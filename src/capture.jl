"""
    capturing(x) -> Bool

Whether the device backing `x` is currently recording a graph capture.

`false` for anything that cannot capture — every CPU array, and any backend
without an extension answering otherwise. A backend extension adds a method on
its array type (`Pol.capturing(::CuArray)`), so this is the one predicate the
rest of the stack asks.
"""
capturing(@nospecialize _) = false

"""
    CaptureViolation

A buffer was materialized from a garbage-collected [`Space`](@ref) while the
device was recording a graph capture.

The allocation does not fail, and that is the problem. A device allocator is
*stream-ordered*, so allocating under capture is recorded into the graph as a
memory node — capture succeeds, instantiation succeeds, and **the first launch
succeeds**. What happens after that depends on where the matching free landed:

- **Freed outside the captured region, or not at all** — a no-op `release!`,
  a frame that closes after capture ends, a buffer that escaped its frame:
  the allocation node has no paired free node, and a graph whose previous
  launch's allocations are still outstanding cannot be relaunched:

      ERROR: CUDA error: invalid argument (code 1, ERROR_INVALID_VALUE)

  The error lands on the second launch — the replay the graph exists for —
  names nothing useful, comes and goes with finalizer timing, and survives
  any smoke test that launches once.

- **Freed inside the captured region**: the free is recorded too, the nodes
  pair, and replay is legal — a `scratchspace` block enclosed whole by the
  capture, `release!` freeing eagerly, does replay correctly. But it stands
  on ground the stack does not promise: CUDA.jl documents allocations as
  non-capturable (capture disables GC for that reason) and the driver is
  doing the rescuing; every replay pays the allocation node's map/unmap; a
  re-capture records fresh addresses. And if the pool is exhausted
  mid-capture, the allocator escalates to reclaim, which device-synchronizes
  — invalidating the capture outright, with GC disabled so Julia-side
  reclamation cannot intervene.

So the split is structural, not situational: one shape fails on exactly the
operation the capture was built to perform, the other works by grace of the
driver. Neither is a foundation.

The capture-safe space is an [`Arena`](@ref): a carve is offset arithmetic over
a slab allocated long before capture began, so it records nothing into the
graph, and an identical allocation sequence yields identical addresses — which
is what replay requires. Warm up eagerly, read the [`watermark`](@ref), size the
slab, *then* capture.

Warmth is not optional for the rest of the step either: a kernel's first launch
loads its module, which capture forbids (`ERROR_STREAM_CAPTURE_UNSUPPORTED`).
Run the step once eagerly before capturing it — the same pass that fills the
watermark.

Reaching this error means a pass fell back to its eager default — a
[`Similar`](@ref) space, or a `scratchspace` on one — inside a captured
region. Pass an arena (or a `Frame` on one) instead.
"""
struct CaptureViolation <: Exception
    space::Any
    T::Type
    dims::Dims
end

function Base.showerror(io::IO, e::CaptureViolation)
    print(io, "CaptureViolation: allocating $(e.T)$(e.dims) from a ",
          nameof(typeof(e.space)), " while the device is capturing a graph.\n",
          "A garbage-collected space allocates through the device's ",
          "stream-ordered allocator, so this would not fail — it would record ",
          "a memory node into the graph, and the graph would then launch once ",
          "and fail on the second launch (ERROR_INVALID_VALUE), which is the ",
          "replay the capture exists for.\n",
          "Carve from an Arena instead: warm up eagerly, size the slab from ",
          "`watermark`, then capture.")
end

# The guard belongs here, not in each caller: `alloc` is the one chokepoint
# every buffer in the stack passes through, so a verb gets the guarantee by
# construction rather than by remembering to ask.
@inline function check_capture(space, ::Type{T}, dims::Dims, exemplar) where {T}
    capturing(exemplar) && throw(CaptureViolation(space, T, dims))
    return nothing
end
