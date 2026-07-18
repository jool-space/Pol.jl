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
succeeds**. The second launch fails:

    ERROR: CUDA error: invalid argument (code 1, ERROR_INVALID_VALUE)

A graph holding an allocation node whose memory is not freed *within the graph*
cannot be relaunched until that memory is freed — and replay is the whole
reason the graph exists. So the error lands on exactly the operation the
capture was built to perform, names nothing useful, and survives any smoke test
that launches once. Freeing inside the captured region does not rescue it
either: the free path errors during capture.

(A lesser hazard rides along: if the pool is exhausted, the allocator escalates
to reclaim, which device-synchronizes — and synchronizing under capture
invalidates the capture outright. That fires only under memory pressure, so it
is the same class of bug, deferred further.)

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
