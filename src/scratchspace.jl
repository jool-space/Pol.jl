"""
    release!(buffer)

Release one frame buffer back to its space when a [`scratchspace`](@ref)
block closes. The default is a no-op — GC-owned buffers just become
garbage. Array families that benefit from eager reclamation overload it
on the buffer type; a package extension already provides
`release!(x::AbstractGPUArray) = GPUArrays.unsafe_free!(x)`, freeing the
device memory immediately instead of when the GC finalizes the array.
"""
release!(x) = nothing

"""
    Frame

The abstract supertype of allocation frames — the framed space a
[`scratchspace`](@ref) block passes. A frame is a [`Space`](@ref) whose
buffers are reclaimed when its block closes, and proof of being inside
one: the one-step [`scratch`](@ref) form materializes against a `Frame`
only.
"""
abstract type Frame <: Space end

# generic frame: buffers recorded, release!d at block close;
# a frame opened on a frame tracks at its own level only
struct TrackedFrame{S} <: Frame
    space::S
    buffers::Vector{Any}
end
Frame(space::Space) = TrackedFrame(space, Any[])
Frame(frame::TrackedFrame) = Frame(frame.space)

function alloc(frame::TrackedFrame, ::Type{T}, dims::Dims) where {T}
    b = alloc(frame.space, T, dims)
    push!(frame.buffers, b)
    return b
end

release!(frame::TrackedFrame) = foreach(release!, frame.buffers)

# an arena's frame: the arena plus the mark its block retracts to
struct ArenaFrame{A<:Arena} <: Frame
    arena::A
    mark::Mark
end
Frame(arena::Arena) = ArenaFrame(arena, mark(arena))
Frame(frame::ArenaFrame) = Frame(frame.arena)

alloc(frame::ArenaFrame, ::Type{T}, dims::Dims) where {T} = alloc(frame.arena, T, dims)

release!(frame::ArenaFrame) = retract!(frame.arena, frame.mark)

"""
    scratchspace(f, space)

An allocation frame on `space`: calls `f` with a [`Frame`](@ref), which is
also the ambient space for the block —

    scratchspace(space) do frame
        softmax!(y, x; scratch(frame, softmax!, x)...)
    end

On an [`Arena`](@ref) the frame carries a [`mark`](@ref), retracted when
the block closes, also on a throw — everything carved inside is reclaimed.
On any other space the frame records each buffer it materializes and
[`release!`](@ref)s them all at close — a no-op for GC-owned arrays, eager
reclamation where an overload provides it. Frames nest: opening a frame
on a frame stacks a new one on the same space.

Inside the block the ambient space ([`ambientspace`](@ref)) is the frame:
once a frame is open, nothing should allocate from outside it implicitly,
so every frame-opening rebinds. The `frame` argument is therefore
redundant for lifetime — but it is the concretely-typed spelling: ambient
reads are dynamically dispatched, `alloc(frame, ...)` infers.

Buffers allocated from the frame are only valid inside the block.
Allocating from the unframed space — [`outputs`](@ref) before the block
opens — is what gives a buffer the caller's lifetime.
"""
function scratchspace(f::Function, space::Space)
    frame = Frame(space)
    try
        return withspace(() -> f(frame), frame)
    finally
        release!(frame)
    end
end

"""
    scratchspace(f)

The ambient form: a frame on [`ambientspace`](@ref), rebound as ambient
for the block — `f` is called with no arguments, and everything inside
allocates from the frame by default:

    scratchspace() do
        tmp = alloc(Float32, n)      # carved from this block's frame
        y = softmax(x)               # an Allocating verb: outputs too
    end                              # frame retracts

The invariant both forms maintain: the ambient space is always the
innermost open frame. The closure receives no frame because it needs no
frame — an allocation that must *outlive* the block is the case that
deserves the explicit space-first spelling.
"""
scratchspace(f::Function) = scratchspace(_ -> f(), ambientspace())
