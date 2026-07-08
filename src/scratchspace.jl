"""
    release!(buffer)

Release one frame buffer back to its space when a [`scratchspace`](@ref)
block closes. The default is a no-op — GC-owned buffers just become
garbage. Array families that benefit from eager reclamation overload it
on the buffer type, e.g. `Pol.release!(x::CuArray) = CUDA.unsafe_free!(x)`
returns memory to the pool without waiting for the GC.
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

An allocation frame on `space`: calls `f` with a [`Frame`](@ref) —

    scratchspace(space) do frame
        softmax!(y, x; scratch(frame, softmax!, x)...)
    end

On an [`Arena`](@ref) the frame carries a [`mark`](@ref), retracted when
the block closes, also on a throw — everything carved inside is reclaimed.
On any other space the frame records each buffer it materializes and
[`release!`](@ref)s them all at close — a no-op for GC-owned arrays, eager
reclamation where an overload provides it. Frames nest: opening a frame
on a frame stacks a new one on the same space.

Buffers allocated from the frame are only valid inside the block.
Allocating from the unframed space — [`outputs`](@ref) before the block
opens — is what gives a buffer the caller's lifetime.
"""
function scratchspace(f::Function, space::Space)
    frame = Frame(space)
    try
        return f(frame)
    finally
        release!(frame)
    end
end
