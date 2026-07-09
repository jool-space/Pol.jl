"""
    Arena(slab::AbstractVector{UInt8}; alignment = 256)

A bump allocator over one preallocated byte `slab`: [`alloc`](@ref) carves
typed arrays from it, [`retract!`](@ref) and [`reset!`](@ref) reclaim them
in bulk, and nothing tracks individual allocations. The arena never grows —
allocating past the end of the slab throws — and identical allocation
sequences yield identical addresses, which is what graph capture requires.

`alignment` (a power of two) is where each carve starts, **relative to the
slab's first byte**; a carve additionally never starts below its element
type's own alignment. Absolute alignment therefore also depends on the
slab's base: CUDA allocators return 256-aligned bases (so the default meets
cuBLASLt's workspace requirement), a Julia `Vector`'s base is only 32/64,
`Mmap.mmap`'s is page-aligned.

Arenas are task-local, so they take no locks.
"""
mutable struct Arena{S<:AbstractVector{UInt8}} <: Space
    const slab::S
    const alignment::Int
    offset::Int
    watermark::Int
    epoch::Int
end

function Arena(slab::AbstractVector{UInt8}; alignment::Integer = 256)
    ispow2(alignment) || throw(ArgumentError("alignment must be a power of two, got $alignment"))
    return Arena(slab, Int(alignment), 0, 0, 0)
end

align(offset::Int, alignment::Int) = (offset + (alignment - 1)) & -alignment

"""
    Mark

A snapshot of an [`Arena`](@ref)'s offset, returned by [`mark`](@ref) and
consumed by [`retract!`](@ref). Carries the arena's epoch, so retracting to
a mark taken before a [`reset!`](@ref) throws instead of silently corrupting
newer allocations.
"""
struct Mark
    offset::Int
    epoch::Int
end

"""
    mark(arena) -> Mark

Snapshot the current offset. Does not change the arena.
"""
mark(a::Arena) = Mark(a.offset, a.epoch)

"""
    retract!(arena, m::Mark)

Move the offset back to `m`: everything allocated since the mark was taken
is garbage at once, however many carves that is. Retracting to the same mark
twice with nothing allocated in between is a no-op. Throws if the mark is
stale (taken before a [`reset!`](@ref)) or above the current offset.
"""
function retract!(a::Arena, m::Mark)
    m.epoch == a.epoch ||
        throw(ArgumentError("stale mark: taken at epoch $(m.epoch), arena is at epoch $(a.epoch) (a reset! happened in between)"))
    m.offset <= a.offset ||
        throw(ArgumentError("mark at offset $(m.offset) is above the current offset $(a.offset): already retracted past it"))
    a.offset = m.offset
    return a
end

"""
    reset!(arena)

Offset back to 0 — every carve is garbage — and the epoch advances, so
existing marks become stale. The [`watermark`](@ref) survives.
"""
function reset!(a::Arena)
    a.offset = 0
    a.epoch += 1
    return a
end

"""
    watermark(arena) -> Int

The peak offset ever reached — the measured byte requirement of everything
run against this arena. Survives [`reset!`](@ref), so it accumulates over
all passes of a warmup.
"""
watermark(a::Arena) = a.watermark

"""
    alloc(space::Arena, T, dims) -> AbstractArray{T}

Carve a `dims`-shaped array of bitstype `T` from the arena: round the offset
up to the arena's alignment (never below `T`'s own), take the next
`sizeof(T) * prod(dims)` bytes as a typed array (see [`carve`](@ref)),
advance the offset, update the watermark. Throws past the end of the slab.
The offset advance is source state, like an RNG's — hence no `!`.
"""
function alloc(a::Arena, ::Type{T}, dims::Dims) where T
    isbitstype(T) || throw(ArgumentError("an arena carves bitstypes only, got $T"))
    start = align(a.offset, max(a.alignment, Base.datatype_alignment(T)))
    nbytes = sizeof(T) * prod(dims)
    stop = start + nbytes
    stop <= length(a.slab) ||
        throw(ArgumentError("arena ceiling: need $stop bytes ($nbytes for $T$dims), slab holds $(length(a.slab)) — raise the budget"))
    a.offset = stop
    a.watermark = max(a.watermark, stop)
    return carve(a.slab, start, T, dims)
end

"""
    carve(slab, offset, T, dims) -> AbstractArray{T}

A `dims`-shaped `T` array over `slab[offset+1 : offset+sizeof(T)*prod(dims)]`,
aliasing the bytes — never copying. The arena's only function that touches
memory, and its per-storage specialization point. The generic method is
`reshape(reinterpret(T, view(slab, …)), dims)`; a `Vector{UInt8}` slab gets
a native `Array` via `unsafe_wrap` instead — faster to index, **not
GC-rooted**: valid only while the arena holding the slab is alive.
"""
function carve(slab::AbstractVector{UInt8}, offset::Int, ::Type{T}, dims::Dims) where T
    nbytes = sizeof(T) * prod(dims)
    bytes = @view slab[offset .+ (1:nbytes)]
    return reshape(reinterpret(T, bytes), dims)
end

carve(slab::Vector{UInt8}, offset::Int, ::Type{T}, dims::Dims) where {T} =
    unsafe_wrap(Array, convert(Ptr{T}, pointer(slab) + offset), dims)
