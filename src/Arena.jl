"""
    Arena(slab::AbstractVector{UInt8}; alignment = 256)

A bump allocator over one preallocated byte `slab`: [`alloc`](@ref) carves
typed arrays from it, [`retract!`](@ref) and [`reset!`](@ref) reclaim them
in bulk, and nothing tracks individual allocations. The arena never grows —
allocating past the end of the slab throws — and identical allocation
sequences yield identical addresses, which is what graph capture requires.

`alignment` (a power of two) is where each carve starts; a carve
additionally never starts below its element type's own alignment. When the
slab's storage exposes a base address ([`slabbase`](@ref)), the arena pads
its origin so that alignment is *absolute*: every carve's address is
divisible by `alignment` no matter where the slab landed, at a one-time
cost of at most `alignment - 1` slab bytes. CUDA allocators return
256-aligned bases — the pad is zero and the default meets cuBLASLt's
workspace requirement — while a Julia `Vector`'s base is only 16/64-aligned,
so the pad is what makes a CPU arena's carves truly 256-aligned. Storage
with no readable base (`slabbase` returning `nothing`) keeps the relative
contract: carves are aligned relative to the slab's first byte only.

Arenas are task-local, so they take no locks.
"""
mutable struct Arena{S<:AbstractVector{UInt8}} <: Space
    const slab::S
    const alignment::Int
    const pad::Int
    offset::Int
    watermark::Int
    epoch::Int
end

"""
    slabbase(slab) -> Union{Nothing, UInt}

The absolute address of a slab's first byte, or `nothing` when the storage
exposes none. An [`Arena`](@ref) pads its origin by `-slabbase mod
alignment`, which is what turns relative alignment into absolute. The
generic method answers `nothing` — no pad, the relative contract — and
dense vectors answer their `pointer`; a new storage family overloads this
alongside [`carve`](@ref). The address is virtual, and that is the right
notion: every consumer of alignment — an API contract, a vectorized load —
checks the virtual address, and pages map whole, so virtual and physical
agree modulo the page size, far above any alignment an arena is asked for.
A storage whose base can move must answer `nothing`: a pad computed from a
stale base is misalignment, silently.
"""
slabbase(::AbstractVector{UInt8}) = nothing
slabbase(slab::DenseVector{UInt8}) = UInt(pointer(slab))

function Arena(slab::AbstractVector{UInt8}; alignment::Integer = 256)
    ispow2(alignment) || throw(ArgumentError("alignment must be a power of two, got $alignment"))
    base = slabbase(slab)
    pad = base === nothing ? 0 : Int(mod(-base, UInt(alignment)))
    return Arena(slab, Int(alignment), pad, 0, 0, 0)
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
run against this arena, not counting the arena's base pad. Survives
[`reset!`](@ref), so it accumulates over all passes of a warmup. When
sizing a fresh slab from a watermark, add `alignment` slack: a
differently-based slab pads differently.
"""
watermark(a::Arena) = a.watermark

"""
    alloc(space::Arena, T, dims) -> AbstractArray{T}

Carve a `dims`-shaped array of bitstype `T` from the arena: round the offset
up to the arena's alignment (never below `T`'s own), take the next
`sizeof(T) * prod(dims)` bytes as a typed array (see [`carve`](@ref)),
advance the offset, update the watermark. Offsets count from the arena's
padded origin, so the pad rides on top of the ceiling. Throws past the end
of the slab. The offset advance is source state, like an RNG's — hence no
`!`.
"""
function alloc(a::Arena, ::Type{T}, dims::Dims) where T
    isbitstype(T) || throw(ArgumentError("an arena carves bitstypes only, got $T"))
    start = align(a.offset, max(a.alignment, Base.datatype_alignment(T)))
    nbytes = sizeof(T) * prod(dims)
    stop = start + nbytes
    a.pad + stop <= length(a.slab) ||
        throw(ArgumentError("arena ceiling: need $(a.pad + stop) bytes ($nbytes for $T$dims on a $(a.pad)-byte base pad), slab holds $(length(a.slab)) — raise the budget"))
    a.offset = stop
    a.watermark = max(a.watermark, stop)
    return carve(a.slab, a.pad + start, T, dims)
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
