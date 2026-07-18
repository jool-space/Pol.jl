"""
    Similar(x)

The GC-owned [`Space`](@ref) around an exemplar array: [`alloc`](@ref)
against it is `similar(x, T, dims)` — same array family and device,
lifetimes by scope. The exemplar is dispatched on, never read.

Eager-only: a carve goes through the device's allocator, so allocating from a
`Similar` under an active graph capture throws [`CaptureViolation`](@ref)
rather than silently recording a memory node into the graph. Under capture,
use an [`Arena`](@ref).
"""
struct Similar{X<:AbstractArray} <: Space
    x::X
end

function alloc(s::Similar, ::Type{T}, dims::Dims) where {T}
    check_capture(s, T, dims, s.x)
    return similar(s.x, T, dims)
end
