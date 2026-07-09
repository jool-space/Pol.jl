"""
    Similar(x)

The GC-owned [`Space`](@ref) around an exemplar array: [`alloc`](@ref)
against it is `similar(x, T, dims)` — same array family and device,
lifetimes by scope. The exemplar is dispatched on, never read.
"""
struct Similar{X<:AbstractArray} <: Space
    x::X
end

alloc(s::Similar, ::Type{T}, dims::Dims) where {T} = similar(s.x, T, dims)
