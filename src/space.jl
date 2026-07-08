"""
    Space

The abstract supertype of allocation spaces — anywhere an uninitialized
array can be materialized from. A space implements the one leaf method
`alloc(space, T, dims::Dims)`; everything else — [`Undef`](@ref) specs,
`NamedTuple` specs, space-first [`outputs`](@ref)/[`checkpoints`](@ref)
and frame-first [`scratch`](@ref) — reduces to it.
"""
abstract type Space end

"""
    Similar(x)

The GC-owned space around an exemplar array: [`alloc`](@ref) against it
is `similar(x, T, dims)` — buffers of `x`'s array family, on `x`'s
device, with lifetimes by scope. The exemplar is dispatched on, never
read.
"""
struct Similar{X<:AbstractArray} <: Space
    x::X
end

"""
    alloc(space, T, dims...) -> AbstractArray{T}
    alloc(space, u::Undef)
    alloc(space, spec::NamedTuple)

Materialize an uninitialized array — or a whole spec of [`Undef`](@ref)
descriptions, name for name — from `space`. Specs nest: a `NamedTuple`
value materializes recursively, so buffers bundled under one name stay
bundled. The spaces:

- an [`Arena`](@ref) carves from its slab, reclaimed by `retract!`/`reset!`
- a [`Similar`](@ref) exemplar gives `similar(x, T, dims)` — GC-owned,
  same array family and device

A custom space implements the one leaf method `alloc(space, T, dims::Dims)`.
"""
alloc(space, ::Type{T}, dims::Integer...) where {T} = alloc(space, T, Dims(dims))

alloc(s::Similar, ::Type{T}, dims::Dims) where {T} = similar(s.x, T, dims)
