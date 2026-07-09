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
    alloc(space::Space, T, dims)
    alloc(space::Space, T, dims...)

Materialize the description of an array into `space`.
A custom space implements the one leaf method `alloc(space, T, dims::Dims)`.
"""
alloc(space::Space, ::Type{T}, dims::Integer...) where {T} = alloc(space, T, Dims(dims))
