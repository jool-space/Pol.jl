```@meta
CurrentModule = Pol
```

# Spaces

Pol ships two spaces: the GC-owned [`Similar`](@ref) default and the
[`Arena`](@ref) bump allocator. The [`Frame`](@ref)s a
[`scratchspace`](@ref) block opens are spaces too.

```@docs
Space
alloc(space::Space, ::Type{T}, dims::Integer...) where {T}
```

## `Similar`

```@docs
Similar
```

## `Arena`

```@docs
Arena
alloc(a::Arena, ::Type{T}, dims::Dims) where T
Mark
mark
retract!
reset!
watermark
carve
```
