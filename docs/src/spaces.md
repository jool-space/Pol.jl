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

## The ambient space

The explicit space-first forms are the primitives; a dynamically-scoped
binding removes the plumbing where it is noise. Every ambient method —
`alloc(T, dims...)`, an [`Allocating`](@ref) verb called without a leading
space, a [`scratchspace`](@ref) block's defaults — forwards to its explicit
form with [`ambientspace`](@ref).

```@docs
withspace
ambientspace
alloc(::Type{T}, dims::Dims) where {T}
```

## Capture safety

```@docs
capturing
CaptureViolation
```
