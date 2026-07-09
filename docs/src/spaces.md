```@meta
CurrentModule = Pol
```

# Spaces

A [`Space`](@ref) is anywhere an array can be materialized from. The one verb
[`alloc`](@ref) turns an [`Undef`](@ref) description — an element type and a
shape, no memory — into a real array against the space the caller chose; every
space reduces to the single leaf method `alloc(space, T, dims::Dims)`.

[`Similar`](@ref) is the GC-owned space around an exemplar array: `alloc`
against it is `similar(x, T, dims)` — same array family and device, lifetimes
by scope — the natural default when no arena is in play. The bump-allocator
[`Arena`](@ref) and the frames a [`scratchspace`](@ref) opens are spaces too;
what a kernel materializes *from* them is covered under [Descriptions](@ref).

```@autodocs
Modules = [Pol]
Pages = ["src/space.jl"]
```
