```@meta
CurrentModule = Pol
```

# Verbs

A mutating verb takes its outputs first, as a NamedTuple it destructures —
`f!((; Y), X; kws...)` — and describes them with [`outputs`](@ref). Its
[`Allocating`](@ref) form is the same verb with the first argument answering
*where results live* instead of *which arrays*: a [`Space`](@ref) the
outputs materialize from, before any frame opens.

```@docs
Allocating
takes_space
@takes_space
```
