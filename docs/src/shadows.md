```@meta
CurrentModule = Pol
```

# Shadows

A [`Shadowed`](@ref) pairs a primal array with a gradient buffer for in-place
reverse-mode kernels: the backward pass reads primals and writes input
gradients into shadows. Shadows are *overwritten*, never accumulated —
cross-call accumulation is the AD backend's job. (This is the shape of
Enzyme's `Duplicated`, deliberately not its name, whose shadows accumulate.)

The shadow need not share the primal's eltype — a narrow-precision forward
usually pairs with wider gradients — and a leading [`Space`](@ref)
materializes the shadow from it. Forward kernels stay agnostic: [`primal`](@ref)
unwraps a `Shadowed` or passes a bare array through, and [`shadow`](@ref)
returns the gradient buffer or `nothing`, so a bare array passed where a
`Shadowed` is accepted reads as a constant with no gradient requested.

```@autodocs
Modules = [Pol]
Pages = ["shadow.jl"]
```
