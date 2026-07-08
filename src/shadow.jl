"""
    Shadowed(primal, shadow)
    Shadowed(primal)
    Shadowed(primal, T::Type)

A primal array paired with a gradient buffer, passed positionally to
in-place `∇`-kernels: the backward pass reads `primal`s and writes input
gradients into `shadow`s. Shadows are **overwritten**, never accumulated —
cross-call accumulation is the AD backend's job. (Same shape as Enzyme's
`Duplicated`, deliberately not its name: Enzyme's shadows accumulate.)

The shadow need not share the primal's eltype — a narrow-precision forward
usually pairs with wider gradients (`Shadowed(x, BFloat16)` for an fp8
`x`); the backward kernel writes in the shadow's eltype. The one-argument
form pairs `x` with an uninitialized `similar(x)` shadow, the type form
with `similar(x, T)`. A leading [`Space`](@ref) materializes the shadow
from it instead: `Shadowed(space, x, [T])`. A shadow needs no description
protocol — its spec is the primal itself.

Forward kernels accept a bare array or a `Shadowed` interchangeably via
[`primal`](@ref).
"""
struct Shadowed{P,S}
    primal::P
    shadow::S
end

Shadowed(x) = Shadowed(x, similar(x))
Shadowed(x, ::Type{T}) where {T} = Shadowed(x, similar(x, T))
Shadowed(space::Space, x::AbstractArray, T::Type=eltype(x)) = Shadowed(x, alloc(space, Undef(x, T)))

"""
    primal(x)

The primal value of `x`: `x` itself for a bare array, or `d.primal` for a
[`Shadowed`](@ref). Lets forward kernels accept either.
"""
@inline primal(x) = x
@inline primal(d::Shadowed) = d.primal

"""
    shadow(x)

The gradient shadow of a [`Shadowed`](@ref), or `nothing` for anything
else: a bare array passed where a `Shadowed` is accepted is a *constant* —
no gradient is requested and none is written.
"""
@inline shadow(@nospecialize _) = nothing
@inline shadow(d::Shadowed) = d.shadow
