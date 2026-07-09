"""
    Undef(T, dims)
    Undef(T, dims...)
    Undef(x, [T=eltype(x)])
    Undef(x, [dims=size(x)])

A description of the element type and size of an array, holding no memory.
`Array{T}(undef, dims)` allocates the array; `Undef{T}(dims)` only describes
one, for [`alloc`](@ref) to allocate against a space of the caller's
choosing.
"""
struct Undef{T,N}
    dims::Dims{N}
end

Undef{T}(dims::Dims{N}) where {T,N} = Undef{T,N}(dims)
Undef{T}(dims::Integer...) where {T} = Undef{T}(Dims(dims))

Undef(T::Type, dims::Dims) = Undef{T}(dims)
Undef(T::Type, dims::Integer...) = Undef{T}(dims...)

Undef(x::AbstractArray, T::Type=eltype(x)) = Undef(T, size(x))
Undef(x::AbstractArray, dims::Dims) = Undef(eltype(x), dims)

"""
    alloc(space, u::Undef)
    alloc(space, spec::NamedTuple)

Materialize an [`Undef`](@ref) description — or a whole `NamedTuple` spec
of them, name for name — from `space`. Specs nest: a `NamedTuple` value
materializes recursively, so buffers bundled under one name stay bundled.
"""
alloc(space::Space, u::Undef{T}) where {T} = alloc(space, T, u.dims)
alloc(space::Space, spec::NamedTuple) = map(u -> alloc(space, u), spec)

"""
    outputs(f, args...; kwargs...) -> NamedTuple
    outputs(space, f, args...; kwargs...)

The arrays a pass produces — the destinations, first in the mutating
signature — described as [`Undef`](@ref)s. Implementations dispatch on the
pass function. A leading [`Space`](@ref) materializes the description on
the spot: `outputs(space, f, args...)` is `alloc(space, outputs(f,
args...))`. Outputs belong to the caller: materialize them from the
unframed space, before the [`scratchspace`](@ref) block they must outlive.
"""
function outputs end

"""
    checkpoints(f, args...; kwargs...) -> NamedTuple
    checkpoints(space, f, args...; kwargs...)

The buffers a primitive saves in its forward pass and reads in its backward
pass (a norm's `Rstd`, attention's `M`/`L`), described as [`Undef`](@ref)s:
the primitive states eltypes and shapes, allocates nothing. Implementations
dispatch on the forward function. A primitive that saves nothing has no
method. The described names are keyword arguments the primitive already
takes; a leading [`Space`](@ref) materializes them — splat the result into
the call.

Distinct from [`scratch`](@ref), which lives and dies within a scope.
"""
function checkpoints end

"""
    scratch(f, args...; kwargs...) -> NamedTuple
    scratch(frame::Frame, f, args...; kwargs...)

The buffers a single pass needs only while it runs (workspace,
partial-reduction buffers), described as [`Undef`](@ref)s. Dispatched per
pass function — `rms_norm!` and `∇rms_norm!` each describe their own; a pass
that needs none has no method. The described names are keyword arguments the
pass already takes; materialize them against the [`Frame`](@ref) a
[`scratchspace`](@ref) block passes — `scratch(frame, f, args...)` is
`alloc(frame, scratch(f, args...))` — and splat into the call. The one-step
form takes a `Frame` only: scratch lifetime is frame lifetime.

Distinct from [`checkpoints`](@ref), which bridges forward → backward.
"""
function scratch end

outputs(space::Space, f, args...; kwargs...) = alloc(space, outputs(f, args...; kwargs...))
checkpoints(space::Space, f, args...; kwargs...) = alloc(space, checkpoints(f, args...; kwargs...))
scratch(frame::Frame, f, args...; kwargs...) = alloc(frame, scratch(f, args...; kwargs...))
