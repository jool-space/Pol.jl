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

# ── the ambient space ────────────────────────────────────────────────────────

using ScopedValues: ScopedValue, with

# The ambient Space, read by `currentspace` and every ambient method.
# Dynamic scope is the right model here: a frame's buffers are valid exactly
# for the frame's dynamic extent, so a binding with that same extent cannot
# name a dead frame. Deliberately not exported — bind with `withspace`.
const SPACE = ScopedValue{Union{Space, Nothing}}(nothing)

"""
    withspace(f, space)

Call `f()` with `space` as the ambient space for its dynamic extent. Inside,
[`ambientspace`](@ref) returns `space`, and every ambient method —
`alloc(T, dims...)`, an [`Allocating`](@ref) verb called without a leading
space — uses it. A [`scratchspace`](@ref) block rebinds the ambient space
to its frame, so the ambient space is always the innermost open frame. The
explicit space-first forms remain the primitives; the ambient forms forward
to them, so this is one semantics with two spellings. An ambient read is
dynamically dispatched (scope is runtime data) — thread spaces explicitly
where full inference matters.

Scoped values flow into spawned tasks, but an [`Arena`](@ref) is task-local
by design: don't `Threads.@spawn` under `withspace` of an arena.
"""
withspace(f, space::Space) = with(f, SPACE => space)

"""
    ambientspace() -> Space
    ambientspace(default)

The ambient space bound by the nearest enclosing [`withspace`](@ref) or
[`scratchspace`](@ref) block. The zero-arg form throws when no space is
bound; `ambientspace(default)` returns `default` instead — the form for
kernel `space` keyword defaults, which must also work unscoped:

    space = ambientspace(Similar(X))
"""
ambientspace() = @something(SPACE[],
    throw(ArgumentError("no ambient space: not inside withspace/scratchspace")))
ambientspace(default) = @something(SPACE[], default)

"""
    alloc(args...)

The ambient form: any `alloc` call without a leading space forwards to
`alloc(ambientspace(), args...)` — every explicit spelling (`T, dims`,
an [`Undef`](@ref), a `NamedTuple` spec) has its ambient counterpart
through this one method.
"""
alloc(args...) = alloc(ambientspace(), args...)

# A Space-first call that matched no explicit method must not fall through
# to the ambient form (alloc(ambient, space, ...) — a loop, or nonsense);
# fail as the missing method it is.
alloc(space::Space, args...) = throw(MethodError(alloc, (space, args...)))
