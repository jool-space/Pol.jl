"""
    Allocating(f!)

The allocating form of a mutating verb: a callable that owns the
outputs/scratch ritual so call sites don't repeat it.

    const decode_attention = Allocating(decode_attention!)
    o = decode_attention(space, q, K, V; lengths)

The first argument answers *where results live* — the mutating verb takes
its output buffers there, the allocating form takes a [`Space`](@ref) to
materialize them from. Calling it allocates [`outputs`](@ref) from `space`
— **before** any frame opens, so on an [`Arena`](@ref) they sit below the
frame's mark and survive its retract — then runs `f!` and returns the
outputs `NamedTuple`. A verb marked [`@takes_space`](@ref) runs inside a
[`scratchspace`](@ref) block and receives the frame as its `space` keyword:
its defaulted scratch is carved there and reclaimed at return.

`space` may itself be a [`Frame`](@ref) — frames are spaces and nest — in
which case the outputs take the caller's frame lifetime.

The `output_override` keyword merges caller-owned arrays over the
[`outputs`](@ref) spec *before* materialization, so an overridden entry is
never allocated — pass the verb's own state array to run partially in
place: `f(space, args..., S; output_override = (; S′ = S))`.
"""
struct Allocating{F}
    f!::F
end

Base.show(io::IO, a::Allocating) = print(io, "Allocating(", a.f!, ")")

"""
    takes_space(f!) -> Bool

Whether the mutating verb `f!` accepts a `space` keyword — the space its
defaulted scratch materializes from. `false` generically; declare with
[`@takes_space`](@ref). [`Allocating`](@ref) branches on it: a verb that
takes no space is called bare, with no frame opened on its behalf.
"""
takes_space(@nospecialize f!) = false

"""
    Pol.@takes_space f!

Declare that the mutating verb `f!` accepts a `space` keyword (see
[`takes_space`](@ref)). One line next to the verb's [`scratch`](@ref)
method:

    Pol.@takes_space decode_attention!
"""
macro takes_space(f!)
    esc(:($Pol.takes_space(::typeof($f!)) = true))
end

function ((; f!)::Allocating)(space::Space, args...; output_override = (;), kws...)
    out = alloc(space, merge(outputs(f!, args...; kws...), output_override))
    if takes_space(f!)
        scratchspace(space) do frame
            f!(out, args...; space = frame, kws...)
        end
    else
        f!(out, args...; kws...)
    end
    return out
end

# the ambient form: results live in the ambient space
(a::Allocating)(args...; kws...) = a(ambientspace(), args...; kws...)
