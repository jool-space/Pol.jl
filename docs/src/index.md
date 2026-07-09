```@meta
CurrentModule = Pol
```

# Pol

Memory management primitives: protocols for kernels to describe the buffers
they need as data, one verb that materializes descriptions into a space of the
caller's choosing, and a bump-allocator with frame-scoped lifetimes.

## Installation

```julia
using Pkg
Registry.add("https://registry.jool.space")
Pkg.add(url = "https://github.com/jool-space/Pol.jl")
```

## Quick start

```julia
using Pol

# a kernel takes its buffers as plain keyword arguments and describes them:
softmax!(y, x; tmp) = (tmp .= exp.(x); y .= tmp ./ sum(tmp, dims=1))
Pol.outputs(::typeof(softmax!), x) = (; y = Pol.Undef(x))
Pol.scratch(::typeof(softmax!), x) = (; tmp = Pol.Undef(x))

x, y = rand(1000), zeros(1000)

# materialize from an arena — the call is its own frame: carve, run, retract
arena = Arena(Vector{UInt8}(undef, 2^20))

scratchspace(arena) do frame
    softmax!(y, x; scratch(frame, softmax!, x)...)
end

# the functional form: output from the unframed space (caller lifetime),
# scratch inside the frame (dies with the block)
function softmax(x; space = Similar(x))
    (; y) = outputs(space, softmax!, x)
    scratchspace(space) do frame
        softmax!(y, x; scratch(frame, softmax!, x)...)
    end
    return y
end
```

Arena allocations are aligned arrays (256-byte by default; see
[`Arena`](@ref)`(slab; alignment)`) aliasing a single preallocated byte slab —
for a plain `Vector{UInt8}` slab they are `unsafe_wrap`ped `Array`s, not
GC-rooted. They are only valid until the [`scratchspace`](@ref) block they
were carved in closes. Arenas are task-local and never grow; allocating past
the end throws.

## Manual

- [Spaces](@ref) — the [`alloc`](@ref) verb, the [`Space`](@ref) it
  materializes against, and the GC-owned [`Similar`](@ref) default.
- [Arenas](@ref) — the [`Arena`](@ref) bump allocator, [`mark`](@ref)s, and
  bulk reclamation.
- [Scratchspaces](@ref) — [`scratchspace`](@ref) blocks and
  frame-scoped lifetimes.
- [Descriptions](@ref) — how a kernel describes the buffers it needs as
  [`Undef`](@ref)s: its [`outputs`](@ref), [`checkpoints`](@ref), and
  [`scratch`](@ref).
- [Shadows](@ref) — pairing primals with gradient buffers for reverse-mode
  kernels.

The full symbol index lives in the [API reference](@ref).
