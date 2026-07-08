```@meta
CurrentModule = Pol
```

# Descriptions

A kernel describes the buffers it needs as data, holding no memory, and lets
the caller decide where they come from. The atom is [`Undef`](@ref) — an
element type and a shape — which [`alloc`](@ref) materializes against a
[`Space`](@ref) of the caller's choosing.

A kernel bundles its `Undef`s into named descriptions, dispatched on the
kernel function and split by lifetime:

- [`outputs`](@ref) — the arrays a pass produces. They belong to the caller,
  so they are materialized from the unframed space and outlive the
  [`scratchspace`](@ref) block.
- [`checkpoints`](@ref) — buffers a forward pass saves and its backward pass
  reads (a norm's `Rstd`, attention's `M`/`L`). They bridge forward →
  backward.
- [`scratch`](@ref) — buffers a single pass needs only while it runs. They
  live and die within a frame; the one-step form materializes against a
  [`Frame`](@ref) only.

Each protocol has a space-first (or frame-first) form that materializes the
description on the spot, ready to splat into the call. A pass that needs none
of a given kind simply defines no method.

```@autodocs
Modules = [Pol]
Pages = ["descriptions.jl"]
```
