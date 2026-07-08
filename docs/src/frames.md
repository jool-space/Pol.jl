```@meta
CurrentModule = Pol
```

# Frames & scratch space

A [`scratchspace`](@ref) block opens a [`Frame`](@ref) on a space and
guarantees that everything allocated from the frame is reclaimed when the
block closes — including on a throw. On an [`Arena`](@ref) the frame carries
a [`mark`](@ref) and retracts to it; on any other space it records each
buffer it hands out and [`release!`](@ref)s them at close.

This is the mechanism behind the lifetime split in the [quick
start](@ref "Quick start"): buffers a pass needs only while it runs are
allocated *inside* the frame and die with the block, while buffers the
caller must outlive are allocated from the unframed space *before* the block
opens. Frames nest — opening one on a frame stacks a new level on the same
space.

[`release!`](@ref) is a no-op for GC-owned arrays but the extension point
for array families with eager reclamation (e.g. `CUDA.unsafe_free!`).

```@autodocs
Modules = [Pol]
Pages = ["scratchspace.jl"]
```
