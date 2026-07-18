```@meta
CurrentModule = Pol
```

# Scratchspaces

This is the mechanism behind the lifetime split in the [quick
start](@ref "Quick start"): buffers a pass needs only while it runs are
allocated *inside* the frame and die with the block, while buffers the
caller must outlive are allocated from the unframed space *before* the block
opens.

Both forms bind their frame as [the ambient space](@ref "The ambient space")
for the block, so the ambient space is always the innermost open frame.

```@docs
scratchspace
Frame
release!
```
