```@meta
CurrentModule = Pol
```

# Arenas

An [`Arena`](@ref) is a bump allocator over one preallocated byte slab:
[`alloc`](@ref) carves typed arrays from it by advancing an offset, and
reclamation happens in bulk rather than per allocation. Because identical
allocation sequences yield identical addresses, an arena is what graph
capture (e.g. CUDA graphs) requires.

Reclamation comes in two forms. [`mark`](@ref) snapshots the current offset
and [`retract!`](@ref) rolls back to it — everything carved since the mark
becomes garbage at once. [`reset!`](@ref) rolls all the way back to zero and
advances the epoch, invalidating outstanding marks. The [`watermark`](@ref)
records the peak offset ever reached and survives `reset!`, so a warmup pass
measures the exact byte budget a workload needs.

Arenas never grow — allocating past the end of the slab throws — and are
task-local, so they take no locks.

```@autodocs
Modules = [Pol]
Pages = ["src/arena.jl"]
```
