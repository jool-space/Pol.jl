module Pol

using Republic

include("space.jl")
@public Space, alloc

include("Similar.jl")
export Similar

include("Arena.jl")
export Arena
@public Mark, reset!, watermark, mark, retract!, carve

include("scratchspace.jl")
export scratchspace
@public Frame, release!

include("descriptions.jl")
export Undef, outputs, checkpoints, scratch

include("shadow.jl")
@public Shadowed, primal, shadow

end
