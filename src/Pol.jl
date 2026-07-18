module Pol

using Republic

include("space.jl")
export withspace, ambientspace
@public Space, alloc

include("capture.jl")
@public capturing, CaptureViolation

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

include("allocating.jl")
export Allocating
@public takes_space, @takes_space

include("shadow.jl")
@public Shadowed, primal, shadow

end
