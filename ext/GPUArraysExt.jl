module GPUArraysExt

using Pol
using GPUArrays

Pol.release!(x::AbstractGPUArray) = unsafe_free!(x)

end
