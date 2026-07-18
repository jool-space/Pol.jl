module CUDACoreExt

using Pol
using CUDACore: CuArray, CuStream, is_capturing, stream

# The one backend answer to Pol's capture question. A CUDA allocation is
# stream-ordered, so under an active capture it is *recorded* rather than
# refused — see `Pol.CaptureViolation` for why that is worse than failing.
# `is_capturing` reads the capture status of the current task's stream, which
# is the stream a `similar` on this array would allocate against.
Pol.capturing(::CuArray) = is_capturing(stream())

end
