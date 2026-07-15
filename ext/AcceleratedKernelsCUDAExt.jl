module AcceleratedKernelsCUDAExt


import AcceleratedKernels as AK
import CUDA
using CUDA: @device_override


# The generic `AK._decoupled_fence` lowers fine on NVPTX, but as `fence.acq_rel.sys` —
# system scope, which needlessly orders against host and peer-GPU agents that take no part
# in the decoupled-lookback protocol. `CUDA.threadfence()` emits `membar.gl` (device scope),
# giving the same guarantee for the blocks that matter at ~1.5x the lookback throughput.
# This is a performance narrowing; the generic fence is already correct.
#
# Both the accumulate decoupled-lookback scan and the radix onesweep scatter go through this
# single definition, so there is one audited fence implementation per backend.
@device_override @inline AK._decoupled_fence() = CUDA.threadfence()


end   # module AcceleratedKernelsCUDAExt
