module AcceleratedKernelsCUDAExt


import AcceleratedKernels as AK
import CUDA
using CUDA: @device_override


# The experimental radix onesweep path uses a device-scope fence in its decoupled look-back.
# On CUDA the NVPTX backend does not select scoped atomic fences (the generic acquire-release
# `AK._os_fence`), so provide the native `membar.gl` threadfence instead.
@device_override @inline AK._os_fence() = CUDA.threadfence()


end   # module AcceleratedKernelsCUDAExt
