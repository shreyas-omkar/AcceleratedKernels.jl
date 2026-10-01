# Prototype: Int32 kernel indexing with unchecked boundary conversions.
#
# Julia's native indexing is Int64. Asking a kernel for Int32 indices only pays off if the index
# conversion at the boundary is UNCHECKED: `Int32(x::Int64)` is a checked (throwing) narrowing, and the
# compiler keeps the never-taken range-check + error path, which inflates register usage. Using
# `unsafe_trunc(Int32, ...)` at the one narrowing boundary removes that path; widening Int32 -> Int64 for
# the pointer offset is free. With that, Int32 indices use fewer registers than Int64, as expected.
#
# Register usage of an indexing-heavy gather kernel (VGPR/SGPR from the compiled GCN ISA, AMD gfx1200):
#   Int64 (native)              : 14 / 24
#   Int32 via checked Int32(...) : 12 / 58   <- error path balloons SGPR
#   Int32 via unsafe_trunc       : 10 / 18   <- fewest of the three (the real win)
#
# FINDING: the register saving does NOT translate into speed for AK's kernels. Measured on AMD RDNA4 and
# NVIDIA RTX 5080, an Int32-indexed reduction is 1.0x (noise, 0.94-1.02x) vs Int64 at every size, because
# these kernels are memory-bound and already at max occupancy, so fewer registers / cheaper index math buy
# nothing (matches the "~0.5% unless at an occupancy boundary" expectation). Int32 indexing is worth it only
# for compute-bound, register-starved kernels. None of AK's throughput primitives (reduce, scan, sort,
# reverse, findall, map) are that. Kept here as a reviewable reference for the technique; the way to apply it
# in practice is an index-type type parameter on the kernel, chosen on the host by the input size (so a
# >2^31-element input stays Int64).
#
# Run: SR_BACKEND=amd|cuda julia --project=<env with AMDGPU/CUDA + KA> prototype/int32_indexing.jl

using KernelAbstractions
const MODE = get(ENV, "SR_BACKEND", "amd")
if MODE == "cuda"
    using CUDA; const GPU = CuArray; const DEVT = @eval g -> CUDA.@elapsed g(); sync() = CUDA.synchronize()
else
    using AMDGPU; const GPU = ROCArray; const DEVT = @eval g -> AMDGPU.@elapsed g(); sync() = AMDGPU.synchronize()
end
const WG = 256; const NBLK = 512; const GSZ = NBLK * WG

# Int64-indexed grid-stride reduction (baseline).
@kernel cpu=false inbounds=true unsafe_indices=true function red_i64!(out, @Const(x), n::Int64)
    I = @index(Global, Linear); L = @index(Local, Linear); G = @index(Group, Linear); g = GSZ
    a = zero(eltype(x)); b = a; c = a; d = a; e = a; ff = a; gg = a; h = a; i = I
    while i + 7g <= n; a += x[i]; b += x[i+g]; c += x[i+2g]; d += x[i+3g]; e += x[i+4g]; ff += x[i+5g]; gg += x[i+6g]; h += x[i+7g]; i += 8g; end
    while i <= n; a += x[i]; i += g; end
    acc = ((a + b) + (c + d)) + ((e + ff) + (gg + h))
    sm = @localmem eltype(x) WG; sm[L] = acc; @synchronize()
    s = WG >> 1; while s > 0; (L <= s) && (sm[L] += sm[L+s]); @synchronize(); s >>= 1; end
    (L == 1) && (out[G] = sm[1])
end

# Int32-indexed: unchecked trunc at the boundary, all loop arithmetic in Int32 (valid for n < 2^31).
@kernel cpu=false inbounds=true unsafe_indices=true function red_i32!(out, @Const(x), n32::Int32)
    I = unsafe_trunc(Int32, @index(Global, Linear)); L = @index(Local, Linear); G = @index(Group, Linear)
    g = Int32(GSZ)
    a = zero(eltype(x)); b = a; c = a; d = a; e = a; ff = a; gg = a; h = a; i = I
    while i + Int32(7) * g <= n32
        a += x[i]; b += x[i+g]; c += x[i+Int32(2)*g]; d += x[i+Int32(3)*g]
        e += x[i+Int32(4)*g]; ff += x[i+Int32(5)*g]; gg += x[i+Int32(6)*g]; h += x[i+Int32(7)*g]; i += Int32(8) * g
    end
    while i <= n32; a += x[i]; i += g; end
    acc = ((a + b) + (c + d)) + ((e + ff) + (gg + h))
    sm = @localmem eltype(x) WG; sm[L] = acc; @synchronize()
    s = WG >> 1; while s > 0; (L <= s) && (sm[L] += sm[L+s]); @synchronize(); s >>= 1; end
    (L == 1) && (out[G] = sm[1])
end

function main()
    be = get_backend(GPU(Float32[]))
    r64(o, x, n) = red_i64!(be, WG)(o, x, n; ndrange = GSZ)
    r32(o, x, n) = red_i32!(be, WG)(o, x, n; ndrange = GSZ)
    best(f) = (t = Inf; for _ in 1:40; sync(); t = min(t, DEVT(f)); end; t)
    gb(n, t) = round(n * 4 / t / 1e9)
    println("backend=$MODE")
    println("| n | Int64 GB/s | Int32 GB/s | Int32 speedup |")
    for n in (1 << 20, 1 << 22, 1 << 23, 1 << 24, 1 << 26)
        x = GPU(rand(Float32, n)); op = GPU(zeros(Float32, NBLK))
        t64 = best(() -> r64(op, x, Int64(n))); t32 = best(() -> r32(op, x, Int32(n)))
        println("| $(n >> 20)M | $(gb(n, t64)) | $(gb(n, t32)) | $(round(t64 / t32, digits=3))x |")
        x = op = nothing; GC.gc()
    end
end

main()
