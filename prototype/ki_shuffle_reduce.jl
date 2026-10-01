# Prototype: subgroup-shuffle block reduction via KernelInterface (KI).
#
# AK's block reduction (`reduce_group!`) folds a workgroup's values in shared memory with a
# `@synchronize()` at every halving step (log2(block_size) barriers). This prototype replaces the
# intra-subgroup part with `KI.shfl_down` (no barrier, subgroup-synchronous) and uses a single
# barrier to combine the per-subgroup partials, cutting the barrier count from log2(block_size) to 1.
#
# FINDING (measured on AMD RDNA4 gfx1200 and NVIDIA RTX 5080): for a full-array reduction this is a
# WASH, not a speedup. A whole-array reduce is memory-bandwidth bound (dominated by the grid-stride
# read of N), and both the shuffle and the shared-memory tree already reach full occupancy, so the
# cheaper block-reduce is noise against the data read:
#   AMD  64M Float32: shuffle 0.86 ms vs tree 0.86 ms  (both ~299 GB/s, bandwidth wall)
# The shuffle path's real advantage is resource use, not time: ~20x less shared memory (one slot per
# subgroup vs one per thread) and 1 barrier vs 8. That only turns into a speedup if a reduction is
# FUSED into a kernel whose occupancy is shared-memory limited (e.g. a radix scatter). As a drop-in
# for the standalone reduce it gives no gain.
#
# It is kept here as a reviewable reference for whether AK should adopt KI subgroup intrinsics: the
# `shfl_down` path is correct and portable across backends that implement it (via KI's @device_override),
# and compiles inside a normal launch. KI is not yet released, so this is a prototype, not an integration.
#
# Run: SR_BACKEND=amd|cuda julia --project=<env with KI + AK + a backend> prototype/ki_shuffle_reduce.jl

import KernelInterface as KI

const MODE = get(ENV, "SR_BACKEND", "amd")
if MODE == "cuda"
    using CUDA; const GPU = CuArray; const DEVT = @eval g -> CUDA.@elapsed g(); sync() = CUDA.synchronize()
else
    using AMDGPU; const GPU = ROCArray; const DEVT = @eval g -> AMDGPU.@elapsed g(); sync() = AMDGPU.synchronize()
end

const WG   = 256
const SG   = 32          # subgroup / wave size (RDNA wave32, CUDA warp 32)
const NSUB = WG ÷ SG     # per-subgroup partials to combine
const NBLK = 512

# Grid-stride accumulate, then: intra-subgroup shfl_down reduction (no barrier), one barrier, and a
# final shfl_down reduction of the NSUB partials in subgroup 0. Each block writes out[group_id].
function shfl_reduce_kernel!(out, x, n::Int64)
    i = KI.get_global_id().x
    g = KI.get_global_size().x
    acc = zero(eltype(x))
    @inbounds while i <= n; acc += x[i]; i += g; end

    off = 0x00000001
    while off < UInt32(SG); acc += KI.shfl_down(acc, off); off <<= 1; end

    smem = KI.localmemory(eltype(x), Val(NSUB))
    slid = KI.get_sub_group_local_id()          # 1-based lane id within the subgroup
    sgid = KI.get_sub_group_id()                # 1-based subgroup id within the block
    @inbounds if slid == 1; smem[sgid] = acc; end
    KI.barrier()

    if sgid == 1
        v = (slid <= NSUB) ? (@inbounds smem[slid]) : zero(eltype(x))
        off = 0x00000001
        while off < UInt32(NSUB); v += KI.shfl_down(v, off); off <<= 1; end
        @inbounds if slid == 1; out[KI.get_group_id().x] = v; end
    end
    return
end

function main()
    be = KI.get_backend(GPU(Float32[]))
    run(o, x, n) = KI.@kernel be numworkgroups=NBLK workgroupsize=WG shfl_reduce_kernel!(o, x, n)
    best(f) = (t = Inf; for _ in 1:30; sync(); t = min(t, DEVT(f)); end; t)
    println("backend=$MODE  WG=$WG  SG=$SG  NSUB=$NSUB")
    println("| n | shuffle reduce ms | GB/s | correct |")
    for n in (1 << 20, 1 << 24, 1 << 26)
        x = GPU(rand(Float32, n)); o = GPU(zeros(Float32, NBLK)); nn = Int64(n)
        run(o, x, nn); sync()
        ok = isapprox(sum(Array(o)), sum(Array(x)); rtol = 1e-3)
        t = best(() -> run(o, x, nn))
        println("| $(n >> 20)M | $(round(t * 1e3, digits=4)) | $(round(n * 4 / t / 1e9)) | $ok |")
    end
end

main()
