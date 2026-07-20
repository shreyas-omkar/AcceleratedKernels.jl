# Standalone correctness + dispatch benchmark for the experimental radix onesweep path.
# NOT part of runtests.jl — run it directly against whichever backend you have:
#
#     julia --project -e 'using Metal;  include("test/onesweep_check.jl"); run_all(MetalBackend(), MtlArray)'
#     julia --project -e 'using CUDA;   include("test/onesweep_check.jl"); run_all(CUDABackend(), CuArray)'
#     julia --project -e 'using oneAPI; include("test/onesweep_check.jl"); run_all(oneAPIBackend(), oneArray)'
#
# WHAT THIS IS TESTING
#
# The default radix sort runs three kernels per 8-bit pass: histogram, a global
# exclusive scan (`accumulate!`), and scatter. Onesweep fuses the global scan into the
# scatter using a non-spinning decoupled look-back, so a pass costs two launches instead
# of three. Measured on an RTX 3060, that is 23 -> 15 launches for Int32 and 43 -> 27 for
# Int64 (~36% fewer dispatches).
#
# It trades device-memory traffic and a cross-block dependency for those saved launches,
# so it is expected to LOSE on CUDA and to WIN only where dispatch dominates. On the 3060
# that crossover is already visible: onesweep is 0.84x (faster) at n=10_000 where launch
# overhead dominates, but 3.0x slower by n=1_000_000 where traffic dominates.
#
# THE QUESTION FOR METAL: where does that crossover sit? If Metal is dispatch-bound as
# expected, the win should extend to much larger n than it does on CUDA — possibly across
# the whole range. If onesweep loses at every size on Metal too, the approach is dead and
# we should say so.
#
# Correctness has been validated on CUDA (352/352: 6 types x 16 sizes, descending,
# duplicate-heavy, skip-pass, plus 150 stress repeats) and on OpenCL/SPIR-V via POCL.
# Metal is unvalidated only because no Metal device was available.

using AcceleratedKernels, KernelAbstractions, Random, Printf
const AK = AcceleratedKernels
const KA = KernelAbstractions


function check_correctness(backend, AT)
    Random.seed!(0)
    npass = 0; nfail = 0
    chk(c, m) = c ? (npass += 1) : (nfail += 1; println("  FAIL: ", m))

    sizes = [1, 2, 31, 32, 255, 256, 257, 511, 512, 513, 1000, 4096, 65_536, 65_537, 1_000_003]
    types = (Int32, UInt32, Float32, Int64, UInt64, Float64)

    println("== correctness ==")
    for T in types, n in sizes
        h = rand(T, n)
        d = AT(h)
        try
            AK.sort!(d; alg=AK.RadixSort(onesweep=true))
            KA.synchronize(backend)
            chk(Array(d) == sort(h), "onesweep $T n=$n")
        catch e
            chk(false, "onesweep $T n=$n THREW: $(first(sprint(showerror, e), 200))")
        end
    end

    # Duplicate-heavy keys stress the look-back and the stable rank hardest.
    for n in (10_000, 1_000_003)
        h = rand(Int32(1):Int32(16), n)
        d = AT(h)
        AK.sort!(d; alg=AK.RadixSort(onesweep=true)); KA.synchronize(backend)
        chk(Array(d) == sort(h), "duplicate-heavy n=$n")
    end

    # The look-back is racy by construction: a single pass proves little.
    for rep in 1:50
        h = rand(Int32, 1_000_003)
        d = AT(h)
        AK.sort!(d; alg=AK.RadixSort(onesweep=true)); KA.synchronize(backend)
        chk(Array(d) == sort(h), "stress rep=$rep")
    end

    @printf("  passed=%d failed=%d -> %s\n\n", npass, nfail,
            nfail == 0 ? "OK" : "FAILURES")
    nfail
end


function bench(backend, AT)
    Random.seed!(0)
    function tmin(f, h; reps=10)
        d = AT(h); f(d); KA.synchronize(backend)
        best = Inf
        for _ in 1:reps
            d = AT(h); KA.synchronize(backend)
            t0 = time_ns(); f(d); KA.synchronize(backend)
            best = min(best, (time_ns() - t0) / 1e9)
        end
        best
    end

    println("== throughput: onesweep vs default ==")
    println("   os/def < 1.0 means onesweep WINS. Watch for where it crosses 1.0.")
    @printf("%-9s %10s %12s %12s %9s\n", "T", "n", "default", "onesweep", "os/def")
    for T in (Int32, Float32, Int64)
        for n in (10_000, 100_000, 1_000_000, 4_000_000, 16_000_000)
            h = rand(T, n)
            td = tmin(d -> AK.sort!(d; alg=AK.RadixSort(onesweep=false)), h)
            to = tmin(d -> AK.sort!(d; alg=AK.RadixSort(onesweep=true)),  h)
            @printf("%-9s %10d %10.3fms %10.3fms %8.2fx%s\n", T, n, td*1e3, to*1e3, to/td,
                    to < td ? "  <-- onesweep wins" : "")
            GC.gc()
        end
    end
end


function run_all(backend, AT)
    println("backend: ", backend)
    println("supports_atomics: ", KA.supports_atomics(backend))
    println()
    nfail = check_correctness(backend, AT)
    nfail == 0 || (println("correctness failed — skipping benchmark"); return)
    bench(backend, AT)
end
