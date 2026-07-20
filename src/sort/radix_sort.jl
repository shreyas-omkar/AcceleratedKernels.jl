# LSD radix sort (8-bit, 256 buckets).  Stable, GPU-only.
#
# Backend-portable: no dependency on sub-group / warp intrinsics.  Where a
# backend reports shared-memory atomics support, faster atomic-based histogram
# and scatter kernels are used; otherwise a scan-based path runs anywhere.
#
# Algorithm (per non-trivial pass over byte k):
#   1. histogram — count, per block, how many elements have each byte-digit.
#                  Layout: hist[k * B + b] = count of digit k in block b; B = num_blocks.
#                  Shared-atomic where supported (_radix_hist_atomic!), portable
#                  per-bucket scan otherwise (_radix_hist!).
#   2. accumulate! — exclusive prefix sum over hist → global per-(digit, block) offsets.
#   3. scatter — stable scatter to the offsets.  Chunked O(32)-rank where shared-memory
#                atomics are available (_radix_scatter_chunked!), O(block_size)-rank
#                broadcast scan otherwise (_radix_scatter!).
#
# Key optimizations:
#   • Fused min/max range (_rs_key_range): one reduction over the sort keys instead of
#     separate minimum + maximum.
#   • Skip-pass via min/max keys: if min and max share the whole byte-suffix from byte k
#     up, every element does too, so the whole pass is skipped (e.g. UInt32 in [0, 255]
#     sorts in a single pass).
#
# Supported element types: UInt32/64, Int32/64, Float32/64.
# Custom lt/by → falls back to merge_sort!.

import Atomix

const _RS_BITS  = UInt32(8)
const _RS_SIZE  = UInt32(256)   # 2^_RS_BITS
const _RS_CHUNK = 32            # chunked-scatter chunk width (smaller = cheaper rank; 32 best measured)


# ─── Make any supported scalar type sortable as an unsigned integer ───────────

@inline _to_sort_key(x::UInt32) = x
@inline _to_sort_key(x::UInt64) = x
@inline _to_sort_key(x::Int32)  = reinterpret(UInt32, x) ⊻ 0x80000000
@inline _to_sort_key(x::Int64)  = reinterpret(UInt64, x) ⊻ 0x8000000000000000

@inline function _to_sort_key(x::Float32)
    u = reinterpret(UInt32, x)
    mask = ((u >> 31) * 0xFFFFFFFF) | 0x80000000
    u ⊻ mask
end

@inline function _to_sort_key(x::Float64)
    u = reinterpret(UInt64, x)
    mask = ((u >> 63) * 0xFFFFFFFFFFFFFFFF) | 0x8000000000000000
    u ⊻ mask
end

@inline _rs_digit(x, shift::UInt32, rev::Bool) =
    ((rev ? ~_to_sort_key(x) : _to_sort_key(x)) >> shift) & (_RS_SIZE - 0x1)


# ─── Phase 1: per-pass histogram — generic scan (all backends) ───────────────
# hist[k * num_blocks + b] = count of elements with digit k in block b.
#
# Each thread loads its element's digit into s_digit, then scans s_digit for each
# of its assigned buckets (bucket t, t+NI, t+2*NI, …).  Uses no shared-memory
# atomics, so it is the portable fallback that runs on every backend.

@kernel inbounds=true cpu=false unsafe_indices=true function _radix_hist!(
    hist, @Const(v), shift::UInt32, rev::Bool,
)
    @uniform NI = Int(@groupsize()[1])
    s_digit = @localmem UInt32 (NI,)

    iblock  = Int(@index(Group, Linear)) - 1
    ithread = Int(@index(Local, Linear)) - 1
    len        = Int(length(v))
    num_blocks = Int(length(hist)) ÷ Int(_RS_SIZE)

    # 0xffffffff doesn't match any valid bucket (0–255); OOB elements are neutral.
    i = iblock * NI + ithread
    s_digit[ithread + 1] = UInt32(i < len ? _rs_digit(v[i + 1], shift, rev) : 0xffffffff)
    @synchronize()

    # Thread t handles buckets t, t+NI, t+2*NI, … (covers all 256 when NI ≤ 256).
    bucket = ithread
    while bucket < Int(_RS_SIZE)
        cnt = UInt32(0)
        for jj in 1:NI
            cnt += UInt32(s_digit[jj] == UInt32(bucket))
        end
        hist[bucket * num_blocks + iblock + 1] = cnt
        bucket += NI
    end
end


# ─── Phase 1b: histogram — shared-atomic (backends with shared-mem atomics) ──
# Same per-block 256-bin output, but O(1)/element via a shared-memory atomic
# increment instead of the O(256)/element per-bucket scan above.  Selected by
# the driver only where the backend reports atomics support.

@kernel inbounds=true cpu=false unsafe_indices=true function _radix_hist_atomic!(
    hist, @Const(v), shift::UInt32, rev::Bool,
)
    @uniform NI = Int(@groupsize()[1])
    s_hist = @localmem UInt32 (Int(_RS_SIZE),)

    iblock  = Int(@index(Group, Linear)) - 1
    ithread = Int(@index(Local, Linear)) - 1
    len        = Int(length(v))
    num_blocks = Int(length(hist)) ÷ Int(_RS_SIZE)

    j = ithread
    while j < Int(_RS_SIZE)
        s_hist[j + 1] = UInt32(0)
        j += NI
    end
    @synchronize()

    i = iblock * NI + ithread
    if i < len
        d = Int(_rs_digit(v[i + 1], shift, rev))
        Atomix.@atomic s_hist[d + 1] += UInt32(1)
    end
    @synchronize()

    bucket = ithread
    while bucket < Int(_RS_SIZE)
        hist[bucket * num_blocks + iblock + 1] = s_hist[bucket + 1]
        bucket += NI
    end
end


# ─── Phase 3: scatter — broadcast-read rank (O(N) per thread) ───────────────
# `hist` is already the exclusive-prefix-summed per-block offsets;
# `hist[k * num_blocks + b]` = global start for bucket k, block b (1-indexed).

@kernel inbounds=true cpu=false unsafe_indices=true function _radix_scatter!(
    v_out, @Const(v_in), @Const(hist), shift::UInt32, rev::Bool,
)
    @uniform N   = @groupsize()[1]
    @uniform NI  = Int(@groupsize()[1])
    s_elem  = @localmem eltype(v_in) (N,)
    s_digit = @localmem UInt32       (N,)
    s_gbase = @localmem UInt32       (256,)

    iblock  = Int(@index(Group, Linear)) - 1
    ithread = Int(@index(Local, Linear)) - 1
    len        = Int(length(v_in))
    num_blocks = Int(length(hist)) ÷ 256

    i = iblock * NI + ithread
    if i < len
        s_elem[ithread + 1] = v_in[i + 1]
    end
    j = ithread
    while j < 256
        s_gbase[j + 1] = hist[j * num_blocks + iblock + 1]
        j += NI
    end
    @synchronize()

    my_digit = UInt32(i < len ? _rs_digit(s_elem[ithread + 1], shift, rev) : 0)
    s_digit[ithread + 1] = my_digit
    @synchronize()

    if i < len
        cnt = UInt32(0)
        for jj in UInt32(1):UInt32(ithread)
            cnt += UInt32(s_digit[jj] == my_digit)
        end
        gpos = Int(s_gbase[my_digit + 1]) + Int(cnt)
        v_out[gpos + 1] = s_elem[ithread + 1]
    end
end


# ─── Phase 3b: scatter — chunked stable rank (O(chunk) per thread) ───────────
# The broadcast scatter's rank is O(block_size)/element.  Split the block into
# 32-wide chunks: per-chunk digit counts (built with shared-memory atomics —
# order doesn't matter for counts) give each chunk's stable base via a cross-chunk
# exclusive prefix, and each element only scans its own chunk (≤32) for the
# intra-chunk part.  Rank = cross-chunk-base[digit] + intra-chunk same-digit
# count — still fully stable, but O(32) instead of O(block_size).

@kernel inbounds=true cpu=false unsafe_indices=true function _radix_scatter_chunked!(
    v_out, @Const(v_in), @Const(hist), shift::UInt32, rev::Bool,
)
    @uniform N   = @groupsize()[1]
    @uniform NI  = Int(@groupsize()[1])
    @uniform NCH = Int(@groupsize()[1]) ÷ _RS_CHUNK   # number of chunks
    s_elem  = @localmem eltype(v_in) (N,)
    s_digit = @localmem UInt32       (N,)
    s_gbase = @localmem UInt32       (256,)
    s_chist = @localmem UInt32       (256 * NCH,)      # per-chunk digit counts → bases

    iblock  = Int(@index(Group, Linear)) - 1
    ithread = Int(@index(Local, Linear)) - 1
    len        = Int(length(v_in))
    num_blocks = Int(length(hist)) ÷ 256

    i = iblock * NI + ithread
    if i < len
        s_elem[ithread + 1] = v_in[i + 1]
    end
    j = ithread
    while j < 256
        s_gbase[j + 1] = hist[j * num_blocks + iblock + 1]
        j += NI
    end
    j = ithread
    while j < 256 * NCH
        s_chist[j + 1] = UInt32(0)
        j += NI
    end
    @synchronize()

    mychunk  = ithread ÷ _RS_CHUNK
    my_digit = UInt32(i < len ? _rs_digit(s_elem[ithread + 1], shift, rev) : 0)
    s_digit[ithread + 1] = my_digit
    if i < len
        Atomix.@atomic s_chist[mychunk * 256 + Int(my_digit) + 1] += UInt32(1)
    end
    @synchronize()

    # cross-chunk exclusive prefix per digit (thread d owns digit d, d+NI, …)
    d = ithread
    while d < 256
        acc = UInt32(0)
        for c in 0:NCH-1
            cnt = s_chist[c * 256 + d + 1]
            s_chist[c * 256 + d + 1] = acc
            acc += cnt
        end
        d += NI
    end
    @synchronize()

    if i < len
        chunk_start = mychunk * _RS_CHUNK
        cnt = UInt32(0)
        for jj in chunk_start:(ithread - 1)
            cnt += UInt32(s_digit[jj + 1] == my_digit)
        end
        rank = s_chist[mychunk * 256 + Int(my_digit) + 1] + cnt
        gpos = Int(s_gbase[my_digit + 1]) + Int(rank)
        v_out[gpos + 1] = s_elem[ithread + 1]
    end
end


# ─── Onesweep (experimental) ─────────────────────────────────────────────────
# Fuses the global prefix-sum into the scatter, cutting the per-pass kernel launches from
# hist + accumulate!(scan) + scatter down to hist + tiny-scan + scatter — a win on
# dispatch-bound backends (e.g. Metal).  Opt-in via ENV["AK_RADIX_ONESWEEP"]; needs shared-
# and global-memory atomics plus a device-scope fence, so the driver only selects it where
# the backend reports atomics support.  Per pass:
#   1. _radix_hist_os!  — per-tile digit histogram AND global per-digit totals, one pass.
#   2. _radix_digit_base! — exclusive scan of the 256 totals → per-digit region base.
#   3. _radix_scatter_onesweep! — each tile finds its exclusive cross-tile prefix per digit by
#      a NON-SPINNING decoupled look-back (accumulate predecessors' published counts until an
#      inclusive aggregate is found; never busy-waits), then scatters.  The look-back only
#      reads already-computed histogram values, so it always makes progress — no forward-
#      progress guarantee needed, unlike a spin-based chain (safe on Metal / weaker backends).

# The look-back publish/consume uses exactly the same cross-block coherence primitives as the
# decoupled-lookback scan — see the commentary above `_decoupled_fence` in
# accumulate/accumulate_1d_gpu.jl for why both a relaxed L1-bypassing access *and* a fence are
# required.  They are shared rather than duplicated so there is a single audited implementation
# per backend, and so the CUDA device-scope override applies here too.


# Per-tile digit histogram hist[d*num_blocks+b] plus global per-digit totals digit_total[d],
# in a single pass over the data (shared-mem atomic tile histogram; global atomic to totals).
@kernel inbounds=true cpu=false unsafe_indices=true function _radix_hist_os!(
    hist, digit_total, @Const(v), shift::UInt32, rev::Bool,
)
    @uniform NI = Int(@groupsize()[1])
    s_hist = @localmem UInt32 (Int(_RS_SIZE),)

    iblock  = Int(@index(Group, Linear)) - 1
    ithread = Int(@index(Local, Linear)) - 1
    len        = Int(length(v))
    num_blocks = Int(length(hist)) ÷ Int(_RS_SIZE)

    j = ithread
    while j < Int(_RS_SIZE)
        s_hist[j + 1] = UInt32(0)
        j += NI
    end
    @synchronize()

    i = iblock * NI + ithread
    if i < len
        d = Int(_rs_digit(v[i + 1], shift, rev))
        Atomix.@atomic s_hist[d + 1] += UInt32(1)
    end
    @synchronize()

    j = ithread
    while j < Int(_RS_SIZE)
        c = s_hist[j + 1]
        hist[j * num_blocks + iblock + 1] = c
        if c != UInt32(0)
            Atomix.@atomic digit_total[j + 1] += c
        end
        j += NI
    end
end


# Exclusive scan of the 256 digit totals → digit_base[d] = start offset of digit d's region.
# Single block; the 256-element serial scan by one thread is negligible (≤8 passes total).
@kernel inbounds=true cpu=false unsafe_indices=true function _radix_digit_base!(
    digit_base, @Const(digit_total),
)
    ithread = Int(@index(Local, Linear)) - 1
    if ithread == 0
        acc = UInt32(0)
        for d in 1:Int(_RS_SIZE)
            digit_base[d] = acc
            acc += digit_total[d]
        end
    end
end


# Fused onesweep scatter with a NON-SPINNING decoupled look-back.  `incl`/`flags` are length
# 256*num_blocks (per tile, per digit); `flags` must be zero-initialised before launch.  `incl`
# needs no init: a consumer only reads incl[b',d] after observing flags[b',d] == 1, set after
# the write to incl[b',d].
#
# For its digit d, tile b walks predecessors b-1, b-2, …: if a predecessor has published its
# inclusive aggregate (flag == 1) we add it and stop; otherwise we add that predecessor's plain
# (already-computed) tile count `hist[d,b']` and keep walking.  Because `hist` is fully computed
# by _radix_hist_os! before this kernel, the walk ALWAYS makes progress and never busy-waits, so
# no cross-block forward-progress guarantee is required (safe on Metal).
@kernel inbounds=true cpu=false unsafe_indices=true function _radix_scatter_onesweep!(
    v_out, @Const(v_in), @Const(hist), @Const(digit_base), incl, flags,
    shift::UInt32, rev::Bool,
)
    @uniform N   = @groupsize()[1]
    @uniform NI  = Int(@groupsize()[1])
    @uniform NCH = Int(@groupsize()[1]) ÷ _RS_CHUNK
    s_elem  = @localmem eltype(v_in) (N,)
    s_digit = @localmem UInt32       (N,)
    s_chist = @localmem UInt32       (256 * NCH,)   # per-chunk digit bases (intra-tile rank)
    s_gbase = @localmem UInt32       (256,)         # global base per digit for this tile

    iblock  = Int(@index(Group, Linear)) - 1
    ithread = Int(@index(Local, Linear)) - 1
    len        = Int(length(v_in))
    num_blocks = Int(length(flags)) ÷ 256

    i = iblock * NI + ithread
    if i < len
        s_elem[ithread + 1] = v_in[i + 1]
    end
    j = ithread
    while j < 256 * NCH
        s_chist[j + 1] = UInt32(0)
        j += NI
    end
    @synchronize()

    mychunk  = ithread ÷ _RS_CHUNK
    my_digit = UInt32(i < len ? _rs_digit(s_elem[ithread + 1], shift, rev) : 0)
    s_digit[ithread + 1] = my_digit
    if i < len
        Atomix.@atomic s_chist[mychunk * 256 + Int(my_digit) + 1] += UInt32(1)
    end
    @synchronize()

    # Per digit (thread d owns digit d, d+NI, …): intra-tile cross-chunk exclusive prefix, then
    # the non-spinning cross-tile look-back for the exclusive per-digit prefix.
    d = ithread
    while d < 256
        # cross-chunk exclusive prefix; `local_count` ends as this tile's total count of digit d
        local_count = UInt32(0)
        for c in 0:NCH-1
            cnt = s_chist[c * 256 + d + 1]
            s_chist[c * 256 + d + 1] = local_count
            local_count += cnt
        end

        # non-spinning decoupled look-back over predecessor tiles for digit d
        excl_partial = UInt32(0)
        excl = UInt32(0)
        found = false
        b = iblock - 1
        while b >= 0
            if _load_relaxed(flags, b * 256 + d + 1) == UInt8(1)
                _decoupled_fence()                                     # value load after flag load
                excl = _load_relaxed(incl, b * 256 + d + 1) + excl_partial
                found = true
                break
            else
                excl_partial += hist[d * num_blocks + b + 1]    # plain count, always available
            end
            b -= 1
        end
        if !found
            excl = digit_base[d + 1] + excl_partial
        end

        _store_relaxed!(incl, iblock * 256 + d + 1, excl + local_count)   # publish inclusive prefix
        _decoupled_fence()                                                 # value before flag
        _store_relaxed!(flags, iblock * 256 + d + 1, UInt8(1))
        s_gbase[d + 1] = excl
        d += NI
    end
    @synchronize()

    # Scatter with the intra-tile chunked stable rank.
    if i < len
        chunk_start = mychunk * _RS_CHUNK
        cnt = UInt32(0)
        for jj in chunk_start:(ithread - 1)
            cnt += UInt32(s_digit[jj + 1] == my_digit)
        end
        rank = s_chist[mychunk * 256 + Int(my_digit) + 1] + cnt
        gpos = Int(s_gbase[Int(my_digit) + 1]) + Int(rank)
        v_out[gpos + 1] = s_elem[ithread + 1]
    end
end


# ─── Implementation ──────────────────────────────────────────────────────────

_rs_supported(::Type{T}) where T =
    T === UInt32 || T === Int32 || T === Float32 ||
    T === UInt64 || T === Int64 || T === Float64


# Return (min_sort_key, max_sort_key) as UInt64, accounting for descending order.
# Used to detect passes where all elements share the same byte-digit (trivial pass).
#
# Single fused reduction: map each element to its (order-preserving) unsigned sort
# key and reduce to (min_key, max_key) in one pass, instead of two separate full
# reductions over the array (minimum + maximum).  ~halves the range-finding cost.
function _rs_key_range(v::AbstractArray{T}, descending::Bool) where T
    K = typeof(_to_sort_key(zero(T)))   # UInt32 for 32-bit types, UInt64 for 64-bit
    ident = (typemax(K), typemin(K))   # identity for (min, max) over keys
    min_k, max_k = mapreduce(
        x -> (k = _to_sort_key(x); (k, k)),
        (a, b) -> (min(a[1], b[1]), max(a[2], b[2])),
        v;
        init=ident,
        neutral=ident,
    )
    if descending
        # rev=true flips all bits: digit = (~key >> shift) & 0xFF
        # Maximum value → minimum key after negation; swap accordingly.
        UInt64(~max_k), UInt64(~min_k)
    else
        UInt64(min_k), UInt64(max_k)
    end
end


"""
    _radix_sort!(v, backend; lt, by, rev, order, block_size, temp)

In-place GPU LSD radix sort (8-bit, 256 buckets per pass).  Supported types:
`UInt32`, `Int32`, `Float32`, `UInt64`, `Int64`, `Float64`.  Falls back to
[`merge_sort!`](@ref) for any other type or when `lt`/`by` are non-default.
"""
function _radix_sort!(
    v::AbstractArray{T}, backend::Backend=get_backend(v);
    lt=isless,
    by=identity,
    rev::Union{Nothing, Bool}=nothing,
    order::Base.Order.Ordering=Base.Forward,
    block_size::Int=256,
    temp::Union{Nothing, AbstractArray}=nothing,
    onesweep::Bool=haskey(ENV, "AK_RADIX_ONESWEEP"),
) where T

    if !_rs_supported(T) || lt !== isless || by !== identity
        return merge_sort!(v, backend; lt, by, rev, order, block_size, temp)
    end

    n = length(v)
    n <= 1 && return v

    @argcheck ispow2(block_size) && block_size >= 1

    descending = (rev === true) || order === Base.Order.Reverse

    num_blocks = cld(n, block_size)
    n_passes   = sizeof(T) * 8 ÷ Int(_RS_BITS)   # 4 for 32-bit, 8 for 64-bit

    # Experimental onesweep path (opt-in): fuses the prefix-sum into the scatter.  Needs
    # shared/global atomics + a device-scope fence, and block_size a multiple of the chunk.
    # Requested via the `onesweep` keyword (default: the AK_RADIX_ONESWEEP env var, which lets
    # a whole test/benchmark run be flipped over without touching call sites); silently
    # declined where the backend cannot support it.
    has_atomics = KernelAbstractions.supports_atomics(backend)
    onesweep = onesweep && has_atomics && block_size % _RS_CHUNK == 0

    if onesweep
        hist        = similar(v, UInt32, Int(_RS_SIZE) * num_blocks)
        digit_total = similar(v, UInt32, Int(_RS_SIZE))
        digit_base  = similar(v, UInt32, Int(_RS_SIZE))
        incl        = similar(v, UInt32, Int(_RS_SIZE) * num_blocks)
        flags       = similar(v, UInt8,  Int(_RS_SIZE) * num_blocks)
    else
        # Single histogram buffer; no need to zero before each pass — the histogram kernel
        # zero-initializes its own shared-memory histogram and writes directly here.
        hist = similar(v, UInt32, Int(_RS_SIZE) * num_blocks)

        # Reusable scratch for accumulate!'s per-block prefixes (ScanPrefixes uses a
        # 256-thread, 2-elems-per-thread grid → 512 elements per block), so the
        # exclusive prefix sum does not re-allocate on every pass.
        acc_temp = similar(v, UInt32, cld(length(hist), 512))
    end

    p1 = v
    p2 = if !isnothing(temp)
        @argcheck length(temp) >= n && eltype(temp) === T
        temp
    else
        similar(v)
    end

    ndrange = (block_size * num_blocks,)

    # Compute (min, max) sort-key to detect passes where all elements share the
    # same digit → skip the full hist+scan+scatter for that byte position.
    min_key, max_key = _rs_key_range(p1, descending)

    # The fast histogram (O(1)/element atomic counting) and the chunked scatter
    # (per-chunk sub-histograms) both use shared-memory atomics; select them
    # where the backend reports atomics support and fall back to the portable
    # scan/broadcast kernels otherwise.  The chunked scatter additionally needs
    # block_size to be a multiple of its 32-wide chunk.
    if onesweep
        hist_os_kern! = _radix_hist_os!(backend, block_size)
        dbase_kern!   = _radix_digit_base!(backend, block_size)
        os_kern!      = _radix_scatter_onesweep!(backend, block_size)
    else
        hist_kern! = has_atomics ?
            _radix_hist_atomic!(backend, block_size) :
            _radix_hist!(backend, block_size)
        scat_kern! = (has_atomics && block_size % _RS_CHUNK == 0) ?
            _radix_scatter_chunked!(backend, block_size) :
            _radix_scatter!(backend, block_size)
    end

    n_actual = 0

    for pass in 0:n_passes - 1
        shift = UInt64(pass) * UInt64(_RS_BITS)

        # Trivial pass: all elements share the same byte k AND all higher bytes
        # are also identical (so no element can have a different byte k).
        # Sufficient condition: (min_key >> shift) == (max_key >> shift).
        # Checking just the single byte is WRONG when higher bytes differ.
        (min_key >> shift) == (max_key >> shift) && continue

        shift32 = UInt32(shift)
        # Kernels run in order on a single backend stream, so no host synchronization is
        # needed between them.
        if onesweep
            # digit_total accumulates via atomics and flags gates the look-back, so both start
            # zeroed each pass; incl is written before it is read.
            fill!(digit_total, UInt32(0))
            fill!(flags, UInt8(0))
            hist_os_kern!(hist, digit_total, p1, shift32, descending; ndrange)
            dbase_kern!(digit_base, digit_total; ndrange=(block_size,))
            os_kern!(p2, p1, hist, digit_base, incl, flags, shift32, descending; ndrange)
        else
            hist_kern!(hist, p1, shift32, descending; ndrange)
            accumulate!(+, hist, backend; init=UInt32(0), inclusive=false, temp=acc_temp)
            scat_kern!(p2, p1, hist, shift32, descending; ndrange)
        end

        p1, p2 = p2, p1
        n_actual += 1
    end

    # p1 holds the result; copy back only if it's in the temp buffer.
    if isodd(n_actual)
        copyto!(v, p1)
    end

    # Block once so the sort is complete on return; the passes only enqueue work.
    KernelAbstractions.synchronize(backend)

    v
end
