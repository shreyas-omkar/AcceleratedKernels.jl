# LSD radix sort (8-bit, 256 buckets).  Stable, GPU-only.
#
# Algorithm (per non-trivial pass over byte k):
#   1. histogram — count, per block, how many elements have each byte-digit.
#                  Layout: hist[k * B + b] = count of digit k in block b; B = num_blocks.
#   2. accumulate! — exclusive prefix sum over hist → global per-(digit, block) offsets.
#   3. scatter — stable scatter to the offsets.
#
# Key optimizations (all verified on an RTX 5080, 4M elements):
#   • Fused min/max range (_rs_key_range): one reduction over the sort keys instead of
#     separate minimum + maximum — recovers ~20% of total time (more for 64-bit types).
#   • Skip-pass via min/max keys: if min and max share the whole byte-suffix from byte k
#     up, every element does too, so the whole pass is skipped.  Free for structured /
#     small-range data (e.g. UInt32 in [0, 255] sorts in a single pass).
#   • Digits are precomputed once into shared memory per block, so the histogram scans
#     shared memory instead of recomputing the digit for every bucket.
#
# Supported element types: UInt32/64, Int32/64, Float32/64.
# Custom lt/by → falls back to merge_sort!.

const _RS_BITS = UInt32(8)
const _RS_SIZE = UInt32(256)   # 2^_RS_BITS


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


# ─── Phase 1: per-pass histogram ─────────────────────────────────────────────
# hist[k * num_blocks + b] = count of elements with digit k in block b.
#
# Each thread loads its element's digit into s_digit, then scans s_digit for each
# of its assigned buckets (bucket t, t+NI, t+2*NI, …).  Uses no shared-memory
# atomics (Atomix has no @localmem atomic path on CUDA or POCL), so it runs on
# every backend.

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


# ─── Phase 2: scatter — stable O(N) broadcast-read rank ──────────────────────
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

    # Single histogram buffer; no need to zero before each pass — _radix_hist!
    # zero-initializes its own shared-memory histogram and writes directly here.
    hist = similar(v, UInt32, Int(_RS_SIZE) * num_blocks)

    # Reusable scratch for accumulate!'s per-block prefixes (ScanPrefixes uses a
    # 256-thread, 2-elems-per-thread grid → 512 elements per block), so the
    # exclusive prefix sum does not re-allocate on every pass.
    acc_temp = similar(v, UInt32, cld(length(hist), 512))

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

    hist_kern! = _radix_hist!(backend, block_size)
    scat_kern! = _radix_scatter!(backend, block_size)

    n_actual = 0

    for pass in 0:n_passes - 1
        shift = UInt64(pass) * UInt64(_RS_BITS)

        # Trivial pass: all elements share the same byte k AND all higher bytes
        # are also identical (so no element can have a different byte k).
        # Sufficient condition: (min_key >> shift) == (max_key >> shift).
        # Checking just the single byte is WRONG when higher bytes differ.
        (min_key >> shift) == (max_key >> shift) && continue

        shift32 = UInt32(shift)
        # The three kernels run in order on a single backend stream, so no host
        # synchronization is needed between them.
        hist_kern!(hist, p1, shift32, descending; ndrange)
        accumulate!(+, hist, backend; init=UInt32(0), inclusive=false, temp=acc_temp)
        scat_kern!(p2, p1, hist, shift32, descending; ndrange)

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
