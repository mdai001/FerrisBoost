#include <stdint.h>

// Exact split evaluation. Integer scans are explicitly modulo 2^64; floating
// operations use round-to-nearest intrinsics (no contraction/reassociation).
struct SplitPair { int64_t grad, hess; };
struct SplitPacket {
    uint32_t feat, bin; float gain; uint32_t flags;
    SplitPair left, right;
    float left_weight, right_weight, left_hess, right_hess;
};
static_assert(sizeof(SplitPacket) == 64, "split packet ABI");
// Categorical split capacity. One CTA handles one feature with one thread per
// bin, so a categorical feature may have up to SPLIT_CTA_THREADS bins, and a
// category set is CAT_SET_WORDS 32-bit words (a bitset over those bins). Both
// size the shared scratch, the per-winner set output and the packet tail, and
// must match `gpu_mem_plan::CAT_SET_WORDS` (checked by a unit test). Wider
// categorical features (more than SPLIT_CTA_THREADS bins) need these raised
// and the per-thread bin loops below turned into strided loops.
#define SPLIT_CTA_THREADS 256
#define CAT_SET_WORDS 8
static_assert(CAT_SET_WORDS * 32 >= SPLIT_CTA_THREADS, "set must cover every bin");
__device__ SplitPair split_add(SplitPair a, SplitPair b) {
    return {(int64_t)((uint64_t)a.grad + (uint64_t)b.grad),
            (int64_t)((uint64_t)a.hess + (uint64_t)b.hess)};
}
__device__ SplitPair split_sub(SplitPair a, SplitPair b) {
    return {(int64_t)((uint64_t)a.grad - (uint64_t)b.grad),
            (int64_t)((uint64_t)a.hess - (uint64_t)b.hess)};
}
__device__ double split_value(int64_t x, double inv) {
    return __dmul_rn(__ll2double_rn(x), inv);
}
__device__ double split_gain(SplitPair a, double ig, double ih, float lambda) {
    double g = split_value(a.grad, ig), h = split_value(a.hess, ih);
    return __ddiv_rn(__dmul_rn(g, g), __dadd_rn(h, (double)lambda));
}
__device__ SplitPacket split_candidate(SplitPair l, SplitPair r,
    uint32_t feat, uint32_t bin, bool missing, double parent,
    double ig, double ih, float lambda, float gamma, float mcw) {
    SplitPacket p{};
    if (split_value(l.hess, ih) < (double)mcw || split_value(r.hess, ih) < (double)mcw)
        return p;
    float gain = __double2float_rn(__dsub_rn(__dsub_rn(
        __dadd_rn(split_gain(l, ig, ih, lambda), split_gain(r, ig, ih, lambda)),
        parent), (double)gamma));
    if (!(gain > 1.e-6f)) return p;
    p.feat = feat; p.bin = bin; p.gain = gain;
    p.flags = 1u | (missing ? 2u : 0u); p.left = l; p.right = r;
    return p;
}
// Monotone constraints (XGBoost 3.4.1 single-output hist semantics). Each
// rounding step mirrors src/split.rs `bounded_split_gain` exactly; explicit
// _rn intrinsics forbid FMA contraction. Unused when has_monotone == 0.
// XGBoost ThresholdL1 + CalcWeight (param.h): alpha == 0 and
// max_delta_step == 0 reduce to -g / (h + lambda) bit for bit.
__device__ double mono_threshold_l1(double g, float alpha) {
    double a = (double)alpha;
    if (g > a) return __dsub_rn(g, a);
    if (g < -a) return __dadd_rn(g, a);
    return 0.0;
}
__device__ float mono_weight(double g, double h, float lambda, float alpha, float mds) {
    if (h <= 0.0) return 0.0f;
    double dw = __ddiv_rn(-mono_threshold_l1(g, alpha), __dadd_rn(h, (double)lambda));
    if (mds != 0.0f && fabs(dw) > (double)mds) dw = copysign((double)mds, dw);
    return __double2float_rn(dw);
}
__device__ float mono_apply(float w, float lower, float upper) {
    if (w < lower) return lower;
    if (w > upper) return upper;
    return w;
}
__device__ float mono_gain_given_weight(double g, double h, float w, float lambda,
    float alpha) {
    if (h <= 0.0) return 0.0f;
    double square = (double)__fmul_rn(w, w);
    double a = __dmul_rn(__dmul_rn(2.0, g), (double)w);
    double b = __dmul_rn(__dadd_rn(h, (double)lambda), square);
    double l1 = __dmul_rn(__dmul_rn(2.0, (double)alpha), (double)fabsf(w));
    return __double2float_rn(-__dadd_rn(__dadd_rn(a, b), l1));
}
__device__ float mono_node_gain(SplitPair s, double ig, double ih, float lambda,
    float alpha, float mds, float lower, float upper) {
    double g = split_value(s.grad, ig), h = split_value(s.hess, ih);
    return mono_gain_given_weight(g, h,
        mono_apply(mono_weight(g, h, lambda, alpha, mds), lower, upper), lambda, alpha);
}
__device__ SplitPacket mono_candidate(SplitPair l, SplitPair r,
    uint32_t feat, uint32_t bin, bool missing, float parent,
    double ig, double ih, float lambda, float gamma, float mcw,
    float alpha, float mds, int32_t constraint, float lower, float upper) {
    SplitPacket p{};
    double hl = split_value(l.hess, ih), hr = split_value(r.hess, ih);
    if (!(hl > 0.0 && hr > 0.0 && hl >= (double)mcw && hr >= (double)mcw)) return p;
    double gl = split_value(l.grad, ig), gr = split_value(r.grad, ig);
    float wl = mono_apply(mono_weight(gl, hl, lambda, alpha, mds), lower, upper);
    float wr = mono_apply(mono_weight(gr, hr, lambda, alpha, mds), lower, upper);
    if ((constraint > 0 && wl > wr) || (constraint < 0 && wl < wr)) return p;
    float gain = __fadd_rn(mono_gain_given_weight(gl, hl, wl, lambda, alpha),
        mono_gain_given_weight(gr, hr, wr, lambda, alpha));
    float loss = __fsub_rn(gain, parent);
    float result = __double2float_rn(__dsub_rn((double)loss, (double)gamma));
    if (!(result > 1.e-6f)) return p;
    p.feat = feat; p.bin = bin; p.gain = result;
    p.flags = 1u | (missing ? 2u : 0u); p.left = l; p.right = r;
    return p;
}
__device__ bool split_better(SplitPacket a, uint32_t ao, SplitPacket b, uint32_t bo) {
    return (a.flags & 1u) && (!(b.flags & 1u) || a.gain > b.gain ||
        (a.gain == b.gain && ao < bo));
}

// Inclusive two-stage CTA scan of one SplitPair per thread. Integer addition
// is modulo 2^64, so this has exactly the serial prefix. Shuffle operands never
// cross a warp; shared cross-warp data is protected by full CTA barriers before
// publication and consumption. All 256 threads must call it (zero-padded bins).
__device__ __forceinline__ SplitPair split_block_scan(SplitPair v, SplitPair* warp_totals) {
    uint32_t lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    SplitPair scanned = v;
    for (uint32_t step = 1; step < 32; step *= 2) {
        SplitPair prev{__shfl_up_sync(0xffffffffu, scanned.grad, step),
                       __shfl_up_sync(0xffffffffu, scanned.hess, step)};
        if (lane >= step) scanned = split_add(scanned, prev);
    }
    if (lane == 31) warp_totals[warp] = scanned;
    __syncthreads();
    if (warp == 0) {
        SplitPair total = lane < 8 ? warp_totals[lane] : SplitPair{};
        for (uint32_t step = 1; step < 8; step *= 2) {
            SplitPair prev{__shfl_up_sync(0xffffffffu, total.grad, step),
                           __shfl_up_sync(0xffffffffu, total.hess, step)};
            if (lane >= step) total = split_add(total, prev);
        }
        if (lane < 8) warp_totals[lane] = total;
    }
    __syncthreads();
    if (warp) scanned = split_add(warp_totals[warp - 1], scanned);
    return scanned;
}
// One candidate, with the monotone / unconstrained form chosen uniformly.
__device__ __forceinline__ SplitPacket split_any_candidate(SplitPair l, SplitPair r,
    uint32_t feat, uint32_t bin, bool missing, double parent, float mono_parent_gain,
    double ig, double ih, float lambda, float gamma, float mcw,
    float alpha, float mds, uint32_t has_monotone, int32_t constraint,
    float lower, float upper) {
    return has_monotone
        ? mono_candidate(l, r, feat, bin, missing, mono_parent_gain, ig, ih, lambda, gamma,
            mcw, alpha, mds, constraint, lower, upper)
        : split_candidate(l, r, feat, bin, missing, parent, ig, ih, lambda, gamma, mcw);
}

// Per-node metadata: working output slot, small-input slot, previous-cache
// parent slot (UINT32_MAX = direct), next-cache slot (UINT32_MAX = uncached).
// Cache generations are disjoint; no CTA writes an input used by another CTA.
// Monotone: `bounds[n*2..]` = node [lower, upper], `monotone[f]` = direction of
// block-local feature f; both read only when has_monotone != 0 (uniform).
//
// Feature enable: `node_enabled` (nullable) is a per-(node, block-local feature)
// byte mask for nested column sampling; otherwise the tree-level `enabled[f]`.
//
// Categorical (CAT instantiation only): `cat_info[f]` is 0 for an ordered
// feature, else 1 + the feature's ordinal among the block's categorical
// features. A categorical CTA enumerates exactly FerrisBoost's host
// `enumerate_categorical`: one-hot when the bin count < max_onehot, else a
// stable (weight, category) sort and forward / backward prefix scans of
// min(max_threshold, bins) - 1 prefixes. Candidate ordinals follow the host's
// offer order, so the strict-greater / first-wins tie rule is identical. The
// winner's category set is written to `cat_out[(n * n_cat + ordinal) * CAT_SET_WORDS ..]`;
// the winning packet carries the categorical flag (4) and the ordinal in `bin`.
template <bool CAT>
__device__ __forceinline__ void split_features_body(
    SplitPair* hist, const SplitPair* previous, SplitPair* next,
    const uint32_t* offsets, const uint32_t* feats, const uint32_t* enabled,
    const SplitPair* sums, const uint32_t* metadata, uint32_t features,
    uint32_t hist_stride, uint32_t cache_stride,
    double ig, double ih, float lambda, float gamma, float mcw,
    float alpha, float mds,
    const float* bounds, const int32_t* monotone, uint32_t has_monotone,
    SplitPacket* output,
    const uint8_t* node_enabled, const uint32_t* cat_info, uint32_t n_cat,
    uint32_t max_onehot, uint32_t max_threshold, uint32_t* cat_out) {
    __shared__ SplitPair prefix[256];
    __shared__ SplitPair warp_totals[8];
    __shared__ double parent_gain;
    __shared__ float mono_parent;
    // `winners` doubles as scratch for the categorical sort (weights, sorted
    // bins); the scratch is dead before the first winners[] write, which is
    // separated from the last scratch read by the scan's CTA barrier.
    __shared__ SplitPacket winners[256];
    __shared__ uint32_t ordinals[256];
    __shared__ uint16_t cat_order[CAT ? 256 : 1];
    __shared__ uint32_t cat_words[CAT ? CAT_SET_WORDS : 1];
    uint32_t t = threadIdx.x, f = blockIdx.x, n = blockIdx.y;
    uint32_t lo = offsets[f], count = offsets[f + 1] - lo;
    const uint32_t* m = metadata + n * 4;
    SplitPair v{};
    if (t < count) {
        size_t i = (size_t)m[0] * hist_stride + lo + t;
        if (m[2] != 0xffffffffu) {
            v = split_sub(previous[(size_t)m[2] * cache_stride + lo + t],
                hist[(size_t)m[1] * hist_stride + lo + t]);
            hist[i] = v;
        } else v = hist[i];
        if (m[3] != 0xffffffffu)
            next[(size_t)m[3] * cache_stride + lo + t] = v;
    }
    float lower = 0.0f, upper = 0.0f;
    int32_t constraint = 0;
    if (has_monotone) {
        lower = bounds[(size_t)n * 2];
        upper = bounds[(size_t)n * 2 + 1];
        constraint = monotone[f];
    }
    if (!t) {
        parent_gain = split_gain(sums[n], ig, ih, lambda);
        mono_parent = has_monotone
            ? mono_node_gain(sums[n], ig, ih, lambda, alpha, mds, lower, upper) : 0.0f;
    }
    const bool on = (node_enabled ? node_enabled[(size_t)n * features + f] : enabled[f]) != 0;
    const uint32_t kind = CAT ? cat_info[f] : 0u;
    SplitPacket best{}; uint32_t ordinal = 0xffffffffu;
    bool onehot = false;
    uint32_t n_scan = 0;
    if (CAT && kind) {
        // ---- categorical feature (block-uniform branch) ----
        onehot = count < max_onehot;
        n_scan = count < max_threshold ? count : max_threshold;
        const bool active = on && (onehot || n_scan >= 2);
        const uint32_t cat_ord = kind - 1;
        if (active) {
            SplitPair sum = sums[n];
            if (onehot) {
                prefix[t] = split_block_scan(v, warp_totals);
                __syncthreads();
                if (t < count) {
                    SplitPair total = prefix[count - 1];
                    SplitPair missing = split_sub(sum, total);
                    // Category t alone goes right; missing rows go left first,
                    // then with the category.
                    best = split_any_candidate(split_sub(sum, v), v, feats[f], cat_ord, true,
                        parent_gain, mono_parent, ig, ih, lambda, gamma, mcw, alpha, mds,
                        has_monotone, constraint, lower, upper);
                    ordinal = 2 * t;
                    SplitPair right = split_add(v, missing);
                    SplitPacket alt = split_any_candidate(split_sub(sum, right), right,
                        feats[f], cat_ord, false, parent_gain, mono_parent, ig, ih, lambda,
                        gamma, mcw, alpha, mds, has_monotone, constraint, lower, upper);
                    if (split_better(alt, 2 * t + 1, best, ordinal)) { best = alt; ordinal = 2 * t + 1; }
                }
            } else {
                float* wts = (float*)winners;
                // Scratch layout inside `winners`: SPLIT_CTA_THREADS weights, then bins.
                static_assert(SPLIT_CTA_THREADS * (sizeof(float) + sizeof(SplitPair))
                              <= SPLIT_CTA_THREADS * sizeof(SplitPacket), "scratch fits");
                SplitPair* sorted = (SplitPair*)((char*)winners + SPLIT_CTA_THREADS * sizeof(float));
                float w = 0.0f;
                if (t < count) {
                    double g = split_value(v.grad, ig), h = split_value(v.hess, ih);
                    w = mono_weight(g, h, lambda, alpha, mds);
                }
                wts[t] = w;
                __syncthreads();
                if (t < count) {
                    // Stable rank by (weight, category): equal weights (incl.
                    // -0.0 / 0.0) keep ascending category order, like the
                    // host's stable sort with `<`.
                    uint32_t rank = 0;
                    for (uint32_t u = 0; u < count; ++u) {
                        float wu = wts[u];
                        rank += (wu < w || (wu == w && u < t)) ? 1u : 0u;
                    }
                    cat_order[rank] = (uint16_t)t;
                    sorted[rank] = v;
                }
                __syncthreads();
                SplitPair x = t < count ? sorted[t] : SplitPair{};
                prefix[t] = split_block_scan(x, warp_totals);
                __syncthreads();
                if (t + 1 < n_scan) {
                    SplitPair total = prefix[count - 1];
                    // Forward: first t+1 sorted categories right, missing left.
                    SplitPair right = prefix[t];
                    best = split_any_candidate(split_sub(sum, right), right, feats[f], cat_ord,
                        true, parent_gain, mono_parent, ig, ih, lambda, gamma, mcw, alpha, mds,
                        has_monotone, constraint, lower, upper);
                    ordinal = t;
                    // Backward: sorted tail j.. on the left, missing right; the
                    // first j sorted categories are the set.
                    uint32_t j = count - 1 - t;
                    SplitPair left = split_sub(total, prefix[j - 1]);
                    SplitPacket alt = split_any_candidate(left, split_sub(sum, left), feats[f],
                        cat_ord, false, parent_gain, mono_parent, ig, ih, lambda, gamma, mcw,
                        alpha, mds, has_monotone, constraint, lower, upper);
                    uint32_t ro = n_scan - 1 + t;
                    if (split_better(alt, ro, best, ordinal)) { best = alt; ordinal = ro; }
                }
            }
            if (best.flags & 1u) best.flags |= 4u;
        }
    } else {
        SplitPair scanned = split_block_scan(v, warp_totals);
        prefix[t] = scanned;
        __syncthreads();
        if (t < count && on) {
            SplitPair sum = sums[n], total = prefix[count - 1];
            best = has_monotone
                ? mono_candidate(prefix[t], split_sub(sum, prefix[t]), feats[f], t + 1, false,
                    mono_parent, ig, ih, lambda, gamma, mcw, alpha, mds, constraint, lower, upper)
                : split_candidate(prefix[t], split_sub(sum, prefix[t]), feats[f], t + 1,
                    false, parent_gain, ig, ih, lambda, gamma, mcw);
            ordinal = t;
            if (total.grad != sum.grad || total.hess != sum.hess) {
                SplitPair before = t ? prefix[t - 1] : SplitPair{};
                SplitPair right = split_sub(total, before);
                SplitPacket reverse = has_monotone
                    ? mono_candidate(split_sub(sum, right), right, feats[f], t, true,
                        mono_parent, ig, ih, lambda, gamma, mcw, alpha, mds, constraint, lower, upper)
                    : split_candidate(split_sub(sum, right), right, feats[f], t,
                        true, parent_gain, ig, ih, lambda, gamma, mcw);
                uint32_t ro = count + count - 1 - t;
                if (split_better(reverse, ro, best, ordinal)) { best = reverse; ordinal = ro; }
            }
        }
    }
    winners[t] = best; ordinals[t] = ordinal; __syncthreads();
    for (uint32_t step = 128; step; step /= 2) {
        if (t < step && split_better(winners[t + step], ordinals[t + step],
                                    winners[t], ordinals[t])) {
            winners[t] = winners[t + step]; ordinals[t] = ordinals[t + step];
        }
        __syncthreads();
    }
    if (!t) output[(size_t)n * features + f] = winners[0];
    if (CAT && kind && (winners[0].flags & 1u)) {
        if (t < CAT_SET_WORDS) cat_words[t] = 0u;
        __syncthreads();
        uint32_t o = ordinals[0];
        if (onehot) {
            if (!t) atomicOr(&cat_words[(o >> 1) >> 5], 1u << ((o >> 1) & 31u));
        } else {
            // Forward ordinal o holds o+1 sorted categories; backward ordinal
            // n_scan-1+e holds the first (count-1-e).
            uint32_t k = o < n_scan - 1 ? o + 1 : count - 1 - (o - (n_scan - 1));
            if (t < k) atomicOr(&cat_words[cat_order[t] >> 5], 1u << (cat_order[t] & 31u));
        }
        __syncthreads();
        if (t < CAT_SET_WORDS) cat_out[((size_t)n * n_cat + (kind - 1)) * CAT_SET_WORDS + t] = cat_words[t];
    }
}
extern "C" __global__ void split_cached_features(
    SplitPair* hist, const SplitPair* previous, SplitPair* next,
    const uint32_t* offsets, const uint32_t* feats, const uint32_t* enabled,
    const SplitPair* sums, const uint32_t* metadata, uint32_t features,
    uint32_t hist_stride, uint32_t cache_stride,
    double ig, double ih, float lambda, float gamma, float mcw,
    float alpha, float mds,
    const float* bounds, const int32_t* monotone, uint32_t has_monotone,
    SplitPacket* output, const uint8_t* node_enabled) {
    split_features_body<false>(hist, previous, next, offsets, feats, enabled, sums, metadata,
        features, hist_stride, cache_stride, ig, ih, lambda, gamma, mcw, alpha, mds, bounds,
        monotone, has_monotone, output, node_enabled, nullptr, 0, 0, 0, nullptr);
}
extern "C" __global__ void split_cached_features_cat(
    SplitPair* hist, const SplitPair* previous, SplitPair* next,
    const uint32_t* offsets, const uint32_t* feats, const uint32_t* enabled,
    const SplitPair* sums, const uint32_t* metadata, uint32_t features,
    uint32_t hist_stride, uint32_t cache_stride,
    double ig, double ih, float lambda, float gamma, float mcw,
    float alpha, float mds,
    const float* bounds, const int32_t* monotone, uint32_t has_monotone,
    SplitPacket* output, const uint8_t* node_enabled,
    const uint32_t* cat_info, uint32_t n_cat, uint32_t max_onehot, uint32_t max_threshold,
    uint32_t* cat_out) {
    split_features_body<true>(hist, previous, next, offsets, feats, enabled, sums, metadata,
        features, hist_stride, cache_stride, ig, ih, lambda, gamma, mcw, alpha, mds, bounds,
        monotone, has_monotone, output, node_enabled, cat_info, n_cat, max_onehot,
        max_threshold, cat_out);
}
extern "C" __global__ void split_cached_nodes(const SplitPacket* features,
    uint32_t n_features, double ig, double ih, float lambda, SplitPacket* output) {
    if (threadIdx.x) return;
    SplitPacket best{};
    for (uint32_t f = 0; f < n_features; ++f) {
        SplitPacket p = features[(size_t)blockIdx.x * n_features + f];
        if ((p.flags & 1u) && (!(best.flags & 1u) || p.gain > best.gain)) best = p;
    }
    // The training consumer uses only winner metadata and fixed-point child
    // sums. Leaf weights are computed once after the global block reduction;
    // avoid repeating unused divisions for every block-local candidate.
    output[blockIdx.x] = best;
}
// Node reduction for blocks with categorical features. Packets are `stride`
// bytes apart: a 64 B packet, then the 256-bit category set of a categorical
// winner (zero otherwise). The packet's `bin` (the feature's categorical
// ordinal) is cleared once the set has been located.
extern "C" __global__ void split_cached_nodes_cat(const SplitPacket* features,
    uint32_t n_features, const uint32_t* cat_sets, uint32_t n_cat,
    uint8_t* output, uint32_t stride) {
    if (threadIdx.x) return;
    SplitPacket best{};
    for (uint32_t f = 0; f < n_features; ++f) {
        SplitPacket p = features[(size_t)blockIdx.x * n_features + f];
        if ((p.flags & 1u) && (!(best.flags & 1u) || p.gain > best.gain)) best = p;
    }
    uint8_t* out = output + (size_t)blockIdx.x * stride;
    uint32_t* words = (uint32_t*)(out + sizeof(SplitPacket));
    if (best.flags & 4u) {
        const uint32_t* src = cat_sets + ((size_t)blockIdx.x * n_cat + best.bin) * CAT_SET_WORDS;
        for (uint32_t w = 0; w < CAT_SET_WORDS; ++w) words[w] = src[w];
        best.bin = 0;
    } else {
        for (uint32_t w = 0; w < CAT_SET_WORDS; ++w) words[w] = 0u;
    }
    *(SplitPacket*)out = best;
}

// R=4 multi-row/thread 的每线程行数。见 hist_build_shared_planar32_fused_r4。
#define ROWS_PER_THREAD 4

// ---------------------------------------------------------------------------
// 定点量化搬到 device
//
// **为什么能保持逐位相同**(这两个 kernel 唯一需要小心的地方):
//
// 1. `GradQuantizer::new` 求的是 `max |grad as f64|` / `max |hess as f64|`。
//    f32→f64 是**精确**加宽,`fabs` 精确,`max` 结合律交换律成立而且精确 ——
//    所以「块内归约 + host 收尾」与串行逐个扫**逐位相同**,与顺序无关。
// 2. `quantize` 是 `(v as f64 * scale).round() as i64`,而 `scale` **恒为
//    2 的幂**(`pow2_scale`)。乘 2 的幂在 f64 里只动指数、**不产生舍入**;
//    CUDA 的 `round(double)` 和 Rust 的 `f64::round` 都是 ties-away-from-zero
//    (已实测 ±0.5/±1.5/±2.5/±3.5 八个 tie 全部一致)。
//
// ⚠️ **梯度本身仍然留在 CPU。** `gradient_of` 含 exp/sigmoid,而 CUDA 的
// `expf` 和 host libm 不保证最后一位相同 —— 搬上来会直接破坏
// 「GPU/CPU 模型逐字节相同」这条核心契约。这里只搬**纯算术**的那两步。
// ---------------------------------------------------------------------------

// 每个 block 归约出自己那段的 max|grad| / max|hess|,写进 out_max[2*blockIdx]。
// host 再对这 n_blocks 对做最终 max —— 很小,而且同样是精确的。
// ---- GPU-local row-major 原型 ----------------------------------------------
//
// **前提**:列主序下 `bins[feat * n_rows + row]` 在 root 上是连续的,但
// partition 之后 `row` 是**散列的子集**,于是一个 warp 的 32 条 lane 各取
// 1 个字节、落在最多 32 个不同 sector 上。实测(ncu,wide 10M×300):
//
//   depth 0  sector 效率 62.2%   82.0 M sectors   21.7 ms
//   depth 2  sector 效率 22.3%  102.6 M sectors   11.6 ms
//   depth 5  sector 效率 **7.1%**  332.9 M sectors  37.6 ms
//
// 也就是深层每取 32 B 只用掉约 2.3 B。行主序把同一行的所有特征放连续,
// **一个线程一次读满一个 sector**,于是效率与行的散列程度无关。
//
// ⚠️ 旧的「flattened (row × feature) 更慢」结论**只对列主序成立**
// (当时 feature lane 相隔 n_rows 字节),这里前提变了,所以可以重开。
extern "C" __global__ void transpose_bins_to_row_major(
    const uint8_t* __restrict__ src,
    uint8_t* __restrict__ dst,
    uint32_t n_rows,
    uint32_t n_feats) {
    // +1 padding 避开 shared bank 冲突。
    __shared__ uint8_t tile[32][33];
    const uint32_t r0 = blockIdx.x * 32u;
    const uint32_t f0 = blockIdx.y * 32u;

    const uint32_t r_in = r0 + threadIdx.x;
    const uint32_t f_in = f0 + threadIdx.y;
    if (r_in < n_rows && f_in < n_feats) {
        tile[threadIdx.y][threadIdx.x] = src[(uint64_t)f_in * n_rows + r_in];
    }
    __syncthreads();
    // 写出时让 threadIdx.x 走 feature 维,保证写也是连续的。
    const uint32_t f_out = f0 + threadIdx.x;
    const uint32_t r_out = r0 + threadIdx.y;
    if (r_out < n_rows && f_out < n_feats) {
        dst[(uint64_t)r_out * n_feats + f_out] = tile[threadIdx.x][threadIdx.y];
    }
}


// 行主序 bins 上的多节点 batched histogram。
//
// 与 `hist_build_shared_planar32_batched` **逐条对应**:同样只批
// `Accumulate` 节点、同样一个 block 管一个节点的一段行、同样的 tile 切分、
// 同样的 Planar32 shared 布局、同样的 `AtomicAdd64As32`、同样的 flush。
// **shared 大小和 tile 几何一个字节都没变。**
//
// 唯一的差别:每个线程**先把自己那一行的全部特征字节一次性读进寄存器**
// (行主序下是连续的,最多 32 B = 8 个 uint32),之后所有 tile 都从寄存器里取。
// 于是 global 读取次数从「每 (行, 特征) 一次」变成「每行一次」,
// 且每次都是满 sector。
//
// ⚠️ 原型限制:`n_feats <= 32`(HIGGS 28 / wide cols_per_block=32 都满足)。
// 超过就回落到列主序路径 —— host 侧负责这个判断。
//
// 位精确:读的是同一批字节、加的是同一批定点数、顺序无关,
// 因此与列主序 kernel 逐位相同。
#define RM_MAX_FEAT_WORDS 8
extern "C" __global__ void hist_build_shared_planar32_batched_rm(
    const uint8_t* bins_rm,
    const int64_t* gpair,
    const uint32_t* row_idx,
    uint32_t n_rows,
    uint32_t n_feats,
    const uint32_t* feat_offsets,
    int64_t* out_hist,
    uint32_t max_tile_bins,
    const uint32_t* batch_idx,
    uint32_t n_nodes,
    uint32_t hist_stride,
    // `colsample_bytree`:本列块里被选中的**局部**特征位图。
    // 行主序一块最多 32 个特征,所以一个 u32 正好装下 —— 不需要额外
    // buffer,也不改 launch 几何。全 1 = 不采样。
    //
    // ⚠️ 这里省的是**计算**(shared 原子加),不是 H2D:块仍然完整常驻,
    // 整行的 bins 也仍然一次读进寄存器。resident 下**不应该**为采样每棵树
    // 重新 compact / 上传 —— 那会把行主序常驻的优势直接抵消掉。
    uint32_t feat_mask) {
    // One tracked allocation: [blk_ptr(n+1) | node_off(n) | node_len(n)].
    const uint32_t* blk_ptr = batch_idx;
    const uint32_t* node_off = batch_idx + n_nodes + 1;
    const uint32_t* node_len = batch_idx + 2 * n_nodes + 1;
    extern __shared__ unsigned int fused_hist_rm[];
    __shared__ uint32_t s_node_rm;

    if (threadIdx.x == 0) {
        const uint32_t bid = blockIdx.x;
        uint32_t lo = 0, hi = n_nodes;
        while (lo + 1 < hi) {
            const uint32_t mid = (lo + hi) >> 1;
            if (blk_ptr[mid] <= bid) { lo = mid; } else { hi = mid; }
        }
        s_node_rm = lo;
    }
    __syncthreads();

    const uint32_t k = s_node_rm;
    const uint32_t local = blockIdx.x - blk_ptr[k];
    // 两条 lane 一行,所以一个 block 覆盖 blockDim.x / 2 行。
    // host 侧的 rows_per_block 必须和这里一致,否则批内节点边界会错位。
    const uint32_t start = node_off[k] + local * (blockDim.x * ROWS_PER_THREAD);
    const uint32_t end = node_off[k] + node_len[k];
    int64_t* out = out_hist + (uint64_t)k * hist_stride;

    // 每线程一行。行主序下这一行的 n_feats 个字节是连续的,按 uint32
    // 向量读进寄存器 —— 一个线程一次吃满一个 sector。
    // **两条 lane 负责一行,每条一次 uint4 取 16 B。**
    //
    // ⚠️ 上一版是一条 lane 读整行 32 B(两次 uint4),实测 depth 0 的
    // sector 数**翻倍**(82.0 M → 166.9 M):行主序里相邻行相隔 32 B,
    // 而一次 uint4 只覆盖 16 B,于是同一个 32 B sector 被取两次,
    // L1 没能把第二次接住。
    //
    // 改成半行/lane 之后:depth 0 上 32 条 lane 覆盖 16 个连续行 ×32 B =
    // 512 B 连续;深层散列时 lane 2i / 2i+1 读同一行的两半、共享一个
    // sector。两种情况都是满 sector。
    // 一条 lane 一行,整行 32 B 用两次 uint4 读进寄存器。
    //
    // ⚠️ 试过「两条 lane 一行、各一次 uint4」来修 depth 0 的 sector 翻倍,
    // **实测 wide 反而慢 22%**:每 block 只覆盖 blockDim.x/2 = 256 行,
    // 而列主序 R=4 覆盖 2048 行 —— shared 的 init/flush 因此多做 8 倍,
    // 远超省下来的那点取数。教训是 coalescing 和 init/flush 摊薄在这里
    // 是**互相拉扯**的,不能只优化一边。
    // **每线程 4 行,和列主序 R=4 完全一致** —— 于是每 block 覆盖的行数、
    // shared 的 init/flush 次数、barrier 次数都和参照臂一模一样,
    // 两条路径**只剩布局和取数方式的差别**。
    //
    // R=1 那一版每 block 只覆盖 blockDim.x 行,init/flush 要多做 4 倍;
    // 实测深层快 2.27× 却被浅层和这笔固定开销吃回去,整轮只有 −1.8%。
    unsigned int g_lo[ROWS_PER_THREAD], g_hi[ROWS_PER_THREAD];
    unsigned int h_lo[ROWS_PER_THREAD], h_hi[ROWS_PER_THREAD];
    uint32_t feat_words[ROWS_PER_THREAD][RM_MAX_FEAT_WORDS];
    bool ok[ROWS_PER_THREAD];
#pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; ++r) {
        const uint32_t sel = start + threadIdx.x + (uint32_t)r * blockDim.x;
        ok[r] = sel < end;
        g_lo[r] = 0; g_hi[r] = 0; h_lo[r] = 0; h_hi[r] = 0;
#pragma unroll
        for (int w = 0; w < RM_MAX_FEAT_WORDS; ++w) { feat_words[r][w] = 0xFFFFFFFFu; }
        if (ok[r]) {
            uint32_t row = row_idx == nullptr ? sel : row_idx[sel];
            const uint32_t orig_row = row;
            const unsigned long long gr =
                static_cast<unsigned long long>(gpair[(uint64_t)orig_row * 2]);
            const unsigned long long hr =
                static_cast<unsigned long long>(gpair[(uint64_t)orig_row * 2 + 1]);
            g_lo[r] = static_cast<unsigned int>(gr);
            g_hi[r] = static_cast<unsigned int>(gr >> 32);
            h_lo[r] = static_cast<unsigned int>(hr);
            h_hi[r] = static_cast<unsigned int>(hr >> 32);
            // `row` 此时是压实下标(未压实时就是原始行号)。
            const uint8_t* rowp = bins_rm + (uint64_t)row * n_feats;
            // 取数分三档,按对齐能力**逐级降级**。
            //
            // ⚠️ 只有 uint4 一条快路的话,`n_feats % 16 != 0` 会整个掉进
            // 逐字节循环 —— HIGGS 是 28 列,于是每行 28 次单字节取数,
            // **实测整轮回退 32~45%**。宽表(32 列)看不到这个坑,
            // 因为它正好整除。
            //
            // 对齐前提:行首偏移是 `row * n_feats`,所以
            // `n_feats % 16 == 0` 才能保证 16 B 对齐,`n_feats % 4 == 0`
            // 保证 4 B 对齐。不满足就只能按字节。
            if ((n_feats & 15u) == 0u && (((uintptr_t)rowp & 15u) == 0u)) {
                const uint4* v4 = reinterpret_cast<const uint4*>(rowp);
                const uint32_t n4 = n_feats >> 4;
#pragma unroll
                for (uint32_t q = 0; q < (RM_MAX_FEAT_WORDS >> 2); ++q) {
                    if (q < n4) {
                        const uint4 v = v4[q];
                        feat_words[r][q * 4 + 0] = v.x; feat_words[r][q * 4 + 1] = v.y;
                        feat_words[r][q * 4 + 2] = v.z; feat_words[r][q * 4 + 3] = v.w;
                    }
                }
            } else if ((n_feats & 3u) == 0u && (((uintptr_t)rowp & 3u) == 0u)) {
                const uint32_t* v1 = reinterpret_cast<const uint32_t*>(rowp);
                const uint32_t nw = n_feats >> 2;
#pragma unroll
                for (uint32_t q = 0; q < RM_MAX_FEAT_WORDS; ++q) {
                    if (q < nw) { feat_words[r][q] = v1[q]; }
                }
            } else {
                for (uint32_t j = 0; j < n_feats; ++j) {
                    feat_words[r][j >> 2] =
                        (feat_words[r][j >> 2] & ~(0xFFu << ((j & 3u) * 8u)))
                        | ((uint32_t)rowp[j] << ((j & 3u) * 8u));
                }
            }
        }
    }

    uint32_t f0 = 0;
    while (f0 < n_feats) {
        const uint32_t hist_begin = feat_offsets[f0];
        uint32_t f1 = f0;
        while (f1 < n_feats && feat_offsets[f1 + 1] - hist_begin <= max_tile_bins) { ++f1; }
        const uint32_t n_slots = feat_offsets[f1] - hist_begin;

        // 整个 tile 都没有选中的特征就**彻底跳过** —— 连 shared 的清零和
        // flush 都不做。只屏蔽原子加是不够的:flush 仍会把整块零写回
        // global,而 `colsample` 越小,这部分零就越多。
        //
        // `feat_mask`、`f0`、`f1` 在整个 block 内一致,所以这个分支是
        // block-uniform 的,跳过 `__syncthreads()` 是安全的。
        // 跳过的 tile 在 out_hist 里保持调用方 memset 出来的 0,而枚举侧
        // 本来就不看这些特征。
        const uint32_t tile_bits = (f1 - f0 >= 32u)
            ? 0xFFFFFFFFu
            : (((1u << (f1 - f0)) - 1u) << f0);
        if ((feat_mask & tile_bits) == 0u) {
            f0 = f1;
            continue;
        }

        unsigned int* grad_lo = fused_hist_rm;
        unsigned int* grad_hi = fused_hist_rm + n_slots;
        unsigned int* hess_lo = fused_hist_rm + (uint64_t)n_slots * 2;
        unsigned int* hess_hi = fused_hist_rm + (uint64_t)n_slots * 3;

        for (uint32_t word = threadIdx.x; word < n_slots * 4; word += blockDim.x) {
            fused_hist_rm[word] = 0;
        }
        __syncthreads();

        for (uint32_t feat = f0; feat < f1; ++feat) {
            if (((feat_mask >> feat) & 1u) == 0u) { continue; }
            const uint32_t off = feat_offsets[feat] - hist_begin;
#pragma unroll
            for (int r = 0; r < ROWS_PER_THREAD; ++r) {
                if (!ok[r]) { continue; }
                const uint32_t bin =
                    (feat_words[r][feat >> 2] >> ((feat & 3u) * 8u)) & 0xFFu;
                if (bin == 255u) { continue; }
                const uint32_t s = off + bin;
                const unsigned int go = atomicAdd(&grad_lo[s], g_lo[r]);
                atomicAdd(&grad_hi[s], g_hi[r] + ((go > (0xFFFFFFFFu - g_lo[r])) ? 1u : 0u));
                const unsigned int ho = atomicAdd(&hess_lo[s], h_lo[r]);
                atomicAdd(&hess_hi[s], h_hi[r] + ((ho > (0xFFFFFFFFu - h_lo[r])) ? 1u : 0u));
            }
        }
        __syncthreads();

        const uint32_t out_words = n_slots * 2;
        for (uint32_t word = threadIdx.x; word < out_words; word += blockDim.x) {
            const uint32_t s = word >> 1;
            const bool is_hess = (word & 1) != 0;
            const unsigned int lo = is_hess ? hess_lo[s] : grad_lo[s];
            const unsigned int hi = is_hess ? hess_hi[s] : grad_hi[s];
            const unsigned long long v = ((unsigned long long)hi << 32) | (unsigned long long)lo;
            atomicAdd(reinterpret_cast<unsigned long long*>(out) + (uint64_t)hist_begin * 2 + word, v);
        }
        __syncthreads();

        f0 = f1;
    }
}


// XGBoost 式 **flattened (row × feature)** 映射,跑在行主序 bins 上。
//
// **和 R=4 行主序的区别只有一个:warp 内先走 feature 维,不是 row 维。**
//
// R=4 行主序里一条 lane 管一整行,于是一个 warp 同时追 **32 个离散 row**
// —— 它修好了「每个 sector 只用 1 B」,却没修「一个 warp 摸 32 个地址」。
// 实测:wide(32 列,整除 16)−15%,但 HIGGS(28 列)**回退 32~45%**。
//
// flattened 之后 `linear = row_slot * gf + feature`,连续 lane 先变 feature:
// `gf = 8` 时一个 warp 覆盖 **4 行 × 8 特征**,每行 8 个连续字节 ——
// 一个 warp 只摸 4 个地址,而不是 32 个。离散 row 只决定这一小段连续
// 读取的基址。
//
// feature group 放到 **blockIdx.y**,不再在 block 内顺序扫 tile ——
// 深层 row 数下降时,feature 维仍然提供并行度。
//
// 位精确:仍是同一批定点数按模 2^64 相加,顺序无关。
#define FLAT_ROWS_PER_BLOCK 512u
// `gpu_math = "fast"` 专用:prediction apply + objective gradient 全部留在
// device 上,于是每轮消掉 prediction D2H(4 B/行)、host 的按行 gradient
// 扫描,以及 gpair H2D(8 B/行)。
//
// ⚠️ **它故意不保证 CPU/GPU 逐字节相同。** CUDA 的 `expf` 与 glibc 的
// `expf` 在 1019 万个样本上有 7.32% 不逐位相同(最大 4 ULP),经
// `s - label` / `s(1-s)` 的抵消放大后约 7.1% 的行会拿到不同的 grad/hess。
// 这正是 `exact` 模式把这段留在 host 的原因;`fast` 是显式放弃这条契约,
// 用来把「契约的价钱」和「实现的差距」分开定价。
//
// 用精确的 `expf` 而不是 `__expf` 快速内在函数:后者误差大得多,而这里
// 要的是「不保证逐位相同」,不是「放弃数值质量」。
//
// 每一步都逐字对应 host 的 `gradient_of`,包括 hess 的 1e-16 下限 ——
// 两边算的必须是同一个函数,否则求 scale 用的上界和真正累加的不一致,
// 定点会溢出。
// SplitMix64 finalizer。和 host 的 `subsample::mix` **必须逐位相同**。
__device__ __forceinline__ unsigned long long fb_mix(unsigned long long z) {
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// Per-block max-abs reduction for the fused fast gradient kernels. Every thread
// of the block must call it (no early return before it). The non-finite
// mapping to +inf is done by the caller, so no NaN can reach fmax. max is
// order independent, so this is bit-identical to the old separate
// `gpair_max_abs` pass over the written f32 gpair.
__device__ __forceinline__ void fb_block_max_pair(double mg, double mh, double* out_max) {
    __shared__ double wg[32];
    __shared__ double wh[32];
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        mg = fmax(mg, __shfl_down_sync(0xffffffffu, mg, o));
        mh = fmax(mh, __shfl_down_sync(0xffffffffu, mh, o));
    }
    const uint32_t lane = threadIdx.x & 31u;
    const uint32_t warp = threadIdx.x >> 5;
    if (lane == 0) {
        wg[warp] = mg;
        wh[warp] = mh;
    }
    __syncthreads();
    if (warp == 0) {
        const uint32_t nw = (blockDim.x + 31u) >> 5;
        mg = lane < nw ? wg[lane] : 0.0;
        mh = lane < nw ? wh[lane] : 0.0;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            mg = fmax(mg, __shfl_down_sync(0xffffffffu, mg, o));
            mh = fmax(mh, __shfl_down_sync(0xffffffffu, mh, o));
        }
        if (lane == 0) {
            out_max[(uint64_t)blockIdx.x * 2] = mg;
            out_max[(uint64_t)blockIdx.x * 2 + 1] = mh;
        }
    }
}

__device__ __forceinline__ double fb_bound(float v) {
    const double d = (double)v;
    return isfinite(d) ? fabs(d) : INFINITY;
}

extern "C" __global__ void apply_delta_and_gradient(
    float* pred,
    const float* delta,
    const float* label,
    // 样本权重(XGBoost:grad、hess 同乘 w,hess 下限在乘之前)。
    // has_weight == 0 时 weight 是占位指针,不读。
    const float* weight,
    uint32_t has_weight,
    uint32_t n_rows,
    uint32_t objective,
    // poisson: max_delta_step; tweedie: variance power; pseudohuber: slope.
    float obj_param,
    float* gpair_out,
    // 行采样:`row_hash >> 11 < threshold` 即选中。阈值由 host 折算好,
    // **device 上不做任何浮点换算** —— 两边比的是同一个整数,
    // 所以不可能因为浮点差异选出不同的行。`~0ull` = 全选。
    unsigned long long subsample_threshold,
    unsigned long long subsample_seed,
    unsigned long long subsample_tree,
    // 每个 block 一对 (max|g|, max|h|),host 在小 D2H 之后归约。
    double* max_out) {
    double mg = 0.0;
    double mh = 0.0;
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n_rows;
         i += (uint32_t)blockDim.x * gridDim.x) {
        // ⚠️⚠️ **预测更新永远无条件执行。**
        // 采样只决定这一行是否**塑造**这棵树,不决定它是否**收到**这棵树。
        // 把下面这两行放进采样分支里,就等于把采样掩码变成了预测掩码,
        // 未选中的行会漏掉这棵树的预测增量 —— 那是语义错误,不是优化。
        const float p = pred[i] + delta[i];
        pred[i] = p;

        const float y = label[i];
        float g, h;
        // Objective codes match `Objective::device_code` (src/objective.rs).
        // Formulas follow XGBoost 3.4.1 regression_obj.cu; CUDA expf is not
        // bit-identical to host libm, which is why this path is fast-only.
        if (objective == 0u) {
            // SquaredError
            g = p - y;
            h = 1.0f;
        } else if (objective == 1u) {
            // Logistic
            const float s = 1.0f / (1.0f + expf(-p));
            g = s - y;
            h = fmaxf(s * (1.0f - s), 1e-16f);
        } else if (objective == 2u) {
            // count:poisson
            g = expf(p) - y;
            h = expf(p + obj_param);
        } else if (objective == 3u) {
            // reg:gamma
            const float e = expf(p);
            g = 1.0f - y / e;
            h = y / e;
        } else if (objective == 4u) {
            // reg:tweedie
            const float rho = obj_param;
            const float a = expf((1.0f - rho) * p);
            const float b = expf((2.0f - rho) * p);
            g = -y * a + b;
            h = -y * (1.0f - rho) * a + (2.0f - rho) * b;
        } else {
            // reg:pseudohubererror
            const float z = p - y;
            const float slope2 = obj_param * obj_param;
            const float scale_sqrt = sqrtf(1.0f + z * z / slope2);
            const float scale = slope2 + z * z;
            g = z / scale_sqrt;
            h = slope2 / (scale * scale_sqrt);
        }
        if (has_weight) {
            const float w = weight[i];
            g = __fmul_rn(g, w);
            h = __fmul_rn(h, w);
        }
        // 未选中的行:**只把工作 gpair 置零**(上面的 prediction 已经更新过了)。
        if (subsample_threshold != ~0ULL) {
            const unsigned long long hsh = fb_mix(
                subsample_seed
                ^ ((subsample_tree * 0x9E3779B97F4A7C15ULL) << 23
                   | (subsample_tree * 0x9E3779B97F4A7C15ULL) >> 41)
                ^ ((unsigned long long)i * 0xD6E8FEB86659FD93ULL));
            if ((hsh >> 11) >= subsample_threshold) {
                g = 0.0f;
                h = 0.0f;
            }
        }
        gpair_out[(uint64_t)i * 2] = g;
        gpair_out[(uint64_t)i * 2 + 1] = h;
        // 采样置零之后的值进 max(和原来单独的 gpair_max_abs 看到的一样)。
        mg = fmax(mg, fb_bound(g));
        mh = fmax(mh, fb_bound(h));
    }
    fb_block_max_pair(mg, mh, max_out);
}


// ---------------------------------------------------------------- 多分类 fast
//
// `gpu_math = "fast"` 的 multi:softprob / multi:softmax。prediction 是 class-major
// 的 `K × n`(`pred[k * n + i]`),所以每个类的 prediction 列、delta 施加都是连续访问。
//
// 一轮的流程(host 在 `train_multiclass_rounds` 里驱动):
//   1. `softmax_row_stats`  —— 读 K 列旧 prediction,每行只算一次 wmax 和 (float)wsum;
//   2. 对每个类 k:`softmax_class_gradient` 只读 pred 的第 k 列 + 上面两个统计量,
//      写出 interleaved [grad, hess] 的 f32 gpair(并按行采样置零),走通用的
//      max → scale → 定点量化尾巴,建第 k 棵树;
//   3. 这棵树的叶子增量加到第 k 列 —— 不再单独 launch:它并进「下一个」kernel
//      (第 k+1 类的 `softmax_class_gradient`,或下一轮的 `softmax_row_stats`),
//      在那里顺手 `pred[k] += delta`。delta buffer 在下一棵树建完之前不会被覆盖,
//      而被并进去的那列不是该 kernel 读来算梯度的那列,所以数值和单独 launch 逐位相同。
//      `add_class_delta` 只剩训练结束 flush 最后一个待施加 delta 时使用。
// 第 k 类的梯度只依赖「这一轮开始时」的 K 个 margin。第 k 列在第 k 棵树之前不会变,
// 其余列 wmax / wsum 已经固化在统计量里,所以第 k 棵树之后立刻施加 delta,
// 不会改变同一轮里后面各类的梯度 —— 和 XGBoost「一轮一次 GetGradient」一致。
//
// 公式逐步对应 `Objective::softmax_gradients`(XGBoost `SoftmaxMultiClassObj`):
// wmax 从 FLT_MIN(最小正规格化数)起取 fmaxf,wsum 在 double 里按类序累加,
// p = expf(m - wmax) / (float)wsum,h = max(2p(1-p)w, 1e-16),g = (p - [y==k]) w。
// 不保证与 host 逐位相同(CUDA expf),但每步算术用显式 _rn,不允许编译器 FMA 收缩;
// 没有浮点原子操作,每行只有一个写者,所以结果逐次运行确定。
extern "C" __global__ void softmax_row_stats(
    float* pred,
    const float* delta,
    // 上一轮最后一棵树的 delta 还没施加:在这里并进来(`~0u` = 没有待施加的)。
    uint32_t pending_class,
    uint32_t n_rows,
    uint32_t n_classes,
    float* wmax_out,
    float* wsum_out) {
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n_rows;
         i += (uint32_t)blockDim.x * gridDim.x) {
        if (pending_class < n_classes) {
            float* slot = pred + (uint64_t)pending_class * n_rows + i;
            *slot = __fadd_rn(*slot, delta[i]);
        }
        float wmax = 1.17549435e-38f;  // FLT_MIN
        for (uint32_t k = 0; k < n_classes; ++k) {
            wmax = fmaxf(pred[(uint64_t)k * n_rows + i], wmax);
        }
        double wsum = 0.0;
        for (uint32_t k = 0; k < n_classes; ++k) {
            wsum = __dadd_rn(wsum, (double)expf(__fsub_rn(pred[(uint64_t)k * n_rows + i], wmax)));
        }
        wmax_out[i] = wmax;
        wsum_out[i] = __double2float_rn(wsum);
    }
}

extern "C" __global__ void softmax_class_gradient(
    float* pred,
    const float* delta,
    // 前一棵树(第 class_id-1 类)的 delta 在这里顺手加到它自己的那一列:
    // 本 kernel 只读第 class_id 列,所以不会改变自己的梯度。`~0u` = 没有。
    // host 保证 pending_class != class_id。
    uint32_t pending_class,
    const float* wmax,
    const float* wsum,
    const float* label,
    const float* weight,
    uint32_t has_weight,
    uint32_t n_rows,
    uint32_t class_id,
    float* gpair_out,
    unsigned long long subsample_threshold,
    unsigned long long subsample_seed,
    unsigned long long subsample_tree,
    double* max_out) {
    double mg = 0.0;
    double mh = 0.0;
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n_rows;
         i += (uint32_t)blockDim.x * gridDim.x) {
        const float m = pred[(uint64_t)class_id * n_rows + i];
        if (pending_class != ~0u) {
            float* slot = pred + (uint64_t)pending_class * n_rows + i;
            *slot = __fadd_rn(*slot, delta[i]);
        }
        float p = __fdiv_rn(expf(__fsub_rn(m, wmax[i])), wsum[i]);
        const float w = has_weight ? weight[i] : 1.0f;
        float h = fmaxf(
            __fmul_rn(__fmul_rn(__fmul_rn(2.0f, p), __fsub_rn(1.0f, p)), w), 1e-16f);
        if (label[i] == (float)class_id) { p = __fsub_rn(p, 1.0f); }
        float g = __fmul_rn(p, w);
        // 与 `apply_delta_and_gradient` 相同的整数行采样:未选中行只把工作
        // gpair 置零。prediction 的更新在别的 kernel 里,无条件执行。
        if (subsample_threshold != ~0ULL) {
            const unsigned long long hsh = fb_mix(
                subsample_seed
                ^ ((subsample_tree * 0x9E3779B97F4A7C15ULL) << 23
                   | (subsample_tree * 0x9E3779B97F4A7C15ULL) >> 41)
                ^ ((unsigned long long)i * 0xD6E8FEB86659FD93ULL));
            if ((hsh >> 11) >= subsample_threshold) {
                g = 0.0f;
                h = 0.0f;
            }
        }
        gpair_out[(uint64_t)i * 2] = g;
        gpair_out[(uint64_t)i * 2 + 1] = h;
        mg = fmax(mg, fb_bound(g));
        mh = fmax(mh, fb_bound(h));
    }
    fb_block_max_pair(mg, mh, max_out);
}

// 只在需要 flush 待施加 delta 时用(训练结束读 prediction 之前)。训练循环里
// delta 已经并进下一个 stats / gradient kernel,不再单独 launch。
extern "C" __global__ void add_class_delta(
    float* pred,
    const float* delta,
    uint32_t n_rows,
    uint32_t class_id) {
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n_rows;
         i += (uint32_t)blockDim.x * gridDim.x) {
        float* slot = pred + (uint64_t)class_id * n_rows + i;
        *slot = __fadd_rn(*slot, delta[i]);
    }
}

// ---------------------------------------------------------------- quantile fast
//
// `gpu_math = "fast"` 的 reg:quantileerror。梯度依赖每轮的全局 scale
// `S = (Σ w·sqrt|p-y| / Σ w)^2`(XGBoost 3.4.1 quantile_obj.cu),所以一轮是三个 kernel:
//   1. `quantile_apply_delta_partials` —— 施加叶子增量(delta 为空则只读),同时对每个 block 归约
//      `Σ w·sqrtf|p-y|` 与 `Σ w`(f64),写进 partials[2*block + {0,1}];
//   2. `quantile_reduce_scale` —— 单 block 按固定顺序归约 partials,写出 S;
//   3. `quantile_gradient_rows` —— 读 S,逐行出 grad/hess(含样本权重和行采样)。
//
// 归约形状是**固定**的:grid = QUANTILE_REDUCE_BLOCKS、block = QUANTILE_REDUCE_THREADS,
// 每个线程按 grid-stride 访问固定的行并在 f64 里顺序累加,block 内用固定步长的
// 共享内存树归约,block 间由单 block kernel 按固定的线程分工/树形顺序归约。
// 没有任何浮点原子操作,也不依赖 block 的调度顺序,所以逐次运行逐位相同。
// 不保证与 host 的串行 f64 求和逐位相同(求和顺序不同,CUDA tanhf 也不同于 libm)。
#define QUANTILE_REDUCE_BLOCKS 256u
#define QUANTILE_REDUCE_THREADS 256u

extern "C" __global__ void quantile_apply_delta_partials(
    float* pred,
    const float* delta,
    const float* label,
    const float* weight,
    uint32_t has_weight,
    uint32_t n_rows,
    double* scratch) {
    // scratch = [per-block (Σ w·sqrt|p-y|, Σ w) × QUANTILE_REDUCE_BLOCKS][scale]
    double* partials = scratch;
    __shared__ double rs[QUANTILE_REDUCE_THREADS];
    __shared__ double rw[QUANTILE_REDUCE_THREADS];
    double stat = 0.0;
    double sum_w = 0.0;
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n_rows;
         i += (uint32_t)blockDim.x * gridDim.x) {
        // 预测更新无条件执行(采样掩码不是预测掩码)。delta 为空指针 = 这一列的
        // 叶子增量已经另行施加(多分位数:每个目标一列,`add_class_delta` 在每棵树之后
        // 立即施加,因为 delta buffer 会被下一棵树覆盖),这里只读。
        float p = pred[i];
        if (delta != nullptr) {
            p = __fadd_rn(p, delta[i]);
            pred[i] = p;
        }
        const float r = __fsqrt_rn(fabsf(__fsub_rn(p, label[i])));
        if (has_weight) {
            const float w = weight[i];
            stat = __dadd_rn(stat, (double)__fmul_rn(w, r));
            sum_w = __dadd_rn(sum_w, (double)w);
        } else {
            stat = __dadd_rn(stat, (double)r);
        }
    }
    rs[threadIdx.x] = stat;
    rw[threadIdx.x] = sum_w;
    __syncthreads();
    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            rs[threadIdx.x] = __dadd_rn(rs[threadIdx.x], rs[threadIdx.x + s]);
            rw[threadIdx.x] = __dadd_rn(rw[threadIdx.x], rw[threadIdx.x + s]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        partials[(uint64_t)blockIdx.x * 2] = rs[0];
        partials[(uint64_t)blockIdx.x * 2 + 1] = rw[0];
    }
}

// 单 block(blockDim.x == QUANTILE_REDUCE_THREADS):线程 t 按 b = t, t+T, ... 的固定顺序
// 累加 partials,再做固定树归约。S(已舍入成 f32 的值,存成 double)写到 partials 之后的那个槽。
extern "C" __global__ void quantile_reduce_scale(
    double* scratch,
    uint32_t n_partials,
    uint32_t n_rows,
    uint32_t has_weight) {
    const double* partials = scratch;
    double* scale_out = scratch + (uint64_t)n_partials * 2;
    __shared__ double rs[QUANTILE_REDUCE_THREADS];
    __shared__ double rw[QUANTILE_REDUCE_THREADS];
    double stat = 0.0;
    double sum_w = 0.0;
    for (uint32_t b = threadIdx.x; b < n_partials; b += blockDim.x) {
        stat = __dadd_rn(stat, partials[(uint64_t)b * 2]);
        sum_w = __dadd_rn(sum_w, partials[(uint64_t)b * 2 + 1]);
    }
    rs[threadIdx.x] = stat;
    rw[threadIdx.x] = sum_w;
    __syncthreads();
    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            rs[threadIdx.x] = __dadd_rn(rs[threadIdx.x], rs[threadIdx.x + s]);
            rw[threadIdx.x] = __dadd_rn(rw[threadIdx.x], rw[threadIdx.x + s]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const double total_w = has_weight ? rw[0] : (double)n_rows;
        float scale = 0.0f;
        if (!(fabs(total_w) < 1e-6)) {
            const double root_mean = __ddiv_rn(rs[0], total_w);
            scale = __double2float_rn(__dmul_rn(root_mean, root_mean));
        }
        scale_out[0] = (double)scale;
    }
}

// 逐行 quantile 梯度,逐步对应 `Objective::quantile_gradient`(显式 _rn,不允许 FMA 收缩)。
// scale 不为正(含 NaN)时输出 (0, 0),与 host 相同。
extern "C" __global__ void quantile_gradient_rows(
    const float* pred,
    const float* label,
    const float* weight,
    uint32_t has_weight,
    uint32_t n_rows,
    float alpha,
    const double* scratch,
    uint32_t n_partials,
    float* gpair_out,
    unsigned long long subsample_threshold,
    unsigned long long subsample_seed,
    unsigned long long subsample_tree) {
    const float scale = (float)scratch[(uint64_t)n_partials * 2];
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n_rows;
         i += (uint32_t)blockDim.x * gridDim.x) {
        float g = 0.0f;
        float h = 0.0f;
        if (scale > 0.0f) {
            const float residual = __fsub_rn(pred[i], label[i]);
            const float x = __fdiv_rn(residual, __fmul_rn(0.04f, scale));
            const float t = tanhf(x);
            float ratio = (x == 0.0f) ? 1.0f : __fdiv_rn(t, x);
            ratio = fmaxf(ratio, 3.0e-4f);
            g = __fmul_rn(__fmul_rn(0.5f, scale),
                          __fsub_rn(__fadd_rn(t, 1.0f), __fmul_rn(2.0f, alpha)));
            h = __fmul_rn(__fdiv_rn(0.5f, 0.04f), ratio);
            if (has_weight) {
                const float w = weight[i];
                g = __fmul_rn(g, w);
                h = __fmul_rn(h, w);
            }
        }
        if (subsample_threshold != ~0ULL) {
            const unsigned long long hsh = fb_mix(
                subsample_seed
                ^ ((subsample_tree * 0x9E3779B97F4A7C15ULL) << 23
                   | (subsample_tree * 0x9E3779B97F4A7C15ULL) >> 41)
                ^ ((unsigned long long)i * 0xD6E8FEB86659FD93ULL));
            if ((hsh >> 11) >= subsample_threshold) {
                g = 0.0f;
                h = 0.0f;
            }
        }
        gpair_out[(uint64_t)i * 2] = g;
        gpair_out[(uint64_t)i * 2 + 1] = h;
    }
}

extern "C" __global__ void gpair_max_abs(
    float* gpair,
    uint32_t n_rows,
    double* out_max,
    unsigned long long subsample_threshold,
    unsigned long long subsample_seed,
    unsigned long long subsample_tree) {
    extern __shared__ double red[];
    double* rg = red;
    double* rh = red + blockDim.x;

    double mg = 0.0;
    double mh = 0.0;
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n_rows;
         i += (uint32_t)blockDim.x * gridDim.x) {
        // Exact objective stays on the host. Fuse only deterministic integer
        // sampling into this existing pass, BEFORE max/scale selection. Each
        // row has one writer; the subsequent quantization sees the masked pair.
        // Prediction buffers are deliberately not involved in this operation.
        if (subsample_threshold != ~0ULL) {
            const unsigned long long hsh = fb_mix(
                subsample_seed
                ^ ((subsample_tree * 0x9E3779B97F4A7C15ULL) << 23
                   | (subsample_tree * 0x9E3779B97F4A7C15ULL) >> 41)
                ^ ((unsigned long long)i * 0xD6E8FEB86659FD93ULL));
            if ((hsh >> 11) >= subsample_threshold) {
                gpair[(uint64_t)i * 2] = 0.0f;
                gpair[(uint64_t)i * 2 + 1] = 0.0f;
            }
        }
        // fmax ignores NaN, so map all non-finite pairs to an infinite bound.
        // The existing small D2H reduction then rejects them before quantization.
        const double g = (double)gpair[(uint64_t)i * 2];
        const double h = (double)gpair[(uint64_t)i * 2 + 1];
        mg = fmax(mg, isfinite(g) ? fabs(g) : INFINITY);
        mh = fmax(mh, isfinite(h) ? fabs(h) : INFINITY);
    }
    rg[threadIdx.x] = mg;
    rh[threadIdx.x] = mh;
    __syncthreads();
    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            rg[threadIdx.x] = fmax(rg[threadIdx.x], rg[threadIdx.x + s]);
            rh[threadIdx.x] = fmax(rh[threadIdx.x], rh[threadIdx.x + s]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out_max[(uint64_t)blockIdx.x * 2] = rg[0];
        out_max[(uint64_t)blockIdx.x * 2 + 1] = rh[0];
    }
}

// f32 GradPair → 定点 GradPairFixed,顺带归约出整棵树的 root node_sum。
//
// scale 由 host 用和 CPU 完全相同的 `pow2_scale` 算好传进来,所以这里只做
// 那一次精确的乘法和取整。root sum 一并算出来,是因为 GPU 路径上 host 唯一
// 还要用到 gpair 的地方就是它(`hist::build` 只在 CPU 分支里跑),
// 这样 host 就不必为了求两个整数而保留整份定点数组。
// 整数加法与顺序无关、溢出同样是 mod 2^64,所以和 host 串行累加逐位相同。
extern "C" __global__ void gpair_quantize(
    const float* gpair,
    uint32_t n_rows,
    double grad_scale,
    double hess_scale,
    long long* out_fixed,
    long long* out_sum) {
    extern __shared__ long long sred[];
    long long* sg = sred;
    long long* sh = sred + blockDim.x;

    long long acc_g = 0;
    long long acc_h = 0;
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n_rows;
         i += (uint32_t)blockDim.x * gridDim.x) {
        const long long gi =
            (long long)round((double)gpair[(uint64_t)i * 2] * grad_scale);
        const long long hi =
            (long long)round((double)gpair[(uint64_t)i * 2 + 1] * hess_scale);
        out_fixed[(uint64_t)i * 2] = gi;
        out_fixed[(uint64_t)i * 2 + 1] = hi;
        acc_g += gi;
        acc_h += hi;
    }
    sg[threadIdx.x] = acc_g;
    sh[threadIdx.x] = acc_h;
    __syncthreads();
    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sg[threadIdx.x] += sg[threadIdx.x + s];
            sh[threadIdx.x] += sh[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out_sum[(uint64_t)blockIdx.x * 2] = sg[0];
        out_sum[(uint64_t)blockIdx.x * 2 + 1] = sh[0];
    }
}

// ---------------------------------------------------------------- fast 量化尾巴
//
// `gpu_math = "fast"` 专用:把「归约 max → host 求 pow2 scale → 量化」之间那次
// host 往返去掉。max 的块间归约、pow2 scale、量化都留在 device,host 只在最后
// 和 root sum 一起做**一次**小 D2H + 一次同步,再用自己的 `GradQuantizer` 校验。
//
// scale 的公式逐步对应 `types::pow2_scale`(预算 2^62/n,向下取 2 的幂);
// device 上用 ilogb 取 floor(log2),只有 raw 恰在 2 的幂正下方 1 ULP 内时才可能和
// host 的 `log2().floor()` 差一档 —— host 拿到结果后会逐位比对 scale,不一致就用
// host 的 scale 重新量化(行为退回到旧路径),所以这里永远不会静默用错 scale。
// 非有限 / 非法 max 时 scale = 1.0;host 随后据 max 报错,不会往下用这棵树。
__device__ __forceinline__ double fb_pow2_scale(double max_abs, uint32_t n_rows) {
    if (!(max_abs > 0.0) || !isfinite(max_abs)) {
        return 1.0;
    }
    const double budget = __ddiv_rn(4611686018427387904.0, (double)(n_rows > 0u ? n_rows : 1u));
    const double raw = __ddiv_rn(budget, max_abs);
    if (!isfinite(raw) || raw <= 0.0) {
        return 1.0;
    }
    int k = ilogb(raw);
    k = k < -1000 ? -1000 : (k > 1000 ? 1000 : k);
    return ldexp(1.0, k);
}

// 单 block。out = [grad_scale, hess_scale, max_grad, max_hess]。
extern "C" __global__ void gpair_scale_finalize(
    const double* block_max,
    uint32_t n_blocks,
    uint32_t n_rows,
    double* out) {
    __shared__ double rg[256];
    __shared__ double rh[256];
    double mg = 0.0;
    double mh = 0.0;
    for (uint32_t b = threadIdx.x; b < n_blocks; b += blockDim.x) {
        mg = fmax(mg, block_max[(uint64_t)b * 2]);
        mh = fmax(mh, block_max[(uint64_t)b * 2 + 1]);
    }
    rg[threadIdx.x] = mg;
    rh[threadIdx.x] = mh;
    __syncthreads();
    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            rg[threadIdx.x] = fmax(rg[threadIdx.x], rg[threadIdx.x + s]);
            rh[threadIdx.x] = fmax(rh[threadIdx.x], rh[threadIdx.x + s]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out[0] = fb_pow2_scale(rg[0], n_rows);
        out[1] = fb_pow2_scale(rh[0], n_rows);
        out[2] = rg[0];
        out[3] = rh[0];
    }
}

// 与 `gpair_quantize` 逐字相同,只是 scale 从 device 读(`gpair_scale_finalize` 的输出)。
extern "C" __global__ void gpair_quantize_dscale(
    const float* gpair,
    uint32_t n_rows,
    const double* scales,
    long long* out_fixed,
    long long* out_sum) {
    extern __shared__ long long sred[];
    long long* sg = sred;
    long long* sh = sred + blockDim.x;
    const double grad_scale = scales[0];
    const double hess_scale = scales[1];

    long long acc_g = 0;
    long long acc_h = 0;
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n_rows;
         i += (uint32_t)blockDim.x * gridDim.x) {
        const long long gi =
            (long long)round((double)gpair[(uint64_t)i * 2] * grad_scale);
        const long long hi =
            (long long)round((double)gpair[(uint64_t)i * 2 + 1] * hess_scale);
        out_fixed[(uint64_t)i * 2] = gi;
        out_fixed[(uint64_t)i * 2 + 1] = hi;
        acc_g += gi;
        acc_h += hi;
    }
    sg[threadIdx.x] = acc_g;
    sh[threadIdx.x] = acc_h;
    __syncthreads();
    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            sg[threadIdx.x] += sg[threadIdx.x + s];
            sh[threadIdx.x] += sh[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        out_sum[(uint64_t)blockIdx.x * 2] = sg[0];
        out_sum[(uint64_t)blockIdx.x * 2 + 1] = sh[0];
    }
}

// 行重分区的三个阶段:分类并计数、确定性块级 exclusive scan、稳定 scatter。
// 这里使用 device-side scan 以保持 checked-in PTX + cudarc 的构建架构;CUB 的
// DeviceScan 是 host API,不能从纯 PTX kernel 内调用。scan 的结果和 CUB
// DeviceScan::ExclusiveSum 的契约相同,并由 Rust 侧逐字节对拍 CPU 版本。

// 第一版只验证数值契约:每个 CUDA block 故意只有一个线程,blockIdx.x
// 就是所选行数组的位置。这样 selected-row 数量由 launch grid 携带,
// n_rows 可以继续表示列主序 bins 的物理行 stride,不需要改既定 FFI。
extern "C" __global__ void hist_build(
    const uint8_t* bins,
    const int64_t* gpair,
    const uint32_t* row_idx,
    uint32_t n_rows,
    uint32_t n_feats,
    const uint32_t* feat_offsets,
    int64_t* out_hist) {
    const uint32_t selected = blockIdx.x;
    const uint32_t row = row_idx == nullptr ? selected : row_idx[selected];

    for (uint32_t feat = 0; feat < n_feats; ++feat) {
        const uint8_t bin = bins[(uint64_t)feat * n_rows + row];
        if (bin == UINT8_MAX) {
            continue;
        }

        const uint64_t slot = (uint64_t)feat_offsets[feat] + bin;
        // CUDA 的 64-bit integer atomicAdd 接受 unsigned long long。二补数
        // 模 2^64 加法与 i64 加法位模式相同;grad 可以为负,hess 通常为正。
        atomicAdd(
            reinterpret_cast<unsigned long long*>(&out_hist[slot * 2]),
            static_cast<unsigned long long>(gpair[(uint64_t)row * 2]));
        atomicAdd(
            reinterpret_cast<unsigned long long*>(&out_hist[slot * 2 + 1]),
            static_cast<unsigned long long>(gpair[(uint64_t)row * 2 + 1]));
    }
}

// 把一次 64-bit shared atomicAdd 拆成两次 32-bit atomicAdd + 手动进位。
//
// **为什么**:sm_86 上 64-bit shared atomic 实测 3.78 wavefronts/指令,
// 而同形状的 32-bit shared store 只有 1.999 —— 64-bit 原子单元本身比
// 32 位慢约 1.9×,而 accumulate 占 shared 管道流量的 95.2%。
// XGBoost 的 `AtomicAdd64As32` 用的是同一个技巧,并且注释说明它**只对
// shared 有效,global 拆开没有好处**,所以这里也只用在 shared 上。
//
// **为什么仍然逐位精确**:最终低 32 位 = (Σ 低位) mod 2^32,与顺序无关;
// 每次 atomicAdd 返回的是这一次的真实旧值,所以 `old > 0xFFFFFFFF - x_lo`
// 恰好判定**这一次**是否进位,累加起来就是低位总和溢出的次数,也就是
// floor((Σ 低位) / 2^32)。因此高 32 位 = (Σ 高位 + 总进位) mod 2^32,
// 合起来正是 (Σ src) mod 2^64 —— 和一次 64-bit 原子加完全相同,
// 而且同样与累加顺序无关(定点整数的位精确契约靠的就是这一条)。
//
// signed grad 以二补数位模式传进来,模 2^64 加法对二补数就是有符号加法,
// 这一点和原来的 64-bit 路径没有区别。
//
// 对齐:shared_hist 是 8 字节对齐的 u64 数组,拆出来的 lo/hi 各自 4 字节
// 对齐,满足 32-bit atomic 的要求。NVIDIA GPU 是小端,lo 在低地址。
__device__ __forceinline__ void atomic_add_shared_u64_as_u32(
    unsigned long long* dst, unsigned long long src) {
    unsigned int* lo = reinterpret_cast<unsigned int*>(dst);
    unsigned int* hi = lo + 1;
    const unsigned int x_lo = static_cast<unsigned int>(src);
    const unsigned int x_hi = static_cast<unsigned int>(src >> 32);
    const unsigned int old = atomicAdd(lo, x_lo);
    const unsigned int carry = (old > (0xFFFFFFFFu - x_lo)) ? 1u : 0u;
    atomicAdd(hi, x_hi + carry);
}

// 正确性优先的 shared-memory 版本。host 按实际 shared-memory 容量把
// 特征切成 tile;每个 block 处理一段连续的 selected rows。block 内先把
// 两个 i64 字段累加到 shared,再原子 flush 到 global。两层都只有整数加法,
// 所以顺序变化不改变位模式。
extern "C" __global__ void hist_build_shared(
    const uint8_t* bins,
    const int64_t* gpair,
    const uint32_t* row_idx,
    uint32_t n_rows,
    uint32_t n_feats,
    const uint32_t* feat_offsets,
    int64_t* out_hist) {
    extern __shared__ unsigned long long shared_hist[];

    const uint32_t hist_begin = feat_offsets[0];
    const uint32_t hist_end = feat_offsets[n_feats];
    const uint32_t shared_words = (hist_end - hist_begin) * 2;

    for (uint32_t word = threadIdx.x; word < shared_words; word += blockDim.x) {
        shared_hist[word] = 0;
    }
    __syncthreads();

    const uint32_t selected = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t row = row_idx == nullptr ? selected : row_idx[selected];
    // Each selected row contributes the same gradient pair to every feature.
    // Load it once into registers instead of issuing two global loads for
    // every feature.  Integer accumulation order and two's-complement bits
    // are unchanged; this only removes redundant memory instructions.
    const unsigned long long row_grad =
        static_cast<unsigned long long>(gpair[(uint64_t)row * 2]);
    const unsigned long long row_hess =
        static_cast<unsigned long long>(gpair[(uint64_t)row * 2 + 1]);
    for (uint32_t feat = 0; feat < n_feats; ++feat) {
        const uint8_t bin = bins[(uint64_t)feat * n_rows + row];
        if (bin == UINT8_MAX) {
            continue;
        }

        const uint32_t local_slot = feat_offsets[feat] + bin - hist_begin;
        // 拆成两次 32-bit + 进位。实测比一次 64-bit shared atomic 快
        // 2.34×(kernel 6.545 → 2.795 ms),见 atomic_add_shared_u64_as_u32
        // 上面的说明和 CLAUDE 的 A/B。
        atomic_add_shared_u64_as_u32(&shared_hist[(uint64_t)local_slot * 2], row_grad);
        atomic_add_shared_u64_as_u32(&shared_hist[(uint64_t)local_slot * 2 + 1], row_hess);
    }
    __syncthreads();

    for (uint32_t word = threadIdx.x; word < shared_words; word += blockDim.x) {
        atomicAdd(
            reinterpret_cast<unsigned long long*>(out_hist) + (uint64_t)hist_begin * 2 + word,
            shared_hist[word]);
    }
}

// Planar32 + **把一个列块的所有 feature tile 融进一次 launch**。
//
// **为什么**:host 侧现在是「每个 feature tile 一次 launch」,而每次 launch
// 的每个线程都要重新读一遍自己那一行的 `gpair`(16 B)和 `row_idx`(4 B)。
// wide 的 `cols_per_block=32` 切成 4 个 tile,于是 gpair/row_idx 被**读 4 遍**。
// 实测一次 8-feature launch 的 global load 是 540 MB,而这 8 个特征的 bins
// 只有 80 MB —— 也就是说约 85% 的 global 流量是被重复读的 gpair/row_idx。
//
// 融合之后每个 block 只读一次 gpair/row_idx,然后在 kernel 内依次处理各个
// tile。**减少的是真正被执行的 load 指令数**,不是把同样的工作换个方式摊派
// (CLAUDE 第八条:重新分配 ≠ 减少昂贵操作)。
//
// bins 的读取量、shared 原子次数、init/flush 次数**都完全不变**,所以这不
// 影响位精确,也不改变 shared 预算和 occupancy。
//
// ⚠️ 这个 kernel 有**八个**参数,比既定的七参数 ABI 多一个 `max_tile_bins`。
// 这是有意的:tile 边界原本由 host 算(依赖 driver 报的 shared 上限),融合
// 之后必须让 kernel 自己走同一套贪心切分,所以那个上限得传进来。既有的
// 七参数 kernel 一个都没改。
extern "C" __global__ void hist_build_shared_planar32_fused(
    const uint8_t* bins,
    const int64_t* gpair,
    const uint32_t* row_idx,
    uint32_t n_rows,
    uint32_t n_feats,
    const uint32_t* feat_offsets,
    int64_t* out_hist,
    uint32_t max_tile_bins) {
    extern __shared__ unsigned int fused_hist[];

    const uint32_t selected = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t row = row_idx == nullptr ? selected : row_idx[selected];
    // 整个 block 的所有 tile 共用这一次读取 —— 这就是融合的全部收益来源。
    const unsigned long long row_grad =
        static_cast<unsigned long long>(gpair[(uint64_t)row * 2]);
    const unsigned long long row_hess =
        static_cast<unsigned long long>(gpair[(uint64_t)row * 2 + 1]);
    const unsigned int grad_lo_v = static_cast<unsigned int>(row_grad);
    const unsigned int grad_hi_v = static_cast<unsigned int>(row_grad >> 32);
    const unsigned int hess_lo_v = static_cast<unsigned int>(row_hess);
    const unsigned int hess_hi_v = static_cast<unsigned int>(row_hess >> 32);

    uint32_t f0 = 0;
    while (f0 < n_feats) {
        // 和 host 侧 train_ctx.rs 里的贪心切分逐字一致:尽量多装连续特征,
        // 直到超过 shared 能放下的 bin 数。
        const uint32_t hist_begin = feat_offsets[f0];
        uint32_t f1 = f0;
        while (f1 < n_feats && feat_offsets[f1 + 1] - hist_begin <= max_tile_bins) {
            ++f1;
        }
        const uint32_t n_slots = feat_offsets[f1] - hist_begin;

        unsigned int* grad_lo = fused_hist;
        unsigned int* grad_hi = fused_hist + n_slots;
        unsigned int* hess_lo = fused_hist + (uint64_t)n_slots * 2;
        unsigned int* hess_hi = fused_hist + (uint64_t)n_slots * 3;

        for (uint32_t word = threadIdx.x; word < n_slots * 4; word += blockDim.x) {
            fused_hist[word] = 0;
        }
        __syncthreads();

        for (uint32_t feat = f0; feat < f1; ++feat) {
            const uint8_t bin = bins[(uint64_t)feat * n_rows + row];
            if (bin == UINT8_MAX) {
                continue;
            }
            const uint32_t s = feat_offsets[feat] + bin - hist_begin;
            const unsigned int g_old = atomicAdd(&grad_lo[s], grad_lo_v);
            atomicAdd(&grad_hi[s],
                      grad_hi_v + ((g_old > (0xFFFFFFFFu - grad_lo_v)) ? 1u : 0u));
            const unsigned int h_old = atomicAdd(&hess_lo[s], hess_lo_v);
            atomicAdd(&hess_hi[s],
                      hess_hi_v + ((h_old > (0xFFFFFFFFu - hess_lo_v)) ? 1u : 0u));
        }
        __syncthreads();

        for (uint32_t word = threadIdx.x; word < n_slots * 2; word += blockDim.x) {
            const uint32_t s = word >> 1;
            const bool is_hess = (word & 1) != 0;
            const unsigned int lo = is_hess ? hess_lo[s] : grad_lo[s];
            const unsigned int hi = is_hess ? hess_hi[s] : grad_hi[s];
            const unsigned long long v =
                ((unsigned long long)hi << 32) | (unsigned long long)lo;
            atomicAdd(
                reinterpret_cast<unsigned long long*>(out_hist) + (uint64_t)hist_begin * 2 + word,
                v);
        }
        // flush 读 shared,下一个 tile 的清零写 shared —— 中间必须再同步一次,
        // 否则跑得快的线程会在别人读完之前把 buffer 清掉。
        __syncthreads();

        f0 = f1;
    }
}

// R=4 multi-row/thread:每个线程处理 4 行,其余全部不变。
//
// **收益来源只有摊薄**:一个 block 的 shared init、flush、barrier 和 launch
// 是固定的,现在摊到 4 倍的行上。**atomic 次数、bins 读取量、gpair 读取量
// 都一模一样**(每个 (row, feature) 仍然 4 次 32-bit atomic)。
//
// coalescing 保持:线程 t 取 `t, t+512, t+1024, t+1536`,所以每个 row-step
// 内一个 warp 仍然读 32 个连续行的 bin —— 和 1 行/线程时的访存模式相同。
//
// 行数不整除 `blockDim.x * 4` 的尾部交回一行/线程的 kernel 处理,
// 所以这里不需要每行判边界。
//
// ⚠️ 2026-09-02 第一次做时实测只有 2.2%(wide),按当时 5% 的门槛回滚了;
// 实现从未进过提交。现在基线快了一倍,同样的约 24 ms 绝对收益约合 3.3%,
// 所以按 memo 重开 —— **这是重写,不是恢复**。
extern "C" __global__ void hist_build_shared_planar32_fused_r4(
    const uint8_t* bins,
    const int64_t* gpair,
    const uint32_t* row_idx,
    uint32_t n_rows,
    uint32_t n_feats,
    const uint32_t* feat_offsets,
    int64_t* out_hist,
    uint32_t max_tile_bins) {
    extern __shared__ unsigned int fused_hist_r4[];

    const uint32_t base = blockIdx.x * (blockDim.x * ROWS_PER_THREAD) + threadIdx.x;
    uint32_t rows_v[ROWS_PER_THREAD];
    unsigned int g_lo[ROWS_PER_THREAD], g_hi[ROWS_PER_THREAD];
    unsigned int h_lo[ROWS_PER_THREAD], h_hi[ROWS_PER_THREAD];
#pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; ++r) {
        const uint32_t sel = base + r * blockDim.x;
        const uint32_t row = row_idx == nullptr ? sel : row_idx[sel];
        rows_v[r] = row;
        const unsigned long long gr =
            static_cast<unsigned long long>(gpair[(uint64_t)row * 2]);
        const unsigned long long hr =
            static_cast<unsigned long long>(gpair[(uint64_t)row * 2 + 1]);
        g_lo[r] = static_cast<unsigned int>(gr);
        g_hi[r] = static_cast<unsigned int>(gr >> 32);
        h_lo[r] = static_cast<unsigned int>(hr);
        h_hi[r] = static_cast<unsigned int>(hr >> 32);
    }

    uint32_t f0 = 0;
    while (f0 < n_feats) {
        const uint32_t hist_begin = feat_offsets[f0];
        uint32_t f1 = f0;
        while (f1 < n_feats && feat_offsets[f1 + 1] - hist_begin <= max_tile_bins) {
            ++f1;
        }
        const uint32_t n_slots = feat_offsets[f1] - hist_begin;

        unsigned int* grad_lo = fused_hist_r4;
        unsigned int* grad_hi = fused_hist_r4 + n_slots;
        unsigned int* hess_lo = fused_hist_r4 + (uint64_t)n_slots * 2;
        unsigned int* hess_hi = fused_hist_r4 + (uint64_t)n_slots * 3;

        for (uint32_t word = threadIdx.x; word < n_slots * 4; word += blockDim.x) {
            fused_hist_r4[word] = 0;
        }
        __syncthreads();

        for (uint32_t feat = f0; feat < f1; ++feat) {
            const uint32_t off = feat_offsets[feat] - hist_begin;
#pragma unroll
            for (int r = 0; r < ROWS_PER_THREAD; ++r) {
                const uint8_t bin = bins[(uint64_t)feat * n_rows + rows_v[r]];
                if (bin == UINT8_MAX) {
                    continue;
                }
                const uint32_t s = off + bin;
                const unsigned int go = atomicAdd(&grad_lo[s], g_lo[r]);
                atomicAdd(&grad_hi[s],
                          g_hi[r] + ((go > (0xFFFFFFFFu - g_lo[r])) ? 1u : 0u));
                const unsigned int ho = atomicAdd(&hess_lo[s], h_lo[r]);
                atomicAdd(&hess_hi[s],
                          h_hi[r] + ((ho > (0xFFFFFFFFu - h_lo[r])) ? 1u : 0u));
            }
        }
        __syncthreads();

        const uint32_t out_words = n_slots * 2;
        for (uint32_t word = threadIdx.x; word < out_words; word += blockDim.x) {
            const uint32_t s = word >> 1;
            const bool is_hess = (word & 1) != 0;
            const unsigned int lo = is_hess ? hess_lo[s] : grad_lo[s];
            const unsigned int hi = is_hess ? hess_hi[s] : grad_hi[s];
            const unsigned long long v =
                ((unsigned long long)hi << 32) | (unsigned long long)lo;
            atomicAdd(
                reinterpret_cast<unsigned long long*>(out_hist) + (uint64_t)hist_begin * 2 + word,
                v);
        }
        __syncthreads();

        f0 = f1;
    }
}


// 一层里多个 Accumulate 节点合并进一次 launch。
//
// **几何完全不变**:仍然是一个 block 负责一个节点的 `blockDim.x *
// ROWS_PER_THREAD` 行,仍然在 block 内顺序扫 feature tile,shared 布局、
// 原子方式、flush 方式和 `..._fused_r4` 逐字相同。**唯一的变化是
// block → (node, rows, out) 从隐式的 blockIdx 算术变成查 `blk_ptr`。**
//
// **为什么值得**:审计(internal-docs/history.md 2026-09-03)显示 launch 本身不贵
// (91 次 launch 之间的纯间隙合计只有 0.11 ms),真正的固定开销是
// **每节点一次 host 往返造成的 3.9 ms GPU 空闲** —— launch → D2H →
// sync → host 枚举分裂 → launch 下一个。整层一次 launch 把 26 段往返
// 压到每层一次。附带好处是 66 次装不满一个 block/SM 的小 launch
// 现在会和同层其它节点共同调度。
//
// **位精确**:定点加法按模 2^64 精确且与顺序无关,合并 launch 只改变
// block 的调度顺序,不改变任何一次加法的操作数。越界行走
// `ok[r] == false` 直接跳过,和「不存在这一行」等价 —— 不是加 0,
// 所以连原子次数都不变。
extern "C" __global__ void hist_build_shared_planar32_batched(
    const uint8_t* bins,
    const int64_t* gpair,
    const uint32_t* row_idx,
    uint32_t n_rows,
    uint32_t n_feats,
    const uint32_t* feat_offsets,
    int64_t* out_hist,
    uint32_t max_tile_bins,
    const uint32_t* batch_idx,
    uint32_t n_nodes,
    uint32_t hist_stride,
    const uint32_t* feature_select,
    uint32_t selection_active) {
    // Match the row-major ABI and retain one safe dependency per allocation.
    const uint32_t* blk_ptr = batch_idx;
    const uint32_t* node_off = batch_idx + n_nodes + 1;
    const uint32_t* node_len = batch_idx + 2 * n_nodes + 1;
    extern __shared__ unsigned int fused_hist_b[];
    __shared__ uint32_t s_node;

    // blk_ptr 是 [n_nodes+1] 的前缀和,找 k 使 blk_ptr[k] <= bid < blk_ptr[k+1]。
    // 整个 block 的结果相同,所以让 0 号线程算一次放进 shared,
    // 避免 blockDim.x 个线程各做一遍同样的二分。
    if (threadIdx.x == 0) {
        const uint32_t bid = blockIdx.x;
        uint32_t lo = 0, hi = n_nodes;
        while (lo + 1 < hi) {
            const uint32_t mid = (lo + hi) >> 1;
            if (blk_ptr[mid] <= bid) {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        s_node = lo;
    }
    __syncthreads();

    const uint32_t k = s_node;
    const uint32_t rows_per_block = blockDim.x * ROWS_PER_THREAD;
    const uint32_t local = blockIdx.x - blk_ptr[k];
    const uint32_t start = node_off[k] + local * rows_per_block;
    const uint32_t end = node_off[k] + node_len[k];
    int64_t* out = out_hist + (uint64_t)k * hist_stride;

    uint32_t rows_v[ROWS_PER_THREAD];
    unsigned int g_lo[ROWS_PER_THREAD], g_hi[ROWS_PER_THREAD];
    unsigned int h_lo[ROWS_PER_THREAD], h_hi[ROWS_PER_THREAD];
    bool ok[ROWS_PER_THREAD];
#pragma unroll
    for (int r = 0; r < ROWS_PER_THREAD; ++r) {
        const uint32_t sel = start + threadIdx.x + (uint32_t)r * blockDim.x;
        ok[r] = sel < end;
        rows_v[r] = 0;
        g_lo[r] = 0; g_hi[r] = 0; h_lo[r] = 0; h_hi[r] = 0;
        if (ok[r]) {
            const uint32_t row = row_idx == nullptr ? sel : row_idx[sel];
            rows_v[r] = row;
            const unsigned long long gr =
                static_cast<unsigned long long>(gpair[(uint64_t)row * 2]);
            const unsigned long long hr =
                static_cast<unsigned long long>(gpair[(uint64_t)row * 2 + 1]);
            g_lo[r] = static_cast<unsigned int>(gr);
            g_hi[r] = static_cast<unsigned int>(gr >> 32);
            h_lo[r] = static_cast<unsigned int>(hr);
            h_hi[r] = static_cast<unsigned int>(hr >> 32);
        }
    }

    uint32_t f0 = 0;
    while (f0 < n_feats) {
        const uint32_t hist_begin = feat_offsets[f0];
        uint32_t f1 = f0;
        while (f1 < n_feats && feat_offsets[f1 + 1] - hist_begin <= max_tile_bins) {
            ++f1;
        }
        const uint32_t n_slots = feat_offsets[f1] - hist_begin;

        unsigned int* grad_lo = fused_hist_b;
        unsigned int* grad_hi = fused_hist_b + n_slots;
        unsigned int* hess_lo = fused_hist_b + (uint64_t)n_slots * 2;
        unsigned int* hess_hi = fused_hist_b + (uint64_t)n_slots * 3;

        for (uint32_t word = threadIdx.x; word < n_slots * 4; word += blockDim.x) {
            fused_hist_b[word] = 0;
        }
        __syncthreads();

        for (uint32_t feat = f0; feat < f1; ++feat) {
            const uint32_t physical_feat =
                selection_active != 0 ? feature_select[feat] : feat;
            const uint32_t off = feat_offsets[feat] - hist_begin;
#pragma unroll
            for (int r = 0; r < ROWS_PER_THREAD; ++r) {
                if (!ok[r]) {
                    continue;
                }
                const uint8_t bin = bins[(uint64_t)physical_feat * n_rows + rows_v[r]];
                if (bin == UINT8_MAX) {
                    continue;
                }
                const uint32_t s = off + bin;
                const unsigned int go = atomicAdd(&grad_lo[s], g_lo[r]);
                atomicAdd(&grad_hi[s],
                          g_hi[r] + ((go > (0xFFFFFFFFu - g_lo[r])) ? 1u : 0u));
                const unsigned int ho = atomicAdd(&hess_lo[s], h_lo[r]);
                atomicAdd(&hess_hi[s],
                          h_hi[r] + ((ho > (0xFFFFFFFFu - h_lo[r])) ? 1u : 0u));
            }
        }
        __syncthreads();

        const uint32_t out_words = n_slots * 2;
        for (uint32_t word = threadIdx.x; word < out_words; word += blockDim.x) {
            const uint32_t s = word >> 1;
            const bool is_hess = (word & 1) != 0;
            const unsigned int lo = is_hess ? hess_lo[s] : grad_lo[s];
            const unsigned int hi = is_hess ? hess_hi[s] : grad_hi[s];
            const unsigned long long v =
                ((unsigned long long)hi << 32) | (unsigned long long)lo;
            atomicAdd(
                reinterpret_cast<unsigned long long*>(out) + (uint64_t)hist_begin * 2 + word,
                v);
        }
        __syncthreads();

        f0 = f1;
    }
}


// 与 hist_build_shared 相同的七参数 ABI,但 shared 里改成**四个 u32 平面**
// (grad_lo / grad_hi / hess_lo / hess_hi),每个平面按 slot 单位步长索引。
//
// **为什么**:交错的 16-byte record 让 slot s 的 grad_lo 落在 word 4s,
// 于是 bank = 4*(s%8) —— 一个 warp 的 32 个 lane **结构性地只能命中 8 个
// bank**,至少 4-way 冲突。实测 `AtomicAdd64As32` 之后 shared atomic
// wavefronts 72,673,570 ≈ 指令 9,999,872 + bank conflicts 62,174,032,
// 也就是 **85% 的 shared 原子流量是冲突重放**。
//
// 拆成四个平面之后 bank = (base + s) % 32,32 个 bank 全部用上,
// 冲突只剩「两个 lane 落进同一个 bin」这种真实的地址碰撞。
//
// ⚠️ 这个布局在 64-bit shared atomic 时代是**非法**的(64-bit atomicAdd
// 要求 8 字节对齐,拆开的平面给不了);是 `AtomicAdd64As32` 把这一维重新
// 打开的 —— CLAUDE 里「padding/bank 布局上限 1.01×」和「20-byte stride
// 根本不可行」两条结论都是 64-bit kernel 的产物,对这里不适用。
//
// shared 字节数完全不变(4 个平面 × 4 B = 16 B/slot),所以 host 侧的
// feature tile 计算、occupancy 和 launch geometry 一个都不用改。
//
// 位精确:每个平面仍是 mod 2^32 的整数累加,进位判定和
// atomic_add_shared_u64_as_u32 逐字相同,因此与顺序无关、与一次
// 64-bit 原子加逐位相同。
extern "C" __global__ void hist_build_shared_planar32(
    const uint8_t* bins,
    const int64_t* gpair,
    const uint32_t* row_idx,
    uint32_t n_rows,
    uint32_t n_feats,
    const uint32_t* feat_offsets,
    int64_t* out_hist) {
    extern __shared__ unsigned int planar_hist[];

    const uint32_t hist_begin = feat_offsets[0];
    const uint32_t hist_end = feat_offsets[n_feats];
    const uint32_t n_slots = hist_end - hist_begin;
    const uint32_t shared_words = n_slots * 4;

    unsigned int* grad_lo = planar_hist;
    unsigned int* grad_hi = planar_hist + n_slots;
    unsigned int* hess_lo = planar_hist + (uint64_t)n_slots * 2;
    unsigned int* hess_hi = planar_hist + (uint64_t)n_slots * 3;

    for (uint32_t word = threadIdx.x; word < shared_words; word += blockDim.x) {
        planar_hist[word] = 0;
    }
    __syncthreads();

    const uint32_t selected = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t row = row_idx == nullptr ? selected : row_idx[selected];
    const unsigned long long row_grad =
        static_cast<unsigned long long>(gpair[(uint64_t)row * 2]);
    const unsigned long long row_hess =
        static_cast<unsigned long long>(gpair[(uint64_t)row * 2 + 1]);
    const unsigned int grad_lo_v = static_cast<unsigned int>(row_grad);
    const unsigned int grad_hi_v = static_cast<unsigned int>(row_grad >> 32);
    const unsigned int hess_lo_v = static_cast<unsigned int>(row_hess);
    const unsigned int hess_hi_v = static_cast<unsigned int>(row_hess >> 32);

    for (uint32_t feat = 0; feat < n_feats; ++feat) {
        const uint8_t bin = bins[(uint64_t)feat * n_rows + row];
        if (bin == UINT8_MAX) {
            continue;
        }

        const uint32_t s = feat_offsets[feat] + bin - hist_begin;

        const unsigned int g_old = atomicAdd(&grad_lo[s], grad_lo_v);
        atomicAdd(&grad_hi[s],
                  grad_hi_v + ((g_old > (0xFFFFFFFFu - grad_lo_v)) ? 1u : 0u));
        const unsigned int h_old = atomicAdd(&hess_lo[s], hess_lo_v);
        atomicAdd(&hess_hi[s],
                  hess_hi_v + ((h_old > (0xFFFFFFFFu - hess_lo_v)) ? 1u : 0u));
    }
    __syncthreads();

    // flush 的 global 写入模式和 Interleaved16 完全一致(每个线程负责连续的
    // u64 word),所以 global RED 的 coalescing 不受这次改动影响;变的只有
    // 读 shared 的地址。
    const uint32_t out_words = n_slots * 2;
    for (uint32_t word = threadIdx.x; word < out_words; word += blockDim.x) {
        const uint32_t s = word >> 1;
        const bool is_hess = (word & 1) != 0;
        const unsigned int lo = is_hess ? hess_lo[s] : grad_lo[s];
        const unsigned int hi = is_hess ? hess_hi[s] : grad_hi[s];
        const unsigned long long v = ((unsigned long long)hi << 32) | (unsigned long long)lo;
        atomicAdd(
            reinterpret_cast<unsigned long long*>(out_hist) + (uint64_t)hist_begin * 2 + word,
            v);
    }
}

// 与 hist_build_shared 使用相同的七参数 ABI,但仅在 block-local shared
// memory 中把 GradPairFixed 拆成两个 i64 数组。每个数组的元素间距是 8 B
// (两个 bank),既保持 64-bit atomic 的对齐要求,也避开 16 B 交错 record
// 的四-bank stride。global memory 的 GradPairFixed 仍为交错布局。
extern "C" __global__ void hist_build_shared_split(
    const uint8_t* bins,
    const int64_t* gpair,
    const uint32_t* row_idx,
    uint32_t n_rows,
    uint32_t n_feats,
    const uint32_t* feat_offsets,
    int64_t* out_hist) {
    extern __shared__ unsigned long long shared_hist[];

    const uint32_t hist_begin = feat_offsets[0];
    const uint32_t hist_end = feat_offsets[n_feats];
    const uint32_t tile_bins = hist_end - hist_begin;
    unsigned long long* const shared_grad = shared_hist;
    unsigned long long* const shared_hess = shared_hist + tile_bins;

    for (uint32_t slot = threadIdx.x; slot < tile_bins; slot += blockDim.x) {
        shared_grad[slot] = 0;
        shared_hess[slot] = 0;
    }
    __syncthreads();

    const uint32_t selected = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t row = row_idx[selected];
    for (uint32_t feat = 0; feat < n_feats; ++feat) {
        const uint8_t bin = bins[(uint64_t)feat * n_rows + row];
        if (bin == UINT8_MAX) {
            continue;
        }

        const uint32_t local_slot = feat_offsets[feat] + bin - hist_begin;
        atomicAdd(
            &shared_grad[local_slot],
            static_cast<unsigned long long>(gpair[(uint64_t)row * 2]));
        atomicAdd(
            &shared_hess[local_slot],
            static_cast<unsigned long long>(gpair[(uint64_t)row * 2 + 1]));
    }
    __syncthreads();

    for (uint32_t slot = threadIdx.x; slot < tile_bins; slot += blockDim.x) {
        const uint64_t global_slot = (uint64_t)hist_begin + slot;
        atomicAdd(
            reinterpret_cast<unsigned long long*>(out_hist) + global_slot * 2,
            shared_grad[slot]);
        atomicAdd(
            reinterpret_cast<unsigned long long*>(out_hist) + global_slot * 2 + 1,
            shared_hess[slot]);
    }
}

// `split_bin == CAT_SPLIT` marks a categorical split: `cat_mask` is the 256-bit
// set of categories that go RIGHT (8 x u32, bit `bin % 32` of word `bin / 32`);
// every other category goes left. Missing rows follow `missing_left` as ever.
// For a numeric split `cat_mask` is never read (any valid pointer will do).
enum : uint32_t { CAT_SPLIT = 0xFFFFFFFFu };

__device__ __forceinline__ uint32_t partition_goes_left(
    uint8_t bin, uint32_t split_bin, uint32_t missing_left, const uint32_t* cat_mask) {
    if (bin == UINT8_MAX) return missing_left;
    if (split_bin == CAT_SPLIT) return ((cat_mask[bin >> 5] >> (bin & 31)) & 1u) ^ 1u;
    return (uint32_t)(bin < split_bin);
}

extern "C" __global__ void partition_count(
    const uint8_t* col,
    const uint32_t* rows,
    uint32_t n_rows,
    uint32_t n_selected,
    uint32_t split_bin,
    uint32_t missing_left,
    uint8_t* flags,
    uint32_t* block_left_counts,
    const uint32_t* cat_mask) {
    extern __shared__ uint32_t counts[];
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t left = 0;
    if (idx < n_selected) {
        const uint32_t row = rows[idx];
        const uint8_t bin = col[(uint64_t)row];
        left = partition_goes_left(bin, split_bin, missing_left, cat_mask);
        flags[idx] = (uint8_t)left;
    }
    counts[threadIdx.x] = left;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            counts[threadIdx.x] += counts[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        block_left_counts[blockIdx.x] = counts[0];
    }
}

// One block is enough for the block-count array (the row grid is normally far
// larger than this). Keeping this scan in device code makes offsets deterministic
// and avoids an atomic reservation that would scramble stable row order.
// scan 顺带把这个节点的左右总数写进 `totals_out[2*slot]` / `[2*slot+1]`。
//
// **为什么只留 totals**:count → scan → scatter 是同一条 stream 上顺序排的,
// 所以节点 i 的 scatter 一定在节点 i+1 的 count 之前跑完 —— offsets 和 flags
// 可以照旧按节点复用,**不需要**整层留着。host 唯一需要活到本层结束的,
// 就是每个节点的左右行数(用来构造下一层的 span)。把这两个数单独攒起来,
// 就能把每节点两次阻塞 D2H 压成整层一次,而不用给 offsets 开一整层的 arena。
__device__ __forceinline__ void partition_scan_segment(
    const uint32_t* block_left_counts,
    uint32_t* left_offsets,
    uint32_t* right_offsets,
    uint32_t n_blocks,
    uint32_t n_selected,
    uint32_t row_block_size,
    uint32_t* totals_out,
    uint32_t slot) {
    // **单 block、多线程**的分块 scan。
    //
    // 以前这里是 `grid(1,1,1) block(1,1,1)` —— 一个线程串行走完本节点的所有
    // block,root 节点上就是循环 20,509 次。ncu 实测它占 partition 三个 kernel
    // 的 45%(0.872 / 1.944 ms),**比旁边两个真正并行的还慢**。
    //
    // 为什么是单 block 而不是多 block:CUB 的 `DeviceScan` 是 **host API**,
    // 从 checked-in PTX 里调不到(见文件顶部说明);多 block scan 需要
    // 跨 block 同步或两趟 kernel。单 block 多线程已经把 O(n) 的串行链变成
    // 「n/p 个 tile × O(log p)」,而且不引入任何新的 launch 或状态。
    //
    // **结果与串行版逐位相同**:整数前缀和,tile 内是标准 Hillis-Steele
    // inclusive scan,tile 之间用 carry 串起来,得到的还是同一组精确前缀和。
    extern __shared__ uint32_t scan_smem[];
    uint32_t* sl = scan_smem;
    uint32_t* sr = scan_smem + blockDim.x;

    if (threadIdx.x == 0) {
        left_offsets[0] = 0;
        right_offsets[0] = 0;
    }

    uint32_t carry_l = 0;
    uint32_t carry_r = 0;
    for (uint32_t base = 0; base < n_blocks; base += blockDim.x) {
        const uint32_t i = base + threadIdx.x;
        uint32_t l = 0;
        uint32_t r = 0;
        if (i < n_blocks) {
            const uint32_t start = i * row_block_size;
            const uint32_t rows_in_block = min(row_block_size, n_selected - start);
            l = block_left_counts[i];
            r = rows_in_block - l;
        }
        sl[threadIdx.x] = l;
        sr[threadIdx.x] = r;
        __syncthreads();

        for (uint32_t off = 1; off < blockDim.x; off <<= 1) {
            const uint32_t add_l = threadIdx.x >= off ? sl[threadIdx.x - off] : 0u;
            const uint32_t add_r = threadIdx.x >= off ? sr[threadIdx.x - off] : 0u;
            __syncthreads();
            sl[threadIdx.x] += add_l;
            sr[threadIdx.x] += add_r;
            __syncthreads();
        }

        if (i < n_blocks) {
            left_offsets[i + 1] = carry_l + sl[threadIdx.x];
            right_offsets[i + 1] = carry_r + sr[threadIdx.x];
        }
        __syncthreads();

        // 本 tile 的总和 = 最后一个有效元素的 inclusive 值。所有线程都算同一个
        // carry,保持一致;必须在下一轮覆写 shared 之前读走。
        const uint32_t last = min(blockDim.x, n_blocks - base) - 1u;
        const uint32_t tile_l = sl[last];
        const uint32_t tile_r = sr[last];
        __syncthreads();
        carry_l += tile_l;
        carry_r += tile_r;
        // Avoid a u32 loop-counter wrap for the supported T=1 boundary when
        // n_blocks approaches UINT32_MAX. The final tile is already complete.
        if (n_blocks - base <= blockDim.x) break;
    }

    if (threadIdx.x == 0) {
        totals_out[slot * 2] = carry_l;
        totals_out[slot * 2 + 1] = carry_r;
    }
}

extern "C" __global__ void partition_scan_blocks(
    const uint32_t* counts, uint32_t* left, uint32_t* right,
    uint32_t blocks, uint32_t selected, uint32_t threads,
    uint32_t* totals, uint32_t slot) {
    partition_scan_segment(counts, left, right, blocks, selected, threads, totals, slot);
}

// One fixed-size AoS descriptor per node:
// input offset, length, original-local feature, split bin, missing-left,
// count base, offset base, output offset, level-total slot, cumulative block end.
// All pointers are ordinary tracked allocations; no per-node pointer table.
enum { PART_DESC_WORDS = 10 };

__device__ __forceinline__ uint32_t partition_batch_node(
    const uint32_t* descriptors, uint32_t nodes, uint32_t block) {
    uint32_t lo = 0, hi = nodes;
    while (lo < hi) {
        const uint32_t mid = lo + (hi - lo) / 2;
        if (block < descriptors[mid * PART_DESC_WORDS + 9]) hi = mid;
        else lo = mid + 1;
    }
    return lo;
}

extern "C" __global__ void partition_count_batched(
    const uint8_t* bins, const uint32_t* rows, const uint32_t* descriptors,
    uint32_t nodes, uint32_t n_rows, uint32_t n_feats, uint32_t row_major,
    uint8_t* flags, uint32_t* counts) {
    const uint32_t node = partition_batch_node(descriptors, nodes, blockIdx.x);
    const uint32_t* d = descriptors + node * PART_DESC_WORDS;
    const uint32_t local_block = blockIdx.x - d[5];
    const uint32_t idx = local_block * blockDim.x + threadIdx.x;
    uint32_t left = 0;
    if (idx < d[1]) {
        const uint32_t arena = d[0] + idx;
        const uint32_t row = rows[arena];
        const uint64_t address = row_major ? (uint64_t)row * n_feats + d[2]
                                          : (uint64_t)d[2] * n_rows + row;
        const uint8_t bin = bins[address];
        left = bin == UINT8_MAX ? d[4] : (uint32_t)(bin < d[3]);
        flags[arena] = (uint8_t)left;
    }
    extern __shared__ uint32_t sums[];
    sums[threadIdx.x] = left;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) counts[blockIdx.x] = sums[0];
}

extern "C" __global__ void partition_scan_batched(
    const uint32_t* counts, const uint32_t* descriptors,
    uint32_t row_block_size, uint32_t* left, uint32_t* right, uint32_t* totals) {
    const uint32_t* d = descriptors + blockIdx.x * PART_DESC_WORDS;
    partition_scan_segment(counts + d[5], left + d[6], right + d[6],
                           d[9] - d[5], d[1], row_block_size, totals, d[8]);
}

extern "C" __global__ void partition_scatter_batched(
    const uint32_t* rows, const uint8_t* flags, const uint32_t* descriptors,
    uint32_t nodes, const uint32_t* left, const uint32_t* right, uint32_t* next_rows) {
    const uint32_t node = partition_batch_node(descriptors, nodes, blockIdx.x);
    const uint32_t* d = descriptors + node * PART_DESC_WORDS;
    const uint32_t block = blockIdx.x - d[5];
    const uint32_t idx = block * blockDim.x + threadIdx.x;
    const bool active = idx < d[1];
    const uint32_t is_left = active ? flags[d[0] + idx] : 0;
    extern __shared__ uint32_t scans[];
    uint32_t* ls = scans;
    uint32_t* rs = scans + blockDim.x;
    ls[threadIdx.x] = is_left;
    rs[threadIdx.x] = active ? 1u - is_left : 0;
    __syncthreads();
    for (uint32_t off = 1; off < blockDim.x; off <<= 1) {
        const uint32_t al = threadIdx.x >= off ? ls[threadIdx.x - off] : 0;
        const uint32_t ar = threadIdx.x >= off ? rs[threadIdx.x - off] : 0;
        __syncthreads();
        ls[threadIdx.x] += al;
        rs[threadIdx.x] += ar;
        __syncthreads();
    }
    if (active) {
        const uint32_t total_left = left[d[6] + d[9] - d[5]];
        const uint32_t dst = is_left ? left[d[6] + block] + ls[threadIdx.x] - is_left
            : total_left + right[d[6] + block] + rs[threadIdx.x] - (1u - is_left);
        next_rows[d[7] + dst] = rows[d[0] + idx];
    }
}

// 行主序下的 partition 分类:**直接按 active row 取 split feature**,
// 不再物化整列。
//
// ⚠️ 上一版为了不改 partition,先把整列 gather 进 `part_col`
// (每个 split feature、每层一次,每次扫满 n_rows)。nsys 实测那个
// gather 占 **GPU 总时间的 43.3%(HIGGS 34.5 ms/轮)**,比 histogram
// kernel 本身还大,把行主序省下的 9 ms 吃干净还倒欠 —— 而且它落在所有
// `FB_PROFILE` 计时段之外,表面上完全看不出来。
//
// 这里的代价改成**只随 active row 走**,不再是
// `n_rows × 不同 split feature 数`。
//
// 除了取 bin 的地址算法,其余(missing 路由、flags 语义、block 局部
// 归约)与 `partition_count` **逐字相同**,所以稳定分区的行顺序契约不变。
extern "C" __global__ void partition_count_rm(
    const uint8_t* __restrict__ bins_rm,
    uint32_t n_feats,
    uint32_t feat,
    const uint32_t* rows,
    uint32_t n_selected,
    uint32_t split_bin,
    uint32_t missing_left,
    uint8_t* flags,
    uint32_t* block_left_counts,
    const uint32_t* cat_mask) {
    extern __shared__ uint32_t counts_rm[];
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t left = 0;
    if (idx < n_selected) {
        const uint32_t row = rows[idx];
        const uint8_t bin = bins_rm[(uint64_t)row * n_feats + feat];
        left = partition_goes_left(bin, split_bin, missing_left, cat_mask);
        flags[idx] = (uint8_t)left;
    }
    counts_rm[threadIdx.x] = left;
    __syncthreads();
    for (uint32_t stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            counts_rm[threadIdx.x] += counts_rm[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        block_left_counts[blockIdx.x] = counts_rm[0];
    }
}


extern "C" __global__ void partition_scatter(
    const uint32_t* rows,
    const uint8_t* flags,
    uint32_t n_selected,
    uint32_t* left_offsets,
    uint32_t* right_offsets,
    uint32_t* left_out,
    uint32_t* right_out,
    uint32_t total_left) {
    extern __shared__ uint32_t scans[];
    uint32_t* left_scan = scans;
    uint32_t* right_scan = scans + blockDim.x;
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = idx < n_selected;
    const uint32_t is_left = active ? flags[idx] : 0;
    left_scan[threadIdx.x] = is_left;
    right_scan[threadIdx.x] = active ? (1u - is_left) : 0u;
    __syncthreads();
    for (uint32_t offset = 1; offset < blockDim.x; offset <<= 1) {
        const uint32_t old_left = threadIdx.x >= offset ? left_scan[threadIdx.x - offset] : 0;
        const uint32_t old_right = threadIdx.x >= offset ? right_scan[threadIdx.x - offset] : 0;
        __syncthreads();
        left_scan[threadIdx.x] += old_left;
        right_scan[threadIdx.x] += old_right;
        __syncthreads();
    }
    if (active) {
        const uint32_t block = blockIdx.x;
        const uint32_t local_left = left_scan[threadIdx.x] - is_left;
        const uint32_t local_right = right_scan[threadIdx.x] - (1u - is_left);
        if (is_left) {
            left_out[left_offsets[block] + local_left] = rows[idx];
        } else {
            right_out[total_left + right_offsets[block] + local_right] = rows[idx];
        }
    }
}

// Training-loop variant: both children are written contiguously into one
// next-level arena. The host only needs total_left (one scalar per node) to
// describe the two spans; row ids never leave the device between levels.
extern "C" __global__ void partition_scatter_layer(
    const uint32_t* rows,
    const uint8_t* flags,
    uint32_t n_selected,
    const uint32_t* left_offsets,
    const uint32_t* right_offsets,
    uint32_t* next_rows) {
    // `total_left` 以前是 host 传进来的标量,而 host 为了拿到它必须在
    // scan 之后、scatter 之前做一次**阻塞** D2H —— 每个节点两次,
    // 实测 HIGGS 上 126 次/轮、23.8 ms/轮(其中绝大部分是 pipeline drain,
    // 不是那 1 KB 的传输)。
    //
    // 而 `partition_scan_blocks` 早就把这个值写进了 `left_offsets[n_blocks]`,
    // 而 scatter 的 gridDim.x 就等于 n_blocks。所以 kernel 自己读一下就行,
    // host 那次同步整个可以去掉;totals 改成整层结束后一次性取回。
    const uint32_t total_left = left_offsets[gridDim.x];
    extern __shared__ uint32_t scans[];
    uint32_t* left_scan = scans;
    uint32_t* right_scan = scans + blockDim.x;
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = idx < n_selected;
    const uint32_t is_left = active ? flags[idx] : 0;
    left_scan[threadIdx.x] = is_left;
    right_scan[threadIdx.x] = active ? (1u - is_left) : 0u;
    __syncthreads();
    for (uint32_t offset = 1; offset < blockDim.x; offset <<= 1) {
        const uint32_t old_left = threadIdx.x >= offset ? left_scan[threadIdx.x - offset] : 0;
        const uint32_t old_right = threadIdx.x >= offset ? right_scan[threadIdx.x - offset] : 0;
        __syncthreads();
        left_scan[threadIdx.x] += old_left;
        right_scan[threadIdx.x] += old_right;
        __syncthreads();
    }
    if (active) {
        const uint32_t block = blockIdx.x;
        const uint32_t local_left = left_scan[threadIdx.x] - is_left;
        const uint32_t local_right = right_scan[threadIdx.x] - (1u - is_left);
        const uint32_t dst = is_left
            ? left_offsets[block] + local_left
            : total_left + right_offsets[block] + local_right;
        next_rows[dst] = rows[idx];
    }
}

extern "C" __global__ void assign_leaf_value(
    const uint32_t* rows,
    uint32_t n_selected,
    float value,
    float* prediction_delta) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_selected) {
        prediction_delta[rows[idx]] = value;
    }
}

extern "C" __global__ void init_tree_rows(
    uint32_t n_rows,
    uint32_t* rows,
    float* prediction_delta) {
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_rows) {
        rows[idx] = idx;
        prediction_delta[idx] = 0.0f;
    }
}

// Prediction-only compact SoA traversal. One thread owns one row and walks
// trees in model order, matching Model::predict_margin_upto's accumulation
// order exactly. The host applies the objective link after D2H so CUDA expf
// cannot change prediction semantics. The categorical variant
// (`predict_margin_soa_cat`) lives in its own section at the end of this file.
extern "C" __global__ void predict_margin_soa(
    const float* __restrict__ rows,
    uint32_t n_rows,
    uint32_t n_features,
    const uint32_t* __restrict__ tree_offsets,
    uint32_t n_trees,
    const uint32_t* __restrict__ features,
    const float* __restrict__ split_conditions,
    const uint32_t* __restrict__ left_children,
    const uint32_t* __restrict__ right_children,
    const uint8_t* __restrict__ flags,
    const float* __restrict__ leaf_values,
    const float* __restrict__ init,
    float base_margin,
    float* __restrict__ output) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;

    const float* values = rows + (uint64_t)row * n_features;
    // A per-row start (`base_margin`) replaces the scalar intercept; a null
    // pointer selects the intercept. Either way the tree sum starts from it.
    float margin = init != nullptr ? init[row] : base_margin;
    for (uint32_t tree = 0; tree < n_trees; ++tree) {
        uint32_t node = tree_offsets[tree];
        if (node == tree_offsets[tree + 1]) continue;
        while ((flags[node] & 1u) == 0u) {
            const float value = values[features[node]];
            const bool go_left = value != value
                ? ((flags[node] & 2u) != 0u)
                : value < split_conditions[node];
            node = go_left ? left_children[node] : right_children[node];
        }
        margin += leaf_values[node];
    }
    output[row] = margin;
}

// ---------------------------------------------------------------- learning to rank fast
//
// `gpu_math = "fast"` for rank:ndcg / rank:pairwise / rank:map (LambdaMART).
// Per round, all on the device and on one stream:
//   1. `rank_apply_delta`   pred += delta (every row; sampling never masks predictions)
//   2. `rank_sort_groups`   stable descending sort of every query group's scores
//                           (one block per group, bottom-up merge sort by rank:
//                           ties keep the lower index, so the order is a pure
//                           function of the scores)
//   3. `rank_map_stat`      rank:map only: relevant counts / sum l_k/k along the list
//   4. memset of the fixed-point accumulators
//   5. `rank_pairs`         one thread per pair initiator; every pair adds its lambda
//                           to both rows as a fixed-point INTEGER (atomicAdd on int64).
//                           Integer addition is exact and order independent, so the
//                           result is identical run to run; there are no float atomics.
//   6. `rank_finalize`      fixed point -> f32, lambdarank normalization, group weight,
//                           row sampling -> interleaved f32 gpair (then the usual
//                           max / scale / quantize tail)
// `lambdarank_unbiased` replaces steps 4 to 6 by the bound / scale / accumulate sequence of
// the "unbiased LambdaMART" section at the end of this file.
// Pair enumeration: topk = every (i, j > i) with i < k on the rank list; mean = each
// document samples `num_pair` partners outside its label bucket with a counter based
// hash of (seed, group, document, sample). The pairs, the arithmetic order and the
// fixed-point rounding differ from the host path, which is why this mode is fast-only
// (the host `gpu_math="exact"` path stays byte-identical to the CPU).
#define RANK_SORT_THREADS 256u
#define RANK_PAIR_THREADS 256u

extern "C" __global__ void rank_apply_delta(float* pred, const float* delta, uint32_t n_rows) {
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n_rows;
         i += (uint32_t)blockDim.x * gridDim.x) {
        pred[i] = __fadd_rn(pred[i], delta[i]);
    }
}

// Largest group index g with ptr[g] <= v (ptr has n + 1 non-decreasing entries, v < ptr[n]).
__device__ __forceinline__ uint32_t rank_group_of(const uint32_t* ptr, uint32_t n, uint32_t v) {
    uint32_t lo = 0, hi = n;  // answer in [lo, hi)
    while (hi - lo > 1u) {
        const uint32_t mid = (lo + hi) >> 1;
        if (ptr[mid] <= v) lo = mid; else hi = mid;
    }
    return lo;
}

extern "C" __global__ void rank_sort_groups(
    const float* __restrict__ pred,
    const uint32_t* __restrict__ group_ptr,
    uint32_t n_groups,
    uint32_t* __restrict__ idx_a,
    uint32_t* __restrict__ idx_b,
    uint32_t* __restrict__ rank_pos) {
    for (uint32_t g = blockIdx.x; g < n_groups; g += gridDim.x) {
        const uint32_t base = group_ptr[g];
        const uint32_t cnt = group_ptr[g + 1] - base;
        const float* p = pred + base;
        uint32_t* src = idx_a + base;
        uint32_t* dst = idx_b + base;
        for (uint32_t i = threadIdx.x; i < cnt; i += blockDim.x) src[i] = i;
        __syncthreads();
        for (uint32_t width = 1; width < cnt; width <<= 1) {
            for (uint32_t i = threadIdx.x; i < cnt; i += blockDim.x) {
                const uint32_t lo = (i / (2u * width)) * (2u * width);
                const uint32_t mid = min(lo + width, cnt);
                const uint32_t hi = min(lo + 2u * width, cnt);
                const uint32_t doc = src[i];
                const float key = p[doc];
                uint32_t pos;
                if (i < mid) {
                    // From run A: the elements of B strictly greater go first.
                    uint32_t l = mid, h = hi;
                    while (l < h) {
                        const uint32_t m = (l + h) >> 1;
                        if (p[src[m]] > key) l = m + 1u; else h = m;
                    }
                    pos = lo + (i - lo) + (l - mid);
                } else {
                    // From run B: the elements of A greater or equal go first (stable).
                    uint32_t l = lo, h = mid;
                    while (l < h) {
                        const uint32_t m = (l + h) >> 1;
                        if (p[src[m]] >= key) l = m + 1u; else h = m;
                    }
                    pos = lo + (i - mid) + (l - lo);
                }
                dst[pos] = doc;
            }
            __syncthreads();
            uint32_t* t = src; src = dst; dst = t;
        }
        for (uint32_t i = threadIdx.x; i < cnt; i += blockDim.x) {
            const uint32_t doc = src[i];
            rank_pos[base + doc] = i;
            idx_a[base + i] = doc;
        }
        __syncthreads();
    }
}

// rank:map statistics along the rank list; one thread per group, sequential f64 scan.
extern "C" __global__ void rank_map_stat(
    const float* __restrict__ label,
    const uint32_t* __restrict__ group_ptr,
    uint32_t n_groups,
    const uint32_t* __restrict__ rank_idx,
    double* __restrict__ n_rel,
    double* __restrict__ acc) {
    for (uint32_t g = blockIdx.x * blockDim.x + threadIdx.x; g < n_groups;
         g += (uint32_t)blockDim.x * gridDim.x) {
        const uint32_t base = group_ptr[g];
        const uint32_t cnt = group_ptr[g + 1] - base;
        double nr = 0.0, ac = 0.0;
        for (uint32_t k = 0; k < cnt; ++k) {
            const double y = (double)label[base + rank_idx[base + k]];
            nr += y;
            ac += y / (double)(k + 1u);
            n_rel[base + k] = nr;
            acc[base + k] = ac;
        }
    }
}

// What every pair evaluation yields: the documents (indices inside the group; `dh` is
// the higher-labelled one), the sigmoid of their score difference and the metric weight
// (|delta metric|, already divided by the score difference when that is on).
struct RankPairEval {
    uint32_t dh, dl;
    double sig, dm;
};

// The arithmetic every pair shares (plain and unbiased paths). False: no pair (equal labels).
__device__ __forceinline__ bool rank_pair_eval(
    uint32_t ra, uint32_t rb, uint32_t base, uint32_t cnt,
    const float* __restrict__ pred, const float* __restrict__ label,
    const uint32_t* __restrict__ rank_idx,
    uint32_t kind, uint32_t exp_gain, uint32_t norm_by_diff, bool best_ne_worst,
    double inv_idcg, const double* __restrict__ n_rel, const double* __restrict__ acc,
    RankPairEval* out) {
    const uint32_t da = rank_idx[base + ra];
    const uint32_t db = rank_idx[base + rb];
    const float ya = label[base + da];
    const float yb = label[base + db];
    if (ya == yb) return false;
    uint32_t rh = ra, rl = rb, dh = da, dl = db;
    float yh = ya, yl = yb;
    if (ya < yb) { rh = rb; rl = ra; dh = db; dl = da; yh = yb; yl = ya; }
    const float sh = pred[base + dh];
    const float sl = pred[base + dl];
    const float diff = __fsub_rn(sh, sl);
    const double delta_score = (double)fabsf(diff);
    const float x = fminf(-diff, 88.7f);
    const float denom = __fadd_rn(__fadd_rn(expf(x), 1.0f), 1e-16f);
    const double sig = (double)__fdiv_rn(1.0f, denom);
    double raw;
    if (kind == 1u) {
        raw = 1.0;
    } else if (kind == 0u) {
        const double gh = exp_gain ? (double)((1u << (uint32_t)yh) - 1u) : (double)yh;
        const double gl = exp_gain ? (double)((1u << (uint32_t)yl) - 1u) : (double)yl;
        const double dh_ = 1.0 / log2((double)rh + 2.0);
        const double dl_ = 1.0 / log2((double)rl + 2.0);
        raw = ((gh * dh_ + gl * dl_) - (gl * dh_ + gh * dl_)) * inv_idcg;
    } else {
        uint32_t r1 = rh, r2 = rl;
        float y1 = yh, y2 = yl;
        if (r1 > r2) { const uint32_t t = r1; r1 = r2; r2 = t; const float ty = y1; y1 = y2; y2 = ty; }
        const double r_h = (double)r1 + 1.0;
        const double r_l = (double)r2 + 1.0;
        const double n_total = n_rel[cnt - 1u];
        const double m = n_rel[r2];
        const double n = n_rel[r1];
        const double b = acc[r2 - 1u] - acc[r1];
        if (y1 < y2) {
            raw = ((m / r_l - (n + 1.0) / r_h) - b) / n_total;
        } else {
            raw = ((n / r_h - m / r_l) + b) / n_total;
        }
    }
    double dm = fabs(raw);
    if (norm_by_diff && best_ne_worst) dm = dm / (delta_score + 0.01);
    out->dh = dh;
    out->dl = dl;
    out->sig = sig;
    out->dm = dm;
    return true;
}

// One pair (rank positions ra, rb): the lambda of the higher-labelled document is added to
// it, the negative to the other, both Hessians; all as fixed-point integers.
__device__ __forceinline__ void rank_pair(
    uint32_t ra, uint32_t rb, uint32_t base, uint32_t cnt,
    const float* __restrict__ pred, const float* __restrict__ label,
    const uint32_t* __restrict__ rank_idx,
    uint32_t kind, uint32_t exp_gain, uint32_t norm_by_diff, bool best_ne_worst,
    double inv_idcg, const double* __restrict__ n_rel, const double* __restrict__ acc,
    double scale,
    long long* __restrict__ acc_g, long long* __restrict__ acc_h,
    long long* __restrict__ group_sum) {
    RankPairEval e;
    if (!rank_pair_eval(ra, rb, base, cnt, pred, label, rank_idx, kind, exp_gain, norm_by_diff,
                        best_ne_worst, inv_idcg, n_rel, acc, &e)) return;
    const uint32_t dh = e.dh, dl = e.dl;
    const double sig = e.sig, dm = e.dm;
    const double lambda_ij = (sig - 1.0) * dm;
    const double hessian_ij = fmax(sig * (1.0 - sig), 1e-16) * dm * 2.0;
    const float pg = (float)lambda_ij;
    const float ph = (float)hessian_ij;
    const long long qg = __double2ll_rn((double)pg * scale);
    const long long qh = __double2ll_rn((double)ph * scale);
    const long long qs = __double2ll_rn(-2.0 * (double)pg * scale);
    atomicAdd((unsigned long long*)&acc_g[base + dh], (unsigned long long)qg);
    atomicAdd((unsigned long long*)&acc_g[base + dl], (unsigned long long)(-qg));
    atomicAdd((unsigned long long*)&acc_h[base + dh], (unsigned long long)qh);
    atomicAdd((unsigned long long*)&acc_h[base + dl], (unsigned long long)qh);
    atomicAdd((unsigned long long*)group_sum, (unsigned long long)qs);
}

// The pairs one initiator slot `t` owns (topk: the rank position `t - init_ptr[g]` against
// every lower one; mean: its sampled partners outside the label bucket), as calls
// `op(ra, rb, g, base, cnt, best_ne_worst, inv_idcg, n_rel, acc)` with rank positions.
// Shared by the plain and the unbiased kernels, so both enumerate the same pairs.
template <typename Op>
__device__ __forceinline__ void rank_slot_pairs(
    uint32_t t, const float* __restrict__ pred, const uint32_t* __restrict__ group_ptr,
    const uint32_t* __restrict__ init_ptr, uint32_t n_groups,
    const uint32_t* __restrict__ rank_idx, const uint32_t* __restrict__ rank_pos,
    const double* __restrict__ inv_idcg, const double* __restrict__ n_rel,
    const double* __restrict__ acc, const uint32_t* __restrict__ y_order,
    const uint32_t* __restrict__ bucket_lo, const uint32_t* __restrict__ bucket_hi,
    uint32_t kind, uint32_t is_topk, uint32_t num_pair, unsigned long long seed, Op&& op) {
    const uint32_t g = rank_group_of(init_ptr, n_groups, t);
    const uint32_t s = t - init_ptr[g];
    const uint32_t base = group_ptr[g];
    const uint32_t cnt = group_ptr[g + 1] - base;
    const bool best_ne_worst =
        pred[base + rank_idx[base]] != pred[base + rank_idx[base + cnt - 1u]];
    const double inv = (kind == 0u) ? inv_idcg[g] : 0.0;
    const double* nr = (kind == 2u) ? n_rel + base : n_rel;
    const double* ac = (kind == 2u) ? acc + base : acc;
    if (is_topk) {
        for (uint32_t j = s + 1u; j < cnt; ++j) {
            op(s, j, g, base, cnt, best_ne_worst, inv, nr, ac);
        }
    } else {
        const uint32_t doc = y_order[base + s];
        const uint32_t lo = bucket_lo[base + s];
        const uint32_t hi = bucket_hi[base + s];
        const uint32_t n_lefts = lo;
        const uint32_t n_rights = cnt - hi;
        if (n_lefts + n_rights == 0u) return;
        for (uint32_t q = 0; q < num_pair; ++q) {
            const unsigned long long h = fb_mix(
                seed ^ ((unsigned long long)g * 0x9E3779B97F4A7C15ULL)
                ^ ((unsigned long long)s * 0xD6E8FEB86659FD93ULL)
                ^ ((unsigned long long)(q + 1u) * 0xBF58476D1CE4E5B9ULL));
            uint32_t r = (uint32_t)(h % (unsigned long long)(n_lefts + n_rights));
            if (r >= n_lefts) r = r - lo + hi;
            const uint32_t pdoc = y_order[base + r];
            op(rank_pos[base + doc], rank_pos[base + pdoc], g, base, cnt, best_ne_worst, inv,
               nr, ac);
        }
    }
}

extern "C" __global__ void rank_pairs(
    const float* __restrict__ pred,
    const float* __restrict__ label,
    const uint32_t* __restrict__ group_ptr,
    const uint32_t* __restrict__ init_ptr,
    uint32_t n_groups,
    uint32_t total_slots,
    const uint32_t* __restrict__ rank_idx,
    const uint32_t* __restrict__ rank_pos,
    const double* __restrict__ inv_idcg,
    const double* __restrict__ n_rel,
    const double* __restrict__ acc,
    const uint32_t* __restrict__ y_order,
    const uint32_t* __restrict__ bucket_lo,
    const uint32_t* __restrict__ bucket_hi,
    uint32_t kind, uint32_t is_topk, uint32_t num_pair, uint32_t exp_gain, uint32_t norm_by_diff,
    unsigned long long seed, double scale,
    long long* __restrict__ acc_g, long long* __restrict__ acc_h,
    long long* __restrict__ group_sum) {
    for (uint32_t t = blockIdx.x * blockDim.x + threadIdx.x; t < total_slots;
         t += (uint32_t)blockDim.x * gridDim.x) {
        rank_slot_pairs(t, pred, group_ptr, init_ptr, n_groups, rank_idx, rank_pos, inv_idcg,
                        n_rel, acc, y_order, bucket_lo, bucket_hi, kind, is_topk, num_pair, seed,
                        [&](uint32_t ra, uint32_t rb, uint32_t g, uint32_t base, uint32_t cnt,
                            bool best_ne_worst, double inv, const double* nr, const double* ac) {
                            rank_pair(ra, rb, base, cnt, pred, label, rank_idx, kind, exp_gain,
                                      norm_by_diff, best_ne_worst, inv, nr, ac, scale, acc_g,
                                      acc_h, group_sum + g);
                        });
    }
}

// Fixed point -> f32 gpair of row `i` (normalization, group weight, row sampling).
// Returns whether both values are finite (always true for sane inputs).
__device__ __forceinline__ bool rank_finalize_row(
    uint32_t i, uint32_t g,
    const long long* __restrict__ acc_g,
    const long long* __restrict__ acc_h,
    const long long* __restrict__ group_sum,
    const float* __restrict__ group_weight,
    uint32_t has_group_weight,
    float weight_norm,
    uint32_t is_topk, uint32_t num_pair, uint32_t normalization,
    double scale,
    float* __restrict__ gpair_out,
    unsigned long long subsample_threshold,
    unsigned long long subsample_seed,
    unsigned long long subsample_tree) {
    float gr = (float)((double)acc_g[i] / scale);
    float he = (float)((double)acc_h[i] / scale);
    if (normalization) {
        double norm = 1.0;
        if (!is_topk) {
            norm = 1.0 / (double)num_pair;
        } else {
            const double sum = (double)group_sum[g] / scale;
            if (sum > 0.0) norm = log2(1.0 + sum) / sum;
        }
        if (norm != 1.0) {
            const float n = (float)norm;
            gr = __fmul_rn(gr, n);
            he = __fmul_rn(he, n);
        }
    }
    const float w = has_group_weight ? group_weight[g] : 1.0f;
    gr = __fmul_rn(__fmul_rn(gr, w), weight_norm);
    he = __fmul_rn(__fmul_rn(he, w), weight_norm);
    const bool finite = isfinite(gr) && isfinite(he);
    if (subsample_threshold != ~0ULL) {
        const unsigned long long hsh = fb_mix(
            subsample_seed
            ^ ((subsample_tree * 0x9E3779B97F4A7C15ULL) << 23
               | (subsample_tree * 0x9E3779B97F4A7C15ULL) >> 41)
            ^ ((unsigned long long)i * 0xD6E8FEB86659FD93ULL));
        if ((hsh >> 11) >= subsample_threshold) {
            gr = 0.0f;
            he = 0.0f;
        }
    }
    gpair_out[(uint64_t)i * 2] = gr;
    gpair_out[(uint64_t)i * 2 + 1] = he;
    return finite;
}

extern "C" __global__ void rank_finalize(
    const uint32_t* __restrict__ group_ptr,
    uint32_t n_groups,
    uint32_t n_rows,
    const long long* __restrict__ acc_g,
    const long long* __restrict__ acc_h,
    const long long* __restrict__ group_sum,
    const float* __restrict__ group_weight,
    uint32_t has_group_weight,
    float weight_norm,
    uint32_t is_topk, uint32_t num_pair, uint32_t normalization,
    double scale,
    float* __restrict__ gpair_out,
    unsigned long long subsample_threshold,
    unsigned long long subsample_seed,
    unsigned long long subsample_tree) {
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n_rows;
         i += (uint32_t)blockDim.x * gridDim.x) {
        const uint32_t g = rank_group_of(group_ptr, n_groups, i);
        rank_finalize_row(i, g, acc_g, acc_h, group_sum, group_weight, has_group_weight,
                          weight_norm, is_topk, num_pair, normalization, scale, gpair_out,
                          subsample_threshold, subsample_seed, subsample_tree);
    }
}

// ---------------------------------------------------------------- unbiased LambdaMART
//
// `lambdarank_unbiased` on the device (XGBoost semantics: pair lambdas / hessians divided
// by t+(i) t-(j), per-position costs li / lj accumulated from the same pairs, biases
// re-estimated after the gradients of every round).
//
// The scaled lambdas are unbounded (t+ and t- go down to 1e-16, so 1 / (t+ t-) reaches
// 1e32), which no fixed scale chosen up front can hold. Each round is therefore two passes
// over the same pairs:
//   `rank_pairs_ub` mode 0  bound: the exact maximum of every quantity that will be added,
//                           as an order independent integer max of the double bit patterns
//   `rank_bias_scales`      one power of two per quantity group such that (maximum x
//                           number of terms that can reach one accumulator) < 2^61
//   `rank_pairs_ub` mode 1  accumulate the fixed-point integers (exact, order independent)
//   `rank_finalize_ub`      like `rank_finalize`, but with the device scale
//   `rank_bias_update`      t+ / t- from the integer cost sums
// A non finite maximum or gradient (diverged scores) sets a status word that the host
// turns into an error; nothing saturates silently. There are no float atomics, so a run
// is bit-reproducible.
//
// The cost of a pair, -log(1 - sigmoid) x metric weight, is infinite once the f32 sigmoid
// rounds to 1 (score gap above ~16.6); with it XGBoost's estimate is NaN. Here 1 - sigmoid
// is floored at its smallest nonzero f32 value (2^-24), the cost of the most separated pair
// the f32 sigmoid can tell apart.
#define RANK_BIAS_REPLICAS 64u
#define RANK_BIAS_CTL_WORDS 8u
#define RANK_CTL_MAX_GH 0u
#define RANK_CTL_MAX_COST 1u
#define RANK_CTL_SCALE_GH 2u
#define RANK_CTL_SCALE_COST 3u
#define RANK_CTL_STATUS 4u
#define RANK_STATUS_BOUND 1ULL
#define RANK_STATUS_GRADIENT 2ULL
#define RANK_BIAS_EPS 1e-16

// Order preserving integer key of a non-negative double; NaN sorts above infinity.
__device__ __forceinline__ unsigned long long rank_key(double v) {
    return (v != v) ? 0x7FF8000000000000ULL : (unsigned long long)__double_as_longlong(fabs(v));
}

__device__ __forceinline__ double rank_pow2_scale(double v) {
    if (!(v > 0.0)) return 1.0;
    int e = (int)floor(61.0 - log2(v));
    e = max(-1000, min(1000, e));
    return ldexp(1.0, e);
}

// One pair of the unbiased path. mode 0: fold the magnitudes into (m_gh, m_cost).
// mode 1: add the fixed-point contributions; `repl` is this block's replica of the cost sums.
__device__ __forceinline__ void rank_pair_ub(
    const RankPairEval& e, uint32_t mode,
    const double* __restrict__ ti, const double* __restrict__ tj, uint32_t n_pos,
    double scale_gh, double scale_cost,
    long long* __restrict__ acc_g, long long* __restrict__ acc_h,
    long long* __restrict__ group_sum, uint32_t base,
    long long* __restrict__ repl,
    unsigned long long* m_gh, unsigned long long* m_cost) {
    const double sig = e.sig, dm = e.dm;
    // 1 - sigmoid is 0 or >= 2^-24 (f32): floor it, see the section comment
    const double one_minus = fmax(1.0 - sig, 5.9604644775390625e-08);
    const double cost = log(1.0 / one_minus) * dm;
    double lambda_ij = (sig - 1.0) * dm;
    double hessian_ij = fmax(sig * (1.0 - sig), 1e-16) * dm * 2.0;
    const bool tracked = e.dh < n_pos && e.dl < n_pos;
    const double tiv = tracked ? ti[e.dh] : 0.0;
    const double tjv = tracked ? tj[e.dl] : 0.0;
    if (tracked && tjv >= RANK_BIAS_EPS && tiv >= RANK_BIAS_EPS) {
        lambda_ij /= tiv * tjv;
        hessian_ij /= tiv * tjv;
    }
    const float pg = (float)lambda_ij;
    const float ph = (float)hessian_ij;
    const double li_c = (tracked && tjv >= RANK_BIAS_EPS) ? cost / tjv : 0.0;
    const double lj_c = (tracked && tiv >= RANK_BIAS_EPS) ? cost / tiv : 0.0;
    if (mode == 0u) {
        const unsigned long long k1 = rank_key(2.0 * (double)pg);
        const unsigned long long k2 = rank_key((double)ph);
        const unsigned long long k3 = rank_key(li_c);
        const unsigned long long k4 = rank_key(lj_c);
        const unsigned long long kg = k1 > k2 ? k1 : k2;
        const unsigned long long kc = k3 > k4 ? k3 : k4;
        if (kg > *m_gh) *m_gh = kg;
        if (kc > *m_cost) *m_cost = kc;
        return;
    }
    const long long qg = __double2ll_rn((double)pg * scale_gh);
    const long long qh = __double2ll_rn((double)ph * scale_gh);
    const long long qs = __double2ll_rn(-2.0 * (double)pg * scale_gh);
    atomicAdd((unsigned long long*)&acc_g[base + e.dh], (unsigned long long)qg);
    atomicAdd((unsigned long long*)&acc_g[base + e.dl], (unsigned long long)(-qg));
    atomicAdd((unsigned long long*)&acc_h[base + e.dh], (unsigned long long)qh);
    atomicAdd((unsigned long long*)&acc_h[base + e.dl], (unsigned long long)qh);
    atomicAdd((unsigned long long*)group_sum, (unsigned long long)qs);
    if (li_c != 0.0) {
        atomicAdd((unsigned long long*)&repl[e.dh],
                  (unsigned long long)__double2ll_rn(li_c * scale_cost));
    }
    if (lj_c != 0.0) {
        atomicAdd((unsigned long long*)&repl[n_pos + e.dl],
                  (unsigned long long)__double2ll_rn(lj_c * scale_cost));
    }
}

extern "C" __global__ void rank_pairs_ub(
    const float* __restrict__ pred,
    const float* __restrict__ label,
    const uint32_t* __restrict__ group_ptr,
    const uint32_t* __restrict__ init_ptr,
    uint32_t n_groups,
    uint32_t total_slots,
    const uint32_t* __restrict__ rank_idx,
    const uint32_t* __restrict__ rank_pos,
    const double* __restrict__ inv_idcg,
    const double* __restrict__ n_rel,
    const double* __restrict__ acc,
    const uint32_t* __restrict__ y_order,
    const uint32_t* __restrict__ bucket_lo,
    const uint32_t* __restrict__ bucket_hi,
    uint32_t kind, uint32_t is_topk, uint32_t num_pair, uint32_t exp_gain, uint32_t norm_by_diff,
    unsigned long long seed,
    const double* __restrict__ ti, const double* __restrict__ tj, uint32_t n_pos,
    uint32_t mode,
    unsigned long long* __restrict__ ctl,
    long long* __restrict__ repl_all,
    long long* __restrict__ acc_g, long long* __restrict__ acc_h,
    long long* __restrict__ group_sum) {
    const double scale_gh = __longlong_as_double((long long)ctl[RANK_CTL_SCALE_GH]);
    const double scale_cost = __longlong_as_double((long long)ctl[RANK_CTL_SCALE_COST]);
    long long* repl = repl_all + (size_t)(blockIdx.x & (RANK_BIAS_REPLICAS - 1u)) * 2u * n_pos;
    unsigned long long m_gh = 0ULL, m_cost = 0ULL;
    for (uint32_t t = blockIdx.x * blockDim.x + threadIdx.x; t < total_slots;
         t += (uint32_t)blockDim.x * gridDim.x) {
        rank_slot_pairs(t, pred, group_ptr, init_ptr, n_groups, rank_idx, rank_pos, inv_idcg,
                        n_rel, acc, y_order, bucket_lo, bucket_hi, kind, is_topk, num_pair, seed,
                        [&](uint32_t ra, uint32_t rb, uint32_t g, uint32_t base, uint32_t cnt,
                            bool best_ne_worst, double inv, const double* nr, const double* ac) {
                            RankPairEval e;
                            if (!rank_pair_eval(ra, rb, base, cnt, pred, label, rank_idx, kind,
                                                exp_gain, norm_by_diff, best_ne_worst, inv, nr,
                                                ac, &e)) return;
                            rank_pair_ub(e, mode, ti, tj, n_pos, scale_gh, scale_cost, acc_g,
                                         acc_h, group_sum + g, base, repl, &m_gh, &m_cost);
                        });
    }
    if (mode == 0u) {
        // warp max, then one atomic max per warp (integer max: order independent)
        for (int d = 16; d > 0; d >>= 1) {
            const unsigned long long a = __shfl_down_sync(0xffffffffu, m_gh, d);
            const unsigned long long b = __shfl_down_sync(0xffffffffu, m_cost, d);
            if (a > m_gh) m_gh = a;
            if (b > m_cost) m_cost = b;
        }
        if ((threadIdx.x & 31u) == 0u) {
            if (m_gh != 0ULL) atomicMax(&ctl[RANK_CTL_MAX_GH], m_gh);
            if (m_cost != 0ULL) atomicMax(&ctl[RANK_CTL_MAX_COST], m_cost);
        }
    }
}

// One thread: the fixed-point scales of this round from the bounds.
// `bound_row`: most terms that can reach one row accumulator or a group sum;
// `bound_total`: terms that can reach one position cost sum (all groups).
extern "C" __global__ void rank_bias_scales(
    unsigned long long* __restrict__ ctl, double bound_row, double bound_total) {
    if (blockIdx.x != 0u || threadIdx.x != 0u) return;
    const unsigned long long inf_key = 0x7FF0000000000000ULL;
    const unsigned long long kg = ctl[RANK_CTL_MAX_GH];
    const unsigned long long kc = ctl[RANK_CTL_MAX_COST];
    double sg = 1.0, sc = 1.0;
    if (kg >= inf_key || kc >= inf_key) {
        ctl[RANK_CTL_STATUS] |= RANK_STATUS_BOUND;
    } else {
        sg = rank_pow2_scale(__longlong_as_double((long long)kg) * bound_row);
        sc = rank_pow2_scale(__longlong_as_double((long long)kc) * bound_total);
    }
    ctl[RANK_CTL_SCALE_GH] = (unsigned long long)__double_as_longlong(sg);
    ctl[RANK_CTL_SCALE_COST] = (unsigned long long)__double_as_longlong(sc);
}

extern "C" __global__ void rank_finalize_ub(
    const uint32_t* __restrict__ group_ptr,
    uint32_t n_groups,
    uint32_t n_rows,
    const long long* __restrict__ acc_g,
    const long long* __restrict__ acc_h,
    const long long* __restrict__ group_sum,
    const float* __restrict__ group_weight,
    uint32_t has_group_weight,
    float weight_norm,
    uint32_t is_topk, uint32_t num_pair, uint32_t normalization,
    unsigned long long* __restrict__ ctl,
    float* __restrict__ gpair_out,
    unsigned long long subsample_threshold,
    unsigned long long subsample_seed,
    unsigned long long subsample_tree) {
    const double scale = __longlong_as_double((long long)ctl[RANK_CTL_SCALE_GH]);
    bool ok = true;
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n_rows;
         i += (uint32_t)blockDim.x * gridDim.x) {
        const uint32_t g = rank_group_of(group_ptr, n_groups, i);
        ok = rank_finalize_row(i, g, acc_g, acc_h, group_sum, group_weight, has_group_weight,
                               weight_norm, is_topk, num_pair, normalization, scale, gpair_out,
                               subsample_threshold, subsample_seed, subsample_tree) && ok;
    }
    if (!ok) atomicOr(&ctl[RANK_CTL_STATUS], RANK_STATUS_GRADIENT);
}

// t+(i) = (li(i) / li(0))^regularizer from the integer cost sums of all replicas (exact),
// the same guards as XGBoost: l(0) must not be tiny. One block.
extern "C" __global__ void rank_bias_update(
    double* __restrict__ ti, double* __restrict__ tj,
    const long long* __restrict__ repl_all, const unsigned long long* __restrict__ ctl,
    uint32_t n_pos, double regularizer) {
    const double sc = __longlong_as_double((long long)ctl[RANK_CTL_SCALE_COST]);
    for (uint32_t k = 0; k < 2u; ++k) {
        double* t = k == 0u ? ti : tj;
        unsigned long long z = 0ULL;
        for (uint32_t r = 0; r < RANK_BIAS_REPLICAS; ++r) {
            z += (unsigned long long)repl_all[((size_t)r * 2u + k) * n_pos];
        }
        const double l0 = (double)(long long)z / sc;
        if (!(l0 >= RANK_BIAS_EPS)) continue;
        for (uint32_t i = threadIdx.x; i < n_pos; i += blockDim.x) {
            unsigned long long a = 0ULL;
            for (uint32_t r = 0; r < RANK_BIAS_REPLICAS; ++r) {
                a += (unsigned long long)repl_all[((size_t)r * 2u + k) * n_pos + i];
            }
            t[i] = pow(((double)(long long)a / sc) / l0, regularizer);
        }
    }
}

// ================================================================ SHAP explanations (shx_*)
//
// `Model.predict(pred_contribs / pred_interactions / pred_leaf, use_gpu=True)`.
//
// The device runs the same fixed 8-point QuadratureTreeSHAP as `src/explain.rs`
// and replays its float32 operation order exactly: one thread owns one
// (row, output group) pair and walks the group's trees in model order, so
// nothing is reduced across threads and no float atomics exist. Every float
// operation is a round-to-nearest intrinsic (no FMA contraction), which makes
// the result bit-identical to the host path.
//
// The tree walk is the host recursion made iterative. SHAP visits every node
// whatever the row, so all threads of a warp execute the same node sequence
// (only the values differ). The per-level frames live in a global scratch laid
// out structure-of-arrays over threads (word w of thread t at w * stride + t),
// which keeps those accesses coalesced; the host sizes it from the deepest tree
// of the requested prefix, so there is no fixed depth limit.
//
// Per node (precomputed on the host, flattened over all trees):
//   child_weight  branch_weight(cover, parent cover) of the edge into the node
//   prev_same     levels up to the nearest ancestor splitting on the same
//                 feature (0 = none); that ancestor's edge probability is the
//                 host's `path_prob[feat]`
//   slot          position of the split feature in the tree's list of distinct
//                 features, where the per-tree contribution accumulates (the
//                 host's `tree_contribs`, summed per tree and then added to the
//                 row so the rounding matches)
// Interactions accumulate straight into the output matrix like the host; two
// small kernels then symmetrise it and turn the diagonal into main effects.
#define SHX_POINTS 8
#define SHX_UNSEEN (-999.0f)
// Frame layout (words per level).
#define SHX_NODE 0u
#define SHX_PHASE 1u
#define SHX_EFEAT 2u
#define SHX_W 3u
#define SHX_PE 4u
#define SHX_POLD 5u
#define SHX_C 6u
#define SHX_H (SHX_C + SHX_POINTS)
#define SHX_FRAME_WORDS (SHX_H + SHX_POINTS)

#define SHX_FMUL(a, b) __fmul_rn((a), (b))
#define SHX_FADD(a, b) __fadd_rn((a), (b))
#define SHX_FSUB(a, b) __fsub_rn((a), (b))
#define SHX_FDIV(a, b) __fdiv_rn((a), (b))

#define SHX_MODEL_PARAMS \
    const uint32_t* __restrict__ tree_node_begin, \
    const uint32_t* __restrict__ node_feature, \
    const float* __restrict__ node_split, \
    const uint32_t* __restrict__ node_left, \
    const uint32_t* __restrict__ node_right, \
    const uint8_t* __restrict__ node_flags, \
    const float* __restrict__ node_leaf, \
    const float* __restrict__ node_child_weight, \
    const uint32_t* __restrict__ node_prev_same, \
    const uint32_t* __restrict__ node_slot, \
    const uint32_t* __restrict__ node_cat_begin, \
    const uint32_t* __restrict__ node_cat_len, \
    const uint32_t* __restrict__ cat_words, \
    const uint32_t* __restrict__ tree_uniq_begin, \
    const uint32_t* __restrict__ uniq_feats, \
    const float* __restrict__ node_mean

struct ShxModel {
    const uint32_t* tree_node_begin;
    const uint32_t* feature;
    const float* split;
    const uint32_t* left;
    const uint32_t* right;
    const uint8_t* flags;
    const float* leaf;
    const float* child_weight;
    const uint32_t* prev_same;
    const uint32_t* slot;
    const uint32_t* cat_begin;
    const uint32_t* cat_len;
    const uint32_t* cat_words;
    const uint32_t* uniq_begin;
    const uint32_t* uniq_feats;
    const float* mean;
};

#define SHX_MODEL_INIT \
    ShxModel m{tree_node_begin, node_feature, node_split, node_left, node_right, node_flags, \
               node_leaf, node_child_weight, node_prev_same, node_slot, node_cat_begin, \
               node_cat_len, cat_words, tree_uniq_begin, uniq_feats, node_mean}

// Node::goes_left: missing follows default_left, a categorical node tests its
// right-going set (everything else, including out-of-range codes, goes left),
// a numeric node compares `v < split_cond`.
__device__ __forceinline__ bool shx_goes_left(const ShxModel& m, uint32_t nd, float v) {
    const uint8_t fl = m.flags[nd];
    if (v != v) return (fl & 2u) != 0u;
    if ((fl & 4u) != 0u) {
        if (v < 0.0f || v >= 16777216.0f) return true;
        const uint32_t cat = (uint32_t)v;
        const uint32_t word = cat >> 5;
        if (word >= m.cat_len[nd]) return true;
        return ((m.cat_words[m.cat_begin[nd] + word] >> (cat & 31u)) & 1u) == 0u;
    }
    return v < m.split[nd];
}

// Scratch word accessors. `su` / `sf` alias one buffer; `stride` is the number
// of threads of the launch and `sid` this thread's slot.
#define SHX_SU(w) su[(uint64_t)(w) * stride + sid]
#define SHX_SF(w) sf[(uint64_t)(w) * stride + sid]
#define SHX_FU(d, f) SHX_SU((d) * SHX_FRAME_WORDS + (f))
#define SHX_FF(d, f) SHX_SF((d) * SHX_FRAME_WORDS + (f))

__device__ __forceinline__ float shx_extract_delta(const float* rn, const float* h,
                                                   float p_enter, float p_exit) {
    float acc = 0.0f;
    if (p_enter != 1.0f) {
        const float alpha = SHX_FSUB(p_enter, 1.0f);
#pragma unroll
        for (int i = 0; i < SHX_POINTS; ++i) {
            acc = SHX_FADD(acc, SHX_FDIV(SHX_FMUL(alpha, h[i]),
                                         SHX_FADD(1.0f, SHX_FMUL(alpha, rn[i]))));
        }
    }
    if (p_exit != 1.0f) {
        const float alpha = SHX_FSUB(p_exit, 1.0f);
#pragma unroll
        for (int i = 0; i < SHX_POINTS; ++i) {
            acc = SHX_FSUB(acc, SHX_FDIV(SHX_FMUL(alpha, h[i]),
                                         SHX_FADD(1.0f, SHX_FMUL(alpha, rn[i]))));
        }
    }
    return acc;
}

__device__ __forceinline__ float shx_interaction_delta(const float* rn, const float* h,
                                                       float p_enter, float p_exit,
                                                       float q_partner) {
    if (q_partner == 1.0f) return 0.0f;
    const float alpha_partner = SHX_FSUB(q_partner, 1.0f);
    const float alpha_enter = SHX_FSUB(p_enter, 1.0f);
    float acc = 0.0f;
    if (p_exit == 1.0f) {
#pragma unroll
        for (int i = 0; i < SHX_POINTS; ++i) {
            const float edge = SHX_FDIV(alpha_enter, SHX_FADD(1.0f, SHX_FMUL(alpha_enter, rn[i])));
            acc = SHX_FADD(acc, SHX_FDIV(SHX_FMUL(SHX_FMUL(alpha_partner, h[i]), edge),
                                         SHX_FADD(1.0f, SHX_FMUL(alpha_partner, rn[i]))));
        }
    } else {
        const float alpha_exit = SHX_FSUB(p_exit, 1.0f);
#pragma unroll
        for (int i = 0; i < SHX_POINTS; ++i) {
            const float edge = SHX_FSUB(
                SHX_FDIV(alpha_enter, SHX_FADD(1.0f, SHX_FMUL(alpha_enter, rn[i]))),
                SHX_FDIV(alpha_exit, SHX_FADD(1.0f, SHX_FMUL(alpha_exit, rn[i]))));
            acc = SHX_FADD(acc, SHX_FDIV(SHX_FMUL(SHX_FMUL(alpha_partner, h[i]), edge),
                                         SHX_FADD(1.0f, SHX_FMUL(alpha_partner, rn[i]))));
        }
    }
    return acc;
}

// Set up the frame of one child of the node `nd` at level `p`: the host's
// `visit_child` up to the recursive call.
__device__ __forceinline__ void shx_descend(const ShxModel& m, const float* feats, const float* rn,
                                            uint32_t p, uint32_t nd, bool right,
                                            uint32_t* su, float* sf, uint64_t stride,
                                            uint64_t sid) {
    const uint32_t feat = m.feature[nd];
    const uint32_t child = right ? m.right[nd] : m.left[nd];
    const float cw = m.child_weight[child];
    const bool gl = shx_goes_left(m, nd, feats[feat]);
    const bool satisfies = right ? !gl : gl;
    const uint32_t off = m.prev_same[nd];
    const float p_old = off != 0u ? SHX_FF(p - off + 1u, SHX_PE) : SHX_UNSEEN;
    const bool seen = p_old != SHX_UNSEEN;
    float p_e = 0.0f;
    if (satisfies) p_e = seen ? SHX_FDIV(p_old, cw) : SHX_FDIV(1.0f, cw);
    const float alpha_e = SHX_FSUB(p_e, 1.0f);
    const float alpha_old = seen ? SHX_FSUB(p_old, 1.0f) : 0.0f;
#pragma unroll
    for (int i = 0; i < SHX_POINTS; ++i) {
        float c = SHX_FF(p, SHX_C + i);
        c = SHX_FMUL(c, SHX_FADD(1.0f, SHX_FMUL(alpha_e, rn[i])));
        if (seen && alpha_old != 0.0f) c = SHX_FDIV(c, SHX_FADD(1.0f, SHX_FMUL(alpha_old, rn[i])));
        SHX_FF(p + 1u, SHX_C + i) = c;
    }
    SHX_FU(p + 1u, SHX_NODE) = child;
    SHX_FU(p + 1u, SHX_EFEAT) = feat;
    SHX_FF(p + 1u, SHX_W) = SHX_FMUL(SHX_FF(p, SHX_W), cw);
    SHX_FF(p + 1u, SHX_PE) = p_e;
    SHX_FF(p + 1u, SHX_POLD) = p_old;
}

// One tree for one row. MODE 0 accumulates per-tree contributions into the
// scratch slots starting at word `tc_base`; MODE 1 adds the diagonal and the
// path-partner deltas straight into `mat` (`nc` x `nc` floats of this row/group).
template <int MODE>
__device__ void shx_run_tree(const ShxModel& m, const float* feats, const float* rn,
                             const float* rw, uint32_t root, uint32_t* su, float* sf,
                             uint64_t stride, uint64_t sid, uint32_t tc_base, float* mat,
                             uint32_t nc) {
    uint32_t d = 0;
    SHX_FU(0, SHX_NODE) = root;
    SHX_FF(0, SHX_W) = 1.0f;
#pragma unroll
    for (int i = 0; i < SHX_POINTS; ++i) SHX_FF(0, SHX_C + i) = 1.0f;
    while (true) {
        const uint32_t nd = SHX_FU(d, SHX_NODE);
        if ((m.flags[nd] & 1u) != 0u) {
            const float leaf_scale = SHX_FMUL(SHX_FF(d, SHX_W), m.leaf[nd]);
#pragma unroll
            for (int i = 0; i < SHX_POINTS; ++i) {
                SHX_FF(d, SHX_H + i) = SHX_FMUL(SHX_FMUL(SHX_FF(d, SHX_C + i), leaf_scale), rw[i]);
            }
        } else {
            SHX_FU(d, SHX_PHASE) = 0u;
            shx_descend(m, feats, rn, d, nd, false, su, sf, stride, sid);
            ++d;
            continue;
        }
        // The node at level d has its result in H; unwind finished edges
        // until a right sibling still has to be visited.
        while (true) {
            if (d == 0u) return;
            const uint32_t p = d - 1u;
            const uint32_t pn = SHX_FU(p, SHX_NODE);
            const uint32_t feat = m.feature[pn];
            const float p_enter = SHX_FF(d, SHX_PE);
            const float p_old = SHX_FF(d, SHX_POLD);
            const float p_exit = p_old == SHX_UNSEEN ? 1.0f : p_old;
            float h[SHX_POINTS];
#pragma unroll
            for (int i = 0; i < SHX_POINTS; ++i) h[i] = SHX_FF(d, SHX_H + i);
            const float delta = shx_extract_delta(rn, h, p_enter, p_exit);
            if (MODE == 0) {
                const uint32_t w = tc_base + m.slot[pn];
                SHX_SF(w) = SHX_FADD(SHX_SF(w), delta);
            } else {
                float* diag = mat + (uint64_t)feat * nc + feat;
                *diag = SHX_FADD(*diag, delta);
                // Partners: the newest occurrence of every other feature on the
                // live path, newest to oldest; older duplicates are shadowed.
                bool skipped_current = false;
                for (uint32_t j = d; j >= 1u; --j) {
                    const uint32_t split = SHX_FU(j, SHX_EFEAT);
                    bool shadowed = false;
                    for (uint32_t jj = j + 1u; jj <= d; ++jj) {
                        if (SHX_FU(jj, SHX_EFEAT) == split) {
                            shadowed = true;
                            break;
                        }
                    }
                    if (shadowed) continue;
                    if (!skipped_current && split == feat) {
                        skipped_current = true;
                        continue;
                    }
                    const float dp = shx_interaction_delta(rn, h, p_enter, p_exit,
                                                           SHX_FF(j, SHX_PE));
                    if (dp != 0.0f) {
                        float* cell = mat + (uint64_t)feat * nc + split;
                        *cell = SHX_FADD(*cell, dp);
                    }
                }
            }
            if (SHX_FU(p, SHX_PHASE) == 0u) {
#pragma unroll
                for (int i = 0; i < SHX_POINTS; ++i) SHX_FF(p, SHX_H + i) = h[i];
                SHX_FU(p, SHX_PHASE) = 1u;
                shx_descend(m, feats, rn, p, pn, true, su, sf, stride, sid);
                break;
            }
#pragma unroll
            for (int i = 0; i < SHX_POINTS; ++i) {
                SHX_FF(p, SHX_H + i) = SHX_FADD(SHX_FF(p, SHX_H + i), h[i]);
            }
            d = p;
        }
    }
}

// Exact contributions: out[(row * k + g) * nc + f], zeroed by the host.
extern "C" __global__ void shx_contribs(
    SHX_MODEL_PARAMS,
    const float* __restrict__ rule, const float* __restrict__ rows, uint32_t n_rows,
    uint32_t n_features, uint32_t n_trees, uint32_t k, uint32_t levels,
    const float* __restrict__ group_root_mean, const float* __restrict__ group_bias,
    const float* __restrict__ margin, float* __restrict__ scratch, float* __restrict__ out) {
    SHX_MODEL_INIT;
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;
    const uint32_t g = blockIdx.y;
    uint32_t* su = reinterpret_cast<uint32_t*>(scratch);
    float* sf = scratch;
    const uint64_t stride = (uint64_t)n_rows * k;
    const uint64_t sid = (uint64_t)g * n_rows + row;
    const uint32_t nc = n_features + 1u;
    const uint32_t tc_base = levels * SHX_FRAME_WORDS;
    const float* feats = rows + (uint64_t)row * n_features;
    float* p = out + ((uint64_t)row * k + g) * nc;
    float rn[SHX_POINTS], rw[SHX_POINTS];
#pragma unroll
    for (int i = 0; i < SHX_POINTS; ++i) {
        rn[i] = rule[i];
        rw[i] = rule[SHX_POINTS + i];
    }
    for (uint32_t t = g; t < n_trees; t += k) {
        const uint32_t root = m.tree_node_begin[t];
        if ((m.flags[root] & 1u) != 0u) continue;
        const uint32_t ub = m.uniq_begin[t];
        const uint32_t un = m.uniq_begin[t + 1u] - ub;
        for (uint32_t u = 0; u < un; ++u) SHX_SF(tc_base + u) = 0.0f;
        shx_run_tree<0>(m, feats, rn, rw, root, su, sf, stride, sid, tc_base, nullptr, nc);
        for (uint32_t u = 0; u < un; ++u) {
            const uint32_t f = m.uniq_feats[ub + u];
            p[f] = SHX_FADD(p[f], SHX_SF(tc_base + u));
        }
    }
    p[nc - 1u] = SHX_FADD(p[nc - 1u], group_root_mean[g]);
    const float bias = margin != nullptr ? margin[(uint64_t)row * k + g] : group_bias[g];
    p[nc - 1u] = SHX_FADD(p[nc - 1u], bias);
}

// Saabas (approx_contribs): the host's `saabas_row`, same slots as above.
extern "C" __global__ void shx_saabas(
    SHX_MODEL_PARAMS,
    const float* __restrict__ rows, uint32_t n_rows, uint32_t n_features, uint32_t n_trees,
    uint32_t k, const float* __restrict__ group_bias, const float* __restrict__ margin,
    float* __restrict__ scratch, float* __restrict__ out) {
    SHX_MODEL_INIT;
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;
    const uint32_t g = blockIdx.y;
    float* sf = scratch;
    const uint64_t stride = (uint64_t)n_rows * k;
    const uint64_t sid = (uint64_t)g * n_rows + row;
    const uint32_t nc = n_features + 1u;
    const float* feats = rows + (uint64_t)row * n_features;
    float* p = out + ((uint64_t)row * k + g) * nc;
    for (uint32_t t = g; t < n_trees; t += k) {
        const uint32_t root = m.tree_node_begin[t];
        const uint32_t ub = m.uniq_begin[t];
        const uint32_t un = m.uniq_begin[t + 1u] - ub;
        for (uint32_t u = 0; u < un; ++u) SHX_SF(u) = 0.0f;
        float node_value = m.mean[root];
        // tree_contribs[bias] starts at zero and receives the root mean.
        p[nc - 1u] = SHX_FADD(p[nc - 1u], SHX_FADD(0.0f, node_value));
        if ((m.flags[root] & 1u) == 0u) {
            uint32_t idx = root;
            uint32_t split_slot = 0;
            while ((m.flags[idx] & 1u) == 0u) {
                split_slot = m.slot[idx];
                const bool left = shx_goes_left(m, idx, feats[m.feature[idx]]);
                idx = left ? m.left[idx] : m.right[idx];
                const float new_value = m.mean[idx];
                SHX_SF(split_slot) = SHX_FADD(SHX_SF(split_slot), SHX_FSUB(new_value, node_value));
                node_value = new_value;
            }
            SHX_SF(split_slot) = SHX_FADD(SHX_SF(split_slot), SHX_FSUB(m.leaf[idx], node_value));
        }
        for (uint32_t u = 0; u < un; ++u) {
            const uint32_t f = m.uniq_feats[ub + u];
            p[f] = SHX_FADD(p[f], SHX_SF(u));
        }
    }
    const float bias = margin != nullptr ? margin[(uint64_t)row * k + g] : group_bias[g];
    p[nc - 1u] = SHX_FADD(p[nc - 1u], bias);
}

// Interactions: matrix[(row * k + g)][a * nc + b], zeroed by the host.
extern "C" __global__ void shx_interactions(
    SHX_MODEL_PARAMS,
    const float* __restrict__ rule, const float* __restrict__ rows, uint32_t n_rows,
    uint32_t n_features, uint32_t n_trees, uint32_t k, float* __restrict__ scratch,
    float* __restrict__ out) {
    SHX_MODEL_INIT;
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;
    const uint32_t g = blockIdx.y;
    uint32_t* su = reinterpret_cast<uint32_t*>(scratch);
    float* sf = scratch;
    const uint64_t stride = (uint64_t)n_rows * k;
    const uint64_t sid = (uint64_t)g * n_rows + row;
    const uint32_t nc = n_features + 1u;
    const float* feats = rows + (uint64_t)row * n_features;
    float* mat = out + ((uint64_t)row * k + g) * nc * nc;
    float rn[SHX_POINTS], rw[SHX_POINTS];
#pragma unroll
    for (int i = 0; i < SHX_POINTS; ++i) {
        rn[i] = rule[i];
        rw[i] = rule[SHX_POINTS + i];
    }
    for (uint32_t t = g; t < n_trees; t += k) {
        const uint32_t root = m.tree_node_begin[t];
        if ((m.flags[root] & 1u) != 0u) continue;
        shx_run_tree<1>(m, feats, rn, rw, root, su, sf, stride, sid, 0u, mat, nc);
    }
}

// Averages the two directional off-diagonal estimates: one thread per
// (row/group, r), looping over the columns right of the diagonal.
extern "C" __global__ void shx_interactions_symmetrize(float* __restrict__ out,
                                                       uint64_t n_groups_total, uint32_t nc) {
    const uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_groups_total * nc) return;
    const uint64_t rg = idx / nc;
    const uint32_t r = (uint32_t)(idx % nc);
    float* mat = out + rg * nc * nc;
    for (uint32_t c = r + 1u; c < nc; ++c) {
        const float sym =
            SHX_FMUL(0.5f, SHX_FADD(mat[(uint64_t)r * nc + c], mat[(uint64_t)c * nc + r]));
        mat[(uint64_t)r * nc + c] = sym;
        mat[(uint64_t)c * nc + r] = sym;
    }
}

// Diagonal = additive SHAP value minus the row's interactions; the bias cell
// also receives the root means and the base margin.
extern "C" __global__ void shx_interactions_diagonal(
    float* __restrict__ out, uint64_t n_groups_total, uint32_t nc, uint32_t k,
    const float* __restrict__ group_root_mean, const float* __restrict__ group_bias,
    const float* __restrict__ margin) {
    const uint64_t idx = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_groups_total * nc) return;
    const uint64_t rg = idx / nc;  // row * k + g
    const uint32_t r = (uint32_t)(idx % nc);
    const uint32_t g = (uint32_t)(rg % k);
    float* mat = out + rg * nc * nc;
    float value = mat[(uint64_t)r * nc + r];
    if (r == nc - 1u) {
        value = SHX_FADD(value, group_root_mean[g]);
        value = SHX_FADD(value, margin != nullptr ? margin[rg] : group_bias[g]);
    }
    for (uint32_t c = 0; c < nc; ++c) {
        if (c != r) value = SHX_FSUB(value, mat[(uint64_t)r * nc + c]);
    }
    mat[(uint64_t)r * nc + r] = value;
}

// pred_leaf: the node id (within its tree) of the leaf each row reaches, as f32.
extern "C" __global__ void shx_leaf(
    SHX_MODEL_PARAMS,
    const float* __restrict__ rows, uint32_t n_rows, uint32_t n_features, uint32_t n_trees,
    float* __restrict__ out) {
    SHX_MODEL_INIT;
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;
    const float* feats = rows + (uint64_t)row * n_features;
    for (uint32_t t = 0; t < n_trees; ++t) {
        const uint32_t base = m.tree_node_begin[t];
        uint32_t idx = base;
        while ((m.flags[idx] & 1u) == 0u) {
            const bool left = shx_goes_left(m, idx, feats[m.feature[idx]]);
            idx = left ? m.left[idx] : m.right[idx];
        }
        out[(uint64_t)row * n_trees + t] = (float)(idx - base);
    }
}
// ================================================================ end SHAP explanations

// ---------------------------------------------------------------- categorical prediction
//
// `predict_margin_soa_cat` is `predict_margin_soa` plus categorical decisions,
// used when the model has any categorical split. Same traversal, same
// accumulation order, same optional per-row start, so device output equals the
// host path bit for bit. Flag bit 2 marks a categorical node.
//
// A categorical node's right-going set is an arbitrary-width bitset of
// little-endian 32-bit words (XGBoost `CatBitField` layout), stored in
// `cat_words` as [n_words, word_0, ..., word_{n_words-1}] per set;
// `cat_slots[node]` is the offset of that record (unused for other nodes).
// Decision (identical to `categorical::goes_left`): NaN follows the default
// direction; a negative value or one >= 2^24 goes left; otherwise the value is
// truncated toward zero and a category inside the set goes RIGHT, any other
// (including one beyond the stored words) goes left.
#define PREDICT_CAT_OUT_OF_RANGE 16777216.0f

__device__ __forceinline__ bool predict_cat_goes_left(
    const uint32_t* __restrict__ cat_words, uint32_t slot, float value) {
    if (value < 0.0f || value >= PREDICT_CAT_OUT_OF_RANGE) return true;
    const uint32_t cat = (uint32_t)value;  // round toward zero
    const uint32_t word = cat >> 5;
    if (word >= cat_words[slot]) return true;
    return ((cat_words[slot + 1u + word] >> (cat & 31u)) & 1u) == 0u;
}

extern "C" __global__ void predict_margin_soa_cat(
    const float* __restrict__ rows,
    uint32_t n_rows,
    uint32_t n_features,
    const uint32_t* __restrict__ tree_offsets,
    uint32_t n_trees,
    const uint32_t* __restrict__ features,
    const float* __restrict__ split_conditions,
    const uint32_t* __restrict__ left_children,
    const uint32_t* __restrict__ right_children,
    const uint8_t* __restrict__ flags,
    const float* __restrict__ leaf_values,
    const uint32_t* __restrict__ cat_slots,
    const uint32_t* __restrict__ cat_words,
    const float* __restrict__ init,
    float base_margin,
    float* __restrict__ output) {
    const uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n_rows) return;

    const float* values = rows + (uint64_t)row * n_features;
    float margin = init != nullptr ? init[row] : base_margin;
    for (uint32_t tree = 0; tree < n_trees; ++tree) {
        uint32_t node = tree_offsets[tree];
        if (node == tree_offsets[tree + 1]) continue;
        uint32_t flag = flags[node];
        while ((flag & 1u) == 0u) {
            const float value = values[features[node]];
            bool go_left;
            if (value != value) {
                go_left = (flag & 2u) != 0u;
            } else if ((flag & 4u) != 0u) {
                go_left = predict_cat_goes_left(cat_words, cat_slots[node], value);
            } else {
                go_left = value < split_conditions[node];
            }
            node = go_left ? left_children[node] : right_children[node];
            flag = flags[node];
        }
        margin += leaf_values[node];
    }
    output[row] = margin;
}
