#include <stdint.h>

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
    const uint32_t* blk_ptr,
    const uint32_t* node_off,
    const uint32_t* node_len,
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

extern "C" __global__ void apply_delta_and_gradient(
    float* pred,
    const float* delta,
    const float* label,
    uint32_t n_rows,
    uint32_t objective,
    float* gpair_out,
    // 行采样:`row_hash >> 11 < threshold` 即选中。阈值由 host 折算好,
    // **device 上不做任何浮点换算** —— 两边比的是同一个整数,
    // 所以不可能因为浮点差异选出不同的行。`~0ull` = 全选。
    unsigned long long subsample_threshold,
    unsigned long long subsample_seed,
    unsigned long long subsample_tree) {
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
        if (objective == 0u) {
            // SquaredError
            g = p - y;
            h = 1.0f;
        } else {
            // Logistic
            const float s = 1.0f / (1.0f + expf(-p));
            g = s - y;
            h = fmaxf(s * (1.0f - s), 1e-16f);
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
    }
}


extern "C" __global__ void gpair_max_abs(
    const float* gpair,
    uint32_t n_rows,
    double* out_max) {
    extern __shared__ double red[];
    double* rg = red;
    double* rh = red + blockDim.x;

    double mg = 0.0;
    double mh = 0.0;
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n_rows;
         i += (uint32_t)blockDim.x * gridDim.x) {
        mg = fmax(mg, fabs((double)gpair[(uint64_t)i * 2]));
        mh = fmax(mh, fabs((double)gpair[(uint64_t)i * 2 + 1]));
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
    const uint32_t* blk_ptr,
    const uint32_t* node_off,
    const uint32_t* node_len,
    uint32_t n_nodes,
    uint32_t hist_stride) {
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
            const uint32_t off = feat_offsets[feat] - hist_begin;
#pragma unroll
            for (int r = 0; r < ROWS_PER_THREAD; ++r) {
                if (!ok[r]) {
                    continue;
                }
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

extern "C" __global__ void partition_count(
    const uint8_t* col,
    const uint32_t* rows,
    uint32_t n_rows,
    uint32_t n_selected,
    uint32_t split_bin,
    uint32_t missing_left,
    uint8_t* flags,
    uint32_t* block_left_counts) {
    extern __shared__ uint32_t counts[];
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t left = 0;
    if (idx < n_selected) {
        const uint32_t row = rows[idx];
        const uint8_t bin = col[(uint64_t)row];
        left = (bin == UINT8_MAX) ? missing_left : (uint32_t)(bin < split_bin);
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
extern "C" __global__ void partition_scan_blocks(
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
    }

    if (threadIdx.x == 0) {
        totals_out[slot * 2] = carry_l;
        totals_out[slot * 2 + 1] = carry_r;
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
    uint32_t* block_left_counts) {
    extern __shared__ uint32_t counts_rm[];
    const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t left = 0;
    if (idx < n_selected) {
        const uint32_t row = rows[idx];
        const uint8_t bin = bins_rm[(uint64_t)row * n_feats + feat];
        left = (bin == UINT8_MAX) ? missing_left : (uint32_t)(bin < split_bin);
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
