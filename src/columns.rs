//! 列块 —— 整个设计的核心数据结构。
//!
//! 关键约束:bin 数据存成扁平的 Vec<u8>,不是 Vec<Vec<_>>,也不带
//! 生命周期的 view 类型。取指针就能直接传给 C++,阶段 3 不用重构。

use crate::types::{Bin, FeatId, GradPairFixed};

/// 一组特征的量化数据,列主序存放。
///
/// 内存布局:`data[local_feat * n_rows + row]`
///
/// 列主序是刻意的 —— 直方图累加时按特征遍历,这样访存是连续的。
/// 行主序(XGBoost 的 Ellpack)在这个访问模式下每次都跨 stride。
#[derive(Clone)]
pub struct ColumnBlock {
    /// 本块包含的全局特征 id,升序。
    pub feat_ids: Vec<FeatId>,
    /// 扁平的 bin 数据,长度 = feat_ids.len() * n_rows。
    pub data: Vec<Bin>,
    pub n_rows: usize,
}

impl ColumnBlock {
    pub fn n_feats(&self) -> usize {
        self.feat_ids.len()
    }

    /// 取某个局部特征的整列。
    pub fn column(&self, local_feat: usize) -> &[Bin] {
        let start = local_feat * self.n_rows;
        &self.data[start..start + self.n_rows]
    }

    /// 传给 FFI 的裸指针。阶段 3 用。
    pub fn as_ptr(&self) -> *const Bin {
        self.data.as_ptr()
    }
}

/// 分箱边界。所有列块共享,常驻内存。
///
/// cuts[feat] 是该特征的切分点,长度 = 该特征实际 bin 数 - 1。
/// 不同特征的 bin 数可以不同(低基数特征用不满 max_bin)。
#[derive(Clone)]
pub struct BinCuts {
    /// 扁平存放,用 offsets 索引。
    pub values: Vec<f32>,
    /// offsets[feat]..offsets[feat+1] 是该特征的切分点范围。
    pub offsets: Vec<u32>,
    /// hist_offsets[feat] = 前面所有特征的 bin 数之和。
    /// 预计算的前缀和,直方图索引每次都要用,不能现算。
    hist_offsets: Vec<u32>,
}

impl BinCuts {
    pub fn new(values: Vec<f32>, offsets: Vec<u32>) -> Self {
        assert!(!offsets.is_empty(), "cuts offsets 至少要有起始 0");
        assert_eq!(offsets[0], 0, "cuts offsets 必须从 0 开始");
        assert_eq!(
            offsets.last().copied().map(|v| v as usize),
            Some(values.len()),
            "cuts offsets 末尾必须等于 values 长度"
        );
        assert!(offsets.windows(2).all(|w| w[0] <= w[1]), "cuts offsets 必须单调不减");
        assert!(values.iter().all(|v| v.is_finite()), "cuts 必须全部是有限值");
        let n_feats = offsets.len().saturating_sub(1);
        let mut hist_offsets = Vec::with_capacity(n_feats + 1);
        let mut acc = 0u32;
        hist_offsets.push(0);
        for f in 0..n_feats {
            // 该特征的 bin 数 = 切分点数 + 1
            let n_cuts = offsets[f + 1] - offsets[f];
            let cuts = &values[offsets[f] as usize..offsets[f + 1] as usize];
            assert!(cuts.windows(2).all(|w| w[0] < w[1]), "每个特征的 cuts 必须严格递增");
            acc = acc.checked_add(n_cuts + 1).expect("直方图总 bin 数超过 u32");
            hist_offsets.push(acc);
        }
        Self { values, offsets, hist_offsets }
    }

    /// 直方图的总长度(所有特征的 bin 数之和)。
    pub fn total_bins(&self) -> usize {
        *self.hist_offsets.last().unwrap_or(&0) as usize
    }

    pub fn n_feats(&self) -> usize {
        self.offsets.len().saturating_sub(1)
    }

    pub fn cuts_for(&self, feat: FeatId) -> &[f32] {
        let f = feat as usize;
        let (a, b) = (self.offsets[f] as usize, self.offsets[f + 1] as usize);
        &self.values[a..b]
    }

    /// 该特征的 bin 数(切分点数 + 1)。
    pub fn n_bins(&self, feat: FeatId) -> usize {
        self.cuts_for(feat).len() + 1
    }

    /// 直方图里该特征的起始偏移。直方图按特征拼接存放。
    pub fn hist_offset(&self, feat: FeatId) -> usize {
        self.hist_offsets[feat as usize] as usize
    }

    /// 一个列块的直方图偏移表,块内局部编号。
    ///
    /// hist_offset() 是全局偏移,但块的直方图只覆盖自己那组特征,
    /// 所以要重新做一次前缀和。长度 = feat_ids.len() + 1,最后一项
    /// 就是这个块的直方图总长度。
    ///
    /// 这张表直接对应 FFI 签名里的 `const uint32_t* feat_offsets`。
    pub fn block_hist_offsets(&self, feat_ids: &[FeatId]) -> Vec<u32> {
        let mut offsets = Vec::with_capacity(feat_ids.len() + 1);
        let mut acc = 0u32;
        offsets.push(0);
        for &f in feat_ids {
            acc += self.n_bins(f) as u32;
            offsets.push(acc);
        }
        offsets
    }

        /// 从 XGBoost 的 `QuantileDMatrix.get_quantile_cut()` 转过来。
    ///
    /// **两边的表示不一样,别直接塞。** XGBoost 给的是 bin **边界**:
    /// 每个特征前面有一个 `-inf`,后面有一个比最大值还大的哨兵。
    /// 8 个唯一值的特征拿到的是 `[-inf, 1, 2, 3, 4, 5, 6, 7, 14]`,
    /// 9 个数 8 个 bin。我们的 `cuts` 只存中间那些内部边界
    /// (`[1..7]`,7 个切分点 8 个 bin),所以两头都要掐掉。
    ///
    /// 掐完剩下的必须严格递增且有限,不满足就是上游给错了,直接报错
    /// 而不是硬着头皮建一个语义错乱的 BinCuts。
    pub fn from_xgboost_edges(indptr: &[u32], edges: &[f32]) -> anyhow::Result<Self> {
        let n_feats = indptr.len().saturating_sub(1);
        let mut values = Vec::new();
        let mut offsets = Vec::with_capacity(n_feats + 1);
        offsets.push(0u32);

        for f in 0..n_feats {
            let (lo, hi) = (indptr[f] as usize, indptr[f + 1] as usize);
            let slice = edges
                .get(lo..hi)
                .ok_or_else(|| anyhow::anyhow!("特征 {f} 的 indptr {lo}..{hi} 越界"))?;

            // 前面的 -inf 和最后的上界哨兵都不是我们的切分点
            let slice = match slice.split_first() {
                Some((first, rest)) if !first.is_finite() => rest,
                _ => slice,
            };
            let slice = slice.split_last().map_or(&[][..], |(_, rest)| rest);

            for (i, &v) in slice.iter().enumerate() {
                if !v.is_finite() {
                    anyhow::bail!("特征 {f} 的第 {i} 个切分点不是有限值:{v}");
                }
                if i > 0 && v <= slice[i - 1] {
                    anyhow::bail!("特征 {f} 的切分点不是严格递增:{} 之后是 {v}", slice[i - 1]);
                }
            }
            values.extend_from_slice(slice);
            offsets.push(values.len() as u32);
        }
        Ok(Self::new(values, offsets))
    }

    /// 给一个值定 bin。语义必须和 sketch::cuts 一致:
    /// cut 是上界,落入第一个满足 v < cuts[i] 的 i。
    ///
    /// NaN(缺失)不走这里 —— 它在训练时由 default direction 决定
    /// 去向,不占 bin。调用方负责先判断。
    pub fn find_bin(&self, feat: FeatId, value: f32) -> u8 {
        let cuts = self.cuts_for(feat);
        cuts.partition_point(|&c| c <= value) as u8
    }
}

/// 分箱边界从哪来。
///
/// `Provided` 不只是测试用的口子:
///
/// - 和 XGBoost 对拍时,注入它的 cuts 就能把「分箱差异」和「训练逻辑
///   差异」彻底分开 —— 两边分箱一样时预测必须逐位相同,不一样时才
///   允许分岔。
/// - 验证集必须复用训练集的 cuts(见 CLAUDE.md)。
/// - 将来跨机训练,各方必须用同一套 cuts,入口就是这里。
pub enum BinningStrategy<'a> {
    /// 自己建草图算(默认)
    Sketch { max_bins: usize },
    /// 用外部给定的切分点
    Provided(&'a BinCuts),
}

/// 列块的划分策略。
///
/// 注意:按列数平分不等于按计算量平分 —— 高基数特征的直方图更大、
/// 枚举更慢。第一版先按列数分,拿到 profile 后再考虑按 bin 数加权。
/// 自动选一个 `cols_per_block`,让**块数尽量对齐线程数**。
///
/// **为什么需要它**:列块是唯一的并行维度,所以块数决定并行度上限,
/// 而**块数不整除线程数就直接变成利用率损失** —— 实测 wide 300 列、
/// `cols_per_block=32` 是 10 个块跑 8 个线程,要两波,利用率只有 67%,
/// 正好等于理论值 10/(2×8)=62.5%。同一形状下 XGBoost 是 98%,
/// 而两边的**每单位 work 效率其实差不多**(CPU 膨胀 2.44× vs 2.62×),
/// 所以那一整段扩展性差距就是这个不对齐造成的。
/// 详见 CLAUDE 的「CPU probe:XGB 赢在利用率」。
///
/// 选法:从 `blocks = nthread` 开始,不够就往 `2×nthread`、`3×nthread` 试,
/// 直到每块的直方图放得进 `HIST_BUDGET_BYTES`。**cache 约束优先于对齐** ——
/// 块太宽会让直方图掉出 L2,实测 300 列一块(1.2 MB)比 16 列一块(65 KB)
/// 连单线程都慢。
///
/// 返回值恒 ≥ 1。`nthread` 传**实际会用的线程数**(0 已经解析过)。
pub fn auto_cols_per_block(n_features: usize, nthread: usize, max_bin: usize) -> usize {
    /// 每个列块的直方图预算。5800H 的 L2 是 512 KiB/core,留一半给
    /// gpair tile 和流式的 bins。
    const HIST_BUDGET_BYTES: usize = 256 * 1024;
    let n_features = n_features.max(1);
    let nthread = nthread.max(1);
    let bytes_per_feature = (max_bin.max(1) + 1) * std::mem::size_of::<GradPairFixed>();
    let cache_cap = (HIST_BUDGET_BYTES / bytes_per_feature.max(1)).max(1);

    for mult in 1..=n_features {
        let blocks = nthread.saturating_mul(mult);
        let cols = n_features.div_ceil(blocks);
        if cols <= cache_cap {
            return cols.max(1);
        }
        if blocks >= n_features {
            break;
        }
    }
    cache_cap.min(n_features).max(1)
}

pub fn partition_features(n_feats: usize, cols_per_block: usize) -> Vec<Vec<FeatId>> {
    (0..n_feats as FeatId)
        .collect::<Vec<_>>()
        .chunks(cols_per_block)
        .map(|c| c.to_vec())
        .collect()
}

#[cfg(test)]
mod auto_block_tests {
    use super::auto_cols_per_block;

    /// 目标形状:300 列 / 8 线程应当落在 8 个块(每块 38 列),
    /// 这正是实测最快的那一档(487 ms vs 默认 32 列/10 块的 535 ms)。
    #[test]
    fn wide_data_aligns_block_count_to_thread_count() {
        let cols = auto_cols_per_block(300, 8, 255);
        assert_eq!(cols, 38);
        assert_eq!(300_usize.div_ceil(cols), 8, "块数应当正好等于线程数");
    }

    /// 线程少的时候对齐会让块变得太宽,直方图掉出 L2 —— 这时 cache 约束
    /// 必须压过对齐,退到 `k × nthread` 个块。
    #[test]
    fn cache_bound_wins_when_alignment_would_make_blocks_too_wide() {
        let cols = auto_cols_per_block(300, 1, 255);
        assert!(cols <= 64, "300 列一块的直方图有 1.2 MB,不能选它:{cols}");
        assert_eq!(300_usize.div_ceil(cols) % 1, 0);
    }

    /// 窄数据:28 列 8 线程,应当切得足够细才能喂满线程。
    #[test]
    fn narrow_data_still_produces_at_least_thread_count_blocks() {
        let cols = auto_cols_per_block(28, 8, 255);
        assert!(cols <= 4, "28 列 8 线程应当切到 ≤4 列/块,实际 {cols}");
        assert!(28_usize.div_ceil(cols) >= 7);
    }

    /// bin 数变小,cache 能放下更宽的块,对齐就更容易达成。
    #[test]
    fn smaller_max_bin_allows_wider_blocks() {
        assert!(auto_cols_per_block(300, 1, 32) >= auto_cols_per_block(300, 1, 255));
    }

    #[test]
    fn degenerate_inputs_never_return_zero() {
        for (f, t, b) in [(1, 1, 255), (1, 64, 255), (0, 0, 0), (7, 1000, 255)] {
            assert!(auto_cols_per_block(f, t, b) >= 1);
        }
    }
}
