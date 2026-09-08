//! 每棵树一次的特征采样(`colsample_bytree`)。
//!
//! **契约:XGBoost 兼容的 API 与语义,FerrisBoost 自己的确定性采样序列。**
//!
//! 和 XGBoost 一致的部分(用户看到的部分):
//!
//! - 参数名、取值范围、count 语义:`max(1, floor(rate * n_features))`,
//!   **截断**而不是四舍五入,至少一个特征;
//! - 每棵树采一次;
//! - 返回的特征 id **升序**;
//! - 保留 `bytree → bylevel → bynode` 的嵌套组合空间。
//!
//! **刻意不一致的部分**:RNG。XGBoost 的 `ColSample` 在
//! `ctx->IsCUDA()` 上分叉 —— CPU 走 `std::shuffle`、GPU 走 `thrust::shuffle`,
//! **同一个 seed 下两边选出的特征集本来就不同**。而且 `std::shuffle` 的
//! 结果是实现定义的。
//!
//! FerrisBoost 的核心承诺是**换 CPU/GPU backend 不改变模型**,所以这里用
//! 一套自己的、完全定义好的序列,CPU / GPU / resident / streaming 共用。
//! 详见 `internal-docs/history.md` 的 Phase 1 审计。
//!
//! ⚠️ **采样基于全局特征 id**,不是块内下标 —— 改 `cols_per_block`
//! **不得**改变选中哪些特征。

use crate::types::FeatId;

/// SplitMix64。选它是因为它**完全由这几行定义**:没有平台差异、没有
/// 标准库实现自由度,换编译器/架构结果都一样。
struct SplitMix64(u64);

impl SplitMix64 {
    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// `[0, n)` 上的均匀整数。用拒绝采样去掉取模偏置 ——
    /// 直接 `% n` 会让小的余数概率偏高,虽然影响很小,但那是**无法解释的**
    /// 偏差,而这是一个要写进模型可复现性契约的函数。
    fn below(&mut self, n: u64) -> u64 {
        debug_assert!(n > 0);
        let zone = u64::MAX - (u64::MAX % n) - 1;
        loop {
            let v = self.next_u64();
            if v <= zone {
                return v % n;
            }
        }
    }
}

/// 本轮(本棵树)选中的特征数。`max(1, floor(rate * n_features))`,
/// 与 XGBoost 的 `std::max(1, static_cast<int>(colsample * n))` 一致。
pub fn sampled_count(n_features: usize, rate: f32) -> usize {
    if n_features == 0 {
        return 0;
    }
    let n = (f64::from(rate) * n_features as f64).floor();
    let n = if n.is_finite() && n >= 1.0 { n as usize } else { 1 };
    n.min(n_features).max(1)
}

/// 一棵树的采样特征集,**升序**。
///
/// `round` 进种子,所以每棵树不同;同一个 `(seed, round)` 永远给出同一个集合,
/// 与线程数、backend、列块划分都无关。
pub fn sample_features(n_features: usize, rate: f32, seed: u64, round: usize) -> Vec<FeatId> {
    let take = sampled_count(n_features, rate);
    if take >= n_features {
        return (0..n_features as FeatId).collect();
    }
    let mut ids: Vec<FeatId> = (0..n_features as FeatId).collect();
    // 把 round 混进种子,而不是让一个 RNG 跨树流动 —— 后者会让「第 k 棵树
    // 的采样集」依赖于之前所有消费者的调用次数(XGBoost 的
    // `ctx->Rng()` 就是这个形状,很难复现)。
    let mut rng = SplitMix64(
        seed ^ (round as u64)
            .wrapping_mul(0x9E37_79B9_7F4A_7C15)
            .rotate_left(17),
    );
    // Fisher-Yates,自后向前。
    for i in (1..ids.len()).rev() {
        let j = rng.below((i + 1) as u64) as usize;
        ids.swap(i, j);
    }
    ids.truncate(take);
    ids.sort_unstable();
    ids
}

/// 采样特征 → **物理上必须读的列块**。
///
/// 这是 colsample 能省下真实字节的关键一步:没有任何选中特征的块**根本不读**,
/// 也就不上传、不转置、不建直方图。
///
/// 返回 `(block_index, 该块内选中的局部特征下标)`,块序升序。
pub fn required_blocks(
    sampled: &[FeatId],
    block_features: &[Vec<FeatId>],
) -> Vec<(usize, Vec<usize>)> {
    let mut out = Vec::new();
    for (b, feats) in block_features.iter().enumerate() {
        let mut local = Vec::new();
        for (i, f) in feats.iter().enumerate() {
            if sampled.binary_search(f).is_ok() {
                local.push(i);
            }
        }
        if !local.is_empty() {
            out.push((b, local));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn count_truncates_and_keeps_at_least_one() {
        // 与 XGBoost 的 `max(1, (int)(rate * n))` 对齐:截断,不是四舍五入。
        assert_eq!(sampled_count(300, 0.1), 30);
        assert_eq!(sampled_count(300, 0.3), 90);
        assert_eq!(sampled_count(28, 0.7), 19); // 19.6 -> 19,不是 20
        assert_eq!(sampled_count(10, 0.99), 9); // 9.9 -> 9
        // 极小比例仍然至少一个。
        assert_eq!(sampled_count(300, 0.0001), 1);
        assert_eq!(sampled_count(300, 0.0), 1);
        assert_eq!(sampled_count(1, 0.5), 1);
    }

    #[test]
    fn rate_one_returns_every_feature_in_order() {
        let all = sample_features(64, 1.0, 7, 3);
        assert_eq!(all, (0..64u32).collect::<Vec<_>>());
    }

    #[test]
    fn sample_is_sorted_deduplicated_and_right_sized() {
        for round in 0..8 {
            let s = sample_features(300, 0.1, 42, round);
            assert_eq!(s.len(), 30);
            assert!(s.windows(2).all(|w| w[0] < w[1]), "必须升序且无重复");
            assert!(s.iter().all(|&f| f < 300));
        }
    }

    #[test]
    fn same_seed_and_round_reproduce_exactly() {
        assert_eq!(
            sample_features(300, 0.3, 11, 5),
            sample_features(300, 0.3, 11, 5)
        );
    }

    #[test]
    fn different_seed_or_round_changes_the_set() {
        let a = sample_features(300, 0.3, 11, 5);
        assert_ne!(a, sample_features(300, 0.3, 12, 5), "换 seed 应该换集合");
        assert_ne!(a, sample_features(300, 0.3, 11, 6), "换轮次应该换集合");
    }

    #[test]
    fn required_blocks_skips_blocks_without_sampled_features() {
        // 4 块 × 3 特征 = 12 个特征。
        let blocks: Vec<Vec<FeatId>> = vec![
            vec![0, 1, 2],
            vec![3, 4, 5],
            vec![6, 7, 8],
            vec![9, 10, 11],
        ];
        // 只选第 0 块的一个和第 2 块的两个 —— 第 1、3 块必须整块跳过。
        let sampled: Vec<FeatId> = vec![1, 6, 8];
        let req = required_blocks(&sampled, &blocks);
        assert_eq!(req, vec![(0, vec![1]), (2, vec![0, 2])]);
    }

    #[test]
    fn sampled_ids_do_not_depend_on_block_layout() {
        // 采样是全局特征 id 上的,和 cols_per_block 无关。
        let s = sample_features(300, 0.1, 99, 2);
        let wide: Vec<Vec<FeatId>> = (0..3).map(|b| (b * 100..(b + 1) * 100).collect()).collect();
        let narrow: Vec<Vec<FeatId>> = (0..30).map(|b| (b * 10..(b + 1) * 10).collect()).collect();
        let resolve = |layout: &[Vec<FeatId>]| -> Vec<FeatId> {
            let mut out = Vec::new();
            for (b, local) in required_blocks(&s, layout) {
                for i in local {
                    out.push(layout[b][i]);
                }
            }
            out
        };
        let from_wide = resolve(&wide);
        let from_narrow = resolve(&narrow);
        assert_eq!(from_wide, s);
        assert_eq!(from_narrow, s);
    }
}
