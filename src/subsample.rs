//! 每棵树一次的**行采样**(`subsample`)。
//!
//! # 语义不变量(审计自 XGBoost 3.4.1,属于公开训练行为)
//!
//! > **采样决定哪些行「塑造」这棵树,不决定哪些行「收到」这棵树。**
//!
//! 展开成可检查的条款:
//!
//! - 每棵树采一次,在**完整梯度已经算出来之后**、root sum / 直方图**之前**;
//! - **逐行独立的 Bernoulli 试验**,概率就是 `subsample`。
//!   选中的行数是**随机**的,不是 `floor(n_rows * subsample)`;
//! - 选中的行 gpair 原样保留;未选中的行**工作 gpair 置零**;
//! - uniform 采样**不做 `1/p` 加权**(那是 gradient-based 采样才需要的);
//! - `subsample = 1.0` 必须**恰好**是原来的路径,一个字节都不差。
//!
//! ⚠️⚠️ **采样掩码永远不能变成预测掩码。**
//! 所有行都要跟着每一次分裂走、都要走到叶子、都要拿到这棵树的预测增量 ——
//! 未选中的行只是**贡献零统计**,不是"不参与这棵树"。
//! 这条在两种 prediction authority 下都必须成立:
//!
//! - `gpu_math="exact"`:host 是权威副本,置零发生在 host 的定点 gpair 上;
//! - `gpu_math="fast"`:device 是权威副本,置零发生在 device 的 gradient
//!   kernel 里 —— 那里**必须只把 gpair 置零,绝不能跳过 `pred += delta`**。
//!   跳过它就等于把采样掩码变成了预测掩码,树会漏掉一部分行的预测更新。
//!
//! # 确定性
//!
//! 采样身份只由 `(seed, tree_id, row_id)` 决定,**逐行独立哈希**,
//! 不是顺序 RNG 流。所以它与线程调度、CUDA 几何、块宽、`colsample`、
//! 以及任何"第几次取随机数"都无关 —— 这正是"换 backend / 换线程数不改变
//! 模型"这条硬契约要求的。
//!
//! 比较在**整数空间**完成:host 把 `rate` 折成一个 u64 阈值,device 用同一个
//! 阈值比同一个哈希。这样两边不需要各自做浮点除法,也就不可能因为浮点差异
//! 选出不同的行。

use crate::types::GradPairFixed;

/// SplitMix64 的 finalizer:把 `(seed, tree, row)` 混成均匀的 u64。
///
/// 和 device 侧 `subsample_hash` 必须逐位相同。
#[inline]
pub fn mix(mut z: u64) -> u64 {
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// 某一行在这棵树里的哈希。device 侧用同一个式子。
#[inline]
pub fn row_hash(row: u64, seed: u64, tree: u64) -> u64 {
    mix(seed
        ^ tree.wrapping_mul(0x9E37_79B9_7F4A_7C15).rotate_left(23)
        ^ row.wrapping_mul(0xD6E8_FEB8_6659_FD93))
}

/// 把 `rate` 折成整数阈值:`row_hash >> 11 < threshold` 即选中。
///
/// 取高 53 位是为了和 f64 尾数宽度对齐;折算只在 host 做一次,
/// device 直接拿这个 u64 比较,**两边不做各自的浮点运算**。
#[inline]
pub fn threshold_for(rate: f32) -> u64 {
    if rate >= 1.0 {
        return u64::MAX; // 全选
    }
    if !(rate > 0.0) {
        return u64::MAX;
    }
    (f64::from(rate) * (1u64 << 53) as f64) as u64
}

/// 某一行是否被选中。
#[inline]
pub fn row_selected(row: u64, threshold: u64, seed: u64, tree: u64) -> bool {
    if threshold == u64::MAX {
        return true;
    }
    (row_hash(row, seed, tree) >> 11) < threshold
}

/// 把未选中行的**工作 gpair** 置零,返回实际选中的行数。
///
/// ⚠️ 这里改的是**工作副本**。完整梯度在概念上是另一份东西 ——
/// 预测更新、叶子值的施加、下一轮的梯度计算走的都是完整那条线,
/// 不受采样影响。
pub fn zero_unselected(gpair: &mut [GradPairFixed], rate: f32, seed: u64, tree: u64) -> usize {
    let threshold = threshold_for(rate);
    if threshold == u64::MAX {
        return gpair.len();
    }
    let mut kept = 0usize;
    for (i, g) in gpair.iter_mut().enumerate() {
        if row_selected(i as u64, threshold, seed, tree) {
            kept += 1;
        } else {
            *g = GradPairFixed::default();
        }
    }
    kept
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rate_one_is_bit_identical_to_no_sampling() {
        let mut g: Vec<GradPairFixed> = (0..64)
            .map(|i| GradPairFixed { grad: i as i64, hess: i as i64 * 2 })
            .collect();
        let orig = g.clone();
        assert_eq!(zero_unselected(&mut g, 1.0, 9, 5), 64);
        assert_eq!(g, orig, "subsample=1.0 必须完全不动数据");
    }

    #[test]
    fn same_seed_and_tree_reproduce_exactly() {
        let t = threshold_for(0.5);
        let a: Vec<bool> = (0..500).map(|r| row_selected(r, t, 11, 2)).collect();
        let b: Vec<bool> = (0..500).map(|r| row_selected(r, t, 11, 2)).collect();
        assert_eq!(a, b);
    }

    #[test]
    fn different_tree_or_seed_changes_the_sample() {
        let t = threshold_for(0.5);
        let a: Vec<bool> = (0..500).map(|r| row_selected(r, t, 11, 2)).collect();
        assert_ne!(a, (0..500).map(|r| row_selected(r, t, 11, 3)).collect::<Vec<_>>());
        assert_ne!(a, (0..500).map(|r| row_selected(r, t, 12, 2)).collect::<Vec<_>>());
    }

    #[test]
    fn selection_does_not_depend_on_traversal_order() {
        // 逐行独立 → 换遍历顺序(也就是换线程数)不能改变结果。
        let t = threshold_for(0.3);
        let fwd: Vec<bool> = (0..1000).map(|r| row_selected(r, t, 5, 1)).collect();
        let mut rev: Vec<bool> = (0..1000).rev().map(|r| row_selected(r, t, 5, 1)).collect();
        rev.reverse();
        assert_eq!(fwd, rev);
    }

    #[test]
    fn realized_count_is_random_not_a_fixed_floor() {
        // XGBoost 的 uniform 采样是逐行 Bernoulli,选中条数是随机的。
        // 不同的树几乎不可能选出完全相同的条数。
        let t = threshold_for(0.5);
        let counts: Vec<usize> = (0..8u64)
            .map(|tree| (0..10_000u64).filter(|&r| row_selected(r, t, 1, tree)).count())
            .collect();
        assert!(
            counts.windows(2).any(|w| w[0] != w[1]),
            "选中条数应当随树波动,而不是固定值:{counts:?}"
        );
        // 但要在合理范围内。
        for c in counts {
            assert!((4700..5300).contains(&c), "选中条数 {c} 偏离 50% 太远");
        }
    }

    #[test]
    fn kept_fraction_tracks_the_requested_rate() {
        for rate in [0.1f32, 0.25, 0.5, 0.9] {
            let t = threshold_for(rate);
            let n = 200_000u64;
            let kept = (0..n).filter(|&r| row_selected(r, t, 3, 0)).count() as f64;
            let frac = kept / n as f64;
            assert!((frac - f64::from(rate)).abs() < 0.01, "rate={rate} 实际 {frac:.4}");
        }
    }

    #[test]
    fn unselected_rows_are_zeroed_and_selected_rows_untouched() {
        let mut g: Vec<GradPairFixed> = (0..200)
            .map(|i| GradPairFixed { grad: i as i64 + 1, hess: 7 })
            .collect();
        let orig = g.clone();
        let t = threshold_for(0.5);
        let kept = zero_unselected(&mut g, 0.5, 4, 0);
        assert!(kept > 0 && kept < 200);
        for i in 0..200 {
            if row_selected(i as u64, t, 4, 0) {
                assert_eq!(g[i], orig[i], "选中的行不能被改动");
            } else {
                assert_eq!(g[i], GradPairFixed::default(), "未选中的行必须置零");
            }
        }
    }
}

