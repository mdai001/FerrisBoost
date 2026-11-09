//! 分裂枚举。

use crate::categorical::CatBits;
use crate::hist::Histogram;
use crate::types::{FeatId, GradPairFixed, GradQuantizer, TrainParams};

/// 挡住浮点噪声搞出来的「零增益」分裂(XGBoost 里叫 kRtEps)。
///
/// 没有这个阈值,梯度全抵消的节点也会分出一堆增益是 1e-9 的裂口,
/// 树白白长深,和 XGBoost 的结构直接对不上。
const MIN_GAIN_EPS: f32 = 1e-6;

#[derive(Clone, Debug)]
pub struct SplitCandidate {
    pub feat: FeatId,
    /// 第一个走右边的 bin:`bin_of(row) < bin` 归左,否则归右。
    ///
    /// 存右边界而不是「左边最后一个 bin」,是为了直接对上 XGBoost 的
    /// `value < split_condition 走左` —— 阈值就是 `cuts[bin - 1]`,
    /// 不用在 tree.rs 里再做一次 ±1 的换算。
    ///
    /// 取值范围 0..=n_bins。两个端点是合法的:0 表示所有非缺失行都走
    /// 右边,n_bins 表示都走左边 —— 只有缺失值独自成一侧时才会选中,
    /// 也就是「按是否缺失分裂」。
    ///
    /// 端点没有对应的切分点(cuts 只有 n_bins - 1 个),tree.rs 落地
    /// 阈值时要特判:bin == n_bins 取一个大于所有取值的数,bin == 0 取
    /// 一个不大于最小值的数。XGBoost 那边不用特判是因为它的 cut 数组
    /// 多带一个上界哨兵,别照着它的下标抄。
    pub bin: u32,
    /// f32,不是 f64 —— XGBoost 在比候选之前有一次
    /// `static_cast<bst_float>`,这里跟着截断才好对上它挑的那个。
    /// 算的过程是 f64,只有存下来这一步降精度。
    pub gain: f32,
    /// 两侧的梯度和,**定点**。精确值,除法留到算叶子权重时再做。
    pub left: GradPairFixed,
    pub right: GradPairFixed,
    /// 缺失值走左还是右。每个分裂独立学,不是简单填补 ——
    /// 这是 XGBoost 精度的重要来源,也是最容易写错的地方。
    pub missing_left: bool,
    /// Categorical split: the categories that go RIGHT (everything else goes
    /// left, XGBoost's convention). `None` for numeric splits. For a
    /// categorical split `bin` is unused (0), `left` is the stats of the
    /// categories outside the set (plus missing rows when `missing_left`) and
    /// `right` those of the set (plus missing rows otherwise).
    pub cats: Option<CatBits>,
}

/// 叶子权重。f64 算完再降到 f32 —— 除法只做一次,不累积。
///
/// hess ≤ 0(节点里全是零权重样本)时是 0,和 XGBoost 的 `CalcWeight` 一致;
/// 否则 `lambda = 0` 时会算出 0/0 = NaN 叶子。
pub fn leaf_weight(sum: GradPairFixed, q: &GradQuantizer, p: &TrainParams) -> f32 {
    let (grad, hess) = q.to_f64(sum);
    xgb_weight(grad, hess, p)
}

/// 分裂增益。f64:定点累加出来的和是精确的,别在最后一步又用 f32
/// 把好不容易攒下的精度丢掉。
fn gain(grad: f64, hess: f64, p: &TrainParams) -> f64 {
    grad * grad / (hess + p.lambda as f64)
}

/// 分裂增益:
///
/// ```text
/// GL²/(HL+λ) + GR²/(HR+λ) - G²/(H+λ) - γ
/// ```
///
/// **没有 0.5。** 教科书(和 XGBoost 自己的文档)写的是
/// `0.5 * [...] - γ`,但**实现里没有那个 0.5** —— XGBoost 的
/// `CalcGain` 返回的就是 `G²/(H+λ)`,`loss_chg` 是三项直接相减,
/// γ(`min_split_loss`)也是拿这个没除以 2 的值去比。
///
/// 补上 0.5 会让 γ 的实际阈值严格一倍(要求 `Δ > 2γ` 而不是 `Δ > γ`),
/// 树比 XGBoost 浅一截。实测过:γ=4 时同样的数据我们剪到 [3,5,1] 个
/// 节点,XGBoost 是 [9,5,3],预测差 0.16。`internal-docs/tools/train_compat.py`
/// 里带 `--gamma` 就是复现这件事的。
///
/// 顺带:导出的 `loss_changes` 也因此和 XGBoost 同一个量纲,
/// `get_score(importance_type="gain")` 两边可比。
///
/// `parent_gain` 是调用方按节点算好一次传进来的:它在整个节点的枚举
/// 里是常数,放进内层循环纯属浪费。
fn split_gain(parent_gain: f64, left: (f64, f64), right: (f64, f64), p: &TrainParams) -> f32 {
    let g = gain(left.0, left.1, p) + gain(right.0, right.1, p) - parent_gain - p.gamma as f64;
    // 和 XGBoost 一样,在这里截断成 f32 再拿去比
    g as f32
}

// ---------------------------------------------------------------- 单调约束
//
// 语义逐条照搬 XGBoost 3.4.1 单输出 hist 路径(`split_evaluator.h` 的
// `TreeEvaluator::SplitEvaluator` + `evaluate_splits.h` 的 `ApplyTreeSplit`):
//
// - 每个节点有 `[lower, upper]`,根节点是 `±f32::MAX`。
// - 候选的左右权重 `w = clamp(f32(-G/(H+λ)), lower, upper)`(当前节点的界)。
// - 约束 `c > 0` 而 `wl > wr`、或 `c < 0` 而 `wl < wr` 的候选直接作废。
// - 增益用**给定权重**的形式 `-(2·G·w + (H+λ)·w²)`,每侧算完截成 f32,
//   两侧 f32 相加,再减父节点同一形式的增益(f32)。权重被夹过时它和
//   `G²/(H+λ)` 是两个数,所以约束模式下**所有**特征都走这个形式 ——
//   和 XGBoost 一样:有约束时它整棵树都用 `CalcGainGivenWeight`。
// - 分裂后左右孩子继承父节点的界,`mid = (wl + wr) / 2`(f32):
//   `c > 0` 时左孩子上界、右孩子下界取 `mid`;`c < 0` 反过来。
// - 叶子值就是夹过的孩子权重 × learning_rate。
//
// 没有任何非 0 约束时完全不走这里:无约束路径的数值一位不动。

/// 节点权重的上下界(XGBoost `TreeEvaluator` 的 lower/upper)。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NodeBounds {
    pub lower: f32,
    pub upper: f32,
}

impl NodeBounds {
    pub const UNBOUNDED: Self = Self {
        lower: -f32::MAX,
        upper: f32::MAX,
    };

    /// XGBoost `ApplyBounds`:先比下界,再比上界。
    pub fn apply(&self, w: f32) -> f32 {
        if w < self.lower {
            self.lower
        } else if w > self.upper {
            self.upper
        } else {
            w
        }
    }

    /// XGBoost `TreeEvaluator::AddSplit`:按分裂特征的约束方向给两个孩子定界。
    /// `wl` / `wr` 是**夹过的**孩子权重(不含 learning_rate)。
    pub fn children(&self, constraint: i8, wl: f32, wr: f32) -> (NodeBounds, NodeBounds) {
        let mid = (wl + wr) / 2.0;
        let (mut left, mut right) = (*self, *self);
        if constraint < 0 {
            left.lower = mid;
            right.upper = mid;
        } else if constraint > 0 {
            left.upper = mid;
            right.lower = mid;
        }
        (left, right)
    }
}

/// 约束模式下一次枚举需要的上下文:全局特征 id → 约束方向,以及当前节点的界。
#[derive(Clone, Copy, Debug)]
pub struct Monotone<'a> {
    pub constraints: &'a [i8],
    pub bounds: NodeBounds,
}

impl Monotone<'_> {
    fn of(&self, feat: FeatId) -> i8 {
        self.constraints.get(feat as usize).copied().unwrap_or(0)
    }
}

/// XGBoost `ThresholdL1`:`sign(G)·max(|G| − α, 0)`。α = 0 时恒等(逐位)。
fn threshold_l1(grad: f64, alpha: f32) -> f64 {
    let a = alpha as f64;
    if grad > a {
        grad - a
    } else if grad < -a {
        grad + a
    } else {
        0.0
    }
}

/// XGBoost `CalcWeight`(`param.h`):`−ThresholdL1(G, α) / (H + λ)`,
/// `max_delta_step ≠ 0` 时再按绝对值夹住;全程 f64,最后截成 f32
/// (evaluator 的标量版本返回 float)。α = 0 且 max_delta_step = 0 时
/// 和以前的 `−G / (H + λ)` 逐位相同。
fn xgb_weight(grad: f64, hess: f64, p: &TrainParams) -> f32 {
    if hess <= 0.0 {
        return 0.0;
    }
    let mut dw = -threshold_l1(grad, p.alpha) / (hess + p.lambda as f64);
    let mds = p.max_delta_step;
    if mds != 0.0 && dw.abs() > mds as f64 {
        dw = (mds as f64).copysign(dw);
    }
    dw as f32
}

/// 约束模式下节点(或孩子)的权重:`ApplyBounds(CalcWeight(sum))`。
/// 叶子值 = 它 × learning_rate。
pub fn bounded_weight(
    sum: GradPairFixed,
    q: &GradQuantizer,
    p: &TrainParams,
    bounds: NodeBounds,
) -> f32 {
    let (grad, hess) = q.to_f64(sum);
    bounds.apply(xgb_weight(grad, hess, p))
}

/// XGBoost `CalcGainGivenWeight`:`-(2·G·w + (H+λ)·w² + 2α|w|)`。
///
/// ⚠️ `w²` 是 **f32 里的平方**(XGBoost `common::Sqr(float)`),再提升到 f64;
/// 返回值截成 f32(evaluator 的标量版本返回 float)。α = 0 那一项保留成
/// `+ 0.0`,和 XGBoost 的求值顺序一致。
fn gain_given_weight(grad: f64, hess: f64, w: f32, p: &TrainParams) -> f32 {
    if hess <= 0.0 {
        return 0.0;
    }
    let square = (w * w) as f64;
    let l1 = 2.0 * p.alpha as f64 * (w.abs() as f64);
    (-((2.0 * grad * w as f64 + (hess + p.lambda as f64) * square) + l1)) as f32
}

/// 约束模式下节点自身的增益(父节点那一项),用节点自己的界。
fn bounded_node_gain(node: (f64, f64), p: &TrainParams, bounds: NodeBounds) -> f32 {
    let w = bounds.apply(xgb_weight(node.0, node.1, p));
    gain_given_weight(node.0, node.1, w, p)
}

/// 约束模式下一个候选的增益;`None` = 不合法(Hessian 不够或违反单调方向)。
///
/// 和无约束的 `split_gain` 一样最后减 γ 再截成 f32(γ = 0 时这一步是恒等)。
/// CUDA `split_candidate` 的约束分支逐步对应这里的每一次舍入。
fn bounded_split_gain(
    parent: f32,
    left: (f64, f64),
    right: (f64, f64),
    constraint: i8,
    bounds: NodeBounds,
    p: &TrainParams,
) -> Option<f32> {
    let mcw = p.min_child_weight as f64;
    // XGBoost `IsValidSplit`:两侧 Hessian 都要 > 0 且 >= min_child_weight。
    if !(left.1 > 0.0 && right.1 > 0.0 && left.1 >= mcw && right.1 >= mcw) {
        return None;
    }
    let wl = bounds.apply(xgb_weight(left.0, left.1, p));
    let wr = bounds.apply(xgb_weight(right.0, right.1, p));
    if (constraint > 0 && wl > wr) || (constraint < 0 && wl < wr) {
        return None;
    }
    let gain =
        gain_given_weight(left.0, left.1, wl, p) + gain_given_weight(right.0, right.1, wr, p);
    let loss = gain - parent;
    Some((loss as f64 - p.gamma as f64) as f32)
}

/// 候选择优。并列时保留**先遇到的**那个。
///
/// 严格大于是刻意的:枚举顺序是「特征升序 → 正向 → 反向」,并列时
/// 保留先来的就等于「特征 id 小的优先、缺失走右优先」,和 XGBoost 的
/// `SplitEntry::Update` 一致,也让结果不依赖遍历顺序的偶然性。
#[allow(clippy::neg_cmp_op_on_partial_ord, clippy::unnecessary_map_or)] // Preserve NaN and strict first-winner semantics.
fn update_best(best: &mut Option<SplitCandidate>, cand: SplitCandidate) {
    // 写成 !(gain > eps) 而不是 gain <= eps,是为了让 NaN 也走这条:
    // λ = 0 且某侧梯度全零时 0/0 会冒出 NaN,不挡住会污染整个归约
    // (任何和 NaN 的比较都是 false,它会一路"赢"到最后)。
    if !(cand.gain > MIN_GAIN_EPS) {
        return;
    }
    if best.as_ref().is_none_or(|b| cand.gain > b.gain) {
        *best = Some(cand);
    }
}

/// 在一个列块的直方图上枚举最优分裂。
///
/// 缺失值的处理:对每个特征扫两遍 —— 一遍假设缺失走左,一遍走右,
/// 取增益大的。缺失值的梯度和 = 节点总和 - 所有非缺失 bin 之和。
///
/// `feat_offsets` 是**块内局部**偏移(见 `BinCuts::block_hist_offsets`),
/// 和 `hist::build` 用的是同一张表。
/// `sampled` 为 `Some` 时只在这批(全局升序的)特征里枚举 —— `colsample_bytree`。
/// `None` 表示全选,和以前完全一样。
///
/// ⚠️ 枚举顺序仍然是 feature 升序,采样只是**跳过**没选中的特征,
/// 不改变剩下那些的相对顺序 —— 并列时「保留先出现者」的语义因此不变。
pub fn best_split_in_block(
    hist: &Histogram,
    feat_ids: &[FeatId],
    feat_offsets: &[u32],
    node_sum: GradPairFixed,
    q: &GradQuantizer,
    p: &TrainParams,
    sampled: Option<&[FeatId]>,
) -> Option<SplitCandidate> {
    best_split_in_block_constrained(hist, feat_ids, feat_offsets, node_sum, q, p, sampled, None)
}

/// `best_split_in_block`,带单调约束。`monotone = None` 就是无约束路径,
/// 和 `best_split_in_block` 逐位相同。约束语义见本文件「单调约束」一节。
#[allow(clippy::too_many_arguments)] // `best_split_in_block` plus the constraint context.
pub fn best_split_in_block_constrained(
    hist: &Histogram,
    feat_ids: &[FeatId],
    feat_offsets: &[u32],
    node_sum: GradPairFixed,
    q: &GradQuantizer,
    p: &TrainParams,
    sampled: Option<&[FeatId]>,
    monotone: Option<Monotone<'_>>,
) -> Option<SplitCandidate> {
    best_split_in_block_typed(
        hist,
        feat_ids,
        feat_offsets,
        node_sum,
        q,
        p,
        sampled,
        monotone,
        None,
    )
}

/// `best_split_in_block_constrained` with categorical features.
///
/// `categorical` is indexed by **global** feature id (`BinCuts::categorical_flags`);
/// `None` or an empty slice means every feature is numeric, and the result is
/// then bit-identical to the numeric-only path. A categorical feature is
/// enumerated like XGBoost's `EnumerateOneHot` / `EnumeratePart` (see
/// [`enumerate_categorical`]) and competes with the numeric features under the
/// same strict-greater, ascending-feature rule.
#[allow(clippy::too_many_arguments)] // The constrained signature plus the feature types.
pub fn best_split_in_block_typed(
    hist: &Histogram,
    feat_ids: &[FeatId],
    feat_offsets: &[u32],
    node_sum: GradPairFixed,
    q: &GradQuantizer,
    p: &TrainParams,
    sampled: Option<&[FeatId]>,
    monotone: Option<Monotone<'_>>,
    categorical: Option<&[bool]>,
) -> Option<SplitCandidate> {
    debug_assert_eq!(feat_offsets.len(), feat_ids.len() + 1);
    debug_assert_eq!(hist.len(), feat_offsets[feat_ids.len()] as usize);

    let node = q.to_f64(node_sum);
    let parent_gain = gain(node.0, node.1, p);
    let bounded_parent = monotone.map(|m| bounded_node_gain(node, p, m.bounds));
    let mcw = p.min_child_weight as f64;
    let mut best: Option<SplitCandidate> = None;
    // 一个候选的增益:无约束走原公式;约束模式走 XGBoost 的给定权重形式。
    let candidate_gain = |feat: FeatId, lf: (f64, f64), rf: (f64, f64)| -> Option<f32> {
        match (monotone, bounded_parent) {
            (Some(m), Some(parent)) => bounded_split_gain(parent, lf, rf, m.of(feat), m.bounds, p),
            _ => {
                // min_child_weight 的剪枝必须在比较增益**之前**做,
                // 不是先挑出最优再回头检查 —— 那样会把一个本来合法的
                // 次优分裂给挤掉。
                if lf.1 < mcw || rf.1 < mcw {
                    return None;
                }
                Some(split_gain(parent_gain, lf, rf, p))
            }
        }
    };

    for (local, &feat) in feat_ids.iter().enumerate() {
        // colsample:没被选中的特征直接跳过,不参与枚举。
        if let Some(sel) = sampled {
            if sel.binary_search(&feat).is_err() {
                continue;
            }
        }
        let lo = feat_offsets[local] as usize;
        let hi = feat_offsets[local + 1] as usize;
        let bins = &hist.bins[lo..hi];

        if categorical.is_some_and(|c| c.get(feat as usize) == Some(&true)) {
            enumerate_categorical(feat, bins, node_sum, q, p, &candidate_gain, &mut best);
            continue;
        }

        // ---- 正向扫描:缺失走右 ----
        //
        // 左边是前 i+1 个 bin,右边由 node_sum 反推。缺失行不在任何
        // bin 里(hist::build 直接跳过),所以这个减法自动把它们算到
        // 右边 —— 不需要为缺失单独留槽位,这正是哨兵 bin 的用意。
        let mut left = GradPairFixed::default();
        for (i, &b) in bins.iter().enumerate() {
            left += b;
            let right = node_sum - left;
            let (lf, rf) = (q.to_f64(left), q.to_f64(right));
            let Some(gain) = candidate_gain(feat, lf, rf) else {
                continue;
            };
            update_best(
                &mut best,
                SplitCandidate {
                    feat,
                    bin: (i + 1) as u32,
                    gain,
                    left,
                    right,
                    missing_left: false,
                    cats: None,
                },
            );
        }

        // 正向扫完,left 就是所有真实 bin 之和,缺失部分 = node_sum - 它。
        //
        // 没有缺失就不用再扫一遍:反向扫出来的分裂和正向逐一对应,
        // 划分和增益完全一样,只有 missing_left 标记不同。这时候
        // XGBoost 留下的是 default_left = false,所以这里必须跳过而不是
        // 「反正增益一样,谁赢都行」—— 训练集没缺失但预测时有的话,
        // 那条数据往哪边走就是靠这个标记定的。
        //
        // 判等要带相对容差:node_sum 按行累加、Σbins 按 bin 累加,顺序
        // 不同,没有缺失时两者也只是「几乎」相等。用 == 的话那点噪声会
        // 让反向扫描算出的增益偶尔高出一个 ulp,default_left 就被翻成
        // true 了 —— 训练集上完全看不出来,喂带缺失的数据才炸。
        // 只看 hess:缺失行的 hess 恒为正(平方损失恒等于 1,logistic
        // 是 p(1-p)),真有缺失就一定顶得出这个阈值。
        // 定点之后这里可以直接判等:两边都是精确整数和,没有缺失时
        // Σbins **就是** node_sum,不再需要那条相对容差
        // (原来靠 1e-6 的容差挡浮点噪声,现在结构上就不会有噪声)。
        if left == node_sum {
            continue;
        }

        // ---- 反向扫描:缺失走左 ----
        //
        // 从高位 bin 往回累加右边,左边同样由 node_sum 反推,于是缺失
        // 落到左边。扫到 i == 0 是刻意的(XGBoost 的反向循环也扫到底):
        // 它给出「非缺失全走右、缺失独自走左」,是合法候选。它其实是
        // 正向最后一格的镜像划分,增益相同、先来的正向会赢,所以真正
        // 选中它的场合很少 —— 但漏掉它就和 XGBoost 的枚举范围不一致了。
        let mut right = GradPairFixed::default();
        for i in (0..bins.len()).rev() {
            right += bins[i];
            let left = node_sum - right;
            let (lf, rf) = (q.to_f64(left), q.to_f64(right));
            let Some(gain) = candidate_gain(feat, lf, rf) else {
                continue;
            };
            update_best(
                &mut best,
                SplitCandidate {
                    feat,
                    bin: i as u32,
                    gain,
                    left,
                    right,
                    missing_left: true,
                    cats: None,
                },
            );
        }
    }

    best
}

/// Categorical split enumeration for one feature, ported from XGBoost 3.4.1
/// `HistEvaluator::EnumerateOneHot` / `EnumeratePart` (`tree/hist/evaluate_splits.h`).
///
/// `bins` holds one histogram slot per category code `0..n` (codes that never
/// occur are zero slots, exactly as XGBoost's cuts list every integer up to the
/// largest category). The categories that go RIGHT form the candidate's set.
///
/// * **One-hot** when `n < max_cat_to_onehot`: each category alone is the right
///   side. For each one, missing rows are tried on the left first, then on the
///   right (`default_left = true` wins a tie).
/// * **Partition** otherwise: categories are stable-sorted by the unregularised
///   leaf weight of their stats (`CalcWeightCat`, which still honours alpha and
///   `max_delta_step`), then prefixes of that order are scanned. The forward scan
///   puts the first `k` categories right with missing on the left; the backward
///   scan puts the first `j` right with missing on the right. At most
///   `max_cat_threshold - 1` prefixes are tried per direction, and a prefix that
///   would hold every category is never tried.
///
/// Candidates are offered in XGBoost's order (one-hot: category ascending,
/// missing-left then missing-right; partition: forward then backward) under the
/// usual strict-greater rule, so ties resolve to the earlier candidate.
fn enumerate_categorical<F>(
    feat: FeatId,
    bins: &[GradPairFixed],
    node_sum: GradPairFixed,
    q: &GradQuantizer,
    p: &TrainParams,
    candidate_gain: &F,
    best: &mut Option<SplitCandidate>,
) where
    F: Fn(FeatId, (f64, f64), (f64, f64)) -> Option<f32>,
{
    let n = bins.len();
    // Like XGBoost, each enumeration keeps its own best candidate without a
    // set and materialises the winner's set once at the end (sized by the
    // feature's codes), then offers it under the usual rule. Offering the
    // first maximum of a sequence is the same as offering each candidate in
    // turn (`update_best` keeps the earlier one on ties), so the result is the
    // per-candidate one, without building a set per candidate (a 20k-category
    // feature would otherwise copy a 2.5 KB set for every prefix).
    struct Local {
        gain: f32,
        left: GradPairFixed,
        right: GradPairFixed,
        missing_left: bool,
        /// Which candidate: one-hot category, or prefix length of the order.
        at: usize,
    }
    let mut local: Option<Local> = None;
    let consider = |local: &mut Option<Local>,
                    left: GradPairFixed,
                    right: GradPairFixed,
                    missing_left: bool,
                    at: usize| {
        let (lf, rf) = (q.to_f64(left), q.to_f64(right));
        if let Some(gain) = candidate_gain(feat, lf, rf) {
            // Same filter and strict-greater rule as `update_best`.
            if gain > MIN_GAIN_EPS && local.as_ref().is_none_or(|l| gain > l.gain) {
                *local = Some(Local {
                    gain,
                    left,
                    right,
                    missing_left,
                    at,
                });
            }
        }
    };
    let finish = |best: &mut Option<SplitCandidate>,
                  local: Option<Local>,
                  set: &dyn Fn(usize) -> CatBits| {
        if let Some(l) = local {
            update_best(
                best,
                SplitCandidate {
                    feat,
                    bin: 0,
                    gain: l.gain,
                    left: l.left,
                    right: l.right,
                    missing_left: l.missing_left,
                    cats: Some(set(l.at)),
                },
            );
        }
    };

    if (n as u64) < p.max_cat_to_onehot as u64 {
        let mut feature_sum = GradPairFixed::default();
        for &b in bins {
            feature_sum += b;
        }
        let missing = node_sum - feature_sum;
        for (i, &b) in bins.iter().enumerate() {
            // Missing rows go with the other categories.
            consider(&mut local, node_sum - b, b, true, i);
            // Missing rows go with the chosen category.
            let mut right = b;
            right += missing;
            consider(&mut local, node_sum - right, right, false, i);
        }
        finish(best, local, &|i| {
            let mut set = CatBits::for_bins(n);
            set.set(i as u32);
            set
        });
        return;
    }

    // Partition-based: sort categories by weight, then scan prefixes.
    let weights: Vec<f32> = bins
        .iter()
        .map(|&b| {
            let (g, h) = q.to_f64(b);
            xgb_weight(g, h, p)
        })
        .collect();
    let mut order: Vec<usize> = (0..n).collect();
    // `<` comparison with a stable sort, as `std::stable_sort` with `<`:
    // equal weights (including -0.0 vs 0.0) keep ascending category order.
    order.sort_by(|&a, &b| {
        weights[a]
            .partial_cmp(&weights[b])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let n_scan = (p.max_cat_threshold as usize).min(n);
    if n_scan < 2 {
        return;
    }
    // The set of a prefix: the first `k` sorted categories.
    let prefix = |k: usize| {
        let mut set = CatBits::for_bins(n);
        for &cat in &order[..k] {
            set.set(cat as u32);
        }
        set
    };

    // Forward: the first `k` sorted categories are the set (right); missing left.
    let mut right = GradPairFixed::default();
    for (k, &cat) in order[..n_scan - 1].iter().enumerate() {
        right += bins[cat];
        consider(&mut local, node_sum - right, right, true, k + 1);
    }
    finish(best, local.take(), &prefix);

    // Backward: the sorted tail accumulates on the left; its complement (the
    // first `j` sorted categories) is the set, with missing rows on the right.
    let mut left = GradPairFixed::default();
    for j in (n - (n_scan - 1)..n).rev() {
        left += bins[order[j]];
        consider(&mut local, left, node_sum - left, false, j);
    }
    finish(best, local, &prefix);
}

/// 跨块归约:各块的最优候选里取全局最优。
///
/// 多 GPU 时这一步就是 allreduce 的内容 —— 交换的是候选,不是数据。
/// 所以并列规则要和块内一致(先来的赢),否则块的划分方式一变,
/// 同一份数据能训出两棵不同的树。
///
/// Equal gains go to the lower feature id (XGBoost `SplitEntry::NeedReplace`),
/// not to the earlier block: blocks are grouped by bin width
/// (`columns::partition_features_by_width`), so a 16-bit block can hold a
/// lower feature than an 8-bit block before it. With contiguous ascending
/// blocks (every all-8-bit grouping) the two rules pick the same candidate.
/// Candidates of one feature never come from two blocks.
pub fn reduce_best(candidates: &[Option<SplitCandidate>]) -> Option<SplitCandidate> {
    let mut best: Option<SplitCandidate> = None;
    for c in candidates.iter().flatten() {
        match &best {
            Some(b) if c.gain == b.gain && c.feat < b.feat && c.gain > MIN_GAIN_EPS => {
                best = Some(c.clone());
            }
            _ => update_best(&mut best, c.clone()),
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hist;
    use crate::types::{Bin, GradPair, MISSING_BIN};

    struct Lcg(u64);

    impl Lcg {
        fn next_u32(&mut self) -> u32 {
            self.0 = self
                .0
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (self.0 >> 33) as u32
        }

        /// [-1, 1) 上的伪随机数。
        fn next_signed(&mut self) -> f32 {
            (self.next_u32() % 20_000) as f32 / 10_000.0 - 1.0
        }
    }

    /// 一个块的全部输入:列主序 bin 数据 + 局部偏移表 + 梯度。
    ///
    /// 梯度存两份:`float` 是给人看的原始值,`gpair` 是量化后真正
    /// 参与累加的定点值。
    struct Case {
        block: Vec<Bin>,
        n_rows: usize,
        feat_ids: Vec<FeatId>,
        feat_offsets: Vec<u32>,
        gpair: Vec<GradPairFixed>,
        q: GradQuantizer,
    }

    impl Case {
        fn new(
            block: Vec<Bin>,
            n_rows: usize,
            feat_ids: Vec<FeatId>,
            feat_offsets: Vec<u32>,
            float: Vec<GradPair>,
        ) -> Self {
            let q = GradQuantizer::new(&float, n_rows);
            let gpair = float.iter().map(|&g| q.to_fixed(g)).collect();
            Self {
                block,
                n_rows,
                feat_ids,
                feat_offsets,
                gpair,
                q,
            }
        }

        fn n_feats(&self) -> usize {
            self.feat_ids.len()
        }

        fn hist(&self) -> Histogram {
            let mut h = Histogram::zeros(*self.feat_offsets.last().unwrap() as usize);
            hist::build(
                &self.block,
                self.n_rows,
                self.n_feats(),
                &self.feat_offsets,
                &self.gpair,
                None,
                &mut h.bins,
            );
            h
        }

        fn node_sum(&self) -> GradPairFixed {
            hist::node_sum(&self.gpair, None)
        }

        fn best(&self, p: &TrainParams) -> Option<SplitCandidate> {
            best_split_in_block(
                &self.hist(),
                &self.feat_ids,
                &self.feat_offsets,
                self.node_sum(),
                &self.q,
                p,
                None,
            )
        }

        /// 定点值转回来给断言看。缩放因子是 2 的幂,所以小整数和
        /// 半整数是精确还原的,可以直接判等。
        fn f(&self, g: GradPairFixed) -> (f64, f64) {
            self.q.to_f64(g)
        }

        /// 按候选描述的规则重新划分行,返回两侧的梯度和。
        /// 用来验证候选里带的 left/right 和它自己的划分自洽。
        fn partition(&self, c: &SplitCandidate) -> (GradPairFixed, GradPairFixed) {
            let local = self.feat_ids.iter().position(|&f| f == c.feat).unwrap();
            let (mut l, mut r) = (GradPairFixed::default(), GradPairFixed::default());
            for row in 0..self.n_rows {
                let b = self.block[local * self.n_rows + row];
                let goes_left = if b == MISSING_BIN {
                    c.missing_left
                } else if let Some(cats) = &c.cats {
                    !cats.contains(b as u32)
                } else {
                    (b as u32) < c.bin
                };
                if goes_left {
                    l += self.gpair[row];
                } else {
                    r += self.gpair[row];
                }
            }
            (l, r)
        }
    }

    /// 随机块。特征的 bin 数刻意不一样,好把偏移表算错的 bug 抖出来;
    /// 全局 feat_id 也刻意不从 0 开始,防止把局部下标当全局用。
    fn random_case(seed: u64, n_rows: usize, n_feats: usize, missing_pct: u32) -> Case {
        let mut rng = Lcg(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1));

        let feat_ids: Vec<FeatId> = (0..n_feats as FeatId).map(|f| f * 3 + 5).collect();
        let n_bins: Vec<usize> = (0..n_feats).map(|f| 2 + f % 5).collect();

        let mut feat_offsets = Vec::with_capacity(n_feats + 1);
        let mut acc = 0u32;
        feat_offsets.push(0);
        for &nb in &n_bins {
            acc += nb as u32;
            feat_offsets.push(acc);
        }

        let mut block = vec![0 as Bin; n_feats * n_rows];
        for f in 0..n_feats {
            for row in 0..n_rows {
                block[f * n_rows + row] = if rng.next_u32() % 100 < missing_pct {
                    MISSING_BIN
                } else {
                    (rng.next_u32() as usize % n_bins[f]) as Bin
                };
            }
        }

        // logistic 那一档的梯度形状:一阶有正有负,二阶恒正且不大。
        let float = (0..n_rows)
            .map(|_| GradPair {
                grad: rng.next_signed(),
                hess: 0.05 + 0.2 * (rng.next_u32() % 1000) as f32 / 1000.0,
            })
            .collect();

        Case::new(block, n_rows, feat_ids, feat_offsets, float)
    }

    /// 暴力参考实现:直接按行划分,枚举所有 (阈值, 缺失方向) 组合。
    ///
    /// 刻意不复用 best_split_in_block 的任何中间量 —— 不走直方图、
    /// 不做「总和 - Σbins」的反推,这样两边错到一起去的概率最小。
    fn brute_force(case: &Case, p: &TrainParams) -> Option<SplitCandidate> {
        let node = case.f(case.node_sum());
        let parent_gain = node.0 * node.0 / (node.1 + p.lambda as f64);
        let mcw = p.min_child_weight as f64;
        let mut best: Option<SplitCandidate> = None;

        for local in 0..case.n_feats() {
            let n_bins = case.feat_offsets[local + 1] - case.feat_offsets[local];
            for bin in 0..=n_bins {
                for &missing_left in &[false, true] {
                    let cand = SplitCandidate {
                        feat: case.feat_ids[local],
                        bin,
                        gain: 0.0,
                        left: GradPairFixed::default(),
                        right: GradPairFixed::default(),
                        missing_left,
                        cats: None,
                    };
                    let (l, r) = case.partition(&cand);
                    let (lf, rf) = (case.f(l), case.f(r));
                    if lf.1 < mcw || rf.1 < mcw {
                        continue;
                    }
                    let g = (lf.0 * lf.0 / (lf.1 + p.lambda as f64)
                        + rf.0 * rf.0 / (rf.1 + p.lambda as f64)
                        - parent_gain
                        - p.gamma as f64) as f32;
                    #[allow(clippy::neg_cmp_op_on_partial_ord)]
                    // Reference rejects NaN just like production.
                    if !(g > MIN_GAIN_EPS) {
                        continue;
                    }
                    #[allow(clippy::unnecessary_map_or)]
                    // Keep the independent strict-gain reference unchanged.
                    if best.as_ref().is_none_or(|b| g > b.gain) {
                        best = Some(SplitCandidate {
                            gain: g,
                            left: l,
                            right: r,
                            ..cand
                        });
                    }
                }
            }
        }
        best
    }

    #[test]
    fn matches_brute_force_on_random_cases() {
        let mut checked = 0;
        for seed in 0..40u64 {
            for &missing_pct in &[0u32, 10, 50] {
                let case = random_case(seed, 64, 4, missing_pct);
                let p = TrainParams {
                    lambda: 1.0 + (seed % 3) as f32,
                    gamma: 0.0,
                    min_child_weight: (seed % 4) as f32 * 0.5,
                    ..Default::default()
                };

                let got = case.best(&p);
                let want = brute_force(&case, &p);

                match (got, want) {
                    (None, None) => {}
                    (Some(g), Some(w)) => {
                        // 定点之后这里可以要求**逐位**相等,不再是
                        // "差在 1e-5 以内":两边的累加路径不同(直方图
                        // 前缀和 vs 逐行划分),但整数加法与顺序无关,
                        // 同一个划分算出来的 f64 输入完全一样。
                        assert_eq!(
                            g.gain.to_bits(),
                            w.gain.to_bits(),
                            "seed {seed} miss {missing_pct}%: gain {} vs brute force {}",
                            g.gain,
                            w.gain
                        );
                        // 增益并列时两边可能挑中不同的候选(枚举顺序
                        // 不一样),那是允许的;挑中同一个就必须一模一样。
                        if (g.feat, g.bin, g.missing_left) == (w.feat, w.bin, w.missing_left) {
                            assert_eq!(g.left, w.left, "seed {seed} miss {missing_pct}%");
                            assert_eq!(g.right, w.right, "seed {seed} miss {missing_pct}%");
                        }
                    }
                    (g, w) => panic!("seed {seed} miss {missing_pct}%: {g:?} vs {w:?}"),
                }
                checked += 1;
            }
        }
        assert_eq!(checked, 120);
    }

    #[test]
    fn candidate_sums_match_its_own_partition() {
        for seed in 0..20u64 {
            let case = random_case(seed, 48, 3, 20);
            let p = TrainParams {
                lambda: 1.0,
                ..Default::default()
            };
            let c = case.best(&p).expect("random data should always split");

            let (l, r) = case.partition(&c);
            // 定点:逐位相等,不是"差在 1e-4 以内"
            assert_eq!(c.left, l, "seed {seed} left");
            assert_eq!(c.right, r, "seed {seed} right");
            // 两边加起来必须是节点总和,缺失行一个都不能漏
            let mut sum = c.left;
            sum += c.right;
            assert_eq!(sum, case.node_sum(), "seed {seed} left + right != total");
        }
    }

    /// 手工构造:缺失行的梯度和哪一侧同号,就该被分到哪一侧。
    fn missing_direction_case(missing_grad: f32) -> Case {
        // 一个特征、2 个 bin。bin 0 的梯度是 -1,bin 1 是 +1,
        // 缺失行的梯度由参数给。
        let float = vec![
            GradPair {
                grad: -1.0,
                hess: 1.0,
            },
            GradPair {
                grad: -1.0,
                hess: 1.0,
            },
            GradPair {
                grad: 1.0,
                hess: 1.0,
            },
            GradPair {
                grad: 1.0,
                hess: 1.0,
            },
            GradPair {
                grad: missing_grad,
                hess: 1.0,
            },
            GradPair {
                grad: missing_grad,
                hess: 1.0,
            },
        ];
        Case::new(
            vec![0, 0, 1, 1, MISSING_BIN, MISSING_BIN],
            6,
            vec![0],
            vec![0, 2],
            float,
        )
    }

    #[test]
    fn missing_goes_to_the_side_it_agrees_with() {
        let p = TrainParams {
            lambda: 1.0,
            min_child_weight: 0.0,
            ..Default::default()
        };

        // 缺失行梯度为负,和 bin 0 那一侧同号 —— 该走左
        let case = missing_direction_case(-1.0);
        let c = case.best(&p).unwrap();
        assert_eq!(c.bin, 1);
        assert!(
            c.missing_left,
            "missing has the same sign as the left side, should go left"
        );
        assert_eq!(case.f(c.left).0, -4.0);
        assert_eq!(case.f(c.right).0, 2.0);

        // 梯度为正 —— 该走右
        let case = missing_direction_case(1.0);
        let c = case.best(&p).unwrap();
        assert_eq!(c.bin, 1);
        assert!(
            !c.missing_left,
            "missing has the same sign as the right side, should go right"
        );
        assert_eq!(case.f(c.left).0, -2.0);
        assert_eq!(case.f(c.right).0, 4.0);
    }

    #[test]
    fn no_missing_means_default_direction_stays_right() {
        // 训练集里一个缺失都没有时,XGBoost 留下的是 default_left = false。
        // 反向扫描能给出增益完全相同的候选,但标记是 true —— 跳过反向
        // 扫描才对得上。这个断言就是在钉住那个 continue。
        //
        // 定点之后判断「有没有缺失」是精确的整数判等,不再靠容差。
        let p = TrainParams {
            lambda: 1.0,
            min_child_weight: 0.0,
            ..Default::default()
        };
        for seed in 0..10u64 {
            let case = random_case(seed, 32, 3, 0);
            let c = case.best(&p).unwrap();
            assert!(
                !c.missing_left,
                "seed {seed}: default direction should be right when nothing is missing"
            );
        }
    }

    #[test]
    fn missing_alone_on_one_side_is_a_valid_split() {
        // 非缺失行的梯度全相同,唯一有意义的分裂就是「缺失 vs 非缺失」。
        // 端点 bin 要是被漏掉,这里就一个分裂都找不出来。
        //
        // 期望值是正向的那一侧(bin == n_bins、缺失走右),不是反向的
        // bin == 0:两者是同一个划分的镜像,增益一样,先枚举到的赢。
        let case = Case::new(
            vec![0, 1, 0, 1, MISSING_BIN, MISSING_BIN],
            6,
            vec![0],
            vec![0, 2],
            vec![
                GradPair {
                    grad: 1.0,
                    hess: 1.0,
                },
                GradPair {
                    grad: 1.0,
                    hess: 1.0,
                },
                GradPair {
                    grad: 1.0,
                    hess: 1.0,
                },
                GradPair {
                    grad: 1.0,
                    hess: 1.0,
                },
                GradPair {
                    grad: -3.0,
                    hess: 1.0,
                },
                GradPair {
                    grad: -3.0,
                    hess: 1.0,
                },
            ],
        );
        let p = TrainParams {
            lambda: 1.0,
            min_child_weight: 0.0,
            ..Default::default()
        };
        let c = case.best(&p).unwrap();

        assert_eq!(
            c.bin, 2,
            "all non-missing rows go left, so the right boundary is the bin count"
        );
        assert!(!c.missing_left, "missing stays alone on the right");
        assert_eq!(case.f(c.left).0, 4.0, "sum of the two real bins");
        assert_eq!(case.f(c.right).0, -6.0, "two missing rows");
    }

    #[test]
    fn constant_feature_without_missing_has_no_split() {
        let case = Case::new(
            vec![0, 0, 0, 0],
            4,
            vec![0],
            vec![0, 1],
            vec![
                GradPair {
                    grad: 1.0,
                    hess: 1.0,
                },
                GradPair {
                    grad: -1.0,
                    hess: 1.0,
                },
                GradPair {
                    grad: 2.0,
                    hess: 1.0,
                },
                GradPair {
                    grad: -2.0,
                    hess: 1.0,
                },
            ],
        );
        let p = TrainParams {
            lambda: 1.0,
            min_child_weight: 0.0,
            ..Default::default()
        };
        assert!(case.best(&p).is_none());
    }

    #[test]
    fn min_child_weight_prunes_before_gain_comparison() {
        // 特征 0 在 bin 1 上有个"完美"分裂,但左边只有 1 行(hess = 1);
        // 特征 1 的分裂增益低一些,两边各 3 行。
        // min_child_weight = 2 时必须选特征 1 —— 先剪枝、再比增益。
        // 反过来(先挑最优再检查)会把特征 0 挑出来然后整个作废。
        let case = Case::new(
            vec![
                0, 1, 1, 1, 1, 1, // feat 0
                0, 0, 0, 1, 1, 1, // feat 1
            ],
            6,
            vec![0, 1],
            vec![0, 2, 4],
            vec![
                GradPair {
                    grad: -5.0,
                    hess: 1.0,
                },
                GradPair {
                    grad: 1.0,
                    hess: 1.0,
                },
                GradPair {
                    grad: 1.0,
                    hess: 1.0,
                },
                GradPair {
                    grad: 1.0,
                    hess: 1.0,
                },
                GradPair {
                    grad: 1.0,
                    hess: 1.0,
                },
                GradPair {
                    grad: 1.0,
                    hess: 1.0,
                },
            ],
        );

        let loose = TrainParams {
            lambda: 1.0,
            min_child_weight: 0.0,
            ..Default::default()
        };
        assert_eq!(
            case.best(&loose).unwrap().feat,
            0,
            "without pruning feature 0 has the highest gain"
        );

        let strict = TrainParams {
            min_child_weight: 2.0,
            ..loose.clone()
        };
        let c = case.best(&strict).unwrap();
        assert_eq!(
            c.feat, 1,
            "min_child_weight should reject feature 0 before gains are compared"
        );
        // hess 恒等于 1 时反量化必须精确还原成整数,不然 mcw 的比较
        // 会在 2.9999999 上翻边。scale 取 2 的幂就是为了这个。
        assert_eq!(case.f(c.left).1, 3.0);
        assert_eq!(case.f(c.right).1, 3.0);
    }

    #[test]
    fn gamma_threshold_uses_the_unhalved_loss_change() {
        // 钉死「增益不除以 2」:XGBoost 的实现里没有那个 0.5,γ 是拿
        // 三项直接相减的结果去比的。补上 0.5 的话 γ 的实际阈值严一倍,
        // 树比 XGBoost 浅,而且训得出来、指标还行,就是对不上。
        let case = Case::new(
            vec![0, 0, 1, 1],
            4,
            vec![0],
            vec![0, 2],
            vec![
                GradPair {
                    grad: -1.0,
                    hess: 1.0,
                },
                GradPair {
                    grad: -1.0,
                    hess: 1.0,
                },
                GradPair {
                    grad: 1.0,
                    hess: 1.0,
                },
                GradPair {
                    grad: 1.0,
                    hess: 1.0,
                },
            ],
        );
        let base = TrainParams {
            lambda: 1.0,
            gamma: 0.0,
            min_child_weight: 0.0,
            ..Default::default()
        };

        // 手算:GL = -2, GR = 2, HL = HR = 2, G = 0, λ = 1
        // 增益 = 4/3 + 4/3 - 0 = 8/3(不是 4/3)
        let g0 = case.best(&base).unwrap().gain;
        assert!(
            (g0 - 8.0 / 3.0).abs() < 1e-6,
            "gain should be 8/3, got {g0}"
        );

        // γ 略小于 8/3:还留得住;略大于:该被剪掉。
        // 要是哪天有人"顺手"把 0.5 补回去,这两条会一起挂。
        let keep = TrainParams {
            gamma: 8.0 / 3.0 - 0.01,
            ..base.clone()
        };
        assert!(case.best(&keep).is_some());

        let prune = TrainParams {
            gamma: 8.0 / 3.0 + 0.01,
            ..base.clone()
        };
        assert!(
            case.best(&prune).is_none(),
            "should prune when gamma exceeds the gain"
        );
    }

    #[test]
    fn bin_is_the_first_bin_that_goes_right() {
        // bin 的语义必须是「右边界」:b < bin 归左。差一位的话
        // tree.rs 取 cuts[bin - 1] 当阈值就会整体错开一个 bin。
        let case = Case::new(
            vec![0, 1, 2, 3],
            4,
            vec![0],
            vec![0, 4],
            vec![
                GradPair {
                    grad: -1.0,
                    hess: 1.0,
                },
                GradPair {
                    grad: -1.0,
                    hess: 1.0,
                },
                GradPair {
                    grad: 1.0,
                    hess: 1.0,
                },
                GradPair {
                    grad: 1.0,
                    hess: 1.0,
                },
            ],
        );
        let p = TrainParams {
            lambda: 1.0,
            min_child_weight: 0.0,
            ..Default::default()
        };
        let c = case.best(&p).unwrap();

        assert_eq!(
            c.bin, 2,
            "bins 0/1 go left and bins 2/3 go right, so the right boundary is 2"
        );
        assert_eq!(case.f(c.left).0, -2.0);
        assert_eq!(case.f(c.right).0, 2.0);
    }

    #[test]
    fn enumeration_is_reproducible_bit_for_bit() {
        // 同 seed 必须逐位一致。定点累加之后这是结构性的,
        // 不再依赖"行按升序遍历"。
        let case = random_case(7, 64, 4, 25);
        let p = TrainParams {
            lambda: 1.5,
            ..Default::default()
        };
        let a = case.best(&p).unwrap();
        let b = case.best(&p).unwrap();
        assert_eq!(a.gain.to_bits(), b.gain.to_bits());
        assert_eq!(a.feat, b.feat);
        assert_eq!(a.bin, b.bin);
        assert_eq!(a.missing_left, b.missing_left);
        assert_eq!(a.left, b.left);
    }

    #[test]
    fn reduce_best_takes_the_max_and_keeps_the_first_on_ties() {
        let mk = |feat: FeatId, gain: f32| {
            Some(SplitCandidate {
                feat,
                bin: 1,
                gain,
                left: GradPairFixed::default(),
                right: GradPairFixed::default(),
                missing_left: false,
                cats: None,
            })
        };

        assert!(reduce_best(&[None, None]).is_none());
        assert_eq!(
            reduce_best(&[None, mk(3, 0.5), mk(9, 2.0)]).unwrap().feat,
            9
        );
        // 并列时保留先来的:块的划分方式变了也不该换一棵树
        assert_eq!(reduce_best(&[mk(3, 2.0), mk(9, 2.0)]).unwrap().feat, 3);
        // 低于 eps 的候选一个都不该留下
        assert!(reduce_best(&[mk(3, 1e-9)]).is_none());
    }

    // ------------------------------------------------------------ categorical

    /// A one-feature block whose bins are category codes `0..n_cats`, with
    /// `missing_pct`% missing rows. Gradients depend on the category so that
    /// some categories are clearly "positive" and others "negative".
    fn categorical_case(seed: u64, n_rows: usize, n_cats: usize, missing_pct: u32) -> Case {
        let mut rng = Lcg(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(7));
        let mut block = vec![0 as Bin; n_rows];
        let mut float = Vec::with_capacity(n_rows);
        for slot in block.iter_mut() {
            let missing = rng.next_u32() % 100 < missing_pct;
            let cat = rng.next_u32() as usize % n_cats;
            *slot = if missing { MISSING_BIN } else { cat as Bin };
            let signal = if cat % 3 == 0 { 0.6 } else { -0.4 };
            float.push(GradPair {
                grad: signal + 0.3 * rng.next_signed(),
                hess: 0.1 + 0.2 * (rng.next_u32() % 1000) as f32 / 1000.0,
            });
        }
        Case::new(block, n_rows, vec![0], vec![0, n_cats as u32], float)
    }

    fn best_categorical(case: &Case, p: &TrainParams) -> Option<SplitCandidate> {
        let flags = vec![true];
        best_split_in_block_typed(
            &case.hist(),
            &case.feat_ids,
            &case.feat_offsets,
            case.node_sum(),
            &case.q,
            p,
            None,
            None,
            Some(&flags),
        )
    }

    /// The gain of a candidate recomputed from its own row partition.
    fn recomputed_gain(case: &Case, c: &SplitCandidate, p: &TrainParams) -> f32 {
        let node = case.f(case.node_sum());
        let parent = node.0 * node.0 / (node.1 + p.lambda as f64);
        let (l, r) = case.partition(c);
        assert_eq!(
            (l, r),
            (c.left, c.right),
            "candidate sums must match its partition"
        );
        let (lf, rf) = (case.f(l), case.f(r));
        (lf.0 * lf.0 / (lf.1 + p.lambda as f64) + rf.0 * rf.0 / (rf.1 + p.lambda as f64)
            - parent
            - p.gamma as f64) as f32
    }

    #[test]
    fn one_hot_matches_brute_force_over_single_categories() {
        // n_cats < max_cat_to_onehot (4): the split is one category vs the rest.
        for seed in 0..30u64 {
            for &missing_pct in &[0u32, 15, 60] {
                let case = categorical_case(seed, 80, 3, missing_pct);
                let p = TrainParams {
                    min_child_weight: 0.0,
                    ..TrainParams::default()
                };
                let got = best_categorical(&case, &p);
                // Brute force: every category alone on the right, both missing sides.
                let mut want: Option<(f32, u32, bool)> = None;
                for cat in 0..3u32 {
                    for &missing_left in &[true, false] {
                        let mut set = CatBits::empty();
                        set.set(cat);
                        let cand = SplitCandidate {
                            feat: 0,
                            bin: 0,
                            gain: 0.0,
                            left: GradPairFixed::default(),
                            right: GradPairFixed::default(),
                            missing_left,
                            cats: Some(set),
                        };
                        let (l, r) = case.partition(&cand);
                        let (lf, rf) = (case.f(l), case.f(r));
                        if lf.1 < p.min_child_weight as f64 || rf.1 < p.min_child_weight as f64 {
                            continue;
                        }
                        let g = recomputed_gain(
                            &case,
                            &SplitCandidate {
                                left: l,
                                right: r,
                                ..cand
                            },
                            &p,
                        );
                        #[allow(clippy::neg_cmp_op_on_partial_ord)]
                        if !(g > MIN_GAIN_EPS) {
                            continue;
                        }
                        if want.is_none_or(|(w, _, _)| g > w) {
                            want = Some((g, cat, missing_left));
                        }
                    }
                }
                match (got, want) {
                    (None, None) => {}
                    (Some(c), Some((g, cat, missing_left))) => {
                        assert_eq!(c.gain, g, "seed {seed} missing {missing_pct}");
                        assert_eq!(c.cats.unwrap().iter().collect::<Vec<_>>(), vec![cat]);
                        assert_eq!(c.missing_left, missing_left);
                    }
                    other => panic!("one-hot mismatch: {other:?}"),
                }
            }
        }
    }

    #[test]
    fn partition_split_is_self_consistent_and_sorted_by_weight() {
        for seed in 0..40u64 {
            for &missing_pct in &[0u32, 10, 40] {
                let case = categorical_case(seed, 200, 9, missing_pct);
                let p = TrainParams {
                    min_child_weight: 0.0,
                    ..TrainParams::default()
                };
                let Some(c) = best_categorical(&case, &p) else {
                    continue;
                };
                let cats = c
                    .cats
                    .clone()
                    .expect("categorical candidate carries its set");
                assert!(cats.count() >= 1 && (cats.count() as usize) < 9);
                // The candidate's sums and gain agree with its own partition.
                assert_eq!(c.gain, recomputed_gain(&case, &c, &p));
                // The set is a prefix or a suffix of the weight order: no
                // outside category has a smaller weight than an inside one on
                // both sides of it. Equivalent check: the sets split the
                // categories by a threshold on g/(h+lambda).
                let hist = case.hist();
                let w: Vec<f64> = hist
                    .bins
                    .iter()
                    .map(|&b| {
                        let (g, h) = case.f(b);
                        -g / (h + p.lambda as f64)
                    })
                    .collect();
                let inside_max = (0..9u32)
                    .filter(|&k| cats.contains(k))
                    .map(|k| w[k as usize])
                    .fold(f64::MIN, f64::max);
                let inside_min = (0..9u32)
                    .filter(|&k| cats.contains(k))
                    .map(|k| w[k as usize])
                    .fold(f64::MAX, f64::min);
                let outside_max = (0..9u32)
                    .filter(|&k| !cats.contains(k))
                    .map(|k| w[k as usize])
                    .fold(f64::MIN, f64::max);
                let outside_min = (0..9u32)
                    .filter(|&k| !cats.contains(k))
                    .map(|k| w[k as usize])
                    .fold(f64::MAX, f64::min);
                assert!(
                    inside_min >= outside_max || inside_max <= outside_min,
                    "seed {seed}: set is not contiguous in weight order"
                );
            }
        }
    }

    #[test]
    fn partition_split_finds_the_obvious_grouping() {
        // Categories 0..6: even ones push the prediction up, odd ones down.
        let n_rows = 6 * 40;
        let mut block = Vec::new();
        let mut float = Vec::new();
        for cat in 0..6u8 {
            for _ in 0..40 {
                block.push(cat);
                float.push(GradPair {
                    grad: if cat % 2 == 0 { -1.0 } else { 1.0 },
                    hess: 0.25,
                });
            }
        }
        let case = Case::new(block, n_rows, vec![0], vec![0, 6], float);
        let p = TrainParams {
            max_cat_to_onehot: 4,
            ..TrainParams::default()
        };
        let c = best_categorical(&case, &p).expect("a clear split exists");
        let set: Vec<u32> = c.cats.unwrap().iter().collect();
        assert!(set == vec![0, 2, 4] || set == vec![1, 3, 5], "got {set:?}");
        assert!(
            !c.missing_left || c.missing_left,
            "no missing rows: direction is free"
        );
    }

    #[test]
    fn no_missing_rows_make_the_forward_scan_win_ties() {
        // Without missing data the forward (default-left) and backward
        // (default-right) scans produce mirror partitions with equal gain;
        // XGBoost's EnumeratePart has no skip for that, so default_left wins.
        let case = categorical_case(3, 120, 7, 0);
        let c = best_categorical(&case, &TrainParams::default()).expect("split");
        assert!(c.missing_left);
        // One-hot likewise evaluates missing-left first.
        let case = categorical_case(4, 120, 3, 0);
        let c = best_categorical(&case, &TrainParams::default()).expect("split");
        assert!(c.missing_left);
    }

    #[test]
    fn max_cat_threshold_limits_the_scan() {
        // 20 categories, threshold 3: at most 2 prefix sizes per direction, so
        // the set holds <= 2 categories (forward) or >= 18 (backward).
        let case = categorical_case(11, 400, 20, 5);
        let p = TrainParams {
            max_cat_threshold: 3,
            min_child_weight: 0.0,
            ..TrainParams::default()
        };
        let c = best_categorical(&case, &p).expect("split");
        let n = c.cats.unwrap().count();
        assert!(n <= 2 || n >= 18, "set size {n}");
        // Threshold 1 scans nothing.
        let p1 = TrainParams {
            max_cat_threshold: 1,
            ..p.clone()
        };
        assert!(best_categorical(&case, &p1).is_none());
    }

    #[test]
    fn onehot_threshold_switches_enumeration() {
        // 5 categories: partition by default, one-hot once max_cat_to_onehot > 5.
        let case = categorical_case(5, 300, 5, 0);
        let part = best_categorical(&case, &TrainParams::default()).unwrap();
        let onehot = best_categorical(
            &case,
            &TrainParams {
                max_cat_to_onehot: 6,
                ..TrainParams::default()
            },
        )
        .unwrap();
        assert_eq!(onehot.cats.unwrap().count(), 1);
        assert!(
            part.gain >= onehot.gain,
            "partition sees a superset of one-hot splits here"
        );
    }

    #[test]
    fn categorical_enumeration_is_reproducible() {
        let case = categorical_case(9, 150, 12, 20);
        let p = TrainParams::default();
        let a = best_categorical(&case, &p).unwrap();
        let b = best_categorical(&case, &p).unwrap();
        assert_eq!(
            (a.gain, a.cats, a.left, a.right),
            (b.gain, b.cats, b.left, b.right)
        );
    }

    #[test]
    fn numeric_only_flags_leave_the_numeric_path_untouched() {
        let case = random_case(5, 64, 4, 10);
        let p = TrainParams::default();
        let flags = vec![false; 20];
        let typed = best_split_in_block_typed(
            &case.hist(),
            &case.feat_ids,
            &case.feat_offsets,
            case.node_sum(),
            &case.q,
            &p,
            None,
            None,
            Some(&flags),
        );
        let plain = case.best(&p);
        assert_eq!(
            typed.map(|c| (c.feat, c.bin, c.gain, c.missing_left)),
            plain.map(|c| (c.feat, c.bin, c.gain, c.missing_left))
        );
    }
}
