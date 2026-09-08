//! 分裂枚举。

use crate::hist::Histogram;
use crate::types::{FeatId, GradPairFixed, GradQuantizer, TrainParams};

/// 挡住浮点噪声搞出来的「零增益」分裂(XGBoost 里叫 kRtEps)。
///
/// 没有这个阈值,梯度全抵消的节点也会分出一堆增益是 1e-9 的裂口,
/// 树白白长深,和 XGBoost 的结构直接对不上。
const MIN_GAIN_EPS: f32 = 1e-6;

#[derive(Clone, Copy, Debug)]
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
}

/// 叶子权重。f64 算完再降到 f32 —— 除法只做一次,不累积。
pub fn leaf_weight(sum: GradPairFixed, q: &GradQuantizer, p: &TrainParams) -> f32 {
    let (grad, hess) = q.to_f64(sum);
    (-grad / (hess + p.lambda as f64)) as f32
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
fn split_gain(
    parent_gain: f64,
    left: (f64, f64),
    right: (f64, f64),
    p: &TrainParams,
) -> f32 {
    let g = gain(left.0, left.1, p) + gain(right.0, right.1, p) - parent_gain - p.gamma as f64;
    // 和 XGBoost 一样,在这里截断成 f32 再拿去比
    g as f32
}

/// 候选择优。并列时保留**先遇到的**那个。
///
/// 严格大于是刻意的:枚举顺序是「特征升序 → 正向 → 反向」,并列时
/// 保留先来的就等于「特征 id 小的优先、缺失走右优先」,和 XGBoost 的
/// `SplitEntry::Update` 一致,也让结果不依赖遍历顺序的偶然性。
#[allow(clippy::neg_cmp_op_on_partial_ord)] // 见下面对 NaN 的处理
fn update_best(best: &mut Option<SplitCandidate>, cand: SplitCandidate) {
    // 写成 !(gain > eps) 而不是 gain <= eps,是为了让 NaN 也走这条:
    // λ = 0 且某侧梯度全零时 0/0 会冒出 NaN,不挡住会污染整个归约
    // (任何和 NaN 的比较都是 false,它会一路"赢"到最后)。
    if !(cand.gain > MIN_GAIN_EPS) {
        return;
    }
    if best.map_or(true, |b| cand.gain > b.gain) {
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
    debug_assert_eq!(feat_offsets.len(), feat_ids.len() + 1);
    debug_assert_eq!(hist.len(), feat_offsets[feat_ids.len()] as usize);

    let node = q.to_f64(node_sum);
    let parent_gain = gain(node.0, node.1, p);
    let mcw = p.min_child_weight as f64;
    let mut best: Option<SplitCandidate> = None;

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
            // min_child_weight 的剪枝必须在比较增益**之前**做,
            // 不是先挑出最优再回头检查 —— 那样会把一个本来合法的
            // 次优分裂给挤掉。
            if lf.1 < mcw || rf.1 < mcw {
                continue;
            }
            update_best(
                &mut best,
                SplitCandidate {
                    feat,
                    bin: (i + 1) as u32,
                    gain: split_gain(parent_gain, lf, rf, p),
                    left,
                    right,
                    missing_left: false,
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
            if lf.1 < mcw || rf.1 < mcw {
                continue;
            }
            update_best(
                &mut best,
                SplitCandidate {
                    feat,
                    bin: i as u32,
                    gain: split_gain(parent_gain, lf, rf, p),
                    left,
                    right,
                    missing_left: true,
                },
            );
        }
    }

    best
}

/// 跨块归约:各块的最优候选里取全局最优。
///
/// 多 GPU 时这一步就是 allreduce 的内容 —— 交换的是候选,不是数据。
/// 所以并列规则要和块内一致(先来的赢),否则块的划分方式一变,
/// 同一份数据能训出两棵不同的树。
pub fn reduce_best(candidates: &[Option<SplitCandidate>]) -> Option<SplitCandidate> {
    let mut best: Option<SplitCandidate> = None;
    for &c in candidates.iter().flatten() {
        update_best(&mut best, c);
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
            Self { block, n_rows, feat_ids, feat_offsets, gpair, q }
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
                    if !(g > MIN_GAIN_EPS) {
                        continue;
                    }
                    if best.map_or(true, |b| g > b.gain) {
                        best = Some(SplitCandidate { gain: g, left: l, right: r, ..cand });
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
                            "seed {seed} miss {missing_pct}%: 增益 {} vs 暴力 {}",
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
            let p = TrainParams { lambda: 1.0, ..Default::default() };
            let c = case.best(&p).expect("随机数据总能分出来");

            let (l, r) = case.partition(&c);
            // 定点:逐位相等,不是"差在 1e-4 以内"
            assert_eq!(c.left, l, "seed {seed} 左侧");
            assert_eq!(c.right, r, "seed {seed} 右侧");
            // 两边加起来必须是节点总和,缺失行一个都不能漏
            let mut sum = c.left;
            sum += c.right;
            assert_eq!(sum, case.node_sum(), "seed {seed} 左 + 右 != 总和");
        }
    }

    /// 手工构造:缺失行的梯度和哪一侧同号,就该被分到哪一侧。
    fn missing_direction_case(missing_grad: f32) -> Case {
        // 一个特征、2 个 bin。bin 0 的梯度是 -1,bin 1 是 +1,
        // 缺失行的梯度由参数给。
        let float = vec![
            GradPair { grad: -1.0, hess: 1.0 },
            GradPair { grad: -1.0, hess: 1.0 },
            GradPair { grad: 1.0, hess: 1.0 },
            GradPair { grad: 1.0, hess: 1.0 },
            GradPair { grad: missing_grad, hess: 1.0 },
            GradPair { grad: missing_grad, hess: 1.0 },
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
        let p = TrainParams { lambda: 1.0, min_child_weight: 0.0, ..Default::default() };

        // 缺失行梯度为负,和 bin 0 那一侧同号 —— 该走左
        let case = missing_direction_case(-1.0);
        let c = case.best(&p).unwrap();
        assert_eq!(c.bin, 1);
        assert!(c.missing_left, "缺失和左侧同号,应该走左");
        assert_eq!(case.f(c.left).0, -4.0);
        assert_eq!(case.f(c.right).0, 2.0);

        // 梯度为正 —— 该走右
        let case = missing_direction_case(1.0);
        let c = case.best(&p).unwrap();
        assert_eq!(c.bin, 1);
        assert!(!c.missing_left, "缺失和右侧同号,应该走右");
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
        let p = TrainParams { lambda: 1.0, min_child_weight: 0.0, ..Default::default() };
        for seed in 0..10u64 {
            let case = random_case(seed, 32, 3, 0);
            let c = case.best(&p).unwrap();
            assert!(!c.missing_left, "seed {seed}: 无缺失时默认方向应该是右");
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
                GradPair { grad: 1.0, hess: 1.0 },
                GradPair { grad: 1.0, hess: 1.0 },
                GradPair { grad: 1.0, hess: 1.0 },
                GradPair { grad: 1.0, hess: 1.0 },
                GradPair { grad: -3.0, hess: 1.0 },
                GradPair { grad: -3.0, hess: 1.0 },
            ],
        );
        let p = TrainParams { lambda: 1.0, min_child_weight: 0.0, ..Default::default() };
        let c = case.best(&p).unwrap();

        assert_eq!(c.bin, 2, "所有非缺失行走左边,右边界就是 bin 数");
        assert!(!c.missing_left, "缺失独自留在右边");
        assert_eq!(case.f(c.left).0, 4.0, "两个真实 bin 的和");
        assert_eq!(case.f(c.right).0, -6.0, "两行缺失");
    }

    #[test]
    fn constant_feature_without_missing_has_no_split() {
        let case = Case::new(
            vec![0, 0, 0, 0],
            4,
            vec![0],
            vec![0, 1],
            vec![
                GradPair { grad: 1.0, hess: 1.0 },
                GradPair { grad: -1.0, hess: 1.0 },
                GradPair { grad: 2.0, hess: 1.0 },
                GradPair { grad: -2.0, hess: 1.0 },
            ],
        );
        let p = TrainParams { lambda: 1.0, min_child_weight: 0.0, ..Default::default() };
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
                GradPair { grad: -5.0, hess: 1.0 },
                GradPair { grad: 1.0, hess: 1.0 },
                GradPair { grad: 1.0, hess: 1.0 },
                GradPair { grad: 1.0, hess: 1.0 },
                GradPair { grad: 1.0, hess: 1.0 },
                GradPair { grad: 1.0, hess: 1.0 },
            ],
        );

        let loose = TrainParams { lambda: 1.0, min_child_weight: 0.0, ..Default::default() };
        assert_eq!(case.best(&loose).unwrap().feat, 0, "不剪枝时特征 0 增益最高");

        let strict = TrainParams { min_child_weight: 2.0, ..loose.clone() };
        let c = case.best(&strict).unwrap();
        assert_eq!(c.feat, 1, "min_child_weight 该在比增益之前就把特征 0 挡掉");
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
                GradPair { grad: -1.0, hess: 1.0 },
                GradPair { grad: -1.0, hess: 1.0 },
                GradPair { grad: 1.0, hess: 1.0 },
                GradPair { grad: 1.0, hess: 1.0 },
            ],
        );
        let base = TrainParams { lambda: 1.0, gamma: 0.0, min_child_weight: 0.0, ..Default::default() };

        // 手算:GL = -2, GR = 2, HL = HR = 2, G = 0, λ = 1
        // 增益 = 4/3 + 4/3 - 0 = 8/3(不是 4/3)
        let g0 = case.best(&base).unwrap().gain;
        assert!((g0 - 8.0 / 3.0).abs() < 1e-6, "增益应为 8/3,实际 {g0}");

        // γ 略小于 8/3:还留得住;略大于:该被剪掉。
        // 要是哪天有人"顺手"把 0.5 补回去,这两条会一起挂。
        let keep = TrainParams { gamma: 8.0 / 3.0 - 0.01, ..base.clone() };
        assert!(case.best(&keep).is_some());

        let prune = TrainParams { gamma: 8.0 / 3.0 + 0.01, ..base.clone() };
        assert!(case.best(&prune).is_none(), "γ 超过增益就该剪掉");
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
                GradPair { grad: -1.0, hess: 1.0 },
                GradPair { grad: -1.0, hess: 1.0 },
                GradPair { grad: 1.0, hess: 1.0 },
                GradPair { grad: 1.0, hess: 1.0 },
            ],
        );
        let p = TrainParams { lambda: 1.0, min_child_weight: 0.0, ..Default::default() };
        let c = case.best(&p).unwrap();

        assert_eq!(c.bin, 2, "bin 0/1 归左、bin 2/3 归右,右边界就是 2");
        assert_eq!(case.f(c.left).0, -2.0);
        assert_eq!(case.f(c.right).0, 2.0);
    }

    #[test]
    fn enumeration_is_reproducible_bit_for_bit() {
        // 同 seed 必须逐位一致。定点累加之后这是结构性的,
        // 不再依赖"行按升序遍历"。
        let case = random_case(7, 64, 4, 25);
        let p = TrainParams { lambda: 1.5, ..Default::default() };
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
            })
        };

        assert!(reduce_best(&[None, None]).is_none());
        assert_eq!(reduce_best(&[None, mk(3, 0.5), mk(9, 2.0)]).unwrap().feat, 9);
        // 并列时保留先来的:块的划分方式变了也不该换一棵树
        assert_eq!(reduce_best(&[mk(3, 2.0), mk(9, 2.0)]).unwrap().feat, 3);
        // 低于 eps 的候选一个都不该留下
        assert!(reduce_best(&[mk(3, 1e-9)]).is_none());
    }
}
