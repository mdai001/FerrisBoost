//! Learning to rank: LambdaMART gradients and ranking metrics.
//!
//! Semantics follow XGBoost 3.4.1 (`src/objective/lambdarank_obj.{h,cc}`,
//! `src/common/ranking_utils.{h,cc}`, `src/metric/rank_metric.cc`), by name and
//! by number, so that models can be bit-exact on losslessly binned data:
//!
//! * the model rank list is a **stable** descending sort of the group's scores;
//! * `lambdarank_pair_method=topk` enumerates every pair `(i, j)` with
//!   `i < k`, `j > i` on that list; `mean` samples `lambdarank_num_pair_per_sample`
//!   partners for every document from outside its label bucket with
//!   `std::minstd_rand` seeded `seed + group`, where `seed` is one draw of the
//!   training `std::mt19937` per round (and one extra draw when the base score
//!   is estimated);
//! * all arithmetic is f32 / f64 exactly where XGBoost uses them;
//! * rows are never reordered: gradients are per row, group offsets are a
//!   separate array.
//!
//! Query groups are a flat `group_ptr` (`n_groups + 1` offsets) that lives
//! beside the labels; nothing here touches feature storage.

use std::sync::Mutex;

use rayon::prelude::*;

use crate::types::GradPair;

/// `ltr::LambdaRankParam::NotSet()`: stored in JSON as 4294967295.
pub const NOT_SET: u32 = u32::MAX;
/// Largest relevance degree usable with the exponential gain (`MaxRel()`).
const MAX_REL: u32 = 31;
const EPS64: f64 = 1e-16;
/// XGBoost `kRtEps`.
const K_RT_EPS: f64 = 1e-6;

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum PairMethod {
    TopK,
    Mean,
}

impl PairMethod {
    pub fn name(self) -> &'static str {
        match self {
            PairMethod::TopK => "topk",
            PairMethod::Mean => "mean",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum RankKind {
    Ndcg,
    Pairwise,
    Map,
}

impl RankKind {
    pub fn name(self) -> &'static str {
        match self {
            RankKind::Ndcg => "rank:ndcg",
            RankKind::Pairwise => "rank:pairwise",
            RankKind::Map => "rank:map",
        }
    }
}

/// `ltr::LambdaRankParam`.
#[derive(Clone, Copy, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct RankParams {
    pub pair_method: PairMethod,
    /// `None` = not set (XGBoost default: 32 for `topk`, 1 for `mean`).
    pub num_pair_per_sample: Option<u32>,
    pub normalization: bool,
    pub score_normalization: bool,
    pub ndcg_exp_gain: bool,
    /// Unbiased LambdaMART (`lambdarank_unbiased`): the position bias of the click
    /// data is estimated every round and divided out of the pair gradients.
    pub unbiased: bool,
    /// `lambdarank_bias_norm`: the position-bias ratios are raised to
    /// `1 / (1 + bias_norm)`.
    pub bias_norm: f64,
}

impl Default for RankParams {
    fn default() -> Self {
        Self {
            pair_method: PairMethod::TopK,
            num_pair_per_sample: None,
            normalization: true,
            score_normalization: true,
            ndcg_exp_gain: true,
            unbiased: false,
            bias_norm: 1.0,
        }
    }
}

impl RankParams {
    /// `NumPair()`.
    pub fn num_pair(&self) -> u32 {
        match self.num_pair_per_sample {
            Some(k) => k,
            None => match self.pair_method {
                PairMethod::Mean => 1,
                PairMethod::TopK => 32,
            },
        }
    }

    pub fn has_truncation(&self) -> bool {
        self.pair_method == PairMethod::TopK
    }

    /// `TopK()`: the truncation level for the metric / ideal DCG; `NOT_SET` = whole list.
    pub fn top_k(&self) -> u32 {
        if self.has_truncation() {
            self.num_pair()
        } else {
            NOT_SET
        }
    }

    pub fn validate(&self) -> anyhow::Result<()> {
        if let Some(k) = self.num_pair_per_sample {
            crate::ensure_input!(
                k >= 1,
                "lambdarank_num_pair_per_sample must be >= 1, got {k}"
            );
        }
        crate::ensure_input!(
            self.bias_norm.is_finite() && self.bias_norm >= 0.0,
            "lambdarank_bias_norm must be a non-negative finite number"
        );
        Ok(())
    }

    /// `Regularizer()`: the exponent of the position-bias update.
    pub fn regularizer(&self) -> f64 {
        1.0 / (1.0 + self.bias_norm)
    }

    /// `RankingCache::MaxPositionSize()`: how many positions of the input list the
    /// unbiased estimation tracks (the truncation level, or at most 32 for `mean`).
    pub fn max_position_size(&self, max_group_size: usize) -> usize {
        if self.has_truncation() {
            self.num_pair() as usize
        } else {
            max_group_size.min(32)
        }
    }
}

// ------------------------------------------------------------------ groups

/// Group offsets from a per-row query id. XGBoost's rule: ids must be
/// non-decreasing along the rows; every run of equal ids is one group.
pub fn group_ptr_from_qid(qid: &[u64]) -> anyhow::Result<Vec<u32>> {
    crate::ensure_input!(
        qid.len() <= u32::MAX as usize,
        "too many rows for query groups"
    );
    let mut ptr = vec![0u32];
    for i in 1..qid.len() {
        if qid[i] < qid[i - 1] {
            crate::bail_input!(
                "qid must be sorted in non-decreasing order along the rows (row {i} has qid {} \
                 after qid {}); sort the data by query id first",
                qid[i],
                qid[i - 1]
            );
        }
        if qid[i] != qid[i - 1] {
            ptr.push(i as u32);
        }
    }
    if !qid.is_empty() {
        ptr.push(qid.len() as u32);
    }
    Ok(ptr)
}

/// Group offsets from group sizes (`group=`). Sizes must be positive and sum to `n_rows`.
pub fn group_ptr_from_sizes(sizes: &[u64], n_rows: usize) -> anyhow::Result<Vec<u32>> {
    let mut ptr = Vec::with_capacity(sizes.len() + 1);
    ptr.push(0u32);
    let mut total = 0u64;
    for (i, &s) in sizes.iter().enumerate() {
        crate::ensure_input!(
            s > 0,
            "group sizes must be positive (group {i} is empty); drop empty query groups"
        );
        total += s;
        crate::ensure_input!(
            total <= n_rows as u64,
            "group sizes sum to more than the {n_rows} data rows"
        );
        ptr.push(total as u32);
    }
    crate::ensure_input!(
        total == n_rows as u64,
        "group sizes sum to {total}, but the data has {n_rows} rows"
    );
    Ok(ptr)
}

/// Validate a `group_ptr` against the row count (used for Rust-API callers).
pub fn validate_group_ptr(ptr: &[u32], n_rows: usize, what: &str) -> anyhow::Result<()> {
    crate::ensure_input!(
        ptr.len() >= 2 && ptr[0] == 0,
        "{what}: group offsets must start at 0 and contain at least one group"
    );
    crate::ensure_input!(
        *ptr.last().unwrap() as usize == n_rows,
        "{what}: group sizes sum to {}, but the data has {n_rows} rows",
        ptr.last().unwrap()
    );
    crate::ensure_input!(
        ptr.windows(2).all(|w| w[0] < w[1]),
        "{what}: every query group must have at least one row"
    );
    Ok(())
}

// ------------------------------------------------------------------ RNG

/// `std::mt19937` (the training context RNG; one draw per round for `mean`).
pub struct Mt19937 {
    state: [u32; 624],
    index: usize,
}

impl Mt19937 {
    pub fn new(seed: u32) -> Self {
        let mut state = [0u32; 624];
        state[0] = seed;
        for i in 1..624 {
            state[i] = 1812433253u32
                .wrapping_mul(state[i - 1] ^ (state[i - 1] >> 30))
                .wrapping_add(i as u32);
        }
        Self { state, index: 624 }
    }

    pub fn next_u32(&mut self) -> u32 {
        if self.index >= 624 {
            for k in 0..624 {
                let y = (self.state[k] & 0x8000_0000) | (self.state[(k + 1) % 624] & 0x7fff_ffff);
                let mut v = self.state[(k + 397) % 624] ^ (y >> 1);
                if y & 1 != 0 {
                    v ^= 0x9908_b0df;
                }
                self.state[k] = v;
            }
            self.index = 0;
        }
        let mut y = self.state[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }
}

/// `std::minstd_rand` (a = 48271, m = 2^31 - 1).
struct MinStd(u64);

impl MinStd {
    const M: u64 = 2_147_483_647;
    fn new(seed: u32) -> Self {
        let s = u64::from(seed) % Self::M;
        Self(if s == 0 { 1 } else { s })
    }
    fn next(&mut self) -> u64 {
        self.0 = self.0 * 48271 % Self::M;
        self.0
    }
    /// `std::uniform_int_distribution<size_t>(0, n - 1)(rng)` as libstdc++ computes it
    /// for a 31-bit engine: rejection sampling on `urngrange / n` buckets.
    fn below(&mut self, n: u64) -> u64 {
        // engine range: [1, M - 1], so range = M - 2.
        let urange = n - 1;
        let urngrange = Self::M - 2;
        debug_assert!(urngrange > urange, "groups are far below 2^31 rows");
        let uerange = urange + 1;
        let scaling = urngrange / uerange;
        let past = uerange * scaling;
        loop {
            let ret = self.next() - 1;
            if ret < past {
                return ret / scaling;
            }
        }
    }
}

// ------------------------------------------------------------------ math

#[inline]
fn dcg_gain(label: f32) -> f64 {
    // `(1u << label) - 1` with the label narrowed to uint32.
    f64::from((1u32 << (label as u32)) - 1)
}

#[inline]
fn dcg_discount(idx: usize) -> f64 {
    1.0 / (idx as f64 + 2.0).log2()
}

#[inline]
fn inv_idcg_of(idcg: f64) -> f64 {
    if idcg == 0.0 {
        0.0
    } else {
        1.0 / idcg
    }
}

/// `common::Sigmoid(float)`.
#[inline]
fn sigmoid32(x: f32) -> f32 {
    let x = (-x).min(88.7f32);
    let denom = x.exp() + 1.0f32 + 1e-16f32;
    1.0f32 / denom
}

fn discounts(n: usize) -> Vec<f64> {
    (0..n).map(dcg_discount).collect()
}

/// Stable argsort of a group's scores, descending (`ArgSort(.., std::greater<>)`).
fn rank_by_score(preds: &[f32], out: &mut Vec<usize>) {
    out.clear();
    out.extend(0..preds.len());
    out.sort_by(|&a, &b| {
        let (x, y) = (preds[a], preds[b]);
        match y.partial_cmp(&x) {
            Some(o) => o,
            None => x.is_nan().cmp(&y.is_nan()),
        }
    });
}

/// Stable argsort of labels, descending.
fn rank_by_label(labels: &[f32], out: &mut Vec<usize>) {
    rank_by_score(labels, out)
}

// ------------------------------------------------------------------ gradient

/// Per-training-set LambdaMART state: group offsets, per-group weights and the
/// label-derived caches (`NDCGCache::inv_idcg_`, discounts).
pub struct RankGradient<'a> {
    kind: RankKind,
    params: RankParams,
    labels: &'a [f32],
    group_ptr: &'a [u32],
    group_weights: Option<&'a [f32]>,
    weight_norm: f64,
    inv_idcg: Vec<f64>,
    discount: Vec<f64>,
    /// Position-bias state of unbiased LambdaMART (`None` when it is off).
    bias: Option<Mutex<BiasState>>,
}

/// The estimated position biases `ti+` / `tj-` (XGBoost `ti_plus_` / `tj_minus_`)
/// and the per-row loss accumulators `li_full_` / `lj_full_` of the round in progress.
struct BiasState {
    ti_plus: Vec<f64>,
    tj_minus: Vec<f64>,
    li_full: Vec<f64>,
    lj_full: Vec<f64>,
}

impl BiasState {
    fn new(n_positions: usize, n_rows: usize) -> Self {
        Self {
            ti_plus: vec![1.0; n_positions],
            tj_minus: vec![1.0; n_positions],
            li_full: vec![0.0; n_rows],
            lj_full: vec![0.0; n_rows],
        }
    }
}

#[derive(Default)]
struct Scratch {
    rank: Vec<usize>,
    y_sorted: Vec<usize>,
    n_rel: Vec<f64>,
    acc: Vec<f64>,
}

impl<'a> RankGradient<'a> {
    pub fn new(
        kind: RankKind,
        params: RankParams,
        labels: &'a [f32],
        group_ptr: &'a [u32],
        group_weights: Option<&'a [f32]>,
    ) -> anyhow::Result<Self> {
        params.validate()?;
        validate_group_ptr(group_ptr, labels.len(), "training set")?;
        let n_groups = group_ptr.len() - 1;
        if let Some(w) = group_weights {
            crate::ensure_input!(
                w.len() == n_groups,
                "ranking weights are per query group: got {} weights for {n_groups} groups",
                w.len()
            );
        }
        let sum_weights: f64 = match group_weights {
            None => n_groups as f64,
            Some(w) => w.iter().fold(0.0f64, |a, &v| a + f64::from(v)),
        };
        let weight_norm = n_groups as f64 / sum_weights;
        let max_group = group_ptr
            .windows(2)
            .map(|w| (w[1] - w[0]) as usize)
            .max()
            .unwrap_or(0);
        let discount = discounts(max_group);
        let mut inv_idcg = Vec::new();
        if kind == RankKind::Ndcg {
            check_ndcg_labels(params.ndcg_exp_gain, labels, "training set")?;
            inv_idcg = ideal_dcg_inverse(
                labels,
                group_ptr,
                &discount,
                params.top_k(),
                params.ndcg_exp_gain,
            );
        }
        if kind == RankKind::Map {
            check_binary_labels("rank:map", labels, "training set")?;
        }
        let bias = params.unbiased.then(|| {
            Mutex::new(BiasState::new(
                params.max_position_size(max_group),
                labels.len(),
            ))
        });
        Ok(Self {
            kind,
            params,
            labels,
            group_ptr,
            group_weights,
            weight_norm,
            inv_idcg,
            discount,
            bias,
        })
    }

    /// The position biases `(ti+, tj-)` as stored in the model JSON (f32, like
    /// XGBoost's `F32Array`); `None` unless `lambdarank_unbiased` is on.
    pub fn position_bias(&self) -> Option<(Vec<f32>, Vec<f32>)> {
        let st = self
            .bias
            .as_ref()?
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        Some((
            st.ti_plus.iter().map(|&v| v as f32).collect(),
            st.tj_minus.iter().map(|&v| v as f32).collect(),
        ))
    }

    pub fn n_groups(&self) -> usize {
        self.group_ptr.len() - 1
    }

    /// Static inputs of the device LambdaMART path (`gpu_math = "fast"`).
    pub fn device_inputs(&self) -> crate::backend::cuda::RankDeviceInputs<'_> {
        crate::backend::cuda::RankDeviceInputs {
            kind: self.kind,
            params: self.params,
            labels: self.labels,
            group_ptr: self.group_ptr,
            group_weights: self.group_weights,
            weight_norm: self.weight_norm,
            inv_idcg: &self.inv_idcg,
        }
    }

    /// Gradients of every row for the scores `preds`. `seed` is the per-round draw of
    /// the training RNG (only used by `lambdarank_pair_method=mean`).
    ///
    /// With `lambdarank_unbiased` this is one XGBoost `GetGradient` call: the pair
    /// gradients use the current position biases, then the biases are re-estimated
    /// from the losses of this round (`UpdatePositionBias`). Call it once per round.
    pub fn gradients(&self, preds: &[f32], seed: u32, out: &mut Vec<GradPair>) {
        match &self.bias {
            None => self.gradients_with(preds, seed, out, None),
            Some(state) => {
                let mut st = state.lock().unwrap_or_else(|e| e.into_inner());
                self.gradients_with(preds, seed, out, Some(&mut st));
                self.update_position_bias(&mut st);
            }
        }
    }

    /// The gradients the base-score estimation uses: XGBoost runs it on a throwaway
    /// copy of the objective, so the position biases are all ones and nothing of the
    /// training state is touched.
    fn initial_gradients(&self, preds: &[f32], seed: u32, out: &mut Vec<GradPair>) {
        match &self.bias {
            None => self.gradients_with(preds, seed, out, None),
            Some(state) => {
                let n_pos = state
                    .lock()
                    .unwrap_or_else(|e| e.into_inner())
                    .ti_plus
                    .len();
                let mut fresh = BiasState::new(n_pos, self.labels.len());
                self.gradients_with(preds, seed, out, Some(&mut fresh));
            }
        }
    }

    fn gradients_with(
        &self,
        preds: &[f32],
        seed: u32,
        out: &mut Vec<GradPair>,
        bias: Option<&mut BiasState>,
    ) {
        debug_assert_eq!(preds.len(), self.labels.len());
        out.clear();
        out.resize(preds.len(), GradPair::default());
        // Disjoint per-group output slices; groups are independent, and inside a
        // group everything is sequential, so the result does not depend on threads.
        let n_groups = self.n_groups();
        let tails = split_by_groups(out.as_mut_slice(), self.group_ptr);
        let unbiased = bias.is_some();
        let (ti, tj, li_tails, lj_tails): (&[f64], &[f64], _, _) = match bias {
            Some(st) => (
                &st.ti_plus,
                &st.tj_minus,
                split_by_groups(st.li_full.as_mut_slice(), self.group_ptr),
                split_by_groups(st.lj_full.as_mut_slice(), self.group_ptr),
            ),
            None => (
                &[],
                &[],
                (0..n_groups).map(|_| <&mut [f64]>::default()).collect(),
                (0..n_groups).map(|_| <&mut [f64]>::default()).collect(),
            ),
        };
        tails
            .into_par_iter()
            .zip(li_tails)
            .zip(lj_tails)
            .enumerate()
            .for_each_init(Scratch::default, |scratch, (g, ((out_g, li_g), lj_g))| {
                let (b, e) = (self.group_ptr[g] as usize, self.group_ptr[g + 1] as usize);
                self.group_gradient(
                    g,
                    &preds[b..e],
                    &self.labels[b..e],
                    self.group_weights.map_or(1.0f32, |w| w[g]),
                    seed,
                    out_g,
                    unbiased.then_some(PositionBias {
                        ti,
                        tj,
                        li: li_g,
                        lj: lj_g,
                    }),
                    scratch,
                );
            });
    }

    /// `LambdaRankObj::UpdatePositionBias` (CPU): fold the per-row losses of every
    /// group into per-position totals (groups in order, f64), then
    /// `t(i) = (l(i) / l(0))^regularizer` (eq. 30 / 31) while `l(0)` is not tiny.
    fn update_position_bias(&self, st: &mut BiasState) {
        let n_pos = st.ti_plus.len();
        let regularizer = self.params.regularizer();
        let mut li = vec![0.0f64; n_pos];
        let mut lj = vec![0.0f64; n_pos];
        for w in self.group_ptr.windows(2) {
            let (b, e) = (w[0] as usize, w[1] as usize);
            let n = (e - b).min(n_pos);
            for i in 0..n {
                li[i] += st.li_full[b + i];
                lj[i] += st.lj_full[b + i];
            }
        }
        for i in 0..n_pos {
            if li[0] >= EPS64 {
                st.ti_plus[i] = (li[i] / li[0]).powf(regularizer);
            }
            if lj[0] >= EPS64 {
                st.tj_minus[i] = (lj[i] / lj[0]).powf(regularizer);
            }
        }
        st.li_full.fill(0.0);
        st.lj_full.fill(0.0);
    }

    #[allow(clippy::too_many_arguments)]
    fn group_gradient(
        &self,
        g: usize,
        preds: &[f32],
        labels: &[f32],
        w: f32,
        seed: u32,
        out: &mut [GradPair],
        mut bias: Option<PositionBias<'_>>,
        scratch: &mut Scratch,
    ) {
        let cnt = preds.len();
        rank_by_score(preds, &mut scratch.rank);
        if self.kind == RankKind::Map {
            map_stat(labels, &scratch.rank, &mut scratch.n_rel, &mut scratch.acc);
        }
        let rank = &scratch.rank;
        let best_score = preds[rank[0]];
        let worst_score = preds[rank[cnt - 1]];
        let norm_by_diff = self.params.score_normalization;
        let inv_idcg = if self.kind == RankKind::Ndcg {
            self.inv_idcg[g]
        } else {
            0.0
        };
        let exp_gain = self.params.ndcg_exp_gain;
        let n_rel = &scratch.n_rel;
        let acc = &scratch.acc;
        let discount = &self.discount;
        let kind = self.kind;
        // Normalization, first used by LightGBM.
        let mut sum_lambda = 0.0f64;
        let (ti, tj): (&[f64], &[f64]) = match &bias {
            Some(b) => (b.ti, b.tj),
            None => (&[], &[]),
        };
        // positions tracked by the unbiased estimation
        let n_tracked = ti.len();
        let unbiased = bias.is_some();

        let mut op = |i: usize, j: usize| {
            let (mut rank_high, mut rank_low) = (i, j);
            let (yi, yj) = (labels[rank[rank_high]], labels[rank[rank_low]]);
            if yi == yj {
                return;
            }
            if yi < yj {
                std::mem::swap(&mut rank_high, &mut rank_low);
            }
            let idx_high = rank[rank_high];
            let idx_low = rank[rank_low];
            let y_high = labels[idx_high];
            let y_low = labels[idx_low];
            let s_high = preds[idx_high];
            let s_low = preds[idx_low];

            let delta_score = f64::from((s_high - s_low).abs());
            let sigmoid = f64::from(sigmoid32(s_high - s_low));
            let raw_delta = match kind {
                RankKind::Pairwise => 1.0,
                RankKind::Ndcg => delta_ndcg(
                    exp_gain, y_high, y_low, rank_high, rank_low, inv_idcg, discount,
                ),
                RankKind::Map => {
                    // DeltaMAP expects the higher model rank first.
                    let (mut rh, mut rl, mut yh, mut yl) = (rank_high, rank_low, y_high, y_low);
                    if rh > rl {
                        std::mem::swap(&mut rh, &mut rl);
                        std::mem::swap(&mut yh, &mut yl);
                    }
                    delta_map(yh, yl, rh, rl, n_rel, acc)
                }
            };
            let mut delta_metric = raw_delta.abs();
            if norm_by_diff && best_score != worst_score {
                delta_metric /= delta_score + 0.01;
            }
            // `LambdaGrad`: the cost of the pair, used by the bias update (eq. 30 / 31).
            let cost = if unbiased {
                (1.0 / (1.0 - sigmoid)).ln() * delta_metric
            } else {
                0.0
            };
            let mut lambda_ij = (sigmoid - 1.0) * delta_metric;
            let mut hessian_ij = (sigmoid * (1.0 - sigmoid)).max(EPS64) * delta_metric * 2.0;
            // Positions beyond the tracked ones, and ratios so small that the
            // division would be unstable, keep the unscaled gradient.
            let tracked = unbiased && idx_high < n_tracked && idx_low < n_tracked;
            if tracked && tj[idx_low] >= EPS64 && ti[idx_high] >= EPS64 {
                lambda_ij /= ti[idx_high] * tj[idx_low];
                hessian_ij /= ti[idx_high] * tj[idx_low];
            }
            let pg_grad = lambda_ij as f32;
            let pg_hess = hessian_ij as f32;
            out[idx_high].grad += pg_grad;
            out[idx_high].hess += pg_hess;
            out[idx_low].grad += -pg_grad;
            out[idx_low].hess += pg_hess;
            if tracked {
                // `idx_high` / `idx_low` are positions in the input list (the rows of
                // the group in file order), the click-position order.
                if let Some(b) = bias.as_mut() {
                    if tj[idx_low] >= EPS64 {
                        b.li[idx_high] += cost / tj[idx_low];
                    }
                    if ti[idx_high] >= EPS64 {
                        b.lj[idx_low] += cost / ti[idx_high];
                    }
                }
            }
            sum_lambda += -2.0 * f64::from(pg_grad);
        };

        if self.params.has_truncation() {
            let n = cnt.min(self.params.num_pair() as usize);
            for i in 0..n {
                for j in (i + 1)..cnt {
                    op(i, j);
                }
            }
        } else {
            make_mean_pairs(
                seed.wrapping_add(g as u32),
                self.params.num_pair(),
                labels,
                rank,
                &mut scratch.y_sorted,
                &mut op,
            );
        }

        if self.params.normalization {
            let mut norm = 1.0f64;
            if self.params.pair_method == PairMethod::Mean {
                norm = 1.0 / f64::from(self.params.num_pair());
            } else if sum_lambda > 0.0 {
                norm = (1.0 + sum_lambda).log2() / sum_lambda;
            }
            if norm != 1.0 {
                let n = norm as f32;
                for p in out.iter_mut() {
                    p.grad *= n;
                    p.hess *= n;
                }
            }
        }
        let wn = self.weight_norm as f32;
        for p in out.iter_mut() {
            p.grad = p.grad * w * wn;
            p.hess = p.hess * w * wn;
        }
    }
}

/// Disjoint per-group slices of a per-row array.
fn split_by_groups<'s, T>(mut rest: &'s mut [T], group_ptr: &[u32]) -> Vec<&'s mut [T]> {
    let mut out = Vec::with_capacity(group_ptr.len().saturating_sub(1));
    for w in group_ptr.windows(2) {
        let (head, tail) = rest.split_at_mut((w[1] - w[0]) as usize);
        out.push(head);
        rest = tail;
    }
    out
}

/// What one group needs of the position-bias state: the current biases (read) and its
/// slice of the per-row loss accumulators (written).
struct PositionBias<'a> {
    ti: &'a [f64],
    tj: &'a [f64],
    li: &'a mut [f64],
    lj: &'a mut [f64],
}

#[inline]
fn delta_ndcg(
    exp_gain: bool,
    y_high: f32,
    y_low: f32,
    rank_high: usize,
    rank_low: usize,
    inv_idcg: f64,
    discount: &[f64],
) -> f64 {
    let gain_high = if exp_gain {
        dcg_gain(y_high)
    } else {
        f64::from(y_high)
    };
    let discount_high = discount[rank_high];
    let gain_low = if exp_gain {
        dcg_gain(y_low)
    } else {
        f64::from(y_low)
    };
    let discount_low = discount[rank_low];
    let original = gain_high * discount_high + gain_low * discount_low;
    let changed = gain_low * discount_high + gain_high * discount_low;
    (original - changed) * inv_idcg
}

#[inline]
fn delta_map(
    y_high: f32,
    y_low: f32,
    rank_high: usize,
    rank_low: usize,
    n_rel: &[f64],
    acc: &[f64],
) -> f64 {
    let r_h = rank_high as f64 + 1.0;
    let r_l = rank_low as f64 + 1.0;
    let n_total = *n_rel.last().unwrap();
    let m = n_rel[rank_low];
    let n = n_rel[rank_high];
    if y_high < y_low {
        let a = m / r_l - (n + 1.0) / r_h;
        let b = acc[rank_low - 1] - acc[rank_high];
        (a - b) / n_total
    } else {
        let a = n / r_h - m / r_l;
        let b = acc[rank_low - 1] - acc[rank_high];
        (a + b) / n_total
    }
}

/// `cpu_impl::MAPStat`: relevant count and `sum l_k / k` along the model rank list.
fn map_stat(labels: &[f32], rank: &[usize], n_rel: &mut Vec<f64>, acc: &mut Vec<f64>) {
    let cnt = rank.len();
    n_rel.clear();
    n_rel.resize(cnt, 0.0);
    acc.clear();
    acc.resize(cnt, 0.0);
    n_rel[0] = f64::from(labels[rank[0]]);
    for k in 1..cnt {
        n_rel[k] = n_rel[k - 1] + f64::from(labels[rank[k]]);
    }
    acc[0] = f64::from(labels[rank[0]]) / 1.0;
    for k in 1..cnt {
        acc[k] = acc[k - 1] + f64::from(labels[rank[k]]) / (k + 1) as f64;
    }
}

/// `MakePairs`, sampling branch (`lambdarank_pair_method=mean`).
fn make_mean_pairs(
    seed: u32,
    num_pair: u32,
    labels: &[f32],
    rank: &[usize],
    y_sorted: &mut Vec<usize>,
    op: &mut impl FnMut(usize, usize),
) {
    let cnt = rank.len();
    let mut rnd = MinStd::new(seed);
    // labels in model-rank order, then a stable descending sort of those positions
    let ranked: Vec<f32> = rank.iter().map(|&r| labels[r]).collect();
    rank_by_label(&ranked, y_sorted);
    let rev = |idx: usize| ranked[y_sorted[idx]];
    let mut i = 0;
    while i < cnt {
        let mut j = i + 1;
        while j < cnt && rev(i) == rev(j) {
            j += 1;
        }
        let n_lefts = i;
        let n_rights = cnt - j;
        if n_lefts + n_rights == 0 {
            i = j;
            continue;
        }
        for _ in 0..num_pair {
            for pair_idx in i..j {
                let mut ridx = rnd.below((n_lefts + n_rights) as u64) as usize;
                if ridx >= n_lefts {
                    ridx = ridx - i + j;
                }
                op(y_sorted[pair_idx], y_sorted[ridx]);
            }
        }
        i = j;
    }
}

/// `NDCGCache::InitOnCPU`: 1 / IDCG@k for every group.
fn ideal_dcg_inverse(
    labels: &[f32],
    group_ptr: &[u32],
    discount: &[f64],
    top_k: u32,
    exp_gain: bool,
) -> Vec<f64> {
    group_ptr
        .par_windows(2)
        .map_init(Vec::new, |idx, w| {
            let g = &labels[w[0] as usize..w[1] as usize];
            rank_by_label(g, idx);
            let mut idcg = 0.0f64;
            for (i, &r) in idx.iter().enumerate().take(g.len().min(top_k as usize)) {
                idcg += if exp_gain {
                    discount[i] * dcg_gain(g[r])
                } else {
                    discount[i] * f64::from(g[r])
                };
            }
            inv_idcg_of(idcg)
        })
        .collect()
}

/// `CheckNDCGLabels`.
pub fn check_ndcg_labels(exp_gain: bool, labels: &[f32], what: &str) -> anyhow::Result<()> {
    if exp_gain {
        crate::ensure_input!(
            labels
                .iter()
                .all(|&v| v >= 0.0 && (v.floor() - v).abs() <= 1e-6),
            "{what}: when using relevance degree as target, labels must be 0 or positive \
             integers (set ndcg_exp_gain=false to use the raw label as gain)"
        );
        crate::ensure_input!(
            labels.iter().all(|&v| (v as u32) <= MAX_REL),
            "{what}: relevance degrees must be at most {MAX_REL} when the exponential NDCG gain \
             is used; set ndcg_exp_gain=false to use a custom DCG gain"
        );
    }
    Ok(())
}

/// `CheckPreLabels` (`map` / `pre` need binary relevance).
pub fn check_binary_labels(name: &str, labels: &[f32], what: &str) -> anyhow::Result<()> {
    crate::ensure_input!(
        labels
            .iter()
            .all(|&y| (y - 1.0).abs() < 1e-6 || y.abs() < 1e-6),
        "{what}: {name} can only be used with binary labels (0 / 1)"
    );
    Ok(())
}

/// One draw sequence of the objective: the training RNG is advanced once for the
/// base-score estimation (only when it is estimated) and once per round, as
/// XGBoost's `ctx->Rng()` is; the draws only matter for `pair_method=mean`.
pub struct RoundSeeds {
    rng: Option<Mt19937>,
}

impl RoundSeeds {
    pub fn new(params: &RankParams, seed: u64) -> Self {
        Self {
            rng: (params.pair_method == PairMethod::Mean).then(|| Mt19937::new(seed as u32)),
        }
    }
    pub fn draw(&mut self) -> u32 {
        self.rng.as_mut().map_or(0, Mt19937::next_u32)
    }
}

/// `FitIntercept::InitEstimation` for a ranking objective: one Newton step on
/// the gradients at score 0 (`-sum g / max(sum h, kRtEps)`, f64 sums in row order).
pub fn estimate_base_score(grad: &RankGradient<'_>, seed: u32) -> f32 {
    let zeros = vec![0.0f32; grad.labels.len()];
    let mut out = Vec::new();
    grad.initial_gradients(&zeros, seed, &mut out);
    let (mut sg, mut sh) = (0.0f64, 0.0f64);
    for p in &out {
        sg += f64::from(p.grad);
        sh += f64::from(p.hess);
    }
    (-sg / sh.max(K_RT_EPS)) as f32
}

// ------------------------------------------------------------------ metrics

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RankMetricKind {
    Ndcg,
    Map,
    Pre,
}

impl RankMetricKind {
    fn base(self) -> &'static str {
        match self {
            RankMetricKind::Ndcg => "ndcg",
            RankMetricKind::Map => "map",
            RankMetricKind::Pre => "pre",
        }
    }
}

/// `ndcg` / `map` / `pre` with XGBoost's `@k` and trailing `-` syntax.
pub struct RankMetric {
    kind: RankMetricKind,
    name: String,
    /// Truncation level; `NOT_SET` = whole list.
    top_k: u32,
    minus: bool,
    exp_gain: bool,
}

/// Is `name` a ranking metric name (`ndcg`, `map`, `pre`, optionally `@k` / `-`)?
pub fn is_rank_metric_name(name: &str) -> bool {
    split_metric_name(name).is_some()
}

fn split_metric_name(name: &str) -> Option<(RankMetricKind, &str)> {
    let (prefix, rest) = match name.find('@') {
        Some(p) => (&name[..p], &name[p + 1..]),
        None => match name.strip_suffix('-') {
            Some(p) => (p, "-"),
            None => (name, ""),
        },
    };
    let kind = match prefix {
        "ndcg" => RankMetricKind::Ndcg,
        "map" => RankMetricKind::Map,
        "pre" => RankMetricKind::Pre,
        _ => return None,
    };
    Some((kind, rest))
}

impl RankMetric {
    /// Parse an `eval_metric` name. An explicit metric without `@k` is
    /// truncated at 32, exactly as in XGBoost (a metric's own parameter is the
    /// default `topk` / 32 unless it is the objective's default metric).
    pub fn parse(name: &str, ndcg_exp_gain: bool) -> anyhow::Result<Self> {
        let (kind, rest) = split_metric_name(name)
            .ok_or_else(|| anyhow::anyhow!("unknown ranking metric {name:?}"))?;
        let minus = rest.ends_with('-');
        let digits = rest.trim_end_matches('-');
        let top_k = if digits.is_empty() {
            RankParams::default().num_pair()
        } else {
            let k: u32 = digits.parse().map_err(|_| {
                anyhow::anyhow!(
                    "invalid ranking metric {name:?}: expected {}@<k>[-]",
                    kind.base()
                )
            })?;
            crate::ensure_input!(k >= 1, "invalid ranking metric {name:?}: k must be >= 1");
            k
        };
        Ok(Self {
            kind,
            name: name.to_string(),
            top_k,
            minus,
            exp_gain: ndcg_exp_gain,
        })
    }

    /// The metric XGBoost picks for a ranking objective when none is configured.
    pub fn default_for(kind: RankKind, params: &RankParams) -> Self {
        let base = if kind == RankKind::Map {
            RankMetricKind::Map
        } else {
            RankMetricKind::Ndcg
        };
        let name = if params.has_truncation() {
            format!("{}@{}", base.base(), params.num_pair())
        } else {
            base.base().to_string()
        };
        // ndcg objectives hand their whole parameter block to the default metric;
        // `map` builds it from the name alone (so mean-mode `map` is truncated at 32).
        let top_k = if base == RankMetricKind::Map {
            if params.has_truncation() {
                params.num_pair()
            } else {
                RankParams::default().num_pair()
            }
        } else {
            params.top_k()
        };
        Self {
            kind: base,
            name,
            top_k,
            minus: false,
            exp_gain: params.ndcg_exp_gain,
        }
    }

    pub fn validate_labels(&self, labels: &[f32], what: &str) -> anyhow::Result<()> {
        match self.kind {
            RankMetricKind::Ndcg => check_ndcg_labels(self.exp_gain, labels, what),
            RankMetricKind::Map | RankMetricKind::Pre => {
                check_binary_labels(self.kind.base(), labels, what)
            }
        }
    }

    pub fn score(
        &self,
        pred: &[f32],
        label: &[f32],
        weight: Option<&[f32]>,
        group_ptr: Option<&[u32]>,
    ) -> f64 {
        let single = [0u32, pred.len() as u32];
        let gptr = group_ptr.unwrap_or(&single);
        let n_groups = gptr.len() - 1;
        let w_of = |g: usize| weight.map_or(1.0f32, |w| w[g]);
        let sum_w = match weight {
            None => n_groups as f64,
            Some(w) => w.iter().fold(0.0f64, |a, &v| a + f64::from(v)),
        };
        let top_k = self.top_k as usize;
        let per_group: Vec<f64> = gptr
            .par_windows(2)
            .enumerate()
            .map_init(Vec::new, |rank, (g, win)| {
                let (b, e) = (win[0] as usize, win[1] as usize);
                let (p, y) = (&pred[b..e], &label[b..e]);
                rank_by_score(p, rank);
                match self.kind {
                    RankMetricKind::Ndcg => self.ndcg_group(y, rank, w_of(g), top_k),
                    RankMetricKind::Map => self.map_group(y, rank, w_of(g), top_k),
                    RankMetricKind::Pre => {
                        let n = top_k.min(y.len());
                        let mut hits = 0.0f64;
                        for &r in rank.iter().take(n) {
                            hits += f64::from(y[r] * w_of(g));
                        }
                        hits / n as f64
                    }
                }
            })
            .collect();
        let sum: f64 = per_group.iter().fold(0.0, |a, &v| a + v);
        // `Finalize`
        let mut score = sum;
        if sum_w > 0.0 {
            score /= sum_w;
        }
        score.min(1.0)
    }

    fn ndcg_group(&self, y: &[f32], rank: &[usize], w: f32, top_k: usize) -> f64 {
        let cnt = y.len();
        let discount: Vec<f64> = (0..cnt.min(top_k)).map(dcg_discount).collect();
        // ideal DCG@k
        let mut idx = Vec::new();
        rank_by_label(y, &mut idx);
        let mut idcg = 0.0f64;
        for (i, &r) in idx.iter().enumerate().take(cnt.min(top_k)) {
            idcg += if self.exp_gain {
                discount[i] * dcg_gain(y[r])
            } else {
                discount[i] * f64::from(y[r])
            };
        }
        let inv = inv_idcg_of(idcg);
        if inv <= 0.0 {
            // groups without any relevant document are not weighted (XGBoost quirk)
            return if self.minus { 0.0 } else { 1.0 };
        }
        let n = cnt.min(top_k);
        let mut ndcg = 0.0f64;
        for (i, &r) in rank.iter().enumerate().take(n) {
            ndcg += if self.exp_gain {
                discount[i] * dcg_gain(y[r]) * inv
            } else {
                discount[i] * f64::from(y[r]) * inv
            };
        }
        ndcg * f64::from(w)
    }

    fn map_group(&self, y: &[f32], rank: &[usize], w: f32, top_k: usize) -> f64 {
        let n = top_k.min(y.len());
        let mut n_hits = 0.0f64;
        let mut map = 0.0f64;
        for (i, &r) in rank.iter().enumerate().take(n) {
            let p = f64::from(y[r]);
            n_hits += p;
            map += n_hits / (i + 1) as f64 * p;
        }
        for &r in rank.iter().skip(n) {
            n_hits += f64::from(y[r]);
        }
        if n_hits > 0.0 {
            map /= n_hits.min(top_k as f64);
        } else {
            map = if self.minus { 0.0 } else { 1.0 };
        }
        map * f64::from(w)
    }
}

impl crate::metrics::Metric for RankMetric {
    fn name(&self) -> &str {
        &self.name
    }

    fn eval(&self, pred: &[f32], label: &[f32]) -> f32 {
        self.score(pred, label, None, None) as f32
    }

    fn supports_weights(&self) -> bool {
        true
    }

    fn eval_weighted(&self, pred: &[f32], label: &[f32], weight: Option<&[f32]>) -> f32 {
        self.score(pred, label, weight, None) as f32
    }

    fn eval_grouped(
        &self,
        pred: &[f32],
        label: &[f32],
        weight: Option<&[f32]>,
        group_ptr: Option<&[u32]>,
    ) -> f32 {
        self.score(pred, label, weight, group_ptr) as f32
    }

    fn check_labels(&self, labels: &[f32], what: &str) -> anyhow::Result<()> {
        self.validate_labels(labels, what)
    }

    fn lower_is_better(&self) -> bool {
        false
    }
}

// ------------------------------------------------------------------ auc / aucpr per group

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GroupAucKind {
    Roc,
    Pr,
}

/// `auc` / `aucpr` for data with query groups (XGBoost `RankingAUC`,
/// `src/metric/auc.cc`): every group is scored on its own, groups that cannot be
/// scored are left out, and the score is the mean over the scorable groups.
///
/// * `auc` counts **all pairs** of a group, ordered by label: a pair scores 1 when
///   the model keeps the label order, 0.5 for tied scores, 0 otherwise. Groups with
///   fewer than 3 documents are not scorable. The group weight enters squared in
///   both numerator and denominator, so it only matters through `0`.
/// * `aucpr` is the binary PR-AUC of the group (labels must lie in `[0, 1]`) with
///   the group weight on every document; groups without a positive or without a
///   negative document are not scorable.
///
/// XGBoost uses this path whenever the data has query groups, whatever the
/// objective. `sigmoid` applies the logistic transform first (`binary:logistic`
/// metrics see probabilities); ranking objectives score the raw margin.
#[derive(Clone, Copy, Debug)]
pub struct GroupAuc {
    kind: GroupAucKind,
    sigmoid: bool,
}

impl GroupAuc {
    pub fn new(kind: GroupAucKind, sigmoid: bool) -> Self {
        Self { kind, sigmoid }
    }

    fn name(&self) -> &'static str {
        match self.kind {
            GroupAucKind::Roc => "auc",
            GroupAucKind::Pr => "aucpr",
        }
    }

    /// `NaN` when no group is scorable (or there is no data).
    pub fn score(
        &self,
        pred: &[f32],
        label: &[f32],
        weight: Option<&[f32]>,
        group_ptr: &[u32],
    ) -> f64 {
        if label.is_empty() {
            return f64::NAN;
        }
        let transformed: Vec<f32>;
        let pred = if self.sigmoid {
            transformed = pred
                .iter()
                .map(|&r| crate::metrics::sigmoid_f32(r))
                .collect();
            &transformed[..]
        } else {
            pred
        };
        let per_group: Vec<Option<f64>> = group_ptr
            .par_windows(2)
            .enumerate()
            .map_init(Vec::new, |idx, (g, win)| {
                let (b, e) = (win[0] as usize, win[1] as usize);
                let w = weight.map_or(1.0f32, |w| w[g]);
                let (p, y) = (&pred[b..e], &label[b..e]);
                let auc = match self.kind {
                    GroupAucKind::Roc => {
                        if y.len() < 3 {
                            return None;
                        }
                        roc_group(p, y, w, idx)
                    }
                    GroupAucKind::Pr => pr_group(p, y, w, idx),
                };
                (!auc.is_nan()).then_some(auc)
            })
            .collect();
        let (mut sum, mut valid) = (0.0f64, 0u32);
        for v in per_group.into_iter().flatten() {
            sum += v;
            valid += 1;
        }
        if valid == 0 {
            f64::NAN
        } else {
            sum / f64::from(valid)
        }
    }
}

/// `GroupRankingROC`: all pairs in label order (stable, descending).
fn roc_group(pred: &[f32], label: &[f32], w: f32, idx: &mut Vec<usize>) -> f64 {
    rank_by_label(label, idx);
    // f32 arithmetic where XGBoost has `float`: the squared weight, the score
    // difference and the product that is added to the f64 sum.
    let w2 = w * w;
    let (mut auc, mut sum_w) = (0.0f64, 0.0f64);
    for i in 0..idx.len() {
        let pi = pred[idx[i]];
        for &j in &idx[i + 1..] {
            let d = pi - pred[j];
            let v: f32 = if d > 0.0 {
                1.0
            } else if d == 0.0 {
                0.5
            } else {
                0.0
            };
            auc += f64::from(v * w2);
            sum_w += f64::from(w2);
        }
    }
    if sum_w != 0.0 {
        auc /= sum_w;
    }
    auc
}

/// `BinaryPRAUC` with a constant weight; `NaN` for a group with one class only.
fn pr_group(pred: &[f32], label: &[f32], w: f32, idx: &mut Vec<usize>) -> f64 {
    rank_by_score(pred, idx);
    let (mut total_pos, mut total_neg) = (0.0f64, 0.0f64);
    for &y in label {
        total_pos += f64::from(w * y);
        total_neg += f64::from(w * (1.0f32 - y));
    }
    if total_pos <= 0.0 || total_neg <= 0.0 {
        return f64::NAN;
    }
    let first = idx[0];
    let y = label[first];
    let mut fp = (1.0 - f64::from(y)) * f64::from(w);
    let mut tp = f64::from(y * w);
    let (mut fp_prev, mut tp_prev) = (0.0f64, 0.0f64);
    let mut area = 0.0f64;
    for k in 1..idx.len() {
        let (i, prev) = (idx[k], idx[k - 1]);
        if pred[i] != pred[prev] {
            area += crate::metrics::delta_pr_auc(fp_prev, fp, tp_prev, tp, total_pos);
            fp_prev = fp;
            tp_prev = tp;
        }
        let y = label[i];
        fp += f64::from((1.0f32 - y) * w);
        tp += f64::from(y * w);
    }
    area += crate::metrics::delta_pr_auc(fp_prev, fp, tp_prev, tp, total_pos);
    if fp <= 0.0 || tp <= 0.0 {
        area = 0.0;
    }
    area
}

impl crate::metrics::Metric for GroupAuc {
    fn name(&self) -> &str {
        GroupAuc::name(self)
    }

    fn eval(&self, pred: &[f32], label: &[f32]) -> f32 {
        self.score(pred, label, None, &[0, pred.len() as u32]) as f32
    }

    fn supports_weights(&self) -> bool {
        true
    }

    fn eval_weighted(&self, pred: &[f32], label: &[f32], weight: Option<&[f32]>) -> f32 {
        assert!(
            weight.is_none(),
            "metric {:?} with weights needs query groups",
            GroupAuc::name(self)
        );
        self.eval(pred, label)
    }

    fn eval_grouped(
        &self,
        pred: &[f32],
        label: &[f32],
        weight: Option<&[f32]>,
        group_ptr: Option<&[u32]>,
    ) -> f32 {
        match group_ptr {
            Some(g) => self.score(pred, label, weight, g) as f32,
            None => self.eval_weighted(pred, label, weight),
        }
    }

    fn check_labels(&self, labels: &[f32], what: &str) -> anyhow::Result<()> {
        if self.kind == GroupAucKind::Pr {
            crate::ensure_input!(
                labels.iter().all(|&y| (0.0..=1.0).contains(&y)),
                "{what}: aucpr supports only binary relevance for learning to rank \
                 (labels in [0, 1])"
            );
        }
        Ok(())
    }

    fn lower_is_better(&self) -> bool {
        false
    }
}

// ------------------------------------------------------------------ input helpers

/// Group offsets from `qid` or `group=` sizes (exactly one may be given).
/// `None` when neither is given.
pub fn group_ptr_from_inputs(
    qid: Option<&[u64]>,
    group: Option<&[u64]>,
    n_rows: usize,
) -> anyhow::Result<Option<Vec<u32>>> {
    match (qid, group) {
        (Some(_), Some(_)) => crate::bail_input!("Either one of `group` or `qid` should be None."),
        (Some(q), None) => {
            crate::ensure_input!(
                q.len() == n_rows,
                "qid has {} entries, the data has {n_rows} rows",
                q.len()
            );
            group_ptr_from_qid(q).map(Some)
        }
        (None, Some(g)) => group_ptr_from_sizes(g, n_rows).map(Some),
        (None, None) => Ok(None),
    }
}

/// Per-row weights for the quantile sketch from per-group weights
/// (XGBoost unrolls group weights onto the rows for ranking).
pub fn unroll_group_weights(group_weights: &[f32], group_ptr: &[u32]) -> Vec<f32> {
    let mut out = Vec::with_capacity(*group_ptr.last().unwrap_or(&0) as usize);
    for (g, w) in group_ptr.windows(2).enumerate() {
        out.extend(std::iter::repeat_n(
            group_weights[g],
            (w[1] - w[0]) as usize,
        ));
    }
    out
}

/// A file weight column holds one value per row; for ranking it must be the group
/// weight repeated on every row of the group. Returns one weight per group.
pub fn group_weights_from_rows(
    row_weights: &[f32],
    group_ptr: &[u32],
    column: &str,
) -> anyhow::Result<Vec<f32>> {
    let mut out = Vec::with_capacity(group_ptr.len().saturating_sub(1));
    for (g, w) in group_ptr.windows(2).enumerate() {
        let rows = &row_weights[w[0] as usize..w[1] as usize];
        let first = rows[0];
        if let Some(k) = rows.iter().position(|&v| v != first) {
            crate::bail_input!(
                "weight column {column:?}: ranking weights are per query group, but group {g} \
                 has different weights on its rows (row {} has {first}, row {} has {})",
                w[0],
                w[0] as usize + k,
                rows[k]
            );
        }
        out.push(first);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metrics::Metric;

    #[test]
    fn mt19937_matches_the_reference_sequence() {
        // std::mt19937(5489) first output is 3499211612; the 10000th is 4123659995.
        let mut r = Mt19937::new(5489);
        assert_eq!(r.next_u32(), 3_499_211_612);
        for _ in 0..9998 {
            r.next_u32();
        }
        assert_eq!(r.next_u32(), 4_123_659_995);
    }

    #[test]
    fn minstd_matches_the_reference_sequence() {
        // std::minstd_rand(1): the 10000th output is 399268537.
        let mut r = MinStd::new(1);
        let mut last = 0;
        for _ in 0..10000 {
            last = r.next();
        }
        assert_eq!(last, 399_268_537);
        // seed 0 is mapped to 1
        assert_eq!(MinStd::new(0).next(), MinStd::new(1).next());
    }

    #[test]
    fn qid_runs_become_groups() {
        assert_eq!(
            group_ptr_from_qid(&[3, 3, 3, 7, 9, 9]).unwrap(),
            vec![0, 3, 4, 6]
        );
        let err = group_ptr_from_qid(&[1, 2, 1]).unwrap_err().to_string();
        assert!(err.contains("non-decreasing"), "{err}");
        assert_eq!(group_ptr_from_sizes(&[2, 1], 3).unwrap(), vec![0, 2, 3]);
        assert!(group_ptr_from_sizes(&[2, 0, 1], 3).is_err());
        assert!(group_ptr_from_sizes(&[2, 2], 3).is_err());
        assert!(group_ptr_from_sizes(&[1, 1], 3).is_err());
    }

    #[test]
    fn default_params_follow_xgboost() {
        let p = RankParams::default();
        assert_eq!(p.num_pair(), 32);
        assert_eq!(p.top_k(), 32);
        let mean = RankParams {
            pair_method: PairMethod::Mean,
            ..p
        };
        assert_eq!(mean.num_pair(), 1);
        assert_eq!(mean.top_k(), NOT_SET);
    }

    #[test]
    fn groups_of_one_and_equal_labels_give_zero_gradients() {
        let labels = [1.0f32, 2.0, 2.0, 2.0];
        let ptr = [0u32, 1, 4];
        let g =
            RankGradient::new(RankKind::Ndcg, RankParams::default(), &labels, &ptr, None).unwrap();
        let mut out = Vec::new();
        g.gradients(&[0.3, 0.1, 0.2, 0.4], 0, &mut out);
        assert!(out.iter().all(|p| p.grad == 0.0 && p.hess == 0.0));
    }

    #[test]
    fn lambda_gradients_sum_to_zero_within_a_group() {
        let labels = [0.0f32, 1.0, 2.0, 0.0, 3.0];
        let ptr = [0u32, 5];
        for kind in [RankKind::Ndcg, RankKind::Pairwise] {
            let g = RankGradient::new(kind, RankParams::default(), &labels, &ptr, None).unwrap();
            let mut out = Vec::new();
            g.gradients(&[0.1, -0.2, 0.3, 0.0, 0.05], 0, &mut out);
            let s: f64 = out.iter().map(|p| f64::from(p.grad)).sum();
            assert!(s.abs() < 1e-5, "{kind:?}: {s}");
            assert!(out.iter().any(|p| p.hess > 0.0));
        }
    }

    #[test]
    fn mean_pairs_are_deterministic_and_seed_dependent() {
        let labels: Vec<f32> = (0..20).map(|i| (i % 4) as f32).collect();
        let ptr = [0u32, 20];
        let params = RankParams {
            pair_method: PairMethod::Mean,
            ..RankParams::default()
        };
        let g = RankGradient::new(RankKind::Ndcg, params, &labels, &ptr, None).unwrap();
        let preds: Vec<f32> = (0..20).map(|i| ((i * 7) % 11) as f32 * 0.1).collect();
        let (mut a, mut b, mut c) = (Vec::new(), Vec::new(), Vec::new());
        g.gradients(&preds, 5, &mut a);
        g.gradients(&preds, 5, &mut b);
        g.gradients(&preds, 6, &mut c);
        assert!(a
            .iter()
            .zip(&b)
            .all(|(x, y)| x.grad == y.grad && x.hess == y.hess));
        assert!(a.iter().zip(&c).any(|(x, y)| x.grad != y.grad));
    }

    #[test]
    fn ndcg_metric_matches_a_hand_computation() {
        // one group: scores rank the docs [label 1, label 3, label 0]
        let m = RankMetric::parse("ndcg@3", true).unwrap();
        let v = f64::from(m.eval(&[0.5, 0.9, 0.1], &[1.0, 3.0, 0.0]));
        let dcg = 7.0 / 2.0f64.log2() + 1.0 / 3.0f64.log2();
        let idcg = 7.0 / 2.0f64.log2() + 1.0 / 3.0f64.log2();
        // perfect order -> 1
        assert!((v - dcg / idcg).abs() < 1e-6);
        let worse = f64::from(m.eval(&[0.9, 0.5, 0.1], &[1.0, 3.0, 0.0]));
        let dcg2 = 1.0 / 2.0f64.log2() + 7.0 / 3.0f64.log2();
        assert!((worse - dcg2 / idcg).abs() < 1e-6);
        // all-zero group counts as a perfect score
        assert_eq!(m.eval(&[0.1, 0.2], &[0.0, 0.0]), 1.0);
        assert_eq!(
            RankMetric::parse("ndcg-", true)
                .unwrap()
                .eval(&[0.1, 0.2], &[0.0, 0.0]),
            0.0
        );
    }

    #[test]
    fn metric_names_follow_xgboost() {
        assert!(RankMetric::parse("ndcg@0", true).is_err());
        assert!(RankMetric::parse("ndcg@x", true).is_err());
        assert!(RankMetric::parse("auc", true).is_err());
        let d = RankMetric::default_for(RankKind::Ndcg, &RankParams::default());
        assert_eq!(d.name(), "ndcg@32");
        let mean = RankParams {
            pair_method: PairMethod::Mean,
            ..RankParams::default()
        };
        assert_eq!(
            RankMetric::default_for(RankKind::Pairwise, &mean).name(),
            "ndcg"
        );
        assert_eq!(RankMetric::default_for(RankKind::Map, &mean).name(), "map");
        let five = RankParams {
            num_pair_per_sample: Some(5),
            ..RankParams::default()
        };
        assert_eq!(
            RankMetric::default_for(RankKind::Map, &five).name(),
            "map@5"
        );
        assert!(!d.lower_is_better());
    }

    // ---------------------------------------------------------------- unbiased

    fn unbiased_fixture() -> (Vec<f32>, Vec<u32>, Vec<f32>) {
        // two groups, clicks ordered by display position
        let labels = vec![1.0f32, 0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0];
        let ptr = vec![0u32, 4, 9];
        let preds = vec![0.3f32, 0.1, 0.2, 0.4, 0.15, -0.1, 0.5, 0.0, 0.25];
        (labels, ptr, preds)
    }

    #[test]
    fn position_bias_sizes_follow_xgboost() {
        let p = RankParams::default();
        assert_eq!(p.max_position_size(5), 32);
        assert_eq!(p.max_position_size(500), 32);
        let k3 = RankParams {
            num_pair_per_sample: Some(3),
            ..p
        };
        assert_eq!(k3.max_position_size(500), 3);
        let mean = RankParams {
            pair_method: PairMethod::Mean,
            ..p
        };
        assert_eq!(mean.max_position_size(5), 5);
        assert_eq!(mean.max_position_size(500), 32);
        assert_eq!(p.regularizer(), 0.5);
        let q = RankParams {
            bias_norm: 0.0,
            ..p
        };
        assert_eq!(q.regularizer(), 1.0);
    }

    #[test]
    fn unbiased_first_round_is_the_plain_gradient_and_the_bias_moves_afterwards() {
        let (labels, ptr, preds) = unbiased_fixture();
        let plain = RankGradient::new(
            RankKind::Pairwise,
            RankParams::default(),
            &labels,
            &ptr,
            None,
        )
        .unwrap();
        assert!(plain.position_bias().is_none());
        let params = RankParams {
            unbiased: true,
            ..RankParams::default()
        };
        let g = RankGradient::new(RankKind::Pairwise, params, &labels, &ptr, None).unwrap();
        let (ti, tj) = g.position_bias().unwrap();
        assert_eq!((ti.len(), tj.len()), (32, 32));
        assert!(ti.iter().chain(&tj).all(|&v| v == 1.0));
        let (mut a, mut b) = (Vec::new(), Vec::new());
        plain.gradients(&preds, 0, &mut a);
        g.gradients(&preds, 0, &mut b);
        // the biases are all ones, so the division is exact: the same gradients
        assert!(a
            .iter()
            .zip(&b)
            .all(|(x, y)| x.grad == y.grad && x.hess == y.hess));
        // the update: position 0 is the reference, the others are ratios in (0, inf)
        let (ti, tj) = g.position_bias().unwrap();
        assert_eq!((ti[0], tj[0]), (1.0, 1.0));
        assert!(
            ti[1..4].iter().chain(&tj[1..4]).any(|&v| v != 1.0),
            "{ti:?} {tj:?}"
        );
        // positions never seen in any group keep a zero loss -> bias 0
        assert!(ti[4..].iter().all(|&v| v == 0.0), "{ti:?}");
        // the next round uses them
        g.gradients(&preds, 0, &mut b);
        assert!(a.iter().zip(&b).any(|(x, y)| x.grad != y.grad));
    }

    #[test]
    fn base_score_estimation_leaves_the_bias_state_alone() {
        let (labels, ptr, _) = unbiased_fixture();
        let params = RankParams {
            unbiased: true,
            ..RankParams::default()
        };
        let g = RankGradient::new(RankKind::Ndcg, params, &labels, &ptr, None).unwrap();
        let plain =
            RankGradient::new(RankKind::Ndcg, RankParams::default(), &labels, &ptr, None).unwrap();
        // the estimate uses all-ones biases (a throwaway copy in XGBoost)
        assert_eq!(estimate_base_score(&g, 0), estimate_base_score(&plain, 0));
        let (ti, tj) = g.position_bias().unwrap();
        assert!(ti.iter().chain(&tj).all(|&v| v == 1.0));
    }

    #[test]
    fn unbiased_gradients_do_not_depend_on_threads() {
        let sizes: Vec<u32> = (0..40).map(|i| 2 + (i * 7) % 23).collect();
        let mut ptr = vec![0u32];
        for s in &sizes {
            ptr.push(ptr.last().unwrap() + s);
        }
        let n = *ptr.last().unwrap() as usize;
        let labels: Vec<f32> = (0..n).map(|i| ((i * 5) % 3 == 0) as u8 as f32).collect();
        let params = RankParams {
            unbiased: true,
            ..RankParams::default()
        };
        let run = |threads: usize| {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            pool.install(|| {
                let g = RankGradient::new(RankKind::Ndcg, params, &labels, &ptr, None).unwrap();
                let mut out = Vec::new();
                let mut last = Vec::new();
                for r in 0..4 {
                    let preds: Vec<f32> = (0..n)
                        .map(|i| ((i * 13 + r * 7) % 17) as f32 * 0.1)
                        .collect();
                    g.gradients(&preds, r as u32, &mut out);
                    last = out.clone();
                }
                (last, g.position_bias().unwrap())
            })
        };
        let (a, ba) = run(1);
        let (b, bb) = run(4);
        assert!(a
            .iter()
            .zip(&b)
            .all(|(x, y)| x.grad == y.grad && x.hess == y.hess));
        assert_eq!(ba, bb);
    }

    // ---------------------------------------------------------------- auc / aucpr per group

    #[test]
    fn group_roc_auc_counts_all_pairs_of_each_group() {
        // group 0: labels 2 > 1 > 0, scores keep the order of two of the three pairs
        // group 1: two documents, no ROC AUC
        let label = [2.0f32, 1.0, 0.0, 1.0, 0.0];
        let pred = [0.9f32, 0.1, 0.5, 0.3, 0.2];
        let ptr = [0u32, 3, 5];
        let m = GroupAuc::new(GroupAucKind::Roc, false);
        assert!((m.score(&pred, &label, None, &ptr) - 2.0 / 3.0).abs() < 1e-15);
        // equal scores count one half
        let tied = [0.5f32; 5];
        assert!((m.score(&tied, &label, None, &ptr) - 0.5).abs() < 1e-15);
        // the weight is squared in both sums: it only matters through zero
        let w = [3.0f32, 0.5];
        assert!((m.score(&pred, &label, Some(&w), &ptr) - 2.0 / 3.0).abs() < 1e-15);
        // a zero-weight group is scorable (and scores 0), so the mean is over both
        let w0 = [0.0f32, 1.0];
        assert_eq!(m.score(&pred, &label, Some(&w0), &ptr), 0.0);
        // no data / only small groups -> NaN
        assert!(m.score(&[], &[], None, &[0]).is_nan());
        assert!(m.score(&pred[3..], &label[3..], None, &[0, 2]).is_nan());
    }

    #[test]
    fn group_pr_auc_needs_both_classes_in_a_group() {
        // group 0 has positives and negatives; group 1 only negatives; group 2 only positives
        let label = [1.0f32, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0];
        let pred = [0.9f32, 0.8, 0.7, 0.1, 0.5, 0.4, 0.3, 0.2];
        let ptr = [0u32, 4, 6, 8];
        let m = GroupAuc::new(GroupAucKind::Pr, false);
        let all = m.score(&pred, &label, None, &ptr);
        let first = m.score(&pred[..4], &label[..4], None, &[0, 4]);
        assert!(first > 0.0 && first <= 1.0);
        assert_eq!(all, first, "groups with one class are left out of the mean");
        assert!(m.score(&pred[4..6], &label[4..6], None, &[0, 2]).is_nan());
        // labels must be binary relevance
        use crate::metrics::Metric;
        assert!(m.check_labels(&[0.0, 2.0], "t").is_err());
        assert!(m.check_labels(&[0.0, 1.0, 0.5], "t").is_ok());
    }

    #[test]
    fn logistic_group_auc_scores_probabilities() {
        use crate::metrics::{Auc, Metric};
        let label = [1.0f32, 0.0, 1.0, 0.0, 1.0, 0.0];
        // margins far apart saturate the f32 sigmoid to the same probability
        let raw = [40.0f32, 30.0, 0.2, -0.1, 0.0, 0.5];
        let ptr = [0u32, 3, 6];
        let grouped = Auc.eval_grouped(&raw, &label, None, Some(&ptr));
        let prob: Vec<f32> = raw
            .iter()
            .map(|&r| crate::metrics::sigmoid_f32(r))
            .collect();
        let want = GroupAuc::new(GroupAucKind::Roc, false).score(&prob, &label, None, &ptr);
        assert_eq!(f64::from(grouped), f64::from(want as f32));
        // without groups the binary AUC is used
        let flat = Auc.eval_grouped(&raw, &label, None, None);
        assert_eq!(flat, Auc.eval(&raw, &label));
    }
}
