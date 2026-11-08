//! Feature importance, SHAP contributions, SHAP interaction values and leaf
//! indices, with the semantics (and, as far as float32 allows, the arithmetic)
//! of XGBoost 3.4.1.
//!
//! * Importance (`feature_scores`) mirrors `GBTree::FeatureScore`: split nodes
//!   are walked in XGBoost's stack order (right child first) and accumulated
//!   in float32, so even the rounding of `gain` / `cover` sums matches.
//! * Exact contributions and interactions use the fixed 8-point
//!   QuadratureTreeSHAP that XGBoost 3.4.1 uses on the CPU (not the classic
//!   path-extension recurrence): same Gauss-Legendre rule, same float32
//!   operation order, same cover handling for degenerate (zero-cover) nodes.
//! * `approx` contributions are the Saabas method (`approx_contribs=True`).
//! * Leaf indices are node ids in the tree arrays, which is how the JSON model
//!   numbers them.
//!
//! This is the host implementation: row-parallel, each row computed serially in
//! model order, so the result does not depend on the worker count. The device
//! kernels (`backend::cuda::explain`) replay this exact float32 operation order
//! per row and group and are bit-identical to it.

// Index loops over the fixed 8-point buffers deliberately mirror XGBoost's
// operation order, which is what keeps the float32 results identical.
#![allow(clippy::needless_range_loop)]

use std::sync::OnceLock;

use anyhow::{bail, Context, Result};
use rayon::prelude::*;

use crate::tree::{Model, Node, Tree};

/// Number of Gauss-Legendre points. Fixed, as in XGBoost's CPU path.
const POINTS: usize = 8;
/// Marks "feature not on the current path" in the per-feature path
/// probability table.
const UNSEEN: f32 = -999.0;
/// Floor for a child's share of its parent's cover (zero-cover children can
/// appear in refreshed or externally produced trees).
const MIN_CHILD_WEIGHT: f32 = 1e-12;

type Buf = [f32; POINTS];

// ---------------------------------------------------------------------------
// Feature importance
// ---------------------------------------------------------------------------

/// `Booster.get_score(importance_type=...)` kinds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Importance {
    Weight,
    Gain,
    Cover,
    TotalGain,
    TotalCover,
}

impl Importance {
    pub fn parse(name: &str) -> Option<Self> {
        match name {
            "weight" => Some(Self::Weight),
            "gain" => Some(Self::Gain),
            "cover" => Some(Self::Cover),
            "total_gain" => Some(Self::TotalGain),
            "total_cover" => Some(Self::TotalCover),
            _ => None,
        }
    }
}

/// Per-feature importance of the whole ensemble: `(feature index, score)` for
/// every feature that is used in at least one split, in feature order
/// (XGBoost omits zero-importance features).
pub fn feature_scores(model: &Model, kind: Importance) -> Vec<(usize, f32)> {
    let n = model.n_features;
    let mut split_counts = vec![0usize; n];
    let mut score = vec![0.0f32; n];
    for tree in &model.trees {
        if tree.nodes.is_empty() {
            continue;
        }
        // XGBoost's WalkTree: pop a node, then push left and right, so the
        // right subtree is visited first. The order only matters for float32
        // rounding of the running sums.
        let mut stack = vec![0usize];
        while let Some(i) = stack.pop() {
            let node = &tree.nodes[i];
            if node.is_leaf {
                continue;
            }
            let f = node.feat as usize;
            split_counts[f] += 1;
            match kind {
                Importance::Weight => score[f] = split_counts[f] as f32,
                Importance::Gain | Importance::TotalGain => score[f] += node.gain,
                Importance::Cover | Importance::TotalCover => score[f] += node.sum_hess,
            }
            stack.push(node.left as usize);
            stack.push(node.right as usize);
        }
    }
    if matches!(kind, Importance::Gain | Importance::Cover) {
        for (s, &c) in score.iter_mut().zip(&split_counts) {
            *s /= 1.0f32.max(c as f32);
        }
    }
    (0..n)
        .filter(|&f| split_counts[f] != 0)
        .map(|f| (f, score[f]))
        .collect()
}

// ---------------------------------------------------------------------------
// Quadrature rule
// ---------------------------------------------------------------------------

struct Rule {
    nodes: Buf,
    weights: Buf,
}

fn legendre(n: usize, x: f64) -> f64 {
    let p0 = 1.0;
    if n == 0 {
        return p0;
    }
    let (mut p0, mut p1) = (p0, x);
    for k in 2..=n {
        let kf = k as f64;
        let pk = ((2.0 * kf - 1.0) * x * p1 - (kf - 1.0) * p0) / kf;
        p0 = p1;
        p1 = pk;
    }
    p1
}

fn legendre_derivative(n: usize, x: f64, pn: f64) -> f64 {
    n as f64 * (x * pn - legendre(n - 1, x)) / (x * x - 1.0)
}

/// Endpoint-squared Gauss-Legendre rule, a port of `MakeEndpointQuadrature`.
fn make_rule() -> Rule {
    let n = POINTS;
    let mut rule = Rule {
        nodes: [0.0; POINTS],
        weights: [0.0; POINTS],
    };
    for i in 0..n {
        let theta = std::f64::consts::PI * (i as f64 + 0.75) / (n as f64 + 0.5);
        let mut x = theta.cos();
        for _ in 0..64 {
            let pn = legendre(n, x);
            let dpn = legendre_derivative(n, x, pn);
            let dx = pn / dpn;
            x -= dx;
            if dx.abs() < 1e-15 {
                break;
            }
        }
        let pn = legendre(n, x);
        let dpn = legendre_derivative(n, x, pn);
        let w = 2.0 / ((1.0 - x * x) * dpn * dpn);
        let s = 0.5 * (x + 1.0);
        let ws = 0.5 * w;
        let out = n - 1 - i;
        rule.nodes[out] = (s * s) as f32;
        rule.weights[out] = (2.0 * s * ws) as f32;
    }
    rule
}

fn rule() -> &'static Rule {
    static RULE: OnceLock<Rule> = OnceLock::new();
    RULE.get_or_init(make_rule)
}

/// The quadrature nodes then weights, as the 16 floats the device kernels read.
#[cfg(feature = "cuda")]
pub(crate) fn rule_values() -> [f32; 2 * POINTS] {
    let r = rule();
    let mut out = [0.0f32; 2 * POINTS];
    out[..POINTS].copy_from_slice(&r.nodes);
    out[POINTS..].copy_from_slice(&r.weights);
    out
}

pub(crate) fn branch_weight(cover: f32, parent_cover: f32) -> f32 {
    if parent_cover <= 0.0 {
        return 0.5;
    }
    let weight = cover / parent_cover;
    if weight < MIN_CHILD_WEIGHT {
        MIN_CHILD_WEIGHT
    } else {
        weight
    }
}

// ---------------------------------------------------------------------------
// Quadrature TreeSHAP runner
// ---------------------------------------------------------------------------

fn extract_delta(rule: &Rule, h: &Buf, p_enter: f32, p_exit: f32) -> f32 {
    let mut acc = 0.0f32;
    if p_enter != 1.0 {
        let alpha = p_enter - 1.0;
        for i in 0..POINTS {
            acc += alpha * h[i] / (1.0 + alpha * rule.nodes[i]);
        }
    }
    if p_exit != 1.0 {
        let alpha = p_exit - 1.0;
        for i in 0..POINTS {
            acc -= alpha * h[i] / (1.0 + alpha * rule.nodes[i]);
        }
    }
    acc
}

fn extract_interaction_delta(
    rule: &Rule,
    h: &Buf,
    p_enter: f32,
    p_exit: f32,
    q_partner: f32,
) -> f32 {
    if q_partner == 1.0 {
        return 0.0;
    }
    let alpha_partner = q_partner - 1.0;
    let alpha_enter = p_enter - 1.0;
    let mut acc = 0.0f32;
    if p_exit == 1.0 {
        for i in 0..POINTS {
            let edge = alpha_enter / (1.0 + alpha_enter * rule.nodes[i]);
            acc += alpha_partner * h[i] * edge / (1.0 + alpha_partner * rule.nodes[i]);
        }
    } else {
        let alpha_exit = p_exit - 1.0;
        for i in 0..POINTS {
            let edge = alpha_enter / (1.0 + alpha_enter * rule.nodes[i])
                - alpha_exit / (1.0 + alpha_exit * rule.nodes[i]);
            acc += alpha_partner * h[i] * edge / (1.0 + alpha_partner * rule.nodes[i]);
        }
    }
    acc
}

/// Where the runner writes its per-edge results (additive or interaction).
trait Formulation {
    fn push(&mut self, feat: usize, p_child: f32);
    fn pop(&mut self);
    fn on_return(&mut self, rule: &Rule, feat: usize, h: &Buf, p_enter: f32, p_exit: f32);
}

struct Additive<'a> {
    phi: &'a mut [f32],
}

impl Formulation for Additive<'_> {
    fn push(&mut self, _: usize, _: f32) {}
    fn pop(&mut self) {}
    fn on_return(&mut self, rule: &Rule, feat: usize, h: &Buf, p_enter: f32, p_exit: f32) {
        self.phi[feat] += extract_delta(rule, h, p_enter, p_exit);
    }
}

struct Interaction<'a> {
    path: &'a mut Vec<(usize, f32)>,
    diag: &'a mut [f32],
    matrix: &'a mut [f32],
    ncols: usize,
}

impl Formulation for Interaction<'_> {
    fn push(&mut self, feat: usize, p_child: f32) {
        self.path.push((feat, p_child));
    }
    fn pop(&mut self) {
        self.path.pop();
    }
    fn on_return(&mut self, rule: &Rule, feat: usize, h: &Buf, p_enter: f32, p_exit: f32) {
        self.diag[feat] += extract_delta(rule, h, p_enter, p_exit);
        // Partners: the newest occurrence of every other feature on the live
        // path, newest to oldest (the current split is the newest entry and is
        // skipped; older duplicates of any feature are shadowed).
        let len = self.path.len();
        let mut skipped_current = false;
        for i in (0..len).rev() {
            let (split, p_child) = self.path[i];
            if self.path[i + 1..].iter().any(|&(s, _)| s == split) {
                continue;
            }
            if !skipped_current && split == feat {
                skipped_current = true;
                continue;
            }
            let delta = extract_interaction_delta(rule, h, p_enter, p_exit, p_child);
            self.matrix[feat * self.ncols + split] += delta;
        }
    }
}

struct Runner<'a, F: Formulation> {
    nodes: &'a [Node],
    feats: &'a [f32],
    rule: &'a Rule,
    path_prob: &'a mut [f32],
    form: F,
}

impl<F: Formulation> Runner<'_, F> {
    fn goes_left(&self, node: &Node) -> bool {
        // Missing follows the default direction; categorical nodes test their
        // right-going set (XGBoost's `GetNextNode`).
        node.goes_left(self.feats[node.feat as usize])
    }

    #[allow(clippy::too_many_arguments)]
    fn visit_child(
        &mut self,
        split_node: usize,
        child: usize,
        child_weight: f32,
        satisfies: bool,
        c_vals: &Buf,
        w_prod: f32,
        out_h: &mut Buf,
    ) {
        let feat = self.nodes[split_node].feat as usize;
        let p_old = self.path_prob[feat];
        let p_e = if satisfies {
            if p_old == UNSEEN {
                1.0 / child_weight
            } else {
                p_old / child_weight
            }
        } else {
            0.0
        };
        let rule = self.rule;
        let mut c_child = *c_vals;
        let alpha_e = p_e - 1.0;
        for i in 0..POINTS {
            c_child[i] *= 1.0 + alpha_e * rule.nodes[i];
        }
        if p_old != UNSEEN {
            let alpha_old = p_old - 1.0;
            if alpha_old != 0.0 {
                for i in 0..POINTS {
                    c_child[i] /= 1.0 + alpha_old * rule.nodes[i];
                }
            }
        }
        self.path_prob[feat] = p_e;
        self.form.push(feat, p_e);
        self.run_node(child, &c_child, w_prod * child_weight, out_h);
        let p_exit = if p_old == UNSEEN { 1.0 } else { p_old };
        self.form.on_return(rule, feat, out_h, p_e, p_exit);
        self.form.pop();
        self.path_prob[feat] = p_old;
    }

    fn run_node(&mut self, idx: usize, c_vals: &Buf, w_prod: f32, out_h: &mut Buf) {
        let nodes = self.nodes;
        let node = &nodes[idx];
        if node.is_leaf {
            let leaf_scale = w_prod * node.leaf_value;
            for i in 0..POINTS {
                out_h[i] = c_vals[i] * leaf_scale * self.rule.weights[i];
            }
            return;
        }
        let (left, right) = (node.left as usize, node.right as usize);
        let left_weight = branch_weight(nodes[left].sum_hess, node.sum_hess);
        let right_weight = branch_weight(nodes[right].sum_hess, node.sum_hess);
        let goes_left = self.goes_left(node);
        let mut right_h = [0.0f32; POINTS];
        self.visit_child(idx, left, left_weight, goes_left, c_vals, w_prod, out_h);
        self.visit_child(
            idx,
            right,
            right_weight,
            !goes_left,
            c_vals,
            w_prod,
            &mut right_h,
        );
        for i in 0..POINTS {
            out_h[i] += right_h[i];
        }
    }

    fn run(&mut self) {
        if self.nodes[0].is_leaf {
            return;
        }
        let ones = [1.0f32; POINTS];
        let mut h = [0.0f32; POINTS];
        self.run_node(0, &ones, 1.0, &mut h);
    }
}

// ---------------------------------------------------------------------------
// Saabas (approx_contribs) helpers
// ---------------------------------------------------------------------------

/// Cover-weighted mean leaf value of every node, in float32 like XGBoost's
/// `FillNodeMeanValues`.
pub(crate) fn fill_node_means(nodes: &[Node], idx: usize, means: &mut [f32]) -> f32 {
    let node = &nodes[idx];
    let value = if node.is_leaf {
        node.leaf_value
    } else {
        let (l, r) = (node.left as usize, node.right as usize);
        let mut result = fill_node_means(nodes, l, means) * nodes[l].sum_hess;
        result += fill_node_means(nodes, r, means) * nodes[r].sum_hess;
        result / node.sum_hess
    };
    means[idx] = value;
    value
}

/// Cover-weighted root mean in double precision (`FillRootMeanValue`).
fn root_mean(nodes: &[Node], idx: usize) -> f64 {
    let node = &nodes[idx];
    if node.is_leaf {
        return node.leaf_value as f64;
    }
    let (l, r) = (node.left as usize, node.right as usize);
    let parent_cover = node.sum_hess;
    let left_mean = root_mean(nodes, l);
    let right_mean = root_mean(nodes, r);
    if parent_cover == 0.0 {
        return 0.5 * (left_mean + right_mean);
    }
    (left_mean * nodes[l].sum_hess as f64 + right_mean * nodes[r].sum_hess as f64)
        / parent_cover as f64
}

fn check_covers(tree: &Tree, index: usize) -> Result<()> {
    for node in tree.nodes.iter().filter(|n| !n.is_leaf) {
        if node.sum_hess < 0.0
            || tree.nodes[node.left as usize].sum_hess < 0.0
            || tree.nodes[node.right as usize].sum_hess < 0.0
        {
            crate::bail_input!(
                "tree {index} has a negative cover (sum_hessian) at a split: SHAP is undefined"
            );
        }
        if node.sum_hess.is_nan()
            || tree.nodes[node.left as usize].sum_hess.is_nan()
            || tree.nodes[node.right as usize].sum_hess.is_nan()
        {
            crate::bail_input!("tree {index} has a NaN cover (sum_hessian): SHAP is undefined");
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Explainer
// ---------------------------------------------------------------------------

/// What to compute for each row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// `(n_features + 1)` values per output group, bias last. `approx`
    /// selects Saabas instead of TreeSHAP.
    Contribs { approx: bool },
    /// `(n_features + 1)^2` values per output group.
    Interactions,
    /// One node id per tree.
    Leaf,
}

/// A model prefix prepared for explanation. Cheap to build (linear in the
/// number of nodes) and shared read-only by all workers.
pub struct Explainer<'a> {
    model: &'a Model,
    n_trees: usize,
    k: usize,
    n_features: usize,
    /// Margin-space base score of each output group.
    group_bias: Vec<f32>,
    /// Sum over the group's trees of the cover-weighted root mean (exact SHAP
    /// bias term).
    group_root_mean: Vec<f32>,
    /// Per-tree node means (Saabas); empty unless built for approx.
    approx_means: Vec<Vec<f32>>,
}

struct Scratch {
    path_prob: Vec<f32>,
    tree_contribs: Vec<f32>,
    path: Vec<(usize, f32)>,
    diag: Vec<f32>,
}

impl<'a> Explainer<'a> {
    /// Prepare the first `n_trees` trees of `model` (clamped to the model).
    pub fn new(model: &'a Model, n_trees: usize, kind: Kind) -> Result<Self> {
        let n_trees = n_trees.min(model.trees.len());
        let k = model.num_outputs();
        if model.n_features == 0 {
            bail!("model has no features");
        }
        let group_bias: Vec<f32> = if model.class_base_margins.is_empty() {
            vec![crate::train::base_margin(model.objective, model.base_score)]
        } else {
            model.class_base_margins.clone()
        };
        let mut group_root_mean = vec![0.0f32; k];
        let mut approx_means = Vec::new();
        match kind {
            Kind::Leaf => {}
            Kind::Contribs { approx: true } => {
                approx_means.reserve(n_trees);
                for (i, tree) in model.trees[..n_trees].iter().enumerate() {
                    if tree.nodes.is_empty() {
                        bail!("tree {i} has no nodes");
                    }
                    let mut means = vec![0.0f32; tree.nodes.len()];
                    fill_node_means(&tree.nodes, 0, &mut means);
                    approx_means.push(means);
                }
            }
            Kind::Contribs { approx: false } | Kind::Interactions => {
                let mut sums = vec![0.0f64; k];
                for (i, tree) in model.trees[..n_trees].iter().enumerate() {
                    if tree.nodes.is_empty() {
                        bail!("tree {i} has no nodes");
                    }
                    check_covers(tree, i)?;
                    sums[i % k] += root_mean(&tree.nodes, 0);
                }
                for (dst, s) in group_root_mean.iter_mut().zip(sums) {
                    *dst = s as f32;
                }
            }
        }
        Ok(Self {
            model,
            n_trees,
            k,
            n_features: model.n_features,
            group_bias,
            group_root_mean,
            approx_means,
        })
    }

    /// Trees explained (the prefix length).
    pub fn n_trees(&self) -> usize {
        self.n_trees
    }

    /// Number of output groups.
    pub fn groups(&self) -> usize {
        self.k
    }

    /// Margin-space base score of each output group.
    #[cfg(feature = "cuda")]
    pub(crate) fn group_bias(&self) -> &[f32] {
        &self.group_bias
    }

    /// Cover-weighted root mean of each group (the exact-SHAP bias term).
    #[cfg(feature = "cuda")]
    pub(crate) fn group_root_mean(&self) -> &[f32] {
        &self.group_root_mean
    }

    /// Number of f32 values produced per input row.
    pub fn row_width(&self, kind: Kind) -> Result<usize> {
        let nc = self.n_features + 1;
        let w = match kind {
            Kind::Contribs { .. } => nc.checked_mul(self.k),
            Kind::Interactions => nc.checked_mul(nc).and_then(|m| m.checked_mul(self.k)),
            Kind::Leaf => Some(self.n_trees),
        };
        w.context("output size overflow")
    }

    fn scratch(&self) -> Scratch {
        Scratch {
            path_prob: vec![UNSEEN; self.n_features],
            tree_contribs: vec![0.0; self.n_features + 1],
            path: Vec::new(),
            diag: vec![0.0; self.n_features + 1],
        }
    }

    /// Compute `kind` for `n_rows` row-major rows (`n_features` values each).
    /// `pool` is built lazily when `workers > 1` and can be reused across
    /// batches.
    pub fn compute(
        &self,
        kind: Kind,
        rows: &[f32],
        n_rows: usize,
        workers: usize,
        pool: &mut Option<std::sync::Arc<rayon::ThreadPool>>,
    ) -> Result<Vec<f32>> {
        self.compute_with_margin(kind, rows, n_rows, workers, pool, None)
    }

    /// [`Self::compute`] with an optional per-row `base_margin` (row-major
    /// `n_rows x num_outputs`) standing in for the model intercept in the bias
    /// column, as XGBoost does: the bias is `root means + base_margin[row, g]`
    /// instead of `root means + base_score`.
    pub fn compute_with_margin(
        &self,
        kind: Kind,
        rows: &[f32],
        n_rows: usize,
        workers: usize,
        pool: &mut Option<std::sync::Arc<rayon::ThreadPool>>,
        margin: Option<&[f32]>,
    ) -> Result<Vec<f32>> {
        let nf = self.n_features;
        if let Some(m) = margin {
            if m.len() != n_rows.checked_mul(self.k).context("margin size overflow")? {
                bail!(
                    "base_margin has {} values, expected {n_rows} rows x {} outputs",
                    m.len(),
                    self.k
                );
            }
        }
        if rows.len() != n_rows.checked_mul(nf).context("input size overflow")? {
            bail!(
                "expected {n_rows} rows x {nf} features = {} values, got {}",
                n_rows * nf,
                rows.len()
            );
        }
        let width = self.row_width(kind)?;
        let total = n_rows.checked_mul(width).context("output size overflow")?;
        let mut out = vec![0.0f32; total];
        if total == 0 {
            return Ok(out);
        }
        if workers <= 1 || n_rows == 1 {
            self.block(kind, rows, &mut out, margin);
            return Ok(out);
        }
        if pool.is_none() {
            *pool = Some(crate::threading::prediction_pool(workers)?);
        }
        let chunk_rows = n_rows.div_ceil(workers.saturating_mul(4)).max(1);
        pool.as_ref().expect("pool just built").install(|| {
            let k = self.k;
            out.par_chunks_mut(chunk_rows * width)
                .zip(rows.par_chunks(chunk_rows * nf))
                .enumerate()
                .for_each(|(i, (o, r))| {
                    let first = i * chunk_rows;
                    let m = margin.map(|m| &m[first * k..(first * k + r.len() / nf * k)]);
                    self.block(kind, r, o, m)
                });
        });
        Ok(out)
    }

    /// Serial computation of one contiguous block of rows.
    fn block(&self, kind: Kind, rows: &[f32], out: &mut [f32], margin: Option<&[f32]>) {
        let nf = self.n_features;
        let width = out.len() / (rows.len() / nf);
        let mut scratch = self.scratch();
        for (i, (feats, o)) in rows
            .chunks_exact(nf)
            .zip(out.chunks_exact_mut(width))
            .enumerate()
        {
            // This row's bias per output group: its own start, else the intercept.
            let bias = margin.map_or(self.group_bias.as_slice(), |m| {
                &m[i * self.k..(i + 1) * self.k]
            });
            match kind {
                Kind::Leaf => self.leaf_row(feats, o),
                Kind::Contribs { approx: false } => self.shap_row(feats, o, &mut scratch, bias),
                Kind::Contribs { approx: true } => self.saabas_row(feats, o, &mut scratch, bias),
                Kind::Interactions => self.interaction_row(feats, o, &mut scratch, bias),
            }
        }
    }

    fn leaf_row(&self, feats: &[f32], out: &mut [f32]) {
        for (slot, tree) in out.iter_mut().zip(&self.model.trees[..self.n_trees]) {
            let mut i = 0usize;
            loop {
                let node = &tree.nodes[i];
                if node.is_leaf {
                    break;
                }
                let left = node.goes_left(feats[node.feat as usize]);
                i = if left { node.left } else { node.right } as usize;
            }
            *slot = i as f32;
        }
    }

    fn shap_row(&self, feats: &[f32], out: &mut [f32], s: &mut Scratch, bias: &[f32]) {
        let nc = self.n_features + 1;
        let rule = rule();
        for g in 0..self.k {
            let p = &mut out[g * nc..(g + 1) * nc];
            for t in (g..self.n_trees).step_by(self.k) {
                s.tree_contribs.fill(0.0);
                let mut runner = Runner {
                    nodes: &self.model.trees[t].nodes,
                    feats,
                    rule,
                    path_prob: &mut s.path_prob,
                    form: Additive {
                        phi: &mut s.tree_contribs,
                    },
                };
                runner.run();
                for ci in 0..nc - 1 {
                    p[ci] += s.tree_contribs[ci];
                }
            }
            p[nc - 1] += self.group_root_mean[g];
            p[nc - 1] += bias[g];
        }
    }

    fn saabas_row(&self, feats: &[f32], out: &mut [f32], s: &mut Scratch, bias: &[f32]) {
        let nc = self.n_features + 1;
        for g in 0..self.k {
            let p = &mut out[g * nc..(g + 1) * nc];
            for t in (g..self.n_trees).step_by(self.k) {
                s.tree_contribs.fill(0.0);
                let nodes = &self.model.trees[t].nodes;
                let means = &self.approx_means[t];
                let mut node_value = means[0];
                s.tree_contribs[nc - 1] += node_value;
                if !nodes[0].is_leaf {
                    let mut idx = 0usize;
                    let mut split = 0usize;
                    while !nodes[idx].is_leaf {
                        let node = &nodes[idx];
                        split = node.feat as usize;
                        let left = node.goes_left(feats[split]);
                        idx = if left { node.left } else { node.right } as usize;
                        let new_value = means[idx];
                        s.tree_contribs[split] += new_value - node_value;
                        node_value = new_value;
                    }
                    s.tree_contribs[split] += nodes[idx].leaf_value - node_value;
                }
                for ci in 0..nc {
                    p[ci] += s.tree_contribs[ci];
                }
            }
            p[nc - 1] += bias[g];
        }
    }

    fn interaction_row(&self, feats: &[f32], out: &mut [f32], s: &mut Scratch, bias: &[f32]) {
        let nc = self.n_features + 1;
        let rule = rule();
        for g in 0..self.k {
            let matrix = &mut out[g * nc * nc..(g + 1) * nc * nc];
            s.diag.fill(0.0);
            for t in (g..self.n_trees).step_by(self.k) {
                s.path.clear();
                let mut runner = Runner {
                    nodes: &self.model.trees[t].nodes,
                    feats,
                    rule,
                    path_prob: &mut s.path_prob,
                    form: Interaction {
                        path: &mut s.path,
                        diag: &mut s.diag,
                        matrix,
                        ncols: nc,
                    },
                };
                runner.run();
            }
            s.diag[nc - 1] += self.group_root_mean[g];
            s.diag[nc - 1] += bias[g];
            // The path-local updates fill the off-diagonal row-wise; average
            // the two directional estimates so the matrix is symmetric.
            for r in 0..nc {
                for c in r + 1..nc {
                    let sym = 0.5 * (matrix[r * nc + c] + matrix[c * nc + r]);
                    matrix[r * nc + c] = sym;
                    matrix[c * nc + r] = sym;
                }
            }
            // Diagonal = additive SHAP value minus the row's interactions.
            for r in 0..nc {
                let mut value = s.diag[r];
                for c in 0..nc {
                    if c != r {
                        value -= matrix[r * nc + c];
                    }
                }
                matrix[r * nc + r] = value;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tree::{SchemaMode, Tree};
    use crate::types::Objective;

    /// Deterministic xorshift for test data (no dependency on a RNG crate).
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn unit(&mut self) -> f32 {
            (self.next() >> 40) as f32 / (1u64 << 24) as f32
        }
    }

    /// Grow a random tree of at most `depth` levels over `n_features`
    /// features. Covers are consistent (children sum to the parent).
    fn random_tree(rng: &mut Rng, n_features: usize, depth: usize) -> Tree {
        let mut t = Tree::default();
        let root_cover = 100.0 + rng.unit() * 100.0;
        let root = t.push_leaf(rng.unit() - 0.5, root_cover);
        let mut frontier = vec![(root, root_cover)];
        for _ in 0..depth {
            let mut next = Vec::new();
            for (node, cover) in frontier {
                if rng.unit() < 0.15 {
                    continue; // stays a leaf
                }
                let feat = (rng.next() % n_features as u64) as u32;
                let frac = 0.1 + 0.8 * rng.unit();
                let (lc, rc) = (cover * frac, cover * (1.0 - frac));
                let (l, r) = t.split_leaf(
                    node,
                    feat,
                    rng.unit(),
                    rng.next() % 2 == 0,
                    rng.unit(),
                    rng.unit() - 0.5,
                    lc,
                    rng.unit() - 0.5,
                    rc,
                );
                next.push((l, lc));
                next.push((r, rc));
            }
            frontier = next;
        }
        t
    }

    fn model_of(trees: Vec<Tree>, n_features: usize, k: usize) -> Model {
        let (objective, margins) = if k > 1 {
            (
                Objective::Softprob {
                    num_class: k as u32,
                },
                (0..k).map(|c| 0.1 * c as f32 - 0.05).collect(),
            )
        } else {
            (Objective::SquaredError, Vec::new())
        };
        Model {
            trees,
            base_score: 0.25,
            objective,
            n_features,
            best_iteration: None,
            best_score: None,
            feature_names: Vec::new(),
            schema_mode: Some(SchemaMode::Positional),
            class_base_margins: margins,
            scale_pos_weight: 1.0,
            feature_types: Vec::new(),
            rank_bias: None,
        }
    }

    fn random_rows(rng: &mut Rng, n: usize, nf: usize) -> Vec<f32> {
        (0..n * nf)
            .map(|_| {
                if rng.unit() < 0.15 {
                    f32::NAN
                } else {
                    rng.unit()
                }
            })
            .collect()
    }

    /// Expected value of one tree given that only the features in `known`
    /// (a bitmask) are fixed to the row's values; unknown splits are averaged
    /// by cover. This is the SHAP value function.
    fn tree_value(nodes: &[Node], idx: usize, feats: &[f32], known: u32) -> f64 {
        let n = &nodes[idx];
        if n.is_leaf {
            return n.leaf_value as f64;
        }
        let (l, r) = (n.left as usize, n.right as usize);
        if known >> n.feat & 1 == 1 {
            let left = n.goes_left(feats[n.feat as usize]);
            return tree_value(nodes, if left { l } else { r }, feats, known);
        }
        let (cl, cr) = (nodes[l].sum_hess as f64, nodes[r].sum_hess as f64);
        (cl * tree_value(nodes, l, feats, known) + cr * tree_value(nodes, r, feats, known))
            / (cl + cr)
    }

    fn model_value(model: &Model, group: usize, feats: &[f32], known: u32) -> f64 {
        let k = model.num_outputs();
        model
            .trees
            .iter()
            .enumerate()
            .filter(|(i, _)| i % k == group)
            .map(|(_, t)| tree_value(&t.nodes, 0, feats, known))
            .sum()
    }

    fn fact(n: usize) -> f64 {
        (1..=n).map(|v| v as f64).product()
    }

    /// Brute-force Shapley values (bias last) of one row and group.
    fn brute_shap(model: &Model, group: usize, feats: &[f32]) -> Vec<f64> {
        let m = model.n_features;
        let mut phi = vec![0.0; m + 1];
        for i in 0..m {
            for s in 0..(1u32 << m) {
                if s >> i & 1 == 1 {
                    continue;
                }
                let size = s.count_ones() as usize;
                let w = fact(size) * fact(m - size - 1) / fact(m);
                phi[i] += w
                    * (model_value(model, group, feats, s | 1 << i)
                        - model_value(model, group, feats, s));
            }
        }
        let bias = model_value(model, group, feats, 0)
            + if model.class_base_margins.is_empty() {
                crate::train::base_margin(model.objective, model.base_score) as f64
            } else {
                model.class_base_margins[group] as f64
            };
        phi[m] = bias;
        phi
    }

    /// Brute-force Shapley interaction values (off-diagonal entries; each
    /// pair is split evenly between (i, j) and (j, i)).
    fn brute_interaction(model: &Model, group: usize, feats: &[f32], i: usize, j: usize) -> f64 {
        let m = model.n_features;
        let mut acc = 0.0;
        for s in 0..(1u32 << m) {
            if s >> i & 1 == 1 || s >> j & 1 == 1 {
                continue;
            }
            let size = s.count_ones() as usize;
            let w = fact(size) * fact(m - size - 2) / (2.0 * fact(m - 1));
            let f = |mask: u32| model_value(model, group, feats, mask);
            acc += w * (f(s | 1 << i | 1 << j) - f(s | 1 << i) - f(s | 1 << j) + f(s));
        }
        acc
    }

    fn run(model: &Model, kind: Kind, rows: &[f32], workers: usize) -> Vec<f32> {
        let n = rows.len() / model.n_features;
        let ex = Explainer::new(model, model.trees.len(), kind).unwrap();
        ex.compute(kind, rows, n, workers, &mut None).unwrap()
    }

    #[test]
    fn quadrature_rule_integrates_polynomials() {
        // The rule approximates the integral over [0, 1] of f(u) with
        // f(u) = u^p for small p (nodes are s^2 of Gauss-Legendre points).
        let r = rule();
        for p in 0..=6 {
            let approx: f64 = (0..POINTS)
                .map(|i| r.weights[i] as f64 * (r.nodes[i] as f64).powi(p))
                .sum();
            assert!((approx - 1.0 / (p as f64 + 1.0)).abs() < 1e-5, "p={p}");
        }
    }

    #[test]
    fn exact_shap_matches_brute_force() {
        let mut rng = Rng(0x1234_5678_9abc_def1);
        for k in [1usize, 3] {
            let nf = 5;
            let trees = (0..6 * k).map(|_| random_tree(&mut rng, nf, 4)).collect();
            let model = model_of(trees, nf, k);
            let rows = random_rows(&mut rng, 12, nf);
            let kind = Kind::Contribs { approx: false };
            let out = run(&model, kind, &rows, 1);
            let nc = nf + 1;
            for (r, feats) in rows.chunks_exact(nf).enumerate() {
                for g in 0..k {
                    let expect = brute_shap(&model, g, feats);
                    for c in 0..nc {
                        let got = out[(r * k + g) * nc + c] as f64;
                        assert!(
                            (got - expect[c]).abs() < 2e-5,
                            "row {r} group {g} col {c}: {got} vs {}",
                            expect[c]
                        );
                    }
                }
            }
        }
    }

    /// Turn about half of the internal nodes into categorical splits over the
    /// codes 0..5, so rows of small integers exercise the set decision.
    fn categorize(rng: &mut Rng, tree: &mut Tree) {
        for node in 0..tree.len() {
            if tree.nodes[node].is_leaf || rng.next() % 2 == 0 {
                continue;
            }
            let mut set = crate::categorical::CatBits::empty();
            for c in 0..5u32 {
                if rng.next() % 2 == 0 {
                    set.set(c);
                }
            }
            if set.is_empty() {
                set.set(1);
            }
            tree.set_categories(node, &set);
        }
    }

    #[test]
    fn categorical_splits_follow_the_set_in_every_explanation() {
        let mut rng = Rng(0xfeed_beef);
        let nf = 4;
        let mut trees: Vec<Tree> = (0..8).map(|_| random_tree(&mut rng, nf, 4)).collect();
        trees.iter_mut().for_each(|t| categorize(&mut rng, t));
        assert!(trees.iter().any(Tree::has_categorical));
        let model = model_of(trees, nf, 1);
        // Codes 0..6 (5 is outside every set), invalid values, and missing.
        let pool = [0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, -1.0, 40.0, f32::NAN];
        let rows: Vec<f32> = (0..16 * nf)
            .map(|_| pool[(rng.next() % 9) as usize])
            .collect();
        let out = run(&model, Kind::Contribs { approx: false }, &rows, 1);
        let nc = nf + 1;
        for (r, feats) in rows.chunks_exact(nf).enumerate() {
            let expect = brute_shap(&model, 0, feats);
            for c in 0..nc {
                let got = out[r * nc + c] as f64;
                assert!(
                    (got - expect[c]).abs() < 2e-5,
                    "row {r} col {c}: {got} vs {}",
                    expect[c]
                );
            }
            // Contributions sum to the margin, which follows the same decisions.
            let sum: f64 = (0..nc).map(|c| out[r * nc + c] as f64).sum();
            assert!((sum - model.predict_margin(feats) as f64).abs() < 1e-4);
        }
        // pred_leaf lands where the model's own traversal does.
        let leaves = run(&model, Kind::Leaf, &rows, 1);
        for (r, feats) in rows.chunks_exact(nf).enumerate() {
            for (t, tree) in model.trees.iter().enumerate() {
                let mut i = 0usize;
                while !tree.nodes[i].is_leaf {
                    let n = &tree.nodes[i];
                    i = if n.goes_left(feats[n.feat as usize]) {
                        n.left
                    } else {
                        n.right
                    } as usize;
                }
                assert_eq!(leaves[r * model.trees.len() + t], i as f32);
            }
        }
    }

    #[test]
    fn contributions_sum_to_margin() {
        let mut rng = Rng(77);
        let nf = 6;
        let trees = (0..10).map(|_| random_tree(&mut rng, nf, 5)).collect();
        let model = model_of(trees, nf, 1);
        let rows = random_rows(&mut rng, 40, nf);
        for approx in [false, true] {
            let out = run(&model, Kind::Contribs { approx }, &rows, 1);
            for (r, feats) in rows.chunks_exact(nf).enumerate() {
                let sum: f32 = out[r * (nf + 1)..(r + 1) * (nf + 1)].iter().sum();
                let margin = model.predict_margin(feats);
                assert!((sum - margin).abs() < 1e-4, "approx={approx} row {r}");
            }
        }
    }

    #[test]
    fn interactions_match_brute_force_and_are_consistent() {
        let mut rng = Rng(991);
        let nf = 4;
        let trees = (0..5).map(|_| random_tree(&mut rng, nf, 3)).collect();
        let model = model_of(trees, nf, 1);
        let rows = random_rows(&mut rng, 10, nf);
        let nc = nf + 1;
        let inter = run(&model, Kind::Interactions, &rows, 1);
        let shap = run(&model, Kind::Contribs { approx: false }, &rows, 1);
        for (r, feats) in rows.chunks_exact(nf).enumerate() {
            let m = &inter[r * nc * nc..(r + 1) * nc * nc];
            for i in 0..nf {
                for j in 0..nf {
                    if i != j {
                        let expect = brute_interaction(&model, 0, feats, i, j);
                        assert!((m[i * nc + j] as f64 - expect).abs() < 5e-5, "({i},{j})");
                        assert_eq!(m[i * nc + j], m[j * nc + i]);
                    }
                }
            }
            for i in 0..nc {
                let row_sum: f32 = m[i * nc..(i + 1) * nc].iter().sum();
                assert!((row_sum - shap[r * nc + i]).abs() < 1e-4, "row sum {i}");
            }
        }
    }

    #[test]
    fn worker_count_does_not_change_results() {
        let mut rng = Rng(5);
        let nf = 7;
        let trees = (0..9).map(|_| random_tree(&mut rng, nf, 5)).collect();
        let model = model_of(trees, nf, 3);
        let rows = random_rows(&mut rng, 101, nf);
        for kind in [
            Kind::Contribs { approx: false },
            Kind::Contribs { approx: true },
            Kind::Interactions,
            Kind::Leaf,
        ] {
            let a = run(&model, kind, &rows, 1);
            let b = run(&model, kind, &rows, 4);
            assert_eq!(a, b, "{kind:?}");
        }
    }

    #[test]
    fn leaf_indices_follow_the_tree() {
        let mut t = Tree::default();
        let root = t.push_leaf(0.0, 10.0);
        let (l, r) = t.split_leaf(root, 0, 0.5, true, 1.0, 1.0, 4.0, 2.0, 6.0);
        let model = model_of(
            vec![
                t.clone(),
                Tree {
                    nodes: vec![Node::leaf(3.0, 1.0)],
                },
            ],
            1,
            1,
        );
        let rows = [0.2f32, 0.9, f32::NAN];
        let out = run(&model, Kind::Leaf, &rows, 1);
        assert_eq!(out, vec![l as f32, 0.0, r as f32, 0.0, l as f32, 0.0]);
    }

    #[test]
    fn importance_types() {
        let mut t = Tree::default();
        let root = t.push_leaf(0.0, 10.0);
        let (l, _) = t.split_leaf(root, 0, 0.5, true, 4.0, 0.0, 6.0, 0.0, 4.0);
        t.split_leaf(l, 1, 0.5, true, 2.0, 0.0, 3.0, 0.0, 3.0);
        let mut t2 = Tree::default();
        let r2 = t2.push_leaf(0.0, 8.0);
        t2.split_leaf(r2, 0, 0.1, true, 1.0, 0.0, 4.0, 0.0, 4.0);
        let model = model_of(vec![t, t2], 3, 1);
        let get = |k| feature_scores(&model, k);
        assert_eq!(get(Importance::Weight), vec![(0, 2.0), (1, 1.0)]);
        assert_eq!(get(Importance::TotalGain), vec![(0, 5.0), (1, 2.0)]);
        assert_eq!(get(Importance::Gain), vec![(0, 2.5), (1, 2.0)]);
        assert_eq!(get(Importance::TotalCover), vec![(0, 18.0), (1, 6.0)]);
        assert_eq!(get(Importance::Cover), vec![(0, 9.0), (1, 6.0)]);
    }

    #[test]
    fn root_only_trees_contribute_to_bias() {
        let model = model_of(
            vec![Tree {
                nodes: vec![Node::leaf(0.75, 5.0)],
            }],
            2,
            1,
        );
        let rows = [0.1f32, 0.2];
        for approx in [false, true] {
            let out = run(&model, Kind::Contribs { approx }, &rows, 1);
            assert_eq!(out, vec![0.0, 0.0, 0.75 + 0.25]);
        }
    }

    /// A per-row `base_margin` replaces the intercept in the bias column (and only
    /// there), for every explanation kind, with any worker count; a margin equal to
    /// the intercept changes nothing.
    #[test]
    fn base_margin_replaces_the_bias_per_row() {
        let mut rng = Rng(5150);
        let nf = 4;
        for k in [1usize, 3] {
            let trees = (0..6 * k).map(|_| random_tree(&mut rng, nf, 4)).collect();
            let model = model_of(trees, nf, k);
            let n = 25;
            let rows = random_rows(&mut rng, n, nf);
            let margin: Vec<f32> = (0..n * k).map(|i| (i as f32 * 0.37).sin()).collect();
            for kind in [
                Kind::Contribs { approx: false },
                Kind::Contribs { approx: true },
                Kind::Interactions,
            ] {
                let ex = Explainer::new(&model, model.trees.len(), kind).unwrap();
                let width = ex.row_width(kind).unwrap() / k;
                let plain = ex.compute(kind, &rows, n, 1, &mut None).unwrap();
                let intercept: Vec<f32> = (0..n).flat_map(|_| ex.group_bias.clone()).collect();
                let same = ex
                    .compute_with_margin(kind, &rows, n, 1, &mut None, Some(&intercept))
                    .unwrap();
                assert_eq!(plain, same, "k={k} {kind:?}: intercept margin is a no-op");
                for workers in [1usize, 4] {
                    let got = ex
                        .compute_with_margin(kind, &rows, n, workers, &mut None, Some(&margin))
                        .unwrap();
                    let nc = nf + 1;
                    let last = match kind {
                        Kind::Interactions => nc * nc - 1,
                        _ => nc - 1,
                    };
                    for r in 0..n {
                        for g in 0..k {
                            let base = (r * k + g) * width;
                            for j in 0..width {
                                let (a, b) = (got[base + j], plain[base + j]);
                                if j == last {
                                    let want = b - ex.group_bias[g] + margin[r * k + g];
                                    assert!((a - want).abs() < 1e-4, "k={k} {kind:?} bias row {r}");
                                } else {
                                    assert_eq!(a.to_bits(), b.to_bits(), "k={k} {kind:?} j={j}");
                                }
                            }
                        }
                    }
                }
            }
        }
        // A margin of the wrong length is an error.
        let model = model_of(vec![random_tree(&mut rng, nf, 3)], nf, 1);
        let ex = Explainer::new(&model, 1, Kind::Contribs { approx: false }).unwrap();
        let rows = random_rows(&mut rng, 3, nf);
        assert!(ex
            .compute_with_margin(
                Kind::Contribs { approx: false },
                &rows,
                3,
                1,
                &mut None,
                Some(&[0.0, 1.0])
            )
            .is_err());
    }

    /// The device kernels replay the host's float32 operation order, so every
    /// explanation must match bit for bit (not just to a tolerance), for
    /// multiclass, categorical, missing-value, root-only and deep models, with
    /// a tree prefix and with per-row base margins.
    #[cfg(feature = "cuda")]
    #[test]
    fn gpu_matches_cpu_bit_for_bit() {
        let mut rng = Rng(0xc0ff_ee11);
        // (outputs, features, depth, categorical)
        let cases = [
            (1usize, 5usize, 4usize, false),
            (3, 4, 5, false),
            (1, 4, 4, true),
            (2, 3, 6, true),
            (1, 2, 9, false),
            (1, 6, 0, false),
        ];
        for (k, nf, depth, cat) in cases {
            let mut trees: Vec<Tree> = (0..5 * k)
                .map(|_| random_tree(&mut rng, nf, depth))
                .collect();
            if cat {
                trees.iter_mut().for_each(|t| categorize(&mut rng, t));
            }
            let model = model_of(trees, nf, k);
            let gpu = match crate::backend::cuda::GpuExplainModel::new(&model, 0) {
                Ok(gpu) => gpu,
                Err(e) => {
                    eprintln!("skipping the device comparison: {e:#}");
                    return;
                }
            };
            let n = 37;
            let pool = [0.0f32, 1.0, 2.0, 3.0, 4.0, 5.0, -1.0, 40.0, f32::NAN];
            let rows: Vec<f32> = if cat {
                (0..n * nf)
                    .map(|_| pool[(rng.next() % 9) as usize])
                    .collect()
            } else {
                random_rows(&mut rng, n, nf)
            };
            let margin: Vec<f32> = (0..n * k).map(|_| rng.unit() - 0.5).collect();
            for kind in [
                Kind::Contribs { approx: false },
                Kind::Contribs { approx: true },
                Kind::Interactions,
                Kind::Leaf,
            ] {
                for upto in [model.trees.len(), model.trees.len() - k, 0] {
                    let ex = Explainer::new(&model, upto, kind).unwrap();
                    for m in [None, Some(margin.as_slice())] {
                        let cpu = ex
                            .compute_with_margin(kind, &rows, n, 1, &mut None, m)
                            .unwrap();
                        let dev = gpu.compute(&ex, kind, &rows, n, m).unwrap();
                        assert_eq!(cpu.len(), dev.len());
                        for (i, (a, b)) in cpu.iter().zip(&dev).enumerate() {
                            assert_eq!(
                                a.to_bits(),
                                b.to_bits(),
                                "k={k} depth={depth} cat={cat} {kind:?} upto={upto} margin={} \
                                 at {i}: {a} vs {b}",
                                m.is_some()
                            );
                        }
                    }
                }
            }
        }
    }
}
