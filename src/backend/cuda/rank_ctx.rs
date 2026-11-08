//! Device LambdaMART for `gpu_math = "fast"` with a `rank:*` objective.
//!
//! The kernels live at the end of `histogram.cu` (see the comment there for the round
//! structure). Everything here is bookkeeping: the buffers are allocated once by
//! `GpuTrainCtx::new` from the shape the planner charged
//! ([`crate::gpu_mem_plan::fast_rank_bytes`]), the static per-group inputs are
//! uploaded once by `init_fast_rank`, and `fast_rank_round` runs the kernels of one
//! round. Nothing is allocated during training.
//!
//! Deterministic by construction: the sort is a pure function of the scores, pair
//! contributions are added as fixed-point int64 (exact, order independent), and the
//! only reductions are integer. There are no float atomics.
//!
//! `lambdarank_unbiased` adds a position-bias state on the device (`t+` / `t-`, the
//! integer cost sums, a control block) and replaces steps 4 to 6 of the round by the
//! bound / scale / accumulate sequence described in `histogram.cu`: the fixed-point scale
//! of the divided-by-bias lambdas cannot be fixed up front, so it is derived each round
//! from an exact device maximum. The biases are copied to the host once, at the end.

use super::*;
use crate::gpu_mem_plan::{FastRankShape, RANK_BIAS_CTL_WORDS, RANK_BIAS_REPLICAS};
use crate::rank::{PairMethod, RankKind, RankParams};

/// Static inputs of the device ranking path, prepared on the host from the training
/// labels and query groups.
pub struct RankDeviceInputs<'a> {
    pub kind: RankKind,
    pub params: RankParams,
    pub labels: &'a [f32],
    pub group_ptr: &'a [u32],
    /// One weight per group, if the data has weights.
    pub group_weights: Option<&'a [f32]>,
    /// `n_groups / sum(weights)` (XGBoost's `WeightNorm`).
    pub weight_norm: f64,
    /// `1 / IDCG@k` per group (only read by `rank:ndcg`).
    pub inv_idcg: &'a [f64],
    /// Display positions tracked by the unbiased estimation; 0 without `lambdarank_unbiased`.
    pub n_positions: usize,
}

/// Per-training constants decided by `init_fast_rank`.
#[derive(Clone, Copy)]
struct RankRun {
    kind: u32,
    is_topk: u32,
    num_pair: u32,
    exp_gain: u32,
    score_norm: u32,
    normalization: u32,
    has_group_weight: u32,
    n_groups: u32,
    total_slots: u32,
    weight_norm: f32,
    scale: f64,
    /// Tracked positions of the unbiased path (0 = plain), `regularizer` of its update and
    /// the term-count bounds of its scales (see `rank_bias_scales`).
    n_pos: u32,
    regularizer: f64,
    bound_row: f64,
    bound_total: f64,
}

/// Device state of `lambdarank_unbiased`.
struct RankBiasDevice {
    pairs: CudaFunction,
    scales: CudaFunction,
    finalize: CudaFunction,
    update: CudaFunction,
    /// `t+` / `t-` (`shape.n_positions` each)
    ti: CudaSlice<f64>,
    tj: CudaSlice<f64>,
    /// `RANK_BIAS_REPLICAS` x 2 x positions fixed-point cost sums
    repl: CudaSlice<i64>,
    /// bounds, scales and status words (`RANK_BIAS_CTL_WORDS`)
    ctl: CudaSlice<u64>,
}

pub(super) struct RankFns {
    apply_delta: CudaFunction,
    sort_groups: CudaFunction,
    map_stat: CudaFunction,
    pairs: CudaFunction,
    finalize: CudaFunction,
}

pub(super) struct RankDevice {
    fns: RankFns,
    shape: FastRankShape,
    group_ptr: CudaSlice<u32>,
    init_ptr: CudaSlice<u32>,
    inv_idcg: CudaSlice<f64>,
    group_weight: CudaSlice<f32>,
    y_order: Option<CudaSlice<u32>>,
    bucket_lo: Option<CudaSlice<u32>>,
    bucket_hi: Option<CudaSlice<u32>>,
    rank_idx: CudaSlice<u32>,
    rank_tmp: CudaSlice<u32>,
    rank_pos: CudaSlice<u32>,
    acc_g: CudaSlice<i64>,
    acc_h: CudaSlice<i64>,
    group_sum: CudaSlice<i64>,
    n_rel: Option<CudaSlice<f64>>,
    acc: Option<CudaSlice<f64>>,
    bias: Option<RankBiasDevice>,
    run: Option<RankRun>,
}

impl RankDevice {
    /// Allocate every buffer of `shape` (exactly `fast_rank_bytes`) and load the kernels.
    pub(super) fn alloc(
        stream: &Arc<CudaStream>,
        module: &Arc<cudarc::driver::CudaModule>,
        n_rows: usize,
        shape: FastRankShape,
    ) -> Result<Self> {
        let load = |name: &str| -> Result<CudaFunction> {
            module
                .load_function(name)
                .with_context(|| format!("{name} not found in PTX"))
        };
        let fns = RankFns {
            apply_delta: load("rank_apply_delta")?,
            sort_groups: load("rank_sort_groups")?,
            map_stat: load("rank_map_stat")?,
            pairs: load("rank_pairs")?,
            finalize: load("rank_finalize")?,
        };
        let ng = usize::try_from(shape.n_groups).context("group count")?;
        let rows = n_rows.max(1);
        let zeros_u32 = |n: usize| stream.alloc_zeros::<u32>(n.max(1));
        Ok(Self {
            fns,
            shape,
            group_ptr: zeros_u32(ng + 1).context("rank group_ptr alloc")?,
            init_ptr: zeros_u32(ng + 1).context("rank init_ptr alloc")?,
            inv_idcg: stream
                .alloc_zeros::<f64>(ng.max(1))
                .context("rank inv_idcg alloc")?,
            group_weight: stream
                .alloc_zeros::<f32>(ng.max(1))
                .context("rank group weight alloc")?,
            y_order: shape
                .mean
                .then(|| zeros_u32(rows))
                .transpose()
                .context("rank y_order alloc")?,
            bucket_lo: shape
                .mean
                .then(|| zeros_u32(rows))
                .transpose()
                .context("rank bucket_lo alloc")?,
            bucket_hi: shape
                .mean
                .then(|| zeros_u32(rows))
                .transpose()
                .context("rank bucket_hi alloc")?,
            rank_idx: zeros_u32(rows).context("rank list alloc")?,
            rank_tmp: zeros_u32(rows).context("rank scratch alloc")?,
            rank_pos: zeros_u32(rows).context("rank position alloc")?,
            acc_g: stream
                .alloc_zeros::<i64>(rows)
                .context("rank grad accumulator alloc")?,
            acc_h: stream
                .alloc_zeros::<i64>(rows)
                .context("rank hess accumulator alloc")?,
            group_sum: stream
                .alloc_zeros::<i64>(ng.max(1))
                .context("rank group sum alloc")?,
            n_rel: shape
                .map
                .then(|| stream.alloc_zeros::<f64>(rows))
                .transpose()
                .context("rank n_rel alloc")?,
            acc: shape
                .map
                .then(|| stream.alloc_zeros::<f64>(rows))
                .transpose()
                .context("rank acc alloc")?,
            bias: shape
                .n_positions
                .gt(&0)
                .then(|| -> Result<RankBiasDevice> {
                    let np = usize::try_from(shape.n_positions).context("position count")?;
                    let repl_len = RANK_BIAS_REPLICAS as usize * 2 * np;
                    Ok(RankBiasDevice {
                        pairs: load("rank_pairs_ub")?,
                        scales: load("rank_bias_scales")?,
                        finalize: load("rank_finalize_ub")?,
                        update: load("rank_bias_update")?,
                        ti: stream.alloc_zeros::<f64>(np).context("rank t+ alloc")?,
                        tj: stream.alloc_zeros::<f64>(np).context("rank t- alloc")?,
                        repl: stream
                            .alloc_zeros::<i64>(repl_len)
                            .context("rank cost sums alloc")?,
                        ctl: stream
                            .alloc_zeros::<u64>(RANK_BIAS_CTL_WORDS as usize)
                            .context("rank control block alloc")?,
                    })
                })
                .transpose()?,
            run: None,
        })
    }
}

/// Fixed-point scale of the integer accumulators: a power of two such that the sum of
/// every pair contribution a row (or a group) can receive stays far below 2^63.
/// One pair adds at most `2 * 100` (|lambda| <= 100 because the metric delta is at most
/// 1 and is divided by at least 0.01; the group sum adds twice that).
pub(crate) fn rank_fixed_scale(group_ptr: &[u32], params: &RankParams) -> Result<f64> {
    let k = u64::from(params.num_pair());
    let n_max = group_ptr
        .windows(2)
        .map(|w| {
            let cnt = u64::from(w[1] - w[0]);
            k.min(cnt) * cnt + (cnt + 1) * (k + 1)
        })
        .max()
        .unwrap_or(1)
        .max(1);
    let bound = 200.0 * n_max as f64;
    let e = (61.0 - bound.log2()).floor().min(40.0);
    ensure!(
        e >= 0.0,
        "a query group is too large for the fast device ranking path (accumulators would overflow); \
         use gpu_math=\"exact\""
    );
    Ok(2f64.powi(e as i32))
}

/// Most terms that can reach one row accumulator (or a group sum) in a round, and, summed
/// over the groups, one position cost sum: the same per-group count `rank_fixed_scale`
/// bounds (`k` partners per initiator as initiator and as partner).
fn rank_term_bounds(group_ptr: &[u32], params: &RankParams) -> (f64, f64) {
    let k = u64::from(params.num_pair());
    let (mut row, mut total) = (1.0f64, 0.0f64);
    for w in group_ptr.windows(2) {
        let cnt = u64::from(w[1] - w[0]);
        let n = (k.min(cnt) * cnt + (cnt + 1) * (k + 1)) as f64;
        row = row.max(n);
        total += n;
    }
    (row, total.max(1.0))
}

impl GpuTrainCtx {
    /// Upload the static ranking inputs (once, before the first round).
    pub fn init_fast_rank(&self, inputs: &RankDeviceInputs<'_>) -> Result<()> {
        let n_groups = inputs.group_ptr.len() - 1;
        let mut guard = self.lock()?;
        let rank = guard
            .rank
            .as_mut()
            .context("the device ranking buffers were not allocated (fast_rank shape missing)")?;
        ensure!(
            rank.shape.n_groups as usize >= n_groups,
            "device ranking buffers hold {} groups, the data has {n_groups}",
            rank.shape.n_groups
        );
        let is_topk = inputs.params.pair_method == PairMethod::TopK;
        // initiator slots: topk = the first min(cnt, k) rank positions; mean = every document
        let k = inputs.params.num_pair();
        let mut init_ptr = Vec::with_capacity(n_groups + 1);
        let mut total: u64 = 0;
        init_ptr.push(0u32);
        for w in inputs.group_ptr.windows(2) {
            let cnt = w[1] - w[0];
            total += u64::from(if is_topk { cnt.min(k) } else { cnt });
            init_ptr.push(u32::try_from(total).context("too many pair initiators")?);
        }
        let stream = &self.stream;
        stream
            .memcpy_htod(inputs.group_ptr, &mut rank.group_ptr)
            .context("rank group_ptr H2D")?;
        stream
            .memcpy_htod(&init_ptr, &mut rank.init_ptr)
            .context("rank init_ptr H2D")?;
        if inputs.kind == RankKind::Ndcg {
            ensure!(inputs.inv_idcg.len() == n_groups, "inv_idcg length");
            stream
                .memcpy_htod(inputs.inv_idcg, &mut rank.inv_idcg)
                .context("rank inv_idcg H2D")?;
        }
        if let Some(w) = inputs.group_weights {
            ensure!(w.len() == n_groups, "group weight length");
            stream
                .memcpy_htod(w, &mut rank.group_weight)
                .context("rank group weight H2D")?;
        }
        if !is_topk {
            // label-sorted order of every group (stable, descending) and the bucket of
            // equal labels each position belongs to: static for the whole training.
            let n = inputs.labels.len();
            let (mut order, mut lo, mut hi) = (vec![0u32; n], vec![0u32; n], vec![0u32; n]);
            let mut idx: Vec<u32> = Vec::new();
            for w in inputs.group_ptr.windows(2) {
                let (b, e) = (w[0] as usize, w[1] as usize);
                let y = &inputs.labels[b..e];
                idx.clear();
                idx.extend(0..(e - b) as u32);
                idx.sort_by(|&a, &c| {
                    y[c as usize]
                        .partial_cmp(&y[a as usize])
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                let mut i = 0usize;
                while i < idx.len() {
                    let mut j = i + 1;
                    while j < idx.len() && y[idx[i] as usize] == y[idx[j] as usize] {
                        j += 1;
                    }
                    for p in i..j {
                        order[b + p] = idx[p];
                        lo[b + p] = i as u32;
                        hi[b + p] = j as u32;
                    }
                    i = j;
                }
            }
            stream
                .memcpy_htod(&order, rank.y_order.as_mut().context("y_order")?)
                .context("rank y_order H2D")?;
            stream
                .memcpy_htod(&lo, rank.bucket_lo.as_mut().context("bucket_lo")?)
                .context("rank bucket_lo H2D")?;
            stream
                .memcpy_htod(&hi, rank.bucket_hi.as_mut().context("bucket_hi")?)
                .context("rank bucket_hi H2D")?;
        }
        let n_pos = inputs.n_positions;
        match (rank.bias.as_mut(), n_pos) {
            (None, 0) => {}
            (Some(b), n) if n > 0 => {
                ensure!(
                    n as u64 <= rank.shape.n_positions,
                    "device ranking buffers track {} positions, the data needs {n}",
                    rank.shape.n_positions
                );
                // every position starts unbiased
                let ones = vec![1.0f64; rank.shape.n_positions as usize];
                stream
                    .memcpy_htod(&ones, &mut b.ti)
                    .context("rank t+ H2D")?;
                stream
                    .memcpy_htod(&ones, &mut b.tj)
                    .context("rank t- H2D")?;
            }
            (Some(_), _) => anyhow::bail!(
                "the device ranking buffers were allocated for lambdarank_unbiased but the \
                 data tracks no position"
            ),
            (None, _) => anyhow::bail!(
                "lambdarank_unbiased needs the position-bias buffers, which were not allocated"
            ),
        }
        let (bound_row, bound_total) = rank_term_bounds(inputs.group_ptr, &inputs.params);
        rank.run = Some(RankRun {
            kind: match inputs.kind {
                RankKind::Ndcg => 0,
                RankKind::Pairwise => 1,
                RankKind::Map => 2,
            },
            is_topk: u32::from(is_topk),
            num_pair: inputs.params.num_pair(),
            exp_gain: u32::from(inputs.params.ndcg_exp_gain),
            score_norm: u32::from(inputs.params.score_normalization),
            normalization: u32::from(inputs.params.normalization),
            has_group_weight: u32::from(inputs.group_weights.is_some()),
            n_groups: n_groups as u32,
            total_slots: u32::try_from(total).context("too many pair initiators")?,
            weight_norm: inputs.weight_norm as f32,
            // the unbiased path derives its scales every round instead
            scale: if n_pos == 0 {
                rank_fixed_scale(inputs.group_ptr, &inputs.params)?
            } else {
                1.0
            },
            n_pos: u32::try_from(n_pos).context("position count")?,
            regularizer: inputs.params.regularizer(),
            bound_row,
            bound_total,
        });
        Ok(())
    }

    /// One round of device LambdaMART: apply the previous tree, sort, accumulate pairs,
    /// finalize into the f32 gpair buffer, then quantize (the shared tail).
    pub fn fast_rank_round(
        &self,
        round_seed: u32,
        subsample_threshold: u64,
        subsample_seed: u64,
        subsample_tree: u64,
    ) -> Result<(GradQuantizer, GradPairFixed, f32)> {
        let mut buf = self.lock()?;
        self.sync_if_timing()?;
        let started = std::time::Instant::now();
        let n_rows = u32::try_from(self.n_rows).context("row count exceeds u32")?;
        {
            let GpuBuffers {
                pred_resident,
                label_resident,
                prediction_delta,
                gpair_f32,
                rank,
                ..
            } = &mut *buf;
            let rank = rank
                .as_mut()
                .context("the device ranking buffers were not allocated")?;
            let run = rank.run.context("init_fast_rank was not called")?;
            let pred = pred_resident
                .as_mut()
                .context("fast pred buffer not allocated")?;
            let label = label_resident
                .as_ref()
                .context("fast label buffer not allocated")?;
            let out = gpair_f32
                .as_mut()
                .context("f32 gpair buffer not allocated")?;
            let grid_rows = n_rows.div_ceil(256).clamp(1, 4096);

            // 1. pred += delta
            {
                let cfg = LaunchConfig {
                    grid_dim: (grid_rows, 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                };
                let mut l = self.stream.launch_builder(&rank.fns.apply_delta);
                l.arg(&mut *pred).arg(&*prediction_delta).arg(&n_rows);
                unsafe { l.launch(cfg) }.context("launch rank_apply_delta")?;
            }
            // 2. stable sort of every group
            {
                let cfg = LaunchConfig {
                    grid_dim: (run.n_groups.clamp(1, 1 << 20), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                };
                let mut l = self.stream.launch_builder(&rank.fns.sort_groups);
                l.arg(&*pred)
                    .arg(&rank.group_ptr)
                    .arg(&run.n_groups)
                    .arg(&mut rank.rank_idx)
                    .arg(&mut rank.rank_tmp)
                    .arg(&mut rank.rank_pos);
                unsafe { l.launch(cfg) }.context("launch rank_sort_groups")?;
            }
            // 3. rank:map statistics
            if run.kind == 2 {
                let cfg = LaunchConfig {
                    grid_dim: (run.n_groups.div_ceil(128).clamp(1, 4096), 1, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: 0,
                };
                let mut l = self.stream.launch_builder(&rank.fns.map_stat);
                l.arg(label)
                    .arg(&rank.group_ptr)
                    .arg(&run.n_groups)
                    .arg(&rank.rank_idx)
                    .arg(rank.n_rel.as_mut().context("n_rel")?)
                    .arg(rank.acc.as_mut().context("acc")?);
                unsafe { l.launch(cfg) }.context("launch rank_map_stat")?;
            }
            if rank.bias.is_some() {
                // 4. - 6. bound, scale, accumulate, finalize, bias update
                self.unbiased_pairs_and_finalize(
                    rank,
                    run,
                    pred,
                    label,
                    out,
                    round_seed,
                    (subsample_threshold, subsample_seed, subsample_tree),
                    grid_rows,
                )?;
            } else {
                // 4. fixed-point accumulators
                self.stream
                    .memset_zeros(&mut rank.acc_g)
                    .context("zero rank grad accumulator")?;
                self.stream
                    .memset_zeros(&mut rank.acc_h)
                    .context("zero rank hess accumulator")?;
                self.stream
                    .memset_zeros(&mut rank.group_sum)
                    .context("zero rank group sums")?;
                // 5. pairs (integer atomics)
                {
                    let cfg = LaunchConfig {
                        grid_dim: (run.total_slots.div_ceil(256).clamp(1, 1 << 16), 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let seed = u64::from(round_seed);
                    let mut l = self.stream.launch_builder(&rank.fns.pairs);
                    // The optional buffers are only read for the kinds / methods that own them;
                    // other launches pass an always-valid dummy of the same type.
                    let n_rel = rank.n_rel.as_ref().unwrap_or(&rank.inv_idcg);
                    let acc = rank.acc.as_ref().unwrap_or(&rank.inv_idcg);
                    let y_order = rank.y_order.as_ref().unwrap_or(&rank.rank_pos);
                    let lo = rank.bucket_lo.as_ref().unwrap_or(&rank.rank_pos);
                    let hi = rank.bucket_hi.as_ref().unwrap_or(&rank.rank_pos);
                    l.arg(&*pred)
                        .arg(label)
                        .arg(&rank.group_ptr)
                        .arg(&rank.init_ptr)
                        .arg(&run.n_groups)
                        .arg(&run.total_slots)
                        .arg(&rank.rank_idx)
                        .arg(&rank.rank_pos)
                        .arg(&rank.inv_idcg)
                        .arg(n_rel)
                        .arg(acc)
                        .arg(y_order)
                        .arg(lo)
                        .arg(hi)
                        .arg(&run.kind)
                        .arg(&run.is_topk)
                        .arg(&run.num_pair)
                        .arg(&run.exp_gain)
                        .arg(&run.score_norm)
                        .arg(&seed)
                        .arg(&run.scale)
                        .arg(&mut rank.acc_g)
                        .arg(&mut rank.acc_h)
                        .arg(&mut rank.group_sum);
                    unsafe { l.launch(cfg) }.context("launch rank_pairs")?;
                }
                // 6. finalize into the f32 gpair (normalization, group weight, row sampling)
                {
                    let cfg = LaunchConfig {
                        grid_dim: (grid_rows, 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    };
                    let mut l = self.stream.launch_builder(&rank.fns.finalize);
                    l.arg(&rank.group_ptr)
                        .arg(&run.n_groups)
                        .arg(&n_rows)
                        .arg(&rank.acc_g)
                        .arg(&rank.acc_h)
                        .arg(&rank.group_sum)
                        .arg(&rank.group_weight)
                        .arg(&run.has_group_weight)
                        .arg(&run.weight_norm)
                        .arg(&run.is_topk)
                        .arg(&run.num_pair)
                        .arg(&run.normalization)
                        .arg(&run.scale)
                        .arg(out)
                        .arg(&subsample_threshold)
                        .arg(&subsample_seed)
                        .arg(&subsample_tree);
                    unsafe { l.launch(cfg) }.context("launch rank_finalize")?;
                }
            }
        }
        // The finalize kernel already applied the row sampling; the quantize tail
        // scans gpair_f32 for the maximum (like the quantile path).
        self.quantize_resident_gpair(&mut buf, started, u64::MAX, 0, 0, false)
    }

    /// Steps 4 to 6 of a round with `lambdarank_unbiased`: zero the accumulators, bound
    /// every magnitude (pass 0), derive the fixed-point scales, accumulate (pass 1),
    /// finalize into the f32 gpair and re-estimate the position biases. Errors when the
    /// round produced a non-finite magnitude.
    #[allow(clippy::too_many_arguments)]
    fn unbiased_pairs_and_finalize(
        &self,
        rank: &mut RankDevice,
        run: RankRun,
        pred: &CudaSlice<f32>,
        label: &CudaSlice<f32>,
        out: &mut CudaSlice<f32>,
        round_seed: u32,
        (subsample_threshold, subsample_seed, subsample_tree): (u64, u64, u64),
        grid_rows: u32,
    ) -> Result<()> {
        let RankDevice {
            acc_g,
            acc_h,
            group_sum,
            group_ptr,
            init_ptr,
            inv_idcg,
            group_weight,
            y_order,
            bucket_lo,
            bucket_hi,
            rank_idx,
            rank_pos,
            n_rel,
            acc,
            bias,
            ..
        } = rank;
        let b = bias.as_mut().context("position-bias buffers")?;
        self.stream
            .memset_zeros(acc_g)
            .context("zero rank grad accumulator")?;
        self.stream
            .memset_zeros(acc_h)
            .context("zero rank hess accumulator")?;
        self.stream
            .memset_zeros(group_sum)
            .context("zero rank group sums")?;
        self.stream
            .memset_zeros(&mut b.repl)
            .context("zero rank cost sums")?;
        self.stream
            .memset_zeros(&mut b.ctl)
            .context("zero rank control block")?;
        let seed = u64::from(round_seed);
        let pair_cfg = LaunchConfig {
            grid_dim: (run.total_slots.div_ceil(256).clamp(1, 1 << 16), 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        // The optional buffers are only read for the kinds / methods that own them;
        // other launches pass an always-valid dummy of the same type.
        let n_rel = n_rel.as_ref().unwrap_or(&*inv_idcg);
        let acc = acc.as_ref().unwrap_or(&*inv_idcg);
        let y_order = y_order.as_ref().unwrap_or(&*rank_pos);
        let lo = bucket_lo.as_ref().unwrap_or(&*rank_pos);
        let hi = bucket_hi.as_ref().unwrap_or(&*rank_pos);
        let RankBiasDevice {
            pairs,
            scales,
            finalize,
            update,
            ti,
            tj,
            repl,
            ctl,
        } = b;
        let mut launch_pairs = |mode: u32,
                                ctl: &mut CudaSlice<u64>,
                                repl: &mut CudaSlice<i64>,
                                acc_g: &mut CudaSlice<i64>,
                                acc_h: &mut CudaSlice<i64>,
                                group_sum: &mut CudaSlice<i64>|
         -> Result<()> {
            let mut l = self.stream.launch_builder(pairs);
            l.arg(pred)
                .arg(label)
                .arg(&*group_ptr)
                .arg(&*init_ptr)
                .arg(&run.n_groups)
                .arg(&run.total_slots)
                .arg(&*rank_idx)
                .arg(&*rank_pos)
                .arg(&*inv_idcg)
                .arg(n_rel)
                .arg(acc)
                .arg(y_order)
                .arg(lo)
                .arg(hi)
                .arg(&run.kind)
                .arg(&run.is_topk)
                .arg(&run.num_pair)
                .arg(&run.exp_gain)
                .arg(&run.score_norm)
                .arg(&seed)
                .arg(&*ti)
                .arg(&*tj)
                .arg(&run.n_pos)
                .arg(&mode)
                .arg(ctl)
                .arg(repl)
                .arg(acc_g)
                .arg(acc_h)
                .arg(group_sum);
            unsafe { l.launch(pair_cfg) }.context("launch rank_pairs_ub")?;
            Ok(())
        };
        // 4. bound pass (reads t+ / t-, writes the maxima into the control block)
        launch_pairs(0, ctl, repl, acc_g, acc_h, group_sum)?;
        // the scales from the bounds
        {
            let cfg = LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (1, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut l = self.stream.launch_builder(scales);
            l.arg(&mut *ctl).arg(&run.bound_row).arg(&run.bound_total);
            unsafe { l.launch(cfg) }.context("launch rank_bias_scales")?;
        }
        // 5. accumulate pass
        launch_pairs(1, ctl, repl, acc_g, acc_h, group_sum)?;
        // 6. finalize with the device scale
        {
            let cfg = LaunchConfig {
                grid_dim: (grid_rows, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            };
            let n_rows = u32::try_from(self.n_rows).context("row count exceeds u32")?;
            let mut l = self.stream.launch_builder(finalize);
            l.arg(&*group_ptr)
                .arg(&run.n_groups)
                .arg(&n_rows)
                .arg(&*acc_g)
                .arg(&*acc_h)
                .arg(&*group_sum)
                .arg(&*group_weight)
                .arg(&run.has_group_weight)
                .arg(&run.weight_norm)
                .arg(&run.is_topk)
                .arg(&run.num_pair)
                .arg(&run.normalization)
                .arg(&mut *ctl)
                .arg(out)
                .arg(&subsample_threshold)
                .arg(&subsample_seed)
                .arg(&subsample_tree);
            unsafe { l.launch(cfg) }.context("launch rank_finalize_ub")?;
        }
        // the biases for the next round (after this round's gradients, like XGBoost)
        {
            let cfg = LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut l = self.stream.launch_builder(update);
            l.arg(&mut *ti)
                .arg(&mut *tj)
                .arg(&*repl)
                .arg(&*ctl)
                .arg(&run.n_pos)
                .arg(&run.regularizer);
            unsafe { l.launch(cfg) }.context("launch rank_bias_update")?;
        }
        // one word back: the status of the round
        let mut status = [0u64; 1];
        self.stream
            .memcpy_dtoh(&ctl.slice(4..5), &mut status)
            .context("rank status D2H")?;
        self.stream.synchronize().context("rank status sync")?;
        ensure!(
            status[0] == 0,
            "lambdarank_unbiased on the device met a non-finite {} (diverged scores, or a \
             position-bias estimate that degenerated); use gpu_math=\"exact\" to reproduce it \
             on the host",
            if status[0] & 1 != 0 {
                "pair statistic"
            } else {
                "gradient"
            }
        );
        Ok(())
    }

    /// The position biases `(t+, t-)` of the device estimation as stored in the model
    /// (f32); `None` without `lambdarank_unbiased`.
    pub fn fast_rank_bias(&self) -> Result<Option<(Vec<f32>, Vec<f32>)>> {
        let guard = self.lock()?;
        let Some(rank) = guard.rank.as_ref() else {
            return Ok(None);
        };
        let (Some(b), Some(run)) = (rank.bias.as_ref(), rank.run) else {
            return Ok(None);
        };
        let n = run.n_pos as usize;
        let (mut ti, mut tj) = (vec![0.0f64; n], vec![0.0f64; n]);
        self.stream
            .memcpy_dtoh(&b.ti.slice(0..n), &mut ti)
            .context("rank t+ D2H")?;
        self.stream
            .memcpy_dtoh(&b.tj.slice(0..n), &mut tj)
            .context("rank t- D2H")?;
        self.stream.synchronize().context("rank bias sync")?;
        Ok(Some((
            ti.iter().map(|&v| v as f32).collect(),
            tj.iter().map(|&v| v as f32).collect(),
        )))
    }
}
