use super::*;
use std::{cmp::Ordering, collections::BinaryHeap, sync::Arc};

use crate::{
    backend::cuda::{GpuBlockSession, GpuRowSpan, GpuTrainCtx},
    colsample::NestedSampler,
    interaction::{InteractionConstraints, InteractionState},
};

// Bound the histograms retained by queued leaves. A cache miss builds both
// child histograms directly. Fixed-point accumulation makes that equivalent
// to building the smaller child and subtracting it from its parent's bins.
const FRONTIER_HISTOGRAM_BYTES: usize = 32 * 1024 * 1024;

type BlockCandidates = Vec<(Option<Histogram>, Option<SplitCandidate>)>;

struct Leaf {
    id: usize,
    rows: NodeRows,
    depth: u32,
    sum: GradPairFixed,
    bounds: split::NodeBounds,
    interaction: Option<Arc<InteractionState>>,
}

struct PendingSplit {
    leaf: Leaf,
    split: SplitCandidate,
    histograms: Option<Vec<Option<Histogram>>>,
}

impl PartialEq for PendingSplit {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other).is_eq()
    }
}

impl Eq for PendingSplit {}

impl PartialOrd for PendingSplit {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PendingSplit {
    fn cmp(&self, other: &Self) -> Ordering {
        self.split
            .gain
            .total_cmp(&other.split.gain)
            .then_with(|| other.leaf.id.cmp(&self.leaf.id))
    }
}

struct Search<'a> {
    source: &'a dyn BlockSource,
    plans: &'a [BlockPlan],
    gpair: &'a [GradPairFixed],
    quantizer: &'a GradQuantizer,
    params: &'a TrainParams,
    comm: &'a dyn Comm,
    gpu: Option<&'a GpuTrainCtx>,
    sampled: Option<&'a [FeatId]>,
    nested: Option<NestedSampler<'a>>,
    tree_features: &'a [FeatId],
    used_bytes: usize,
    pending: BinaryHeap<PendingSplit>,
    device_leaves: Vec<(GpuRowSpan, f32)>,
}

impl Search<'_> {
    #[allow(clippy::too_many_arguments)]
    fn evaluate(
        &mut self,
        leaves: Vec<Leaf>,
        parent_id: Option<usize>,
        parent: Option<Vec<Option<Histogram>>>,
        remaining_splits: u32,
        tree: &Tree,
        rows: &[RowId],
        predictions: &mut [f32],
    ) -> anyhow::Result<()> {
        let depth = leaves[0].depth;
        let retain_histograms = remaining_splits > 1 && depth + 1 < self.params.max_depth;
        let level_set = self.nested.map(|n| n.level_set(self.tree_features, depth));
        let masks: Vec<Option<Vec<FeatId>>> = leaves
            .iter()
            .map(|leaf| {
                let mut features = match (self.nested, &level_set) {
                    (Some(n), Some(set)) if n.samples_nodes() => Some(n.node_set(set, leaf.id)),
                    (_, Some(set)) => Some(set.clone()),
                    _ => self.sampled.map(<[_]>::to_vec),
                };
                if let Some(state) = &leaf.interaction {
                    let allowed = state.eligible();
                    if let Some(mask) = &mut features {
                        mask.retain(|feature| allowed.binary_search(feature).is_ok());
                    } else {
                        features = Some(allowed.to_vec());
                    }
                }
                features
            })
            .collect();
        let small_child = parent_id.map(|_| {
            debug_assert_eq!(leaves.len(), 2);
            usize::from(leaves[0].rows.len() > leaves[1].rows.len())
        });
        let parents = parent.unwrap_or_else(|| (0..self.plans.len()).map(|_| None).collect());
        let cuts = self.source.cuts();
        let categories = cuts.has_categorical().then(|| cuts.categorical_flags());
        let phase = prof::Timer::new(&prof::HIST_PHASE_WALL);
        #[allow(unused_mut)] // Only CUDA populates the packet results.
        let mut packets: Vec<Option<BlockCandidates>> =
            (0..self.plans.len()).map(|_| None).collect();
        #[cfg(feature = "cuda")]
        if let Some(gpu) = self.gpu.filter(|gpu| {
            self.plans.len() > 1
                && self
                    .plans
                    .iter()
                    .all(|plan| !plan.compact && plan.physical < gpu.resident_block_count())
        }) {
            let mut submitted = Vec::with_capacity(self.plans.len());
            for (index, plan) in self.plans.iter().enumerate() {
                let started = std::time::Instant::now();
                let Some(mut session) = gpu.block_session_resident_width(
                    plan.physical,
                    cuts.block_width(&plan.handed_feats),
                    self.source.n_rows(),
                    plan.handed_feats.len(),
                    &plan.hist_offsets,
                    self.sampled.map(|_| plan.orig_local.as_slice()),
                )?
                else {
                    continue;
                };
                if let Some(pending) = self.submit_packet_splits(
                    &mut session,
                    plan,
                    &leaves,
                    &masks,
                    categories,
                    parent_id,
                    small_child,
                    retain_histograms,
                )? {
                    submitted.push((index, started, session, pending));
                }
            }
            for (index, started, mut session, pending) in submitted {
                packets[index] = Some(collect_packet_splits(
                    &mut session,
                    pending,
                    &leaves,
                    parent_id,
                )?);
                record_histogram_timing(&session, started);
            }
        }
        let results = self
            .plans
            .par_iter()
            .zip(parents.into_par_iter())
            .zip(packets.into_par_iter())
            .map(|((plan, parent_hist), packets)| {
                if let Some(results) = packets {
                    return Ok(results);
                }
                let physical = plan.physical;
                let selected_local =
                    (self.sampled.is_some() && !plan.compact).then_some(plan.orig_local.as_slice());
                let run = |block: Option<&ColumnBlock>, resident: Option<GpuBlockSession<'_>>| {
                    if let Some(block) = block {
                        validate_planned_block(self.source, plan, block)?;
                    }
                    let small_host_child = parent_hist.as_ref().and(small_child);
                    let width = *plan.hist_offsets.last().expect("histogram width") as usize;
                    let mut histograms: Vec<_> =
                        (0..leaves.len()).map(|_| Histogram::zeros(0)).collect();
                    let _timer = prof::Timer::new(&prof::HIST);
                    if let Some(gpu) = self.gpu {
                        let started = std::time::Instant::now();
                        let mut session = match resident {
                            Some(session) => session,
                            None => {
                                let block =
                                    block.context("GPU cache miss requires source payload")?;
                                gpu.block_session_bins(
                                    physical,
                                    block.bins(),
                                    block.n_rows,
                                    block.n_feats(),
                                    &plan.hist_offsets,
                                    selected_local,
                                )?
                            }
                        };
                        #[cfg(feature = "cuda")]
                        if let Some(pending) = self.submit_packet_splits(
                            &mut session,
                            plan,
                            &leaves,
                            &masks,
                            categories,
                            parent_id,
                            small_child,
                            retain_histograms,
                        )? {
                            let results =
                                collect_packet_splits(&mut session, pending, &leaves, parent_id)?;
                            record_histogram_timing(&session, started);
                            return Ok(results);
                        }

                        let (spans, mut output): (Vec<_>, Vec<_>) = leaves
                            .iter()
                            .zip(&mut histograms)
                            .enumerate()
                            .filter(|(i, _)| small_host_child.is_none_or(|small| *i == small))
                            .map(|(_, (leaf, histogram))| (leaf.rows.device(), &mut histogram.bins))
                            .unzip();
                        let capacity = session.hist_nodes_batch_capacity();
                        if capacity > 1 || session.requires_batched_hist() {
                            for (spans, output) in
                                spans.chunks(capacity).zip(output.chunks_mut(capacity))
                            {
                                session.histogram_device_batch(spans, output)?;
                            }
                        } else {
                            for (span, output) in spans.into_iter().zip(output) {
                                session.histogram_device(span, output)?;
                            }
                        }
                        record_histogram_timing(&session, started);
                    } else {
                        let block = block.context("CPU histogram requires source payload")?;
                        crate::with_bins!(block.bins(), |data| {
                            let all_cols = hist::block_columns(data, block.n_rows, block.n_feats());
                            let cols: Vec<&[_]> = if let Some(indices) = selected_local {
                                indices.iter().map(|&i| all_cols[i]).collect()
                            } else {
                                all_cols
                            };
                            let mut targets: Vec<_> = leaves
                                .iter()
                                .zip(&mut histograms)
                                .enumerate()
                                .filter(|(i, _)| small_host_child.is_none_or(|small| *i == small))
                                .map(|(_, (leaf, histogram))| {
                                    *histogram = Histogram::zeros(width);
                                    (leaf.rows.host(rows), histogram.bins.as_mut_slice())
                                })
                                .collect();
                            hist::build_nodes_cols(
                                &cols,
                                &plan.hist_offsets,
                                self.gpair,
                                &mut targets,
                            );
                        });
                    }
                    drop(_timer);
                    if let (Some(small), Some(mut histogram)) = (small_host_child, parent_hist) {
                        let _timer = prof::Timer::new(&prof::SUBTRACT);
                        hist::subtract_in_place(&mut histogram, &histograms[small]);
                        histograms[1 - small] = histogram;
                    }
                    let _timer = prof::Timer::new(&prof::ENUM);
                    Ok(histograms
                        .into_iter()
                        .zip(leaves.iter().zip(&masks))
                        .map(|(histogram, (leaf, mask))| {
                            let candidate = split::best_split_in_block_typed(
                                &histogram,
                                &plan.hist_feats,
                                &plan.hist_offsets,
                                leaf.sum,
                                self.quantizer,
                                self.params,
                                mask.as_deref(),
                                monotone_at(self.params, leaf.bounds),
                                categories,
                            );
                            (retain_histograms.then_some(histogram), candidate)
                        })
                        .collect::<Vec<_>>())
                };
                if !plan.compact {
                    if let Some(gpu) = self.gpu {
                        if let Some(resident) = gpu.block_session_resident_width(
                            physical,
                            cuts.block_width(&plan.handed_feats),
                            self.source.n_rows(),
                            plan.handed_feats.len(),
                            &plan.hist_offsets,
                            selected_local,
                        )? {
                            return run(None, Some(resident));
                        }
                    }
                }
                if plan.compact {
                    self.source
                        .with_block_cols_ret(physical, &plan.orig_local, |block| {
                            run(Some(block), None)
                        })?
                } else {
                    self.source
                        .with_block_ret(physical, |block| run(Some(block), None))?
                }
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        drop(phase);

        let mut candidates = vec![None; leaves.len()];
        let mut leaf_histograms: Vec<_> = (0..leaves.len()).map(|_| Vec::new()).collect();
        for block in results {
            for (index, (histogram, candidate)) in block.into_iter().enumerate() {
                candidates[index] = split::reduce_best(&[candidates[index].take(), candidate]);
                leaf_histograms[index].push(histogram);
            }
        }
        for ((leaf, candidate), histograms) in
            leaves.into_iter().zip(candidates).zip(leaf_histograms)
        {
            if let Some(split) = self.comm.allreduce_best_split(candidate) {
                let bytes = histogram_bytes(&histograms);
                let histograms =
                    if bytes != 0 && bytes <= FRONTIER_HISTOGRAM_BYTES - self.used_bytes {
                        self.used_bytes += bytes;
                        Some(histograms)
                    } else {
                        None
                    };
                self.pending.push(PendingSplit {
                    leaf,
                    split,
                    histograms,
                });
            } else {
                self.finish_leaf(tree, &leaf, rows, predictions);
            }
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[allow(clippy::too_many_arguments)]
    fn submit_packet_splits(
        &self,
        session: &mut GpuBlockSession<'_>,
        plan: &BlockPlan,
        leaves: &[Leaf],
        masks: &[Option<Vec<FeatId>>],
        categories: Option<&[bool]>,
        parent_id: Option<usize>,
        small_child: Option<usize>,
        retain_histograms: bool,
    ) -> anyhow::Result<Option<crate::backend::cuda::GpuSplitLevel>> {
        use crate::backend::cuda::GpuSplitGroup;

        if !session.begin_lossguide_split(parent_id.is_none())? {
            return Ok(None);
        }
        let group = |index: usize| GpuSplitGroup {
            small: index,
            span: leaves[index].rows.device(),
            sum: leaves[index].sum,
            bounds: leaves[index].bounds,
            large: None,
        };
        let groups = match (
            small_child,
            parent_id.and_then(|id| session.cached_parent_slot(id)),
        ) {
            (Some(small), Some(slot)) => {
                let big = 1 - small;
                vec![GpuSplitGroup {
                    large: Some((big, slot, leaves[big].sum, leaves[big].bounds)),
                    ..group(small)
                }]
            }
            _ => (0..leaves.len()).map(group).collect(),
        };
        let enabled = |i: usize, f: FeatId| {
            masks[i]
                .as_ref()
                .is_none_or(|mask| mask.binary_search(&f).is_ok())
        };
        let has_node_masks = self.nested.is_some() || leaves[0].interaction.is_some();
        let pending = session.split_cached_submit(
            &groups,
            &plan.hist_feats,
            self.sampled,
            categories,
            has_node_masks.then_some(&enabled as &dyn Fn(usize, FeatId) -> bool),
            self.quantizer,
            self.params,
            retain_histograms,
        )?;
        Ok(Some(pending))
    }

    fn finish_leaf(&mut self, tree: &Tree, leaf: &Leaf, rows: &[RowId], predictions: &mut [f32]) {
        let value = tree.nodes[leaf.id].leaf_value;
        match leaf.rows {
            NodeRows::Device(span) => self.device_leaves.push((span, value)),
            NodeRows::Host(span) => {
                let _timer = prof::Timer::new(&prof::UPDATE_PRED);
                apply_host_leaf_updates(predictions, &[(span.of(rows), value)]);
            }
        }
    }

    fn partition_gpu(
        &self,
        gpu: &GpuTrainCtx,
        span: GpuRowSpan,
        split: &SplitCandidate,
    ) -> anyhow::Result<(NodeRows, NodeRows)> {
        let started = std::time::Instant::now();
        let block_and_column = self.plans.iter().find_map(|plan| {
            let index = plan.sampled_global.binary_search(&split.feat).ok()?;
            Some((plan.physical, plan.orig_local[index]))
        });
        let resident = match block_and_column {
            Some((block, local)) => gpu.partition_session_resident(block, local)?,
            None => None,
        };
        let owned_column;
        let mut session = match resident {
            Some(session) => session,
            None => {
                let _fetch = prof::Timer::new(&prof::PART_FETCH);
                let column = match self.source.resident_feature_column(split.feat) {
                    Some(column) => column,
                    None => {
                        owned_column = self.source.feature_column(split.feat)?;
                        owned_column.as_slice()
                    }
                };
                gpu.partition_session_bins(column)?
            }
        };
        match &split.cats {
            Some(categories) => {
                session.partition_categorical(span, categories, split.missing_left)?
            }
            None => session.partition(span, split.bin, split.missing_left)?,
        };
        let timing = session.timing();
        drop(session);
        prof::record_gpu(
            &timing,
            started.elapsed().as_nanos() as u64,
            &prof::GPU_PART_COL_H2D,
            &prof::GPU_PART_ROWS_H2D,
            &prof::GPU_PART_KERNEL,
            &prof::GPU_PART_D2H,
            &prof::GPU_PART_HOST,
        );
        let (left, right) = gpu.commit_leaf_partition(span)?;
        Ok((NodeRows::Device(left), NodeRows::Device(right)))
    }
}

#[cfg(feature = "cuda")]
fn collect_packet_splits(
    session: &mut GpuBlockSession<'_>,
    pending: crate::backend::cuda::GpuSplitLevel,
    leaves: &[Leaf],
    parent_id: Option<usize>,
) -> anyhow::Result<BlockCandidates> {
    let results = session.split_cached_finish(pending)?;
    let mut decoded: Vec<_> = (0..leaves.len()).map(|_| (None, None)).collect();
    for (i, candidate, histogram) in results {
        decoded[i] = (histogram.map(|bins| Histogram { bins }), candidate);
    }
    let ids: Vec<_> = leaves.iter().map(|leaf| leaf.id).collect();
    session.retain_lossguide_histograms(parent_id, &ids)?;
    Ok(decoded)
}

fn record_histogram_timing(session: &GpuBlockSession<'_>, started: std::time::Instant) {
    prof::record_gpu(
        &session.timing(),
        started.elapsed().as_nanos() as u64,
        &prof::GPU_HIST_BLOCK_H2D,
        &prof::GPU_HIST_ROWS_H2D,
        &prof::GPU_HIST_KERNEL,
        &prof::GPU_HIST_D2H,
        &prof::GPU_HIST_HOST,
    );
}

fn histogram_bytes(histograms: &[Option<Histogram>]) -> usize {
    histograms
        .iter()
        .flatten()
        .map(|histogram| histogram.len() * std::mem::size_of::<GradPairFixed>())
        .sum()
}

#[allow(clippy::too_many_arguments)]
pub(super) fn grow(
    source: &dyn BlockSource,
    plans: &[BlockPlan],
    gpair: &[GradPairFixed],
    quantizer: &GradQuantizer,
    params: &TrainParams,
    comm: &dyn Comm,
    gpu: Option<&GpuTrainCtx>,
    sampled: Option<&[FeatId]>,
    interactions: Option<&InteractionConstraints>,
    nested: Option<NestedSampler<'_>>,
    tree_features: &[FeatId],
    tree: &mut Tree,
    root: LevelNode,
    rows: &mut [RowId],
    predictions: &mut [f32],
    drop_zero_rows: bool,
    split_info: &mut Vec<DropSplit>,
) -> anyhow::Result<()> {
    let root = Leaf {
        id: root.node,
        rows: root.rows,
        depth: 0,
        sum: root.sum,
        bounds: root.bounds,
        interaction: root.interaction,
    };
    let mut search = Search {
        source,
        plans,
        gpair,
        quantizer,
        params,
        comm,
        gpu,
        sampled,
        nested,
        tree_features,
        used_bytes: 0,
        pending: BinaryHeap::new(),
        device_leaves: Vec::new(),
    };
    let remaining = |count| {
        if params.max_leaves == 0 {
            u32::MAX
        } else {
            params.max_leaves - count
        }
    };
    if params.max_depth == 0 || params.max_leaves == 1 {
        search.finish_leaf(tree, &root, rows, predictions);
    } else {
        search.evaluate(
            vec![root],
            None,
            None,
            remaining(1),
            tree,
            rows,
            predictions,
        )?;
    }
    let mut leaf_count = 1;
    let mut partition_buffer = Vec::new();

    while params.max_leaves == 0 || leaf_count < params.max_leaves {
        let Some(PendingSplit {
            leaf,
            split,
            histograms,
        }) = search.pending.pop()
        else {
            break;
        };
        if let Some(histograms) = &histograms {
            search.used_bytes -= histogram_bytes(histograms);
        }

        let (left_rows, right_rows, split) = match leaf.rows {
            NodeRows::Device(span) => {
                let (left, right) =
                    search.partition_gpu(gpu.expect("device rows require CUDA"), span, &split)?;
                (left, right, split)
            }
            NodeRows::Host(input) => {
                let span = HostSpan {
                    start: 0,
                    len: input.len,
                };
                let nodes = [(span, &split)];
                let range = input.start..input.start + input.len;
                partition_buffer.resize(span.len, 0);
                let partitions = {
                    let _phase = crate::diagnostic::Phase::new(b"fb.host.partition\0");
                    partition_level_host(
                        &rows[range.clone()],
                        &mut partition_buffer,
                        &nodes,
                        PartCols::for_nodes(source, &nodes),
                        comm,
                    )?
                };
                rows[range].copy_from_slice(&partition_buffer);
                let (mut left_rows, mut right_rows) = partitions[0];
                left_rows.start += input.start;
                right_rows.start += input.start;
                (NodeRows::Host(left_rows), NodeRows::Host(right_rows), split)
            }
        };

        let values = child_leaves(&split, quantizer, params, leaf.bounds);
        let threshold = if split.cats.is_some() {
            0.0
        } else {
            tree::split_threshold(source.cuts(), split.feat, split.bin)
        };
        let (left, right) = tree.split_leaf(
            leaf.id,
            split.feat,
            threshold,
            split.missing_left,
            split.gain,
            values.left_value,
            hess_of(split.left, quantizer),
            values.right_value,
            hess_of(split.right, quantizer),
        );
        if let Some(categories) = &split.cats {
            tree.set_categories(leaf.id, categories);
        }
        if drop_zero_rows {
            if split_info.len() <= leaf.id {
                split_info.resize(leaf.id + 1, DropSplit::default());
            }
            split_info[leaf.id] = DropSplit {
                feat: split.feat,
                bin: split.bin,
                missing_left: split.missing_left,
                cats: split.cats,
            };
        }
        leaf_count += 1;
        let depth = leaf.depth + 1;
        let interaction = interactions.map(|rules| {
            Arc::new(
                rules
                    .child_after_split(
                        leaf.interaction.as_ref().expect("interaction state"),
                        split.feat,
                    )
                    .expect("selected split must satisfy interaction constraints"),
            )
        });
        let children = vec![
            Leaf {
                id: left,
                rows: left_rows,
                depth,
                sum: split.left,
                bounds: values.left_bounds,
                interaction: interaction.clone(),
            },
            Leaf {
                id: right,
                rows: right_rows,
                depth,
                sum: split.right,
                bounds: values.right_bounds,
                interaction,
            },
        ];
        if depth == params.max_depth || leaf_count == params.max_leaves {
            for child in &children {
                search.finish_leaf(tree, child, rows, predictions);
            }
        } else {
            search.evaluate(
                children,
                Some(leaf.id),
                histograms,
                remaining(leaf_count),
                tree,
                rows,
                predictions,
            )?;
        }
    }
    for pending in std::mem::take(&mut search.pending) {
        search.finish_leaf(tree, &pending.leaf, rows, predictions);
    }
    if let Some(gpu) = gpu {
        gpu.assign_leaf_values(&search.device_leaves)?;
    }
    Ok(())
}
