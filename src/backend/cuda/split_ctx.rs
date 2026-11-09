//! Bounded, per-resident-block split state. No device allocations occur at a level.
use super::*;
use crate::gpu_mem_plan::{CAT_SET_WORDS, SPLIT_PACKET_BYTES, SPLIT_PACKET_WINDOW_BATCHES};
use crate::split::SplitCandidate;
use crate::types::{FeatId, TrainParams};

#[cfg(test)]
#[path = "packet_tests.rs"]
mod packet_tests;

/// Most bins per feature the one-thread-per-bin split kernels take (every
/// 8-bit feature; `SPLIT_CTA_THREADS` = 256). A block holding a wider feature
/// (16-bit blocks: max_bin > 255, more than 254 categories) runs the tiled
/// `split_cached_features_wide*` kernels instead.
const SPLIT_KERNEL_MAX_BINS: u32 = 255;

pub(super) struct ResidentSplitBuffers {
    previous: CudaSlice<i64>,
    next: CudaSlice<i64>,
    candidates: CudaSlice<u8>,
    packets: CudaSlice<u8>,
    /// Pinned host mirror of `packets`: the last window of a level is copied
    /// asynchronously right after its kernels are queued, so collecting it
    /// costs a stream wait instead of a synchronous pageable D2H.
    packets_host: PinnedHostSlice<u8>,
    /// Bytes of `packets` already queued into `packets_host`.
    packets_in_flight: Option<usize>,
    sums: CudaSlice<i64>,
    metadata: CudaSlice<u32>,
    feats: CudaSlice<u32>,
    enabled: CudaSlice<u32>,
    /// 单调约束:每节点 `[lower, upper]`(和 `sums` 同序),每个块内特征的方向。
    /// 只在有约束时上传和读取。
    bounds: CudaSlice<f32>,
    monotone: CudaSlice<i32>,
    /// Categorical path (allocated only when the data has categorical
    /// features): per block-local feature `0` = ordered, else `1 + ordinal`
    /// among the block's categorical features; per-(node, categorical feature)
    /// winning category sets (8 words each).
    cat_info: CudaSlice<u32>,
    cat_sets: CudaSlice<u32>,
    cat_enabled: bool,
    /// Per-(node, block-local feature) enable bytes for bylevel / bynode column
    /// sampling (a one-byte dummy otherwise).
    node_enabled: CudaSlice<u8>,
    node_masks: bool,
    /// Bytes between consecutive node packets of the current block:
    /// `SPLIT_PACKET_BYTES`, plus a category set on the categorical path (8
    /// words, or the widest set of a 16-bit block's categorical features).
    packet_stride: usize,
    /// `packet_stride` of 8-bit blocks.
    base_packet_stride: usize,
    n_cat: u32,
    /// The current block has a feature above `SPLIT_KERNEL_MAX_BINS` bins
    /// and runs the wide kernels (decided per tree in `begin_split_level`).
    wide: bool,
    /// Wide categorical path (allocated only for 16-bit categorical
    /// features, priced by `wide_categorical_split_cache_bytes`):
    /// per-(node, categorical feature) weight keys and category sets, and the
    /// per-ordinal layout `[set offset, set words, key offset]`.
    wide_keys: CudaSlice<u32>,
    wide_sets: CudaSlice<u32>,
    wide_layout: CudaSlice<u32>,
    /// Per-node set words / key elements of the current wide block.
    wide_set_words: u32,
    wide_key_elems: u32,
    /// Allocated per-node capacity of `wide_sets` / `wide_keys` and of the
    /// layout (in categorical features).
    wide_set_capacity: usize,
    wide_key_capacity: usize,
    wide_cat_capacity: usize,
    /// Set words of each categorical feature of the current wide block (the
    /// packet decode trims the set to its feature's own size, like the host
    /// `CatBits::for_bins`).
    wide_cat_words: Vec<(FeatId, usize)>,
    stride: usize,
    capacity: usize,
    previous_ids: Vec<(usize, usize)>,
    next_ids: Vec<(usize, usize)>,
    depth: Option<u32>,
    layout: Vec<u32>,
    features_ready: bool,
}

impl ResidentSplitBuffers {
    pub(super) fn new(
        stream: &Arc<CudaStream>,
        bins: usize,
        feats: usize,
        cap: usize,
        categorical: bool,
        node_masks: bool,
        shape: &crate::gpu_mem_plan::GpuBlockShape,
    ) -> Result<Self> {
        let packet_stride = SPLIT_PACKET_BYTES + if categorical { CAT_SET_WORDS * 4 } else { 0 };
        // 16-bit categorical sets widen the packet window to the widest set
        // (`wide_categorical_split_cache_bytes` prices the difference).
        let wide_cat = categorical && shape.wide_cat_feats > 0;
        let max_packet_stride = if wide_cat {
            SPLIT_PACKET_BYTES + CAT_SET_WORDS.max(shape.cat_mask_words) * 4
        } else {
            packet_stride
        };
        let packet_bytes = cap * max_packet_stride * SPLIT_PACKET_WINDOW_BATCHES;
        let (wide_set_capacity, wide_key_capacity, wide_cat_capacity) = if wide_cat {
            (
                shape.wide_cat_set_words,
                shape.wide_cat_bins,
                shape.wide_cat_feats,
            )
        } else {
            (0, 0, 0)
        };
        // SAFETY: the pinned bytes are only read after a D2H has written them
        // (`collect_split_packets` waits for the copy's event first).
        let packets_host = unsafe { stream.context().alloc_pinned::<u8>(packet_bytes.max(1)) }?;
        Ok(Self {
            previous: stream.alloc_zeros::<i64>((bins * 2 * cap).max(1))?,
            next: stream.alloc_zeros::<i64>((bins * 2 * cap).max(1))?,
            candidates: stream.alloc_zeros::<u8>((feats * cap * 64).max(1))?,
            packets: stream.alloc_zeros::<u8>(packet_bytes)?,
            packets_host,
            packets_in_flight: None,
            sums: stream.alloc_zeros::<i64>(cap * 2)?,
            metadata: stream.alloc_zeros::<u32>(cap * 4)?,
            feats: stream.alloc_zeros::<u32>(feats.max(1))?,
            enabled: stream.alloc_zeros::<u32>(feats.max(1))?,
            bounds: stream.alloc_zeros::<f32>(cap * 2)?,
            monotone: stream.alloc_zeros::<i32>(feats.max(1))?,
            cat_info: stream.alloc_zeros::<u32>(if categorical { feats.max(1) } else { 1 })?,
            cat_sets: stream.alloc_zeros::<u32>(if categorical {
                (cap * feats * CAT_SET_WORDS).max(1)
            } else {
                1
            })?,
            cat_enabled: categorical,
            node_enabled: stream.alloc_zeros::<u8>(if node_masks {
                (cap * feats).max(1)
            } else {
                1
            })?,
            node_masks,
            packet_stride,
            base_packet_stride: packet_stride,
            n_cat: 0,
            wide: false,
            wide_keys: stream.alloc_zeros::<u32>((cap * wide_key_capacity).max(1))?,
            wide_sets: stream.alloc_zeros::<u32>((cap * wide_set_capacity).max(1))?,
            wide_layout: stream.alloc_zeros::<u32>((wide_cat_capacity * 3).max(1))?,
            wide_set_words: 0,
            wide_key_elems: 0,
            wide_set_capacity,
            wide_key_capacity,
            wide_cat_capacity,
            wide_cat_words: Vec::new(),
            stride: bins,
            capacity: cap,
            previous_ids: Vec::with_capacity(cap),
            next_ids: Vec::with_capacity(cap),
            depth: None,
            layout: Vec::new(),
            features_ready: false,
        })
    }
}

pub(crate) struct GpuSplitGroup {
    pub small: usize,
    pub span: GpuRowSpan,
    pub sum: GradPairFixed,
    /// 单调约束下 small 节点的权重界(无约束时不读)。
    pub bounds: crate::split::NodeBounds,
    /// (large child index, cached parent slot, large child sum, large child bounds).
    pub large: Option<(usize, usize, GradPairFixed, crate::split::NodeBounds)>,
}
pub(crate) type GpuSplitResult = (usize, Option<SplitCandidate>, Option<Vec<GradPairFixed>>);

/// A level whose last packet window is queued but not yet collected
/// ([`GpuBlockSession::split_cached_submit`]).
pub(crate) struct GpuSplitLevel {
    pending: Vec<PendingSplitBatch>,
    packet_nodes: usize,
    results: Vec<GpuSplitResult>,
    keep_next: bool,
}

struct PendingSplitBatch {
    indices: Vec<usize>,
    working_slots: Vec<usize>,
    cached: Vec<bool>,
}

impl GpuBlockSession<'_> {
    pub(crate) fn begin_split_level(&mut self, depth: u32) -> Result<bool> {
        if self.n_feats == 0 || self.n_hist == 0 {
            return Ok(false);
        }
        let has_wide_split = self.ctx.wide.as_ref().is_some_and(|w| w.split.is_some());
        let slot = self.slot.as_mut().expect("histogram slot lease missing");
        let super::GpuHistBuffers {
            split_cache,
            offsets_host,
            ..
        } = &mut **slot;
        let Some(cache) = split_cache.as_mut() else {
            return Ok(false);
        };
        // A block holding a feature wider than one CTA of bins (a 16-bit
        // block: max_bin > 255 or more than 254 categories) runs the tiled
        // wide kernels; they are loaded whenever a 16-bit block exists.
        let wide = offsets_host
            .windows(2)
            .any(|w| w[1] - w[0] > SPLIT_KERNEL_MAX_BINS);
        anyhow::ensure!(
            !wide || has_wide_split,
            "GPU split cache: wide split kernels missing for a 16-bit block"
        );
        if depth == 0 {
            cache.previous_ids.clear();
            cache.layout.clone_from(offsets_host);
            cache.features_ready = false;
            cache.wide = wide;
            cache.packet_stride = cache.base_packet_stride;
        } else {
            anyhow::ensure!(
                cache.depth == Some(depth - 1),
                "GPU split cache generation mismatch"
            );
            anyhow::ensure!(
                cache.layout == *offsets_host,
                "GPU split cache layout changed within a tree"
            );
        }
        cache.next_ids.clear();
        // A copy queued by an abandoned level must never stand in for a new one.
        cache.packets_in_flight = None;
        Ok(true)
    }

    pub(crate) fn cached_parent_slot(&mut self, parent: usize) -> Option<usize> {
        let found = self
            .slot
            .as_ref()?
            .split_cache
            .as_ref()?
            .previous_ids
            .iter()
            .find(|(id, _)| *id == parent)
            .map(|(_, slot)| *slot);
        if found.is_some() {
            self.timing.cache_parent_hits += 1;
        } else {
            self.timing.cache_parent_misses += 1;
        }
        found
    }

    pub(crate) fn finish_split_level(&mut self, depth: u32) {
        if let Some(cache) = self
            .slot
            .as_mut()
            .and_then(|slot| slot.split_cache.as_mut())
        {
            std::mem::swap(&mut cache.previous, &mut cache.next);
            std::mem::swap(&mut cache.previous_ids, &mut cache.next_ids);
            cache.depth = Some(depth);
        }
    }

    /// Build only small children, subtract cached parents on device, and return
    /// compact winners. Uncached histograms are downloaded only if another
    /// level can consume them; this is the exact bounded-capacity fallback.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn split_cached_batch(
        &mut self,
        groups: &[GpuSplitGroup],
        feats: &[FeatId],
        sampled: Option<&[FeatId]>,
        categorical: Option<&[bool]>,
        node_enabled: Option<&dyn Fn(usize, FeatId) -> bool>,
        q: &GradQuantizer,
        p: &TrainParams,
        keep_next: bool,
    ) -> Result<Vec<GpuSplitResult>> {
        let level = self.split_cached_submit(
            groups,
            feats,
            sampled,
            categorical,
            node_enabled,
            q,
            p,
            keep_next,
        )?;
        self.split_cached_finish(level)
    }

    /// First half of [`Self::split_cached_batch`]: queue every batch of the
    /// level, collecting only packet windows that must be drained before the
    /// next producer (full window or an uncached spill). The last window stays
    /// queued, so the caller can submit other blocks' levels before waiting in
    /// [`Self::split_cached_finish`].
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn split_cached_submit(
        &mut self,
        groups: &[GpuSplitGroup],
        feats: &[FeatId],
        sampled: Option<&[FeatId]>,
        categorical: Option<&[bool]>,
        node_enabled: Option<&dyn Fn(usize, FeatId) -> bool>,
        q: &GradQuantizer,
        p: &TrainParams,
        keep_next: bool,
    ) -> Result<GpuSplitLevel> {
        let cap = self.ctx.hist_nodes_per_batch;
        let mut pending = Vec::with_capacity(SPLIT_PACKET_WINDOW_BATCHES);
        let mut packet_nodes = 0;
        let mut results = Vec::new();
        for chunk in groups.chunks((cap / 2).max(1)) {
            let nodes: usize = chunk
                .iter()
                .map(|g| 1 + usize::from(g.large.is_some()))
                .sum();
            let cache = self
                .slot
                .as_ref()
                .and_then(|s| s.split_cache.as_ref())
                .context("resident split cache unavailable")?;
            let spills = keep_next && cache.next_ids.len() + nodes > cache.capacity;
            // An uncached histogram lives in reusable hist_out: collect before
            // another producer can overwrite it. Cached winners need only packets.
            // A full packet window is drained only when another batch needs
            // its slots, so a level that exactly fills the window stays queued
            // for `split_cached_finish`.
            if !pending.is_empty() && (spills || pending.len() == SPLIT_PACKET_WINDOW_BATCHES) {
                results.extend(self.collect_split_packets(&pending, packet_nodes, keep_next)?);
                pending.clear();
                packet_nodes = 0;
            }
            pending.push(self.submit_split_cached_batch(
                chunk,
                feats,
                sampled,
                categorical,
                node_enabled,
                q,
                p,
                keep_next,
                packet_nodes,
            )?);
            packet_nodes += nodes;
            if spills {
                results.extend(self.collect_split_packets(&pending, packet_nodes, keep_next)?);
                pending.clear();
                packet_nodes = 0;
            }
        }
        if !pending.is_empty() {
            self.enqueue_split_packets(packet_nodes)?;
        }
        Ok(GpuSplitLevel {
            pending,
            packet_nodes,
            results,
            keep_next,
        })
    }

    /// Queue the async copy of the first `nodes` packets into the pinned
    /// mirror on the slot's stream (no host wait).
    fn enqueue_split_packets(&mut self, nodes: usize) -> Result<()> {
        let slot = self.slot.as_mut().expect("histogram slot lease missing");
        let stream = slot.stream.clone();
        let cache = slot
            .split_cache
            .as_mut()
            .context("resident split cache unavailable")?;
        let bytes = nodes * cache.packet_stride;
        anyhow::ensure!(
            bytes <= cache.packets_host.len(),
            "GPU split packet window exceeds its pinned mirror"
        );
        let mut dst = super::PinnedPrefix {
            inner: &mut cache.packets_host,
            len: bytes,
        };
        stream.memcpy_dtoh(&cache.packets.slice(..bytes), &mut dst)?;
        cache.packets_in_flight = Some(bytes);
        self.timing.d2h_calls += 1;
        self.timing.d2h_bytes += bytes as u64;
        self.timing.split_d2h_calls += 1;
        self.timing.split_d2h_bytes += bytes as u64;
        Ok(())
    }

    /// Second half of [`Self::split_cached_batch`]: drain the last packet
    /// window and return all of the level's winners in submission order.
    pub(crate) fn split_cached_finish(
        &mut self,
        level: GpuSplitLevel,
    ) -> Result<Vec<GpuSplitResult>> {
        let GpuSplitLevel {
            pending,
            packet_nodes,
            mut results,
            keep_next,
        } = level;
        if !pending.is_empty() {
            results.extend(self.collect_split_packets(&pending, packet_nodes, keep_next)?);
        }
        Ok(results)
    }

    #[allow(clippy::too_many_arguments)] // Explicit packet offsets and feature identities prevent routing ambiguity.
    fn submit_split_cached_batch(
        &mut self,
        groups: &[GpuSplitGroup],
        feats: &[FeatId],
        sampled: Option<&[FeatId]>,
        categorical: Option<&[bool]>,
        node_enabled: Option<&dyn Fn(usize, FeatId) -> bool>,
        q: &GradQuantizer,
        p: &TrainParams,
        keep_next: bool,
        packet_offset: usize,
    ) -> Result<PendingSplitBatch> {
        anyhow::ensure!(!groups.is_empty(), "empty GPU split batch");
        anyhow::ensure!(
            feats.len() == self.n_feats,
            "GPU split feature shape mismatch"
        );
        let wide = self
            .slot
            .as_ref()
            .and_then(|s| s.split_cache.as_ref())
            .is_some_and(|c| c.wide);
        anyhow::ensure!(
            wide || self
                .feat_offsets()
                .windows(2)
                .all(|x| x[1] - x[0] <= SPLIT_KERNEL_MAX_BINS),
            "GPU split: a feature above 255 bins needs the wide split kernels"
        );
        let nodes = groups
            .iter()
            .map(|g| 1 + usize::from(g.large.is_some()))
            .sum::<usize>();
        anyhow::ensure!(
            nodes <= self.ctx.hist_nodes_per_batch,
            "GPU split batch exceeds capacity"
        );
        anyhow::ensure!(
            packet_offset + nodes <= self.ctx.hist_nodes_per_batch * SPLIT_PACKET_WINDOW_BATCHES,
            "GPU split packet window exceeds capacity"
        );
        let spans: Vec<_> = groups.iter().map(|g| g.span).collect();
        self.produce_histogram_device_batch(&spans)?;
        let slot = self.slot.as_mut().expect("histogram slot lease missing");
        let stream = slot.stream.clone();
        let GpuHistBuffers {
            split_cache,
            hist_out,
            offsets,
            offsets_host,
            start_event,
            end_event,
            ..
        } = &mut **slot;
        let cache = split_cache
            .as_mut()
            .context("resident split cache unavailable")?;
        let mut metadata = Vec::with_capacity(nodes * 4);
        let mut sums = Vec::with_capacity(nodes * 2);
        let mut bounds = Vec::with_capacity(nodes * 2);
        let mut indices = Vec::with_capacity(nodes);
        let mut working_slots = Vec::with_capacity(nodes);
        let mut cached = Vec::with_capacity(nodes);
        let mut large_slot = groups.len();
        for (small_slot, g) in groups.iter().enumerate() {
            if let Some((_, parent, _, _)) = g.large {
                anyhow::ensure!(
                    parent < cache.capacity,
                    "GPU parent cache slot out of range"
                );
            }
            let mut append = |index: usize,
                              work: usize,
                              sum: GradPairFixed,
                              node_bounds: crate::split::NodeBounds,
                              parent: u32| {
                let destination = if keep_next && cache.next_ids.len() < cache.capacity {
                    let dest = cache.next_ids.len();
                    cache.next_ids.push((index, dest));
                    dest as u32
                } else {
                    u32::MAX
                };
                metadata.extend_from_slice(&[work as u32, small_slot as u32, parent, destination]);
                sums.extend_from_slice(&[sum.grad, sum.hess]);
                bounds.extend_from_slice(&[node_bounds.lower, node_bounds.upper]);
                indices.push(index);
                working_slots.push(work);
                cached.push(destination != u32::MAX);
            };
            append(g.small, small_slot, g.sum, g.bounds, u32::MAX);
            if let Some((index, parent, sum, large_bounds)) = g.large {
                append(index, large_slot, sum, large_bounds, parent as u32);
                large_slot += 1;
            }
        }
        if self.ctx.timing {
            stream.synchronize()?;
        }
        let metadata_start = std::time::Instant::now();
        // Feature IDs and sampling are immutable within one tree's physical
        // block. Reset at depth zero, then reuse across batches and levels.
        if !cache.features_ready {
            let enabled: Vec<u32> = feats
                .iter()
                .map(|f| u32::from(sampled.is_none_or(|s| s.binary_search(f).is_ok())))
                .collect();
            stream.memcpy_htod(feats, &mut cache.feats.slice_mut(..feats.len()))?;
            stream.memcpy_htod(&enabled, &mut cache.enabled.slice_mut(..enabled.len()))?;
            self.timing.split_metadata_h2d_calls += 2;
            self.timing.split_metadata_h2d_bytes += (feats.len() * 8) as u64;
            if cache.cat_enabled {
                let is_cat = |f: &FeatId| {
                    categorical.is_some_and(|c| c.get(*f as usize).copied().unwrap_or(false))
                };
                let mut ordinal = 0u32;
                let info: Vec<u32> = feats
                    .iter()
                    .map(|f| {
                        if is_cat(f) {
                            ordinal += 1;
                            ordinal
                        } else {
                            0
                        }
                    })
                    .collect();
                cache.n_cat = ordinal;
                stream.memcpy_htod(&info, &mut cache.cat_info.slice_mut(..info.len()))?;
                self.timing.split_metadata_h2d_calls += 1;
                self.timing.split_metadata_h2d_bytes += (info.len() * 4) as u64;
                if cache.wide {
                    // Per categorical ordinal: set offset and words inside a
                    // node's sets, key offset inside a node's keys. Sets are
                    // whole 64-bit words (CatBits::words64), so packet tails
                    // stay 8-byte aligned.
                    let mut layout = Vec::with_capacity(ordinal as usize * 3);
                    let (mut set_words, mut key_elems, mut widest) = (0usize, 0usize, 0usize);
                    cache.wide_cat_words.clear();
                    for (local, f) in feats.iter().enumerate() {
                        if !is_cat(f) {
                            continue;
                        }
                        let bins = (offsets_host[local + 1] - offsets_host[local]) as usize;
                        let words = bins.div_ceil(64) * 2;
                        layout.extend_from_slice(&[
                            set_words as u32,
                            words as u32,
                            key_elems as u32,
                        ]);
                        cache.wide_cat_words.push((*f, words));
                        set_words += words;
                        key_elems += bins;
                        widest = widest.max(words);
                    }
                    anyhow::ensure!(
                        ordinal as usize <= cache.wide_cat_capacity
                            && set_words <= cache.wide_set_capacity
                            && key_elems <= cache.wide_key_capacity
                            && SPLIT_PACKET_BYTES + widest * 4
                                <= cache.packets_host.len()
                                    / (cache.capacity * SPLIT_PACKET_WINDOW_BATCHES),
                        "GPU split cache: 16-bit categorical block exceeds the planned \
                         split buffers"
                    );
                    if !layout.is_empty() {
                        stream.memcpy_htod(
                            &layout,
                            &mut cache.wide_layout.slice_mut(..layout.len()),
                        )?;
                        self.timing.split_metadata_h2d_calls += 1;
                        self.timing.split_metadata_h2d_bytes += (layout.len() * 4) as u64;
                    }
                    cache.wide_set_words = set_words as u32;
                    cache.wide_key_elems = key_elems as u32;
                    cache.packet_stride = SPLIT_PACKET_BYTES + widest * 4;
                }
            } else {
                anyhow::ensure!(
                    !categorical.is_some_and(|c| feats
                        .iter()
                        .any(|f| c.get(*f as usize).copied().unwrap_or(false))),
                    "categorical features need the categorical GPU split cache \
                     (SharedHistogramConfig::has_categorical)"
                );
            }
            if p.has_monotone() {
                let monotone: Vec<i32> = feats.iter().map(|&f| p.monotone_of(f) as i32).collect();
                stream.memcpy_htod(&monotone, &mut cache.monotone.slice_mut(..monotone.len()))?;
                self.timing.split_metadata_h2d_calls += 1;
                self.timing.split_metadata_h2d_bytes += (monotone.len() * 4) as u64;
            }
            cache.features_ready = true;
        }
        // Given-weight mode (always on, see `TrainParams::uses_given_weight`).
        // 没有单调约束时 monotone 缓冲区保持全 0、界是 ±f32::MAX。
        let has_monotone = u32::from(p.uses_given_weight());
        if has_monotone != 0 {
            stream.memcpy_htod(&bounds, &mut cache.bounds.slice_mut(..bounds.len()))?;
            self.timing.split_metadata_h2d_calls += 1;
            self.timing.split_metadata_h2d_bytes += (bounds.len() * 4) as u64;
        }
        if let Some(is_enabled) = node_enabled {
            anyhow::ensure!(
                cache.node_masks,
                "nested column sampling needs the per-node feature masks \
                 (SharedHistogramConfig::node_feature_masks)"
            );
            let mut mask = Vec::with_capacity(indices.len() * feats.len());
            for &index in &indices {
                mask.extend(feats.iter().map(|&f| u8::from(is_enabled(index, f))));
            }
            stream.memcpy_htod(&mask, &mut cache.node_enabled.slice_mut(..mask.len()))?;
            self.timing.split_metadata_h2d_calls += 1;
            self.timing.split_metadata_h2d_bytes += mask.len() as u64;
        }
        stream.memcpy_htod(&sums, &mut cache.sums.slice_mut(..sums.len()))?;
        stream.memcpy_htod(&metadata, &mut cache.metadata.slice_mut(..metadata.len()))?;
        self.timing.split_metadata_h2d_calls += 2;
        self.timing.split_metadata_h2d_bytes += (metadata.len() * 4 + sums.len() * 8) as u64;
        self.timing.split_evaluated_nodes += nodes as u64;
        self.timing.cache_admitted_nodes += cached.iter().filter(|x| **x).count() as u64;
        if keep_next {
            self.timing.cache_spilled_nodes += cached.iter().filter(|x| !**x).count() as u64;
        }
        if self.ctx.timing {
            stream.synchronize()?;
            self.timing.split_metadata_h2d_ms += metadata_start.elapsed().as_secs_f32() * 1e3;
        }
        let features = self.n_feats as u32;
        let stride = self.n_hist as u32;
        let cache_stride = cache.stride as u32;
        let ig = 1.0 / q.grad_scale();
        let ih = 1.0 / q.hess_scale();
        if self.ctx.timing {
            start_event.record(&stream)?;
        }
        // Without nested sampling a null mask makes the tree-level `enabled`
        // words decide.
        let no_node_mask = 0u64;
        let n_cat = cache.n_cat;
        let max_onehot = p.max_cat_to_onehot;
        let max_threshold = p.max_cat_threshold;
        let wide_fns = self.ctx.wide.as_ref();
        let split_fn = match (cache.wide, cache.cat_enabled) {
            (true, true) => wide_fns.and_then(|w| w.split_cat.as_ref()).map(|w| &w.0),
            (true, false) => wide_fns.and_then(|w| w.split.as_ref()),
            (false, true) => self.ctx.split_features_cat_fn.as_ref(),
            (false, false) => self.ctx.split_features_fn.as_ref(),
        };
        let mut launch = stream.launch_builder(split_fn.context("GPU split kernel unavailable")?);
        launch
            .arg(&mut *hist_out)
            .arg(&cache.previous)
            .arg(&mut cache.next)
            .arg(offsets)
            .arg(&cache.feats)
            .arg(&cache.enabled)
            .arg(&cache.sums)
            .arg(&cache.metadata)
            .arg(&features)
            .arg(&stride)
            .arg(&cache_stride)
            .arg(&ig)
            .arg(&ih)
            .arg(&p.lambda)
            .arg(&p.gamma)
            .arg(&p.min_child_weight)
            .arg(&p.alpha)
            .arg(&p.max_delta_step)
            .arg(&cache.bounds)
            .arg(&cache.monotone)
            .arg(&has_monotone)
            .arg(&mut cache.candidates);
        if node_enabled.is_some() {
            launch.arg(&cache.node_enabled);
        } else {
            launch.arg(&no_node_mask);
        }
        let wide_set_words = cache.wide_set_words;
        let wide_key_elems = cache.wide_key_elems;
        if cache.cat_enabled && cache.wide {
            launch
                .arg(&cache.cat_info)
                .arg(&cache.wide_layout)
                .arg(&max_onehot)
                .arg(&max_threshold)
                .arg(&wide_set_words)
                .arg(&wide_key_elems)
                .arg(&mut cache.wide_keys)
                .arg(&mut cache.wide_sets);
        } else if cache.cat_enabled {
            launch
                .arg(&cache.cat_info)
                .arg(&n_cat)
                .arg(&max_onehot)
                .arg(&max_threshold)
                .arg(&mut cache.cat_sets);
        }
        // Bounds above cover all CTA loads/stores; previous/next/work allocations
        // are disjoint and feature CTAs own disjoint histogram ranges.
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (features, nodes as u32, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            })
        }?;
        let _submit_phase = crate::diagnostic::Phase::new(b"fb.host.packet_submit\0");
        let packet_stride = cache.packet_stride;
        let stride_arg = packet_stride as u32;
        let mut packet_output = cache
            .packets
            .slice_mut(packet_offset * packet_stride..(packet_offset + nodes) * packet_stride);
        let nodes_fn = match (cache.wide, cache.cat_enabled) {
            (true, true) => wide_fns.and_then(|w| w.split_cat.as_ref()).map(|w| &w.1),
            (_, true) => self.ctx.split_nodes_cat_fn.as_ref(),
            (_, false) => self.ctx.split_nodes_fn.as_ref(),
        };
        let mut launch =
            stream.launch_builder(nodes_fn.context("GPU split reduction unavailable")?);
        launch.arg(&cache.candidates).arg(&features);
        if cache.cat_enabled && cache.wide {
            launch
                .arg(&cache.wide_sets)
                .arg(&cache.wide_layout)
                .arg(&wide_set_words)
                .arg(&mut packet_output)
                .arg(&stride_arg);
        } else if cache.cat_enabled {
            launch
                .arg(&cache.cat_sets)
                .arg(&n_cat)
                .arg(&mut packet_output)
                .arg(&stride_arg);
        } else {
            launch
                .arg(&ig)
                .arg(&ih)
                .arg(&p.lambda)
                .arg(&mut packet_output);
        }
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (nodes as u32, 1, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            })
        }?;
        self.timing.kernel_calls += 2;
        self.timing.split_kernel_calls += 2;
        self.timing.split_cat_evaluated += (nodes * n_cat as usize) as u64;
        if self.ctx.timing {
            end_event.record(&stream)?;
            stream.synchronize()?;
            let ms = start_event.elapsed_ms(end_event)?;
            self.timing.kernel_ms += ms;
            self.timing.split_kernel_ms += ms;
        }
        Ok(PendingSplitBatch {
            indices,
            working_slots,
            cached,
        })
    }

    fn collect_split_packets(
        &mut self,
        pending: &[PendingSplitBatch],
        nodes: usize,
        keep_next: bool,
    ) -> Result<Vec<GpuSplitResult>> {
        let slot = self.slot.as_mut().expect("histogram slot lease missing");
        let stream = slot.stream.clone();
        let GpuHistBuffers {
            split_cache,
            hist_out,
            ..
        } = &mut **slot;
        let cache = split_cache
            .as_mut()
            .context("resident split cache unavailable")?;
        anyhow::ensure!(
            pending.len() <= SPLIT_PACKET_WINDOW_BATCHES
                && pending.iter().map(|b| b.indices.len()).sum::<usize>() == nodes,
            "GPU split packet collection shape mismatch"
        );
        anyhow::ensure!(
            !keep_next
                || pending.len() == 1
                || pending
                    .iter()
                    .all(|b| b.cached.iter().all(|&cached| cached)),
            "uncached histograms cannot outlive their producer batch"
        );
        let start = std::time::Instant::now();
        let packet_ready_phase = crate::diagnostic::Phase::new(b"fb.host.packet_ready\0");
        let stride = cache.packet_stride;
        let bytes = nodes * stride;
        if cache.packets_in_flight.take() != Some(bytes) {
            // Mid-level window (drained before the next batch): copy now.
            let mut dst = super::PinnedPrefix {
                inner: &mut cache.packets_host,
                len: bytes,
            };
            stream.memcpy_dtoh(&cache.packets.slice(..bytes), &mut dst)?;
            self.timing.d2h_calls += 1;
            self.timing.d2h_bytes += bytes as u64;
            self.timing.split_d2h_calls += 1;
            self.timing.split_d2h_bytes += bytes as u64;
        }
        stream.synchronize()?;
        // `as_slice` also waits for the copy's own event.
        let packets: Vec<u8> = cache.packets_host.as_slice()?[..bytes].to_vec();
        drop(packet_ready_phase);
        if self.ctx.timing {
            let ms = start.elapsed().as_secs_f32() * 1e3;
            self.timing.d2h_ms += ms;
            self.timing.split_d2h_ms += ms;
        }
        let _decode_phase = crate::diagnostic::Phase::new(b"fb.host.packet_decode\0");
        let mut results = Vec::with_capacity(nodes);
        let mut n = 0;
        for batch in pending {
            for local in 0..batch.indices.len() {
                let packet = &packets[n * stride..(n + 1) * stride];
                let u32_at = |i| u32::from_ne_bytes(packet[i..i + 4].try_into().unwrap());
                let i64_at = |i| i64::from_ne_bytes(packet[i..i + 8].try_into().unwrap());
                let flags = u32_at(12);
                let candidate = (flags & 1 != 0).then(|| SplitCandidate {
                    feat: u32_at(0),
                    bin: u32_at(4),
                    gain: f32::from_bits(u32_at(8)),
                    left: GradPairFixed {
                        grad: i64_at(16),
                        hess: i64_at(24),
                    },
                    right: GradPairFixed {
                        grad: i64_at(32),
                        hess: i64_at(40),
                    },
                    missing_left: flags & 2 != 0,
                    cats: (flags & 4 != 0).then(|| {
                        if cache.wide {
                            // The winner's own set words (the feature's bins
                            // in 64-bit words), sized like `CatBits::for_bins`.
                            let feat = u32_at(0);
                            let words = cache
                                .wide_cat_words
                                .iter()
                                .find(|(f, _)| *f == feat)
                                .map_or(0, |&(_, w)| w);
                            let words: Vec<u32> = (0..words)
                                .map(|w| u32_at(SPLIT_PACKET_BYTES + w * 4))
                                .collect();
                            crate::categorical::CatBits::from_words(&words)
                        } else {
                            let words: [u32; CAT_SET_WORDS] =
                                std::array::from_fn(|w| u32_at(SPLIT_PACKET_BYTES + w * 4));
                            crate::categorical::CatBits::Small(std::array::from_fn(|i| {
                                u64::from(words[2 * i]) | u64::from(words[2 * i + 1]) << 32
                            }))
                        }
                    }),
                });
                let hist = if keep_next && !batch.cached[local] {
                    let _fallback_phase =
                        crate::diagnostic::Phase::new(b"fb.host.packet_fallback\0");
                    let start = std::time::Instant::now();
                    let mut out = vec![GradPairFixed::default(); self.n_hist];
                    let lo = batch.working_slots[local] * self.n_hist * 2;
                    stream.memcpy_dtoh(
                        &hist_out.slice(lo..lo + self.n_hist * 2),
                        GradPairFixed::flatten_mut(&mut out),
                    )?;
                    stream.synchronize()?;
                    self.timing.d2h_calls += 1;
                    self.timing.d2h_bytes += (self.n_hist * 16) as u64;
                    if self.ctx.timing {
                        self.timing.d2h_ms += start.elapsed().as_secs_f32() * 1e3;
                    }
                    Some(out)
                } else {
                    None
                };
                results.push((batch.indices[local], candidate, hist));
                n += 1;
            }
        }
        Ok(results)
    }
}
