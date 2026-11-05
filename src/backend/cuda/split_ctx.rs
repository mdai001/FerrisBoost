//! Bounded, per-resident-block split state. No device allocations occur at a level.
use super::*;
use crate::gpu_mem_plan::SPLIT_PACKET_WINDOW_BATCHES;
use crate::split::SplitCandidate;
use crate::types::{FeatId, TrainParams};

#[cfg(test)]
#[path = "packet_tests.rs"]
mod packet_tests;

pub(super) struct ResidentSplitBuffers {
    previous: CudaSlice<i64>,
    next: CudaSlice<i64>,
    candidates: CudaSlice<u8>,
    packets: CudaSlice<u8>,
    sums: CudaSlice<i64>,
    metadata: CudaSlice<u32>,
    feats: CudaSlice<u32>,
    enabled: CudaSlice<u32>,
    /// 单调约束:每节点 `[lower, upper]`(和 `sums` 同序),每个块内特征的方向。
    /// 只在有约束时上传和读取。
    bounds: CudaSlice<f32>,
    monotone: CudaSlice<i32>,
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
    ) -> Result<Self> {
        Ok(Self {
            previous: stream.alloc_zeros::<i64>((bins * 2 * cap).max(1))?,
            next: stream.alloc_zeros::<i64>((bins * 2 * cap).max(1))?,
            candidates: stream.alloc_zeros::<u8>((feats * cap * 64).max(1))?,
            packets: stream.alloc_zeros::<u8>(cap * 64 * SPLIT_PACKET_WINDOW_BATCHES)?,
            sums: stream.alloc_zeros::<i64>(cap * 2)?,
            metadata: stream.alloc_zeros::<u32>(cap * 4)?,
            feats: stream.alloc_zeros::<u32>(feats.max(1))?,
            enabled: stream.alloc_zeros::<u32>(feats.max(1))?,
            bounds: stream.alloc_zeros::<f32>(cap * 2)?,
            monotone: stream.alloc_zeros::<i32>(feats.max(1))?,
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
        let slot = self.slot.as_mut().expect("histogram slot lease missing");
        let super::GpuHistBuffers {
            split_cache,
            offsets_host,
            ..
        } = &mut **slot;
        let Some(cache) = split_cache.as_mut() else {
            return Ok(false);
        };
        if depth == 0 {
            cache.previous_ids.clear();
            cache.layout.clone_from(offsets_host);
            cache.features_ready = false;
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
    pub(crate) fn split_cached_batch(
        &mut self,
        groups: &[GpuSplitGroup],
        feats: &[FeatId],
        sampled: Option<&[FeatId]>,
        q: &GradQuantizer,
        p: &TrainParams,
        keep_next: bool,
    ) -> Result<Vec<GpuSplitResult>> {
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
            if spills && !pending.is_empty() {
                results.extend(self.collect_split_packets(&pending, packet_nodes, keep_next)?);
                pending.clear();
                packet_nodes = 0;
            }
            pending.push(self.submit_split_cached_batch(
                chunk,
                feats,
                sampled,
                q,
                p,
                keep_next,
                packet_nodes,
            )?);
            packet_nodes += nodes;
            if spills || pending.len() == SPLIT_PACKET_WINDOW_BATCHES {
                results.extend(self.collect_split_packets(&pending, packet_nodes, keep_next)?);
                pending.clear();
                packet_nodes = 0;
            }
        }
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
        anyhow::ensure!(
            self.feat_offsets().windows(2).all(|x| x[1] - x[0] <= 255),
            "GPU split requires at most 255 bins per feature"
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
            if p.has_monotone() {
                let monotone: Vec<i32> = feats.iter().map(|&f| p.monotone_of(f) as i32).collect();
                stream.memcpy_htod(&monotone, &mut cache.monotone.slice_mut(..monotone.len()))?;
                self.timing.split_metadata_h2d_calls += 1;
                self.timing.split_metadata_h2d_bytes += (monotone.len() * 4) as u64;
            }
            cache.features_ready = true;
        }
        let has_monotone = u32::from(p.has_monotone());
        if has_monotone != 0 {
            stream.memcpy_htod(&bounds, &mut cache.bounds.slice_mut(..bounds.len()))?;
            self.timing.split_metadata_h2d_calls += 1;
            self.timing.split_metadata_h2d_bytes += (bounds.len() * 4) as u64;
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
        let mut launch = stream.launch_builder(
            self.ctx
                .split_features_fn
                .as_ref()
                .context("GPU split kernel unavailable")?,
        );
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
            .arg(&cache.bounds)
            .arg(&cache.monotone)
            .arg(&has_monotone)
            .arg(&mut cache.candidates);
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
        let mut packet_output = cache
            .packets
            .slice_mut(packet_offset * 64..(packet_offset + nodes) * 64);
        let mut launch = stream.launch_builder(
            self.ctx
                .split_nodes_fn
                .as_ref()
                .context("GPU split reduction unavailable")?,
        );
        launch
            .arg(&cache.candidates)
            .arg(&features)
            .arg(&ig)
            .arg(&ih)
            .arg(&p.lambda)
            .arg(&mut packet_output);
        unsafe {
            launch.launch(LaunchConfig {
                grid_dim: (nodes as u32, 1, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            })
        }?;
        self.timing.kernel_calls += 2;
        self.timing.split_kernel_calls += 2;
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
            .as_ref()
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
        let mut packets = vec![0u8; nodes * 64];
        stream.memcpy_dtoh(
            &cache.packets.slice(..packets.len()),
            packets.as_mut_slice(),
        )?;
        stream.synchronize()?;
        drop(packet_ready_phase);
        self.timing.d2h_calls += 1;
        self.timing.d2h_bytes += packets.len() as u64;
        self.timing.split_d2h_calls += 1;
        self.timing.split_d2h_bytes += packets.len() as u64;
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
                let packet = &packets[n * 64..(n + 1) * 64];
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
