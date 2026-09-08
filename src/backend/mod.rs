//! 计算后端。
//!
//! 阶段 1 只有 cpu。阶段 3 加 cuda —— 那时 hist::build 和
//! split::best_split_in_block 各有两个实现,满足同一签名契约,
//! 用同样的输入对拍验证。
//!
//! 边界原则:C++ 侧只暴露 C 接口(裸指针 + 长度),无 struct、
//! 无模板、无析构。所有内存的分配和生命周期归 Rust 管。
//! 这样 unsafe 全部集中在一层薄封装里,而且将来 hipify 只动
//! .cu 内部,签名不变。

pub mod cpu;

#[cfg(feature = "cuda")]
pub mod cuda;

// Keep the CPU-only build free of the CUDA dependency while allowing the
// shared training state machine to compile with an optional GPU backend.
#[cfg(not(feature = "cuda"))]
pub mod cuda {
    use anyhow::Result;
    use crate::types::{Bin, GradPair, GradPairFixed, GradQuantizer, RowId};


    #[derive(Clone, Copy, Debug, Default)]
    pub struct SharedHistogramConfig {
        pub runtime_log: bool,
        pub hist_streams: Option<usize>,
        pub resident_blocks: Option<usize>,
        pub gpu_memory_budget: Option<usize>,
        pub device_quantize: Option<bool>,
        pub hist_nodes_per_batch: Option<usize>,
        pub gpu_math: crate::types::GpuMath,
    }
    #[derive(Clone, Copy, Debug)]
    pub struct SharedHistogramProfile {
        pub kernel_ms: f32,
        pub h2d_ms: f32,
        pub d2h_ms: f32,
    }
    #[derive(Clone, Copy, Debug)]
    pub struct PartitionProfile { pub kernel_ms: f32 }

    /// CPU-only 构建下的占位:`train_with_backend` 是同一份状态机,
    /// 只在 `gpu` 为 `Some` 时才会碰到它,而没有 CUDA 时它构造不出来。
    #[derive(Clone, Copy, Debug, Default)]
    pub struct GpuSegmentTiming {
        pub kernel_ms: f32,
        pub block_h2d_ms: f32,
        pub rows_h2d_ms: f32,
        pub d2h_ms: f32,
        pub block_h2d_calls: u64,
        pub rows_h2d_calls: u64,
        pub d2h_calls: u64,
        pub block_h2d_bytes: u64,
        pub rows_h2d_bytes: u64,
        pub d2h_bytes: u64,
        pub kernel_calls: u64,
    }
    impl GpuSegmentTiming {
        pub fn h2d_ms(&self) -> f32 { 0.0 }
        pub fn add(&mut self, _other: &Self) {}
    }

    #[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
    pub struct GpuRowSpan {
        pub offset: usize,
        pub len: usize,
    }

    #[derive(Clone, Copy, Debug, Default)]
    pub struct GpuMemoryBudget;
    impl GpuMemoryBudget {
        pub fn total(&self) -> usize { 0 }
        pub fn per_row_bytes(&self) -> usize { 0 }
        pub fn report(&self) -> String { String::new() }
    }

    #[derive(Clone, Copy, Debug, Default)]
    pub struct GpuHostMemoryBudget;
    impl GpuHostMemoryBudget {
        pub fn total(&self) -> usize { 0 }
        pub fn report(&self) -> String { String::new() }
    }

    pub struct GpuTrainCtx(std::convert::Infallible);
    impl GpuTrainCtx {
        pub fn fast_math_enabled(&self) -> bool { match self.0 {} }
        pub fn prediction_authority(&self) -> crate::types::PredictionAuthority { match self.0 {} }
        pub fn init_fast_math(&self, _labels: &[f32], _margin: f32) -> Result<()> { match self.0 {} }
        pub fn fast_gradient_and_quantize(
            &self, _objective: crate::types::Objective,
            _t: u64, _s: u64, _tree: u64,
        ) -> Result<(crate::types::GradQuantizer, GradPairFixed, f32)> { match self.0 {} }
        pub fn sync_prediction_to_host(&self, _out: &mut [f32]) -> Result<()> { match self.0 {} }
        #[allow(clippy::too_many_arguments)]
        pub fn new(
            _device: usize, _n_rows: usize, _max_block_feats: usize, _max_hist_bins: usize,
            _n_blocks: usize,
            _partition_threads: u32, _config: SharedHistogramConfig, _timing: bool,
        ) -> Result<Self> {
            anyhow::bail!("CUDA backend 未启用")
        }
        pub fn budget(&self) -> GpuMemoryBudget { match self.0 {} }
        pub fn host_budget(&self) -> GpuHostMemoryBudget { match self.0 {} }
        pub fn timing_enabled(&self) -> bool { match self.0 {} }
        pub fn hist_stream_count(&self) -> usize { match self.0 {} }
        pub fn resident_block_count(&self) -> usize { match self.0 {} }
        pub fn device_mem_used(&self) -> Result<(usize, usize)> { match self.0 {} }
        pub fn upload_gpair(&self, _gpair: &[GradPairFixed]) -> Result<f32> { match self.0 {} }
        pub fn device_quantize_enabled(&self) -> bool { match self.0 {} }
        pub fn upload_and_quantize_gpair(
            &self,
            _gpair: &[GradPair],
        ) -> Result<(GradQuantizer, GradPairFixed, f32)> {
            match self.0 {}
        }
        pub fn begin_tree_rows(&self) -> Result<GpuRowSpan> { match self.0 {} }
        pub fn finish_partition_level(&self) -> Result<()> { match self.0 {} }
        pub fn resolve_partitions(&self) -> Result<Vec<(GpuRowSpan, GpuRowSpan)>> {
            match self.0 {}
        }
        pub fn download_rows(&self, _span: GpuRowSpan) -> Result<Vec<RowId>> { match self.0 {} }
        pub fn assign_leaf_value(&self, _span: GpuRowSpan, _value: f32) -> Result<()> {
            match self.0 {}
        }
        pub fn download_prediction_delta(&self) -> Result<f32> { match self.0 {} }
        pub fn with_prediction_delta(&self, _f: &mut dyn FnMut(&[f32])) -> Result<()> {
            match self.0 {}
        }
        pub fn block_session(
            &self, _block: usize, _data: &[Bin], _rows: usize, _feats: usize, _offsets: &[u32],
        ) -> Result<GpuBlockSession<'_>> { match self.0 {} }
        pub fn partition_session(&self, _col: &[Bin]) -> Result<GpuPartitionSession<'_>> {
            match self.0 {}
        }
        pub fn partition_session_resident(
            &self,
            _block: usize,
            _local_feat: usize,
        ) -> Result<Option<GpuPartitionSession<'_>>> {
            match self.0 {}
        }
    }

    pub struct GpuBlockSession<'a>(std::convert::Infallible, std::marker::PhantomData<&'a ()>);
    impl GpuBlockSession<'_> {
        pub fn histogram(
            &mut self, _rows: Option<&[RowId]>, _out: &mut Vec<GradPairFixed>,
        ) -> Result<()> { match self.0 {} }
        pub fn histogram_device(
            &mut self, _span: GpuRowSpan, _out: &mut Vec<GradPairFixed>,
        ) -> Result<()> { match self.0 {} }
        pub fn histogram_device_at_depth(
            &mut self, _span: GpuRowSpan, _out: &mut Vec<GradPairFixed>,
            _depth: u32, _node_index: usize,
        ) -> Result<()> { match self.0 {} }
        pub fn histogram_device_batch(
            &mut self, _spans: &[GpuRowSpan], _outs: &mut [&mut Vec<GradPairFixed>],
            _feat_mask: u32,
        ) -> Result<()> { match self.0 {} }
        pub fn hist_nodes_batch_capacity(&self) -> usize { match self.0 {} }
        pub fn requires_batched_hist(&self) -> bool { match self.0 {} }
        pub fn timing(&self) -> GpuSegmentTiming { match self.0 {} }
        pub fn feature_tiles(&self) -> usize { match self.0 {} }
    }

    pub struct GpuPartitionSession<'a>(std::convert::Infallible, std::marker::PhantomData<&'a ()>);
    impl GpuPartitionSession<'_> {
        pub fn partition(
            &mut self, _span: GpuRowSpan, _split_bin: u32, _missing_left: bool,
        ) -> Result<usize> { match self.0 {} }
        pub fn timing(&self) -> GpuSegmentTiming { match self.0 {} }
    }

    #[allow(clippy::too_many_arguments)]
    pub fn build_histogram_shared_with_config(
        _device: usize, _data: &[Bin], _rows: usize, _feats: usize,
        _offsets: &[u32], _gpair: &[GradPairFixed], _row_idx: Option<&[RowId]>,
        _config: SharedHistogramConfig,
    ) -> Result<(Vec<GradPairFixed>, SharedHistogramProfile)> {
        anyhow::bail!("CUDA backend 未启用")
    }
    #[allow(clippy::too_many_arguments)]
    pub fn partition_rows_gpu_with_profile(
        _device: usize, _data: &[Bin], _rows: usize, _feature: usize,
        _row_idx: &[RowId], _split_bin: u32, _missing_left: bool, _threads: u32,
    ) -> Result<((Vec<RowId>, Vec<RowId>), PartitionProfile)> {
        anyhow::bail!("CUDA backend 未启用")
    }
}
