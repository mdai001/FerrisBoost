//! 训练级 GPU 资源上下文。
//!
//! 改造前每次 histogram / partition 调用都自己 `CudaContext::new` + 解析
//! PTX + `cudaMalloc` 一整套 device buffer,结束再 `cudaFree`。这两个都是
//! **同步**调用,会把整个 stream 挡住;实测 wrapper 是 kernel 的几百倍,
//! 也就是说当时量到的是分配器,不是架构。
//!
//! 这里把所有**长生命周期**的 device 分配收进一个训练级上下文:gpair 每轮
//! 传一次、row index 每个节点更新一次、histogram / partition 的 buffer 按
//! 最坏情况在训练开始时分配一次。
//!
//! **量化列块的生命周期不变**:仍然按 `BlockSource::with_block` 逐块 H2D、
//! 用完即弃。复用的是**装块的那个 buffer**(分配一次、内容反复覆写),
//! 不是让块常驻显存 —— 后者会直接推翻「小显卡跑大数据」这个前提。
//!
//! 并发:`build_tree` 按列块 `par_iter`;两套 histogram stream/buffer 允许
//! 下一块 H2D 和当前块 kernel 重叠。gpair、row arena 和 partition buffer
//! 仍只有一份，因为它们在整个 histogram phase 都是只读的，不需要复制。

use anyhow::{bail, ensure, Context, Result};
use cudarc::driver::{
    sys, CudaEvent, CudaFunction, CudaSlice, CudaStream, HostSlice, LaunchConfig, PinnedHostSlice,
    PushKernelArg, SyncOnDrop,
};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};

use rayon::prelude::*;

use crate::types::{Bin, GradPair, GradPairFixed, GradQuantizer, RowId};

use super::{device_state, DeviceState, SharedHistogramConfig};

/// `gpair_max_abs` / `gpair_quantize` 的 launch 几何。两个 kernel 都是
/// grid-stride,所以 block 数只要够填满卡,不随行数线性增长。
const GPAIR_THREADS: usize = 256;
const GPAIR_BLOCKS: usize = 1024;

/// 诊断/实验开关的统一判定。
///
/// ⚠️ **不要用 `var_os(..).is_some()`。** shell 里 `FOO= cmd` 会把变量设成
/// **空字符串**,而 `is_some()` 对空字符串返回 true —— 于是 A/B 里「关掉」
/// 的那一档其实也走了打开的路径,变成自己和自己比。这个坑真踩过:
/// 两档都跑了 repack,量出约 1% 的「无效应」,差点据此否掉一个 7.5% 的改动。
/// **症状和「真的没效果」完全一样**,所以只能在这里堵死。
/// kernel 要不要加载,和 buffer 要不要分配必须用同一个判断,否则会出现
/// 「分配了 buffer 但没加载 kernel」这种半开状态。
fn device_quantize_wanted(config: &SharedHistogramConfig) -> bool {
    match config.device_quantize {
        Some(v) => v,
        None => !matches!(
            std::env::var("FB_GPU_DEVICE_QUANT").as_deref(),
            Ok("0") | Ok("false")
        ),
    }
}

fn env_flag(name: &str) -> bool {
    matches!(std::env::var(name).as_deref(), Ok("1") | Ok("true"))
}

/// 预分配的显存构成。训练开始就能算出来,是显存预算的一部分。
///
/// `per_row_bytes()` 是**列分块切不动**的那一部分:它只随行数走,
/// 决定单卡能处理的最大行数(也就是多卡该扩展列维度而不是行维度)。
#[derive(Clone, Copy, Debug, Default)]
pub struct GpuMemoryBudget {
    /// 并行复用的 histogram stream/buffer 套数。
    pub hist_streams: usize,
    /// 定点梯度,O(行数),每轮覆写一次。
    pub gpair_bytes: usize,
    /// 当前节点的行索引,O(行数)。
    pub row_idx_bytes: usize,
    /// partition 的左右输出,O(行数)。
    pub partition_out_bytes: usize,
    /// partition 的方向 flag / 块计数 / 偏移,O(行数) + O(块数)。
    pub partition_scratch_bytes: usize,
    /// partition 取的那一整列,O(行数)。
    pub partition_col_bytes: usize,
    /// 装一个量化列块的 buffer(不是让块常驻)。
    pub block_bins_bytes: usize,
    /// 直方图输出。multi-node batching 打开时是 `hist_nodes_per_batch` 份切片。
    pub hist_bytes: usize,
    /// multi-node batching 的 block -> node 映射数组。关闭时为 0。
    pub hist_nodes_batch_idx_bytes: usize,
    /// `gpu_math = "fast"` 的常驻 prediction + label,各 O(行数)。exact 下为 0。
    pub fast_math_bytes: usize,
    /// 行主序原型的转置暂存区(一块,不随 resident 块数增长)。关闭时为 0。
    pub rm_stage_bytes: usize,
    pub offsets_bytes: usize,
    /// 每个 histogram slot 的 colsample 选择表,O(max block features)。
    pub hist_feature_select_bytes: usize,
    /// 每棵树的叶子预测增量；写完一次 D2H，供下一轮算梯度。
    pub prediction_bytes: usize,
    /// device-side quantization 的 f32 gpair 与两个归约 buffer。关闭时为 0。
    ///
    /// ⚠️ **每一项预分配都必须出现在预算报告里。** 上一次做 device
    /// quantization 时正是漏了这一块(报告仍打 1142.1 MB),导致它的显存代价
    /// 在评估时被整个忽略掉。
    pub gpair_f32_bytes: usize,
}

/// 训练期间锁在物理内存里的 host staging buffer。它不能被 swap，必须和
/// device 预算分开报告。
#[derive(Clone, Copy, Debug, Default)]
pub struct GpuHostMemoryBudget {
    /// 列主序臂的整块 pinned staging(每个 slot 一份)。
    /// **行主序不用它**,那条路有自己的分块 pinned 暂存,见下。
    pub block_bins_bytes: usize,
    /// 行主序分块落地的 pinned 暂存(每个 slot 两块轮转)。
    pub rm_stage_bytes: usize,
    /// 一轮定点梯度的 pinned staging buffer。
    pub gpair_bytes: usize,
}

impl GpuHostMemoryBudget {
    pub fn total(&self) -> usize {
        self.block_bins_bytes + self.rm_stage_bytes + self.gpair_bytes
    }

    pub fn report(&self) -> String {
        let mb = |b: usize| b as f64 / (1024.0 * 1024.0);
        format!(
            "pinned host {:.1} MB = 列块 staging {:.1} + 行主序暂存 {:.1} + gpair staging {:.1}",
            mb(self.total()),
            mb(self.block_bins_bytes),
            mb(self.rm_stage_bytes),
            mb(self.gpair_bytes),
        )
    }
}

impl GpuMemoryBudget {
    pub fn total(&self) -> usize {
        self.gpair_bytes
            + self.row_idx_bytes
            + self.partition_out_bytes
            + self.partition_scratch_bytes
            + self.partition_col_bytes
            + self.block_bins_bytes
            + self.hist_bytes
            + self.offsets_bytes
            + self.prediction_bytes
            + self.gpair_f32_bytes
            + self.hist_nodes_batch_idx_bytes
            + self.fast_math_bytes
            + self.rm_stage_bytes
            + self.hist_feature_select_bytes
    }

    /// 只随行数走、列分块压不掉的那部分。
    pub fn per_row_bytes(&self) -> usize {
        self.gpair_bytes
            + self.row_idx_bytes
            + self.partition_out_bytes
            + self.partition_scratch_bytes
            + self.partition_col_bytes
            + self.prediction_bytes
            + self.gpair_f32_bytes
            + self.fast_math_bytes
    }

    pub fn report(&self) -> String {
        let mb = |b: usize| b as f64 / (1024.0 * 1024.0);
        format!(
            "预分配显存 {:.1} MB({} 条 histogram stream) = gpair {:.1} + row_idx {:.1} + partition 输出 {:.1} \
             + partition 临时 {:.1} + partition 取列 {:.1} + 列块 buffer {:.1} \
             + 直方图 {:.1} + offsets {:.3} + colsample 选择 {:.3} + prediction {:.1} + device-quant {:.1} + batch 索引 {:.3} + fast-math {:.1} + rm-stage {:.1};其中 O(行数) 部分 {:.1} MB",
            mb(self.total()),
            self.hist_streams,
            mb(self.gpair_bytes),
            mb(self.row_idx_bytes),
            mb(self.partition_out_bytes),
            mb(self.partition_scratch_bytes),
            mb(self.partition_col_bytes),
            mb(self.block_bins_bytes),
            mb(self.hist_bytes),
            mb(self.offsets_bytes),
            mb(self.hist_feature_select_bytes),
            mb(self.prediction_bytes),
            mb(self.gpair_f32_bytes),
            mb(self.hist_nodes_batch_idx_bytes),
            mb(self.fast_math_bytes),
            mb(self.rm_stage_bytes),
            mb(self.per_row_bytes()),
        )
    }
}

/// 一次 GPU 段的分段耗时。**口径写在字段上,不要混着加。**
///
/// timing 打开时每个传输段前后都显式 `stream.synchronize()`,所以这些墙钟
/// 是真实传输时间,不含「等前面 kernel 排完」的时间;代价是额外的 sync,
/// 因此默认关闭(只有 `FB_PROFILE=1` 打开)。
#[derive(Clone, Copy, Debug, Default)]
pub struct GpuSegmentTiming {
    /// CUDA event 圈的纯 kernel 时间。
    pub kernel_ms: f32,
    /// 大块数据的 H2D:histogram 段是量化列块,partition 段是选中的那一列。
    pub block_h2d_ms: f32,
    /// 行索引的 H2D。
    pub rows_h2d_ms: f32,
    /// D2H 墙钟。
    pub d2h_ms: f32,
    /// Number of H2D / D2H submissions represented by this segment.  These
    /// are deliberately separate from kernel launches: one block upload can
    /// serve every active node at a level.
    pub block_h2d_calls: u64,
    pub rows_h2d_calls: u64,
    pub d2h_calls: u64,
    /// Bytes submitted in the corresponding transfer categories.  The block
    /// number includes the small histogram-offset upload, so a profile can
    /// account for every H2D byte rather than silently dropping metadata.
    pub block_h2d_bytes: u64,
    pub rows_h2d_bytes: u64,
    pub d2h_bytes: u64,
    /// CUDA kernel launches represented by this segment.
    pub kernel_calls: u64,
}

/// 一个已经 launch、但 totals 还没取回的 partition。
///
/// scatter 已经在 device 上按正确位置写完了行;host 只是还不知道左右各有
/// 多少行。那两个数只有在构造下一层的 `GpuRowSpan` 时才需要,所以整层
/// 攒到最后一次性取回 —— 这就是把 126 次阻塞 D2H 压到每层一次的全部原理。
#[derive(Clone, Copy, Debug)]
struct PendingPartition {
    /// 输出在下一层 arena 中的起点。
    out_offset: usize,
    /// 该节点的行数(左右之和)。
    n_selected: usize,
}

/// 一层最多多少个节点。depth 12 就有 4096 个,远超本项目的实际使用范围;
/// 超了会显式报错,不会越界。
const MAX_LEVEL_NODES: usize = 8192;

/// multi-node histogram batching 的硬上界。审计给的可兑现区间是 4/8/16 个
/// 节点;设 32 是为了让「批大小」永远是**有界**的显存项,而不是随树深
/// 无限增长(depth 10 的一层就有 512 个节点)。超出的部分自动分成多批。
const MAX_HIST_NODES_PER_BATCH: usize = 32;

/// 行主序转置的暂存预算。**按行分块转置**,所以它和列块大小无关。
///
/// ⚠️ 第一版是整块大小的暂存区(wide 上 305 MB),而它**只在 warm-up
/// 用一次** —— 常驻一整块只为了转置那一下,是白占。分块之后这笔常驻
/// 从 305 MB 降到 32 MB,而转置本身仍然是那个两维都合并的 tiled kernel。
const RM_STAGE_BYTES: usize = 32 * 1024 * 1024;

/// 行主序流式上传的暂存区,**每个 slot 独占一份**。
///
/// H2D 要真正藏到 kernel 后面,CUDA 要求**四个条件同时成立**:
/// pinned host 内存、非默认 stream、异步 copy、彼此独立的 buffer。
/// 这条路径以前一个都没占全:
///
/// * buffer 是**全局一块**,还配一把持有到 `synchronize()` 的锁 → 所有
///   slot 串成一条(已修);
/// * 源是**可分页**内存 —— `cuMemcpyHtoDAsync` 在可分页内存上是
///   **实质同步**的,driver 必须先经自己的暂存区搬一道。
///   所以"异步 copy"这条当时也没成立,只是看起来成立。
///
/// pinned 打开时用**两组** buffer 轮转:host 填第 i 块的同时,第 i-1 块的
/// H2D + 转置还在 GPU 上跑。等待用的是 `PinnedHostSlice` 自带的 per-buffer
/// event(`as_mut_slice()` 只等这一块自己的前一次使用),**不是**整条流的
/// `synchronize()` —— 后者会把刚建立的重叠再毁掉。
struct RmStaging {
    /// device 落地区,pinned 模式下是 2 块(轮转),否则 1 块。
    dev: Vec<CudaSlice<u8>>,
    /// pinned host 暂存,和 `dev` 一一对应;可分页模式下为空。
    host: Vec<PinnedHostSlice<u8>>,
    /// 每块暂存能装的行数。
    chunk_rows: usize,
}

impl RmStaging {
    fn pinned(&self) -> bool {
        !self.host.is_empty()
    }
}
/// 每个 slot 的暂存下限。slot 很多时按总预算平分会把 chunk 切得过碎,
/// 使 warm-up 的 H2D 次数无谓上升;低于这个值就不再切。
const RM_STAGE_MIN_PER_SLOT: usize = 4 * 1024 * 1024;

/// `partition()` 的返回值:本层内的登记号,`resolve_partitions()` 后才有行数。
pub type PartitionToken = usize;

/// A contiguous node-row span in the current device-side level arena.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GpuRowSpan {
    pub offset: usize,
    pub len: usize,
}

impl GpuSegmentTiming {
    pub fn h2d_ms(&self) -> f32 {
        self.block_h2d_ms + self.rows_h2d_ms
    }

    pub fn add(&mut self, other: &Self) {
        self.kernel_ms += other.kernel_ms;
        self.block_h2d_ms += other.block_h2d_ms;
        self.rows_h2d_ms += other.rows_h2d_ms;
        self.d2h_ms += other.d2h_ms;
        self.block_h2d_calls += other.block_h2d_calls;
        self.rows_h2d_calls += other.rows_h2d_calls;
        self.d2h_calls += other.d2h_calls;
        self.block_h2d_bytes += other.block_h2d_bytes;
        self.rows_h2d_bytes += other.rows_h2d_bytes;
        self.d2h_bytes += other.d2h_bytes;
        self.kernel_calls += other.kernel_calls;
    }
}

struct GpuBuffers {
    /// 摊平的交错 [grad, hess, ...],每轮覆写一次。
    gpair: CudaSlice<i64>,
    /// device-side quantization:上传的 **f32** 交错 [grad, hess, ...]。
    /// 走 PCIe 的从 16 B/行 变成 8 B/行。
    gpair_f32: Option<CudaSlice<f32>>,
    /// max / sum 归约的每 block 局部结果,以及取回它们的 host buffer。
    gpair_max: Option<CudaSlice<f64>>,
    gpair_sum: Option<CudaSlice<i64>>,
    gpair_max_host: Vec<f64>,
    gpair_sum_host: Vec<i64>,
    /// selected-row packing:`原始行号 → 压实下标`,每棵树上传一次。
    /// 未启用时为 None。
    /// `gpu_math = "fast"`:raw-margin prediction 常驻 device,不再每轮 D2H。
    pred_resident: Option<CudaSlice<f32>>,
    /// `gpu_math = "fast"`:label 常驻 device,整个训练只上传一次。
    label_resident: Option<CudaSlice<f32>>,
    /// 当前节点的行索引。
    rows: CudaSlice<u32>,
    part_col: CudaSlice<u8>,
    part_flags: CudaSlice<u8>,
    part_block_counts: CudaSlice<u32>,
    part_left_offsets: CudaSlice<u32>,
    part_right_offsets: CudaSlice<u32>,
    part_right_out: CudaSlice<u32>,
    /// 下一层 row arena 在 `part_right_out` 中顺序分配到的位置。
    next_rows_cursor: usize,
    /// 每个节点左右行数的 device 侧收集区,整层结束一次取回。
    part_totals: CudaSlice<u32>,
    /// 本层每个已 launch 节点的解析信息,整层结束后一次性取回 totals。
    part_pending: Vec<PendingPartition>,
    part_offsets_host: Vec<u32>,
    prediction_delta: CudaSlice<f32>,
    prediction_host: Vec<f32>,
    /// H2D staging。两块都是训练开始分配、训练结束释放的 page-locked
    /// memory；量化数据本身仍然逐块流式覆写。
    gpair_host: Option<PinnedHostSlice<i64>>,
}

/// 每条 histogram stream 独占的一整套可覆写资源。共享这些 buffer 会让
/// 下一块的 H2D 覆盖仍在被上一块 kernel 读取的数据，形成低概率竞态。
struct GpuHistBuffers {
    stream: Arc<CudaStream>,
    bins: CudaSlice<u8>,
    offsets: CudaSlice<u32>,
    /// Column-major colsample index scratch, one u32 per local feature.
    feature_select: CudaSlice<u32>,
    hist_out: CudaSlice<i64>,
    /// Multi-node batching 的 block -> node 映射,布局是
    /// `[blk_ptr(batch+1) | node_off(batch) | node_len(batch)]`。
    /// batching 关闭时长度为 1(cudarc 不接受零长分配)。
    batch_idx: CudaSlice<u32>,
    hist_host: Vec<i64>,
    bins_host: Option<PinnedHostSlice<u8>>,
    /// 行主序转置的分块暂存区,**每个 slot 一份**(见 `RmStaging`)。
    rm_stage: Option<RmStaging>,
    start_event: CudaEvent,
    end_event: CudaEvent,
    /// 常驻块的**实际特征数**。
    ///
    /// ⚠️ 不能拿 `bins.len() / n_rows` 反推 —— `bins` 是按**最坏情况**
    /// (`max_block_feats`)分配的,反推出来的永远是最宽的那个块。
    /// 最后一块通常更窄(28 特征 / 每块 16 → 16 + 12),行主序下
    /// 行距就是特征数,用错会**整块读偏**,而且只在"最后那块也常驻"时
    /// 才发作 —— 等宽的块碰巧对,所以这个 bug 藏得很深。
    resident_feats: usize,
    /// `Some(block)` marks an immutable, device-resident training block.
    /// Streaming slots always leave this as None.
    resident_block: Option<usize>,
}

struct HistSlotPool {
    available: Mutex<Vec<usize>>,
    ready: Condvar,
}

impl HistSlotPool {
    fn new(n: usize) -> Self {
        Self {
            available: Mutex::new((0..n).rev().collect()),
            ready: Condvar::new(),
        }
    }

    fn acquire(&self) -> Result<HistSlotLease<'_>> {
        let mut available = self
            .available
            .lock()
            .map_err(|_| anyhow::anyhow!("GPU histogram slot pool poisoned"))?;
        loop {
            if let Some(index) = available.pop() {
                return Ok(HistSlotLease { pool: self, index });
            }
            available = self
                .ready
                .wait(available)
                .map_err(|_| anyhow::anyhow!("GPU histogram slot pool poisoned"))?;
        }
    }
}

struct HistSlotLease<'a> {
    pool: &'a HistSlotPool,
    index: usize,
}

impl Drop for HistSlotLease<'_> {
    fn drop(&mut self) {
        if let Ok(mut available) = self.pool.available.lock() {
            available.push(self.index);
            self.pool.ready.notify_one();
        }
    }
}

/// cudarc 的 pinned allocation 没有 prefix view；这个内部适配器保留它的
/// event 同步语义，同时只传当前列块的有效前缀。
struct PinnedPrefix<'a, T> {
    inner: &'a mut PinnedHostSlice<T>,
    len: usize,
}

impl<T> HostSlice<T> for PinnedPrefix<'_, T> {
    fn len(&self) -> usize {
        self.len
    }

    unsafe fn stream_synced_slice<'a>(
        &'a self,
        stream: &'a CudaStream,
    ) -> (&'a [T], SyncOnDrop<'a>) {
        let (slice, sync) = unsafe { self.inner.stream_synced_slice(stream) };
        (&slice[..self.len], sync)
    }

    unsafe fn stream_synced_mut_slice<'a>(
        &'a mut self,
        stream: &'a CudaStream,
    ) -> (&'a mut [T], SyncOnDrop<'a>) {
        let (slice, sync) = unsafe { self.inner.stream_synced_mut_slice(stream) };
        (&mut slice[..self.len], sync)
    }
}

pub struct GpuTrainCtx {
    #[allow(dead_code)]
    state: Arc<DeviceState>,
    stream: Arc<CudaStream>,
    hist_fn: CudaFunction,
    /// **临时 A/B 开关**(`FB_HIST_FUSED=1`):把一个列块的所有 feature tile
    /// 融进一次 launch,让 gpair / row_idx 每个 block 只读一次。定下默认
    /// 之后连同这个字段一起删掉 —— 仓库里不留实验旋钮。
    hist_fused_fn: Option<CudaFunction>,
    /// 临时 A/B:R=4 multi-row/thread 的融合 kernel。定下之后要么成为默认、
    /// 要么连同开关一起删。
    hist_fused_r4_fn: Option<CudaFunction>,
    /// 整层多节点合并 launch 的 kernel。`hist_nodes_per_batch <= 1` 时为 None。
    hist_batched_fn: Option<CudaFunction>,
    /// `gpu_math = "fast"` 的 prediction-apply + gradient kernel。
    apply_grad_fn: Option<CudaFunction>,
    /// GPU-local row-major 原型:转置、行主序 histogram、行主序取列。
    /// 三个要么都有要么都没有。
    rm_transpose_fn: Option<CudaFunction>,
    rm_hist_fn: Option<CudaFunction>,
    /// XGBoost 式 flattened (row × feature) 映射。`Some` 时取代 `rm_hist_fn`。
    /// 行主序下按 active row 直接取 split feature 的 partition 分类 kernel。
    rm_count_fn: Option<CudaFunction>,
    gpu_math: crate::types::GpuMath,
    gpair_max_fn: Option<CudaFunction>,
    gpair_quantize_fn: Option<CudaFunction>,
    count_fn: CudaFunction,
    scan_fn: CudaFunction,
    scatter_layer_fn: CudaFunction,
    assign_leaf_fn: CudaFunction,
    init_tree_fn: CudaFunction,
    start_event: CudaEvent,
    end_event: CudaEvent,
    config: SharedHistogramConfig,
    /// 一个 feature tile 最多放多少个 bin,由 driver 报的 shared-memory
    /// 上限和配置预算共同决定。
    max_bins_per_tile: usize,
    /// 一次 launch 最多合并多少个 `Accumulate` 节点。1 表示关闭。
    /// **它改变内存计划**(每个节点要一份独立的直方图切片),所以和
    /// `hist_streams` 一样必须登记进 `GpuMemoryBudget`。
    hist_nodes_per_batch: usize,
    /// 行主序转置一次处理多少行。0 表示未启用。
    rm_chunk_rows: usize,
    n_rows: usize,
    partition_threads: u32,
    /// 是否记录分段耗时。打开会在 H2D/D2H 前后显式同步,产生额外 sync ——
    /// 所以默认关闭,只在 `FB_PROFILE=1` 时打开。
    timing: bool,
    /// Opt-in diagnostic only. It records a CUDA event pair around **every**
    /// histogram launch, which serializes them — so it must never be on in a
    /// normal training run, and never on the same event pair as `timing`.
    bucket_profile: bool,
    use_pinned: bool,
    budget: GpuMemoryBudget,
    host_budget: GpuHostMemoryBudget,
    buf: Mutex<GpuBuffers>,
    hist_slots: Vec<Mutex<GpuHistBuffers>>,
    hist_slot_pool: HistSlotPool,
    hist_stream_count: usize,
    resident_block_count: usize,
    /// 物理列块总数,用来在路径回显里区分 pure streaming / hybrid / full resident。
    n_blocks: usize,
    /// 规划器实际使用的显存预算(字节),仅用于回显。
    gpu_memory_budget: Option<usize>,
    /// 这次训练实际选中的 GPU ordinal,仅用于回显。
    device_ordinal: usize,
    echoed_resident_col: std::sync::atomic::AtomicBool,
    echoed_upload_col: std::sync::atomic::AtomicBool,
}

impl GpuTrainCtx {
    /// 按**最坏情况**预分配:`max_block_feats` 是最宽的列块,
    /// `max_hist_bins` 是最宽的那块直方图。这样整个训练过程中不再有
    /// 任何 device 分配 / 释放。
    pub fn new(
        device_ordinal: usize,
        n_rows: usize,
        max_block_feats: usize,
        max_hist_bins: usize,
        // 物理列块总数。**常驻块数要夹在这个数以内** —— 比物理块还多的
        // 常驻 slot 会被真的分配出来,然后永远用不上。
        n_blocks: usize,
        partition_threads: u32,
        config: SharedHistogramConfig,
        timing: bool,
    ) -> Result<Self> {
        if n_rows == 0 {
            bail!("GPU 训练上下文需要非空数据");
        }
        u32::try_from(n_rows).context("GPU 训练行数超过 u32")?;
        if partition_threads == 0
            || partition_threads > 1024
            || !partition_threads.is_power_of_two()
        {
            bail!("partition threads 必须是 1..=1024 的 2 次幂");
        }
        if config.threads_per_block == 0 || config.threads_per_block > 1024 {
            bail!("threads_per_block 要在 1..=1024");
        }

        let bucket_profile = env_flag("FB_HIST_BUCKET_PROFILE");
        let state = device_state(device_ordinal)?;
        let stream = state.ctx.default_stream();
        let hist_fn = state
            .module
            .load_function(config.layout.kernel_name())
            .with_context(|| format!("PTX 中找不到 {}", config.layout.kernel_name()))?;
        // 融合 kernel 是默认路径。唯一的例外是 bucket profiler:它按
        // **每次 launch** 记录一行 CSV,而融合之后一个列块只有一次 launch,
        // 那个诊断就失去意义 —— 所以开 `FB_HIST_BUCKET_PROFILE` 时退回
        // 逐 tile launch 的老路径。这不是性能旋钮,是诊断模式。
        let hist_fused_fn = if !bucket_profile {
            Some(
                state
                    .module
                    .load_function("hist_build_shared_planar32_fused")
                    .context("PTX 中找不到 hist_build_shared_planar32_fused")?,
            )
        } else {
            None
        };
        // R=4 multi-row/thread。只在融合路径上有意义(逐 tile 的诊断回退
        // 路径不走它)。
        //
        // 收益只来自摊薄 shared init / flush / barrier / launch —— atomic 次数、
        // bins 和 gpair 的读取量都不变。所以它**不占额外显存**,也就不像
        // `hist_streams` 那样需要变成用户参数:它是纯 kernel 调度选择。
        //
        // 实测(counterbalanced,取最小):wide resident 0.723 → **0.687(−5.0%)**、
        // HIGGS 0.174 → 0.169(−2.9%,不退)。
        let hist_fused_r4_fn = if hist_fused_fn.is_some() {
            Some(
                state
                    .module
                    .load_function("hist_build_shared_planar32_fused_r4")
                    .context("PTX 中找不到 hist_build_shared_planar32_fused_r4")?,
            )
        } else {
            None
        };
        // Multi-node histogram batching。整层的 `Accumulate` 节点合并进一次
        // launch,几何不变(仍是一个 block 管一个节点的 rows_per_block 行、
        // 块内顺序扫 tile),变的只是 block → node 从 blockIdx 算术变成查
        // `blk_ptr`。
        //
        // ⚠️ **它改变内存计划**:批内每个节点要一份独立的直方图切片,
        // 所以容量是显式旋钮而不是编译期常量,并且必须登记进预算。
        // 解析顺序和 hist_streams / device_quantize 一致:
        // **显式参数 > 环境变量 > 默认 16**。env 只是 benchmark override,
        // 永远不得压过用户明确传进来的值。
        let hist_nodes_per_batch = match config.hist_nodes_per_batch {
            Some(n) => n,
            None => match std::env::var("FB_HIST_NODES_PER_BATCH") {
                Ok(v) => v.trim().parse::<usize>().context("FB_HIST_NODES_PER_BATCH 必须是整数")?,
                Err(_) => 16,
            },
        };
        if hist_nodes_per_batch == 0 {
            bail!("hist_nodes_per_batch 至少是 1(1 表示关闭合并),收到 0");
        }
        if hist_nodes_per_batch > MAX_HIST_NODES_PER_BATCH {
            bail!("hist_nodes_per_batch 上限是 {MAX_HIST_NODES_PER_BATCH},收到 {hist_nodes_per_batch}");
        }
        // `gpu_math = "fast"`:prediction / gradient / quantization 全程留在
        // device。只有 Fast 才加载 kernel 和分配那两块常驻 buffer。
        let gpu_math = config.gpu_math;
        let fast_math = matches!(gpu_math, crate::types::GpuMath::Fast);
        let apply_grad_fn = if fast_math {
            Some(
                state
                    .module
                    .load_function("apply_delta_and_gradient")
                    .context("PTX 中找不到 apply_delta_and_gradient")?,
            )
        } else {
            None
        };
        // GPU-local row-major 原型。**只是 GPU 侧的表示**:Parquet / Arrow /
        // host 量化 cache / FFI 一个都不动,列块照旧按列主序上传,上传后在
        // device 上转置一次。
        //
        // 限制:只支持 `n_feats <= 32` 且能整除 16 的向量取数快路径之外的
        // 情况会回落到标量路径;`hist_nodes_per_batch > 1` 才有意义
        // (复用 batched 的 blk_ptr 语义,batching 本身不变)。
        // ⚠️ 行主序 kernel 用「两条 lane 一行、每条一次 uint4」的映射,
        // 一行最多覆盖 **32 个特征**。超过就会**静默丢掉**后面的特征 ——
        // 已经因此让 `gpu_training_reuses_resources_without_changing_the_model`
        // 挂过一次。所以这里是硬门槛,不满足就整个关掉回到列主序,
        // 而不是让它悄悄算错。
        // **默认开启**(2026-09-04)。判据全部满足:wide fast −22.4%、
        // wide exact −15.6%,HIGGS 两种模式在真实轮数下打平,
        // exact 仍与 CPU 逐字节相同,常驻显存只多 32 MB。
        //
        // `FB_ROW_MAJOR` 现在是**三态**:未设 = 默认开;`0`/`false` = 关
        // (列主序参照臂);`1`/`true` = 开。
        // ⚠️ 不能用 `env_flag`:那样「未设」和「显式关掉」会变成同一个值,
        // 参照臂就再也测不出来了。
        let row_major_requested = match std::env::var("FB_ROW_MAJOR") {
            Ok(v) => match v.trim() {
                "0" | "false" => false,
                "1" | "true" => true,
                other => bail!("FB_ROW_MAJOR 只接受 0/1/true/false,收到 {other:?}"),
            },
            Err(_) => true,
        };
        // 一行最多 32 个特征是 kernel 的硬上界,超过就明确回退到列主序。
        // ⚠️ 上界用 `gpu_mem_plan` 里那个**共享常量** —— 显存规划器要靠同一个
        // 值预测运行时会不会开第二条流,两边各写一个 32 就会悄悄漂移。
        let row_major = row_major_requested
            && hist_fused_fn.is_some()
            && max_block_feats as u64 <= crate::gpu_mem_plan::ROW_MAJOR_MAX_FEATS;
        let (rm_transpose_fn, rm_hist_fn, rm_count_fn) = if row_major {
            (
                Some(state.module.load_function("transpose_bins_to_row_major")
                    .context("PTX 中找不到 transpose_bins_to_row_major")?),
                Some(state.module.load_function("hist_build_shared_planar32_batched_rm")
                    .context("PTX 中找不到 hist_build_shared_planar32_batched_rm")?),
                Some(state.module.load_function("partition_count_rm")
                    .context("PTX 中找不到 partition_count_rm")?),
            )
        } else {
            (None, None, None)
        };
        let hist_batched_fn = if hist_fused_fn.is_some() {
            Some(
                state
                    .module
                    .load_function("hist_build_shared_planar32_batched")
                    .context("PTX 中找不到 hist_build_shared_planar32_batched")?,
            )
        } else {
            None
        };
        let (gpair_max_fn, gpair_quantize_fn) = if device_quantize_wanted(&config) {
            (
                Some(
                    state
                        .module
                        .load_function("gpair_max_abs")
                        .context("PTX 中找不到 gpair_max_abs")?,
                ),
                Some(
                    state
                        .module
                        .load_function("gpair_quantize")
                        .context("PTX 中找不到 gpair_quantize")?,
                ),
            )
        } else {
            (None, None)
        };
        let count_fn = state
            .module
            .load_function("partition_count")
            .context("PTX 中找不到 partition_count")?;
        let scan_fn = state
            .module
            .load_function("partition_scan_blocks")
            .context("PTX 中找不到 partition_scan_blocks")?;
        let scatter_layer_fn = state
            .module
            .load_function("partition_scatter_layer")
            .context("PTX 中找不到 partition_scatter_layer")?;
        let assign_leaf_fn = state
            .module
            .load_function("assign_leaf_value")
            .context("PTX 中找不到 assign_leaf_value")?;
        let init_tree_fn = state
            .module
            .load_function("init_tree_rows")
            .context("PTX 中找不到 init_tree_rows")?;

        // shared-memory tile 上限只跟设备和配置有关,和数据无关,所以
        // 在这里算一次就够,不必每次 launch 重新问 driver。
        // ⚠️ **>48 KiB 的 opt-in shared memory 已经测过并否决**,不要重开:
        // 在 sm_86 上 32 KiB → 4 个 feature tile 但每 SM 常驻 3 个 block
        // (满 occupancy);要降到 2 个 tile 需要 ≥56 KiB,而每 SM 只有
        // 100 KiB shared,于是只剩 1 个 block、occupancy 掉到 33%。
        // 实测 HIGGS 打平、wide **回退 14%**。而且 R=4 已经先把
        // shared init/flush 的固定成本摊薄了一轮,tile 数减半能拿的余量本来
        // 就不多。细节见 `internal-docs/history.md`;换到 shared/SM 比例不同的卡再重测。
        let device_shared_bytes = state
            .ctx
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)
            .context("读取每 block shared-memory 上限")?;
        let static_shared_bytes = hist_fn
            .shared_size_bytes()
            .context("读取 kernel 静态 shared-memory 大小")?;
        let device_shared_limit = usize::try_from(device_shared_bytes - static_shared_bytes)
            .context("shared-memory 上限无效")?;
        let shared_limit_bytes = config
            .shared_memory_budget_bytes
            .unwrap_or(device_shared_limit)
            .min(device_shared_limit);
        let max_bins_per_tile = shared_limit_bytes / std::mem::size_of::<GradPairFixed>();
        if max_bins_per_tile == 0 {
            bail!("设备没有足够 shared memory 放一个 GradPairFixed");
        }

        let n_part_blocks = (n_rows as u32).div_ceil(partition_threads) as usize;
        let bins_len = max_block_feats * n_rows;
        let hist_words = max_hist_bins * 2;
        let offsets_len = max_block_feats + 1;
        // [blk_ptr(batch+1) | node_off(batch) | node_len(batch)]
        //
        // ⚠️ 行主序**只有 batched 一个入口**,所以哪怕
        // `hist_nodes_per_batch == 1` 也要按批分配 —— 否则批里那一个节点
        // 要写 4 个 u32 到一个长度为 1 的 buffer 上,cudarc 直接 panic。
        let batch_capable = hist_fused_fn.is_some() || row_major;
        let batch_idx_len = if batch_capable { hist_nodes_per_batch.max(1) * 3 + 1 } else { 1 };

        // 默认双流；同一 binary 可用 1 做严格 A/B。只开放 1/2，避免把
        // 实验开关误当成任意并发度配置，显存也因此始终可精确预算。
        // A deliberately simple cache fast path: retain the first N logical
        // column blocks.  It is opt-in, fixed-size, and every other block
        // continues through the original streaming buffers.
        //
        // 解析顺序:**显式参数(通常来自显存规划器)> env > 0**。
        //
        // ⚠️ 规划器解出来的值是**权威**的:块宽、stream 数、常驻块数是在
        // 同一个内存模型里一起解的,运行时不能自己再挑一个,否则预算和
        // 实际分配对不上 —— 那正是"规划器按单流选宽块、运行时按路径开双流"
        // 那个 OOM 的形状。
        //
        // `FB_CUDA_RESIDENT_BLOCKS` 保留为开发/参照 override,但**不再是
        // 正常的控制路径**(正式 API 是 `TrainParams::resident_blocks`)。
        let requested_resident = match config.resident_blocks {
            Some(n) => n,
            None => std::env::var("FB_CUDA_RESIDENT_BLOCKS")
                .ok()
                .map(|v| v.parse::<usize>())
                .transpose()
                .context("FB_CUDA_RESIDENT_BLOCKS 必须是整数")?
                .unwrap_or(0),
        };
        // 夹到物理块数:多出来的 slot 只会白占显存。
        let resident_block_count = requested_resident.min(n_blocks);

        // stream 数的解析顺序:**用户显式参数 > 环境变量 > planner 默认**。
        //
        // 环境变量只是 benchmark/debug 的 override,**不能压过用户明确传进来的
        // 值** —— 正式 API 是 `TrainParams::hist_streams`,不是 env。
        //
        // planner 默认按模式选,因为这不只是调度参数,它直接参与内存计划:
        // 第二条 stream 要多一整套列块 buffer,c=28 时是 280 MB(约占预分配
        // 的 25%)。resident 命中之后块 H2D 已经很少,没什么可重叠的,实测
        // wide resident 单流反而快 4.2%;streaming 仍然是传输受限的。
        let hist_stream_count = match config.hist_streams {
            Some(n) => n,
            None => match std::env::var("FB_CUDA_HIST_STREAMS") {
                Ok(v) => v
                    .parse::<usize>()
                    .context("FB_CUDA_HIST_STREAMS 必须是整数")?,
                // **按模式选,两档的证据完全不同**(2026-09-05 重测;
                // 旧的「两种模式都不值」是在**全局 staging 锁**下量的,
                // 那把锁把所有 slot 串成一条,双流结构上不可能有收益。
                // 锁去掉之后两边都变了 —— 这就是「负结论只在其前提下成立」):
                //
                // | 配置 | s=1 | s=2 | 第二条 stream |
                // |---|---:|---:|---:|
                // | HIGGS resident | **0.112** | 0.114 | 打平,−280 MB 更划算 |
                // | wide resident | **0.706** | 0.709 | 打平 |
                // | HIGGS streaming | 0.6185 | **0.512** | **−17.2%**,+282 MB |
                // | wide streaming | 3.6965 | **3.2195** | **−12.9%**,+307 MB |
                //
                // resident:块 H2D 只在 warm-up 发生,稳态几乎没有可重叠的
                // 传输,所以是打平 —— 打平不值 280 MB,保持单流。
                // (旧记录说 wide resident 单流快 4.2%;那 4.2% 里有一部分
                // 其实是锁在惩罚双流那一档,现在是干净的打平。)
                //
                // streaming:仍然是传输受限的,双流拿到稳定的两位数收益,
                // 而**显存规划器已经把 stream 数算进 `per_col_bytes`**,
                // 预算不够时它会自己收窄列块或明确失败,所以默认双流
                // 不会让本来能训的数据训不了 —— 能力边界由规划器守,
                // 不是由这个默认值守。
                // **按执行路径分派,不是按 resident/streaming 分派:**
                //
                //   行主序分块 streaming → 2
                //   列主序 streaming     → 1
                //   resident             → 1
                //
                // 机制:**行主序是分块上传的** —— 每块切成多个 chunk 依次
                // 上传 / 转置 / 建直方图,独立的 slot 之间因此有东西可以
                // 流水起来。第二条流让**本来就要做的传输**真正并发,
                // 而不是为了 overlap 再复制一遍数据(后者正是 pinned 被
                // 否掉的原因)。收益随 chunk 数走:
                // c=8 **−28%**、c=28 −17%、c=32 −13%。
                //
                // **列主序一块一次大 memcpy,没有 chunk 可交错,实测打平**
                // (wide c=64:3.237 vs 3.230)—— 那一档 +307 MB 换 0.2%。
                // resident 同样打平(HIGGS 0.112/0.114、wide 0.706/0.709),
                // 稳态几乎没有块 H2D 可重叠。
                //
                // 显存代价:+282 MB(HIGGS)/ +307 MB(wide),全是列块
                // buffer 翻倍。规划器 `resolve_gpu_streaming_plan` 用同一条
                // 判据预测这个决定,并在第二条流会把块宽夹到拐点以下时
                // 降回单流(拐点 −42.6% 比第二条流值钱)。
                Err(_) if resident_block_count == 0 && row_major => 2,
                Err(_) => 1,
            },
        };
        if !matches!(hist_stream_count, 1 | 2) {
            bail!("hist_streams 只支持 1 或 2,收到 {hist_stream_count}");
        }
        // 必须在任何共享 CudaSlice 分配之前进入 multi-stream mode。
        // cudarc 只会在分配当下为 slice 创建 read/write events；顺序反了，
        // gpair/rows 就没有跨 stream 依赖，测试可能偶然通过但真实训练会竞态。
        let mut hist_streams = Vec::with_capacity(hist_stream_count);
        for slot in 0..hist_stream_count {
            hist_streams.push(
                state
                    .ctx
                    .new_stream()
                    .with_context(|| format!("创建 histogram stream {slot}"))?,
            );
        }

        let gpair = stream
            .alloc_zeros::<i64>(n_rows * 2)
            .context("gpair buffer 分配")?;
        let rows = stream
            .alloc_zeros::<u32>(n_rows)
            .context("row_idx buffer 分配")?;
        let part_col = stream
            .alloc_zeros::<u8>(n_rows)
            .context("partition 列 buffer 分配")?;
        let part_flags = stream
            .alloc_zeros::<u8>(n_rows)
            .context("partition flags 分配")?;
        // counts/offsets 仍然按**一个节点**定容量:同一条 stream 上
        // count → scan → scatter 是顺序的,节点 i 的 scatter 一定在
        // 节点 i+1 的 count 之前跑完,所以这两块照旧可以按节点复用。
        let part_block_counts = stream
            .alloc_zeros::<u32>(n_part_blocks)
            .context("partition block counts 分配")?;
        let part_left_offsets = stream
            .alloc_zeros::<u32>(n_part_blocks + 1)
            .context("partition left offsets 分配")?;
        let part_right_offsets = stream
            .alloc_zeros::<u32>(n_part_blocks + 1)
            .context("partition right offsets 分配")?;
        // **唯一需要活到本层结束的东西**:每个节点的左右行数。
        // 一层最多 MAX_LEVEL_NODES 个节点,每个两个 u32。
        let part_totals = stream
            .alloc_zeros::<u32>(MAX_LEVEL_NODES * 2)
            .context("partition totals 分配")?;
        let part_right_out = stream
            .alloc_zeros::<u32>(n_rows)
            .context("下一层 rows arena 分配")?;
        let prediction_delta = stream
            .alloc_zeros::<f32>(n_rows)
            .context("prediction delta 分配")?;
        // device-side quantization 的 O(行数) buffer。只在开关打开时分配,
        // 关闭时一个字节都不占。
        // 解析顺序和 hist_streams 一致:**显式参数 > 环境变量 > 默认**。
        // 默认开:实测 HIGGS −27.1%、wide −5.5%,代价是 +8 B/行 device。
        // fast 模式的 gradient kernel 直接写 device 上的 `gpair_f32`,
        // 而那块 buffer 只在 device quantization 打开时才分配 ——
        // 所以 fast 蕴含 device quantize。显式给
        // `device_quantize=false` + `gpu_math="fast"` 是自相矛盾的组合,
        // 与其悄悄回退成 host 路径(那正是「两档跑成同一条路」那个 bug 的
        // 形状),不如直接报错。
        let fast_math_wanted = matches!(config.gpu_math, crate::types::GpuMath::Fast);
        if fast_math_wanted && config.device_quantize == Some(false) {
            bail!("gpu_math=\"fast\" 需要 device quantization,不能同时 device_quantize=false");
        }
        let device_quantize = device_quantize_wanted(&config) || fast_math_wanted;
        let n_gpair_blocks = GPAIR_BLOCKS.min(n_rows.div_ceil(GPAIR_THREADS).max(1));
        let (gpair_f32, gpair_max, gpair_sum) = if device_quantize {
            (
                Some(
                    stream
                        .alloc_zeros::<f32>(n_rows * 2)
                        .context("f32 gpair buffer 分配")?,
                ),
                Some(
                    stream
                        .alloc_zeros::<f64>(n_gpair_blocks * 2)
                        .context("gpair max 归约 buffer 分配")?,
                ),
                Some(
                    stream
                        .alloc_zeros::<i64>(n_gpair_blocks * 2)
                        .context("gpair sum 归约 buffer 分配")?,
                ),
            )
        } else {
            (None, None, None)
        };
        // 同一 binary 的严格 A/B 开关。HIGGS 实测 pinned 的 DMA 只快 2%，
        // 却多付 110 ms/轮的 pageable→pinned staging，并锁住 441 MB host
        // 内存，所以生产默认仍走 pageable；实验时显式打开。
        // 两块 staging 都在第一次 H2D 前完整覆写有效区间，不读取
        // alloc_pinned 的未初始化内容。
        let use_pinned = env_flag("FB_CUDA_PINNED");
        let gpair_host = if use_pinned {
            Some(
                unsafe { state.ctx.alloc_pinned::<i64>(n_rows * 2) }
                    .context("pinned gpair staging 分配")?,
            )
        } else {
            None
        };

        // 行主序转置暂存:**每个 slot 一份,但总预算不变**。
        // 以前是全局一块 32 MiB 配一把 `self.lock()`,而那把锁一直持有到
        // `stream.synchronize()`,把所有 slot 的上传串成一条 —— 多流之下
        // 重叠因此**结构上不可能出现**。现在把同一份预算切给每个 slot,
        // 显存总量一字不变,换到的是各自独立的 source lifetime。
        let rm_slot_count = hist_stream_count + resident_block_count;
        let rm_stage_bytes_per_slot = if row_major {
            (RM_STAGE_BYTES / rm_slot_count.max(1)).max(RM_STAGE_MIN_PER_SLOT)
        } else {
            0
        };
        // pinned 模式要两块轮转,所以把**同一份**每-slot 预算切成两半,
        // device 总量不变;多出来的只有 pinned host 内存(单独登记)。
        let rm_bufs = if row_major && use_pinned { 2usize } else { 1 };
        let rm_chunk_rows = if row_major {
            (rm_stage_bytes_per_slot / rm_bufs / max_block_feats.max(1))
                .clamp(1, n_rows)
                .max(1)
        } else {
            0
        };
        let rm_stage_len = rm_chunk_rows * max_block_feats;
        let alloc_rm_staging = |stream: &Arc<CudaStream>, what: &str| -> Result<Option<RmStaging>> {
            if !row_major {
                return Ok(None);
            }
            let mut dev = Vec::with_capacity(rm_bufs);
            let mut host = Vec::with_capacity(if use_pinned { rm_bufs } else { 0 });
            for b in 0..rm_bufs {
                dev.push(
                    stream
                        .alloc_zeros::<u8>(rm_stage_len.max(1))
                        .with_context(|| format!("{what} 行主序暂存 {b} 分配"))?,
                );
                if use_pinned {
                    // ⚠️ 双 buffer 轮转的**全部**安全性都压在
                    // `PinnedHostSlice` 自带的那个 event 上:H2D 之后由
                    // cudarc 记录,下一轮 `as_mut_slice()` 等它。
                    // 但 cudarc 只在 `is_managing_stream_synchronization()`
                    // 为真时才**真的记录**;为假时那个 event 从未被 record,
                    // `synchronize()` 于是立刻返回 —— host 会覆写一块仍在
                    // 被 DMA 读的 buffer,而且**静默**,只表现为偶发错值。
                    // 这个前提必须显式检查,不能默认它一直成立。
                    ensure!(
                        state.ctx.is_managing_stream_synchronization(),
                        "行主序 pinned 双 buffer 需要 cudarc 的 stream 事件跟踪,\
                         但当前 context 未开启;继续下去会静默覆写在飞的 pinned buffer"
                    );
                    host.push(
                        unsafe { state.ctx.alloc_pinned::<u8>(rm_stage_len.max(1)) }
                            .with_context(|| format!("{what} 行主序 pinned 暂存 {b} 分配"))?,
                    );
                }
            }
            Ok(Some(RmStaging { dev, host, chunk_rows: rm_chunk_rows }))
        };

        let mut hist_slots = Vec::with_capacity(hist_stream_count);
        for (slot, hist_stream) in hist_streams.into_iter().enumerate() {
            let bins = hist_stream
                .alloc_zeros::<u8>(bins_len.max(1))
                .with_context(|| format!("histogram slot {slot} 列块 buffer 分配"))?;
            let offsets = hist_stream
                .alloc_zeros::<u32>(offsets_len)
                .with_context(|| format!("histogram slot {slot} offsets buffer 分配"))?;
            let feature_select = hist_stream
                .alloc_zeros::<u32>(max_block_feats.max(1))
                .with_context(|| format!("histogram slot {slot} colsample buffer 分配"))?;
            let hist_out = hist_stream
                .alloc_zeros::<i64>((hist_words * hist_nodes_per_batch).max(1))
                .with_context(|| format!("histogram slot {slot} 输出 buffer 分配"))?;
            let batch_idx = hist_stream
                .alloc_zeros::<u32>(batch_idx_len)
                .with_context(|| format!("histogram slot {slot} batch 索引分配"))?;
            // 只有**列主序**那条臂会用 `bins_host`;行主序有自己的分块 pinned
            // 暂存(`RmStaging.host`),再留一份整块大小的 pinned buffer 是
            // 纯浪费(而且是按最宽块 × 行数算的,不小)。
            let bins_host = if use_pinned && !row_major {
                Some(
                    unsafe { state.ctx.alloc_pinned::<u8>(bins_len.max(1)) }
                        .with_context(|| format!("histogram slot {slot} pinned staging 分配"))?,
                )
            } else {
                None
            };
            let start_event = state
                .ctx
                .new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
                .with_context(|| format!("创建 histogram slot {slot} 起始 event"))?;
            let end_event = state
                .ctx
                .new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
                .with_context(|| format!("创建 histogram slot {slot} 结束 event"))?;
            let rm_stage = alloc_rm_staging(&hist_stream, &format!("histogram slot {slot}"))?;
            hist_slots.push(Mutex::new(GpuHistBuffers {
                stream: hist_stream,
                bins,
                offsets,
                feature_select,
                hist_out,
                batch_idx,
                hist_host: Vec::new(),
                bins_host,
                rm_stage,
                start_event,
                end_event,
                resident_feats: 0,
                resident_block: None,
            }));
        }
        for cache_slot in 0..resident_block_count {
            let hist_stream = state.ctx.new_stream()
                .with_context(|| format!("创建 resident block stream {cache_slot}"))?;
            let bins = hist_stream.alloc_zeros::<u8>(bins_len.max(1))
                .with_context(|| format!("resident block {cache_slot} buffer 分配"))?;
            let offsets = hist_stream.alloc_zeros::<u32>(offsets_len)
                .with_context(|| format!("resident block {cache_slot} offsets 分配"))?;
            let feature_select = hist_stream.alloc_zeros::<u32>(max_block_feats.max(1))
                .with_context(|| format!("resident block {cache_slot} colsample buffer 分配"))?;
            let hist_out = hist_stream.alloc_zeros::<i64>((hist_words * hist_nodes_per_batch).max(1))
                .with_context(|| format!("resident block {cache_slot} histogram 分配"))?;
            let batch_idx = hist_stream.alloc_zeros::<u32>(batch_idx_len)
                .with_context(|| format!("resident block {cache_slot} batch 索引分配"))?;
            let start_event = state.ctx.new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
            let end_event = state.ctx.new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))?;
            let rm_stage = alloc_rm_staging(&hist_stream, &format!("resident block {cache_slot}"))?;
            // ⚠️ resident slot 也要 pinned staging:它同样要**加载一次**自己的
            // 块,走的是和 streaming slot 完全相同的 `block_session`。
            // 以前这里恒为 None,于是 `FB_CUDA_PINNED=1` + 常驻块直接报
            // 「pinned 列块 staging 未分配」—— 在 residency 还是 env-only、
            // 默认 0 的年代没人同时打开这两个开关,所以一直没暴露。
            let bins_host = if use_pinned && !row_major {
                Some(
                    unsafe { state.ctx.alloc_pinned::<u8>(bins_len.max(1)) }
                        .with_context(|| format!("resident block {cache_slot} pinned staging 分配"))?,
                )
            } else {
                None
            };
            hist_slots.push(Mutex::new(GpuHistBuffers {
                stream: hist_stream, bins, offsets, feature_select, hist_out, batch_idx,
                hist_host: Vec::new(),
                bins_host, rm_stage, start_event, end_event,
                resident_feats: 0, resident_block: None,
            }));
        }

        // fast 模式的两块常驻 O(行数) buffer。必须登记进预算 ——
        // 「每一项预分配都必须出现在预算报告里」。
        let (pred_resident, label_resident) = if fast_math {
            (
                Some(stream.alloc_zeros::<f32>(n_rows.max(1)).context("fast pred 分配")?),
                Some(stream.alloc_zeros::<f32>(n_rows.max(1)).context("fast label 分配")?),
            )
        } else {
            (None, None)
        };
        // 行主序转置的落地区:H2D 先进这里(列主序),转置后写进 slot.bins
        let budget = GpuMemoryBudget {
            fast_math_bytes: if fast_math { n_rows * 4 * 2 } else { 0 },
            rm_stage_bytes: if row_major { rm_stage_len * rm_bufs * rm_slot_count } else { 0 },
            hist_streams: hist_stream_count + resident_block_count,
            gpair_bytes: n_rows * 2 * 8,
            row_idx_bytes: n_rows * 4,
            partition_out_bytes: n_rows * 4,
            // flags(n_rows) + counts/offsets(3 组)+ 整层 totals 收集区。
            // **每一项预分配都必须出现在预算报告里** —— 少报一块曾经害得
            // 一次 A/B 的显存代价被漏掉,见 internal-docs/history.md。
            // ⚠️ device quantization 的三块 buffer **必须**出现在预算里。
            // 上一次尝试正是漏报了它们(报告仍打 1142.1 MB),事后才发现。
            gpair_f32_bytes: if device_quantize { n_rows * 2 * 4 } else { 0 }
                + if device_quantize { n_gpair_blocks * 2 * (8 + 8) } else { 0 },
            partition_scratch_bytes: n_rows
                + (n_part_blocks * 3 + 2) * 4
                + MAX_LEVEL_NODES * 2 * 4,
            partition_col_bytes: n_rows,
            block_bins_bytes: bins_len * (hist_stream_count + resident_block_count),
            hist_bytes: hist_words * 8 * hist_nodes_per_batch * (hist_stream_count + resident_block_count),
            hist_nodes_batch_idx_bytes: if batch_capable {
                batch_idx_len * 4 * (hist_stream_count + resident_block_count)
            } else {
                0
            },
            offsets_bytes: offsets_len * 4 * (hist_stream_count + resident_block_count),
            hist_feature_select_bytes: max_block_feats.max(1) * 4
                * (hist_stream_count + resident_block_count),
            prediction_bytes: n_rows * 4,
        };
        // ⚠️ **预算必须report实际分配的东西,不能按公式猜。**
        // 列块 staging 只在**列主序 + pinned** 下存在,而且 streaming slot 和
        // resident slot 各一份;行主序那条路一份都不分配,改用自己的分块
        // pinned 暂存(每 slot 两块)。之前这里恒按 `use_pinned × bins_len ×
        // hist_stream_count` 算,于是行主序下报了 160 MB 根本没分配的内存,
        // 同时把真正分配了的行主序暂存漏掉 —— 两个方向都错。
        let pinned_col_slots = if use_pinned && !row_major {
            hist_stream_count + resident_block_count
        } else {
            0
        };
        let host_budget = GpuHostMemoryBudget {
            block_bins_bytes: pinned_col_slots * bins_len.max(1),
            rm_stage_bytes: if use_pinned && row_major {
                rm_stage_len.max(1) * rm_bufs * rm_slot_count
            } else {
                0
            },
            gpair_bytes: usize::from(use_pinned) * n_rows * 2 * 8,
        };

        let start_event = state
            .ctx
            .new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .context("创建 GPU 计时起始 event")?;
        let end_event = state
            .ctx
            .new_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
            .context("创建 GPU 计时结束 event")?;

        let ctx = Self {
            rm_transpose_fn,
            rm_hist_fn,
            rm_count_fn,
            apply_grad_fn,
            gpu_math,
            state: state.clone(),
            stream,
            hist_fn,
            hist_fused_fn,
            hist_fused_r4_fn,
            hist_batched_fn,
            gpair_max_fn,
            gpair_quantize_fn,
            count_fn,
            scan_fn,
            scatter_layer_fn,
            assign_leaf_fn,
            init_tree_fn,
            start_event,
            end_event,
            config,
            max_bins_per_tile,
            hist_nodes_per_batch,
            rm_chunk_rows,
            n_rows,
            partition_threads,
            timing,
            bucket_profile,
            use_pinned,
            budget,
            host_budget,
            buf: Mutex::new(GpuBuffers {
                pred_resident,
                label_resident,
                gpair,
                rows,
                part_col,
                part_flags,
                part_block_counts,
                part_left_offsets,
                part_right_offsets,
                part_right_out,
                next_rows_cursor: 0,
                part_totals,
                part_pending: Vec::new(),
                part_offsets_host: Vec::new(),
                prediction_delta,
                prediction_host: vec![0.0; n_rows],
                gpair_f32,
                gpair_max,
                gpair_sum,
                gpair_max_host: vec![0.0; n_gpair_blocks * 2],
                gpair_sum_host: vec![0; n_gpair_blocks * 2],
                gpair_host,
            }),
            hist_slots,
            hist_slot_pool: HistSlotPool::new(hist_stream_count),
            hist_stream_count,
            resident_block_count,
            n_blocks,
            gpu_memory_budget: config.gpu_memory_budget,
            device_ordinal,
            echoed_resident_col: std::sync::atomic::AtomicBool::new(false),
            echoed_upload_col: std::sync::atomic::AtomicBool::new(false),
        };
        // 见 `path_summary`:A/B 只信这一行,不信环境变量。
        if ctx.config.runtime_log {
            eprintln!("{}", ctx.path_summary());
        }
        Ok(ctx)
    }

    pub fn budget(&self) -> GpuMemoryBudget {
        self.budget
    }

    pub fn host_budget(&self) -> GpuHostMemoryBudget {
        self.host_budget
    }

    /// 这次训练**实际走了哪条路**。
    ///
    /// ⚠️ **A/B 必须打印它,不能只相信自己设了什么环境变量。**
    /// 真踩过:A/B 脚本用 `FOO= cmd` 关开关,而空字符串被 `is_some()`
    /// 判成 true,于是两档跑的是同一条路径,量出「约 1% 无效应」——
    /// 差点据此否掉一个 7.5% 的改动,也确实据此错误地否掉过一次
    /// device-side quantization。**症状和「真的没效果」无法区分**,
    /// 所以唯一可靠的办法是让被测代码自己报出它走的路径。
    ///
    /// 这里的每个值都取自**已经解析好的字段**,不是重新读一遍 env ——
    /// 重新读 env 只会把同一个 bug 再复现一遍。
    pub fn path_summary(&self) -> String {
        format!(
            // ⚠️ **回显实际选中的卡**。v0.0.1 支持 `cuda` / `cuda:N`,
            // 一次训练只用一张卡;日志里必须能看出用的是哪一张,
            // 否则多卡机器上无从判断 `cuda:1` 有没有生效。
            "GPU_PATH device=cuda:{} fused={} layout={} pinned={} hist_streams={} resident_blocks={} \
             partition_threads={} threads_per_block={}",
            self.device_ordinal,
            self.hist_fused_fn.is_some(),
            self.config.layout.label(),
            self.use_pinned,
            self.hist_stream_count,
            self.resident_block_count,
            self.partition_threads,
            self.config.threads_per_block,
        ) + if self.hist_fused_r4_fn.is_some() { " rows_per_thread=4" } else { " rows_per_thread=1" }
            + &format!(" hist_nodes_per_batch={}", if self.hist_batched_fn.is_some() { self.hist_nodes_per_batch } else { 1 })
            + if self.rm_hist_fn.is_some() {
                " layout_gpu=row_major hist_map=row_per_lane"
            } else {
                " layout_gpu=col_major hist_map=row_per_lane"
            }
            + match self.gpu_math {
                crate::types::GpuMath::Exact => " gpu_math=exact",
                crate::types::GpuMath::Fast => " gpu_math=fast",
            }
            + if self.gpair_quantize_fn.is_some() { " quantize=device" } else { " quantize=host" }
            // ⚠️ 这一段回显的是**列块 H2D 实际执行的那条路**,不是
            // `use_pinned` 这个字段。两者曾经不一致:行主序(默认)臂
            // 根本没有 pinned 分支,于是 `FB_CUDA_PINNED=1` 那一档跟
            // 对照组跑的是同一条路,量出来的差异全是噪声。
            // **A/B 脚本 grep 的必须是这里,不是环境变量。**
            // 常驻/流式的实际构成。⚠️ **术语要一致**:几乎全常驻的时候
            // 不要再笼统叫它 streaming,report 里直接给出三档之一。
            // ⚠️ 直接说出**约束了这次规划的那个数**。它以前只以
            // `free_vram` 的形式出现在 plan report 里,很容易被读成
            // "卡上还剩多少",而不是"用户给了多少上限"。
            + &match self.gpu_memory_budget {
                Some(b) => format!(" gpu_mem_budget={:.1}MB", b as f64 / (1024.0 * 1024.0)),
                None => " gpu_mem_budget=auto(free_vram)".to_string(),
            }
            + &format!(
                " residency={} resident_blocks={}/{} streamed_blocks={}",
                match (self.resident_block_count, self.n_blocks) {
                    (0, _) => "pure_streaming",
                    (r, t) if r >= t => "full_resident",
                    _ => "hybrid_partial_resident",
                },
                self.resident_block_count,
                self.n_blocks,
                self.n_blocks.saturating_sub(self.resident_block_count),
            )
            + &format!(
                " block_upload={} rm_stage_per_slot={}",
                match (self.rm_transpose_fn.is_some(), self.use_pinned) {
                    (true, true) => "rm_chunked_pinned_async_x2",
                    (true, false) => "rm_chunked_pageable",
                    (false, true) => "colmajor_pinned",
                    (false, false) => "colmajor_pageable",
                },
                self.rm_chunk_rows,
            )
            + &format!(" max_bins_per_tile={}", self.max_bins_per_tile)
    }

    pub fn timing_enabled(&self) -> bool {
        self.timing
    }

    /// 实际启用的 histogram stream 数。调用方需要据此决定 host submission
    /// 线程数；不能重新解析参数或环境变量，否则两边的默认值和优先级可能漂移。
    /// 有多少个**物理块**常驻显存。常驻块用物理块号做身份,
    /// 因此**不能**接收 colsample 压实过的块(压实块的内容随每棵树的
    /// 采样集变化,而 slot 是按物理块号缓存的)。
    pub fn resident_block_count(&self) -> usize {
        self.resident_block_count
    }

    /// Compact selected-feature histograms are available only on the normal
    /// batched column-major path. Bucket profiling keeps the legacy kernels.
    pub fn supports_compact_column_histogram(&self) -> bool {
        self.rm_hist_fn.is_none() && self.hist_batched_fn.is_some()
    }

    pub fn hist_stream_count(&self) -> usize {
        self.hist_stream_count
    }

    /// 当前 device 已用显存(总量 - 空闲)。含 driver / context 的固定开销,
    /// 所以它不等于 `budget().total()`,两个都要报。
    fn ensure_host_authority(&self, what: &str) -> Result<()> {
        if self.prediction_authority() == crate::types::PredictionAuthority::Device {
            bail!(
                "{what}:gpu_math=\"fast\" 下 device 才是 prediction 的权威副本,\
                 不要把它拉回 host;确实需要预测值时用 sync_prediction_to_host"
            );
        }
        Ok(())
    }

    pub fn device_mem_used(&self) -> Result<(usize, usize)> {
        let (free, total) = self.state.ctx.mem_get_info().context("读取显存用量")?;
        Ok((total - free, total))
    }

    /// Start a tree with identity rows resident on device and a cleared leaf
    /// prediction buffer. No root-row H2D is needed.
    pub fn begin_tree_rows(&self) -> Result<GpuRowSpan> {
        let mut buf = self.lock()?;
        buf.next_rows_cursor = 0;
        let n_rows = self.n_rows as u32;
        let threads = 256u32;
        let cfg = LaunchConfig {
            grid_dim: (n_rows.div_ceil(threads), 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        let GpuBuffers {
            rows,
            prediction_delta,
            ..
        } = &mut *buf;
        let mut launch = self.stream.launch_builder(&self.init_tree_fn);
        launch.arg(&n_rows).arg(rows).arg(prediction_delta);
        unsafe { launch.launch(cfg) }.context("launch init_tree_rows")?;
        Ok(GpuRowSpan {
            offset: 0,
            len: self.n_rows,
        })
    }

    /// 整层一次性取回所有节点的左右行数。
    ///
    /// **这是本层唯一一次阻塞 D2H。** 以前是每个节点两次(scan 之后、
    /// scatter 之前),因为 scatter 要 total_left 当标量;现在 scatter 自己
    /// 从 device 读那个值,host 就只剩「构造下一层 span」这一个用途,
    /// 而它整层攒到最后再做完全没问题 —— scatter 早就把行写到正确位置了。
    ///
    /// 实测代价:HIGGS 126 次/轮、23.8 ms/轮(profile 关闭,含 pipeline
    /// drain);压到每层一次之后是 6 次/轮。
    ///
    /// 取回的是整段 scratch 前缀(一次 memcpy),不是每个节点各取一小段 ——
    /// 那样只会把 126 次同步换成 126 次小拷贝。
    pub fn resolve_partitions(&self) -> Result<Vec<(GpuRowSpan, GpuRowSpan)>> {
        let mut buf = self.lock()?;
        let n = buf.part_pending.len();
        if n == 0 {
            return Ok(Vec::new());
        }
        let started = std::time::Instant::now();
        buf.part_offsets_host.clear();
        buf.part_offsets_host.resize(n * 2, 0);
        {
            let GpuBuffers { part_totals, part_offsets_host, .. } = &mut *buf;
            self.stream
                .memcpy_dtoh(&part_totals.slice(..n * 2), part_offsets_host.as_mut_slice())
                .context("partition totals D2H(整层一次)")?;
        }

        let mut out = Vec::with_capacity(n);
        for (slot, p) in buf.part_pending.iter().enumerate() {
            if p.n_selected == 0 {
                let empty = GpuRowSpan { offset: p.out_offset, len: 0 };
                out.push((empty, empty));
                continue;
            }
            let total_left = buf.part_offsets_host[slot * 2] as usize;
            let total_right = buf.part_offsets_host[slot * 2 + 1] as usize;
            debug_assert_eq!(total_left + total_right, p.n_selected);
            out.push((
                GpuRowSpan { offset: p.out_offset, len: total_left },
                GpuRowSpan { offset: p.out_offset + total_left, len: total_right },
            ));
        }
        Ok(out)
    }

    /// Make the just-built next-row arena current for the following depth.
    pub fn finish_partition_level(&self) -> Result<()> {
        let mut buf = self.lock()?;
        let GpuBuffers {
            rows,
            part_right_out,
            ..
        } = &mut *buf;
        std::mem::swap(rows, part_right_out);
        buf.next_rows_cursor = 0;
        buf.part_pending.clear();
        Ok(())
    }

    /// Diagnostic/correctness hook; the training loop itself never downloads
    /// row ids. Useful for asserting stable partition order against the CPU.
    pub fn download_rows(&self, span: GpuRowSpan) -> Result<Vec<RowId>> {
        if span.offset.saturating_add(span.len) > self.n_rows {
            bail!("download row span 越界:{span:?}");
        }
        let buf = self.lock()?;
        let view = buf.rows.slice(span.offset..span.offset + span.len);
        self.stream
            .clone_dtoh(&view)
            .context("row span D2H diagnostic")
    }

    /// Assign one terminal node's value to every row it owns on device.
    pub fn assign_leaf_value(&self, span: GpuRowSpan, value: f32) -> Result<()> {
        if span.offset.saturating_add(span.len) > self.n_rows {
            bail!("leaf row span 越界:{span:?}");
        }
        if span.len == 0 {
            return Ok(());
        }
        let mut buf = self.lock()?;
        let n_selected = span.len as u32;
        let threads = 256u32;
        let cfg = LaunchConfig {
            grid_dim: (n_selected.div_ceil(threads), 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };
        let GpuBuffers {
            rows,
            prediction_delta,
            ..
        } = &mut *buf;
        let rows_view = rows.slice(span.offset..span.offset + span.len);
        let mut launch = self.stream.launch_builder(&self.assign_leaf_fn);
        launch
            .arg(&rows_view)
            .arg(&n_selected)
            .arg(&value)
            .arg(prediction_delta);
        unsafe { launch.launch(cfg) }.context("launch assign_leaf_value")?;
        Ok(())
    }

    /// One prediction D2H per tree, then host gradients can start the next round.
    /// 只把 delta 取回 host,**不施加**。融合路径用它:施加和求梯度合成
    /// 下一轮开头的一次遍历。
    /// ⚠️ 只在 [`PredictionAuthority::Host`] 下合法。device 权威时
    /// 调用它说明有人想把 prediction 拉回 host —— 那是一次没必要的
    /// O(行数) D2H,直接报错而不是悄悄付钱。
    ///
    /// [`PredictionAuthority::Host`]: crate::types::PredictionAuthority::Host
    pub fn download_prediction_delta(&self) -> Result<f32> {
        self.ensure_host_authority("download_prediction_delta")?;
        let mut buf = self.lock()?;
        self.sync_if_timing()?;
        let started = std::time::Instant::now();
        let GpuBuffers { prediction_delta, prediction_host, .. } = &mut *buf;
        self.stream
            .memcpy_dtoh(&*prediction_delta, prediction_host.as_mut_slice())
            .context("leaf prediction D2H")?;
        self.stream
            .synchronize()
            .context("等待 leaf prediction D2H")?;
        Ok(if self.timing {
            started.elapsed().as_secs_f32() * 1e3
        } else {
            0.0
        })
    }

    /// 借出刚取回的 delta。调用方在同一次遍历里施加它并算出梯度。
    /// ⚠️ 同 [`Self::download_prediction_delta`]:device 权威时报错。
    pub fn with_prediction_delta(&self, f: &mut dyn FnMut(&[f32])) -> Result<()> {
        self.ensure_host_authority("with_prediction_delta")?;
        let buf = self.lock()?;
        f(buf.prediction_host.as_slice());
        Ok(())
    }


    /// 每轮传一次定点梯度。整个训练期间 buffer 不重新分配。
    pub fn upload_gpair(&self, gpair: &[GradPairFixed]) -> Result<f32> {
        if gpair.len() != self.n_rows {
            bail!("gpair 长度 {} 与训练行数 {} 不符", gpair.len(), self.n_rows);
        }
        // `&[GradPairFixed]` **已经就是**交错的 [grad, hess, ...] i64
        // (repr(C) + types.rs 里的 const 布局断言),所以 pageable 路径
        // 不需要再摊平一次 —— 旧代码那个循环产出的字节和输入逐位相同,
        // 等于每轮白读一遍、白写一遍整个 O(行数) 数组(HIGGS 上 2 × 168 MB),
        // 而且它落在计时器外面,在 profile 里完全看不见。
        let flat = GradPairFixed::flatten(gpair);
        let mut buf = self.lock()?;
        if self.use_pinned {
            // pinned staging 必须是**另一块**锁页内存,这一次拷贝省不掉;
            // 但可以从逐行循环换成一次 bulk copy。
            let host = buf
                .gpair_host
                .as_mut()
                .context("pinned gpair staging 未分配")?
                .as_mut_slice()
                .context("访问 pinned gpair staging")?;
            host[..flat.len()].copy_from_slice(flat);
        }
        self.sync_if_timing()?;
        let started = std::time::Instant::now();
        let GpuBuffers {
            gpair_host,
            gpair: gpair_dev,
            ..
        } = &mut *buf;
        if self.use_pinned {
            self.stream
                .memcpy_htod(
                    &*gpair_host.as_mut().context("pinned gpair staging 未分配")?,
                    gpair_dev,
                )
                .context("gpair H2D")?;
        } else {
            self.stream
                .memcpy_htod(flat, gpair_dev)
                .context("gpair H2D")?;
        }
        self.sync_if_timing()?;
        Ok(if self.timing {
            started.elapsed().as_secs_f32() * 1e3
        } else {
            0.0
        })
    }

    /// device-side quantization 是否可用(kernel 和 buffer 都已就绪)。
    pub fn device_quantize_enabled(&self) -> bool {
        self.gpair_quantize_fn.is_some()
    }

    /// 上传 **f32** 梯度,在 device 上求 scale、转定点,并归约出 root node_sum。
    ///
    /// 替代「host 转定点 + 上传 16 B/行」:走 PCIe 的从 `GradPairFixed`
    /// (16 B/行)变成 `GradPair`(8 B/行),**H2D 直接减半**;host 侧那两遍
    /// O(行数) 的扫描(求 max、逐元素量化)也一并没了。
    ///
    /// **为什么仍然和 CPU 逐位相同** —— 这是这个函数存在的全部前提:
    ///
    /// - `max |g as f64|`:f32→f64 精确、`fabs` 精确、`max` 精确且与顺序无关,
    ///   所以「device 分块求局部最大 + host 收尾」和串行扫一遍**逐位相同**;
    /// - scale 仍由 **host 的 `pow2_scale`** 算(同一个函数、同一个输入),
    ///   2 的幂这条不变量不受影响;
    /// - `(v as f64 * scale).round()`:scale 是 2 的幂 → 乘法在 f64 里
    ///   **不产生舍入**,而 CUDA 的 `round(double)` 与 Rust 的 `f64::round`
    ///   都是 ties-away-from-zero(已实测八个 tie 全部一致);
    /// - root sum 是整数加法,与顺序无关、溢出同样 mod 2^64。
    ///
    /// ⚠️ **梯度本身仍在 CPU 算。** `gradient_of` 含 exp/sigmoid,CUDA 的
    /// `expf` 不保证和 host libm 最后一位相同,搬上来会破坏
    /// 「GPU/CPU 模型逐字节相同」。这里只搬**纯算术**的两步。
    pub fn upload_and_quantize_gpair(
        &self,
        gpair: &[GradPair],
    ) -> Result<(GradQuantizer, GradPairFixed, f32)> {
        if gpair.len() != self.n_rows {
            bail!("gpair 长度 {} 与训练行数 {} 不符", gpair.len(), self.n_rows);
        }
        let max_fn = self
            .gpair_max_fn
            .as_ref()
            .context("device quantization 未启用")?;
        let quant_fn = self
            .gpair_quantize_fn
            .as_ref()
            .context("device quantization 未启用")?;

        // `GradPair` 是 #[repr(C)] 的两个 f32,所以整个 slice 本来就是交错的
        // [grad, hess, ...] —— 和 GradPairFixed::flatten 同一个道理,
        // host 侧不需要再打包一遍。
        let flat: &[f32] = unsafe {
            std::slice::from_raw_parts(gpair.as_ptr().cast::<f32>(), gpair.len() * 2)
        };
        let mut buf = self.lock()?;
        self.sync_if_timing()?;
        let started = std::time::Instant::now();
        {
            let f32_buf = buf.gpair_f32.as_mut().context("f32 gpair buffer 未分配")?;
            self.stream.memcpy_htod(flat, f32_buf).context("f32 gpair H2D")?;
        }
        self.quantize_resident_gpair(&mut buf, started)
    }

    /// `gpu_math = "fast"`:一次 launch 完成 prediction apply + objective
    /// gradient,然后走和 exact 完全相同的 device 量化尾巴。
    ///
    /// 相对 exact 每轮消掉的**真实数据搬运**:
    /// prediction D2H(4 B/行)、host 的按行 gradient 扫描(读 8 B/行 +
    /// 写 8 B/行)、gpair H2D(8 B/行)。HIGGS 上合计约 210 MB/轮。
    ///
    /// ⚠️ **不保证 CPU/GPU 逐字节相同** —— 这是这个模式的定义,不是缺陷。
    /// `subsample_*`:行采样。阈值由 host 折算(见 `subsample::threshold_for`),
    /// device 上只做整数比较,所以两边选出的行**必然相同**。
    /// `u64::MAX` = 不采样。
    pub fn fast_gradient_and_quantize(
        &self,
        objective: crate::types::Objective,
        subsample_threshold: u64,
        subsample_seed: u64,
        subsample_tree: u64,
    ) -> Result<(GradQuantizer, GradPairFixed, f32)> {
        let apply_fn = self
            .apply_grad_fn
            .as_ref()
            .context("gpu_math=fast 未启用")?;
        let mut buf = self.lock()?;
        self.sync_if_timing()?;
        let started = std::time::Instant::now();
        let n_rows_u32 = u32::try_from(self.n_rows).context("行数超过 u32")?;
        let obj_code: u32 = match objective {
            crate::types::Objective::SquaredError => 0,
            crate::types::Objective::Logistic => 1,
        };
        {
            let threads = 256u32;
            let blocks = n_rows_u32.div_ceil(threads).min(65535);
            let GpuBuffers {
                pred_resident,
                label_resident,
                prediction_delta,
                gpair_f32,
                ..
            } = &mut *buf;
            let pred = pred_resident.as_mut().context("fast pred buffer 未分配")?;
            let label = label_resident.as_ref().context("fast label buffer 未分配")?;
            let out = gpair_f32.as_mut().context("f32 gpair buffer 未分配")?;
            let cfg = LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: 0,
            };
            let mut launch = self.stream.launch_builder(apply_fn);
            launch
                .arg(pred)
                .arg(&*prediction_delta)
                .arg(label)
                .arg(&n_rows_u32)
                .arg(&obj_code)
                .arg(out)
                .arg(&subsample_threshold)
                .arg(&subsample_seed)
                .arg(&subsample_tree);
            unsafe { launch.launch(cfg) }.context("launch apply_delta_and_gradient")?;
        }
        self.quantize_resident_gpair(&mut buf, started)
    }

    pub fn fast_math_enabled(&self) -> bool {
        self.apply_grad_fn.is_some()
    }

    /// 训练期间 prediction 的权威副本在哪一侧。见
    /// [`crate::types::PredictionAuthority`]。
    pub fn prediction_authority(&self) -> crate::types::PredictionAuthority {
        if self.fast_math_enabled() {
            crate::types::PredictionAuthority::Device
        } else {
            crate::types::PredictionAuthority::Host
        }
    }

    /// 把 device 上的累计 raw-margin prediction 显式同步回 host。
    ///
    /// **只在外部真正需要预测值时调用**(比如训练后要拿训练集预测)。
    /// 训练循环本身不需要它 —— fast 模式下 device 才是权威副本。
    /// 它是 O(行数) 的一次 D2H,所以特意起了一个让代价显而易见的名字,
    /// 而不是藏在某个 getter 后面。
    pub fn sync_prediction_to_host(&self, out: &mut [f32]) -> Result<()> {
        if out.len() != self.n_rows {
            bail!("prediction 长度 {} 与训练行数 {} 不符", out.len(), self.n_rows);
        }
        let buf = self.lock()?;
        let pred = buf
            .pred_resident
            .as_ref()
            .context("device 上没有常驻 prediction(gpu_math=exact 时 host 已是权威副本)")?;
        self.stream.memcpy_dtoh(pred, out).context("prediction D2H")?;
        self.stream.synchronize().context("等待 prediction D2H")?;
        Ok(())
    }

    pub fn init_fast_math(&self, labels: &[f32], margin: f32) -> Result<()> {
        if labels.len() != self.n_rows {
            bail!("label 长度 {} 与训练行数 {} 不符", labels.len(), self.n_rows);
        }
        let mut buf = self.lock()?;
        let GpuBuffers { pred_resident, label_resident, .. } = &mut *buf;
        let label = label_resident.as_mut().context("fast label buffer 未分配")?;
        self.stream.memcpy_htod(labels, label).context("label H2D")?;
        let pred = pred_resident.as_mut().context("fast pred buffer 未分配")?;
        let host_pred = vec![margin; self.n_rows];
        self.stream.memcpy_htod(&host_pred, pred).context("初始 prediction H2D")?;
        self.stream.synchronize().context("等待 fast-math 初始化")?;
        Ok(())
    }

    /// exact 与 fast 共用的量化尾巴:max 归约 → host 求 pow2 scale →
    /// 定点量化 → root sum 归约。gpair_f32 此时已经在 device 上,
    /// 两条路径从这里开始逐字相同。
    fn quantize_resident_gpair(
        &self,
        buf: &mut GpuBuffers,
        started: std::time::Instant,
    ) -> Result<(GradQuantizer, GradPairFixed, f32)> {
        let max_fn = self.gpair_max_fn.as_ref().context("device quantization 未启用")?;
        let quant_fn = self.gpair_quantize_fn.as_ref().context("device quantization 未启用")?;
        let n_rows_u32 = u32::try_from(self.n_rows).context("行数超过 u32")?;
        let threads = GPAIR_THREADS as u32;
        let blocks = u32::try_from(buf.gpair_max_host.len() / 2).context("归约 block 数超过 u32")?;
        {
            let GpuBuffers { gpair_f32, gpair_max, .. } = &mut *buf;
            let src = gpair_f32.as_ref().context("f32 gpair buffer 未分配")?;
            let dst = gpair_max.as_mut().context("gpair max buffer 未分配")?;
            let cfg = LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: threads * 2 * std::mem::size_of::<f64>() as u32,
            };
            let mut launch = self.stream.launch_builder(max_fn);
            launch.arg(src).arg(&n_rows_u32).arg(dst);
            unsafe { launch.launch(cfg) }.context("launch gpair_max_abs")?;
        }
        {
            let GpuBuffers { gpair_max, gpair_max_host, .. } = &mut *buf;
            let src = gpair_max.as_ref().context("gpair max buffer 未分配")?;
            self.stream
                .memcpy_dtoh(src, gpair_max_host.as_mut_slice())
                .context("gpair max D2H")?;
        }
        self.stream.synchronize().context("等待 gpair max D2H")?;

        // host 收尾。很小(block 数级别),而且 max 精确,所以整体结果和
        // 「host 串行扫全部行」逐位相同。
        let (mut max_grad, mut max_hess) = (0.0f64, 0.0f64);
        for pair in buf.gpair_max_host.as_chunks::<2>().0 {
            max_grad = max_grad.max(pair[0]);
            max_hess = max_hess.max(pair[1]);
        }
        let quantizer = GradQuantizer::from_bounds(max_grad, max_hess, self.n_rows);

        {
            let grad_scale = quantizer.grad_scale();
            let hess_scale = quantizer.hess_scale();
            let GpuBuffers { gpair_f32, gpair: gpair_dev, gpair_sum, .. } = &mut *buf;
            let src = gpair_f32.as_ref().context("f32 gpair buffer 未分配")?;
            let sums = gpair_sum.as_mut().context("gpair sum buffer 未分配")?;
            let cfg = LaunchConfig {
                grid_dim: (blocks, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: threads * 2 * std::mem::size_of::<i64>() as u32,
            };
            let mut launch = self.stream.launch_builder(quant_fn);
            launch
                .arg(src)
                .arg(&n_rows_u32)
                .arg(&grad_scale)
                .arg(&hess_scale)
                .arg(gpair_dev)
                .arg(sums);
            unsafe { launch.launch(cfg) }.context("launch gpair_quantize")?;
        }
        {
            let GpuBuffers { gpair_sum, gpair_sum_host, .. } = &mut *buf;
            let src = gpair_sum.as_ref().context("gpair sum buffer 未分配")?;
            self.stream
                .memcpy_dtoh(src, gpair_sum_host.as_mut_slice())
                .context("gpair sum D2H")?;
        }
        self.stream.synchronize().context("等待 gpair sum D2H")?;
        // 用 `+=`(AddAssign)而不是 wrapping_add:debug 下仍然检查溢出。
        let mut root_sum = GradPairFixed::default();
        for pair in buf.gpair_sum_host.as_chunks::<2>().0 {
            root_sum += GradPairFixed { grad: pair[0], hess: pair[1] };
        }

        self.sync_if_timing()?;
        let elapsed = if self.timing {
            started.elapsed().as_secs_f32() * 1e3
        } else {
            0.0
        };
        Ok((quantizer, root_sum, elapsed))
    }

    /// 借出一个列块:H2D 一次,然后本块内所有活跃节点的直方图共用它。
    ///
    /// 返回值独占两套 histogram slot 之一；另一列块可以同时在另一条
    /// stream 上传或执行 kernel。块结束后 slot 回到池里继续复用。
    pub fn block_session(
        &self,
        block_index: usize,
        block_data: &[Bin],
        block_n_rows: usize,
        n_feats: usize,
        feat_offsets: &[u32],
        selected_local: Option<&[usize]>,
    ) -> Result<GpuBlockSession<'_>> {
        // selected-row packing 之后,交给 histogram 的块**行数会更少**
        // (只含采样行),所以这里不能再要求等于训练行数。
        //
        // ⚠️ 但仍然要卡上界:比训练行数还多一定是算错了。
        // 行主序 kernel 不用 `n_rows` 当 stride(它按 `row * n_feats` 寻址),
        // 所以少行数是安全的;列主序参照臂会用它当 stride,因此那条路上
        // 压实是关掉的(见 train.rs 里只在 streaming + 行主序时压实)。
        if block_n_rows > self.n_rows {
            bail!(
                "列块行数 {block_n_rows} 超过训练行数 {}",
                self.n_rows
            );
        }
        if block_data.len() != n_feats.saturating_mul(block_n_rows) {
            bail!("bins 长度与 n_feats × n_rows 不符");
        }
        if feat_offsets.is_empty() || feat_offsets.first() != Some(&0) {
            bail!("feat_offsets 必须从 0 开始且非空");
        }
        let hist_n_feats = feat_offsets.len() - 1;
        if feat_offsets.windows(2).any(|pair| pair[0] > pair[1]) {
            bail!("feat_offsets 必须单调不减");
        }
        if let Some(selected) = selected_local {
            if selected.windows(2).any(|pair| pair[0] >= pair[1])
                || selected.iter().any(|&feat| feat >= n_feats)
            {
                bail!("colsample 局部特征必须严格递增且小于物理 n_feats");
            }
        }
        let column_major = self.rm_transpose_fn.is_none();
        if column_major {
            if let Some(selected) = selected_local {
                if selected.len() != hist_n_feats {
                    bail!("column-major selected_local 必须与紧凑 histogram 特征数相等");
                }
            } else if hist_n_feats != n_feats {
                bail!("无 selected_local 时 histogram 特征数必须等于物理 n_feats");
            }
        } else if hist_n_feats != n_feats {
            bail!("row-major histogram offsets 必须保持物理全宽");
        }
        let n_hist = feat_offsets[hist_n_feats] as usize;
        // 独立入口每次调用都全量扫一遍 bin 范围;那是 O(行数 × 特征数),
        // 和 kernel 本身一个量级,接进训练循环就不能每块都付。
        //
        // 不变量本来就在构造边界:`ColumnBlock` 的 bin 由 `BinCuts::find_bin`
        // 产生,要么 < n_bins 要么是 MISSING_BIN。debug 构建仍然验一遍,
        // 因为越界的后果是 shared memory 越界写,不是一个错数字。
        debug_assert!(
            (0..hist_n_feats).all(|hist_feat| {
                let physical_feat = if column_major {
                    selected_local.map_or(hist_feat, |selected| selected[hist_feat])
                } else {
                    hist_feat
                };
                let n_bins = feat_offsets[hist_feat + 1] - feat_offsets[hist_feat];
                block_data[physical_feat * block_n_rows..(physical_feat + 1) * block_n_rows]
                    .iter()
                    .all(|&bin| bin == crate::types::MISSING_BIN || u32::from(bin) < n_bins)
            }),
            "列块里有超出 feat_offsets 宽度的 bin"
        );

        let (slot_index, lease) = if block_index < self.resident_block_count {
            (self.hist_stream_count + block_index, None)
        } else {
            let lease = self.hist_slot_pool.acquire()?;
            (lease.index, Some(lease))
        };
        let mut slot = self.hist_slots[slot_index]
            .lock()
            .map_err(|_| anyhow::anyhow!("GPU histogram slot {slot_index} poisoned"))?;
        if block_data.len() > slot.bins.len() {
            bail!(
                "列块 {} 字节超过预分配的 {} 字节 buffer",
                block_data.len(),
                slot.bins.len()
            );
        }
        if n_hist * 2 > slot.hist_out.len() {
            bail!("直方图宽度 {n_hist} 超过预分配上限");
        }

        let cache_hit = slot.resident_block == Some(block_index);
        let selection_active = selected_local.is_some_and(|selected| selected.len() < n_feats);
        let row_major_feat_mask = if selection_active {
            let mut mask = 0u32;
            for &feat in selected_local.expect("selection_active 蕴含 selected_local") {
                if feat < 32 {
                    mask |= 1u32 << feat;
                }
            }
            mask
        } else {
            !0u32
        };
        // 这里是唯一一次量化数据的 H2D。块本身在调用方的 `with_block`
        // 闭包结束时就还回去了,device 上留下的只是 slot 的可覆写 buffer。
        // 行主序的 pinned 落地在下面的分块循环里做(`RmStaging.host`),
        // 这里这次预拷贝**只服务列主序那条臂**。不加这个判据的话,行主序
        // 下会白白把整块再搬进一个根本不会被读的 buffer。
        if self.use_pinned && !cache_hit && self.rm_transpose_fn.is_none() {
            let host = slot
                .bins_host
                .as_mut()
                .context("pinned 列块 staging 未分配")?
                .as_mut_slice()
                .context("访问 pinned 列块 staging")?;
            host[..block_data.len()].copy_from_slice(block_data);
        }
        let hist_stream = slot.stream.clone();
        self.sync_hist_if_timing(&hist_stream)?;
        let started = std::time::Instant::now();
        if !cache_hit {
        {
            let GpuHistBuffers {
                bins,
                bins_host,
                rm_stage,
                ..
            } = &mut *slot;
            // 行主序原型:H2D 仍然是**原封不动的列主序**(host / Parquet /
            // Arrow / 量化 cache / FFI 全不动),只是先落到暂存区,再在
            // device 上转置一次进 slot.bins。转置**每个 resident 块只做一次**,
            // 之后所有层、所有轮直接复用 —— 这是能不能摊薄的关键。
            match self.rm_transpose_fn.as_ref() {
                Some(tfn) => {
                    // **按行分块**上传 + 转置,暂存区因此和列块大小无关。
                    // 每块的列主序源数据在 host 上是 n_feats 段、段内连续,
                    // 所以一个 chunk 要 n_feats 次 H2D;这只在 warm-up 发生
                    // (resident 块每块一次),之后全是 cache hit。
                    // ⚠️ 这里**没有** `self.lock()`。暂存区是 slot 自己的,
                    // 所以下面的 `synchronize()` 只等自己这条流 —— 全局锁
                    // 曾经把它变成"等所有流",多流重叠因此结构上不可能。
                    let staging = rm_stage.as_mut().context("行主序暂存未分配")?;
                    let pinned = staging.pinned();
                    let n_bufs = staging.dev.len();
                    let n_rows_total = block_n_rows;
                    let chunk = staging.chunk_rows.max(1);
                    let n_feats_u = u32::try_from(n_feats).context("特征数超过 u32")?;
                    let mut r0 = 0usize;
                    let mut turn = 0usize;
                    while r0 < n_rows_total {
                        let r1 = (r0 + chunk).min(n_rows_total);
                        let len = r1 - r0;
                        let b = turn % n_bufs;
                        if pinned {
                            // ⚠️ `as_mut_slice()` 只等**这一块**自己的 event,
                            // 不是整条流。所以上一块的 H2D + 转置仍在飞,
                            // host 这边已经可以填下一块 —— 重叠就在这里。
                            // 换成 `hist_stream.synchronize()` 会当场毁掉它。
                            let hbuf = staging.host[b]
                                .as_mut_slice()
                                .context("访问行主序 pinned 暂存")?;
                            for f in 0..n_feats {
                                let src =
                                    &block_data[f * n_rows_total + r0..f * n_rows_total + r1];
                                hbuf[f * len..(f + 1) * len].copy_from_slice(src);
                            }
                            // pinned 源 → 这一次 `cuMemcpyHtoDAsync` 才**真的**异步,
                            // 而且整块只发一次,不是 n_feats 次。
                            let src = PinnedPrefix {
                                inner: &mut staging.host[b],
                                len: len * n_feats,
                            };
                            let mut dst = staging.dev[b].slice_mut(..len * n_feats);
                            hist_stream
                                .memcpy_htod(&src, &mut dst)
                                .context("量化列块 H2D(行主序 pinned)")?;
                        } else {
                            for f in 0..n_feats {
                                let src =
                                    &block_data[f * n_rows_total + r0..f * n_rows_total + r1];
                                let mut dst = staging.dev[b].slice_mut(f * len..(f + 1) * len);
                                hist_stream
                                    .memcpy_htod(src, &mut dst)
                                    .context("量化列块 H2D(行主序分块)")?;
                            }
                        }
                        let src = staging.dev[b].slice(..len * n_feats);
                        let mut dst = bins.slice_mut(r0 * n_feats..r1 * n_feats);
                        let len_u = u32::try_from(len).context("chunk 行数超过 u32")?;
                        let cfg = LaunchConfig {
                            grid_dim: (len_u.div_ceil(32), n_feats_u.div_ceil(32), 1),
                            block_dim: (32, 32, 1),
                            shared_mem_bytes: 0,
                        };
                        let mut lb = hist_stream.launch_builder(tfn);
                        lb.arg(&src).arg(&mut dst).arg(&len_u).arg(&n_feats_u);
                        unsafe { lb.launch(cfg) }
                            .context("launch transpose_bins_to_row_major")?;
                        // ⚠️ 暂存区下一个 chunk 就要覆写,而转置是异步的。
                        // **enqueue 完成不等于 buffer 可复用。**
                        //
                        // 试过删掉它:同流顺序**确实**保证了正确性(暂存区
                        // 已经是 per-slot,一个 slot 只有一条流),但墙钟上
                        // 单流只快 1.2~1.5%(噪声量级),而**双流反而变慢**
                        // (HIGGS 0.481 → 0.546、wide 3.220 → 3.365)——
                        // 少了这个节流,两条流的可分页 H2D 互相抢 driver 的
                        // 暂存池。所以保留:它不是双流收益的来源
                        //(那个假设已被这次实测推翻),删了也不划算。
                        if !pinned {
                            hist_stream.synchronize().context("等待行主序转置 chunk")?;
                        }
                        r0 = r1;
                        turn += 1;
                    }
                    if pinned {
                        // 块级收尾:两块暂存都要在离开前退休,否则下一块的
                        // host 填充会覆写仍在被读的 pinned buffer。
                        hist_stream.synchronize().context("等待行主序 pinned 收尾")?;
                    }
                }
                None => {
                    let mut bins_view = bins.slice_mut(..block_data.len());
                    if self.use_pinned {
                        let pinned = PinnedPrefix {
                            inner: bins_host.as_mut().context("pinned 列块 staging 未分配")?,
                            len: block_data.len(),
                        };
                        hist_stream.memcpy_htod(&pinned, &mut bins_view).context("量化列块 H2D")?;
                    } else {
                        hist_stream.memcpy_htod(block_data, &mut bins_view).context("量化列块 H2D")?;
                    }
                }
            }
        }
        if block_index < self.resident_block_count {
            // 宽度和块号**必须一起写** —— 只写块号就等于让下游去反推宽度,
            // 而那正是上面说的那个读偏。
            slot.resident_feats = n_feats;
            slot.resident_block = Some(block_index);
        }
        }
        {
            let mut offsets_view = slot.offsets.slice_mut(..feat_offsets.len());
            hist_stream
                .memcpy_htod(feat_offsets, &mut offsets_view)
                .context("feat_offsets H2D")?;
        }
        if selection_active {
            let selected = selected_local.expect("selection_active 蕴含 selected_local");
            let selected_u32: Vec<u32> = selected
                .iter()
                .map(|&feat| u32::try_from(feat).context("局部特征下标超过 u32"))
                .collect::<Result<_>>()?;
            let mut feature_select_view = slot.feature_select.slice_mut(..selected_u32.len());
            hist_stream
                .memcpy_htod(&selected_u32, &mut feature_select_view)
                .context("colsample feature indices H2D")?;
        }
        self.sync_hist_if_timing(&hist_stream)?;
        let block_h2d_ms = if self.timing {
            started.elapsed().as_secs_f32() * 1e3
        } else {
            0.0
        };

        let timing = GpuSegmentTiming {
            block_h2d_ms,
            block_h2d_calls: u64::from(!cache_hit) + 1 + u64::from(selection_active),
            block_h2d_bytes: u64::from(!cache_hit) * block_data.len() as u64
                + std::mem::size_of_val(feat_offsets) as u64
                + selected_local.map_or(0, |selected| {
                    (selected.len() * std::mem::size_of::<u32>()) as u64
                }),
            ..Default::default()
        };
        Ok(GpuBlockSession {
            ctx: self,
            slot: Some(slot),
            lease,
            physical_n_feats: n_feats,
            n_feats: hist_n_feats,
            n_hist,
            feat_offsets: feat_offsets.to_vec(),
            selection_active,
            row_major_feat_mask,
            timing,
            feature_tiles: 0,
        })
    }

    /// resident 命中时的 partition:**一个字节都不传**。
    ///
    /// 这一列的 bin 早就在 `slot.bins` 里(histogram 那一趟上传的,而且
    /// resident slot 不会被覆写),偏移就是 `local_feat * n_rows`。
    /// 以前这里无条件走 host `feature_column()` 新建一个 `Vec<Bin>` 再
    /// `memcpy_htod` —— 实测 HIGGS 上是 `PART_FETCH` 25.9 +
    /// `GPU_PART_COL_H2D` 52.1 = **78 ms/轮**,占全部 H2D 流量的 **58%**,
    /// 而这些字节**已经在卡上**。
    ///
    /// partition kernel 的签名本来就是 `const uint8_t* col` + `n_rows`
    /// stride,所以 device subview 直接就能喂进去 —— kernel 没改、
    /// 显存没增加、streaming 路径一行没动。
    ///
    /// 返回 `Ok(None)` 表示这个块此刻不是 resident 命中,调用方应当回退到
    /// `partition_session`(自己去 host 取列)。
    ///
    /// 锁序:**先 slot 再 buf**,和 `block_session` 一致。两者其实是分相位的
    /// (整层 histogram 做完才进 partition),但顺序统一了就不用依赖那个前提。
    pub fn partition_session_resident(
        &self,
        block_index: usize,
        local_feat: usize,
    ) -> Result<Option<GpuPartitionSession<'_>>> {
        if block_index >= self.resident_block_count {
            return Ok(None);
        }
        let slot_index = self.hist_stream_count + block_index;
        let slot = self.hist_slots[slot_index]
            .lock()
            .map_err(|_| anyhow::anyhow!("GPU histogram slot {slot_index} poisoned"))?;
        if slot.resident_block != Some(block_index) {
            return Ok(None);
        }
        // ⚠️ 两种布局的越界判据不同,不能共用一个:
        // 列主序取的是连续一列 `[feat*n_rows, (feat+1)*n_rows)`;
        // 行主序按 `row * n_feats + feat` 寻址,约束是 `feat < n_feats`。
        let feats = slot.resident_feats;
        if feats == 0 || feats > slot.bins.len() / self.n_rows.max(1) {
            bail!("resident 块 {block_index} 的特征数 {feats} 和 buffer 对不上");
        }
        if self.rm_count_fn.is_some() {
            if local_feat >= feats {
                bail!("resident 列 {local_feat} 越过块宽 {feats}");
            }
        } else {
            let end = local_feat
                .checked_add(1)
                .and_then(|f| f.checked_mul(self.n_rows))
                .context("resident 列偏移溢出")?;
            if end > slot.bins.len() {
                bail!("resident 列 {local_feat} 越过块 buffer");
            }
        }
        self.echo_part_col_source(true);
        let buf = self.lock()?;
        Ok(Some(GpuPartitionSession {
            ctx: self,
            buf,
            resident: Some((slot, local_feat)),
            timing: GpuSegmentTiming::default(),
        }))
    }

    /// A/B 必须能看出实际走了哪条路,不能只看自己传了什么参数。
    /// 每种来源只打一次,不刷屏。
    fn echo_part_col_source(&self, resident: bool) {
        if !crate::train::prof::enabled() {
            return;
        }
        use std::sync::atomic::Ordering;
        let flag = if resident { &self.echoed_resident_col } else { &self.echoed_upload_col };
        if !flag.swap(true, Ordering::Relaxed) {
            eprintln!(
                "PART_COL_SOURCE={}",
                if resident { "resident" } else { "upload" }
            );
        }
    }

    /// 借出一整列给行重分区:选中的特征列 H2D 一次,本层所有用到它的
    /// 节点共用。build_tree 的第二趟本来就是「外层特征、内层节点」。
    ///
    /// **这是回退路径**:streaming 块,或者 resident 尚未命中时才走。
    /// resident 命中请走 `partition_session_resident`。
    pub fn partition_session(&self, col: &[Bin]) -> Result<GpuPartitionSession<'_>> {
        if col.len() != self.n_rows {
            bail!(
                "partition 列长度 {} 与训练行数 {} 不符",
                col.len(),
                self.n_rows
            );
        }
        let mut buf = self.lock()?;
        self.sync_if_timing()?;
        let started = std::time::Instant::now();
        self.stream
            .memcpy_htod(col, &mut buf.part_col)
            .context("partition 列 H2D")?;
        self.sync_if_timing()?;
        let h2d_ms = if self.timing {
            started.elapsed().as_secs_f32() * 1e3
        } else {
            0.0
        };
        let timing = GpuSegmentTiming {
            block_h2d_ms: h2d_ms,
            block_h2d_calls: 1,
            block_h2d_bytes: col.len() as u64,
            ..Default::default()
        };
        self.echo_part_col_source(false);
        Ok(GpuPartitionSession {
            ctx: self,
            buf,
            resident: None,
            timing,
        })
    }

    fn lock(&self) -> Result<MutexGuard<'_, GpuBuffers>> {
        self.buf
            .lock()
            .map_err(|_| anyhow::anyhow!("GPU 训练 buffer 锁 poisoned"))
    }

    /// 只在 timing 打开时同步。目的是把「真实传输」和「等前面的 kernel」
    /// 分开;关掉 timing 的正常路径不会因为计时而多出 sync。
    fn sync_if_timing(&self) -> Result<()> {
        if self.timing {
            self.stream.synchronize().context("GPU 分段计时同步")?;
        }
        Ok(())
    }

    fn sync_hist_if_timing(&self, stream: &CudaStream) -> Result<()> {
        if self.timing {
            stream.synchronize().context("GPU histogram 分段计时同步")?;
        }
        Ok(())
    }
}

/// 一个已经在显存里的列块。生命周期结束就把 GPU 锁还回去,但 buffer
/// 本身留着给下一个块覆写。
pub struct GpuBlockSession<'a> {
    ctx: &'a GpuTrainCtx,
    // Option 只为 Drop 时先释放 mutex guard、再把 slot 号还给 pool。
    slot: Option<MutexGuard<'a, GpuHistBuffers>>,
    lease: Option<HistSlotLease<'a>>,
    /// Number of columns in the resident/streamed bins allocation.
    physical_n_feats: usize,
    /// Number of logical columns represented in the histogram output.
    n_feats: usize,
    n_hist: usize,
    feat_offsets: Vec<u32>,
    selection_active: bool,
    row_major_feat_mask: u32,
    timing: GpuSegmentTiming,
    feature_tiles: usize,
}

impl Drop for GpuBlockSession<'_> {
    fn drop(&mut self) {
        drop(self.slot.take());
        drop(self.lease.take());
    }
}

impl GpuBlockSession<'_> {
    /// Histogram a node whose rows already live in the current device arena.
    pub fn histogram_device(
        &mut self,
        span: GpuRowSpan,
        out: &mut Vec<GradPairFixed>,
    ) -> Result<()> {
        if span.offset.saturating_add(span.len) > self.ctx.n_rows {
            bail!("histogram device row span 越界:{span:?}");
        }
        self.histogram_span(span, out, u32::MAX, usize::MAX)
    }

    /// Same as [`Self::histogram_device`], but carries the tree position so the
    /// opt-in bucket profiler can attribute each launch. `depth` / `node_index`
    /// are ignored unless `FB_HIST_BUCKET_PROFILE` is set.
    pub fn histogram_device_at_depth(
        &mut self,
        span: GpuRowSpan,
        out: &mut Vec<GradPairFixed>,
        depth: u32,
        node_index: usize,
    ) -> Result<()> {
        if span.offset.saturating_add(span.len) > self.ctx.n_rows {
            bail!("histogram device row span 越界:{span:?}");
        }
        self.histogram_span(span, out, depth, node_index)
    }

    fn histogram_span(
        &mut self,
        span: GpuRowSpan,
        out: &mut Vec<GradPairFixed>,
        depth: u32,
        node_index: usize,
    ) -> Result<()> {
        out.clear();
        out.resize(self.n_hist, GradPairFixed::default());
        let n_selected = span.len;
        if self.n_hist == 0 || n_selected == 0 {
            return Ok(());
        }

        let n_rows_u32 = self.ctx.n_rows as u32;
        let threads_per_block = self.ctx.config.threads_per_block;
        let full_rows = n_selected / threads_per_block as usize * threads_per_block as usize;
        let full_blocks = u32::try_from(full_rows / threads_per_block as usize)
            .context("CUDA histogram grid 超过 u32")?;
        let tail_rows = (n_selected - full_rows) as u32;
        let bytes_per_bin = std::mem::size_of::<GradPairFixed>();
        let stream = self
            .slot
            .as_ref()
            .expect("histogram slot lease missing")
            .stream
            .clone();

        {
            let slot = self.slot.as_mut().expect("histogram slot lease missing");
            let mut view = slot.hist_out.slice_mut(..self.n_hist * 2);
            stream
                .memset_zeros(&mut view)
                .context("清零直方图 buffer")?;
        }

        if self.ctx.timing && !self.ctx.bucket_profile {
            self.slot
                .as_ref()
                .expect("histogram slot lease missing")
                .start_event
                .record(&stream)
                .context("记录 shared histogram 起始 event")?;
        }

        // 每个节点要 launch 几个 feature tile。只有 bucket profile 需要它,
        // 所以平时不算。
        let feature_tile_count = if self.ctx.bucket_profile {
            let mut count = 0usize;
            let mut begin = 0usize;
            while begin < self.n_feats {
                let hist_begin = self.feat_offsets[begin] as usize;
                let mut end = begin;
                while end < self.n_feats
                    && self.feat_offsets[end + 1] as usize - hist_begin
                        <= self.ctx.max_bins_per_tile
                {
                    end += 1;
                }
                debug_assert!(end > begin);
                begin = end;
                count += 1;
            }
            count
        } else {
            0
        };

        // gpair/rows 在整个 histogram phase 都只读。锁只保护 host 侧取 view
        // 和 launch 参数组装；kernel 排进各自 stream 后立即释放，让另一
        // slot 可以并发提交。进入 partition / 下一轮前，调用方已经收齐了
        // 每块 D2H 结果，因此不会和后续写操作竞态。
        let core = self.ctx.lock()?;

        // 融合路径:一次 launch 处理整个列块的所有 feature tile。
        //
        // 每个线程只读一次自己那一行的 gpair(16 B)和 row_idx(4 B),
        // 而不是每个 tile 各读一次。wide 的 `cols_per_block=32` 切成 4 个
        // tile,所以这一项直接少 4 倍;实测一次 8-feature launch 的 global
        // load 是 540 MB,而这 8 个特征的 bins 只有 80 MB。
        //
        // bins 读取量、shared 原子次数、init/flush 次数完全不变,所以
        // 位精确不受影响。
        if let Some(fused_fn) = self.ctx.hist_fused_fn.as_ref() {
            // R=4 时每个 block 覆盖 threads*4 行,余数交回 R=1 的融合 kernel。
            // 两个 kernel 的数值行为完全相同(同样的 atomic、同样的进位),
            // 所以拆不拆边界都不影响位精确。
            let rows_per_thread = if self.ctx.hist_fused_r4_fn.is_some() { 4usize } else { 1 };
            let rows_per_block = threads_per_block as usize * rows_per_thread;
            let r4_rows = if rows_per_thread > 1 {
                n_selected / rows_per_block * rows_per_block
            } else {
                0
            };
            let r4_blocks = u32::try_from(r4_rows / rows_per_block)
                .context("R4 grid 超过 u32")?;
            // shared 按这个列块**实际用到的最大 tile** 来要,不要一律按
            // `max_bins_per_tile` 顶格要 —— 窄列块顶格会白占 shared,
            // 把每个 SM 能常驻的 block 数压下去。这里跑的贪心切分和
            // kernel 内的逐字相同。
            let mut max_tile_slots = 0usize;
            let mut probe = 0usize;
            while probe < self.n_feats {
                let hist_begin = self.feat_offsets[probe] as usize;
                let mut end = probe;
                while end < self.n_feats
                    && self.feat_offsets[end + 1] as usize - hist_begin
                        <= self.ctx.max_bins_per_tile
                {
                    end += 1;
                }
                debug_assert!(end > probe, "单个特征的 bin 数超过 tile 上限");
                max_tile_slots =
                    max_tile_slots.max(self.feat_offsets[end] as usize - hist_begin);
                probe = end;
            }
            let shared_mem_bytes = u32::try_from(max_tile_slots * bytes_per_bin)
                .context("shared-memory tile 大小超过 u32")?;
            let n_feats_u32 = u32::try_from(self.n_feats).context("列块特征数超过 u32")?;
            let max_tile_bins =
                u32::try_from(self.ctx.max_bins_per_tile).context("tile bin 上限超过 u32")?;

            let slot = self.slot.as_mut().expect("histogram slot lease missing");
            let GpuHistBuffers { bins, offsets, hist_out, .. } = &mut **slot;
            let GpuBuffers { gpair, rows, .. } = &*core;
            let bins_view = bins.slice(0..self.n_feats * self.ctx.n_rows);
            let offsets_view = offsets.slice(0..=self.n_feats);
            let mut out_view = hist_out.slice_mut(..self.n_hist * 2);

            // R=4 段:整块整块地吃掉 rows_per_block 的整数倍。
            if r4_blocks > 0 {
                let r4_fn = self
                    .ctx
                    .hist_fused_r4_fn
                    .as_ref()
                    .expect("r4_blocks > 0 时 R=4 kernel 必然已加载");
                let rows_view = rows.slice(span.offset..span.offset + r4_rows);
                let cfg = LaunchConfig {
                    grid_dim: (r4_blocks, 1, 1),
                    block_dim: (threads_per_block, 1, 1),
                    shared_mem_bytes,
                };
                let mut launch = stream.launch_builder(r4_fn);
                launch
                    .arg(&bins_view)
                    .arg(gpair)
                    .arg(&rows_view)
                    .arg(&n_rows_u32)
                    .arg(&n_feats_u32)
                    .arg(&offsets_view)
                    .arg(&mut out_view)
                    .arg(&max_tile_bins);
                unsafe { launch.launch(cfg) }.context("launch fused histogram R=4")?;
                self.timing.kernel_calls += 1;
            }
            // 余下的行(含 R=1 时的全部行)走原来的一行/线程融合 kernel。
            let rest_begin = r4_rows;
            let rest_len = n_selected - r4_rows;
            let full_rows = rest_begin + rest_len / threads_per_block as usize
                * threads_per_block as usize;
            let full_blocks = u32::try_from((full_rows - rest_begin) / threads_per_block as usize)
                .context("CUDA histogram grid 超过 u32")?;
            let tail_rows = u32::try_from(n_selected - full_rows).context("tail 超过 u32")?;
            if full_blocks > 0 {
                let rows_view = rows.slice(span.offset + rest_begin..span.offset + full_rows);
                let cfg = LaunchConfig {
                    grid_dim: (full_blocks, 1, 1),
                    block_dim: (threads_per_block, 1, 1),
                    shared_mem_bytes,
                };
                let mut launch = stream.launch_builder(fused_fn);
                launch
                    .arg(&bins_view)
                    .arg(gpair)
                    .arg(&rows_view)
                    .arg(&n_rows_u32)
                    .arg(&n_feats_u32)
                    .arg(&offsets_view)
                    .arg(&mut out_view)
                    .arg(&max_tile_bins);
                unsafe { launch.launch(cfg) }.context("launch fused histogram full blocks")?;
                self.timing.kernel_calls += 1;
            }
            if tail_rows > 0 {
                let rows_view = rows.slice(span.offset + full_rows..span.offset + n_selected);
                let cfg = LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (tail_rows, 1, 1),
                    shared_mem_bytes,
                };
                let mut launch = stream.launch_builder(fused_fn);
                launch
                    .arg(&bins_view)
                    .arg(gpair)
                    .arg(&rows_view)
                    .arg(&n_rows_u32)
                    .arg(&n_feats_u32)
                    .arg(&offsets_view)
                    .arg(&mut out_view)
                    .arg(&max_tile_bins);
                unsafe { launch.launch(cfg) }.context("launch fused histogram tail")?;
                self.timing.kernel_calls += 1;
            }
            self.feature_tiles = 1;
        } else {

        let mut feature_tiles = 0usize;
        let mut feat_begin = 0usize;
        while feat_begin < self.n_feats {
            let hist_begin = self.feat_offsets[feat_begin] as usize;
            let mut feat_end = feat_begin;
            while feat_end < self.n_feats
                && self.feat_offsets[feat_end + 1] as usize - hist_begin
                    <= self.ctx.max_bins_per_tile
            {
                feat_end += 1;
            }
            if feat_end == feat_begin {
                let bins = self.feat_offsets[feat_begin + 1] - self.feat_offsets[feat_begin];
                bail!(
                    "特征 {feat_begin} 的 {bins} 个 bin 超过 shared-memory tile 上限 {}",
                    self.ctx.max_bins_per_tile
                );
            }
            let tile_bins = self.feat_offsets[feat_end] as usize - hist_begin;
            let shared_mem_bytes = u32::try_from(tile_bins * bytes_per_bin)
                .context("shared-memory tile 大小超过 u32")?;
            let tile_n_feats = (feat_end - feat_begin) as u32;

            let slot = self.slot.as_mut().expect("histogram slot lease missing");
            let GpuHistBuffers {
                bins,
                offsets,
                hist_out,
                start_event,
                end_event,
                ..
            } = &mut **slot;
            let GpuBuffers { gpair, rows, .. } = &*core;
            let bins_view = bins.slice(feat_begin * self.ctx.n_rows..feat_end * self.ctx.n_rows);
            let offsets_view = offsets.slice(feat_begin..=feat_end);
            let mut out_view = hist_out.slice_mut(..self.n_hist * 2);

            if full_blocks > 0 {
                let rows_view = rows.slice(span.offset..span.offset + full_rows);
                let cfg = LaunchConfig {
                    grid_dim: (full_blocks, 1, 1),
                    block_dim: (threads_per_block, 1, 1),
                    shared_mem_bytes,
                };
                let mut launch = stream.launch_builder(&self.ctx.hist_fn);
                launch
                    .arg(&bins_view)
                    .arg(gpair)
                    .arg(&rows_view)
                    .arg(&n_rows_u32)
                    .arg(&tile_n_feats)
                    .arg(&offsets_view)
                    .arg(&mut out_view);
                // 参数顺序和 histogram.cu 的七项 C ABI 逐项对应;所有 device
                // view 活到 D2H 完成,device row span 已在入口验过范围,
                // feature tile 保证动态 shared memory 不越 driver 上限。
                if self.ctx.bucket_profile {
                    start_event
                        .record(&stream)
                        .context("记录 histogram bucket 起始 event")?;
                }
                unsafe { launch.launch(cfg) }.context("launch hist_build_shared full blocks")?;
                if self.ctx.bucket_profile {
                    end_event
                        .record(&stream)
                        .context("记录 histogram bucket 结束 event")?;
                    let ms = start_event
                        .elapsed_ms(end_event)
                        .context("计算 histogram bucket kernel 时间")?;
                    self.timing.kernel_ms += ms;
                    eprintln!(
                        "FB_HIST_BUCKET,depth={depth},node={node_index},node_rows={n_selected},kind=full,tile={feature_tiles},node_tiles={feature_tile_count},tile_features={tile_n_feats},grid_blocks={full_blocks},launch_rows={full_rows},shared_bytes={shared_mem_bytes},kernel_ms={ms:.6}"
                    );
                }
                self.timing.kernel_calls += 1;
            }
            if tail_rows > 0 {
                let rows_view = rows.slice(span.offset + full_rows..span.offset + n_selected);
                let cfg = LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (tail_rows, 1, 1),
                    shared_mem_bytes,
                };
                let mut launch = stream.launch_builder(&self.ctx.hist_fn);
                launch
                    .arg(&bins_view)
                    .arg(gpair)
                    .arg(&rows_view)
                    .arg(&n_rows_u32)
                    .arg(&tile_n_feats)
                    .arg(&offsets_view)
                    .arg(&mut out_view);
                if self.ctx.bucket_profile {
                    start_event
                        .record(&stream)
                        .context("记录 histogram tail bucket 起始 event")?;
                }
                unsafe { launch.launch(cfg) }.context("launch hist_build_shared tail")?;
                if self.ctx.bucket_profile {
                    end_event
                        .record(&stream)
                        .context("计算 histogram tail bucket 结束 event")?;
                    let ms = start_event
                        .elapsed_ms(end_event)
                        .context("计算 histogram tail bucket kernel 时间")?;
                    self.timing.kernel_ms += ms;
                    eprintln!(
                        "FB_HIST_BUCKET,depth={depth},node={node_index},node_rows={n_selected},kind=tail,tile={feature_tiles},node_tiles={feature_tile_count},tile_features={tile_n_feats},grid_blocks=1,launch_rows={tail_rows},shared_bytes={shared_mem_bytes},kernel_ms={ms:.6}"
                    );
                }
                self.timing.kernel_calls += 1;
            }
            feat_begin = feat_end;
            feature_tiles += 1;
        }
        self.feature_tiles = feature_tiles;
        }
        drop(core);

        if self.ctx.timing && !self.ctx.bucket_profile {
            let slot = self.slot.as_ref().expect("histogram slot lease missing");
            slot.end_event
                .record(&stream)
                .context("记录 shared histogram 结束 event")?;
            self.timing.kernel_ms += self
                .slot
                .as_ref()
                .expect("histogram slot lease missing")
                .start_event
                .elapsed_ms(&slot.end_event)
                .context("计算 shared histogram kernel 时间")?;
        }

        let started = std::time::Instant::now();
        {
            let slot = self.slot.as_mut().expect("histogram slot lease missing");
            slot.hist_host.clear();
            slot.hist_host.resize(self.n_hist * 2, 0);
            let GpuHistBuffers {
                hist_out,
                hist_host,
                ..
            } = &mut **slot;
            let view = hist_out.slice(..hist_host.len());
            stream
                .memcpy_dtoh(&view, hist_host.as_mut_slice())
                .context("histogram D2H")?;
        }
        // Host 枚举马上消费结果；只等本 slot，不阻塞另一条 stream。
        stream.synchronize().context("等待 histogram D2H")?;
        if self.ctx.timing {
            self.timing.d2h_ms += started.elapsed().as_secs_f32() * 1e3;
        }
        self.timing.d2h_calls += 1;
        self.timing.d2h_bytes += (self.n_hist * std::mem::size_of::<GradPairFixed>()) as u64;
        let slot = self.slot.as_ref().expect("histogram slot lease missing");
        for (dst, pair) in out.iter_mut().zip(slot.hist_host.as_chunks::<2>().0) {
            dst.grad = pair[0];
            dst.hess = pair[1];
        }
        Ok(())
    }

    /// 本层的多个 `Accumulate` 节点合并进**一次** launch。
    ///
    /// 语义和「对每个 span 依次调用 [`Self::histogram_device_at_depth`]」
    /// **逐位相同**:同样的行、同样的 bin、同样的定点加法。定点加法按模
    /// 2^64 与顺序无关,所以合并只改变 block 的调度顺序。
    ///
    /// 消掉的是**每节点一次 host 往返**(launch → D2H → sync → 枚举 →
    /// 再 launch)。审计实测那条链每轮占 3.9 ms GPU 空闲,其中只有 11%
    /// 是 memcpy。
    ///
    /// 调用方保证 `spans.len() == outs.len()`,且不超过 `hist_nodes_per_batch`。
    /// Column-major consumes a compact logical feature list; row-major keeps
    /// its scalar mask ABI while retaining the full output shape.
    pub fn histogram_device_batch(
        &mut self,
        spans: &[GpuRowSpan],
        outs: &mut [&mut Vec<GradPairFixed>],
    ) -> Result<()> {
        if spans.len() != outs.len() {
            bail!("histogram batch:spans {} 与 outs {} 不等长", spans.len(), outs.len());
        }
        // 行主序 kernel 每线程一行(整行 32 B 一次向量取数进寄存器),
        // 列主序 kernel 每线程四行。**每 block 覆盖的行数因此不同**,
        // blk_ptr 必须按各自的 rows_per_block 算,否则批内节点边界会错位。
        let row_major = self.ctx.rm_hist_fn.is_some();
        let batched_fn = if row_major {
            self.ctx.rm_hist_fn.as_ref().expect("row_major 蕴含 kernel 已加载")
        } else {
            self
                .ctx
                .hist_batched_fn
                .as_ref()
                .context("histogram batch 需要 hist_nodes_per_batch > 1")?
        };
        if spans.len() > self.ctx.hist_nodes_per_batch {
            bail!(
                "histogram batch:{} 个节点超过容量 {}",
                spans.len(),
                self.ctx.hist_nodes_per_batch
            );
        }
        for out in outs.iter_mut() {
            out.clear();
            out.resize(self.n_hist, GradPairFixed::default());
        }
        for span in spans {
            if span.offset.saturating_add(span.len) > self.ctx.n_rows {
                bail!("histogram device row span 越界:{span:?}");
            }
        }
        let n_nodes = spans.len();
        if self.n_hist == 0 || n_nodes == 0 {
            return Ok(());
        }

        let threads_per_block = self.ctx.config.threads_per_block;
        // 批 kernel 固定走 R=4 的几何(和 `..._fused_r4` 完全一致),
        // 余数由 kernel 内的 `ok[r]` 边界判断吃掉 —— 所以不再需要
        // 单独的 remainder / tail launch,那 66 个装不满 SM 的小 launch
        // 一并消失。
        // 列主序 kernel 每线程 4 行;行主序 kernel 两条 lane 一行,
        // 所以每 block 只覆盖 threads/2 行。
        // 行主序 R=4 和列主序同一个几何。
        let rows_per_block = threads_per_block as usize * 4;
        let mut blk_ptr: Vec<u32> = Vec::with_capacity(n_nodes + 1);
        let mut node_off: Vec<u32> = Vec::with_capacity(n_nodes);
        let mut node_len: Vec<u32> = Vec::with_capacity(n_nodes);
        let mut total_blocks: u32 = 0;
        blk_ptr.push(0);
        for span in spans {
            let blocks = u32::try_from(span.len.div_ceil(rows_per_block))
                .context("histogram batch grid 超过 u32")?;
            total_blocks = total_blocks
                .checked_add(blocks)
                .context("histogram batch grid 溢出 u32")?;
            blk_ptr.push(total_blocks);
            node_off.push(u32::try_from(span.offset).context("row span offset 超过 u32")?);
            node_len.push(u32::try_from(span.len).context("row span len 超过 u32")?);
        }
        // 整批都空(可能发生在极端的行分布上)时没有任何 block 要跑,
        // 但 outs 已经清成零直方图,语义正确。
        if total_blocks == 0 {
            return Ok(());
        }

        let n_rows_u32 = self.ctx.n_rows as u32;
        let bytes_per_bin = std::mem::size_of::<GradPairFixed>();
        let hist_stride = self.n_hist * 2;
        let stream = self
            .slot
            .as_ref()
            .expect("histogram slot lease missing")
            .stream
            .clone();

        {
            let slot = self.slot.as_mut().expect("histogram slot lease missing");
            let mut view = slot.hist_out.slice_mut(..hist_stride * n_nodes);
            stream.memset_zeros(&mut view).context("清零批直方图 buffer")?;
        }

        if self.ctx.timing {
            self.slot
                .as_ref()
                .expect("histogram slot lease missing")
                .start_event
                .record(&stream)
                .context("记录批 histogram 起始 event")?;
        }

        // shared 仍然按这个列块**实际用到的最大 tile** 要,和融合路径逐字
        // 相同的贪心切分 —— 几何不变是这次实验的前提。
        let mut max_tile_slots = 0usize;
        let mut probe = 0usize;
        while probe < self.n_feats {
            let hist_begin = self.feat_offsets[probe] as usize;
            let mut end = probe;
            while end < self.n_feats
                && self.feat_offsets[end + 1] as usize - hist_begin <= self.ctx.max_bins_per_tile
            {
                end += 1;
            }
            debug_assert!(end > probe, "单个特征的 bin 数超过 tile 上限");
            max_tile_slots = max_tile_slots.max(self.feat_offsets[end] as usize - hist_begin);
            probe = end;
        }
        let shared_mem_bytes = u32::try_from(max_tile_slots * bytes_per_bin)
            .context("shared-memory tile 大小超过 u32")?;
        let n_feats_u32 = u32::try_from(self.n_feats).context("列块特征数超过 u32")?;
        let max_tile_bins =
            u32::try_from(self.ctx.max_bins_per_tile).context("tile bin 上限超过 u32")?;
        let n_nodes_u32 = u32::try_from(n_nodes).context("批节点数超过 u32")?;
        let hist_stride_u32 = u32::try_from(hist_stride).context("直方图 stride 超过 u32")?;

        let core = self.ctx.lock()?;
        {
            let slot = self.slot.as_mut().expect("histogram slot lease missing");
            // [blk_ptr(n+1) | node_off(n) | node_len(n)],一次 H2D 打包上传。
            let mut idx_host: Vec<u32> = Vec::with_capacity(n_nodes * 3 + 1);
            idx_host.extend_from_slice(&blk_ptr);
            idx_host.extend_from_slice(&node_off);
            idx_host.extend_from_slice(&node_len);
            let mut idx_view = slot.batch_idx.slice_mut(..idx_host.len());
            stream
                .memcpy_htod(&idx_host, &mut idx_view)
                .context("批 block->node 索引 H2D")?;

            let GpuHistBuffers {
                bins,
                offsets,
                feature_select,
                hist_out,
                batch_idx,
                ..
            } = &mut **slot;
            let GpuBuffers { gpair, rows, .. } = &*core;
            let bins_view = bins.slice(0..self.physical_n_feats * self.ctx.n_rows);
            let offsets_view = offsets.slice(0..=self.n_feats);
            let feature_select_view = feature_select.slice(0..self.n_feats.max(1));
            let blk_ptr_view = batch_idx.slice(0..n_nodes + 1);
            let node_off_view = batch_idx.slice(n_nodes + 1..n_nodes * 2 + 1);
            let node_len_view = batch_idx.slice(n_nodes * 2 + 1..n_nodes * 3 + 1);
            let rows_view = rows.slice(0..self.ctx.n_rows);
            let mut out_view = hist_out.slice_mut(..hist_stride * n_nodes);

            let cfg = LaunchConfig {
                grid_dim: (total_blocks, 1, 1),
                block_dim: (threads_per_block, 1, 1),
                shared_mem_bytes,
            };
            let selection_active_u32 = u32::from(self.selection_active);
            let mut launch = stream.launch_builder(batched_fn);
            launch
                .arg(&bins_view)
                .arg(gpair)
                .arg(&rows_view)
                .arg(&n_rows_u32)
                .arg(&n_feats_u32)
                .arg(&offsets_view)
                .arg(&mut out_view)
                .arg(&max_tile_bins)
                .arg(&blk_ptr_view)
                .arg(&node_off_view)
                .arg(&node_len_view)
                .arg(&n_nodes_u32)
                .arg(&hist_stride_u32);
            if row_major {
                launch.arg(&self.row_major_feat_mask);
            } else {
                launch.arg(&feature_select_view).arg(&selection_active_u32);
            }
            unsafe { launch.launch(cfg) }.context("launch batched histogram")?;
            self.timing.kernel_calls += 1;
        }
        drop(core);

        if self.ctx.timing {
            let slot = self.slot.as_ref().expect("histogram slot lease missing");
            slot.end_event.record(&stream).context("记录批 histogram 结束 event")?;
            stream.synchronize().context("等待批 histogram kernel")?;
            self.timing.kernel_ms += slot
                .start_event
                .elapsed_ms(&slot.end_event)
                .context("计算批 histogram kernel 时间")?;
        }

        // **整批一次 D2H**。原来是每个节点一次,而且每次都要 host 等
        // scatter 退休 —— 那正是审计里 3.47 ms host 延迟的来源。
        let started = std::time::Instant::now();
        {
            let slot = self.slot.as_mut().expect("histogram slot lease missing");
            slot.hist_host.clear();
            slot.hist_host.resize(hist_stride * n_nodes, 0);
            let GpuHistBuffers { hist_out, hist_host, .. } = &mut **slot;
            let view = hist_out.slice(..hist_host.len());
            stream
                .memcpy_dtoh(&view, hist_host.as_mut_slice())
                .context("批 histogram D2H")?;
        }
        stream.synchronize().context("等待批 histogram D2H")?;
        if self.ctx.timing {
            self.timing.d2h_ms += started.elapsed().as_secs_f32() * 1e3;
        }
        self.timing.d2h_calls += 1;
        self.timing.d2h_bytes += (self.n_hist * n_nodes * bytes_per_bin) as u64;

        let slot = self.slot.as_ref().expect("histogram slot lease missing");
        for (node, out) in outs.iter_mut().enumerate() {
            let chunk = &slot.hist_host[node * hist_stride..(node + 1) * hist_stride];
            for (dst, pair) in out.iter_mut().zip(chunk.as_chunks::<2>().0) {
                dst.grad = pair[0];
                dst.hess = pair[1];
            }
        }
        Ok(())
    }

    /// 本 session 一次最多能合并多少个节点。1 表示 batching 关闭。
    pub fn hist_nodes_batch_capacity(&self) -> usize {
        if self.ctx.hist_batched_fn.is_some() || self.ctx.rm_hist_fn.is_some() {
            self.ctx.hist_nodes_per_batch
        } else {
            1
        }
    }

    /// 是否必须走 batched 入口。
    ///
    /// ⚠️ 行主序 kernel **只有 batched 这一个入口** —— 逐节点的
    /// `histogram_device_at_depth` 仍然按列主序寻址 bins。所以哪怕
    /// `hist_nodes_per_batch == 1`(批里只有一个节点),行主序也必须走
    /// batched 路径,否则会拿行主序的字节按列主序解释,**静默算错**。
    /// 这个 bug 被 `gpu_multi_node_batching_crosses_the_batch_boundary`
    /// 抓到过一次。
    pub fn requires_batched_hist(&self) -> bool {
        self.ctx.rm_hist_fn.is_some() || self.selection_active
    }

    pub fn timing(&self) -> GpuSegmentTiming {
        self.timing
    }

    pub fn feature_tiles(&self) -> usize {
        self.feature_tiles
    }
}

/// 一列已经在显存里,本层所有选中它的节点共用。
pub struct GpuPartitionSession<'a> {
    ctx: &'a GpuTrainCtx,
    buf: MutexGuard<'a, GpuBuffers>,
    /// resident 命中时借住那个 slot,直接拿它 bins 里的那一列;
    /// `usize` 是该特征在块内的局部下标。None 表示走老的 `part_col` 上传。
    resident: Option<(MutexGuard<'a, GpuHistBuffers>, usize)>,
    timing: GpuSegmentTiming,
}

impl GpuPartitionSession<'_> {
    /// 稳定地把 device row span 分到下一层 arena；只把左右计数传回 host。
    pub fn partition(
        &mut self,
        span: GpuRowSpan,
        split_bin: u32,
        missing_left: bool,
    ) -> Result<PartitionToken> {
        if span.len == 0 {
            self.buf.part_pending.push(PendingPartition {
                out_offset: span.offset,
                n_selected: 0,
            });
            return Ok(self.buf.part_pending.len() - 1);
        }
        if span.offset.saturating_add(span.len) > self.ctx.n_rows {
            bail!("partition device row span 越界:{span:?}");
        }
        let threads = self.ctx.partition_threads;
        let n_selected = span.len as u32;
        let n_blocks = n_selected.div_ceil(threads);
        let n_rows_u32 = self.ctx.n_rows as u32;
        let n_rows_usize = self.ctx.n_rows;

        if self.ctx.timing {
            self.ctx
                .start_event
                .record(&self.ctx.stream)
                .context("记录 partition 起始 event")?;
        }
        let missing_left_u32 = missing_left as u32;
        // 本节点在整层 totals 收集区里的槽位。
        let slot = self.buf.part_pending.len();
        anyhow::ensure!(
            slot < MAX_LEVEL_NODES,
            "一层的 partition 节点数超过 {MAX_LEVEL_NODES}"
        );
        let slot_u32 = slot as u32;
        {
            let GpuBuffers {
                part_col,
                rows: rows_dev,
                part_flags,
                part_block_counts,
                ..
            } = &mut *self.buf;
            // resident 命中就直接看常驻 bins 里的那一列,不用 `part_col`。
            // 两条路给出的都是长度 n_rows 的 `CudaView<u8>`,kernel 侧
            // 完全看不出区别 —— 契约仍是 `const uint8_t* col` + n_rows stride。
            // 行主序:没有连续的一列,但也**不需要**物化一列 ——
            // 直接按 active row 算地址 `row * n_feats + feat`。
            // 代价只随 active row 走,不随 n_rows 走。
            let rm = self.ctx.rm_count_fn.is_some();
            let (col_view, rm_feat) = match self.resident.as_ref() {
                Some((slot, local_feat)) if rm => {
                    // 用**这个块自己**的宽度,不是 buffer 容量反推的宽度。
                    let n_feats = slot.resident_feats;
                    debug_assert!(n_feats > 0 && *local_feat < n_feats);
                    (slot.bins.slice(..n_rows_usize * n_feats), Some((n_feats, *local_feat)))
                }
                Some((slot, local_feat)) => (
                    slot.bins.slice(local_feat * n_rows_usize..(local_feat + 1) * n_rows_usize),
                    None,
                ),
                None => (part_col.slice(..n_rows_usize), None),
            };
            let rows_view = rows_dev.slice(span.offset..span.offset + span.len);
            // flags 按**绝对 arena 位置**切,而不是从 0 开始 —— 一层里各
            // 节点的 span 互不相交,这样它们的 flags 也互不覆盖,可以一直
            // 活到 scatter。以前是每个节点都用前缀,所以必须立刻消费掉。
            let mut flags_view = part_flags.slice_mut(..span.len);
            let mut counts_view = part_block_counts.slice_mut(..n_blocks as usize);
            let cfg = LaunchConfig {
                grid_dim: (n_blocks, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: threads * 4,
            };
            match rm_feat {
                Some((n_feats, feat)) => {
                    let n_feats_u32 = u32::try_from(n_feats).context("列块特征数超过 u32")?;
                    let feat_u32 = u32::try_from(feat).context("列下标超过 u32")?;
                    let rm_fn = self
                        .ctx
                        .rm_count_fn
                        .as_ref()
                        .expect("rm_feat 蕴含 kernel 已加载");
                    let mut launch = self.ctx.stream.launch_builder(rm_fn);
                    launch
                        .arg(&col_view)
                        .arg(&n_feats_u32)
                        .arg(&feat_u32)
                        .arg(&rows_view)
                        .arg(&n_selected)
                        .arg(&split_bin)
                        .arg(&missing_left_u32)
                        .arg(&mut flags_view)
                        .arg(&mut counts_view);
                    unsafe { launch.launch(cfg) }.context("launch partition_count_rm")?;
                }
                None => {
                    let mut launch = self.ctx.stream.launch_builder(&self.ctx.count_fn);
                    launch
                        .arg(&col_view)
                        .arg(&rows_view)
                        .arg(&n_rows_u32)
                        .arg(&n_selected)
                        .arg(&split_bin)
                        .arg(&missing_left_u32)
                        .arg(&mut flags_view)
                        .arg(&mut counts_view);
                    // 行下标已在上面验过范围,列 buffer 就是本层选中的整列,
                    // 三个输出 view 长度都按 n_selected / n_blocks 裁好。
                    unsafe { launch.launch(cfg) }.context("launch partition_count")?;
                }
            }
            self.timing.kernel_calls += 1;
        }
        {
            let GpuBuffers {
                part_block_counts,
                part_left_offsets,
                part_right_offsets,
                part_totals: totals,
                ..
            } = &mut *self.buf;
            let counts_view = part_block_counts.slice(..n_blocks as usize);
            let mut left_view = part_left_offsets.slice_mut(..n_blocks as usize + 1);
            let mut right_view = part_right_offsets.slice_mut(..n_blocks as usize + 1);
            // 单 block、多线程的分块 scan(kernel 里有推导)。以前是一个线程
            // 串行走完所有 block —— root 节点上 20,509 次迭代。
            let scan_threads: u32 = 1024;
            let cfg = LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (scan_threads, 1, 1),
                shared_mem_bytes: scan_threads * 2 * std::mem::size_of::<u32>() as u32,
            };
            let mut launch = self.ctx.stream.launch_builder(&self.ctx.scan_fn);
            launch
                .arg(&counts_view)
                .arg(&mut left_view)
                .arg(&mut right_view)
                .arg(&n_blocks)
                .arg(&n_selected)
                .arg(&threads)
                .arg(totals)
                .arg(&slot_u32);
            unsafe { launch.launch(cfg) }.context("launch partition_scan_blocks")?;
            self.timing.kernel_calls += 1;
        }

        // **这里以前有两次阻塞 D2H**:scatter 需要 total_left 作为标量,
        // 而它只能从 device 读回来。实测那是 126 次/轮、23.8 ms/轮
        // (HIGGS,profile 关闭),其中绝大部分是 pipeline drain,
        // 不是那 1 KB 的传输本身。
        //
        // 现在 scatter 自己从 `left_offsets[gridDim.x]` 读这个值,
        // 所以 launch 阶段一次同步都不需要;host 要的左右行数攒到整层
        // 结束由 `resolve_partitions()` 一次取回。
        let next_offset = self.buf.next_rows_cursor;
        anyhow::ensure!(
            next_offset.saturating_add(span.len) <= self.ctx.n_rows,
            "下一层 row arena 溢出"
        );
        {
            let GpuBuffers {
                rows: rows_dev,
                part_flags,
                part_left_offsets,
                part_right_offsets,
                part_right_out,
                ..
            } = &mut *self.buf;
            let rows_view = rows_dev.slice(span.offset..span.offset + span.len);
            let flags_view = part_flags.slice(..span.len);
            let left_off_view = part_left_offsets.slice(..n_blocks as usize + 1);
            let right_off_view = part_right_offsets.slice(..n_blocks as usize + 1);
            let mut next_view = part_right_out.slice_mut(next_offset..next_offset + span.len);
            let cfg = LaunchConfig {
                grid_dim: (n_blocks, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: threads * 2 * 4,
            };
            let mut launch = self.ctx.stream.launch_builder(&self.ctx.scatter_layer_fn);
            launch
                .arg(&rows_view)
                .arg(&flags_view)
                .arg(&n_selected)
                .arg(&left_off_view)
                .arg(&right_off_view)
                .arg(&mut next_view);
            unsafe { launch.launch(cfg) }.context("launch partition_scatter_layer")?;
            self.timing.kernel_calls += 1;
        }
        self.buf.next_rows_cursor += span.len;
        if self.ctx.timing {
            self.ctx
                .end_event
                .record(&self.ctx.stream)
                .context("记录 partition 结束 event")?;
            self.timing.kernel_ms += self
                .ctx
                .start_event
                .elapsed_ms(&self.ctx.end_event)
                .context("计算 partition kernel 时间")?;
        }

        // 只记账,不同步。左右各多少行由 `resolve_partitions()` 整层取回。
        self.buf.part_pending.push(PendingPartition {
            out_offset: next_offset,
            n_selected: span.len,
        });
        Ok(self.buf.part_pending.len() - 1)
    }

    pub fn timing(&self) -> GpuSegmentTiming {
        self.timing
    }
}
