//! CUDA 后端的最小正确性版本。
//!
//! PTX 预编译进 crate,所以默认 CPU 构建和 `--features cuda` 的编译都不
//! 要求本机有 nvcc。只有实际调用时才需要 NVIDIA driver 和 GPU。

use anyhow::{bail, Context, Result};
use cudarc::{
    driver::{sys, CudaContext, LaunchConfig, PushKernelArg},
    nvrtc::Ptx,
};
use std::{collections::HashMap, sync::{Arc, Mutex, OnceLock}};

use crate::types::{Bin, GradPairFixed, RowId, MISSING_BIN};

mod inference;
mod train_ctx;
pub use inference::GpuInferenceModel;
pub use train_ctx::{
    GpuBlockSession, GpuHostMemoryBudget, GpuMemoryBudget, GpuPartitionSession, GpuRowSpan,
    GpuSegmentTiming, GpuTrainCtx,
};

const HISTOGRAM_PTX: &str = include_str!("histogram.ptx");

struct DeviceState {
    ctx: Arc<CudaContext>,
    module: Arc<cudarc::driver::CudaModule>,
}

static DEVICE_STATES: OnceLock<Mutex<HashMap<usize, Arc<DeviceState>>>> = OnceLock::new();

fn device_state(device_ordinal: usize) -> Result<Arc<DeviceState>> {
    let cache = DEVICE_STATES.get_or_init(|| Mutex::new(HashMap::new()));
    let mut states = cache.lock().map_err(|_| anyhow::anyhow!("CUDA device cache poisoned"))?;
    if let Some(state) = states.get(&device_ordinal) {
        return Ok(state.clone());
    }
    let ctx = CudaContext::new(device_ordinal).context("创建 CUDA context")?;
    let module = ctx
        .load_module(Ptx::from_src(HISTOGRAM_PTX))
        .context("加载预编译 histogram PTX")?;
    let state = Arc::new(DeviceState { ctx, module });
    states.insert(device_ordinal, state.clone());
    Ok(state)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SharedHistogramLayout {
    /// 一个 16-byte 交错 GradPairFixed 对应一个 logical bin。
    Interleaved16,
    /// block-local shared memory 分开存放 8-byte grad / hess 数组。
    Split8,
    /// block-local shared memory 分成四个 u32 平面(grad_lo/hi、hess_lo/hi),
    /// 每个平面按 slot 单位步长索引,从而用满 32 个 bank。字节数与
    /// `Interleaved16` 相同。
    Planar32,
}

impl SharedHistogramLayout {
    pub const ALL: [Self; 3] = [Self::Interleaved16, Self::Split8, Self::Planar32];

    pub const fn kernel_name(self) -> &'static str {
        match self {
            Self::Interleaved16 => "hist_build_shared",
            Self::Split8 => "hist_build_shared_split",
            Self::Planar32 => "hist_build_shared_planar32",
        }
    }

    pub const fn label(self) -> &'static str {
        match self {
            Self::Interleaved16 => "interleaved16",
            Self::Split8 => "split8",
            Self::Planar32 => "planar32",
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SharedHistogramConfig {
    /// Whether to emit the concise public runtime path summary.
    pub runtime_log: bool,
    /// CUDA threads per row block.只用于测量阶段扫 occupancy,不改变数值语义。
    pub threads_per_block: u32,
    /// 限制 feature tile 的动态 shared memory。None 使用设备默认上限。
    pub shared_memory_budget_bytes: Option<usize>,
    /// shared-memory 内部布局。它独立于 global GradPairFixed 的 FFI 表示。
    pub layout: SharedHistogramLayout,
    /// histogram stream 数(1 或 2)。`None` = 默认 1；环境变量只作为
    /// benchmark/debug override。
    ///
    /// 用户通过 `TrainParams::hist_streams` 显式给值时,这里就是 `Some(n)`,
    /// **planner 和环境变量都不得覆盖它**。
    pub hist_streams: Option<usize>,
    /// 定点量化是否在 device 上做。`None` = 默认开。见
    /// `TrainParams::device_quantize`:它改变内存计划(+8 B/行),所以是参数。
    pub device_quantize: Option<bool>,
    /// `TrainParams::hist_nodes_per_batch`:一次 launch 合并多少个 Accumulate 节点。
    /// 批内每个节点要一份独立直方图切片,**改变内存计划**,所以是参数。
    pub hist_nodes_per_batch: Option<usize>,
    /// `TrainParams::gpu_math`。`Fast` 会让 prediction 和 label 常驻 device
    /// (+8 B/行),**改变内存计划**。
    pub gpu_math: crate::types::GpuMath,
    /// 规划器实际使用的显存预算(字节)。`None` = 没有规划器参与
    /// (显式块宽 / CPU),此时没有"生效预算"可言。
    ///
    /// 只用于**回显**:让 `GPU_PATH` 直接说出约束了这次规划的那个数,
    /// 而不是让人去 plan report 里把它认成 `free_vram`。
    pub gpu_memory_budget: Option<usize>,
    /// 常驻物理列块数。`None` = 交给显存规划器 / env / 默认 0。
    ///
    /// **规划器解出来的值是权威的**:它是在"块宽 + stream 数 + 常驻块数"
    /// 同一个内存模型里一起解出来的,运行时不得自己再改一个,
    /// 否则预算就和实际分配对不上。
    pub resident_blocks: Option<usize>,
}

impl Default for SharedHistogramConfig {
    fn default() -> Self {
        Self {
            runtime_log: true,
            // 在参考 sm_86 occupancy profile 上,32 KiB 比 48 KiB 能保留更多
            // 512-thread resident blocks;基准 sweep 因此选择这个保守默认值。
            threads_per_block: 512,
            shared_memory_budget_bytes: Some(32 * 1024),
            layout: SharedHistogramLayout::Planar32,
            hist_streams: None,
            device_quantize: None,
            hist_nodes_per_batch: None,
            gpu_memory_budget: None,
            resident_blocks: None,
            gpu_math: crate::types::GpuMath::Exact,
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct SharedHistogramProfile {
    /// 只含 kernel launches,不含 context/PTX/H2D/D2H/host validation。
    pub kernel_ms: f32,
    /// 当前调用中量化 bins、gpair、row_idx、offsets 的 H2D wall time。
    pub h2d_ms: f32,
    /// 当前调用中 histogram 输出的 D2H wall time。
    pub d2h_ms: f32,
    pub feature_tiles: usize,
    pub row_launches_per_feature_tile: usize,
    pub threads_per_block: u32,
    pub shared_memory_budget_bytes: usize,
    pub active_blocks_per_sm: u32,
    pub theoretical_occupancy: f32,
    pub layout: SharedHistogramLayout,
}

#[derive(Clone, Copy, Debug)]
pub struct PartitionProfile {
    /// 只含三个 partition kernel 的 device event 时间。
    pub kernel_ms: f32,
    pub threads_per_block: u32,
    pub blocks: u32,
}

/// 在 GPU 上按一个量化特征稳定地把行分到左右两边。
///
/// 三个 kernel 分别完成方向分类/块计数、确定性 exclusive scan 和稳定
/// scatter。返回的两个数组保持输入 `rows` 的相对顺序,不改动 CPU partition
/// 或训练循环;这是当前阶段的独立 characterization 入口。
#[allow(clippy::too_many_arguments)]
pub fn partition_rows_gpu(
    device_ordinal: usize,
    block_data: &[Bin],
    block_n_rows: usize,
    feature: usize,
    rows: &[RowId],
    split_bin: u32,
    missing_left: bool,
    threads_per_block: u32,
) -> Result<(Vec<RowId>, Vec<RowId>)> {
    Ok(partition_rows_gpu_with_profile(
        device_ordinal,
        block_data,
        block_n_rows,
        feature,
        rows,
        split_bin,
        missing_left,
        threads_per_block,
    )?
    .0)
}

#[allow(clippy::too_many_arguments)]
pub fn partition_rows_gpu_with_profile(
    device_ordinal: usize,
    block_data: &[Bin],
    block_n_rows: usize,
    feature: usize,
    rows: &[RowId],
    split_bin: u32,
    missing_left: bool,
    threads_per_block: u32,
) -> Result<((Vec<RowId>, Vec<RowId>), PartitionProfile)> {
    if block_n_rows == 0 || feature >= block_data.len().checked_div(block_n_rows).unwrap_or(0)
    {
        bail!("partition GPU 的 feature 不在 bins 列范围内");
    }
    if threads_per_block == 0 || threads_per_block > 1024 || !threads_per_block.is_power_of_two() {
        bail!("partition threads_per_block 必须是 1..=1024 的 2 次幂");
    }
    if rows.is_empty() {
        return Ok(((Vec::new(), Vec::new()), PartitionProfile {
            kernel_ms: 0.0,
            threads_per_block,
            blocks: 0,
        }));
    }
    if rows.iter().any(|&row| row as usize >= block_n_rows) {
        bail!("partition row_idx 含越界行");
    }
    let n_rows_u32 = u32::try_from(block_n_rows).context("partition 物理行数超过 u32")?;
    let n_selected = u32::try_from(rows.len()).context("partition 行数超过 u32")?;
    let n_blocks = (rows.len() as u32).div_ceil(threads_per_block);
    let state = device_state(device_ordinal)?;
    let ctx = &state.ctx;
    let stream = ctx.default_stream();
    let module = &state.module;
    let count_fn = module
        .load_function("partition_count")
        .context("PTX 中找不到 partition_count")?;
    let scan_fn = module
        .load_function("partition_scan_blocks")
        .context("PTX 中找不到 partition_scan_blocks")?;
    let scatter_fn = module
        .load_function("partition_scatter")
        .context("PTX 中找不到 partition_scatter")?;

    let col_dev = stream
        .clone_htod(&block_data[feature * block_n_rows..(feature + 1) * block_n_rows])
        .context("partition bins H2D")?;
    let rows_dev = stream.clone_htod(rows).context("partition rows H2D")?;
    let mut flags_dev = stream
        .alloc_zeros::<u8>(rows.len())
        .context("partition flags allocation")?;
    let mut block_counts_dev = stream
        .alloc_zeros::<u32>(n_blocks as usize)
        .context("partition block counts allocation")?;
    let mut left_offsets_dev = stream
        .alloc_zeros::<u32>(n_blocks as usize + 1)
        .context("partition left offsets allocation")?;
    let mut right_offsets_dev = stream
        .alloc_zeros::<u32>(n_blocks as usize + 1)
        .context("partition right offsets allocation")?;
    let mut left_out_dev = stream
        .alloc_zeros::<u32>(rows.len())
        .context("partition left output allocation")?;
    let mut right_out_dev = stream
        .alloc_zeros::<u32>(rows.len())
        .context("partition right output allocation")?;
    let start = stream
        .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
        .context("记录 partition 起始 event")?;

    let count_config = LaunchConfig {
        grid_dim: (n_blocks, 1, 1),
        block_dim: (threads_per_block, 1, 1),
        shared_mem_bytes: threads_per_block * 4,
    };
    let missing_left_u32 = missing_left as u32;
    let mut launch = stream.launch_builder(&count_fn);
    launch
        .arg(&col_dev)
        .arg(&rows_dev)
        .arg(&n_rows_u32)
        .arg(&n_selected)
        .arg(&split_bin)
        .arg(&missing_left_u32)
        .arg(&mut flags_dev)
        .arg(&mut block_counts_dev);
    unsafe { launch.launch(count_config) }.context("launch partition_count")?;

    // 和训练路径同一个 kernel:单 block、多线程分块 scan。
    let scan_threads: u32 = 1024;
    let scan_config = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (scan_threads, 1, 1),
        shared_mem_bytes: scan_threads * 2 * std::mem::size_of::<u32>() as u32,
    };
    // scan 现在顺带把左右总数写进一个 totals 槽位(训练路径用它把每节点
    // 两次阻塞 D2H 压成整层一次)。独立 characterization 入口不需要那个
    // 优化,但必须满足同一个 kernel 签名,所以给它一个一次性的单槽 buffer。
    let mut totals_dev = stream
        .alloc_zeros::<u32>(2)
        .context("partition totals 分配")?;
    let totals_slot = 0u32;
    let mut launch = stream.launch_builder(&scan_fn);
    launch
        .arg(&block_counts_dev)
        .arg(&mut left_offsets_dev)
        .arg(&mut right_offsets_dev)
        .arg(&n_blocks)
        .arg(&n_selected)
        .arg(&threads_per_block)
        .arg(&mut totals_dev)
        .arg(&totals_slot);
    unsafe { launch.launch(scan_config) }.context("launch partition_scan_blocks")?;

    let scatter_config = LaunchConfig {
        grid_dim: (n_blocks, 1, 1),
        block_dim: (threads_per_block, 1, 1),
        shared_mem_bytes: threads_per_block * 2 * 4,
    };
    // total_left is filled below after the scan; right output starts immediately
    // after the left region, so scatter is launched only after reading offsets.
    let left_offsets = stream
        .clone_dtoh(&left_offsets_dev)
        .context("partition left offsets D2H")?;
    let right_offsets = stream
        .clone_dtoh(&right_offsets_dev)
        .context("partition right offsets D2H")?;
    let total_left = *left_offsets.last().unwrap();
    let mut launch = stream.launch_builder(&scatter_fn);
    launch
        .arg(&rows_dev)
        .arg(&flags_dev)
        .arg(&n_selected)
        .arg(&left_offsets_dev)
        .arg(&right_offsets_dev)
        .arg(&mut left_out_dev)
        .arg(&mut right_out_dev)
        .arg(&total_left);
    unsafe { launch.launch(scatter_config) }.context("launch partition_scatter")?;

    let end = stream
        .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
        .context("记录 partition 结束 event")?;
    let kernel_ms = start.elapsed_ms(&end).context("计算 partition kernel 时间")?;

    let left = stream.clone_dtoh(&left_out_dev).context("partition left D2H")?;
    let right = stream.clone_dtoh(&right_out_dev).context("partition right D2H")?;
    let total_right = *right_offsets.last().unwrap();
    Ok((
        (
            left[..total_left as usize].to_vec(),
            right[total_left as usize..(total_left + total_right) as usize].to_vec(),
        ),
        PartitionProfile {
            kernel_ms,
            threads_per_block,
            blocks: n_blocks,
        },
    ))
}

/// 用最朴素的 global-memory atomic kernel 构建一个节点的直方图。
///
/// 参数语义与 `hist::build` 相同。返回新 Vec 是为了让第一版的显存生命
/// 周期完全留在这一层;训练循环尚未接入 CUDA。
pub fn build_histogram(
    device_ordinal: usize,
    block_data: &[Bin],
    block_n_rows: usize,
    n_feats: usize,
    feat_offsets: &[u32],
    gpair: &[GradPairFixed],
    row_idx: Option<&[RowId]>,
) -> Result<Vec<GradPairFixed>> {
    validate_inputs(
        block_data,
        block_n_rows,
        n_feats,
        feat_offsets,
        gpair,
        row_idx,
    )?;

    let n_hist = feat_offsets[n_feats] as usize;
    let n_selected = row_idx.map_or(block_n_rows, |rows| rows.len());
    if n_hist == 0 || n_selected == 0 {
        return Ok(vec![GradPairFixed::default(); n_hist]);
    }

    let n_rows_u32 = u32::try_from(block_n_rows).context("CUDA histogram 行数超过 u32")?;
    let n_feats_u32 = u32::try_from(n_feats).context("CUDA histogram 特征数超过 u32")?;
    let n_selected_u32 = u32::try_from(n_selected).context("CUDA histogram 子集超过 u32")?;

    // FFI 明确把梯度和输出描述为交错 int64。这里显式摊平,不要求
    // cudarc 理解 Rust struct,也让 host/device 布局契约一眼可见。
    let mut gpair_words = Vec::with_capacity(gpair.len() * 2);
    for pair in gpair {
        gpair_words.push(pair.grad);
        gpair_words.push(pair.hess);
    }

    let state = device_state(device_ordinal)?;
    let ctx = &state.ctx;
    let stream = ctx.default_stream();
    let module = &state.module;
    let function = module
        .load_function("hist_build")
        .context("PTX 中找不到 hist_build")?;

    let bins_dev = stream.clone_htod(block_data).context("bins H2D")?;
    let gpair_dev = stream.clone_htod(&gpair_words).context("gpair H2D")?;
    let rows_dev = match row_idx {
        Some(rows) => stream.clone_htod(rows).context("row_idx H2D")?,
        None => stream.null::<u32>().context("创建 null row_idx")?,
    };
    let offsets_dev = stream
        .clone_htod(feat_offsets)
        .context("feat_offsets H2D")?;
    let mut out_dev = stream
        .alloc_zeros::<i64>(n_hist * 2)
        .context("分配 CUDA histogram")?;

    // 一个 block 只有一个线程,因此 grid.x 精确等于 selected-row 数量。
    // 这是第一版用 launch geometry 携带 subset 长度、同时保持既定 FFI
    // 中 n_rows 为列 stride 的办法;不作为性能实现保留。
    let config = LaunchConfig {
        grid_dim: (n_selected_u32, 1, 1),
        block_dim: (1, 1, 1),
        shared_mem_bytes: 0,
    };
    let mut launch = stream.launch_builder(&function);
    launch
        .arg(&bins_dev)
        .arg(&gpair_dev)
        .arg(&rows_dev)
        .arg(&n_rows_u32)
        .arg(&n_feats_u32)
        .arg(&offsets_dev)
        .arg(&mut out_dev);
    // 参数顺序/类型和 histogram.cu 的 C ABI 已逐项对应;所有 device slice
    // 活到 D2H 完成,且 validate_inputs 排除了 kernel 越界。
    unsafe { launch.launch(config) }.context("launch hist_build")?;

    let words = stream.clone_dtoh(&out_dev).context("histogram D2H")?;
    Ok(words
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| GradPairFixed {
            grad: pair[0],
            hess: pair[1],
        })
        .collect())
}

/// 用 block-local shared memory 构建一个节点的直方图。
///
/// 这是和 [`build_histogram`] 并存的第二个正确性实现:host 根据设备报告的
/// 默认每-block shared-memory 上限切连续 feature tile,每个 CUDA block
/// 先在 shared memory 里累加自己的 row tile,再把整数结果 atomic flush 到
/// global。训练循环仍未接入 CUDA。
pub fn build_histogram_shared(
    device_ordinal: usize,
    block_data: &[Bin],
    block_n_rows: usize,
    n_feats: usize,
    feat_offsets: &[u32],
    gpair: &[GradPairFixed],
    row_idx: Option<&[RowId]>,
) -> Result<Vec<GradPairFixed>> {
    Ok(build_histogram_shared_with_config(
        device_ordinal,
        block_data,
        block_n_rows,
        n_feats,
        feat_offsets,
        gpair,
        row_idx,
        SharedHistogramConfig::default(),
    )?
    .0)
}

/// 和 [`build_histogram_shared`] 相同,但允许扫描 launch 形状并返回纯 kernel
/// CUDA-event 时间。这个入口用于当前 characterization,不接训练循环。
#[allow(clippy::too_many_arguments)]
pub fn build_histogram_shared_with_config(
    device_ordinal: usize,
    block_data: &[Bin],
    block_n_rows: usize,
    n_feats: usize,
    feat_offsets: &[u32],
    gpair: &[GradPairFixed],
    row_idx: Option<&[RowId]>,
    config: SharedHistogramConfig,
) -> Result<(Vec<GradPairFixed>, SharedHistogramProfile)> {
    validate_inputs(
        block_data,
        block_n_rows,
        n_feats,
        feat_offsets,
        gpair,
        row_idx,
    )?;

    let n_hist = feat_offsets[n_feats] as usize;
    let n_selected = row_idx.map_or(block_n_rows, |rows| rows.len());
    if n_hist == 0 || n_selected == 0 {
        return Ok((
            vec![GradPairFixed::default(); n_hist],
            SharedHistogramProfile {
                kernel_ms: 0.0,
                h2d_ms: 0.0,
                d2h_ms: 0.0,
                feature_tiles: 0,
                row_launches_per_feature_tile: 0,
                threads_per_block: config.threads_per_block,
                shared_memory_budget_bytes: 0,
                active_blocks_per_sm: 0,
                theoretical_occupancy: 0.0,
                layout: config.layout,
            },
        ));
    }
    if config.threads_per_block == 0 || config.threads_per_block > 1024 {
        bail!("threads_per_block 要在 1..=1024");
    }

    let n_rows_u32 = u32::try_from(block_n_rows).context("CUDA histogram 行数超过 u32")?;
    let mut gpair_words = Vec::with_capacity(gpair.len() * 2);
    for pair in gpair {
        gpair_words.push(pair.grad);
        gpair_words.push(pair.hess);
    }

    // context 和 PTX module 按 device 缓存:早期版本在这里每次调用都
    // `CudaContext::new` + 重新解析 PTX,那是 wrapper 时间几百倍于 kernel
    // 的主因之一。训练循环走 `GpuTrainCtx`,这个入口留给独立 characterization。
    let state = device_state(device_ordinal)?;
    let ctx = &state.ctx;
    let stream = ctx.default_stream();
    let module = &state.module;
    let function = module
        .load_function(config.layout.kernel_name())
        .with_context(|| format!("PTX 中找不到 {}", config.layout.kernel_name()))?;

    // 先只使用无需 cuFuncSetAttribute opt-in 的默认上限。Ampere 通常是
    // 48 KiB,但这里以 driver 报告值为准;将来扫 100 KiB 档时再显式 opt-in。
    let device_shared_bytes = ctx
        .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK)
        .context("读取每 block shared-memory 上限")?;
    let max_threads_per_sm = ctx
        .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR)
        .context("读取每 SM 最大线程数")? as u32;
    let static_shared_bytes = function
        .shared_size_bytes()
        .context("读取 kernel 静态 shared-memory 大小")?;
    let device_shared_limit = usize::try_from(device_shared_bytes - static_shared_bytes)
        .context("shared-memory 上限无效")?;
    let shared_limit_bytes = config
        .shared_memory_budget_bytes
        .unwrap_or(device_shared_limit)
        .min(device_shared_limit);
    let bytes_per_bin = std::mem::size_of::<GradPairFixed>();
    let max_bins_per_tile = shared_limit_bytes / bytes_per_bin;
    if max_bins_per_tile == 0 {
        bail!("设备没有足够 shared memory 放一个 GradPairFixed");
    }

    let h2d_start = std::time::Instant::now();
    let bins_dev = stream.clone_htod(block_data).context("bins H2D")?;
    let gpair_dev = stream.clone_htod(&gpair_words).context("gpair H2D")?;
    // 七参数 FFI 没有 selected-row 长度。shared 版把根节点也显式展开成
    // row ids,完整 256-row blocks 和不足 256 的 tail 分两次 launch,从而
    // 不给 kernel 增加隐藏的第八个参数。
    let selected_rows: Vec<RowId> = match row_idx {
        Some(rows) => rows.to_vec(),
        None => (0..n_rows_u32).collect(),
    };
    let rows_dev = stream.clone_htod(&selected_rows).context("row_idx H2D")?;
    let offsets_dev = stream
        .clone_htod(feat_offsets)
        .context("feat_offsets H2D")?;
    let h2d_ms = h2d_start.elapsed().as_secs_f32() * 1e3;
    let mut out_dev = stream
        .alloc_zeros::<i64>(n_hist * 2)
        .context("分配 CUDA histogram")?;

    let threads_per_block = config.threads_per_block;
    let full_rows = n_selected / threads_per_block as usize * threads_per_block as usize;
    let full_blocks = u32::try_from(full_rows / threads_per_block as usize)
        .context("CUDA histogram grid 超过 u32")?;
    let tail_rows = u32::try_from(n_selected - full_rows).unwrap();
    let row_launches_per_feature_tile = usize::from(full_blocks > 0) + usize::from(tail_rows > 0);
    let start = stream
        .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
        .context("记录 shared histogram 起始 event")?;
    let mut feature_tiles = 0usize;
    let mut active_blocks_per_sm = u32::MAX;
    let mut feat_begin = 0usize;
    while feat_begin < n_feats {
        let hist_begin = feat_offsets[feat_begin] as usize;
        let mut feat_end = feat_begin;
        while feat_end < n_feats
            && feat_offsets[feat_end + 1] as usize - hist_begin <= max_bins_per_tile
        {
            feat_end += 1;
        }
        if feat_end == feat_begin {
            let bins = feat_offsets[feat_begin + 1] - feat_offsets[feat_begin];
            bail!(
                "特征 {feat_begin} 的 {bins} 个 bin 超过 shared-memory tile 上限 {max_bins_per_tile}"
            );
        }

        let tile_bins = feat_offsets[feat_end] as usize - hist_begin;
        let shared_mem_bytes =
            u32::try_from(tile_bins * bytes_per_bin).context("shared-memory tile 大小超过 u32")?;
        active_blocks_per_sm = active_blocks_per_sm.min(
            function
                .occupancy_max_active_blocks_per_multiprocessor(
                    threads_per_block,
                    shared_mem_bytes as usize,
                    None,
                )
                .context("计算 shared histogram occupancy")?,
        );
        let tile_n_feats = u32::try_from(feat_end - feat_begin).unwrap();
        let bins_view = bins_dev.slice(feat_begin * block_n_rows..feat_end * block_n_rows);
        let offsets_view = offsets_dev.slice(feat_begin..=feat_end);

        if full_blocks > 0 {
            let rows_view = rows_dev.slice(..full_rows);
            let config = LaunchConfig {
                grid_dim: (full_blocks, 1, 1),
                block_dim: (threads_per_block, 1, 1),
                shared_mem_bytes,
            };
            let mut launch = stream.launch_builder(&function);
            launch
                .arg(&bins_view)
                .arg(&gpair_dev)
                .arg(&rows_view)
                .arg(&n_rows_u32)
                .arg(&tile_n_feats)
                .arg(&offsets_view)
                .arg(&mut out_dev);
            // validate_inputs 保证所有 row/bin 有效,feature tile 保证动态
            // shared memory 不越过 driver 上限,device views 保持原 FFI。
            unsafe { launch.launch(config) }.context("launch hist_build_shared full blocks")?;
        }
        if tail_rows > 0 {
            let rows_view = rows_dev.slice(full_rows..);
            let config = LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (tail_rows, 1, 1),
                shared_mem_bytes,
            };
            let mut launch = stream.launch_builder(&function);
            launch
                .arg(&bins_view)
                .arg(&gpair_dev)
                .arg(&rows_view)
                .arg(&n_rows_u32)
                .arg(&tile_n_feats)
                .arg(&offsets_view)
                .arg(&mut out_dev);
            unsafe { launch.launch(config) }.context("launch hist_build_shared tail")?;
        }
        feat_begin = feat_end;
        feature_tiles += 1;
    }

    let end = stream
        .record_event(Some(sys::CUevent_flags::CU_EVENT_DEFAULT))
        .context("记录 shared histogram 结束 event")?;
    let kernel_ms = start
        .elapsed_ms(&end)
        .context("计算 shared histogram kernel 时间")?;

    let d2h_start = std::time::Instant::now();
    let words = stream.clone_dtoh(&out_dev).context("histogram D2H")?;
    let d2h_ms = d2h_start.elapsed().as_secs_f32() * 1e3;
    Ok((
        words
            .as_chunks::<2>()
            .0
            .iter()
            .map(|pair| GradPairFixed {
                grad: pair[0],
                hess: pair[1],
            })
            .collect(),
        SharedHistogramProfile {
            kernel_ms,
            h2d_ms,
            d2h_ms,
            feature_tiles,
            row_launches_per_feature_tile,
            threads_per_block,
            shared_memory_budget_bytes: shared_limit_bytes,
            active_blocks_per_sm,
            theoretical_occupancy: (active_blocks_per_sm * threads_per_block) as f32
                / max_threads_per_sm as f32,
            layout: config.layout,
        },
    ))
}

fn validate_inputs(
    block_data: &[Bin],
    block_n_rows: usize,
    n_feats: usize,
    feat_offsets: &[u32],
    gpair: &[GradPairFixed],
    row_idx: Option<&[RowId]>,
) -> Result<()> {
    if block_data.len() != n_feats.saturating_mul(block_n_rows) {
        bail!("bins 长度与 n_feats × n_rows 不符");
    }
    if feat_offsets.len() != n_feats + 1 || feat_offsets.first() != Some(&0) {
        bail!("feat_offsets 必须从 0 开始且长度为 n_feats + 1");
    }
    if feat_offsets.windows(2).any(|pair| pair[0] > pair[1]) {
        bail!("feat_offsets 必须单调不减");
    }
    if gpair.len() != block_n_rows {
        bail!("gpair 长度与 block_n_rows 不符");
    }
    if let Some(rows) = row_idx {
        if rows.iter().any(|&row| row as usize >= block_n_rows) {
            bail!("row_idx 含越界行");
        }
    }

    // CPU 路径依赖 ColumnBlock 保证 bin 合法;GPU OOB 的后果更重,第一版
    // 在 host 明确验一次。性能版接训练循环时再把这个检查移到构造边界。
    let selected: Box<dyn Iterator<Item = usize> + '_> = match row_idx {
        Some(rows) => Box::new(rows.iter().map(|&row| row as usize)),
        None => Box::new(0..block_n_rows),
    };
    let selected: Vec<usize> = selected.collect();
    for feat in 0..n_feats {
        let n_bins = feat_offsets[feat + 1] - feat_offsets[feat];
        let col = &block_data[feat * block_n_rows..(feat + 1) * block_n_rows];
        if selected.iter().any(|&row| {
            let bin = col[row];
            bin != MISSING_BIN && u32::from(bin) >= n_bins
        }) {
            bail!("特征 {feat} 含超出 feat_offsets 宽度的 bin");
        }
    }
    Ok(())
}
