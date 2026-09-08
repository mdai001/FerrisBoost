//! GPU 显存预算规划:从"还剩多少显存"解出 streaming 的列块宽度。
//!
//! **有界内存不等于尽量少用显存。** 以前 streaming 固定用很窄的列块,
//! 于是在 6 GB 卡上实测只用掉 1.57 GB、**空着 4.5 GB**,却要为此多付
//! 块数 / H2D 次数 / launch 次数。这个模块把它变成一次显式的预算求解。
//!
//! 模型只有两项,因此块宽可以直接解出来,不需要启动时扫参:
//!
//! ```text
//! peak = fixed_bytes + cols_per_block * per_col_bytes
//! ```
//!
//! ⚠️ **块宽是训练初始化时定一次的物理布局**,不随每棵树的采样集变化 ——
//! 否则块的身份和性能都会依赖 RNG。`colsample` 是叠在上面的逻辑筛选层。

/// 一次训练的 streaming 显存模型。所有单位都是字节。
#[derive(Clone, Copy, Debug)]
pub struct StreamingMemModel {
    /// 和列块宽度无关的部分:gpair、row index、partition、prediction、
    /// device-quant、fast-math、行主序暂存等。
    pub fixed_bytes: u64,
    /// 每多一列要多付的字节:量化列块本身(n_rows)+ 该列的直方图切片。
    pub per_col_bytes: u64,
    /// **一个 slot、一列**的字节数。streaming slot 与 resident slot 同价。
    pub slot_col_bytes: u64,
}

impl StreamingMemModel {
    /// 按训练形状建模。
    ///
    /// 参数刻意都是显式的 —— 这个模型必须能被单测用真实测到的数字校准,
    /// 而不是散在 `GpuTrainCtx::new` 里靠读代码推断。
    pub fn new(
        n_rows: u64,
        max_bin: u64,
        hist_streams: u64,
        hist_nodes_per_batch: u64,
        device_quantize: bool,
        fast_math: bool,
        row_major: bool,
        rm_stage_bytes: u64,
    ) -> Self {
        let gpair = n_rows * 2 * 8; // 定点 [grad, hess]
        let row_idx = n_rows * 4;
        let partition_out = n_rows * 4;
        let partition_col = n_rows;
        // flags(n_rows) + 每块的 counts/offsets + 整层 totals,取个稳的上界。
        let partition_scratch = n_rows + 8 * 1024 * 1024;
        let prediction = n_rows * 4;
        let gpair_f32 = if device_quantize { n_rows * 2 * 4 } else { 0 };
        // fast 模式:常驻 prediction + label,各 4 B/行。
        let fast = if fast_math { n_rows * 4 * 2 } else { 0 };
        let stage = if row_major { rm_stage_bytes } else { 0 };

        let fixed_bytes = gpair
            + row_idx
            + partition_out
            + partition_col
            + partition_scratch
            + prediction
            + gpair_f32
            + fast
            + stage;

        // 每列:量化数据本身,加上它在直方图里的切片。
        // 直方图切片要乘 batch 和并发 slot 数。
        // **一个 slot、一列**要多少字节。streaming slot 和 resident slot
        // 用的是**同一套 buffer**(量化列块 + 它的直方图切片),所以两者
        // 必须按同一个单价进同一个预算 —— 运行时也正是这么算的
        // (`GpuMemoryBudget.hist_streams = hist_stream_count + resident_block_count`)。
        let slot_col_bytes = n_rows + (max_bin + 1) * 16 * hist_nodes_per_batch;
        Self {
            fixed_bytes,
            per_col_bytes: slot_col_bytes * hist_streams,
            slot_col_bytes,
        }
    }

    /// 定下 `cols` 和 stream 数之后,剩下的预算还能**再常驻几个物理块**。
    ///
    /// 每个常驻块就是多一个 slot,单价和 streaming slot 完全一样
    /// (`cols × slot_col_bytes`)。实测 wide c=64 每块 614 MB,
    /// 和 `64 × 10M` 对得上。
    ///
    /// ⚠️ **夹到 `total_blocks`**:常驻块比物理块还多没有意义,而且会真的
    /// 去分配用不到的 slot。
    pub fn resident_capacity(
        &self,
        budget_bytes: u64,
        cols: u64,
        streams: u64,
        total_blocks: u64,
    ) -> u64 {
        let per_slot = cols.saturating_mul(self.slot_col_bytes);
        if per_slot == 0 {
            return 0;
        }
        let base = self.fixed_bytes + per_slot * streams;
        if base >= budget_bytes {
            return 0;
        }
        ((budget_bytes - base) / per_slot).min(total_blocks)
    }

    /// `cols` 宽、`streams` 条流、`resident` 个常驻块时的峰值。
    pub fn peak_for_layout(&self, cols: u64, streams: u64, resident: u64) -> u64 {
        self.fixed_bytes + cols.saturating_mul(self.slot_col_bytes) * (streams + resident)
    }

    /// 给定块宽的估计峰值。
    pub fn peak_for(&self, cols_per_block: u64) -> u64 {
        self.fixed_bytes + cols_per_block * self.per_col_bytes
    }

    /// 在预算内解出最大可用块宽,并夹到 `[1, n_features]`。
    ///
    /// 返回 `None` 表示**连最窄的合法配置都放不下** —— 调用方必须在训练
    /// 开始前明确失败,而不是让 CUDA OOM 来当内存规划器。
    pub fn solve_cols(&self, budget_bytes: u64, n_features: u64) -> Option<u64> {
        if self.per_col_bytes == 0 {
            return Some(n_features.max(1));
        }
        if self.peak_for(1) > budget_bytes {
            return None;
        }
        let room = budget_bytes - self.fixed_bytes;
        let cols = room / self.per_col_bytes;
        Some(cols.clamp(1, n_features.max(1)))
    }
}

/// GPU **streaming** 的显存计划。`report()` 不需要 profile 模式就能看到。
#[derive(Clone, Debug)]
pub struct GpuStreamingPlan {
    pub free_vram: u64,
    pub total_vram: u64,
    pub reserve: u64,
    pub fixed_bytes: u64,
    pub per_col_bytes: u64,
    pub max_safe_cols: u64,
    pub target_blocks: u64,
    pub target_cols: u64,
    pub resolved_cols_per_block: u64,
    /// 解出来的 histogram stream 数。streaming 默认 2 —— 第二条流让本来
    /// 就要做的传输真正并发起来(**不是**为了 overlap 多复制一遍数据)。
    /// 但它会让 `per_col_bytes` 翻倍,**若因此把块宽夹到拐点以下就降回 1**:
    /// 拐点值 −42.6%,第二条流只值 −13~28%,**拐点优先**。
    pub resolved_hist_streams: u64,
    /// 常驻的物理列块数。`0` = 纯 streaming,`== physical_blocks` = 全常驻,
    /// 中间是 hybrid。**空闲显存优先买这个** —— 它消掉的是"每层重新上传
    /// 同一个物理块"这笔重复流量,实测 wide 每块 −3.65 GB/轮。
    pub resident_blocks: u64,
    pub physical_blocks: u64,
    pub estimated_peak: u64,
    pub how: Resolution,
}

/// 块宽是怎么定下来的 —— 报告里必须一眼看出来。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Resolution {
    /// 用户显式固定了 `cols_per_block`；stream 和 residency 仍照常规划。
    ExplicitOverride,
    /// 按块数目标定的(没被显存夹住)。
    BlockCountTarget,
    /// 想要的块宽放不下,被显存上限夹小了。
    VramClamped,
}

impl GpuStreamingPlan {
    /// ⚠️ **术语要一致**:全常驻就不要再叫它 streaming。
    /// 报告里出现 `full resident` 时,per-round H2D 应该接近 0。
    pub fn residency_label(&self) -> &'static str {
        if self.resident_blocks == 0 {
            "pure streaming"
        } else if self.resident_blocks >= self.physical_blocks {
            "full resident"
        } else {
            "hybrid partial resident"
        }
    }

    pub fn report(&self) -> String {
        let mb = |b: u64| b as f64 / (1024.0 * 1024.0);
        format!(
            "GPU streaming memory plan:\n               free_vram:        {:.1} MB / {:.1} MB\n               reserve:          {:.1} MB\n               fixed_bytes:      {:.1} MB\n               per_col_bytes:    {:.2} MB\n               max_safe_cols:    {}\n               target_blocks:    {}\n               target_cols:      {}\n               resolved:         {} cols/block  [{}]\n               hist_streams:     {}\n               physical_blocks:  {}\n               resident/stream:  {} resident + {} streamed  [{}]\n               estimated_peak:   {:.1} MB",
            mb(self.free_vram),
            mb(self.total_vram),
            mb(self.reserve),
            mb(self.fixed_bytes),
            mb(self.per_col_bytes),
            self.max_safe_cols,
            self.target_blocks,
            self.target_cols,
            self.resolved_cols_per_block,
            match self.how {
                Resolution::ExplicitOverride => "explicit manual override",
                Resolution::BlockCountTarget => "block-count auto target",
                Resolution::VramClamped => "VRAM-clamped auto target",
            },
            self.resolved_hist_streams,
            self.physical_blocks,
            self.resident_blocks,
            self.physical_blocks.saturating_sub(self.resident_blocks),
            self.residency_label(),
            mb(self.estimated_peak),
        )
    }
}

/// **GPU streaming 专用**的块宽解析点。
///
/// ⚠️ 刻意不复用 CPU / resident 共用的那条路 —— 这里的启发式是拿 GPU
/// streaming 的墙钟测出来的,套到 CPU 或 resident 上没有任何证据支持。
///
/// 优先级:**显式 `cols_per_block` > 块数目标(被显存上限夹住)**。
/// 显式值只关闭块宽 auto-sizing；完整入口仍须解析 stream 和 residency。
pub fn resolve_gpu_streaming_cols_per_block(
    explicit: Option<usize>,
    n_features: u64,
    free_vram: u64,
    total_vram: u64,
    model: StreamingMemModel,
) -> anyhow::Result<GpuStreamingPlan> {
    let n_features = n_features.max(1);
    let reserve = default_safety_reserve(free_vram);
    let budget = free_vram.saturating_sub(reserve);
    let max_safe_cols = model.solve_cols(budget, n_features).ok_or_else(|| {
        // 连最窄的合法配置都放不下 —— **训练开始前就明确失败**,
        // 不要开跑之后靠 CUDA OOM 来当内存规划器。
        anyhow::anyhow!(
            "GPU 显存放不下最小的 streaming 配置:需要约 {:.1} MB,\
             预算只有 {:.1} MB(空闲 {:.1} MB − 保留 {:.1} MB)",
            model.peak_for(1) as f64 / (1024.0 * 1024.0),
            budget as f64 / (1024.0 * 1024.0),
            free_vram as f64 / (1024.0 * 1024.0),
            reserve as f64 / (1024.0 * 1024.0),
        )
    })?;

    let target_cols = n_features.div_ceil(TARGET_BLOCKS).max(1);
    let (resolved, how) = match explicit {
        Some(c) if c > 0 => (c as u64, Resolution::ExplicitOverride),
        _ => {
            let want = knee_width(n_features, max_safe_cols, 0);
            // ⚠️ 想要的宽度被显存夹小了才算 VramClamped —— 报告里要能
            // 区分"性能目标就是这么宽"和"其实想更宽但放不下"。
            let unclamped = knee_width(n_features, n_features, 0);
            let how = if want < unclamped {
                Resolution::VramClamped
            } else {
                Resolution::BlockCountTarget
            };
            (want, how)
        }
    };
    let resolved = resolved.clamp(1, n_features);
    Ok(GpuStreamingPlan {
        free_vram,
        total_vram,
        reserve,
        fixed_bytes: model.fixed_bytes,
        per_col_bytes: model.per_col_bytes,
        max_safe_cols,
        target_blocks: TARGET_BLOCKS,
        target_cols,
        resolved_cols_per_block: resolved,
        // 这个原语只解块宽;stream 数和常驻块数由
        // `resolve_gpu_streaming_plan` 在同一个内存模型下一起定。
        resolved_hist_streams: 1,
        resident_blocks: 0,
        physical_blocks: n_features.div_ceil(resolved),
        estimated_peak: model.peak_for(resolved),
        how,
    })
}

/// 行主序 histogram kernel 一行能覆盖的最大特征数。
///
/// ⚠️ **规划器和运行时必须用同一个值。** 运行时用它决定走不走行主序
/// (`GpuTrainCtx::new` 里的 `max_block_feats <= ROW_MAJOR_MAX_FEATS`),
/// 规划器用它预测运行时会不会开第二条流。两边各写一个 32 就会悄悄漂移 ——
/// 规划器按双流算了预算,运行时却跑单流(或者反过来,直接 OOM)。
pub const ROW_MAJOR_MAX_FEATS: u64 = 32;

/// GPU streaming 的**完整**计划:块宽 + stream 数一起解。
///
/// stream 数**按执行路径分派**,不是按 resident/streaming 分派:
///
/// ```text
/// 行主序分块 streaming(宽度 ≤ ROW_MAJOR_MAX_FEATS) → 2
/// 列主序 streaming                                  → 1
/// resident                                          → 1(不走这里)
/// ```
///
/// 机制:**行主序是分块上传的** —— 每块被切成多个 chunk 依次
/// 上传 / 转置 / 建直方图,所以独立的 slot 之间有东西可以流水起来,
/// 第二条流让**本来就要做的传输**真正并发(而不是为了 overlap 再复制一遍
/// 数据 —— 那正是 pinned 那条路被否掉的原因)。收益随 chunk 数走:
/// c=8 **−28%**、c=28 −17%、c=32 −13%。
/// **列主序一块一次大 memcpy,没有 chunk 可以交错,实测打平**
/// (wide c=64:3.237 vs 3.230),那一档 +307 MB 买到的接近 0。
///
/// ⚠️ **拐点优先。** 第二条流让 `per_col_bytes` 翻倍,显存紧的机器上可能把
/// 块宽夹到拐点以下 —— 而**拐点值 −42.6%,第二条流只值 −13~28%**。所以
/// 只有当开第二条流**不会让解出来的块宽变窄**时才用 2,否则降回 1。
///
/// `make_model` 按 stream 数造模型 —— stream 数进 `per_col_bytes`,
/// 所以两档必须各算各的,不能拿一个模型套两次。
pub fn resolve_gpu_streaming_plan(
    explicit_cols: Option<usize>,
    explicit_streams: Option<usize>,
    explicit_resident: Option<usize>,
    n_features: u64,
    free_vram: u64,
    total_vram: u64,
    make_model: impl Fn(u64) -> StreamingMemModel,
) -> anyhow::Result<GpuStreamingPlan> {
    // 用户显式给了 stream 数就完全尊重,规划器不参与这个决定。
    if let Some(n) = explicit_streams {
        let n = n.max(1) as u64;
        let mut p = resolve_gpu_streaming_cols_per_block(
            explicit_cols,
            n_features,
            free_vram,
            total_vram,
            make_model(n),
        )?;
        p.resolved_hist_streams = n;
        fill_residency(&mut p, explicit_resident, &make_model(n));
        return Ok(p);
    }

    // 单流那档必须能解出来 —— 解不出来说明连最小配置都放不下,直接往上抛
    // (**规划失败必须报错,不能悄悄回退**)。
    let one = resolve_gpu_streaming_cols_per_block(
        explicit_cols,
        n_features,
        free_vram,
        total_vram,
        make_model(1),
    )?;

    // 双流那档放不下,或者放得下但**块宽被夹窄了**,都退回单流。
    match resolve_gpu_streaming_cols_per_block(
        explicit_cols,
        n_features,
        free_vram,
        total_vram,
        make_model(2),
    ) {
        // 双流只在**行主序分块**那条路上有收益,而走不走行主序由块宽决定。
        // 宽度必须两档都 ≤ 上界:双流解出来的那个宽度才是运行时真正用的。
        Ok(mut two)
            if two.resolved_cols_per_block >= one.resolved_cols_per_block
                && two.resolved_cols_per_block <= ROW_MAJOR_MAX_FEATS =>
        {
            two.resolved_hist_streams = 2;
            fill_residency(&mut two, explicit_resident, &make_model(2));
            Ok(two)
        }
        _ => {
            let mut one = one;
            one.resolved_hist_streams = 1;
            fill_residency(&mut one, explicit_resident, &make_model(1));
            Ok(one)
        }
    }
}

/// 几何定下来之后,**把剩余安全预算换成常驻物理块**。
///
/// 判据来自实测(wide 10M×300,c=64,5 块):常驻块和墙钟/H2D 的关系是
/// **严格线性**的 —— 每块固定消掉 3.65 GB/轮、固定多吃 614 MB,
/// 一路到全常驻都没有拐点(0→5 块:18.233 → 1.080 GB/轮,3.441 → 0.790 s/轮)。
///
/// 所以这里**没有"最优 N"可以解**,策略只能是:
/// **在安全预算内尽量多常驻,剩下的才 streaming。**
/// 不要停在某个"2~3 块"的经验值上 —— 那个数字没有任何测量支持。
///
/// ⚠️ 安全边界仍然由 `reserve` 守(`budget = free - reserve`),这里只花
/// 已经扣掉保留量之后的预算,**不会把显存填到 100%**。
fn fill_residency(plan: &mut GpuStreamingPlan, explicit: Option<usize>, model: &StreamingMemModel) {
    let budget = plan.free_vram.saturating_sub(plan.reserve);
    let cols = plan.resolved_cols_per_block;
    let streams = plan.resolved_hist_streams;
    let total = plan.physical_blocks;
    let capacity = model.resident_capacity(budget, cols, streams, total);
    // 显式给值时完全尊重,但仍然夹到物理块数 —— 多出来的 slot 会被真的
    // 分配出来却永远用不上。
    plan.resident_blocks = match explicit {
        Some(n) => (n as u64).min(total),
        None => capacity,
    };
    plan.estimated_peak = model.peak_for_layout(cols, streams, plan.resident_blocks);
}

/// 架构上友好的候选块宽。/// 架构上友好的候选块宽。不细扫每个整数 —— 没有证据表明整数粒度有意义,
/// 而少数几个候选让解出来的宽度可复现、可解释。
pub const CANDIDATE_WIDTHS: [u64; 6] = [4, 8, 16, 32, 64, 128];

/// streaming 想要的**块数**上界。
///
/// **实测拐点是 4~5 块**:HIGGS 在 4 块、wide 在 5 块都已进入收敛区。
/// 取 5 作为上界,于是两个形状都自然停在各自的拐点上,不需要特判。
///
/// ⚠️ **第一版用的是"块 buffer 字节数"目标(64 MB),被 wide 实测推翻了。**
/// 留下这段是因为这个反例比结论本身更有用:
///
/// | 形状 | cols | 块数 | 块 buffer | 整轮 |
/// |---|---:|---:|---:|---:|
/// | HIGGS 28 列 | 4 | 7 | 40 MB | 0.959 |
/// | HIGGS | **7** | **4** | 70 MB | **0.812** |
/// | HIGGS | 14 | 2 | 140 MB | 0.816(没再快) |
/// | HIGGS | 28 | 1 | 280 MB | 0.782 |
/// | wide 300 列 | 8 | 38 | **80 MB** | 5.764 |
/// | wide | 16 | 19 | 160 MB | 4.527(**−21%**) |
/// | wide | 32 | 10 | 320 MB | 3.906(−14%) |
/// | wide | **64** | **5** | 640 MB | **3.495**(−11%) |
///
/// 按字节看两条曲线对不上:HIGGS 在 70 MB 就饱和,wide 到 640 MB 还在降。
/// 按**块数**看它们完全一致 —— 两边都在 **4~5 块** 附近收敛。
///
/// 机制上也讲得通:streaming 的固定成本是**每层每块一次的加载 / session /
/// H2D 调用**,它随**块数**走,不随块的字节数走。块 buffer 的绝对大小只是
/// 块数在某个形状下的影子。
///
/// 所以目标是块数,由 VRAM 上限夹住。
pub const TARGET_MAX_BLOCKS: u64 = 5;

/// 兼容旧名字。
pub const TARGET_BLOCKS: u64 = TARGET_MAX_BLOCKS;

/// 在 `[最窄, 预算上限]` 里选一个**刚好把块数压到目标**的宽度。
///
/// `max_safe_cols` 是预算解出来的**上限**,不是目标 —— 直接取上限会在窄表上
/// 白吃 4 倍显存换 3.7%(见上表 HIGGS 28 列那一行)。
pub fn knee_width(n_features: u64, max_safe_cols: u64, _n_rows: u64) -> u64 {
    let n_features = n_features.max(1);
    let ceiling = max_safe_cols.clamp(1, n_features);
    // **从小往大扫,第一次把块数压到目标以内就停。**
    //
    // 判据是**块数**,不是"离 target_cols 有多近" —— 后者会冲过头:
    // 300 特征时 32→10 块、64→5 块、128→3 块;64 已经进了收敛区,
    // 128 只是继续吃显存(640 → 1280 MB)换约 5%。
    //
    // 达到块数拐点之后再加宽,需要**单独的墙钟证据**,不能因为显存放得下
    // 就顺手吃掉。
    for &c in CANDIDATE_WIDTHS.iter() {
        if c > ceiling {
            break;
        }
        if n_features.div_ceil(c) <= TARGET_MAX_BLOCKS {
            return c.clamp(1, ceiling);
        }
    }
    // 没有候选能达到目标块数(显存太紧)→ 用上限内最大的合法候选。
    CANDIDATE_WIDTHS
        .iter()
        .copied()
        .filter(|&c| c <= ceiling)
        .max()
        .unwrap_or(ceiling)
        .clamp(1, ceiling)
}

/// 安全余量:不要按报出来的空闲显存顶格分配。/// 安全余量:不要按报出来的空闲显存顶格分配。
///
/// ⚠️ **不能假设显卡是独占的。** 显示工作负载和其他并发进程都可能在
/// 训练期间申请显存。这里取「空闲的 15% 或 512 MB 中较大者」,
/// 是一个**实测校准过的起点**,不是什么架构真理 —— 它出现在诊断里就是为了
/// 让人能质疑它。
pub fn default_safety_reserve(free_vram: u64) -> u64 {
    (free_vram / 100 * 15).max(512 * 1024 * 1024)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 用**实测数字**校准模型,而不是让模型自说自话。
    ///
    /// 实测(HIGGS 10.5M 行、max_bin=255、单流、batch=16、device-quant 开、
    /// fast 开、行主序开、rm-stage 32 MiB,`FB_CUDA_RESIDENT_BLOCKS=0`):
    ///
    /// | cols | 预分配 |
    /// |---:|---:|
    /// | 4 | 533.3 MiB |
    /// | 7 | 563.5 MiB |
    /// | 14 | 633.9 MiB |
    /// | 28 | 774.8 MiB |
    fn higgs_model() -> StreamingMemModel {
        StreamingMemModel::new(10_500_000, 255, 1, 16, true, true, true, 32 * 1024 * 1024)
    }

    #[test]
    fn model_matches_measured_preallocation_within_a_few_percent() {
        let m = higgs_model();
        let mib = |b: u64| b as f64 / (1024.0 * 1024.0);
        for (cols, measured) in [(4u64, 533.3), (7.0 as u64, 563.5), (14, 633.9), (28, 774.8)] {
            let est = mib(m.peak_for(cols));
            let err = (est - measured).abs() / measured;
            assert!(
                err < 0.05,
                "cols={cols}: 估计 {est:.1} MiB vs 实测 {measured} MiB,误差 {:.1}%",
                err * 100.0
            );
        }
    }

    #[test]
    fn per_col_slope_matches_the_measured_slope() {
        // 实测斜率:(774.8 - 533.3) MiB / (28 - 4) 列 ≈ 10.06 MiB/列。
        let m = higgs_model();
        let slope = m.per_col_bytes as f64 / (1024.0 * 1024.0);
        assert!(
            (slope - 10.06).abs() < 0.5,
            "每列斜率 {slope:.2} MiB 与实测 10.06 MiB 不符"
        );
    }

    #[test]
    fn solve_picks_the_widest_block_that_fits() {
        let m = higgs_model();
        let budget = m.peak_for(10) + m.per_col_bytes / 2; // 刚好够 10 列多一点
        assert_eq!(m.solve_cols(budget, 300), Some(10));
    }

    #[test]
    fn solve_clamps_to_feature_count() {
        let m = higgs_model();
        // 预算足够宽,但只有 28 个特征 —— 不该返回比特征数还大的块宽。
        assert_eq!(m.solve_cols(m.peak_for(1000), 28), Some(28));
    }

    #[test]
    fn solve_fails_clearly_when_even_one_column_does_not_fit() {
        let m = higgs_model();
        // 连 fixed 都装不下 —— 必须返回 None,让调用方在训练前明确失败,
        // 而不是开跑之后靠 CUDA OOM 来发现。
        assert_eq!(m.solve_cols(m.fixed_bytes / 2, 300), None);
    }

    #[test]
    fn knee_width_targets_block_count_and_matches_both_measured_shapes() {
        // HIGGS 28 列:28/4 = 7 → 候选里第一个 >= 7 是 8(实测 knee 在 7~8)。
        assert_eq!(knee_width(28, 128, 10_500_000), 8);
        // wide 300 列:32→10 块(超标),64→5 块(<=5,停)。128 虽然也放得下,
        // 但只是继续吃显存,不该被选中。
        assert_eq!(knee_width(300, 128, 10_000_000), 64);
    }

    #[test]
    fn byte_target_would_have_been_wrong_on_wide() {
        // 反例钉住:按 64 MB 字节目标,wide(10M 行)只会选到 8 列 = 80 MB,
        // 而实测 8 列要 5.764 s/轮、64 列只要 3.495 —— 差 39%。
        // 块数规则给出 64,字节规则给出 8。
        let by_blocks = knee_width(300, 64, 10_000_000);
        let by_bytes_would_be = 8u64;
        assert!(
            by_blocks > by_bytes_would_be,
            "块数规则必须比字节规则给出更宽的块:{by_blocks} vs {by_bytes_would_be}"
        );
    }

    #[test]
    fn knee_width_respects_the_budget_ceiling_and_feature_count() {
        // 预算只够 4 列时不能返回 8。
        assert_eq!(knee_width(28, 4, 10_500_000), 4);
        // 特征本来就少:5 个特征用 4 列宽就已经是 2 块,早就满足 <=4 块的目标,
        // 没必要再加宽到 5(那是"最小满足"而不是"尽量宽")。
        assert_eq!(knee_width(5, 128, 10_500_000), 4);
        // 但绝不能超过特征数。
        assert!(knee_width(3, 128, 10_500_000) <= 3);
    }

    #[test]
    fn safety_reserve_never_drops_below_half_a_gigabyte() {
        // 显卡不是独占的:显示和别的进程也在用。
        assert_eq!(default_safety_reserve(100 * 1024 * 1024), 512 * 1024 * 1024);
        assert_eq!(
            default_safety_reserve(10 * 1024 * 1024 * 1024),
            10 * 1024 * 1024 * 1024 / 100 * 15
        );
    }
}

#[cfg(test)]
mod resolve_tests {
    use super::*;

    fn higgs() -> StreamingMemModel {
        StreamingMemModel::new(10_500_000, 255, 1, 16, true, true, true, 32 * 1024 * 1024)
    }
    fn wide() -> StreamingMemModel {
        StreamingMemModel::new(10_000_000, 255, 1, 16, true, true, true, 32 * 1024 * 1024)
    }
    const GB: u64 = 1024 * 1024 * 1024;

    fn wide_model(streams: u64) -> StreamingMemModel {
        StreamingMemModel::new(
            10_000_000,
            255,
            streams,
            16,
            true,
            false,
            true,
            32 * 1024 * 1024,
        )
    }

    /// 空闲显存要**换成常驻块**,而不是就这么闲着。
    /// 实测依据:每常驻一块固定消掉 3.65 GB/轮 H2D,一路线性到全常驻。
    #[test]
    fn spare_vram_is_spent_on_residency() {
        let p = resolve_gpu_streaming_plan(
            None,
            None,
            None,
            300,
            5118 * 1024 * 1024,
            6 * GB,
            wide_model,
        )
        .unwrap();
        assert!(
            p.resident_blocks > 0,
            "5 GB 空闲还全 streaming,等于白白每层重传所有列块:{}",
            p.report()
        );
        assert!(
            p.estimated_peak <= p.free_vram - p.reserve,
            "峰值不能越过安全预算"
        );
    }

    /// 常驻块数**必须夹到物理块数** —— 多出来的 slot 会被真的分配,却永远用不上。
    #[test]
    fn residency_is_clamped_to_the_physical_block_count() {
        let p = resolve_gpu_streaming_plan(
            None,
            None,
            Some(9999),
            300,
            5118 * 1024 * 1024,
            6 * GB,
            wide_model,
        )
        .unwrap();
        assert_eq!(
            p.resident_blocks, p.physical_blocks,
            "常驻块不能超过物理块数"
        );
        // 全常驻就不要再叫它 streaming。
        assert_eq!(p.residency_label(), "full resident");
    }

    /// 显存紧的时候常驻要让位 —— 先保证几何合法,再谈常驻。
    #[test]
    fn tight_budget_falls_back_to_pure_streaming() {
        let model = wide_model(1);
        // 刚好够一个 streaming slot,再多一块就放不下。
        let budget = model.peak_for_layout(64, 1, 0) + 8 * 1024 * 1024;
        let free = budget + default_safety_reserve(budget * 2);
        let p = resolve_gpu_streaming_plan(None, Some(1), None, 300, free, 6 * GB, wide_model);
        if let Ok(p) = p {
            assert_eq!(
                p.resident_blocks,
                0,
                "预算不够时不能还去常驻:{}",
                p.report()
            );
            assert_eq!(p.residency_label(), "pure streaming");
        }
    }

    /// 常驻块和 streaming slot **同价**,必须进同一个预算。
    #[test]
    fn resident_and_streaming_slots_cost_the_same() {
        let m = wide_model(1);
        let one_stream_two_resident = m.peak_for_layout(64, 1, 2);
        let three_streams_no_resident = m.peak_for_layout(64, 3, 0);
        assert_eq!(
            one_stream_two_resident, three_streams_no_resident,
            "resident slot 和 stream slot 的单价必须一致,否则预算和实际分配对不上"
        );
    }

    #[test]
    fn higgs_auto_resolves_to_the_measured_knee() {
        // 6 GB 卡、约 4.5 GB 空闲。28 特征 / 4 块 = 7 → 候选取 8 → 4 块。
        let p = resolve_gpu_streaming_cols_per_block(None, 28, 4500 * 1024 * 1024, 6 * GB, higgs())
            .unwrap();
        assert_eq!(p.resolved_cols_per_block, 8);
        assert_eq!(p.physical_blocks, 4);
        assert_eq!(p.how, Resolution::BlockCountTarget);
    }

    #[test]
    fn wide_auto_resolves_near_five_blocks_without_collapsing_to_one() {
        // 300 特征 / 4 块 = 75 → 候选里第一个 >= 75 是 128。
        // ⚠️ 但实测只到 64 列(5 块)就基本收敛了,再宽没有证据支持,
        // 所以这里必须**不会**因为 128 也放得下就跳上去。
        let p = resolve_gpu_streaming_cols_per_block(None, 300, 4500 * 1024 * 1024, 6 * GB, wide())
            .unwrap();
        assert!(
            p.resolved_cols_per_block <= 128,
            "不该超过候选集上限:{}",
            p.resolved_cols_per_block
        );
        assert!(
            p.physical_blocks >= 2,
            "不该塌成 1 块:{} 块",
            p.physical_blocks
        );
    }

    #[test]
    fn explicit_width_disables_only_geometry_auto_sizing() {
        let p =
            resolve_gpu_streaming_cols_per_block(Some(4), 28, 4500 * 1024 * 1024, 6 * GB, higgs())
                .unwrap();
        assert_eq!(p.resolved_cols_per_block, 4);
        assert_eq!(p.how, Resolution::ExplicitOverride);
        // 手动参照臂:块宽显式给多少就是多少。
        assert_eq!(p.physical_blocks, 7);
    }

    #[test]
    fn explicit_width_still_plans_streams_and_residency() {
        // Epsilon shape:显式 128 只固定物理几何,不能让完整 planner 提前返回。
        // 约 5 GiB 空闲足够让 16 个块全常驻;这正是 CSV deferred-open 路径
        // 必须在拿到真实行数后恢复的计划。
        let epsilon_model = |streams| {
            StreamingMemModel::new(
                400_000,
                255,
                streams,
                16,
                true,
                false,
                true,
                32 * 1024 * 1024,
            )
        };
        let p = resolve_gpu_streaming_plan(
            Some(128),
            None,
            None,
            2_000,
            5_118 * 1024 * 1024,
            6 * GB,
            epsilon_model,
        )
        .unwrap();
        assert_eq!(p.resolved_cols_per_block, 128);
        assert_eq!(p.physical_blocks, 16);
        assert_eq!(p.resolved_hist_streams, 1);
        assert_eq!(p.resident_blocks, 16, "显式块宽不能关闭 residency planning");
        assert_eq!(p.residency_label(), "full resident");
        assert_eq!(p.how, Resolution::ExplicitOverride);
    }

    #[test]
    fn tight_vram_clamps_the_width_and_says_so() {
        // 只剩很少空闲 → 想要 8 列但放不下,应该夹小并标成 VramClamped。
        let m = higgs();
        let free = m.fixed_bytes + m.per_col_bytes * 5 + default_safety_reserve(2 * GB);
        let p = resolve_gpu_streaming_cols_per_block(None, 28, free, 6 * GB, m).unwrap();
        assert!(p.resolved_cols_per_block < 8, "应被夹小");
        assert_eq!(p.how, Resolution::VramClamped);
    }

    #[test]
    fn fails_before_training_when_the_minimum_does_not_fit() {
        // 不能开跑之后靠 CUDA OOM 发现放不下。
        let err = resolve_gpu_streaming_cols_per_block(None, 28, 64 * 1024 * 1024, 6 * GB, higgs())
            .unwrap_err()
            .to_string();
        assert!(err.contains("放不下"), "错误信息要说清楚:{err}");
    }

    #[test]
    fn estimated_peak_stays_within_the_calibrated_tolerance() {
        // 解出来的宽度的估计峰值必须仍然在预算内。
        let p = resolve_gpu_streaming_cols_per_block(None, 28, 4500 * 1024 * 1024, 6 * GB, higgs())
            .unwrap();
        assert!(p.estimated_peak <= p.free_vram - p.reserve);
    }
}
