//! 基础类型。刻意保持具体 —— 数值代码抽象过度会非常难调。

use rayon::prelude::*;

/// 训练期间 raw-margin prediction 的**权威副本**在哪一侧。
///
/// 这不是一个性能旋钮,而是一条所有权约定:
///
/// - [`PredictionAuthority::Host`](Self::Host)(`gpu_math = "exact"`):
///   host 的 `predictions` 是权威副本,每棵树结束后 device 把 delta 交回
///   host 施加。device 上只存在**本棵树的 delta**,不存在累计值。
/// - [`PredictionAuthority::Device`](Self::Device)(`gpu_math = "fast"`):
///   **训练期间 device 是权威副本**,累计 raw margin 常驻显存,delta 由
///   device 上的 gradient kernel 直接消费。host 那份 `predictions` 停在
///   初始 margin,**故意不同步** —— 只有外部真正需要时才显式同步一次。
///
/// ⚠️ 写下这条约定的原因:`gpu_math="fast"` 第一版只在一个地方加了
/// 「fast 就跳过 D2H」,结果 `build_tree` 里另一条路径照样每棵树拉回
/// 40 MB(实测 16.5 ms/轮,吃掉一半以上收益)。所以现在由
/// [`GpuTrainCtx::download_prediction_delta`] 一侧**主动报错**来兜底,
/// 而不是指望每个调用点都记得判断。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PredictionAuthority {
    Host,
    Device,
}

/// GPU 目标函数数学模式。见 [`TrainParams::gpu_math`]。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum GpuMath {
    /// CPU/GPU 模型逐字节相同。gradient 留在 host。
    #[default]
    Exact,
    /// device 上算 objective;**不保证** CPU/GPU 逐字节相同。
    Fast,
}

impl std::str::FromStr for GpuMath {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "exact" => Ok(Self::Exact),
            "fast" => Ok(Self::Fast),
            other => Err(format!("gpu_math 只支持 \"exact\" 或 \"fast\",收到 {other:?}")),
        }
    }
}

/// 一阶和二阶梯度。布局与 GPU 端一致(交错存放)。
#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct GradPair {
    pub grad: f32,
    pub hess: f32,
}

impl std::ops::AddAssign for GradPair {
    fn add_assign(&mut self, rhs: Self) {
        self.grad += rhs.grad;
        self.hess += rhs.hess;
    }
}

/// 反推用的减法:split.rs 靠「节点总和 - Σbins」拿到缺失部分,
/// hist.rs 靠「父 - 子」拿到兄弟节点。两处都是热路径上的常数优化。
impl std::ops::Sub for GradPair {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self { grad: self.grad - rhs.grad, hess: self.hess - rhs.hess }
    }
}

/// 定点梯度对。**直方图累加走这个,不走 GradPair。**
///
/// 整数加法没有舍入误差,而且和顺序无关。两件事因此是结构性成立的,
/// 不靠"小心一点":
///
/// 1. 近似平局的两个分裂不会因为累加噪声选错。f32 直方图的相对误差
///    在 1e-6 量级,深树里节点样本少、相对误差更大,正好落在
///    候选增益的差距上 —— 实测 depth 3 才开始翻,depth 1/2 全清。
/// 2. 阶段 3 的 GPU kernel 可以和 CPU **逐位**对拍,比"差在 1e-6
///    以内"强得多。
///
/// XGBoost 的 CPU hist 是拿 double 累加的,GPU 后端
/// (`GradientPairInt64`)则就是这个定点做法 —— 不是我们发明的。
/// 这里不跟 double 那条路,是因为直方图正是列分块要压的那一项
/// (n_bins × n_features × n_nodes),翻倍内存和项目目标直接冲突。
///
/// i64 不是 i32:上亿行累加会溢出 i32。
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[repr(C)]
pub struct GradPairFixed {
    pub grad: i64,
    pub hess: i64,
}

// `GradPairFixed` 的内存布局是**对外承诺**,不是实现细节:
//
// - GPU 的 device buffer 是 `CudaSlice<i64>`,kernel 签名是 `const int64_t*`,
//   两边都按「交错的 [grad, hess, ...]」解释同一段内存;
// - `flatten()` 正是靠这一点把 `&[GradPairFixed]` 零拷贝地看成 `&[i64]`。
//
// 所以这里用 const assert 把 size/align 钉死。**将来给这个 struct 加字段、
// 换类型或改 repr,都会在编译期直接失败**,而不是让 `flatten()` 悄悄按
// 错误的布局把内存喂给 H2D —— 那种错误只会表现成数字不对,极难定位。
const _: () = {
    assert!(std::mem::size_of::<GradPairFixed>() == 2 * std::mem::size_of::<i64>());
    assert!(std::mem::align_of::<GradPairFixed>() == std::mem::align_of::<i64>());
};

impl GradPairFixed {
    /// 把 `&[GradPairFixed]` 零拷贝看成交错的 `&[i64]`。
    ///
    /// **为什么需要它**:GPU 那边要的就是 `[grad, hess, grad, hess, ...]`,
    /// 而 `#[repr(C)]` 的两个连续 i64 **本来就是**这个布局。以前那里用一个
    /// 逐行的循环把它「摊平」成另一个 `Vec<i64>`,产出的字节和输入逐位相同
    /// —— 等于每轮多读一遍、多写一遍整个 O(行数) 数组。
    ///
    /// **为什么是 `&[i64]` 而不是 byte slice**:cudarc 的 `memcpy_htod` 是按
    /// 元素类型定型的,目标 buffer 是 `CudaSlice<i64>`,kernel 也声明成
    /// `const int64_t*`。走 byte view 就得把 device buffer 改成
    /// `CudaSlice<u8>`,反而要把类型信息从 FFI 边界上抹掉。
    ///
    /// 安全性:上面的 const assert 保证 size = 2 × i64、align = i64,
    /// `#[repr(C)]` 保证字段顺序是 grad 在前;因此 `2n` 个 i64 的视图
    /// 覆盖的正是同一段有效内存,且对齐合法。
    pub fn flatten(pairs: &[Self]) -> &[i64] {
        // SAFETY: 见上面的 const assert 与 repr(C)。长度按 2 倍换算,
        // 生命周期由入参 slice 借出,不产生悬垂。
        unsafe { std::slice::from_raw_parts(pairs.as_ptr().cast::<i64>(), pairs.len() * 2) }
    }
}

impl std::ops::AddAssign for GradPairFixed {
    fn add_assign(&mut self, rhs: Self) {
        // 溢出是这个方案唯一的失败模式,不能静默截断成一个"看起来
        // 挺正常"的数。debug 下明确报出来;release 下走普通加法
        // (scale 算对了就不会溢出),由下面的 debug_assert 兜底。
        if cfg!(debug_assertions) {
            self.grad = self.grad.checked_add(rhs.grad).expect(OVERFLOW_MSG);
            self.hess = self.hess.checked_add(rhs.hess).expect(OVERFLOW_MSG);
        } else {
            self.grad = self.grad.wrapping_add(rhs.grad);
            self.hess = self.hess.wrapping_add(rhs.hess);
        }
    }
}

impl std::ops::Sub for GradPairFixed {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self { grad: self.grad - rhs.grad, hess: self.hess - rhs.hess }
    }
}

const OVERFLOW_MSG: &str =
    "定点梯度累加溢出。GradQuantizer 的 scale 留的 headroom 不够 —— \
     它是按 n_rows × max|g| 算的,喂进来的行数或梯度量级和建 quantizer \
     时不是同一批就会这样。见 types.rs::GradQuantizer::new";

/// 梯度 → 定点的缩放因子。
///
/// **每轮重算**:梯度量级随训练变化,拿第一轮的 scale 套到第 50 轮
/// 要么溢出要么丢精度。
pub struct GradQuantizer {
    grad_scale: f64,
    hess_scale: f64,
}

impl GradQuantizer {
    pub fn new(gpair: &[GradPair], n_rows: usize) -> Self {
        // `max` 是结合律和交换律都成立的,而且对 f64 是**精确**运算
        // (不像求和会引入舍入),所以并行归约和串行逐位相同 ——
        // scale 不变,量化结果就不变,位精确契约不受影响。
        let (max_grad, max_hess) = gpair
            .par_iter()
            .fold(
                || (0.0f64, 0.0f64),
                |(g_acc, h_acc), g| {
                    (g_acc.max((g.grad as f64).abs()), h_acc.max((g.hess as f64).abs()))
                },
            )
            .reduce(|| (0.0f64, 0.0f64), |a, b| (a.0.max(b.0), a.1.max(b.1)));
        Self::from_bounds(max_grad, max_hess, n_rows)
    }

    /// 从已知的上界直接构造。
    ///
    /// 有它就不必为了求上界而把整个 f32 梯度数组物化出来 —— 扫一遍算
    /// 上界、再扫一遍直接产出定点值,峰值内存少一整份浮点梯度。
    pub fn from_bounds(max_grad: f64, max_hess: f64, n_rows: usize) -> Self {
        // grad 和 hess 分开定 scale:logistic 的 hess 最大 0.25,
        // 和 grad 不是一个量级,共用一个 scale 白扔几位精度。
        Self {
            grad_scale: pow2_scale(max_grad, n_rows),
            hess_scale: pow2_scale(max_hess, n_rows),
        }
    }

    pub fn to_fixed(&self, g: GradPair) -> GradPairFixed {
        GradPairFixed {
            grad: quantize(g.grad, self.grad_scale),
            hess: quantize(g.hess, self.hess_scale),
        }
    }

    /// 反量化成 f64。算 gain 和叶子权重走这条 —— 整条累加链路是整数,
    /// 只在最后一步转浮点,中途不要来回转。
    pub fn to_f64(&self, g: GradPairFixed) -> (f64, f64) {
        (g.grad as f64 / self.grad_scale, g.hess as f64 / self.hess_scale)
    }

    /// 反量化成 f32。存进树里的 sum_hess 之类走这条。
    pub fn to_float(&self, g: GradPairFixed) -> GradPair {
        let (grad, hess) = self.to_f64(g);
        GradPair { grad: grad as f32, hess: hess as f32 }
    }

    pub fn grad_scale(&self) -> f64 {
        self.grad_scale
    }

    pub fn hess_scale(&self) -> f64 {
        self.hess_scale
    }
}

fn quantize(v: f32, scale: f64) -> i64 {
    // round 是"四舍五入away from zero",确定的;as i64 的截断不是。
    (v as f64 * scale).round() as i64
}

/// 缩放因子,**取 2 的幂**。
///
/// 取 2 的幂不是为了好看:这样量化和反量化都只动指数,没有尾数误差。
/// 特别是平方损失的 hess 恒等于 1,`round(1 × 2^k) / 2^k` 必须精确
/// 还原成 1.0 —— 不然 `min_child_weight = 1` 的比较会在
/// 0.9999999 上翻边,比原来的浮点噪声还难查。
///
/// 预算:i64 有 63 位,留一位余量给父子相减,单值上限取
/// `2^62 / n_rows`。n_rows 上亿(约 2^27)时单值还能占 2^35,
/// 精度远超 f32 的 24 位尾数。
#[allow(clippy::neg_cmp_op_on_partial_ord)] // !(x > 0) 是为了让 NaN 也走这条
fn pow2_scale(max_abs: f64, n_rows: usize) -> f64 {
    if !(max_abs > 0.0) || !max_abs.is_finite() {
        return 1.0; // 梯度全零(或者是 NaN/inf,那是上游的错)
    }
    let budget = (1u64 << 62) as f64 / (n_rows.max(1) as f64);
    let raw = budget / max_abs;
    if !raw.is_finite() || raw <= 0.0 {
        return 1.0;
    }
    // 向下取到 2 的幂:宁可少用一位精度,也不能越过预算
    let scale = (raw.log2().floor() as i32).clamp(-1000, 1000);
    let scale = f64::exp2(scale as f64);
    debug_assert!(
        max_abs * scale * (n_rows.max(1) as f64) <= (1u64 << 62) as f64,
        "scale {scale} 超预算:max|g| {max_abs},n_rows {n_rows}"
    );
    scale
}

/// 量化后的 bin 编号。0..=254 是真实 bin,255 是缺失哨兵。
pub type Bin = u8;

/// 缺失值的哨兵 bin。
///
/// 缺失不占真实 bin:直方图累加时直接跳过,缺失的梯度和用
/// 「节点总和 - 所有非缺失 bin 之和」反推。这正是 split.rs 双向扫描
/// 需要的形式,也是 XGBoost 的做法。
///
/// 代价:真实 bin 只剩 255 个,所以 max_bin 上限是 255 而不是 256。
/// 想要 256 得把 Bin 换成 u16,但那样 ColumnBlock.data 就不是
/// Vec<u8> 了,内存翻倍、FFI 契约也要跟着改 —— 不值得。
/// 和 XGBoost 对拍时两边都设 max_bin=255 即可。
pub const MISSING_BIN: Bin = u8::MAX;

/// 单个特征的真实 bin 数上限。见 MISSING_BIN。
pub const MAX_BIN_LIMIT: u32 = 255;

/// `max_depth` 的上限,**两个后端共用一个数**。合法区间是 `0..=14`,
/// 其中 **0 表示只有根节点的树**(合法,有专门测试)。
///
/// 卡在这里的是 GPU:一层的 partition 节点数由 `MAX_LEVEL_NODES = 8192`
/// 的 `part_totals` buffer 兜着(每个节点两个 u32)。partition 跑在
/// 第 `0..max_depth-1` 层,第 L 层最多 `2^L` 个节点,所以最深要装
/// `2^(max_depth-1)` 个 —— **depth 14 正好是 8192,一个不多**;
/// depth 15 需要 16384,直接超。
///
/// ⚠️ **CPU 本身没有这个限制**(实测跑到 depth 17 都正常),但公开契约取
/// **两个后端的共同安全值**:否则 CPU 上训得出来的模型在 GPU 上复现不了,
/// 而「CPU/GPU 逐字节相同」是这个项目的核心不变量。
///
/// 要提高它就得把 `MAX_LEVEL_NODES` 一起提高:每 +1 层,那个 buffer 翻倍
/// (depth 15 → 16384 节点 → 128 KB,并不大),但要重新验证整条 partition
/// 路径,不是改个常量就完事。
pub const MAX_DEPTH_LIMIT: u32 = 14;

/// 行在当前训练集中的下标。
pub type RowId = u32;

/// 特征在全局特征列表中的下标(不是块内下标)。
pub type FeatId = u32;

#[derive(Clone, Debug)]
pub struct TrainParams {
    pub n_rounds: usize,
    pub max_depth: u32,
    pub max_bin: u32,
    pub learning_rate: f32,
    pub lambda: f32,
    pub gamma: f32,
    pub min_child_weight: f32,
    pub subsample: f32,
    pub colsample_bytree: f32,
    /// 每个列块包含多少个特征。决定内存工作集和并行粒度。
    ///
    /// 注意它同时是**并行度的上限**:并行维度就是列块,
    /// 能同时干活的线程最多 ceil(n_features / cols_per_block) 个。
    ///
    /// 块数不整除线程数会直接变成利用率损失 —— 实测 300 列 / 8 线程下
    /// `32`(10 块)只有 67% 利用率、`38`(8 块)有 78%,快 9%。
    /// 选值可以用 `columns::auto_cols_per_block`(**目前还不是默认**,
    /// 见 CLAUDE 的「thread-aware column blocking」)。
    pub cols_per_block: usize,
    /// 线程数。0 = 用满所有核,和 XGBoost 的 `nthread` 同义。
    ///
    /// 线程数**不影响结果**:块内是定点整数累加(与顺序无关),
    /// 块间的归约按块序固定。改这里之前先看
    /// tests/bit_exact_vs_xgboost.rs。
    pub nthread: usize,
    /// GPU histogram 用几条 stream(1 或 2)。`None` = 由 planner 按模式选。
    ///
    /// **这是内存计划的一部分,不只是调度参数。** 第二条 stream 要多一整套
    /// 列块 buffer,代价随 `cols_per_block` 走:c=7 时约 70 MB,
    /// c=28(HIGGS resident)时是 **280 MB,占整个预分配的约 25%**。
    ///
    /// **默认 1。** resident 上块 H2D 已经很少、没什么可重叠,单流实测反而
    /// 更快(wide resident 快 4.2%,HIGGS 打平);streaming 上双流只快
    /// 0.1–2.5%(不稳定、低于保留门槛),却要多吃 305 MB —— 而 streaming
    /// 是能力模式,指标是能装下多少数据。
    ///
    /// 用户显式给值时**完全尊重**,planner 不参与。
    pub hist_streams: Option<usize>,
    /// 常驻物理列块数。`None` = auto(显存规划器按剩余预算解)。
    ///
    /// **它改变内存计划,所以才暴露成参数**(和 `hist_streams` 同一条判据):
    /// 每个常驻块多占一整套列块 buffer,单价和一条 stream 完全相同
    /// (wide 10M×300、c=64 实测 **614 MB/块**)。
    ///
    /// 换来的是**消掉"每层重新上传同一个物理块"这笔重复流量**:
    /// 重复量是 `层数 × 块数 × 块字节`,wide 上 6×5×610 MB ≈ 18.2 GB/轮。
    /// 实测每常驻一块固定少 **3.65 GB/轮**,一路到全常驻**严格线性、没有拐点**
    /// (0 → 5 块:18.233 → 1.080 GB/轮,3.441 → 0.790 s/轮,**−77%**)。
    ///
    /// 所以 auto 的策略是**在安全预算内尽量多常驻**,而不是停在某个经验值。
    pub resident_blocks: Option<usize>,
    /// 定点量化是否在 device 上做。`None` = 默认开。
    ///
    /// **它改变内存计划,所以才暴露成参数**(和 `hist_streams` 同一条判据):
    /// 需要一份 f32 gpair 常驻 device,**+8 B/行**;HIGGS 上 O(行数) 部分从
    /// 28.6 B/行 涨到 36.3 B/行,也就是**固定显存下能训练的最大行数少约 21%**。
    ///
    /// 换来的是:host 的 max 扫描 / 逐元素量化 / root_sum 扫描全部消失,
    /// gpair H2D 减半(16 → 8 B/行),host 那份定点数组不再物化。
    /// 实测 HIGGS **−27.1%**、wide **−5.5%**,host RSS 少约 160 MB。
    ///
    /// 显存吃紧、更看重「能装下多少数据」的场景可以显式关掉。
    pub device_quantize: Option<bool>,
    /// 一次 GPU histogram launch 合并多少个 `Accumulate` 节点。`None` = 默认 16。
    ///
    /// **它改变内存计划,所以才暴露成参数**(和 `hist_streams`、
    /// `device_quantize` 同一条判据):批内每个节点要一份独立的直方图切片,
    /// 代价随 histogram slot 数走 —— HIGGS(2 slot)+2.8 MB,
    /// wide 10M×300(11 slot,cols=32)**+20.6 MB**。
    ///
    /// 换来的是消掉**每节点一次 host 往返**(launch → D2H → sync → 枚举 →
    /// 再 launch)。审计实测那条链每轮占 3.9 ms GPU 空闲,其中只有 11%
    /// 是 memcpy,其余是 host 延迟。实测 HIGGS **−3.9%**、wide 打平。
    ///
    /// `1` 表示关闭,回到逐节点 launch。
    pub hist_nodes_per_batch: Option<usize>,
    /// GPU streaming 可以使用的显存上限(字节)。`None` = 从运行时空闲显存推断。
    ///
    /// 语义是**「最多可以用到大约这么多」,不是「就分配这么多」**:它是一个
    /// 安全上限,真正的块宽由**块数拐点**决定(实测 4~5 块),达到拐点之后
    /// 不会因为预算还有余就继续加宽。
    ///
    /// 显式 `cols_per_block` 优先级更高,给了就完全关掉自动规划。
    ///
    /// **单位是字节。** 语义是「最多用到大约这么多」,不是「就分配这么多」。
    ///
    /// ✅ **已接通(2026-09-05)。** Python 侧公开暴露(`gpu_memory_budget`,
    /// 正整数字节),路径入口在建 source 之前调用
    /// `gpu_mem_plan::resolve_gpu_streaming_plan()`,把它作为可用显存传进去,
    /// 并**夹到实际空闲显存**(`min(requested, free)`)—— 给的比卡上还多不会
    /// 让规划器乐观。规划器再从中扣掉安全保留量
    /// (`default_safety_reserve`:15% 或 512 MiB 取大者)。
    ///
    /// 实测(wide 10M×300,预算 1 GiB):规划器解出 8 列/块、38 个物理块、
    /// **0 常驻 / 38 流式(pure streaming)**,估计峰值 479 MB;
    /// 而不设预算时是 64 列/块、5 块、**全常驻**、估计峰值 4088 MB。
    /// **公开参数确实改变执行模式。**
    ///
    /// ⚠️ **规划之后这个字段保存的是"生效值"**,不是原始请求值 ——
    /// 运行时回显(`GPU_PATH gpu_mem_budget=`)读的就是它,
    /// 这样日志里看到的就是真正约束了规划的那个数。
    ///
    /// ⚠️ **只影响自动规划。** 显式给了 `cols_per_block` 就完全绕开这次
    /// 显存查询,预算也就不起作用。
    pub gpu_memory_budget: Option<usize>,
    /// GPU 上目标函数数学的模式。默认 [`GpuMath::Exact`]。
    ///
    /// - [`GpuMath::Exact`]:当前行为。prediction D2H 回 host,gradient 在
    ///   host 上算,gpair 再 H2D 回去 —— 换来 **CPU/GPU 模型逐字节相同**。
    /// - [`GpuMath::Fast`]:prediction → gradient → quantization 全程留在
    ///   device 上,用 CUDA 的 `expf`。**不保证 CPU/GPU 逐字节相同。**
    ///
    /// 它存在的目的不是单纯让 HIGGS 更快,而是**把「位精确契约的价钱」
    /// 和「实现本身的差距」分开定价**:实测边界占 HIGGS 整轮 45%,
    /// 而 FB 的 device 侧工作已经比 XGBoost 的整轮快约 33%。
    pub gpu_math: GpuMath,
    pub seed: u64,
}

impl Default for TrainParams {
    fn default() -> Self {
        Self {
            n_rounds: 100,
            max_depth: 6,
            max_bin: 255,
            learning_rate: 0.3,
            lambda: 1.0,
            gamma: 0.0,
            min_child_weight: 1.0,
            subsample: 1.0,
            colsample_bytree: 1.0,
            cols_per_block: 64,
            nthread: 0,
            hist_streams: None,
            resident_blocks: None,
            hist_nodes_per_batch: None,
            gpu_memory_budget: None,
            gpu_math: GpuMath::default(),
            device_quantize: None,
            seed: 0,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Objective {
    SquaredError,
    Logistic,
}

#[cfg(test)]
mod layout_tests {
    use super::*;

    /// `GradPairFixed::flatten` 把整个 GPU H2D 路径的正确性押在
    /// 「repr(C) 的两个 i64 == 交错的 [grad, hess]」这条布局假设上。
    /// const assert 已经在编译期钉住 size/align,但它证明不了**字段顺序**
    /// —— 把 grad/hess 调换,size 和 align 都不变,而喂给 GPU 的数据会
    /// 整体错位,只表现成「数字不对」。这个测试钉住顺序。
    #[test]
    fn flatten_matches_manual_interleave_including_field_order() {
        let pairs = vec![
            GradPairFixed { grad: 1, hess: 2 },
            GradPairFixed { grad: -3, hess: 4 },
            GradPairFixed { grad: i64::MIN, hess: i64::MAX },
        ];
        // 这正是旧代码那个循环干的事;新路径必须逐位等价。
        let mut manual = Vec::with_capacity(pairs.len() * 2);
        for p in &pairs {
            manual.extend_from_slice(&[p.grad, p.hess]);
        }
        assert_eq!(GradPairFixed::flatten(&pairs), manual.as_slice());
    }

    #[test]
    fn flatten_handles_empty_without_dangling() {
        assert!(GradPairFixed::flatten(&[]).is_empty());
    }
}
