//! 训练循环。
//!
//! 注意循环顺序:外层是列块,内层是活跃节点。这是刻意的 ——
//! 反过来写(外层节点、内层列块)会让每层都重新加载所有列块,
//! 传输量变成 O(深度 × 数据)。

use crate::callback::{Callback, RoundMetrics};
use crate::columns::{BinCuts, ColumnBlock};
use crate::comm::Comm;
use rayon::prelude::*;

/// 分段计时。`FB_PROFILE=1` 时在训练结束打一张耗时分布表。
///
/// 只在每层/每轮的边界取时间(不是每行),开销可以忽略。留着是因为
/// 优化前后需要同一把尺子 —— 没有前后对比的优化等于没做。
pub mod prof {
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::{Mutex, MutexGuard};
    use std::time::Instant;

    macro_rules! counters {
        ($($name:ident),* $(,)?) => {
            $(pub static $name: AtomicU64 = AtomicU64::new(0);)*
            fn reset_timers() {
                $($name.store(0, Ordering::Relaxed);)*
            }
            pub fn report(rounds: usize) {
                let all: &[(&str, &AtomicU64)] = &[$((stringify!($name), &$name)),*];
                let total: u64 = all.iter().map(|(_, c)| c.load(Ordering::Relaxed)).sum();
                eprintln!("\n=== 耗时分布({rounds} 轮)===");
                eprintln!(
                    "⚠️ 口径:HIST / ENUM / SUBTRACT / GPU_* 是**各块 worker 累加**,\n                     　 段和可以大于整轮,**不能当墙钟占比或收益上限**读。\n                     　 同一段的真实墙钟见 HIST_PHASE_WALL;GRAD / QUANTIZE /\n                     　 UPDATE_PRED 每轮一次,本来就是墙钟。"
                );
                for (name, c) in all {
                    let ns = c.load(Ordering::Relaxed);
                    eprintln!("{name:<22}{:>9.2} s{:>8.1}%{:>10.1} ms/轮",
                        ns as f64 / 1e9, ns as f64 / total as f64 * 100.0,
                        ns as f64 / 1e6 / rounds as f64);
                }
                eprintln!("{:<22}{:>9.2} s", "合计", total as f64 / 1e9);
            }
        };
    }

    // ⚠️⚠️ 注意 HIST / ENUM / 所有 GPU_* 都是**各线程累加**,和墙钟不是一回事:
    // 块级 rayon 并行时它们的和会**大于**整轮墙钟。
    // 这条注释一直在,但**报表本身没印出来**,于是照样被误读成墙钟占比
    // (把 1130 ms 的 HIST 当成 475 ms 那一轮的 30%)。
    // 现在 `HIST_PHASE_WALL` 给出同一段的**真实墙钟**,报表也会显式标注口径。
    counters!(
        // HIST_PHASE_WALL:直方图并行段的**真实墙钟**(每层进出一次,
        // 不随 worker 数放大)。它是 HIST / ENUM / SUBTRACT / 全部 GPU_*
        // 那些累加值的**共同上界**,也是这一段任何优化的**收益上限**。
        HIST_PHASE_WALL,
        GRAD,
        QUANTIZE,
        HIST,
        ENUM,
        SUBTRACT,
        // PARTITION 拆开:取列 / 算分区位 / 切分行下标
        PART_FETCH,
        PART_BITS,
        PART_SPLIT,
        NEXT_LEVEL,
        UPDATE_PRED,
        EVAL,
        // GPU 分段。口径:H2D / D2H 是显式同步夹出来的真实传输墙钟,
        // KERNEL 是 CUDA event,HOST 是「整段墙钟 - 上面这些」的残差
        // (wrapper 自己的开销:launch、锁、host 侧摊平/拷贝)。
        GPU_GPAIR_H2D,
        GPU_HIST_BLOCK_H2D,
        GPU_HIST_ROWS_H2D,
        GPU_HIST_KERNEL,
        GPU_HIST_D2H,
        GPU_HIST_HOST,
        GPU_PART_COL_H2D,
        GPU_PART_ROWS_H2D,
        GPU_PART_KERNEL,
        GPU_PART_D2H,
        GPU_PART_HOST,
        GPU_PRED_D2H,
    );

    // Profile-only accounting. These are transfer submissions / bytes, not
    // durations, so they must not be folded into the timing table above.
    pub static GPU_GPAIR_CALLS: AtomicU64 = AtomicU64::new(0);
    pub static GPU_GPAIR_BYTES: AtomicU64 = AtomicU64::new(0);
    pub static GPU_HIST_BLOCK_CALLS: AtomicU64 = AtomicU64::new(0);
    pub static GPU_HIST_BLOCK_BYTES: AtomicU64 = AtomicU64::new(0);
    pub static GPU_HIST_ROWS_CALLS: AtomicU64 = AtomicU64::new(0);
    pub static GPU_HIST_ROWS_BYTES: AtomicU64 = AtomicU64::new(0);
    pub static GPU_HIST_D2H_CALLS: AtomicU64 = AtomicU64::new(0);
    pub static GPU_HIST_D2H_BYTES: AtomicU64 = AtomicU64::new(0);
    pub static GPU_PART_COL_CALLS: AtomicU64 = AtomicU64::new(0);
    pub static GPU_PART_COL_BYTES: AtomicU64 = AtomicU64::new(0);
    pub static GPU_PART_ROWS_CALLS: AtomicU64 = AtomicU64::new(0);
    pub static GPU_PART_ROWS_BYTES: AtomicU64 = AtomicU64::new(0);
    pub static GPU_PART_D2H_CALLS: AtomicU64 = AtomicU64::new(0);
    pub static GPU_PART_D2H_BYTES: AtomicU64 = AtomicU64::new(0);
    pub static GPU_PRED_D2H_CALLS: AtomicU64 = AtomicU64::new(0);
    pub static GPU_PRED_D2H_BYTES: AtomicU64 = AtomicU64::new(0);
    pub static GPU_KERNEL_CALLS: AtomicU64 = AtomicU64::new(0);
    pub static ROUND_WALL: AtomicU64 = AtomicU64::new(0);

    /// partition 的规模:取了多少次列、处理了多少行。
    pub static PART_COLS: AtomicU64 = AtomicU64::new(0);
    pub static PART_ROWS: AtomicU64 = AtomicU64::new(0);
    static PROFILE_RUN: Mutex<()> = Mutex::new(());

    /// profiler 是进程级静态计数器；每次训练必须从零开始，否则同一进程
    /// 里的 A/B、GPU/CPU 对拍和测试会把多次训练混在一张表里。
    pub fn reset() {
        reset_timers();
        PART_COLS.store(0, Ordering::Relaxed);
        PART_ROWS.store(0, Ordering::Relaxed);
        HIST_UPDATES.store(0, Ordering::Relaxed);
        VISITS.store(0, Ordering::Relaxed);
        ADVANCES.store(0, Ordering::Relaxed);
        BLOCK_SCANS.store(0, Ordering::Relaxed);
        EMPTY_SCANS.store(0, Ordering::Relaxed);
        for c in [
            &GPU_GPAIR_CALLS,
            &GPU_GPAIR_BYTES,
            &GPU_HIST_BLOCK_CALLS,
            &GPU_HIST_BLOCK_BYTES,
            &GPU_HIST_ROWS_CALLS,
            &GPU_HIST_ROWS_BYTES,
            &GPU_HIST_D2H_CALLS,
            &GPU_HIST_D2H_BYTES,
            &GPU_PART_COL_CALLS,
            &GPU_PART_COL_BYTES,
            &GPU_PART_ROWS_CALLS,
            &GPU_PART_ROWS_BYTES,
            &GPU_PART_D2H_CALLS,
            &GPU_PART_D2H_BYTES,
            &GPU_PRED_D2H_CALLS,
            &GPU_PRED_D2H_BYTES,
            &GPU_KERNEL_CALLS,
            &ROUND_WALL,
        ] {
            c.store(0, Ordering::Relaxed);
        }
    }

    /// 计数器是进程级的；profile 打开时完整训练必须串行，否则并发调用
    /// 会互相 reset / 累加，最后打印出一张看似合理的假表。
    pub fn begin() -> Option<MutexGuard<'static, ()>> {
        if !enabled() {
            return None;
        }
        let guard = PROFILE_RUN
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        reset();
        Some(guard)
    }

    pub fn report_partition(rounds: usize) {
        let c = PART_COLS.load(Ordering::Relaxed);
        let r = PART_ROWS.load(Ordering::Relaxed);
        if c == 0 {
            return;
        }
        let ns = PART_FETCH.load(Ordering::Relaxed)
            + PART_BITS.load(Ordering::Relaxed)
            + PART_SPLIT.load(Ordering::Relaxed);
        eprintln!("\n=== 行重分区 ===");
        eprintln!("取列次数        {:>12.1} 次/轮", c as f64 / rounds as f64);
        eprintln!(
            "处理的行        {:>12.1} 亿/轮",
            r as f64 / 1e8 / rounds as f64
        );
        eprintln!("每行            {:>12.2} ns", ns as f64 / r as f64);
    }

    /// 直方图累加的次数(不是时间):每「行 × 特征」一次。
    /// 用来算每次累加多少纳秒,再和理论下界比。
    pub static HIST_UPDATES: AtomicU64 = AtomicU64::new(0);

    pub fn report_hist(rounds: usize) {
        let n = HIST_UPDATES.load(Ordering::Relaxed);
        let ns = HIST.load(Ordering::Relaxed);
        if n == 0 {
            return;
        }
        eprintln!("\n=== 直方图累加 ===");
        eprintln!(
            "累加次数        {:>12.1} 亿/轮",
            n as f64 / 1e8 / rounds as f64
        );
        eprintln!("每次累加        {:>12.2} ns", ns as f64 / n as f64);
        // 每次累加要读 1 字节 bin + 16 字节梯度,读写 16 字节直方图槽位
        let bytes = n as f64 * (1.0 + 16.0);
        eprintln!(
            "流过的数据      {:>12.2} GB/轮  → {:.1} GB/s",
            bytes / 1e9 / rounds as f64,
            bytes / 1e9 / (ns as f64 / 1e9)
        );
    }

    pub fn report_gpu(rounds: usize) {
        let names: [(&str, &AtomicU64); 12] = [
            ("H2D:gpair(每轮一次)", &GPU_GPAIR_H2D),
            ("H2D:量化列块", &GPU_HIST_BLOCK_H2D),
            ("H2D:histogram 行索引", &GPU_HIST_ROWS_H2D),
            ("hist_build kernel", &GPU_HIST_KERNEL),
            ("D2H:直方图", &GPU_HIST_D2H),
            ("histogram wrapper 残差", &GPU_HIST_HOST),
            ("H2D:partition 取列", &GPU_PART_COL_H2D),
            ("H2D:partition 行索引", &GPU_PART_ROWS_H2D),
            ("partition kernel", &GPU_PART_KERNEL),
            ("D2H:partition offsets", &GPU_PART_D2H),
            ("partition wrapper 残差", &GPU_PART_HOST),
            ("D2H:leaf prediction", &GPU_PRED_D2H),
        ];
        if names.iter().all(|(_, c)| c.load(Ordering::Relaxed) == 0) {
            return;
        }
        let total: u64 = names.iter().map(|(_, c)| c.load(Ordering::Relaxed)).sum();
        eprintln!("\n=== GPU 分段({rounds} 轮)===");
        for (name, c) in names {
            let ns = c.load(Ordering::Relaxed);
            eprintln!(
                "{name:<28}{:>9.2} ms/轮{:>8.1}%",
                ns as f64 / 1e6 / rounds as f64,
                ns as f64 / total as f64 * 100.0
            );
        }
        // 这一行就是这次改造的验收指标:改造前 wrapper 是 kernel 的
        // 500 倍以上,说明量到的是分配器不是架构。
        let hist_kernel = GPU_HIST_KERNEL.load(Ordering::Relaxed).max(1);
        let hist_wrap = GPU_HIST_BLOCK_H2D.load(Ordering::Relaxed)
            + GPU_HIST_ROWS_H2D.load(Ordering::Relaxed)
            + GPU_HIST_D2H.load(Ordering::Relaxed)
            + GPU_HIST_HOST.load(Ordering::Relaxed);
        let part_kernel = GPU_PART_KERNEL.load(Ordering::Relaxed).max(1);
        let part_wrap = GPU_PART_COL_H2D.load(Ordering::Relaxed)
            + GPU_PART_ROWS_H2D.load(Ordering::Relaxed)
            + GPU_PART_D2H.load(Ordering::Relaxed)
            + GPU_PART_HOST.load(Ordering::Relaxed)
            + GPU_PRED_D2H.load(Ordering::Relaxed);
        eprintln!(
            "histogram wrapper / kernel = {:.1}×;partition wrapper / kernel = {:.1}×",
            hist_wrap as f64 / hist_kernel as f64,
            part_wrap as f64 / part_kernel as f64
        );
        let transfer_rows: [(&str, &AtomicU64, &AtomicU64); 8] = [
            ("H2D:gpair", &GPU_GPAIR_CALLS, &GPU_GPAIR_BYTES),
            (
                "H2D:histogram block",
                &GPU_HIST_BLOCK_CALLS,
                &GPU_HIST_BLOCK_BYTES,
            ),
            (
                "H2D:histogram rows",
                &GPU_HIST_ROWS_CALLS,
                &GPU_HIST_ROWS_BYTES,
            ),
            ("D2H:histogram", &GPU_HIST_D2H_CALLS, &GPU_HIST_D2H_BYTES),
            (
                "H2D:partition column",
                &GPU_PART_COL_CALLS,
                &GPU_PART_COL_BYTES,
            ),
            (
                "H2D:partition rows",
                &GPU_PART_ROWS_CALLS,
                &GPU_PART_ROWS_BYTES,
            ),
            (
                "D2H:partition offsets",
                &GPU_PART_D2H_CALLS,
                &GPU_PART_D2H_BYTES,
            ),
            (
                "D2H:leaf prediction",
                &GPU_PRED_D2H_CALLS,
                &GPU_PRED_D2H_BYTES,
            ),
        ];
        eprintln!("\n=== GPU transfer volume({rounds} 轮) ===");
        let mut h2d_bytes = 0u64;
        let mut d2h_bytes = 0u64;
        for (name, calls, bytes) in transfer_rows {
            let calls = calls.load(Ordering::Relaxed);
            let bytes = bytes.load(Ordering::Relaxed);
            if name.starts_with("H2D") {
                h2d_bytes += bytes;
            } else {
                d2h_bytes += bytes;
            }
            eprintln!(
                "{name:<28}{:>8.1} calls/轮{:>10.3} GB/轮",
                calls as f64 / rounds as f64,
                bytes as f64 / 1e9 / rounds as f64
            );
        }
        eprintln!(
            "H2D total{:>25.3} GB/轮",
            h2d_bytes as f64 / 1e9 / rounds as f64
        );
        eprintln!(
            "D2H total{:>25.3} GB/轮",
            d2h_bytes as f64 / 1e9 / rounds as f64
        );
        eprintln!(
            "CUDA kernels{:>21.1} launches/轮",
            GPU_KERNEL_CALLS.load(Ordering::Relaxed) as f64 / rounds as f64
        );
    }

    /// 把一次 GPU session 的分段耗时记进计数器。`wall` 是整段墙钟,
    /// 减掉已归因的部分就是 wrapper 自己的残差。
    pub fn record_gpu(
        t: &crate::backend::cuda::GpuSegmentTiming,
        wall_ns: u64,
        block_h2d: &'static AtomicU64,
        rows_h2d: &'static AtomicU64,
        kernel: &'static AtomicU64,
        d2h: &'static AtomicU64,
        host: &'static AtomicU64,
    ) {
        let ms_to_ns = |ms: f32| (ms as f64 * 1e6) as u64;
        let parts = ms_to_ns(t.block_h2d_ms)
            + ms_to_ns(t.rows_h2d_ms)
            + ms_to_ns(t.kernel_ms)
            + ms_to_ns(t.d2h_ms);
        block_h2d.fetch_add(ms_to_ns(t.block_h2d_ms), Ordering::Relaxed);
        rows_h2d.fetch_add(ms_to_ns(t.rows_h2d_ms), Ordering::Relaxed);
        kernel.fetch_add(ms_to_ns(t.kernel_ms), Ordering::Relaxed);
        d2h.fetch_add(ms_to_ns(t.d2h_ms), Ordering::Relaxed);
        host.fetch_add(wall_ns.saturating_sub(parts), Ordering::Relaxed);

        // The call site identifies histogram versus partition; the segment
        // itself carries exact counts/bytes from the CUDA boundary.
        if std::ptr::eq(block_h2d, &GPU_HIST_BLOCK_H2D) {
            GPU_HIST_BLOCK_CALLS.fetch_add(t.block_h2d_calls, Ordering::Relaxed);
            GPU_HIST_BLOCK_BYTES.fetch_add(t.block_h2d_bytes, Ordering::Relaxed);
            GPU_HIST_ROWS_CALLS.fetch_add(t.rows_h2d_calls, Ordering::Relaxed);
            GPU_HIST_ROWS_BYTES.fetch_add(t.rows_h2d_bytes, Ordering::Relaxed);
            GPU_HIST_D2H_CALLS.fetch_add(t.d2h_calls, Ordering::Relaxed);
            GPU_HIST_D2H_BYTES.fetch_add(t.d2h_bytes, Ordering::Relaxed);
        } else {
            GPU_PART_COL_CALLS.fetch_add(t.block_h2d_calls, Ordering::Relaxed);
            GPU_PART_COL_BYTES.fetch_add(t.block_h2d_bytes, Ordering::Relaxed);
            GPU_PART_ROWS_CALLS.fetch_add(t.rows_h2d_calls, Ordering::Relaxed);
            GPU_PART_ROWS_BYTES.fetch_add(t.rows_h2d_bytes, Ordering::Relaxed);
            GPU_PART_D2H_CALLS.fetch_add(t.d2h_calls, Ordering::Relaxed);
            GPU_PART_D2H_BYTES.fetch_add(t.d2h_bytes, Ordering::Relaxed);
        }
        GPU_KERNEL_CALLS.fetch_add(t.kernel_calls, Ordering::Relaxed);
    }

    /// 下降那一趟的计数(不是时间):
    /// VISITS   = (趟 × 块 × 行)的迭代次数,也就是实际扫了多少次
    /// ADVANCES = 真的往下走了一层的次数,理论下界是 行数 × 树深
    /// BLOCK_SCANS = 扫了多少个「块 × 趟」
    /// EMPTY_SCANS = 其中整块一次都没推动任何行的次数
    pub static VISITS: AtomicU64 = AtomicU64::new(0);
    pub static ADVANCES: AtomicU64 = AtomicU64::new(0);
    pub static BLOCK_SCANS: AtomicU64 = AtomicU64::new(0);
    pub static EMPTY_SCANS: AtomicU64 = AtomicU64::new(0);

    pub fn report_descent() {
        let v = VISITS.load(Ordering::Relaxed) as f64;
        let a = ADVANCES.load(Ordering::Relaxed) as f64;
        let b = BLOCK_SCANS.load(Ordering::Relaxed);
        let e = EMPTY_SCANS.load(Ordering::Relaxed);
        if v == 0.0 {
            return;
        }
        eprintln!("\n=== 沿树下降的扫描量 ===");
        eprintln!("行访问次数      {:>12.0} M", v / 1e6);
        eprintln!(
            "其中真的推进了  {:>12.0} M  ({:.1}%)",
            a / 1e6,
            a / v * 100.0
        );
        eprintln!(
            "白扫的行访问    {:>12.0} M  ({:.1}%)",
            (v - a) / 1e6,
            (v - a) / v * 100.0
        );
        eprintln!("块 × 趟         {:>12}", b);
        eprintln!(
            "其中整块白扫    {:>12}  ({:.1}%)",
            e,
            e as f64 / b as f64 * 100.0
        );
    }

    pub fn enabled() -> bool {
        std::env::var("FB_PROFILE").is_ok()
    }

    pub struct Timer(&'static AtomicU64, Instant);
    impl Timer {
        pub fn new(c: &'static AtomicU64) -> Self {
            Timer(c, Instant::now())
        }
    }
    impl Drop for Timer {
        fn drop(&mut self) {
            self.0
                .fetch_add(self.1.elapsed().as_nanos() as u64, Ordering::Relaxed);
        }
    }
}

use crate::hist::{self, Histogram};
use crate::metrics::Metric;
use crate::split::{self, SplitCandidate};
use crate::tree::{self, Model, Tree};
use crate::types::{
    Bin, FeatId, GradPair, GradPairFixed, GradQuantizer, Objective, PredictionAuthority, RowId,
    TrainParams, MISSING_BIN,
};

pub struct TrainConfig {
    pub params: TrainParams,
    pub objective: Objective,
    /// Suppress normal runtime summaries. Explicit profiling remains controlled
    /// by `FB_PROFILE` and is intentionally independent.
    pub quiet: bool,
    /// 为空时用 objective 的默认指标。名字沿用 XGBoost 的写法。
    pub eval_metrics: Vec<Box<dyn Metric>>,
    /// 和 XGBoost 的 base_score 同义:**变换前**空间里的初始预测。
    /// 回归时就是预测值本身,二分类时是概率。
    ///
    /// 注意 XGBoost 2.0 起不给这个参数时会用标签均值估一个,不再是
    /// 固定 0.5。对拍时两边都要显式设死,否则第一棵树的梯度就对不上。
    pub base_score: f32,
}

impl TrainConfig {
    /// XGBoost 老版本的默认值,两个 objective 都是 0.5。
    pub fn new(params: TrainParams, objective: Objective) -> Self {
        Self {
            params,
            objective,
            quiet: false,
            eval_metrics: vec![crate::metrics::default_metric(objective)],
            base_score: 0.5,
        }
    }
}

/// 一个验证集。名字会出现在日志和 evals_result 里。
///
/// early stopping 默认盯**最后一个** eval 的第一个 metric —— 语义
/// 和 XGBoost 一致,所以顺序有意义。
pub struct EvalSet<'a> {
    pub name: String,
    pub source: &'a dyn BlockSource,
    pub labels: &'a [f32],
}

/// 让 `with_block` 能带返回值。
///
/// 分成两个 trait 是因为 object safety:带泛型返回值的方法不能出现在
/// `dyn` trait 里,而训练循环全程用的是 `&dyn BlockSource`。
pub trait BlockSourceExt: BlockSource {
    /// `with_block_cols` 的取返回值版本。见 `with_block_ret`。
    fn with_block_cols_ret<R>(
        &self,
        i: usize,
        cols: &[usize],
        f: impl FnOnce(&ColumnBlock) -> R,
    ) -> anyhow::Result<R> {
        let mut slot = None;
        let mut f = Some(f);
        self.with_block_cols(i, cols, &mut |block| {
            let f = f.take().expect("with_block_cols 的实现必须恰好调用一次 f");
            slot = Some(f(block));
        })?;
        Ok(slot.expect("with_block_cols 的实现必须恰好调用一次 f"))
    }

    fn with_block_ret<R>(&self, i: usize, f: impl FnOnce(&ColumnBlock) -> R) -> anyhow::Result<R> {
        let mut slot = None;
        let mut f = Some(f);
        self.with_block(i, &mut |block| {
            let f = f.take().expect("with_block 的实现必须恰好调用一次 f");
            slot = Some(f(block));
        })?;
        Ok(slot.expect("with_block 的实现必须恰好调用一次 f"))
    }
}

impl<T: BlockSource + ?Sized> BlockSourceExt for T {}

/// 数据源:按列块产出数据。实现见 source 模块。
///
/// `Send + Sync`:建树时各线程并行读不同的列块,数据源本身只读。
/// 将来接 S3 流式加载时,内部的缓存要自己保证线程安全(比如 Mutex),
/// 别把可变状态暴露成 &self 上的裸字段。
pub trait BlockSource: Send + Sync {
    fn n_blocks(&self) -> usize;
    fn n_rows(&self) -> usize;
    fn n_features(&self) -> usize;
    fn cuts(&self) -> &BinCuts;
    /// **借用**第 i 个列块,闭包结束之后这个块就不保证还在了。
    ///
    /// 实现必须**恰好调用一次** `f`。
    ///
    /// 这个形状是两边的最小公倍数:
    /// - 常驻的 source(Dense / Arrow)直接把自己持有的块借出去,**零拷贝**
    /// - 按需加载的 source(Parquet)在闭包里加载,闭包一结束就 drop
    ///
    /// 早期签名是 `-> ColumnBlock`(交出所有权),为的是让按需加载可行
    /// —— 返回引用等于承诺块一直活在 source 里。代价是常驻 source 只能
    /// 克隆整块,于是**同一个 bug 出现了三次**(`feature_column`、
    /// `block_features`、`leaf_values`),每次都在调用点修。
    /// 改成借用之后,「克隆整块」在类型层面就不可能了,不靠每个实现
    /// 记得别这么做。
    ///
    /// 用 `&mut dyn FnMut` 而不是泛型,是为了让 trait 保持 object safe
    /// —— 训练循环全程拿的是 `&dyn BlockSource`。要拿返回值用
    /// `BlockSourceExt::with_block_ret`。
    fn with_block(&self, i: usize, f: &mut dyn FnMut(&ColumnBlock)) -> anyhow::Result<()>;

    /// 只借出块里**选中的那几列**(`cols` 是块内局部下标,升序)。
    ///
    /// `colsample_bytree` 要真的省字节,选中的特征 id 就必须一路穿透到
    /// **source 这一层**。只在 H2D 之前挑列是不够的:那样传输量降了,
    /// **读取量没降** —— `read 32 列 → pack 3 列 → H2D 3 列` 不能算作
    /// source 流量降到 3/32。
    ///
    /// **默认实现读整块**,这是刻意的诚实回退:物理格式无法部分读取的
    /// source(或还没实现的)会走这里,并且在
    /// [`Self::source_bytes_read`] 上如实记成整块的字节数,而不是假装
    /// 只读了选中的列。
    fn with_block_cols(
        &self,
        i: usize,
        _cols: &[usize],
        f: &mut dyn FnMut(&ColumnBlock),
    ) -> anyhow::Result<()> {
        self.with_block(i, f)
    }

    /// 这个 source 能否真的只读选中的列。
    ///
    /// 默认 `false` —— 调用方据此决定要不要压实块。**不能靠
    /// `with_block_cols` 交回来的东西去猜**:默认实现交的是整块,
    /// 如果调用方已经按压实算好了 offsets,就会对不上(这条断言真的
    /// 抓到过一次)。
    fn supports_column_selection(&self) -> bool {
        false
    }

    /// 本 source 到目前为止**真实读入**的量化字节数。
    ///
    /// 和 H2D 字节分开报:两者会因为块粒度、部分读取能力而不同,
    /// 混成一个数字就看不出「省的到底是读取还是传输」。
    /// 不统计的实现返回 `None`。
    fn source_bytes_read(&self) -> Option<u64> {
        None
    }

    /// 取**单个特征**的那一列 bin。
    ///
    /// 行重分区只需要选中的那一列,不需要整个块:全量驻留的 source 直接
    /// 切片,Parquet 这类按列存的只读那一个 column chunk。
    ///
    /// **没有默认实现是刻意的。** 曾经有一个(「加载整块再切一列」),
    /// 结果是每个全量驻留的 source 都静默继承了一个 6 倍的性能悬崖 ——
    /// HIGGS 上为了取 1 列克隆 294MB,吃掉单轮一半时间,profile 才看出来。
    /// 现在编译器会逼每个新 source 想清楚这件事。
    fn feature_column(&self, feat: FeatId) -> anyhow::Result<Vec<Bin>>;

    /// 第 i 块持有哪些特征。**只是元数据,不该碰数据。**
    ///
    /// 同样没有默认实现:默认实现只能是「加载整块再读 `feat_ids`」,
    /// 而那对按需加载的 source 意味着建树前先把数据集读一遍。
    /// 文档里写「按需加载的实现一定要覆盖它」本身就说明这个默认值是错的。
    fn block_features(&self, i: usize) -> anyhow::Result<Vec<FeatId>>;
}

pub fn train(
    source: &dyn BlockSource,
    labels: &[f32],
    evals: &[EvalSet],
    cfg: &TrainConfig,
    callbacks: &mut [Box<dyn Callback>],
    comm: &dyn Comm,
) -> anyhow::Result<Model> {
    train_with_backend(source, labels, evals, cfg, callbacks, comm, None)
}

/// 完整多轮/多层 GPU 训练入口。量化列块仍按 `BlockSource::with_block`
/// 流式处理;GPU 只替换 histogram 与 partition,CPU 的梯度、枚举、树结构
/// 和 callback 语义保持不变。
#[cfg(feature = "cuda")]
pub fn train_gpu(
    device_ordinal: usize,
    source: &dyn BlockSource,
    labels: &[f32],
    evals: &[EvalSet],
    cfg: &TrainConfig,
    callbacks: &mut [Box<dyn Callback>],
    comm: &dyn Comm,
) -> anyhow::Result<Model> {
    train_with_backend(
        source,
        labels,
        evals,
        cfg,
        callbacks,
        comm,
        Some(device_ordinal),
    )
}

fn validate_source_metadata(source: &dyn BlockSource, name: &str) -> anyhow::Result<()> {
    let n_features = source.n_features();
    anyhow::ensure!(source.n_blocks() > 0, "{name} 没有列块");
    anyhow::ensure!(
        source.cuts().n_feats() == n_features,
        "{name} 的 cuts 有 {} 个特征,数据源声明 {n_features} 个",
        source.cuts().n_feats()
    );
    let mut seen = vec![false; n_features];
    for block in 0..source.n_blocks() {
        let feats = source.block_features(block)?;
        anyhow::ensure!(!feats.is_empty(), "{name} 的列块 {block} 没有特征");
        anyhow::ensure!(
            feats.windows(2).all(|w| w[0] < w[1]),
            "{name} 的列块 {block} 特征必须严格递增且不能重复"
        );
        for feat in feats {
            let feat = feat as usize;
            anyhow::ensure!(feat < n_features, "{name} 的列块 {block} 含越界特征 {feat}");
            anyhow::ensure!(!seen[feat], "{name} 的特征 {feat} 出现在多个列块");
            seen[feat] = true;
        }
    }
    if let Some(feat) = seen.iter().position(|&present| !present) {
        anyhow::bail!("{name} 的特征 {feat} 不在任何列块中");
    }
    Ok(())
}

fn train_with_backend(
    source: &dyn BlockSource,
    labels: &[f32],
    evals: &[EvalSet],
    cfg: &TrainConfig,
    callbacks: &mut [Box<dyn Callback>],
    comm: &dyn Comm,
    gpu_device: Option<usize>,
) -> anyhow::Result<Model> {
    let n = source.n_rows();
    anyhow::ensure!(n > 0, "训练数据必须至少有一行");
    anyhow::ensure!(source.n_features() > 0, "训练数据必须至少有一个特征");
    validate_source_metadata(source, "训练集")?;
    anyhow::ensure!(
        labels.len() == n,
        "标签有 {} 行,训练数据有 {n} 行",
        labels.len()
    );
    anyhow::ensure!(labels.iter().all(|v| v.is_finite()), "标签必须全部是有限值");
    if cfg.objective == Objective::Logistic {
        anyhow::ensure!(
            labels.iter().all(|&v| (0.0..=1.0).contains(&v)),
            "binary:logistic 的标签必须在 [0, 1]"
        );
        anyhow::ensure!(
            cfg.base_score.is_finite() && cfg.base_score > 0.0 && cfg.base_score < 1.0,
            "binary:logistic 的 base_score 必须是 (0, 1) 内的有限概率"
        );
    } else {
        anyhow::ensure!(cfg.base_score.is_finite(), "base_score 必须是有限值");
    }
    let p = &cfg.params;
    anyhow::ensure!(
        (1..=crate::types::MAX_BIN_LIMIT).contains(&p.max_bin),
        "max_bin 要在 1..=255"
    );
    // ⚠️ **必须在共享路径上校验**,不能只在 GPU 里。以前完全没有校验:
    // CPU 会安静地接受 depth 16,GPU 则在训练深处报
    // 「一层的 partition 节点数超过 8192」——同一份参数两个后端行为不同,
    // 而且 GPU 那个错误信息离原因很远。
    // ⚠️ 下界是 **0**,不是 1:`max_depth = 0` 是合法的「只有根节点」的树,
    // 有专门的测试钉着它的语义(`root_only_tree_updates_the_next_round_predictions`)。
    anyhow::ensure!(
        p.max_depth <= crate::types::MAX_DEPTH_LIMIT,
        "max_depth 要在 0..={}(收到 {});0 表示只有根节点,上限由 GPU 的\
         每层 partition 节点 buffer 决定,见 types::MAX_DEPTH_LIMIT",
        crate::types::MAX_DEPTH_LIMIT,
        p.max_depth
    );
    anyhow::ensure!(p.cols_per_block > 0, "cols_per_block 必须 > 0");
    anyhow::ensure!(
        p.learning_rate.is_finite() && p.learning_rate >= 0.0,
        "learning_rate 必须是非负有限值"
    );
    anyhow::ensure!(
        p.lambda.is_finite() && p.lambda >= 0.0,
        "lambda 必须是非负有限值"
    );
    anyhow::ensure!(
        p.gamma.is_finite() && p.gamma >= 0.0,
        "gamma 必须是非负有限值"
    );
    anyhow::ensure!(
        p.min_child_weight.is_finite() && p.min_child_weight >= 0.0,
        "min_child_weight 必须是非负有限值"
    );
    anyhow::ensure!(
        p.subsample > 0.0 && p.subsample <= 1.0 && p.subsample.is_finite(),
        "subsample 必须在 (0, 1],收到 {}",
        p.subsample
    );
    anyhow::ensure!(
        p.colsample_bytree > 0.0 && p.colsample_bytree <= 1.0 && p.colsample_bytree.is_finite(),
        "colsample_bytree 必须在 (0, 1],收到 {}",
        p.colsample_bytree
    );
    for ev in evals {
        anyhow::ensure!(
            ev.labels.len() == ev.source.n_rows(),
            "验证集 {:?} 的标签有 {} 行,数据有 {} 行",
            ev.name,
            ev.labels.len(),
            ev.source.n_rows()
        );
        anyhow::ensure!(
            ev.source.n_features() == source.n_features(),
            "验证集 {:?} 有 {} 个特征,训练集有 {} 个",
            ev.name,
            ev.source.n_features(),
            source.n_features()
        );
        anyhow::ensure!(
            ev.labels.iter().all(|v| v.is_finite()),
            "验证集 {:?} 的标签必须全部是有限值",
            ev.name
        );
        if cfg.objective == Objective::Logistic {
            anyhow::ensure!(
                ev.labels.iter().all(|&v| (0.0..=1.0).contains(&v)),
                "验证集 {:?} 的 binary:logistic 标签必须在 [0, 1]",
                ev.name
            );
        }
        validate_source_metadata(ev.source, &format!("验证集 {:?}", ev.name))?;
        anyhow::ensure!(
            ev.source.cuts().offsets == source.cuts().offsets
                && ev.source.cuts().values == source.cuts().values,
            "验证集 {:?} 没有复用训练集的 cuts",
            ev.name
        );
    }
    let _profile_guard = prof::begin();
    let margin = base_margin(cfg.objective, cfg.base_score);
    let mut predictions = vec![margin; n];
    let mut model = Model {
        trees: Vec::with_capacity(cfg.params.n_rounds),
        base_score: cfg.base_score,
        objective: cfg.objective,
        n_features: source.n_features(),
        // 训练层不认识"列名"这个概念(它只见到量化后的块),所以留空,
        // 由知道 schema 的入口层填。空 = 位置模式,也是 numpy 入口的正解。
        feature_names: Vec::new(),
        schema_mode: None,
        best_iteration: None,
        best_score: None,
    };

    // 每个验证集缓存一份 raw margin,每轮只把**新加的那棵树**累上去。
    //
    // 原来是每轮拿整个模型重算一遍,那是 O(轮数²):100 轮的任务后半程
    // 光评估就比建树还贵,而 early stopping 的全部意义就是快速迭代。
    // CLAUDE.md 说这个优化要等数值对齐之后再做 —— 现在对齐了,而且有
    // tests/bit_exact_vs_xgboost.rs 兜底,可以动了。
    let mut eval_preds: Vec<Vec<f32>> = evals
        .iter()
        .map(|e| vec![base_margin(cfg.objective, cfg.base_score); e.source.n_rows()])
        .collect();

    // GPU 资源按**最坏情况**在训练开始分配一次:gpair、row index、
    // 装列块的 buffer、直方图和 partition 的临时区。训练过程中不再有任何
    // device 分配 / 释放 —— cudaMalloc/cudaFree 是同步的,夹在 kernel 前后
    // 会把整个 stream 挡住(改造前 wrapper 是 kernel 的几百倍)。
    //
    // 注意这里预留的是**装一个列块的 buffer**,不是让量化数据常驻显存:
    // 列块仍然每层每块 H2D 一次、用完覆写。
    let gpu = match gpu_device {
        Some(device) => {
            let cuts = source.cuts();
            let mut max_block_feats = 0usize;
            let mut max_hist_bins = 0usize;
            for b in 0..source.n_blocks() {
                let feats = source.block_features(b)?;
                max_block_feats = max_block_feats.max(feats.len());
                let offsets = cuts.block_hist_offsets(&feats);
                max_hist_bins = max_hist_bins.max(*offsets.last().unwrap() as usize);
            }
            let ctx = crate::backend::cuda::GpuTrainCtx::new(
                device,
                n,
                max_block_feats,
                max_hist_bins,
                source.n_blocks(),
                GPU_PARTITION_THREADS,
                crate::backend::cuda::SharedHistogramConfig {
                    runtime_log: !cfg.quiet,
                    // 用户显式给的 stream 数走正式参数,不走 env。
                    hist_streams: cfg.params.hist_streams,
                    resident_blocks: cfg.params.resident_blocks,
                    // 规划之后这个字段保存的是**生效值**(见 TrainParams 文档)。
                    gpu_memory_budget: cfg.params.gpu_memory_budget,
                    hist_nodes_per_batch: cfg.params.hist_nodes_per_batch,
                    gpu_math: cfg.params.gpu_math,
                    device_quantize: cfg.params.device_quantize,
                    ..Default::default()
                },
                prof::enabled(),
            )?;
            if prof::enabled() {
                eprintln!("\n=== GPU 显存预算 ===\n{}", ctx.budget().report());
                eprintln!("{}", ctx.host_budget().report());
                if let Ok((used, total)) = ctx.device_mem_used() {
                    eprintln!(
                        "分配后 device 已用 {:.1} MB / {:.1} MB(含 driver/context 固定开销)",
                        used as f64 / 1048576.0,
                        total as f64 / 1048576.0
                    );
                }
            }
            Some(ctx)
        }
        None => None,
    };

    // 线程池按 nthread 建一次,每轮 install 进去跑建树。
    // 0 = 交给 rayon 用满所有核(和 XGBoost 的 nthread 语义一致)。GPU
    // 双流在 nthread=1 时仍需要两条 host submission 线程。是否双流必须读取
    // GpuTrainCtx **已经解析好的结果**，不能在这里再次读取 env；否则显式参数
    // 和默认单流会与实际 context 不一致。CPU 路径继续严格尊重单线程配置。
    let worker_threads = if gpu.as_ref().is_some_and(|ctx| ctx.hist_stream_count() == 2)
        && cfg.params.nthread == 1
    {
        2
    } else {
        cfg.params.nthread
    };
    let pool = crate::threading::build_pool(worker_threads)?;

    // `gpu_math = "fast"`:prediction 归 device 所有。label 上传一次,
    // 常驻 prediction 初始化成 raw-margin base score。host 的 `predictions`
    // 从此不再更新 —— 训练结束后没人读它(eval 走独立的 `eval_preds`),
    // 所以这不是「两份状态不同步」,而是所有权转移。
    let fast_math = gpu.as_ref().is_some_and(|c| c.fast_math_enabled());
    if fast_math {
        gpu.as_ref()
            .expect("fast_math 蕴含 gpu 存在")
            .init_fast_math(labels, margin)?;
    }

    // 融合路径把「施加 delta」推迟到下一轮开头,所以要记着是否欠一次。
    let mut pending_delta = false;
    for round in 0..cfg.params.n_rounds {
        let round_started = std::time::Instant::now();
        // 浮点梯度只用来求 scale 和转定点,**转完就扔**。
        //
        // 留着它等于在 build_tree(整轮内存最紧的地方)白占一份
        // n_rows × 8 字节 —— 实测 500 万行时峰值 669 → 630 MB。
        // 早期版本是让它活到本轮结束的。
        //
        // 试过「两遍扫描、根本不物化 f32」:峰值一样(全局峰值在
        // build_tree 里,不在转换那一刻),但多扫一遍要多花约 2% 的时间。
        // 所以维持一遍扫描 + 提前 drop。
        //
        // 每轮重建 quantizer:梯度量级随训练变化,拿第一轮的 scale
        // 套到后面要么溢出要么把精度丢光。
        // 这两段都在训练线程池里跑,所以它们服从 `nthread`,不会越过它去
        // 抢 Rayon 全局池。三步(梯度、求 scale、转定点)全部与顺序无关:
        // 前后两步是逐元素 map 且 collect 保序,中间那步是 f64 max 归约,
        // 而 max 是结合律+交换律成立且**精确**的(不像浮点求和会有舍入),
        // 所以并行结果和串行**逐位相同**。位精确基线守着这一条。
        // ---- host 侧那一遍 O(行数) ----
        //
        // 非融合时是**两遍**:上一轮结尾 `pred[i] += delta[i]`,这一轮开头
        // `grad[i] = f(pred[i], label[i])`。两遍都是逐行独立的,中间那次
        // 对 `predictions` 的完整读取纯属多余,而且各自要过一次线程池
        // (两次 fork/join 屏障)。
        //
        // 融合之后一遍就够:施加 delta,**就地**用更新后的值算梯度。
        // 每行只依赖同下标的输入,没有任何重结合,所以**逐位相同**。
        //
        // `predictions` 在两点之间没有任何读者(eval 走 `eval_preds`,
        // 回调和 metric 也只看 eval),所以把施加推迟到这里是安全的;
        // 训练结束时若还欠一次施加,循环外会补上,不留下过期状态。
        let fuse = pending_delta;
        // fast 模式下 host 侧完全不产生 gpair:prediction apply、objective
        // gradient 和量化在 device 上一次做完。
        let gpair_f32 = if fast_math {
            Vec::new()
        } else if fuse {
            let ctx = gpu.as_ref().expect("pending_delta 蕴含 gpu 存在");
            let obj = cfg.objective;
            let mut out: Vec<GradPair> = Vec::new();
            pool.install(|| -> anyhow::Result<()> {
                let _t = prof::Timer::new(&prof::GRAD);
                ctx.with_prediction_delta(&mut |delta| {
                    out = predictions
                        .par_iter_mut()
                        .zip(delta.par_iter())
                        .zip(labels.par_iter())
                        .map(|((p, d), &y)| {
                            *p += *d;
                            gradient_of(*p, y, obj)
                        })
                        .collect();
                })
            })?;
            pending_delta = false;
            out
        } else {
            pool.install(|| {
                let _t = prof::Timer::new(&prof::GRAD);
                compute_gradients(&predictions, labels, cfg.objective)
            })
        };
        // ⬅ **行采样在这里发生**:完整梯度已经算出来,root sum / 直方图还没开始。
        //
        // ⚠️ 改的是**工作副本**:未选中的行 gpair 置零,于是它们对直方图和
        // node_sum 贡献零(定点零加上去什么都不改)。**它们仍然跟着每一次
        // 分裂走、仍然走到叶子、仍然拿到这棵树的预测增量** ——
        // 采样掩码**不是**预测掩码。
        //
        // uniform 采样**不做 `1/p` 加权**(那是 gradient-based 采样才需要的)。
        let mut gpair_f32 = gpair_f32;
        if cfg.params.subsample < 1.0 && !fast_math {
            let kept = {
                // f32 侧置零:量化在这之后发生,所以 scale 也只看选中的行。
                let th = crate::subsample::threshold_for(cfg.params.subsample);
                let mut kept = 0usize;
                for (i, g) in gpair_f32.iter_mut().enumerate() {
                    if crate::subsample::row_selected(i as u64, th, cfg.params.seed, round as u64) {
                        kept += 1;
                    } else {
                        *g = GradPair { grad: 0.0, hess: 0.0 };
                    }
                }
                kept
            };
            if prof::enabled() && round == 0 {
                eprintln!(
                    "SUBSAMPLE rate={} kept={kept}/{} ({:.1}%)",
                    cfg.params.subsample,
                    gpair_f32.len(),
                    100.0 * kept as f64 / gpair_f32.len().max(1) as f64
                );
            }
        }
        // device-side quantization:求 scale、转定点、归约 root sum 都在 device 上,
        // 走 PCIe 的是 f32(8 B/行)而不是定点(16 B/行)。位精确的理由见
        // `GpuTrainCtx::upload_and_quantize_gpair`。CPU 路径完全不变。
        let device_quant = gpu.as_ref().is_some_and(|c| c.device_quantize_enabled());
        let (quantizer, fixed, dev_root_sum) = if fast_math {
            // prediction D2H、host 按行 gradient、gpair H2D —— 三项都不存在。
            // ⚠️ **不保证 CPU/GPU 逐字节相同**,这是 `gpu_math="fast"` 的定义。
            let ctx = gpu.as_ref().expect("fast_math 蕴含 gpu 存在");
            let _t = prof::Timer::new(&prof::QUANTIZE);
            // 行采样在 device 上完成:阈值 host 折算,device 只比整数。
            // ⚠️ kernel 里 prediction 更新是**无条件**的,只有 gpair 被置零 ——
            // 采样掩码不是预测掩码。
            let (q, root, _ms) = ctx.fast_gradient_and_quantize(
                cfg.objective,
                crate::subsample::threshold_for(cfg.params.subsample),
                cfg.params.seed,
                round as u64,
            )?;
            // delta 已经被 device kernel 吃掉,不再欠 host 一次施加。
            pending_delta = false;
            (q, Vec::new(), Some(root))
        } else if device_quant {
            let ctx = gpu.as_ref().expect("device_quant 蕴含 gpu 存在");
            let _t = prof::Timer::new(&prof::QUANTIZE);
            let (q, root, ms) = ctx.upload_and_quantize_gpair(&gpair_f32)?;
            drop(_t);
            if prof::enabled() {
                prof::GPU_GPAIR_H2D
                    .fetch_add((ms as f64 * 1e6) as u64, std::sync::atomic::Ordering::Relaxed);
                prof::GPU_GPAIR_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                prof::GPU_GPAIR_BYTES.fetch_add(
                    (gpair_f32.len() * std::mem::size_of::<GradPair>()) as u64,
                    std::sync::atomic::Ordering::Relaxed,
                );
            }
            (q, Vec::new(), Some(root))
        } else {
            pool.install(|| {
                let _t = prof::Timer::new(&prof::QUANTIZE);
                let quantizer = GradQuantizer::new(&gpair_f32, n);
                let fixed: Vec<GradPairFixed> =
                    gpair_f32.par_iter().map(|&g| quantizer.to_fixed(g)).collect();
                (quantizer, fixed, None)
            })
        };
        drop(gpair_f32);
        // 定点梯度每轮传一次,不是每个列块重传一次。
        if let (Some(ctx), false) = (gpu.as_ref(), device_quant) {
            let h2d_ms = ctx.upload_gpair(&fixed)?;
            if prof::enabled() {
                prof::GPU_GPAIR_H2D.fetch_add(
                    (h2d_ms as f64 * 1e6) as u64,
                    std::sync::atomic::Ordering::Relaxed,
                );
                prof::GPU_GPAIR_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                prof::GPU_GPAIR_BYTES.fetch_add(
                    (fixed.len() * std::mem::size_of::<GradPairFixed>()) as u64,
                    std::sync::atomic::Ordering::Relaxed,
                );
            }
        }
        // build_tree 结束时已经知道每行落在哪个叶子。直接用那份行分区
        // 累加训练预测,不要再把整棵树走一遍。验证集没有参与建树,
        // 下面仍然需要正常下降。
        // **每棵树采一次**,基于全局特征 id,与线程数 / backend /
        // resident-vs-streaming / cols_per_block 都无关。
        // 语义对齐 XGBoost(count = max(1, floor(rate * n)),升序),
        // 但序列是 FerrisBoost 自己的确定性实现 —— 见 `colsample` 模块。
        let sampled: Option<Vec<FeatId>> = if cfg.params.colsample_bytree < 1.0 {
            Some(crate::colsample::sample_features(
                source.n_features(),
                cfg.params.colsample_bytree,
                cfg.params.seed,
                round,
            ))
        } else {
            None
        };
        let tree = pool.install(|| {
            build_tree(
                source,
                &fixed,
                dev_root_sum,
                &quantizer,
                &mut predictions,
                cfg,
                comm,
                gpu.as_ref(),
                sampled.as_deref(),
            )
        })?;
        // GPU 路径的叶子增量留在 device buffer 里没有施加,记一笔债;
        // CPU 路径是按行 scatter 进 `predictions` 的,没有可推迟的 delta。
        if gpu.is_some() {
            pending_delta = true;
        }
        {
            let _t = prof::Timer::new(&prof::EVAL);
            for (ev, pred) in evals.iter().zip(eval_preds.iter_mut()) {
                update_predictions(pred, &tree, ev.source)?;
            }
        }
        model.trees.push(tree);

        let m = evaluate(&eval_preds, evals, round, cfg);
        // 任一 callback 返回 false 就停 —— early stopping 走这条路。
        let keep_going = callbacks
            .iter_mut()
            .map(|cb| cb.after_iteration(&m))
            .fold(true, |a, b| a && b);
        if !keep_going {
            if prof::enabled() {
                prof::ROUND_WALL.fetch_add(
                    round_started.elapsed().as_nanos() as u64,
                    std::sync::atomic::Ordering::Relaxed,
                );
            }
            break;
        }
        if prof::enabled() {
            prof::ROUND_WALL.fetch_add(
                round_started.elapsed().as_nanos() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
        }
    }

    // 融合路径把最后一棵树的 delta 推迟了,而训练结束时没人再读 `predictions`。
    // 但把它留成过期状态是个陷阱:以后有人加一个「训练后读预测」的功能就会
    // 静默拿到少一棵树的值。整次训练补一遍 O(行数),代价可以忽略。
    // fast 模式下 `predictions` 的所有权在 device,host 那份本来就不参与
    // 训练,补一次 D2H 只会把一份没人读的数组写成「差一棵树」的状态,
    // 白付一次 O(行数) 传输。
    if pending_delta && !fast_math {
        if let Some(ctx) = gpu.as_ref() {
            let mut delta_applied = Ok(());
            pool.install(|| {
                delta_applied = ctx.with_prediction_delta(&mut |delta| {
                    predictions
                        .par_iter_mut()
                        .zip(delta.par_iter())
                        .for_each(|(p, d)| *p += *d);
                });
            });
            delta_applied?;
        }
        pending_delta = false;
    }
    let _ = pending_delta;

    if prof::enabled() {
        let completed_rounds = model.trees.len().max(1);
        if let Some(ctx) = gpu.as_ref() {
            if let Ok((used, total)) = ctx.device_mem_used() {
                eprintln!(
                    "\n训练结束时 device 已用 {:.1} MB / {:.1} MB;其中本进程预分配 {:.1} MB",
                    used as f64 / 1048576.0,
                    total as f64 / 1048576.0,
                    ctx.budget().total() as f64 / 1048576.0
                );
            }
        }
        prof::report(completed_rounds);
        eprintln!(
            "实际整轮 wall{:>19.2} ms/轮",
            prof::ROUND_WALL.load(std::sync::atomic::Ordering::Relaxed) as f64
                / 1e6
                / completed_rounds as f64
        );
        prof::report_hist(completed_rounds);
        prof::report_partition(completed_rounds);
        prof::report_gpu(completed_rounds);
    }
    for cb in callbacks.iter_mut() {
        cb.after_training();
    }
    // 早停的结果记进模型:导出时进 attributes,和 XGBoost 一个位置。
    // 注意**预测默认仍然用全部树**(XGBoost 的 Booster 也是),
    // 要截断得显式调 predict_margin_upto。
    if let Some((iter, score)) = callbacks.iter().find_map(|cb| cb.best_iteration()) {
        model.best_iteration = Some(iter);
        model.best_score = Some(score);
    }

    Ok(model)
}

/// GPU characterization 的最小端到端闭环:流式处理每个列块,构建一棵树。
///
/// 这不是训练循环的最终接线(暂不支持多轮/多层),但它刻意不把量化数据
/// 常驻显存:每个 block 在 `with_block` 闭包中 H2D、GPU histogram、D2H 后
/// 立即释放。用于先验收模型位精确和“数据可超过显存”的调度边界。
#[cfg(feature = "cuda")]
pub fn train_gpu_one_tree(
    device_ordinal: usize,
    source: &dyn BlockSource,
    labels: &[f32],
    cfg: &TrainConfig,
    comm: &dyn Comm,
) -> anyhow::Result<Model> {
    use crate::backend::cuda::build_histogram_shared;
    if labels.len() != source.n_rows() {
        anyhow::bail!("labels 长度与 source 行数不符");
    }
    let margin = base_margin(cfg.objective, cfg.base_score);
    let predictions = vec![margin; source.n_rows()];
    let gpair = compute_gradients(&predictions, labels, cfg.objective);
    let quantizer = GradQuantizer::new(&gpair, source.n_rows());
    let fixed: Vec<GradPairFixed> = gpair.iter().map(|&g| quantizer.to_fixed(g)).collect();
    let root_sum = hist::node_sum(&fixed, None);
    let mut tree = Tree::default();
    let root = tree.push_leaf(
        leaf_of(root_sum, &quantizer, &cfg.params),
        hess_of(root_sum, &quantizer),
    );

    let cuts = source.cuts();
    let mut best: Option<SplitCandidate> = None;
    for b in 0..source.n_blocks() {
        let block_best = source.with_block_ret(b, |block| {
            let offsets = cuts.block_hist_offsets(&block.feat_ids);
            let hist = build_histogram_shared(
                device_ordinal,
                &block.data,
                block.n_rows,
                block.n_feats(),
                &offsets,
                &fixed,
                None,
            )?;
            Ok::<_, anyhow::Error>(split::best_split_in_block(
                &Histogram { bins: hist },
                &block.feat_ids,
                &offsets,
                root_sum,
                &quantizer,
                &cfg.params,
                None,
            ))
        })??;
        best = split::reduce_best(&[best, block_best]);
    }

    let Some(chosen) = comm.allreduce_best_split(best) else {
        return Ok(Model {
        // 训练层没有列名概念;由知道 schema 的入口层填。
        feature_names: Vec::new(),
        schema_mode: None,
            trees: vec![tree],
            base_score: cfg.base_score,
            objective: cfg.objective,
            n_features: source.n_features(),
            best_iteration: None,
            best_score: None,
        });
    };
    let (left_node, right_node) = tree.split_leaf(
        root,
        chosen.feat,
        tree::split_threshold(cuts, chosen.feat, chosen.bin),
        chosen.missing_left,
        chosen.gain,
        leaf_of(chosen.left, &quantizer, &cfg.params),
        hess_of(chosen.left, &quantizer),
        leaf_of(chosen.right, &quantizer, &cfg.params),
        hess_of(chosen.right, &quantizer),
    );
    // 选中的列也按需读取;GPU partition 只接收这一个 column,不把整块
    // 常驻。结果用于闭环验收,后续多层接线会把它挂到 LevelNode。
    let col = source.feature_column(chosen.feat)?;
    let rows: Vec<RowId> = (0..source.n_rows() as RowId).collect();
    let (left_rows, right_rows) = crate::backend::cuda::partition_rows_gpu(
        device_ordinal,
        &col,
        source.n_rows(),
        0,
        &rows,
        chosen.bin,
        chosen.missing_left,
        512,
    )?;
    anyhow::ensure!(left_rows.len() + right_rows.len() == source.n_rows());
    anyhow::ensure!(left_node > root && right_node > root);
    Ok(Model {
        // 训练层没有列名概念;由知道 schema 的入口层填。
        feature_names: Vec::new(),
        schema_mode: None,
        trees: vec![tree],
        base_score: cfg.base_score,
        objective: cfg.objective,
        n_features: source.n_features(),
        best_iteration: None,
        best_score: None,
    })
}

/// 在每个验证集上算每个指标。
///
/// 预测值是外面增量维护的(每轮只累加新树),这里只负责套指标。
/// 顺序是「eval 外层、metric 内层」,所以 `RoundMetrics::last()` 拿到的
/// 是**最后一个 eval 的最后一个 metric**;early stopping 盯的是
/// 最后一个 eval 的**第一个** metric,由 `first_of_last_eval` 负责。
fn evaluate(
    eval_preds: &[Vec<f32>],
    evals: &[EvalSet],
    round: usize,
    cfg: &TrainConfig,
) -> RoundMetrics {
    let mut entries = Vec::new();
    for (ev, pred) in evals.iter().zip(eval_preds) {
        for m in &cfg.eval_metrics {
            entries.push((
                ev.name.clone(),
                m.name().to_string(),
                m.eval(pred, ev.labels),
            ));
        }
    }
    RoundMetrics { round, entries }
}

fn predict_on(model: &Model, source: &dyn BlockSource) -> anyhow::Result<Vec<f32>> {
    // 返回的是 raw margin(链接函数之前)。metrics.rs 里的 logloss
    // 自己会过 sigmoid,这里再变换一次就成了 sigmoid(sigmoid(x))。
    let mut pred = vec![base_margin(model.objective, model.base_score); source.n_rows()];
    for t in &model.trees {
        for (p, v) in pred.iter_mut().zip(leaf_values(t, source)?) {
            *p += v;
        }
    }
    Ok(pred)
}

/// 每行走到的叶子值(已含 learning_rate)。
///
/// `predict_on` 和 `update_predictions` 共用这一份下降逻辑。**两处
/// 必须完全一致** —— 各写一遍的话,训练时累加的 pred 和验证时算的
/// pred 会悄悄地不是同一个东西,而且只在缺失值或边界 bin 上差一点。
///
/// 手里只有量化后的 bin,所以先把每个节点的浮点阈值换回 bin 右边界
/// (`tree::split_bin`),再按 bin 比。
///
/// **逐层推进,一次只驻留一个列块。** 原来是先把所有块拿到手再逐行
/// 走树 —— 那样按需加载就白做了,预测这一步会把整个数据集拉回内存。
/// 现在每加载一个块,就把"当前节点的分裂特征正好在这个块里"的行尽量
/// 往下推;一个块能连着推几层就推几层。最多扫 max_depth 遍块即可全部
/// 落到叶子(最坏情况每遍每行只前进一层)。
fn leaf_values(t: &Tree, source: &dyn BlockSource) -> anyhow::Result<Vec<f32>> {
    let cuts = source.cuts();
    let n_rows = source.n_rows();

    let thresholds: Vec<u32> = t
        .nodes
        .iter()
        .map(|n| {
            if n.is_leaf {
                0
            } else {
                tree::split_bin(cuts, n.feat, n.split_cond)
            }
        })
        .collect();

    // 特征 → 它在哪个块。分桶要用。
    let mut block_of = vec![usize::MAX; source.n_features()];
    let mut block_feats = Vec::with_capacity(source.n_blocks());
    for b in 0..source.n_blocks() {
        let feats = source.block_features(b)?;
        for &f in &feats {
            block_of[f as usize] = b;
        }
        block_feats.push(feats);
    }

    // 每行当前停在哪个节点
    let mut at = vec![0u32; n_rows];
    let mut done = t.nodes[0].is_leaf;

    // 遍数的上界是**树的深度**:一遍之内每行都会走到它需要的那个块,
    // 所以至少往下一层。
    fn depth_of(t: &Tree, node: usize) -> usize {
        let n = &t.nodes[node];
        if n.is_leaf {
            0
        } else {
            1 + depth_of(t, n.left as usize).max(depth_of(t, n.right as usize))
        }
    }
    let max_passes = depth_of(t, 0);
    let mut bucket: Vec<Vec<RowId>> = vec![Vec::new(); source.n_blocks()];

    for _pass in 0..=max_passes {
        if done {
            break;
        }
        // 先分桶:每行按「它当前需要哪个块」归位,一次 O(行数)。
        //
        // 原来是「每个块都把所有行扫一遍」,复杂度 O(深度 × 块数 × 行数)
        // —— 14 个块时 8.8 亿次行访问,其中绝大多数是「这个块里没有我
        // 要的特征」然后立刻退出。分桶之后每行每趟只被碰一次,
        // **块数这个维度就没了**,和只有一个块时一样。
        for b in bucket.iter_mut() {
            b.clear();
        }
        for (row, &node) in at.iter().enumerate() {
            let n = &t.nodes[node as usize];
            if !n.is_leaf {
                bucket[block_of[n.feat as usize]].push(row as RowId);
            }
        }

        for b in 0..source.n_blocks() {
            if bucket[b].is_empty() {
                continue; // 没有行等这个块,就根本不加载它
            }
            source.with_block_ret(b, |block| -> anyhow::Result<()> {
                anyhow::ensure!(
                    block.n_rows == n_rows,
                    "预测时列块 {b} 有 {} 行,数据源声明 {n_rows} 行",
                    block.n_rows
                );
                anyhow::ensure!(
                    block.data.len() == block.n_rows.saturating_mul(block.n_feats()),
                    "预测时列块 {b} 数据长度与行数 × 特征数不一致"
                );
                anyhow::ensure!(
                    block.feat_ids == block_feats[b],
                    "预测时列块 {b} 的实际特征与 block_features 元数据不一致"
                );
                let mut local_of = vec![usize::MAX; source.n_features()];
                for (local, &f) in block.feat_ids.iter().enumerate() {
                    local_of[f as usize] = local;
                }
                for &row in &bucket[b] {
                    let node = &mut at[row as usize];
                    // 父子的分裂特征常常在同一块,能连着推几层就推几层
                    loop {
                        let n = &t.nodes[*node as usize];
                        if n.is_leaf {
                            break;
                        }
                        let local = local_of[n.feat as usize];
                        if local == usize::MAX {
                            break;
                        }
                        let bin = block.column(local)[row as usize];
                        let go_left = if bin == MISSING_BIN {
                            n.default_left
                        } else {
                            (bin as u32) < thresholds[*node as usize]
                        };
                        *node = if go_left {
                            n.left as u32
                        } else {
                            n.right as u32
                        };
                    }
                }
                Ok(())
            })??;
        }
        done = at.iter().all(|&n| t.nodes[n as usize].is_leaf);
    }
    anyhow::ensure!(done, "有行没走到叶子,树的结构不对");

    Ok(at.iter().map(|&n| t.nodes[n as usize].leaf_value).collect())
}

/// GPU 行重分区的 block 大小。512 是 1M 行粗扫里最快的一档
/// (见 CLAUDE.md「GPU 行重分区」),和 CPU 的 `cols_per_block` 无关。
const GPU_PARTITION_THREADS: u32 = 512;

/// 本层里的一个活跃节点。
struct LevelNode {
    /// 在 tree.nodes 里的下标
    node: usize,
    /// 该节点持有的行
    rows: NodeRows,
    /// 梯度总和,**含缺失行** —— split.rs 靠它反推缺失部分
    sum: GradPairFixed,
    hist_from: HistFrom,
}

enum NodeRows {
    Host(Vec<RowId>),
    Device(crate::backend::cuda::GpuRowSpan),
}

impl NodeRows {
    fn len(&self) -> usize {
        match self {
            Self::Host(rows) => rows.len(),
            Self::Device(span) => span.len,
        }
    }

    fn host(&self) -> &[RowId] {
        match self {
            Self::Host(rows) => rows,
            Self::Device(_) => panic!("device rows 被送进 CPU 路径"),
        }
    }

    fn device(&self) -> crate::backend::cuda::GpuRowSpan {
        match self {
            Self::Device(span) => *span,
            Self::Host(_) => panic!("host rows 被送进 GPU 路径"),
        }
    }
}

/// 这个节点的直方图怎么来。
enum HistFrom {
    /// 老老实实按行累加
    Accumulate,
    /// 父 - 兄弟。GBDT 最重要的常数优化:一对兄弟里只有行少的那个
    /// 真的去扫数据,另一个是一次减法。
    ///
    /// `parent` 是上一层的槽位,`sibling` 是本层的槽位。
    Subtract { parent: usize, sibling: usize },
}

fn build_tree(
    source: &dyn BlockSource,
    gpair: &[GradPairFixed],
    // GPU device-quantization 路径上 root 的 node_sum 已经在 device 上归约好,
    // host 那份定点数组因此不再物化;CPU 路径传 None,照常从 `gpair` 求。
    dev_root_sum: Option<GradPairFixed>,
    q: &GradQuantizer,
    predictions: &mut [f32],
    cfg: &TrainConfig,
    comm: &dyn Comm,
    gpu: Option<&crate::backend::cuda::GpuTrainCtx>,
    // 本棵树的采样特征集(全局 id,升序)。`None` = 全选。
    sampled: Option<&[FeatId]>,
) -> anyhow::Result<Tree> {
    let p = &cfg.params;

    let cuts = source.cuts();
    let n_blocks_total = source.n_blocks();
    // 两种身份**必须分开**,不能互相推导:
    //
    // - `physical`:source / cache / resident 缓存里的**原始块号**。
    //   resident slot 就是按它缓存的。
    // - 压实块:当前这棵树的**临时物化**,只含被采样的列。它的内容随
    //   每棵树的采样集变化,所以**绝不能**进 resident 缓存,也绝不能拿
    //   物理块号当作它的"内容身份"。
    //
    // ⚠️ 上一版用「过滤后的位置」去推物理块号,结果 resident slot 认错块
    // (轻则退回整列上传,重则读到别的块的数据)。所以这里把三套下标
    // **显式带在一起**,不再靠位置推导。
    struct BlockPlan {
        /// source / cache / resident 缓存里的原始块号。
        physical: usize,
        /// 本块里**被采样中**的全局特征 id(升序)。
        sampled_global: Vec<FeatId>,
        /// 上面那些特征在**原始块**里的局部下标 —— partition 查 resident
        /// 列时用的是这个,和压实与否无关。
        orig_local: Vec<usize>,
        /// 这个块**实际会交回来**的特征列表:压实时是采样子集,
        /// 否则是整块。offsets 和那条一致性断言都按它来。
        ///
        /// ⚠️ 曾经把它和 `sampled_global` 合成一个字段,结果不支持部分读的
        /// source 交回整块、而 offsets 按子集算,直接对不上 —— 断言抓到了。
        handed_feats: Vec<FeatId>,
        /// 直方图 offsets,按**实际交给 histogram 的那个块**算。
        offsets: Vec<u32>,
        /// 是否物化成压实块(只读选中列)。resident 块恒为 false。
        compact: bool,
    }
    // 只把"每块有哪些特征"这张表留着(几百个 u32),块本身按需加载。
    let all_block_feats: Vec<Vec<FeatId>> = (0..n_blocks_total)
        .map(|i| source.block_features(i))
        .collect::<anyhow::Result<_>>()?;

    // **colsample 的关键一步:先算出哪些块物理上需要,再决定读什么。**
    //
    // 没有任何选中特征的块**根本不进入下面的循环** —— 不读、不上传、
    // 不转置、不建直方图。这就是「采样必须减少真实字节」的落点:
    // 放到 histogram kernel 里再跳过特征的话,前面那些字节已经花掉了
    // (XGBoost 就是那样:采样只作用于 split 枚举)。
    // GPU resident 的块保持原样(整块常驻,采样只影响 compute);
    // 其余块允许按采样列压实,从而在 source 侧就少读。
    let resident_upto = gpu.map_or(0, |c| c.resident_block_count());
    let supports_cols = source.supports_column_selection();
    let plans: Vec<BlockPlan> = match sampled {
        Some(sel) => crate::colsample::required_blocks(sel, &all_block_feats)
            .into_iter()
            .map(|(physical, orig_local)| {
                // 只有 source 真的支持部分读、块又不常驻,压实才有意义。
                // 不支持的 source 会交回整块,这时按压实算的 offsets 会对不上。
                let compact = supports_cols
                    && physical >= resident_upto
                    && orig_local.len() < all_block_feats[physical].len();
                let sampled_global: Vec<FeatId> = orig_local
                    .iter()
                    .map(|&l| all_block_feats[physical][l])
                    .collect();
                let handed_feats = if compact {
                    sampled_global.clone()
                } else {
                    all_block_feats[physical].clone()
                };
                let offsets = cuts.block_hist_offsets(&handed_feats);
                BlockPlan {
                    physical,
                    sampled_global,
                    orig_local,
                    handed_feats,
                    offsets,
                    compact,
                }
            })
            .collect(),
        None => (0..n_blocks_total)
            .map(|physical| BlockPlan {
                physical,
                sampled_global: all_block_feats[physical].clone(),
                orig_local: (0..all_block_feats[physical].len()).collect(),
                handed_feats: all_block_feats[physical].clone(),
                offsets: cuts.block_hist_offsets(&all_block_feats[physical]),
                compact: false,
            })
            .collect(),
    };
    let n_blocks = plans.len();

    let mut t = Tree::default();
    let root_sum = dev_root_sum.unwrap_or_else(|| hist::node_sum(gpair, None));
    let root = t.push_leaf(leaf_of(root_sum, q, p), hess_of(root_sum, q));

    let root_rows = match gpu {
        Some(ctx) => NodeRows::Device(ctx.begin_tree_rows()?),
        None => NodeRows::Host((0..source.n_rows() as RowId).collect()),
    };
    let mut level = vec![LevelNode {
        node: root,
        rows: root_rows,
        sum: root_sum,
        hist_from: HistFrom::Accumulate,
    }];
    // 上一层的直方图,[列块][槽位]。兄弟相减要用,所以整层都得留着。
    let mut parent_hists: Vec<Vec<Histogram>> = Vec::new();

    for depth in 0..p.max_depth {
        if level.is_empty() {
            break;
        }
        // ★ 外层列块、内层节点。反过来写(外层节点、内层列块)每个
        // 节点都要重新过一遍所有列块,传输量从 O(数据) 变成
        // O(节点数 × 数据) —— 整个方案就没意义了。改这里要特别小心。
        //
        // 并行维度就是列块本身(internal-docs/ARCHITECTURE.md「切分维度 = 并行维度」):
        // 每个块算自己那组特征的直方图、枚举自己的候选,块之间不共享
        // 任何可变状态,所以 par_iter 直接就能用。阶段 4 换成多卡时,
        // 这一层换成"每卡一组列 + 候选 allreduce",训练逻辑不用动。
        //
        // 并行度上限 = 块数 = ceil(n_features / cols_per_block)。列少的
        // 时候调小 cols_per_block 才吃得满核。
        // 父直方图**按块交给各线程**,用完就地释放,不再整层留着。
        // `into_par_iter` 让每个线程拿到自己那块的所有权,不用加锁。
        let parents: Vec<Option<Vec<Histogram>>> = if parent_hists.is_empty() {
            (0..n_blocks).map(|_| None).collect()
        } else {
            std::mem::take(&mut parent_hists)
                .into_iter()
                .map(Some)
                .collect()
        };

        // 真实墙钟:整个块级并行段进出各一次。**不要**用累加计时器去推它。
        let _phase = prof::Timer::new(&prof::HIST_PHASE_WALL);
        let per_block: anyhow::Result<Vec<(Vec<Histogram>, Vec<Option<SplitCandidate>>)>> = parents
            .into_par_iter()
            .enumerate()
            .map(|(b, mut parent_hist)| {
                // `b` 只是 plan 数组里的位置,**不承载任何身份**。
                // 物理块号、原始局部下标、压实后的 offsets 全部从 plan 里取。
                // `b` 只是 plan 数组里的位置,**不承载任何身份**。
                // 物理块号、原始局部下标、offsets 全部从 plan 里显式取。
                let plan = &plans[b];
                let phys = plan.physical;
                let run = |block: &ColumnBlock| -> anyhow::Result<_> {
                    anyhow::ensure!(
                        block.n_rows == source.n_rows(),
                        "列块 {phys} 有 {} 行,数据源声明 {} 行",
                        block.n_rows,
                        source.n_rows()
                    );
                    anyhow::ensure!(
                        block.feat_ids == plan.handed_feats,
                        "列块 {phys} 交回的特征与 plan 不一致"
                    );
                    anyhow::ensure!(
                        block.data.len() == block.n_rows.saturating_mul(block.n_feats()),
                        "列块 {b} 数据长度与行数 × 特征数不一致"
                    );
                    let width = *plan.offsets.last().unwrap() as usize;
                    // 要相减的槽位先放个空壳,缓冲在下面 subtract 那一趟才分配。
                    // 原来这里给每个槽位都分配 + 清零一个完整直方图,相减节点
                    // 那一半分配完立刻被扔掉。
                    //
                    // 老实说:实测**量不出差别**(这台机器上噪声比效果大)。
                    // 留着是因为它确实少干一份活,不是因为它更快。
                    let mut per_node: Vec<Histogram> = level
                        .iter()
                        .map(|ln| match ln.hist_from {
                            HistFrom::Accumulate => Histogram::zeros(width),
                            HistFrom::Subtract { .. } => Histogram::zeros(0),
                        })
                        .collect();

                    // 先算要实际累加的那些
                    if let Some(ctx) = gpu {
                        // 量化列块 H2D **一次**,本块里所有活跃节点共用它。
                        // 早期版本是每个节点重新上传一遍整块,那是纯浪费。
                        let _t = prof::Timer::new(&prof::HIST);
                        let started = std::time::Instant::now();
                // ⚠️ **必须用物理块号**。resident 缓存按物理块号索引,而
                // `b` 是 colsample 过滤之后的位置 —— 传 `b` 会把 slot 认错:
                // partition 那边用 `block_ids[b]` 查,两边对不上,轻则
                // resident 命中失败退回整列上传(实测 wide 上出现
                // `PART_COL_SOURCE=upload`),重则不同轮次的采样集让同一个
                // 位置映射到不同物理块,直接读到别的块的数据。
                let (hist_data, hist_rows) = (&block.data[..], block.n_rows);
                let mut session = ctx.block_session(
                    phys,
                    hist_data,
                            hist_rows,
                            block.n_feats(),
                            &plan.offsets,
                        )?;
                        // 交给 kernel 的位图,按**这个块实际交回来的特征**算。
                        //
                        // - 压实块:每一列都是选中的 → 全 1;
                        // - resident 整块:采样只影响 compute,这里才真正
                        //   起到跳过未选中特征的作用。
                        let feat_mask: u32 = match sampled {
                            Some(_) if plan.compact => !0u32,
                            Some(sel) => {
                                let mut m = 0u32;
                                for (l, f) in block.feat_ids.iter().enumerate() {
                                    if l < 32 && sel.binary_search(f).is_ok() {
                                        m |= 1u32 << l;
                                    }
                                }
                                m
                            }
                            None => !0u32,
                        };
                        // 整层的 Accumulate 节点按容量分批合并 launch。
                        // 容量 1 就是原来的逐节点路径,一个字都没变。
                        let cap = session.hist_nodes_batch_capacity();
                        // 行主序 kernel 只有 batched 一个入口,批里只有一个
                        // 节点也必须走它。
                        let must_batch = session.requires_batched_hist();
                        let todo: Vec<usize> = level
                            .iter()
                            .enumerate()
                            .filter(|(_, ln)| matches!(ln.hist_from, HistFrom::Accumulate))
                            .map(|(i, _)| i)
                            .collect();
                        if cap <= 1 && !must_batch {
                            for &i in &todo {
                                session.histogram_device_at_depth(
                                    level[i].rows.device(),
                                    &mut per_node[i].bins,
                                    depth,
                                    i,
                                )?;
                            }
                        } else {
                            for chunk in todo.chunks(cap.max(1)) {
                                let spans: Vec<_> =
                                    chunk.iter().map(|&i| level[i].rows.device()).collect();
                                // 借出这一批的输出槽位。`per_node` 的其它
                                // 元素不参与,所以按下标取可变引用是安全的,
                                // 但借用检查器看不出来 —— 用 split 逐个取。
                                let mut outs: Vec<&mut Vec<GradPairFixed>> =
                                    Vec::with_capacity(chunk.len());
                                let mut rest = &mut per_node[..];
                                let mut taken = 0usize;
                                for &i in chunk {
                                    let (_, tail) = rest.split_at_mut(i - taken);
                                    let (head, tail) = tail.split_at_mut(1);
                                    outs.push(&mut head[0].bins);
                                    rest = tail;
                                    taken = i + 1;
                                }
                                session.histogram_device_batch(&spans, &mut outs, feat_mask)?;
                            }
                        }
                        let timing = session.timing();
                        drop(session);
                        prof::record_gpu(
                            &timing,
                            started.elapsed().as_nanos() as u64,
                            &prof::GPU_HIST_BLOCK_H2D,
                            &prof::GPU_HIST_ROWS_H2D,
                            &prof::GPU_HIST_KERNEL,
                            &prof::GPU_HIST_D2H,
                            &prof::GPU_HIST_HOST,
                        );
                    } else {
                        for (i, ln) in level.iter().enumerate() {
                            if matches!(ln.hist_from, HistFrom::Accumulate) {
                                // 根节点走 None 那条路:少一层行下标的间接寻址,
                                // 结果和传全部行完全一样(hist.rs 有断言钉着)。
                                let rows = if depth == 0 {
                                    None
                                } else {
                                    Some(ln.rows.host())
                                };
                                let _t = prof::Timer::new(&prof::HIST);
                                hist::build(
                                    &block.data,
                                    block.n_rows,
                                    block.n_feats(),
                                    &plan.offsets,
                                    gpair,
                                    rows,
                                    &mut per_node[i].bins,
                                );
                            }
                        }
                    }
                    // 再用父 - 兄弟补上剩下的。分两趟是为了不依赖兄弟在本层
                    // 里的先后顺序。
                    for i in 0..level.len() {
                        if let HistFrom::Subtract { parent, sibling } = level[i].hist_from {
                            let ph = parent_hist
                                .as_mut()
                                .expect("第一层没有父亲,不该走到相减这条路");
                            let _t = prof::Timer::new(&prof::SUBTRACT);
                            // 父的存储**搬**给大 child,原地减去兄弟。
                            //
                            // 这个线程通过 `into_par_iter` **拥有**本块的父
                            // 直方图,而一个父只会被它唯一的 `Subtract` 子节点
                            // 认领一次,所以可以直接取走 —— 不再分配一份新的
                            // 再清零(那份清零紧接着就被整份覆写,是纯白写)。
                            let mut out = std::mem::replace(&mut ph[parent], Histogram::zeros(0));
                            // ⚠️ 认领两次会在这里当场炸,而不是悄悄返回零直方图:
                            // 取走之后原位留下的是长度 0 的壳。
                            assert_eq!(
                                out.len(),
                                width,
                                "父直方图 {parent} 被认领了不止一次(或宽度不符)"
                            );
                            hist::subtract_in_place(&mut out, &per_node[sibling]);
                            per_node[i] = out;
                        }
                    }

                    let _t = prof::Timer::new(&prof::ENUM);
                    let cands: Vec<Option<SplitCandidate>> = level
                        .iter()
                        .enumerate()
                        .map(|(i, ln)| {
                            split::best_split_in_block(
                                &per_node[i],
                                &block.feat_ids,
                                &plan.offsets,
                                ln.sum,
                                q,
                                p,
                                sampled,
                            )
                        })
                        .collect();
                    Ok((per_node, cands))
                };
                // 压实块只读选中的列(source 侧就少读);resident 块整块借出,
                // 采样只影响 compute。**两条路都用 `plan.physical` 寻址**,
                // 绝不从数组位置反推身份。
                if plan.compact {
                    source.with_block_cols_ret(phys, &plan.orig_local, run)?
                } else {
                    source.with_block_ret(phys, run)?
                }
                // 本块的父直方图到此为止 —— 闭包一结束,它和刚借来的
                // 块一起释放,这一层内存最紧的时候正好都不在手上。
            })
            .collect();
        let per_block = per_block?;
        drop(_phase);

        // 跨块归约按**块序**做,和单线程逐块推进完全一样。
        //
        // 线程数不影响结果,靠的是两件事叠起来:块内是定点整数累加
        // (与顺序无关),块间的归约顺序在这里固定死。少了任何一件,
        // 并列的候选就会因为"哪个线程先跑完"而胜负不同 ——
        // tests/bit_exact_vs_xgboost.rs 盯着这条。
        let mut best: Vec<Option<SplitCandidate>> = vec![None; level.len()];
        let mut hists: Vec<Vec<Histogram>> = Vec::with_capacity(n_blocks);
        for (per_node, cands) in per_block {
            for (i, cand) in cands.into_iter().enumerate() {
                best[i] = split::reduce_best(&[best[i], cand]);
            }
            hists.push(per_node);
        }

        // 多机/多卡时各方只看得到自己那几列,全局最优要在这里汇总。
        // 单机 Local 是恒等,但训练逻辑里不能直接挑本地最优 ——
        // 那样换个 Comm 就不对了。
        let chosen: Vec<Option<SplitCandidate>> = best
            .into_iter()
            .map(|c| comm.allreduce_best_split(c))
            .collect();

        // ---- 第二趟:行重分区 ----
        //
        // 循环顺序和直方图那趟一样:**外层数据、内层节点**。一个节点
        // 一个节点地去取数据,就退化成 O(节点数 × 数据) —— 深度 6 的
        // 末层有 64 个节点,而特征总共才几十上百列。
        //
        // 为什么必须是**两趟**:重分区要先知道选中了哪个特征,而那要等
        // 所有块的直方图都算完、全局归约出最优分裂之后。所以一层之内
        // 至少两趟,合并不了。
        //
        // 但第二趟只需要**选中的那几列**,不是整块:一层里不同节点选中的
        // 特征往往重复,去重之后通常只有个位数列。所以传输量是
        // O(深度 × 数据) + 一点零头,不是 O(2 × 深度 × 数据)。
        let mut partitions: Vec<Option<(NodeRows, NodeRows)>> =
            (0..level.len()).map(|_| None).collect();
        // GPU 路径上 partition 只 launch 不同步,先拿登记号;整层结束后
        // 一次 D2H 把所有节点的左右行数取回来再填 `partitions`。
        let mut part_tokens: Vec<Option<usize>> = vec![None; level.len()];
        let mut by_feat: std::collections::BTreeMap<FeatId, Vec<usize>> = Default::default();
        for (i, c) in chosen.iter().enumerate() {
            if let Some(c) = c {
                by_feat.entry(c.feat).or_default().push(i);
            }
        }
        for (feat, node_idxs) in by_feat {
            // resident 命中时,这一列的 bin 已经在 device 上了。**必须在取列
            // 之前就判断**,否则 host 那份 `Vec<Bin>`(每次 O(行数))照样会被
            // 物化出来 —— 实测 HIGGS 上 PART_FETCH 25.9 ms/轮 + 列 H2D
            // 52.1 ms/轮,合计占整轮 20%,而且传的是已经在卡上的字节。
            let resident_session = match gpu {
                Some(ctx) => {
                    // 同样:`block_feats` 是过滤后的,resident 缓存按物理
                    // 块号索引,所以这里要换回 `block_ids[b]`。
                    // ⚠️ 这里要的是**原始块里的局部下标**,和压实与否无关 ——
                    // resident 缓存按物理块号 + 原始局部下标寻址。
                    // 压实块的物理块号 >= resident_block_count,
                    // `partition_session_resident` 会直接返回 None,
                    // 所以两条身份不会串。
                    let loc = plans.iter().find_map(|pl| {
                        pl.sampled_global
                            .iter()
                            .position(|&f| f == feat)
                            .map(|i| (pl.physical, pl.orig_local[i]))
                    });
                    match loc {
                        Some((b, local)) => ctx.partition_session_resident(b, local)?,
                        None => None,
                    }
                }
                None => None,
            };
            if let Some(mut session) = resident_session {
                let started = std::time::Instant::now();
                for i in node_idxs {
                    let c = chosen[i].expect("by_feat 里只放了有候选的节点");
                    // 只登记,不同步:左右行数整层结束后一次取回。
                    part_tokens[i] =
                        Some(session.partition(level[i].rows.device(), c.bin, c.missing_left)?);
                }
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
                continue;
            }
            let t = prof::Timer::new(&prof::PART_FETCH);
            let col = source.feature_column(feat)?;
            drop(t);
            if prof::enabled() {
                use std::sync::atomic::Ordering;
                prof::PART_COLS.fetch_add(1, Ordering::Relaxed);
                prof::PART_ROWS.fetch_add(
                    node_idxs
                        .iter()
                        .map(|&i| level[i].rows.len() as u64)
                        .sum::<u64>(),
                    Ordering::Relaxed,
                );
            }
            if let Some(ctx) = gpu {
                // 选中的这一列 H2D **一次**,本层所有选中它的节点共用 ——
                // 和「外层特征、内层节点」的循环顺序天然对齐。
                let started = std::time::Instant::now();
                let mut session = ctx.partition_session(&col)?;
                for i in node_idxs {
                    let c = chosen[i].expect("by_feat 里只放了有候选的节点");
                    part_tokens[i] =
                        Some(session.partition(level[i].rows.device(), c.bin, c.missing_left)?);
                }
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
            } else {
                for i in node_idxs {
                    let c = chosen[i].expect("by_feat 里只放了有候选的节点");
                    let (left, right) = partition_rows(level[i].rows.host(), &col, &c, comm);
                    partitions[i] = Some((NodeRows::Host(left), NodeRows::Host(right)));
                }
            }
        }

        if let Some(ctx) = gpu {
            let started = std::time::Instant::now();
            let resolved = ctx.resolve_partitions()?;
            if prof::enabled() {
                prof::GPU_PART_D2H.fetch_add(
                    started.elapsed().as_nanos() as u64,
                    std::sync::atomic::Ordering::Relaxed,
                );
            }
            for (i, token) in part_tokens.iter().enumerate() {
                if let Some(t) = token {
                    let (left, right) = resolved[*t];
                    partitions[i] = Some((NodeRows::Device(left), NodeRows::Device(right)));
                }
            }
        }

        let _t = prof::Timer::new(&prof::NEXT_LEVEL);
        let mut next: Vec<LevelNode> = Vec::new();
        for (i, ln) in level.into_iter().enumerate() {
            let Some(c) = chosen[i] else {
                // 这个节点到此就是终叶。行分区已经是答案,现在直接累加
                // 比训练结束后再沿树下降一遍便宜得多。
                let value = t.nodes[ln.node].leaf_value;
                if let Some(ctx) = gpu {
                    ctx.assign_leaf_value(ln.rows.device(), value)?;
                } else {
                    let _t_pred = prof::Timer::new(&prof::UPDATE_PRED);
                    for &row in ln.rows.host() {
                        predictions[row as usize] += value;
                    }
                }
                continue; // 分不动就留成叶子,值在 push_leaf 时就写好了
            };

            let (left_node, right_node) = t.split_leaf(
                ln.node,
                c.feat,
                tree::split_threshold(cuts, c.feat, c.bin),
                c.missing_left,
                c.gain,
                leaf_of(c.left, q, p),
                hess_of(c.left, q),
                leaf_of(c.right, q, p),
                hess_of(c.right, q),
            );

            let (left_rows, right_rows) = partitions[i].take().expect("上面按块分好了");

            // 行少的那个真算,另一个用父 - 兄弟。省下来的是整个训练
            // 里最大的一块常数。
            let (small, big) = if left_rows.len() <= right_rows.len() {
                (0, 1)
            } else {
                (1, 0)
            };
            let slot = next.len();
            let mut pair = [
                LevelNode {
                    node: left_node,
                    rows: left_rows,
                    sum: c.left,
                    hist_from: HistFrom::Accumulate,
                },
                LevelNode {
                    node: right_node,
                    rows: right_rows,
                    sum: c.right,
                    hist_from: HistFrom::Accumulate,
                },
            ];
            pair[big].hist_from = HistFrom::Subtract {
                parent: i,
                sibling: slot + small,
            };
            next.extend(pair);
        }

        drop(_t);
        parent_hists = hists;
        if let Some(ctx) = gpu {
            ctx.finish_partition_level()?;
        }
        level = next;
    }

    // max_depth 用尽时,最后一次分裂产生的孩子还留在 level 里;
    // max_depth == 0 时这里处理根叶子。其它提前停下的叶子已在上面写过。
    {
        let _t = prof::Timer::new(&prof::UPDATE_PRED);
        for ln in level {
            debug_assert!(t.nodes[ln.node].is_leaf);
            let value = t.nodes[ln.node].leaf_value;
            if let Some(ctx) = gpu {
                ctx.assign_leaf_value(ln.rows.device(), value)?;
            } else if let NodeRows::Host(rows) = ln.rows {
                for row in rows {
                    predictions[row as usize] += value;
                }
            }
        }
        // `gpu_math = "fast"` 下 delta 由 device 上的 gradient kernel 直接
        // 消费,host 永远不需要看它 —— 这一次 4 B/行 的 D2H 整个消失。
        // 漏掉这个判断的话 fast 模式仍然每棵树付一次 40 MB D2H
        // (实测 16.5 ms/轮),收益被吃掉一半还多。
        // exact:host 是权威副本,delta 要交回去施加。
        // fast:device 是权威副本,delta 由 device 的 gradient kernel 直接
        // 消费,这一次 4 B/行 的 D2H 整个不存在。
        if let Some(ctx) =
            gpu.filter(|c| c.prediction_authority() == PredictionAuthority::Host)
        {
            // 只取回 delta;施加推迟到下一轮开头,和求梯度合并成一次遍历
            // (见 train_with_backend 里那一段)。
            let d2h_ms = ctx.download_prediction_delta()?;
            if prof::enabled() {
                prof::GPU_PRED_D2H.fetch_add(
                    (d2h_ms as f64 * 1e6) as u64,
                    std::sync::atomic::Ordering::Relaxed,
                );
                prof::GPU_PRED_D2H_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                prof::GPU_PRED_D2H_BYTES.fetch_add(
                    (predictions.len() * std::mem::size_of::<f32>()) as u64,
                    std::sync::atomic::Ordering::Relaxed,
                );
                // One f32 delta per training row is downloaded once per tree.
                // Applying it to `predictions` is intentionally kept on CPU
                // for now and remains visible in UPDATE_PRED / round wall.
            }
        }
    }

    Ok(t)
}

/// 叶子权重 × learning_rate。收缩在这里做,不在预测时 ——
/// 见 tree.rs 里 `Node::leaf_value` 的说明。
fn leaf_of(sum: GradPairFixed, q: &GradQuantizer, p: &TrainParams) -> f32 {
    split::leaf_weight(sum, q, p) * p.learning_rate
}

/// 节点的 hess,只为导出用(XGBoost 的 cover)。
fn hess_of(sum: GradPairFixed, q: &GradQuantizer) -> f32 {
    q.to_float(sum).hess
}

#[derive(Clone, Copy)]
struct PartitionOrder {
    left: [u8; 8],
    right: [u8; 8],
    n_left: u8,
}

/// 一个 byte 已经给出了 8 行的方向,预先把左右两边的稳定拷贝顺序
/// 排好。这样真正拆行下标时不再为每一行走一次不可预测分支。
fn partition_orders() -> &'static [PartitionOrder; 256] {
    use std::sync::OnceLock;
    static ORDERS: OnceLock<[PartitionOrder; 256]> = OnceLock::new();
    ORDERS.get_or_init(|| {
        std::array::from_fn(|mask| {
            let mut left = [0u8; 8];
            let mut right = [0u8; 8];
            let (mut nl, mut nr) = (0usize, 0usize);
            for j in 0..8u8 {
                if mask & (1usize << j) != 0 {
                    left[nl] = j;
                    nl += 1;
                } else {
                    right[nr] = j;
                    nr += 1;
                }
            }
            PartitionOrder {
                left,
                right,
                n_left: nl as u8,
            }
        })
    })
}

/// 按选中的分裂把行分到左右两边。
///
/// 分区用的是 **bin**(训练侧手里就是量化数据),和 `leaf_values` 里
/// 把阈值换回 bin 再比是同一套语义,`tree::split_bin` /
/// `split_threshold` 的往返测试钉着这一点。
fn partition_rows(
    rows: &[RowId],
    col: &[Bin],
    c: &SplitCandidate,
    comm: &dyn Comm,
) -> (Vec<RowId>, Vec<RowId>) {
    // 每行 1 bit。多卡时只有持有该特征的一方知道每行去哪边,这个
    // 结果必须广播出去 —— 这是主要通信开销,阶段 4 要单独 benchmark。
    // 单机 Local 是空操作,但走一遍这条路,换后端时训练逻辑不用动。
    let _t = prof::Timer::new(&prof::PART_BITS);
    let mut bits = vec![0u8; rows.len().div_ceil(8)];
    // 按**字节**攒够 8 个 bit 再写一次,而不是每行都对 bits[k/8] 做一次
    // 读-改-写。同样的结果,内存流量少 8 倍。
    let fill_byte = |byte: &mut u8, chunk: &[RowId]| {
        let mut acc = 0u8;
        for (j, &r) in chunk.iter().enumerate() {
            let bin = col[r as usize];
            let go_left = if bin == MISSING_BIN {
                c.missing_left
            } else {
                (bin as u32) < c.bin
            };
            acc |= (go_left as u8) << j;
        }
        *byte = acc;
    };
    if rows.len() >= PARTITION_MIN_ROWS_PER_SEGMENT {
        bits.par_iter_mut().enumerate().for_each(|(i, byte)| {
            let start = i * 8;
            fill_byte(byte, &rows[start..(start + 8).min(rows.len())]);
        });
    } else {
        for (byte, chunk) in bits.iter_mut().zip(rows.chunks(8)) {
            fill_byte(byte, chunk);
        }
    }
    comm.broadcast_partition(owner_rank(c.feat), &mut bits);
    drop(_t);

    let _t = prof::Timer::new(&prof::PART_SPLIT);
    let (left, right) = split_rows_by_prefix(rows, &bits);
    drop(_t);
    (left, right)
}

/// 小节点直接串行更便宜;大节点最多切成当前 pool 的线程数份。
const PARTITION_MIN_ROWS_PER_SEGMENT: usize = 16 * 1024;

/// 按已经广播好的方向 bit 做稳定分区。
///
/// 每段覆盖连续、按 byte 对齐的输入。先并行计数,再串行前缀和得到每段
/// 的左右写入区间,最后并行写不相交的输出切片。段内和段间都保持原顺序,
/// 所以结果和串行 stable partition 逐字节相同。
fn split_rows_by_prefix(rows: &[RowId], bits: &[u8]) -> (Vec<RowId>, Vec<RowId>) {
    debug_assert_eq!(bits.len(), rows.len().div_ceil(8));
    if rows.is_empty() {
        return (Vec::new(), Vec::new());
    }

    let max_segments = rayon::current_num_threads().min(bits.len());
    let n_segments = rows
        .len()
        .div_ceil(PARTITION_MIN_ROWS_PER_SEGMENT)
        .clamp(1, max_segments);
    let bytes_per_segment = bits.len().div_ceil(n_segments);
    let segments: Vec<(usize, usize)> = (0..bits.len())
        .step_by(bytes_per_segment)
        .map(|start| (start, (start + bytes_per_segment).min(bits.len())))
        .collect();

    let left_counts: Vec<usize> = segments
        .par_iter()
        .map(|&(start, end)| {
            bits[start..end]
                .iter()
                .map(|b| b.count_ones() as usize)
                .sum()
        })
        .collect();

    let mut left_offsets = Vec::with_capacity(segments.len() + 1);
    let mut right_offsets = Vec::with_capacity(segments.len() + 1);
    left_offsets.push(0usize);
    right_offsets.push(0usize);
    for (&(start_byte, end_byte), &n_left) in segments.iter().zip(&left_counts) {
        let start_row = start_byte * 8;
        let end_row = (end_byte * 8).min(rows.len());
        left_offsets.push(left_offsets.last().unwrap() + n_left);
        right_offsets.push(right_offsets.last().unwrap() + (end_row - start_row - n_left));
    }

    let mut left = vec![0 as RowId; *left_offsets.last().unwrap()];
    let mut right = vec![0 as RowId; *right_offsets.last().unwrap()];
    let left_parts = split_output_by_offsets(&mut left, &left_offsets);
    let right_parts = split_output_by_offsets(&mut right, &right_offsets);
    let orders = partition_orders();

    segments
        .into_par_iter()
        .zip(left_parts)
        .zip(right_parts)
        .for_each(|(((start_byte, end_byte), left_out), right_out)| {
            let (mut li, mut ri) = (0usize, 0usize);
            for byte in start_byte..end_byte {
                let start = byte * 8;
                let chunk = &rows[start..(start + 8).min(rows.len())];
                let mask = bits[byte];
                if chunk.len() == 8 {
                    let order = &orders[mask as usize];
                    for &j in &order.left[..order.n_left as usize] {
                        left_out[li] = chunk[j as usize];
                        li += 1;
                    }
                    for &j in &order.right[..8 - order.n_left as usize] {
                        right_out[ri] = chunk[j as usize];
                        ri += 1;
                    }
                } else {
                    // 只有全局最后一个 byte 可能不足 8 行。高位不是输入,
                    // 不能直接按完整查表写。
                    for (j, &row) in chunk.iter().enumerate() {
                        if mask & (1 << j) != 0 {
                            left_out[li] = row;
                            li += 1;
                        } else {
                            right_out[ri] = row;
                            ri += 1;
                        }
                    }
                }
            }
            debug_assert_eq!(li, left_out.len());
            debug_assert_eq!(ri, right_out.len());
        });

    (left, right)
}

/// 根据前缀 offset 把一个输出切成互不重叠的可变段。这样并行写阶段
/// 不需要锁,也不需要裸指针。
fn split_output_by_offsets<'a>(output: &'a mut [RowId], offsets: &[usize]) -> Vec<&'a mut [RowId]> {
    let mut parts = Vec::with_capacity(offsets.len().saturating_sub(1));
    let mut rest = output;
    for range in offsets.windows(2) {
        let len = range[1] - range[0];
        let (part, tail) = rest.split_at_mut(len);
        parts.push(part);
        rest = tail;
    }
    debug_assert!(rest.is_empty());
    parts
}

/// 哪个 rank 持有这个特征。单机永远是 0;阶段 4 换成列 → rank 的映射。
fn owner_rank(_feat: FeatId) -> usize {
    0
}

/// base_score 转成 raw margin —— 数值对齐时最常踩的坑之一。
///
/// 训练循环里累加的 predictions 是 raw margin(链接函数**之前**的值),
/// 但 base_score 和 XGBoost 一样是在变换**之后**的空间里给的。
/// 二分类忘了取 logit 的话,起点就从 0.0 变成 0.5,第一棵树的梯度
/// 全错,而且错得很小 —— 训得出来,就是对不上。
pub fn base_margin(obj: Objective, base_score: f32) -> f32 {
    match obj {
        Objective::SquaredError => base_score,
        // logit(0.5) = 0.0
        Objective::Logistic => {
            assert!(
                base_score > 0.0 && base_score < 1.0,
                "binary:logistic 的 base_score 是概率,必须在 (0, 1) 里,给的是 {base_score}"
            );
            (base_score / (1.0 - base_score)).ln()
        }
    }
}

/// 单行的梯度。`gradient_bounds` 和量化那一趟共用它,保证两遍算的
/// 是同一个函数 —— 各写一遍的话,求上界用的和真正累加的可能不一致,
/// 而那会让 scale 偏小、定点溢出。
#[inline]
fn gradient_of(pred: f32, label: f32, obj: Objective) -> GradPair {
    match obj {
        Objective::SquaredError => GradPair {
            grad: pred - label,
            hess: 1.0,
        },
        Objective::Logistic => {
            let s = 1.0 / (1.0 + (-pred).exp());
            GradPair {
                grad: s - label,
                hess: (s * (1.0 - s)).max(1e-16),
            }
        }
    }
}

/// 每行独立,没有归约,所以并行**不可能**改变任何一个元素的值 ——
/// `collect` 保序,第 i 个输出永远只由第 i 个输入决定。
///
/// 这一段和下面的量化是 Amdahl 分析里那部分「按行的串行工作」的主要成分:
/// 它们只随行数走、完全不进列块那一维,所以以前无论开多少线程都是单线程跑。
/// GPU 路径上尤其明显 —— HIGGS 1050 万行时 GRAD + QUANTIZE 合计约
/// 145 ms/轮,占整轮 0.345 s 的四成以上,而那时 GPU 在干等。
fn compute_gradients(pred: &[f32], label: &[f32], obj: Objective) -> Vec<GradPair> {
    pred.par_iter()
        .zip(label.par_iter())
        .map(|(&p, &y)| gradient_of(p, y, obj))
        .collect()
}

/// 把新加的这棵树累加到 raw margin 上。
///
/// 叶子值**已经含 learning_rate**(建树时就收缩好了,和 XGBoost 存
/// 模型的方式一致),所以这里只是加,不再乘 eta。
/// 把一棵**完整的树**施加到**全部行**上:每一行都从 root 重新 routing。
///
/// row-subsample 第二阶段需要它:建树只用采样行时,未采样的行必须在树建完
/// 之后从 root 重走一遍 —— **不能**从采样行的 partition 状态去推断它们落在
/// 哪个叶子(那些状态里根本没有它们)。
pub fn apply_tree_to_all_rows(
    pred: &mut [f32],
    tree: &Tree,
    source: &dyn BlockSource,
) -> anyhow::Result<()> {
    update_predictions(pred, tree, source)
}

fn update_predictions(
    pred: &mut [f32],
    tree: &Tree,
    source: &dyn BlockSource,
) -> anyhow::Result<()> {
    for (p, v) in pred.iter_mut().zip(leaf_values(tree, source)?) {
        *p += v;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    /// `max_depth` 的公开契约是 **1..=14**,两个后端同一个数。
    ///
    /// 上限卡在 GPU:`part_totals` 由 `MAX_LEVEL_NODES = 8192` 兜着,
    /// partition 跑第 `0..max_depth-1` 层,第 L 层最多 `2^L` 个节点,
    /// 所以要装 `2^(max_depth-1)` 个 —— depth 14 正好 8192,depth 15 要 16384。
    ///
    /// ⚠️ **CPU 自己没有这个限制**(实测 depth 17 正常),但契约取共同安全值:
    /// 否则 CPU 上能训的模型在 GPU 上复现不了,而 CPU/GPU 逐字节相同是
    /// 这个项目的核心不变量。实测 depth 15/16 在 GPU 上确实报
    /// 「一层的 partition 节点数超过 8192」。
    #[test]
    fn max_depth_contract_is_one_to_fourteen() {
        assert_eq!(crate::types::MAX_DEPTH_LIMIT, 14);
        // 下界 0 是合法的(根节点树),不该被校验挡掉。
        let (x, y, n_rows, n_feats) = tiny_fixture();
        let mut p = TrainParams { max_depth: 0, n_rounds: 1, ..Default::default() };
        p.cols_per_block = n_feats;
        let cfg = TrainConfig::new(p, Objective::SquaredError);
        let src = DenseSource::from_row_major(&x, n_rows, n_feats, 32, n_feats);
        train(&src, &y, &[], &cfg, &mut [], &Local).expect("max_depth=0 是合法的根节点树");
        // 上限就是那个 buffer 装得下的最深一层。
        let deepest_partitioned_level = crate::types::MAX_DEPTH_LIMIT - 1;
        assert_eq!(1usize << deepest_partitioned_level, 8192, "depth 14 正好用满 8192 个槽位");
        assert!(1usize << crate::types::MAX_DEPTH_LIMIT > 8192, "depth 15 一定超");
    }

    /// 越界的 `max_depth` 必须在**共享校验**里被挡下来,而不是留给某个后端
    /// 在训练深处报一个和原因无关的错。
    ///
    /// ⚠️ 下界是 **0**(只有根节点的树,合法),所以这里只测上界。
    #[test]
    fn out_of_range_max_depth_is_rejected_before_training() {
        let (x, y, n_rows, n_feats) = tiny_fixture();
        // ⚠️ 0 **不在**这里 —— 它是合法的根节点树,不是越界值。
        for bad in [crate::types::MAX_DEPTH_LIMIT + 1, 32, 100] {
            let mut p = TrainParams { max_depth: bad, n_rounds: 1, ..Default::default() };
            p.cols_per_block = n_feats;
            let cfg = TrainConfig::new(p, Objective::SquaredError);
            let src = DenseSource::from_row_major(&x, n_rows, n_feats, 32, n_feats);
            let err = train(&src, &y, &[], &cfg, &mut [], &Local)
                .expect_err(&format!("max_depth={bad} 应当被拒"));
            assert!(
                err.to_string().contains("max_depth"),
                "错误信息要指出是 max_depth 的问题,收到:{err}"
            );
        }
    }

    /// 边界值必须真的能训 —— 否则上面那条只是把限制写死,没有证明它可用。
    #[test]
    fn max_depth_at_the_limit_trains_on_cpu() {
        let (x, y, n_rows, n_feats) = tiny_fixture();
        let mut p = TrainParams {
            max_depth: crate::types::MAX_DEPTH_LIMIT,
            n_rounds: 1,
            ..Default::default()
        };
        p.cols_per_block = n_feats;
        let cfg = TrainConfig::new(p, Objective::SquaredError);
        let src = DenseSource::from_row_major(&x, n_rows, n_feats, 32, n_feats);
        train(&src, &y, &[], &cfg, &mut [], &Local).expect("depth 14 必须能训");
    }

    /// 小夹具:行少所以深树会自然剪枝,这几条测的是**校验**不是性能。
    fn tiny_fixture() -> (Vec<f32>, Vec<f32>, usize, usize) {
        let (n_rows, n_feats) = (64usize, 4usize);
        let x: Vec<f32> = (0..n_rows * n_feats).map(|i| (i % 17) as f32).collect();
        let y: Vec<f32> = (0..n_rows).map(|i| (i % 3) as f32).collect();
        (x, y, n_rows, n_feats)
    }

    use super::*;
    use crate::callback::{EarlyStopping, RecordHistory, RoundMetrics, VerboseEval};
    use crate::columns::{BinCuts, ColumnBlock};
    use crate::comm::Local;
    use crate::metrics::{LogLoss, Rmse};
    use crate::source::DenseSource;

    const MISSING: f32 = f32::NAN;

    fn params(max_depth: u32, n_rounds: usize) -> TrainParams {
        TrainParams {
            max_depth,
            n_rounds,
            max_bin: 16,
            learning_rate: 0.3,
            lambda: 1.0,
            min_child_weight: 0.0,
            cols_per_block: 2,
            ..Default::default()
        }
    }

    /// 一个阶梯函数 + 两个噪声特征 + 一列缺失。
    /// cols_per_block = 2 时它会切成两个列块,顺便压到多块的路径。
    fn step_data(n: usize) -> (Vec<f32>, Vec<f32>, usize) {
        let n_features = 4;
        let mut x = Vec::with_capacity(n * n_features);
        let mut y = Vec::with_capacity(n);
        for i in 0..n {
            let t = i as f32 / n as f32;
            x.push(t); // 决定性特征
            x.push((i % 7) as f32); // 噪声
            x.push(if i % 5 == 0 { MISSING } else { -t }); // 带缺失
            x.push(((i * 13) % 11) as f32 * 0.1); // 噪声
            y.push(if t > 0.5 { 1.0 } else { 0.0 });
        }
        (x, y, n_features)
    }

    fn train_on(
        x: &[f32],
        y: &[f32],
        n_features: usize,
        p: TrainParams,
        obj: Objective,
    ) -> (Model, DenseSource) {
        let n_rows = y.len();
        let src = DenseSource::from_row_major(x, n_rows, n_features, p.max_bin, p.cols_per_block);
        let cfg = TrainConfig::new(p, obj);
        let model = train(&src, y, &[], &cfg, &mut [], &Local).unwrap();
        (model, src)
    }

    /// 最要紧的一条:训练侧按 bin 下降,导出的模型按浮点阈值下降,
    /// 两条路必须走到同一个叶子。差一位的话训练指标好看、
    /// 模型给出去就不对 —— 而且只在边界 bin 上错,很难发现。
    #[test]
    fn bin_descent_matches_float_descent() {
        let (x, y, nf) = step_data(60);
        let (model, src) = train_on(&x, &y, nf, params(3, 8), Objective::Logistic);

        let by_bins = predict_on(&model, &src).unwrap();
        for (row, got) in by_bins.iter().enumerate() {
            let raw = &x[row * nf..(row + 1) * nf];
            let by_floats = model.predict_margin(raw);
            assert_eq!(
                got.to_bits(),
                by_floats.to_bits(),
                "行 {row} {raw:?}:量化下降 {got} != 浮点下降 {by_floats}"
            );
        }
    }

    #[test]
    fn logistic_learns_the_step() {
        let (x, y, nf) = step_data(80);
        let (model, src) = train_on(&x, &y, nf, params(3, 30), Objective::Logistic);
        let pred = predict_on(&model, &src).unwrap();

        let ll = crate::metrics::LogLoss.eval(&pred, &y);
        assert!(ll < 0.2, "30 轮之后 logloss 应该很小,实际 {ll}");
        for (i, (&p, &label)) in pred.iter().zip(&y).enumerate() {
            let prob = 1.0 / (1.0 + (-p).exp());
            assert_eq!(prob > 0.5, label > 0.5, "第 {i} 行分错了(prob {prob})");
        }
    }

    #[test]
    fn regression_error_shrinks_every_round() {
        let (x, y, nf) = step_data(60);
        let mut last = f32::INFINITY;
        for rounds in [1usize, 2, 4, 8, 16] {
            let (model, src) = train_on(&x, &y, nf, params(3, rounds), Objective::SquaredError);
            let pred = predict_on(&model, &src).unwrap();
            let rmse = crate::metrics::Rmse.eval(&pred, &y);
            assert!(
                rmse < last,
                "{rounds} 轮的 rmse {rmse} 没比上一档 {last} 小"
            );
            last = rmse;
        }
    }

    /// 父子相减 + 行重分区的自洽性:所有叶子的 hess 加起来必须等于
    /// 根节点的 hess。相减写错、或者分区漏行/重复行,这里立刻就崩。
    #[test]
    fn leaf_hessians_sum_back_to_the_root() {
        let (x, y, nf) = step_data(100);
        let (model, _) = train_on(&x, &y, nf, params(4, 3), Objective::Logistic);

        for (k, t) in model.trees.iter().enumerate() {
            let root = t.nodes[0].sum_hess;
            let leaves: f32 = t
                .nodes
                .iter()
                .filter(|n| n.is_leaf)
                .map(|n| n.sum_hess)
                .sum();
            assert!(
                (leaves - root).abs() < 1e-3 * root.max(1.0),
                "第 {k} 棵树:叶子 hess 之和 {leaves} != 根 {root}"
            );
        }
    }

    #[test]
    fn respects_max_depth() {
        let (x, y, nf) = step_data(100);
        for max_depth in [0u32, 1, 2, 4] {
            let (model, _) = train_on(&x, &y, nf, params(max_depth, 2), Objective::Logistic);
            for t in &model.trees {
                assert!(depth_of(t, 0) <= max_depth, "max_depth {max_depth} 被突破");
            }
        }
    }

    #[test]
    fn root_only_tree_updates_the_next_round_predictions() {
        // max_depth=0 不进生长循环,根节点就是终叶。第一轮的叶子值必须
        // 直接累进 predictions,否则第二轮会看到同一份梯度、长出完全
        // 相同的叶子 —— 这是省掉训练侧树下降后最容易漏掉的端点。
        let x = vec![0.0f32; 8];
        let y = vec![1.0f32; 8];
        let mut p = params(0, 2);
        p.learning_rate = 0.3;
        let (model, _) = train_on(&x, &y, 1, p, Objective::SquaredError);

        let first = model.trees[0].nodes[0].leaf_value;
        let second = model.trees[1].nodes[0].leaf_value;
        assert!(
            second > 0.0 && second < first,
            "第二轮应该看到已更新的预测:{first} → {second}"
        );
    }

    #[test]
    fn rejects_mismatched_training_and_eval_labels_before_training() {
        let (x, y, nf) = step_data(20);
        let src = DenseSource::from_row_major(&x, 20, nf, 16, 2);
        let cfg = TrainConfig::new(params(2, 2), Objective::Logistic);

        let err = train(&src, &y[..19], &[], &cfg, &mut [], &Local)
            .expect_err("训练标签长度不一致必须报错");
        assert!(err.to_string().contains("标签有 19 行"), "{err:#}");

        let eval = EvalSet {
            name: "valid".into(),
            source: &src,
            labels: &y[..18],
        };
        let err = train(&src, &y, &[eval], &cfg, &mut [], &Local)
            .expect_err("验证标签长度不一致必须报错");
        assert!(err.to_string().contains("验证集 \"valid\""), "{err:#}");
    }

    #[test]
    fn partition_order_table_preserves_both_stable_orders() {
        for mask in 0u16..=255 {
            let order = &partition_orders()[mask as usize];
            let want_left: Vec<u8> = (0..8).filter(|&j| mask & (1 << j) != 0).collect();
            let want_right: Vec<u8> = (0..8).filter(|&j| mask & (1 << j) == 0).collect();
            assert_eq!(&order.left[..order.n_left as usize], want_left);
            assert_eq!(&order.right[..8 - order.n_left as usize], want_right);
        }
    }

    #[test]
    fn prefix_partition_is_thread_count_independent_and_stable() {
        let n = 100_003usize; // 多线程阈值以上,并带不足 8 行的尾巴
        let rows: Vec<RowId> = (0..n as RowId).rev().collect();
        let col: Vec<Bin> = (0..n)
            .map(|r| {
                if r % 17 == 0 {
                    MISSING_BIN
                } else {
                    (r % 5) as Bin
                }
            })
            .collect();

        for missing_left in [false, true] {
            let cand = SplitCandidate {
                feat: 0,
                bin: 3,
                gain: 1.0,
                left: GradPairFixed::default(),
                right: GradPairFixed::default(),
                missing_left,
            };
            let run = |nthread| {
                crate::threading::build_pool(nthread)
                    .unwrap()
                    .install(|| partition_rows(&rows, &col, &cand, &Local))
            };
            let one = run(1);
            let eight = run(8);
            assert_eq!(one, eight, "1/8 线程的行索引必须逐字节相同");

            let goes_left = |&row: &RowId| {
                let bin = col[row as usize];
                if bin == MISSING_BIN {
                    missing_left
                } else {
                    (bin as u32) < cand.bin
                }
            };
            assert_eq!(
                one.0,
                rows.iter()
                    .filter(|r| goes_left(r))
                    .copied()
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                one.1,
                rows.iter()
                    .filter(|r| !goes_left(r))
                    .copied()
                    .collect::<Vec<_>>()
            );
        }
    }

    fn depth_of(t: &Tree, node: usize) -> u32 {
        let n = &t.nodes[node];
        if n.is_leaf {
            0
        } else {
            1 + depth_of(t, n.left as usize).max(depth_of(t, n.right as usize))
        }
    }

    #[test]
    fn nothing_to_learn_gives_a_single_leaf() {
        // 标签恒等于 base_score:梯度全 0,任何分裂的增益都是 0,
        // 应该一刀都不分。分了就说明 eps 那道关没挡住浮点噪声。
        let (x, _, nf) = step_data(40);
        let y = vec![0.5f32; 40];
        let mut p = params(4, 1);
        p.learning_rate = 1.0;
        let src = DenseSource::from_row_major(&x, 40, nf, p.max_bin, p.cols_per_block);
        let cfg = TrainConfig {
            base_score: 0.5,
            ..TrainConfig::new(p, Objective::SquaredError)
        };
        let model = train(&src, &y, &[], &cfg, &mut [], &Local).unwrap();

        assert_eq!(model.trees[0].len(), 1, "不该分裂");
        assert!(model.trees[0].nodes[0].is_leaf);
    }

    #[test]
    fn training_is_reproducible() {
        // 同样输入训两遍,预测要逐位相同。哪天把归约改成并行/乱序,
        // 这条会第一个挂。
        let (x, y, nf) = step_data(70);
        let (a, src) = train_on(&x, &y, nf, params(3, 5), Objective::Logistic);
        let (b, _) = train_on(&x, &y, nf, params(3, 5), Objective::Logistic);
        let pa = predict_on(&a, &src).unwrap();
        let pb = predict_on(&b, &src).unwrap();
        for (i, (x, y)) in pa.iter().zip(&pb).enumerate() {
            assert_eq!(x.to_bits(), y.to_bits(), "第 {i} 行两次训练结果不同");
        }
    }

    /// 建一个"越训越差"的验证集:标签和训练集相反,logloss 一路上升,
    /// 早停应该在第 1 轮之后就开始数,数满 rounds 就停。
    fn run_early_stopping(rounds: usize, n_rounds: usize) -> (Model, usize) {
        let (x, y, nf) = step_data(80);
        let flipped: Vec<f32> = y.iter().map(|v| 1.0 - v).collect();
        let mut p = params(3, n_rounds);
        p.learning_rate = 0.5;

        let train_src = DenseSource::from_row_major(&x, 80, nf, p.max_bin, p.cols_per_block);
        let valid_src = DenseSource::with_cuts(
            &x,
            80,
            nf,
            DenseSource::from_row_major(&x, 80, nf, p.max_bin, p.cols_per_block).into_cuts(),
            p.cols_per_block,
        );
        let cfg = TrainConfig::new(p, Objective::Logistic);
        let evals = [EvalSet {
            name: "valid".into(),
            source: &valid_src,
            labels: &flipped,
        }];

        let mut cbs: Vec<Box<dyn Callback>> = vec![
            Box::new(EarlyStopping::new(rounds, true)),
            Box::new(VerboseEval { period: 0 }),
        ];
        let model = train(&train_src, &y, &evals, &cfg, &mut cbs, &Local).unwrap();
        let n_trees = model.trees.len();
        (model, n_trees)
    }

    #[test]
    fn early_stopping_stops_and_records_the_best_round() {
        let (model, n_trees) = run_early_stopping(3, 50);
        // 指标一路变差,best 停在第 0 轮,再数 3 轮没改善 → 一共 4 棵树
        assert_eq!(model.best_iteration, Some(0), "最佳轮次应该是第一轮");
        assert_eq!(n_trees, 4, "best(1 棵) + 连续 3 轮没改善 = 4 棵树");
        assert!(model.best_score.is_some());
        assert!(n_trees < 50, "早停没生效,跑满了 50 轮");
    }

    #[test]
    fn early_stopping_rounds_controls_the_patience() {
        for rounds in [1usize, 2, 5] {
            let (_, n_trees) = run_early_stopping(rounds, 50);
            assert_eq!(
                n_trees,
                rounds + 1,
                "耐心 {rounds} 轮应该留下 {} 棵树",
                rounds + 1
            );
        }
    }

    #[test]
    fn without_evals_early_stopping_never_fires() {
        // 没有验证集时不该早停 —— 否则第一轮就停了
        let (x, y, nf) = step_data(60);
        let p = params(3, 7);
        let src = DenseSource::from_row_major(&x, 60, nf, p.max_bin, p.cols_per_block);
        let cfg = TrainConfig::new(p, Objective::Logistic);
        let mut cbs: Vec<Box<dyn Callback>> = vec![Box::new(EarlyStopping::new(1, true))];
        let model = train(&src, &y, &[], &cfg, &mut cbs, &Local).unwrap();
        assert_eq!(model.trees.len(), 7);
        assert_eq!(model.best_iteration, None);
    }

    /// 增量评估必须和"每轮拿整个模型重算"**逐位相同**。
    ///
    /// 这是把 O(轮数²) 降成 O(轮数) 的那个优化,便宜但容易错:漏更新
    /// 一轮、或者把 base_margin 加两次,指标会小小地偏,而训练照常收敛,
    /// 光看曲线发现不了。
    #[test]
    fn incremental_eval_matches_full_recompute() {
        let (x, y, nf) = step_data(70);
        let p = params(3, 6);
        let train_src = DenseSource::from_row_major(&x, 70, nf, p.max_bin, p.cols_per_block);
        let valid_src = DenseSource::with_cuts(
            &x,
            70,
            nf,
            DenseSource::from_row_major(&x, 70, nf, p.max_bin, p.cols_per_block).into_cuts(),
            p.cols_per_block,
        );
        let cfg = TrainConfig::new(p, Objective::Logistic);
        let evals = [EvalSet {
            name: "valid".into(),
            source: &valid_src,
            labels: &y,
        }];

        let recorder = RecordHistory::new();
        let history = recorder.handle();
        let mut cbs: Vec<Box<dyn Callback>> = vec![Box::new(recorder)];
        let model = train(&train_src, &y, &evals, &cfg, &mut cbs, &Local).unwrap();

        // 把每轮的模型重建一遍,用全量预测算同一个指标
        let recorded = history.lock().unwrap();
        assert_eq!(recorded.len(), 6);
        for (round, m) in recorded.iter().enumerate() {
            let partial = Model {
                trees: model.trees[..=round].to_vec(),
                ..model.clone()
            };
            let full = predict_on(&partial, &valid_src).unwrap();
            let want = LogLoss.eval(&full, &y);
            let got = m.get("valid", "logloss").unwrap();
            assert_eq!(
                got.to_bits(),
                want.to_bits(),
                "第 {round} 轮的增量指标和全量重算不一致"
            );
        }
    }

    #[test]
    fn watched_metric_is_the_first_of_the_last_eval() {
        // entries 是「eval 外层、metric 内层」,所以 last() 是最后一个
        // metric,watched() 才是第一个 —— 配两个 metric 时才看得出区别
        let m = RoundMetrics {
            round: 0,
            entries: vec![
                ("train".into(), "logloss".into(), 1.0),
                ("train".into(), "rmse".into(), 2.0),
                ("valid".into(), "logloss".into(), 3.0),
                ("valid".into(), "rmse".into(), 4.0),
            ],
        };
        assert_eq!(m.watched().unwrap().1, "logloss");
        assert_eq!(m.watched().unwrap().2, 3.0);
        assert_eq!(m.last().unwrap().2, 4.0, "last() 是最后一个 metric");
    }

    #[test]
    fn multiple_metrics_all_get_recorded() {
        let (x, y, nf) = step_data(50);
        let p = params(2, 3);
        let src = DenseSource::from_row_major(&x, 50, nf, p.max_bin, p.cols_per_block);
        let valid = DenseSource::with_cuts(
            &x,
            50,
            nf,
            DenseSource::from_row_major(&x, 50, nf, p.max_bin, p.cols_per_block).into_cuts(),
            p.cols_per_block,
        );
        let cfg = TrainConfig {
            eval_metrics: vec![Box::new(LogLoss), Box::new(Rmse)],
            ..TrainConfig::new(p, Objective::Logistic)
        };
        let evals = [EvalSet {
            name: "valid".into(),
            source: &valid,
            labels: &y,
        }];
        let recorder = RecordHistory::new();
        let history = recorder.handle();
        let mut cbs: Vec<Box<dyn Callback>> = vec![Box::new(recorder)];
        train(&src, &y, &evals, &cfg, &mut cbs, &Local).unwrap();

        // 两个 metric 都要出现,顺序就是配置的顺序(early stopping 盯第一个)
        let recorded = history.lock().unwrap();
        assert_eq!(recorded.len(), 3);
        for m in recorded.iter() {
            assert_eq!(m.entries.len(), 2, "一个 eval × 两个 metric");
            assert_eq!(m.entries[0].1, "logloss");
            assert_eq!(m.entries[1].1, "rmse");
            assert_eq!(m.watched().unwrap().1, "logloss", "盯的必须是第一个");
        }
    }

    /// 加载次数必须跟着**块数 / 列数**走,不是节点数。
    ///
    /// 两趟各有各的形状:
    /// - 直方图那趟:每层每块一次 → `block()` ≤ 块数 × 层数 × 轮数
    /// - 行重分区那趟:该层选中的特征**去重后**每列一次 →
    ///   `feature_column()` ≤ 特征数 × 层数 × 轮数
    ///
    /// 第二条同时钉住了去重:深树末层节点数远多于特征数,少了去重,
    /// 同一个强特征会被几十个节点各读一遍,次数立刻超上限。
    #[test]
    fn loads_scale_with_blocks_and_columns_not_nodes() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct Counting {
            inner: DenseSource,
            blocks: AtomicUsize,
            cols: AtomicUsize,
        }
        impl BlockSource for Counting {
            fn n_blocks(&self) -> usize {
                self.inner.n_blocks()
            }
            fn n_rows(&self) -> usize {
                self.inner.n_rows()
            }
            fn n_features(&self) -> usize {
                self.inner.n_features()
            }
            fn cuts(&self) -> &BinCuts {
                self.inner.cuts()
            }
            // 元数据不算加载
            fn block_features(&self, i: usize) -> anyhow::Result<Vec<FeatId>> {
                self.inner.block_features(i)
            }
            fn with_block(&self, i: usize, f: &mut dyn FnMut(&ColumnBlock)) -> anyhow::Result<()> {
                self.blocks.fetch_add(1, Ordering::Relaxed);
                self.inner.with_block(i, f)
            }
            fn feature_column(&self, f: FeatId) -> anyhow::Result<Vec<Bin>> {
                self.cols.fetch_add(1, Ordering::Relaxed);
                self.inner.feature_column(f)
            }
        }

        // 标签必须**带噪声**:干净的阶跃函数第一刀就分完了,树只剩一个
        // 内部节点,这条测试就等于没测(step_data 正是这种)。
        // 这里让标签只有部分可预测,树才会一直往下分。
        let (n_rows, nf) = (400usize, 6usize);
        let mut rng = 0x243f_6a88_85a3_08d3u64;
        let mut next = move || {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (rng >> 33) as u32
        };
        let mut x = Vec::with_capacity(n_rows * nf);
        let mut y = Vec::with_capacity(n_rows);
        for _ in 0..n_rows {
            let row: Vec<f32> = (0..nf).map(|_| (next() % 11) as f32).collect();
            let signal = (row[0] - row[1]) / 3.0;
            let p_pos = 1.0 / (1.0 + (-signal).exp());
            y.push(if (next() % 1000) as f32 / 1000.0 < p_pos {
                1.0
            } else {
                0.0
            });
            x.extend_from_slice(&row);
        }
        let mut p = params(5, 2);
        // 每列一块:块数 = 特征数,加载次数的两个上限都最紧
        p.cols_per_block = 1;
        p.min_child_weight = 0.0;

        let inner = DenseSource::from_row_major(&x, n_rows, nf, p.max_bin, p.cols_per_block);
        let n_blocks = inner.n_blocks();
        let src = Counting {
            inner,
            blocks: AtomicUsize::new(0),
            cols: AtomicUsize::new(0),
        };

        let (depth, rounds) = (p.max_depth as usize, p.n_rounds);
        let cfg = TrainConfig::new(p, Objective::Logistic);
        let model = train(&src, &y, &[], &cfg, &mut [], &Local).unwrap();

        let blocks = src.blocks.load(Ordering::Relaxed);
        let cols = src.cols.load(Ordering::Relaxed);
        let internal = model.trees[0].nodes.iter().filter(|n| !n.is_leaf).count();

        // 这条测试有意义的前提:节点数确实比块数多
        assert!(
            internal > n_blocks,
            "内部节点才 {internal} 个、块 {n_blocks} 个 —— 数据太简单,测不出按节点加载"
        );
        // 训练预测直接用 build_tree 留下的叶子行分区,不再加载块下降。
        // 所以整块加载只剩建直方图:每层每块一次。
        let ceiling = n_blocks * depth * rounds;
        assert!(
            blocks <= ceiling,
            "block() 调了 {blocks} 次,上限 {n_blocks} 块 × {depth} 层 × {rounds} 轮 \
             = {ceiling}(只剩建直方图)"
        );
        assert!(
            cols <= nf * depth * rounds,
            "feature_column() 调了 {cols} 次,上限 {nf} 列 × {depth} 层 × {rounds} 轮 = {}。\
             超了说明同一层里同一个特征被重复读了(去重没生效)",
            nf * depth * rounds
        );
    }

    /// 验证集必须复用训练集的 cuts,这里顺带验证 evals 那条路能跑通。
    #[test]
    fn eval_set_reuses_training_cuts() {
        let (x, y, nf) = step_data(60);
        let p = params(3, 3);
        let train_src = DenseSource::from_row_major(&x, 60, nf, p.max_bin, p.cols_per_block);
        let valid_src = DenseSource::with_cuts(
            &x,
            60,
            nf,
            DenseSource::from_row_major(&x, 60, nf, p.max_bin, p.cols_per_block).into_cuts(),
            p.cols_per_block,
        );

        let cfg = TrainConfig::new(p, Objective::Logistic);
        let evals = [EvalSet {
            name: "valid".into(),
            source: &valid_src,
            labels: &y,
        }];
        let model = train(&train_src, &y, &evals, &cfg, &mut [], &Local).unwrap();

        // 同一份数据、同一套 cuts,训练集和验证集的预测必须一模一样
        let a = predict_on(&model, &train_src).unwrap();
        let b = predict_on(&model, &valid_src).unwrap();
        for (i, (x, y)) in a.iter().zip(&b).enumerate() {
            assert_eq!(x.to_bits(), y.to_bits(), "第 {i} 行");
        }
    }
}
