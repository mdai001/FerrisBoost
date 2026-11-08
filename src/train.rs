//! 训练循环。
//!
//! 注意循环顺序:外层是列块,内层是活跃节点。这是刻意的 ——
//! 反过来写(外层节点、内层列块)会让每层都重新加载所有列块,
//! 传输量变成 O(深度 × 数据)。

use crate::callback::{Callback, RoundMetrics};
use crate::columns::{BinCuts, ColumnBlock};
use crate::comm::Comm;
use anyhow::Context;
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
                eprintln!("\n=== time breakdown ({rounds} rounds) ===");
                eprintln!(
                    "note: HIST / ENUM / SUBTRACT / GPU_* are **summed across block workers**,\n      so a stage can exceed the whole round; **do not read them as wall-clock share or as an upper bound on savings**.\n      For the real wall clock of the same stage see HIST_PHASE_WALL; GRAD / QUANTIZE /\n      UPDATE_PRED run once per round and are already wall clock."
                );
                for (name, c) in all {
                    let ns = c.load(Ordering::Relaxed);
                    eprintln!("{name:<22}{:>9.2} s{:>8.1}%{:>10.1} ms/round",
                        ns as f64 / 1e9, ns as f64 / total as f64 * 100.0,
                        ns as f64 / 1e6 / rounds as f64);
                }
                eprintln!("{:<22}{:>9.2} s", "total", total as f64 / 1e9);
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
        // 行采样置零(每轮一次,墙钟)。
        SAMPLE,
        // build_tree 开头:root 梯度和 + root 行下标(每棵树一次,墙钟)。
        TREE_SETUP,
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
        GPU_SPLIT_METADATA_H2D,
        GPU_SPLIT_KERNEL,
        GPU_SPLIT_D2H,
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
    pub static GPU_SPLIT_KERNEL_CALLS: AtomicU64 = AtomicU64::new(0);
    pub static GPU_SPLIT_D2H_CALLS: AtomicU64 = AtomicU64::new(0);
    pub static GPU_SPLIT_D2H_BYTES: AtomicU64 = AtomicU64::new(0);
    pub static GPU_SPLIT_METADATA_H2D_CALLS: AtomicU64 = AtomicU64::new(0);
    pub static GPU_SPLIT_METADATA_H2D_BYTES: AtomicU64 = AtomicU64::new(0);
    pub static GPU_PART_COL_CALLS: AtomicU64 = AtomicU64::new(0);
    pub static GPU_PART_COL_BYTES: AtomicU64 = AtomicU64::new(0);
    pub static GPU_PART_ROWS_CALLS: AtomicU64 = AtomicU64::new(0);
    pub static GPU_PART_ROWS_BYTES: AtomicU64 = AtomicU64::new(0);
    pub static GPU_PART_D2H_CALLS: AtomicU64 = AtomicU64::new(0);
    pub static GPU_PART_D2H_BYTES: AtomicU64 = AtomicU64::new(0);
    pub static GPU_PRED_D2H_CALLS: AtomicU64 = AtomicU64::new(0);
    pub static GPU_PRED_D2H_BYTES: AtomicU64 = AtomicU64::new(0);
    pub static GPU_KERNEL_CALLS: AtomicU64 = AtomicU64::new(0);
    // GPU split-cache telemetry is per-training accounting, separate from timings.
    pub static GPU_SPLIT_CACHE_PARENT_HITS: AtomicU64 = AtomicU64::new(0);
    pub static GPU_SPLIT_CACHE_PARENT_MISSES: AtomicU64 = AtomicU64::new(0);
    pub static GPU_SPLIT_CACHE_ADMITTED_NODES: AtomicU64 = AtomicU64::new(0);
    pub static GPU_SPLIT_CACHE_SPILLED_NODES: AtomicU64 = AtomicU64::new(0);
    pub static GPU_SPLIT_EVALUATED_NODES: AtomicU64 = AtomicU64::new(0);
    pub static GPU_SPLIT_CAT_EVALUATED: AtomicU64 = AtomicU64::new(0);
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
            &GPU_SPLIT_KERNEL_CALLS,
            &GPU_SPLIT_D2H_CALLS,
            &GPU_SPLIT_D2H_BYTES,
            &GPU_SPLIT_METADATA_H2D_CALLS,
            &GPU_SPLIT_METADATA_H2D_BYTES,
            &GPU_PART_COL_CALLS,
            &GPU_PART_COL_BYTES,
            &GPU_PART_ROWS_CALLS,
            &GPU_PART_ROWS_BYTES,
            &GPU_PART_D2H_CALLS,
            &GPU_PART_D2H_BYTES,
            &GPU_PRED_D2H_CALLS,
            &GPU_PRED_D2H_BYTES,
            &GPU_KERNEL_CALLS,
            &GPU_SPLIT_CACHE_PARENT_HITS,
            &GPU_SPLIT_CACHE_PARENT_MISSES,
            &GPU_SPLIT_CACHE_ADMITTED_NODES,
            &GPU_SPLIT_CACHE_SPILLED_NODES,
            &GPU_SPLIT_EVALUATED_NODES,
            &GPU_SPLIT_CAT_EVALUATED,
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
        eprintln!("\n=== row repartition ===");
        eprintln!("column fetches  {:>12.1} /round", c as f64 / rounds as f64);
        eprintln!(
            "rows processed  {:>12.1} x1e8/round",
            r as f64 / 1e8 / rounds as f64
        );
        eprintln!("per row         {:>12.2} ns", ns as f64 / r as f64);
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
        eprintln!("\n=== histogram accumulation ===");
        eprintln!(
            "accumulations   {:>12.1} x1e8/round",
            n as f64 / 1e8 / rounds as f64
        );
        eprintln!("per accumulate  {:>12.2} ns", ns as f64 / n as f64);
        // 每次累加要读 1 字节 bin + 16 字节梯度,读写 16 字节直方图槽位
        let bytes = n as f64 * (1.0 + 16.0);
        eprintln!(
            "data streamed   {:>12.2} GB/round  -> {:.1} GB/s",
            bytes / 1e9 / rounds as f64,
            bytes / 1e9 / (ns as f64 / 1e9)
        );
    }

    pub fn report_gpu(rounds: usize) {
        let names: [(&str, &AtomicU64); 15] = [
            ("H2D:gpair (once per round)", &GPU_GPAIR_H2D),
            ("H2D:quantized col blocks", &GPU_HIST_BLOCK_H2D),
            ("H2D:histogram row index", &GPU_HIST_ROWS_H2D),
            ("hist_build kernels", &GPU_HIST_KERNEL),
            ("GPU split metadata H2D", &GPU_SPLIT_METADATA_H2D),
            ("GPU split kernels", &GPU_SPLIT_KERNEL),
            ("GPU split D2H", &GPU_SPLIT_D2H),
            ("D2H:histogram remainder", &GPU_HIST_D2H),
            ("histogram wrapper residual", &GPU_HIST_HOST),
            ("H2D:partition col fetch", &GPU_PART_COL_H2D),
            ("H2D:partition row index", &GPU_PART_ROWS_H2D),
            ("partition kernel", &GPU_PART_KERNEL),
            ("D2H:partition offsets", &GPU_PART_D2H),
            ("partition wrapper residual", &GPU_PART_HOST),
            ("D2H:leaf prediction", &GPU_PRED_D2H),
        ];
        if names.iter().all(|(_, c)| c.load(Ordering::Relaxed) == 0) {
            return;
        }
        let total: u64 = names.iter().map(|(_, c)| c.load(Ordering::Relaxed)).sum();
        eprintln!("\n=== GPU stages ({rounds} rounds) ===");
        for (name, c) in names {
            let ns = c.load(Ordering::Relaxed);
            eprintln!(
                "{name:<28}{:>9.2} ms/round{:>8.1}%",
                ns as f64 / 1e6 / rounds as f64,
                ns as f64 / total as f64 * 100.0
            );
        }
        // 这一行就是这次改造的验收指标:改造前 wrapper 是 kernel 的
        // 500 倍以上,说明量到的是分配器不是架构。
        let hist_kernel = (GPU_HIST_KERNEL.load(Ordering::Relaxed)
            + GPU_SPLIT_KERNEL.load(Ordering::Relaxed))
        .max(1);
        let hist_wrap = GPU_HIST_BLOCK_H2D.load(Ordering::Relaxed)
            + GPU_HIST_ROWS_H2D.load(Ordering::Relaxed)
            + GPU_HIST_D2H.load(Ordering::Relaxed)
            + GPU_SPLIT_METADATA_H2D.load(Ordering::Relaxed)
            + GPU_SPLIT_D2H.load(Ordering::Relaxed)
            + GPU_HIST_HOST.load(Ordering::Relaxed);
        let part_kernel = GPU_PART_KERNEL.load(Ordering::Relaxed).max(1);
        let part_wrap = GPU_PART_COL_H2D.load(Ordering::Relaxed)
            + GPU_PART_ROWS_H2D.load(Ordering::Relaxed)
            + GPU_PART_D2H.load(Ordering::Relaxed)
            + GPU_PART_HOST.load(Ordering::Relaxed)
            + GPU_PRED_D2H.load(Ordering::Relaxed);
        eprintln!(
            "histogram + split wrapper / kernels = {:.1}×;partition wrapper / kernel = {:.1}×",
            hist_wrap as f64 / hist_kernel as f64,
            part_wrap as f64 / part_kernel as f64
        );
        let transfer_rows: [(&str, &AtomicU64, &AtomicU64); 10] = [
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
            (
                "D2H:histogram remainder",
                &GPU_HIST_D2H_CALLS,
                &GPU_HIST_D2H_BYTES,
            ),
            (
                "H2D:split metadata",
                &GPU_SPLIT_METADATA_H2D_CALLS,
                &GPU_SPLIT_METADATA_H2D_BYTES,
            ),
            (
                "D2H:split packets",
                &GPU_SPLIT_D2H_CALLS,
                &GPU_SPLIT_D2H_BYTES,
            ),
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
        eprintln!("\n=== GPU transfer volume ({rounds} rounds) ===");
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
                "{name:<28}{:>8.1} calls/round{:>10.3} GB/round",
                calls as f64 / rounds as f64,
                bytes as f64 / 1e9 / rounds as f64
            );
        }
        eprintln!(
            "H2D total{:>25.3} GB/round",
            h2d_bytes as f64 / 1e9 / rounds as f64
        );
        eprintln!(
            "D2H total{:>25.3} GB/round",
            d2h_bytes as f64 / 1e9 / rounds as f64
        );
        eprintln!(
            "Histogram/split/partition kernels{:>5.1} launches/round",
            GPU_KERNEL_CALLS.load(Ordering::Relaxed) as f64 / rounds as f64
        );
        eprintln!(
            "  split kernel subset{:>18.1} launches/round",
            GPU_SPLIT_KERNEL_CALLS.load(Ordering::Relaxed) as f64 / rounds as f64
        );
        let cache_parent_hits = GPU_SPLIT_CACHE_PARENT_HITS.load(Ordering::Relaxed);
        let cache_parent_misses = GPU_SPLIT_CACHE_PARENT_MISSES.load(Ordering::Relaxed);
        let cache_admitted_nodes = GPU_SPLIT_CACHE_ADMITTED_NODES.load(Ordering::Relaxed);
        let cache_spilled_nodes = GPU_SPLIT_CACHE_SPILLED_NODES.load(Ordering::Relaxed);
        let split_evaluated_nodes = GPU_SPLIT_EVALUATED_NODES.load(Ordering::Relaxed);
        let split_cat_evaluated = GPU_SPLIT_CAT_EVALUATED.load(Ordering::Relaxed);
        eprintln!(
            "GPU_SPLIT_CACHE parents_hit={cache_parent_hits} parents_miss={cache_parent_misses} admitted_nodes={cache_admitted_nodes} spilled_nodes={cache_spilled_nodes} evaluated_nodes={split_evaluated_nodes} categorical_feature_nodes={split_cat_evaluated}"
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
        let is_hist = std::ptr::eq(block_h2d, &GPU_HIST_BLOCK_H2D);
        let hist_kernel_ms = if is_hist {
            (t.kernel_ms - t.split_kernel_ms).max(0.0)
        } else {
            t.kernel_ms
        };
        let hist_d2h_ms = if is_hist {
            (t.d2h_ms - t.split_d2h_ms).max(0.0)
        } else {
            t.d2h_ms
        };
        // `kernel_ms` and `d2h_ms` are inclusive. Their split-specific
        // counters are subsets, so include each total only once in the wall sum.
        let parts = ms_to_ns(t.block_h2d_ms)
            + ms_to_ns(t.rows_h2d_ms)
            + ms_to_ns(t.kernel_ms)
            + ms_to_ns(t.d2h_ms)
            + ms_to_ns(t.split_metadata_h2d_ms);
        block_h2d.fetch_add(ms_to_ns(t.block_h2d_ms), Ordering::Relaxed);
        rows_h2d.fetch_add(ms_to_ns(t.rows_h2d_ms), Ordering::Relaxed);
        kernel.fetch_add(ms_to_ns(hist_kernel_ms), Ordering::Relaxed);
        d2h.fetch_add(ms_to_ns(hist_d2h_ms), Ordering::Relaxed);
        host.fetch_add(wall_ns.saturating_sub(parts), Ordering::Relaxed);
        GPU_SPLIT_METADATA_H2D.fetch_add(ms_to_ns(t.split_metadata_h2d_ms), Ordering::Relaxed);
        GPU_SPLIT_KERNEL.fetch_add(ms_to_ns(t.split_kernel_ms), Ordering::Relaxed);
        GPU_SPLIT_D2H.fetch_add(ms_to_ns(t.split_d2h_ms), Ordering::Relaxed);
        GPU_SPLIT_KERNEL_CALLS.fetch_add(t.split_kernel_calls, Ordering::Relaxed);
        GPU_SPLIT_D2H_CALLS.fetch_add(t.split_d2h_calls, Ordering::Relaxed);
        GPU_SPLIT_D2H_BYTES.fetch_add(t.split_d2h_bytes, Ordering::Relaxed);
        GPU_SPLIT_METADATA_H2D_CALLS.fetch_add(t.split_metadata_h2d_calls, Ordering::Relaxed);
        GPU_SPLIT_METADATA_H2D_BYTES.fetch_add(t.split_metadata_h2d_bytes, Ordering::Relaxed);
        GPU_SPLIT_CACHE_PARENT_HITS.fetch_add(t.cache_parent_hits, Ordering::Relaxed);
        GPU_SPLIT_CACHE_PARENT_MISSES.fetch_add(t.cache_parent_misses, Ordering::Relaxed);
        GPU_SPLIT_CACHE_ADMITTED_NODES.fetch_add(t.cache_admitted_nodes, Ordering::Relaxed);
        GPU_SPLIT_CACHE_SPILLED_NODES.fetch_add(t.cache_spilled_nodes, Ordering::Relaxed);
        GPU_SPLIT_EVALUATED_NODES.fetch_add(t.split_evaluated_nodes, Ordering::Relaxed);
        GPU_SPLIT_CAT_EVALUATED.fetch_add(t.split_cat_evaluated, Ordering::Relaxed);

        // The call site identifies histogram versus partition; the segment
        // itself carries exact counts/bytes from the CUDA boundary.
        if is_hist {
            GPU_HIST_BLOCK_CALLS.fetch_add(t.block_h2d_calls, Ordering::Relaxed);
            GPU_HIST_BLOCK_BYTES.fetch_add(t.block_h2d_bytes, Ordering::Relaxed);
            GPU_HIST_ROWS_CALLS.fetch_add(t.rows_h2d_calls, Ordering::Relaxed);
            GPU_HIST_ROWS_BYTES.fetch_add(t.rows_h2d_bytes, Ordering::Relaxed);
            GPU_HIST_D2H_CALLS.fetch_add(
                t.d2h_calls.saturating_sub(t.split_d2h_calls),
                Ordering::Relaxed,
            );
            GPU_HIST_D2H_BYTES.fetch_add(
                t.d2h_bytes.saturating_sub(t.split_d2h_bytes),
                Ordering::Relaxed,
            );
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
        eprintln!("\n=== tree descent scan volume ===");
        eprintln!("row visits      {:>12.0} M", v / 1e6);
        eprintln!(
            "  advanced      {:>12.0} M  ({:.1}%)",
            a / 1e6,
            a / v * 100.0
        );
        eprintln!(
            "wasted visits   {:>12.0} M  ({:.1}%)",
            (v - a) / 1e6,
            (v - a) / v * 100.0
        );
        eprintln!("blocks x passes {:>12}", b);
        eprintln!(
            "  fully wasted  {:>12}  ({:.1}%)",
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

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn split_timing_is_subtracted_from_totals_and_reported_separately() {
            let _guard = PROFILE_RUN
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            reset();
            struct ResetOnDrop;
            impl Drop for ResetOnDrop {
                fn drop(&mut self) {
                    reset();
                }
            }
            let _cleanup = ResetOnDrop;

            let timing = crate::backend::cuda::GpuSegmentTiming {
                kernel_ms: 10.0,
                split_kernel_ms: 3.0,
                split_kernel_calls: 2,
                d2h_ms: 5.0,
                split_d2h_ms: 2.0,
                split_d2h_calls: 1,
                split_d2h_bytes: 192,
                split_metadata_h2d_ms: 1.0,
                split_metadata_h2d_calls: 1,
                split_metadata_h2d_bytes: 64,
                block_h2d_ms: 2.0,
                rows_h2d_ms: 1.0,
                d2h_calls: 4,
                d2h_bytes: 1000,
                kernel_calls: 5,
                ..Default::default()
            };
            record_gpu(
                &timing,
                20_000_000,
                &GPU_HIST_BLOCK_H2D,
                &GPU_HIST_ROWS_H2D,
                &GPU_HIST_KERNEL,
                &GPU_HIST_D2H,
                &GPU_HIST_HOST,
            );

            assert_eq!(GPU_HIST_KERNEL.load(Ordering::Relaxed), 7_000_000);
            assert_eq!(GPU_SPLIT_KERNEL.load(Ordering::Relaxed), 3_000_000);
            assert_eq!(GPU_HIST_D2H.load(Ordering::Relaxed), 3_000_000);
            assert_eq!(GPU_SPLIT_D2H.load(Ordering::Relaxed), 2_000_000);
            assert_eq!(GPU_SPLIT_METADATA_H2D.load(Ordering::Relaxed), 1_000_000);
            assert_eq!(GPU_HIST_HOST.load(Ordering::Relaxed), 1_000_000);

            assert_eq!(GPU_HIST_D2H_CALLS.load(Ordering::Relaxed), 3);
            assert_eq!(GPU_HIST_D2H_BYTES.load(Ordering::Relaxed), 808);
            assert_eq!(GPU_SPLIT_D2H_CALLS.load(Ordering::Relaxed), 1);
            assert_eq!(GPU_SPLIT_D2H_BYTES.load(Ordering::Relaxed), 192);
            assert_eq!(GPU_SPLIT_METADATA_H2D_CALLS.load(Ordering::Relaxed), 1);
            assert_eq!(GPU_SPLIT_METADATA_H2D_BYTES.load(Ordering::Relaxed), 64);
            assert_eq!(GPU_KERNEL_CALLS.load(Ordering::Relaxed), 5);
            assert_eq!(GPU_SPLIT_KERNEL_CALLS.load(Ordering::Relaxed), 2);
        }

        #[test]
        fn split_cache_counters_record_and_reset_per_profile_run() {
            let _guard = PROFILE_RUN
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            reset();
            struct ResetOnDrop;
            impl Drop for ResetOnDrop {
                fn drop(&mut self) {
                    reset();
                }
            }
            let _cleanup = ResetOnDrop;

            let timing = crate::backend::cuda::GpuSegmentTiming {
                cache_parent_hits: 7,
                cache_parent_misses: 3,
                cache_admitted_nodes: 5,
                cache_spilled_nodes: 2,
                split_evaluated_nodes: 11,
                ..Default::default()
            };
            record_gpu(
                &timing,
                0,
                &GPU_HIST_BLOCK_H2D,
                &GPU_HIST_ROWS_H2D,
                &GPU_HIST_KERNEL,
                &GPU_HIST_D2H,
                &GPU_HIST_HOST,
            );

            assert_eq!(GPU_SPLIT_CACHE_PARENT_HITS.load(Ordering::Relaxed), 7);
            assert_eq!(GPU_SPLIT_CACHE_PARENT_MISSES.load(Ordering::Relaxed), 3);
            assert_eq!(GPU_SPLIT_CACHE_ADMITTED_NODES.load(Ordering::Relaxed), 5);
            assert_eq!(GPU_SPLIT_CACHE_SPILLED_NODES.load(Ordering::Relaxed), 2);
            assert_eq!(GPU_SPLIT_EVALUATED_NODES.load(Ordering::Relaxed), 11);

            reset();
            assert_eq!(GPU_SPLIT_CACHE_PARENT_HITS.load(Ordering::Relaxed), 0);
            assert_eq!(GPU_SPLIT_CACHE_PARENT_MISSES.load(Ordering::Relaxed), 0);
            assert_eq!(GPU_SPLIT_CACHE_ADMITTED_NODES.load(Ordering::Relaxed), 0);
            assert_eq!(GPU_SPLIT_CACHE_SPILLED_NODES.load(Ordering::Relaxed), 0);
            assert_eq!(GPU_SPLIT_EVALUATED_NODES.load(Ordering::Relaxed), 0);
        }
    }
}

use crate::hist::{self, Histogram};
use crate::metrics::Metric;
use crate::split::{self, SplitCandidate};
use crate::tree::{self, Model, Tree};
use crate::types::{
    FeatId, GradPair, GradPairFixed, GradQuantizer, Objective, PredictionAuthority, RowId,
    TrainParams,
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
    /// `None`(默认)和 XGBoost 2.0+ 一样从训练标签估计
    /// (`Objective::estimate_base_score`);给了值就原样用。
    /// 训练出来的 `Model.base_score` 永远是解析后的那个数。
    pub base_score: Option<f32>,
    /// Per-row starting raw margin for the training set (XGBoost `base_margin`),
    /// class-major: `base_margin[k * n + i]` is row `i`, output `k` (so a
    /// single-output objective is simply `n` values). It **replaces** the
    /// `base_score` as the starting prediction; the intercept is still
    /// estimated from the labels (or taken from `base_score`) and stored in the
    /// model, exactly as XGBoost does. `None` = start from the intercept.
    pub base_margin: Option<Vec<f32>>,
    /// Per-validation-set starting margins, parallel to the `evals` slice
    /// (same layout as `base_margin`). A missing tail or a `None` entry means
    /// that set starts from the intercept.
    pub eval_base_margins: Vec<Option<Vec<f32>>>,
    /// Training continuation: the model whose trees, intercept (`base_score` /
    /// class margins), feature names and best-iteration metadata are carried
    /// into the result, with the newly trained trees appended. The caller must
    /// supply `base_margin` / `eval_base_margins` equal to this model's raw
    /// margin on the respective data (the starting predictions); the model
    /// itself is never evaluated here, because the data arrives binned with
    /// freshly sketched cuts while its splits are float thresholds.
    pub prior_model: Option<Model>,
    /// Query-group row offsets (`n_groups + 1` values, first 0, last = rows) of the
    /// training set. Required by the `rank:*` objectives and rejected by every
    /// other objective. With ranking, the `weights` argument of the train entry
    /// points holds one weight per query group instead of one per row.
    pub group_ptr: Option<Vec<u32>>,
}

impl TrainConfig {
    /// `base_score` 默认不设:和 XGBoost 一样从标签估计。
    pub fn new(params: TrainParams, objective: Objective) -> Self {
        Self {
            params,
            objective,
            quiet: false,
            eval_metrics: vec![crate::metrics::default_metric(objective)],
            base_score: None,
            base_margin: None,
            eval_base_margins: Vec::new(),
            prior_model: None,
            group_ptr: None,
        }
    }

    /// The starting margins of validation set `i`, if it has any.
    fn eval_margin(&self, i: usize) -> Option<&[f32]> {
        self.eval_base_margins.get(i).and_then(|m| m.as_deref())
    }

    /// 解析 base_score:用户给的值,或从训练标签/权重估计。
    pub fn resolve_base_score(&self, labels: &[f32], weights: Option<&[f32]>) -> f32 {
        self.base_score.unwrap_or_else(|| {
            let spw = self.params.scale_pos_weight;
            let g = gradient_weights(self.objective, labels, weights, spw);
            self.objective
                .initial_base_score(labels, weights, g.as_deref().or(weights), spw)
        })
    }
}

/// 一个验证集。名字会出现在日志和 evals_result 里。
///
/// early stopping 默认盯**最后一个** eval 的最后一个 metric —— 语义
/// 和 XGBoost 一致,所以顺序有意义。
pub struct EvalSet<'a> {
    pub name: String,
    pub source: &'a dyn BlockSource,
    pub labels: &'a [f32],
    /// 每行样本权重,`None` 等价于全 1(XGBoost 的 DMatrix `weight`)。
    /// 加权 metric 是 `sum(w·loss) / sum(w)`。
    pub weights: Option<&'a [f32]>,
    /// Query-group row offsets of this validation set (required by ranking
    /// objectives; with ranking, `weights` is one value per group).
    pub group_ptr: Option<&'a [u32]>,
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
            let f = f
                .take()
                .expect("with_block_cols implementations must call f exactly once");
            slot = Some(f(block));
        })?;
        Ok(slot.expect("with_block_cols implementations must call f exactly once"))
    }

    fn with_block_ret<R>(&self, i: usize, f: impl FnOnce(&ColumnBlock) -> R) -> anyhow::Result<R> {
        let mut slot = None;
        let mut f = Some(f);
        self.with_block(i, &mut |block| {
            let f = f
                .take()
                .expect("with_block implementations must call f exactly once");
            slot = Some(f(block));
        })?;
        Ok(slot.expect("with_block implementations must call f exactly once"))
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

    /// Source block materializations, separate from raw file ingestion and GPU
    /// uploads. Implementations that do not track this return None.
    fn source_block_reads(&self) -> Option<u64> {
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
    ///
    /// The column comes back at its feature's storage width (`crate::bins`).
    fn feature_column(&self, feat: FeatId) -> anyhow::Result<crate::bins::BinData>;

    /// 直接借出**驻留在内存里的**单个特征整列;拿不到(落盘 cache 等)返回
    /// `None`,调用方退回 `feature_column`。
    ///
    /// 行重分区每层对每个选中的特征取一次列,而且只读。`feature_column`
    /// 交出所有权,所以全量驻留的 source 也要为一次只读的 gather 复制
    /// n_rows 字节(HIGGS:每棵树约 30 次 × 10.5 MB,串行 memcpy,外加
    /// 每次一块新分配的缺页)。驻留的 source 覆盖本方法零拷贝借出;
    /// 默认 `None`,所以只能交出拷贝的 source(落盘 cache、测试里的
    /// 包装器)行为和以前完全一样。
    fn resident_feature_column(&self, _feat: FeatId) -> Option<crate::bins::BinSlice<'_>> {
        None
    }

    /// 是否**每一列**都能通过 `resident_feature_column` 借出(全量驻留的
    /// source)。训练据此决定能不能对零梯度行做随机访问下降。
    fn has_resident_columns(&self) -> bool {
        false
    }

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
    train_with_backend(source, labels, None, evals, cfg, callbacks, comm, None)
}

/// `train` 加每行样本权重(XGBoost 的 DMatrix `weight`)。
///
/// 语义与 XGBoost 一致:objective 算出 grad / hess 之后**两者同乘 w**
/// (f32),再进入采样掩码和定点量化。`None` 与全 1 权重逐字节相同。
pub fn train_weighted(
    source: &dyn BlockSource,
    labels: &[f32],
    weights: Option<&[f32]>,
    evals: &[EvalSet],
    cfg: &TrainConfig,
    callbacks: &mut [Box<dyn Callback>],
    comm: &dyn Comm,
) -> anyhow::Result<Model> {
    train_with_backend(source, labels, weights, evals, cfg, callbacks, comm, None)
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
        None,
        evals,
        cfg,
        callbacks,
        comm,
        Some(device_ordinal),
    )
}

/// `train_gpu` 加每行样本权重。语义见 [`train_weighted`]。
#[cfg(feature = "cuda")]
#[allow(clippy::too_many_arguments)] // Mirrors train_gpu plus the weight slice.
pub fn train_gpu_weighted(
    device_ordinal: usize,
    source: &dyn BlockSource,
    labels: &[f32],
    weights: Option<&[f32]>,
    evals: &[EvalSet],
    cfg: &TrainConfig,
    callbacks: &mut [Box<dyn Callback>],
    comm: &dyn Comm,
) -> anyhow::Result<Model> {
    train_with_backend(
        source,
        labels,
        weights,
        evals,
        cfg,
        callbacks,
        comm,
        Some(device_ordinal),
    )
}

/// 权重校验:长度一致、有限、非负。XGBoost 同样拒绝负权重。
fn validate_weights(weights: &[f32], n: usize, name: &str) -> anyhow::Result<()> {
    crate::ensure_input!(
        weights.len() == n,
        "{name}: weights have {} rows, but the data has {n} rows",
        weights.len()
    );
    crate::ensure_input!(
        weights.iter().all(|w| w.is_finite() && *w >= 0.0),
        "{name}: weights must all be finite and non-negative"
    );
    // 单个权重有限,加起来(以及乘上梯度后)仍可能溢出 f32,
    // 训练会悄悄变成 inf / NaN。总和超出 f32 范围直接拒绝。
    let total: f64 = weights.iter().map(|&w| f64::from(w)).sum();
    crate::ensure_input!(
        total <= f64::from(f32::MAX),
        "{name}: weights sum to {total:e}, which exceeds the f32 range; scale the weights down"
    );
    Ok(())
}

fn validate_source_metadata(source: &dyn BlockSource, name: &str) -> anyhow::Result<()> {
    let n_features = source.n_features();
    anyhow::ensure!(source.n_blocks() > 0, "{name} has no column blocks");
    anyhow::ensure!(
        source.cuts().n_feats() == n_features,
        "{name}: cuts have {} features, but the source declares {n_features}",
        source.cuts().n_feats()
    );
    let mut seen = vec![false; n_features];
    for block in 0..source.n_blocks() {
        let feats = source.block_features(block)?;
        anyhow::ensure!(
            !feats.is_empty(),
            "{name}: column block {block} has no features"
        );
        anyhow::ensure!(
            feats.windows(2).all(|w| w[0] < w[1]),
            "{name}: column block {block} features must be strictly increasing with no duplicates"
        );
        for feat in feats {
            let feat = feat as usize;
            anyhow::ensure!(
                feat < n_features,
                "{name}: column block {block} contains out-of-range feature {feat}"
            );
            anyhow::ensure!(
                !seen[feat],
                "{name}: feature {feat} appears in more than one column block"
            );
            seen[feat] = true;
        }
    }
    if let Some(feat) = seen.iter().position(|&present| !present) {
        anyhow::bail!("{name}: feature {feat} is not in any column block");
    }
    Ok(())
}

/// Checks for the starting-margin inputs (`base_margin`, per-eval margins and
/// the continuation model): shapes, finiteness, and that a continued model is
/// compatible with the new run. Shared by every backend.
fn validate_start_margins(
    cfg: &TrainConfig,
    source: &dyn BlockSource,
    evals: &[EvalSet],
) -> anyhow::Result<()> {
    let k = cfg.objective.num_outputs();
    let check = |margin: &[f32], rows: usize, what: &str| -> anyhow::Result<()> {
        crate::ensure_input!(
            margin.len() == rows * k,
            "{what}: base_margin has {} values, expected {rows} rows × {k} outputs = {}",
            margin.len(),
            rows * k
        );
        crate::ensure_input!(
            margin.iter().all(|v| v.is_finite()),
            "{what}: base_margin values must all be finite"
        );
        Ok(())
    };
    if let Some(m) = &cfg.base_margin {
        check(m, source.n_rows(), "training set")?;
    }
    crate::ensure_input!(
        cfg.eval_base_margins.len() <= evals.len(),
        "eval_base_margins has {} entries, but there are only {} validation sets",
        cfg.eval_base_margins.len(),
        evals.len()
    );
    for (i, ev) in evals.iter().enumerate() {
        if let Some(m) = cfg.eval_margin(i) {
            check(
                m,
                ev.source.n_rows(),
                &format!("validation set {:?}", ev.name),
            )?;
        }
    }
    let Some(prior) = &cfg.prior_model else {
        return Ok(());
    };
    // XGBoost refuses to continue unbiased LambdaMART (`CHECK_EQ(iter, 0)` in
    // lambdarank_obj.cc): the position biases are estimated from round 0 and
    // restarting them would silently change the model.
    let unbiased = |o: &Objective| matches!(o, Objective::Rank { params, .. } if params.unbiased);
    crate::ensure_input!(
        !unbiased(&cfg.objective) && prior.rank_bias.is_none(),
        "lambdarank_unbiased models cannot be trained further (same as XGBoost); train from scratch"
    );
    crate::ensure_input!(
        prior.objective.name() == cfg.objective.name(),
        "continued training uses objective {:?}, but the existing model uses {:?}",
        cfg.objective.name(),
        prior.objective.name()
    );
    // Each output of a quantile model is tied to its alpha: continuing with other
    // quantiles would silently mix two different targets.
    crate::ensure_input!(
        prior.objective.quantile_alphas() == cfg.objective.quantile_alphas(),
        "continued training uses quantile_alpha {:?}, but the existing model uses {:?}; continued training must use the same quantiles",
        cfg.objective
            .quantile_alphas()
            .map(|a| a.as_slice().to_vec()),
        prior
            .objective
            .quantile_alphas()
            .map(|a| a.as_slice().to_vec())
    );
    crate::ensure_input!(
        prior.num_outputs() == k,
        "continued training has {k} outputs (num_class), but the existing model has {}",
        prior.num_outputs()
    );
    crate::ensure_input!(
        prior.n_features == source.n_features(),
        "continued training data has {} features, but the existing model has {}",
        source.n_features(),
        prior.n_features
    );
    crate::ensure_input!(
        prior.trees.len() % k == 0,
        "the existing model has {} trees, which is not a whole number of rounds for {k} outputs",
        prior.trees.len()
    );
    // An empty `feature_types` means every feature is numeric.
    let flags = |f: &[bool], n: usize| -> Vec<bool> {
        if f.is_empty() {
            vec![false; n]
        } else {
            f.to_vec()
        }
    };
    let n_features = source.n_features();
    crate::ensure_input!(
        flags(&prior.feature_types, n_features)
            == flags(source.cuts().categorical_flags(), n_features),
        "categorical features (feature_types) of the continued training data do not match the existing model"
    );
    // The new trees can only split on codes the current training supports
    // (`MAX_CAT_CODE_WIDE`); refuse a model that already splits on larger ones
    // rather than carrying (or truncating) them.
    for (t, tree) in prior.trees.iter().enumerate() {
        for node in tree.nodes.iter().filter(|n| !n.is_leaf) {
            let Some(words) = &node.cats else { continue };
            if let Some(&big) = crate::categorical::categories_from_words(words)
                .iter()
                .find(|&&c| c > crate::categorical::MAX_CAT_CODE_WIDE)
            {
                crate::bail_input!(
                    "tree {t} of the existing model splits categorical feature {} on category code {big}, above the largest code training supports ({}); cannot continue training",
                    node.feat,
                    crate::categorical::MAX_CAT_CODE_WIDE
                );
            }
        }
    }
    if let Some(b) = cfg.base_score {
        let model_b = if k > 1 {
            prior.class_base_margins.first().copied().unwrap_or(b)
        } else {
            prior.base_score
        };
        crate::ensure_input!(
            k > 1 || b.to_bits() == model_b.to_bits(),
            "continued training keeps the existing model's base_score {model_b}; it cannot be changed to {b}"
        );
    }
    if !prior.trees.is_empty() {
        crate::ensure_input!(
            cfg.base_margin.is_some(),
            "continued training requires the existing model's raw margin on the training set as base_margin"
        );
        for (i, ev) in evals.iter().enumerate() {
            crate::ensure_input!(
                cfg.eval_margin(i).is_some(),
                "continued training requires the existing model's raw margin on validation set {:?} as base_margin",
                ev.name
            );
        }
    }
    Ok(())
}

/// Row-major `n x k` host margins -> class-major `k x n`.
pub fn class_major_from_rows(rows: &[f32], n: usize, k: usize) -> Vec<f32> {
    if k == 1 {
        return rows.to_vec();
    }
    let mut out = vec![0.0f32; n * k];
    for (i, row) in rows.chunks_exact(k).enumerate() {
        for (c, &v) in row.iter().enumerate() {
            out[c * n + i] = v;
        }
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn train_with_backend(
    source: &dyn BlockSource,
    labels: &[f32],
    weights: Option<&[f32]>,
    evals: &[EvalSet],
    cfg: &TrainConfig,
    callbacks: &mut [Box<dyn Callback>],
    comm: &dyn Comm,
    gpu_device: Option<usize>,
) -> anyhow::Result<Model> {
    let n = source.n_rows();
    crate::ensure_input!(n > 0, "training data must have at least one row");
    // 行号是 u32(`RowId`):超过就会静默截断成错的行。
    crate::ensure_input!(
        n <= crate::types::RowId::MAX as usize,
        "training data has {n} rows, above the supported limit {}",
        crate::types::RowId::MAX
    );
    crate::ensure_input!(
        source.n_features() > 0,
        "training data must have at least one feature"
    );
    validate_source_metadata(source, "training set")?;
    crate::ensure_input!(
        labels.len() == n,
        "labels have {} rows, but the training data has {n} rows",
        labels.len()
    );
    crate::ensure_input!(
        labels.iter().all(|v| v.is_finite()),
        "labels must all be finite"
    );
    let ranking = cfg.objective.rank();
    match (ranking.is_some(), cfg.group_ptr.as_deref()) {
        (true, None) => crate::bail_input!(
            "{} needs query groups: pass qid= or group= with the training data",
            cfg.objective.name()
        ),
        // Without a ranking objective the groups do not change training (the
        // training set only keeps them if it is also evaluated); they are checked all
        // the same.
        (_, Some(g)) => crate::rank::validate_group_ptr(g, n, "training set")?,
        (false, None) => {}
    }
    // Ranking weights are per query group (XGBoost): they never reach the per-row
    // machinery (gradient weights, base score, device labels).
    let n_weights = match (&cfg.group_ptr, ranking.is_some()) {
        (Some(g), true) => g.len() - 1,
        _ => n,
    };
    if let Some(w) = weights {
        validate_weights(w, n_weights, "training set")?;
    }
    let (group_weights, weights) = if ranking.is_some() {
        (weights, None)
    } else {
        (None, weights)
    };
    cfg.objective.validate_labels(labels, "training set")?;
    cfg.objective.validate_params()?;
    let fast_requested = cfg.params.gpu_math == crate::types::GpuMath::Fast;
    let rank_grad = match (ranking, cfg.group_ptr.as_deref()) {
        (Some((kind, params)), Some(groups)) => Some(crate::rank::RankGradient::new(
            kind,
            params,
            labels,
            groups,
            group_weights,
        )?),
        _ => None,
    };
    // One draw of the training RNG per round (XGBoost seeds the sampled pairs of
    // `lambdarank_pair_method=mean` from it); the base-score estimation takes the
    // first one when the base score is not given.
    let mut round_seeds =
        ranking.map(|(_, params)| crate::rank::RoundSeeds::new(&params, cfg.params.seed));
    // Continuation: skip the draws the earlier rounds (and the base-score
    // estimation) would have consumed, so the pair sampling of N+M rounds equals
    // N continued by M. A first run with an explicit `base_score` drew one less;
    // for it the continued stream is shifted by one draw (still deterministic).
    if let (Some(seeds), Some(prior)) = (round_seeds.as_mut(), cfg.prior_model.as_ref()) {
        for _ in 0..=prior.trees.len() {
            seeds.draw();
        }
    }
    let num_class = cfg.objective.num_outputs();
    validate_start_margins(cfg, source, evals)?;
    // 多分类:每类一个 base margin。用户给了 base_score 就每类都用它
    // (XGBoost 把标量复制成 K 份),否则按类别频率估计。
    // 续训:截距永远是已有模型的(XGBoost 同样保留模型的 base_score)。
    let class_margins: Vec<f32> = if num_class > 1 {
        cfg.objective.validate_labels(labels, "training set")?;
        match (&cfg.prior_model, cfg.base_score) {
            (Some(prior), _) => prior.class_base_margins.clone(),
            (None, Some(b)) => vec![b; num_class],
            // 多分位数:每个 alpha 一个标签分位数截距。
            (None, None) if cfg.objective.is_multi_quantile() => {
                cfg.objective.initial_quantile_margins(labels, weights)
            }
            (None, None) => cfg.objective.initial_class_margins(labels, weights),
        }
    } else {
        Vec::new()
    };
    let base_score = if let Some(prior) = &cfg.prior_model {
        prior.base_score
    } else if num_class > 1 {
        class_margins[0]
    } else if let Some(rg) = rank_grad.as_ref() {
        match cfg.base_score {
            Some(b) => b,
            None => {
                let seed = round_seeds.as_mut().map_or(0, |r| r.draw());
                crate::rank::estimate_base_score(rg, seed)
            }
        }
    } else {
        cfg.resolve_base_score(labels, weights)
    };
    // 梯度用的权重:样本权重 × (标签 == 1 时 scale_pos_weight)。只进梯度,
    // metric 和分箱仍然只看样本权重 —— 和 XGBoost `RegLossObj` 一致。
    // spw == 1 时不复制(乘 1 是恒等,结果逐位相同)。
    let spw_weights = gradient_weights(cfg.objective, labels, weights, cfg.params.scale_pos_weight);
    let grad_weights: Option<&[f32]> = spw_weights.as_deref().or(weights);
    crate::ensure_input!(base_score.is_finite(), "base_score must be finite");
    cfg.objective.validate_base_score(base_score)?;
    let source_bytes_start = source.source_bytes_read();
    let source_reads_start = source.source_block_reads();
    let p = &cfg.params;
    crate::ensure_input!(
        (1..=crate::types::MAX_BIN_LIMIT).contains(&p.max_bin),
        "max_bin must be in 1..={}, got {}",
        crate::types::MAX_BIN_LIMIT,
        p.max_bin
    );
    // ⚠️ **必须在共享路径上校验**,不能只在 GPU 里。以前完全没有校验:
    // CPU 会安静地接受 depth 16,GPU 则在训练深处报
    // 「一层的 partition 节点数超过 8192」——同一份参数两个后端行为不同,
    // 而且 GPU 那个错误信息离原因很远。
    // ⚠️ 下界是 **0**,不是 1:`max_depth = 0` 是合法的「只有根节点」的树,
    // 有专门的测试钉着它的语义(`root_only_tree_updates_the_next_round_predictions`)。
    crate::ensure_input!(
        p.max_depth <= crate::types::MAX_DEPTH_LIMIT,
        "max_depth must be in 0..={}, got {}; 0 means a root-only tree, and the upper bound \
         is set by the GPU per-level partition node buffer (see types::MAX_DEPTH_LIMIT)",
        crate::types::MAX_DEPTH_LIMIT,
        p.max_depth
    );
    crate::ensure_input!(p.cols_per_block > 0, "cols_per_block must be > 0");
    // XGBoost:约束比特征多是错误;比特征少的部分视为 0。
    crate::ensure_input!(
        p.monotone_constraints.len() <= source.n_features(),
        "monotone_constraints has {} entries, but there are only {} features",
        p.monotone_constraints.len(),
        source.n_features()
    );
    crate::ensure_input!(
        p.monotone_constraints.iter().all(|c| (-1..=1).contains(c)),
        "monotone_constraints accepts only -1 / 0 / 1 (after sign normalization)"
    );
    crate::ensure_input!(
        p.learning_rate.is_finite() && p.learning_rate >= 0.0,
        "learning_rate must be finite and non-negative"
    );
    crate::ensure_input!(
        p.lambda.is_finite() && p.lambda >= 0.0,
        "lambda must be finite and non-negative"
    );
    crate::ensure_input!(
        p.gamma.is_finite() && p.gamma >= 0.0,
        "gamma must be finite and non-negative"
    );
    crate::ensure_input!(
        p.min_child_weight.is_finite() && p.min_child_weight >= 0.0,
        "min_child_weight must be finite and non-negative"
    );
    crate::ensure_input!(
        p.alpha.is_finite() && p.alpha >= 0.0,
        "alpha must be finite and non-negative"
    );
    crate::ensure_input!(
        p.max_delta_step.is_finite() && p.max_delta_step >= 0.0,
        "max_delta_step must be finite and non-negative"
    );
    crate::ensure_input!(
        p.scale_pos_weight.is_finite() && p.scale_pos_weight >= 0.0,
        "scale_pos_weight must be finite and non-negative"
    );
    crate::ensure_input!(
        p.subsample > 0.0 && p.subsample <= 1.0 && p.subsample.is_finite(),
        "subsample must be in (0, 1], got {}",
        p.subsample
    );
    crate::ensure_input!(
        p.colsample_bytree > 0.0 && p.colsample_bytree <= 1.0 && p.colsample_bytree.is_finite(),
        "colsample_bytree must be in (0, 1], got {}",
        p.colsample_bytree
    );
    crate::ensure_input!(
        p.colsample_bylevel > 0.0 && p.colsample_bylevel <= 1.0 && p.colsample_bylevel.is_finite(),
        "colsample_bylevel must be in (0, 1], got {}",
        p.colsample_bylevel
    );
    crate::ensure_input!(
        p.colsample_bynode > 0.0 && p.colsample_bynode <= 1.0 && p.colsample_bynode.is_finite(),
        "colsample_bynode must be in (0, 1], got {}",
        p.colsample_bynode
    );
    for ev in evals {
        crate::ensure_input!(
            ev.source.n_rows() > 0,
            "validation set {:?} is empty",
            ev.name
        );
        crate::ensure_input!(
            ev.source.n_rows() <= crate::types::RowId::MAX as usize,
            "validation set {:?} has {} rows, above the supported limit {}",
            ev.name,
            ev.source.n_rows(),
            crate::types::RowId::MAX
        );
        crate::ensure_input!(
            ev.labels.len() == ev.source.n_rows(),
            "validation set {:?}: labels have {} rows, but the data has {} rows",
            ev.name,
            ev.labels.len(),
            ev.source.n_rows()
        );
        crate::ensure_input!(
            ev.source.n_features() == source.n_features(),
            "validation set {:?} has {} features, but the training set has {}",
            ev.name,
            ev.source.n_features(),
            source.n_features()
        );
        crate::ensure_input!(
            ev.labels.iter().all(|v| v.is_finite()),
            "validation set {:?}: labels must all be finite",
            ev.name
        );
        cfg.objective
            .validate_labels(ev.labels, &format!("validation set {:?}", ev.name))?;
        match (ranking.is_some(), ev.group_ptr) {
            (true, None) => crate::bail_input!(
                "validation set {:?} needs query groups (eval qid / group) for {}",
                ev.name,
                cfg.objective.name()
            ),
            (_, Some(g)) => crate::rank::validate_group_ptr(
                g,
                ev.source.n_rows(),
                &format!("validation set {:?}", ev.name),
            )?,
            (false, None) => {}
        }
        for m in &cfg.eval_metrics {
            m.check_labels(ev.labels, &format!("validation set {:?}", ev.name))?;
            // `aucpr` over query groups needs binary relevance (XGBoost `InvalidLabels`).
            if ev.group_ptr.is_some() && m.name() == "aucpr" {
                crate::rank::GroupAuc::new(crate::rank::GroupAucKind::Pr, false)
                    .check_labels(ev.labels, &format!("validation set {:?}", ev.name))?;
            }
        }
        if let Some(w) = ev.weights {
            let n_ev_weights = match (ev.group_ptr, ranking.is_some()) {
                (Some(g), true) => g.len() - 1,
                _ => ev.source.n_rows(),
            };
            // XGBoost scores `auc` / `aucpr` of data with query groups per group, with
            // one weight per group, which a per-row weight vector cannot be.
            if ev.group_ptr.is_some() && ranking.is_none() {
                if let Some(m) = cfg
                    .eval_metrics
                    .iter()
                    .find(|m| matches!(m.name(), "auc" | "aucpr"))
                {
                    crate::bail_input!(
                        "validation set {:?}: {} with query groups is scored per group and needs \
                         one weight per group, but {} uses per-row weights; drop the weights or \
                         the groups of this set",
                        ev.name,
                        m.name(),
                        cfg.objective.name()
                    );
                }
            }
            validate_weights(w, n_ev_weights, &format!("validation set {:?}", ev.name))?;
            for m in &cfg.eval_metrics {
                crate::ensure_input!(
                    m.supports_weights(),
                    "metric {:?} does not support weighted evaluation, but validation set {:?} has weights",
                    m.name(),
                    ev.name
                );
            }
        }
        validate_source_metadata(ev.source, &format!("validation set {:?}", ev.name))?;
        crate::ensure_input!(
            ev.source.cuts().offsets == source.cuts().offsets
                && ev.source.cuts().values == source.cuts().values
                && ev.source.cuts().categorical_flags() == source.cuts().categorical_flags(),
            "validation set {:?} does not reuse the training set cuts",
            ev.name
        );
    }
    let _profile_guard = prof::begin();
    let margin = base_margin(cfg.objective, base_score);
    // host 的训练预测;`gpu_math="fast"` 时 device 是权威副本、host 这份从不
    // 读写,所以下面确定了 fast 之后才按需分配(省 4 B/行的常驻内存)。
    let mut predictions: Vec<f32> = Vec::new();
    // 续训:已有的树先放进模型,新树接在后面;早停的轮次是**整个模型里**的
    // 绝对序号(XGBoost 的 `EarlyStopping` 同样加上 `starting_round`)。
    let prior_trees: &[Tree] = cfg.prior_model.as_ref().map_or(&[], |m| m.trees.as_slice());
    let prior_rounds = prior_trees.len() / num_class;
    let mut model = Model {
        trees: {
            let mut trees = Vec::with_capacity(prior_trees.len() + cfg.params.n_rounds * num_class);
            trees.extend_from_slice(prior_trees);
            trees
        },
        base_score,
        objective: cfg.objective,
        n_features: source.n_features(),
        // 训练层不认识"列名"这个概念(它只见到量化后的块),所以留空,
        // 由知道 schema 的入口层填。空 = 位置模式,也是 numpy 入口的正解。
        feature_names: Vec::new(),
        schema_mode: None,
        class_base_margins: Vec::new(),
        scale_pos_weight: cfg.params.scale_pos_weight,
        feature_types: source.cuts().categorical_flags().to_vec(),
        rank_bias: None,
        // 和 XGBoost 一样,续训时旧模型的 best_iteration / best_score 属性原样保留,
        // 本次若开了早停再覆盖。
        best_iteration: cfg.prior_model.as_ref().and_then(|m| m.best_iteration),
        best_score: cfg.prior_model.as_ref().and_then(|m| m.best_score),
    };

    // 每个验证集缓存一份 raw margin,每轮只把**新加的那棵树**累上去。
    //
    // 原来是每轮拿整个模型重算一遍,那是 O(轮数²):100 轮的任务后半程
    // 光评估就比建树还贵,而 early stopping 的全部意义就是快速迭代。
    // CLAUDE.md 说这个优化要等数值对齐之后再做 —— 现在对齐了,而且有
    // tests/bit_exact_vs_xgboost.rs 兜底,可以动了。
    // `base_margin`(含续训的既有预测)给了就从它起步,否则从截距起步。
    let mut eval_preds: Vec<Vec<f32>> = evals
        .iter()
        .enumerate()
        .map(|(i, e)| match cfg.eval_margin(i) {
            Some(m) if num_class == 1 => m.to_vec(),
            _ => vec![base_margin(cfg.objective, base_score); e.source.n_rows()],
        })
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
            // 16-bit blocks run on the device too (column-major, byte-sized
            // slots); the context allocates for the real block shape.
            let shape = gpu_block_shape(source)?;
            let ctx = crate::backend::cuda::GpuTrainCtx::new_with_shape(
                device,
                n,
                shape,
                source.n_blocks(),
                crate::gpu_mem_plan::GPU_PARTITION_THREADS,
                crate::backend::cuda::SharedHistogramConfig {
                    runtime_log: !cfg.quiet,
                    // 用户显式给的 stream 数走正式参数,不走 env。
                    hist_streams: cfg.params.hist_streams,
                    resident_blocks: cfg.params.resident_blocks,
                    // 规划之后这个字段保存的是**生效值**(见 TrainParams 文档)。
                    gpu_memory_budget: cfg.params.gpu_memory_budget,
                    hist_nodes_per_batch: cfg.params.hist_nodes_per_batch,
                    // Categorical features are enumerated by the device split
                    // cache too (one-hot and sorted-partition kernels); the
                    // flag adds their set outputs to the cache allocation.
                    gpu_split_cache: cfg.params.gpu_split_cache,
                    has_categorical: source.cuts().has_categorical(),
                    node_feature_masks: cfg.params.colsample_bylevel < 1.0
                        || cfg.params.colsample_bynode < 1.0,
                    gpu_math: cfg.params.gpu_math,
                    fast_weights: grad_weights.is_some(),
                    // 规划器(python.rs)和这里必须用同一个 K 计费。
                    n_outputs: num_class,
                    multi_quantile: cfg.objective.is_multi_quantile(),
                    // Device LambdaMART buffers (fast ranking only); the planner
                    // charges the same shape.
                    fast_rank: match (ranking, cfg.group_ptr.as_deref()) {
                        (Some((kind, params)), Some(g)) if fast_requested => Some(
                            crate::gpu_mem_plan::FastRankShape::for_groups(kind, &params, g),
                        ),
                        _ => None,
                    },
                    device_quantize: cfg.params.device_quantize,
                    ..Default::default()
                },
                prof::enabled(),
            )?;
            // The histogram kernels tile a block by whole features. A 16-bit
            // feature wider than one shared-memory tile is accumulated in
            // global memory; an 8-bit feature always fits a tile unless the
            // shared-memory budget is tiny, which is a hard error here, before
            // any launch (never a kernel loop without progress).
            let max_u8_feature_bins = (0..cuts.n_feats() as FeatId)
                .filter(|&f| cuts.bin_width(f) == crate::bins::BinWidth::U8)
                .map(|f| cuts.n_bins(f))
                .max()
                .unwrap_or(0);
            ctx.ensure_features_fit_tile(max_u8_feature_bins)?;
            if prof::enabled() {
                eprintln!("\n=== GPU memory budget ===\n{}", ctx.budget().report());
                eprintln!("{}", ctx.host_budget().report());
                if let Ok((used, total)) = ctx.device_mem_used() {
                    eprintln!(
                        "device memory used after allocation {:.1} MB / {:.1} MB (including fixed driver/context overhead)",
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
    // 多分类有自己的 K 份预测(train_multiclass_rounds),也不需要这一份。
    if !fast_math && num_class == 1 {
        predictions = match &cfg.base_margin {
            Some(m) => m.clone(),
            None => vec![margin; n],
        };
    }
    if fast_math {
        let ctx = gpu.as_ref().expect("fast_math implies a GPU context");
        match &cfg.base_margin {
            // device prediction 直接从逐行 margin 起步(class-major,K × n):
            // 缓冲区本来就按 K × n 分配,规划器不用多记一项。
            Some(m) => ctx.init_fast_math_rows(labels, grad_weights, m)?,
            None => ctx.init_fast_math(
                labels,
                grad_weights,
                // 多分类:K 个 base margin(class-major 的 K 份 device prediction)。
                if num_class > 1 {
                    &class_margins
                } else {
                    std::slice::from_ref(&margin)
                },
            )?,
        }
    }

    // fast ranking: static per-group inputs go to the device once
    if let (true, Some(rg)) = (fast_math, rank_grad.as_ref()) {
        gpu.as_ref()
            .expect("fast_math implies a GPU context")
            .init_fast_rank(&rg.device_inputs())?;
    }

    if num_class > 1 {
        model.class_base_margins = class_margins;
        return train_multiclass_rounds(
            MulticlassRun {
                source,
                labels,
                weights,
                evals,
                cfg,
                comm,
                gpu: gpu.as_ref(),
                pool: &pool,
            },
            model,
            callbacks,
        );
    }

    // 融合路径把「施加 delta」推迟到下一轮开头,所以要记着是否欠一次。
    let mut pending_delta = false;
    // host 量化的定点梯度缓冲,跨轮复用(见 `quantize_host`)。
    let mut fixed_buf: Vec<GradPairFixed> = Vec::new();
    for round in 0..cfg.params.n_rounds {
        let round_started = std::time::Instant::now();
        // 采样种子按模型里的**绝对**轮次取:续训接着旧模型的轮次往下走,
        // 于是 N 轮 + M 轮和一次训 N+M 轮在同一份数据上完全相同(含 subsample /
        // colsample);`round` 仍是本次的相对轮次,给 callback 和 evals_result 用。
        let key_round = prior_rounds + round;
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
        let round_seed = round_seeds.as_mut().map_or(0, |r| r.draw());
        // fast 模式下 host 侧完全不产生 gpair:prediction apply、objective
        // gradient 和量化在 device 上一次做完。
        let exact_gradient_phase =
            (!fast_math).then(|| crate::diagnostic::Phase::new(b"fb.host.exact.gradient\0"));
        let gpair_f32 = if fast_math {
            Vec::new()
        } else if fuse {
            let ctx = gpu.as_ref().expect("pending_delta implies a GPU context");
            let obj = cfg.objective;
            let mut out: Vec<GradPair> = Vec::new();
            pool.install(|| -> anyhow::Result<()> {
                let _t = prof::Timer::new(&prof::GRAD);
                ctx.with_prediction_delta(&mut |delta| {
                    if let Some(rg) = rank_grad.as_ref() {
                        // Group gradients need every updated score: apply the
                        // delta first, then compute them.
                        predictions
                            .par_iter_mut()
                            .zip(delta.par_iter())
                            .for_each(|(p, d)| *p += *d);
                        rg.gradients(&predictions, round_seed, &mut out);
                        return;
                    }
                    if obj.is_quantile() {
                        // 梯度依赖全体行的更新后预测(全局 scale):先施加 delta
                        // 再整体求梯度,不能边施加边算。
                        predictions
                            .par_iter_mut()
                            .zip(delta.par_iter())
                            .for_each(|(p, d)| *p += *d);
                        let qs = Objective::quantile_scale(&predictions, labels, weights);
                        out = match grad_weights {
                            None => compute_gradients(&predictions, labels, obj, qs),
                            Some(w) => compute_gradients_weighted(&predictions, labels, w, obj, qs),
                        };
                        return;
                    }
                    out = match grad_weights {
                        None => predictions
                            .par_iter_mut()
                            .zip(delta.par_iter())
                            .zip(labels.par_iter())
                            .map(|((p, d), &y)| {
                                *p += *d;
                                gradient_of(*p, y, obj, 0.0)
                            })
                            .collect(),
                        Some(w) => predictions
                            .par_iter_mut()
                            .zip(delta.par_iter())
                            .zip(labels.par_iter().zip(w.par_iter()))
                            .map(|((p, d), (&y, &w))| {
                                *p += *d;
                                weighted(gradient_of(*p, y, obj, 0.0), w)
                            })
                            .collect(),
                    };
                })
            })?;
            pending_delta = false;
            out
        } else {
            pool.install(|| {
                let _t = prof::Timer::new(&prof::GRAD);
                if let Some(rg) = rank_grad.as_ref() {
                    let mut out = Vec::new();
                    rg.gradients(&predictions, round_seed, &mut out);
                    return out;
                }
                // quantile 的 scale 是对全体行预测的串行 f64 归约(其它
                // objective 不用,传 0)。
                let qs = if cfg.objective.is_quantile() {
                    Objective::quantile_scale(&predictions, labels, weights)
                } else {
                    0.0
                };
                match grad_weights {
                    None => compute_gradients(&predictions, labels, cfg.objective, qs),
                    Some(w) => {
                        compute_gradients_weighted(&predictions, labels, w, cfg.objective, qs)
                    }
                }
            })
        };
        drop(exact_gradient_phase);
        // ⬅ **行采样在这里发生**:完整梯度已经算出来,root sum / 直方图还没开始。
        //
        // ⚠️ 改的是**工作副本**:未选中的行 gpair 置零,于是它们对直方图和
        // node_sum 贡献零(定点零加上去什么都不改)。**它们仍然跟着每一次
        // 分裂走、仍然走到叶子、仍然拿到这棵树的预测增量** ——
        // 采样掩码**不是**预测掩码。
        //
        // uniform 采样**不做 `1/p` 加权**(那是 gradient-based 采样才需要的)。
        let mut gpair_f32 = gpair_f32;
        let device_quant = gpu.as_ref().is_some_and(|c| c.device_quantize_enabled());
        // Device quantization masks in its existing max scan before selecting
        // scale. CPU and explicit host quantization mask in `mask_and_bounds`
        // below, fused with the quantization bound scan.
        // device-side quantization:求 scale、转定点、归约 root sum 都在 device 上,
        // 走 PCIe 的是 f32(8 B/行)而不是定点(16 B/行)。位精确的理由见
        // `GpuTrainCtx::upload_and_quantize_gpair`。CPU 路径完全不变。
        let quantize_phase = crate::diagnostic::Phase::new(b"fb.host.quantize\0");
        // `known_root_sum`:根节点的和已经顺手求出(device 量化、或 host 转定点
        // 那一趟);`host_nonzero`:host 转定点时记下的非零行位图。
        let (quantizer, fixed, known_root_sum, host_nonzero) = if fast_math {
            // prediction D2H、host 按行 gradient、gpair H2D —— 三项都不存在。
            // ⚠️ **不保证 CPU/GPU 逐字节相同**,这是 `gpu_math="fast"` 的定义。
            let ctx = gpu.as_ref().expect("fast_math implies a GPU context");
            let _t = prof::Timer::new(&prof::QUANTIZE);
            // 行采样在 device 上完成:阈值 host 折算,device 只比整数。
            // ⚠️ kernel 里 prediction 更新是**无条件**的,只有 gpair 被置零 ——
            // 采样掩码不是预测掩码。
            let (q, root, _ms) = if rank_grad.is_some() {
                ctx.fast_rank_round(
                    round_seed,
                    crate::subsample::threshold_for(cfg.params.subsample),
                    cfg.params.seed,
                    key_round as u64,
                )?
            } else {
                ctx.fast_gradient_and_quantize(
                    cfg.objective,
                    crate::subsample::threshold_for(cfg.params.subsample),
                    cfg.params.seed,
                    key_round as u64,
                )?
            };
            // delta 已经被 device kernel 吃掉,不再欠 host 一次施加。
            pending_delta = false;
            (q, Vec::new(), Some(root), None)
        } else if device_quant {
            let ctx = gpu.as_ref().expect("device_quant implies a GPU context");
            let _t = prof::Timer::new(&prof::QUANTIZE);
            let (q, root, ms) = ctx.upload_and_quantize_gpair_sampled(
                &gpair_f32,
                crate::subsample::threshold_for(cfg.params.subsample),
                cfg.params.seed,
                key_round as u64,
            )?;
            drop(_t);
            if prof::enabled() {
                prof::GPU_GPAIR_H2D.fetch_add(
                    (ms as f64 * 1e6) as u64,
                    std::sync::atomic::Ordering::Relaxed,
                );
                prof::GPU_GPAIR_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                prof::GPU_GPAIR_BYTES.fetch_add(
                    (gpair_f32.len() * std::mem::size_of::<GradPair>()) as u64,
                    std::sync::atomic::Ordering::Relaxed,
                );
            }
            (q, Vec::new(), Some(root), None)
        } else {
            pool.install(|| {
                // 采样掩码和求 scale 的 max 合成一趟(以前是采样一趟、max 一趟)。
                // f32 侧置零:量化在这之后发生,所以 scale 也只看选中的行。
                let mask = (cfg.params.subsample < 1.0).then(|| {
                    (
                        crate::subsample::threshold_for(cfg.params.subsample),
                        cfg.params.seed,
                        key_round as u64,
                    )
                });
                let (max_grad, max_hess, kept) = {
                    let _phase = crate::diagnostic::Phase::new(b"fb.host.exact.subsample\0");
                    let _t = prof::Timer::new(&prof::SAMPLE);
                    mask_and_bounds(&mut gpair_f32, mask)
                };
                if prof::enabled() && round == 0 && mask.is_some() {
                    eprintln!(
                        "SUBSAMPLE rate={} kept={kept}/{} ({:.1}%)",
                        cfg.params.subsample,
                        gpair_f32.len(),
                        100.0 * kept as f64 / gpair_f32.len().max(1) as f64
                    );
                }
                let _t = prof::Timer::new(&prof::QUANTIZE);
                let quantizer = GradQuantizer::try_from_bounds(max_grad, max_hess, n)?;
                // 定点缓冲跨轮复用:每轮新分配 16 B/行要重新缺页一遍。
                // 它本来就活过整个 build_tree,复用不抬高峰值。
                let mut fixed = std::mem::take(&mut fixed_buf);
                let (root, nonzero) = quantize_host(&gpair_f32, &quantizer, &mut fixed);
                Ok::<_, anyhow::Error>((quantizer, fixed, Some(root), Some(nonzero)))
            })?
        };
        drop(quantize_phase);
        {
            let _phase = crate::diagnostic::Phase::new(b"fb.host.gpair_release\0");
            drop(gpair_f32);
        }
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
                key_round,
            ))
        } else {
            None
        };
        let tree = pool.install(|| {
            build_tree(
                source,
                &fixed,
                known_root_sum,
                host_nonzero,
                &quantizer,
                &mut predictions,
                cfg,
                comm,
                gpu.as_ref(),
                sampled.as_deref(),
                crate::colsample::NestedSampler::new(
                    cfg.params.colsample_bylevel,
                    cfg.params.colsample_bynode,
                    cfg.params.seed,
                    key_round,
                ),
            )
        })?;
        fixed_buf = fixed;
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
        // Every callback must run even after an earlier callback requests stopping.
        #[allow(clippy::unnecessary_fold)]
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
                    "\ndevice memory used at end of training {:.1} MB / {:.1} MB; preallocated by this process {:.1} MB",
                    used as f64 / 1048576.0,
                    total as f64 / 1048576.0,
                    ctx.budget().total() as f64 / 1048576.0
                );
            }
        }
        prof::report(completed_rounds);
        eprintln!(
            "actual round wall{:>15.2} ms/round",
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
        model.best_iteration = Some(iter + prior_rounds);
        model.best_score = Some(score);
    }
    // Unbiased LambdaMART: the position biases as of the last round, saved with the
    // objective config like XGBoost does.
    // With `gpu_math = "fast"` the estimate lives on the device (the host state was
    // never advanced).
    let rank_bias = match (fast_math, gpu.as_ref()) {
        (true, Some(ctx)) if rank_grad.is_some() => ctx.fast_rank_bias()?,
        _ => rank_grad.as_ref().and_then(|rg| rg.position_bias()),
    };
    if let Some((ti_plus, tj_minus)) = rank_bias {
        model.rank_bias = Some(crate::tree::RankBias { ti_plus, tj_minus });
    }

    if !cfg.quiet {
        if let (Some(before), Some(after)) = (source_bytes_start, source.source_bytes_read()) {
            eprintln!(
                "TRAIN_SOURCE quantized_bytes_read={}",
                after.saturating_sub(before)
            );
        }
        if let (Some(before), Some(after)) = (source_reads_start, source.source_block_reads()) {
            eprintln!("TRAIN_SOURCE block_reads={}", after.saturating_sub(before));
        }
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
        anyhow::bail!("labels length does not match the source row count");
    }
    let base_score = cfg.resolve_base_score(labels, None);
    let margin = base_margin(cfg.objective, base_score);
    let predictions = vec![margin; source.n_rows()];
    let qs = if cfg.objective.is_quantile() {
        Objective::quantile_scale(&predictions, labels, None)
    } else {
        0.0
    };
    let gpair = compute_gradients(&predictions, labels, cfg.objective, qs);
    let quantizer = GradQuantizer::try_new(&gpair, source.n_rows())?;
    let fixed: Vec<GradPairFixed> = gpair.iter().map(|&g| quantizer.to_fixed(g)).collect();
    let root_sum = hist::node_sum(&fixed, None);
    let mut tree = Tree::default();
    let root = tree.push_leaf(
        leaf_of(root_sum, &quantizer, &cfg.params),
        hess_of(root_sum, &quantizer),
    );

    let cuts = source.cuts();
    anyhow::ensure!(
        !cuts.has_categorical(),
        "train_gpu_one_tree does not support categorical features"
    );
    let mut best: Option<SplitCandidate> = None;
    for b in 0..source.n_blocks() {
        let block_best = source.with_block_ret(b, |block| {
            let offsets = cuts.block_hist_offsets(&block.feat_ids);
            let hist = build_histogram_shared(
                device_ordinal,
                block.bins().require_u8("GPU training")?,
                block.n_rows,
                block.n_feats(),
                &offsets,
                &fixed,
                None,
            )?;
            Ok::<_, anyhow::Error>(split::best_split_in_block_constrained(
                &Histogram { bins: hist },
                &block.feat_ids,
                &offsets,
                root_sum,
                &quantizer,
                &cfg.params,
                None,
                monotone_at(&cfg.params, split::NodeBounds::UNBOUNDED),
            ))
        })??;
        best = split::reduce_best(&[best, block_best]);
    }

    let Some(chosen) = comm.allreduce_best_split(best) else {
        return Ok(Model {
            // 训练层没有列名概念;由知道 schema 的入口层填。
            feature_names: Vec::new(),
            schema_mode: None,
            class_base_margins: Vec::new(),
            scale_pos_weight: cfg.params.scale_pos_weight,
            feature_types: Vec::new(),
            rank_bias: None,
            trees: vec![tree],
            base_score,
            objective: cfg.objective,
            n_features: source.n_features(),
            best_iteration: None,
            best_score: None,
        });
    };
    let children = child_leaves(
        &chosen,
        &quantizer,
        &cfg.params,
        split::NodeBounds::UNBOUNDED,
    );
    let (left_node, right_node) = tree.split_leaf(
        root,
        chosen.feat,
        tree::split_threshold(cuts, chosen.feat, chosen.bin),
        chosen.missing_left,
        chosen.gain,
        children.left_value,
        hess_of(chosen.left, &quantizer),
        children.right_value,
        hess_of(chosen.right, &quantizer),
    );
    // 选中的列也按需读取;GPU partition 只接收这一个 column,不把整块
    // 常驻。结果用于闭环验收,后续多层接线会把它挂到 LevelNode。
    let col = source.feature_column(chosen.feat)?;
    let rows: Vec<RowId> = (0..source.n_rows() as RowId).collect();
    let (left_rows, right_rows) = crate::backend::cuda::partition_rows_gpu(
        device_ordinal,
        col.as_slice().require_u8("GPU training")?,
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
        class_base_margins: Vec::new(),
        scale_pos_weight: cfg.params.scale_pos_weight,
        feature_types: Vec::new(),
        rank_bias: None,
        trees: vec![tree],
        base_score,
        objective: cfg.objective,
        n_features: source.n_features(),
        best_iteration: None,
        best_score: None,
    })
}

/// 多分类训练需要的只读上下文。
struct MulticlassRun<'a> {
    source: &'a dyn BlockSource,
    labels: &'a [f32],
    weights: Option<&'a [f32]>,
    evals: &'a [EvalSet<'a>],
    cfg: &'a TrainConfig,
    comm: &'a dyn Comm,
    gpu: Option<&'a crate::backend::cuda::GpuTrainCtx>,
    pool: &'a rayon::ThreadPool,
}

/// 多分类(`multi:softprob` / `multi:softmax`,XGBoost `one_output_per_tree`)。
///
/// 每轮:按行算一次 softmax 梯度(K 列),然后 k = 0..K 各建一棵单输出树,
/// 复用和单输出完全相同的建树路径。和单输出循环的区别:
/// - 预测是 K 份 `n` 长的 margin;第 k 棵树只累加到第 k 份;
/// - 行采样 / 列采样的「轮次」键是**树序号** `round·K + k`,每棵树独立抽样
///   (XGBoost 也是每棵树各采一次);K = 1 时就是原来的 `round`;
/// - GPU 的叶子增量在每棵树之后立即施加(单输出路径那种「推迟到下一轮
///   梯度那一遍」的融合在这里不适用:下一棵树要用的是另一类的梯度);
/// - `gpu_math="fast"`:device 持有 `K × n` 的权威 prediction(host 的
///   `class_preds` 不分配、不更新)。每轮一个 kernel 算一次每行的 softmax 统计量,
///   每个类再一个 kernel 写出该类的 f32 gpair(含行采样,键同样是树序号)并走
///   通用 device 量化;建完树后 delta 在 device 上直接加到该类的列。
///   梯度每轮只算一次(而不是 K 次),且 host 没有任何 O(n·K) 的搬运。
fn train_multiclass_rounds(
    run: MulticlassRun<'_>,
    mut model: Model,
    callbacks: &mut [Box<dyn Callback>],
) -> anyhow::Result<Model> {
    let MulticlassRun {
        source,
        labels,
        weights,
        evals,
        cfg,
        comm,
        gpu,
        pool,
    } = run;
    let n = source.n_rows();
    let k_count = cfg.objective.num_outputs();
    let margins = model.class_base_margins.clone();
    // Absolute round of the first new tree (continuation): sampling seeds key on
    // it, so continuing equals uninterrupted training on the same data.
    let prior_rounds = model.trees.len() / k_count;
    let model_prior_rounds = prior_rounds;
    // fast:device 是 prediction 的权威副本,host 这份不存在。
    let fast_math = gpu.is_some_and(|c| c.fast_math_enabled());
    // 逐行 base_margin(class-major,K × n)给了就按类切开作为起点,否则每类一个截距。
    let split_classes = |m: &[f32], rows: usize| -> Vec<Vec<f32>> {
        m.chunks_exact(rows).map(<[f32]>::to_vec).collect()
    };
    let mut class_preds: Vec<Vec<f32>> = if fast_math {
        Vec::new()
    } else if let Some(m) = &cfg.base_margin {
        split_classes(m, n)
    } else {
        margins.iter().map(|&m| vec![m; n]).collect()
    };
    let mut eval_class_preds: Vec<Vec<Vec<f32>>> = evals
        .iter()
        .enumerate()
        .map(|(i, e)| match cfg.eval_margin(i) {
            Some(m) => split_classes(m, e.source.n_rows()),
            None => margins
                .iter()
                .map(|&m| vec![m; e.source.n_rows()])
                .collect(),
        })
        .collect();
    let mut grads: Vec<Vec<GradPair>> = vec![Vec::new(); k_count];
    let device_quant = gpu.is_some_and(|c| c.device_quantize_enabled());
    let multi_quantile = cfg.objective.is_multi_quantile();
    let mut fixed_buf: Vec<GradPairFixed> = Vec::new();

    for round in 0..cfg.params.n_rounds {
        let round_started = std::time::Instant::now();
        if fast_math {
            // 多分位数没有每轮的行统计量:每个目标的 scale 在它自己的 kernel 里归约。
            if !multi_quantile {
                let _t = prof::Timer::new(&prof::GRAD);
                gpu.expect("fast_math implies a GPU context")
                    .fast_multiclass_begin_round()?;
            }
        } else {
            pool.install(|| {
                let _t = prof::Timer::new(&prof::GRAD);
                match cfg.objective.quantile_alphas() {
                    Some(alphas) => quantile_multi_gradients(
                        alphas.as_slice(),
                        &class_preds,
                        labels,
                        weights,
                        &mut grads,
                    ),
                    None => {
                        cfg.objective
                            .softmax_gradients(&class_preds, labels, weights, &mut grads)
                    }
                }
            });
        }
        for class in 0..k_count {
            let tree_index = ((prior_rounds + round) * k_count + class) as u64;
            let mut gpair_f32 = if fast_math {
                Vec::new()
            } else {
                std::mem::take(&mut grads[class])
            };
            // 行采样:和单输出相同的确定性掩码,键是树序号。host 量化路径上
            // 掩码与求 scale 的 max 合成一趟(见 `mask_and_bounds`)。
            let (quantizer, fixed, dev_root_sum, host_nonzero) = if fast_math {
                // ⚠️ **不保证 CPU/GPU 逐字节相同**,这是 `gpu_math="fast"` 的定义。
                // 行采样在 gradient kernel 里完成;prediction 的更新是另一个
                // kernel、无条件执行 —— 采样掩码不是预测掩码。
                let ctx = gpu.expect("fast_math implies a GPU context");
                let _t = prof::Timer::new(&prof::QUANTIZE);
                let (q, root, _ms) = if multi_quantile {
                    ctx.fast_multiquantile_gradient_and_quantize(
                        class,
                        cfg.objective
                            .quantile_alphas()
                            .expect("multi quantile")
                            .as_slice()[class],
                        crate::subsample::threshold_for(cfg.params.subsample),
                        cfg.params.seed,
                        tree_index,
                    )?
                } else {
                    ctx.fast_multiclass_gradient_and_quantize(
                        class,
                        crate::subsample::threshold_for(cfg.params.subsample),
                        cfg.params.seed,
                        tree_index,
                    )?
                };
                (q, Vec::new(), Some(root), None)
            } else if device_quant {
                let ctx = gpu.expect("device_quant implies a GPU context");
                let _t = prof::Timer::new(&prof::QUANTIZE);
                let (q, root, _ms) = ctx.upload_and_quantize_gpair_sampled(
                    &gpair_f32,
                    crate::subsample::threshold_for(cfg.params.subsample),
                    cfg.params.seed,
                    tree_index,
                )?;
                (q, Vec::new(), Some(root), None)
            } else {
                pool.install(|| {
                    let mask = (cfg.params.subsample < 1.0).then(|| {
                        (
                            crate::subsample::threshold_for(cfg.params.subsample),
                            cfg.params.seed,
                            tree_index,
                        )
                    });
                    let (max_grad, max_hess, _) = {
                        let _t = prof::Timer::new(&prof::SAMPLE);
                        mask_and_bounds(&mut gpair_f32, mask)
                    };
                    let _t = prof::Timer::new(&prof::QUANTIZE);
                    let quantizer = GradQuantizer::try_from_bounds(max_grad, max_hess, n)?;
                    // 定点缓冲在各类、各轮之间复用(见 `quantize_host`)。
                    let mut fixed = std::mem::take(&mut fixed_buf);
                    let (root, nonzero) = quantize_host(&gpair_f32, &quantizer, &mut fixed);
                    Ok::<_, anyhow::Error>((quantizer, fixed, Some(root), Some(nonzero)))
                })?
            };
            drop(gpair_f32);
            if let (Some(ctx), false) = (gpu, device_quant) {
                ctx.upload_gpair(&fixed)?;
            }
            let sampled: Option<Vec<FeatId>> = if cfg.params.colsample_bytree < 1.0 {
                Some(crate::colsample::sample_features(
                    source.n_features(),
                    cfg.params.colsample_bytree,
                    cfg.params.seed,
                    tree_index as usize,
                ))
            } else {
                None
            };
            // fast:GPU 建树不读写 host prediction,给空切片。
            let mut no_host_pred: [f32; 0] = [];
            let tree = pool.install(|| {
                build_tree(
                    source,
                    &fixed,
                    dev_root_sum,
                    host_nonzero,
                    &quantizer,
                    if fast_math {
                        &mut no_host_pred
                    } else {
                        &mut class_preds[class]
                    },
                    cfg,
                    comm,
                    gpu,
                    sampled.as_deref(),
                    crate::colsample::NestedSampler::new(
                        cfg.params.colsample_bylevel,
                        cfg.params.colsample_bynode,
                        cfg.params.seed,
                        tree_index as usize,
                    ),
                )
            })?;
            fixed_buf = fixed;
            // fast:delta 在 device 上直接加到第 `class` 列,没有任何 D2H。
            if fast_math && multi_quantile {
                // 下一棵树(下一个目标)会覆盖 delta buffer:立刻并进这个目标的列。
                gpu.expect("fast_math implies a GPU context")
                    .fast_multiquantile_apply_delta(class)?;
            } else if fast_math {
                gpu.expect("fast_math implies a GPU context")
                    .fast_multiclass_apply_delta(class)?;
            }
            // GPU 路径的叶子增量还在 device 回传的 buffer 里:立即施加到这一类。
            else if let Some(ctx) = gpu {
                let preds = &mut class_preds[class];
                let mut applied = Ok(());
                pool.install(|| {
                    applied = ctx.with_prediction_delta(&mut |delta| {
                        preds
                            .par_iter_mut()
                            .zip(delta.par_iter())
                            .for_each(|(p, d)| *p += *d);
                    });
                });
                applied?;
            }
            {
                let _t = prof::Timer::new(&prof::EVAL);
                for (ev, preds) in evals.iter().zip(eval_class_preds.iter_mut()) {
                    update_predictions(&mut preds[class], &tree, ev.source)?;
                }
            }
            model.trees.push(tree);
        }

        // 指标要按行排的 n × K margin。
        let interleaved: Vec<Vec<f32>> = eval_class_preds
            .iter()
            .map(|per_class| interleave_classes(per_class))
            .collect();
        let m = evaluate(&interleaved, evals, round, cfg);
        #[allow(clippy::unnecessary_fold)]
        let keep_going = callbacks
            .iter_mut()
            .map(|cb| cb.after_iteration(&m))
            .fold(true, |a, b| a && b);
        if prof::enabled() {
            prof::ROUND_WALL.fetch_add(
                round_started.elapsed().as_nanos() as u64,
                std::sync::atomic::Ordering::Relaxed,
            );
        }
        if !keep_going {
            break;
        }
    }
    for cb in callbacks.iter_mut() {
        cb.after_training();
    }
    if let Some((iter, score)) = callbacks.iter().find_map(|cb| cb.best_iteration()) {
        model.best_iteration = Some(iter + model_prior_rounds);
        model.best_score = Some(score);
    }
    Ok(model)
}

/// 多分位数一轮的梯度:每个 alpha(= 目标)各自的 scale 和逐行梯度,
/// `out[k]` 是第 k 个 alpha 的梯度列(已乘样本权重)。
///
/// XGBoost `QuantileRegression::GetGradient`:每个目标独立求
/// `S_j = (Σ w·sqrt|p_j − y| / Σ w)²`(f32 逐行乘积、f64 串行累加),所以和单 alpha
/// 路径逐位相同,而且目标之间互不影响 —— 第 k 棵树只改第 k 列预测,
/// 一轮开头一次算完和每棵树前各算一次结果一样。
fn quantile_multi_gradients(
    alphas: &[f32],
    class_preds: &[Vec<f32>],
    labels: &[f32],
    weights: Option<&[f32]>,
    out: &mut [Vec<GradPair>],
) {
    debug_assert_eq!(alphas.len(), class_preds.len());
    out.par_iter_mut()
        .zip(class_preds.par_iter())
        .zip(alphas.par_iter())
        .for_each(|((col, preds), &alpha)| {
            let qs = Objective::quantile_scale(preds, labels, weights);
            let obj = Objective::quantile(alpha);
            *col = match weights {
                None => compute_gradients(preds, labels, obj, qs),
                Some(w) => compute_gradients_weighted(preds, labels, w, obj, qs),
            };
        });
}

/// K 份按类排的预测 → 按行排的 `n × K`。
fn interleave_classes(per_class: &[Vec<f32>]) -> Vec<f32> {
    let k = per_class.len();
    let n = per_class.first().map_or(0, Vec::len);
    let mut out = vec![0.0f32; n * k];
    for (c, col) in per_class.iter().enumerate() {
        for (i, &v) in col.iter().enumerate() {
            out[i * k + c] = v;
        }
    }
    out
}

/// 在每个验证集上算每个指标。
///
/// 预测值是外面增量维护的(每轮只累加新树),这里只负责套指标。
/// 顺序是「eval 外层、metric 内层」,所以 `RoundMetrics::last()` 拿到的
/// 是**最后一个 eval 的最后一个 metric**,也是 early stopping 的默认指标。
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
                m.eval_grouped(pred, ev.labels, ev.weights, ev.group_ptr),
            ));
        }
    }
    RoundMetrics { round, entries }
}

#[cfg(test)]
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
            if n.is_leaf || n.cats.is_some() {
                0
            } else {
                tree::split_bin(cuts, n.feat, n.split_cond)
            }
        })
        .collect();
    // Categorical nodes compare the bin (= category code) against a set.
    let node_cats: Vec<Option<crate::categorical::CatBits>> = t
        .nodes
        .iter()
        .map(|n| {
            n.cats.as_ref().map(|words| {
                // Only trained codes (`< n_bins`) can be a bin of this feature;
                // a larger member (from a continued model) must never match
                // the width's unseen-category bin.
                let n_bins = cuts.n_bins(n.feat);
                let mut bits = crate::categorical::CatBits::for_bins(n_bins);
                for cat in crate::categorical::categories_from_words(words) {
                    if (cat as usize) < n_bins {
                        bits.set(cat);
                    }
                }
                bits
            })
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
                    "prediction: column block {b} has {} rows, but the source declares {n_rows}",
                    block.n_rows
                );
                anyhow::ensure!(
                    block.data.len() == block.n_rows.saturating_mul(block.n_feats()),
                    "prediction: column block {b} data length does not match rows × features"
                );
                anyhow::ensure!(
                    block.width() == source.cuts().block_width(&block.feat_ids),
                    "prediction: column block {b} bin width does not match the cuts"
                );
                anyhow::ensure!(
                    block.feat_ids == block_feats[b],
                    "prediction: column block {b} features do not match the block_features metadata"
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
                        let col = block.column(local);
                        let bin = col.get(row as usize);
                        let go_left = if bin == col.width().missing() {
                            n.default_left
                        } else if let Some(cats) = &node_cats[*node as usize] {
                            !cats.contains(bin as u32)
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
    anyhow::ensure!(
        done,
        "some rows did not reach a leaf; the tree structure is invalid"
    );

    Ok(at.iter().map(|&n| t.nodes[n as usize].leaf_value).collect())
}

/// 本层里的一个活跃节点。
struct LevelNode {
    /// 在 tree.nodes 里的下标
    node: usize,
    /// 该节点持有的行
    rows: NodeRows,
    /// 梯度总和,**含缺失行** —— split.rs 靠它反推缺失部分
    sum: GradPairFixed,
    hist_from: HistFrom,
    /// 单调约束下这个节点的权重上下界。无约束时恒为 `UNBOUNDED`、不被读取。
    bounds: split::NodeBounds,
}

enum NodeRows {
    /// 本层宿主行缓冲里的一段(见 [`HostSpan`])。
    Host(HostSpan),
    Device(crate::backend::cuda::GpuRowSpan),
}

impl NodeRows {
    fn len(&self) -> usize {
        match self {
            Self::Host(span) => span.len,
            Self::Device(span) => span.len,
        }
    }

    fn host<'a>(&self, buf: &'a [RowId]) -> &'a [RowId] {
        match self {
            Self::Host(span) => span.of(buf),
            Self::Device(_) => panic!("device rows were sent to the CPU path"),
        }
    }

    fn device(&self) -> crate::backend::cuda::GpuRowSpan {
        match self {
            Self::Device(span) => *span,
            Self::Host(_) => panic!("host rows were sent to the GPU path"),
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

#[allow(clippy::too_many_arguments)] // Keep the established source/backend ownership interface explicit.
/// Split results a two-phase GPU level submission hands to the per-block pass.
#[cfg(feature = "cuda")]
type PreSplit = Vec<crate::backend::cuda::GpuSplitResult>;
#[cfg(not(feature = "cuda"))]
type PreSplit = std::convert::Infallible;

fn build_tree(
    source: &dyn BlockSource,
    gpair: &[GradPairFixed],
    // GPU device-quantization 路径上 root 的 node_sum 已经在 device 上归约好,
    // host 那份定点数组因此不再物化;CPU 路径传 None,照常从 `gpair` 求。
    known_root_sum: Option<GradPairFixed>,
    // host 转定点那一趟记下的非零行位图(见 `quantize_host`);`None` 时
    // 下面自己扫 `gpair`。
    host_nonzero: Option<NonzeroBits>,
    q: &GradQuantizer,
    predictions: &mut [f32],
    cfg: &TrainConfig,
    comm: &dyn Comm,
    gpu: Option<&crate::backend::cuda::GpuTrainCtx>,
    // 本棵树的采样特征集(全局 id,升序)。`None` = 全选。
    sampled: Option<&[FeatId]>,
    // `colsample_bylevel` / `colsample_bynode`:只缩小 split 枚举的特征集,
    // 直方图仍按 `sampled` 建。`None` = 两者都是 1。
    nested: Option<crate::colsample::NestedSampler>,
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
        /// 直方图的逻辑特征。column-major GPU 可只含采样列,即使
        /// resident `handed_feats` 仍保持完整物理块。
        hist_feats: Vec<FeatId>,
        /// 与 `hist_feats` 一一对应的紧凑 histogram offsets。
        hist_offsets: Vec<u32>,
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
    // 直方图只建采样中的列:CPU 恒如此(内核按列切片取列,不需要压实块),
    // column-major GPU 视后端能力而定。没采样的列根本不进 split 枚举,
    // 给它们建直方图是纯白算 —— colsample_bytree=0.6 时那是近一半的累加。
    let compact_hist = |physical: usize| {
        gpu.map_or(true, |c| {
            c.supports_compact_column_histogram_for(cuts.block_width(&all_block_feats[physical]))
        })
    };
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
                let hist_feats = if compact_hist(physical) {
                    sampled_global.clone()
                } else {
                    handed_feats.clone()
                };
                let hist_offsets = cuts.block_hist_offsets(&hist_feats);
                BlockPlan {
                    physical,
                    sampled_global,
                    orig_local,
                    handed_feats,
                    hist_feats,
                    hist_offsets,
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
                hist_feats: all_block_feats[physical].clone(),
                hist_offsets: cuts.block_hist_offsets(&all_block_feats[physical]),
                compact: false,
            })
            .collect(),
    };
    let n_blocks = plans.len();

    let mut t = Tree::default();
    let setup_timer = prof::Timer::new(&prof::TREE_SETUP);
    let root_sum = known_root_sum.unwrap_or_else(|| hist::node_sum(gpair, None));
    let root = t.push_leaf(leaf_of(root_sum, q, p), hess_of(root_sum, q));

    // 梯度全零的行(行采样没选中的行,或梯度恰好为零)对每个直方图 bin 和
    // 每个节点和都贡献零,建树只靠这些和,所以它们不必进入行集合:
    // 直方图和分区都只处理其余的行,而最终模型逐位相同。它们照常拿到
    // 这棵树的预测增量(采样掩码不是预测掩码),只是建完树之后沿树
    // 一次性下降,而不是随每层分区一起搬 —— 见 `apply_dropped_rows`。
    // 只在 CPU、source 的每一列都能随机访问、且确实能省下可观的行数时启用。
    let kept_rows: Option<Vec<RowId>> = if gpu.is_none() && source.has_resident_columns() {
        let worth = |kept: usize| kept <= gpair.len() - gpair.len() / 8;
        match &host_nonzero {
            // 位图已经给出个数:不划算时连行下标都不写。
            Some(bits) => worth(bits.count()).then(|| bits.rows()),
            None => {
                let kept = nonzero_rows(gpair);
                worth(kept.len()).then_some(kept)
            }
        }
    } else {
        None
    };
    let drop_zero_rows = kept_rows.is_some();
    // 每个已分裂节点的 (特征, 分区用的 bin 边界, 缺失走左):下降必须和分区
    // 用同一套比较,不经过浮点阈值往返。
    let mut split_info: Vec<DropSplit> = Vec::new();

    // CPU:本层所有节点的行下标住在 `rows_cur` 里,分区写进 `rows_next`
    // 同一段位置,每层交换(见 `partition_level_host`)。GPU 两个都空着。
    let mut rows_cur: Vec<RowId> = Vec::new();
    let mut rows_next: Vec<RowId> = Vec::new();
    let root_rows = match gpu {
        Some(ctx) => NodeRows::Device(ctx.begin_tree_rows()?),
        None => {
            rows_cur = match kept_rows {
                Some(kept) => kept,
                None => identity_rows(source.n_rows()),
            };
            NodeRows::Host(HostSpan {
                start: 0,
                len: rows_cur.len(),
            })
        }
    };
    drop(setup_timer);
    let mut level = vec![LevelNode {
        node: root,
        rows: root_rows,
        sum: root_sum,
        hist_from: HistFrom::Accumulate,
        bounds: split::NodeBounds::UNBOUNDED,
    }];
    // 上一层的直方图,[列块][槽位]。兄弟相减要用,所以整层都得留着。
    let mut parent_hists: Vec<Vec<Histogram>> = Vec::new();
    // bylevel / bynode 的起点:本树的 bytree 集合(没采样 = 全部特征)。
    let tree_features: Vec<FeatId> = match (nested, sampled) {
        (Some(_), Some(s)) => s.to_vec(),
        (Some(_), None) => (0..source.n_features() as FeatId).collect(),
        (None, _) => Vec::new(),
    };

    for depth in 0..p.max_depth {
        if level.is_empty() {
            break;
        }
        // ---- 本层每个节点的 split 枚举特征集 ----
        //
        // 层集合每层采一次,节点集合每节点采一次(嵌套:节点 ⊂ 层 ⊂ 树)。
        // 种子只看 (seed, 树, 深度 / 节点号),CPU 与 GPU 同一个集合。
        // 嵌套采样激活时 device split cache 按 (节点, 特征) 上传 host 算好的
        // 字节掩码,与 host 枚举用同一个 `enum_features` 集合。
        let level_set: Option<Vec<FeatId>> = nested.map(|n| n.level_set(&tree_features, depth));
        let node_sets: Option<Vec<Vec<FeatId>>> = match (nested, level_set.as_deref()) {
            (Some(n), Some(ls)) if n.samples_nodes() => Some(
                level
                    .iter()
                    .map(|ln| n.node_set(ls, ln.node))
                    .collect::<Vec<_>>(),
            ),
            _ => None,
        };
        // 节点 i 枚举用的特征集:节点集 > 层集 > bytree 集。
        let enum_features = |i: usize| -> Option<&[FeatId]> {
            match (&node_sets, &level_set) {
                (Some(ns), _) => Some(ns[i].as_slice()),
                (None, Some(ls)) => Some(ls.as_slice()),
                (None, None) => sampled,
            }
        };
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
        let hist_phase = crate::diagnostic::Phase::new(b"fb.host.hist_phase\0");

        // 本层要交给 device split cache 的 (small, large) 组。两阶段提交和
        // 逐块路径共用它。
        #[cfg(feature = "cuda")]
        let gpu_split_groups = |session: &mut crate::backend::cuda::GpuBlockSession<'_>| {
            use crate::backend::cuda::GpuSplitGroup;
            let mut groups = Vec::new();
            if depth == 0 {
                groups.push(GpuSplitGroup {
                    small: 0,
                    span: level[0].rows.device(),
                    sum: level[0].sum,
                    bounds: level[0].bounds,
                    large: None,
                });
            } else {
                for (i, ln) in level.iter().enumerate() {
                    if let HistFrom::Subtract { parent, sibling } = ln.hist_from {
                        if let Some(cache_slot) = session.cached_parent_slot(parent) {
                            groups.push(GpuSplitGroup {
                                small: sibling,
                                span: level[sibling].rows.device(),
                                sum: level[sibling].sum,
                                bounds: level[sibling].bounds,
                                large: Some((i, cache_slot, ln.sum, ln.bounds)),
                            });
                        }
                    }
                }
            }
            groups
        };
        #[cfg(feature = "cuda")]
        let node_enabled = |node: usize, feat: FeatId| {
            enum_features(node).is_none_or(|s| s.binary_search(&feat).is_ok())
        };

        // Two-phase GPU submission when every block is device resident: queue
        // all blocks' histogram + split kernels from this thread first, then
        // collect their packets. Per-block worker threads submitted (and
        // waited) one block after another as they woke up, leaving the device
        // idle between blocks; the remaining per-block host work is trivial, so
        // the level then runs sequentially. Results are the same packets.
        #[allow(unused_mut)] // filled only by the CUDA two-phase path
        let mut pre_split: Vec<Option<PreSplit>> = (0..n_blocks).map(|_| None).collect();
        #[cfg(feature = "cuda")]
        let two_phase = gpu.is_some_and(|ctx| {
            n_blocks > 1
                && plans
                    .iter()
                    .all(|pl| !pl.compact && pl.physical < ctx.resident_block_count())
        });
        #[cfg(not(feature = "cuda"))]
        let two_phase = false;
        #[cfg(feature = "cuda")]
        if two_phase {
            let ctx = gpu.expect("two-phase submission implies a GPU context");
            let started = std::time::Instant::now();
            let mut submitted = Vec::with_capacity(n_blocks);
            for (b, plan) in plans.iter().enumerate() {
                let selected_local = sampled.map(|_| plan.orig_local.as_slice());
                let Some(mut session) = ctx.block_session_resident_width(
                    plan.physical,
                    cuts.block_width(&plan.handed_feats),
                    source.n_rows(),
                    plan.handed_feats.len(),
                    &plan.hist_offsets,
                    selected_local,
                )?
                else {
                    continue; // not resident yet: the per-block path uploads it
                };
                if !session.begin_split_level(depth)? {
                    continue;
                }
                let groups = gpu_split_groups(&mut session);
                let pending = session.split_cached_submit(
                    &groups,
                    &plan.hist_feats,
                    sampled,
                    cuts.has_categorical().then(|| cuts.categorical_flags()),
                    nested
                        .is_some()
                        .then_some(&node_enabled as &dyn Fn(usize, FeatId) -> bool),
                    q,
                    p,
                    depth + 1 < p.max_depth,
                )?;
                submitted.push((b, session, pending));
            }
            for (b, mut session, pending) in submitted {
                pre_split[b] = Some(session.split_cached_finish(pending)?);
                session.finish_split_level(depth);
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
            }
        }

        let items: Vec<_> = parents.into_iter().zip(pre_split).enumerate().collect();
        #[allow(clippy::type_complexity)]
        let block_fn = |(b, (mut parent_hist, pre)): (
            usize,
            (Option<Vec<Histogram>>, Option<PreSplit>),
        )|
         -> anyhow::Result<(Vec<Histogram>, Vec<Option<SplitCandidate>>)> {
            #[cfg(feature = "cuda")]
            let mut pre = pre;
            #[cfg(not(feature = "cuda"))]
            let _ = pre;
            // `b` 只是 plan 数组里的位置,**不承载任何身份**。
            // 物理块号、原始局部下标、压实后的 offsets 全部从 plan 里取。
            // `b` 只是 plan 数组里的位置,**不承载任何身份**。
            // 物理块号、原始局部下标、offsets 全部从 plan 里显式取。
            let plan = &plans[b];
            let phys = plan.physical;
            let selected_local = match sampled {
                Some(_) if !plan.compact => Some(plan.orig_local.as_slice()),
                _ => None,
            };
            let mut run = |block: Option<&ColumnBlock>,
                           resident: Option<crate::backend::cuda::GpuBlockSession<'_>>|
             -> anyhow::Result<_> {
                if let Some(block) = block {
                    anyhow::ensure!(
                        block.n_rows == source.n_rows(),
                        "column block {phys} has {} rows, but the source declares {}",
                        block.n_rows,
                        source.n_rows()
                    );
                    anyhow::ensure!(
                        block.feat_ids == plan.handed_feats,
                        "column block {phys} returned features that do not match the plan"
                    );
                    anyhow::ensure!(
                        block.data.len() == block.n_rows.saturating_mul(block.n_feats()),
                        "column block {b} data length does not match rows × features"
                    );
                    anyhow::ensure!(
                        block.width() == source.cuts().block_width(&block.feat_ids),
                        "column block {phys} bin width does not match the cuts"
                    );
                }
                let width = *plan.hist_offsets.last().unwrap() as usize;
                // 要相减的槽位先放个空壳,缓冲在下面 subtract 那一趟才分配。
                // 原来这里给每个槽位都分配 + 清零一个完整直方图,相减节点
                // 那一半分配完立刻被扔掉。
                //
                // 老实说:实测**量不出差别**(这台机器上噪声比效果大)。
                // 留着是因为它确实少干一份活,不是因为它更快。
                let mut per_node: Vec<Histogram> = level
                    .iter()
                    .map(|ln| match ln.hist_from {
                        // CPU builds accumulate in place; GPU downloads
                        // supply complete values and allocate their outputs.
                        HistFrom::Accumulate if gpu.is_none() => Histogram::zeros(width),
                        _ => Histogram::zeros(0),
                    })
                    .collect();

                let mut gpu_done = vec![false; level.len()];
                let mut gpu_candidates = vec![None; level.len()];

                // 先算要实际累加的那些
                if let Some(ctx) = gpu {
                    // 量化列块 H2D **一次**,本块里所有活跃节点共用它。
                    // 早期版本是每个节点重新上传一遍整块,那是纯浪费。
                    let _t = prof::Timer::new(&prof::HIST);
                    let started = std::time::Instant::now();
                    // Physical identity and original-local feature identity
                    // come from this plan; neither is inferred from its
                    // colsample-filtered position. Partition lookup uses
                    // the same plan mapping.
                    let mut session = match resident {
                        Some(session) => session,
                        None => {
                            let block = block.context("GPU cache miss requires source payload")?;
                            ctx.block_session_bins(
                                phys,
                                block.bins(),
                                block.n_rows,
                                block.n_feats(),
                                &plan.hist_offsets,
                                selected_local,
                            )?
                        }
                    };
                    // 整层的 Accumulate 节点按容量分批合并 launch。
                    // 容量 1 就是原来的逐节点路径,一个字都没变。
                    let cap = session.hist_nodes_batch_capacity();
                    // 行主序 kernel 只有 batched 一个入口,批里只有一个
                    // 节点也必须走它。
                    let must_batch = session.requires_batched_hist();
                    #[cfg(feature = "cuda")]
                    if let Some(results) = pre.take() {
                        // Already submitted and collected by the two-phase path.
                        for (i, candidate, histogram) in results {
                            gpu_done[i] = true;
                            gpu_candidates[i] = candidate;
                            if let Some(bins) = histogram {
                                per_node[i].bins = bins;
                            }
                        }
                    } else if session.begin_split_level(depth)? {
                        let groups = gpu_split_groups(&mut session);
                        for (i, candidate, histogram) in session.split_cached_batch(
                            &groups,
                            &plan.hist_feats,
                            sampled,
                            cuts.has_categorical().then(|| cuts.categorical_flags()),
                            // bylevel / bynode: the level node's feature set is
                            // the host's `enum_features`, uploaded as a byte mask.
                            nested
                                .is_some()
                                .then_some(&node_enabled as &dyn Fn(usize, FeatId) -> bool),
                            q,
                            p,
                            depth + 1 < p.max_depth,
                        )? {
                            gpu_done[i] = true;
                            gpu_candidates[i] = candidate;
                            if let Some(bins) = histogram {
                                per_node[i].bins = bins;
                            }
                        }
                        session.finish_split_level(depth);
                    }
                    let todo: Vec<usize> = level
                        .iter()
                        .enumerate()
                        .filter(|(i, ln)| {
                            !gpu_done[*i] && matches!(ln.hist_from, HistFrom::Accumulate)
                        })
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
                            session.histogram_device_batch(&spans, &mut outs)?;
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
                    let block = block.context("CPU histogram requires source payload")?;
                    let _t = prof::Timer::new(&prof::HIST);
                    // `hist_feats` 的每一列在交回块里的位置:整块交回时是
                    // 采样列的原始局部下标,压实块或不采样时就是全部列。
                    // Every column of a block has the same width: dispatch once
                    // per block so the inner loops are monomorphised.
                    crate::with_bins!(block.bins(), |data| {
                        let all_cols = hist::block_columns(data, block.n_rows, block.n_feats());
                        let cols: Vec<&[_]> = match selected_local {
                            Some(sel) => sel.iter().map(|&l| all_cols[l]).collect(),
                            None => all_cols,
                        };
                        debug_assert_eq!(cols.len(), plan.hist_feats.len());
                        if depth == 0 && !drop_zero_rows {
                            // 根节点走 None 那条路:少一层行下标的间接寻址,
                            // 结果和传全部行完全一样(hist.rs 有断言钉着)。
                            hist::build_cols(
                                &cols,
                                &plan.hist_offsets,
                                gpair,
                                None,
                                &mut per_node[0].bins,
                            );
                        } else {
                            // 本层所有要实际累加的节点一起走:行 id 区间在外、
                            // 节点在内,同一段 bin/gpair 被所有节点复用
                            // (见 `hist::build_nodes`)。逐位结果与逐节点
                            // 调 `build` 相同。
                            let mut targets: Vec<(&[RowId], &mut [GradPairFixed])> = level
                                .iter()
                                .zip(per_node.iter_mut())
                                .filter(|(ln, _)| matches!(ln.hist_from, HistFrom::Accumulate))
                                .map(|(ln, h)| (ln.rows.host(&rows_cur), h.bins.as_mut_slice()))
                                .collect();
                            hist::build_nodes_cols(&cols, &plan.hist_offsets, gpair, &mut targets);
                        }
                    });
                }
                // 再用父 - 兄弟补上剩下的。分两趟是为了不依赖兄弟在本层
                // 里的先后顺序。
                for i in 0..level.len() {
                    if gpu_done[i] {
                        continue;
                    }
                    if let HistFrom::Subtract { parent, sibling } = level[i].hist_from {
                        let ph = parent_hist.as_mut().expect(
                            "the first level has no parent and must not take the subtraction path",
                        );
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
                                "parent histogram {parent} was claimed more than once (or has the wrong width)"
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
                        if gpu_done[i] {
                            return gpu_candidates[i].clone();
                        }
                        split::best_split_in_block_typed(
                            &per_node[i],
                            &plan.hist_feats,
                            &plan.hist_offsets,
                            ln.sum,
                            q,
                            p,
                            enum_features(i),
                            monotone_at(p, ln.bounds),
                            cuts.has_categorical().then(|| cuts.categorical_flags()),
                        )
                    })
                    .collect();
                Ok((per_node, cands))
            };
            // Check the dedicated resident slot before asking the source to
            // materialize bins. Its payload passed source/shape validation
            // on first upload; current tree metadata still goes through the
            // same GPU session validation. Physical identity is never a
            // filtered plan position. Streaming and cold misses keep the
            // source read/checksum path.
            if !plan.compact {
                if let Some(ctx) = gpu {
                    if let Some(session) = ctx.block_session_resident_width(
                        phys,
                        cuts.block_width(&plan.handed_feats),
                        source.n_rows(),
                        plan.handed_feats.len(),
                        &plan.hist_offsets,
                        selected_local,
                    )? {
                        return run(None, Some(session));
                    }
                }
            }
            if plan.compact {
                source
                    .with_block_cols_ret(phys, &plan.orig_local, |block| run(Some(block), None))?
            } else {
                source.with_block_ret(phys, |block| run(Some(block), None))?
            }
            // 本块的父直方图到此为止 —— 闭包一结束,它和刚借来的
            // 块一起释放,这一层内存最紧的时候正好都不在手上。
        };
        // Paired histogram and split results retain per-block ownership.
        #[allow(clippy::type_complexity)]
        let per_block: anyhow::Result<Vec<(Vec<Histogram>, Vec<Option<SplitCandidate>>)>> =
            if two_phase {
                items.into_iter().map(&block_fn).collect()
            } else {
                items.into_par_iter().map(&block_fn).collect()
            };
        let per_block = per_block?;
        drop(hist_phase);
        drop(_phase);

        // 跨块归约按**块序**做,和单线程逐块推进完全一样。
        //
        // 线程数不影响结果,靠的是两件事叠起来:块内是定点整数累加
        // (与顺序无关),块间的归约顺序在这里固定死。少了任何一件,
        // 并列的候选就会因为"哪个线程先跑完"而胜负不同 ——
        // tests/bit_exact_vs_xgboost.rs 盯着这条。
        let split_reduce_phase = crate::diagnostic::Phase::new(b"fb.host.split_reduce\0");
        let mut best: Vec<Option<SplitCandidate>> = vec![None; level.len()];
        let mut hists: Vec<Vec<Histogram>> = Vec::with_capacity(n_blocks);
        for (per_node, cands) in per_block {
            for (i, cand) in cands.into_iter().enumerate() {
                best[i] = split::reduce_best(&[best[i].take(), cand]);
            }
            hists.push(per_node);
        }

        // 多机/多卡时各方只看得到自己那几列,全局最优要在这里汇总。
        // 单机 Local 是恒等,但训练逻辑里不能直接挑本地最优 ——
        // 那样换个 Comm 就不对了。
        let mut chosen: Vec<Option<SplitCandidate>> = best
            .into_iter()
            .map(|c| comm.allreduce_best_split(c))
            .collect();
        drop(split_reduce_phase);
        let partition_phase = crate::diagnostic::Phase::new(b"fb.host.partition\0");

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
        // Group resident requests by PHYSICAL block, not selected feature.
        // Original-local feature IDs survive colsample/compaction unchanged.
        // Keep the existing per-feature upload fallback for nonresident blocks.
        if let Some(ctx) = gpu {
            let mut resident_groups: std::collections::BTreeMap<usize, Vec<(usize, usize)>> =
                Default::default();
            for (&feat, node_idxs) in &by_feat {
                let loc = plans.iter().find_map(|pl| {
                    pl.sampled_global
                        .iter()
                        .position(|&f| f == feat)
                        .map(|i| (pl.physical, pl.orig_local[i]))
                });
                if let Some((block, local)) =
                    loc.filter(|&(block, _)| block < ctx.resident_block_count())
                {
                    resident_groups
                        .entry(block)
                        .or_default()
                        .extend(node_idxs.iter().map(|&i| (i, local)));
                }
            }
            // Categorical nodes take the single-node route of their block's
            // session (the batched kernel only knows ordered bins). Numeric
            // nodes of every resident block go into level-wide batches: one
            // descriptor upload and one count/scan/scatter triple per
            // PARTITION_BATCH_CAPACITY nodes instead of per block.
            let mut numeric: Vec<(usize, usize, usize)> = Vec::new();
            for (block, nodes) in resident_groups {
                let (cat_nodes, nodes): (Vec<_>, Vec<_>) = nodes
                    .into_iter()
                    .partition(|&(i, _)| chosen[i].as_ref().is_some_and(|c| c.cats.is_some()));
                numeric.extend(nodes.into_iter().map(|(i, local)| (i, block, local)));
                if cat_nodes.is_empty() {
                    continue;
                }
                if let Some(mut session) = ctx.partition_session_resident(block, 0)? {
                    let started = std::time::Instant::now();
                    for (i, local) in cat_nodes {
                        let c = chosen[i]
                            .as_ref()
                            .expect("resident group contains chosen nodes");
                        let cats = c.cats.as_ref().expect("partitioned by categorical");
                        part_tokens[i] = Some(session.partition_resident_categorical(
                            level[i].rows.device(),
                            local,
                            cats,
                            c.missing_left,
                        )?);
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
                }
            }
            let mut blocks: Vec<usize> = numeric.iter().map(|&(_, b, _)| b).collect();
            blocks.dedup(); // `numeric` is grouped by ascending block
                            // A batch reads blocks of one bin width (8-bit blocks may be
                            // row-major on the device, 16-bit ones are column-major): 8-bit
                            // blocks first, exactly as before, then the 16-bit blocks.
            let wide_block = |block: usize| {
                plans.iter().any(|pl| {
                    pl.physical == block
                        && cuts.block_width(&pl.handed_feats) == crate::bins::BinWidth::U16
                })
            };
            let (wide_blocks, narrow_blocks): (Vec<usize>, Vec<usize>) =
                blocks.into_iter().partition(|&b| wide_block(b));
            let block_chunks = narrow_blocks
                .chunks(crate::gpu_mem_plan::PARTITION_MAX_BLOCKS)
                .chain(wide_blocks.chunks(crate::gpu_mem_plan::PARTITION_MAX_BLOCKS));
            for block_chunk in block_chunks {
                let Some(mut session) = ctx.partition_session_resident_blocks(block_chunk)? else {
                    continue; // nonresident: the per-feature fallback below
                };
                let started = std::time::Instant::now();
                let nodes: Vec<_> = numeric
                    .iter()
                    .filter(|(_, b, _)| block_chunk.binary_search(b).is_ok())
                    .collect();
                for chunk in nodes.chunks(crate::gpu_mem_plan::PARTITION_BATCH_CAPACITY) {
                    let requests: Vec<_> = chunk
                        .iter()
                        .map(|&&(i, block, local)| {
                            let c = chosen[i]
                                .as_ref()
                                .expect("resident group contains chosen nodes");
                            (level[i].rows.device(), block, local, c.bin, c.missing_left)
                        })
                        .collect();
                    let tokens = session.partition_resident_multi(&requests)?;
                    for (&&(i, _, _), token) in chunk.iter().zip(tokens) {
                        part_tokens[i] = Some(token);
                    }
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
            }
        }
        if gpu.is_none() {
            // CPU:整层一起分(见 `partition_level_host`)。节点按层内顺序,
            // 也就是按行缓冲里的位置升序。
            let nodes: Vec<(HostSpan, SplitCandidate)> = level
                .iter()
                .zip(&chosen)
                .filter_map(|(ln, c)| match (&ln.rows, c) {
                    (NodeRows::Host(span), Some(c)) => Some((*span, c.clone())),
                    _ => None,
                })
                .collect();
            if !nodes.is_empty() {
                if rows_next.len() != rows_cur.len() {
                    rows_next = vec![0 as RowId; rows_cur.len()];
                }
                let spans = partition_level_host(
                    &rows_cur,
                    &mut rows_next,
                    &nodes,
                    PartCols::for_nodes(source, &nodes),
                    comm,
                )?;
                let mut spans = spans.into_iter();
                for (i, c) in chosen.iter().enumerate() {
                    if c.is_some() {
                        let (left, right) = spans
                            .next()
                            .expect("each split node has a pair of children");
                        partitions[i] = Some((NodeRows::Host(left), NodeRows::Host(right)));
                    }
                }
            }
            by_feat.clear();
        }
        for (feat, node_idxs) in by_feat {
            let node_idxs: Vec<_> = node_idxs
                .into_iter()
                .filter(|&i| part_tokens[i].is_none())
                .collect();
            if node_idxs.is_empty() {
                continue;
            }
            // 只读借用这一列:驻留 source 零拷贝;落盘 cache 等只能交出拷贝的
            // source 照旧付那份拷贝(PART_FETCH 只计这部分)。
            let fetch_timer = prof::Timer::new(&prof::PART_FETCH);
            let owned_col;
            let col: crate::bins::BinSlice<'_> = match source.resident_feature_column(feat) {
                Some(col) => col,
                None => {
                    owned_col = source.feature_column(feat)?;
                    owned_col.as_slice()
                }
            };
            drop(fetch_timer);
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
            {
                let ctx = gpu.expect("CPU partitioning was already done for the whole level above");
                // 选中的这一列 H2D **一次**,本层所有选中它的节点共用 ——
                // 和「外层特征、内层节点」的循环顺序天然对齐。
                let started = std::time::Instant::now();
                let mut session = ctx.partition_session_bins(col)?;
                for i in node_idxs {
                    let c = chosen[i]
                        .as_ref()
                        .expect("by_feat only holds nodes with a candidate");
                    part_tokens[i] = Some(match &c.cats {
                        Some(cats) => session.partition_categorical(
                            level[i].rows.device(),
                            cats,
                            c.missing_left,
                        )?,
                        None => session.partition(level[i].rows.device(), c.bin, c.missing_left)?,
                    });
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

        drop(partition_phase);
        let _next_phase = crate::diagnostic::Phase::new(b"fb.host.next_level\0");
        let _t = prof::Timer::new(&prof::NEXT_LEVEL);
        let mut next: Vec<LevelNode> = Vec::new();
        // GPU 终叶整层攒起来,一次 launch 写完(见 `assign_leaf_values`)。
        let mut gpu_leaves: Vec<(crate::backend::cuda::GpuRowSpan, f32)> = Vec::new();
        for (i, ln) in level.into_iter().enumerate() {
            let Some(c) = chosen[i].take() else {
                // 这个节点到此就是终叶。行分区已经是答案,现在直接累加
                // 比训练结束后再沿树下降一遍便宜得多。
                let value = t.nodes[ln.node].leaf_value;
                if gpu.is_some() {
                    gpu_leaves.push((ln.rows.device(), value));
                } else {
                    let _t_pred = prof::Timer::new(&prof::UPDATE_PRED);
                    for &row in ln.rows.host(&rows_cur) {
                        predictions[row as usize] += value;
                    }
                }
                continue; // 分不动就留成叶子,值在 push_leaf 时就写好了
            };

            let children = child_leaves(&c, q, p, ln.bounds);
            let ln_node = ln.node;
            let (left_node, right_node) = t.split_leaf(
                ln.node,
                c.feat,
                if c.cats.is_some() {
                    0.0
                } else {
                    tree::split_threshold(cuts, c.feat, c.bin)
                },
                c.missing_left,
                c.gain,
                children.left_value,
                hess_of(c.left, q),
                children.right_value,
                hess_of(c.right, q),
            );
            if let Some(cats) = &c.cats {
                t.set_categories(ln.node, cats);
            }

            if drop_zero_rows {
                if split_info.len() <= ln_node {
                    split_info.resize(ln_node + 1, DropSplit::default());
                }
                split_info[ln_node] = DropSplit {
                    feat: c.feat,
                    bin: c.bin,
                    missing_left: c.missing_left,
                    cats: c.cats,
                };
            }
            let (left_rows, right_rows) = partitions[i].take().expect("partitioned by block above");

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
                    bounds: children.left_bounds,
                },
                LevelNode {
                    node: right_node,
                    rows: right_rows,
                    sum: c.right,
                    hist_from: HistFrom::Accumulate,
                    bounds: children.right_bounds,
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
            // 这些行还在本层的行缓冲里;下一层的分区在同一条 stream 上排在后面。
            ctx.assign_leaf_values(&gpu_leaves)?;
            ctx.finish_partition_level()?;
        }
        // 孩子的行已经在 `rows_next` 里;本层不分裂的叶子刚才已从 `rows_cur`
        // 施加完,旧缓冲从此只当下一层的输出用。
        if gpu.is_none() {
            std::mem::swap(&mut rows_cur, &mut rows_next);
        }
        level = next;
    }

    // max_depth 用尽时,最后一次分裂产生的孩子还留在 level 里;
    // max_depth == 0 时这里处理根叶子。其它提前停下的叶子已在上面写过。
    {
        let _t = prof::Timer::new(&prof::UPDATE_PRED);
        if let Some(ctx) = gpu {
            let leaves: Vec<_> = level
                .iter()
                .map(|ln| {
                    debug_assert!(t.nodes[ln.node].is_leaf);
                    (ln.rows.device(), t.nodes[ln.node].leaf_value)
                })
                .collect();
            ctx.assign_leaf_values(&leaves)?;
        } else {
            let leaves: Vec<(&[RowId], f32)> = level
                .iter()
                .map(|ln| {
                    debug_assert!(t.nodes[ln.node].is_leaf);
                    (ln.rows.host(&rows_cur), t.nodes[ln.node].leaf_value)
                })
                .collect();
            apply_host_leaf_updates(predictions, &leaves);
            drop(leaves);
            // Keep row-buffer release in the same profiled phase as before.
            drop(level);
        }
        // `gpu_math = "fast"` 下 delta 由 device 上的 gradient kernel 直接
        // 消费,host 永远不需要看它 —— 这一次 4 B/行 的 D2H 整个消失。
        // 漏掉这个判断的话 fast 模式仍然每棵树付一次 40 MB D2H
        // (实测 16.5 ms/轮),收益被吃掉一半还多。
        // exact:host 是权威副本,delta 要交回去施加。
        // fast:device 是权威副本,delta 由 device 的 gradient kernel 直接
        // 消费,这一次 4 B/行 的 D2H 整个不存在。
        if let Some(ctx) = gpu.filter(|c| c.prediction_authority() == PredictionAuthority::Host) {
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
                    std::mem::size_of_val(predictions) as u64,
                    std::sync::atomic::Ordering::Relaxed,
                );
                // One f32 delta per training row is downloaded once per tree.
                // Applying it to `predictions` is intentionally kept on CPU
                // for now and remains visible in UPDATE_PRED / round wall.
            }
        }
    }

    if drop_zero_rows {
        apply_dropped_rows(
            &t,
            &split_info,
            source,
            gpair,
            host_nonzero.as_ref(),
            predictions,
        )?;
    }

    Ok(t)
}

/// The bin geometry GPU training allocates for: block widths in features and
/// bytes per row, the largest block histogram, the widest 8-bit block (the
/// device row-major gate) and the widest category set a partition ships.
/// Pure metadata: no block payload is read.
pub(crate) fn gpu_block_shape(
    source: &dyn BlockSource,
) -> anyhow::Result<crate::gpu_mem_plan::GpuBlockShape> {
    use crate::bins::BinWidth;
    let cuts = source.cuts();
    let mut shape = crate::gpu_mem_plan::GpuBlockShape::all_u8(0, 0);
    for b in 0..source.n_blocks() {
        let feats = source.block_features(b)?;
        let width = cuts.block_width(&feats);
        shape.max_block_feats = shape.max_block_feats.max(feats.len());
        shape.max_block_row_bytes = shape.max_block_row_bytes.max(feats.len() * width.bytes());
        let offsets = cuts.block_hist_offsets(&feats);
        shape.max_hist_bins = shape.max_hist_bins.max(*offsets.last().unwrap() as usize);
        match width {
            BinWidth::U8 => {
                shape.max_u8_block_feats = shape.max_u8_block_feats.max(feats.len());
            }
            BinWidth::U16 => {
                shape.max_cell_bytes = 2;
                let (mut cat_feats, mut cat_bins, mut set_words) = (0, 0, 0);
                for &f in &feats {
                    if cuts.is_categorical(f) {
                        // Two 32-bit words per 64-bit set word (CatBits::words64).
                        let words = cuts.n_bins(f).div_ceil(64) * 2;
                        shape.cat_mask_words = shape.cat_mask_words.max(words);
                        cat_feats += 1;
                        cat_bins += cuts.n_bins(f);
                        set_words += words;
                    }
                }
                // Device split buffers of the block's categorical features.
                shape.wide_cat_feats = shape.wide_cat_feats.max(cat_feats);
                shape.wide_cat_bins = shape.wide_cat_bins.max(cat_bins);
                shape.wide_cat_set_words = shape.wide_cat_set_words.max(set_words);
            }
        }
    }
    Ok(shape)
}

/// `apply_dropped_rows` 需要的、和分区完全同一套的分裂判定。
#[derive(Clone, Default)]
struct DropSplit {
    feat: FeatId,
    bin: u32,
    missing_left: bool,
    cats: Option<crate::categorical::CatBits>,
}

/// 沿已建好的树给**没进行集合**的行(梯度全零的行)施加叶子值。
///
/// 每行一次随机访问下降,最多 `max_depth` 步;分裂用的比较和分区那一遍
/// 完全相同(`bin < 边界`,缺失走记下的方向),所以每行落到的叶子、
/// 以及对 `predictions` 的那一次加法都和「随每层分区一起搬」逐位相同。
/// 按行块并行,每块独占自己那段 `predictions`。
///
/// 下降是一条依赖链(节点 → 列 → bin → 方向 → 下一节点),方向又是
/// 一半一半的随机分支。所以一次推进 `DROP_LANES` 行、每行走固定的
/// 「树深」步:叶子指回自己,每一步对所有行都无分支,各行的链彼此独立,
/// 访存延迟可以重叠。
///
/// `nonzero` 是转定点时记下的非零位图(有就用,省得每行读 16 B 的定点
/// 梯度);没有时按 `gpair` 判断。
fn apply_dropped_rows(
    t: &Tree,
    split_info: &[DropSplit],
    source: &dyn BlockSource,
    gpair: &[GradPairFixed],
    nonzero: Option<&NonzeroBits>,
    predictions: &mut [f32],
) -> anyhow::Result<()> {
    anyhow::ensure!(gpair.len() == predictions.len());
    if t.nodes.is_empty() {
        return Ok(());
    }
    // 扁平下降表。叶子:两个孩子都是自己,列借根节点的(读到什么都不影响)。
    struct Step<'a> {
        col: crate::bins::BinSlice<'a>,
        /// 这一列宽度的缺失哨兵。
        missing: u32,
        bound: u32,
        missing_left: bool,
        /// 类别分裂走右的集合(64 位字)。
        cats: Option<&'a [u64]>,
        next: [u32; 2],
        value: f32,
    }
    let root_col = if t.nodes[0].is_leaf {
        crate::bins::BinSlice::U8(&[])
    } else {
        source
            .resident_feature_column(t.nodes[0].feat)
            .context("zero-gradient row descent needs resident columns")?
    };
    let mut steps: Vec<Step<'_>> = Vec::with_capacity(t.nodes.len());
    for (i, n) in t.nodes.iter().enumerate() {
        if n.is_leaf {
            steps.push(Step {
                col: root_col,
                missing: root_col.width().missing(),
                bound: 0,
                missing_left: true,
                cats: None,
                next: [i as u32; 2],
                value: n.leaf_value,
            });
        } else {
            let sp = &split_info[i];
            let col = source
                .resident_feature_column(sp.feat)
                .context("zero-gradient row descent needs resident columns")?;
            steps.push(Step {
                col,
                missing: col.width().missing(),
                bound: sp.bin,
                missing_left: sp.missing_left,
                cats: sp.cats.as_ref().map(|c| c.words64()),
                next: [n.left as u32, n.right as u32],
                value: 0.0,
            });
        }
    }
    fn depth(t: &Tree, i: usize) -> usize {
        let n = &t.nodes[i];
        if n.is_leaf {
            0
        } else {
            1 + depth(t, n.left as usize).max(depth(t, n.right as usize))
        }
    }
    let depth = depth(t, 0);
    const DROP_LANES: usize = 8;
    // 64 的倍数:块与非零位图的字对齐。
    const CHUNK: usize = 1 << 15;
    let steps = &steps;
    predictions
        .par_chunks_mut(CHUNK)
        .zip(gpair.par_chunks(CHUNK))
        .enumerate()
        .for_each(|(c, (pred, g))| {
            let base = c * CHUNK;
            // 本块里被丢掉的行(块内偏移,升序)。
            let mut dropped: Vec<u32> = Vec::with_capacity(pred.len());
            match nonzero {
                Some(bits) => {
                    let words = &bits.words[base / 64..(base + pred.len()).div_ceil(64)];
                    for (wi, &w) in words.iter().enumerate() {
                        let valid = (pred.len() - wi * 64).min(64);
                        let mut z = !w;
                        if valid < 64 {
                            z &= (1u64 << valid) - 1;
                        }
                        while z != 0 {
                            dropped.push((wi * 64) as u32 + z.trailing_zeros());
                            z &= z - 1;
                        }
                    }
                }
                None => {
                    let zero = GradPairFixed::default();
                    dropped.extend(
                        g.iter()
                            .enumerate()
                            .filter(|(_, gp)| **gp == zero)
                            .map(|(i, _)| i as u32),
                    );
                }
            }
            for lanes in dropped.chunks(DROP_LANES) {
                let mut node = [0u32; DROP_LANES];
                for _ in 0..depth {
                    for (k, &i) in lanes.iter().enumerate() {
                        let st = &steps[node[k] as usize];
                        // 树深 ≥ 1 时根是分裂节点,叶子借的根列与其它列
                        // 同长,读取不会越界(树深 0 时根本不进这个循环)。
                        let bin = st.col.get(base + i as usize);
                        // 与 `fill_partition_bits` 逐条相同:缺失走记下的方向;
                        // 类别分裂里集合走右、其余(含未见过的编码)走左;
                        // 数值分裂比 bin。
                        let go_left = if bin == st.missing {
                            st.missing_left
                        } else if let Some(cats) = st.cats {
                            !crate::categorical::contains_in(cats, bin)
                        } else {
                            bin < st.bound
                        };
                        node[k] = st.next[(!go_left) as usize];
                    }
                }
                for (k, &i) in lanes.iter().enumerate() {
                    pred[i as usize] += steps[node[k] as usize].value;
                }
            }
        });
    Ok(())
}

/// host 量化那几趟逐行工作的分块大小(行)。是 64 的倍数,每块的非零位图
/// 恰好占整数个 `u64`。
const PREP_CHUNK: usize = 1 << 16;

/// 行采样掩码和量化上界合成一趟:未选中的行置零(工作副本,见调用处),
/// 同时对置零之后的值求 `GradQuantizer::try_new` 要的 `(max |grad|,
/// max |hess|)`。返回 `(max_grad, max_hess, 选中行数)`。
///
/// 采样身份只由 `(seed, tree, row)` 逐行独立哈希决定,max 精确、行数是整数
/// 求和,三者都与分块和线程数无关;上界与「先置零、再 `try_new`」逐位相同。
fn mask_and_bounds(gpair: &mut [GradPair], mask: Option<(u64, u64, u64)>) -> (f64, f64, usize) {
    gpair
        .par_chunks_mut(PREP_CHUNK)
        .enumerate()
        .map(|(c, chunk)| {
            let base = (c * PREP_CHUNK) as u64;
            let (mut mg, mut mh, mut kept) = (0.0f64, 0.0f64, 0usize);
            for (i, g) in chunk.iter_mut().enumerate() {
                if let Some((th, seed, tree)) = mask {
                    if !crate::subsample::row_selected(base + i as u64, th, seed, tree) {
                        *g = GradPair {
                            grad: 0.0,
                            hess: 0.0,
                        };
                    } else {
                        kept += 1;
                    }
                } else {
                    kept += 1;
                }
                let (gb, hb) = GradQuantizer::bound_of(*g);
                mg = mg.max(gb);
                mh = mh.max(hb);
            }
            (mg, mh, kept)
        })
        .reduce(
            || (0.0, 0.0, 0),
            |a, b| (a.0.max(b.0), a.1.max(b.1), a.2 + b.2),
        )
}

/// 定点梯度不为零的行,一行一位(`PREP_CHUNK` 行一块),外加每块的个数。
struct NonzeroBits {
    words: Vec<u64>,
    chunk_counts: Vec<usize>,
}

impl NonzeroBits {
    fn count(&self) -> usize {
        self.chunk_counts.iter().sum()
    }

    /// 位为 1 的行下标(升序),与 `nonzero_rows` 逐位相同。各块写自己那段。
    fn rows(&self) -> Vec<RowId> {
        let mut rows = vec![0 as RowId; self.count()];
        let mut parts: Vec<&mut [RowId]> = Vec::with_capacity(self.chunk_counts.len());
        let mut rest = rows.as_mut_slice();
        for &c in &self.chunk_counts {
            let (head, tail) = rest.split_at_mut(c);
            parts.push(head);
            rest = tail;
        }
        parts
            .into_par_iter()
            .zip(self.words.par_chunks(PREP_CHUNK / 64))
            .enumerate()
            .for_each(|(ci, (out, words))| {
                let base = ci * PREP_CHUNK;
                let mut k = 0usize;
                for (wi, &w) in words.iter().enumerate() {
                    let mut w = w;
                    while w != 0 {
                        out[k] = (base + wi * 64 + w.trailing_zeros() as usize) as RowId;
                        k += 1;
                        w &= w - 1;
                    }
                }
                debug_assert_eq!(k, out.len());
            });
        rows
    }
}

/// 转定点的同一趟里顺手求根节点的和与非零位图:以前这是三趟
/// (转定点、`node_sum`、`nonzero_rows` 的计数和写入)各读一遍 16 B/行的
/// 定点数组。`out` 跨轮复用,长度不符时才重新分配。
///
/// 定点加法按模 2^64 精确、与顺序无关,所以分块求和与 `hist::node_sum`
/// 逐位相同;每行的定点值与 `to_fixed` 逐行 map 相同。
fn quantize_host(
    gpair: &[GradPair],
    q: &GradQuantizer,
    out: &mut Vec<GradPairFixed>,
) -> (GradPairFixed, NonzeroBits) {
    let n = gpair.len();
    if out.len() != n {
        *out = vec![GradPairFixed::default(); n];
    }
    let mut words = vec![0u64; n.div_ceil(64)];
    let zero = GradPairFixed::default();
    let parts: Vec<(GradPairFixed, usize)> = out
        .par_chunks_mut(PREP_CHUNK)
        .zip(gpair.par_chunks(PREP_CHUNK))
        .zip(words.par_chunks_mut(PREP_CHUNK / 64))
        .map(|((dst, src), bits)| {
            let mut sum = GradPairFixed::default();
            let mut count = 0usize;
            for ((d, s), w) in dst.chunks_mut(64).zip(src.chunks(64)).zip(bits.iter_mut()) {
                let mut word = 0u64;
                for (j, (d, &s)) in d.iter_mut().zip(s).enumerate() {
                    let f = q.to_fixed(s);
                    *d = f;
                    sum += f;
                    word |= ((f != zero) as u64) << j;
                }
                *w = word;
                count += word.count_ones() as usize;
            }
            (sum, count)
        })
        .collect();
    let mut sum = GradPairFixed::default();
    for &(s, _) in &parts {
        sum += s;
    }
    let chunk_counts = parts.into_iter().map(|(_, c)| c).collect();
    (
        sum,
        NonzeroBits {
            words,
            chunk_counts,
        },
    )
}

/// 梯度不全为零的行下标(升序)。按块并行:先数、再前缀和、最后各块写
/// 自己那一段,结果与串行 `filter` 逐位相同。
fn nonzero_rows(gpair: &[GradPairFixed]) -> Vec<RowId> {
    const CHUNK: usize = 1 << 16;
    let zero = GradPairFixed::default();
    let counts: Vec<usize> = gpair
        .par_chunks(CHUNK)
        .map(|c| c.iter().filter(|g| **g != zero).count())
        .collect();
    let mut offsets = Vec::with_capacity(counts.len() + 1);
    offsets.push(0usize);
    for &c in &counts {
        offsets.push(offsets.last().unwrap() + c);
    }
    let mut rows = vec![0 as RowId; *offsets.last().unwrap()];
    let mut parts: Vec<&mut [RowId]> = Vec::with_capacity(counts.len());
    let mut rest = rows.as_mut_slice();
    for &c in &counts {
        let (head, tail) = rest.split_at_mut(c);
        parts.push(head);
        rest = tail;
    }
    parts
        .into_par_iter()
        .zip(gpair.par_chunks(CHUNK))
        .enumerate()
        .for_each(|(ci, (out, g))| {
            let base = ci * CHUNK;
            let mut k = 0usize;
            for (i, gp) in g.iter().enumerate() {
                if *gp != zero {
                    out[k] = (base + i) as RowId;
                    k += 1;
                }
            }
        });
    rows
}

/// `0..n` 的行下标。每棵树一次,1050 万行是 42 MB:串行 `collect()` 把
/// 写入和首次触碰的缺页都压在一条线程上,所以按块并行填(与 XGBoost 的
/// 并行 Iota 同形)。内容逐位相同。
fn identity_rows(n: usize) -> Vec<RowId> {
    const CHUNK: usize = 1 << 16;
    let mut rows = vec![0 as RowId; n];
    if n < 4 * CHUNK {
        for (i, r) in rows.iter_mut().enumerate() {
            *r = i as RowId;
        }
        return rows;
    }
    rows.par_chunks_mut(CHUNK)
        .enumerate()
        .for_each(|(c, chunk)| {
            let base = c * CHUNK;
            for (i, r) in chunk.iter_mut().enumerate() {
                *r = (base + i) as RowId;
            }
        });
    rows
}

/// 约束模式下 split 枚举要的上下文;没有非 0 约束时是 `None`(无约束路径)。
fn monotone_at(p: &TrainParams, bounds: split::NodeBounds) -> Option<split::Monotone<'_>> {
    p.uses_given_weight().then_some(split::Monotone {
        constraints: &p.monotone_constraints,
        bounds,
    })
}

/// 一次分裂落地后两个孩子的叶子值和权重界。
struct ChildLeaves {
    left_value: f32,
    right_value: f32,
    left_bounds: split::NodeBounds,
    right_bounds: split::NodeBounds,
}

/// Apply terminal leaves to host predictions. Stable partitioning keeps each
/// leaf's row IDs sorted. Split the destination into disjoint contiguous slices
/// so workers need neither unsafe scatter writes nor an O(rows) delta buffer.
/// Each destination retains the original leaf/row addition order, even for
/// repeated row IDs; only independent destination ranges execute concurrently.
fn apply_host_leaf_updates(predictions: &mut [f32], leaves: &[(&[RowId], f32)]) {
    const CHUNK_ROWS: usize = 1 << 16;
    const MIN_PARALLEL_ROWS: usize = 2 * CHUNK_ROWS;
    assert!(
        leaves.iter().all(|(rows, _)| rows
            .last()
            .is_none_or(|&row| (row as usize) < predictions.len())),
        "leaf row index exceeds prediction buffer"
    );
    debug_assert!(leaves
        .iter()
        .all(|(rows, _)| rows.windows(2).all(|pair| pair[0] <= pair[1])));
    let active_rows: usize = leaves.iter().map(|(rows, _)| rows.len()).sum();
    // Avoid a chunk-by-leaf search explosion for deep trees or sparse tails.
    // Both bounds are work estimates; the old serial path remains available.
    let leaf_searches = leaves
        .len()
        .saturating_mul(predictions.len().div_ceil(CHUNK_ROWS));
    if active_rows < MIN_PARALLEL_ROWS
        || leaf_searches > active_rows / 16
        || rayon::current_num_threads() == 1
    {
        for &(rows, value) in leaves {
            for &row in rows {
                predictions[row as usize] += value;
            }
        }
        return;
    }
    predictions
        .par_chunks_mut(CHUNK_ROWS)
        .enumerate()
        .for_each(|(chunk, output)| {
            let start = chunk * CHUNK_ROWS;
            let end = start + output.len();
            for &(rows, value) in leaves {
                let first = rows.partition_point(|&row| (row as usize) < start);
                let count = rows[first..].partition_point(|&row| (row as usize) < end);
                for &row in &rows[first..first + count] {
                    output[row as usize - start] += value;
                }
            }
        });
}

/// 无约束:叶子值就是 `leaf_of`,界保持 `UNBOUNDED`(一位不动)。
/// 有约束:孩子权重先夹进**父节点**的界(XGBoost `ApplyTreeSplit` 用
/// `candidate.nid` 的界),叶子值 = 夹过的权重 × learning_rate;孩子的界按
/// 分裂特征的约束方向从 `(wl + wr) / 2` 处切开。
fn child_leaves(
    c: &SplitCandidate,
    q: &GradQuantizer,
    p: &TrainParams,
    parent: split::NodeBounds,
) -> ChildLeaves {
    if !p.uses_given_weight() {
        return ChildLeaves {
            left_value: leaf_of(c.left, q, p),
            right_value: leaf_of(c.right, q, p),
            left_bounds: split::NodeBounds::UNBOUNDED,
            right_bounds: split::NodeBounds::UNBOUNDED,
        };
    }
    let wl = split::bounded_weight(c.left, q, p, parent);
    let wr = split::bounded_weight(c.right, q, p, parent);
    let (left_bounds, right_bounds) = parent.children(p.monotone_of(c.feat), wl, wr);
    ChildLeaves {
        left_value: wl * p.learning_rate,
        right_value: wr * p.learning_rate,
        left_bounds,
        right_bounds,
    }
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

/// 一个节点在宿主行缓冲里占的连续一段。
///
/// CPU 路径上一层所有节点的行下标住在**同一个**缓冲里,按节点顺序首尾相接;
/// 分区把左右孩子写进另一个缓冲的同一段位置(左在前、右在后),两缓冲每层
/// 交换一次。不再为每个节点分配两条新 `Vec`。
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct HostSpan {
    start: usize,
    len: usize,
}

impl HostSpan {
    fn of<'a>(&self, buf: &'a [RowId]) -> &'a [RowId] {
        &buf[self.start..self.start + self.len]
    }
}

/// 整层分区的一个任务:某节点的一段连续行。必须是 8 的倍数,任务的方向
/// bit 才按 byte 对齐,各任务写互不重叠的 byte。
const PART_TASK_ROWS: usize = 8 * 1024;

struct PartTask {
    /// 在 `nodes` 里的下标
    node: usize,
    /// 行缓冲里的绝对位置
    rows: std::ops::Range<usize>,
}

/// 一段行的方向 bit:第 `j` 行走左则 `out[j / 8]` 的第 `j % 8` 位为 1。
fn fill_partition_bits<B: crate::bins::BinCode>(
    rows: &[RowId],
    col: &[B],
    c: &SplitCandidate,
    out: &mut [u8],
) {
    debug_assert_eq!(out.len(), rows.len().div_ceil(8));
    let cats = c.cats.as_ref().map(|s| s.words64());
    for (byte, chunk) in out.iter_mut().zip(rows.chunks(8)) {
        let mut acc = 0u8;
        for (j, &r) in chunk.iter().enumerate() {
            let bin = col[r as usize];
            let go_left = if bin == B::MISSING {
                c.missing_left
            } else if let Some(cats) = cats {
                // Categorical: the set goes right, every other category left.
                !crate::categorical::contains_in(cats, bin.idx() as u32)
            } else {
                (bin.idx() as u32) < c.bin
            };
            acc |= (go_left as u8) << j;
        }
        *byte = acc;
    }
}

/// 按选中的分裂把**一整层**节点的行分到左右两边。
///
/// 分区用的是 **bin**(训练侧手里就是量化数据),和 `leaf_values` 里
/// 把阈值换回 bin 再比是同一套语义,`tree::split_bin` /
/// `split_threshold` 的往返测试钉着这一点。
///
/// `nodes` 按 `start` 升序、互不重叠(本层的节点顺序)。节点 `k` 的左孩子
/// 写进 `next[start..start + n_left]`,右孩子紧随其后;两边都保持原顺序
/// (稳定分区),所以孩子的行集合和逐节点分区逐字节相同。
///
/// 任务是 (节点, 一段行) 的扁平列表,整层只有两次 fork-join:
/// 第一趟并行算方向 bit,第二趟并行把行写到各自的输出段。以前是每个节点
/// 三次(bit、计数、写),深层几十个小节点时线程大多在等 join。
///
/// 分区要读的列从哪来。
enum PartCols<'a> {
    /// 驻留 source:每个节点分裂特征的列(与 `nodes` 一一对应),零拷贝。
    Resident(Vec<crate::bins::BinSlice<'a>>),
    /// 只能交出拷贝的 source:按特征分批取,任何时刻只持有一列
    /// (峰值内存同逐列分区)。
    Fetch(&'a dyn BlockSource),
}

impl<'a> PartCols<'a> {
    fn for_nodes(source: &'a dyn BlockSource, nodes: &[(HostSpan, SplitCandidate)]) -> Self {
        match nodes
            .iter()
            .map(|(_, c)| source.resident_feature_column(c.feat))
            .collect()
        {
            Some(cols) => Self::Resident(cols),
            None => Self::Fetch(source),
        }
    }
}

fn partition_level_host(
    rows: &[RowId],
    next: &mut [RowId],
    nodes: &[(HostSpan, SplitCandidate)],
    cols: PartCols<'_>,
    comm: &dyn Comm,
) -> anyhow::Result<Vec<(HostSpan, HostSpan)>> {
    debug_assert_eq!(rows.len(), next.len());
    debug_assert!(nodes
        .windows(2)
        .all(|w| w[0].0.start + w[0].0.len <= w[1].0.start));
    let bits_timer = prof::Timer::new(&prof::PART_BITS);
    // 节点 k 的方向 bit 占 `node_bits[k]` 这段 byte(按节点首尾相接)。
    let mut tasks: Vec<PartTask> = Vec::new();
    let mut node_bits: Vec<std::ops::Range<usize>> = Vec::with_capacity(nodes.len());
    let mut n_bytes = 0usize;
    for (k, (span, _)) in nodes.iter().enumerate() {
        let mut off = 0usize;
        while off < span.len {
            let end = (off + PART_TASK_ROWS).min(span.len);
            tasks.push(PartTask {
                node: k,
                rows: span.start + off..span.start + end,
            });
            off = end;
        }
        node_bits.push(n_bytes..n_bytes + span.len.div_ceil(8));
        n_bytes += span.len.div_ceil(8);
    }
    let mut bits = vec![0u8; n_bytes];
    {
        // 每个任务独占自己那段 byte:任务按 byte 位置升序,顺次切开即可。
        let mut work: Vec<(&PartTask, &mut [u8])> = Vec::with_capacity(tasks.len());
        let mut rest = bits.as_mut_slice();
        for t in &tasks {
            let (head, tail) = rest.split_at_mut(t.rows.len().div_ceil(8));
            work.push((t, head));
            rest = tail;
        }
        debug_assert!(rest.is_empty());
        match cols {
            PartCols::Resident(cols) => work.into_par_iter().for_each(|(t, out)| {
                let c = &nodes[t.node].1;
                crate::with_bins!(cols[t.node], |col| fill_partition_bits(
                    &rows[t.rows.clone()],
                    col,
                    c,
                    out
                ));
            }),
            PartCols::Fetch(source) => {
                let mut feats: Vec<FeatId> = nodes.iter().map(|(_, c)| c.feat).collect();
                feats.sort_unstable();
                feats.dedup();
                for feat in feats {
                    let fetch_timer = prof::Timer::new(&prof::PART_FETCH);
                    let col = source.feature_column(feat)?;
                    drop(fetch_timer);
                    let (this, rest): (Vec<_>, Vec<_>) = work
                        .into_iter()
                        .partition(|(t, _)| nodes[t.node].1.feat == feat);
                    work = rest;
                    let col = col.as_slice();
                    this.into_par_iter().for_each(|(t, out)| {
                        let c = &nodes[t.node].1;
                        crate::with_bins!(col, |col| fill_partition_bits(
                            &rows[t.rows.clone()],
                            col,
                            c,
                            out
                        ));
                    });
                }
                debug_assert!(work.is_empty());
            }
        }
    }
    if prof::enabled() {
        use std::sync::atomic::Ordering;
        let mut feats: Vec<FeatId> = nodes.iter().map(|(_, c)| c.feat).collect();
        feats.sort_unstable();
        feats.dedup();
        prof::PART_COLS.fetch_add(feats.len() as u64, Ordering::Relaxed);
        prof::PART_ROWS.fetch_add(
            nodes.iter().map(|(s, _)| s.len as u64).sum::<u64>(),
            Ordering::Relaxed,
        );
    }
    // 每行 1 bit。多卡时只有持有该特征的一方知道每行去哪边,这个
    // 结果必须广播出去 —— 这是主要通信开销,阶段 4 要单独 benchmark。
    // 单机 Local 是空操作,但走一遍这条路,换后端时训练逻辑不用动。
    for ((_, c), range) in nodes.iter().zip(&node_bits) {
        comm.broadcast_partition(owner_rank(c.feat), &mut bits[range.clone()]);
    }
    drop(bits_timer);

    let _t = prof::Timer::new(&prof::PART_SPLIT);
    // 计数在广播**之后**按 bit 数:非持有方的方向全来自广播。
    // 整层 bit 只有行数 / 8 个 byte,串行数一遍可以忽略。
    let mut byte = 0usize;
    let left_counts: Vec<usize> = tasks
        .iter()
        .map(|t| {
            let n = t.rows.len();
            let bytes = &bits[byte..byte + n.div_ceil(8)];
            byte += n.div_ceil(8);
            let full: usize = bytes[..n / 8].iter().map(|b| b.count_ones() as usize).sum();
            let tail = if n % 8 != 0 {
                (bytes[n / 8] & ((1u8 << (n % 8)) - 1)).count_ones() as usize
            } else {
                0
            };
            full + tail
        })
        .collect();

    // 输出切片:节点 k 的那段是 [各任务的左段..., 各任务的右段...],
    // 节点之间的空隙(本层不分裂的叶子)跳过。全部互不重叠,无需加锁。
    let mut outs: Vec<(&mut [RowId], &mut [RowId])> = Vec::with_capacity(tasks.len());
    let mut spans = Vec::with_capacity(nodes.len());
    let mut rest = next;
    let mut pos = 0usize;
    let mut ti = 0usize;
    for (k, (span, _)) in nodes.iter().enumerate() {
        let (_, tail) = rest.split_at_mut(span.start - pos);
        rest = tail;
        let first = ti;
        while ti < tasks.len() && tasks[ti].node == k {
            ti += 1;
        }
        let n_left: usize = left_counts[first..ti].iter().sum();
        let (mut left, tail) = rest.split_at_mut(n_left);
        let (mut right, tail) = tail.split_at_mut(span.len - n_left);
        rest = tail;
        pos = span.start + span.len;
        for (t, &nl) in tasks[first..ti].iter().zip(&left_counts[first..ti]) {
            let (l, lt) = left.split_at_mut(nl);
            let (r, rt) = right.split_at_mut(t.rows.len() - nl);
            left = lt;
            right = rt;
            outs.push((l, r));
        }
        spans.push((
            HostSpan {
                start: span.start,
                len: n_left,
            },
            HostSpan {
                start: span.start + n_left,
                len: span.len - n_left,
            },
        ));
    }

    let mut byte_starts = Vec::with_capacity(tasks.len());
    let mut byte = 0usize;
    for t in &tasks {
        byte_starts.push(byte);
        byte += t.rows.len().div_ceil(8);
    }
    let orders = partition_orders();
    let bits = &bits;
    tasks
        .par_iter()
        .zip(byte_starts.par_iter())
        .zip(outs.into_par_iter())
        .for_each(|((t, &b0), (left_out, right_out))| {
            let rows = &rows[t.rows.clone()];
            let (mut li, mut ri) = (0usize, 0usize);
            for (chunk, &mask) in rows.chunks(8).zip(&bits[b0..]) {
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
                    // 只有节点最后一个 byte 可能不足 8 行。高位不是输入,
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
    Ok(spans)
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
    obj.base_margin(base_score)
}

/// 单行的梯度;定义在 `Objective::gradient`。`quantile_scale` 只给
/// `reg:quantileerror` 用(每轮的全局 scale),其它 objective 忽略。
#[inline]
fn gradient_of(pred: f32, label: f32, obj: Objective, quantile_scale: f32) -> GradPair {
    match obj {
        // 单目标路径(多个 alpha 时按目标拆成单 alpha 的 objective 再调用)。
        Objective::Quantile { alphas } => {
            Objective::quantile_gradient(alphas.as_slice()[0], quantile_scale, pred, label)
        }
        _ => obj.gradient(pred, label),
    }
}

/// 训练会不会有一份逐行的梯度权重(样本权重,或折进 `scale_pos_weight`
/// 之后的隐式权重)。`gpu_math="fast"` 时它常驻显存(4 B/行),所以显存
/// 规划器必须用**同一个**判断 —— 不能只看用户有没有传样本权重。
/// 与 `gradient_weights(..).is_some() || weights.is_some()` 逐项一致。
pub(crate) fn has_gradient_weights(obj: Objective, has_sample_weights: bool, spw: f32) -> bool {
    has_sample_weights || (spw != 1.0 && obj.uses_scale_pos_weight())
}

/// 把 `scale_pos_weight` 折进梯度权重(XGBoost `RegLossObj::GetGradient`:
/// `w = weight or 1; if label == 1.0 { w *= spw }`)。spw == 1 返回 `None`。
fn gradient_weights(
    obj: Objective,
    labels: &[f32],
    weights: Option<&[f32]>,
    spw: f32,
) -> Option<Vec<f32>> {
    if spw == 1.0 || !obj.uses_scale_pos_weight() {
        return None;
    }
    Some(
        labels
            .par_iter()
            .enumerate()
            .map(|(i, &y)| {
                let w = weights.map_or(1.0, |w| w[i]);
                if y == 1.0 {
                    w * spw
                } else {
                    w
                }
            })
            .collect(),
    )
}

/// XGBoost `RegLossObj::GetGradient`:`GradientPair(grad * w, hess * w)`,
/// hess 的 `max(.., 1e-16)` 在乘权重**之前**。f32 乘法,与 XGBoost 同。
#[inline]
fn weighted(g: GradPair, w: f32) -> GradPair {
    GradPair {
        grad: g.grad * w,
        hess: g.hess * w,
    }
}

fn compute_gradients_weighted(
    pred: &[f32],
    label: &[f32],
    weight: &[f32],
    obj: Objective,
    quantile_scale: f32,
) -> Vec<GradPair> {
    pred.par_iter()
        .zip(label.par_iter().zip(weight.par_iter()))
        .map(|(&p, (&y, &w))| weighted(gradient_of(p, y, obj, quantile_scale), w))
        .collect()
}

/// 每行独立,没有归约,所以并行**不可能**改变任何一个元素的值 ——
/// `collect` 保序,第 i 个输出永远只由第 i 个输入决定。
///
/// 这一段和下面的量化是 Amdahl 分析里那部分「按行的串行工作」的主要成分:
/// 它们只随行数走、完全不进列块那一维,所以以前无论开多少线程都是单线程跑。
/// GPU 路径上尤其明显 —— HIGGS 1050 万行时 GRAD + QUANTIZE 合计约
/// 145 ms/轮,占整轮 0.345 s 的四成以上,而那时 GPU 在干等。
fn compute_gradients(
    pred: &[f32],
    label: &[f32],
    obj: Objective,
    quantile_scale: f32,
) -> Vec<GradPair> {
    pred.par_iter()
        .zip(label.par_iter())
        .map(|(&p, &y)| gradient_of(p, y, obj, quantile_scale))
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
    #[test]
    fn identity_rows_matches_the_serial_range() {
        for n in [0usize, 1, 7, (1 << 18) - 1, (1 << 18) + 5, 700_001] {
            let expected: Vec<RowId> = (0..n as RowId).collect();
            assert_eq!(super::identity_rows(n), expected, "n = {n}");
        }
    }

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
        let mut p = TrainParams {
            max_depth: 0,
            n_rounds: 1,
            ..Default::default()
        };
        p.cols_per_block = n_feats;
        let cfg = TrainConfig::new(p, Objective::SquaredError);
        let src = DenseSource::from_row_major(&x, n_rows, n_feats, 32, n_feats);
        train(&src, &y, &[], &cfg, &mut [], &Local).expect("max_depth=0 is a valid root-only tree");
        // 上限就是那个 buffer 装得下的最深一层。
        let deepest_partitioned_level = crate::types::MAX_DEPTH_LIMIT - 1;
        assert_eq!(
            1usize << deepest_partitioned_level,
            8192,
            "depth 14 uses exactly 8192 slots"
        );
        assert_eq!(
            1usize << crate::types::MAX_DEPTH_LIMIT,
            16384,
            "depth 15 needs 16384 slots"
        );
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
            let mut p = TrainParams {
                max_depth: bad,
                n_rounds: 1,
                ..Default::default()
            };
            p.cols_per_block = n_feats;
            let cfg = TrainConfig::new(p, Objective::SquaredError);
            let src = DenseSource::from_row_major(&x, n_rows, n_feats, 32, n_feats);
            let err = train(&src, &y, &[], &cfg, &mut [], &Local)
                .expect_err(&format!("max_depth={bad} should be rejected"));
            assert!(
                err.to_string().contains("max_depth"),
                "the error should mention max_depth, got: {err}"
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
        train(&src, &y, &[], &cfg, &mut [], &Local).expect("depth 14 must train");
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
    use crate::columns::{BinCuts, BinningStrategy, ColumnBlock};
    use crate::comm::Local;
    use crate::metrics::{LogLoss, Metric, Rmse};
    use crate::source::DenseSource;
    use crate::types::{Bin, MISSING_BIN};

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
                "row {row} {raw:?}: quantized descent {got} != float descent {by_floats}"
            );
        }
    }

    /// Two numeric columns, then two categorical code columns (5 and 12
    /// levels). The label depends on category membership, so categorical
    /// splits are worth taking.
    fn categorical_data(n: usize) -> (Vec<f32>, Vec<f32>, usize, Vec<bool>) {
        let mut x = Vec::new();
        let mut y = Vec::new();
        for i in 0..n {
            let a = ((i * 7) % 9) as f32;
            let b = if i % 11 == 0 {
                MISSING
            } else {
                ((i * 5) % 6) as f32
            };
            let c1 = ((i * 3 + i / 7) % 5) as f32;
            let c2 = if i % 13 == 0 {
                MISSING
            } else {
                ((i * 17 + i / 5) % 12) as f32
            };
            let up = matches!(c1 as i32, 1 | 3) || matches!(c2 as i32, 2 | 5 | 9);
            x.extend([a, b, c1, c2]);
            y.push(if up { 1.0 } else { 0.0 } + 0.05 * a);
        }
        (x, y, 4, vec![false, false, true, true])
    }

    fn train_categorical(
        x: &[f32],
        y: &[f32],
        nf: usize,
        flags: &[bool],
        p: TrainParams,
    ) -> (Model, DenseSource) {
        let src = DenseSource::new(
            x,
            y.len(),
            nf,
            BinningStrategy::SketchTyped {
                max_bins: p.max_bin as usize,
                categorical: flags,
            },
            p.cols_per_block,
        );
        let cfg = TrainConfig::new(p, Objective::SquaredError);
        let model = train(&src, y, &[], &cfg, &mut [], &Local).unwrap();
        (model, src)
    }

    #[test]
    fn categorical_bin_descent_matches_float_descent() {
        let (x, y, nf, flags) = categorical_data(400);
        let (model, src) = train_categorical(&x, &y, nf, &flags, params(4, 6));
        assert_eq!(model.feature_types, flags);
        assert!(
            model.has_categorical_splits(),
            "the data rewards categorical splits"
        );
        let by_bins = predict_on(&model, &src).unwrap();
        for (row, got) in by_bins.iter().enumerate() {
            let raw = &x[row * nf..(row + 1) * nf];
            assert_eq!(
                got.to_bits(),
                model.predict_margin(raw).to_bits(),
                "row {row} {raw:?}"
            );
        }
    }

    #[test]
    fn categorical_training_ignores_block_width_and_threads() {
        let (x, y, nf, flags) = categorical_data(600);
        let mut reference = None;
        for (cols, nthread) in [(1usize, 1usize), (2, 2), (4, 3)] {
            let p = TrainParams {
                cols_per_block: cols,
                nthread,
                ..params(4, 4)
            };
            let (m, _) = train_categorical(&x, &y, nf, &flags, p);
            let json = m.to_xgboost_json().unwrap();
            match &reference {
                None => reference = Some(json),
                Some(r) => assert_eq!(&json, r, "cols {cols} threads {nthread}"),
            }
        }
    }

    #[test]
    fn categorical_validation_set_with_an_unseen_code_is_scored_left() {
        let (x, y, nf, flags) = categorical_data(300);
        let (model, train_src) = train_categorical(&x, &y, nf, &flags, params(3, 4));
        // Same rows with the last column's codes pushed past the trained range.
        let mut vx = x.clone();
        for row in 0..y.len() {
            vx[row * nf + 3] = 40.0;
        }
        let valid = DenseSource::with_cuts(&vx, y.len(), nf, train_src.cuts().clone(), 2);
        let by_bins = predict_on(&model, &valid).unwrap();
        for (row, got) in by_bins.iter().enumerate() {
            let raw = &vx[row * nf..(row + 1) * nf];
            assert_eq!(
                got.to_bits(),
                model.predict_margin(raw).to_bits(),
                "row {row}"
            );
        }
    }

    #[test]
    fn invalid_categorical_codes_are_rejected_when_binning_the_training_data() {
        let (mut x, y, nf, flags) = categorical_data(50);
        x[2] = 2.5;
        let r = std::panic::catch_unwind(|| {
            DenseSource::new(
                &x,
                y.len(),
                nf,
                BinningStrategy::SketchTyped {
                    max_bins: 16,
                    categorical: &flags,
                },
                2,
            )
        });
        assert!(r.is_err());
    }

    #[test]
    fn logistic_learns_the_step() {
        let (x, y, nf) = step_data(80);
        let (model, src) = train_on(&x, &y, nf, params(3, 30), Objective::Logistic);
        let pred = predict_on(&model, &src).unwrap();

        let ll = crate::metrics::LogLoss.eval(&pred, &y);
        assert!(
            ll < 0.2,
            "logloss should be small after 30 rounds, got {ll}"
        );
        for (i, (&p, &label)) in pred.iter().zip(&y).enumerate() {
            let prob = 1.0 / (1.0 + (-p).exp());
            assert_eq!(
                prob > 0.5,
                label > 0.5,
                "row {i} misclassified (prob {prob})"
            );
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
                "rmse {rmse} at {rounds} rounds is not smaller than the previous {last}"
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
                "tree {k}: sum of leaf hess {leaves} != root {root}"
            );
        }
    }

    #[test]
    fn respects_max_depth() {
        let (x, y, nf) = step_data(100);
        for max_depth in [0u32, 1, 2, 4] {
            let (model, _) = train_on(&x, &y, nf, params(max_depth, 2), Objective::Logistic);
            for t in &model.trees {
                assert!(
                    depth_of(t, 0) <= max_depth,
                    "max_depth {max_depth} exceeded"
                );
            }
        }
    }

    #[test]
    fn planner_weight_predicate_matches_training_allocation() {
        let y = vec![0.0f32, 1.0, 1.0, 0.0];
        let w = vec![0.5f32; 4];
        for obj in Objective::ALL {
            for spw in [1.0f32, 0.0, 2.5] {
                for sample in [None, Some(w.as_slice())] {
                    let allocated =
                        gradient_weights(obj, &y, sample, spw).is_some() || sample.is_some();
                    assert_eq!(
                        has_gradient_weights(obj, sample.is_some(), spw),
                        allocated,
                        "{obj:?} spw={spw} sample={}",
                        sample.is_some()
                    );
                }
            }
        }
    }

    #[test]
    fn default_base_score_is_the_weighted_label_mean() {
        let x: Vec<f32> = (0..40).map(|i| (i % 7) as f32).collect();
        let y: Vec<f32> = (0..40).map(|i| f32::from(i % 3 == 0)).collect();
        let w: Vec<f32> = (0..40)
            .map(|i| {
                if i % 3 == 0 {
                    3.0
                } else {
                    0.5 + (i % 4) as f32
                }
            })
            .collect();
        let src = DenseSource::from_row_major(&x, 40, 1, 255, 1);
        // 走加权均值(FitInterceptGlmLike)的 objective;gamma 要求标签为正,
        // pseudohuber 走 Newton,quantile 取分位数,各有自己的测试。
        let glm_like = Objective::ALL.into_iter().filter(|o| {
            !matches!(
                o,
                Objective::Gamma | Objective::PseudoHuber { .. } | Objective::Quantile { .. }
            )
        });
        for obj in glm_like {
            let cfg = TrainConfig::new(params(2, 1), obj);
            let m = train(&src, &y, &[], &cfg, &mut [], &Local).unwrap();
            let mean = y.iter().fold(0.0f64, |a, &v| a + v as f64 / 40.0) as f32;
            assert_eq!(m.base_score, mean, "{obj:?}");
            let mw = train_weighted(&src, &y, Some(&w), &[], &cfg, &mut [], &Local).unwrap();
            assert_eq!(
                mw.base_score,
                obj.estimate_base_score(&y, Some(&w)),
                "{obj:?}"
            );
            assert_ne!(mw.base_score, mean);
        }
        // 全零权重:退回 0.5 而不是 NaN。
        let zero = vec![0.0f32; 40];
        assert_eq!(
            Objective::Logistic.estimate_base_score(&y, Some(&zero)),
            0.5
        );
        // 标签全相同的 logistic:估出 0 / 1,margin 被夹住而不是 ±inf。
        let ones = vec![1.0f32; 40];
        let cfg = TrainConfig::new(params(2, 1), Objective::Logistic);
        let m = train(&src, &ones, &[], &cfg, &mut [], &Local).unwrap();
        assert_eq!(m.base_score, 1.0);
        assert!(Objective::Logistic.base_margin(m.base_score).is_finite());
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
        // 显式 base_score:估计出的标签均值正好等于全部标签,梯度为 0。
        let src = DenseSource::from_row_major(&x, 8, 1, p.max_bin, p.cols_per_block);
        let cfg = TrainConfig {
            base_score: Some(0.5),
            ..TrainConfig::new(p, Objective::SquaredError)
        };
        let model = train(&src, &y, &[], &cfg, &mut [], &Local).unwrap();

        let first = model.trees[0].nodes[0].leaf_value;
        let second = model.trees[1].nodes[0].leaf_value;
        assert!(
            second > 0.0 && second < first,
            "the second round should see the updated predictions: {first} -> {second}"
        );
    }

    #[test]
    fn rejects_mismatched_training_and_eval_labels_before_training() {
        let (x, y, nf) = step_data(20);
        let src = DenseSource::from_row_major(&x, 20, nf, 16, 2);
        let cfg = TrainConfig::new(params(2, 2), Objective::Logistic);

        let err = train(&src, &y[..19], &[], &cfg, &mut [], &Local)
            .expect_err("mismatched training label length must error");
        assert!(err.to_string().contains("labels have 19 rows"), "{err:#}");

        let eval = EvalSet {
            name: "valid".into(),
            source: &src,
            labels: &y[..18],
            weights: None,
            group_ptr: None,
        };
        let err = train(&src, &y, &[eval], &cfg, &mut [], &Local)
            .expect_err("mismatched validation label length must error");
        assert!(
            err.to_string().contains("validation set \"valid\""),
            "{err:#}"
        );
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
    fn fused_host_quantization_matches_the_separate_passes() {
        // 跨多个块并带不足 64 行的尾巴;含零梯度行和被采样掉的行。
        let n = 3 * PREP_CHUNK + 77;
        let base: Vec<GradPair> = (0..n)
            .map(|i| GradPair {
                grad: if i % 11 == 0 {
                    0.0
                } else {
                    ((i % 97) as f32 - 48.0) * 0.013
                },
                hess: if i % 11 == 0 {
                    0.0
                } else {
                    0.05 + (i % 13) as f32 * 0.01
                },
            })
            .collect();
        for mask in [
            None,
            Some((crate::subsample::threshold_for(0.6), 7u64, 3u64)),
        ] {
            let run = |nthread| {
                crate::threading::build_pool(nthread).unwrap().install(|| {
                    let mut g = base.clone();
                    let (mg, mh, kept) = mask_and_bounds(&mut g, mask);
                    let q = GradQuantizer::try_from_bounds(mg, mh, n).unwrap();
                    let mut fixed = Vec::new();
                    let (sum, bits) = quantize_host(&g, &q, &mut fixed);
                    (g, kept, fixed, sum, bits.rows(), bits.count())
                })
            };
            let (g, kept, fixed, sum, rows, count) = run(1);
            let eight = run(8);
            assert_eq!(fixed, eight.2);
            assert_eq!(rows, eight.4);
            assert_eq!((kept, sum, count), (eight.1, eight.3, eight.5));

            // 分开做的参照:先置零、再 try_new、逐行 to_fixed、node_sum、nonzero_rows。
            let mut want = base.clone();
            let mut want_kept = n;
            if let Some((th, seed, tree)) = mask {
                want_kept = 0;
                for (i, w) in want.iter_mut().enumerate() {
                    if crate::subsample::row_selected(i as u64, th, seed, tree) {
                        want_kept += 1;
                    } else {
                        *w = GradPair {
                            grad: 0.0,
                            hess: 0.0,
                        };
                    }
                }
            }
            let bits_of = |v: &[GradPair]| {
                v.iter()
                    .map(|g| (g.grad.to_bits(), g.hess.to_bits()))
                    .collect::<Vec<_>>()
            };
            assert_eq!(bits_of(&g), bits_of(&want));
            assert_eq!(kept, want_kept);
            let q = GradQuantizer::try_new(&want, n).unwrap();
            let want_fixed: Vec<GradPairFixed> = want.iter().map(|&g| q.to_fixed(g)).collect();
            assert_eq!(fixed, want_fixed);
            assert_eq!(sum, hist::node_sum(&want_fixed, None));
            assert_eq!(rows, nonzero_rows(&want_fixed));
            assert_eq!(count, rows.len());
        }
    }

    #[test]
    fn level_partition_is_thread_count_independent_and_stable() {
        // 多个任务大小以上,并带不足 8 行的尾巴;节点之间留空隙(不分裂的叶子)。
        let n = 100_003usize;
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
        let spans = [
            HostSpan {
                start: 0,
                len: 40_001,
            },
            HostSpan {
                start: 40_001,
                len: 3,
            },
            HostSpan {
                start: 50_000,
                len: n - 50_000,
            },
        ];

        for missing_left in [false, true] {
            let cand = |bin| SplitCandidate {
                feat: 0,
                bin,
                gain: 1.0,
                left: GradPairFixed::default(),
                right: GradPairFixed::default(),
                missing_left,
                cats: None,
            };
            let nodes: Vec<(HostSpan, SplitCandidate)> = spans
                .iter()
                .zip([3, 1, 4])
                .map(|(s, b)| (*s, cand(b)))
                .collect();
            let run = |nthread| {
                let mut next = vec![RowId::MAX; n];
                let out = crate::threading::build_pool(nthread)
                    .unwrap()
                    .install(|| {
                        partition_level_host(
                            &rows,
                            &mut next,
                            &nodes,
                            PartCols::Resident(vec![crate::bins::BinSlice::U8(&col); nodes.len()]),
                            &Local,
                        )
                    })
                    .unwrap();
                (out, next)
            };
            let one = run(1);
            let eight = run(8);
            assert_eq!(
                one, eight,
                "row indices must be byte-identical with 1 and 8 threads"
            );

            let (out, next) = one;
            for ((span, c), (left, right)) in nodes.iter().zip(&out) {
                let goes_left = |&row: &RowId| {
                    let bin = col[row as usize];
                    if bin == MISSING_BIN {
                        missing_left
                    } else {
                        (bin as u32) < c.bin
                    }
                };
                let input = span.of(&rows);
                assert_eq!(left.start, span.start);
                assert_eq!(right.start, left.start + left.len);
                assert_eq!(left.len + right.len, span.len);
                assert_eq!(
                    left.of(&next),
                    input
                        .iter()
                        .filter(|r| goes_left(r))
                        .copied()
                        .collect::<Vec<_>>()
                );
                assert_eq!(
                    right.of(&next),
                    input
                        .iter()
                        .filter(|r| !goes_left(r))
                        .copied()
                        .collect::<Vec<_>>()
                );
            }
            // 空隙(不分裂的节点)不被写。
            assert!(next[40_004..50_000].iter().all(|&r| r == RowId::MAX));
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
            base_score: Some(0.5),
            ..TrainConfig::new(p, Objective::SquaredError)
        };
        let model = train(&src, &y, &[], &cfg, &mut [], &Local).unwrap();

        assert_eq!(model.trees[0].len(), 1, "should not split");
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
            assert_eq!(
                x.to_bits(),
                y.to_bits(),
                "row {i} differs between two training runs"
            );
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
            weights: None,
            group_ptr: None,
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
        assert_eq!(
            model.best_iteration,
            Some(0),
            "the best iteration should be the first round"
        );
        assert_eq!(
            n_trees, 4,
            "best (1 tree) + 3 rounds without improvement = 4 trees"
        );
        assert!(model.best_score.is_some());
        assert!(
            n_trees < 50,
            "early stopping did not trigger, ran all 50 rounds"
        );
    }

    #[test]
    fn early_stopping_rounds_controls_the_patience() {
        for rounds in [1usize, 2, 5] {
            let (_, n_trees) = run_early_stopping(rounds, 50);
            assert_eq!(
                n_trees,
                rounds + 1,
                "patience of {rounds} rounds should leave {} trees",
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
            weights: None,
            group_ptr: None,
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
                "round {round}: incremental metric differs from full recomputation"
            );
        }
    }

    #[test]
    fn watched_metric_is_the_last_of_the_last_eval() {
        // entries 是「eval 外层、metric 内层」,early stopping 应盯最后
        // 一个 eval 下的最后一个 metric。
        let m = RoundMetrics {
            round: 0,
            entries: vec![
                ("train".into(), "logloss".into(), 1.0),
                ("train".into(), "rmse".into(), 2.0),
                ("valid".into(), "logloss".into(), 3.0),
                ("valid".into(), "rmse".into(), 4.0),
            ],
        };
        assert_eq!(m.watched().unwrap().1, "rmse");
        assert_eq!(m.watched().unwrap().2, 4.0);
        assert_eq!(m.last().unwrap().2, 4.0, "last() is the last metric");
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
            weights: None,
            group_ptr: None,
        }];
        let recorder = RecordHistory::new();
        let history = recorder.handle();
        let mut cbs: Vec<Box<dyn Callback>> = vec![Box::new(recorder)];
        train(&src, &y, &evals, &cfg, &mut cbs, &Local).unwrap();

        // 两个 metric 都要出现且保留配置顺序;watched 选最后一个。
        let recorded = history.lock().unwrap();
        assert_eq!(recorded.len(), 3);
        for m in recorded.iter() {
            assert_eq!(m.entries.len(), 2, "one eval × two metrics");
            assert_eq!(m.entries[0].1, "logloss");
            assert_eq!(m.entries[1].1, "rmse");
            assert_eq!(
                m.watched().unwrap().1,
                "rmse",
                "the watched metric must be the last one"
            );
            assert_eq!(m.watched().unwrap().2, m.entries[1].2);
        }
    }

    #[test]
    fn early_stopping_uses_last_metric_even_when_first_metric_keeps_improving() {
        struct ConstantMetric;
        impl Metric for ConstantMetric {
            fn name(&self) -> &str {
                "constant"
            }

            fn eval(&self, _pred: &[f32], _label: &[f32]) -> f32 {
                1.0
            }
        }

        let (x, y, n_features) = step_data(60);
        let p = params(2, 20);
        let train_src =
            DenseSource::from_row_major(&x, 60, n_features, p.max_bin, p.cols_per_block);
        let valid = DenseSource::with_cuts(
            &x,
            60,
            n_features,
            DenseSource::from_row_major(&x, 60, n_features, p.max_bin, p.cols_per_block)
                .into_cuts(),
            p.cols_per_block,
        );
        let cfg = TrainConfig {
            eval_metrics: vec![Box::new(LogLoss), Box::new(ConstantMetric)],
            ..TrainConfig::new(p, Objective::Logistic)
        };
        let evals = [EvalSet {
            name: "valid".into(),
            source: &valid,
            labels: &y,
            weights: None,
            group_ptr: None,
        }];
        let recorder = RecordHistory::new();
        let history = recorder.handle();
        let mut callbacks: Vec<Box<dyn Callback>> =
            vec![Box::new(recorder), Box::new(EarlyStopping::new(2, true))];

        let model = train(&train_src, &y, &evals, &cfg, &mut callbacks, &Local).unwrap();
        assert_eq!(
            model.trees.len(),
            3,
            "equal watched scores stop after patience two"
        );
        assert_eq!(model.best_iteration, Some(0));
        assert_eq!(model.best_score, Some(1.0));

        let recorded = history.lock().unwrap();
        assert_eq!(recorded.len(), 3);
        let first_metric: Vec<f32> = recorded
            .iter()
            .map(|round| {
                assert_eq!(round.entries.len(), 2);
                assert_eq!(round.entries[0].1, "logloss");
                assert_eq!(round.entries[1].1, "constant");
                assert_eq!(round.entries[1].2, 1.0);
                assert_eq!(round.watched().unwrap().1, "constant");
                round.entries[0].2
            })
            .collect();
        assert!(
            first_metric.windows(2).all(|pair| pair[1] < pair[0]),
            "the earlier logloss metric must keep improving over the three recorded rounds: {first_metric:?}"
        );
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
            fn feature_column(&self, f: FeatId) -> anyhow::Result<crate::bins::BinData> {
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
            "only {internal} internal nodes and {n_blocks} blocks: data is too simple to test per-node loading"
        );
        // 训练预测直接用 build_tree 留下的叶子行分区,不再加载块下降。
        // 所以整块加载只剩建直方图:每层每块一次。
        let ceiling = n_blocks * depth * rounds;
        assert!(
            blocks <= ceiling,
            "block() was called {blocks} times, limit {n_blocks} blocks × {depth} levels × {rounds} rounds \
             = {ceiling} (histogram building only)"
        );
        assert!(
            cols <= nf * depth * rounds,
            "feature_column() was called {cols} times, limit {nf} columns × {depth} levels × {rounds} rounds = {}. \
             Exceeding it means a feature was read twice within a level (dedup did not work)",
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
            weights: None,
            group_ptr: None,
        }];
        let model = train(&train_src, &y, &evals, &cfg, &mut [], &Local).unwrap();

        // 同一份数据、同一套 cuts,训练集和验证集的预测必须一模一样
        let a = predict_on(&model, &train_src).unwrap();
        let b = predict_on(&model, &valid_src).unwrap();
        for (i, (x, y)) in a.iter().zip(&b).enumerate() {
            assert_eq!(x.to_bits(), y.to_bits(), "row {i}");
        }
    }

    #[test]
    fn quantized_leaf_values_clear_feature_lookup_between_blocks_and_passes() {
        use crate::tree::{Node, Tree};

        let rows = [
            0.0,
            0.0,
            3.0, // f0 -> f1 -> f0, left at the last split
            1.0,
            0.0,
            4.0, // f0 -> f1 -> f0, right at the last split
            2.0,
            0.0,
            5.0, // root right
            f32::NAN,
            0.0,
            6.0, // missing f0 follows the root's left default
            f32::NAN,
            f32::NAN,
            7.0, // missing f0 and f1 follow distinct defaults
            1.0,
            2.0,
            8.0, // f1 threshold boundary
            1.0,
            f32::NAN,
            9.0, // missing f1 follows its right default
        ];
        let source = DenseSource::from_row_major(&rows, 7, 3, 32, 1);

        // Each feature is in its own block. The route through nodes 0, 1, 2
        // returns to feature 0 in a later pass after feature 1 was loaded.
        let mut root = Node::leaf(0.0, 1.0);
        root.feat = 0;
        root.split_cond = 2.0;
        root.left = 1;
        root.right = 6;
        root.default_left = true;
        root.is_leaf = false;

        let mut node1 = Node::leaf(0.0, 1.0);
        node1.feat = 1;
        node1.split_cond = 2.0;
        node1.left = 2;
        node1.right = 5;
        node1.default_left = false;
        node1.is_leaf = false;

        let mut node2 = Node::leaf(0.0, 1.0);
        node2.feat = 0;
        node2.split_cond = 1.0;
        node2.left = 3;
        node2.right = 4;
        node2.default_left = false;
        node2.is_leaf = false;

        let tree = Tree {
            nodes: vec![
                root,
                node1,
                node2,
                Node::leaf(10.0, 1.0),
                Node::leaf(20.0, 1.0),
                Node::leaf(30.0, 1.0),
                Node::leaf(99.0, 1.0),
            ],
        };
        let expected = [10.0f32, 20.0, 99.0, 20.0, 30.0, 30.0, 30.0];

        for _ in 0..2 {
            let got = leaf_values(&tree, &source).unwrap();
            assert_eq!(
                got.iter().map(|value| value.to_bits()).collect::<Vec<_>>(),
                expected
                    .iter()
                    .map(|value| value.to_bits())
                    .collect::<Vec<_>>(),
            );
        }

        let single_leaf = Tree {
            nodes: vec![Node::leaf(0.25, 1.0)],
        };
        let got = leaf_values(&single_leaf, &source).unwrap();
        assert_eq!(got.len(), 7);
        assert!(got.iter().all(|value| value.to_bits() == 0.25f32.to_bits()));

        let empty = DenseSource::from_row_major(&[], 0, 3, 32, 1);
        assert!(leaf_values(&tree, &empty).unwrap().is_empty());
    }

    // ---- 样本权重 ----

    fn weight_fixture(n: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>, usize) {
        let (x, y, nf) = step_data(n);
        // 非平凡权重:含 0、小数和大于 1 的值,与标签不相关。
        let w = (0..n)
            .map(|i| match i % 6 {
                0 => 0.0,
                1 => 0.25,
                2 => 1.0,
                3 => 2.5,
                4 => 1.75,
                _ => 4.0,
            })
            .collect();
        (x, y, w, nf)
    }

    fn json(m: &Model) -> String {
        m.to_xgboost_json().unwrap()
    }

    /// Hides the resident-column accessors, forcing the pre-skip row flow.
    struct Opaque(DenseSource);
    impl BlockSource for Opaque {
        fn n_blocks(&self) -> usize {
            self.0.n_blocks()
        }
        fn n_rows(&self) -> usize {
            self.0.n_rows()
        }
        fn n_features(&self) -> usize {
            self.0.n_features()
        }
        fn cuts(&self) -> &BinCuts {
            self.0.cuts()
        }
        fn block_features(&self, i: usize) -> anyhow::Result<Vec<FeatId>> {
            self.0.block_features(i)
        }
        fn with_block(&self, i: usize, f: &mut dyn FnMut(&ColumnBlock)) -> anyhow::Result<()> {
            self.0.with_block(i, f)
        }
        fn feature_column(&self, f: FeatId) -> anyhow::Result<crate::bins::BinData> {
            self.0.feature_column(f)
        }
    }

    /// 梯度全零的行不进行集合、建完树再沿树下降:模型必须和「所有行随每层
    /// 分区一起走」逐字节相同(后者靠一个不暴露驻留列的包装 source 触发)。
    /// 覆盖行采样、权重为零的行、缺失值,以及 monotone 约束。
    #[test]
    fn dropping_zero_gradient_rows_keeps_models_byte_identical() {
        let (n_rows, nf) = (4000usize, 6usize);
        let mut rng = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move || {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (rng >> 33) as u32
        };
        let mut x = Vec::with_capacity(n_rows * nf);
        let mut y = Vec::with_capacity(n_rows);
        let mut w = Vec::with_capacity(n_rows);
        for _ in 0..n_rows {
            let row: Vec<f32> = (0..nf)
                .map(|_| {
                    if next() % 13 == 0 {
                        f32::NAN
                    } else {
                        (next() % 17) as f32
                    }
                })
                .collect();
            let signal = (row[0].max(0.0) - row[1].max(0.0)) / 4.0;
            let p_pos = 1.0 / (1.0 + (-signal).exp());
            y.push(if (next() % 1000) as f32 / 1000.0 < p_pos {
                1.0
            } else {
                0.0
            });
            // 约三成的行权重为零 → 梯度全零,即使不做行采样也会被丢出行集合。
            w.push(if next() % 10 < 3 { 0.0f32 } else { 1.0 });
            x.extend_from_slice(&row);
        }
        for (subsample, use_weights, monotone) in [
            (0.5f32, false, false),
            (0.7, true, false),
            (1.0, true, false),
            (0.6, false, true),
        ] {
            let mut p = params(5, 5);
            p.subsample = subsample;
            p.seed = 11;
            p.colsample_bytree = 0.8;
            if monotone {
                p.monotone_constraints = vec![1, -1, 0, 0, 0, 0];
            }
            let dense = DenseSource::from_row_major(&x, n_rows, nf, p.max_bin, p.cols_per_block);
            let opaque = Opaque(DenseSource::from_row_major(
                &x,
                n_rows,
                nf,
                p.max_bin,
                p.cols_per_block,
            ));
            assert!(dense.has_resident_columns() && !opaque.has_resident_columns());
            let weights = use_weights.then_some(w.as_slice());
            for threads in [1usize, 4] {
                let mut q = p.clone();
                q.nthread = threads;
                let cfg = TrainConfig::new(q, Objective::Logistic);
                let a = train_weighted(&dense, &y, weights, &[], &cfg, &mut [], &Local).unwrap();
                let b = train_weighted(&opaque, &y, weights, &[], &cfg, &mut [], &Local).unwrap();
                assert_eq!(
                    json(&a),
                    json(&b),
                    "subsample={subsample} weights={use_weights} monotone={monotone} threads={threads}"
                );
            }
        }
    }

    /// The zero-gradient row skip must also hold for categorical splits
    /// (set membership goes right, unseen / invalid codes left, NaN follows
    /// the learned direction) and for `reg:quantileerror` (global scale from
    /// all rows' predictions, per-round deltas for dropped rows).
    #[test]
    fn dropping_zero_gradient_rows_is_categorical_and_quantile_safe() {
        let (x, y, nf, flags) = categorical_data(3000);
        let n_rows = y.len();
        let w: Vec<f32> = (0..n_rows)
            .map(|i| if i % 7 < 2 { 0.0 } else { 1.0 })
            .collect();
        for (objective, subsample, use_weights) in [
            (Objective::SquaredError, 0.5f32, false),
            (Objective::SquaredError, 0.7, true),
            (Objective::Logistic, 0.6, true),
            (Objective::quantile(0.7), 0.5, false),
            (Objective::quantile(0.3), 0.8, true),
        ] {
            let mut p = params(5, 5);
            p.subsample = subsample;
            p.seed = 5;
            p.colsample_bytree = 0.75;
            p.colsample_bylevel = 0.8;
            p.colsample_bynode = 0.8;
            let make = || {
                DenseSource::new(
                    &x,
                    n_rows,
                    nf,
                    BinningStrategy::SketchTyped {
                        max_bins: p.max_bin as usize,
                        categorical: &flags,
                    },
                    p.cols_per_block,
                )
            };
            let dense = make();
            let opaque = Opaque(make());
            assert!(dense.has_resident_columns() && !opaque.has_resident_columns());
            let labels: Vec<f32> = if matches!(objective, Objective::Logistic) {
                y.iter().map(|&v| if v > 0.5 { 1.0 } else { 0.0 }).collect()
            } else {
                y.clone()
            };
            let weights = use_weights.then_some(w.as_slice());
            for threads in [1usize, 4] {
                let mut q = p.clone();
                q.nthread = threads;
                let cfg = TrainConfig::new(q, objective);
                let a =
                    train_weighted(&dense, &labels, weights, &[], &cfg, &mut [], &Local).unwrap();
                let b =
                    train_weighted(&opaque, &labels, weights, &[], &cfg, &mut [], &Local).unwrap();
                assert!(
                    objective.is_quantile() || a.has_categorical_splits(),
                    "{objective:?}: the data rewards categorical splits"
                );
                assert_eq!(
                    json(&a),
                    json(&b),
                    "{objective:?} subsample={subsample} weights={use_weights} threads={threads}"
                );
            }
        }
    }

    #[test]
    fn weights_none_and_all_ones_are_byte_identical() {
        let (x, y, nf) = step_data(97);
        let ones = vec![1.0f32; y.len()];
        for obj in [Objective::Logistic, Objective::SquaredError] {
            for subsample in [1.0, 0.7] {
                let mut p = params(3, 4);
                p.subsample = subsample;
                p.seed = 7;
                let src = DenseSource::from_row_major(&x, y.len(), nf, p.max_bin, p.cols_per_block);
                let cfg = TrainConfig::new(p, obj);
                let plain = train(&src, &y, &[], &cfg, &mut [], &Local).unwrap();
                let none = train_weighted(&src, &y, None, &[], &cfg, &mut [], &Local).unwrap();
                let one =
                    train_weighted(&src, &y, Some(&ones), &[], &cfg, &mut [], &Local).unwrap();
                assert_eq!(json(&plain), json(&none), "{obj:?} subsample={subsample}");
                assert_eq!(json(&plain), json(&one), "{obj:?} subsample={subsample}");
            }
        }
    }

    #[test]
    fn weights_change_the_model_and_are_thread_invariant() {
        let (x, y, w, nf) = weight_fixture(150);
        let p = params(3, 3);
        let src = DenseSource::from_row_major(&x, y.len(), nf, p.max_bin, p.cols_per_block);
        let cfg = TrainConfig::new(p.clone(), Objective::Logistic);
        let plain = train(&src, &y, &[], &cfg, &mut [], &Local).unwrap();
        let weighted = train_weighted(&src, &y, Some(&w), &[], &cfg, &mut [], &Local).unwrap();
        assert_ne!(
            json(&plain),
            json(&weighted),
            "weights must actually take effect"
        );
        for nthread in [1, 3] {
            for cols_per_block in [1, 4] {
                let mut q = p.clone();
                q.nthread = nthread;
                q.cols_per_block = cols_per_block;
                let s2 = DenseSource::from_row_major(&x, y.len(), nf, q.max_bin, cols_per_block);
                let c2 = TrainConfig::new(q, Objective::Logistic);
                let m = train_weighted(&s2, &y, Some(&w), &[], &c2, &mut [], &Local).unwrap();
                assert_eq!(
                    json(&weighted),
                    json(&m),
                    "nthread={nthread} cpb={cols_per_block}"
                );
            }
        }
    }

    #[test]
    fn zero_weight_rows_do_not_shape_the_tree() {
        // 零权重行 ≡ 从梯度里拿掉:同一份 cuts 下,与删掉这些行的数据集
        // 应得到相同的树结构。定点 scale 依赖行数,所以叶子值只比近似。
        let (x, y, w, nf) = weight_fixture(180);
        let keep: Vec<usize> = (0..y.len()).filter(|&i| w[i] != 0.0).collect();
        let p = params(3, 2);
        let full = DenseSource::from_row_major(&x, y.len(), nf, p.max_bin, p.cols_per_block);
        let cuts = full.cuts().clone();
        let xk: Vec<f32> = keep
            .iter()
            .flat_map(|&i| x[i * nf..(i + 1) * nf].to_vec())
            .collect();
        let yk: Vec<f32> = keep.iter().map(|&i| y[i]).collect();
        let wk: Vec<f32> = keep.iter().map(|&i| w[i]).collect();
        let reduced = DenseSource::with_cuts(&xk, keep.len(), nf, cuts, p.cols_per_block);
        let cfg = TrainConfig::new(p, Objective::Logistic);
        let a = train_weighted(&full, &y, Some(&w), &[], &cfg, &mut [], &Local).unwrap();
        let b = train_weighted(&reduced, &yk, Some(&wk), &[], &cfg, &mut [], &Local).unwrap();
        assert_eq!(a.trees.len(), b.trees.len());
        for (ta, tb) in a.trees.iter().zip(&b.trees) {
            assert_eq!(ta.nodes.len(), tb.nodes.len());
            for (na, nb) in ta.nodes.iter().zip(&tb.nodes) {
                assert_eq!(
                    (
                        na.is_leaf,
                        na.feat,
                        na.split_cond.to_bits(),
                        na.default_left
                    ),
                    (
                        nb.is_leaf,
                        nb.feat,
                        nb.split_cond.to_bits(),
                        nb.default_left
                    )
                );
                let tol = 1e-5 * na.leaf_value.abs().max(1e-3);
                assert!(
                    (na.leaf_value - nb.leaf_value).abs() <= tol,
                    "{na:?} vs {nb:?}"
                );
            }
        }
    }

    #[test]
    fn rejects_invalid_weights() {
        let (x, y, nf) = step_data(20);
        let src = DenseSource::from_row_major(&x, 20, nf, 16, 2);
        let cfg = TrainConfig::new(params(2, 2), Objective::Logistic);
        for (w, msg) in [
            (vec![1.0f32; 19], "weights have 19 rows"),
            (
                {
                    let mut v = vec![1.0f32; 20];
                    v[3] = -0.5;
                    v
                },
                "non-negative",
            ),
            (
                {
                    let mut v = vec![1.0f32; 20];
                    v[4] = f32::NAN;
                    v
                },
                "non-negative",
            ),
            (
                {
                    let mut v = vec![1.0f32; 20];
                    v[5] = f32::INFINITY;
                    v
                },
                "non-negative",
            ),
        ] {
            let err = train_weighted(&src, &y, Some(&w), &[], &cfg, &mut [], &Local)
                .expect_err("invalid weights must error");
            assert!(err.to_string().contains(msg), "{err:#}");
        }
        let bad = vec![-1.0f32; 20];
        let eval = EvalSet {
            name: "valid".into(),
            source: &src,
            labels: &y,
            weights: Some(&bad),
            group_ptr: None,
        };
        let err = train(&src, &y, &[eval], &cfg, &mut [], &Local)
            .expect_err("validation weights must be checked too");
        assert!(
            err.to_string().contains("validation set \"valid\""),
            "{err:#}"
        );
    }

    #[test]
    fn weighted_eval_metrics_follow_xgboost_formula() {
        use crate::metrics::{LogLoss, Metric, Rmse};
        let pred = [0.5f32, -1.0, 2.0, 0.0];
        let label = [1.0f32, 0.0, 1.0, 0.0];
        let w = [1.0f32, 3.0, 0.5, 0.0];
        let ones = [1.0f32; 4];
        for m in [&Rmse as &dyn Metric, &LogLoss] {
            assert_eq!(
                m.eval_weighted(&pred, &label, Some(&ones)).to_bits(),
                m.eval(&pred, &label).to_bits(),
                "{} all-ones weights must equal unweighted",
                m.name()
            );
        }
        let wsum: f64 = w.iter().map(|&v| v as f64).sum();
        let se: f64 = pred
            .iter()
            .zip(&label)
            .zip(&w)
            .map(|((&p, &y), &w)| w as f64 * ((p - y) as f64).powi(2))
            .sum();
        assert_eq!(
            Rmse.eval_weighted(&pred, &label, Some(&w)),
            (se / wsum).sqrt() as f32
        );
        let ll: f64 = pred
            .iter()
            .zip(&label)
            .zip(&w)
            .map(|((&r, &y), &w)| {
                let p = (1.0 / (1.0 + (-r as f64).exp())).clamp(1e-15, 1.0 - 1e-15);
                w as f64 * -((y as f64) * p.ln() + (1.0 - y as f64) * (1.0 - p).ln())
            })
            .sum();
        assert_eq!(
            LogLoss.eval_weighted(&pred, &label, Some(&w)),
            (ll / wsum) as f32
        );
    }

    // ------------------------------------------------- quantile / 嵌套列采样

    #[test]
    fn quantile_objective_trains_reproducibly_and_tracks_the_quantile() {
        let (x, y, nf) = wiggly_data(2000, 4);
        for alpha in [0.1f32, 0.5, 0.9] {
            let obj = Objective::quantile(alpha);
            let mut p = params(4, 30);
            p.nthread = 1;
            let (a, src) = train_on(&x, &y, nf, p.clone(), obj);
            p.nthread = 3;
            let (b, _) = train_on(&x, &y, nf, p, obj);
            assert_eq!(
                json(&a),
                json(&b),
                "alpha={alpha}: thread count must not change the model"
            );
            let pred = predict_on(&a, &src).unwrap();
            let below = y.iter().zip(&pred).filter(|(y, p)| *y <= *p).count();
            let frac = below as f32 / y.len() as f32;
            assert!((frac - alpha).abs() < 0.12, "alpha={alpha} frac={frac}");
        }
    }

    #[test]
    fn multi_quantile_trains_one_tree_per_alpha_and_equals_single_alpha_models() {
        let (x, y, nf) = wiggly_data(2000, 5);
        let alphas = [0.1f32, 0.5, 0.9];
        let multi_obj = Objective::quantile(0.5)
            .with_quantile_alphas(&alphas)
            .unwrap();
        let mut p = params(4, 6);
        p.nthread = 1;
        let (multi, src) = train_on(&x, &y, nf, p.clone(), multi_obj);
        assert_eq!(multi.trees.len(), 6 * alphas.len());
        assert_eq!(multi.num_outputs(), alphas.len());
        // The thread count never changes the model.
        let mut p4 = p.clone();
        p4.nthread = 4;
        let (multi4, _) = train_on(&x, &y, nf, p4, multi_obj);
        assert_eq!(json(&multi), json(&multi4));
        // Column j is the single-alpha model trained alone (no sampling): same trees, same
        // per-alpha intercept.
        for (j, &alpha) in alphas.iter().enumerate() {
            let (single, _) = train_on(&x, &y, nf, p.clone(), Objective::quantile(alpha));
            let view = multi.class_view(j);
            assert_eq!(view.base_score, single.base_score, "alpha={alpha}");
            assert_eq!(view.trees.len(), single.trees.len());
            let a = predict_on(&view, &src).unwrap();
            let b = predict_on(&single, &src).unwrap();
            assert_eq!(a, b, "alpha={alpha}");
        }
        // JSON: num_target = K, vector base_score, lossless round trip.
        let doc: serde_json::Value = serde_json::from_str(&json(&multi)).unwrap();
        let mp = &doc["learner"]["learner_model_param"];
        assert_eq!(mp["num_target"], "3");
        assert_eq!(mp["num_class"], "0");
        let back = Model::from_xgboost_json(&json(&multi)).unwrap();
        assert_eq!(back.objective, multi_obj);
        assert_eq!(back.class_base_margins, multi.class_base_margins);
        assert_eq!(json(&back), json(&multi));
        // Row predictions are sorted per row (non-crossing quantiles).
        let row: Vec<f32> = x[..nf].to_vec();
        let pred = back.predict_row(&row);
        assert!(pred.windows(2).all(|w| w[0] <= w[1]), "{pred:?}");
    }

    #[test]
    fn nested_column_sampling_is_thread_and_blocking_invariant() {
        let (x, y, nf) = wiggly_data(1500, 6);
        let mut base = params(5, 6);
        base.seed = 9;
        let full = {
            let (m, _) = train_on(&x, &y, nf, base.clone(), Objective::SquaredError);
            json(&m)
        };
        for (bylevel, bynode) in [(0.6f32, 1.0f32), (1.0, 0.6), (0.8, 0.5)] {
            let mut p = base.clone();
            p.colsample_bylevel = bylevel;
            p.colsample_bynode = bynode;
            p.nthread = 1;
            let reference = {
                let (m, _) = train_on(&x, &y, nf, p.clone(), Objective::SquaredError);
                json(&m)
            };
            assert_ne!(
                reference, full,
                "({bylevel}, {bynode}) should change the model"
            );
            for (threads, cols) in [(4usize, 2usize), (2, 1), (3, 5)] {
                let mut q = p.clone();
                q.nthread = threads;
                q.cols_per_block = cols;
                let (m, _) = train_on(&x, &y, nf, q, Objective::SquaredError);
                assert_eq!(
                    json(&m),
                    reference,
                    "({bylevel}, {bynode}) threads={threads} cols_per_block={cols}"
                );
            }
        }
    }

    #[test]
    fn invalid_nested_rates_are_rejected() {
        let (x, y, nf) = wiggly_data(100, 1);
        for (bylevel, bynode) in [(0.0f32, 1.0f32), (1.0, 0.0), (1.5, 1.0), (1.0, f32::NAN)] {
            let mut p = params(2, 1);
            p.colsample_bylevel = bylevel;
            p.colsample_bynode = bynode;
            let src = DenseSource::from_row_major(&x, y.len(), nf, p.max_bin, p.cols_per_block);
            let cfg = TrainConfig::new(p, Objective::SquaredError);
            let err = train(&src, &y, &[], &cfg, &mut [], &Local).unwrap_err();
            assert!(err.to_string().contains("colsample_by"), "{err}");
        }
    }

    // ------------------------------------------------------------ 单调约束

    /// 非单调的真实关系(x0 上先升后降、x1 上锯齿),约束才会真的起作用。
    fn wiggly_data(n: usize, seed: u64) -> (Vec<f32>, Vec<f32>, usize) {
        let nf = 5;
        let mut state = seed.wrapping_mul(0x9E37_79B9_7F4A_7C15).wrapping_add(1);
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32) / (1u64 << 24) as f32
        };
        let mut x = Vec::with_capacity(n * nf);
        let mut y = Vec::with_capacity(n);
        for i in 0..n {
            let row: Vec<f32> = (0..nf).map(|_| next() * 4.0 - 2.0).collect();
            let signal =
                (row[0] * 1.7).sin() + 0.8 * (row[1] * 3.0).cos() - 0.5 * row[2] + 0.3 * next();
            for (f, v) in row.iter().enumerate() {
                x.push(if (i + f) % 13 == 0 { MISSING } else { *v });
            }
            y.push(signal);
        }
        (x, y, nf)
    }

    fn assert_monotone(model: &Model, x: &[f32], nf: usize, constraints: &[i8], what: &str) {
        let grid: Vec<f32> = (0..61).map(|k| -2.2 + k as f32 * (4.4 / 60.0)).collect();
        for base in (0..x.len() / nf).step_by(7).take(40) {
            let mut row = x[base * nf..(base + 1) * nf].to_vec();
            for (f, &c) in constraints.iter().enumerate() {
                if c == 0 {
                    continue;
                }
                let original = row[f];
                let preds: Vec<f32> = grid
                    .iter()
                    .map(|&v| {
                        row[f] = v;
                        model.predict_margin(&row)
                    })
                    .collect();
                row[f] = original;
                for pair in preds.windows(2) {
                    let ok = if c > 0 {
                        pair[1] >= pair[0]
                    } else {
                        pair[1] <= pair[0]
                    };
                    assert!(
                        ok,
                        "{what}: feature {f} (c={c}) row {base} not monotone: {preds:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn monotone_unset_and_all_zero_are_byte_identical() {
        let (x, y, nf) = wiggly_data(300, 1);
        for obj in [Objective::Logistic, Objective::SquaredError] {
            let labels: Vec<f32> = match obj {
                Objective::Logistic => y.iter().map(|&v| f32::from(v > 0.0)).collect(),
                _ => y.clone(),
            };
            let p = params(5, 4);
            let src =
                DenseSource::from_row_major(&x, labels.len(), nf, p.max_bin, p.cols_per_block);
            let plain = train(
                &src,
                &labels,
                &[],
                &TrainConfig::new(p.clone(), obj),
                &mut [],
                &Local,
            )
            .unwrap();
            for zeros in [vec![0i8; nf], vec![0i8; 2]] {
                let mut q = p.clone();
                q.monotone_constraints = zeros;
                let m = train(
                    &src,
                    &labels,
                    &[],
                    &TrainConfig::new(q, obj),
                    &mut [],
                    &Local,
                )
                .unwrap();
                assert_eq!(
                    json(&plain),
                    json(&m),
                    "{obj:?}: all-zero constraints must equal no constraints"
                );
            }
        }
    }

    #[test]
    fn monotone_predictions_are_monotone_and_thread_invariant() {
        for (seed, constraints) in [
            (2u64, vec![1i8, -1, 0, 0, 0]),
            (3, vec![-1, 1, 1, 0, 0]),
            (4, vec![1, 1, -1, -1, 1]),
        ] {
            let (x, y, nf) = wiggly_data(600, seed);
            for obj in [Objective::Logistic, Objective::SquaredError] {
                let labels: Vec<f32> = match obj {
                    Objective::Logistic => y.iter().map(|&v| f32::from(v > 0.0)).collect(),
                    _ => y.clone(),
                };
                for (depth, subsample, colsample) in [(6, 1.0, 1.0), (8, 0.7, 0.8), (0, 1.0, 1.0)] {
                    let mut p = params(depth, 6);
                    p.subsample = subsample;
                    p.colsample_bytree = colsample;
                    p.seed = seed;
                    p.monotone_constraints = constraints.clone();
                    let src = DenseSource::from_row_major(
                        &x,
                        labels.len(),
                        nf,
                        p.max_bin,
                        p.cols_per_block,
                    );
                    let cfg = TrainConfig::new(p.clone(), obj);
                    let m = train(&src, &labels, &[], &cfg, &mut [], &Local).unwrap();
                    let what =
                        format!("seed={seed} {obj:?} depth={depth} ss={subsample} cs={colsample}");
                    assert_monotone(&m, &x, nf, &constraints, &what);
                    let mut q = p.clone();
                    q.nthread = 3;
                    q.cols_per_block = 1;
                    let s2 = DenseSource::from_row_major(&x, labels.len(), nf, q.max_bin, 1);
                    let m2 = train(
                        &s2,
                        &labels,
                        &[],
                        &TrainConfig::new(q, obj),
                        &mut [],
                        &Local,
                    )
                    .unwrap();
                    assert_eq!(
                        json(&m),
                        json(&m2),
                        "{what}: thread count / block width must not change the constrained model"
                    );
                }
            }
        }
    }

    #[test]
    fn unconstrained_wiggly_model_is_not_monotone() {
        // 反证夹具:不加约束时同样的检查必须失败,否则上面的测试什么也没证明。
        let (x, y, nf) = wiggly_data(600, 2);
        let p = params(6, 6);
        let src = DenseSource::from_row_major(&x, y.len(), nf, p.max_bin, p.cols_per_block);
        let m = train(
            &src,
            &y,
            &[],
            &TrainConfig::new(p, Objective::SquaredError),
            &mut [],
            &Local,
        )
        .unwrap();
        let caught = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            assert_monotone(&m, &x, nf, &[1, -1, 0, 0, 0], "unconstrained");
        }));
        assert!(
            caught.is_err(),
            "the fixture must violate monotonicity without constraints"
        );
    }

    #[test]
    fn rejects_too_many_monotone_constraints() {
        let (x, y, nf) = step_data(50);
        let mut p = params(2, 1);
        p.monotone_constraints = vec![1; nf + 1];
        let src = DenseSource::from_row_major(&x, y.len(), nf, p.max_bin, p.cols_per_block);
        let err = train(
            &src,
            &y,
            &[],
            &TrainConfig::new(p, Objective::Logistic),
            &mut [],
            &Local,
        )
        .unwrap_err();
        assert!(err.to_string().contains("monotone_constraints"), "{err}");
    }

    fn assert_host_leaf_updates_match(initial: Vec<f32>, owned_leaves: Vec<(Vec<RowId>, f32)>) {
        let leaves: Vec<(&[RowId], f32)> = owned_leaves
            .iter()
            .map(|(rows, value)| (rows.as_slice(), *value))
            .collect();
        for (leaf_idx, (rows, _)) in leaves.iter().enumerate() {
            assert!(
                rows.windows(2).all(|pair| pair[0] <= pair[1]),
                "test leaf {leaf_idx} must be sorted"
            );
            assert!(rows.iter().all(|&row| (row as usize) < initial.len()));
        }

        // This is the reference contract: preserve the original leaf-major,
        // row-minor sequence of f32 additions, including repeated row IDs.
        let mut expected = initial.clone();
        for (rows, value) in &leaves {
            for &row in *rows {
                expected[row as usize] += *value;
            }
        }

        for threads in [1, 2, 8] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            let mut actual = initial.clone();
            pool.install(|| apply_host_leaf_updates(&mut actual, &leaves));
            assert_eq!(actual.len(), expected.len());
            for (row, (actual, expected)) in actual.iter().zip(&expected).enumerate() {
                assert_eq!(
                    actual.to_bits(),
                    expected.to_bits(),
                    "prediction row {row} differs with a {threads}-thread pool"
                );
            }
        }
    }

    #[test]
    fn host_leaf_updates_match_serial_f32_order_at_small_sizes_and_chunk_edges() {
        for n_rows in [0usize, 1, 65_535, 65_536, 65_537] {
            let initial: Vec<f32> = (0..n_rows)
                .map(|row| match row % 7 {
                    0 => -0.0,
                    1 => f32::MAX,
                    2 => f32::MIN,
                    3 => 1.0e20,
                    4 => -1.0e20,
                    _ => row as f32 * 0.25,
                })
                .collect();
            let mut all_rows: Vec<RowId> = (0..n_rows as RowId).collect();
            let sparse: Vec<RowId> = (0..n_rows).step_by(97).map(|row| row as RowId).collect();
            let mut reverse_order_subset = sparse.clone();
            reverse_order_subset.sort_unstable();
            // Empty lists and duplicate IDs are valid inputs. Duplicates are
            // deliberate here so the oracle also checks repeated additions.
            let mut repeated = vec![];
            for &row in sparse.iter().take(4) {
                repeated.extend([row, row]);
            }
            repeated.sort_unstable();
            if n_rows > 0 {
                all_rows.extend([0, 0, (n_rows - 1) as RowId]);
                all_rows.sort_unstable();
            }
            let leaves = vec![
                (Vec::new(), 99.0),
                (all_rows, -0.0),
                (sparse, 3.0),
                (repeated, -1.0e20),
                (reverse_order_subset, 1.0e20),
                (Vec::new(), f32::MAX),
            ];
            assert_host_leaf_updates_match(initial, leaves);
        }
    }

    #[test]
    fn host_leaf_updates_preserve_order_for_large_interleaved_and_late_sparse_leaves() {
        let n_rows = 262_147usize;
        let initial: Vec<f32> = (0..n_rows)
            .map(|row| if row % 13 == 0 { 1.0e20 } else { -0.0 })
            .collect();

        // Eight balanced, sorted, interleaved row lists make the large update
        // enter the chunked path. Every row receives exactly one update here.
        let balanced: Vec<(Vec<RowId>, f32)> = (0..8usize)
            .map(|lane| {
                let rows = (lane..n_rows).step_by(8).map(|row| row as RowId).collect();
                (rows, lane as f32 + 0.5)
            })
            .collect();
        assert_host_leaf_updates_match(initial.clone(), balanced);

        // Exercise overlap and addition order after earlier leaves have
        // already changed the same rows. The final sparse leaf is intentionally
        // later in leaf order and uses non-associative values.
        let all_rows: Vec<RowId> = (0..n_rows as RowId).collect();
        let overlap: Vec<RowId> = (0..n_rows).step_by(3).map(|row| row as RowId).collect();
        let later_subset: Vec<RowId> = (0..n_rows).step_by(11).map(|row| row as RowId).collect();
        let mut leaves = vec![
            (all_rows, 1.0e20),
            (overlap.clone(), -1.0e20),
            (overlap, 3.0),
            (Vec::new(), 7.0),
            (later_subset, -1.0e20),
        ];
        // Add repeated occurrences and cross-leaf overlap while keeping the
        // affected lists sorted.
        leaves[2].0.extend([0, 0, 33, 33]);
        leaves[2].0.sort_unstable();
        assert_host_leaf_updates_match(initial, leaves);
    }

    #[test]
    fn host_leaf_updates_many_contiguous_leaves_match_serial_oracle() {
        let n_rows = 262_144usize;
        let initial: Vec<f32> = (0..n_rows)
            .map(|row| match row % 5 {
                0 => -0.0,
                1 => 1.0e20,
                2 => -1.0e20,
                _ => row as f32 * 0.125,
            })
            .collect();
        let leaves: Vec<(Vec<RowId>, f32)> = (0..4_096usize)
            .map(|leaf| {
                let start = leaf * 64;
                let rows = (start..start + 64).map(|row| row as RowId).collect();
                (rows, (leaf % 17) as f32 - 8.0)
            })
            .collect();
        assert_host_leaf_updates_match(initial, leaves);
    }

    #[test]
    fn host_leaf_updates_large_buffer_with_few_rows_preserves_untouched_bits() {
        let n_rows = 262_147usize;
        let initial: Vec<f32> = (0..n_rows)
            .map(|row| match row % 4 {
                0 => -0.0,
                1 => f32::MAX,
                2 => f32::MIN,
                _ => row as f32 * 0.25,
            })
            .collect();
        let leaves = vec![
            (vec![0, 65_535, 65_536, 262_146], 3.0),
            (vec![65_536, 262_146], -1.0e20),
        ];
        assert_host_leaf_updates_match(initial, leaves);
    }

    #[test]
    fn host_leaf_updates_reject_out_of_range_rows_with_serial_and_parallel_pools() {
        let n_rows = 131_072usize;
        let initial = vec![0.0f32; n_rows];
        let invalid_rows: Vec<RowId> = (0..=n_rows as RowId).collect();
        let leaves = [(invalid_rows.as_slice(), 1.0f32)];

        for threads in [1, 8] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            let mut predictions = initial.clone();
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                pool.install(|| apply_host_leaf_updates(&mut predictions, &leaves));
            }));
            assert!(
                result.is_err(),
                "out-of-range row must panic with a {threads}-thread pool"
            );
        }
    }

    // ---- base_margin / training continuation -------------------------------

    fn float_margins(model: &Model, x: &[f32], n_features: usize) -> Vec<f32> {
        x.chunks_exact(n_features)
            .map(|row| model.predict_margin(row))
            .collect()
    }

    fn trees_bits(model: &Model) -> Vec<(u32, u32, u32)> {
        model
            .trees
            .iter()
            .flat_map(|t| &t.nodes)
            .map(|n| (n.feat, n.split_cond.to_bits(), n.leaf_value.to_bits()))
            .collect()
    }

    /// A margin equal to the intercept's margin is the same start as no margin:
    /// bit-identical model.
    #[test]
    fn base_margin_equal_to_the_intercept_changes_nothing() {
        let (x, y, nf) = step_data(80);
        let p = params(3, 6);
        for obj in [Objective::SquaredError, Objective::Logistic] {
            let src = DenseSource::from_row_major(&x, 80, nf, p.max_bin, p.cols_per_block);
            let plain = train(
                &src,
                &y,
                &[],
                &TrainConfig::new(p.clone(), obj),
                &mut [],
                &Local,
            )
            .unwrap();
            let margin = base_margin(obj, plain.base_score);
            let cfg = TrainConfig {
                base_margin: Some(vec![margin; 80]),
                ..TrainConfig::new(p.clone(), obj)
            };
            let with = train(&src, &y, &[], &cfg, &mut [], &Local).unwrap();
            assert_eq!(trees_bits(&plain), trees_bits(&with), "{obj:?}");
            assert_eq!(plain.base_score.to_bits(), with.base_score.to_bits());
        }
    }

    /// The margin is the starting point: with label = margin + 1 everywhere the
    /// first leaf is the step towards +1, and the stored intercept is still the
    /// label estimate (XGBoost does not look at base_margin there).
    #[test]
    fn base_margin_replaces_the_start_but_not_the_estimated_intercept() {
        let n = 16usize;
        let x = vec![0.0f32; n];
        let margin: Vec<f32> = (0..n).map(|i| i as f32 * 0.5).collect();
        let y: Vec<f32> = margin.iter().map(|m| m + 1.0).collect();
        let mut p = params(0, 1);
        p.lambda = 0.0;
        p.learning_rate = 1.0;
        let src = DenseSource::from_row_major(&x, n, 1, p.max_bin, p.cols_per_block);
        let cfg = TrainConfig {
            base_margin: Some(margin.clone()),
            ..TrainConfig::new(p, Objective::SquaredError)
        };
        let model = train(&src, &y, &[], &cfg, &mut [], &Local).unwrap();
        let mean = y.iter().sum::<f32>() / n as f32;
        assert!(
            (model.base_score - mean).abs() < 1e-5,
            "{}",
            model.base_score
        );
        // gradient = pred - y = -1 for every row, hessian 1: leaf = +1.
        assert_eq!(model.trees[0].nodes[0].leaf_value, 1.0);
    }

    #[test]
    fn base_margin_shape_and_values_are_validated() {
        let (x, y, nf) = step_data(20);
        let p = params(2, 2);
        let src = DenseSource::from_row_major(&x, 20, nf, p.max_bin, p.cols_per_block);
        let bad = |margin: Vec<f32>| {
            let cfg = TrainConfig {
                base_margin: Some(margin),
                ..TrainConfig::new(p.clone(), Objective::SquaredError)
            };
            train(&src, &y, &[], &cfg, &mut [], &Local)
                .expect_err("must be rejected")
                .to_string()
        };
        assert!(bad(vec![0.0; 19]).contains("base_margin"));
        let mut nan = vec![0.0; 20];
        nan[3] = f32::NAN;
        assert!(bad(nan).contains("finite"));
        let eval = EvalSet {
            name: "v".into(),
            source: &src,
            labels: &y,
            weights: None,
            group_ptr: None,
        };
        let cfg = TrainConfig {
            eval_base_margins: vec![Some(vec![0.0; 3])],
            ..TrainConfig::new(p.clone(), Objective::SquaredError)
        };
        let err = train(&src, &y, &[eval], &cfg, &mut [], &Local).unwrap_err();
        assert!(err.to_string().contains("validation set"), "{err:#}");
    }

    /// Multiclass margins are class-major `K x n`: class `k` of row `i` starts at
    /// `margin[k * n + i]`.
    #[test]
    fn multiclass_base_margin_is_class_major() {
        let n = 30usize;
        let x: Vec<f32> = (0..n).map(|i| (i % 5) as f32).collect();
        let y: Vec<f32> = (0..n).map(|i| (i % 3) as f32).collect();
        let obj = Objective::Softprob { num_class: 3 };
        let p = params(2, 3);
        let src = DenseSource::from_row_major(&x, n, 1, p.max_bin, p.cols_per_block);
        let plain = train(
            &src,
            &y,
            &[],
            &TrainConfig::new(p.clone(), obj),
            &mut [],
            &Local,
        )
        .unwrap();
        // Starting every row at the class intercepts reproduces the plain model.
        let mut margin = Vec::new();
        for k in 0..3 {
            margin.extend(std::iter::repeat_n(plain.class_base_margins[k], n));
        }
        let cfg = TrainConfig {
            base_margin: Some(margin),
            ..TrainConfig::new(p.clone(), obj)
        };
        let with = train(&src, &y, &[], &cfg, &mut [], &Local).unwrap();
        assert_eq!(trees_bits(&plain), trees_bits(&with));
        // A genuinely different start changes the trees.
        let skew: Vec<f32> = (0..3 * n).map(|i| if i < n { 2.0 } else { 0.0 }).collect();
        let cfg = TrainConfig {
            base_margin: Some(skew),
            ..TrainConfig::new(p, obj)
        };
        let skewed = train(&src, &y, &[], &cfg, &mut [], &Local).unwrap();
        assert_ne!(trees_bits(&plain), trees_bits(&skewed));
        assert_eq!(skewed.class_base_margins, plain.class_base_margins);
    }

    #[allow(clippy::too_many_arguments)]
    fn continue_from(
        model: Model,
        src: &DenseSource,
        x: &[f32],
        y: &[f32],
        n_features: usize,
        p: TrainParams,
        obj: Objective,
    ) -> anyhow::Result<Model> {
        let k = obj.num_outputs();
        let rows = y.len();
        let mut margins = Vec::with_capacity(rows * k);
        for row in x.chunks_exact(n_features) {
            margins.extend(model.predict_margins_upto(row, model.trees.len()));
        }
        let cfg = TrainConfig {
            base_margin: Some(class_major_from_rows(&margins, rows, k)),
            prior_model: Some(model),
            ..TrainConfig::new(p, obj)
        };
        train(src, y, &[], &cfg, &mut [], &Local)
    }

    /// On the same data (same cuts), N rounds + M continued rounds is exactly N+M
    /// rounds, including row / column sampling (seeds key on the absolute round).
    #[test]
    fn continuation_on_the_same_data_equals_uninterrupted_training() {
        let (x, y, nf) = step_data(120);
        for obj in [Objective::SquaredError, Objective::Logistic] {
            let mut full_p = params(3, 8);
            full_p.subsample = 0.8;
            full_p.colsample_bytree = 0.75;
            let src =
                DenseSource::from_row_major(&x, 120, nf, full_p.max_bin, full_p.cols_per_block);
            let full = train(
                &src,
                &y,
                &[],
                &TrainConfig::new(full_p.clone(), obj),
                &mut [],
                &Local,
            )
            .unwrap();
            let first_p = TrainParams {
                n_rounds: 3,
                ..full_p.clone()
            };
            let first = train(
                &src,
                &y,
                &[],
                &TrainConfig::new(first_p, obj),
                &mut [],
                &Local,
            )
            .unwrap();
            let tail_p = TrainParams {
                n_rounds: 5,
                ..full_p
            };
            let joined = continue_from(first, &src, &x, &y, nf, tail_p, obj).unwrap();
            assert_eq!(joined.trees.len(), 8);
            assert_eq!(trees_bits(&full), trees_bits(&joined), "{obj:?}");
            assert_eq!(full.base_score.to_bits(), joined.base_score.to_bits());
        }
    }

    #[test]
    fn multiclass_continuation_equals_uninterrupted_training() {
        let n = 90usize;
        let x: Vec<f32> = (0..n).map(|i| ((i * 7) % 11) as f32).collect();
        let y: Vec<f32> = (0..n).map(|i| ((i * 7) % 11 / 4) as f32).collect();
        let obj = Objective::Softprob { num_class: 3 };
        let full_p = params(2, 6);
        let src = DenseSource::from_row_major(&x, n, 1, full_p.max_bin, full_p.cols_per_block);
        let full = train(
            &src,
            &y,
            &[],
            &TrainConfig::new(full_p.clone(), obj),
            &mut [],
            &Local,
        )
        .unwrap();
        let first_p = TrainParams {
            n_rounds: 2,
            ..full_p.clone()
        };
        let first = train(
            &src,
            &y,
            &[],
            &TrainConfig::new(first_p, obj),
            &mut [],
            &Local,
        )
        .unwrap();
        let tail_p = TrainParams {
            n_rounds: 4,
            ..full_p
        };
        let joined = continue_from(first, &src, &x, &y, 1, tail_p, obj).unwrap();
        assert_eq!(joined.trees.len(), 18);
        assert_eq!(trees_bits(&full), trees_bits(&joined));
        assert_eq!(full.class_base_margins, joined.class_base_margins);
    }

    #[test]
    fn continuation_rejects_incompatible_models_and_missing_margins() {
        let (x, y, nf) = step_data(40);
        let p = params(2, 2);
        let src = DenseSource::from_row_major(&x, 40, nf, p.max_bin, p.cols_per_block);
        let prior = train(
            &src,
            &y,
            &[],
            &TrainConfig::new(p.clone(), Objective::SquaredError),
            &mut [],
            &Local,
        )
        .unwrap();
        // A model's intercept is not up for change.
        let cfg = TrainConfig {
            prior_model: Some(prior.clone()),
            base_margin: Some(float_margins(&prior, &x, nf)),
            base_score: Some(prior.base_score + 1.0),
            ..TrainConfig::new(p.clone(), Objective::SquaredError)
        };
        let err = train(&src, &y, &[], &cfg, &mut [], &Local).unwrap_err();
        assert!(err.to_string().contains("base_score"), "{err:#}");
        // The prior's predictions must come with it.
        let cfg = TrainConfig {
            prior_model: Some(prior.clone()),
            ..TrainConfig::new(p.clone(), Objective::SquaredError)
        };
        let err = train(&src, &y, &[], &cfg, &mut [], &Local).unwrap_err();
        assert!(err.to_string().contains("base_margin"), "{err:#}");
        // Different objective.
        let cfg = TrainConfig {
            prior_model: Some(prior.clone()),
            base_margin: Some(float_margins(&prior, &x, nf)),
            ..TrainConfig::new(p.clone(), Objective::Logistic)
        };
        let err = train(&src, &y, &[], &cfg, &mut [], &Local).unwrap_err();
        assert!(err.to_string().contains("objective"), "{err:#}");
        // Different feature count.
        let narrow = DenseSource::from_row_major(&x[..], 80, nf / 2, p.max_bin, p.cols_per_block);
        let y2: Vec<f32> = (0..80).map(|i| (i % 2) as f32).collect();
        let cfg = TrainConfig {
            prior_model: Some(prior.clone()),
            base_margin: Some(vec![0.0; 80]),
            ..TrainConfig::new(p, Objective::SquaredError)
        };
        let err = train(&narrow, &y2, &[], &cfg, &mut [], &Local).unwrap_err();
        assert!(err.to_string().contains("features"), "{err:#}");
    }

    /// A multi-quantile model continues like a multiclass one (K margins per row, the
    /// per-alpha intercepts kept), and only with the same quantiles.
    #[test]
    fn multi_quantile_continuation_equals_uninterrupted_training_and_needs_same_alphas() {
        let (x, y, nf) = step_data(120);
        let obj = Objective::quantile(0.5)
            .with_quantile_alphas(&[0.2, 0.5, 0.8])
            .unwrap();
        let mut full_p = params(3, 6);
        full_p.subsample = 0.8;
        let src = DenseSource::from_row_major(&x, 120, nf, full_p.max_bin, full_p.cols_per_block);
        let train_rounds = |rounds: usize| {
            let p = TrainParams {
                n_rounds: rounds,
                ..full_p.clone()
            };
            train(&src, &y, &[], &TrainConfig::new(p, obj), &mut [], &Local).unwrap()
        };
        let full = train_rounds(6);
        let first = train_rounds(2);
        let tail_p = TrainParams {
            n_rounds: 4,
            ..full_p.clone()
        };
        let joined = continue_from(first.clone(), &src, &x, &y, nf, tail_p.clone(), obj).unwrap();
        assert_eq!(joined.trees.len(), 18);
        assert_eq!(trees_bits(&full), trees_bits(&joined));
        assert_eq!(full.class_base_margins, joined.class_base_margins);
        // Other quantiles (same count or not) are refused.
        for alphas in [&[0.1f32, 0.5, 0.9][..], &[0.2, 0.8][..]] {
            let other = Objective::quantile(0.5)
                .with_quantile_alphas(alphas)
                .unwrap();
            let cfg = TrainConfig {
                base_margin: Some(vec![0.0; 120 * alphas.len()]),
                prior_model: Some(first.clone()),
                ..TrainConfig::new(tail_p.clone(), other)
            };
            let err = train(&src, &y, &[], &cfg, &mut [], &Local).unwrap_err();
            assert!(err.to_string().contains("quantile_alpha"), "{err:#}");
        }
    }

    /// Early stopping while continuing reports the absolute round in the whole
    /// model, and evaluation starts from the prior model's predictions.
    #[test]
    fn continuation_best_iteration_is_absolute() {
        let (x, y, nf) = step_data(100);
        let p = params(2, 2);
        let src = DenseSource::from_row_major(&x, 100, nf, p.max_bin, p.cols_per_block);
        let prior = train(
            &src,
            &y,
            &[],
            &TrainConfig::new(p.clone(), Objective::Logistic),
            &mut [],
            &Local,
        )
        .unwrap();
        let margins = float_margins(&prior, &x, nf);
        let eval = EvalSet {
            name: "valid".into(),
            source: &src,
            labels: &y,
            weights: None,
            group_ptr: None,
        };
        let cfg = TrainConfig {
            base_margin: Some(margins.clone()),
            eval_base_margins: vec![Some(margins)],
            prior_model: Some(prior.clone()),
            ..TrainConfig::new(TrainParams { n_rounds: 6, ..p }, Objective::Logistic)
        };
        let mut cbs: Vec<Box<dyn Callback>> = vec![Box::new(EarlyStopping::new(2, true))];
        let model = train(&src, &y, &[eval], &cfg, &mut cbs, &Local).unwrap();
        let best = model
            .best_iteration
            .expect("early stopping records the best round");
        assert!(best < model.trees.len());
        assert!(model.trees.len() > 2);
        // The best round is a round of the *whole* model: it is at least the
        // prior's last round whenever any new round improved on it.
        assert!(best >= 1, "absolute round in the whole model, got {best}");
    }
}
