//! open 阶段的分段计时。默认关闭,`FB_OPEN_PROFILE=1` 打开。
//!
//! 和 `train::prof` 分开,因为关心的问题不同:训练那边要「每轮分布」,
//! 这边要「一次性的 open 里那 180 多秒花在哪一段」。打点只在 open 路径上,
//! `quantize_block` 也只被 `build_cache` 调用,所以训练期间不会有点被触发。
use std::sync::Mutex;
use std::time::Instant;

/// `/proc/self/stat` 的 utime/stime 单位是时钟节拍。Linux 上
/// `getconf CLK_TCK` 恒为 100,而仓库没有 libc 依赖、调不到 sysconf,
/// 所以写死。**自检方式**:已知串行的段(metadata / label / cuts)
/// cpu/wall 应该落在 1.0 附近,明显偏离就说明这个假设不成立,
/// 报告里的 cpu 列不能用。
const CLK_TCK: f64 = 100.0;

#[derive(Default)]
struct Bucket {
    wall_ns: u128,
    cpu_ticks: u64,
    rchar: u64,
    calls: usize,
    vmrss_kb: u64,
    vmhwm_kb: u64,
}

static BUCKETS: Mutex<Vec<(&'static str, Bucket)>> = Mutex::new(Vec::new());
static SHAPE: Mutex<Option<(usize, usize, usize)>> = Mutex::new(None);

pub fn enabled() -> bool {
    // 缓存住:细分计时每列都要问一次,10M × 300 是上百万次,
    // 而 env::var_os 每次都分配一个 OsString。
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("FB_OPEN_PROFILE").is_ok_and(|v| v == "1" || v == "true"))
}

fn read_kv(path: &str, key: &str) -> u64 {
    let Ok(text) = std::fs::read_to_string(path) else { return 0 };
    for line in text.lines() {
        if let Some(rest) = line.strip_prefix(key) {
            return rest.split_whitespace().next().and_then(|t| t.parse().ok()).unwrap_or(0);
        }
    }
    0
}

/// 整个进程(含所有线程)的 CPU 节拍。
fn cpu_ticks() -> u64 {
    let Ok(text) = std::fs::read_to_string("/proc/self/stat") else { return 0 };
    // comm 字段可能含空格和括号,所以从最后一个 ')' 之后开始切。
    let Some((_, rest)) = text.rsplit_once(')') else { return 0 };
    let fields: Vec<&str> = rest.split_whitespace().collect();
    // rest 的第 0 项是 state(原第 3 字段),所以 utime/stime 是 11/12。
    let get = |i: usize| fields.get(i).and_then(|v| v.parse::<u64>().ok()).unwrap_or(0);
    get(11) + get(12)
}

pub struct Guard {
    name: &'static str,
    wall: Instant,
    cpu0: u64,
    rchar0: u64,
}

/// 开一段计时。返回 `None` 表示没开 profiling,调用点几乎零开销。
pub fn start(name: &'static str) -> Option<Guard> {
    if !enabled() {
        return None;
    }
    Some(Guard {
        name,
        wall: Instant::now(),
        cpu0: cpu_ticks(),
        rchar0: read_kv("/proc/self/io", "rchar:"),
    })
}

impl Drop for Guard {
    fn drop(&mut self) {
        let wall = self.wall.elapsed().as_nanos();
        let cpu = cpu_ticks().saturating_sub(self.cpu0);
        let rchar = read_kv("/proc/self/io", "rchar:").saturating_sub(self.rchar0);
        // VmHWM 是单调的,所以这里记的是「本段结束时的全局高水位」,
        // **不是本段自己的增量**。报告要按这个口径读。
        let vmrss = read_kv("/proc/self/status", "VmRSS:");
        let vmhwm = read_kv("/proc/self/status", "VmHWM:");
        let mut buckets = BUCKETS.lock().unwrap();
        if !buckets.iter().any(|(n, _)| *n == self.name) {
            buckets.push((self.name, Bucket::default()));
        }
        let slot = &mut buckets.iter_mut().find(|(n, _)| *n == self.name).unwrap().1;
        slot.wall_ns += wall;
        slot.cpu_ticks += cpu;
        slot.rchar += rchar;
        slot.calls += 1;
        slot.vmrss_kb = vmrss;
        slot.vmhwm_kb = vmhwm;
    }
}

/// 记下数据形状,报告里带上,免得回头分不清是哪次跑的。
pub fn shape(n_rows: usize, n_features: usize, n_blocks: usize) {
    if enabled() {
        *SHAPE.lock().unwrap() = Some((n_rows, n_features, n_blocks));
    }
}

/// 细分计时:**只测墙钟**,不采 CPU / 逻辑读 / RSS。
///
/// 内层循环(每 batch 每列一次)在 10M × 300 上要跑上百万次,
/// 而 coarse 段那套要读三个 `/proc` 文件 —— 采样开销会盖过被测对象。
/// 所以细分段只留 `Instant`,报告分成两张表。
static SUB: Mutex<Vec<(&'static str, u128, usize)>> = Mutex::new(Vec::new());

pub struct Sub {
    name: &'static str,
    wall: Instant,
}

pub fn sub(name: &'static str) -> Option<Sub> {
    if !enabled() {
        return None;
    }
    Some(Sub { name, wall: Instant::now() })
}

impl Drop for Sub {
    fn drop(&mut self) {
        let ns = self.wall.elapsed().as_nanos();
        let mut subs = SUB.lock().unwrap();
        match subs.iter_mut().find(|(n, _, _)| *n == self.name) {
            Some(slot) => {
                slot.1 += ns;
                slot.2 += 1;
            }
            None => subs.push((self.name, ns, 1)),
        }
    }
}

pub fn report() {
    if !enabled() {
        return;
    }
    let buckets = BUCKETS.lock().unwrap();
    let total: u128 = buckets.iter().map(|(_, b)| b.wall_ns).sum();
    if total == 0 {
        return;
    }
    match *SHAPE.lock().unwrap() {
        Some((rows, feats, blocks)) => {
            eprintln!("\n=== open 分段({rows} 行 × {feats} 列,{blocks} 块)===")
        }
        None => eprintln!("\n=== open 分段 ==="),
    }
    eprintln!(
        "{:<12}{:>9}{:>7}{:>10}{:>11}{:>12}{:>11}{:>6}",
        "段", "wall", "占比", "cpu/wall", "逻辑读", "VmHWM@末", "VmRSS@末", "次数"
    );
    for (name, b) in buckets.iter() {
        let wall_s = b.wall_ns as f64 / 1e9;
        eprintln!(
            "{name:<12}{wall_s:>8.1}s{:>6.0}%{:>10.2}{:>10.2}G{:>11.0}M{:>10.0}M{:>6}",
            b.wall_ns as f64 / total as f64 * 100.0,
            (b.cpu_ticks as f64 / CLK_TCK) / wall_s.max(1e-9),
            b.rchar as f64 / 1e9,
            b.vmhwm_kb as f64 / 1024.0,
            b.vmrss_kb as f64 / 1024.0,
            b.calls,
        );
    }
    eprintln!("{:<12}{:>8.1}s", "合计", total as f64 / 1e9);

    let subs = SUB.lock().unwrap();
    if subs.is_empty() {
        return;
    }
    let sub_total: u128 = subs.iter().map(|(_, ns, _)| ns).sum();
    eprintln!("\n--- 细分(只测墙钟)---");
    eprintln!("{:<14}{:>9}{:>7}{:>14}", "段", "wall", "占比", "次数");
    for (name, ns, calls) in subs.iter() {
        eprintln!("{name:<14}{:>8.1}s{:>6.0}%{:>14}",
            *ns as f64 / 1e9, *ns as f64 / sub_total as f64 * 100.0, calls);
    }
}
