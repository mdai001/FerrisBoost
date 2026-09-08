//! 所有 CPU 阶段共用的 Rayon 线程池构造,以及 `nthread=0` 的解析。
//!
//! `nthread=0` 的语义是「自己选一个合理的线程数」。**它解析成物理核数,
//! 不是逻辑 CPU 数** —— 实测在 8 核 16 线程的 5800H 上,16 个逻辑线程比
//! 8 个物理核**慢 7–9%**(wide 1M×300 563.7 → 525.5 ms、
//! HIGGS 897.4 → 826.9 ms),SMT 兄弟线程共享执行单元,对这种
//! 访存密集的直方图累加帮不上忙,只是多了调度和争用。
//!
//! **线程数选择和块宽选择是两件事,不要合成一个 heuristic。**
//! 这里只负责回答「起几个线程」;`columns::auto_cols_per_block` 拿到
//! **正确的**线程数之后会自动跟着对。

/// `nthread` 解析成实际要起的线程数。
///
/// **用户显式给了 >0 就原样返回**,自动探测绝不覆盖。
pub(crate) fn effective_threads(nthread: usize) -> usize {
    resolve(nthread, physical_cores(), fallback_parallelism())
}

/// Static I/O worker resolver shared by ingest and model file I/O.
///
/// `requested == 0` follows the process-visible physical-CPU policy above;
/// positive values bypass that CPU policy, but both modes remain bounded by
/// independent work and a conservative quarter of currently available memory.
pub(crate) fn bounded_io_workers(
    requested: usize,
    work_units: usize,
    bytes_per_worker: usize,
) -> usize {
    let cpu_limit = if requested == 0 {
        effective_threads(0)
    } else {
        requested
    };
    let memory_limit = available_memory_bytes()
        .map(|bytes| (bytes / 4) / bytes_per_worker.max(1))
        .unwrap_or(usize::MAX)
        .max(1);
    cpu_limit.max(1).min(work_units.max(1)).min(memory_limit)
}

/// Best-effort process-visible memory headroom. Linux uses MemAvailable and,
/// when a real cgroup limit exists, also clamps to limit minus current usage.
pub(crate) fn available_memory_bytes() -> Option<usize> {
    let host = std::fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|text| {
            text.lines().find_map(|line| {
                let value = line
                    .strip_prefix("MemAvailable:")?
                    .split_whitespace()
                    .next()?
                    .parse::<u64>()
                    .ok()?;
                usize::try_from(value.saturating_mul(1024)).ok()
            })
        });
    let cgroup = cgroup_memory_headroom();
    match (host, cgroup) {
        (Some(a), Some(b)) => Some(a.min(b)),
        (Some(a), None) => Some(a),
        (None, Some(b)) => Some(b),
        (None, None) => None,
    }
}

fn cgroup_memory_headroom() -> Option<usize> {
    fn pair(base: &std::path::Path, limit_name: &str, current_name: &str) -> Option<usize> {
        let limit = std::fs::read_to_string(base.join(limit_name)).ok()?;
        let limit = limit.trim();
        if limit == "max" {
            return None;
        }
        let limit = limit.parse::<u64>().ok()?;
        // cgroup v1 commonly uses a near-i64::MAX sentinel for "unlimited".
        if limit >= (i64::MAX as u64) / 2 {
            return None;
        }
        let used = std::fs::read_to_string(base.join(current_name))
            .ok()?
            .trim()
            .parse::<u64>()
            .ok()?;
        usize::try_from(limit.saturating_sub(used)).ok()
    }
    let membership = std::fs::read_to_string("/proc/self/cgroup").unwrap_or_default();
    let mut candidates = Vec::new();
    for line in membership.lines() {
        let mut fields = line.splitn(3, ':');
        let _hierarchy = fields.next();
        let controllers = fields.next().unwrap_or_default();
        let relative = fields.next().unwrap_or("/").trim_start_matches('/');
        if controllers.is_empty() {
            candidates.push((
                std::path::PathBuf::from("/sys/fs/cgroup").join(relative),
                "memory.max",
                "memory.current",
            ));
            candidates.push((
                std::path::PathBuf::from("/sys/fs/cgroup/unified").join(relative),
                "memory.max",
                "memory.current",
            ));
        } else if controllers.split(',').any(|c| c == "memory") {
            candidates.push((
                std::path::PathBuf::from("/sys/fs/cgroup/memory").join(relative),
                "memory.limit_in_bytes",
                "memory.usage_in_bytes",
            ));
        }
    }
    candidates
        .into_iter()
        .find_map(|(base, limit, current)| pair(&base, limit, current))
}

/// 纯函数形式,方便直接测:不读 sysfs,也不碰真实机器。
///
/// ⚠️ **物理核数必须夹到进程"真正能调度到"的并行度上。**
/// `physical_cores()` 读的是 sysfs / procfs 里的**机器**拓扑,它不知道
/// cgroup CPU quota,也不知道 `sched_setaffinity`;而
/// `available_parallelism()` 两者都认。容器里限 2 CPU、或者
/// `taskset -c 0-3` 起来的进程,机器上仍然是 8 个物理核 ——
/// 不夹的话 `nthread=0` 会起 8 个 rayon worker 去抢 4 条能跑的核,
/// **多出来的 worker 只会制造调度争用**。
///
/// 夹完仍然保持既定策略:**优先物理核,不是逻辑 CPU**
/// (本机 8 物理 / 16 逻辑,实测 16 线程比 8 线程慢 7–9%)。
fn resolve(nthread: usize, physical: Option<usize>, available: usize) -> usize {
    if nthread > 0 {
        return nthread;
    }
    let available = available.max(1);
    physical
        .filter(|&p| p > 0)
        .map_or(available, |p| p.min(available))
        .max(1)
}

fn fallback_parallelism() -> usize {
    std::thread::available_parallelism().map_or(1, |n| n.get())
}

/// 数物理核。两条 Linux 路径都只读 sysfs / procfs,**不引新依赖**;
/// 非 Linux 或读不到就返回 None,由调用方退回逻辑 CPU 数。
fn physical_cores() -> Option<usize> {
    sibling_groups()
        .or_else(cpuinfo_core_ids)
        .filter(|&n| n > 0)
}

/// 首选:`/sys/.../topology/thread_siblings_list`。同一个物理核上的
/// 兄弟线程共享同一份列表,所以**去重之后的组数**就是物理核数。
fn sibling_groups() -> Option<usize> {
    let mut groups = std::collections::HashSet::new();
    for entry in std::fs::read_dir("/sys/devices/system/cpu").ok()? {
        let path = entry.ok()?.path();
        let name = path.file_name()?.to_str()?.to_owned();
        // 只要 cpu0 / cpu1 …,跳过 cpuidle / cpufreq 这类同前缀目录。
        if !name.starts_with("cpu") || !name[3..].chars().all(|c| c.is_ascii_digit()) {
            continue;
        }
        if name.len() == 3 {
            continue;
        }
        let list = std::fs::read_to_string(path.join("topology/thread_siblings_list")).ok()?;
        groups.insert(list.trim().to_owned());
    }
    (!groups.is_empty()).then(|| groups.len())
}

/// 兜底:`/proc/cpuinfo` 里按 `(physical id, core id)` 去重。
/// WSL / 容器里 sysfs 的 topology 有时是缺的。
fn cpuinfo_core_ids() -> Option<usize> {
    let text = std::fs::read_to_string("/proc/cpuinfo").ok()?;
    let mut seen = std::collections::HashSet::new();
    let (mut pkg, mut core) = (None, None);
    for line in text.lines() {
        let Some((k, v)) = line.split_once(':') else {
            // 空行分隔两个 processor 块。
            if line.trim().is_empty() {
                if let (Some(p), Some(c)) = (pkg.take(), core.take()) {
                    seen.insert((p, c));
                }
            }
            continue;
        };
        match k.trim() {
            "physical id" => pkg = v.trim().parse::<i64>().ok(),
            "core id" => core = v.trim().parse::<i64>().ok(),
            _ => {}
        }
    }
    if let (Some(p), Some(c)) = (pkg, core) {
        seen.insert((p, c));
    }
    (!seen.is_empty()).then(|| seen.len())
}

pub(crate) fn build_pool(nthread: usize) -> anyhow::Result<rayon::ThreadPool> {
    let resolved = effective_threads(nthread);
    // ⚠️ **回显解析后的线程数,而不是用户传进来的那个。** `nthread=0` 解析成
    // 物理核数并夹到进程实际能调度到的并行度上,所以「传 0」和「真的起了几个
    // worker」可以差一倍(本机 8 物理 / 16 逻辑)。跨实现比较时,两边各自
    // 写「默认/全部线程」而实际数字不同,就是一次不公平的对照 ——
    // 这条日志正是为了让那种情况能被看见。
    //
    // logging 只负责观察:打不打印都不改变解析结果(见 CLAUDE.md 的
    // 「logging 只负责观察,不能控制执行」)。
    if crate::train::prof::enabled() {
        eprintln!("CPU_THREADS requested={nthread} resolved={resolved}");
    }
    rayon::ThreadPoolBuilder::new()
        // 0 会让 Rayon 自己按逻辑 CPU 数来,所以这里先解析成物理核数。
        .num_threads(resolved)
        .build()
        .map_err(anyhow::Error::from)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_thread_count_is_never_overridden() {
        for n in [1, 2, 3, 8, 64, 1000] {
            assert_eq!(effective_threads(n), n);
        }
    }

    #[test]
    fn auto_resolves_to_something_sane() {
        let auto = effective_threads(0);
        assert!(auto >= 1, "自动解析不能返回 0");
        // 物理核不会比逻辑 CPU 多;探测失败时退回逻辑数,所以取等也合法。
        assert!(
            auto <= fallback_parallelism(),
            "物理核 {auto} 不该超过逻辑 CPU {}",
            fallback_parallelism()
        );
    }

    /// 探测到的话,物理核数应当整除逻辑 CPU 数(SMT 每核的线程数一致)。
    /// 这条在没有 SMT 的机器上退化成相等,同样成立。
    #[test]
    fn physical_cores_divide_logical_cpus_when_detected() {
        if let Some(phys) = physical_cores() {
            let logical = fallback_parallelism();
            assert!(phys <= logical);
            assert_eq!(logical % phys, 0, "逻辑 {logical} / 物理 {phys} 不整除");
        }
    }

    /// 显式 `nthread > 0` 永远原样返回 —— 自动探测不得覆盖用户的设置,
    /// 即使它比机器能提供的还多(那是用户的选择,也是可复现基线的前提)。
    #[test]
    fn resolver_never_overrides_an_explicit_thread_count() {
        assert_eq!(resolve(1, Some(8), 16), 1);
        assert_eq!(resolve(3, Some(8), 16), 3);
        assert_eq!(resolve(64, Some(8), 16), 64, "显式值高于机器能力也照给");
    }

    /// 默认策略不变:优先物理核,不是逻辑 CPU。
    #[test]
    fn auto_prefers_physical_cores_over_logical() {
        assert_eq!(resolve(0, Some(8), 16), 8, "8 物理 / 16 逻辑要选 8");
    }

    /// ⚠️ 这条是本次修的 bug:cgroup quota / taskset 把进程限死之后,
    /// sysfs 仍然报整机的物理核数,不夹就会起过多 worker。
    #[test]
    fn auto_is_clamped_to_schedulable_parallelism() {
        assert_eq!(
            resolve(0, Some(8), 4),
            4,
            "taskset 限 4 核时不能起 8 个 worker"
        );
        assert_eq!(resolve(0, Some(16), 2), 2, "容器限 2 CPU 时不能起 16 个");
    }

    /// 读不到拓扑(非 Linux / WSL / 容器里 sysfs 缺失)就退回可用并行度。
    #[test]
    fn missing_topology_falls_back_to_available_parallelism() {
        assert_eq!(resolve(0, None, 12), 12);
        assert_eq!(resolve(0, Some(0), 12), 12, "探测出 0 视同探测失败");
    }

    /// 永远至少一个线程 —— 0 个 worker 的线程池跑不了任何东西。
    #[test]
    fn never_resolves_to_zero_threads() {
        assert_eq!(resolve(0, None, 0), 1);
        assert_eq!(resolve(0, Some(0), 0), 1);
    }

    #[test]
    fn io_workers_are_bounded_by_work_even_for_explicit_requests() {
        assert_eq!(bounded_io_workers(64, 3, 1), 3);
        assert_eq!(bounded_io_workers(1, 100, usize::MAX), 1);
    }
}
