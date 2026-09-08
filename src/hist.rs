//! 直方图构建 —— 热点所在。
//!
//! 这个模块的函数签名刻意写成 FFI 的形状(裸切片、扁平数组、显式长度),
//! 这样阶段 3 的 CUDA kernel 满足同一契约,可以直接对拍:
//! 同样输入喂给 CPU 和 GPU 两个实现,断言输出一致。

use rayon::prelude::*;

use crate::types::{Bin, GradPairFixed, RowId, MISSING_BIN};

/// 一个节点在一个列块上的直方图。
///
/// 布局:`bins[hist_offset(feat) + bin]`,按特征拼接。
pub struct Histogram {
    pub bins: Vec<GradPairFixed>,
}

impl Histogram {
    pub fn zeros(total_bins: usize) -> Self {
        Self { bins: vec![GradPairFixed::default(); total_bins] }
    }

    pub fn len(&self) -> usize {
        self.bins.len()
    }

    pub fn is_empty(&self) -> bool {
        self.bins.is_empty()
    }
}

/// 核心累加。签名对应将来的 C 接口:
///
/// ```c
/// void hist_build(const uint8_t* bins, const int64_t* gpair,
///                 const uint32_t* row_idx, uint32_t n_rows,
///                 uint32_t n_feats, const uint32_t* feat_offsets,
///                 int64_t* out_hist);
/// ```
///
/// gpair 和 out_hist 是**定点整数**(见 types.rs::GradPairFixed),
/// 不是 float。GPU kernel 照这个契约写,就能和 CPU 逐位对拍。
///
/// `row_idx` 是当前节点包含的行下标。为 None 时表示全部行(根节点)。
///
/// `feat_offsets` 是**块内局部**偏移(见 `BinCuts::block_hist_offsets`),
/// 长度 n_feats + 1,`out.len() == feat_offsets[n_feats]`。
///
/// ⚠️ **`out` 必须由调用方交进来时就是清零的**,本函数只累加,不再清一遍。
///
/// 契约放在调用方是因为**调用方本来就在清**:直方图的来源只有
/// `Histogram::zeros()`(以及测试里的 `vec![default; n]`),那已经是一趟
/// 全零写。本函数以前再清一次,等于同一段内存**连写两遍零**,
/// 而第二遍永远写在刚写完的零上面。
///
/// 这么定契约而不是加一个 `clear: bool` 参数:所有权本来就在调用方手里,
/// 多一个开关只会让"谁负责清"这件事变成两处都要看。
/// debug 构建里有断言兜底,违约会当场炸而不是悄悄算错。
///
/// 缺失值(bin == MISSING_BIN)整个跳过,不进任何 bin。缺失的梯度和
/// 由 split.rs 用「节点总和 - 所有 bin 之和」反推 —— 这样这里不用
/// 为它留槽位,直方图宽度就是真实 bin 数。
pub fn build(
    block_data: &[Bin],
    block_n_rows: usize,
    n_feats: usize,
    feat_offsets: &[u32],
    gpair: &[GradPairFixed],
    row_idx: Option<&[RowId]>,
    out: &mut [GradPairFixed],
) {
    debug_assert_eq!(block_data.len(), n_feats * block_n_rows);
    debug_assert_eq!(feat_offsets.len(), n_feats + 1);
    debug_assert_eq!(out.len(), feat_offsets[n_feats] as usize);
    debug_assert_eq!(gpair.len(), block_n_rows);
    // 契约:调用方交进来时就是清零的。违约会静默多算一份上一轮的直方图,
    // 所以在 debug 里直接断言,不留给"结果看起来怪怪的"去发现。
    debug_assert!(
        out.iter().all(|g| *g == GradPairFixed::default()),
        "hist::build 要求 out 进来就是清零的(见函数文档)"
    );

    // 每「行 × 特征」一次累加。一次性加,不进内层循环。
    if crate::train::prof::enabled() {
        let rows = row_idx.map_or(block_n_rows, |r| r.len());
        crate::train::prof::HIST_UPDATES
            .fetch_add((rows * n_feats) as u64, std::sync::atomic::Ordering::Relaxed);
    }

    // ---- 外层**行分块**,块内再「外层特征、内层行」----
    //
    // 列主序让 bin 的内层访存是顺序的(这是列主序的初衷),但外层特征
    // 意味着**梯度数组每个特征都要从头流一遍**:HIGGS 上 1050 万行 ×
    // 16 字节 = 168MB,乘 28 个特征 = 每层 4.7GB。实测每次累加 3.51 ns,
    // 而同样的代码在梯度塞得进 L3 的小数据上只要 1.40 ns —— 差的 2.5×
    // 全是在重复搬同一份数据。
    //
    // 按行分块之后,一个 tile 的梯度(TILE × 16 字节)在 L2 里被 28 个
    // 特征复用,重复流量从 28 遍降到 1 遍。**不改布局、不改分块结构**,
    // 纯循环变换。
    //
    // 定点累加与顺序无关,所以这个变换**不可能改变任何数值** ——
    // 位精确基线是这条性质的守门人。
    let tile = tile_rows();
    match row_idx {
        None => {
            let mut start = 0usize;
            while start < block_n_rows {
                let end = (start + tile).min(block_n_rows);
                for f in 0..n_feats {
                    let col = &block_data[f * block_n_rows..(f + 1) * block_n_rows];
                    let hist = &mut out[feat_offsets[f] as usize..feat_offsets[f + 1] as usize];
                    for row in start..end {
                        let b = col[row];
                        if b == MISSING_BIN {
                            continue;
                        }
                        hist[b as usize] += gpair[row];
                    }
                }
                start = end;
            }
        }
        Some(rows) => {
            // 行下标是升序的(来自上一层的分区),所以一个 tile 里的
            // 梯度访问也集中在一段地址里。
            for chunk in rows.chunks(tile) {
                for f in 0..n_feats {
                    let col = &block_data[f * block_n_rows..(f + 1) * block_n_rows];
                    let hist = &mut out[feat_offsets[f] as usize..feat_offsets[f + 1] as usize];
                    for &r in chunk {
                        let b = col[r as usize];
                        if b == MISSING_BIN {
                            continue;
                        }
                        hist[b as usize] += gpair[r as usize];
                    }
                }
            }
        }
    }
}

/// 行分块的块大小。
///
/// 目标是让「一个 tile 的梯度」留在缓存里被所有特征复用:
/// TILE × 16 字节就是要驻留的量。4096 行 = 64KB。
///
/// HIGGS 上实测(每次累加的耗时,越小越好):
/// 无分块 3.53 / 4k **1.34–1.41** / 8k 1.44 / 16k 1.41–1.49 /
/// 32k 1.44 / 64k 1.55 ns。4k 到 32k 之间很平,64k 掉出 L2 开始变差;
/// 4k 稳定领先约 5%(重复测过)。曲线见 CLAUDE.md「直方图行分块」。
///
/// **GPU 上要重新定**:shared memory 只有 48–100KB,tile 会小一个量级。
/// `FB_TILE` 环境变量可以覆盖,扫参数用。
fn tile_rows() -> usize {
    use std::sync::OnceLock;
    static TILE: OnceLock<usize> = OnceLock::new();
    *TILE.get_or_init(|| {
        std::env::var("FB_TILE")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&v: &usize| v > 0)
            .unwrap_or(4_096)
    })
}

/// 节点的梯度总和 —— 包含缺失行,所以**不能**由直方图求和得到。
///
/// split.rs 靠 `总和 - Σbins` 反推缺失部分,那个减法的被减数必须
/// 独立算出来,否则恒等于 0,缺失方向就永远学不出来。
pub fn node_sum(gpair: &[GradPairFixed], row_idx: Option<&[RowId]>) -> GradPairFixed {
    // 每棵树都要在这里把整份定点梯度扫一遍,只为拿两个整数。
    // HIGGS(1050 万行 × 16 B = 168 MB)上串行扫一遍约 18 ms/轮,而整轮才
    // 169 ms —— 列数越少它越扎眼(它只随行数走,和特征数无关,所以 wide 上
    // 被 300 列的 histogram 盖住,HIGGS 上就浮出来了)。
    //
    // **并行是位精确的**:定点加法按模 2^64 精确、结合律和交换律都成立,
    // 所以分块求和再合并和串行逐个累加**逐位相同**(和定点直方图同一条性质)。
    //
    // ⚠️ 累加必须走 `+=`(`AddAssign`),**不能写成 `wrapping_add`**:
    // `AddAssign` 在 debug 下是 `checked_add` + panic、release 下才 wrapping。
    // 直接用 wrapping 会让 debug 构建**悄悄不再检查溢出**,变成 build-mode
    // 行为差异。并行只改变「哪一次部分加法先触发检查」,不改变「真实总和
    // 溢出会不会被发现」—— scale 预算把总和钉在 2^62 以内。
    const PARALLEL_MIN_ROWS: usize = 1 << 16;
    let sum_all = |g: &[GradPairFixed]| -> GradPairFixed {
        if g.len() < PARALLEL_MIN_ROWS {
            // 小输入直接串行:rayon 的分块开销比这点活还大,而且单测里
            // 大量小调用会被它拖慢。
            let mut acc = GradPairFixed::default();
            for &x in g {
                acc += x;
            }
            acc
        } else {
            g.par_iter()
                .fold(GradPairFixed::default, |mut a, &x| {
                    a += x;
                    a
                })
                .reduce(GradPairFixed::default, |mut a, b| {
                    a += b;
                    a
                })
        }
    };
    match row_idx {
        None => sum_all(gpair),
        Some(rows) => {
            if rows.len() < PARALLEL_MIN_ROWS {
                let mut acc = GradPairFixed::default();
                for &r in rows {
                    acc += gpair[r as usize];
                }
                acc
            } else {
                rows.par_iter()
                    .fold(GradPairFixed::default, |mut a, &r| {
                        a += gpair[r as usize];
                        a
                    })
                    .reduce(GradPairFixed::default, |mut a, b| {
                        a += b;
                        a
                    })
            }
        }
    }
}

/// 父子相减:兄弟节点的直方图 = 父 - 自己。
///
/// 这是 GBDT 最重要的常数优化,只需要对较小的那个子节点做实际累加。
pub fn subtract(parent: &Histogram, child: &Histogram, out: &mut Histogram) {
    for i in 0..out.bins.len() {
        out.bins[i] = parent.bins[i] - child.bins[i];
    }
}

/// 原地相减:`parent -= child`,父的存储**就地变成**大 child 的直方图。
///
/// 调用方把父直方图的所有权**搬**过来,所以这里不分配、不清零,也不需要
/// 第三个 buffer。一个父只会被它那**唯一**的 `Subtract` 子节点认领一次。
///
/// 定点加减按模 `2^64` 精确且顺序无关,所以原地做和三方相减**逐位相同**。
pub fn subtract_in_place(parent: &mut Histogram, child: &Histogram) {
    debug_assert_eq!(parent.bins.len(), child.bins.len(), "相减两边宽度必须一致");
    for i in 0..parent.bins.len() {
        parent.bins[i] = parent.bins[i] - child.bins[i];
    }
}

/// 批量构建:一个列块加载后,把所有活跃节点的直方图一次算完。
///
/// 这就是 internal-docs/ARCHITECTURE.md 里说的「块复用」—— 传输量从
/// O(深度 × 数据) 降到 O(数据)。
pub fn build_all_nodes(
    block_data: &[Bin],
    block_n_rows: usize,
    n_feats: usize,
    feat_offsets: &[u32],
    gpair: &[GradPairFixed],
    // node_rows[i] 是第 i 个活跃节点的行下标
    node_rows: &[Vec<RowId>],
    out: &mut [Histogram],
) {
    debug_assert_eq!(node_rows.len(), out.len());

    // 顺序跑。并行的正确轴是**列块**(见 internal-docs/ARCHITECTURE.md「切分维度 =
    // 并行维度」),不是这里的节点 —— 根节点只有一个,按节点并行时
    // 第一层完全没有并行度。各节点写各自的 out,将来要加 rayon 也安全。
    for (rows, hist) in node_rows.iter().zip(out.iter_mut()) {
        build(
            block_data,
            block_n_rows,
            n_feats,
            feat_offsets,
            gpair,
            Some(rows),
            &mut hist.bins,
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 原地相减必须和三方相减**逐位相同** —— 它换掉的是存储,不是算法。
    #[test]
    fn in_place_subtract_matches_three_way_form() {
        let mut parent = Histogram::zeros(64);
        let mut child = Histogram::zeros(64);
        for i in 0..64 {
            // 故意混进负数和大值,盖住模 2^64 回绕。
            parent.bins[i] = GradPairFixed { grad: (i as i64) * 7 - 13, hess: i64::MAX - i as i64 };
            child.bins[i] = GradPairFixed { grad: -(i as i64) * 3, hess: i as i64 };
        }
        let mut three_way = Histogram::zeros(64);
        subtract(&parent, &child, &mut three_way);

        let mut moved = Histogram { bins: parent.bins.clone() };
        subtract_in_place(&mut moved, &child);
        assert_eq!(moved.bins, three_way.bins, "原地相减必须逐位等于三方相减");
    }

    use super::*;
    use crate::types::{GradPair, GradQuantizer};

    /// 3 个特征、每个 4 个 bin 的块,行数 8。
    /// feat 2 里塞了缺失。
    fn fixture() -> (Vec<Bin>, usize, usize, Vec<u32>, Vec<GradPairFixed>, GradQuantizer) {
        let n_rows = 8;
        let n_feats = 3;
        // 列主序:data[local_feat * n_rows + row]
        let block: Vec<Bin> = vec![
            0, 1, 2, 3, 0, 1, 2, 3, // feat 0
            0, 0, 1, 1, 2, 2, 3, 3, // feat 1
            MISSING_BIN, 1, MISSING_BIN, 2, 0, 0, 3, 1, // feat 2
        ];
        let feat_offsets = vec![0, 4, 8, 12];
        let float: Vec<GradPair> = (0..n_rows)
            .map(|i| GradPair {
                grad: (i as f32) + 1.0,
                hess: 0.5 * (i as f32) + 2.0,
            })
            .collect();
        let q = GradQuantizer::new(&float, n_rows);
        let gpair = float.iter().map(|&g| q.to_fixed(g)).collect();
        (block, n_rows, n_feats, feat_offsets, gpair, q)
    }

    /// 最直白的参考实现:逐行逐特征往对应槽位加。
    fn reference(
        block: &[Bin],
        n_rows: usize,
        n_feats: usize,
        feat_offsets: &[u32],
        gpair: &[GradPairFixed],
        rows: &[RowId],
    ) -> Vec<GradPairFixed> {
        let mut out = vec![GradPairFixed::default(); *feat_offsets.last().unwrap() as usize];
        for &r in rows {
            for f in 0..n_feats {
                let b = block[f * n_rows + r as usize];
                if b == MISSING_BIN {
                    continue;
                }
                out[feat_offsets[f] as usize + b as usize] += gpair[r as usize];
            }
        }
        out
    }

    /// 定点之后这就是**逐位**相等,不再是"差在 1e-5 以内"。
    fn assert_same(a: &[GradPairFixed], b: &[GradPairFixed]) {
        assert_eq!(a, b);
    }

    #[test]
    fn all_rows_matches_reference() {
        let (block, n_rows, n_feats, offs, gpair, _q) = fixture();
        let mut out = vec![GradPairFixed::default(); 12];
        build(&block, n_rows, n_feats, &offs, &gpair, None, &mut out);

        let all: Vec<RowId> = (0..n_rows as RowId).collect();
        assert_same(&out, &reference(&block, n_rows, n_feats, &offs, &gpair, &all));
    }

    #[test]
    fn row_subset_matches_reference() {
        let (block, n_rows, n_feats, offs, gpair, _q) = fixture();
        let rows: Vec<RowId> = vec![1, 3, 4, 7];
        let mut out = vec![GradPairFixed::default(); 12];
        build(&block, n_rows, n_feats, &offs, &gpair, Some(&rows), &mut out);

        assert_same(
            &out,
            &reference(&block, n_rows, n_feats, &offs, &gpair, &rows),
        );
    }

    #[test]
    fn none_and_full_row_list_agree() {
        let (block, n_rows, n_feats, offs, gpair, _q) = fixture();
        let all: Vec<RowId> = (0..n_rows as RowId).collect();

        let mut a = vec![GradPairFixed::default(); 12];
        let mut b = vec![GradPairFixed::default(); 12];
        build(&block, n_rows, n_feats, &offs, &gpair, None, &mut a);
        build(&block, n_rows, n_feats, &offs, &gpair, Some(&all), &mut b);
        assert_same(&a, &b);
    }

    /// 定点方案的核心价值:换个行顺序累加,结果**逐位相同**。
    ///
    /// 浮点做不到这一点 —— 加法不结合,乱序归约必然飘。有了这条,
    /// 将来把行循环换成 rayon 并行、或者 GPU 上按 warp 归约,都不用
    /// 担心结果会变;阶段 3 的 CPU/GPU 对拍也能直接断言相等。
    #[test]
    fn row_order_does_not_change_the_result() {
        let (block, n_rows, n_feats, offs, gpair, _q) = fixture();
        let ascending: Vec<RowId> = (0..n_rows as RowId).collect();
        let mut shuffled = vec![5u32, 0, 7, 2, 6, 1, 4, 3];

        let mut a = vec![GradPairFixed::default(); 12];
        let mut b = vec![GradPairFixed::default(); 12];
        build(&block, n_rows, n_feats, &offs, &gpair, Some(&ascending), &mut a);
        build(&block, n_rows, n_feats, &offs, &gpair, Some(&shuffled), &mut b);
        assert_eq!(a, b, "换行顺序结果就变了,定点累加白做了");

        // 反着来也一样
        shuffled.reverse();
        let mut c = vec![GradPairFixed::default(); 12];
        build(&block, n_rows, n_feats, &offs, &gpair, Some(&shuffled), &mut c);
        assert_eq!(a, c);

        // node_sum 同理
        assert_eq!(node_sum(&gpair, Some(&ascending)), node_sum(&gpair, Some(&shuffled)));
    }

    /// 极端量级下既不能溢出,也不能把精度丢光。
    ///
    /// scale 是按 max|g| × n_rows 反推的,所以梯度整体放大/缩小
    /// 6 个数量级都应该照常工作 —— 溢出是这个方案唯一的失败模式,
    /// 得有测试守着。
    #[test]
    fn extreme_gradient_magnitudes_neither_overflow_nor_lose_the_signal() {
        for mult in [1e-6f32, 1.0, 1e6] {
            let n_rows = 1000;
            let float: Vec<GradPair> = (0..n_rows)
                .map(|i| GradPair {
                    grad: ((i % 17) as f32 - 8.0) * mult,
                    hess: ((i % 5) as f32 + 1.0) * mult,
                })
                .collect();
            let q = GradQuantizer::new(&float, n_rows);
            let fixed: Vec<GradPairFixed> = float.iter().map(|&g| q.to_fixed(g)).collect();

            // debug 构建下溢出会 panic,跑到这儿就说明没溢出
            let sum = node_sum(&fixed, None);
            let (grad, hess) = q.to_f64(sum);

            let want_grad: f64 = float.iter().map(|g| g.grad as f64).sum();
            let want_hess: f64 = float.iter().map(|g| g.hess as f64).sum();
            let tol = 1e-9 * want_hess.abs().max(1e-30 * mult as f64);
            assert!(
                (grad - want_grad).abs() <= tol.max(1e-9 * want_grad.abs()),
                "mult {mult}:grad {grad} vs {want_grad}"
            );
            assert!(
                (hess - want_hess).abs() <= tol.max(1e-9 * want_hess.abs()),
                "mult {mult}:hess {hess} vs {want_hess}"
            );
        }
    }

    #[test]
    fn missing_rows_are_excluded_from_every_bin() {
        let (block, n_rows, n_feats, offs, gpair, _q) = fixture();
        let mut out = vec![GradPairFixed::default(); 12];
        build(&block, n_rows, n_feats, &offs, &gpair, None, &mut out);

        // feat 2 的行 0 和行 2 是缺失,它们的梯度不该出现在任何 bin 里
        let feat2: i64 = out[8..12].iter().map(|g| g.grad).sum();
        let total: i64 = gpair.iter().map(|g| g.grad).sum();
        let missing: i64 = gpair[0].grad + gpair[2].grad;
        assert_eq!(feat2, total - missing);

        // 没有缺失的 feat 0 则必须收齐
        let feat0: i64 = out[0..4].iter().map(|g| g.grad).sum();
        assert_eq!(feat0, total);
    }

    /// `node_sum` 的并行分支只在 >= 65536 行时才走,而其它单测的夹具都远小于
    /// 这个阈值 —— 也就是说**并行那条路在此之前一次都没被跑到过**。
    /// 这里专门造一个跨过阈值的输入,和串行参考逐位对拍。
    ///
    /// 同时构造成**混合正负、量级接近定点上界**,让部分和确实来回穿过 0,
    /// 否则「分块求和」和「逐个累加」的差别根本表现不出来。
    #[test]
    fn node_sum_parallel_branch_matches_serial_reference() {
        let n = (1usize << 16) + 1234; // 跨过阈值,而且不是整块
        let gpair: Vec<GradPairFixed> = (0..n)
            .map(|i| {
                let sign = if i % 3 == 0 { -1 } else { 1 };
                GradPairFixed {
                    grad: sign * ((i as i64 % 9973) << 20),
                    hess: (i as i64 % 7919) << 18,
                }
            })
            .collect();

        let mut want = GradPairFixed::default();
        for &g in &gpair {
            want += g;
        }
        assert_eq!(node_sum(&gpair, None), want, "并行全量求和必须逐位等于串行");

        // 行子集那条分支同样要过阈值才并行。
        let rows: Vec<RowId> = (0..n as RowId).filter(|r| r % 2 == 0).collect();
        let mut want_rows = GradPairFixed::default();
        for &r in &rows {
            want_rows += gpair[r as usize];
        }
        assert_eq!(
            node_sum(&gpair, Some(&rows)),
            want_rows,
            "并行子集求和必须逐位等于串行"
        );
    }

    #[test]
    fn node_sum_includes_missing_rows() {
        let (block, n_rows, n_feats, offs, gpair, _q) = fixture();
        let mut out = vec![GradPairFixed::default(); 12];
        build(&block, n_rows, n_feats, &offs, &gpair, None, &mut out);

        // 这就是 split.rs 的反推:总和 - Σbins = 缺失部分。
        // 定点之后这个反推是精确的,不再有"减出个 1e-7 的假缺失"。
        let sum = node_sum(&gpair, None);
        let binned: i64 = out[8..12].iter().map(|g| g.grad).sum();
        assert_eq!(sum.grad - binned, gpair[0].grad + gpair[2].grad);
    }

    /// `build` **不再自己清零**,契约改成"调用方交进来时就是零"。
    ///
    /// 这条替换掉原来的 `build_zeroes_out_first`:那条往里塞一个脏 buffer
    /// 再断言结果和干净 buffer 相同,钉的正是被删掉的那趟清零。
    /// 钉的是**结果没变**:去掉那趟清零之后,干净 buffer 上的直方图
    /// 必须和逐行手算的参考逐位相同。
    ///
    /// ⚠️ 不要拿"同一个 buffer 调两次看是否翻倍"来测累加语义 ——
    /// 那正好违反新契约(第二次进去时 buffer 已经不是零),debug 断言
    /// 会先炸。**契约本身由下面那条 `should_panic` 钉。**
    #[test]
    fn build_result_is_unchanged_without_the_clear_pass() {
        let (block, n_rows, n_feats, offs, gpair, _q) = fixture();
        let mut got = vec![GradPairFixed::default(); 12];
        build(&block, n_rows, n_feats, &offs, &gpair, None, &mut got);

        // 逐行手算的参考:跳过缺失,其余按 (特征偏移 + bin) 累加。
        let mut want = vec![GradPairFixed::default(); 12];
        for row in 0..n_rows {
            for f in 0..n_feats {
                let bin = block[f * n_rows + row];
                if bin == crate::types::MISSING_BIN {
                    continue;
                }
                let slot = offs[f] as usize + bin as usize;
                want[slot].grad += gpair[row].grad;
                want[slot].hess += gpair[row].hess;
            }
        }
        assert_eq!(got, want, "去掉清零之后直方图变了");
        assert!(got.iter().any(|g| g.grad != 0), "参考数据太弱,全零测不出东西");
    }

    /// debug 断言真的会拦住违约的调用方(否则契约只是注释)。
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "out 进来就是清零的")]
    fn build_rejects_a_dirty_buffer_in_debug() {
        let (block, n_rows, n_feats, offs, gpair, _q) = fixture();
        let mut dirty = vec![GradPairFixed { grad: 99, hess: 99 }; 12];
        build(&block, n_rows, n_feats, &offs, &gpair, None, &mut dirty);
    }

    #[test]
    fn siblings_subtract_to_parent() {
        let (block, n_rows, n_feats, offs, gpair, _q) = fixture();
        let left: Vec<RowId> = vec![0, 2, 5, 6];
        let right: Vec<RowId> = vec![1, 3, 4, 7];

        let mut parent = Histogram::zeros(12);
        let mut lh = Histogram::zeros(12);
        let mut rh = Histogram::zeros(12);
        build(&block, n_rows, n_feats, &offs, &gpair, None, &mut parent.bins);
        build(&block, n_rows, n_feats, &offs, &gpair, Some(&left), &mut lh.bins);
        build(&block, n_rows, n_feats, &offs, &gpair, Some(&right), &mut rh.bins);

        // 缺失行在两边都不进 bin,所以父 = 左 + 右 依然成立。
        // 定点之后是精确相等 —— 父子相减不再往下传误差。
        let mut derived = Histogram::zeros(12);
        subtract(&parent, &lh, &mut derived);
        assert_eq!(derived.bins, rh.bins);
    }

    #[test]
    fn build_all_nodes_matches_per_node_build() {
        let (block, n_rows, n_feats, offs, gpair, _q) = fixture();
        let node_rows: Vec<Vec<RowId>> = vec![vec![0, 1, 2], vec![3, 4], vec![5, 6, 7]];

        let mut batched: Vec<Histogram> = (0..3).map(|_| Histogram::zeros(12)).collect();
        build_all_nodes(&block, n_rows, n_feats, &offs, &gpair, &node_rows, &mut batched);

        for (rows, got) in node_rows.iter().zip(&batched) {
            let mut want = vec![GradPairFixed::default(); 12];
            build(&block, n_rows, n_feats, &offs, &gpair, Some(rows), &mut want);
            assert_same(&got.bins, &want);
        }
    }
}
