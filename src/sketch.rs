//! 加权分位数草图。
//!
//! 核心性质是**可合并**:每个 batch、每个列块各自建草图,最后 merge
//! 出全局切分点 —— 这是分批处理能成立的基础。
//!
//! # 为什么是层级归并,不是每批压缩
//!
//! 直觉写法是「攒满一个 buffer 就并进主草图并压缩」。这样误差会随
//! 压缩次数**线性**累积:10 万个点、buffer 128,就是 780 次压缩,
//! 均匀分布上的分位误差滚到 0.03 以上,远超 1/max_bins。
//!
//! 这里用 log-structured merge:同一层有两个 summary 就合并并晋升到
//! 上一层,只在晋升时压缩。压缩次数从 O(n/batch) 降到 O(log(n/batch)),
//! 误差跟着从线性变成对数。XGBoost 的 WQuantileSketch 也是这个结构。
//!
//! 实测(10 万点、max_bins=16):
//!   每批压缩      max_err ≈ 0.054
//!   层级 ratio=4  max_err ≈ 0.045
//!   层级 ratio=16 max_err ≈ 0.012

use crate::columns::BinCuts;
use crate::types::FeatId;

/// 每层 summary 的容量相对 bin 数的倍数。
///
/// 内存代价:max_bins=256 时每特征 4096 个 entry × 16 字节 = 64KB,
/// 500 特征约 32MB。可接受。调小会明显掉精度,见文件头的实测数字。
const SKETCH_RATIO: usize = 16;

/// 草图里的一个条目。
///
/// 不变式:按 value 升序;rmin <= rmax。
/// rmin 是严格小于 value 的权重和,rmax 是小于等于 value 的权重和。
#[derive(Clone, Copy, Debug)]
struct Entry {
    value: f32,
    weight: f32,
    rmin: f32,
    rmax: f32,
}

impl Entry {
    fn rmid(&self) -> f32 {
        0.5 * (self.rmin + self.rmax)
    }
}

/// 单个特征的草图。
pub struct FeatureSketch {
    /// 未排序的新输入。攒满 limit 个才建 summary。
    buffer: Vec<(f32, f32)>,
    /// levels[i] 是第 i 层的 summary。None 表示该层空着。
    levels: Vec<Option<Vec<Entry>>>,
    /// 每层 summary 压缩后的容量。
    limit: usize,
    total_weight: f32,
}

impl FeatureSketch {
    pub fn new(max_bins: usize) -> Self {
        let limit = (max_bins * SKETCH_RATIO).max(32);
        Self {
            buffer: Vec::with_capacity(limit),
            levels: Vec::new(),
            limit,
            total_weight: 0.0,
        }
    }

    pub fn total_weight(&self) -> f32 {
        self.total_weight
    }

    /// 加入一个观测值。
    ///
    /// NaN 直接丢弃 —— 缺失值不参与分箱,训练时走 default direction,
    /// 不占 bin。这一点和 XGBoost 一致。
    pub fn push(&mut self, value: f32, weight: f32) {
        if value.is_nan() || weight <= 0.0 {
            return;
        }
        self.total_weight += weight;
        self.buffer.push((value, weight));
        if self.buffer.len() >= self.limit {
            self.drain_buffer();
        }
    }

    fn drain_buffer(&mut self) {
        if self.buffer.is_empty() {
            return;
        }
        let mut pairs = std::mem::take(&mut self.buffer);
        let summary = compress(&build_summary(&mut pairs), self.limit);
        self.buffer = pairs;
        self.buffer.clear();
        self.insert_level(summary);
    }

    /// 把一个 summary 放进层级结构。同层已占用就合并晋升。
    fn insert_level(&mut self, mut cur: Vec<Entry>) {
        let mut lvl = 0;
        loop {
            if lvl == self.levels.len() {
                self.levels.push(Some(cur));
                return;
            }
            match self.levels[lvl].take() {
                None => {
                    self.levels[lvl] = Some(cur);
                    return;
                }
                Some(existing) => {
                    cur = compress(&merge_entries(&existing, &cur), self.limit);
                    lvl += 1;
                }
            }
        }
    }

    /// 把所有层合并成一个 summary。查询和 merge 前都要先做这一步。
    fn collapse(&mut self) -> &[Entry] {
        self.drain_buffer();

        let mut acc: Vec<Entry> = Vec::new();
        for lv in self.levels.drain(..).flatten() {
            acc = if acc.is_empty() {
                lv
            } else {
                merge_entries(&acc, &lv)
            };
        }

        if !acc.is_empty() {
            self.levels.push(Some(compress(&acc, self.limit)));
        }

        self.levels
            .first()
            .and_then(|o| o.as_deref())
            .unwrap_or(&[])
    }

    /// 合并另一个草图。多线程、多机、分批处理都走这条。
    pub fn merge(&mut self, other: &FeatureSketch) {
        let mut tmp = other.snapshot();
        let b: Vec<Entry> = tmp.collapse().to_vec();
        let a: Vec<Entry> = self.collapse().to_vec();

        self.levels.clear();
        if !a.is_empty() || !b.is_empty() {
            self.levels
                .push(Some(compress(&merge_entries(&a, &b), self.limit)));
        }
        self.total_weight += other.total_weight;
    }

    /// merge 需要对方 collapse 过,但不该改动对方。复制一份来做。
    fn snapshot(&self) -> FeatureSketch {
        FeatureSketch {
            buffer: self.buffer.clone(),
            levels: self.levels.clone(),
            limit: self.limit,
            // 权重由调用方累加,这里置零免得重复计
            total_weight: 0.0,
        }
    }

    /// 输出切分点,最多 max_bins - 1 个。
    ///
    /// 语义和 XGBoost 一致:cut 是**上界**,值 v 落入的 bin 是第一个
    /// 满足 v < cuts[i] 的 i;都不满足则落入最后一个 bin。
    pub fn cuts(&mut self, max_bins: usize) -> Vec<f32> {
        let n_cuts = max_bins.saturating_sub(1);
        if n_cuts == 0 {
            return Vec::new();
        }

        let entries = self.collapse();
        if entries.is_empty() {
            return Vec::new();
        }

        // 低基数:唯一值本身就少于需要的 bin 数,直接用相邻唯一值的
        // 中点。这样 one-hot、类别编码这类特征分箱是精确的,不丢信息。
        if entries.len() <= n_cuts {
            return entries
                .windows(2)
                .map(|w| midpoint(w[0].value, w[1].value))
                .collect();
        }

        // 高基数:按等权分位取点。
        let total = entries.last().unwrap().rmax;
        let mut cuts: Vec<f32> = Vec::with_capacity(n_cuts);
        for k in 1..=n_cuts {
            let target = total * (k as f32) / (max_bins as f32);
            let idx = entries
                .partition_point(|e| e.rmid() < target)
                .min(entries.len() - 1);
            let v = entries[idx].value;
            // 去重:相邻分位点可能落在同一个值上
            if cuts.last().map_or(true, |&last| v > last) {
                cuts.push(v);
            }
        }
        cuts
    }

    #[cfg(test)]
    fn n_entries(&mut self) -> usize {
        self.collapse().len()
    }
}

/// 两个值之间的切分点。
///
/// 用中点而不是直接取右值,是为了让边界上的样本落在预期的一侧。
/// 中点因浮点精度等于左值时退化为取右值。
fn midpoint(a: f32, b: f32) -> f32 {
    let m = a + (b - a) * 0.5;
    if m <= a {
        b
    } else {
        m
    }
}

/// 把 (value, weight) 列表变成有序 summary,相同值合并。
fn build_summary(pairs: &mut [(f32, f32)]) -> Vec<Entry> {
    pairs.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    let mut out = Vec::with_capacity(pairs.len());
    let mut acc = 0.0f32;
    let mut i = 0;
    while i < pairs.len() {
        let v = pairs[i].0;
        let mut w = 0.0;
        while i < pairs.len() && pairs[i].0 == v {
            w += pairs[i].1;
            i += 1;
        }
        out.push(Entry {
            value: v,
            weight: w,
            rmin: acc,
            rmax: acc + w,
        });
        acc += w;
    }
    out
}

/// 归并两个有序 summary,重算 rmin/rmax。
///
/// 关键点:合并后某个值的排名下界 = 它在 A 中的下界 + B 中已经完全
/// 走过的权重。这是排名区间能跨草图正确传递的原因,也是整个可合并
/// 性质的数学基础。
fn merge_entries(a: &[Entry], b: &[Entry]) -> Vec<Entry> {
    if a.is_empty() {
        return b.to_vec();
    }
    if b.is_empty() {
        return a.to_vec();
    }

    let mut out = Vec::with_capacity(a.len() + b.len());
    let (mut i, mut j) = (0usize, 0usize);
    let (mut pass_a, mut pass_b) = (0.0f32, 0.0f32);

    while i < a.len() || j < b.len() {
        let take_a = match (a.get(i), b.get(j)) {
            (Some(x), Some(y)) => x.value <= y.value,
            (Some(_), None) => true,
            (None, Some(_)) => false,
            (None, None) => break,
        };

        if take_a {
            let e = a[i];
            out.push(Entry {
                value: e.value,
                weight: e.weight,
                rmin: e.rmin + pass_b,
                rmax: e.rmax + pass_b,
            });
            pass_a = e.rmax;
            i += 1;
        } else {
            let e = b[j];
            out.push(Entry {
                value: e.value,
                weight: e.weight,
                rmin: e.rmin + pass_a,
                rmax: e.rmax + pass_a,
            });
            pass_b = e.rmax;
            j += 1;
        }
    }

    dedup_same_value(out)
}

fn dedup_same_value(entries: Vec<Entry>) -> Vec<Entry> {
    let mut out: Vec<Entry> = Vec::with_capacity(entries.len());
    for e in entries {
        match out.last_mut() {
            Some(last) if last.value == e.value => {
                last.weight += e.weight;
                last.rmin = last.rmin.min(e.rmin);
                last.rmax = last.rmax.max(e.rmax);
            }
            _ => out.push(e),
        }
    }
    out
}

/// 压缩到 max_size 个条目。
///
/// 在 [0, total] 上取等距目标排名,每个目标取第一个 rmid >= target
/// 的条目。**用递增指针**保证每个目标取到不同的条目 —— 早期版本用
/// 「找到后再去重」,多个目标落到同一条目时槽位被丢掉,64 个目标只
/// 剩 39 个,分布出现空洞。这是个很隐蔽的 bug,分位数看起来"差不多"
/// 但误差是设计值的三倍。
///
/// 首尾必须保留:它们是特征的最小值和最大值,丢了会让分箱边界越界。
fn compress(entries: &[Entry], max_size: usize) -> Vec<Entry> {
    let n = entries.len();
    if n <= max_size {
        return entries.to_vec();
    }

    let total = entries[n - 1].rmax;
    let slots = max_size - 1;
    let mut out = Vec::with_capacity(max_size);
    out.push(entries[0]);
    let mut taken = 0usize;

    for k in 1..slots {
        let target = total * (k as f32) / (slots as f32);
        let mut j = taken + 1;
        while j < n - 1 && entries[j].rmid() < target {
            j += 1;
        }
        // 保证严格递增,同时给后面的目标留足条目
        let ceiling = n - 1 - (slots - 1 - k);
        let j = j.max(taken + 1).min(ceiling);
        if j > taken {
            out.push(entries[j]);
            taken = j;
        }
    }

    if taken < n - 1 {
        out.push(entries[n - 1]);
    }
    out
}

/// 全部特征的草图集合。
pub struct SketchSet {
    pub per_feat: Vec<FeatureSketch>,
    max_bins: usize,
}

impl SketchSet {
    pub fn new(n_feats: usize, max_bins: usize) -> Self {
        Self {
            per_feat: (0..n_feats).map(|_| FeatureSketch::new(max_bins)).collect(),
            max_bins,
        }
    }

    /// 喂入一行(稠密)。NaN 表示缺失,会被 push 丢弃。
    /// 特征数。流式输入时用来校验各批的列数一致。
    pub fn n_feats(&self) -> usize {
        self.per_feat.len()
    }

    pub fn push_row(&mut self, row: &[f32], weight: f32) {
        assert_eq!(row.len(), self.per_feat.len(), "row 的特征数和 sketch 不一致");
        for (sk, &v) in self.per_feat.iter_mut().zip(row) {
            sk.push(v, weight);
        }
    }

    /// 喂入一整列。列主序数据走这条,访存连续。
    pub fn push_column(&mut self, feat: FeatId, values: &[f32], weights: Option<&[f32]>) {
        let sk = &mut self.per_feat[feat as usize];
        match weights {
            Some(w) => {
                assert_eq!(values.len(), w.len(), "values 和 weights 长度必须一致");
                for (&v, &wt) in values.iter().zip(w) {
                    sk.push(v, wt);
                }
            }
            None => {
                for &v in values {
                    sk.push(v, 1.0);
                }
            }
        }
    }

    pub fn merge(&mut self, other: &SketchSet) {
        assert_eq!(self.per_feat.len(), other.per_feat.len(), "合并的 sketch 特征数不一致");
        for (a, b) in self.per_feat.iter_mut().zip(&other.per_feat) {
            a.merge(b);
        }
    }

    /// 拼接成扁平的 BinCuts。
    pub fn finalize(mut self) -> BinCuts {
        let max_bins = self.max_bins;
        let mut values = Vec::new();
        let mut offsets = Vec::with_capacity(self.per_feat.len() + 1);
        offsets.push(0u32);

        for sk in self.per_feat.iter_mut() {
            values.extend_from_slice(&sk.cuts(max_bins));
            offsets.push(values.len() as u32);
        }

        BinCuts::new(values, offsets)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 用 cuts 给一个值定 bin。必须和训练时的量化语义一致。
    fn find_bin(cuts: &[f32], v: f32) -> usize {
        cuts.partition_point(|&c| c <= v)
    }

    fn max_quantile_err(cuts: &[f32], max_bins: usize) -> f32 {
        cuts.iter()
            .enumerate()
            .map(|(k, &v)| (v - (k + 1) as f32 / max_bins as f32).abs())
            .fold(0.0, f32::max)
    }

    #[test]
    fn low_cardinality_is_exact() {
        let mut sk = FeatureSketch::new(256);
        for _ in 0..1000 {
            for v in [1.0, 2.0, 3.0, 5.0] {
                sk.push(v, 1.0);
            }
        }
        let cuts = sk.cuts(256);
        assert_eq!(cuts.len(), 3, "4 个唯一值应该有 3 个切分点");

        let bins: Vec<usize> = [1.0, 2.0, 3.0, 5.0]
            .iter()
            .map(|&v| find_bin(&cuts, v))
            .collect();
        assert_eq!(bins, vec![0, 1, 2, 3], "每个唯一值独占一个 bin");
    }

    #[test]
    fn uniform_quantiles_sorted_input() {
        let n = 100_000;
        let max_bins = 16;
        let mut sk = FeatureSketch::new(max_bins);
        for i in 0..n {
            sk.push(i as f32 / n as f32, 1.0);
        }
        let cuts = sk.cuts(max_bins);
        assert!(cuts.len() <= max_bins - 1);
        let e = max_quantile_err(&cuts, max_bins);
        assert!(e < 0.02, "有序输入分位误差 {e}");
    }

    #[test]
    fn uniform_quantiles_shuffled_input() {
        // 打乱顺序比有序更难 —— 每批都跨全域,归并时重叠严重
        let n = 100_000;
        let max_bins = 16;
        let mut sk = FeatureSketch::new(max_bins);
        for i in 0..n {
            let v = ((i * 7919) % n) as f32 / n as f32;
            sk.push(v, 1.0);
        }
        let cuts = sk.cuts(max_bins);
        let e = max_quantile_err(&cuts, max_bins);
        assert!(e < 0.02, "乱序输入分位误差 {e}");
    }

    #[test]
    fn merge_matches_single_pass() {
        let n = 50_000;
        let max_bins = 32;

        let mut whole = FeatureSketch::new(max_bins);
        let mut a = FeatureSketch::new(max_bins);
        let mut b = FeatureSketch::new(max_bins);

        for i in 0..n {
            let v = ((i * 7919) % n) as f32 / n as f32;
            whole.push(v, 1.0);
            if i % 2 == 0 {
                a.push(v, 1.0);
            } else {
                b.push(v, 1.0);
            }
        }

        a.merge(&b);
        let c_whole = whole.cuts(max_bins);
        let c_merged = a.cuts(max_bins);

        assert_eq!(c_whole.len(), c_merged.len());
        for (x, y) in c_whole.iter().zip(&c_merged) {
            assert!((x - y).abs() < 0.03, "分批 {y} vs 一次性 {x}");
        }
    }

    #[test]
    fn merge_preserves_total_weight() {
        let mut a = FeatureSketch::new(16);
        let mut b = FeatureSketch::new(16);
        for i in 0..100 {
            a.push(i as f32, 1.0);
            b.push(i as f32, 2.0);
        }
        a.merge(&b);
        assert_eq!(a.total_weight(), 300.0);
    }

    #[test]
    fn weights_shift_quantiles() {
        let mut unweighted = FeatureSketch::new(8);
        let mut weighted = FeatureSketch::new(8);
        for i in 0..1000 {
            let v = i as f32;
            unweighted.push(v, 1.0);
            weighted.push(v, if v > 500.0 { 10.0 } else { 1.0 });
        }
        let cu = unweighted.cuts(8);
        let cw = weighted.cuts(8);
        assert!(cw[3] > cu[3], "加权后中位数应该更大: {} vs {}", cw[3], cu[3]);
    }

    #[test]
    fn nan_is_ignored() {
        let mut sk = FeatureSketch::new(8);
        for i in 0..100 {
            sk.push(i as f32, 1.0);
            sk.push(f32::NAN, 1.0);
        }
        assert_eq!(sk.total_weight(), 100.0, "NaN 不计入权重");
    }

    #[test]
    fn compression_bounds_size() {
        let max_bins = 16;
        let mut sk = FeatureSketch::new(max_bins);
        for i in 0..200_000 {
            sk.push((i % 99991) as f32, 1.0);
        }
        let n = sk.n_entries();
        assert!(n <= max_bins * SKETCH_RATIO + 2, "压缩后条目数 {n}");
    }

    #[test]
    fn constant_feature_has_no_cuts() {
        let mut sk = FeatureSketch::new(256);
        for _ in 0..1000 {
            sk.push(42.0, 1.0);
        }
        assert!(sk.cuts(256).is_empty(), "常量特征不应产生切分点");
    }

    #[test]
    fn cuts_are_strictly_increasing() {
        // 极端偏斜:99% 是 0,剩下发散
        let mut sk = FeatureSketch::new(16);
        for i in 0..100_000 {
            sk.push(if i % 100 == 0 { i as f32 } else { 0.0 }, 1.0);
        }
        let cuts = sk.cuts(16);
        for w in cuts.windows(2) {
            assert!(w[0] < w[1], "切分点必须严格递增: {:?}", cuts);
        }
    }

    #[test]
    fn empty_sketch_is_safe() {
        let mut sk = FeatureSketch::new(16);
        assert!(sk.cuts(16).is_empty());
        assert_eq!(sk.total_weight(), 0.0);
    }

    #[test]
    fn sketch_set_finalize() {
        let mut set = SketchSet::new(3, 8);
        for i in 0..1000 {
            set.push_row(&[i as f32, (i % 3) as f32, 7.0], 1.0);
        }
        let cuts = set.finalize();
        assert_eq!(cuts.n_feats(), 3);
        assert!(cuts.n_bins(0) > 4, "连续特征应该有多个 bin");
        assert_eq!(cuts.n_bins(1), 3, "3 个唯一值 -> 3 个 bin");
        assert_eq!(cuts.n_bins(2), 1, "常量特征 -> 1 个 bin");
    }
}
