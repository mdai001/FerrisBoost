//! 内存里的稠密 f32 矩阵入口 —— 阶段 1a 专用。
//!
//! 刻意不走 Arrow:数值对齐是阶段 1a 唯一的目的,schema、null 表示、
//! chunk 边界这些和算法无关的问题混进来会让调试非常痛苦。等 hist /
//! split / tree 都和 XGBoost 对上了,再换输入层就是低风险改动。
//!
//! 缺失值用 NaN 表示,量化成 MISSING_BIN。

use crate::columns::{partition_features, BinCuts, BinningStrategy, ColumnBlock};
use crate::sketch::FeatureSketch;
use rayon::prelude::*;
use crate::train::BlockSource;
use crate::types::{Bin, FeatId, MAX_BIN_LIMIT, MISSING_BIN};

pub struct DenseSource {
    cuts: BinCuts,
    blocks: Vec<ColumnBlock>,
    n_rows: usize,
    n_features: usize,
}

impl DenseSource {
    /// 从行主序矩阵构建。`data.len() == n_rows * n_features`,
    /// 元素按 `data[row * n_features + feat]` 排列 —— 这是用户手里
    /// numpy / Vec<Vec<f32>> 摊平后最自然的形状。
    ///
    /// 内部转成列主序:量化后的 `ColumnBlock.data` 是
    /// `data[local_feat * n_rows + row]`。转置在这里发生一次,
    /// 之后每轮训练都吃连续访存的红利。
    pub fn from_row_major(
        data: &[f32],
        n_rows: usize,
        n_features: usize,
        max_bins: u32,
        cols_per_block: usize,
    ) -> Self {
        Self::from_row_major_with_nthread(
            data, n_rows, n_features, max_bins, cols_per_block, 0,
        )
    }

    /// `from_row_major`,但分箱明确服从 `nthread`。0 = 用满可用核。
    pub fn from_row_major_with_nthread(
        data: &[f32],
        n_rows: usize,
        n_features: usize,
        max_bins: u32,
        cols_per_block: usize,
        nthread: usize,
    ) -> Self {
        Self::new_with_nthread(
            data,
            n_rows,
            n_features,
            BinningStrategy::Sketch { max_bins: max_bins as usize },
            cols_per_block,
            nthread,
        )
    }

    /// 总入口:分箱边界自己算还是外部给,由 `binning` 决定。
    pub fn new(
        data: &[f32],
        n_rows: usize,
        n_features: usize,
        binning: BinningStrategy,
        cols_per_block: usize,
    ) -> Self {
        Self::new_with_nthread(data, n_rows, n_features, binning, cols_per_block, 0)
    }

    /// `new`,但用独立 Rayon pool 让分箱线程数和训练参数保持一致。
    pub fn new_with_nthread(
        data: &[f32],
        n_rows: usize,
        n_features: usize,
        binning: BinningStrategy,
        cols_per_block: usize,
        nthread: usize,
    ) -> Self {
        assert_eq!(
            data.len(),
            n_rows * n_features,
            "行主序矩阵长度应为 n_rows * n_features"
        );
        assert!(cols_per_block > 0, "cols_per_block 必须 > 0");

        let cuts = match binning {
            BinningStrategy::Sketch { max_bins } => {
                assert!(
                    max_bins as u32 <= MAX_BIN_LIMIT,
                    "max_bin 上限是 {MAX_BIN_LIMIT}(255 留给缺失哨兵),给的是 {max_bins}"
                );
                // 第一遍:建草图。**按特征并行** —— 每个特征的草图完全
                // 由一个线程从头建到尾,线程之间不共享任何东西:
                // 不用锁,也不用合并。
                //
                // 「不合并」是关键:草图合并是有损的(见 sketch.rs 的
                // 层级归并),合并顺序变了 cuts 就可能变。按行分块并行
                // 就得合并,那样 cuts 会跟着线程数走 —— 而 cuts 变了
                // 整个模型就变了。按特征切没有这个问题,
                // `binning_is_thread_count_independent` 钉着这条。
                //
                // 代价:每个特征要跨 stride 读(行主序下同一特征的相邻
                // 两个值隔着 n_features 个 f32)。宽数据上这不划算,
                // 但它换来的是零同步。
                let pool = crate::threading::build_pool(nthread)
                    .expect("分箱线程池创建失败");
                let per_feat: Vec<Vec<f32>> = pool.install(|| {
                    (0..n_features)
                        .into_par_iter()
                        .map(|f| {
                            let mut sk = FeatureSketch::new(max_bins);
                            for r in 0..n_rows {
                                sk.push(data[r * n_features + f], 1.0);
                            }
                            sk.cuts(max_bins)
                        })
                        .collect()
                });

                let mut values = Vec::new();
                let mut offsets = Vec::with_capacity(n_features + 1);
                offsets.push(0u32);
                for c in per_feat {
                    values.extend_from_slice(&c);
                    offsets.push(values.len() as u32);
                }
                BinCuts::new(values, offsets)
            }
            BinningStrategy::Provided(cuts) => {
                assert_eq!(cuts.n_feats(), n_features, "给定的 cuts 特征数和数据对不上");
                cuts.clone()
            }
        };

        Self::quantize_row_major(data, n_rows, n_features, cuts, cols_per_block)
    }

    /// 用给定的 cuts 量化 —— 验证集必须走这条,不能自己重新建草图。
    ///
    /// 验证集用自己的分箱边界会让同一个特征值在训练/验证里落进不同
    /// 的 bin,模型直接失效。这个错误跑起来不报错,只是指标莫名其妙
    /// 地差,所以单独开一个入口把它变成显式选择。
    pub fn with_cuts(
        data: &[f32],
        n_rows: usize,
        n_features: usize,
        cuts: BinCuts,
        cols_per_block: usize,
    ) -> Self {
        assert_eq!(data.len(), n_rows * n_features);
        assert_eq!(
            cuts.n_feats(),
            n_features,
            "cuts 的特征数和数据对不上"
        );
        Self::quantize_row_major(data, n_rows, n_features, cuts, cols_per_block)
    }

    fn quantize_row_major(
        data: &[f32],
        n_rows: usize,
        n_features: usize,
        cuts: BinCuts,
        cols_per_block: usize,
    ) -> Self {
        let blocks = partition_features(n_features, cols_per_block)
            .into_iter()
            .map(|feat_ids| {
                let mut block = vec![0 as Bin; feat_ids.len() * n_rows];
                for (local, &f) in feat_ids.iter().enumerate() {
                    debug_assert!(
                        cuts.n_bins(f) <= MAX_BIN_LIMIT as usize,
                        "特征 {f} 的 bin 数超过哨兵留下的空间"
                    );
                    let dst = &mut block[local * n_rows..(local + 1) * n_rows];
                    for (row, slot) in dst.iter_mut().enumerate() {
                        let v = data[row * n_features + f as usize];
                        // NaN 是唯一的缺失表示。find_bin 不接受 NaN
                        // (partition_point 的比较对 NaN 永远是 false,
                        // 会静默落到 bin 0),所以必须在这里拦掉。
                        *slot = if v.is_nan() {
                            MISSING_BIN
                        } else {
                            cuts.find_bin(f, v)
                        };
                    }
                }
                ColumnBlock { feat_ids, data: block, n_rows }
            })
            .collect();

        Self { cuts, blocks, n_rows, n_features }
    }

    /// 交出分箱边界的所有权,用来构建验证集。
    pub fn into_cuts(self) -> BinCuts {
        self.cuts
    }

    pub fn blocks(&self) -> &[ColumnBlock] {
        &self.blocks
    }

    /// 某一行某个特征的 bin。测试和调试用,训练热路径不走这里。
    pub fn bin_at(&self, row: usize, feat: FeatId) -> Bin {
        for b in &self.blocks {
            if let Some(local) = b.feat_ids.iter().position(|&f| f == feat) {
                return b.column(local)[row];
            }
        }
        panic!("特征 {feat} 不在任何列块里");
    }
}

impl BlockSource for DenseSource {
    fn n_blocks(&self) -> usize {
        self.blocks.len()
    }

    fn n_rows(&self) -> usize {
        self.n_rows
    }

    fn n_features(&self) -> usize {
        self.n_features
    }

    fn cuts(&self) -> &BinCuts {
        &self.cuts
    }



    /// 纯元数据,不碰数据。
    fn block_features(&self, i: usize) -> anyhow::Result<Vec<FeatId>> {
        self.blocks
            .get(i)
            .map(|b| b.feat_ids.clone())
            .ok_or_else(|| anyhow::anyhow!("列块 {i} 不存在,共 {} 个", self.blocks.len()))
    }

    /// 直接从常驻的块里切一列出来。
    ///
    /// **必须覆盖 trait 的默认实现** —— 默认实现是「`block(i)` 拿所有权
    /// 再切一列」,对全量驻留的 source 就是「为了 1 列克隆整块」。
    /// HIGGS 上 28 列 × 1050 万行 = 每次克隆 294MB,行重分区因此吃掉
    /// 单轮一半的时间(profile 抓出来的,见 CLAUDE.md「6.9× 花在哪」)。
    fn feature_column(&self, feat: FeatId) -> anyhow::Result<Vec<Bin>> {
        for b in &self.blocks {
            if let Some(local) = b.feat_ids.iter().position(|&f| f == feat) {
                return Ok(b.column(local).to_vec());
            }
        }
        anyhow::bail!("特征 {feat} 不在任何列块里,共 {} 个特征", self.n_features)
    }

    /// 全量持有,直接借出去,**零拷贝**。
    fn with_block(&self, i: usize, f: &mut dyn FnMut(&ColumnBlock)) -> anyhow::Result<()> {
        let block = self
            .blocks
            .get(i)
            .ok_or_else(|| anyhow::anyhow!("列块下标 {i} 越界,共 {} 块", self.blocks.len()))?;
        f(block);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::columns::BinningStrategy;

    /// 4 行 3 列,行主序。
    fn toy() -> (Vec<f32>, usize, usize) {
        let data = vec![
            1.0, 10.0, 0.0, //
            2.0, 20.0, 0.0, //
            3.0, 30.0, 1.0, //
            4.0, 40.0, 1.0, //
        ];
        (data, 4, 3)
    }

    #[test]
    fn blocks_cover_every_feature_once() {
        let (data, n_rows, n_feats) = toy();
        let src = DenseSource::from_row_major(&data, n_rows, n_feats, 16, 2);

        assert_eq!(src.n_blocks(), 2); // 3 列切成 [0,1] 和 [2]
        let mut seen: Vec<FeatId> = src
            .blocks()
            .iter()
            .flat_map(|b| b.feat_ids.iter().copied())
            .collect();
        seen.sort_unstable();
        assert_eq!(seen, vec![0, 1, 2]);
        assert_eq!(src.n_rows(), 4);
        assert_eq!(src.n_features(), 3);
    }

    #[test]
    fn layout_is_column_major() {
        let (data, n_rows, n_feats) = toy();
        let src = DenseSource::from_row_major(&data, n_rows, n_feats, 16, 2);
        src.with_block(0, &mut |b| {
            assert_eq!(b.data.len(), b.n_feats() * n_rows);
            // 一整列是连续的一段
            assert_eq!(b.column(0), &b.data[0..n_rows]);
            assert_eq!(b.column(1), &b.data[n_rows..2 * n_rows]);
        })
        .unwrap();
    }

    #[test]
    fn quantization_matches_find_bin() {
        let (data, n_rows, n_feats) = toy();
        let src = DenseSource::from_row_major(&data, n_rows, n_feats, 16, 2);

        for r in 0..n_rows {
            for f in 0..n_feats as FeatId {
                let v = data[r * n_feats + f as usize];
                assert_eq!(
                    src.bin_at(r, f),
                    src.cuts().find_bin(f, v),
                    "row {r} feat {f}"
                );
            }
        }
    }

    #[test]
    fn monotonic_values_give_monotonic_bins() {
        let (data, n_rows, n_feats) = toy();
        let src = DenseSource::from_row_major(&data, n_rows, n_feats, 16, 2);

        // 特征 0 严格递增,bin 也必须不降
        for r in 1..n_rows {
            assert!(src.bin_at(r, 0) >= src.bin_at(r - 1, 0));
        }
        // 低基数特征(只有 0/1)应当被精确分开
        assert_ne!(src.bin_at(0, 2), src.bin_at(3, 2));
    }

    #[test]
    fn nan_becomes_missing_sentinel() {
        let data = vec![
            1.0, f32::NAN, //
            2.0, 5.0, //
            f32::NAN, 6.0, //
        ];
        let src = DenseSource::from_row_major(&data, 3, 2, 16, 8);

        assert_eq!(src.bin_at(0, 1), MISSING_BIN);
        assert_eq!(src.bin_at(2, 0), MISSING_BIN);
        // 非缺失的没有被哨兵污染
        assert_ne!(src.bin_at(0, 0), MISSING_BIN);
        assert_ne!(src.bin_at(1, 1), MISSING_BIN);
    }

    /// 注入外部 cuts:和 `with_cuts` 走同一条路,但入口是正式 API。
    /// 和 XGBoost 对拍、以及将来跨机训练统一 cuts,都靠这个口子。
    /// 分箱是按特征并行的,**cuts 必须和线程数无关**。
    ///
    /// 每个特征的草图由单个线程建完,不做跨线程合并 —— 草图合并是有损
    /// 的,合并顺序一变 cuts 就可能变,而 cuts 变了整个模型就变了。
    /// 这条挂了说明有人把并行改成了按行分块 + 合并。
    #[test]
    fn binning_is_thread_count_independent() {
        let (n_rows, nf) = (2000usize, 12usize);
        let mut data = Vec::with_capacity(n_rows * nf);
        let mut state = 0x243f_6a88_85a3_08d3u64;
        for _ in 0..n_rows * nf {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            data.push((state >> 40) as f32 / 1000.0);
        }

        let cuts_of = |threads: usize| {
            DenseSource::from_row_major_with_nthread(&data, n_rows, nf, 64, 4, threads)
                .into_cuts()
        };

        let a = cuts_of(1);
        let b = cuts_of(8);
        for f in 0..nf as u32 {
            assert_eq!(
                a.cuts_for(f),
                b.cuts_for(f),
                "特征 {f}:1 线程和 8 线程分箱出的 cuts 不一样"
            );
        }
    }

    #[test]
    fn provided_binning_uses_the_given_cuts_verbatim() {
        let data: Vec<f32> = vec![0.0, 10.0, 1.0, 20.0, 2.0, 30.0, 3.0, 40.0];
        // 自己建草图会得到完全不同的边界;这里指定死
        let cuts = BinCuts::new(vec![1.5, 2.5, 15.0], vec![0, 2, 3]);
        let src = DenseSource::new(&data, 4, 2, BinningStrategy::Provided(&cuts), 2);

        // feat 0 的切分点是 [1.5, 2.5]:0 和 1 落 bin 0,2 落 bin 1,3 落 bin 2
        assert_eq!(src.bin_at(0, 0), 0);
        assert_eq!(src.bin_at(1, 0), 0);
        assert_eq!(src.bin_at(2, 0), 1);
        assert_eq!(src.bin_at(3, 0), 2);
        // feat 1 的切分点是 [15.0]:10 落 bin 0,其余落 bin 1
        assert_eq!(src.bin_at(0, 1), 0);
        assert_eq!(src.bin_at(1, 1), 1);
        assert_eq!(src.cuts().cuts_for(0), &[1.5, 2.5]);
    }

    /// XGBoost 给的是 bin **边界**(前面 -inf、后面一个上界哨兵),
    /// 两头都要掐掉才是我们的切分点。掐错一头整个分箱就偏一格。
    #[test]
    fn xgboost_edges_convert_to_interior_cuts() {
        // 8 个唯一值的特征 + 一个常数特征,照 3.4.1 实际输出的形状
        let indptr = vec![0u32, 9, 11];
        let edges = vec![
            f32::NEG_INFINITY, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 14.0, // feat 0
            f32::NEG_INFINITY, 6.0, // feat 1:常数,没有内部切分点
        ];
        let cuts = BinCuts::from_xgboost_edges(&indptr, &edges).unwrap();

        assert_eq!(cuts.cuts_for(0), &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
        assert_eq!(cuts.n_bins(0), 8, "8 个唯一值就该是 8 个 bin");
        assert_eq!(cuts.cuts_for(1), &[] as &[f32]);
        assert_eq!(cuts.n_bins(1), 1, "常数特征只有一个 bin");

        // 每个唯一值各占一个 bin —— 这是"低基数分箱无损"的判据
        for v in 0..8 {
            assert_eq!(cuts.find_bin(0, v as f32), v as u8);
        }

        // 非递增的输入要报错,不能硬建一个语义错乱的 BinCuts
        assert!(BinCuts::from_xgboost_edges(&[0, 4], &[f32::NEG_INFINITY, 2.0, 1.0, 9.0]).is_err());
    }

    #[test]
    fn with_cuts_reuses_training_boundaries() {
        let (train, n_rows, n_feats) = toy();
        let src = DenseSource::from_row_major(&train, n_rows, n_feats, 16, 8);
        let cuts = src.cuts();

        // 验证集的值域完全不同 —— 自己建草图会得到另一套边界
        let valid = vec![
            100.0, 1000.0, 1.0, //
            1.5, 15.0, 0.0, //
        ];
        let expected: Vec<Bin> = (0..2)
            .flat_map(|r| (0..n_feats as FeatId).map(move |f| (r, f)))
            .map(|(r, f)| cuts.find_bin(f, valid[r * n_feats + f as usize]))
            .collect();

        let cuts = DenseSource::from_row_major(&train, n_rows, n_feats, 16, 8).into_cuts();
        let vsrc = DenseSource::with_cuts(&valid, 2, n_feats, cuts, 8);

        let got: Vec<Bin> = (0..2)
            .flat_map(|r| (0..n_feats as FeatId).map(move |f| (r, f)))
            .map(|(r, f)| vsrc.bin_at(r, f))
            .collect();
        assert_eq!(got, expected);
    }

    #[test]
    #[should_panic(expected = "max_bin")]
    fn rejects_max_bin_above_limit() {
        let (data, n_rows, n_feats) = toy();
        DenseSource::from_row_major(&data, n_rows, n_feats, 256, 8);
    }
}
