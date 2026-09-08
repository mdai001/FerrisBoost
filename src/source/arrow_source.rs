//! 从 Arrow RecordBatch 构建列块。
//!
//! Arrow 是**唯一**的输入接口:Parquet、S3、DuckDB、Polars、任何数据库
//! 的 ADBC 驱动都能吐 RecordBatch,接住这一层就等于全都支持了。
//!
//! 两遍扫描:第一遍建草图定分箱边界,第二遍按边界量化。两遍都按
//! batch 流式处理,不需要把整个数据集拉进内存 —— 这正是「内存需求从
//! O(数据集) 降到 O(列块)」那条主张在输入侧的落点。

use arrow::array::{Array, Float32Array, Float64Array, Int32Array, Int64Array};
use arrow::datatypes::DataType;
use arrow::record_batch::RecordBatch;
use rayon::prelude::*;

use crate::columns::{partition_features, BinCuts, ColumnBlock};
use crate::sketch::SketchSet;
use crate::types::{Bin, FeatId, MAX_BIN_LIMIT, MISSING_BIN};

/// 把一列取成 f32,null 变 NaN。
///
/// 只认数值列。**null 和 NaN 走同一条路**(都变成 MISSING_BIN),
/// 因为 XGBoost 的语义里两者都是"缺失",分裂时由 default direction
/// 决定去向 —— 不是填 0,更不是丢掉整行。
pub(crate) fn column_as_f32(batch: &RecordBatch, col: usize) -> anyhow::Result<Vec<f32>> {
    let array = batch.column(col);
    let n = array.len();
    let mut out = Vec::with_capacity(n);

    macro_rules! pull {
        ($ty:ty) => {{
            let a = array
                .as_any()
                .downcast_ref::<$ty>()
                .expect("类型已经 match 过了");
            for i in 0..n {
                out.push(if a.is_null(i) {
                    f32::NAN
                } else {
                    a.value(i) as f32
                });
            }
        }};
    }

    match array.data_type() {
        DataType::Float32 => pull!(Float32Array),
        DataType::Float64 => pull!(Float64Array),
        DataType::Int32 => pull!(Int32Array),
        DataType::Int64 => pull!(Int64Array),
        other => anyhow::bail!(
            "第 {col} 列是 {other:?},只支持数值列。\
             类别特征要先在上游编码成数值 —— 第一版不做 categorical 分裂"
        ),
    }
    Ok(out)
}

/// 第一遍:扫描建草图,得到分箱边界。
///
/// 每个 batch 喂进同一组草图。草图本身是流式的(见 sketch.rs 的层级
/// 归并),所以这一遍的内存是 O(特征数 × 草图大小),和行数无关。
pub fn build_cuts(
    batches: impl Iterator<Item = anyhow::Result<RecordBatch>>,
    max_bins: usize,
) -> anyhow::Result<BinCuts> {
    build_cuts_with_nthread(batches, max_bins, crate::source::DEFAULT_INGEST_THREADS)
}

/// `build_cuts`,但同一 batch 内按特征并行。`nthread=0` 交给 Rayon
/// 使用可用线程数。
///
/// batch 仍然严格顺序消费,每个特征也始终由同一个草图按 batch 顺序
/// 接收数据。这样不需要合并草图,线程数不会改变 cuts。
pub fn build_cuts_with_nthread(
    batches: impl Iterator<Item = anyhow::Result<RecordBatch>>,
    max_bins: usize,
    nthread: usize,
) -> anyhow::Result<BinCuts> {
    assert!(
        max_bins as u32 <= MAX_BIN_LIMIT,
        "max_bin 上限是 {MAX_BIN_LIMIT}(255 留给缺失哨兵),给的是 {max_bins}"
    );

    // 整次扫描只建一个 pool。每 batch 重建一次在线程数较多、batch 很小
    // 时会让调度开销盖过 sketch 本身。
    let pool = crate::threading::build_pool(nthread)
        .map_err(|e| anyhow::anyhow!("Parquet cuts 线程池创建失败:{e}"))?;
    let profile_columns = pool.current_num_threads() == 1;
    let mut sketches: Option<SketchSet> = None;
    // 手动迭代:`next()` 里才是 Parquet 解压,要和 sketch 分开计时。
    let mut batches = batches;
    loop {
        let next = {
            let _p = crate::source::openprof::sub("read+decode");
            batches.next()
        };
        let Some(batch) = next else { break };
        let batch = batch?;
        let n_feats = batch.num_columns();
        let sk = sketches.get_or_insert_with(|| SketchSet::new(n_feats, max_bins));
        anyhow::ensure!(
            n_feats == sk.n_feats(),
            "batch 之间的列数不一致:先前 {},这个 batch {n_feats}",
            sk.n_feats()
        );
        if profile_columns {
            // 单线程保留原来的细分口径。多线程不能让每个 worker 在任务
            // 结束时争全局 profiler 锁,否则 profile 本身会串行化热循环。
            for col in 0..n_feats {
                let values = {
                    let _p = crate::source::openprof::sub("to_f32");
                    column_as_f32(&batch, col)?
                };
                let _p = crate::source::openprof::sub("sketch");
                sk.push_column(col as FeatId, &values, None);
            }
        } else {
            // 这里的计时是整个 batch 的 elapsed wall,不是各 worker 用时之和。
            let _p = crate::source::openprof::sub("parallel batch");
            pool.install(|| {
                sk.per_feat.par_iter_mut().enumerate().try_for_each(
                    |(col, sketch)| -> anyhow::Result<()> {
                        let values = column_as_f32(&batch, col)?;
                        for value in values {
                            sketch.push(value, 1.0);
                        }
                        Ok(())
                    },
                )
            })?;
        }
    }
    Ok(sketches
        .ok_or_else(|| anyhow::anyhow!("一个 batch 都没有,建不了分箱边界"))?
        .finalize())
}

/// 第二遍:按 cuts 量化,产出列块。
///
/// 列主序,`data[local_feat * n_rows + row]`,和 DenseSource 出来的
/// 布局完全一样 —— 训练侧看不出数据是从哪来的。
pub fn quantize(
    batches: impl Iterator<Item = anyhow::Result<RecordBatch>>,
    cuts: &BinCuts,
    cols_per_block: usize,
) -> anyhow::Result<(Vec<ColumnBlock>, usize)> {
    quantize_with_nthread(
        batches,
        cuts,
        cols_per_block,
        crate::source::DEFAULT_INGEST_THREADS,
    )
}

/// Quantize one bounded RecordBatch at a time while distributing independent
/// column blocks over an ingest-only Rayon pool.  Output columns and rows are
/// appended in canonical order, so scheduling cannot affect the bins.
pub fn quantize_with_nthread(
    batches: impl Iterator<Item = anyhow::Result<RecordBatch>>,
    cuts: &BinCuts,
    cols_per_block: usize,
    nthread: usize,
) -> anyhow::Result<(Vec<ColumnBlock>, usize)> {
    assert!(cols_per_block > 0, "cols_per_block 必须 > 0");
    let n_features = cuts.n_feats();
    let groups = partition_features(n_features, cols_per_block);

    // 每个块先攒成"按特征分段"的 Vec,行数未知所以只能边走边 push。
    // 列主序的好处在这里也成立:同一个特征的 bin 是连续追加的。
    let mut per_block: Vec<Vec<Vec<Bin>>> =
        groups.iter().map(|g| vec![Vec::new(); g.len()]).collect();
    let mut n_rows = 0usize;
    let pool = crate::threading::build_pool(nthread)
        .map_err(|e| anyhow::anyhow!("Arrow quantize ingest 线程池创建失败:{e}"))?;

    for batch in batches {
        let batch = batch?;
        anyhow::ensure!(
            batch.num_columns() == n_features,
            "batch 有 {} 列,cuts 是按 {n_features} 列建的",
            batch.num_columns()
        );
        n_rows += batch.num_rows();

        pool.install(|| {
            per_block
                .par_iter_mut()
                .zip(groups.par_iter())
                .try_for_each(|(block, feats)| -> anyhow::Result<()> {
                    for (local, &f) in feats.iter().enumerate() {
                        let values = column_as_f32(&batch, f as usize)?;
                        let dst = &mut block[local];
                        dst.reserve(values.len());
                        for v in values {
                            // null and NaN share the missing sentinel.
                            dst.push(if v.is_nan() {
                                MISSING_BIN
                            } else {
                                cuts.find_bin(f, v)
                            });
                        }
                    }
                    Ok(())
                })
        })?;
    }

    let blocks = groups
        .into_iter()
        .zip(per_block)
        .map(|(feat_ids, cols)| {
            let mut data = Vec::with_capacity(feat_ids.len() * n_rows);
            for col in cols {
                debug_assert_eq!(col.len(), n_rows, "某一列的行数和别的列对不上");
                data.extend_from_slice(&col);
            }
            ColumnBlock {
                feat_ids,
                data,
                n_rows,
            }
        })
        .collect();

    Ok((blocks, n_rows))
}

/// Arrow 来的数据源。布局和 `DenseSource` 完全一致 —— 训练侧看不出
/// 数据是从内存矩阵来的还是从 Parquet 流进来的。
///
/// 这一版**量化结果全驻内存**(u8 的 bin,比原始 f32 小 4 倍)。
/// 按需从对象存储加载列块是 `ParquetBlockSource` 的事。
pub struct ArrowSource {
    cuts: BinCuts,
    blocks: Vec<ColumnBlock>,
    n_rows: usize,
    n_features: usize,
}

impl ArrowSource {
    /// 两遍扫描:先建草图定边界,再量化。
    ///
    /// `batches` 是个工厂而不是迭代器 —— 要扫两遍,而 RecordBatch 的
    /// 迭代器通常是一次性的(Parquet reader、网络流都是)。让调用方
    /// 提供"再来一遍"的能力,比我们把所有 batch 缓存下来诚实得多:
    /// 缓存就等于把整个数据集拉进内存,正是这个项目要避免的。
    pub fn new<I, F>(batches: F, max_bins: u32, cols_per_block: usize) -> anyhow::Result<Self>
    where
        I: Iterator<Item = anyhow::Result<RecordBatch>>,
        F: FnMut() -> anyhow::Result<I>,
    {
        Self::new_with_nthread(
            batches,
            max_bins,
            cols_per_block,
            crate::source::DEFAULT_INGEST_THREADS,
        )
    }

    pub fn new_with_nthread<I, F>(
        mut batches: F,
        max_bins: u32,
        cols_per_block: usize,
        nthread: usize,
    ) -> anyhow::Result<Self>
    where
        I: Iterator<Item = anyhow::Result<RecordBatch>>,
        F: FnMut() -> anyhow::Result<I>,
    {
        let cuts = build_cuts_with_nthread(batches()?, max_bins as usize, nthread)?;
        Self::with_cuts_and_nthread(batches, cuts, cols_per_block, nthread)
    }

    /// 用给定的 cuts 量化,只扫一遍。
    ///
    /// **验证集走这条**(和 `DenseSource::with_cuts` 同理):自己重新
    /// 建草图会让同一个特征值在训练/验证里落进不同的 bin,模型直接
    /// 失效,而且跑起来不报错,只是指标莫名其妙地差。
    pub fn with_cuts<I, F>(batches: F, cuts: BinCuts, cols_per_block: usize) -> anyhow::Result<Self>
    where
        I: Iterator<Item = anyhow::Result<RecordBatch>>,
        F: FnMut() -> anyhow::Result<I>,
    {
        Self::with_cuts_and_nthread(
            batches,
            cuts,
            cols_per_block,
            crate::source::DEFAULT_INGEST_THREADS,
        )
    }

    pub fn with_cuts_and_nthread<I, F>(
        mut batches: F,
        cuts: BinCuts,
        cols_per_block: usize,
        nthread: usize,
    ) -> anyhow::Result<Self>
    where
        I: Iterator<Item = anyhow::Result<RecordBatch>>,
        F: FnMut() -> anyhow::Result<I>,
    {
        let (blocks, n_rows) = quantize_with_nthread(batches()?, &cuts, cols_per_block, nthread)?;
        let n_features = cuts.n_feats();
        Ok(Self {
            cuts,
            blocks,
            n_rows,
            n_features,
        })
    }

    pub fn into_cuts(self) -> BinCuts {
        self.cuts
    }

    /// 某一行某个特征的 bin。测试和调试用。
    pub fn bin_at(&self, row: usize, feat: FeatId) -> Bin {
        for b in &self.blocks {
            if let Some(local) = b.feat_ids.iter().position(|&f| f == feat) {
                return b.column(local)[row];
            }
        }
        panic!("特征 {feat} 不在任何列块里");
    }
}

impl crate::train::BlockSource for ArrowSource {
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

    /// 量化结果全量持有,直接借出去,**零拷贝**。ArrowSource 收的是
    /// "喂进来的 batch",没有回头再读的能力,所以做不了按需加载 ——
    /// 那是 `ParquetBlockSource` 的事(它能按 column chunk 重新读)。
    fn with_block(&self, i: usize, f: &mut dyn FnMut(&ColumnBlock)) -> anyhow::Result<()> {
        let block = self
            .blocks
            .get(i)
            .ok_or_else(|| anyhow::anyhow!("列块 {i} 不存在,一共 {} 个", self.blocks.len()))?;
        f(block);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::DenseSource;
    use crate::train::BlockSource as _;
    use arrow::array::{Float32Array, Float64Array, Int64Array, StringArray};
    use arrow::datatypes::{Field, Schema};
    use std::sync::Arc;

    /// 3 列 × 6 行,拆成两个 batch;第 2 列带 null。
    fn batches() -> Vec<RecordBatch> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Float32, false),
            Field::new("b", DataType::Float64, true),
            Field::new("c", DataType::Int64, false),
        ]));
        let mk = |a: Vec<f32>, b: Vec<Option<f64>>, c: Vec<i64>| {
            RecordBatch::try_new(
                schema.clone(),
                vec![
                    Arc::new(Float32Array::from(a)),
                    Arc::new(Float64Array::from(b)),
                    Arc::new(Int64Array::from(c)),
                ],
            )
            .unwrap()
        };
        vec![
            mk(
                vec![1.0, 2.0, 3.0],
                vec![Some(10.0), None, Some(30.0)],
                vec![7, 8, 9],
            ),
            mk(
                vec![4.0, 5.0, 6.0],
                vec![Some(40.0), Some(50.0), None],
                vec![1, 2, 3],
            ),
        ]
    }

    fn feed(bs: &[RecordBatch]) -> impl Iterator<Item = anyhow::Result<RecordBatch>> + '_ {
        bs.iter().cloned().map(Ok)
    }

    #[test]
    fn cut_sketch_is_thread_count_independent() {
        let bs = batches();
        let one = build_cuts_with_nthread(feed(&bs), 16, 1).unwrap();
        let eight = build_cuts_with_nthread(feed(&bs), 16, 8).unwrap();
        let rayon_default = build_cuts_with_nthread(feed(&bs), 16, 0).unwrap();

        assert_eq!(one.offsets, eight.offsets);
        assert_eq!(
            one.values.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            eight.values.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
        );
        assert_eq!(one.offsets, rayon_default.offsets);
        assert_eq!(
            one.values.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            rayon_default
                .values
                .iter()
                .map(|v| v.to_bits())
                .collect::<Vec<_>>()
        );
    }

    /// 同一份数据,从 Arrow 进和从稠密矩阵进,量化结果必须**逐位相同**。
    ///
    /// 这是换输入层最要紧的一条:阶段 1a 的位精确基线是拿 DenseSource
    /// 建的,只要这一条成立,训练那一半完全不用重验。
    #[test]
    fn arrow_and_dense_agree_bin_for_bin() {
        let bs = batches();
        let src = ArrowSource::new(|| Ok(feed(&bs)), 16, 2).unwrap();

        // 同样的数据摆成行主序喂 DenseSource,null 用 NaN
        let n = f32::NAN;
        let row_major: Vec<f32> = vec![
            1.0, 10.0, 7.0, //
            2.0, n, 8.0, //
            3.0, 30.0, 9.0, //
            4.0, 40.0, 1.0, //
            5.0, 50.0, 2.0, //
            6.0, n, 3.0,
        ];
        let dense = DenseSource::from_row_major(&row_major, 6, 3, 16, 2);

        assert_eq!(src.n_rows(), 6);
        assert_eq!(src.n_features(), 3);
        assert_eq!(src.n_blocks(), dense.n_blocks());
        for row in 0..6 {
            for f in 0..3u32 {
                assert_eq!(
                    src.bin_at(row, f),
                    dense.bin_at(row, f),
                    "行 {row} 特征 {f}:Arrow 和 Dense 的 bin 不一致"
                );
            }
        }
        // 分箱边界本身也要一样
        for f in 0..3u32 {
            assert_eq!(
                src.cuts().cuts_for(f),
                dense.cuts().cuts_for(f),
                "特征 {f} 的 cuts"
            );
        }
    }

    /// null 和 NaN 都要变成缺失哨兵,不能填 0 也不能丢行。
    #[test]
    fn nulls_become_the_missing_sentinel() {
        let bs = batches();
        let src = ArrowSource::new(|| Ok(feed(&bs)), 16, 3).unwrap();
        // 第 2 列的第 1 行和第 5 行是 null
        assert_eq!(src.bin_at(1, 1), MISSING_BIN);
        assert_eq!(src.bin_at(5, 1), MISSING_BIN);
        // 其余行不是缺失
        for row in [0usize, 2, 3, 4] {
            assert_ne!(src.bin_at(row, 1), MISSING_BIN, "行 {row} 不该是缺失");
        }
    }

    /// batch 的切法不该影响结果 —— 一个 batch 和三个 batch 一样。
    #[test]
    fn batch_boundaries_do_not_change_anything() {
        let bs = batches();
        let split = ArrowSource::new(|| Ok(feed(&bs)), 16, 2).unwrap();

        let merged = arrow::compute::concat_batches(&bs[0].schema(), &bs).unwrap();
        let one = vec![merged];
        let whole = ArrowSource::new(|| Ok(feed(&one)), 16, 2).unwrap();

        assert_eq!(split.n_rows(), whole.n_rows());
        for row in 0..6 {
            for f in 0..3u32 {
                assert_eq!(
                    split.bin_at(row, f),
                    whole.bin_at(row, f),
                    "行 {row} 特征 {f}"
                );
            }
        }
    }

    /// 验证集必须复用训练集的 cuts。
    #[test]
    fn with_cuts_reuses_the_given_boundaries() {
        let bs = batches();
        let train = ArrowSource::new(|| Ok(feed(&bs)), 16, 2).unwrap();
        let cuts = train.cuts().clone();

        // 只拿后半段当"验证集",自己建草图的话边界会完全不同
        let tail = vec![bs[1].clone()];
        let valid = ArrowSource::with_cuts(|| Ok(feed(&tail)), cuts, 2).unwrap();

        assert_eq!(valid.n_rows(), 3);
        for f in 0..3u32 {
            assert_eq!(valid.cuts().cuts_for(f), train.cuts().cuts_for(f));
        }
        // 同一个值在两边落进同一个 bin
        for f in 0..3u32 {
            assert_eq!(valid.bin_at(0, f), train.bin_at(3, f), "特征 {f}");
        }
    }

    #[test]
    fn non_numeric_columns_are_rejected_with_a_clear_message() {
        let schema = Arc::new(Schema::new(vec![Field::new("s", DataType::Utf8, false)]));
        let batch = RecordBatch::try_new(schema, vec![Arc::new(StringArray::from(vec!["a", "b"]))])
            .unwrap();
        let bs = vec![batch];
        let Err(err) = build_cuts_with_nthread(feed(&bs), 16, 8) else {
            panic!("字符串列应该被拒绝");
        };
        let msg = err.to_string();
        assert!(msg.contains("只支持数值列"), "报错要说人话,实际是:{msg}");
    }

    #[test]
    fn mismatched_column_counts_are_caught() {
        let bs = batches();
        let one_col = Arc::new(Schema::new(vec![Field::new("a", DataType::Float32, false)]));
        let odd = RecordBatch::try_new(one_col, vec![Arc::new(Float32Array::from(vec![1.0f32]))])
            .unwrap();
        let mixed = vec![bs[0].clone(), odd];
        let Err(err) = build_cuts_with_nthread(feed(&mixed), 16, 8) else {
            panic!("列数不一致应该被拒绝");
        };
        assert!(err.to_string().contains("列数不一致"), "{err}");
    }

    #[test]
    fn empty_parallel_cut_input_is_rejected() {
        let empty = std::iter::empty::<anyhow::Result<RecordBatch>>();
        let Err(err) = build_cuts_with_nthread(empty, 16, 8) else {
            panic!("空输入应该被拒绝");
        };
        assert!(err.to_string().contains("一个 batch 都没有"), "{err}");
    }
}
