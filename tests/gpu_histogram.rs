#![cfg(feature = "cuda")]

use ferrisboost::{
    backend::cuda::{
        build_histogram, build_histogram_shared, build_histogram_shared_with_config,
        partition_rows_gpu, GpuTrainCtx, SharedHistogramConfig, SharedHistogramLayout,
    },
    hist,
    source::DenseSource,
    train::{train, train_gpu, train_gpu_one_tree, TrainConfig},
    comm::Local,
    types::{GradPairFixed, RowId, MISSING_BIN},
};

fn assert_cpu_global_shared_equal(
    bins: &[u8],
    n_rows: usize,
    n_feats: usize,
    offsets: &[u32],
    gpair: &[GradPairFixed],
    rows: Option<&[RowId]>,
) {
    let mut cpu = vec![GradPairFixed::default(); offsets[n_feats] as usize];
    hist::build(bins, n_rows, n_feats, offsets, gpair, rows, &mut cpu);
    let global = build_histogram(0, bins, n_rows, n_feats, offsets, gpair, rows).unwrap();
    let shared = build_histogram_shared(0, bins, n_rows, n_feats, offsets, gpair, rows).unwrap();
    assert_eq!(global, cpu, "CPU/global-atomic GPU 必须逐位相同");
    assert_eq!(shared, cpu, "CPU/shared-memory GPU 必须逐位相同");
    assert_eq!(shared, global, "两个 GPU 实现必须逐位相同");
}

#[test]
/// `partition_scan_blocks` 是**单 block 多线程的分块 scan**:tile 内做
/// Hillis-Steele,tile 之间用 carry 串起来。**carry 那条路只有在
/// `n_blocks > blockDim.x`(1024)时才会走到**,而其它 partition 测试的夹具
/// 只有几十到几千行 —— `n_blocks` 个位数,永远是单 tile。
///
/// 所以这里专门造一个跨过阈值的规模:partition 的 row block 是 512 行,
/// 要 `n_blocks > 1024` 就需要 > 524,288 行。取 1,200,000 行 →
/// n_blocks = 2344 → 3 个 tile,carry 被真正累加两次。
///
/// (这条测试的由来:`node_sum` 并行分支当初就是因为所有夹具都低于阈值而
/// 完全没有 CI 覆盖。凡是「够大才启用」的快路径,都要有跨过阈值的测试。)
fn gpu_partition_scan_spans_multiple_scan_tiles() {
    let n_rows = 1_200_000usize;
    // 逆序行下标:顺便钉住 stable 顺序不依赖输入是否升序。
    let rows: Vec<RowId> = (0..n_rows as RowId).rev().collect();
    // 让左右都不是平凡的全有/全无,并掺入缺失。
    let bins: Vec<u8> = (0..n_rows)
        .map(|row| {
            if row % 97 == 0 {
                MISSING_BIN
            } else {
                (row % 7) as u8
            }
        })
        .collect();
    for missing_left in [false, true] {
        let (left, right) =
            partition_rows_gpu(0, &bins, n_rows, 0, &rows, 3, missing_left, 512).unwrap();
        let goes_left = |&row: &RowId| {
            let bin = bins[row as usize];
            if bin == MISSING_BIN {
                missing_left
            } else {
                (bin as u32) < 3
            }
        };
        let want_left: Vec<_> = rows.iter().filter(|r| goes_left(*r)).copied().collect();
        let want_right: Vec<_> = rows.iter().filter(|r| !goes_left(*r)).copied().collect();
        assert_eq!(left.len() + right.len(), n_rows, "行数必须守恒");
        assert!(left.len() > 1000 && right.len() > 1000, "两侧都要非平凡");
        assert_eq!(left, want_left, "跨 scan tile 的左侧顺序必须和串行参考一致");
        assert_eq!(right, want_right, "跨 scan tile 的右侧顺序必须和串行参考一致");
    }
}

#[test]
fn gpu_partition_matches_stable_cpu_order() {
    let n_rows = 37usize;
    let n_feats = 2usize;
    let rows: Vec<RowId> = (0..n_rows as RowId).rev().collect();
    let bins: Vec<u8> = (0..n_feats)
        .flat_map(|feat| {
            (0..n_rows).map(move |row| {
                if row % 11 == 0 {
                    MISSING_BIN
                } else {
                    ((row + feat * 2) % 7) as u8
                }
            })
        })
        .collect();
    for missing_left in [false, true] {
        let (left, right) = partition_rows_gpu(
            0, &bins, n_rows, 1, &rows, 3, missing_left, 8,
        )
        .unwrap();
        let goes_left = |&row: &RowId| {
            let bin = bins[n_rows + row as usize];
            if bin == MISSING_BIN {
                missing_left
            } else {
                (bin as u32) < 3
            }
        };
        let want_left: Vec<_> = rows.iter().filter(|row| goes_left(*row)).copied().collect();
        let want_right: Vec<_> = rows.iter().filter(|row| !goes_left(*row)).copied().collect();
        assert_eq!(left, want_left);
        assert_eq!(right, want_right);
        let (left_again, right_again) = partition_rows_gpu(
            0, &bins, n_rows, 1, &rows, 3, missing_left, 8,
        )
        .unwrap();
        assert_eq!(left_again, left);
        assert_eq!(right_again, right);
    }
}

#[test]
fn training_partition_keeps_stable_rows_on_device() {
    let n_rows = 1_037usize;
    let col: Vec<u8> = (0..n_rows)
        .map(|row| if row % 17 == 0 { MISSING_BIN } else { (row % 7) as u8 })
        .collect();
    let ctx = GpuTrainCtx::new(
        0,
        n_rows,
        1,
        8,
        1,
        512,
        SharedHistogramConfig::default(),
        false,
    )
    .unwrap();
    let root = ctx.begin_tree_rows().unwrap();
    let token = {
        let mut session = ctx.partition_session(&col).unwrap();
        session.partition(root, 3, true).unwrap()
    };
    // partition 现在只 launch;左右行数由整层一次 D2H 取回。
    let (left, right) = ctx.resolve_partitions().unwrap()[token];
    ctx.finish_partition_level().unwrap();
    let got_left = ctx.download_rows(left).unwrap();
    let got_right = ctx.download_rows(right).unwrap();
    let want_left: Vec<_> = (0..n_rows as RowId)
        .filter(|&row| col[row as usize] == MISSING_BIN || col[row as usize] < 3)
        .collect();
    let want_right: Vec<_> = (0..n_rows as RowId)
        .filter(|&row| col[row as usize] != MISSING_BIN && col[row as usize] >= 3)
        .collect();
    assert_eq!(got_left, want_left);
    assert_eq!(got_right, want_right);
}

#[test]
fn gpu_one_tree_streaming_matches_cpu_model() {
    let n_rows = 200usize;
    let n_feats = 5usize;
    let data: Vec<f32> = (0..n_rows)
        .flat_map(|row| (0..n_feats).map(move |feat| ((row * 7 + feat * 3) % 19) as f32))
        .collect();
    let labels: Vec<f32> = (0..n_rows)
        .map(|row| if (row * 11) % 23 < 9 { 1.0 } else { 0.0 })
        .collect();
    let source = DenseSource::from_row_major(&data, n_rows, n_feats, 16, 2);
    let params = ferrisboost::types::TrainParams {
        n_rounds: 1,
        max_depth: 1,
        nthread: 1,
        ..Default::default()
    };
    let cfg = TrainConfig::new(params, ferrisboost::types::Objective::Logistic);
    let cpu = train(&source, &labels, &[], &cfg, &mut [], &Local).unwrap();
    let gpu = train_gpu_one_tree(0, &source, &labels, &cfg, &Local).unwrap();
    assert_eq!(
        serde_json::to_vec(&gpu).unwrap(),
        serde_json::to_vec(&cpu).unwrap(),
        "流式 GPU 单树模型必须与 CPU 逐字节相同"
    );
}

#[test]
fn gpu_multilevel_training_matches_cpu_model() {
    let n_rows = 200usize;
    let n_feats = 5usize;
    let data: Vec<f32> = (0..n_rows)
        .flat_map(|row| (0..n_feats).map(move |feat| ((row * 5 + feat * 7) % 23) as f32))
        .collect();
    let labels: Vec<f32> = (0..n_rows)
        .map(|row| if (row * 13) % 29 < 12 { 1.0 } else { 0.0 })
        .collect();
    let source = DenseSource::from_row_major(&data, n_rows, n_feats, 16, 2);
    let params = ferrisboost::types::TrainParams {
        n_rounds: 2,
        max_depth: 2,
        nthread: 1,
        ..Default::default()
    };
    let cfg = TrainConfig::new(params, ferrisboost::types::Objective::Logistic);
    let cpu = train(&source, &labels, &[], &cfg, &mut [], &Local).unwrap();
    let gpu = train_gpu(0, &source, &labels, &[], &cfg, &mut [], &Local).unwrap();
    assert_eq!(serde_json::to_vec(&gpu).unwrap(), serde_json::to_vec(&cpu).unwrap());
}

/// 训练级资源复用的验收:多轮 × 多层 × 多列块 × 带缺失值。
///
/// 盯的是资源生命周期改造**一个数字都不能动**。这些形状专门去踩复用
/// 逻辑里容易出错的地方:
///
/// - 多个列块在两套 stream-local bins buffer 间复用(不能跨 stream 覆写)
/// - 一个列块内多个活跃节点共用一次 H2D(块只上传一次)
/// - depth 0 的 identity 行索引在块之间复用、但每轮要重新装
/// - partition 的输出 buffer 跨节点复用(只读写各自那一段)
/// - 行数不是 threads_per_block 的整数倍,tail launch 必须走到
#[test]
fn gpu_training_reuses_resources_without_changing_the_model() {
    let n_rows = 1_037usize; // 刻意不是 512 的倍数,逼出 tail launch
    let n_feats = 13usize;
    let data: Vec<f32> = (0..n_rows)
        .flat_map(|row| {
            (0..n_feats).map(move |feat| {
                if (row * 7 + feat) % 17 == 0 {
                    f32::NAN // 缺失值:默认方向要在 GPU/CPU 两边一致
                } else {
                    ((row * 11 + feat * 5) % 41) as f32
                }
            })
        })
        .collect();
    let labels: Vec<f32> = (0..n_rows)
        .map(|row| if (row * 19) % 37 < 15 { 1.0 } else { 0.0 })
        .collect();
    // 4 个列块 × 4 层 × 3 轮:bins buffer 被覆写 48 次以上。
    let source = DenseSource::from_row_major(&data, n_rows, n_feats, 32, 4);
    let params = ferrisboost::types::TrainParams {
        n_rounds: 3,
        max_depth: 4,
        max_bin: 32,
        cols_per_block: 4,
        nthread: 1,
        ..Default::default()
    };
    let cfg = TrainConfig::new(params, ferrisboost::types::Objective::Logistic);
    let cpu = train(&source, &labels, &[], &cfg, &mut [], &Local).unwrap();
    let gpu = train_gpu(0, &source, &labels, &[], &cfg, &mut [], &Local).unwrap();
    assert!(gpu.trees.iter().any(|t| t.nodes.len() > 1), "这个夹具应该真的分裂");
    assert_eq!(
        gpu.to_xgboost_json().unwrap(),
        cpu.to_xgboost_json().unwrap(),
        "复用 device buffer 之后模型仍必须和 CPU 逐字节相同"
    );
}

/// 同时覆盖 200 行尾部、不同特征 bin 宽、缺失值和根节点 null row_idx。
#[test]
fn gpu_histogram_matches_cpu_for_all_rows() {
    let n_rows = 200usize;
    let widths = [2u32, 5, 1, 7, 3];
    let mut offsets = vec![0u32];
    for width in widths {
        offsets.push(offsets.last().unwrap() + width);
    }

    let mut bins = Vec::with_capacity(n_rows * widths.len());
    for (feat, &width) in widths.iter().enumerate() {
        for row in 0..n_rows {
            let bin = if (row * 11 + feat * 7) % 29 == 0 {
                MISSING_BIN
            } else {
                ((row * 3 + feat) % width as usize) as u8
            };
            bins.push(bin);
        }
    }
    let gpair: Vec<_> = (0..n_rows)
        .map(|row| GradPairFixed {
            grad: (row as i64 % 17) - 8,
            hess: (row as i64 % 13) + 1,
        })
        .collect();

    assert_cpu_global_shared_equal(&bins, n_rows, widths.len(), &offsets, &gpair, None);
}

/// 子集长度和内容都与物理列 stride 不同,守住 FFI 中两者不能混用。
#[test]
fn gpu_histogram_matches_cpu_for_row_subset() {
    let n_rows = 200usize;
    let n_feats = 5usize;
    let offsets = [0u32, 2, 7, 8, 15, 18];
    let bins: Vec<u8> = (0..n_feats)
        .flat_map(|feat| {
            let width = (offsets[feat + 1] - offsets[feat]) as usize;
            (0..n_rows).map(move |row| {
                if (row + feat * 5) % 31 == 0 {
                    MISSING_BIN
                } else {
                    ((row * 7 + feat * 3) % width) as u8
                }
            })
        })
        .collect();
    let gpair: Vec<_> = (0..n_rows)
        .map(|row| GradPairFixed {
            grad: 101 - row as i64,
            hess: row as i64 * 2 + 3,
        })
        .collect();
    let rows: Vec<RowId> = (0..n_rows as RowId)
        .filter(|row| row % 3 == 1 || row % 17 == 0)
        .collect();

    assert_cpu_global_shared_equal(&bins, n_rows, n_feats, &offsets, &gpair, Some(&rows));
}

#[test]
fn gpu_histogram_handles_single_feature_single_row() {
    let bins = [2u8];
    let offsets = [0u32, 3];
    let gpair = [GradPairFixed { grad: -7, hess: 11 }];
    assert_cpu_global_shared_equal(&bins, 1, 1, &offsets, &gpair, None);

    let missing = [MISSING_BIN];
    let rows = [0u32];
    assert_cpu_global_shared_equal(&missing, 1, 1, &offsets, &gpair, Some(&rows));
}

/// 30 × 255 bins 需要 122,400 bytes,超过测试路径配置的单-block
/// shared-memory budget,保证 host 真的跨多个 feature tile 启动。
/// 257 行也保证第二个 row block 带尾部。
#[test]
fn shared_histogram_tiles_features_and_rows() {
    let n_rows = 257usize;
    let n_feats = 30usize;
    let width = 255u32;
    let offsets: Vec<u32> = (0..=n_feats).map(|feat| feat as u32 * width).collect();
    let bins: Vec<u8> = (0..n_feats)
        .flat_map(|feat| {
            (0..n_rows).map(move |row| {
                if (row * 13 + feat * 17) % 41 == 0 {
                    MISSING_BIN
                } else {
                    ((row * 7 + feat * 11) % width as usize) as u8
                }
            })
        })
        .collect();
    let gpair: Vec<_> = (0..n_rows)
        .map(|row| GradPairFixed {
            grad: (row as i64 % 31) - 15,
            hess: row as i64 * 3 + 1,
        })
        .collect();

    assert_cpu_global_shared_equal(&bins, n_rows, n_feats, &offsets, &gpair, None);

    let mut cpu = vec![GradPairFixed::default(); offsets[n_feats] as usize];
    hist::build(&bins, n_rows, n_feats, &offsets, &gpair, None, &mut cpu);
    for layout in SharedHistogramLayout::ALL {
        let config = SharedHistogramConfig {
            runtime_log: false,
            resident_blocks: None,
            gpu_memory_budget: None,
            hist_streams: None,
            device_quantize: None,
            hist_nodes_per_batch: None,
            gpu_math: ferrisboost::types::GpuMath::Exact,
            threads_per_block: 64,
            shared_memory_budget_bytes: Some(16 * 1024),
            layout,
        };
        let (configured, profile) = build_histogram_shared_with_config(
            0, &bins, n_rows, n_feats, &offsets, &gpair, None, config,
        )
        .unwrap();
        assert_eq!(configured, cpu, "{layout:?} 必须逐位相同");
        assert!(profile.feature_tiles > 1);
        assert_eq!(profile.threads_per_block, 64);
        assert_eq!(profile.shared_memory_budget_bytes, 16 * 1024);
        assert_eq!(profile.layout, layout);
        assert!(profile.active_blocks_per_sm > 0);
        assert!(profile.theoretical_occupancy > 0.0);
        assert!(profile.theoretical_occupancy <= 1.0);
    }
}

/// 逼出 **32-bit 拆分累加的进位路径**。
///
/// `Interleaved16Atomic32` 把一次 64-bit shared atomicAdd 拆成两次 32-bit
/// 加一次手动进位。**进位只在低 32 位溢出时才发生**,而上面那个 layout
/// 夹具的 grad 只有 -15..15、hess 不到 800,每个 bin 的和离 2^32 差着好几
/// 个数量级 —— 也就是说它**从来没走到过进位那一支**,全绿也证明不了什么。
///
/// 这里按真实定点量级构造:`pow2_scale` 的预算是 `2^62 / n_rows`,
/// 所以单值量级本来就在 2^38 附近。再让大量行落进同一个 bin,低 32 位会
/// 反复溢出;负 grad 的二补数低位接近 0xFFFFFFFF,更是每加一次就进位。
#[test]
fn atomic32_layout_is_bit_exact_when_low_word_overflows() {
    let n_rows = 4_096usize;
    let n_feats = 8usize;
    let width = 16u32; // 故意窄:行多、bin 少,单个 bin 累加上千次
    let offsets: Vec<u32> = (0..=n_feats).map(|feat| feat as u32 * width).collect();
    let bins: Vec<u8> = (0..n_feats)
        .flat_map(|feat| {
            (0..n_rows).map(move |row| {
                if (row * 3 + feat) % 97 == 0 {
                    MISSING_BIN
                } else {
                    ((row + feat) % width as usize) as u8
                }
            })
        })
        .collect();
    // 量级贴着真实定点预算(2^62 / n_rows);混合正负,并特意让一部分值的
    // 低 32 位紧贴 0xFFFFFFFF,保证进位分支被反复走到。
    let gpair: Vec<_> = (0..n_rows)
        .map(|row| {
            let big = 1i64 << 38;
            let grad = match row % 4 {
                0 => big + row as i64,
                1 => -(big + row as i64),
                2 => 0xFFFF_FFF0i64 - (row as i64 % 8), // 低位贴着 u32 上界
                _ => -(0xFFFF_FFFFi64) + row as i64,
            };
            GradPairFixed { grad, hess: big + (row as i64 * 7) }
        })
        .collect();

    let mut cpu = vec![GradPairFixed::default(); offsets[n_feats] as usize];
    hist::build(&bins, n_rows, n_feats, &offsets, &gpair, None, &mut cpu);
    // 先确认夹具真的会溢出低 32 位,否则这个测试等于没测。
    let overflowed = cpu.iter().any(|g| {
        (g.grad as u64) > u32::MAX as u64 || (g.hess as u64) > u32::MAX as u64
    });
    assert!(overflowed, "夹具没有让低 32 位溢出,进位分支没被覆盖");

    for layout in SharedHistogramLayout::ALL {
        let config = SharedHistogramConfig {
            runtime_log: false,
            resident_blocks: None,
            gpu_memory_budget: None,
            hist_streams: None,
            device_quantize: None,
            hist_nodes_per_batch: None,
            gpu_math: ferrisboost::types::GpuMath::Exact,
            threads_per_block: 256,
            shared_memory_budget_bytes: Some(16 * 1024),
            layout,
        };
        let (got, _) = build_histogram_shared_with_config(
            0, &bins, n_rows, n_feats, &offsets, &gpair, None, config,
        )
        .unwrap();
        assert_eq!(got, cpu, "{layout:?} 在低位溢出时必须仍然逐位相同");
    }
}

/// multi-node histogram batching 必须**跨过批容量边界**。
///
/// 深度 5 的一层有 16 个节点,配上 `hist_nodes_per_batch = 4` 就会切成 4 批 ——
/// 这条路径(`todo.chunks(cap)` 走多于一次)和单批路径是不同的代码,
/// 单批测试覆盖不到批与批之间的 `hist_out` 复用和 `blk_ptr` 重建。
///
/// 同时钉住三件事:批容量不改变模型、和 CPU 逐字节相同、
/// 关闭合并(`hist_nodes_per_batch = 1`)与打开合并结果相同。
///
/// ⚠️ 「按规模阈值启用的 fast path 必须有跨过阈值的测试」是这个仓库
/// 反复吃过亏的一条规则,不要因为默认值现在是 16 就删掉这个用例。
#[test]
fn gpu_multi_node_batching_crosses_the_batch_boundary() {
    let n_rows = 4000usize;
    let n_feats = 6usize;
    let data: Vec<f32> = (0..n_rows)
        .flat_map(|row| (0..n_feats).map(move |feat| ((row * 11 + feat * 17) % 41) as f32))
        .collect();
    let labels: Vec<f32> = (0..n_rows)
        .map(|row| if (row * 7) % 31 < 14 { 1.0 } else { 0.0 })
        .collect();
    let source = DenseSource::from_row_major(&data, n_rows, n_feats, 32, 3);

    let build = |hist_nodes_per_batch: Option<usize>| {
        let params = ferrisboost::types::TrainParams {
            n_rounds: 2,
            // depth 5 => 最深一层 16 个节点,cap=4 时一定要分 4 批。
            max_depth: 5,
            nthread: 1,
            hist_nodes_per_batch,
            ..Default::default()
        };
        let cfg = TrainConfig::new(params, ferrisboost::types::Objective::Logistic);
        serde_json::to_vec(&train_gpu(0, &source, &labels, &[], &cfg, &mut [], &Local).unwrap())
            .unwrap()
    };

    let cpu_params = ferrisboost::types::TrainParams {
        n_rounds: 2,
        max_depth: 5,
        nthread: 1,
        ..Default::default()
    };
    let cpu_cfg = TrainConfig::new(cpu_params, ferrisboost::types::Objective::Logistic);
    let cpu = serde_json::to_vec(
        &train(&source, &labels, &[], &cpu_cfg, &mut [], &Local).unwrap(),
    )
    .unwrap();

    let off = build(Some(1));
    let split = build(Some(4)); // 16 个节点 / 4 = 4 批,跨边界
    let single = build(Some(16)); // 一批装下

    assert_eq!(off, cpu, "关闭合并时必须和 CPU 逐字节相同");
    assert_eq!(split, cpu, "分成多批时必须和 CPU 逐字节相同");
    assert_eq!(single, cpu, "单批时必须和 CPU 逐字节相同");
}

/// `gpu_math = "fast"` 的契约:**它不承诺和 CPU 逐字节相同**,所以不能用
/// 逐字节断言。但「不保证位精确」不等于「什么都不保证」,这里钉住三件事:
///
/// 1. **同一档内确定性**:两次 fast 运行必须逐字节相同。丢掉这一条就没法
///    做 A/B,也没法复现 bug。
/// 2. **树结构不变**:HIGGS 上实测 20 棵树的 split feature 和 split
///    condition 与 exact 完全一致 —— `expf` 的 4 ULP 差异没有翻转任何一次
///    分裂决策。这里在小夹具上钉住同样的性质。
/// 3. **叶子值只在末位漂移**:实测最大相对差 9.8e-6。
///
/// ⚠️ 第 2 条在别的数据上**可能**不成立(接近并列的分裂会翻转),所以断言
/// 写在这个固定夹具上,不是当成普遍定律。
#[test]
fn gpu_math_fast_keeps_structure_and_stays_deterministic() {
    let n_rows = 3000usize;
    let n_feats = 6usize;
    let data: Vec<f32> = (0..n_rows)
        .flat_map(|row| (0..n_feats).map(move |feat| ((row * 13 + feat * 29) % 37) as f32))
        .collect();
    let labels: Vec<f32> = (0..n_rows)
        .map(|row| if (row * 17) % 23 < 11 { 1.0 } else { 0.0 })
        .collect();
    let source = DenseSource::from_row_major(&data, n_rows, n_feats, 32, 3);

    let run = |math: ferrisboost::types::GpuMath| {
        let params = ferrisboost::types::TrainParams {
            n_rounds: 3,
            max_depth: 4,
            nthread: 1,
            gpu_math: math,
            ..Default::default()
        };
        let cfg = TrainConfig::new(params, ferrisboost::types::Objective::Logistic);
        train_gpu(0, &source, &labels, &[], &cfg, &mut [], &Local).unwrap()
    };

    let exact = run(ferrisboost::types::GpuMath::Exact);
    let fast = run(ferrisboost::types::GpuMath::Fast);
    let fast_again = run(ferrisboost::types::GpuMath::Fast);

    // 1. fast 自身必须确定性。
    assert_eq!(
        serde_json::to_vec(&fast).unwrap(),
        serde_json::to_vec(&fast_again).unwrap(),
        "gpu_math=fast 两次运行必须逐字节相同"
    );

    // 2. 树结构必须一致。
    assert_eq!(exact.trees.len(), fast.trees.len());
    for (te, tf) in exact.trees.iter().zip(fast.trees.iter()) {
        assert_eq!(te.nodes.len(), tf.nodes.len(), "节点数不同");
        for (ne, nf) in te.nodes.iter().zip(tf.nodes.iter()) {
            assert_eq!(ne.is_leaf, nf.is_leaf, "叶子/内点结构不同");
            if !ne.is_leaf {
                assert_eq!(ne.feat, nf.feat, "分裂特征不同");
                assert_eq!(ne.split_cond, nf.split_cond, "分裂阈值不同");
                assert_eq!(ne.default_left, nf.default_left, "缺失方向不同");
                assert_eq!((ne.left, ne.right), (nf.left, nf.right), "子节点连接不同");
            }
        }
    }

    // 3. 叶子值只允许末位漂移。
    for (te, tf) in exact.trees.iter().zip(fast.trees.iter()) {
        for (ne, nf) in te.nodes.iter().zip(tf.nodes.iter()) {
            if ne.is_leaf {
                let d = (ne.leaf_value - nf.leaf_value).abs();
                let rel = d / ne.leaf_value.abs().max(1e-12);
                assert!(
                    rel < 1e-3,
                    "叶子值相对差 {rel:.3e} 太大:exact {} vs fast {}",
                    ne.leaf_value,
                    nf.leaf_value
                );
            }
        }
    }
}

/// 宽列块(>32 特征/块)在行主序原型下**必须回退到列主序**,不能静默丢特征。
///
/// ⚠️ 这个用例来自一次真实的静默错误:行主序 kernel 用「两条 lane 一行、
/// 每条一次 uint4」的映射,一行最多覆盖 32 个特征,而 `feat_hi` 当时写成
/// `min(feat_lo + 16, n_feats)` —— 于是第 33 个及以后的特征被
/// **悄悄丢掉**,直方图少算而没有任何报错。
///
/// 现在的约定是:`max_block_feats > 32` 时整个关掉行主序(见
/// `GpuTrainCtx::new` 里的硬门槛),**明确 fallback,而不是用 `min()`
/// 把不支持的形状伪装成支持**。将来要支持 64+ 特征/块,应当单独设计
/// lanes_per_row / 多段映射,而不是放宽这里的判断。
///
/// 在 `FB_ROW_MAJOR=1` 下运行时这条用例才真正走到 fallback;不设该变量时
/// 它退化成一个普通的宽列块正确性用例,两种情况都应该通过。
#[test]
fn wide_blocks_fall_back_from_row_major_without_dropping_features() {
    let n_rows = 3000usize;
    // 64 特征/块 —— 超过行主序 kernel 一行能覆盖的 32。
    let n_feats = 64usize;
    let data: Vec<f32> = (0..n_rows)
        .flat_map(|row| (0..n_feats).map(move |feat| ((row * 7 + feat * 13) % 43) as f32))
        .collect();
    let labels: Vec<f32> = (0..n_rows)
        .map(|row| if (row * 11) % 19 < 9 { 1.0 } else { 0.0 })
        .collect();
    // cols_per_block = 64 => 一个列块就装下全部 64 个特征。
    let source = DenseSource::from_row_major(&data, n_rows, n_feats, 64, 3);
    let params = ferrisboost::types::TrainParams {
        n_rounds: 2,
        max_depth: 4,
        nthread: 1,
        ..Default::default()
    };
    let cfg = TrainConfig::new(params, ferrisboost::types::Objective::Logistic);
    let cpu = train(&source, &labels, &[], &cfg, &mut [], &Local).unwrap();
    let gpu = train_gpu(0, &source, &labels, &[], &cfg, &mut [], &Local).unwrap();
    assert_eq!(
        serde_json::to_vec(&gpu).unwrap(),
        serde_json::to_vec(&cpu).unwrap(),
        "宽列块下 GPU 必须和 CPU 逐字节相同 —— 行主序要么正确处理,要么明确回退"
    );
}

/// 运行时用的是**签入的 PTX**,不是 `.cu`。改了 `.cu` 却忘记跑 nvcc,
/// 训练会拿旧 kernel 跑并且**没有任何报错**。
///
/// ⚠️ 已经踩过一次:把行主序 kernel 的 `start` 从 `local * blockDim.x` 改成
/// `local * (blockDim.x >> 1)`,host 侧同步改了每 block 行数,但那一轮只跑了
/// `cargo build` 没跑 nvcc —— 于是 kernel 每个 block 跳过一半的行,
/// 直方图**静默少算约一半**,而两边代码看起来都是对的。
///
/// mtime 在 clean checkout 里没有语义,所以 PTX 生成器嵌入源文件的稳定
/// FNV-1a64;测试比较内容指纹,不再靠 checkout 恰好先写了哪个文件。
#[test]
fn checked_in_ptx_is_not_older_than_its_source() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("src/backend/cuda");
    let source = std::fs::read(dir.join("histogram.cu")).expect("histogram.cu");
    let mut hash = 0xcbf29ce484222325u64;
    for byte in source {
        hash = (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3);
    }
    let ptx = std::fs::read_to_string(dir.join("histogram.ptx")).expect("histogram.ptx");
    let marker = format!("// FerrisBoost source FNV-1a64: {hash:016x}");
    assert!(
        ptx.lines().any(|line| line == marker),
        "histogram.ptx 不是由当前 histogram.cu 生成的。请运行 scripts/build_ptx.sh"
    );
}

/// `colsample_bytree` 的跨后端契约。
///
/// FerrisBoost 的采样序列是自己的(见 `colsample` 模块的说明:XGBoost 自己
/// 的 CPU/GPU 就选不出同一批特征),所以这里钉的**不是**和 XGBoost 逐位相同,
/// 而是 FerrisBoost 自己的承诺:
///
/// 1. `colsample_bytree = 1.0` 与不设它完全一样(不能因为加了这条路径就漂);
/// 2. **CPU 与 GPU 用同一个采样集** —— exact 模式下模型仍然逐字节相同;
/// 3. 同 seed 可复现,换 seed 会变;
/// 4. 换 `cols_per_block` 不改变选中的全局特征 id(所以模型也不变)。
///
/// 第 4 条尤其重要:采样是在**全局特征 id** 上做的,如果误写成块内下标,
/// 换一个 `cols_per_block` 就会静默训出另一个模型。
#[test]
fn colsample_bytree_is_backend_and_blocking_invariant() {
    let n_rows = 3000usize;
    let n_feats = 40usize;
    let data: Vec<f32> = (0..n_rows)
        .flat_map(|row| (0..n_feats).map(move |feat| ((row * 7 + feat * 13) % 47) as f32))
        .collect();
    let labels: Vec<f32> = (0..n_rows)
        .map(|row| if (row * 11) % 17 < 8 { 1.0 } else { 0.0 })
        .collect();

    // ⚠️ 签名是 (data, n_rows, n_features, **max_bins**, **cols_per_block**)。
    // 之前把这两个参数写反,变成在改量化而不是改分块 —— 测试因此"失败",
    // 但那是测试自己的 bug,不是采样的。
    let train_with = |rate: f32, seed: u64, cols_per_block: usize, on_gpu: bool| -> Vec<u8> {
        let source = DenseSource::from_row_major(&data, n_rows, n_feats, 32, cols_per_block);
        let params = ferrisboost::types::TrainParams {
            n_rounds: 3,
            max_depth: 4,
            nthread: 1,
            colsample_bytree: rate,
            seed,
            ..Default::default()
        };
        let cfg = TrainConfig::new(params, ferrisboost::types::Objective::Logistic);
        let model = if on_gpu {
            train_gpu(0, &source, &labels, &[], &cfg, &mut [], &Local).unwrap()
        } else {
            train(&source, &labels, &[], &cfg, &mut [], &Local).unwrap()
        };
        serde_json::to_vec(&model).unwrap()
    };

    // 1. rate = 1.0 必须和默认路径完全一致。
    assert_eq!(
        train_with(1.0, 0, 8, false),
        train_with(1.0, 7, 8, false),
        "colsample=1.0 时 seed 不该影响任何东西"
    );

    // 2. CPU 与 GPU 必须用同一个采样集 —— exact 下模型逐字节相同。
    let cpu = train_with(0.3, 42, 8, false);
    let gpu = train_with(0.3, 42, 8, true);
    assert_eq!(cpu, gpu, "CPU 与 GPU 必须使用同一个采样集");

    // 3. 同 seed 可复现;换 seed 要变。
    assert_eq!(cpu, train_with(0.3, 42, 8, false));
    assert_ne!(cpu, train_with(0.3, 43, 8, false), "换 seed 应该换采样集");

    // 4. 换 cols_per_block 不改变全局采样集,因此模型不变。
    assert_eq!(
        cpu,
        train_with(0.3, 42, 5, false),
        "cols_per_block 不得影响选中哪些全局特征"
    );
    assert_eq!(cpu, train_with(0.3, 42, 40, false));

    // 极小比例仍然至少一个特征,并且能正常训练。
    let tiny = train_with(0.001, 42, 8, false);
    assert!(!tiny.is_empty());
    assert_eq!(tiny, train_with(0.001, 42, 13, false));
}

/// **改变显存几何只允许改变时间,不允许改变模型。**
///
/// 三档物理块宽:显式窄、规划器自动解出的、以及一个更宽的合法值。
/// exact 模式下三者必须**逐字节相同** —— 块宽是内存计划的属性,
/// 不是训练语义的一部分。
///
/// 同时钉住:`colsample` 选中的**全局特征 id 不随块宽变化**
/// (采样在全局 id 上做;写成块内下标的话,换块宽就会静默训出另一个模型)。
#[test]
fn streaming_block_width_changes_timing_not_the_model() {
    use ferrisboost::gpu_mem_plan::{resolve_gpu_streaming_cols_per_block, StreamingMemModel};

    let n_rows = 4000usize;
    let n_feats = 32usize;
    let data: Vec<f32> = (0..n_rows)
        .flat_map(|row| (0..n_feats).map(move |feat| ((row * 5 + feat * 11) % 53) as f32))
        .collect();
    let labels: Vec<f32> = (0..n_rows)
        .map(|row| if (row * 13) % 23 < 11 { 1.0 } else { 0.0 })
        .collect();

    // 规划器在这个形状上会解出什么宽度(用一个宽裕的预算,让它按块数目标走)。
    let model = StreamingMemModel::new(n_rows as u64, 32, 1, 16, true, false, true, 0);
    let plan = resolve_gpu_streaming_cols_per_block(
        None,
        n_feats as u64,
        4 * 1024 * 1024 * 1024,
        6 * 1024 * 1024 * 1024,
        model,
    )
    .unwrap();
    let auto_w = plan.resolved_cols_per_block as usize;
    assert!(auto_w >= 1 && auto_w <= n_feats, "auto 宽度要合法:{auto_w}");

    let run = |cols_per_block: usize, rate: f32| -> Vec<u8> {
        let source = DenseSource::from_row_major(&data, n_rows, n_feats, 32, cols_per_block);
        let params = ferrisboost::types::TrainParams {
            n_rounds: 3,
            max_depth: 4,
            nthread: 1,
            colsample_bytree: rate,
            seed: 9,
            ..Default::default()
        };
        let cfg = TrainConfig::new(params, ferrisboost::types::Objective::Logistic);
        serde_json::to_vec(&train_gpu(0, &source, &labels, &[], &cfg, &mut [], &Local).unwrap())
            .unwrap()
    };

    // 三档宽度:窄 / auto / 更宽。
    let narrow = 4usize;
    let wider = n_feats; // 单块,合法的另一端
    for rate in [1.0f32, 0.5] {
        let a = run(narrow, rate);
        let b = run(auto_w, rate);
        let c = run(wider, rate);
        assert_eq!(a, b, "colsample={rate}:窄({narrow}) 与 auto({auto_w}) 必须逐字节相同");
        assert_eq!(a, c, "colsample={rate}:窄({narrow}) 与宽({wider}) 必须逐字节相同");
    }
}

/// `subsample` 的核心不变量与跨后端一致性。
///
/// > **采样决定哪些行「塑造」这棵树,不决定哪些行「收到」这棵树。**
///
/// 钉三件事:
/// 1. `subsample = 1.0` 与不设它**逐字节相同**(不能因为加了这条路径就漂);
/// 2. CPU 与 GPU 在同一个 seed 下选出**同一批行** —— exact 模式下模型逐字节相同。
///    ⚠️ 这条最关键:host 在 f32 gpair 上置零,device 在 gradient kernel 里置零,
///    两条完全不同的代码路径必须选出同一批行(靠同一个整数哈希 + 同一个阈值);
/// 3. **所有行都拿到预测** —— 未选中的行不是"不参与这棵树"。
///    如果采样掩码被误当成预测掩码,预测里会出现一批没被这棵树更新过的行,
///    模型质量会塌,而且不会有任何报错。
#[test]
fn subsample_shapes_the_tree_without_masking_predictions() {
    let n_rows = 4000usize;
    let n_feats = 12usize;
    let data: Vec<f32> = (0..n_rows)
        .flat_map(|row| (0..n_feats).map(move |feat| ((row * 7 + feat * 13) % 41) as f32))
        .collect();
    let labels: Vec<f32> = (0..n_rows)
        .map(|row| if (row * 11) % 19 < 9 { 1.0 } else { 0.0 })
        .collect();
    let source = DenseSource::from_row_major(&data, n_rows, n_feats, 32, 4);

    let run = |rate: f32, on_gpu: bool| {
        let params = ferrisboost::types::TrainParams {
            n_rounds: 3,
            max_depth: 4,
            nthread: 1,
            subsample: rate,
            seed: 21,
            ..Default::default()
        };
        let cfg = TrainConfig::new(params, ferrisboost::types::Objective::Logistic);
        if on_gpu {
            train_gpu(0, &source, &labels, &[], &cfg, &mut [], &Local).unwrap()
        } else {
            train(&source, &labels, &[], &cfg, &mut [], &Local).unwrap()
        }
    };
    let bytes = |m: &ferrisboost::tree::Model| serde_json::to_vec(m).unwrap();

    // 1. rate = 1.0 必须和默认路径完全一致。
    let base = run(1.0, false);
    let default_params = ferrisboost::types::TrainParams {
        n_rounds: 3,
        max_depth: 4,
        nthread: 1,
        seed: 21,
        ..Default::default()
    };
    let default_cfg = TrainConfig::new(default_params, ferrisboost::types::Objective::Logistic);
    let untouched = train(&source, &labels, &[], &default_cfg, &mut [], &Local).unwrap();
    assert_eq!(bytes(&base), bytes(&untouched), "subsample=1.0 必须是原来的路径");

    for rate in [0.5f32, 0.25] {
        // 2. CPU 与 GPU 必须选出同一批行 → 模型逐字节相同。
        let cpu = run(rate, false);
        let gpu = run(rate, true);
        assert_eq!(
            bytes(&cpu),
            bytes(&gpu),
            "subsample={rate}:CPU 与 GPU 必须选出同一批行"
        );

        // 3. 采样必须真的改变了树(否则上面两条都是空的)。
        assert_ne!(
            bytes(&cpu),
            bytes(&base),
            "subsample={rate} 应该训出和全量不同的树"
        );

        // 4. **所有行都收到预测**:没有任何一行落在"这棵树没更新过"的状态。
        //    用一个恒等检查:预测必须全部有限,且不是全体相同的常数
        //    (常数意味着树退化 / 大批行没被更新)。
        let preds: Vec<f32> = (0..n_rows)
            .map(|r| cpu.predict(&data[r * n_feats..(r + 1) * n_feats]))
            .collect();
        assert_eq!(preds.len(), n_rows, "每一行都必须有预测");
        assert!(preds.iter().all(|p| p.is_finite()), "预测不能有 NaN/Inf");
        let first = preds[0];
        assert!(
            preds.iter().any(|p| (p - first).abs() > 1e-9),
            "subsample={rate}:预测不该是常数 —— 那说明大批行没拿到这棵树的更新"
        );
    }
}

/// 常驻块的**行主序行距必须来自这个块自己的宽度**,不能拿 buffer 容量反推。
///
/// `slot.bins` 是按**最坏情况**(`max_block_feats`)分配的,所以
/// `bins.len() / n_rows` 恒等于最宽的块。等宽切分时它碰巧等于真实宽度,
/// 一旦**最后一块更窄**(30 特征 / 每块 16 → 16 + 14)且那一块也常驻,
/// 行主序 partition 就会按 16 的行距去读一个行距 14 的块,**整块读偏**。
///
/// 它只在"最后那块也常驻"时发作 —— 而在 residency 还是 env-only、默认 0 的
/// 年代根本没人跑到这个组合,所以藏了很久。现在规划器会主动选满常驻,
/// 这条路成了默认路径。
///
/// 钉的是:**不等宽切分 + 全常驻,模型必须和纯 streaming 逐字节相同。**
#[test]
fn resident_narrow_last_block_keeps_the_model_bit_exact() {
    let n_rows = 3000usize;
    // 30 不能被 16 整除:块宽 16 → [16, 14],最后一块更窄。
    let n_feats = 30usize;
    let data: Vec<f32> = (0..n_rows)
        .flat_map(|row| (0..n_feats).map(move |feat| ((row * 7 + feat * 13) % 61) as f32))
        .collect();
    let labels: Vec<f32> = (0..n_rows)
        .map(|row| if (row * 17) % 29 < 14 { 1.0 } else { 0.0 })
        .collect();

    let run = |cols_per_block: usize, resident: usize| -> Vec<u8> {
        let source = DenseSource::from_row_major(&data, n_rows, n_feats, 64, cols_per_block);
        let params = ferrisboost::types::TrainParams {
            n_rounds: 3,
            max_depth: 5,
            nthread: 1,
            resident_blocks: Some(resident),
            seed: 4,
            ..Default::default()
        };
        let cfg = TrainConfig::new(params, ferrisboost::types::Objective::Logistic);
        serde_json::to_vec(&train_gpu(0, &source, &labels, &[], &cfg, &mut [], &Local).unwrap())
            .unwrap()
    };

    // 16 → [16, 14]:第 2 块比第 1 块窄。
    let streaming = run(16, 0);
    let partial = run(16, 1); // 只常驻等宽的第一块
    let full = run(16, 2); // 连那块更窄的也常驻 —— 曾经在这里读偏
    assert_eq!(streaming, partial, "部分常驻不该改变模型");
    assert_eq!(
        streaming, full,
        "更窄的最后一块常驻之后模型变了 —— 行主序行距用错了(拿 buffer 容量反推,\
         而不是用这个块自己的宽度)"
    );

    // 等宽切分作为对照:这一档**过去也是通过的**,所以它不能证明修复,
    // 只用来说明上面那条失败确实来自"不等宽",不是来自"常驻"本身。
    let even_streaming = run(15, 0); // 30 / 15 = [15, 15]
    let even_full = run(15, 2);
    assert_eq!(even_streaming, even_full, "等宽切分下全常驻本来就该一致");
}
