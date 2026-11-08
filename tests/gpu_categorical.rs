#![cfg(feature = "cuda")]
//! Categorical features on the GPU: partition by category set, and the exact
//! mode staying byte-identical to the CPU.

use ferrisboost::{
    backend::cuda::partition_rows_gpu_categorical,
    categorical::CatBits,
    columns::BinningStrategy,
    comm::Local,
    source::DenseSource,
    train::{train, train_gpu, TrainConfig},
    types::{GpuMath, Objective, RowId, TrainParams, MISSING_BIN},
};

#[test]
fn gpu_categorical_partition_matches_stable_cpu_order() {
    let n_rows = 200_003usize;
    // Reverse order checks stability independent of ascending input.
    let rows: Vec<RowId> = (0..n_rows as RowId).rev().collect();
    let bins: Vec<u8> = (0..n_rows)
        .map(|row| {
            if row % 53 == 0 {
                MISSING_BIN
            } else {
                ((row * 7 + row / 11) % 200) as u8
            }
        })
        .collect();
    let mut set = CatBits::empty();
    for c in [0u32, 3, 31, 32, 63, 64, 127, 128, 199] {
        set.set(c);
    }
    for missing_left in [false, true] {
        for threads in [8u32, 512] {
            let (left, right) = partition_rows_gpu_categorical(
                0,
                &bins,
                n_rows,
                0,
                &rows,
                &set,
                missing_left,
                threads,
            )
            .unwrap();
            let goes_left = |&row: &RowId| {
                let bin = bins[row as usize];
                if bin == MISSING_BIN {
                    missing_left
                } else {
                    !set.contains(bin as u32)
                }
            };
            let want_left: Vec<_> = rows.iter().filter(|r| goes_left(r)).copied().collect();
            let want_right: Vec<_> = rows.iter().filter(|r| !goes_left(r)).copied().collect();
            assert!(want_left.len() > 1000 && want_right.len() > 1000);
            assert_eq!(left, want_left);
            assert_eq!(right, want_right);
        }
    }
}

fn data(n: usize) -> (Vec<f32>, Vec<f32>, usize, Vec<bool>) {
    let mut x = Vec::new();
    let mut y = Vec::new();
    for i in 0..n {
        let a = ((i * 7) % 9) as f32;
        let b = if i % 11 == 0 {
            f32::NAN
        } else {
            ((i * 5) % 6) as f32
        };
        let c1 = ((i * 3 + i / 7) % 5) as f32;
        let c2 = if i % 13 == 0 {
            f32::NAN
        } else {
            ((i * 17 + i / 5) % 40) as f32
        };
        let up = matches!(c1 as i32, 1 | 3) || matches!(c2 as i32, 2 | 5 | 9 | 21 | 33);
        x.extend([a, b, c1, c2]);
        y.push(if up { 1.0 } else { 0.0 } + 0.05 * a);
    }
    (x, y, 4, vec![false, false, true, true])
}

fn config(cols_per_block: usize, math: GpuMath) -> TrainConfig {
    let mut cfg = TrainConfig::new(
        TrainParams {
            n_rounds: 5,
            max_depth: 5,
            max_bin: 32,
            cols_per_block,
            nthread: 2,
            gpu_math: math,
            ..Default::default()
        },
        Objective::SquaredError,
    );
    cfg.quiet = true;
    cfg
}

fn source(x: &[f32], n: usize, nf: usize, flags: &[bool], cols: usize) -> DenseSource {
    DenseSource::new(
        x,
        n,
        nf,
        BinningStrategy::SketchTyped {
            max_bins: 32,
            categorical: flags,
        },
        cols,
    )
}

#[test]
fn gpu_exact_categorical_model_is_byte_identical_to_cpu() {
    let (x, y, nf, flags) = data(3000);
    for cols in [1usize, 2, 4] {
        let src = source(&x, y.len(), nf, &flags, cols);
        let cfg = config(cols, GpuMath::Exact);
        let cpu = train(&src, &y, &[], &cfg, &mut [], &Local).unwrap();
        let gpu = train_gpu(0, &src, &y, &[], &cfg, &mut [], &Local).unwrap();
        assert!(cpu.has_categorical_splits());
        assert_eq!(
            cpu.to_xgboost_json().unwrap(),
            gpu.to_xgboost_json().unwrap(),
            "cols_per_block {cols}"
        );
    }
}

#[test]
fn gpu_fast_categorical_is_deterministic() {
    let (x, y, nf, flags) = data(3000);
    let src = source(&x, y.len(), nf, &flags, 2);
    let cfg = config(2, GpuMath::Fast);
    let a = train_gpu(0, &src, &y, &[], &cfg, &mut [], &Local).unwrap();
    let b = train_gpu(0, &src, &y, &[], &cfg, &mut [], &Local).unwrap();
    assert!(a.has_categorical_splits());
    assert_eq!(a.to_xgboost_json().unwrap(), b.to_xgboost_json().unwrap());
}
