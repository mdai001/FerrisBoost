//! Learning to rank through the Rust API: query groups, group weights, early
//! stopping on ranking metrics, XGBoost JSON parameters, thread independence.
//! (Bit-exactness against XGBoost is covered by `python/tests/test_ranking.py`.)

use ferrisboost::callback::RecordHistory;
use ferrisboost::comm::Local;
use ferrisboost::rank::{PairMethod, RankKind, RankParams};
use ferrisboost::train::{train, EvalSet, TrainConfig};
use ferrisboost::types::{Objective, TrainParams};
use ferrisboost::{DenseSource, Model};

const SIZES: [usize; 8] = [1, 2, 3, 5, 8, 13, 21, 34];

struct Data {
    x: Vec<f32>,
    y: Vec<f32>,
    group_ptr: Vec<u32>,
    n_features: usize,
}

fn data(seed: u64) -> Data {
    // small deterministic generator
    let mut s = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
    let mut next = move || {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((s >> 33) as f64) / ((1u64 << 31) as f64)
    };
    let n_features = 4;
    let (mut x, mut y, mut group_ptr) = (Vec::new(), Vec::new(), vec![0u32]);
    let mut total = 0u32;
    for &n in &SIZES {
        for _ in 0..n {
            let row: Vec<f32> = (0..n_features).map(|_| (next() * 10.0).floor() as f32).collect();
            let score = f64::from(row[0]) * 0.5 + f64::from(row[1]) * 0.3 + next() * 2.0;
            y.push((score / 2.0).floor().clamp(0.0, 4.0) as f32);
            x.extend(row);
        }
        total += n as u32;
        group_ptr.push(total);
    }
    Data {
        x,
        y,
        group_ptr,
        n_features,
    }
}

fn params(nthread: usize) -> TrainParams {
    TrainParams {
        n_rounds: 6,
        max_depth: 3,
        max_bin: 16,
        nthread,
        ..Default::default()
    }
}

fn config(kind: RankKind, rank: RankParams, nthread: usize, groups: &[u32]) -> TrainConfig {
    let mut c = TrainConfig::new(params(nthread), Objective::Rank { kind, params: rank });
    c.group_ptr = Some(groups.to_vec());
    c
}

fn source(d: &Data) -> DenseSource {
    DenseSource::from_row_major(&d.x, d.y.len(), d.n_features, 16, 2)
}

fn fit(d: &Data, kind: RankKind, rank: RankParams, nthread: usize, weights: Option<&[f32]>) -> Model {
    let src = source(d);
    let cfg = config(kind, rank, nthread, &d.group_ptr);
    ferrisboost::train::train_weighted(&src, &d.y, weights, &[], &cfg, &mut [], &Local).unwrap()
}

fn predictions(m: &Model, d: &Data) -> Vec<f32> {
    d.x.chunks(d.n_features).map(|r| m.predict_margin(r)).collect()
}

#[test]
fn every_kind_and_pair_method_trains_and_is_thread_independent() {
    let d = data(1);
    let binary = Data {
        y: d.y.iter().map(|&v| f32::from(v > 1.0)).collect(),
        x: d.x.clone(),
        group_ptr: d.group_ptr.clone(),
        n_features: d.n_features,
    };
    for kind in [RankKind::Ndcg, RankKind::Pairwise, RankKind::Map] {
        for method in [PairMethod::TopK, PairMethod::Mean] {
            let rank = RankParams {
                pair_method: method,
                ..RankParams::default()
            };
            let data = if kind == RankKind::Map { &binary } else { &d };
            let one = fit(data, kind, rank, 1, None);
            let four = fit(data, kind, rank, 4, None);
            assert_eq!(
                predictions(&one, data)
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                predictions(&four, data)
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                "{kind:?} {method:?}"
            );
            assert_eq!(one.trees.len(), 6);
            assert_eq!(one.objective.name(), kind.name());
        }
    }
}

#[test]
fn training_improves_the_ranking_metric_and_records_it() {
    let d = data(2);
    let src = source(&d);
    let cfg = config(RankKind::Ndcg, RankParams::default(), 1, &d.group_ptr);
    let hist = RecordHistory::default();
    let handle = hist.handle();
    let mut cbs: Vec<Box<dyn ferrisboost::Callback>> = vec![Box::new(hist)];
    let eval = EvalSet {
        name: "t".into(),
        source: &src,
        labels: &d.y,
        weights: None,
        group_ptr: Some(&d.group_ptr),
    };
    let m = train(&src, &d.y, &[eval], &cfg, &mut cbs, &Local).unwrap();
    assert_eq!(m.trees.len(), 6);
    let rounds = handle.lock().unwrap().clone();
    assert_eq!(rounds[0].entries[0].1, "ndcg@32");
    let first = rounds[0].entries[0].2;
    let last = rounds[5].entries[0].2;
    assert!(last > first, "ndcg should improve: {first} -> {last}");
}

#[test]
fn group_weights_scale_gradients_and_must_match_the_group_count() {
    let d = data(3);
    let n_groups = d.group_ptr.len() - 1;
    let uniform = vec![1.0f32; n_groups];
    let plain = fit(&d, RankKind::Pairwise, RankParams::default(), 1, None);
    let ones = fit(&d, RankKind::Pairwise, RankParams::default(), 1, Some(&uniform));
    assert_eq!(predictions(&plain, &d), predictions(&ones, &d));
    let varying: Vec<f32> = (0..n_groups).map(|g| 0.5 + g as f32 * 0.25).collect();
    let weighted = fit(&d, RankKind::Pairwise, RankParams::default(), 1, Some(&varying));
    assert_ne!(predictions(&plain, &d), predictions(&weighted, &d));
    // one weight per row is an error for ranking
    let src = source(&d);
    let cfg = config(RankKind::Pairwise, RankParams::default(), 1, &d.group_ptr);
    let per_row = vec![1.0f32; d.y.len()];
    let err = ferrisboost::train::train_weighted(&src, &d.y, Some(&per_row), &[], &cfg, &mut [], &Local)
        .unwrap_err()
        .to_string();
    assert!(err.contains("权重"), "{err}");
}

#[test]
fn groups_are_required_and_validated() {
    let d = data(4);
    let src = source(&d);
    let rank = Objective::Rank {
        kind: RankKind::Ndcg,
        params: RankParams::default(),
    };
    let none = TrainConfig::new(params(1), rank);
    let err = train(&src, &d.y, &[], &none, &mut [], &Local).unwrap_err().to_string();
    assert!(err.contains("needs query groups"), "{err}");
    let mut short = TrainConfig::new(params(1), rank);
    short.group_ptr = Some(vec![0, 3, 5]);
    let err = train(&src, &d.y, &[], &short, &mut [], &Local).unwrap_err().to_string();
    assert!(err.contains("group sizes sum"), "{err}");
    let mut other = TrainConfig::new(params(1), Objective::SquaredError);
    other.group_ptr = Some(d.group_ptr.clone());
    let err = train(&src, &d.y, &[], &other, &mut [], &Local).unwrap_err().to_string();
    assert!(err.contains("only used by the rank"), "{err}");
    // an evaluation set needs its own groups
    let cfg = config(RankKind::Ndcg, RankParams::default(), 1, &d.group_ptr);
    let eval = EvalSet {
        name: "v".into(),
        source: &src,
        labels: &d.y,
        weights: None,
        group_ptr: None,
    };
    let err = train(&src, &d.y, &[eval], &cfg, &mut [], &Local).unwrap_err().to_string();
    assert!(err.contains("needs query groups"), "{err}");
}

#[test]
fn objective_parameters_round_trip_through_the_xgboost_json() {
    let params = RankParams {
        pair_method: PairMethod::Mean,
        num_pair_per_sample: Some(3),
        normalization: false,
        score_normalization: false,
        ndcg_exp_gain: false,
        ..RankParams::default()
    };
    let o = Objective::Rank {
        kind: RankKind::Map,
        params,
    };
    let json = o.json_params(1.0);
    let back = Objective::from_json("rank:map", &json).unwrap();
    assert_eq!(back, o);
    let block = &json["lambdarank_param"];
    assert_eq!(block["lambdarank_pair_method"], "mean");
    assert_eq!(block["lambdarank_num_pair_per_sample"], "3");
    assert_eq!(block["lambdarank_normalization"], "0");
    assert_eq!(block["ndcg_exp_gain"], "0");
    // XGBoost writes an unset pair count as the unsigned maximum
    let default = Objective::Rank {
        kind: RankKind::Ndcg,
        params: RankParams::default(),
    }
    .json_params(1.0);
    assert_eq!(
        default["lambdarank_param"]["lambdarank_num_pair_per_sample"],
        "4294967295"
    );
    // and a trained model keeps them through save / load
    let d = data(5);
    let m = fit(&d, RankKind::Ndcg, params, 1, None);
    let again = Model::from_xgboost_json(&m.to_xgboost_json().unwrap()).unwrap();
    assert_eq!(again.objective, m.objective);
    assert_eq!(predictions(&again, &d), predictions(&m, &d));
}

#[test]
fn ranking_metric_validation_follows_xgboost() {
    let d = data(6);
    let src = source(&d);
    // fractional relevance needs a linear gain
    let fractional: Vec<f32> = d.y.iter().map(|v| v + 0.5).collect();
    let cfg = config(RankKind::Ndcg, RankParams::default(), 1, &d.group_ptr);
    let err = train(&src, &fractional, &[], &cfg, &mut [], &Local)
        .unwrap_err()
        .to_string();
    assert!(err.contains("integers"), "{err}");
    let linear = RankParams {
        ndcg_exp_gain: false,
        ..RankParams::default()
    };
    let cfg = config(RankKind::Ndcg, linear, 1, &d.group_ptr);
    train(&src, &fractional, &[], &cfg, &mut [], &Local).unwrap();
    // rank:map and the map metric need binary labels
    let cfg = config(RankKind::Map, RankParams::default(), 1, &d.group_ptr);
    let err = train(&src, &d.y, &[], &cfg, &mut [], &Local).unwrap_err().to_string();
    assert!(err.contains("binary"), "{err}");
    // unbiased LambdaMART is rejected with a clear message
    let unbiased = RankParams {
        unbiased: true,
        ..RankParams::default()
    };
    let cfg = config(RankKind::Ndcg, unbiased, 1, &d.group_ptr);
    let err = train(&src, &d.y, &[], &cfg, &mut [], &Local).unwrap_err().to_string();
    assert!(err.contains("unbiased"), "{err}");
}
