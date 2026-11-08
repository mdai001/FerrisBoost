"""count:poisson / reg:gamma / reg:tweedie / reg:pseudohubererror 和它们的 metric:和 XGBoost 逐位对齐。"""
import json

import numpy as np
import pytest

import ferrisboost as fb

xgb = pytest.importorskip("xgboost")


def make(objective, n=3000, m=6, seed=0, missing=0.05):
    # 低基数特征(取值数 < max_bin):分箱无损,和 XGBoost 可以逐位比。
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 9, size=(n, m)).astype(np.float32)
    x[rng.random(x.shape) < missing] = np.nan
    eta = 0.15 * np.nan_to_num(x[:, 0], nan=4.0) - 0.1 * np.nan_to_num(x[:, 1], nan=4.0) + 0.3
    if objective == "count:poisson":
        y = rng.poisson(np.exp(eta)).astype(np.float32)
    elif objective == "reg:gamma":
        y = rng.gamma(2.0, np.exp(eta) / 2.0).astype(np.float32) + np.float32(1e-3)
    elif objective == "reg:tweedie":
        # 复合泊松-伽马:约 40% 的 0,其余为正
        counts = rng.poisson(np.exp(eta - 0.5))
        y = np.array([rng.gamma(2.0, 0.7, c).sum() for c in counts], dtype=np.float32)
    else:  # pseudohuber:重尾噪声
        y = (3.0 + eta * 4 + rng.standard_t(2, n)).astype(np.float32)
    return x, y


OBJECTIVES = [
    ("count:poisson", {}),
    ("count:poisson", {"max_delta_step": 0.3}),
    ("reg:gamma", {}),
    ("reg:tweedie", {}),
    ("reg:tweedie", {"tweedie_variance_power": 1.2}),
    ("reg:pseudohubererror", {}),
    ("reg:pseudohubererror", {"huber_slope": 2.5}),
]
IDS = [o + "".join(f",{k}={v}" for k, v in p.items()) for o, p in OBJECTIVES]


def xgb_train(params, x, y, w, rounds, monotone=None, evals=()):
    p = {**params, "tree_method": "hist", "nthread": 1}
    if monotone is not None:
        p["monotone_constraints"] = "(" + ",".join(map(str, monotone)) + ")"
    d = xgb.DMatrix(x, label=y, weight=w)
    res = {}
    bst = xgb.train(p, d, num_boost_round=rounds,
                    evals=[(xgb.DMatrix(ex, label=ey, weight=ew), name) for ex, ey, ew, name in evals],
                    evals_result=res, verbose_eval=False)
    return bst, res


def fb_base_score(model, tmp_path):
    path = tmp_path / "m.json"
    model.save_model(str(path), quiet=True)
    raw = json.loads(path.read_text())["learner"]["learner_model_param"]["base_score"]
    return np.float32(float(raw.strip("[]")))


@pytest.mark.parametrize("objective,extra", OBJECTIVES, ids=IDS)
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("monotone", [None, (1, -1, 0, 0, 0, 0)])
def test_bit_identical_to_xgboost(tmp_path, objective, extra, weighted, monotone):
    x, y = make(objective, seed=3)
    w = np.random.default_rng(3).uniform(0.2, 3, len(y)).astype(np.float32) if weighted else None
    p = {"objective": objective, "max_depth": 5, "eta": 0.3, "max_bin": 255, **extra}
    bst, _ = xgb_train(p, x, y, w, 10, monotone)
    fp = {**p, "nthread": 1}
    if monotone is not None:
        fp["monotone_constraints"] = monotone
    ours = fb.train(fp, x, label=y, weight=w, num_boost_round=10, quiet=True)
    ref_bs = json.loads(bst.save_config())["learner"]["learner_model_param"]["base_score"]
    assert fb_base_score(ours, tmp_path) == np.float32(float(ref_bs.strip("[]")))
    np.testing.assert_array_equal(ours.predict(x, output_margin=True),
                                  bst.predict(xgb.DMatrix(x), output_margin=True))
    np.testing.assert_array_equal(ours.predict(x), bst.predict(xgb.DMatrix(x)))


METRICS = {
    "count:poisson": ["poisson-nloglik", "rmse", "mae"],
    "reg:gamma": ["gamma-deviance", "gamma-nloglik", "mape", "rmse"],
    "reg:tweedie": ["tweedie-nloglik@1.5", "tweedie-nloglik@1.2", "mae"],
    "reg:pseudohubererror": ["mphe", "mae", "rmse"],
}


@pytest.mark.parametrize("objective", list(METRICS))
@pytest.mark.parametrize("weighted", [False, True])
def test_metrics_match_xgboost(objective, weighted):
    x, y = make(objective, seed=5)
    vx, vy = make(objective, n=1500, seed=6)
    rng = np.random.default_rng(5)
    w = rng.uniform(0.2, 3, len(y)).astype(np.float32) if weighted else None
    vw = rng.uniform(0.2, 3, len(vy)).astype(np.float32) if weighted else None
    metrics = METRICS[objective]
    p = {"objective": objective, "max_depth": 4, "eta": 0.3, "max_bin": 255, "eval_metric": metrics}
    _, ref = xgb_train(p, x, y, w, 6, evals=[(vx, vy, vw, "v")])
    ours = {}
    ev = (vx, vy, vw, "v") if weighted else (vx, vy, "v")
    fb.train({**p, "nthread": 1}, x, label=y, weight=w, evals=[ev], num_boost_round=6,
             evals_result=ours, quiet=True)
    assert list(ours["v"]) == list(ref["v"])
    for name in metrics:
        np.testing.assert_allclose(ours["v"][name], ref["v"][name], rtol=2e-6, atol=1e-7,
                                   err_msg=name)


def test_default_metric_follows_xgboost():
    names = {}
    for objective in ["count:poisson", "reg:gamma", "reg:tweedie", "reg:pseudohubererror"]:
        x, y = make(objective, n=400)
        res = {}
        fb.train({"objective": objective}, x, label=y, evals=[(x, y, "t")], num_boost_round=1,
                 evals_result=res, quiet=True)
        names[objective] = list(res["t"])
    assert names == {"count:poisson": ["poisson-nloglik"], "reg:gamma": ["gamma-deviance"],
                     "reg:tweedie": ["tweedie-nloglik@1.5"], "reg:pseudohubererror": ["mphe"]}


@pytest.mark.parametrize("objective,extra", OBJECTIVES, ids=IDS)
def test_model_interchange_with_xgboost(tmp_path, objective, extra):
    x, y = make(objective, seed=7)
    p = {"objective": objective, "max_depth": 4, **extra}
    ours = fb.train({**p, "nthread": 1}, x, label=y, num_boost_round=5, quiet=True)
    path = tmp_path / "fb.json"
    ours.save_model(str(path), quiet=True)
    bst = xgb.Booster(model_file=str(path))
    np.testing.assert_allclose(bst.predict(xgb.DMatrix(x)), ours.predict(x), rtol=1e-6)
    ref, _ = xgb_train(p, x, y, None, 5)
    ref_path = tmp_path / "xgb.json"
    ref.save_model(str(ref_path))
    loaded = fb.Model.load_model(str(ref_path), quiet=True)
    np.testing.assert_array_equal(loaded.predict(x), ref.predict(xgb.DMatrix(x)))


@pytest.mark.parametrize("objective,bad", [
    ("count:poisson", -1.0), ("reg:tweedie", -0.5), ("reg:gamma", 0.0), ("reg:gamma", -2.0)])
def test_invalid_labels_are_rejected(objective, bad):
    x, y = make(objective, n=200)
    y = y.copy()
    y[3] = bad
    with pytest.raises(Exception, match="labels must be"):
        fb.train({"objective": objective}, x, label=y, num_boost_round=1, quiet=True)


@pytest.mark.parametrize("params", [
    {"objective": "reg:tweedie", "tweedie_variance_power": 2.0},
    {"objective": "reg:tweedie", "tweedie_variance_power": 0.9},
    {"objective": "reg:pseudohubererror", "huber_slope": 0.0},
    {"objective": "count:poisson", "max_delta_step": -1.0},
])
def test_invalid_objective_params_are_rejected(params):
    x, y = make(params["objective"], n=200)
    with pytest.raises(ValueError):
        fb.train(params, x, label=y, num_boost_round=1, quiet=True)


def test_all_zero_poisson_labels_need_explicit_base_score():
    x, _ = make("count:poisson", n=200)
    y = np.zeros(len(x), np.float32)
    with pytest.raises(Exception, match="base_score"):
        fb.train({"objective": "count:poisson"}, x, label=y, num_boost_round=1, quiet=True)
    m = fb.train({"objective": "count:poisson", "base_score": 0.1}, x, label=y,
                 num_boost_round=2, quiet=True)
    assert np.isfinite(m.predict(x)).all()


def test_file_input_matches_arrays(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    x, y = make("count:poisson", seed=9)
    pq.write_table(pa.table({**{f"f{i}": x[:, i] for i in range(x.shape[1])}, "y": y}),
                   str(tmp_path / "t.parquet"))
    p = {"objective": "count:poisson", "nthread": 1}
    a = fb.train(p, str(tmp_path / "t.parquet"), label="y", num_boost_round=5, quiet=True)
    b = fb.train(p, x, label=y, num_boost_round=5, quiet=True)
    np.testing.assert_array_equal(a.predict(x), b.predict(x))


# ---------------------------------------------------------------- GPU
def _has_cuda():
    try:
        x, y = make("reg:gamma", 64, 3)
        fb.train({"objective": "reg:gamma", "device": "cuda", "max_depth": 2}, x, y,
                 num_boost_round=1, quiet=True)
        return True
    except Exception as e:  # noqa: BLE001
        if "CUDA" in str(e) or "cuda" in str(e):
            return False
        raise


cuda = pytest.mark.skipif(not _has_cuda(), reason="没有可用的 CUDA 构建或设备")


def model_bytes(model, tmp_path, name):
    path = tmp_path / f"{name}.json"
    model.save_model(str(path), quiet=True)
    return path.read_bytes()


@cuda
@pytest.mark.parametrize("objective,extra", OBJECTIVES, ids=IDS)
def test_gpu_exact_matches_cpu(tmp_path, objective, extra):
    x, y = make(objective, n=4000, seed=11)
    w = np.random.default_rng(11).uniform(0.2, 3, len(y)).astype(np.float32)
    p = {"objective": objective, "max_depth": 6, "subsample": 0.8, "colsample_bytree": 0.8,
         "seed": 1, **extra}
    cpu = fb.train(p, x, label=y, weight=w, num_boost_round=6, quiet=True)
    gpu = fb.train({**p, "device": "cuda", "gpu_math": "exact"}, x, label=y, weight=w,
                   num_boost_round=6, quiet=True)
    assert model_bytes(cpu, tmp_path, "cpu") == model_bytes(gpu, tmp_path, "gpu")


@cuda
@pytest.mark.parametrize("objective,extra", OBJECTIVES, ids=IDS)
def test_gpu_fast_is_deterministic_and_close_to_cpu(tmp_path, objective, extra):
    x, y = make(objective, n=4000, seed=12)
    w = np.random.default_rng(12).uniform(0.2, 3, len(y)).astype(np.float32)
    p = {"objective": objective, "max_depth": 6, **extra}
    a = fb.train({**p, "device": "cuda", "gpu_math": "fast"}, x, label=y, weight=w,
                 num_boost_round=8, quiet=True)
    b = fb.train({**p, "device": "cuda", "gpu_math": "fast"}, x, label=y, weight=w,
                 num_boost_round=8, quiet=True)
    assert model_bytes(a, tmp_path, "a") == model_bytes(b, tmp_path, "b")
    cpu = fb.train(p, x, label=y, weight=w, num_boost_round=8, quiet=True)
    np.testing.assert_allclose(a.predict(x), cpu.predict(x), rtol=1e-4, atol=1e-5)
