"""`alpha` / `reg_alpha`、`max_delta_step`、`scale_pos_weight`:语义和 XGBoost 逐位对齐。"""
import json

import numpy as np
import pytest

import ferrisboost as fb

xgb = pytest.importorskip("xgboost")


def low_cardinality(n=3000, m=6, seed=0, missing=0.05):
    # 每个特征的取值数 < max_bin:分箱无损,和 XGBoost 可以逐位比。
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 9, size=(n, m)).astype(np.float32)
    x[rng.random(x.shape) < missing] = np.nan
    s = np.nan_to_num(x[:, 0], nan=4.0) - np.nan_to_num(x[:, 1], nan=4.0) + 0.5 * x[:, 2]
    y = (s + rng.normal(0, 1.5, n) > 1.0).astype(np.float32)
    return x, y


def model_bytes(model, tmp_path, name):
    path = tmp_path / f"{name}.json"
    model.save_model(str(path), quiet=True)
    return path.read_bytes()


def xgb_margin(params, x, y, w, rounds, monotone=None):
    p = {**params, "tree_method": "hist", "nthread": 1}
    if monotone is not None:
        p["monotone_constraints"] = "(" + ",".join(map(str, monotone)) + ")"
    bst = xgb.train(p, xgb.DMatrix(x, label=y, weight=w), num_boost_round=rounds)
    return bst.predict(xgb.DMatrix(x), output_margin=True), bst


REG_CASES = [
    {"alpha": 0.5},
    {"alpha": 5.0},
    {"max_delta_step": 0.3},
    {"max_delta_step": 2.0},
    {"scale_pos_weight": 0.3},
    {"scale_pos_weight": 4.0},
    {"alpha": 1.0, "max_delta_step": 0.5, "scale_pos_weight": 2.5},
]


@pytest.mark.parametrize("extra", REG_CASES, ids=lambda d: ",".join(f"{k}={v}" for k, v in d.items()))
@pytest.mark.parametrize("objective", ["binary:logistic", "reg:squarederror"])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("monotone", [None, (1, -1, 0, 1, 0, 0)])
def test_bit_identical_to_xgboost(extra, objective, weighted, monotone):
    x, y = low_cardinality(seed=3)
    w = np.random.default_rng(3).uniform(0.2, 3, len(y)).astype(np.float32) if weighted else None
    # base_score 不设:spw ≠ 1 时两边都走一步 Newton 估计。
    p = {"max_depth": 5, "eta": 0.3, "objective": objective, "max_bin": 255, **extra}
    ref, _ = xgb_margin(p, x, y, w, 10, monotone)
    fp = {**p, "nthread": 1}
    if monotone is not None:
        fp["monotone_constraints"] = monotone
    ours = fb.train(fp, x, label=y, weight=w, num_boost_round=10, quiet=True)
    np.testing.assert_array_equal(ours.predict(x, output_margin=True), ref)


def test_reg_alpha_alias_matches_alpha(tmp_path):
    x, y = low_cardinality(seed=1)
    a = fb.train({"alpha": 2.0, "nthread": 1}, x, label=y, num_boost_round=4, quiet=True)
    b = fb.train({"reg_alpha": 2.0, "nthread": 1}, x, label=y, num_boost_round=4, quiet=True)
    assert model_bytes(a, tmp_path, "a") == model_bytes(b, tmp_path, "b")


def test_defaults_equal_explicit_neutral_values(tmp_path):
    x, y = low_cardinality(seed=2)
    a = fb.train({"nthread": 1}, x, label=y, num_boost_round=5, quiet=True)
    b = fb.train({"nthread": 1, "alpha": 0.0, "max_delta_step": 0.0, "scale_pos_weight": 1.0},
                 x, label=y, num_boost_round=5, quiet=True)
    assert model_bytes(a, tmp_path, "a") == model_bytes(b, tmp_path, "b")


@pytest.mark.parametrize("spw", [0.3, 4.0])
@pytest.mark.parametrize("weighted", [False, True])
def test_scale_pos_weight_base_score_matches_xgboost(tmp_path, spw, weighted):
    x, y = low_cardinality(n=20_001, seed=5)
    w = np.random.default_rng(5).uniform(0.2, 3, len(y)).astype(np.float32) if weighted else None
    _, bst = xgb_margin({"objective": "binary:logistic", "scale_pos_weight": spw}, x, y, w, 1)
    m = fb.train({"objective": "binary:logistic", "scale_pos_weight": spw}, x, label=y, weight=w,
                 num_boost_round=1, quiet=True)
    path = tmp_path / "m.json"
    m.save_model(str(path), quiet=True)
    ours = json.loads(path.read_text())["learner"]["learner_model_param"]["base_score"]
    ref = json.loads(bst.save_config())["learner"]["learner_model_param"]["base_score"]
    assert np.float32(float(ours.strip("[]"))) == np.float32(float(ref.strip("[]")))


def test_scale_pos_weight_does_not_touch_metrics():
    x, y = low_cardinality(seed=4)
    res = {}
    m = fb.train({"scale_pos_weight": 3.0, "eval_metric": "logloss", "nthread": 1}, x, label=y,
                 evals=[(x, y, "t")], num_boost_round=3, evals_result=res, quiet=True)
    p = np.clip(m.predict(x), 1e-7, 1 - 1e-7).astype(np.float64)
    expected = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))
    assert abs(res["t"]["logloss"][-1] - expected) < 1e-5


def test_max_delta_step_bounds_every_leaf(tmp_path):
    x, y = low_cardinality(seed=6)
    eta, mds = 0.3, 0.2
    m = fb.train({"max_delta_step": mds, "eta": eta, "lambda": 0.0, "nthread": 1}, x, label=y,
                 num_boost_round=5, quiet=True)
    path = tmp_path / "m.json"
    m.save_model(str(path), quiet=True)
    trees = json.loads(path.read_text())["learner"]["gradient_booster"]["model"]["trees"]
    for t in trees:
        leaves = [v for v, l in zip(t["split_conditions"], t["left_children"]) if l == -1]
        assert max(abs(v) for v in leaves) <= mds * eta * (1 + 1e-6)


@pytest.mark.parametrize("bad", [{"alpha": -1.0}, {"max_delta_step": -0.1},
                                 {"scale_pos_weight": -2.0}, {"alpha": float("inf")}])
def test_invalid_values_are_rejected(bad):
    x, y = low_cardinality(n=200)
    with pytest.raises(Exception):
        fb.train(bad, x, label=y, num_boost_round=1, quiet=True)


# ---------------------------------------------------------------- GPU
def _has_cuda():
    try:
        x, y = low_cardinality(64, 3)
        fb.train({"device": "cuda", "max_depth": 2}, x, y, num_boost_round=1, quiet=True)
        return True
    except Exception as e:  # noqa: BLE001
        if "CUDA" in str(e) or "cuda" in str(e):
            return False
        raise


cuda = pytest.mark.skipif(not _has_cuda(), reason="没有可用的 CUDA 构建或设备")


@cuda
@pytest.mark.parametrize("extra", REG_CASES, ids=lambda d: ",".join(f"{k}={v}" for k, v in d.items()))
@pytest.mark.parametrize("split_cache", ["1", "0"])
def test_gpu_exact_matches_cpu(tmp_path, monkeypatch, extra, split_cache):
    monkeypatch.setenv("FB_GPU_SPLIT_CACHE", split_cache)
    x, y = low_cardinality(n=4000, seed=7)
    w = np.random.default_rng(7).uniform(0.2, 3, len(y)).astype(np.float32)
    p = {"max_depth": 6, "eta": 0.3, "subsample": 0.8, "colsample_bytree": 0.8, "seed": 2,
         "monotone_constraints": (1, 0, 0, 0, 0, -1), **extra}
    cpu = fb.train(p, x, label=y, weight=w, num_boost_round=6, quiet=True)
    gpu = fb.train({**p, "device": "cuda", "gpu_math": "exact"}, x, label=y, weight=w,
                   num_boost_round=6, quiet=True)
    assert model_bytes(cpu, tmp_path, "cpu") == model_bytes(gpu, tmp_path, "gpu")
    # 不带单调约束:「给定权重」模式只由 alpha / max_delta_step 打开
    p.pop("monotone_constraints")
    cpu = fb.train(p, x, label=y, weight=w, num_boost_round=6, quiet=True)
    gpu = fb.train({**p, "device": "cuda", "gpu_math": "exact"}, x, label=y, weight=w,
                   num_boost_round=6, quiet=True)
    assert model_bytes(cpu, tmp_path, "cpu2") == model_bytes(gpu, tmp_path, "gpu2")


@cuda
@pytest.mark.parametrize("extra", [{"alpha": 1.0, "max_delta_step": 0.5, "scale_pos_weight": 2.5},
                                   {"scale_pos_weight": 4.0}])
def test_gpu_fast_is_deterministic_and_close(tmp_path, extra):
    x, y = low_cardinality(n=4000, seed=8)
    p = {"max_depth": 6, "eta": 0.3, **extra}
    a = fb.train({**p, "device": "cuda", "gpu_math": "fast"}, x, label=y, num_boost_round=8, quiet=True)
    b = fb.train({**p, "device": "cuda", "gpu_math": "fast"}, x, label=y, num_boost_round=8, quiet=True)
    assert model_bytes(a, tmp_path, "a") == model_bytes(b, tmp_path, "b")
    cpu = fb.train(p, x, label=y, num_boost_round=8, quiet=True)
    np.testing.assert_allclose(a.predict(x), cpu.predict(x), atol=1e-4)


# Split gains without any constraint also use XGBoost's given-weight form in f32
# (each side rounded to f32, summed in f32, minus the f32 parent gain). The f64
# G²/(H+λ) form is algebraically equal but, when two candidates tie in f32
# (large label offsets, many categories), picked a different split than XGBoost.
def _near_tie_numeric():
    rng = np.random.default_rng(99)
    x = rng.integers(0, 40, (3000, 4)).astype(np.float32)
    y = (30 + 0.05 * x[:, 0] - 0.03 * x[:, 1] + rng.normal(0, 0.5, 3000)).astype(np.float32)
    return x, y, None


def _near_tie_categorical():
    rng = np.random.default_rng(47)
    c = rng.integers(0, 30, 3000)
    c[:30] = np.arange(30)
    q = rng.integers(0, 50, 3000).astype(np.float32)
    x = np.stack([q, c.astype(np.float32)], axis=1)
    y = (30 + rng.normal(0, 1, 30)[c] + 0.05 * q + rng.normal(0, 0.5, 3000)).astype(np.float32)
    return x, y, ["q", "c"]


@pytest.mark.parametrize("data", [_near_tie_numeric, _near_tie_categorical],
                         ids=["numeric", "categorical"])
def test_unconstrained_gain_ties_resolve_like_xgboost(data, tmp_path):
    x, y, ft = data()
    p = {"objective": "reg:squarederror", "max_depth": 6, "nthread": 1, "base_score": 0.5}
    d = xgb.DMatrix(x, label=y, feature_types=ft, enable_categorical=ft is not None)
    bst = xgb.train({**p, "tree_method": "hist"}, d, num_boost_round=4)
    ours = fb.train(p, x, label=y, feature_types=ft, num_boost_round=4, quiet=True)
    np.testing.assert_array_equal(ours.predict(x, output_margin=True),
                                  bst.predict(d, output_margin=True))
    if _has_cuda():
        gpu = fb.train({**p, "device": "cuda", "gpu_math": "exact"}, x, label=y,
                       feature_types=ft, num_boost_round=4, quiet=True)
        assert model_bytes(gpu, tmp_path, "g") == model_bytes(ours, tmp_path, "c")
