"""reg:quantileerror 和 quantile metric:和 XGBoost 3.4.1 逐位对齐。"""
import json

import numpy as np
import pytest

import ferrisboost as fb

xgb = pytest.importorskip("xgboost")

ALPHAS = [0.1, 0.5, 0.9, 0.0, 1.0]


def make(n=3000, m=6, seed=0, missing=0.05):
    # 低基数特征(取值数 < max_bin):分箱无损,和 XGBoost 可以逐位比。
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 9, size=(n, m)).astype(np.float32)
    x[rng.random(x.shape) < missing] = np.nan
    mean = 0.7 * np.nan_to_num(x[:, 0], nan=4.0) - 0.4 * np.nan_to_num(x[:, 1], nan=4.0) + 2.0
    y = (mean + rng.standard_t(3, n) * (1.0 + 0.2 * np.nan_to_num(x[:, 2], nan=4.0))).astype(
        np.float32
    )
    return x, y


def xgb_train(params, x, y, w, rounds, evals=()):
    p = {**params, "tree_method": "hist", "nthread": 1}
    d = xgb.DMatrix(x, label=y, weight=w)
    res = {}
    bst = xgb.train(p, d, num_boost_round=rounds,
                    evals=[(xgb.DMatrix(ex, label=ey, weight=ew), name) for ex, ey, ew, name in evals],
                    evals_result=res, verbose_eval=False)
    return bst, res


def saved_base_score(model, tmp_path):
    path = tmp_path / "m.json"
    model.save_model(str(path), quiet=True)
    raw = json.loads(path.read_text())["learner"]["learner_model_param"]["base_score"]
    return np.float32(float(raw.strip("[]")))


@pytest.mark.parametrize("alpha", ALPHAS)
@pytest.mark.parametrize("weighted", [False, True])
def test_bit_identical_to_xgboost(tmp_path, alpha, weighted):
    x, y = make(seed=3)
    w = np.random.default_rng(3).uniform(0.2, 3, len(y)).astype(np.float32) if weighted else None
    p = {"objective": "reg:quantileerror", "quantile_alpha": alpha, "max_depth": 5, "eta": 0.3}
    bst, _ = xgb_train(p, x, y, w, 10)
    ours = fb.train({**p, "nthread": 1}, x, label=y, weight=w, num_boost_round=10, quiet=True)
    ref_bs = json.loads(bst.save_config())["learner"]["learner_model_param"]["base_score"]
    assert saved_base_score(ours, tmp_path) == np.float32(float(ref_bs.strip("[]")))
    np.testing.assert_array_equal(ours.predict(x, output_margin=True),
                                  bst.predict(xgb.DMatrix(x), output_margin=True))
    np.testing.assert_array_equal(ours.predict(x), bst.predict(xgb.DMatrix(x)))


def test_alpha_as_length_one_list_and_explicit_base_score():
    x, y = make(n=500, seed=4)
    a = fb.train({"objective": "reg:quantileerror", "quantile_alpha": 0.3, "nthread": 1,
                  "base_score": 1.5}, x, label=y, num_boost_round=3, quiet=True)
    b = fb.train({"objective": "reg:quantileerror", "quantile_alpha": [0.3], "nthread": 1,
                  "base_score": 1.5}, x, label=y, num_boost_round=3, quiet=True)
    np.testing.assert_array_equal(a.predict(x), b.predict(x))
    bst, _ = xgb_train({"objective": "reg:quantileerror", "quantile_alpha": 0.3,
                        "base_score": 1.5}, x, y, None, 3)
    np.testing.assert_array_equal(a.predict(x), bst.predict(xgb.DMatrix(x)))


@pytest.mark.parametrize("weighted", [False, True])
def test_quantile_metric_matches_xgboost(weighted):
    x, y = make(seed=5)
    vx, vy = make(n=1500, seed=6)
    rng = np.random.default_rng(5)
    w = rng.uniform(0.2, 3, len(y)).astype(np.float32) if weighted else None
    vw = rng.uniform(0.2, 3, len(vy)).astype(np.float32) if weighted else None
    p = {"objective": "reg:quantileerror", "quantile_alpha": 0.8, "max_depth": 4, "eta": 0.3,
         "eval_metric": ["quantile", "mae", "rmse"]}
    _, ref = xgb_train(p, x, y, w, 6, evals=[(vx, vy, vw, "v")])
    ours = {}
    ev = (vx, vy, vw, "v") if weighted else (vx, vy, "v")
    fb.train({**p, "nthread": 1}, x, label=y, weight=w, evals=[ev], num_boost_round=6,
             evals_result=ours, quiet=True)
    assert list(ours["v"]) == list(ref["v"])
    for name in ("quantile", "mae", "rmse"):
        np.testing.assert_allclose(ours["v"][name], ref["v"][name], rtol=2e-6, atol=1e-7,
                                   err_msg=name)


def test_default_metric_is_quantile():
    x, y = make(n=400)
    res = {}
    fb.train({"objective": "reg:quantileerror", "quantile_alpha": 0.5}, x, label=y,
             evals=[(x, y, "t")], num_boost_round=1, evals_result=res, quiet=True)
    assert list(res["t"]) == ["quantile"]


@pytest.mark.parametrize("alpha", [0.2, 0.9])
def test_model_interchange_with_xgboost(tmp_path, alpha):
    x, y = make(seed=7)
    p = {"objective": "reg:quantileerror", "quantile_alpha": alpha, "max_depth": 4}
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
    assert loaded.objective == "reg:quantileerror"


def test_quantile_property_coverage_is_near_alpha():
    # 不依赖 XGBoost 的健全性检查:训练集上低于预测的比例应接近 alpha。
    x, y = make(n=6000, seed=8)
    for alpha in (0.1, 0.5, 0.9):
        m = fb.train({"objective": "reg:quantileerror", "quantile_alpha": alpha, "max_depth": 4,
                      "eta": 0.3}, x, label=y, num_boost_round=40, quiet=True)
        frac = float(np.mean(y <= m.predict(x)))
        assert abs(frac - alpha) < 0.08, (alpha, frac)


def test_multi_quantile_and_missing_alpha_are_rejected():
    x, y = make(n=200)
    with pytest.raises(ValueError, match="多分位数|quantile_alpha"):
        fb.train({"objective": "reg:quantileerror", "quantile_alpha": [0.1, 0.9]}, x, label=y,
                 num_boost_round=1, quiet=True)
    with pytest.raises(ValueError, match="quantile_alpha"):
        fb.train({"objective": "reg:quantileerror"}, x, label=y, num_boost_round=1, quiet=True)
    with pytest.raises(ValueError, match="quantile_alpha"):
        fb.train({"objective": "reg:quantileerror", "quantile_alpha": 1.5}, x, label=y,
                 num_boost_round=1, quiet=True)


def test_quantile_metric_needs_the_quantile_objective():
    x, y = make(n=200)
    with pytest.raises(ValueError, match="quantile"):
        fb.train({"objective": "reg:squarederror", "eval_metric": "quantile"}, x, label=y,
                 num_boost_round=1, quiet=True)


# ---------------------------------------------------------------- GPU
def _has_cuda():
    try:
        x, y = make(64, 3)
        fb.train({"objective": "reg:quantileerror", "quantile_alpha": 0.5, "device": "cuda",
                  "max_depth": 2}, x, y, num_boost_round=1, quiet=True)
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
@pytest.mark.parametrize("alpha", [0.1, 0.5, 0.9])
def test_gpu_exact_matches_cpu(tmp_path, alpha):
    x, y = make(n=4000, seed=11)
    w = np.random.default_rng(11).uniform(0.2, 3, len(y)).astype(np.float32)
    p = {"objective": "reg:quantileerror", "quantile_alpha": alpha, "max_depth": 6,
         "subsample": 0.8, "colsample_bytree": 0.8, "seed": 1}
    cpu = fb.train(p, x, label=y, weight=w, num_boost_round=6, quiet=True)
    gpu = fb.train({**p, "device": "cuda", "gpu_math": "exact"}, x, label=y, weight=w,
                   num_boost_round=6, quiet=True)
    assert model_bytes(cpu, tmp_path, "cpu") == model_bytes(gpu, tmp_path, "gpu")


@cuda
def test_gpu_fast_is_rejected_with_a_clear_error():
    x, y = make(n=500)
    with pytest.raises(Exception, match="gpu_math"):
        fb.train({"objective": "reg:quantileerror", "quantile_alpha": 0.5, "device": "cuda",
                  "gpu_math": "fast"}, x, label=y, num_boost_round=2, quiet=True)
