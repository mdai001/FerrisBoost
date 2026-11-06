"""默认 base_score:和 XGBoost 3.x 一样从训练标签估计(加权标签均值)。"""
import json

import numpy as np
import pytest

import ferrisboost as fb

xgb = pytest.importorskip("xgboost")


def fb_base_score(model, tmp_path):
    p = tmp_path / "m.json"
    model.save_model(str(p), quiet=True)
    raw = json.loads(p.read_text())["learner"]["learner_model_param"]["base_score"]
    return np.float32(float(raw.strip("[]")))


def xgb_base_score(booster):
    raw = json.loads(booster.save_config())["learner"]["learner_model_param"]["base_score"]
    return np.float32(float(raw.strip("[]")))


def data(n, seed, objective):
    rng = np.random.default_rng(seed)
    # 低基数特征:分箱无损,模型可以逐位对拍
    x = rng.integers(0, 12, size=(n, 4)).astype(np.float32)
    signal = x[:, 0] - 0.5 * x[:, 1] + rng.normal(0, 2, n)
    if objective == "binary:logistic":
        y = (signal > np.quantile(signal, 0.7)).astype(np.float32)
    else:
        y = (3.0 + 0.4 * signal).astype(np.float32)
    w = rng.uniform(0.2, 3.0, n).astype(np.float32)
    return x, y, w


@pytest.mark.parametrize("objective", ["binary:logistic", "reg:squarederror"])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("n", [37, 5000, 200_003])
def test_estimated_base_score_matches_xgboost_bitwise(tmp_path, objective, weighted, n):
    x, y, w = data(n, n, objective)
    kw = {"weight": w} if weighted else {}
    m = fb.train({"objective": objective, "max_depth": 2, "nthread": 0}, x, label=y,
                 num_boost_round=1, quiet=True, **kw)
    b = xgb.train({"objective": objective, "max_depth": 2, "tree_method": "hist", "nthread": 1},
                  xgb.DMatrix(x, label=y, weight=w if weighted else None), 1)
    assert fb_base_score(m, tmp_path) == xgb_base_score(b)


@pytest.mark.parametrize("objective", ["binary:logistic", "reg:squarederror"])
@pytest.mark.parametrize("weighted", [False, True])
def test_default_base_score_models_match_xgboost(objective, weighted):
    x, y, w = data(3000, 7, objective)
    p = {"objective": objective, "max_depth": 4, "eta": 0.3, "max_bin": 255}
    kw = {"weight": w} if weighted else {}
    ours = fb.train({**p, "nthread": 1}, x, label=y, num_boost_round=8, quiet=True, **kw)
    bst = xgb.train({**p, "tree_method": "hist", "nthread": 1},
                    xgb.DMatrix(x, label=y, weight=w if weighted else None), 8)
    np.testing.assert_array_equal(ours.predict(x, output_margin=True),
                                  bst.predict(xgb.DMatrix(x), output_margin=True))


def test_estimate_is_thread_invariant(tmp_path):
    x, y, w = data(200_003, 3, "reg:squarederror")
    got = {fb_base_score(fb.train({"objective": "reg:squarederror", "nthread": t, "max_depth": 1},
                                  x, label=y, weight=w, num_boost_round=1, quiet=True), tmp_path)
           for t in (1, 2, 7)}
    assert len(got) == 1


def test_explicit_base_score_is_kept(tmp_path):
    x, y, _ = data(500, 1, "binary:logistic")
    m = fb.train({"objective": "binary:logistic", "base_score": 0.5}, x, label=y,
                 num_boost_round=1, quiet=True)
    assert fb_base_score(m, tmp_path) == np.float32(0.5)


def test_constant_logistic_labels_match_xgboost():
    x, _, _ = data(400, 2, "binary:logistic")
    y = np.ones(len(x), np.float32)
    ours = fb.train({"objective": "binary:logistic", "nthread": 1}, x, label=y,
                    num_boost_round=2, quiet=True)
    bst = xgb.train({"objective": "binary:logistic", "tree_method": "hist", "nthread": 1},
                    xgb.DMatrix(x, label=y), 2)
    np.testing.assert_array_equal(ours.predict(x, output_margin=True),
                                  bst.predict(xgb.DMatrix(x), output_margin=True))


def test_file_input_estimates_from_the_label_column(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    x, y, w = data(2000, 5, "reg:squarederror")
    cols = {f"f{i}": x[:, i] for i in range(4)}
    pq.write_table(pa.table({**cols, "y": y, "w": w}), str(tmp_path / "t.parquet"))
    from_file = fb.train({"objective": "reg:squarederror"}, str(tmp_path / "t.parquet"),
                         label="y", weight="w", num_boost_round=1, quiet=True)
    from_array = fb.train({"objective": "reg:squarederror"}, x, label=y, weight=w,
                          num_boost_round=1, quiet=True)
    assert fb_base_score(from_file, tmp_path) == fb_base_score(from_array, tmp_path)
