"""`max_bin` above 255 (16-bit column blocks for the features that need them).

Where binning is lossless (every feature has fewer distinct values than
`max_bin`) the trees and predictions are bit-identical to XGBoost 3.4.1
`tree_method="hist"` with the same `max_bin`, for features with up to ~1000
bins. The same holds at `max_bin=255` for the 8-bit features of the same data,
so mixed-width training is not a different algorithm.
"""
import json

import numpy as np
import pytest

import ferrisboost as fb

xgb = pytest.importorskip("xgboost")

BASE = {"max_depth": 5, "nthread": 1, "base_score": 0.5, "eta": 0.3}
# Distinct values per feature: wide (> 255) features between 8-bit ones.
DISTINCT = [8, 600, 40, 300, 1000, 5]


def make(n=4000, seed=0, objective="reg:squarederror"):
    rng = np.random.default_rng(seed)
    x = np.stack([rng.integers(0, d, n) for d in DISTINCT], axis=1).astype(np.float32)
    x = x * 0.25 - 3.0
    x[rng.random(x.shape) < 0.02] = np.nan
    v = np.nan_to_num(x)
    s = (v[:, 1] > 70) + 0.5 * (v[:, 4] > 200) - (v[:, 3] > 40) + 0.1 * v[:, 0]
    s = s + 0.3 * rng.standard_normal(n)
    y = (s > 0.6).astype(np.float32) if objective == "binary:logistic" else s.astype(np.float32)
    return x, y


def trees(model, tmp_path, name):
    path = str(tmp_path / f"{name}.json")
    model.save_model(path, quiet=True)
    with open(path) as f:
        return json.load(f)["learner"]["gradient_booster"]["model"]["trees"]


def struct_key(tree):
    # FerrisBoost stores split thresholds as cut midpoints and writes no
    # base_weights; compare structure here, leaf values via exact predictions.
    return {k: tree[k] for k in ("left_children", "right_children", "split_indices",
                                 "default_left")}


@pytest.mark.parametrize("objective", ["reg:squarederror", "binary:logistic"])
@pytest.mark.parametrize("max_bin", [1024, 4096])
def test_bit_exact_vs_xgboost_with_wide_bins(objective, max_bin, tmp_path):
    x, y = make(objective=objective)
    params = {**BASE, "objective": objective, "max_bin": max_bin}
    bst = xgb.train({"tree_method": "hist", **params}, xgb.DMatrix(x, label=y),
                    num_boost_round=8)
    ours = fb.train(params, x, label=y, num_boost_round=8, quiet=True)
    ref = json.loads(bst.save_raw("json"))["learner"]["gradient_booster"]["model"]["trees"]
    mine = trees(ours, tmp_path, "m")
    assert len(mine) == len(ref)
    for i, (u, v) in enumerate(zip(mine, ref)):
        assert struct_key(u) == struct_key(v), f"tree {i} differs"
    np.testing.assert_array_equal(
        ours.predict(x, output_margin=True),
        bst.predict(xgb.DMatrix(x), output_margin=True),
    )
    # The wide features really are split above the 8-bit range somewhere.
    wide = {1, 3, 4}
    assert any(
        idx in wide and c > 255 * 0.25 - 3.0
        for t in mine
        for idx, c, left in zip(t["split_indices"], t["split_conditions"], t["left_children"])
        if left != -1
    )


def test_wide_model_round_trips_and_predicts(tmp_path):
    x, y = make(seed=3, objective="binary:logistic")
    params = {**BASE, "objective": "binary:logistic", "max_bin": 1024}
    model = fb.train(params, x, label=y, num_boost_round=5, quiet=True)
    path = tmp_path / "wide.json"
    model.save_model(str(path), quiet=True)
    loaded = fb.Model.load_model(str(path), quiet=True)
    np.testing.assert_array_equal(loaded.predict(x), model.predict(x))
    # XGBoost reads the same JSON and predicts the same margins.
    bst = xgb.Booster(model_file=str(path))
    np.testing.assert_array_equal(
        bst.predict(xgb.DMatrix(x), output_margin=True),
        model.predict(x, output_margin=True),
    )


def test_max_bin_up_to_255_is_unchanged(tmp_path):
    # Below the 16-bit tier nothing changes: same model as before the change
    # would give, checked here against XGBoost at the 8-bit limit.
    x, y = make(seed=5)
    x = x[:, [0, 2, 5]]
    params = {**BASE, "objective": "reg:squarederror", "max_bin": 255}
    bst = xgb.train({"tree_method": "hist", **params}, xgb.DMatrix(x, label=y),
                    num_boost_round=6)
    ours = fb.train(params, x, label=y, num_boost_round=6, quiet=True)
    np.testing.assert_array_equal(
        ours.predict(x, output_margin=True),
        bst.predict(xgb.DMatrix(x), output_margin=True),
    )
