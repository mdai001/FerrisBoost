"""`aucpr` eval metric (binary:logistic): value parity with XGBoost, weights, ties, early stopping."""
import re

import numpy as np
import pytest

import ferrisboost as fb

xgb = pytest.importorskip("xgboost")

BASE = {"objective": "binary:logistic", "max_depth": 3, "eta": 0.3, "nthread": 1}


def toy(n=400, m=5, seed=0, card=8):
    # Integer features with few distinct values: binning is lossless and predictions
    # contain many exact ties.
    rng = np.random.default_rng(seed)
    x = rng.integers(0, card, size=(n, m)).astype(np.float32)
    y = (x[:, 0] - x[:, 1] + rng.normal(0, 1.0, n) > 0).astype(np.float32)
    w = rng.uniform(0.1, 3.0, n).astype(np.float32)
    return x, y, w


def xgb_aucpr_per_round(model, tmp_path, xv, yv, wv, rounds):
    path = tmp_path / "m.json"
    model.save_model(str(path), quiet=True)
    bst = xgb.Booster(model_file=str(path))
    bst.set_param([("eval_metric", "aucpr")])
    dm = xgb.DMatrix(xv, label=yv, weight=wv)
    out = []
    for k in range(1, rounds + 1):
        line = bst[:k].eval(dm, "v")
        out.append(float(re.search(r"v-aucpr:([0-9.eE+-]+)", line).group(1)))
    return np.array(out)


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("card", [3, 8, 40])
def test_aucpr_matches_xgboost_per_round(tmp_path, weighted, card):
    x, y, w = toy(card=card)
    xv, yv, wv = toy(300, seed=4, card=card)
    rounds = 8
    res = {}
    if weighted:
        ev, wref = (xv, yv, wv, "v"), wv
    else:
        ev, wref = (xv, yv, "v"), None
    model = fb.train({**BASE, "eval_metric": "aucpr"}, x, label=y,
                     weight=w if weighted else None, num_boost_round=rounds,
                     evals=[ev], evals_result=res, quiet=True)
    ours = np.array(res["v"]["aucpr"])
    theirs = xgb_aucpr_per_round(model, tmp_path, xv, yv, wref, rounds)
    np.testing.assert_allclose(ours, theirs, rtol=2e-6)


def test_aucpr_all_tied_predictions_match_xgboost(tmp_path):
    # Constant feature -> every prediction identical (a single tie group).
    rng = np.random.default_rng(7)
    x = np.zeros((200, 2), dtype=np.float32)
    y = (rng.random(200) < 0.3).astype(np.float32)
    w = rng.uniform(0.5, 2.0, 200).astype(np.float32)
    res = {}
    model = fb.train({**BASE, "eval_metric": "aucpr"}, x, label=y, num_boost_round=2,
                     evals=[(x, y, w, "v")], evals_result=res, quiet=True)
    theirs = xgb_aucpr_per_round(model, tmp_path, x, y, w, 2)
    np.testing.assert_allclose(res["v"]["aucpr"], theirs, rtol=2e-6)


def test_aucpr_single_class_is_nan():
    x, y, _ = toy()
    res = {}
    fb.train({**BASE, "eval_metric": "aucpr"}, x, label=y, num_boost_round=2,
             evals=[(x[:50], np.ones(50, dtype=np.float32), "v")], evals_result=res, quiet=True)
    assert np.isnan(res["v"]["aucpr"]).all()


def test_aucpr_early_stopping_matches_xgboost_best_iteration():
    x, y, w = toy(600, card=40)
    xv, yv, wv = toy(300, seed=2, card=40)
    res = {}
    model = fb.train({**BASE, "eval_metric": "aucpr", "eta": 0.5}, x, label=y, weight=w,
                     num_boost_round=60, evals=[(xv, yv, wv, "v")],
                     early_stopping_rounds=3, evals_result=res, quiet=True)
    curve = res["v"]["aucpr"]
    assert model.best_iteration == int(np.argmax(curve))
    assert model.best_score == pytest.approx(max(curve))

    ref_res = {}
    ref = xgb.train({**BASE, "tree_method": "hist", "eval_metric": "aucpr", "eta": 0.5,
                     "max_bin": 256},
                    xgb.DMatrix(x, label=y, weight=w), num_boost_round=60,
                    evals=[(xgb.DMatrix(xv, label=yv, weight=wv), "v")],
                    early_stopping_rounds=3, evals_result=ref_res, verbose_eval=False)
    assert model.best_iteration == ref.best_iteration
    np.testing.assert_allclose(curve, ref_res["v"]["aucpr"], rtol=2e-6)


def test_aucpr_rejected_for_regression():
    x, y, _ = toy()
    with pytest.raises(Exception, match="binary:logistic"):
        fb.train({"objective": "reg:squarederror", "eval_metric": "aucpr"}, x, label=y,
                 num_boost_round=1, evals=[(x, y, "v")], quiet=True)
