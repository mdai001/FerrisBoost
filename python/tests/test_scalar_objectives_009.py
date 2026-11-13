"""Built-in scalar objectives added in the 0.0.9 development cycle."""
import json
import os

import numpy as np
import pytest

import ferrisboost as fb


OBJECTIVES = ("binary:logitraw", "reg:squaredlogerror", "binary:hinge")


def data(objective, n=120, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 8, size=(n, 4)).astype(np.float32)
    signal = 0.25 * (x[:, 0] - 3) + 0.1 * (x[:, 1] - 4)
    if objective in ("binary:logitraw", "binary:hinge"):
        y = (signal > 0).astype(np.float32)
        y[:4] = [0.0, 1.0, 0.0, 1.0]
    else:
        y = (1.0 + 0.2 * x[:, 0] + 0.1 * x[:, 1]).astype(np.float32)
    return x, y


BASE = {"max_depth": 2, "eta": 0.25, "nthread": 1, "max_bin": 16,
        "base_score": 0.2, "seed": 5}


def model_bytes(model, path):
    model.save_model(str(path), quiet=True)
    return path.read_bytes()


@pytest.mark.parametrize("objective,metric", [
    ("binary:logitraw", "logloss"),
    ("reg:squaredlogerror", "rmsle"),
    ("binary:hinge", "error"),
])
def test_identity_margin_predictions_and_default_metric(objective, metric):
    x, y = data(objective)
    result = {}
    model = fb.train({**BASE, "objective": objective}, x, label=y, num_boost_round=3,
                     evals=[(x, y, "train")], evals_result=result, quiet=True)
    margin = model.predict(x, output_margin=True).astype(np.float64)
    pred = model.predict(x).astype(np.float64)
    assert list(result["train"]) == [metric]

    if objective == "binary:logitraw":
        np.testing.assert_array_equal(pred, margin)
        prob = 1.0 / (1.0 + np.exp(-margin))
        yy = y.astype(np.float64)
        expected = -np.mean(yy * np.log(np.clip(prob, 1e-15, 1.0))
                            + (1.0 - yy) * np.log(np.clip(1.0 - prob, 1e-15, 1.0)))
        assert result["train"][metric][-1] == pytest.approx(expected, rel=1e-6)
        raw_result = {}
        fb.train({**BASE, "objective": objective, "eval_metric": "rmse"}, x, label=y,
                 num_boost_round=3, evals=[(x, y, "train")], evals_result=raw_result,
                 quiet=True)
        expected_raw_rmse = np.sqrt(np.mean((yy - margin) ** 2))
        assert raw_result["train"]["rmse"][-1] == pytest.approx(expected_raw_rmse, rel=1e-6)
    elif objective == "reg:squaredlogerror":
        np.testing.assert_array_equal(pred, margin)
        expected = np.sqrt(np.mean((np.log1p(pred) - np.log1p(y.astype(np.float64))) ** 2))
        assert result["train"][metric][-1] == pytest.approx(expected, rel=1e-6)
    else:
        expected = (margin > 0.0).astype(np.float64)
        np.testing.assert_array_equal(pred, expected)
        assert set(np.unique(pred)) <= {0.0, 1.0}
        assert result["train"][metric][-1] == pytest.approx(np.mean(pred != y), abs=1e-7)


@pytest.mark.parametrize("objective", OBJECTIVES)
def test_objective_accepts_finite_raw_base_score_and_rejects_nonfinite(objective):
    x, y = data(objective)
    model = fb.train({**BASE, "objective": objective, "base_score": -0.35}, x, label=y,
                     num_boost_round=1, quiet=True)
    margin = model.predict(x, output_margin=True)
    prediction = model.predict(x)
    if objective != "binary:hinge":
        np.testing.assert_array_equal(prediction, margin)
    else:
        np.testing.assert_array_equal(prediction, (margin > 0.0).astype(np.float32))

    for invalid in (np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError, match="base_score"):
            fb.train({**BASE, "objective": objective, "base_score": invalid}, x, label=y,
                     num_boost_round=1, quiet=True)


@pytest.mark.parametrize("objective,bad", [
    ("binary:logitraw", -0.01),
    ("binary:logitraw", 1.01),
    ("binary:hinge", -0.01),
    ("binary:hinge", 1.01),
    ("reg:squaredlogerror", -1.0),
    ("reg:squaredlogerror", -1.01),
])
def test_objective_rejects_labels_outside_its_domain(objective, bad):
    x, y = data(objective)
    y[0] = bad
    with pytest.raises(ValueError, match="label|between|\[0, ?1\]|greater"):
        fb.train({**BASE, "objective": objective}, x, label=y, num_boost_round=1,
                 quiet=True)


@pytest.mark.parametrize("objective", OBJECTIVES)
def test_save_load_preserves_name_and_continuation(tmp_path, objective):
    x, y = data(objective)
    params = {**BASE, "objective": objective}
    uninterrupted = fb.train(params, x, label=y, num_boost_round=4, quiet=True)
    first = fb.train(params, x, label=y, num_boost_round=2, quiet=True)
    path = tmp_path / "first.json"
    first.save_model(str(path), quiet=True)
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["learner"]["objective"]["name"] == objective

    loaded = fb.Model.load_model(str(path), quiet=True)
    continued = fb.train(params, x, label=y, num_boost_round=2, init_model=loaded,
                          quiet=True)
    np.testing.assert_array_equal(continued.predict(x, output_margin=True),
                                  uninterrupted.predict(x, output_margin=True))
    final_path = tmp_path / "continued.json"
    continued.save_model(str(final_path), quiet=True)
    final = json.loads(final_path.read_text(encoding="utf-8"))
    assert final["learner"]["objective"]["name"] == objective


@pytest.mark.parametrize("objective", ["binary:logitraw", "binary:hinge"])
def test_sklearn_classifier_prediction_probability_and_model_load(tmp_path, objective):
    pytest.importorskip("sklearn")
    from ferrisboost.sklearn import FerrisBoostClassifier

    x, y = data(objective, n=100)
    clf = FerrisBoostClassifier(objective=objective, n_estimators=3, max_depth=2,
                                n_jobs=1, random_state=5).fit(x, y)
    pred = clf.predict(x)
    assert pred.shape == y.shape
    assert np.all(np.isin(pred, clf.classes_))

    if objective == "binary:logitraw":
        margin = clf.predict(x, output_margin=True)
        np.testing.assert_array_equal(pred, clf.classes_[(margin > 0.0).astype(np.intp)])
        extreme_margin = np.where(np.arange(len(x)) % 2 == 0, 1000.0, -1000.0).astype(np.float32)
        proba = clf.predict_proba(x, base_margin=extreme_margin)
        assert proba.shape == (len(x), 2)
        assert np.all(np.isfinite(proba))
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-7)
    else:
        assert set(np.unique(pred)) <= {0.0, 1.0}
        with pytest.raises(ValueError, match="class decisions, not probabilities"):
            clf.predict_proba(x)

    path = tmp_path / f"{objective.replace(':', '-')}.json"
    clf.get_booster().save_model(str(path), quiet=True)
    loaded = FerrisBoostClassifier().load_model(path)
    loaded_pred = loaded.predict(x)
    assert loaded_pred.shape == y.shape
    assert np.all(np.isin(loaded_pred, loaded.classes_))
    if objective == "binary:logitraw":
        loaded_proba = loaded.predict_proba(x)
        assert loaded_proba.shape == (len(x), 2)
        assert np.all(np.isfinite(loaded_proba))
    else:
        with pytest.raises(ValueError, match="class decisions, not probabilities"):
            loaded.predict_proba(x)


def _cuda_available():
    try:
        x, y = data("binary:logitraw", n=32)
        fb.train({"objective": "binary:logitraw", "device": "cuda", "max_depth": 1},
                 x, label=y, num_boost_round=1, quiet=True)
        return True
    except Exception as exc:  # noqa: BLE001
        if "cuda" in str(exc).lower():
            return False
        raise


cuda_opt_in = pytest.mark.skipif(
    os.environ.get("FB_TEST_CUDA") != "1" or not _cuda_available(),
    reason="requires FB_TEST_CUDA=1 and an available CUDA build/device",
)


@cuda_opt_in
@pytest.mark.parametrize("objective", OBJECTIVES)
def test_cuda_exact_matches_cpu_and_fast_is_repeatable(tmp_path, objective):
    x, y = data(objective, n=160, seed=9)
    params = {**BASE, "objective": objective}
    cpu = fb.train(params, x, label=y, num_boost_round=3, quiet=True)
    exact = fb.train({**params, "device": "cuda", "gpu_math": "exact"}, x, label=y,
                     num_boost_round=3, quiet=True)
    assert model_bytes(cpu, tmp_path / "cpu.json") == model_bytes(exact, tmp_path / "exact.json")
    np.testing.assert_array_equal(cpu.predict(x), exact.predict(x))

    fast_params = {**params, "device": "cuda", "gpu_math": "fast"}
    fast_a = fb.train(fast_params, x, label=y, num_boost_round=3, quiet=True)
    fast_b = fb.train(fast_params, x, label=y, num_boost_round=3, quiet=True)
    assert model_bytes(fast_a, tmp_path / "fast-a.json") == model_bytes(
        fast_b, tmp_path / "fast-b.json")
    np.testing.assert_array_equal(fast_a.predict(x), fast_b.predict(x))
