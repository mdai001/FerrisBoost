"""Probability-valued regression with the `reg:logistic` objective."""
import json
import os

import numpy as np
import pytest

import ferrisboost as fb


def data(n=180, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 8, size=(n, 4)).astype(np.float32)
    # Keep labels fractional while tying them to the features.
    y = np.clip(0.12 + 0.09 * x[:, 0] + 0.025 * x[:, 1], 0.05, 0.95)
    if n >= 4:
        y[:4] = [0.0, 1.0, 0.25, 0.75]
    return x, y.astype(np.float32)


PARAMS = {
    "max_depth": 2,
    "eta": 0.25,
    "nthread": 1,
    "max_bin": 16,
    "base_score": 0.4,
    "scale_pos_weight": 2.5,
    "seed": 7,
}


def test_fractional_labels_predict_in_probability_range_and_default_metric_is_rmse():
    x, y = data()
    result = {}
    model = fb.train({**PARAMS, "objective": "reg:logistic"}, x, label=y,
                     num_boost_round=3, evals=[(x, y, "train")], evals_result=result,
                     quiet=True)

    pred = model.predict(x)
    assert np.all((pred >= 0.0) & (pred <= 1.0))
    assert list(result["train"]) == ["rmse"]
    expected = np.sqrt(np.mean((y.astype(np.float64) - pred.astype(np.float64)) ** 2))
    assert result["train"]["rmse"][-1] == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize("sampling_method", ["uniform", "gradient_based"])
def test_reg_logistic_matches_binary_logistic_training_and_prediction(sampling_method):
    x, y = data()
    weight = np.linspace(0.4, 2.0, len(y), dtype=np.float32)
    common = {**PARAMS, "objective": "reg:logistic", "sampling_method": sampling_method, "subsample": 0.65}
    reg = fb.train(common, x, label=y, weight=weight, num_boost_round=4, quiet=True)
    binary = fb.train({**common, "objective": "binary:logistic"}, x, label=y,
                      weight=weight, num_boost_round=4, quiet=True)

    np.testing.assert_array_equal(reg.predict(x, output_margin=True),
                                  binary.predict(x, output_margin=True))
    np.testing.assert_array_equal(reg.predict(x), binary.predict(x))


@pytest.mark.parametrize("labels", [
    lambda n: np.full(n, -0.01, dtype=np.float32),
    lambda n: np.full(n, 1.01, dtype=np.float32),
])
def test_labels_outside_unit_interval_are_rejected(labels):
    x, _ = data()
    with pytest.raises(ValueError, match=r"\[0, ?1\]|between 0 and 1|label"):
        fb.train({**PARAMS, "objective": "reg:logistic"}, x, label=labels(len(x)),
                 num_boost_round=1, quiet=True)


@pytest.mark.parametrize("base_score", [-0.1, 1.1, np.nan, np.inf])
def test_base_score_outside_closed_unit_interval_is_rejected(base_score):
    x, y = data()
    with pytest.raises(ValueError, match="base_score"):
        fb.train({**PARAMS, "objective": "reg:logistic", "base_score": base_score},
                 x, label=y, num_boost_round=1, quiet=True)


@pytest.mark.parametrize("base_score", [0.0, 1.0])
def test_closed_interval_base_scores_produce_finite_probability_predictions(base_score):
    x, y = data()
    model = fb.train({**PARAMS, "objective": "reg:logistic", "base_score": base_score},
                     x, label=y, num_boost_round=1, quiet=True)
    pred = model.predict(x)
    assert np.all(np.isfinite(pred))
    assert np.all((pred >= 0.0) & (pred <= 1.0))


def _saved_objective(path):
    return json.loads(path.read_text(encoding="utf-8"))["learner"]["objective"]["name"]


def test_save_load_preserves_reg_logistic_name_and_continuation(tmp_path):
    x, y = data()
    params = {**PARAMS, "objective": "reg:logistic"}
    uninterrupted = fb.train(params, x, label=y, num_boost_round=4, quiet=True)
    first = fb.train(params, x, label=y, num_boost_round=2, quiet=True)
    path = tmp_path / "reg-logistic.json"
    first.save_model(str(path), quiet=True)

    assert _saved_objective(path) == "reg:logistic"
    loaded = fb.Model.load_model(str(path), quiet=True)
    continued = fb.train(params, x, label=y, num_boost_round=2, init_model=loaded,
                          quiet=True)
    assert continued.num_trees == 4
    np.testing.assert_array_equal(continued.predict(x, output_margin=True),
                                  uninterrupted.predict(x, output_margin=True))
    continued_path = tmp_path / "continued.json"
    continued.save_model(str(continued_path), quiet=True)
    assert _saved_objective(continued_path) == "reg:logistic"


def test_sklearn_regressor_accepts_reg_logistic_fractional_targets():
    pytest.importorskip("sklearn")
    from ferrisboost.sklearn import FerrisBoostRegressor

    x, y = data()
    reg = FerrisBoostRegressor(objective="reg:logistic", n_estimators=3, max_depth=2,
                               n_jobs=1, base_score=0.4, scale_pos_weight=2.5,
                               max_bin=16, random_state=7).fit(x, y)
    pred = reg.predict(x)
    assert np.all((pred >= 0.0) & (pred <= 1.0))


def _cuda_available():
    try:
        x, y = data(n=32)
        fb.train({"objective": "reg:logistic", "device": "cuda", "max_depth": 1,
                  "nthread": 1}, x, label=y, num_boost_round=1, quiet=True)
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
def test_cuda_exact_matches_cpu_and_fast_is_repeatable(tmp_path):
    x, y = data()
    params = {**PARAMS, "objective": "reg:logistic", "max_depth": 2}
    cpu = fb.train(params, x, label=y, num_boost_round=3, quiet=True)
    exact = fb.train({**params, "device": "cuda", "gpu_math": "exact"}, x, label=y,
                     num_boost_round=3, quiet=True)
    np.testing.assert_array_equal(cpu.predict(x), exact.predict(x))
    cpu_path = tmp_path / "cpu.json"
    exact_path = tmp_path / "exact.json"
    cpu.save_model(str(cpu_path), quiet=True)
    exact.save_model(str(exact_path), quiet=True)
    assert cpu_path.read_bytes() == exact_path.read_bytes()

    fast_params = {**params, "device": "cuda", "gpu_math": "fast"}
    fast_a = fb.train(fast_params, x, label=y, num_boost_round=3, quiet=True)
    fast_b = fb.train(fast_params, x, label=y, num_boost_round=3, quiet=True)
    np.testing.assert_array_equal(fast_a.predict(x), fast_b.predict(x))
    fast_a_path = tmp_path / "fast-a.json"
    fast_b_path = tmp_path / "fast-b.json"
    fast_a.save_model(str(fast_a_path), quiet=True)
    fast_b.save_model(str(fast_b_path), quiet=True)
    assert fast_a_path.read_bytes() == fast_b_path.read_bytes()
