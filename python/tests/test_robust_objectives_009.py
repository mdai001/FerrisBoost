"""Expectile and robust absolute-error objective behavior."""
import json
import os

import numpy as np
import pytest

import ferrisboost as fb


OBJECTIVES = ("reg:expectileerror", "reg:absoluteerror")
ALPHA = 0.7
BASE = {"max_depth": 2, "eta": 0.25, "nthread": 1, "max_bin": 16,
        "base_score": 0.5, "seed": 13}


def data(objective, n=150, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 8, size=(n, 4)).astype(np.float32)
    y = (0.3 + 0.4 * x[:, 0] - 0.2 * x[:, 1] + 0.1 * x[:, 2]).astype(np.float32)
    if objective == "reg:absoluteerror" and n >= 2:
        y[:2] = [2.0, 2.0]
    weight = np.linspace(0.25, 2.0, n, dtype=np.float32)
    return x, y, weight


def params(objective, **extra):
    p = {**BASE, "objective": objective, **extra}
    if objective == "reg:expectileerror":
        p["expectile_alpha"] = ALPHA
    return p


def _model_base_score(path):
    value = json.loads(path.read_text(encoding="utf-8"))["learner"][
        "learner_model_param"]["base_score"]
    if isinstance(value, list):
        return float(value[0])
    text = str(value).strip().strip("[]")
    return float(text.split(",")[0])


def _initial_score(objective, y, weight):
    mean = np.average(y.astype(np.float64), weights=weight.astype(np.float64))
    residual = np.float32(mean) - y
    if objective == "reg:expectileerror":
        asymmetry = np.where(residual >= 0.0, 1.0 - ALPHA, ALPHA).astype(np.float32)
        grad = asymmetry * residual
        hess = asymmetry
    else:
        delta = np.float32(np.average(np.sqrt(np.abs(residual).astype(np.float64)),
                                      weights=weight.astype(np.float64)) ** 2)
        curvature = delta / np.hypot(delta, residual)
        grad = residual * curvature
        hess = curvature
    step = -np.sum(weight * grad, dtype=np.float64) / np.sum(weight * hess, dtype=np.float64)
    return float(np.float32(mean) + np.float32(step))


@pytest.mark.parametrize("objective,metric", [
    ("reg:expectileerror", "expectile"),
    ("reg:absoluteerror", "mae"),
])
def test_weighted_default_metric_matches_independent_numpy(objective, metric):
    x, y, weight = data(objective)
    result = {}
    model = fb.train(params(objective), x, label=y, weight=weight, num_boost_round=3,
                     evals=[(x, y, weight, "train")], evals_result=result, quiet=True)
    pred = model.predict(x).astype(np.float64)
    label = y.astype(np.float64)
    w = weight.astype(np.float64)
    if objective == "reg:expectileerror":
        asymmetry = np.where(pred >= label, 1.0 - ALPHA, ALPHA)
        loss = asymmetry * (pred - label) ** 2
    else:
        loss = np.abs(pred - label)
    expected = np.sum(w * loss) / np.sum(w)
    assert list(result["train"]) == [metric]
    assert result["train"][metric][-1] == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize("objective", OBJECTIVES)
def test_weighted_mean_initialization_and_scale_pos_weight_is_ignored(tmp_path, objective):
    x, y, weight = data(objective)
    p = params(objective)
    p.pop("base_score")
    plain = fb.train(p, x, label=y, weight=weight, num_boost_round=2, quiet=True)
    ignored = fb.train({**p, "scale_pos_weight": 3.5}, x, label=y, weight=weight,
                       num_boost_round=2, quiet=True)
    np.testing.assert_array_equal(plain.predict(x, output_margin=True),
                                  ignored.predict(x, output_margin=True))

    path = tmp_path / f"{objective.replace(':', '-')}.json"
    plain.save_model(str(path), quiet=True)
    assert _model_base_score(path) == pytest.approx(_initial_score(objective, y, weight),
                                                    rel=1e-6)


@pytest.mark.parametrize("objective", OBJECTIVES)
def test_weighted_mean_initialization_and_single_leaf_newton_step(objective):
    x = np.zeros((5, 1), dtype=np.float32)
    y = np.array([-1.0, 0.5, 1.0, 3.0, 5.0], dtype=np.float32)
    weight = np.array([0.5, 1.0, 2.0, 1.5, 0.75], dtype=np.float32)
    p = {"objective": objective, "max_depth": 1, "max_bin": 4, "eta": 1.0,
         "reg_lambda": 0.0, "min_child_weight": 0.0, "nthread": 1}
    if objective == "reg:expectileerror":
        p["expectile_alpha"] = ALPHA
    model = fb.train(p, x, label=y, weight=weight, num_boost_round=1, quiet=True)

    initial = np.float32(_initial_score(objective, y, weight))
    residual = initial - y
    if objective == "reg:expectileerror":
        asymmetry = np.where(residual >= 0.0, 1.0 - ALPHA, ALPHA).astype(np.float32)
        grad = weight * asymmetry * residual
        hess = weight * asymmetry
    else:
        delta = np.float32(np.average(np.sqrt(np.abs(residual).astype(np.float64)),
                                      weights=weight.astype(np.float64)) ** 2)
        curvature = delta / np.hypot(delta, residual)
        grad = weight * residual * curvature
        hess = weight * curvature
    expected = initial - np.sum(grad, dtype=np.float64) / np.sum(hess, dtype=np.float64)
    margin = model.predict(x, output_margin=True)
    np.testing.assert_allclose(margin, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("alpha", [0.0, 0.2, 0.5, 1.0])
def test_expectile_accepts_scalar_alpha_on_closed_unit_interval(alpha):
    x, y, _ = data("reg:expectileerror", n=40)
    model = fb.train({**BASE, "objective": "reg:expectileerror", "expectile_alpha": alpha},
                     x, label=y, num_boost_round=1, quiet=True)
    assert np.all(np.isfinite(model.predict(x)))


@pytest.mark.parametrize("bad_alpha", [-0.01, 1.01, np.nan, np.inf, [0.2, 0.8]])
def test_expectile_rejects_out_of_range_or_nonscalar_alpha(bad_alpha):
    x, y, _ = data("reg:expectileerror", n=40)
    with pytest.raises((TypeError, ValueError),
                       match="expectile_alpha|scalar|alpha|real number|sequence|list"):
        fb.train({**BASE, "objective": "reg:expectileerror", "expectile_alpha": bad_alpha},
                 x, label=y, num_boost_round=1, quiet=True)


def test_expectile_alpha_is_required():
    x, y, _ = data("reg:expectileerror", n=40)
    p = {**BASE, "objective": "reg:expectileerror"}
    p.pop("expectile_alpha", None)
    with pytest.raises((TypeError, ValueError), match="expectile_alpha|alpha"):
        fb.train(p, x, label=y, num_boost_round=1, quiet=True)


def test_expectile_rejects_multi_alpha_explicitly():
    x, y, _ = data("reg:expectileerror", n=40)
    with pytest.raises((TypeError, ValueError), match="scalar|one expectile|expectile_alpha"):
        fb.train({**BASE, "objective": "reg:expectileerror", "expectile_alpha": [0.2, 0.8]},
                 x, label=y, num_boost_round=1, quiet=True)


def test_absolute_error_zero_residual_has_finite_unit_prediction(tmp_path):
    x, _y, weight = data("reg:absoluteerror", n=60)
    y = np.full(len(x), 2.0, dtype=np.float32)
    model = fb.train({**BASE, "objective": "reg:absoluteerror", "base_score": 2.0},
                     x, label=y, weight=weight, num_boost_round=3, quiet=True)
    pred = model.predict(x)
    assert np.all(np.isfinite(pred))
    np.testing.assert_array_equal(pred, np.full(len(x), 2.0, dtype=np.float32))
    path = tmp_path / "absolute-zero.json"
    model.save_model(str(path), quiet=True)
    assert json.loads(path.read_text(encoding="utf-8"))["learner"]["objective"]["name"] \
        == "reg:absoluteerror"


@pytest.mark.parametrize("objective", OBJECTIVES)
def test_save_load_name_and_continuation(tmp_path, objective):
    x, y, weight = data(objective)
    p = params(objective)
    uninterrupted = fb.train(p, x, label=y, weight=weight, num_boost_round=4, quiet=True)
    first = fb.train(p, x, label=y, weight=weight, num_boost_round=2, quiet=True)
    path = tmp_path / "first.json"
    first.save_model(str(path), quiet=True)
    assert json.loads(path.read_text(encoding="utf-8"))["learner"]["objective"]["name"] \
        == objective

    loaded = fb.Model.load_model(str(path), quiet=True)
    continued = fb.train(p, x, label=y, weight=weight, num_boost_round=2,
                          init_model=loaded, quiet=True)
    np.testing.assert_array_equal(continued.predict(x, output_margin=True),
                                  uninterrupted.predict(x, output_margin=True))
    final_path = tmp_path / "continued.json"
    continued.save_model(str(final_path), quiet=True)
    assert json.loads(final_path.read_text(encoding="utf-8"))["learner"]["objective"]["name"] \
        == objective


def _cuda_available():
    try:
        x, y, weight = data("reg:expectileerror", n=32)
        fb.train(params("reg:expectileerror", device="cuda"), x, label=y, weight=weight,
                 num_boost_round=1, quiet=True)
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
def test_cuda_exact_cpu_parity_and_fast_repeatability(tmp_path, objective):
    x, y, weight = data(objective, n=100, seed=17)
    p = params(objective)
    cpu = fb.train(p, x, label=y, weight=weight, num_boost_round=3, quiet=True)
    exact = fb.train({**p, "device": "cuda", "gpu_math": "exact"}, x, label=y,
                     weight=weight, num_boost_round=3, quiet=True)
    cpu_path, exact_path = tmp_path / "cpu.json", tmp_path / "exact.json"
    cpu.save_model(str(cpu_path), quiet=True)
    exact.save_model(str(exact_path), quiet=True)
    assert cpu_path.read_bytes() == exact_path.read_bytes()
    np.testing.assert_array_equal(cpu.predict(x), exact.predict(x))

    fast_params = {**p, "device": "cuda", "gpu_math": "fast"}
    fast_a = fb.train(fast_params, x, label=y, weight=weight, num_boost_round=3, quiet=True)
    fast_b = fb.train(fast_params, x, label=y, weight=weight, num_boost_round=3, quiet=True)
    fast_a_path, fast_b_path = tmp_path / "fast-a.json", tmp_path / "fast-b.json"
    fast_a.save_model(str(fast_a_path), quiet=True)
    fast_b.save_model(str(fast_b_path), quiet=True)
    assert fast_a_path.read_bytes() == fast_b_path.read_bytes()
    np.testing.assert_array_equal(fast_a.predict(x), fast_b.predict(x))



def test_expectile_continuation_rejects_changed_alpha():
    x = np.arange(24, dtype=np.float32).reshape(12, 2)
    y = np.arange(12, dtype=np.float32)
    p = {"objective": "reg:expectileerror", "expectile_alpha": 0.3, "nthread": 1}
    model = fb.train(p, x, label=y, num_boost_round=1, quiet=True)
    with pytest.raises(ValueError, match="objective parameters"):
        fb.train({**p, "expectile_alpha": 0.7}, x, label=y,
                 init_model=model, num_boost_round=1, quiet=True)
