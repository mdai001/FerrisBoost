"""Independent one-leaf checks for new GPU fast-math loss kernels."""
import json
import os

import numpy as np
import pytest

import ferrisboost as fb


cuda = pytest.mark.skipif(
    os.environ.get("FB_TEST_CUDA") != "1", reason="requires explicit CUDA test opt-in"
)


def _loss_data(objective):
    # A constant feature makes each boosting round a single leaf. Repeated,
    # varied labels let the second round independently check updated margins.
    x = np.zeros((80, 2), dtype=np.float32)
    if objective == "binary:hinge":
        y = np.array(([1.0] * 64) + ([0.0] * 16), dtype=np.float32)
        base_score = 0.6  # First update crosses the positive hinge kink at +1.
    elif objective == "reg:squaredlogerror":
        y = np.resize(np.array([-0.5, 0.0, 0.4, 1.5, 3.0], dtype=np.float32), len(x))
        base_score = 0.3
    elif objective == "reg:absoluteerror":
        y = np.resize(np.array([-0.75, 0.1, 0.8, 1.7, 3.2], dtype=np.float32), len(x))
        base_score = 0.4
    else:  # reg:expectileerror
        y = np.resize(np.array([-1.0, -0.1, 0.5, 1.2, 2.5], dtype=np.float32), len(x))
        base_score = 0.2
    weight = np.resize(np.array([0.25, 0.5, 1.0, 1.5, 2.0], dtype=np.float32), len(x))
    return x, y, weight, base_score


def _gradient_pair(objective, margin, labels, weight):
    residual = margin - labels.astype(np.float64)
    if objective == "reg:squaredlogerror":
        pred = np.maximum(margin, -1.0 + 1e-6)
        denom = pred + 1.0
        delta = np.log1p(pred) - np.log1p(labels.astype(np.float64))
        grad = delta / denom
        hess = np.maximum((1.0 - delta) / (denom * denom), 1e-6)
    elif objective == "binary:hinge":
        sign = 2.0 * labels.astype(np.float64) - 1.0
        active = margin * sign < 1.0
        grad = np.where(active, -sign, 0.0)
        hess = np.where(active, 1.0, np.finfo(np.float32).tiny)
    elif objective == "reg:absoluteerror":
        delta = (np.sum(weight.astype(np.float64) * np.sqrt(np.abs(residual)))
                 / np.sum(weight.astype(np.float64))) ** 2
        norm = np.hypot(delta, residual)
        curvature = np.where(norm > 0.0, delta / np.maximum(norm, 1e-300), 1.0)
        grad = residual * curvature
        hess = curvature
    else:
        alpha = 0.3
        asymmetry = np.where(residual >= 0.0, 1.0 - alpha, alpha)
        grad = asymmetry * residual
        hess = asymmetry
    return grad, hess


def _one_leaf_oracle(objective, labels, weight, base_score):
    margin = float(np.float32(base_score))
    rounds = []
    for _ in range(2):
        grad, hess = _gradient_pair(objective, margin, labels, weight)
        update = -np.sum(weight.astype(np.float64) * grad) / np.sum(
            weight.astype(np.float64) * hess
        )
        margin += update
        rounds.append(margin)
    return rounds


def _train_one_leaf(objective, x, y, weight, base_score):
    params = {
        "objective": objective,
        "max_depth": 1,
        "max_bin": 4,
        "eta": 1.0,
        "lambda": 0.0,
        "min_child_weight": 0.0,
        "base_score": base_score,
        "nthread": 1,
        "device": "cuda",
        "gpu_math": "fast",
        "subsample": 1.0,
    }
    if objective == "reg:expectileerror":
        params["expectile_alpha"] = 0.3
    return fb.train(params, x, label=y, weight=weight, num_boost_round=2, quiet=True)


@cuda
@pytest.mark.parametrize("objective", [
    "reg:squaredlogerror", "binary:hinge", "reg:absoluteerror", "reg:expectileerror",
])
def test_fast_loss_kernels_match_independent_two_round_leaf_oracle(objective):
    x, y, weight, base_score = _loss_data(objective)
    model = _train_one_leaf(objective, x, y, weight, base_score)
    expected = _one_leaf_oracle(objective, y, weight, base_score)
    first_margin = model.predict(x, output_margin=True, iteration_range=(0, 1))
    final_margin = model.predict(x, output_margin=True)
    np.testing.assert_allclose(first_margin, expected[0], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(final_margin, expected[1], rtol=1e-5, atol=1e-5)
    assert first_margin.shape == final_margin.shape == (len(y),)
    assert np.all(np.isfinite(first_margin))
    assert np.all(np.isfinite(final_margin))


def _root_cover(model, path):
    model.save_model(str(path), quiet=True)
    tree = json.loads(path.read_text(encoding="utf-8"))["learner"]["gradient_booster"][
        "model"]["trees"][0]
    return float(tree["sum_hessian"][0])


@cuda
def test_fast_gradient_sampler_reweights_importance_samples_not_just_dispatch(tmp_path):
    n = 1024
    x = np.zeros((n, 1), dtype=np.float32)
    y = np.full(n, 4.0, dtype=np.float32)
    shared = {
        "objective": "reg:squarederror",
        "max_depth": 1,
        "max_bin": 4,
        "eta": 1.0,
        "lambda": 0.0,
        "min_child_weight": 0.0,
        "base_score": 0.0,
        "subsample": 0.25,
        "seed": 71,
        "nthread": 1,
        "device": "cuda",
        "gpu_math": "fast",
    }
    gradient = fb.train({**shared, "sampling_method": "gradient_based"}, x, label=y,
                        num_boost_round=1, quiet=True)
    uniform = fb.train({**shared, "sampling_method": "uniform"}, x, label=y,
                       num_boost_round=1, quiet=True)

    gradient_cover = _root_cover(gradient, tmp_path / "gradient.json")
    uniform_cover = _root_cover(uniform, tmp_path / "uniform.json")
    assert uniform_cover > 0.0
    assert gradient_cover / uniform_cover == pytest.approx(4.0, rel=0.05)
    np.testing.assert_array_equal(gradient.predict(x), uniform.predict(x))
    np.testing.assert_allclose(gradient.predict(x), 4.0, rtol=0.0, atol=1e-6)
