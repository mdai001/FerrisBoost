"""Opt-in CUDA regression tests for the parallel fast gradient sampler."""

import json
import os
import re

import numpy as np
import pytest

import ferrisboost as fb


cuda = pytest.mark.skipif(
    os.environ.get("FB_TEST_CUDA") != "1", reason="requires explicit CUDA test opt-in"
)


def _data(n_rows, objective, seed=17):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_rows, 4)).astype(np.float32)
    margin = 0.9 * x[:, 0] - 0.4 * x[:, 1] * x[:, 2] + 0.2 * x[:, 3]
    if objective == "binary:logistic":
        y = (margin > np.median(margin)).astype(np.float32)
    else:
        y = margin.astype(np.float32)
    return x, y


def _params(objective="reg:squarederror", **overrides):
    params = {
        "objective": objective,
        "max_depth": 2,
        "eta": 0.2,
        "max_bin": 32,
        "nthread": 1,
        "seed": 17,
        "device": "cuda",
        "gpu_math": "fast",
        "sampling_method": "gradient_based",
        "subsample": 0.6,
    }
    params.update(overrides)
    return params


def _fit_with_path(capfd, params, x, y, rounds=1, **kwargs):
    capfd.readouterr()
    model = fb.train(params, x, label=y, num_boost_round=rounds, quiet=False, **kwargs)
    log = capfd.readouterr().err
    line = next((row for row in log.splitlines() if row.startswith("GPU_PATH")), None)
    assert line is not None, log[-2000:]
    sampler = re.search(r"\bgradient_sampler=(\S+)", line)
    scratch = re.search(r"\bgradient_sampling_bytes=(\d+)", line)
    assert sampler and scratch, line
    return model, sampler.group(1), int(scratch.group(1)), line


def _model_bytes(model, tmp_path, name):
    path = tmp_path / f"{name}.json"
    model.save_model(str(path), quiet=True)
    return path.read_bytes()


@cuda
@pytest.mark.parametrize("n_rows", [65535, 65536, 65537])
@pytest.mark.parametrize("objective", ["reg:squarederror", "binary:logistic"])
def test_parallel_sampler_dispatches_at_fixed_row_cutoff(capfd, n_rows, objective):
    x, y = _data(n_rows, objective)
    _, sampler, scratch, line = _fit_with_path(capfd, _params(objective), x, y, rounds=1)
    if n_rows < 65536:
        assert sampler == "single_cta", line
        assert scratch == 0, line
    else:
        assert sampler == "parallel_tile", line
        assert scratch == 8256, line


@cuda
def test_parallel_sampler_is_repeatable_and_continuation_matches(tmp_path):
    x, y = _data(65537, "reg:squarederror", seed=29)
    params = _params()
    whole = fb.train(params, x, label=y, num_boost_round=3, quiet=True)
    repeated = fb.train(params, x, label=y, num_boost_round=3, quiet=True)
    first = fb.train(params, x, label=y, num_boost_round=1, quiet=True)
    continued = fb.train(params, x, label=y, num_boost_round=2,
                          init_model=first, quiet=True)

    expected = _model_bytes(whole, tmp_path, "whole")
    assert _model_bytes(repeated, tmp_path, "repeated") == expected
    assert _model_bytes(continued, tmp_path, "continued") == expected


@cuda
def test_parallel_sampler_allocation_is_absent_for_other_paths(capfd):
    x, y = _data(65537, "reg:squarederror", seed=31)
    common = {
        "objective": "reg:squarederror",
        "max_depth": 2,
        "eta": 0.2,
        "max_bin": 32,
        "nthread": 1,
        "seed": 17,
        "device": "cuda",
        "gpu_math": "fast",
        "subsample": 0.6,
    }
    cases = [
        ({**common, "sampling_method": "uniform"}, "none"),
        ({**common, "sampling_method": "gradient_based", "subsample": 1.0}, "none"),
        ({**common, "sampling_method": "gradient_based", "gpu_math": "exact"}, "none"),
    ]
    for params, expected_sampler in cases:
        _, sampler, scratch, line = _fit_with_path(capfd, params, x, y, rounds=1)
        assert sampler == expected_sampler, line
        assert scratch == 0, line


def _root_leaf_value(model, tmp_path):
    path = tmp_path / "one_leaf.json"
    model.save_model(str(path), quiet=True)
    tree = json.loads(path.read_text(encoding="utf-8"))["learner"]["gradient_booster"][
        "model"]["trees"][0]
    leaf = next(index for index, child in enumerate(tree["left_children"]) if child == -1)
    return float(tree["split_conditions"][leaf])


@cuda
def test_parallel_sampler_matches_independent_weighted_one_leaf_oracle(tmp_path):
    # Every positive-weight row has the same squared-error gradient/Hessian
    # ratio. Importance inclusion and inverse-probability reweighting change
    # total cover but cannot change the exact root-leaf Newton update: 4.
    n_rows = 65537
    x = np.zeros((n_rows, 4), dtype=np.float32)
    y = np.full(n_rows, 4.0, dtype=np.float32)
    weight = np.linspace(0.2, 2.0, n_rows, dtype=np.float32)
    weight[::17] = 0.0
    params = _params(
        max_depth=0,
        eta=1.0,
        max_bin=32,
        min_child_weight=0.0,
        base_score=0.0,
        **{"lambda": 0.0},
    )
    model = fb.train(params, x, label=y, weight=weight, num_boost_round=1, quiet=True)

    expected_leaf = 4.0  # -sum(g) / sum(h), with g=-4w and h=w.
    assert _root_leaf_value(model, tmp_path) == pytest.approx(expected_leaf, rel=1e-6, abs=1e-6)
    margin = model.predict(x, output_margin=True)
    assert margin.shape == (n_rows,)
    np.testing.assert_allclose(margin, expected_leaf, rtol=1e-6, atol=1e-6)

@cuda
def test_parallel_sampler_reweighting_matches_uniform_mask_cover(tmp_path):
    # Equal scores give the same inclusion mask as uniform sampling at 1/4,
    # with fourfold Hessian reweighting. Cover detects a sampler no-op even
    # though the constant-label leaf prediction itself would be unchanged.
    n_rows = 65537
    x = np.zeros((n_rows, 1), dtype=np.float32)
    y = np.full(n_rows, 4.0, dtype=np.float32)
    params = _params(base_score=0.0, subsample=0.25, seed=71, max_depth=1,
                     eta=1.0, min_child_weight=0.0, **{"lambda": 0.0})
    covers = {}
    for sampling in ("uniform", "gradient_based"):
        model = fb.train({**params, "sampling_method": sampling}, x, label=y,
                         num_boost_round=1, quiet=True)
        path = tmp_path / f"cover_{sampling}.json"
        model.save_model(str(path), quiet=True)
        tree = json.loads(path.read_text(encoding="utf-8"))["learner"][
            "gradient_booster"]["model"]["trees"][0]
        covers[sampling] = float(tree["sum_hessian"][0])
    assert 0 < covers["uniform"] < n_rows
    assert covers["gradient_based"] == pytest.approx(4 * covers["uniform"], rel=1e-6)
    assert covers["gradient_based"] != n_rows