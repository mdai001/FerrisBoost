"""Opt-in CUDA tests for fast-math gradient-based row sampling."""
import os

import numpy as np
import pytest

import ferrisboost as fb


OBJECTIVES = (
    "reg:squarederror",
    "binary:logistic",
    "reg:quantileerror",
    "reg:absoluteerror",
    "reg:expectileerror",
    "multi:softprob",
)


def _data(objective, n_rows=512, seed=29):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_rows, 8)).astype(np.float32)
    score = 1.7 * x[:, 0] - 0.8 * x[:, 1] * x[:, 2] + 0.4 * x[:, 5]
    if objective == "binary:logistic":
        y = (score > np.median(score)).astype(np.float32)
    elif objective == "multi:softprob":
        labels = np.column_stack((score, -score + x[:, 3], x[:, 1] - x[:, 4]))
        y = np.argmax(labels, axis=1).astype(np.float32)
    else:
        y = score.astype(np.float32)
    return x, y


def _params(objective, seed=17, subsample=0.4, **extra):
    params = {
        "objective": objective,
        "max_depth": 3,
        "eta": 0.2,
        "max_bin": 24,
        "nthread": 1,
        "seed": seed,
        "device": "cuda",
        "gpu_math": "fast",
        "sampling_method": "gradient_based",
        "subsample": subsample,
    }
    if objective == "reg:quantileerror":
        params["quantile_alpha"] = 0.5
    if objective == "reg:expectileerror":
        params["expectile_alpha"] = 0.3
    if objective == "multi:softprob":
        params["num_class"] = 3
    params.update(extra)
    return params


def _model_bytes(model, tmp_path, name):
    path = tmp_path / f"{name}.json"
    model.save_model(str(path), quiet=True)
    return path.read_bytes()


def _assert_fast_combo(tmp_path, objective, x, y, fit_kwargs=None, **extra):
    fit_kwargs = dict(fit_kwargs or {})
    params = _params(objective, seed=67, subsample=0.4, **extra)

    def fit(training_params, rounds, init_model=None):
        return fb.train(training_params, x, label=y, num_boost_round=rounds,
                        init_model=init_model, quiet=True, **fit_kwargs)

    subsampled = fit(params, 3)
    repeated = fit(params, 3)
    assert _model_bytes(repeated, tmp_path, "repeat") == _model_bytes(
        subsampled, tmp_path, "subsampled")

    full_params = {**params, "subsample": 1.0}
    full_gradient = fit(full_params, 3)
    full_uniform = fit({**full_params, "sampling_method": "uniform"}, 3)
    assert _model_bytes(full_gradient, tmp_path, "full_gradient") == _model_bytes(
        full_uniform, tmp_path, "full_uniform")

    first = fit(params, 1)
    continued = fit(params, 2, init_model=first)
    assert _model_bytes(continued, tmp_path, "continued") == _model_bytes(
        subsampled, tmp_path, "uninterrupted")

    prediction = subsampled.predict(x)
    margin = subsampled.predict(x, output_margin=True)
    assert prediction.shape[0] == margin.shape[0] == len(y)
    assert np.all(np.isfinite(prediction))
    assert np.all(np.isfinite(margin))


cuda = pytest.mark.skipif(
    os.environ.get("FB_TEST_CUDA") != "1", reason="requires explicit CUDA test opt-in"
)


@cuda
@pytest.mark.parametrize("objective", OBJECTIVES)
def test_fast_sampling_is_repeatable_seeded_and_updates_every_row(tmp_path, objective):
    x, y = _data(objective)
    params = _params(objective)
    first = fb.train(params, x, label=y, num_boost_round=5, quiet=True)
    repeated = fb.train(params, x, label=y, num_boost_round=5, quiet=True)
    changed_seed = fb.train(_params(objective, seed=18), x, label=y,
                            num_boost_round=5, quiet=True)
    expected = _model_bytes(first, tmp_path, "first")
    assert _model_bytes(repeated, tmp_path, "repeated") == expected
    assert _model_bytes(changed_seed, tmp_path, "changed_seed") != expected

    margin = first.predict(x, output_margin=True)
    prediction = first.predict(x)
    assert margin.shape[0] == len(y)
    assert prediction.shape[0] == len(y)
    assert np.all(np.isfinite(margin))
    assert np.all(np.isfinite(prediction))


@cuda
@pytest.mark.parametrize("objective", OBJECTIVES)
def test_fast_sampling_rate_one_matches_uniform_and_subsampled_differs(tmp_path, objective):
    x, y = _data(objective, seed=31)
    full_gradient = fb.train(_params(objective, seed=22, subsample=1.0), x, label=y,
                             num_boost_round=4, quiet=True)
    full_uniform = fb.train(_params(objective, seed=22, subsample=1.0,
                                    sampling_method="uniform"),
                            x, label=y, num_boost_round=4, quiet=True)
    assert _model_bytes(full_gradient, tmp_path, "full_gradient") == _model_bytes(
        full_uniform, tmp_path, "full_uniform")

    subsampled = fb.train(_params(objective, seed=22, subsample=0.4), x, label=y,
                          num_boost_round=4, quiet=True)
    uniform_subsampled = fb.train(_params(objective, seed=22, subsample=0.4,
                                          sampling_method="uniform"),
                                  x, label=y, num_boost_round=4, quiet=True)
    assert _model_bytes(subsampled, tmp_path, "subsampled_gradient") != _model_bytes(
        uniform_subsampled, tmp_path, "subsampled_uniform")


@cuda
@pytest.mark.parametrize("objective", OBJECTIVES)
def test_fast_sampling_continuation_matches_uninterrupted_training(tmp_path, objective):
    x, y = _data(objective, seed=41)
    params = _params(objective, seed=23, subsample=0.4)
    first = fb.train(params, x, label=y, num_boost_round=2, quiet=True)
    continued = fb.train(params, x, label=y, num_boost_round=2,
                          init_model=first, quiet=True)
    uninterrupted = fb.train(params, x, label=y, num_boost_round=4, quiet=True)
    assert _model_bytes(continued, tmp_path, "continued") == _model_bytes(
        uninterrupted, tmp_path, "uninterrupted")


@cuda
@pytest.mark.parametrize("resident_blocks", [0, 1])
@pytest.mark.parametrize("gpu_split_cache", [False, True])
def test_fast_sampling_residency_and_split_cache_switches_are_repeatable(
    tmp_path, monkeypatch, resident_blocks, gpu_split_cache
):
    x, y = _data("reg:squarederror", n_rows=256, seed=51)
    monkeypatch.setenv("FB_GPU_SPLIT_CACHE", "1" if gpu_split_cache else "0")
    params = _params("reg:squarederror", seed=37, subsample=0.45,
                     resident_blocks=resident_blocks)
    first = fb.train(params, x, label=y, num_boost_round=3, quiet=True)
    second = fb.train(params, x, label=y, num_boost_round=3, quiet=True)
    assert _model_bytes(first, tmp_path, "first") == _model_bytes(second, tmp_path, "second")
    assert np.all(np.isfinite(first.predict(x, output_margin=True)))


@cuda
def test_fast_sampling_ranking_with_query_groups_and_weights(tmp_path):
    rng = np.random.default_rng(73)
    sizes = np.array([20, 24, 28, 32])
    qid = np.repeat(np.arange(len(sizes), dtype=np.uint32), sizes)
    x = rng.normal(size=(int(sizes.sum()), 6)).astype(np.float32)
    relevance = 0.9 * x[:, 0] - 0.5 * x[:, 1] + 0.2 * x[:, 2]
    y = np.concatenate([
        np.clip(np.rint(relevance[qid == group] - relevance[qid == group].min()), 0, 4)
        for group in range(len(sizes))
    ]).astype(np.float32)
    group_weight = np.array([0.5, 1.0, 1.75, 1.25], dtype=np.float32)
    _assert_fast_combo(
        tmp_path, "rank:ndcg", x, y,
        fit_kwargs={"qid": qid, "weight": group_weight},
        max_depth=3,
    )


@cuda
def test_fast_sampling_multi_alpha_quantile(tmp_path):
    x, y = _data("reg:quantileerror", n_rows=192, seed=79)
    _assert_fast_combo(
        tmp_path, "reg:quantileerror", x, y,
        quantile_alpha=[0.2, 0.8],
        max_depth=2,
    )


@cuda
def test_fast_sampling_sample_weights_with_zeros_and_varying_values(tmp_path):
    x, y = _data("reg:squarederror", n_rows=192, seed=83)
    weight = np.linspace(0.2, 2.0, len(y), dtype=np.float32)
    weight[::7] = 0.0
    _assert_fast_combo(
        tmp_path, "reg:squarederror", x, y,
        fit_kwargs={"weight": weight},
        max_depth=2,
    )


@cuda
def test_fast_sampling_wide_max_bin_300(tmp_path):
    x, y = _data("reg:squarederror", n_rows=384, seed=89)
    _assert_fast_combo(tmp_path, "reg:squarederror", x, y, max_bin=300, max_depth=2)


@cuda
def test_fast_sampling_interaction_constraints_with_max_leaves(tmp_path):
    x, y = _data("reg:squarederror", n_rows=192, seed=97)
    _assert_fast_combo(
        tmp_path, "reg:squarederror", x, y,
        interaction_constraints=[[0, 1], [2, 3, 4, 5, 6, 7]],
        max_leaves=5,
        max_depth=4,
    )
