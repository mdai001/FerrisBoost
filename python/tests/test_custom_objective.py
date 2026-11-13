"""Python API checks for user supplied gradient and Hessian objectives."""

import copy
import csv
import json
import os

import numpy as np
import pytest

import ferrisboost as fb


BASE = {
    "objective": "reg:squarederror",
    "max_depth": 2,
    "eta": 0.25,
    "max_bin": 16,
    "nthread": 1,
    "base_score": 0.0,
}


def data(n=96):
    x = np.arange(n * 3, dtype=np.float32).reshape(n, 3) / np.float32(17)
    y = (0.35 * x[:, 0] - 0.2 * x[:, 1] + 0.1).astype(np.float32)
    return x, y


def squared_error(raw_margin, labels):
    return np.ascontiguousarray(raw_margin - labels), np.ones(len(labels), dtype=np.float32)


def custom_params(callback=squared_error, name="squared_error_callback", **params):
    return {
        **BASE,
        **params,
        "custom_objective": fb.CustomObjective(name, callback),
    }


def train_custom(x, y, callback=squared_error, **kwargs):
    rounds = kwargs.pop("num_boost_round", 4)
    name = kwargs.pop("objective_name", "squared_error_callback")
    return fb.train(
        custom_params(callback, name, **kwargs.pop("params", {})),
        x,
        label=y,
        num_boost_round=rounds,
        quiet=True,
        **kwargs,
    )


def read_json(model, path):
    model.save_model(str(path), quiet=True)
    return json.loads(path.read_text(encoding="utf-8"))


def without_objective_marker(model_json):
    model_json = copy.deepcopy(model_json)
    model_json["learner"]["attributes"].pop("ferrisboost_custom_objective", None)
    return model_json


def test_squarederror_callback_matches_builtin_with_weights_subsample_and_early_stopping(tmp_path):
    x, y = data()
    weight = np.linspace(0.5, 2.0, len(y), dtype=np.float32)
    valid_x, valid_y = x[::2].copy(), y[::2].copy()
    params = {**BASE, "subsample": 0.8, "seed": 19}
    calls = []

    def capture(raw_margin, labels):
        calls.append((raw_margin.copy(), labels.copy()))
        return squared_error(raw_margin, labels)

    custom = fb.train(
        custom_params(capture, "squarederror", subsample=0.8, seed=19),
        x,
        label=y,
        weight=weight,
        num_boost_round=10,
        evals=[(valid_x, valid_y, "valid")],
        early_stopping_rounds=2,
        quiet=True,
    )
    builtin = fb.train(
        params,
        x,
        label=y,
        weight=weight,
        num_boost_round=10,
        evals=[(valid_x, valid_y, "valid")],
        early_stopping_rounds=2,
        quiet=True,
    )

    assert custom.num_trees == builtin.num_trees == len(calls)
    for raw_margin, labels in calls:
        assert raw_margin.dtype == labels.dtype == np.float32
        assert raw_margin.shape == labels.shape == (len(y),)
        assert raw_margin.flags.c_contiguous and labels.flags.c_contiguous
        np.testing.assert_array_equal(labels, y)
    assert any(not np.array_equal(calls[0][0], margin) for margin, _ in calls[1:])
    np.testing.assert_array_equal(custom.predict(x, output_margin=True), builtin.predict(x, output_margin=True))

    custom_json = read_json(custom, tmp_path / "custom.json")
    builtin_json = read_json(builtin, tmp_path / "builtin.json")
    assert custom_json["learner"]["attributes"]["ferrisboost_custom_objective"] == "squarederror"
    assert without_objective_marker(custom_json) == without_objective_marker(builtin_json)


def test_custom_objective_defaults_base_score_to_zero(tmp_path):
    x, y = data()
    params = {key: value for key, value in BASE.items() if key != "base_score"}
    params["custom_objective"] = fb.CustomObjective("zero_start", squared_error)
    model = fb.train(params, x, label=y, num_boost_round=1, quiet=True)
    model_json = read_json(model, tmp_path / "zero_start.json")
    encoded = model_json["learner"]["learner_model_param"]["base_score"]
    assert float(str(encoded).strip("[]")) == 0.0


def test_root_only_callback_margins_follow_regularized_newton_recurrence():
    x, y = data(9)
    eta = 0.3
    reg_lambda = 1.0
    calls = []

    def record(raw_margin, labels):
        calls.append(raw_margin.copy())
        return squared_error(raw_margin, labels)

    params = {
        "objective": "reg:squarederror",
        "custom_objective": fb.CustomObjective("root_recurrence", record),
        "max_depth": 0,
        "base_score": 0.0,
        "eta": eta,
        "reg_lambda": reg_lambda,
        "nthread": 1,
    }
    fb.train(params, x, label=y, num_boost_round=4, quiet=True)

    expected_margin = 0.0
    expected_calls = []
    for _ in range(4):
        expected_calls.append(expected_margin)
        # One root leaf receives G = n * (margin - mean(y)) and H = n.
        # The learning-rate-scaled Newton update is -eta * G / (H + lambda).
        expected_margin += eta * (float(np.mean(y, dtype=np.float64)) - expected_margin) * len(y) / (
            len(y) + reg_lambda
        )
    assert len(calls) == len(expected_calls)
    for observed, expected in zip(calls, expected_calls):
        assert observed.dtype == np.float32
        assert observed.shape == (len(y),)
        np.testing.assert_allclose(observed, expected, rtol=0, atol=2e-6)


@pytest.mark.parametrize(
    "factory",
    [
        pytest.param(lambda n: (np.zeros(n, dtype=np.float64), np.ones(n, dtype=np.float32)), id="float64"),
        pytest.param(lambda n: (np.zeros((n, 1), dtype=np.float32), np.ones(n, dtype=np.float32)), id="2d"),
        pytest.param(lambda n: (np.zeros(2 * n, dtype=np.float32)[::2], np.ones(n, dtype=np.float32)), id="noncontiguous"),
        pytest.param(lambda n: (np.zeros(n - 1, dtype=np.float32), np.ones(n, dtype=np.float32)), id="wrong-length"),
    ],
)
@pytest.mark.parametrize("component", ["gradient", "hessian"])
def test_custom_objective_rejects_invalid_derivative_dtype_or_layout(factory, component):
    x, y = data(12)

    def invalid(_margin, labels):
        bad = factory(len(labels))[0]
        good = np.ones(len(labels), dtype=np.float32)
        return (bad, good) if component == "gradient" else (good, bad)

    with pytest.raises((TypeError, ValueError)):
        train_custom(x, y, invalid, num_boost_round=1)


@pytest.mark.parametrize(
    "bad_result",
    [
        pytest.param(None, id="none"),
        pytest.param(np.zeros(4, dtype=np.float32), id="single-array"),
        pytest.param((np.zeros(4, dtype=np.float32),), id="one-item-tuple"),
        pytest.param(
            (np.zeros(4, dtype=np.float32), np.ones(4, dtype=np.float32), np.ones(4, dtype=np.float32)),
            id="three-item-tuple",
        ),
    ],
)
def test_custom_objective_rejects_non_tuple_or_wrong_tuple_length(bad_result):
    x, y = data(4)
    with pytest.raises((TypeError, ValueError)):
        train_custom(x, y, lambda _margin, _labels: bad_result, num_boost_round=1)


def test_custom_objective_constructor_validates_name_and_callable():
    with pytest.raises(ValueError):
        fb.CustomObjective("  ", squared_error)
    with pytest.raises(ValueError, match="must be callable"):
        fb.CustomObjective("not_callable", 42)


@pytest.mark.parametrize(
    "which,bad_value",
    [
        ("gradient", np.nan),
        ("gradient", np.inf),
        ("hessian", np.nan),
        ("hessian", np.inf),
        ("hessian", -1.0),
    ],
)
def test_custom_objective_rejects_nonfinite_derivatives_and_negative_hessian(which, bad_value):
    x, y = data(12)

    def invalid(margin, labels):
        grad, hess = squared_error(margin, labels)
        (grad if which == "gradient" else hess)[3] = bad_value
        return grad, hess

    with pytest.raises(ValueError):
        train_custom(x, y, invalid, num_boost_round=1)


def test_custom_objective_allows_zero_hessian_with_regularization():
    x, y = data(12)

    def zero_curvature(margin, labels):
        return np.ascontiguousarray(margin - labels), np.zeros(len(labels), dtype=np.float32)

    model = train_custom(x, y, zero_curvature, num_boost_round=1, params={"reg_lambda": 1.0})
    assert model.num_trees == 1
    assert np.isfinite(model.predict(x)).all()


def test_custom_objective_rejects_overflow_after_sample_weighting():
    x, y = data(8)
    weight = np.full(len(y), 2.0, dtype=np.float32)

    def finite_but_overflowing(_margin, labels):
        return (
            np.full(len(labels), np.finfo(np.float32).max, dtype=np.float32),
            np.ones(len(labels), dtype=np.float32),
        )

    with pytest.raises(ValueError):
        fb.train(
            custom_params(finite_but_overflowing, "overflow"),
            x,
            label=y,
            weight=weight,
            num_boost_round=1,
            quiet=True,
        )


@pytest.mark.parametrize("objective", ["binary:logistic", "reg:logistic", "rank:pairwise"])
def test_custom_objective_rejects_other_named_objectives(objective):
    x, y = data(12)
    params = {**BASE, "objective": objective, "custom_objective": fb.CustomObjective("custom", squared_error)}
    with pytest.raises(ValueError):
        fb.train(params, x, label=y, num_boost_round=1, quiet=True)


def test_custom_objective_rejects_scale_pos_weight():
    x, y = data(12)
    params = {**BASE, "scale_pos_weight": 2.0, "custom_objective": fb.CustomObjective("custom", squared_error)}
    with pytest.raises(ValueError):
        fb.train(params, x, label=y, num_boost_round=1, quiet=True)


class ObjectiveMarkerError(RuntimeError):
    pass


def test_custom_objective_preserves_callback_exception_type():
    x, y = data(8)

    def fail(_margin, _labels):
        raise ObjectiveMarkerError("objective callback marker")

    with pytest.raises(ObjectiveMarkerError, match="objective callback marker"):
        train_custom(x, y, fail, num_boost_round=2)


def test_custom_objective_callback_arrays_are_owned_snapshots():
    x, y = data(16)
    retained = []

    def retain(margin, labels):
        retained.append((margin, margin.copy(), labels, labels.copy()))
        return squared_error(margin, labels)

    train_custom(x, y, retain, num_boost_round=3)
    assert len(retained) == 3
    for index, (margin, _, labels, _) in enumerate(retained):
        for other_margin, _, other_labels, _ in retained[index + 1 :]:
            assert not np.shares_memory(margin, other_margin)
            assert not np.shares_memory(labels, other_labels)
    for margin, margin_copy, labels, labels_copy in retained:
        assert margin.dtype == labels.dtype == np.float32
        assert margin.flags.c_contiguous and labels.flags.c_contiguous
        np.testing.assert_array_equal(margin, margin_copy)
        np.testing.assert_array_equal(labels, labels_copy)
    for margin, _, labels, _ in retained:
        margin[:] = 123.0
        labels[:] = -123.0
    for margin, _, labels, _ in retained:
        assert np.all(margin == 123.0)
        assert np.all(labels == -123.0)
    for _, margin_copy, _, labels_copy in retained:
        assert not np.all(margin_copy == 123.0)
        assert not np.all(labels_copy == -123.0)


def test_same_name_objective_descriptors_are_reusable_and_copyable():
    x, y = data(24)
    first = fb.CustomObjective("reusable", squared_error)
    second = fb.CustomObjective("reusable", lambda margin, labels: squared_error(margin, labels))

    for descriptor in (first, second, copy.copy(first), copy.deepcopy(first)):
        params = {**BASE, "custom_objective": descriptor}
        model = fb.train(params, x, label=y, num_boost_round=1, quiet=True)
        assert model.num_trees == 1


def test_sklearn_clone_runs_same_custom_objective_callback():
    pytest.importorskip("sklearn")
    from sklearn.base import clone

    from ferrisboost.sklearn import FerrisBoostRegressor

    calls = []

    def counted(raw_margin, labels):
        calls.append(len(labels))
        return squared_error(raw_margin, labels)

    objective = fb.CustomObjective("clone_objective", counted)
    estimator = FerrisBoostRegressor(
        n_estimators=2, max_depth=0, n_jobs=1, custom_objective=objective
    )
    cloned = clone(estimator)
    x, y = data(24)
    estimator.fit(x, y)
    cloned.fit(x, y)
    assert calls == [len(y)] * 4
    np.testing.assert_array_equal(estimator.predict(x), cloned.predict(x))


@pytest.mark.parametrize("file_format", ["csv", "parquet"])
def test_file_training_save_load_and_custom_objective_continuation(tmp_path, file_format):
    if file_format == "parquet":
        pa = pytest.importorskip("pyarrow")
        pq = pytest.importorskip("pyarrow.parquet")
    x, y = data(24)
    path = tmp_path / f"train.{file_format}"
    if file_format == "parquet":
        pq.write_table(pa.table({"f0": x[:, 0], "f1": x[:, 1], "f2": x[:, 2], "target": y}), path)
    else:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(["f0", "f1", "f2", "target"])
            writer.writerows(np.column_stack((x, y)).tolist())

    descriptor = fb.CustomObjective("file_objective", squared_error)
    params = {**BASE, "custom_objective": descriptor}
    first = fb.train(params, str(path), label="target", num_boost_round=2, quiet=True)
    model_path = tmp_path / f"{file_format}.json"
    first.save_model(str(model_path), quiet=True)
    loaded = fb.Model.load_model(str(model_path), quiet=True)
    np.testing.assert_array_equal(loaded.predict(x), first.predict(x))

    continued = fb.train(
        {**BASE, "custom_objective": fb.CustomObjective("file_objective", squared_error)},
        str(path),
        label="target",
        num_boost_round=1,
        init_model=loaded,
        quiet=True,
    )
    assert continued.num_trees == first.num_trees + 1
    np.testing.assert_array_equal(loaded.predict(x), first.predict(x))


def test_loaded_custom_objective_model_predicts_without_callable(tmp_path):
    x, y = data(16)
    model = train_custom(x, y, num_boost_round=2)
    path = tmp_path / "saved.json"
    model.save_model(str(path), quiet=True)
    loaded = fb.Model.load_model(str(path), quiet=True)
    np.testing.assert_array_equal(loaded.predict(x), model.predict(x))


def test_custom_objective_continuation_requires_same_name_provider():
    x, y = data(16)
    first = train_custom(x, y, num_boost_round=1)
    with pytest.raises(ValueError):
        fb.train(BASE, x, label=y, num_boost_round=1, init_model=first, quiet=True)
    with pytest.raises(ValueError):
        fb.train(
            {**BASE, "custom_objective": fb.CustomObjective("different", squared_error)},
            x,
            label=y,
            num_boost_round=1,
            init_model=first,
            quiet=True,
        )


def test_builtin_model_cannot_be_continued_with_custom_objective():
    x, y = data(16)
    builtin = fb.train(BASE, x, label=y, num_boost_round=1, quiet=True)
    params = {**BASE, "custom_objective": fb.CustomObjective("custom", squared_error)}
    with pytest.raises(ValueError):
        fb.train(params, x, label=y, num_boost_round=1, init_model=builtin, quiet=True)


@pytest.mark.skipif(os.environ.get("FB_TEST_CUDA") != "1", reason="requires explicit CUDA test opt-in")
def test_cuda_exact_custom_objective_matches_cpu_and_callback_margins():
    x, y = data(64)
    cpu_calls = []
    gpu_calls = []

    def cpu_callback(margin, labels):
        cpu_calls.append(margin.copy())
        return squared_error(margin, labels)

    def gpu_callback(margin, labels):
        gpu_calls.append(margin.copy())
        return squared_error(margin, labels)

    cpu = fb.train(
        {**BASE, "custom_objective": fb.CustomObjective("cuda_squarederror", cpu_callback)},
        x,
        label=y,
        num_boost_round=3,
        quiet=True,
    )
    gpu = fb.train(
        {**BASE, "device": "cuda:0", "gpu_math": "exact", "custom_objective": fb.CustomObjective("cuda_squarederror", gpu_callback)},
        x,
        label=y,
        num_boost_round=3,
        quiet=True,
    )
    assert len(cpu_calls) == len(gpu_calls) == 3
    for cpu_margin, gpu_margin in zip(cpu_calls, gpu_calls):
        np.testing.assert_array_equal(cpu_margin, gpu_margin)
    np.testing.assert_array_equal(cpu.predict(x, output_margin=True), gpu.predict(x, output_margin=True))


@pytest.mark.skipif(os.environ.get("FB_TEST_CUDA") != "1", reason="requires explicit CUDA test opt-in")
def test_cuda_fast_custom_objective_is_rejected():
    x, y = data(16)
    params = {
        **BASE,
        "device": "cuda:0",
        "gpu_math": "fast",
        "custom_objective": fb.CustomObjective("unsupported_gpu_fast", squared_error),
    }
    with pytest.raises(ValueError):
        fb.train(params, x, label=y, num_boost_round=1, quiet=True)


def test_callback_mutation_does_not_change_training_state():
    x, y = data(24)
    original_y = y.copy()

    def mutate_after_derivatives(margin, labels):
        grad, hess = squared_error(margin, labels)
        margin[:] = np.nan
        labels[:] = -999.0
        return grad, hess

    custom = train_custom(x, y, mutate_after_derivatives, num_boost_round=3)
    builtin = fb.train(BASE, x, label=y, num_boost_round=3, quiet=True)
    assert custom.predict(x, use_gpu=False).tobytes() == builtin.predict(x, use_gpu=False).tobytes()
    np.testing.assert_array_equal(y, original_y)
