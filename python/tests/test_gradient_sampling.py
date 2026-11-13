"""Checks for deterministic gradient-based row sampling."""

import csv
import os

import numpy as np
import pytest

import ferrisboost as fb


BASE = {
    "objective": "reg:squarederror",
    "max_depth": 3,
    "eta": 0.2,
    "max_bin": 24,
    "nthread": 1,
    "seed": 17,
}


def _data(n_rows=384, seed=29):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n_rows, 8)).astype(np.float32)
    y = (1.7 * x[:, 0] - 0.8 * x[:, 1] * x[:, 2] + 0.4 * x[:, 5]).astype(np.float32)
    return x, y


def _model_bytes(model, tmp_path, name="model"):
    path = tmp_path / f"{name}.json"
    model.save_model(str(path), quiet=True)
    return path.read_bytes()


cuda = pytest.mark.skipif(
    os.environ.get("FB_TEST_CUDA") != "1", reason="requires explicit CUDA test opt-in"
)


def test_uniform_default_and_gradient_rate_one_preserve_model_json(tmp_path):
    x, y = _data()
    default = fb.train(BASE, x, label=y, num_boost_round=5, quiet=True)
    uniform = fb.train({**BASE, "sampling_method": "uniform"}, x, label=y,
                       num_boost_round=5, quiet=True)
    gradient = fb.train({**BASE, "sampling_method": "gradient_based", "subsample": 1.0},
                        x, label=y, num_boost_round=5, quiet=True)
    expected = _model_bytes(default, tmp_path, "default")
    assert _model_bytes(uniform, tmp_path, "uniform") == expected
    assert _model_bytes(gradient, tmp_path, "gradient_one") == expected


def test_sampling_method_rejects_unknown_value():
    x, y = _data(32)
    with pytest.raises(ValueError, match="sampling_method"):
        fb.train({**BASE, "sampling_method": "weighted"}, x, label=y,
                 num_boost_round=1, quiet=True)


def test_gradient_sampling_is_repeatable_across_threads_and_blocks(tmp_path):
    x, y = _data()
    params = {**BASE, "sampling_method": "gradient_based", "subsample": 0.57}
    one = fb.train({**params, "nthread": 1, "cols_per_block": 2}, x, label=y,
                   num_boost_round=6, quiet=True)
    repeat = fb.train({**params, "nthread": 1, "cols_per_block": 2}, x, label=y,
                      num_boost_round=6, quiet=True)
    parallel = fb.train({**params, "nthread": 4, "cols_per_block": 4}, x, label=y,
                        num_boost_round=6, quiet=True)
    expected = _model_bytes(one, tmp_path, "one")
    assert _model_bytes(repeat, tmp_path, "repeat") == expected
    assert _model_bytes(parallel, tmp_path, "parallel") == expected

    # Fast math on CPU still uses host importance sampling; the flag affects
    # CUDA arithmetic only.
    cpu_fast_flag = fb.train({**params, "gpu_math": "fast"}, x, label=y,
                             num_boost_round=6, quiet=True)
    assert _model_bytes(cpu_fast_flag, tmp_path, "cpu_fast_flag") == expected


def test_gradient_sampling_changes_subsampled_model_without_accuracy_gate():
    x, y = _data()
    common = {**BASE, "subsample": 0.3}
    uniform = fb.train(common, x, label=y, num_boost_round=8, quiet=True)
    gradient = fb.train({**common, "sampling_method": "gradient_based"}, x, label=y,
                        num_boost_round=8, quiet=True)
    assert not np.array_equal(uniform.predict(x), gradient.predict(x))


def test_custom_objective_receives_all_rows_under_gradient_sampling():
    x, y = _data(96)
    seen = []

    def objective(raw_margin, labels):
        seen.append((raw_margin.copy(), labels.copy()))
        return np.ascontiguousarray(raw_margin - labels), np.ones(len(labels), dtype=np.float32)

    params = {
        **BASE,
        "sampling_method": "gradient_based",
        "subsample": 0.45,
        "custom_objective": fb.CustomObjective("all_rows_sampling", objective),
    }
    model = fb.train(params, x, label=y, num_boost_round=3, quiet=True)
    assert model.num_trees == 3
    assert len(seen) == 3
    for margin, labels in seen:
        assert margin.shape == labels.shape == (len(y),)
        assert margin.dtype == labels.dtype == np.float32
        np.testing.assert_array_equal(labels, y)
    assert any(not np.array_equal(seen[0][0], margin) for margin, _ in seen[1:])


def test_gradient_sampling_accepts_named_csv_and_parquet_inputs(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    x, y = _data(128)
    table = pa.table({**{f"f{i}": x[:, i] for i in range(x.shape[1])}, "target": y})
    parquet = tmp_path / "train.parquet"
    pq.write_table(table, parquet)
    csv_path = tmp_path / "train.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([*(f"f{i}" for i in range(x.shape[1])), "target"])
        writer.writerows(np.column_stack((x, y)))
    params = {**BASE, "sampling_method": "gradient_based", "subsample": 0.6}
    for path in (parquet, csv_path):
        model = fb.train(params, path, label="target", num_boost_round=3, quiet=True)
        assert len(model.predict(x)) == len(y)


def test_sklearn_clone_and_fit_forward_sampling_method():
    pytest.importorskip("sklearn")
    from sklearn.base import clone
    from ferrisboost.sklearn import FerrisBoostRegressor

    x, y = _data(160)
    estimator = FerrisBoostRegressor(
        n_estimators=4, max_depth=2, n_jobs=1, sampling_method="gradient_based",
        subsample=0.55, random_state=3,
    )
    cloned = clone(estimator)
    assert cloned.sampling_method == "gradient_based"
    cloned.fit(x, y)
    assert len(cloned.predict(x)) == len(y)


def test_multiclass_gradient_sampling_is_repeatable_and_outputs_all_rows(tmp_path):
    pytest.importorskip("sklearn")
    from sklearn.datasets import make_classification

    x, y = make_classification(n_samples=300, n_features=8, n_informative=6,
                               n_redundant=0, n_classes=3, random_state=12)
    x = x.astype(np.float32)
    params = {
        "objective": "multi:softprob", "num_class": 3, "max_depth": 3,
        "eta": 0.2, "max_bin": 20, "nthread": 1, "seed": 8,
        "sampling_method": "gradient_based", "subsample": 0.6,
    }
    first = fb.train(params, x, label=y, num_boost_round=4, quiet=True)
    second = fb.train(params, x, label=y, num_boost_round=4, quiet=True)
    assert _model_bytes(first, tmp_path, "first") == _model_bytes(second, tmp_path, "second")
    prediction = first.predict(x)
    assert prediction.shape == (len(y), 3)
    assert np.all(np.isfinite(prediction))


def test_gradient_sampling_continuation_matches_uninterrupted_training(tmp_path):
    x, y = _data()
    params = {**BASE, "sampling_method": "gradient_based", "subsample": 0.62}
    first = fb.train(params, x, label=y, num_boost_round=2, quiet=True)
    continued = fb.train(params, x, label=y, num_boost_round=2, init_model=first, quiet=True)
    whole = fb.train(params, x, label=y, num_boost_round=4, quiet=True)
    assert _model_bytes(continued, tmp_path, "continued") == _model_bytes(whole, tmp_path, "whole")


@cuda
@pytest.mark.parametrize("device_quantize", [True, False])
def test_gradient_sampling_cpu_and_exact_gpu_models_match(tmp_path, device_quantize):
    x, y = _data(256)
    params = {**BASE, "sampling_method": "gradient_based", "subsample": 0.58}
    cpu = fb.train(params, x, label=y, num_boost_round=4, quiet=True)
    gpu = fb.train({**params, "device": "cuda", "gpu_math": "exact",
                    "device_quantize": device_quantize}, x, label=y,
                   num_boost_round=4, quiet=True)
    assert _model_bytes(cpu, tmp_path, f"cpu_{device_quantize}") == _model_bytes(
        gpu, tmp_path, f"gpu_{device_quantize}"
    )


@cuda
def test_fast_gradient_sampling_at_rate_one_is_uniform_and_repeatable(tmp_path):
    x, y = _data(64)
    params = {**BASE, "subsample": 1.0, "device": "cuda", "gpu_math": "fast"}
    gradient_params = {**params, "sampling_method": "gradient_based"}
    uniform_params = {**params, "sampling_method": "uniform"}
    gradient = fb.train(gradient_params, x, label=y, num_boost_round=4, quiet=True)
    repeat = fb.train(gradient_params, x, label=y, num_boost_round=4, quiet=True)
    uniform = fb.train(uniform_params, x, label=y, num_boost_round=4, quiet=True)
    expected = _model_bytes(gradient, tmp_path, "gradient_rate_one")
    assert _model_bytes(repeat, tmp_path, "gradient_rate_one_repeat") == expected
    assert _model_bytes(uniform, tmp_path, "uniform_rate_one") == expected


@cuda
def test_gradient_sampling_custom_objective_cpu_gpu_exact_weights_and_margins_match(tmp_path):
    x, y = _data(128)
    weight = np.linspace(0.35, 1.8, len(y), dtype=np.float32)
    cpu_margins = []
    gpu_margins = []

    def callback(capture):
        def objective(raw_margin, labels):
            capture.append((raw_margin.copy(), labels.copy()))
            return (np.ascontiguousarray(raw_margin - labels),
                    np.ones(len(labels), dtype=np.float32))
        return objective

    params = {
        **BASE,
        "base_score": 0.0,
        "sampling_method": "gradient_based",
        "subsample": 0.45,
        "custom_objective": fb.CustomObjective(
            "gradient_sampling_exact_gpu", callback(cpu_margins)
        ),
    }
    cpu = fb.train(params, x, label=y, weight=weight, num_boost_round=4, quiet=True)
    gpu_params = {
        **params,
        "device": "cuda",
        "gpu_math": "exact",
        "custom_objective": fb.CustomObjective(
            "gradient_sampling_exact_gpu", callback(gpu_margins)
        ),
    }
    gpu = fb.train(gpu_params, x, label=y, weight=weight, num_boost_round=4, quiet=True)

    assert len(cpu_margins) == len(gpu_margins) == 4
    for (cpu_margin, cpu_labels), (gpu_margin, gpu_labels) in zip(cpu_margins, gpu_margins):
        assert cpu_margin.shape == gpu_margin.shape == cpu_labels.shape == gpu_labels.shape == (len(y),)
        np.testing.assert_array_equal(cpu_labels, y)
        np.testing.assert_array_equal(gpu_labels, y)
        np.testing.assert_array_equal(gpu_margin, cpu_margin)
    assert _model_bytes(cpu, tmp_path, "custom_cpu") == _model_bytes(gpu, tmp_path, "custom_gpu")


@cuda
def test_multiclass_gradient_sampling_exact_gpu_and_continuation_match(tmp_path):
    rng = np.random.default_rng(43)
    x = rng.normal(size=(240, 7)).astype(np.float32)
    logits = np.column_stack((x[:, 0] + x[:, 1], -x[:, 0] + x[:, 2], -x[:, 1] - x[:, 2]))
    y = np.argmax(logits, axis=1).astype(np.float32)
    params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "base_score": 0.5,
        "max_depth": 3,
        "eta": 0.2,
        "max_bin": 24,
        "nthread": 1,
        "seed": 43,
        "sampling_method": "gradient_based",
        "subsample": 0.57,
    }
    cpu = fb.train(params, x, label=y, num_boost_round=4, quiet=True)
    gpu_params = {**params, "device": "cuda", "gpu_math": "exact"}
    gpu = fb.train(gpu_params, x, label=y, num_boost_round=4, quiet=True)
    first = fb.train(gpu_params, x, label=y, num_boost_round=2, quiet=True)
    continued = fb.train(gpu_params, x, label=y, num_boost_round=2,
                         init_model=first, quiet=True)

    expected = _model_bytes(cpu, tmp_path, "multi_cpu")
    assert _model_bytes(gpu, tmp_path, "multi_gpu") == expected
    assert _model_bytes(continued, tmp_path, "multi_gpu_continued") == _model_bytes(
        gpu, tmp_path, "multi_gpu_whole"
    )
