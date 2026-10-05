"""Explicit CUDA training must fail cleanly when no platform driver is loadable."""
import ctypes
import csv
import re

import numpy as np
import pytest

import ferrisboost as fb


def _platform_driver_loadable():
    if sys_platform() == "win32":
        soname = "nvcuda.dll"
    elif sys_platform() == "linux":
        soname = "libcuda.so.1"
    else:
        pytest.skip(f"driver-absent contract is not defined for {sys_platform()}")
    try:
        ctypes.CDLL(soname)
    except OSError:
        return False
    return True


def sys_platform():
    # Keep platform-specific loader checks local to this narrowly scoped test.
    import sys

    return sys.platform


def _fixtures(tmp_path):
    rng = np.random.default_rng(9031)
    features = rng.integers(0, 32, size=(192, 8)).astype(np.float32)
    labels = ((features[:, 0] + features[:, 3] + features[:, 6]) > 45).astype(np.float32)
    names = [f"f{i}" for i in range(features.shape[1])]
    csv_path = tmp_path / "no-driver.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow([*names, "target"])
        for row in range(len(features)):
            writer.writerow([*(format(float(value), ".9g") for value in features[row]),
                             format(float(labels[row]), ".9g")])

    parquet_path = tmp_path / "no-driver.parquet"
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError:
        pa = None
        pq = None
    if pa is not None:
        pq.write_table(
            pa.table({**{name: features[:, i] for i, name in enumerate(names)},
                      "target": labels}),
            parquet_path,
        )
    return features, labels, csv_path, parquet_path


def _train(kind, params, features, labels, csv_path, parquet_path):
    if kind == "ndarray":
        return fb.train(params, features, label=labels, num_boost_round=3, quiet=True)
    path = csv_path if kind == "csv" else parquet_path
    return fb.train(params, path, label="target", num_boost_round=3, quiet=True)


@pytest.mark.parametrize("kind", ["ndarray", "csv", "parquet"])
@pytest.mark.parametrize("device", ["cuda", "cuda:0"])
def test_explicit_cuda_training_without_driver_is_catchable_and_cpu_recovers(
    tmp_path, capfd, kind, device
):
    if _platform_driver_loadable():
        pytest.skip("platform CUDA driver is loadable; this test covers missing-driver behavior only")

    features, labels, csv_path, parquet_path = _fixtures(tmp_path)
    if kind == "parquet" and not parquet_path.exists():
        pytest.importorskip("pyarrow")
        pytest.importorskip("pyarrow.parquet")

    base_params = {
        "objective": "binary:logistic",
        "max_depth": 3,
        "max_bin": 32,
        "eta": 0.2,
        "lambda": 1.0,
        "seed": 17,
        "nthread": 1,
        "ingest_threads": 1,
        # CSV cannot report its row count before open; supply the required
        # geometry so the test reaches optional-driver initialization.
        "cols_per_block": 4,
    }
    baseline = _train(kind, base_params, features, labels, csv_path, parquet_path)
    baseline_path = tmp_path / "baseline.json"
    baseline.save_model(str(baseline_path), quiet=True)
    baseline_model_bytes = baseline_path.read_bytes()
    prediction_data = features if kind == "ndarray" else (
        csv_path if kind == "csv" else parquet_path
    )
    baseline_margin = baseline.predict(
        prediction_data, output_margin=True, predict_threads=1
    )
    assert baseline_margin.dtype == np.float32
    assert baseline_margin.shape == (len(features),)

    with pytest.raises((RuntimeError, ValueError), match=re.compile(r"CUDA.*driver", re.IGNORECASE)):
        _train(kind, {**base_params, "device": device},
               features, labels, csv_path, parquet_path)
    captured = capfd.readouterr()
    combined_output = captured.out + "\n" + captured.err
    lowered_output = combined_output.casefold()
    assert "panicked" not in lowered_output
    assert "panicexception" not in lowered_output
    assert "ferrisboost 内部错误" not in lowered_output

    recovered = _train(kind, base_params, features, labels, csv_path, parquet_path)
    recovered_path = tmp_path / "recovered.json"
    recovered.save_model(str(recovered_path), quiet=True)
    recovered_margin = recovered.predict(
        prediction_data, output_margin=True, predict_threads=1
    )
    assert recovered.num_trees == baseline.num_trees == 3
    assert recovered_path.read_bytes() == baseline_model_bytes
    assert recovered_margin.dtype == np.float32
    assert recovered_margin.shape == baseline_margin.shape
    assert recovered_margin.tobytes() == baseline_margin.tobytes()
