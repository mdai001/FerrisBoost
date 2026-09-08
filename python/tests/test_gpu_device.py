"""Python 侧 GPU 入口的最小验收。

契约:Python 只说 `device="cuda"`;**GPU source 的构造和物理块宽都在 Rust
侧决定**(显存规划器),Python 不碰。所以这里测的是"接线通不通"和
"两条后端给的是不是同一个模型",不是性能。
"""
import re

import numpy as np
import pytest

import ferrisboost as fb


def _data(n_rows=800, n_feats=12, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 40, size=(n_rows, n_feats)).astype(np.float32)
    y = ((x[:, 0] + x[:, 3]) > 38).astype(np.float32)
    return x, y


def _has_cuda():
    try:
        fb.train({"device": "cuda", "max_depth": 2}, *_data(64, 4), num_boost_round=1)
        return True
    except Exception as e:  # noqa: BLE001
        # 没有 CUDA 构建 / 没有卡 —— 跳过而不是失败。
        if "CUDA" in str(e) or "cuda" in str(e):
            return False
        raise


cuda = pytest.mark.skipif(not _has_cuda(), reason="没有可用的 CUDA 构建或设备")


def test_device_parameter_is_validated():
    x, y = _data()
    with pytest.raises(ValueError, match="device"):
        fb.train({"device": "tpu"}, x, y, num_boost_round=1)


@cuda
def test_cpu_and_gpu_smoke():
    """两条后端都能跑通,并且都给出可用的模型。"""
    x, y = _data()
    p = {"max_depth": 3, "eta": 0.3, "max_bin": 32}
    cpu = fb.train(p, x, y, num_boost_round=3)
    gpu = fb.train({**p, "device": "cuda"}, x, y, num_boost_round=3)
    assert len(cpu.predict(x)) == len(x)
    assert len(gpu.predict(x)) == len(x)


@cuda
def test_gpu_prediction_is_bit_identical_and_cache_reusable(tmp_path):
    """Inference backend is independent of the backend that trained the model."""
    rng = np.random.default_rng(20260908)
    train_x = rng.normal(size=(4000, 16)).astype(np.float32)
    train_y = (
        train_x[:, 0]
        + train_x[:, 1] * train_x[:, 2]
        - 0.5 * train_x[:, 5]
        + 0.25 * train_x[:, 9]
        > 0
    ).astype(np.float32)
    model = fb.train(
        {"max_depth": 6, "eta": 0.2, "max_bin": 64, "nthread": 1},
        train_x,
        train_y,
        num_boost_round=30,
    )
    predict_x = np.tile(train_x, (25, 1))
    predict_x[::97, 0] = np.nan
    predict_x[::131, 7] = np.nan

    cpu_margin = model.predict(predict_x, output_margin=True, predict_threads=1)
    gpu_margin_first = model.predict(predict_x, output_margin=True, use_gpu=True)
    gpu_margin_reuse = model.predict(predict_x, output_margin=True, use_gpu=True)
    assert cpu_margin.tobytes() == gpu_margin_first.tobytes()
    assert cpu_margin.tobytes() == gpu_margin_reuse.tobytes()
    assert model.predict(predict_x, predict_threads=1).tobytes() == model.predict(
        predict_x, use_gpu=True
    ).tobytes()
    assert cpu_margin.tobytes() == model.predict(
        np.asfortranarray(predict_x), output_margin=True, use_gpu=True
    ).tobytes()
    strided_storage = np.empty((len(predict_x), predict_x.shape[1] * 2), dtype=np.float32)
    strided_storage[:, ::2] = predict_x
    strided_storage[:, 1::2] = -1.0
    assert cpu_margin.tobytes() == model.predict(
        strided_storage[:, ::2], output_margin=True, use_gpu=True
    ).tobytes()

    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    file_path = tmp_path / "predict.parquet"
    pq.write_table(
        pa.table({f"f{i}": predict_x[:, i] for i in range(predict_x.shape[1])}),
        file_path,
    )
    file_cpu = model.predict(file_path, output_margin=True, predict_threads=1)
    file_gpu = model.predict(file_path, output_margin=True, use_gpu=True)
    assert file_cpu.tobytes() == file_gpu.tobytes()


@cuda
def test_gpu_is_byte_identical_to_cpu_in_exact_mode():
    """exact 模式(默认)下,换后端**不允许**改变模型。

    这是 FerrisBoost 的核心承诺,也是为什么 colsample 用的是自己那套
    确定性采样、而不是 XGBoost 的(它的 CPU/GPU 本来就选不出同一批特征)。
    """
    import tempfile, pathlib as _p

    x, y = _data(1200, 16, seed=7)
    p = {"max_depth": 4, "eta": 0.2, "max_bin": 32, "seed": 5}
    with tempfile.TemporaryDirectory() as d:
        for rate in (1.0, 0.5):
            q = {**p, "colsample_bytree": rate}
            cpu = fb.train(q, x, y, num_boost_round=3)
            gpu = fb.train({**q, "device": "cuda"}, x, y, num_boost_round=3)
            a, b = _p.Path(d) / "cpu.json", _p.Path(d) / "gpu.json"
            cpu.save_model(str(a))
            gpu.save_model(str(b))
            assert a.read_bytes() == b.read_bytes(), (
                f"colsample={rate}:CPU 与 GPU 模型必须逐字节相同"
            )
            # 预测也必须一致(模型相同 => 预测相同,这条是冗余的保险)。
            assert np.array_equal(cpu.predict(x), gpu.predict(x))


@cuda
def test_csv_deferred_row_count_keeps_full_gpu_planning(tmp_path, capfd):
    """显式块宽只固定几何;CSV open 后仍须补完 stream/residency plan。"""
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")

    x, y = _data(1200, 16, seed=11)
    names = [f"f{i}" for i in range(x.shape[1])]
    columns = {name: x[:, i] for i, name in enumerate(names)}
    columns["target"] = y
    table = pa.table(columns)
    parquet = tmp_path / "epsilon-shape.parquet"
    csv = tmp_path / "epsilon-shape.csv.gz"
    pq.write_table(table, parquet)
    # NumPy transparently gzip-compresses paths ending in .gz.  Keep this test on
    # the exact compressed-input route whose deferred planner wiring regressed.
    np.savetxt(
        csv,
        np.column_stack((x, y)),
        delimiter=",",
        header=",".join([*names, "target"]),
        comments="",
    )

    params = {
        "device": "cuda",
        "gpu_math": "exact",
        "max_depth": 3,
        "max_bin": 32,
        "cols_per_block": 4,
        "gpu_memory_budget": 1024 * 1024 * 1024,
        "seed": 7,
    }
    parquet_model = fb.train(params, parquet, label="target", num_boost_round=3)
    parquet_log = capfd.readouterr().err
    csv_model = fb.train(params, csv, label="target", num_boost_round=3)
    csv_log = capfd.readouterr().err

    expected_blocks = x.shape[1] // params["cols_per_block"]
    runtime_streams = []
    for log in (parquet_log, csv_log):
        assert "explicit manual override" in log
        assert "[full resident]" in log
        assert f"resident_blocks={expected_blocks}/{expected_blocks}" in log
        match = re.search(r"GPU_PATH .* hist_streams=(\d+)", log)
        assert match, log
        runtime_streams.append(int(match.group(1)))
    assert runtime_streams == [2, 2]

    parquet_json = tmp_path / "parquet.json"
    csv_json = tmp_path / "csv.json"
    parquet_model.save_model(str(parquet_json))
    csv_model.save_model(str(csv_json))
    assert parquet_json.read_bytes() == csv_json.read_bytes()
