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


# ----------------------------------------------------------------------------
# GPU prediction: categorical models and per-row base_margin.
#
# Contract (same as numeric models): the device margin equals the CPU margin bit
# for bit. Batches must be large enough for the transfer-aware selector to pick
# the device (rows x tree depth >= 8M work units), so these inputs are tiled.
# ----------------------------------------------------------------------------
BIG_ROWS = 70_000


def _tile(x, rows=BIG_ROWS):
    return np.ascontiguousarray(np.tile(x, (rows // len(x) + 1, 1))[:rows])


def _cat_data(n=3000, levels=(3, 12, 40), seed=0, n_num=2, objective="reg:squarederror"):
    """Numeric columns first, then integer category codes, with some NaN."""
    rng = np.random.default_rng(seed)
    cols = [rng.normal(size=n).astype(np.float32) for _ in range(n_num)]
    cats = [rng.integers(0, k, n).astype(np.float32) for k in levels]
    signal = cols[0] * 0.7
    for c, k in zip(cats, levels):
        signal = signal + rng.normal(0, 1, k)[c.astype(int)]
    if objective == "multi:softprob":
        y = np.digitize(signal + rng.normal(0, 0.4, n), [-0.7, 0.7]).astype(np.float32)
    elif objective == "binary:logistic":
        y = (signal + rng.normal(0, 0.5, n) > 0).astype(np.float32)
    else:
        y = (signal + rng.normal(0, 0.3, n)).astype(np.float32)
    x = np.stack(cols + cats, axis=1)
    x[rng.random(x.shape) < 0.05] = np.nan
    return x, y, ["q"] * n_num + ["c"] * len(levels)


def _probe(x, n_num, seed=1):
    """Training-like rows plus the awkward category values scattered through them."""
    rng = np.random.default_rng(seed)
    p = _tile(x).copy()
    awkward = np.array([-1.0, -0.5, 0.0, 2.7, 3.0, 11.9, 255.0, 256.0, 300.0, 65535.0, 65536.0,
                        16777215.0, 16777216.0, 1e30, np.inf, -np.inf, np.nan], dtype=np.float32)
    for col in range(n_num, x.shape[1]):
        rows = rng.random(len(p)) < 0.2
        p[rows, col] = rng.choice(awkward, rows.sum())
    return p


def _same_bits(a, b):
    assert a.shape == b.shape and a.tobytes() == b.tobytes()


@cuda
@pytest.mark.parametrize("params", [
    {},                                              # one-hot (<= 3 categories) and partition splits
    {"max_cat_to_onehot": 64},                       # one-hot everywhere
    {"max_cat_to_onehot": 1, "max_depth": 7},        # partition everywhere
])
def test_gpu_prediction_of_categorical_models_is_bit_identical(params):
    x, y, ft = _cat_data()
    model = fb.train({"objective": "reg:squarederror", "max_depth": 5, "eta": 0.3, "nthread": 2,
                      **params}, x, label=y,
                     num_boost_round=40, feature_types=ft, quiet=True)
    probe = _probe(x, n_num=2)
    cpu = model.predict(probe, output_margin=True, predict_threads=1)
    gpu = model.predict(probe, output_margin=True, use_gpu=True)
    _same_bits(cpu, gpu)
    # Repeated call reuses the cached device model and staging.
    _same_bits(cpu, model.predict(probe, output_margin=True, use_gpu=True))
    # Non-C-contiguous input and a tree prefix take other host paths into the same kernel.
    _same_bits(cpu, model.predict(np.asfortranarray(probe), output_margin=True, use_gpu=True))
    _same_bits(model.predict(probe, output_margin=True, n_trees=17, predict_threads=1),
               model.predict(probe, output_margin=True, n_trees=17, use_gpu=True))
    _same_bits(model.predict(probe, predict_threads=1), model.predict(probe, use_gpu=True))


@cuda
def test_gpu_prediction_of_categorical_binary_model_from_files(tmp_path):
    pd = pytest.importorskip("pandas")
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    x, y, ft = _cat_data(objective="binary:logistic", seed=3)
    frame = pd.DataFrame({f"f{i}": x[:, i] for i in range(x.shape[1])})
    model = fb.train({"objective": "binary:logistic", "max_depth": 5, "nthread": 2},
                     frame, label=y, num_boost_round=40, feature_types=ft, quiet=True)
    probe = _probe(x, n_num=2)
    path = tmp_path / "probe.parquet"
    pq.write_table(pa.table({f"f{i}": probe[:, i] for i in range(probe.shape[1])}), path)
    cpu = model.predict(str(path), output_margin=True, predict_threads=1)
    _same_bits(cpu, model.predict(str(path), output_margin=True, use_gpu=True))
    probe_frame = pd.DataFrame({f"f{i}": probe[:, i] for i in range(probe.shape[1])})
    _same_bits(cpu, model.predict(probe_frame, output_margin=True, use_gpu=True))
    _same_bits(model.predict(str(path), predict_threads=1),
               model.predict(str(path), use_gpu=True))


@cuda
def test_gpu_prediction_of_multiclass_categorical_model_is_bit_identical():
    x, y, ft = _cat_data(objective="multi:softprob", seed=4)
    model = fb.train({"objective": "multi:softprob", "num_class": 3, "max_depth": 5,
                      "nthread": 2}, x, label=y, num_boost_round=30, feature_types=ft, quiet=True)
    probe = _probe(x, n_num=2)
    cpu = model.predict(probe, output_margin=True, predict_threads=1)
    gpu = model.predict(probe, output_margin=True, use_gpu=True)
    assert gpu.shape == (len(probe), 3)
    _same_bits(cpu, gpu)
    _same_bits(model.predict(probe, predict_threads=1), model.predict(probe, use_gpu=True))


@cuda
def test_gpu_prediction_of_xgboost_categorical_model_with_wide_category_sets(tmp_path):
    xgb = pytest.importorskip("xgboost")
    rng = np.random.default_rng(13)
    n = 6000
    x = np.stack([rng.integers(0, 600, n), rng.integers(0, 5, n),
                  rng.normal(size=n)], axis=1).astype(np.float32)
    y = ((x[:, 0] % 7 == 0) * 2.0 + (x[:, 0] > 400) + (x[:, 1] == 3)
         + 0.3 * x[:, 2] + rng.normal(0, 0.1, n)).astype(np.float32)
    ft = ["c", "c", "q"]
    d = xgb.DMatrix(x, label=y, feature_types=ft, enable_categorical=True)
    bst = xgb.train({"tree_method": "hist", "max_depth": 6, "max_cat_to_onehot": 1,
                     "max_bin": 512, "nthread": 1}, d, num_boost_round=40)
    path = str(tmp_path / "wide.json")
    bst.save_model(path)
    import json
    sets = [t["categories"] for t in json.load(open(path))["learner"]["gradient_booster"]
            ["model"]["trees"]]
    assert max(max(s, default=0) for s in sets) > 255, "sets must hold ids beyond 8 bits"
    model = fb.Model.load_model(path, quiet=True)

    probe = _tile(x).copy()
    rows = rng.random(len(probe)) < 0.25
    probe[rows, 0] = rng.choice(
        np.array([-3.0, 0.5, 599.0, 600.0, 4000.0, 70000.0, 16777216.0, np.nan], np.float32),
        rows.sum())
    probe[rng.random(len(probe)) < 0.03, 1] = np.nan
    cpu = model.predict(probe, output_margin=True, predict_threads=1)
    _same_bits(cpu, model.predict(probe, output_margin=True, use_gpu=True))
    # The CPU path itself agrees with XGBoost (small slice: XGBoost is the slow side).
    head = probe[:2000]
    np.testing.assert_array_equal(
        model.predict(head, output_margin=True),
        bst.predict(xgb.DMatrix(head, feature_types=ft, enable_categorical=True),
                    output_margin=True))


def _margin_data(objective, n=BIG_ROWS, m=12, seed=2):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, m)).astype(np.float32)
    x[rng.random(x.shape) < 0.03] = np.nan
    signal = np.nan_to_num(x[:, 0]) + 0.5 * np.nan_to_num(x[:, 1]) * np.nan_to_num(x[:, 2])
    if objective == "multi:softprob":
        y = np.digitize(signal + rng.normal(0, 0.5, n), [-0.6, 0.6]).astype(np.float32)
    elif objective == "binary:logistic":
        y = (signal + rng.normal(0, 0.5, n) > 0).astype(np.float32)
    else:
        y = (signal + rng.normal(0, 0.3, n)).astype(np.float32)
    return x, y


MARGIN_CASES = [
    ("reg:squarederror", {}, 1),
    ("binary:logistic", {}, 1),
    ("multi:softprob", {"num_class": 3}, 3),
    ("reg:quantileerror", {"quantile_alpha": [0.2, 0.5, 0.8]}, 3),
]


@cuda
@pytest.mark.parametrize("objective, extra, k", MARGIN_CASES)
def test_gpu_prediction_with_base_margin_is_bit_identical(objective, extra, k, tmp_path):
    pd = pytest.importorskip("pandas")
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    x, y = _margin_data(objective)
    frame = pd.DataFrame({f"f{i}": x[:5000, i] for i in range(x.shape[1])})
    model = fb.train({"objective": objective, "max_depth": 6, "eta": 0.2, "nthread": 2, **extra},
                     frame, label=y[:5000], num_boost_round=25, quiet=True)
    rng = np.random.default_rng(9)
    shape = (len(x),) if k == 1 else (len(x), k)
    margin = rng.normal(0, 1.5, size=shape).astype(np.float32)

    cpu = model.predict(x, output_margin=True, base_margin=margin, predict_threads=1)
    gpu = model.predict(x, output_margin=True, base_margin=margin, use_gpu=True)
    _same_bits(cpu, gpu)
    # The margin actually replaced the intercept (and differs from a plain call).
    assert cpu.tobytes() != model.predict(x, output_margin=True, predict_threads=1).tobytes()
    _same_bits(model.predict(x, base_margin=margin, predict_threads=1),
               model.predict(x, base_margin=margin, use_gpu=True))
    _same_bits(cpu, model.predict(np.asfortranarray(x), output_margin=True,
                                  base_margin=margin, use_gpu=True))
    # Plain GPU prediction (no margin) still works on the same cached model afterwards.
    _same_bits(model.predict(x, output_margin=True, predict_threads=1),
               model.predict(x, output_margin=True, use_gpu=True))

    # File input: the margin comes from column(s) next to the features.
    table = {f"f{i}": x[:, i] for i in range(x.shape[1])}
    if k == 1:
        table["offset"] = margin
        cols = "offset"
    else:
        cols = [f"offset{c}" for c in range(k)]
        for c, name in enumerate(cols):
            table[name] = margin[:, c]
    path = str(tmp_path / "margin.parquet")
    pq.write_table(pa.table(table), path)
    file_cpu = model.predict(path, output_margin=True, base_margin=cols, predict_threads=1)
    _same_bits(file_cpu, cpu)
    _same_bits(file_cpu, model.predict(path, output_margin=True, base_margin=cols, use_gpu=True))
    _same_bits(model.predict(path, base_margin=cols, predict_threads=1),
               model.predict(path, base_margin=cols, use_gpu=True))


@cuda
def test_gpu_prediction_with_base_margin_on_a_categorical_model():
    x, y, ft = _cat_data(seed=6, objective="binary:logistic")
    model = fb.train({"objective": "binary:logistic", "max_depth": 5, "nthread": 2}, x, label=y,
                     num_boost_round=40, feature_types=ft, quiet=True)
    probe = _probe(x, n_num=2)
    margin = np.random.default_rng(1).normal(0, 2, len(probe)).astype(np.float32)
    cpu = model.predict(probe, output_margin=True, base_margin=margin, predict_threads=1)
    _same_bits(cpu, model.predict(probe, output_margin=True, base_margin=margin, use_gpu=True))
    _same_bits(model.predict(probe, base_margin=margin, predict_threads=1),
               model.predict(probe, base_margin=margin, use_gpu=True))
    with pytest.raises(ValueError, match="base_margin has"):
        model.predict(probe, base_margin=margin[:-1], use_gpu=True)
