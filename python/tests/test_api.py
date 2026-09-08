"""API 表面:参数解析、early stopping、evals_result、模型互通、错误处理。"""
import csv as csv_module
import gzip
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import ferrisboost as fb

FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures"


def toy(n=300, m=5, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 8, size=(n, m)).astype(np.float32)
    y = (x[:, 0] - x[:, 1] + rng.normal(0, 0.5, n) > 0).astype(np.float32)
    return x, y


BASE = {"objective": "binary:logistic", "max_depth": 3, "eta": 0.3, "nthread": 1}


# ---------------------------------------------------------------- 参数
def test_unknown_param_raises():
    x, y = toy()
    with pytest.raises(ValueError, match="不认识的参数"):
        fb.train({**BASE, "max_dpeth": 3}, x, label=y, num_boost_round=2)


def test_aliases_are_accepted():
    x, y = toy()
    a = fb.train({**BASE, "eta": 0.1, "lambda": 2.0}, x, label=y, num_boost_round=3)
    b = fb.train({"objective": "binary:logistic", "max_depth": 3, "nthread": 1,
                  "learning_rate": 0.1, "reg_lambda": 2.0},
                 x, label=y, num_boost_round=3)
    assert a.predict(x).tobytes() == b.predict(x).tobytes(), "别名和本名结果应该一样"


def test_unimplemented_param_is_explicit():
    x, y = toy()
    with pytest.raises(ValueError, match="L1 正则"):
        fb.train({**BASE, "alpha": 1.0}, x, label=y, num_boost_round=2)


def test_bad_objective_lists_the_supported_ones():
    x, y = toy()
    with pytest.raises(ValueError) as e:
        fb.train({**BASE, "objective": "rank:pairwise"}, x, label=y, num_boost_round=2)
    assert "binary:logistic" in str(e.value)


def test_max_bin_over_255_rejected():
    x, y = toy()
    with pytest.raises(ValueError, match="max_bin"):
        fb.train({**BASE, "max_bin": 256}, x, label=y, num_boost_round=2)


# ---------------------------------------------------------------- 训练控制
def test_early_stopping_and_evals_result():
    x, y = toy()
    xv, yv = toy(n=120, seed=1)
    result = {}
    model = fb.train(BASE, x, label=y, num_boost_round=100,
                     evals=[(xv, yv, "valid")], early_stopping_rounds=3,
                     evals_result=result)

    assert model.best_iteration is not None
    assert model.num_trees == model.best_iteration + 1 + 3, "耐心数错了"
    assert model.num_trees < 100, "早停没生效"
    assert list(result) == ["valid"]
    assert list(result["valid"]) == ["logloss"]
    assert len(result["valid"]["logloss"]) == model.num_trees
    assert result["valid"]["logloss"][model.best_iteration] == pytest.approx(
        model.best_score, rel=1e-6)


def test_callbacks_can_stop_training():
    x, y = toy()
    seen = []

    def stop_at_four(round_idx, metrics):
        seen.append(round_idx)
        return round_idx < 3          # 返回 False 就停

    model = fb.train(BASE, x, label=y, num_boost_round=50, callbacks=[stop_at_four])
    assert seen == [0, 1, 2, 3]
    assert model.num_trees == 4


def test_callback_exception_propagates():
    x, y = toy()

    def boom(round_idx, metrics):
        raise KeyError("回调炸了")

    with pytest.raises(KeyError, match="回调炸了"):
        fb.train(BASE, x, label=y, num_boost_round=5, callbacks=[boom])


def test_callback_sees_metrics():
    x, y = toy()
    xv, yv = toy(n=100, seed=2)
    seen = {}

    def watch(round_idx, metrics):
        seen.update(metrics)
        return True

    fb.train(BASE, x, label=y, num_boost_round=3,
             evals=[(xv, yv, "valid")], callbacks=[watch])
    assert "valid" in seen and "logloss" in seen["valid"]


def test_verbose_eval_prints(capfd):
    x, y = toy()
    xv, yv = toy(n=100, seed=3)
    fb.train(BASE, x, label=y, num_boost_round=4,
             evals=[(xv, yv, "valid")], verbose_eval=2)
    out = capfd.readouterr().out
    assert "valid-logloss" in out


def test_normal_runtime_summary_and_quiet_mode(capfd):
    x, y = toy(n=80)
    fb.train(BASE, x, label=y, num_boost_round=1)
    normal = capfd.readouterr()
    assert "CPU_THREADS" in normal.err
    assert "SETUP" in normal.err and "TRAIN" in normal.err and "END_TO_END" in normal.err

    fb.train(BASE, x, label=y, num_boost_round=1, quiet=True)
    quiet = capfd.readouterr()
    assert quiet.out == ""
    assert quiet.err == ""


# ---------------------------------------------------------------- 模型互通
def test_save_load_round_trip(tmp_path):
    x, y = toy()
    model = fb.train(BASE, x, label=y, num_boost_round=5)
    path = tmp_path / "m.json"
    model.save_model(str(path))

    back = fb.Model.load_model(str(path))
    assert back.num_trees == model.num_trees
    assert back.predict(x, output_margin=True).tobytes() == \
        model.predict(x, output_margin=True).tobytes()


def test_model_io_progress_and_quiet(tmp_path, capfd):
    x, y = toy(n=80)
    model = fb.train(BASE, x, label=y, num_boost_round=2)
    capfd.readouterr()
    path = tmp_path / "logged.json"
    model.save_model(str(path))
    saved = capfd.readouterr().err
    assert "MODEL_SAVE start" in saved and "MODEL_SAVE complete" in saved
    assert "bytes_total=" in saved and "model_io_threads=auto" in saved and "workers=1" in saved

    fb.Model.load_model(str(path))
    loaded = capfd.readouterr().err
    assert "MODEL_LOAD start" in loaded and "MODEL_LOAD complete" in loaded

    model.save_model(str(tmp_path / "quiet.json"), quiet=True)
    fb.Model.load_model(str(path), quiet=True)
    assert capfd.readouterr().err == ""

    model.save_model(str(tmp_path / "auto.json"), model_io_threads=0)
    fb.Model.load_model(str(path), model_io_threads=0)


def test_xgboost_can_load_our_model(tmp_path):
    xgb = pytest.importorskip("xgboost")
    x, y = toy()
    model = fb.train(BASE, x, label=y, num_boost_round=5)
    path = tmp_path / "m.json"
    model.save_model(str(path))

    bst = xgb.Booster(model_file=str(path))
    theirs = bst.predict(xgb.DMatrix(x), output_margin=True)
    ours = model.predict(x, output_margin=True)
    assert np.allclose(ours, theirs, atol=1e-6), "XGBoost 读我们的模型算出来不一样"


def test_we_can_load_an_xgboost_model(tmp_path):
    xgb = pytest.importorskip("xgboost")
    x, y = toy()
    path = tmp_path / "x.json"
    bst = xgb.train({"objective": "binary:logistic", "max_depth": 3, "eta": 0.3,
                     "base_score": 0.5, "tree_method": "hist", "nthread": 1},
                    xgb.DMatrix(x, label=y), num_boost_round=5)
    bst.save_model(str(path))

    ours = fb.Model.load_model(str(path))
    assert np.allclose(ours.predict(x, output_margin=True),
                       bst.predict(xgb.DMatrix(x), output_margin=True), atol=1e-6)


# ---------------------------------------------------------------- 文件入口
def test_csv_and_parquet_entries(tmp_path):
    pd = pytest.importorskip("pandas")
    x, y = toy(n=200)
    df = pd.DataFrame(x, columns=[f"f{i}" for i in range(x.shape[1])])
    df["target"] = y

    csv_path = tmp_path / "d.csv"
    pq_path = tmp_path / "d.parquet"
    df.to_csv(csv_path, index=False)
    df.to_parquet(pq_path, index=False)

    from_mem = fb.train(BASE, x, label=y, num_boost_round=4)
    from_csv = fb.train(BASE, str(csv_path), label="target", num_boost_round=4)
    from_pq = fb.train(BASE, str(pq_path), label="target", num_boost_round=4)

    ref = from_mem.predict(x, output_margin=True)
    assert from_csv.predict(x, output_margin=True).tobytes() == ref.tobytes()
    assert from_pq.predict(x, output_margin=True).tobytes() == ref.tobytes()


def test_float64_file_inputs_share_float32_quantization_semantics(tmp_path):
    """Parquet/CSV/CSV.gz cast numeric input to f32 before cuts are built."""
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")

    n = 320
    row = np.arange(n, dtype=np.float64)
    # Adjacent f64 values within each pair differ, but the perturbation is below
    # one f32 ULP around 1.0. Comparing against the explicitly rounded NumPy
    # path makes the precision boundary observable rather than incidental.
    f0 = 1.0 + (row // 2) * np.float64(2.0**-20) + (row % 2) * np.float64(2.0**-40)
    f1 = np.sin(row / 17.0).astype(np.float64) + (row % 3) * np.float64(2.0**-45)
    f2 = ((row % 11) - 5.0).astype(np.float64)
    f1[::37] = np.nan
    x64 = np.column_stack((f0, f1, f2))
    y64 = ((row % 7) < 3).astype(np.float64)

    parquet = tmp_path / "precision.parquet"
    csv = tmp_path / "precision.csv"
    csv_gz = tmp_path / "precision.csv.gz"
    table = pa.table({"f0": f0, "f1": f1, "f2": f2, "target": y64})
    pq.write_table(table, parquet, row_group_size=43)
    rows = np.column_stack((x64, y64))
    header = ["f0", "f1", "f2", "target"]

    def write_csv(path, opener):
        with opener(path, "wt", newline="") as stream:
            writer = csv_module.writer(stream)
            writer.writerow(header)
            for values in rows:
                writer.writerow(["" if np.isnan(value) else format(value, ".17g")
                                 for value in values])

    write_csv(csv, open)
    write_csv(csv_gz, gzip.open)

    params = {**BASE, "cols_per_block": 2, "max_bin": 64}
    models = [
        fb.train(params, x64.astype(np.float32), label=y64.astype(np.float32),
                 num_boost_round=6, quiet=True),
        *(fb.train(params, path, label="target", num_boost_round=6, quiet=True)
          for path in (parquet, csv, csv_gz)),
    ]
    model_bytes = []
    for index, model in enumerate(models):
        path = tmp_path / f"precision-model-{index}.json"
        model.save_model(str(path), quiet=True)
        model_bytes.append(path.read_bytes())
    # NumPy is positional while files carry names, so its JSON metadata differs;
    # all three named file formats must still be byte-identical.
    assert model_bytes[1] == model_bytes[2] == model_bytes[3]

    expected = models[0].predict(x64.astype(np.float32), output_margin=True).tobytes()
    for model, path in zip(models[1:], (parquet, csv, csv_gz)):
        assert model.predict(path, output_margin=True).tobytes() == expected


@pytest.mark.parametrize("fmt", ["parquet", "csv", "csv.gz"])
@pytest.mark.parametrize("multi", [False, True])
def test_ingest_threads_preserve_models_and_row_order(tmp_path, fmt, multi):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    x, y = toy(n=800, m=12, seed=23)
    names = [f"f{i}" for i in range(x.shape[1])]

    ranges = [(0, len(x))] if not multi else [(0, 317), (317, len(x))]
    paths = []
    for index, (start, stop) in enumerate(ranges):
        path = tmp_path / f"part-{index}.{fmt}"
        if fmt == "parquet":
            columns = {name: x[start:stop, col] for col, name in enumerate(names)}
            columns["target"] = y[start:stop]
            pq.write_table(pa.table(columns), path, row_group_size=73)
        else:
            np.savetxt(
                path,
                np.column_stack((x[start:stop], y[start:stop])),
                delimiter=",",
                header=",".join([*names, "target"]),
                comments="",
            )
        paths.append(path)

    data = paths if multi else paths[0]
    models = []
    for ingest_threads in (0, 1, 8):
        models.append(
            fb.train(
                {**BASE, "ingest_threads": ingest_threads, "cols_per_block": 4},
                data,
                label="target",
                num_boost_round=4,
                quiet=True,
            )
        )
    saved = []
    for index, model in enumerate(models):
        path = tmp_path / f"model-{index}-{fmt}-{multi}.json"
        model.save_model(str(path), quiet=True)
        saved.append(path.read_bytes())
    assert saved[0] == saved[1] == saved[2]
    reference = models[0].predict(data, predict_threads=1).tobytes()
    predictions = [
        model.predict(data, predict_threads=threads).tobytes()
        for model in models
        for threads in (0, 1, 2, 4)
    ]
    assert all(prediction == reference for prediction in predictions)


def test_file_prediction_compacts_used_features_but_validates_full_schema(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    rng = np.random.default_rng(20260908)
    x = np.zeros((1200, 8), dtype=np.float32)
    x[:, 0] = rng.normal(size=len(x))
    y = (x[:, 0] > 0).astype(np.float32)
    names = [f"f{i}" for i in range(x.shape[1])]
    train_path = tmp_path / "compact-train.parquet"
    pq.write_table(
        pa.table({**{name: x[:, i] for i, name in enumerate(names)}, "target": y}),
        train_path,
    )
    model = fb.train(
        {**BASE, "max_depth": 3, "nthread": 1},
        train_path,
        label="target",
        num_boost_round=8,
        quiet=True,
    )

    before = tmp_path / "canonical-before.json"
    model.save_model(str(before), quiet=True)
    want = model.predict(x, output_margin=True, predict_threads=1)
    got = model.predict(train_path, output_margin=True, predict_threads=4)
    assert got.tobytes() == want.tobytes()
    after = tmp_path / "canonical-after.json"
    model.save_model(str(after), quiet=True)
    assert before.read_bytes() == after.read_bytes()

    # f7 is constant and cannot participate in a split, but it remains part of
    # the canonical model schema. Compaction must never weaken schema checks.
    missing_unused = tmp_path / "missing-unused.parquet"
    pq.write_table(
        pa.table({name: x[:, i] for i, name in enumerate(names[:-1])}),
        missing_unused,
    )
    with pytest.raises(ValueError, match="f7"):
        model.predict(missing_unused)


def test_io_thread_auto_and_negative_validation():
    x, y = toy(n=20)
    fb.train({**BASE, "ingest_threads": 0}, x, label=y, num_boost_round=1)
    fb.train({**BASE, "model_io_threads": 0}, x, label=y, num_boost_round=1)
    with pytest.raises(ValueError, match="ingest_threads"):
        fb.train({**BASE, "ingest_threads": -1}, x, label=y, num_boost_round=1)
    with pytest.raises(ValueError, match="model_io_threads"):
        fb.train({**BASE, "model_io_threads": -1}, x, label=y, num_boost_round=1)


def test_headless_csv_uses_positional_label_and_predicts_files(tmp_path):
    """header=False must not consume the first data row as a header."""
    x, y = toy(n=200)
    rows = np.column_stack((x[:, :2], y, x[:, 2:]))
    path = tmp_path / "headless.csv"
    np.savetxt(path, rows, delimiter=",")

    model = fb.train(BASE, path, label=2, header=False, num_boost_round=4)
    want = model.predict(x, output_margin=True)

    predict_path = tmp_path / "headless-predict.csv"
    np.savetxt(predict_path, x, delimiter=",")
    got = model.predict(predict_path, output_margin=True, header=False)
    assert isinstance(got, np.ndarray)
    assert got.dtype == np.float32
    assert got.shape == (len(x),)
    assert got.tobytes() == want.tobytes()

    saved = tmp_path / "positional.json"
    model.save_model(str(saved))
    loaded = fb.Model.load_model(str(saved))
    assert loaded.predict(predict_path, output_margin=True, header=False).tobytes() == want.tobytes()


def test_file_ingest_progress_reports_resolved_workers_and_quiet(tmp_path, capfd):
    x, y = toy(n=200)
    path = tmp_path / "progress.csv.gz"
    names = [f"f{i}" for i in range(x.shape[1])]
    np.savetxt(
        path,
        np.column_stack((x, y)),
        delimiter=",",
        header=",".join([*names, "target"]),
        comments="",
    )
    fb.train(
        {**BASE, "ingest_threads": 4, "cols_per_block": 2},
        path,
        label="target",
        num_boost_round=1,
    )
    log = capfd.readouterr().err
    assert "INGEST start" in log and "INGEST complete" in log
    assert "files=1" in log and "ingest_threads=4" in log and "workers=3" in log
    assert "rows=200" in log

    fb.train(
        {**BASE, "ingest_threads": 0, "cols_per_block": 2},
        path,
        label="target",
        num_boost_round=1,
    )
    auto_log = capfd.readouterr().err
    assert "ingest_threads=auto" in auto_log and "workers=" in auto_log

    fb.train(
        {**BASE, "ingest_threads": 4, "cols_per_block": 2},
        path,
        label="target",
        num_boost_round=1,
        quiet=True,
    )
    assert capfd.readouterr().err == ""


def test_headless_csv_rejects_bad_label_and_schema_mode_mix(tmp_path):
    x, y = toy(n=40)
    rows = np.column_stack((x, y))
    path = tmp_path / "headless.csv"
    np.savetxt(path, rows, delimiter=",")

    with pytest.raises(ValueError, match="label|下标|越界"):
        fb.train(BASE, path, label=99, header=False, num_boost_round=2)
    with pytest.raises(ValueError, match="label|下标"):
        fb.train(BASE, path, label="target", header=False, num_boost_round=2)

    model = fb.train(BASE, path, label=x.shape[1], header=False, num_boost_round=2)
    with pytest.raises(ValueError, match="header=False"):
        model.predict(path)

    named = tmp_path / "named.csv"
    np.savetxt(named, rows, delimiter=",", header="f0,f1,f2,f3,f4,target", comments="")
    named_model = fb.train(BASE, named, label="target", num_boost_round=2)
    with pytest.raises(ValueError, match="命名模式"):
        named_model.predict(path, header=False)


def test_missing_file_is_a_clear_error():
    with pytest.raises((ValueError, FileNotFoundError)):
        fb.train(BASE, "/nope/does_not_exist.parquet", label="target", num_boost_round=2)


# ---------------------------------------------------------------- 形状 / GIL
def test_shape_mismatch_messages():
    x, y = toy()
    with pytest.raises(ValueError, match="标签"):
        fb.train(BASE, x, label=y[:10], num_boost_round=2)
    model = fb.train(BASE, x, label=y, num_boost_round=2)
    with pytest.raises(ValueError, match="特征"):
        model.predict(x[:, :2])


@pytest.mark.parametrize("objective", ["binary:logistic", "reg:squarederror"])
def test_predict_accepts_all_numpy_2d_layouts(objective):
    """`Model.predict` 是直接暴露的 Rust API，不能假定调用者先走 train。"""
    x, y = toy(n=120, m=5, seed=7)
    if objective == "reg:squarederror":
        y = (x[:, 0] * 0.5 - x[:, 1] * 0.25).astype(np.float32)
    model = fb.train({**BASE, "objective": objective}, x, label=y, num_boost_round=4)

    fortran = np.asfortranarray(x)
    padded = np.empty((x.shape[0], x.shape[1] + 2), dtype=np.float32)
    padded[:, :x.shape[1]] = x
    strided = padded[:, :x.shape[1]]
    assert fortran.flags["F_CONTIGUOUS"] and not fortran.flags["C_CONTIGUOUS"]
    assert not strided.flags["C_CONTIGUOUS"]

    want = model.predict(x, output_margin=True, predict_threads=1).tobytes()
    for threads in (0, 1, 2, 4):
        assert model.predict(
            x, output_margin=True, predict_threads=threads
        ).tobytes() == want
        assert model.predict(
            fortran, output_margin=True, predict_threads=threads
        ).tobytes() == want
        assert model.predict(
            strided, output_margin=True, predict_threads=threads
        ).tobytes() == want

def test_gil_is_released_during_training():
    """训练时要放掉 GIL,否则调用方的其它线程会被卡住。"""
    import threading
    import time

    x, y = toy(n=4000, m=20)
    ticks = []
    stop = threading.Event()

    def ticker():
        while not stop.is_set():
            ticks.append(time.perf_counter())
            time.sleep(0.001)

    t = threading.Thread(target=ticker)
    t.start()
    fb.train({**BASE, "max_depth": 6, "nthread": 1}, x, label=y, num_boost_round=30)
    stop.set()
    t.join()
    assert len(ticks) > 5, f"训练期间别的线程只跑了 {len(ticks)} 次,GIL 大概没放开"


def test_gil_is_released_during_numpy_prediction():
    """NumPy scoring can be long-running and must not retain the Python GIL."""
    import threading
    import time

    x, y = toy(n=4000, m=20, seed=91)
    model = fb.train(
        {**BASE, "max_depth": 6, "nthread": 1},
        x,
        label=y,
        num_boost_round=30,
    )
    predict_x = np.tile(x, (20, 1))
    ticks = []
    stop = threading.Event()

    def ticker():
        while not stop.is_set():
            ticks.append(time.perf_counter())
            time.sleep(0.001)

    thread = threading.Thread(target=ticker)
    thread.start()
    model.predict(predict_x, predict_threads=1)
    stop.set()
    thread.join()
    assert len(ticks) > 5, (
        f"prediction 期间别的线程只跑了 {len(ticks)} 次,GIL 大概没放开"
    )


# ---------------------------------------------------------------- 量化缓存
def _write_parquet(tmp_path, n=400, m=6, seed=0):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    x, y = toy(n=n, m=m, seed=seed)
    df = pd.DataFrame(x, columns=[f"f{i}" for i in range(m)])
    df["target"] = y
    path = tmp_path / "d.parquet"
    df.to_parquet(path, index=False)
    return path, x, y


def test_cache_budget_must_not_be_negative(tmp_path):
    path, x, y = _write_parquet(tmp_path)
    with pytest.raises(ValueError, match="cache_budget_bytes"):
        fb.train({**BASE, "cache_budget_bytes": -1}, str(path), label="target",
                 num_boost_round=2)


def test_cache_path_accepts_pathlib(tmp_path):
    """pathlib.Path 和 str 都该收 —— 用户手里多半是 Path。"""
    path, x, y = _write_parquet(tmp_path)
    model = fb.train({**BASE, "cache_path": tmp_path / "cache_dir"}, str(path),
                     label="target", num_boost_round=3)
    assert (tmp_path / "cache_dir" / "manifest.json").exists()
    assert model.num_trees == 3


def test_same_cache_path_twice_gives_identical_margins(tmp_path):
    """第二次跑跳过 cuts 扫描和量化,但预测必须**逐位相同**。

    缓存复用最危险的失败方式不是报错,是悄悄用了不同的 bin —— 那样
    模型会变而没人知道。
    """
    path, x, y = _write_parquet(tmp_path)
    cache = tmp_path / "reuse"
    p = {**BASE, "cache_path": cache}
    first = fb.train(p, str(path), label="target", num_boost_round=5)
    second = fb.train(p, str(path), label="target", num_boost_round=5)
    assert (first.predict(x, output_margin=True).tobytes()
            == second.predict(x, output_margin=True).tobytes())


def test_cache_rejects_a_different_max_bin(tmp_path):
    """换了 max_bin,缓存里的 bin 就没意义了,必须报错不是静默复用。"""
    path, x, y = _write_parquet(tmp_path)
    cache = tmp_path / "mb"
    fb.train({**BASE, "max_bin": 64, "cache_path": cache}, str(path),
             label="target", num_boost_round=2)
    with pytest.raises((ValueError, RuntimeError), match="max_bin|缓存"):
        fb.train({**BASE, "max_bin": 32, "cache_path": cache}, str(path),
                 label="target", num_boost_round=2)


def test_cache_matches_no_cache_bit_for_bit(tmp_path):
    """开不开缓存都只是取数路径,数值不该有任何差别。"""
    path, x, y = _write_parquet(tmp_path)
    plain = fb.train(BASE, str(path), label="target", num_boost_round=4)
    cached = fb.train({**BASE, "cache_path": tmp_path / "c"}, str(path),
                      label="target", num_boost_round=4)
    # 0 = 强制落盘
    spilled = fb.train({**BASE, "cache_budget_bytes": 0}, str(path),
                       label="target", num_boost_round=4)
    ref = plain.predict(x, output_margin=True).tobytes()
    assert cached.predict(x, output_margin=True).tobytes() == ref
    assert spilled.predict(x, output_margin=True).tobytes() == ref
