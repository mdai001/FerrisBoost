"""multi:softprob / multi:softmax:训练、预测形状、指标、模型交换,和 XGBoost 逐位对齐。"""
import json

import numpy as np
import pytest

import ferrisboost as fb

xgb = pytest.importorskip("xgboost")


def make(n=3000, k=4, m=5, seed=0, missing=0.05):
    # 低基数特征:分箱无损,和 XGBoost 可以逐位比。
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 9, size=(n, m)).astype(np.float32)
    x[rng.random(x.shape) < missing] = np.nan
    s = np.nan_to_num(x[:, 0], nan=4.0) - np.nan_to_num(x[:, 1], nan=4.0) + rng.normal(0, 2, n)
    edges = np.quantile(s, np.linspace(0, 1, k + 1)[1:-1])
    y = np.digitize(s, edges).astype(np.float32)
    return x, y


def xgb_train(params, x, y, w, rounds, evals=(), monotone=None, **kw):
    p = {**params, "tree_method": "hist", "nthread": 1}
    if monotone is not None:
        p["monotone_constraints"] = "(" + ",".join(map(str, monotone)) + ")"
    res = {}
    bst = xgb.train(p, xgb.DMatrix(x, label=y, weight=w), num_boost_round=rounds,
                    evals=[(xgb.DMatrix(ex, label=ey, weight=ew), nm) for ex, ey, ew, nm in evals],
                    evals_result=res, verbose_eval=False, **kw)
    return bst, res


@pytest.mark.parametrize("k", [2, 3, 7])
@pytest.mark.parametrize("objective", ["multi:softprob", "multi:softmax"])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("monotone", [None, (1, -1, 0, 0, 0)])
def test_bit_identical_to_xgboost(k, objective, weighted, monotone):
    x, y = make(k=k, seed=k)
    w = np.random.default_rng(k).uniform(0.2, 3, len(y)).astype(np.float32) if weighted else None
    p = {"objective": objective, "num_class": k, "max_depth": 4, "eta": 0.3, "max_bin": 255}
    bst, _ = xgb_train(p, x, y, w, 6, monotone=monotone)
    fp = {**p, "nthread": 1}
    if monotone is not None:
        fp["monotone_constraints"] = monotone
    m = fb.train(fp, x, label=y, weight=w, num_boost_round=6, quiet=True)
    d = xgb.DMatrix(x)
    np.testing.assert_array_equal(m.predict(x, output_margin=True),
                                  bst.predict(d, output_margin=True))
    ours, ref = m.predict(x), bst.predict(d)
    assert ours.shape == ref.shape == ((len(y), k) if objective == "multi:softprob" else (len(y),))
    np.testing.assert_array_equal(ours, ref)
    assert m.num_trees == 6 * k


@pytest.mark.parametrize("weighted", [False, True])
def test_metrics_and_early_stopping_match_xgboost(weighted):
    x, y = make(k=5, seed=11)
    vx, vy = make(n=1200, k=5, seed=12)
    rng = np.random.default_rng(3)
    w = rng.uniform(0.2, 3, len(y)).astype(np.float32) if weighted else None
    vw = rng.uniform(0.2, 3, len(vy)).astype(np.float32) if weighted else None
    p = {"objective": "multi:softprob", "num_class": 5, "max_depth": 5, "eta": 0.6,
         "max_bin": 255, "eval_metric": ["merror", "mlogloss"]}
    bst, ref = xgb_train(p, x, y, w, 200, evals=[(vx, vy, vw, "v")], early_stopping_rounds=4)
    ours = {}
    ev = (vx, vy, vw, "v") if weighted else (vx, vy, "v")
    m = fb.train({**p, "nthread": 1}, x, label=y, weight=w, evals=[ev], num_boost_round=200,
                 early_stopping_rounds=4, evals_result=ours, quiet=True)
    for name in ["merror", "mlogloss"]:
        np.testing.assert_allclose(ours["v"][name], ref["v"][name], rtol=2e-6, atol=1e-7)
    assert m.best_iteration == bst.best_iteration
    # 截断到最佳轮次:n_trees = (best_iteration + 1) * num_class
    upto = (m.best_iteration + 1) * 5
    np.testing.assert_array_equal(
        m.predict(vx, n_trees=upto),
        bst.predict(xgb.DMatrix(vx), iteration_range=(0, bst.best_iteration + 1)))


def test_default_metric_is_mlogloss():
    x, y = make(n=500, k=3)
    res = {}
    fb.train({"objective": "multi:softmax", "num_class": 3}, x, label=y, evals=[(x, y, "t")],
             num_boost_round=2, evals_result=res, quiet=True)
    assert list(res["t"]) == ["mlogloss"]


@pytest.mark.parametrize("objective", ["multi:softprob", "multi:softmax"])
def test_model_interchange_with_xgboost(tmp_path, objective):
    x, y = make(k=4, seed=5)
    p = {"objective": objective, "num_class": 4, "max_depth": 4}
    m = fb.train({**p, "nthread": 1}, x, label=y, num_boost_round=5, quiet=True)
    path = tmp_path / "fb.json"
    m.save_model(str(path), quiet=True)
    bst = xgb.Booster(model_file=str(path))
    np.testing.assert_allclose(bst.predict(xgb.DMatrix(x), output_margin=True),
                               m.predict(x, output_margin=True), rtol=0, atol=1e-6)
    again = fb.Model.load_model(str(path), quiet=True)
    np.testing.assert_array_equal(again.predict(x), m.predict(x))
    ref, _ = xgb_train(p, x, y, None, 5)
    ref_path = tmp_path / "xgb.json"
    ref.save_model(str(ref_path))
    loaded = fb.Model.load_model(str(ref_path), quiet=True)
    np.testing.assert_array_equal(loaded.predict(x), ref.predict(xgb.DMatrix(x)))


def test_file_training_and_prediction_match_arrays(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    x, y = make(k=3, seed=8)
    cols = {f"f{i}": x[:, i] for i in range(x.shape[1])}
    pq.write_table(pa.table({**cols, "y": y}), str(tmp_path / "t.parquet"))
    pq.write_table(pa.table(cols), str(tmp_path / "x.parquet"))
    p = {"objective": "multi:softprob", "num_class": 3, "nthread": 1}
    a = fb.train(p, str(tmp_path / "t.parquet"), label="y", num_boost_round=4, quiet=True)
    b = fb.train(p, x, label=y, num_boost_round=4, quiet=True)
    np.testing.assert_array_equal(a.predict(x), b.predict(x))
    np.testing.assert_array_equal(a.predict(str(tmp_path / "x.parquet")), b.predict(x))
    n = a.predict_to_parquet(str(tmp_path / "x.parquet"), str(tmp_path / "out.parquet"))
    out = pq.read_table(str(tmp_path / "out.parquet"))
    assert n == len(y) and out.column_names == ["prediction_0", "prediction_1", "prediction_2"]
    np.testing.assert_array_equal(np.column_stack([out[c].to_numpy() for c in out.column_names]),
                                  b.predict(x))


def test_softmax_parquet_writes_class_index(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    x, y = make(k=3, seed=9)
    pq.write_table(pa.table({f"f{i}": x[:, i] for i in range(x.shape[1])}),
                   str(tmp_path / "x.parquet"))
    m = fb.train({"objective": "multi:softmax", "num_class": 3}, x, label=y,
                 num_boost_round=3, quiet=True)
    m.predict_to_parquet(str(tmp_path / "x.parquet"), str(tmp_path / "o.parquet"),
                         prediction_column="cls")
    out = pq.read_table(str(tmp_path / "o.parquet"))
    assert out.column_names == ["cls"]
    np.testing.assert_array_equal(out["cls"].to_numpy(), m.predict(x))


@pytest.mark.parametrize("params, labels, match", [
    ({"objective": "multi:softprob"}, None, "num_class"),
    ({"objective": "multi:softprob", "num_class": 1}, None, "num_class"),
    ({"objective": "multi:softprob", "num_class": 3}, [0, 1, 3], "整数"),
    ({"objective": "multi:softprob", "num_class": 3}, [0, 1.5, 2], "整数"),
    ({"objective": "multi:softprob", "num_class": 3, "eval_metric": "auc"}, None, "mlogloss"),
    ({"objective": "binary:logistic", "eval_metric": "mlogloss"}, [0, 1, 1], "多分类"),
])
def test_invalid_configurations_are_rejected(params, labels, match):
    x, y = make(n=300, k=3)
    if labels is not None:
        y = np.resize(np.array(labels, np.float32), len(y))
    with pytest.raises(Exception, match=match):
        fb.train(params, x, label=y, num_boost_round=1, quiet=True)


def test_predict_array_points_to_predict():
    x, y = make(n=300, k=3)
    m = fb.train({"objective": "multi:softprob", "num_class": 3}, x, label=y,
                 num_boost_round=1, quiet=True)
    with pytest.raises(ValueError, match="predict"):
        m.predict_array(x)


# ---------------------------------------------------------------- GPU
def _has_cuda():
    try:
        x, y = make(64, 3)
        fb.train({"objective": "multi:softprob", "num_class": 3, "device": "cuda", "max_depth": 2},
                 x, y, num_boost_round=1, quiet=True)
        return True
    except Exception as e:  # noqa: BLE001
        if "CUDA" in str(e) or "cuda" in str(e):
            return False
        raise


cuda = pytest.mark.skipif(not _has_cuda(), reason="没有可用的 CUDA 构建或设备")


def model_bytes(model, tmp_path, name):
    path = tmp_path / f"{name}.json"
    model.save_model(str(path), quiet=True)
    return path.read_bytes()


EXTRAS = [{}, {"subsample": 0.8, "colsample_bytree": 0.7},
          {"monotone_constraints": (1, 0, 0, -1, 0)}]


@cuda
@pytest.mark.parametrize("gpu_math", ["exact", "fast"])
@pytest.mark.parametrize("extra", EXTRAS)
def test_gpu_matches_cpu(tmp_path, capfd, gpu_math, extra):
    x, y = make(n=4000, k=5, seed=13)
    w = np.random.default_rng(13).uniform(0.2, 3, len(y)).astype(np.float32)
    p = {"objective": "multi:softprob", "num_class": 5, "max_depth": 6, "seed": 4, **extra}
    cpu = fb.train(p, x, label=y, weight=w, num_boost_round=5, quiet=True)
    gpu_p = {**p, "device": "cuda", "gpu_math": gpu_math}
    gpu = fb.train(gpu_p, x, label=y, weight=w, num_boost_round=5, quiet=gpu_math == "exact")
    err = capfd.readouterr().err
    if gpu_math == "exact":
        # exact:host 梯度,CPU/GPU 逐字节相同。
        assert model_bytes(cpu, tmp_path, "cpu") == model_bytes(gpu, tmp_path, "gpu")
        return
    # fast:不再降级成 exact,回显明确是 fast。
    assert "effective=exact" not in err and "GPU_MATH" not in err
    assert "gpu_math=fast" in err
    # 每次运行确定,但不保证和 CPU 逐字节相同(CUDA expf)。
    again = fb.train(gpu_p, x, label=y, weight=w, num_boost_round=5, quiet=True)
    assert model_bytes(gpu, tmp_path, "a") == model_bytes(again, tmp_path, "b")
    assert gpu.num_trees == cpu.num_trees == 25
    np.testing.assert_allclose(gpu.predict(x), cpu.predict(x), rtol=0, atol=1e-4)


@cuda
@pytest.mark.parametrize("k", [2, 7])
@pytest.mark.parametrize("objective", ["multi:softprob", "multi:softmax"])
@pytest.mark.parametrize("weighted", [False, True])
def test_gpu_fast_class_counts(tmp_path, k, objective, weighted):
    x, y = make(n=3000, k=k, seed=20 + k)
    w = (np.random.default_rng(k).uniform(0.2, 3, len(y)).astype(np.float32)
         if weighted else None)
    p = {"objective": objective, "num_class": k, "max_depth": 5, "seed": 2,
         "subsample": 0.8, "colsample_bytree": 0.8}
    cpu = fb.train(p, x, label=y, weight=w, num_boost_round=4, quiet=True)
    fast = {**p, "device": "cuda", "gpu_math": "fast"}
    a = fb.train(fast, x, label=y, weight=w, num_boost_round=4, quiet=True)
    b = fb.train(fast, x, label=y, weight=w, num_boost_round=4, quiet=True)
    assert model_bytes(a, tmp_path, "a") == model_bytes(b, tmp_path, "b")
    assert a.num_trees == 4 * k
    if objective == "multi:softprob":
        np.testing.assert_allclose(a.predict(x), cpu.predict(x), rtol=0, atol=1e-4)
    else:
        # softmax 预测是类别下标:允许极少数近平局的行翻转。
        assert np.mean(a.predict(x) == cpu.predict(x)) > 0.995


@cuda
def test_gpu_fast_early_stopping_with_evals():
    x, y = make(n=4000, k=5, seed=31)
    vx, vy = make(n=1500, k=5, seed=32)
    p = {"objective": "multi:softprob", "num_class": 5, "max_depth": 5, "eta": 0.6,
         "eval_metric": ["mlogloss", "merror"]}
    kw = dict(evals=[(vx, vy, "v")], num_boost_round=200, early_stopping_rounds=4, quiet=True)
    cpu_res, a_res, b_res = {}, {}, {}
    cpu = fb.train(p, x, label=y, evals_result=cpu_res, **kw)
    a = fb.train({**p, "device": "cuda", "gpu_math": "fast"}, x, label=y, evals_result=a_res, **kw)
    b = fb.train({**p, "device": "cuda", "gpu_math": "fast"}, x, label=y, evals_result=b_res, **kw)
    assert a_res == b_res and a.best_iteration == b.best_iteration
    assert abs(a.best_iteration - cpu.best_iteration) <= 2
    n = min(len(a_res["v"]["mlogloss"]), len(cpu_res["v"]["mlogloss"]))
    np.testing.assert_allclose(a_res["v"]["mlogloss"][:n], cpu_res["v"]["mlogloss"][:n],
                               rtol=1e-3)


@cuda
def test_gpu_fast_echo_reports_multiclass_allocation(capfd):
    n, k = 3000, 5
    x, y = make(n=n, k=k, seed=41)
    w = np.random.default_rng(1).uniform(0.2, 3, n).astype(np.float32)
    fast = {"objective": "multi:softprob", "num_class": k, "device": "cuda",
            "gpu_math": "fast", "max_depth": 4}
    fb.train(fast, x, label=y, weight=w, num_boost_round=1)
    err = capfd.readouterr().err
    line = next(ln for ln in err.splitlines() if ln.startswith("GPU_PATH"))
    assert "gpu_math=fast" in line and f"multiclass_outputs={k}" in line
    # (K-1) 份额外 prediction + 每行 softmax 的 wmax / wsum(各 4 B)。
    assert f"multiclass_bytes={n * 4 * (k - 1) + n * 8}" in line
    # exact 模式不分配这块,回显里也不应出现。
    fb.train({**fast, "gpu_math": "exact"}, x, label=y, weight=w, num_boost_round=1)
    line = next(ln for ln in capfd.readouterr().err.splitlines() if ln.startswith("GPU_PATH"))
    assert "gpu_math=exact" in line and "multiclass_bytes" not in line


@cuda
def test_gpu_prediction_matches_cpu():
    x, y = make(n=4000, k=4, seed=14)
    m = fb.train({"objective": "multi:softprob", "num_class": 4}, x, label=y,
                 num_boost_round=5, quiet=True)
    np.testing.assert_allclose(m.predict(x, use_gpu=True), m.predict(x), rtol=0, atol=1e-6)


def test_multiclass_parquet_output_follows_single_output_rules(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    x, y = make(n=400, k=3, seed=21)
    src = str(tmp_path / "x.parquet")
    pq.write_table(pa.table({f"f{i}": x[:, i] for i in range(x.shape[1])}), src)
    m = fb.train({"objective": "multi:softprob", "num_class": 3}, x, label=y,
                 num_boost_round=2, quiet=True)
    with pytest.raises(ValueError, match="扩展名"):
        m.predict_to_parquet(src, str(tmp_path / "out.csv"))
    with pytest.raises(ValueError, match="prediction_column"):
        m.predict_to_parquet(src, str(tmp_path / "a.parquet"), prediction_column=" ")
    existing = tmp_path / "exists.parquet"
    existing.write_bytes(b"keep")
    with pytest.raises(FileExistsError):
        m.predict_to_parquet(src, str(existing))
    assert existing.read_bytes() == b"keep"
    # 预测失败(特征对不上)时不留下半成品
    pq.write_table(pa.table({"zz": x[:, 0]}), str(tmp_path / "bad.parquet"))
    with pytest.raises(Exception):
        m.predict_to_parquet(str(tmp_path / "bad.parquet"), str(tmp_path / "partial.parquet"))
    assert not (tmp_path / "partial.parquet").exists()
