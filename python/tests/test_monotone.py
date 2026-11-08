"""`monotone_constraints`:XGBoost 同名参数的写法、语义,以及和 XGBoost 的对齐。"""
import numpy as np
import pytest

import ferrisboost as fb


def wiggly(n=800, m=4, seed=0, missing=0.08):
    # 非单调的真实关系:不加约束时模型会跟着起伏,约束才有东西可管。
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2, 2, size=(n, m)).astype(np.float32)
    signal = np.sin(1.7 * x[:, 0]) + 0.8 * np.cos(3 * x[:, 1]) - 0.5 * x[:, 2]
    y = (signal + rng.normal(0, 0.3, n) > 0).astype(np.float32)
    x[rng.random(x.shape) < missing] = np.nan
    return x, y


BASE = {"objective": "binary:logistic", "max_depth": 5, "eta": 0.3, "nthread": 1}


def model_bytes(model, tmp_path, name):
    path = tmp_path / f"{name}.json"
    model.save_model(str(path), quiet=True)
    return path.read_bytes()


def sweep_is_monotone(model, x, constraints):
    grid = np.linspace(-2.2, 2.2, 61, dtype=np.float32)
    for base in x[::23][:30]:
        for f, c in enumerate(constraints):
            if c == 0:
                continue
            rows = np.repeat(base[None, :], len(grid), axis=0)
            rows[:, f] = grid
            d = np.diff(model.predict(rows, output_margin=True))
            if (c > 0 and (d < 0).any()) or (c < 0 and (d > 0).any()):
                return False
    return True


# ---------------------------------------------------------------- 语义
def test_constrained_predictions_are_monotone():
    x, y = wiggly()
    mc = (1, -1, 0, 0)
    free = fb.train(BASE, x, label=y, num_boost_round=20, quiet=True)
    tied = fb.train({**BASE, "monotone_constraints": mc}, x, label=y,
                    num_boost_round=20, quiet=True)
    assert not sweep_is_monotone(free, x, mc), "夹具不加约束时必须违反单调性"
    assert sweep_is_monotone(tied, x, mc)


def test_unset_and_all_zero_are_byte_identical(tmp_path):
    x, y = wiggly()
    a = fb.train(BASE, x, label=y, num_boost_round=4, quiet=True)
    for zeros in [(0, 0, 0, 0), (0,), "(0,0)", []]:
        b = fb.train({**BASE, "monotone_constraints": zeros}, x, label=y,
                     num_boost_round=4, quiet=True)
        assert model_bytes(a, tmp_path, "a") == model_bytes(b, tmp_path, "b"), zeros


@pytest.mark.parametrize("form", [
    (1, -1, 0, 0), [1, -1, 0, 0], "(1,-1,0,0)", "(1, -1, 0, 0)", np.array([1, -1, 0, 0]),
    (1, -1), (5, -2, 0, 0),
])
def test_accepted_forms_are_equivalent(tmp_path, form):
    # 元组 / 列表 / 字符串 / numpy、比特征数短(补 0)、只看符号(5 == 1)。
    x, y = wiggly()
    ref = fb.train({**BASE, "monotone_constraints": (1, -1, 0, 0)}, x, label=y,
                   num_boost_round=4, quiet=True)
    got = fb.train({**BASE, "monotone_constraints": form}, x, label=y,
                   num_boost_round=4, quiet=True)
    assert model_bytes(ref, tmp_path, "ref") == model_bytes(got, tmp_path, "got")


def test_works_with_weights_and_regression():
    x, y = wiggly()
    rng = np.random.default_rng(1)
    w = rng.uniform(0.2, 3, len(y)).astype(np.float32)
    target = np.nan_to_num(x[:, 0]) * 0.5 + np.sin(3 * np.nan_to_num(x[:, 1]))
    mc = (1, 1, 0, 0)
    m = fb.train({**BASE, "objective": "reg:squarederror", "monotone_constraints": mc},
                 x, label=target, weight=w, num_boost_round=15, quiet=True)
    assert sweep_is_monotone(m, x, mc)


# ---------------------------------------------------------------- 错误
@pytest.mark.parametrize("bad, match", [
    ((1, 0, 0, 0, 1), "only 4 features"),
    ((1.5, 0), "integer"),
    ((True, 0), "integer"),
    ("1,0,-1", "string must look like"),
    ("(1,x)", "non-integer"),
    (3.0, "tuple"),
])
def test_invalid_constraints_are_rejected(bad, match):
    x, y = wiggly()
    with pytest.raises(ValueError, match=match):
        fb.train({**BASE, "monotone_constraints": bad}, x, label=y,
                 num_boost_round=1, quiet=True)


def test_named_constraints_need_feature_names():
    x, y = wiggly()
    with pytest.raises(ValueError, match="column names"):
        fb.train({**BASE, "monotone_constraints": {"f0": 1}}, x, label=y,
                 num_boost_round=1, quiet=True)


# ---------------------------------------------------------------- 按列名 / 文件
def write_parquet(path, x, y):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    cols = {f"f{i}": x[:, i] for i in range(x.shape[1])}
    cols["label"] = y
    pq.write_table(pa.table(cols), str(path))


def write_csv(path, x, y):
    with open(path, "w") as f:
        f.write(",".join([f"f{i}" for i in range(x.shape[1])] + ["label"]) + "\n")
        for row, label in zip(x, y):
            f.write(",".join("" if np.isnan(v) else repr(float(v)) for v in row)
                    + f",{int(label)}\n")


@pytest.mark.parametrize("fmt", ["parquet", "csv", "parquet-multi"])
def test_named_constraints_on_files_match_positional_arrays(tmp_path, fmt):
    x, y = wiggly()
    ref = fb.train({**BASE, "monotone_constraints": (0, 1, 0, -1)}, x, label=y,
                   num_boost_round=4, quiet=True)
    if fmt == "csv":
        write_csv(tmp_path / "t.csv", x, y)
        spec = str(tmp_path / "t.csv")
    elif fmt == "parquet":
        write_parquet(tmp_path / "t.parquet", x, y)
        spec = str(tmp_path / "t.parquet")
    else:
        half = len(y) // 2
        write_parquet(tmp_path / "a.parquet", x[:half], y[:half])
        write_parquet(tmp_path / "b.parquet", x[half:], y[half:])
        spec = [str(tmp_path / "a.parquet"), str(tmp_path / "b.parquet")]
    got = fb.train({**BASE, "monotone_constraints": {"f3": -1, "f1": 1}}, spec,
                   label="label", num_boost_round=4, quiet=True)
    np.testing.assert_array_equal(got.predict(x), ref.predict(x))


def test_named_constraints_on_pyarrow_table():
    pa = pytest.importorskip("pyarrow")
    x, y = wiggly()
    table = pa.table({f"f{i}": x[:, i] for i in range(x.shape[1])})
    ref = fb.train({**BASE, "monotone_constraints": (1, 0, 0, 0)}, x, label=y,
                   num_boost_round=4, quiet=True)
    got = fb.train({**BASE, "monotone_constraints": {"f0": 1}}, table, label=y,
                   num_boost_round=4, quiet=True)
    np.testing.assert_array_equal(got.predict(x), ref.predict(x))


def test_unknown_named_column_is_rejected(tmp_path):
    x, y = wiggly()
    write_parquet(tmp_path / "t.parquet", x, y)
    with pytest.raises(ValueError, match="is not a feature column"):
        fb.train({**BASE, "monotone_constraints": {"nope": 1}}, str(tmp_path / "t.parquet"),
                 label="label", num_boost_round=1, quiet=True)
    with pytest.raises(ValueError, match="is not a feature column"):
        # 标签列不是特征
        fb.train({**BASE, "monotone_constraints": {"label": 1}}, str(tmp_path / "t.parquet"),
                 label="label", num_boost_round=1, quiet=True)


# ---------------------------------------------------------------- XGBoost
def low_cardinality(n=600, m=6, seed=0, missing=0.12):
    # 每列 8 个取值:分箱无损,两边比的纯粹是约束语义。
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 8, size=(n, m)).astype(np.float32)
    x[rng.random(x.shape) < missing] = np.nan
    s = np.nan_to_num(x[:, 0], nan=4.0) - np.nan_to_num(x[:, 1], nan=4.0)
    y = (s + rng.normal(0, 1.5, n) > 0).astype(np.float32)
    return x, y


@pytest.mark.parametrize("objective", ["binary:logistic", "reg:squarederror"])
@pytest.mark.parametrize("weighted", [False, True])
def test_bit_identical_to_xgboost(objective, weighted):
    xgb = pytest.importorskip("xgboost")
    x, y = low_cardinality(seed=3)
    w = np.random.default_rng(3).uniform(0.2, 3, len(y)).astype(np.float32) if weighted else None
    mc = (1, -1, 0, 1, 0, -1)
    p = {"max_depth": 5, "eta": 0.3, "objective": objective, "base_score": 0.5,
         "reg_lambda": 1.0, "min_child_weight": 1.0, "max_bin": 255}
    bst = xgb.train({**p, "tree_method": "hist", "monotone_constraints": "(1,-1,0,1,0,-1)"},
                    xgb.DMatrix(x, label=y, weight=w), num_boost_round=10)
    ref = bst.predict(xgb.DMatrix(x), output_margin=True)
    ours = fb.train({**p, "nthread": 1, "monotone_constraints": mc}, x, label=y, weight=w,
                    num_boost_round=10, quiet=True).predict(x, output_margin=True)
    np.testing.assert_array_equal(ours, ref)


def test_xgboost_loads_constrained_model_and_agrees(tmp_path):
    xgb = pytest.importorskip("xgboost")
    x, y = wiggly()
    m = fb.train({**BASE, "monotone_constraints": (1, -1, 0, 0)}, x, label=y,
                 num_boost_round=8, quiet=True)
    path = tmp_path / "m.json"
    m.save_model(str(path), quiet=True)
    bst = xgb.Booster(model_file=str(path))
    np.testing.assert_allclose(bst.predict(xgb.DMatrix(x), output_margin=True),
                               m.predict(x, output_margin=True), rtol=0, atol=1e-6)


# ---------------------------------------------------------------- GPU
def _has_cuda():
    try:
        x, y = wiggly(64, 4)
        fb.train({"device": "cuda", "max_depth": 2}, x, y, num_boost_round=1, quiet=True)
        return True
    except Exception as e:  # noqa: BLE001
        if "CUDA" in str(e) or "cuda" in str(e):
            return False
        raise


cuda = pytest.mark.skipif(not _has_cuda(), reason="没有可用的 CUDA 构建或设备")


@cuda
@pytest.mark.parametrize("extra", [{}, {"subsample": 0.8, "colsample_bytree": 0.7},
                                   {"hist_nodes_per_batch": 1}])
def test_gpu_constrained_model_matches_cpu(tmp_path, extra):
    x, y = wiggly(3000, 6)
    p = {**BASE, **extra, "monotone_constraints": (1, -1, 0, 1, 0, -1), "seed": 3}
    cpu = fb.train(p, x, label=y, num_boost_round=6, quiet=True)
    gpu = fb.train({**p, "device": "cuda", "gpu_math": "exact"}, x, label=y,
                   num_boost_round=6, quiet=True)
    assert model_bytes(cpu, tmp_path, "cpu") == model_bytes(gpu, tmp_path, "gpu")


@cuda
def test_gpu_fast_constrained_is_monotone_and_deterministic(tmp_path):
    x, y = wiggly(3000, 4)
    mc = (1, -1, 0, 0)
    p = {**BASE, "device": "cuda", "gpu_math": "fast", "monotone_constraints": mc}
    a = fb.train(p, x, label=y, num_boost_round=10, quiet=True)
    b = fb.train(p, x, label=y, num_boost_round=10, quiet=True)
    assert model_bytes(a, tmp_path, "a") == model_bytes(b, tmp_path, "b")
    assert sweep_is_monotone(a, x, mc)
