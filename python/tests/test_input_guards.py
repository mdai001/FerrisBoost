"""输入防护的回归测试:验证集按列名绑定、回调异常不丢、±inf / 极端值 / 空验证集等。"""
import numpy as np
import pytest

import ferrisboost as fb

BASE = {"objective": "binary:logistic", "max_depth": 3, "eta": 0.3, "nthread": 1}


def data(n=600, m=4, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, m)).astype(np.float32)
    y = (x[:, 0] + 0.5 * x[:, 1] + rng.normal(0, 0.3, n) > 0).astype(np.float32)
    return x, y


def write_parquet(path, x, y, order=None, rename=None):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    names = [f"f{i}" for i in range(x.shape[1])]
    cols = {n: x[:, i] for i, n in enumerate(names)}
    cols["label"] = y
    keys = order or list(cols)
    # from_arrays,不用 dict:重命名成重复列名时 dict 会把列吞掉。
    table = pa.Table.from_arrays([pa.array(cols[k]) for k in keys],
                                 names=[(rename or {}).get(k, k) for k in keys])
    pq.write_table(table, str(path))


def write_csv(path, x, y, order=None, rename=None):
    names = [f"f{i}" for i in range(x.shape[1])] + ["label"]
    cols = {n: (x[:, i] if i < x.shape[1] else y) for i, n in enumerate(names)}
    keys = order or names
    with open(path, "w") as f:
        f.write(",".join((rename or {}).get(k, k) for k in keys) + "\n")
        for r in range(len(y)):
            f.write(",".join(repr(float(cols[k][r])) for k in keys) + "\n")


WRITERS = {"parquet": write_parquet, "csv": write_csv}


# ------------------------------------------------- 验证集按列名绑定
@pytest.mark.parametrize("fmt", ["parquet", "csv"])
@pytest.mark.parametrize("multi", [False, True])
def test_eval_file_binds_columns_by_name(tmp_path, fmt, multi):
    x, y = data()
    ex, ey = data(300, seed=1)
    write = WRITERS[fmt]
    if multi:
        half = len(y) // 2
        write(tmp_path / f"a.{fmt}", x[:half], y[:half])
        write(tmp_path / f"b.{fmt}", x[half:], y[half:])
        train = [str(tmp_path / f"a.{fmt}"), str(tmp_path / f"b.{fmt}")]
    else:
        write(tmp_path / f"t.{fmt}", x, y)
        train = str(tmp_path / f"t.{fmt}")
    write(tmp_path / f"same.{fmt}", ex, ey)
    write(tmp_path / f"shuffled.{fmt}", ex, ey, order=["label", "f3", "f1", "f0", "f2"])

    results = {}
    for name in ["same", "shuffled"]:
        res = {}
        fb.train({**BASE, "eval_metric": "logloss"}, train, label="label",
                 num_boost_round=5, evals=[(str(tmp_path / f"{name}.{fmt}"), "v")],
                 evals_result=res, quiet=True)
        results[name] = res["v"]["logloss"]
    assert results["same"] == results["shuffled"]


@pytest.mark.parametrize("fmt", ["parquet", "csv"])
def test_eval_file_with_different_columns_is_rejected(tmp_path, fmt):
    x, y = data()
    write = WRITERS[fmt]
    write(tmp_path / f"t.{fmt}", x, y)
    write(tmp_path / f"renamed.{fmt}", x, y, rename={"f2": "zz"})
    with pytest.raises(Exception, match="对不上"):
        fb.train(BASE, str(tmp_path / f"t.{fmt}"), label="label", num_boost_round=2,
                 evals=[(str(tmp_path / f"renamed.{fmt}"), "v")], quiet=True)


@pytest.mark.parametrize("fmt", ["parquet", "csv"])
def test_duplicate_column_names_are_rejected(tmp_path, fmt):
    x, y = data()
    write = WRITERS[fmt]
    write(tmp_path / f"dup.{fmt}", x, y, rename={"f2": "f1"})
    with pytest.raises(Exception, match="重复"):
        fb.train(BASE, str(tmp_path / f"dup.{fmt}"), label="label",
                 num_boost_round=1, quiet=True)


def test_pyarrow_eval_binds_columns_by_name():
    pa = pytest.importorskip("pyarrow")
    x, y = data()
    ex, ey = data(300, seed=1)
    names = [f"f{i}" for i in range(4)]
    train = pa.table({n: x[:, i] for i, n in enumerate(names)})
    same = pa.table({n: ex[:, i] for i, n in enumerate(names)})
    shuffled = same.select(["f2", "f0", "f3", "f1"])
    out = {}
    for tag, ev in [("same", same), ("shuffled", shuffled)]:
        res = {}
        fb.train(BASE, train, label=y, num_boost_round=4, evals=[(ev, ey, "v")],
                 evals_result=res, quiet=True)
        out[tag] = res["v"]
    assert out["same"] == out["shuffled"]
    with pytest.raises(ValueError, match="对不上"):
        fb.train(BASE, train, label=y, num_boost_round=1,
                 evals=[(same.rename_columns(["f0", "f1", "f2", "zz"]), ey, "v")], quiet=True)


def test_named_constraints_on_record_batch_reader():
    pa = pytest.importorskip("pyarrow")
    x, y = data()
    table = pa.table({f"f{i}": x[:, i] for i in range(4)})
    ref = fb.train({**BASE, "monotone_constraints": (1, 0, 0, 0)}, x, label=y,
                   num_boost_round=4, quiet=True)
    reader = pa.RecordBatchReader.from_batches(table.schema, table.to_batches())
    got = fb.train({**BASE, "monotone_constraints": {"f0": 1}}, reader, label=y,
                   num_boost_round=4, quiet=True)
    np.testing.assert_array_equal(got.predict(x), ref.predict(x))


# ------------------------------------------------- 回调
def test_callback_exception_survives_later_callbacks():
    x, y = data()
    calls = []

    def bad(i, m):
        if i == 2:
            raise RuntimeError("boom from callback")

    def ok(i, m):
        calls.append(i)

    with pytest.raises(RuntimeError, match="boom from callback"):
        fb.train(BASE, x, label=y, num_boost_round=10, evals=[(x, y, "t")],
                 callbacks=[bad, ok], quiet=True)
    assert max(calls) <= 2, "抛异常之后不该再调用后面的回调"


# ------------------------------------------------- 数值输入
@pytest.mark.parametrize("bad", [np.inf, -np.inf])
def test_inf_features_are_rejected_cleanly(tmp_path, bad):
    x, y = data()
    x[5, 1] = bad
    with pytest.raises(ValueError, match="inf"):
        fb.train(BASE, x, label=y, num_boost_round=1, quiet=True)
    good, _ = data()
    with pytest.raises(ValueError, match="inf"):
        fb.train(BASE, good, label=y, num_boost_round=1, evals=[(x, y, "v")], quiet=True)
    write_parquet(tmp_path / "inf.parquet", x, y)
    with pytest.raises(Exception, match="inf"):
        fb.train(BASE, str(tmp_path / "inf.parquet"), label="label",
                 num_boost_round=1, quiet=True)


def test_wide_range_finite_values_train_like_xgboost():
    # 低基数:分箱无损,和 XGBoost 逐位比。以前 -2e38/2e38 的中点溢出成 inf 而 panic。
    xgb = pytest.importorskip("xgboost")
    x, y = data(400, 3)
    x = np.round(x * 2).astype(np.float32)
    x[:, 0] = np.where(x[:, 0] > 0, 2e38, -2e38).astype(np.float32)
    below_max = np.nextafter(np.float32(np.finfo(np.float32).max), np.float32(0.0))
    x[:4, 2] = [-3.0e38, 3.0e38, below_max, -np.finfo(np.float32).max]
    p = {**BASE, "base_score": 0.5, "max_bin": 255}
    ours = fb.train(p, x, label=y, num_boost_round=5, quiet=True).predict(x, output_margin=True)
    bst = xgb.train({**p, "tree_method": "hist"}, xgb.DMatrix(x, label=y), num_boost_round=5)
    assert np.isfinite(ours).all()
    np.testing.assert_array_equal(ours, bst.predict(xgb.DMatrix(x), output_margin=True))


def test_exact_float32_max_feature_is_rejected_for_numpy_and_arrow():
    x = np.array([[np.finfo(np.float32).max], [1.0]], dtype=np.float32)
    y = np.array([1.0, 0.0], dtype=np.float32)
    with pytest.raises(ValueError, match="f32::MAX"):
        fb.train(BASE, x, label=y, num_boost_round=1, quiet=True)

    pa = pytest.importorskip("pyarrow")
    table = pa.table({"f0": x[:, 0]})
    with pytest.raises(Exception, match="f32::MAX"):
        fb.train(BASE, table, label=y, num_boost_round=1, quiet=True)


@pytest.mark.parametrize("fmt", ["csv", "parquet"])
def test_exact_float32_max_feature_is_rejected_from_training_file(tmp_path, fmt):
    values = np.array([[np.finfo(np.float32).max], [1.0]], dtype=np.float32)
    labels = np.array([1.0, 0.0], dtype=np.float32)
    path = tmp_path / f"max.{fmt}"
    WRITERS[fmt](path, values, labels)

    with pytest.raises(Exception, match="f32::MAX"):
        fb.train(BASE, str(path), label="label", num_boost_round=1, quiet=True)


@pytest.mark.parametrize("fmt", ["csv", "parquet"])
def test_exact_float32_max_feature_is_rejected_from_eval_file_with_training_cuts(
    tmp_path, fmt
):
    safe = np.array([[1.0], [2.0]], dtype=np.float32)
    bad = np.array([[np.finfo(np.float32).max], [3.0]], dtype=np.float32)
    labels = np.array([1.0, 0.0], dtype=np.float32)
    train_path = tmp_path / f"train.{fmt}"
    eval_path = tmp_path / f"eval.{fmt}"
    WRITERS[fmt](train_path, safe, labels)
    WRITERS[fmt](eval_path, bad, labels)

    with pytest.raises(Exception, match="f32::MAX"):
        fb.train(
            BASE,
            str(train_path),
            label="label",
            num_boost_round=1,
            evals=[(str(eval_path), "validation")],
            quiet=True,
        )


def test_exact_float32_max_feature_is_rejected_from_numpy_eval():
    train_x = np.array([[1.0], [2.0]], dtype=np.float32)
    eval_x = np.array([[np.finfo(np.float32).max], [3.0]], dtype=np.float32)
    labels = np.array([1.0, 0.0], dtype=np.float32)

    with pytest.raises(ValueError, match="f32::MAX"):
        fb.train(
            BASE,
            train_x,
            label=labels,
            num_boost_round=1,
            evals=[(eval_x, labels, "validation")],
            quiet=True,
        )


def test_empty_eval_set_is_rejected():
    x, y = data()
    with pytest.raises(Exception, match="空"):
        fb.train(BASE, x, label=y, num_boost_round=1,
                 evals=[(x[:0], y[:0], "v")], quiet=True)


def test_all_zero_weights_with_zero_lambda_give_no_nan():
    x, y = data()
    m = fb.train({**BASE, "reg_lambda": 0.0, "min_child_weight": 0.0}, x, label=y,
                 weight=np.zeros(len(y), np.float32), num_boost_round=3, quiet=True)
    assert np.isfinite(m.predict(x, output_margin=True)).all()


def test_weight_sum_overflowing_f32_is_rejected():
    x, y = data()
    w = np.full(len(y), 1e36, np.float32)
    with pytest.raises(Exception, match="权重总和"):
        fb.train(BASE, x, label=y, weight=w, num_boost_round=1, quiet=True)
