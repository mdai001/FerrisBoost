"""`missing=`: XGBoost's DMatrix `missing` sentinel, as a real core parameter.

A feature value equal to `missing` must behave exactly like NaN (missing
direction) in the training data, every validation set and prediction, for every
input kind. Two kinds of evidence:

* sentinel data with `missing=S` is bit-for-bit the same as the same data with
  the sentinel replaced by NaN (trees, predictions, eval history) -- for numpy,
  pandas, pyarrow, Parquet and CSV, so the sketch / quantizer never turn a
  sentinel into a bin;
* where binning is lossless (small integer features) the trees and predictions
  are bit-identical to XGBoost trained with `DMatrix(x, missing=S)`.
"""
import json

import numpy as np
import pytest

import ferrisboost as fb

xgb = pytest.importorskip("xgboost")

SENTINELS = [-999.0, 0.0]


def make_ints(n=700, m=4, seed=0, sentinel=-999.0, frac=0.15, objective="reg:squarederror"):
    """Small-integer features (lossless binning). Returns (x_sentinel, x_nan, y).

    For sentinel 0 the data already contains zeros as ordinary values, and those
    are the values that must become missing; for other sentinels a random subset
    of cells is overwritten with the sentinel.
    """
    rng = np.random.default_rng(seed)
    x = rng.integers(1, 9, size=(n, m)).astype(np.float32)
    signal = 0.4 * (x[:, 0] - 4) + 0.3 * (x[:, 1] % 3) - 0.2 * x[:, 2]
    hole = rng.random(x.shape) < frac
    x_sent = x.copy()
    x_sent[hole] = sentinel
    x_nan = x.copy()
    x_nan[hole] = np.nan
    if sentinel == 0.0:
        # Zero is also a legitimate-looking value: make some real zeros too.
        extra = rng.random(x.shape) < 0.1
        x_sent[extra] = 0.0
        x_nan[extra] = np.nan
    if objective == "binary:logistic":
        y = (signal + rng.normal(0, 0.7, n) > 0).astype(np.float32)
    elif objective.startswith("multi:"):
        y = np.digitize(signal + rng.normal(0, 0.4, n), [-0.7, 0.7]).astype(np.float32)
    else:
        y = (signal + rng.normal(0, 0.3, n)).astype(np.float32)
    return x_sent, x_nan, y


def make_floats(n=900, m=5, seed=1, sentinel=-999.0, frac=0.2):
    """Continuous features (sketch-binned). Returns (x_sentinel, x_nan, y)."""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, m)).astype(np.float32)
    y = (x[:, 0] + 0.5 * x[:, 1] - 0.3 * x[:, 2] + rng.normal(0, 0.3, n) > 0).astype(np.float32)
    hole = rng.random(x.shape) < frac
    xs, xn = x.copy(), x.copy()
    xs[hole] = sentinel
    xn[hole] = np.nan
    return xs, xn, y


BASE = {"objective": "reg:squarederror", "max_depth": 4, "nthread": 1, "base_score": 0.5}


def fb_fit(x, y, params=None, rounds=6, **kw):
    return fb.train({**BASE, **(params or {})}, x, label=y, num_boost_round=rounds, quiet=True, **kw)


def model_json(model, tmp_path, name):
    path = str(tmp_path / f"{name}.json")
    model.save_model(path, quiet=True)
    return json.load(open(path))


def trees(model, tmp_path, name):
    return model_json(model, tmp_path, name)["learner"]["gradient_booster"]["model"]["trees"]


def tree_key(tree):
    return {k: tree[k] for k in ("left_children", "right_children", "split_indices",
                                 "split_conditions", "default_left", "base_weights")}


def struct_key(tree):
    # FerrisBoost writes no base_weights and stores split thresholds as cut midpoints,
    # so against XGBoost compare the structure; leaf values are covered by bit-exact
    # predictions.
    return {k: tree[k] for k in ("left_children", "right_children", "split_indices",
                                 "default_left")}


def assert_same_model(a, b, tmp_path):
    ta, tb = trees(a, tmp_path, "a"), trees(b, tmp_path, "b")
    assert len(ta) == len(tb)
    for i, (u, v) in enumerate(zip(ta, tb)):
        assert tree_key(u) == tree_key(v), f"tree {i} differs"


# ------------------------------------------------------------ bit-exact vs XGBoost
@pytest.mark.parametrize("sentinel", SENTINELS)
@pytest.mark.parametrize("objective", ["reg:squarederror", "binary:logistic"])
def test_bit_exact_vs_xgboost(sentinel, objective, tmp_path):
    xs, xn, y = make_ints(sentinel=sentinel, objective=objective)
    params = {**BASE, "objective": objective}
    d = xgb.DMatrix(xs, label=y, missing=sentinel)
    bst = xgb.train({"tree_method": "hist", "max_bin": 255, **params}, d, num_boost_round=6)
    ours = fb.train({**params, "missing": sentinel}, xs, label=y, num_boost_round=6, quiet=True)
    ref = json.loads(bst.save_raw("json"))["learner"]["gradient_booster"]["model"]["trees"]
    mine = trees(ours, tmp_path, "m")
    assert len(mine) == len(ref)
    for i, (u, v) in enumerate(zip(mine, ref)):
        assert struct_key(u) == struct_key(v), f"tree {i} differs"
    np.testing.assert_array_equal(
        ours.predict(xs, output_margin=True, missing=sentinel),
        bst.predict(xgb.DMatrix(xs, missing=sentinel), output_margin=True),
    )
    # The sentinel is not NaN-equivalent unless told: without `missing` the same
    # matrix is a different dataset.
    assert not np.array_equal(
        fb.train(params, xs, label=y, num_boost_round=6, quiet=True).predict(xs, output_margin=True),
        ours.predict(xs, output_margin=True, missing=sentinel),
    )


@pytest.mark.parametrize("sentinel", SENTINELS)
def test_bit_exact_vs_xgboost_with_eval_and_weights(sentinel, tmp_path):
    xs, _, y = make_ints(sentinel=sentinel, seed=4)
    vx, _, vy = make_ints(n=300, sentinel=sentinel, seed=5)
    w = np.random.default_rng(6).uniform(0.3, 2.5, len(y)).astype(np.float32)
    d = xgb.DMatrix(xs, label=y, weight=w, missing=sentinel)
    dv = xgb.DMatrix(vx, label=vy, missing=sentinel)
    hist = {}
    bst = xgb.train({"tree_method": "hist", "max_bin": 255, **BASE, "eval_metric": "rmse"}, d,
                    num_boost_round=8, evals=[(dv, "v")], evals_result=hist, verbose_eval=False)
    mine_hist = {}
    ours = fb.train({**BASE, "missing": sentinel, "eval_metric": "rmse"}, xs, label=y, weight=w,
                    num_boost_round=8, evals=[(vx, vy, "v")], evals_result=mine_hist, quiet=True)
    np.testing.assert_array_equal(
        ours.predict(vx, output_margin=True, missing=sentinel),
        bst.predict(dv, output_margin=True),
    )
    np.testing.assert_allclose(mine_hist["v"]["rmse"], hist["v"]["rmse"], rtol=1e-6)


def test_multiclass_bit_exact_vs_xgboost():
    xs, _, y = make_ints(sentinel=-999.0, seed=7, objective="multi:softprob")
    params = {**BASE, "objective": "multi:softprob", "num_class": 3, "max_depth": 3}
    d = xgb.DMatrix(xs, label=y, missing=-999.0)
    bst = xgb.train({"tree_method": "hist", "max_bin": 255, **params}, d, num_boost_round=4)
    ours = fb.train({**params, "missing": -999.0}, xs, label=y, num_boost_round=4, quiet=True)
    np.testing.assert_array_equal(
        ours.predict(xs, output_margin=True, missing=-999.0),
        bst.predict(xgb.DMatrix(xs, missing=-999.0), output_margin=True),
    )


def test_categorical_sentinel_bit_exact_vs_xgboost(tmp_path):
    # XGBoost applies `missing` to categorical codes too: with missing=-1 the
    # code -1 is a missing value, not an invalid category.
    rng = np.random.default_rng(8)
    n = 800
    num = rng.integers(0, 8, n).astype(np.float32)
    cat = rng.integers(0, 5, n).astype(np.float32)
    y = (0.3 * (num - 4) + np.array([0.5, -1, 0.2, 1.0, -0.4])[cat.astype(int)]
         + rng.normal(0, 0.3, n)).astype(np.float32)
    hole = rng.random(n) < 0.15
    xs = np.stack([num, cat], axis=1)
    xs[hole, 1] = -1.0
    ft = ["q", "c"]
    d = xgb.DMatrix(xs, label=y, feature_types=ft, enable_categorical=True, missing=-1.0)
    bst = xgb.train({"tree_method": "hist", "max_bin": 255, **BASE}, d, num_boost_round=5)
    ours = fb.train({**BASE, "missing": -1.0}, xs, label=y, num_boost_round=5, quiet=True,
                    feature_types=ft)
    np.testing.assert_array_equal(
        ours.predict(xs, output_margin=True, missing=-1.0), bst.predict(d, output_margin=True))
    # Same codes with the sentinel replaced by NaN: same model.
    xn = xs.copy()
    xn[hole, 1] = np.nan
    plain = fb.train(BASE, xn, label=y, num_boost_round=5, quiet=True, feature_types=ft)
    assert_same_model(ours, plain, tmp_path)


def test_zero_sentinel_hits_category_zero():
    # missing=0 makes category 0 itself missing (it is a code like any other).
    rng = np.random.default_rng(9)
    n = 600
    cat = rng.integers(0, 4, n).astype(np.float32)
    y = (np.array([2.0, -1.0, 0.5, 1.5])[cat.astype(int)] + rng.normal(0, 0.2, n)).astype(np.float32)
    x = cat[:, None]
    ours = fb.train({**BASE, "missing": 0.0}, x, label=y, num_boost_round=4, quiet=True,
                    feature_types=["c"])
    xn = np.where(x == 0, np.nan, x).astype(np.float32)
    plain = fb.train(BASE, xn, label=y, num_boost_round=4, quiet=True, feature_types=["c"])
    np.testing.assert_array_equal(ours.predict(x, output_margin=True, missing=0.0),
                                  plain.predict(xn, output_margin=True))


# ------------------------------------------------------------ equivalence with NaN
@pytest.mark.parametrize("sentinel", SENTINELS + [-1.0, 12345.0])
def test_numpy_sentinel_equals_nan_including_eval_and_early_stopping(sentinel, tmp_path):
    xs, xn, y = make_floats(sentinel=sentinel)
    vs, vn, vy = make_floats(n=300, seed=2, sentinel=sentinel)
    p = {"objective": "binary:logistic", "eval_metric": ["logloss", "auc"]}
    h1, h2 = {}, {}
    a = fb_fit(xs, y, {**p, "missing": sentinel}, rounds=30, evals=[(vs, vy, "v")],
               early_stopping_rounds=4, evals_result=h1)
    b = fb_fit(xn, y, p, rounds=30, evals=[(vn, vy, "v")], early_stopping_rounds=4, evals_result=h2)
    assert h1 == h2
    assert a.best_iteration == b.best_iteration
    assert_same_model(a, b, tmp_path)
    np.testing.assert_array_equal(a.predict(vs, missing=sentinel), b.predict(vn))


def test_sentinel_never_becomes_a_cut_point(tmp_path):
    # Mostly sentinel values: if they reached the sketch, a split threshold would
    # land at or below -999. Every threshold must be a real data value.
    xs, xn, y = make_floats(sentinel=-999.0, frac=0.6)
    m = fb_fit(xs, y, {"missing": -999.0}, rounds=8)
    for t in trees(m, tmp_path, "m"):
        for left, cond in zip(t["left_children"], t["split_conditions"]):
            if left != -1:
                assert cond > -900.0
    assert_same_model(m, fb_fit(xn, y, rounds=8), tmp_path)


def test_sentinel_with_weights_and_missing_in_sketch(tmp_path):
    xs, xn, y = make_floats(sentinel=-7.5, seed=3)
    w = np.random.default_rng(3).uniform(0.2, 3, len(y)).astype(np.float32)
    a = fb_fit(xs, y, {"missing": -7.5}, weight=w)
    b = fb_fit(xn, y, weight=w)
    assert_same_model(a, b, tmp_path)


def test_nan_stays_missing_alongside_a_sentinel(tmp_path):
    xs, xn, y = make_floats(sentinel=-999.0)
    xs2, xn2 = xs.copy(), xn.copy()
    xs2[::7, 3] = np.nan
    xn2[::7, 3] = np.nan
    a = fb_fit(xs2, y, {"missing": -999.0})
    b = fb_fit(xn2, y)
    assert_same_model(a, b, tmp_path)


def test_caller_array_is_never_modified():
    xs, _, y = make_floats(sentinel=-999.0)
    before = xs.copy()
    m = fb_fit(xs, y, {"missing": -999.0})
    m.predict(xs, missing=-999.0)
    np.testing.assert_array_equal(xs, before)


def test_float64_and_int_inputs(tmp_path):
    xs, xn, y = make_ints(sentinel=-999.0)
    a = fb_fit(xs.astype(np.float64), y, {"missing": -999.0})
    b = fb_fit(xs.astype(np.int64), y, {"missing": -999.0})
    c = fb_fit(xn, y)
    assert_same_model(a, c, tmp_path)
    assert_same_model(b, c, tmp_path)


def test_pandas_frame(tmp_path):
    pd = pytest.importorskip("pandas")
    xs, xn, y = make_floats(sentinel=-999.0)
    vs, vn, vy = make_floats(n=200, seed=2, sentinel=-999.0)
    cols = [f"c{i}" for i in range(xs.shape[1])]
    a = fb_fit(pd.DataFrame(xs, columns=cols), y, {"missing": -999.0},
               evals=[(pd.DataFrame(vs, columns=cols), vy, "v")])
    b = fb_fit(pd.DataFrame(xn, columns=cols), y, evals=[(pd.DataFrame(vn, columns=cols), vy, "v")])
    assert_same_model(a, b, tmp_path)
    np.testing.assert_array_equal(
        a.predict(pd.DataFrame(vs, columns=cols), missing=-999.0), b.predict(vn))
    # nullable integer column: pd.NA stays missing, the sentinel joins it
    df = pd.DataFrame({"a": pd.array([1, -1, 3, 4, 5, 6, 7, -1], dtype="Int64"),
                       "b": np.arange(8, dtype=np.float32)})
    ya = np.arange(8, dtype=np.float32)
    mm = fb.train({**BASE, "missing": -1.0, "max_depth": 2}, df, label=ya, num_boost_round=2,
                  quiet=True)
    assert mm.n_features == 2


def test_pyarrow_table(tmp_path):
    pa = pytest.importorskip("pyarrow")
    xs, xn, y = make_floats(sentinel=-999.0)
    cols = [f"c{i}" for i in range(xs.shape[1])]
    ts = pa.table({c: xs[:, i] for i, c in enumerate(cols)})
    tn = pa.table({c: xn[:, i] for i, c in enumerate(cols)})
    a = fb_fit(ts, y, {"missing": -999.0}, evals=[(ts, y, "v")])
    b = fb_fit(tn, y, evals=[(tn, y, "v")])
    assert_same_model(a, b, tmp_path)
    np.testing.assert_array_equal(a.predict(ts, missing=-999.0), b.predict(tn))


# ------------------------------------------------------------ files
def write_parquet(path, x, y, dtype="f32", null_nan=False):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    arrays, names = [], []
    for i in range(x.shape[1]):
        col = x[:, i]
        if dtype == "f64":
            arr = pa.array(col.astype(np.float64))
        elif dtype == "i64":
            arr = pa.array(col.astype(np.int64))
        else:
            arr = pa.array(col)
        if null_nan:
            arr = pa.array(col, mask=np.isnan(col))
        arrays.append(arr)
        names.append(f"f{i}")
    if y is not None:  # a positional-mode model scores feature-only files
        arrays.append(pa.array(y))
        names.append("label")
    pq.write_table(pa.Table.from_arrays(arrays, names=names), str(path), row_group_size=250)


def write_csv(path, x, y, header=True):
    names = [f"f{i}" for i in range(x.shape[1])] + (["label"] if y is not None else [])
    with open(path, "w") as f:
        if header:
            f.write(",".join(names) + "\n")
        for r in range(len(x)):
            cells = ["" if np.isnan(v) else repr(float(v)) for v in x[r]]
            f.write(",".join(cells + ([repr(float(y[r]))] if y is not None else [])) + "\n")


@pytest.mark.parametrize("fmt", ["parquet", "csv"])
@pytest.mark.parametrize("sentinel", [-999.0, 0.0])
def test_file_train_eval_predict_match_nan_encoding(tmp_path, fmt, sentinel):
    xs, xn, y = make_ints(sentinel=sentinel, seed=11)
    vs, vn, vy = make_ints(n=300, sentinel=sentinel, seed=12)
    write = {"parquet": write_parquet, "csv": write_csv}[fmt]
    write(tmp_path / f"train_s.{fmt}", xs, y)
    write(tmp_path / f"valid_s.{fmt}", vs, vy)
    write(tmp_path / f"train_n.{fmt}", xn, y)
    write(tmp_path / f"valid_n.{fmt}", vn, vy)
    for kw in ({}, {"nthread": 3, "ingest_threads": 3}):
        h1, h2 = {}, {}
        a = fb_fit(str(tmp_path / f"train_s.{fmt}"), "label", {**kw, "missing": sentinel},
                   evals=[(str(tmp_path / f"valid_s.{fmt}"), "v")], evals_result=h1)
        b = fb_fit(str(tmp_path / f"train_n.{fmt}"), "label", kw,
                   evals=[(str(tmp_path / f"valid_n.{fmt}"), "v")], evals_result=h2)
        assert h1 == h2
        assert_same_model(a, b, tmp_path)
        # and the same as the in-memory run
        c = fb_fit(xs, y, {**kw, "missing": sentinel})
        assert_same_model(a, c, tmp_path)
        expect = c.predict(vn)
        np.testing.assert_array_equal(a.predict(vn), expect)
        # file prediction with the sentinel
        np.testing.assert_array_equal(
            a.predict(str(tmp_path / f"valid_s.{fmt}"), missing=sentinel), expect)
        # ... which differs without it
        assert not np.array_equal(a.predict(str(tmp_path / f"valid_s.{fmt}")), expect)


@pytest.mark.parametrize("dtype", ["f64", "i64"])
def test_parquet_column_types(tmp_path, dtype):
    xs, xn, y = make_ints(sentinel=-999.0, seed=13)
    write_parquet(tmp_path / "t.parquet", xs, y, dtype=dtype)
    a = fb_fit(str(tmp_path / "t.parquet"), "label", {"missing": -999.0})
    assert_same_model(a, fb_fit(xn, y), tmp_path)


def test_parquet_null_and_sentinel_together(tmp_path):
    xs, xn, y = make_floats(sentinel=-999.0)
    xs2, xn2 = xs.copy(), xn.copy()
    xs2[::5, 1] = np.nan
    xn2[::5, 1] = np.nan
    write_parquet(tmp_path / "t.parquet", xs2, y, null_nan=True)
    a = fb_fit(str(tmp_path / "t.parquet"), "label", {"missing": -999.0})
    assert_same_model(a, fb_fit(xn2, y), tmp_path)


def test_multi_file_input_and_directory(tmp_path):
    xs, xn, y = make_ints(sentinel=-999.0, seed=14)
    half = len(y) // 2
    d = tmp_path / "data"
    d.mkdir()
    write_parquet(d / "a.parquet", xs[:half], y[:half])
    write_parquet(d / "b.parquet", xs[half:], y[half:])
    a = fb_fit(str(d), "label", {"missing": -999.0})
    assert_same_model(a, fb_fit(xn, y), tmp_path)
    np.testing.assert_array_equal(a.predict(str(d), missing=-999.0), a.predict(xn))


def test_weight_column_and_label_are_not_subject_to_missing(tmp_path):
    # The sentinel is a *feature* marker: a label / weight equal to it is kept.
    xs, xn, y = make_ints(sentinel=0.0, seed=15)
    y = y.copy()
    y[:50] = 0.0
    w = np.random.default_rng(1).uniform(0.5, 2, len(y)).astype(np.float32)
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    cols = {f"f{i}": xs[:, i] for i in range(xs.shape[1])}
    cols["label"] = y
    cols["w"] = w
    pq.write_table(pa.table(cols), str(tmp_path / "t.parquet"))
    a = fb.train({**BASE, "missing": 0.0}, str(tmp_path / "t.parquet"), label="label",
                 weight="w", num_boost_round=4, quiet=True)
    b = fb.train(BASE, xn, label=y, weight=w, num_boost_round=4, quiet=True)
    assert_same_model(a, b, tmp_path)


def test_quantized_cache_is_only_reused_with_the_same_missing(tmp_path):
    xs, xn, y = make_ints(sentinel=-999.0, seed=16)
    write_parquet(tmp_path / "t.parquet", xs, y)
    cache = str(tmp_path / "cache")
    p = {"missing": -999.0, "cache_path": cache, "cache_budget_bytes": 1}
    a = fb_fit(str(tmp_path / "t.parquet"), "label", p)
    # same sentinel: reuse works and gives the same model
    b = fb_fit(str(tmp_path / "t.parquet"), "label", p)
    assert_same_model(a, b, tmp_path)
    # a different one must not silently reuse bins built for another sentinel
    with pytest.raises(Exception, match="missing"):
        fb_fit(str(tmp_path / "t.parquet"), "label", {**p, "missing": 0.0})
    with pytest.raises(Exception, match="missing"):
        fb_fit(str(tmp_path / "t.parquet"), "label", {**p, "missing": float("nan")})


# ------------------------------------------------------------ predict surfaces
@pytest.mark.parametrize("objective", ["reg:squarederror", "binary:logistic", "multi:softprob",
                                       "multi:softmax"])
def test_predict_variants_with_missing(tmp_path, objective):
    xs, xn, y = make_ints(sentinel=-999.0, seed=17, objective=objective)
    params = {"objective": objective}
    if objective.startswith("multi:"):
        params["num_class"] = 3
    m = fb_fit(xs, y, {**params, "missing": -999.0}, rounds=4)
    expect = m.predict(xn)
    np.testing.assert_array_equal(m.predict(xs, missing=-999.0), expect)
    np.testing.assert_array_equal(
        m.predict(xs, missing=-999.0, output_margin=True), m.predict(xn, output_margin=True))
    np.testing.assert_array_equal(
        m.predict(xs, missing=-999.0, iteration_range=(1, 3)), m.predict(xn, iteration_range=(1, 3)))
    np.testing.assert_array_equal(
        m.predict(xs, missing=-999.0, n_trees=3), m.predict(xn, n_trees=3))
    # non-contiguous input
    xf = np.asfortranarray(xs)
    np.testing.assert_array_equal(m.predict(xf, missing=-999.0), expect)
    # file input
    write_parquet(tmp_path / "p.parquet", xs, None)
    write_csv(tmp_path / "p.csv", xs, None, header=False)
    np.testing.assert_array_equal(m.predict(str(tmp_path / "p.parquet"), missing=-999.0), expect)
    np.testing.assert_array_equal(m.predict(str(tmp_path / "p.csv"), header=False, missing=-999.0), expect)
    # predict_to_parquet
    pq = pytest.importorskip("pyarrow.parquet")
    out = str(tmp_path / "out.parquet")
    n = m.predict_to_parquet(str(tmp_path / "p.parquet"), out, missing=-999.0)
    assert n == len(y)
    table = pq.read_table(out)
    got = np.stack([table.column(c).to_numpy() for c in table.column_names], axis=1)
    np.testing.assert_array_equal(got.reshape(expect.shape) if got.shape[1] == 1 else got, expect)


def test_explanations_honor_missing(tmp_path):
    xs, xn, y = make_ints(sentinel=-999.0, seed=18)
    m = fb_fit(xs, y, {"missing": -999.0}, rounds=3)
    for kw in ({"pred_contribs": True}, {"pred_leaf": True}, {"pred_interactions": True}):
        np.testing.assert_array_equal(m.predict(xs, missing=-999.0, **kw), m.predict(xn, **kw))
    write_parquet(tmp_path / "p.parquet", xs, None)
    np.testing.assert_array_equal(
        m.predict(str(tmp_path / "p.parquet"), missing=-999.0, pred_contribs=True),
        m.predict(xn, pred_contribs=True))


def test_predict_does_not_know_the_training_sentinel():
    # XGBoost does not store `missing` in the model either: predicting with the
    # default treats the sentinel as an ordinary (very small) number.
    xs, xn, y = make_floats(sentinel=-999.0)
    m = fb_fit(xs, y, {"missing": -999.0})
    assert not np.array_equal(m.predict(xs), m.predict(xn))
    np.testing.assert_array_equal(m.predict(xs, missing=-999.0), m.predict(xn))


def test_model_round_trip_does_not_store_missing(tmp_path):
    xs, xn, y = make_floats(sentinel=-999.0)
    m = fb_fit(xs, y, {"missing": -999.0})
    path = str(tmp_path / "m.json")
    m.save_model(path, quiet=True)
    text = open(path).read()
    assert "-999" not in text
    loaded = fb.Model.load_model(path, quiet=True)
    np.testing.assert_array_equal(loaded.predict(xs, missing=-999.0), m.predict(xn))
    bst = xgb.Booster(model_file=path)
    np.testing.assert_array_equal(
        bst.predict(xgb.DMatrix(xs, missing=-999.0)), m.predict(xs, missing=-999.0))


def test_pandas_predict_with_missing():
    pd = pytest.importorskip("pandas")
    xs, xn, y = make_floats(sentinel=-999.0)
    m = fb_fit(xs, y, {"missing": -999.0})
    df = pd.DataFrame(xs, columns=list("abcde"))
    np.testing.assert_array_equal(m.predict(df, missing=-999.0), m.predict(xn))
    # explanations through a frame too
    np.testing.assert_array_equal(m.predict(df, missing=-999.0, pred_leaf=True),
                                  m.predict(xn, pred_leaf=True))


# ------------------------------------------------------------ errors and defaults
@pytest.mark.parametrize("bad", [float("inf"), float("-inf"), 3.4028234663852886e38, 1e39, -1e39])
def test_reserved_or_non_finite_sentinels_are_rejected(bad):
    xs, _, y = make_floats()
    with pytest.raises(ValueError, match="missing"):
        fb_fit(xs, y, {"missing": bad})
    m = fb_fit(xs, y)
    with pytest.raises(ValueError, match="missing"):
        m.predict(xs, missing=bad)


@pytest.mark.parametrize("bad", [None, "nan", "abc", [1.0]])
def test_non_numeric_sentinel_is_rejected(bad):
    xs, _, y = make_floats()
    with pytest.raises((ValueError, TypeError)):
        fb_fit(xs, y, {"missing": bad})


def test_predict_to_parquet_validates_missing(tmp_path):
    xs, _, y = make_floats()
    m = fb_fit(xs, y)
    write_parquet(tmp_path / "p.parquet", xs, y)
    with pytest.raises(ValueError, match="missing"):
        m.predict_to_parquet(str(tmp_path / "p.parquet"), str(tmp_path / "o.parquet"),
                             missing=float("inf"))


def test_other_reserved_values_still_rejected_with_a_sentinel():
    xs, _, y = make_floats(sentinel=-999.0)
    bad = xs.copy()
    bad[3, 1] = np.inf
    with pytest.raises(ValueError, match="inf"):
        fb_fit(bad, y, {"missing": -999.0})
    bad = xs.copy()
    bad[3, 1] = np.finfo(np.float32).max
    with pytest.raises(ValueError, match="f32::MAX"):
        fb_fit(bad, y, {"missing": -999.0})
    m = fb_fit(xs, y, {"missing": -999.0})
    with pytest.raises(ValueError, match="f32::MAX"):
        m.predict(bad, missing=-999.0)


def test_negative_largest_float_is_a_valid_sentinel():
    xs, xn, y = make_floats(sentinel=float(-np.finfo(np.float32).max))
    a = fb_fit(xs, y, {"missing": float(-np.finfo(np.float32).max)})
    b = fb_fit(xn, y)
    np.testing.assert_array_equal(a.predict(xn), b.predict(xn))


def test_default_is_nan_and_unchanged():
    xs, xn, y = make_floats()
    a = fb_fit(xn, y)
    b = fb_fit(xn, y, {"missing": float("nan")})
    np.testing.assert_array_equal(a.predict(xn), b.predict(xn))
    np.testing.assert_array_equal(a.predict(xn, missing=float("nan")), a.predict(xn))


def test_validation_set_gets_the_sentinel_too():
    # If the eval set ignored `missing`, its metric would differ from the
    # NaN-encoded run; identical histories prove it is applied.
    xs, xn, y = make_floats(sentinel=-999.0)
    h1, h2 = {}, {}
    fb_fit(xs, y, {"missing": -999.0, "eval_metric": "rmse"}, evals=[(xs, y, "v")], evals_result=h1)
    fb_fit(xn, y, {"eval_metric": "rmse"}, evals=[(xn, y, "v")], evals_result=h2)
    assert h1 == h2 and h1["v"]["rmse"]


def test_pandas_category_codes_are_subject_to_missing(tmp_path):
    # XGBoost drops the *codes* equal to `missing` for pandas categoricals too
    # (DMatrix num_nonmissing counts them out); FerrisBoost does the same.
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(21)
    n = 600
    cat = pd.Categorical(rng.choice(["a", "b", "c", "d"], n))
    num = rng.integers(1, 9, n).astype(np.float32)
    y = (np.array([1.0, -1.0, 0.5, 0.0])[cat.codes] + 0.2 * num + rng.normal(0, 0.2, n)).astype(
        np.float32)
    df = pd.DataFrame({"cat": cat, "num": num})
    params = {**BASE, "enable_categorical": True}
    a = fb.train({**params, "missing": 0.0}, df, label=y, num_boost_round=4, quiet=True)
    codes = np.stack([np.where(cat.codes == 0, np.nan, cat.codes).astype(np.float32), num], axis=1)
    b = fb.train(BASE, codes, label=y, num_boost_round=4, quiet=True, feature_types=["c", "q"])
    assert_same_model(a, b, tmp_path)
    d = xgb.DMatrix(df, label=y, enable_categorical=True, missing=0.0)
    assert d.num_nonmissing() == int((cat.codes != 0).sum() + (num != 0).sum())
