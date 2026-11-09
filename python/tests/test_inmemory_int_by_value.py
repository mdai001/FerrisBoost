"""In-memory integer columns mean what they mean in a file.

* A model with an integer category list for a feature looks up a plain integer
  column of a pandas DataFrame or an Arrow table / RecordBatch by value, as it
  looks up a CSV or plain Parquet integer column: same outputs, the same
  `ValueError` for a value outside the list, float columns stay codes.
* A model without a list reads a pandas integer `category` column by its
  values, as it reads an Arrow or Parquet integer dictionary; a pandas string
  category keeps its codes.
* NumPy arrays are codes in both cases.

Prediction (margin, leaf indices, SHAP; CPU and GPU), validation sets with
early stopping and the scikit-learn estimators all follow these rules.
"""
import numpy as np
import pytest

import ferrisboost as fb

pd = pytest.importorskip("pandas")
pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

FB = {"objective": "reg:squarederror", "max_depth": 4, "nthread": 2, "base_score": 0.5,
      "max_cat_to_onehot": 1, "max_bin": 255, "enable_categorical": True}
LEVELS = [30, 10, 20, 0, 41, 7, 25]


def data(n=1000, seed=0):
    rng = np.random.default_rng(seed)
    k = rng.choice(LEVELS, n).astype(np.float64)
    k[rng.random(n) < 0.05] = np.nan
    x = rng.integers(0, 8, n).astype(np.float32)
    y = (np.nan_to_num(k, nan=15.0) % 7 + 0.3 * x + rng.normal(0, 0.1, n)).astype(np.float32)
    return k, x, y


def category_frame(k, x, order=LEVELS):
    return pd.DataFrame({"k": pd.Categorical(k, categories=order), "x": x})


def nullable(k):
    return pd.array([pd.NA if np.isnan(v) else int(v) for v in k], dtype="Int64")


def arrow_ints(k, chunks=1, type_=pa.int64()):
    values = [None if np.isnan(v) else int(v) for v in k]
    return pa.chunked_array([pa.array(list(p), type_) for p in np.array_split(values, chunks)])


def write_csv(path, k, x, fmt="{:d}"):
    with open(path, "w") as f:
        f.write("k,x\n")
        for a, b in zip(k, x):
            f.write(("" if np.isnan(a) else fmt.format(int(a) if fmt == "{:d}" else a))
                    + f",{int(b)}\n")
    return str(path)


def outputs(model, d, use_gpu=False):
    return [model.predict(d, output_margin=True, use_gpu=use_gpu),
            model.predict(d, pred_leaf=True, use_gpu=use_gpu),
            model.predict(d, pred_contribs=True, use_gpu=use_gpu)]


def assert_same(a, b):
    for u, v in zip(a, b):
        np.testing.assert_array_equal(u, v)


# ------------------------------------------------- a model with integer categories
def list_model(seed=0, rounds=4, **kw):
    k, x, y = data(seed=seed)
    model = fb.train(FB, category_frame(k, x), label=y, num_boost_round=rounds, quiet=True,
                     **kw)
    assert model.get_categories() == [LEVELS, None]
    return model


def plain_inputs(k, x, tmp_path):
    """The same rows as plain integer columns in memory and in files."""
    xs = arrow_ints(x, 3, pa.int32())
    pq_path = str(tmp_path / "plain.parquet")
    pq.write_table(pa.table({"k": arrow_ints(k), "x": arrow_ints(x)}), pq_path)
    assert pa.types.is_integer(pq.read_schema(pq_path).field("k").type)
    return {
        "pandas Int64": pd.DataFrame({"k": nullable(k), "x": x}),
        "arrow chunked": pa.table({"k": arrow_ints(k, 3), "x": xs}),
        "arrow uint16": pa.table({"k": arrow_ints(k, 1, pa.uint16()), "x": x}),
        "record batch": pa.table({"k": arrow_ints(k), "x": x}).to_batches()[0],
        "parquet": pq_path,
        "csv": write_csv(tmp_path / "plain.csv", k, x),
    }


def test_plain_integers_are_looked_up_by_value_like_files(tmp_path):
    model = list_model()
    k, x, _ = data(n=600, seed=1)
    want = outputs(model, category_frame(k, x, sorted(LEVELS)))
    for d in plain_inputs(k, x, tmp_path).values():
        assert_same(outputs(model, d), want)
    # numpy int64 column without missing values, columns in another order.
    full = np.where(np.isnan(k), 10, k).astype(np.int64)
    np.testing.assert_array_equal(
        model.predict(pd.DataFrame({"x": x, "k": full})),
        model.predict(category_frame(full.astype(float), x)))
    # The missing sentinel applies to the looked-up codes, as for a file.
    inputs = plain_inputs(k, x, tmp_path)
    np.testing.assert_array_equal(model.predict(inputs["pandas Int64"], missing=1.0),
                                  model.predict(inputs["parquet"], missing=1.0))
    np.testing.assert_array_equal(model.predict(inputs["arrow chunked"], missing=1.0),
                                  model.predict(inputs["parquet"], missing=1.0))


def test_numpy_and_float_columns_stay_codes(tmp_path):
    model = list_model()
    k, x, _ = data(n=600, seed=2)
    codes = np.column_stack([k, x]).astype(np.float32)  # values used as codes
    want = outputs(model, codes)
    assert not np.array_equal(want[0], model.predict(plain_inputs(k, x, tmp_path)["parquet"],
                                                     output_margin=True))
    pq_path = str(tmp_path / "float.parquet")
    pq.write_table(pa.table({"k": k, "x": x}), pq_path)
    for d in (pd.DataFrame({"k": k, "x": x}), pa.table({"k": k, "x": x}), pq_path,
              write_csv(tmp_path / "float.csv", k, x, fmt="{:.1f}")):
        assert_same(outputs(model, d), want)


def test_unseen_values_raise_like_files(tmp_path):
    model = list_model()
    k, x, _ = data(n=50, seed=3)
    k[7] = 99
    for d in plain_inputs(k, x, tmp_path).values():
        with pytest.raises(ValueError, match="99"):
            model.predict(d)
    with pytest.raises(ValueError, match="not in the training categories"):
        model.predict(pa.table({"k": pa.array([10, -1], pa.int8()), "x": [1.0, 2.0]}))
    with pytest.raises(ValueError, match="not in the training categories"):
        model.predict(pa.table({"k": pa.array([10, 2**63], pa.uint64()), "x": [1.0, 2.0]}))


def test_string_categories_keep_plain_integers_as_codes(tmp_path):
    k, x, y = data(seed=4)
    names = [None if np.isnan(v) else str(int(v)) for v in k]
    model = fb.train(FB, pd.DataFrame({"k": pd.Categorical(names), "x": x}), label=y,
                     num_boost_round=3, quiet=True)
    codes = np.column_stack([np.where(np.isnan(k), np.nan, k % 5), x]).astype(np.float32)
    plain = pd.DataFrame({"k": nullable(codes[:, 0]), "x": x})
    path = str(tmp_path / "s.parquet")
    pq.write_table(pa.table({"k": arrow_ints(codes[:, 0]), "x": x}), path)
    assert_same(outputs(model, plain), outputs(model, codes))
    assert_same(outputs(model, path), outputs(model, codes))


def test_validation_sets_look_up_plain_integers(tmp_path):
    k, x, y = data(seed=5)
    vk, vx, vy = data(n=400, seed=6)
    runs = {}
    for name, ev in (("category", category_frame(vk, vx, sorted(LEVELS))),
                     ("pandas", pd.DataFrame({"k": nullable(vk), "x": vx})),
                     ("arrow", pa.table({"k": arrow_ints(vk, 2), "x": vx}))):
        model = fb.train(FB, category_frame(k, x), label=y, num_boost_round=10, quiet=True,
                         evals=[(ev, vy, "v")], early_stopping_rounds=2)
        runs[name] = (model.evals_result(), model.best_iteration)
    assert runs["pandas"] == runs["category"] and runs["arrow"] == runs["category"]
    # A column forced numeric by feature_types is not looked up.
    model = fb.train(FB, category_frame(k, x), label=y, num_boost_round=3, quiet=True,
                     feature_types=["q", "q"],
                     evals=[(pd.DataFrame({"k": nullable(vk), "x": vx}), vy, "v")])
    ref = fb.train(FB, category_frame(k, x), label=y, num_boost_round=3, quiet=True,
                   feature_types=["q", "q"],
                   evals=[(np.column_stack([vk, vx]).astype(np.float32), vy, "v")])
    assert model.evals_result() == ref.evals_result()


@pytest.mark.filterwarnings("ignore:X has feature names")
def test_sklearn_looks_up_plain_integers(tmp_path):
    pytest.importorskip("sklearn")
    k, x, y = data(seed=7)
    vk, vx, vy = data(n=300, seed=8)
    frame = category_frame(k, x)
    cat_eval = category_frame(vk, vx, sorted(LEVELS))
    plain_eval = pd.DataFrame({"k": nullable(vk), "x": vx})
    table_eval = pa.table({"k": arrow_ints(vk), "x": vx})
    kw = dict(n_estimators=4, max_depth=4, max_cat_to_onehot=1, enable_categorical=True)
    est = fb.FerrisBoostRegressor(**kw).fit(frame, y, eval_set=[(plain_eval, vy)])
    ref = fb.FerrisBoostRegressor(**kw).fit(frame, y, eval_set=[(cat_eval, vy)])
    assert est.evals_result() == ref.evals_result()
    for d in (plain_eval, table_eval):
        np.testing.assert_array_equal(est.predict(d), ref.predict(cat_eval))
        np.testing.assert_array_equal(est.apply(d), ref.apply(cat_eval))
    clf = fb.FerrisBoostClassifier(**kw).fit(frame, (y > np.median(y)).astype(int))
    np.testing.assert_array_equal(clf.predict_proba(plain_eval), clf.predict_proba(cat_eval))
    with pytest.raises(ValueError, match="not in the training categories"):
        est.predict(pd.DataFrame({"k": [99], "x": [1.0]}))
    # A loaded estimator brings the booster's categories.
    path = str(tmp_path / "est.json")
    est.save_model(path)
    loaded = fb.FerrisBoostRegressor()
    loaded.load_model(path)
    np.testing.assert_array_equal(loaded.predict(table_eval), ref.predict(cat_eval))


@pytest.mark.parametrize("prior_kind", ["integer", "string"])
@pytest.mark.parametrize("current_types", [("c", "q"), ("q", "q")])
@pytest.mark.filterwarnings("ignore:X has feature names")
def test_sklearn_refit_does_not_reuse_prior_categories(prior_kind, current_types):
    pytest.importorskip("sklearn")
    k, x, y = data(seed=15)
    if prior_kind == "integer":
        prior_frame = category_frame(k, x)
    else:
        names = [None if np.isnan(v) else str(int(v)) for v in k]
        prior_frame = pd.DataFrame({"k": pd.Categorical(names), "x": x})
    # Code 99 is valid raw categorical input for the refit, but is absent from
    # either prior category list. A stale integer list would reject it while
    # preparing either pandas or Arrow validation data.
    k[0] = 99
    raw = np.column_stack([k, x]).astype(np.float32)

    plain_frame = pd.DataFrame({"k": nullable(k), "x": x})
    plain_arrow = pa.table({"k": arrow_ints(k), "x": x})
    kw = dict(n_estimators=3, max_depth=3, enable_categorical=True)

    for evaluation in (plain_frame, plain_arrow):
        est = fb.FerrisBoostRegressor(**kw).fit(prior_frame, y)
        fresh = fb.FerrisBoostRegressor(**kw)
        # Exercise a current-fit explicit feature-type override too. The prior
        # model still has categories, but this fit's eval conversion follows
        # its own empty category state and the explicit q/c feature types.
        est.feature_types = list(current_types)
        fresh.feature_types = list(current_types)
        est.fit(raw, y, eval_set=[(evaluation, y)])
        fresh.fit(raw, y, eval_set=[(evaluation, y)])
        assert est.evals_result() == fresh.evals_result()
        np.testing.assert_array_equal(est.predict(plain_frame), fresh.predict(raw))
        np.testing.assert_array_equal(est.predict(plain_arrow), fresh.predict(raw))


# --------------------------------------------------------- a model without a list
def codes_model(seed=9, rounds=4, **kw):
    k, x, y = data(seed=seed)
    # Named numeric columns let the CSV fixtures bind by header while keeping
    # the categorical feature's codes raw and without a stored category list.
    model = fb.train(FB, pd.DataFrame({"k": k, "x": x}), label=y,
                     feature_types=["c", "q"], num_boost_round=rounds, quiet=True, **kw)
    assert model.get_categories() is None
    return model


def value_inputs(k, x, tmp_path):
    """Integer categories / dictionaries in any order, and the plain values."""
    order = [41, 0, 99, 7, 30, 25, 10, 20]  # unordered, one unused
    pos = {v: i for i, v in enumerate(order)}
    keys = pa.array([None if np.isnan(v) else pos[int(v)] for v in k], pa.int8())
    table = pa.table({"k": pa.DictionaryArray.from_arrays(keys, pa.array(order, pa.int32())),
                      "x": x})
    path = str(tmp_path / "dict.parquet")
    pq.write_table(table, path)
    return {
        "pandas category": category_frame(k, x, order),
        "pandas uint8 category": pd.DataFrame(
            {"k": pd.Categorical(k, categories=pd.Index(order, dtype="uint8")), "x": x}),
        "arrow dictionary": table,
        "parquet dictionary": path,
        "csv": write_csv(tmp_path / "v.csv", k, x),
    }


def test_integer_categories_are_read_by_value_without_a_list(tmp_path):
    model = codes_model()
    k, x, _ = data(n=600, seed=10)
    want = outputs(model, np.column_stack([k, x]).astype(np.float32))
    for d in value_inputs(k, x, tmp_path).values():
        assert_same(outputs(model, d), want)
    # Bound by name: column order does not matter.
    assert_same(outputs(model, category_frame(k, x)[["x", "k"]]), want)


def test_string_categories_keep_codes_without_a_list():
    model = codes_model()
    names = ["p", "q", None, "r", "q"]
    frame = pd.DataFrame({"k": pd.Categorical(names, categories=["r", "q", "p"]),
                          "x": np.arange(5, dtype=np.float32)})
    arr = np.column_stack([[2, 1, np.nan, 0, 1], np.arange(5)]).astype(np.float32)
    np.testing.assert_array_equal(model.predict(frame), model.predict(arr))


def test_validation_sets_read_category_values_without_a_list(tmp_path):
    k, x, y = data(seed=11)
    vk, vx, vy = data(n=400, seed=12)
    runs = {}
    for name, ev in (("codes", np.column_stack([vk, vx]).astype(np.float32)),
                     ("pandas", category_frame(vk, vx, sorted(LEVELS, reverse=True)))):
        model = fb.train(FB, np.column_stack([k, x]).astype(np.float32), label=y,
                         feature_types=["c", "q"], num_boost_round=10, quiet=True,
                         evals=[(ev, vy, "v")], early_stopping_rounds=2)
        runs[name] = (model.evals_result(), model.best_iteration)
    assert runs["pandas"] == runs["codes"]


@pytest.mark.filterwarnings("ignore:X has feature names")
def test_sklearn_reads_category_values_without_a_list():
    pytest.importorskip("sklearn")
    k, x, y = data(seed=13)
    arr = np.column_stack([k, x]).astype(np.float32)
    frame = category_frame(k, x, sorted(LEVELS, reverse=True))
    kw = dict(n_estimators=4, max_depth=4, max_cat_to_onehot=1, feature_types=["c", "q"],
              enable_categorical=True)
    est = fb.FerrisBoostRegressor(**kw).fit(arr, y, eval_set=[(frame, y)])
    ref = fb.FerrisBoostRegressor(**kw).fit(arr, y, eval_set=[(arr, y)])
    assert est.evals_result() == ref.evals_result()
    np.testing.assert_array_equal(est.predict(frame), est.predict(arr))
    np.testing.assert_array_equal(est.apply(frame), est.apply(arr))


# --------------------------------------------------------------------------- GPU
def _has_cuda():
    try:
        fb.train({"device": "cuda", "max_depth": 2, "objective": "reg:squarederror"},
                 np.zeros((64, 2), np.float32), np.zeros(64, np.float32), num_boost_round=1,
                 quiet=True)
        return True
    except Exception as e:  # noqa: BLE001
        if "CUDA" in str(e) or "cuda" in str(e):
            return False
        raise


@pytest.mark.skipif(not _has_cuda(), reason="no usable CUDA build or device")
def test_gpu_prediction_follows_the_same_rules(tmp_path):
    k, x, _ = data(n=600, seed=14)
    model = list_model()
    want = outputs(model, category_frame(k, x))
    for d in plain_inputs(k, x, tmp_path).values():
        assert_same(outputs(model, d, use_gpu=True), want)
    model = codes_model()
    want = outputs(model, np.column_stack([k, x]).astype(np.float32))
    for d in value_inputs(k, x, tmp_path).values():
        assert_same(outputs(model, d, use_gpu=True), want)
