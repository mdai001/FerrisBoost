"""In-memory Arrow integer dictionaries scored by a model without category lists.

A model with no `cats` for a feature (trained on integer codes: NumPy, a plain
integer column, a Parquet integer dictionary) takes an Arrow table's integer
dictionary by its **values**, exactly as it reads the same column from a
Parquet file: predictions, explanations and validation sets agree with the
Parquet file and with the plain integer codes, on CPU and GPU, whatever the
dictionary order and however the table is chunked. String dictionaries have no
values to look up in such a model and keep being read by index.
"""
import numpy as np
import pytest

import ferrisboost as fb

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

FB = {"objective": "reg:squarederror", "max_depth": 4, "nthread": 2, "base_score": 0.5,
      "max_cat_to_onehot": 1, "max_bin": 255}
LEVELS = [30, 10, 20, 0, 41, 7, 25]


def data(n=1200, seed=0):
    rng = np.random.default_rng(seed)
    k = rng.choice(LEVELS, n).astype(np.float64)
    k[rng.random(n) < 0.05] = np.nan
    x = rng.integers(0, 8, n).astype(np.float32)
    y = (np.nan_to_num(k, nan=15.0) % 7 + 0.3 * x + rng.normal(0, 0.1, n)).astype(np.float32)
    return k, x, y


def dict_column(k, order, index_type=pa.int16(), value_type=pa.int32()):
    """`k` (floats, NaN = missing) as a dictionary over `order`; a missing row
    is a null index."""
    pos = {v: i for i, v in enumerate(order)}
    keys = pa.array([None if np.isnan(v) else pos[int(v)] for v in k], index_type)
    return pa.DictionaryArray.from_arrays(keys, pa.array(order, value_type))


def chunked(k, orders):
    """A chunked column whose every chunk has its own dictionary order."""
    parts = np.array_split(k, len(orders))
    return pa.chunked_array([dict_column(p, o) for p, o in zip(parts, orders)])


def codes_model(k, x, y, rounds=4, **kw):
    return fb.train(FB, np.column_stack([k, x]).astype(np.float32), label=y,
                    feature_types=["c", "q"], num_boost_round=rounds, quiet=True, **kw)


def outputs(model, data_, use_gpu=False):
    return [model.predict(data_, output_margin=True, use_gpu=use_gpu),
            model.predict(data_, pred_leaf=True, use_gpu=use_gpu),
            model.predict(data_, pred_contribs=True, use_gpu=use_gpu)]


def assert_same(a, b):
    for u, v in zip(a, b):
        np.testing.assert_array_equal(u, v)


def inputs(k, x, tmp_path):
    """The same rows as plain codes, an unordered dictionary table, a table of
    chunks with different dictionaries, and a Parquet file."""
    arr = np.column_stack([k, x]).astype(np.float32)
    table = pa.table({"k": dict_column(k, LEVELS), "x": x})
    chunks = pa.table({"k": chunked(k, [LEVELS, LEVELS[::-1], [41, 0, 7, 30, 25, 10, 20, 99]]),
                       "x": pa.chunked_array(np.array_split(x, 3))})
    assert chunks.column("k").num_chunks == 3
    path = str(tmp_path / "k.parquet")
    pq.write_table(table, path)
    return arr, table, chunks, path


def test_unordered_and_chunked_integer_dictionaries_score_like_parquet_and_codes(tmp_path):
    k, x, y = data()
    model = codes_model(k, x, y)
    assert model.get_categories() is None
    arr, table, chunks, path = inputs(k, x, tmp_path)
    want = outputs(model, arr)
    assert_same(outputs(model, table), want)
    assert_same(outputs(model, chunks), want)
    assert_same(outputs(model, table.to_batches()[0]), want)
    assert_same(outputs(model, path), want)
    np.testing.assert_array_equal(model.predict(table, predict_threads=1),
                                  model.predict(arr, predict_threads=1))


def test_a_model_trained_from_a_parquet_integer_dictionary(tmp_path):
    k, x, y = data(seed=1)
    path = str(tmp_path / "train.parquet")
    pq.write_table(pa.table({"k": dict_column(k, LEVELS), "x": x, "label": y}), path)
    model = fb.train(FB, path, label="label", num_boost_round=4, quiet=True)
    assert model.get_categories() is None and model.feature_names == ["k", "x"]
    arr, table, chunks, test_path = inputs(k, x, tmp_path)
    want = outputs(model, test_path)
    assert_same(outputs(model, arr), want)
    # Bound by name: column order of the table does not matter.
    assert_same(outputs(model, table.select(["x", "k"])), want)
    assert_same(outputs(model, chunks), want)


def test_value_rules_match_parquet(tmp_path):
    """Out-of-range, negative and fractional-free large values, null indices and
    null dictionary entries are read as Parquet reads them: missing goes the
    missing direction, other values follow the model's category sets."""
    k, x, y = data(seed=2)
    model = codes_model(k, x, y)
    rng = np.random.default_rng(3)

    def case(order):
        keys = pa.array(rng.integers(0, len(order), 600).tolist() + [None] * 10, pa.int32())
        column = pa.DictionaryArray.from_arrays(keys, pa.array(order, pa.int64()))
        xs = rng.integers(0, 8, len(keys)).astype(np.float32)
        plain = np.array([np.nan if v is None else float(v) for v in column.to_pylist()],
                         np.float64).astype(np.float32)
        return pa.table({"k": column, "x": xs}), np.column_stack([plain, xs])

    table, arr = case([-5, 30, 70000, 10, 2**40, 20])
    path = str(tmp_path / "odd.parquet")
    pq.write_table(table, path)
    want = outputs(model, arr)
    assert_same(outputs(model, path), want)
    assert_same(outputs(model, table), want)
    # The missing sentinel applies to the values, as for a file.
    np.testing.assert_array_equal(model.predict(table, missing=30.0),
                                  model.predict(path, missing=30.0))
    # A null dictionary entry (Parquet cannot store one) is missing too.
    table, arr = case([30, None, 10])
    assert np.isnan(arr[:, 0]).sum() > 10
    assert_same(outputs(model, table), outputs(model, arr))


def test_validation_sets_read_dictionary_values(tmp_path):
    k, x, y = data(seed=4)
    vk, vx, vy = data(n=500, seed=5)
    arr, table, chunks, _ = inputs(vk, vx, tmp_path)
    runs = {}
    for name, ev in (("codes", arr), ("table", table), ("chunks", chunks)):
        model = codes_model(k, x, y, rounds=8, evals=[(ev, vy, "v")],
                            early_stopping_rounds=2)
        runs[name] = (model.evals_result(), model.best_iteration)
    assert runs["table"] == runs["codes"] and runs["chunks"] == runs["codes"]


def test_mixed_model_reads_values_only_for_features_without_a_list():
    """A model with a list for one feature re-codes that one by value and reads
    another feature's integer dictionary values as its codes."""
    pd = pytest.importorskip("pandas")
    k, x, y = data(seed=6)
    rng = np.random.default_rng(6)
    names = rng.choice(["a", "b", "c"], len(k))
    df = pd.DataFrame({"s": pd.Categorical(names, categories=["a", "b", "c"]), "k": k})
    model = fb.train({**FB, "enable_categorical": True}, df, label=y,
                     feature_types=["c", "c"], num_boost_round=4, quiet=True)
    assert model.get_categories() == [["a", "b", "c"], None]
    table = pa.table({"k": dict_column(k, LEVELS),
                      "s": pa.array(names).dictionary_encode()})
    np.testing.assert_array_equal(model.predict(table), model.predict(df))


def test_string_dictionaries_on_a_model_without_lists_keep_their_indices():
    k, x, y = data(seed=7)
    model = codes_model(np.where(np.isnan(k), np.nan, k % 3), x, y)
    column = pa.DictionaryArray.from_arrays(
        pa.array([0, 2, 1, None, 1], pa.int8()), pa.array(["p", "q", "r"]))
    table = pa.table({"k": column, "x": x[:5]})
    arr = np.column_stack([[0, 2, 1, np.nan, 1], x[:5]]).astype(np.float32)
    np.testing.assert_array_equal(model.predict(table), model.predict(arr))


@pytest.mark.filterwarnings("ignore:X has feature names")
def test_sklearn_estimator_reads_dictionary_values(tmp_path):
    pytest.importorskip("sklearn")
    k, x, y = data(seed=8)
    arr, table, chunks, _ = inputs(k, x, tmp_path)
    est = fb.FerrisBoostRegressor(n_estimators=4, max_depth=4, max_cat_to_onehot=1,
                                  feature_types=["c", "q"], enable_categorical=True)
    est.fit(arr, y, eval_set=[(table, y)])
    ref = fb.FerrisBoostRegressor(n_estimators=4, max_depth=4, max_cat_to_onehot=1,
                                  feature_types=["c", "q"], enable_categorical=True)
    ref.fit(arr, y, eval_set=[(arr, y)])
    assert est.evals_result() == ref.evals_result()
    np.testing.assert_array_equal(est.predict(table), est.predict(arr))
    np.testing.assert_array_equal(est.predict(chunks), est.predict(arr))
    np.testing.assert_array_equal(est.apply(table), est.apply(arr))
    clf = fb.FerrisBoostClassifier(n_estimators=3, max_depth=3, feature_types=["c", "q"],
                                   enable_categorical=True)
    clf.fit(arr, (y > np.median(y)).astype(int))
    np.testing.assert_array_equal(clf.predict_proba(table), clf.predict_proba(arr))


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
def test_gpu_prediction_reads_dictionary_values(tmp_path):
    k, x, y = data(seed=9)
    model = codes_model(k, x, y)
    arr, table, chunks, path = inputs(k, x, tmp_path)
    want = outputs(model, arr)
    for d in (arr, table, chunks, path):
        assert_same(outputs(model, d, use_gpu=True), want)


def test_sklearn_fit_on_a_dictionary_table_codes_like_train():
    """An estimator fitted on an Arrow table codes it as `fb.train` does (an
    integer dictionary is a named categorical feature; a null stays missing)."""
    pytest.importorskip("sklearn")
    k, x, y = data(seed=10)
    table = pa.table({"k": dict_column(k, LEVELS), "x": x})
    est = fb.FerrisBoostRegressor(n_estimators=4, max_depth=4, max_cat_to_onehot=1,
                                  enable_categorical=True, n_jobs=2, base_score=0.5)
    est.fit(table, y)
    ref = fb.train({**FB, "enable_categorical": True}, table, label=y, num_boost_round=4,
                   quiet=True)
    assert est.get_booster().get_categories() == ref.get_categories() == [LEVELS, None]
    np.testing.assert_array_equal(est.predict(table), ref.predict(table))
    resorted = pa.table({"k": dict_column(k, sorted(LEVELS)), "x": x})
    np.testing.assert_array_equal(est.predict(resorted), ref.predict(table))
