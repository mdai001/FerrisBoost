"""Named categories: category values stored in the model, re-coded by value.

Training from pandas string / integer categories, PyArrow dictionary tables and
Parquet string dictionaries (one or many files, every file / row group with its
own dictionary order) builds one category list per feature, stores it in the
model in XGBoost 3.x's `cats` layout and re-codes every later input by value.

Bit-exactness against XGBoost 3.4.1 uses lossless binning (categorical codes,
small-integer numeric columns): XGBoost trains on a DataFrame whose categories
are the list FerrisBoost stored, so both see the same codes.
"""
import json

import numpy as np
import pytest

import ferrisboost as fb

pd = pytest.importorskip("pandas")
pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
xgb = pytest.importorskip("xgboost")

FB = {"objective": "reg:squarederror", "max_depth": 4, "nthread": 2, "base_score": 0.5,
      "max_cat_to_onehot": 1, "max_bin": 255}
XGB = {"tree_method": "hist", "max_depth": 4, "nthread": 2, "base_score": 0.5,
       "max_cat_to_onehot": 1, "max_bin": 255}


def make(n=3000, k=40, seed=0, prefix="c"):
    rng = np.random.default_rng(seed)
    cats = [f"{prefix}{i}" for i in range(k)]
    v = rng.choice(cats, n).astype(object)
    v[rng.random(n) < 0.03] = None
    effect = {c: rng.normal() for c in cats}
    x1 = rng.integers(0, 8, n).astype(np.float32)
    signal = np.array([effect.get(c, 0.0) for c in v])
    y = (signal + 0.3 * x1 + rng.normal(0, 0.1, n)).astype(np.float32)
    return v, x1, y, cats


def frame(v, x1, categories):
    return pd.DataFrame({"x1": x1, "c": pd.Categorical(v, categories=categories)})


def xgb_model(df, y, rounds=5, params=None):
    return xgb.train({**XGB, **(params or {})}, xgb.DMatrix(df, y, enable_categorical=True), rounds)


def xgb_margin(bst, df):
    return bst.predict(xgb.DMatrix(df, enable_categorical=True), output_margin=True)


def model_json(model, tmp_path, name="m"):
    path = tmp_path / f"{name}.json"
    model.save_model(str(path), quiet=True)
    return json.loads(path.read_text())


def cats_of(doc):
    return doc["learner"]["gradient_booster"]["model"].get("cats")


def write_parts(tmp_path, v, x1, y, n_parts=3, seed=1, row_group_size=None, name="p"):
    """Parquet files whose string dictionaries list the values in different orders."""
    rng = np.random.default_rng(seed)
    files = []
    for k, idx in enumerate(np.array_split(np.arange(len(v)), n_parts)):
        vv = v[idx]
        order = list(rng.permutation(sorted({s for s in vv if s is not None})))
        keys = pa.array([None if s is None else order.index(s) for s in vv], pa.int32())
        arr = pa.DictionaryArray.from_arrays(keys, pa.array(order))
        path = str(tmp_path / f"{name}{k}.parquet")
        pq.write_table(pa.table({"x1": x1[idx], "c": arr, "label": y[idx]}), path,
                       row_group_size=row_group_size)
        files.append(path)
    return files


# --------------------------------------------------------------------------- pandas
def test_pandas_string_categories_match_xgboost_and_its_cats_layout(tmp_path):
    v, x1, y, cats = make()
    order = cats[::-1]
    df = frame(v, x1, order)
    ours = fb.train({**FB, "enable_categorical": True}, df, label=y, num_boost_round=5,
                    quiet=True)
    bst = xgb_model(df, y)
    assert ours.get_categories() == [None, order]
    np.testing.assert_array_equal(ours.predict(df, output_margin=True), xgb_margin(bst, df))
    doc = model_json(ours, tmp_path)
    ref = json.loads(bst.save_raw("json"))
    assert cats_of(doc) == cats_of(ref)
    ours_trees = doc["learner"]["gradient_booster"]["model"]["trees"]
    ref_trees = ref["learner"]["gradient_booster"]["model"]["trees"]
    for a, b in zip(ours_trees, ref_trees):
        for key in ("split_indices", "categories", "categories_nodes", "categories_sizes",
                    "left_children", "default_left"):
            assert a[key] == b[key], key


def test_prediction_recodes_reordered_and_subset_categories():
    v, x1, y, cats = make(seed=2)
    df = frame(v, x1, cats)
    m = fb.train({**FB, "enable_categorical": True}, df, label=y, num_boost_round=4, quiet=True)
    base = m.predict(df, output_margin=True)
    shuffled = frame(v, x1, list(np.random.default_rng(0).permutation(cats)))[["c", "x1"]]
    np.testing.assert_array_equal(m.predict(shuffled, output_margin=True), base)
    # Categories the frame does not use may be missing from its dictionary.
    sub = [c for c in cats if c in set(v[:50])]
    np.testing.assert_array_equal(
        m.predict(frame(v[:50], x1[:50], sub[::-1]), output_margin=True), base[:50])
    # pred_leaf / pred_contribs go through the same re-coding.
    codes = np.stack([x1, pd.Categorical(v, categories=cats).codes.astype(np.float32)], 1)
    codes[codes[:, 1] < 0, 1] = np.nan
    np.testing.assert_array_equal(m.predict(shuffled, pred_leaf=True),
                                  m.predict(codes, pred_leaf=True))
    np.testing.assert_array_equal(m.predict(shuffled, pred_contribs=True),
                                  m.predict(codes, pred_contribs=True))
    # An unused category the model does not know is ignored.
    extra = frame(v, x1, cats + ["never"])
    np.testing.assert_array_equal(m.predict(extra, output_margin=True), base)


def test_unseen_and_mismatched_categories_raise():
    v, x1, y, cats = make(seed=3)
    m = fb.train({**FB, "enable_categorical": True}, frame(v, x1, cats), label=y,
                 num_boost_round=2, quiet=True)
    with pytest.raises(ValueError, match="'zzz'.*not in the training categories"):
        m.predict(frame(np.array(["c1", "zzz"], object), x1[:2], ["c1", "zzz"]))
    ints = pd.DataFrame({"x1": x1[:2], "c": pd.Categorical([1, 2])})
    with pytest.raises(ValueError, match="must match"):
        m.predict(ints)
    with pytest.raises(ValueError, match="not in the training categories"):
        fb.train({**FB, "enable_categorical": True}, frame(v, x1, cats), label=y,
                 num_boost_round=1, quiet=True,
                 evals=[(frame(np.array(["c1", "new"], object), x1[:2], ["c1", "new"]),
                         y[:2], "v")])


def test_integer_categories_are_stored_with_their_dtype(tmp_path):
    rng = np.random.default_rng(4)
    n = 2000
    levels = np.array([30, -2, 7, 1000, 5], dtype=np.int32)
    v = rng.choice(levels, n)
    y = (v % 7 + rng.normal(0, 0.1, n)).astype(np.float32)
    df = pd.DataFrame({"k": pd.Categorical(v, categories=levels)})
    ours = fb.train({**FB, "enable_categorical": True}, df, label=y, num_boost_round=4,
                    quiet=True)
    bst = xgb_model(df, y, 4)
    np.testing.assert_array_equal(ours.predict(df, output_margin=True), xgb_margin(bst, df))
    assert cats_of(model_json(ours, tmp_path)) == cats_of(json.loads(bst.save_raw("json")))
    resorted = pd.DataFrame({"k": pd.Categorical(v, categories=np.sort(levels))})
    np.testing.assert_array_equal(ours.predict(resorted, output_margin=True), xgb_margin(bst, df))


def test_float_categories_are_rejected_like_xgboost():
    df = pd.DataFrame({"c": pd.Categorical([1.5, 2.5, 1.5, 2.5])})
    with pytest.raises(ValueError, match="strings or integers"):
        fb.train({**FB, "enable_categorical": True}, df, label=np.arange(4.0),
                 num_boost_round=1, quiet=True)


def test_model_json_round_trips_through_xgboost(tmp_path):
    v, x1, y, cats = make(seed=5)
    df = frame(v, x1, cats[::3] + [c for i, c in enumerate(cats) if i % 3])
    reordered = frame(v, x1, sorted(cats))
    ours = fb.train({**FB, "enable_categorical": True}, df, label=y, num_boost_round=4,
                    quiet=True)
    path = str(tmp_path / "fb.json")
    ours.save_model(path, quiet=True)
    loaded = xgb.Booster()
    loaded.load_model(path)
    np.testing.assert_array_equal(xgb_margin(loaded, reordered),
                                  ours.predict(reordered, output_margin=True))
    bst = xgb_model(df, y, 4)
    xpath = str(tmp_path / "x.json")
    bst.save_model(xpath)
    back = fb.Model.load_model(xpath, quiet=True)
    assert back.get_categories() == ours.get_categories()
    np.testing.assert_array_equal(back.predict(reordered, output_margin=True),
                                  xgb_margin(bst, reordered))


def test_integer_coded_models_have_no_cats(tmp_path):
    rng = np.random.default_rng(6)
    x = np.stack([rng.integers(0, 6, 500), rng.integers(0, 5, 500)], 1).astype(np.float32)
    y = x[:, 0] + rng.normal(0, 0.1, 500).astype(np.float32)
    m = fb.train(FB, x, label=y, feature_types=["c", "q"], num_boost_round=2, quiet=True)
    assert m.get_categories() is None
    assert cats_of(model_json(m, tmp_path)) is None


# --------------------------------------------------------------------------- Arrow
def test_arrow_table_with_differently_ordered_chunks():
    v, x1, y, cats = make(seed=7)
    half = len(v) // 2
    a = pa.array(v[:half]).dictionary_encode()
    b = pa.DictionaryArray.from_arrays(
        pa.array([None if s is None else sorted(cats).index(s) for s in v[half:]], pa.int32()),
        pa.array(sorted(cats)))
    table = pa.table({"x1": x1, "c": pa.chunked_array([a, b.cast(a.type)])})
    m = fb.train({**FB, "enable_categorical": True}, table, label=y, num_boost_round=4,
                 quiet=True)
    stored = m.get_categories()[1]
    df = frame(v, x1, stored)
    bst = xgb_model(df, y, 4)
    np.testing.assert_array_equal(m.predict(table, output_margin=True), xgb_margin(bst, df))
    np.testing.assert_array_equal(m.predict(frame(v, x1, cats), output_margin=True),
                                  xgb_margin(bst, df))


# --------------------------------------------------------------------------- Parquet
def test_parquet_files_with_different_dictionary_orders_match_xgboost(tmp_path):
    v, x1, y, cats = make(n=4000, seed=8)
    files = write_parts(tmp_path, v, x1, y, row_group_size=500)
    m = fb.train(FB, files, label="label", num_boost_round=5, quiet=True)
    stored = m.get_categories()[1]
    # The list is the order of first appearance in the rows (pandas.unique).
    assert stored == list(pd.unique(pd.Series(v).dropna()))
    # ... whatever the batching / thread count.
    single = fb.train({**FB, "nthread": 1}, files, label="label", num_boost_round=5, quiet=True)
    assert model_json(single, tmp_path, "n1") == model_json(m, tmp_path, "n2")
    df = frame(v, x1, stored)
    bst = xgb_model(df, y)
    want = xgb_margin(bst, df)
    np.testing.assert_array_equal(m.predict(files, output_margin=True), want)
    np.testing.assert_array_equal(m.predict(frame(v, x1, cats), output_margin=True), want)
    out = tmp_path / "pred.parquet"
    m.predict_to_parquet(files, str(out), output_margin=True)
    np.testing.assert_array_equal(pq.read_table(str(out)).column(0).to_numpy(), want)
    # Each file alone predicts its own rows the same way.
    first = np.array_split(np.arange(len(v)), 3)[0]
    np.testing.assert_array_equal(m.predict(files[0], output_margin=True), want[first])


def test_single_parquet_validation_and_cache(tmp_path):
    v, x1, y, cats = make(n=3000, seed=9)
    train_files = write_parts(tmp_path, v[:2000], x1[:2000], y[:2000], n_parts=1,
                              row_group_size=300, name="t")
    val_files = write_parts(tmp_path, v[2000:], x1[2000:], y[2000:], n_parts=1, seed=5,
                            name="v")
    hist = {}
    p = {**FB, "cache_path": str(tmp_path / "cache")}
    m = fb.train(p, train_files[0], label="label", num_boost_round=4, quiet=True,
                 evals=[(val_files[0], "v")], evals_result=hist)
    stored = m.get_categories()[1]
    df = frame(v[:2000], x1[:2000], stored)
    dval = xgb.DMatrix(frame(v[2000:], x1[2000:], stored), y[2000:], enable_categorical=True)
    res = {}
    bst = xgb.train(XGB, xgb.DMatrix(df, y[:2000], enable_categorical=True), 4,
                    evals=[(dval, "v")], evals_result=res, verbose_eval=False)
    np.testing.assert_array_equal(m.predict(df, output_margin=True), xgb_margin(bst, df))
    np.testing.assert_allclose(hist["v"]["rmse"], res["v"]["rmse"], rtol=1e-6)
    manifest = json.loads((tmp_path / "cache" / "manifest.json").read_text())
    assert manifest["categories"]["enc"][1]["offsets"][-1] > 0
    again = fb.train(p, train_files[0], label="label", num_boost_round=4, quiet=True)
    assert model_json(again, tmp_path, "a") == model_json(m, tmp_path, "b")
    # One reader (batches spanning row groups) or one per row group: same list.
    for threads in (1, 3):
        other_threads = fb.train({**FB, "nthread": threads}, train_files[0], label="label",
                                 num_boost_round=4, quiet=True)
        assert model_json(other_threads, tmp_path, "t") == model_json(m, tmp_path, "b")
    # Reusing the cache with other category lists (a continued model's) is refused.
    other = fb.train({**FB, "enable_categorical": True}, frame(v, x1, sorted(cats)),
                     label=y, num_boost_round=1, quiet=True)
    with pytest.raises(ValueError, match="cached category names"):
        fb.train(p, train_files[0], label="label", num_boost_round=1, quiet=True,
                 init_model=other)


def test_high_cardinality_string_dictionaries(tmp_path):
    rng = np.random.default_rng(10)
    n, k = 60000, 20000
    names = np.array([f"id_{i:05d}" for i in rng.permutation(k)], object)
    v = names[rng.integers(0, k, n)]
    effect = dict(zip(names, rng.normal(size=k)))
    x1 = rng.integers(0, 8, n).astype(np.float32)
    y = (np.array([effect[s] for s in v]) + 0.2 * x1 + rng.normal(0, 0.1, n)).astype(np.float32)
    files = write_parts(tmp_path, v, x1, y, n_parts=2, row_group_size=10000)
    p = {**FB, "max_cat_to_onehot": 4, "max_depth": 3}
    m = fb.train(p, files, label="label", num_boost_round=3, quiet=True)
    stored = m.get_categories()[1]
    assert len(stored) == len(set(v))
    df = frame(v, x1, stored)
    bst = xgb_model(df, y, 3, {"max_cat_to_onehot": 4, "max_depth": 3})
    np.testing.assert_array_equal(m.predict(files, output_margin=True), xgb_margin(bst, df))


def test_integer_dictionaries_in_files_keep_their_values_as_codes(tmp_path):
    rng = np.random.default_rng(11)
    codes = rng.integers(0, 6, 800)
    path = str(tmp_path / "i.parquet")
    y = (codes + rng.normal(0, 0.1, 800)).astype(np.float32)
    pq.write_table(pa.table({"k": pa.array(codes).dictionary_encode(), "label": y}), path)
    m = fb.train(FB, path, label="label", num_boost_round=2, quiet=True)
    assert m.get_categories() is None
    ref = fb.train(FB, codes.astype(np.float32)[:, None], label=y, feature_types=["c"],
                   num_boost_round=2, quiet=True)
    np.testing.assert_array_equal(m.predict(codes.astype(np.float32)[:, None]),
                                  ref.predict(codes.astype(np.float32)[:, None]))


def test_arrow_tables_and_parquet_files_code_integer_dictionaries_differently(tmp_path):
    """The documented split: an in-memory Arrow table's integer dictionary is a
    named categorical feature (like a pandas integer category: the categories
    are the dictionary values in dictionary order, stored with their dtype), a
    Parquet file's integer dictionary is not (its values are the codes)."""
    rng = np.random.default_rng(18)
    n = 1500
    levels = np.array([40, 3, 17, 9], dtype=np.int16)
    k = rng.choice(levels, n)
    y = (k % 5 + rng.normal(0, 0.1, n)).astype(np.float32)
    keys = pa.array([int(np.flatnonzero(levels == x)[0]) for x in k], pa.int8())
    column = pa.DictionaryArray.from_arrays(keys, pa.array(levels))
    table = pa.table({"k": column})
    from_table = fb.train({**FB, "enable_categorical": True}, table, label=y,
                          num_boost_round=3, quiet=True)
    assert from_table.get_categories() == [levels.tolist()]
    df = pd.DataFrame({"k": pd.Categorical(k, categories=levels)})
    bst = xgb_model(df, y, 3)
    want = xgb_margin(bst, df)
    np.testing.assert_array_equal(from_table.predict(table, output_margin=True), want)
    assert cats_of(model_json(from_table, tmp_path, "t")) == cats_of(json.loads(bst.save_raw("json")))
    # Re-coded by value from any other dictionary order, and from a Parquet file.
    resorted = pa.table({"k": pa.array(k).dictionary_encode()})
    np.testing.assert_array_equal(from_table.predict(resorted, output_margin=True), want)
    path = str(tmp_path / "k.parquet")
    pq.write_table(pa.table({"k": column, "label": y}), path)
    np.testing.assert_array_equal(from_table.predict(path, output_margin=True), want)
    # The same column in a Parquet file trains on its values as codes: no list,
    # one bin per code 0..40, the same model as the plain integer codes.
    from_file = fb.train(FB, path, label="label", num_boost_round=3, quiet=True)
    assert from_file.get_categories() is None
    ref = fb.train(FB, k.astype(np.float32)[:, None], label=y, feature_types=["c"],
                   num_boost_round=3, quiet=True)
    trees = lambda m, name: model_json(m, tmp_path, name)["learner"]["gradient_booster"]["model"]["trees"]  # noqa: E731
    assert trees(from_file, "f") == trees(ref, "r")


def test_plain_string_column_declared_categorical(tmp_path):
    v, x1, y, cats = make(n=1500, seed=12)
    path = str(tmp_path / "s.parquet")
    pq.write_table(pa.table({"x1": x1, "c": pa.array(v, pa.string()), "label": y}), path)
    m = fb.train(FB, path, label="label", feature_types=["q", "c"], num_boost_round=3,
                 quiet=True)
    df = frame(v, x1, m.get_categories()[1])
    np.testing.assert_array_equal(m.predict(path, output_margin=True),
                                  xgb_margin(xgb_model(df, y, 3), df))
    with pytest.raises(ValueError, match="only numeric columns"):
        fb.train(FB, path, label="label", num_boost_round=1, quiet=True)


def test_unsupported_dictionary_values_and_unseen_file_categories_raise(tmp_path):
    path = str(tmp_path / "b.parquet")
    pq.write_table(pa.table({"c": pa.array([b"a", b"b", b"a"]).dictionary_encode(),
                             "label": pa.array([0.0, 1.0, 0.0])}), path)
    with pytest.raises(ValueError):
        fb.train(FB, path, label="label", num_boost_round=1, quiet=True)
    v, x1, y, cats = make(n=600, seed=13)
    files = write_parts(tmp_path, v, x1, y, n_parts=1, name="u")
    m = fb.train(FB, files[0], label="label", num_boost_round=2, quiet=True)
    bad = str(tmp_path / "bad.parquet")
    pq.write_table(pa.table({"x1": x1[:3], "c": pa.array(["c1", "nope", None]).dictionary_encode()}),
                   bad)
    with pytest.raises(ValueError, match="'nope'|\"nope\""):
        m.predict(bad)


# --------------------------------------------------------------------------- continuation
def test_continuation_keeps_categories_and_recodes_new_data(tmp_path):
    v, x1, y, cats = make(seed=14)
    first = fb.train({**FB, "enable_categorical": True}, frame(v, x1, cats), label=y,
                     num_boost_round=2, quiet=True)
    v2, x2, y2, _ = make(seed=15)
    same = fb.train({**FB, "enable_categorical": True}, frame(v2, x2, cats), label=y2,
                    num_boost_round=2, quiet=True, init_model=first)
    other = fb.train({**FB, "enable_categorical": True}, frame(v2, x2, sorted(cats)),
                     label=y2, num_boost_round=2, quiet=True, init_model=first)
    assert other.get_categories() == first.get_categories()
    assert model_json(other, tmp_path, "o") == model_json(same, tmp_path, "s")
    # XGBoost gives the same model when the new frame keeps the model's order.
    bst = xgb_model(frame(v, x1, cats), y, 2)
    cont = xgb.train(XGB, xgb.DMatrix(frame(v2, x2, cats), y2, enable_categorical=True), 2,
                     xgb_model=bst)
    np.testing.assert_array_equal(same.predict(frame(v2, x2, cats), output_margin=True),
                                  xgb_margin(cont, frame(v2, x2, cats)))
    with pytest.raises(ValueError, match="not in the training categories"):
        fb.train({**FB, "enable_categorical": True},
                 frame(np.array(["c1", "brand-new"], object), x2[:2], ["c1", "brand-new"]),
                 label=y2[:2], num_boost_round=1, quiet=True, init_model=first)
    # Files continue the same way.
    files = write_parts(tmp_path, v2, x2, y2, n_parts=2, name="cont")
    from_files = fb.train(FB, files, label="label", num_boost_round=2, quiet=True,
                          init_model=first)
    assert from_files.get_categories() == first.get_categories()
    np.testing.assert_array_equal(from_files.predict(frame(v2, x2, cats), output_margin=True),
                                  same.predict(frame(v2, x2, cats), output_margin=True))


# --------------------------------------------------------------------------- sklearn
def test_sklearn_estimator_stores_categories_in_the_booster(tmp_path):
    pytest.importorskip("sklearn")
    v, x1, y, cats = make(seed=16)
    est = fb.FerrisBoostRegressor(n_estimators=4, max_depth=4, enable_categorical=True,
                                  max_cat_to_onehot=1)
    est.fit(frame(v, x1, cats), y)
    path = str(tmp_path / "est.json")
    est.save_model(path)
    loaded = fb.FerrisBoostRegressor(enable_categorical=True)
    loaded.load_model(path)
    shuffled = frame(v, x1, sorted(cats))
    np.testing.assert_array_equal(loaded.predict(shuffled), est.predict(frame(v, x1, cats)))
    with pytest.raises(ValueError, match="not in the training categories"):
        loaded.predict(frame(np.array(["c1", "zz"], object), x1[:2], ["c1", "zz"]))


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
@pytest.mark.parametrize("k", [40, 3000])
def test_gpu_training_and_prediction_match_the_cpu(tmp_path, k):
    v, x1, y, cats = make(n=6000, k=k, seed=17)
    files = write_parts(tmp_path, v, x1, y, row_group_size=700)
    cpu = fb.train(FB, files, label="label", num_boost_round=4, quiet=True)
    gpu = fb.train({**FB, "device": "cuda"}, files, label="label", num_boost_round=4, quiet=True)
    assert model_json(gpu, tmp_path, "g") == model_json(cpu, tmp_path, "c")
    df = frame(v, x1, sorted(cats))
    np.testing.assert_array_equal(cpu.predict(df, use_gpu=True, output_margin=True),
                                  cpu.predict(df, output_margin=True))
    np.testing.assert_array_equal(cpu.predict(files, use_gpu=True, output_margin=True),
                                  cpu.predict(files, output_margin=True))
