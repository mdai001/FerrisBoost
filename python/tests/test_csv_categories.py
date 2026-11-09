"""CSV text columns as named categories.

A CSV column declared categorical (`"c"` in `feature_types`) whose fields are
not all numbers is read as text: training learns its category list in the
order the values first appear in the rows (files in order), like a Parquet
string column, stores it in the model `cats` and every later read (validation,
prediction from CSV, pandas, Arrow or Parquet) re-codes by value.

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
TYPES = ["q", "c"]


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


def write_csv(path, v, x1, y=None):
    """x1, c (text, empty = missing) and an optional label column."""
    cols = {"x1": x1.astype(np.int64), "c": pd.Series(v, dtype=object)}
    if y is not None:
        cols["label"] = y
    pd.DataFrame(cols).to_csv(path, index=False, na_rep="")
    return str(path)


def xgb_model(df, y, rounds=4, params=None):
    return xgb.train({**XGB, **(params or {})}, xgb.DMatrix(df, y, enable_categorical=True), rounds)


def xgb_margin(bst, df):
    return bst.predict(xgb.DMatrix(df, enable_categorical=True), output_margin=True)


def model_json(model, tmp_path, name="m"):
    path = tmp_path / f"{name}.json"
    model.save_model(str(path), quiet=True)
    return path.read_bytes()


def first_seen(v):
    return list(pd.unique(pd.Series(v).dropna()))


def test_csv_text_column_trains_named_categories_and_matches_xgboost(tmp_path):
    v, x1, y, cats = make()
    path = write_csv(tmp_path / "t.csv", v, x1, y)
    m = fb.train(FB, path, label="label", feature_types=TYPES, num_boost_round=4, quiet=True)
    stored = m.get_categories()[1]
    assert m.get_categories()[0] is None
    assert stored == first_seen(v)
    df = frame(v, x1, stored)
    bst = xgb_model(df, y)
    want = xgb_margin(bst, df)
    np.testing.assert_array_equal(m.predict(df, output_margin=True), want)
    # Every input form re-codes by value: CSV, pandas with another category
    # order, an Arrow dictionary table and a Parquet string dictionary.
    np.testing.assert_array_equal(m.predict(path, output_margin=True), want)
    np.testing.assert_array_equal(m.predict(frame(v, x1, sorted(cats)), output_margin=True), want)
    table = pa.table({"x1": x1, "c": pa.array(v, pa.string()).dictionary_encode()})
    np.testing.assert_array_equal(m.predict(table, output_margin=True), want)
    pq_path = str(tmp_path / "p.parquet")
    pq.write_table(table, pq_path)
    np.testing.assert_array_equal(m.predict(pq_path, output_margin=True), want)
    # The model loads in XGBoost 3.4.1 with its categories.
    saved = str(tmp_path / "fb.json")
    m.save_model(saved, quiet=True)
    loaded = xgb.Booster()
    loaded.load_model(saved)
    reordered = frame(v, x1, sorted(cats))
    np.testing.assert_array_equal(xgb_margin(loaded, reordered), want)
    # The same rows from a Parquet string column train the same model.
    pq_train = str(tmp_path / "train.parquet")
    pq.write_table(pa.table({"x1": x1.astype(np.int64), "c": pa.array(v, pa.string()),
                             "label": y}), pq_train)
    from_pq = fb.train(FB, pq_train, label="label", feature_types=TYPES, num_boost_round=4,
                       quiet=True)
    assert model_json(from_pq, tmp_path, "pq") == model_json(m, tmp_path, "csv")


def test_multi_file_csv_learns_one_list_in_row_order(tmp_path):
    v, x1, y, cats = make(n=6000, seed=1)
    parts = tmp_path / "parts"
    parts.mkdir()
    idx = np.array_split(np.arange(len(v)), 3)
    for k, i in enumerate(idx):
        # Columns in another order in one file: CSV binds by name.
        p = write_csv(parts / f"part{k}.csv", v[i], x1[i], y[i])
        if k == 1:
            pd.read_csv(p, keep_default_na=False)[["c", "label", "x1"]].to_csv(p, index=False)
    m = fb.train(FB, str(parts), label="label", feature_types=TYPES, num_boost_round=4,
                 quiet=True)
    assert m.get_categories()[1] == first_seen(v)
    # Independent of the reader thread count (single-reader and parallel chunks).
    for threads in (1, 4):
        other = fb.train({**FB, "nthread": threads}, str(parts), label="label",
                         feature_types=TYPES, num_boost_round=4, quiet=True)
        assert model_json(other, tmp_path, "t") == model_json(m, tmp_path, "m")
    single = write_csv(tmp_path / "all.csv", v, x1, y)
    for threads in (1, 4):
        one = fb.train({**FB, "nthread": threads}, single, label="label", feature_types=TYPES,
                       num_boost_round=4, quiet=True)
        assert model_json(one, tmp_path, "one") == model_json(m, tmp_path, "m")
    df = frame(v, x1, m.get_categories()[1])
    np.testing.assert_array_equal(m.predict(str(parts), output_margin=True),
                                  xgb_margin(xgb_model(df, y), df))


def test_numeric_looking_categories_are_text_in_every_file(tmp_path):
    """One feature never mixes codes and names: if any training file holds text,
    the column is text everywhere, and later files are read as text whatever
    their own fields look like."""
    rng = np.random.default_rng(2)
    n = 1200
    nums = rng.choice(["1", "2", "7"], n).astype(object)
    text = rng.choice(["1", "x", "7", "yy"], n).astype(object)
    x1 = rng.integers(0, 4, 2 * n).astype(np.float32)
    v = np.concatenate([nums, text])
    y = (np.array([{"1": 0.0, "2": 1.0, "7": 2.0, "x": 3.0, "yy": -1.0}[s] for s in v])
         + 0.1 * x1).astype(np.float32)
    parts = tmp_path / "mixed"
    parts.mkdir()
    write_csv(parts / "a.csv", nums, x1[:n], y[:n])      # all integers
    write_csv(parts / "b.csv", text, x1[n:], y[n:])      # text
    m = fb.train(FB, str(parts), label="label", feature_types=TYPES, num_boost_round=3,
                 quiet=True)
    stored = m.get_categories()[1]
    assert stored == first_seen(v) and sorted(stored) == ["1", "2", "7", "x", "yy"]
    df = frame(v, x1, stored)
    want = xgb_margin(xgb_model(df, y, 3), df)
    np.testing.assert_array_equal(m.predict(str(parts), output_margin=True), want)
    # A validation / prediction file holding only "1", "2" and "7" (inferred as
    # integers) is still coded by value, not read as the codes 1, 2 and 7.
    a = str(parts / "a.csv")
    np.testing.assert_array_equal(m.predict(a, output_margin=True), want[:n])
    hist = {}
    fb.train(FB, str(parts), label="label", feature_types=TYPES, num_boost_round=3,
             quiet=True, evals=[(a, "v")], evals_result=hist)
    res = {}
    dval = xgb.DMatrix(frame(nums, x1[:n], stored), y[:n], enable_categorical=True)
    xgb.train(XGB, xgb.DMatrix(df, y, enable_categorical=True), 3, evals=[(dval, "v")],
              evals_result=res, verbose_eval=False)
    np.testing.assert_allclose(hist["v"]["rmse"], res["v"]["rmse"], rtol=1e-6)
    # All-integer categorical columns keep their values as codes (no names).
    codes = fb.train(FB, a, label="label", feature_types=TYPES, num_boost_round=2, quiet=True)
    assert codes.get_categories() is None


def test_csv_validation_set_and_unseen_categories(tmp_path):
    v, x1, y, cats = make(n=3000, seed=3)
    train = write_csv(tmp_path / "t.csv", v[:2000], x1[:2000], y[:2000])
    val = write_csv(tmp_path / "v.csv", v[2000:], x1[2000:], y[2000:])
    hist = {}
    m = fb.train(FB, train, label="label", feature_types=TYPES, num_boost_round=4, quiet=True,
                 evals=[(val, "v")], evals_result=hist)
    stored = m.get_categories()[1]
    df = frame(v[:2000], x1[:2000], stored)
    dval = xgb.DMatrix(frame(v[2000:], x1[2000:], stored), y[2000:], enable_categorical=True)
    res = {}
    xgb.train(XGB, xgb.DMatrix(df, y[:2000], enable_categorical=True), 4, evals=[(dval, "v")],
              evals_result=res, verbose_eval=False)
    np.testing.assert_allclose(hist["v"]["rmse"], res["v"]["rmse"], rtol=1e-6)
    # An unseen category raises ValueError naming the feature column and the value.
    bad = write_csv(tmp_path / "bad.csv", np.array(["c1", "zebra", None], object), x1[:3],
                    y[:3])
    with pytest.raises(ValueError, match=r'"c".*"zebra".*not in the model\'s categories'):
        m.predict(bad)
    with pytest.raises(ValueError, match=r'"c".*"zebra"'):
        fb.train(FB, train, label="label", feature_types=TYPES, num_boost_round=1, quiet=True,
                 evals=[(bad, "v")])
    with pytest.raises(ValueError, match=r"'c'.*'zebra'.*not in the training categories"):
        m.predict(frame(np.array(["c1", "zebra"], object), x1[:2], ["c1", "zebra"]))


def test_headerless_csv_text_categories(tmp_path):
    v, x1, y, cats = make(n=1500, seed=4)
    path = str(tmp_path / "h.csv")
    pd.DataFrame({"a": x1.astype(np.int64), "b": pd.Series(v, dtype=object), "y": y}).to_csv(
        path, index=False, header=False, na_rep="")
    m = fb.train(FB, path, label=2, header=False, feature_types=TYPES, num_boost_round=3,
                 quiet=True)
    stored = m.get_categories()[1]
    assert stored == first_seen(v)
    df = frame(v, x1, stored)
    np.testing.assert_array_equal(m.predict(df, output_margin=True),
                                  xgb_margin(xgb_model(df, y, 3), df))
    feats = str(tmp_path / "hf.csv")
    pd.DataFrame({"a": x1.astype(np.int64), "b": pd.Series(v, dtype=object)}).to_csv(
        feats, index=False, header=False, na_rep="")
    np.testing.assert_array_equal(m.predict(feats, header=False, output_margin=True),
                                  m.predict(df, output_margin=True))


def test_text_columns_need_a_categorical_declaration(tmp_path):
    v, x1, y, cats = make(n=300, seed=5)
    path = write_csv(tmp_path / "t.csv", v, x1, y)
    with pytest.raises(ValueError, match="only numeric columns are supported"):
        fb.train(FB, path, label="label", num_boost_round=1, quiet=True)
    with pytest.raises(ValueError, match="only numeric columns are supported"):
        fb.train(FB, path, label="label", feature_types=["c", "q"], num_boost_round=1,
                 quiet=True)
    # A model trained on integer codes cannot score text.
    codes = np.stack([x1, pd.Categorical(v, categories=cats).codes.astype(np.float32)], 1)
    codes[codes[:, 1] < 0, 1] = np.nan
    m = fb.train(FB, codes, label=y, feature_types=TYPES, num_boost_round=1, quiet=True)
    feats = str(tmp_path / "f.csv")
    pd.DataFrame({"x1": x1, "c": pd.Series(v, dtype=object)}).to_csv(
        feats, index=False, header=False, na_rep="")
    with pytest.raises(ValueError):
        m.predict(feats, header=False)


def test_continuation_from_csv_and_integer_categories(tmp_path):
    v, x1, y, cats = make(n=2000, seed=6)
    first = fb.train({**FB, "enable_categorical": True}, frame(v, x1, cats), label=y,
                     num_boost_round=2, quiet=True)
    v2, x2, y2, _ = make(n=2000, seed=7)
    path = write_csv(tmp_path / "c.csv", v2, x2, y2)
    cont = fb.train(FB, path, label="label", feature_types=TYPES, num_boost_round=2,
                    quiet=True, init_model=first)
    assert cont.get_categories() == first.get_categories()
    same = fb.train({**FB, "enable_categorical": True}, frame(v2, x2, sorted(cats)), label=y2,
                    num_boost_round=2, quiet=True, init_model=first)
    assert model_json(cont, tmp_path, "a") == model_json(same, tmp_path, "b")
    with pytest.raises(ValueError, match="brand-new"):
        fb.train(FB, write_csv(tmp_path / "n.csv", np.array(["c1", "brand-new"], object),
                               x2[:2], y2[:2]),
                 label="label", feature_types=TYPES, num_boost_round=1, quiet=True,
                 init_model=first)
    # Integer categories (a pandas int category model): CSV integers and plain
    # Parquet integers are category values, re-coded like an integer dictionary.
    rng = np.random.default_rng(8)
    levels = np.array([30, -2, 7, 1000, 5], dtype=np.int64)
    k = rng.choice(levels, 1500)
    yk = (k % 7 + rng.normal(0, 0.1, 1500)).astype(np.float32)
    ints = fb.train({**FB, "enable_categorical": True},
                    pd.DataFrame({"k": pd.Categorical(k, categories=levels)}), label=yk,
                    num_boost_round=3, quiet=True)
    want = ints.predict(pd.DataFrame({"k": pd.Categorical(k, categories=levels)}),
                        output_margin=True)
    kcsv = str(tmp_path / "k.csv")
    pd.DataFrame({"k": k}).to_csv(kcsv, index=False)
    np.testing.assert_array_equal(ints.predict(kcsv, output_margin=True), want)
    kpq = str(tmp_path / "k.parquet")
    pq.write_table(pa.table({"k": pa.array(k)}), kpq)
    np.testing.assert_array_equal(ints.predict(kpq, output_margin=True), want)
    pd.DataFrame({"k": [30, 31]}).to_csv(kcsv, index=False)
    with pytest.raises(ValueError, match=r'"k".*31'):
        ints.predict(kcsv)


def test_csv_category_limit(tmp_path):
    """Up to 65,534 categories per feature; one more is an error."""
    n = 65535
    names = np.array([f"k{i}" for i in range(n)], object)
    y = np.zeros(n, np.float32)
    path = write_csv(tmp_path / "many.csv", names, np.zeros(n, np.float32), y)
    with pytest.raises(ValueError, match="more than 65534 distinct categories"):
        fb.train(FB, path, label="label", feature_types=TYPES, num_boost_round=1, quiet=True)
    ok = write_csv(tmp_path / "ok.csv", names[:-1], np.zeros(n - 1, np.float32), y[:-1])
    m = fb.train({**FB, "max_depth": 1}, ok, label="label", feature_types=TYPES,
                 num_boost_round=1, quiet=True)
    assert len(m.get_categories()[1]) == 65534


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
def test_gpu_csv_training_and_prediction_match_the_cpu(tmp_path, k):
    v, x1, y, cats = make(n=6000, k=k, seed=9)
    path = write_csv(tmp_path / "g.csv", v, x1, y)
    cpu = fb.train(FB, path, label="label", feature_types=TYPES, num_boost_round=4, quiet=True)
    # Without cols_per_block the GPU plan is made after the first pass (CSV has
    # no row count before open); the learned categories decide the 16-bit width.
    gpu = fb.train({**FB, "device": "cuda", "gpu_math": "exact"}, path, label="label",
                   feature_types=TYPES, num_boost_round=4, quiet=True)
    assert model_json(gpu, tmp_path, "g") == model_json(cpu, tmp_path, "c")
    assert len(cpu.get_categories()[1]) == len(set(v) - {None})
    np.testing.assert_array_equal(cpu.predict(path, use_gpu=True, output_margin=True),
                                  cpu.predict(path, output_margin=True))
    df = frame(v, x1, sorted(cats))
    np.testing.assert_array_equal(cpu.predict(df, use_gpu=True, output_margin=True),
                                  cpu.predict(path, output_margin=True))


# ------------------------------------------------- past the type inference sample
def _late_text(n, late, seed):
    """Integer-looking fields for the first `late` rows, then text mixed in."""
    rng = np.random.default_rng(seed)
    v = rng.choice([str(i) for i in range(12)], n).astype(object)
    tail = np.arange(late, n)
    v[tail[rng.random(len(tail)) < 0.5]] = rng.choice(["a", "bb", "c c"], 1)[0]
    v[tail[rng.random(len(tail)) < 0.3]] = "zz"
    v[rng.random(n) < 0.03] = None
    x1 = rng.integers(0, 8, n).astype(np.float32)
    effect = {s: rng.normal() for s in set(v) - {None}}
    y = (np.array([effect.get(s, 0.0) for s in v]) + 0.3 * x1
         + rng.normal(0, 0.1, n)).astype(np.float32)
    return v, x1, y


def _parquet(path, v, x1, y):
    pq.write_table(pa.table({"x1": x1.astype(np.int64), "c": pa.array(v, pa.string()),
                             "label": y}), str(path), row_group_size=7000)
    return str(path)


@pytest.mark.parametrize("late", [10_050, 100_050])
def test_text_first_appearing_past_the_inference_sample(tmp_path, late):
    """The codes-or-names decision for a declared "c" column reads every field,
    not the 10,000-record inference sample."""
    n = late + 2000
    v, x1, y = _late_text(n, late, seed=late)
    path = write_csv(tmp_path / "late.csv", v, x1, y)
    m = fb.train(FB, path, label="label", feature_types=TYPES, num_boost_round=3, quiet=True)
    assert m.get_categories()[1] == first_seen(v)
    want = model_json(m, tmp_path, "m")
    for threads in (1, 4):
        other = fb.train({**FB, "nthread": threads}, path, label="label",
                         feature_types=TYPES, num_boost_round=3, quiet=True)
        assert model_json(other, tmp_path, "t") == want
    # The same rows as a Parquet string column train the same model.
    from_pq = fb.train(FB, _parquet(tmp_path / "late.parquet", v, x1, y), label="label",
                       feature_types=TYPES, num_boost_round=3, quiet=True)
    assert model_json(from_pq, tmp_path, "pq") == want
    # Text only in the second of three files, past its own sample.
    parts = tmp_path / "parts"
    parts.mkdir()
    head, _, _ = _late_text(3000, 3000, seed=1)  # integers only
    hx, hy = x1[:3000] + 1, y[:3000] - 1
    write_csv(parts / "a.csv", head, hx, hy)
    write_csv(parts / "b.csv", v, x1, y)
    write_csv(parts / "c.csv", head, hx, hy)
    allv = np.concatenate([head, v, head])
    allx, ally = np.concatenate([hx, x1, hx]), np.concatenate([hy, y, hy])
    multi = fb.train(FB, str(parts), label="label", feature_types=TYPES, num_boost_round=3,
                     quiet=True)
    assert multi.get_categories()[1] == first_seen(allv)
    single = fb.train(FB, _parquet(tmp_path / "all.parquet", allv, allx, ally), label="label",
                      feature_types=TYPES, num_boost_round=3, quiet=True)
    assert model_json(multi, tmp_path, "multi") == model_json(single, tmp_path, "single")
    np.testing.assert_array_equal(multi.predict(str(parts), output_margin=True),
                                  single.predict(_parquet(tmp_path / "p.parquet", allv, allx,
                                                          ally), output_margin=True))


def test_all_integer_column_past_the_sample_stays_codes(tmp_path):
    """Integers in every field keep their values as the codes (no names), larger
    codes past the sample included, as from a Parquet integer column."""
    rng = np.random.default_rng(11)
    n = 30_000
    k = rng.integers(0, 20, n).astype(object)
    k[20_000:][rng.random(n - 20_000) < 0.2] = 300
    k[rng.random(n) < 0.03] = None
    x1 = rng.integers(0, 8, n).astype(np.float32)
    y = (np.array([0.0 if c is None else (c % 5) for c in k]) + 0.2 * x1).astype(np.float32)
    path = write_csv(tmp_path / "ints.csv", k, x1, y)
    m = fb.train(FB, path, label="label", feature_types=TYPES, num_boost_round=3, quiet=True)
    assert m.get_categories() is None
    kp = str(tmp_path / "ints.parquet")
    pq.write_table(pa.table({"x1": x1.astype(np.int64), "c": pa.array(list(k), pa.int64()),
                             "label": y}), kp)
    from_pq = fb.train(FB, kp, label="label", feature_types=TYPES, num_boost_round=3, quiet=True)
    assert model_json(m, tmp_path, "csv") == model_json(from_pq, tmp_path, "pq")


def test_unnamed_codes_with_a_bad_field_past_the_sample(tmp_path):
    """A model without names reads its categorical CSV columns as codes; a field
    that is not a number past the inference sample names the file, the column,
    the value and its row instead of failing inside the CSV parser."""
    rng = np.random.default_rng(12)
    n = 20_000
    k = rng.integers(0, 9, n)
    x1 = rng.integers(0, 8, n).astype(np.float32)
    y = (k % 4 + 0.2 * x1).astype(np.float32)
    train = write_csv(tmp_path / "t.csv", k.astype(object), x1, y)
    m = fb.train(FB, train, label="label", feature_types=TYPES, num_boost_round=3, quiet=True)
    assert m.get_categories() is None
    bad_k = k.astype(object)
    bad_k[15_000] = "oops"
    bad = write_csv(tmp_path / "bad.csv", bad_k, x1, y)
    pattern = r'bad\.csv.*"c".*"oops".*data row 15001.*integer category codes'
    with pytest.raises(ValueError, match=pattern):
        m.predict(bad)
    with pytest.raises(ValueError, match=pattern):
        fb.train(FB, train, label="label", feature_types=TYPES, num_boost_round=1, quiet=True,
                 evals=[(bad, "v")])
    # A fractional code past the sample is its integer part, as everywhere at
    # prediction time.
    frac_k = k.astype(object)
    frac_k[15_000] = f"{k[15_000]}.5"
    frac = write_csv(tmp_path / "frac.csv", frac_k, x1)
    np.testing.assert_array_equal(m.predict(frac, output_margin=True),
                                  m.predict(train, output_margin=True))
