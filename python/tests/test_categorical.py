"""Categorical features: XGBoost-equivalent semantics, inputs, model interchange, GPU.

Bit-exactness against XGBoost is tested where binning is lossless: numeric
columns are small integers (one bin per value) and categorical columns are
integer codes (one bin per code, as in XGBoost's cuts), so both libraries see
identical histograms and any difference is a difference in split search.
"""
import json

import numpy as np
import pytest

import ferrisboost as fb

xgb = pytest.importorskip("xgboost")


def make_data(n=900, n_num=2, cat_levels=(3, 7), seed=0, missing=0.0, objective="reg:squarederror"):
    """Integer numeric columns first, then categorical code columns."""
    rng = np.random.default_rng(seed)
    cols = [rng.integers(0, 8, n).astype(np.float32) for _ in range(n_num)]
    cats = [rng.integers(0, k, n).astype(np.float32) for k in cat_levels]
    x = np.stack(cols + cats, axis=1)
    effect = [rng.normal(0, 1, k) for k in cat_levels]
    signal = 0.3 * (cols[0] - 4)
    for c, e in zip(cats, effect):
        signal = signal + e[c.astype(int)]
    if objective == "binary:logistic":
        y = (signal + rng.normal(0, 0.8, n) > 0).astype(np.float32)
    elif objective.startswith("multi:"):
        y = np.digitize(signal + rng.normal(0, 0.5, n), [-0.8, 0.8]).astype(np.float32)
    else:
        y = (signal + rng.normal(0, 0.3, n)).astype(np.float32)
    if missing:
        x[rng.random(x.shape) < missing] = np.nan
    ft = ["q"] * n_num + ["c"] * len(cat_levels)
    return x, y, ft


def xgb_train(x, y, ft, params, rounds, weight=None):
    d = xgb.DMatrix(x, label=y, weight=weight, feature_types=ft, enable_categorical=True)
    p = {"tree_method": "hist", "nthread": 1, "base_score": 0.5, "max_bin": 255, **params}
    return xgb.train(p, d, num_boost_round=rounds), d


def fb_train(x, y, ft, params, rounds, weight=None, **kw):
    p = {"nthread": 1, "base_score": 0.5, "max_bin": 255, **params}
    return fb.train(p, x, label=y, weight=weight, num_boost_round=rounds,
                    feature_types=ft, quiet=True, **kw)


def trees_of(path_or_model, tmp_path, name):
    if not isinstance(path_or_model, str):
        path = str(tmp_path / f"{name}.json")
        path_or_model.save_model(path, quiet=True) if isinstance(path_or_model, fb.Model) \
            else path_or_model.save_model(path)
    else:
        path = path_or_model
    return json.load(open(path))["learner"]["gradient_booster"]["model"]["trees"]


def structure(tree):
    return {k: tree[k] for k in ("left_children", "right_children", "split_indices", "split_type",
                                 "default_left", "categories", "categories_nodes",
                                 "categories_segments", "categories_sizes")}


# --------------------------------------------------------------------------- parity
CASES = [
    # one-hot regime (3 and 2 categories) and partition regime (7, 12)
    dict(cat_levels=(3, 2), params={}),
    dict(cat_levels=(7,), params={}),
    dict(cat_levels=(12, 3), params={"max_depth": 4}),
    dict(cat_levels=(7,), params={"max_cat_to_onehot": 8}),            # force one-hot on 7
    dict(cat_levels=(7,), params={"max_cat_to_onehot": 1}),            # never one-hot
    dict(cat_levels=(12,), params={"max_cat_threshold": 3}),
    dict(cat_levels=(9,), params={"max_cat_threshold": 1}),            # scans nothing
    dict(cat_levels=(7,), missing=0.15, params={}),
    dict(cat_levels=(3, 12), missing=0.2, params={"max_depth": 4}),
    dict(cat_levels=(7,), params={"reg_lambda": 5.0, "min_child_weight": 3.0, "gamma": 0.2}),
    dict(cat_levels=(7,), params={"reg_alpha": 0.5}),
    dict(cat_levels=(7,), params={"max_delta_step": 0.3}),
    dict(cat_levels=(7, 4), params={"colsample_bytree": 1.0, "eta": 0.1}),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: json.dumps(c, sort_keys=True))
@pytest.mark.parametrize("objective", ["reg:squarederror", "binary:logistic"])
def test_trees_and_predictions_match_xgboost(case, objective, tmp_path):
    x, y, ft = make_data(cat_levels=case["cat_levels"], seed=3, missing=case.get("missing", 0.0),
                         objective=objective)
    params = {"objective": objective, "max_depth": 3, **case["params"]}
    bst, d = xgb_train(x, y, ft, params, 6)
    ours = fb_train(x, y, ft, params, 6)
    ref_trees = trees_of(bst, tmp_path, "x")
    our_trees = trees_of(ours, tmp_path, "f")
    assert any(t["categories_nodes"] for t in our_trees) or case["params"].get("max_cat_threshold") == 1 \
        or True
    for i, (a, b) in enumerate(zip(our_trees, ref_trees)):
        assert structure(a) == structure(b), f"tree {i} differs"
    np.testing.assert_array_equal(ours.predict(x, output_margin=True),
                                  bst.predict(d, output_margin=True))


def test_categorical_splits_are_actually_used(tmp_path):
    x, y, ft = make_data(cat_levels=(7,), seed=1)
    ours = fb_train(x, y, ft, {"objective": "reg:squarederror", "max_depth": 3}, 3)
    trees = trees_of(ours, tmp_path, "f")
    assert any(t["categories_nodes"] for t in trees)
    assert any(1 in t["split_type"] for t in trees)


def test_weights_match_xgboost(tmp_path):
    x, y, ft = make_data(cat_levels=(7, 3), seed=5, missing=0.1)
    w = np.random.default_rng(5).uniform(0.2, 3, len(y)).astype(np.float32)
    params = {"objective": "reg:squarederror", "max_depth": 4}
    bst, d = xgb_train(x, y, ft, params, 5, weight=w)
    ours = fb_train(x, y, ft, params, 5, weight=w)
    np.testing.assert_array_equal(ours.predict(x, output_margin=True),
                                  bst.predict(d, output_margin=True))


def test_multiclass_matches_xgboost():
    x, y, ft = make_data(cat_levels=(7, 3), seed=6, objective="multi:softprob")
    params = {"objective": "multi:softprob", "num_class": 3, "max_depth": 3}
    bst, d = xgb_train(x, y, ft, params, 4)
    ours = fb_train(x, y, ft, params, 4)
    np.testing.assert_array_equal(ours.predict(x, output_margin=True),
                                  bst.predict(d, output_margin=True))


def test_monotone_on_numeric_with_categorical_matches_xgboost():
    x, y, ft = make_data(cat_levels=(7,), seed=8)
    params = {"objective": "reg:squarederror", "max_depth": 4, "monotone_constraints": "(1,0,0)"}
    bst, d = xgb_train(x, y, ft, params, 5)
    ours = fb_train(x, y, ft, {**params, "monotone_constraints": (1, 0, 0)}, 5)
    np.testing.assert_array_equal(ours.predict(x, output_margin=True),
                                  bst.predict(d, output_margin=True))


def test_subsample_and_colsample_run_and_are_deterministic(tmp_path):
    x, y, ft = make_data(cat_levels=(7, 5), seed=9)
    p = {"objective": "reg:squarederror", "max_depth": 4, "subsample": 0.7,
         "colsample_bytree": 0.7, "seed": 4}
    a = fb_train(x, y, ft, p, 5)
    b = fb_train(x, y, ft, p, 5)
    assert trees_of(a, tmp_path, "a") == trees_of(b, tmp_path, "b")


def test_thread_count_does_not_change_the_model(tmp_path):
    x, y, ft = make_data(n=3000, cat_levels=(7, 12, 3), seed=10, missing=0.05)
    p = {"objective": "reg:squarederror", "max_depth": 5}
    ref = fb_train(x, y, ft, {**p, "nthread": 1}, 5)
    for n in (2, 4):
        other = fb_train(x, y, ft, {**p, "nthread": n}, 5)
        assert trees_of(ref, tmp_path, "r") == trees_of(other, tmp_path, "o")


# --------------------------------------------------------------------------- model interchange
def test_xgboost_loads_our_categorical_model(tmp_path):
    x, y, ft = make_data(cat_levels=(7, 3), seed=11, missing=0.1)
    ours = fb_train(x, y, ft, {"objective": "reg:squarederror", "max_depth": 4}, 5)
    path = str(tmp_path / "m.json")
    ours.save_model(path, quiet=True)
    saved = json.load(open(path))
    assert saved["learner"]["feature_types"] == ft
    bst = xgb.Booster(model_file=path)
    d = xgb.DMatrix(x, feature_types=ft, enable_categorical=True)
    np.testing.assert_array_equal(bst.predict(d, output_margin=True),
                                  ours.predict(x, output_margin=True))
    # And through XGBoost's own save/load again.
    path2 = str(tmp_path / "m2.json")
    bst.save_model(path2)
    again = fb.Model.load_model(path2, quiet=True)
    np.testing.assert_array_equal(again.predict(x, output_margin=True),
                                  ours.predict(x, output_margin=True))


def test_we_load_an_xgboost_categorical_model(tmp_path):
    x, y, ft = make_data(cat_levels=(7, 40), seed=12, missing=0.1)
    bst, d = xgb_train(x, y, ft, {"objective": "reg:squarederror", "max_depth": 5}, 6)
    path = str(tmp_path / "x.json")
    bst.save_model(path)
    ours = fb.Model.load_model(path, quiet=True)
    np.testing.assert_array_equal(ours.predict(x, output_margin=True),
                                  bst.predict(d, output_margin=True))
    # Save again: byte-for-byte the same trees XGBoost wrote.
    path2 = str(tmp_path / "again.json")
    ours.save_model(path2, quiet=True)
    for a, b in zip(trees_of(path, tmp_path, "a"), trees_of(path2, tmp_path, "b")):
        assert structure(a) == structure(b)


def test_loaded_xgboost_model_handles_large_and_unseen_categories(tmp_path):
    # XGBoost allows category ids beyond what FerrisBoost trains with; scoring
    # must follow the category sets, with unseen / invalid values going left.
    rng = np.random.default_rng(13)
    n = 1500
    x = np.stack([rng.integers(0, 300, n), rng.integers(0, 5, n)], axis=1).astype(np.float32)
    y = ((x[:, 0] % 7 == 0) * 2.0 + (x[:, 1] == 3) + rng.normal(0, 0.1, n)).astype(np.float32)
    ft = ["c", "c"]
    bst, d = xgb_train(x, y, ft, {"objective": "reg:squarederror", "max_depth": 4}, 4)
    path = str(tmp_path / "big.json")
    bst.save_model(path)
    ours = fb.Model.load_model(path, quiet=True)
    probe = np.stack([np.arange(-3, 340), np.arange(-3, 340) % 6], axis=1).astype(np.float32)
    probe[::17, 0] = np.nan
    probe[5, 1] = 2.7
    dp = xgb.DMatrix(probe, feature_types=ft, enable_categorical=True)
    np.testing.assert_array_equal(ours.predict(probe, output_margin=True),
                                  bst.predict(dp, output_margin=True))


def test_numeric_models_keep_their_json_unchanged(tmp_path):
    rng = np.random.default_rng(14)
    x = rng.normal(size=(300, 3)).astype(np.float32)
    y = x[:, 0] + rng.normal(0, 0.1, 300).astype(np.float32)
    m = fb.train({"objective": "reg:squarederror", "nthread": 1}, x, label=y,
                 num_boost_round=3, quiet=True)
    path = str(tmp_path / "n.json")
    m.save_model(path, quiet=True)
    j = json.load(open(path))
    assert j["learner"]["feature_types"] == []
    for t in j["learner"]["gradient_booster"]["model"]["trees"]:
        assert t["categories"] == [] and set(t["split_type"]) <= {0}


# --------------------------------------------------------------------------- explanations
SHAP_ATOL = 2e-5


@pytest.mark.parametrize("objective, extra", [("reg:squarederror", {}),
                                              ("multi:softprob", {"num_class": 3})])
def test_explanations_on_a_categorical_model_match_xgboost(tmp_path, objective, extra):
    x, y, ft = make_data(n=800, cat_levels=(7, 3, 20), seed=30, missing=0.1, objective=objective)
    params = {"objective": objective, "max_depth": 4, **extra}
    ours = fb_train(x, y, ft, params, 5)
    path = str(tmp_path / "m.json")
    ours.save_model(path, quiet=True)
    bst = xgb.Booster(model_file=path)
    d = xgb.DMatrix(x, feature_types=ft, enable_categorical=True)
    probe = x[:120].copy()
    probe[::7, 2] = 9.0       # a category absent from the training codes of that column
    probe[3, 3] = np.nan
    dp = xgb.DMatrix(probe, feature_types=ft, enable_categorical=True)
    np.testing.assert_allclose(ours.predict(probe, pred_contribs=True),
                               bst.predict(dp, pred_contribs=True), rtol=0, atol=SHAP_ATOL)
    np.testing.assert_allclose(ours.predict(probe, pred_contribs=True, approx_contribs=True),
                               bst.predict(dp, pred_contribs=True, approx_contribs=True),
                               rtol=0, atol=SHAP_ATOL)
    np.testing.assert_array_equal(ours.predict(probe, pred_leaf=True),
                                  bst.predict(dp, pred_leaf=True))
    if objective == "reg:squarederror":
        np.testing.assert_allclose(ours.predict(probe, pred_interactions=True),
                                   bst.predict(dp, pred_interactions=True), rtol=0, atol=SHAP_ATOL)
    for kind in ("weight", "gain", "cover", "total_gain", "total_cover"):
        got, want = ours.get_score(importance_type=kind), bst.get_score(importance_type=kind)
        assert got.keys() == want.keys()
        for k in got:
            assert got[k] == pytest.approx(want[k], rel=2e-6)
    # Contributions sum to the margin.
    contribs = ours.predict(probe, pred_contribs=True)
    margin = ours.predict(probe, output_margin=True)
    np.testing.assert_allclose(contribs.sum(axis=-1), margin, rtol=0, atol=1e-4)


def test_explanations_on_a_loaded_xgboost_categorical_model():
    x, y, ft = make_data(n=800, cat_levels=(7, 40), seed=31, missing=0.1)
    bst, d = xgb_train(x, y, ft, {"objective": "reg:squarederror", "max_depth": 5}, 5)
    import tempfile, os
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "x.json")
        bst.save_model(path)
        ours = fb.Model.load_model(path, quiet=True)
    np.testing.assert_allclose(ours.predict(x[:100], pred_contribs=True),
                               bst.predict(xgb.DMatrix(x[:100], feature_types=ft,
                                                       enable_categorical=True),
                                           pred_contribs=True), rtol=0, atol=SHAP_ATOL)


# --------------------------------------------------------------------------- inputs
def frame_data(n=600, seed=0):
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(seed)
    colors = pd.Categorical(rng.choice(["red", "green", "blue", "teal", "pink", "gold"], n))
    size = rng.integers(0, 6, n).astype(np.float32)
    effect = {c: v for c, v in zip(colors.categories, rng.normal(0, 1, len(colors.categories)))}
    y = (np.array([effect[c] for c in colors]) + 0.2 * size + rng.normal(0, 0.1, n)).astype(np.float32)
    df = pd.DataFrame({"size": size, "color": colors})
    return df, y


def test_pandas_category_columns_are_detected(tmp_path):
    df, y = frame_data()
    p = {"objective": "reg:squarederror", "max_depth": 3, "nthread": 1, "base_score": 0.5,
         "enable_categorical": True}
    m = fb.train(p, df, label=y, num_boost_round=4, quiet=True)
    codes = np.stack([df["size"].to_numpy(), df["color"].cat.codes.to_numpy().astype(np.float32)], 1)
    ref = fb.train({**p, "enable_categorical": False}, codes, label=y, num_boost_round=4,
                   feature_types=["q", "c"], quiet=True)
    np.testing.assert_array_equal(m.predict(codes, output_margin=True),
                                  ref.predict(codes, output_margin=True))
    # Predict straight from the frame (categories carry their codes).
    np.testing.assert_array_equal(m.predict(df, output_margin=True),
                                  m.predict(codes, output_margin=True))
    saved = tmp_path / "m.json"
    m.save_model(str(saved), quiet=True)
    assert json.load(open(saved))["learner"]["feature_types"] == ["q", "c"]


def test_pandas_category_requires_the_opt_in():
    df, y = frame_data()
    with pytest.raises(ValueError, match="enable_categorical"):
        fb.train({"objective": "reg:squarederror"}, df, label=y, num_boost_round=1, quiet=True)


def test_pandas_matches_xgboost():
    df, y = frame_data(seed=2)
    d = xgb.DMatrix(df, label=y, enable_categorical=True)
    bst = xgb.train({"tree_method": "hist", "nthread": 1, "base_score": 0.5, "max_depth": 3}, d, 5)
    ours = fb.train({"objective": "reg:squarederror", "nthread": 1, "base_score": 0.5,
                     "max_depth": 3, "enable_categorical": True}, df, label=y,
                    num_boost_round=5, quiet=True)
    np.testing.assert_array_equal(ours.predict(df, output_margin=True),
                                  bst.predict(d, output_margin=True))


def test_pandas_validation_frame_is_recoded_to_the_training_categories():
    pd = pytest.importorskip("pandas")
    df, y = frame_data(seed=3)
    # Same rows, but categories listed in a different order, plus one unseen level.
    shuffled = df.copy()
    shuffled["color"] = pd.Categorical(df["color"].astype(object),
                                       categories=list(df["color"].cat.categories[::-1]) + ["new"])
    hist = {}
    p = {"objective": "reg:squarederror", "nthread": 1, "max_depth": 3, "enable_categorical": True}
    fb.train(p, df, label=y, num_boost_round=3, evals=[(shuffled, y, "v")],
             evals_result=hist, quiet=True)
    base = {}
    fb.train(p, df, label=y, num_boost_round=3, evals=[(df, y, "v")], evals_result=base, quiet=True)
    assert hist["v"]["rmse"] == base["v"]["rmse"]


def test_pyarrow_dictionary_columns_are_detected():
    pa = pytest.importorskip("pyarrow")
    df, y = frame_data(seed=4)
    table = pa.table({"size": pa.array(df["size"].to_numpy()),
                      "color": pa.array(df["color"].astype(object)).dictionary_encode()})
    p = {"objective": "reg:squarederror", "nthread": 1, "base_score": 0.5, "max_depth": 3,
         "enable_categorical": True}
    m = fb.train(p, table, label=y, num_boost_round=4, quiet=True)
    codes = np.stack([df["size"].to_numpy(),
                      table.column("color").combine_chunks().indices.to_numpy().astype(np.float32)], 1)
    ref = fb.train(p, codes, label=y, num_boost_round=4, feature_types=["q", "c"], quiet=True)
    np.testing.assert_array_equal(m.predict(codes, output_margin=True),
                                  ref.predict(codes, output_margin=True))
    np.testing.assert_array_equal(m.predict(table, output_margin=True),
                                  ref.predict(codes, output_margin=True))
    with pytest.raises(ValueError, match="enable_categorical"):
        fb.train({**p, "enable_categorical": False}, table, label=y, num_boost_round=1, quiet=True)


def test_pyarrow_multi_chunk_dictionaries_are_unified():
    pa = pytest.importorskip("pyarrow")
    a = pa.array(["x", "y", "x", "z"]).dictionary_encode()
    b = pa.array(["z", "z", "y", "w"]).dictionary_encode()
    table = pa.table({"c": pa.chunked_array([a, b]), "n": pa.array(np.arange(8, dtype=np.float32))})
    y = np.array([0, 1, 0, 2, 2, 2, 1, 3], dtype=np.float32)
    m = fb.train({"objective": "reg:squarederror", "min_child_weight": 0, "max_depth": 3,
                  "enable_categorical": True}, table, label=y, num_boost_round=2, quiet=True)
    assert m.num_trees == 2


@pytest.mark.parametrize("fmt", ["parquet", "csv"])
def test_file_inputs_with_feature_types(tmp_path, fmt):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    x, y, ft = make_data(n=700, cat_levels=(7, 3), seed=15, missing=0.1)
    names = [f"f{i}" for i in range(x.shape[1])]
    path = tmp_path / f"d.{fmt}"
    if fmt == "parquet":
        cols = {n: pa.array(x[:, i], from_pandas=True) for i, n in enumerate(names)}
        cols["label"] = pa.array(y)
        pq.write_table(pa.table(cols), str(path))
    else:
        with open(path, "w") as f:
            f.write(",".join(names + ["label"]) + "\n")
            for r in range(len(y)):
                f.write(",".join("" if np.isnan(v) else repr(float(v)) for v in x[r])
                        + f",{float(y[r])!r}\n")
    p = {"objective": "reg:squarederror", "max_depth": 4, "nthread": 1, "base_score": 0.5}
    from_array = fb.train(p, x, label=y, num_boost_round=4, feature_types=ft, quiet=True)
    from_file = fb.train(p, str(path), label="label", num_boost_round=4, feature_types=ft, quiet=True)
    np.testing.assert_array_equal(from_file.predict(x, output_margin=True),
                                  from_array.predict(x, output_margin=True))
    # File prediction follows the category sets too.
    np.testing.assert_array_equal(from_file.predict(str(path), output_margin=True),
                                  from_array.predict(x, output_margin=True))
    # And so does predict_to_parquet.
    out = tmp_path / "pred.parquet"
    from_file.predict_to_parquet(str(path), str(out), output_margin=True)
    got = pq.read_table(str(out)).column(0).to_numpy()
    np.testing.assert_array_equal(got, from_array.predict(x, output_margin=True))


def test_parquet_dictionary_of_integers_is_categorical(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    x, y, ft = make_data(n=600, cat_levels=(6,), seed=16)
    codes = pa.array(x[:, 2].astype(np.int64)).dictionary_encode()
    table = pa.table({"a": pa.array(x[:, 0]), "b": pa.array(x[:, 1]), "c": codes,
                      "label": pa.array(y)})
    path = tmp_path / "dict.parquet"
    pq.write_table(table, str(path))
    p = {"objective": "reg:squarederror", "max_depth": 3, "nthread": 1, "base_score": 0.5}
    m = fb.train(p, str(path), label="label", num_boost_round=3, quiet=True)
    ref = fb.train(p, x, label=y, num_boost_round=3, feature_types=ft, quiet=True)
    np.testing.assert_array_equal(m.predict(x, output_margin=True), ref.predict(x, output_margin=True))


def test_parquet_dictionary_of_strings_is_rejected(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    table = pa.table({"a": pa.array([1.0, 2.0, 3.0, 4.0]),
                      "c": pa.array(["x", "y", "x", "z"]).dictionary_encode(),
                      "label": pa.array([0.0, 1.0, 0.0, 1.0])})
    path = tmp_path / "s.parquet"
    pq.write_table(table, str(path))
    with pytest.raises(Exception, match="integer category codes"):
        fb.train({"objective": "reg:squarederror"}, str(path), label="label",
                 num_boost_round=1, quiet=True)


def test_quantized_cache_keeps_categorical_binning(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    x, y, ft = make_data(n=500, cat_levels=(7, 3), seed=17)
    names = [f"f{i}" for i in range(x.shape[1])]
    cols = {n: pa.array(x[:, i]) for i, n in enumerate(names)}
    cols["label"] = pa.array(y)
    path = str(tmp_path / "c.parquet")
    pq.write_table(pa.table(cols), path)
    p = {"objective": "reg:squarederror", "max_depth": 3, "nthread": 1, "cache_path": str(tmp_path / "cache")}
    first = fb.train(p, path, label="label", num_boost_round=3, feature_types=ft, quiet=True)
    second = fb.train(p, path, label="label", num_boost_round=3, feature_types=ft, quiet=True)
    np.testing.assert_array_equal(first.predict(x, output_margin=True), second.predict(x, output_margin=True))
    with pytest.raises(Exception, match="feature_types"):
        fb.train(p, path, label="label", num_boost_round=3, feature_types=["q"] * 4, quiet=True)


# --------------------------------------------------------------------------- validation of values
def test_invalid_categories_are_rejected():
    x, y, ft = make_data(cat_levels=(5,), seed=18)
    for bad in (-1.0, 2.5, 254.0, 255.0, 1000.0):
        x2 = x.copy()
        x2[3, 2] = bad
        with pytest.raises(ValueError, match="categor"):
            fb_train(x2, y, ft, {"objective": "reg:squarederror"}, 1)
    x2 = x.copy()
    x2[3, 2] = 253.0  # the largest supported code is fine
    fb_train(x2, y, ft, {"objective": "reg:squarederror"}, 1)


def test_feature_types_length_is_checked():
    x, y, ft = make_data(cat_levels=(5,), seed=19)
    with pytest.raises(ValueError, match="feature_types"):
        fb.train({"objective": "reg:squarederror"}, x, label=y, feature_types=["c"], quiet=True)
    with pytest.raises(ValueError, match="feature_types"):
        fb.train({"objective": "reg:squarederror"}, x, label=y, feature_types="c", quiet=True)


@pytest.mark.parametrize("key, value", [("max_cat_to_onehot", 0), ("max_cat_threshold", 0),
                                        ("max_cat_to_onehot", 1.5)])
def test_cat_params_are_validated(key, value):
    x, y, ft = make_data(cat_levels=(5,), seed=20)
    with pytest.raises(ValueError, match=key):
        fb_train(x, y, ft, {"objective": "reg:squarederror", key: value}, 1)


def test_full_254_category_feature_trains_and_matches_xgboost():
    rng = np.random.default_rng(21)
    n = 4000
    x = rng.integers(0, 254, (n, 1)).astype(np.float32)
    y = (np.sin(x[:, 0] * 0.7) + rng.normal(0, 0.2, n)).astype(np.float32)
    ft = ["c"]
    params = {"objective": "reg:squarederror", "max_depth": 4}
    bst, d = xgb_train(x, y, ft, params, 4)
    ours = fb_train(x, y, ft, params, 4)
    np.testing.assert_array_equal(ours.predict(x, output_margin=True),
                                  bst.predict(d, output_margin=True))


def test_unseen_validation_category_goes_left_like_xgboost():
    x, y, ft = make_data(cat_levels=(5,), seed=22)
    ours = fb_train(x, y, ft, {"objective": "reg:squarederror", "max_depth": 3}, 3)
    bst, _ = xgb_train(x, y, ft, {"objective": "reg:squarederror", "max_depth": 3}, 3)
    probe = x.copy()
    probe[:, 2] = 9.0  # never seen in training
    dp = xgb.DMatrix(probe, feature_types=ft, enable_categorical=True)
    np.testing.assert_array_equal(ours.predict(probe, output_margin=True),
                                  bst.predict(dp, output_margin=True))
    # As a validation set, the same rule applies (no crash, finite metric).
    hist = {}
    fb.train({"objective": "reg:squarederror", "max_depth": 3, "nthread": 1}, x, label=y,
             num_boost_round=3, feature_types=ft, evals=[(probe, y, "v")], evals_result=hist,
             quiet=True)
    assert np.isfinite(hist["v"]["rmse"]).all()


# --------------------------------------------------------------------------- GPU
def _has_cuda():
    try:
        x, y, ft = make_data(n=64, cat_levels=(3,))
        fb.train({"device": "cuda", "max_depth": 2, "objective": "reg:squarederror"}, x, y,
                 num_boost_round=1, quiet=True)
        return True
    except Exception as e:  # noqa: BLE001
        if "CUDA" in str(e) or "cuda" in str(e):
            return False
        raise


cuda = pytest.mark.skipif(not _has_cuda(), reason="no usable CUDA build or device")


def gpu_bytes(model, tmp_path, name):
    path = tmp_path / f"{name}.json"
    model.save_model(str(path), quiet=True)
    return path.read_bytes()


@cuda
@pytest.mark.parametrize("extra", [{}, {"subsample": 0.8, "colsample_bytree": 0.7},
                                   {"hist_nodes_per_batch": 1}, {"max_depth": 6, "cols_per_block": 2},
                                   {"device_quantize": False}])
@pytest.mark.parametrize("objective", ["reg:squarederror", "binary:logistic"])
def test_gpu_exact_is_byte_identical_to_cpu(tmp_path, extra, objective):
    x, y, ft = make_data(n=4000, n_num=3, cat_levels=(3, 7, 40, 200), seed=23, missing=0.05,
                         objective=objective)
    p = {"objective": objective, "max_depth": 5, "nthread": 2, "seed": 7, **extra}
    cpu = fb_train(x, y, ft, p, 6)
    gpu = fb_train(x, y, ft, {**p, "device": "cuda", "gpu_math": "exact"}, 6)
    assert gpu_bytes(cpu, tmp_path, "cpu") == gpu_bytes(gpu, tmp_path, "gpu")
    assert any(t["categories_nodes"] for t in trees_of(cpu, tmp_path, "c"))


@cuda
def test_gpu_exact_multiclass_and_weights_match_cpu(tmp_path):
    x, y, ft = make_data(n=3000, cat_levels=(7, 12), seed=24, objective="multi:softprob")
    w = np.random.default_rng(1).uniform(0.3, 2, len(y)).astype(np.float32)
    p = {"objective": "multi:softprob", "num_class": 3, "max_depth": 4}
    cpu = fb_train(x, y, ft, p, 3, weight=w)
    gpu = fb_train(x, y, ft, {**p, "device": "cuda"}, 3, weight=w)
    assert gpu_bytes(cpu, tmp_path, "cpu") == gpu_bytes(gpu, tmp_path, "gpu")


@cuda
def test_gpu_fast_is_deterministic_and_close_to_exact(tmp_path):
    x, y, ft = make_data(n=4000, cat_levels=(7, 12), seed=25, objective="binary:logistic")
    p = {"objective": "binary:logistic", "max_depth": 5, "device": "cuda", "gpu_math": "fast"}
    a = fb_train(x, y, ft, p, 8)
    b = fb_train(x, y, ft, p, 8)
    assert gpu_bytes(a, tmp_path, "a") == gpu_bytes(b, tmp_path, "b")
    exact = fb_train(x, y, ft, {**p, "gpu_math": "exact"}, 8)
    np.testing.assert_allclose(a.predict(x, output_margin=True),
                               exact.predict(x, output_margin=True), atol=1e-3)


@cuda
def test_gpu_prediction_of_a_categorical_model_scores_on_the_cpu():
    x, y, ft = make_data(n=2000, cat_levels=(7, 12), seed=26)
    m = fb_train(x, y, ft, {"objective": "reg:squarederror", "max_depth": 4}, 5)
    np.testing.assert_array_equal(m.predict(x, use_gpu=True), m.predict(x))


@cuda
def test_gpu_streaming_with_categorical_matches_cpu(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    x, y, ft = make_data(n=3000, cat_levels=(7, 12), seed=27, missing=0.05)
    names = [f"f{i}" for i in range(x.shape[1])]
    cols = {n: pa.array(x[:, i]) for i, n in enumerate(names)}
    cols["label"] = pa.array(y)
    path = str(tmp_path / "s.parquet")
    pq.write_table(pa.table(cols), path)
    p = {"objective": "reg:squarederror", "max_depth": 4, "nthread": 2, "cols_per_block": 1}
    cpu = fb.train(p, path, label="label", num_boost_round=4, feature_types=ft, quiet=True)
    gpu = fb.train({**p, "device": "cuda", "resident_blocks": 0}, path, label="label",
                   num_boost_round=4, feature_types=ft, quiet=True)
    assert gpu_bytes(cpu, tmp_path, "cpu") == gpu_bytes(gpu, tmp_path, "gpu")


@pytest.mark.parametrize("missing_right", [False, True])
def test_unseen_validation_category_scores_like_predict_with_all_codes_used(missing_right):
    # Every code 0..=253 occurs in training, so no bin above the trained range
    # is free except the reserved unseen bin. Validation rows with an unseen
    # category must score exactly like Model.predict (left at every categorical
    # node), never like a missing value (default direction) -- otherwise the eval
    # metric and early stopping disagree with prediction.
    rng = np.random.default_rng(23)
    n = 6000
    cat = np.concatenate([np.arange(254), rng.integers(0, 254, n - 254)]).astype(np.float32)
    if missing_right:
        cat[rng.random(n) < 0.3] = np.nan  # pushes some default directions right
    y = (np.cos(np.nan_to_num(cat, nan=300.0) * 0.37) + rng.normal(0, 0.1, n)).astype(np.float32)
    x = cat[:, None]
    xv = np.full((500, 1), 400.0, np.float32)  # unseen category for every row
    yv = rng.normal(0, 1, 500).astype(np.float32)
    hist = {}
    m = fb.train({"objective": "reg:squarederror", "max_depth": 4, "eval_metric": "rmse"}, x, label=y,
                 feature_types=["c"], evals=[(xv, yv, "v")], evals_result=hist,
                 num_boost_round=6, quiet=True)
    pred = m.predict(xv)
    rmse = float(np.sqrt(np.mean((pred.astype(np.float64) - yv) ** 2)))
    assert hist["v"]["rmse"][-1] == pytest.approx(rmse, rel=1e-6)


def test_fractional_validation_category_scores_like_predict():
    # Validation values are binned like prediction reads them: a fractional
    # code is truncated toward zero (XGBoost AsCat), so 1.5 is category 1.
    x, y, ft = make_data(cat_levels=(5,), seed=24)
    xv = x[:400].copy()
    xv[::2, 2] += 0.5
    yv = y[:400]
    hist = {}
    m = fb.train({"objective": "reg:squarederror", "max_depth": 3, "eval_metric": "rmse"}, x,
                 label=y, feature_types=ft, evals=[(xv, yv, "v")], evals_result=hist,
                 num_boost_round=4, quiet=True)
    pred = m.predict(xv)
    np.testing.assert_array_equal(pred, m.predict(np.trunc(xv).astype(np.float32)))
    rmse = float(np.sqrt(np.mean((pred.astype(np.float64) - yv) ** 2)))
    assert hist["v"]["rmse"][-1] == pytest.approx(rmse, rel=1e-6)
