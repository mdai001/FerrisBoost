"""High-cardinality categorical features (codes above 253, stored at 16 bits).

Category codes are lossless bins (one bin per code up to the largest), so the
trees, category sets and predictions are bit-identical to XGBoost 3.4.1
`tree_method="hist"` (partition-based splits with `max_cat_threshold`,
one-hot below `max_cat_to_onehot`) with 1,000 and 20,000 categories. 8-bit
categorical and numeric features of the same data keep their 8-bit columns.
"""
import json

import numpy as np
import pytest

import ferrisboost as fb

xgb = pytest.importorskip("xgboost")


def make(n, levels, seed=0, missing=0.02, objective="reg:squarederror"):
    """Small-integer numeric column, then one categorical column per level
    count (every code up to the largest occurs)."""
    rng = np.random.default_rng(seed)
    cols = [rng.integers(0, 8, n).astype(np.float32)]
    for k in levels:
        c = rng.integers(0, k, n)
        c[: min(k, n)] = np.arange(min(k, n))
        rng.shuffle(c)
        cols.append(c.astype(np.float32))
    x = np.stack(cols, axis=1)
    signal = 0.2 * (cols[0] - 4)
    for k, c in zip(levels, cols[1:]):
        signal = signal + rng.normal(0, 1, k)[c.astype(int)]
    if objective == "binary:logistic":
        y = (signal + rng.normal(0, 0.8, n) > 0).astype(np.float32)
    else:
        y = (signal + rng.normal(0, 0.3, n)).astype(np.float32)
    if missing:
        x[rng.random(x.shape) < missing] = np.nan
    return x, y, ["q"] + ["c"] * len(levels)


def xgb_train(x, y, ft, params, rounds):
    d = xgb.DMatrix(x, label=y, feature_types=ft, enable_categorical=True)
    p = {"tree_method": "hist", "nthread": 1, "base_score": 0.5, "max_bin": 255, **params}
    return xgb.train(p, d, num_boost_round=rounds), d


def fb_train(x, y, ft, params, rounds, **kw):
    p = {"nthread": 1, "base_score": 0.5, "max_bin": 255, **params}
    return fb.train(p, x, label=y, num_boost_round=rounds, feature_types=ft, quiet=True, **kw)


def trees(model, tmp_path, name):
    path = str(tmp_path / f"{name}.json")
    model.save_model(path, quiet=True) if isinstance(model, fb.Model) else model.save_model(path)
    with open(path) as f:
        return json.load(f)["learner"]["gradient_booster"]["model"]["trees"]


def structure(tree):
    return {k: tree[k] for k in ("left_children", "right_children", "split_indices", "split_type",
                                 "default_left", "categories", "categories_nodes",
                                 "categories_segments", "categories_sizes")}


def max_set_code(tree_list, feature):
    best = -1
    for t in tree_list:
        for n, s, z in zip(t["categories_nodes"], t["categories_segments"], t["categories_sizes"]):
            if t["split_indices"][n] == feature and z:
                best = max(best, max(t["categories"][s:s + z]))
    return best


CASES = [
    # (levels, n_rows, extra params)
    ((1000, 12), 20_000, {}),
    ((1000,), 20_000, {"max_cat_threshold": 300}),
    ((1000,), 20_000, {"max_cat_to_onehot": 1001}),      # one-hot over 1,000 codes
    ((1000, 300), 20_000, {"reg_lambda": 3.0, "min_child_weight": 2.0}),
    ((20_000,), 60_000, {}),
    ((20_000, 7), 60_000, {"max_cat_threshold": 2000}),
]


@pytest.mark.parametrize("case", CASES, ids=lambda c: f"{c[0]}-{c[2]}")
@pytest.mark.parametrize("objective", ["reg:squarederror", "binary:logistic"])
def test_bit_exact_vs_xgboost_with_many_categories(case, objective, tmp_path):
    levels, n, extra = case
    x, y, ft = make(n, levels, seed=len(levels) + n, objective=objective)
    params = {"objective": objective, "max_depth": 4, **extra}
    bst, d = xgb_train(x, y, ft, params, 4)
    ours = fb_train(x, y, ft, params, 4)
    mine, ref = trees(ours, tmp_path, "f"), trees(bst, tmp_path, "x")
    assert len(mine) == len(ref)
    for i, (a, b) in enumerate(zip(mine, ref)):
        assert structure(a) == structure(b), f"tree {i} differs"
    np.testing.assert_array_equal(ours.predict(x, output_margin=True),
                                  bst.predict(d, output_margin=True))
    # The wide feature really is split with codes above the 8-bit range.
    assert max_set_code(mine, 1) > 255


def test_wide_categorical_model_round_trips_and_loads_in_xgboost(tmp_path):
    x, y, ft = make(20_000, (1000, 12), seed=4, objective="binary:logistic")
    model = fb_train(x, y, ft, {"objective": "binary:logistic", "max_depth": 4}, 5)
    path = tmp_path / "wide_cat.json"
    model.save_model(str(path), quiet=True)
    loaded = fb.Model.load_model(str(path), quiet=True)
    np.testing.assert_array_equal(loaded.predict(x), model.predict(x))
    assert trees(loaded, tmp_path, "again") == trees(model, tmp_path, "orig")
    bst = xgb.Booster(model_file=str(path))
    d = xgb.DMatrix(x, feature_types=ft, enable_categorical=True)
    np.testing.assert_array_equal(bst.predict(d, output_margin=True),
                                  model.predict(x, output_margin=True))
    # Unseen codes (beyond the trained range) go left in both.
    probe = x.copy()
    probe[:, 1] = 5000.0
    dp = xgb.DMatrix(probe, feature_types=ft, enable_categorical=True)
    np.testing.assert_array_equal(bst.predict(dp, output_margin=True),
                                  model.predict(probe, output_margin=True))


def test_validation_binning_of_a_wide_feature_scores_like_predict():
    x, y, ft = make(20_000, (1000, 12), seed=5)
    xv, yv, _ = make(4000, (1000, 12), seed=6)
    xv[::7, 1] = 1500.0   # unseen in training
    xv[::11, 1] += 0.5    # fractional: read as the integer part
    hist = {}
    model = fb_train(x, y, ft, {"objective": "reg:squarederror", "max_depth": 4}, 4,
                     evals=[(xv, yv, "v")], evals_result=hist)
    pred = model.predict(xv)
    rmse = float(np.sqrt(np.mean((pred.astype(np.float64) - yv) ** 2)))
    assert hist["v"]["rmse"][-1] == pytest.approx(rmse, rel=1e-6)


def test_model_does_not_depend_on_threads_or_blocking(tmp_path):
    x, y, ft = make(20_000, (1000, 12, 300), seed=7)
    p = {"objective": "reg:squarederror", "max_depth": 5, "subsample": 0.8, "colsample_bytree": 0.8}
    ref = fb_train(x, y, ft, p, 4)
    for extra in ({"nthread": 4}, {"cols_per_block": 1}, {"cols_per_block": 3, "nthread": 3}):
        got = fb_train(x, y, ft, {**p, **extra}, 4)
        assert trees(got, tmp_path, "g") == trees(ref, tmp_path, "r"), extra


def test_largest_code_and_the_limit():
    rng = np.random.default_rng(8)
    x = rng.integers(0, 50, (3000, 1)).astype(np.float32)
    x[0, 0] = 65533.0
    y = rng.normal(0, 1, 3000).astype(np.float32)
    fb_train(x, y, ["c"], {"objective": "reg:squarederror", "max_depth": 2}, 1)
    x[0, 0] = 65534.0
    with pytest.raises(ValueError, match="65533"):
        fb_train(x, y, ["c"], {"objective": "reg:squarederror", "max_depth": 2}, 1)


def test_pandas_and_pyarrow_with_many_levels_match_numpy(tmp_path):
    pd = pytest.importorskip("pandas")
    pa = pytest.importorskip("pyarrow")
    x, y, ft = make(20_000, (1000,), seed=9, missing=0.0)
    params = {"objective": "reg:squarederror", "max_depth": 4, "enable_categorical": True}
    ref = fb_train(x, y, ft, params, 3)
    levels = [f"L{i:04d}" for i in range(1000)]
    df = pd.DataFrame({
        "num": x[:, 0],
        "cat": pd.Categorical.from_codes(x[:, 1].astype(int), categories=levels),
    })
    from_pandas = fb.train({"nthread": 1, "base_score": 0.5, **params}, df, label=y,
                           num_boost_round=3, quiet=True)
    assert trees(from_pandas, tmp_path, "pd") == trees(ref, tmp_path, "np")
    table = pa.table({
        "num": pa.array(x[:, 0]),
        "cat": pa.DictionaryArray.from_arrays(pa.array(x[:, 1].astype(np.int32)), pa.array(levels)),
    })
    from_arrow = fb.train({"nthread": 1, "base_score": 0.5, **params}, table, label=y,
                          num_boost_round=3, quiet=True)
    assert trees(from_arrow, tmp_path, "pa") == trees(ref, tmp_path, "np")
    np.testing.assert_array_equal(from_arrow.predict(table), ref.predict(x))


def test_parquet_dictionary_column_and_cache_match_numpy(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    x, y, ft = make(20_000, (1000, 12), seed=10)
    params = {"objective": "reg:squarederror", "max_depth": 4}
    ref = fb_train(x, y, ft, params, 3)

    def dict_col(v):
        codes = np.where(np.isnan(v), 0, v).astype(np.int32)
        return pa.DictionaryArray.from_arrays(
            pa.array(codes, mask=np.isnan(v)), pa.array(np.arange(int(np.nanmax(v)) + 1, dtype=np.int32)))

    path = str(tmp_path / "wide_cat.parquet")
    pq.write_table(pa.table({"num": pa.array(x[:, 0]), "big": dict_col(x[:, 1]),
                             "small": dict_col(x[:, 2]), "label": pa.array(y)}), path)
    p = {"nthread": 1, "base_score": 0.5, "max_bin": 255, **params}
    from_file = fb.train(p, path, label="label", num_boost_round=3, quiet=True)
    assert trees(from_file, tmp_path, "file") == trees(ref, tmp_path, "np")
    cache = tmp_path / "cache"
    for i in range(2):  # build, then reuse the 16-bit (version 3) cache
        cached = fb.train({**p, "cache_path": str(cache), "cache_budget_bytes": 0}, path,
                          label="label", num_boost_round=3, quiet=True)
        assert trees(cached, tmp_path, f"c{i}") == trees(ref, tmp_path, "np"), i
    manifest = json.loads((cache / "manifest.json").read_text())
    assert manifest["version"] == 3 and 2 in manifest["block_widths"]


def _has_cuda():
    try:
        x, y, ft = make(64, (3,))
        fb.train({"device": "cuda", "max_depth": 2, "objective": "reg:squarederror"}, x, label=y,
                 feature_types=ft,
                 num_boost_round=1, quiet=True)
        return True
    except Exception as e:  # noqa: BLE001
        if "CUDA" in str(e) or "cuda" in str(e):
            return False
        raise


@pytest.mark.skipif(not _has_cuda(), reason="no usable CUDA build or device")
def test_gpu_trains_the_cpu_model_with_16_bit_categorical_columns(tmp_path):
    # 1,000 and 20,000 categories (16-bit blocks; 20,000 slots exceed one
    # shared-memory histogram tile) next to 8-bit features.
    x, y, ft = make(4000, (1000, 20000, 12), seed=11)
    p = {"objective": "reg:squarederror", "max_depth": 5, "subsample": 0.8}

    def raw(model, name):
        path = tmp_path / f"{name}.json"
        model.save_model(str(path), quiet=True)
        return path.read_bytes()

    cpu = raw(fb_train(x, y, ft, p, 3), "cpu")
    for resident in (0, 8):
        gpu = fb_train(x, y, ft, {**p, "device": "cuda", "resident_blocks": resident}, 3)
        assert raw(gpu, "gpu") == cpu, f"resident_blocks={resident}"
    # Up to 254 categories the GPU trains as before.
    x, y, ft = make(4000, (254,), seed=11)
    fb_train(x, y, ft, {"device": "cuda", "objective": "reg:squarederror"}, 1)
