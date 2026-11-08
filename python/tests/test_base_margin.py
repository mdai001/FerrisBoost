"""`base_margin=`: XGBoost's per-row starting margin, for training, validation sets,
prediction and SHAP.

Where binning is lossless (small integer features) the margins and SHAP values are
bit-identical to XGBoost trained / evaluated with `DMatrix(x, base_margin=...)`.
XGBoost semantics pinned here: the margin replaces `base_score` as the starting
point (it is not added to it), the intercept is still estimated from the labels and
stored in the model, and a plain `predict` (no margin) starts from that intercept.
"""
import json

import numpy as np
import pytest

import ferrisboost as fb

xgb = pytest.importorskip("xgboost")

ROUNDS = 6
OBJECTIVES = ["binary:logistic", "reg:squarederror", "count:poisson", "multi:softprob"]
EXTRA = {"multi:softprob": {"num_class": 3}}


def make(objective, n=900, m=5, seed=0):
    """Small-integer features (lossless binning), labels and a margin for `objective`."""
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 16, size=(n, m)).astype(np.float32)
    signal = 0.3 * (x[:, 0] - 8) + 0.2 * (x[:, 1] % 4) - 0.1 * x[:, 2]
    k = 3 if objective == "multi:softprob" else 1
    if objective == "binary:logistic":
        y = (signal + rng.normal(0, 1.0, n) > 0).astype(np.float32)
    elif objective == "count:poisson":
        y = rng.poisson(np.exp(0.15 * signal)).astype(np.float32)
    elif objective == "multi:softprob":
        y = np.digitize(signal + rng.normal(0, 0.6, n), [-0.8, 0.8]).astype(np.float32)
    else:
        y = (signal + rng.normal(0, 0.5, n)).astype(np.float32)
    margin = rng.normal(0, 0.4, size=(n,) if k == 1 else (n, k)).astype(np.float32)
    return x, y, margin


def params(objective, **extra):
    p = {"objective": objective, "max_depth": 3, "eta": 0.3, "nthread": 1,
         "tree_method": "hist", "max_bin": 255, **EXTRA.get(objective, {}), **extra}
    return p


def fb_params(p):
    return {k: v for k, v in p.items() if k != "tree_method"}


def xgb_train(objective, x, y, margin, rounds=ROUNDS, evals=(), **extra):
    d = xgb.DMatrix(x, y, base_margin=margin)
    watch = [(xgb.DMatrix(ex, ey, base_margin=em), name) for ex, ey, em, name in evals]
    res = {}
    bst = xgb.train(params(objective, **extra), d, rounds, evals=watch, evals_result=res,
                    verbose_eval=False)
    return bst, res


def xgb_margin(bst, x, margin=None):
    return bst.predict(xgb.DMatrix(x, base_margin=margin), output_margin=True)


def parse_scores(text):
    """'[3.7E-1,...]' / '0.37' -> list of float32 values (spelling differs, value must not)."""
    return [float(np.float32(v)) for v in text.strip("[]").split(",")]


def fb_base_score(model, tmp_path):
    path = tmp_path / "m.json"
    model.save_model(str(path), quiet=True)
    cfg = json.loads(path.read_text())["learner"]["learner_model_param"]["base_score"]
    return cfg


@pytest.mark.parametrize("objective", OBJECTIVES)
def test_training_with_base_margin_is_bit_exact_vs_xgboost(objective, tmp_path):
    x, y, margin = make(objective)
    bst, _ = xgb_train(objective, x, y, margin)
    model = fb.train(fb_params(params(objective)), x, label=y, num_boost_round=ROUNDS,
                     base_margin=margin, quiet=True)
    # The same margin at prediction: margin + trees, accumulated in tree order.
    np.testing.assert_array_equal(
        model.predict(x, output_margin=True, base_margin=margin), xgb_margin(bst, x, margin))
    # Without a margin the model starts from the (label-estimated) intercept.
    np.testing.assert_array_equal(model.predict(x, output_margin=True), xgb_margin(bst, x))
    xgb_path = tmp_path / "x.json"
    bst.save_model(str(xgb_path))
    xgb_bs = json.loads(xgb_path.read_text())["learner"]["learner_model_param"]["base_score"]
    assert parse_scores(fb_base_score(model, tmp_path)) == parse_scores(xgb_bs)
    # Transformed output as well (sigmoid / exp / softmax).
    np.testing.assert_array_equal(
        model.predict(x, base_margin=margin),
        bst.predict(xgb.DMatrix(x, base_margin=margin)))


def test_the_margin_replaces_the_intercept_and_is_not_added_to_it():
    x, y, margin = make("reg:squarederror")
    model = fb.train(fb_params(params("reg:squarederror")), x, label=y, num_boost_round=4,
                     base_margin=margin, quiet=True)
    with_margin = model.predict(x, base_margin=margin)
    plain = model.predict(x)
    intercept = model.predict(x[:1], n_trees=0)[0]
    # plain = intercept + trees, with_margin = margin + trees (same trees).
    np.testing.assert_allclose(with_margin - margin, plain - intercept, rtol=0, atol=1e-5)
    assert abs(float((with_margin - margin).mean() - (plain - intercept).mean())) < 1e-5


@pytest.mark.parametrize("objective", ["binary:logistic", "multi:softprob"])
def test_validation_margins_drive_the_eval_history(objective):
    x, y, margin = make(objective)
    vx, vy, vm = make(objective, n=300, seed=3)
    p = params(objective)
    _, xres = xgb_train(objective, x, y, margin, evals=[(vx, vy, vm, "v")])
    res = {}
    fb.train(fb_params(p), x, label=y, num_boost_round=ROUNDS, base_margin=margin,
             evals=[(vx, vy, "v")], base_margin_eval_set=[vm], evals_result=res, quiet=True)
    metric = next(iter(res["v"]))
    np.testing.assert_allclose(res["v"][metric], xres["v"][metric], rtol=1e-6)
    # Without the eval margin the history is different (the margin is really used).
    res2 = {}
    fb.train(fb_params(p), x, label=y, num_boost_round=ROUNDS, base_margin=margin,
             evals=[(vx, vy, "v")], evals_result=res2, quiet=True)
    assert res["v"][metric] != res2["v"][metric]


@pytest.mark.parametrize("objective", ["binary:logistic", "multi:softprob", "count:poisson"])
def test_shap_bias_includes_the_margin_like_xgboost(objective):
    x, y, margin = make(objective, n=300)
    bst, _ = xgb_train(objective, x, y, margin, rounds=4)
    model = fb.train(fb_params(params(objective)), x, label=y, num_boost_round=4,
                     base_margin=margin, quiet=True)
    got = model.predict(x, pred_contribs=True, base_margin=margin)
    want = bst.predict(xgb.DMatrix(x, base_margin=margin), pred_contribs=True)
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-6)
    # Rows sum to the margin-based raw prediction.
    raw = model.predict(x, output_margin=True, base_margin=margin)
    summed = got.sum(axis=-1)
    np.testing.assert_allclose(summed, raw, rtol=1e-4, atol=1e-4)
    # Approximate (Saabas) contributions use the margin too.
    approx = model.predict(x, pred_contribs=True, approx_contribs=True, base_margin=margin)
    want_approx = bst.predict(xgb.DMatrix(x, base_margin=margin), pred_contribs=True,
                              approx_contribs=True)
    np.testing.assert_allclose(approx, want_approx, rtol=1e-5, atol=1e-6)


def test_shap_interactions_use_the_margin():
    x, y, margin = make("binary:logistic", n=120)
    bst, _ = xgb_train("binary:logistic", x, y, margin, rounds=3)
    model = fb.train(fb_params(params("binary:logistic")), x, label=y, num_boost_round=3,
                     base_margin=margin, quiet=True)
    got = model.predict(x, pred_interactions=True, base_margin=margin)
    want = bst.predict(xgb.DMatrix(x, base_margin=margin), pred_interactions=True)
    np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-5)


def test_array_likes_and_shapes():
    x, y, margin = make("binary:logistic", n=200)
    p = fb_params(params("binary:logistic"))
    a = fb.train(p, x, label=y, num_boost_round=3, base_margin=margin, quiet=True)
    b = fb.train(p, x, label=y, num_boost_round=3, base_margin=margin.tolist(), quiet=True)
    c = fb.train(p, x, label=y, num_boost_round=3, base_margin=margin.reshape(-1, 1).astype(
        np.float64), quiet=True)
    pd = pytest.importorskip("pandas")
    d = fb.train(p, x, label=y, num_boost_round=3, base_margin=pd.Series(margin), quiet=True)
    ref = a.predict(x, base_margin=margin)
    for m in (b, c, d):
        np.testing.assert_array_equal(m.predict(x, base_margin=margin), ref)
    # Prediction accepts the same forms.
    np.testing.assert_array_equal(a.predict(x, base_margin=margin.tolist()), ref)
    np.testing.assert_array_equal(a.predict(x, base_margin=pd.Series(margin)), ref)


def test_predict_through_dataframe_and_noncontiguous_arrays():
    pd = pytest.importorskip("pandas")
    x, y, margin = make("reg:squarederror", n=150)
    model = fb.train(fb_params(params("reg:squarederror")), x, label=y, num_boost_round=3,
                     base_margin=margin, quiet=True)
    ref = model.predict(x, base_margin=margin)
    np.testing.assert_array_equal(model.predict(pd.DataFrame(x), base_margin=margin), ref)
    xf = np.asfortranarray(x)
    np.testing.assert_array_equal(model.predict(xf, base_margin=margin), ref)


def test_invalid_margins_are_rejected():
    x, y, margin = make("binary:logistic", n=100)
    p = fb_params(params("binary:logistic"))
    with pytest.raises(ValueError, match="base_margin has 99 rows"):
        fb.train(p, x, label=y, num_boost_round=1, base_margin=margin[:99], quiet=True)
    bad = margin.copy()
    bad[3] = np.nan
    with pytest.raises(ValueError, match="finite"):
        fb.train(p, x, label=y, num_boost_round=1, base_margin=bad, quiet=True)
    with pytest.raises(ValueError, match="shape"):
        fb.train(p, x, label=y, num_boost_round=1, base_margin=np.zeros((100, 2)), quiet=True)
    mc = fb_params(params("multi:softprob"))
    xm, ym, _ = make("multi:softprob", n=100)
    with pytest.raises(ValueError, match=r"shape \(n, 3\)"):
        fb.train(mc, xm, label=ym, num_boost_round=1, base_margin=np.zeros(100), quiet=True)
    with pytest.raises(ValueError, match="base_margin_eval_set has 2 entries"):
        fb.train(p, x, label=y, num_boost_round=1, evals=[(x, y, "v")],
                 base_margin_eval_set=[None, None], quiet=True)
    model = fb.train(p, x, label=y, num_boost_round=1, quiet=True)
    with pytest.raises(ValueError, match="base_margin has 99 rows"):
        model.predict(x, base_margin=margin[:99])
    with pytest.raises(ValueError, match="use_gpu"):
        model.predict(x, base_margin=margin, use_gpu=True)


def test_iteration_range_applies_on_top_of_the_margin():
    x, y, margin = make("binary:logistic", n=300)
    bst, _ = xgb_train("binary:logistic", x, y, margin, rounds=5)
    model = fb.train(fb_params(params("binary:logistic")), x, label=y, num_boost_round=5,
                     base_margin=margin, quiet=True)
    for rng_ in [(0, 2), (1, 4), (2, 0)]:
        got = model.predict(x, output_margin=True, base_margin=margin, iteration_range=rng_)
        want = bst.predict(xgb.DMatrix(x, base_margin=margin), output_margin=True,
                           iteration_range=rng_)
        np.testing.assert_array_equal(got, want)


# ----------------------------------------------------------------------- files
def write_table(tmp_path, x, y, margin, name="data"):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    cols = {f"f{i}": x[:, i] for i in range(x.shape[1])}
    cols["label"] = y
    if margin.ndim == 1:
        cols["offset"] = margin
    else:
        for c in range(margin.shape[1]):
            cols[f"offset{c}"] = margin[:, c]
    path = tmp_path / f"{name}.parquet"
    pq.write_table(pa.table(cols), path)
    return str(path)


@pytest.mark.parametrize("objective", ["binary:logistic", "multi:softprob"])
def test_file_input_reads_the_margin_column(objective, tmp_path):
    x, y, margin = make(objective, n=400)
    path = write_table(tmp_path, x, y, margin)
    cols = "offset" if margin.ndim == 1 else [f"offset{c}" for c in range(margin.shape[1])]
    p = fb_params(params(objective))
    from_array = fb.train(p, x, label=y, num_boost_round=4, base_margin=margin, quiet=True)
    from_file = fb.train(p, path, label="label", num_boost_round=4, base_margin=cols, quiet=True)
    assert from_file.feature_names == [f"f{i}" for i in range(x.shape[1])]
    np.testing.assert_array_equal(
        from_file.predict(x, output_margin=True, base_margin=margin),
        from_array.predict(x, output_margin=True, base_margin=margin))
    # Prediction from the file picks the margin up from the same column.
    np.testing.assert_array_equal(
        from_file.predict(path, output_margin=True, base_margin=cols),
        from_file.predict(x, output_margin=True, base_margin=margin))
    # ... and so do SHAP values and the Parquet writer.
    np.testing.assert_allclose(
        from_file.predict(path, pred_contribs=True, base_margin=cols),
        from_file.predict(x, pred_contribs=True, base_margin=margin), rtol=1e-6, atol=1e-6)
    out = tmp_path / "pred.parquet"
    n = from_file.predict_to_parquet(path, str(out), output_margin=True, base_margin=cols)
    assert n == len(x)
    table = pytest.importorskip("pyarrow.parquet").read_table(out)
    got = np.column_stack([table.column(c).to_numpy() for c in table.column_names])
    want = from_file.predict(x, output_margin=True, base_margin=margin).reshape(len(x), -1)
    np.testing.assert_array_equal(got, want)


def test_file_validation_sets_read_the_same_margin_column(tmp_path):
    x, y, margin = make("binary:logistic", n=400)
    vx, vy, vm = make("binary:logistic", n=150, seed=5)
    train_path = write_table(tmp_path, x, y, margin, "train")
    valid_path = write_table(tmp_path, vx, vy, vm, "valid")
    p = fb_params(params("binary:logistic"))
    res_file, res_arr = {}, {}
    fb.train(p, train_path, label="label", num_boost_round=4, base_margin="offset",
             evals=[(valid_path, "v")], evals_result=res_file, quiet=True)
    fb.train(p, x, label=y, num_boost_round=4, base_margin=margin, evals=[(vx, vy, "v")],
             base_margin_eval_set=[vm], evals_result=res_arr, quiet=True)
    assert res_file == res_arr


def test_file_margin_errors(tmp_path):
    x, y, margin = make("binary:logistic", n=100)
    path = write_table(tmp_path, x, y, margin)
    p = fb_params(params("binary:logistic"))
    with pytest.raises(ValueError):
        fb.train(p, path, label="label", num_boost_round=1, base_margin="offset_typo", quiet=True)
    with pytest.raises(ValueError, match="column name"):
        fb.train(p, path, label="label", num_boost_round=1, base_margin=margin, quiet=True)
    model = fb.train(p, path, label="label", num_boost_round=1, base_margin="offset", quiet=True)
    with pytest.raises(ValueError):
        model.predict(path, base_margin="nope")
    with pytest.raises(ValueError, match="feature name"):
        model.predict(path, base_margin="f1")
    # A positional (numpy-trained) model has nowhere to put a margin column.
    pos = fb.train(p, x, label=y, num_boost_round=1, quiet=True)
    with pytest.raises(ValueError):
        pos.predict(path, base_margin="offset")


def test_csv_margin_column_by_name_and_index(tmp_path):
    x, y, margin = make("reg:squarederror", n=200)
    header = ",".join([f"f{i}" for i in range(x.shape[1])] + ["label", "offset"])
    rows = np.column_stack([x, y, margin])
    named = tmp_path / "named.csv"
    np.savetxt(named, rows, delimiter=",", header=header, comments="", fmt="%.9g")
    bare = tmp_path / "bare.csv"
    np.savetxt(bare, rows, delimiter=",", fmt="%.9g")
    p = fb_params(params("reg:squarederror"))
    ref = fb.train(p, x, label=y, num_boost_round=3, base_margin=margin, quiet=True)
    by_name = fb.train(p, str(named), label="label", num_boost_round=3, base_margin="offset",
                       quiet=True)
    by_index = fb.train(p, str(bare), label=x.shape[1], num_boost_round=3, header=False,
                        base_margin=x.shape[1] + 1, quiet=True)
    want = ref.predict(x, output_margin=True, base_margin=margin)
    for m in (by_name, by_index):
        np.testing.assert_array_equal(m.predict(x, output_margin=True, base_margin=margin), want)
    np.testing.assert_array_equal(
        by_name.predict(str(named), output_margin=True, base_margin="offset"), want)


# ---------------------------------------------------------------------- sklearn
def test_sklearn_passthrough():
    x, y, margin = make("binary:logistic", n=500)
    vx, vy, vm = make("binary:logistic", n=200, seed=7)
    xgbc = xgb.XGBClassifier(n_estimators=5, max_depth=3, learning_rate=0.3, n_jobs=1,
                             tree_method="hist", max_bin=255)
    xgbc.fit(x, y, base_margin=margin, eval_set=[(vx, vy)], base_margin_eval_set=[vm],
             verbose=False)
    clf = fb.FerrisBoostClassifier(n_estimators=5, max_depth=3, learning_rate=0.3, n_jobs=1)
    clf.fit(x, y, base_margin=margin, eval_set=[(vx, vy)], base_margin_eval_set=[vm])
    metric = next(iter(clf.evals_result()["validation_0"]))
    np.testing.assert_allclose(
        clf.evals_result()["validation_0"][metric], xgbc.evals_result()["validation_0"][metric],
        rtol=1e-6)
    np.testing.assert_array_equal(clf.predict_proba(vx, base_margin=vm),
                                  xgbc.predict_proba(vx, base_margin=vm))
    np.testing.assert_array_equal(clf.predict(vx, base_margin=vm), xgbc.predict(vx, base_margin=vm))
    np.testing.assert_array_equal(clf.predict(vx, output_margin=True, base_margin=vm),
                                  xgbc.predict(vx, output_margin=True, base_margin=vm))
    with pytest.raises(ValueError, match="base_margin_eval_set"):
        clf.fit(x, y, base_margin=margin, base_margin_eval_set=[vm])


def test_sklearn_multiclass_and_regressor_passthrough():
    x, y, margin = make("multi:softprob", n=500)
    xgbc = xgb.XGBClassifier(n_estimators=4, max_depth=3, learning_rate=0.3, n_jobs=1,
                             tree_method="hist", max_bin=255)
    xgbc.fit(x, y, base_margin=margin)
    clf = fb.FerrisBoostClassifier(n_estimators=4, max_depth=3, learning_rate=0.3, n_jobs=1)
    clf.fit(x, y, base_margin=margin)
    np.testing.assert_array_equal(clf.predict_proba(x, base_margin=margin),
                                  xgbc.predict_proba(x, base_margin=margin))
    xr, yr, mr = make("count:poisson", n=500)
    xgbr = xgb.XGBRegressor(n_estimators=4, max_depth=3, learning_rate=0.3, n_jobs=1,
                            tree_method="hist", max_bin=255, objective="count:poisson")
    xgbr.fit(xr, yr, base_margin=mr)
    reg = fb.FerrisBoostRegressor(n_estimators=4, max_depth=3, learning_rate=0.3, n_jobs=1,
                                  objective="count:poisson")
    reg.fit(xr, yr, base_margin=mr)
    np.testing.assert_array_equal(reg.predict(xr, base_margin=mr), xgbr.predict(xr, base_margin=mr))


# -------------------------------------------------------------------------- GPU
def _has_cuda():
    try:
        fb.train({"objective": "reg:squarederror", "device": "cuda", "max_depth": 2}, *make("reg:squarederror", n=64)[:2],
                 num_boost_round=1, quiet=True)
        return True
    except Exception as e:  # noqa: BLE001
        if "CUDA" in str(e) or "cuda" in str(e):
            return False
        raise


cuda = pytest.mark.skipif(not _has_cuda(), reason="no usable CUDA build or device")


@cuda
@pytest.mark.parametrize("objective", ["binary:logistic", "multi:softprob", "count:poisson"])
def test_gpu_exact_matches_cpu_byte_for_byte_with_margins(objective, tmp_path):
    x, y, margin = make(objective, n=2000)
    p = fb_params(params(objective, gpu_math="exact"))
    cpu = fb.train({k: v for k, v in p.items() if k != "gpu_math"}, x, label=y,
                   num_boost_round=5, base_margin=margin, quiet=True)
    gpu = fb.train({**p, "device": "cuda"}, x, label=y, num_boost_round=5, base_margin=margin,
                   quiet=True)
    a, b = tmp_path / "cpu.json", tmp_path / "gpu.json"
    cpu.save_model(str(a), quiet=True)
    gpu.save_model(str(b), quiet=True)
    assert a.read_bytes() == b.read_bytes()


@cuda
@pytest.mark.parametrize("objective", ["binary:logistic", "multi:softprob"])
def test_gpu_fast_starts_from_the_margins_and_is_deterministic(objective, tmp_path):
    x, y, margin = make(objective, n=2000)
    p = {**fb_params(params(objective)), "device": "cuda", "gpu_math": "fast"}
    one = fb.train(p, x, label=y, num_boost_round=5, base_margin=margin, quiet=True)
    two = fb.train(p, x, label=y, num_boost_round=5, base_margin=margin, quiet=True)
    a, b = tmp_path / "one.json", tmp_path / "two.json"
    one.save_model(str(a), quiet=True)
    two.save_model(str(b), quiet=True)
    assert a.read_bytes() == b.read_bytes()
    # The margins really enter the device predictions: a different margin gives a
    # different model, and the quality tracks the exact path.
    other = fb.train(p, x, label=y, num_boost_round=5, base_margin=margin[::-1].copy(), quiet=True)
    c = tmp_path / "other.json"
    other.save_model(str(c), quiet=True)
    assert a.read_bytes() != c.read_bytes()
    exact = fb.train({**p, "gpu_math": "exact"}, x, label=y, num_boost_round=5,
                     base_margin=margin, quiet=True)
    np.testing.assert_allclose(
        one.predict(x, output_margin=True, base_margin=margin),
        exact.predict(x, output_margin=True, base_margin=margin), rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------- ranking
def make_rank(seed=0, sizes=(3, 5, 8, 13, 21, 34, 40, 7, 9, 30)):
    rng = np.random.default_rng(seed)
    n = int(sum(sizes))
    x = rng.integers(0, 12, size=(n, 6)).astype(np.float32)
    s = x[:, 0] * 0.5 + x[:, 1] * 0.3 - x[:, 2] * 0.2 + rng.normal(0, 1.5, n)
    y = np.clip(np.round((s - s.min()) / (s.max() - s.min()) * 4), 0, 4).astype(np.float32)
    qid = np.repeat(np.arange(len(sizes)), sizes)
    margin = rng.normal(0, 0.5, n).astype(np.float32)
    return x, y, qid, margin


@pytest.mark.parametrize("objective", ["rank:ndcg", "rank:pairwise"])
@pytest.mark.parametrize("pair_method", ["topk", "mean"])
def test_base_margin_with_ranking_objectives_matches_xgboost(objective, pair_method):
    x, y, qid, margin = make_rank()
    extra = {"lambdarank_pair_method": pair_method}
    d = xgb.DMatrix(x, y, qid=qid, base_margin=margin)
    bst = xgb.train({"objective": objective, "tree_method": "hist", "nthread": 1,
                     "max_depth": 3, "max_bin": 16, **extra}, d, 5)
    model = fb.train({"objective": objective, "nthread": 1, "max_depth": 3, "max_bin": 16, **extra},
                     x, label=y, qid=qid, base_margin=margin, num_boost_round=5, quiet=True)
    np.testing.assert_array_equal(
        model.predict(x, output_margin=True, base_margin=margin),
        bst.predict(xgb.DMatrix(x, base_margin=margin), output_margin=True))
    # The margin matters (a plain model starts from the intercept instead).
    assert not np.array_equal(model.predict(x), model.predict(x, base_margin=margin))


def test_ranker_estimator_passes_base_margin_through():
    x, y, qid, margin = make_rank()
    kw = dict(n_estimators=4, max_depth=3, max_bin=16, n_jobs=1, objective="rank:ndcg")
    xr = xgb.XGBRanker(tree_method="hist", **kw)
    xr.fit(x, y, qid=qid, base_margin=margin)
    fr = fb.FerrisBoostRanker(**kw)
    fr.fit(x, y, qid=qid, base_margin=margin)
    np.testing.assert_array_equal(fr.predict(x, base_margin=margin),
                                  xr.predict(x, base_margin=margin))
