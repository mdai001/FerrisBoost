"""Training continuation (`xgb_model=`).

What holds, and is pinned here:

* On lossless-binned data (small integer features) continuing a model is
  bit-identical to XGBoost 3.4.1's own continuation, whether the model to
  continue is a FerrisBoost `Model`, a path to FerrisBoost / XGBoost JSON or an
  `xgboost.Booster`, for binary / regression / poisson / multiclass.
* `num_boost_round` NEW trees are appended (XGBoost 3.4.1 semantics); the intercept
  of the continued model is kept; the objective follows the model.
* On the SAME data, N rounds followed by M continued rounds is byte-identical to
  N+M uninterrupted rounds, including `subsample` / `colsample_*` (FerrisBoost keys
  its sampling on the absolute round). XGBoost makes no such promise.
* On other data the cuts are sketched afresh while the existing trees keep their
  float thresholds, so the old trees are scored on the raw values; lossless data
  still matches XGBoost bit for bit.
"""
import json

import numpy as np
import pytest

import ferrisboost as fb

xgb = pytest.importorskip("xgboost")

OBJECTIVES = ["binary:logistic", "reg:squarederror", "count:poisson", "multi:softprob"]
EXTRA = {"multi:softprob": {"num_class": 3}}


def make(objective, n=900, m=5, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 16, size=(n, m)).astype(np.float32)
    signal = 0.3 * (x[:, 0] - 8) + 0.2 * (x[:, 1] % 4) - 0.1 * x[:, 2]
    if objective == "binary:logistic":
        y = (signal + rng.normal(0, 1.0, n) > 0).astype(np.float32)
    elif objective == "count:poisson":
        y = rng.poisson(np.exp(0.15 * signal)).astype(np.float32)
    elif objective == "multi:softprob":
        y = np.digitize(signal + rng.normal(0, 0.6, n), [-0.8, 0.8]).astype(np.float32)
    else:
        y = (signal + rng.normal(0, 0.5, n)).astype(np.float32)
    return x, y


def params(objective, **extra):
    return {"objective": objective, "max_depth": 3, "eta": 0.3, "nthread": 1,
            "max_bin": 255, **EXTRA.get(objective, {}), **extra}


def xparams(objective, **extra):
    return {**params(objective, **extra), "tree_method": "hist"}


def raw(model, x, **kw):
    return model.predict(x, output_margin=True, **kw)


def xraw(bst, x, margin=None):
    return bst.predict(xgb.DMatrix(x, base_margin=margin), output_margin=True)


def saved(model, tmp_path, name):
    path = tmp_path / f"{name}.json"
    model.save_model(str(path), quiet=True)
    return path


@pytest.mark.parametrize("objective", OBJECTIVES)
def test_continuation_is_bit_exact_vs_xgboost(objective):
    x, y = make(objective)
    d = xgb.DMatrix(x, y)
    first = xgb.train(xparams(objective), d, 5)
    cont = xgb.train(xparams(objective), d, 5, xgb_model=first)
    m1 = fb.train(params(objective), x, label=y, num_boost_round=5, quiet=True)
    m2 = fb.train(params(objective), x, label=y, num_boost_round=5, xgb_model=m1, quiet=True)
    assert m2.num_trees == cont.num_boosted_rounds() * EXTRA.get(objective, {}).get("num_class", 1)
    np.testing.assert_array_equal(raw(m2, x), xraw(cont, x))
    # The model passed in is not modified.
    assert m1.num_trees == 5 * EXTRA.get(objective, {}).get("num_class", 1)


@pytest.mark.parametrize("objective", ["binary:logistic", "multi:softprob"])
def test_xgboost_models_are_continued_from_path_and_from_booster(objective, tmp_path):
    x, y = make(objective)
    d = xgb.DMatrix(x, y)
    first = xgb.train(xparams(objective), d, 4)
    cont = xgb.train(xparams(objective), d, 4, xgb_model=first)
    path = tmp_path / "xgb.json"
    first.save_model(str(path))
    from_path = fb.train(params(objective), x, label=y, num_boost_round=4,
                         xgb_model=str(path), quiet=True)
    from_booster = fb.train(params(objective), x, label=y, num_boost_round=4,
                            xgb_model=first, quiet=True)
    from_loaded = fb.train(params(objective), x, label=y, num_boost_round=4,
                           xgb_model=fb.Model.load_model(str(path), quiet=True), quiet=True)
    want = xraw(cont, x)
    for m in (from_path, from_booster, from_loaded):
        np.testing.assert_array_equal(raw(m, x), want)
    # A continued FerrisBoost model is still XGBoost-loadable and agrees.
    out = saved(from_path, tmp_path, "cont")
    reloaded = xgb.Booster()
    reloaded.load_model(str(out))
    np.testing.assert_array_equal(xraw(reloaded, x), want)


def test_continuing_on_different_data_matches_xgboost():
    x, y = make("binary:logistic", seed=0)
    x2, y2 = make("binary:logistic", n=700, seed=11)
    first = xgb.train(xparams("binary:logistic"), xgb.DMatrix(x, y), 4)
    cont = xgb.train(xparams("binary:logistic"), xgb.DMatrix(x2, y2), 4, xgb_model=first)
    m1 = fb.train(params("binary:logistic"), x, label=y, num_boost_round=4, quiet=True)
    m2 = fb.train(params("binary:logistic"), x2, label=y2, num_boost_round=4, xgb_model=m1,
                  quiet=True)
    np.testing.assert_array_equal(raw(m2, x2), xraw(cont, x2))
    np.testing.assert_array_equal(raw(m2, x), xraw(cont, x))
    # The first four trees are untouched and keep scoring on their float thresholds.
    np.testing.assert_array_equal(raw(m2, x, iteration_range=(0, 4)), raw(m1, x))


def test_continuation_with_validation_sets_matches_xgboost_history():
    x, y = make("binary:logistic")
    vx, vy = make("binary:logistic", n=300, seed=5)
    d, dv = xgb.DMatrix(x, y), xgb.DMatrix(vx, vy)
    first = xgb.train(xparams("binary:logistic"), d, 3)
    xres = {}
    xgb.train(xparams("binary:logistic", eval_metric=["logloss", "auc"]), d, 5, xgb_model=first,
              evals=[(dv, "v")], evals_result=xres, verbose_eval=False)
    m1 = fb.train(params("binary:logistic"), x, label=y, num_boost_round=3, quiet=True)
    res = {}
    m2 = fb.train(params("binary:logistic", eval_metric=["logloss", "auc"]), x, label=y,
                  num_boost_round=5, xgb_model=m1, evals=[(vx, vy, "v")], evals_result=res,
                  quiet=True)
    assert m2.num_trees == 8
    for metric in ("logloss", "auc"):
        assert len(res["v"][metric]) == 5, "evals_result covers the new rounds only"
        np.testing.assert_allclose(res["v"][metric], xres["v"][metric], rtol=1e-6)


def test_early_stopping_reports_the_absolute_best_iteration_like_xgboost():
    x, y = make("binary:logistic", n=500)
    vx, vy = make("binary:logistic", n=500, seed=9)
    d, dv = xgb.DMatrix(x, y), xgb.DMatrix(vx, vy)
    p = {"eta": 0.5, "max_depth": 4}
    first = xgb.train(xparams("binary:logistic", **p), d, 3)
    cont = xgb.train(xparams("binary:logistic", **p), d, 60, xgb_model=first,
                     evals=[(dv, "v")], early_stopping_rounds=3, verbose_eval=False)
    m1 = fb.train(params("binary:logistic", **p), x, label=y, num_boost_round=3, quiet=True)
    m2 = fb.train(params("binary:logistic", **p), x, label=y, num_boost_round=60, xgb_model=m1,
                  evals=[(vx, vy, "v")], early_stopping_rounds=3, quiet=True)
    assert m2.best_iteration == cont.best_iteration
    assert m2.best_iteration >= 3, "an absolute round of the whole model"
    assert m2.num_trees == cont.num_boosted_rounds()
    assert m2.best_score == pytest.approx(cont.best_score, rel=1e-6)


def test_a_previous_best_iteration_is_kept_when_not_early_stopping():
    x, y = make("binary:logistic", n=500)
    vx, vy = make("binary:logistic", n=500, seed=9)
    m1 = fb.train(params("binary:logistic", eta=0.5), x, label=y, num_boost_round=40,
                  evals=[(vx, vy, "v")], early_stopping_rounds=2, quiet=True)
    assert m1.best_iteration is not None
    m2 = fb.train(params("binary:logistic", eta=0.5), x, label=y, num_boost_round=2,
                  xgb_model=m1, quiet=True)
    assert m2.best_iteration == m1.best_iteration  # XGBoost keeps the stored attribute


@pytest.mark.parametrize("objective", ["binary:logistic", "multi:softprob", "count:poisson"])
def test_same_data_n_plus_m_equals_n_plus_m_rounds_byte_for_byte(objective, tmp_path):
    x, y = make(objective, n=700)
    p = params(objective, subsample=0.7, colsample_bytree=0.8, colsample_bylevel=0.9, seed=3)
    full = fb.train(p, x, label=y, num_boost_round=9, quiet=True)
    part = fb.train(p, x, label=y, num_boost_round=4, quiet=True)
    joined = fb.train(p, x, label=y, num_boost_round=5, xgb_model=part, quiet=True)
    assert saved(full, tmp_path, "full").read_bytes() == saved(joined, tmp_path, "joined").read_bytes()
    # Three legs, too.
    mid = fb.train(p, x, label=y, num_boost_round=3, xgb_model=part, quiet=True)
    last = fb.train(p, x, label=y, num_boost_round=2, xgb_model=mid, quiet=True)
    assert saved(full, tmp_path, "full2").read_bytes() == saved(last, tmp_path, "last").read_bytes()


def test_base_margin_and_continuation_combine_like_xgboost():
    x, y = make("binary:logistic")
    rng = np.random.default_rng(4)
    margin = rng.normal(0, 0.4, len(y)).astype(np.float32)
    d = xgb.DMatrix(x, y, base_margin=margin)
    first = xgb.train(xparams("binary:logistic"), d, 3)
    cont = xgb.train(xparams("binary:logistic"), d, 3, xgb_model=first)
    m1 = fb.train(params("binary:logistic"), x, label=y, num_boost_round=3, base_margin=margin,
                  quiet=True)
    m2 = fb.train(params("binary:logistic"), x, label=y, num_boost_round=3, base_margin=margin,
                  xgb_model=m1, quiet=True)
    np.testing.assert_array_equal(raw(m2, x, base_margin=margin), xraw(cont, x, margin))
    np.testing.assert_array_equal(raw(m2, x), xraw(cont, x))


def test_the_model_decides_the_objective_and_keeps_its_intercept(tmp_path):
    x, y = make("count:poisson")
    m1 = fb.train(params("count:poisson"), x, label=y, num_boost_round=3, quiet=True)
    # No objective in params: the model's is used (here the default would be logistic).
    m2 = fb.train({"max_depth": 3, "eta": 0.3, "nthread": 1}, x, label=y, num_boost_round=2,
                  xgb_model=m1, quiet=True)
    assert m2.objective == "count:poisson"
    assert m2.num_trees == 5
    base = lambda m, n: json.loads(saved(m, tmp_path, n).read_text())["learner"][
        "learner_model_param"]["base_score"]
    assert base(m1, "a") == base(m2, "b")
    mc = fb.train(params("multi:softprob"), *make("multi:softprob"), num_boost_round=2, quiet=True)
    mc2 = fb.train({"max_depth": 2}, *make("multi:softprob"), num_boost_round=1, xgb_model=mc,
                   quiet=True)
    assert mc2.num_class == 3 and mc2.num_trees == 9


def test_incompatible_models_are_rejected():
    x, y = make("binary:logistic", n=300)
    m = fb.train(params("binary:logistic"), x, label=y, num_boost_round=2, quiet=True)
    with pytest.raises(ValueError, match="objective"):
        fb.train(params("reg:squarederror"), x, label=y, num_boost_round=1, xgb_model=m, quiet=True)
    xm, ym = make("multi:softprob", n=300)
    with pytest.raises(ValueError, match="objective"):
        fb.train(params("multi:softprob"), xm, label=ym, num_boost_round=1, xgb_model=m, quiet=True)
    mc = fb.train(params("multi:softprob"), xm, label=ym, num_boost_round=1, quiet=True)
    with pytest.raises(ValueError, match="output"):
        fb.train(params("multi:softprob", num_class=4), xm, label=ym, num_boost_round=1,
                 xgb_model=mc, quiet=True)
    with pytest.raises(ValueError, match="features"):
        fb.train(params("binary:logistic"), x[:, :3], label=y, num_boost_round=1, xgb_model=m,
                 quiet=True)
    with pytest.raises((ValueError, RuntimeError), match="base_score"):
        fb.train(params("binary:logistic", base_score=0.123), x, label=y, num_boost_round=1,
                 xgb_model=m, quiet=True)
    with pytest.raises(TypeError, match="xgb_model"):
        fb.train(params("binary:logistic"), x, label=y, num_boost_round=1, xgb_model=3, quiet=True)
    with pytest.raises((ValueError, OSError)):
        fb.train(params("binary:logistic"), x, label=y, num_boost_round=1,
                 xgb_model="/nonexistent/model.json", quiet=True)


def test_named_data_is_bound_to_the_model_names_and_mismatches_are_rejected():
    pd = pytest.importorskip("pandas")
    x, y = make("binary:logistic", n=400)
    cols = [f"c{i}" for i in range(x.shape[1])]
    df = pd.DataFrame(x, columns=cols)
    m1 = fb.train(params("binary:logistic"), df, label=y, num_boost_round=3, quiet=True)
    assert m1.feature_names == cols
    shuffled = df[cols[::-1]]
    m2 = fb.train(params("binary:logistic"), shuffled, label=y, num_boost_round=2, xgb_model=m1,
                  quiet=True)
    assert m2.feature_names == cols, "columns are bound by name, in the model's order"
    ref = fb.train(params("binary:logistic"), df, label=y, num_boost_round=2, xgb_model=m1,
                   quiet=True)
    np.testing.assert_array_equal(raw(m2, df), raw(ref, df))
    renamed = df.rename(columns={"c0": "zzz"})
    with pytest.raises(ValueError, match="missing the features"):
        fb.train(params("binary:logistic"), renamed, label=y, num_boost_round=1, xgb_model=m1,
                 quiet=True)
    # numpy data continues a named model by position and the model keeps its names.
    m3 = fb.train(params("binary:logistic"), x, label=y, num_boost_round=1, xgb_model=m1, quiet=True)
    assert m3.feature_names == cols


def test_categorical_features_continue_and_mismatched_types_are_rejected(tmp_path):
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(1)
    n = 800
    cat = rng.integers(0, 6, n)
    num = rng.normal(size=n).astype(np.float32)
    y = ((cat % 3 == 0) ^ (num > 0.3)).astype(np.float32)
    df = pd.DataFrame({"cat": pd.Categorical(cat, categories=range(6)), "num": num})
    p = params("binary:logistic", enable_categorical=True, max_cat_to_onehot=2)
    full = fb.train(p, df, label=y, num_boost_round=6, quiet=True)
    part = fb.train(p, df, label=y, num_boost_round=3, quiet=True)
    joined = fb.train(p, df, label=y, num_boost_round=3, xgb_model=part, quiet=True)
    assert saved(full, tmp_path, "f").read_bytes() == saved(joined, tmp_path, "j").read_bytes()
    # Declaring the feature as numeric is an error, not a silent re-interpretation.
    with pytest.raises((ValueError, RuntimeError), match="feature_types"):
        fb.train(params("binary:logistic"), df[["cat", "num"]].astype(np.float32).to_numpy(),
                 label=y, num_boost_round=1, xgb_model=part, quiet=True)
    # Validation frames are re-coded onto the training categories as usual.
    res = {}
    fb.train(p, df, label=y, num_boost_round=2, xgb_model=part, evals=[(df, y, "v")],
             evals_result=res, quiet=True)
    assert len(res["v"]["logloss"]) == 2


def write_parquet(tmp_path, x, y, name):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    cols = {f"f{i}": x[:, i] for i in range(x.shape[1])}
    cols["label"] = y
    path = tmp_path / f"{name}.parquet"
    pq.write_table(pa.table(cols), path)
    return str(path)


def test_file_inputs_continue_like_arrays(tmp_path):
    x, y = make("binary:logistic", n=600)
    vx, vy = make("binary:logistic", n=200, seed=2)
    path = write_parquet(tmp_path, x, y, "train")
    vpath = write_parquet(tmp_path, vx, vy, "valid")
    p = params("binary:logistic")
    first = fb.train(p, path, label="label", num_boost_round=3, quiet=True)
    assert first.feature_names == [f"f{i}" for i in range(5)]
    res_file, res_arr = {}, {}
    from_file = fb.train(p, path, label="label", num_boost_round=3, xgb_model=first,
                         evals=[(vpath, "v")], evals_result=res_file, quiet=True)
    from_arr = fb.train(p, x, label=y, num_boost_round=3, xgb_model=first,
                        evals=[(vx, vy, "v")], evals_result=res_arr, quiet=True)
    np.testing.assert_array_equal(raw(from_file, x), raw(from_arr, x))
    assert res_file == res_arr
    # ... and from a model JSON path.
    jpath = saved(first, tmp_path, "first")
    again = fb.train(p, path, label="label", num_boost_round=3, xgb_model=str(jpath), quiet=True)
    np.testing.assert_array_equal(raw(again, x), raw(from_arr, x))


def test_file_continuation_with_a_margin_column(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    x, y = make("binary:logistic", n=500)
    margin = np.random.default_rng(8).normal(0, 0.3, len(y)).astype(np.float32)
    cols = {f"f{i}": x[:, i] for i in range(5)}
    cols.update(label=y, offset=margin)
    path = str(tmp_path / "t.parquet")
    pq.write_table(pa.table(cols), path)
    p = params("binary:logistic")
    first = fb.train(p, path, label="label", num_boost_round=3, base_margin="offset", quiet=True)
    cont_file = fb.train(p, path, label="label", num_boost_round=3, base_margin="offset",
                         xgb_model=first, quiet=True)
    cont_arr = fb.train(p, x, label=y, num_boost_round=3, base_margin=margin, xgb_model=first,
                        quiet=True)
    np.testing.assert_array_equal(raw(cont_file, x, base_margin=margin),
                                  raw(cont_arr, x, base_margin=margin))


def test_sklearn_xgb_model_passthrough():
    x, y = make("binary:logistic")
    xc1 = xgb.XGBClassifier(n_estimators=4, max_depth=3, learning_rate=0.3, n_jobs=1,
                            tree_method="hist", max_bin=255).fit(x, y)
    xc2 = xgb.XGBClassifier(n_estimators=4, max_depth=3, learning_rate=0.3, n_jobs=1,
                            tree_method="hist", max_bin=255).fit(x, y, xgb_model=xc1.get_booster())
    c1 = fb.FerrisBoostClassifier(n_estimators=4, max_depth=3, learning_rate=0.3, n_jobs=1).fit(x, y)
    c2 = fb.FerrisBoostClassifier(n_estimators=4, max_depth=3, learning_rate=0.3, n_jobs=1)
    c2.fit(x, y, xgb_model=c1)  # an estimator, a Model or a path all work
    np.testing.assert_array_equal(c2.predict_proba(x), xc2.predict_proba(x))
    assert c2.get_booster().num_trees == 8
    c3 = fb.FerrisBoostClassifier(n_estimators=4, max_depth=3, learning_rate=0.3, n_jobs=1)
    c3.fit(x, y, xgb_model=c1.get_booster())
    np.testing.assert_array_equal(c3.predict_proba(x), c2.predict_proba(x))
    xm, ym = make("multi:softprob")
    m1 = fb.FerrisBoostClassifier(n_estimators=3, max_depth=3, n_jobs=1).fit(xm, ym)
    m2 = fb.FerrisBoostClassifier(n_estimators=3, max_depth=3, n_jobs=1).fit(xm, ym, xgb_model=m1)
    assert m2.get_booster().num_trees == 18
    xr, yr = make("reg:squarederror")
    r1 = fb.FerrisBoostRegressor(n_estimators=3, max_depth=3, n_jobs=1).fit(xr, yr)
    r2 = fb.FerrisBoostRegressor(n_estimators=3, max_depth=3, n_jobs=1).fit(xr, yr, xgb_model=r1)
    assert r2.get_booster().num_trees == 6


# -------------------------------------------------------------------------- GPU
def _has_cuda():
    try:
        fb.train({"objective": "reg:squarederror", "device": "cuda", "max_depth": 2}, *make("reg:squarederror", n=64),
                 num_boost_round=1, quiet=True)
        return True
    except Exception as e:  # noqa: BLE001
        if "CUDA" in str(e) or "cuda" in str(e):
            return False
        raise


cuda = pytest.mark.skipif(not _has_cuda(), reason="no usable CUDA build or device")


@cuda
@pytest.mark.parametrize("objective", ["binary:logistic", "multi:softprob"])
def test_gpu_exact_continuation_equals_cpu_and_uninterrupted(objective, tmp_path):
    x, y = make(objective, n=2000)
    p = params(objective, subsample=0.8)
    gp = {**p, "device": "cuda", "gpu_math": "exact"}
    cpu_first = fb.train(p, x, label=y, num_boost_round=3, quiet=True)
    cpu = fb.train(p, x, label=y, num_boost_round=3, xgb_model=cpu_first, quiet=True)
    gpu_first = fb.train(gp, x, label=y, num_boost_round=3, quiet=True)
    gpu = fb.train(gp, x, label=y, num_boost_round=3, xgb_model=gpu_first, quiet=True)
    full = fb.train(gp, x, label=y, num_boost_round=6, quiet=True)
    a, b, c = (saved(m, tmp_path, n).read_bytes() for m, n in
               ((cpu, "cpu"), (gpu, "gpu"), (full, "full")))
    assert a == b == c


@cuda
def test_gpu_fast_continuation_is_deterministic(tmp_path):
    x, y = make("binary:logistic", n=2000)
    gp = {**params("binary:logistic"), "device": "cuda", "gpu_math": "fast"}
    first = fb.train(gp, x, label=y, num_boost_round=3, quiet=True)
    one = fb.train(gp, x, label=y, num_boost_round=3, xgb_model=first, quiet=True)
    two = fb.train(gp, x, label=y, num_boost_round=3, xgb_model=first, quiet=True)
    assert saved(one, tmp_path, "one").read_bytes() == saved(two, tmp_path, "two").read_bytes()
    assert one.num_trees == 6
