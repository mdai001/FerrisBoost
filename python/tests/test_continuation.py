"""Training continuation (`init_model=`).

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
* That includes categorical splits: an old tree's category set may hold codes the
  new training cuts do not cover (binned as "unseen" for the NEW trees); the old
  trees still score the training and validation starting predictions on the raw
  codes, so trees, predictions and evals_result match XGBoost's continuation.
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
    m2 = fb.train(params(objective), x, label=y, num_boost_round=5, init_model=m1, quiet=True)
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
                         init_model=str(path), quiet=True)
    from_booster = fb.train(params(objective), x, label=y, num_boost_round=4,
                            init_model=first, quiet=True)
    from_loaded = fb.train(params(objective), x, label=y, num_boost_round=4,
                           init_model=fb.Model.load_model(str(path), quiet=True), quiet=True)
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
    m2 = fb.train(params("binary:logistic"), x2, label=y2, num_boost_round=4, init_model=m1,
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
                  num_boost_round=5, init_model=m1, evals=[(vx, vy, "v")], evals_result=res,
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
    m2 = fb.train(params("binary:logistic", **p), x, label=y, num_boost_round=60, init_model=m1,
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
                  init_model=m1, quiet=True)
    assert m2.best_iteration == m1.best_iteration  # XGBoost keeps the stored attribute


@pytest.mark.parametrize("objective", ["binary:logistic", "multi:softprob", "count:poisson"])
def test_same_data_n_plus_m_equals_n_plus_m_rounds_byte_for_byte(objective, tmp_path):
    x, y = make(objective, n=700)
    p = params(objective, subsample=0.7, colsample_bytree=0.8, colsample_bylevel=0.9, seed=3)
    full = fb.train(p, x, label=y, num_boost_round=9, quiet=True)
    part = fb.train(p, x, label=y, num_boost_round=4, quiet=True)
    joined = fb.train(p, x, label=y, num_boost_round=5, init_model=part, quiet=True)
    assert saved(full, tmp_path, "full").read_bytes() == saved(joined, tmp_path, "joined").read_bytes()
    # Three legs, too.
    mid = fb.train(p, x, label=y, num_boost_round=3, init_model=part, quiet=True)
    last = fb.train(p, x, label=y, num_boost_round=2, init_model=mid, quiet=True)
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
                  init_model=m1, quiet=True)
    np.testing.assert_array_equal(raw(m2, x, base_margin=margin), xraw(cont, x, margin))
    np.testing.assert_array_equal(raw(m2, x), xraw(cont, x))


def test_the_model_decides_the_objective_and_keeps_its_intercept(tmp_path):
    x, y = make("count:poisson")
    m1 = fb.train(params("count:poisson"), x, label=y, num_boost_round=3, quiet=True)
    # No objective in params: the model's is used (here the default would be logistic).
    m2 = fb.train({"max_depth": 3, "eta": 0.3, "nthread": 1}, x, label=y, num_boost_round=2,
                  init_model=m1, quiet=True)
    assert m2.objective == "count:poisson"
    assert m2.num_trees == 5
    base = lambda m, n: json.loads(saved(m, tmp_path, n).read_text())["learner"][
        "learner_model_param"]["base_score"]
    assert base(m1, "a") == base(m2, "b")
    mc = fb.train(params("multi:softprob"), *make("multi:softprob"), num_boost_round=2, quiet=True)
    mc2 = fb.train({"max_depth": 2}, *make("multi:softprob"), num_boost_round=1, init_model=mc,
                   quiet=True)
    assert mc2.num_class == 3 and mc2.num_trees == 9


def test_incompatible_models_are_rejected():
    x, y = make("binary:logistic", n=300)
    m = fb.train(params("binary:logistic"), x, label=y, num_boost_round=2, quiet=True)
    with pytest.raises(ValueError, match="objective"):
        fb.train(params("reg:squarederror"), x, label=y, num_boost_round=1, init_model=m, quiet=True)
    xm, ym = make("multi:softprob", n=300)
    with pytest.raises(ValueError, match="objective"):
        fb.train(params("multi:softprob"), xm, label=ym, num_boost_round=1, init_model=m, quiet=True)
    mc = fb.train(params("multi:softprob"), xm, label=ym, num_boost_round=1, quiet=True)
    with pytest.raises(ValueError, match="output"):
        fb.train(params("multi:softprob", num_class=4), xm, label=ym, num_boost_round=1,
                 init_model=mc, quiet=True)
    with pytest.raises(ValueError, match="features"):
        fb.train(params("binary:logistic"), x[:, :3], label=y, num_boost_round=1, init_model=m,
                 quiet=True)
    with pytest.raises(ValueError, match="base_score"):
        fb.train(params("binary:logistic", base_score=0.123), x, label=y, num_boost_round=1,
                 init_model=m, quiet=True)
    with pytest.raises(TypeError, match="init_model"):
        fb.train(params("binary:logistic"), x, label=y, num_boost_round=1, init_model=3, quiet=True)
    with pytest.raises((ValueError, OSError)):
        fb.train(params("binary:logistic"), x, label=y, num_boost_round=1,
                 init_model="/nonexistent/model.json", quiet=True)


def test_named_data_is_bound_to_the_model_names_and_mismatches_are_rejected():
    pd = pytest.importorskip("pandas")
    x, y = make("binary:logistic", n=400)
    cols = [f"c{i}" for i in range(x.shape[1])]
    df = pd.DataFrame(x, columns=cols)
    m1 = fb.train(params("binary:logistic"), df, label=y, num_boost_round=3, quiet=True)
    assert m1.feature_names == cols
    shuffled = df[cols[::-1]]
    m2 = fb.train(params("binary:logistic"), shuffled, label=y, num_boost_round=2, init_model=m1,
                  quiet=True)
    assert m2.feature_names == cols, "columns are bound by name, in the model's order"
    ref = fb.train(params("binary:logistic"), df, label=y, num_boost_round=2, init_model=m1,
                   quiet=True)
    np.testing.assert_array_equal(raw(m2, df), raw(ref, df))
    renamed = df.rename(columns={"c0": "zzz"})
    with pytest.raises(ValueError, match="missing the features"):
        fb.train(params("binary:logistic"), renamed, label=y, num_boost_round=1, init_model=m1,
                 quiet=True)
    # numpy data continues a named model by position and the model keeps its names.
    m3 = fb.train(params("binary:logistic"), x, label=y, num_boost_round=1, init_model=m1, quiet=True)
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
    joined = fb.train(p, df, label=y, num_boost_round=3, init_model=part, quiet=True)
    assert saved(full, tmp_path, "f").read_bytes() == saved(joined, tmp_path, "j").read_bytes()
    # Declaring the feature as numeric is an error, not a silent re-interpretation.
    with pytest.raises(ValueError, match="feature_types"):
        fb.train(params("binary:logistic"), df[["cat", "num"]].astype(np.float32).to_numpy(),
                 label=y, num_boost_round=1, init_model=part, quiet=True)
    # Validation frames are re-coded onto the training categories as usual.
    res = {}
    fb.train(p, df, label=y, num_boost_round=2, init_model=part, evals=[(df, y, "v")],
             evals_result=res, quiet=True)
    assert len(res["v"]["logloss"]) == 2


# A categorical feature whose codes 0..20 occur in the first run but only 0..10 in
# the continued run: the old trees hold category sets with codes the new training
# cuts do not cover, and the validation set still contains those codes.
CAT_FT = ["c", "q", "q"]


def make_cat(n, hi, seed):
    rng = np.random.default_rng(seed)
    effect = np.random.default_rng(0).normal(0, 1.5, 21)
    cat = rng.integers(0, hi + 1, n)
    q = rng.integers(0, 16, size=(n, 2))
    x = np.column_stack([cat, q]).astype(np.float32)
    y = (effect[cat] + 0.2 * (q[:, 0] - 8) + rng.normal(0, 0.7, n) > 0).astype(np.float32)
    return x, y


def cat_trees(model, tmp_path, name):
    path = tmp_path / f"{name}.json"
    model.save_model(str(path)) if isinstance(model, xgb.Booster) \
        else model.save_model(str(path), quiet=True)
    out = []
    for t in json.loads(path.read_text())["learner"]["gradient_booster"]["model"]["trees"]:
        keep = {k: t[k] for k in ("left_children", "right_children", "split_indices", "split_type",
                                  "default_left", "categories", "categories_nodes",
                                  "categories_segments", "categories_sizes", "sum_hessian")}
        keep["leaves"] = [c for c, l in zip(t["split_conditions"], t["left_children"]) if l == -1]
        out.append(keep)
    return out


def cat_continuation(device="cpu"):
    x1, y1 = make_cat(1500, 20, 1)
    x2, y2 = make_cat(1200, 10, 2)
    xv, yv = make_cat(600, 20, 3)
    p = params("binary:logistic", max_cat_to_onehot=1, eval_metric=["logloss", "auc"],
               device=device, **({"gpu_math": "exact"} if device == "cuda" else {}))
    m1 = fb.train(p, x1, label=y1, num_boost_round=4, feature_types=CAT_FT, quiet=True)
    res = {}
    m2 = fb.train(p, x2, label=y2, num_boost_round=5, init_model=m1, feature_types=CAT_FT,
                  evals=[(x2, y2, "t"), (xv, yv, "v")], evals_result=res, quiet=True)
    return (x1, y1, x2, y2, xv, yv), m1, m2, res


def test_categorical_continuation_with_codes_outside_the_new_cuts_matches_xgboost(tmp_path):
    """The old trees score the starting predictions of the training and validation
    sets on the raw category values (category membership, as predict does), never
    on codes binned with the new cuts, where code 17 would be "unseen" (left)."""
    (x1, y1, x2, y2, xv, yv), m1, m2, res = cat_continuation()
    first = cat_trees(m1, tmp_path, "m1")
    assert any(17 in t["categories"] for t in first), "an old tree must send code 17 right"
    assert (xv[:, 0] == 17).any() and not (x2[:, 0] > 10).any()

    def d(x, y=None):
        return xgb.DMatrix(x, y, feature_types=CAT_FT, enable_categorical=True)

    xp = xparams("binary:logistic", max_cat_to_onehot=1, eval_metric=["logloss", "auc"])
    b1 = xgb.train(xp, d(x1, y1), 4)
    xres = {}
    b2 = xgb.train(xp, d(x2, y2), 5, xgb_model=b1, evals=[(d(x2, y2), "t"), (d(xv, yv), "v")],
                   evals_result=xres, verbose_eval=False)
    assert cat_trees(m1, tmp_path, "a") == cat_trees(b1, tmp_path, "b")
    assert cat_trees(m2, tmp_path, "c") == cat_trees(b2, tmp_path, "d")
    for x in (x1, x2, xv):
        np.testing.assert_array_equal(raw(m2, x), b2.predict(d(x), output_margin=True))
    for name in ("t", "v"):
        for metric in ("logloss", "auc"):
            np.testing.assert_allclose(res[name][metric], xres[name][metric], rtol=1e-6)

    # The validation history is the metric of the final model's own predictions
    # round by round; scoring the old trees with code 17 sent left would be off by
    # orders of magnitude more than the tolerance.
    def logloss(margin):
        prob = np.clip(1 / (1 + np.exp(-margin.astype(np.float64))), 1e-16, 1 - 1e-16)
        return -np.mean(yv * np.log(prob) + (1 - yv) * np.log(1 - prob))

    want = [logloss(raw(m2, xv, iteration_range=(0, 4 + r))) for r in range(1, 6)]
    np.testing.assert_allclose(res["v"]["logloss"], want, rtol=1e-6)
    unseen = xv.copy()
    unseen[unseen[:, 0] == 17, 0] = -1.0  # invalid code: left at every node
    shift = raw(m1, unseen) - raw(m1, xv)
    assert np.count_nonzero(shift) > 0
    wrong = [logloss(raw(m2, xv, iteration_range=(0, 4 + r)) + shift) for r in range(1, 6)]
    assert np.max(np.abs(np.subtract(wrong, want))) > 1e-3


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
    from_file = fb.train(p, path, label="label", num_boost_round=3, init_model=first,
                         evals=[(vpath, "v")], evals_result=res_file, quiet=True)
    from_arr = fb.train(p, x, label=y, num_boost_round=3, init_model=first,
                        evals=[(vx, vy, "v")], evals_result=res_arr, quiet=True)
    np.testing.assert_array_equal(raw(from_file, x), raw(from_arr, x))
    assert res_file == res_arr
    # ... and from a model JSON path.
    jpath = saved(first, tmp_path, "first")
    again = fb.train(p, path, label="label", num_boost_round=3, init_model=str(jpath), quiet=True)
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
                         init_model=first, quiet=True)
    cont_arr = fb.train(p, x, label=y, num_boost_round=3, base_margin=margin, init_model=first,
                        quiet=True)
    np.testing.assert_array_equal(raw(cont_file, x, base_margin=margin),
                                  raw(cont_arr, x, base_margin=margin))


def test_sklearn_init_model_passthrough():
    x, y = make("binary:logistic")
    xc1 = xgb.XGBClassifier(n_estimators=4, max_depth=3, learning_rate=0.3, n_jobs=1,
                            tree_method="hist", max_bin=255).fit(x, y)
    xc2 = xgb.XGBClassifier(n_estimators=4, max_depth=3, learning_rate=0.3, n_jobs=1,
                            tree_method="hist", max_bin=255).fit(x, y, xgb_model=xc1.get_booster())
    c1 = fb.FerrisBoostClassifier(n_estimators=4, max_depth=3, learning_rate=0.3, n_jobs=1).fit(x, y)
    c2 = fb.FerrisBoostClassifier(n_estimators=4, max_depth=3, learning_rate=0.3, n_jobs=1)
    c2.fit(x, y, init_model=c1)  # an estimator, a Model or a path all work
    np.testing.assert_array_equal(c2.predict_proba(x), xc2.predict_proba(x))
    assert c2.get_booster().num_trees == 8
    c3 = fb.FerrisBoostClassifier(n_estimators=4, max_depth=3, learning_rate=0.3, n_jobs=1)
    c3.fit(x, y, init_model=c1.get_booster())
    np.testing.assert_array_equal(c3.predict_proba(x), c2.predict_proba(x))
    xm, ym = make("multi:softprob")
    m1 = fb.FerrisBoostClassifier(n_estimators=3, max_depth=3, n_jobs=1).fit(xm, ym)
    m2 = fb.FerrisBoostClassifier(n_estimators=3, max_depth=3, n_jobs=1).fit(xm, ym, init_model=m1)
    assert m2.get_booster().num_trees == 18
    xr, yr = make("reg:squarederror")
    r1 = fb.FerrisBoostRegressor(n_estimators=3, max_depth=3, n_jobs=1).fit(xr, yr)
    r2 = fb.FerrisBoostRegressor(n_estimators=3, max_depth=3, n_jobs=1).fit(xr, yr, init_model=r1)
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
    cpu = fb.train(p, x, label=y, num_boost_round=3, init_model=cpu_first, quiet=True)
    gpu_first = fb.train(gp, x, label=y, num_boost_round=3, quiet=True)
    gpu = fb.train(gp, x, label=y, num_boost_round=3, init_model=gpu_first, quiet=True)
    full = fb.train(gp, x, label=y, num_boost_round=6, quiet=True)
    a, b, c = (saved(m, tmp_path, n).read_bytes() for m, n in
               ((cpu, "cpu"), (gpu, "gpu"), (full, "full")))
    assert a == b == c


@cuda
def test_gpu_exact_categorical_continuation_with_codes_outside_the_new_cuts_equals_cpu(tmp_path):
    _, c1, c2, cres = cat_continuation("cpu")
    _, g1, g2, gres = cat_continuation("cuda")
    for (c, g), n in (((c1, g1), "1"), ((c2, g2), "2")):
        assert saved(c, tmp_path, "c" + n).read_bytes() == saved(g, tmp_path, "g" + n).read_bytes()
    assert cres == gres


@cuda
def test_gpu_fast_continuation_is_deterministic(tmp_path):
    x, y = make("binary:logistic", n=2000)
    gp = {**params("binary:logistic"), "device": "cuda", "gpu_math": "fast"}
    first = fb.train(gp, x, label=y, num_boost_round=3, quiet=True)
    one = fb.train(gp, x, label=y, num_boost_round=3, init_model=first, quiet=True)
    two = fb.train(gp, x, label=y, num_boost_round=3, init_model=first, quiet=True)
    assert saved(one, tmp_path, "one").read_bytes() == saved(two, tmp_path, "two").read_bytes()
    assert one.num_trees == 6


# ---------------------------------------------------------------------- ranking
def make_rank(seed=0, sizes=(3, 5, 8, 13, 21, 34, 40, 7, 9, 30)):
    rng = np.random.default_rng(seed)
    n = int(sum(sizes))
    x = rng.integers(0, 12, size=(n, 6)).astype(np.float32)
    s = x[:, 0] * 0.5 + x[:, 1] * 0.3 - x[:, 2] * 0.2 + rng.normal(0, 1.5, n)
    y = np.clip(np.round((s - s.min()) / (s.max() - s.min()) * 4), 0, 4).astype(np.float32)
    return x, y, np.repeat(np.arange(len(sizes)), sizes)


@pytest.mark.parametrize("objective", ["rank:ndcg", "rank:pairwise", "rank:map"])
def test_ranking_model_continuation_matches_xgboost(objective, tmp_path):
    x, y, qid = make_rank()
    if objective == "rank:map":
        y = (y > 1).astype(np.float32)
    d = xgb.DMatrix(x, y, qid=qid)
    xp = {"objective": objective, "tree_method": "hist", "nthread": 1, "max_depth": 3, "max_bin": 16}
    first = xgb.train(xp, d, 3)
    cont = xgb.train(xp, d, 3, xgb_model=first)
    p = {"objective": objective, "nthread": 1, "max_depth": 3, "max_bin": 16}
    m1 = fb.train(p, x, label=y, qid=qid, num_boost_round=3, quiet=True)
    m2 = fb.train(p, x, label=y, qid=qid, num_boost_round=3, init_model=m1, quiet=True)
    np.testing.assert_array_equal(raw(m2, x), xraw(cont, x))
    # Same data: 3 + 3 rounds are the 6 uninterrupted rounds, byte for byte.
    full = fb.train(p, x, label=y, qid=qid, num_boost_round=6, quiet=True)
    assert saved(full, tmp_path, "f").read_bytes() == saved(m2, tmp_path, "j").read_bytes()


def test_ranking_continuation_with_validation_sets_and_margins():
    x, y, qid = make_rank()
    vx, vy, vq = make_rank(seed=4, sizes=(6, 11, 17, 25))
    margin = np.random.default_rng(2).normal(0, 0.4, len(y)).astype(np.float32)
    p = {"objective": "rank:ndcg", "nthread": 1, "max_depth": 3, "max_bin": 16}
    m1 = fb.train(p, x, label=y, qid=qid, base_margin=margin, num_boost_round=3, quiet=True)
    res = {}
    m2 = fb.train(p, x, label=y, qid=qid, base_margin=margin, num_boost_round=3, init_model=m1,
                  evals=[(vx, vy, "v")], eval_qid=[vq], evals_result=res, quiet=True)
    assert m2.num_trees == 6
    assert all(len(v) == 3 for v in res["v"].values())
    d = xgb.DMatrix(x, y, qid=qid, base_margin=margin)
    xp = {"objective": "rank:ndcg", "tree_method": "hist", "nthread": 1, "max_depth": 3,
          "max_bin": 16}
    cont = xgb.train(xp, d, 3, xgb_model=xgb.train(xp, d, 3))
    np.testing.assert_array_equal(raw(m2, x, base_margin=margin), xraw(cont, x, margin))


def test_unbiased_ranking_cannot_be_continued():
    # XGBoost blocks continuing unbiased LambdaMART; so do we, with a clear error.
    rng = np.random.default_rng(41)
    x = rng.integers(0, 8, (400, 3)).astype(np.float32)
    y = rng.integers(0, 3, 400).astype(np.float32)
    qid = np.repeat(np.arange(20), 20)
    p = {"objective": "rank:ndcg", "lambdarank_unbiased": True, "max_depth": 3}
    m = fb.train(p, x, label=y, qid=qid, num_boost_round=2, quiet=True)
    with pytest.raises(Exception, match="unbiased"):
        fb.train(p, x, label=y, qid=qid, num_boost_round=2, init_model=m, quiet=True)
    with pytest.raises(Exception, match="unbiased"):
        fb.train({**p, "lambdarank_unbiased": False}, x, label=y, qid=qid, num_boost_round=2,
                 init_model=m, quiet=True)


def test_xgb_model_is_an_alias_of_init_model(tmp_path):
    x, y = make("binary:logistic")
    m1 = fb.train(params("binary:logistic"), x, label=y, num_boost_round=3, quiet=True)
    a = fb.train(params("binary:logistic"), x, label=y, num_boost_round=2, init_model=m1, quiet=True)
    b = fb.train(params("binary:logistic"), x, label=y, num_boost_round=2, xgb_model=m1, quiet=True)
    assert saved(a, tmp_path, "a").read_bytes() == saved(b, tmp_path, "b").read_bytes()
    with pytest.raises(TypeError, match="not both"):
        fb.train(params("binary:logistic"), x, label=y, num_boost_round=1, init_model=m1,
                 xgb_model=m1, quiet=True)
    c1 = fb.FerrisBoostClassifier(n_estimators=2, max_depth=3, n_jobs=1).fit(x, y)
    ca = fb.FerrisBoostClassifier(n_estimators=2, max_depth=3, n_jobs=1).fit(x, y, init_model=c1)
    cb = fb.FerrisBoostClassifier(n_estimators=2, max_depth=3, n_jobs=1).fit(x, y, xgb_model=c1)
    np.testing.assert_array_equal(ca.predict_proba(x), cb.predict_proba(x))
    with pytest.raises(TypeError, match="not both"):
        fb.FerrisBoostRegressor(n_estimators=1).fit(x, y, init_model=m1, xgb_model=m1)
