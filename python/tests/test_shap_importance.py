"""get_score, pred_contribs / approx_contribs / pred_interactions / pred_leaf.

Everything is checked against XGBoost on the *same* model (FerrisBoost model
saved as JSON and loaded by XGBoost, and XGBoost models loaded by FerrisBoost).
Data are small and synthetic; no local datasets are used.
"""
import numpy as np
import pytest

import ferrisboost as fb

xgb = pytest.importorskip("xgboost")

# Contributions are float32 sums of float32 products evaluated in XGBoost's
# own order, so the two sides agree to float32 rounding of values of order 1.
SHAP_ATOL = 2e-5
SCORE_RTOL = 2e-6


def make(n=1500, m=6, k=None, seed=0, missing=0.08):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, m)).astype(np.float32)
    x[rng.random(x.shape) < missing] = np.nan
    s = (np.nan_to_num(x[:, 0]) - 0.7 * np.nan_to_num(x[:, 1]) * np.nan_to_num(x[:, 2])
         + rng.normal(0, 0.5, n))
    if k:
        edges = np.quantile(s, np.linspace(0, 1, k + 1)[1:-1])
        return x, np.digitize(s, edges).astype(np.float32)
    return x, s.astype(np.float32)


def train_fb(params, x, y, w=None, rounds=6):
    return fb.train({**params, "nthread": 1}, x, label=y, weight=w, num_boost_round=rounds,
                    quiet=True)


def booster_of(model, tmp_path, name="m.json"):
    path = str(tmp_path / name)
    model.save_model(path)
    return xgb.Booster(model_file=path)


CASES = {
    "logistic": ({"objective": "binary:logistic", "max_depth": 5}, 1),
    "squarederror": ({"objective": "reg:squarederror", "max_depth": 6}, 1),
    "poisson": ({"objective": "count:poisson", "max_depth": 4}, 1),
    "softprob": ({"objective": "multi:softprob", "num_class": 3, "max_depth": 4}, 3),
}


def case_data(name):
    params, k = CASES[name]
    if k > 1:
        x, y = make(k=k, seed=2)
    else:
        x, y = make(seed=1)
        if name == "logistic":
            y = (y > np.median(y)).astype(np.float32)
        elif name == "poisson":
            y = np.random.default_rng(5).poisson(np.exp(0.4 * np.nan_to_num(x[:, 0]))).astype(
                np.float32)
    return params, k, x, y


@pytest.fixture(scope="module", params=sorted(CASES))
def pair(request, tmp_path_factory):
    params, k, x, y = case_data(request.param)
    model = train_fb(params, x, y)
    return model, booster_of(model, tmp_path_factory.mktemp(request.param)), x, k


@pytest.mark.parametrize("importance_type",
                         ["weight", "gain", "cover", "total_gain", "total_cover"])
def test_get_score_matches_xgboost(pair, importance_type):
    model, bst, _, _ = pair
    ours = model.get_score(importance_type=importance_type)
    ref = bst.get_score(importance_type=importance_type)
    assert list(ours) == list(ref)
    np.testing.assert_allclose(list(ours.values()), list(ref.values()), rtol=SCORE_RTOL)


def test_get_score_default_is_weight_and_names_are_f_index():
    x, y = make()
    model = train_fb({"objective": "reg:squarederror", "max_depth": 3}, x, y)
    score = model.get_score()
    assert score == model.get_score("weight")
    assert all(name.startswith("f") and name[1:].isdigit() for name in score)
    # weight is the number of splits on the feature (a whole number); features
    # come in index order
    assert all(v >= 1 and v == int(v) for v in score.values())
    assert list(score) == sorted(score, key=lambda n: int(n[1:]))


def test_get_score_unknown_type_raises():
    x, y = make(n=200)
    model = train_fb({"objective": "reg:squarederror"}, x, y, rounds=2)
    with pytest.raises(ValueError, match="importance_type"):
        model.get_score("permutation")


def test_pred_leaf_matches_xgboost(pair):
    model, bst, x, k = pair
    ours = model.predict(x, pred_leaf=True)
    ref = bst.predict(xgb.DMatrix(x), pred_leaf=True)
    assert ours.dtype == np.float32
    assert ours.shape == ref.shape == (len(x), model.num_trees)
    np.testing.assert_array_equal(ours, ref)


@pytest.mark.parametrize("approx", [False, True])
def test_contribs_match_xgboost_and_sum_to_margin(pair, approx):
    model, bst, x, k = pair
    ours = model.predict(x, pred_contribs=True, approx_contribs=approx)
    ref = bst.predict(xgb.DMatrix(x), pred_contribs=True, approx_contribs=approx)
    expected_shape = (len(x), x.shape[1] + 1) if k == 1 else (len(x), k, x.shape[1] + 1)
    assert ours.shape == ref.shape == expected_shape
    assert ours.dtype == np.float32
    np.testing.assert_allclose(ours, ref, rtol=0, atol=SHAP_ATOL)
    margin = model.predict(x, output_margin=True)
    np.testing.assert_allclose(ours.sum(axis=-1), margin, rtol=0, atol=2e-4)


def test_interactions_match_xgboost(pair):
    model, bst, x, k = pair
    x = x[:200]
    ours = model.predict(x, pred_interactions=True)
    ref = bst.predict(xgb.DMatrix(x), pred_interactions=True)
    f1 = x.shape[1] + 1
    assert ours.shape == ref.shape == ((len(x), f1, f1) if k == 1 else (len(x), k, f1, f1))
    np.testing.assert_allclose(ours, ref, rtol=0, atol=SHAP_ATOL)
    # symmetric, and row sums are the SHAP values
    np.testing.assert_allclose(ours, np.swapaxes(ours, -1, -2), atol=1e-6)
    contribs = model.predict(x, pred_contribs=True)
    np.testing.assert_allclose(ours.sum(axis=-1), contribs, atol=1e-4)


def test_n_trees_limits_contribs_and_leaves(pair):
    model, bst, x, k = pair
    rounds = 3
    d = xgb.DMatrix(x)
    for approx in (False, True):
        ours = model.predict(x, pred_contribs=True, approx_contribs=approx, n_trees=rounds * k)
        ref = bst.predict(d, pred_contribs=True, approx_contribs=approx, iteration_range=(0, rounds))
        np.testing.assert_allclose(ours, ref, rtol=0, atol=SHAP_ATOL)
        full_margin = model.predict(x, output_margin=True, n_trees=rounds * k)
        np.testing.assert_allclose(ours.sum(axis=-1), full_margin, atol=2e-4)
    leaves = model.predict(x, pred_leaf=True, n_trees=rounds * k)
    ref = bst.predict(d, pred_leaf=True, iteration_range=(0, rounds))
    assert leaves.shape == (len(x), rounds * k)
    np.testing.assert_array_equal(leaves, ref)


@pytest.mark.parametrize("max_depth", [1, 2, 3, 4, 6, 8])
def test_depth_sweep_with_weights_and_missing(max_depth, tmp_path):
    x, y = make(n=1200, seed=11, missing=0.2)
    y = (y > 0).astype(np.float32)
    w = np.random.default_rng(3).uniform(0.2, 3, len(y)).astype(np.float32)
    model = train_fb({"objective": "binary:logistic", "max_depth": max_depth}, x, y, w=w, rounds=5)
    bst = booster_of(model, tmp_path)
    d = xgb.DMatrix(x)
    np.testing.assert_allclose(model.predict(x, pred_contribs=True),
                               bst.predict(d, pred_contribs=True), rtol=0, atol=SHAP_ATOL)
    np.testing.assert_array_equal(model.predict(x, pred_leaf=True), bst.predict(d, pred_leaf=True))
    for t in ["weight", "gain", "cover"]:
        assert model.get_score(t).keys() == bst.get_score(importance_type=t).keys()
    xi = x[:100]
    np.testing.assert_allclose(model.predict(xi, pred_interactions=True),
                               bst.predict(xgb.DMatrix(xi), pred_interactions=True),
                               rtol=0, atol=SHAP_ATOL)


@pytest.mark.parametrize("name", ["binary", "multi", "lossguide"])
def test_xgboost_trained_model_loaded_into_ferrisboost(name, tmp_path):
    if name == "multi":
        x, y = make(seed=4, k=4)
        params = {"objective": "multi:softprob", "num_class": 4, "max_depth": 5}
        k = 4
    else:
        x, y = make(seed=3)
        y = (y > 0).astype(np.float32)
        params = {"objective": "binary:logistic", "max_depth": 7}
        if name == "lossguide":
            params.update(grow_policy="lossguide", max_leaves=24, max_depth=0)
        k = 1
    bst = xgb.train({**params, "tree_method": "hist", "nthread": 1}, xgb.DMatrix(x, label=y), 6)
    path = str(tmp_path / "x.json")
    bst.save_model(path)
    model = fb.Model.load_model(path, quiet=True)
    d = xgb.DMatrix(x)
    for t in ["weight", "gain", "cover", "total_gain", "total_cover"]:
        ours, ref = model.get_score(t), bst.get_score(importance_type=t)
        assert list(ours) == list(ref)
        np.testing.assert_allclose(list(ours.values()), list(ref.values()), rtol=SCORE_RTOL)
    np.testing.assert_array_equal(model.predict(x, pred_leaf=True), bst.predict(d, pred_leaf=True))
    for approx in (False, True):
        np.testing.assert_allclose(
            model.predict(x, pred_contribs=True, approx_contribs=approx),
            bst.predict(d, pred_contribs=True, approx_contribs=approx), rtol=0, atol=SHAP_ATOL)
    if name != "lossguide":
        xi = x[:100]
        np.testing.assert_allclose(model.predict(xi, pred_interactions=True),
                                   bst.predict(xgb.DMatrix(xi), pred_interactions=True),
                                   rtol=0, atol=SHAP_ATOL)


def test_trained_by_both_libraries_gives_same_explanations():
    # Low-cardinality features: binning is lossless, so both libraries grow the
    # same trees with the same node numbering.
    rng = np.random.default_rng(1)
    x = rng.integers(0, 9, size=(2000, 6)).astype(np.float32)
    x[rng.random(x.shape) < 0.05] = np.nan
    s = np.nan_to_num(x[:, 0], nan=4) - np.nan_to_num(x[:, 1], nan=4) + rng.normal(0, 2, len(x))
    y = (s > 0).astype(np.float32)
    p = {"objective": "binary:logistic", "max_depth": 6, "eta": 0.3, "nthread": 1}
    bst = xgb.train({**p, "tree_method": "hist"}, xgb.DMatrix(x, label=y), 6)
    model = fb.train(p, x, label=y, num_boost_round=6, quiet=True)
    d = xgb.DMatrix(x)
    np.testing.assert_array_equal(model.predict(x, pred_leaf=True), bst.predict(d, pred_leaf=True))
    np.testing.assert_allclose(model.predict(x, pred_contribs=True),
                               bst.predict(d, pred_contribs=True), rtol=0, atol=SHAP_ATOL)
    for t in ["weight", "gain", "cover", "total_gain", "total_cover"]:
        ours, ref = model.get_score(t), bst.get_score(importance_type=t)
        assert list(ours) == list(ref)
        np.testing.assert_allclose(list(ours.values()), list(ref.values()), rtol=1e-5)


def test_explain_results_do_not_depend_on_thread_count():
    x, y = make(n=900, seed=8)
    model = train_fb({"objective": "reg:squarederror", "max_depth": 5}, x, y, rounds=8)
    for kw in ({"pred_contribs": True}, {"pred_contribs": True, "approx_contribs": True},
               {"pred_interactions": True}, {"pred_leaf": True}):
        ref = model.predict(x[:300], predict_threads=1, **kw)
        for threads in (0, 3, 8):
            np.testing.assert_array_equal(model.predict(x[:300], predict_threads=threads, **kw), ref)


def test_non_contiguous_input_matches_contiguous():
    x, y = make(n=300, seed=9)
    model = train_fb({"objective": "reg:squarederror", "max_depth": 4}, x, y)
    f_order = np.asfortranarray(x)
    sliced = np.zeros((600, x.shape[1]), np.float32)
    sliced[::2] = x
    for kw in ({"pred_contribs": True}, {"pred_leaf": True}):
        ref = model.predict(x, **kw)
        np.testing.assert_array_equal(model.predict(f_order, **kw), ref)
        np.testing.assert_array_equal(model.predict(sliced[::2], **kw), ref)


def test_file_input_matches_numpy(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    x, y = make(n=800, seed=10)
    cols = {f"feat_{i}": x[:, i] for i in range(x.shape[1])}
    path = str(tmp_path / "train.parquet")
    pq.write_table(pa.table({**cols, "label": y}), path)
    model = fb.train({"objective": "reg:squarederror", "max_depth": 4, "nthread": 1}, path,
                     label="label", num_boost_round=5, quiet=True)
    # names come from the file and are what get_score reports
    assert set(model.get_score()) <= set(cols)
    bst = booster_of(model, tmp_path)
    assert list(model.get_score("gain")) == list(bst.get_score(importance_type="gain"))
    for kw in ({"pred_contribs": True}, {"pred_contribs": True, "approx_contribs": True},
               {"pred_leaf": True}):
        from_file = model.predict(path, **kw)
        from_array = model.predict(x, **kw)
        np.testing.assert_array_equal(from_file, from_array)
    inter_file = model.predict(path, pred_interactions=True)
    assert inter_file.shape == (len(x), x.shape[1] + 1, x.shape[1] + 1)


def test_constant_models_and_root_only_trees():
    x = np.random.default_rng(0).normal(size=(50, 3)).astype(np.float32)
    y = np.full(50, 2.0, np.float32)
    model = train_fb({"objective": "reg:squarederror", "max_depth": 3}, x, y, rounds=2)
    contribs = model.predict(x, pred_contribs=True)
    assert contribs.shape == (50, 4)
    np.testing.assert_allclose(contribs.sum(axis=1), model.predict(x, output_margin=True), atol=1e-6)
    assert model.get_score() == {}


def test_empty_input_shapes():
    x, y = make(n=200)
    model = train_fb({"objective": "reg:squarederror", "max_depth": 3}, x, y, rounds=3)
    empty = np.zeros((0, x.shape[1]), np.float32)
    assert model.predict(empty, pred_contribs=True).shape == (0, x.shape[1] + 1)
    assert model.predict(empty, pred_leaf=True).shape == (0, 3)
    assert model.predict(empty, pred_interactions=True).shape == (0, x.shape[1] + 1, x.shape[1] + 1)


def test_argument_errors():
    x, y = make(n=300)
    model = train_fb({"objective": "reg:squarederror", "max_depth": 3}, x, y, rounds=3)
    with pytest.raises(ValueError, match="one type of prediction"):
        model.predict(x, pred_contribs=True, pred_leaf=True)
    with pytest.raises(ValueError, match="one type of prediction"):
        model.predict(x, pred_contribs=True, output_margin=True)
    with pytest.raises(ValueError, match="approx_contribs"):
        model.predict(x, pred_interactions=True, approx_contribs=True)
    with pytest.raises(ValueError, match="features"):
        model.predict(x[:, :3], pred_contribs=True)
    bad = x.copy()
    bad[0, 1] = np.finfo(np.float32).max
    with pytest.raises(ValueError, match="f32::MAX"):
        model.predict(bad, pred_contribs=True)
    # approx_contribs alone is ignored, as in XGBoost
    np.testing.assert_array_equal(model.predict(x, approx_contribs=True), model.predict(x))


def test_use_gpu_is_ignored_for_explanations():
    x, y = make(n=300)
    model = train_fb({"objective": "reg:squarederror", "max_depth": 3}, x, y, rounds=3)
    np.testing.assert_array_equal(model.predict(x, pred_contribs=True, use_gpu=True),
                                  model.predict(x, pred_contribs=True))


# ---------------------------------------------------------------------------
# strict_shape: XGBoost's CalcPredictShape, case by case
# ---------------------------------------------------------------------------

STRICT_KINDS = {
    "value": {},
    "margin": {"output_margin": True},
    "contribs": {"pred_contribs": True},
    "approx": {"pred_contribs": True, "approx_contribs": True},
    "interactions": {"pred_interactions": True},
    "leaf": {"pred_leaf": True},
}


def _ranking_data(seed=2):
    rng = np.random.default_rng(seed)
    sizes = rng.integers(8, 30, 40)
    qid = np.repeat(np.arange(len(sizes)), sizes)
    x = rng.normal(size=(len(qid), 5)).astype(np.float32)
    y = np.clip((x[:, 0] * 1.5 + rng.normal(0, 1, len(qid)) + 1.5).round(), 0, 4).astype(np.float32)
    return x, y, qid


@pytest.fixture(scope="module", params=["logistic", "squarederror", "softprob", "softmax",
                                        "quantiles", "single_quantile", "ranking"])
def strict_pair(request, tmp_path_factory):
    name = request.param
    path = tmp_path_factory.mktemp(name)
    kw = {}
    if name == "ranking":
        x, y, qid = _ranking_data()
        params = {"objective": "rank:ndcg", "max_depth": 3}
        kw["qid"] = qid
    elif name in ("softprob", "softmax"):
        x, y = make(k=3, seed=2)
        params = {"objective": f"multi:{name}", "num_class": 3, "max_depth": 3}
    elif name == "quantiles":
        x, y = make(seed=1)
        params = {"objective": "reg:quantileerror", "quantile_alpha": [0.2, 0.5, 0.8],
                  "max_depth": 3}
    elif name == "single_quantile":
        x, y = make(seed=1)
        params = {"objective": "reg:quantileerror", "quantile_alpha": 0.5, "max_depth": 3}
    else:
        x, y = make(seed=1)
        if name == "logistic":
            y = (y > np.median(y)).astype(np.float32)
        params = {"objective": f"{'binary:logistic' if name == 'logistic' else 'reg:squarederror'}",
                  "max_depth": 3}
    model = fb.train({**params, "nthread": 1}, x[:600], label=y[:600], num_boost_round=4,
                     quiet=True, **{k: v[:600] for k, v in kw.items()})
    return model, booster_of(model, path), x[:150], name


@pytest.mark.parametrize("kind", sorted(STRICT_KINDS))
def test_strict_shape_matches_xgboost(strict_pair, kind):
    model, bst, x, name = strict_pair
    flags = STRICT_KINDS[kind]
    d = xgb.DMatrix(x)
    ours = model.predict(x, strict_shape=True, **flags)
    ref = bst.predict(d, strict_shape=True, **flags)
    assert ours.shape == ref.shape, (name, kind)
    assert ours.dtype == np.float32
    atol = 1e-5 if "pred_interactions" in flags or "pred_contribs" in flags else 1e-6
    np.testing.assert_allclose(ours, ref, rtol=1e-5, atol=atol)
    # strict_shape only reshapes: the values are the default ones
    plain = model.predict(x, **flags)
    np.testing.assert_array_equal(ours.reshape(-1), plain.reshape(-1))
    # and the default shapes still match XGBoost's
    assert plain.shape == bst.predict(d, **flags).shape


@pytest.mark.parametrize("kind", ["value", "margin", "contribs", "leaf"])
def test_strict_shape_with_iteration_range(strict_pair, kind):
    model, bst, x, name = strict_pair
    flags = STRICT_KINDS[kind]
    d = xgb.DMatrix(x)
    # XGBoost's explanations only support ranges that begin at 0 (it asks for
    # model slicing instead); values and margins take any range.
    for rng in ((0, 2), (0, 3)) + ((1, 3),) * (kind in ("value", "margin")):
        ours = model.predict(x, strict_shape=True, iteration_range=rng, **flags)
        ref = bst.predict(d, strict_shape=True, iteration_range=rng, **flags)
        assert ours.shape == ref.shape, (name, kind, rng)
        np.testing.assert_allclose(ours, ref, rtol=1e-5, atol=1e-5)


def test_strict_shape_leaf_forest_axis_and_n_trees():
    x, y = make(k=3, seed=2)
    model = train_fb({"objective": "multi:softprob", "num_class": 3, "max_depth": 3}, x, y,
                     rounds=4)
    assert model.predict(x, pred_leaf=True, strict_shape=True).shape == (len(x), 4, 3, 1)
    assert model.predict(x, pred_leaf=True, strict_shape=True, n_trees=6).shape == (len(x), 2, 3, 1)
    with pytest.raises(ValueError, match="whole boosting rounds"):
        model.predict(x, pred_leaf=True, strict_shape=True, n_trees=5)


def test_strict_shape_file_input_and_empty_flags(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    x, y = make(n=400, seed=12)
    cols = {f"feat_{i}": x[:, i] for i in range(x.shape[1])}
    path = str(tmp_path / "t.parquet")
    pq.write_table(pa.table({**cols, "label": y}), path)
    model = fb.train({"objective": "reg:squarederror", "max_depth": 3, "nthread": 1}, path,
                     label="label", num_boost_round=3, quiet=True)
    assert model.predict(path, strict_shape=True).shape == (len(x), 1)
    assert model.predict(path, strict_shape=True, pred_contribs=True).shape == (len(x), 1, 7)
    # default stays flat
    assert model.predict(path).shape == (len(x),)


# ---------------------------------------------------------------------------
# GPU explanations: bit-identical to the CPU path
# ---------------------------------------------------------------------------


def _has_cuda_predict():
    try:
        x, y = make(n=64, seed=0)
        m = train_fb({"objective": "reg:squarederror", "max_depth": 2}, x, y, rounds=1)
        m.predict(x, pred_contribs=True, use_gpu=True)
        return True
    except Exception as e:  # noqa: BLE001
        if "CUDA" in str(e) or "cuda" in str(e) or "GPU" in str(e):
            return False
        raise


gpu = pytest.mark.skipif(not _has_cuda_predict(), reason="no usable CUDA build or device")

EXPLAIN_KINDS = {
    "contribs": {"pred_contribs": True},
    "approx": {"pred_contribs": True, "approx_contribs": True},
    "interactions": {"pred_interactions": True},
    "leaf": {"pred_leaf": True},
}


def bits(a):
    return np.ascontiguousarray(a).view(np.uint32)


def assert_identical(cpu, dev, what=""):
    assert cpu.shape == dev.shape and cpu.dtype == dev.dtype, what
    np.testing.assert_array_equal(bits(cpu), bits(dev), err_msg=what)


@gpu
@pytest.mark.parametrize("kind", sorted(EXPLAIN_KINDS))
def test_gpu_explanations_are_bit_identical_to_cpu(pair, kind):
    model, bst, x, k = pair
    x = x[:300]
    flags = EXPLAIN_KINDS[kind]
    cpu = model.predict(x, **flags)
    dev = model.predict(x, use_gpu=True, **flags)
    assert_identical(cpu, dev, kind)
    # and deterministic: a second launch gives the same bits
    assert_identical(dev, model.predict(x, use_gpu=True, **flags), kind)


@gpu
@pytest.mark.parametrize("max_depth", [0, 1, 2, 4, 6, 8, 10])
def test_gpu_depth_sweep_matches_cpu_and_xgboost(max_depth, tmp_path):
    x, y = make(n=900, seed=20 + max_depth, missing=0.2)
    y = (y > 0).astype(np.float32)
    model = train_fb({"objective": "binary:logistic", "max_depth": max_depth}, x, y, rounds=5)
    bst = booster_of(model, tmp_path)
    for kind in ("contribs", "approx"):
        flags = EXPLAIN_KINDS[kind]
        dev = model.predict(x, use_gpu=True, **flags)
        assert_identical(model.predict(x, **flags), dev, kind)
        np.testing.assert_allclose(dev, bst.predict(xgb.DMatrix(x), **flags), rtol=0, atol=SHAP_ATOL)
    xi = x[:60]
    dev = model.predict(xi, use_gpu=True, pred_interactions=True)
    assert_identical(model.predict(xi, pred_interactions=True), dev, "interactions")
    np.testing.assert_allclose(dev, bst.predict(xgb.DMatrix(xi), pred_interactions=True),
                               rtol=0, atol=SHAP_ATOL)
    assert_identical(model.predict(x, pred_leaf=True), model.predict(x, use_gpu=True, pred_leaf=True))


@gpu
def test_gpu_repeated_features_on_a_path():
    # Many splits on few features force the previous-occurrence bookkeeping.
    rng = np.random.default_rng(31)
    x = rng.normal(size=(1500, 2)).astype(np.float32)
    y = (np.sin(3 * x[:, 0]) + np.cos(2 * x[:, 1]) + rng.normal(0, 0.1, len(x))).astype(np.float32)
    model = train_fb({"objective": "reg:squarederror", "max_depth": 9}, x, y, rounds=4)
    for kind, flags in EXPLAIN_KINDS.items():
        assert_identical(model.predict(x[:200], **flags),
                         model.predict(x[:200], use_gpu=True, **flags), kind)


@gpu
def test_gpu_base_margin_and_ranges():
    x, y = make(k=3, seed=2)
    model = train_fb({"objective": "multi:softprob", "num_class": 3, "max_depth": 4}, x, y)
    margin = np.random.default_rng(1).normal(size=(len(x), 3)).astype(np.float32)
    xs, ms = x[:200], margin[:200]
    for kind in ("contribs", "approx", "interactions"):
        flags = EXPLAIN_KINDS[kind]
        for extra in ({"base_margin": ms}, {"iteration_range": (1, 4)}, {"n_trees": 7}):
            assert_identical(model.predict(xs, **flags, **extra),
                             model.predict(xs, use_gpu=True, **flags, **extra), (kind, extra))
    # base_margin lands in the bias column
    dev = model.predict(xs, use_gpu=True, pred_contribs=True, base_margin=ms)
    np.testing.assert_allclose(dev.sum(axis=-1), model.predict(xs, output_margin=True, base_margin=ms),
                               atol=2e-4)


@gpu
def test_gpu_categorical_splits(tmp_path):
    rng = np.random.default_rng(5)
    n = 1500
    num = rng.normal(size=(n, 2)).astype(np.float32)
    cat1 = rng.integers(0, 9, n).astype(np.float32)
    cat2 = rng.integers(0, 40, n).astype(np.float32)
    x = np.column_stack([num, cat1, cat2])
    x[rng.random(x.shape) < 0.07] = np.nan
    eff1, eff2 = rng.normal(0, 1, 9), rng.normal(0, 1, 40)
    y = (num[:, 0] + eff1[np.nan_to_num(cat1).astype(int)] + eff2[np.nan_to_num(cat2).astype(int)]
         + rng.normal(0, 0.3, n)).astype(np.float32)
    model = fb.train({"objective": "reg:squarederror", "max_depth": 5, "nthread": 1,
                      "max_cat_to_onehot": 2}, x, label=y, num_boost_round=5, quiet=True,
                     feature_types=["q", "q", "c", "c"])
    assert model.predict(x[:10], pred_leaf=True).size  # sanity
    probe = x[:300].copy()
    probe[5, 2] = 77.0       # beyond the trained categories
    probe[6, 2] = -3.0       # negative code
    probe[7, 3] = 1e9        # absurd code
    probe[8, 2] = 2.9        # fractional code truncates
    for kind, flags in EXPLAIN_KINDS.items():
        assert_identical(model.predict(probe, **flags), model.predict(probe, use_gpu=True, **flags), kind)
    bst = booster_of(model, tmp_path)
    d = xgb.DMatrix(probe, feature_types=["q", "q", "c", "c"], enable_categorical=True)
    np.testing.assert_allclose(model.predict(probe, use_gpu=True, pred_contribs=True),
                               bst.predict(d, pred_contribs=True), rtol=0, atol=SHAP_ATOL)


@gpu
@pytest.mark.parametrize("name", ["binary", "multi", "lossguide"])
def test_gpu_on_xgboost_trained_models(name, tmp_path):
    if name == "multi":
        x, y = make(seed=4, k=4)
        params = {"objective": "multi:softprob", "num_class": 4, "max_depth": 5}
    else:
        x, y = make(seed=3)
        y = (y > 0).astype(np.float32)
        params = {"objective": "binary:logistic", "max_depth": 7}
        if name == "lossguide":
            params.update(grow_policy="lossguide", max_leaves=24, max_depth=0)
    bst = xgb.train({**params, "tree_method": "hist", "nthread": 1}, xgb.DMatrix(x, label=y), 6)
    path = str(tmp_path / "x.json")
    bst.save_model(path)
    model = fb.Model.load_model(path, quiet=True)
    d = xgb.DMatrix(x[:150])
    dev = model.predict(x[:150], use_gpu=True, pred_contribs=True)
    assert_identical(model.predict(x[:150], pred_contribs=True), dev)
    np.testing.assert_allclose(dev, bst.predict(d, pred_contribs=True), rtol=0, atol=SHAP_ATOL)
    if name != "lossguide":
        dev = model.predict(x[:60], use_gpu=True, pred_interactions=True)
        assert_identical(model.predict(x[:60], pred_interactions=True), dev)


@gpu
def test_gpu_batches_more_rows_than_one_device_batch_and_empty_input():
    x, y = make(n=500, seed=40)
    model = train_fb({"objective": "reg:squarederror", "max_depth": 4}, x, y)
    big = np.tile(x, (40, 1))
    assert_identical(model.predict(big, pred_contribs=True),
                     model.predict(big, use_gpu=True, pred_contribs=True))
    empty = np.zeros((0, x.shape[1]), np.float32)
    assert model.predict(empty, use_gpu=True, pred_contribs=True).shape == (0, x.shape[1] + 1)


@gpu
def test_gpu_explanations_from_a_file(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    x, y = make(n=700, seed=41)
    cols = {f"feat_{i}": x[:, i] for i in range(x.shape[1])}
    path = str(tmp_path / "t.parquet")
    pq.write_table(pa.table({**cols, "label": y}), path)
    model = fb.train({"objective": "reg:squarederror", "max_depth": 4, "nthread": 1}, path,
                     label="label", num_boost_round=4, quiet=True)
    for kind, flags in EXPLAIN_KINDS.items():
        assert_identical(model.predict(path, **flags), model.predict(path, use_gpu=True, **flags), kind)
