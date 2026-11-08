"""reg:quantileerror 和 quantile metric:和 XGBoost 3.4.1 逐位对齐。"""
import json

import numpy as np
import pytest

import ferrisboost as fb

xgb = pytest.importorskip("xgboost")

ALPHAS = [0.1, 0.5, 0.9, 0.0, 1.0]


def make(n=3000, m=6, seed=0, missing=0.05):
    # 低基数特征(取值数 < max_bin):分箱无损,和 XGBoost 可以逐位比。
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 9, size=(n, m)).astype(np.float32)
    x[rng.random(x.shape) < missing] = np.nan
    mean = 0.7 * np.nan_to_num(x[:, 0], nan=4.0) - 0.4 * np.nan_to_num(x[:, 1], nan=4.0) + 2.0
    y = (mean + rng.standard_t(3, n) * (1.0 + 0.2 * np.nan_to_num(x[:, 2], nan=4.0))).astype(
        np.float32
    )
    return x, y


def xgb_train(params, x, y, w, rounds, evals=()):
    p = {**params, "tree_method": "hist", "nthread": 1}
    d = xgb.DMatrix(x, label=y, weight=w)
    res = {}
    bst = xgb.train(p, d, num_boost_round=rounds,
                    evals=[(xgb.DMatrix(ex, label=ey, weight=ew), name) for ex, ey, ew, name in evals],
                    evals_result=res, verbose_eval=False)
    return bst, res


def saved_base_score(model, tmp_path):
    path = tmp_path / "m.json"
    model.save_model(str(path), quiet=True)
    raw = json.loads(path.read_text())["learner"]["learner_model_param"]["base_score"]
    return np.float32(float(raw.strip("[]")))


@pytest.mark.parametrize("alpha", ALPHAS)
@pytest.mark.parametrize("weighted", [False, True])
def test_bit_identical_to_xgboost(tmp_path, alpha, weighted):
    x, y = make(seed=3)
    w = np.random.default_rng(3).uniform(0.2, 3, len(y)).astype(np.float32) if weighted else None
    p = {"objective": "reg:quantileerror", "quantile_alpha": alpha, "max_depth": 5, "eta": 0.3}
    bst, _ = xgb_train(p, x, y, w, 10)
    ours = fb.train({**p, "nthread": 1}, x, label=y, weight=w, num_boost_round=10, quiet=True)
    ref_bs = json.loads(bst.save_config())["learner"]["learner_model_param"]["base_score"]
    assert saved_base_score(ours, tmp_path) == np.float32(float(ref_bs.strip("[]")))
    np.testing.assert_array_equal(ours.predict(x, output_margin=True),
                                  bst.predict(xgb.DMatrix(x), output_margin=True))
    np.testing.assert_array_equal(ours.predict(x), bst.predict(xgb.DMatrix(x)))


def test_alpha_as_length_one_list_and_explicit_base_score():
    x, y = make(n=500, seed=4)
    a = fb.train({"objective": "reg:quantileerror", "quantile_alpha": 0.3, "nthread": 1,
                  "base_score": 1.5}, x, label=y, num_boost_round=3, quiet=True)
    b = fb.train({"objective": "reg:quantileerror", "quantile_alpha": [0.3], "nthread": 1,
                  "base_score": 1.5}, x, label=y, num_boost_round=3, quiet=True)
    np.testing.assert_array_equal(a.predict(x), b.predict(x))
    bst, _ = xgb_train({"objective": "reg:quantileerror", "quantile_alpha": 0.3,
                        "base_score": 1.5}, x, y, None, 3)
    np.testing.assert_array_equal(a.predict(x), bst.predict(xgb.DMatrix(x)))


@pytest.mark.parametrize("weighted", [False, True])
def test_quantile_metric_matches_xgboost(weighted):
    x, y = make(seed=5)
    vx, vy = make(n=1500, seed=6)
    rng = np.random.default_rng(5)
    w = rng.uniform(0.2, 3, len(y)).astype(np.float32) if weighted else None
    vw = rng.uniform(0.2, 3, len(vy)).astype(np.float32) if weighted else None
    p = {"objective": "reg:quantileerror", "quantile_alpha": 0.8, "max_depth": 4, "eta": 0.3,
         "eval_metric": ["quantile", "mae", "rmse"]}
    _, ref = xgb_train(p, x, y, w, 6, evals=[(vx, vy, vw, "v")])
    ours = {}
    ev = (vx, vy, vw, "v") if weighted else (vx, vy, "v")
    fb.train({**p, "nthread": 1}, x, label=y, weight=w, evals=[ev], num_boost_round=6,
             evals_result=ours, quiet=True)
    assert list(ours["v"]) == list(ref["v"])
    for name in ("quantile", "mae", "rmse"):
        np.testing.assert_allclose(ours["v"][name], ref["v"][name], rtol=2e-6, atol=1e-7,
                                   err_msg=name)


def test_default_metric_is_quantile():
    x, y = make(n=400)
    res = {}
    fb.train({"objective": "reg:quantileerror", "quantile_alpha": 0.5}, x, label=y,
             evals=[(x, y, "t")], num_boost_round=1, evals_result=res, quiet=True)
    assert list(res["t"]) == ["quantile"]


@pytest.mark.parametrize("alpha", [0.2, 0.9])
def test_model_interchange_with_xgboost(tmp_path, alpha):
    x, y = make(seed=7)
    p = {"objective": "reg:quantileerror", "quantile_alpha": alpha, "max_depth": 4}
    ours = fb.train({**p, "nthread": 1}, x, label=y, num_boost_round=5, quiet=True)
    path = tmp_path / "fb.json"
    ours.save_model(str(path), quiet=True)
    bst = xgb.Booster(model_file=str(path))
    np.testing.assert_allclose(bst.predict(xgb.DMatrix(x)), ours.predict(x), rtol=1e-6)
    ref, _ = xgb_train(p, x, y, None, 5)
    ref_path = tmp_path / "xgb.json"
    ref.save_model(str(ref_path))
    loaded = fb.Model.load_model(str(ref_path), quiet=True)
    np.testing.assert_array_equal(loaded.predict(x), ref.predict(xgb.DMatrix(x)))
    assert loaded.objective == "reg:quantileerror"


def test_quantile_property_coverage_is_near_alpha():
    # 不依赖 XGBoost 的健全性检查:训练集上低于预测的比例应接近 alpha。
    x, y = make(n=6000, seed=8)
    for alpha in (0.1, 0.5, 0.9):
        m = fb.train({"objective": "reg:quantileerror", "quantile_alpha": alpha, "max_depth": 4,
                      "eta": 0.3}, x, label=y, num_boost_round=40, quiet=True)
        frac = float(np.mean(y <= m.predict(x)))
        assert abs(frac - alpha) < 0.08, (alpha, frac)


def test_missing_and_invalid_alpha_are_rejected():
    x, y = make(n=200)
    with pytest.raises(ValueError, match="quantile_alpha"):
        fb.train({"objective": "reg:quantileerror"}, x, label=y, num_boost_round=1, quiet=True)
    with pytest.raises(ValueError, match="quantile_alpha"):
        fb.train({"objective": "reg:quantileerror", "quantile_alpha": 1.5}, x, label=y,
                 num_boost_round=1, quiet=True)
    with pytest.raises(ValueError, match="quantile_alpha"):
        fb.train({"objective": "reg:quantileerror", "quantile_alpha": [0.1, -0.2]}, x, label=y,
                 num_boost_round=1, quiet=True)
    with pytest.raises(ValueError, match="quantile_alpha"):
        fb.train({"objective": "reg:quantileerror", "quantile_alpha": []}, x, label=y,
                 num_boost_round=1, quiet=True)


def test_quantile_metric_needs_the_quantile_objective():
    x, y = make(n=200)
    with pytest.raises(ValueError, match="quantile"):
        fb.train({"objective": "reg:squarederror", "eval_metric": "quantile"}, x, label=y,
                 num_boost_round=1, quiet=True)


# ---------------------------------------------------------------- GPU
def _has_cuda():
    try:
        x, y = make(64, 3)
        fb.train({"objective": "reg:quantileerror", "quantile_alpha": 0.5, "device": "cuda",
                  "max_depth": 2}, x, y, num_boost_round=1, quiet=True)
        return True
    except Exception as e:  # noqa: BLE001
        if "CUDA" in str(e) or "cuda" in str(e):
            return False
        raise


cuda = pytest.mark.skipif(not _has_cuda(), reason="没有可用的 CUDA 构建或设备")


def model_bytes(model, tmp_path, name):
    path = tmp_path / f"{name}.json"
    model.save_model(str(path), quiet=True)
    return path.read_bytes()


@cuda
@pytest.mark.parametrize("alpha", [0.1, 0.5, 0.9])
def test_gpu_exact_matches_cpu(tmp_path, alpha):
    x, y = make(n=4000, seed=11)
    w = np.random.default_rng(11).uniform(0.2, 3, len(y)).astype(np.float32)
    p = {"objective": "reg:quantileerror", "quantile_alpha": alpha, "max_depth": 6,
         "subsample": 0.8, "colsample_bytree": 0.8, "seed": 1}
    cpu = fb.train(p, x, label=y, weight=w, num_boost_round=6, quiet=True)
    gpu = fb.train({**p, "device": "cuda", "gpu_math": "exact"}, x, label=y, weight=w,
                   num_boost_round=6, quiet=True)
    assert model_bytes(cpu, tmp_path, "cpu") == model_bytes(gpu, tmp_path, "gpu")


@cuda
@pytest.mark.parametrize("alpha", [0.1, 0.5, 0.9])
@pytest.mark.parametrize("weighted", [False, True])
def test_gpu_fast_is_deterministic_and_calibrated(tmp_path, alpha, weighted):
    x, y = make(n=6000, seed=21)
    w = np.random.default_rng(21).uniform(0.2, 3, len(y)).astype(np.float32) if weighted else None
    p = {"objective": "reg:quantileerror", "quantile_alpha": alpha, "max_depth": 5,
         "device": "cuda", "gpu_math": "fast"}
    runs = [fb.train(p, x, label=y, weight=w, num_boost_round=20, quiet=True) for _ in range(2)]
    # fast: byte-identical from run to run (not necessarily equal to the CPU).
    assert model_bytes(runs[0], tmp_path, "a") == model_bytes(runs[1], tmp_path, "b")
    exact = fb.train({**p, "gpu_math": "exact"}, x, label=y, weight=w, num_boost_round=20,
                     quiet=True)
    cover = float(np.mean(y <= runs[0].predict(x)))
    cover_exact = float(np.mean(y <= exact.predict(x)))
    assert abs(cover - alpha) < 0.06
    assert abs(cover - cover_exact) < 0.03


# ---------------------------------------------------------------- 多个分位数(多输出)
MULTI = [0.1, 0.5, 0.9]


def xgb_json_param(bst):
    return json.loads(bst.save_config())["learner"]["learner_model_param"]


def fb_json(model, tmp_path, name="m"):
    path = tmp_path / f"{name}.json"
    model.save_model(str(path), quiet=True)
    return json.loads(path.read_text())


@pytest.mark.parametrize("alphas", [MULTI, [0.05, 0.25, 0.5, 0.75, 0.95], [0.0, 0.5, 1.0], [0.3, 0.3]])
@pytest.mark.parametrize("weighted", [False, True])
def test_multi_quantile_bit_identical_to_xgboost(tmp_path, alphas, weighted):
    x, y = make(seed=3)
    w = np.random.default_rng(3).uniform(0.2, 3, len(y)).astype(np.float32) if weighted else None
    p = {"objective": "reg:quantileerror", "quantile_alpha": alphas, "max_depth": 5, "eta": 0.3}
    bst, _ = xgb_train(p, x, y, w, 8)
    ours = fb.train({**p, "nthread": 1}, x, label=y, weight=w, num_boost_round=8, quiet=True)
    k = len(alphas)
    assert ours.num_trees == 8 * k
    # per-alpha intercept = the label quantile, like XGBoost
    ref_bs = np.array(
        [float(v) for v in xgb_json_param(bst)["base_score"].strip("[]").split(",")], np.float32
    )
    got = fb_json(ours, tmp_path)["learner"]["learner_model_param"]
    np.testing.assert_array_equal(
        np.array([float(v) for v in got["base_score"].strip("[]").split(",")], np.float32), ref_bs
    )
    assert (got["num_class"], got["num_target"]) == ("0", str(k))
    d = xgb.DMatrix(x)
    margin = ours.predict(x, output_margin=True)
    assert margin.shape == (len(y), k)
    np.testing.assert_array_equal(margin, bst.predict(d, output_margin=True))
    np.testing.assert_array_equal(ours.predict(x), bst.predict(d))
    assert ours.predict(x).shape == (len(y), k)


def test_multi_quantile_columns_equal_single_alpha_models():
    # One tree per alpha per round, each on its own gradient column: column j is exactly the
    # model trained with alpha_j alone (no sampling, so the trees are comparable).
    x, y = make(n=1500, seed=4)
    w = np.random.default_rng(4).uniform(0.2, 3, len(y)).astype(np.float32)
    base = {"objective": "reg:quantileerror", "max_depth": 4, "nthread": 1}
    multi = fb.train({**base, "quantile_alpha": MULTI}, x, label=y, weight=w,
                     num_boost_round=5, quiet=True)
    got = multi.predict(x, output_margin=True)
    for j, alpha in enumerate(MULTI):
        single = fb.train({**base, "quantile_alpha": alpha}, x, label=y, weight=w,
                          num_boost_round=5, quiet=True)
        np.testing.assert_array_equal(got[:, j], single.predict(x, output_margin=True))


def test_multi_quantile_predictions_are_sorted_but_margins_are_not(tmp_path):
    # XGBoost sorts every row of the predictions (PredTransform) so quantiles never cross;
    # output_margin=True is the raw per-quantile score. Force crossing margins by reversing
    # the per-alpha intercepts of a trained model, then compare with XGBoost loading the file.
    x, y = make(n=800, seed=9)
    p = {"objective": "reg:quantileerror", "quantile_alpha": MULTI, "max_depth": 3}
    bst, _ = xgb_train(p, x, y, None, 3)
    path = tmp_path / "cross.json"
    bst.save_model(str(path))
    doc = json.loads(path.read_text())
    doc["learner"]["learner_model_param"]["base_score"] = "[9E0,5E0,1E0]"
    path.write_text(json.dumps(doc))
    ref = xgb.Booster(model_file=str(path))
    ours = fb.Model.load_model(str(path), quiet=True)
    d = xgb.DMatrix(x)
    ref_margin = ref.predict(d, output_margin=True)
    assert (np.diff(ref_margin, axis=1) < 0).any(), "the fixture must produce crossing margins"
    np.testing.assert_array_equal(ours.predict(x, output_margin=True), ref_margin)
    np.testing.assert_array_equal(ours.predict(x), ref.predict(d))
    assert (np.diff(ours.predict(x), axis=1) >= 0).all()


@pytest.mark.parametrize("weighted", [False, True])
def test_multi_quantile_metric_and_early_stopping_match_xgboost(weighted):
    x, y = make(seed=5)
    vx, vy = make(n=1500, seed=6)
    rng = np.random.default_rng(5)
    w = rng.uniform(0.2, 3, len(y)).astype(np.float32) if weighted else None
    vw = rng.uniform(0.2, 3, len(vy)).astype(np.float32) if weighted else None
    p = {"objective": "reg:quantileerror", "quantile_alpha": MULTI, "max_depth": 4, "eta": 0.3}
    _, ref = xgb_train(p, x, y, w, 6, evals=[(vx, vy, vw, "v")])
    ours = {}
    ev = (vx, vy, vw, "v") if weighted else (vx, vy, "v")
    m = fb.train({**p, "nthread": 1}, x, label=y, weight=w, evals=[ev], num_boost_round=6,
                 evals_result=ours, quiet=True)
    assert list(ours["v"]) == list(ref["v"]) == ["quantile"]
    np.testing.assert_allclose(ours["v"]["quantile"], ref["v"]["quantile"], rtol=2e-6, atol=1e-7)
    assert m.num_trees == 18
    # early stopping works on the averaged metric and picks XGBoost's best iteration
    stopped = fb.train({**p, "nthread": 1}, x, label=y, weight=w, evals=[ev],
                       num_boost_round=300, early_stopping_rounds=3, quiet=True)
    ref_es = xgb.train({**p, "tree_method": "hist", "nthread": 1},
                       xgb.DMatrix(x, label=y, weight=w), 300,
                       evals=[(xgb.DMatrix(vx, label=vy, weight=vw), "v")],
                       early_stopping_rounds=3, verbose_eval=False)
    assert stopped.best_iteration == ref_es.best_iteration


def test_multi_quantile_rejects_other_metrics_and_unsorted_alpha():
    x, y = make(n=300)
    ok = {"objective": "reg:quantileerror", "quantile_alpha": MULTI}
    with pytest.raises(ValueError, match="quantile"):
        fb.train({**ok, "eval_metric": "mae"}, x, label=y, evals=[(x, y, "t")],
                 num_boost_round=1, quiet=True)
    with pytest.raises(ValueError, match="升序|ascending"):
        fb.train({**ok, "quantile_alpha": [0.9, 0.1]}, x, label=y, num_boost_round=1, quiet=True)
    with pytest.raises(ValueError, match="quantile_alpha"):
        fb.train({**ok, "quantile_alpha": [0.5] * 65}, x, label=y, num_boost_round=1, quiet=True)
    # an explicit base_score is replicated for every quantile (XGBoost: scalar -> vector)
    m = fb.train({**ok, "base_score": 2.0, "nthread": 1}, x, label=y, num_boost_round=1,
                 quiet=True)
    ref, _ = xgb_train({**ok, "base_score": 2.0}, x, y, None, 1)
    np.testing.assert_array_equal(m.predict(x), ref.predict(xgb.DMatrix(x)))


@pytest.mark.parametrize("weighted", [False, True])
def test_multi_quantile_model_interchange_with_xgboost(tmp_path, weighted):
    x, y = make(seed=7)
    w = np.random.default_rng(7).uniform(0.2, 3, len(y)).astype(np.float32) if weighted else None
    p = {"objective": "reg:quantileerror", "quantile_alpha": MULTI, "max_depth": 4}
    ours = fb.train({**p, "nthread": 1}, x, label=y, weight=w, num_boost_round=5, quiet=True)
    # FerrisBoost -> XGBoost
    path = tmp_path / "fb.json"
    ours.save_model(str(path), quiet=True)
    bst = xgb.Booster(model_file=str(path))
    d = xgb.DMatrix(x)
    np.testing.assert_allclose(bst.predict(d), ours.predict(x), rtol=1e-6)
    np.testing.assert_allclose(bst.predict(d, output_margin=True),
                               ours.predict(x, output_margin=True), rtol=1e-6)
    cfg = json.loads(bst.save_config())["learner"]
    assert json.loads(cfg["objective"]["quantile_loss_param"]["quantile_alpha"]) == MULTI
    # XGBoost -> FerrisBoost, exact; then FerrisBoost -> FerrisBoost is lossless
    ref, _ = xgb_train(p, x, y, w, 5)
    ref_path = tmp_path / "xgb.json"
    ref.save_model(str(ref_path))
    loaded = fb.Model.load_model(str(ref_path), quiet=True)
    assert loaded.objective == "reg:quantileerror" and loaded.num_class == 3
    np.testing.assert_array_equal(loaded.predict(x), ref.predict(d))
    np.testing.assert_array_equal(loaded.predict(x, output_margin=True),
                                  ref.predict(d, output_margin=True))
    again = tmp_path / "again.json"
    loaded.save_model(str(again), quiet=True)
    back = fb.Model.load_model(str(again), quiet=True)
    np.testing.assert_array_equal(back.predict(x), ref.predict(d))
    # a multi-target model that is not a multi-alpha quantile model is still refused
    doc = json.loads(ref_path.read_text())
    doc["learner"]["objective"] = {"name": "reg:squarederror",
                                   "reg_loss_param": {"scale_pos_weight": "1"}}
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="num_target|多目标|objective"):
        fb.Model.load_model(str(bad), quiet=True)


def test_multi_quantile_iteration_range_shap_leaf_and_score_match_xgboost():
    x, y = make(n=1200, seed=12)
    p = {"objective": "reg:quantileerror", "quantile_alpha": MULTI, "max_depth": 4}
    ref, _ = xgb_train(p, x, y, None, 6)
    ours = fb.train({**p, "nthread": 1}, x, label=y, num_boost_round=6, quiet=True)
    d = xgb.DMatrix(x)
    for rng in [(0, 3), (2, 5), (4, 6), (3, 3)]:
        np.testing.assert_array_equal(ours.predict(x, iteration_range=rng),
                                      ref.predict(d, iteration_range=rng))
        np.testing.assert_array_equal(ours.predict(x, iteration_range=rng, output_margin=True),
                                      ref.predict(d, iteration_range=rng, output_margin=True))
    contribs = ours.predict(x, pred_contribs=True)
    ref_contribs = ref.predict(d, pred_contribs=True)
    assert contribs.shape == ref_contribs.shape == (len(y), 3, x.shape[1] + 1)
    np.testing.assert_array_equal(contribs, ref_contribs)
    np.testing.assert_array_equal(ours.predict(x, pred_contribs=True, approx_contribs=True),
                                  ref.predict(d, pred_contribs=True, approx_contribs=True))
    # the contributions add up to the margin of each quantile
    np.testing.assert_allclose(contribs.sum(axis=2), ours.predict(x, output_margin=True),
                               rtol=1e-4, atol=1e-4)
    leaf = ours.predict(x, pred_leaf=True)
    assert leaf.shape == ref.predict(d, pred_leaf=True).shape
    np.testing.assert_array_equal(leaf, ref.predict(d, pred_leaf=True))
    inter = ours.predict(x[:50], pred_interactions=True)
    ref_inter = ref.predict(xgb.DMatrix(x[:50]), pred_interactions=True)
    assert inter.shape == ref_inter.shape == (50, 3, x.shape[1] + 1, x.shape[1] + 1)
    np.testing.assert_array_equal(inter, ref_inter)
    for kind in ("weight", "gain", "cover", "total_gain", "total_cover"):
        a, b = ours.get_score(importance_type=kind), ref.get_score(importance_type=kind)
        assert list(a) == list(b)
        np.testing.assert_allclose(list(a.values()), list(b.values()), rtol=1e-5)


def test_multi_quantile_sklearn_regressor_predicts_one_column_per_quantile():
    x, y = make(n=1000, seed=13)
    est = fb.FerrisBoostRegressor(
        objective="reg:quantileerror", quantile_alpha=MULTI, n_estimators=4, max_depth=3,
        n_jobs=1,
    ).fit(x, y)
    pred = est.predict(x)
    assert pred.shape == (len(y), 3)
    assert (np.diff(pred, axis=1) >= 0).all()
    ref = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=np.array(MULTI),
                           n_estimators=4, max_depth=3, tree_method="hist", n_jobs=1).fit(x, y)
    np.testing.assert_array_equal(pred, ref.predict(x))


def test_multi_quantile_coverage_is_near_each_alpha():
    x, y = make(n=6000, seed=8)
    m = fb.train({"objective": "reg:quantileerror", "quantile_alpha": MULTI, "max_depth": 4,
                  "eta": 0.3}, x, label=y, num_boost_round=40, quiet=True)
    pred = m.predict(x)
    for j, alpha in enumerate(MULTI):
        assert abs(float(np.mean(y <= pred[:, j])) - alpha) < 0.08, (alpha, j)


@cuda
@pytest.mark.parametrize("alphas", [MULTI, [0.05, 0.5, 0.95]])
def test_gpu_exact_multi_quantile_matches_cpu(tmp_path, alphas):
    x, y = make(n=4000, seed=11)
    w = np.random.default_rng(11).uniform(0.2, 3, len(y)).astype(np.float32)
    p = {"objective": "reg:quantileerror", "quantile_alpha": alphas, "max_depth": 6,
         "subsample": 0.8, "colsample_bytree": 0.8, "seed": 1}
    cpu = fb.train(p, x, label=y, weight=w, num_boost_round=6, quiet=True)
    gpu = fb.train({**p, "device": "cuda", "gpu_math": "exact"}, x, label=y, weight=w,
                   num_boost_round=6, quiet=True)
    assert model_bytes(cpu, tmp_path, "cpu") == model_bytes(gpu, tmp_path, "gpu")


@cuda
@pytest.mark.parametrize("weighted", [False, True])
def test_gpu_fast_multi_quantile_is_deterministic_and_close_to_exact(tmp_path, weighted):
    x, y = make(n=6000, seed=21)
    w = np.random.default_rng(21).uniform(0.2, 3, len(y)).astype(np.float32) if weighted else None
    p = {"objective": "reg:quantileerror", "quantile_alpha": MULTI, "max_depth": 5,
         "device": "cuda", "gpu_math": "fast", "subsample": 0.9, "seed": 2}
    runs = [fb.train(p, x, label=y, weight=w, num_boost_round=20, quiet=True) for _ in range(2)]
    assert model_bytes(runs[0], tmp_path, "a") == model_bytes(runs[1], tmp_path, "b")
    exact = fb.train({**p, "gpu_math": "exact"}, x, label=y, weight=w, num_boost_round=20,
                     quiet=True)
    fast_pred, exact_pred = runs[0].predict(x), exact.predict(x)
    assert fast_pred.shape == (len(y), 3)
    for j, alpha in enumerate(MULTI):
        assert abs(float(np.mean(y <= fast_pred[:, j])) - alpha) < 0.06
        assert abs(float(np.mean(y <= fast_pred[:, j])) -
                   float(np.mean(y <= exact_pred[:, j]))) < 0.03
    # the fast metric is close to the exact one
    ev = [(x, y, "t")] if w is None else [(x, y, w, "t")]
    res_f, res_e = {}, {}
    fb.train(p, x, label=y, weight=w, evals=ev, num_boost_round=10, evals_result=res_f, quiet=True)
    fb.train({**p, "gpu_math": "exact"}, x, label=y, weight=w, evals=ev, num_boost_round=10,
             evals_result=res_e, quiet=True)
    np.testing.assert_allclose(res_f["t"]["quantile"], res_e["t"]["quantile"], rtol=0.05)


@cuda
def test_gpu_fast_multi_quantile_planner_counts_the_prediction_columns(capfd):
    x, y = make(n=3000, seed=22)
    fb.train({"objective": "reg:quantileerror", "quantile_alpha": MULTI, "max_depth": 3,
              "device": "cuda", "gpu_math": "fast"}, x, label=y, num_boost_round=1)
    err = capfd.readouterr().err
    # (K - 1) * n * 4 extra prediction bytes, and no softmax statistics
    assert f"multiclass_outputs=3 multiclass_bytes={2 * 3000 * 4}" in err
