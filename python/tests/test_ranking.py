"""Learning to rank: rank:ndcg / rank:pairwise / rank:map vs XGBoost 3.4.1.

The data is integer-valued (lossless binning), so with ``nthread=1`` the models are
bit-identical to ``xgb.train(tree_method="hist")``: predictions are compared bit for bit,
ranking metrics to float precision, and models cross-load in both directions.
"""
import itertools
import json
import os

import numpy as np
import pytest

import ferrisboost as fb

xgb = pytest.importorskip("xgboost")

SIZES = [1, 2, 3, 5, 8, 13, 21, 34, 40, 50, 7, 9]


def make(seed=0, sizes=SIZES, n_features=6, weights=False, binary=False):
    rng = np.random.default_rng(seed)
    n = int(sum(sizes))
    x = rng.integers(0, 12, size=(n, n_features)).astype(np.float32)
    s = x[:, 0] * 0.5 + x[:, 1] * 0.3 - x[:, 2] * 0.2 + rng.normal(0, 1.5, n)
    y = np.clip(np.round((s - s.min()) / (s.max() - s.min()) * 4), 0, 4).astype(np.float32)
    if binary:
        y = (y > 1).astype(np.float32)
    qid = np.repeat(np.arange(len(sizes)), sizes)
    w = rng.uniform(0.5, 2.0, len(sizes)).astype(np.float32) if weights else None
    return x, y, qid, np.asarray(sizes), w


PARAMS = {"max_depth": 3, "max_bin": 16, "nthread": 1}


def xgb_train(obj, x, y, qid, w, rounds, extra=None, evals=True):
    d = xgb.DMatrix(x, y, qid=qid, weight=w)
    p = dict(objective=obj, tree_method="hist", nthread=1, max_depth=3, max_bin=16, **(extra or {}))
    res = {}
    b = xgb.train(p, d, rounds, evals=[(d, "t")] if evals else [], evals_result=res, verbose_eval=False)
    return b, d, res


def fb_train(obj, x, y, qid, w, rounds, extra=None, evals=True, **kw):
    p = {**PARAMS, "objective": obj, **(extra or {})}
    res = {}
    m = fb.train(
        p, x, label=y, qid=qid, weight=w, num_boost_round=rounds, quiet=True,
        evals=[(x, y, w, "t")] if evals and w is not None else ([(x, y, "t")] if evals else []),
        eval_qid=[qid] if evals else None, evals_result=res, **kw,
    )
    return m, res


def f32_bits(a):
    return np.asarray(a, dtype=np.float32).view(np.uint32)


CASES = list(itertools.product(
    ["rank:ndcg", "rank:pairwise", "rank:map"], ["topk", "mean"], [False, True]))


@pytest.mark.parametrize("obj,pair_method,weighted", CASES)
def test_bit_exact_vs_xgboost(obj, pair_method, weighted):
    x, y, qid, _, w = make(weights=weighted, binary=obj == "rank:map")
    extra = {"lambdarank_pair_method": pair_method}
    b, d, xres = xgb_train(obj, x, y, qid, w, 6, extra)
    m, fres = fb_train(obj, x, y, qid, w, 6, extra)
    np.testing.assert_array_equal(f32_bits(m.predict(x)), f32_bits(b.predict(d)))
    # the same metric, evaluated every round on the training data
    (name, xvals), = xres["t"].items()
    (fname, fvals), = fres["t"].items()
    assert fname == name
    np.testing.assert_allclose(fvals, xvals, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("obj", ["rank:ndcg", "rank:pairwise", "rank:map"])
@pytest.mark.parametrize("extra", [
    {"lambdarank_num_pair_per_sample": 3},
    {"lambdarank_num_pair_per_sample": 4, "lambdarank_pair_method": "mean"},
    {"lambdarank_normalization": False},
    {"lambdarank_score_normalization": False},
    {"ndcg_exp_gain": False},
])
def test_parameters_vs_xgboost(obj, extra):
    x, y, qid, _, w = make(seed=3, binary=obj == "rank:map")
    b, d, _ = xgb_train(obj, x, y, qid, None, 4, extra, evals=False)
    m, _ = fb_train(obj, x, y, qid, None, 4, extra, evals=False)
    np.testing.assert_array_equal(f32_bits(m.predict(x)), f32_bits(b.predict(d)))


def test_group_sizes_equal_qid():
    x, y, qid, sizes, _ = make()
    a, _ = fb_train("rank:ndcg", x, y, qid, None, 3, evals=False)
    p = dict(objective="rank:ndcg", **PARAMS)
    b = fb.train(p, x, label=y, group=sizes, num_boost_round=3, quiet=True)
    np.testing.assert_array_equal(f32_bits(a.predict(x)), f32_bits(b.predict(x)))


def test_validation_set_and_early_stopping():
    x, y, qid, _, _ = make(seed=1)
    vx, vy, vqid, _, _ = make(seed=2, sizes=[6, 12, 30, 4, 18])
    d = xgb.DMatrix(x, y, qid=qid)
    dv = xgb.DMatrix(vx, vy, qid=vqid)
    p = dict(objective="rank:ndcg", tree_method="hist", nthread=1, max_depth=3, max_bin=16,
             eval_metric="ndcg@5")
    xres = {}
    xb = xgb.train(p, d, 40, evals=[(dv, "v")], evals_result=xres, verbose_eval=False,
                   early_stopping_rounds=5)
    res = {}
    m = fb.train({"objective": "rank:ndcg", "eval_metric": "ndcg@5", **PARAMS}, x, label=y,
                 qid=qid, num_boost_round=40, evals=[(vx, vy, "v")], eval_qid=[vqid],
                 early_stopping_rounds=5, evals_result=res, quiet=True)
    assert m.best_iteration == xb.best_iteration
    np.testing.assert_allclose(res["v"]["ndcg@5"], xres["v"]["ndcg@5"], rtol=1e-6)
    np.testing.assert_array_equal(f32_bits(m.predict(vx)), f32_bits(xb.predict(dv)))


METRICS = ["ndcg", "ndcg@3", "ndcg@5-", "ndcg-", "ndcg@1000", "map", "map@4", "map@5-", "pre@3",
           "pre@10", "pre"]


@pytest.mark.parametrize("weighted", [False, True])
def test_metrics_vs_xgboost(weighted):
    x, y, qid, _, w = make(seed=5, weights=weighted)
    yb = (y > 1).astype(np.float32)
    # a model with some structure, evaluated on both label kinds
    m, _ = fb_train("rank:pairwise", x, yb, qid, w, 5, evals=False)
    scores = m.predict(x)
    for metric in METRICS:
        labels = yb if metric.startswith(("map", "pre")) else y
        d = xgb.DMatrix(x, labels, qid=qid, weight=w)
        # XGBoost's value for the same raw scores: base_score = 0 model with the scores as margin
        d.set_base_margin(scores)
        booster = xgb.train(dict(objective="rank:ndcg", tree_method="hist", nthread=1,
                                 base_score=0.0, eval_metric=metric), d, 0)
        text = booster.eval(d, "t")
        want = float(text.split(":")[-1])
        got = fb._core.rank_score(metric, scores.astype(np.float32), labels,
                                  qid.astype(np.uint64), w, True)
        assert got == pytest.approx(want, rel=1e-12, abs=1e-12), (metric, got, want)


def test_default_metric_names_follow_xgboost():
    x, y, qid, _, _ = make()
    for obj, extra, name in [
        ("rank:ndcg", {}, "ndcg@32"),
        ("rank:pairwise", {}, "ndcg@32"),
        ("rank:map", {}, "map@32"),
        ("rank:ndcg", {"lambdarank_num_pair_per_sample": 7}, "ndcg@7"),
        ("rank:ndcg", {"lambdarank_pair_method": "mean"}, "ndcg"),
        ("rank:map", {"lambdarank_pair_method": "mean"}, "map"),
    ]:
        yy = (y > 1).astype(np.float32) if obj == "rank:map" else y
        _, res = fb_train(obj, x, yy, qid, None, 1, extra)
        assert list(res["t"]) == [name], (obj, extra, res)


@pytest.mark.parametrize("obj", ["rank:ndcg", "rank:pairwise", "rank:map"])
def test_json_round_trip_with_xgboost(obj, tmp_path):
    x, y, qid, _, w = make(weights=True, binary=obj == "rank:map")
    extra = {"lambdarank_pair_method": "mean", "lambdarank_num_pair_per_sample": 2,
             "lambdarank_score_normalization": False}
    b, d, _ = xgb_train(obj, x, y, qid, w, 4, extra, evals=False)
    m, _ = fb_train(obj, x, y, qid, w, 4, extra, evals=False)
    # FerrisBoost model in XGBoost
    path = os.fspath(tmp_path / "fb.json")
    m.save_model(path, quiet=True)
    loaded = xgb.Booster()
    loaded.load_model(path)
    np.testing.assert_array_equal(f32_bits(loaded.predict(d)), f32_bits(m.predict(x)))
    cfg = loaded.save_config()
    assert obj in cfg and '"lambdarank_pair_method":"mean"' in cfg.replace(" ", "")
    # XGBoost model in FerrisBoost
    xpath = os.fspath(tmp_path / "xgb.json")
    b.save_model(xpath)
    back = fb.Model.load_model(xpath, quiet=True)
    assert back.objective == obj
    np.testing.assert_array_equal(f32_bits(back.predict(x)), f32_bits(b.predict(d)))
    # and FerrisBoost re-saves the parameters it read
    again = os.fspath(tmp_path / "again.json")
    back.save_model(again, quiet=True)
    again_booster = xgb.Booster()
    again_booster.load_model(again)
    np.testing.assert_array_equal(f32_bits(again_booster.predict(d)), f32_bits(b.predict(d)))


def test_user_base_score_skips_estimation():
    x, y, qid, _, _ = make()
    b, d, _ = xgb_train("rank:ndcg", x, y, qid, None, 3, {"base_score": 0.25}, evals=False)
    m, _ = fb_train("rank:ndcg", x, y, qid, None, 3, {"base_score": 0.25}, evals=False)
    np.testing.assert_array_equal(f32_bits(m.predict(x)), f32_bits(b.predict(d)))


def test_mean_pairs_with_seed():
    x, y, qid, _, _ = make()
    extra = {"lambdarank_pair_method": "mean", "seed": 7}
    b, d, _ = xgb_train("rank:ndcg", x, y, qid, None, 4, extra, evals=False)
    m, _ = fb_train("rank:ndcg", x, y, qid, None, 4, extra, evals=False)
    np.testing.assert_array_equal(f32_bits(m.predict(x)), f32_bits(b.predict(d)))


def test_models_do_not_depend_on_threads():
    x, y, qid, _, w = make(weights=True)
    p1, _ = fb_train("rank:ndcg", x, y, qid, w, 4, {"nthread": 1}, evals=False)
    p4, _ = fb_train("rank:ndcg", x, y, qid, w, 4, {"nthread": 4}, evals=False)
    np.testing.assert_array_equal(f32_bits(p1.predict(x)), f32_bits(p4.predict(x)))


def test_subsample_and_colsample_run():
    x, y, qid, _, _ = make()
    m, _ = fb_train("rank:ndcg", x, y, qid, None, 4,
                    {"subsample": 0.7, "colsample_bytree": 0.8, "seed": 3}, evals=False)
    assert np.isfinite(m.predict(x)).all()


# ---------------------------------------------------------------- input validation
def test_qid_must_be_sorted_and_groups_must_match():
    x, y, qid, sizes, _ = make()
    p = dict(objective="rank:ndcg", **PARAMS)
    bad = qid.copy()
    bad[[0, -1]] = bad[[-1, 0]]
    with pytest.raises(ValueError, match="non-decreasing"):
        fb.train(p, x, label=y, qid=bad, quiet=True)
    with pytest.raises(ValueError, match="group sizes sum"):
        fb.train(p, x, label=y, group=sizes[:-1], quiet=True)
    with pytest.raises(ValueError, match="Either one of"):
        fb.train(p, x, label=y, qid=qid, group=sizes, quiet=True)
    with pytest.raises(ValueError, match="needs query groups"):
        fb.train(p, x, label=y, quiet=True)
    # other objectives accept query groups (as XGBoost does) and ignore them in training
    a = fb.train(dict(objective="reg:squarederror", **PARAMS), x, label=y, qid=qid,
                 num_boost_round=3, quiet=True)
    b = fb.train(dict(objective="reg:squarederror", **PARAMS), x, label=y,
                 num_boost_round=3, quiet=True)
    np.testing.assert_array_equal(f32_bits(a.predict(x)), f32_bits(b.predict(x)))
    with pytest.raises(ValueError, match="per query group"):
        fb.train(p, x, label=y, qid=qid, weight=np.ones(len(y), np.float32), quiet=True)
    with pytest.raises(ValueError, match="at least 1|>= 1"):
        fb.train({**p, "lambdarank_num_pair_per_sample": 0}, x, label=y, qid=qid, quiet=True)


def test_label_rules_and_unsupported_options():
    x, y, qid, _, _ = make()
    p = dict(objective="rank:ndcg", **PARAMS)
    with pytest.raises(ValueError, match="integers"):
        fb.train(p, x, label=y + 0.5, qid=qid, quiet=True)
    # fractional relevance is fine with a linear gain
    fb.train({**p, "ndcg_exp_gain": False}, x, label=y + 0.5, qid=qid, num_boost_round=2, quiet=True)
    with pytest.raises(ValueError, match="binary"):
        fb.train(dict(p, objective="rank:map"), x, label=y, qid=qid, quiet=True)
    with pytest.raises(ValueError, match="ranking metric|ranking objective"):
        fb.train({**p, "eval_metric": "rmse"}, x, label=y, qid=qid, quiet=True)
    with pytest.raises(ValueError, match="rank:ndcg"):
        fb.train(dict(objective="reg:squarederror", eval_metric="ndcg@3"), x, label=y, quiet=True)


def test_eval_set_needs_groups():
    x, y, qid, _, _ = make()
    p = dict(objective="rank:ndcg", **PARAMS)
    with pytest.raises(ValueError, match="query groups"):
        fb.train(p, x, label=y, qid=qid, evals=[(x, y, "v")], quiet=True)


# ---------------------------------------------------------------- file inputs
def _write(tmp_path, x, y, qid, w=None, fmt="parquet"):
    import pyarrow as pa
    import pyarrow.csv as pacsv
    import pyarrow.parquet as pq

    cols = {"query_id": qid.astype(np.int64), "rel": y}
    if w is not None:
        cols["w"] = np.repeat(w, np.bincount(qid))
    for j in range(x.shape[1]):
        cols[f"f{j}"] = x[:, j]
    table = pa.table(cols)
    path = os.fspath(tmp_path / f"d.{fmt}")
    if fmt == "parquet":
        pq.write_table(table, path, row_group_size=37)
    else:
        pacsv.write_csv(table, path)
    return path


@pytest.mark.parametrize("fmt", ["parquet", "csv"])
@pytest.mark.parametrize("weighted", [False, True])
def test_file_inputs_with_qid_column(tmp_path, fmt, weighted):
    x, y, qid, _, w = make(weights=weighted)
    path = _write(tmp_path, x, y, qid, w, fmt)
    p = dict(objective="rank:ndcg", **PARAMS)
    res = {}
    m = fb.train(p, path, label="rel", qid="query_id", weight="w" if weighted else None,
                 num_boost_round=5, evals=[(path, "t")], evals_result=res, quiet=True)
    ref, rres = fb_train("rank:ndcg", x, y, qid, w, 5)
    np.testing.assert_array_equal(f32_bits(m.predict(path)), f32_bits(ref.predict(x)))
    np.testing.assert_allclose(res["t"]["ndcg@32"], rres["t"]["ndcg@32"], rtol=1e-6)
    assert m.n_features == x.shape[1]
    with pytest.raises(ValueError, match="needs query groups"):
        fb.train(p, path, label="rel", quiet=True)


def test_file_unsorted_qid_and_varying_weights_rejected(tmp_path):
    x, y, qid, _, w = make(weights=True)
    p = dict(objective="rank:ndcg", **PARAMS)
    bad = qid.copy()
    bad[[0, -1]] = bad[[-1, 0]]
    path = _write(tmp_path, x, y, bad, w)
    with pytest.raises(Exception, match="non-decreasing"):
        fb.train(p, path, label="rel", qid="query_id", quiet=True)
    path = _write(tmp_path, x, y, qid, w)
    import pyarrow.parquet as pq

    t = pq.read_table(path)
    wcol = t["w"].to_numpy().copy()
    wcol[1] += 1.0
    t = t.set_column(t.schema.get_field_index("w"), "w", __import__("pyarrow").array(wcol))
    pq.write_table(t, path)
    with pytest.raises(Exception, match="per query group"):
        fb.train(p, path, label="rel", qid="query_id", weight="w", quiet=True)


# ---------------------------------------------------------------- sklearn
def test_ranker_fit_predict_score():
    from ferrisboost.sklearn import FerrisBoostRanker

    x, y, qid, sizes, w = make(weights=True)
    vx, vy, vqid, _, _ = make(seed=9, sizes=[10, 20, 15])
    r = FerrisBoostRanker(n_estimators=8, max_depth=3, max_bin=16, n_jobs=1)
    r.fit(x, y, qid=qid, sample_weight=w, eval_set=[(vx, vy)], eval_qid=[vqid])
    ref, _ = fb_train("rank:ndcg", x, y, qid, w, 8, evals=False)
    np.testing.assert_array_equal(f32_bits(r.predict(x)), f32_bits(ref.predict(x)))
    assert "validation_0" in r.evals_result()
    import pandas as pd

    df = pd.DataFrame(vx, columns=[f"c{i}" for i in range(vx.shape[1])])
    df["qid"] = vqid
    score = r.score(df, vy)
    assert score == pytest.approx(r.evals_result()["validation_0"]["ndcg@32"][-1], rel=1e-5)
    # qid as a DataFrame column, group sizes: the same model
    tr = pd.DataFrame(x, columns=df.columns[:-1])
    tr["qid"] = qid
    r2 = FerrisBoostRanker(n_estimators=8, max_depth=3, max_bin=16, n_jobs=1)
    r2.fit(tr, y, sample_weight=w)
    np.testing.assert_array_equal(f32_bits(r2.predict(tr)), f32_bits(r.predict(x)))
    r3 = FerrisBoostRanker(n_estimators=8, max_depth=3, max_bin=16, n_jobs=1)
    r3.fit(x, y, group=sizes, sample_weight=w)
    np.testing.assert_array_equal(f32_bits(r3.predict(x)), f32_bits(r.predict(x)))
    with pytest.raises(ValueError, match="qid"):
        FerrisBoostRanker(n_estimators=2).fit(x, y)


def test_ranker_early_stopping_and_params(tmp_path):
    from ferrisboost.sklearn import FerrisBoostRanker

    x, y, qid, _, _ = make(seed=1)
    vx, vy, vqid, _, _ = make(seed=2, sizes=[6, 12, 30, 4, 18])
    r = FerrisBoostRanker(n_estimators=60, max_depth=3, max_bin=16, n_jobs=1,
                          objective="rank:pairwise", lambdarank_pair_method="mean",
                          eval_metric="ndcg@5", early_stopping_rounds=4)
    r.fit(x, y, qid=qid, eval_set=[(vx, vy)], eval_qid=[vqid])
    assert r.best_iteration < 59
    assert r.get_booster().objective == "rank:pairwise"
    path = os.fspath(tmp_path / "r.json")
    r.save_model(path)
    again = FerrisBoostRanker.from_file(path)
    np.testing.assert_array_equal(f32_bits(again.predict(vx)), f32_bits(r.predict(vx)))


# ---------------------------------------------------------------- GPU
def _has_cuda():
    x, y, qid, _, _ = make(sizes=[4, 4])
    try:
        fb.train({"objective": "rank:ndcg", "device": "cuda", "max_depth": 2}, x, label=y,
                 qid=qid, num_boost_round=1, quiet=True)
        return True
    except Exception as e:  # noqa: BLE001
        if "CUDA" in str(e) or "cuda" in str(e):
            return False
        raise


cuda = pytest.mark.skipif(not _has_cuda(), reason="no CUDA build or device")


@cuda
@pytest.mark.parametrize("obj,pair_method,weighted", CASES)
def test_gpu_exact_is_byte_identical_to_cpu(obj, pair_method, weighted):
    x, y, qid, _, w = make(weights=weighted, binary=obj == "rank:map")
    extra = {"lambdarank_pair_method": pair_method}
    cpu, _ = fb_train(obj, x, y, qid, w, 5, extra, evals=False)
    gpu, _ = fb_train(obj, x, y, qid, w, 5, {**extra, "device": "cuda", "gpu_math": "exact"},
                      evals=False)
    np.testing.assert_array_equal(f32_bits(cpu.predict(x)), f32_bits(gpu.predict(x)))


def _big(n_groups=300, seed=11):
    rng = np.random.default_rng(seed)
    sizes = rng.integers(2, 60, size=n_groups)
    return make(seed=seed, sizes=list(sizes))


@cuda
@pytest.mark.parametrize("obj,pair_method", list(itertools.product(
    ["rank:ndcg", "rank:pairwise", "rank:map"], ["topk", "mean"])))
def test_gpu_fast_is_deterministic_and_close_to_exact(obj, pair_method):
    x, y, qid, _, w = _big()
    if obj == "rank:map":
        y = (y > 1).astype(np.float32)
    extra = {"lambdarank_pair_method": pair_method, "max_depth": 4}
    fast = [fb_train(obj, x, y, qid, w, 12, {**extra, "device": "cuda", "gpu_math": "fast"},
                     evals=False)[0] for _ in range(2)]
    np.testing.assert_array_equal(f32_bits(fast[0].predict(x)), f32_bits(fast[1].predict(x)))
    exact, _ = fb_train(obj, x, y, qid, w, 12, extra, evals=False)
    name = "map@32" if obj == "rank:map" else "ndcg@32"

    def score(m):
        return fb._core.rank_score(name, m.predict(x), y, qid.astype(np.uint64), None, True)

    assert score(fast[0]) == pytest.approx(score(exact), abs=0.03)
    base, _ = fb_train(obj, x, y, qid, w, 1, {**extra, "eta": 0.0001}, evals=False)
    assert score(fast[0]) > score(base)


@cuda
def test_gpu_fast_with_validation_and_subsample():
    x, y, qid, _, _ = _big()
    vx, vy, vqid, _, _ = _big(40, seed=5)
    res = {}
    fb.train({"objective": "rank:ndcg", "device": "cuda", "gpu_math": "fast", "max_depth": 3,
              "subsample": 0.8, "seed": 2}, x, label=y, qid=qid, num_boost_round=6, quiet=True,
             evals=[(vx, vy, "v")], eval_qid=[vqid], evals_result=res)
    assert len(res["v"]["ndcg@32"]) == 6


# ---------------------------------------------------------------- unbiased LTR
def _xgb_bias(booster):
    obj = json.loads(booster.save_raw("json"))["learner"]["objective"]
    return obj["ti+"], obj["tj-"]


def _fb_bias(model, tmp_path, name="m.json"):
    path = os.fspath(tmp_path / name)
    model.save_model(path, quiet=True)
    with open(path) as f:
        obj = json.load(f)["learner"]["objective"]
    return obj["ti+"], obj["tj-"]


UNBIASED = list(itertools.product(
    ["rank:ndcg", "rank:pairwise", "rank:map"], ["topk", "mean"], [False, True]))


@pytest.mark.parametrize("obj,pair_method,weighted", UNBIASED)
def test_unbiased_bit_exact_vs_xgboost(obj, pair_method, weighted, tmp_path):
    x, y, qid, _, w = make(weights=weighted, binary=True)
    extra = {"lambdarank_unbiased": True, "lambdarank_pair_method": pair_method}
    b, d, xres = xgb_train(obj, x, y, qid, w, 8, extra)
    m, fres = fb_train(obj, x, y, qid, w, 8, extra)
    np.testing.assert_array_equal(f32_bits(m.predict(x)), f32_bits(b.predict(d)))
    (name, xvals), = xres["t"].items()
    np.testing.assert_allclose(fres["t"][name], xvals, rtol=1e-6, atol=1e-7)
    # the estimated position biases are part of the model, like XGBoost's `ti+` / `tj-`
    xti, xtj = _xgb_bias(b)
    fti, ftj = _fb_bias(m, tmp_path)
    assert fti == xti and ftj == xtj
    assert any(v != 1.0 for v in fti[1:]) and fti[0] == 1.0
    # and the unbiased model differs from the plain one
    plain, _ = fb_train(obj, x, y, qid, w, 8, {"lambdarank_pair_method": pair_method},
                        evals=False)
    assert not np.array_equal(f32_bits(plain.predict(x)), f32_bits(m.predict(x)))


@pytest.mark.parametrize("extra", [
    {"lambdarank_bias_norm": 0.0},
    {"lambdarank_bias_norm": 0.5},
    {"lambdarank_bias_norm": 3.0},
    {"lambdarank_num_pair_per_sample": 3},
    {"lambdarank_num_pair_per_sample": 80},
    {"lambdarank_pair_method": "mean", "lambdarank_num_pair_per_sample": 4, "seed": 5},
    {"lambdarank_normalization": False, "lambdarank_score_normalization": False},
])
def test_unbiased_parameters_vs_xgboost(extra, tmp_path):
    x, y, qid, _, w = make(seed=4, weights=True, binary=True)
    extra = {"lambdarank_unbiased": True, **extra}
    b, d, _ = xgb_train("rank:ndcg", x, y, qid, w, 10, extra, evals=False)
    m, _ = fb_train("rank:ndcg", x, y, qid, w, 10, extra, evals=False)
    np.testing.assert_array_equal(f32_bits(m.predict(x)), f32_bits(b.predict(d)))
    assert _fb_bias(m, tmp_path) == _xgb_bias(b)


def test_unbiased_graded_labels_and_early_stopping(tmp_path):
    x, y, qid, _, _ = make(seed=1)
    vx, vy, vqid, _, _ = make(seed=2, sizes=[6, 12, 30, 4, 18])
    extra = {"lambdarank_unbiased": True, "eval_metric": "ndcg@5"}
    d, dv = xgb.DMatrix(x, y, qid=qid), xgb.DMatrix(vx, vy, qid=vqid)
    p = dict(objective="rank:ndcg", tree_method="hist", nthread=1, max_depth=3, max_bin=16, **extra)
    xb = xgb.train(p, d, 40, evals=[(dv, "v")], verbose_eval=False, early_stopping_rounds=4)
    m = fb.train({"objective": "rank:ndcg", **PARAMS, **extra}, x, label=y, qid=qid,
                 num_boost_round=40, evals=[(vx, vy, "v")], eval_qid=[vqid],
                 early_stopping_rounds=4, quiet=True)
    assert m.best_iteration == xb.best_iteration
    np.testing.assert_array_equal(f32_bits(m.predict(vx)), f32_bits(xb.predict(dv)))
    assert _fb_bias(m, tmp_path) == _xgb_bias(xb)


def test_unbiased_json_round_trip_with_xgboost(tmp_path):
    x, y, qid, _, w = make(weights=True, binary=True)
    extra = {"lambdarank_unbiased": True, "lambdarank_bias_norm": 0.7}
    b, d, _ = xgb_train("rank:pairwise", x, y, qid, w, 5, extra, evals=False)
    m, _ = fb_train("rank:pairwise", x, y, qid, w, 5, extra, evals=False)
    # FerrisBoost model in XGBoost
    path = os.fspath(tmp_path / "fb.json")
    m.save_model(path, quiet=True)
    loaded = xgb.Booster()
    loaded.load_model(path)
    np.testing.assert_array_equal(f32_bits(loaded.predict(d)), f32_bits(m.predict(x)))
    cfg = json.loads(loaded.save_config())["learner"]["objective"]
    assert cfg["lambdarank_param"]["lambdarank_unbiased"] == "1"
    assert cfg["ti+"] == _xgb_bias(b)[0]
    # XGBoost model in FerrisBoost, re-saved with the same biases
    xpath = os.fspath(tmp_path / "xgb.json")
    b.save_model(xpath)
    back = fb.Model.load_model(xpath, quiet=True)
    np.testing.assert_array_equal(f32_bits(back.predict(x)), f32_bits(b.predict(d)))
    assert _fb_bias(back, tmp_path, "again.json") == _xgb_bias(b)


def test_unbiased_models_do_not_depend_on_threads():
    x, y, qid, _, w = make(weights=True, binary=True)
    extra = {"lambdarank_unbiased": True}
    p1, _ = fb_train("rank:ndcg", x, y, qid, w, 6, {**extra, "nthread": 1}, evals=False)
    p4, _ = fb_train("rank:ndcg", x, y, qid, w, 6, {**extra, "nthread": 4}, evals=False)
    np.testing.assert_array_equal(f32_bits(p1.predict(x)), f32_bits(p4.predict(x)))


def test_unbiased_sklearn_ranker():
    from ferrisboost.sklearn import FerrisBoostRanker

    x, y, qid, _, _ = make(binary=True)
    r = FerrisBoostRanker(n_estimators=5, max_depth=3, max_bin=16, n_jobs=1,
                          lambdarank_unbiased=True, lambdarank_bias_norm=0.5)
    r.fit(x, y, qid=qid)
    ref, _ = fb_train("rank:ndcg", x, y, qid, None, 5,
                      {"lambdarank_unbiased": True, "lambdarank_bias_norm": 0.5}, evals=False)
    np.testing.assert_array_equal(f32_bits(r.predict(x)), f32_bits(ref.predict(x)))


@cuda
@pytest.mark.parametrize("obj,pair_method", list(itertools.product(
    ["rank:ndcg", "rank:pairwise", "rank:map"], ["topk", "mean"])))
def test_unbiased_gpu_exact_is_byte_identical_to_cpu(obj, pair_method, tmp_path):
    x, y, qid, _, w = make(weights=True, binary=True)
    extra = {"lambdarank_unbiased": True, "lambdarank_pair_method": pair_method}
    cpu, _ = fb_train(obj, x, y, qid, w, 6, extra, evals=False)
    gpu, _ = fb_train(obj, x, y, qid, w, 6, {**extra, "device": "cuda", "gpu_math": "exact"},
                      evals=False)
    np.testing.assert_array_equal(f32_bits(cpu.predict(x)), f32_bits(gpu.predict(x)))
    assert _fb_bias(cpu, tmp_path, "c.json") == _fb_bias(gpu, tmp_path, "g.json")


@cuda
def test_unbiased_gpu_fast_is_an_error():
    x, y, qid, _, _ = make(binary=True)
    with pytest.raises(Exception, match="lambdarank_unbiased.*gpu_math"):
        fb_train("rank:ndcg", x, y, qid, None, 2,
                 {"lambdarank_unbiased": True, "device": "cuda", "gpu_math": "fast"}, evals=False)


# ---------------------------------------------------------------- auc / aucpr per group
def _xgb_metric_on_scores(metric, scores, labels, qid, w):
    d = xgb.DMatrix(np.zeros((len(labels), 1), np.float32), labels, qid=qid, weight=w)
    d.set_base_margin(scores)
    booster = xgb.train(dict(objective="rank:ndcg", tree_method="hist", nthread=1, base_score=0.0,
                             eval_metric=metric), d, 0)
    return float(booster.eval(d, "t").split(":")[-1])


@pytest.mark.parametrize("metric", ["auc", "aucpr"])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_group_auc_metrics_vs_xgboost(metric, weighted, seed):
    x, y, qid, _, w = make(seed=seed, weights=weighted)
    yb = (y > 1).astype(np.float32)
    m, _ = fb_train("rank:pairwise", x, yb, qid, w, 4, evals=False)
    scores = m.predict(x)
    # also with tied scores (a constant model) and a heavily tied one
    for s in (scores, np.zeros_like(scores), np.round(scores, 1).astype(np.float32)):
        want = _xgb_metric_on_scores(metric, s, yb, qid, w)
        got = fb._core.rank_score(metric, s, yb, qid.astype(np.uint64), w, True)
        assert got == pytest.approx(want, rel=1e-9, abs=1e-12), (metric, got, want)


@pytest.mark.parametrize("metric", ["auc", "aucpr"])
@pytest.mark.parametrize("obj", ["rank:ndcg", "rank:map"])
def test_group_auc_in_training_and_early_stopping(metric, obj):
    x, y, qid, _, _ = make(seed=1, binary=True)
    vx, vy, vqid, _, vw = make(seed=2, sizes=[6, 12, 30, 4, 18, 3, 9], binary=True, weights=True)
    d = xgb.DMatrix(x, y, qid=qid)
    dv = xgb.DMatrix(vx, vy, qid=vqid, weight=vw)
    p = dict(objective=obj, tree_method="hist", nthread=1, max_depth=3, max_bin=16,
             eval_metric=metric)
    xres = {}
    xb = xgb.train(p, d, 40, evals=[(d, "t"), (dv, "v")], evals_result=xres, verbose_eval=False,
                   early_stopping_rounds=5)
    res = {}
    m = fb.train({"objective": obj, "eval_metric": metric, **PARAMS}, x, label=y, qid=qid,
                 num_boost_round=40, evals=[(x, y, "t"), (vx, vy, vw, "v")],
                 eval_qid=[qid, vqid], early_stopping_rounds=5, evals_result=res, quiet=True)
    assert m.best_iteration == xb.best_iteration
    for name in ("t", "v"):
        np.testing.assert_allclose(res[name][metric], xres[name][metric], rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(f32_bits(m.predict(vx)), f32_bits(xb.predict(dv)))


def test_group_auc_degenerate_groups():
    # groups of size < 3 have no ROC AUC; groups without a positive or a negative have no PR-AUC
    qid = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2])
    y = np.array([1, 0, 1, 0, 0, 0, 0, 0, 0], np.float32)
    s = np.array([0.3, 0.1, 0.9, 0.5, 0.2, 0.4, 0.3, 0.2, 0.1], np.float32)
    for metric in ("auc", "aucpr"):
        want = _xgb_metric_on_scores(metric, s, y, qid, None)
        assert fb._core.rank_score(metric, s, y, qid.astype(np.uint64), None, True) == \
            pytest.approx(want, abs=1e-12)
    # nothing scorable -> NaN, like XGBoost
    allneg = np.zeros_like(y)
    assert np.isnan(fb._core.rank_score("aucpr", s, allneg, qid.astype(np.uint64), None, True))
    with pytest.raises(ValueError, match="binary relevance"):
        fb._core.rank_score("aucpr", s, y * 3, qid.astype(np.uint64), None, True)


def test_ranking_objective_without_groups_says_how_to_pass_them():
    x, y, _, _, _ = make()
    with pytest.raises(ValueError, match="qid.*group|group.*qid"):
        fb.train(dict(objective="rank:ndcg", **PARAMS), x, label=y, quiet=True)


@pytest.mark.parametrize("metric", ["auc", "aucpr"])
def test_binary_logistic_with_query_groups_scores_auc_per_group(metric):
    # XGBoost scores auc / aucpr per group whenever the data has qid, whatever the objective
    x, y, qid, _, _ = make(seed=1, binary=True)
    vx, vy, vqid, _, _ = make(seed=2, sizes=[6, 12, 30, 4, 18, 3, 9], binary=True)
    d, dv = xgb.DMatrix(x, y, qid=qid), xgb.DMatrix(vx, vy, qid=vqid)
    p = dict(objective="binary:logistic", tree_method="hist", nthread=1, max_depth=3, max_bin=16,
             eval_metric=metric)
    xres = {}
    xb = xgb.train(p, d, 30, evals=[(d, "t"), (dv, "v")], evals_result=xres, verbose_eval=False,
                   early_stopping_rounds=4)
    res = {}
    m = fb.train({"objective": "binary:logistic", "eval_metric": metric, **PARAMS}, x, label=y,
                 qid=qid, num_boost_round=30, evals=[(x, y, "t"), (vx, vy, "v")],
                 eval_qid=[qid, vqid], early_stopping_rounds=4, evals_result=res, quiet=True)
    assert m.best_iteration == xb.best_iteration
    for name in ("t", "v"):
        np.testing.assert_allclose(res[name][metric], xres[name][metric], rtol=1e-6, atol=1e-7)
    np.testing.assert_array_equal(f32_bits(m.predict(vx)), f32_bits(xb.predict(dv)))
    # without groups the same metric is the pooled binary one
    plain = {}
    fb.train({"objective": "binary:logistic", "eval_metric": metric, **PARAMS}, x, label=y,
             num_boost_round=3, evals=[(x, y, "t")], evals_result=plain, quiet=True)
    assert plain["t"][metric][0] != pytest.approx(res["t"][metric][0], abs=1e-9)


def test_binary_logistic_groups_with_row_weights_and_auc_is_an_error():
    x, y, qid, _, _ = make(binary=True)
    w = np.ones(len(y), np.float32)
    with pytest.raises(Exception, match="one weight per group"):
        fb.train({"objective": "binary:logistic", "eval_metric": "auc", **PARAMS}, x, label=y,
                 num_boost_round=2, evals=[(x, y, w, "t")], eval_qid=[qid], quiet=True)
    # other metrics keep the per-row weights
    fb.train({"objective": "binary:logistic", "eval_metric": "logloss", **PARAMS}, x, label=y,
             num_boost_round=2, evals=[(x, y, w, "t")], eval_qid=[qid], quiet=True)


def test_file_qid_column_is_for_ranking_objectives_only(tmp_path):
    x, y, qid, _, _ = make(binary=True)
    path = _write(tmp_path, x, y, qid)
    with pytest.raises(ValueError, match="only used by the rank"):
        fb.train(dict(objective="binary:logistic", **PARAMS), path, label="rel",
                 qid="query_id", quiet=True)
