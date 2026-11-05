"""样本权重(`weight=`)和 `eval_metric`:API、语义,以及和 XGBoost 的对齐。"""
import csv as csv_module
import re

import numpy as np
import pytest

import ferrisboost as fb


def toy(n=400, m=5, seed=0):
    # 整数特征:cuts 只取决于出现过哪些值,删行/置零权重不会改变分箱。
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 8, size=(n, m)).astype(np.float32)
    y = (x[:, 0] - x[:, 1] + rng.normal(0, 0.5, n) > 0).astype(np.float32)
    w = rng.uniform(0.1, 3.0, n).astype(np.float32)
    return x, y, w


BASE = {"objective": "binary:logistic", "max_depth": 3, "eta": 0.3, "nthread": 1}


def model_bytes(model, tmp_path, name):
    path = tmp_path / f"{name}.json"
    model.save_model(str(path), quiet=True)
    return path.read_bytes()


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z.astype(np.float64)))


# ---------------------------------------------------------------- 语义
def test_all_ones_weight_is_byte_identical_to_unweighted(tmp_path):
    x, y, _ = toy()
    a = fb.train(BASE, x, label=y, num_boost_round=5, quiet=True)
    b = fb.train(BASE, x, label=y, weight=np.ones(len(y)), num_boost_round=5, quiet=True)
    assert model_bytes(a, tmp_path, "a") == model_bytes(b, tmp_path, "b")


def test_weights_change_the_model(tmp_path):
    x, y, w = toy()
    a = fb.train(BASE, x, label=y, num_boost_round=5, quiet=True)
    b = fb.train(BASE, x, label=y, weight=w, num_boost_round=5, quiet=True)
    assert model_bytes(a, tmp_path, "a") != model_bytes(b, tmp_path, "b")


def test_zero_weight_rows_match_dropping_them():
    x, y, w = toy()
    keep = np.arange(len(y)) % 3 != 0
    w0 = np.where(keep, w, 0.0).astype(np.float32)
    zeroed = fb.train(BASE, x, label=y, weight=w0, num_boost_round=5, quiet=True)
    dropped = fb.train(BASE, x[keep], label=y[keep], weight=w[keep],
                       num_boost_round=5, quiet=True)
    np.testing.assert_array_equal(zeroed.predict(x), dropped.predict(x))


def test_weighted_eval_metrics_follow_the_xgboost_formula():
    x, y, w = toy()
    xv, yv, wv = toy(200, seed=1)
    res = {}
    model = fb.train({**BASE, "eval_metric": ["logloss", "rmse", "error"]}, x, label=y,
                     weight=w, num_boost_round=4, evals=[(xv, yv, wv, "v")],
                     evals_result=res, quiet=True)
    assert list(res["v"]) == ["logloss", "rmse", "error"]
    p = model.predict(xv).astype(np.float64)
    wd = wv.astype(np.float64)
    ll = -(yv * np.log(np.clip(p, 1e-15, 1 - 1e-15))
           + (1 - yv) * np.log(np.clip(1 - p, 1e-15, 1 - 1e-15)))
    np.testing.assert_allclose(res["v"]["logloss"][-1], (wd * ll).sum() / wd.sum(), rtol=1e-5)
    rmse = np.sqrt((wd * (yv - p) ** 2).sum() / wd.sum())
    np.testing.assert_allclose(res["v"]["rmse"][-1], rmse, rtol=1e-5)
    err = (wd * np.where(p > 0.5, 1 - yv, yv)).sum() / wd.sum()
    np.testing.assert_allclose(res["v"]["error"][-1], err, rtol=1e-5)


def test_unweighted_and_weighted_eval_tuples_can_be_mixed():
    x, y, w = toy()
    xv, yv, wv = toy(200, seed=1)
    res = {}
    fb.train(BASE, x, label=y, num_boost_round=3, quiet=True, evals_result=res,
             evals=[(xv, yv, "plain"), (xv, yv, wv, "weighted"), (xv, yv, None, "none")])
    assert res["plain"]["logloss"] == res["none"]["logloss"]
    assert res["plain"]["logloss"] != res["weighted"]["logloss"]


def test_auc_early_stopping_maximizes():
    x, y, w = toy(600)
    xv, yv, _ = toy(300, seed=2)
    res = {}
    model = fb.train({**BASE, "eval_metric": ["logloss", "auc"], "eta": 0.5}, x, label=y,
                     num_boost_round=40, evals=[(xv, yv, "v")],
                     early_stopping_rounds=3, evals_result=res, quiet=True)
    auc = res["v"]["auc"]
    assert model.best_iteration == int(np.argmax(auc))
    assert model.best_score == pytest.approx(max(auc))


def test_auc_matches_pairwise_definition():
    x, y, w = toy()
    xv, yv, wv = toy(150, seed=3)
    res = {}
    model = fb.train({**BASE, "eval_metric": "auc"}, x, label=y, num_boost_round=3,
                     evals=[(xv, yv, wv, "v")], evals_result=res, quiet=True)
    p = model.predict(xv)
    pos, neg = yv == 1, yv == 0
    diff = p[pos][:, None] - p[neg][None, :]
    score = np.where(diff > 0, 1.0, np.where(diff == 0, 0.5, 0.0))
    pw = wv[pos][:, None].astype(np.float64) * wv[neg][None, :]
    np.testing.assert_allclose(res["v"]["auc"][-1], (score * pw).sum() / pw.sum(), rtol=1e-5)


def test_rmse_metric_with_logistic_uses_probabilities():
    x, y, _ = toy()
    res = {}
    model = fb.train({**BASE, "eval_metric": "rmse"}, x, label=y, num_boost_round=3,
                     evals=[(x, y, "t")], evals_result=res, quiet=True)
    p = model.predict(x).astype(np.float64)
    np.testing.assert_allclose(res["t"]["rmse"][-1], np.sqrt(((y - p) ** 2).mean()), rtol=1e-5)


def test_metrics_match_xgboost_on_the_same_model(tmp_path):
    xgb = pytest.importorskip("xgboost")
    x, y, w = toy()
    xv, yv, wv = toy(250, seed=4)
    res = {}
    metrics = ["logloss", "error", "auc", "rmse"]
    model = fb.train({**BASE, "eval_metric": metrics}, x, label=y, weight=w,
                     num_boost_round=6, evals=[(xv, yv, wv, "v")], evals_result=res,
                     quiet=True)
    path = tmp_path / "m.json"
    model.save_model(str(path), quiet=True)
    bst = xgb.Booster(model_file=str(path))
    bst.set_param([("eval_metric", m) for m in metrics])
    line = bst.eval(xgb.DMatrix(xv, label=yv, weight=wv), "v")
    theirs = {k: float(v) for k, v in re.findall(r"v-(\w+):([0-9.eE+-]+)", line)}
    for m in metrics:
        assert res["v"][m][-1] == pytest.approx(theirs[m], rel=1e-5, abs=1e-6), (m, line)


# ---------------------------------------------------------------- 错误
@pytest.mark.parametrize("bad, match", [
    (lambda n: -np.ones(n), "负数"),
    (lambda n: np.full(n, np.nan), "NaN"),
    (lambda n: np.full(n, np.inf), "NaN 或 inf"),
    (lambda n: np.ones(n - 1), "数据有"),
])
def test_invalid_array_weights_are_rejected(bad, match):
    x, y, _ = toy()
    with pytest.raises(ValueError, match=match):
        fb.train(BASE, x, label=y, weight=bad(len(y)), num_boost_round=1, quiet=True)


def test_column_name_weight_with_array_input_is_rejected():
    x, y, _ = toy()
    with pytest.raises(ValueError, match="一维数组"):
        fb.train(BASE, x, label=y, weight="w", num_boost_round=1, quiet=True)


def test_invalid_eval_weights_are_rejected():
    x, y, _ = toy()
    with pytest.raises(ValueError, match="evals\\[v\\] 的 weight"):
        fb.train(BASE, x, label=y, num_boost_round=1, quiet=True,
                 evals=[(x, y, -np.ones(len(y)), "v")])


@pytest.mark.parametrize("params, match", [
    ({"eval_metric": "mae"}, "不支持"),
    ({"eval_metric": "auc", "objective": "reg:squarederror"}, "binary:logistic"),
    ({"eval_metric": []}, "字符串"),
    ({"eval_metric": 3}, "字符串"),
])
def test_invalid_eval_metric_is_rejected(params, match):
    x, y, _ = toy()
    with pytest.raises(ValueError, match=match):
        fb.train({**BASE, **params}, x, label=y, num_boost_round=1, quiet=True)


# ---------------------------------------------------------------- 文件入口
def write_csv(path, x, y, w, header=True):
    with open(path, "w", newline="") as f:
        out = csv_module.writer(f)
        if header:
            out.writerow([f"f{i}" for i in range(x.shape[1])] + ["w", "label"])
        for row, label, weight in zip(x, y, w):
            out.writerow([*(repr(float(v)) for v in row), repr(float(weight)), int(label)])


def write_parquet(path, x, y, w):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    cols = {f"f{i}": x[:, i] for i in range(x.shape[1])}
    cols["w"] = w
    cols["label"] = y
    pq.write_table(pa.table(cols), str(path))


@pytest.mark.parametrize("fmt", ["csv", "parquet"])
def test_file_weight_column_matches_array_weights(tmp_path, fmt):
    x, y, w = toy()
    xv, yv, wv = toy(200, seed=1)
    train_path, valid_path = tmp_path / f"t.{fmt}", tmp_path / f"v.{fmt}"
    writer = write_csv if fmt == "csv" else write_parquet
    writer(train_path, x, y, w)
    writer(valid_path, xv, yv, wv)
    res_file, res_arr = {}, {}
    from_file = fb.train(BASE, str(train_path), label="label", weight="w",
                         num_boost_round=4, evals=[(str(valid_path), "v")],
                         evals_result=res_file, quiet=True)
    from_arr = fb.train(BASE, x, label=y, weight=w, num_boost_round=4,
                        evals=[(xv, yv, wv, "v")], evals_result=res_arr, quiet=True)
    # 权重列不进特征
    assert from_file.n_features == x.shape[1]
    np.testing.assert_array_equal(from_file.predict(xv), from_arr.predict(xv))
    assert res_file == res_arr


def test_multi_file_parquet_weights(tmp_path):
    x, y, w = toy()
    half = len(y) // 2
    write_parquet(tmp_path / "a.parquet", x[:half], y[:half], w[:half])
    write_parquet(tmp_path / "b.parquet", x[half:], y[half:], w[half:])
    from_files = fb.train(BASE, [str(tmp_path / "a.parquet"), str(tmp_path / "b.parquet")],
                          label="label", weight="w", num_boost_round=4, quiet=True)
    from_arr = fb.train(BASE, x, label=y, weight=w, num_boost_round=4, quiet=True)
    np.testing.assert_array_equal(from_files.predict(x), from_arr.predict(x))


def test_headerless_csv_weight_index(tmp_path):
    x, y, w = toy()
    path = tmp_path / "nohdr.csv"
    write_csv(path, x, y, w, header=False)
    m = x.shape[1]
    model = fb.train(BASE, str(path), label=m + 1, weight=m, header=False,
                     num_boost_round=4, quiet=True)
    ref = fb.train(BASE, x, label=y, weight=w, num_boost_round=4, quiet=True)
    np.testing.assert_array_equal(model.predict(x), ref.predict(x))


def test_parquet_quantized_cache_records_the_weight_column(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    x, y, w = toy()
    cols = {f"f{i}": x[:, i] for i in range(x.shape[1])}
    cols["w"], cols["w2"], cols["label"] = w, w * 2, y
    path = tmp_path / "t.parquet"
    pq.write_table(pa.table(cols), str(path))
    cache = tmp_path / "cache"
    fb.train({**BASE, "cache_path": str(cache)}, str(path), label="label", weight="w",
             num_boost_round=1, quiet=True)
    # 换一个权重列:特征数相同,但特征集合不同 —— 旧缓存必须拒绝。
    with pytest.raises(Exception, match="权重列"):
        fb.train({**BASE, "cache_path": str(cache)}, str(path), label="label",
                 weight="w2", num_boost_round=1, quiet=True)


@pytest.mark.parametrize("weight, match", [
    ("label", "同一列"),
    ("nope", "nope"),
    (3, "列名字符串"),
])
def test_bad_file_weight_column(tmp_path, weight, match):
    x, y, w = toy()
    path = tmp_path / "t.csv"
    write_csv(path, x, y, w)
    with pytest.raises(ValueError, match=match):
        fb.train(BASE, str(path), label="label", weight=weight, num_boost_round=1, quiet=True)


def test_negative_weight_in_file_is_rejected(tmp_path):
    x, y, w = toy()
    w[7] = -1.0
    path = tmp_path / "t.csv"
    write_csv(path, x, y, w)
    with pytest.raises(Exception, match="非负"):
        fb.train(BASE, str(path), label="label", weight="w", num_boost_round=1, quiet=True)


def test_runtime_echo_reports_weights(capfd):
    x, y, w = toy()
    fb.train(BASE, x, label=y, weight=w, num_boost_round=1)
    assert "weights=on" in capfd.readouterr().err
    fb.train(BASE, x, label=y, num_boost_round=1)
    assert "weights=off" in capfd.readouterr().err


# ---------------------------------------------------------------- GPU
def _has_cuda():
    try:
        x, y, _ = toy(64, 4)
        fb.train({"device": "cuda", "max_depth": 2}, x, y, num_boost_round=1, quiet=True)
        return True
    except Exception as e:  # noqa: BLE001
        if "CUDA" in str(e) or "cuda" in str(e):
            return False
        raise


cuda = pytest.mark.skipif(not _has_cuda(), reason="没有可用的 CUDA 构建或设备")


@cuda
def test_gpu_exact_weights_match_cpu(tmp_path):
    x, y, w = toy(2000, 8)
    cpu = fb.train(BASE, x, label=y, weight=w, num_boost_round=5, quiet=True)
    gpu = fb.train({**BASE, "device": "cuda", "gpu_math": "exact"}, x, label=y, weight=w,
                   num_boost_round=5, quiet=True)
    assert model_bytes(cpu, tmp_path, "cpu") == model_bytes(gpu, tmp_path, "gpu")


@cuda
def test_gpu_fast_weights_are_deterministic_and_planned(tmp_path, capfd):
    x, y, w = toy(2000, 8)
    p = {**BASE, "device": "cuda", "gpu_math": "fast"}
    a = fb.train(p, x, label=y, weight=w, num_boost_round=5)
    capfd.readouterr()
    b = fb.train(p, x, label=y, weight=w, num_boost_round=5, quiet=True)
    plain = fb.train(p, x, label=y, num_boost_round=5, quiet=True)
    assert model_bytes(a, tmp_path, "a") == model_bytes(b, tmp_path, "b")
    assert model_bytes(a, tmp_path, "a") != model_bytes(plain, tmp_path, "plain")


# ---------------------------------------------------------------- 加权分箱
def continuous(n=3000, m=4, seed=0):
    # 连续特征:唯一值远多于 max_bin,cuts 由 sketch 决定 —— 权重会移动 cuts。
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, m)).astype(np.float32)
    y = (x[:, 0] + 0.5 * x[:, 1] + rng.normal(0, 0.5, n) > 0.8).astype(np.float32)
    w = np.where(x[:, 0] > 1.0, 15.0, 1.0).astype(np.float32)
    return x, y, w


CONT = {**BASE, "max_bin": 16}


def test_weighted_binning_all_ones_is_byte_identical(tmp_path):
    x, y, _ = continuous()
    a = fb.train(CONT, x, label=y, num_boost_round=5, quiet=True)
    b = fb.train(CONT, x, label=y, weight=np.ones(len(y)), num_boost_round=5, quiet=True)
    assert model_bytes(a, tmp_path, "a") == model_bytes(b, tmp_path, "b")


def test_weighted_binning_zero_weight_rows_match_dropping_them_on_continuous_data():
    # 0 权重的行既不进梯度也不进 sketch,所以连续数据上也和删行逐位一致。
    x, y, w = continuous()
    keep = np.arange(len(y)) % 4 != 0
    zeroed = fb.train(CONT, x, label=y, weight=np.where(keep, w, 0).astype(np.float32),
                      num_boost_round=5, quiet=True)
    dropped = fb.train(CONT, x[keep], label=y[keep], weight=w[keep],
                       num_boost_round=5, quiet=True)
    np.testing.assert_array_equal(zeroed.predict(x), dropped.predict(x))


def test_weights_move_the_split_thresholds():
    # 重权重集中在 x0 > 1 的尾部:加权 cuts 在那里更密,树能切出更细的阈值。
    x, y, w = continuous()
    plain = fb.train(CONT, x, label=y, num_boost_round=3, quiet=True)
    weighted = fb.train(CONT, x, label=y, weight=w, num_boost_round=3, quiet=True)
    ones = fb.train(CONT, x, label=y, weight=np.ones(len(y)), num_boost_round=3, quiet=True)

    def thresholds(model, path):
        import json
        model.save_model(str(path), quiet=True)
        trees = json.loads(path.read_text())["learner"]["gradient_booster"]["model"]["trees"]
        return {round(v, 6) for t in trees
                for v, f, l in zip(t["split_conditions"], t["split_indices"], t["left_children"])
                if l != -1 and f == 0}

    import tempfile
    from pathlib import Path
    with tempfile.TemporaryDirectory() as d:
        tp, tw, to = (thresholds(m, Path(d) / f"{i}.json")
                      for i, m in enumerate((plain, weighted, ones)))
    assert tp == to
    assert tw != tp, "加权 cuts 应该给出不同的 x0 阈值"


@pytest.mark.parametrize("fmt", ["csv", "parquet", "parquet-multi"])
def test_weighted_binning_is_identical_across_input_paths(tmp_path, fmt):
    x, y, w = continuous()
    xv, yv, wv = continuous(800, seed=1)
    ref_res, res = {}, {}
    ref = fb.train(CONT, x, label=y, weight=w, num_boost_round=5,
                   evals=[(xv, yv, wv, "v")], evals_result=ref_res, quiet=True)
    if fmt == "csv":
        write_csv(tmp_path / "t.csv", x, y, w)
        write_csv(tmp_path / "v.csv", xv, yv, wv)
        train_spec, valid = str(tmp_path / "t.csv"), str(tmp_path / "v.csv")
    else:
        write_parquet(tmp_path / "v.parquet", xv, yv, wv)
        valid = str(tmp_path / "v.parquet")
        if fmt == "parquet":
            write_parquet(tmp_path / "t.parquet", x, y, w)
            train_spec = str(tmp_path / "t.parquet")
        else:
            cut = len(y) // 3
            write_parquet(tmp_path / "a.parquet", x[:cut], y[:cut], w[:cut])
            write_parquet(tmp_path / "b.parquet", x[cut:], y[cut:], w[cut:])
            train_spec = [str(tmp_path / "a.parquet"), str(tmp_path / "b.parquet")]
    model = fb.train(CONT, train_spec, label="label", weight="w", num_boost_round=5,
                     evals=[(valid, "v")], evals_result=res, quiet=True)
    np.testing.assert_array_equal(model.predict(xv), ref.predict(xv))
    assert res == ref_res


def test_weighted_binning_parquet_cache_reuse_keeps_weighted_cuts(tmp_path):
    x, y, w = continuous()
    write_parquet(tmp_path / "t.parquet", x, y, w)
    p = {**CONT, "cache_path": str(tmp_path / "cache")}
    first = fb.train(p, str(tmp_path / "t.parquet"), label="label", weight="w",
                     num_boost_round=4, quiet=True)
    reused = fb.train(p, str(tmp_path / "t.parquet"), label="label", weight="w",
                      num_boost_round=4, quiet=True)
    np.testing.assert_array_equal(first.predict(x), reused.predict(x))
    ref = fb.train(CONT, x, label=y, weight=w, num_boost_round=4, quiet=True)
    np.testing.assert_array_equal(first.predict(x), ref.predict(x))


@cuda
def test_gpu_weighted_binning_matches_cpu(tmp_path):
    x, y, w = continuous(4000, 8)
    cpu = fb.train(CONT, x, label=y, weight=w, num_boost_round=5, quiet=True)
    gpu = fb.train({**CONT, "device": "cuda", "gpu_math": "exact"}, x, label=y, weight=w,
                   num_boost_round=5, quiet=True)
    assert model_bytes(cpu, tmp_path, "cpu") == model_bytes(gpu, tmp_path, "gpu")
