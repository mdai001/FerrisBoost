"""colsample_bylevel / colsample_bynode:XGBoost 的嵌套采样语义,FerrisBoost 自己的确定性序列。

随机序列和 XGBoost 不同(它的 CPU/GPU 序列本来就不同),所以这里不和 XGBoost 逐位比,
只验证语义(嵌套、个数、只限制 split 枚举)和确定性(线程数 / backend 无关)。
"""
import json

import numpy as np
import pytest

import ferrisboost as fb

N_FEATURES = 12


def make(n=3000, m=N_FEATURES, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, m)).astype(np.float32)
    x[rng.random(x.shape) < 0.03] = np.nan
    z = x[:, 0] + 0.5 * x[:, 1] * np.nan_to_num(x[:, 2]) + 0.3 * np.nan_to_num(x[:, 3])
    y = (z + rng.normal(scale=0.5, size=n) > 0).astype(np.float32)
    return x, y


def model_json(model, tmp_path, name="m"):
    path = tmp_path / f"{name}.json"
    model.save_model(str(path), quiet=True)
    return path.read_bytes()


def trees_of(model, tmp_path):
    doc = json.loads(model_json(model, tmp_path))
    return doc["learner"]["gradient_booster"]["model"]["trees"]


def depth_to_features(tree):
    """{深度: 该深度所有分裂用到的特征集合}"""
    left, right = tree["left_children"], tree["right_children"]
    feat = tree["split_indices"]
    out: dict[int, set[int]] = {}
    stack = [(0, 0)]
    while stack:
        node, depth = stack.pop()
        if left[node] == -1:
            continue
        out.setdefault(depth, set()).add(feat[node])
        stack.append((left[node], depth + 1))
        stack.append((right[node], depth + 1))
    return out


BASE = {"objective": "binary:logistic", "max_depth": 5, "eta": 0.3, "seed": 3, "nthread": 1}


def test_rate_one_is_identical_to_unset(tmp_path):
    x, y = make()
    a = fb.train(BASE, x, label=y, num_boost_round=5, quiet=True)
    b = fb.train({**BASE, "colsample_bylevel": 1.0, "colsample_bynode": 1.0}, x, label=y,
                 num_boost_round=5, quiet=True)
    assert model_json(a, tmp_path, "a") == model_json(b, tmp_path, "b")


@pytest.mark.parametrize("extra", [{"colsample_bylevel": 0.5}, {"colsample_bynode": 0.5},
                                   {"colsample_bylevel": 0.7, "colsample_bynode": 0.6,
                                    "colsample_bytree": 0.8}])
def test_deterministic_thread_invariant_and_different_from_full(tmp_path, extra):
    x, y = make()
    p = {**BASE, **extra}
    a = fb.train(p, x, label=y, num_boost_round=8, quiet=True)
    b = fb.train(p, x, label=y, num_boost_round=8, quiet=True)
    c = fb.train({**p, "nthread": 4}, x, label=y, num_boost_round=8, quiet=True)
    d = fb.train({**p, "nthread": 3, "cols_per_block": 5}, x, label=y, num_boost_round=8,
                 quiet=True)
    full = fb.train(BASE, x, label=y, num_boost_round=8, quiet=True)
    ja = model_json(a, tmp_path, "a")
    assert ja == model_json(b, tmp_path, "b")
    assert ja == model_json(c, tmp_path, "c"), "线程数不能改变模型"
    assert ja == model_json(d, tmp_path, "d"), "列块划分不能改变模型"
    assert ja != model_json(full, tmp_path, "full")
    assert ja != model_json(
        fb.train({**p, "seed": 4}, x, label=y, num_boost_round=8, quiet=True), tmp_path, "s4")


def test_bylevel_uses_one_feature_set_per_depth(tmp_path):
    # 12 个特征、bylevel = 1/12 → 每层只采 1 个特征:同一层所有分裂必须是同一个特征。
    x, y = make()
    m = fb.train({**BASE, "colsample_bylevel": 1.0 / N_FEATURES + 1e-3}, x, label=y,
                 num_boost_round=6, quiet=True)
    seen_multi_node_level = False
    for tree in trees_of(m, tmp_path):
        for feats in depth_to_features(tree).values():
            assert len(feats) == 1, feats
        left = tree["left_children"]
        # 至少某棵树的某一层有 >= 2 个分裂节点,上面的断言才有意义。
        seen_multi_node_level |= sum(1 for v in left if v != -1) >= 3
    assert seen_multi_node_level


def test_sets_are_nested_bytree_bylevel_bynode(tmp_path):
    x, y = make()
    # bytree 6 个特征;每层再采 3 个。
    m = fb.train({**BASE, "colsample_bytree": 0.5, "colsample_bylevel": 0.5}, x, label=y,
                 num_boost_round=8, quiet=True)
    for tree in trees_of(m, tmp_path):
        per_depth = depth_to_features(tree)
        used = set().union(*per_depth.values()) if per_depth else set()
        assert len(used) <= 6
        for feats in per_depth.values():
            assert len(feats) <= 3
    # bynode 在 bytree 里面:整棵树用到的特征不超过 bytree 个数。
    m = fb.train({**BASE, "colsample_bytree": 0.5, "colsample_bynode": 0.5}, x, label=y,
                 num_boost_round=8, quiet=True)
    for tree in trees_of(m, tmp_path):
        used = set().union(*depth_to_features(tree).values()) if depth_to_features(tree) else set()
        assert len(used) <= 6


def test_bynode_gives_different_features_within_a_level(tmp_path):
    x, y = make()
    m = fb.train({**BASE, "colsample_bynode": 0.2, "max_depth": 6}, x, label=y,
                 num_boost_round=10, quiet=True)
    multi = any(len(f) > 1 for t in trees_of(m, tmp_path) for f in depth_to_features(t).values())
    assert multi, "bynode 应该让同层不同节点用到不同特征"


def test_quality_is_sane():
    x, y = make(n=6000, seed=2)
    vx, vy = make(n=3000, seed=9)

    def logloss(params):
        res = {}
        fb.train({**BASE, **params, "eval_metric": "logloss"}, x, label=y, num_boost_round=30,
                 evals=[(vx, vy, "v")], evals_result=res, quiet=True)
        return res["v"]["logloss"][-1]

    base = logloss({})
    for extra in ({"colsample_bylevel": 0.5}, {"colsample_bynode": 0.5},
                  {"colsample_bylevel": 0.7, "colsample_bynode": 0.7}):
        assert logloss(extra) < base * 1.15, extra


def test_multiclass_supports_nested_sampling(tmp_path):
    rng = np.random.default_rng(1)
    x = rng.normal(size=(1500, N_FEATURES)).astype(np.float32)
    y = ((x[:, 0] > 0).astype(int) + (x[:, 1] > 0.3).astype(int)).astype(np.float32)
    p = {"objective": "multi:softprob", "num_class": 3, "max_depth": 4, "colsample_bylevel": 0.5,
         "colsample_bynode": 0.7, "seed": 2, "nthread": 1}
    a = fb.train(p, x, label=y, num_boost_round=4, quiet=True)
    b = fb.train({**p, "nthread": 3}, x, label=y, num_boost_round=4, quiet=True)
    assert model_json(a, tmp_path, "a") == model_json(b, tmp_path, "b")
    assert (a.predict(x).argmax(axis=1) == y).mean() > 0.8


@pytest.mark.parametrize("key", ["colsample_bylevel", "colsample_bynode"])
@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5])
def test_invalid_rates_are_rejected(key, bad):
    x, y = make(n=200)
    with pytest.raises(Exception, match=key):
        fb.train({**BASE, key: bad}, x, label=y, num_boost_round=1, quiet=True)


# ---------------------------------------------------------------- GPU
def _has_cuda():
    try:
        x, y = make(64, 4)
        fb.train({**BASE, "device": "cuda", "max_depth": 2}, x, y, num_boost_round=1, quiet=True)
        return True
    except Exception as e:  # noqa: BLE001
        if "CUDA" in str(e) or "cuda" in str(e):
            return False
        raise


cuda = pytest.mark.skipif(not _has_cuda(), reason="没有可用的 CUDA 构建或设备")


@cuda
@pytest.mark.parametrize("extra", [
    {"colsample_bylevel": 0.5},
    {"colsample_bynode": 0.5},
    {"colsample_bytree": 0.8, "colsample_bylevel": 0.7, "colsample_bynode": 0.6},
])
@pytest.mark.parametrize("objective", ["binary:logistic", "reg:squarederror"])
def test_gpu_exact_matches_cpu(tmp_path, extra, objective):
    x, y = make(n=4000, seed=11)
    w = np.random.default_rng(11).uniform(0.2, 3, len(y)).astype(np.float32)
    p = {"objective": objective, "max_depth": 6, "subsample": 0.8, "seed": 1, **extra}
    cpu = fb.train(p, x, label=y, weight=w, num_boost_round=6, quiet=True)
    gpu = fb.train({**p, "device": "cuda", "gpu_math": "exact"}, x, label=y, weight=w,
                   num_boost_round=6, quiet=True)
    assert model_json(cpu, tmp_path, "cpu") == model_json(gpu, tmp_path, "gpu")


@cuda
def test_gpu_multiclass_exact_matches_cpu(tmp_path):
    rng = np.random.default_rng(1)
    x = rng.normal(size=(2000, N_FEATURES)).astype(np.float32)
    y = ((x[:, 0] > 0).astype(int) + (x[:, 1] > 0.3).astype(int)).astype(np.float32)
    p = {"objective": "multi:softprob", "num_class": 3, "max_depth": 4, "colsample_bylevel": 0.6,
         "colsample_bynode": 0.7, "seed": 2}
    cpu = fb.train(p, x, label=y, num_boost_round=4, quiet=True)
    gpu = fb.train({**p, "device": "cuda"}, x, label=y, num_boost_round=4, quiet=True)
    assert model_json(cpu, tmp_path, "cpu") == model_json(gpu, tmp_path, "gpu")


@cuda
def test_gpu_fast_with_nested_sampling_is_deterministic(tmp_path):
    x, y = make(n=4000, seed=12)
    p = {**BASE, "nthread": 0, "max_depth": 6, "colsample_bylevel": 0.5, "colsample_bynode": 0.7,
         "device": "cuda", "gpu_math": "fast"}
    a = fb.train(p, x, label=y, num_boost_round=6, quiet=True)
    b = fb.train(p, x, label=y, num_boost_round=6, quiet=True)
    assert model_json(a, tmp_path, "a") == model_json(b, tmp_path, "b")
