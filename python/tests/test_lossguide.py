"""CPU loss-guided growth policy contract tests."""
import json

import numpy as np
import pytest

import ferrisboost as fb


BASE = {
    "objective": "reg:squarederror",
    "max_depth": 8,
    "max_bin": 16,
    "eta": 0.3,
    "lambda": 0.0,
    "min_child_weight": 0.0,
    "base_score": 0.0,
    "nthread": 1,
}


def _data(labels):
    y = np.asarray(labels, dtype=np.float32)
    x = np.arange(y.size, dtype=np.float32).reshape(-1, 1)
    return x, y


def _tree(model, tmp_path):
    path = tmp_path / "model.json"
    model.save_model(str(path), quiet=True)
    return json.loads(path.read_text())["learner"]["gradient_booster"]["model"]["trees"][0]


def _leaf_count(tree):
    return sum(child == -1 for child in tree["left_children"])


def test_train_accepts_both_growth_policy_names():
    x, y = _data([0, 0, 1, 1, 10, 10, 14, 20])
    for policy in ("depthwise", "lossguide"):
        model = fb.train({**BASE, "grow_policy": policy}, x, label=y,
                         num_boost_round=1, quiet=True)
        assert model.num_trees == 1


def test_unknown_growth_policy_is_rejected():
    x, y = _data([0, 0, 1, 1])
    with pytest.raises(ValueError, match="grow_policy"):
        fb.train({**BASE, "grow_policy": "breadthwise"}, x, label=y,
                 num_boost_round=1, quiet=True)


def test_sklearn_constructor_and_parameter_translation_include_growth_policy():
    pytest.importorskip("sklearn")
    from ferrisboost.sklearn import FerrisBoostRegressor

    estimator = FerrisBoostRegressor(grow_policy="lossguide", max_leaves=3)
    assert estimator.get_params()["grow_policy"] == "lossguide"
    params = estimator.get_ferrisboost_params()
    assert params["grow_policy"] == "lossguide"
    assert params["max_leaves"] == 3


def test_lossguide_uses_highest_gain_under_leaf_cap(tmp_path):
    x, y = _data([0, 0, 1, 1, 10, 10, 14, 20])
    model = fb.train({**BASE, "grow_policy": "lossguide", "max_leaves": 3},
                     x, label=y, num_boost_round=1, quiet=True)
    tree = _tree(model, tmp_path)
    # Both root children are splittable; the right child's gain is larger.
    assert _leaf_count(tree) == 3
    assert tree["left_children"][1] == -1, "the lower-gain left candidate should stay a leaf"
    assert tree["left_children"][2] != -1, "the higher-gain right candidate should be chosen"


def test_lossguide_ties_choose_lower_node_id(tmp_path):
    x, y = _data([0, 0, 1, 1, 3, 3, 4, 4])
    model = fb.train({**BASE, "grow_policy": "lossguide", "max_leaves": 3},
                     x, label=y, num_boost_round=1, quiet=True)
    tree = _tree(model, tmp_path)
    assert _leaf_count(tree) == 3
    assert tree["left_children"][1] != -1
    assert tree["left_children"][2] == -1


@pytest.mark.parametrize("depth, leaves", [(0, 0), (8, 1)])
def test_root_only_limits_apply_to_lossguide(tmp_path, depth, leaves):
    x, y = _data([0, 0, 1, 1, 3, 3, 4, 4])
    model = fb.train({**BASE, "grow_policy": "lossguide", "max_depth": depth,
                      "max_leaves": leaves}, x, label=y, num_boost_round=1, quiet=True)
    assert _leaf_count(_tree(model, tmp_path)) == 1


def test_zero_max_leaves_is_unlimited_with_finite_depth(tmp_path):
    y = np.repeat(np.square(np.arange(8, dtype=np.float32)), 4)
    x = np.arange(y.size, dtype=np.float32).reshape(-1, 1)
    unlimited = fb.train({**BASE, "grow_policy": "lossguide", "max_leaves": 0},
                         x, label=y, num_boost_round=1, quiet=True)
    capped = fb.train({**BASE, "grow_policy": "lossguide", "max_leaves": 3},
                      x, label=y, num_boost_round=1, quiet=True)
    assert _leaf_count(_tree(unlimited, tmp_path)) > 3
    assert _leaf_count(_tree(capped, tmp_path)) == 3


def test_lossguide_save_load_and_continue(tmp_path):
    x, y = _data([0, 0, 1, 1, 10, 10, 14, 20])
    params = {**BASE, "grow_policy": "lossguide", "max_leaves": 3}
    first = fb.train(params, x, label=y, num_boost_round=2, quiet=True)
    path = tmp_path / "lossguide.json"
    first.save_model(str(path), quiet=True)
    loaded = fb.Model.load_model(str(path), quiet=True)
    np.testing.assert_array_equal(first.predict(x), loaded.predict(x))

    continued = fb.train(params, x, label=y, num_boost_round=1,
                         init_model=loaded, quiet=True)
    assert continued.num_trees == 3
    np.testing.assert_array_equal(
        continued.predict(x),
        fb.train(params, x, label=y, num_boost_round=3, quiet=True).predict(x),
    )


def _has_cuda():
    x, y = _data([0, 0, 1, 1])
    try:
        fb.train({**BASE, "device": "cuda", "max_depth": 2}, x, label=y,
                 num_boost_round=1, quiet=True)
        return True
    except Exception as e:  # noqa: BLE001
        if "CUDA" in str(e) or "cuda" in str(e):
            return False
        raise


cuda = pytest.mark.skipif(not _has_cuda(), reason="no usable CUDA build or device")


@cuda
@pytest.mark.parametrize("gpu_math", ["exact", "fast"])
def test_lossguide_cuda_uses_best_first_topology(gpu_math, tmp_path):
    x, y = _data([0, 0, 1, 1, 10, 10, 14, 20])
    params = {**BASE, "grow_policy": "lossguide", "max_leaves": 4}
    cpu = fb.train(params, x, label=y, num_boost_round=1, quiet=True)
    gpu = fb.train({**params, "device": "cuda", "gpu_math": gpu_math}, x,
                   label=y, num_boost_round=1, quiet=True)
    cpu_tree = _tree(cpu, tmp_path)
    gpu_tree = _tree(gpu, tmp_path)
    assert _leaf_count(gpu_tree) == 4
    children = gpu_tree["left_children"]
    assert children[0] != -1
    assert children[1] == -1, "the lower-gain shallow candidate should stay a leaf"
    assert children[2] != -1, "the higher-gain candidate should be selected"
    assert children[3] != -1, "best-first growth should continue at a deeper level"

    if gpu_math == "exact":
        assert gpu_tree["left_children"] == cpu_tree["left_children"]
        assert gpu_tree["right_children"] == cpu_tree["right_children"]
        np.testing.assert_array_equal(gpu.predict(x), cpu.predict(x))
    else:
        exact_gpu = fb.train({**params, "device": "cuda", "gpu_math": "exact"}, x,
                             label=y, num_boost_round=1, quiet=True)
        np.testing.assert_allclose(gpu.predict(x), exact_gpu.predict(x), rtol=1e-4, atol=1e-4)

    root_only = fb.train({**params, "device": "cuda", "gpu_math": gpu_math,
                          "max_leaves": 1}, x, label=y, num_boost_round=1, quiet=True)
    assert _leaf_count(_tree(root_only, tmp_path)) == 1


@cuda
def test_lossguide_cuda_exact_matches_resident_and_streamed_parquet(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    x, y = _data([0, 0, 1, 1, 10, 10, 14, 20])
    table = pa.table({"x": x[:, 0], "y": y})
    paths = [tmp_path / "part1.parquet", tmp_path / "part2.parquet"]
    for i, path in enumerate(paths):
        pq.write_table(table.slice(4 * i, 4), path)
    params = {**BASE, "grow_policy": "lossguide", "max_leaves": 4,
              "device": "cuda", "gpu_math": "exact"}
    resident = fb.train(params, x, label=y, num_boost_round=2, quiet=True)
    streamed = fb.train({**params, "cache_budget_bytes": 0, "resident_blocks": 0},
                        [str(path) for path in paths], label="y",
                        num_boost_round=2, quiet=True)
    np.testing.assert_array_equal(resident.predict(x), streamed.predict(x))


def test_lossguide_matches_resident_and_spilled_parquet(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    x, y = _data([0, 0, 1, 1, 10, 10, 14, 20])
    table = pa.table({"x": x[:, 0], "y": y})
    paths = [tmp_path / "part1.parquet", tmp_path / "part2.parquet"]
    for i, path in enumerate(paths):
        pq.write_table(table.slice(4 * i, 4), path)
    params = {**BASE, "grow_policy": "lossguide", "max_leaves": 4}
    resident = fb.train(params, x, label=y, num_boost_round=4, quiet=True)
    spilled = fb.train(
        {**params, "cache_budget_bytes": 0}, [str(path) for path in paths],
        label="y", num_boost_round=4, quiet=True,
    )
    np.testing.assert_array_equal(resident.predict(x), spilled.predict(x))


@cuda
def test_lossguide_cuda_exact_matches_cpu_with_categories_and_sampling(tmp_path):
    rng = np.random.default_rng(281)
    x = rng.integers(0, 9, (2048, 4)).astype(np.float32)
    y = (3.0 * (x[:, 0] == 4) + x[:, 1] * (x[:, 2] > 2) - x[:, 3]).astype(np.float32)
    x[::13, 0] = np.nan
    x[::17, 1] = np.nan
    params = {
        **BASE, "grow_policy": "lossguide", "max_leaves": 8, "cols_per_block": 1,
        "subsample": 0.8, "colsample_bylevel": 0.75, "colsample_bytree": 0.75,
        "interaction_constraints": [[0, 1, 2], [3]],
    }
    options = dict(label=y, feature_types=["c", "q", "q", "q"], num_boost_round=4, quiet=True)
    cpu = fb.train(params, x, **options)
    gpu = fb.train({**params, "device": "cuda", "gpu_math": "exact"}, x, **options)
    assert _tree(cpu, tmp_path) == _tree(gpu, tmp_path)
    np.testing.assert_array_equal(cpu.predict(x), gpu.predict(x))


@cuda
def test_lossguide_cuda_split_cache_eviction_matches_uncached(tmp_path, monkeypatch):
    rng = np.random.default_rng(719)
    n = 768
    # Put narrow and wide quantized features in separate physical blocks, and
    # give best-first growth enough useful columns to keep several candidates
    # live while the two-node histogram batch advances the frontier.
    x = np.column_stack([
        rng.integers(0, 8, n),
        rng.integers(0, 400, n),
        rng.integers(0, 2, n),
        rng.integers(0, 5, n),
        rng.integers(0, 11, n),
        rng.integers(0, 19, n),
        rng.integers(0, 31, n),
        rng.integers(0, 47, n),
    ]).astype(np.float32)
    y = (2.5 * (x[:, 1] % 13 == 3) + 1.7 * (x[:, 4] > 6)
         - 1.3 * (x[:, 6] > 22) + 0.4 * x[:, 7] + rng.normal(0, 0.2, n))
    y = y.astype(np.float32)
    params = {
        **BASE, "grow_policy": "lossguide", "max_leaves": 12,
        "device": "cuda", "gpu_math": "exact", "max_bin": 64,
        "cols_per_block": 1, "resident_blocks": 2, "hist_nodes_per_batch": 2,
    }
    options = dict(
        label=y, feature_types=["c", "c", "q", "q", "q", "q", "q", "q"],
        num_boost_round=1, quiet=True,
    )

    monkeypatch.setenv("FB_GPU_SPLIT_CACHE", "1")
    cached = fb.train(params, x, **options)
    monkeypatch.setenv("FB_GPU_SPLIT_CACHE", "0")
    uncached = fb.train(params, x, **options)

    cached_tree = _tree(cached, tmp_path)
    uncached_tree = _tree(uncached, tmp_path)
    assert _leaf_count(cached_tree) == 12
    assert cached_tree == uncached_tree  # includes split gains and thresholds
    np.testing.assert_array_equal(cached.predict(x), uncached.predict(x))
