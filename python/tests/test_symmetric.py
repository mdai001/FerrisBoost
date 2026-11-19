"""Public API and model-format coverage for symmetric growth."""
import json

import numpy as np
import pytest

import ferrisboost as fb


BASE = {
    "objective": "reg:squarederror",
    "grow_policy": "symmetric",
    "max_depth": 2,
    "max_bin": 16,
    "eta": 1.0,
    "lambda": 0.0,
    "base_score": 0.0,
    "nthread": 1,
}


def _train(x, y, params=None, **kwargs):
    return fb.train(
        {**BASE, **(params or {})},
        x,
        label=y,
        num_boost_round=1,
        quiet=True,
        **kwargs,
    )


def _save(model, path):
    model.save_model(str(path), quiet=True)
    return json.loads(path.read_text())["learner"]["gradient_booster"]["model"]["trees"][0]


def _assert_levels(tree, features):
    categories = {
        node: tuple(tree["categories"][start : start + size])
        for node, start, size in zip(
            tree["categories_nodes"], tree["categories_segments"], tree["categories_sizes"]
        )
    }
    left, right = tree["left_children"], tree["right_children"]
    frontier = [0]
    visited = set()
    for feature in features:
        splits = set()
        children = []
        for node in frontier:
            assert node not in visited
            visited.add(node)
            assert left[node] >= 0 and right[node] >= 0
            assert tree["split_indices"][node] == feature
            splits.add((
                tree["split_conditions"][node],
                tuple(sorted(categories.get(node, ()))),
                tree["default_left"][node],
            ))
            children.extend((left[node], right[node]))
        assert len(splits) == 1
        frontier = children
    for node in frontier:
        assert node not in visited
        visited.add(node)
        assert left[node] == right[node] == -1
    assert visited == set(range(len(left)))


def test_symmetric_shares_the_missing_direction(tmp_path):
    # Feature 1's missing rows favor opposite directions in the two leaves.
    # Sending them right has the larger sum of gains over both leaves.
    x = np.array([[0, 0], [0, 1], [0, np.nan], [1, 0], [1, 1], [1, np.nan]],
                 dtype=np.float32)
    y = np.array([-20, -12, -20, 10, 22, 22], dtype=np.float32)
    model = _train(x, y)
    tree = _save(model, tmp_path / "numeric.json")
    _assert_levels(tree, [0, 1])
    for node in (tree["left_children"][0], tree["right_children"][0]):
        assert not tree["default_left"][node]
    np.testing.assert_array_equal(model.predict(x), [-20, -16, -16, 10, 22, 22])


def test_symmetric_categorical_partition_survives_save_load(tmp_path):
    categories = np.array([0, 1, 2, 3, np.nan], dtype=np.float32)
    x = np.column_stack((
        np.repeat(np.array([0, 1], dtype=np.float32), categories.size),
        np.tile(categories, 2),
    ))
    # Two categories and the missing row share a target in each root branch.
    y = np.array([-20, -12, -20, -12, -20, 12, 20, 12, 20, 12], dtype=np.float32)
    model = _train(x, y, {"max_cat_to_onehot": 1}, feature_types=["q", "c"])
    path = tmp_path / "categorical.json"
    tree = _save(model, path)
    _assert_levels(tree, [0, 1])
    assert sorted(tree["categories_nodes"]) == sorted(
        [tree["left_children"][0], tree["right_children"][0]]
    )
    assert tree["categories_sizes"] == [2, 2]
    np.testing.assert_array_equal(model.predict(x), y)

    loaded = fb.Model.load_model(str(path), quiet=True)
    np.testing.assert_array_equal(loaded.predict(x), y)


@pytest.mark.parametrize("depth, cap, expected_depth", [
    (5, 0, 5),
    (5, 31, 4),
    (5, 1, 0),
    (0, 0, 0),
])
def test_symmetric_leaf_caps_keep_complete_levels(tmp_path, depth, cap, expected_depth):
    rows = np.arange(32)
    x = ((rows[:, None] >> np.arange(4, -1, -1)) & 1).astype(np.float32)
    model = _train(x, rows.astype(np.float32), {"max_depth": depth, "max_leaves": cap})
    _assert_levels(_save(model, tmp_path / "capped.json"), range(expected_depth))


def test_sklearn_forwards_symmetric_growth():
    pytest.importorskip("sklearn")
    from ferrisboost.sklearn import FerrisBoostRegressor

    estimator = FerrisBoostRegressor(grow_policy="symmetric", max_leaves=3)
    params = estimator.get_ferrisboost_params()
    assert params["grow_policy"] == "symmetric"
    assert params["max_leaves"] == 3


def test_symmetric_spilled_parquet_matches_dense(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    rows = np.arange(32)
    x = ((rows[:, None] >> np.arange(4, -1, -1)) & 1).astype(np.float32)
    y = rows.astype(np.float32)
    table = pa.table({**{f"x{i}": column for i, column in enumerate(x.T)}, "label": y})
    paths = [tmp_path / f"part{i}.parquet" for i in range(2)]
    for i, path in enumerate(paths):
        pq.write_table(table.slice(i * 16, 16), path, row_group_size=4)

    params = {"max_depth": 5, "max_leaves": 31, "cols_per_block": 1}
    dense = _train(x, y, params)
    spilled = fb.train(
        {**BASE, **params, "cache_budget_bytes": 0},
        [str(path) for path in paths],
        label="label",
        num_boost_round=1,
        quiet=True,
    )
    spilled_tree = _save(spilled, tmp_path / "spilled.json")
    _assert_levels(spilled_tree, range(4))
    assert spilled_tree == _save(dense, tmp_path / "dense.json")
    np.testing.assert_array_equal(spilled.predict(x), dense.predict(x))


def test_xgboost_loads_symmetric_json(tmp_path):
    xgb = pytest.importorskip("xgboost")
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    y = np.array([-8, -4, 4, 8], dtype=np.float32)
    model = _train(x, y)
    path = tmp_path / "symmetric.json"
    _assert_levels(_save(model, path), [0, 1])

    booster = xgb.Booster(model_file=str(path))
    probe = np.vstack((x, [[0, np.nan], [1, np.nan]])).astype(np.float32)
    np.testing.assert_array_equal(
        booster.predict(xgb.DMatrix(probe, nthread=1)), model.predict(probe)
    )
