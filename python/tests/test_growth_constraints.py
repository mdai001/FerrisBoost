"""Python API and model-path checks for per-tree growth constraints."""
import json

import numpy as np
import pytest

import ferrisboost as fb


BASE = {"objective": "binary:logistic", "max_depth": 4, "eta": 0.5, "nthread": 1}


def _toy(n=512, seed=17):
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 2, size=(n, 4)).astype(np.float32)
    y = (((x[:, 0] > 0) & (x[:, 1] > 0)) |
         ((x[:, 2] > 0) & (x[:, 3] > 0))).astype(np.float32)
    return x, y


def _trees(model, tmp_path):
    path = tmp_path / "model.json"
    model.save_model(str(path), quiet=True)
    return json.loads(path.read_text())["learner"]["gradient_booster"]["model"]["trees"]


def _path_features(tree):
    left, right = tree["left_children"], tree["right_children"]
    splits = tree["split_indices"]
    stack = [(0, frozenset())]
    paths = []
    while stack:
        node, features = stack.pop()
        if left[node] == -1:
            paths.append(features)
            continue
        path = features | {splits[node]}
        stack.append((left[node], path))
        stack.append((right[node], path))
    return paths


def test_negative_max_leaves_is_rejected():
    x, y = _toy()
    with pytest.raises(ValueError, match="max_leaves must be >= 0"):
        fb.train({**BASE, "max_leaves": -1}, x, label=y, num_boost_round=1, quiet=True)


def test_max_leaves_one_makes_root_only_trees(tmp_path):
    x, y = _toy()
    model = fb.train({**BASE, "max_leaves": 1}, x, label=y,
                     num_boost_round=3, quiet=True)
    for tree in _trees(model, tmp_path):
        assert sum(child == -1 for child in tree["left_children"]) == 1


def test_sklearn_get_set_params_forwards_growth_constraints():
    pytest.importorskip("sklearn")
    from ferrisboost.sklearn import FerrisBoostClassifier

    estimator = FerrisBoostClassifier(max_leaves=3, interaction_constraints=[[0, 1]])
    params = estimator.get_params()
    assert params["max_leaves"] == 3
    assert params["interaction_constraints"] == [[0, 1]]

    estimator.set_params(max_leaves=5, interaction_constraints=[["a", 1]])
    forwarded = estimator.get_ferrisboost_params()
    assert forwarded["max_leaves"] == 5
    assert forwarded["interaction_constraints"] == [["a", 1]]


def test_interaction_groups_restrict_every_root_to_leaf_path(tmp_path):
    x, y = _toy()
    model = fb.train(
        {**BASE, "interaction_constraints": [[0, 1], [2, 3]]},
        x, label=y, num_boost_round=8, quiet=True,
    )
    paths = [path for tree in _trees(model, tmp_path) for path in _path_features(tree)]
    allowed = ({0, 1}, {2, 3})
    assert any(len(path) >= 2 for path in paths), "fixture should exercise multi-feature paths"
    assert all(any(path <= group for group in allowed) for path in paths)


def test_interaction_names_resolve_with_named_inputs():
    pd = pytest.importorskip("pandas")
    x, y = _toy()
    frame = pd.DataFrame(x, columns=["a", "b", "c", "d"])
    model = fb.train({**BASE, "interaction_constraints": [["a", 1], ["c", "d"]]},
                     frame, label=y, num_boost_round=2, quiet=True)
    assert model.num_trees == 2


def test_interaction_unknown_names_are_rejected():
    pd = pytest.importorskip("pandas")
    x, y = _toy()
    frame = pd.DataFrame(x, columns=["a", "b", "c", "d"])
    with pytest.raises(ValueError, match="not a feature column"):
        fb.train({**BASE, "interaction_constraints": [["missing"]]},
                 frame, label=y, num_boost_round=1, quiet=True)


def test_interaction_names_require_a_named_input():
    x, y = _toy()
    with pytest.raises(ValueError, match="given by name need column names"):
        fb.train({**BASE, "interaction_constraints": [["a", 1]]},
                 x, label=y, num_boost_round=1, quiet=True)


@pytest.mark.parametrize("constraints, match", [
    ([[True, 1]], "got bool"),
    ([[-1]], "non-negative"),
    ([[]], "must not be empty"),
])
def test_invalid_interaction_groups_are_rejected(constraints, match):
    x, y = _toy()
    with pytest.raises(ValueError, match=match):
        fb.train({**BASE, "interaction_constraints": constraints},
                 x, label=y, num_boost_round=1, quiet=True)


def test_interaction_feature_ids_must_be_in_range():
    x, y = _toy()
    with pytest.raises(ValueError, match="out of range"):
        fb.train({**BASE, "interaction_constraints": [[4]]},
                 x, label=y, num_boost_round=1, quiet=True)


def test_ambiguous_interaction_names_are_rejected(tmp_path):
    path = tmp_path / "duplicate_names.csv"
    path.write_text("dup,dup,target\n0,1,0\n1,0,1\n0,1,0\n1,0,1\n")
    with pytest.raises(ValueError, match="ambiguous"):
        fb.train({**BASE, "interaction_constraints": [["dup"]]},
                 str(path), label="target", num_boost_round=1, quiet=True)
