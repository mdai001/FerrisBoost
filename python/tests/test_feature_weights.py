"""Feature weight validation and weighted column sampling behavior."""

import csv
import json
import os

import numpy as np
import pytest

import ferrisboost as fb


BASE = {
    "objective": "binary:logistic",
    "max_depth": 4,
    "eta": 0.35,
    "max_bin": 16,
    "seed": 7,
    "nthread": 1,
}


def data(n=768, seed=7):
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 2, size=(n, 6)).astype(np.float32)
    # Feature 0 is dominant; feature 1 provides weaker positive-weight signal.
    noise = rng.normal(0, 0.7, n)
    y = (4.0 * (x[:, 0] - 0.5) + 0.7 * (x[:, 1] - 0.5) + noise > 0).astype(np.float32)
    return x, y


def model_doc(model, tmp_path, name="model"):
    path = tmp_path / f"{name}.json"
    model.save_model(str(path), quiet=True)
    return json.loads(path.read_text(encoding="utf-8"))


def model_bytes(model, tmp_path, name="model"):
    path = tmp_path / f"{name}.json"
    model.save_model(str(path), quiet=True)
    return path.read_bytes()


def trees(model, tmp_path, name="model"):
    return model_doc(model, tmp_path, name)["learner"]["gradient_booster"]["model"]["trees"]


def split_ids(tree):
    return {int(feature) for feature, left in zip(tree["split_indices"], tree["left_children"]) if left != -1}


def path_features(tree):
    left, right = tree["left_children"], tree["right_children"]
    split = tree["split_indices"]
    paths = []
    stack = [(0, frozenset())]
    while stack:
        node, used = stack.pop()
        if left[node] == -1:
            paths.append(used)
        else:
            current = used | {int(split[node])}
            stack.append((left[node], current))
            stack.append((right[node], current))
    return paths


@pytest.mark.parametrize(
    "weights",
    [
        pytest.param([-1.0, 1, 1, 1, 1, 1], id="negative"),
        pytest.param([np.nan, 1, 1, 1, 1, 1], id="nan"),
        pytest.param([np.inf, 1, 1, 1, 1, 1], id="infinity"),
        pytest.param([0, 0, 0, 0, 0, 0], id="all-zero"),
        pytest.param([1, 1, 1, 1, 1], id="wrong-length"),
        pytest.param([True, 1, 1, 1, 1, 1], id="bool"),
        pytest.param([np.bool_(True), 1, 1, 1, 1, 1], id="numpy-bool"),
        pytest.param(np.ones(6, dtype=np.bool_), id="bool-array"),
        pytest.param([], id="explicit-empty"),
    ],
)
def test_invalid_feature_weights_are_rejected(weights):
    x, y = data(32)
    with pytest.raises((TypeError, ValueError)):
        fb.train({**BASE, "feature_weights": weights}, x, label=y, num_boost_round=1, quiet=True)


def test_equal_positive_weights_preserve_unweighted_seeded_model(tmp_path):
    x, y = data()
    params = {**BASE, "colsample_bytree": 0.8, "colsample_bylevel": 0.75, "colsample_bynode": 0.7}
    plain = fb.train(params, x, label=y, num_boost_round=6, quiet=True)
    ones = fb.train({**params, "feature_weights": [1.0] * x.shape[1]}, x, label=y,
                    num_boost_round=6, quiet=True)
    assert trees(plain, tmp_path, "plain") == trees(ones, tmp_path, "ones")
    np.testing.assert_array_equal(plain.predict(x, output_margin=True), ones.predict(x, output_margin=True))


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_numpy_float_weight_arrays_match_equivalent_list(tmp_path, dtype):
    x, y = data()
    params = {**BASE, "colsample_bytree": 0.8, "colsample_bylevel": 0.75}
    weights = np.asarray([0.0, 0.5, 2.0, 1.0, 3.0, 0.25], dtype=dtype)
    from_array = fb.train({**params, "feature_weights": weights}, x, label=y,
                          num_boost_round=5, quiet=True)
    from_list = fb.train({**params, "feature_weights": weights.tolist()}, x, label=y,
                         num_boost_round=5, quiet=True)
    assert trees(from_array, tmp_path, "array") == trees(from_list, tmp_path, "list")
    np.testing.assert_array_equal(from_array.predict(x), from_list.predict(x))


def test_zero_weight_is_not_a_global_feature_ban_but_is_excluded_when_sampled(tmp_path):
    x, y = data()
    weights = [0.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    full = fb.train({**BASE, "max_depth": 1, "feature_weights": weights},
                    x, label=y, num_boost_round=1, quiet=True)
    assert split_ids(trees(full, tmp_path, "full")[0]) == {0}

    # At rate 0.5, floor(0.5 * 6) = 3 features are eligible for selection.
    # Feature 0 has weight zero and must never appear in a sampled tree.
    sampled = fb.train({**BASE, "max_depth": 1, "colsample_bytree": 0.5,
                        "feature_weights": weights}, x, label=y, num_boost_round=12, quiet=True)
    sampled_trees = trees(sampled, tmp_path, "sampled")
    assert len(sampled_trees) == 12
    assert all(0 not in split_ids(tree) for tree in sampled_trees)
    # Only three candidates can be selected per tree; the tree records a subset
    # of candidates that actually won a split.
    assert all(len(split_ids(tree)) <= 3 for tree in sampled_trees)


def test_weighted_sampling_respects_explicit_singleton_interaction_groups(tmp_path):
    x, y = data()
    params = {
        **BASE,
        "max_depth": 3,
        "feature_weights": [0.0, 0.5, 3.0, 1.5, 0.75, 2.0],
        "interaction_constraints": [[0], [1], [2], [3], [4], [5]],
        "colsample_bytree": 0.8,
        "colsample_bynode": 0.7,
    }
    model = fb.train(params, x, label=y, num_boost_round=8, quiet=True)
    model_trees = trees(model, tmp_path)
    paths = [path for tree in model_trees for path in path_features(tree)]
    assert any(path for path in paths), "fixture should produce at least one split"
    assert all(len(path) <= 1 for path in paths)


def test_asymmetric_weights_sampling_is_repeatable_across_threads_and_blocks(tmp_path):
    x, y = data(seed=11)
    params = {
        **BASE,
        "seed": 29,
        "feature_weights": [0.0, 0.25, 4.0, 1.5, 0.75, 2.0],
        "colsample_bytree": 0.8,
        "colsample_bylevel": 0.7,
        "colsample_bynode": 0.6,
    }
    first = fb.train(params, x, label=y, num_boost_round=7, quiet=True)
    repeat = fb.train(params, x, label=y, num_boost_round=7, quiet=True)
    threaded = fb.train({**params, "nthread": 4}, x, label=y, num_boost_round=7, quiet=True)
    blocked = fb.train({**params, "nthread": 3, "cols_per_block": 2}, x, label=y,
                       num_boost_round=7, quiet=True)
    expected = model_doc(first, tmp_path, "first")
    assert expected == model_doc(repeat, tmp_path, "repeat")
    assert expected == model_doc(threaded, tmp_path, "threaded")
    assert expected == model_doc(blocked, tmp_path, "blocked")


def test_feature_weights_do_not_change_leaf_math_when_sampling_is_disabled(tmp_path):
    x, y = data()
    plain = fb.train({**BASE, "colsample_bytree": 1.0, "colsample_bylevel": 1.0,
                      "colsample_bynode": 1.0}, x, label=y, num_boost_round=4, quiet=True)
    weighted = fb.train({**BASE, "feature_weights": [0, 1, 2, 3, 4, 5],
                         "colsample_bytree": 1.0, "colsample_bylevel": 1.0,
                         "colsample_bynode": 1.0}, x, label=y, num_boost_round=4, quiet=True)
    np.testing.assert_array_equal(plain.predict(x, output_margin=True), weighted.predict(x, output_margin=True))
    assert trees(plain, tmp_path, "plain") == trees(weighted, tmp_path, "weighted")


def test_named_feature_weights_match_positional_weights_and_validate_schema(tmp_path):
    pd = pytest.importorskip("pandas")
    x, y = data(384)
    names = ["dominant", "weak", "noise_a", "noise_b", "noise_c", "noise_d"]
    frame = pd.DataFrame(x, columns=names)
    base = {**BASE, "colsample_bytree": 0.5}
    by_name = fb.train({**base, "feature_weights": {"dominant": 0.0}}, frame,
                       label=y, num_boost_round=5, quiet=True)
    by_position = fb.train({**base, "feature_weights": [0, 1, 1, 1, 1, 1]}, x,
                           label=y, num_boost_round=5, quiet=True)
    np.testing.assert_array_equal(by_name.predict(frame), by_position.predict(x))
    assert [split_ids(tree) for tree in trees(by_name, tmp_path, "named")] == [
        split_ids(tree) for tree in trees(by_position, tmp_path, "positional")
    ]

    with pytest.raises(ValueError):
        fb.train({**base, "feature_weights": {"missing": 1.0}}, frame,
                 label=y, num_boost_round=1, quiet=True)
    with pytest.raises(ValueError):
        fb.train({**base, "feature_weights": {"dominant": 0.0}}, x,
                 label=y, num_boost_round=1, quiet=True)
    duplicate = pd.DataFrame(x[:, :2], columns=["duplicate", "duplicate"])
    with pytest.raises(ValueError):
        fb.train({**BASE, "feature_weights": {"duplicate": 1.0}}, duplicate,
                 label=y, num_boost_round=1, quiet=True)


def test_csv_named_feature_weights_match_positional_reference(tmp_path):
    x, y = data(384)
    names = [f"f{i}" for i in range(x.shape[1])]
    path = tmp_path / "named_weights.csv"
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow([*names, "target"])
        writer.writerows(np.column_stack((x, y)).tolist())

    params = {**BASE, "colsample_bytree": 0.5}
    by_name = fb.train({**params, "feature_weights": {"f0": 0.0}}, str(path),
                       label="target", num_boost_round=5, quiet=True)
    by_position = fb.train({**params, "feature_weights": [0, 1, 1, 1, 1, 1]}, x,
                           label=y, num_boost_round=5, quiet=True)
    assert [split_ids(tree) for tree in trees(by_name, tmp_path, "csv-named")] == [
        split_ids(tree) for tree in trees(by_position, tmp_path, "csv-positional")
    ]
    np.testing.assert_array_equal(by_name.predict(x), by_position.predict(x))


def test_sklearn_clone_and_fit_forward_feature_weights():
    pytest.importorskip("sklearn")
    from sklearn.base import clone

    from ferrisboost.sklearn import FerrisBoostClassifier

    x, y = data(256)
    weights = [0.0, 2.0, 1.0, 1.0, 1.0, 1.0]
    estimator = FerrisBoostClassifier(n_estimators=3, max_depth=2, n_jobs=1,
                                      feature_weights=weights)
    cloned = clone(estimator)
    assert cloned.get_params()["feature_weights"] == weights
    assert cloned.get_ferrisboost_params()["feature_weights"] == weights
    cloned.fit(x, y)
    assert cloned.get_booster().num_trees == 3


def _weighted_multiclass_data():
    rng = np.random.default_rng(41)
    x = rng.integers(0, 2, size=(384, 6)).astype(np.float32)
    y = ((x[:, 0].astype(np.int32) + 2 * x[:, 1].astype(np.int32)) % 3).astype(np.float32)
    return x, y


def _weighted_multiclass_params():
    return {
        "objective": "multi:softprob",
        "num_class": 3,
        "max_depth": 3,
        "eta": 0.3,
        "max_bin": 16,
        "seed": 43,
        "nthread": 1,
        "feature_weights": [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        "colsample_bytree": 0.5,
        "colsample_bylevel": 0.5,
        "colsample_bynode": 0.5,
    }


def test_weighted_multiclass_sampling_uses_only_positive_weight_feature(tmp_path):
    x, y = _weighted_multiclass_data()
    model = fb.train(_weighted_multiclass_params(), x, label=y, num_boost_round=3, quiet=True)
    model_trees = trees(model, tmp_path)
    assert len(model_trees) == 9
    actual_splits = [split_ids(tree) for tree in model_trees]
    assert any(features for features in actual_splits)
    assert all(features <= {1} for features in actual_splits)


@pytest.mark.skipif(os.environ.get("FB_TEST_CUDA") != "1", reason="requires explicit CUDA test opt-in")
def test_cuda_exact_weighted_multiclass_matches_cpu_json(tmp_path):
    x, y = _weighted_multiclass_data()
    params = _weighted_multiclass_params()
    cpu = fb.train(params, x, label=y, num_boost_round=3, quiet=True)
    gpu = fb.train({**params, "device": "cuda:0", "gpu_math": "exact"},
                   x, label=y, num_boost_round=3, quiet=True)
    assert model_bytes(cpu, tmp_path, "multi-cpu") == model_bytes(gpu, tmp_path, "multi-gpu")


def test_weighted_seeded_continuation_matches_uninterrupted_training(tmp_path):
    x, y = data(512, seed=47)
    params = {
        **BASE,
        "seed": 53,
        "subsample": 0.8,
        "feature_weights": [0.2, 3.0, 0.5, 1.25, 2.0, 0.75],
        "colsample_bytree": 0.8,
        "colsample_bylevel": 0.75,
        "colsample_bynode": 0.6,
    }
    uninterrupted = fb.train(params, x, label=y, num_boost_round=4, quiet=True)
    first = fb.train(params, x, label=y, num_boost_round=2, quiet=True)
    continued = fb.train(params, x, label=y, num_boost_round=2, init_model=first, quiet=True)

    assert model_bytes(uninterrupted, tmp_path, "uninterrupted") == model_bytes(continued, tmp_path, "continued")
    assert uninterrupted.predict(x, output_margin=True).tobytes() == continued.predict(
        x, output_margin=True
    ).tobytes()


def test_feature_weights_work_with_interaction_constraints_and_max_leaves(tmp_path):
    rng = np.random.default_rng(18)
    x = rng.integers(0, 2, size=(1200, 4)).astype(np.float32)
    y = (((x[:, 0] > 0) & (x[:, 1] > 0)) |
         ((x[:, 2] > 0) & (x[:, 3] > 0))).astype(np.float32)
    params = {
        **BASE,
        "max_depth": 4,
        "max_leaves": 3,
        "interaction_constraints": [[0, 1], [2, 3]],
        "feature_weights": [2.0, 1.0, 1.5, 0.0],
        "colsample_bytree": 0.75,
    }
    model = fb.train(params, x, label=y, num_boost_round=8, quiet=True)
    allowed = ({0, 1}, {2, 3})
    all_paths = []
    for tree in trees(model, tmp_path):
        assert sum(child == -1 for child in tree["left_children"]) <= 3
        paths = path_features(tree)
        all_paths.extend(paths)
        assert all(any(path <= group for group in allowed) for path in paths)
    assert any(len(path) >= 2 for path in all_paths)


@pytest.mark.skipif(os.environ.get("FB_TEST_CUDA") != "1", reason="requires explicit CUDA test opt-in")
def test_cuda_exact_weighted_sampling_matches_cpu_for_resident_modes(tmp_path):
    x, y = data(320, seed=22)
    params = {
        **BASE,
        "seed": 31,
        "feature_weights": [0.0, 0.2, 3.0, 1.5, 0.75, 2.0],
        "colsample_bytree": 0.8,
        "colsample_bylevel": 0.7,
        "colsample_bynode": 0.6,
    }
    cpu = fb.train(params, x, label=y, num_boost_round=4, quiet=True)
    expected = model_doc(cpu, tmp_path, "cpu")
    for resident_blocks in (0, 1):
        gpu = fb.train({**params, "device": "cuda:0", "gpu_math": "exact",
                        "resident_blocks": resident_blocks}, x, label=y,
                       num_boost_round=4, quiet=True)
        assert expected == model_doc(gpu, tmp_path, f"gpu-{resident_blocks}")


@pytest.mark.skipif(os.environ.get("FB_TEST_CUDA") != "1", reason="requires explicit CUDA test opt-in")
def test_cuda_fast_weighted_sampling_is_reproducible(tmp_path):
    x, y = data(320, seed=23)
    params = {
        **BASE,
        "device": "cuda:0",
        "gpu_math": "fast",
        "seed": 37,
        "feature_weights": [0.0, 0.1, 5.0, 0.5, 2.0, 1.0],
        "colsample_bytree": 0.75,
        "colsample_bylevel": 0.8,
    }
    first = fb.train(params, x, label=y, num_boost_round=4, quiet=True)
    repeat = fb.train(params, x, label=y, num_boost_round=4, quiet=True)
    assert model_doc(first, tmp_path, "fast-a") == model_doc(repeat, tmp_path, "fast-b")
