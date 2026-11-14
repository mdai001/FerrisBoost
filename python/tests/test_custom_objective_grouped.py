"""Python adapters for grouped and multi-output custom objectives."""

import os

import numpy as np
import pytest

import ferrisboost as fb


def multiclass_data(n=36, k=3):
    x = np.arange(n * 2, dtype=np.float32).reshape(n, 2) / np.float32(11)
    y = (np.arange(n) % k).astype(np.float32)
    return x, y


def grouped_data():
    sizes = np.array([2, 3, 2], dtype=np.int64)
    n = int(sizes.sum())
    x = np.arange(n * 2, dtype=np.float32).reshape(n, 2) / np.float32(7)
    y = np.arange(n, dtype=np.float32) % 3
    qid = np.repeat(np.arange(len(sizes)), sizes)
    return x, y, qid, sizes


def test_custom_multiclass_callback_shape_raw_margin_and_continuation():
    x, y = multiclass_data()
    calls = []

    def custom_gradient(margins, labels):
        assert margins.dtype == labels.dtype == np.float32
        assert margins.shape == (len(labels), 3)
        assert margins.flags.c_contiguous
        calls.append((margins.copy(), labels.copy()))
        target = np.eye(3, dtype=np.float32)[labels.astype(np.int64)]
        return (
            np.ascontiguousarray(margins - target),
            np.ones_like(margins, dtype=np.float32),
        )

    params = {
        "objective": "multi:softprob",
        "num_class": 3,
        "max_depth": 1,
        "eta": 0.3,
        "max_bin": 16,
        "nthread": 1,
        "custom_objective": fb.CustomObjective(
            "multi_shape_objective", custom_gradient, kind="multiclass"
        ),
    }
    first = fb.train(params, x, label=y, num_boost_round=2, quiet=True)
    assert len(calls) == 2
    np.testing.assert_array_equal(calls[0][0], np.zeros((len(y), 3), np.float32))
    assert first.predict(x, output_margin=True).shape == (len(y), 3)
    prior_margin = first.predict(x, output_margin=True)

    continued = fb.train(
        params,
        x,
        label=y,
        num_boost_round=1,
        init_model=first,
        quiet=True,
    )
    assert len(calls) == 3
    np.testing.assert_allclose(calls[2][0], prior_margin, rtol=0, atol=2e-6)
    assert continued.num_trees == first.num_trees + 3
    assert continued.predict(x, output_margin=True).shape == (len(y), 3)


def test_custom_ranking_callback_groups_weights_raw_margin_and_continuation():
    x, y, qid, sizes = grouped_data()
    group_ptr = np.concatenate(([0], np.cumsum(sizes))).astype(np.uint32)
    calls = []

    def custom_gradient(margins, labels, groups):
        assert margins.dtype == labels.dtype == np.float32
        assert groups.dtype == np.uint32
        assert margins.shape == labels.shape == (len(y),)
        assert groups.flags.c_contiguous
        calls.append((margins.copy(), labels.copy(), groups.copy()))
        return np.ascontiguousarray(margins - labels), np.ones_like(labels)

    params = {
        "objective": "rank:pairwise",
        "max_depth": 1,
        "eta": 0.3,
        "max_bin": 16,
        "nthread": 1,
        "custom_objective": fb.CustomObjective(
            "rank_shape_objective", custom_gradient, kind="ranking"
        ),
    }
    weight = np.array([0.5, 1.5, 2.0], dtype=np.float32)
    first = fb.train(
        params, x, label=y, qid=qid, weight=weight, num_boost_round=2, quiet=True
    )
    assert len(calls) == 2
    np.testing.assert_array_equal(calls[0][0], np.zeros(len(y), dtype=np.float32))
    np.testing.assert_array_equal(calls[0][2], group_ptr)
    margin = first.predict(x, output_margin=True)
    continued = fb.train(
        params,
        x,
        label=y,
        qid=qid,
        weight=weight,
        num_boost_round=1,
        init_model=first,
        quiet=True,
    )
    assert len(calls) == 3
    np.testing.assert_allclose(calls[2][0], margin, rtol=0, atol=2e-6)
    np.testing.assert_array_equal(calls[2][2], group_ptr)
    assert continued.num_trees == first.num_trees + 1


@pytest.mark.skipif(os.environ.get("FB_TEST_CUDA") != "1", reason="requires explicit CUDA test opt-in")
@pytest.mark.parametrize("kind", ["multiclass", "ranking"])
def test_grouped_custom_objective_cuda_exact(kind):
    if kind == "multiclass":
        x, y = multiclass_data(48)
        k = 3
        seen = []

        def callback(margin, labels):
            seen.append(margin.copy())
            one_hot = np.eye(k, dtype=np.float32)[labels.astype(np.int64)]
            return np.ascontiguousarray(margin - one_hot), np.ones_like(margin)

        params = {
            "objective": "multi:softprob",
            "num_class": k,
            "max_depth": 1,
            "eta": 0.2,
            "nthread": 1,
        }
        descriptor = fb.CustomObjective("cuda_multi", callback, kind=kind)
        kwargs = {}
    else:
        x, y, qid, _sizes = grouped_data()
        seen = []

        def callback(margin, labels, groups):
            seen.append((margin.copy(), groups.copy()))
            return np.ascontiguousarray(margin - labels), np.ones_like(labels)

        params = {
            "objective": "rank:pairwise",
            "max_depth": 1,
            "eta": 0.2,
            "nthread": 1,
        }
        descriptor = fb.CustomObjective("cuda_rank", callback, kind=kind)
        kwargs = {"qid": qid}

    params.update(device="cuda:0", gpu_math="exact", custom_objective=descriptor)
    model = fb.train(params, x, label=y, num_boost_round=2, quiet=True, **kwargs)
    assert len(seen) == 2
    assert model.num_trees == (6 if kind == "multiclass" else 2)


@pytest.mark.skipif(os.environ.get("FB_TEST_CUDA") != "1", reason="requires explicit CUDA test opt-in")
@pytest.mark.parametrize("kind", ["multiclass", "ranking"])
def test_grouped_custom_objective_cuda_fast(kind):
    seen = []
    if kind == "multiclass":
        x, y = multiclass_data(12)
        params = {"objective": "multi:softprob", "num_class": 3}

        def callback(margin, labels):
            seen.append(margin.copy())
            one_hot = np.eye(3, dtype=np.float32)[labels.astype(np.int64)]
            return np.ascontiguousarray(margin - one_hot), np.ones_like(margin)

        descriptor = fb.CustomObjective(
            "fast_multi", callback, kind=kind
        )
        kwargs = {}
    else:
        x, y, qid, _sizes = grouped_data()
        params = {"objective": "rank:pairwise"}

        def callback(margin, labels, groups):
            seen.append((margin.copy(), groups.copy()))
            return np.ascontiguousarray(margin - labels), np.ones_like(labels)

        descriptor = fb.CustomObjective(
            "fast_rank", callback, kind=kind
        )
        kwargs = {"qid": qid}
    params.update(device="cuda:0", gpu_math="fast", custom_objective=descriptor)
    model = fb.train(params, x, label=y, num_boost_round=2, quiet=True, **kwargs)
    assert len(seen) == 2
    assert model.num_trees == (6 if kind == "multiclass" else 2)
