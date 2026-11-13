"""Python callback metrics: raw margins, evaluation context, and error behavior."""

import csv
import os

import numpy as np
import pytest

import ferrisboost as fb


def data(n=8):
    x = np.arange(n, dtype=np.float32).reshape(-1, 1)
    y = (np.arange(n) >= n // 2).astype(np.float32)
    return x, y


def params(objective="binary:logistic", **extra):
    return {
        "objective": objective,
        "max_depth": 1,
        "max_bin": 8,
        "nthread": 1,
        **extra,
    }


def test_custom_metric_receives_raw_margins_and_matches_hand_computed_rmse():
    x, y = data()
    received = []

    def raw_rmse(pred, labels, sample_weight, group_ptr):
        received.append((pred.copy(), labels.copy(), sample_weight, group_ptr))
        assert sample_weight is None
        assert group_ptr is None
        return float(np.sqrt(np.mean((pred - labels) ** 2)))

    metric = fb.CustomMetric("raw_rmse", raw_rmse)
    result = {}
    model = fb.train(
        params(eval_metric=metric),
        x,
        label=y,
        num_boost_round=1,
        evals=[(x, y, "valid")],
        evals_result=result,
        quiet=True,
    )

    raw = model.predict(x, output_margin=True)
    expected = float(np.sqrt(np.mean((raw - y) ** 2)))
    assert result["valid"]["raw_rmse"] == pytest.approx([expected], rel=0, abs=1e-7)
    assert len(received) == 1
    seen_pred, seen_labels, _, _ = received[0]
    assert seen_pred.dtype == np.float32
    assert seen_pred.shape == seen_labels.shape == y.shape
    np.testing.assert_array_equal(seen_pred, raw)
    np.testing.assert_array_equal(seen_labels, y)


def test_custom_metric_receives_weights_once_without_hidden_reweighting():
    x, y = data()
    weight = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.float32)
    received = []

    def weighted_sum(pred, labels, sample_weight, group_ptr):
        assert group_ptr is None
        assert sample_weight is not None
        received.append(sample_weight.copy())
        # Return a weighted sum rather than an average, making any implicit
        # outer reweighting visible in evals_result.
        return float(np.sum(sample_weight * np.abs(pred - labels)))

    result = {}
    model = fb.train(
        params(eval_metric=fb.CustomMetric("weighted_sum", weighted_sum)),
        x,
        label=y,
        num_boost_round=1,
        evals=[(x, y, weight, "valid")],
        evals_result=result,
        quiet=True,
    )

    assert len(received) == 1
    np.testing.assert_array_equal(received[0], weight)
    # The callback's scalar is recorded directly; it is not multiplied by the
    # row weights a second time by the evaluation layer.
    model_raw = model.predict(x, output_margin=True)
    expected = float(np.sum(weight * np.abs(model_raw - y)))
    assert result["valid"]["weighted_sum"] == pytest.approx([expected], rel=0, abs=1e-7)


def test_greater_is_better_custom_metric_controls_early_stopping():
    x, y = data()
    values = iter([1.0, 2.0, 3.0, 2.0, 1.0])
    metric_calls = []

    def increasing_score(pred, labels, sample_weight, group_ptr):
        assert sample_weight is None and group_ptr is None
        value = next(values)
        metric_calls.append(value)
        return value

    callback_rounds = []

    def count_rounds(round_index, _metrics):
        callback_rounds.append(round_index)
        return True

    model = fb.train(
        params(eval_metric=fb.CustomMetric("score", increasing_score, greater_is_better=True)),
        x,
        label=y,
        num_boost_round=10,
        evals=[(x, y, "valid")],
        early_stopping_rounds=2,
        callbacks=[count_rounds],
        quiet=True,
    )

    assert metric_calls == [1.0, 2.0, 3.0, 2.0, 1.0]
    assert callback_rounds == [0, 1, 2, 3, 4]
    assert model.best_iteration == 2
    assert model.best_score == 3.0
    assert model.num_trees == 5


def test_custom_metric_mixed_order_and_multiple_eval_order_are_preserved():
    x, y = data()
    left_labels = y.copy()
    right_labels = 1.0 - y
    calls = []

    def record_eval(pred, labels, sample_weight, group_ptr):
        assert sample_weight is None and group_ptr is None
        calls.append(labels.copy())
        return float(np.mean(pred))

    metric = fb.CustomMetric("custom_mean", record_eval)
    result = {}
    fb.train(
        params(eval_metric=[metric, "rmse"]),
        x,
        label=y,
        num_boost_round=2,
        evals=[(x, left_labels, "left"), (x, right_labels, "right")],
        evals_result=result,
        quiet=True,
    )

    assert list(result) == ["left", "right"]
    assert list(result["left"]) == ["custom_mean", "rmse"]
    assert list(result["right"]) == ["custom_mean", "rmse"]
    assert len(calls) == 4
    for observed, expected in zip(calls, [left_labels, right_labels] * 2):
        np.testing.assert_array_equal(observed, expected)


def test_metric_exception_keeps_python_type_and_skips_history_and_callbacks():
    x, y = data()
    result = {}
    callback_rounds = []

    class MetricMarkerError(LookupError):
        pass

    def fail(_pred, _labels, _sample_weight, _group_ptr):
        raise MetricMarkerError("metric callback marker")

    def after_iteration(round_index, _metrics):
        callback_rounds.append(round_index)
        return True

    with pytest.raises(MetricMarkerError, match="metric callback marker"):
        fb.train(
            params(eval_metric=fb.CustomMetric("raises", fail)),
            x,
            label=y,
            num_boost_round=2,
            evals=[(x, y, "valid")],
            evals_result=result,
            callbacks=[after_iteration],
            quiet=True,
        )

    assert result == {}
    assert callback_rounds == []


@pytest.mark.parametrize(
    "bad_value",
    [float("nan"), float("inf"), [1.0, 2.0], np.array([1.0])],
)
def test_metric_rejects_nonfinite_or_nonscalar_results(bad_value):
    x, y = data()
    metric = fb.CustomMetric("invalid", lambda *_args: bad_value)

    with pytest.raises(ValueError):
        fb.train(
            params(eval_metric=metric),
            x,
            label=y,
            num_boost_round=1,
            evals=[(x, y, "valid")],
            quiet=True,
        )


def test_retained_callback_arrays_remain_independent_snapshots():
    x, y = data()
    weight = np.linspace(1.0, 2.0, len(y), dtype=np.float32)
    retained = []

    def retain(pred, labels, sample_weight, group_ptr):
        retained.append(
            (
                pred,
                pred.copy(),
                labels,
                labels.copy(),
                sample_weight,
                sample_weight.copy(),
                group_ptr,
            )
        )
        return 0.0

    fb.train(
        params(eval_metric=fb.CustomMetric("retain", retain)),
        x,
        label=y,
        num_boost_round=3,
        evals=[(x, y, weight, "valid")],
        quiet=True,
    )

    assert len(retained) == 3
    for pred, pred_copy, labels, labels_copy, sample_weight, weight_copy, group_ptr in retained:
        assert pred.dtype == labels.dtype == sample_weight.dtype == np.float32
        assert group_ptr is None
        np.testing.assert_array_equal(pred, pred_copy)
        np.testing.assert_array_equal(labels, labels_copy)
        np.testing.assert_array_equal(sample_weight, weight_copy)


def test_multiclass_custom_metric_receives_row_major_margin_matrix():
    x = np.arange(18, dtype=np.float32).reshape(6, 3)
    labels = np.array([0, 1, 2, 0, 1, 2], dtype=np.float32)
    received = []

    def matrix_shape(pred, y, sample_weight, group_ptr):
        received.append((pred.copy(), y.copy(), sample_weight, group_ptr))
        assert sample_weight is None and group_ptr is None
        return float(np.mean(pred))

    metric = fb.CustomMetric("margin_mean", matrix_shape)
    result = {}
    model = fb.train(
        params("multi:softprob", num_class=3, eval_metric=metric),
        x,
        label=labels,
        num_boost_round=1,
        evals=[(x, labels, "valid")],
        evals_result=result,
        quiet=True,
    )

    raw = model.predict(x, output_margin=True)
    assert raw.shape == (len(labels), 3)
    assert len(received) == 1
    seen_pred, seen_labels, _, _ = received[0]
    assert seen_pred.dtype == np.float32
    assert seen_pred.shape == (len(labels), 3)
    assert seen_labels.shape == (len(labels),)
    np.testing.assert_array_equal(seen_pred, raw)
    np.testing.assert_array_equal(seen_labels, labels)
    assert result["valid"]["margin_mean"] == pytest.approx([float(np.mean(raw))])


def test_ranking_custom_metric_receives_group_pointer_offsets():
    x = np.arange(8, dtype=np.float32).reshape(8, 1)
    labels = np.array([0, 1, 2, 0, 2, 1, 0, 1], dtype=np.float32)
    qid = np.repeat(np.arange(4), 2)
    eval_qid = np.repeat(np.arange(4), 2)
    group_weight = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    group_ptrs = []
    received_weights = []

    def with_groups(pred, y, sample_weight, group_ptr):
        assert sample_weight is not None
        group_ptrs.append(group_ptr.copy())
        received_weights.append(sample_weight.copy())
        return float(np.mean(pred))

    fb.train(
        params("rank:ndcg", eval_metric=fb.CustomMetric("grouped", with_groups)),
        x,
        label=labels,
        qid=qid,
        num_boost_round=1,
        evals=[(x, labels, group_weight, "valid")],
        eval_qid=[eval_qid],
        quiet=True,
    )

    assert len(group_ptrs) == 1
    np.testing.assert_array_equal(group_ptrs[0], np.array([0, 2, 4, 6, 8], dtype=np.uint32))
    assert len(received_weights) == 1
    np.testing.assert_array_equal(received_weights[0], group_weight)


def test_nonranking_custom_metric_keeps_row_weights_with_query_groups():
    x, y = data()
    qid = np.repeat(np.arange(4), 2)
    eval_qid = np.repeat(np.arange(4), 2)
    row_weight = np.arange(1, 9, dtype=np.float32)
    received = []

    def check_row_weights_and_groups(pred, labels, sample_weight, group_ptr):
        assert sample_weight is not None
        assert group_ptr is not None
        received.append((pred.copy(), labels.copy(), sample_weight.copy(), group_ptr.copy()))
        return 0.0

    fb.train(
        params(eval_metric=fb.CustomMetric("row_weighted", check_row_weights_and_groups)),
        x,
        label=y,
        qid=qid,
        num_boost_round=1,
        evals=[(x, y, row_weight, "valid")],
        eval_qid=[eval_qid],
        quiet=True,
    )

    assert len(received) == 1
    pred, labels, weights, group_ptr = received[0]
    assert pred.shape == labels.shape == weights.shape == y.shape
    np.testing.assert_array_equal(weights, row_weight)
    np.testing.assert_array_equal(group_ptr, np.array([0, 2, 4, 6, 8], dtype=np.uint32))


def test_sklearn_clone_preserves_custom_metric_descriptor():
    pytest.importorskip("sklearn")
    from sklearn.base import clone

    from ferrisboost.sklearn import FerrisBoostRanker

    metric = fb.CustomMetric("clone_metric", lambda *_args: 2.5, greater_is_better=True)
    estimator = FerrisBoostRanker(n_estimators=1, eval_metric=metric)
    cloned = clone(estimator)

    assert cloned.eval_metric is not metric
    assert cloned.eval_metric.name == "clone_metric"
    assert cloned.eval_metric.greater_is_better is True


@pytest.mark.parametrize(
    ("greater_is_better", "expected_score"),
    [(False, -2.5), (True, 2.5)],
)
def test_ranker_fit_and_score_support_custom_metric_direction(greater_is_better, expected_score):
    pd = pytest.importorskip("pandas")
    pytest.importorskip("sklearn")
    from ferrisboost.sklearn import FerrisBoostRanker

    x = np.arange(6, dtype=np.float32)
    qid = np.repeat(np.arange(3), 2)
    frame = pd.DataFrame({"feature": x, "qid": qid})
    labels = np.array([0, 1, 2, 0, 2, 1], dtype=np.float32)
    received = []

    def fixed_score(pred, y, sample_weight, group_ptr):
        assert sample_weight is None
        np.testing.assert_array_equal(
            group_ptr, np.array([0, 2, 4, 6], dtype=np.uint32)
        )
        received.append((pred.shape, y.shape))
        return 2.5

    metric = fb.CustomMetric(
        "constant_custom_score", fixed_score, greater_is_better=greater_is_better
    )
    ranker = FerrisBoostRanker(
        n_estimators=1,
        max_depth=1,
        n_jobs=1,
        eval_metric=metric,
    )
    ranker.fit(frame, labels, eval_set=[(frame, labels)])

    assert ranker.evals_result_["validation_0"]["constant_custom_score"] == [2.5]
    assert ranker.score(frame, labels) == expected_score
    assert received == [((6,), (6,)), ((6,), (6,))]


@pytest.mark.skipif(os.environ.get("FB_TEST_CUDA") != "1", reason="requires explicit CUDA test opt-in")
@pytest.mark.parametrize("gpu_math", ["exact", "fast"])
def test_gpu_custom_metric_receives_each_round_raw_margins(gpu_math):
    x, y = data()
    received = []

    def capture_mean(pred, labels, sample_weight, group_ptr):
        assert sample_weight is None and group_ptr is None
        received.append(pred.copy())
        return float(np.mean(pred))

    result = {}
    model = fb.train(
        params(
            eval_metric=fb.CustomMetric("gpu_margin_mean", capture_mean),
            device="cuda:0",
            gpu_math=gpu_math,
        ),
        x,
        label=y,
        num_boost_round=3,
        evals=[(x, y, "valid")],
        evals_result=result,
        quiet=True,
    )

    assert len(received) == 3
    assert len(result["valid"]["gpu_margin_mean"]) == 3
    for round_index, observed in enumerate(received):
        expected = model.predict(
            x,
            output_margin=True,
            iteration_range=(0, round_index + 1),
        )
        assert observed.dtype == np.float32
        np.testing.assert_array_equal(observed, expected)
        assert result["valid"]["gpu_margin_mean"][round_index] == pytest.approx(
            float(np.mean(expected))
        )


@pytest.mark.skipif(os.environ.get("FB_TEST_CUDA") != "1", reason="requires explicit CUDA test opt-in")
def test_gpu_custom_metric_exception_skips_after_iteration_and_releases_for_next_train():
    x, y = data()
    callback_rounds = []

    class MarkerError(RuntimeError):
        pass

    def fail(_pred, _labels, _sample_weight, _group_ptr):
        raise MarkerError("GPU metric callback marker")

    def after_iteration(round_index, _metrics):
        callback_rounds.append(round_index)
        return True

    with pytest.raises(MarkerError, match="GPU metric callback marker"):
        fb.train(
            params(eval_metric=fb.CustomMetric("gpu_raises", fail), device="cuda:0"),
            x,
            label=y,
            num_boost_round=2,
            evals=[(x, y, "valid")],
            callbacks=[after_iteration],
            quiet=True,
        )

    assert callback_rounds == []
    model = fb.train(
        params(device="cuda:0"),
        x,
        label=y,
        num_boost_round=1,
        quiet=True,
    )
    assert model.num_trees == 1


@pytest.mark.parametrize("file_format", ["csv", "parquet"])
def test_custom_metric_runs_from_native_file_train_and_eval_paths(tmp_path, file_format):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")

    def write_file(path, x, labels, weights):
        table = pa.table({"feature": x[:, 0], "target": labels, "weight": weights})
        if file_format == "parquet":
            pq.write_table(table, path)
        else:
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.writer(stream)
                writer.writerow(["feature", "target", "weight"])
                writer.writerows(zip(x[:, 0], labels, weights))

    train_x, train_y = data()
    train_weight = np.linspace(0.5, 1.5, len(train_y), dtype=np.float32)
    valid_x = (np.arange(6, dtype=np.float32) + 0.25).reshape(-1, 1)
    valid_y = np.array([0, 0, 0, 1, 1, 1], dtype=np.float32)
    valid_weight = np.array([1, 2, 3, 4, 5, 6], dtype=np.float32)
    train_path = tmp_path / f"train.{file_format}"
    valid_path = tmp_path / f"valid.{file_format}"
    write_file(train_path, train_x, train_y, train_weight)
    write_file(valid_path, valid_x, valid_y, valid_weight)
    received = []

    def weighted_absolute_sum(pred, labels, sample_weight, group_ptr):
        assert group_ptr is None
        assert sample_weight is not None
        received.append((pred.copy(), labels.copy(), sample_weight.copy()))
        return float(np.sum(sample_weight * np.abs(pred - labels)))

    metric = fb.CustomMetric("file_weighted_abs_sum", weighted_absolute_sum)
    result = {}
    model = fb.train(
        params(eval_metric=metric),
        str(train_path),
        label="target",
        weight="weight",
        num_boost_round=1,
        evals=[(str(valid_path), "valid")],
        evals_result=result,
        quiet=True,
    )

    assert len(received) == 1
    pred, labels, weights = received[0]
    assert pred.dtype == labels.dtype == weights.dtype == np.float32
    np.testing.assert_array_equal(labels, valid_y)
    np.testing.assert_array_equal(weights, valid_weight)
    np.testing.assert_array_equal(
        pred, model.predict(str(valid_path), output_margin=True)
    )
    expected_score = float(np.sum(valid_weight * np.abs(pred - valid_y)))
    assert result["valid"]["file_weighted_abs_sum"] == pytest.approx(
        [expected_score], rel=0, abs=1e-7
    )

    # The callback belongs to the training request; it is not part of the
    # saved model state. File prediction after reload uses only the model.
    model_path = tmp_path / f"model-{file_format}.json"
    model.save_model(str(model_path), quiet=True)
    loaded = fb.Model.load_model(str(model_path), quiet=True)
    np.testing.assert_array_equal(
        loaded.predict(str(valid_path), output_margin=True),
        model.predict(str(valid_path), output_margin=True),
    )

    # Reusing the descriptor for explicit continuation creates a fresh metric
    # adapter; the same file-based callback contract still applies.
    continuation_result = {}
    continued = fb.train(
        params(eval_metric=metric),
        str(train_path),
        label="target",
        weight="weight",
        num_boost_round=1,
        evals=[(str(valid_path), "valid")],
        evals_result=continuation_result,
        init_model=model,
        quiet=True,
    )
    assert continued.num_trees == model.num_trees + 1
    assert len(received) == 2
    assert len(continuation_result["valid"]["file_weighted_abs_sum"]) == 1
