"""Prediction links attached to models trained with custom objectives."""

import csv

import numpy as np
import pytest

import ferrisboost as fb


def scalar_case(n=32):
    x = np.arange(n * 2, dtype=np.float32).reshape(n, 2) / np.float32(9)
    y = (0.4 * x[:, 0] - 0.15 * x[:, 1]).astype(np.float32)
    return x, y


def scalar_callback(margin, labels):
    return np.ascontiguousarray(margin - labels), np.ones_like(labels)


def scalar_model(n=32):
    x, y = scalar_case(n)
    params = {
        "objective": "reg:squarederror",
        "max_depth": 2,
        "eta": 0.3,
        "max_bin": 16,
        "nthread": 1,
        "custom_objective": fb.CustomObjective("link_scalar", scalar_callback),
    }
    return x, y, params, fb.train(params, x, label=y, num_boost_round=3, quiet=True)


def test_builtin_links_transform_custom_scalar_and_multiclass_outputs():
    x, _y, _params, model = scalar_model()
    raw = model.predict(x, output_margin=True)
    model.set_prediction_link("identity")
    np.testing.assert_array_equal(model.predict(x), raw)
    model.set_prediction_link("sigmoid")
    np.testing.assert_allclose(model.predict(x), 1.0 / (1.0 + np.exp(-raw)), rtol=1e-6)
    model.set_prediction_link("exp")
    np.testing.assert_allclose(model.predict(x), np.exp(raw), rtol=1e-6)
    model.set_prediction_link(None)

    n, k = 24, 3
    mx = np.arange(n * 2, dtype=np.float32).reshape(n, 2) / np.float32(7)
    my = (np.arange(n) % k).astype(np.float32)

    def multi_grad(margin, labels):
        target = np.eye(k, dtype=np.float32)[labels.astype(np.int64)]
        return np.ascontiguousarray(margin - target), np.ones_like(margin)

    multi = fb.train(
        {
            "objective": "multi:softprob",
            "num_class": k,
            "max_depth": 1,
            "nthread": 1,
            "custom_objective": fb.CustomObjective(
                "link_multi", multi_grad, kind="multiclass"
            ),
        },
        mx,
        label=my,
        num_boost_round=2,
        quiet=True,
    )
    margins = multi.predict(mx, output_margin=True)
    multi.set_prediction_link("softmax")
    probs = multi.predict(mx)
    assert probs.shape == (n, k)
    centered = margins - margins.max(axis=1, keepdims=True)
    expected = np.exp(centered) / np.exp(centered).sum(axis=1, keepdims=True)
    np.testing.assert_allclose(probs, expected, rtol=1e-6, atol=1e-7)


def test_named_link_preserves_shape_iteration_range_and_bypasses_margins_and_shap():
    x, _y, _params, model = scalar_model(20)
    inputs = []

    def double(raw):
        inputs.append(raw.copy())
        return np.ascontiguousarray(raw * np.float32(2))

    model.set_prediction_link("double", callback=double)
    raw_prefix = model.predict(x, output_margin=True, iteration_range=(0, 1))
    got = model.predict(x, iteration_range=(0, 1))
    np.testing.assert_array_equal(got, raw_prefix * np.float32(2))
    assert inputs[-1].shape == (len(x),)

    raw = model.predict(x, output_margin=True)
    np.testing.assert_array_equal(model.predict(x, output_margin=True), raw)
    contributions = model.predict(x, pred_contribs=True)
    np.testing.assert_allclose(contributions.sum(axis=1), raw, rtol=1e-6, atol=1e-6)


def test_named_link_streams_file_and_parquet_predictions_in_batches(tmp_path):
    pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    x, y = scalar_case(24)
    source = tmp_path / "data.csv"
    with source.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["f0", "f1", "target"])
        writer.writerows(np.column_stack((x, y)).tolist())
    params = {
        "objective": "reg:squarederror",
        "max_depth": 1,
        "nthread": 1,
        "custom_objective": fb.CustomObjective("link_file", scalar_callback),
    }
    model = fb.train(params, str(source), label="target", num_boost_round=2, quiet=True)
    observed = []

    def offset(raw):
        observed.append(raw.shape)
        return np.ascontiguousarray(raw + np.float32(0.25))

    model.set_prediction_link("offset", callback=offset)
    raw_file = model.predict(str(source), output_margin=True)
    np.testing.assert_allclose(model.predict(str(source)), raw_file + 0.25, rtol=0, atol=1e-7)
    parquet_out = tmp_path / "pred.parquet"
    assert model.predict_to_parquet(str(source), str(parquet_out)) == len(x)
    table = pq.read_table(parquet_out)
    np.testing.assert_allclose(table["prediction"].to_numpy(), raw_file + 0.25, rtol=0, atol=1e-7)
    assert observed and all(len(shape) == 1 for shape in observed)


def test_named_link_save_load_rebind_and_exact_name(tmp_path):
    x, y, params, model = scalar_model(18)
    raw = model.predict(x, output_margin=True)

    def square(raw_margin):
        return np.ascontiguousarray(raw_margin * raw_margin)

    model.set_prediction_link("square", callback=square)
    path = tmp_path / "named.json"
    model.save_model(str(path), quiet=True)
    loaded = fb.Model.load_model(str(path), quiet=True)
    np.testing.assert_array_equal(loaded.predict(x, output_margin=True), raw)
    with pytest.raises(ValueError, match="callback|rebind|link"):
        loaded.predict(x)
    with pytest.raises(ValueError, match="name|link"):
        loaded.set_prediction_link("wrong_name", callback=square)
    loaded.set_prediction_link("square", callback=square)
    np.testing.assert_array_equal(loaded.predict(x), square(raw))

    continued = fb.train(
        params,
        x,
        label=y,
        base_margin=raw,
        num_boost_round=1,
        init_model=model,
        quiet=True,
    )
    with pytest.raises(ValueError, match="callback|rebind|link"):
        continued.predict(x)
    continued.set_prediction_link("square", callback=square)
    np.testing.assert_array_equal(continued.predict(x), square(continued.predict(x, output_margin=True)))


def test_builtin_link_roundtrip_and_continuation_retain_metadata(tmp_path):
    x, y, params, model = scalar_model(24)
    model.set_prediction_link("sigmoid")
    path = tmp_path / "builtin.json"
    model.save_model(str(path), quiet=True)
    loaded = fb.Model.load_model(str(path), quiet=True)
    np.testing.assert_array_equal(loaded.predict(x), model.predict(x))

    margins = model.predict(x, output_margin=True)
    continued = fb.train(
        params,
        x,
        label=y,
        base_margin=margins,
        num_boost_round=1,
        init_model=model,
        quiet=True,
    )
    np.testing.assert_allclose(continued.predict(x), 1 / (1 + np.exp(-continued.predict(x, output_margin=True))), rtol=1e-6)


def test_link_callback_errors_and_invalid_shapes_are_reported():
    x, _y, _params, model = scalar_model(12)
    model.set_prediction_link(
        "wrong_shape", callback=lambda raw: np.zeros((len(raw), 1), dtype=np.float32)
    )
    with pytest.raises(ValueError, match="shape"):
        model.predict(x)

    def fail(_raw):
        raise LookupError("callback sentinel")

    model.set_prediction_link("raises", callback=fail)
    with pytest.raises((LookupError, RuntimeError), match="callback sentinel"):
        model.predict(x)


def test_builtin_links_rejected_for_non_custom_models():
    x, y = scalar_case(12)
    model = fb.train(
        {"objective": "reg:squarederror", "max_depth": 1},
        x,
        label=y,
        num_boost_round=1,
        quiet=True,
    )
    with pytest.raises(ValueError, match="custom objective"):
        model.set_prediction_link("sigmoid")
