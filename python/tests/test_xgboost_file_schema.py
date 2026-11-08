"""File inputs and XGBoost-trained models.

An XGBoost model JSON records no FerrisBoost schema mode. It is bound to file
columns **by name** when it carries feature names (the names say which column
feeds which feature, so the file's column order is free) and is refused with a
clear `ValueError` when it has none (binding by position would silently
mis-score a file whose columns are in another order).

Pinned here: continuation from an XGBoost JSON by file input is bit-exact vs
XGBoost's own continuation on lossless data, file prediction works the same way,
the nameless case is an error with guidance, and a FerrisBoost model's behaviour
does not depend on any of this.
"""
import json

import numpy as np
import pandas as pd
import pytest

import ferrisboost as fb

xgb = pytest.importorskip("xgboost")

NAMES = ["alpha", "beta", "gamma", "delta", "eps"]
OBJECTIVES = ["binary:logistic", "multi:softprob"]


def make(objective, n=900, m=5, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 16, size=(n, m)).astype(np.float32)
    signal = 0.3 * (x[:, 0] - 8) + 0.2 * (x[:, 1] % 4) - 0.1 * x[:, 2]
    if objective == "binary:logistic":
        y = (signal + rng.normal(0, 1.0, n) > 0).astype(np.float32)
    else:
        y = np.digitize(signal + rng.normal(0, 0.6, n), [-0.8, 0.8]).astype(np.float32)
    return x, y


def params(objective, **extra):
    out = {"objective": objective, "max_depth": 3, "eta": 0.3, "nthread": 1, "max_bin": 255}
    if objective == "multi:softprob":
        out["num_class"] = 3
    return {**out, **extra}


def trained_xgb(objective, tmp_path, names):
    x, y = make(objective)
    d = xgb.DMatrix(x, y, feature_names=names)
    first = xgb.train({**params(objective), "tree_method": "hist"}, d, 4)
    path = tmp_path / "xgb.json"
    first.save_model(str(path))
    return x, y, d, first, str(path)


def shuffled_frame(x, y, names, order=(3, 0, 4, 2, 1)):
    """Feature columns in another order than the model's, label in the middle."""
    cols = {names[i]: x[:, i] for i in order}
    frame = pd.DataFrame(cols)
    frame.insert(2, "label", y)
    return frame


def write_parquet(tmp_path, frame, name="data"):
    path = tmp_path / f"{name}.parquet"
    frame.to_parquet(path)
    return str(path)


def write_csv(tmp_path, frame, name="data"):
    path = tmp_path / f"{name}.csv"
    frame.to_csv(path, index=False)
    return str(path)


@pytest.mark.parametrize("fmt", ["parquet", "csv"])
@pytest.mark.parametrize("objective", OBJECTIVES)
def test_named_xgboost_model_continues_from_a_file_bit_exact(objective, fmt, tmp_path):
    x, y, d, first, path = trained_xgb(objective, tmp_path, NAMES)
    cont = xgb.train({**params(objective), "tree_method": "hist"}, d, 4, xgb_model=first)
    want = cont.predict(xgb.DMatrix(x, feature_names=NAMES), output_margin=True)

    frame = shuffled_frame(x, y, NAMES)
    data = (write_parquet if fmt == "parquet" else write_csv)(tmp_path, frame)
    from_file = fb.train(params(objective), data, label="label", num_boost_round=4,
                         xgb_model=path, quiet=True)
    assert from_file.feature_names == NAMES
    np.testing.assert_array_equal(from_file.predict(x, output_margin=True), want)

    # An `xgboost.Booster` carries its names the same way.
    from_booster = fb.train(params(objective), data, label="label", num_boost_round=4,
                            xgb_model=first, quiet=True)
    np.testing.assert_array_equal(from_booster.predict(x, output_margin=True), want)

    # Validation files are bound by name too.
    res = {}
    fb.train(params(objective), data, label="label", num_boost_round=2, xgb_model=path,
             evals=[(data, "v")], evals_result=res, quiet=True)
    assert "v" in res


def test_named_xgboost_model_scores_files_by_name(tmp_path):
    x, y, d, first, path = trained_xgb("binary:logistic", tmp_path, NAMES)
    want = first.predict(xgb.DMatrix(x, feature_names=NAMES), output_margin=True)
    model = fb.Model.load_model(path, quiet=True)
    assert model.feature_names == NAMES

    frame = shuffled_frame(x, y, NAMES)
    for data in (write_parquet(tmp_path, frame), write_csv(tmp_path, frame)):
        np.testing.assert_array_equal(model.predict(data, output_margin=True), want)
    out = str(tmp_path / "pred.parquet")
    model.predict_to_parquet(write_parquet(tmp_path, frame, "again"), out, output_margin=True)
    got = pd.read_parquet(out)["prediction"].to_numpy()
    np.testing.assert_array_equal(got, want)
    contribs = model.predict(write_parquet(tmp_path, frame, "c"), pred_contribs=True)
    np.testing.assert_array_equal(contribs, model.predict(x, pred_contribs=True))

    # A named model cannot read a positional (headerless) CSV, as ever.
    headerless = str(tmp_path / "h.csv")
    frame.to_csv(headerless, index=False, header=False)
    with pytest.raises(ValueError, match="header=False"):
        model.predict(headerless, header=False)


def test_a_missing_feature_column_is_an_error(tmp_path):
    x, y, d, first, path = trained_xgb("binary:logistic", tmp_path, NAMES)
    frame = shuffled_frame(x, y, NAMES).drop(columns=["gamma"])
    data = write_parquet(tmp_path, frame)
    with pytest.raises(ValueError, match="gamma"):
        fb.Model.load_model(path, quiet=True).predict(data)
    with pytest.raises(ValueError, match="gamma"):
        fb.train(params("binary:logistic"), data, label="label", num_boost_round=2,
                 xgb_model=path, quiet=True)


def test_a_surplus_training_column_is_an_error_not_a_silent_feature(tmp_path):
    x, y, d, first, path = trained_xgb("binary:logistic", tmp_path, NAMES)
    frame = shuffled_frame(x, y, NAMES)
    frame["extra"] = 1.0
    with pytest.raises(ValueError, match="extra"):
        fb.train(params("binary:logistic"), write_parquet(tmp_path, frame), label="label",
                 num_boost_round=2, xgb_model=path, quiet=True)


@pytest.mark.parametrize("objective", OBJECTIVES)
def test_xgboost_model_without_names_needs_arrays_not_files(objective, tmp_path):
    x, y = make(objective)
    first = xgb.train({**params(objective), "tree_method": "hist"}, xgb.DMatrix(x, y), 4)
    path = tmp_path / "nameless.json"
    first.save_model(str(path))
    model = fb.Model.load_model(str(path), quiet=True)
    assert model.feature_names == []

    frame = pd.DataFrame({f"f{i}": x[:, i] for i in range(5)})
    frame["label"] = y
    data = write_parquet(tmp_path, frame)
    with pytest.raises(ValueError, match="feature names"):
        model.predict(data)
    with pytest.raises(ValueError, match="feature names"):
        model.predict(write_csv(tmp_path, frame))
    with pytest.raises(ValueError, match="feature names"):
        model.predict_to_parquet(data, str(tmp_path / "o.parquet"))
    with pytest.raises(ValueError, match="feature names"):
        model.predict(data, pred_contribs=True)
    with pytest.raises(ValueError, match="feature names"):
        fb.train(params(objective), data, label="label", num_boost_round=2,
                 xgb_model=str(path), quiet=True)
    assert not (tmp_path / "o.parquet").exists()

    # Arrays keep working: the caller owns the column order.
    cont = fb.train(params(objective), x, label=y, num_boost_round=2, xgb_model=str(path),
                    quiet=True)
    new_trees = 2 * (model.num_class if model.num_class > 1 else 1)
    assert cont.num_trees == model.num_trees + new_trees
    # Adding the names to the model (what the error suggests) makes files work.
    booster = xgb.Booster()
    booster.load_model(str(path))
    booster.feature_names = [f"f{i}" for i in range(5)]
    named = str(tmp_path / "named.json")
    booster.save_model(named)
    again = fb.train(params(objective), data, label="label", num_boost_round=2,
                     xgb_model=named, quiet=True)
    np.testing.assert_array_equal(again.predict(x, output_margin=True),
                                  cont.predict(x, output_margin=True))


def test_incomplete_or_duplicate_names_are_not_trusted(tmp_path):
    x, y, d, first, path = trained_xgb("binary:logistic", tmp_path, NAMES)
    frame = shuffled_frame(x, y, NAMES)
    data = write_parquet(tmp_path, frame)
    doc = json.loads(open(path).read())
    for bad in (NAMES[:-1], NAMES[:-1] + [NAMES[0]]):
        doc["learner"]["feature_names"] = bad
        broken = tmp_path / "broken.json"
        broken.write_text(json.dumps(doc))
        with pytest.raises(ValueError, match="feature names"):
            fb.Model.load_model(str(broken), quiet=True).predict(data)


def test_ferrisboost_models_keep_their_recorded_mode(tmp_path):
    x, y = make("binary:logistic")
    frame = shuffled_frame(x, y, NAMES, order=(0, 1, 2, 3, 4))
    data = write_parquet(tmp_path, frame)
    named = fb.train(params("binary:logistic"), data, label="label", num_boost_round=3, quiet=True)
    positional = fb.train(params("binary:logistic"), x, label=y, num_boost_round=3, quiet=True)
    want = named.predict(x, output_margin=True)
    np.testing.assert_array_equal(named.predict(data, output_margin=True), want)
    # Positional models still refuse a headered file (and never silently bind by name).
    with pytest.raises(ValueError):
        positional.predict(write_csv(tmp_path, frame))
    # An FB model whose saved schema mode is stripped (an old file) is bound by name
    # when it has names and refused when it has none: the recorded-mode rule is unchanged.
    for model, expect_named in ((named, True), (positional, False)):
        path = tmp_path / ("n.json" if expect_named else "p.json")
        model.save_model(str(path), quiet=True)
        doc = json.loads(path.read_text())
        doc["learner"]["attributes"].pop("ferrisboost_schema_mode")
        path.write_text(json.dumps(doc))
        old = fb.Model.load_model(str(path), quiet=True)
        if expect_named:
            np.testing.assert_array_equal(old.predict(data, output_margin=True), want)
        else:
            with pytest.raises(ValueError, match="feature names"):
                old.predict(data)
