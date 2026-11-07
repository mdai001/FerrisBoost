"""Feature names from in-memory pandas / PyArrow training, and explanation
options on DataFrame / Table input."""
import numpy as np
import pytest

import ferrisboost as fb

pd = pytest.importorskip("pandas")
pa = pytest.importorskip("pyarrow")

PARAMS = {"objective": "binary:logistic", "max_depth": 3, "eta": 0.3}


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(2000, 4)).astype(np.float32)
    y = (x[:, 0] + 0.5 * x[:, 2] - x[:, 3] > 0).astype(np.float32)
    df = pd.DataFrame(x, columns=["alpha", "beta", "gamma", "delta"])
    return x, y, df


def test_numpy_training_keeps_positional_names(data):
    x, y, _ = data
    m = fb.train(PARAMS, x, label=y, num_boost_round=5, quiet=True)
    assert all(k.startswith("f") for k in m.get_score())


@pytest.mark.parametrize("kind", ["pandas", "arrow"])
def test_named_training_reports_column_names(data, kind):
    x, y, df = data
    src = df if kind == "pandas" else pa.Table.from_pandas(df, preserve_index=False)
    m = fb.train(PARAMS, src, label=y, num_boost_round=5, quiet=True)
    keys = set(m.get_score())
    assert keys and keys <= {"alpha", "beta", "gamma", "delta"}
    positional = fb.train(PARAMS, x, label=y, num_boost_round=5, quiet=True)
    # Same model apart from the names: scores map one to one.
    by_pos = positional.get_score(importance_type="gain")
    by_name = m.get_score(importance_type="gain")
    names = list(df.columns)
    assert {names[int(k[1:])]: v for k, v in by_pos.items()} == by_name


def test_dataframe_predict_binds_by_name(data):
    x, y, df = data
    m = fb.train(PARAMS, df, label=y, num_boost_round=5, quiet=True)
    base = m.predict(x)
    shuffled = df[["delta", "alpha", "gamma", "beta"]].assign(extra=1.0)
    np.testing.assert_array_equal(m.predict(df), base)
    np.testing.assert_array_equal(m.predict(shuffled), base)
    table = pa.Table.from_pandas(shuffled, preserve_index=False)
    np.testing.assert_array_equal(m.predict(table), base)
    with pytest.raises(ValueError, match="gamma"):
        m.predict(df.drop(columns=["gamma"]))


def test_names_survive_save_load(data, tmp_path):
    _, y, df = data
    m = fb.train(PARAMS, df, label=y, num_boost_round=5, quiet=True)
    p = tmp_path / "m.json"
    m.save_model(str(p), quiet=True)
    loaded = fb.Model.load_model(str(p), quiet=True)
    assert loaded.get_score() == m.get_score()


@pytest.mark.parametrize("opt", ["pred_contribs", "pred_interactions", "pred_leaf"])
@pytest.mark.parametrize("kind", ["pandas", "arrow"])
def test_explanations_accept_tables(data, opt, kind):
    x, y, df = data
    m = fb.train(PARAMS, df, label=y, num_boost_round=5, quiet=True)
    src = df[::-1].iloc[::-1] if kind == "pandas" else pa.Table.from_pandas(df, preserve_index=False)
    np.testing.assert_array_equal(m.predict(src, **{opt: True}), m.predict(x, **{opt: True}))


def test_explanations_accept_frames_for_positional_models(data):
    x, y, df = data
    m = fb.train(PARAMS, x, label=y, num_boost_round=5, quiet=True)
    np.testing.assert_array_equal(m.predict(df, pred_contribs=True), m.predict(x, pred_contribs=True))


def test_xgboost_reads_the_names(data, tmp_path):
    xgb = pytest.importorskip("xgboost")
    x, y, df = data
    m = fb.train(PARAMS, df, label=y, num_boost_round=5, quiet=True)
    p = tmp_path / "m.json"
    m.save_model(str(p), quiet=True)
    b = xgb.Booster(model_file=str(p))
    assert b.feature_names == list(df.columns)
    assert b.get_score(importance_type="gain") == pytest.approx(m.get_score(importance_type="gain"))
