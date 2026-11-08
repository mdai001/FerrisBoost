"""File prediction reads only the feature columns the trees split on.

Observable proof: a column no tree uses may hold a type the readers reject
(strings) and a file prediction still equals the array prediction bit for bit,
for single-output, multiclass and multi-quantile models, `predict_to_parquet`
and the explanation outputs. The schema rule stays strict: every feature name of
the model must exist in the file, used or not, and a positional model needs the
exact width.
"""
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import ferrisboost as fb

N, M = 300, 12
RNG = np.random.default_rng(11)
X = RNG.integers(0, 10, size=(N, M)).astype(np.float32)
SIGNAL = X[:, 0] - X[:, 1] + 0.5 * X[:, 2]
Y01 = (SIGNAL + RNG.normal(0, 1, N) > 0.5).astype(np.float32)
Y3 = np.digitize(SIGNAL + RNG.normal(0, 1, N), [-1.0, 2.0]).astype(np.float32)
YREG = (SIGNAL + RNG.normal(0, 1, N)).astype(np.float32)
NAMES = [f"c{i:02d}" for i in range(M)]

BASE = {"max_depth": 2, "eta": 0.3, "nthread": 1, "colsample_bytree": 0.5, "seed": 3}
MODELS = {
    "binary": ({"objective": "binary:logistic"}, Y01),
    "softprob": ({"objective": "multi:softprob", "num_class": 3}, Y3),
    "softmax": ({"objective": "multi:softmax", "num_class": 3}, Y3),
    "quantiles": ({"objective": "reg:quantileerror", "quantile_alpha": [0.2, 0.5, 0.8]}, YREG),
}


def train(kind):
    extra, y = MODELS[kind]
    frame = pd.DataFrame(X, columns=NAMES)
    return fb.train({**BASE, **extra}, frame, label=y, num_boost_round=3, quiet=True)


def used_features(model):
    return sorted(int(name[1:]) for name in model.get_score())


def unused_column(model):
    used = set(used_features(model))
    return next(i for i in range(M) if i not in used)


def write_files(tmp_path, unused, shuffle=True):
    """Parquet and CSV with columns shuffled, a label column, and the unused feature
    column replaced by strings (a type the readers cannot decode)."""
    order = list(range(M))
    if shuffle:
        order = order[::-1]
    cols = {NAMES[i]: X[:, i] for i in order}
    cols["label"] = Y01
    cols[NAMES[unused]] = [f"s{v}" for v in range(N)]
    table = pa.table(cols)
    parquet = str(tmp_path / "wide.parquet")
    pq.write_table(table, parquet)
    csv = str(tmp_path / "wide.csv")
    table.to_pandas().to_csv(csv, index=False)
    return parquet, csv


@pytest.fixture(params=sorted(MODELS))
def case(request, tmp_path):
    model = train(request.param)
    unused = unused_column(model)
    parquet, csv = write_files(tmp_path, unused)
    return request.param, model, unused, parquet, csv


def array_input(unused):
    return X  # the model is named, so the array is in the model's column order


def test_predict_from_files_equals_arrays_without_reading_unused_columns(case):
    kind, model, unused, parquet, csv = case
    assert len(used_features(model)) < M
    for margin in (False, True):
        want = model.predict(X, output_margin=margin)
        for path in (parquet, csv):
            np.testing.assert_array_equal(model.predict(path, output_margin=margin), want)


def test_predict_to_parquet_equals_arrays(case, tmp_path):
    kind, model, unused, parquet, csv = case
    want = np.asarray(model.predict(X, output_margin=True))
    got_rows = model.predict_to_parquet(parquet, str(tmp_path / "o.parquet"), output_margin=True)
    assert got_rows == N
    out = pd.read_parquet(tmp_path / "o.parquet").to_numpy()
    np.testing.assert_array_equal(out.reshape(want.shape), want)
    # A CSV source streams the same way.
    model.predict_to_parquet(csv, str(tmp_path / "o2.parquet"), output_margin=True)
    out = pd.read_parquet(tmp_path / "o2.parquet").to_numpy()
    np.testing.assert_array_equal(out.reshape(want.shape), want)


@pytest.mark.parametrize("option", ["pred_contribs", "pred_leaf", "pred_interactions"])
def test_explanations_from_files_equal_arrays(case, option):
    kind, model, unused, parquet, csv = case
    want = model.predict(X, **{option: True})
    got = model.predict(parquet, **{option: True})
    assert got.shape == want.shape
    np.testing.assert_array_equal(got, want)
    if option == "pred_contribs":
        # a feature no tree splits on contributes exactly zero
        zero = got[..., unused]
        assert not np.any(zero)
        np.testing.assert_array_equal(model.predict(csv, **{option: True}), want)


def test_explanations_honour_iteration_range_and_margin_files(tmp_path):
    model = train("binary")
    unused = unused_column(model)
    parquet, _ = write_files(tmp_path, unused)
    want = model.predict(X, pred_contribs=True, iteration_range=(0, 2))
    got = model.predict(parquet, pred_contribs=True, iteration_range=(0, 2))
    np.testing.assert_array_equal(got, want)


def test_every_model_feature_must_exist_even_if_no_tree_uses_it(case, tmp_path):
    kind, model, unused, parquet, csv = case
    table = pq.read_table(parquet).drop_columns([NAMES[unused]])
    path = str(tmp_path / "without_unused.parquet")
    pq.write_table(table, path)
    for call in (
        lambda: model.predict(path),
        lambda: model.predict_to_parquet(path, str(tmp_path / "x.parquet")),
        lambda: model.predict(path, pred_contribs=True),
    ):
        with pytest.raises(ValueError, match=NAMES[unused]):
            call()
    assert not (tmp_path / "x.parquet").exists()


def test_positional_models_still_need_the_exact_width(tmp_path):
    extra, y = MODELS["softprob"]
    model = fb.train({**BASE, **extra}, X, label=y, num_boost_round=3, quiet=True)
    wide = tmp_path / "wide.csv"
    pd.DataFrame(np.hstack([X, X[:, :1]])).to_csv(wide, index=False, header=False)
    with pytest.raises(ValueError, match="positional mode"):
        model.predict(str(wide), header=False)
    exact = tmp_path / "exact.csv"
    pd.DataFrame(X).to_csv(exact, index=False, header=False)
    np.testing.assert_array_equal(model.predict(str(exact), header=False), model.predict(X))
    np.testing.assert_array_equal(
        model.predict(str(exact), header=False, pred_contribs=True),
        model.predict(X, pred_contribs=True),
    )


def test_a_model_with_no_split_still_predicts_from_files(tmp_path):
    # Root-only trees use no feature: the reader still needs one column to count rows.
    frame = pd.DataFrame(X, columns=NAMES)
    model = fb.train({**BASE, "objective": "multi:softprob", "num_class": 3, "max_depth": 0},
                     frame, label=Y3, num_boost_round=2, quiet=True)
    path = str(tmp_path / "d.parquet")
    frame.assign(label=Y3).to_parquet(path)
    np.testing.assert_array_equal(model.predict(path), model.predict(X))
    np.testing.assert_array_equal(model.predict(path, pred_contribs=True),
                                  model.predict(X, pred_contribs=True))
