"""predict(iteration_range=(begin, end)):XGBoost 语义(半开区间,按轮次;多分类一轮 K 棵树)。"""
import numpy as np
import pytest

import ferrisboost as fb

xgb = pytest.importorskip("xgboost")


def data(n=1500, m=6, seed=0, k=None):
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 12, size=(n, m)).astype(np.float32)
    x[rng.random(x.shape) < 0.04] = np.nan
    z = np.nan_to_num(x[:, 0], nan=5.0) - 0.6 * np.nan_to_num(x[:, 1], nan=5.0)
    if k is None:
        return x, (z + rng.normal(size=n) > 2).astype(np.float32)
    return x, np.clip(np.digitize(z + rng.normal(size=n), np.quantile(z, np.linspace(0, 1, k + 1)[1:-1])), 0, k - 1).astype(np.float32)


def train_pair(objective, rounds=12, **extra):
    k = extra.get("num_class")
    x, y = data(k=k)
    p = {"objective": objective, "max_depth": 4, "eta": 0.3, "nthread": 1, **extra}
    ours = fb.train(p, x, label=y, num_boost_round=rounds, quiet=True)
    ref = xgb.train({**p, "tree_method": "hist"}, xgb.DMatrix(x, label=y), num_boost_round=rounds)
    return x, ours, ref


RANGES = [(0, 0), (0, 5), (3, 9), (5, 12), (4, 4), (0, 12), (11, 12), (0, 1)]


@pytest.mark.parametrize("objective,extra", [
    ("binary:logistic", {}),
    ("reg:squarederror", {}),
    ("count:poisson", {}),
    ("multi:softprob", {"num_class": 3}),
    ("multi:softmax", {"num_class": 3}),
])
@pytest.mark.parametrize("margin", [False, True])
def test_matches_xgboost(objective, extra, margin):
    if objective in ("count:poisson", "reg:squarederror"):
        x, ours, ref = train_pair(objective, **extra)
    else:
        x, ours, ref = train_pair(objective, **extra)
    d = xgb.DMatrix(x)
    for rng_ in RANGES:
        a = ours.predict(x, output_margin=margin, iteration_range=rng_)
        b = ref.predict(d, output_margin=margin, iteration_range=rng_)
        np.testing.assert_array_equal(a, b, err_msg=str(rng_))


def test_prefix_equals_n_trees_and_does_not_mix():
    x, ours, _ = train_pair("binary:logistic")
    np.testing.assert_array_equal(ours.predict(x, iteration_range=(0, 5)),
                                  ours.predict(x, n_trees=5))
    with pytest.raises(ValueError, match="n_trees"):
        ours.predict(x, n_trees=3, iteration_range=(0, 5))


def test_multiclass_counts_rounds_not_trees():
    x, ours, ref = train_pair("multi:softprob", num_class=3)
    assert ours.num_trees == 36
    a = ours.predict(x, iteration_range=(2, 4))
    np.testing.assert_array_equal(a, ref.predict(xgb.DMatrix(x), iteration_range=(2, 4)))
    # (2, 4) 是 2 轮 = 6 棵树;不等于前缀。
    assert not np.array_equal(a, ours.predict(x, iteration_range=(0, 2)))


@pytest.mark.parametrize("bad", [(0, 13), (13, 14), (5, 3)])
def test_out_of_range_is_rejected(bad):
    x, ours, _ = train_pair("binary:logistic")
    with pytest.raises(ValueError, match="iteration_range"):
        ours.predict(x, iteration_range=bad)


def test_file_input_and_parquet_output(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    x, y = data()
    feats = {f"f{i}": x[:, i] for i in range(x.shape[1])}
    train_path, pred_path = tmp_path / "train.parquet", tmp_path / "pred.parquet"
    pq.write_table(pa.table({**feats, "y": y}), str(train_path))
    pq.write_table(pa.table(feats), str(pred_path))
    m = fb.train({"objective": "binary:logistic", "nthread": 1, "max_depth": 3}, str(train_path),
                 label="y", num_boost_round=8, quiet=True)
    for rng_ in [(0, 5), (2, 7), (3, 3)]:
        want = m.predict(x, iteration_range=rng_)
        np.testing.assert_array_equal(m.predict(str(pred_path), iteration_range=rng_), want)
        out = tmp_path / f"out_{rng_[0]}_{rng_[1]}.parquet"
        assert m.predict_to_parquet(str(pred_path), str(out), iteration_range=rng_) == len(x)
        got = pq.read_table(str(out)).column("prediction").to_numpy()
        np.testing.assert_array_equal(got, want)
