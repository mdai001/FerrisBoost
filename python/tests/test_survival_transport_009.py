"""AFT target-bound transport across arrays, files, Arrow and sklearn."""

import csv

import numpy as np
import pytest

import ferrisboost as fb


BASE = {
    "objective": "survival:aft",
    "max_depth": 2,
    "eta": 0.2,
    "max_bin": 16,
    "nthread": 1,
    "seed": 19,
    "aft_loss_distribution": "logistic",
    "aft_loss_distribution_scale": 0.8,
}


def _data(n=96, seed=17):
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 8, size=(n, 3)).astype(np.float32)
    lo = np.exp(0.13 * x[:, 0] - 0.08 * x[:, 1] + rng.normal(0, 0.3, n))
    lo = lo.astype(np.float32)
    hi = lo.copy()
    interval = np.arange(n) % 3 == 0
    censored = np.arange(n) % 3 == 1
    hi[interval] = lo[interval] * np.float32(1.6)
    hi[censored] = np.inf
    weight = np.linspace(0.5, 1.7, n, dtype=np.float32)
    margin = np.linspace(-0.2, 0.25, n, dtype=np.float32)
    return x, lo, hi, weight, margin


def _write_csv(path, x, lo, hi, weight, margin, *, header):
    names = ["f0", "f1", "f2", "lo", "hi", "weight", "offset"]
    rows = np.column_stack((x, lo, hi, weight, margin))
    if header:
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.writer(stream)
            writer.writerow(names)
            writer.writerows(rows)
    else:
        np.savetxt(path, rows, delimiter=",", fmt="%.17g")
    return path


@pytest.mark.parametrize("header", [True, False])
def test_csv_aft_bounds_eval_weights_and_margin_match_dense(tmp_path, header):
    x, lo, hi, weight, margin = _data()
    xv, lov, hiv, weightv, marginv = _data(n=47, seed=23)
    train_path = _write_csv(tmp_path / "train.csv", x, lo, hi, weight, margin, header=header)
    eval_path = _write_csv(tmp_path / "eval.csv", xv, lov, hiv, weightv, marginv, header=header)
    lower, upper = ("lo", "hi") if header else (3, 4)
    weight_col, margin_col = ("weight", "offset") if header else (5, 6)
    file_history, dense_history = {}, {}

    file_model = fb.train(
        BASE,
        train_path,
        # Natural file shorthand: `label` selects the lower-bound column.
        label=lower,
        label_upper_bound=upper,
        weight=weight_col,
        base_margin=margin_col,
        evals=[(eval_path, "valid")],
        num_boost_round=5,
        evals_result=file_history,
        quiet=True,
        header=header,
    )
    dense_model = fb.train(
        BASE,
        x,
        label=(lo, hi),
        weight=weight,
        base_margin=margin,
        evals=[(xv, (lov, hiv), weightv, "valid")],
        base_margin_eval_set=[marginv],
        num_boost_round=5,
        evals_result=dense_history,
        quiet=True,
    )

    assert file_model.n_features == x.shape[1]
    assert file_model.feature_names == (["f0", "f1", "f2"] if header else [])
    np.testing.assert_array_equal(
        file_model.predict(x, output_margin=True, base_margin=margin),
        dense_model.predict(x, output_margin=True, base_margin=margin),
    )
    np.testing.assert_allclose(
        file_history["valid"]["aft-nloglik"],
        dense_history["valid"]["aft-nloglik"],
        rtol=1e-6,
        atol=1e-7,
    )


def test_aft_upper_column_errors_and_selector_collisions(tmp_path):
    x, lo, hi, weight, margin = _data(n=40)
    path = _write_csv(tmp_path / "targets.csv", x, lo, hi, weight, margin, header=True)
    with pytest.raises(ValueError, match="column|upper|hi_missing"):
        fb.train(BASE, path, label="lo", label_upper_bound="hi_missing", quiet=True)
    with pytest.raises(ValueError, match="weight column"):
        fb.train(BASE, path, label="lo", label_upper_bound="weight", weight="weight", quiet=True)
    with pytest.raises(ValueError, match="base_margin column"):
        fb.train(BASE, path, label="lo", label_upper_bound="hi", base_margin="hi", quiet=True)
    with pytest.raises(ValueError, match="lower_bound|label"):
        fb.train(BASE, path, label="lo", label_lower_bound="lo", label_upper_bound="hi", quiet=True)


def test_aft_dense_native_rejects_eval_upper_row_mismatch():
    x, lo, hi, _, _ = _data(n=32)
    params = fb._normalize_params(BASE, 1)
    params["_quiet"] = True
    with pytest.raises(ValueError, match="eval_upper_bounds.*entries|eval_upper_bounds.*rows"):
        fb._core.train_dense(
            x,
            lo,
            params,
            evals=[(x, lo, None, "valid", None, None)],
            label_upper_bound=hi,
            eval_upper_bounds=[hi[:-1]],
        )


def test_aft_custom_metrics_are_rejected_for_interval_targets():
    x, lo, hi, _, _ = _data(n=32)
    metric = fb.CustomMetric("custom_interval_metric", lambda pred, label: float(np.mean(pred)))
    with pytest.raises(ValueError, match="interval-aware|survival:aft|aft"):
        fb.train(
            {**BASE, "eval_metric": metric},
            x,
            label=(lo, hi),
            evals=[(x, (lo, hi), "valid")],
            num_boost_round=1,
            quiet=True,
        )


def test_aft_nondefault_distribution_continues_and_rejects_changed_params(tmp_path):
    x, lo, hi, _, _ = _data(n=72)
    first = fb.train(BASE, x, label=(lo, hi), num_boost_round=2, quiet=True)
    path = tmp_path / "aft.json"
    first.save_model(str(path), quiet=True)
    loaded = fb.Model.load_model(str(path), quiet=True)
    continued = fb.train(BASE, x, label=(lo, hi), init_model=loaded, num_boost_round=2, quiet=True)
    assert continued.num_trees == first.num_trees + 2
    assert np.all(np.isfinite(continued.predict(x)))
    with pytest.raises(ValueError, match="objective|continue|different|AFT|aft"):
        fb.train(
            {**BASE, "aft_loss_distribution_scale": 1.3},
            x,
            label=(lo, hi),
            init_model=loaded,
            num_boost_round=1,
            quiet=True,
        )


@pytest.mark.parametrize("distribution", ["normal", "logistic", "extreme"])
def test_sklearn_keeps_infinite_upper_bounds_and_weighted_eval_targets(
    tmp_path, distribution
):
    pytest.importorskip("sklearn")
    from ferrisboost.sklearn import FerrisBoostRegressor

    x, lo, hi, weight, _ = _data(n=72)
    xv, lov, hiv, weightv, _ = _data(n=36, seed=29)
    assert np.isinf(hi).any() and np.isinf(hiv).any()
    train_y = np.column_stack((lo, hi)).astype(np.float32)
    eval_y = np.column_stack((lov, hiv)).astype(np.float32)
    model = FerrisBoostRegressor(
        objective="survival:aft",
        aft_loss_distribution=distribution,
        aft_loss_distribution_scale=0.8,
        n_estimators=3,
        max_depth=2,
        n_jobs=1,
        random_state=19,
        eval_metric="aft-nloglik",
    ).fit(
        x,
        train_y,
        sample_weight=weight,
        eval_set=[(xv, eval_y)],
        sample_weight_eval_set=[weightv],
    )
    assert model.evals_result_["validation_0"]["aft-nloglik"]
    assert np.all(np.isfinite(model.predict(x)))
    expected = -model.evals_result_["validation_0"]["aft-nloglik"][-1]
    assert model.score(xv, eval_y, sample_weight=weightv) == pytest.approx(expected, rel=1e-6)

    # The loaded model's native scorer must use the distribution and scale
    # stored in that model, even though the new estimator has no such params.
    model_path = tmp_path / f"aft-{distribution}.json"
    model.save_model(str(model_path))
    loaded = FerrisBoostRegressor().load_model(model_path)
    assert loaded.score(xv, eval_y, sample_weight=weightv) == pytest.approx(expected, rel=1e-6)


def test_sklearn_cox_score_is_negative_weighted_eval_nll():
    pytest.importorskip("sklearn")
    from ferrisboost.sklearn import FerrisBoostRegressor

    x, lo, _, weight, _ = _data(n=72)
    xv, lov, _, weightv, _ = _data(n=36, seed=29)
    y = np.where(np.arange(len(lo)) % 3 == 0, lo, -lo).astype(np.float32)
    yv = np.where(np.arange(len(lov)) % 3 == 0, lov, -lov).astype(np.float32)
    model = FerrisBoostRegressor(
        objective="survival:cox", n_estimators=3, max_depth=2, n_jobs=1,
        random_state=19, eval_metric="cox-nloglik",
    ).fit(
        x, y, sample_weight=weight, eval_set=[(xv, yv)],
        sample_weight_eval_set=[weightv],
    )
    expected = -model.evals_result_["validation_0"]["cox-nloglik"][-1]
    assert model.score(xv, yv, sample_weight=weightv) == pytest.approx(expected, rel=1e-6)


def test_aft_parquet_cache_reuses_then_invalidates_when_upper_bounds_change(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    x, lo, hi_a, weight, margin = _data(n=80)
    path = tmp_path / "aft.parquet"
    def write(upper):
        pq.write_table(pa.table({
            "f0": pa.array(x[:, 0]),
            "f1": pa.array(x[:, 1]),
            "f2": pa.array(x[:, 2]),
            "lo": pa.array(lo),
            "hi": pa.array(upper),
            "weight": pa.array(weight),
            "offset": pa.array(margin),
        }), path)

    write(hi_a)
    cache = tmp_path / "aft-cache"
    params = {**BASE, "cache_path": cache}
    history_a, history_replay = {}, {}
    first = fb.train(
        params, path, label="lo", label_upper_bound="hi", weight="weight",
        base_margin="offset", evals=[(path, "valid")], num_boost_round=4,
        evals_result=history_a, quiet=True,
    )
    replay = fb.train(
        params, path, label="lo", label_upper_bound="hi", weight="weight",
        base_margin="offset", evals=[(path, "valid")], num_boost_round=4,
        evals_result=history_replay, quiet=True,
    )
    np.testing.assert_array_equal(first.predict(x), replay.predict(x))
    assert history_a == history_replay

    # Keep every feature byte identical and change only the target upper bounds.
    # The source identity changes, so stale cache reuse must be rejected.
    hi_b = lo * np.float32(1.8)
    hi_b[np.arange(len(lo)) % 4 == 2] = np.inf
    write(hi_b)
    with pytest.raises(ValueError, match="cache.*changed|cache.*current file"):
        fb.train(
            params, path, label="lo", label_upper_bound="hi", weight="weight",
            base_margin="offset", evals=[(path, "valid")], num_boost_round=4,
            quiet=True,
        )
    history_b = {}
    second = fb.train(
        {**BASE, "cache_path": tmp_path / "aft-cache-updated"}, path,
        label="lo", label_upper_bound="hi", weight="weight", base_margin="offset",
        evals=[(path, "valid")], num_boost_round=4, evals_result=history_b, quiet=True,
    )
    assert first.n_features == second.n_features == x.shape[1]
    assert first.feature_names == second.feature_names == ["f0", "f1", "f2"]
    assert not np.array_equal(
        first.predict(x, output_margin=True, base_margin=margin),
        second.predict(x, output_margin=True, base_margin=margin),
    )
    assert history_a["valid"]["aft-nloglik"] != history_b["valid"]["aft-nloglik"]


def test_aft_accepts_arrow_table_and_parquet_eval_files(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    x, lo, hi, weight, margin = _data(n=64)
    xv, lov, hiv, weightv, marginv = _data(n=31, seed=31)
    training = pa.table({
        **{f"f{i}": pa.array(x[:, i]) for i in range(x.shape[1])},
    })
    reference = fb.train(BASE, x, label=(lo, hi), num_boost_round=3, quiet=True)
    arrow = fb.train(
        BASE, training, label_lower_bound=lo, label_upper_bound=hi,
        num_boost_round=3, quiet=True,
    )
    np.testing.assert_array_equal(reference.predict(x), arrow.predict(x))
    assert arrow.feature_names == ["f0", "f1", "f2"]

    # Write the file pair separately so the file reader exercises per-eval
    # upper-bound reads while reusing the training label/weight/margin columns.
    train_path = tmp_path / "train.parquet"
    eval_path = tmp_path / "eval.parquet"
    pq.write_table(pa.table({
        **{f"f{i}": pa.array(x[:, i]) for i in range(x.shape[1])},
        "lo": pa.array(lo), "hi": pa.array(hi),
        "weight": pa.array(weight), "offset": pa.array(margin),
    }), train_path)
    pq.write_table(pa.table({
        **{f"f{i}": pa.array(xv[:, i]) for i in range(xv.shape[1])},
        "lo": pa.array(lov), "hi": pa.array(hiv),
        "weight": pa.array(weightv), "offset": pa.array(marginv),
    }), eval_path)
    history = {}
    file_model = fb.train(
        BASE, train_path, label="lo", label_upper_bound="hi", weight="weight",
        base_margin="offset", evals=[(eval_path, "valid")], num_boost_round=3,
        evals_result=history, quiet=True,
    )
    assert file_model.n_features == x.shape[1]
    assert file_model.feature_names == ["f0", "f1", "f2"]
    assert np.isfinite(history["valid"]["aft-nloglik"]).all()
