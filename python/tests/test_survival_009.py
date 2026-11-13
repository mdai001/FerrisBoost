"""Cox and accelerated-failure-time survival objectives."""
import csv
import json
import math
import os

import numpy as np
import pytest

import ferrisboost as fb


BASE = {"max_depth": 2, "eta": 0.2, "max_bin": 16, "nthread": 1, "seed": 11}


def _data(n=144, seed=4):
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 9, size=(n, 3)).astype(np.float32)
    score = 0.25 * x[:, 0] - 0.15 * x[:, 1] + 0.1 * x[:, 2]
    time = np.exp(0.2 * score + rng.normal(0.0, 0.45, n)).astype(np.float32)
    event = (rng.random(n) < 0.72)
    cox_label = np.where(event, time, -time).astype(np.float32)

    lower = time.copy()
    upper = time.copy()
    interval = np.arange(n) % 4 == 1
    right = np.arange(n) % 4 == 2
    left = np.arange(n) % 4 == 3
    lower[interval] = time[interval] * 0.7
    upper[interval] = time[interval] * 1.25
    upper[right] = np.inf
    lower[left] = 0.0
    return x, cox_label, lower.astype(np.float32), upper.astype(np.float32)


def _weights(n):
    return np.linspace(0.4, 1.8, n, dtype=np.float32)


def _params(objective, distribution="normal", scale=1.1, **extra):
    params = {**BASE, "objective": objective, **extra}
    if objective == "survival:aft":
        params.update(aft_loss_distribution=distribution,
                      aft_loss_distribution_scale=scale)
    return params


def _raw_base_score(path):
    value = json.loads(path.read_text(encoding="utf-8"))["learner"][
        "learner_model_param"]["base_score"]
    if isinstance(value, list):
        return float(value[0])
    return float(str(value).strip().strip("[]").split(",")[0])


def _cox_nll(margin, label, weight):
    times = np.abs(label.astype(np.float64))
    event = label > 0.0
    unique_event_times = np.unique(times[event])
    total_events = np.sum(weight[event], dtype=np.float64)
    if total_events <= 0.0:
        return 0.0
    loss = 0.0
    risk_score = np.exp(margin.astype(np.float64))
    for time in unique_event_times:
        at_time = event & (times == time)
        risk = times >= time
        event_mass = np.sum(weight[at_time], dtype=np.float64)
        denom = np.sum(weight[risk].astype(np.float64) * risk_score[risk])
        loss += event_mass * math.log(denom)
        loss -= np.sum(weight[at_time].astype(np.float64) * margin[at_time], dtype=np.float64)
    return loss / total_events


def _cdf(distribution, z):
    if distribution == "normal":
        return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
    if distribution == "logistic":
        return 1.0 / (1.0 + math.exp(-z))
    return 1.0 - math.exp(-math.exp(z))


def _pdf(distribution, z):
    if distribution == "normal":
        return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    if distribution == "logistic":
        cdf = _cdf(distribution, z)
        return cdf * (1.0 - cdf)
    return math.exp(z - math.exp(z))


def _aft_loss(distribution, scale, margin, lower, upper):
    loss = np.empty(len(margin), dtype=np.float64)
    for row, (m, lo, hi) in enumerate(zip(margin, lower, upper)):
        if lo == hi:
            z = (math.log(float(lo)) - float(m)) / scale
            loss[row] = -math.log(_pdf(distribution, z) / (scale * float(lo)))
        else:
            cdf_hi = 1.0 if np.isinf(hi) else _cdf(
                distribution, (math.log(float(hi)) - float(m)) / scale)
            cdf_lo = 0.0 if lo == 0.0 else _cdf(
                distribution, (math.log(float(lo)) - float(m)) / scale)
            loss[row] = -math.log(max(cdf_hi - cdf_lo, 1e-12))
    return loss


def test_cox_weighted_breslow_metric_and_log_link(tmp_path):
    x, label, _, _ = _data()
    weight = _weights(len(label))
    result = {}
    model = fb.train(_params("survival:cox"), x, label=label, weight=weight,
                     num_boost_round=4, evals=[(x, label, weight, "train")],
                     evals_result=result, quiet=True)
    margin = model.predict(x, output_margin=True).astype(np.float64)
    risk = model.predict(x).astype(np.float64)
    np.testing.assert_allclose(risk, np.exp(margin), rtol=2e-7)
    assert list(result["train"]) == ["cox-nloglik"]
    expected = _cox_nll(margin, label, weight)
    assert result["train"]["cox-nloglik"][-1] == pytest.approx(expected, rel=1e-6)

    path = tmp_path / "cox-base.json"
    model.save_model(str(path), quiet=True)
    assert _raw_base_score(path) == pytest.approx(1.0)


@pytest.mark.parametrize("distribution", ["normal", "logistic", "extreme"])
def test_aft_bounds_default_metric_and_log_time_link(tmp_path, distribution):
    x, _, lower, upper = _data()
    result = {}
    scale = 0.9
    model = fb.train(_params("survival:aft", distribution, scale), x,
                     label=(lower, upper), num_boost_round=3,
                     evals=[(x, (lower, upper), "valid")], evals_result=result, quiet=True)
    margin = model.predict(x, output_margin=True).astype(np.float64)
    predicted_time = model.predict(x).astype(np.float64)
    np.testing.assert_allclose(predicted_time, np.exp(margin), rtol=2e-7)
    assert np.all(predicted_time > 0.0)
    assert list(result["valid"]) == ["aft-nloglik"]
    loss = _aft_loss(distribution, scale, margin, lower, upper)
    expected = np.mean(loss)
    assert result["valid"]["aft-nloglik"][-1] == pytest.approx(expected, rel=2e-5)
    path = tmp_path / f"aft-{distribution}.json"
    model.save_model(str(path), quiet=True)
    assert _raw_base_score(path) == pytest.approx(0.5)


def test_aft_bounds_keyword_and_tuple_forms_and_named_file_exclude_label_columns(tmp_path):
    x, _, lower, upper = _data(n=64)
    p = _params("survival:aft", "logistic", 1.2)
    tuple_model = fb.train(p, x, label=(lower, upper), num_boost_round=2, quiet=True)
    keyword_model = fb.train(p, x, label_lower_bound=lower, label_upper_bound=upper,
                             num_boost_round=2, quiet=True)
    np.testing.assert_array_equal(tuple_model.predict(x), keyword_model.predict(x))

    csv_path = tmp_path / "aft-named.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["f0", "f1", "f2", "lower", "upper"])
        writer.writerows(np.column_stack((x, lower, upper)))
    file_model = fb.train(p, csv_path, label_lower_bound="lower",
                          label_upper_bound="upper", num_boost_round=2, quiet=True)
    assert file_model.n_features == x.shape[1]
    assert file_model.feature_names == ["f0", "f1", "f2"]
    assert len(file_model.predict(x)) == len(x)


@pytest.mark.parametrize("lower,upper", [
    ([-0.1], [1.0]),
    ([np.nan], [1.0]),
    ([0.0], [0.0]),
    ([2.0], [1.0]),
    ([0.0], [-np.inf]),
])
def test_aft_rejects_invalid_bounds(lower, upper):
    x = np.zeros((1, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="bound|lower|upper|survival|label"):
        fb.train(_params("survival:aft"), x,
                 label_lower_bound=np.asarray(lower, dtype=np.float32),
                 label_upper_bound=np.asarray(upper, dtype=np.float32),
                 num_boost_round=1, quiet=True)


def test_aft_scale_must_be_positive_and_distribution_must_be_known():
    x, _, lower, upper = _data(n=32)
    for scale in (0.0, -1.0, np.nan, np.inf):
        with pytest.raises(ValueError, match="scale|distribution"):
            fb.train(_params("survival:aft", scale=scale), x, label=(lower, upper),
                     num_boost_round=1, quiet=True)
    with pytest.raises(ValueError, match="distribution"):
        fb.train(_params("survival:aft", distribution="unknown"), x,
                 label=(lower, upper), num_boost_round=1, quiet=True)


@pytest.mark.parametrize("objective", ["survival:cox", "survival:aft"])
def test_survival_save_load_and_continuation(tmp_path, objective):
    x, cox_label, lower, upper = _data()
    weight = _weights(len(x))
    p = _params(objective)
    label = cox_label if objective == "survival:cox" else (lower, upper)
    uninterrupted = fb.train(p, x, label=label, weight=weight,
                             num_boost_round=4, quiet=True)
    first = fb.train(p, x, label=label, weight=weight, num_boost_round=2, quiet=True)
    path = tmp_path / "first.json"
    first.save_model(str(path), quiet=True)
    assert json.loads(path.read_text(encoding="utf-8"))["learner"]["objective"]["name"] \
        == objective
    loaded = fb.Model.load_model(str(path), quiet=True)
    continued = fb.train(p, x, label=label, weight=weight, num_boost_round=2,
                         init_model=loaded, quiet=True)
    np.testing.assert_array_equal(continued.predict(x, output_margin=True),
                                  uninterrupted.predict(x, output_margin=True))


def test_sklearn_regressor_accepts_two_column_aft_targets():
    pytest.importorskip("sklearn")
    from ferrisboost.sklearn import FerrisBoostRegressor

    x, _, lower, upper = _data(n=96)
    upper[np.isinf(upper)] = lower[np.isinf(upper)] * 1.4
    target = np.column_stack((lower, upper)).astype(np.float32)
    reg = FerrisBoostRegressor(objective="survival:aft", aft_loss_distribution="normal",
                               aft_loss_distribution_scale=1.1, n_estimators=3,
                               max_depth=2, n_jobs=1, random_state=11).fit(x, target)
    pred = reg.predict(x)
    assert pred.shape == (len(x),)
    assert np.all(np.isfinite(pred))
    assert np.all(pred > 0.0)


cuda = pytest.mark.skipif(
    os.environ.get("FB_TEST_CUDA") != "1", reason="requires explicit CUDA test opt-in"
)


@cuda
@pytest.mark.parametrize("objective", ["survival:cox", "survival:aft"])
def test_survival_exact_cuda_matches_cpu_and_fast_is_rejected(tmp_path, objective):
    x, cox_label, lower, upper = _data(n=96)
    weight = _weights(len(x))
    label = cox_label if objective == "survival:cox" else (lower, upper)
    p = _params(objective)
    cpu = fb.train(p, x, label=label, weight=weight, num_boost_round=3, quiet=True)
    exact = fb.train({**p, "device": "cuda", "gpu_math": "exact"}, x,
                     label=label, weight=weight, num_boost_round=3, quiet=True)
    np.testing.assert_array_equal(cpu.predict(x), exact.predict(x))
    cpu_path, exact_path = tmp_path / "cpu.json", tmp_path / "exact.json"
    cpu.save_model(str(cpu_path), quiet=True)
    exact.save_model(str(exact_path), quiet=True)
    assert cpu_path.read_bytes() == exact_path.read_bytes()
    with pytest.raises((ValueError, RuntimeError), match="survival|fast|gpu_math"):
        fb.train({**p, "device": "cuda", "gpu_math": "fast"}, x,
                 label=label, weight=weight, num_boost_round=1, quiet=True)
