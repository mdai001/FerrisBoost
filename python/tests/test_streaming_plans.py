"""Streaming training beyond GPU memory (release gate, run every release).

Full-resident, hybrid (1..t-1 resident blocks) and pure-streaming plans must train
byte-identical `gpu_math="exact"` models that also match the CPU, for every
objective family and the features that touch the histogram / partition / split
paths (column sampling at all three levels, categorical splits, weights,
base_margin, init_model continuation, eval sets with early stopping, missing
values). Narrow and wide column-major blocks are covered, each with a last block
narrower than the others. `gpu_math="fast"` must be deterministic per plan; it is
also checked to be plan-invariant (the plan only changes where the quantized bins
live, not the arithmetic). Every GPU run confirms the executed plan from the runtime `GPU_PATH`
echo, never from the requested configuration.

Budget-driven planning (no resident-block override) and a forced host-cache spill
are covered too. Larger datasets (large column-major upload, auto block width,
observed device memory against the planner estimate) run with FB_SLOW_TESTS=1.
"""
import os
import re
import subprocess
import threading
import time

import numpy as np
import pytest

import ferrisboost as fb

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

SLOW = os.environ.get("FB_SLOW_TESTS") == "1"
MB = 1024 * 1024


def _has_cuda():
    x = np.zeros((64, 2), dtype=np.float32)
    y = np.zeros(64, dtype=np.float32)
    try:
        fb.train({"device": "cuda", "max_depth": 2, "objective": "reg:squarederror"}, x, y,
                 num_boost_round=1, quiet=True)
        return True
    except Exception as e:  # noqa: BLE001
        if "CUDA" in str(e) or "cuda" in str(e):
            return False
        raise


cuda = pytest.mark.skipif(not _has_cuda(), reason="no usable CUDA build or device")
slow = pytest.mark.skipif(not SLOW, reason="set FB_SLOW_TESTS=1 for the large streaming cases")

# Physical geometry, forced with cols_per_block. Both have a narrower last block.
LAYOUTS = {
    # 3, 3, 3, 2 -> four narrow column-major blocks
    "narrow": {"n_feats": 11, "cols_per_block": 3, "blocks": 4},
    # 33, 33, 4 -> three wide column-major blocks
    "wide": {"n_feats": 70, "cols_per_block": 33, "blocks": 3},
}
CAT_FEATURES = (1, 3)  # categorical columns when a case asks for them

PATH_RE = re.compile(
    r"layout_gpu=(\w+).*?residency=(\w+) resident_blocks=(\d+)/(\d+) streamed_blocks=(\d+)")
STREAMS_RE = re.compile(r"hist_streams=(\d+)")


def _write(tmp_path, name, objective, n, n_feats, seed, n_cat=0, weights=False,
           base_margin=False, num_class=4):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, n_feats)).astype(np.float32)
    cats = CAT_FEATURES[:n_cat]
    for c, k in zip(cats, (12, 40)):
        x[:, c] = rng.integers(0, k, n).astype(np.float32)
    x[rng.random(x.shape) < 0.03] = np.nan
    xs = np.nan_to_num(x)
    s = xs[:, 0] - 0.7 * xs[:, 5] + 0.4 * xs[:, n_feats - 1]
    for c in cats:
        s += 0.5 * np.sin(1.7 * xs[:, c])
    cols = {f"f{i}": pa.array(x[:, i]) for i in range(n_feats)}
    ranking = objective.startswith("rank:")
    if ranking:
        qid = np.repeat(np.arange(n // 20 + 1), 20)[:n].astype(np.int64)
        cols["qid"] = pa.array(qid)
    if objective == "binary:logistic":
        y = (s + rng.normal(0, 0.5, n) > 0).astype(np.float32)
    elif objective == "multi:softprob":
        y = np.digitize(s, np.quantile(s, [0.25, 0.5, 0.75])).astype(np.float32)
    elif objective == "rank:map":
        y = (s + rng.normal(0, 0.5, n) > 0.5).astype(np.float32)
    elif ranking:
        y = np.clip(np.round(s + 1.5), 0, 4).astype(np.float32)
    elif objective == "count:poisson":
        y = rng.poisson(np.exp(np.clip(0.4 * s, -3, 3))).astype(np.float32)
    else:
        y = (s + rng.normal(0, 0.3, n)).astype(np.float32)
    cols["label"] = pa.array(y)
    if weights:
        if ranking:  # ranking weights are per query group
            cols["w"] = pa.array(rng.uniform(0.5, 2.0, qid.max() + 1).astype(np.float32)[qid])
        else:
            cols["w"] = pa.array(rng.uniform(0.2, 3.0, n).astype(np.float32))
    if base_margin:
        k = num_class if objective == "multi:softprob" else 1
        for j in range(k):
            cols[f"bm{j}"] = pa.array(rng.normal(0, 0.3, n).astype(np.float32))
    path = str(tmp_path / f"{name}.parquet")
    pq.write_table(pa.table(cols), path, row_group_size=1000)
    return path


def _bytes(model, tmp_path, name):
    p = tmp_path / f"{name}.json"
    model.save_model(str(p), quiet=True)
    return p.read_bytes()


def _setup(tmp_path, objective, layout, n=4000, seed=0, sampling=None, n_cat=0,
           weights=False, base_margin=False, evals=False, init_model=False, rounds=4,
           max_depth=5, **extra):
    """Data, params and train kwargs for one matrix case (CPU params; add the GPU keys)."""
    lay = LAYOUTS[layout]
    unbiased = objective.endswith("+unbiased")
    objective = objective.split("+")[0]
    tag = f"{objective.replace(':', '_')}_{layout}_{seed}"
    path = _write(tmp_path, tag, objective, n, lay["n_feats"], seed, n_cat, weights, base_margin)
    p = {"objective": objective, "max_depth": max_depth, "eta": 0.3, "nthread": 2,
         "cols_per_block": lay["cols_per_block"], "seed": 7, "max_bin": 64,
         **(sampling or {}), **extra}
    if objective == "multi:softprob":
        p["num_class"] = 4
    if objective == "reg:quantileerror":
        p["quantile_alpha"] = [0.2, 0.5, 0.8]
    if unbiased:
        p["lambdarank_unbiased"] = True
    kw = {"label": "label", "num_boost_round": rounds}
    if n_cat:
        kw["feature_types"] = ["c" if i in CAT_FEATURES[:n_cat] else "q"
                               for i in range(lay["n_feats"])]
    if objective.startswith("rank:"):
        kw["qid"] = "qid"
    if weights:
        kw["weight"] = "w"
    if base_margin:
        kw["base_margin"] = (
            [f"bm{j}" for j in range(4)] if objective == "multi:softprob" else "bm0")
    if evals:
        vpath = _write(tmp_path, tag + "_val", objective, n // 2, lay["n_feats"], seed + 1,
                       n_cat, weights, base_margin)
        kw.update(evals=[(vpath, "val")], early_stopping_rounds=2, num_boost_round=12)
    if init_model:
        prior_kw = {**kw, "num_boost_round": 2}
        prior_kw.pop("early_stopping_rounds", None)
        kw["init_model"] = fb.train(p, path, quiet=True, **prior_kw)
    return path, p, kw, lay["blocks"]


def _gpu(capfd, p, path, kw, **gpu):
    """Train on the GPU and return (model, executed plan) from the GPU_PATH echo."""
    capfd.readouterr()
    model = fb.train({**p, "device": "cuda", **gpu}, path, **kw)
    log = capfd.readouterr().err
    m = PATH_RE.search(log)
    assert m, log[-2000:]
    s = STREAMS_RE.search(log)
    return model, {"layout": m.group(1), "residency": m.group(2), "resident": int(m.group(3)),
                   "total": int(m.group(4)), "streamed": int(m.group(5)),
                   "streams": int(s.group(1)), "log": log}


def _expected_label(resident, total):
    return ("pure_streaming" if resident == 0
            else "full_resident" if resident >= total else "hybrid_partial_resident")


def _plans(total):
    """Full residency, every hybrid split 1..t-1, and pure streaming."""
    return list(range(total, -1, -1))


def _check_plan(info, layout, resident, total):
    assert info["layout"] == "col_major", info["log"][-1500:]
    assert info["total"] == total
    assert info["resident"] == resident
    assert info["streamed"] == total - resident
    assert info["residency"] == _expected_label(resident, total)


def _assert_exact_matrix(tmp_path, capfd, path, p, kw, layout, total, streams=(None,)):
    ref = _bytes(fb.train(p, path, quiet=True, **kw), tmp_path, "cpu")
    for resident in _plans(total):
        for hs in streams:
            gpu = {"gpu_math": "exact", "resident_blocks": resident}
            if hs is not None:
                gpu["hist_streams"] = hs
            model, info = _gpu(capfd, p, path, kw, **gpu)
            _check_plan(info, layout, resident, total)
            if hs is not None:
                assert info["streams"] == hs
            assert _bytes(model, tmp_path, "gpu") == ref, (
                f"{info['residency']} {resident}/{total} hist_streams={info['streams']} "
                "differs from the CPU model")


SAMPLE_ALL = {"subsample": 0.8, "colsample_bytree": 0.7, "colsample_bylevel": 0.8,
              "colsample_bynode": 0.7}

OBJECTIVES = ["binary:logistic", "reg:squarederror", "count:poisson", "multi:softprob",
              "reg:quantileerror", "rank:ndcg", "rank:pairwise", "rank:map",
              "rank:ndcg+unbiased"]


@cuda
@pytest.mark.parametrize("layout", list(LAYOUTS))
@pytest.mark.parametrize("objective", OBJECTIVES)
def test_every_plan_trains_the_cpu_model_for_every_objective(tmp_path, capfd, objective, layout):
    path, p, kw, total = _setup(tmp_path, objective, layout, sampling=SAMPLE_ALL, n_cat=1)
    _assert_exact_matrix(tmp_path, capfd, path, p, kw, layout, total)


FEATURES = {
    "categorical_nested_colsample": dict(
        n_cat=2, sampling={"colsample_bytree": 0.8, "colsample_bylevel": 0.7,
                           "colsample_bynode": 0.6}),
    "weights_base_margin_subsample": dict(weights=True, base_margin=True,
                                          sampling={"subsample": 0.7}),
    "eval_early_stopping": dict(evals=True, sampling={"colsample_bytree": 0.6}),
    "init_model": dict(init_model=True, sampling={"colsample_bytree": 0.6}),
    "everything": dict(n_cat=2, weights=True, evals=True, init_model=True,
                       sampling=SAMPLE_ALL),
    # deeper trees: more than 16 nodes per level (batched launches, split-cache
    # overflow) and more than 32 partitions per resident block
    "deep_trees": dict(n=12000, max_depth=8, n_cat=1, sampling={"subsample": 0.7}),
    "regularized_monotone": dict(
        sampling={"colsample_bytree": 0.8}, reg_lambda=3.0, reg_alpha=0.5, gamma=0.2,
        min_child_weight=5, max_delta_step=0.7),
}


@cuda
@pytest.mark.parametrize("layout", list(LAYOUTS))
@pytest.mark.parametrize("objective", ["binary:logistic", "multi:softprob", "rank:ndcg"])
@pytest.mark.parametrize("feature", list(FEATURES))
def test_every_plan_trains_the_cpu_model_with_training_features(
        tmp_path, capfd, feature, objective, layout):
    opts = dict(FEATURES[feature])
    if feature == "regularized_monotone":
        nf = LAYOUTS[layout]["n_feats"]
        opts["monotone_constraints"] = tuple([1, 0, -1] + [0] * (nf - 4) + [1])
    path, p, kw, total = _setup(tmp_path, objective, layout, **opts)
    _assert_exact_matrix(tmp_path, capfd, path, p, kw, layout, total)


@cuda
@pytest.mark.parametrize("layout", list(LAYOUTS))
@pytest.mark.parametrize("objective", ["binary:logistic", "reg:quantileerror"])
def test_one_and_two_histogram_streams_train_the_cpu_model(tmp_path, capfd, objective, layout):
    path, p, kw, total = _setup(tmp_path, objective, layout, sampling=SAMPLE_ALL, n_cat=1)
    _assert_exact_matrix(tmp_path, capfd, path, p, kw, layout, total, streams=(1, 2))


@cuda
@pytest.mark.parametrize("layout", list(LAYOUTS))
@pytest.mark.parametrize("objective", ["binary:logistic", "rank:ndcg", "multi:softprob"])
def test_forced_host_spill_with_gpu_streaming_trains_the_cpu_model(
        tmp_path, capfd, objective, layout):
    # A spilled cache can hand compact (column-selected) blocks to the streamed part.
    path, p, kw, total = _setup(tmp_path, objective, layout, sampling=SAMPLE_ALL, n_cat=1)
    ref = _bytes(fb.train(p, path, quiet=True, **kw), tmp_path, "cpu")
    spill = {**p, "cache_budget_bytes": 0}
    assert _bytes(fb.train(spill, path, quiet=True, **kw), tmp_path, "cpu_spill") == ref
    _assert_exact_matrix(tmp_path, capfd, path, spill, kw, layout, total)


FAST_CASES = [
    ("binary:logistic", dict(sampling=SAMPLE_ALL, n_cat=1, weights=True, base_margin=True)),
    ("multi:softprob", dict(sampling={"subsample": 0.8, "colsample_bytree": 0.7},
                            evals=True, init_model=True)),
    ("rank:ndcg+unbiased", dict(sampling=SAMPLE_ALL)),
    ("reg:quantileerror", dict(sampling=SAMPLE_ALL, weights=True, init_model=True)),
    ("count:poisson", dict(sampling={"colsample_bynode": 0.7})),
]


@cuda
@pytest.mark.parametrize("layout", list(LAYOUTS))
@pytest.mark.parametrize("objective,opts", FAST_CASES, ids=[c[0] for c in FAST_CASES])
def test_fast_math_is_deterministic_and_plan_invariant(tmp_path, capfd, objective, opts, layout):
    path, p, kw, total = _setup(tmp_path, objective, layout, **opts)
    seen = {}
    for resident in _plans(total):
        runs = []
        for _ in range(2):
            model, info = _gpu(capfd, p, path, kw, gpu_math="fast", resident_blocks=resident)
            _check_plan(info, layout, resident, total)
            runs.append(_bytes(model, tmp_path, "fast"))
        assert runs[0] == runs[1], f"fast is not deterministic under {info['residency']}"
        seen[resident] = runs[0]
    # The plan decides where the bins live, not the arithmetic: fast models agree too.
    assert len(set(seen.values())) == 1, "fast models differ between residency plans"


@cuda
@pytest.mark.parametrize("layout", list(LAYOUTS))
def test_budget_driven_plans_without_override_train_the_cpu_model(tmp_path, capfd, layout):
    """No resident-block override: the planner alone picks the plan from the budget.

    The budgets are derived from the planner's own estimates for each forced plan
    (block width and stream count are pinned so only residency is left to plan),
    so the case does not depend on the device or its free memory.
    """
    n = 200_000 if layout == "narrow" else 80_000
    path, p, kw, total = _setup(tmp_path, "binary:logistic", layout, n=n, rounds=3,
                                sampling={"subsample": 0.8, "colsample_bytree": 0.8}, n_cat=1)
    p = {**p, "hist_streams": 1}
    ref = _bytes(fb.train(p, path, quiet=True, **kw), tmp_path, "cpu")
    estimate = {}
    for resident in (total, 1, 0):
        _, info = _gpu(capfd, p, path, kw, resident_blocks=resident)
        estimate[resident] = float(re.search(r"estimated_peak:\s+([\d.]+) MB",
                                             info["log"]).group(1))
    assert estimate[0] < estimate[1] < estimate[total]
    reserve = 512.0  # the planner's minimum safety reserve, below 3.4 GB budgets
    for resident in (total, 1, 0):
        budget = int((reserve + estimate[resident] + 0.25) * MB)
        model, info = _gpu(capfd, p, path, kw, gpu_memory_budget=budget)
        _check_plan(info, layout, resident, total)
        assert _bytes(model, tmp_path, "gpu") == ref, f"budget {budget} ({info['residency']})"
    with pytest.raises((ValueError, RuntimeError), match="does not fit"):
        fb.train({**p, "device": "cuda", "gpu_memory_budget": int((reserve + estimate[0] / 2) * MB)},
                 path, quiet=True, **kw)



@cuda
def test_override_over_a_set_budget_raises_before_training(tmp_path, capfd):
    """A user-set gpu_memory_budget is a limit: an explicit resident_blocks that
    needs more memory fails up front; the same override without a budget trains."""
    path, p, kw, total = _setup(tmp_path, "binary:logistic", "narrow", n=200_000, rounds=2)
    p = {**p, "hist_streams": 1}
    estimate = {}
    for resident in (total, 0):
        _, info = _gpu(capfd, p, path, kw, resident_blocks=resident)
        estimate[resident] = float(re.search(r"estimated_peak:\s+([\d.]+) MB",
                                             info["log"]).group(1))
    assert estimate[0] < estimate[total]
    reserve = 512.0
    budget = int((reserve + (estimate[0] + estimate[total]) / 2) * MB)  # fits streaming only
    with pytest.raises(ValueError, match=r"resident_blocks=\d+ training needs about"):
        fb.train({**p, "device": "cuda", "gpu_memory_budget": budget,
                  "resident_blocks": total}, path, quiet=True, **kw)
    # The budget alone plans a plan that fits.
    _, info = _gpu(capfd, p, path, kw, gpu_memory_budget=budget)
    assert info["residency"] != "full_resident"

@cuda
def test_csv_and_multi_file_parquet_stream_like_a_single_parquet(tmp_path, capfd):
    path, p, kw, total = _setup(tmp_path, "binary:logistic", "narrow",
                                sampling={"subsample": 0.8, "colsample_bytree": 0.7})
    ref = _bytes(fb.train(p, path, quiet=True, **kw), tmp_path, "cpu")
    table = pq.read_table(path)
    parts = tmp_path / "parts"
    parts.mkdir()
    step = table.num_rows // 3 + 1
    for i in range(3):
        pq.write_table(table.slice(i * step, step), str(parts / f"part{i}.parquet"))
    csv = str(tmp_path / "data.csv")
    table.to_pandas().to_csv(csv, index=False, na_rep="")
    for src in (str(parts), csv):
        assert _bytes(fb.train(p, src, quiet=True, **kw), tmp_path, "cpu_src") == ref
        for resident in (total, 1, 0):
            for spill in ({}, {"cache_budget_bytes": 0}):
                model, info = _gpu(capfd, {**p, **spill}, src, kw, resident_blocks=resident)
                _check_plan(info, "narrow", resident, total)
                assert _bytes(model, tmp_path, "gpu") == ref, (src, resident, spill)



@cuda
def test_gpu_csv_without_cols_per_block_uses_a_default_width(tmp_path, capfd):
    """CSV has no row count before open: without cols_per_block the GPU path uses
    a conservative default width (announced in the log) instead of failing, and
    the model still equals the CPU model."""
    path, p, kw, _ = _setup(tmp_path, "binary:logistic", "narrow")
    p = {k: v for k, v in p.items() if k != "cols_per_block"}
    csv = str(tmp_path / "data.csv")
    pq.read_table(path).to_pandas().to_csv(csv, index=False, na_rep="")
    ref = _bytes(fb.train(p, csv, quiet=True, **kw), tmp_path, "cpu")
    capfd.readouterr()
    gpu = fb.train({**p, "device": "cuda"}, csv, **kw)
    log = capfd.readouterr().err
    assert "GPU_BLOCKING cols_per_block=8" in log, log[-2000:]
    assert _bytes(gpu, tmp_path, "gpu") == ref

# Fast ranking keeps per-query-group buffers on the device (two u32 offsets, f64 +
# f32 + i64 per group: 28 bytes), so the planner must charge the real groups for
# every input kind, not one group per row.
RANK_GROUP_BYTES = 28
RANK_CASES = {
    "ndcg": {"objective": "rank:ndcg"},
    "pairwise_mean": {"objective": "rank:pairwise", "lambdarank_pair_method": "mean"},
    "map": {"objective": "rank:map"},
    "unbiased": {"objective": "rank:ndcg", "lambdarank_unbiased": True},
}


def _rank_table(n, n_feats, seed, binary=False):
    """Ranking data with varying group sizes (5..40 rows) and sorted query ids."""
    rng = np.random.default_rng(seed)
    sizes = rng.integers(5, 41, n // 5)
    sizes = sizes[np.cumsum(sizes) <= n]
    sizes = np.r_[sizes, n - sizes.sum()] if sizes.sum() < n else sizes
    qid = np.repeat(np.arange(len(sizes)), sizes).astype(np.int64)
    x = rng.normal(size=(n, n_feats)).astype(np.float32)
    s = x[:, 0] - 0.6 * x[:, 3 % n_feats] + rng.normal(0, 0.7, n)
    y = (s > 0.3).astype(np.float32) if binary else np.clip(np.round(s + 1.5), 0, 4).astype(
        np.float32)
    cols = {**{f"f{i}": x[:, i] for i in range(n_feats)}, "qid": qid, "label": y}
    return pa.table(cols), len(sizes)


def _plan_mb(log, key):
    return float(re.search(rf"{key}:\s+([\d.]+) MB", log).group(1))


@cuda
@pytest.mark.parametrize("case", list(RANK_CASES))
def test_fast_ranking_plans_the_real_query_groups_for_every_input(tmp_path, capfd, case):
    """Single and multi-file Parquet, CSV and arrays plan the same fast ranking
    buffers: the query groups read from the qid column, not one group per row. The
    same rows with one row per query are charged exactly the extra group buffers."""
    n = 60_000
    table, n_groups = _rank_table(n, 6, 0, binary=case == "map")
    path = str(tmp_path / "rank.parquet")
    pq.write_table(table, path, row_group_size=10_000)
    parts = tmp_path / "parts"
    parts.mkdir()
    for i in range(3):
        pq.write_table(table.slice(i * 20_000, 20_000), str(parts / f"part{i}.parquet"))
    csv = str(tmp_path / "rank.csv")
    table.to_pandas().to_csv(csv, index=False)
    per_row = table.set_column(table.schema.get_field_index("qid"), "qid",
                               pa.array(np.arange(n, dtype=np.int64)))
    per_row_path = str(tmp_path / "per_row.parquet")
    pq.write_table(per_row, per_row_path, row_group_size=10_000)

    p = {**RANK_CASES[case], "device": "cuda", "gpu_math": "fast", "max_depth": 4,
         "max_bin": 32, "seed": 3, "cols_per_block": 3, "hist_streams": 1,
         "resident_blocks": 0}
    kw = {"num_boost_round": 2}
    fixed, models = {}, {}
    for name, src in (("parquet", path), ("multi_parquet", str(parts)), ("csv", csv),
                      ("per_row", per_row_path)):
        capfd.readouterr()
        model = fb.train(p, src, label="label", qid="qid", **kw)
        fixed[name] = _plan_mb(capfd.readouterr().err, "fixed_bytes")
        models[name] = _bytes(model, tmp_path, name)
    x = np.column_stack([table.column(f"f{i}").to_numpy() for i in range(6)])
    capfd.readouterr()
    model = fb.train(p, x, label=table.column("label").to_numpy(),
                     qid=table.column("qid").to_numpy(), **kw)
    fixed["array"] = _plan_mb(capfd.readouterr().err, "fixed_bytes")
    assert fixed["parquet"] == fixed["multi_parquet"] == fixed["csv"] == fixed["array"], fixed
    # The file inputs train the same model; one query per row is another problem.
    assert models["parquet"] == models["multi_parquet"] == models["csv"]
    extra = (n - n_groups) * RANK_GROUP_BYTES / MB
    assert fixed["per_row"] - fixed["parquet"] == pytest.approx(extra, abs=0.11), fixed


# ---------------------------------------------------------------- slow (FB_SLOW_TESTS=1)

def _used_mb():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout
        return int(out.strip().splitlines()[0])
    except Exception:  # noqa: BLE001
        return None


class _PeakVram:
    def __init__(self):
        self.peak = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        while not self._stop.is_set():
            used = _used_mb()
            if used is not None:
                self.peak = max(self.peak, used)
            time.sleep(0.05)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()


@pytest.fixture(scope="module")
def large_parquet(tmp_path_factory):
    """3M x 23 rows (two categorical): an auto plan of 8-column blocks (8, 8, 7) whose
    column-major upload uses several reusable allocation slots."""
    n, f = 3_000_000, 23
    rng = np.random.default_rng(0)
    x = rng.normal(size=(n, f)).astype(np.float32)
    x[:, 2] = rng.integers(0, 6, n)
    x[:, 7] = rng.integers(0, 50, n)
    x[rng.random(x.shape) < 0.02] = np.nan
    xs = np.nan_to_num(x)
    s = xs[:, 0] - 0.6 * xs[:, 5] + 0.3 * xs[:, f - 1] + 0.5 * np.sin(xs[:, 7])
    y = (s + rng.normal(0, 0.7, n) > 0).astype(np.float32)
    path = str(tmp_path_factory.mktemp("large") / "large.parquet")
    pq.write_table(pa.table({**{f"f{i}": x[:, i] for i in range(f)}, "label": y}), path,
                   row_group_size=200_000)
    return path, ["c" if i in (2, 7) else "q" for i in range(f)]


@cuda
@slow
@pytest.mark.parametrize("math", ["exact", "fast"])
def test_large_auto_plans_match_and_stay_within_the_planner_estimate(
        large_parquet, tmp_path, capfd, math):
    """Fully automatic planning (width, streams, residency) on a dataset large enough
    for the budget to force hybrid and pure streaming. Exact matches the CPU model,
    fast is plan-invariant, and the observed device memory never exceeds the estimate."""
    path, ftypes = large_parquet
    p = {"objective": "binary:logistic", "max_depth": 6, "nthread": 4, "seed": 1,
         "subsample": 0.8, "colsample_bytree": 0.8, "colsample_bynode": 0.8}
    kw = {"label": "label", "num_boost_round": 3, "feature_types": ftypes}
    ref = _bytes(fb.train(p, path, quiet=True, **kw), tmp_path, "cpu") if math == "exact" else None
    fb.train({"device": "cuda", "max_depth": 1}, np.zeros((64, 2), np.float32),
             np.zeros(64, np.float32), num_boost_round=1, quiet=True)
    base = _used_mb()
    seen = {}
    labels = set()
    for budget in (None, 760, 710, 690):
        gpu = {"gpu_math": math, **({"gpu_memory_budget": budget * MB} if budget else {})}
        try:
            with _PeakVram() as peak:
                model, info = _gpu(capfd, p, path, kw, **gpu)
        except (ValueError, RuntimeError) as e:
            assert "does not fit" in str(e) and budget is not None
            continue
        labels.add(info["residency"])
        est = float(re.search(r"estimated_peak:\s+([\d.]+) MB", info["log"]).group(1))
        if base is not None:
            assert peak.peak - base <= est + 1, (budget, peak.peak - base, est)
        out = _bytes(model, tmp_path, "gpu")
        if ref is not None:
            assert out == ref, f"budget {budget}: {info['residency']} differs from the CPU model"
        seen[budget] = out
    assert {"full_resident", "pure_streaming"} <= labels, labels
    assert len(set(seen.values())) == 1


@cuda
@slow
@pytest.mark.parametrize("pinned", [False, True])
def test_large_forced_plans_with_large_column_major_upload(
        large_parquet, tmp_path, capfd, monkeypatch, pinned):
    path, ftypes = large_parquet
    if pinned:
        monkeypatch.setenv("FB_CUDA_PINNED", "1")
    p = {"objective": "binary:logistic", "max_depth": 6, "nthread": 4, "seed": 2,
         "cols_per_block": 8, "subsample": 0.5, "colsample_bylevel": 0.8}
    kw = {"label": "label", "num_boost_round": 3, "feature_types": ftypes}
    ref = _bytes(fb.train(p, path, quiet=True, **kw), tmp_path, "cpu")
    for resident in (3, 2, 1, 0):
        for hs in (1, 2):
            model, info = _gpu(capfd, p, path, kw, resident_blocks=resident, hist_streams=hs)
            _check_plan(info, "narrow", resident, 3)
            assert _bytes(model, tmp_path, "gpu") == ref, (resident, hs, pinned)


@pytest.fixture(scope="module")
def large_rank_parquet(tmp_path_factory):
    """3M ranking rows x 20 features, query groups of 5..40 rows (sorted qid), graded
    and binary labels (rank:map)."""
    d = tmp_path_factory.mktemp("large_rank")
    out = {}
    for binary in (False, True):
        table, n_groups = _rank_table(3_000_000, 20, 1, binary=binary)
        path = str(d / f"rank_{int(binary)}.parquet")
        pq.write_table(table, path, row_group_size=200_000)
        out[binary] = (path, n_groups)
    return out


@cuda
@slow
@pytest.mark.parametrize("case", list(RANK_CASES))
def test_large_fast_ranking_estimate_covers_observed_memory_and_tight_budgets_train(
        large_rank_parquet, tmp_path, capfd, case):
    """On 3M rows the fast ranking estimate stays above the observed device memory
    without the old one-group-per-row overcharge, and a budget between the real need
    and that overcharge now trains: fast stays plan-invariant, exact matches the CPU."""
    path, n_groups = large_rank_parquet[case == "map"]
    n = 3_000_000
    p = {**RANK_CASES[case], "max_depth": 6, "nthread": 4, "seed": 1, "cols_per_block": 4}
    kw = {"label": "label", "qid": "qid", "num_boost_round": 3}
    fb.train({"device": "cuda", "max_depth": 1}, np.zeros((64, 2), np.float32),
             np.zeros(64, np.float32), num_boost_round=1, quiet=True)
    base = _used_mb()
    with _PeakVram() as peak:
        ref, info = _gpu(capfd, p, path, kw, gpu_math="fast")
    est = _plan_mb(info["log"], "estimated_peak")
    if base is not None:
        assert peak.peak - base <= est + 1, (peak.peak - base, est)
        # the old charge added (n - n_groups) x 28 bytes on top of this estimate
        assert est - (peak.peak - base) < (n - n_groups) * RANK_GROUP_BYTES / MB, (
            peak.peak - base, est)
    ref = _bytes(ref, tmp_path, "fast_ref")
    # The minimum plan at this width: pure streaming with one histogram stream.
    _, info = _gpu(capfd, p, path, kw, gpu_math="fast", resident_blocks=0, hist_streams=1)
    minimum = _plan_mb(info["log"], "estimated_peak")
    overcharge = (n - n_groups) * RANK_GROUP_BYTES / MB
    reserve = 512.0
    budget = int((reserve + minimum + overcharge / 2) * MB)  # refused before the fix
    model, info = _gpu(capfd, p, path, kw, gpu_math="fast", hist_streams=1,
                       gpu_memory_budget=budget)
    assert _bytes(model, tmp_path, "fast_budget") == ref, info["residency"]
    cpu = _bytes(fb.train(p, path, quiet=True, **kw), tmp_path, "cpu")
    model, _ = _gpu(capfd, p, path, kw, gpu_math="exact", hist_streams=1,
                    gpu_memory_budget=budget)
    assert _bytes(model, tmp_path, "exact_budget") == cpu
