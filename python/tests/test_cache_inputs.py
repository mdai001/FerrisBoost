"""`cache_path` / `cache_budget_bytes` on every input kind.

The quantized column cache is built only by single-file Parquet training.
Every other input used to ignore the two options silently; it now raises a
ValueError that names the input and the way out. Where the cache works, a
second run must reuse it (files untouched) and give the same model.
"""
import numpy as np
import pytest

import ferrisboost as fb

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

BASE = {"objective": "binary:logistic", "max_depth": 3, "eta": 0.3, "nthread": 1}


def data(n=400, m=4, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 8, size=(n, m)).astype(np.float32)
    y = (x[:, 0] - x[:, 1] + rng.normal(0, 0.5, n) > 0).astype(np.float32)
    return x, y


def table(x, y):
    cols = {f"f{i}": pa.array(x[:, i]) for i in range(x.shape[1])}
    cols["target"] = pa.array(y)
    return pa.table(cols)


def write_parquet(path, x, y):
    pq.write_table(table(x, y), str(path))
    return str(path)


def write_parts(dirpath, x, y, n_parts=3):
    dirpath.mkdir()
    t = table(x, y)
    step = t.num_rows // n_parts + 1
    return [write_parquet_table(dirpath / f"part{i}.parquet", t.slice(i * step, step))
            for i in range(n_parts)]


def write_parquet_table(path, t):
    pq.write_table(t, str(path))
    return str(path)


def write_csv(path, x, y, header=True):
    rows = np.column_stack([x, y])
    head = ",".join([f"f{i}" for i in range(x.shape[1])] + ["target"])
    np.savetxt(str(path), rows, delimiter=",", fmt="%g",
               header=head if header else "", comments="")
    if not header:  # savetxt writes an empty first line for header=""
        text = path.read_text().lstrip("\n")
        path.write_text(text)
    return str(path)


def snapshot(cache):
    files = sorted(p for p in cache.iterdir())
    return {p.name: p.stat().st_mtime_ns for p in files}


def margins(model, x):
    return model.predict(x, output_margin=True).tobytes()


CACHE_OPTIONS = [
    ("cache_path", lambda tmp: {"cache_path": str(tmp / "cache")}),
    ("cache_budget_bytes", lambda tmp: {"cache_budget_bytes": 0}),
]


# ------------------------------------------------------------- raising inputs
@pytest.mark.parametrize("name,opts", CACHE_OPTIONS)
def test_several_parquet_files_raise(tmp_path, name, opts):
    x, y = data()
    parts = write_parts(tmp_path / "parts", x, y)
    p = {**BASE, **opts(tmp_path)}
    for spec in (parts, str(tmp_path / "parts"), str(tmp_path / "parts" / "*.parquet")):
        with pytest.raises(ValueError, match=rf"{name} supports a single Parquet file; "
                                             r"got 3 Parquet files .* merge them into one file "
                                             rf"or drop {name}"):
            fb.train(p, spec, label="target", num_boost_round=2, quiet=True)
    assert not (tmp_path / "cache").exists()


def test_both_options_are_named(tmp_path):
    x, y = data()
    parts = write_parts(tmp_path / "parts", x, y, n_parts=2)
    p = {**BASE, "cache_path": str(tmp_path / "cache"), "cache_budget_bytes": 0}
    with pytest.raises(ValueError, match=r"cache_path and cache_budget_bytes support a single "
                                         r"Parquet file; got 2 Parquet files .* "
                                         r"drop cache_path / cache_budget_bytes"):
        fb.train(p, parts, label="target", num_boost_round=2, quiet=True)


@pytest.mark.parametrize("name,opts", CACHE_OPTIONS)
def test_csv_raises(tmp_path, name, opts):
    x, y = data()
    one = write_csv(tmp_path / "one.csv", x, y)
    two = [write_csv(tmp_path / f"p{i}.csv", x[i::2], y[i::2]) for i in range(2)]
    bare = write_csv(tmp_path / "bare.csv", x, y, header=False)
    p = {**BASE, **opts(tmp_path)}
    for spec, kw, got in ((one, {"label": "target"}, "1 CSV file"),
                          (two, {"label": "target"}, "2 CSV files"),
                          (bare, {"label": x.shape[1], "header": False}, "1 CSV file")):
        with pytest.raises(ValueError, match=rf"{name} supports a single Parquet file; got {got} "
                                             r".* convert the data to one Parquet file"):
            fb.train(p, spec, num_boost_round=2, quiet=True, **kw)
    assert not (tmp_path / "cache").exists()


def _in_memory_inputs(x):
    pd = pytest.importorskip("pandas")
    t = table(x, np.zeros(len(x), np.float32)).drop(["target"])
    return {
        "numpy": x,
        "pandas": pd.DataFrame({f"f{i}": x[:, i] for i in range(x.shape[1])}),
        "arrow_table": t,
        "arrow_batch": t.to_batches()[0],
        "arrow_reader": pa.RecordBatchReader.from_batches(t.schema, t.to_batches()),
    }


@pytest.mark.parametrize("name,opts", CACHE_OPTIONS)
@pytest.mark.parametrize("kind", ["numpy", "pandas", "arrow_table", "arrow_batch",
                                  "arrow_reader"])
def test_in_memory_input_raises(tmp_path, name, opts, kind):
    x, y = data()
    p = {**BASE, **opts(tmp_path)}
    with pytest.raises(ValueError, match=rf"{name} supports a single Parquet file; got in-memory "
                                         r"data .* write it to one Parquet file"):
        fb.train(p, _in_memory_inputs(x)[kind], label=y, num_boost_round=2, quiet=True)
    assert not (tmp_path / "cache").exists()


def test_in_memory_continuation_raises(tmp_path):
    x, y = data()
    prior = fb.train(BASE, x, label=y, num_boost_round=2, quiet=True)
    with pytest.raises(ValueError, match="cache_path supports a single Parquet file"):
        fb.train({**BASE, "cache_path": str(tmp_path / "cache")}, x, label=y,
                 num_boost_round=2, init_model=prior, quiet=True)


def test_several_parquet_files_continuation_raises(tmp_path):
    x, y = data()
    parts = write_parts(tmp_path / "parts", x, y)
    prior = fb.train(BASE, parts, label="target", num_boost_round=2, quiet=True)
    with pytest.raises(ValueError, match="got 3 Parquet files"):
        fb.train({**BASE, "cache_path": str(tmp_path / "cache")}, parts, label="target",
                 num_boost_round=2, init_model=prior, quiet=True)


def test_sklearn_has_no_cache_options():
    # The estimators train on in-memory data only and do not take the cache
    # options at all (an unknown constructor argument is a TypeError).
    pytest.importorskip("sklearn")
    from ferrisboost.sklearn import (FerrisBoostClassifier, FerrisBoostRanker,
                                     FerrisBoostRegressor)
    for est in (FerrisBoostClassifier, FerrisBoostRegressor, FerrisBoostRanker):
        for opt in ({"cache_path": "cache"}, {"cache_budget_bytes": 0}):
            with pytest.raises(TypeError):
                est(**opt)


# ------------------------------------------------------------- working inputs
def _build_then_reuse(tmp_path, spec, x, extra=None, **kw):
    cache = tmp_path / "cache"
    p = {**BASE, "cache_path": str(cache), "cache_budget_bytes": 0, **(extra or {})}
    first = fb.train(p, spec, label="target", num_boost_round=3, quiet=True, **kw)
    assert (cache / "manifest.json").exists()
    assert any(f.name.startswith("block-") for f in cache.iterdir())
    before = snapshot(cache)
    second = fb.train(p, spec, label="target", num_boost_round=3, quiet=True, **kw)
    assert snapshot(cache) == before, "the cache was rewritten instead of reused"
    assert margins(first, x) == margins(second, x)
    return first


def test_single_parquet_file_builds_and_reuses(tmp_path):
    x, y = data()
    path = write_parquet(tmp_path / "t.parquet", x, y)
    cached = _build_then_reuse(tmp_path, path, x)
    plain = fb.train(BASE, path, label="target", num_boost_round=3, quiet=True)
    assert margins(cached, x) == margins(plain, x)


def test_directory_or_glob_of_one_parquet_file_reuses(tmp_path):
    x, y = data()
    (tmp_path / "one").mkdir()
    write_parquet(tmp_path / "one" / "t.parquet", x, y)
    _build_then_reuse(tmp_path, str(tmp_path / "one"), x)
    # The glob resolves to the same single file and reuses the same cache.
    before = snapshot(tmp_path / "cache")
    fb.train({**BASE, "cache_path": str(tmp_path / "cache"), "cache_budget_bytes": 0},
             str(tmp_path / "one" / "*.parquet"), label="target", num_boost_round=3, quiet=True)
    assert snapshot(tmp_path / "cache") == before


def test_cache_budget_alone_on_a_single_parquet_file_still_works(tmp_path):
    x, y = data()
    path = write_parquet(tmp_path / "t.parquet", x, y)
    plain = fb.train(BASE, path, label="target", num_boost_round=3, quiet=True)
    spilled = fb.train({**BASE, "cache_budget_bytes": 0}, path, label="target",
                       num_boost_round=3, quiet=True)
    assert margins(spilled, x) == margins(plain, x)


def test_single_parquet_file_with_validation_set_reuses(tmp_path):
    # The training set is cached; the validation set uses the training cuts and
    # never writes into the training cache (it inherits only the budget).
    x, y = data()
    xv, yv = data(n=200, seed=1)
    path = write_parquet(tmp_path / "t.parquet", x, y)
    valid = write_parquet(tmp_path / "v.parquet", xv, yv)
    hist = {}
    _build_then_reuse(tmp_path, path, x, evals=[(valid, "valid")], evals_result=hist)
    assert len(hist["valid"]["logloss"]) == 3
    plain_hist = {}
    fb.train(BASE, path, label="target", num_boost_round=3, quiet=True,
             evals=[(valid, "valid")], evals_result=plain_hist)
    assert hist == plain_hist


def test_single_parquet_file_continuation_reuses(tmp_path):
    x, y = data()
    path = write_parquet(tmp_path / "t.parquet", x, y)
    prior = fb.train(BASE, path, label="target", num_boost_round=2, quiet=True)
    cont = _build_then_reuse(tmp_path, path, x, init_model=prior)
    assert cont.num_trees == 5


def _has_cuda():
    try:
        x, y = data(n=64)
        fb.train({**BASE, "device": "cuda", "max_depth": 2}, x, y, num_boost_round=1, quiet=True)
        return True
    except Exception as e:  # noqa: BLE001
        if "CUDA" in str(e) or "cuda" in str(e):
            return False
        raise


cuda = pytest.mark.skipif(not _has_cuda(), reason="no usable CUDA build or device")


@cuda
def test_single_parquet_file_reuses_on_gpu(tmp_path):
    # A cache is tied to cols_per_block: pin it so the GPU reuses the same cache.
    x, y = data()
    path = write_parquet(tmp_path / "t.parquet", x, y)
    gpu = _build_then_reuse(tmp_path, path, x,
                            {"device": "cuda", "gpu_math": "exact", "cols_per_block": 2})
    cpu = fb.train({**BASE, "cols_per_block": 2}, path, label="target", num_boost_round=3,
                   quiet=True)
    assert margins(gpu, x) == margins(cpu, x)


@cuda
def test_gpu_rejects_unsupported_input_too(tmp_path):
    x, y = data()
    parts = write_parts(tmp_path / "parts", x, y)
    with pytest.raises(ValueError, match="got 3 Parquet files"):
        fb.train({**BASE, "device": "cuda", "cache_path": str(tmp_path / "cache")}, parts,
                 label="target", num_boost_round=2, quiet=True)
    with pytest.raises(ValueError, match="got in-memory data"):
        fb.train({**BASE, "device": "cuda", "cache_path": str(tmp_path / "cache")}, x,
                 label=y, num_boost_round=2, quiet=True)
