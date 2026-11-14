"""Multi-file Parquet spill caches preserve the ordinary training path."""
import hashlib
import json

import numpy as np
import pytest

import ferrisboost as fb

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

BASE = {"objective": "binary:logistic", "max_depth": 3, "eta": 0.25,
        "nthread": 1, "max_bin": 16, "seed": 3}
NAMES = ["alpha", "beta", "gamma", "delta"]


def data(n=240, seed=19):
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 9, size=(n, len(NAMES))).astype(np.float32)
    x[::23, 1] = np.nan
    x[::31, 3] = np.nan
    y = ((np.nan_to_num(x[:, 0]) - np.nan_to_num(x[:, 2])
          + rng.normal(0, 0.6, n)) > 0).astype(np.float32)
    w = (0.5 + (np.arange(n) % 7) * 0.25).astype(np.float32)
    return x, y, w


def write_parts(directory, x, y, w, n_parts=3):
    directory.mkdir()
    table = pa.table({**{name: pa.array(x[:, i]) for i, name in enumerate(NAMES)},
                      "label": pa.array(y), "weight": pa.array(w)})
    step = (len(y) + n_parts - 1) // n_parts
    paths = []
    for i in range(n_parts):
        part = table.slice(i * step, step)
        # The first shard establishes canonical feature order. Later shards
        # can have a different physical order, including their target columns.
        if i % 2 == 0:
            part = part.select(list(reversed(part.column_names)))
        part = part.replace_schema_metadata({b"dataset": b"spill-regression-v1",
                                             b"schema_version": b"1"})
        path = directory / f"part{i}.parquet"
        pq.write_table(part, str(path))
        paths.append(str(path))
    return paths


def margins(model, x):
    return np.asarray(model.predict(x, output_margin=True)).tobytes()


def cache_snapshot(path):
    return {
        item.name: (item.stat().st_size, item.stat().st_mtime_ns,
                    hashlib.sha256(item.read_bytes()).hexdigest())
        for item in sorted(path.iterdir()) if item.is_file()
    }


@pytest.mark.parametrize("source_kind", ["list", "glob", "directory"])
@pytest.mark.parametrize("cache_option", ["cache_path", "budget"])
def test_multi_file_spill_matches_array_and_reuses_blocks(
        tmp_path, source_kind, cache_option):
    x, y, w = data()
    paths = write_parts(tmp_path / "parts", x, y, w)
    source = {"list": paths,
              "glob": str(tmp_path / "parts" / "*.parquet"),
              "directory": str(tmp_path / "parts")}[source_kind]
    cache = tmp_path / "persistent-cache"
    options = ({"cache_path": str(cache)} if cache_option == "cache_path"
               else {"cache_budget_bytes": 0})
    if cache_option == "cache_path":
        options["cache_budget_bytes"] = 0
    cached = fb.train({**BASE, **options}, source, label="label", weight="weight",
                      num_boost_round=4, quiet=True)
    if cache_option == "cache_path":
        assert (cache / "manifest.json").exists()
        manifest = json.loads((cache / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["n_features"] == len(NAMES)
        before = cache_snapshot(cache)
        reused = fb.train({**BASE, **options}, source, label="label", weight="weight",
                          num_boost_round=4, quiet=True)
        assert cache_snapshot(cache) == before, "persistent cache blocks were rewritten"
        assert margins(cached, x[:, ::-1]) == margins(reused, x[:, ::-1])

    # File fields are physically reversed, so the array reference uses the
    # matching logical feature order. The weighted training paths must agree
    # bit for bit, including rows with NaNs.
    reference = fb.train(BASE, x[:, ::-1], label=y, weight=w,
                         num_boost_round=4, quiet=True)
    assert margins(cached, x[:, ::-1]) == margins(reference, x[:, ::-1])
    assert list(cached.feature_names) == list(reversed(NAMES))


@pytest.mark.parametrize("mismatch", ["source", "mtime", "single", "max_bin"])
def test_multi_file_cache_rejects_changed_source_or_training_parameters(tmp_path, mismatch):
    x, y, w = data()
    paths = write_parts(tmp_path / "parts", x, y, w)
    cache = tmp_path / "cache"
    fb.train({**BASE, "cache_path": str(cache)}, paths, label="label", weight="weight",
             num_boost_round=2, quiet=True)

    if mismatch == "source":
        x2, y2, w2 = data(seed=20)
        changed = write_parts(tmp_path / "changed", x2, y2, w2)
        params = {**BASE, "cache_path": str(cache)}
        source = changed
    elif mismatch == "mtime":
        import os
        mtime = os.stat(paths[1]).st_mtime_ns
        os.utime(paths[1], ns=(mtime, mtime + 2_000_000_000))
        source = paths
        params = {**BASE, "cache_path": str(cache)}
    elif mismatch == "single":
        source = paths[0]
        params = {**BASE, "cache_path": str(cache)}
    else:
        params = {**BASE, "max_bin": BASE["max_bin"] + 8, "cache_path": str(cache)}
        source = paths
    with pytest.raises((ValueError, RuntimeError), match="cache|source|changed|max_bin"):
        fb.train(params, source, label="label", weight="weight", num_boost_round=2,
                 quiet=True)


def test_multi_file_validation_uses_training_cuts_and_does_not_touch_training_cache(tmp_path):
    x, y, w = data()
    xv, yv, wv = data(n=120, seed=29)
    train_parts = write_parts(tmp_path / "train", x, y, w)
    write_parts(tmp_path / "eval", xv, yv, wv)
    cache = tmp_path / "train-cache"
    params = {**BASE, "cache_path": str(cache), "cache_budget_bytes": 0}
    cached_history = {}
    cached = fb.train(params, train_parts, label="label", weight="weight", num_boost_round=4,
                      evals=[(str(tmp_path / "eval" / "*.parquet"), "valid")], evals_result=cached_history, quiet=True)
    before = cache_snapshot(cache)
    reused_history = {}
    fb.train(params, train_parts, label="label", weight="weight", num_boost_round=4,
             evals=[(str(tmp_path / "eval"), "valid")], evals_result=reused_history, quiet=True)
    assert cached_history == reused_history
    assert cache_snapshot(cache) == before, "validation or reuse changed the training cache"

    plain_history = {}
    reference = fb.train(BASE, x[:, ::-1], label=y, weight=w, num_boost_round=4,
                         evals=[(xv[:, ::-1], yv, wv, "valid")],
                         evals_result=plain_history, quiet=True)
    assert cached_history == plain_history
    assert margins(cached, x[:, ::-1]) == margins(reference, x[:, ::-1])
