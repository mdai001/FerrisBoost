"""Public API smoke suite for an installed FerrisBoost wheel.

Run this with the Python executable from a fresh environment, outside the
checkout.  NumPy is the only package runtime dependency; PyArrow is optional
fixture tooling that enables the Parquet cases.
"""

from __future__ import annotations

import argparse
import gzip
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import ferrisboost as fb


def named_csv(path: Path, x: np.ndarray, y: np.ndarray | None = None) -> None:
    values = x if y is None else np.column_stack((x, y))
    names = [f"f{i}" for i in range(x.shape[1])]
    if y is not None:
        names.append("target")
    np.savetxt(path, values, delimiter=",", header=",".join(names), comments="")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--forbid-import-prefix",
        help="Fail if ferrisboost was imported from this checkout/source prefix",
    )
    args = parser.parse_args()
    if args.forbid_import_prefix:
        assert not str(Path(fb.__file__).resolve()).startswith(
            str(Path(args.forbid_import_prefix).resolve())
        ), fb.__file__
    assert fb.__version__ == "0.0.1-post1"

    rng = np.random.default_rng(7)
    x = rng.integers(0, 20, size=(120, 4)).astype(np.float32)
    y = ((x[:, 0] + x[:, 2]) > 18).astype(np.float32)
    params = {"objective": "binary:logistic", "max_depth": 3, "nthread": 1}
    passed: list[str] = []
    skipped: list[str] = []

    with tempfile.TemporaryDirectory(prefix="ferrisboost-wheel-smoke-") as tmp:
        root = Path(tmp)
        train_csv = root / "train.csv"
        named_csv(train_csv, x, y)
        model = fb.train(params, train_csv, label="target", num_boost_round=3, quiet=True)
        pred = model.predict(train_csv)
        assert isinstance(pred, np.ndarray)
        assert pred.shape == (len(x),) and pred.dtype == np.float32
        passed += ["csv", "numpy-output"]

        reordered = root / "reordered.csv"
        np.savetxt(
            reordered,
            np.column_stack((x[:, 2], x[:, 0], x[:, 3], x[:, 1])),
            delimiter=",",
            header="f2,f0,f3,f1",
            comments="",
        )
        assert model.predict(reordered).tobytes() == pred.tobytes()
        missing = root / "missing.csv"
        named_csv(missing, x[:, :3])
        try:
            model.predict(missing)
        except ValueError:
            pass
        else:
            raise AssertionError("missing prediction feature was accepted")
        passed += ["named-reorder", "missing-feature-error"]

        gz_path = root / "train.csv.gz"
        with train_csv.open("rb") as src, gzip.open(gz_path, "wb") as dst:
            dst.write(src.read())
        fb.train(params, gz_path, label="target", num_boost_round=1, quiet=True)
        passed.append("csv-gz")

        parts = root / "parts"
        parts.mkdir()
        named_csv(parts / "a.csv", x[:50], y[:50])
        named_csv(parts / "b.csv", x[50:], y[50:])
        a = fb.train(params, [parts / "b.csv", parts / "a.csv"], label="target",
                     num_boost_round=2, quiet=True)
        b = fb.train(params, str(parts / "*.csv"), label="target",
                     num_boost_round=2, quiet=True)
        ap, bp = root / "a.json", root / "b.json"
        a.save_model(str(ap))
        b.save_model(str(bp))
        assert ap.read_bytes() == bp.read_bytes()
        assert a.predict(parts).shape == (len(x),)
        passed += ["multi-file", "directory", "glob", "deterministic-order"]

        headless_train = root / "headless-train.csv"
        np.savetxt(headless_train, np.column_stack((x[:, :2], y, x[:, 2:])), delimiter=",")
        positional = fb.train(
            params, headless_train, label=2, header=False,
            num_boost_round=3, quiet=True,
        )
        headless_pred = root / "headless-predict.csv"
        np.savetxt(headless_pred, x, delimiter=",")
        pos_pred = positional.predict(headless_pred, header=False)
        try:
            fb.train(params, headless_train, label=99, header=False,
                     num_boost_round=1, quiet=True)
        except ValueError:
            pass
        else:
            raise AssertionError("invalid positional label was accepted")
        passed += ["headless-positional", "invalid-label-error"]

        model_path = root / "model.json"
        positional.save_model(str(model_path))
        loaded = fb.Model.load_model(str(model_path))
        assert loaded.predict(headless_pred, header=False).tobytes() == pos_pred.tobytes()
        child_code = (
            "import ferrisboost as f,numpy as n;"
            f"m=f.Model.load_model({str(model_path)!r});"
            f"p=m.predict({str(headless_pred)!r},header=False);"
            "assert isinstance(p,n.ndarray) and p.dtype==n.float32 and p.shape==(120,)"
        )
        child = subprocess.run(
            [sys.executable, "-c", child_code], cwd="/", text=True, capture_output=True,
        )
        assert child.returncode == 0, child.stderr
        passed += ["save-load", "fresh-process-load"]

        quiet_code = (
            "import ferrisboost as f,numpy as n;"
            "x=n.ones((20,2),dtype=n.float32);y=n.zeros(20,dtype=n.float32);"
            "f.train({'nthread':1},x,y,num_boost_round=1,quiet=True)"
        )
        quiet = subprocess.run(
            [sys.executable, "-c", quiet_code], cwd="/", text=True, capture_output=True,
        )
        assert quiet.returncode == 0 and quiet.stdout == "" and quiet.stderr == "", quiet.stderr
        normal = subprocess.run(
            [sys.executable, "-c", quiet_code.replace(",quiet=True", "")],
            cwd="/", text=True, capture_output=True,
        )
        assert normal.returncode == 0
        for marker in ("CPU_THREADS", "SETUP", "TRAIN", "END_TO_END"):
            assert marker in normal.stderr, normal.stderr
        passed += ["quiet", "normal-runtime-log"]

        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            skipped.append("parquet (install pyarrow as smoke fixture tooling)")
        else:
            parquet = root / "train.parquet"
            pq.write_table(pa.table({
                "f0": x[:, 0], "f1": x[:, 1], "f2": x[:, 2], "f3": x[:, 3],
                "target": y,
            }), parquet)
            pm = fb.train(params, parquet, label="target", num_boost_round=2, quiet=True)
            pp = pm.predict(parquet)
            pp_path = root / "parquet.json"
            pm.save_model(str(pp_path))
            assert fb.Model.load_model(str(pp_path)).predict(parquet).tobytes() == pp.tobytes()
            passed.append("parquet-lifecycle")

        try:
            fb.train({**params, "device": "cuda"}, x, y, num_boost_round=1, quiet=True)
        except Exception as exc:  # CUDA is optional on the smoke host.
            skipped.append(f"gpu ({type(exc).__name__})")
        else:
            exact = fb.train({**params, "device": "cuda", "gpu_math": "exact"},
                             x, y, num_boost_round=2, quiet=True)
            fast_a = fb.train({**params, "device": "cuda:0", "gpu_math": "fast"},
                              x, y, num_boost_round=2, quiet=True)
            fast_b = fb.train({**params, "device": "cuda:0", "gpu_math": "fast"},
                              x, y, num_boost_round=2, quiet=True)
            fa, fb_path = root / "fast-a.json", root / "fast-b.json"
            fast_a.save_model(str(fa)); fast_b.save_model(str(fb_path))
            assert fa.read_bytes() == fb_path.read_bytes()
            assert exact.predict(x).shape == fast_a.predict(x).shape
            passed += ["gpu-exact", "gpu-fast", "cuda-ordinal"]

    print(json.dumps({
        "status": "PASS",
        "version": fb.__version__,
        "module": fb.__file__,
        "passed": passed,
        "skipped": skipped,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
