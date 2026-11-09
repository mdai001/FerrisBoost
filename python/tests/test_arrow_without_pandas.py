"""Arrow category adapters do not require pandas at runtime."""
import subprocess
import sys

import pytest

pytest.importorskip("pyarrow")


def test_arrow_category_train_eval_predict_without_pandas_import():
    code = '''import builtins\nimport numpy as np\nimport pyarrow as pa\nimport ferrisboost as fb\noriginal_import = builtins.__import__\ndef no_pandas(name, *args, **kwargs):\n    if name == "pandas" or name.startswith("pandas."):\n        raise ModuleNotFoundError("pandas blocked for optional-dependency check")\n    return original_import(name, *args, **kwargs)\nbuiltins.__import__ = no_pandas\nn = 121\nkeys = pa.array([i % 2 for i in range(n - 1)] + [None], type=pa.int8())\nstrings = pa.DictionaryArray.from_arrays(keys, pa.array(["a", "b"]))\nintegers = pa.DictionaryArray.from_arrays(keys, pa.array([10, 20], type=pa.int64()))\nx = pa.table({"s": strings, "i": integers})\ny = np.arange(n, dtype=np.float32)\nm = fb.train({"objective": "reg:squarederror", "enable_categorical": True,\n              "max_depth": 2, "nthread": 1}, x, label=y, num_boost_round=2,\n             evals=[(x, y, "validation")], quiet=True)\nm.predict(x, output_margin=True)\n'''
    run = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
