# FerrisBoost Python API How-To

FerrisBoost is a histogram-based gradient-boosted tree library with NumPy,
PyArrow, Parquet, and CSV training inputs. Its parameter names and JSON model
format are designed to be familiar to XGBoost users, but FerrisBoost does not
use a `DMatrix` object.

The public Python API supports CPU training and CUDA-enabled builds. Select a
GPU with `device="cuda"` or `device="cuda:N"`.

## Choose the input form

Use the source-oriented API as the default:

- **Large-data training, especially on GPU: pass a file path.** Parquet is the
  preferred format. A path lets the planner inspect the schema and choose
  quantization, block width, resident versus streaming execution, and spill
  behavior without materializing the full table first.

For CUDA training, this is an adaptive residency decision rather than a fixed
mode switch: FerrisBoost automatically trades GPU memory for training speed,
from full residency through hybrid execution to pure streaming when memory is
under pressure.
- **In-memory convenience: NumPy and PyArrow are optional.** Use them when the
  data is already in memory and comfortably fits there.

Do not put pandas on the main large-data path. In particular, avoid loading a
complete Parquet dataset into a `DataFrame` and then converting it to NumPy or
Arrow before calling FerrisBoost. That materializes the dataset before the
planner can apply its memory budget and introduces avoidable host-memory and
copying costs.

```python
# Recommended for large data. The label names a column in the file.
model = fb.train(params, "train.parquet", label="target")

# Optional convenience for data that is already resident in memory.
model = fb.train(params, X_numpy, label=y_numpy)
```

The same path-first input contract is the primary interface for large-data GPU
training; GPU selection does not require users to redesign the loading plan.

## Install

Install the prebuilt wheel from [PyPI](https://pypi.org/project/ferrisboost/):

```bash
pip install ferrisboost
```

### Install from the source tree

FerrisBoost requires Python 3.9 or newer and a Rust toolchain. For development,
create or activate a virtual environment and install Maturin:

```bash
python -m pip install --upgrade pip maturin

# CPU-only build:
maturin develop --release

# CUDA GPU-enabled build (Turing+ / sm_75+; precompiled PTX bundled):
maturin develop --release --features "cuda,python,pyo3/extension-module"
```

Use a release build for training or benchmarking. A debug build is
substantially slower.

To build a wheel instead:

```bash
# CPU wheel:
maturin build --release

# CUDA GPU wheel:
maturin build --release --features "cuda,python,pyo3/extension-module"
python -m pip install target/wheels/ferrisboost-*.whl
```

If a release bundle provides both `generic/` and `x86-64-v3/`, install the
generic wheel unless the target CPU is known to implement the x86-64-v3 ISA
level. The optimized build is not a runtime-dispatched universal wheel. The two
files have the same Python compatibility tag and are therefore distributed in
separate directories.

Check the installed package:

```python
import ferrisboost as fb

print(fb.__version__)
```

## Train from NumPy arrays

Pass a two-dimensional feature matrix and a one-dimensional label array:

```python
import numpy as np
import ferrisboost as fb

rng = np.random.default_rng(42)
X = rng.normal(size=(10_000, 20)).astype(np.float32)
y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(np.float32)

model = fb.train(
    {
        "objective": "binary:logistic",
        "max_depth": 6,
        "eta": 0.1,
        "max_bin": 255,
        "nthread": 0,
    },
    X,
    label=y,
    num_boost_round=100,
)

probability = model.predict(X)
margin = model.predict(X, output_margin=True)
```

Training converts array-like input to a C-contiguous `float32` matrix when
necessary. Labels are flattened and converted to contiguous `float32`.
Missing feature values may be represented by `NaN`; labels must not contain
`NaN`.

`nthread=0` lets FerrisBoost choose the physical core count. Set a positive
integer to request an exact thread count.

## Regression

Use `reg:squarederror` for regression:

```python
target = (2.0 * X[:, 0] - 0.25 * X[:, 1]).astype(np.float32)

model = fb.train(
    {
        "objective": "reg:squarederror",
        "max_depth": 5,
        "eta": 0.05,
        "nthread": 0,
    },
    X,
    label=target,
    num_boost_round=200,
)

prediction = model.predict(X)
```

For squared-error models, normal predictions and raw-margin predictions are
the same.

## Validation and early stopping

For in-memory data, each validation entry is a
`(features, labels, name)` tuple:

```python
X_train, X_valid = X[:8_000], X[8_000:]
y_train, y_valid = y[:8_000], y[8_000:]
history = {}

model = fb.train(
    {
        "objective": "binary:logistic",
        "max_depth": 6,
        "eta": 0.1,
    },
    X_train,
    label=y_train,
    num_boost_round=500,
    evals=[(X_valid, y_valid, "valid")],
    early_stopping_rounds=20,
    evals_result=history,
    verbose_eval=10,
)

print(model.best_iteration)
print(model.best_score)
print(history["valid"]["logloss"])
```

FerrisBoost automatically reuses the training cuts for validation data. This
is required for consistent bin assignments.

Early stopping watches the first metric of the last validation set. The
currently supported objectives each expose one default metric:

- `binary:logistic`: `logloss`
- `reg:squarederror`: `rmse`

Like `xgboost.Booster.predict`, `Model.predict()` uses all trees by default,
including trees built after the best iteration. To predict with the early-
stopping optimum:

```python
best_prediction = model.predict(
    X_valid,
    n_trees=model.best_iteration + 1,
)
```

## Custom callbacks

A callback receives the zero-based round index and the metrics for that round.
Return `True` to continue or `False` to stop training:

```python
def monitor(round_index, metrics):
    if "valid" in metrics:
        print(round_index, metrics["valid"])
    return round_index < 49

model = fb.train(
    {"objective": "binary:logistic"},
    X_train,
    label=y_train,
    num_boost_round=500,
    evals=[(X_valid, y_valid, "valid")],
    callbacks=[monitor],
)
```

Exceptions raised by callbacks propagate to the caller. The main training loop
releases Python's GIL; it reacquires the GIL when invoking a Python callback.

## Train from Parquet or CSV

Pass the file path as `data` and the label column name as `label`:

```python
model = fb.train(
    {
        "objective": "binary:logistic",
        "max_depth": 6,
        "eta": 0.1,
        "nthread": 0,
    },
    "train.parquet",
    label="target",
    num_boost_round=100,
)
```

The format is inferred from the file extension. Supported extensions are
`.parquet`, `.pq`, `.csv`, and `.csv.gz`.

Inputs can also be multi-file or partitioned datasets: pass a list of file
paths, a directory path, or a glob pattern (`*`, `?`). Files are sorted
lexicographically to ensure deterministic row ordering across runs:

```python
# Pass a list of files, a directory path, or a glob pattern
model = fb.train(
    {"objective": "binary:logistic", "eta": 0.1},
    ["part_0.parquet", "part_1.parquet"],  # or "data/partitions/" or "data/*.parquet"
    label="target",
    num_boost_round=100,
)
```

CSV files without a header use positional mode (`header=False`) with a zero-based integer index for `label`:

```python
# Train from headerless CSV with label at column index 4:
model = fb.train(
    {"objective": "binary:logistic", "eta": 0.1},
    "train.csv",
    label=4,
    header=False,
    num_boost_round=100,
)
```

All supported numeric file columns use the same training precision boundary:
FerrisBoost converts Arrow/Parquet `float64`, CSV numeric values, and integer
values to `float32` before building quantization cuts. Parquet null/`NaN` and
empty CSV fields then use the same missing-value path. Consequently, file input does not retain full
`float64` distinctions below `float32` precision; convert or rescale upstream
if those distinctions are significant. Parquet, CSV, and CSV.gz apply this
conversion identically.

File loading uses `ingest_threads=0` (system-adaptive auto) by default,
independently of training `nthread`. Auto respects process CPU affinity/cgroup
limits, available work, and memory safety bounds. Readers preserve file and row
order for Parquet, CSV, and CSV.gz inputs.

For file-backed validation data, entries are `(path, name)` tuples. Validation
files must use the same label column name as the training file:

```python
history = {}

model = fb.train(
    {"objective": "binary:logistic", "eta": 0.1},
    "train.parquet",
    label="target",
    num_boost_round=500,
    evals=[("valid.parquet", "valid")],
    early_stopping_rounds=20,
    evals_result=history,
)
```

Parquet is the preferred large-data input. FerrisBoost quantizes it once into
column blocks and then trains from the quantized cache. CSV cannot project
individual column blocks efficiently, so it is fully quantized in memory.

## Control the Parquet quantized cache

Cache settings belong in the parameter dictionary.

`cache_budget_bytes` limits how much quantized feature data may remain in
memory. Data beyond the budget spills to local files:

```python
model = fb.train(
    {
        "objective": "binary:logistic",
        "cache_budget_bytes": 2 * 1024**3,
    },
    "train.parquet",
    label="target",
    num_boost_round=100,
)
```

Use zero to force the quantized cache to spill:

```python
params = {
    "objective": "binary:logistic",
    "cache_budget_bytes": 0,
}
```

When the parameter is omitted, FerrisBoost uses half of Linux
`MemAvailable`, with a 4 GiB fallback if that value cannot be read.

Use `cache_path` to persist and reuse the quantized training cache across
processes:

```python
from pathlib import Path

params = {
    "objective": "binary:logistic",
    "cache_path": Path("./ferrisboost-cache"),
}

first = fb.train(params, "train.parquet", label="target", num_boost_round=100)
second = fb.train(params, "train.parquet", label="target", num_boost_round=100)
```

The second call can reuse cuts and quantized blocks. Cache metadata is checked;
incompatible settings such as a different `max_bin` are rejected rather than
silently reused.

`cache_path` applies only to the training set. Validation sets reuse the
training cuts but do not share the training cache directory.

## Train from PyArrow

PyArrow `Table`, `RecordBatch`, and `RecordBatchReader` inputs are accepted with labels supplied
separately:

```python
import pyarrow as pa

table = pa.table({
    "f0": X[:, 0],
    "f1": X[:, 1],
    "f2": X[:, 2],
})

model = fb.train(
    {"objective": "binary:logistic"},
    table,
    label=y,
    num_boost_round=50,
)
```

The current Python wrapper materializes PyArrow columns into a contiguous
NumPy `float32` matrix. For data that should not be fully materialized, use the
Parquet path API instead.

## Train on GPU

When FerrisBoost is built with CUDA support (`--features "cuda,python,pyo3/extension-module"`), training runs on NVIDIA GPUs (Turing architecture or newer, compute capability sm_75+):

```python
model = fb.train(
    {
        "objective": "binary:logistic",
        "device": "cuda",  # or a specific ordinal: "cuda:0", "cuda:1"
        "max_depth": 6,
        "eta": 0.1,
    },
    "train.parquet",
    label="target",
    num_boost_round=100,
)
```

FerrisBoost's memory planner automatically trades GPU memory for training speed without manual chunk or batch tuning:
- **Adaptive residency**: Dynamically adapts across full VRAM residency, hybrid CPU/GPU caching, and pure streaming under extreme memory pressure.
- **`gpu_memory_budget`**: Sets the memory ceiling in bytes for GPU operations. When omitted, FerrisBoost defaults to available free VRAM on the device.
- **`gpu_math`**: `"exact"` (default) produces a byte-by-byte exact model matching CPU training. Setting `"fast"` is deterministic but does not guarantee byte-identical CPU/GPU models, trading bit-level CPU reproducibility for maximum GPU throughput.
- **`device_quantize`**: `True` (default) performs feature binning directly on the GPU.
- **`hist_streams`**: Defaults to automatic (`None`), resolved by the GPU planner based on execution path, block width, and VRAM budget (selects 2 streams for chunked row-major streaming when memory allows, and 1 stream for resident or column-major paths); can be explicitly set to `1` or `2`.
- **`device` selection**: Training currently targets one selected GPU (`device="cuda"` or `device="cuda:N"`). Multi-GPU scaling is in progress (WIP).
- **`max_depth`**: Depth-wise tree growth on GPU is constrained to `0..=14` by partition buffer limits.

## Prediction

`Model.predict()` supports both file paths (the recommended path for large test sets) and in-memory NumPy arrays. The return value is always a one-dimensional `numpy.ndarray` of `float32`.

### File prediction

Passing file paths streams records directly through Rust readers without materializing the full dataset in Python host memory:

```python
# Single file (.parquet, .pq, .csv, or .csv.gz):
prediction = model.predict("test.parquet")

# Multi-file or partitioned dataset (list of files, directory path, or glob):
prediction = model.predict(["test_part_0.parquet", "test_part_1.parquet"])
prediction = model.predict("data/test_partitions/")
prediction = model.predict("data/test_*.parquet")

# Headerless CSV (positional mode):
prediction = model.predict("test.csv", header=False)
```

In named mode (Parquet or CSV with headers), FerrisBoost binds columns by the feature names recorded during training. Surplus columns—including leftover label columns—are ignored automatically. In positional mode (`header=False`), the file must have exactly `model.n_features` columns.

### In-memory array prediction

Prediction also accepts a two-dimensional NumPy `float32` array with the same number and order of features used for training:

```python
X_test = np.asarray(X_test, dtype=np.float32)

prediction = model.predict(X_test)
raw_margin = model.predict(X_test, output_margin=True)
first_20_trees = model.predict(X_test, n_trees=20)
```

C-contiguous input uses the direct row-major fast path. Fortran-contiguous and strided `float32` arrays are also interpreted correctly. Non-`float32` arrays, pandas `DataFrame`s, or PyArrow `Table`s must be converted explicitly with `np.asarray(..., dtype=np.float32)`.

In the current release, FerrisBoost model prediction executes single-threaded on the CPU. If inference throughput or latency is a primary concern, models can be loaded directly into XGBoost for multi-threaded CPU or GPU serving.

The model exposes these properties:

```python
print(model.num_trees)
print(model.n_features)
print(model.best_iteration)
print(model.best_score)
print(model.evals_result())
```

## Save, load, and exchange models with XGBoost

FerrisBoost saves XGBoost-compatible JSON:

```python
model.save_model("model.json")

restored = fb.Model.load_model("model.json")
prediction = restored.predict(X.astype(np.float32))
```

Model files use bounded positional I/O. `model_io_threads=0` selects a
system-adaptive worker count and is independent of file ingest and training
threads. Small models automatically use one worker; override per operation when
needed:

```python
model.save_model("model.json", model_io_threads=2)
restored = fb.Model.load_model("model.json", model_io_threads=2)
```

Save/load and file ingest emit concise progress on stderr every few seconds.
The start record shows the requested mode and resolved count, for example
`ingest_threads=auto workers=8` or `model_io_threads=auto workers=1`. Auto uses
CPUs actually available to the process (including affinity/cgroup limits),
independent work, and a conservative transient-memory bound. A positive value
overrides CPU auto-selection but remains bounded by available work and memory.
Pass `quiet=True` to suppress routine messages.

XGBoost can load a model written by FerrisBoost:

```python
import xgboost as xgb

booster = xgb.Booster(model_file="model.json")
xgb_prediction = booster.predict(xgb.DMatrix(X))
```

FerrisBoost can also load compatible XGBoost JSON models:

```python
xgb_model = xgb.train(
    {
        "objective": "binary:logistic",
        "tree_method": "hist",
        "base_score": 0.5,
    },
    xgb.DMatrix(X, label=y),
    num_boost_round=20,
)
xgb_model.save_model("xgboost-model.json")

ferris_model = fb.Model.load_model("xgboost-model.json")
```

Evaluation history is not stored in model JSON, so a loaded model returns an
empty `evals_result()`.

## Supported parameters

| Parameter | Default | Notes |
|---|---:|---|
| `objective` | `binary:logistic` | Also supports `reg:squarederror` |
| `max_depth` | `6` | Depth-wise tree growth (`0..=14`) |
| `max_bin` | `255` | Must be between 1 and 255 |
| `eta` | `0.3` | Alias: `learning_rate` |
| `lambda` | `1.0` | Alias: `reg_lambda` |
| `gamma` | `0.0` | Minimum split loss |
| `min_child_weight` | `1.0` | Minimum child Hessian |
| `subsample` | `1.0` | Deterministic per-tree row sampling in `(0, 1]`; all rows still receive the learned tree |
| `colsample_bytree` | `1.0` | Deterministic per-tree feature sampling in `(0, 1]`; column-major GPU histograms use compact selected-feature output |
| `cols_per_block` | automatic | Positive integer; normally leave unset |
| `nthread` | `0` | Zero selects physical cores; alias: `n_jobs` |
| `device` | `"cpu"` | Compute device: `"cpu"`, `"cuda"`, or `"cuda:N"` |
| `gpu_memory_budget` | automatic | Positive integer byte limit for GPU memory (defaults to free VRAM) |
| `gpu_math` | `"exact"` | `"exact"` (byte-by-byte identical to CPU) or `"fast"` (matches XGBoost GPU behavior) |
| `device_quantize` | `True` | Direct GPU feature quantization |
| `hist_streams` | automatic | Planned by GPU planner based on execution path, block width, and VRAM; override with `1` or `2` |
| `hist_nodes_per_batch` | automatic | Multi-node merge batch size (`1..=32`) |
| `ingest_threads` | `0` (auto) | Process-visible CPUs, independent work, and memory bound file ingest; independent of `nthread` |
| `model_io_threads` | `0` (auto) | Size/work/memory bound model-file I/O; small models use one worker |
| `seed` | `0` | Alias: `random_state` |
| `base_score` | `0.5` | Value before the objective transform |
| `cache_budget_bytes` | automatic | Parquet quantized-cache memory budget |
| `cache_path` | `None` | Persistent Parquet cache directory |

Unknown parameters raise `ValueError`; they are not silently ignored.

The following familiar XGBoost parameters are recognized but not implemented:

- `alpha` / `reg_alpha`
- `max_leaves`
- `scale_pos_weight`
- `tree_method` (FerrisBoost currently has only histogram training)
- `num_parallel_tree`

Call options accepted by `fb.train()`, `fb.Model.save_model()`, and `fb.Model.load_model()`:
- `quiet`: Set `True` to suppress routine progress logging on stderr.
- `header`: Set `False` for headerless CSV files using positional column indices.

## Common errors

### The label argument has the wrong type

- NumPy or PyArrow input: pass an array-like label vector.
- Parquet or CSV path: pass the label column name as a string.

### Prediction rejects the input dtype

Convert prediction input explicitly:

```python
X_test = np.asarray(X_test, dtype=np.float32)
prediction = model.predict(X_test)
```

### Prediction reports the wrong number of features

The prediction matrix must have exactly `model.n_features` columns in training
feature order. File label columns are excluded automatically during training;
do not include the label as a prediction feature.

### Early stopping does not stop

Early stopping requires at least one validation set. Without `evals`, there is
no metric for the early-stopping callback to watch.

### A persistent cache is rejected

Do not reuse a cache directory after changing data identity, feature schema,
label column, `max_bin`, or another cache-defining setting. Use a new directory
or remove the obsolete cache deliberately.

### CUDA is requested on a CPU-only build

Passing `device="cuda"` on a build compiled without the CUDA feature raises:
```text
ValueError: 这个构建没有 CUDA 支持:请用 --features cuda 重新构建,或改用 device="cpu"
```
Rebuild with:
```bash
maturin develop --release --features "cuda,python,pyo3/extension-module"
```

### Header and label specification mismatch

- When `header=True` (default), `label` must be a column name string (e.g. `label="target"`).
- When `header=False`, `label` must be a 0-based integer column index (e.g. `label=4`).
- `header=False` is only supported for CSV inputs; Parquet schemas always supply column names.

### Positional CSV prediction column count mismatch

In positional mode (`header=False`), test CSV files must contain exactly `model.n_features` columns. In contrast to named mode (which can safely filter and ignore surplus label columns), positional mode binds by column position, so a leftover training label column in the test file will cause a column count mismatch error.

### max_depth is out of range

`max_depth` must be an integer between 0 and 14 (`0 <= max_depth <= 14`). Depth 0 corresponds to a root-only tree; the upper limit of 14 is constrained by the partition buffer on GPU.
