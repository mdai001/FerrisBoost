# FerrisBoost

FerrisBoost is a Rust-backed gradient-boosted tree library for large tabular
data. It uses a column-blocked histogram engine, XGBoost-compatible training
semantics for supported objectives, and a Python API with parameter names familiar
to XGBoost. Training runs on a single GPU today; the architecture is multi-GPU
ready (WIP).

## Why FerrisBoost

In large-scale GBDT workflows, GPU acceleration is frequently bottlenecked not by GPU compute kernels, but by host-memory overhead, data ingest, and rigid memory limits. FerrisBoost focuses on reducing ingest, memory, and execution overhead in large tabular workflows:

- **Eliminating Host-Memory Bottlenecks & Slow Setup**: In XGBoost, GPU training often incurs high host memory and multi-pass materialization when constructing a `DMatrix`, `QuantileDMatrix`, or data iterator. FerrisBoost streams and quantizes directly from Parquet or CSV in Rust, making setup **2–8× faster** while slashing peak host RAM by **2–4×** (`0.24–0.48× XGBoost`).
- **Adaptive VRAM Management**: Instead of requiring manual external-memory configurations when datasets exceed physical VRAM, FerrisBoost features an adaptive GPU memory planner that automatically manages residency—seamlessly transitioning between full VRAM residency, hybrid CPU/GPU caching, and streaming execution under memory pressure.
- **Physically Selective Column Sampling**: On the column-major GPU path, `colsample_bytree` builds compact selected-feature histograms over unchanged resident blocks. Unselected features are omitted from histogram initialization, accumulation, subtraction, and device-to-host output without rematerializing resident data for each tree.
- **Optimized for Wide Data (High Feature Counts)**: Datasets with hundreds or thousands of features (wide tables) severely strain GPU histogram construction and setup time. FerrisBoost's **column-blocked histogram engine** tiles features into cache-conscious blocks, ensuring bounded GPU-memory planning, strong cache locality, and sustained throughput on high-dimensional datasets.
- **Single-GPU Now, Multi-GPU Ready (WIP)**: Training targets a single selected GPU today (`device="cuda"` or `device="cuda:N"`). The column-blocked architecture is engineered to scale across multiple GPUs, with multi-GPU support currently in progress (WIP).
- **Deterministic Parity & Flexible Math Modes**: FerrisBoost provides deterministic training semantics. Use `gpu_math="exact"` (default) for byte-identical CPU/GPU models, or `gpu_math="fast"` for maximum GPU throughput; `fast` is deterministic but does not guarantee byte-identical CPU/GPU models.
- **File-First Rust Data Pipeline**: Train directly from Parquet, CSV, or CSV.gz (single or partitioned files), PyArrow tables, or NumPy arrays without constructing an intermediate `DMatrix`, `QuantileDMatrix`, or custom iterator. Passing file paths lets Rust inspect schemas and stream-quantize required columns directly.

## Performance vs XGBoost

FerrisBoost focuses its performance engineering on GPU acceleration, memory efficiency, and data pipelining. The CPU engine provides a reference implementation with the supported exactness contract, but is not yet micro-optimized.

On the tested workloads, FerrisBoost GPU fast used **0.89–1.34×** XGBoost training time, with **0.24–0.48×** peak host RAM and **0.39–0.69×** combined peak footprint (RAM + active VRAM):

- **GPU fast training**: 0.89–1.34× XGBoost time (deterministic; maximum throughput)
- **GPU exact training**: 0.98–1.81× XGBoost time (byte-for-byte CPU-identical)
- **GPU setup**: 0.12–0.49× XGBoost time — approximately 2–8× faster setup before boosting
- **Peak host RAM**: 0.24–0.48× XGBoost
- **Combined RAM + active VRAM**: 0.39–0.69× XGBoost (observed peak footprint during benchmarks; not a hard memory budget guarantee)
- **GPU VRAM**: Adaptive; full residency when memory is available, hybrid or streaming under tighter budgets
- **CPU training**: 1.59–2.42× XGBoost time (reference implementation; not yet optimized)

### v0.0.1-post1 column-sampling validation

`v0.0.1-post1` resolves the column-major resident-path issue where
`colsample_bytree` reduced split enumeration but left histogram construction
and device-to-host output effectively full-width. Selected features now define
the compact histogram initialization, accumulation, sibling subtraction, host
transfer, and enumeration width while resident feature blocks remain unchanged.

A clean-installed generic wheel was tested on HIGGS Parquet and Epsilon CSV
with 100 rounds, depth 6, `eta=0.2`, `subsample=0.6`, and
`colsample_bytree=0.6`. Values are medians of two fresh-process
forward/reverse runs after full input-cache and per-backend warm-ups:

| Dataset | Backend | Setup | Train | Seconds/round | End-to-end | Peak RAM | Active VRAM |
|---|---|---:|---:|---:|---:|---:|---:|
| HIGGS 10.5M × 28 Parquet | CPU | 5.205 s | 67.723 s | 0.6772 | 72.927 s | 816 MiB | 0 |
| HIGGS 10.5M × 28 Parquet | GPU exact | 6.259 s | 15.475 s | 0.1548 | 21.734 s | 676 MiB | 1,092 MiB |
| HIGGS 10.5M × 28 Parquet | GPU fast | 6.296 s | 6.176 s | 0.0618 | 12.472 s | 521 MiB | 1,135 MiB |
| Epsilon 380K × 2,000 CSV | CPU | 52.643 s | 117.610 s | 1.1761 | 170.253 s | 3,468 MiB | 0 |
| Epsilon 380K × 2,000 CSV | GPU exact | 55.399 s | 15.466 s | 0.1547 | 70.865 s | 3,671 MiB | 1,136 MiB |
| Epsilon 380K × 2,000 CSV | GPU fast | 54.397 s | 14.744 s | 0.1474 | 69.141 s | 3,417 MiB | 1,070 MiB |

CPU and GPU-exact models and validation predictions were byte-identical on
both datasets. GPU fast was deterministic; its maximum probability difference
from exact was `1.1920929e-7`. Validation accuracy/AUC were
`0.737168/0.818614` on HIGGS and `0.853400/0.930840` on Epsilon for all three
backends at the shown precision. Prediction/scoring was performed after and
excluded from timing and memory accounting. Active VRAM is sampled above the
pre-arm idle baseline and may miss sub-sampling-interval spikes.

This post-fix table is FerrisBoost-only; the cross-implementation ranges above
come from the broader v0.0.1 release matrix.

## Install

FerrisBoost is available on [PyPI](https://pypi.org/project/ferrisboost/):

```bash
pip install ferrisboost
```

The release tag is `v0.0.1-post1`; Python package metadata uses the normalized
PEP 440 version `0.0.1.post1`.

One wheel, CPU and NVIDIA GPU support. GPU acceleration is optional; CPU training and inference work without NVIDIA hardware or drivers.

The default package is the portable x86-64 build. Systems that implement the
x86-64-v3 ISA level can explicitly install the optimized distribution instead;
both distributions provide the same `ferrisboost` Python API and model format:

```bash
# Portable default, including CPU support and optional NVIDIA GPU acceleration:
pip install ferrisboost

# Explicit x86-64-v3 opt-in (install instead of, not alongside, ferrisboost):
pip install ferrisboost-v3
```

CPU microarchitecture levels are not encoded in stable wheel compatibility
tags, so the optimized build uses a separate distribution name rather than a
custom platform tag. Do not install both distributions into the same environment.

### Install from source

FerrisBoost requires Python 3.9+ and a Rust toolchain when installing from source:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip maturin

# CPU-only build:
maturin develop --release

# CUDA GPU-enabled build (Turing+ / sm_75+; precompiled PTX bundled, CUDA toolkit not required):
maturin develop --release --features "cuda,python,pyo3/extension-module"
```

## Quick start

### File-first training & prediction

Pass file paths directly to `fb.train()` and `model.predict()`. Python acts as the control plane passing paths and parameters, while Rust handles streaming ingest, column selection, and quantization without materializing full tables in Python:

```python
import ferrisboost as fb

# Train directly from a Parquet file:
model = fb.train(
    {
        "objective": "reg:squarederror",
        "max_depth": 8,
        "nthread": 0,
    },
    "train.parquet",
    label="target",
    num_boost_round=200,
)

# Predict returns a 1-D NumPy float32 array:
prediction = model.predict("test.parquet")
model.save_model("model.json")
```

Train on GPU:

```python
# Enable GPU training on a selected device with device="cuda" (or "cuda:0", "cuda:1"):
model = fb.train(
    {
        "objective": "reg:squarederror",
        "device": "cuda",
        "max_depth": 8,
    },
    "train.parquet",
    label="target",
    num_boost_round=200,
)
```

- **`device`**: Single GPU today (`device="cuda"` or `device="cuda:N"`); multi-GPU scaling is in progress (WIP).
- **Adaptive VRAM**: Automatically manages GPU memory, dynamically adapting between full residency and streaming execution based on dataset size and available VRAM.
- **`colsample_bytree`**: Selects a deterministic feature subset per tree. Column-major GPU execution uses that subset as the logical histogram execution/output width while preserving the physical resident-block layout and model compatibility.
- **`gpu_math`**: `"exact"` (default) guarantees byte-by-byte exact CPU reproducibility; `"fast"` is deterministic but does not guarantee byte-identical CPU/GPU models.
- **`nthread`**: `0` selects available physical CPU parallelism.

Multi-file and partitioned datasets (list of files, directory, or glob pattern):

```python
model = fb.train(
    {"objective": "reg:squarederror", "max_depth": 8},
    ["train_part_0.parquet", "train_part_1.parquet"],  # or "data/partitions/" or "data/*.parquet"
    label="target",
    num_boost_round=200,
)
prediction = model.predict(["test_part_0.parquet", "test_part_1.parquet"])
```

CSV files with no header use positional column indices:

```python
model = fb.train(params, "train.csv", label=4, header=False)
prediction = model.predict("test.csv", header=False)
```

### Convenience APIs (NumPy & PyArrow)

For in-memory arrays or exploratory workflows where data is already loaded in Python:

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
        "nthread": 0,
    },
    X,
    label=y,
    num_boost_round=100,
)

probability = model.predict(X)
```

PyArrow tables are also accepted:

```python
import pyarrow as pa

table = pa.table({"f0": X[:, 0], "f1": X[:, 1], "f2": X[:, 2]})
model = fb.train(
    {"objective": "binary:logistic", "max_depth": 6, "eta": 0.1},
    table,
    label=y,
    num_boost_round=100,
)
```

### XGBoost Interchange

Because FerrisBoost models are serialized to standard XGBoost-compatible JSON:

- A FerrisBoost GPU-trained model can be loaded by XGBoost on CPU or GPU:
  ```python
  import xgboost as xgb

  booster = xgb.Booster(model_file="fb_gpu_model.json")
  preds = booster.predict(xgb.DMatrix(X))
  ```

- Similarly, FerrisBoost can load models originally trained by XGBoost (`fb.Model.load_model("xgb_model.json")`) and predict using FerrisBoost's CPU engine.

In the current release, FerrisBoost model prediction runs single-threaded on the CPU. If inference throughput or latency is a primary concern, models can be loaded directly into XGBoost for multi-threaded CPU or GPU serving.

### Known limitations

* **CPU prediction:** tree scoring is currently single-threaded. Large-batch inference may be faster through XGBoost model interchange.
* **Row-sampling efficiency:** `subsample` currently preserves full-row routing so every row receives every learned tree. Column sampling is physically selective on the column-major GPU path; the row-major path skips unselected accumulation but retains full-width histogram buffers.
* **GPU inference:** post-training prediction currently runs on CPU, including models trained on GPU.
* **Multi-GPU:** training currently uses one selected GPU per job; multi-GPU execution is not yet implemented.

These are performance and feature limitations, not correctness failures. Future work includes parallel CPU prediction, more selective row-sampling execution, and multi-GPU support.

## Learn more

See the [Python API How-To](documents/howto.md) for installation details,
Parquet/CSV input formats, validation, early stopping, model persistence,
logging, and the complete parameter reference.

## License

Apache-2.0.
