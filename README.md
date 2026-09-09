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
- **File-First Rust Data Pipeline**: Train directly from Parquet, CSV, or CSV.gz (single or partitioned files), PyArrow tables, or NumPy arrays without constructing an intermediate `DMatrix`, `QuantileDMatrix`, or custom iterator. **Parquet is the recommended file format** for production and repeated training or prediction because it preserves typed schemas, supports efficient column projection, and avoids CSV text-parsing overhead. CSV and CSV.gz remain fully supported compatibility inputs for existing pipelines. The Rust CSV path reuses schema preflight, computes cuts and the exact row count on its first pass, then extracts labels and quantizes directly into final column-major buffers on its second pass. File prediction can stream deterministic CPU results directly into one Parquet output without materializing a full NumPy result.

## Performance vs XGBoost

FerrisBoost focuses its performance engineering on GPU acceleration, memory efficiency, and data pipelining. The CPU engine provides a reference implementation with the supported exactness contract, but is not yet micro-optimized.

The following ratios compare FerrisBoost with the corresponding XGBoost backend on standardized HIGGS Parquet and Epsilon CSV workloads. Timing and memory values are **FerrisBoost / XGBoost**, so values below `1.0×` favor FerrisBoost. Accuracy is also **FerrisBoost / XGBoost**, where values close to `1.0×` indicate equivalent predictive quality.

| Workload | FerrisBoost backend | Setup time | Training time | Setup + training | Peak host RAM | RAM + active VRAM | Accuracy |
|---|---|---:|---:|---:|---:|---:|---:|
| HIGGS, narrow Parquet | CPU | **0.29×** | 1.46× | 1.14× | **0.46×** | — | 1.0002× |
| HIGGS, narrow Parquet | GPU exact | **0.35×** | 2.65× | **0.96×** | **0.43×** | **0.68×** | 0.9999× |
| HIGGS, narrow Parquet | GPU fast | **0.36×** | **0.97×** | **0.52×** | **0.33×** | **0.65×** | 0.9999× |
| Epsilon, wide CSV | CPU | **0.14×** | **0.59×** | **0.31×** | **0.42×** | — | 0.9964× |
| Epsilon, wide CSV | GPU exact | **0.15×** | **0.98×** | **0.19×** | **0.42×** | **0.47×** | 0.9987× |
| Epsilon, wide CSV | GPU fast | **0.15×** | **0.99×** | **0.19×** | **0.43×** | **0.48×** | 0.9987× |

On these workloads, FerrisBoost used approximately **0.14–0.36×** XGBoost setup time, **0.33–0.46×** peak host RAM, and **0.47–0.68×** combined RAM plus active VRAM on GPU. GPU-fast training used **0.97–0.99×** XGBoost GPU training time, while setup plus GPU-fast training used **0.19–0.52×** total time. Accuracy ratios remained within **0.996–1.001×** of XGBoost.

These ratios describe the tested workloads and hardware, not universal performance guarantees. Setup includes input processing, quantization, planner initialization, and initial GPU preparation. Combined memory is an observed secondary footprint indicator, not a hard memory-budget guarantee. Metric scoring is excluded from training time. GPU memory remains adaptive: full residency is used when possible, with hybrid or streaming execution under tighter budgets.

## Install

FerrisBoost is available on [PyPI](https://pypi.org/project/ferrisboost/):

```bash
pip install ferrisboost
```

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
prediction = model.predict("test.parquet", predict_threads=0)
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
- **Prediction CPU threads**: `model.predict(..., predict_threads=0)` uses process-aware automatic CPU parallelism independently of training `nthread`; a positive value overrides it.
- **GPU prediction**: `model.predict(..., use_gpu=True)` opts into single-GPU inference. The compact tree model is uploaded lazily and cached for reuse; small or transfer-heavy batches automatically remain on CPU.

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

FerrisBoost prediction is independent of the training backend. CPU prediction parallelizes deterministic row ranges with `predict_threads=0` by default. CUDA-enabled builds can opt into single-GPU prediction with `use_gpu=True`; repeated calls reuse a lazy compact device model. File prediction validates the complete model schema, then materializes only features used by the trees. The canonical model feature IDs are never rewritten: compact IDs exist only in the cached inference representation, preserving save/load and XGBoost compatibility. A transfer-aware selector keeps small or transfer-heavy batches on CPU. CPU single-thread, CPU multi-thread, and GPU outputs preserve the same row order and prediction bits for the same model.

File-to-file CPU scoring is also available through `Model.predict_to_parquet()`; it uses bounded batches, preserves canonical input order, and writes one `float32` prediction column without collecting the complete result in Python memory.

### Known limitations

* **GPU inference:** prediction currently targets GPU 0 only and uses a bounded single-stream staging path. Multi-stream H2D/kernel overlap remains future optimization work.
* **Multi-GPU:** training and prediction currently use at most one selected GPU per job; multi-GPU execution is not yet implemented.

These are performance and feature limitations, not correctness failures. Future work includes GPU inference pipeline overlap and multi-GPU support.

## Learn more

Project homepage and source: [github.com/mdai001/FerrisBoost](https://github.com/mdai001/FerrisBoost).

See the [Python API How-To](documents/howto.md) for installation details,
Parquet/CSV input formats, validation, early stopping, model persistence,
logging, and the complete parameter reference.

## License

Apache-2.0.
