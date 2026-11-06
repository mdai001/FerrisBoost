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

For CUDA training, FerrisBoost chooses a stable residency plan from the
training shape and memory budget: full residency, hybrid execution, or
streaming. It does not continuously migrate data between these modes during a
training run.
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
uv venv .venv
source .venv/bin/activate
uv pip install ferrisboost
```

### Install from the source tree

FerrisBoost requires Python 3.9 or newer and a Rust toolchain. For development,
create a UV-managed virtual environment and install Maturin into it:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install "maturin>=1.7,<2"

# CPU-only build:
uv run --no-sync maturin develop --release

# CUDA GPU-enabled build (Turing+ / sm_75+; precompiled PTX bundled):
uv run --no-sync maturin develop --release --features "cuda,python,pyo3/extension-module"
```

Use a release build for training or benchmarking. A debug build is
substantially slower.

To build a wheel instead:

```bash
# Release wheel. This applies the required feature set, path remapping,
# stripping, wheel sanitization, and RECORD regeneration.
WHEEL_PATH=$(MATURIN=.venv/bin/maturin FB_PY=.venv/bin/python \
  bash scripts/build_wheel.sh | tail -1)
# Install exactly the newly built artifact, never a stale wildcard match.
uv pip install "$WHEEL_PATH"
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

## Count, positive, and robust regression objectives

Besides `reg:squarederror`, these regression objectives follow XGBoost's
definitions:

| Objective | Use it for | Labels | Link | Default metric |
|---|---|---|---|---|
| `count:poisson` | Non-negative counts | `>= 0` | `exp` | `poisson-nloglik` |
| `reg:gamma` | Strictly positive, skewed targets | `> 0` | `exp` | `gamma-deviance` |
| `reg:tweedie` | Non-negative targets with many zeros and a positive tail (insurance claims, for example) | `>= 0` | `exp` | `tweedie-nloglik@rho` |
| `reg:pseudohubererror` | Targets with outliers | any | identity | `mphe` |

The three log-link objectives (`count:poisson`, `reg:gamma`, `reg:tweedie`)
predict `exp(margin)`, so predictions are positive; `output_margin=True`
returns the margin before `exp`. Labels that violate the requirement in the
table are rejected before training starts.

Two parameters belong to specific objectives:

- `tweedie_variance_power` (default `1.5`, must be in `[1, 2)`) selects the
  Tweedie distribution. Values near `1` behave like Poisson and values near
  `2` like gamma. The default metric is named after it, for example
  `tweedie-nloglik@1.5`.
- `huber_slope` (default `1`, must be non-zero) is the pseudo-Huber slope. The
  `mphe` metric uses the same value.

```python
rng = np.random.default_rng(7)
Xr = rng.normal(size=(5_000, 8)).astype(np.float32)
rate = np.exp(0.5 * Xr[:, 0] - 0.3 * Xr[:, 1])

counts = rng.poisson(rate).astype(np.float32)
model = fb.train(
    {"objective": "count:poisson", "max_depth": 4, "eta": 0.1},
    Xr, label=counts, num_boost_round=100,
)
expected_counts = model.predict(Xr)          # exp(margin), always positive

durations = rng.gamma(2.0, rate / 2.0).astype(np.float32) + 1e-3
model = fb.train(
    {"objective": "reg:gamma", "max_depth": 4, "eta": 0.1},
    Xr, label=durations, num_boost_round=100,
)

noisy = (3.0 * Xr[:, 0] + rng.standard_t(2, size=5_000)).astype(np.float32)
model = fb.train(
    {"objective": "reg:pseudohubererror", "huber_slope": 2.0, "max_depth": 4},
    Xr, label=noisy, num_boost_round=100,
)

claims = np.where(rng.random(5_000) < 0.4, 0.0, rng.gamma(2.0, rate)).astype(np.float32)
model = fb.train(
    {"objective": "reg:tweedie", "tweedie_variance_power": 1.3, "max_depth": 4},
    Xr, label=claims, num_boost_round=100,
)
```

For `count:poisson`, an unset `max_delta_step` becomes `0.7`, as in XGBoost's
learner. It is applied to the Poisson Hessian and to the tree's leaf-weight
clamp (see [Regularization and class imbalance](#regularization-and-class-imbalance)).
Passing `max_delta_step` explicitly, including `0`, overrides that default.

The log-link objectives need a positive `base_score`, because the starting
margin is `log(base_score)`. The estimated default is positive whenever the
labels are not all zero. If every label is `0` (possible for Poisson and
Tweedie), pass an explicit positive `base_score`. See
[Starting prediction](#starting-prediction-base_score).

All of these objectives work with sample weights, monotone constraints,
`alpha`, `max_delta_step`, GPU training, and XGBoost model exchange. On the
CPU and with `gpu_math="exact"`, models match XGBoost exactly when every
feature has fewer distinct values than `max_bin`.

## Regularization and class imbalance

Three parameters have the semantics of their XGBoost namesakes:

- `alpha` (alias `reg_alpha`, default `0`): L1 regularization on leaf weights.
  The leaf weight is soft-thresholded, so small gradients no longer produce a
  leaf value, and the split gain accounts for the L1 term.
- `max_delta_step` (default `0`, meaning no limit): the largest absolute leaf
  weight, before it is scaled by `eta`. A small positive value (1 to 10) can
  stabilize logistic regression on very unbalanced data. For `count:poisson`
  the unset default is `0.7` (see above).
- `scale_pos_weight` (default `1`): multiplies the gradient and Hessian of rows
  whose label is `1`. A common choice for unbalanced binary data is
  `sum(negative) / sum(positive)`.

```python
neg, pos = (y == 0).sum(), (y == 1).sum()

model = fb.train(
    {
        "objective": "binary:logistic",
        "alpha": 0.5,
        "max_delta_step": 1.0,
        "scale_pos_weight": neg / pos,
    },
    X,
    label=y,
    num_boost_round=200,
)
```

Notes:

- `scale_pos_weight` only changes the gradients. Evaluation metrics and the
  weighted quantile sketch use the sample weights alone. It applies to
  `binary:logistic`, `reg:squarederror`, and `reg:gamma`, and is ignored by the
  other objectives, as in XGBoost.
- Predictions from a model trained with `scale_pos_weight` are shifted toward
  the positive class and are no longer calibrated probabilities.
- When `scale_pos_weight` is not `1` and `base_score` is unset, the default
  `base_score` comes from one Newton step instead of the label mean, as in
  XGBoost.
- All three parameters work on CPU and GPU. With `gpu_math="exact"`, GPU models
  stay byte-identical to CPU models.

## Starting prediction (`base_score`)

When `base_score` is not set, FerrisBoost estimates it from the training labels,
as XGBoost 2.0 and later do:

| Objective | Default `base_score` |
|---|---|
| `reg:squarederror`, `binary:logistic`, `count:poisson`, `reg:gamma`, `reg:tweedie` | The label mean (weighted if weights are given) |
| `reg:pseudohubererror` | One Newton step from margin `0` |
| `reg:squarederror`, `binary:logistic`, `reg:gamma` with `scale_pos_weight != 1` | One Newton step from margin `0` |
| `multi:softprob`, `multi:softmax` | Per-class log-frequency margins, centered to mean zero |

Because the estimate depends on the data, default models of `binary:logistic`
and `reg:squarederror` differ from those trained with FerrisBoost 0.0.4, which
started from a fixed `0.5`. They now match XGBoost's default models. To
reproduce a 0.0.4 model, set `"base_score": 0.5` explicitly. An explicit
`base_score` is used as given.

`base_score` is in the space after the objective's transform: a probability for
`binary:logistic` (any value in `[0, 1]`, so that all-0 or all-1 labels train),
a mean for the log-link objectives (it must be positive), and a margin for
multiclass objectives, where a single value is used for every class. The
trained value is stored in the model and used again at prediction time.

## Validation and early stopping

For in-memory data, each validation entry is a
`(features, labels, name)` tuple, or `(features, labels, weights, name)` for a
weighted validation set (see [Sample weights](#sample-weights)):

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

When both the training data and a validation table are pyarrow tables, the
validation columns are matched to the training features by name, so their order
may differ. A validation table with missing or extra columns is rejected. NumPy
arrays have no column names and are matched by position.

Feature values must be finite or missing (`NaN`). The exact largest `float32`
value (`numpy.finfo(numpy.float32).max`) is reserved for split boundaries and is
rejected in training and validation features; rescale such a feature first.
Finite values that overflow while computing gradients or Hessians produce a
training error instead of being silently converted to fixed-point integers.
Values of `±inf` are rejected
during training, as in XGBoost.

Early stopping watches the last configured metric of the last validation set,
as in XGBoost. Choose metrics with `params["eval_metric"]`, either one name or a
list:

```python
model = fb.train(
    {"objective": "binary:logistic", "eval_metric": ["logloss", "auc"]},
    X_train,
    label=y_train,
    num_boost_round=500,
    evals=[(X_valid, y_valid, "valid")],
    early_stopping_rounds=20,  # watches auc, the last metric
)
```

| Metric | Objectives | Definition |
|---|---|---|
| `rmse` | all except multiclass | Root mean squared error on the transformed prediction (probabilities for `binary:logistic`, `exp(margin)` for log-link objectives), as in XGBoost |
| `mae` | all except multiclass | Mean absolute error on the transformed prediction |
| `mape` | all except multiclass | Mean absolute percentage error, `abs((label - prediction) / label)` |
| `mphe` | all except multiclass | Mean pseudo-Huber error; uses `huber_slope` for `reg:pseudohubererror` and `1` otherwise |
| `logloss` | `binary:logistic` | Negative log-likelihood |
| `error` | `binary:logistic` | Error rate at threshold 0.5 (`p > 0.5` counts as positive) |
| `error@t` | `binary:logistic` | Error rate at threshold `t`, for example `error@0.7` |
| `auc` | `binary:logistic` | Area under the ROC curve; tied predictions share a trapezoid. Larger is better, so early stopping maximizes it |
| `aucpr` | `binary:logistic` | Area under the precision-recall curve. Larger is better, so early stopping maximizes it |
| `poisson-nloglik` | `count:poisson` | Negative Poisson log-likelihood |
| `gamma-nloglik` | `reg:gamma` | Negative gamma log-likelihood |
| `gamma-deviance` | `reg:gamma` | Gamma residual deviance |
| `tweedie-nloglik@rho` | `reg:tweedie` | Negative Tweedie log-likelihood at variance power `rho` in `[1, 2)`, for example `tweedie-nloglik@1.5`. The `@rho` part is required |
| `mlogloss` | `multi:softprob`, `multi:softmax` | Multiclass negative log-likelihood |
| `merror` | `multi:softprob`, `multi:softmax` | Multiclass error rate |

The likelihood and deviance metrics are meant for the objective they are listed
with: they are computed on the transformed prediction and assume it is
positive. Multiclass objectives accept only `mlogloss` and `merror`, and those
two are rejected for other objectives. The binary metrics are rejected for
every other objective, with a `ValueError` before training starts. The key in
`evals_result` is the metric name as written, for example
`history["valid"]["tweedie-nloglik@1.5"]`.

Without `eval_metric`, the objective selects the default: `logloss` for
`binary:logistic`, `rmse` for `reg:squarederror`, `poisson-nloglik` for
`count:poisson`, `gamma-deviance` for `reg:gamma`, `tweedie-nloglik@rho` (with
the objective's `tweedie_variance_power`) for `reg:tweedie`, `mphe` for
`reg:pseudohubererror`, and `mlogloss` for both multiclass objectives. With
weighted validation data every metric is weighted the same way XGBoost does it.

Like `xgboost.Booster.predict`, `Model.predict()` uses all trees by default,
including trees built after the best iteration. To predict with the early-
stopping optimum:

```python
best_prediction = model.predict(
    X_valid,
    n_trees=model.best_iteration + 1,
)
```

## Sample weights

Pass per-row weights with `weight=`. They work like XGBoost's
`DMatrix(weight=...)`: each row's gradient and Hessian are multiplied by its
weight, so `min_child_weight` and the split gain both see weighted sums, and
the quantile sketch that chooses bin boundaries is weighted too, so heavily
weighted regions of a feature get finer bins.

```python
w_train = np.where(y_train == 1, 5.0, 1.0)   # e.g. up-weight the rare class
w_valid = np.where(y_valid == 1, 5.0, 1.0)

model = fb.train(
    {"objective": "binary:logistic", "eval_metric": "auc"},
    X_train,
    label=y_train,
    weight=w_train,
    num_boost_round=200,
    evals=[(X_valid, y_valid, w_valid, "valid")],
)
```

For Parquet and CSV input, `weight` names a column, which is excluded from the
features. File validation sets reuse the same label and weight column names.
Headerless CSV takes a zero-based column index, as `label` does:

```python
model = fb.train(params, "train.parquet", label="target", weight="w",
                 evals=[("valid.parquet", "valid")])
model = fb.train(params, "train.csv", label=4, weight=5, header=False)
```

Rules:

- Weights must be finite and non-negative. A row with weight `0` has no effect
  on the trees or the bin boundaries: the model is identical to one trained
  without that row.
- All-ones weights give the same model, byte for byte, as no weights.
- Arrays, Parquet (single file, multiple files, or a reused quantized cache),
  and CSV produce identical weighted bin boundaries and models for the same
  data.
- Weighted metrics are `sum(w * loss) / sum(w)`; weighted `auc` uses
  `w * label` and `w * (1 - label)` as the positive and negative masses.
- CPU and `gpu_math="exact"` GPU training stay byte-identical with weights.
  `gpu_math="fast"` keeps the weights on the GPU, which costs 4 bytes per row;
  the GPU memory planner accounts for it.

**Relation to XGBoost:** when every feature has fewer distinct values than
`max_bin`, the bins are the same and weighted models match XGBoost exactly.
For continuous features, FerrisBoost and XGBoost use different quantile-sketch
implementations, with or without weights, so individual bin boundaries can sit
slightly apart; both are accurate weighted quantiles.

## Monotone constraints

`params["monotone_constraints"]` forces the model's prediction to move in one
direction along chosen features, as XGBoost's parameter of the same name does:
`1` means non-decreasing, `-1` non-increasing, `0` unconstrained.

```python
model = fb.train(
    {"objective": "binary:logistic", "monotone_constraints": (1, -1, 0)},
    X_train,
    label=y_train,
    num_boost_round=200,
)
```

Accepted forms, all with the same meaning:

- a tuple or list by feature position, e.g. `(1, -1, 0)`; a shorter list
  leaves the remaining features unconstrained, a longer one is an error;
- a string `"(1,-1,0)"`;
- a dict by feature name, e.g. `{"age": 1, "debt_ratio": -1}`, for Parquet,
  CSV with a header, or a PyArrow table. NumPy arrays and headerless CSV have
  no feature names, so use the positional form there.

Only the sign of each value matters (`2` behaves like `1`). All-zero
constraints are the same as no constraints, byte for byte.

The constraint applies to the whole model: for any fixed values of the other
features, predictions never decrease (or never increase) as a constrained
feature grows. Missing values follow each split's learned default direction
and are not ordered relative to real values.

The rules are XGBoost's (`hist`, single output): each node keeps lower and
upper bounds on its weight, candidate splits whose child weights go the wrong
way are discarded, and children inherit bounds split at the midpoint of
their weights. Constrained models match XGBoost exactly when every feature has
fewer distinct values than `max_bin`, on CPU and GPU, with or without
weights. CPU and `gpu_math="exact"` GPU models stay byte-identical. The
constraints are not stored in the saved model, as in XGBoost; prediction does
not need them.

## Multiclass classification

Use `multi:softprob` or `multi:softmax` with `num_class`, the number of classes
(at least 2). Labels must be integers in `[0, num_class)`:

```python
rng = np.random.default_rng(3)
Xm = rng.normal(size=(6_000, 10)).astype(np.float32)
score = Xm[:, 0] - Xm[:, 1] + rng.normal(scale=0.5, size=6_000)
ym = np.digitize(score, [-1.0, 0.0, 1.0]).astype(np.float32)   # classes 0..3

model = fb.train(
    {"objective": "multi:softprob", "num_class": 4, "max_depth": 4, "eta": 0.2},
    Xm[:5_000],
    label=ym[:5_000],
    num_boost_round=100,
    evals=[(Xm[5_000:], ym[5_000:], "valid")],   # default metric: mlogloss
    early_stopping_rounds=10,
)
```

Each boosting round builds `num_class` trees, one per class, as XGBoost does.
`model.num_trees` therefore counts trees, not rounds, and equals
`num_class` times the number of rounds. Row and column sampling are drawn per
tree. Sample weights, `monotone_constraints`, `alpha`, `max_delta_step`, GPU
training, and file input work as for the other objectives;
`scale_pos_weight` is ignored.

The default `base_score` is a per-class margin computed from the log of the
(weighted) class frequencies. An explicit `base_score` is a margin that is
used for every class.

Metrics are `mlogloss` (the default) and `merror`; no other metric is
accepted for a multiclass objective.

### Multiclass prediction

| Call | Shape and meaning |
|---|---|
| `predict(X)` with `multi:softprob` | `(n, num_class)` class probabilities |
| `predict(X)` with `multi:softmax` | `(n,)` predicted class index, as `float32` |
| `predict(X, output_margin=True)` | `(n, num_class)` raw margins for either objective |

```python
proba = model.predict(Xm[5_000:])                      # shape (1000, 4)
labels = proba.argmax(axis=1)
margins = model.predict(Xm[5_000:], output_margin=True)
```

`n_trees` counts trees, so truncating to the early-stopping round takes
`num_class` trees per round:

```python
best = model.predict(
    Xm[5_000:],
    n_trees=(model.best_iteration + 1) * 4,   # num_class = 4
)
```

File prediction works as for single-output models:
`model.predict("test.parquet")` returns the same arrays. `predict_to_parquet`
writes one column per class named `prediction_0`, `prediction_1`, ... for
`multi:softprob` and for `output_margin=True`, and a single class-index column
(named by `prediction_column`) for `multi:softmax`:

```python
model.predict_to_parquet("test.parquet", "proba.parquet")
# columns: prediction_0, prediction_1, prediction_2, prediction_3
```

Limitations in this version:

- `predict_to_parquet` for multiclass models builds the whole output in memory
  before writing; it is not streamed in bounded batches like the single-output
  path. Like the single-output path, it refuses to overwrite an existing file,
  requires a `.parquet` or `.pq` extension, and removes a partial file if
  prediction fails.
- The lower-level `predict_array()` method rejects multiclass models; use
  `predict()`.

File prediction reads the input once: each batch is scored for every class
before the next batch is read.

### Multiclass models and XGBoost

Multiclass models are saved with `num_class`, per-tree class ids, and a vector
`base_score`, and are exchanged with XGBoost in both directions: XGBoost loads
a FerrisBoost multiclass model, and FerrisBoost loads one trained by XGBoost.

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
files must use the same label column name (and `weight` column name, if any) as
the training file. Feature columns are matched to the training features by
name, so their order may differ, but a validation file with missing or extra
feature columns is rejected. Headerless CSV files are matched by position.

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

FerrisBoost's memory planner chooses a training plan from the input shape and
memory budget, without manual chunk or batch tuning:
- **Residency plan**: The planner selects full GPU residency, hybrid residency,
  or streaming from the training shape and budget. The plan stays stable during
  training; data is not continuously migrated between modes.
- **`gpu_memory_budget`**: Sets the planner's budget for GPU training memory.
  When omitted, the planner derives a budget from currently free VRAM and keeps
  safety headroom. It is not a hard cap on all GPU memory used by this process
  or by other processes.
- **`gpu_math`**: `"exact"` (default) produces a byte-by-byte exact model
  matching CPU training. `"fast"` is deterministic, but CPU and GPU models are
  not guaranteed to be bit-identical; FerrisBoost does not promise universal
  parity with XGBoost GPU training.
- **Objectives and parameters on GPU**: Every objective (including the
  multiclass ones) and the `alpha`, `max_delta_step`, `scale_pos_weight`,
  sample-weight, and monotone-constraint options work with `device="cuda"`.
  `gpu_math="exact"` stays byte-identical to CPU training for all of them.
- **`gpu_math="fast"` and objectives**: `"fast"` supports `reg:squarederror`,
  `binary:logistic`, `count:poisson`, `reg:gamma`, `reg:tweedie`, and
  `reg:pseudohubererror`. It is deterministic from run to run but not
  byte-identical to CPU. Support for the newer regression objectives is a
  first version; it is correct, and its speed will be optimized in 0.0.6.
- **`gpu_math="fast"` with multiclass**: `gpu_math="fast"` with multiclass objectives is supported (device softmax gradients); optimization continues in 0.0.6.
- **`device_quantize`**: `True` (default) performs fixed-point gradient and
  Hessian quantization on the GPU. Feature binning and feature bins remain on
  the host/source side.
- **Bounded split cache**: A bounded resident histogram/split cache is enabled
  by default and included in the GPU memory plan. Unsupported cases and cache
  overflow use the exact host fallback. This is managed internally and is not
  a public Python parameter.
- **`hist_streams`**: Defaults to automatic (`None`), resolved by the GPU planner based on execution path, block width, and VRAM budget (selects 2 streams for chunked row-major streaming when memory allows, and 1 stream for resident or column-major paths); can be explicitly set to `1` or `2`.
- **`device` selection**: Training currently targets one selected GPU (`device="cuda"` or `device="cuda:N"`). Multi-GPU scaling is in progress (WIP).
- **`max_depth`**: Depth-wise tree growth on GPU is constrained to `0..=14` by partition buffer limits.

## Prediction

`Model.predict()` supports both file paths (the recommended path for large test sets) and in-memory NumPy arrays. The return value is a `float32` `numpy.ndarray`. It is one-dimensional, with shape `(n,)`, except for multiclass models, which return two-dimensional `(n, num_class)` probabilities or margins; `multi:softmax` returns a one-dimensional array of class indices unless `output_margin=True` (see [Multiclass prediction](#multiclass-prediction)). Predictions are in the objective's output space: probabilities for `binary:logistic`, `exp(margin)` for `count:poisson`, `reg:gamma`, and `reg:tweedie`, and the margin itself for the other regression objectives; `output_margin=True` returns the raw margin. Prediction has its own CPU thread control and does not inherit training `nthread`:

```python
prediction = model.predict(data, predict_threads=0)  # process-aware auto
reference = model.predict(data, predict_threads=1)   # explicit serial path
gpu_allowed = model.predict(data, use_gpu=True)      # CUDA build; GPU 0 when worthwhile
```

`predict_threads=0` uses available physical CPU parallelism and keeps small workloads serial; a positive value is an explicit override. Workers own disjoint row slices and each row traverses trees in model order, so output is bit-identical across serial, explicit multi-thread, and auto CPU execution. NumPy scoring releases the Python GIL.

`use_gpu=False` is the default and never initializes CUDA. With `use_gpu=True`, a CUDA-enabled build lazily creates a compact prediction-only SoA model on GPU 0 and reuses it on later calls. It transfers only tree-used features, while the selector also accounts for host projection and launch costs and can keep a batch on CPU. The inference backend is independent of whether the model was trained by FerrisBoost CPU, FerrisBoost GPU, or XGBoost, and independent of training `gpu_math`.

### File prediction

Passing file paths streams records directly through Rust readers without materializing the full dataset in Python host memory:

```python
# Single file (.parquet, .pq, .csv, or .csv.gz):
prediction = model.predict("test.parquet", predict_threads=0)

# Multi-file or partitioned dataset (list of files, directory path, or glob):
prediction = model.predict(["test_part_0.parquet", "test_part_1.parquet"])
prediction = model.predict("data/test_partitions/")
prediction = model.predict("data/test_*.parquet")

# Headerless CSV (positional mode):
prediction = model.predict("test.csv", header=False)
```

```python
# Stream file prediction directly to one Parquet output:
rows_written = model.predict_to_parquet(
    "data/test_partitions/",
    "predictions.parquet",
    output_margin=False,
    n_trees=None,
    predict_threads=0,
    prediction_column="prediction",
)
```

In named mode (Parquet or CSV with headers), FerrisBoost binds columns by the feature names recorded during training. Surplus columns—including leftover label columns—are ignored automatically. In positional mode (`header=False`), the file must have exactly `model.n_features` columns.

File readers validate the full canonical schema before projecting batches to the sorted set of features actually used by the ensemble. This reduces Parquet decode/materialization and GPU H2D width without weakening missing-feature checks. The canonical training/serialized model is immutable; feature remapping exists only in a lazy inference copy, so save/load and XGBoost interchange retain original feature IDs.

Readers keep deterministic file/row order and bounded batches. In auto CPU mode, ingest and scoring share one process-visible CPU budget so `ingest_threads` and `predict_threads` do not each consume the full machine. If both are explicitly set above that budget, FerrisBoost preserves the overrides and emits an oversubscription warning. GPU file prediction keeps input in the Rust batch path and does not first materialize a full NumPy matrix.

`predict_to_parquet()` accepts the same file, file-list, directory, and glob inputs as file-based `predict()`; it does not accept a NumPy input. This first implementation uses CPU scoring and writes one non-null `float32` column to a single `.parquet` or `.pq` file with Snappy compression. It returns the number of rows written. For multiclass models it writes one column per class instead (see [Multiclass prediction](#multiclass-prediction)). The target must not already exist, and an incomplete output created by the call is removed if schema validation, reading, scoring, encoding, or finalization fails.

The direct-output path keeps a bounded two-batch channel between scoring and one ordered Parquet writer thread. The writer can overlap encoding/output with the next input batch while preserving file order and row order. Auto mode shares one process-visible CPU budget across ingest, scoring, and the writer; explicit ingest or prediction thread counts are preserved and produce an oversubscription warning when their total plus the writer exceeds that budget.

### In-memory array prediction

Prediction also accepts a two-dimensional NumPy `float32` array with the same number and order of features used for training:

```python
X_test = np.asarray(X_test, dtype=np.float32)

prediction = model.predict(X_test)
raw_margin = model.predict(X_test, output_margin=True)
first_20_trees = model.predict(X_test, n_trees=20)
```

C-contiguous input uses the direct row-major fast path. Fortran-contiguous and strided `float32` arrays are also interpreted correctly. Non-`float32` arrays, pandas `DataFrame`s, or PyArrow `Table`s must be converted explicitly with `np.asarray(..., dtype=np.float32)`.

For GPU prediction, the first eligible call includes CUDA context initialization and one model upload; subsequent calls reuse the cached model. Staging memory remains per-call and bounded to available VRAM. GPU 0 is the only inference target in this version; multi-GPU prediction is intentionally not enabled.

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

Model save/load converts independent trees in parallel with fixed-index ordering,
keeps single-document JSON serialization/parsing sequential, and uses bounded
positional file I/O. `model_io_threads=0` selects system-adaptive workers and is
independent of file ingest and training threads. Small models automatically use
one conversion worker; override per operation when needed:

```python
model.save_model("model.json", model_io_threads=2)
restored = fb.Model.load_model("model.json", model_io_threads=2)
```

Save/load and file ingest emit concise progress on stderr every few seconds.
The start record shows the requested mode and resolved count, for example
`ingest_threads=auto workers=8` or `model_io_threads=auto workers=1`. The
model record reports the existing positional-I/O worker count; ordered tree
conversion independently bounds the same request by tree work. Auto uses CPUs
actually available to the process (including affinity/cgroup limits),
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
| `objective` | `binary:logistic` | `binary:logistic`, `reg:squarederror`, `count:poisson`, `reg:gamma`, `reg:tweedie`, `reg:pseudohubererror`, `multi:softprob`, `multi:softmax` |
| `num_class` | `None` | Number of classes, at least 2. Required for `multi:softprob` and `multi:softmax`, ignored otherwise. Labels are integers in `[0, num_class)` |
| `tweedie_variance_power` | `1.5` | Variance power of `reg:tweedie`, in `[1, 2)` |
| `huber_slope` | `1.0` | Slope of `reg:pseudohubererror` and its `mphe` metric; must be non-zero |
| `max_depth` | `6` | Depth-wise tree growth (`0..=14`) |
| `max_bin` | `255` | Must be between 1 and 255 |
| `eta` | `0.3` | Alias: `learning_rate` |
| `lambda` | `1.0` | Alias: `reg_lambda` |
| `gamma` | `0.0` | Minimum split loss |
| `min_child_weight` | `1.0` | Minimum child Hessian |
| `alpha` | `0.0` | L1 regularization on leaf weights; alias: `reg_alpha` |
| `max_delta_step` | `0.0` | Maximum absolute leaf weight before `eta` (`0` = no limit); can help logistic regression with very unbalanced classes. For `count:poisson` an unset value becomes `0.7`, for the tree and the objective alike |
| `scale_pos_weight` | `1.0` | Multiplies the gradient weight of rows with label `1`; metrics and binning are not affected. Used by `binary:logistic`, `reg:squarederror`, and `reg:gamma`, ignored by the other objectives. When it is not `1` and `base_score` is unset, the default `base_score` comes from one Newton step, as in XGBoost |
| `subsample` | `1.0` | Deterministic per-tree row sampling in `(0, 1]`; all rows still receive the learned tree |
| `colsample_bytree` | `1.0` | Deterministic per-tree feature sampling in `(0, 1]`; column-major GPU histograms use compact selected-feature output |
| `cols_per_block` | automatic | Positive integer; normally leave unset |
| `nthread` | `0` | Zero selects physical cores; alias: `n_jobs` |
| `device` | `"cpu"` | Compute device: `"cpu"`, `"cuda"`, or `"cuda:N"` |
| `gpu_memory_budget` | automatic | Positive integer planner budget for GPU training memory; auto derives from free VRAM with safety headroom |
| `gpu_math` | `"exact"` | `"exact"` (byte-by-byte identical to CPU) or deterministic `"fast"` (CPU/GPU models need not be bit-identical); see [Train on GPU](#train-on-gpu) for objective support |
| `device_quantize` | `True` | Fixed-point gradient/Hessian quantization on GPU; feature binning remains on host/source |
| `hist_streams` | automatic | Planned by GPU planner based on execution path, block width, and VRAM; override with `1` or `2` |
| `hist_nodes_per_batch` | automatic | Multi-node merge batch size (`1..=32`) |
| `ingest_threads` | `0` (auto) | Process-visible CPUs, independent work, and memory bound file ingest; independent of `nthread` |
| `model_io_threads` | `0` (auto) | Work/memory-bound ordered tree conversion and model-file I/O; JSON parse/dump remains serial |
| `seed` | `0` | Alias: `random_state` |
| `base_score` | estimated | Starting prediction in the objective's output space (a probability for `binary:logistic`, a positive mean for log-link objectives, a per-class margin for multiclass). If unset, it is estimated from the training labels as XGBoost 2.0+ does; see [Starting prediction](#starting-prediction-base_score) |
| `cache_budget_bytes` | automatic | Parquet quantized-cache memory budget |
| `cache_path` | `None` | Persistent Parquet cache directory |
| `eval_metric` | objective default | One name or a list; see the metric table under [Validation and early stopping](#validation-and-early-stopping). Early stopping watches the last one |
| `monotone_constraints` | `None` | `1` / `-1` / `0` per feature: tuple, list, `"(1,-1,0)"` string, or `{feature_name: dir}` dict |

Unknown parameters raise `ValueError`; they are not silently ignored.

The following familiar XGBoost parameters are recognized but not implemented:

- `max_leaves`
- `tree_method` (FerrisBoost currently has only histogram training)
- `num_parallel_tree`

Call options accepted by `fb.train()`, `fb.Model.save_model()`, and `fb.Model.load_model()`:
- `quiet`: Set `True` to suppress routine progress logging on stderr.
- `header`: Set `False` for headerless CSV files using positional column indices.
- `weight` (`fb.train()` only): per-row sample weights; an array for array input, or a column name (column index for headerless CSV) for file input.

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

### Labels do not fit the objective

Training validates labels before the first round: `binary:logistic` needs
labels in `[0, 1]`, `count:poisson` and `reg:tweedie` need non-negative labels,
`reg:gamma` needs strictly positive labels, and multiclass objectives need
integer labels in `[0, num_class)`.

### A multiclass objective is missing `num_class`

`multi:softprob` and `multi:softmax` need `num_class >= 2` in the parameters.
`num_class` must also be larger than the largest label.

### A metric does not fit the objective

Binary metrics (`logloss`, `error`, `error@t`, `auc`, `aucpr`) require
`binary:logistic`. `mlogloss` and `merror` require a multiclass objective, and
multiclass objectives accept only those two. `tweedie-nloglik` must be written
with its variance power, such as `tweedie-nloglik@1.5`.

### All labels are zero with a log-link objective

`count:poisson`, `reg:gamma`, and `reg:tweedie` start from `log(base_score)`, so
`base_score` must be positive. The label mean is zero when every label is `0`,
so pass an explicit positive value, for example `"base_score": 0.1`.

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
uv run --no-sync maturin develop --release --features "cuda,python,pyo3/extension-module"
```

### Header and label specification mismatch

- When `header=True` (default), `label` must be a column name string (e.g. `label="target"`).
- When `header=False`, `label` must be a 0-based integer column index (e.g. `label=4`).
- `header=False` is only supported for CSV inputs; Parquet schemas always supply column names.

### Positional CSV prediction column count mismatch

In positional mode (`header=False`), test CSV files must contain exactly `model.n_features` columns. In contrast to named mode (which can safely filter and ignore surplus label columns), positional mode binds by column position, so a leftover training label column in the test file will cause a column count mismatch error.

### max_depth is out of range

`max_depth` must be an integer between 0 and 14 (`0 <= max_depth <= 14`). Depth 0 corresponds to a root-only tree; the upper limit of 14 is constrained by the partition buffer on GPU.
