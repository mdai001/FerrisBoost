# FerrisBoost Python API How-To

FerrisBoost is a histogram-based gradient-boosted tree library with NumPy,
PyArrow, Parquet, and CSV training inputs. Its parameter names and JSON model
format are designed to be familiar to XGBoost users, but FerrisBoost does not
use a `DMatrix` object.

The published 0.0.8 packages support CPU training and CUDA-enabled builds. This
guide also documents APIs implemented for the unreleased 0.0.9 development
candidate; sections marked **0.0.9 development** are not available from the
current PyPI wheels or Rust crate. Release and cross-platform validation are
still pending. Select one GPU with `device="cuda"` or `device="cuda:N"`;
Multi-GPU support is future 1.0 work.

## Choose the input form

Use the source-oriented API as the default:

- **Large-data training, especially on GPU: pass a file path.** Parquet is the
  preferred format. A path lets the planner inspect the schema and choose
  quantization, block width, resident versus streaming execution, and spill
  behavior without materializing the full table first.
- **In-memory convenience: NumPy, PyArrow, and pandas are optional.** Use them
  when the data is already in memory and comfortably fits there.

For CUDA training, FerrisBoost chooses a stable residency plan from the
training shape and memory budget: full residency, hybrid execution, or
streaming. It does not continuously migrate data between these modes during a
training run.

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
Missing feature values may be represented by `NaN` (or by a sentinel you
declare with `missing`, see [Missing values](#missing-values-and-the-missing-sentinel));
labels must not contain `NaN`.

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

### Additional scalar objectives (0.0.9 development)

These scalar objectives are available in development builds on CPU and CUDA
with `gpu_math="exact"` or `gpu_math="fast"`. They are not in the published
0.0.8 packages.

| Objective | Use it for | Labels | Prediction | Default metric |
|---|---|---|---|---|
| `binary:logitraw` | Binary classification when raw margins are useful | Values in `[0, 1]` | Raw margin | `logloss` (applies sigmoid); `rmse` uses the raw margin |
| `binary:hinge` | Binary classification decisions without probabilities | Values in `[0, 1]` | Class decision `0` or `1` (`margin > 0` is positive) | `error` |
| `reg:squaredlogerror` | Regression with squared log error | Finite and `> -1` | Raw margin | `rmsle` |
| `reg:absoluteerror` | Regression with absolute error | Any finite value | Identity | `mae` |
| `reg:expectileerror` | Asymmetric squared regression | Any finite value | Identity | `expectile` |

`binary:logitraw` uses logistic gradients but returns the raw margin from
`predict`; `predict_proba` in the sklearn classifier applies sigmoid. Its
`logloss` metric also applies sigmoid, while `rmse` measures the raw margin.
For `binary:hinge`, sklearn `predict_proba` raises `ValueError` because the
objective produces class decisions, not probabilities.

`reg:squaredlogerror` requires labels greater than `-1`; its predictions stay
in margin space. The objective's gradient guards its logarithm near `-1`, but
the `rmsle` metric requires predictions greater than `-1` to compute `log1p`.

`reg:absoluteerror` uses an adaptive, round-level smooth majorization of MAE
with a weighted residual scale, then passes gradient/Hessian pairs to the
regular tree fitter. It does not choose leaves by taking a weighted median.
When `base_score` is unset, its starting score is one Newton correction from
the weighted label mean.

`reg:expectileerror` requires the scalar parameter `expectile_alpha` in
`[0, 1]`; arrays are not supported. For residual `prediction - label`, the
positive-residual side is weighted by `1 - expectile_alpha` and the
negative-residual side by `expectile_alpha`. Its default `expectile` metric
uses the same alpha. Its estimated starting score is one Newton correction
from the weighted label mean.

### Probability regression (0.0.9 development)

Use `objective="reg:logistic"` for continuous targets in `[0, 1]`. Predictions
are probabilities; `output_margin=True` returns raw margins. It shares the
native logistic gradients and CPU/CUDA exact/fast paths with `binary:logistic`,
but keeps its own objective name when saved and uses probability-space `rmse`
as its default evaluation metric. Explicit `base_score` values also lie in
`[0, 1]`. Use `FerrisBoostRegressor(objective="reg:logistic")` with sklearn.

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

### Quantile regression

`reg:quantileerror` fits conditional quantiles and requires `quantile_alpha`
in `[0, 1]` (XGBoost has no default either): one value, or an ascending list
for several quantiles in one model (see [Several quantiles in one
model](#several-quantiles-in-one-model)):

```python
upper = fb.train(
    {"objective": "reg:quantileerror", "quantile_alpha": 0.9, "max_depth": 4},
    Xr, label=noisy, num_boost_round=100,
    evals=[(Xr, noisy, "train")],              # default metric: quantile
)
q90 = upper.predict(Xr)                        # identity link
```

This is the XGBoost 3.4.1 objective, which is a smoothed pinball loss rather
than the textbook subgradient. Every round it computes one global scale from
all rows' current predictions, so its gradients are not a function of a single
row. The starting prediction (`base_score`, when unset) is the `quantile_alpha`
quantile of the labels (linear interpolation without weights, a step function
with weights). The `quantile` metric is the pinball loss at the same alpha, and
it is the default metric for this objective. On the CPU and with
`gpu_math="exact"`, models match XGBoost exactly when every feature has fewer
distinct values than `max_bin` and XGBoost runs with `nthread=1`. XGBoost sums
the global scale in a thread-dependent order; FerrisBoost sums it serially, so
its models do not depend on `nthread`.

`gpu_math="fast"` is supported: the scale is reduced on the device in a fixed
shape (fixed block count, per-block partial sums, one final single-block
reduction in a fixed order, no float atomics), so a fast run is deterministic
from run to run, but it is not byte-identical to the CPU (the summation order
and the device `tanh` differ); `gpu_math="exact"` stays byte-identical to the
CPU.

#### Several quantiles in one model

Pass an ascending list to train one model with one output per quantile, as
XGBoost does with `quantile_alpha=[0.1, 0.5, 0.9]`:

```python
model = fb.train(
    {"objective": "reg:quantileerror", "quantile_alpha": [0.1, 0.5, 0.9], "max_depth": 4},
    Xr, label=noisy, num_boost_round=100,
    evals=[(Xr, noisy, "train")],              # one averaged "quantile" metric
)
q = model.predict(Xr)                          # shape (n, 3), one column per alpha
```

Each round builds one tree per quantile (`num_class`-style, XGBoost's
`one_output_per_tree`), so `model.num_trees` is `3 * num_boost_round`. Every
quantile has its own global scale and its own starting prediction: the
`quantile_alpha_j` quantile of the labels (an explicit `base_score` is used for
every quantile). `predict` sorts each row ascending so quantiles never cross,
exactly as XGBoost does; `output_margin=True` returns the unsorted per-quantile
scores. The `quantile` metric averages the pinball loss over all rows and
quantiles (and the early-stopping rule uses it); other metrics are rejected for
several quantiles because the prediction and the label have different shapes.
`quantile_alpha` values must be in `[0, 1]` and ascending (repeats are allowed),
and at most 64 are accepted. SHAP values, `pred_leaf`, `iteration_range`
(rounds, `num_boost_round`-style) and `get_score` work as for multiclass models:
`pred_contribs` returns `(n, n_quantiles, n_features + 1)`. The model JSON
stores `num_target` and a vector `base_score`, and XGBoost loads it and
FerrisBoost loads XGBoost's. On the CPU and with `gpu_math="exact"` the models
match XGBoost exactly (same conditions as above); `gpu_math="fast"` keeps one
prediction column per quantile on the device (the planner counts them) and
reduces each quantile's scale in turn in the same fixed shape. `predict_to_parquet`
writes one column per quantile, `prediction_0`, `prediction_1`, ...

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
  `binary:logistic`, `reg:logistic`, `reg:squarederror`, and `reg:gamma`, and is ignored by the
  other objectives, as in XGBoost.
- Predictions from a model trained with `scale_pos_weight` are shifted toward
  the positive class and are no longer calibrated probabilities.
- When `scale_pos_weight` is not `1` and `base_score` is unset, the default
  `base_score` comes from one Newton step instead of the label mean, as in
  XGBoost.
- All three parameters work on CPU and GPU. With `gpu_math="exact"`, GPU models
  stay byte-identical to CPU models.

## Survival objectives

Available in 0.0.9 development builds. The losses and derivatives are native
Rust implementations. Both objectives support CPU and CUDA exact training;
CUDA fast currently raises an error. Predictions use `exp(raw_margin)`.

`survival:cox` accepts signed times: positive labels indicate events, while
nonpositive labels indicate censoring at the absolute time. The initial risk
score is `1`. Tied events use grouped Breslow risk sets. Sample weights enter
both the risk sets and event counts; zero-weight rows have no effect on the
loss. This weighted, tied-event training convention differs from XGBoost's
convention, so model interchange does not imply identical training results.
The default metric is `cox-nloglik`, normalized by total event weight.

`survival:aft` accepts separate lower and upper time bounds. Lower bounds
must be finite and nonnegative; upper bounds must be positive and may be
`+inf`. Equal bounds specify an exact positive event time; `(0, upper)` is
left-censored, `(lower, +inf)` is right-censored, and finite unequal bounds
specify interval censoring. `(0, +inf)` is uninformative.

```python
model = fb.train(
    {"objective": "survival:aft", "aft_loss_distribution": "normal",
     "aft_loss_distribution_scale": 1.0},
    X, label=(lower, upper),
    evals=[(X_valid, (lower_valid, upper_valid), "valid")],
)
```

The explicit form is `label_lower_bound=lower, label_upper_bound=upper`.
For files, use column names (or indices with `header=False`); both columns
are excluded from features. `label="lower", label_upper_bound="upper"` is
also accepted for file input. File evaluation sets use the same column
selectors. The sklearn regressor accepts targets with shape `(n_samples, 2)`; its
`score` returns negative AFT log likelihood, including optional sample weights.
For Cox models, `score` returns negative Cox log likelihood. Both use the
parameters stored in the fitted model and are maximized during model selection.

`aft_loss_distribution` accepts `normal`, `logistic` or `extreme`; the scale
must be finite and positive. The initial predicted time is `0.5`. The default
metric is `aft-nloglik`; `interval-regression-accuracy` is also supported and
is maximized by early stopping. Point-label and ordinary custom metrics are
rejected for AFT to avoid silently discarding censoring bounds. AFT gradients
are clipped to `[-15, 15]`, Hessians to `[1e-16, 15]`, and reported likelihoods
use a `1e-12` floor. Exact-event likelihoods are densities and may exceed
`1`, so their negative log likelihood may be negative. Continuation keeps
the AFT distribution and scale, or the expectile alpha, fixed; changing
these objective parameters raises an error.

## Starting prediction (`base_score`)

When `base_score` is not set, FerrisBoost estimates it from the training labels,
as XGBoost 2.0 and later do:

| Objective | Default `base_score` |
|---|---|
| `reg:squarederror`, `binary:logistic`, `reg:logistic`, `binary:logitraw`, `reg:squaredlogerror`, `count:poisson`, `reg:gamma`, `reg:tweedie` | The label mean (weighted if weights are given) |
| `reg:pseudohubererror`, `binary:hinge` | One Newton step from margin `0` |
| `reg:absoluteerror`, `reg:expectileerror` | One Newton correction from the weighted label mean |
| `reg:squarederror`, `binary:logistic`, `reg:logistic`, `binary:logitraw`, `reg:squaredlogerror`, `reg:gamma` with `scale_pos_weight != 1` | One Newton step from margin `0` |
| `multi:softprob`, `multi:softmax` | Per-class log-frequency margins, centered to mean zero |
| `rank:ndcg`, `rank:pairwise`, `rank:map` | One Newton step from score `0` on the objective's gradients (close to `0`) |

Because the estimate depends on the data, default models of `binary:logistic`
and `reg:squarederror` differ from those trained with FerrisBoost 0.0.4, which
started from a fixed `0.5`. They now match XGBoost's default models. To
reproduce a 0.0.4 model, set `"base_score": 0.5` explicitly. An explicit
`base_score` is used as given.

`base_score` is in the space after the objective's transform: a probability for
`binary:logistic` and `reg:logistic` (any value in `[0, 1]`, so that all-0 or all-1 labels train),
a raw margin for `binary:logitraw` and `binary:hinge`, a mean for the log-link
objectives (it must be positive), and a margin for multiclass objectives, where
a single value is used for every class. Other scalar objectives use their
identity prediction space. The
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
rejected in training, validation and prediction features alike, so the two can
never route it differently; rescale such a feature first.
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
| `rmse` | single-output objectives except AFT and ranking | Root mean squared error on the transformed prediction (probabilities for `binary:logistic` and `reg:logistic`, `exp(margin)` for log-link objectives); `binary:logitraw` uses its raw margin, as in XGBoost |
| `mae` | single-output objectives except AFT and ranking | Mean absolute error on the transformed prediction; the default for `reg:absoluteerror` |
| `rmsle` | `reg:squaredlogerror` | Root mean squared error between `log1p` predictions and labels; predictions must be greater than `-1` |
| `expectile` | `reg:expectileerror` | Expectile loss using `expectile_alpha` |
| `mape` | single-output objectives except AFT and ranking | Mean absolute percentage error, `abs((label - prediction) / label)` |
| `mphe` | single-output objectives except AFT and ranking | Mean pseudo-Huber error; uses `huber_slope` for `reg:pseudohubererror` and `1` otherwise |
| `logloss` | `binary:logistic`, `reg:logistic`, `binary:logitraw` | Negative log-likelihood; `binary:logitraw` margins are passed through sigmoid |
| `error` | `binary:logistic`, `reg:logistic`, `binary:logitraw`, `binary:hinge` | Error rate at threshold 0.5 for probability outputs; hinge uses its class decisions |
| `error@t` | `binary:logistic`, `reg:logistic`, `binary:logitraw`, `binary:hinge` | Error rate at threshold `t`, for example `error@0.7` |
| `auc` | `binary:logistic`, `reg:logistic`, `binary:logitraw`, ranking objectives | Area under the ROC curve; tied predictions share a trapezoid. Larger is better, so early stopping maximizes it. With query groups: mean over the groups, see [Learning to rank](#learning-to-rank) |
| `aucpr` | `binary:logistic`, `reg:logistic`, `binary:logitraw`, ranking objectives | Area under the precision-recall curve. Larger is better, so early stopping maximizes it. With query groups: mean over the groups |
| `poisson-nloglik` | `count:poisson` | Negative Poisson log-likelihood |
| `gamma-nloglik` | `reg:gamma` | Negative gamma log-likelihood |
| `gamma-deviance` | `reg:gamma` | Gamma residual deviance |
| `tweedie-nloglik@rho` | `reg:tweedie` | Negative Tweedie log-likelihood at variance power `rho` in `[1, 2)`, for example `tweedie-nloglik@1.5`. The `@rho` part is required |
| `quantile` | `reg:quantileerror` | Pinball loss at `quantile_alpha` (the mean over all quantiles of a multi-quantile model); the objective's default metric |
| `cox-nloglik` | `survival:cox` | Weighted grouped Breslow negative log likelihood per event weight |
| `aft-nloglik` | `survival:aft` | Mean negative interval/event log likelihood |
| `interval-regression-accuracy` | `survival:aft` | Fraction of predictions inside the target intervals; higher is better |
| `mlogloss` | `multi:softprob`, `multi:softmax` | Multiclass negative log-likelihood |
| `merror` | `multi:softprob`, `multi:softmax` | Multiclass error rate |
| `ndcg`, `ndcg@k`, `ndcg@k-` | ranking objectives | Mean NDCG over the query groups (group weights if given); `@k` truncates, `-` scores a group without relevant documents as 0; see [Learning to rank](#learning-to-rank) |
| `map`, `map@k`, `map@k-` | ranking objectives | Mean average precision (binary labels) |
| `pre@k` | ranking objectives | Precision at `k` (binary labels) |

The likelihood and deviance metrics are meant for the objective they are listed
with: they are computed on the transformed prediction and assume it is
positive. Multiclass objectives accept only `mlogloss` and `merror`, and those
two are rejected for other objectives. Binary metrics are accepted for
`binary:logistic`, `reg:logistic`, `binary:logitraw`, and `binary:hinge` as
applicable (`auc` / `aucpr` are also accepted for ranking), and are
rejected for other objectives with a `ValueError` before training starts. The key in
`evals_result` is the metric name as written, for example
`history["valid"]["tweedie-nloglik@1.5"]`.

Without `eval_metric`, the objective selects the default: `logloss` for
`binary:logistic` and `binary:logitraw`, `error` for `binary:hinge`, `rmse` for `reg:logistic` and `reg:squarederror`, `rmsle` for `reg:squaredlogerror`, `mae` for `reg:absoluteerror`, `expectile` for `reg:expectileerror`, `poisson-nloglik` for
`count:poisson`, `gamma-deviance` for `reg:gamma`, `tweedie-nloglik@rho` (with
the objective's `tweedie_variance_power`) for `reg:tweedie`, `mphe` for
`reg:pseudohubererror`, `quantile` for `reg:quantileerror`, `mlogloss` for both multiclass objectives, and `ndcg@32` (`map@32` for `rank:map`) for the ranking objectives. With
weighted validation data the built-in metrics apply their documented weighting rules.

Like `xgboost.Booster.predict`, `Model.predict()` uses all trees by default,
including trees built after the best iteration. To predict with the early-
stopping optimum:

```python
best_prediction = model.predict(
    X_valid,
    n_trees=model.best_iteration + 1,
)
```

## Custom evaluation metrics

Available in 0.0.9 development builds. Wrap a batch function in `fb.CustomMetric`
and pass it through `eval_metric`, alone or in a list with built-in metric names:

```python
def raw_rmse(predictions, labels, sample_weight, group_ptr):
    squared_error = (predictions - labels) ** 2
    return float(np.sqrt(np.average(squared_error, weights=sample_weight)))

metric = fb.CustomMetric("raw_rmse", raw_rmse)
model = fb.train(
    {"objective": "reg:squarederror", "eval_metric": ["rmse", metric]},
    X_train, label=y_train, num_boost_round=200,
    evals=[(X_valid, y_valid, "valid")], early_stopping_rounds=10,
)
```

The function receives **raw margins**, including any base margin. Predictions
have shape `(rows,)` for one output and `(rows, outputs)` for multiclass or
multi-quantile models. Labels and predictions are `float32` arrays. Weights are
`None` or the validation weights: per row for ordinary objectives, per query
group for ranking. Query groups are `None` or a `uint32` array of row offsets.
The function applies any weighting it needs and returns one finite scalar;
FerrisBoost does not weight that scalar again.

Set `greater_is_better=True` for a metric to maximize. The last metric of the
last validation set still controls early stopping. Rust owns evaluation
ordering, result validation, history and stopping; Python is called once per
metric and validation set each round. Built-in metrics stay native. Callback
exceptions propagate with their original Python type and stop the round before
it is recorded.

Input arrays are owned snapshots that the callback can retain safely. Creating
them has a copy cost; Python metric callbacks run on the host and are not
compiled into CUDA. The descriptor also works with sklearn estimators.
`FerrisBoostRanker.score` negates a lower-is-better custom metric so that a
larger sklearn score is always better. Callback code is not stored in model
files; supply the metric again when continuing training.

Rust users implement the public `ferrisboost::metrics::Metric` trait and pass
`Box<dyn Metric>` values through `TrainConfig.eval_metrics`. The fallible
`try_eval_grouped` hook can return evaluation errors to training; implement
weight/group support explicitly when the metric needs it.

## Custom objectives

Available in 0.0.9 development builds. A custom objective supplies an
unweighted gradient and Hessian for each row. Pass a descriptor through
`custom_objective`; `objective="reg:squarederror"` is required alongside it
to define identity-link prediction and the default evaluation metric. The `fb.train` default remains
`binary:logistic`.

```python
def squared_error(raw_margin, labels):
    grad = np.ascontiguousarray(raw_margin - labels, dtype=np.float32)
    hess = np.ones(labels.shape, dtype=np.float32)
    return grad, hess

custom_obj = fb.CustomObjective("squared_error", squared_error)
model = fb.train(
    {"objective": "reg:squarederror", "custom_objective": custom_obj},
    X_train, label=y_train, num_boost_round=200,
    evals=[(X_valid, y_valid, "valid")],
)
```

The callback receives `(raw_margin, labels)` as owned one-dimensional
`float32` NumPy snapshots and returns exactly `(gradient, hessian)`: two
one-dimensional, C-contiguous `float32` arrays with one value per row. Return
unweighted values; FerrisBoost validates finite gradients and non-negative
finite Hessians, applies sample weights once, then samples rows for the tree.
The custom objective uses the shared Newton leaf rules. Its default starting
margin is `0` (set `base_score` to override it), predictions use the identity
link, and its default metric is RMSE.

Python runs the callback once per boosting round for all training rows
together; callback code is not compiled. The native extension point is the public Rust
`ferrisboost::custom_objective::CustomObjective` trait, whose `gradients` method fills a
`GradPair` slice. The Python adapter implements that same contract. Support is
limited to `reg:squarederror` on CPU and CUDA with `gpu_math="exact"`; other
named objectives, fast GPU math, and `scale_pos_weight` values other than `1`
are rejected.

Model JSON stores the custom objective name, not callback code. Loaded models
can predict without the callback. Continuing training with `init_model`
requires passing a custom objective descriptor with the same name again; the
name does not verify that its callback code is equivalent, which is the user's
responsibility.

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

## Starting margins (`base_margin`)

`base_margin` gives every row its own starting raw margin, like XGBoost's
`DMatrix(base_margin=...)`. Training starts each row's prediction from it
instead of from `base_score`; an offset (log exposure for Poisson counts, the
output of an earlier model, ...) therefore enters the model without being a
feature.

```python
model = fb.train(
    {"objective": "count:poisson"},
    X_train, label=y_train, base_margin=np.log(exposure_train),
    evals=[(X_valid, y_valid, "valid")],
    base_margin_eval_set=[np.log(exposure_valid)],
    num_boost_round=100,
)
pred = model.predict(X_new, base_margin=np.log(exposure_new))
```

- Shape: `(n,)` (or `(n, 1)`) for one output, `(n, num_class)` for
  `multi:softprob` / `multi:softmax`. Lists, pandas Series and PyArrow arrays work.
  Values are raw margins (before the sigmoid, exp or softmax) and must be finite.
- It **replaces** the intercept; it is not added to it. As in XGBoost the
  intercept is still estimated from the labels (or taken from `base_score`) and
  stored in the model; the margin only changes where each row starts. A
  `predict` without `base_margin` therefore starts from that intercept, so give
  prediction the same kind of margin you trained with.
- Validation sets take `base_margin_eval_set`: a list parallel to `evals` with a
  margin or `None` per set.
- Prediction: `Model.predict(..., base_margin=)` for arrays, DataFrames and
  tables; `output_margin=True`, `iteration_range`, and the transformed output
  all apply on top of it. `pred_contribs` / `pred_interactions` put the margin
  in the bias column, as XGBoost does (rows still sum to the raw prediction);
  `pred_leaf` ignores it.
- Files: `base_margin` names a column (a list of `num_class` names for
  multiclass; a zero-based index for headerless CSV training). The column is
  excluded from the features, like `label` and `weight`, and file validation
  sets read the same column from their own files. `Model.predict` and
  `predict_to_parquet` read the column next to the features, which needs a model
  with feature names (trained from a table or file).

  ```python
  model = fb.train(params, "train.parquet", label="y", base_margin="offset",
                   evals=[("valid.parquet", "valid")])
  model.predict("new.parquet", base_margin="offset")
  ```
- The margin lives in a plain per-row array next to the labels and weights; it is
  not part of the column-blocked feature storage. With `gpu_math="fast"` the
  device prediction buffer is initialized directly from it (the buffer exists
  anyway, so the memory plan is unchanged). CPU and `gpu_math="exact"` GPU
  training stay byte-identical with margins; fast mode stays deterministic.
- `use_gpu=True` prediction takes `base_margin` too (arrays and files, one value per
  row and output, so multiclass and multi-quantile models work): the device starts
  each row from its margin instead of the intercept and adds leaf values in tree
  order, so the result equals the CPU result bit for bit. The per-row margins
  are staged in the same bounded device buffers as the features.

With lossless bins the margins, trees and SHAP values are bit-identical to XGBoost
trained with `DMatrix(x, base_margin=...)`.

## Continue training (`init_model`)

`init_model` continues an existing model: the new trees are fitted to the
residual of the old ones. XGBoost code that passes `xgb_model=` works unchanged
(it is the same argument under XGBoost's name; give one or the other).

```python
first = fb.train(params, X, label=y, num_boost_round=100)
more = fb.train(params, X_new, label=y_new, num_boost_round=50, init_model=first)
assert more.num_trees == 150          # 100 old trees + 50 new ones

fb.train(params, X, label=y, num_boost_round=50, init_model="model.json")
```

`init_model` is a FerrisBoost `Model`, a path to a FerrisBoost **or XGBoost** model
JSON, or an `xgboost.Booster`. The model passed in is not modified.

- `num_boost_round` counts the **new** rounds (XGBoost 3.4.1 does the same); the
  result holds the old trees followed by the new ones.
- Training starts from the old model's prediction on the new training data and on
  every validation set, and keeps its `base_score` / per-class margins. A given
  `base_margin` replaces the old intercept (the start is `base_margin` plus the
  old trees, as in XGBoost). The old trees are scored on the raw feature values
  through their stored float thresholds, so they do not depend on bin boundaries.
- The bin boundaries of the new trees are sketched afresh from the new data
  (XGBoost does the same). Categorical features and multiclass models work; a
  model with stored category names codes the new data with them (see
  [Named categories](#named-categories)).
- The objective follows the model: leave `params["objective"]` (and `num_class`) out
  to use the model's. A different objective, `num_class`, feature count, feature
  names (a DataFrame / table is bound to the model's names, in any column order;
  numpy data continues by position) or categorical feature types raise
  `ValueError`. An explicit `base_score` that differs from the model's is an error.
- `evals_result` and the round numbers passed to callbacks cover the new rounds
  only. Early stopping reports `best_iteration` as the round in the **whole**
  model, so `predict(iteration_range=(0, best_iteration + 1))` is the best model,
  as in XGBoost. A `best_iteration` already stored in the continued model stays if
  early stopping is not used this time.
- Files work too (`fb.train(params, "more.parquet", label="y", init_model=m)`),
  also from an **XGBoost** JSON or `Booster`. File columns are bound to the model
  **by name**, so the model needs feature names: a FerrisBoost model trained from
  a DataFrame, table, Parquet or headered CSV has them, and so does an XGBoost
  model trained with `feature_names` (a DataFrame, or `DMatrix(feature_names=...)`).
  The file's column order is free; a missing feature column, or a surplus column
  other than the label / weight / `qid` / `base_margin` columns, is an error. An
  XGBoost model **without** feature names raises `ValueError` for file input: only
  a positional binding would be left, and that mis-scores silently when the file's
  columns are in another order. Continue it from an array (you own the column
  order), or set `feature_names` on the `Booster`, save it again and pass that.
  The same rule governs `predict`, `predict_to_parquet` and explanations on files.
- `sklearn`: `fit(X, y, init_model=)` (or `xgb_model=`) accepts a `Model`, another estimator, a path
  or a `Booster`, and adds `n_estimators` new rounds.

What holds: on lossless bins, continuing is bit-identical to XGBoost's own
continuation (also from an XGBoost-trained JSON). On the **same** data, N rounds
followed by M continued rounds is byte-identical to N+M uninterrupted rounds,
including `subsample` and `colsample_*`, because FerrisBoost keys its sampling on
the absolute round. On different data the old trees stay as they are and the new
trees learn from the new data's cuts, so there is no uninterrupted equivalent.
CPU and `gpu_math="exact"` GPU continuation are byte-identical; `gpu_math="fast"`
initializes the device predictions from the old model's margins and stays
deterministic.

A model trained with `lambdarank_unbiased=True` cannot be continued, as in XGBoost; train it again from the start.

## Feature sampling

Three nested parameters follow XGBoost's `ColumnSampler`: `colsample_bytree`
draws a feature set once per tree, `colsample_bylevel` draws a subset of it for
each tree level (shared by all nodes of the level), and `colsample_bynode`
draws a subset of that for every node. Each stage keeps
`max(1, floor(rate * previous_count))` features, and the rates multiply:

```python
model = fb.train(
    {"objective": "binary:logistic", "colsample_bytree": 0.8,
     "colsample_bylevel": 0.7, "colsample_bynode": 0.5, "seed": 3},
    X, label=y, num_boost_round=100,
)
```

The draws are deterministic and depend only on `seed`, the tree number, the
depth, and the node number: thread count, column blocking, and CPU versus GPU
(`gpu_math="exact"`) do not change the model. The random sequence is
FerrisBoost's own, so for the same `seed` the chosen features are **not**
the same as XGBoost's (XGBoost's CPU and GPU sequences already differ from each
other); the semantics, counts, and nesting are the same. The same is true of
`colsample_bytree`.

`colsample_bylevel` and `colsample_bynode` only narrow the split search, as in
XGBoost. Histograms are still built for every feature of the per-tree set, so
unlike `colsample_bytree` they do not reduce input reads.

### Feature weights

Available in 0.0.9 development builds. `feature_weights` biases feature
sampling at any stage whose `colsample_*` rate is below `1`:

```python
model = fb.train(
    {"objective": "binary:logistic", "colsample_bytree": 0.7,
     "feature_weights": {"important_feature": 4.0, "rare_feature": 0.2}},  # named-column input
    X, label=y, num_boost_round=100,
)
```

Pass either a nonempty positional vector with one finite, non-negative
`float32` weight per feature, or a dictionary from feature names to weights.
Named weights require a complete named schema; names omitted from the
dictionary default to `1`. Booleans (including NumPy booleans), nonnumeric
values, and all-zero weights are rejected. At least one weight must be
positive, even when every sampling rate is `1`.

At each stage whose rate is below `1`, zero-weight features are ineligible.
The stage selects without replacement
`min(max(1, floor(rate * incoming_count)), positive_count)` features. A rate
of `1` passes every incoming feature through, including zero-weight features,
so weights are not a global feature ban. The usual nesting and interaction
constraints still apply; their intersection can leave a leaf with no eligible
features.

Weights are resolved to feature order once and borrowed by the native nested
samplers.
Equal positive weights preserve the existing seeded uniform feature sequence
exactly; `None` keeps the existing default behavior. Sampling uses FerrisBoost's
own deterministic sequence, independent of XGBoost's nonuniform sampling
algorithm. CPU, exact GPU, and fast GPU use the same host sampler. Weighted
ordering uses host floating-point logarithms, so cross-architecture bitwise
agreement is not guaranteed.
Fast mode remains deterministic across repeated fast runs; this does not
promise CPU/GPU whole-model equality. Feature weights are training parameters
and are not stored in model metadata. For equivalent continued training, pass
the same weights and sampling parameters again.

### Gradient-based row sampling

Available in 0.0.9 development builds.

`sampling_method="gradient_based"` changes how `subsample` chooses training
rows. The default, `"uniform"`, preserves the existing uniform row sampler.
Gradient-based sampling scores each row from its current gradient and Hessian
as `sqrt(g² + 0.1 h²)`, after sample weights have been applied once. The
constant `0.1` does not depend on `reg_lambda`. Rows with larger scores are
more likely to be sampled. Inclusion probabilities are normalized toward an
expected sample count of `min(subsample * row_count, nonzero_score_count)`;
the actual rows are selected independently from a deterministic hash of the
seed, tree, and row. Sampling is therefore repeatable, but the realized row
count can vary.

Nonzero probabilities are represented on a 53-bit grid and rounded down, with
at least one probability-grid slot for each nonzero score. Sampled gradients
and Hessians are scaled by the inverse of each row's actual inclusion
probability. Rows with zero score remain zero. Every row still receives a
prediction from each resulting tree, whether or not it shaped that tree.
`subsample=1` is an exact no-op for this method as well.

The method is available on CPU and CUDA exact, and for supported native
objectives in CUDA fast mode. Exact mode uses a fixed 48 KiB histogram;
fast mode solves the inclusion probabilities on the device with bounded
shared memory. Neither path adds a row-sized sampling scratch buffer.
Fast mode is repeatable but need not match CPU sampling bit for bit.
`subsample=1` remains a no-op. This is a memory bound, not a speedup claim.
`fb.train`, sklearn estimators and file inputs forward the setting.
Continuation must use the same sampling parameters to reproduce an
uninterrupted run.

## Tree growth and feature interactions

Available in 0.0.9 development builds. `max_leaves` limits the total leaves in
each tree, including branches that stopped growing earlier. The default `0`
adds no leaf limit; `1` produces a root-only tree. Trees retain depthwise growth,
with stable node order when a level has more candidates than remaining leaf
slots. `max_depth` remains an independent limit; its value `0` still means a
root-only tree. Loss-guided growth is not implemented.

`interaction_constraints` restricts which features may appear together along
a root-to-leaf path:

```python
model = fb.train(
    {"objective": "binary:logistic", "max_leaves": 16,
     "interaction_constraints": [[0, 1], [2, 3]]},
    X_train, label=y_train, num_boost_round=100,
)
```

Use global feature indices, or names for inputs with named columns. Groups may
overlap: a path must remain within an original group containing all features
already used on that path. Features omitted from every group can only interact
with themselves. Reusing a feature is allowed. An empty outer list or `None`
disables constraints; empty inner groups and invalid feature references raise
an error. Duplicate entries are normalized in Rust.

Both controls apply to numerical and categorical splits and combine with
column sampling and monotone constraints. Constraints narrow split candidates;
they do not automatically eliminate histogram reads needed by other nodes.

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

## Missing values and the `missing` sentinel

`NaN` (and a Parquet / Arrow null, or an empty CSV field) is a missing value:
the row follows each split's learned default direction. If your data marks
missing values with a sentinel instead, declare it with `missing`, exactly like
XGBoost's `DMatrix(..., missing=...)` (default `NaN`):

```python
model = fb.train(
    {"objective": "reg:squarederror", "missing": -999.0},
    X, label=y,
    evals=[(X_valid, y_valid, "valid")],
)
pred = model.predict(X_test, missing=-999.0)     # pass the same value again
model.predict_to_parquet("test.parquet", "pred.parquet", missing=-999.0)
```

A feature value equal to `missing` is treated **exactly like `NaN`**, and `NaN`
stays missing as well. This holds in the training data and in every validation
set, and in prediction, for NumPy arrays, pandas DataFrames, PyArrow
tables and Parquet / CSV files (training converts any NumPy dtype to `float32`
first; prediction takes `float32` arrays). For files the Parquet and CSV readers apply the
sentinel while streaming, so nothing is materialized in Python. A NumPy array is
never modified: when some value equals the sentinel, one corrected float32 copy
is made inside the call.

What the sentinel does and does not touch:

- Sentinel values are skipped by the quantile sketch like `NaN`, so they never
  become a bin boundary or shift the cuts; the resulting model is
  bit-identical to training on the same data with the sentinel replaced by `NaN`.
- Only **feature** values are affected. Labels and the weight column are read as
  they are.
- Categorical features: as in XGBoost, `missing` is compared with the integer
  category codes too, so `missing=0` makes category code `0` a missing value.
  Choose a sentinel outside your code range (for example `-1`).
- The value is compared after conversion to `float32`, as in XGBoost.
- `missing` must be `NaN` or a finite `float32` value. `inf`, `-inf`,
  `numpy.finfo(numpy.float32).max` (which is reserved for split boundaries) and
  values that overflow `float32` raise `ValueError`. XGBoost accepts `inf` as a
  sentinel; FerrisBoost rejects it because infinite features are an error here.
  Features equal to `+/-inf` or `float32` max are still rejected when a
  sentinel is set.
- **The model does not store `missing`** (nor does XGBoost's JSON), so predict
  with the same `missing` you trained with; otherwise the sentinel is scored as
  an ordinary number. The scikit-learn estimators hold it as a constructor
  parameter and do this for you.
- A persistent Parquet `cache_path` is bound to the `missing` value it was
  built with; opening it with another value is an error rather than a silent
  reuse of bins built for a different sentinel.

## Categorical features

FerrisBoost splits categorical features the way XGBoost's `hist` method does.
A categorical feature holds **integer category codes**, one bin per code, and a
split sends a chosen set of categories to the right child and every other
category to the left.

```python
model = fb.train(
    {"objective": "binary:logistic", "max_cat_to_onehot": 4, "max_cat_threshold": 64},
    X,                                  # NumPy array of floats
    label=y,
    feature_types=["q", "c", "c", "q"],  # "c" = categorical, "q" = numeric
    num_boost_round=100,
)
```

How a feature is declared:

- **`feature_types`** (`fb.train` argument): one `"c"` or `"q"` per feature, as in
  XGBoost's `DMatrix`. It works for NumPy arrays, PyArrow tables, and Parquet
  or CSV files. For files the list is positional over the feature columns (the
  label and weight columns are not features).
- **pandas `category` columns and PyArrow dictionary columns** are detected
  automatically when `params["enable_categorical"]` is `True`; without it they
  are rejected so a categorical column is never trained as one by accident.
  A category becomes its integer code (its position in the category list), and
  the category list is **stored in the model** (see *Named categories* below);
  missing values stay missing. An explicit `feature_types` overrides detection.
  Categories must be strings or integers, as in XGBoost; other category types
  (floats, booleans, ...) raise `ValueError`.
- **Parquet dictionary columns** are categorical automatically, unless an
  explicit `feature_types` says otherwise. A dictionary of **strings** is a
  named categorical feature (below). A dictionary of **integers** keeps its
  values as the category codes, as before (unlike an in-memory Arrow table, see
  *Integer dictionaries: Arrow tables and Parquet files* below). Other
  dictionary value types raise `ValueError` when the feature is categorical.
- **CSV columns** are categorical only when `feature_types` declares them
  `"c"`. A declared column of integers holds category codes; a declared column
  of text is a named categorical feature (see *CSV text columns* below).

Category codes must be integers in `0..=65533`, with `NaN` / null for a missing
value; negative, fractional, or larger values raise `ValueError`. One feature
can have at most 65,534 categories, because the two largest bins are reserved:
one for missing values and one for categories seen only at validation or
prediction time, which always go left, as in prediction. At validation and prediction time a fractional value is read as its integer part (`1.5` is category `1`), as in XGBoost; training data must hold whole codes. A feature
whose largest code is at most 253 is stored with one byte per value; a larger
code stores that feature (only) with two bytes per value. CPU and GPU training
support the full range. XGBoost allows codes up to 2^24.
Codes need not be contiguous, but each code up to the feature's largest costs one
histogram bin, so prefer compact codes.

Validation sets use the training codes. For integer codes, a code that never
appeared in training is "not in any set" and goes to the left child, as it does
in XGBoost. Named validation data (pandas, PyArrow, Parquet string
dictionaries, CSV text columns) is first re-coded by value onto the training
categories, and a category the training data did not have raises `ValueError`
(see *Named categories*). The rules for every input type are in
[How categorical input is coded](#how-categorical-input-is-coded).

### Named categories

When the input names its categories (pandas `category` columns, PyArrow
dictionary columns, Parquet string dictionaries, CSV text columns declared
`"c"`), the model stores one category list per feature, in XGBoost 3.x's `cats`
layout, and every later read re-codes by **value**:

- **Files stay streaming.** For Parquet the Rust reader collects the
  dictionaries while it streams the data (nothing is materialized in Python):
  each feature's list holds the values in the order they first appear in the
  rows (files in order, then rows), i.e. `pandas.unique` of the column;
  dictionary order and unused dictionary entries play no part, so the list does
  not depend on row groups, batching or thread count. Every file is coded onto
  that one list, so files with differently ordered dictionaries train on
  consistent codes. A plain string column declared `"c"` in
  `feature_types` works the same way, and so do CSV text columns (below). At
  most 65,534 categories per feature; more than 254 store the feature with two
  bytes per value, on CPU and GPU.
- **Validation sets and prediction** (`predict`, `pred_contribs`, `pred_leaf`,
  `predict_to_parquet`, CPU or GPU) re-code DataFrames, Arrow tables, Parquet
  and CSV files with the stored lists, so their categories may be listed in any
  order or be a subset. String and integer categories do not mix: the category
  data type must match the training data. NumPy input is taken as integer
  codes. A feature with integer categories looks up plain integer columns by
  value, from files, DataFrames and Arrow tables alike. A feature without a
  stored list (trained on integer codes) takes integer values as its codes,
  including the values of a pandas integer category or an Arrow integer
  dictionary, and a pandas string category or Arrow string dictionary by its
  codes. The
  full rules per input type are in
  [How categorical input is coded](#how-categorical-input-is-coded).
- **Unseen categories raise `ValueError`**, as in XGBoost: a validation or
  prediction row holding a category the model does not know is an error, not a
  silent left turn. The message names the column and the value, for example
  `feature column "city" holds the category "Oslo", which is not in the
  model's categories for this feature` (files) or `predict data column 'city'
  holds the category 'Oslo', which is not in the training categories`
  (DataFrames and Arrow tables). Dictionary entries no row uses are ignored, and
  a missing value (null, `NaN`, an empty CSV field) is never unseen.
- **Model JSON**: XGBoost 3.4.1 loads a FerrisBoost model with categories and
  re-codes DataFrames the same way; FerrisBoost loads an XGBoost model trained
  on pandas categories with its lists. `Model.get_categories()` returns them.
- **Continuation** (`init_model`) keeps the model's lists, as XGBoost does:
  named training and validation data are coded with them, and a category
  outside them raises `ValueError`. (When the new frame lists its categories in
  another order, XGBoost 3.4.1 trains the new trees on the frame's own codes;
  FerrisBoost re-codes by value, so old and new trees agree.)
- **Quantized cache**: a Parquet cache records the category lists; reusing it
  with other lists (for example a different `init_model`) is rejected.

#### CSV text columns

CSV has no types, so FerrisBoost infers each column's type from the first
10,000 records of every file. A column declared `"c"` in `feature_types` is
decided from **every** record of every training file instead, during the first
pass that also learns the categories: it is read as **text** (category names)
when any non-empty field is not a number, and then in every file, so one feature
never mixes codes and names; a declared column whose fields are all numbers
keeps them as category codes, without names, as before. Text columns that are
not declared `"c"` raise `ValueError`.

```python
# city,x1,label
# Oslo,3,0.5
# Lima,,1.0
# ,7,0.2        <- empty field: missing
model = fb.train(params, "train.csv", label="label", feature_types=["c", "q"])
model.get_categories()  # [['Oslo', 'Lima', ...], None]
```

- The list holds the values in order of first appearance in the rows (files in
  order, then rows), the same rule as Parquet string columns, independent of
  batching and thread count. The same rows as CSV or as Parquet train the same
  model, on CPU and GPU (`gpu_math="exact"` is byte-identical to the CPU under
  every residency plan, and the GPU plan CSV makes after its first pass is the
  Parquet plan).
- Fields are taken verbatim (no trimming; numbers, `true` or dates in a text
  column are the strings as written). Only an empty field is missing: `NA` or
  `null` are categories (`pandas.read_csv` reads them as missing by default).
- Validation and prediction files read every feature with string categories in
  the model as text, whatever their own fields look like: a file holding only
  `1` and `2` of the categories `["1", "2", "x"]` is coded by value, not as the
  codes 1 and 2. A model with **integer** categories (trained on pandas integer
  categories) re-codes CSV integers, and plain Parquet integer columns, by value
  too.
- Validation and prediction files read the categorical features of a model
  without category names (trained on integer codes) as codes from every
  record: a field that is not a number raises `ValueError` naming the file, the
  column, the value and its row.

#### Integer dictionaries: Arrow tables and Parquet files

An integer dictionary column trains differently in memory and in a file, but
is scored the same way:

| Input | Category list stored in the model | Codes trained on | Read by a model without a list for the feature |
|---|---|---|---|
| PyArrow table / record batch, integer dictionary | the dictionary values in dictionary order (chunks are unified first: the first chunk's dictionary, then values new in later chunks), with the dictionary's integer type, like a pandas integer category | dictionary indices | the dictionary **values** |
| Parquet file, integer dictionary | none (no `cats`) | the dictionary **values**, which must be codes in `0..=65533` | the dictionary **values** |

- An Arrow table's integer dictionary is a named categorical feature: later
  input is re-coded by value, so `[40, 3, 17]` and `[3, 17, 40]` dictionaries
  score alike, and the model's `cats` hold `[40, 3, 17]` as XGBoost writes them
  for the same pandas column.
- A Parquet file's integer dictionary trains exactly like a plain integer
  column declared `"c"`: the value `40` is category code 40. To get named
  integer categories from a file, train from the table in memory
  (`pq.read_table(path)`), or from pandas.
- String dictionaries follow the same split in ordering only: an Arrow table's
  list is its (unified) dictionary order, a Parquet file's list is the order of
  first appearance in the rows. Either way the codes are consistent and later
  input is re-coded by value.
- Prediction and validation re-code integer dictionaries by value when the
  model has integer categories for the feature. Without a list (a model
  trained on integer codes, NumPy codes or a Parquet integer dictionary), an
  Arrow table's integer dictionary is read by its **values**, exactly as the
  same column in a Parquet file: `[30, 10, 20]` and `[10, 20, 30]`
  dictionaries, any chunking, the Parquet file and the plain codes all score
  alike. The values are taken as codes under the usual prediction rules (a
  null index or entry is missing; a negative value, or one the feature never
  had in training, goes left at a categorical split). A string dictionary has
  no values such a model could look up and is read by its indices.

Split search follows XGBoost:

- a feature with fewer than `max_cat_to_onehot` categories (default 4) is split
  one-versus-rest (a large `max_cat_to_onehot` over thousands of categories is
  slow);
- otherwise the categories are sorted by their leaf weight and prefixes of that
  order are scanned, at most `max_cat_threshold` (default 64) per direction;
- both missing-value directions are tried, and ties resolve to the earlier
  candidate, so training is deterministic and does not depend on thread count.

`lambda`, `alpha`, `max_delta_step`, `min_child_weight`, `gamma`, sample weights,
`monotone_constraints` on the numeric features, `subsample` and `colsample_bytree`
all apply. With lossless binning (numeric columns with fewer distinct values than
`max_bin`, categorical codes) the trees and predictions match XGBoost
`tree_method="hist"` exactly, for regression, binary, and multiclass objectives.

Prediction handles category sets on every path (arrays, pandas / PyArrow, files,
`predict_to_parquet`, multiclass). Named input is re-coded by value with the
stored categories (see *Named categories*); the scikit-learn estimators do the
same. With `use_gpu=True` categorical models are scored on the GPU as well,
with the same category decisions and bit-identical results (see
[Prediction](#prediction)).

Saved models use XGBoost's JSON layout (`split_type`, `categories`, the
`feature_types` list and, for named categories, `cats`) and load in XGBoost in
both directions. A loaded XGBoost
model may use category ids above 65533; prediction follows its sets, with
unknown or invalid values going left.

On GPU, categorical features work with `gpu_math="exact"` (byte-identical to
CPU) and `"fast"`. The resident GPU split cache enumerates categorical features
on the device (one-hot and sorted partition, with the same tie-breaking and
missing-value direction as the CPU), so histogram building, split search, row
partitioning, and gradients all stay on the device. Cache overflow and pure
streaming (`resident_blocks=0`) fall back to the exact host enumeration over
downloaded histograms.

## Learning to rank

`rank:ndcg`, `rank:pairwise` and `rank:map` train LambdaMART rankers with
XGBoost's semantics, parameter names and model JSON. The rows of one query
(search, user, session) form a **query group**; the model learns to order the
documents inside a group by relevance.

```python
import numpy as np
import ferrisboost as fb

# 3 queries with 4, 3 and 5 documents; graded relevance labels 0-4.
# The rows must be sorted by query.
qid = np.array([0] * 4 + [1] * 3 + [2] * 5)
model = fb.train(
    {"objective": "rank:ndcg", "eta": 0.1, "max_depth": 4,
     "eval_metric": "ndcg@5"},
    X, label=relevance, qid=qid, num_boost_round=200,
    evals=[(X_valid, relevance_valid, "valid")], eval_qid=[qid_valid],
    early_stopping_rounds=10,
)
scores = model.predict(X_valid)      # raw ranking scores, higher = more relevant
```

**Query groups.** Array inputs take `qid` (one query id per row, rows sorted by
it: every run of equal ids is one group) or `group` (the size of every group,
in row order); give one of them. Validation sets take `eval_qid` / `eval_group`,
lists parallel to `evals`. For Parquet and CSV files, `qid` is the **name of a
column** (the column index for a headerless CSV): it is excluded from the
features like `label`, and the validation files reuse it. The ids must be
non-decreasing along the rows (an unsorted `qid` is an error, not a silent
regrouping), and a group cannot be empty. A ranking objective without query
groups raises `ValueError` (XGBoost would treat the data as one group; here
the message asks for `qid=` / `group=`). Array inputs of other objectives
accept query groups too, as in XGBoost: they do not change training, but
`auc` and `aucpr` on data with groups are scored per group (see Metrics).

**Weights are per query group**, as in XGBoost: `weight` has one value per group
(not per row). For files, the weight column repeats the group's weight on every
row of the group (different values inside a group are an error). Group weights
also weight the quantile sketch of the bin boundaries, like row weights do.

**Objectives and parameters.** XGBoost's names and defaults:

| Parameter | Default | Meaning |
|---|---:|---|
| `lambdarank_pair_method` | `"topk"` | `"topk"`: every pair of a top-`k` document with a lower-ranked one, `k` = `lambdarank_num_pair_per_sample`. `"mean"`: sample `lambdarank_num_pair_per_sample` partners with a different label for every document |
| `lambdarank_num_pair_per_sample` | `32` (`topk`), `1` (`mean`) | Truncation level / number of sampled pairs; at least 1 |
| `lambdarank_normalization` | `True` | Normalize the gradient of each group (by `log2(1 + sum)` for `topk`, by the pair count for `mean`) |
| `lambdarank_score_normalization` | `True` | Divide the metric change by the score difference |
| `ndcg_exp_gain` | `True` | NDCG gain `2^label - 1` (labels must be integers in `[0, 31]`); `False` uses the label itself |
| `lambdarank_unbiased` | `False` | Unbiased LambdaMART: estimate the position bias of click data and divide it out of the pair gradients (see below) |
| `lambdarank_bias_norm` | `1.0` | Exponent control of the position-bias update: the ratios are raised to `1 / (1 + lambdarank_bias_norm)`; at least 0 |

`rank:map` needs binary labels (0 / 1). `rank:pairwise` optimizes the RankNet
pairwise loss without a metric weighting. Documents that share a label inside a
group form no pair; a group of one document or with equal labels contributes
nothing. `subsample` and `colsample_*` work as for every objective: rows are
sampled after the group gradients are computed.

With `lambdarank_pair_method="mean"` the sampled pairs come from XGBoost's
generator (one draw of the training `std::mt19937` per round, `seed` selects
the stream), so models match XGBoost's on losslessly binned data when
`subsample` and `colsample_*` are 1. The integer sampling adapter follows the
native XGBoost platform (libstdc++ on Linux, MSVC STL on Windows), so `mean`
training can produce different models across operating systems. Saved models
remain portable; CPU and GPU exact remain byte-identical on each platform.

**Unbiased learning to rank.** With `lambdarank_unbiased=True` the rows of a
query group are taken to be in the order the documents were **displayed**, and
the labels to be clicks. Every round FerrisBoost estimates how likely a click
at each display position is (`ti+` for relevant documents, `tj-` for the
others, following the Unbiased LambdaMART paper and XGBoost) and divides these
position biases out of the pair gradients. Positions beyond
`lambdarank_num_pair_per_sample` (`topk`; at most 32 for `mean`) are not
tracked. The estimates are stored with the objective in the model JSON as
`ti+` / `tj-` (32-bit arrays, as in XGBoost); they do not change predictions.
Models match XGBoost's bit for bit on losslessly binned data, on the CPU and
with `gpu_math="exact"` (byte-identical to the CPU). With `gpu_math="fast"` the
bias estimation runs on the device too: reproducible from run to run, close to
but not byte-identical to the CPU (the same caveat as fast ranking in general).
Every round it makes two passes over the pairs, the first finding the exact
maximum of what will be added so the fixed-point scale fits the bias-divided
gradients, and a gradient or statistic that is not finite is an error.

In fast unbiased ranking, a sigmoid that rounds to 1 uses a minimum
`1 - sigmoid` of `2^-24` when estimating position bias. This keeps that cost
finite; saturated scores can therefore differ from XGBoost and exact mode.

**Metrics.** `ndcg`, `map` and `pre`, with XGBoost's syntax and values: `ndcg@k`
and `map@k` truncate at `k`, a trailing `-` (`ndcg-`, `ndcg@5-`) scores a group
without relevant documents as 0 instead of 1. A metric without `@k` is
truncated at 32 (as in XGBoost, so `ndcg` is `ndcg@32`; write
`ndcg@1000000` for the whole list). `map` and `pre` need binary labels. The
default metric is `ndcg@32` for `rank:ndcg` and `rank:pairwise` and `map@32` for
`rank:map` (`k` follows `lambdarank_num_pair_per_sample`; with the `mean` method
`ndcg` covers the whole list). Early stopping maximizes them. With group weights
the group scores are averaged with those weights. Ranking objectives accept
these metrics plus `auc` and `aucpr`, and the other ranking metrics are
rejected for other objectives.

`auc` and `aucpr` on data with query groups are computed **per group** and
averaged, as in XGBoost. `auc` counts every pair of a group ordered by label (1
when the scores keep the label order, 0.5 for tied scores, 0 otherwise; groups
with fewer than 3 documents are skipped), so it is not the pooled ROC AUC;
`aucpr` is the binary PR-AUC of each group (labels in `[0, 1]`; groups without a
positive or without a negative document are skipped). The group weight
multiplies both sides of the ratio of `auc`, so it only matters as 0. The mean
is over the groups that can be scored, and the value is `NaN` when none can.
Both work for early stopping. XGBoost uses the same per-group path for
`binary:logistic` whenever the validation data has `qid` / `group`, and so does
FerrisBoost (arrays only; the weights of that set must then be per group, which
`binary:logistic` cannot use, so such a set takes no weights).

**Starting score.** Without `base_score`, the estimate is one Newton step on the
objective's gradients at score 0, as in XGBoost; it is close to 0.

**scikit-learn.** `FerrisBoostRanker` (`XGBRanker`) takes `qid` or `group` in
`fit`, `eval_qid` / `eval_group` for `eval_set`, and also accepts a pandas
`qid` column in `X`. `sample_weight` is per group. `predict` returns the scores
and `score(X, y)` (with a `qid` column in `X`) returns the ranking metric.

```python
from ferrisboost.sklearn import FerrisBoostRanker

ranker = FerrisBoostRanker(n_estimators=200, learning_rate=0.1, max_depth=4)
ranker.fit(X, relevance, qid=qid, eval_set=[(X_valid, relevance_valid)],
           eval_qid=[qid_valid])
ranker.predict(X_valid)
```

XGBoost `rank:*` models load with `Model.load_model` (and FerrisBoost ones in
XGBoost); the objective parameters are stored in the model JSON. On the GPU,
`gpu_math="exact"` computes the group gradients on the host and stays
byte-identical to the CPU; `gpu_math="fast"` computes them on the device (see
[Train on GPU](#train-on-gpu)).

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

- The lower-level `predict_array()` method rejects multiclass models; use
  `predict()`.

`predict_to_parquet` streams multiclass output like the single-output path: the
input is read once in bounded batches, each batch is scored for every class and
written before the next one is read, so memory does not grow with the number of
rows. Like the single-output path, it refuses to overwrite an existing file,
requires a `.parquet` or `.pq` extension, and removes a partial file if
prediction fails. The output file is the same whatever the thread counts or
batch sizes were.

File prediction reads the input once: each batch is scored for every class
before the next batch is read.

### Multiclass models and XGBoost

Multiclass models are saved with `num_class`, per-tree class ids, and a vector
`base_score`, and are exchanged with XGBoost in both directions: XGBoost loads
a FerrisBoost multiclass model, and FerrisBoost loads one trained by XGBoost.

## scikit-learn estimators

`FerrisBoostClassifier` and `FerrisBoostRegressor` follow the `XGBClassifier` /
`XGBRegressor` interface without taking the XGBoost names (so both libraries can
be imported together):

| XGBoost | FerrisBoost |
|---|---|
| `xgboost.XGBClassifier` | `ferrisboost.FerrisBoostClassifier` |
| `xgboost.XGBRegressor` | `ferrisboost.FerrisBoostRegressor` |

scikit-learn is optional: `import ferrisboost` never imports it, and the two
names are loaded on first access.

```python
from ferrisboost.sklearn import FerrisBoostClassifier

clf = FerrisBoostClassifier(
    n_estimators=500, learning_rate=0.1, max_depth=6,
    early_stopping_rounds=20, eval_metric="logloss",
)
clf.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
proba = clf.predict_proba(X_valid)        # (n, 2); (n, K) for multiclass
labels = clf.predict(X_valid)
print(clf.best_iteration, clf.evals_result()["validation_0"]["logloss"][-1])
clf.save_model("clf.json")                # XGBoost-compatible JSON
```

- `fit(X, y, sample_weight=, base_margin=, eval_set=, sample_weight_eval_set=, base_margin_eval_set=, init_model=, verbose=)`,
  `predict`, `predict_proba`, `apply`, `score`, `get_params` / `set_params`, `clone`,
  pickling, `GridSearchCV`, `cross_val_score`, and `Pipeline` work. Constructor
  parameters use XGBoost's names (`n_estimators`, `learning_rate`, `max_depth`,
  `subsample`, `colsample_by*`, `reg_alpha`, `reg_lambda`, `n_jobs`,
  `random_state`, `eval_metric`, `early_stopping_rounds`, ...) plus FerrisBoost's
  `device`, `gpu_math`, `cols_per_block`, `quantile_alpha`,
  `tweedie_variance_power`, `huber_slope`, and `importance_type`. `None` means "use FerrisBoost's default".
- `missing` (default `NaN`) is a constructor parameter as in XGBoost. It is
  applied to the training data, every `eval_set`, and to `predict`,
  `predict_proba`, and `apply`, so a fitted estimator always uses the value it
  was trained with. After `load_model`, give the same `missing` to the
  constructor (the model file does not store it). See
  [Missing values](#missing-values-and-the-missing-sentinel) for the rules.
- Categorical features: `enable_categorical`, `feature_types`,
  `max_cat_to_onehot`, and `max_cat_threshold` are constructor parameters that
  go to `ferrisboost.train`. pandas `category` columns and PyArrow dictionary
  columns work in `fit`, `eval_set`, `predict`, `predict_proba`, and `apply`,
  with `feature_names_in_` set as for any frame, and are coded as
  `ferrisboost.train` and `Model.predict` code them (see
  [How categorical input is coded](#how-categorical-input-is-coded); a model
  fitted on plain arrays reads a pandas integer category or an Arrow integer
  dictionary by its values);
  with `enable_categorical=False` such a column in `fit` raises
  `ValueError`. The training categories are stored in the booster and later
  frames are re-coded by value, so a different category order is harmless; a
  category unseen in training raises `ValueError`, as in XGBoost. A model
  restored with `load_model` keeps its categories. Each fresh `fit` learns a
  new category state and does not reuse lists from a previous fit; use
  `init_model` to continue from that model and its categories.
  For plain arrays, mark categorical columns with `feature_types=["c", "q", ...]`.
  `GridSearchCV`, `cross_val_score`, and `clone` work with categorical frames.
- `apply(X, iteration_range=None)` returns the leaf node reached in every tree,
  like `XGBModel.apply`: a `float32` array of shape `(n_samples, n_trees)` (a
  multiclass model has `num_class` trees per boosting round). Without
  `iteration_range`, a model trained with early stopping uses its
  `best_iteration`. It is `Model.predict(..., pred_leaf=True)` underneath.
- Binary classification uses `binary:logistic`; more than two classes switch to
  `multi:softprob` (or `multi:softmax` if you ask for it) and set `num_class`.
  Labels can be any sortable class values; they are encoded internally and
  `classes_` keeps the originals (XGBoost requires `0..K-1`; for such labels
  the behavior is the same).
- After early stopping, `predict` / `predict_proba` use trees up to
  `best_iteration` automatically, as XGBoost's sklearn layer does (the raw
  `Model.predict` still uses all trees). `early_stopping_rounds` requires an
  `eval_set`.
- `evals_result()` is keyed `validation_0`, `validation_1`, ...
  `feature_importances_` is `Model.get_score(importance_type)` normalized to sum
  to 1 over all input features (`importance_type` defaults to `gain`, as in
  XGBoost). `verbose` defaults to `False` (set it to see the validation metrics
  per round).
- `load_model(path)` / `from_file(path)` read XGBoost-compatible JSON.
- `fit(..., base_margin=, base_margin_eval_set=, init_model=)` and
  `predict` / `predict_proba(..., base_margin=)` follow XGBoost; see
  [Starting margins](#starting-margins-base_margin) and
  [Continue training](#continue-training-init_model).
- Not supported: `booster=` and `tree_method=`.
  `sample_weight` follows FerrisBoost's weighted-sketch semantics, so a weight of
  `k` is not the same as repeating a row `k` times.

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
memory (one byte per value, two for features with more than 255 bins or
categorical codes above 253). Data beyond the budget spills to local files:

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
incompatible settings such as a different `max_bin`, `missing`, or set of
categorical features (from `feature_types` or dictionary columns) are rejected
rather than silently reused.

`cache_path` applies only to the training set. Validation sets reuse the
training cuts but do not share the training cache directory.

Which inputs use the cache:

| Training input | `cache_path` / `cache_budget_bytes` |
|---|---|
| One Parquet file (also a directory or glob that matches exactly one file), CPU or GPU, with or without validation sets or `init_model` | Supported |
| Several Parquet files (a list, directory, or glob) | `ValueError`: merge them into one file or drop the option |
| CSV (one or several files, with or without a header) | `ValueError`: convert the data to one Parquet file or drop the option |
| numpy, pandas, PyArrow `Table` / `RecordBatch` / `RecordBatchReader` | `ValueError`: write the data to one Parquet file or drop the option |

The scikit-learn estimators train on in-memory data and do not take these
options. Several Parquet files and CSV are quantized once and held in memory.

## Train from PyArrow

PyArrow is optional, and Arrow table training, evaluation, and prediction work
without pandas installed; the Arrow input path does not import pandas.

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

## Train from pandas

A pandas `DataFrame` is accepted as `data`, with the labels supplied
separately. Columns are converted to `float32` (nullable values become `NaN`),
and `category` columns are trained as categorical features when
`params["enable_categorical"]` is `True` (see
[Categorical features](#categorical-features)). Column names align validation
frames and resolve a `monotone_constraints` dict, but, as with NumPy and PyArrow
input, the trained model records no feature names. pandas is an optional
dependency that FerrisBoost only imports when you pass a frame. A column that
is neither numeric nor `category` raises `ValueError`.

```python
model = fb.train(params, df[feature_columns], label=df["target"].to_numpy(),
                 num_boost_round=100)
```

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
  or by other processes. When you set it together with an explicit
  `cols_per_block`, `hist_streams` or `resident_blocks`, the planned training
  memory must fit: an override that needs more raises `ValueError` before
  training starts, naming the override and the memory it needs. Without a budget,
  explicit overrides are used as given.
- **`gpu_math`**: `"exact"` (default) produces a byte-by-byte exact model
  matching CPU training. `"fast"` is deterministic, but CPU and GPU models are
  not guaranteed to be bit-identical; FerrisBoost does not promise universal
  parity with XGBoost GPU training.
- **Objectives and parameters on GPU**: Built-in objectives, growth constraints,
  sampling controls and sample weights work with `device="cuda"` and
  `gpu_math="exact"`. Exact mode computes objective derivatives on the host
  and preserves CPU/GPU model parity.
- **`gpu_math="fast"`**: Native scalar, multiclass, quantile and ranking
  objectives compute derivatives, quantization scales and prediction updates
  on the device. Runs are deterministic but need not match CPU bit for bit.
  `reg:absoluteerror` and `reg:quantileerror` reduce their residual scale on
  the device. Fast mode requires `device_quantize=True`; custom objectives
  and survival objectives currently require CPU or CUDA exact.
- **Ranking on the GPU**: with `gpu_math="exact"` the group gradients are
  computed on the host and the model is byte-identical to the CPU model. With
  `gpu_math="fast"` the whole LambdaMART step runs on the device: a stable
  segmented sort of every group's scores, pair generation (`topk` or `mean`),
  the NDCG / MAP metric change and the per-row accumulation, which uses
  fixed-point integer additions only, so a run is reproducible but not
  byte-identical to the CPU (the sampled pairs of `mean` also differ). It needs
  about 28 bytes per row plus 16 for `rank:map`, 12 for the `mean` method and
  28 per query group; the planner counts them for the real query groups (file
  inputs read the query-id column before planning). Very large groups can overflow
  the fixed-point accumulators and are rejected with an error.
  `lambdarank_unbiased=True` also runs in fast mode: the position biases live on
  the device (a few KB per tracked position), each round makes a second pass over
  the pairs, and the estimate is saved with the model at the end.
- **Extra GPU memory in fast mode**: `gpu_math="fast"` keeps the running
  prediction and the labels on the device (4 bytes per row each) and, with
  sample weights, the weights (4 more bytes per row). Multiclass keeps one
  prediction column per class plus two per-row softmax statistics; several
  `quantile_alpha` values keep one prediction column per quantile.
  `reg:quantileerror` adds a small fixed scratch buffer that does not depend
  on the row count (it is reused by every quantile in turn). The memory planner accounts for all of these.
- **`colsample_bylevel` / `colsample_bynode`**: supported with `device="cuda"` and byte-identical to the CPU under `gpu_math="exact"`. The per-level and per-node feature sets are drawn on the host exactly as on the CPU and applied by the device split cache.
- **Categorical features**: supported with both `gpu_math` modes. Split search
  for them runs in the device split cache (see
  [Categorical features](#categorical-features)).
- **`device_quantize`**: `True` (default) performs fixed-point gradient and
  Hessian quantization on the GPU. Feature binning and feature bins remain on
  the host/source side.
- **Bounded split cache**: A bounded resident histogram/split cache is enabled
  by default and included in the GPU memory plan. Cache overflow and pure
  streaming use the exact host fallback. This is managed internally and is not
  a public Python parameter.
- **`hist_streams`**: Defaults to automatic (`None`); the GPU planner resolves the stream count from the execution path, block width, and VRAM budget. The current column-major path uses one stream by default; set `1` or `2` to override it.
- **`device` selection**: Training currently targets one selected GPU (`device="cuda"` or `device="cuda:N"`). Multi-GPU support is future 1.0 work and is not implemented.
- **`max_depth`**: Depth-wise tree growth on GPU is constrained to `0..=14` by partition buffer limits.

### What runs where

| Step | Where it runs |
|---|---|
| Histograms and row partition | GPU for both math modes; objective gradients run on the host in exact mode and on the device in fast mode |
| Split search | GPU (resident cache) by default, including categorical features and `colsample_bylevel` / `colsample_bynode`; on the host from downloaded histograms for cache overflow and pure streaming |
| Feature binning and file ingest | Host |
| Prediction (`use_gpu=True`) | GPU 0 when worthwhile, including categorical models and `base_margin`; CPU for small batches and `predict_to_parquet` |
| Explanations (`use_gpu=True`) | `pred_contribs` (exact and `approx_contribs`), `pred_interactions` and `pred_leaf` run on GPU 0 for every call, categorical splits and `base_margin` included, with values bit-identical to the CPU path. With `use_gpu=False` they run on the CPU |

What raises an error instead of falling back: `device="cuda"` on a build without
CUDA, `max_depth` above 14 (on CPU too), `gpu_math="fast"` together with
`device_quantize=False`, custom objectives or survival objectives.

## Prediction

`Model.predict()` supports both file paths (the recommended path for large test sets) and in-memory NumPy arrays. The return value is a `float32` `numpy.ndarray`. It is one-dimensional, with shape `(n,)`, except for multiclass models, which return two-dimensional `(n, num_class)` probabilities or margins, and models with several `quantile_alpha` values, which return `(n, n_quantiles)` (each row sorted ascending); `multi:softmax` returns a one-dimensional array of class indices unless `output_margin=True` (see [Multiclass prediction](#multiclass-prediction)). Predictions are in the objective's output space: probabilities for `binary:logistic` and `reg:logistic`, class decisions for `binary:hinge`, raw margins for `binary:logitraw`, `exp(margin)` for `count:poisson`, `reg:gamma`, `reg:tweedie`, `survival:cox` and `survival:aft`, and the margin itself for other regression objectives; `output_margin=True` returns the raw margin. Prediction has its own CPU thread control and does not inherit training `nthread`:

```python
prediction = model.predict(data, predict_threads=0)  # process-aware auto
reference = model.predict(data, predict_threads=1)   # explicit serial path
gpu_allowed = model.predict(data, use_gpu=True)      # CUDA build; GPU 0 when worthwhile
```

`predict_threads=0` uses available physical CPU parallelism and keeps small workloads serial; a positive value is an explicit override. Workers own disjoint row slices and each row traverses trees in model order, so output is bit-identical across serial, explicit multi-thread, and auto CPU execution. NumPy scoring releases the Python GIL.

`use_gpu=False` is the default and never initializes CUDA. With `use_gpu=True`, a CUDA-enabled build lazily creates a compact prediction-only SoA model on GPU 0 and reuses it on later calls. It transfers only tree-used features, while the selector also accounts for host projection and launch costs and can keep a batch on CPU. The inference backend is independent of whether the model was trained by FerrisBoost CPU, FerrisBoost GPU, or XGBoost, and independent of training `gpu_math`.

GPU prediction covers numeric and categorical splits and an optional per-row `base_margin`. Each row is scored by one thread that starts from the intercept (or the row's `base_margin`) and adds leaf values in tree order, with the same decisions as the CPU: missing values follow the node's default direction, numeric splits send `value < threshold` left, and a categorical split sends a category in the node's set right (negative values, values at or above 2^24, and categories beyond the stored set go left; fractional values are truncated toward zero). Category sets may be any width, including the wide sets of XGBoost-trained models. The result equals CPU prediction bit for bit. A `base_margin` adds one `float32` per row to the staging buffers, which stay bounded to available VRAM.

### File prediction

Passing file paths streams records directly through Rust readers without materializing the full dataset in Python host memory:

```python
# Single file (.parquet, .pq, .csv, or .csv.gz):
prediction = model.predict("test.parquet", predict_threads=0)

# Multi-file or partitioned dataset (list of files, directory path, or glob):
prediction = model.predict(["test_part_0.parquet", "test_part_1.parquet"])
prediction = model.predict("data/test_partitions/")
prediction = model.predict("data/test_*.parquet")

# Headerless CSV (positional mode; for a model trained without feature names):
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

The projection applies to every file prediction: single-output models, multiclass and multi-quantile models (the union of the features the outputs use is read once), `predict_to_parquet`, and `pred_contribs` / `pred_interactions` / `pred_leaf` (a feature no tree splits on contributes exactly zero, so its column is not read and its output stays zero). A wide file therefore costs I/O and decoding for the used columns only, however many columns it holds. The schema check still covers all of the model's features: **every feature name of the model must be present in the file, including features no tree uses** (XGBoost validates feature names the same way). Because unused columns are never decoded, an unused column holding an unsupported type or a reserved value is not an error.

A model is bound to file columns by name when it has feature names, including an XGBoost model trained with `feature_names` (its JSON records no FerrisBoost schema mode; the names decide). A model without feature names that was not trained by this version of FerrisBoost (for example an XGBoost model trained on an array) is refused for file input with `ValueError`: see [An XGBoost model has no feature names](#an-xgboost-model-has-no-feature-names).

Readers keep deterministic file/row order and bounded batches. In auto CPU mode, ingest and scoring share one process-visible CPU budget so `ingest_threads` and `predict_threads` do not each consume the full machine. If both are explicitly set above that budget, FerrisBoost preserves the overrides and emits an oversubscription warning. GPU file prediction keeps input in the Rust batch path and does not first materialize a full NumPy matrix.

`predict_to_parquet()` accepts the same file, file-list, directory, and glob inputs as file-based `predict()`; it does not accept a NumPy input. This first implementation uses CPU scoring and writes one non-null `float32` column to a single `.parquet` or `.pq` file with Snappy compression. It returns the number of rows written. For multiclass models and for models with several `quantile_alpha` values it writes one column per class or quantile instead (see [Multiclass prediction](#multiclass-prediction)), streamed in the same bounded batches. The target must not already exist, and an incomplete output created by the call is removed if schema validation, reading, scoring, encoding, or finalization fails.

The direct-output path keeps a bounded two-batch channel between scoring and one ordered Parquet writer thread. The writer can overlap encoding/output with the next input batch while preserving file order and row order. Auto mode shares one process-visible CPU budget across ingest, scoring, and the writer; explicit ingest or prediction thread counts are preserved and produce an oversubscription warning when their total plus the writer exceeds that budget.

### The `missing` sentinel in prediction

`predict(data, missing=...)` and `predict_to_parquet(..., missing=...)` take the
same `missing` as training (default `NaN`) and apply it to arrays, DataFrames,
tables and files, including `pred_contribs`, `pred_interactions`, `pred_leaf`,
`iteration_range`, and multiclass models. The model file does not record it, so
pass the value you trained with. See
[Missing values](#missing-values-and-the-missing-sentinel).

### In-memory array prediction

Prediction also accepts a two-dimensional NumPy `float32` array with the same number and order of features used for training:

```python
X_test = np.asarray(X_test, dtype=np.float32)

prediction = model.predict(X_test)
raw_margin = model.predict(X_test, output_margin=True)
first_20_trees = model.predict(X_test, n_trees=20)
rounds_10_to_19 = model.predict(X_test, iteration_range=(10, 20))
```

`iteration_range=(begin, end)` follows XGBoost: it uses the half-open range of
boosting **rounds** `[begin, end)`, so a multiclass model uses `num_class`
trees per round, whereas `n_trees` counts trees. `end=0` means "to the last
round"; a range beyond the model or with `begin > end` raises `ValueError`, and
`begin == end` is an empty range (only `base_score`). The base prediction is
always included. It works for arrays, files, and `predict_to_parquet`, and
cannot be combined with `n_trees`. With `begin > 0` the call scores a copy of
the selected trees, so repeated calls with a nonzero `begin` are best batched.

C-contiguous input uses the direct row-major fast path. Fortran-contiguous and strided `float32` arrays are also interpreted correctly. Non-`float32` arrays must be converted explicitly with `np.asarray(..., dtype=np.float32)`. pandas `DataFrame`s and PyArrow `Table`s are accepted directly and converted to `float32` (categorical columns become the model's category codes; see [How categorical input is coded](#how-categorical-input-is-coded)). For a model trained on named columns (a DataFrame, a PyArrow table, Parquet, or a CSV with a header), their columns are matched to the model's features by name: order does not matter, extra columns are ignored, and a missing one is an error. This also applies to the explanation options (`pred_contribs`, `pred_interactions`, `pred_leaf`).

For GPU prediction, the first eligible call includes CUDA context initialization and one model upload; subsequent calls reuse the cached model. Staging memory remains per-call and bounded to available VRAM. GPU 0 is the only inference target in this version; multi-GPU prediction is intentionally not enabled.

The model exposes these properties:

```python
print(model.num_trees)
print(model.n_features)
print(model.best_iteration)
print(model.best_score)
print(model.evals_result())
```

### How categorical input is coded

Prediction (`predict` with every option, `predict_to_parquet`, CPU or GPU),
validation sets (`evals`, `eval_set`) and the scikit-learn estimators'
`predict`, `predict_proba` and `apply` turn a categorical feature's input into
the model's category codes by the same rules. What applies depends on whether
the model stores a category list for the feature (`Model.get_categories()`):

| Input for a categorical feature | Feature with string categories | Feature with integer categories | Feature without a list |
|---|---|---|---|
| NumPy array | codes as given | codes as given | codes as given |
| pandas string `category` column, PyArrow string dictionary | re-coded by value | `ValueError` (type mismatch) | the column's codes (positions in its categories, dictionary indices) |
| Parquet string dictionary or string column | re-coded by value | `ValueError` (type mismatch) | `ValueError` |
| pandas integer `category` column, PyArrow or Parquet integer dictionary | `ValueError` (type mismatch) | re-coded by value | the **values** as codes |
| pandas / PyArrow / Parquet plain integer column | codes as given | looked up **by value** | codes as given |
| pandas / PyArrow / Parquet float column | codes as given | codes as given | codes as given |
| CSV column | read as text and re-coded by value (`1` is the category `"1"`) | integers looked up **by value**; a decimal column is codes as given | integers as codes; text raises `ValueError` |

- **Re-coded / looked up by value**: the value's position in the model's list
  is the code, so input may list its categories in any order or a subset. A
  category not in the list raises `ValueError`, as in XGBoost; a missing value
  (null, `NaN`, `pd.NA`, an empty CSV field) stays missing.
- **Codes**: integers `0..` taken as the category codes the model was trained
  on. A null or `NaN` is missing; a negative value, a value the feature never
  had in training, or a value at or above 2^24 goes left at a categorical
  split, and a fractional value is truncated toward zero. An integer
  `category` column or dictionary on a feature without a list is read by its
  values exactly like the same column in a Parquet file, so category order
  and chunking do not matter.
- A column means the same whether it comes from a file, a pandas DataFrame or
  a PyArrow table: the column's type decides, not its source. Only NumPy
  arrays, which carry no column types, are always codes (as in XGBoost).
- A float column is codes as given even when its values are whole numbers
  (`10.0`), in files and in memory. A pandas integer column with a `NaN` is
  `float64`; use the nullable `Int64` dtype to have it looked up by value.
- The `missing` sentinel applies to the codes (after decoding a dictionary or
  looking up a value).

## Feature importance, SHAP values, and leaf indices

These follow XGBoost 3.4.1 (`Booster.get_score` and the `pred_contribs`, `approx_contribs`, `pred_interactions`, and `pred_leaf` options of `Booster.predict`) and work on models trained by FerrisBoost or loaded from XGBoost JSON.

```python
model.get_score()                           # {"f0": 12.0, "f3": 7.0, ...}
model.get_score(importance_type="gain")

contribs = model.predict(X_test, pred_contribs=True)          # (n, n_features + 1)
saabas = model.predict(X_test, pred_contribs=True, approx_contribs=True)
inter = model.predict(X_test, pred_interactions=True)         # (n, n_features + 1, n_features + 1)
leaves = model.predict(X_test, pred_leaf=True)                # (n, n_trees)
```

`get_score(importance_type=...)` returns `{feature_name: score}` for every feature used in at least one split, in feature order; features that are never used are omitted. `importance_type` is one of:

- `"weight"` (default): the number of splits that use the feature across all trees.
- `"gain"` / `"total_gain"`: the average / summed split gain of those splits.
- `"cover"` / `"total_cover"`: the average / summed Hessian sum (`sum_hessian`) of the split nodes.

Names are the model's feature names (the column names of a pandas DataFrame, PyArrow table, Parquet file, or headered CSV), or `f0`, `f1`, ... when the model was trained from a NumPy array. The scores come from the gain and Hessian sum that every tree node stores, so they also work on models loaded from XGBoost JSON.

`predict(..., pred_contribs=True)` returns exact TreeSHAP values in margin space. Every row sums to the raw margin (`output_margin=True`), and the last column is the bias term: the model's `base_score` (as a margin) plus the cover-weighted mean of the trees. With `approx_contribs=True` it returns the faster Saabas attribution instead. Multiclass models return `(n, num_class, n_features + 1)`, one attribution matrix per class; multi-quantile models return `(n, n_quantiles, n_features + 1)`.

`pred_interactions=True` returns SHAP interaction values `(n, n_features + 1, n_features + 1)` (multiclass: `(n, num_class, n_features + 1, n_features + 1)`). The matrices are symmetric, each row sums to the matching `pred_contribs` row, and the whole matrix sums to the margin. Memory grows with `n * (n_features + 1)^2`, so score wide data in batches. Combining `pred_interactions=True` with `approx_contribs=True` raises an error: XGBoost 3.4.1 returns a diagonal-only matrix there.

`pred_leaf=True` returns, for every row, the node index of the leaf it reaches in each tree, as `float32` like XGBoost; indices are node ids within the tree. The result has shape `(n, n_trees)` (a multiclass model has `num_class` trees per round).

These options accept a two-dimensional `float32` array, a pandas DataFrame, a PyArrow table, or file paths (bound by column name, streamed in batches). `n_trees` limits the trees used, with the same meaning as in normal prediction, and `predict_threads` controls CPU workers; results are identical for every thread count. Only one of `pred_contribs`, `pred_interactions`, and `pred_leaf` can be set per call, and none combines with `output_margin`. Exact SHAP uses the same fixed 8-point quadrature TreeSHAP as XGBoost 3.4.1 on CPU. A per-row `base_margin` replaces the intercept in the bias column, as in XGBoost (see [Starting margins](#starting-margins-base_margin)); `pred_leaf` ignores it.

#### `strict_shape`

`predict(..., strict_shape=True)` gives the output a shape that does not depend on the model type, exactly as XGBoost's `strict_shape`. The default shapes drop the output-group axis when there is a single group; with `strict_shape=True` it is always present:

| Output | Default | `strict_shape=True` |
|---|---|---|
| Prediction / `output_margin` | `(n,)`, or `(n, groups)` for multiclass probabilities, multiclass margins and multi-quantile models | `(n, groups)`; `groups` is 1 for regression, binary, ranking, single-quantile and `multi:softmax` class indices |
| `pred_contribs` | `(n, F + 1)`, or `(n, groups, F + 1)` | `(n, groups, F + 1)` |
| `pred_interactions` | `(n, F + 1, F + 1)`, or `(n, groups, F + 1, F + 1)` | `(n, groups, F + 1, F + 1)` |
| `pred_leaf` | `(n, n_trees)` | `(n, rounds, groups, 1)` (the last axis is the trees per round and group, always 1 here); `n_trees` must be a whole number of rounds |

It only reshapes the finished array (a view of the same values) and applies to every input type and to `iteration_range`; `rounds` counts the rounds actually used. `predict_to_parquet` has its own column layout and no such option.

#### SHAP on the GPU

`predict(..., use_gpu=True, pred_contribs=True)` (also `approx_contribs=True`, `pred_interactions=True` and `pred_leaf=True`) computes the explanations on GPU 0 in CUDA builds, for every call with rows (there is no small-batch fallback, unlike plain scoring), and for every kind of model, including models with categorical splits and models loaded from XGBoost. The result is **bit-identical** to the CPU path (and so to XGBoost 3.4.1 on the CPU in the same cases): the device runs the same fixed 8-point quadrature TreeSHAP with the same `float32` operation order, one thread per row and output group walking the trees in model order, with no atomics and no cross-thread reductions, so results are also identical from run to run. A per-row `base_margin` works on the device too.

Device memory is bounded: the ensemble is uploaded once per model, and rows are processed in batches sized from the free device memory (at most about 512 MiB per batch, including the per-thread traversal scratch, which grows with the deepest tree used). A model so wide that even one row's interaction matrix does not fit raises an error; use the CPU for it. Prediction scoring (`use_gpu=True` without an explanation option) is unchanged.

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
| `objective` | `binary:logistic` | 0.0.9 development: `reg:logistic`, `binary:logitraw`, `binary:hinge`, `reg:squaredlogerror`, `reg:absoluteerror`, `reg:expectileerror`, `survival:cox`, `survival:aft`; also `binary:logistic`, `reg:squarederror`, `count:poisson`, `reg:gamma`, `reg:tweedie`, `reg:pseudohubererror`, `reg:quantileerror`, `multi:softprob`, `multi:softmax`, `rank:ndcg`, `rank:pairwise`, `rank:map` |
| `lambdarank_pair_method`, `lambdarank_num_pair_per_sample`, `lambdarank_normalization`, `lambdarank_score_normalization`, `ndcg_exp_gain` | `"topk"`, unset, `True`, `True`, `True` | Ranking objectives only; see [Learning to rank](#learning-to-rank) |
| `lambdarank_unbiased`, `lambdarank_bias_norm` | `False`, `1.0` | Unbiased LambdaMART with position-bias estimation; see [Learning to rank](#learning-to-rank) |
| `expectile_alpha` | required | Required scalar for `reg:expectileerror`, in `[0, 1]`; arrays are unsupported |
| `quantile_alpha` | `None` | Required for `reg:quantileerror`: a quantile in `[0, 1]`, or an ascending list of up to 64 quantiles for one multi-output model (one tree per quantile per round; `predict` returns `(n, n_quantiles)`). Unsorted or out-of-range values raise `ValueError` |
| `num_class` | `None` | Number of classes, at least 2. Required for `multi:softprob` and `multi:softmax`, ignored otherwise. Labels are integers in `[0, num_class)` |
| `tweedie_variance_power` | `1.5` | Variance power of `reg:tweedie`, in `[1, 2)` |
| `huber_slope` | `1.0` | Slope of `reg:pseudohubererror` and its `mphe` metric; must be non-zero |
| `aft_loss_distribution` | `"normal"` | AFT noise distribution: `"normal"`, `"logistic"`, or `"extreme"` |
| `aft_loss_distribution_scale` | `1.0` | Positive finite AFT distribution scale |
| `custom_objective` | `None` | 0.0.9 development: an `fb.CustomObjective` descriptor; requires `objective="reg:squarederror"`. See [Custom objectives](#custom-objectives) |
| `max_depth` | `6` | Depth-wise tree growth (`0..=14`) |
| `max_bin` | `255` | Must be between 1 and 65535. Features with more than 255 bins use 2 bytes per value instead of 1 (CPU and GPU) |
| `eta` | `0.3` | Alias: `learning_rate` |
| `lambda` | `1.0` | Alias: `reg_lambda` |
| `gamma` | `0.0` | Minimum split loss |
| `min_child_weight` | `1.0` | Minimum child Hessian |
| `alpha` | `0.0` | L1 regularization on leaf weights; alias: `reg_alpha` |
| `max_delta_step` | `0.0` | Maximum absolute leaf weight before `eta` (`0` = no limit); can help logistic regression with very unbalanced classes. For `count:poisson` an unset value becomes `0.7`, for the tree and the objective alike |
| `scale_pos_weight` | `1.0` | Multiplies the gradient weight of rows with label `1`; metrics and binning are not affected. Used by `binary:logistic`, `reg:logistic`, `binary:logitraw`, `reg:squaredlogerror`, `reg:squarederror`, and `reg:gamma`, ignored by the other objectives. When it is not `1` and `base_score` is unset, the default `base_score` comes from one Newton step, as in XGBoost |
| `subsample` | `1.0` | Deterministic per-tree row sampling in `(0, 1]`; rows that were not sampled do not shape the tree, but still receive its leaf values |
| `sampling_method` | `"uniform"` | `"uniform"` or `"gradient_based"`; the latter uses current gradient/Hessian scores for row sampling; see [Gradient-based row sampling](#gradient-based-row-sampling) |
| `colsample_bytree` | `1.0` | Deterministic per-tree feature sampling in `(0, 1]`; column-major GPU histograms use compact selected-feature output |
| `colsample_bylevel` | `1.0` | Per-level feature sampling in `(0, 1]` from the per-tree set; see [Feature sampling](#feature-sampling) |
| `colsample_bynode` | `1.0` | Per-node feature sampling in `(0, 1]` from the per-level set; see [Feature sampling](#feature-sampling) |
| `feature_weights` | `None` | 0.0.9 development: nonnegative feature sampling weights as a positional vector or named dictionary; see [Feature weights](#feature-weights) |
| `cols_per_block` | automatic | Positive integer; normally leave unset. On GPU the memory planner chooses it from the row count; for CSV / CSV.gz it uses the exact count of the first reading pass, so CSV and the equivalent Parquet get the same plan |
| `nthread` | `0` | Zero selects physical cores; alias: `n_jobs` |
| `device` | `"cpu"` | Compute device: `"cpu"`, `"cuda"`, or `"cuda:N"` |
| `gpu_memory_budget` | automatic | Positive integer planner budget for GPU training memory; auto derives from free VRAM with safety headroom. Explicit `cols_per_block` / `hist_streams` / `resident_blocks` that exceed a set budget raise `ValueError` |
| `gpu_math` | `"exact"` | `"exact"` (byte-by-byte identical to CPU) or deterministic `"fast"` (CPU/GPU models need not be bit-identical); see [Train on GPU](#train-on-gpu) for objective support |
| `device_quantize` | `True` | Fixed-point gradient/Hessian quantization on GPU; feature binning remains on host/source. `gpu_math="fast"` requires it |
| `hist_streams` | automatic | Planned by GPU planner based on execution path, block width, and VRAM; override with `1` or `2` |
| `resident_blocks` | automatic | Number of physical column blocks kept resident on the GPU; `0` means pure streaming. A larger value uses more VRAM and reduces repeated uploads; normally leave unset (the planner chooses and clamps it to the block count). Must fit `gpu_memory_budget` when one is set |
| `hist_nodes_per_batch` | automatic | GPU only: number of nodes merged into one histogram launch (`1..=32`, `1` disables merging; the GPU default is 16) |
| `ingest_threads` | `0` (auto) | Process-visible CPUs, independent work, and memory bound file ingest; independent of `nthread` |
| `model_io_threads` | `0` (auto) | Work/memory-bound ordered tree conversion and model-file I/O; JSON parse/dump remains serial |
| `max_leaves` | `0` | 0.0.9 development: total leaf cap per depthwise tree; zero adds no cap |
| `interaction_constraints` | `None` | 0.0.9 development: allowed feature groups by index or name |
| `seed` | `0` | Alias: `random_state` |
| `base_score` | estimated | Starting prediction in the objective's output space (a probability for `binary:logistic` and `reg:logistic`, a raw margin for `binary:logitraw` and `binary:hinge`, a positive mean for log-link objectives, a per-class margin for multiclass). If unset, it is estimated from the training labels as XGBoost 2.0+ does; see [Starting prediction](#starting-prediction-base_score) |
| `cache_budget_bytes` | automatic | Parquet quantized-cache memory budget; single Parquet file only, other inputs raise `ValueError` (see [Control the Parquet quantized cache](#control-the-parquet-quantized-cache)) |
| `cache_path` | `None` | Persistent Parquet cache directory; single Parquet file only, other inputs raise `ValueError` (see [Control the Parquet quantized cache](#control-the-parquet-quantized-cache)) |
| `eval_metric` | objective default | One name or a list; see the metric table under [Validation and early stopping](#validation-and-early-stopping). Early stopping watches the last one |
| `monotone_constraints` | `None` | `1` / `-1` / `0` per feature: tuple, list, `"(1,-1,0)"` string, or `{feature_name: dir}` dict |
| `missing` | `NaN` | Feature value treated exactly like `NaN` (XGBoost's `DMatrix` `missing`), in training, validation, and prediction. Finite `float32` or `NaN`; `inf` and `float32` max are rejected. Not stored in the model: pass the same `missing` to `predict`. See [Missing values](#missing-values-and-the-missing-sentinel) |
| `enable_categorical` | `False` | Detect pandas `category` and PyArrow dictionary columns as categorical features; see [Categorical features](#categorical-features) |
| `max_cat_to_onehot` | `4` | Categorical features with fewer categories than this use one-vs-rest splits, the others sorted-partition splits (`>= 1`). Large values over thousands of categories are slow |
| `max_cat_threshold` | `64` | Most categories a partition split scans per direction (`>= 1`) |

Unknown parameters raise `ValueError`; they are not silently ignored.

The following familiar XGBoost parameters are recognized but not implemented:

- `tree_method` (FerrisBoost currently has only histogram training)
- `num_parallel_tree`

Arguments of `fb.train(params, data, label=None, num_boost_round=10, evals=(), early_stopping_rounds=None, evals_result=None, verbose_eval=False, callbacks=None, header=True, quiet=False, weight=None, feature_types=None, qid=None, group=None, eval_qid=None, eval_group=None, base_margin=None, base_margin_eval_set=None, init_model=None, xgb_model=None)`:
- `num_boost_round`: number of boosting rounds (default `10`; multiclass builds `num_class` trees per round).
- `evals`, `early_stopping_rounds`, `evals_result`, `verbose_eval`, `callbacks`: validation, early stopping, and monitoring; see [Validation and early stopping](#validation-and-early-stopping) and [Custom callbacks](#custom-callbacks).
- `quiet`: Set `True` to suppress routine progress logging on stderr.
- `header`: Set `False` for headerless CSV files using positional column indices.
- `weight`: per-row sample weights; an array for array input, or a column name (column index for headerless CSV) for file input.
- `feature_types`: `"c"` / `"q"` per feature; see [Categorical features](#categorical-features).
- `base_margin`, `base_margin_eval_set`: per-row starting margins; see [Starting margins](#starting-margins-base_margin).
- `init_model` (alias `xgb_model`): a model, a model JSON path or an `xgboost.Booster` to continue; see [Continue training](#continue-training-init_model).
- `qid`, `group`, `eval_qid`, `eval_group`: query groups for the ranking objectives; see [Learning to rank](#learning-to-rank).

Keyword options of `Model.predict(data, output_margin=False, n_trees=None, header=True, *, predict_threads=0, use_gpu=False, iteration_range=None, pred_contribs=False, approx_contribs=False, pred_interactions=False, pred_leaf=False, missing=nan, base_margin=None, strict_shape=False)` are described under [Prediction](#prediction) (categorical input: [How categorical input is coded](#how-categorical-input-is-coded)), [Feature importance, SHAP values, and leaf indices](#feature-importance-shap-values-and-leaf-indices), and [The `missing` sentinel in prediction](#the-missing-sentinel-in-prediction). `Model.predict_to_parquet` takes `output_margin`, `n_trees`, `header`, `predict_threads`, `prediction_column`, `iteration_range`, `missing`, and `base_margin` (a column name). `Model.save_model(path, quiet=None, model_io_threads=None)` and `Model.load_model(path, quiet=False, model_io_threads=0, ingest_threads=0)` take the thread and logging options described under [Save, load, and exchange models with XGBoost](#save-load-and-exchange-models-with-xgboost).

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

Training validates labels before the first round: `binary:logistic`, `binary:logitraw`,
and `binary:hinge` need labels in `[0, 1]`; `reg:squaredlogerror` needs finite
labels greater than `-1`; `count:poisson` and `reg:tweedie` need non-negative labels,
`reg:gamma` needs strictly positive labels, and multiclass objectives need
integer labels in `[0, num_class)`. `rank:ndcg` with `ndcg_exp_gain=True` needs
integer relevance in `[0, 31]`, and `rank:map` (and the `map` / `pre` metrics)
need binary 0 / 1 labels.

### Which exception type is raised

Rejected input raises `ValueError`: invalid parameters, labels that do not fit the
objective (including ranking relevance labels), invalid weights or `base_margin`,
a `base_score`, `feature_types` or objective that conflicts with the model being
continued, missing or surplus columns in a file, unsupported column types or
reserved values in a file, and a persistent cache that does not match the call.
`RuntimeError` is reserved for failures that are not the caller's input: I/O
errors while reading or writing, CUDA and device failures, and internal errors. A
model file that does not exist raises `FileNotFoundError` (a data path that matches
no file is rejected input, `ValueError`), an exception raised inside a callback
propagates with its own type, and a wrong Python type for an argument raises
`TypeError`.

### An XGBoost model has no feature names

An XGBoost model JSON trained on an array has no feature names and records no
FerrisBoost schema mode, so `predict`, `predict_to_parquet`, explanations and
`fb.train(..., init_model=)` on **files** raise `ValueError`. Use an array (you own
the column order), or give the model names so that columns are bound by name:

```python
booster = xgboost.Booster()
booster.load_model("model.json")
booster.feature_names = ["age", "income", "tenure"]   # the training column order
booster.save_model("model_named.json")
```

### Ranking needs query groups

`rank:ndcg`, `rank:pairwise` and `rank:map` raise `ValueError` (asking for
`qid=` / `group=`) when neither `qid` nor `group` is given (a `qid` column name for file input), when `qid` is
not sorted along the rows, when group sizes do not add up to the row count, or
when the weights are not one per query group. Validation sets need
`eval_qid` / `eval_group` too. See [Learning to rank](#learning-to-rank).

### A multiclass objective is missing `num_class`

`multi:softprob` and `multi:softmax` need `num_class >= 2` in the parameters.
`num_class` must also be larger than the largest label.

### A metric does not fit the objective

Binary metrics (`logloss`, `error`, `error@t`) require `binary:logistic` or
`reg:logistic`; `auc` and `aucpr` also accept the ranking objectives. `mlogloss` and `merror` require a multiclass objective, and
multiclass objectives accept only those two. `quantile` requires
`reg:quantileerror`. `tweedie-nloglik` must be written with its variance power,
such as `tweedie-nloglik@1.5`.

### All labels are zero with a log-link objective

`count:poisson`, `reg:gamma`, and `reg:tweedie` start from `log(base_score)`, so
`base_score` must be positive. The label mean is zero when every label is `0`,
so pass an explicit positive value, for example `"base_score": 0.1`.

### A feature equals the largest float32 value

Feature values exactly equal to `numpy.finfo(numpy.float32).max` are rejected
in training, validation and prediction, and the error names the feature
column (or, for NumPy input, the row and column). That value is reserved for
split boundaries. Rescale or clip the feature, for example to
`numpy.nextafter(numpy.float32(numpy.finfo(numpy.float32).max), 0)`, which is
accepted. File prediction only reads the columns the model uses, so a value in
a column the model never reads is not checked and cannot affect the result.

### `missing` is rejected

`missing` must be `NaN` or a finite `float32` value. `inf`, `-inf`,
`numpy.finfo(numpy.float32).max`, and numbers that overflow `float32` raise
`ValueError` naming `missing`, in `fb.train`, `Model.predict`,
`predict_to_parquet`, and the scikit-learn constructors (at `fit`). Use a
different sentinel such as `-999`. Setting a sentinel does not relax the other
feature rules: a feature that is itself `+/-inf` or `float32` max is still an
error.

### `reg:quantileerror` needs a valid `quantile_alpha`

`reg:quantileerror` raises `ValueError` without `quantile_alpha`, with a value
outside `[0, 1]`, with a list that is not in ascending order (XGBoost has the
same rule), or with more than 64 values. Pass one quantile, for example `0.9`,
or a sorted list such as `[0.1, 0.5, 0.9]`.

### A categorical column is rejected

- `categorical feature ... contains the value ...`: categorical columns must
  hold integer codes in `0..=65533`, with `NaN` or null for a missing value.
  Negative, fractional, or larger codes are rejected; re-encode the categories
  compactly. A feature can have at most 65,534 categories.
- `... is a category dtype` / `... is a dictionary type`: set
  `params["enable_categorical"] = True` (or declare the column with
  `feature_types`) so a categorical column is never trained as one by accident.
- `feature_types` must have one `"c"` or `"q"` per feature; for files it counts
  feature columns only (not the label or weight column).
- `... was inferred as Utf8; only numeric columns are supported ...` (CSV):
  declare the text column `"c"` in `feature_types` to train it as named
  categories, or fix stray characters in a numeric column.
- `... holds the category ..., which is not in the model's categories` /
  `... not in the training categories`: a validation or prediction row holds a
  category the training data did not have. XGBoost rejects it too; map such
  values to a known category or to missing before scoring.
- `... the category data type must match the training data`: the model's
  categories are strings and the input's are integers, or the reverse.

### An explanation option is rejected

`pred_contribs`, `pred_interactions`, and `pred_leaf` allow only one per call and
cannot be combined with `output_margin`. `pred_interactions=True` together with
`approx_contribs=True` is rejected because XGBoost 3.4.1 returns only a diagonal
matrix there.

### `iteration_range` is rejected

`iteration_range=(begin, end)` counts boosting rounds. It raises `ValueError`
when `end` is beyond the model's rounds, when `begin > end`, or when `n_trees`
is also given.

### `cache_path` supports a single Parquet file

The quantized cache is built only when training on one Parquet file. With
several Parquet files, CSV, or in-memory data, `cache_path` and
`cache_budget_bytes` would have no effect, so they raise instead of being
ignored. Merge the files into one Parquet file, or drop the option.

### A persistent cache is rejected with a `missing` message

The cache was quantized with a different `missing` than the current call.
Bins built for one sentinel are wrong for another, so use a new `cache_path` or
remove the old cache.

### Training stops with a non-finite gradient error

Some valid but extreme inputs make a gradient or Hessian overflow to `inf` or
`NaN` during training, for example `reg:gamma` with a tiny explicit
`base_score`, very large sample weights, or huge labels with a log-link
objective. Training then stops with an error mentioning a non-finite gradient
or Hessian instead of producing a corrupt model. Reduce the extreme labels,
weights, `base_score`, or objective parameters; the estimated default
`base_score` avoids the most common case.

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
