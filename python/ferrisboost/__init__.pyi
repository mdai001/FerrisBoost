from typing import Any, Callable, Iterable, Mapping, Sequence

from ._core import Model as Model

__version__: str

def train(
    params: Mapping[str, Any],
    data: Any,
    label: Any = ...,
    num_boost_round: int = ...,
    evals: Sequence[tuple] = ...,
    early_stopping_rounds: int | None = ...,
    evals_result: dict | None = ...,
    verbose_eval: bool | int = ...,
    callbacks: Iterable[Callable[[int, dict], bool]] | None = ...,
    header: bool = ...,
    quiet: bool = ...,
    weight: Any = ...,
) -> Model:
    """Train a model from arrays, file paths, or a PyArrow table.

    weight gives per-row sample weights (XGBoost's DMatrix weight): a 1-D array
    for array input, or a column name (column index for headerless CSV) for
    file input. Array evals accept (data, label, name) or
    (data, label, weight, name); file evals are (path, name) and reuse the
    training label and weight columns.

    params["objective"] is one of binary:logistic, reg:squarederror,
    count:poisson, reg:gamma, reg:tweedie, reg:pseudohubererror,
    multi:softprob, multi:softmax. Multiclass objectives require
    params["num_class"] (>= 2) and integer labels in [0, num_class); each
    round builds num_class trees. Objective-specific parameters:
    tweedie_variance_power (default 1.5, in [1, 2)) and huber_slope (default 1,
    non-zero). Other regularization parameters follow XGBoost: alpha (alias
    reg_alpha), max_delta_step (count:poisson defaults to 0.7 when unset) and
    scale_pos_weight (binary:logistic, reg:squarederror, reg:gamma only).
    When params["base_score"] is unset it is estimated from the training
    labels, as in XGBoost 2.0+; an explicit value is used as given.

    params["eval_metric"] takes a metric name or list of names: rmse, mae,
    mape, mphe, logloss, error, error@t, auc, aucpr, poisson-nloglik,
    gamma-nloglik, gamma-deviance, tweedie-nloglik@rho, mlogloss, merror.
    Multiclass objectives accept only mlogloss and merror. Without
    eval_metric the objective picks the default metric. params["monotone_constraints"]
    takes 1 / -1 / 0 per feature as a tuple, list, "(1,-1,0)" string, or
    {feature_name: direction} dict.

    Early stopping takes effect with validation data, watches the last metric
    of the last eval set, and retains all trees trained through the stopping
    round; best_iteration identifies the best round (a round index, so a
    multiclass model has (best_iteration + 1) * num_class trees up to it).
    Weights also shape the quantile sketch that sets bin boundaries, as in
    XGBoost.
    """
    ...
