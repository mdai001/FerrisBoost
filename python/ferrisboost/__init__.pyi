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

    params["eval_metric"] takes a metric name or list of names: rmse, logloss,
    error, auc. Early stopping takes effect with validation data, watches the
    last metric of the last eval set, and retains all trees trained through the
    stopping round; best_iteration identifies the best round. Weights also
    shape the quantile sketch that sets bin boundaries, as in XGBoost.
    """
    ...
