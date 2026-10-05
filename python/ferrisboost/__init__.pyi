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
) -> Model:
    """Train a model from arrays, file paths, or a PyArrow table.

    Early stopping takes effect with validation data and retains all trees
    trained through the stopping round; best_iteration identifies the best
    round. The Python API currently uses the objective's one default metric and
    has no eval_metric parameter.
    """
    ...
