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
    """训一个模型。data 收 numpy 二维数组、.parquet / .csv 路径,或 pyarrow Table。"""
    ...
