"""Rust 扩展的类型声明。用户一般不直接用这一层,走 ferrisboost.train。"""
from os import PathLike
from typing import Any, Callable, Sequence

import numpy as np
from numpy.typing import NDArray

__version__: str

class Model:
    @property
    def best_iteration(self) -> int | None: ...
    @property
    def best_score(self) -> float | None: ...
    @property
    def num_trees(self) -> int: ...
    @property
    def n_features(self) -> int: ...
    def predict(
        self,
        x: NDArray[np.float32] | str | PathLike[str] | Sequence[str | PathLike[str]],
        output_margin: bool = False,
        n_trees: int | None = None,
        header: bool = True,
        *,
        predict_threads: int = 0,
        use_gpu: bool = False,
    ) -> NDArray[np.float32]:
        """Return ordered 1-D float32 predictions from an array or path specs.

        File inputs accept PathLike, lists, directories, and globs. Pass
        header=False for headerless CSV. All trees are used unless n_trees is
        set; output_margin=True returns raw scores. predict_threads=0 selects
        auto CPU workers. use_gpu=True permits GPU 0 scoring in CUDA builds,
        with CPU selection for batches where GPU overhead is not worthwhile.
        """
        ...
    def predict_to_parquet(
        self,
        data: str | PathLike[str] | Sequence[str | PathLike[str]],
        output_path: str,
        output_margin: bool = False,
        n_trees: int | None = None,
        header: bool = True,
        *,
        predict_threads: int = 0,
        prediction_column: str = "prediction",
    ) -> int:
        """CPU-score file inputs and write ordered float32 predictions to Parquet.

        The target must not exist. Return the number of rows written.
        """
        ...
    def save_model(
        self, path: str, quiet: bool | None = None, model_io_threads: int | None = None
    ) -> None:
        """Save XGBoost-compatible JSON using the model's thread setting by default.

        model_io_threads=None inherits the model setting; zero selects auto
        workers. JSON serialization remains serial.
        """
        ...
    @staticmethod
    def load_model(
        path: str, quiet: bool = False, model_io_threads: int = 0,
        ingest_threads: int = 0,
    ) -> Model:
        """Load XGBoost-compatible JSON; zero thread counts select auto workers, while JSON parsing remains serial."""
        ...
    def evals_result(self) -> dict[str, dict[str, list[float]]]: ...

def train_dense(
    x: NDArray[np.float32],
    y: NDArray[np.float32],
    params: dict[str, Any],
    evals: Sequence[
        tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32] | None, str]
    ] | None = ...,
    early_stopping_rounds: int | None = ...,
    verbose_eval: int | None = ...,
    callbacks: Sequence[Callable[[int, dict], bool]] | None = ...,
    weight: NDArray[np.float32] | None = ...,
) -> Model:
    """Train a dense model; early stopping needs evals and retains all trained trees."""
    ...
def train_file(
    path: Sequence[str],
    label: str | int,
    params: dict[str, Any],
    header: bool = ...,
    evals: Sequence[tuple[str, str]] | None = ...,
    early_stopping_rounds: int | None = ...,
    verbose_eval: int | None = ...,
    callbacks: Sequence[Callable[[int, dict], bool]] | None = ...,
    weight: str | int | None = ...,
) -> Model:
    """Train from files; early stopping needs evals and retains all trained trees."""
    ...
