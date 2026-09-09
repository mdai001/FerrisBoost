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
        """从二维数组或文件路径预测；无表头 CSV 传 header=False。"""
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
    ) -> int: ...
    def save_model(
        self, path: str, quiet: bool | None = None, model_io_threads: int | None = None
    ) -> None: ...
    @staticmethod
    def load_model(
        path: str, quiet: bool = False, model_io_threads: int = 0,
        ingest_threads: int = 0,
    ) -> Model: ...
    def evals_result(self) -> dict[str, dict[str, list[float]]]: ...

def train_dense(
    x: NDArray[np.float32],
    y: NDArray[np.float32],
    params: dict[str, Any],
    evals: Sequence[tuple[NDArray[np.float32], NDArray[np.float32], str]] | None = ...,
    early_stopping_rounds: int | None = ...,
    verbose_eval: int | None = ...,
    callbacks: Sequence[Callable[[int, dict], bool]] | None = ...,
) -> Model: ...
def train_file(
    path: Sequence[str],
    label: str | int,
    params: dict[str, Any],
    header: bool = ...,
    evals: Sequence[tuple[str, str]] | None = ...,
    early_stopping_rounds: int | None = ...,
    verbose_eval: int | None = ...,
    callbacks: Sequence[Callable[[int, dict], bool]] | None = ...,
) -> Model: ...
