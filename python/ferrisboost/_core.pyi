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
    @property
    def feature_names(self) -> list[str]:
        """Feature names; empty for a model trained from positional (numpy) data."""
        ...
    @property
    def objective(self) -> str:
        """XGBoost objective name, for example "binary:logistic"."""
        ...
    @property
    def quantile_alpha(self) -> list[float] | None:
        """The quantile_alpha list of a reg:quantileerror model; None otherwise."""
        ...
    @property
    def num_class(self) -> int:
        """Number of outputs: classes for multiclass models, quantiles for a
        multi-quantile model (several quantile_alpha values); 1 otherwise."""
        ...
    def predict(
        self,
        data: NDArray[np.float32] | str | PathLike[str] | Sequence[str | PathLike[str]] | Any,
        output_margin: bool = False,
        n_trees: int | None = None,
        header: bool = True,
        *,
        predict_threads: int = 0,
        use_gpu: bool = False,
        iteration_range: tuple[int, int] | None = None,
        pred_contribs: bool = False,
        approx_contribs: bool = False,
        pred_interactions: bool = False,
        pred_leaf: bool = False,
        missing: float = ...,
        base_margin: Any = None,
        strict_shape: bool = False,
    ) -> NDArray[np.float32]:
        """Return ordered float32 predictions from an array or path specs.

        base_margin (XGBoost semantics) is a per-row starting raw margin that
        replaces the model's base_score: each row is base_margin plus its leaf
        values, accumulated in tree order. Arrays give shape (n,) ((n,
        num_class) for multiclass); file inputs give the column name (a list of
        num_class names for multiclass) of a column that sits next to the
        features, which requires a model with feature names. It also becomes
        the bias term of pred_contribs / pred_interactions. pred_leaf ignores
        it. use_gpu=True accepts base_margin too (the device starts each row
        from its margin; results equal the CPU).

        The result is 1-D with shape (n,), except for multiclass models:
        multi:softprob returns (n, num_class) probabilities, multi:softmax
        returns (n,) class indices (as float32), and output_margin=True returns
        (n, num_class) raw margins for either multiclass objective. A model with
        several quantile_alpha values returns (n, n_quantiles), each row sorted
        ascending (output_margin=True gives the unsorted per-quantile scores).

        File inputs accept PathLike, lists, directories, and globs. Pass
        header=False for headerless CSV. All trees are used unless n_trees is
        set; n_trees counts trees, so a multiclass model uses num_class trees
        per boosting round. output_margin=True returns raw scores (before the
        objective's transform). iteration_range=(begin, end) uses the half-open
        range of boosting rounds [begin, end) as XGBoost does (a multiclass round
        is num_class trees); end=0 means up to the last round, an out-of-range
        value raises ValueError, begin == end gives base_score only, and it cannot
        be combined with n_trees. predict_threads=0 selects auto CPU workers.
        use_gpu=True permits GPU 0 scoring in CUDA builds, with CPU selection
        for batches where GPU overhead is not worthwhile.

        Feature values exactly equal to the largest float32 are rejected, as in
        training (the value is reserved for split boundaries); +/-inf and NaN
        follow the usual rules.

        missing (default NaN) is XGBoost's DMatrix `missing`: feature values
        equal to it are scored exactly like NaN (the missing direction), for
        arrays, DataFrames, tables and file inputs (the file readers apply it
        while streaming). It is not stored in the model, so pass the same value
        used in training; it must be NaN or a finite float32 (+/-inf and the
        reserved f32::MAX are rejected). It also applies to the integer codes of
        categorical columns.

        pandas DataFrames and PyArrow tables are accepted too. For a model
        trained on named columns they are matched to the model's features by
        name (any order; extra columns ignored; a missing one is an error).
        Categorical columns are scored through their category codes, so they must carry the
        training categories in the training order. Models with categorical
        splits always score on the CPU, even with use_gpu=True.

        strict_shape=True (XGBoost's strict_shape) makes the shape independent
        of the model type: the output-group axis is always present. Values and
        margins are (n, groups) (groups is 1 for regression, binary, ranking,
        single-quantile and multi:softmax), pred_contribs (n, groups, F + 1),
        pred_interactions (n, groups, F + 1, F + 1) and pred_leaf (n, rounds,
        groups, 1). It reshapes the finished result (a view); pred_leaf needs
        n_trees to be a whole number of rounds.

        Explanations (XGBoost 3.4.1 semantics; at most one per call, not
        combinable with output_margin; n_trees and predict_threads apply; same
        inputs as plain prediction). They run on the CPU, or on GPU 0 with
        use_gpu=True (every call with rows, categorical splits and base_margin
        included) with results bit-identical to the CPU, in bounded device
        memory:

        * pred_contribs=True: exact TreeSHAP in margin space, shape
          (n, n_features + 1) with the bias last, or (n, num_class,
          n_features + 1) for multiclass. Each row sums to the raw margin.
        * approx_contribs=True (with pred_contribs): Saabas attribution
          instead of TreeSHAP. Ignored unless pred_contribs is set; raises
          with pred_interactions, where XGBoost 3.4.1 returns only a diagonal.
        * pred_interactions=True: SHAP interaction values, shape
          (n, n_features + 1, n_features + 1), multiclass adds a class axis
          after the row axis. Memory is n * (n_features + 1) ** 2 floats.
        * pred_leaf=True: float32 node index of the leaf reached in each
          tree, shape (n, n_trees).
        """
        ...
    def get_score(self, importance_type: str = "weight") -> dict[str, float]:
        """Feature importance as in XGBoost's Booster.get_score.

        importance_type is "weight" (split count), "gain", "cover" (averages
        per split), "total_gain" or "total_cover". Features never used in a
        split are omitted; keys are the model's feature names, or "f0",
        "f1", ... when it has none.
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
        iteration_range: tuple[int, int] | None = None,
        missing: float = ...,
        base_margin: str | Sequence[str] | None = None,
    ) -> int:
        """CPU-score file inputs and write ordered float32 predictions to Parquet.

        The target must not exist. Return the number of rows written. Multiclass
        models and multi-quantile models write columns {prediction_column}_0,
        {prediction_column}_1, ... (multi:softprob, output_margin=True, or one
        column per quantile) or one class-index column (multi:softmax). Every
        output streams in bounded batches (the input is read once; each batch is
        scored for all outputs and written before the next is read). missing has
        the same meaning as in predict. base_margin names the file column(s) holding
        the per-row starting margin (see predict).
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
    feature_names: Sequence[str] | None = ...,
    qid: NDArray[np.uint64] | None = ...,
    group: NDArray[np.uint64] | None = ...,
    base_margin: NDArray[np.float32] | None = ...,
    eval_base_margins: Sequence[NDArray[np.float32] | None] | None = ...,
    prior_model: Model | None = ...,
) -> Model:
    """Train a dense model; early stopping needs evals and retains all trained trees.

    params["_feature_types"] is an optional list of bools (True = categorical,
    one per feature); ferrisboost.train fills it from feature_types and dtypes.
    """
    ...
def rank_score(
    metric: str,
    pred: NDArray[np.float32],
    label: NDArray[np.float32],
    qid: NDArray[np.uint64],
    weight: NDArray[np.float32] | None = ...,
    ndcg_exp_gain: bool = ...,
) -> float:
    """A ranking metric (ndcg, ndcg@k, ndcg@k-, map, map@k, pre@k) with XGBoost's semantics.

    qid gives the query groups (sorted); weight is one value per group.
    """
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
    qid: str | int | None = ...,
    base_margin: str | int | Sequence[str] | Sequence[int] | None = ...,
    start_margin: NDArray[np.float32] | None = ...,
    eval_start_margins: Sequence[NDArray[np.float32] | None] | None = ...,
    prior_model: Model | None = ...,
) -> Model:
    """Train from files; early stopping needs evals and retains all trained trees."""
    ...
