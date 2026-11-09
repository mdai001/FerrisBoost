from typing import Any, Callable, Iterable, Mapping, Sequence

from ._core import Model as Model
from .sklearn import FerrisBoostClassifier as FerrisBoostClassifier
from .sklearn import FerrisBoostRegressor as FerrisBoostRegressor
from .sklearn import FerrisBoostRanker as FerrisBoostRanker

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
    feature_types: Sequence[str] | None = ...,
    base_margin: Any = ...,
    base_margin_eval_set: Sequence[Any] | None = ...,
    init_model: Any = ...,
    xgb_model: Any = ...,
    qid: Any = ...,
    group: Any = ...,
    eval_qid: Sequence[Any] | None = ...,
    eval_group: Sequence[Any] | None = ...,
) -> Model:
    """Train a model from arrays, file paths, a pandas DataFrame, or a PyArrow table.

    base_margin (XGBoost's DMatrix base_margin) is a per-row starting raw
    margin: training starts every row's prediction from it instead of from
    base_score. Arrays give shape (n,) ((n, num_class) for multiclass); file
    inputs give the column name (a list of num_class names for multiclass, a
    column index for headerless CSV), and that column is not a feature. As in
    XGBoost the intercept is still estimated from the labels (or taken from
    params["base_score"]) and stored in the model; the margin only replaces it
    as the starting point, so prediction must be given the same margin
    (Model.predict(..., base_margin=...)). base_margin_eval_set is a list
    parallel to evals with one margin (or None) per validation set; file evals
    read the same column name(s) from their own files.

    init_model continues training (xgb_model= is accepted too, XGBoost's
    name for it): a ferrisboost.Model, a path to a FerrisBoost
    or XGBoost model JSON, or an xgboost.Booster. Training starts from that
    model's prediction on the new training data and on every validation set,
    keeps its base_score and objective (params["objective"] defaults to the
    model's), and appends num_boost_round NEW trees; the returned model holds
    all of them. The objective, num_class, feature count, feature names (data
    with names is bound to the model's names) and categorical feature types
    must agree with the model or ValueError is raised. Bin boundaries are
    sketched afresh from the new data, which is safe because the existing
    trees split on stored float thresholds and are scored on the raw values.
    Combined with base_margin the starting point is base_margin plus the
    existing trees. evals_result covers only the new rounds, round indices
    passed to callbacks start at 0, and early stopping's best_iteration is the
    absolute round in the whole model (as in XGBoost). A stale best_iteration
    of the continued model is kept if early stopping is not used. File inputs
    are bound to the model by feature name (any column order), so the model
    needs feature names; an XGBoost model has them when it was trained with
    feature_names. A model without them raises ValueError for file input.

    weight gives per-row sample weights (XGBoost's DMatrix weight): a 1-D array
    for array input, or a column name (column index for headerless CSV) for
    file input. Array evals accept (data, label, name) or
    (data, label, weight, name); file evals are (path, name) and reuse the
    training label and weight columns.

    params["objective"] is one of binary:logistic, reg:squarederror,
    count:poisson, reg:gamma, reg:tweedie, reg:pseudohubererror,
    reg:quantileerror, multi:softprob, multi:softmax, rank:ndcg, rank:pairwise,
    rank:map. Multiclass objectives require
    params["num_class"] (>= 2) and integer labels in [0, num_class); each
    round builds num_class trees. Objective-specific parameters:
    tweedie_variance_power (default 1.5, in [1, 2)), huber_slope (default 1,
    non-zero) and quantile_alpha (required for reg:quantileerror: a quantile in
    [0, 1], or an ascending list of quantiles for one multi-output model with
    one tree per quantile per round; gpu_math="fast" is supported).
    Other regularization parameters follow XGBoost: alpha (alias
    reg_alpha), max_delta_step (count:poisson defaults to 0.7 when unset) and
    scale_pos_weight (binary:logistic, reg:squarederror, reg:gamma only).
    When params["base_score"] is unset it is estimated from the training
    labels, as in XGBoost 2.0+; an explicit value is used as given.

    params["eval_metric"] takes a metric name or list of names: rmse, mae,
    mape, mphe, logloss, error, error@t, auc, aucpr, poisson-nloglik,
    gamma-nloglik, gamma-deviance, tweedie-nloglik@rho, quantile, mlogloss, merror.
    Multiclass objectives accept only mlogloss and merror; the ranking
    objectives accept ndcg, ndcg@k, ndcg@k-, map, map@k and pre@k. Without
    eval_metric the objective picks the default metric. params["monotone_constraints"]
    takes 1 / -1 / 0 per feature as a tuple, list, "(1,-1,0)" string, or
    {feature_name: direction} dict.

    Ranking (rank:ndcg, rank:pairwise, rank:map) needs query groups: qid (one
    query id per row, rows sorted by it) or group (the size of every group) for
    array inputs, and a qid column name for files (excluded from the features,
    like label). Weights are per query group. Validation sets take eval_qid /
    eval_group, lists parallel to evals. XGBoost's lambdarank_pair_method
    ("topk" or "mean"), lambdarank_num_pair_per_sample, lambdarank_normalization,
    lambdarank_score_normalization, ndcg_exp_gain and lambdarank_unbiased /
    lambdarank_bias_norm (unbiased LambdaMART) are
    supported. The default metric is ndcg@32 (map@32 for rank:map); auc and
    aucpr are scored per query group.

    colsample_bytree, colsample_bylevel and colsample_bynode (each in (0, 1])
    nest as in XGBoost (tree set, then per-level subset, then per-node subset);
    the random sequence is FerrisBoost's own and independent of threads and
    backend. bylevel/bynode only narrow the split search.

    Early stopping takes effect with validation data, watches the last metric
    of the last eval set, and retains all trees trained through the stopping
    round; best_iteration identifies the best round (a round index, so a
    multiclass model has (best_iteration + 1) * num_class trees up to it).
    Features must be finite or NaN (missing); the exact largest float32 value
    is reserved for split boundaries and rejected in training, validation and
    prediction. If extreme inputs make a gradient or Hessian non-finite,
    training raises an error instead of producing a corrupt model.
    Weights also shape the quantile sketch that sets bin boundaries, as in
    XGBoost.

    Categorical features follow XGBoost's hist semantics. feature_types gives
    "c" (categorical) or "q" (numeric) per feature, for arrays, tables, and
    files (positional over the feature columns). With
    params["enable_categorical"] = True, pandas category and PyArrow dictionary
    columns are detected as categorical; Parquet dictionary columns of integers
    are always categorical. Category values are integer codes in 0..=65533
    (NaN / null = missing), so a feature has at most 65,534 categories, on CPU
    and GPU; anything else raises ValueError. params["max_bin"] is 1..=65535. params["max_cat_to_onehot"]
    (default 4) and params["max_cat_threshold"] (default 64) match XGBoost.

    params["missing"] (default NaN) is XGBoost's DMatrix `missing`: a feature
    value equal to it is treated exactly like NaN in the training data and in
    every validation set, for arrays, DataFrames, PyArrow tables and files
    (Parquet / CSV are handled by the Rust readers while streaming, so nothing
    is materialized in Python). Sentinel values never become sketch cut points
    or categories. It must be NaN or a finite float32; +/-inf and the reserved
    f32::MAX raise ValueError. It also applies to the integer codes of
    categorical columns. The model does not store it (XGBoost does not either):
    pass the same `missing` to Model.predict / predict_to_parquet. A quantized
    cache_path is only reused with the same `missing`.

    params["cache_path"] and params["cache_budget_bytes"] apply only to training
    on a single Parquet file; several files, CSV and in-memory data raise
    ValueError instead of ignoring them.
    """
    ...
