"""scikit-learn 风格的估计器:`FerrisBoostClassifier` / `FerrisBoostRegressor` / `FerrisBoostRanker`。

和 XGBoost 的 `XGBClassifier` / `XGBRegressor` / `XGBRanker` 同一套用法,但**不占用 xgboost
的名字**(两个库可以同时 import):

=====================  ==============================
XGBoost                FerrisBoost
=====================  ==============================
``XGBClassifier``      ``FerrisBoostClassifier``
``XGBRegressor``       ``FerrisBoostRegressor``
``XGBRanker``          ``FerrisBoostRanker``
=====================  ==============================

    from ferrisboost.sklearn import FerrisBoostClassifier

    clf = FerrisBoostClassifier(n_estimators=200, learning_rate=0.1,
                                early_stopping_rounds=10)
    clf.fit(X, y, eval_set=[(Xv, yv)])
    clf.predict_proba(Xv)

scikit-learn 是**可选依赖**:`import ferrisboost` 不需要它;访问
``ferrisboost.FerrisBoostClassifier`` 或 import 本模块时才会导入 scikit-learn,
缺了就给出明确的 ImportError。

这一层只做参数搬运、标签编码和 sklearn 的校验约定,**不碰任何数值** ——
训练和预测全部走 `ferrisboost.train` / `Model.predict`。与 XGBoost 一致的行为:

* `predict` / `predict_proba` 在早停后**自动只用到 best_iteration**
  (`Booster.predict` 则默认用全部树);
* 二分类 `predict_proba` 是 `(n, 2)`;多分类用 `multi:softprob`;
* `feature_importances_` 按 `importance_type` 归一化;
* `evals_result()` 的键是 `validation_0`、`validation_1`……

刻意的差异:

* 分类标签可以是任意可排序的类别(内部做 `LabelEncoder` 式编码,`classes_` 保存
  原始类别);XGBoost 要求标签恰好是 `0..K-1`。标签已经是 `0..K-1` 时两者行为一致。
* 不支持 `booster=` / `tree_method=` 等 FerrisBoost 没有的选项。

`quantile_alpha` 可以是一个分位数,也可以是升序列表(多输出模型,和 XGBoost 一样:
每轮每个分位数一棵树,`predict` 返回 `(n_samples, n_quantiles)`)。

`missing`、类别特征和 `apply` 与 XGBoost 一致:

* `missing=`(默认 NaN)传给训练(含 `eval_set`)、`predict` / `predict_proba` /
  `apply`;模型里不保存它,所以同一个估计器自动用同一个值;`load_model` 之后要在
  构造时再给同一个 `missing`。
* `enable_categorical` / `feature_types` / `max_cat_to_onehot` / `max_cat_threshold`
  转给 `ferrisboost.train`。pandas `category` 列在 `fit` / `predict` /
  `predict_proba` / `apply` 里都可用;估计器记住训练时的类别表,预测时按**取值**
  重新编码(训练没见过的类别走缺失方向的另一侧,和 XGBoost 一致)。每个特征最多
  65534 个类别(码 0..=65533),`max_bin` 取 1..=65535,CPU 和 GPU 相同。
* `fit(..., base_margin=, base_margin_eval_set=, init_model=)`(XGBoost 的 `xgb_model=` 同样接受)和
  `predict` / `predict_proba(..., base_margin=)` 与 XGBoost 一致:`base_margin` 是每行的
  起始 raw margin(替换 `base_score` 作为起点),`init_model` 接已有的 `Model` / 估计器 /
  模型 JSON 路径 / `xgboost.Booster`,在它后面续训 `n_estimators` 棵新树。
* `apply(X, iteration_range=None)` 返回每棵树的叶子节点号,形状 `(n_samples, n_trees)`,
  float32,与 `XGBModel.apply` 相同;有早停时默认用到 best_iteration。

Learning to rank: `FerrisBoostRanker` (`XGBRanker`), default `rank:ndcg`.
`fit(X, y, qid=..., eval_set=..., eval_qid=...)` takes a per-row query id (rows sorted by
it) or `group=` sizes; a pandas `qid` column in `X` works too. Weights are per query
group. `predict` returns raw scores; `score(X, y)` is the ranking metric (ndcg@32 by
default, or the last `eval_metric`) and needs the query ids as a `qid` column of `X`.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Sequence
from typing import Any

import numpy as np

try:  # pragma: no cover - 依赖缺失时的提示
    from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
    from sklearn.utils.validation import _check_sample_weight, check_is_fitted
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "ferrisboost.sklearn needs scikit-learn; install it first (for example "
        "`uv pip install scikit-learn`). The core ferrisboost.train / Model API does not "
        "depend on scikit-learn."
    ) from exc

try:  # scikit-learn >= 1.6
    from sklearn.utils.validation import validate_data as _validate_data
except ImportError:  # pragma: no cover
    _validate_data = None

from . import (Model, _arrow_table, _as_ref, _categories_wire, _frame_matrix, _is_pandas_frame,
               _load_prior, _model_refs, _table_matrix, _value_ref, train)

__all__ = ["FerrisBoostClassifier", "FerrisBoostRegressor", "FerrisBoostRanker"]

# sklearn 构造参数 -> `ferrisboost.train` 的参数名。值为 None 的一律不传,
# 让 FerrisBoost 自己的默认值生效(和 XGBoost 的 `None = 用库默认` 一致)。
_PARAM_MAP = {
    "max_depth": "max_depth",
    "max_bin": "max_bin",
    "learning_rate": "eta",
    "n_jobs": "nthread",
    "gamma": "gamma",
    "min_child_weight": "min_child_weight",
    "max_delta_step": "max_delta_step",
    "subsample": "subsample",
    "colsample_bytree": "colsample_bytree",
    "colsample_bylevel": "colsample_bylevel",
    "colsample_bynode": "colsample_bynode",
    "reg_alpha": "alpha",
    "reg_lambda": "lambda",
    "scale_pos_weight": "scale_pos_weight",
    "base_score": "base_score",
    "monotone_constraints": "monotone_constraints",
    "eval_metric": "eval_metric",
    "device": "device",
    "gpu_math": "gpu_math",
    "cols_per_block": "cols_per_block",
    "quantile_alpha": "quantile_alpha",
    "tweedie_variance_power": "tweedie_variance_power",
    "huber_slope": "huber_slope",
    "missing": "missing",
    "enable_categorical": "enable_categorical",
    "max_cat_to_onehot": "max_cat_to_onehot",
    "max_cat_threshold": "max_cat_threshold",
    "lambdarank_pair_method": "lambdarank_pair_method",
    "lambdarank_num_pair_per_sample": "lambdarank_num_pair_per_sample",
    "lambdarank_normalization": "lambdarank_normalization",
    "lambdarank_score_normalization": "lambdarank_score_normalization",
    "lambdarank_unbiased": "lambdarank_unbiased",
    "lambdarank_bias_norm": "lambdarank_bias_norm",
    "ndcg_exp_gain": "ndcg_exp_gain",
}

_IMPORTANCE_TYPES = ("weight", "gain", "cover", "total_gain", "total_cover")


def _check_X(est: Any, X: Any, y: Any = None, *, reset: bool, **kwargs: Any):
    """sklearn 的输入校验(NaN 当缺失值放行,inf 拒绝)。"""
    if _validate_data is not None:
        if y is None:
            return _validate_data(est, X, reset=reset, ensure_all_finite="allow-nan", **kwargs)
        return _validate_data(
            est, X, y, reset=reset, ensure_all_finite="allow-nan", **kwargs
        )
    if y is None:  # pragma: no cover
        return est._validate_data(X, reset=reset, force_all_finite="allow-nan", **kwargs)
    return est._validate_data(  # pragma: no cover
        X, y, reset=reset, force_all_finite="allow-nan", **kwargs
    )


def _has_category_column(frame: Any) -> bool:
    import pandas as pd

    return any(isinstance(dt, pd.CategoricalDtype) for dt in frame.dtypes)


def _dictionary_table(X: Any) -> Any:
    """A PyArrow Table / RecordBatch with a dictionary column, else None."""
    if _is_pandas_frame(X) or not hasattr(X, "column_names"):  # not a stream reader either
        return None
    table = _arrow_table(X)
    if table is None:
        return None
    import pyarrow as pa

    return table if any(pa.types.is_dictionary(t) for t in table.schema.types) else None


def _seed_of(random_state: Any) -> int | None:
    if random_state is None:
        return None
    if isinstance(random_state, np.random.RandomState):
        return int(random_state.randint(np.iinfo(np.int32).max))
    if isinstance(random_state, np.random.Generator):
        return int(random_state.integers(np.iinfo(np.int32).max))
    return int(random_state)


class _FerrisBoostBase(BaseEstimator):
    """两个估计器共用的参数、拟合和持久化逻辑。"""

    def __init__(
        self,
        *,
        n_estimators: int | None = None,
        max_depth: int | None = None,
        max_bin: int | None = None,
        learning_rate: float | None = None,
        objective: str | None = None,
        n_jobs: int | None = None,
        gamma: float | None = None,
        min_child_weight: float | None = None,
        max_delta_step: float | None = None,
        subsample: float | None = None,
        colsample_bytree: float | None = None,
        colsample_bylevel: float | None = None,
        colsample_bynode: float | None = None,
        reg_alpha: float | None = None,
        reg_lambda: float | None = None,
        scale_pos_weight: float | None = None,
        base_score: float | None = None,
        random_state: Any = None,
        monotone_constraints: Any = None,
        eval_metric: str | list[str] | tuple[str, ...] | None = None,
        early_stopping_rounds: int | None = None,
        callbacks: list | None = None,
        device: str | None = None,
        gpu_math: str | None = None,
        cols_per_block: int | None = None,
        quantile_alpha: float | Sequence[float] | None = None,
        tweedie_variance_power: float | None = None,
        huber_slope: float | None = None,
        importance_type: str | None = None,
        missing: float = np.nan,
        enable_categorical: bool = False,
        feature_types: Any = None,
        max_cat_to_onehot: int | None = None,
        max_cat_threshold: int | None = None,
        lambdarank_pair_method: str | None = None,
        lambdarank_num_pair_per_sample: int | None = None,
        lambdarank_normalization: bool | None = None,
        lambdarank_score_normalization: bool | None = None,
        lambdarank_unbiased: bool | None = None,
        lambdarank_bias_norm: float | None = None,
        ndcg_exp_gain: bool | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.max_bin = max_bin
        self.learning_rate = learning_rate
        self.objective = objective
        self.n_jobs = n_jobs
        self.gamma = gamma
        self.min_child_weight = min_child_weight
        self.max_delta_step = max_delta_step
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.colsample_bylevel = colsample_bylevel
        self.colsample_bynode = colsample_bynode
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.scale_pos_weight = scale_pos_weight
        self.base_score = base_score
        self.random_state = random_state
        self.monotone_constraints = monotone_constraints
        self.eval_metric = eval_metric
        self.early_stopping_rounds = early_stopping_rounds
        self.callbacks = callbacks
        self.device = device
        self.gpu_math = gpu_math
        self.cols_per_block = cols_per_block
        self.quantile_alpha = quantile_alpha
        self.tweedie_variance_power = tweedie_variance_power
        self.huber_slope = huber_slope
        self.importance_type = importance_type
        self.missing = missing
        self.enable_categorical = enable_categorical
        self.feature_types = feature_types
        self.max_cat_to_onehot = max_cat_to_onehot
        self.max_cat_threshold = max_cat_threshold
        self.lambdarank_pair_method = lambdarank_pair_method
        self.lambdarank_num_pair_per_sample = lambdarank_num_pair_per_sample
        self.lambdarank_normalization = lambdarank_normalization
        self.lambdarank_score_normalization = lambdarank_score_normalization
        self.lambdarank_unbiased = lambdarank_unbiased
        self.lambdarank_bias_norm = lambdarank_bias_norm
        self.ndcg_exp_gain = ndcg_exp_gain

    # ------------------------------------------------------------------ tags
    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags

    # ------------------------------------------------------------------ params
    def _default_objective(self) -> str:  # pragma: no cover - 子类实现
        raise NotImplementedError

    def get_num_boosting_rounds(self) -> int:
        return 100 if self.n_estimators is None else int(self.n_estimators)

    def get_ferrisboost_params(self) -> dict[str, Any]:
        """翻译成 `ferrisboost.train` 的 params 字典(None 的不传)。"""
        params: dict[str, Any] = {
            "objective": self.objective if self.objective is not None else self._default_objective()
        }
        for name, target in _PARAM_MAP.items():
            value = getattr(self, name)
            if value is not None:
                params[target] = value
        seed = _seed_of(self.random_state)
        if seed is not None:
            params["seed"] = seed
        # sklearn's n_jobs=-1 (and any negative value) means "all cores", which is
        # nthread=0 here; Rust takes an unsigned thread count.
        if isinstance(params.get("nthread"), int) and params["nthread"] < 0:
            params["nthread"] = 0
        return params

    # ------------------------------------------------------------------ 输入
    def _prepare_X(
        self, X: Any, y: Any = None, *, reset: bool, prior: Any = None, **kwargs: Any
    ) -> tuple[Any, Any, list[str] | None]:
        """sklearn validation plus pandas `category` handling.

        Returns `(X, y, feature_types)`: `X` is the validated float32 matrix
        (category columns as their integer codes, NaN for a missing value) and
        `feature_types` the per-feature "c" / "q" list detected from the frame
        (None when the frame has no category column). Training stores the
        categories (in the booster too); later calls re-code onto them by
        value. A continued model (`prior`) codes the training frame with its
        own categories. PyArrow tables with dictionary columns are coded as
        `ferrisboost.train` and `Model.predict` code them (an integer
        dictionary of a feature without categories is read by its values).
        Prediction and eval sets read frames and tables as files are read: a
        plain integer column of a feature with integer categories is looked
        up by value, and an integer `category` column of a feature without
        categories yields its values. NumPy input is codes.
        """
        flags = None
        table = _dictionary_table(X)
        frame = _is_pandas_frame(X) and _has_category_column(X)
        prior_model = None
        prior_feature_names = None
        prior_levels = None
        if reset and prior is not None:
            # A continuation may receive plain integer columns whose values
            # are categories in the prior model. Route those frames through
            # the same value-to-code adapters used by prediction/eval.
            prior_levels = getattr(prior, "_cat_levels_", None)
            estimator_names = getattr(prior, "feature_names_in_", None)
            if estimator_names is not None:
                prior_feature_names = [str(n) for n in estimator_names]
            if hasattr(prior, "get_booster"):
                prior = prior.get_booster()
            prior_model = _load_prior(prior)
            if prior_feature_names is None:
                prior_feature_names = list(prior_model.feature_names or []) or None
            if prior_levels is None:
                prior_levels = _model_refs(prior_model)
        eval_levels = None
        if not reset:
            eval_levels = getattr(self, "_cat_levels_", None)
            if eval_levels is None:
                eval_levels = _model_refs(getattr(self, "_Booster", None))
        has_category_refs = bool(prior_levels if reset else eval_levels)
        if reset and prior is not None and has_category_refs:
            if _is_pandas_frame(X):
                frame = True
            elif hasattr(X, "column_names"):
                table = _arrow_table(X)
        feature_order = ((prior_feature_names if reset else
                          [str(n) for n in self.feature_names_in_]
                          if hasattr(self, "feature_names_in_") else None)
                         if has_category_refs else None)
        if feature_order and _is_pandas_frame(X):
            columns = [str(c) for c in X.columns]
            missing = [name for name in feature_order if name not in columns]
            extra = [name for name in columns if name not in feature_order]
            if missing or extra:
                raise ValueError(
                    f"feature names do not match the training data: missing {missing}, extra {extra}"
                )
            X = X[[c for name in feature_order for c in X.columns if str(c) == name]]
            frame = True
        categorical = levels = None
        if not reset:
            # This fit's categories first (eval sets are prepared before the
            # booster exists); a loaded model brings its own.
            levels = eval_levels
            if self.feature_types is not None:
                categorical = [t == "c" for t in self.feature_types]
            if table is None and not frame:
                if _is_pandas_frame(X):
                    # Plain integer columns may be category values. An empty
                    # _cat_levels_ is meaningful: this fit has no category
                    # lists, even if the estimator still holds an older booster.
                    frame = any(
                        _value_ref(_as_ref(lv), i, categorical) is not None
                        for i, lv in (levels or {}).items()
                    )
                elif hasattr(X, "column_names"):
                    # sklearn's validation sets may be Arrow tables with only
                    # numeric columns too; route them through the same dtype-
                    # aware adapter, which preserves integer codes without refs.
                    table = _arrow_table(X)
        if table is not None or frame:
            if reset:
                levels = prior_levels
                if self.feature_types is not None:
                    categorical = [t == "c" for t in self.feature_types]
            if table is not None:
                import pyarrow as pa

                matrix, names, flag_list, new_levels = _table_matrix(
                    table, feature_order, bool(self.enable_categorical), levels, "X",
                    scoring=(not reset or prior is not None),
                    categorical=categorical,
                )
                X = pa.table({n: matrix[:, i] for i, n in enumerate(names)})
            else:
                import pandas as pd

                matrix, _, flag_list, new_levels = _frame_matrix(
                    X, bool(self.enable_categorical), levels, "X",
                    scoring=(not reset or prior is not None),
                    categorical=categorical,
                )
                # A numeric stand-in with the same labels: sklearn then sets
                # `n_features_in_` / `feature_names_in_` exactly as for any frame.
                X = pd.DataFrame(matrix, columns=X.columns, index=X.index)
            flags = ["c" if f else "q" for f in flag_list]
            new = new_levels if reset else None
        else:
            new = {} if reset else None
        if y is None:
            X = _check_X(self, X, reset=reset, **kwargs)
        else:
            X, y = _check_X(self, X, y, reset=reset, **kwargs)
        if new is not None:
            self._cat_levels_ = new
        return X, y, flags

    # ------------------------------------------------------------------ fit
    def _fit_booster(
        self,
        X: np.ndarray,
        y: np.ndarray,
        params: dict[str, Any],
        sample_weight: Any,
        eval_set: Any,
        sample_weight_eval_set: Any,
        verbose: Any,
        eval_labels: Any,
        feature_types: Any = None,
        base_margin: Any = None,
        base_margin_eval_set: Any = None,
        init_model: Any = None,
        ranking: dict | None = None,
    ) -> None:
        # Ranking: `ranking` carries qid / group, the validation sets' query ids and
        # tells that sample weights are per query group (not per row).
        per_group = ranking is not None
        if base_margin_eval_set is not None and not eval_set:
            raise ValueError("base_margin_eval_set was given without eval_set")
        if base_margin_eval_set is not None and len(base_margin_eval_set) != len(eval_set):
            raise ValueError(
                "base_margin_eval_set must have the same length as eval_set: "
                f"{len(base_margin_eval_set)} != {len(eval_set)}"
            )
        if hasattr(init_model, "get_booster"):
            init_model = init_model.get_booster()
        # The booster stores the category names the training frame was coded with.
        types = feature_types if self.feature_types is None else self.feature_types
        wire = _categories_wire(
            getattr(self, "_cat_levels_", None) or {},
            None if types is None else [t == "c" for t in types],
            X.shape[1],
        )
        if wire is not None:
            params = {**params, "_categories": wire}
        if sample_weight is not None:
            if per_group:
                sample_weight = np.asarray(sample_weight, dtype=np.float32).ravel()
            else:
                sample_weight = _check_sample_weight(sample_weight, X, dtype=np.float32)
        evals = []
        if eval_set:
            if sample_weight_eval_set is not None and len(sample_weight_eval_set) != len(eval_set):
                raise ValueError(
                    "sample_weight_eval_set must have the same length as eval_set: "
                    f"{len(sample_weight_eval_set)} != {len(eval_set)}"
                )
            for i, pair in enumerate(eval_set):
                if len(pair) != 2:
                    raise ValueError("Each eval_set element must be an (X, y) pair")
                ex, ey = pair
                ex, _, _ = self._prepare_X(ex, reset=False)
                ey = eval_labels(ey)
                ew = None if sample_weight_eval_set is None else sample_weight_eval_set[i]
                if ew is None:
                    evals.append((ex, ey, f"validation_{i}"))
                else:
                    if per_group:
                        ew = np.asarray(ew, dtype=np.float32).ravel()
                    else:
                        ew = _check_sample_weight(ew, ex, dtype=np.float32)
                    evals.append((ex, ey, ew, f"validation_{i}"))
        elif sample_weight_eval_set is not None:
            raise ValueError("sample_weight_eval_set was given without eval_set")
        if self.early_stopping_rounds is not None and not evals:
            raise ValueError("early_stopping_rounds requires eval_set (at least one validation set)")
        if verbose is True:
            verbose_eval: bool | int = True
        elif not verbose:
            verbose_eval = False
        else:
            verbose_eval = int(verbose)
        result: dict = {}
        self._Booster = train(
            params,
            X,
            label=y,
            num_boost_round=self.get_num_boosting_rounds(),
            evals=evals,
            early_stopping_rounds=self.early_stopping_rounds,
            evals_result=result,
            verbose_eval=verbose_eval,
            callbacks=self.callbacks,
            quiet=not verbose,
            weight=None if sample_weight is None else np.asarray(sample_weight, dtype=np.float32),
            feature_types=feature_types if self.feature_types is None else self.feature_types,
            base_margin=base_margin,
            base_margin_eval_set=base_margin_eval_set,
            init_model=init_model,
            **({} if ranking is None else ranking),
        )
        self.evals_result_ = result

    # ------------------------------------------------------------------ 访问器
    def get_booster(self) -> Model:
        """底层的 `ferrisboost.Model`(未拟合时抛 `NotFittedError`)。"""
        check_is_fitted(self, "_Booster")
        return self._Booster

    @property
    def best_iteration(self) -> int:
        """早停选中的轮次(0 起)。没有早停时和 XGBoost 一样抛 `AttributeError`。"""
        best = self.get_booster().best_iteration
        if best is None:
            raise AttributeError("`best_iteration` is only available on a model trained with early stopping")
        return int(best)

    @property
    def best_score(self) -> float:
        best = self.get_booster().best_score
        if best is None:
            raise AttributeError("`best_score` is only available on a model trained with early stopping")
        return float(best)

    def evals_result(self) -> dict[str, dict[str, list[float]]]:
        """`{"validation_0": {"logloss": [...]}, ...}`;没有 eval_set 时是空字典。"""
        check_is_fitted(self, "evals_result_")
        return self.evals_result_

    @property
    def n_estimators_(self) -> int:
        """实际训练了多少轮(早停时包含 best_iteration 之后的那些)。"""
        model = self.get_booster()
        return model.num_trees // max(1, getattr(model, "num_class", 1) or 1)

    @property
    def feature_importances_(self) -> np.ndarray:
        """Feature importances normalized to sum to 1, one entry per input feature.

        Uses `Model.get_score(importance_type)`; `importance_type` defaults to
        "gain" as in XGBoost's sklearn layer. Multiclass importances are
        accumulated over all classes' trees. All zeros when no split exists.
        """
        kind = self.importance_type or "gain"
        if kind not in _IMPORTANCE_TYPES:
            raise ValueError(f"importance_type must be one of {_IMPORTANCE_TYPES}, got {kind!r}")
        score = self.get_booster().get_score(importance_type=kind)
        # The booster is trained from arrays, so its keys are positional (f0, f1, ...).
        out = np.array(
            [score.get(f"f{i}", 0.0) for i in range(self.n_features_in_)], dtype=np.float32
        )
        total = out.sum()
        return out / total if total > 0 else out

    # ------------------------------------------------------------------ 预测
    def _iteration_range(self, iteration_range: tuple[int, int] | None) -> tuple[int, int]:
        """XGBoost `_get_iteration_range`:没给(或 end == 0)且有早停,就用到最佳轮次。"""
        if iteration_range is None or iteration_range[1] == 0:
            best = self.get_booster().best_iteration
            if best is not None:
                return (0, int(best) + 1)
            return (0, 0)
        return (int(iteration_range[0]), int(iteration_range[1]))

    def _raw_predict(
        self,
        X: Any,
        output_margin: bool = False,
        iteration_range: tuple[int, int] | None = None,
        base_margin: Any = None,
    ) -> np.ndarray:
        check_is_fitted(self, "_Booster")
        X, _, _ = self._prepare_X(X, reset=False)
        return self._Booster.predict(
            np.ascontiguousarray(X, dtype=np.float32),
            output_margin=output_margin,
            iteration_range=self._iteration_range(iteration_range),
            missing=self.missing,
            base_margin=base_margin,
        )

    def apply(
        self, X: Any, iteration_range: tuple[int, int] | None = None
    ) -> np.ndarray:
        """Leaf node index reached in every tree, like `XGBModel.apply`.

        Returns float32 `(n_samples, n_trees)` (a multiclass model has
        `num_class` trees per boosting round). Without `iteration_range`, a model
        trained with early stopping uses its `best_iteration` automatically.
        """
        check_is_fitted(self, "_Booster")
        X, _, _ = self._prepare_X(X, reset=False)
        return self._Booster.predict(
            np.ascontiguousarray(X, dtype=np.float32),
            pred_leaf=True,
            iteration_range=self._iteration_range(iteration_range),
            missing=self.missing,
        )

    # ------------------------------------------------------------------ 持久化
    def save_model(self, fname: str | os.PathLike) -> None:
        """写 XGBoost 兼容的 JSON(只含 booster)。"""
        self.get_booster().save_model(os.fspath(fname), quiet=True)

    def _load_booster(self, fname: str | os.PathLike) -> Model:
        return Model.load_model(os.fspath(fname), quiet=True)

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        booster = state.pop("_Booster", None)
        if booster is not None:
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "model.json")
                booster.save_model(path, quiet=True)
                with open(path, "rb") as fh:
                    state["_booster_json"] = fh.read()
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        state = dict(state)
        raw = state.pop("_booster_json", None)
        self.__dict__.update(state)
        if raw is not None:
            with tempfile.TemporaryDirectory() as tmp:
                path = os.path.join(tmp, "model.json")
                with open(path, "wb") as fh:
                    fh.write(raw)
                self._Booster = self._load_booster(path)


class FerrisBoostRegressor(RegressorMixin, _FerrisBoostBase):
    """回归估计器(对应 `xgboost.XGBRegressor`),默认 `reg:squarederror`。

    支持 `reg:squarederror`、`count:poisson`、`reg:gamma`、`reg:tweedie`、
    `reg:pseudohubererror`、`reg:quantileerror`(需要 `quantile_alpha`:标量,或升序列表得到多输出模型)。
    """

    def _default_objective(self) -> str:
        return "reg:squarederror"

    def fit(
        self,
        X: Any,
        y: Any,
        *,
        sample_weight: Any = None,
        base_margin: Any = None,
        eval_set: Any = None,
        sample_weight_eval_set: Any = None,
        base_margin_eval_set: Any = None,
        init_model: Any = None,
        xgb_model: Any = None,
        verbose: bool | int | None = False,
    ) -> FerrisBoostRegressor:
        from . import _pick_init_model

        init_model = _pick_init_model(init_model, xgb_model)
        if y is None:
            raise ValueError("This FerrisBoostRegressor estimator requires y to be passed, but the target y is None.")
        X, y, flags = self._prepare_X(X, y, reset=True, prior=init_model, y_numeric=True, dtype=np.float32)
        params = self.get_ferrisboost_params()
        self._fit_booster(
            X,
            np.asarray(y, dtype=np.float32),
            params,
            sample_weight,
            eval_set,
            sample_weight_eval_set,
            verbose,
            eval_labels=lambda ey: np.asarray(ey, dtype=np.float32).ravel(),
            feature_types=flags,
            base_margin=base_margin,
            base_margin_eval_set=base_margin_eval_set,
            init_model=init_model,
        )
        return self

    def predict(
        self,
        X: Any,
        *,
        output_margin: bool = False,
        iteration_range: tuple[int, int] | None = None,
        base_margin: Any = None,
    ) -> np.ndarray:
        return self._raw_predict(X, output_margin, iteration_range, base_margin)

    @classmethod
    def from_file(cls, fname: str | os.PathLike, **params: Any) -> FerrisBoostRegressor:
        est = cls(**params)
        est.load_model(fname)
        return est

    def load_model(self, fname: str | os.PathLike) -> FerrisBoostRegressor:
        """读 XGBoost 兼容的 JSON(也能读 XGBoost 训的单输出回归模型)。"""
        self._Booster = self._load_booster(fname)
        if self._Booster.num_class > 1 and self._Booster.objective != "reg:quantileerror":
            raise ValueError("This is a multiclass model; use FerrisBoostClassifier.load_model")
        self.n_features_in_ = self._Booster.n_features
        self.evals_result_ = {}
        self.__dict__.pop("_cat_levels_", None)
        return self


class FerrisBoostClassifier(ClassifierMixin, _FerrisBoostBase):
    """分类估计器(对应 `xgboost.XGBClassifier`),默认 `binary:logistic`。

    类别数 > 2 时自动切到 `multi:softprob`(或你指定的 `multi:softmax`)并设置
    `num_class`。标签可以是任意可排序类别,`classes_` 保存原始类别。
    """

    def _default_objective(self) -> str:
        return "binary:logistic"

    def fit(
        self,
        X: Any,
        y: Any,
        *,
        sample_weight: Any = None,
        base_margin: Any = None,
        eval_set: Any = None,
        sample_weight_eval_set: Any = None,
        base_margin_eval_set: Any = None,
        init_model: Any = None,
        xgb_model: Any = None,
        verbose: bool | int | None = False,
    ) -> FerrisBoostClassifier:
        from . import _pick_init_model

        init_model = _pick_init_model(init_model, xgb_model)
        from sklearn.utils.multiclass import check_classification_targets

        if y is None:
            raise ValueError("This FerrisBoostClassifier estimator requires y to be passed, but the target y is None.")
        X, y, flags = self._prepare_X(X, y, reset=True, prior=init_model, dtype=np.float32)
        check_classification_targets(y)
        classes, encoded = np.unique(y, return_inverse=True)
        if len(classes) < 2:
            raise ValueError(
                "Classification needs samples of at least 2 classes, but the data contains "
                f"only one class: {classes[0]!r}"
            )
        self.classes_ = classes
        self.n_classes_ = len(classes)
        params = self.get_ferrisboost_params()
        if hasattr(init_model, "get_booster"):
            init_model = init_model.get_booster()
        if init_model is not None and self.objective is None:
            # Continuation keeps the objective of the model being continued.
            from . import _load_prior

            params["objective"] = _load_prior(init_model).objective
        if self.n_classes_ > 2:
            if params["objective"] != "multi:softmax":
                params["objective"] = "multi:softprob"
            params["num_class"] = self.n_classes_
        elif params["objective"].startswith("multi:"):
            raise ValueError(
                f"objective={params['objective']!r} needs more than 2 classes, "
                f"but the labels contain {self.n_classes_}"
            )

        def encode_eval(ey: Any) -> np.ndarray:
            ey = np.asarray(ey).ravel()
            idx = np.searchsorted(self.classes_, ey)
            idx = np.clip(idx, 0, self.n_classes_ - 1)
            if not (self.classes_[idx] == ey).all():
                unseen = np.unique(ey[self.classes_[idx] != ey])
                raise ValueError(f"eval_set contains classes not seen in training: {unseen}")
            return idx.astype(np.float32)

        self._fit_booster(
            X,
            encoded.astype(np.float32),
            params,
            sample_weight,
            eval_set,
            sample_weight_eval_set,
            verbose,
            eval_labels=encode_eval,
            feature_types=flags,
            base_margin=base_margin,
            base_margin_eval_set=base_margin_eval_set,
            init_model=init_model,
        )
        self._objective_resolved = params["objective"]
        return self

    def _decode(self, index: np.ndarray) -> np.ndarray:
        return self.classes_[np.asarray(index, dtype=np.intp)]

    def predict(
        self,
        X: Any,
        *,
        output_margin: bool = False,
        iteration_range: tuple[int, int] | None = None,
        base_margin: Any = None,
    ) -> np.ndarray:
        check_is_fitted(self, "_Booster")
        raw = self._raw_predict(X, output_margin, iteration_range, base_margin)
        if output_margin:
            return raw
        if raw.ndim == 2:  # multi:softprob
            return self._decode(np.argmax(raw, axis=1))
        if self._objective_resolved == "multi:softmax":
            return self._decode(raw.astype(np.intp))
        return self._decode((raw > 0.5).astype(np.intp))

    def predict_proba(
        self,
        X: Any,
        *,
        iteration_range: tuple[int, int] | None = None,
        base_margin: Any = None,
    ) -> np.ndarray:
        """每个类别的概率,形状 `(n_samples, n_classes_)`。"""
        check_is_fitted(self, "_Booster")
        if self._objective_resolved == "multi:softmax":
            margin = self._raw_predict(X, True, iteration_range, base_margin)
            shifted = margin - margin.max(axis=1, keepdims=True)
            expd = np.exp(shifted)
            return expd / expd.sum(axis=1, keepdims=True)
        raw = self._raw_predict(X, False, iteration_range, base_margin)
        if raw.ndim == 2:
            return raw
        return np.vstack((1.0 - raw, raw)).T

    def load_model(self, fname: str | os.PathLike) -> FerrisBoostClassifier:
        """读 XGBoost 兼容的 JSON(binary:logistic / multi:softprob / multi:softmax)。"""
        booster = self._load_booster(fname)
        name = booster.objective
        if name not in ("binary:logistic", "multi:softprob", "multi:softmax"):
            raise ValueError(f"{name} is not a classification model; use FerrisBoostRegressor.load_model")
        self._Booster = booster
        self._objective_resolved = name
        self.n_classes_ = booster.num_class if booster.num_class > 1 else 2
        self.classes_ = np.arange(self.n_classes_)
        self.n_features_in_ = booster.n_features
        self.evals_result_ = {}
        self.__dict__.pop("_cat_levels_", None)
        return self

    @classmethod
    def from_file(cls, fname: str | os.PathLike, **params: Any) -> FerrisBoostClassifier:
        est = cls(**params)
        est.load_model(fname)
        return est


def _split_qid(X: Any, qid: Any) -> tuple[Any, Any]:
    """The special `qid` column of a pandas DataFrame (XGBoost convention)."""
    if _is_pandas_frame(X) and "qid" in list(X.columns):
        if qid is not None:
            raise ValueError(
                "Found both the special column `qid` in `X` and the `qid` from the `fit` "
                "method. Please remove one of them."
            )
        return X.drop(columns="qid"), X["qid"].to_numpy()
    return X, qid


class FerrisBoostRanker(_FerrisBoostBase):
    """Learning-to-rank estimator (`xgboost.XGBRanker`), default objective `rank:ndcg`.

    `rank:ndcg`, `rank:pairwise` and `rank:map` with XGBoost's `lambdarank_pair_method`,
    `lambdarank_num_pair_per_sample`, `lambdarank_normalization`,
    `lambdarank_score_normalization`, `ndcg_exp_gain` and the unbiased
    `lambdarank_unbiased` / `lambdarank_bias_norm`.

    `fit(X, y, qid=...)` needs the query groups: `qid` (one id per row, rows sorted by it) or
    `group` (the size of every group). A pandas `qid` column in `X` is used like the `qid`
    argument (also for `eval_set` and `score`). `sample_weight` is **per query group**. The
    default metric is `ndcg@32` (`map@32` for `rank:map`), as in XGBoost. `predict` returns
    the raw ranking scores.
    """

    def _default_objective(self) -> str:
        return "rank:ndcg"

    def fit(
        self,
        X: Any,
        y: Any,
        *,
        qid: Any = None,
        group: Any = None,
        sample_weight: Any = None,
        base_margin: Any = None,
        eval_set: Any = None,
        eval_qid: Any = None,
        eval_group: Any = None,
        sample_weight_eval_set: Any = None,
        base_margin_eval_set: Any = None,
        init_model: Any = None,
        xgb_model: Any = None,
        verbose: bool | int | None = False,
    ) -> FerrisBoostRanker:
        from . import _pick_init_model

        init_model = _pick_init_model(init_model, xgb_model)
        if y is None:
            raise ValueError("This FerrisBoostRanker estimator requires y to be passed, but the target y is None.")
        objective = self.objective if self.objective is not None else self._default_objective()
        if callable(objective) or "rank:" not in str(objective):
            raise ValueError("please use FerrisBoostRegressor / FerrisBoostClassifier for a non-ranking objective")
        X, qid = _split_qid(X, qid)
        if qid is None and group is None:
            raise ValueError("Either `group` or `qid` is required for ranking task")
        if qid is not None and group is not None:
            raise ValueError("Either one of `group` or `qid` should be None.")
        X, y, flags = self._prepare_X(X, y, reset=True, prior=init_model, y_numeric=True, dtype=np.float32)
        eval_set = list(eval_set) if eval_set else None
        eq_list: list = []
        eg_list: list = []
        if eval_set:
            if eval_qid is not None and len(eval_qid) != len(eval_set):
                raise ValueError("eval_qid must have the same length as eval_set")
            if eval_group is not None and len(eval_group) != len(eval_set):
                raise ValueError("eval_group must have the same length as eval_set")
            for i, pair in enumerate(eval_set):
                if len(pair) != 2:
                    raise ValueError("Each eval_set element must be an (X, y) pair")
                ex, ey = pair
                ex, eq = _split_qid(ex, None if eval_qid is None else eval_qid[i])
                eg = None if eval_group is None else eval_group[i]
                if eq is not None and eg is not None:
                    raise ValueError(
                        "Only one of the eval_qid or eval_group for each evaluation dataset should be provided."
                    )
                if eq is None and eg is None:
                    raise ValueError(
                        f"eval_set[{i}] needs query groups: pass eval_qid / eval_group "
                        "or a `qid` column in its X"
                    )
                eval_set[i] = (ex, ey)
                eq_list.append(eq)
                eg_list.append(eg)
        elif eval_qid is not None or eval_group is not None:
            raise ValueError("eval_qid / eval_group were given without eval_set")
        params = self.get_ferrisboost_params()
        self._fit_booster(
            X,
            np.asarray(y, dtype=np.float32),
            params,
            sample_weight,
            eval_set,
            sample_weight_eval_set,
            verbose,
            eval_labels=lambda ey: np.asarray(ey, dtype=np.float32).ravel(),
            feature_types=flags,
            base_margin=base_margin,
            base_margin_eval_set=base_margin_eval_set,
            init_model=init_model,
            ranking={
                "qid": qid,
                "group": group,
                "eval_qid": eq_list if eval_set else None,
                "eval_group": eg_list if eval_set else None,
            },
        )
        return self

    def predict(
        self,
        X: Any,
        *,
        output_margin: bool = False,
        iteration_range: tuple[int, int] | None = None,
        base_margin: Any = None,
    ) -> np.ndarray:
        X, _ = _split_qid(X, None)
        return self._raw_predict(X, output_margin, iteration_range, base_margin)

    def apply(self, X: Any, iteration_range: tuple[int, int] | None = None) -> np.ndarray:
        X, _ = _split_qid(X, None)
        return super().apply(X, iteration_range)

    def _score_metric(self) -> str:
        """The metric `score` evaluates: the last `eval_metric`, else XGBoost's default."""
        if self.eval_metric is not None:
            names = [self.eval_metric] if isinstance(self.eval_metric, str) else list(self.eval_metric)
            return str(names[-1])
        objective = self.objective if self.objective is not None else self._default_objective()
        base = "map" if objective == "rank:map" else "ndcg"
        method = self.lambdarank_pair_method or "topk"
        if method == "mean":
            # whole-list ndcg; XGBoost's default `map` for `mean` is map@32
            return "ndcg@4294967295" if base == "ndcg" else "map"
        k = self.lambdarank_num_pair_per_sample or 32
        return f"{base}@{int(k)}"

    def score(self, X: Any, y: Any) -> float:
        """The ranking metric (last `eval_metric`, default ndcg@32) of the model on `X`.

        The query ids must be the `qid` column of the DataFrame `X`. With early stopping the
        model is evaluated up to `best_iteration`. Higher is better.
        """
        from . import _core

        check_is_fitted(self, "_Booster")
        X, qid = _split_qid(X, None)
        if qid is None:
            raise ValueError("The special column `qid` is required in `X` for ranking task.")
        scores = np.ascontiguousarray(self._raw_predict(X), dtype=np.float32)
        labels = np.ascontiguousarray(np.asarray(y, dtype=np.float32).ravel())
        from . import _as_query_array

        gain = True if self.ndcg_exp_gain is None else bool(self.ndcg_exp_gain)
        return float(
            _core.rank_score(
                self._score_metric(), scores, labels,
                _as_query_array(qid, "qid", labels.size), None, gain,
            )
        )

    def load_model(self, fname: str | os.PathLike) -> FerrisBoostRanker:
        """Read an XGBoost-compatible JSON (also an XGBoost `rank:*` model)."""
        booster = self._load_booster(fname)
        if not str(booster.objective).startswith("rank:"):
            raise ValueError(
                f"{booster.objective} is not a ranking model; use FerrisBoostRegressor / "
                "FerrisBoostClassifier.load_model"
            )
        self._Booster = booster
        self.n_features_in_ = booster.n_features
        self.evals_result_ = {}
        self.__dict__.pop("_cat_levels_", None)
        return self

    @classmethod
    def from_file(cls, fname: str | os.PathLike, **params: Any) -> FerrisBoostRanker:
        est = cls(**params)
        est.load_model(fname)
        return est
