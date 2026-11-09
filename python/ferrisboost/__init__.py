"""ferrisboost —— 列分块的梯度提升树,数值对齐 XGBoost。

参数名沿用 XGBoost 的写法,数据入口按自己的数据流设计:**没有 DMatrix**。
那个对象模型不可变、假设全量入内存,正是这个项目要摆脱的东西。

    import ferrisboost as fb
    import numpy as np

    model = fb.train({"max_depth": 6, "eta": 0.3, "objective": "binary:logistic",
                      "eval_metric": ["logloss", "auc"]},
                     X, label=y, weight=w, num_boost_round=100,
                     evals=[(Xv, yv, wv, "valid")], early_stopping_rounds=10)
    model.save_model("m.json")     # XGBoost 能加载
    fb.Model.load_model("m.json")  # 也能读 XGBoost 训的

这一层只做参数解析、dtype 归一化和输入分发,**不碰任何数值**。
计算全在 Rust 里 —— `python/tests/test_bit_exact.py` 就是守这条的:
同一份 fixture 从 Python 训出来,预测值要和基准逐位相同。
"""

from __future__ import annotations

import os
from typing import Any, Callable, Iterable, Mapping, Sequence

import numpy as np

from . import _core
from ._core import Model

__all__ = ["train", "Model", "__version__"]
__version__ = _core.__version__

_SKLEARN_NAMES = ("FerrisBoostClassifier", "FerrisBoostRegressor", "FerrisBoostRanker")


def __getattr__(name: str) -> Any:
    """scikit-learn 估计器按需导入:scikit-learn 是可选依赖,`import ferrisboost` 不需要它。"""
    if name in _SKLEARN_NAMES:
        from . import sklearn as _sk

        return getattr(_sk, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# 规范名 -> 默认值。**未知参数报错**,不像 XGBoost 那样静默忽略 ——
# 参数名打错了不给任何提示,是最难查的一类问题。
_DEFAULTS: dict[str, Any] = {
    "objective": "binary:logistic",
    "max_depth": 6,
    "max_bin": 255,
    "eta": 0.3,
    "lambda": 1.0,
    "gamma": 0.0,
    # L1 正则(alias reg_alpha)、叶子步长上限、正类梯度权重:语义同 XGBoost。
    "alpha": 0.0,
    # None = 树不限(0);count:poisson 的 Hessian 稳定项另用 0.7(XGBoost 默认)。
    "max_delta_step": None,
    "tweedie_variance_power": 1.5,
    "huber_slope": 1.0,
    # reg:quantileerror 必须给:[0, 1] 内的分位数,标量或升序列表
    # (多个值 = 多输出模型,每个分位数一棵树)。
    "quantile_alpha": None,
    # multi:softprob / multi:softmax 必须给;其它 objective 忽略。
    "num_class": None,
    "scale_pos_weight": 1.0,
    "min_child_weight": 1.0,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    # XGBoost 的嵌套采样(bytree → bylevel → bynode)。只限制 split 枚举。
    "colsample_bylevel": 1.0,
    "colsample_bynode": 1.0,
    # None = 自动:等拿到 schema 再按 nthread / max_bin 选块宽。
    # 显式给一个正整数就完全按用户的来,启发式不参与。
    "cols_per_block": None,
    "nthread": 0,
    # 0 = system-adaptive auto; independent of training nthread.
    "ingest_threads": 0,
    # 0 = auto for ordered tree conversion and positional model I/O;
    # independent of ingest/training. Single-document JSON serde stays serial.
    "model_io_threads": 0,
    # None = 用默认值 1。第二条 stream 要多一整套列块 buffer
    # (resident 280 MB / streaming 305 MB),而实测 resident 上单流反而更快、
    # streaming 上双流只快 0.1–2.5%(不稳定)。想要双流显式传 2。
    "hist_streams": None,
    # 常驻物理列块数。None = auto(按显存预算解)。
    # 它改变内存计划:每块多占一整套列块 buffer(wide 10M×300、c=64 实测
    # 614 MB/块),换来的是消掉「每层重新上传同一个物理块」的重复流量。
    "resident_blocks": None,
    "hist_nodes_per_batch": None,
    "gpu_math": None,
    # "cpu"(默认)/ "cuda" / "cuda:N"。
    #
    # ⚠️ Python 只说**用哪块卡**;GPU source 的构造和物理块宽由 Rust 侧的
    # 显存规划器决定,Python 不碰这两件事。
    "device": None,
    # GPU streaming 可用的显存上限(字节)。语义是"最多用到大约这么多",
    # 不是"就分配这么多"。
    "gpu_memory_budget": None,
    # None = 默认开。它改变内存计划(+8 B/行 device),换来 host 扫描消失、
    # gpair H2D 减半;实测 HIGGS −27%、wide −5.5%。显存吃紧可以关掉。
    "device_quantize": None,
    "seed": 0,
    # None = 和 XGBoost 2.0+ 一样从训练标签估计(加权标签均值);给值就原样用。
    "base_score": None,
    "cache_budget_bytes": None,
    "cache_path": None,
    # XGBoost 同名参数:字符串或字符串列表。None = objective 的默认指标。
    # 多个时 early stopping 盯最后一个(和 XGBoost 一致)。
    "eval_metric": None,
    # XGBoost 同名参数:元组/列表(按特征位置)、"(1,0,-1)" 字符串,或
    # {特征名: 方向} 字典(需要列名)。1 递增、-1 递减、0 不约束;只看符号。
    "monotone_constraints": None,
    # Categorical features (XGBoost semantics). `enable_categorical` lets pandas
    # `category` columns and PyArrow dictionary columns be detected as
    # categorical; `feature_types` ("c" / "q" per feature) declares them
    # explicitly for numpy and file inputs. Categories are integer codes in
    # 0..=65533. `max_cat_to_onehot`: features with fewer categories than this are
    # split one-vs-rest, otherwise by sorted partition; `max_cat_threshold`
    # bounds the partition scan.
    "enable_categorical": False,
    # XGBoost `missing`: a feature value equal to it is treated exactly like NaN
    # (missing direction). NaN (default) = no extra sentinel. Applies to the
    # training data and every validation set, for arrays, DataFrames, tables
    # and files. Not stored in the model: pass the same `missing` to predict.
    "missing": float("nan"),
    "max_cat_to_onehot": 4,
    "max_cat_threshold": 64,
    # Learning to rank (rank:ndcg / rank:pairwise / rank:map), XGBoost parameter
    # names and defaults. `lambdarank_num_pair_per_sample` None = XGBoost's
    # default (32 for "topk", 1 for "mean").
    "lambdarank_pair_method": "topk",
    "lambdarank_num_pair_per_sample": None,
    "lambdarank_normalization": True,
    "lambdarank_score_normalization": True,
    "lambdarank_unbiased": False,
    "lambdarank_bias_norm": 1.0,
    "ndcg_exp_gain": True,
    # Private wire field set by `train` from `feature_types` / the frame dtypes:
    # one bool per feature, or None.
    "_feature_types": None,
    # Private wire field: category names of named categorical columns (pandas
    # categories, Arrow dictionaries) the data was coded with, one entry per
    # feature (None or (kind, values)); stored in the model.
    "_categories": None,
}

# XGBoost 的别名,两种写法都收
_ALIASES = {
    "learning_rate": "eta",
    "reg_lambda": "lambda",
    "reg_alpha": "alpha",
    "n_jobs": "nthread",
    "random_state": "seed",
    "max_leaves": "max_leaves",
}

_OBJECTIVES = ("binary:logistic", "reg:squarederror", "count:poisson", "reg:gamma",
               "reg:tweedie", "reg:pseudohubererror", "reg:quantileerror", "multi:softprob",
               "multi:softmax", "rank:ndcg", "rank:pairwise", "rank:map")

# 认识但还没实现的:显式报错,别让用户以为设了就生效了
_UNIMPLEMENTED = {
    "max_leaves": "leaf-wise growth is not implemented yet (trees grow depthwise)",
    "tree_method": "hist is the only tree method; do not set it",
    "num_parallel_tree": "random forest mode is not supported",
}


def _normalize_params(params: Mapping[str, Any], num_boost_round: int) -> dict[str, Any]:
    out = dict(_DEFAULTS)
    for key, value in dict(params).items():
        canon = _ALIASES.get(key, key)
        if canon in _UNIMPLEMENTED:
            raise ValueError(f"parameter {key!r}: {_UNIMPLEMENTED[canon]}")
        if canon not in _DEFAULTS:
            known = ", ".join(sorted(_DEFAULTS) + sorted(_ALIASES))
            raise ValueError(f"unknown parameter {key!r}; supported parameters: {known}")
        out[canon] = value

    if out["objective"] not in _OBJECTIVES:
        raise ValueError(
            f"objective {out['objective']!r} is not supported; supported objectives: {', '.join(_OBJECTIVES)}"
        )
    if not 0 < out["max_bin"] <= 65535:
        # Features with more than 255 bins are stored as 16-bit columns; the
        # largest 16-bit code is the missing sentinel.
        raise ValueError(f"max_bin must be in 1..=65535, got {out['max_bin']}")
    if num_boost_round < 1:
        raise ValueError(f"num_boost_round must be >= 1, got {num_boost_round}")

    out["num_boost_round"] = int(num_boost_round)
    for key in (
        "max_depth", "max_bin", "nthread", "ingest_threads", "model_io_threads",
        "seed", "num_boost_round"
    ):
        out[key] = int(out[key])
    if out["ingest_threads"] < 0:
        raise ValueError(f"ingest_threads must be >= 0, got {out['ingest_threads']}")
    if out["model_io_threads"] < 0:
        raise ValueError(f"model_io_threads must be >= 0, got {out['model_io_threads']}")
    # cols_per_block 保留 None 语义,不要被 int() 变成 0 之后和「用户真的
    # 传了 0」分不开。
    if out["cols_per_block"] is not None:
        out["cols_per_block"] = int(out["cols_per_block"])
        if out["cols_per_block"] < 1:
            raise ValueError(
                f"cols_per_block must be >= 1, got {out['cols_per_block']};"
                " omit it to let it be chosen automatically"
            )
    # 和 cols_per_block 同样的 None 语义:显式给了就完全按用户的来。
    if out["hist_streams"] is not None:
        out["hist_streams"] = int(out["hist_streams"])
        if out["hist_streams"] not in (1, 2):
            raise ValueError(
                f"hist_streams must be 1 or 2, got {out['hist_streams']};"
                " omit it to let it be chosen automatically"
            )
    # 同样的 None 语义:auto 让规划器按剩余显存解,显式给了就按用户的来
    # (仍会被夹到物理块数 —— 比块还多的常驻 slot 只会白占显存)。
    if out["resident_blocks"] is not None:
        out["resident_blocks"] = int(out["resident_blocks"])
        if out["resident_blocks"] < 0:
            raise ValueError(
                f"resident_blocks must not be negative, got {out['resident_blocks']};"
                " omit it to let it be chosen automatically; 0 means pure streaming"
            )
    if out["hist_nodes_per_batch"] is not None:
        out["hist_nodes_per_batch"] = int(out["hist_nodes_per_batch"])
        if not 1 <= out["hist_nodes_per_batch"] <= 32:
            raise ValueError(
                f"hist_nodes_per_batch must be in 1..=32, got {out['hist_nodes_per_batch']};"
                " 1 disables multi-node batching"
            )
    if out["device"] is not None:
        d = str(out["device"])
        if d not in ("cpu", "cuda", "gpu") and not d.startswith("cuda:"):
            raise ValueError(
                f'device must be "cpu", "cuda" or "cuda:N", got {d!r}'
            )
        out["device"] = d
    # max_depth 的上限由 GPU 每层节点 buffer 决定(见 Rust 的
    # `types::MAX_DEPTH_LIMIT`)。CPU 自己能跑更深,但契约取两个后端的共同
    # 安全值 —— 否则 CPU 训出来的模型在 GPU 上复现不了。
    # ⚠️ 下界是 0(只有根节点的树,合法),不是 1。
    if not 0 <= int(out["max_depth"]) <= 14:
        raise ValueError(
            f"max_depth must be in 0..=14, got {out['max_depth']};"
            " 0 means a root-only tree; the limit comes from the GPU per-level partition node buffer"
            " (depth 14 fills all 8192 slots)"
        )
    if out["gpu_memory_budget"] is not None:
        out["gpu_memory_budget"] = int(out["gpu_memory_budget"])
        if out["gpu_memory_budget"] <= 0:
            raise ValueError("gpu_memory_budget must be a positive integer number of bytes")
    if out["gpu_math"] is not None:
        out["gpu_math"] = str(out["gpu_math"])
        if out["gpu_math"] not in ("exact", "fast"):
            raise ValueError(
                f"gpu_math must be \"exact\" or \"fast\", got {out['gpu_math']!r};"
                " fast does not guarantee byte-identical CPU/GPU models"
            )
    if out["device_quantize"] is not None:
        out["device_quantize"] = bool(out["device_quantize"])
    if out["num_class"] is not None:
        if isinstance(out["num_class"], bool) or int(out["num_class"]) != out["num_class"]:
            raise ValueError(f"num_class must be an integer, got {out['num_class']!r}")
        out["num_class"] = int(out["num_class"])
        if out["num_class"] < 0:
            raise ValueError("num_class must not be negative")
    if out["max_delta_step"] is not None:
        out["max_delta_step"] = float(out["max_delta_step"])
    out["quantile_alpha"] = _normalize_quantile_alpha(out["quantile_alpha"], out["objective"])
    for key in ("eta", "lambda", "gamma", "min_child_weight", "alpha", "tweedie_variance_power",
                "huber_slope", "scale_pos_weight", "subsample",
                "colsample_bytree", "colsample_bylevel", "colsample_bynode"):
        out[key] = float(out[key])
    if out["base_score"] is not None:
        out["base_score"] = float(out["base_score"])
    if out["cache_budget_bytes"] is not None:
        out["cache_budget_bytes"] = int(out["cache_budget_bytes"])
        if out["cache_budget_bytes"] < 0:
            raise ValueError("cache_budget_bytes must not be negative")
    if out["cache_path"] is not None:
        out["cache_path"] = os.fspath(out["cache_path"])
    if out["eval_metric"] is not None:
        m = out["eval_metric"]
        names = [m] if isinstance(m, str) else (
            list(m) if isinstance(m, (list, tuple)) else None
        )
        if not names or not all(isinstance(n, str) for n in names):
            raise ValueError(
                f"eval_metric must be a string or a list of strings, got {m!r}"
            )
        # 名字和 objective 的组合由 Rust 校验(唯一一份规则)。
        out["eval_metric"] = names
    out["monotone_constraints"] = _normalize_monotone(out["monotone_constraints"])
    _normalize_lambdarank(out)
    out["enable_categorical"] = bool(out["enable_categorical"])
    out["missing"] = _normalize_missing(out["missing"])
    for key in ("max_cat_to_onehot", "max_cat_threshold"):
        if isinstance(out[key], bool) or int(out[key]) != out[key]:
            raise ValueError(f"{key} must be an integer, got {out[key]!r}")
        out[key] = int(out[key])
        if out[key] < 1:
            raise ValueError(f"{key} must be >= 1, got {out[key]}")
    return out


def _normalize_lambdarank(out: dict[str, Any]) -> None:
    """Validate the XGBoost `lambdarank_*` / `ndcg_exp_gain` parameters in place."""
    if out["lambdarank_pair_method"] not in ("topk", "mean"):
        raise ValueError(
            "lambdarank_pair_method must be \"topk\" or \"mean\", "
            f"got {out['lambdarank_pair_method']!r}"
        )
    k = out["lambdarank_num_pair_per_sample"]
    if k is not None:
        if isinstance(k, bool) or int(k) != k or int(k) < 1:
            raise ValueError(
                f"lambdarank_num_pair_per_sample must be an integer >= 1, got {k!r}"
            )
        out["lambdarank_num_pair_per_sample"] = int(k)
    for key in ("lambdarank_normalization", "lambdarank_score_normalization",
                "lambdarank_unbiased", "ndcg_exp_gain"):
        out[key] = bool(out[key])
    out["lambdarank_bias_norm"] = float(out["lambdarank_bias_norm"])
    if not out["lambdarank_bias_norm"] >= 0.0:
        raise ValueError("lambdarank_bias_norm must be >= 0")


def _normalize_missing(value: Any) -> float:
    """`missing` -> float. NaN means "no sentinel"; ±inf and f32::MAX are rejected.

    The full rule (finite float32 below the reserved f32::MAX) is enforced once,
    in Rust; this only turns the value into a float and fails early on junk.
    """
    if value is None or isinstance(value, (str, bytes)):
        raise ValueError(f"missing must be a number (default NaN), got {value!r}")
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"missing must be a number (default NaN), got {value!r}") from None


def _normalize_quantile_alpha(value: Any, objective: str) -> list[float] | None:
    """`quantile_alpha` -> list[float] | None. A scalar is a one-element list; several
    values train one model with one output (and one tree per round) per quantile."""
    if value is None:
        if objective == "reg:quantileerror":
            raise ValueError(
                "reg:quantileerror needs quantile_alpha (one or more ascending quantiles in [0, 1], "
                "e.g. 0.9 or [0.1, 0.5, 0.9])"
            )
        return None
    if isinstance(value, (list, tuple, np.ndarray)):
        flat = np.asarray(value, dtype=np.float64).ravel().tolist()
    else:
        flat = [float(value)]
    if not flat:
        raise ValueError("quantile_alpha must not be an empty list")
    for alpha in flat:
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"quantile_alpha must be in [0, 1], got {alpha}")
    if any(a > b for a, b in zip(flat, flat[1:])):
        raise ValueError(f"quantile_alpha must be in ascending order (as XGBoost requires), got {flat}")
    return flat


def _monotone_sign(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"monotone_constraints {where} must be an integer (1 / 0 / -1), got {value!r}")
    # XGBoost 只看符号:2 等价于 1,-3 等价于 -1。
    return int(np.sign(int(value)))


def _normalize_monotone(mc: Any) -> list[int] | dict[str, int] | None:
    """归一成按位置的整数列表,或 {列名: 方向}。全 0 / 空等价于不约束。"""
    if mc is None:
        return None
    if isinstance(mc, str):
        text = mc.strip()
        if not (text.startswith("(") and text.endswith(")")):
            raise ValueError(f'a monotone_constraints string must look like "(1,0,-1)", got {mc!r}')
        body = text[1:-1].strip()
        try:
            values = [int(v) for v in body.split(",")] if body else []
        except ValueError:
            raise ValueError(f"monotone_constraints string contains a non-integer: {mc!r}") from None
        return [_monotone_sign(v, "element") for v in values]
    if isinstance(mc, Mapping):
        out = {}
        for name, value in mc.items():
            if not isinstance(name, str):
                raise ValueError(f"monotone_constraints dict keys must be column name strings, got {name!r}")
            out[name] = _monotone_sign(value, f"value for column {name!r}")
        return out
    if isinstance(mc, (list, tuple, np.ndarray)):
        return [_monotone_sign(v, "element") for v in np.asarray(mc, dtype=object).ravel()]
    raise ValueError(
        "monotone_constraints must be a tuple/list, a \"(1,0,-1)\" string or a {column name: direction} dict, "
        f"got {type(mc).__name__}"
    )


def _resolve_named_monotone(mc: dict[str, int], names: list[str]) -> list[int]:
    unknown = [k for k in mc if k not in names]
    if unknown:
        raise ValueError(f"monotone_constraints columns {unknown} are not feature columns; feature columns: {names}")
    return [mc.get(n, 0) for n in names]


def _as_matrix(x: Any, name: str = "data") -> np.ndarray:
    """转成 C 连续的 f32 二维数组。

    **显式转换**,不是隐式截断:f64 进来会掉精度,但那是分箱之前的事,
    分箱本身只保留 max_bin 个边界,所以这一步的损失在训练里看不见。
    真正要紧的是布局 —— 非连续的数组必须先拷贝,否则底下按行主序取
    数会取错。
    """
    arr = np.asarray(x)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array, got {arr.ndim} dimensions")
    if arr.dtype == np.float32 and arr.flags["C_CONTIGUOUS"]:
        return arr
    return np.ascontiguousarray(arr, dtype=np.float32)


def _as_labels(y: Any, n_rows: int) -> np.ndarray:
    arr = np.ascontiguousarray(np.asarray(y).ravel(), dtype=np.float32)
    if arr.size != n_rows:
        raise ValueError(f"label has {arr.size} entries, the data has {n_rows} rows")
    if np.isnan(arr).any():
        raise ValueError("label contains NaN; handle missing labels upstream "
                         "(filling them with 0 would train on 0 as a real label)")
    return arr


def _as_weights(w: Any, n_rows: int | None, name: str = "weight") -> np.ndarray:
    """样本权重 -> C 连续 f32 一维数组。规则和 XGBoost 一致:有限、非负。

    `n_rows=None` skips the length check: ranking weights are one per query
    group, and the group count is checked against them in Rust.
    """
    arr = np.ascontiguousarray(np.asarray(w, dtype=np.float32).ravel())
    if n_rows is not None and arr.size != n_rows:
        raise ValueError(f"{name} has {arr.size} entries, the data has {n_rows} rows")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or inf; sample weights must be finite")
    if (arr < 0).any():
        raise ValueError(f"{name} contains negative values; sample weights must be non-negative")
    return arr


def _as_query_array(v: Any, what: str, n_rows: int | None = None) -> np.ndarray:
    """`qid` / `group` -> contiguous uint64 1-D array (integers >= 0, no NaN)."""
    arr = np.asarray(v)
    if arr.ndim != 1:
        arr = arr.ravel()
    if arr.dtype.kind == "f":
        if not np.isfinite(arr).all() or (arr != np.floor(arr)).any():
            raise ValueError(f"{what} must hold integers (no NaN / fractions)")
    elif arr.dtype.kind not in "iub":
        raise ValueError(f"{what} must hold integers, got dtype {arr.dtype}")
    if (arr < 0).any():
        raise ValueError(f"{what} must be non-negative")
    if n_rows is not None and arr.size != n_rows:
        raise ValueError(f"{what} has {arr.size} entries, the data has {n_rows} rows")
    return np.ascontiguousarray(arr, dtype=np.uint64)


def _as_path_specs(data: Any) -> list[str] | None:
    """把公开的输入写法归一成一串路径规格,不是路径就返回 None。

    支持:单个路径、路径序列。目录和 glob 只是**字符串**,展开发生在 Rust ——
    Python 不碰文件系统,否则「Python 是控制面」这条就只是口号。
    """
    if isinstance(data, (str, os.PathLike)):
        return [os.fspath(data)]
    if isinstance(data, (list, tuple)) and data and all(
        isinstance(d, (str, os.PathLike)) for d in data
    ):
        return [os.fspath(d) for d in data]
    return None


def _file_format(path: str) -> str:
    lower = path.lower()
    if lower.endswith(".parquet") or lower.endswith(".pq"):
        return "parquet"
    if lower.endswith(".csv"):
        return "csv"
    raise ValueError(f"cannot tell the format of {path!r} from its extension; supported: .parquet / .csv")


def _arrow_table(obj: Any) -> Any:
    """pyarrow Table / RecordBatch 原样返回,RecordBatchReader 读成 Table;不是 pyarrow 返回 None。"""
    mod = type(obj).__module__ or ""
    if not mod.startswith("pyarrow"):
        return None
    if hasattr(obj, "column_names"):
        if type(obj).__name__ == "RecordBatch":
            import pyarrow as pa

            return pa.Table.from_batches([obj])
        return obj
    if hasattr(obj, "read_all"):
        return obj.read_all()
    return None


def _feature_flags(feature_types: Any, n_features: int | None) -> list[bool] | None:
    """`feature_types` ("c" / "q" per feature, XGBoost style) -> list of bools.

    `None` = nothing declared. Anything other than "c" is numeric ("q", "float",
    "int", "i"). The length is checked against `n_features` when it is known.
    """
    if feature_types is None:
        return None
    if isinstance(feature_types, str) or not isinstance(feature_types, (list, tuple, np.ndarray)):
        raise ValueError(
            f'feature_types must be a list of "c" / "q" (one per feature), got {feature_types!r}'
        )
    flags = []
    for t in feature_types:
        if not isinstance(t, str):
            raise ValueError(f'feature_types elements must be strings ("c" / "q"), got {t!r}')
        flags.append(t == "c")
    if n_features is not None and len(flags) != n_features:
        raise ValueError(f"feature_types has {len(flags)} entries, the data has {n_features} features")
    return flags


def _is_pandas_frame(obj: Any) -> bool:
    return (type(obj).__module__ or "").startswith("pandas") and hasattr(obj, "dtypes") \
        and hasattr(obj, "columns")


def _codes_to_f32(codes: np.ndarray, missing: np.ndarray) -> np.ndarray:
    out = codes.astype(np.float32)
    out[missing] = np.nan
    return out


# XGBoost array-interface type codes of integer categories (XGBoost 3.x `cats`).
_INT_TYPE_CODES = {"int8": 9, "uint8": 10, "int16": 11, "uint16": 12,
                   "int32": 13, "uint32": 14, "int64": 15, "uint64": 16}
_I64_MAX = np.iinfo(np.int64).max


def _check_int_categories(values: list, where: str) -> list:
    if any(v > _I64_MAX for v in values):
        raise ValueError(f"{where}: integer categories must fit the signed 64-bit range (as in XGBoost)")
    return values


def _pandas_category_values(categories: Any, where: str) -> tuple:
    """pandas categories -> (kind, values): kind is "str" or an XGBoost integer
    type code. Other category types are rejected, as in XGBoost."""
    dtype = categories.dtype
    if getattr(dtype, "kind", "") in "iu" and getattr(dtype, "kind", "") != "":
        np_dtype = np.dtype(getattr(dtype, "numpy_dtype", dtype))
        return _INT_TYPE_CODES[str(np_dtype)], _check_int_categories(
            [int(v) for v in categories], where)
    values = list(categories)
    if all(isinstance(v, str) for v in values):
        return "str", [str(v) for v in values]
    raise ValueError(
        f"{where}: categories must be strings or integers (as in XGBoost), got {dtype} categories; "
        "convert them, e.g. with .cat.rename_categories(str)"
    )


def _arrow_category_values(dictionary: Any, where: str) -> tuple:
    """A pyarrow dictionary -> (kind, values), like `_pandas_category_values`."""
    import pyarrow as pa

    t = dictionary.type
    if pa.types.is_string(t) or pa.types.is_large_string(t) or (
        hasattr(pa.types, "is_string_view") and pa.types.is_string_view(t)
    ):
        values = dictionary.to_pylist()
        if any(v is None for v in values):
            raise ValueError(f"{where}: a dictionary entry is null; categories must be values")
        return "str", values
    if pa.types.is_integer(t):
        values = dictionary.to_pylist()
        if any(v is None for v in values):
            raise ValueError(f"{where}: a dictionary entry is null; categories must be values")
        return _INT_TYPE_CODES[str(t)], _check_int_categories(values, where)
    raise ValueError(
        f"{where}: dictionary values must be strings or integers to be categories (as in XGBoost), "
        f"got {t}"
    )


def _as_ref(level: Any) -> tuple | None:
    """A category reference -> (kind, values). Accepts (kind, values) pairs,
    plain lists (a model's `get_categories()` entries) and, for estimators
    pickled by older versions, a pandas Index / pyarrow array."""
    if level is None:
        return None
    if isinstance(level, tuple) and len(level) == 2:
        return level
    if hasattr(level, "dtype") and hasattr(level, "tolist") and not isinstance(level, list):
        try:
            return _pandas_category_values(level, "categories")
        except (TypeError, AttributeError):
            pass
    if hasattr(level, "to_pylist"):
        return _arrow_category_values(level, "categories")
    values = list(level)
    if all(isinstance(v, str) for v in values):
        return "str", values
    return 15, [int(v) for v in values]


def _recode_map(ref: tuple, data_kind: Any, data_values: list, where: str) -> np.ndarray:
    """Index of every data category in the reference list (-1 = unseen)."""
    import pandas as pd

    ref_kind, ref_values = ref
    if (ref_kind == "str") != (data_kind == "str"):
        theirs = "strings" if data_kind == "str" else "integers"
        ours = "strings" if ref_kind == "str" else "integers"
        raise ValueError(
            f"{where}: its categories are {theirs}, the training categories are {ours}; "
            "the category data type must match the training data"
        )
    if not data_values:
        return np.empty(0, np.int64)
    return np.asarray(pd.Index(ref_values).get_indexer(pd.Index(data_values)), np.int64)


def _apply_map(mapping: np.ndarray, codes: np.ndarray, data_values: list, where: str) -> np.ndarray:
    """Data codes (-1 = missing) -> reference codes; a row holding a category
    outside the reference list is an error (XGBoost rejects it too)."""
    out = np.full(codes.shape, -1, np.int64)
    present = codes >= 0
    out[present] = mapping[codes[present]]
    unseen = present & (out < 0)
    if unseen.any():
        value = data_values[int(codes[np.argmax(unseen)])]
        raise ValueError(
            f"{where} holds the category {value!r}, which is not in the training categories "
            "(categories unseen in training are rejected, as in XGBoost)"
        )
    return out


def _frame_matrix(
    df: Any, enable_categorical: bool, ref_levels: dict | None = None, what: str = "data"
) -> tuple[np.ndarray, list[str], list[bool], dict]:
    """pandas DataFrame -> (f32 matrix, column names, categorical flags, levels).

    A `category` column becomes its integer codes (NaN for a missing value) and
    is flagged categorical; `levels[i]` is its category list as `(kind, values)`
    (kind "str" or an XGBoost integer type code). `ref_levels` (category lists
    by column position, e.g. the training data's or a model's) re-codes those
    columns by value onto the reference codes; a category outside the reference
    list is an error, as in XGBoost.
    """
    import pandas as pd

    names = [str(c) for c in df.columns]
    if len(set(names)) != len(names):
        dups = sorted({n for n in names if names.count(n) > 1})
        raise ValueError(f"{what} has duplicate column names {dups}; column names must be unique")
    cols, flags, levels = [], [], {}
    for i, name in enumerate(names):
        s = df.iloc[:, i]
        if isinstance(s.dtype, pd.CategoricalDtype):
            if not enable_categorical:
                raise ValueError(
                    f"{what} column {name!r} has category dtype; to train it as a categorical feature, "
                    "set params['enable_categorical'] = True"
                )
            where = f"{what} column {name!r}"
            codes = s.cat.codes.to_numpy().astype(np.int64)
            ref = _as_ref(ref_levels.get(i)) if ref_levels else None
            if ref is not None:
                kind, values = _pandas_category_values(s.cat.categories, where)
                codes = _apply_map(_recode_map(ref, kind, values, where), codes, values, where)
                levels[i] = ref
            else:
                levels[i] = _pandas_category_values(s.cat.categories, where)
            cols.append(_codes_to_f32(codes, codes < 0))
            flags.append(True)
        else:
            try:
                cols.append(s.to_numpy(dtype=np.float32, na_value=np.nan))
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"{what} column {name!r} has dtype {s.dtype}, which is not a numeric feature; "
                    "convert it to numeric, or to category and set enable_categorical=True"
                ) from e
            flags.append(False)
    x = np.ascontiguousarray(np.stack(cols, axis=1)) if cols else np.empty((len(df), 0), np.float32)
    return x, names, flags, levels


def _table_matrix(
    table: Any,
    order: list[str] | None,
    enable_categorical: bool,
    ref_levels: dict | None = None,
    what: str = "data",
    scoring: bool = False,
) -> tuple[np.ndarray, list[str], list[bool], dict]:
    """pyarrow Table -> (f32 matrix, names, categorical flags, levels).

    A dictionary column is categorical: its codes are the dictionary indices
    (dictionaries are unified across chunks first) and `levels[i]` is the
    dictionary as `(kind, values)`. `order` binds columns by name to a training
    table; `ref_levels` re-codes dictionary columns by value the way
    `_frame_matrix` does.

    `scoring` (prediction and validation data) fixes the categories: a column
    without a reference list keeps them as the model has them, so an integer
    dictionary yields its values (the codes, as a Parquet file's dictionary
    does) and a string dictionary its indices. Missing entries become NaN.
    """
    import pyarrow as pa

    names = list(table.column_names)
    dups = sorted({n for n in names if names.count(n) > 1})
    if dups:
        raise ValueError(f"{what} has duplicate column names {dups}; column names must be unique")
    if order is not None:
        missing = [n for n in order if n not in names]
        extra = [n for n in names if n not in order]
        if missing or extra:
            raise ValueError(
                f"{what} columns do not match the training data (bound by name): missing {missing}, extra {extra}"
            )
        names = order
    if any(pa.types.is_dictionary(table.schema.field(n).type) for n in names):
        table = table.unify_dictionaries()
    cols, flags, levels = [], [], {}
    for i, name in enumerate(names):
        col = table.column(name)
        ref = _as_ref(ref_levels.get(i)) if ref_levels else None
        if (pa.types.is_dictionary(col.type) and ref is None and scoring
                and pa.types.is_integer(col.type.value_type)):
            # No list to re-code onto: the dictionary values are the codes,
            # read exactly as from a Parquet file (int -> f64 -> f32, a null
            # index or entry is missing). Range checks are the prediction's.
            values = pa.chunked_array(
                [c.dictionary_decode() for c in col.chunks], col.type.value_type
            ).cast(pa.float64())
            cols.append(np.asarray(values.to_numpy(zero_copy_only=False), np.float64)
                        .astype(np.float32))
            flags.append(True)
            continue
        if pa.types.is_dictionary(col.type):
            if not enable_categorical:
                raise ValueError(
                    f"{what} column {name!r} has dictionary type; to train it as a categorical feature, "
                    "set params['enable_categorical'] = True"
                )
            where = f"{what} column {name!r}"
            chunks = col.chunks
            dictionary = chunks[0].dictionary if chunks else pa.array([], col.type.value_type)
            # Dictionary indices may be any integer width (pandas categories
            # convert to int8 indices); nulls become -1.
            codes = np.concatenate(
                [np.asarray(c.indices.cast(pa.int64()).fill_null(-1).to_numpy(zero_copy_only=False),
                            np.int64) for c in chunks]
            ) if chunks else np.empty(0, np.int64)
            if ref is not None:
                kind, values = _arrow_category_values(dictionary, where)
                codes = _apply_map(_recode_map(ref, kind, values, where), codes, values, where)
                levels[i] = ref
            else:
                levels[i] = _arrow_category_values(dictionary, where)
            cols.append(_codes_to_f32(codes, codes < 0))
            flags.append(True)
        else:
            cols.append(np.asarray(col.to_numpy(zero_copy_only=False), dtype=np.float32))
            flags.append(False)
    x = np.ascontiguousarray(np.stack(cols, axis=1)) if cols else np.empty((table.num_rows, 0), np.float32)
    return x, names, flags, levels


def _categories_wire(levels: dict, flags: list[bool] | None, n: int) -> list | None:
    """Category lists of the final categorical features -> the private
    `_categories` parameter (None when no categorical feature has names)."""
    if not levels or not flags:
        return None
    wire = [levels[i] if flags[i] and i in levels else None for i in range(n)]
    return wire if any(w is not None for w in wire) else None


def _model_refs(model: Any) -> dict:
    """A model's category lists by feature position (for re-coding)."""
    cats = model.get_categories() if model is not None else None
    if not cats:
        return {}
    return {i: _as_ref(c) for i, c in enumerate(cats) if c is not None}


def _merge_flags(auto: list[bool] | None, explicit: list[bool] | None, n: int) -> list[bool] | None:
    """Explicit `feature_types` win over dtype detection; None if nothing is categorical."""
    flags = explicit if explicit is not None else auto
    if flags is None or not any(flags):
        return None
    if len(flags) != n:
        raise ValueError(f"feature_types has {len(flags)} entries, the data has {n} features")
    return list(flags)


def _is_named_table(data: Any) -> bool:
    """True for a pandas DataFrame or a PyArrow Table / RecordBatch."""
    return _is_pandas_frame(data) or _arrow_table(data) is not None


def _as_predict_matrix(
    data: Any, names: Sequence[str] | None = None, categories: Sequence[Any] | None = None
) -> np.ndarray:
    """Convert a pandas DataFrame / PyArrow Table to the f32 matrix `Model.predict` scores.

    With `names` (the model's feature names), columns are selected and ordered
    by name, as XGBoost does for a DataFrame; extra columns are ignored and a
    missing one is an error. Without names the columns are taken in order.
    `categories` (the model's `get_categories()`) re-codes categorical columns
    by value onto the model's codes, so their categories may be listed in any
    order; a category the model does not know is an error. A categorical
    feature without stored names uses the column's codes as they are.
    """
    refs = {i: _as_ref(c) for i, c in enumerate(categories or []) if c is not None}
    if _is_pandas_frame(data):
        if names:
            cols = [str(c) for c in data.columns]
            absent = [n for n in names if n not in cols]
            if absent:
                raise ValueError(f"predict data is missing the model's feature columns {absent}")
            data = data[[c for n in names for c in data.columns if str(c) == n]]
        return _frame_matrix(data, True, refs, what="predict data")[0]
    table = _arrow_table(data)
    if table is not None:
        if names:
            absent = [n for n in names if n not in table.column_names]
            if absent:
                raise ValueError(f"predict data is missing the model's feature columns {absent}")
            table = table.select(list(names))
        return _table_matrix(table, list(names) if names else None, True, refs,
                             what="predict data", scoring=True)[0]
    raise ValueError(
        "predict accepts file paths (single / list / directory / glob), a 2-D f32 numpy array, "
        "a pandas DataFrame or a pyarrow Table"
    )


def _num_outputs(p: Mapping[str, Any]) -> int:
    """Outputs per row: `num_class` for the multiclass objectives, one per
    `quantile_alpha` for `reg:quantileerror`, else 1."""
    if str(p["objective"]).startswith("multi:"):
        return int(p["num_class"] or 1)
    if p["objective"] == "reg:quantileerror" and p.get("quantile_alpha"):
        return len(p["quantile_alpha"])
    return 1


def _as_base_margin(margin: Any, k: int, what: str = "base_margin") -> np.ndarray:
    """`base_margin` -> C-contiguous float32 `(rows, k)` matrix.

    XGBoost shapes: `(n,)` (or `(n, 1)`) for one output, `(n, num_class)` for
    multiclass. Lists, pandas Series / DataFrames and pyarrow arrays are
    accepted; every value must be finite.
    """
    if hasattr(margin, "to_numpy") and not isinstance(margin, np.ndarray):
        margin = margin.to_numpy()
    elif hasattr(margin, "to_pandas") and not isinstance(margin, np.ndarray):
        margin = margin.to_pandas().to_numpy()
    arr = np.asarray(margin, dtype=np.float32)
    if arr.ndim == 1 and k == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2 or arr.shape[1] != k:
        want = "(n,) or (n, 1)" if k == 1 else f"(n, {k})"
        raise ValueError(f"{what} must have shape {want}, got {tuple(arr.shape)}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{what} contains NaN or inf; every starting margin must be finite")
    return np.ascontiguousarray(arr)


def _pick_init_model(init_model: Any, xgb_model: Any) -> Any:
    """`init_model`, or `xgb_model` (XGBoost's name for it); not both."""
    if init_model is not None and xgb_model is not None:
        raise TypeError("pass init_model or xgb_model (its XGBoost name), not both")
    return xgb_model if init_model is None else init_model


def _load_prior(model: Any) -> Model | None:
    """`init_model` -> a Model: a FerrisBoost Model, a path to FerrisBoost or
    XGBoost JSON, or an object with `save_raw` (an `xgboost.Booster`)."""
    if model is None:
        return None
    if isinstance(model, Model):
        return model
    if isinstance(model, (str, os.PathLike)):
        return Model.load_model(os.fspath(model), quiet=True)
    if hasattr(model, "save_raw"):
        import tempfile

        raw = bytes(model.save_raw(raw_format="json"))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "model.json")
            with open(path, "wb") as f:
                f.write(raw)
            return Model.load_model(path, quiet=True)
    raise TypeError(
        "init_model must be a ferrisboost.Model, a path to a model JSON "
        f"(FerrisBoost or XGBoost) or an xgboost.Booster, got {type(model).__name__}"
    )


def _adopt_prior_params(params: Mapping[str, Any], prior: Model) -> dict[str, Any]:
    """Continuation keeps the model's objective: without an explicit one in
    `params` the prior's `objective` / `num_class` are used."""
    out = dict(params)
    if "objective" not in out:
        out["objective"] = prior.objective
        if prior.num_class > 1 and "num_class" not in out:
            out["num_class"] = prior.num_class
    # A quantile model keeps its quantiles: they are part of the objective.
    if (
        out.get("objective") == "reg:quantileerror"
        and out.get("quantile_alpha") is None
        and prior.quantile_alpha is not None
    ):
        out["quantile_alpha"] = list(prior.quantile_alpha)
    return out


def _check_prior(prior: Model, p: Mapping[str, Any], n_features: int | None) -> None:
    """Fail early (before scoring the prior on the new data) when the model to
    continue cannot be continued with these parameters / this data."""
    if prior.objective != p["objective"]:
        raise ValueError(
            f"init_model was trained with objective {prior.objective!r}, "
            f"params ask for {p['objective']!r}"
        )
    if prior.quantile_alpha is not None or p["objective"] == "reg:quantileerror":
        ours = np.asarray(p.get("quantile_alpha") or [], dtype=np.float32)
        theirs = np.asarray(prior.quantile_alpha or [], dtype=np.float32)
        if ours.shape != theirs.shape or not np.array_equal(ours, theirs):
            raise ValueError(
                f"init_model was trained with quantile_alpha={theirs.tolist()}, params give "
                f"{ours.tolist()}: continuing needs the same quantiles"
            )
    k = _num_outputs(p)
    if prior.num_class != k:
        raise ValueError(
            f"init_model has {prior.num_class} output(s) per row, params give num_class={k}"
        )
    if n_features is not None and prior.n_features != n_features:
        raise ValueError(
            f"init_model was trained on {prior.n_features} features, the new data has {n_features}"
        )


def _prior_margin(
    prior: Model, data: Any, base_margin: Any, p: Mapping[str, Any], *, header: bool = True
) -> np.ndarray:
    """The prior model's raw margin on `data` as a `(rows, k)` matrix: the
    starting predictions of a continued run. A `base_margin` replaces the
    prior's intercept (XGBoost: starting margin + the existing trees)."""
    out = prior.predict(
        data,
        output_margin=True,
        header=header,
        predict_threads=p["nthread"],
        missing=p["missing"],
        base_margin=base_margin,
    )
    return np.ascontiguousarray(np.asarray(out, dtype=np.float32).reshape(-1, _num_outputs(p)))


def train(
    params: Mapping[str, Any],
    data: Any,
    label: Any = None,
    num_boost_round: int = 10,
    evals: Sequence[tuple] = (),
    early_stopping_rounds: int | None = None,
    evals_result: dict | None = None,
    verbose_eval: bool | int = False,
    callbacks: Iterable[Callable[[int, dict], bool]] | None = None,
    header: bool = True,
    quiet: bool = False,
    weight: Any = None,
    feature_types: Sequence[str] | None = None,
    base_margin: Any = None,
    base_margin_eval_set: Sequence[Any] | None = None,
    init_model: Any = None,
    xgb_model: Any = None,
    qid: Any = None,
    group: Any = None,
    eval_qid: Sequence[Any] | None = None,
    eval_group: Sequence[Any] | None = None,
) -> Model:
    """训一个模型。

    `data` 收这几种:

    * numpy 二维数组(f32 / f64),`label` 给 numpy 一维数组
    * `.parquet` / 有表头 `.csv` 路径,`label` 给**列名字符串**
    * 无表头 `.csv` 路径传 `header=False`,`label` 给从 0 开始的列下标
    * pyarrow Table / RecordBatchReader,`label` 给 numpy 一维数组
    * pandas DataFrame(`category` 列需要 `params["enable_categorical"] = True`),
      `label` 给一维数组;列名记录为模型的特征名(和 pyarrow 一样),预测时按列名绑定

    `weight` 是每行样本权重(XGBoost 的 DMatrix `weight`):数组入口给
    一维数组,文件入口给**列名**(无表头 CSV 给列下标),该列不进特征。

    `evals` 是 `[(data, label, name)]` 或带权重的 `[(data, label, weight, name)]`;
    数据是路径时写 `[(path, name)]`,标签列名和权重列名都和训练集共用。
    **验证集会自动复用训练集的分箱边界** —— 各自建边界会让同一个值落进
    不同的 bin,模型直接失效。

    权重同时作用于梯度、指标和分箱边界(quantile sketch 按权重加权),
    语义和 XGBoost 一致。

    `params["missing"]`(XGBoost 的 `missing`,默认 NaN):等于它的特征值和 NaN 完全等价
    (走缺失方向),同时作用于训练集和所有验证集 —— numpy、pandas、pyarrow 和文件入口
    (Parquet / CSV 由 Rust 读取器流式处理,不在 Python 里物化)。它必须是 NaN 或有限
    的 float32 值,±inf 和保留的 f32::MAX 会报错。**模型里不保存 `missing`**(XGBoost
    也不存),预测时要传同一个 `missing`。类别列的整数码同样受它影响。

    Learning to rank (`rank:ndcg` / `rank:pairwise` / `rank:map`): give the query
    groups with `qid` (one query id per row, rows sorted by it) or `group` (the size
    of every group, array inputs only); for Parquet / CSV files `qid` is a **column
    name** (a column index for a headerless CSV), excluded from the features like
    `label`. Ranking weights are **per query group** (array inputs: one value per
    group), as in XGBoost; for files the weight column holds the group weight on
    every row of the group. Validation sets take `eval_qid` / `eval_group`, lists
    parallel to `evals` (file validation sets reuse the training `qid` column).
    The default metric is `ndcg@32` (or `map@32`), as in XGBoost.

    类别特征(XGBoost 语义):`feature_types` 给每个特征 "c"(类别)或 "q"(数值);
    pandas `category` 列和 PyArrow dictionary 列在 `params["enable_categorical"]`
    为 True 时自动识别。类别值是 0..=65533 的整数码(缺失用 NaN / null),超出范围、
    负数或非整数会报错;每个特征最多 65534 个类别(CPU 和 GPU 相同)。`max_bin` 取 1..=65535。

    `base_margin`(XGBoost 的 DMatrix `base_margin`):每行的起始 raw margin,训练从它起步
    而不是从 `base_score` 起步(截距仍然按标签估计并存进模型,和 XGBoost 一致)。数组入口给
    `(n,)`(多分类 `(n, num_class)`);文件入口给**列名**(多分类给 `num_class` 个列名的列表,
    无表头 CSV 给列下标),该列不进特征。验证集的 margin 用 `base_margin_eval_set`
    (和 `evals` 平行的列表,没有的项给 `None`);文件入口的验证集沿用同样的列名。预测时
    要自己给同样的 margin(`Model.predict(..., base_margin=...)`)。

    `init_model`(训练继续;XGBoost 的写法 `xgb_model=` 同样接受):已有的 FerrisBoost `Model`、FerrisBoost / XGBoost 模型 JSON 的路径,
    或 `xgboost.Booster`。训练从该模型在新数据(和每个验证集)上的预测起步,沿用它的
    `base_score` 和 objective,`num_boost_round` 棵**新**树接在它后面;objective / num_class /
    特征数 / 特征名 / 类别特征不一致会报错。分箱边界按新数据重新采样(旧树用的是浮点阈值,
    所以旧树的预测不依赖新边界)。`evals_result` 只含新的轮次;早停的 `best_iteration` 是
    整个模型里的绝对轮次。文件入口按**列名**绑定到模型的特征名(列序随意),所以模型要有
    特征名(带 `feature_names` 的 XGBoost 模型也行);没有特征名的 XGBoost 模型只能用数组继续,
    文件入口会报 `ValueError`。
    """
    prior = _load_prior(_pick_init_model(init_model, xgb_model))
    if prior is not None:
        params = _adopt_prior_params(params, prior)
    p = _normalize_params(params, num_boost_round)
    k_out = _num_outputs(p)
    if base_margin_eval_set is not None and len(base_margin_eval_set) != len(evals):
        raise ValueError(
            f"base_margin_eval_set has {len(base_margin_eval_set)} entries, evals has {len(evals)}"
        )
    ranking = p["objective"].startswith("rank:")
    if qid is not None and group is not None:
        raise ValueError("Either one of `group` or `qid` should be None.")
    # Private wire field: quiet is a public call option, not a tree parameter.
    p["_quiet"] = bool(quiet)
    explicit_types = _feature_flags(feature_types, None)
    cb_list = list(callbacks) if callbacks else None
    verbose = None
    if verbose_eval is True:
        verbose = 1
    elif isinstance(verbose_eval, int) and not isinstance(verbose_eval, bool):
        verbose = int(verbose_eval)
    if quiet:
        verbose = None

    specs = _as_path_specs(data)
    if specs is not None:
        if header and not isinstance(label, str):
            raise ValueError("with header=True, a file label must be a column name string")
        if not header and (not isinstance(label, int) or isinstance(label, bool)):
            raise ValueError("with header=False, a file label must be a zero-based column index")
        if weight is not None:
            if header and not isinstance(weight, str):
                raise ValueError("file inputs take weight as a column name string (header=True)")
            if not header and (not isinstance(weight, int) or isinstance(weight, bool)):
                raise ValueError("with header=False, a file weight must be a zero-based column index")
        if group is not None or eval_group is not None or eval_qid is not None:
            raise ValueError(
                "file inputs take the query groups as a qid column name (qid=\"query_id\"); "
                "group= / eval_group= / eval_qid= are for array inputs"
            )
        if qid is not None:
            if not ranking:
                raise ValueError(f"qid is only used by the rank:* objectives, not {p['objective']}")
            if header and not isinstance(qid, str):
                raise ValueError("with header=True, a file qid must be a column name string")
            if not header and (not isinstance(qid, int) or isinstance(qid, bool)):
                raise ValueError("with header=False, a file qid must be a zero-based column index")
        elif ranking:
            raise ValueError(
                f"{p['objective']} needs query groups: pass qid=<column name> for file inputs"
            )
        eval_specs = []
        for item in evals:
            if len(item) != 2:
                raise ValueError(
                    "file input evals elements must be (path, name); "
                    "the label and weight columns reuse the training column names"
                )
            eval_specs.append((os.fspath(item[0]), item[1]))
        # Declared categorical features are positional over the file's feature
        # columns (label / weight excluded); the length is checked in Rust, where
        # the schema is known. Dictionary columns are detected there.
        p["_feature_types"] = explicit_types or None
        if base_margin_eval_set is not None:
            raise ValueError(
                "file inputs read the base_margin column from every eval file: "
                "base_margin_eval_set is for array inputs"
            )
        if base_margin is not None:
            names = base_margin if isinstance(base_margin, (list, tuple)) else [base_margin]
            ok = all(isinstance(n, str) for n in names) if header else all(
                isinstance(n, int) and not isinstance(n, bool) for n in names
            )
            if not ok:
                raise ValueError(
                    "file inputs take base_margin as column name(s)"
                    + ("" if header else " (zero-based column indices for header=False)")
                )
            base_margin = list(names)
        start = eval_start = None
        if prior is not None:
            _check_prior(prior, p, None)
            # Starting margins of a continued run: the prior scored over the files.
            # A named base_margin column (not usable for a positional prior's
            # predict) replaces the prior's intercept, as in XGBoost.
            start = _prior_margin(prior, specs, base_margin, p, header=header)
            eval_start = [
                _prior_margin(prior, spec, base_margin, p, header=header)
                for spec, _ in eval_specs
            ]
        # 格式由 Rust 从解析出的文件判定 —— Python 不再按后缀猜,也不做
        # 文件系统判断(目录/glob 展开、排序、schema 一致性全在 Rust)。
        model = _core.train_file(specs, label, p, header, eval_specs,
                                 early_stopping_rounds, verbose, cb_list,
                                 weight=weight, qid=qid, base_margin=base_margin,
                                 start_margin=start, eval_start_margins=eval_start,
                                 prior_model=prior)
    else:
        table = _arrow_table(data)
        frame = data if _is_pandas_frame(data) else None
        enable_cat = p["enable_categorical"]
        auto_flags, levels, names = None, {}, None
        # A continued model indexes features by position: named data is bound to
        # the model's feature names (any column order), like evals are.
        prior_names = list(prior.feature_names) if prior is not None else []
        # A continued model keeps its category lists: named columns are coded
        # with them (the other categorical columns bring their own).
        prior_refs = _model_refs(prior)
        if table is not None:
            x, names, auto_flags, levels = _table_matrix(
                table, prior_names or None, enable_cat, prior_refs
            )
        elif frame is not None:
            if prior_names:
                cols = [str(c) for c in frame.columns]
                absent = [n for n in prior_names if n not in cols]
                if absent:
                    raise ValueError(f"data is missing the features of init_model: {absent}")
                frame = frame[[c for n in prior_names for c in frame.columns if str(c) == n]]
            x, names, auto_flags, levels = _frame_matrix(frame, enable_cat, prior_refs)
        else:
            x = _as_matrix(data)
        p["_feature_types"] = _merge_flags(auto_flags, explicit_types, x.shape[1])
        if p["_categories"] is None:
            p["_categories"] = _categories_wire(levels, p["_feature_types"], x.shape[1])
        if isinstance(p["monotone_constraints"], dict):
            if names is None:
                raise ValueError(
                    "monotone_constraints given by column name needs column names: a numpy array has none; "
                    "use a positional tuple, or a pyarrow Table / a file with a header"
                )
            p["monotone_constraints"] = _resolve_named_monotone(p["monotone_constraints"], names)
        x = _as_matrix(x)
        y = _as_labels(label, x.shape[0])
        if isinstance(weight, str):
            raise ValueError("array inputs take weight as a 1-D array; column names are only for file inputs")
        if ranking and qid is None and group is None:
            raise ValueError(
                f"{p['objective']} needs query groups: pass qid= (one id per row) or group= "
                "(the size of every group)"
            )
        q = None if qid is None else _as_query_array(qid, "qid", x.shape[0])
        g = None if group is None else _as_query_array(group, "group")
        # Ranking weights are per query group; Rust checks their count.
        w = None if weight is None else _as_weights(weight, None if ranking else x.shape[0])
        if eval_qid is not None and len(eval_qid) != len(evals):
            raise ValueError("eval_qid must have one entry per evals element")
        if eval_group is not None and len(eval_group) != len(evals):
            raise ValueError("eval_group must have one entry per evals element")
        bm = None if base_margin is None else _as_base_margin(base_margin, k_out)
        if bm is not None and bm.shape[0] != x.shape[0]:
            raise ValueError(f"base_margin has {bm.shape[0]} rows, data has {x.shape[0]}")
        if prior is not None:
            _check_prior(prior, p, x.shape[1])
        eval_bms: list[Any] = []
        eval_specs = []
        for n_eval, item in enumerate(evals):
            if len(item) == 3:
                ed, el, name = item
                ew = None
            elif len(item) == 4:
                ed, el, ew, name = item
            else:
                raise ValueError(
                    "evals elements must be (data, label, name) or (data, label, weight, name)"
                )
            et = _arrow_table(ed)
            if et is not None:
                # 训练集也有列名时按列名绑定;否则(numpy 训练集)只能按位置。
                ed = _table_matrix(et, names, enable_cat, levels, f"evals[{name}]", scoring=True)[0]
            elif _is_pandas_frame(ed):
                if names is not None:
                    missing = [n for n in names if n not in [str(c) for c in ed.columns]]
                    if missing:
                        raise ValueError(f"evals[{name}] is missing training columns {missing}")
                    ed = ed[[c for n in names for c in ed.columns if str(c) == n]]
                ed = _frame_matrix(ed, enable_cat, levels, f"evals[{name}]")[0]
            ex = _as_matrix(ed, f"evals[{name}]")
            if ew is not None:
                ew = _as_weights(ew, None if ranking else ex.shape[0],
                                 f"evals[{name}] weight")
            eq = None if eval_qid is None or eval_qid[n_eval] is None else _as_query_array(
                eval_qid[n_eval], f"eval_qid[{n_eval}]", ex.shape[0])
            eg = None if eval_group is None or eval_group[n_eval] is None else _as_query_array(
                eval_group[n_eval], f"eval_group[{n_eval}]")
            if eq is not None and eg is not None:
                raise ValueError("Only one of the eval_qid or eval_group for each evaluation "
                                 "dataset should be provided.")
            if ranking and eq is None and eg is None:
                raise ValueError(
                    f"validation set {name!r} needs query groups: pass eval_qid / eval_group"
                )
            eval_specs.append((ex, _as_labels(el, ex.shape[0]), ew, str(name), eq, eg))
            ebm = None
            if base_margin_eval_set is not None and base_margin_eval_set[n_eval] is not None:
                ebm = _as_base_margin(
                    base_margin_eval_set[n_eval], k_out, f"base_margin_eval_set[{name}]"
                )
                if ebm.shape[0] != ex.shape[0]:
                    raise ValueError(
                        f"base_margin_eval_set[{name}] has {ebm.shape[0]} rows, "
                        f"the set has {ex.shape[0]}"
                    )
            eval_bms.append(ebm)
        if prior is not None:
            # Starting predictions: the prior's margin on each data set (on top of a
            # supplied base_margin, which replaces the prior's intercept).
            bm = _prior_margin(prior, x, bm, p)
            eval_bms = [
                _prior_margin(prior, spec[0], ebm, p) for spec, ebm in zip(eval_specs, eval_bms)
            ]
        model = _core.train_dense(x, y, p, eval_specs, early_stopping_rounds,
                                  verbose, cb_list, weight=w,
                                  feature_names=None if names is None else [str(n) for n in names],
                                  qid=q, group=g, base_margin=bm,
                                  eval_base_margins=eval_bms if any(
                                      m is not None for m in eval_bms) else None,
                                  prior_model=prior)

    if evals_result is not None:
        evals_result.clear()
        evals_result.update(model.evals_result())
    return model
