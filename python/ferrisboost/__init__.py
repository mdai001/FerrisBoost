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

_SKLEARN_NAMES = ("FerrisBoostClassifier", "FerrisBoostRegressor")


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
    # reg:quantileerror 必须给:单个 [0, 1] 内的分位数(长度 1 的列表也收)。
    # 向量(多分位数 = 多输出)不支持。
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
    # 0..=254. `max_cat_to_onehot`: features with fewer categories than this are
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
    # Private wire field set by `train` from `feature_types` / the frame dtypes:
    # one bool per feature, or None.
    "_feature_types": None,
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
               "multi:softmax")

# 认识但还没实现的:显式报错,别让用户以为设了就生效了
_UNIMPLEMENTED = {
    "max_leaves": "leaf-wise 生长还没实现(现在是 depthwise)",
    "tree_method": "只有 hist 一种,不用设",
    "num_parallel_tree": "随机森林模式不支持",
}


def _normalize_params(params: Mapping[str, Any], num_boost_round: int) -> dict[str, Any]:
    out = dict(_DEFAULTS)
    for key, value in dict(params).items():
        canon = _ALIASES.get(key, key)
        if canon in _UNIMPLEMENTED:
            raise ValueError(f"参数 {key!r}:{_UNIMPLEMENTED[canon]}")
        if canon not in _DEFAULTS:
            known = ", ".join(sorted(_DEFAULTS) + sorted(_ALIASES))
            raise ValueError(f"不认识的参数 {key!r}。支持的有:{known}")
        out[canon] = value

    if out["objective"] not in _OBJECTIVES:
        raise ValueError(
            f"objective {out['objective']!r} 不支持,现在只有:{', '.join(_OBJECTIVES)}"
        )
    if not 0 < out["max_bin"] <= 255:
        # 255 是上限,bin 255 留给缺失哨兵
        raise ValueError(f"max_bin 要在 1..=255,给的是 {out['max_bin']}")
    if num_boost_round < 1:
        raise ValueError(f"num_boost_round 要 >= 1,给的是 {num_boost_round}")

    out["num_boost_round"] = int(num_boost_round)
    for key in (
        "max_depth", "max_bin", "nthread", "ingest_threads", "model_io_threads",
        "seed", "num_boost_round"
    ):
        out[key] = int(out[key])
    if out["ingest_threads"] < 0:
        raise ValueError(f"ingest_threads 要 >= 0,给的是 {out['ingest_threads']}")
    if out["model_io_threads"] < 0:
        raise ValueError(f"model_io_threads 要 >= 0,给的是 {out['model_io_threads']}")
    # cols_per_block 保留 None 语义,不要被 int() 变成 0 之后和「用户真的
    # 传了 0」分不开。
    if out["cols_per_block"] is not None:
        out["cols_per_block"] = int(out["cols_per_block"])
        if out["cols_per_block"] < 1:
            raise ValueError(
                f"cols_per_block 要 >= 1,给的是 {out['cols_per_block']};"
                " 想让它自动选就别传这个参数"
            )
    # 和 cols_per_block 同样的 None 语义:显式给了就完全按用户的来。
    if out["hist_streams"] is not None:
        out["hist_streams"] = int(out["hist_streams"])
        if out["hist_streams"] not in (1, 2):
            raise ValueError(
                f"hist_streams 只支持 1 或 2,给的是 {out['hist_streams']};"
                " 想让它自动选就别传这个参数"
            )
    # 同样的 None 语义:auto 让规划器按剩余显存解,显式给了就按用户的来
    # (仍会被夹到物理块数 —— 比块还多的常驻 slot 只会白占显存)。
    if out["resident_blocks"] is not None:
        out["resident_blocks"] = int(out["resident_blocks"])
        if out["resident_blocks"] < 0:
            raise ValueError(
                f"resident_blocks 不能是负数,给的是 {out['resident_blocks']};"
                " 想让它自动选就别传这个参数,0 表示纯 streaming"
            )
    if out["hist_nodes_per_batch"] is not None:
        out["hist_nodes_per_batch"] = int(out["hist_nodes_per_batch"])
        if not 1 <= out["hist_nodes_per_batch"] <= 32:
            raise ValueError(
                f"hist_nodes_per_batch 要在 1..=32,给的是 {out['hist_nodes_per_batch']};"
                "1 表示关闭多节点合并"
            )
    if out["device"] is not None:
        d = str(out["device"])
        if d not in ("cpu", "cuda", "gpu") and not d.startswith("cuda:"):
            raise ValueError(
                f'device 只支持 "cpu" / "cuda" / "cuda:N",给的是 {d!r}'
            )
        out["device"] = d
    # max_depth 的上限由 GPU 每层节点 buffer 决定(见 Rust 的
    # `types::MAX_DEPTH_LIMIT`)。CPU 自己能跑更深,但契约取两个后端的共同
    # 安全值 —— 否则 CPU 训出来的模型在 GPU 上复现不了。
    # ⚠️ 下界是 0(只有根节点的树,合法),不是 1。
    if not 0 <= int(out["max_depth"]) <= 14:
        raise ValueError(
            f"max_depth 要在 0..=14,给的是 {out['max_depth']};"
            " 0 表示只有根节点;上限来自 GPU 每层 partition 节点 buffer"
            "(depth 14 正好用满 8192)"
        )
    if out["gpu_memory_budget"] is not None:
        out["gpu_memory_budget"] = int(out["gpu_memory_budget"])
        if out["gpu_memory_budget"] <= 0:
            raise ValueError("gpu_memory_budget 必须是正整数字节")
    if out["gpu_math"] is not None:
        out["gpu_math"] = str(out["gpu_math"])
        if out["gpu_math"] not in ("exact", "fast"):
            raise ValueError(
                f"gpu_math 只支持 \"exact\" 或 \"fast\",给的是 {out['gpu_math']!r};"
                "fast 不保证 CPU/GPU 模型逐字节相同"
            )
    if out["device_quantize"] is not None:
        out["device_quantize"] = bool(out["device_quantize"])
    if out["num_class"] is not None:
        if isinstance(out["num_class"], bool) or int(out["num_class"]) != out["num_class"]:
            raise ValueError(f"num_class 要是整数,给的是 {out['num_class']!r}")
        out["num_class"] = int(out["num_class"])
        if out["num_class"] < 0:
            raise ValueError("num_class 不能是负数")
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
            raise ValueError("cache_budget_bytes 不能是负数")
    if out["cache_path"] is not None:
        out["cache_path"] = os.fspath(out["cache_path"])
    if out["eval_metric"] is not None:
        m = out["eval_metric"]
        names = [m] if isinstance(m, str) else (
            list(m) if isinstance(m, (list, tuple)) else None
        )
        if not names or not all(isinstance(n, str) for n in names):
            raise ValueError(
                f"eval_metric 要是字符串或字符串列表,给的是 {m!r}"
            )
        # 名字和 objective 的组合由 Rust 校验(唯一一份规则)。
        out["eval_metric"] = names
    out["monotone_constraints"] = _normalize_monotone(out["monotone_constraints"])
    out["enable_categorical"] = bool(out["enable_categorical"])
    out["missing"] = _normalize_missing(out["missing"])
    for key in ("max_cat_to_onehot", "max_cat_threshold"):
        if isinstance(out[key], bool) or int(out[key]) != out[key]:
            raise ValueError(f"{key} 要是整数,给的是 {out[key]!r}")
        out[key] = int(out[key])
        if out[key] < 1:
            raise ValueError(f"{key} 要 >= 1,给的是 {out[key]}")
    return out


def _normalize_missing(value: Any) -> float:
    """`missing` -> float. NaN means "no sentinel"; ±inf and f32::MAX are rejected.

    The full rule (finite float32 below the reserved f32::MAX) is enforced once,
    in Rust; this only turns the value into a float and fails early on junk.
    """
    if value is None or isinstance(value, (str, bytes)):
        raise ValueError(f"missing 要是数字(默认 NaN),给的是 {value!r}")
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"missing 要是数字(默认 NaN),给的是 {value!r}") from None


def _normalize_quantile_alpha(value: Any, objective: str) -> float | None:
    """`quantile_alpha` -> float | None。长度 1 的序列等价于标量;多个值是多输出,不支持。"""
    if value is None:
        if objective == "reg:quantileerror":
            raise ValueError(
                "reg:quantileerror 需要 quantile_alpha(单个 [0, 1] 内的分位数,例如 0.9)"
            )
        return None
    if isinstance(value, (list, tuple, np.ndarray)):
        flat = np.asarray(value, dtype=np.float64).ravel()
        if flat.size != 1:
            raise ValueError(
                f"quantile_alpha 给了 {flat.size} 个值;多分位数(多输出)不支持,"
                "请每个分位数训一个模型"
            )
        value = flat[0]
    alpha = float(value)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"quantile_alpha 要在 [0, 1] 内,给的是 {alpha}")
    return alpha


def _monotone_sign(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"monotone_constraints 的{where}要是整数(1 / 0 / -1),给的是 {value!r}")
    # XGBoost 只看符号:2 等价于 1,-3 等价于 -1。
    return int(np.sign(int(value)))


def _normalize_monotone(mc: Any) -> list[int] | dict[str, int] | None:
    """归一成按位置的整数列表,或 {列名: 方向}。全 0 / 空等价于不约束。"""
    if mc is None:
        return None
    if isinstance(mc, str):
        text = mc.strip()
        if not (text.startswith("(") and text.endswith(")")):
            raise ValueError(f'monotone_constraints 字符串要写成 "(1,0,-1)",给的是 {mc!r}')
        body = text[1:-1].strip()
        try:
            values = [int(v) for v in body.split(",")] if body else []
        except ValueError:
            raise ValueError(f"monotone_constraints 字符串里有非整数:{mc!r}") from None
        return [_monotone_sign(v, "元素") for v in values]
    if isinstance(mc, Mapping):
        out = {}
        for name, value in mc.items():
            if not isinstance(name, str):
                raise ValueError(f"monotone_constraints 字典的键要是列名字符串,给的是 {name!r}")
            out[name] = _monotone_sign(value, f"列 {name!r} 的值")
        return out
    if isinstance(mc, (list, tuple, np.ndarray)):
        return [_monotone_sign(v, "元素") for v in np.asarray(mc, dtype=object).ravel()]
    raise ValueError(
        "monotone_constraints 要是元组/列表、\"(1,0,-1)\" 字符串或 {列名: 方向} 字典,"
        f"给的是 {type(mc).__name__}"
    )


def _resolve_named_monotone(mc: dict[str, int], names: list[str]) -> list[int]:
    unknown = [k for k in mc if k not in names]
    if unknown:
        raise ValueError(f"monotone_constraints 里的列 {unknown} 不是特征列;特征列有 {names}")
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
        raise ValueError(f"{name} 要是二维数组,给的是 {arr.ndim} 维")
    if arr.dtype == np.float32 and arr.flags["C_CONTIGUOUS"]:
        return arr
    return np.ascontiguousarray(arr, dtype=np.float32)


def _as_labels(y: Any, n_rows: int) -> np.ndarray:
    arr = np.ascontiguousarray(np.asarray(y).ravel(), dtype=np.float32)
    if arr.size != n_rows:
        raise ValueError(f"标签有 {arr.size} 个,数据有 {n_rows} 行")
    if np.isnan(arr).any():
        raise ValueError("标签里有 NaN。缺失的标签要在上游处理掉 —— "
                         "填 0 会被当成一个真实标签学进去")
    return arr


def _as_weights(w: Any, n_rows: int, name: str = "weight") -> np.ndarray:
    """样本权重 -> C 连续 f32 一维数组。规则和 XGBoost 一致:有限、非负。"""
    arr = np.ascontiguousarray(np.asarray(w, dtype=np.float32).ravel())
    if arr.size != n_rows:
        raise ValueError(f"{name} 有 {arr.size} 个,数据有 {n_rows} 行")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} 里有 NaN 或 inf;样本权重必须是有限值")
    if (arr < 0).any():
        raise ValueError(f"{name} 里有负数;样本权重必须非负")
    return arr


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
    raise ValueError(f"从后缀看不出 {path!r} 是什么格式,支持 .parquet / .csv")


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
            f'feature_types 要是 "c" / "q" 的列表(每个特征一项),给的是 {feature_types!r}'
        )
    flags = []
    for t in feature_types:
        if not isinstance(t, str):
            raise ValueError(f'feature_types 的元素要是字符串("c" / "q"),给的是 {t!r}')
        flags.append(t == "c")
    if n_features is not None and len(flags) != n_features:
        raise ValueError(f"feature_types 有 {len(flags)} 项,数据有 {n_features} 个特征")
    return flags


def _is_pandas_frame(obj: Any) -> bool:
    return (type(obj).__module__ or "").startswith("pandas") and hasattr(obj, "dtypes") \
        and hasattr(obj, "columns")


def _codes_to_f32(codes: np.ndarray, missing: np.ndarray) -> np.ndarray:
    out = codes.astype(np.float32)
    out[missing] = np.nan
    return out


def _frame_matrix(
    df: Any, enable_categorical: bool, ref_levels: dict | None = None, what: str = "data"
) -> tuple[np.ndarray, list[str], list[bool], dict]:
    """pandas DataFrame -> (f32 matrix, column names, categorical flags, levels).

    A `category` column becomes its integer codes (NaN for a missing value) and
    is flagged categorical. `ref_levels` (the training frame's levels by column
    position) re-codes a validation frame's categories onto the training codes;
    a category unseen in training gets a code past the trained range, which is
    in no category set and so goes left, as in XGBoost.
    """
    import pandas as pd

    names = [str(c) for c in df.columns]
    if len(set(names)) != len(names):
        dups = sorted({n for n in names if names.count(n) > 1})
        raise ValueError(f"{what} 里有重复的列名 {dups};列名必须唯一")
    cols, flags, levels = [], [], {}
    for i, name in enumerate(names):
        s = df.iloc[:, i]
        if isinstance(s.dtype, pd.CategoricalDtype):
            if not enable_categorical:
                raise ValueError(
                    f"{what} 的列 {name!r} 是 category 类型;要把它当类别特征训练,"
                    "请设置 params['enable_categorical'] = True"
                )
            if ref_levels is not None and i in ref_levels:
                ref = ref_levels[i]
                codes = pd.Categorical(s.astype(object), categories=ref).codes.astype(np.int64)
                unseen = (codes < 0) & s.notna().to_numpy()
                codes[unseen] = len(ref)
                levels[i] = ref
            else:
                codes = s.cat.codes.to_numpy().astype(np.int64)
                levels[i] = s.cat.categories
            cols.append(_codes_to_f32(codes, ~s.notna().to_numpy()))
            flags.append(True)
        else:
            try:
                cols.append(s.to_numpy(dtype=np.float32, na_value=np.nan))
            except (TypeError, ValueError) as e:
                raise ValueError(
                    f"{what} 的列 {name!r} 的类型是 {s.dtype},不能当数值特征;"
                    "请先转成数值,或转成 category 并设置 enable_categorical=True"
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
) -> tuple[np.ndarray, list[str], list[bool], dict]:
    """pyarrow Table -> (f32 matrix, names, categorical flags, levels).

    A dictionary column is categorical: its codes are the dictionary indices
    (dictionaries are unified across chunks first). `order` binds columns by
    name to a training table; `ref_levels` re-codes dictionary columns onto the
    training dictionary the way `_frame_matrix` does.
    """
    import pyarrow as pa
    import pyarrow.compute as pc

    names = list(table.column_names)
    dups = sorted({n for n in names if names.count(n) > 1})
    if dups:
        raise ValueError(f"{what} 里有重复的列名 {dups};列名必须唯一")
    if order is not None:
        missing = [n for n in order if n not in names]
        extra = [n for n in names if n not in order]
        if missing or extra:
            raise ValueError(
                f"{what} 的列和训练集对不上(按列名绑定):缺少 {missing},多出 {extra}"
            )
        names = order
    if any(pa.types.is_dictionary(table.schema.field(n).type) for n in names):
        table = table.unify_dictionaries()
    cols, flags, levels = [], [], {}
    for i, name in enumerate(names):
        col = table.column(name)
        if pa.types.is_dictionary(col.type):
            if not enable_categorical:
                raise ValueError(
                    f"{what} 的列 {name!r} 是 dictionary 类型;要把它当类别特征训练,"
                    "请设置 params['enable_categorical'] = True"
                )
            chunks = col.chunks
            dictionary = chunks[0].dictionary if chunks else pa.array([])
            if ref_levels is not None and i in ref_levels:
                ref = ref_levels[i]
                decoded = pa.chunked_array([c.dictionary_decode() for c in chunks], col.type.value_type)
                idx = pc.index_in(decoded, value_set=ref)
                arr = idx.cast(pa.float32()).to_numpy(zero_copy_only=False).astype(np.float32)
                null = np.asarray(decoded.is_null().to_numpy(zero_copy_only=False))
                arr = np.where(np.isnan(arr) & ~null, np.float32(len(ref)), arr)
                arr[null] = np.nan
                levels[i] = ref
            else:
                indices = pa.chunked_array([c.indices for c in chunks], pa.int32()) \
                    if chunks else pa.chunked_array([], pa.int32())
                arr = indices.cast(pa.float32()).to_numpy(zero_copy_only=False).astype(np.float32)
                levels[i] = dictionary
            cols.append(np.asarray(arr, dtype=np.float32))
            flags.append(True)
        else:
            cols.append(np.asarray(col.to_numpy(zero_copy_only=False), dtype=np.float32))
            flags.append(False)
    x = np.ascontiguousarray(np.stack(cols, axis=1)) if cols else np.empty((table.num_rows, 0), np.float32)
    return x, names, flags, levels


def _merge_flags(auto: list[bool] | None, explicit: list[bool] | None, n: int) -> list[bool] | None:
    """Explicit `feature_types` win over dtype detection; None if nothing is categorical."""
    flags = explicit if explicit is not None else auto
    if flags is None or not any(flags):
        return None
    if len(flags) != n:
        raise ValueError(f"feature_types 有 {len(flags)} 项,数据有 {n} 个特征")
    return list(flags)


def _is_named_table(data: Any) -> bool:
    """True for a pandas DataFrame or a PyArrow Table / RecordBatch."""
    return _is_pandas_frame(data) or _arrow_table(data) is not None


def _as_predict_matrix(data: Any, names: Sequence[str] | None = None) -> np.ndarray:
    """Convert a pandas DataFrame / PyArrow Table to the f32 matrix `Model.predict` scores.

    With `names` (the model's feature names), columns are selected and ordered
    by name, as XGBoost does for a DataFrame; extra columns are ignored and a
    missing one is an error. Without names the columns are taken in order.
    Categorical columns use their codes, so the frame must carry the same
    categories (in the same order) as the training data.
    """
    if _is_pandas_frame(data):
        if names:
            cols = [str(c) for c in data.columns]
            absent = [n for n in names if n not in cols]
            if absent:
                raise ValueError(f"predict data 缺少模型的特征列 {absent}")
            data = data[[c for n in names for c in data.columns if str(c) == n]]
        return _frame_matrix(data, True, what="predict data")[0]
    table = _arrow_table(data)
    if table is not None:
        return _table_matrix(table, list(names) if names else None, True, what="predict data")[0]
    raise ValueError(
        "predict 只接受文件路径(单个 / 列表 / 目录 / glob)、二维 f32 numpy 数组、"
        "pandas DataFrame 或 pyarrow Table"
    )


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
) -> Model:
    """训一个模型。

    `data` 收这几种:

    * numpy 二维数组(f32 / f64),`label` 给 numpy 一维数组
    * `.parquet` / 有表头 `.csv` 路径,`label` 给**列名字符串**
    * 无表头 `.csv` 路径传 `header=False`,`label` 给从 0 开始的列下标
    * pyarrow Table / RecordBatchReader,`label` 给 numpy 一维数组
    * pandas DataFrame(`category` 列需要 `params["enable_categorical"] = True`),
      `label` 给一维数组;和 numpy 一样,模型里不记录特征名

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

    类别特征(XGBoost 语义):`feature_types` 给每个特征 "c"(类别)或 "q"(数值);
    pandas `category` 列和 PyArrow dictionary 列在 `params["enable_categorical"]`
    为 True 时自动识别。类别值是 0..=254 的整数码(缺失用 NaN / null),超出范围、
    负数或非整数会报错;每个特征最多 255 个类别。
    """
    p = _normalize_params(params, num_boost_round)
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
            raise ValueError("header=True 时文件 label 要给列名字符串")
        if not header and (not isinstance(label, int) or isinstance(label, bool)):
            raise ValueError("header=False 时文件 label 要给从 0 开始的列下标")
        if weight is not None:
            if header and not isinstance(weight, str):
                raise ValueError("文件入口的 weight 要给列名字符串(header=True)")
            if not header and (not isinstance(weight, int) or isinstance(weight, bool)):
                raise ValueError("header=False 时文件 weight 要给从 0 开始的列下标")
        eval_specs = []
        for item in evals:
            if len(item) != 2:
                raise ValueError(
                    "文件入口的 evals 元素要是 (path, name);"
                    "标签列和权重列沿用训练集的列名"
                )
            eval_specs.append((os.fspath(item[0]), item[1]))
        # Declared categorical features are positional over the file's feature
        # columns (label / weight excluded); the length is checked in Rust, where
        # the schema is known. Dictionary columns are detected there.
        p["_feature_types"] = explicit_types or None
        # 格式由 Rust 从解析出的文件判定 —— Python 不再按后缀猜,也不做
        # 文件系统判断(目录/glob 展开、排序、schema 一致性全在 Rust)。
        model = _core.train_file(specs, label, p, header, eval_specs,
                                 early_stopping_rounds, verbose, cb_list,
                                 weight=weight)
    else:
        table = _arrow_table(data)
        frame = data if _is_pandas_frame(data) else None
        enable_cat = p["enable_categorical"]
        auto_flags, levels, names = None, {}, None
        if table is not None:
            x, names, auto_flags, levels = _table_matrix(table, None, enable_cat)
        elif frame is not None:
            x, names, auto_flags, levels = _frame_matrix(frame, enable_cat)
        else:
            x = _as_matrix(data)
        p["_feature_types"] = _merge_flags(auto_flags, explicit_types, x.shape[1])
        if isinstance(p["monotone_constraints"], dict):
            if names is None:
                raise ValueError(
                    "按列名给的 monotone_constraints 需要列名:numpy 数组没有列名,"
                    "请改用按位置的元组,或用 pyarrow Table / 带表头的文件"
                )
            p["monotone_constraints"] = _resolve_named_monotone(p["monotone_constraints"], names)
        x = _as_matrix(x)
        y = _as_labels(label, x.shape[0])
        if isinstance(weight, str):
            raise ValueError("数组入口的 weight 要给一维数组;列名只用于文件入口")
        w = None if weight is None else _as_weights(weight, x.shape[0])
        eval_specs = []
        for item in evals:
            if len(item) == 3:
                ed, el, name = item
                ew = None
            elif len(item) == 4:
                ed, el, ew, name = item
            else:
                raise ValueError(
                    "evals 的元素要是 (data, label, name) 或 (data, label, weight, name)"
                )
            et = _arrow_table(ed)
            if et is not None:
                # 训练集也有列名时按列名绑定;否则(numpy 训练集)只能按位置。
                ed = _table_matrix(et, names, enable_cat, levels, f"evals[{name}]")[0]
            elif _is_pandas_frame(ed):
                if names is not None:
                    missing = [n for n in names if n not in [str(c) for c in ed.columns]]
                    if missing:
                        raise ValueError(f"evals[{name}] 缺少训练集的列 {missing}")
                    ed = ed[[c for n in names for c in ed.columns if str(c) == n]]
                ed = _frame_matrix(ed, enable_cat, levels, f"evals[{name}]")[0]
            ex = _as_matrix(ed, f"evals[{name}]")
            if ew is not None:
                ew = _as_weights(ew, ex.shape[0], f"evals[{name}] 的 weight")
            eval_specs.append((ex, _as_labels(el, ex.shape[0]), ew, str(name)))
        model = _core.train_dense(x, y, p, eval_specs, early_stopping_rounds,
                                  verbose, cb_list, weight=w,
                                  feature_names=None if names is None else [str(n) for n in names])

    if evals_result is not None:
        evals_result.clear()
        evals_result.update(model.evals_result())
    return model
