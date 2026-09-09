"""ferrisboost —— 列分块的梯度提升树,数值对齐 XGBoost。

参数名沿用 XGBoost 的写法,数据入口按自己的数据流设计:**没有 DMatrix**。
那个对象模型不可变、假设全量入内存,正是这个项目要摆脱的东西。

    import ferrisboost as fb
    import numpy as np

    model = fb.train({"max_depth": 6, "eta": 0.3, "objective": "binary:logistic"},
                     X, label=y, num_boost_round=100,
                     evals=[(Xv, yv, "valid")], early_stopping_rounds=10)
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

# 规范名 -> 默认值。**未知参数报错**,不像 XGBoost 那样静默忽略 ——
# 参数名打错了不给任何提示,是最难查的一类问题。
_DEFAULTS: dict[str, Any] = {
    "objective": "binary:logistic",
    "max_depth": 6,
    "max_bin": 255,
    "eta": 0.3,
    "lambda": 1.0,
    "gamma": 0.0,
    "min_child_weight": 1.0,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
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
    "base_score": 0.5,
    "cache_budget_bytes": None,
    "cache_path": None,
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

_OBJECTIVES = ("binary:logistic", "reg:squarederror")

# 认识但还没实现的:显式报错,别让用户以为设了就生效了
_UNIMPLEMENTED = {
    "alpha": "L1 正则还没实现",
    "max_leaves": "leaf-wise 生长还没实现(现在是 depthwise)",
    "scale_pos_weight": "还没实现",
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
    for key in ("eta", "lambda", "gamma", "min_child_weight", "subsample",
                "colsample_bytree", "base_score"):
        out[key] = float(out[key])
    if out["cache_budget_bytes"] is not None:
        out["cache_budget_bytes"] = int(out["cache_budget_bytes"])
        if out["cache_budget_bytes"] < 0:
            raise ValueError("cache_budget_bytes 不能是负数")
    if out["cache_path"] is not None:
        out["cache_path"] = os.fspath(out["cache_path"])
    return out


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


def _from_pyarrow(obj: Any) -> np.ndarray | None:
    """pyarrow Table / RecordBatch -> f32 矩阵。不是 pyarrow 就返回 None。"""
    mod = type(obj).__module__ or ""
    if not mod.startswith("pyarrow"):
        return None
    try:
        table = obj if hasattr(obj, "column_names") else obj.read_all()
    except AttributeError:
        return None
    cols = [np.asarray(table.column(c).to_numpy(zero_copy_only=False), dtype=np.float32)
            for c in table.column_names]
    return np.ascontiguousarray(np.stack(cols, axis=1))


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
) -> Model:
    """训一个模型。

    `data` 收这几种:

    * numpy 二维数组(f32 / f64),`label` 给 numpy 一维数组
    * `.parquet` / 有表头 `.csv` 路径,`label` 给**列名字符串**
    * 无表头 `.csv` 路径传 `header=False`,`label` 给从 0 开始的列下标
    * pyarrow Table / RecordBatchReader,`label` 给 numpy 一维数组

    `evals` 是 `[(data, label, name)]`;数据是路径时写 `[(path, name)]`,
    标签列名和训练集共用。**验证集会自动复用训练集的分箱边界** ——
    各自建边界会让同一个值落进不同的 bin,模型直接失效。
    """
    p = _normalize_params(params, num_boost_round)
    # Private wire field: quiet is a public call option, not a tree parameter.
    p["_quiet"] = bool(quiet)
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
        eval_specs = [(os.fspath(d), name) for d, name in evals]
        # 格式由 Rust 从解析出的文件判定 —— Python 不再按后缀猜,也不做
        # 文件系统判断(目录/glob 展开、排序、schema 一致性全在 Rust)。
        model = _core.train_file(specs, label, p, header, eval_specs,
                                 early_stopping_rounds, verbose, cb_list)
    else:
        arrow = _from_pyarrow(data)
        x = _as_matrix(arrow if arrow is not None else data)
        y = _as_labels(label, x.shape[0])
        eval_specs = []
        for item in evals:
            if len(item) != 3:
                raise ValueError("evals 的元素要是 (data, label, name)")
            ed, el, name = item
            ea = _from_pyarrow(ed)
            ex = _as_matrix(ea if ea is not None else ed, f"evals[{name}]")
            eval_specs.append((ex, _as_labels(el, ex.shape[0]), str(name)))
        model = _core.train_dense(x, y, p, eval_specs, early_stopping_rounds,
                                  verbose, cb_list)

    if evals_result is not None:
        evals_result.clear()
        evals_result.update(model.evals_result())
    return model
