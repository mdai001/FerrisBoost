"""位精确基线在 Python 侧复现。

**这是绑定层的核心验收。** `tests/fixtures/` 里的 16 个配置,通过
Python API 训出来的预测值必须和 fixture(xgboost 3.4.1 算的)**位相同**。

绑定层但凡动了一点数值 —— dtype 转换、内存布局、参数默认值传错 ——
这条就会挂。它变红不要放宽容差,去查绑定层哪里改了数。
"""
import json
from pathlib import Path

import numpy as np
import pytest

import ferrisboost as fb

FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures"


def load_index():
    return json.loads((FIXTURES / "index.json").read_text())["cases"]


def read_f32(name):
    return np.fromfile(FIXTURES / name, dtype="<f4")


def case_params(c):
    return {
        "objective": c["objective"],
        "max_depth": c["max_depth"],
        "max_bin": c["max_bin"],
        "eta": c["eta"],
        "lambda": c["lambda"],
        "gamma": c["gamma"],
        "min_child_weight": c["min_child_weight"],
        "cols_per_block": c["cols_per_block"],
        "base_score": c["base_score"],
        "nthread": 1,
    }


@pytest.mark.parametrize("case", load_index(), ids=lambda c: c["name"])
def test_bit_exact_vs_xgboost(case):
    x = read_f32(f"{case['name']}.x").reshape(case["n_rows"], case["n_features"])
    y = read_f32(f"{case['name']}.y")
    want = read_f32(f"{case['name']}.margin")

    model = fb.train(case_params(case), x, label=y,
                     num_boost_round=case["n_rounds"])
    got = model.predict(x, output_margin=True)

    assert got.dtype == np.float32
    # 逐位相同,不是 allclose
    assert got.tobytes() == want.tobytes(), (
        f"{case['name']}:Python 侧训的和基准不是位相同。"
        f"最大差 {np.abs(got.astype(np.float64) - want.astype(np.float64)).max():.3e}"
    )


def test_float64_input_is_converted_explicitly():
    """f64 输入要显式转 f32,而且结果和直接传 f32 一样。"""
    case = load_index()[0]
    x32 = read_f32(f"{case['name']}.x").reshape(case["n_rows"], case["n_features"])
    y = read_f32(f"{case['name']}.y")
    want = read_f32(f"{case['name']}.margin")

    model = fb.train(case_params(case), x32.astype(np.float64), label=y.astype(np.float64),
                     num_boost_round=case["n_rounds"])
    got = model.predict(x32, output_margin=True)
    assert got.tobytes() == want.tobytes(), "f64 转 f32 这一步改了数值"


def test_non_contiguous_input_still_correct():
    """切片出来的非连续数组要先拷贝 —— 不然按行主序取数会取错。"""
    case = load_index()[0]
    x = read_f32(f"{case['name']}.x").reshape(case["n_rows"], case["n_features"])
    y = read_f32(f"{case['name']}.y")
    want = read_f32(f"{case['name']}.margin")

    padded = np.zeros((case["n_rows"], case["n_features"] + 3), dtype=np.float32)
    padded[:, :case["n_features"]] = x
    view = padded[:, :case["n_features"]]
    assert not view.flags["C_CONTIGUOUS"]

    model = fb.train(case_params(case), view, label=y, num_boost_round=case["n_rounds"])
    assert model.predict(x, output_margin=True).tobytes() == want.tobytes()
