"""位精确基线在 Python 侧复现。

**这是绑定层的核心验收。** `tests/fixtures/` 里的 16 个配置,通过
Python API 训出来的预测值必须和 fixture(xgboost 3.4.1 算的)**位相同**。
Windows deep_logistic 使用同平台 XGBoost 生成的精确参考：该参考与 Linux
只有 13 行的最后几位不同。保留原 Linux 文件，并校验平台参考的完整哈希。

绑定层但凡动了一点数值 —— dtype 转换、内存布局、参数默认值传错 ——
这条就会挂。它变红不要放宽容差,去查绑定层哪里改了数。
"""
import hashlib
import json
import platform
from pathlib import Path
import sys

import numpy as np
import pytest

import ferrisboost as fb

FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "fixtures"
INDEX_PATH = FIXTURES / "index.json"


def load_index():
    return json.loads(INDEX_PATH.read_text(encoding="utf-8"))["cases"]


def read_f32(name):
    return np.fromfile(FIXTURES / name, dtype="<f4")


def apply_windows_reference(case, margin, reference):
    assert margin.dtype == np.dtype("<f4")
    assert margin.ndim == 1 and margin.flags.c_contiguous
    index = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    assert reference["case"] == case["name"]
    assert case["name"] == "deep_logistic"
    assert reference["xgboost"] == index["xgboost"] == "3.4.1"
    assert reference["platform"] == "win32-amd64"
    assert reference["rows"] == case["n_rows"] == len(margin)
    original_bytes = margin.tobytes()
    assert hashlib.sha256(original_bytes).hexdigest() == reference["base_sha256"]

    replacements = reference["replacements"]
    assert isinstance(replacements, list) and len(replacements) == 13
    rows = [replacement["row"] for replacement in replacements]
    assert all(type(row) is int for row in rows)
    assert rows == sorted(set(rows)), "replacement rows must be unique and sorted"
    assert all(0 <= row < len(margin) for row in rows)

    result = margin.copy()
    original_bits = margin.view("<u4")
    result_bits = result.view("<u4")
    for replacement in replacements:
        row = replacement["row"]
        base_bits = replacement["base_bits"]
        bits = replacement["bits"]
        assert type(base_bits) is int and 0 <= base_bits <= 0xFFFFFFFF
        assert base_bits == int(original_bits[row]), f"base bits mismatch at row {row}"
        assert type(bits) is int and 0 <= bits <= 0xFFFFFFFF
        assert bits != base_bits, f"replacement does not change row {row}"
        result_bits[row] = bits
    assert hashlib.sha256(result.tobytes()).hexdigest() == reference["expected_sha256"]
    return result


def read_expected_margin(case, platform_name=sys.platform):
    margin = read_f32(f"{case['name']}.margin")
    if (platform_name == "win32" and case["name"] == "deep_logistic"
            and platform.machine().lower() in ("amd64", "x86_64")):
        reference = json.loads(
            (FIXTURES / "deep_logistic.windows.json").read_text(encoding="utf-8")
        )
        return apply_windows_reference(case, margin, reference)
    return margin


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
    want = read_expected_margin(case)

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
    want = read_expected_margin(case)

    model = fb.train(case_params(case), x32.astype(np.float64), label=y.astype(np.float64),
                     num_boost_round=case["n_rounds"])
    got = model.predict(x32, output_margin=True)
    assert got.tobytes() == want.tobytes(), "f64 转 f32 这一步改了数值"


def test_non_contiguous_input_still_correct():
    """切片出来的非连续数组要先拷贝 —— 不然按行主序取数会取错。"""
    case = load_index()[0]
    x = read_f32(f"{case['name']}.x").reshape(case["n_rows"], case["n_features"])
    y = read_f32(f"{case['name']}.y")
    want = read_expected_margin(case)

    padded = np.zeros((case["n_rows"], case["n_features"] + 3), dtype=np.float32)
    padded[:, :case["n_features"]] = x
    view = padded[:, :case["n_features"]]
    assert not view.flags["C_CONTIGUOUS"]

    model = fb.train(case_params(case), view, label=y, num_boost_round=case["n_rounds"])
    assert model.predict(x, output_margin=True).tobytes() == want.tobytes()


def test_linux_margin_reference_keeps_original_fixture_bytes():
    case = next(c for c in load_index() if c["name"] == "deep_logistic")
    original = read_f32("deep_logistic.margin")
    assert read_expected_margin(case, platform_name="linux").tobytes() == original.tobytes()


def test_windows_deep_logistic_reference_is_exact_sparse_patch(monkeypatch):
    case = next(c for c in load_index() if c["name"] == "deep_logistic")
    original = read_f32("deep_logistic.margin")
    monkeypatch.setattr(platform, "machine", lambda: "AMD64")
    expected = read_expected_margin(case, platform_name="win32")
    assert hashlib.sha256(expected.tobytes()).hexdigest() == json.loads(
        (FIXTURES / "deep_logistic.windows.json").read_text(encoding="utf-8")
    )["expected_sha256"]
    assert np.count_nonzero(expected.view("<u4") != original.view("<u4")) == 13


def test_windows_reference_is_not_selected_for_other_cases(monkeypatch):
    case = load_index()[0]
    original = read_f32(f"{case['name']}.margin")
    monkeypatch.setattr(platform, "machine", lambda: "AMD64")
    assert read_expected_margin(case, platform_name="win32").tobytes() == original.tobytes()


def test_windows_reference_requires_amd64(monkeypatch):
    case = next(c for c in load_index() if c["name"] == "deep_logistic")
    original = read_f32("deep_logistic.margin")
    monkeypatch.setattr(platform, "machine", lambda: "ARM64")
    assert read_expected_margin(case, platform_name="win32").tobytes() == original.tobytes()


@pytest.mark.parametrize(
    "corruption",
    [
        "duplicate_row",
        "unsorted_rows",
        "out_of_range_row",
        "invalid_uint32",
        "boolean_bits",
        "wrong_base_bits",
        "unchanged_replacement",
        "bad_base_hash",
        "bad_final_hash",
        "wrong_metadata",
    ],
)
def test_windows_reference_rejects_invalid_patch(corruption):
    case = next(c for c in load_index() if c["name"] == "deep_logistic")
    original = read_f32("deep_logistic.margin")
    reference = json.loads(
        (FIXTURES / "deep_logistic.windows.json").read_text(encoding="utf-8")
    )
    reference["replacements"] = [dict(item) for item in reference["replacements"]]

    replacements = reference["replacements"]
    if corruption == "duplicate_row":
        replacements[1]["row"] = replacements[0]["row"]
    elif corruption == "unsorted_rows":
        replacements[0]["row"], replacements[1]["row"] = (
            replacements[1]["row"], replacements[0]["row"]
        )
    elif corruption == "out_of_range_row":
        replacements[-1]["row"] = len(original)
    elif corruption == "invalid_uint32":
        replacements[0]["bits"] = 0x1_0000_0000
    elif corruption == "boolean_bits":
        replacements[0]["bits"] = True
    elif corruption == "wrong_base_bits":
        replacements[0]["base_bits"] ^= 1
    elif corruption == "unchanged_replacement":
        replacements[0]["bits"] = replacements[0]["base_bits"]
    elif corruption == "bad_base_hash":
        reference["base_sha256"] = "0" * 64
    elif corruption == "bad_final_hash":
        reference["expected_sha256"] = "0" * 64
    elif corruption == "wrong_metadata":
        reference["xgboost"] = "0.0.0"

    with pytest.raises(AssertionError):
        apply_windows_reference(case, original, reference)
