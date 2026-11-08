"""Exception type of every class of rejected input.

Input the caller must fix raises `ValueError`, whichever layer (Python or the
Rust core) notices it. `RuntimeError` is for failures that are not the caller's
input: I/O, a corrupt file, CUDA. The table below pins one case per class; a new
validation error in the core should be tagged with `crate::error` so it lands in
the first group.
"""
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import ferrisboost as fb

xgb = pytest.importorskip("xgboost")

RNG = np.random.default_rng(3)
N = 120
X = RNG.integers(0, 8, size=(N, 4)).astype(np.float32)
Y01 = (X[:, 0] + RNG.normal(0, 1, N) > 3.5).astype(np.float32)
Y3 = np.digitize(X[:, 0], [2.5, 5.5]).astype(np.float32)
YREG = X[:, 0] + RNG.normal(0, 1, N).astype(np.float32)
QID = np.repeat(np.arange(N // 10), 10)
YRANK = RNG.integers(0, 3, N).astype(np.float32)

BASE = {"max_depth": 3, "eta": 0.3, "nthread": 1}


def p(objective, **extra):
    return {**BASE, "objective": objective, **extra}


def train(objective, y, **kw):
    extra = kw.pop("params", {})
    return fb.train(p(objective, **extra), X, label=y, num_boost_round=2, quiet=True, **kw)


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    d = tmp_path_factory.mktemp("errfiles")
    cols = {f"f{i}": X[:, i] for i in range(4)}

    def write(name, **override):
        t = pa.table({**cols, "label": Y01, **override})
        path = str(d / f"{name}.parquet")
        pq.write_table(t, path)
        return path

    out = {
        "good": write("good"),
        "nan_label": write("nan_label", label=np.where(np.arange(N) == 3, np.nan, Y01)
                           .astype(np.float32)),
        "string_feature": write("string_feature", f1=pa.array(["a"] * N)),
        "max_feature": write("max_feature", f2=np.where(np.arange(N) == 5, np.finfo(np.float32).max,
                                                        X[:, 2]).astype(np.float32)),
        "negative_weight": write("negative_weight", w=np.where(np.arange(N) == 2, -1.0, 1.0)
                                 .astype(np.float32)),
        "float_qid": write("float_qid", q=np.arange(N, dtype=np.float32) * 0.5),
        "no_f3": str(d / "no_f3.parquet"),
        "dir": str(d),
    }
    pq.write_table(pa.table({k: v for k, v in cols.items() if k != "f3"} | {"label": Y01}),
                   out["no_f3"])
    pq.write_table(pa.table({"only_features": X[:, 0]}), str(d / "no_label.parquet"))
    out["no_label"] = str(d / "no_label.parquet")
    pq.write_table(pa.table({**{k: v[:0] for k, v in cols.items()}, "label": Y01[:0]}),
                   str(d / "empty.parquet"))
    out["empty"] = str(d / "empty.parquet")
    pd.DataFrame({**cols, "label": Y01}).to_csv(d / "mixed.csv", index=False)
    out["csv"] = str(d / "mixed.csv")
    return out


def file_train(files, key, objective="binary:logistic", label="label", **kw):
    extra = kw.pop("params", {})
    return fb.train(p(objective, **extra), files[key], label=label, num_boost_round=2,
                    quiet=True, **kw)


@pytest.fixture(scope="module")
def prior():
    return train("binary:logistic", Y01)


# --- the table: (id, callable(files, prior)) raising ValueError ---------------------------

VALUE_ERRORS = {
    # objective / label validation (Rust core)
    "binary_label_out_of_range": lambda f, m: train("binary:logistic", Y01 + 1.0),
    "poisson_negative_label": lambda f, m: train("count:poisson", YREG - 5.0),
    "gamma_zero_label": lambda f, m: train("reg:gamma", np.zeros(N, np.float32)),
    "multiclass_label_out_of_range": lambda f, m: train("multi:softprob", Y3 + 5,
                                                         params={"num_class": 3}),
    "multiclass_fractional_label": lambda f, m: train("multi:softprob", Y3 + 0.5,
                                                       params={"num_class": 3}),
    "nan_label": lambda f, m: train("reg:squarederror", np.where(np.arange(N) == 1, np.nan, YREG)
                                    .astype(np.float32)),
    "negative_weight": lambda f, m: train("reg:squarederror", YREG,
                                          weight=np.where(np.arange(N) == 0, -1.0, 1.0)),
    "bad_base_score": lambda f, m: train("binary:logistic", Y01, params={"base_score": 2.0}),
    "max_depth_too_large": lambda f, m: train("binary:logistic", Y01, params={"max_depth": 99}),
    "tweedie_power": lambda f, m: train("reg:tweedie", np.abs(YREG),
                                        params={"tweedie_variance_power": 3.0}),
    "quantile_alpha_range": lambda f, m: train("reg:quantileerror", YREG,
                                               params={"quantile_alpha": 1.5}),
    # ranking (Rust core)
    "ndcg_fractional_label": lambda f, m: fb.train(
        p("rank:ndcg"), X, label=YRANK + 0.5, qid=QID, num_boost_round=2, quiet=True),
    "map_non_binary_label": lambda f, m: fb.train(
        p("rank:map"), X, label=YRANK, qid=QID, num_boost_round=2, quiet=True),
    "unsorted_qid": lambda f, m: fb.train(
        p("rank:ndcg"), X, label=YRANK, qid=QID[::-1], num_boost_round=2, quiet=True),
    # continuation (Rust core and Python layer)
    "continuation_base_score_conflict": lambda f, m: train(
        "binary:logistic", Y01, xgb_model=m, params={"base_score": 0.123}),
    "continuation_feature_types_mismatch": lambda f, m: fb.train(
        p("binary:logistic"), np.stack([RNG.integers(0, 3, N), X[:, 1]], 1).astype(np.float32),
        label=Y01, num_boost_round=2, quiet=True, xgb_model=_categorical_model()),
    "continuation_objective_conflict": lambda f, m: train(
        "reg:squarederror", YREG, xgb_model=m),
    "continuation_feature_count": lambda f, m: fb.train(
        p("binary:logistic"), X[:, :3], label=Y01, num_boost_round=2, quiet=True, xgb_model=m),
    # file input (Rust core)
    "file_missing_label_column": lambda f, m: file_train(f, "good", label="nope"),
    "file_no_label_column_in_file": lambda f, m: file_train(f, "no_label"),
    "file_nan_label": lambda f, m: file_train(f, "nan_label"),
    "file_without_rows": lambda f, m: file_train(f, "empty"),
    "file_string_feature": lambda f, m: file_train(f, "string_feature"),
    "file_reserved_max_value": lambda f, m: file_train(f, "max_feature"),
    "file_negative_weight": lambda f, m: file_train(f, "negative_weight", weight="w"),
    "file_non_integer_qid": lambda f, m: file_train(f, "float_qid", objective="rank:ndcg",
                                                    qid="q"),
    "file_feature_types_length": lambda f, m: file_train(f, "good",
                                                         feature_types=["q"] * 9),
    "file_eval_columns_differ": lambda f, m: file_train(f, "good", evals=[(f["no_f3"], "v")]),
    "file_mixed_formats": lambda f, m: fb.train(p("binary:logistic"), [f["good"], f["csv"]],
                                                label="label", num_boost_round=2, quiet=True),
    "file_nonexistent_input": lambda f, m: file_train(
        {"x": f["dir"] + "/does_not_exist.parquet"}, "x"),
    # prediction
    "predict_file_missing_feature": lambda f, m: _named_model().predict(f["no_f3"]),
    "predict_array_wrong_width": lambda f, m: m.predict(X[:, :3]),
    "predict_explain_combo": lambda f, m: m.predict(X, pred_contribs=True, pred_leaf=True),
    "predict_to_parquet_bad_extension": lambda f, m: _named_model().predict_to_parquet(
        f["good"], f["dir"] + "/out.txt"),
}


def _named_model():
    frame = pd.DataFrame({f"f{i}": X[:, i] for i in range(4)})
    return fb.train(p("binary:logistic"), frame, label=Y01, num_boost_round=2, quiet=True)


def _categorical_model():
    frame = pd.DataFrame({"a": pd.Categorical(RNG.integers(0, 3, N)), "b": X[:, 1]})
    return fb.train({**p("binary:logistic"), "enable_categorical": True}, frame, label=Y01,
                    num_boost_round=2, quiet=True)


@pytest.mark.parametrize("case", sorted(VALUE_ERRORS))
def test_rejected_input_is_a_value_error(case, files, prior):
    with pytest.raises(ValueError) as info:
        VALUE_ERRORS[case](files, prior)
    assert not isinstance(info.value, RuntimeError)


def test_a_persistent_cache_that_does_not_match_is_a_value_error(tmp_path, files):
    cache = str(tmp_path / "cache")
    fb.train({**p("binary:logistic"), "max_bin": 32, "cache_path": cache}, files["good"],
             label="label", num_boost_round=1, quiet=True)
    with pytest.raises(ValueError, match="max_bin|缓存"):
        fb.train({**p("binary:logistic"), "max_bin": 64, "cache_path": cache}, files["good"],
                 label="label", num_boost_round=1, quiet=True)


# --- failures that are not the caller's input keep their own types -------------------

def test_a_callback_exception_keeps_its_type(files):
    def boom(epoch, history):
        raise RuntimeError("boom from callback")

    with pytest.raises(RuntimeError, match="boom from callback"):
        fb.train(p("binary:logistic"), X, label=Y01, num_boost_round=2, quiet=True,
                 callbacks=[boom])


def test_a_missing_model_file_is_a_file_not_found_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        fb.Model.load_model(str(tmp_path / "absent.json"), quiet=True)
