"""scikit-learn 估计器(ferrisboost.sklearn)。"""
import pickle
import subprocess
import sys

import numpy as np
import pytest

import ferrisboost as fb

sklearn = pytest.importorskip("sklearn")
from sklearn.base import clone  # noqa: E402
from sklearn.datasets import make_classification, make_regression  # noqa: E402
from sklearn.model_selection import GridSearchCV, cross_val_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.utils.estimator_checks import parametrize_with_checks  # noqa: E402

from ferrisboost.sklearn import FerrisBoostClassifier, FerrisBoostRegressor  # noqa: E402


def small_clf(**kw):
    return FerrisBoostClassifier(n_estimators=10, max_depth=3, n_jobs=1, **kw)


def small_reg(**kw):
    return FerrisBoostRegressor(n_estimators=10, max_depth=3, n_jobs=1, **kw)


# ---------------------------------------------------------------- estimator checks
#
# Checks expected to fail (the rest pass):
#   check_sample_weight_equivalence_on_dense_data -- requires "weight k" to equal "the row
#     repeated k times"; the quantile sketch that sets the bin boundaries (and
#     min_child_weight handling) makes that impossible for a binned GBDT. It happens to
#     pass for the regressor on this data, so only the classifier is listed.
def _expected_failed_checks(estimator):
    if isinstance(estimator, FerrisBoostClassifier):
        return {
            "check_sample_weight_equivalence_on_dense_data":
                "weights are not replicated rows for a binned GBDT (sketch cuts)",
        }
    return {}


@parametrize_with_checks(
    [small_clf(), small_reg()],
    expected_failed_checks=_expected_failed_checks,
)
def test_sklearn_estimator_checks(estimator, check):
    check(estimator)


# ---------------------------------------------------------------- 基本行为
def test_package_imports_without_sklearn():
    code = (
        "import sys; sys.modules['sklearn'] = None; import ferrisboost as fb; "
        "assert fb.train; "
        "\ntry:\n    fb.FerrisBoostClassifier\nexcept ImportError as e:\n    print('ImportError-ok')"
    )
    out = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert "ImportError-ok" in out.stdout


def test_lazy_top_level_names():
    assert fb.FerrisBoostClassifier is FerrisBoostClassifier
    assert fb.FerrisBoostRegressor is FerrisBoostRegressor


def test_binary_classifier_matches_train():
    X, y = make_classification(n_samples=800, n_features=8, random_state=0)
    clf = FerrisBoostClassifier(n_estimators=15, max_depth=4, learning_rate=0.2, n_jobs=1,
                                random_state=3).fit(X, y)
    ref = fb.train({"objective": "binary:logistic", "max_depth": 4, "eta": 0.2, "nthread": 1,
                    "seed": 3}, X, label=y, num_boost_round=15, quiet=True)
    p = clf.predict_proba(X)
    assert p.shape == (800, 2)
    np.testing.assert_array_equal(p[:, 1], ref.predict(X.astype(np.float32)))
    np.testing.assert_allclose(p.sum(axis=1), 1.0, rtol=1e-6)
    np.testing.assert_array_equal(clf.predict(X), (p[:, 1] > 0.5).astype(int))
    assert list(clf.classes_) == [0, 1]
    assert clf.n_features_in_ == 8
    assert clf.score(X, y) > 0.9


def test_multiclass_classifier_and_string_labels():
    X, y = make_classification(n_samples=900, n_features=10, n_informative=6, n_classes=3,
                               random_state=1)
    names = np.array(["a", "b", "c"])[y]
    clf = small_clf().fit(X, names)
    assert list(clf.classes_) == ["a", "b", "c"]
    proba = clf.predict_proba(X)
    assert proba.shape == (900, 3)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)
    assert set(clf.predict(X)) <= {"a", "b", "c"}
    assert (clf.predict(X) == names).mean() > 0.8
    assert clf.get_booster().objective == "multi:softprob"
    assert clf.get_booster().num_class == 3


def test_multi_softmax_objective():
    X, y = make_classification(n_samples=600, n_features=8, n_informative=5, n_classes=3,
                               random_state=2)
    clf = small_clf(objective="multi:softmax").fit(X, y)
    proba = clf.predict_proba(X)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, rtol=1e-5)
    np.testing.assert_array_equal(clf.predict(X), clf.classes_[proba.argmax(axis=1)])


def test_regressor_matches_train_and_score():
    X, y = make_regression(n_samples=700, n_features=6, noise=5.0, random_state=0)
    reg = FerrisBoostRegressor(n_estimators=20, max_depth=4, n_jobs=1).fit(X, y)
    ref = fb.train({"objective": "reg:squarederror", "max_depth": 4, "nthread": 1}, X, label=y,
                   num_boost_round=20, quiet=True)
    np.testing.assert_array_equal(reg.predict(X), ref.predict(X.astype(np.float32)))
    assert reg.score(X, y) > 0.9


def test_quantile_regressor():
    X, y = make_regression(n_samples=1500, n_features=5, noise=20.0, random_state=0)
    lo = FerrisBoostRegressor(objective="reg:quantileerror", quantile_alpha=0.1, n_estimators=40,
                              max_depth=3, n_jobs=1).fit(X, y)
    hi = FerrisBoostRegressor(objective="reg:quantileerror", quantile_alpha=0.9, n_estimators=40,
                              max_depth=3, n_jobs=1).fit(X, y)
    assert np.mean(lo.predict(X) < hi.predict(X)) > 0.95
    assert abs(np.mean(y <= lo.predict(X)) - 0.1) < 0.1
    assert abs(np.mean(y <= hi.predict(X)) - 0.9) < 0.1


def test_sample_weight_and_eval_set_and_evals_result():
    X, y = make_classification(n_samples=1000, n_features=8, random_state=0)
    w = np.random.default_rng(0).uniform(0.5, 2.0, len(y))
    clf = small_clf(eval_metric=["logloss", "auc"]).fit(
        X[:800], y[:800], sample_weight=w[:800],
        eval_set=[(X[800:], y[800:]), (X[:100], y[:100])],
        sample_weight_eval_set=[w[800:], None], verbose=False)
    res = clf.evals_result()
    assert list(res) == ["validation_0", "validation_1"]
    assert list(res["validation_0"]) == ["logloss", "auc"]
    assert len(res["validation_0"]["logloss"]) == 10
    unweighted = small_clf().fit(X[:800], y[:800])
    assert not np.array_equal(clf.predict_proba(X[800:]), unweighted.predict_proba(X[800:]))


def test_early_stopping_and_best_iteration_used_by_predict():
    X, y = make_classification(n_samples=1500, n_features=10, flip_y=0.3, random_state=0)
    clf = FerrisBoostClassifier(n_estimators=300, learning_rate=0.5, max_depth=5, n_jobs=1,
                                early_stopping_rounds=5, eval_metric="logloss")
    clf.fit(X[:1000], y[:1000], eval_set=[(X[1000:], y[1000:])], verbose=False)
    best = clf.best_iteration
    assert 0 <= best < 299
    assert clf.n_estimators_ == best + 1 + 5 or clf.n_estimators_ <= 300
    booster = clf.get_booster()
    # predict 自动截断到 best_iteration(和 XGBoost sklearn 层一致),Booster 默认用全部树。
    np.testing.assert_array_equal(
        clf.predict_proba(X[1000:])[:, 1],
        booster.predict(X[1000:].astype(np.float32), iteration_range=(0, best + 1)))
    assert not np.array_equal(clf.predict_proba(X[1000:])[:, 1],
                              booster.predict(X[1000:].astype(np.float32)))
    # 显式 iteration_range 优先。
    np.testing.assert_array_equal(
        clf.predict_proba(X[1000:], iteration_range=(0, 3))[:, 1],
        booster.predict(X[1000:].astype(np.float32), iteration_range=(0, 3)))
    assert np.isfinite(clf.best_score)


def test_best_iteration_missing_without_early_stopping():
    X, y = make_classification(n_samples=200, random_state=0)
    clf = small_clf().fit(X, y)
    with pytest.raises(AttributeError):
        clf.best_iteration  # noqa: B018


def test_early_stopping_requires_eval_set():
    X, y = make_classification(n_samples=200, random_state=0)
    with pytest.raises(ValueError, match="eval_set"):
        small_clf(early_stopping_rounds=3).fit(X, y)


def test_get_set_params_clone_and_unfitted():
    est = FerrisBoostClassifier(n_estimators=7, learning_rate=0.1, colsample_bynode=0.5)
    params = est.get_params()
    assert params["n_estimators"] == 7 and params["colsample_bynode"] == 0.5
    c = clone(est)
    assert c.get_params() == params
    c.set_params(max_depth=2, subsample=0.7)
    assert c.max_depth == 2 and c.subsample == 0.7 and est.max_depth is None
    with pytest.raises(Exception):  # NotFittedError
        est.predict(np.zeros((2, 3)))
    with pytest.raises(ValueError):
        est.set_params(not_a_param=1)


def test_pandas_feature_names_and_feature_importances():
    pd = pytest.importorskip("pandas")
    X, y = make_classification(n_samples=500, n_features=5, random_state=0)
    df = pd.DataFrame(X, columns=list("abcde"))
    clf = small_clf().fit(df, y)
    assert list(clf.feature_names_in_) == list("abcde")
    p = clf.predict_proba(df)
    with pytest.raises(ValueError, match="feature names"):
        clf.predict_proba(df.rename(columns={"a": "z"}))
    imp = clf.feature_importances_
    assert imp.shape == (5,) and abs(imp.sum() - 1.0) < 1e-5
    score = clf.get_booster().get_score(importance_type="gain")
    total = sum(score.values())
    for i in range(5):
        np.testing.assert_allclose(imp[i], score.get(f"f{i}", 0.0) / total, rtol=1e-5)
    other = small_clf(importance_type="weight").fit(df, y).feature_importances_
    assert abs(other.sum() - 1.0) < 1e-5 and not np.allclose(other, imp)
    del p


def test_save_load_round_trip_and_xgboost_interchange(tmp_path):
    xgb = pytest.importorskip("xgboost")
    X, y = make_classification(n_samples=600, n_features=6, random_state=0)
    clf = small_clf().fit(X, y)
    path = tmp_path / "clf.json"
    clf.save_model(path)
    again = FerrisBoostClassifier().load_model(path)
    np.testing.assert_array_equal(again.predict_proba(X), clf.predict_proba(X))
    assert again.n_features_in_ == 6 and list(again.classes_) == [0, 1]
    booster = xgb.Booster(model_file=str(path))
    np.testing.assert_allclose(booster.predict(xgb.DMatrix(X.astype(np.float32))),
                               clf.predict_proba(X)[:, 1], rtol=1e-6)
    # 反向:XGBoost 训的模型读进来
    xclf = xgb.XGBClassifier(n_estimators=8, max_depth=3, tree_method="hist", n_jobs=1).fit(X, y)
    xpath = tmp_path / "x.json"
    xclf.save_model(str(xpath))
    loaded = FerrisBoostClassifier.from_file(xpath)
    np.testing.assert_allclose(loaded.predict_proba(X), xclf.predict_proba(X), rtol=1e-6)
    reg = small_reg().fit(X, y.astype(float))
    reg.save_model(tmp_path / "r.json")
    np.testing.assert_array_equal(FerrisBoostRegressor.from_file(tmp_path / "r.json").predict(X),
                                  reg.predict(X))
    with pytest.raises(ValueError):
        FerrisBoostRegressor().load_model(path) if False else FerrisBoostClassifier().load_model(
            tmp_path / "r.json")


def test_pickle_round_trip():
    X, y = make_classification(n_samples=300, random_state=0)
    clf = small_clf().fit(X, y)
    again = pickle.loads(pickle.dumps(clf))
    np.testing.assert_array_equal(again.predict_proba(X), clf.predict_proba(X))


def test_matches_xgboost_sklearn_predictions():
    xgb = pytest.importorskip("xgboost")
    rng = np.random.default_rng(0)
    X = rng.integers(0, 9, size=(2000, 6)).astype(np.float32)  # 分箱无损
    y = (X[:, 0] - X[:, 1] + rng.normal(size=2000) > 0).astype(int)
    kw = {"n_estimators": 12, "max_depth": 4, "learning_rate": 0.3}
    ours = FerrisBoostClassifier(n_jobs=1, **kw).fit(X, y)
    ref = xgb.XGBClassifier(tree_method="hist", n_jobs=1, **kw).fit(X, y)
    np.testing.assert_array_equal(ours.predict_proba(X), ref.predict_proba(X))
    yr = X[:, 0] * 0.5 + rng.normal(size=2000).astype(np.float32)
    ours_r = FerrisBoostRegressor(n_jobs=1, **kw).fit(X, yr)
    ref_r = xgb.XGBRegressor(tree_method="hist", n_jobs=1, **kw).fit(X, yr)
    np.testing.assert_array_equal(ours_r.predict(X), ref_r.predict(X))


# ---------------------------------------------------------------- sklearn 生态
def test_cross_val_score_gridsearch_and_pipeline():
    X, y = make_classification(n_samples=500, n_features=8, random_state=0)
    scores = cross_val_score(small_clf(), X, y, cv=3)
    assert scores.shape == (3,) and scores.mean() > 0.8
    grid = GridSearchCV(small_clf(), {"max_depth": [2, 4], "learning_rate": [0.1, 0.3]}, cv=3)
    grid.fit(X, y)
    assert grid.best_params_["max_depth"] in (2, 4)
    pipe = make_pipeline(StandardScaler(), small_clf())
    assert pipe.fit(X, y).score(X, y) > 0.9
    Xr, yr = make_regression(n_samples=400, n_features=5, noise=3.0, random_state=0)
    assert cross_val_score(small_reg(), Xr, yr, cv=3).mean() > 0.7


def test_gridsearch_parallel_pickles_estimators():
    X, y = make_classification(n_samples=300, n_features=6, random_state=0)
    grid = GridSearchCV(small_clf(), {"max_depth": [2, 3]}, cv=2, n_jobs=2).fit(X, y)
    assert grid.best_score_ > 0.7


# ---------------------------------------------------------------- missing / apply / categorical
def _int_data(n=900, m=5, seed=0, sentinel=-999.0, frac=0.15):
    rng = np.random.default_rng(seed)
    x = rng.integers(1, 9, size=(n, m)).astype(np.float32)
    signal = 0.4 * (x[:, 0] - 4) + 0.3 * (x[:, 1] % 3)
    y = (signal + rng.normal(0, 0.6, n) > 0).astype(int)
    hole = rng.random(x.shape) < frac
    xs, xn = x.copy(), x.copy()
    xs[hole] = sentinel
    xn[hole] = np.nan
    return xs, xn, y


def test_missing_is_a_constructor_param_and_passes_through():
    xgb = pytest.importorskip("xgboost")
    assert np.isnan(FerrisBoostClassifier().get_params()["missing"])
    xs, xn, y = _int_data()
    kw = {"n_estimators": 8, "max_depth": 4, "learning_rate": 0.3}
    ours = FerrisBoostClassifier(n_jobs=1, missing=-999.0, **kw).fit(xs, y)
    plain = FerrisBoostClassifier(n_jobs=1, **kw).fit(xn, y)
    np.testing.assert_array_equal(ours.predict_proba(xs), plain.predict_proba(xn))
    np.testing.assert_array_equal(ours.predict(xs), plain.predict(xn))
    ref = xgb.XGBClassifier(tree_method="hist", n_jobs=1, missing=-999.0, **kw).fit(xs, y)
    np.testing.assert_array_equal(ours.predict_proba(xs), ref.predict_proba(xs))
    # regressor, eval_set (the sentinel applies to the validation data too)
    yr = np.nan_to_num(xn[:, 0]) * 0.5
    r = FerrisBoostRegressor(n_jobs=1, missing=-999.0, n_estimators=6, max_depth=3,
                             eval_metric="rmse")
    r.fit(xs, yr, eval_set=[(xs, yr)])
    r2 = FerrisBoostRegressor(n_jobs=1, n_estimators=6, max_depth=3, eval_metric="rmse")
    r2.fit(xn, yr, eval_set=[(xn, yr)])
    assert r.evals_result() == r2.evals_result()
    np.testing.assert_array_equal(r.predict(xs), r2.predict(xn))
    # clone keeps it; pickle keeps it
    assert clone(ours).missing == -999.0
    np.testing.assert_array_equal(pickle.loads(pickle.dumps(ours)).predict_proba(xs),
                                  ours.predict_proba(xs))


def test_missing_with_multiclass_and_bad_values():
    xs, xn, _ = _int_data()
    y = np.digitize(np.nan_to_num(xn[:, 0]), [3, 6])
    a = FerrisBoostClassifier(n_estimators=5, max_depth=3, n_jobs=1, missing=-999.0).fit(xs, y)
    b = FerrisBoostClassifier(n_estimators=5, max_depth=3, n_jobs=1).fit(xn, y)
    np.testing.assert_array_equal(a.predict_proba(xs), b.predict_proba(xn))
    with pytest.raises(ValueError, match="missing"):
        FerrisBoostClassifier(missing=np.inf).fit(xs, y)


def test_apply_matches_xgboost_on_the_same_model(tmp_path):
    xgb = pytest.importorskip("xgboost")
    xs, xn, y = _int_data(seed=2)
    kw = {"n_estimators": 7, "max_depth": 4, "learning_rate": 0.3}
    ours = FerrisBoostClassifier(n_jobs=1, missing=-999.0, **kw).fit(xs, y)
    ref = xgb.XGBClassifier(tree_method="hist", n_jobs=1, missing=-999.0, **kw).fit(xs, y)
    leaves = ours.apply(xs)
    expected = ref.apply(xs)
    assert leaves.dtype == expected.dtype == np.float32
    assert leaves.shape == expected.shape == (len(y), 7)
    np.testing.assert_array_equal(leaves, expected)
    # XGBoost's pred_leaf only accepts begin == 0; a nonzero begin here is the
    # matching columns of the full result.
    np.testing.assert_array_equal(ours.apply(xs, iteration_range=(0, 5)),
                                  ref.apply(xs, iteration_range=(0, 5)))
    np.testing.assert_array_equal(ours.apply(xs, iteration_range=(2, 5)), leaves[:, 2:5])
    # the same saved model through XGBoost's own estimator
    path = tmp_path / "m.json"
    ours.save_model(path)
    again = xgb.XGBClassifier()
    again.load_model(str(path))
    again.set_params(missing=-999.0)
    np.testing.assert_array_equal(again.apply(xs), leaves)
    # regressor and multiclass shapes
    yr = np.nan_to_num(xn[:, 0]) + np.nan_to_num(xn[:, 1])
    reg = FerrisBoostRegressor(n_jobs=1, missing=-999.0, **kw).fit(xs, yr)
    xreg = xgb.XGBRegressor(tree_method="hist", n_jobs=1, missing=-999.0, **kw).fit(xs, yr)
    np.testing.assert_array_equal(reg.apply(xs), xreg.apply(xs))
    ym = np.digitize(np.nan_to_num(xn[:, 0]), [3, 6])
    mc = FerrisBoostClassifier(n_jobs=1, missing=-999.0, **kw).fit(xs, ym)
    xmc = xgb.XGBClassifier(tree_method="hist", n_jobs=1, missing=-999.0, **kw).fit(xs, ym)
    assert mc.apply(xs).shape == xmc.apply(xs).shape
    np.testing.assert_array_equal(mc.apply(xs), xmc.apply(xs))


def test_apply_uses_best_iteration_with_early_stopping():
    xgb = pytest.importorskip("xgboost")
    rng = np.random.default_rng(1)
    X = rng.integers(0, 9, size=(800, 6)).astype(np.float32)  # lossless binning
    y = (X[:, 0] - X[:, 1] + rng.normal(0, 2.0, 800) > 0).astype(int)
    kw = {"n_estimators": 60, "max_depth": 3, "learning_rate": 0.4, "early_stopping_rounds": 3}
    ours = FerrisBoostClassifier(n_jobs=1, **kw).fit(X[:500], y[:500], eval_set=[(X[500:], y[500:])])
    best = ours.best_iteration
    assert ours.apply(X).shape == (800, best + 1)
    assert ours.apply(X, iteration_range=(0, 0)).shape == (800, best + 1)
    assert ours.apply(X, iteration_range=(0, 2)).shape == (800, 2)
    ref = xgb.XGBClassifier(tree_method="hist", n_jobs=1, **kw)
    ref.fit(X[:500], y[:500], eval_set=[(X[500:], y[500:])], verbose=False)
    assert ref.best_iteration == best
    np.testing.assert_array_equal(ours.apply(X), ref.apply(X))


def test_apply_requires_a_fitted_estimator_and_matching_features():
    from sklearn.exceptions import NotFittedError

    with pytest.raises(NotFittedError):
        small_clf().apply(np.zeros((2, 3)))
    X, y = make_classification(n_samples=200, n_features=4, random_state=0)
    clf = small_clf().fit(X, y)
    with pytest.raises(ValueError, match="features"):
        clf.apply(X[:, :3])


def _cat_frame(n=1200, seed=0, missing_frac=0.1):
    pd = pytest.importorskip("pandas")
    rng = np.random.default_rng(seed)
    colour = rng.choice(["red", "green", "blue", "pink", "grey", "teal", "gold"], n)
    size = rng.choice(["S", "M", "L"], n)
    x1 = rng.normal(size=n)
    effect = {"red": 1.0, "green": -1.0, "blue": 0.5, "pink": 0.0, "grey": -0.5, "teal": 1.5,
              "gold": -1.5}
    score = np.array([effect[c] for c in colour]) + 0.8 * (size == "L") + 0.7 * x1
    y = (score + rng.normal(0, 0.5, n) > 0).astype(int)
    df = pd.DataFrame({"colour": pd.Categorical(colour), "size": pd.Categorical(size), "x1": x1})
    df.loc[rng.random(n) < missing_frac, "colour"] = np.nan
    return df, y, score


def test_categorical_dataframe_end_to_end():
    df, y, score = _cat_frame()
    kw = {"n_estimators": 20, "max_depth": 4, "n_jobs": 1}
    clf = FerrisBoostClassifier(enable_categorical=True, **kw).fit(df, y)
    assert list(clf.feature_names_in_) == ["colour", "size", "x1"]
    assert clf.n_features_in_ == 3
    proba = clf.predict_proba(df)
    assert proba.shape == (len(y), 2) and np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    assert clf.score(df, y) > 0.85
    # the codes are what was trained on: same as an explicit code matrix + feature_types
    codes = np.stack([df["colour"].cat.codes.where(df["colour"].notna(), np.nan).to_numpy(np.float32),
                      df["size"].cat.codes.to_numpy(np.float32), df["x1"].to_numpy(np.float32)], axis=1)
    manual = FerrisBoostClassifier(feature_types=["c", "c", "q"], enable_categorical=True,
                                   **kw).fit(codes, y)
    np.testing.assert_array_equal(manual.predict_proba(codes), proba)
    # prediction re-codes by value even when the frame's category order differs
    shuffled = df.copy()
    shuffled["colour"] = shuffled["colour"].cat.reorder_categories(
        list(reversed(df["colour"].cat.categories)))
    np.testing.assert_array_equal(clf.predict_proba(shuffled), proba)
    # a category never seen in training does not crash
    novel = df.copy()
    novel["colour"] = novel["colour"].cat.add_categories(["purple"])
    novel.loc[novel.index[:5], "colour"] = "purple"
    assert clf.predict_proba(novel).shape == (len(y), 2)
    # apply, regressor, early stopping with a validation frame
    assert clf.apply(df).shape == (len(y), 20)
    reg = FerrisBoostRegressor(enable_categorical=True, **kw).fit(df, score)
    assert reg.predict(df).shape == (len(y),)
    es = FerrisBoostClassifier(enable_categorical=True, early_stopping_rounds=3,
                               **{**kw, "n_estimators": 80})
    es.fit(df.iloc[:900], y[:900], eval_set=[(df.iloc[900:], y[900:])])
    assert es.best_iteration is not None
    assert es.apply(df).shape[1] == es.best_iteration + 1


def test_categorical_dataframe_matches_xgboost():
    xgb = pytest.importorskip("xgboost")
    df, y, _ = _cat_frame(seed=3)
    kw = {"n_estimators": 10, "max_depth": 4, "learning_rate": 0.3, "max_bin": 255}
    # Lossless binning on the numeric column: both libraries must agree bit for bit.
    df = df.copy()
    df["x1"] = np.round(df["x1"] * 4).astype(np.float32)
    ours = FerrisBoostClassifier(n_jobs=1, enable_categorical=True, **kw).fit(df, y)
    ref = xgb.XGBClassifier(tree_method="hist", n_jobs=1, enable_categorical=True, **kw).fit(df, y)
    np.testing.assert_array_equal(ours.predict_proba(df), ref.predict_proba(df))
    np.testing.assert_array_equal(ours.apply(df), ref.apply(df))


def test_categorical_constructor_params_are_forwarded():
    df, y, _ = _cat_frame(seed=4)
    kw = {"n_estimators": 6, "max_depth": 4, "n_jobs": 1, "enable_categorical": True}
    default = FerrisBoostClassifier(**kw).fit(df, y)
    onehot = FerrisBoostClassifier(max_cat_to_onehot=8, **kw).fit(df, y)
    narrow = FerrisBoostClassifier(max_cat_threshold=1, **kw).fit(df, y)
    assert not np.array_equal(default.predict_proba(df), onehot.predict_proba(df))
    assert not np.array_equal(default.predict_proba(df), narrow.predict_proba(df))
    assert onehot.get_params()["max_cat_to_onehot"] == 8
    # feature_types as a constructor param for plain arrays
    X = df["colour"].cat.codes.where(df["colour"].notna(), np.nan).to_numpy(np.float32)[:, None]
    X = np.hstack([X, df[["x1"]].to_numpy(np.float32)])
    a = FerrisBoostClassifier(feature_types=["c", "q"], enable_categorical=True, n_estimators=5,
                              n_jobs=1).fit(X, y)
    b = FerrisBoostClassifier(n_estimators=5, n_jobs=1).fit(X, y)
    assert not np.array_equal(a.predict_proba(X), b.predict_proba(X))


def test_category_columns_need_enable_categorical():
    df, y, _ = _cat_frame()
    with pytest.raises(ValueError, match="enable_categorical"):
        small_clf().fit(df, y)


def test_categorical_pickle_clone_and_loaded_model(tmp_path):
    df, y, _ = _cat_frame(seed=5)
    kw = {"n_estimators": 8, "max_depth": 3, "n_jobs": 1, "enable_categorical": True}
    clf = FerrisBoostClassifier(**kw).fit(df, y)
    again = pickle.loads(pickle.dumps(clf))
    np.testing.assert_array_equal(again.predict_proba(df), clf.predict_proba(df))
    assert clone(clf).get_params()["enable_categorical"] is True
    path = tmp_path / "c.json"
    clf.save_model(path)
    loaded = FerrisBoostClassifier(**kw).load_model(path)
    # a loaded model has no stored category table: the frame's own codes are used
    np.testing.assert_array_equal(loaded.predict_proba(df), clf.predict_proba(df))


def test_gridsearch_with_categorical_dataframe():
    df, y, _ = _cat_frame(n=600, seed=6)
    grid = GridSearchCV(
        FerrisBoostClassifier(n_estimators=8, n_jobs=1, enable_categorical=True),
        {"max_depth": [2, 4], "max_cat_to_onehot": [1, 8]},
        cv=3,
    )
    grid.fit(df, y)
    assert grid.best_score_ > 0.7
    assert set(grid.best_params_) == {"max_depth", "max_cat_to_onehot"}
    assert grid.predict(df).shape == (len(y),)
    parallel = GridSearchCV(
        FerrisBoostClassifier(n_estimators=5, n_jobs=1, enable_categorical=True),
        {"max_depth": [2, 3]}, cv=2, n_jobs=2,
    ).fit(df, y)
    assert parallel.best_score_ > 0.7
    scores = cross_val_score(
        FerrisBoostRegressor(n_estimators=8, n_jobs=1, enable_categorical=True), df,
        np.asarray(y, dtype=float), cv=3)
    assert scores.shape == (3,)
