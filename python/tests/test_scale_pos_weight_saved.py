"""The trained `scale_pos_weight` is written into the saved model (XGBoost does the same)."""
import json

import numpy as np
import pytest

import ferrisboost as fb

xgb = pytest.importorskip("xgboost")


def toy(n=600, m=5, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.integers(0, 8, size=(n, m)).astype(np.float32)
    y = (x[:, 0] - x[:, 1] + rng.normal(0, 1.0, n) > 1).astype(np.float32)
    return x, y


def saved_learner(model, tmp_path, name="m"):
    path = tmp_path / f"{name}.json"
    model.save_model(str(path), quiet=True)
    return str(path), json.loads(path.read_text())["learner"]


@pytest.mark.parametrize("objective", ["binary:logistic", "reg:squarederror"])
def test_scale_pos_weight_is_saved_and_xgboost_reads_it(tmp_path, objective):
    x, y = toy()
    params = {"objective": objective, "max_depth": 3, "nthread": 1, "scale_pos_weight": 3}
    model = fb.train(params, x, label=y, num_boost_round=5, quiet=True)
    path, learner = saved_learner(model, tmp_path)
    assert learner["objective"]["reg_loss_param"]["scale_pos_weight"] == "3"

    bst = xgb.Booster(model_file=path)
    cfg = json.loads(bst.save_config())
    assert cfg["learner"]["objective"]["reg_loss_param"]["scale_pos_weight"] == "3"


def test_non_integer_value_uses_shortest_representation(tmp_path):
    x, y = toy()
    model = fb.train({"objective": "binary:logistic", "scale_pos_weight": 0.3, "nthread": 1},
                     x, label=y, num_boost_round=2, quiet=True)
    _, learner = saved_learner(model, tmp_path)
    assert learner["objective"]["reg_loss_param"]["scale_pos_weight"] == "0.3"


def test_predictions_survive_save_load_and_value_round_trips(tmp_path):
    x, y = toy()
    model = fb.train({"objective": "binary:logistic", "max_depth": 3, "nthread": 1,
                      "scale_pos_weight": 3}, x, label=y, num_boost_round=5, quiet=True)
    path, _ = saved_learner(model, tmp_path)
    before = model.predict(x)

    loaded = fb.Model.load_model(path)
    np.testing.assert_array_equal(loaded.predict(x), before)
    # Re-saving the loaded model keeps the value (and the bytes).
    path2, learner2 = saved_learner(loaded, tmp_path, "m2")
    assert learner2["objective"]["reg_loss_param"]["scale_pos_weight"] == "3"
    assert open(path, "rb").read() == open(path2, "rb").read()

    bst = xgb.Booster(model_file=path)
    np.testing.assert_allclose(bst.predict(xgb.DMatrix(x)), before, rtol=1e-6)


def test_default_still_writes_one(tmp_path):
    x, y = toy()
    model = fb.train({"objective": "binary:logistic", "nthread": 1}, x, label=y,
                     num_boost_round=2, quiet=True)
    _, learner = saved_learner(model, tmp_path)
    assert learner["objective"]["reg_loss_param"]["scale_pos_weight"] == "1"
