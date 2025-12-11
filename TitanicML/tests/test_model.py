from __future__ import annotations

import pandas as pd

from TitanicML.src.config import DEFAULT_CONFIG
from TitanicML.src.model import build_model, evaluate_predictions


def _make_tiny_dataset():
    cfg = DEFAULT_CONFIG
    df = pd.DataFrame({
        cfg.columns.target_column: [0, 1, 0, 1],
        "Age": [22, 38, 26, 35],
        "SibSp": [1, 1, 0, 1],
        "Parch": [0, 0, 0, 0],
        "Fare": [7.25, 71.2833, 7.925, 53.1],
        "Pclass": [3, 1, 3, 1],
        "Sex": ["male", "female", "female", "female"],
        "Embarked": ["S", "C", "S", "S"],
    })
    X = df.drop(columns=[cfg.columns.target_column])
    y = df[cfg.columns.target_column]
    return X, y


def test_build_model_fits_and_predicts():
    X, y = _make_tiny_dataset()
    model = build_model(DEFAULT_CONFIG)
    model.fit(X, y)
    preds = model.predict(X)
    assert len(preds) == len(y)


def test_evaluate_predictions_values():
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 1, 1]
    acc, f1 = evaluate_predictions(y_true, y_pred)
    assert 0.0 <= acc <= 1.0
    assert 0.0 <= f1 <= 1.0
