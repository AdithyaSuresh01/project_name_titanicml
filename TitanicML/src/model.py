from __future__ import annotations

from typing import Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

from .config import DEFAULT_CONFIG, Config
from .preprocessing import build_preprocessing_pipeline


def build_model(cfg: Config | None = None) -> Pipeline:
    """Create a full sklearn Pipeline: preprocessing + classifier.

    Model type and hyperparameters are taken from the configuration.
    """

    if cfg is None:
        cfg = DEFAULT_CONFIG

    preprocessor = build_preprocessing_pipeline(
        numerical_features=cfg.columns.numerical_features,
        categorical_features=cfg.columns.categorical_features,
    )

    if cfg.model.model_type == "logistic_regression":
        clf = LogisticRegression(**cfg.model.logreg_params)
    elif cfg.model.model_type == "random_forest":
        clf = RandomForestClassifier(**cfg.model.rf_params)
    else:
        raise ValueError(f"Unknown model_type: {cfg.model.model_type}")

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", clf),
        ]
    )
    return pipeline


def evaluate_predictions(y_true, y_pred) -> Tuple[float, float]:
    """Compute basic classification metrics.

    Returns
    -------
    accuracy, f1
    """

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return acc, f1
