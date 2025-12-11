from __future__ import annotations

import pandas as pd

from TitanicML.src.config import DEFAULT_CONFIG
from TitanicML.src.preprocessing import (
    build_preprocessing_pipeline,
    split_features_target,
    validate_required_columns,
)


def test_split_features_target_basic():
    cfg = DEFAULT_CONFIG
    df = pd.DataFrame({
        cfg.columns.target_column: [0, 1, 0],
        "Age": [22, 38, 26],
        "SibSp": [1, 1, 0],
        "Parch": [0, 0, 0],
        "Fare": [7.25, 71.2833, 7.925],
        "Pclass": [3, 1, 3],
        "Sex": ["male", "female", "female"],
        "Embarked": ["S", "C", "S"],
    })

    X, y = split_features_target(df, cfg)

    assert cfg.columns.target_column not in X.columns
    assert len(X) == len(y) == 3


def test_validate_required_columns_passes():
    cfg = DEFAULT_CONFIG
    df = pd.DataFrame({
        "Age": ,
        "SibSp": [1],
        "Parch": ,
        "Fare": [7.25],
        "Pclass": [3],
        "Sex": ["male"],
        "Embarked": ["S"],
    })

    # Should not raise
    validate_required_columns(df, cfg)


def test_validate_required_columns_raises():
    cfg = DEFAULT_CONFIG
    df = pd.DataFrame({
        "Age": ,
        "SibSp": [1],
        # Missing Parch
        "Fare": [7.25],
        "Pclass": [3],
        "Sex": ["male"],
        "Embarked": ["S"],
    })

    try:
        validate_required_columns(df, cfg)
    except ValueError as e:
        assert "Missing required columns" in str(e)
    else:  # pragma: no cover - if no error is raised the test should fail
        assert False, "Expected ValueError for missing columns"


def test_build_preprocessing_pipeline_runs():
    cfg = DEFAULT_CONFIG
    preprocessor = build_preprocessing_pipeline(
        numerical_features=cfg.columns.numerical_features,
        categorical_features=cfg.columns.categorical_features,
    )

    # Fit-transform on a tiny dataframe
    df = pd.DataFrame({
        "Age": [22, 38],
        "SibSp": [1, 1],
        "Parch": [0, 0],
        "Fare": [7.25, 71.2833],
        "Pclass": [3, 1],
        "Sex": ["male", "female"],
        "Embarked": ["S", "C"],
    })

    X_processed = preprocessor.fit_transform(df)
    # Should return a 2D array with as many rows as inputs
    assert X_processed.shape == 2
