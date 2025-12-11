from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import DEFAULT_CONFIG, Config


def build_preprocessing_pipeline(
    numerical_features: List[str] | None = None,
    categorical_features: List[str] | None = None,
) -> ColumnTransformer:
    """Create a sklearn ColumnTransformer for Titanic preprocessing.

    - Numerical features: median imputation + standard scaling
    - Categorical features: most frequent imputation + one-hot encoding

    Parameters
    ----------
    numerical_features:
        List of numerical feature column names.
    categorical_features:
        List of categorical feature column names.

    Returns
    -------
    ColumnTransformer
        A transformer suitable for use in a sklearn Pipeline.
    """

    if numerical_features is None or categorical_features is None:
        cfg = DEFAULT_CONFIG
        if numerical_features is None:
            numerical_features = cfg.columns.numerical_features
        if categorical_features is None:
            categorical_features = cfg.columns.categorical_features

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    return preprocessor


def split_features_target(
    df: pd.DataFrame, cfg: Config | None = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """Split a Titanic dataframe into features and target.

    Parameters
    ----------
    df:
        Input dataframe containing the target column.
    cfg:
        Optional configuration instance.

    Returns
    -------
    X, y
        Features dataframe and target series.
    """

    if cfg is None:
        cfg = DEFAULT_CONFIG

    target_col = cfg.columns.target_column
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataframe")

    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y


def validate_required_columns(df: pd.DataFrame, cfg: Config | None = None) -> None:
    """Ensure that all configured features exist in the dataframe.

    Raises a ValueError if any required columns are missing.
    """

    if cfg is None:
        cfg = DEFAULT_CONFIG

    required = set(cfg.columns.numerical_features + cfg.columns.categorical_features)
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def infer_feature_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Infer numerical and categorical column lists from a DataFrame.

    This is a convenience for notebooks and experiments and is not used
    by the main training pipeline (which relies on config-based columns).
    """

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]
    return num_cols, cat_cols
