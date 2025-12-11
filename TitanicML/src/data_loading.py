from __future__ import annotations

from typing import Tuple

import pandas as pd

from .config import DEFAULT_CONFIG, Config


def load_raw_train(cfg: Config | None = None) -> pd.DataFrame:
    """Load the raw training CSV.

    Parameters
    ----------
    cfg:
        Optional configuration instance.

    Returns
    -------
    pd.DataFrame
        Raw training data.
    """

    if cfg is None:
        cfg = DEFAULT_CONFIG
    return pd.read_csv(cfg.paths.train_csv)


def load_raw_test(cfg: Config | None = None) -> pd.DataFrame:
    """Load the raw test CSV (Kaggle-style)."""

    if cfg is None:
        cfg = DEFAULT_CONFIG
    return pd.read_csv(cfg.paths.test_csv)


def save_processed_splits(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_valid: pd.DataFrame,
    y_valid: pd.Series,
    cfg: Config | None = None,
) -> None:
    """Save processed train/validation splits as CSV files.

    The target column is included so they can be conveniently reloaded
    for evaluation or further experimentation.
    """

    if cfg is None:
        cfg = DEFAULT_CONFIG

    train = X_train.copy()
    train[cfg.columns.target_column] = y_train
    valid = X_valid.copy()
    valid[cfg.columns.target_column] = y_valid

    train.to_csv(cfg.paths.processed_train_csv, index=False)
    valid.to_csv(cfg.paths.processed_valid_csv, index=False)


def load_processed_splits(cfg: Config | None = None) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load processed train/validation splits from CSV files."""

    if cfg is None:
        cfg = DEFAULT_CONFIG

    train = pd.read_csv(cfg.paths.processed_train_csv)
    valid = pd.read_csv(cfg.paths.processed_valid_csv)

    target = cfg.columns.target_column
    X_train = train.drop(columns=[target])
    y_train = train[target]
    X_valid = valid.drop(columns=[target])
    y_valid = valid[target]

    return X_train, y_train, X_valid, y_valid
