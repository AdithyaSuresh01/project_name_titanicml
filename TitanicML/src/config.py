from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List, Dict, Any


# Base paths are computed relative to this file so the package works when installed
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
FIGURES_DIR = os.path.join(REPORTS_DIR, "figures")


@dataclass
class PathsConfig:
    """Configuration for file system paths used in the project."""

    base_dir: str = BASE_DIR
    data_dir: str = DATA_DIR
    raw_data_dir: str = RAW_DATA_DIR
    processed_data_dir: str = PROCESSED_DATA_DIR
    models_dir: str = MODELS_DIR
    reports_dir: str = REPORTS_DIR
    figures_dir: str = FIGURES_DIR

    train_csv: str = field(
        default_factory=lambda: os.path.join(RAW_DATA_DIR, "train.csv")
    )
    test_csv: str = field(
        default_factory=lambda: os.path.join(RAW_DATA_DIR, "test.csv")
    )
    processed_train_csv: str = field(
        default_factory=lambda: os.path.join(PROCESSED_DATA_DIR, "train_processed.csv")
    )
    processed_valid_csv: str = field(
        default_factory=lambda: os.path.join(PROCESSED_DATA_DIR, "valid_processed.csv")
    )
    model_path: str = field(
        default_factory=lambda: os.path.join(MODELS_DIR, "model.pkl")
    )
    metrics_path: str = field(
        default_factory=lambda: os.path.join(REPORTS_DIR, "metrics.txt")
    )


@dataclass
class ColumnsConfig:
    """Configuration for dataset column names."""

    id_column: str = "PassengerId"
    target_column: str = "Survived"

    numerical_features: List[str] = field(
        default_factory=lambda: ["Age", "SibSp", "Parch", "Fare"]
    )
    categorical_features: List[str] = field(
        default_factory=lambda: ["Pclass", "Sex", "Embarked"]
    )


@dataclass
class ModelConfig:
    """Configuration for model selection and hyperparameters."""

    # Options: "logistic_regression", "random_forest"
    model_type: str = "logistic_regression"

    # Hyperparameters for logistic regression
    logreg_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "C": 1.0,
            "max_iter": 1000,
            "solver": "lbfgs",
        }
    )

    # Hyperparameters for random forest
    rf_params: Dict[str, Any] = field(
        default_factory=lambda: {
            "n_estimators": 200,
            "max_depth": None,
            "random_state": 42,
        }
    )

    test_size: float = 0.2
    random_state: int = 42


@dataclass
class Config:
    """Top-level configuration object.

    In a larger project you might read this from a YAML/JSON file instead
    of hard-coding, but for clarity we keep it simple here.
    """

    paths: PathsConfig = field(default_factory=PathsConfig)
    columns: ColumnsConfig = field(default_factory=ColumnsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)


# A singleton-style default config used by the rest of the codebase
DEFAULT_CONFIG = Config()


def ensure_directories(cfg: Config | None = None) -> None:
    """Ensure that all necessary directories exist.

    Parameters
    ----------
    cfg:
        Optional configuration instance. If omitted, uses DEFAULT_CONFIG.
    """

    if cfg is None:
        cfg = DEFAULT_CONFIG

    for directory in [
        cfg.paths.data_dir,
        cfg.paths.raw_data_dir,
        cfg.paths.processed_data_dir,
        cfg.paths.models_dir,
        cfg.paths.reports_dir,
        cfg.paths.figures_dir,
    ]:
        os.makedirs(directory, exist_ok=True)
