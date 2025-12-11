from __future__ import annotations

import argparse

from sklearn.model_selection import train_test_split

from .config import DEFAULT_CONFIG, Config, ensure_directories
from .data_loading import load_raw_train, save_processed_splits
from .model import build_model, evaluate_predictions
from .preprocessing import split_features_target, validate_required_columns
from .utils import save_model, write_text


def train(cfg: Config | None = None) -> None:
    """Train a model on the Titanic training data.

    Steps
    -----
    1. Load raw training CSV.
    2. Split into train/validation.
    3. Fit preprocessing + model pipeline.
    4. Evaluate on validation set.
    5. Save model and processed splits.
    6. Write metrics to reports.
    """

    if cfg is None:
        cfg = DEFAULT_CONFIG

    ensure_directories(cfg)

    df = load_raw_train(cfg)
    validate_required_columns(df, cfg)

    X, y = split_features_target(df, cfg)

    X_train, X_valid, y_train, y_valid = train_test_split(
        X,
        y,
        test_size=cfg.model.test_size,
        random_state=cfg.model.random_state,
        stratify=y,
    )

    pipeline = build_model(cfg)
    pipeline.fit(X_train, y_train)

    # Evaluate
    y_pred = pipeline.predict(X_valid)
    acc, f1 = evaluate_predictions(y_valid, y_pred)

    # Persist artifacts
    save_model(pipeline, cfg.paths.model_path)
    save_processed_splits(X_train, y_train, X_valid, y_valid, cfg)

    metrics_text = f"Accuracy: {acc:.4f}\nF1-score: {f1:.4f}\n"
    write_text(cfg.paths.metrics_path, metrics_text)

    print("Training complete.")
    print(metrics_text)
    print(f"Model saved to: {cfg.paths.model_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train TitanicML model")
    return parser.parse_args()


def main() -> None:
    _ = _parse_args()
    train()


if __name__ == "__main__":  # pragma: no cover
    main()
