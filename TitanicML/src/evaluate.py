from __future__ import annotations

import argparse

from .config import DEFAULT_CONFIG, Config
from .data_loading import load_processed_splits
from .model import evaluate_predictions
from .utils import load_model


def evaluate(cfg: Config | None = None) -> None:
    """Evaluate the saved model on the stored validation split."""

    if cfg is None:
        cfg = DEFAULT_CONFIG

    X_train, y_train, X_valid, y_valid = load_processed_splits(cfg)

    model = load_model(cfg.paths.model_path)
    y_pred = model.predict(X_valid)
    acc, f1 = evaluate_predictions(y_valid, y_pred)

    print("Evaluation on stored validation split:")
    print(f"Accuracy: {acc:.4f}")
    print(f"F1-score: {f1:.4f}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate TitanicML model")
    return parser.parse_args()


def main() -> None:
    _ = _parse_args()
    evaluate()


if __name__ == "__main__":  # pragma: no cover
    main()
