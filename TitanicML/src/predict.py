from __future__ import annotations

import argparse
from typing import Optional

import pandas as pd

from .config import DEFAULT_CONFIG, Config, ensure_directories
from .preprocessing import validate_required_columns
from .utils import load_model


def predict(
    input_path: str,
    output_path: Optional[str] = None,
    cfg: Config | None = None,
) -> pd.DataFrame:
    """Generate predictions for a new Titanic-style CSV.

    Parameters
    ----------
    input_path:
        Path to input CSV containing feature columns.
    output_path:
        Optional path to write predictions as CSV. If omitted, no file is written.
    cfg:
        Optional configuration instance.

    Returns
    -------
    pd.DataFrame
        DataFrame with at least two columns: PassengerId, Survived.
    """

    if cfg is None:
        cfg = DEFAULT_CONFIG

    ensure_directories(cfg)

    df = pd.read_csv(input_path)
    validate_required_columns(df, cfg)

    model = load_model(cfg.paths.model_path)
    preds = model.predict(df)

    id_col = cfg.columns.id_column
    if id_col in df.columns:
        result = pd.DataFrame({id_col: df[id_col], cfg.columns.target_column: preds})
    else:
        # Fall back to simple index if PassengerId is not available
        result = pd.DataFrame({cfg.columns.target_column: preds})

    if output_path is not None:
        result.to_csv(output_path, index=False)
        print(f"Predictions saved to {output_path}")

    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict Titanic survival on new data")
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument("--output", required=False, help="Path to output predictions CSV")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    predict(input_path=args.input, output_path=args.output)


if __name__ == "__main__":  # pragma: no cover
    main()
