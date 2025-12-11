from __future__ import annotations

import argparse

from TitanicML.src import train as train_module
from TitanicML.src import evaluate as evaluate_module
from TitanicML.src import predict as predict_module


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="TitanicML command line interface")

    subparsers = parser.add_subparsers(dest="command", required=True)

    # train
    subparsers.add_parser("train", help="Train the model")

    # evaluate
    subparsers.add_parser("evaluate", help="Evaluate the model on validation data")

    # predict
    predict_parser = subparsers.add_parser("predict", help="Predict on a new CSV file")
    predict_parser.add_argument("--input", required=True, help="Path to input CSV")
    predict_parser.add_argument("--output", required=False, help="Optional path to output predictions CSV")

    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.command == "train":
        train_module.train()
    elif args.command == "evaluate":
        evaluate_module.evaluate()
    elif args.command == "predict":
        predict_module.predict(input_path=args.input, output_path=args.output)
    else:  # pragma: no cover
        parser.print_help()


if __name__ == "__main__":  # pragma: no cover
    main()
