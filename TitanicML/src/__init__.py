"""TitanicML package.

Provides utilities to load data, preprocess features, train models,
run evaluations, and generate predictions on the Titanic dataset.

Modules
-------
config
    Central configuration (paths, column names, hyperparameters).
data_loading
    Functions for loading raw and processed datasets from disk.
preprocessing
    Preprocessing pipelines (imputation, encoding, scaling).
model
    Model definition and model selection helpers.
train
    Training entrypoint.
evaluate
    Evaluation entrypoint.
predict
    Prediction entrypoint.
utils
    Small helper utilities (I/O, logging).
"""

__all__ = [
    "config",
    "data_loading",
    "preprocessing",
    "model",
    "train",
    "evaluate",
    "predict",
    "utils",
]
