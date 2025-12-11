# TitanicML

A small but production-style machine learning project for the classic **Titanic: Machine Learning from Disaster** problem.

This repository demonstrates a clean, testable, and reproducible structure:

- `src/` for all Python source code (package-style)
- `data/` for raw and processed datasets
- `models/` for serialized trained models
- `reports/` for evaluation metrics and figures
- `notebooks/` for exploratory work
- `tests/` for unit tests

## Features

- Config-driven training (paths, model hyperparameters, preprocessing options)
- Reusable data loading and preprocessing utilities
- Simple baseline models (Logistic Regression, Random Forest)
- CLI-style Python entrypoints for:
  - training (`python -m TitanicML.src.train` or `python TitanicML/main.py train`)
  - evaluation (`python -m TitanicML.src.evaluate`)
  - prediction on new CSV data (`python -m TitanicML.src.predict`)
- Basic unit tests for preprocessing and models

## Getting started

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate       # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Data

Place the Kaggle Titanic CSV files in `TitanicML/data/raw/` with the standard names:

- `train.csv`
- `test.csv` (optional, used mainly for prediction)

If you use different filenames or locations, you can update them in `src/config.py`.

### 4. Training

```bash
python -m TitanicML.src.train
```

This will:

- Load raw `train.csv`
- Split into train/validation
- Fit preprocessing and model
- Evaluate on the validation set
- Persist the trained pipeline under `models/model.pkl`
- Write basic metrics to `reports/metrics.txt`

### 5. Evaluation

Re-run evaluation on the saved model and processed validation split:

```bash
python -m TitanicML.src.evaluate
```

### 6. Prediction

Predict on a new CSV file containing the same columns as Kaggle's `test.csv`:

```bash
python -m TitanicML.src.predict --input TitanicML/data/raw/test.csv --output TitanicML/data/processed/predictions.csv
```

The script loads the persisted preprocessing + model pipeline and writes a CSV with `PassengerId` and `Survived` prediction.

### 7. Project layout

```text
TitanicML/
├── data
│   ├── raw
│   │   └── train.csv (not committed)
│   └── processed
├── models
│   └── model.pkl
├── notebooks
│   ├── 01_exploration.ipynb
│   ├── 02_baseline_model.ipynb
│   └── 03_feature_engineering.ipynb
├── reports
│   ├── figures
│   └── metrics.txt
├── src
│   ├── __init__.py
│   ├── config.py
│   ├── data_loading.py
│   ├── preprocessing.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── utils.py
├── tests
│   ├── __init__.py
│   ├── test_preprocessing.py
│   └── test_model.py
├── main.py
├── pyproject.toml
├── requirements.txt
├── setup.cfg
└── README.md
```

## Running tests

From the project root (`TitanicML/`):

```bash
pytest
```

This will automatically discover and run tests in the `tests/` directory.

## Notes

- The goal is clarity and reproducibility rather than maximizing Kaggle score.
- You can safely extend the configuration (`config.py`) and models (`model.py`) to try more advanced approaches.
