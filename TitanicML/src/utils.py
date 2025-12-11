from __future__ import annotations

import os
import pickle
from typing import Any


def save_model(model: Any, path: str) -> None:
    """Serialize a model (or sklearn pipeline) to disk using pickle."""

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)


def load_model(path: str) -> Any:
    """Load a pickled model from disk."""

    with open(path, "rb") as f:
        return pickle.load(f)


def write_text(path: str, content: str) -> None:
    """Write plain text content to a file, creating directories as needed."""

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
