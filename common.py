"""Shared helpers used by the training scripts and the demo.

Holds the stateless pieces that would otherwise be duplicated: model-path
resolution (used by ``lr_train.py``, ``nn_train.py`` and ``demo.py``), and
embedding loading plus evaluation-metric computation and reporting (used by
``lr_train.py`` and ``nn_train.py``).
"""

import numpy as np
import os

# Globals
EMBEDDINGS_PATH = os.path.join(os.path.dirname(__file__), "embeddings")
MODELS_PATH = os.path.join(os.path.dirname(__file__), "models")


def load_embeddings() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the processed embedding arrays from ``EMBEDDINGS_PATH``.

    The arrays are returned exactly as stored (features as rows, examples as
    columns for ``X``; shape ``(1, m)`` for ``Y``). Callers apply any transpose
    or flatten their model expects.

    Returns:
        tuple: ``(X_train, Y_train, X_test, Y_test)`` as numpy arrays.
    """
    X_train = np.load(os.path.join(EMBEDDINGS_PATH, "X_train.npy"))
    Y_train = np.load(os.path.join(EMBEDDINGS_PATH, "Y_train.npy"))
    X_test = np.load(os.path.join(EMBEDDINGS_PATH, "X_test.npy"))
    Y_test = np.load(os.path.join(EMBEDDINGS_PATH, "Y_test.npy"))

    return X_train, Y_train, X_test, Y_test


def resolve_model_path(
    file_path: str, expected_suffix: str | None = None
) -> str | None:
    """
    Validate that a model file exists (and has the expected type) before loading.

    Args:
        file_path (str): Path to the model file.
        expected_suffix (str | None): If given, the file extension the calling
            script can load (e.g. ``".pkl"`` for ``lr_train.py`` or ``".keras"``
            for ``nn_train.py``). A mismatch is rejected before any load attempt.

    Returns:
        str | None: ``file_path`` if it exists and matches ``expected_suffix``,
            otherwise ``None`` after printing an explanatory message.
    """
    if not os.path.exists(file_path):
        print(f"[✗] Model file not found: {file_path}")
        return None

    if expected_suffix and os.path.splitext(file_path)[1] != expected_suffix:
        print(f"[✗] {file_path} does not look like a {expected_suffix} model file. ")
        return None

    return file_path


def f1_score(recall: float, precision: float) -> float:
    """
    Compute the F1 score from recall and precision.

    Args:
        recall (float): Recall for the injection class.
        precision (float): Precision for the injection class.

    Returns:
        float: The F1 score, or 0.0 if recall and precision are both 0.
    """
    if (recall + precision) == 0:
        return 0.0

    return 2 * (recall * precision / (recall + precision))


def report_metrics(split: str, recall: float, f1: float, precision: float) -> None:
    """
    Print evaluation metrics for the prompt injection class in a fixed format.

    Recall is the primary metric and F1 the secondary metric; precision is
    included for context.

    Args:
        split (str): Name of the data split being reported (e.g. "Test").
        recall (float): Recall for the injection class.
        f1 (float): F1 score for the injection class.
        precision (float): Precision for the injection class.
    """
    print(f"{split} recall (injection):    {recall:.4f}")
    print(f"{split} F1 (injection):        {f1:.4f}")
    print(f"{split} precision (injection): {precision:.4f}")
