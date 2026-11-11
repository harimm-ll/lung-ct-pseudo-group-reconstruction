from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from .configs import CLASS_ORDER


def prediction_row(
    image_path: str | Path,
    true_index: int,
    predicted_index: int,
) -> dict[str, str]:
    """Return the three public fields written to prediction CSV files."""
    return {
        "filename": Path(image_path).name,
        "true_label": CLASS_ORDER[int(true_index)],
        "predicted_label": CLASS_ORDER[int(predicted_index)],
    }


def classification_metrics(
    true_indices: np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, object]:
    predicted = np.argmax(probabilities, axis=1)
    labels = list(range(len(CLASS_ORDER)))
    precision, recall, f1, support = precision_recall_fscore_support(
        true_indices,
        predicted,
        labels=labels,
        zero_division=0,
    )
    weighted_f1 = float(
        np.average(f1, weights=support) if int(support.sum()) else 0.0
    )
    try:
        auc = float(
            roc_auc_score(
                true_indices,
                probabilities,
                labels=labels,
                multi_class="ovr",
                average="macro",
            )
        )
    except ValueError:
        auc = math.nan
    return {
        "accuracy": float(accuracy_score(true_indices, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(true_indices, predicted)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "weighted_f1": weighted_f1,
        "macro_roc_auc_ovr": auc,
        "class_order": list(CLASS_ORDER),
        "confusion_matrix": confusion_matrix(
            true_indices, predicted, labels=labels
        ).tolist(),
        "per_class": {
            label: {
                "support": int(support[index]),
                "precision": float(precision[index]),
                "recall": float(recall[index]),
                "f1": float(f1[index]),
            }
            for index, label in enumerate(CLASS_ORDER)
        },
    }

