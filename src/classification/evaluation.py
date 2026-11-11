from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.nn import functional as F

from .configs import CLASS_ORDER
from .metrics import classification_metrics, prediction_row


@dataclass(frozen=True)
class EvaluationResult:
    metrics: dict[str, object]
    predictions: list[dict[str, object]]


def evaluate_loader(
    model: torch.nn.Module,
    loader,
    device: torch.device,
) -> EvaluationResult:
    model.eval()
    all_true: list[int] = []
    all_probabilities: list[np.ndarray] = []
    prediction_rows: list[dict[str, object]] = []
    total_loss = 0.0
    total_count = 0

    with torch.no_grad():
        for batch in loader:
            targets = batch["target_images"].to(device)
            contexts = batch["context_images"].to(device)
            context_mask = batch["context_mask"].to(device)
            labels = batch["labels"].to(device)
            logits = model(
                targets,
                contexts,
                context_mask,
            )
            total_loss += F.cross_entropy(logits, labels, reduction="sum").item()
            total_count += int(labels.numel())
            probabilities = logits.softmax(dim=1).cpu().numpy()
            truth = labels.cpu().numpy()
            predictions = probabilities.argmax(axis=1)
            all_true.extend(int(value) for value in truth)
            all_probabilities.extend(probabilities)
            for row, prediction in enumerate(predictions):
                prediction_rows.append(
                    prediction_row(
                        batch["image_paths"][row], truth[row], prediction
                    )
                )
    if total_count == 0:
        raise ValueError("Evaluation loader is empty")
    true_array = np.asarray(all_true, dtype=np.int64)
    probability_array = np.asarray(all_probabilities, dtype=np.float32)
    metrics = classification_metrics(true_array, probability_array)
    metrics["loss"] = total_loss / total_count
    metrics["images"] = total_count
    return EvaluationResult(metrics=metrics, predictions=prediction_rows)

