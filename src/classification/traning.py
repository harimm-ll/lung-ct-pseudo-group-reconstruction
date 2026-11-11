from __future__ import annotations

import csv
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from .classifier import PseudoGroupContextClassifier
from .configs import CLASS_ORDER, ARCHITECTURE
from .dataset import ContextAwareCTDataset, collate_context_samples
from .device import device_audit, resolve_device
from .evaluation import evaluate_loader
from .io_utils import write_csv, write_json
from .schema import audit_splits, read_split_csv


@dataclass(frozen=True)
class TrainingConfig:
    train_csv: str
    val_csv: str
    dataset_root: str
    output_dir: str
    image_size: int = 224
    batch_size: int = 16
    learning_rate: float = 1e-4
    dropout: float = 0.1
    hidden_dim: int = 256
    epochs: int = 30
    patience: int = 8
    seed: int = 42
    num_workers: int = 0
    device: str = "auto"
    weights: str = "imagenet"
    verify_images: bool = True


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _make_loader(
    dataset: ContextAwareCTDataset,
    config: TrainingConfig,
    shuffle: bool,
) -> DataLoader:
    generator = torch.Generator().manual_seed(config.seed)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        collate_fn=collate_context_samples,
        generator=generator,
        persistent_workers=config.num_workers > 0,
    )


def _train_epoch(model, loader, optimizer, device) -> float:  # noqa: ANN001
    model.train()
    total_loss = 0.0
    total_count = 0
    for batch in loader:
        targets = batch["target_images"].to(device)
        contexts = batch["context_images"].to(device)
        context_mask = batch["context_mask"].to(device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(targets, contexts, context_mask)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()
        count = int(labels.numel())
        total_loss += float(loss.detach()) * count
        total_count += count
    return total_loss / max(total_count, 1)


def _save_checkpoint(
    path: Path,
    model: PseudoGroupContextClassifier,
    optimizer: torch.optim.Optimizer,
    config: TrainingConfig,
    epoch: int,
    validation_metrics: dict[str, object],
) -> None:
    payload = {
        "format_version": 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "model_config": {
            "dropout": config.dropout,
            "hidden_dim": config.hidden_dim,
            "num_classes": len(CLASS_ORDER),
            "image_size": config.image_size,
        },
        "training_config": asdict(config),
        "class_order": CLASS_ORDER,
        "epoch": epoch,
        "validation_metrics": validation_metrics,
        "paper_architecture": ARCHITECTURE,
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def train(config: TrainingConfig) -> dict[str, object]:
    if config.weights not in {"imagenet", "none"}:
        raise ValueError("weights must be imagenet or none")
    if config.epochs < 1 or config.patience < 1 or config.batch_size < 1:
        raise ValueError("epochs, patience and batch_size must be positive")
    if not 0.0 <= config.dropout < 1.0:
        raise ValueError("dropout must be in [0,1)")
    seed_everything(config.seed)
    device = resolve_device(config.device)
    output_dir = Path(config.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_records = read_split_csv(
        Path(config.train_csv), Path(config.dataset_root), config.verify_images
    )
    val_records = read_split_csv(
        Path(config.val_csv), Path(config.dataset_root), config.verify_images
    )
    split_audit = audit_splits({"train": train_records, "val": val_records})
    write_json(output_dir / "split_audit.json", split_audit)
    write_json(output_dir / "run_config.json", asdict(config))

    train_loader = _make_loader(
        ContextAwareCTDataset(train_records, config.image_size), config, shuffle=True
    )
    val_loader = _make_loader(
        ContextAwareCTDataset(val_records, config.image_size), config, shuffle=False
    )
    model = PseudoGroupContextClassifier(
        pretrained=config.weights == "imagenet",
        dropout=config.dropout,
        hidden_dim=config.hidden_dim,
        num_classes=len(CLASS_ORDER),
    ).to(device)
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config.learning_rate,
    )
    model_audit = {
        "paper_architecture": ARCHITECTURE,
        "device": device_audit(device),
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "total_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_backbone_parameter_names": list(
            model.backbone.trainable_parameter_names()
        ),
    }
    write_json(output_dir / "model_audit.json", model_audit)

    history: list[dict[str, object]] = []
    best_loss = float("inf")
    stale_epochs = 0
    checkpoint_path = output_dir / "best.pt"
    for epoch in range(1, config.epochs + 1):
        start = time.perf_counter()
        train_loss = _train_epoch(model, train_loader, optimizer, device)
        validation = evaluate_loader(model, val_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": validation.metrics["loss"],
            "val_accuracy": validation.metrics["accuracy"],
            "val_macro_f1": validation.metrics["macro_f1"],
            "val_weighted_f1": validation.metrics["weighted_f1"],
            "seconds": time.perf_counter() - start,
        }
        history.append(row)
        write_json(output_dir / "history.json", history)
        with (output_dir / "history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        print(json.dumps(row, ensure_ascii=False), flush=True)

        if float(validation.metrics["loss"]) < best_loss:
            best_loss = float(validation.metrics["loss"])
            stale_epochs = 0
            _save_checkpoint(
                checkpoint_path,
                model,
                optimizer,
                config,
                epoch,
                validation.metrics,
            )
            write_json(output_dir / "best_validation_metrics.json", validation.metrics)
            write_csv(output_dir / "best_validation_predictions.csv", validation.predictions)
        else:
            stale_epochs += 1
        if stale_epochs >= config.patience:
            break

    summary = {
        "status": "complete",
        "epochs_completed": len(history),
        "best_epoch": int(min(history, key=lambda item: item["val_loss"])["epoch"]),
        "best_val_loss": best_loss,
        "checkpoint": str(checkpoint_path),
        "test_was_read": False,
    }
    write_json(output_dir / "training_summary.json", summary)
    return summary


def load_checkpoint_model(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[PseudoGroupContextClassifier, dict[str, object]]:
    checkpoint = torch.load(
        Path(checkpoint_path).expanduser().resolve(),
        map_location=device,
        weights_only=False,
    )
    if tuple(checkpoint["class_order"]) != CLASS_ORDER:
        raise ValueError("Checkpoint class order does not match this package")
    model_config = checkpoint["model_config"]
    model = PseudoGroupContextClassifier(
        pretrained=False,
        dropout=float(model_config["dropout"]),
        hidden_dim=int(model_config["hidden_dim"]),
        num_classes=int(model_config["num_classes"]),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return model, checkpoint

