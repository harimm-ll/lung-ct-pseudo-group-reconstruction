from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image

from .configs import CLASS_ORDER
from .metrics import prediction_row
from .schema import ImageRecord, audit_splits, read_split_csv


@dataclass(frozen=True)
class KerasGatedConfig:
    train_csv: str
    val_csv: str
    test_csv: str
    dataset_root: str
    output_dir: str
    imagenet_weights: str
    image_size: int = 224
    batch_size: int = 16
    backbone_learning_rate: float = 1e-5
    head_learning_rate: float = 1e-4
    dropout: float = 0.1
    hidden_dim: int = 256
    epochs: int = 30
    patience: int = 8
    reduce_lr_patience: int = 3
    seed: int = 20260715
    verify_images: bool = True
    skip_full_split_audit: bool = False
    reuse_existing_checkpoint: bool = False
    class_weights: tuple[float, ...] | None = None


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=True) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f"Cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _seed_everything(tf: Any, seed: int) -> None:
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)


def _load_image(record: ImageRecord, image_size: int, preprocess_input: Any) -> np.ndarray:
    with Image.open(record.image_path) as image:
        image = image.convert("RGB").resize((image_size, image_size))
        array = np.asarray(image, dtype=np.float32)
    return preprocess_input(array)


def _group_records(records: list[ImageRecord]) -> list[list[ImageRecord]]:
    groups: dict[str, list[ImageRecord]] = defaultdict(list)
    for record in records:
        groups[record.group_id].append(record)
    return list(groups.values())


def _make_group_packs(
    records: list[ImageRecord],
    image_budget: int,
    *,
    shuffle: bool,
    rng: random.Random,
) -> list[list[list[ImageRecord]]]:
   
    groups = _group_records(records)
    if shuffle:
        rng.shuffle(groups)
    packs: list[list[list[ImageRecord]]] = []
    current: list[list[ImageRecord]] = []
    current_size = 0
    for group in groups:
        if current and current_size + len(group) > image_budget:
            packs.append(current)
            current = []
            current_size = 0
        current.append(group)
        current_size += len(group)
    if current:
        packs.append(current)
    return packs


def _load_pack(
    groups: list[list[ImageRecord]],
    image_size: int,
    preprocess_input: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[ImageRecord]]:
    records: list[ImageRecord] = []
    group_indices: list[int] = []
    for group_index, group in enumerate(groups):
        records.extend(group)
        group_indices.extend([group_index] * len(group))
    images = np.stack(
        [_load_image(record, image_size, preprocess_input) for record in records]
    )
    labels = np.asarray([record.label_index for record in records], dtype=np.int64)
    return images, np.asarray(group_indices, dtype=np.int32), labels, records


def _set_conv5_block3_trainability(base: Any) -> list[str]:
    base.trainable = True
    trainable: list[str] = []
    for layer in base.layers:
        is_batch_norm = "BatchNormalization" in layer.__class__.__name__
        layer.trainable = layer.name.startswith("conv5_block3") and not is_batch_norm
        if layer.trainable and layer.weights:
            trainable.append(layer.name)
    return trainable


def _build_backbone(tf: Any, config: KerasGatedConfig) -> tuple[Any, list[str]]:
    #imagenet基础上只训练5_3
    base = tf.keras.applications.resnet50.ResNet50(
        include_top=False,
        weights=str(Path(config.imagenet_weights).expanduser().resolve()),
        input_shape=(config.image_size, config.image_size, 3),
        pooling="avg",
    )
    return base, _set_conv5_block3_trainability(base)


def _build_model(tf: Any, backbone: Any, config: KerasGatedConfig) -> Any:
    class KerasGatedContextClassifier(tf.keras.Model):
        #拼接逻辑

        def __init__(self) -> None:
            super().__init__(name="keras_pseudo_group_gated_context")
            self.backbone = backbone
            self.gate = tf.keras.layers.Dense(1, name="scalar_gate")
            self.hidden = tf.keras.layers.Dense(
                config.hidden_dim, activation="relu", name="classifier_hidden"
            )
            self.dropout = tf.keras.layers.Dropout(config.dropout, name="classifier_dropout")
            self.output_layer = tf.keras.layers.Dense(
                len(CLASS_ORDER), name="classifier_logits"
            )

        def call(
            self,
            inputs: tuple[Any, Any],
            training: bool = False,
            return_gate: bool = False,
        ) -> Any:
            images, group_indices = inputs
           
            target_features = self.backbone(images, training=False)
            group_indices = tf.cast(group_indices, tf.int32)
            group_count = tf.reduce_max(group_indices) + 1
            sums = tf.math.unsorted_segment_sum(
                target_features, group_indices, group_count
            )
            counts = tf.math.unsorted_segment_sum(
                tf.ones_like(group_indices, dtype=target_features.dtype),
                group_indices,
                group_count,
            )
            other_sums = tf.gather(sums, group_indices) - target_features
            other_counts = tf.gather(counts, group_indices) - 1.0
            context_features = tf.math.divide_no_nan(
                other_sums, tf.expand_dims(other_counts, axis=1)
            )
            # 单一组
            gate_input = tf.concat((target_features, context_features), axis=1)
            alpha = tf.sigmoid(self.gate(gate_input))
            gated_context = alpha * context_features
            fused = tf.concat((target_features, gated_context), axis=1)
            hidden = self.hidden(fused)
            hidden = self.dropout(hidden, training=training)
            logits = self.output_layer(hidden)
            return (logits, alpha) if return_gate else logits

    return KerasGatedContextClassifier()


def _binary_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    positive = y_true.astype(bool)
    n_pos = int(positive.sum())
    n_neg = int((~positive).sum())
    if n_pos == 0 or n_neg == 0:
        return math.nan
    order = np.argsort(scores, kind="mergesort")
    sorted_scores = scores[order]
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(scores):
        end = start + 1
        while end < len(scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    rank_sum = float(ranks[positive].sum())
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, object]:
    predictions = probabilities.argmax(axis=1)
    confusion = np.zeros((len(CLASS_ORDER), len(CLASS_ORDER)), dtype=np.int64)
    for truth, prediction in zip(labels, predictions):
        confusion[int(truth), int(prediction)] += 1
    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    recall_values: list[float] = []
    auc_values: list[float] = []
    for index, label in enumerate(CLASS_ORDER):
        tp = int(confusion[index, index])
        support = int(confusion[index].sum())
        predicted_count = int(confusion[:, index].sum())
        precision = tp / predicted_count if predicted_count else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        auc = _binary_auc(labels == index, probabilities[:, index])
        per_class[label] = {
            "support": support,
            "correct": tp,
            "accuracy": recall,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "roc_auc_ovr": auc,
        }
        f1_values.append(f1)
        recall_values.append(recall)
        auc_values.append(auc)
    return {
        "accuracy": float(np.mean(labels == predictions)),
        "balanced_accuracy": float(np.mean(recall_values)),
        "macro_f1": float(np.mean(f1_values)),
        "macro_roc_auc_ovr": float(np.nanmean(auc_values)),
        "class_order": list(CLASS_ORDER),
        "confusion_matrix": confusion.tolist(),
        "per_class": per_class,
    }


def _evaluate(
    tf: Any,
    model: Any,
    records: list[ImageRecord],
    config: KerasGatedConfig,
) -> tuple[dict[str, object], list[dict[str, object]]]:
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction="sum"
    )
    preprocess_input = tf.keras.applications.resnet50.preprocess_input
    all_labels: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    total_loss = 0.0
    total_count = 0
    packs = _make_group_packs(
        records, config.batch_size, shuffle=False, rng=random.Random(config.seed)
    )
    for groups in packs:
        images, group_indices, labels, pack_records = _load_pack(
            groups, config.image_size, preprocess_input
        )
        logits, _ = model((images, group_indices), training=False, return_gate=True)
        total_loss += float(loss_fn(labels, logits).numpy())
        probabilities = tf.nn.softmax(logits, axis=1).numpy()
        all_labels.append(labels)
        all_probabilities.append(probabilities)
        for record, truth, probability in zip(pack_records, labels, probabilities):
            prediction = int(np.argmax(probability))
            rows.append(prediction_row(record.image_path, truth, prediction))
        total_count += len(labels)
    labels_array = np.concatenate(all_labels)
    probability_array = np.concatenate(all_probabilities)
    metrics = _metrics(labels_array, probability_array)
    metrics["loss"] = total_loss / total_count
    metrics["images"] = total_count
    return metrics, rows


def train(config: KerasGatedConfig) -> dict[str, object]:
    import tensorflow as tf

    if config.batch_size < 1 or config.epochs < 1 or config.patience < 1:
        raise ValueError("batch_size, epochs, and patience must be positive")
    if not 0.0 <= config.dropout < 1.0:
        raise ValueError("dropout must be in [0,1)")
    if config.class_weights is not None:
        if len(config.class_weights) != len(CLASS_ORDER):
            raise ValueError(
                f"class_weights must contain {len(CLASS_ORDER)} values in "
                f"CLASS_ORDER={CLASS_ORDER} order"
            )
        if any(weight <= 0 for weight in config.class_weights):
            raise ValueError("class_weights must all be positive")
    if not Path(config.imagenet_weights).expanduser().is_file():
        raise FileNotFoundError(config.imagenet_weights)

    _seed_everything(tf, config.seed)
    output_dir = Path(config.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    root = Path(config.dataset_root)
    best_path = output_dir / "best.weights.h5"

    if config.reuse_existing_checkpoint:
        if not best_path.is_file():
            raise FileNotFoundError(
                f"--reuse-existing-checkpoint requested, but no checkpoint exists: {best_path}"
            )
        backbone, _ = _build_backbone(tf, config)
        model = _build_model(tf, backbone, config)
        dummy_images = tf.zeros((2, config.image_size, config.image_size, 3))
        model((dummy_images, tf.constant([0, 0], dtype=tf.int32)), training=False)
        model.load_weights(str(best_path))
        test_records = read_split_csv(
            Path(config.test_csv), root, config.verify_images
        )
        test_metrics, test_rows = _evaluate(tf, model, test_records, config)
        reuse_summary = {
            "status": "complete",
            "training_skipped": True,
            "checkpoint_reused": str(best_path),
            "cross_split_audit_performed": False,
            "test_metrics": test_metrics,
        }
        _write_json(
            output_dir / "checkpoint_reuse_config.json", asdict(config)
        )
        _write_json(
            output_dir / "reused_checkpoint_test_metrics.json", reuse_summary
        )
        _write_csv(
            output_dir / "reused_checkpoint_test_predictions.csv", test_rows
        )
        print(json.dumps(reuse_summary, indent=2, ensure_ascii=False), flush=True)
        return reuse_summary

    train_records = read_split_csv(Path(config.train_csv), root, config.verify_images)
    val_records = read_split_csv(Path(config.val_csv), root, config.verify_images)
    _write_json(output_dir / "run_config.json", asdict(config))
    if config.skip_full_split_audit:
        _write_json(
            output_dir / "split_audit_skipped.json",
            {
                "performed": False,
                "scope": "train/validation",
                "warning": (
                    "Train/validation group and image independence was not checked. "
                    "Validation metrics may contain leakage."
                ),
            },
        )
    else:
        split_audit = audit_splits({"train": train_records, "val": val_records})
        _write_json(output_dir / "split_audit.json", split_audit)

    backbone, trainable_backbone_layers = _build_backbone(tf, config)
    model = _build_model(tf, backbone, config)
    
    dummy_images = tf.zeros((2, config.image_size, config.image_size, 3))
    model((dummy_images, tf.constant([0, 0], dtype=tf.int32)), training=False)

    backbone_variables = list(backbone.trainable_variables)
    backbone_ids = {id(variable) for variable in backbone_variables}
    head_variables = [
        variable for variable in model.trainable_variables if id(variable) not in backbone_ids
    ]
    backbone_optimizer = tf.keras.optimizers.Adam(config.backbone_learning_rate)
    head_optimizer = tf.keras.optimizers.Adam(config.head_learning_rate)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True,
        reduction="none",
    )
    class_weights = (
        tf.constant(config.class_weights, dtype=tf.float32)
        if config.class_weights is not None
        else None
    )
    preprocess_input = tf.keras.applications.resnet50.preprocess_input
    model_audit = {
        "framework": f"tensorflow {tf.__version__}",
        "trainable_scope": "conv5_block3 + scalar_gate + classifier",
        "trainable_backbone_layers": trainable_backbone_layers,
        "backbone_trainable_parameters": int(
            sum(np.prod(variable.shape) for variable in backbone_variables)
        ),
        "head_trainable_parameters": int(
            sum(np.prod(variable.shape) for variable in head_variables)
        ),
        "context": "same-split leave-one-out pseudo-group feature mean",
        "singleton_context": "zero vector via divide_no_nan",
        "availability_indicator_m_i": False,
        "gate": "alpha_i=sigmoid(W_g[z_i||g_i]+b_g)",
        "fusion": "[z_i||alpha_i*g_i]",
        "head": f"Dense({config.hidden_dim},relu)-Dropout({config.dropout})-Dense(3 logits)",
        "loss": (
            "class-weighted sparse categorical cross entropy"
            if class_weights is not None
            else "ordinary sparse categorical cross entropy"
        ),
        "class_order": list(CLASS_ORDER),
        "class_weights": (
            list(config.class_weights) if config.class_weights is not None else None
        ),
        "checkpoint_monitor": "validation loss",
    }
    _write_json(output_dir / "model_audit.json", model_audit)

    history: list[dict[str, object]] = []
    best_loss = float("inf")
    stale_epochs = 0
    lr_stale_epochs = 0
    rng = random.Random(config.seed)
    for epoch in range(1, config.epochs + 1):
        start = time.perf_counter()
        train_loss_sum = 0.0
        train_count = 0
        packs = _make_group_packs(
            train_records, config.batch_size, shuffle=True, rng=rng
        )
        for groups in packs:
            images, group_indices, labels, _ = _load_pack(
                groups, config.image_size, preprocess_input
            )
            with tf.GradientTape() as tape:
                logits = model((images, group_indices), training=True)
                per_image_loss = loss_fn(labels, logits)
                if class_weights is not None:
                    sample_weights = tf.gather(class_weights, labels)
                    per_image_loss = per_image_loss * sample_weights
                loss = tf.reduce_mean(per_image_loss)
            variables = backbone_variables + head_variables
            gradients = tape.gradient(loss, variables)
            split_at = len(backbone_variables)
            backbone_pairs = [
                (gradient, variable)
                for gradient, variable in zip(
                    gradients[:split_at], backbone_variables
                )
                if gradient is not None
            ]
            head_pairs = [
                (gradient, variable)
                for gradient, variable in zip(gradients[split_at:], head_variables)
                if gradient is not None
            ]
            backbone_optimizer.apply_gradients(backbone_pairs)
            head_optimizer.apply_gradients(head_pairs)
            train_loss_sum += float(tf.reduce_sum(per_image_loss).numpy())
            train_count += len(labels)

        validation_metrics, validation_rows = _evaluate(
            tf, model, val_records, config
        )
        row = {
            "epoch": epoch,
            "train_loss": train_loss_sum / train_count,
            "val_loss": validation_metrics["loss"],
            "val_accuracy": validation_metrics["accuracy"],
            "val_macro_f1": validation_metrics["macro_f1"],
            "val_macro_roc_auc_ovr": validation_metrics["macro_roc_auc_ovr"],
            "backbone_learning_rate": float(
                backbone_optimizer.learning_rate.numpy()
            ),
            "head_learning_rate": float(head_optimizer.learning_rate.numpy()),
            "seconds": time.perf_counter() - start,
        }
        history.append(row)
        _write_json(output_dir / "history.json", history)
        _write_csv(output_dir / "history.csv", history)
        print(json.dumps(row, ensure_ascii=False), flush=True)

        current_loss = float(validation_metrics["loss"])
        if current_loss < best_loss:
            best_loss = current_loss
            stale_epochs = 0
            lr_stale_epochs = 0
            model.save_weights(str(best_path))
            _write_json(
                output_dir / "best_validation_metrics.json", validation_metrics
            )
            _write_csv(
                output_dir / "best_validation_predictions.csv", validation_rows
            )
        else:
            stale_epochs += 1
            lr_stale_epochs += 1
            if lr_stale_epochs >= config.reduce_lr_patience:
                for optimizer in (backbone_optimizer, head_optimizer):
                    optimizer.learning_rate.assign(
                        max(float(optimizer.learning_rate.numpy()) * 0.2, 1e-7)
                    )
                lr_stale_epochs = 0
        if stale_epochs >= config.patience:
            break

    model.load_weights(str(best_path))
    
    test_records = read_split_csv(Path(config.test_csv), root, config.verify_images)
    if config.skip_full_split_audit:
        _write_json(
            output_dir / "full_split_audit_skipped.json",
            {
                "performed": False,
                "warning": (
                    "Cross-split group/image independence was not checked. "
                    "Test metrics may contain leakage."
                ),
            },
        )
    else:
        full_split_audit = audit_splits(
            {"train": train_records, "val": val_records, "test": test_records}
        )
        _write_json(output_dir / "full_split_audit.json", full_split_audit)
    test_metrics, test_rows = _evaluate(tf, model, test_records, config)
    _write_json(output_dir / "test_metrics.json", test_metrics)
    _write_csv(output_dir / "test_predictions.csv", test_rows)
    summary = {
        "status": "complete",
        "epochs_completed": len(history),
        "best_epoch": int(min(history, key=lambda item: item["val_loss"])["epoch"]),
        "best_val_loss": best_loss,
        "checkpoint": str(best_path),
        "test_metrics": test_metrics,
    }
    _write_json(output_dir / "training_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return summary


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Keras gated context model initialized from ImageNet ResNet50"
    )
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--imagenet-weights", required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help=(
            "Approximate image budget per optimization step. Complete pseudo-groups "
            "are never split, so a larger group may exceed this value."
        ),
    )
    parser.add_argument("--backbone-learning-rate", type=float, default=1e-5)
    parser.add_argument("--head-learning-rate", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--reduce-lr-patience", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument(
        "--class-weights",
        nargs=3,
        type=float,
        metavar=("BENIGN", "MALIGNANT", "NORMAL"),
        help="Optional training-loss weights in CLASS_ORDER order",
    )
    parser.add_argument(
        "--verify-images", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--skip-full-split-audit",
        action="store_true",
        help=(
            "Skip both the initial train/val and final train/val/test overlap "
            "checks. This permits training/evaluation on overlapping splits "
            "and can produce leaked metrics."
        ),
    )
    parser.add_argument(
        "--reuse-existing-checkpoint",
        action="store_true",
        help=(
            "If OUTPUT_DIR/best.weights.h5 exists, load it and evaluate test "
            "without entering the training loop. Fail if it does not exist."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.class_weights is not None:
        args.class_weights = tuple(args.class_weights)
    train(KerasGatedConfig(**vars(args)))
