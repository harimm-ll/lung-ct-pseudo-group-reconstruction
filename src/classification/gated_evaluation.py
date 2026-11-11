from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .keras_gated_training import (
    KerasGatedConfig,
    _build_model,
    _evaluate,
    _build_backbone,
    _seed_everything,
    _write_csv,
    _write_json,
)
from .schema import read_split_csv


@dataclass(frozen=True)
class KerasGatedEvaluationConfig:
    test_csv: str
    dataset_root: str
    gated_checkpoint: str
    imagenet_weights: str
    output_dir: str
    image_size: int = 224
    batch_size: int = 8
    dropout: float = 0.1
    hidden_dim: int = 256
    seed: int = 20260715
    verify_images: bool = True


def evaluate(config: KerasGatedEvaluationConfig) -> dict[str, object]:
    import tensorflow as tf

    for path in (
        config.test_csv,
        config.gated_checkpoint,
        config.imagenet_weights,
    ):
        if not Path(path).expanduser().is_file():
            raise FileNotFoundError(path)
    _seed_everything(tf, config.seed)
    output_dir = Path(config.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    model_config = KerasGatedConfig(
        train_csv="NOT_READ",
        val_csv="NOT_READ",
        test_csv=config.test_csv,
        dataset_root=config.dataset_root,
        output_dir=config.output_dir,
        imagenet_weights=config.imagenet_weights,
        image_size=config.image_size,
        batch_size=config.batch_size,
        dropout=config.dropout,
        hidden_dim=config.hidden_dim,
        seed=config.seed,
        verify_images=config.verify_images,
    )
    backbone, _ = _build_backbone(tf, model_config)
    model = _build_model(tf, backbone, model_config)
    dummy_images = tf.zeros((2, config.image_size, config.image_size, 3))
    model((dummy_images, tf.constant([0, 0], dtype=tf.int32)), training=False)
    model.load_weights(str(Path(config.gated_checkpoint).expanduser().resolve()))

    test_records = read_split_csv(
        Path(config.test_csv), Path(config.dataset_root), config.verify_images
    )
    metrics, rows = _evaluate(tf, model, test_records, model_config)
    result = {
        "status": "complete",
        "evaluation_scope": "test_only_without_cross_split_audit",
        "cross_split_audit_performed": False,
        "warning": (
            "This evaluation does not establish split independence. Do not report "
            "its metrics as leakage-free results."
        ),
        "gated_checkpoint": str(
            Path(config.gated_checkpoint).expanduser().resolve()
        ),
        "metrics": metrics,
    }
    _write_json(output_dir / "evaluation_config.json", asdict(config))
    _write_json(output_dir / "test_metrics_no_audit.json", result)
    _write_csv(output_dir / "test_predictions_no_audit.csv", rows)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate an existing Keras gated checkpoint without split audit"
    )
    parser.add_argument("--test-csv", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--gated-checkpoint", required=True)
    parser.add_argument("--imagenet-weights", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument(
        "--verify-images", action=argparse.BooleanOptionalAction, default=True
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    evaluate(KerasGatedEvaluationConfig(**vars(args)))
