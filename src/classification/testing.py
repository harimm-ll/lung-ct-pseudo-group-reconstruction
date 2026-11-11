from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from torch.utils.data import DataLoader

from .dataset import ContextAwareCTDataset, collate_context_samples
from .device import device_audit, resolve_device
from .evaluation import evaluate_loader
from .io_utils import write_csv, write_json
from .schema import read_split_csv
from .training import load_checkpoint_model


@dataclass(frozen=True)
class EvaluationConfig:
    test_csv: str
    dataset_root: str
    checkpoint: str
    output_dir: str
    batch_size: int = 16
    num_workers: int = 0
    device: str = "auto"
    verify_images: bool = True


def evaluate_checkpoint(config: EvaluationConfig) -> dict[str, object]:
    device = resolve_device(config.device)
    model, checkpoint = load_checkpoint_model(Path(config.checkpoint), device)
    image_size = int(checkpoint["model_config"]["image_size"])
    records = read_split_csv(
        Path(config.test_csv), Path(config.dataset_root), config.verify_images
    )
    dataset = ContextAwareCTDataset(records, image_size=image_size)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=collate_context_samples,
        persistent_workers=config.num_workers > 0,
    )
    result = evaluate_loader(model, loader, device)
    output_dir = Path(config.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "evaluation_config.json", asdict(config))
    write_json(output_dir / "device_audit.json", device_audit(device))
    write_json(output_dir / "metrics.json", result.metrics)
    write_csv(output_dir / "predictions.csv", result.predictions)
    return result.metrics

