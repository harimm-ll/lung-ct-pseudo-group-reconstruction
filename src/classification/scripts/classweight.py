from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from classification.keras_gated_training import main

SPLIT1_BALANCED_CLASS_WEIGHTS = (
    "3.0934959349593494",
    "0.645462256149279",
    "0.8869463869463869",
)


if __name__ == "__main__":
    main(
        [
            "--class-weights",
            *SPLIT1_BALANCED_CLASS_WEIGHTS,
            *sys.argv[1:],
        ]
    )
