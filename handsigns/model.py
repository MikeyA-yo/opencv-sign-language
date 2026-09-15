"""Download the MediaPipe Hand Landmarker bundle on first use."""

from __future__ import annotations

import urllib.request
from pathlib import Path

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

PACKAGE_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PACKAGE_ROOT / "models"
MODEL_PATH = MODEL_DIR / "hand_landmarker.task"


def ensure_model(path: Path | None = None) -> Path:
    target = path or MODEL_PATH
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.stat().st_size > 1_000_000:
        return target
    print(f"Downloading hand landmarker model to {target} ...")
    urllib.request.urlretrieve(MODEL_URL, target)
    if not target.exists() or target.stat().st_size < 1_000_000:
        raise RuntimeError(f"failed to download model from {MODEL_URL}")
    return target
