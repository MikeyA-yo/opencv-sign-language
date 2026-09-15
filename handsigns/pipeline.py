"""One-shot recognize(frame) used by the API and the webcam loop."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .classifier import MIN_CONFIDENCE, SignResult, classify_hands
from .detector import HandDetector
from .draw import draw_hands, draw_sign
from .landmarks import HandLandmarks
from .smoothing import SignSmoother


class HandSignRecognizer:
    def __init__(
        self,
        model_path: Path | None = None,
        max_hands: int = 2,
        min_confidence: float = MIN_CONFIDENCE,
        video_mode: bool = True,
        smooth: bool = True,
    ) -> None:
        self.min_confidence = min_confidence
        self.detector = HandDetector(
            model_path=model_path,
            max_hands=max_hands,
            video_mode=video_mode,
        )
        self.smoother = SignSmoother() if smooth else None

    def close(self) -> None:
        self.detector.close()

    def __enter__(self) -> "HandSignRecognizer":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def recognize(self, frame_bgr: np.ndarray) -> tuple[SignResult, list[HandLandmarks]]:
        hands = self.detector.detect(frame_bgr)
        result = classify_hands(hands, min_confidence=self.min_confidence)
        if self.smoother is not None:
            result = self.smoother.update(result)
        return result, hands

    def annotate(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, SignResult]:
        result, hands = self.recognize(frame_bgr)
        drawn = draw_hands(frame_bgr, hands)
        drawn = draw_sign(drawn, result)
        return drawn, result
