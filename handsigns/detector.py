"""OpenCV frame in → 21 MediaPipe landmarks out."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from .landmarks import HandLandmarks, Landmark
from .model import ensure_model

try:
    import cv2
except ImportError as exc:  # pragma: no cover
    raise SystemExit("opencv-python is required: pip install opencv-python") from exc


class HandDetector:
    """Wraps MediaPipe Hand Landmarker. Feed BGR frames from OpenCV."""

    def __init__(
        self,
        model_path: Path | None = None,
        max_hands: int = 2,
        min_detection_confidence: float = 0.6,
        min_presence_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        video_mode: bool = True,
    ) -> None:
        try:
            import mediapipe as mp
        except ImportError as exc:  # pragma: no cover
            raise SystemExit("mediapipe is required: pip install mediapipe") from exc

        self._mp = mp
        path = ensure_model(model_path)
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        RunningMode = mp.tasks.vision.RunningMode

        mode = RunningMode.VIDEO if video_mode else RunningMode.IMAGE
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(path)),
            running_mode=mode,
            num_hands=max_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = HandLandmarker.create_from_options(options)
        self._video_mode = video_mode
        self._t0 = time.perf_counter()
        self._last_ts = -1

    def close(self) -> None:
        closer = getattr(self._landmarker, "close", None)
        if callable(closer):
            closer()

    def __enter__(self) -> "HandDetector":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def detect(self, frame_bgr: np.ndarray) -> list[HandLandmarks]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb = np.ascontiguousarray(rgb)
        mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        if self._video_mode:
            ts = int((time.perf_counter() - self._t0) * 1000)
            if ts <= self._last_ts:
                ts = self._last_ts + 1
            self._last_ts = ts
            result = self._landmarker.detect_for_video(mp_image, ts)
        else:
            result = self._landmarker.detect(mp_image)
        return _parse(result)


def _parse(result: object) -> list[HandLandmarks]:
    hands: list[HandLandmarks] = []
    landmarks_list = getattr(result, "hand_landmarks", None) or []
    handedness_list = getattr(result, "handedness", None) or []
    for i, lms in enumerate(landmarks_list):
        points = [Landmark(float(p.x), float(p.y), float(getattr(p, "z", 0.0))) for p in lms]
        name = "Unknown"
        score = 1.0
        if i < len(handedness_list) and handedness_list[i]:
            cat = handedness_list[i][0]
            name = getattr(cat, "category_name", None) or getattr(cat, "display_name", None) or "Unknown"
            score = float(getattr(cat, "score", 1.0))
        hands.append(HandLandmarks(points, handedness=name, score=score))
    return hands
