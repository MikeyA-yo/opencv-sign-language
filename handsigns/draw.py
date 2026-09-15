"""OpenCV overlays: skeleton + large sign caption."""

from __future__ import annotations

import cv2
import numpy as np

from .classifier import SignResult
from .landmarks import HAND_CONNECTIONS, HandLandmarks

SIGN_COLOR = {
    "yes": (80, 186, 90),
    "no": (70, 70, 220),
    "okay": (40, 170, 230),
    "hello": (220, 180, 70),
    "peace": (200, 140, 220),
    "point": (180, 180, 180),
    "fist": (140, 140, 160),
    "ily": (180, 90, 200),
    "none": (160, 160, 160),
}


def draw_hands(frame_bgr: np.ndarray, hands: list[HandLandmarks]) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    out = frame_bgr
    for hand in hands:
        pts = [(int(p.x * w), int(p.y * h)) for p in hand.landmarks]
        for a, b in HAND_CONNECTIONS:
            cv2.line(out, pts[a], pts[b], (40, 200, 255), 2, cv2.LINE_AA)
        for x, y in pts:
            cv2.circle(out, (x, y), 4, (245, 245, 245), -1, cv2.LINE_AA)
            cv2.circle(out, (x, y), 4, (20, 20, 20), 1, cv2.LINE_AA)
    return out


def draw_sign(frame_bgr: np.ndarray, result: SignResult) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    color = SIGN_COLOR.get(result.sign, SIGN_COLOR["none"])
    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (0, 0), (w, 86), (18, 16, 14), -1)
    cv2.addWeighted(overlay, 0.72, frame_bgr, 0.28, 0, frame_bgr)

    label = result.label if result.sign != "none" else "waiting..."
    cv2.putText(
        frame_bgr,
        label,
        (18, 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.4,
        color,
        3,
        cv2.LINE_AA,
    )
    if result.sign != "none":
        meta = f"{result.meaning}   {result.confidence:.0%}"
        if result.handedness:
            meta += f"   {result.handedness} hand"
        cv2.putText(
            frame_bgr,
            meta,
            (20, 76),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (210, 210, 210),
            1,
            cv2.LINE_AA,
        )
    return frame_bgr
