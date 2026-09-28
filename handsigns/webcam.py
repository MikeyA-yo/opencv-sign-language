"""Local OpenCV window: webcam → landmarks → sign caption. Press Q to quit."""

from __future__ import annotations

import cv2

from .pipeline import HandSignRecognizer


def run_webcam(camera: int = 0, mirror: bool = True) -> None:
    cap = cv2.VideoCapture(camera)
    if not cap.isOpened():
        raise SystemExit(
            f"could not open camera {camera}. Close other apps using the webcam and retry."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    with HandSignRecognizer(video_mode=True, smooth=True) as rec:
        print("HandSigns webcam. Hold a sign still. Press Q to quit.")
        while True:
            ok, frame = cap.read()
            if not ok:
                print("empty camera frame, retrying...")
                continue
            if mirror:
                frame = cv2.flip(frame, 1)
            annotated, result = rec.annotate(frame)
            cv2.imshow("HandSigns", annotated)
            if result.sign != "none":
                print(f"\r{result.label:<12} {result.confidence:5.0%}   ", end="", flush=True)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()
    print()
