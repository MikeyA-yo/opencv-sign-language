"""CLI: serve | webcam | image | download-model."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="handsigns",
        description="Recognize basic communicative hand signs (yes / no / okay).",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    serve = sub.add_parser("serve", help="Run the HTTP API and demo website")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)

    cam = sub.add_parser("webcam", help="OpenCV desktop demo")
    cam.add_argument("--camera", type=int, default=0)
    cam.add_argument("--no-mirror", action="store_true")

    img = sub.add_parser("image", help="Classify a still image")
    img.add_argument("path", type=Path)
    img.add_argument("--out", type=Path, default=None, help="optional annotated output path")

    sub.add_parser("download-model", help="Fetch the hand landmarker model")

    args = parser.parse_args(argv)

    if args.cmd == "serve":
        from .server import run_server

        print(f"HandSigns demo: http://{args.host}:{args.port}/")
        run_server(host=args.host, port=args.port)
        return

    if args.cmd == "webcam":
        from .webcam import run_webcam

        run_webcam(camera=args.camera, mirror=not args.no_mirror)
        return

    if args.cmd == "download-model":
        from .model import ensure_model

        path = ensure_model()
        print(f"model ready: {path}")
        return

    if args.cmd == "image":
        from .pipeline import HandSignRecognizer

        frame = cv2.imread(str(args.path))
        if frame is None:
            raise SystemExit(f"could not read image: {args.path}")
        with HandSignRecognizer(video_mode=False, smooth=False) as rec:
            annotated, result = rec.annotate(frame)
        print(result.to_dict())
        if args.out:
            cv2.imwrite(str(args.out), annotated)
            print(f"wrote {args.out}")
        return


if __name__ == "__main__":
    main()
