"""FastAPI app: JSON API + static demo that embeds the browser widget."""

from __future__ import annotations

import base64
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .classifier import MIN_CONFIDENCE
from .landmarks import signs_payload
from .model import MODEL_PATH
from .pipeline import HandSignRecognizer

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"

_image_recognizer: HandSignRecognizer | None = None


def get_image_recognizer() -> HandSignRecognizer:
    global _image_recognizer
    if _image_recognizer is None:
        _image_recognizer = HandSignRecognizer(video_mode=False, smooth=False)
    return _image_recognizer


class RecognizeJson(BaseModel):
    image: str = Field(..., description="Base64-encoded image (optionally a data URL)")
    min_confidence: float | None = None


def decode_image_bytes(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(status_code=400, detail="could not decode image")
    return frame


def decode_base64_image(payload: str) -> np.ndarray:
    raw = payload.strip()
    if "," in raw and raw.lower().startswith("data:"):
        raw = raw.split(",", 1)[1]
    try:
        data = base64.b64decode(raw, validate=False)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="invalid base64 image") from exc
    return decode_image_bytes(data)


def create_app() -> FastAPI:
    app = FastAPI(
        title="HandSigns",
        version="1.0.0",
        description="Recognize a small set of communicative hand signs (yes, no, okay, …).",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/signs")
    def list_signs() -> dict[str, object]:
        return {"signs": signs_payload()}

    @app.post("/api/recognize")
    async def recognize_upload(
        file: UploadFile | None = File(default=None),
        min_confidence: float = MIN_CONFIDENCE,
    ) -> dict[str, object]:
        if file is None:
            raise HTTPException(status_code=400, detail="upload an image file as `file`")
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="empty file")
        frame = decode_image_bytes(data)
        return _run(frame, min_confidence)

    @app.post("/api/recognize-json")
    def recognize_json(body: RecognizeJson) -> dict[str, object]:
        frame = decode_base64_image(body.image)
        conf = body.min_confidence if body.min_confidence is not None else MIN_CONFIDENCE
        return _run(frame, conf)

    def _run(frame: np.ndarray, min_confidence: float) -> dict[str, object]:
        rec = get_image_recognizer()
        rec.min_confidence = min_confidence
        result, hands = rec.recognize(frame)
        return {
            "result": result.to_dict(),
            "hands": [
                {
                    "handedness": h.handedness,
                    "score": round(h.score, 4),
                    "landmarks": [{"x": p.x, "y": p.y, "z": p.z} for p in h.landmarks],
                }
                for h in hands
            ],
        }

    @app.get("/")
    def index() -> FileResponse:
        page = STATIC_DIR / "index.html"
        if not page.exists():
            raise HTTPException(status_code=404, detail="demo page missing")
        return FileResponse(page)

    @app.api_route("/models/hand_landmarker.task", methods=["GET", "HEAD"])
    def hand_model() -> FileResponse:
        if not MODEL_PATH.exists():
            raise HTTPException(status_code=404, detail="model not downloaded")
        return FileResponse(
            MODEL_PATH,
            media_type="application/octet-stream",
            headers={"Cache-Control": "public, max-age=31536000, immutable"},
        )

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    return app


app = create_app()


def run_server(host: str = "127.0.0.1", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run("handsigns.server:app", host=host, port=port, reload=False)
