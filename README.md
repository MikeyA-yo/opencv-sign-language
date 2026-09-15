# HandSigns

A small, pluggable interpreter for a handful of **held** hand signs used in basic yes / no / okay communication. It is **not** a full sign-language translator.

It ships two ways to use the same classifier:

1. **Website widget** — one `<hand-signs>` tag. MediaPipe Hands runs in the browser. The camera never leaves the visitor’s device.
2. **Python + OpenCV module / HTTP API** — webcam window, still-image CLI, and a REST endpoint you can call from any backend.

Python was chosen over C, C++, and TensorFlow.js-from-scratch because OpenCV + MediaPipe Hands already gives stable 21-point landmarks. The signs in this project are then recognized with explicit geometry (finger curl, pinch, thumb direction) rather than a large trained network. That keeps yes / no / okay predictable without a custom dataset. TensorFlow is unnecessary here; you can still swap the classifier later for a trained model on the same 21 landmarks.

## What it recognizes

| Sign | How to hold it | Meaning |
| --- | --- | --- |
| **YES** | Thumbs up, other fingers folded | Yes |
| **NO** | Thumbs down, other fingers folded | No |
| **OKAY** | Thumb tip touches index tip; middle, ring, pinky up | Okay / all good |
| HELLO | Open palm, fingers extended | Hello |
| PEACE | Index + middle up (V) | Peace / two |
| POINT | Index up, others folded | Pointing / one |
| FIST | Closed hand | Closed hand |
| I LOVE YOU | Thumb, index, and pinky extended (ASL) | I love you |

Primary target: **YES / NO / OKAY**. Hold the pose still for about half a second (the smoother ignores one-frame flickers). Dynamic signs (nodding a fist, waving) are out of scope.

## Quick start

Python 3.9+ (tested on 3.13). Create a virtualenv and install:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

`mediapipe` pulls in OpenCV (`opencv-contrib-python`).

### Demo website (widget + API)

```bash
python -m handsigns serve
```

Open [http://127.0.0.1:8000/](http://127.0.0.1:8000/). The widget starts downloading the hand model as soon as the page loads (~8 MB, first visit only; later visits use the browser cache). Click **Start camera** — the webcam should open immediately, even if the model is still finishing. Then hold YES, NO, or OKAY in frame.

### OpenCV desktop window

```bash
python -m handsigns webcam
```

Mirror view, skeleton overlay, large caption. Press **Q** to quit.

### Classify a still image

```bash
python -m handsigns image path\to\hand.jpg --out annotated.jpg
```

### Download the landmark model (optional)

The first detect call downloads `models/hand_landmarker.task` (~few MB) from Google’s MediaPipe model host. To fetch it ahead of time:

```bash
python -m handsigns download-model
```

## Plug the widget into another website

Copy `static/handsigns-widget.js` (or serve it from this app) and add:

```html
<script type="module" src="https://your-host/static/handsigns-widget.js"></script>
<hand-signs speak></hand-signs>

<script type="module">
  const el = document.querySelector("hand-signs");
  el.addEventListener("signchange", (event) => {
    // event.detail = { sign, label, meaning, confidence }
    if (event.detail.sign === "yes") {
      // confirm a form, etc.
    }
  });
</script>
```

Attributes:

- `speak` — read the meaning aloud with the Web Speech API when the sign changes.

Methods on the element: `start()`, `stop()`.

The widget loads MediaPipe Tasks Vision from a CDN and the official `hand_landmarker.task` model. Recognition is local. A minimal copy-paste page lives at `static/embed.html`.

You can also mount it from JavaScript:

```js
import { mount } from "./handsigns-widget.js";
mount("#slot", {
  speak: true,
  onSign: (result) => console.log(result),
});
```

## HTTP API

Base URL: `http://127.0.0.1:8000` (CORS is open so a page on another origin can call it).

| Method | Path | Purpose |
| --- | --- | --- |
| GET | `/api/health` | `{ "status": "ok" }` |
| GET | `/api/signs` | Catalog of supported signs |
| POST | `/api/recognize` | Multipart file field `file` |
| POST | `/api/recognize-json` | JSON `{ "image": "<base64 or data URL>" }` |

Example:

```bash
curl -F "file=@hand.jpg" http://127.0.0.1:8000/api/recognize
```

Response shape:

```json
{
  "result": {
    "sign": "yes",
    "label": "YES",
    "meaning": "Yes",
    "confidence": 0.91,
    "handedness": "Right",
    "hand_score": 0.98,
    "details": { "thumb": 0.82, "pinch": 0.01, "thumb_up": 0.9 }
  },
  "hands": [
    {
      "handedness": "Right",
      "score": 0.98,
      "landmarks": [{ "x": 0.51, "y": 0.72, "z": 0.0 }]
    }
  ]
}
```

`sign` is one of `yes`, `no`, `okay`, `hello`, `peace`, `point`, `fist`, `ily`, or `none`. Landmarks are normalized to `[0, 1]` in image coordinates (MediaPipe Hands, 21 points).

For live video on a website, prefer the **widget** (no frame uploads). Use the API for server-side stills, batch jobs, or a backend that already has images.

## Python module

```python
from handsigns.classifier import classify_hand
from handsigns.landmarks import landmarks_from_xy

# 21 (x, y) points in MediaPipe order, normalized 0–1
hand = landmarks_from_xy(points, handedness="Right")
print(classify_hand(hand).sign)

from handsigns.pipeline import HandSignRecognizer
import cv2

frame = cv2.imread("hand.jpg")
with HandSignRecognizer(video_mode=False, smooth=False) as rec:
    annotated, result = rec.annotate(frame)
print(result.label, result.confidence)
```

Layout:

```
handsigns/           Python package
  classifier.py      geometric rules (yes/no/okay/…)
  detector.py        OpenCV BGR frame → MediaPipe landmarks
  pipeline.py        detect + classify + smooth
  server.py          FastAPI app
  webcam.py          cv2.imshow loop
static/              demo site + embeddable widget
tests/               synthetic-landmark unit tests
models/              downloaded .task file (gitignored)
```

## How classification works

1. **Detect** 21 landmarks per hand (wrist, four thumb joints, four joints on each finger).
2. **Score finger curl** from how far each fingertip sits from the wrist versus the PIP / MCP — this still works if the palm is slightly tilted.
3. **Score thumb direction** in image space (negative Y = up) for YES vs NO.
4. **Score pinch** (thumb tip to index tip) for OKAY.
5. Pick the highest-scoring rule above a confidence floor (`0.55`).
6. **Smooth** over ~6 frames so a pose must hold before the label flips.

OKAY is checked before open-palm HELLO so a circle is not read as a five-finger hand.

## Tests

The classifier does not need OpenCV or MediaPipe:

```bash
python -m unittest tests.test_classifier tests.test_smoothing -v
```

Run these from the repo root with `PYTHONPATH` set to that root if you are not in an editable install.

## Tips for reliable YES / NO / OKAY

- Sit a meter from the camera, hand filling a large part of the frame.
- Plain background, even light, sleeve not covering the wrist.
- Palm roughly toward the camera. YES/NO need a clearly vertical thumb.
- Hold still; waving is ignored on purpose.

## Limits

- A few static poses, not ASL/ISL grammar, fingerspelling, or motion signs.
- Two hands: the higher-confidence hand wins.
- Webcam quality, occlusion, and extreme angles will yield `none`.
- This is an accessibility helper for basic confirmation, not a medical or legal interpreter.

## License notes

MediaPipe models are Apache 2.0 (Google). OpenCV is Apache 2.0. This project’s code is yours to use in the site you plug it into.
