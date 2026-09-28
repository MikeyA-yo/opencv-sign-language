"""OpenCV desktop demo.

You can run this file with the system Python; it will switch to the project
virtualenv automatically if OpenCV is missing there:

    python webcam.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from handsigns.deps import reexec_in_venv_if_needed  # noqa: E402

if __name__ == "__main__":
    # If this file was launched as `python webcam.py`, rewrite argv so the
    # venv re-exec becomes `python -m handsigns webcam`.
    if len(sys.argv) == 1 or Path(sys.argv[0]).name.lower() == "webcam.py":
        sys.argv = ["handsigns", "webcam", *sys.argv[1:]]
    reexec_in_venv_if_needed()
    from handsigns.webcam import run_webcam

    camera = 0
    if "--camera" in sys.argv:
        i = sys.argv.index("--camera")
        if i + 1 < len(sys.argv):
            camera = int(sys.argv[i + 1])
    run_webcam(camera=camera, mirror="--no-mirror" not in sys.argv)
