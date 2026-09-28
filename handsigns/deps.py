"""Make sure the OpenCV/MediaPipe venv is the interpreter actually running."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VENV_PY = ROOT / ".venv" / "Scripts" / "python.exe"
INSTALL_HINT = f"""OpenCV (cv2) is not installed for this Python:

  {sys.executable}

Use the project virtualenv (it already has OpenCV + MediaPipe):

  {ROOT}\\.venv\\Scripts\\activate
  python -m handsigns webcam

Or, without activating:

  {ROOT}\\.venv\\Scripts\\python.exe -m handsigns webcam
"""


def running_in_project_venv() -> bool:
    try:
        return Path(sys.executable).resolve() == VENV_PY.resolve()
    except OSError:
        return False


def reexec_in_venv_if_needed() -> None:
    """If cv2 is missing, hop into .venv automatically when it exists."""
    try:
        import cv2  # noqa: F401
        return
    except ImportError:
        pass

    if VENV_PY.exists() and not running_in_project_venv():
        os.execv(str(VENV_PY), [str(VENV_PY), "-m", "handsigns", *sys.argv[1:]])

    raise SystemExit(INSTALL_HINT)
