"""HandSigns: basic hand-sign recognition for web and OpenCV apps."""

from .classifier import SignResult, classify_hand, classify_hands
from .landmarks import HAND_CONNECTIONS, SIGNS, HandLandmarks, Landmark
from .smoothing import SignSmoother

__version__ = "1.0.0"
__all__ = [
    "SignResult",
    "SignSmoother",
    "classify_hand",
    "classify_hands",
    "HAND_CONNECTIONS",
    "Landmark",
    "HandLandmarks",
    "SIGNS",
]
