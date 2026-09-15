"""Temporal smoothing so a flicker of a wrong pose does not flip the label."""

from __future__ import annotations

from collections import deque

from .classifier import SignResult
from .landmarks import SIGNS


class SignSmoother:
    """Majority vote over a short window. `none` is sticky until a sign holds."""

    def __init__(self, window: int = 6, min_hold: int = 3) -> None:
        if window < 1:
            raise ValueError("window must be >= 1")
        if min_hold < 1:
            raise ValueError("min_hold must be >= 1")
        self.window = window
        self.min_hold = min_hold
        self._buf: deque[SignResult] = deque(maxlen=window)
        self._stable = _empty()

    def reset(self) -> None:
        self._buf.clear()
        self._stable = _empty()

    def update(self, result: SignResult) -> SignResult:
        self._buf.append(result)
        votes: dict[str, list[SignResult]] = {}
        for item in self._buf:
            votes.setdefault(item.sign, []).append(item)

        best_sign = "none"
        best_count = 0
        for sign, items in votes.items():
            if sign == "none":
                continue
            if len(items) > best_count:
                best_sign = sign
                best_count = len(items)

        if best_count >= self.min_hold:
            chosen = max(votes[best_sign], key=lambda r: r.confidence)
            self._stable = SignResult(
                sign=chosen.sign,
                label=chosen.label,
                meaning=chosen.meaning,
                confidence=chosen.confidence,
                handedness=chosen.handedness,
                hand_score=chosen.hand_score,
                details=dict(chosen.details),
            )
            return self._stable

        if len(self._buf) >= self.window and best_count == 0:
            self._stable = _empty()
        return self._stable


def _empty() -> SignResult:
    info = SIGNS["none"]
    return SignResult(
        sign="none",
        label=info.label,
        meaning=info.meaning,
        confidence=0.0,
    )
