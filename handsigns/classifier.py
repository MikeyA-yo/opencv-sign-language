"""Geometric classifier over 21 hand landmarks.

Maps a small, high-confidence set of everyday signs used for basic
yes/no/okay communication. Rules are orientation-aware for thumbs
(up vs down) and pinch-aware for OK. Finger curl uses wrist distance
so a slightly tilted palm still classifies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .landmarks import (
    INDEX_MCP,
    INDEX_PIP,
    INDEX_TIP,
    MIDDLE_MCP,
    MIDDLE_PIP,
    MIDDLE_TIP,
    PINKY_MCP,
    PINKY_PIP,
    PINKY_TIP,
    RING_MCP,
    RING_PIP,
    RING_TIP,
    SIGNS,
    THUMB_IP,
    THUMB_MCP,
    THUMB_TIP,
    WRIST,
    HandLandmarks,
    Landmark,
)

# Tuned against synthetic hands and typical webcam poses.
PINCH_OK = 0.38
EXT_ON = 0.42
EXT_OFF = 0.38
THUMB_EXT_ON = 0.48
MIN_CONFIDENCE = 0.55


@dataclass
class SignResult:
    sign: str
    label: str
    meaning: str
    confidence: float
    handedness: str | None = None
    hand_score: float = 0.0
    details: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "sign": self.sign,
            "label": self.label,
            "meaning": self.meaning,
            "confidence": round(self.confidence, 4),
            "handedness": self.handedness,
            "hand_score": round(self.hand_score, 4),
            "details": {k: round(v, 4) for k, v in self.details.items()},
        }


def _none(hand: HandLandmarks | None = None, confidence: float = 0.0) -> SignResult:
    info = SIGNS["none"]
    return SignResult(
        sign="none",
        label=info.label,
        meaning=info.meaning,
        confidence=confidence,
        handedness=None if hand is None else hand.handedness,
        hand_score=0.0 if hand is None else hand.score,
    )


def _dist(a: Landmark, b: Landmark) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def _palm_size(hand: HandLandmarks) -> float:
    wrist = hand[WRIST]
    across = _dist(hand[INDEX_MCP], hand[PINKY_MCP])
    length = _dist(wrist, hand[MIDDLE_MCP])
    return max(across, length, 1e-6)


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if value < lo else hi if value > hi else value


def _finger_extension(hand: HandLandmarks, tip: int, pip: int, mcp: int) -> float:
    """1.0 = straight out, 0.0 = curled toward the palm."""
    palm = _palm_size(hand)
    wrist = hand[WRIST]
    tip_d = _dist(hand[tip], wrist)
    pip_d = _dist(hand[pip], wrist)
    mcp_d = _dist(hand[mcp], wrist)
    # Tip farther from the wrist than PIP → extended.
    along = (tip_d - pip_d) / (palm * 0.55)
    # TIP far from MCP relative to PIP-MCP also indicates extension.
    reach = (_dist(hand[tip], hand[mcp]) - _dist(hand[pip], hand[mcp])) / (palm * 0.45)
    # Folded fingers often have tip closer to wrist than MCP.
    unfold = (tip_d - mcp_d) / (palm * 0.7)
    return _clamp(0.45 * along + 0.35 * reach + 0.20 * unfold)


def _thumb_extension(hand: HandLandmarks) -> float:
    palm = _palm_size(hand)
    span = _dist(hand[THUMB_TIP], hand[THUMB_MCP]) / (palm * 0.85)
    away_from_index = _dist(hand[THUMB_TIP], hand[INDEX_MCP]) / (palm * 0.9)
    ip_span = _dist(hand[THUMB_TIP], hand[THUMB_IP]) / (palm * 0.4)
    return _clamp(0.5 * span + 0.3 * away_from_index + 0.2 * ip_span)


def _thumb_direction(hand: HandLandmarks) -> tuple[float, float]:
    """Return (up_score, down_score) in image coordinates (y grows downward)."""
    mcp = hand[THUMB_MCP]
    tip = hand[THUMB_TIP]
    dx = tip.x - mcp.x
    dy = tip.y - mcp.y
    length = math.hypot(dx, dy) or 1e-6
    ux, uy = dx / length, dy / length
    # Image up is negative y.
    up = _clamp((-uy) * 1.15)
    down = _clamp(uy * 1.15)
    # Penalize mostly-horizontal thumbs.
    vertical = abs(uy)
    return up * vertical, down * vertical


def _pinch(hand: HandLandmarks) -> float:
    palm = _palm_size(hand)
    d = _dist(hand[THUMB_TIP], hand[INDEX_TIP]) / palm
    return _clamp(1.0 - d / PINCH_OK)


def _finger_states(hand: HandLandmarks) -> dict[str, float]:
    return {
        "thumb": _thumb_extension(hand),
        "index": _finger_extension(hand, INDEX_TIP, INDEX_PIP, INDEX_MCP),
        "middle": _finger_extension(hand, MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP),
        "ring": _finger_extension(hand, RING_TIP, RING_PIP, RING_MCP),
        "pinky": _finger_extension(hand, PINKY_TIP, PINKY_PIP, PINKY_MCP),
    }


def _on(score: float) -> float:
    return _clamp((score - EXT_ON) / (1.0 - EXT_ON))


def _off(score: float) -> float:
    return _clamp((EXT_OFF - score) / EXT_OFF)


def _result(sign_id: str, confidence: float, hand: HandLandmarks, details: dict[str, float]) -> SignResult:
    info = SIGNS[sign_id]
    return SignResult(
        sign=sign_id,
        label=info.label,
        meaning=info.meaning,
        confidence=_clamp(confidence),
        handedness=hand.handedness,
        hand_score=hand.score,
        details=details,
    )


def classify_hand(hand: HandLandmarks, min_confidence: float = MIN_CONFIDENCE) -> SignResult:
    """Classify a single hand. Returns sign `none` when nothing is confident."""
    f = _finger_states(hand)
    pinch = _pinch(hand)
    up, down = _thumb_direction(hand)
    thumb_on = _clamp((f["thumb"] - THUMB_EXT_ON) / (1.0 - THUMB_EXT_ON))
    idx_on, idx_off = _on(f["index"]), _off(f["index"])
    mid_on, mid_off = _on(f["middle"]), _off(f["middle"])
    ring_on, ring_off = _on(f["ring"]), _off(f["ring"])
    pinky_on, pinky_off = _on(f["pinky"]), _off(f["pinky"])
    others_off = (idx_off + mid_off + ring_off + pinky_off) / 4.0
    three_up = (mid_on + ring_on + pinky_on) / 3.0

    details = {
        "thumb": f["thumb"],
        "index": f["index"],
        "middle": f["middle"],
        "ring": f["ring"],
        "pinky": f["pinky"],
        "pinch": pinch,
        "thumb_up": up,
        "thumb_down": down,
    }

    candidates: list[tuple[str, float]] = []

    # OK: thumb-index pinch + remaining fingers up. Most specific; check first.
    okay = 0.55 * pinch + 0.45 * three_up
    if pinch > 0.45 and three_up > 0.35:
        candidates.append(("okay", okay))

    # I love you: thumb + index + pinky, middle and ring folded.
    ily = (thumb_on + idx_on + pinky_on + mid_off + ring_off) / 5.0
    if thumb_on > 0.35 and idx_on > 0.4 and pinky_on > 0.4 and mid_off > 0.35 and ring_off > 0.35:
        candidates.append(("ily", ily))

    # Peace: index + middle up, ring + pinky folded. Thumb may be tucked or relaxed.
    peace = (idx_on + mid_on + ring_off + pinky_off) / 4.0
    if idx_on > 0.45 and mid_on > 0.45 and ring_off > 0.35 and pinky_off > 0.35 and pinch < 0.4:
        candidates.append(("peace", peace))

    # Point: only index up.
    point = (idx_on + mid_off + ring_off + pinky_off) / 4.0
    if idx_on > 0.5 and mid_off > 0.4 and ring_off > 0.4 and pinky_off > 0.4 and pinch < 0.4:
        candidates.append(("point", point))

    # YES: thumbs up, other fingers folded, thumb pointing up.
    yes = 0.4 * thumb_on + 0.35 * others_off + 0.25 * up
    if thumb_on > 0.35 and others_off > 0.45 and up > 0.45 and pinch < 0.5:
        candidates.append(("yes", yes))

    # NO: thumbs down.
    no = 0.4 * thumb_on + 0.35 * others_off + 0.25 * down
    if thumb_on > 0.35 and others_off > 0.45 and down > 0.45 and pinch < 0.5:
        candidates.append(("no", no))

    # Open palm / hello: four fingers extended, no pinch.
    hello = (idx_on + mid_on + ring_on + pinky_on) / 4.0
    if hello > 0.55 and pinch < 0.35:
        candidates.append(("hello", 0.7 * hello + 0.3 * max(thumb_on, 0.4)))

    # Fist: four fingers folded and the thumb is not clearly extended.
    # A tucked thumb can still point "up" in image space, so do not use
    # thumb direction here — thumbs-up already requires thumb_on + verticality.
    if others_off > 0.55 and f["thumb"] < THUMB_EXT_ON:
        fist = 0.65 * others_off + 0.35 * (1.0 - f["thumb"])
        candidates.append(("fist", fist))

    if not candidates:
        return _none(hand, 0.0)

    sign_id, confidence = max(candidates, key=lambda item: item[1])
    if confidence < min_confidence:
        return _none(hand, confidence)
    return _result(sign_id, confidence, hand, details)


def classify_hands(
    hands: list[HandLandmarks],
    min_confidence: float = MIN_CONFIDENCE,
) -> SignResult:
    """Pick the highest-confidence hand. Empty list → none."""
    if not hands:
        return _none()
    results = [classify_hand(hand, min_confidence=min_confidence) for hand in hands]
    ranked = sorted(results, key=lambda r: (r.sign != "none", r.confidence), reverse=True)
    return ranked[0]
