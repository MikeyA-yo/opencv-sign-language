"""21-point hand landmark schema (MediaPipe Hands) and sign catalog."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Sequence

WRIST = 0
THUMB_CMC, THUMB_MCP, THUMB_IP, THUMB_TIP = 1, 2, 3, 4
INDEX_MCP, INDEX_PIP, INDEX_DIP, INDEX_TIP = 5, 6, 7, 8
MIDDLE_MCP, MIDDLE_PIP, MIDDLE_DIP, MIDDLE_TIP = 9, 10, 11, 12
RING_MCP, RING_PIP, RING_DIP, RING_TIP = 13, 14, 15, 16
PINKY_MCP, PINKY_PIP, PINKY_DIP, PINKY_TIP = 17, 18, 19, 20

HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (5, 9),
    (9, 13),
    (13, 17),
)

FINGERS: dict[str, tuple[int, int, int]] = {
    "thumb": (THUMB_TIP, THUMB_IP, THUMB_MCP),
    "index": (INDEX_TIP, INDEX_PIP, INDEX_MCP),
    "middle": (MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP),
    "ring": (RING_TIP, RING_PIP, RING_MCP),
    "pinky": (PINKY_TIP, PINKY_PIP, PINKY_MCP),
}


@dataclass(frozen=True)
class Landmark:
    x: float
    y: float
    z: float = 0.0


@dataclass
class HandLandmarks:
    landmarks: Sequence[Landmark]
    handedness: str = "Unknown"
    score: float = 1.0

    def __post_init__(self) -> None:
        if len(self.landmarks) != 21:
            raise ValueError(f"expected 21 landmarks, got {len(self.landmarks)}")

    def __getitem__(self, index: int) -> Landmark:
        return self.landmarks[index]


@dataclass(frozen=True)
class SignInfo:
    id: str
    label: str
    meaning: str
    how: str
    primary: bool = False


SIGNS: dict[str, SignInfo] = {
    "yes": SignInfo(
        id="yes",
        label="YES",
        meaning="Yes",
        how="Thumbs up. Fold the other four fingers; point the thumb straight up.",
        primary=True,
    ),
    "no": SignInfo(
        id="no",
        label="NO",
        meaning="No",
        how="Thumbs down. Fold the other four fingers; point the thumb straight down.",
        primary=True,
    ),
    "okay": SignInfo(
        id="okay",
        label="OKAY",
        meaning="Okay / all good",
        how="Join thumb tip to index tip in a circle. Keep middle, ring, and pinky up.",
        primary=True,
    ),
    "hello": SignInfo(
        id="hello",
        label="HELLO",
        meaning="Hello / open hand",
        how="Open palm facing the camera, all five digits extended.",
        primary=False,
    ),
    "peace": SignInfo(
        id="peace",
        label="PEACE",
        meaning="Peace / two",
        how="Index and middle fingers up in a V. Fold ring and pinky.",
        primary=False,
    ),
    "point": SignInfo(
        id="point",
        label="POINT",
        meaning="Pointing / one",
        how="Index finger up. Fold the other fingers.",
        primary=False,
    ),
    "fist": SignInfo(
        id="fist",
        label="FIST",
        meaning="Closed hand",
        how="Make a fist. All fingers folded.",
        primary=False,
    ),
    "ily": SignInfo(
        id="ily",
        label="I LOVE YOU",
        meaning="I love you (ASL)",
        how="Thumb, index, and pinky extended. Fold middle and ring.",
        primary=False,
    ),
    "none": SignInfo(
        id="none",
        label="—",
        meaning="No recognized sign",
        how="Hold a supported gesture still in the camera.",
        primary=False,
    ),
}


def signs_payload() -> list[dict[str, object]]:
    return [
        {
            "id": s.id,
            "label": s.label,
            "meaning": s.meaning,
            "how": s.how,
            "primary": s.primary,
        }
        for s in SIGNS.values()
        if s.id != "none"
    ]


def landmarks_from_xy(
    points: Iterable[tuple[float, float] | tuple[float, float, float]],
    handedness: str = "Right",
    score: float = 1.0,
) -> HandLandmarks:
    lm = []
    for p in points:
        if len(p) == 2:
            lm.append(Landmark(p[0], p[1], 0.0))
        else:
            lm.append(Landmark(p[0], p[1], p[2]))
    return HandLandmarks(lm, handedness=handedness, score=score)
