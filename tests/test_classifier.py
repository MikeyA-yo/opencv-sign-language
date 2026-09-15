"""Synthetic-landmark tests for the geometric sign classifier."""

from __future__ import annotations

import unittest

from handsigns.classifier import classify_hand
from handsigns.landmarks import landmarks_from_xy


def _base_palm() -> list[list[float]]:
    """Right hand, palm toward camera, wrist at bottom. y grows downward."""
    pts = [[0.0, 0.0] for _ in range(21)]
    pts[0] = [0.50, 0.82]  # wrist
    # thumb resting along the side
    pts[1] = [0.43, 0.76]
    pts[2] = [0.37, 0.70]
    pts[3] = [0.33, 0.64]
    pts[4] = [0.30, 0.58]
    # index
    pts[5] = [0.44, 0.56]
    pts[6] = [0.43, 0.44]
    pts[7] = [0.43, 0.34]
    pts[8] = [0.42, 0.24]
    # middle
    pts[9] = [0.50, 0.54]
    pts[10] = [0.50, 0.40]
    pts[11] = [0.50, 0.28]
    pts[12] = [0.50, 0.16]
    # ring
    pts[13] = [0.56, 0.56]
    pts[14] = [0.57, 0.44]
    pts[15] = [0.58, 0.34]
    pts[16] = [0.58, 0.24]
    # pinky
    pts[17] = [0.62, 0.58]
    pts[18] = [0.64, 0.48]
    pts[19] = [0.65, 0.40]
    pts[20] = [0.66, 0.32]
    return pts


def _fold(pts: list[list[float]], mcp: int, pip: int, dip: int, tip: int) -> None:
    """Curl a finger toward the palm / wrist."""
    wrist = pts[0]
    base = pts[mcp]
    pts[pip] = [base[0] * 0.7 + wrist[0] * 0.3, base[1] * 0.55 + wrist[1] * 0.45]
    pts[dip] = [base[0] * 0.55 + wrist[0] * 0.45, base[1] * 0.4 + wrist[1] * 0.6]
    pts[tip] = [base[0] * 0.45 + wrist[0] * 0.55, base[1] * 0.3 + wrist[1] * 0.7]


def _hand(pts: list[list[float]]):
    return landmarks_from_xy((tuple(p) for p in pts), handedness="Right")


class ClassifierTests(unittest.TestCase):
    def test_yes_thumbs_up(self) -> None:
        pts = _base_palm()
        for mcp, pip, dip, tip in ((5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)):
            _fold(pts, mcp, pip, dip, tip)
        pts[2] = [0.40, 0.68]
        pts[3] = [0.39, 0.52]
        pts[4] = [0.38, 0.32]
        result = classify_hand(_hand(pts))
        self.assertEqual(result.sign, "yes", result)

    def test_no_thumbs_down(self) -> None:
        pts = _base_palm()
        for mcp, pip, dip, tip in ((5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)):
            _fold(pts, mcp, pip, dip, tip)
        pts[2] = [0.40, 0.70]
        pts[3] = [0.40, 0.84]
        pts[4] = [0.40, 0.98]
        result = classify_hand(_hand(pts))
        self.assertEqual(result.sign, "no", result)

    def test_okay_pinch(self) -> None:
        pts = _base_palm()
        # Circle: thumb tip meets index tip; other three fingers stay up.
        pts[4] = [0.40, 0.42]
        pts[6] = [0.41, 0.48]
        pts[7] = [0.40, 0.45]
        pts[8] = [0.40, 0.42]
        result = classify_hand(_hand(pts))
        self.assertEqual(result.sign, "okay", result)

    def test_hello_open_palm(self) -> None:
        pts = _base_palm()
        pts[4] = [0.26, 0.50]
        result = classify_hand(_hand(pts))
        self.assertEqual(result.sign, "hello", result)

    def test_peace(self) -> None:
        pts = _base_palm()
        _fold(pts, 13, 14, 15, 16)
        _fold(pts, 17, 18, 19, 20)
        pts[4] = [0.34, 0.62]
        result = classify_hand(_hand(pts))
        self.assertEqual(result.sign, "peace", result)

    def test_point(self) -> None:
        pts = _base_palm()
        _fold(pts, 9, 10, 11, 12)
        _fold(pts, 13, 14, 15, 16)
        _fold(pts, 17, 18, 19, 20)
        pts[4] = [0.34, 0.62]
        result = classify_hand(_hand(pts))
        self.assertEqual(result.sign, "point", result)

    def test_fist(self) -> None:
        pts = _base_palm()
        for mcp, pip, dip, tip in ((5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)):
            _fold(pts, mcp, pip, dip, tip)
        pts[2] = [0.42, 0.70]
        pts[3] = [0.46, 0.64]
        pts[4] = [0.50, 0.60]
        result = classify_hand(_hand(pts))
        self.assertEqual(result.sign, "fist", result)

    def test_ily(self) -> None:
        pts = _base_palm()
        _fold(pts, 9, 10, 11, 12)
        _fold(pts, 13, 14, 15, 16)
        pts[2] = [0.34, 0.66]
        pts[3] = [0.28, 0.60]
        pts[4] = [0.22, 0.52]
        result = classify_hand(_hand(pts))
        self.assertEqual(result.sign, "ily", result)


if __name__ == "__main__":
    unittest.main()
