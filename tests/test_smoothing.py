import unittest

from handsigns.classifier import SignResult
from handsigns.smoothing import SignSmoother


def _sign(name: str, confidence: float = 0.9) -> SignResult:
    labels = {
        "yes": ("YES", "Yes"),
        "no": ("NO", "No"),
        "none": ("—", "No recognized sign"),
    }
    label, meaning = labels[name]
    return SignResult(sign=name, label=label, meaning=meaning, confidence=confidence)


class SmootherTests(unittest.TestCase):
    def test_requires_hold(self) -> None:
        sm = SignSmoother(window=5, min_hold=3)
        self.assertEqual(sm.update(_sign("yes")).sign, "none")
        self.assertEqual(sm.update(_sign("yes")).sign, "none")
        self.assertEqual(sm.update(_sign("yes")).sign, "yes")

    def test_ignores_single_flicker(self) -> None:
        sm = SignSmoother(window=5, min_hold=3)
        for _ in range(3):
            sm.update(_sign("yes"))
        self.assertEqual(sm.update(_sign("no")).sign, "yes")

    def test_clears_when_hand_leaves(self) -> None:
        sm = SignSmoother(window=4, min_hold=3)
        for _ in range(3):
            sm.update(_sign("yes"))
        for _ in range(4):
            last = sm.update(_sign("none", 0.0))
        self.assertEqual(last.sign, "none")


if __name__ == "__main__":
    unittest.main()
