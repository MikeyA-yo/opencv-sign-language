"""HTTP surface that does not need a camera."""

from __future__ import annotations

import unittest

from fastapi.testclient import TestClient

from handsigns.server import app


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def test_health(self) -> None:
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.json()["status"], "ok")

    def test_signs_include_primary(self) -> None:
        res = self.client.get("/api/signs")
        self.assertEqual(res.status_code, 200)
        ids = {item["id"] for item in res.json()["signs"]}
        self.assertTrue({"yes", "no", "okay"}.issubset(ids))

    def test_demo_page(self) -> None:
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        self.assertIn("hand-signs", res.text)

    def test_widget_js(self) -> None:
        res = self.client.get("/static/handsigns-widget.js")
        self.assertEqual(res.status_code, 200)
        self.assertIn("customElements", res.text)

    def test_local_model_route(self) -> None:
        res = self.client.get("/models/hand_landmarker.task")
        self.assertIn(res.status_code, {200, 404})
        if res.status_code == 200:
            self.assertGreater(len(res.content), 1_000_000)


if __name__ == "__main__":
    unittest.main()
