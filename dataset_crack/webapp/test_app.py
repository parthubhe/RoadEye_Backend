"""Fast local API checks; no checkpoint loading or GPU required."""

from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np
from fastapi.testclient import TestClient

from . import app as server
from .engine import InferenceResult, encode_jpeg


def fake_inference(frame, model_id, confidence):
    return InferenceResult(frame.copy(), [{
        "xyxy": [1.0, 1.0, 8.0, 8.0], "confidence": 0.8,
        "class_id": 0, "class_name": "crack",
    }], 3.5, 0.0)


class WebAppTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(server.app)
        cls.model_id = next(identity for identity, spec in server.catalog.items() if spec.checkpoint)

    def test_catalog_has_evidence_and_unavailable_cloud_model(self):
        response = self.client.get("/api/models")
        self.assertEqual(response.status_code, 200)
        models = response.json()["models"]
        self.assertGreaterEqual(len(models), 15)
        self.assertTrue(all(model["parameters"] for model in models if model["available"]))
        self.assertTrue(any(model["id"] == "cloud__e3" and not model["available"] for model in models))
        self.assertEqual(self.client.get("/").status_code, 200)
        self.assertEqual(self.client.get("/api/health").status_code, 200)

    def test_image_upload_and_output(self):
        frame = np.zeros((32, 48, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(server, "JOBS_ROOT", Path(directory)), mock.patch.object(server.manager, "infer", side_effect=fake_inference):
            response = self.client.post(
                "/api/infer/image", params={"model_id": self.model_id, "confidence": .25},
                files={"file": ("pipe.jpg", encode_jpeg(frame), "image/jpeg")},
            )
            self.assertEqual(response.status_code, 200, response.text)
            payload = response.json()
            self.assertEqual(payload["detections"][0]["class_name"], "crack")
            result = self.client.get(payload["image_url"])
            self.assertEqual(result.status_code, 200)
            self.assertEqual(result.headers["content-type"], "image/jpeg")

    def test_unknown_model_rejected(self):
        response = self.client.post(
            "/api/infer/image", params={"model_id": "../../model"},
            files={"file": ("pipe.jpg", b"bad image", "image/jpeg")},
        )
        self.assertEqual(response.status_code, 404)

    def test_browser_live_websocket(self):
        frame = np.zeros((32, 48, 3), dtype=np.uint8)
        with mock.patch.object(server.manager, "infer", side_effect=fake_inference):
            with self.client.websocket_connect(f"/ws/live/browser?model_id={self.model_id}&confidence=0.25") as socket:
                socket.send_bytes(encode_jpeg(frame))
                result = socket.receive_json()
                self.assertEqual(result["type"], "frame")
                self.assertEqual(result["detections"][0]["class_name"], "crack")
                self.assertTrue(result["image"])

    def test_live_source_rejects_invalid_usb_index(self):
        with self.client.websocket_connect(f"/ws/live/source?kind=usb&source=9&model_id={self.model_id}") as socket:
            result = socket.receive_json()
            self.assertEqual(result["type"], "error")
            self.assertIn("0", result["message"])

    def test_video_job_progress_and_download(self):
        with tempfile.TemporaryDirectory() as directory:
            input_path = Path(directory) / "source.mp4"
            writer = cv2.VideoWriter(str(input_path), cv2.VideoWriter_fourcc(*"mp4v"), 5, (48, 32))
            self.assertTrue(writer.isOpened())
            for _ in range(3):
                writer.write(np.zeros((32, 48, 3), dtype=np.uint8))
            writer.release()
            with mock.patch.object(server, "JOBS_ROOT", Path(directory) / "jobs"), mock.patch.object(server.manager, "infer", side_effect=fake_inference), mock.patch.object(server.shutil, "which", return_value=None):
                response = self.client.post(
                    "/api/infer/video", params={"model_id": self.model_id, "stride": 1},
                    files={"file": ("source.mp4", input_path.read_bytes(), "video/mp4")},
                )
                self.assertEqual(response.status_code, 202, response.text)
                job_id = response.json()["id"]
                for _ in range(50):
                    result = self.client.get(f"/api/jobs/{job_id}").json()
                    if result["status"] in {"complete", "error"}:
                        break
                    time.sleep(.1)
                self.assertEqual(result["status"], "complete", result)
                self.assertEqual(result["frames_written"], 3)
                self.assertEqual(self.client.get(result["output_url"]).status_code, 200)


if __name__ == "__main__":
    unittest.main()
