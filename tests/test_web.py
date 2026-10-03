import json
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import cv2

from lpr.models import PlateCNN
from lpr.runtime import save_checkpoint
from lpr.web import LPRServer, ModelStore


class WebTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.store = ModelStore(root / "recognizer.pt", root / "detector.pt")
        self.server = LPRServer(("127.0.0.1", 0), self.store)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=3)
        self.temp.cleanup()

    def request_json(self, path, data=None, content_type="image/png"):
        request = Request(self.base + path, data=data, headers={"Content-Type": content_type} if data is not None else {})
        try:
            response = urlopen(request, timeout=3)
        except HTTPError as error:
            response = error
        with response:
            return response.status, json.load(response)

    def test_page_and_missing_model_status(self):
        with urlopen(self.base + "/", timeout=3) as response:
            self.assertIn("车牌识别练习".encode(), response.read())
        with urlopen(self.base + "/app.js", timeout=3) as response:
            self.assertIn(b"renderConvolution", response.read())
        code, status = self.request_json("/api/status")
        self.assertEqual(code, 200)
        self.assertFalse(status["recognizer_ready"])
        self.assertFalse(status["detector_ready"])
        code, payload = self.request_json("/api/predict?mode=crop", data=b"invalid")
        self.assertEqual(code, 409)
        self.assertIn("模型", payload["error"])

    def test_new_checkpoint_is_loaded_and_crop_endpoint_runs(self):
        save_checkpoint(self.store.paths["recognizer"], PlateCNN(), "recognizer")
        code, status = self.request_json("/api/status")
        self.assertEqual(code, 200)
        self.assertTrue(status["recognizer_ready"])
        self.assertFalse(status["detector_ready"])
        image = np.zeros((64, 192, 3), dtype=np.uint8)
        ok, encoded = cv2.imencode(".png", image)
        self.assertTrue(ok)
        code, result = self.request_json("/api/predict?mode=crop", data=encoded.tobytes())
        self.assertEqual(code, 200)
        self.assertEqual(result["status"], "ok")
        self.assertEqual(len(result["plate"]), 7)
        code, missing_detector = self.request_json("/api/predict?mode=full", data=encoded.tobytes())
        self.assertEqual(code, 409)
        self.assertIn("定位模型", missing_detector["error"])


if __name__ == "__main__":
    unittest.main()
