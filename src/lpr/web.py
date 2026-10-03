"""Local web interface for model inference and CNN learning."""

import argparse
import json
import threading
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.resources import files
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import cv2
import numpy as np

from .predict import infer
from .runtime import load_checkpoint

MAX_IMAGE_BYTES = 10 * 1024 * 1024
STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
}


class ModelStore:
    def __init__(self, recognizer_path: str | Path, detector_path: str | Path, device: str = "cpu"):
        self.paths = {"recognizer": Path(recognizer_path), "detector": Path(detector_path)}
        self.device = device
        self.models = {"recognizer": None, "detector": None}
        self.versions = {"recognizer": None, "detector": None}
        self.errors = {"recognizer": None, "detector": None}
        self.lock = threading.Lock()

    def snapshot(self):
        with self.lock:
            for kind, path in self.paths.items():
                version = path.stat().st_mtime_ns if path.is_file() else None
                if version == self.versions[kind]:
                    continue
                self.versions[kind] = version
                self.models[kind] = None
                self.errors[kind] = None
                if version is not None:
                    try:
                        self.models[kind], _ = load_checkpoint(path, kind, self.device)
                    except (OSError, RuntimeError, ValueError, KeyError) as exc:
                        self.errors[kind] = f"模型加载失败：{exc}"
            return dict(self.models), dict(self.errors)

    def status(self) -> dict:
        models, errors = self.snapshot()
        return {
            "recognizer_ready": models["recognizer"] is not None,
            "detector_ready": models["detector"] is not None,
            "recognizer_error": errors["recognizer"],
            "detector_error": errors["detector"],
            "max_image_mb": MAX_IMAGE_BYTES // (1024 * 1024),
        }


class LPRServer(ThreadingHTTPServer):
    def __init__(self, address, store: ModelStore):
        super().__init__(address, LPRHandler)
        self.store = store


class LPRHandler(BaseHTTPRequestHandler):
    server: LPRServer

    def _respond(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, payload: dict) -> None:
        self._respond(code, json.dumps(payload, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self) -> None:
        route = urlsplit(self.path).path
        if route == "/api/status":
            self._json(HTTPStatus.OK, self.server.store.status())
        elif route in STATIC_FILES:
            filename, content_type = STATIC_FILES[route]
            body = files("lpr").joinpath("static", filename).read_bytes()
            self._respond(HTTPStatus.OK, body, content_type)
        elif route == "/favicon.ico":
            self._respond(HTTPStatus.NO_CONTENT, b"", "image/x-icon")
        else:
            self._json(HTTPStatus.NOT_FOUND, {"error": "页面不存在"})

    def do_POST(self) -> None:
        url = urlsplit(self.path)
        if url.path != "/api/predict":
            self._json(HTTPStatus.NOT_FOUND, {"error": "接口不存在"})
            return
        mode = parse_qs(url.query).get("mode", ["full"])[0]
        if mode not in ("full", "crop"):
            self._json(HTTPStatus.BAD_REQUEST, {"error": "识别模式无效"})
            return
        models, errors = self.server.store.snapshot()
        if models["recognizer"] is None:
            self._json(HTTPStatus.CONFLICT, {"error": errors["recognizer"] or "识字模型尚未训练。请先按 README 训练并保存到 outputs/recognizer.pt。"})
            return
        if mode == "full" and models["detector"] is None:
            self._json(HTTPStatus.CONFLICT, {"error": errors["detector"] or "定位模型尚未训练。完整车辆图片需要 outputs/detector.pt；也可先试裁剪车牌模式。"})
            return
        if self.headers.get_content_type() not in ("image/jpeg", "image/png", "image/webp"):
            self._json(HTTPStatus.UNSUPPORTED_MEDIA_TYPE, {"error": "请选择 JPG、PNG 或 WebP 图片"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if not 0 < length <= MAX_IMAGE_BYTES:
            self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "图片大小需在 10 MB 以内"})
            return
        encoded = np.frombuffer(self.rfile.read(length), dtype=np.uint8)
        bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if bgr is None:
            self._json(HTTPStatus.BAD_REQUEST, {"error": "图片无法解码"})
            return
        if bgr.shape[0] * bgr.shape[1] > 25_000_000:
            self._json(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, {"error": "图片像素过大，请缩小后重试"})
            return
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        try:
            result = infer(rgb, models["recognizer"], models["detector"] if mode == "full" else None, self.server.store.device)
        except (RuntimeError, ValueError) as exc:
            self._json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"模型推理失败：{exc}"})
            return
        self._json(HTTPStatus.OK, result)


def main() -> None:
    parser = argparse.ArgumentParser(description="Start the local LPR learning page")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--recognizer", default="outputs/recognizer.pt")
    parser.add_argument("--detector", default="outputs/detector.pt")
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps"))
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    store = ModelStore(args.recognizer, args.detector, args.device)
    with LPRServer((args.host, args.port), store) as server:
        print(f"LPR page: http://{args.host}:{server.server_port}", flush=True)
        print("The page will load model files automatically when they appear.", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
