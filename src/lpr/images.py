"""Image loading and consistent plate preprocessing."""

from pathlib import Path

import cv2
import numpy as np

PLATE_WIDTH = 192
PLATE_HEIGHT = 64


def read_rgb(path: str | Path) -> np.ndarray:
    # imdecode handles Chinese characters in paths on every supported platform.
    encoded = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot decode image: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def write_rgb(path: str | Path, image: np.ndarray) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    suffix = destination.suffix or ".jpg"
    ok, encoded = cv2.imencode(suffix, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not ok:
        raise ValueError(f"cannot encode image: {destination}")
    encoded.tofile(str(destination))


def crop_box(image: np.ndarray, box: tuple[int, int, int, int], inclusive: bool = False) -> np.ndarray:
    x1, y1, x2, y2 = (int(value) for value in box)
    if inclusive:
        x2 += 1
        y2 += 1
    height, width = image.shape[:2]
    if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
        raise ValueError(f"box {box} is outside {width}x{height} image")
    return image[y1:y2, x1:x2].copy()


def normalize_plate(image: np.ndarray) -> np.ndarray:
    """Resize without stretching and pad to a fixed CNN input size."""
    height, width = image.shape[:2]
    if height < 1 or width < 1:
        raise ValueError("empty plate image")
    scale = min(PLATE_WIDTH / width, PLATE_HEIGHT / height)
    new_width = max(1, round(width * scale))
    new_height = max(1, round(height * scale))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    canvas = np.full((PLATE_HEIGHT, PLATE_WIDTH, 3), 114, dtype=np.uint8)
    left = (PLATE_WIDTH - new_width) // 2
    top = (PLATE_HEIGHT - new_height) // 2
    canvas[top:top + new_height, left:left + new_width] = resized
    return canvas


def draw_box(image: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    marked = image.copy()
    x1, y1, x2, y2 = (int(value) for value in box)
    cv2.rectangle(marked, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return marked

