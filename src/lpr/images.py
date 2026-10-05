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


def blue_plate_box(
    image: np.ndarray, reference_box: tuple[int, int, int, int]
) -> tuple[int, int, int, int] | None:
    """Find a blue plate-shaped region overlapping a detector candidate."""
    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (90, 60, 30), (140, 255, 255))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(3, width // 100), max(3, height // 100)))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        x, y, box_width, box_height = cv2.boundingRect(contour)
        area = box_width * box_height
        aspect = box_width / box_height
        if not (0.003 * width * height <= area <= 0.6 * width * height and 1.5 <= aspect <= 5.5):
            continue
        fill = cv2.countNonZero(mask[y:y + box_height, x:x + box_width]) / area
        if fill < 0.45:
            continue
        rx1, ry1, rx2, ry2 = reference_box
        intersection = max(0, min(x + box_width, rx2) - max(x, rx1)) * max(0, min(y + box_height, ry2) - max(y, ry1))
        union = area + (rx2 - rx1) * (ry2 - ry1) - intersection
        overlap = intersection / union if union > 0 else 0
        if overlap < 0.4:
            continue
        candidates.append((overlap, x, y, box_width, box_height))
    if not candidates:
        return None
    _, x, y, box_width, box_height = max(candidates)
    inset_x = max(1, round(box_width * 0.008))
    inset_y = max(1, round(box_height * 0.015))
    return x + inset_x, y + inset_y, x + box_width - inset_x, y + box_height - inset_y


def draw_box(image: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    marked = image.copy()
    x1, y1, x2, y2 = (int(value) for value in box)
    cv2.rectangle(marked, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return marked
