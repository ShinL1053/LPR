"""Run trained models on a plate crop or an entire vehicle image."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

from .ccpd import decode_indices
from .datasets import image_tensor
from .images import blue_plate_box, crop_box, draw_box, normalize_plate, read_rgb, write_rgb
from .models import predicted_indices
from .runtime import load_checkpoint, top_detection

MIN_BLUE_FALLBACK_SCORE = 0.2


@torch.no_grad()
def infer(image: np.ndarray, recognizer, detector=None, device: str = "cpu", score_threshold: float = 0.5) -> dict:
    if detector is None:
        plate_image = image
        box = None
        detection_score = None
        detection_source = None
    else:
        detection = detector([image_tensor(image).to(device)])[0]
        selected = top_detection(detection, score_threshold)
        if selected is None:
            candidate = top_detection(detection, MIN_BLUE_FALLBACK_SCORE)
            raw_box = blue_plate_box(image, tuple(int(value) for value in candidate[0])) if candidate else None
            if raw_box is None:
                return {"status": "no_plate", "reason": "detector found no blue plate with positional support"}
            detection_score = None
            detection_source = "blue_color"
        else:
            raw_box, detection_score = selected
            detection_source = "cnn"
            nearby_blue = blue_plate_box(image, tuple(int(value) for value in raw_box))
            if nearby_blue is not None:
                # Extend only a clearly clipped side; small color/detector
                # disagreements are common and should leave the crop intact.
                detected_width = raw_box[2] - raw_box[0]
                left_gap = (raw_box[0] - nearby_blue[0]) / detected_width
                right_gap = (nearby_blue[2] - raw_box[2]) / detected_width
                if 0.08 <= left_gap <= 0.2 and abs(right_gap) <= 0.05:
                    raw_box = (nearby_blue[0], raw_box[1], raw_box[2], raw_box[3])
                    detection_source = "cnn_blue_refined"
                elif 0.08 <= right_gap <= 0.2 and abs(left_gap) <= 0.05:
                    raw_box = (raw_box[0], raw_box[1], nearby_blue[2], raw_box[3])
                    detection_source = "cnn_blue_refined"
        height, width = image.shape[:2]
        x1 = max(0, min(width - 1, int(raw_box[0])))
        y1 = max(0, min(height - 1, int(raw_box[1])))
        x2 = max(x1 + 1, min(width, int(raw_box[2] + 0.999)))
        y2 = max(y1 + 1, min(height, int(raw_box[3] + 0.999)))
        box = (x1, y1, x2, y2)
        plate_image = crop_box(image, box)
    plate_input = image_tensor(normalize_plate(plate_image)).unsqueeze(0).to(device)
    logits = recognizer(plate_input)
    indices = predicted_indices(logits)[0].cpu().tolist()
    province_score = float(logits[0].softmax(dim=1)[0].max())
    return {
        "status": "ok",
        "plate": decode_indices(indices),
        "province_uncertain": province_score < 0.6,
        "box": list(box) if box is not None else None,
        "detection_score": detection_score,
        "detection_source": detection_source,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Recognize a Chinese seven-character plate")
    parser.add_argument("--image", required=True)
    parser.add_argument("--recognizer", required=True, help="trained recognizer checkpoint")
    parser.add_argument("--detector", help="trained detector checkpoint; omit for an already-cropped plate")
    parser.add_argument("--save-image", help="write input image with the predicted box")
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps"))
    args = parser.parse_args()
    if not 0 <= args.score_threshold <= 1:
        parser.error("score-threshold must be between 0 and 1")
    recognizer, _ = load_checkpoint(args.recognizer, "recognizer", args.device)
    detector = load_checkpoint(args.detector, "detector", args.device)[0] if args.detector else None
    image = read_rgb(args.image)
    result = infer(image, recognizer, detector, args.device, args.score_threshold)
    result["image"] = str(Path(args.image))
    if args.save_image and result["status"] == "ok":
        output = draw_box(image, tuple(result["box"])) if result["box"] else image
        write_rgb(args.save_image, output)
        result["saved_image"] = str(Path(args.save_image))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["status"] != "ok":
        sys.exit(2)


if __name__ == "__main__":
    main()
