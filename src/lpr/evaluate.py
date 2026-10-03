"""Evaluate held-out recognition, localization, and end-to-end accuracy."""

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .ccpd import load_split, parse_filename
from .datasets import DetectionDataset, RecognitionDataset, detection_collate
from .images import read_rgb
from .predict import infer
from .runtime import box_iou, load_checkpoint
from .train import score_detector, score_recognizer


def evaluate_end_to_end(records, recognizer, detector, device: str, threshold: float) -> dict:
    samples = len(records)
    found = correct_plates = correct_characters = accurate_boxes = 0
    failures = []
    for record in records:
        image = read_rgb(record.path)
        parse_filename(record.path, (image.shape[1], image.shape[0]))
        result = infer(image, recognizer, detector, device, threshold)
        if result["status"] != "ok":
            failures.append({"image": str(record.path), "expected": record.plate, "reason": "not detected"})
            continue
        found += 1
        predicted = result["plate"]
        correct_plates += predicted == record.plate
        correct_characters += sum(left == right for left, right in zip(predicted, record.plate))
        x1, y1, x2, y2 = record.box
        accurate_boxes += box_iou(result["box"], (x1, y1, x2 + 1, y2 + 1)) >= 0.5
        if predicted != record.plate:
            failures.append({"image": str(record.path), "expected": record.plate, "predicted": predicted})
    return {
        "samples": samples,
        "detection_rate": found / samples,
        "box_iou_at_0.5": accurate_boxes / samples,
        "character_accuracy": correct_characters / (7 * samples),
        "plate_accuracy": correct_plates / samples,
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate models on an untouched CCPD split")
    parser.add_argument("kind", choices=("recognizer", "detector", "end-to-end"))
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--split", required=True, help="held-out CCPD split file")
    parser.add_argument("--recognizer", help="recognizer checkpoint")
    parser.add_argument("--detector", help="detector checkpoint")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--score-threshold", type=float, default=0.5)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps"))
    parser.add_argument("--output", help="optional JSON report path")
    args = parser.parse_args()
    if args.batch_size < 1 or not 0 <= args.score_threshold <= 1:
        parser.error("batch-size must be positive and score-threshold must be between 0 and 1")
    if args.kind in ("recognizer", "end-to-end") and not args.recognizer:
        parser.error("this evaluation requires --recognizer")
    if args.kind in ("detector", "end-to-end") and not args.detector:
        parser.error("this evaluation requires --detector")
    records = load_split(args.data_root, args.split, args.limit)
    if args.kind == "recognizer":
        model, _ = load_checkpoint(args.recognizer, "recognizer", args.device)
        loader = DataLoader(RecognitionDataset(records), batch_size=args.batch_size)
        report = score_recognizer(model, loader, args.device)
    elif args.kind == "detector":
        model, _ = load_checkpoint(args.detector, "detector", args.device)
        loader = DataLoader(DetectionDataset(records), batch_size=1, collate_fn=detection_collate)
        report = score_detector(model, loader, args.device, args.score_threshold)
    else:
        recognizer, _ = load_checkpoint(args.recognizer, "recognizer", args.device)
        detector, _ = load_checkpoint(args.detector, "detector", args.device)
        report = evaluate_end_to_end(records, recognizer, detector, args.device, args.score_threshold)
    report["kind"] = args.kind
    report["split"] = str(Path(args.split))
    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    print(serialized)
    if args.output:
        destination = Path(args.output)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(serialized + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
