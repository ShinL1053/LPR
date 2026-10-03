"""Shared checkpoint, seeding, and detection utilities."""

import random
import os
import tempfile
from pathlib import Path

import numpy as np
import torch

from .models import PlateCNN, make_detector


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(path: str | Path, model, kind: str, **metadata) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent, delete=False) as handle:
            temporary = Path(handle.name)
            torch.save({"kind": kind, "state_dict": model.state_dict(), "metadata": metadata}, handle)
        os.replace(temporary, destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def load_checkpoint(path: str | Path, expected_kind: str, device: str):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint.get("kind") != expected_kind:
        raise ValueError(f"expected {expected_kind} checkpoint: {path}")
    model = PlateCNN() if expected_kind == "recognizer" else make_detector()
    model.load_state_dict(checkpoint["state_dict"])
    return model.to(device).eval(), checkpoint.get("metadata", {})


def box_iou(box_a, box_b) -> float:
    ax1, ay1, ax2, ay2 = (float(value) for value in box_a)
    bx1, by1, bx2, by2 = (float(value) for value in box_b)
    intersection = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


def top_detection(prediction: dict, score_threshold: float):
    if len(prediction["scores"]) == 0 or float(prediction["scores"][0]) < score_threshold:
        return None
    return prediction["boxes"][0].detach().cpu().tolist(), float(prediction["scores"][0])
