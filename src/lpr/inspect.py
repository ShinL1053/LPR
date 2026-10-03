"""Save intermediate CNN feature maps for learning and debugging."""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from .datasets import image_tensor
from .images import normalize_plate, read_rgb, write_rgb
from .runtime import load_checkpoint


LAYERS = {"conv1_relu": 2, "conv2_relu": 6, "conv3_relu": 10}


def feature_grid(features: torch.Tensor, max_channels: int = 16) -> np.ndarray:
    """Render feature channels independently, so patterns are visible."""
    channels = features[0].detach().cpu().numpy()[:max_channels]
    tile_width, tile_height = 96, 48
    grid = np.full((4 * tile_height, 4 * tile_width, 3), 25, dtype=np.uint8)
    for index, channel in enumerate(channels):
        low, high = float(channel.min()), float(channel.max())
        if high > low:
            channel = (channel - low) / (high - low)
        else:
            channel = np.zeros_like(channel)
        tile = cv2.resize((channel * 255).astype(np.uint8), (tile_width, tile_height))
        row, column = divmod(index, 4)
        grid[row * tile_height:(row + 1) * tile_height, column * tile_width:(column + 1) * tile_width] = np.repeat(tile[:, :, None], 3, axis=2)
    return grid


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect CNN activations from a cropped plate")
    parser.add_argument("--image", required=True, help="cropped plate image")
    parser.add_argument("--recognizer", required=True, help="trained recognition checkpoint")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps"))
    args = parser.parse_args()
    model, _ = load_checkpoint(args.recognizer, "recognizer", args.device)
    activations = {}
    handles = [model.features[index].register_forward_hook(lambda _module, _input, output, name=name: activations.update({name: output.detach()})) for name, index in LAYERS.items()]
    try:
        image = normalize_plate(read_rgb(args.image))
        with torch.no_grad():
            model(image_tensor(image).unsqueeze(0).to(args.device))
    finally:
        for handle in handles:
            handle.remove()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_rgb(output_dir / "input.png", image)
    shapes = {}
    for name, features in activations.items():
        write_rgb(output_dir / f"{name}.png", feature_grid(features))
        shapes[name] = list(features.shape)
    print(json.dumps({"input_shape": [1, 3, 64, 192], "feature_shapes": shapes, "output_dir": str(output_dir)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
