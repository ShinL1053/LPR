"""Fine-tune only the province head with a user-verified plate example."""

import argparse
import copy
import json
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from lpr.ccpd import PROVINCES, assert_disjoint, load_split
from lpr.datasets import RecognitionDataset, image_tensor
from lpr.images import crop_box, normalize_plate, read_rgb
from lpr.runtime import load_checkpoint, save_checkpoint, seed_everything


@torch.no_grad()
def features_for_records(model, records, device, batch_size):
    features = []
    labels = []
    for images, targets in DataLoader(RecognitionDataset(records), batch_size=batch_size):
        features.append(model.shared(model.features(images.to(device))).cpu())
        labels.append(targets[:, 0])
    return torch.cat(features), torch.cat(labels)


@torch.no_grad()
def features_for_correction(model, image, box, device):
    height, width = image.shape[:2]
    variants = []
    for dx in (-4, 0, 4):
        for dy in (-4, 0, 4):
            x1, y1, x2, y2 = box
            shifted = (max(0, x1 + dx), max(0, y1 + dy), min(width, x2 + dx), min(height, y2 + dy))
            crop = normalize_plate(crop_box(image, shifted))
            for brightness in (0.8, 1.0, 1.2):
                variants.append(image_tensor(np.clip(crop.astype(np.float32) * brightness, 0, 255).astype(np.uint8)))
    original = image_tensor(normalize_plate(crop_box(image, box))).unsqueeze(0)
    all_features = model.shared(model.features(torch.stack(variants).to(device))).cpu()
    original_feature = model.shared(model.features(original.to(device))).cpu()
    return all_features, original_feature


@torch.no_grad()
def score(head, features, labels):
    head.eval()
    guesses = head(features).argmax(dim=1)
    correct = guesses.eq(labels)
    non_anhui = labels.ne(PROVINCES.index("皖"))
    return {
        "province_accuracy": float(correct.float().mean()),
        "non_anhui_province_accuracy": float(correct[non_anhui].float().mean()),
    }


def validate_splits(train, val):
    """Keep validation images and plate numbers out of correction training."""
    assert_disjoint(train, val)
    shared_plates = {record.plate for record in train} & {record.plate for record in val}
    if shared_plates:
        raise ValueError(f"train and validation share plate numbers: {sorted(shared_plates)[:3]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--recognizer", required=True)
    parser.add_argument("--correction-image", required=True)
    parser.add_argument("--correction-box", nargs=4, type=int, required=True, metavar=("X1", "Y1", "X2", "Y2"))
    parser.add_argument("--province", choices=PROVINCES, required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    seed_everything(args.seed)
    train = load_split(args.data_root, args.train)
    val = load_split(args.data_root, args.val)
    validate_splits(train, val)
    model, _ = load_checkpoint(args.recognizer, "recognizer", args.device)
    model.eval()
    print("extracting fixed CNN features", flush=True)
    train_features, train_labels = features_for_records(model, train, args.device, args.batch_size)
    val_features, val_labels = features_for_records(model, val, args.device, args.batch_size)
    correction_image = read_rgb(args.correction_image)
    correction_features, original_feature = features_for_correction(model, correction_image, tuple(args.correction_box), args.device)
    province_index = PROVINCES.index(args.province)
    labels = torch.cat((train_labels, torch.full((len(correction_features),), province_index)))
    features = torch.cat((train_features, correction_features))
    counts = Counter(train_labels.tolist())
    weights = [counts[int(label)] ** -0.5 for label in train_labels]
    weights += [0.5] * len(correction_features)
    sampler = WeightedRandomSampler(weights, len(features), replacement=True, generator=torch.Generator().manual_seed(args.seed))
    loader = DataLoader(TensorDataset(features, labels), batch_size=512, sampler=sampler)
    head = copy.deepcopy(model.heads[0]).cpu()
    initial = [parameter.detach().clone() for parameter in head.parameters()]
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.001)
    baseline = score(head, val_features, val_labels)
    print(json.dumps({"baseline": baseline, "correction_prediction": PROVINCES[int(head(original_feature).argmax())]}, ensure_ascii=False), flush=True)
    best = (-1.0, -1.0)
    for epoch in range(1, args.epochs + 1):
        head.train()
        for batch_features, batch_labels in loader:
            optimizer.zero_grad()
            penalty = sum((parameter - old).square().mean() for parameter, old in zip(head.parameters(), initial))
            loss = torch.nn.functional.cross_entropy(head(batch_features), batch_labels) + 0.01 * penalty
            loss.backward()
            optimizer.step()
        metrics = score(head, val_features, val_labels)
        with torch.no_grad():
            probabilities = head(original_feature).softmax(dim=1)[0]
        predicted = PROVINCES[int(probabilities.argmax())]
        report = {"epoch": epoch, "validation": metrics, "correction_prediction": predicted, "correction_score": float(probabilities[province_index])}
        print(json.dumps(report, ensure_ascii=False), flush=True)
        acceptable = metrics["province_accuracy"] >= baseline["province_accuracy"] - 0.02 and metrics["non_anhui_province_accuracy"] >= baseline["non_anhui_province_accuracy"] - 0.02
        grade = (metrics["province_accuracy"] + metrics["non_anhui_province_accuracy"], metrics["non_anhui_province_accuracy"])
        if predicted == args.province and acceptable and grade > best:
            best = grade
            model.heads[0].load_state_dict(head.state_dict())
            save_checkpoint(args.output, model, "recognizer", epoch=epoch, correction_province=args.province, validation=metrics)
    if best[0] < 0:
        print("no candidate passed both validation gates and the correction example", flush=True)


if __name__ == "__main__":
    main()
