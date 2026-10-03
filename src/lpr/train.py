"""Train the recognition CNN or a single-class plate detector."""

import argparse
import json

import torch
from torch.utils.data import DataLoader

from .ccpd import assert_disjoint, load_split
from .datasets import DetectionDataset, RecognitionDataset, detection_collate
from .models import PlateCNN, make_detector, predicted_indices, recognition_loss
from .runtime import box_iou, save_checkpoint, seed_everything, top_detection


@torch.no_grad()
def score_recognizer(model, loader, device: str) -> dict:
    model.eval()
    characters = plates = count = 0
    for images, targets in loader:
        predictions = predicted_indices(model(images.to(device))).cpu()
        equal = predictions.eq(targets)
        characters += int(equal.sum())
        plates += int(equal.all(dim=1).sum())
        count += len(targets)
    return {"character_accuracy": characters / (7 * count), "plate_accuracy": plates / count, "samples": count}


@torch.no_grad()
def score_detector(model, loader, device: str, threshold: float = 0.5) -> dict:
    model.eval()
    found = quality = iou_sum = count = 0
    for images, targets in loader:
        predictions = model([image.to(device) for image in images])
        for prediction, target in zip(predictions, targets):
            count += 1
            selected = top_detection(prediction, threshold)
            if selected is None:
                continue
            found += 1
            iou = box_iou(selected[0], target["boxes"][0])
            iou_sum += iou
            quality += iou >= 0.5
    return {"detection_rate": found / count, "iou_at_0.5": quality / count, "mean_iou": iou_sum / count, "samples": count}


def train_recognizer(args, train_records, val_records) -> None:
    train_loader = DataLoader(RecognitionDataset(train_records), batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(RecognitionDataset(val_records), batch_size=args.batch_size, num_workers=args.workers)
    model = PlateCNN().to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for images, targets in train_loader:
            images, targets = images.to(args.device), targets.to(args.device)
            optimizer.zero_grad()
            loss = recognition_loss(model(images), targets)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(targets)
        metrics = score_recognizer(model, val_loader, args.device)
        print(json.dumps({"epoch": epoch, "train_loss": total_loss / len(train_records), "validation": metrics}, ensure_ascii=False))
        if metrics["plate_accuracy"] > best:
            best = metrics["plate_accuracy"]
            save_checkpoint(args.output, model, "recognizer", epoch=epoch, seed=args.seed, validation=metrics)


def train_detector(args, train_records, val_records) -> None:
    train_loader = DataLoader(DetectionDataset(train_records), batch_size=args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=detection_collate)
    val_loader = DataLoader(DetectionDataset(val_records), batch_size=1, num_workers=args.workers, collate_fn=detection_collate)
    model = make_detector(pretrained_backbone=args.pretrained_backbone).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    best = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for images, targets in train_loader:
            images = [image.to(args.device) for image in images]
            targets = [{key: value.to(args.device) for key, value in target.items()} for target in targets]
            optimizer.zero_grad()
            losses = model(images, targets)
            loss = sum(losses.values())
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite detection loss: {losses}")
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach()) * len(images)
        metrics = score_detector(model, val_loader, args.device)
        print(json.dumps({"epoch": epoch, "train_loss": total_loss / len(train_records), "validation": metrics}, ensure_ascii=False))
        if metrics["iou_at_0.5"] > best:
            best = metrics["iou_at_0.5"]
            save_checkpoint(args.output, model, "detector", epoch=epoch, seed=args.seed, validation=metrics)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a CCPD license plate model")
    parser.add_argument("kind", choices=("recognizer", "detector"))
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--train", required=True, help="official CCPD train.txt")
    parser.add_argument("--val", required=True, help="official CCPD val.txt")
    parser.add_argument("--output", required=True, help="best checkpoint path")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--limit", type=int, help="first N entries per split for a smoke run")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps"))
    parser.add_argument("--pretrained-backbone", action="store_true", help="download ImageNet backbone weights for detector")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.lr <= 0 or args.workers < 0:
        parser.error("epochs, batch-size, lr must be positive; workers must be nonnegative")
    seed_everything(args.seed)
    train_records = load_split(args.data_root, args.train, args.limit)
    val_records = load_split(args.data_root, args.val, args.limit)
    assert_disjoint(train_records, val_records)
    print(json.dumps({"kind": args.kind, "train": len(train_records), "val": len(val_records), "device": args.device}))
    if args.kind == "recognizer":
        train_recognizer(args, train_records, val_records)
    else:
        train_detector(args, train_records, val_records)


if __name__ == "__main__":
    main()
