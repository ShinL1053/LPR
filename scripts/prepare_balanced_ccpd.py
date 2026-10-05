"""Add held-out non-Anhui CCPD examples to the existing seven-character splits."""

import argparse
import json
import random
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from lpr.ccpd import parse_filename

from prepare_hf_ccpd import extract, image_entries, source_id, split_file, validate_image


def read_split(split_dir: Path, image_dir: Path, name: str):
    names = (split_dir / f"{name}.txt").read_text(encoding="utf-8").splitlines()
    return [(None, parse_filename(image_dir / entry)) for entry in names]


def main():
    parser = argparse.ArgumentParser(description="Prepare province-balanced CCPD practice splits")
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/hf-ccpd"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    image_dir = args.data_root / "images"
    split_dir = args.data_root / "split"
    original = {name: read_split(split_dir, image_dir, name) for name in ("train", "val", "test")}
    existing = [record for group in original.values() for _, record in group]
    blocked_ids = {source_id(record.path.name) for record in existing}
    blocked_plates = {record.plate for record in existing}

    with zipfile.ZipFile(args.archive) as archive:
        candidates, _ = image_entries(archive)
        by_province = defaultdict(list)
        invalid = 0
        for entry in candidates:
            _, record = entry
            if record.indices[0] == 0 or source_id(record.path.name) in blocked_ids or record.plate in blocked_plates:
                continue
            try:
                validate_image(archive, entry)
            except ValueError:
                invalid += 1
                continue
            by_province[record.indices[0]].append(entry)

        rng = random.Random(args.seed)
        extra_train = []
        extra_val = []
        for province in sorted(by_province):
            by_plate = defaultdict(list)
            for entry in by_province[province]:
                by_plate[entry[1].plate].append(entry)
            groups = list(by_plate.values())
            rng.shuffle(groups)
            val_target = round(len(by_province[province]) * 0.2)
            val_count = 0
            for group in groups:
                if val_count < val_target:
                    extra_val.extend(group)
                    val_count += len(group)
                else:
                    extra_train.extend(group)
        extract(archive, extra_train + extra_val, image_dir)

    train = original["train"] + extra_train
    val = original["val"] + extra_val
    test = original["test"]
    groups = [{record.plate for _, record in entries} for entries in (train, val, test)]
    if groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2]:
        raise ValueError("plate numbers overlap across balanced splits")
    split_file(split_dir / "train-balanced.txt", train)
    split_file(split_dir / "val-balanced.txt", val)
    report = {
        "source": "https://huggingface.co/datasets/zenitsu09/ccpd-100k-yolo",
        "seed": args.seed,
        "added_non_anhui": {"train": len(extra_train), "val": len(extra_val)},
        "skipped_invalid_image_or_bounds": invalid,
        "samples": {"train": len(train), "val": len(val), "test": len(test)},
        "non_anhui_samples": {
            "train": sum(record.indices[0] != 0 for _, record in train),
            "val": sum(record.indices[0] != 0 for _, record in val),
            "test": sum(record.indices[0] != 0 for _, record in test),
        },
        "train_provinces": dict(Counter(str(record.indices[0]) for _, record in train)),
    }
    (args.data_root / "manifest-balanced.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
