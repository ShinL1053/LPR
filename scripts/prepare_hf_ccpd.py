"""Build reproducible seven-character practice splits from the CCPD mirror archives.

The source archives are public files from:
https://huggingface.co/datasets/zenitsu09/ccpd-100k-yolo
"""

import argparse
import json
import os
import random
import shutil
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np

from lpr.ccpd import parse_filename


def image_entries(archive: zipfile.ZipFile):
    valid = []
    excluded = 0
    for info in archive.infolist():
        if info.is_dir() or not info.filename.lower().endswith(".jpg"):
            continue
        name = Path(info.filename).name
        try:
            record = parse_filename(name)
        except ValueError:
            excluded += 1
            continue
        valid.append((info, record))
    return valid, excluded


def source_id(name: str) -> str:
    """Preserve the original CCPD subset and image number across mirror renames."""
    stem = Path(name).stem
    return stem.split("_ccpd_", 1)[-1] if "_ccpd_" in stem else stem


def validate_image(archive: zipfile.ZipFile, entry) -> None:
    info, record = entry
    image = cv2.imdecode(np.frombuffer(archive.read(info), dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot decode image: {info.filename}")
    parse_filename(record.path, (image.shape[1], image.shape[0]))


def take_valid(archive: zipfile.ZipFile, entries, count: int, blocked_ids=None, blocked_plates=None):
    selected = []
    invalid = 0
    blocked_ids = blocked_ids or set()
    blocked_plates = blocked_plates or set()
    for position, entry in enumerate(entries):
        _, record = entry
        if source_id(record.path.name) in blocked_ids or record.plate in blocked_plates:
            continue
        try:
            validate_image(archive, entry)
        except ValueError:
            invalid += 1
            continue
        selected.append(entry)
        if len(selected) == count:
            return selected, entries[position + 1:], invalid
    raise ValueError(f"source archive lacks {count} valid disjoint seven-character images")


def extract(archive: zipfile.ZipFile, entries, image_dir: Path):
    for info, record in entries:
        destination = image_dir / record.path.name
        with tempfile.NamedTemporaryFile(prefix=".ccpd-", dir=image_dir, delete=False) as output:
            temporary = Path(output.name)
            try:
                with archive.open(info) as source:
                    shutil.copyfileobj(source, output)
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        os.replace(temporary, destination)


def split_file(path: Path, entries):
    path.write_text("".join(record.path.name + "\n" for _, record in entries), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Prepare a bounded real CCPD practice set")
    parser.add_argument("--train-archive", type=Path, required=True)
    parser.add_argument("--test-archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/hf-ccpd"))
    parser.add_argument("--train-count", type=int, default=20_000)
    parser.add_argument("--val-count", type=int, default=1_000)
    parser.add_argument("--detector-train-count", type=int, default=1_000)
    parser.add_argument("--detector-val-count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.train_count, args.val_count, args.detector_train_count, args.detector_val_count) < 1:
        parser.error("all split sizes must be positive")
    if args.detector_train_count > args.train_count or args.detector_val_count > args.val_count:
        parser.error("detector split sizes must fit inside the recognition splits")

    image_dir = args.output / "images"
    split_dir = args.output / "split"
    image_dir.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.train_archive) as train_archive, zipfile.ZipFile(args.test_archive) as test_archive:
        candidates, train_excluded = image_entries(train_archive)
        test_candidates, test_excluded = image_entries(test_archive)
        test = []
        test_invalid = 0
        for entry in test_candidates:
            try:
                validate_image(test_archive, entry)
            except ValueError:
                test_invalid += 1
                continue
            test.append(entry)
        if len({record.path.name for _, record in candidates + test}) != len(candidates) + len(test):
            raise ValueError("duplicate image basenames across source archives")
        test_ids = {source_id(record.path.name) for _, record in test}
        test_plates = {record.plate for _, record in test}
        overlapping_images = sum(source_id(record.path.name) in test_ids for _, record in candidates)
        overlapping_plates = sum(record.plate in test_plates for _, record in candidates)
        eligible = [entry for entry in candidates if source_id(entry[1].path.name) not in test_ids and entry[1].plate not in test_plates]
        random.Random(args.seed).shuffle(eligible)
        train, remaining, train_invalid = take_valid(train_archive, eligible, args.train_count)
        train_plates = {record.plate for _, record in train}
        train_ids = {source_id(record.path.name) for _, record in train}
        val, _, val_invalid = take_valid(train_archive, remaining, args.val_count, train_ids, train_plates)
        if not test:
            raise ValueError("source archives lack enough disjoint seven-character images")
        extract(train_archive, train + val, image_dir)
        extract(test_archive, test, image_dir)

    split_file(split_dir / "train.txt", train)
    split_file(split_dir / "val.txt", val)
    split_file(split_dir / "test.txt", test)
    split_file(split_dir / "detector-train.txt", train[:args.detector_train_count])
    split_file(split_dir / "detector-val.txt", val[:args.detector_val_count])
    report = {
        "source": "https://huggingface.co/datasets/zenitsu09/ccpd-100k-yolo",
        "seed": args.seed,
        "source_seven_character": {"train_archive": len(candidates), "test_archive": len(test)},
        "source_excluded_non_seven_character": {"train_archive": train_excluded, "test_archive": test_excluded},
        "source_excluded_invalid_image_or_bounds": {"selected_train": train_invalid, "selected_val": val_invalid, "test_archive": test_invalid},
        "source_train_images_matching_test_ids": overlapping_images,
        "source_train_images_matching_test_plates": overlapping_plates,
        "splits": {"train": len(train), "val": len(val), "test": len(test), "detector_train": args.detector_train_count, "detector_val": args.detector_val_count},
        "plate_overlap": {"train_val": len(train_plates & {record.plate for _, record in val}), "train_test": len(train_plates & test_plates), "val_test": len({record.plate for _, record in val} & test_plates)},
    }
    (args.output / "manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
