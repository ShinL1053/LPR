"""Extend the practice training set and reserve a new plate-disjoint check set."""

import argparse
import json
import random
import zipfile
import zlib
from collections import defaultdict
from pathlib import Path

from lpr.ccpd import parse_filename

from prepare_hf_ccpd import extract, image_entries, source_id, split_file, validate_image


def read_records(split_dir: Path, image_dir: Path, name: str):
    return [parse_filename(image_dir / line) for line in (split_dir / f"{name}.txt").read_text(encoding="utf-8").splitlines() if line]


def matches_archive_entry(path: Path, info: zipfile.ZipInfo) -> bool:
    if path.stat().st_size != info.file_size:
        return False
    checksum = 0
    with path.open("rb") as image:
        for chunk in iter(lambda: image.read(1024 * 1024), b""):
            checksum = zlib.crc32(chunk, checksum)
    return checksum == info.CRC


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/hf-ccpd"))
    parser.add_argument("--final-count", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    if args.final_count < 1:
        parser.error("final-count must be positive")

    image_dir = args.data_root / "images"
    split_dir = args.data_root / "split"
    existing = {name: read_records(split_dir, image_dir, name) for name in ("train-balanced", "val-balanced", "test")}
    blocked_ids = {source_id(record.path.name) for records in existing.values() for record in records}
    heldout_plates = {record.plate for name in ("val-balanced", "test") for record in existing[name]}
    training_plates = {record.plate for record in existing["train-balanced"]}

    with zipfile.ZipFile(args.archive) as archive:
        candidates, non_seven_character = image_entries(archive)
        groups = defaultdict(list)
        invalid = 0
        for entry in candidates:
            _, record = entry
            if source_id(record.path.name) in blocked_ids or record.plate in heldout_plates:
                continue
            try:
                validate_image(archive, entry)
            except ValueError:
                invalid += 1
                continue
            groups[record.plate].append(entry)

        available_for_final = [group for plate, group in groups.items() if plate not in training_plates]
        random.Random(args.seed).shuffle(available_for_final)
        final_groups = []
        final_size = 0
        for group in available_for_final:
            if final_size >= args.final_count:
                break
            final_groups.append(group)
            final_size += len(group)
        if final_size < args.final_count:
            raise ValueError(f"only {final_size} plate-disjoint images are available for final evaluation")

        final_plates = {group[0][1].plate for group in final_groups}
        final = [entry for group in final_groups for entry in group]
        additional_train = [entry for plate, group in groups.items() if plate not in final_plates for entry in group]
        missing = []
        for info, record in final + additional_train:
            destination = image_dir / record.path.name
            if destination.exists():
                if not matches_archive_entry(destination, info):
                    raise ValueError(f"existing image differs from source archive: {destination}")
            else:
                missing.append((info, record))
        extract(archive, missing, image_dir)

    training_names = [record.path.name for record in existing["train-balanced"]]
    (split_dir / "train-full.txt").write_text("".join(name + "\n" for name in training_names), encoding="utf-8")
    with (split_dir / "train-full.txt").open("a", encoding="utf-8") as handle:
        handle.writelines(record.path.name + "\n" for _, record in additional_train)
    split_file(split_dir / "final-independent.txt", final)
    report = {
        "source": str(args.archive), "seed": args.seed,
        "samples": {"train": len(training_names) + len(additional_train), "validation": len(existing["val-balanced"]), "development": len(existing["test"]), "final": len(final)},
        "new_train": len(additional_train), "source_non_seven_character": non_seven_character,
        "skipped_invalid_image_or_bounds": invalid,
        "final_plate_overlap_with_training": len(final_plates & (training_plates | {entry[1].plate for entry in additional_train})),
    }
    (args.data_root / "manifest-full.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
