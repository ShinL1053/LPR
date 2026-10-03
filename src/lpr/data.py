"""Inspect CCPD partitions before training."""

import argparse
import json

from .ccpd import assert_disjoint, load_split, parse_filename
from .images import read_rgb


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect CCPD data and split integrity")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--train", required=True, help="CCPD train.txt")
    parser.add_argument("--val", required=True, help="CCPD val.txt")
    parser.add_argument("--test", help="optional held-out CCPD test.txt")
    parser.add_argument("--limit", type=int, help="limit entries per split for a smoke run")
    parser.add_argument("--verify-images", action="store_true", help="decode images and check annotation bounds")
    args = parser.parse_args()

    files = {"train": args.train, "val": args.val}
    if args.test:
        files["test"] = args.test
    groups = {name: load_split(args.data_root, split_file, args.limit) for name, split_file in files.items()}
    assert_disjoint(*groups.values())
    if args.verify_images:
        for records in groups.values():
            for record in records:
                image = read_rgb(record.path)
                parse_filename(record.path, (image.shape[1], image.shape[0]))
    plates = {name: {record.plate for record in records} for name, records in groups.items()}
    report = {
        "samples": {name: len(records) for name, records in groups.items()},
        "unique_plates": {name: len(values) for name, values in plates.items()},
        "shared_plate_numbers": {
            f"{left}/{right}": len(plates[left] & plates[right])
            for left in plates for right in plates if left < right
        },
        "image_bounds_checked": bool(args.verify_images),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

