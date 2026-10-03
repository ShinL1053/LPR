import tempfile
import unittest
from pathlib import Path

from lpr.ccpd import assert_disjoint, decode_indices, load_split, parse_filename


NAME = "025-95_113-154&383_386&473-386&473_177&454_154&383_363&402-0_0_22_27_27_33_16-37-15.jpg"


class CCPDTests(unittest.TestCase):
    def test_official_example(self):
        record = parse_filename(NAME, (600, 600))
        self.assertEqual(record.box, (154, 383, 386, 473))
        self.assertEqual(record.indices, (0, 0, 22, 27, 27, 33, 16))
        self.assertEqual(record.plate, decode_indices(list(record.indices)))
        self.assertTrue(record.plate.startswith("皖A"))
        self.assertEqual(len(record.plate), 7)

    def test_invalid_annotations_are_rejected(self):
        with self.assertRaises(ValueError):
            parse_filename("not-a-ccpd-image.jpg")
        with self.assertRaises(ValueError):
            parse_filename(NAME, (386, 474))
        with self.assertRaises(ValueError):
            parse_filename(NAME.replace("0_0_22_27_27_33_16", "0_0_22_27_27_33_16_0"))
        with self.assertRaises(ValueError):
            parse_filename(NAME.replace("154&383_386&473", "386&473_154&383"))

    def test_split_resolution_and_overlap(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "ccpd_base" / NAME
            image.parent.mkdir()
            image.touch()
            split = root / "train.txt"
            split.write_text(f"ccpd_base/{NAME}\n", encoding="utf-8")
            records = load_split(root, split)
            self.assertEqual(records[0].path, image.resolve())
            with self.assertRaises(ValueError):
                assert_disjoint(records, records)
            with self.assertRaises(ValueError):
                load_split(root, split, limit=0)


if __name__ == "__main__":
    unittest.main()

