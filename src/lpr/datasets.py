"""Torch datasets backed by CCPD annotations."""

import torch
from torch.utils.data import Dataset

from .ccpd import PlateRecord, parse_filename
from .images import crop_box, normalize_plate, read_rgb


def image_tensor(image):
    return torch.from_numpy(image.copy()).permute(2, 0, 1).float().div_(255.0)


class RecognitionDataset(Dataset):
    def __init__(self, records: list[PlateRecord]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image = read_rgb(record.path)
        parse_filename(record.path, (image.shape[1], image.shape[0]))
        plate = crop_box(image, record.box, inclusive=True)
        return image_tensor(normalize_plate(plate)), torch.tensor(record.indices, dtype=torch.long)


class DetectionDataset(Dataset):
    def __init__(self, records: list[PlateRecord]):
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        image = read_rgb(record.path)
        parse_filename(record.path, (image.shape[1], image.shape[0]))
        x1, y1, x2, y2 = record.box
        box = torch.tensor([[x1, y1, x2 + 1, y2 + 1]], dtype=torch.float32)
        target = {
            "boxes": box,
            "labels": torch.ones(1, dtype=torch.int64),
            "image_id": torch.tensor([index]),
            "area": torch.tensor([(x2 + 1 - x1) * (y2 + 1 - y1)], dtype=torch.float32),
            "iscrowd": torch.zeros(1, dtype=torch.int64),
        }
        return image_tensor(image), target


def detection_collate(batch):
    return tuple(zip(*batch))

