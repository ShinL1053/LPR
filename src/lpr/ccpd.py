"""CCPD filename annotations and split files.

Reference: https://github.com/detectRecog/CCPD#dataset-annotations
"""

from dataclasses import dataclass
from pathlib import Path

PROVINCES = tuple("皖沪津渝冀晋蒙辽吉黑苏浙京闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新") + ("警", "学")
LETTERS = tuple("ABCDEFGHJKLMNPQRSTUVWXYZ")
ALNUM = LETTERS + tuple("0123456789")
VOCABS = (PROVINCES, LETTERS) + (ALNUM,) * 5


def decode_indices(indices: tuple[int, ...] | list[int]) -> str:
    if len(indices) != 7 or any(index < 0 or index >= len(VOCABS[pos]) for pos, index in enumerate(indices)):
        raise ValueError("invalid seven-character plate indices")
    return "".join(VOCABS[pos][index] for pos, index in enumerate(indices))


@dataclass(frozen=True)
class PlateRecord:
    path: Path
    box: tuple[int, int, int, int]  # left, top, right, bottom; inclusive in CCPD
    corners: tuple[tuple[int, int], ...]
    indices: tuple[int, ...]

    @property
    def plate(self) -> str:
        return decode_indices(self.indices)


def _point(value: str) -> tuple[int, int]:
    # The official CCPD archive uses "&"; some redistributed images use ",".
    separator = "&" if "&" in value else ","
    parts = value.split(separator)
    if len(parts) != 2:
        raise ValueError(f"invalid coordinate: {value}")
    x, y = (int(part) for part in parts)
    if x < 0 or y < 0:
        raise ValueError(f"negative coordinate: {value}")
    return x, y


def parse_filename(path: str | Path, image_size: tuple[int, int] | None = None) -> PlateRecord:
    """Read a seven-character CCPD label without opening the image.

``image_size`` is (width, height) and enables bounds checking.
"""
    file_path = Path(path)
    fields = file_path.stem.split("-")
    if len(fields) != 7:
        raise ValueError(f"expected 7 CCPD filename fields: {file_path.name}")
    try:
        top_left, bottom_right = (_point(p) for p in fields[2].split("_"))
        corners = tuple(_point(p) for p in fields[3].split("_"))
        indices = tuple(int(value) for value in fields[4].split("_"))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"invalid CCPD annotation: {file_path.name}") from exc
    if len(corners) != 4 or len(indices) != 7:
        raise ValueError(f"expected 4 corners and 7 characters: {file_path.name}")
    if any(index < 0 or index >= len(VOCABS[position]) for position, index in enumerate(indices)):
        raise ValueError(f"character index outside seven-character vocabulary: {file_path.name}")
    x1, y1 = top_left
    x2, y2 = bottom_right
    if x1 >= x2 or y1 >= y2:
        raise ValueError(f"empty or reversed bounding box: {file_path.name}")
    if image_size is not None:
        width, height = image_size
        if width <= x2 or height <= y2 or any(x >= width or y >= height for x, y in corners):
            raise ValueError(f"annotation outside image: {file_path.name}")
    return PlateRecord(file_path, (x1, y1, x2, y2), corners, indices)


def load_split(data_root: str | Path, split_file: str | Path, limit: int | None = None) -> list[PlateRecord]:
    """Resolve CCPD split entries against an extracted dataset directory."""
    root = Path(data_root).expanduser().resolve()
    split = Path(split_file).expanduser()
    if not root.is_dir():
        raise FileNotFoundError(f"data directory does not exist: {root}")
    if not split.is_file():
        raise FileNotFoundError(f"split file does not exist: {split}")
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")

    entries = [line.strip() for line in split.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not entries:
        raise ValueError(f"split file is empty: {split}")
    if limit is not None:
        entries = entries[:limit]

    records: list[PlateRecord] = []
    seen: set[Path] = set()
    image_index: dict[str, Path] | None = None
    for entry in entries:
        candidate = root / entry
        if not candidate.is_file():
            if image_index is None:
                image_index = {}
                for image in root.rglob("*.jpg"):
                    if image.name in image_index:
                        raise ValueError(f"duplicate image filename under data root: {image.name}")
                    image_index[image.name] = image
            candidate = image_index.get(Path(entry).name, candidate)
        if not candidate.is_file():
            raise FileNotFoundError(f"split entry missing from data directory: {entry}")
        candidate = candidate.resolve()
        if candidate in seen:
            raise ValueError(f"duplicate split entry: {entry}")
        seen.add(candidate)
        records.append(parse_filename(candidate))
    return records


def assert_disjoint(*groups: list[PlateRecord]) -> None:
    """Prevent the same image from entering more than one data partition."""
    seen: set[str] = set()
    for group in groups:
        names = {record.path.name for record in group}
        overlap = seen & names
        if overlap:
            raise ValueError(f"images occur in more than one split: {sorted(overlap)[:3]}")
        seen.update(names)
