"""Page images: loading, binarization, and the per-voice stack layout."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

_initialized = False


def init_gamera():
    """Register Gamera's plugins exactly once."""
    global _initialized
    if not _initialized:
        from gamera.core import init_gamera as _init  # noqa: PLC0415

        _init()
        _initialized = True


@dataclass
class StackPage:
    path: Path
    image: object  # gamera GREYSCALE image
    slots: list[tuple[int, str]] = field(default_factory=list)

    @property
    def nrows(self) -> int:
        return self.image.nrows

    @property
    def ncols(self) -> int:
        return self.image.ncols

    def binary(self):
        """Otsu-binarized ONEBIT image (black = 1)."""
        init_gamera()
        if self.image.data.pixel_type == 0:  # already ONEBIT
            return self.image
        return self.image.otsu_threshold(0)

    def array(self):
        """Numpy view of the binary image (uint16, 0/1)."""
        return self.binary().to_numpy()

    def save_png(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.image.save_PNG(str(path))


def read_slots(path: Path) -> list[tuple[int, str]]:
    """Rows of a stack .list file: `page slot source-path` (1-based page).

    The list is per voice, e.g. S-01.png uses S.list.
    """
    stem = Path(path).stem
    base = stem.rsplit("-", 1)[0] if "-" in stem else stem
    listing = Path(path).with_name(base + ".list")
    if not listing.exists():
        return []
    rows = []
    for line in listing.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) >= 3:
            rows.append((int(parts[0]), parts[2]))
    return rows


def load_stack(path: str | Path) -> StackPage:
    init_gamera()
    from gamera.core import load_image  # noqa: PLC0415

    page = Path(path)
    image = load_image(str(page))
    return StackPage(page, image, read_slots(page))


def from_array(arr, page: StackPage | None = None):
    """Wrap a numpy 0/1 array as a Gamera ONEBIT image."""
    init_gamera()
    from gamera.plugins import numpy_io  # noqa: PLC0415

    image = numpy_io.from_numpy(arr.astype("uint16"))
    if page is None:
        return image
    return StackPage(page.path, image, page.slots)


def save_array_png(arr, path: Path):
    """Save a 0/1 array as a PNG via Gamera (no PIL dependency for this path)."""
    init_gamera()
    image = from_array(arr)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save_PNG(str(path))
