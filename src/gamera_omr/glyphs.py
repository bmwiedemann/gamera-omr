"""Glyph extraction, normalization, clustering and contact sheets.

The classifier is gamera's kNN.  Labelling is headless: components are
clustered, cluster representatives are rendered into contact sheets, and the
label mapping is written as JSON (see tools/label_clusters.py).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import segments
from .staffs import Staff, remove_staff_lines


@dataclass
class Glyph:
    uid: str
    page: str
    staff: int
    x0: int
    y0: int
    x1: int
    y1: int
    space: float
    ink: np.ndarray = field(repr=False)  # 2-D uint8, 1 = ink, bbox-sized

    @property
    def w(self) -> int:
        return self.x1 - self.x0 + 1

    @property
    def h(self) -> int:
        return self.y1 - self.y0 + 1

    @property
    def cx(self) -> float:
        return (self.x0 + self.x1) / 2

    @property
    def cy(self) -> float:
        return (self.y0 + self.y1) / 2


def extract(page, staves: list[Staff], page_name: str | None = None,
            *, repair: bool = False) -> list[Glyph]:
    cleaned = remove_staff_lines(page, staves, repair=repair)
    comps = segments.components(cleaned.image)
    name = page_name or page.path.stem
    out = []
    for staff in staves:
        kept, _dropped = segments.filter_components(comps, staff)
        for c in kept:
            ink = np.asarray(c.glyph.to_numpy()).astype("uint8")
            ink = (ink > 0).astype("uint8")
            uid = f"{name}-s{staff.index}-{c.x0}-{c.y0}"
            out.append(Glyph(uid, name, staff.index, c.x0, c.y0, c.x1, c.y1,
                             staff.space, ink))
    return out


def normalize(ink: np.ndarray, space: float, size: int = 64) -> np.ndarray:
    """Scale by staff space, preserving relative size, and pad to a square."""
    h, w = ink.shape
    scale = min(16.0 / max(space, 1.0), (size - 4) / max(h, w, 1))
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    resized = cv2.resize(ink, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size), dtype=np.uint8)
    x = (size - nw) // 2
    y = (size - nh) // 2
    canvas[y:y + nh, x:x + nw] = resized
    return canvas


def kmeans(x: np.ndarray, k: int, iters: int = 40, seed: int = 0):
    """Tiny k-means (k-means++ init) for small glyph sets."""
    rng = np.random.default_rng(seed)
    k = max(1, min(k, len(x)))
    centers = [x[rng.integers(len(x))]]
    for _ in range(1, k):
        d = np.min(((x[:, None, :] - np.array(centers)[None, :, :]) ** 2).sum(-1), axis=1)
        prob = d / d.sum() if d.sum() else np.full(len(x), 1 / len(x))
        centers.append(x[rng.choice(len(x), p=prob)])
    centers = np.array(centers, dtype=np.float32)
    labels = np.zeros(len(x), dtype=int)
    for _ in range(iters):
        d = ((x[:, None, :] - centers[None, :, :]) ** 2).sum(-1)
        new_labels = d.argmin(1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            members = x[labels == j]
            if len(members):
                centers[j] = members.mean(0)
    return labels, centers


def cluster(glyphs: list[Glyph], k: int = 32, seed: int = 0):
    x = np.stack([normalize(g.ink, g.space).reshape(-1) for g in glyphs]).astype(np.float32)
    x /= 255.0
    labels, centers = kmeans(x, k, seed=seed)
    return labels, centers


def _font(size: int = 18):
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def contact_sheet(glyphs: list[Glyph], labels: np.ndarray, out_png: Path,
                  out_json: Path, per_cluster: int = 6, cell: int = 88,
                  max_clusters: int | None = None, start: int = 0):
    """One row per cluster: cluster id, then up to `per_cluster` examples."""
    font = _font()
    order = sorted(set(int(v) for v in labels))
    order = order[start:start + max_clusters] if max_clusters is not None else order[start:]
    cols = per_cluster + 1
    sheets = []
    row_of = {c: i for i, c in enumerate(order)}
    img = Image.new("RGB", (cols * cell, len(order) * cell), "white")
    draw = ImageDraw.Draw(img)
    mapping = {}
    for c in order:
        row = row_of[c]
        members = [i for i, v in enumerate(labels) if int(v) == c][:per_cluster]
        draw.text((4, row * cell + cell // 2 - 8), f"c{c}", fill=(0, 0, 0), font=font)
        uids = []
        for j, idx in enumerate(members):
            g = glyphs[idx]
            norm = normalize(g.ink, g.space, size=cell - 8)
            tile = Image.fromarray(np.where(norm, 0, 255).astype("uint8")).convert("RGB")
            img.paste(tile, ((j + 1) * cell + 4, row * cell + 4))
            uids.append(g.uid)
        mapping[str(c)] = uids
    out_png.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_png)
    sheets.append(str(out_png))
    out_json.write_text(json.dumps({
        "sheet": str(out_png),
        "clusters": mapping,
        "assignments": {g.uid: int(labels[i]) for i, g in enumerate(glyphs)},
    }, indent=1), encoding="utf-8")
    return mapping


def save_corpus(glyphs: list[Glyph], out_dir: Path):
    gdir = out_dir / "glyphs"
    gdir.mkdir(parents=True, exist_ok=True)
    entries = []
    for g in glyphs:
        path = gdir / f"{g.uid}.png"
        Image.fromarray(np.where(g.ink, 0, 255).astype("uint8")).save(path)
        entries.append({"uid": g.uid, "page": g.page, "staff": g.staff,
                        "x0": g.x0, "y0": g.y0, "x1": g.x1, "y1": g.y1,
                        "space": g.space, "png": str(path.relative_to(out_dir))})
    (out_dir / "glyphs.json").write_text(json.dumps(entries, indent=1), encoding="utf-8")
    return entries


def load_corpus(corpus_dir: Path) -> list[Glyph]:
    entries = json.loads((corpus_dir / "glyphs.json").read_text(encoding="utf-8"))
    out = []
    for e in entries:
        ink = np.asarray(Image.open(corpus_dir / e["png"]).convert("L"))
        ink = (ink < 128).astype("uint8")
        out.append(Glyph(e["uid"], e["page"], e["staff"], e["x0"], e["y0"],
                         e["x1"], e["y1"], e["space"], ink))
    return out
