"""Note detection: split noteheads from stems/beams and derive pitch/duration.

The input is the staff-removed image.  Stems and beams are separated
morphologically; the remaining blobs are noteheads, which are classified as
filled/open by their ink ratio and hole, and associated with their stem/beam
to get the duration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
from fractions import Fraction

import cv2
import numpy as np

from .staffs import Staff


@dataclass
class Note:
    x0: int
    y0: int
    x1: int
    y1: int
    cx: float
    cy: float
    filled: bool
    open_hole: bool
    stem: tuple[int, int, int, int] | None = None
    beams: int = 0
    dot: bool = False
    measure: int | None = None
    position: int | None = None      # staff steps above the bottom line
    pitch: str | None = None
    duration: Fraction = Fraction(1)
    @property
    def w(self): return self.x1 - self.x0 + 1
    @property
    def h(self): return self.y1 - self.y0 + 1


@dataclass
class Masks:
    ink: np.ndarray
    stems: np.ndarray
    beams: np.ndarray
    heads: np.ndarray
    notes: list[Note] = field(default_factory=list)


def _u8(mask: np.ndarray) -> np.ndarray:
    return (mask.astype(np.uint8) * 255)


def _components(mask: np.ndarray):
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(_u8(mask), 8)
    return n, labels, stats, centroids


def build_masks(arr: np.ndarray, staff: Staff) -> Masks:
    ink = (arr > 0).astype(np.uint8)
    space = max(1.0, staff.space)
    stem_len = max(5, int(round(2.0 * space)))
    beam_len = max(5, int(round(1.5 * space)))
    stems = cv2.morphologyEx(_u8(ink), cv2.MORPH_OPEN,
                             np.ones((stem_len, 1), np.uint8)) > 0
    beams = cv2.morphologyEx(_u8(ink), cv2.MORPH_OPEN,
                             np.ones((1, beam_len), np.uint8)) > 0
    # Subtract stems and beams, but keep a small margin around them so a
    # notehead is not bitten into at the attachment point.
    grown_stems = cv2.dilate(_u8(stems), np.ones((1, 3), np.uint8)) > 0
    grown_beams = cv2.dilate(_u8(beams), np.ones((3, 1), np.uint8)) > 0
    heads = ink & ~grown_stems & ~grown_beams
    # Remove tiny specks; keep components a notehead could be.
    heads = cv2.morphologyEx(_u8(heads), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8)) > 0
    return Masks(ink, stems, beams, heads)


def _has_hole(component: np.ndarray) -> bool:
    contours, hierarchy = cv2.findContours(_u8(component), cv2.RETR_CCOMP,
                                           cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return False
    return any(h[3] != -1 for h in hierarchy[0])


def _stem_table(stems: np.ndarray):
    n, labels, stats, cents = _components(stems)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if h < 6 or w > 8:
            continue
        out.append((x, y, x + w - 1, y + h - 1, cents[i][0]))
    return out


def _beam_table(beams: np.ndarray):
    n, labels, stats, cents = _components(beams)
    out = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        out.append((x, y, x + w - 1, y + h - 1))
    return out


def detect(arr: np.ndarray, staff: Staff) -> Masks:
    space = max(1.0, staff.space)
    masks = build_masks(arr, staff)
    n, labels, stats, cents = _components(masks.heads)
    stems = _stem_table(masks.stems)
    beams = _beam_table(masks.beams)
    notes = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        x1, y1 = x + w - 1, y + h - 1
        cx, cy = cents[i]
        # Notehead shape gates.
        if not (0.45 * space <= w <= 2.6 * space and 0.4 * space <= h <= 1.8 * space):
            continue
        if area < 0.12 * space * space:
            continue
        if cx < staff.left + 5.0 * space or cx > staff.right:
            continue
        ratio = area / float(w * h)
        component = (labels[y:y + h, x:x + w] == i).astype(np.uint8)
        hole = _has_hole(component)
        filled = ratio > 0.55 and not hole
        # Attach the nearest stem at the left or right edge.
        stem = None
        best = 1e9
        for sx0, sy0, sx1, sy1, scx in stems:
            if not (y - 0.6 * space <= sy1 and sy0 <= y1 + 0.6 * space):
                continue
            # stem is attached if its x is at the head's left or right edge
            edge = min(abs(scx - x), abs(scx - x1))
            if edge <= 2.5 and sy0 <= cy <= sy1 + 0.5 * space:
                d = abs(scx - cx)
                if d < best:
                    best, stem = d, (sx0, sy0, sx1, sy1)
        # Count beam levels touching the stem (distinct y bands).
        beam_count = 0
        if stem:
            sx0, sy0, sx1, sy1 = stem
            ys = []
            for bx0, by0, bx1, by1 in beams:
                overlap = min(sx1, bx1) >= max(sx0, bx0) - 2
                near = by0 <= sy0 + 2 or by1 >= sy1 - 2
                if overlap and near:
                    ys.append((by0 + by1) / 2)
            ys.sort()
            beam_count = len([1 for a, b in zip(ys, ys[1:] + [1e9]) if b - a > 1.0])
        # A dot to the right of the head.
        dot = False
        for j in range(1, n):
            if j == i:
                continue
            dx, dy, dw, dh, da = stats[j]
            if (0.25 * space <= dh <= 0.5 * space and 0.25 * space <= dw <= 0.5 * space
                    and x1 < dx <= x1 + 1.2 * space
                    and abs((dy + dh / 2) - cy) <= 0.5 * space):
                dot = True
                break
        note = Note(x, y, x1, y1, cx, cy, filled, hole, stem, beam_count, dot)
        if stem is None:
            note.duration = Fraction(4) if hole else Fraction(1)
        elif hole:
            note.duration = Fraction(2)
        elif beam_count:
            note.duration = Fraction(1, 2 ** min(beam_count, 4))
        else:
            note.duration = Fraction(1)
        if dot:
            note.duration = note.duration * Fraction(3, 2)
        notes.append(note)
    masks.notes = sorted(notes, key=lambda nn: (nn.cy, nn.cx))
    return masks


def head_template(ink: np.ndarray, space: float):
    """Crop the notehead (without stem) as a template."""
    from .decode import head_region  # local import avoids a cycle
    a, b = head_region(ink, space)
    region = ink[a:b + 1]
    if region.size == 0:
        return None
    colsum = region.sum(axis=0)
    cols = np.flatnonzero(colsum >= 0.4 * region.shape[0])
    if len(cols) == 0:
        return None
    tpl = region[:, cols[0]:cols[-1] + 1]
    if tpl.shape[0] < 4 or tpl.shape[1] < 4:
        return None
    return tpl


def make_templates(glyphs, predictions, limit: int = 12):
    templates, seen = [], set()
    for g in glyphs:
        if predictions.get(g.uid, {}).get("label") not in ("note.black", "note.open"):
            continue
        tpl = head_template(g.ink, g.space)
        if tpl is None:
            continue
        # A notehead is about one staff space in both directions; a rest,
        # accidental or clef leaf makes a very different template.
        h, w = tpl.shape
        if not (0.4 * g.space <= w <= 1.8 * g.space
                and 0.4 * g.space <= h <= 1.6 * g.space):
            continue
        key = (tpl.shape, int(tpl.sum()))
        if key in seen:
            continue
        seen.add(key)
        templates.append(tpl)
        if len(templates) >= limit:
            break
    return templates


def match_heads(ink: np.ndarray, space: float, templates, threshold: float = 0.6):
    peaks = []
    for tpl in templates:
        if tpl.shape[0] > ink.shape[0] or tpl.shape[1] > ink.shape[1]:
            continue
        res = cv2.matchTemplate(ink.astype(np.float32), tpl.astype(np.float32),
                                cv2.TM_CCOEFF_NORMED)
        ys, xs = np.where(res >= threshold)
        for y, x in zip(ys, xs):
            peaks.append((float(res[y, x]), x + tpl.shape[1] / 2,
                          y + tpl.shape[0] / 2))
    peaks.sort(reverse=True)
    kept = []
    for score, x, y in peaks:
        if all(not (abs(x - kx) < 1.0 * space and abs(y - ky) < 1.0 * space)
               for _s, kx, ky in kept):
            kept.append((score, x, y))
    return kept[:6]


def _distance_heads(ink: np.ndarray, space: float) -> list[tuple[float, float]]:
    """Noteheads as thick blobs (distance-transform peaks).

    Stems and beams are thin, so a distance transform suppresses them while
    filled noteheads - including chords on one long stem - remain as blobs.
    """
    dt = cv2.distanceTransform(_u8(ink), cv2.DIST_L2, 5)
    mask = (dt > max(1.5, 0.16 * space)).astype(np.uint8)
    n, _labels, stats, cents = _components(mask)
    heads = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if (0.35 * space <= bw <= 1.7 * space
                and 0.25 * space <= bh <= 2.2 * space
                and bw <= 2.2 * bh and area >= 0.25 * bw * bh
                and (bh <= 1.4 * space or area >= 0.45 * bw * bh)):
            heads.append((float(cents[i][0]), float(cents[i][1])))
    return heads


def split_beam_group(ink: np.ndarray, space: float, templates=None):
    """Split a beamed group into (head_x, head_y, duration) per notehead."""
    h, w = ink.shape
    stem_k = np.ones((max(4, int(0.9 * space)), 1), np.uint8)
    stems = cv2.morphologyEx(_u8(ink), cv2.MORPH_OPEN, stem_k) > 0
    beam_k = np.ones((1, max(5, int(0.8 * space))), np.uint8)
    beams = cv2.morphologyEx(_u8(ink), cv2.MORPH_OPEN, beam_k) > 0
    n, _labels, stats, cents = _components(stems)
    stem_boxes = []
    for i in range(1, n):
        x, y, bw, bh, _area = stats[i]
        # Long piano stems are often merged with their noteheads by the
        # vertical opening, so accept a wide box when it is clearly vertical.
        if (bh >= 0.8 * space and bw <= max(4, 1.6 * space)
                and bh >= 2.0 * bw):
            stem_boxes.append((x, y, x + bw - 1, y + bh - 1, float(cents[i][0])))
    nb, _blabels, bstats, _bcents = _components(beams)
    beam_boxes = []
    for j in range(1, nb):
        bx, by, bw, bh, _area = bstats[j]
        # A filled notehead also survives the horizontal opening; keep only
        # genuinely long, thin beam strokes so head rows are not excluded.
        if bh <= 0.6 * space and bw >= max(1.2 * space, 2.0 * bh):
            beam_boxes.append((bx, by, bx + bw - 1, by + bh - 1))
    matched = match_heads(ink, space, templates) if templates else []
    # One notehead per stem: heads sharing a stem are duplicates/false peaks.
    by_stem: dict[int, tuple[float, float, float]] = {}
    for score, x, y in matched:
        best = None
        for si, (x0, y0, x1, y1, scx) in enumerate(stem_boxes):
            if (abs(scx - x) <= 1.0 * space
                    and y0 - 1.0 * space <= y <= y1 + 1.0 * space):
                d = abs(scx - x)
                if best is None or d < best[0]:
                    best = (d, si)
        if best is not None:
            si = best[1]
            if si not in by_stem or score > by_stem[si][0]:
                by_stem[si] = (score, x, y)
    matched_heads = [(x, y) for _s, x, y in by_stem.values()]
    if not matched_heads:
        matched_heads = [(x, y) for _s, x, y in matched[:4]]
    blob_heads = _distance_heads(ink, space)
    # The blob detector handles long stems and chords; templates are better
    # for clean single-note groups.  Prefer whichever finds more heads.
    heads = blob_heads if len(blob_heads) > len(matched_heads) else matched_heads
    group_levels = sorted((by0 + by1) / 2 for _a, by0, _b, by1 in beam_boxes)
    group_count = 1
    if group_levels:
        group_count = 1
        for a, b in zip(group_levels, group_levels[1:]):
            if b - a > 1.0 * space:
                group_count += 1
    out = []
    for hx, hy in heads:
        out.append((hx, hy, Fraction(1, 2 ** min(group_count, 4))))
    return out


def overlay(page_path, staff: Staff, masks: Masks, out_path, *, image=None):
    if image is not None:
        img = cv2.cvtColor(
            np.where(np.asarray(image).astype(bool), 0, 255).astype("uint8"),
            cv2.COLOR_GRAY2BGR,
        )
    else:
        img = cv2.imread(str(page_path), cv2.IMREAD_COLOR)
    if img is None:
        return
    for n in masks.notes:
        cv2.circle(img, (int(n.cx), int(n.cy)), max(3, int(staff.space / 4)),
                   (255, 0, 0), 1)
        if n.stem:
            x0, y0, x1, y1 = n.stem
            cv2.rectangle(img, (x0, y0), (x1, y1), (0, 170, 0), 1)
        cv2.putText(img, f"{n.pitch or '?'}/{n.duration}", (n.x0, n.y0 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (200, 0, 0), 1, cv2.LINE_AA)
    cv2.imwrite(str(out_path), img)
