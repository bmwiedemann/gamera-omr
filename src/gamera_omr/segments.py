"""Segmentation: connected components, text filtering, barlines, overlays."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from .staffs import Staff


@dataclass
class Component:
    glyph: object
    x0: int
    y0: int
    x1: int
    y1: int

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

    @property
    def area(self) -> int:
        return self.w * self.h

    @property
    def solidity(self) -> float:
        ink = np.asarray(self.glyph.to_numpy())
        return float((ink > 0).sum()) / max(1, self.area)


def components(image) -> list[Component]:
    out = []
    for glyph in image.cc_analysis():
        out.append(Component(glyph, glyph.offset_x, glyph.offset_y,
                             glyph.offset_x + glyph.ncols - 1,
                             glyph.offset_y + glyph.nrows - 1))
    return out


def filter_components(comps: list[Component], staff: Staff):
    """Keep musical symbols; drop lyrics, noise, staff remnants and barlines."""
    top, bottom = staff.band(above=4.5, below=1.2)
    kept, dropped = [], {"outside": 0, "outside_x": 0, "tiny": 0, "text": 0,
                         "staffline": 0, "barline": 0}
    for c in comps:
        if c.cy < top or c.cy > bottom:
            dropped["outside"] += 1
            continue
        if c.cx < staff.left - 0.5 * staff.space or c.cx > staff.right + 0.5 * staff.space:
            dropped["outside_x"] += 1
            continue
        if c.area < 4:
            dropped["tiny"] += 1
            continue
        if c.w <= max(4.0, 0.35 * staff.space) and c.h >= 3.0 * staff.space:
            dropped["barline"] += 1
            continue
        # Letters (lyrics) are short; noteheads+stems are at least ~2 spaces
        # tall, beams are wide.  Dots are picked up separately by the decoder.
        if c.h < 0.75 * staff.space and c.w < 1.5 * staff.space:
            dropped["text"] += 1
            continue
        # A very tall, narrow component is a barline fragment or a column of
        # text.  A tall *wide* one can be a piano beamed group spanning the
        # whole staff and must not be discarded.
        if c.h > 10 * staff.space and c.w < 2.0 * staff.space:
            dropped["text"] += 1
            continue
        # a staff-line remnant is a long, very thin horizontal stroke
        if c.w >= 0.6 * (staff.right - staff.left) and c.h <= staff.thickness + 2:
            dropped["staffline"] += 1
            continue
        kept.append(c)
    return kept, dropped


def find_rest_bars(comps: list[Component], staff: Staff) -> list[Component]:
    """Solid, short horizontal bars: whole/half (measure) rests."""
    top, bottom = staff.band(above=4.0, below=1.2)
    bars = []
    for c in comps:
        if not (top <= c.cy <= bottom):
            continue
        if (0.35 * staff.space <= c.w <= 1.8 * staff.space
                and 0.12 * staff.space <= c.h <= 0.5 * staff.space
                and c.solidity >= 0.6):
            bars.append(c)
    return bars


def _attached_stroke(arr: np.ndarray, staff: Staff, x0: int, x1: int) -> bool:
    """True when horizontal ink touches a full-height vertical candidate.

    A real barline is a bare vertical stroke.  A long stem of a beamed group
    reaches the same height, but a notehead or beam joins it horizontally.
    Staff lines are masked, and the search stays within half a staff space of
    the staff, so rehearsal figures above and lyrics below cannot look like
    attachments.
    """
    margin = max(2, int(round(0.5 * staff.space)))
    top = max(0, int(round(staff.top)) - margin)
    bottom = min(arr.shape[0] - 1, int(round(staff.bottom)) + margin)
    span = max(1, int(round(2.5 * staff.space)))
    lo = max(0, x0 - span)
    hi = min(arr.shape[1], x1 + span + 1)
    need = (x1 - x0 + 1) + max(4, int(round(0.25 * staff.space)))
    masked = set()
    thickness = max(1, int(round(staff.thickness))) + 2
    for line in staff.lines:
        y = int(round(line))
        masked.update(range(y - thickness, y + thickness + 1))
    # Printed lines are sometimes a little thicker than the measured value;
    # any row that still spans most of the staff is a line as well.
    x_left = max(0, int(round(staff.left)))
    x_right = min(arr.shape[1] - 1, int(round(staff.right)))
    if x_right > x_left:
        counts = arr[top:bottom + 1, x_left:x_right + 1].sum(axis=1)
        wide = np.flatnonzero(counts >= 0.6 * (x_right - x_left + 1))
        masked.update(int(row) + top for row in wide)
    for y in range(top, bottom + 1):
        if y in masked:
            continue
        row = arr[y, lo:hi]
        idx = np.flatnonzero(row)
        if len(idx) < need:
            continue
        breaks = np.flatnonzero(np.diff(idx) > 1)
        starts = np.r_[idx[0], idx[breaks + 1]]
        ends = np.r_[idx[breaks], idx[-1]]
        for a, b in zip(starts, ends):
            start = int(a) + lo
            stop = int(b) + lo
            if (stop >= x0 - 2 and start <= x1 + 2
                    and stop - start + 1 >= need):
                return True
    return False


def find_barlines(arr: np.ndarray, staff: Staff, loose: bool = False,
                  flush: bool = False, flush_bottom: bool = False,
                  reject_stems: bool = False) -> list[float]:
    """Column centres of full-height vertical strokes (barlines).

    Modern engravings extend barlines above/below the staff (strict test).
    Older vocal scores often draw them flush with the staff; `flush` accepts
    a stroke that reaches the bottom line, and `loose` widens the top and
    lowers the bottom tolerance further.
    """
    top = max(0, int(round(staff.top)) - 3)
    bottom = min(arr.shape[0] - 1, int(round(staff.bottom)) + 3)
    band = arr[top:bottom + 1, :]
    top_off = int(round(staff.top)) - top
    bot_off = int(round(staff.bottom)) - top
    staff_px = bot_off - top_off + 1
    if loose:
        # The band adds ~3 px above/below; a flush barline is only as tall
        # as the staff itself.
        staff_px = max(1, int(round(staff.height)) + 1)
    x0 = max(0, int(round(staff.left)))
    x1 = min(arr.shape[1] - 1, int(round(staff.right)))
    candidates = []
    for x in range(x0, x1 + 1):
        column = band[:, x]
        idx = np.flatnonzero(column)
        if len(idx) == 0:
            continue
        breaks = np.flatnonzero(np.diff(idx) > 1)
        starts = np.r_[idx[0], idx[breaks + 1]]
        ends = np.r_[idx[breaks], idx[-1]]
        lengths = ends - starts + 1
        k = int(np.argmax(lengths))
        if lengths[k] < (0.92 if loose else 0.95) * staff_px:
            continue
        # Modern barlines extend below the bottom line; older ones may be
        # flush with it.  A one-pixel band-origin change after deskewing must
        # not turn every flush barline into a miss.
        if loose:
            lower = bot_off - 1
        elif flush:
            lower = bot_off
        elif flush_bottom:
            lower = bot_off + 1
        else:
            lower = bot_off + 2
        upper_start = (top_off + max(2, int(round(0.6 * staff.space)))
                       if loose else top_off + 2)
        if starts[k] > upper_start or ends[k] < lower:
            continue
        candidates.append(x)
    groups = []
    for x in candidates:
        if groups and x - groups[-1][-1] <= 1:
            groups[-1].append(x)
        else:
            groups.append([x])
    barlines = []
    rejected = []
    start = staff.left + 5.0 * staff.space   # skip the clef/key signature zone
    for group in groups:
        width = group[-1] - group[0] + 1
        centre = (group[0] + group[-1]) / 2
        if width > max(2.0, 0.6 * staff.space):
            continue
        if centre < start:
            continue
        if reject_stems and _attached_stroke(arr, staff, group[0], group[-1]):
            rejected.append(centre)
            continue
        barlines.append(centre)
    # A double bar/repeat is two strokes a few pixels apart; one of them may
    # touch a note or beam.  Keep a rejected stroke when it belongs to such a
    # pair, so the incipit/double-bar handling still sees both.
    for centre in rejected:
        if any(abs(centre - kept) <= 2.2 * staff.space
               for kept in barlines):
            barlines.append(centre)
    return sorted(barlines)


def find_grand_barlines(arr: np.ndarray, upper: Staff, lower: Staff,
                        *, min_coverage: float = 0.92) -> list[float]:
    """Barlines that span the whole grand staff.

    Piano stems and beams can be as tall as a staff; requiring a single
    vertical stroke from the upper staff's top to the lower staff's bottom
    separates real barlines from long stems.
    """
    top = max(0, int(round(upper.top)) - 3)
    bottom = min(arr.shape[0] - 1, int(round(lower.bottom)) + 3)
    need = bottom - top + 1
    x0 = max(0, int(round(min(upper.left, lower.left))))
    x1 = min(arr.shape[1] - 1, int(round(max(upper.right, lower.right))))
    space = max(upper.space, lower.space)
    candidates = []
    for x in range(x0, x1 + 1):
        column = arr[top:bottom + 1, x]
        idx = np.flatnonzero(column)
        if len(idx) == 0:
            continue
        breaks = np.flatnonzero(np.diff(idx) > 1)
        starts = np.r_[idx[0], idx[breaks + 1]]
        ends = np.r_[idx[breaks], idx[-1]]
        k = int(np.argmax(ends - starts + 1))
        if ends[k] - starts[k] + 1 < min_coverage * need:
            continue
        if starts[k] > 6 or ends[k] < need - 7:
            continue
        candidates.append(x)
    groups = []
    for x in candidates:
        if groups and x - groups[-1][-1] <= 1:
            groups[-1].append(x)
        else:
            groups.append([x])
    start = min(upper.left, lower.left) + 5.0 * space
    bars = []
    for group in groups:
        width = group[-1] - group[0] + 1
        centre = (group[0] + group[-1]) / 2
        if width > max(2.0, 0.6 * space) or centre < start:
            continue
        bars.append(centre)
    return bars


def measure_bounds(staff: Staff, barlines: list[float]) -> list[tuple[float, float]]:
    right = staff.right
    start = staff.left + 5.0 * staff.space
    bars = list(barlines)
    # An instrumental incipit at the start of the first system ends with a
    # double bar; the vocal part begins at the second line.  Starting at that
    # bar avoids a phantom measure made of the clef/time signature.
    nearby = [b for b in bars if b < staff.left + 15.0 * staff.space]
    if len(nearby) >= 2 and nearby[1] - nearby[0] <= 1.5 * staff.space:
        start = nearby[1]
        bars = [b for b in bars if b > start]
    edges = [start]
    for x in bars:
        if x <= edges[-1] + 1.0 * staff.space:
            continue
        # A double bar/repeat at a section end is two strokes a few pixels
        # apart; treating both as measures creates a tiny phantom measure.
        # Move the boundary to the outer stroke instead.
        if len(edges) > 1 and x - edges[-1] <= 2.2 * staff.space:
            edges[-1] = x
            continue
        edges.append(x)
    if right > edges[-1] + 1.0 * staff.space:
        edges.append(right)
    bounds = list(zip(edges[:-1], edges[1:]))
    return [(a, b) for a, b in bounds if b - a >= 1.5 * staff.space]


def assign_measures(comps: list[Component], bounds: list[tuple[float, float]]):
    out = {i: [] for i in range(len(bounds))}
    for c in comps:
        for i, (x0, x1) in enumerate(bounds):
            if x0 <= c.cx < x1:
                out[i].append(c)
                break
    return out


def save_overlay(page_path: str | Path, staves: list[Staff], barlines_by_staff: dict[int, list[float]],
                 comps: list[Component], out_path: str | Path, *,
                 image=None):
    if image is not None:
        img = Image.fromarray(
            np.where(np.asarray(image).astype(bool), 0, 255).astype("uint8")
        ).convert("RGB")
    else:
        img = Image.open(page_path).convert("RGB")
    draw = ImageDraw.Draw(img)
    for staff in staves:
        for y in staff.lines:
            draw.line([(staff.left, y), (staff.right, y)], fill=(0, 160, 0), width=1)
        for x in barlines_by_staff.get(staff.index, []):
            draw.line([(x, staff.top - 4), (x, staff.bottom + 4)], fill=(220, 0, 0), width=2)
    for c in comps:
        draw.rectangle([c.x0, c.y0, c.x1, c.y1], outline=(0, 0, 220), width=1)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def save_mask(arr: np.ndarray, out_path: str | Path):
    """Save a 0/1 array as a black-on-white PNG."""
    img = Image.fromarray(np.where(arr.astype(bool), 0, 255).astype("uint8"))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
