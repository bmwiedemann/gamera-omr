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
    # Set by `mark_head_like`: a compact ellipse whose stem is either a
    # separate near-vertical fragment or part of a larger connected group.
    head_like: bool = False

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


def _has_nearby_stem(c, comps: list[Component], space: float) -> bool:
    """True when a vertical stroke touches or almost touches `c`.

    Isolated noteheads keep a short stem fragment after staff-line removal;
    rest blobs are accompanied by a short diagonal stroke instead, which is
    wider than a stem for its height.
    """
    for s in comps:
        if s is c:
            continue
        if not (s.w <= max(4.0, 0.45 * space)
                and s.h >= max(0.5 * space, 2.0 * s.w)):
            continue
        if abs(s.cx - c.cx) > 0.9 * space:
            continue
        if s.y0 <= c.y1 + 0.6 * space and s.y1 >= c.y0 - 0.6 * space:
            return True
    return False


def _inside_larger(c, comps: list[Component], space: float) -> bool:
    """True when a larger component's box encloses `c`.

    A notehead cut off from a beamed group sits inside that group's bounding
    box; its stem is part of the group, so no separate stem exists.
    """
    tol = 0.25 * space
    for s in comps:
        if s is c:
            continue
        if s.h < 1.5 * space and s.w < 1.5 * space:
            continue
        if not (s.x0 - tol <= c.x0 and s.x1 + tol >= c.x1
                and s.y0 - tol <= c.y0 and s.y1 + tol >= c.y1):
            continue
        if s.w > c.w + 0.5 * space or s.h > c.h + 0.5 * space:
            return True
    return False


def mark_head_like(comps: list[Component], space: float):
    """Flag compact notehead ellipses that still have a stem context."""
    for c in comps:
        aspect = c.w / max(1.0, c.h)
        if not (0.45 * space <= c.h <= 1.1 * space
                and 0.9 * space <= c.w <= 1.9 * space
                and 1.1 <= aspect <= 3.0
                and c.solidity >= 0.8):
            c.head_like = False
            continue
        c.head_like = (_has_nearby_stem(c, comps, space)
                       or _inside_larger(c, comps, space))


def filter_components(comps: list[Component], staff: Staff):
    """Keep musical symbols; drop lyrics, noise, staff remnants and barlines."""
    mark_head_like(comps, staff.space)
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
        # A notehead whose stem was cut away by staff-line removal is a
        # compact, solid ellipse about half a space high; keep it only when
        # `mark_head_like` found a stem context, so quarter-rest blobs and
        # accidental fragments stay out.
        if c.h < 0.75 * staff.space and c.w < 1.5 * staff.space:
            if not c.head_like:
                dropped["text"] += 1
                continue
        # A very tall, narrow component is a barline fragment or a column of
        # text.  A tall *wide* one can be a piano beamed group spanning the
        # whole staff and must not be discarded.
        if c.h > 10 * staff.space and c.w < 2.0 * staff.space:
            dropped["text"] += 1
            continue
        # The system-start brace joins every staff; its column is a little
        # wider than the strict limit above but still far taller than wide.
        if (c.h > 8 * staff.space and c.w < 3.0 * staff.space
                and c.h > 4.0 * c.w):
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


def _attachment_score(arr: np.ndarray, staff: Staff, x0: int, x1: int) -> int:
    """True when wide horizontal ink joins a full-height vertical candidate.

    A real barline is a bare vertical stroke.  A long stem of a beamed group
    reaches the same height, but a beam or notehead joins it horizontally.
    Only wide runs (a beam or a head) count, and they must cover at least
    three rows: a slur or rehearsal figure touching a barline is thin or
    narrow and must not discard the barline.  Staff lines are masked, and the
    search stays within 2.5 staff spaces of the staff.
    """
    margin = max(2, int(round(2.5 * staff.space)))
    top = max(0, int(round(staff.top)) - margin)
    bottom = min(arr.shape[0] - 1, int(round(staff.bottom)) + margin)
    span = max(1, int(round(2.5 * staff.space)))
    lo = max(0, x0 - span)
    hi = min(arr.shape[1], x1 + span + 1)
    need = max(int(round(0.9 * staff.space)), (x1 - x0 + 1) + 6)
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
    # Only ink touching the stroke itself counts: a tie or slur passing
    # just below a barline is not attached to it, while a beam or head is
    # joined to its stem within the stem's own extent.
    xc = (x0 + x1) // 2
    stroke_top, stroke_bottom = top, bottom
    if 0 <= xc < arr.shape[1]:
        column = arr[top:bottom + 1, max(0, x0):x1 + 1].any(axis=1)
        idx = np.flatnonzero(column)
        if len(idx):
            breaks = np.flatnonzero(np.diff(idx) > 1)
            starts = np.r_[idx[0], idx[breaks + 1]]
            ends = np.r_[idx[breaks], idx[-1]]
            k = int(np.argmax(ends - starts))
            # a head is centred on its stem's end
            reach = max(2, int(round(0.6 * staff.space)))
            stroke_top = top + int(starts[k]) - reach
            stroke_bottom = top + int(ends[k]) + reach
    joined = 0
    for y in range(max(top, stroke_top), min(bottom, stroke_bottom) + 1):
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
                joined += 1
                break
        if joined >= 3:
            return 3
    return joined


def _stroke_extent(arr: np.ndarray, staff: Staff, x0: int, x1: int):
    """(top, bottom) rows of the longest vertical run in columns x0..x1."""
    margin = max(2, int(round(2.5 * staff.space)))
    top = max(0, int(round(staff.top)) - margin)
    bottom = min(arr.shape[0] - 1, int(round(staff.bottom)) + margin)
    column = arr[top:bottom + 1, max(0, x0):x1 + 1].any(axis=1)
    idx = np.flatnonzero(column)
    if len(idx) == 0:
        return None
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]]
    k = int(np.argmax(ends - starts))
    return top + int(starts[k]), top + int(ends[k])


def _beam_on_line(arr: np.ndarray, staff: Staff, x0: int, x1: int) -> bool:
    """True when a beam lying on a staff line leaves a long stem's end.

    The line mask of `_attachment_score` hides such a beam.  Beside the
    stroke, a bare line is a thin run; a beam on it makes the run through
    the line much thicker all along one side, right from the stroke.
    """
    extent = _stroke_extent(arr, staff, x0, x1)
    if extent is None:
        return False
    space = staff.space
    thick = max(1.0, staff.thickness)
    width = max(4, int(round(1.0 * space)))
    h, w = arr.shape
    for line in staff.lines:
        y = int(round(line))
        # only a line at one end of the stroke, where a beam joins a stem
        if min(abs(y - extent[0]), abs(y - extent[1])) > 0.6 * space:
            continue
        for cols in (range(x1 + 1, min(w, x1 + 1 + width)),
                     range(max(0, x0 - width), x0)):
            cols = list(cols)
            if len(cols) < width:
                continue
            thick_cols = 0
            for x in cols:
                if not arr[y, x]:
                    break
                a = b = y
                while a - 1 >= 0 and arr[a - 1, x]:
                    a -= 1
                while b + 1 < h and arr[b + 1, x]:
                    b += 1
                if b - a + 1 < thick + 0.35 * space:
                    break
                thick_cols += 1
            if thick_cols == len(cols):
                return True
    return False


def _head_at_end(arr: np.ndarray, staff: Staff, x0: int, x1: int) -> bool:
    """True when a notehead sits at an end of the stroke (it is a stem).

    Beside one end of a stem, a filled or open head fills much of a
    head-sized box and touches the stroke; beside a barline there are only
    staff lines and the odd thin tie.
    """
    extent = _stroke_extent(arr, staff, x0, x1)
    if extent is None:
        return False
    space = staff.space
    h, w = arr.shape
    reach = int(round(0.6 * space))
    box = int(round(1.1 * space))
    half = int(staff.thickness) // 2 + 1
    line_rows = set()
    for line in staff.lines:
        y = int(round(line))
        line_rows.update(range(y - half, y + half + 1))
    # the stroke's own columns: those carrying most of its length
    length = extent[1] - extent[0] + 1

    def long_run(x):
        if not 0 <= x < w:
            return False
        col = arr[extent[0]:extent[1] + 1, x] > 0
        return col.mean() >= 0.8

    while x0 - 1 >= 0 and long_run(x0 - 1) and x1 - x0 < 6:
        x0 -= 1
    while long_run(x1 + 1) and x1 - x0 < 6:
        x1 += 1
    if length < 2 * space:
        return False
    for end in extent:
        rows = [y for y in range(max(0, end - reach), min(h, end + reach + 1))
                if y not in line_rows]
        if len(rows) < 0.4 * space:
            continue
        for cols in ((max(0, x0 - box), x0), (x1 + 1, min(w, x1 + 1 + box))):
            if cols[1] - cols[0] < box:
                continue
            patch = arr[rows, cols[0]:cols[1]] > 0
            touch = patch[:, -1] if cols[1] == x0 else patch[:, 0]
            if patch.mean() >= 0.35 and touch.sum() >= 0.25 * space:
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
        # Some NBA barlines stop a little short of the bottom line.
        short = flush_bottom and not (loose or flush)
        if lengths[k] < (0.92 if loose else 0.88 if short else 0.95) * staff_px:
            continue
        # Modern barlines extend below the bottom line; older ones may be
        # flush with it.  A one-pixel band-origin change after deskewing must
        # not turn every flush barline into a miss.
        if loose:
            lower = bot_off - 1
        elif flush:
            lower = bot_off
        elif flush_bottom:
            lower = bot_off - max(1, int(round(0.4 * staff.space)))
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
    scored = []
    rejected = []
    start = staff.left + 5.0 * staff.space   # skip the clef/key signature zone
    for group in groups:
        width = group[-1] - group[0] + 1
        centre = (group[0] + group[-1]) / 2
        if width > max(2.0, 0.6 * staff.space):
            continue
        if centre < start:
            continue
        score = (_attachment_score(arr, staff, group[0], group[-1])
                 if reject_stems else 0)
        if (reject_stems and score < 3
                and (_beam_on_line(arr, staff, group[0], group[-1])
                     or _head_at_end(arr, staff, group[0], group[-1]))):
            score = 3
        if score >= 3:
            rejected.append((centre, score))
            continue
        scored.append((centre, score))
    # A double bar/repeat is two strokes a few pixels apart; one of them may
    # touch a note or beam.  Keep a rejected stroke when it belongs to such a
    # pair, so the incipit/double-bar handling still sees both.
    for centre, score in rejected:
        if any(abs(centre - kept) <= 0.8 * staff.space
               for kept, _score in scored):
            scored.append((centre, score))
    # A double bar/repeat can have both strokes touching beams (the opening
    # incipit often does); keep such a close pair as one boundary.
    restored = set()
    for i, (centre, score) in enumerate(rejected):
        for other, other_score in rejected[i + 1:]:
            if abs(other - centre) <= 0.8 * staff.space:
                for c, sc in ((centre, score), (other, other_score)):
                    if c not in restored:
                        scored.append((c, sc))
                        restored.add(c)
    # A system can open with a short pickup whose beam crosses the following
    # barline.  Keep an attached stroke near the left edge when no other
    # barline is close, or that first measure disappears.
    for centre, score in list(rejected):
        x = int(round(centre))
        if _head_at_end(arr, staff, x, x):
            continue  # a stem: its head hangs at one end
        if (centre < staff.left + 10.0 * staff.space
                and all(abs(centre - kept) > 4.0 * staff.space
                        for kept, _score in scored)):
            scored.append((centre, score))
    # A stem close to a clean barline can pass the row threshold; when two
    # strokes are close, keep the cleaner one.
    barlines = []
    for centre, score in scored:
        if 0 < score < 3:
            neighbours = [
                other_score for other, other_score in scored
                if other != centre and abs(other - centre) <= 6.0 * staff.space
            ]
            if neighbours and min(neighbours) < score:
                continue
        barlines.append(centre)
    return sorted(barlines)


def _split_grand_barline(column, top: int, upper: Staff, lower: Staff,
                         gap: int) -> bool:
    """A barline left open between the staves for a dynamic mark.

    It still runs unbroken through each staff and covers most of the gap.
    """
    def through(staff):
        a = int(round(staff.top)) - top
        b = int(round(staff.bottom)) - top
        seg = column[max(0, a - 1):b + 2]
        if len(seg) == 0:
            return False
        idx = np.flatnonzero(seg)
        return (len(idx) >= 0.97 * len(seg)
                and (len(idx) < 2 or np.diff(idx).max() <= gap))

    if not (through(upper) and through(lower)):
        return False
    # From each staff the stroke continues into the gap without a break;
    # together the two stubs cover most of it.
    a = int(round(upper.bottom)) - top
    b = int(round(lower.top)) - top
    if b - a < 2:
        return False
    down = 0
    while a + down < b and column[a + down]:
        down += 1
    up = 0
    while b - up > a and column[b - up]:
        up += 1
    return down + up >= 0.55 * (b - a) and min(down, up) >= 0.1 * (b - a)


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
    # A barline printed across the inter-staff gap is sometimes broken by a
    # beam or a white speck; merge gaps smaller than 0.6 staff space.
    gap = max(2, int(round(0.6 * space)))
    start_margin = max(6, int(round(0.4 * space)))
    end_margin = max(7, int(round(0.5 * space)))
    candidates = []
    for x in range(x0, x1 + 1):
        column = arr[top:bottom + 1, x]
        idx = np.flatnonzero(column)
        if len(idx) == 0:
            continue
        breaks = np.flatnonzero(np.diff(idx) > gap)
        starts = np.r_[idx[0], idx[breaks + 1]]
        ends = np.r_[idx[breaks], idx[-1]]
        k = int(np.argmax(ends - starts + 1))
        whole = (ends[k] - starts[k] + 1 >= min_coverage * need
                 and starts[k] <= start_margin
                 and ends[k] >= need - end_margin)
        if whole or (_split_grand_barline(column, top, upper, lower, gap)
                     and not _head_at_end(arr, upper, x, x)
                     and not _head_at_end(arr, lower, x, x)):
            # (two stems meeting from both staves have heads at their ends)
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
