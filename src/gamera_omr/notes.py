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
        label = predictions.get(g.uid, {}).get("label")
        if label not in ("note.black", "note.open"):
            # Staff-line removal can leave an isolated head that the
            # classifier calls text.  Its clean ellipse is a good template
            # for matching heads in signature/tall-narrow glyphs.
            if not (label in ("text", "skip") and g.head_like):
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


def stem_direction(ink: np.ndarray, space: float) -> str | None:
    """Stem direction of a note, chord or beamed group.

    Stems, beams and flags all sit on one side of the noteheads, so the ink
    outside the head band tells the direction without tracing each stem.  A
    whole note (nothing outside the heads) returns None.  This is what
    separates two independent voices sharing a staff.
    """
    heads = _distance_heads(ink, space)
    if not heads:
        return None
    ys = [y for _x, y in heads]
    top, bottom = min(ys), max(ys)
    # Leave the heads themselves (and their outline) out of the comparison.
    margin = max(1, int(round(0.3 * space)))
    above = int(ink[:max(0, int(top) - margin)].sum())
    below = int(ink[min(ink.shape[0], int(bottom) + margin + 1):].sum())
    if above > 1.4 * below + 3:
        return "up"
    if below > 1.4 * above + 3:
        return "down"
    return None


def _erosion_heads(ink: np.ndarray, space: float) -> list[tuple[float, float]]:
    """Noteheads as blobs thick enough to survive a head-sized erosion.

    A notehead joined to a long stem or beam is one distance-transform blob
    whose bounding box is the whole stem, so the blob filter misses it.  An
    elliptical erosion of about half a staff space removes stems and leaves
    the head cores; a beam that survives is far too wide to be a head.
    """
    side = max(3, int(round(0.48 * space)))
    if side % 2 == 0:
        side += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (side, side))
    eroded = cv2.erode(_u8(ink), kernel) > 0
    n, _labels, stats, cents = _components(eroded.astype(np.uint8))
    h, w = ink.shape
    margin = max(2, int(round(0.25 * space)))
    heads = []
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        cx, cy = cents[i]
        if not (margin <= cx <= w - 1 - margin
                and margin <= cy <= h - 1 - margin):
            continue  # a beam fragment at the glyph edge
        if (0.25 * space <= bw <= 1.7 * space
                and 0.2 * space <= bh <= 1.5 * space
                and bw <= 2.0 * bh and bh <= 2.0 * bw
                and area >= 0.35 * bw * bh):
            heads.append((float(cents[i][0]), float(cents[i][1])))
    return heads


def _rowrun_heads(ink: np.ndarray, space: float, stem_boxes):
    """Noteheads as clusters of rows whose ink run is about head-wide.

    A sloping beam is long in every row, and a stem is only a few pixels
    wide, so a run between roughly 0.65 and 2.2 staff spaces is a head.  A
    head split by an erased staff line yields two clusters; that is why the
    merge with the other detectors uses a generous vertical tolerance.
    """
    h, w = ink.shape
    lo, hi = 0.65 * space, 2.2 * space
    recs = []
    for y in range(h):
        idx = np.flatnonzero(ink[y])
        if len(idx) == 0:
            continue
        splits = np.flatnonzero(np.diff(idx) > 1)
        starts = np.r_[idx[0], idx[splits + 1]]
        ends = np.r_[idx[splits], idx[-1]]
        for a, b in zip(starts, ends):
            length = b - a + 1
            if lo <= length <= hi:
                recs.append((y, (a + b) / 2.0))
    heads = []
    used = [False] * len(recs)
    recs.sort()
    for i, (y, x) in enumerate(recs):
        if used[i]:
            continue
        group = [(y, x)]
        used[i] = True
        changed = True
        while changed:
            changed = False
            ys = [g[0] for g in group]
            xs = [g[1] for g in group]
            for j, (y2, x2) in enumerate(recs):
                if used[j]:
                    continue
                if (abs(y2 - max(ys)) <= 2 or abs(y2 - min(ys)) <= 2):
                    if any(abs(x2 - x3) <= 0.5 * space for x3 in xs):
                        group.append((y2, x2))
                        used[j] = True
                        changed = True
        ys = [g[0] for g in group]
        if max(ys) - min(ys) + 1 < 0.35 * space:
            continue
        cx = float(np.mean([g[1] for g in group]))
        cy = (min(ys) + max(ys)) / 2.0
        margin = max(2, int(round(0.25 * space)))
        if cy < margin or cy > h - 1 - margin:
            continue  # a beam or staff remnant at the glyph's top/bottom edge
        if not stem_boxes:
            heads.append((cx, cy))
            continue
        for x0, y0, x1, y1, scx in stem_boxes:
            if abs(scx - cx) > 0.8 * space:
                continue
            if not (y0 - 1.0 * space <= cy <= y1 + 1.0 * space):
                continue
            # A head sits at the free end of its stem, not on the beam end.
            if cy <= y0 + 1.2 * space or cy >= y1 - 1.2 * space:
                heads.append((cx, cy))
                break
    return heads


def _head_end_ok(x: float, y: float, stem_boxes, beam_boxes,
                 space: float) -> bool:
    """True when (x, y) is at the free end of a stem, not on its beam end."""
    for x0, y0, x1, y1, scx in stem_boxes:
        if abs(scx - x) > 0.9 * space:
            continue
        if not (y0 - 1.1 * space <= y <= y1 + 1.1 * space):
            continue
        top = any(min(x1, bx1) >= max(x0, bx0) - 2
                  and abs((by0 + by1) / 2.0 - y0) <= 1.3 * space
                  for bx0, by0, bx1, by1 in beam_boxes)
        bottom = any(min(x1, bx1) >= max(x0, bx0) - 2
                     and abs((by0 + by1) / 2.0 - y1) <= 1.3 * space
                     for bx0, by0, bx1, by1 in beam_boxes)
        if top and not bottom:
            if y >= y1 - 1.1 * space:
                return True
        elif bottom and not top:
            if y <= y0 + 1.1 * space:
                return True
        elif y <= y0 + 1.1 * space or y >= y1 - 1.1 * space:
            return True
    return False


def split_beam_group(ink: np.ndarray, space: float, templates=None,
                     *, use_rowrun: bool = True):
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
    rowrun_all = _rowrun_heads(ink, space, []) if use_rowrun else []
    rowrun_valid = (_rowrun_heads(ink, space, stem_boxes)
                    if use_rowrun and stem_boxes else [])
    if stem_boxes and use_rowrun:
        # A thick sloping beam is also a distance-transform blob in places;
        # keep a blob only when a row-run cluster agrees with it (the head
        # detectors found the group's other heads) or it sits at a stem end.
        def supported(hx, hy):
            if any(abs(hx - rx) <= 0.55 * space and abs(hy - ry) <= 0.7 * space
                   for rx, ry in rowrun_all):
                return True
            return _head_end_ok(hx, hy, stem_boxes, beam_boxes, space)

        blob_heads = [h for h in blob_heads if supported(h[0], h[1])]
    eroded_heads = _erosion_heads(ink, space)
    # The distance transform handles long stems and chords, the erosion
    # recovers heads swallowed by a beam, and templates are better for clean
    # single-note groups.  Prefer whichever finds more heads.
    heads = max((matched_heads, blob_heads, eroded_heads), key=len)
    # Beams that slope into a head make it one thick blob with the beam, so
    # the shape detectors above miss it.  Row-run clusters recover these,
    # after requiring the candidate to sit at a stem end so beam fragments
    # and accidental strokes are not taken for heads.  A wide glyph that was
    # classified as a note may be a rest or a text figure; for those, only
    # the shape detectors are trustworthy.
    if use_rowrun:
        for hx, hy in rowrun_valid:
            if all(not (abs(hx - x) <= 0.55 * space
                        and abs(hy - y) <= 0.7 * space)
                   for x, y in heads):
                heads.append((hx, hy))
    group_levels = sorted((by0 + by1) / 2 for _a, by0, _b, by1 in beam_boxes)
    group_count = 1
    if group_levels:
        group_count = 1
        for a, b in zip(group_levels, group_levels[1:]):
            if b - a > 1.0 * space:
                group_count += 1
    out = []
    for hx, hy in heads:
        beams = _beams_at(ink, space, hx, hy, stem_boxes)
        if not beams:
            beams = group_count
        out.append((hx, hy, Fraction(1, 2 ** min(beams, 4))))
    return out


def _beams_at(ink: np.ndarray, space: float, hx: float, hy: float,
              stem_boxes) -> int | None:
    """Number of beams on the stem of the head at (hx, hy).

    Beams cross the stem near its free end; just beside the stem each beam
    is a separate thick run in the column.  Both sides are tried (a 16th
    hook points one way only) and the larger count is taken.
    """
    h, w = ink.shape
    best = None
    for x0, y0, x1, y1, scx in stem_boxes:
        if abs(scx - hx) > 1.0 * space:
            continue
        if not (y0 - 1.0 * space <= hy <= y1 + 1.0 * space):
            continue
        d = abs(scx - hx)
        if best is None or d < best[0]:
            best = (d, x0, y0, x1, y1)
    if best is None:
        return None
    _d, x0, y0, x1, y1 = best
    reach = int(round(2.5 * space))
    if abs(hy - y0) > abs(hy - y1):
        lo, hi = y0, min(y1, y0 + reach)       # beams at the top end
    else:
        lo, hi = max(y0, y1 - reach), y1 + 1   # beams at the bottom end
    if hi - lo < 0.5 * space:
        return None
    count = 0
    for cx in (x0 - 3, x1 + 3):
        if not 0 <= cx < w:
            continue
        runs = [b - a + 1 for a, b in _column_runs(ink[lo:hi, cx])]
        count = max(count, sum(1 for r in runs if r >= 0.2 * space))
    return count or None


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


def stemmed_heads(ink: np.ndarray, space: float, templates=None):
    """Heads of a single stemmed note or chord, with their duration.

    Returns ([(head_x, head_y, duration)], stem direction), or ([], None)
    when the glyph has no stem with a head at its end; the direction is
    "accidental" for a sharp/natural and "nohead" when blobs were found but
    none hangs off the stem like a notehead.  The row-width head finder in `decode`
    takes the widest band of rows, which on a note with a flag mixes the
    head with the flag; here the stem is found first, heads are looked for
    as thick blobs (or template matches for open heads) at its end, and the
    opposite end tells a flag from a plain stem.
    """
    h, w = ink.shape
    # A sharp or natural has two parallel vertical strokes about half a
    # space apart (often broken where a staff line crossed them); a note or
    # chord has a single stem.
    if _two_strokes(ink, space):
        return [], "accidental"
    stem = _longest_vertical(ink)
    if stem is None or stem[2] - stem[1] + 1 < 2.0 * space:
        return [], None
    scx, top, bottom = stem
    sx, sw = int(scx) - 1, 3
    candidates = list(_distance_heads(ink, space))
    for x, y in _erosion_heads(ink, space):
        if all(abs(x - cx) > 0.5 * space or abs(y - cy) > 0.5 * space
               for cx, cy in candidates):
            candidates.append((x, y))
    filled = [True] * len(candidates)
    if templates:
        for score, x, y in match_heads(ink, space, templates, threshold=0.75):
            if all(abs(x - cx) > 0.6 * space or abs(y - cy) > 0.6 * space
                   for cx, cy in candidates):
                candidates.append((float(x), float(y)))
                filled.append(False)
    at_top, at_bottom = [], []
    centred = 0
    for (x, y), is_filled in zip(candidates, filled):
        # A head hangs off one side of its stem; a blob centred on the
        # stroke is the crossbar of an accidental.
        if abs(x - scx) < 0.3 * space:
            centred += 1
            continue
        if abs(x - scx) > 1.1 * space:
            continue
        if not (top - 0.8 * space <= y <= bottom + 0.8 * space):
            continue
        # Heads sit at one end of the stem (a chord stacks them from there);
        # the other end is free or carries a flag.
        if y - top <= bottom - y:
            at_top.append((x, y, is_filled))
        else:
            at_bottom.append((x, y, is_filled))
    if at_top and at_bottom:
        # A flag's thick root looks like a blob too.  The head end is the
        # one whose blobs look like the page's noteheads; without templates
        # the end with the more compact blob wins.
        def end_score(group):
            return max(_head_likeness(ink, space, x, y, templates)
                       for x, y, _f in group)

        if end_score(at_top) >= end_score(at_bottom):
            at_bottom = []
        else:
            at_top = []
    heads = at_top or at_bottom
    # Heads hang left of an up-stem and right of a down-stem; a blob on the
    # other side is only a head as the second of an interval (next to a
    # proper head), otherwise it is the end of a flag.
    proper = [hd for hd in heads
              if (hd[0] > scx if at_top else hd[0] < scx)]
    heads = [hd for hd in heads
             if hd in proper
             or any(abs(hd[1] - p[1]) <= 0.8 * space for p in proper)]
    if not heads:
        if centred and w <= 1.4 * space:
            return [], "accidental"
        # A stem but no head hanging off it the way heads do: a clef body,
        # a flag fragment or a letter, not a note.
        return [], "nohead" if candidates else None
    if at_top:
        direction = "down"
        free_rows = (int(max(y for _x, y, _f in heads) + 0.8 * space),
                     bottom + 1)
    else:
        direction = "up"
        free_rows = (top, int(min(y for _x, y, _f in heads) - 0.8 * space))
    a, b = max(0, free_rows[0]), min(h, free_rows[1])
    flagged = False
    if b - a >= 1.0 * space:
        region = ink[a:b].copy()
        lo = max(0, int(sx - 1))
        hi = min(w, int(sx + sw + 1))
        region[:, lo:hi] = 0
        cols = np.flatnonzero(region.any(axis=0))
        flagged = len(cols) > 0 and (cols[-1] - cols[0] + 1) >= 0.45 * space
    # A head found only by template matching may still be filled (a blob
    # joined to an accidental is too wide for the blob detectors); an open
    # head has a light centre.
    def solid(x, y):
        r = max(1, int(round(0.25 * space)))
        core = ink[max(0, int(y) - r):int(y) + r + 1,
                   max(0, int(x) - r):int(x) + r + 1]
        return core.size > 0 and float(core.mean()) >= 0.75

    heads = [(x, y, f or solid(x, y)) for x, y, f in heads]
    # Open heads are half notes; a filled head with a flag is an eighth.
    open_head = not any(f for _x, _y, f in heads)
    if open_head:
        duration = Fraction(2)
    elif flagged:
        duration = Fraction(1, 2)
    else:
        duration = Fraction(1)
    return [(x, y, duration) for x, y, _f in heads], direction


def _longest_vertical(ink: np.ndarray):
    """(x, top, bottom) of the longest vertical ink run: a note's stem.

    A morphological opening would merge the stem with a flag that hugs it;
    the single longest column run does not.
    """
    h, w = ink.shape
    best = None
    for x in range(w):
        runs = _column_runs(ink[:, x])
        for a, b in runs:
            if best is None or b - a > best[2] - best[1]:
                best = (float(x), a, b)
    if best is None:
        return None
    # Centre the stem across its width (columns with nearly as long a run).
    x0 = x1 = int(best[0])
    length = best[2] - best[1]

    def long_run(x):
        return any(b - a >= 0.85 * length for a, b in _column_runs(ink[:, x]))

    while x0 - 1 >= 0 and long_run(x0 - 1) and x1 - x0 < 6:
        x0 -= 1
    while x1 + 1 < w and long_run(x1 + 1) and x1 - x0 < 6:
        x1 += 1
    return ((x0 + x1) / 2.0, best[1], best[2])


def _column_runs(column: np.ndarray):
    idx = np.flatnonzero(column)
    if len(idx) == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]]
    return list(zip(starts.tolist(), ends.tolist()))


def _two_strokes(ink: np.ndarray, space: float) -> bool:
    """Two parallel vertical strokes about half a space apart (# or natural)."""
    h, w = ink.shape
    if w > 1.6 * space:
        return False
    # Ink coverage per column, tolerant of small breaks in the strokes.
    cover = ink.astype(bool).sum(axis=0)
    tall = cover >= 1.8 * space
    groups = []
    start = None
    for x in range(w + 1):
        on = x < w and tall[x]
        if on and start is None:
            start = x
        elif not on and start is not None:
            groups.append((start, x - 1))
            start = None
    if len(groups) != 2:
        return False
    (a0, a1), (b0, b1) = groups
    gap = (b0 + b1) / 2.0 - (a0 + a1) / 2.0
    return (0.25 * space <= gap <= 1.0 * space
            and max(a1 - a0, b1 - b0) + 1 <= 0.35 * space)


def _head_likeness(ink: np.ndarray, space: float, x: float, y: float,
                   templates) -> float:
    """How much the ink around (x, y) looks like a notehead."""
    r = max(2, int(round(0.8 * space)))
    x0, x1 = max(0, int(x) - r), min(ink.shape[1], int(x) + r + 1)
    y0, y1 = max(0, int(y) - r), min(ink.shape[0], int(y) + r + 1)
    patch = ink[y0:y1, x0:x1].astype(np.float32)
    best = 0.0
    for tpl in templates or ():
        if tpl.shape[0] > patch.shape[0] or tpl.shape[1] > patch.shape[1]:
            continue
        res = cv2.matchTemplate(patch, tpl.astype(np.float32),
                                cv2.TM_CCOEFF_NORMED)
        best = max(best, float(res.max()))
    if templates:
        return best
    # No templates: a head is a solid ellipse, a flag root a thin wedge.
    dt = cv2.distanceTransform(_u8(ink[y0:y1, x0:x1] > 0), cv2.DIST_L2, 5)
    return float(dt.max()) / max(1.0, space)


def accidental_kind(ink: np.ndarray, space: float) -> str | None:
    """'sharp', 'natural' or 'flat' for an isolated accidental glyph.

    Sharps and naturals both have two vertical strokes; a natural is
    narrower (about 0.8 of a space against 1.1).  A flat is one stroke on
    the left with a closed bowl at its foot.
    """
    h, w = ink.shape
    if not (1.6 * space <= h <= 3.8 * space):
        return None
    if _two_strokes(ink, space) and h >= 2.3 * space:
        if 0.95 * space <= w <= 1.45 * space:
            return "sharp"
        if 0.55 * space <= w < 0.95 * space:
            return "natural"
        return None
    if not (0.5 * space <= w <= 1.05 * space and h <= 2.9 * space):
        return None
    stem = _longest_vertical(ink)
    if stem is None or stem[2] - stem[1] + 1 < 1.5 * space:
        return None
    if stem[0] > 0.4 * w or stem[2] < h - 0.25 * space:
        return None
    contours, hierarchy = cv2.findContours(_u8(ink), cv2.RETR_CCOMP,
                                           cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return None
    for contour, hh in zip(contours, hierarchy[0]):
        if hh[3] == -1:
            continue
        _x, y, _w, ch = cv2.boundingRect(contour)
        if y + ch / 2.0 >= 0.5 * h:
            return "flat"
    return None


def lone_head(ink: np.ndarray, space: float):
    """(x, y) of a notehead that lost its stem, or None.

    The blob is wider than tall, solid, about one space high, and at most a
    short stem stub remains; an eighth rest is taller than wide and a
    quarter rest or a letter has no head-sized solid core.
    """
    h, w = ink.shape
    if not (0.8 * space <= w <= 1.8 * space and 0.6 * space <= h <= 2.8 * space):
        return None
    blobs = _erosion_heads(ink, space) or _distance_heads(ink, space)
    if len(blobs) != 1:
        return None
    x, y = blobs[0]
    rows = ink[max(0, int(y - 0.7 * space)):int(y + 0.7 * space) + 1]
    cols = np.flatnonzero(rows.any(axis=0))
    if len(cols) == 0:
        return None
    width = cols[-1] - cols[0] + 1
    band = rows[:, cols[0]:cols[-1] + 1]
    filled = [np.flatnonzero(r) for r in band if r.any()]
    height = len(filled)
    if not (0.9 * space <= width <= 1.7 * space
            and 0.55 * space <= height <= 1.3 * space
            and width >= 0.95 * height):
        return None
    if float(band.mean()) < 0.6:
        return None
    # Anything beyond the head must be a thin stem stub, not a rest's tail.
    rest = ink.copy()
    rest[max(0, int(y - 0.7 * space)):int(y + 0.7 * space) + 1] = 0
    extra_cols = np.flatnonzero(rest.any(axis=0))
    if len(extra_cols) and extra_cols[-1] - extra_cols[0] + 1 > 0.35 * space:
        return None
    return float(x), float(y)
