"""Staff finding, validation, and staff-line removal.

Gamera's ported MusicStaves toolkit provides the staff *finders* only
(`StaffFinder_projections`, `StaffFinder_miyao`; the Dalitz finder needs a C++
plugin that is not in the ported tree).  Removal is implemented here with the
classic run-length rule: erase a staff line where it is a thin horizontal run,
but keep the parts of vertical strokes that cross it.
"""

from __future__ import annotations

import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import config
from .stacks import StackPage, from_array, init_gamera


class StaffFindError(RuntimeError):
    pass


@dataclass
class Staff:
    index: int
    lines: list[float]
    left: float
    right: float
    space: float
    thickness: float
    backend: str = ""
    reasons: list[str] = field(default_factory=list)
    # Measure x-bounds computed before a deskew; x is unchanged by a vertical
    # shear, so reusing them keeps measure segmentation stable.
    bounds: list[tuple[float, float]] = field(default_factory=list)

    @property
    def top(self) -> float:
        return min(self.lines)

    @property
    def bottom(self) -> float:
        return max(self.lines)

    @property
    def height(self) -> float:
        return self.bottom - self.top

    @property
    def center(self) -> float:
        return (self.top + self.bottom) / 2

    def band(self, above: float = 4.5, below: float = 2.5) -> tuple[float, float]:
        return self.top - above * self.space, self.bottom + below * self.space


def _convert_average(averages, thickness: float, backend: str) -> list[Staff]:
    staves = []
    for i, staff in enumerate(averages):
        ys = sorted(float(line.average_y) for line in staff)
        lefts = [float(line.left_x) for line in staff if line.left_x is not None]
        rights = [float(line.right_x) for line in staff if line.right_x is not None]
        if len(ys) < 2:
            continue
        space = float(statistics.median(np.diff(ys)))
        staves.append(Staff(
            index=len(staves),
            lines=ys,
            left=float(statistics.median(lefts)) if lefts else 0.0,
            right=float(statistics.median(rights)) if rights else 0.0,
            space=space,
            thickness=max(1.0, float(thickness)),
            backend=backend,
        ))
    return staves


def _run_finder(image, backend: str) -> list[Staff]:
    init_gamera()
    if backend == "projections":
        from gamera.toolkits.musicstaves.stafffinder_projections import (  # noqa: PLC0415
            StaffFinder_projections,
        )
        finder = StaffFinder_projections(image, 0, 0)
        finder.find_staves(num_lines=5)
    elif backend == "miyao":
        from gamera.toolkits.musicstaves.stafffinder_miyao import (  # noqa: PLC0415
            StaffFinder_miyao,
        )
        finder = StaffFinder_miyao(image, 0, 0)
        finder.find_staves(num_lines=5, scanlines=20, blackness=0.8, tolerance=-1)
    else:
        raise ValueError(f"unknown staff finder {backend!r}")
    return _convert_average(finder.get_average(), finder.staffline_height, backend)


def validate_staves(staves: list[Staff], image) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    if not 1 <= len(staves) <= 6:
        reasons.append(f"found {len(staves)} staves (expected 1..6)")
    width = image.ncols
    for staff in staves:
        if len(staff.lines) != 5:
            reasons.append(f"staff {staff.index}: {len(staff.lines)} lines")
            continue
        diffs = np.diff(staff.lines)
        if diffs.min() <= 0:
            reasons.append(f"staff {staff.index}: lines not increasing")
        if diffs.mean() > 0:
            cv = float(diffs.std() / diffs.mean())
            if cv > 0.15:
                reasons.append(f"staff {staff.index}: spacing CV {cv:.3f}")
        if not 8 <= staff.space <= 40:
            reasons.append(f"staff {staff.index}: space {staff.space:.1f}")
        # Height must match the line count and space, not an absolute pixel
        # range: small 1900s vocal-score prints have ~52 px staves, while
        # 600-dpi scans can have >90 px.
        expected = 4.0 * staff.space + staff.thickness
        if not 0.80 * expected <= staff.height <= 1.20 * expected:
            reasons.append(f"staff {staff.index}: height {staff.height:.1f} "
                           f"vs expected {expected:.1f}")
        if staff.right - staff.left < 0.55 * width:
            reasons.append(f"staff {staff.index}: x coverage {staff.right - staff.left:.0f}")
    for i in range(len(staves) - 1):
        gap = staves[i + 1].center - staves[i].center
        if gap < 1.8 * max(staves[i].height, staves[i + 1].height):
            reasons.append(f"staves {i}/{i + 1} too close ({gap:.0f}px)")
    return not reasons, reasons


def oracle_staves(path: str | Path) -> list[Staff]:
    """Line positions reconstructed from the project's OpenCV detector."""
    if str(config.SCRIPTS) not in sys.path:
        sys.path.insert(0, str(config.SCRIPTS))
    import staves as oracle  # noqa: PLC0415

    out = []
    for i, (cx, cy, width, height, _ok) in enumerate(oracle.find_staves(str(path))):
        space = height / 4.0
        out.append(Staff(
            index=i,
            lines=[cy - 2 * space, cy - space, cy, cy + space, cy + 2 * space],
            left=cx - width / 2,
            right=cx + width / 2,
            space=space,
            thickness=max(1.0, height / 24.0),
            backend="oracle",
        ))
    return out


def oracle_mae(staves: list[Staff], path: str | Path) -> float | None:
    try:
        reference = oracle_staves(path)
    except Exception:
        return None
    if not reference:
        return None
    errors = []
    for staff in staves:
        ref = min(reference, key=lambda r: abs(r.center - staff.center))
        errors.extend(abs(a - b) for a, b in zip(staff.lines, ref.lines))
    return float(np.mean(errors)) if errors else None


def find_staves(page: StackPage, backend: str = "auto", *,
                allow_oracle: bool = True) -> tuple[list[Staff], dict]:
    """Find the (monophonic) staff systems of one stack page.

    Returns the staves and a diagnostics dict; raises StaffFindError if no
    backend validates and the OpenCV oracle is unavailable.

    `allow_oracle=False` is for corrected pages: the oracle and its x extents
    are measured on the original file, so they are in the wrong frame after a
    deskew.  In that mode the Gamera finders and `refine_extent` are used
    alone.
    """
    backends = ["projections", "miyao"] if backend == "auto" else [backend]
    tried = []
    for name in backends:
        try:
            staves = _run_finder(page.image, name)
        except Exception as exc:  # the finder itself failed
            tried.append({"backend": name, "error": f"{type(exc).__name__}: {exc}"})
            continue
        if allow_oracle:
            if not apply_oracle_extent(page, staves):
                for staff in staves:
                    refine_extent(page, staff)
        else:
            for staff in staves:
                refine_extent(page, staff)
        ok, reasons = validate_staves(staves, page.image)
        tried.append({
            "backend": name,
            "staves": len(staves),
            "spaces": [round(s.space, 1) for s in staves],
            "valid": ok,
            "reasons": reasons,
        })
        if ok:
            diag = {
                "backend": name,
                "staves": len(staves),
                "tried": tried,
                "oracle_mae": oracle_mae(staves, page.path) if allow_oracle else None,
            }
            if not allow_oracle:
                diag["oracle_frame"] = "disabled_deskewed_frame"
            return staves, diag

    if not allow_oracle:
        raise StaffFindError(f"all staff finders failed: {tried}")

    # Fall back to the proven OpenCV detector (positions only, no pixel removal).
    try:
        staves = oracle_staves(page.path)
    except Exception as exc:
        raise StaffFindError(f"all staff finders failed: {tried}") from exc
    ok, reasons = validate_staves(staves, page.image)
    if not ok:
        raise StaffFindError(f"all staff finders failed: {tried}; oracle: {reasons}")
    return staves, {"backend": "oracle", "staves": len(staves), "tried": tried,
                    "oracle_mae": None}


def refine_extent(page: StackPage, staff: Staff, min_coverage: int = 4) -> Staff:
    """Clamp left/right to the longest run where >= 4 of 5 lines are inked.

    The finder occasionally extrapolates a staff through instrument names or
    other text to the left of the real staff; this keeps components of that
    text out of the note stream.  The row window follows the staff space so a
    small residual skew/rounding error after deskewing does not truncate the
    extent.
    """
    arr = page.array()
    ys = [int(round(y)) for y in staff.lines]
    h, w = arr.shape
    tolerance = max(1, int(round(0.15 * staff.space)))
    good = np.zeros(w, dtype=bool)
    for x in range(w):
        count = 0
        for y in ys:
            if 0 <= y < h and arr[max(0, y - tolerance):y + tolerance + 1, x].any():
                count += 1
        good[x] = count >= min_coverage
    idx = np.flatnonzero(good)
    if len(idx) == 0:
        return staff
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]]
    k = int(np.argmax(ends - starts))
    start, end = int(starts[k]), int(ends[k])
    # Only adopt the refined extent when it is still wide enough to validate;
    # on a coarsely corrected page a residual bow can otherwise truncate it.
    if end - start >= 0.58 * arr.shape[1]:
        staff.left, staff.right = float(start), float(end)
    return staff


def apply_oracle_extent(page: StackPage, staves: list[Staff]) -> bool:
    """Use the OpenCV detector's robust left/right (it removes text first)."""
    try:
        reference = oracle_staves(page.path)
    except Exception:
        return False
    if not reference:
        return False
    for staff in staves:
        ref = min(reference, key=lambda r: abs(r.center - staff.center))
        if abs(ref.center - staff.center) < 2 * max(staff.space, 1.0):
            staff.left, staff.right = ref.left, ref.right
    return True


def _runs(column: np.ndarray) -> list[tuple[int, int]]:
    """Black (1) runs of a 1-D column, as inclusive (start, end) pairs."""
    idx = np.flatnonzero(column)
    if len(idx) == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]]
    return list(zip(starts.tolist(), ends.tolist()))


def _nearest_run(runs: list[tuple[int, int]], y: float, tolerance: float):
    best = None
    best_d = tolerance + 1
    for a, b in runs:
        if a - tolerance <= y <= b + tolerance:
            d = 0 if a <= y <= b else min(abs(y - a), abs(y - b))
            if d < best_d:
                best, best_d = (a, b), d
    return best


def _repair_line_crossings(arr: np.ndarray, staff: Staff):
    """Reconnect noteheads/stems split by erasing a staff line.

    The run-length rule removes the line band through a notehead that sits on
    a line, leaving its upper and lower halves as separate components.  Fill
    the erased band only where both sides are wide enough to be a notehead,
    so stems and beams are not merged.
    """
    h, w = arr.shape
    half = max(1, int(round(staff.thickness / 2.0)) + 1)
    min_width = max(2, int(round(0.30 * staff.space)))
    x0 = max(0, int(np.floor(staff.left)))
    x1 = min(w - 1, int(np.ceil(staff.right)))

    def run_width(row: np.ndarray, x: int) -> int:
        lo = x
        while lo > 0 and row[lo - 1]:
            lo -= 1
        hi = x
        while hi + 1 < len(row) and row[hi + 1]:
            hi += 1
        return hi - lo + 1

    for y in staff.lines:
        y = int(round(y))
        top = y - half - 1
        bottom = y + half + 1
        if top < 0 or bottom >= h:
            continue
        for x in range(x0, x1 + 1):
            if not (arr[top, x] and arr[bottom, x]):
                continue
            if (run_width(arr[top], x) < min_width
                    or run_width(arr[bottom], x) < min_width):
                continue
            arr[top + 1:bottom, x] = 1


def remove_staff_lines(page: StackPage, staves: list[Staff], tolerance: float = 1.0,
                       *, repair: bool = False):
    """Return a new image with the staff lines removed.

    A column's black run is erased when it is no thicker than the line plus
    tolerance.  Thicker runs (noteheads, clefs, stems, barlines) keep their
    non-line pixels; runs spanning most of the staff (barlines, stems) are left
    untouched.

    `repair=True` reconnects noteheads cut by the erased line.  It helps the
    old prints, where heads sit on thick lines, but slightly harms the clean
    NBA engravings, so callers choose it per edition.
    """
    arr = page.array().copy()
    for staff in staves:
        thick = max(1, int(round(staff.thickness)))
        x0 = max(0, int(round(staff.left)))
        x1 = min(arr.shape[1] - 1, int(round(staff.right)))
        height = staff.height
        for x in range(x0, x1 + 1):
            column = arr[:, x]
            runs = _runs(column)
            if not runs:
                continue
            for y in staff.lines:
                run = _nearest_run(runs, y, thick + tolerance)
                if run is None:
                    continue
                a, b = run
                length = b - a + 1
                if length >= 0.80 * height:
                    continue  # barline, clef, or long stem: keep it
                if length <= thick + 2 * tolerance:
                    column[a:b + 1] = 0
                else:
                    lo = max(a, int(round(y - thick / 2)))
                    hi = min(b, int(round(y + thick / 2)))
                    column[lo:hi + 1] = 0
    for staff in staves:
        if repair:
            _repair_line_crossings(arr, staff)
    return from_array(arr, page)
