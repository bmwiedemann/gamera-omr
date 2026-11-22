"""Per-system skew correction by integer vertical shear.

Each stack page is a vertical concatenation of independently cropped systems,
so every system can have its own skew.  The engine stores staff lines as
scalar y values; a sloped line drifts out of the removal window away from the
centre and biases pitch with x.  Correcting before staff finding/removal makes
the existing constant-y model valid again.

A vertical shear ``y' = y - d(x)`` is used instead of a rotation: it preserves
x (measures, x-rhythm and component x positions), keeps vertical barlines and
stems vertical, and can be applied as integer per-column row shifts without
resampling.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .stacks import StackPage, from_array
from .staffs import Staff


@dataclass(frozen=True)
class DeskewOptions:
    enabled: bool = True
    merge_gap: int = 48           # blank gap still belonging to one system
    min_band_height: int = 40     # px; drops lyric/text slivers
    max_bands: int = 6
    min_edge_spaces: float = 0.20  # auto no-op guard (NBA is below this)
    min_edge_px: float = 1.5
    max_angle_deg: float = 1.5
    max_edge_spaces: float = 2.5
    max_curvature_spaces: float = 0.75
    max_shift_spaces: float = 2.5
    linear_residual_px: float = 0.75
    quadratic_gain_px: float = 0.25
    max_line_slope_spread_deg: float = 0.15
    max_band_residual_px: float = 1.5
    keep_original_on_regression: bool = True
    max_passes: int = 2


@dataclass
class Band:
    index: int
    y0: int
    y1: int
    x0: int
    x1: int
    ink: int
    space: float | None = None

    @property
    def height(self) -> int:
        return self.y1 - self.y0 + 1


@dataclass
class LineFit:
    intercept: float
    slope: float
    quadratic: float
    residual_rms: float
    samples: int
    inliers: int


@dataclass
class BandSkew:
    band: Band
    x_ref: float
    space: float
    slope: float = 0.0
    quadratic: float = 0.0
    angle_deg: float = 0.0
    edge_deviation_px: float = 0.0
    edge_deviation_spaces: float = 0.0
    curvature_px: float = 0.0
    residual_rms_px: float | None = None
    model: str = "none"
    shifts: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int32))
    applied: bool = False
    reason: str | None = None
    line_fits: list[LineFit] = field(default_factory=list)
    line_slope_spread_deg: float = 0.0

    def correction(self, x: np.ndarray | float) -> np.ndarray | float:
        u = np.asarray(x, dtype=float) - self.x_ref
        return self.slope * u + self.quadratic * u * u


def _ink_runs(mask: np.ndarray, min_gap: int) -> list[tuple[int, int]]:
    """Inclusive runs of True values, merging gaps smaller than `min_gap`."""
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.r_[idx[0], idx[breaks + 1]]
    ends = np.r_[idx[breaks], idx[-1]]
    runs: list[list[int]] = [[int(starts[0]), int(ends[0])]]
    for a, b in zip(starts[1:], ends[1:]):
        if int(a) - runs[-1][1] - 1 < min_gap:
            runs[-1][1] = int(b)
        else:
            runs.append([int(a), int(b)])
    return [(a, b) for a, b in runs]


def _x_extent(arr: np.ndarray, y0: int, y1: int) -> tuple[int, int, int]:
    sub = arr[y0:y1 + 1]
    cols = np.flatnonzero(sub.any(axis=0))
    if len(cols) == 0:
        return 0, arr.shape[1] - 1, 0
    return int(cols[0]), int(cols[-1]), int(sub.sum())


def _band_from_rows(arr: np.ndarray, index: int, y0: int, y1: int,
                    space: float | None = None) -> Band:
    y0 = max(0, y0)
    y1 = min(arr.shape[0] - 1, y1)
    x0, x1, ink = _x_extent(arr, y0, y1)
    return Band(index, y0, y1, x0, x1, ink, space)


def find_system_bands(
    arr: np.ndarray,
    *,
    staves: list[Staff] | None = None,
    options: DeskewOptions = DeskewOptions(),
) -> list[Band]:
    """Row bands, one per system, separated by blank paper.

    With staff hints the band is the staff window, snapped to nearby blank
    rows.  Without hints the bands come from blank-row segmentation, which is
    reliable because stack.py pastes the per-system crops with large gaps.
    """
    h = arr.shape[0]
    blank = arr.sum(axis=1) == 0
    if staves:
        windows = []
        for staff in staves:
            top, bottom = staff.band(above=4.5, below=2.5)
            y0 = max(0, int(np.floor(top)))
            y1 = min(h - 1, int(np.ceil(bottom)))
            # Snap to a nearby blank row so the whole system (including
            # lyrics) moves with the staff; stay within a small margin.
            limit = max(4, int(round(2.0 * staff.space)))
            for _ in range(limit):
                if y0 > 0 and not blank[y0]:
                    y0 -= 1
                else:
                    break
            for _ in range(limit):
                if y1 < h - 1 and not blank[y1]:
                    y1 += 1
                else:
                    break
            windows.append((y0, y1, staff.space))
        windows.sort()
        merged: list[list] = []
        for y0, y1, space in windows:
            # One band per staff even when their generous windows nearly
            # touch; the 120 px stack gaps keep adjacent systems separate.
            if merged and y0 <= merged[-1][1] + 1:
                merged[-1][1] = max(merged[-1][1], y1)
            else:
                merged.append([y0, y1, space])
        bands = []
        for y0, y1, space in merged:
            band = _band_from_rows(arr, len(bands), int(y0), int(y1), space)
            if band.ink:
                bands.append(band)
        return bands

    runs = _ink_runs(~blank, options.merge_gap)
    bands = []
    for y0, y1 in runs:
        if y1 - y0 + 1 < options.min_band_height:
            continue
        band = _band_from_rows(arr, len(bands), y0, y1)
        if band.ink:
            bands.append(band)
        if len(bands) >= options.max_bands:
            break
    return bands


def estimate_staff_space(arr: np.ndarray, band: Band) -> float | None:
    """Staff space from the periodicity of the row projection."""
    prof = arr[band.y0:band.y1 + 1, band.x0:band.x1 + 1].sum(axis=1).astype(float)
    prof = prof - prof.min()
    if prof.max() <= 0:
        return None
    prof /= prof.max()
    best: tuple[float, int] | None = None
    max_lag = min(40, (len(prof) - 2) // 4)
    for lag in range(6, max_lag + 1):
        a = prof[:-lag]
        b = prof[lag:]
        denom = float(np.sqrt((a * a).sum() * (b * b).sum()))
        if denom <= 0:
            continue
        score = float((a * b).sum() / denom)
        if best is None or score > best[0]:
            best = (score, lag)
    if best is None or best[0] < 0.30:
        return None
    return float(best[1])


def apply_shear(page: StackPage, plans: list[BandSkew], *,
                fill: int = 0) -> StackPage:
    """Shear each band by its planned column shifts."""
    arr = page.array()
    out = arr.copy()
    h, w = arr.shape
    active = [p for p in plans if p.applied]
    targets = []
    for plan in active:
        shifts = plan.shifts
        maxs = int(np.abs(shifts).max()) if shifts.size else 0
        targets.append((plan.band.y0 - maxs, plan.band.y1 + maxs))
    targets.sort()
    for (a0, a1), (b0, _b1) in zip(targets, targets[1:]):
        if b0 <= a1:
            raise ValueError("deskew bands overlap after shifting")

    for plan in active:
        band = plan.band
        shifts = plan.shifts
        maxs = int(np.abs(shifts).max()) if shifts.size else 0
        lo = max(0, band.y0 - maxs)
        hi = min(h - 1, band.y1 + maxs)
        out[lo:hi + 1, band.x0:band.x1 + 1] = fill
        for x in range(band.x0, band.x1 + 1):
            d = int(shifts[x])
            src_lo = max(band.y0, d)
            src_hi = min(band.y1, h - 1 + d)
            if src_lo > src_hi:
                continue
            out[src_lo - d:src_hi - d + 1, x] = arr[src_lo:src_hi + 1, x]
    return from_array(np.ascontiguousarray(out.astype(np.uint16)), page)
