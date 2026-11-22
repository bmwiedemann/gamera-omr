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

import cv2
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


@dataclass
class Trajectory:
    line_fits: list[LineFit]
    slope: float
    quadratic: float
    residual_rms: float
    space: float
    line_slope_spread_deg: float


def _open_profile(sub: np.ndarray, space: float) -> tuple[np.ndarray, np.ndarray]:
    """Horizontally open a strip, then return its row projection.

    Staff lines survive the opening even when slightly sloped; stems and
    most noteheads are narrower than the kernel.
    """
    k = max(3, int(round(1.2 * space)))
    k = min(k, max(3, sub.shape[1] - 1))
    kernel = np.ones((1, k), dtype=np.uint8)
    opened = cv2.morphologyEx(sub.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    return opened, opened.sum(axis=1).astype(float)


def strip_line_centers(
    arr: np.ndarray,
    band: Band,
    *,
    space: float,
    staff: Staff | None = None,
    coarse_slope: float = 0.0,
    x_ref: float | None = None,
    strip_width: int | None = None,
    min_peak_fraction: float = 0.2,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect the five staff lines in narrow vertical strips."""
    x0, x1 = band.x0, band.x1
    if x_ref is None:
        x_ref = 0.5 * (x0 + x1)
    width = strip_width or max(24, int(round(2.0 * space)))
    if x1 - x0 + 1 < width:
        return np.zeros(0), np.zeros((0, 5))
    xs: list[float] = []
    ys: list[list[float]] = []
    half = max(2.0, 0.6 * space)
    lag = max(4, int(round(space)))
    for xa in range(x0, x1 - width + 2, width):
        xc = xa + (width - 1) / 2
        sub = arr[band.y0:band.y1 + 1, xa:xa + width]
        _opened, prof = _open_profile(sub, space)
        if prof.max() <= 0:
            continue
        peaks: list[float] = []
        if staff is not None:
            for line in staff.lines:
                pred = line - band.y0 + coarse_slope * (xc - x_ref)
                lo = max(0, int(np.floor(pred - half)))
                hi = min(len(prof) - 1, int(np.ceil(pred + half)))
                if hi < lo or prof[lo:hi + 1].max() < min_peak_fraction * prof.max():
                    peaks.append(float("nan"))
                    continue
                peaks.append(float(lo + int(np.argmax(prof[lo:hi + 1]))))
        else:
            best: tuple[float, list[int]] | None = None
            for phase in range(lag):
                idxs = [phase + k * lag for k in range(5) if phase + k * lag < len(prof)]
                if len(idxs) < 5:
                    break
                score = float(sum(prof[i] for i in idxs))
                if best is None or score > best[0]:
                    best = (score, idxs)
            if best is None:
                continue
            if any(prof[i] < min_peak_fraction * prof.max() for i in best[1]):
                continue
            peaks = [float(i) for i in best[1]]
        xs.append(xc)
        ys.append(peaks)
    xs_arr = np.asarray(xs, dtype=float)
    ys_arr = np.asarray(ys, dtype=float).reshape(-1, 5)
    return xs_arr, ys_arr


def _fit_model(xs: np.ndarray, ys: np.ndarray, x_ref: float, degree: int,
               space: float) -> tuple[list[LineFit], float, float, float,
                                       np.ndarray] | None:
    """Robust shared-shape fit with one intercept per line.

    ``y_i(x) = a_i + m*u + q*u**2``, ``u = x - x_ref``.  Fitting in scaled
    coordinates keeps the quadratic column well conditioned.
    """
    n = ys.shape[1]
    finite = np.isfinite(ys)
    if finite.sum() < 10:
        return None
    usable = [i for i in range(n) if finite[:, i].sum() >= 3]
    if len(usable) < 3:
        return None
    scale = max(1.0, float(np.abs(xs - x_ref).max()))
    t = (xs - x_ref) / scale
    col = {line: k for k, line in enumerate(usable)}
    mask = finite.copy()

    def solve(current_mask):
        sample_line: list[int] = []
        sample_idx: list[int] = []
        target: list[float] = []
        for line in usable:
            for k in np.flatnonzero(current_mask[:, line]):
                sample_line.append(line)
                sample_idx.append(int(k))
                target.append(float(ys[k, line]))
        if len(target) < len(usable) + degree:
            return None
        line_arr = np.asarray(sample_line)
        idx_arr = np.asarray(sample_idx)
        yv = np.asarray(target)
        design = np.zeros((len(yv), len(usable) + degree))
        design[np.arange(len(yv)), [col[line] for line in line_arr]] = 1.0
        tt = t[idx_arr]
        design[:, len(usable)] = tt
        if degree == 2:
            design[:, len(usable) + 1] = tt * tt
        coef, *_ = np.linalg.lstsq(design, yv, rcond=None)
        return coef, line_arr, idx_arr, yv, design @ coef

    for _ in range(3):
        built = solve(mask)
        if built is None:
            return None
        _coef, line_arr, idx_arr, yv, pred = built
        keep = np.abs(yv - pred) <= max(1.0, 0.35 * space)
        if keep.sum() < max(10, len(usable) + degree):
            return None
        new_mask = np.zeros_like(finite)
        new_mask[idx_arr[keep], line_arr[keep]] = True
        mask = new_mask

    built = solve(mask)
    if built is None:
        return None
    coef, _line_arr, _idx_arr, yv, pred = built
    resid = yv - pred
    rms = float(np.sqrt(np.mean(resid ** 2)))
    intercepts = {line: float(coef[col[line]]) for line in usable}
    slope = float(coef[len(usable)] / scale)
    quad = float(coef[len(usable) + 1] / scale ** 2) if degree == 2 else 0.0
    fits = []
    for line in usable:
        sel = mask[:, line]
        if sel.sum() < 3:
            continue
        line_pred = (intercepts[line] + slope * (xs[sel] - x_ref)
                     + quad * (xs[sel] - x_ref) ** 2)
        line_rms = float(np.sqrt(np.mean((ys[sel, line] - line_pred) ** 2)))
        fits.append(LineFit(intercepts[line], slope, quad, line_rms,
                             int(sel.sum()), int(sel.sum())))
    if len(fits) < 3:
        return None
    return fits, slope, quad, rms, mask


def _independent_slope_spread(xs: np.ndarray, ys: np.ndarray,
                              mask: np.ndarray, x_ref: float) -> float:
    slopes = []
    for i in range(ys.shape[1]):
        sel = mask[:, i]
        if sel.sum() >= 4:
            p = np.polyfit(xs[sel] - x_ref, ys[sel, i], 1)
            slopes.append(p[0])
    if len(slopes) < 3:
        return 0.0
    ang = np.degrees(np.arctan(slopes))
    return float(ang.max() - ang.min())


def fit_band_trajectory(
    xs: np.ndarray,
    ys: np.ndarray,
    *,
    space: float,
    x_ref: float,
    options: DeskewOptions = DeskewOptions(),
) -> Trajectory | None:
    linear = _fit_model(xs, ys, x_ref, 1, space)
    if linear is None:
        return None
    fits, slope, _quad, rms, mask = linear
    quad = 0.0
    if rms > options.linear_residual_px:
        quadratic = _fit_model(xs, ys, x_ref, 2, space)
        if quadratic is not None:
            _fits2, slope2, quad2, rms2, mask2 = quadratic
            if (rms2 <= 0.85 * rms
                    or rms - rms2 >= options.quadratic_gain_px):
                fits, slope, quad, rms, mask = _fits2, slope2, quad2, rms2, mask2
    intercepts = sorted(fit.intercept for fit in fits)
    measured = (float(np.median(np.diff(intercepts)))
                if len(intercepts) >= 2 else space)
    spread = _independent_slope_spread(xs, ys, mask, x_ref)
    return Trajectory(fits, slope, quad, rms, measured, spread)


def _profile_sharpness(opened: np.ndarray, d: np.ndarray) -> float:
    h = opened.shape[0]
    pad = int(np.abs(d).max()) + 1 if d.size else 1
    idx = (np.arange(h)[:, None] + pad - d[None, :]).ravel()
    prof = np.bincount(idx, weights=opened.reshape(-1).astype(float),
                       minlength=h + 2 * pad)
    box = np.convolve(prof, np.ones(5) / 5.0, mode="same")
    return float(((prof - box) ** 2).sum())


def _gamera_angle_guesses(sub: np.ndarray, options: DeskewOptions) -> list[float]:
    try:
        img = from_array(np.ascontiguousarray(sub.astype(np.uint16)))
        result = img.rotation_angle_projections(-options.max_angle_deg,
                                                options.max_angle_deg)
        return [float(result[0])]
    except Exception:
        return []


def coarse_band_skew(
    arr: np.ndarray,
    band: Band,
    *,
    space: float | None = None,
    options: DeskewOptions = DeskewOptions(),
) -> BandSkew | None:
    """Projection-profile angle estimate, used as a bootstrap or fallback."""
    space = float(space or band.space or estimate_staff_space(arr, band) or 0)
    if not 8 <= space <= 40:
        return None
    sub = arr[band.y0:band.y1 + 1, band.x0:band.x1 + 1]
    opened, _prof = _open_profile(sub, space)
    x_ref = 0.5 * (band.x0 + band.x1)
    d_x = np.arange(band.x0, band.x1 + 1) - x_ref
    max_t = float(np.tan(np.radians(options.max_angle_deg)))
    coarse = np.linspace(-max_t, max_t, 17)
    guesses = _gamera_angle_guesses(sub, options)
    if guesses:
        extra = [np.tan(np.radians(a)) for a in guesses]
        coarse = np.unique(np.r_[coarse, extra, -np.asarray(extra)])
    scores = [_profile_sharpness(opened, np.rint(s * d_x).astype(int))
              for s in coarse]
    best = float(coarse[int(np.argmax(scores))])
    step = float(coarse[1] - coarse[0]) if len(coarse) > 1 else max_t / 8
    fine = np.linspace(best - step, best + step, 9)
    scores = [_profile_sharpness(opened, np.rint(s * d_x).astype(int))
              for s in fine]
    slope = float(fine[int(np.argmax(scores))])
    plan = BandSkew(band=band, x_ref=x_ref, space=space, slope=slope,
                    angle_deg=float(np.degrees(np.arctan(slope))),
                    model="projection")
    plan.shifts = _shifts_for(plan, arr.shape[1])
    return _finalize(plan, options)


def _shifts_for(plan: BandSkew, width: int) -> np.ndarray:
    shifts = np.zeros(width, dtype=np.int32)
    xs = np.arange(plan.band.x0, plan.band.x1 + 1)
    values = plan.correction(xs)
    shifts[plan.band.x0:plan.band.x1 + 1] = np.rint(values).astype(np.int32)
    return shifts


def _finalize(plan: BandSkew, options: DeskewOptions) -> BandSkew:
    band = plan.band
    xs = np.array([band.x0, 0.5 * (band.x0 + band.x1), band.x1], dtype=float)
    c = np.asarray(plan.correction(xs), dtype=float)
    plan.edge_deviation_px = float(c.max() - c.min())
    plan.edge_deviation_spaces = plan.edge_deviation_px / max(plan.space, 1e-6)
    plan.curvature_px = float(abs(c[1] - 0.5 * (c[0] + c[2])))
    plan.angle_deg = float(np.degrees(np.arctan(plan.slope)))
    if (plan.edge_deviation_spaces < options.min_edge_spaces
            and plan.edge_deviation_px < options.min_edge_px):
        plan.applied = False
        plan.reason = "below_threshold"
    elif abs(plan.angle_deg) > options.max_angle_deg:
        plan.applied = False
        plan.reason = "angle_too_large"
    elif plan.edge_deviation_spaces > options.max_edge_spaces:
        plan.applied = False
        plan.reason = "edge_too_large"
    elif plan.curvature_px / max(plan.space, 1e-6) > options.max_curvature_spaces:
        plan.applied = False
        plan.reason = "curvature_too_large"
    elif (plan.shifts.size and
          np.abs(plan.shifts).max() > options.max_shift_spaces * plan.space):
        plan.applied = False
        plan.reason = "shift_too_large"
    elif (plan.residual_rms_px is not None
          and plan.residual_rms_px > options.max_band_residual_px):
        plan.applied = False
        plan.reason = "high_residual"
    elif plan.line_slope_spread_deg > options.max_line_slope_spread_deg:
        plan.applied = False
        plan.reason = "line_slope_spread"
    else:
        plan.applied = True
        plan.reason = None
    return plan


def fit_band_skew(
    arr: np.ndarray,
    band: Band,
    *,
    staff: Staff | None = None,
    space: float | None = None,
    options: DeskewOptions = DeskewOptions(),
) -> BandSkew | None:
    """Estimate one band's staff-centre trajectory and shear plan."""
    space = float(space or (staff.space if staff else 0)
                  or band.space or estimate_staff_space(arr, band) or 0)
    if not 8 <= space <= 40:
        return None
    x_ref = 0.5 * (band.x0 + band.x1)
    coarse = coarse_band_skew(arr, band, space=space, options=options)
    coarse_slope = coarse.slope if coarse is not None else 0.0
    xs, ys = strip_line_centers(arr, band, space=space, staff=staff,
                                coarse_slope=coarse_slope, x_ref=x_ref)
    trajectory = fit_band_trajectory(xs, ys, space=space, x_ref=x_ref,
                                     options=options)
    if trajectory is None:
        return coarse
    if not 0.75 * space <= trajectory.space <= 1.25 * space:
        return coarse
    plan = BandSkew(
        band=band, x_ref=x_ref, space=trajectory.space,
        slope=trajectory.slope, quadratic=trajectory.quadratic,
        residual_rms_px=trajectory.residual_rms,
        model="quadratic" if trajectory.quadratic else "linear",
        line_fits=trajectory.line_fits,
        line_slope_spread_deg=trajectory.line_slope_spread_deg,
    )
    plan.shifts = _shifts_for(plan, arr.shape[1])
    return _finalize(plan, options)


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
