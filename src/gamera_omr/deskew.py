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
from pathlib import Path

import cv2
import numpy as np

from .stacks import StackPage, from_array, load_stack
from .staffs import Staff, StaffFindError, find_staves


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


@dataclass
class DeskewResult:
    page: StackPage
    staves: list[Staff] | None
    applied: bool
    bands: list[BandSkew]
    diagnostics: dict
    staff_diag: dict = field(default_factory=dict)


def _band_diag(plan: BandSkew) -> dict:
    shifts = plan.shifts[plan.band.x0:plan.band.x1 + 1]
    return {
        "index": plan.band.index,
        "rows": [plan.band.y0, plan.band.y1],
        "x": [plan.band.x0, plan.band.x1],
        "space": round(plan.space, 1),
        "angle_deg": round(plan.angle_deg, 3),
        "slope": round(plan.slope, 6),
        "quadratic": plan.quadratic,
        "model": plan.model,
        "edge_deviation_px": round(plan.edge_deviation_px, 2),
        "edge_deviation_spaces": round(plan.edge_deviation_spaces, 2),
        "curvature_px": round(plan.curvature_px, 2),
        "residual_rms_px": (round(plan.residual_rms_px, 2)
                            if plan.residual_rms_px is not None else None),
        "shift_min_px": int(shifts.min()) if shifts.size else 0,
        "shift_max_px": int(shifts.max()) if shifts.size else 0,
        "line_slope_spread_deg": round(plan.line_slope_spread_deg, 3),
        "applied": plan.applied,
        "reason": plan.reason,
    }


def _aggregate(plans: list[BandSkew]) -> dict:
    applied = [p for p in plans if p.applied]
    residuals = [p.residual_rms_px for p in applied
                 if p.residual_rms_px is not None]
    reasons: dict[str, int] = {}
    for plan in plans:
        if not plan.applied and plan.reason:
            reasons[plan.reason] = reasons.get(plan.reason, 0) + 1
    return {
        "bands": len(plans),
        "applied_bands": len(applied),
        "max_angle_deg": (round(max(abs(p.angle_deg) for p in applied), 3)
                          if applied else 0.0),
        "max_edge_spaces": (round(max(p.edge_deviation_spaces for p in applied), 2)
                            if applied else 0.0),
        "max_curvature_px": (round(max(p.curvature_px for p in applied), 2)
                             if applied else 0.0),
        "max_residual_rms_px": (round(max(residuals), 2) if residuals else None),
        "reasons": reasons,
    }


def _summary(staves: list[Staff] | None, diag: dict | None,
             error: str | None = None) -> dict:
    if staves is None:
        return {"found": False, "staves": 0, "backend": None,
                "valid": False, "error": error}
    return {"found": True, "staves": len(staves),
            "backend": (diag or {}).get("backend"), "valid": True}


def _deskew_diag(plans: list[BandSkew], *, applied: bool, reason: str | None,
                 pre: dict, post: dict, passes: int = 1) -> dict:
    return {
        "enabled": True,
        "applied": applied,
        "reason": reason,
        "passes": passes,
        "pre": pre,
        "post": post,
        "aggregate": _aggregate(plans),
        "bands": [_band_diag(p) for p in plans],
    }


def _post_quality(arr: np.ndarray, staves: list[Staff],
                  options: DeskewOptions) -> list[BandSkew | None]:
    bands = find_system_bands(arr, staves=staves, options=options)
    plans: list[BandSkew | None] = []
    for band, staff in zip(bands, staves):
        plans.append(fit_band_skew(arr, band, staff=staff, options=options))
    return plans


def _quality(plans: list[BandSkew | None], indices: set[int]) -> tuple[float, float | None]:
    edges = []
    residuals = []
    for plan in plans:
        if plan is None or plan.band.index not in indices:
            continue
        edges.append(plan.edge_deviation_px)
        if plan.residual_rms_px is not None:
            residuals.append(plan.residual_rms_px)
    if len(edges) != len(indices):
        return float("inf"), None
    return max(edges), (max(residuals) if residuals else None)


def _accepted(plans: list[BandSkew], post: list[BandSkew | None]) -> bool:
    applied = {p.band.index for p in plans if p.applied}
    if not applied:
        return False
    pre_edge, pre_res = _quality(plans, applied)
    post_edge, post_res = _quality(post, applied)
    edge_ok = post_edge <= max(1.0, 0.9 * pre_edge)
    res_ok = (pre_res is not None and post_res is not None
              and post_res <= max(1.0, 0.85 * pre_res))
    return edge_ok or res_ok


def _rejected(page: StackPage, staves: list[Staff] | None,
              plans: list[BandSkew], reason: str, pre: dict,
              post: dict | None = None, passes: int = 1) -> DeskewResult:
    return DeskewResult(page, staves, False, plans,
                        _deskew_diag(plans, applied=False, reason=reason,
                                     pre=pre, post=post or {}, passes=passes))


def deskew_page(
    page: StackPage,
    *,
    backend: str = "auto",
    initial_staves: list[Staff] | None = None,
    initial_diag: dict | None = None,
    options: DeskewOptions = DeskewOptions(),
) -> DeskewResult:
    """Estimate per-system shear, apply it, and validate the result.

    With staff hints, correction is accepted only when the corrected page
    still yields the same number of validated staves and the fitted residual
    or edge deviation improves.  Without hints (the raw finder failed), a
    coarse per-band projection shear is used to bootstrap staff finding.
    """
    arr = page.array()
    if initial_staves:
        pre = _summary(initial_staves, initial_diag)
        bands = find_system_bands(arr, staves=initial_staves, options=options)
        plans = [fit_band_skew(arr, band, staff=staff, options=options)
                 for band, staff in zip(bands, initial_staves)]
        plans = [p for p in plans if p is not None]
        if not plans:
            return _rejected(page, initial_staves, [], "no_fit", pre)
        if not any(p.applied for p in plans):
            return _rejected(page, initial_staves, plans, "below_threshold", pre)
        try:
            candidate = apply_shear(page, plans)
        except ValueError:
            return _rejected(page, initial_staves, plans, "bands_overlap", pre)
        try:
            post_staves, post_diag = find_staves(candidate, backend,
                                                 allow_oracle=False)
        except StaffFindError as exc:
            return _rejected(page, initial_staves, plans,
                             "staff_finding_failed_after_deskew", pre,
                             {"error": str(exc)})
        post = _summary(post_staves, post_diag)
        if len(post_staves) != len(initial_staves):
            return _rejected(page, initial_staves, plans, "staff_count_changed",
                             pre, post)
        post_plans = _post_quality(candidate.array(), post_staves, options)
        if not _accepted(plans, post_plans):
            return _rejected(page, initial_staves, plans, "post_quality_regressed",
                             pre, post)
        return DeskewResult(candidate, post_staves, True, plans,
                            _deskew_diag(plans, applied=True, reason=None,
                                         pre=pre, post=post),
                            post_diag)

    return _deskew_without_staves(page, backend, options)


def _deskew_without_staves(page: StackPage, backend: str,
                           options: DeskewOptions) -> DeskewResult:
    """Bootstrap staff finding: fit each band without staff hints, then shear.

    The robust strip fit does not actually need staff hints (the comb of five
    lines provides the line identities), so it can correct bowed systems that
    a linear projection angle alone cannot.  The coarse projection plan is
    only used when the trajectory fit is unavailable.
    """
    arr = page.array()
    pre = {"found": False, "staves": 0, "backend": None, "valid": False}
    bands = find_system_bands(arr, options=options)
    precise = [fit_band_skew(arr, band, options=options) for band in bands]
    precise = [p for p in precise if p is not None]
    fitted = [p for p in precise if p.model != "projection"]
    # The hint-free comb fit is reliable for moderately skewed systems; a
    # majority of projection-only bands means it could not identify the five
    # lines, so bootstrap staff finding with the coarse shear instead.
    if fitted and 2 * len(fitted) >= len(precise) and any(p.applied for p in fitted):
        try:
            candidate = apply_shear(page, fitted)
        except ValueError:
            candidate = None
        if candidate is not None:
            try:
                staves, staff_diag = find_staves(candidate, backend,
                                                 allow_oracle=False)
            except StaffFindError:
                staves = None
            if staves is not None:
                post = _summary(staves, staff_diag)
                return DeskewResult(
                    candidate, staves, True, fitted,
                    _deskew_diag(fitted, applied=True, reason=None,
                                 pre=pre, post=post, passes=1),
                    staff_diag)

    coarse = [coarse_band_skew(arr, band, options=options) for band in bands]
    coarse = [p for p in coarse if p is not None]
    if not any(p.applied for p in coarse):
        return _rejected(page, None, precise or coarse, "below_threshold", pre)
    try:
        coarse_page = apply_shear(page, coarse)
    except ValueError:
        return _rejected(page, None, precise or coarse, "bands_overlap", pre)
    try:
        coarse_staves, coarse_diag = find_staves(coarse_page, backend,
                                                 allow_oracle=False)
    except StaffFindError as exc:
        plans = precise or coarse
        return _rejected(page, None, plans, "staff_finding_failed_after_deskew",
                         pre, {"error": str(exc)})

    # Refit precise trajectories on the original page, using the coarse
    # staves for line identities.  At x_ref the coarse shear is zero, so the
    # staff line positions are still valid for band/prediction.
    orig_bands = find_system_bands(arr, staves=coarse_staves, options=options)
    refined: list[BandSkew] = []
    for band, staff in zip(orig_bands, coarse_staves):
        plan = fit_band_skew(arr, band, staff=staff, options=options)
        if plan is not None:
            refined.append(plan)
    if any(p.applied for p in refined):
        try:
            refined_page = apply_shear(page, refined)
        except ValueError:
            refined_page = None
        if refined_page is not None:
            try:
                refined_staves, refined_diag = find_staves(
                    refined_page, backend, allow_oracle=False)
            except StaffFindError:
                refined_staves = None
            if (refined_staves is not None
                    and len(refined_staves) == len(coarse_staves)):
                post = _summary(refined_staves, refined_diag)
                return DeskewResult(
                    refined_page, refined_staves, True, refined,
                    _deskew_diag(refined, applied=True, reason=None,
                                 pre=pre, post=post, passes=2),
                    refined_diag)

    post = _summary(coarse_staves, coarse_diag)
    return DeskewResult(
        coarse_page, coarse_staves, True, coarse,
        _deskew_diag(coarse, applied=True, reason=None, pre=pre,
                     post=post, passes=1),
        coarse_diag)


def prepare_page(
    page_or_path,
    *,
    backend: str = "auto",
    enabled: bool = True,
    options: DeskewOptions = DeskewOptions(),
) -> tuple[StackPage, list[Staff], dict]:
    """Load, deskew and staff-find a page; return merged diagnostics."""
    page = (load_stack(page_or_path)
            if isinstance(page_or_path, (str, Path)) else page_or_path)
    if not enabled:
        staves, diag = find_staves(page, backend)
        diag["deskew"] = {
            "enabled": False, "applied": False, "reason": "disabled",
            "passes": 0, "pre": {}, "post": {}, "aggregate": {}, "bands": [],
        }
        return page, staves, diag

    pre_staves: list[Staff] | None = None
    pre_diag: dict = {}
    pre_error: str | None = None
    try:
        pre_staves, pre_diag = find_staves(page, backend)
    except StaffFindError as exc:
        pre_error = str(exc)

    result = deskew_page(page, backend=backend, initial_staves=pre_staves,
                         initial_diag=pre_diag, options=options)
    if result.staves is None:
        raise StaffFindError(pre_error or "; ".join(
            str(b.get("error")) for b in result.diagnostics.get("bands", [])))

    if pre_staves is not None:
        diag = dict(result.staff_diag) if result.applied else dict(pre_diag)
        diag["oracle_mae"] = pre_diag.get("oracle_mae")
        diag["oracle_frame"] = "pre_deskew" if result.applied else "original"
    else:
        diag = dict(result.staff_diag)
        diag["oracle_mae"] = None
        diag["oracle_frame"] = "unavailable"
    diag["deskew"] = result.diagnostics
    return result.page, result.staves, diag


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
