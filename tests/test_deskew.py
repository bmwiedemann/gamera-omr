"""Deskew unit tests: band detection, staff-space estimation, integer shear."""

from pathlib import Path

import numpy as np
import pytest

from gamera_omr import config, deskew
from gamera_omr.stacks import StackPage, from_array, load_stack
from gamera_omr.staffs import Staff, find_staves, validate_staves


def synthetic_page(slopes=(0.0087, -0.0052, 0.0), space=14,
                   width=1200, margin=80, gap=120, bows=None):
    """Build a page of independent systems with known slopes and bows."""
    systems = len(slopes)
    bows = bows or tuple(0.0 for _ in slopes)
    height = margin * 2 + systems * (4 * space + 1) + (systems - 1) * gap + 60
    ideal = np.zeros((height, width), dtype=np.uint16)
    centers = []
    for i in range(systems):
        y = margin + i * (4 * space + 1 + gap)
        centers.append(y)
        for k in range(5):
            ideal[y + k * space:y + k * space + 2, 40:width - 40] = 1
    skewed = np.zeros_like(ideal)
    x_ref = width / 2
    half = (width - 40) / 2
    for i, (slope, bow) in enumerate(zip(slopes, bows)):
        y0 = centers[i]
        band = ideal[y0:y0 + 4 * space + 2]
        for x in range(40, width - 40):
            q = (x - x_ref) / half
            d = int(round(slope * (x - x_ref) + bow * q * q))
            col = band[:, x]
            rows = np.flatnonzero(col)
            dst = rows + y0 + d
            ok = (dst >= 0) & (dst < height)
            skewed[dst[ok], x] = col[rows[ok]]
    page = StackPage(Path("/nonexistent/synthetic.png"), from_array(skewed), [])
    return page, skewed, centers


def line_y(arr, x, y_hint, space):
    col = arr[:, x]
    lo = max(0, int(y_hint - 1.5 * space))
    hi = min(len(col), int(y_hint + 1.5 * space))
    idx = np.flatnonzero(col[lo:hi])
    if len(idx) == 0:
        return None
    return lo + float(idx.mean())


def test_band_detection_finds_each_system():
    _page, arr, centers = synthetic_page()
    bands = deskew.find_system_bands(arr)
    assert len(bands) == 3
    for band, y in zip(bands, centers):
        assert abs(band.y0 - y) <= 8
        assert abs(band.y1 - (y + 4 * 14 + 1)) <= 8


def test_band_detection_uses_staff_windows():
    page, arr, centers = synthetic_page()
    from gamera_omr.staffs import Staff

    staves = [Staff(i, [float(y + k * 14) for k in range(5)], 40, 1160, 14, 2)
              for i, y in enumerate(centers)]
    bands = deskew.find_system_bands(arr, staves=staves)
    assert len(bands) == 3
    assert all(band.space == 14 for band in bands)


def test_staff_space_from_projection():
    _page, arr, _centers = synthetic_page()
    for band in deskew.find_system_bands(arr):
        assert deskew.estimate_staff_space(arr, band) == 14


def test_shear_flattens_known_lines_and_keeps_x():
    slopes = (0.0087, -0.0052, 0.0)
    page, skewed, centers = synthetic_page(slopes=slopes)
    bands = deskew.find_system_bands(skewed)
    width = skewed.shape[1]
    x_ref = width / 2
    plans = []
    for band, slope in zip(bands, slopes):
        shifts = np.rint(slope * (np.arange(width) - x_ref)).astype(np.int32)
        plan = deskew.BandSkew(band=band, x_ref=x_ref, space=14.0,
                               slope=slope, shifts=shifts, applied=True)
        plans.append(plan)
    fixed = deskew.apply_shear(page, plans).array()

    left, right = 80, width - 80
    for y, slope in zip(centers, slopes):
        for k in range(5):
            before = (line_y(skewed, left, y + k * 14, 14),
                      line_y(skewed, right, y + k * 14, 14))
            assert before[0] is not None and before[1] is not None
            if abs(slope) * (right - left) > 3:
                assert abs(before[0] - before[1]) > 2
            after = (line_y(fixed, left, y + k * 14, 14),
                     line_y(fixed, right, y + k * 14, 14))
            assert after[0] is not None and after[1] is not None
            assert abs(after[0] - after[1]) <= 1.0
    assert fixed.shape == skewed.shape


def test_fit_trajectory_recovers_slope_and_bow():
    xs = np.arange(50, 1150, 30, dtype=float)
    x_ref = 600.0
    space = 19.5
    slope = 0.012
    bow = 4.0
    quad = bow / 550.0 ** 2
    ys = np.column_stack([
        100 + k * space + slope * (xs - x_ref) + quad * (xs - x_ref) ** 2
        for k in range(5)
    ])
    ys += np.random.default_rng(0).normal(0, 0.15, ys.shape)
    traj = deskew.fit_band_trajectory(xs, ys, space=space, x_ref=x_ref)
    assert traj is not None
    assert abs(traj.slope - slope) < 0.001
    assert abs(traj.quadratic - quad) <= 0.25 * quad
    assert traj.residual_rms < 0.5
    assert abs(traj.space - space) < 0.5


def test_coarse_projection_recovers_injected_slope():
    slopes = (0.0087, -0.0052)
    page, arr, _centers = synthetic_page(slopes=slopes)
    bands = deskew.find_system_bands(arr)
    for band, slope in zip(bands, slopes):
        plan = deskew.coarse_band_skew(arr, band, space=14)
        assert plan is not None
        assert abs(plan.slope - slope) < 0.001
        assert plan.applied


def test_fit_band_skew_with_staff_hints():
    from gamera_omr.staffs import Staff

    slopes = (0.0087, -0.0052, 0.0)
    page, arr, centers = synthetic_page(slopes=slopes)
    bands = deskew.find_system_bands(arr)
    for band, y, slope in zip(bands, centers, slopes):
        staff = Staff(band.index, [float(y + k * 14) for k in range(5)],
                      40, 1160, 14, 2)
        plan = deskew.fit_band_skew(arr, band, staff=staff)
        assert plan is not None
        assert abs(plan.slope - slope) < 0.001
        assert plan.model in ("linear", "quadratic")


def test_bowed_band_selects_quadratic_and_flattens():
    slopes = (0.0, -0.0052, 0.0)
    bows = (0.0, 4.0, 0.0)
    page, arr, centers = synthetic_page(slopes=slopes, bows=bows)
    bands = deskew.find_system_bands(arr)
    plan = deskew.fit_band_skew(arr, bands[1], space=14)
    assert plan is not None
    assert plan.model == "quadratic"
    assert plan.applied
    fixed = deskew.apply_shear(page, [plan]).array()
    y = centers[1]
    left, right = 80, 1120
    for k in range(5):
        a = line_y(fixed, left, y + k * 14, 14)
        b = line_y(fixed, right, y + k * 14, 14)
        assert a is not None and b is not None
        assert abs(a - b) <= 1.5


def test_deskew_page_with_staff_hints():
    slopes = (0.0087, -0.0052, 0.0)
    bows = (0.0, 4.0, 0.0)
    page, _arr, centers = synthetic_page(slopes=slopes, bows=bows)
    staves = [Staff(i, [float(y + k * 14) for k in range(5)], 40, 1160, 14, 2)
              for i, y in enumerate(centers)]
    result = deskew.deskew_page(page, initial_staves=staves,
                                initial_diag={"backend": "synthetic"})
    assert result.applied
    assert result.staves is not None and len(result.staves) == 3
    assert "quadratic" in [b["model"] for b in result.diagnostics["bands"]]
    assert result.diagnostics["aggregate"]["applied_bands"] == 2
    assert result.diagnostics["post"]["staves"] == 3


def test_deskew_page_without_staff_hints():
    page, _arr, _centers = synthetic_page(bows=(0.0, 4.0, 0.0))
    result = deskew.deskew_page(page)
    assert result.applied
    assert result.staves is not None and len(result.staves) == 3
    assert result.diagnostics["reason"] is None


def test_flat_page_is_a_noop():
    page, arr, _centers = synthetic_page(slopes=(0.0, 0.0, 0.0),
                                        bows=(0.0, 0.0, 0.0))
    result = deskew.deskew_page(page)
    assert not result.applied
    assert result.diagnostics["reason"] == "below_threshold"
    assert np.array_equal(result.page.array(), arr)


SIBLEY1 = config.WORK / "stacks" / "sibley" / "S-01.png"
SIBLEY2 = config.WORK / "stacks" / "sibley" / "S-02.png"
ROSLER1 = config.WORK / "stacks" / "rosler" / "S-01.png"
CHORALE = config.WORK / "mvt" / "5" / "stacks" / "nba" / "S-01.png"


def _needs(path):
    return pytest.mark.skipif(not path.exists(), reason=f"missing test image {path}")


@_needs(SIBLEY1)
def test_prepare_page_corrects_large_skew():
    page, staves, diag = deskew.prepare_page(SIBLEY1)
    assert diag["deskew"]["applied"]
    assert len(staves) == 6
    assert diag["deskew"]["aggregate"]["max_edge_spaces"] > 0.5
    ok, reasons = validate_staves(staves, page.image)
    assert ok, reasons


@_needs(SIBLEY2)
def test_prepare_page_bootstraps_when_raw_finder_fails():
    page, staves, diag = deskew.prepare_page(SIBLEY2)
    assert diag["deskew"]["applied"]
    assert len(staves) == 6
    assert diag["oracle_frame"] == "unavailable"


@_needs(ROSLER1)
def test_prepare_page_keeps_rosler_geometry():
    page, staves, diag = deskew.prepare_page(ROSLER1)
    assert diag["deskew"]["applied"]
    assert len(staves) == 6
    ok, reasons = validate_staves(staves, page.image)
    assert ok, reasons


@_needs(CHORALE)
def test_prepare_page_preserves_oracle_mae_for_nba():
    raw = load_stack(CHORALE)
    _raw_staves, raw_diag = find_staves(raw)
    _page, staves, diag = deskew.prepare_page(CHORALE)
    assert len(staves) == 3
    assert diag["oracle_frame"] in ("pre_deskew", "original")
    assert diag["oracle_mae"] == raw_diag["oracle_mae"]
