"""Deskew unit tests: band detection, staff-space estimation, integer shear."""

from pathlib import Path

import numpy as np

from gamera_omr import deskew
from gamera_omr.stacks import StackPage, from_array


def synthetic_page(slopes=(0.0087, -0.0052, 0.0), space=14,
                   width=1200, margin=80, gap=120):
    """Build a page of independent systems with known staff-line slopes."""
    systems = len(slopes)
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
    for i, slope in enumerate(slopes):
        y0 = centers[i]
        band = ideal[y0:y0 + 4 * space + 2]
        for x in range(40, width - 40):
            d = int(round(slope * (x - x_ref)))
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
