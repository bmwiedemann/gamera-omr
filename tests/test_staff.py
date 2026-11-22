"""Stage 0/1 tests: build, staff finding, removal, segmentation."""

from pathlib import Path

import numpy as np
import pytest

from gamera_omr import config
from gamera_omr.segments import find_barlines, find_grand_barlines, measure_bounds
from gamera_omr.stacks import load_stack
from gamera_omr.staffs import Staff, find_staves, remove_staff_lines, validate_staves

CHORALE = config.WORK / "mvt" / "5" / "stacks" / "nba" / "S-01.png"
CHORUS = config.WORK / "stacks" / "nba" / "S-01.png"


def _needs(path):
    return pytest.mark.skipif(not path.exists(), reason=f"missing test image {path}")


@_needs(CHORALE)
def test_staff_finding_chorale():
    page = load_stack(CHORALE)
    staves, diag = find_staves(page)
    assert 1 <= len(staves) <= 6
    assert diag["backend"] in {"projections", "miyao", "oracle"}
    ok, reasons = validate_staves(staves, page.image)
    assert ok, reasons
    for staff in staves:
        assert len(staff.lines) == 5
        assert 10 <= staff.space <= 28
        assert 55 <= staff.height <= 90


@_needs(CHORALE)
def test_removal_removes_lines_keeps_ink():
    page = load_stack(CHORALE)
    staves, _ = find_staves(page)
    before = page.array()
    cleaned = remove_staff_lines(page, staves).image.to_numpy()
    assert cleaned.shape == before.shape
    # Lines are removed: the ink in the five line bands drops sharply.
    staff = staves[0]
    def line_ink(a):
        return sum(int(a[max(0, int(y) - 1):int(y) + 2, :].sum()) for y in staff.lines)
    assert line_ink(cleaned) < 0.3 * line_ink(before)


@_needs(CHORALE)
def test_barlines_and_measures():
    page = load_stack(CHORALE)
    staves, _ = find_staves(page)
    arr = page.array()
    counts = []
    for staff in staves:
        bars = find_barlines(arr, staff)
        bounds = measure_bounds(staff, bars)
        counts.append(len(bounds))
    assert all(n >= 1 for n in counts)
    # The chorale is 14 bars in the truth; homr reads 15 (incl. the incipit).
    assert 12 <= sum(counts) <= 18, counts


@_needs(CHORALE)
def test_staff_finding_without_oracle_frame():
    page = load_stack(CHORALE)
    staves, diag = find_staves(page, allow_oracle=False)
    ok, reasons = validate_staves(staves, page.image)
    assert ok, reasons
    assert diag["oracle_mae"] is None
    assert diag["oracle_frame"] == "disabled_deskewed_frame"


@_needs(CHORUS)
def test_staff_finding_six_systems():
    page = load_stack(CHORUS)
    staves, diag = find_staves(page)
    assert len(staves) == 6
    assert diag["oracle_mae"] is None or diag["oracle_mae"] < 3.0


def _barline_page(barline_end: int):
    """Five-line staff with one barline, a long stem and a beamed stem."""
    lines = [40, 60, 80, 100, 120]
    staff = Staff(0, lines, left=20, right=540, space=20.0, thickness=2.0)
    arr = np.zeros((150, 560), np.uint8)
    for y in lines:
        arr[y:y + 2, 20:540] = 1
    arr[38:barline_end, 300:303] = 1          # real barline
    arr[30:135, 420:422] = 1                  # long beamed stem
    arr[28:35, 420:455] = 1                   # beam above the staff
    arr[126:135, 412:422] = 1                 # notehead below the staff
    return staff, arr


def test_beamed_stem_is_not_a_barline():
    staff, arr = _barline_page(123)
    bars = find_barlines(arr, staff, flush_bottom=True, reject_stems=True)
    assert any(abs(b - 301) <= 1.5 for b in bars)
    assert not any(abs(b - 421) <= 1.5 for b in bars)


def test_grand_barline_bridges_small_scan_gap():
    upper = Staff(0, [40, 60, 80, 100, 120], left=20, right=560,
                  space=20.0, thickness=2.0)
    lower = Staff(1, [200, 220, 240, 260, 280], left=20, right=560,
                  space=20.0, thickness=2.0)
    arr = np.zeros((320, 600), np.uint8)
    for y in upper.lines + lower.lines:
        arr[y:y + 2, 20:560] = 1
    arr[37:150, 300:303] = 1          # grand barline, broken by a 3 px gap
    arr[153:283, 300:303] = 1
    arr[37:283, 450:453] = 1          # continuous grand barline
    bars = find_grand_barlines(arr, upper, lower)
    assert any(abs(b - 301) <= 1.5 for b in bars)
    assert any(abs(b - 451) <= 1.5 for b in bars)


def test_flush_bottom_barline_option():
    # A barline ending one pixel below the bottom line is accepted only when
    # the page is read with the option used for the NBA continuo.
    staff, arr = _barline_page(122)
    assert not any(abs(b - 301) <= 1.5 for b in find_barlines(arr, staff))
    bars = find_barlines(arr, staff, flush_bottom=True)
    assert any(abs(b - 301) <= 1.5 for b in bars)
