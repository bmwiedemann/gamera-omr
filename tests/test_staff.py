"""Stage 0/1 tests: build, staff finding, removal, segmentation."""

from pathlib import Path

import numpy as np
import pytest

from gamera_omr import config
from gamera_omr.segments import find_barlines, measure_bounds
from gamera_omr.stacks import load_stack
from gamera_omr.staffs import find_staves, remove_staff_lines, validate_staves

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
