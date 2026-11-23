"""Unit tests for barline/stem separation, local line offsets and accidentals."""

import cv2
import numpy as np

from gamera_omr import decode, notes, segments
from gamera_omr.glyphs import Glyph
from gamera_omr.staffs import Staff

SPACE = 16


def _staff(top=60, index=0, left=20.0, right=780.0):
    return Staff(index, [float(top + k * SPACE) for k in range(5)], left, right,
                 float(SPACE), 2.0)


def _page(staves, width=800, height=200):
    arr = np.zeros((height, width), np.uint8)
    for staff in staves:
        for y in staff.lines:
            arr[int(y) - 1:int(y) + 1, int(staff.left):int(staff.right)] = 1
    return arr


def test_stem_with_beam_on_bottom_line_is_no_barline():
    staff = _staff()
    arr = _page([staff])
    top, bottom = int(staff.top), int(staff.bottom)
    arr[top - 2:bottom + 2, 400:402] = 1                 # a real barline
    # a stem from a head above the staff down to a beam on the bottom line
    # the head sits on the top line, so the line mask hides most of it
    cv2.ellipse(arr, (594, top), (8, 5), 0, 0, 360, 1, -1)
    arr[top:bottom + 3, 600:602] = 1
    arr[bottom - 3:bottom + 6, 600:680] = 1               # beam on the line
    arr[top + 10:bottom + 3, 678:680] = 1
    bars = segments.find_barlines(arr, staff, flush_bottom=True,
                                  reject_stems=True)
    assert any(abs(b - 400.5) <= 1 for b in bars)
    assert not any(590 <= b <= 610 for b in bars)


def test_grand_barline_broken_for_a_dynamic_mark():
    upper, lower = _staff(30, 0), _staff(150, 1)
    arr = _page([upper, lower], height=260)
    x = 500
    arr[int(upper.top) - 2:int(upper.bottom) + 30, x:x + 2] = 1
    arr[int(lower.top) - 12:int(lower.bottom) + 2, x:x + 2] = 1
    bars = segments.find_grand_barlines(arr, upper, lower)
    assert any(abs(b - x - 0.5) <= 1 for b in bars)


def test_line_offset_measures_local_tilt():
    staff = _staff()
    arr = np.zeros((200, 800), np.uint8)
    for y in staff.lines:
        arr[int(y) + 2:int(y) + 4, 20:780] = 1           # lines 3 px lower
    off = decode.line_offset(arr > 0, staff, 400)
    assert 2.0 <= off <= 3.5
    assert decode.line_offset(_page([staff]) > 0, staff, 400) == 0.0


def _glyph(uid, x0, y0, ink):
    return Glyph(uid=uid, page="p", staff=0, x0=x0, y0=y0,
                 x1=x0 + ink.shape[1] - 1, y1=y0 + ink.shape[0] - 1,
                 space=float(SPACE), ink=ink)


def test_flat_cut_into_stroke_and_bowl_is_found():
    staff = _staff()
    hx, hy = 300, int(staff.lines[2] - SPACE / 2)
    stroke = np.ones((int(2.4 * SPACE), 3), np.uint8)
    bowl = np.zeros((int(0.9 * SPACE), 10), np.uint8)
    cv2.ellipse(bowl, (3, 7), (6, 5), 0, -90, 90, 1, 2)
    flat_x = hx - int(1.7 * SPACE)
    glyphs = [_glyph("s", flat_x, hy + 6 - stroke.shape[0], stroke),
              _glyph("b", flat_x + 4, hy - 7, bowl)]
    assert decode.nearby_accidental(glyphs, staff, hx, hy, notes) == "flat"


def test_natural_and_sharp_strokes():
    def strokes(left, right, w=int(1.1 * SPACE), h=int(3.0 * SPACE)):
        ink = np.zeros((h, w), np.uint8)
        ink[left[0]:left[1], 2:4] = 1
        ink[right[0]:right[1], w - 4:w - 2] = 1
        ink[h // 3:h // 3 + 3, 2:w - 2] = 1
        ink[2 * h // 3:2 * h // 3 + 3, 2:w - 2] = 1
        return ink

    h = int(3.0 * SPACE)
    assert notes.accidental_kind(strokes((0, h - 10), (10, h)), SPACE) == "natural"
    assert notes.accidental_kind(strokes((2, h - 2), (0, h - 4)), SPACE) == "sharp"


def test_stems_meeting_between_the_staves_are_no_barline():
    upper, lower = _staff(30, 0), _staff(150, 1)
    arr = _page([upper, lower], height=260)
    x = 500
    # a down-stem from a head in the upper staff, an up-stem from a head in
    # the lower staff, both reaching into the gap at the same x
    cv2.ellipse(arr, (x + 10, int(upper.top) - 3), (10, 7), -20, 0, 360, 1, -1)
    arr[int(upper.top) - 3:int(upper.bottom) + 30, x:x + 2] = 1
    cv2.ellipse(arr, (x - 9, int(lower.bottom) + 3), (10, 7), -20, 0, 360, 1, -1)
    arr[int(lower.top) - 12:int(lower.bottom) + 3, x:x + 2] = 1
    bars = segments.find_grand_barlines(arr, upper, lower)
    assert not any(abs(b - x - 0.5) <= 2 for b in bars)
