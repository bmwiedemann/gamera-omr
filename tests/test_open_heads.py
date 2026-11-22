"""Unit tests for open heads, eighth rests, head centres and signatures."""

from fractions import Fraction

import cv2
import numpy as np

from gamera_omr import decode, notes
from gamera_omr.glyphs import Glyph
from gamera_omr.staffs import Staff, validate_staves

SPACE = 18


def _staff(top=60, space=SPACE, left=20.0, right=780.0):
    return Staff(0, [float(top + k * space) for k in range(5)], left, right,
                 float(space), 2.0)


def _page(staff, width=800, height=240):
    arr = np.zeros((height, width), np.uint8)
    for y in staff.lines:
        arr[int(y) - 1:int(y) + 1, int(staff.left):int(staff.right)] = 1
    return arr


def _ring(arr, cx, cy, space=SPACE, stem=None):
    """A whole/half notehead: a thick oval ring, optionally with a stem."""
    outer = np.zeros_like(arr)
    cv2.ellipse(outer, (cx, cy), (int(0.85 * space), int(0.5 * space)), 0,
                0, 360, 1, -1)
    inner = np.zeros_like(arr)
    cv2.ellipse(inner, (cx, cy), (int(0.45 * space), int(0.3 * space)), 0,
                0, 360, 1, -1)
    arr[(outer > 0) & (inner == 0)] = 1
    if stem == "up":
        x = cx + int(0.75 * space)
        arr[cy - int(3.3 * space):cy + 1, x:x + 2] = 1


def test_open_heads_whole_and_half_notes():
    staff = _staff()
    arr = _page(staff)
    _ring(arr, 200, int(staff.lines[2]))                  # whole on a line
    _ring(arr, 400, int(staff.lines[3] - SPACE / 2), stem="up")  # half
    found = notes.open_heads(arr > 0, staff, [300.0])
    by_x = {round(x / 50): (x, y, d, direction) for x, y, d, direction, _s in found}
    assert set(by_x) == {4, 8}
    x, y, duration, direction = by_x[4]
    assert abs(y - staff.lines[2]) <= 2 and duration == Fraction(4)
    assert direction is None
    x, y, duration, direction = by_x[8]
    assert duration == Fraction(2) and direction == "up"


def test_open_heads_ignore_filled_heads_and_rectangles():
    staff = _staff()
    arr = _page(staff)
    cv2.ellipse(arr, (200, int(staff.lines[2])), (15, 9), 0, 0, 360, 1, -1)
    # two stems closing the space between two lines: a rectangle, no head
    arr[int(staff.lines[1]):int(staff.lines[2]) + 1, 400:402] = 1
    arr[int(staff.lines[1]):int(staff.lines[2]) + 1, 416:418] = 1
    assert notes.open_heads(arr > 0, staff, []) == []


def _eighth_rest(space=SPACE):
    ink = np.zeros((int(1.9 * space), int(1.1 * space)), np.uint8)
    cv2.circle(ink, (int(0.3 * space), int(0.3 * space)), int(0.27 * space), 1, -1)
    ink[int(0.25 * space):int(0.4 * space), int(0.3 * space):] = 1
    for i in range(int(1.5 * space)):
        x = ink.shape[1] - 2 - int(i * 0.35)
        ink[int(0.4 * space) + i, x - 1:x + 1] = 1
    return ink


def test_eighth_rest_shape():
    assert notes.eighth_rest(_eighth_rest(), SPACE)
    head = np.zeros((int(2.0 * SPACE), int(1.3 * SPACE)), np.uint8)
    cv2.ellipse(head, (int(0.65 * SPACE), int(0.5 * SPACE)), (11, 8), 0, 0,
                360, 1, -1)
    head[int(0.5 * SPACE):, 1:3] = 1                  # stem down on the left
    assert not notes.eighth_rest(head, SPACE)


def test_flat_with_bowl_cut_open_by_line_removal():
    ink = np.zeros((int(2.4 * SPACE), int(0.9 * SPACE)), np.uint8)
    ink[:, 0:3] = 1
    bowl = np.zeros_like(ink)
    cv2.ellipse(bowl, (6, int(1.8 * SPACE)), (8, 6), -30, 0, 360, 1, 3)
    ink |= bowl
    ink[int(1.8 * SPACE) - 7:int(1.8 * SPACE) - 4, 4:] = 0   # cut its top
    assert notes.accidental_kind(ink, SPACE) == "flat"


def test_refine_head_y_trims_line_under_head_above_staff():
    staff = _staff()
    top = int(staff.top)
    # a head in the space above the top line, carrying the line's piece
    ink = np.zeros((40, 24), np.uint8)
    cv2.ellipse(ink, (12, 11), (11, 8), 0, 0, 360, 1, -1)
    ink[19:22, 2:22] = 1
    g = Glyph(uid="h", page="p", staff=0, x0=100, y0=top - 20,
              x1=100 + ink.shape[1] - 1, y1=top - 20 + ink.shape[0] - 1,
              space=float(SPACE), ink=ink)
    y = decode.refine_head_y(g, staff, 112, top - 6)
    assert abs(y - (top - SPACE / 2)) <= 2.5


def test_close_recitative_grand_staff_is_valid():
    image = type("I", (), {"ncols": 800})()
    upper = Staff(0, [40.0 + k * 18 for k in range(5)], 20, 780, 18.0, 3.0)
    lower = Staff(1, [40.0 + 122 + k * 18 for k in range(5)], 20, 780, 18.0,
                  3.0)
    ok, reasons = validate_staves([upper, lower], image)
    assert ok, reasons
