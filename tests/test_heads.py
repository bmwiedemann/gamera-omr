"""Unit tests for grand-staff straightening, stem-first heads and accidentals."""

from fractions import Fraction
from pathlib import Path

import numpy as np
from music21 import pitch

from gamera_omr import decode, deskew, notes
from gamera_omr.stacks import StackPage, from_array
from gamera_omr.staffs import Staff, remove_staff_lines

SPACE = 16


def _staff(index, top, left=40.0, right=1160.0, space=SPACE):
    return Staff(index, [float(top + k * space) for k in range(5)], left, right,
                 float(space), 2.0)


def test_straighten_staves_removes_linear_drift():
    width, height = 1200, 260
    arr = np.zeros((height, width), np.uint16)
    tops = (40, 160)
    slope = 0.012          # about 13 px over the system: most of a space
    for top in tops:
        for k in range(5):
            for x in range(40, width - 40):
                y = int(round(top + k * SPACE + slope * (x - width / 2)))
                arr[y:y + 2, x] = 1
    page = StackPage(Path("/nonexistent/grand.png"), from_array(arr), [])
    staves = [_staff(i, top) for i, top in enumerate(tops)]
    straight = deskew.straighten_staves(page, staves)
    assert straight is not None
    out = straight.array()
    for top in tops:
        for x in (80, width // 2, width - 80):
            rows = np.flatnonzero(out[top - 6:top + 6, x]) + top - 6
            assert len(rows) and abs(rows.mean() - (top + 0.5)) <= 1.5


def test_straighten_staves_leaves_level_page():
    arr = np.zeros((200, 1200), np.uint16)
    for k in range(5):
        arr[40 + k * SPACE:42 + k * SPACE, 40:1160] = 1
    page = StackPage(Path("/nonexistent/level.png"), from_array(arr), [])
    assert deskew.straighten_staves(page, [_staff(0, 40)]) is None


def _eighth_stem_down(space=SPACE):
    """Head at the top, stem down on its left, a flag at the stem's foot."""
    ink = np.zeros((int(4.5 * space), int(2.2 * space)), np.uint8)
    head_h, head_w = int(0.9 * space), int(1.3 * space)
    ink[0:head_h, 2:2 + head_w] = 1
    stem_x = 3
    ink[head_h // 2:int(4.2 * space), stem_x:stem_x + 2] = 1
    for i in range(int(1.6 * space)):          # a thick flag
        y = int(4.2 * space) - i
        x = stem_x + 2 + i // 2
        ink[y - 2:y + 2, x:x + 3] = 1
    return ink


def test_stemmed_heads_keeps_flag_out_of_the_head():
    ink = _eighth_stem_down()
    heads, direction = notes.stemmed_heads(ink, SPACE)
    assert direction == "down"
    assert len(heads) == 1
    x, y, duration = heads[0]
    assert y < 1.0 * SPACE                      # the head, not the flag
    assert duration == Fraction(1, 2)


def test_stemmed_heads_rejects_a_sharp():
    ink = np.zeros((int(3.0 * SPACE), int(1.1 * SPACE)), np.uint8)
    ink[:, 3:5] = 1
    ink[:, 12:14] = 1
    ink[14:20, :] = 1
    ink[28:34, :] = 1
    assert notes.stemmed_heads(ink, SPACE) == ([], "accidental")
    assert notes.accidental_kind(ink, SPACE) == "sharp"


def test_accidental_kind_tells_natural_from_sharp():
    ink = np.zeros((int(3.0 * SPACE), int(0.8 * SPACE)), np.uint8)
    ink[0:36, 1:3] = 1
    ink[12:48, 10:12] = 1
    ink[14:18, :] = 1
    ink[30:34, :] = 1
    assert notes.accidental_kind(ink, SPACE) == "natural"


def test_lone_head_accepts_head_and_rejects_eighth_rest():
    head = np.zeros((SPACE, int(1.3 * SPACE)), np.uint8)
    cv_y, cv_x = np.ogrid[:SPACE, :int(1.3 * SPACE)]
    mask = (((cv_x - 10) / 10.0) ** 2 + ((cv_y - 8) / 7.5) ** 2) <= 1.0
    head[mask] = 1
    assert notes.lone_head(head, SPACE) is not None
    rest = np.zeros((int(1.8 * SPACE), int(1.0 * SPACE)), np.uint8)
    rest[0:7, 0:7] = 1                          # the blob
    for i in range(int(1.8 * SPACE) - 4):        # the long tail
        rest[i + 4, max(0, 12 - i // 3):max(0, 12 - i // 3) + 2] = 1
    assert notes.lone_head(rest, SPACE) is None


def test_keep_crossings_leaves_head_on_a_line_whole():
    arr = np.zeros((140, 300), np.uint16)
    top = 40
    for k in range(5):
        arr[top + k * SPACE:top + k * SPACE + 2, 20:280] = 1
    line = top + 2 * SPACE
    arr[line - 7:line + 8, 100:120] = 1        # head centred on the middle line
    page = StackPage(Path("/nonexistent/head.png"), from_array(arr), [])
    staves = [_staff(0, top, 20.0, 280.0)]
    cut = remove_staff_lines(page, staves).image.to_numpy()
    kept = remove_staff_lines(page, staves,
                              keep_crossings=True).image.to_numpy()
    assert not cut[line, 110] and not cut[line + 1, 110]
    assert kept[line - 7:line + 8, 100:120].all()
    assert not kept[line, 50]                  # the line itself is gone


def test_has_ledger_requires_every_ledger():
    staff = _staff(0, 60)
    arr = np.zeros((200, 300), np.uint16)
    below = staff.bottom + SPACE
    arr[int(below):int(below) + 2, 140:166] = 1     # one ledger below
    assert decode.has_ledger(arr, staff, 153, below)            # on it
    assert not decode.has_ledger(arr, staff, 153, below + SPACE)  # needs two
    assert not decode.has_ledger(arr, staff, 153, staff.top - 2 * SPACE)
    assert decode.has_ledger(arr, staff, 153, staff.top + SPACE)


def _note(x, name, duration, measure=0):
    return decode.Event(measure, float(x), pitch.Pitch(name), duration, "note")


def test_sequence_onsets_follow_durations_when_they_fill_the_bar():
    # Uneven spacing would mislead the x interpolation; the eighths add up
    # to the 2/4 bar, so their durations place them.
    events = [_note(10, "D3", Fraction(1, 2)), _note(30, "D2", Fraction(1, 2)),
              _note(80, "D2", Fraction(1, 2)), _note(90, "D3", Fraction(1, 2))]
    out = decode.assign_onsets_from_x(events, [(0.0, 120.0)], Fraction(2),
                                      grid=Fraction(1, 8), chord_tol=4.0,
                                      sequence=True)
    onsets = [e.onset for e in out if e.kind == "note"]
    assert onsets == [0, Fraction(1, 2), 1, Fraction(3, 2)]


def test_voices_split_only_where_stems_overlap():
    def ev(x, direction, measure):
        e = _note(x, "C3", Fraction(1, 2), measure)
        e.direction = direction
        return e

    events = [ev(10, "up", 0), ev(40, "down", 0),       # one line, bar 1
              ev(110, "up", 1), ev(111, "down", 1)]     # two voices, bar 2
    groups = decode.split_staff_voices(events, space=10)
    assert groups is not None
    assert {e.voice for e in events if e.measure == 0} == {0}
    assert {e.voice for e in events if e.measure == 1} == {0, 1}


def test_accidental_applies_to_following_notes_in_the_measure():
    from gamera_omr.glyphs import Glyph

    staff = _staff(0, 60)
    sharp = np.zeros((int(3.0 * SPACE), int(1.1 * SPACE)), np.uint8)
    sharp[:, 3:5] = 1
    sharp[:, 12:14] = 1
    sharp[14:20, :] = 1
    sharp[28:34, :] = 1
    head_y = staff.bottom                       # G2 on the bass staff's bottom line
    y0 = int(head_y - sharp.shape[0] / 2)
    glyph = Glyph("acc", "test", 0, 200, y0, 200 + sharp.shape[1] - 1,
                  y0 + sharp.shape[0] - 1, SPACE, sharp)
    first = _note(200 + sharp.shape[1] + 0.6 * SPACE + 4, "G2", Fraction(1, 2))
    later = _note(300, "G2", Fraction(1, 2))
    other = _note(300, "G2", Fraction(1, 2), measure=1)
    for e in (first, later, other):
        e.hy = head_y
    decode.apply_accidentals([first, later, other], staff, [glyph], 2)
    assert first.pitch.name == "G#" and later.pitch.name == "G#"
    assert other.pitch.name == "G"
