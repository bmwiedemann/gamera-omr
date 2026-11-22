"""Rest/rhythm unit tests."""

from fractions import Fraction

import numpy as np
from music21 import pitch

from gamera_omr import decode, notes
from gamera_omr.musicxml import _duration_parts


def test_duration_parts_are_standard_values():
    assert _duration_parts(Fraction(5, 4)) == [Fraction(1), Fraction(1, 4)]
    assert _duration_parts(Fraction(3, 2)) == [Fraction(3, 2)]
    assert sum(_duration_parts(Fraction(7, 4))) == Fraction(7, 4)


def test_tacet_measure_becomes_full_rest():
    false_note = decode.Event(0, 10.0, None, Fraction(1), "note")
    events = decode.normalize_measures([false_note], 1, Fraction(3, 2),
                                       tacet={0})
    assert len(events) == 1
    assert events[0].kind == "rest"
    assert events[0].duration == Fraction(3, 2)


def test_empty_measures_become_rests():
    events = decode.normalize_measures([], 3, Fraction(4))
    assert [e.kind for e in events] == ["rest", "rest", "rest"]


def test_onsets_and_gaps_from_x():
    a = decode.Event(0, 10.0, None, Fraction(1), "note")
    b = decode.Event(0, 50.0, None, Fraction(1), "note")
    out = decode.assign_onsets_from_x([a, b], [(0.0, 100.0)], Fraction(4))
    notes = [e for e in out if e.kind == "note"]
    rests = [e for e in out if e.kind == "rest"]
    # The first and last noteheads anchor the measure: no leading rest, and
    # the last note ends with the bar.
    assert [e.onset for e in notes] == [Fraction(0), Fraction(3)]
    assert notes[0].duration == Fraction(1)
    assert sum(e.duration for e in out) == Fraction(4)
    assert any(r.onset == Fraction(1) and r.duration == Fraction(2)
               for r in rests)


def test_chord_notes_share_onset_and_duration():
    a = decode.Event(0, 10.0, None, Fraction(1), "note")
    b = decode.Event(0, 11.0, None, Fraction(1), "note")
    out = decode.assign_onsets_from_x(
        [a, b], [(0.0, 100.0)], Fraction(4), chord_tol=2.0)
    notes_ = [e for e in out if e.kind == "note"]
    assert len(notes_) == 2
    assert notes_[0].onset == notes_[1].onset
    assert notes_[0].duration == notes_[1].duration


def test_measure_fill_counts_voices_once():
    upper = decode.Event(0, 10.0, None, Fraction(3, 4), "note",
                         onset=Fraction(0), voice=0)
    lower = decode.Event(0, 10.0, None, Fraction(3, 4), "note",
                         onset=Fraction(0), voice=1)
    chord_note = decode.Event(0, 11.0, None, Fraction(1, 2), "note",
                              onset=Fraction(1, 2), voice=1)
    assert decode.measure_fill([upper, lower, chord_note], 1,
                               Fraction(5, 4)) == [Fraction(5, 4)]


def test_stem_direction_follows_stem_side():
    space = 10
    up = np.zeros((40, 20), np.uint8)
    up[26:34, 4:14] = 1          # filled head near the bottom
    up[6:34, 13] = 1             # stem rises from it
    assert notes.stem_direction(up, space) == "up"

    down = np.zeros_like(up)
    down[6:14, 4:14] = 1         # head near the top
    down[6:34, 13] = 1           # stem falls from it
    assert notes.stem_direction(down, space) == "down"


def _voice_event(x, direction, measure=0, hy=0.0):
    event = decode.Event(measure, x, pitch.Pitch("C5"), Fraction(1), "note")
    event.direction = direction
    event.hy = hy
    return event


def test_simultaneous_opposite_stems_split_into_voices():
    events = [_voice_event(10, "up"), _voice_event(12, "down"),
              _voice_event(30, "up"), _voice_event(32, "down")]
    groups = decode.split_staff_voices(events, space=10)
    assert groups is not None
    assert [len(g) for g in groups] == [2, 2]
    assert all(e.direction == "up" and e.voice == 0 for e in groups[0])
    assert all(e.direction == "down" and e.voice == 1 for e in groups[1])


def test_alternating_stem_directions_stay_one_voice():
    # A single line flips stem direction with the note's staff position.
    events = [_voice_event(10, "up"), _voice_event(40, "down"),
              _voice_event(70, "up"), _voice_event(100, "down")]
    assert decode.split_staff_voices(events, space=10) is None
    assert {e.voice for e in events} == {0}
