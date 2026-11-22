"""Rest/rhythm unit tests."""

from fractions import Fraction

from gamera_omr import decode
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
    notes = [e for e in out if e.kind == "note"]
    assert len(notes) == 2
    assert notes[0].onset == notes[1].onset
    assert notes[0].duration == notes[1].duration
