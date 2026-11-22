"""MusicXML writer tests."""

from fractions import Fraction

from music21 import pitch

from gamera_omr import decode, musicxml


def test_piano_writer_emits_two_staves_and_voices(tmp_path):
    rh = decode.Event(0, 10.0, pitch.Pitch("C5"), Fraction(1), "note",
                      onset=Fraction(0), voice=0)
    rh_lower = decode.Event(0, 10.0, pitch.Pitch("E5"), Fraction(1), "note",
                            onset=Fraction(0), voice=1)
    lh = decode.Event(0, 10.0, pitch.Pitch("C3"), Fraction(1), "note",
                      onset=Fraction(0), voice=0)
    out = tmp_path / "piano.musicxml"
    musicxml.write_piano([rh, rh_lower], [lh], 2, 2, "4/4", out,
                         lh_clef_changes={1: "G2"})
    text = out.read_text()
    assert text.count("<part id=") == 1
    assert "<staves>2</staves>" in text
    assert {"<voice>1</voice>", "<voice>2</voice>", "<voice>5</voice>"} <= set(
        line.strip() for line in text.splitlines())
    assert "<staff>1</staff>" in text and "<staff>2</staff>" in text
    assert "<backup>" in text
    # The lower staff has its initial F4 clef and the change to G2.
    assert text.count('<clef number="2">') == 2
