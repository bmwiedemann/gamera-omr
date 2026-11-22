"""MusicXML output via music21."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

from fractions import Fraction

from music21 import clef as mclef
from music21 import duration as mduration
from music21 import key as mkey
from music21 import meter, note, stream, tie

from .decode import Event

# Standard note values (quarter lengths), longest first.
_STANDARD = [
    Fraction(4), Fraction(3), Fraction(2), Fraction(3, 2), Fraction(1),
    Fraction(3, 4), Fraction(1, 2), Fraction(3, 8), Fraction(1, 4),
    Fraction(3, 16), Fraction(1, 8), Fraction(1, 16),
]


def _duration_parts(quarter_length):
    """Greedy split into note values MusicXML can write (e.g. 5/4 = 1 + 1/4)."""
    remaining = Fraction(quarter_length)
    parts = []
    for value in _STANDARD:
        while remaining >= value:
            parts.append(value)
            remaining -= value
    if remaining > 0:
        parts.append(remaining)
    return parts


def write(events: list[Event], nmeasures: int, clef: str, fifths: int,
          time_signature: str, out_path: str | Path, part_name: str = "Voice"):
    score = stream.Score()
    part = stream.Part(id="P1")
    part.partName = part_name
    measures = [stream.Measure(number=i + 1) for i in range(nmeasures)]
    measures[0].insert(0, mclef.clefFromString(clef))
    measures[0].insert(0, mkey.KeySignature(fifths or 0))
    measures[0].insert(0, meter.TimeSignature(time_signature))
    for e in events:
        offset = e.onset
        parts = _duration_parts(e.duration)
        for k, value in enumerate(parts):
            n = note.Rest() if e.kind == "rest" else note.Note()
            if e.kind != "rest":
                n.pitch = e.pitch
                if len(parts) > 1:
                    if k == 0:
                        n.tie = tie.Tie("start")
                    elif k == len(parts) - 1:
                        n.tie = tie.Tie("stop")
                    else:
                        n.tie = tie.Tie("continue")
            n.duration = mduration.Duration(quarterLength=value)
            measures[e.measure].insert(offset, n)
            offset += value
    for m in measures:
        part.append(m)
    score.insert(0, part)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    score.write("musicxml", fp=str(out), makeNotation=False)
    return out
