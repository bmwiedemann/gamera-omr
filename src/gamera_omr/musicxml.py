"""MusicXML output via music21."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

from music21 import clef as mclef
from music21 import key as mkey
from music21 import meter, note, stream

from .decode import Event


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
        n = note.Rest() if e.kind == "rest" else note.Note()
        if e.kind != "rest":
            n.pitch = e.pitch
        n.duration.quarterLength = e.duration
        measures[e.measure].insert(e.onset, n)
    for m in measures:
        part.append(m)
    score.insert(0, part)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    score.write("musicxml", fp=str(out), makeNotation=False)
    return out
