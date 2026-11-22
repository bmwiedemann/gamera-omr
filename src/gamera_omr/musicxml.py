"""MusicXML output via music21."""

from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
from pathlib import Path

from music21 import chord as mchord
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


def _fill_part(name: str, events: list[Event], nmeasures: int, clef: str,
               fifths: int, time_signature: str) -> stream.Part:
    part = stream.Part()
    part.partName = name
    measures = [stream.Measure(number=i + 1) for i in range(nmeasures)]
    measures[0].insert(0, mclef.clefFromString(clef))
    measures[0].insert(0, mkey.KeySignature(fifths or 0))
    measures[0].insert(0, meter.TimeSignature(time_signature))

    # Notes sharing measure/onset/duration are a chord (piano, divisi).
    grouped: dict[tuple[int, Fraction, Fraction], list[Event]] = defaultdict(list)
    for e in events:
        grouped[(e.measure, e.onset, e.duration)].append(e)

    for (measure, onset0, duration), group in grouped.items():
        notes = [e for e in group if e.kind == "note"]
        rests = [e for e in group if e.kind == "rest"]
        if not notes:
            if rests:
                offset = onset0
                for value in _duration_parts(duration):
                    n = note.Rest()
                    n.duration = mduration.Duration(quarterLength=value)
                    measures[measure].insert(offset, n)
                    offset += value
            continue
        pitches = [e.pitch for e in notes]
        n = mchord.Chord(pitches) if len(pitches) > 1 else note.Note(pitches[0])
        offset = onset0
        parts = _duration_parts(duration)
        for k, value in enumerate(parts):
            piece = n if len(parts) == 1 else (
                mchord.Chord(pitches) if len(pitches) > 1 else note.Note(pitches[0]))
            if len(parts) > 1:
                if k == 0:
                    piece.tie = tie.Tie("start")
                elif k == len(parts) - 1:
                    piece.tie = tie.Tie("stop")
                else:
                    piece.tie = tie.Tie("continue")
            piece.duration = mduration.Duration(quarterLength=value)
            measures[measure].insert(offset, piece)
            offset += value
    for m in measures:
        part.append(m)
    return part


def write(events: list[Event], nmeasures: int, clef: str, fifths: int,
          time_signature: str, out_path: str | Path, part_name: str = "Voice"):
    score = stream.Score()
    score.insert(0, _fill_part(part_name, events, nmeasures, clef, fifths,
                               time_signature))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    score.write("musicxml", fp=str(out), makeNotation=False)
    return out


def write_parts(groups: list[tuple[list[Event], str, str]], nmeasures: int,
                fifths: int, time_signature: str, out_path: str | Path):
    """Write several parts (e.g. piano right/left hand) into one score."""
    score = stream.Score()
    for events, clef, name in groups:
        score.insert(0, _fill_part(name, events, nmeasures, clef, fifths,
                                   time_signature))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    score.write("musicxml", fp=str(out), makeNotation=False)
    return out
