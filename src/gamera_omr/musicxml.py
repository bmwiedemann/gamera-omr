"""MusicXML output via music21."""

from __future__ import annotations

from collections import defaultdict
from fractions import Fraction
from pathlib import Path

from music21 import chord as mchord
from music21 import clef as mclef
from music21 import duration as mduration
from music21 import key as mkey
from music21 import layout, meter, note, stream, tie

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


def _fill_part(name: str | None, events: list[Event], nmeasures: int, clef: str,
               fifths: int, time_signature: str, *,
               part_factory=stream.Part, staff: int | None = None,
               voice_base: int = 1,
               clef_changes: dict[int, str] | None = None) -> stream.Part:
    """Build one part (or one staff of a grand staff) from decoded events.

    Events carry a `voice` index.  A single-voice part is written directly
    into the measures; a part with two or more voices uses <voice> streams so
    the voices can overlap (MusicXML writes the needed backups).  On a grand
    staff (`staff` set), every note is tagged with its staff number, and
    voice numbers are offset per staff (1-4 upper, 5-8 lower) so the voices
    are unique within the part.
    """
    part = part_factory()
    if name:
        part.partName = name
    measures = [stream.Measure(number=i + 1) for i in range(nmeasures)]
    measures[0].insert(0, mclef.clefFromString(clef))
    measures[0].insert(0, mkey.KeySignature(fifths or 0))
    measures[0].insert(0, meter.TimeSignature(time_signature))
    for measure, changed in (clef_changes or {}).items():
        if 0 <= measure < nmeasures:
            measures[measure].insert(0, mclef.clefFromString(changed))

    # Notes sharing measure/voice/onset/duration are a chord (piano, divisi).
    grouped: dict[tuple[int, int, Fraction, Fraction], list[Event]] = defaultdict(list)
    for e in events:
        grouped[(e.measure, e.voice, e.onset, e.duration)].append(e)

    voices_in_measure: dict[int, set[int]] = defaultdict(set)
    for measure, voice, _onset, _duration in grouped:
        voices_in_measure[measure].add(voice)
    voice_streams: dict[tuple[int, int], stream.Voice] = {}

    def container(measure: int, voice: int):
        # A grand staff always gets explicit voice streams so the two staff
        # voices receive stable numbers (1..4 upper, 5..8 lower) instead of
        # whichever number music21 happens to assign next.
        if len(voices_in_measure[measure]) <= 1 and staff is None:
            return measures[measure]
        key = (measure, voice)
        if key not in voice_streams:
            stream_voice = stream.Voice(id=str(voice_base + voice))
            measures[measure].insert(0, stream_voice)
            voice_streams[key] = stream_voice
        return voice_streams[key]

    for (measure, voice, onset0, duration), group in sorted(
            grouped.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2])):
        target = container(measure, voice)
        notes = [e for e in group if e.kind == "note"]
        rests = [e for e in group if e.kind == "rest"]
        if not notes:
            if rests:
                offset = onset0
                for value in _duration_parts(duration):
                    n = note.Rest()
                    n.duration = mduration.Duration(quarterLength=value)
                    if staff is not None:
                        n.staff = staff
                    target.insert(offset, n)
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
            if staff is not None:
                piece.staff = staff
            target.insert(offset, piece)
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


def write_piano(rh_events: list[Event], lh_events: list[Event], nmeasures: int,
                fifths: int, time_signature: str, out_path: str | Path, *,
                rh_clef: str = "G2", lh_clef: str = "F4",
                rh_clef_changes: dict[int, str] | None = None,
                lh_clef_changes: dict[int, str] | None = None):
    """Write a grand staff as one piano part with two staves and voices.

    The hands are separate PartStaff objects inside one part, as piano
    MusicXML expects: one <part> with <staves>2</staves>, each note tagged
    with its staff, and voices numbered 1-4 (right hand) and 5-8 (left hand).
    """
    score = stream.Score()
    upper = _fill_part("Piano", rh_events, nmeasures, rh_clef, fifths,
                       time_signature, part_factory=stream.PartStaff,
                       staff=1, voice_base=1, clef_changes=rh_clef_changes)
    lower = _fill_part(None, lh_events, nmeasures, lh_clef, fifths,
                       time_signature, part_factory=stream.PartStaff,
                       staff=2, voice_base=5, clef_changes=lh_clef_changes)
    score.insert(0, upper)
    score.insert(0, lower)
    score.insert(0, layout.StaffGroup([upper, lower], symbol="brace",
                                      name="Piano"))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    score.write("musicxml", fp=str(out), makeNotation=False)
    return out
