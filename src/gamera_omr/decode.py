"""Notation decoding: notehead position -> pitch, glyph class -> duration."""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

import numpy as np
from music21 import key as mkey
from music21 import pitch as mpitch

from .glyphs import Glyph
from .staffs import Staff

# Bottom-line diatonic number for the clefs used by the project (music21
# convention: C4 = 29, D4 = 30, E4 = 31, ...).
CLEF_BOTTOM = {"G2": 31, "F4": 19, "C1": 29, "C3": 25, "C4": 23}
REST_DURATIONS = {
    "rest.whole": Fraction(4),
    "rest.half": Fraction(2),
    "rest.quarter": Fraction(1),
    "rest.8": Fraction(1, 2),
    "rest.16": Fraction(1, 4),
    "rest.32": Fraction(1, 8),
}


@dataclass
class Event:
    measure: int
    x: float
    pitch: object
    duration: Fraction
    kind: str = "note"       # note | rest
    glyph: Glyph | None = None
    onset: Fraction = Fraction(0)
    flexible: bool = False   # beamed note; duration may be fitted to the meter
    hy: float = 0.0          # absolute y of the notehead

    @property
    def midi(self) -> int:
        return int(self.pitch.midi) if self.pitch is not None else -1

    @property
    def name(self) -> str:
        return self.pitch.nameWithOctave if self.pitch is not None else "R"


def head_region(ink: np.ndarray, space: float) -> tuple[int, int]:
    """Rows of the notehead inside a composite glyph."""
    widths = ink.sum(axis=1).astype(float)
    threshold = max(2.0, 0.35 * space)
    rows = np.flatnonzero(widths >= threshold)
    if len(rows) == 0:
        return 0, max(0, ink.shape[0] - 1)
    runs = []
    start = prev = int(rows[0])
    for row in rows[1:]:
        row = int(row)
        if row - prev > 0.8 * space:
            runs.append((start, prev))
            start = row
        prev = row
    runs.append((start, prev))
    return max(runs, key=lambda ab: widths[ab[0]:ab[1] + 1].sum())


def head_center(ink: np.ndarray, space: float) -> float:
    """Y centre of the notehead inside a composite glyph."""
    a, b = head_region(ink, space)
    return (a + b) / 2


def head_fill(ink: np.ndarray, space: float) -> tuple[float, float, int, int]:
    """(fill ratio, centre y, height, width) of the notehead region."""
    a, b = head_region(ink, space)
    region = ink[a:b + 1]
    cols = np.flatnonzero(region.any(axis=0))
    width = int(cols[-1] - cols[0] + 1) if len(cols) else 0
    region = region[:, cols[0]:cols[-1] + 1] if len(cols) else region
    fill = float(region.sum()) / max(1, region.size)
    return fill, (a + b) / 2, b - a + 1, width


def staff_position(staff: Staff, head_y: float) -> int:
    # Half-up rounding: Python's round() is banker's rounding, which turns a
    # position of 2.5 into 2 and reads the note a second too low.
    return int(np.floor(2.0 * (staff.bottom - head_y) / staff.space + 0.5))


def pitch_at(staff: Staff, clef: str, position: int, fifths: int = 0):
    diatonic = CLEF_BOTTOM[clef] + position
    p = mpitch.Pitch()
    p.diatonicNoteNum = diatonic
    accidental = mkey.KeySignature(fifths or 0).accidentalByStep(p.step)
    if accidental is not None:
        p.accidental = accidental
    return p


def duration_for(label: str, ink: np.ndarray, space: float,
                 legacy: bool = False) -> Fraction | None:
    if label not in ("note.black", "note.open"):
        return None
    fill, _cy, _h, _w = head_fill(ink, space)
    # Modern engravings separate filled and open heads cleanly by ink ratio.
    # In 19th/early-20th-century prints the open head's counter is often
    # inked, so the glyph class is the more reliable signal there.
    if legacy:
        open_head = label == "note.open" or fill < 0.45
    else:
        open_head = fill < 0.55
    has_stem = ink.shape[0] > 2.2 * space
    if open_head:
        return Fraction(4) if not has_stem else Fraction(2)
    return Fraction(1)


def decode_staff(staff: Staff, glyphs: list[Glyph], predictions: dict[str, dict],
                 bounds: list[tuple[float, float]], clef: str,
                 fifths: int = 0, templates=None,
                 dots: list[tuple[float, float]] | None = None,
                 legacy: bool = False,
                 bar_length: Fraction = Fraction(4),
                 require_head: bool = False) -> list[Event]:
    from . import notes as notes_module  # local import avoids a cycle

    if templates is None:
        templates = notes_module.make_templates(glyphs, predictions)
    events = []
    for g in glyphs:
        if g.staff != staff.index:
            continue
        band_top, band_bottom = staff.band(above=4.0, below=1.2)
        if not (band_top <= g.cy <= band_bottom):
            continue
        # A clef (and the initial key/time signature) is a tall, narrow glyph
        # at the very start of the system; older prints often misclassify it
        # as a note.  Skip that zone (the first real note is further right).
        if (g.cx < staff.left + 3.0 * staff.space
                and g.h >= 2.5 * staff.space and g.w <= 1.6 * staff.space):
            continue
        label = predictions.get(g.uid, {}).get("label", "skip")
        if label in ("note.black", "note.open"):
            # Beams, slurs and ornaments above the staff are sometimes
            # classified as a note; they are wide/flat compared with a head.
            if g.w > 2.5 * g.space or g.h < 0.6 * g.space:
                continue
        if label.startswith("rest."):
            duration = (bar_length if label == "rest.measure"
                        else REST_DURATIONS.get(label))
            if duration is None:
                continue
            measure = None
            for i, (x0, x1) in enumerate(bounds):
                if x0 <= g.cx < x1:
                    measure = i
                    break
            if measure is not None:
                events.append(Event(measure, g.cx, None, duration, "rest", g,
                                    hy=g.y0 + g.h / 2))
            continue
        if label == "beam.group":
            for hx, hy, duration in notes_module.split_beam_group(g.ink, g.space, templates):
                head_y = g.y0 + hy
                position = staff_position(staff, head_y)
                pitch = pitch_at(staff, clef, position, fifths)
                x = g.x0 + hx
                measure = None
                for i, (x0, x1) in enumerate(bounds):
                    if x0 <= x < x1:
                        measure = i
                        break
                if measure is not None:
                    events.append(Event(measure, x, pitch, duration, "note", g,
                                        flexible=True, hy=head_y))
            continue
        duration = duration_for(label, g.ink, g.space, legacy=legacy)
        if duration is None:
            # A notehead merged with a slur/fermata is classified slur/text;
            # the tall narrow glyph still contains the head.
            if label in ("slur", "text") and g.h >= 2.2 * g.space and g.w <= 1.6 * g.space:
                fill, cy, h, w = head_fill(g.ink, g.space)
                if w >= 0.4 * g.space and h >= 0.25 * g.space:
                    # merged slurs/fermatas hide the head outline; these are
                    # the half/quarter notes whose head was cut by the slur
                    duration = Fraction(1) if fill >= 0.75 else Fraction(2)
                    head_y = g.y0 + cy
                    position = staff_position(staff, head_y)
                    pitch = pitch_at(staff, clef, position, fifths)
                    measure = None
                    for i, (x0, x1) in enumerate(bounds):
                        if x0 <= g.cx < x1:
                            measure = i
                            break
                    if measure is not None:
                        events.append(Event(measure, g.cx, pitch, duration,
                                            "note", g, hy=head_y))
            continue
        head_y = g.y0 + head_center(g.ink, g.space)
        # A rest or ornament misclassified as a note has no notehead that
        # matches the page's notehead templates; do not emit it as a note.
        if require_head and templates:
            if (not notes_module.match_heads(g.ink, g.space, templates)
                    and g.w <= 1.1 * g.space and g.h >= 0.8 * g.space):
                # A narrow, tall headless glyph is a rest or accidental; a
                # wide headless one is usually a note merged with a slur and
                # is kept (the slur/text recovery handles those separately).
                continue
        position = staff_position(staff, head_y)
        pitch = pitch_at(staff, clef, position, fifths)
        measure = None
        for i, (x0, x1) in enumerate(bounds):
            if x0 <= g.cx < x1:
                measure = i
                break
        if measure is None:
            continue
        events.append(Event(measure, g.cx, pitch, duration, "note", g, hy=head_y))
    events.sort(key=lambda e: (e.measure, e.x))
    if dots:
        apply_dots(events, dots, staff.space)
    return events


def apply_dots(events: list[Event], dots: list[tuple[float, float]], space: float):
    """Dot to the right of a notehead makes the note dotted."""
    for event in events:
        if event.kind != "note":
            continue
        for dot_x, dot_y in dots:
            if (0.3 * space <= dot_x - event.x <= 1.4 * space
                    and abs(dot_y - event.hy) <= 0.45 * space):
                event.duration = event.duration * Fraction(3, 2)
                break


def assign_onsets_from_x(events: list[Event], bounds: list[tuple[float, float]],
                         bar_length: Fraction,
                         grid: Fraction = Fraction(1, 4)) -> list[Event]:
    """Place notes by their printed x position, fill the gaps with rests.

    The horizontal position of a notehead is a reliable rhythm cue even when
    the rest glyphs are missed (or the edition prints them unusually).  The
    glyph duration caps the note; any remaining gap becomes a rest.
    """
    by_measure: dict[int, list[Event]] = {i: [] for i in range(len(bounds))}
    for event in events:
        if event.kind == "note":
            by_measure[event.measure].append(event)
    out = []
    for measure, (x0, x1) in enumerate(bounds):
        notes = sorted(by_measure[measure], key=lambda e: e.x)
        if not notes:
            rest = next((e for e in events
                         if e.measure == measure and e.kind == "rest"), None)
            out.append(rest or Event(measure, x0, None, bar_length, "rest"))
            continue
        span = max(1.0, float(x1 - x0))
        limit = bar_length - grid if bar_length >= grid else bar_length
        onsets = []
        for event in notes:
            frac = Fraction((event.x - x0) / span).limit_denominator(64)
            steps = round(frac * bar_length / grid)
            onsets.append(min(max(Fraction(steps) * grid, Fraction(0)), limit))
        for i in range(1, len(onsets)):
            if onsets[i] <= onsets[i - 1]:
                onsets[i] = min(onsets[i - 1] + grid, limit)
        if onsets[0] > 0:
            rest = Event(measure, x0, None, onsets[0], "rest")
            rest.onset = Fraction(0)
            out.append(rest)
        for i, event in enumerate(notes):
            event.onset = onsets[i]
            gap = (onsets[i + 1] if i + 1 < len(notes) else bar_length) - onsets[i]
            if event.duration is None or event.duration <= 0:
                event.duration = gap
            if event.duration > gap:
                event.duration = gap
            if event.duration <= 0:
                event.duration = grid
            out.append(event)
            if gap > event.duration:
                rest = Event(measure, event.x + 1.0, None,
                             gap - event.duration, "rest")
                rest.onset = event.onset + event.duration
                out.append(rest)
    return out


def normalize_measures(events: list[Event], nmeasures: int,
                       bar_length: Fraction,
                       tacet: set[int] | None = None) -> list[Event]:
    """A measure without notes is a tacet measure: one full-measure rest.

    `tacet` may name measures with a printed whole-rest bar; any notes found
    there are misclassifications and are dropped.
    """
    tacet = tacet or set()
    by_measure: dict[int, list[Event]] = {i: [] for i in range(nmeasures)}
    for event in events:
        by_measure[event.measure].append(event)
    out = []
    for measure in range(nmeasures):
        if measure in tacet:
            out.append(Event(measure, 0.0, None, bar_length, "rest"))
        elif any(e.kind == "note" for e in by_measure[measure]):
            out.extend(by_measure[measure])
        else:
            out.append(Event(measure, 0.0, None, bar_length, "rest"))
    return out


def fit_durations(events: list[Event], nmeasures: int, bar_length: Fraction):
    """Fit beamed durations (halves/quarters) to the meter of each measure."""
    from itertools import product

    for measure in range(nmeasures):
        in_bar = [e for e in events if e.measure == measure]
        flexible = [e for e in in_bar if e.flexible]
        if not flexible:
            continue
        fixed = sum(e.duration for e in in_bar if not e.flexible)
        target = bar_length - fixed
        options = [Fraction(1, 2), Fraction(1, 4), Fraction(1)]
        best = None
        for combo in product(options, repeat=len(flexible)):
            error = abs(sum(combo) - target)
            preference = sum(abs(c - Fraction(1, 2)) for c in combo)
            key = (error, preference)
            if best is None or key < best[0]:
                best = (key, combo)
        for event, duration in zip(flexible, best[1]):
            event.duration = duration


def assign_onsets(events: list[Event], nmeasures: int, bar_length: Fraction):
    """Cumulative onsets within each measure; pad with rests if short."""
    by_measure: dict[int, list[Event]] = {i: [] for i in range(nmeasures)}
    for e in events:
        by_measure[e.measure].append(e)
    out = []
    for i in range(nmeasures):
        onset = Fraction(0)
        for e in sorted(by_measure[i], key=lambda x: x.x):
            e.onset = onset
            onset += e.duration
            out.append(e)
    return out


def measure_fill(events: list[Event], nmeasures: int, bar_length: Fraction):
    sums = [Fraction(0) for _ in range(nmeasures)]
    for e in events:
        sums[e.measure] += e.duration
    return sums
