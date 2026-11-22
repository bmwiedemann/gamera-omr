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
    direction: str | None = None  # stem direction ("up"/"down") when read
    voice: int = 0           # 0 = upper/main voice, 1 = lower second voice

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


def _has_flag(ink: np.ndarray, space: float) -> bool:
    """True when a filled note's stem carries a flag (an eighth note)."""
    a, _b = head_region(ink, space)
    if a <= 2:
        return False
    top = ink[:a, :]
    cols = np.flatnonzero(top.any(axis=0))
    if len(cols) == 0:
        return False
    # A quarter-note stem is one or two pixels wide; a flag is a curved
    # stroke spanning a good fraction of a staff space.
    return (cols[-1] - cols[0] + 1) >= 0.55 * space


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
    if _has_flag(ink, space):
        return Fraction(1, 2)
    return Fraction(1)


def decode_staff(staff: Staff, glyphs: list[Glyph], predictions: dict[str, dict],
                 bounds: list[tuple[float, float]], clef: str,
                 fifths: int = 0, templates=None,
                 dots: list[tuple[float, float]] | None = None,
                 legacy: bool = False,
                 bar_length: Fraction = Fraction(4),
                 require_head: bool = False,
                 recover_edges: bool = False,
                 clef_changes: list[tuple[float, str]] | None = None,
                 voices: bool = False,
                 beam_row_heads: bool = False) -> list[Event]:
    """Decode one staff's glyphs into note events.

    `beam_row_heads` enables the row-run detector for heads fused to a sloping
    beam; it is used for the NBA continuo crops, where it is reliable, and is
    off for the old prints, whose system-start clefs it can mistake for groups.
    """
    from . import notes as notes_module  # local import avoids a cycle

    if templates is None:
        templates = notes_module.make_templates(glyphs, predictions)
    events = []

    def clef_at(x: float) -> str:
        chosen = clef
        for change_x, changed in clef_changes or ():
            if x >= change_x:
                chosen = changed
            else:
                break
        return chosen

    band_top, band_bottom = staff.band(above=4.0, below=1.2)
    # The clef, key and time signature at the start of a system are tall and
    # narrow like a note+stem.  Find where the printed music really starts;
    # only glyphs before that which contain no notehead are signature glyphs.
    signature_limit = None
    if templates:
        for g in glyphs:
            if g.staff != staff.index:
                continue
            if not (band_top <= g.cy <= band_bottom):
                continue
            label = predictions.get(g.uid, {}).get("label", "skip")
            if label == "beam.group":
                if g.w < 2.5 * staff.space:
                    continue
                if len(notes_module.split_beam_group(
                        g.ink, g.space, templates,
                        use_rowrun=beam_row_heads)) < 2:
                    continue
                x = g.cx
            elif label in ("note.black", "note.open"):
                if not notes_module.match_heads(g.ink, g.space, templates):
                    continue
                x = g.cx
            else:
                continue
            signature_limit = x if signature_limit is None else min(
                signature_limit, x)

    def measure_of(x: float) -> int | None:
        if not bounds:
            return None
        for i, (x0, x1) in enumerate(bounds):
            if x0 <= x < x1:
                return i
        # Notes immediately after the clef or before the last barline fall
        # outside the trimmed bounds; keep those close to the edge in the
        # edge measures, but do not pull in margin/clef components.
        if x < bounds[0][0]:
            left_limit = (staff.left + 1.5 * staff.space if recover_edges
                          else bounds[0][0] - 2.0 * staff.space)
            return 0 if x >= left_limit else None
        if x >= bounds[-1][1]:
            right_limit = (staff.right if recover_edges
                           else bounds[-1][1] + 2.0 * staff.space)
            return len(bounds) - 1 if x <= right_limit else None
        return None

    for g in glyphs:
        if g.staff != staff.index:
            continue
        if not (band_top <= g.cy <= band_bottom):
            continue
        # A clef (and the initial key/time signature) is a tall, narrow glyph
        # at the very start of the system; older prints often misclassify it
        # as a note.  Skip that zone (the first real note is further right).
        if (g.cx < staff.left + 3.0 * staff.space
                and g.h >= 2.5 * staff.space and g.w <= 1.6 * staff.space):
            continue
        if (signature_limit is not None and g.cx < signature_limit
                and g.h >= 2.2 * staff.space and g.w <= 1.6 * staff.space
                and not notes_module.match_heads(g.ink, g.space, templates)):
            continue
        label = predictions.get(g.uid, {}).get("label", "skip")
        wide_note = False
        if label in ("note.black", "note.open"):
            # Beams, slurs and ornaments above the staff are sometimes
            # classified as a note; they are flat compared with a head.  A
            # wide but tall glyph is a beamed group misclassified as one
            # note, so split it below instead of dropping its heads.
            if g.h < 0.6 * g.space:
                continue
            wide_note = g.w > 2.5 * g.space
        if label.startswith("rest."):
            duration = (bar_length if label == "rest.measure"
                        else REST_DURATIONS.get(label))
            if duration is None:
                continue
            measure = measure_of(g.cx)
            if measure is not None:
                events.append(Event(measure, g.cx, None, duration, "rest", g,
                                    hy=g.y0 + g.h / 2))
            continue
        if label == "beam.group" or wide_note:
            heads = notes_module.split_beam_group(
                g.ink, g.space, templates,
                use_rowrun=beam_row_heads and not wide_note)
            # Only a genuinely polyphonic wide glyph is a misclassified
            # beamed group.  A single split head from a wide note+slur is a
            # false positive in the NBA/Rösler/Sibley prints.
            if not heads or (wide_note and len(heads) < 2):
                continue
            direction = (notes_module.stem_direction(g.ink, g.space)
                         if voices else None)
            for hx, hy, duration in heads:
                head_y = g.y0 + hy
                x = g.x0 + hx
                position = staff_position(staff, head_y)
                pitch = pitch_at(staff, clef_at(x), position, fifths)
                measure = measure_of(x)
                if measure is not None:
                    events.append(Event(measure, x, pitch, duration, "note", g,
                                        flexible=True, hy=head_y,
                                        direction=direction))
            continue
        duration = duration_for(label, g.ink, g.space, legacy=legacy)
        if duration is None:
            # A large piano figure of stacked chords and beams can be one
            # connected component and get classified as text.  It still
            # contains noteheads, so split it like a beamed group.
            if (voices and label in ("text", "skip")
                    and g.w >= 2.0 * g.space and g.h >= 2.5 * g.space):
                heads = notes_module.split_beam_group(
                    g.ink, g.space, templates, use_rowrun=beam_row_heads)
                if len(heads) >= 2:
                    direction = notes_module.stem_direction(g.ink, g.space)
                    for hx, hy, head_duration in heads:
                        head_y = g.y0 + hy
                        x = g.x0 + hx
                        position = staff_position(staff, head_y)
                        pitch = pitch_at(staff, clef_at(x), position, fifths)
                        measure = measure_of(x)
                        if measure is not None:
                            events.append(Event(
                                measure, x, pitch, head_duration, "note", g,
                                flexible=True, hy=head_y, direction=direction))
                    continue
            # Staff-line removal can cut an isolated notehead away from its
            # stem; the remaining blob is a solid ellipse and the classifier
            # calls it text.  `head_like` verifies both shape and stem context.
            if label in ("text", "skip") and g.head_like:
                fill, cy, h, w = head_fill(g.ink, g.space)
                if w >= 0.8 * g.space:
                    duration = Fraction(1) if fill >= 0.6 else Fraction(2)
                    head_y = g.y0 + cy
                    position = staff_position(staff, head_y)
                    pitch = pitch_at(staff, clef_at(g.cx), position, fifths)
                    measure = measure_of(g.cx)
                    if measure is not None:
                        events.append(Event(measure, g.cx, pitch, duration,
                                            "note", g, flexible=True,
                                            hy=head_y,
                                            direction=(notes_module.stem_direction(
                                                g.ink, g.space) if voices else None)))
                    continue
            # A notehead merged with a slur/fermata is classified slur/text;
            # the tall narrow glyph still contains the head.  A bare stem or
            # barline is also text and is only a few pixels wide, so require a
            # head-sized horizontal extent before recovering a note.
            if (label in ("slur", "text") and g.h >= 2.2 * g.space
                    and 0.6 * g.space <= g.w <= 1.6 * g.space):
                fill, cy, h, w = head_fill(g.ink, g.space)
                if (w >= 0.6 * g.space and 0.25 * g.space <= h <= 2.0 * g.space):
                    # merged slurs/fermatas hide the head outline; these are
                    # the half/quarter notes whose head was cut by the slur
                    duration = Fraction(1) if fill >= 0.75 else Fraction(2)
                    head_y = g.y0 + cy
                    position = staff_position(staff, head_y)
                    pitch = pitch_at(staff, clef_at(g.cx), position, fifths)
                    measure = measure_of(g.cx)
                    if measure is not None:
                        events.append(Event(measure, g.cx, pitch, duration,
                                            "note", g, hy=head_y,
                                            direction=(notes_module.stem_direction(
                                                g.ink, g.space) if voices else None)))
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
        pitch = pitch_at(staff, clef_at(g.cx), position, fifths)
        measure = measure_of(g.cx)
        if measure is None:
            continue
        direction = (notes_module.stem_direction(g.ink, g.space)
                     if voices else None)
        events.append(Event(measure, g.cx, pitch, duration, "note", g,
                            hy=head_y, direction=direction))
    events.sort(key=lambda e: (e.measure, e.x))
    if dots:
        apply_dots(events, dots, staff.space)
    return events


def split_staff_voices(events: list[Event], space: float,
                       chord_tol: float = 0.9,
                       min_pairs: int = 1) -> list[list[Event]] | None:
    """Split one staff's events into independent musical voices.

    Stem direction separates the voices, but only when notes of opposite
    direction actually sound together: a single homophonic line also flips
    stem direction with the note's position on the staff, and must stay one
    voice.  Returns two event lists (the main voice first, the one with more
    notes) or None for a single voice.  The `voice` field of every event is
    set either way.
    """
    notes = [e for e in events if e.kind == "note"]
    up = [e for e in notes if e.direction == "up"]
    down = [e for e in notes if e.direction == "down"]
    if not up or not down:
        return None
    tolerance = chord_tol * space
    overlapping = sum(
        1 for d in down
        if any(abs(d.x - u.x) <= tolerance and d.measure == u.measure
               for u in up)
    )
    if overlapping < min_pairs:
        return None

    known = up + down
    for event in events:
        if event.kind == "note" and event.direction in ("up", "down"):
            event.voice = 0 if event.direction == "up" else 1
        elif event.kind == "note":
            # A head cut by staff removal has no stem left; give it to the
            # voice whose notes are closest in the staff.
            nearest = min(
                known,
                key=lambda n: (n.x - event.x) ** 2
                + (0.5 * (n.hy - event.hy)) ** 2,
            )
            event.voice = 0 if nearest.direction == "up" else 1
        else:
            event.voice = 0
    voices = ([e for e in events if e.voice == 0],
              [e for e in events if e.voice == 1])
    if not voices[0] or not voices[1]:
        for event in events:
            event.voice = 0
        return None
    if len(up) < len(down):
        # The main voice is the larger group; number it first so the melody
        # stays voice 1 even when its stems point down.
        for event in events:
            event.voice = 1 - event.voice
        voices = (voices[1], voices[0])
    return list(voices)


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
                         grid: Fraction = Fraction(1, 4),
                         anchor: bool = True,
                         chord_tol: float = 0.0) -> list[Event]:
    """Place notes by their printed x position, fill the gaps with rests.

    The horizontal position of a notehead is a reliable rhythm cue even when
    the rest glyphs are missed (or the edition prints them unusually).  Notes
    are interpolated between the first and last notehead rather than across
    the full barline-to-barline width, because printed measures carry left
    (clef/rest) and right margins that otherwise shift every onset.  Detected
    leading/trailing rests anchor the interpolation; the glyph duration caps
    the note and any remaining gap becomes a rest.
    """
    by_measure: dict[int, list[Event]] = {i: [] for i in range(len(bounds))}
    for event in events:
        by_measure[event.measure].append(event)
    out = []
    for measure, (x0, x1) in enumerate(bounds):
        items = by_measure[measure]
        notes = sorted((e for e in items if e.kind == "note"),
                       key=lambda e: e.x)
        rests = [e for e in items if e.kind == "rest"]
        if chord_tol > 0 and notes:
            # Simultaneous heads form one chord and share its duration.
            groups: list[list[Event]] = [[notes[0]]]
            for e in notes[1:]:
                if abs(e.x - groups[-1][-1].x) <= chord_tol:
                    groups[-1].append(e)
                else:
                    groups.append([e])
            for group in groups:
                durations = [e.duration for e in group if e.duration]
                if durations:
                    value = max(durations)
                    for e in group:
                        e.duration = value
        if not notes:
            out.append(rests[0] if rests
                       else Event(measure, x0, None, bar_length, "rest"))
            continue
        limit = bar_length - grid if bar_length >= grid else bar_length
        x_first, x_last = notes[0].x, notes[-1].x
        lead = sum((r.duration for r in rests if r.x < x_first),
                   Fraction(0))
        trail = sum((r.duration for r in rests if r.x > x_last),
                    Fraction(0))
        first = min(max(Fraction(0), lead), limit)
        last_dur = notes[-1].duration
        if last_dur is None or last_dur <= 0:
            last_dur = grid
        last = min(max(first, bar_length - trail - last_dur), limit)

        onsets: list[Fraction] = []
        if anchor and x_last > x_first and last > first:
            span = float(x_last - x_first)
            for event in notes:
                frac = Fraction((event.x - x_first) / span).limit_denominator(64)
                raw = first + frac * (last - first)
                onsets.append(Fraction(round(raw / grid)) * grid)
            onsets[0], onsets[-1] = first, last
        else:
            span = max(1.0, float(x1 - x0))
            for event in notes:
                frac = Fraction((event.x - x0) / span).limit_denominator(64)
                raw = frac * bar_length
                onsets.append(min(max(Fraction(round(raw / grid)) * grid,
                                      Fraction(0)), limit))
        for i in range(1, len(onsets)):
            same_chord = (chord_tol > 0
                          and abs(notes[i].x - notes[i - 1].x) <= chord_tol)
            if same_chord:
                onsets[i] = onsets[i - 1]
            elif onsets[i] <= onsets[i - 1]:
                onsets[i] = min(onsets[i - 1] + grid, limit)
        if onsets[0] > 0:
            rest = Event(measure, x0, None, onsets[0], "rest")
            rest.onset = Fraction(0)
            out.append(rest)
        # Emit one onset group at a time so simultaneous heads become a chord
        # with one shared duration and one gap rest.
        groups: list[list[int]] = []
        for i, onset in enumerate(onsets):
            if groups and onset == onsets[groups[-1][0]]:
                groups[-1].append(i)
            else:
                groups.append([i])
        for gi, idxs in enumerate(groups):
            onset = onsets[idxs[0]]
            next_onset = (onsets[groups[gi + 1][0]]
                          if gi + 1 < len(groups) else bar_length)
            gap = next_onset - onset
            durations = [notes[i].duration for i in idxs
                         if notes[i].duration and notes[i].duration > 0]
            duration = max(durations) if durations else gap
            if duration > gap:
                duration = gap
            if duration <= 0:
                duration = grid
            for i in idxs:
                notes[i].onset = onset
                notes[i].duration = duration
                out.append(notes[i])
            if gap > duration:
                rest = Event(measure, notes[idxs[0]].x + 1.0, None,
                             gap - duration, "rest")
                rest.onset = onset + duration
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
    """Sounding length per measure: the longest voice, chords counted once.

    Chord members share an onset within one voice and must not be added
    together, and a second voice sounds alongside the first rather than after
    it, so the measure's fill is the maximum over its voices.
    """
    grouped: dict[tuple[int, int, Fraction], Fraction] = {}
    for e in events:
        key = (e.measure, e.voice, e.onset)
        grouped[key] = max(grouped.get(key, Fraction(0)), e.duration)
    voices: dict[tuple[int, int], Fraction] = {}
    for (measure, voice, _onset), duration in grouped.items():
        voices[(measure, voice)] = voices.get((measure, voice), Fraction(0)) + duration
    sums = [Fraction(0) for _ in range(nmeasures)]
    for (measure, _voice), duration in voices.items():
        if 0 <= measure < nmeasures:
            sums[measure] = max(sums[measure], duration)
    return sums
