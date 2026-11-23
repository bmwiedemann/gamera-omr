"""Notation decoding: notehead position -> pitch, glyph class -> duration."""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction

import cv2
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
                 beam_row_heads: bool = False,
                 binary: np.ndarray | None = None,
                 stem_heads: bool = False) -> list[Event]:
    """Decode one staff's glyphs into note events.

    `stem_heads` reads single notes and chords stem-first (see
    notes.stemmed_heads); `binary` (the page before staff removal) enables
    the ledger-line check for heads outside the staff.

    `beam_row_heads` enables the row-run detector for heads fused to a sloping
    beam; it is used for the NBA continuo crops, where it is reliable, and is
    off for the old prints, whose system-start clefs it can mistake for groups.
    """
    from . import notes as notes_module  # local import avoids a cycle

    if templates is None:
        templates = notes_module.make_templates(glyphs, predictions,
                                                oval=stem_heads)
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
    time_signature = time_signature_glyphs(staff, glyphs)
    lone: list[Event] = []
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
    signature_end = (_signature_end(staff, glyphs, time_signature,
                                    notes_module) if stem_heads else None)
    if stem_heads and signature_limit is not None:
        # The page's head templates miss some heads (on ledger lines, with
        # a flag), and every note before the first matching one would be
        # dropped as part of the signature.  The clef, key and time
        # signature are a run of closely spaced glyphs from the staff start;
        # the first glyph after a gap that is no accidental ends it.
        signature_limit = min(signature_limit, signature_end)

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
        if g.uid in time_signature:
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
        if (stem_heads and label != "rest.8"
                and g.x0 >= signature_end
                and staff.top - 0.5 * staff.space <= g.cy
                <= staff.bottom + 0.5 * staff.space
                and notes_module.eighth_rest(g.ink, g.space)):
            # The glyph models often call an eighth rest text or a slur.
            measure = measure_of(g.cx)
            if measure is not None:
                events.append(Event(measure, g.cx, None, Fraction(1, 2),
                                    "rest", g, hy=g.cy))
            continue
        if stem_heads and label in ("text", "skip", "slur") or (
                stem_heads and label.startswith("rest.")):
            # A head whose stem broke off (thin stems in old prints) is a
            # small solid blob, often called a rest or text.
            head = notes_module.lone_head(g.ink, g.space)
            if head is not None:
                hx, hy = head
                head_y = g.y0 + hy
                x = g.x0 + hx
                measure = measure_of(x)
                if measure is not None:
                    pitch = pitch_at(staff, clef_at(x),
                                     staff_position(staff, head_y), fifths)
                    lone.append(Event(measure, x, pitch, Fraction(1), "note",
                                      g, flexible=True, hy=head_y))
                continue
        if label.startswith("rest."):
            if stem_heads and g.x0 < signature_end:
                continue  # a piece of the clef or a common-time C
            duration = (bar_length if label == "rest.measure"
                        else REST_DURATIONS.get(label))
            if duration is None:
                continue
            measure = measure_of(g.cx)
            if measure is not None:
                events.append(Event(measure, g.cx, None, duration, "rest", g,
                                    hy=g.y0 + g.h / 2))
            continue
        # A single stemmed note or chord called a beamed group is read from
        # its stem: the group splitter would count its flag or stem end as
        # a beam and halve the duration.
        single_stem = (stem_heads and label == "beam.group"
                       and g.w <= 1.8 * g.space and g.h >= 2.2 * g.space)
        if (stem_heads and not voices and label == "beam.group"
                and not wide_note and g.h < 2.0 * g.space):
            continue  # a tie or slur: a beamed group has stems
        if (label == "beam.group" and not single_stem) or wide_note:
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
        # A flagged note past the system start is often called a clef,
        # slur or text by the glyph models.
        stemmed_label = (label in ("note.black", "note.open") or single_stem
                         or (label in ("text", "slur", "skip", "clef")
                             and g.h >= 2.2 * g.space
                             # a ledger line widens a note on it
                             and 0.6 * g.space <= g.w
                             <= (3.0 if stem_heads else 2.6) * g.space))
        if label == "clef" and _clef_dots(g, glyphs, staff) and not (
                stem_heads and signature_limit is not None
                and g.x0 > signature_limit
                and notes_module.stemmed_heads(g.ink, g.space, templates)[0]):
            continue  # a real bass clef, not a note with a staccato dot
        if stem_heads and stemmed_label:
            # Find the stem first, then the heads at its end.  This keeps a
            # flag out of the head position, reads every head of a chord,
            # and recovers flagged notes the classifier mislabels.
            heads, direction = notes_module.stemmed_heads(
                g.ink, g.space, templates)
            if direction in ("accidental", "nohead"):
                continue
            if not heads and label == "note.black" and g.h > 1.6 * g.space:
                # Taller than a head but without a stem: a clef body,
                # a rest or a figure, not a filled note.
                continue
            if heads:
                for hx, hy, head_duration in heads:
                    if (duration is not None and duration >= 2
                            and head_duration >= 1):
                        # an open head the stem reader took for a filled
                        # one; a flagged or beamed note is never open
                        head_duration = duration
                    head_y = g.y0 + hy
                    x = g.x0 + hx
                    position = staff_position(staff, head_y)
                    pitch = pitch_at(staff, clef_at(x), position, fifths)
                    measure = measure_of(x)
                    if measure is not None:
                        events.append(Event(measure, x, pitch, head_duration,
                                            "note", g, hy=head_y,
                                            direction=direction))
                continue
        if duration is None:
            # A large piano figure of stacked chords and beams can be one
            # connected component and get classified as text.  It still
            # contains noteheads, so split it like a beamed group.
            # A beamed group joined to its slur is often called a slur.
            if (((voices and label in ("text", "skip"))
                 or (stem_heads and label in ("text", "skip", "slur")))
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
                    and 0.6 * g.space <= g.w <= 1.6 * g.space
                    and not (stem_heads and notes_module.accidental_kind(
                        g.ink, g.space))):
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
    for head in lone:
        # Take the rhythm and stem direction of the beamed group the head
        # broke away from, when there is one next to it.
        near = [e for e in events
                if e.kind == "note" and e.flexible and e.measure == head.measure
                and abs(e.x - head.x) <= 2.5 * staff.space]
        if near:
            nearest = min(near, key=lambda e: abs(e.x - head.x))
            head.duration = nearest.duration
            head.direction = nearest.direction
        events.append(head)
    if stem_heads and binary is not None:
        events = _merge_open_heads(
            events, staff, glyphs, binary, bounds, measure_of, clef_at, fifths,
            signature_limit, signature_end, clef_changes, notes_module)
    if stem_heads:
        offsets: dict[int, float] = {}
        for e in events:
            if e.kind != "note":
                continue
            # A slightly tilted staff (below the deskew threshold) puts the
            # lines a quarter space off their average height at the ends.
            off = 0.0
            if binary is not None:
                bucket = int(e.x // (2 * staff.space))
                if bucket not in offsets:
                    offsets[bucket] = line_offset(binary, staff, e.x)
                off = offsets[bucket]
            y = e.hy
            if e.glyph is not None:
                y = refine_head_y(e.glyph, staff, e.x, e.hy, off)
            if y != e.hy or off:
                e.hy = y
                e.pitch = pitch_at(staff, clef_at(e.x),
                                   staff_position(staff, y - off), fifths)
        apply_accidentals(events, staff, glyphs, fifths,
                          start=(signature_end + staff.space
                                 if signature_end is not None else None))
    if binary is not None:
        # A head beyond the first ledger-line position needs that ledger
        # line; dynamics, fingerings and the other staff's notes in the gap
        # between the staves do not have one.
        # Rests of this staff sit on it; one in the gap between the staves
        # belongs to the other staff's voice.
        events = [e for e in events
                  if (has_ledger(binary, staff, e.x, e.hy) if e.kind == "note"
                      else staff.top - staff.space <= e.hy
                      <= staff.bottom + staff.space)]
    events.sort(key=lambda e: (e.measure, e.x))
    if dots:
        apply_dots(events, dots, staff.space)
    return events


def _merge_open_heads(events, staff, glyphs, binary, bounds, measure_of, clef_at,
                      fifths, signature_limit, signature_end, clef_changes,
                      notes_module):
    """Give half and whole notes found by their counters their duration.

    A head the glyph readers found already takes the open head's duration;
    one they missed is added.  Rests and duplicate notes read from the
    pieces of a whole note cut by line removal are dropped.
    """
    space = staff.space
    # The first measure starts after the clef and signature, not at a
    # barline.
    barlines = sorted({x for b in bounds[1:] for x in b}
                      | ({bounds[-1][1]} if bounds else set()))
    found = notes_module.open_heads(binary, staff, barlines)
    # A flat's bowl is a counter too.
    accidentals = [g for g in glyphs if g.staff == staff.index
                   and notes_module.accidental_kind(g.ink, g.space)]
    start = max(signature_limit if signature_limit is not None else 0.0,
                signature_end, staff.left + 3.0 * space)
    drop = set()
    for hx, hy, duration, direction, sure in found:
        if hx < start:
            continue
        if any(g.x0 <= hx <= g.x0 + g.w and g.y0 <= hy <= g.y0 + g.h
               for g in accidentals):
            continue
        if any(abs(hx - cx) <= 2.5 * space for cx, _c in clef_changes or ()):
            continue
        measure = measure_of(hx)
        if measure is None:
            continue
        # The glyph readers' head for this counter, if any.
        near = [e for e in events
                if e.kind == "note" and id(e) not in drop
                and abs(e.x - hx) <= 0.6 * space
                and abs(e.hy - hy) <= 0.7 * space]
        if not near and not sure:
            continue
        for e in events:
            if (e.kind == "rest" and abs(e.x - hx) <= 1.0 * space
                    and abs(e.hy - hy) <= 1.2 * space):
                drop.add(id(e))
        if near:
            keep = min(near, key=lambda e: abs(e.hy - hy) + abs(e.x - hx))
            if abs(keep.hy - hy) > 0.25 * space:
                # The glyph readers miss the middle of a hollow head; the
                # counter's centre is exact.
                keep.hy = hy
                keep.pitch = pitch_at(staff, clef_at(keep.x),
                                      staff_position(staff, hy), fifths)
                keep.glyph = None
            if keep.duration < duration:
                keep.duration = duration
                keep.flexible = False
            if keep.direction is None:
                keep.direction = direction
            for e in near:
                if e is not keep and abs(e.hy - hy) <= 0.35 * space:
                    drop.add(id(e))
            continue
        pitch = pitch_at(staff, clef_at(hx), staff_position(staff, hy), fifths)
        events.append(Event(measure, hx, pitch, duration, "note", None,
                            hy=hy, direction=direction))
    return [e for e in events if id(e) not in drop]


def _signature_end(staff: Staff, glyphs: list[Glyph], time_signature: set[str],
                   notes_module) -> float:
    """x where the clef, key and time signature at the staff start end."""
    space = staff.space
    band = [g for g in glyphs
            if g.staff == staff.index
            and staff.top - 0.5 * space <= g.cy <= staff.bottom + 0.5 * space
            and g.h >= 0.8 * space]
    band.sort(key=lambda g: g.x0)
    end = staff.left
    for g in band:
        if g.x0 + g.w <= end:
            continue
        # The key signature can stand two spaces off the clef.
        if g.x0 > end + 2.0 * space:
            break
        clef = g.x0 < staff.left + 2.5 * space
        # A common-time C fills the middle of the staff and has no stem.
        # (line removal may split it into a body and its tips)
        common = (1.5 * space <= g.h <= 2.6 * space
                  and 0.5 * space <= g.w <= 1.8 * space
                  and abs(g.cy - staff.center) <= 0.5 * space
                  and not notes_module.stemmed_heads(g.ink, g.space)[0])
        if (clef or common or g.uid in time_signature
                or notes_module.accidental_kind(g.ink, g.space) is not None):
            end = max(end, g.x0 + g.w)
            continue
        if g.x0 < end + 0.3 * space and g.w <= 1.2 * space:
            # pieces of the clef (its dots) and of a broken accidental
            end = max(end, g.x0 + g.w)
            continue
        break
    return end + 0.5 * space


def line_offset(binary: np.ndarray, staff: Staff, x: float) -> float:
    """How far the staff lines near x lie below their average heights.

    Each line is found as the rows that are inked across most of a window
    four spaces wide; the median shift of the lines found is returned, or 0
    when fewer than three lines are clear (beams, chords, the staff end).
    """
    space = staff.space
    h, w = binary.shape
    x0, x1 = int(round(x - 2 * space)), int(round(x + 2 * space))
    if x0 < 0 or x1 > w or x0 < staff.left or x1 > staff.right:
        x0, x1 = max(0, int(staff.left), x0), min(w, int(staff.right), x1)
    if x1 - x0 < 2 * space:
        return 0.0
    reach = max(1, int(round(0.4 * space)))
    shifts = []
    for line in staff.lines:
        y = int(round(line))
        a, b = max(0, y - reach), min(h, y + reach + 1)
        counts = (binary[a:b, x0:x1] > 0).mean(axis=1)
        rows = np.flatnonzero(counts >= 0.8)
        if len(rows) == 0 or len(rows) > max(4, staff.thickness + 3):
            continue
        shifts.append(a + rows.mean() - line)
    if len(shifts) < 3:
        return 0.0
    off = float(np.median(shifts))
    return off if abs(off) >= 1.0 else 0.0


def refine_head_y(g: Glyph, staff: Staff, x: float, y: float,
                  offset: float = 0.0) -> float:
    """Head centre from the head's own top and bottom edges.

    With the staff-line crossings kept, a head touching a line from one side
    only (just above the top line or below the bottom line) carries that
    line's piece along; the blob detectors then put the centre a third of a
    space towards the line and the pitch a step off.  Take the vertical ink
    run through the head's middle columns and trim a line lying at one end
    of a run that is too tall for a head.
    """
    space = staff.space
    thick = max(1.0, staff.thickness)
    ink = g.ink
    h, w = ink.shape
    cx, cy = int(round(x - g.x0)), int(round(y - g.y0))
    if not (0 <= cy < h):
        return y
    half = max(1, int(round(0.2 * space)))
    tops, bottoms = [], []
    for col in range(max(0, cx - half), min(w, cx + half + 1)):
        column = ink[:, col]
        if not column[cy]:
            continue
        a = cy
        while a - 1 >= 0 and column[a - 1]:
            a -= 1
        b = cy
        while b + 1 < h and column[b + 1]:
            b += 1
        tops.append(a)
        bottoms.append(b)
    if len(tops) < 2:
        return y
    top = float(np.median(tops)) + g.y0
    bottom = float(np.median(bottoms)) + g.y0
    height = bottom - top + 1
    if 0.7 * space <= height <= 1.05 * space:
        return (top + bottom) / 2.0
    if not (1.05 * space < height <= 1.6 * space):
        return y

    def on_line(edge):
        # The line's average height can be off by a pixel or two locally.
        return any(abs(edge - line - offset) <= 0.5 * thick + 0.2 * space
                   for line in staff.lines)

    # A head in a space between two lines spans both: its middle is right.
    top_line, bottom_line = on_line(top), on_line(bottom)
    if not (top_line or bottom_line):
        return y
    if top_line and not bottom_line:
        top += thick + 1
    elif bottom_line and not top_line:
        bottom -= thick + 1
    return (top + bottom) / 2.0


def nearby_accidental(glyphs: list[Glyph], staff: Staff, x: float, y: float,
                      notes_module) -> str | None:
    """Kind of an accidental just left of a head, from any glyph pieces.

    An accidental close to its head is often one glyph with the head's beam
    group, or cut into stroke and bowl by line removal.  The window ends
    left of the head (and of a down-stem on its left) and spans the
    accidental's height: a flat's bowl sits at the head with its stroke
    rising two spaces; sharps and naturals are centred on it.  The pieces
    are pasted together and gaps the size of a staff line are closed.
    """
    space = staff.space
    x0, x1 = int(round(x - 2.2 * space)), int(round(x - 0.6 * space))
    y0, y1 = int(round(y - 2.3 * space)), int(round(y + 1.7 * space))
    if x1 - x0 < 0.5 * space:
        return None
    canvas = np.zeros((y1 - y0 + 1, x1 - x0 + 1), np.uint8)
    for g in glyphs:
        if g.staff != staff.index:
            continue
        gh, gw = g.ink.shape
        ax, bx = max(x0, g.x0), min(x1 + 1, g.x0 + gw)
        ay, by = max(y0, g.y0), min(y1 + 1, g.y0 + gh)
        if ax >= bx or ay >= by:
            continue
        canvas[ay - y0:by - y0, ax - x0:bx - x0] |= (
            g.ink[ay - g.y0:by - g.y0, ax - g.x0:bx - g.x0] > 0)
    if not canvas.any():
        return None
    gap = max(3, int(round(staff.thickness)) + 2)
    closed = cv2.morphologyEx(canvas, cv2.MORPH_CLOSE,
                              np.ones((gap, 1), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(closed,
                                                           connectivity=8)
    hy = y - y0
    # single pieces, and a flat's stroke with its bowl cut off it
    groups = [[j] for j in range(1, n)]
    for i in range(1, n):
        for j in range(1, n):
            gap = stats[j][0] - (stats[i][0] + stats[i][2])
            if i != j and 0 <= gap <= 0.35 * space:
                groups.append([i, j])
    for group in groups:
        bx = min(stats[j][0] for j in group)
        by = min(stats[j][1] for j in group)
        ex = max(stats[j][0] + stats[j][2] for j in group)
        ey = max(stats[j][1] + stats[j][3] for j in group)
        bw, bh = ex - bx, ey - by
        if bh < 1.6 * space:
            continue
        part = np.isin(labels[by:ey, bx:ex], group).astype(np.uint8)
        kind = notes_module.accidental_kind(part, space)
        if kind is None or (len(group) > 1 and kind != "flat"):
            continue
        centre = by + (0.68 * bh if kind == "flat" else 0.5 * bh)
        if abs(centre - hy) <= 0.45 * space:
            return kind
    return None


def apply_accidentals(events: list[Event], staff: Staff, glyphs: list[Glyph],
                      fifths: int, start: float | None = None) -> None:
    """Apply printed sharps, naturals and flats to the notes they precede.

    An accidental stands just left of its head at the same height and holds
    for later notes on the same staff position in the same measure.  The
    first glyphs of the staff that match the key signature's count are the
    key signature itself and are skipped.
    """
    from . import notes as notes_module  # local import avoids a cycle

    space = staff.space
    found = []
    for g in glyphs:
        if g.staff != staff.index:
            continue
        if not (staff.top - 3 * space <= g.cy <= staff.bottom + 3 * space):
            continue
        kind = notes_module.accidental_kind(g.ink, g.space)
        if kind is not None:
            found.append((g, kind))
    found.sort(key=lambda gk: gk[0].x0)
    key_count = abs(fifths or 0)
    signature = [gk for gk in found
                 if gk[0].x0 < staff.left + 8.0 * space][:key_count]
    found = [gk for gk in found if gk not in signature]
    notes = [e for e in events if e.kind == "note" and e.pitch is not None]
    taken: set[int] = set()
    marked: list[tuple[Event, str]] = []
    for g, kind in found:
        centre = g.y0 + (0.68 * g.h if kind == "flat" else 0.5 * g.h)
        best = None
        for i, e in enumerate(notes):
            if i in taken:
                continue
            gap = (e.x - 0.6 * space) - (g.x0 + g.w)
            dy = abs(e.hy - centre)
            if -0.4 * space <= gap <= 2.0 * space and dy <= 0.45 * space:
                if best is None or (dy, gap) < best[0]:
                    best = ((dy, gap), i)
        if best is not None:
            taken.add(best[1])
            marked.append((notes[best[1]], kind))
    # Accidentals joined to a beam group or cut apart by line removal.
    for i, e in enumerate(notes):
        if i in taken or e.x < (start if start is not None
                                else staff.left + 8.0 * space):
            continue
        kind = nearby_accidental(glyphs, staff, e.x, e.hy, notes_module)
        if kind is not None:
            marked.append((e, kind))
    alter = {"sharp": 1, "natural": 0, "flat": -1}
    for event, kind in sorted(marked, key=lambda ek: ek[0].x):
        step = event.pitch.diatonicNoteNum
        for e in notes:
            if (e is event or (e.measure == event.measure and e.x > event.x
                               and e.pitch.diatonicNoteNum == step)):
                p = mpitch.Pitch()
                p.diatonicNoteNum = step
                if alter[kind]:
                    p.accidental = mpitch.Accidental(alter[kind])
                e.pitch = p


def has_ledger(binary: np.ndarray, staff: Staff, x: float, y: float) -> bool:
    """False when a head outside the staff lacks one of its ledger lines.

    Every ledger position between the staff and the head must carry a
    horizontal stroke clearly wider than a stem around the head's x.  A beam
    or a stray stroke can fake one of them, rarely all of them.
    """
    space = staff.space
    if y < staff.top - 0.75 * space:
        count = int(np.floor((staff.top - y) / space + 0.25))
        ledgers = [staff.top - k * space for k in range(1, count + 1)]
    elif y > staff.bottom + 0.75 * space:
        count = int(np.floor((y - staff.bottom) / space + 0.25))
        ledgers = [staff.bottom + k * space for k in range(1, count + 1)]
    else:
        return True
    h, w = binary.shape
    half = max(1, int(round(0.15 * space)) + 1)
    x0, x1 = int(round(x - 1.4 * space)), int(round(x + 1.4 * space)) + 1
    if x0 < 0 or x1 > w:
        return True
    for ledger in ledgers:
        y0, y1 = int(round(ledger)) - half, int(round(ledger)) + half + 1
        if y0 < 0 or y1 > h:
            return True
        cols = binary[y0:y1, x0:x1].any(axis=0)
        best = run = 0
        for on in cols:
            run = run + 1 if on else 0
            best = max(best, run)
        if best < 1.1 * space:
            return False
    return True


def _clef_dots(g: Glyph, glyphs: list[Glyph], staff: Staff) -> bool:
    """True when two dots stand right of `g`, one above the other (F clef)."""
    space = staff.space
    dots = [d for d in glyphs
            if d.staff == g.staff and d.uid != g.uid
            and g.x0 + g.w - 0.2 * space <= d.x0 <= g.x0 + g.w + 1.5 * space
            and d.w <= 0.65 * space and d.h <= 0.65 * space
            and staff.top - 0.5 * space <= d.cy <= staff.bottom]
    return any(0.6 * space <= abs(a.cy - b.cy) <= 1.4 * space
               and abs(a.cx - b.cx) <= 0.6 * space
               for i, a in enumerate(dots) for b in dots[i + 1:])


def time_signature_glyphs(staff: Staff, glyphs: list[Glyph]) -> set[str]:
    """Uids of numeric time-signature digits on this staff.

    The two digits of a time signature fill the upper and the lower half of
    the staff exactly, one above the other, and have no stem.  The glyph
    models have no class for them, so they are often read as notes.
    """
    space = staff.space
    middle = (staff.top + staff.bottom) / 2.0
    tol = 0.45 * space

    def digit(g, top, bottom):
        return (abs(g.y0 - top) <= tol and abs(g.y0 + g.h - 1 - bottom) <= tol
                and 0.6 * space <= g.w <= 1.9 * space)

    own = [g for g in glyphs if g.staff == staff.index]
    uppers = [g for g in own if digit(g, staff.top, middle)]
    lowers = [g for g in own if digit(g, middle, staff.bottom)]
    out = set()
    # Both digits joined into one glyph through the kept middle line: it
    # fills the staff exactly and, unlike a chord, has no stem.
    from . import notes as notes_module  # local import avoids a cycle
    for g in own:
        if digit(g, staff.top, staff.bottom):
            stem = notes_module._longest_vertical(g.ink)
            if stem is None or stem[2] - stem[1] + 1 < 2.6 * space:
                out.add(g.uid)
    for a in uppers:
        for b in lowers:
            if abs(a.cx - b.cx) <= 0.5 * space and a.uid != b.uid:
                out.update((a.uid, b.uid))
    return out


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
    overlapping = [d for d in down
                   if any(abs(d.x - u.x) <= tolerance and d.measure == u.measure
                          for u in up)]
    if len(overlapping) < min_pairs:
        return None
    # Only measures where opposite stems really sound together are split;
    # elsewhere the staff is one line whose stems follow the note height,
    # and splitting it would leave each half full of gaps.
    split = {d.measure for d in overlapping}
    in_split = [e for e in notes if e.measure in split]
    known = [e for e in in_split if e.direction in ("up", "down")]
    n_up = sum(1 for e in known if e.direction == "up")
    # The main voice is the larger group; number it first so the melody
    # stays voice 1 even when its stems point down.
    main = "up" if n_up >= len(known) - n_up else "down"
    for event in events:
        if event.kind != "note" or event.measure not in split:
            event.voice = 0
        elif event.direction in ("up", "down"):
            event.voice = 0 if event.direction == main else 1
        else:
            # A head cut by staff removal has no stem left; give it to the
            # voice whose notes are closest in the staff.
            nearest = min(
                known,
                key=lambda n: (n.x - event.x) ** 2
                + (0.5 * (n.hy - event.hy)) ** 2,
            )
            event.voice = 0 if nearest.direction == main else 1
    voices = ([e for e in events if e.voice == 0],
              [e for e in events if e.voice == 1])
    if not voices[0] or not voices[1]:
        for event in events:
            event.voice = 0
        return None
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
                         chord_tol: float = 0.0,
                         sequence: bool = False,
                         proportional: bool = False,
                         metric_snap: bool = False) -> list[Event]:
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
    widths = sorted(x1 - x0 for x0, x1 in bounds if x1 > x0)
    ppq = (widths[len(widths) // 2] / float(bar_length)
           if widths and bar_length > 0 else None)
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
        if sequence and chord_tol > 0:
            placed = _sequence_onsets(notes, rests, bar_length, chord_tol)
            if placed is not None:
                out.extend(placed)
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
        if proportional and ppq and not trail:
            # A sparse measure whose closing rest was missed: the last note
            # does not end the bar, so place it by the system's typical
            # horizontal scale instead (engraved spacing is close to
            # proportional when there are only a few notes).
            groups_x = []
            for e in notes:
                if not groups_x or e.x - groups_x[-1] > max(5.0, chord_tol):
                    groups_x.append(e.x)
            if len(groups_x) <= 3:
                raw = first + Fraction((x_last - x_first) / ppq
                                       ).limit_denominator(64)
                coarse = Fraction(round(raw * 2), 2)
                estimate = (coarse if abs(raw - coarse) <= Fraction(3, 20)
                            else Fraction(round(raw / grid)) * grid)
                # A last note read too long (an open head, a stray dot)
                # leaves no room at all; the estimate still places it.
                if first < estimate and (estimate < last or last <= first):
                    last = estimate

        onsets: list[Fraction] = []
        if anchor and x_last > x_first and last > first:
            span = float(x_last - x_first)
            for event in notes:
                frac = Fraction((event.x - x_first) / span).limit_denominator(64)
                raw = first + frac * (last - first)
                onsets.append(_snap_onset(raw, grid) if metric_snap
                              else Fraction(round(raw / grid)) * grid)
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


def _snap_onset(raw: Fraction, grid: Fraction) -> Fraction:
    """Round an interpolated onset, preferring the stronger beat positions.

    Engraved spacing is only roughly proportional, so an onset that lands
    near an eighth position is taken as that eighth rather than the nearest
    sixteenth or thirty-second.
    """
    level = Fraction(1, 2)
    while level > grid:
        candidate = Fraction(round(raw / level)) * level
        if abs(raw - candidate) <= 0.3 * level:
            return candidate
        level /= 2
    return Fraction(round(raw / grid)) * grid


def _sequence_onsets(notes: list[Event], rests: list[Event],
                     bar_length: Fraction,
                     chord_tol: float) -> list[Event] | None:
    """Onsets as running sums of the read durations, when they fill the bar.

    When the chords and rests of one voice add up to exactly one measure, the
    durations themselves place every note; x interpolation is only needed
    when something was missed.  Returns None when the sum does not match.
    """
    items: list[list[Event]] = []
    for e in sorted(notes + rests, key=lambda e: e.x):
        if (e.kind == "note" and items and items[-1][0].kind == "note"
                and abs(e.x - items[-1][-1].x) <= chord_tol):
            items[-1].append(e)
        else:
            items.append([e])
    durations = []
    for group in items:
        values = [e.duration for e in group if e.duration and e.duration > 0]
        if not values:
            return None
        durations.append(max(values))
    if sum(durations, Fraction(0)) != bar_length:
        return None
    out = []
    onset = Fraction(0)
    for group, duration in zip(items, durations):
        for e in group:
            e.onset = onset
            e.duration = duration
            out.append(e)
        onset += duration
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
