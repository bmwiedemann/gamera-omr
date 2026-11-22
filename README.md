# gamera-omr

An optical music recognition engine for the BWV 248 project, built on
**Gamera 4 + the MusicStaves toolkit** (the recognition core that Rodan wraps).

Status: **Stage 1 + first reader** — Python 3.13 port, per-system skew/curvature
correction, grand-staff (piano) reading, staff finding/validation, staff-line
removal, connected-component segmentation, barline/measure detection, a kNN glyph
classifier, note/beam/chord decoding, MusicXML output, and debug overlays.

## Rest detection (Rösler/Sibley first)

The 1900s vocal scores are rest-heavy, so rests are handled explicitly:

- a measure with no detected notes becomes one **full-measure rest**;
- classifier rest classes (`rest.8`, `rest.quarter`, ...) become rest events;
- for Rösler/Sibley the **printed x position** of each notehead sets the onset;
  the remaining gaps become rests.  Notes are capped by their glyph duration, so a
  quarter followed by a rest yields a quarter plus a rest, not a long note;
- rest and note durations are split into standard note values (e.g. 5/4 = 1 + 1/4,
  with ties) so MusicXML export always succeeds.

The old "centred printed bar means tacet" override is still available as
`--tacet-bars`, but it is **off by default**: on the checked pages the bar detector
also matches beams and accidentals, and dropping notes there lowered F1 on both
Rösler and Sibley.  Empty measures and classifier rests already cover the reliable
cases.  Current readings are in "Per-system deskewing" below; NBA output is unchanged
at note F1 0.95 / pitch F1 0.98.

## Per-font models

`read`/`report` pick `models/<font>/model.xml` automatically (`bga` covers `bga2`), so
the older editions have font-specific models rather than the NBA one:

| source | model | training data | page-1 note / pitch F1 |
|---|---|---|---|
| Rösler | `models/rosler/model.xml` | raw soprano p1 | 0.38 / 0.74 |
| Sibley | `models/sibley/model.xml` | deskewed S-02/S-03 (S-01 held out) | 0.33 / 0.58 |
| Sibley, old model | `models/sibley/model-raw.xml` | raw soprano p1 | 0.25 / 0.46 |
| BGA, dense pages | `models/bga/model.xml` | raw BGA pages | 0.31 / 0.63 |
| BGA, rest-heavy pages | `models/bga/model-deskew.xml` | deskewed BGA S-02–S-05 | 0.55 / 0.55 |

These are current-pipeline numbers, so they include the per-system deskewing and, for
Sibley, the notehead repair below.  The Sibley model was retrained on deskewed pages
because the old one, trained on raw glyphs, classified heading text and beam fragments
as notes; the old model is kept for A/B (`--model models/sibley/model-raw.xml`).
The Sibley figure is held out (trained on S-02/S-03).  Page S-04, which is in no
training split, scores 0.19 / 0.43 (old model 0.19 / 0.47): the note gains generalise
better than the pitch gains, and page 4 remains harder.  BGA keeps the raw-trained
model for dense pages and switches to `model-deskew.xml` when the page's mean glyph
confidence is below 0.85 (the rest-heavy late entries); an explicit `--model` wins.  The class sets include
`rest.8`/`rest.quarter`, `beam.group`, `accidental`, `text`, `beam`, `slur` and `skip`.

## Per-system deskewing

Each stack page is a vertical concatenation of separately cropped systems, so skew is
per system.  Measured on the sample stacks (edge deviation at the staff ends):

| source | max angle | max edge deviation |
|---|---|---|
| NBA | 0.11° | 0.29 staff spaces |
| BGA | 0.28° | 0.44 |
| Rösler | 0.35° | 0.76 |
| Sibley | 0.89° | 1.88 (plus a 3–5 px bow) |

A single page rotation is therefore wrong, and several systems are bowed rather than
merely rotated.  The engine now:

1. finds one row band per system (blank-row segmentation, or staff hints snapped to
   blank rows);
2. fits the five-line staff centre trajectory `y(x)` in each band with a robust
   shared-shape model (independent line intercepts, common slope, optional quadratic);
   the five-line comb is detected over the full band, so a biased row-projection
   finder line is corrected rather than trusted;
3. applies an **integer vertical shear** `y' = y - round(y(x))` per column.  X is
   unchanged, so measures, x-rhythm and component positions stay put; vertical
   barlines and stems stay vertical; no resampling is needed;
4. re-fits the corrected page and accepts only when the residual or edge deviation
   improves and the staff geometry still validates.  If the raw page cannot be
   staff-found at all (Sibley S-02), coarse projection shears bootstrap staff
   finding, then line trajectories are re-fitted.

The shear can be disabled with `--no-deskew`, which is also the A/B baseline.  Skew
diagnostics are reported under `deskew` (per band: angle, edge deviation in px and
staff spaces, curvature, residual, model, shifts, applied/reason), and `debug` writes
`00-deskew.png`.  Small NBA skew is deliberately a no-op.

Current soprano page results (truth = the checked soprano part over the page range):

| page | notes truth/omr | note F1 raw → deskew | pitch F1 raw → deskew |
|---|---|---|---|
| Rösler p1 (bars 33–70) | 105 / 71 → 105 / 84 | 0.250 → **0.381** | 0.534 → **0.741** |
| Sibley p1 | 74 / 60 → 74 / 98 | 0.240 → **0.326** | 0.416 → **0.581** |
| BGA p1 (bars 1–39, mostly rests) | 15 / 15 → 15 / 14 | 0.333 → **0.552** | 0.600 → 0.552 |
| NBA mvt 5 (no-op) | 45 / 43 → 45 / 43 | 0.9545 → 0.9545 | 0.9773 → 0.9773 |

Sibley S-02 now yields six validated staves where the raw finder failed, and the
retrained model plus notehead repair (below) raises Sibley page-1 pitch F1 from 0.42
to 0.58.  Rösler uses flush barlines and note-anchored onsets; its page spans bars
33–70 (38 measures), not the 29-measure window assumed earlier.  The remaining
Sibley/Rösler errors are mostly rhythm/onset placement and measure-level differences,
not skew or glyph identity.

When a notehead sits on a staff line, staff removal cuts it in half and shifts the
detected centre by ~a quarter space (often one pitch step).  `remove_staff_lines`
can rejoin the halves where both sides are wide enough to be a notehead.  This is on
for Sibley (and via `--repair-crossings`) and off for the small Rösler staves and the
clean NBA engravings, where it slightly reduced F1.

Remaining bottlenecks on these pages: Rösler still misses about a fifth of the sung
notes, and symbol-level rest/accidental and flagged-duration semantics still need
work.  The flush-barline and note-anchored onset paths are specific to Rösler; the
Sibley and NBA prints keep the strict/measure-fraction paths, which score better
there.

## Piano grand staffs

Per-system piano crops (`work/piano_all/p-*-sN.png`, `work/mvt/*/piano/*.png`) are
read with `--source piano`:

```bash
.venv/bin/python -m gamera_omr.cli read \
    --image ~/git/music/work/piano_all/p-001-s0.png \
    --source piano --voice P --movement 1 -o /tmp/piano.musicxml
```

- The two staves are paired into one system and share measure boundaries; the
  upper staff uses G2 and the lower F4, and the result is a two-part score
  (`Piano RH`, `Piano LH`).
- A `models/piano/model.xml` font model (trained on deskewed piano systems) is
  selected automatically.  Staff removal still runs on RGB/greyscale crops;
  `StackPage.binary()` now converts RGB piano scans through `to_onebit()` instead
  of calling Otsu on RGB.
- **Long stems:** beam groups are split with a distance-transform head detector,
  not only template matching.  Filled noteheads remain thick blobs while long
  stems and beams are thin; several heads on one long stem become a chord.  The
  old `w <= 0.6*space` stem filter is relaxed to an aspect-ratio test so long
  stems are not discarded.
- Simultaneous heads within `0.6*space` share an onset and duration, and the
  MusicXML writer emits them as chords.

Current status: `p-001-s0` reads 8 measures with 29 notes and 4 chords (the raw
page previously produced 1 note).  The homr reference for the same system has 44
notes and 14 chords, so piano reading is usable but still less complete.  Known
gaps: clef changes inside a system are not followed, voices are approximated by
simultaneous onsets, and grand-staff deskewing is not yet implemented.

## Current results

Scored against the checked NBA-derived readings (`partitur/BWV248/gemeinfrei/...`) with
`sheetscan/bench/evalscore.py`:

| page | source / font | truth notes | omr notes | note F1 | pitch F1 |
|---|---|---|---|---|---|
| Nr. 5 Choral, soprano | NBA, 1960 | 45 | 43 | **0.95** | **0.98** |
| Nr. 5 Choral, soprano | BGA, 1856 | 45 | 38 | 0.31 | 0.63 |
| Nr. 12 Choral, soprano | NBA, 1960 | 49 | 46 | 0.44 | 0.80 |
| Nr. 1, soprano page 1 | BGA, 1856 | 15 | 14 | 0.55 | 0.55 |
| Nr. 1, soprano page 1 | Rösler, c.1900 | 105 | 84 | 0.38 | 0.74 |
| Nr. 1, soprano page 1 | Sibley, c.1900 | 74 | 98 | 0.33 | 0.58 |

The 1900s scores are far more variable than NBA.  Rest placement is now structural
(empty measures become rests, printed rests are classified, gaps become rests), and
the remaining errors are mostly rhythm/onset quantization, in-measure accidentals,
and measure-level differences between the editions.  The optional printed-rest-bar
tacet override (`--tacet-bars`) remains available but is off by default.
Known gaps (v1): in-measure accidentals, ties/slurs/fermatas, beamed-group rhythm on
dense pages, perspective or more than one staff per system, and exact onset placement
on the 1900s prints.

## Reliability tooling

- Staff validation is relative to staff space (`height ≈ 4·space + thickness`), not an
  absolute pixel range; this is what makes the small Rösler staves (13–14.5 px space,
  52–54 px height) pass. Spacing CV tolerance is 0.15.
- A model is chosen automatically per edition: `models/<font>/model.xml` if present,
  otherwise `models/common/model.xml` (`bga` covers `bga2`).
- `report` prints per-page metrics and review flags:
  ```bash
  .venv/bin/python -m gamera_omr.cli report \
      --image ~/git/music/work/stacks/rosler/S-01.png \
      --source rosler --voice S --movement 1
  ```
  It includes the staff backend/space/oracle MAE, the per-system `deskew` block
  (angle, edge deviation, curvature, residual, applied/rejected bands), glyph class
  counts, mean and low-confidence glyphs, text/segmentation drops, measure-fill
  mismatches, empty measures, and flags such as `low_mean_confidence`,
  `many_measure_fill_mismatches`, `deskew_rejected` and
  `deskew_high_residual`.  `oracle_mae` is measured on the raw page and labelled
  via `oracle_frame`.
- The event band excludes text above the staff (instrument names) and a tall/narrow
  clef zone at each system start; the old prints otherwise turn those into phantom
  notes.
- Older prints use a class-primary quarter/half decision (`legacy=True`), because their
  open noteheads often have an inked counter; modern NBA prints use the ink-ratio rule.

## Why

Audiveris and homr are already used for the BWV 248 scans. This project is the third,
independent reading, so that disputed bars can be checked against two different engines.
It is not meant to replace homr.

## Python 3.13 port

`vendor/gamera4-rodan` is pinned to `8f625ffa` and ported to Python 3.13 on branch
`py313`; the five commits are exported to `third_party/patches/` and applied by
`build.sh`. See `third_party/PINS.md`. In short:

- `imp` → `importlib.util` (removed in 3.12)
- old buffer API → `PyObject_GetBuffer` via `include/pycompat.hpp` (removed in 3.13)
- legacy `io.py`: binary `np.fromstring`/`tostring()` → `frombuffer`/`tobytes()`
  (removed in numpy 2); the never-existing `np.complex32` → `np.complex64`
- `ndarray.tostring()` → `tobytes()` (removed in numpy 2)
- MusicStaves finders (projections, Miyao, Dalitz): Python 3 integer division/slicing
  and dict methods

`StaffFinder_dalitz` is not usable (its C++ plugin is absent from the ported tree);
`StaffFinder_projections` and `StaffFinder_miyao` work and are what the engine uses.

## Build

```bash
distrobox enter tumbleweed
./build.sh
```

This creates `.venv` (Python 3.13), builds Gamera and MusicStaves, installs the engine
editable, and runs a smoke test.

## Usage

```bash
# check a stack: build, staff finding, validation, oracle comparison
.venv/bin/python -m gamera_omr.cli doctor \
    --image ~/git/music/work/mvt/5/stacks/nba/S-01.png

# write 01-binary.png, 02-clean.png (staffs removed) and 03-overlay.png
.venv/bin/python -m gamera_omr.cli debug \
    --image ~/git/music/work/mvt/5/stacks/nba/S-01.png --out /tmp/debug5

# read a page to MusicXML
.venv/bin/python -m gamera_omr.cli read \
    --image ~/git/music/work/mvt/5/stacks/nba/S-01.png \
    --source nba --voice S --movement 5 -o /tmp/S5.musicxml

# compare without the per-system skew correction
.venv/bin/python -m gamera_omr.cli read ... --no-deskew -o /tmp/S5-raw.musicxml

# restore the old printed-rest-bar tacet override (off by default)
.venv/bin/python -m gamera_omr.cli read ... --tacet-bars -o /tmp/S5-tacet.musicxml

# notehead repair is automatic for Sibley; force it for another noisy print
.venv/bin/python -m gamera_omr.cli read ... --repair-crossings -o /tmp/old.musicxml

# piano grand-staff crop (two staves, two output parts)
.venv/bin/python -m gamera_omr.cli read \
    --image ~/git/music/work/piano_all/p-001-s0.png \
    --source piano --voice P --movement 1 -o /tmp/piano.musicxml

# rebuilding the glyph model (headless labelling)
.venv/bin/python -m gamera_omr.cli corpus --image IMAGE.png --out /tmp/corpus
.venv/bin/python -m gamera_omr.cli sheet --corpus /tmp/corpus \
    --clusters 40 --out /tmp/corpus/s0
# ... view /tmp/corpus/s0/sheet.png, write cluster labels, then:
.venv/bin/python -m gamera_omr.cli train --corpus /tmp/corpus \
    --labels models/common/labels.json --sheet /tmp/corpus/s0/sheet.json \
    --out models/common/model.xml
```

The input stacks are the existing per-voice stack pages from the BWV 248 pipeline
(`work/stacks/<src>/<V>-NN.png` for movement 1,
`work/mvt/<NR>/stacks/<src>/<V>-NN.png` otherwise). The engine reuses
`partitur/BWV248/gemeinfrei/scripts/{staves,prepare_mvt,crop,stack,norm,merge}.py`
rather than duplicating the PDF→stack front end or the normalization/vote back end.

## Tests

```bash
.venv/bin/python -m pytest -q
```

## License

GPL-2.0-or-later (Gamera and MusicStaves are GPL; see `third_party/PINS.md`).
