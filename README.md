# gamera-omr

An optical music recognition engine for the BWV 248 project, built on
**Gamera 4 + the MusicStaves toolkit** (the recognition core that Rodan wraps).

Status: **Stage 1 + first reader** — Python 3.13 port, staff finding/validation,
staff-line removal, connected-component segmentation, barline/measure detection, a
kNN glyph classifier, note/beam decoding, MusicXML output, and debug overlays.

## Rest detection (Rösler/Sibley first)

The 1900s vocal scores are rest-heavy, so rests are now handled before anything else:

- a measure with no detected notes becomes one **full-measure rest**;
- a solid, short, **centred rest bar** (whole/half rest) marks a tacet measure, even
  when the classifier produced false notes there (Rösler/Sibley);
- for Rösler/Sibley the **printed x position** of each notehead sets the onset;
  the remaining gaps become rests.  Notes are capped by their glyph duration, so a
  quarter followed by a rest yields a quarter plus a rest, not a long note;
- rest and note durations are split into standard note values (e.g. 5/4 = 1 + 1/4,
  with ties) so MusicXML export always succeeds.

Measured on the 1900s pages (the first page starts at printed bar 33; truth =
`data/01/voices/S.json` measures 33+):

| page | truth notes | omr notes | rests | measure fills correct | note F1 | pitch F1 |
|---|---|---|---|---|---|---|
| Rösler soprano p1 (bars 33–61) | 74 | 45 | 27 | 27/29 | 0.22 | 0.40 |
| Sibley soprano p1 (bars 33–58) | 65 | 32 | 28 | 23/26 | 0.10 | 0.33 |

Rest placement is now structurally sound (all 29/26 measures carry rests, and nearly
all measures sum to 3/8); the remaining note F1 is limited by the common (NBA) glyph
model on these fonts.  Per-font models and symbol-level quarter/eighth-rest classes are
the next step.  NBA output is unchanged at note F1 0.95 / pitch F1 0.98.

## Current results

Scored against the checked NBA-derived readings (`partitur/BWV248/gemeinfrei/...`) with
`sheetscan/bench/evalscore.py`:

| page | source / font | truth notes | omr notes | note F1 | pitch F1 |
|---|---|---|---|---|---|
| Nr. 5 Choral, soprano | NBA, 1960 | 45 | 43 | **0.95** | **0.98** |
| Nr. 5 Choral, soprano | BGA, 1856 | 45 | 37 | 0.37 | 0.63 |
| Nr. 12 Choral, soprano | NBA, 1960 | 49 | 46 | 0.44 | 0.80 |
| Nr. 1, soprano page 1 | Rösler, c.1900 | 0 (rests) | 72 false | — | — |
| Nr. 1, soprano page 1 | Sibley, c.1900 | 0 (rests) | 60 false | — | — |

The two 1900s vocal-score pages are the instrumental opening where the soprano rests;
all 132 emitted notes are false. **Rest detection is the largest missing class for
1900s vocal scores.**

Known gaps (v1): rests (infrastructure is in place, models need rest labels),
in-measure accidentals, notes whose head is merged with a slur/fermata, beamed-group
rhythm on dense pages, measure boundaries on pages with repeat/double bars, and
per-font models for Rösler/Sibley (only BGA has one so far).

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
  It includes the staff backend/space/oracle MAE, glyph class counts, mean and
  low-confidence glyphs, text/segmentation drops, measure-fill mismatches, empty
  measures, and flags such as `low_mean_confidence`, `many_measure_fill_mismatches`.
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
