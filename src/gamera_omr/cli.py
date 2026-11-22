"""Command line interface: doctor, debug, and (later) read/bench."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

from music21 import meter

from . import classify, config, decode, deskew, glyphs, musicxml, notes, segments
from .staffs import remove_staff_lines


def _flush_barlines(source) -> bool:
    # Rösler draws its barlines flush with the staff; the Sibley/NBA prints
    # have normal or extended barlines and lose measures with this relaxation.
    return source == "rosler"


def _prepare(args):
    source = getattr(args, "source", None)
    # Piano systems are grand staves: the per-system band model does not
    # apply yet, so they use the original image and per-staff clefs below.
    enabled = not args.no_deskew and source != "piano"
    return deskew.prepare_page(args.image, backend=args.backend,
                               enabled=enabled,
                               flush_barlines=_flush_barlines(source))


def _load_predictions(corpus: Path, args, source: str):
    """Classify a corpus, with a low-confidence BGA fallback model.

    The raw-trained BGA model handles the dense chorale pages best, but the
    deskewed-page model is much better on the rest-heavy pages where the old
    model's confidence collapses (e.g. the soprano's late entrance).
    """
    model_path = Path(args.model) if args.model else default_model(source)
    predictions = classify.predict(classify.load_model(model_path), corpus)
    if not args.model and source.startswith("bga"):
        scores = [p["confidence"] for p in predictions.values()]
        if scores and sum(scores) / len(scores) < 0.85:
            alternative = Path("models/bga/model-deskew.xml")
            if alternative.exists():
                model_path = alternative
                predictions = classify.predict(
                    classify.load_model(alternative), corpus)
    return predictions, model_path


def _gamera_version() -> str:
    try:
        import gamera  # noqa: PLC0415

        return getattr(gamera, "__version__", "unknown")
    except Exception as exc:  # pragma: no cover - diagnostic only
        return f"unavailable ({exc})"


def _detect_piano_clefs(comps, staff, starts, initial):
    """Find clef glyphs at system/measure starts and whether they are F4.

    A clef is a glyph roughly 2.2-4.2 staff spaces tall and 0.8-2.2 wide.
    A bass clef is recognised by its two dots to the right; otherwise the
    clef is treble (G2).  Returns (x, clef) changes from `initial`.
    """
    space = staff.space
    changes = []
    current = initial
    for x0 in starts:
        candidates = [
            c for c in comps
            if (x0 - 0.5 * space <= c.cx <= x0 + 5.5 * space
                and staff.top - 0.5 * space <= c.cy <= staff.bottom + 0.5 * space
                and 2.2 * space <= c.h <= 4.2 * space
                and 0.8 * space <= c.w <= 2.2 * space
                and c.area >= 0.25 * c.w * c.h)
        ]
        if not candidates:
            continue
        clef = min(candidates, key=lambda c: abs(c.cx - x0))
        dots = [
            d for d in comps
            if (clef.cx + 0.2 * space <= d.cx <= clef.cx + 3.0 * space
                and staff.top - 0.6 * space <= d.cy <= staff.bottom + 0.6 * space
                and 6 <= d.area <= 140
                and 2 <= d.w <= 0.45 * space and 2 <= d.h <= 0.75 * space
                and d.solidity >= 0.45)
        ]
        bass = any(
            0.2 * space <= abs(a.cy - b.cy) <= 1.5 * space
            and abs(a.cx - b.cx) <= 0.8 * space
            for i, a in enumerate(dots) for b in dots[i + 1:]
        )
        detected = "F4" if bass else "G2"
        if detected != current:
            changes.append((clef.cx, detected))
            current = detected
    return changes


def _decode_piano(staves, items, predictions, dots, fifths, bar, arr, cleaned):
    """Decode a grand-staff system into right/left-hand event streams.

    The two staves share measure boundaries; each staff gets its own clef.
    Notes carry their stem direction so independent voices can be separated
    afterwards; the clef changes are returned as measure -> clef maps for the
    MusicXML writer.
    """
    rh, lh, bounds_all = [], [], []
    rh_clefs, lh_clefs = {}, {}
    comps = segments.components(cleaned.image)
    offset = 0
    pairs = [(staves[i], staves[i + 1] if i + 1 < len(staves) else None)
             for i in range(0, len(staves), 2)]
    for upper, lower in pairs:
        if lower is not None:
            bars = segments.find_grand_barlines(arr, upper, lower)
            bounds = (segments.measure_bounds(upper, bars) if bars
                      else upper.bounds or segments.measure_bounds(
                          upper, segments.find_barlines(arr, upper)))
        else:
            bounds = upper.bounds or segments.measure_bounds(
                upper, segments.find_barlines(arr, upper))
        if lower is not None and len(bounds) <= 1:
            alt = lower.bounds or segments.measure_bounds(
                lower, segments.find_barlines(arr, lower))
            if len(alt) > len(bounds):
                bounds = alt
        starts = [upper.left] + [a for a, _b in bounds[1:]]
        upper_changes = _detect_piano_clefs(comps, upper, starts, "G2")
        lower_changes = (_detect_piano_clefs(comps, lower, starts, "F4")
                         if lower is not None else [])

        def measure_at(x):
            for i, (x0, x1) in enumerate(bounds):
                if x0 <= x < x1:
                    return i
            return min(range(len(bounds)),
                       key=lambda i: abs(bounds[i][0] - x)) if bounds else 0

        for staff, clef, changes, target, clef_map in (
                (upper, "G2", upper_changes, rh, rh_clefs),
                (lower, "F4", lower_changes, lh, lh_clefs)):
            if staff is None:
                continue
            ev = decode.decode_staff(
                staff, items, predictions, bounds, clef, fifths, dots=dots,
                legacy=True, bar_length=bar, require_head=False,
                recover_edges=False, clef_changes=changes, voices=True)
            for event in ev:
                event.measure += offset
            target.extend(ev)
            for x, changed in changes:
                clef_map[offset + measure_at(x)] = changed
        bounds_all.extend(bounds)
        offset += len(bounds)
    return rh, lh, bounds_all, offset, rh_clefs, lh_clefs


def _assign_piano_voices(part, bounds_all, bar, space, offset):
    """Give a piano staff its independent voices their own rhythms.

    The staff is split by stem direction only when opposite-direction notes
    sound together; each voice is then fitted to the printed x positions on
    its own, so an accompaniment chord no longer merges into the melody's
    onset and each voice can carry a rest where the other has notes.
    """
    groups = decode.split_staff_voices(part, space)
    if groups is None:
        groups = [part]
    # The piano reductions use 32nd notes; 1/8 of a quarter keeps their
    # printed x positions distinct instead of collapsing a run into a chord.
    grid = Fraction(1, 8)
    out = []
    for voice, events in enumerate(groups):
        events = decode.normalize_measures(events, offset, bar)
        events = decode.assign_onsets_from_x(
            events, bounds_all, bar, grid=grid, anchor=True,
            chord_tol=0.6 * space)
        for event in events:
            event.voice = voice
        out.extend(events)
    return out


def cmd_doctor(args) -> int:
    print(f"gamera {_gamera_version()}")
    page, staves, diag = _prepare(args)
    print(f"page {page.path} {page.ncols}x{page.nrows} slots={len(page.slots)}")
    print(json.dumps(diag, indent=2))
    for staff in staves:
        print(f"staff {staff.index}: lines={[round(y, 1) for y in staff.lines]} "
              f"space={staff.space:.1f} thickness={staff.thickness:.1f} "
              f"x={staff.left:.0f}..{staff.right:.0f} backend={staff.backend}")
    return 0


def cmd_debug(args) -> int:
    page, staves, diag = _prepare(args)
    out = Path(args.out) if args.out else page.path.parent / f"{page.path.stem}-debug"
    out.mkdir(parents=True, exist_ok=True)

    binary = page.array()
    if diag.get("deskew", {}).get("applied"):
        segments.save_mask(binary, out / "00-deskew.png")
    segments.save_mask(binary, out / "01-binary.png")

    cleaned = remove_staff_lines(page, staves)
    cleaned_arr = cleaned.image.to_numpy()
    segments.save_mask(cleaned_arr, out / "02-clean.png")

    comps = segments.components(cleaned.image)
    kept_all = []
    barlines_by_staff = {}
    for staff in staves:
        kept, dropped = segments.filter_components(comps, staff)
        kept_all.extend(kept)
        barlines = segments.find_barlines(binary, staff)
        barlines_by_staff[staff.index] = barlines
        bounds = staff.bounds or segments.measure_bounds(staff, barlines)
        print(f"staff {staff.index}: {len(kept)} components {dropped} "
              f"barlines={len(barlines)} measures={len(bounds)} "
              f"space={staff.space:.1f}")

    segments.save_overlay(page.path, staves, barlines_by_staff, kept_all,
                          out / "03-overlay.png", image=binary)
    print(f"backend={diag['backend']} wrote {out}")
    return 0


def cmd_corpus(args) -> int:
    page, staves, diag = _prepare(args)
    items = glyphs.extract(page, staves,
                           repair=getattr(args, "repair_crossings", False))
    out = Path(args.out)
    glyphs.save_corpus(items, out)
    print(f"{len(items)} glyphs from {page.path.name} "
          f"({len(staves)} systems, backend {diag['backend']}) -> {out}")
    return 0


def cmd_sheet(args) -> int:
    corpus_dir = Path(args.corpus)
    items = glyphs.load_corpus(corpus_dir)
    out = Path(args.out) if args.out else corpus_dir / "sheets"
    labels, _centers = glyphs.cluster(items, k=args.clusters)
    mapping = glyphs.contact_sheet(items, labels, out / "sheet.png",
                                   out / "sheet.json", per_cluster=args.per,
                                   max_clusters=args.max_clusters,
                                   start=args.start)
    print(f"{len(items)} glyphs, {len(mapping)} clusters -> {out/'sheet.png'}")
    return 0


def cmd_notes(args) -> int:
    page, staves, diag = _prepare(args)
    out = Path(args.out) if args.out else page.path.parent / f"{page.path.stem}-notes"
    out.mkdir(parents=True, exist_ok=True)
    arr = page.array()
    for staff in staves:
        masks = notes.detect(arr, staff)
        notes.overlay(page.path, staff, masks, out / f"notes-s{staff.index}.png",
                      image=arr)
        durations = " ".join(str(n.duration) for n in masks.notes)
        print(f"staff {staff.index}: {len(masks.notes)} heads  durations: {durations}")
    print(f"wrote {out}")
    return 0


def cmd_train(args) -> int:
    _clf, used = classify.train(Path(args.corpus), Path(args.labels),
                                [Path(p) for p in args.sheet], Path(args.out))
    print(f"trained {sum(used.values())} glyphs {used} -> {args.out}")
    return 0


def cmd_classify(args) -> int:
    model = classify.load_model(Path(args.model))
    preds = classify.predict(model, Path(args.corpus))
    counts = {}
    for p in preds.values():
        counts[p["label"]] = counts.get(p["label"], 0) + 1
    out = Path(args.out) if args.out else Path(args.corpus) / "predictions.json"
    out.write_text(json.dumps(preds, indent=1), encoding="utf-8")
    print(f"{counts} -> {out}")
    return 0


def default_model(source: str) -> Path:
    """Prefer a model trained for the edition's font, else the common one."""
    base = "bga" if source.startswith("bga") else source
    candidate = Path("models") / base / "model.xml"
    return candidate if candidate.exists() else Path("models/common/model.xml")


def cmd_read(args) -> int:
    page, staves, diag = _prepare(args)
    legacy = not args.source.startswith("nba")
    repair = ((args.source in ("sibley", "piano") or args.repair_crossings)
              and not args.no_repair_crossings)
    items = glyphs.extract(page, staves, repair=repair)
    with tempfile.TemporaryDirectory() as td:
        corpus = Path(td)
        glyphs.save_corpus(items, corpus)
        predictions, _model_path = _load_predictions(corpus, args, args.source)

    piano = args.source == "piano"
    clef = "G2" if piano else config.clef(args.source, args.voice)
    fifths = config.key_fifths(args.movement) or 0
    time_signature = config.time_signature(args.movement)
    arr = page.array()
    forced_x = args.source in ("rosler", "sibley")
    x_rhythm = forced_x
    cleaned = remove_staff_lines(page, staves, repair=repair)
    all_comps = segments.components(cleaned.image)
    dots = []
    if not args.no_dots:
        for staff in staves:
            for c in all_comps:
                if (2 <= c.area <= int(0.35 * staff.space ** 2)
                        and 0.1 * staff.space <= c.w <= 0.5 * staff.space
                        and 0.1 * staff.space <= c.h <= 0.5 * staff.space
                        and staff.top - staff.space <= c.cy <= staff.bottom + staff.space):
                    dots.append((c.cx, c.cy))
    events = []
    offset = 0
    bounds_all = []
    tacet = set()
    bar = Fraction(meter.TimeSignature(time_signature).barDuration.quarterLength)
    if piano:
        rh, lh, bounds_all, offset, rh_clefs, lh_clefs = _decode_piano(
            staves, items, predictions, dots, fifths, bar, arr, cleaned)
        space = sum(s.space for s in staves) / max(1, len(staves))
        rh = _assign_piano_voices(rh, bounds_all, bar, space, offset)
        lh = _assign_piano_voices(lh, bounds_all, bar, space, offset)
        events = rh + lh
        musicxml.write_piano(rh, lh, offset, fifths, time_signature, args.out,
                             rh_clef_changes=rh_clefs,
                             lh_clef_changes=lh_clefs)
        counts = {}
        for event in events:
            counts[event.kind] = counts.get(event.kind, 0) + 1
        print(f"{len(events)} events {counts}, {offset} measures, "
              f"backend {diag['backend']} -> {args.out}")
        fills = decode.measure_fill(rh, offset, bar)
        bad = [(i + 1, str(s)) for i, s in enumerate(fills) if s != bar]
        if bad:
            print(f"measure-fill mismatches ({len(bad)}): {bad[:16]}")
        return 0
    for staff in staves:
        if staff.bounds:
            bounds = staff.bounds
        else:
            barlines = segments.find_barlines(arr, staff, flush=_flush_barlines(args.source))
            bounds = segments.measure_bounds(staff, barlines)
        bounds_all.extend(bounds)
        if x_rhythm and args.tacet_bars:
            for c in segments.find_rest_bars(all_comps, staff):
                for i, (x0, x1) in enumerate(bounds):
                    if (x0 <= c.cx < x1
                            and abs(c.cx - (x0 + x1) / 2) <= 0.25 * (x1 - x0)):
                        tacet.add(offset + i)
        staff_events = decode.decode_staff(staff, items, predictions, bounds,
                                           clef, fifths, dots=dots, legacy=legacy,
                                           bar_length=bar, require_head=x_rhythm,
                                           recover_edges=args.source.startswith("bga"))
        for event in staff_events:
            event.measure += offset
        events.extend(staff_events)
        offset += len(bounds)

    events = decode.normalize_measures(events, offset, bar, tacet=tacet)
    note_measures = len({e.measure for e in events if e.kind == "note"})
    auto_x = (args.source.startswith("bga")
              and note_measures < 0.5 * max(offset, 1))
    if forced_x or auto_x:
        events = decode.assign_onsets_from_x(
            events, bounds_all, bar,
            anchor=args.source in ("rosler", "bga", "bga2"))
    else:
        decode.fit_durations(events, offset, bar)
        events = decode.assign_onsets(events, offset, bar)
    fills = decode.measure_fill(events, offset, bar)
    out = musicxml.write(events, offset, clef, fifths, time_signature,
                         args.out, args.voice)
    counts = {}
    for event in events:
        counts[event.kind] = counts.get(event.kind, 0) + 1
    print(f"{len(events)} events {counts}, {offset} measures, "
          f"backend {diag['backend']} -> {out}")
    bad = [(i + 1, str(s)) for i, s in enumerate(fills) if s != bar]
    if bad:
        print(f"measure-fill mismatches ({len(bad)}): {bad[:16]}")
    return 0


def cmd_report(args) -> int:
    """Per-page reliability metrics; flags pages that need manual review."""
    page, staves, diag = _prepare(args)
    legacy = not args.source.startswith("nba")
    repair = ((args.source in ("sibley", "piano") or args.repair_crossings)
              and not args.no_repair_crossings)
    items = glyphs.extract(page, staves, repair=repair)
    with tempfile.TemporaryDirectory() as td:
        corpus = Path(td)
        glyphs.save_corpus(items, corpus)
        predictions, model_path = _load_predictions(corpus, args, args.source)

    arr = page.array()
    cleaned = remove_staff_lines(page, staves, repair=repair)
    all_comps = segments.components(cleaned.image)
    piano = args.source == "piano"
    clef = "G2" if piano else config.clef(args.source, args.voice)
    fifths = config.key_fifths(args.movement) or 0
    time_signature = config.time_signature(args.movement)
    bar = Fraction(meter.TimeSignature(time_signature).barDuration.quarterLength)
    forced_x = args.source in ("rosler", "sibley")
    x_rhythm = forced_x

    if piano:
        rh, lh, bounds_all, offset, _rh_clefs, _lh_clefs = _decode_piano(
            staves, items, predictions, [], fifths, bar, arr, cleaned)
        space = sum(s.space for s in staves) / max(1, len(staves))
        rh = _assign_piano_voices(rh, bounds_all, bar, space, offset)
        lh = _assign_piano_voices(lh, bounds_all, bar, space, offset)
        events = rh + lh
        drops, barlines, measures = {}, {}, {}
        fills = decode.measure_fill(rh, offset, bar)
    else:
        events: list = []
        drops, barlines, measures = {}, {}, {}
        offset = 0
        bounds_all = []
        tacet = set()
        for staff in staves:
            _kept, dropped = segments.filter_components(all_comps, staff)
            drops[str(staff.index)] = dropped
            bars = segments.find_barlines(arr, staff, flush=_flush_barlines(args.source))
            if staff.bounds:
                bounds = staff.bounds
            else:
                bounds = segments.measure_bounds(staff, bars)
            bounds_all.extend(bounds)
            if x_rhythm and args.tacet_bars:
                for c in segments.find_rest_bars(all_comps, staff):
                    for i, (x0, x1) in enumerate(bounds):
                        if (x0 <= c.cx < x1
                                and abs(c.cx - (x0 + x1) / 2) <= 0.25 * (x1 - x0)):
                            tacet.add(offset + i)
            barlines[str(staff.index)] = len(bars)
            measures[str(staff.index)] = len(bounds)
            staff_events = decode.decode_staff(staff, items, predictions, bounds,
                                               clef, fifths, dots=[],
                                               legacy=legacy, bar_length=bar,
                                               require_head=x_rhythm,
                                               recover_edges=args.source.startswith("bga"))
            for event in staff_events:
                event.measure += offset
            events.extend(staff_events)
            offset += len(bounds)

        events = decode.normalize_measures(events, offset, bar, tacet=tacet)
        note_measures = len({e.measure for e in events if e.kind == "note"})
        auto_x = (args.source.startswith("bga")
                  and note_measures < 0.5 * max(offset, 1))
        if forced_x or auto_x:
            events = decode.assign_onsets_from_x(
                events, bounds_all, bar,
                anchor=args.source in ("rosler", "bga", "bga2"))
        else:
            decode.fit_durations(events, offset, bar)
            events = decode.assign_onsets(events, offset, bar)
        fills = decode.measure_fill(events, offset, bar)
    mismatches = [(i + 1, str(s)) for i, s in enumerate(fills) if s != bar]

    classes, confidences = {}, []
    for prediction in predictions.values():
        label = prediction["label"]
        classes[label] = classes.get(label, 0) + 1
        confidences.append(prediction["confidence"])
    mean_conf = round(sum(confidences) / len(confidences), 3) if confidences else None

    flags = []
    if diag.get("backend") == "oracle":
        flags.append("staff_fallback_oracle")
    if mean_conf is not None and mean_conf < 0.85:
        flags.append("low_mean_confidence")
    if sum(1 for c in confidences if c < 0.75) > max(3, len(confidences) // 10):
        flags.append("low_confidence_glyphs")
    if len(mismatches) > max(2, offset // 4):
        flags.append("many_measure_fill_mismatches")
    if any(s == 0 for s in fills):
        flags.append("empty_measures")
    if not any(k.startswith("note.") for k in classes):
        flags.append("no_notes")
    deskew_diag = diag.get("deskew", {})
    skew_agg = deskew_diag.get("aggregate", {})
    if (deskew_diag.get("enabled") and not deskew_diag.get("applied")
            and deskew_diag.get("reason") not in (None, "below_threshold")):
        flags.append("deskew_rejected")
    if (skew_agg.get("max_residual_rms_px") or 0) > 1.5:
        flags.append("deskew_high_residual")
    if any(reason != "below_threshold"
           for reason in skew_agg.get("reasons", {})):
        flags.append("deskew_band_rejected")

    report = {
        "image": str(page.path), "source": args.source, "voice": args.voice,
        "movement": args.movement,
        "model": str(model_path),
        "staff": {"backend": diag.get("backend"), "count": len(staves),
                  "spaces": [round(s.space, 1) for s in staves],
                  "oracle_mae": diag.get("oracle_mae"),
                  "oracle_frame": diag.get("oracle_frame"),
                  "tried": diag.get("tried")},
        "deskew": deskew_diag,
        "glyphs": {"total": len(items), "classes": classes,
                   "mean_confidence": mean_conf,
                   "low_confidence": sum(1 for c in confidences if c < 0.75)},
        "segmentation": {"drops": drops, "barlines": barlines, "measures": measures},
        "rhythm": {"measures": offset, "fill_mismatches": len(mismatches),
                   "details": mismatches[:20], "empty": sum(1 for s in fills if s == 0)},
        "events": {"notes": sum(1 for e in events if e.kind == "note"),
                   "rests": sum(1 for e in events if e.kind == "rest")},
        "flags": flags,
    }
    print(json.dumps(report, indent=1))
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="gamera-omr",
                                     description="Gamera-based OMR for BWV 248")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("doctor", help="check the build, staff finding and validation")
    p.add_argument("--image", required=True, help="stack PNG")
    p.add_argument("--backend", default="auto", choices=["auto", "projections", "miyao"])
    p.add_argument("--no-deskew", action="store_true",
                   help="do not correct per-system skew before processing")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("debug", help="write staff/removal/segmentation overlays")
    p.add_argument("--image", required=True, help="stack PNG")
    p.add_argument("--out", help="output directory (default: next to the image)")
    p.add_argument("--backend", default="auto", choices=["auto", "projections", "miyao"])
    p.add_argument("--no-deskew", action="store_true",
                   help="do not correct per-system skew before processing")
    p.set_defaults(func=cmd_debug)

    p = sub.add_parser("corpus", help="extract glyphs from a stack page")
    p.add_argument("--image", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--backend", default="auto", choices=["auto", "projections", "miyao"])
    p.add_argument("--no-deskew", action="store_true",
                   help="do not correct per-system skew before processing")
    p.add_argument("--repair-crossings", action="store_true",
                   help="rejoin noteheads split by staff-line removal")
    p.set_defaults(func=cmd_corpus)

    p = sub.add_parser("sheet", help="cluster a corpus and write a contact sheet")
    p.add_argument("--corpus", required=True)
    p.add_argument("--out")
    p.add_argument("--clusters", type=int, default=32)
    p.add_argument("--per", type=int, default=6)
    p.add_argument("--max-clusters", type=int)
    p.add_argument("--start", type=int, default=0)
    p.set_defaults(func=cmd_sheet)

    p = sub.add_parser("notes", help="detect noteheads/stems/beams and overlay them")
    p.add_argument("--image", required=True)
    p.add_argument("--out")
    p.add_argument("--backend", default="auto", choices=["auto", "projections", "miyao"])
    p.add_argument("--no-deskew", action="store_true",
                   help="do not correct per-system skew before processing")
    p.set_defaults(func=cmd_notes)

    p = sub.add_parser("train", help="train the kNN from a corpus and cluster labels")
    p.add_argument("--corpus", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--sheet", action="append", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("classify", help="classify a corpus with a trained model")
    p.add_argument("--corpus", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--out")
    p.set_defaults(func=cmd_classify)

    p = sub.add_parser("report", help="per-page reliability metrics and review flags")
    p.add_argument("--image", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--voice", required=True)
    p.add_argument("--movement", type=int, required=True)
    p.add_argument("--model")
    p.add_argument("--backend", default="auto", choices=["auto", "projections", "miyao"])
    p.add_argument("--no-deskew", action="store_true",
                   help="do not correct per-system skew before processing")
    p.add_argument("--tacet-bars", action="store_true",
                   help="treat a centred printed whole/half-rest bar as tacet")
    p.add_argument("--repair-crossings", action="store_true",
                   help="rejoin noteheads split by staff-line removal")
    p.add_argument("--no-repair-crossings", action="store_true",
                   help="disable the Sibley notehead repair")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("read", help="read a stack into MusicXML (Stage 1)")
    p.add_argument("--image", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--voice", required=True)
    p.add_argument("--movement", type=int, required=True)
    p.add_argument("--model", help="glyph model (default: per-font, else common)")
    p.add_argument("--no-dots", action="store_true",
                   help="do not apply dotted-note detection")
    p.add_argument("--backend", default="auto", choices=["auto", "projections", "miyao"])
    p.add_argument("--no-deskew", action="store_true",
                   help="do not correct per-system skew before processing")
    p.add_argument("--tacet-bars", action="store_true",
                   help="treat a centred printed whole/half-rest bar as tacet")
    p.add_argument("--repair-crossings", action="store_true",
                   help="rejoin noteheads split by staff-line removal")
    p.add_argument("--no-repair-crossings", action="store_true",
                   help="disable the Sibley notehead repair")
    p.add_argument("-o", "--out", required=True)
    p.set_defaults(func=cmd_read)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
