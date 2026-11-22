"""Command line interface: doctor, debug, and (later) read/bench."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

from music21 import meter

from . import classify, config, decode, glyphs, musicxml, notes, segments
from .stacks import load_stack
from .staffs import find_staves, remove_staff_lines


def _gamera_version() -> str:
    try:
        import gamera  # noqa: PLC0415

        return getattr(gamera, "__version__", "unknown")
    except Exception as exc:  # pragma: no cover - diagnostic only
        return f"unavailable ({exc})"


def cmd_doctor(args) -> int:
    print(f"gamera {_gamera_version()}")
    page = load_stack(args.image)
    print(f"page {page.path} {page.ncols}x{page.nrows} slots={len(page.slots)}")
    staves, diag = find_staves(page, args.backend)
    print(json.dumps(diag, indent=2))
    for staff in staves:
        print(f"staff {staff.index}: lines={[round(y, 1) for y in staff.lines]} "
              f"space={staff.space:.1f} thickness={staff.thickness:.1f} "
              f"x={staff.left:.0f}..{staff.right:.0f} backend={staff.backend}")
    return 0


def cmd_debug(args) -> int:
    page = load_stack(args.image)
    staves, diag = find_staves(page, args.backend)
    out = Path(args.out) if args.out else page.path.parent / f"{page.path.stem}-debug"
    out.mkdir(parents=True, exist_ok=True)

    binary = page.array()
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
        bounds = segments.measure_bounds(staff, barlines)
        print(f"staff {staff.index}: {len(kept)} components {dropped} "
              f"barlines={len(barlines)} measures={len(bounds)} "
              f"space={staff.space:.1f}")

    segments.save_overlay(page.path, staves, barlines_by_staff, kept_all,
                          out / "03-overlay.png")
    print(f"backend={diag['backend']} wrote {out}")
    return 0


def cmd_corpus(args) -> int:
    page = load_stack(args.image)
    staves, diag = find_staves(page, args.backend)
    items = glyphs.extract(page, staves)
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
    page = load_stack(args.image)
    staves, diag = find_staves(page, args.backend)
    out = Path(args.out) if args.out else page.path.parent / f"{page.path.stem}-notes"
    out.mkdir(parents=True, exist_ok=True)
    arr = page.array()
    for staff in staves:
        masks = notes.detect(arr, staff)
        notes.overlay(page.path, staff, masks, out / f"notes-s{staff.index}.png")
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
    page = load_stack(args.image)
    staves, diag = find_staves(page, args.backend)
    items = glyphs.extract(page, staves)
    with tempfile.TemporaryDirectory() as td:
        corpus = Path(td)
        glyphs.save_corpus(items, corpus)
        model = classify.load_model(args.model and Path(args.model) or default_model(args.source))
        predictions = classify.predict(model, corpus)

    clef = config.clef(args.source, args.voice)
    fifths = config.key_fifths(args.movement) or 0
    time_signature = config.time_signature(args.movement)
    arr = page.array()
    legacy = not args.source.startswith("nba")
    cleaned = remove_staff_lines(page, staves)
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
    bar = Fraction(meter.TimeSignature(time_signature).barDuration.quarterLength)
    for staff in staves:
        barlines = segments.find_barlines(arr, staff)
        bounds = segments.measure_bounds(staff, barlines)
        staff_events = decode.decode_staff(staff, items, predictions, bounds,
                                           clef, fifths, dots=dots, legacy=legacy,
                                           bar_length=bar)
        for event in staff_events:
            event.measure += offset
        events.extend(staff_events)
        offset += len(bounds)

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
    page = load_stack(args.image)
    staves, diag = find_staves(page, args.backend)
    items = glyphs.extract(page, staves)
    with tempfile.TemporaryDirectory() as td:
        corpus = Path(td)
        glyphs.save_corpus(items, corpus)
        model = classify.load_model(Path(args.model) if args.model else default_model(args.source))
        predictions = classify.predict(model, corpus)

    arr = page.array()
    cleaned = remove_staff_lines(page, staves)
    all_comps = segments.components(cleaned.image)
    clef = config.clef(args.source, args.voice)
    fifths = config.key_fifths(args.movement) or 0
    time_signature = config.time_signature(args.movement)
    bar = Fraction(meter.TimeSignature(time_signature).barDuration.quarterLength)
    legacy = not args.source.startswith("nba")

    events: list = []
    drops, barlines, measures = {}, {}, {}
    offset = 0
    for staff in staves:
        _kept, dropped = segments.filter_components(all_comps, staff)
        drops[str(staff.index)] = dropped
        bars = segments.find_barlines(arr, staff)
        bounds = segments.measure_bounds(staff, bars)
        barlines[str(staff.index)] = len(bars)
        measures[str(staff.index)] = len(bounds)
        staff_events = decode.decode_staff(staff, items, predictions, bounds,
                                           clef, fifths, dots=[],
                                           legacy=legacy, bar_length=bar)
        for event in staff_events:
            event.measure += offset
        events.extend(staff_events)
        offset += len(bounds)

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

    report = {
        "image": str(page.path), "source": args.source, "voice": args.voice,
        "movement": args.movement,
        "model": str(Path(args.model) if args.model else default_model(args.source)),
        "staff": {"backend": diag.get("backend"), "count": len(staves),
                  "spaces": [round(s.space, 1) for s in staves],
                  "oracle_mae": diag.get("oracle_mae"), "tried": diag.get("tried")},
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
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("debug", help="write staff/removal/segmentation overlays")
    p.add_argument("--image", required=True, help="stack PNG")
    p.add_argument("--out", help="output directory (default: next to the image)")
    p.add_argument("--backend", default="auto", choices=["auto", "projections", "miyao"])
    p.set_defaults(func=cmd_debug)

    p = sub.add_parser("corpus", help="extract glyphs from a stack page")
    p.add_argument("--image", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--backend", default="auto", choices=["auto", "projections", "miyao"])
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
    p.add_argument("-o", "--out", required=True)
    p.set_defaults(func=cmd_read)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
