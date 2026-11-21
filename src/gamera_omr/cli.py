"""Command line interface: doctor, debug, and (later) read/bench."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import config, segments
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


def cmd_read(args) -> int:
    print("read: the MusicXML decoder is not implemented yet "
          "(Stage 1 of the plan: staff removal + segmentation are in place)",
          file=sys.stderr)
    return 2


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

    p = sub.add_parser("read", help="read a stack into MusicXML (Stage 1)")
    p.add_argument("--image", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--voice", required=True)
    p.add_argument("--movement", type=int, required=True)
    p.add_argument("-o", "--out", required=True)
    p.set_defaults(func=cmd_read)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
