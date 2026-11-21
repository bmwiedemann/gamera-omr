"""Paths and per-movement settings shared with the BWV 248 pipeline.

Everything is read from the existing project files; this module deliberately
does not duplicate the clef/key/voice tables.  `voices.py` is imported from
`partitur/BWV248/gemeinfrei/scripts` (it has no import-time side effects).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

MUSIC = Path.home() / "git" / "music"
SCRIPTS = MUSIC / "partitur" / "BWV248" / "gemeinfrei" / "scripts"
MOVEMENTS = SCRIPTS.parent / "movements.json"
WORK = MUSIC / "work"

_voices = None


def voices_module():
    """Import the project's voices.py lazily."""
    global _voices
    if _voices is None:
        if str(SCRIPTS) not in sys.path:
            sys.path.insert(0, str(SCRIPTS))
        import voices as module  # noqa: PLC0415

        _voices = module
    return _voices


def movements():
    return json.loads(MOVEMENTS.read_text(encoding="utf-8"))


def movement(nr: int) -> dict:
    return movements()[str(int(nr))]


def voice_order(nr: int) -> str:
    return "".join(movement(nr).get("voices", "SATB"))


def clef(source: str, voice: str) -> str:
    table = voices_module().BGA_CLEF if source.startswith("bga") else voices_module().NBA_CLEF
    return table[voice]


def key_fifths(nr: int):
    key = movement(nr).get("key")
    return key if isinstance(key, int) else None


def time_signature(nr: int) -> str:
    return movement(nr)["time"]


def stack_dir(nr: int, source: str) -> Path:
    if int(nr) == 1:
        return WORK / "stacks" / source
    return WORK / "mvt" / str(int(nr)) / "stacks" / source


def stack_list(nr: int, source: str) -> Path:
    return stack_dir(nr, source) / f"{source}.list"


def output_dir(nr: int, source: str) -> Path:
    if int(nr) == 1:
        return WORK / "omr" / "gamera" / source
    return WORK / "mvt" / str(int(nr)) / "omr" / "gamera" / source


def truth_path(nr: int, voice: str) -> Path:
    return SCRIPTS.parent / "data" / f"{int(nr):02d}" / "voices" / f"{voice}.json"
