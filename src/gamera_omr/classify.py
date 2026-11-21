"""Gamera kNN training and classification for the glyph corpus."""

from __future__ import annotations

import json
from pathlib import Path

from gamera import knn
from gamera.core import init_gamera, load_image

from .glyphs import load_corpus

FEATURES = [
    "area", "aspect_ratio", "black_area", "moments", "nholes_extended",
    "skeleton_features", "volume64regions", "zernike_moments",
]


def _glyph_image(corpus_dir: Path, uid: str):
    init_gamera()
    return load_image(str(corpus_dir / "glyphs" / f"{uid}.png")).to_onebit()


def read_assignments(sheet_jsons: list[Path]) -> dict[str, int]:
    out = {}
    for path in sheet_jsons:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        out.update({k: int(v) for k, v in data.get("assignments", {}).items()})
    return out


def train(corpus_dir: Path, labels_json: Path, sheet_jsons: list[Path], model_path: Path):
    items = load_corpus(Path(corpus_dir))
    assignments = read_assignments([Path(p) for p in sheet_jsons])
    labels = json.loads(Path(labels_json).read_text(encoding="utf-8"))
    glyphs = []
    used = {}
    for g in items:
        cluster = assignments.get(g.uid)
        if cluster is None:
            continue
        label = labels.get(str(cluster))
        if not label or label == "skip":
            continue
        glyph = _glyph_image(Path(corpus_dir), g.uid)
        glyph.classify_manual(label)
        glyphs.append(glyph)
        used[label] = used.get(label, 0) + 1
    if not glyphs:
        raise SystemExit("no labelled glyphs")
    classifier = knn.kNNNonInteractive(glyphs, features=FEATURES, num_k=1,
                                       normalize=True, perform_splits=False)
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    classifier.to_xml_filename(str(model_path))
    return classifier, used


def load_model(model_path: Path):
    init_gamera()
    return knn.kNNNonInteractive(str(model_path))


def predict(model, corpus_dir: Path) -> dict[str, dict]:
    out = {}
    for g in load_corpus(Path(corpus_dir)):
        glyph = _glyph_image(Path(corpus_dir), g.uid)
        id_name, _confidence = model.guess_glyph_automatic(glyph)
        if not id_name:
            out[g.uid] = {"label": "skip", "confidence": 0.0}
        else:
            conf, name = id_name[0]
            out[g.uid] = {"label": name, "confidence": float(conf)}
    return out
