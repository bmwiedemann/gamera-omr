# Pinned third-party sources

## Gamera 4 + MusicStaves (Rodan port)

- Repository: <https://github.com/DDMAL/gamera4-rodan>
- Pin: `8f625ffa7f9115d8971a4db22b7d386348592fd1` (`main`, merge of DDMAL/update-python)
- Local branch: `py313`
- Gamera version: 4.0.0 (vendored `gamera-4/version`); upstream v4.1.0 is only a reference.
- MusicStaves version: 1.4.0

### Python 3.13 port commits

Applied as `third_party/patches/000*.patch`:

| commit | subject |
|---|---|
| `9ed9de4` | Port paths to importlib (imp was removed in Python 3.12) |
| `41847bd` | Port kNN and GUI buffer handling to Python 3.13 |
| `766498c` | Use np.complex64 instead of the removed np.complex32 alias |
| `b528fd7` | Port the staff finders to Python 3 integer semantics |
| `c0dc965` | Use ndarray.tobytes() in numpy_io (tostring was removed in numpy 2) |

Build/test status on 2026-11-21 with Python 3.13.14, numpy 2.5.5, gcc 16.2:

- `import gamera`, `gamera.knn`, `gamera.classify`, `gamera.gamera_xml` — ok
- `gamera.toolkits.musicstaves.stafffinder_{projections,miyao}` — ok
- `StaffFinder_projections.find_staves(num_lines=5)` on
  `work/mvt/5/stacks/nba/S-01.png` — 3 staves × 5 lines, ~0.02 s
- `StaffFinder_miyao.find_staves(num_lines=5)` — 3 staves × 5 lines, ~0.02 s
- `StaffFinder_dalitz` — **not usable**: its C++ plugin
  (`Image.get_staff_skeleton_list`) is missing from the ported tree.
- `tests/test_classifier.py` — 2 passed (exercises the buffer-API port)

## Other references

- `gamera-4` upstream v4.1.0 at `~/git/music/gamera-4` — read-only reference, not built.
- `sheetscan/bench/evalscore.py` (AGPL-3.0) — algorithm reference only; the scorer is
  reimplemented.
- Rodan wrappers (`Rodan/.../jobs/gamera_rodan`, AGPL-3.0) — API reference only.
