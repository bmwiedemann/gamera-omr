#!/bin/bash
# Build gamera-4 + MusicStaves (Python 3.13 port) into ./.venv and install gamera-omr.
# Run inside 'distrobox enter tumbleweed' (uv, gcc, libpng/libvips headers are there).
set -euo pipefail
cd "$(dirname "$0")"

PIN=8f625ffa7f9115d8971a4db22b7d386348592fd1
VENDOR=vendor/gamera4-rodan

if [ ! -d "$VENDOR/.git" ]; then
  mkdir -p vendor
  git clone https://github.com/DDMAL/gamera4-rodan.git "$VENDOR"
  git -C "$VENDOR" checkout "$PIN"
  git -C "$VENDOR" am third_party/patches/*.patch
fi

if [ ! -x .venv/bin/python ]; then
  uv venv --python 3.13 .venv
fi
uv pip install --python .venv/bin/python "setuptools<80" wheel numpy Pillow \
  opencv-python-headless music21 lxml pytest PyYAML

# Gamera
( cd "$VENDOR/gamera-4" && \
  ../../../.venv/bin/python setup.py --nowx build_ext --inplace && \
  ../../../.venv/bin/python setup.py --nowx install )

# MusicStaves needs Gamera's headers; the system include dir is read-only, so use
# CPLUS_INCLUDE_PATH instead of copying into $PREFIX/include.
( cd "$VENDOR/musicstaves" && \
  CPLUS_INCLUDE_PATH="$PWD/../gamera-4/include" ../../../.venv/bin/python setup.py install )

# MusicStaves installs as an isolated egg; merge it into gamera/toolkits.
GSP=$(.venv/bin/python -c "import gamera,os;print(os.path.dirname(gamera.__file__))")
cp -r .venv/lib/python3.13/site-packages/musicstaves-*.egg/gamera/toolkits/musicstaves "$GSP/toolkits/"

# The engine itself (editable)
uv pip install --python .venv/bin/python -e .

.venv/bin/python -W ignore -c "
import gamera
from gamera.toolkits.musicstaves.stafffinder_projections import StaffFinder_projections
from gamera.toolkits.musicstaves.stafffinder_miyao import StaffFinder_miyao
import gamera.knn, gamera.classify, gamera.gamera_xml
print('gamera', gamera.__version__, 'ok')"
