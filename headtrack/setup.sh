#!/bin/sh
# One-time setup for the side paper's analysis tools (HEADTRACK_STUDY.md):
# Python venv with pinned packages, and the MediaPipe face landmarker model
# checked against its recorded sha256. Analysis only; nothing here ships.
set -e
cd "$(dirname "$0")"
[ -x .venv/bin/python ] || python3 -m venv .venv
.venv/bin/python -m pip install --quiet --upgrade pip
.venv/bin/python -m pip install --quiet -r analysis/requirements.txt
mkdir -p models
M=models/face_landmarker.task
[ -f $M ] || curl -sSL -o $M \
    https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
echo "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff  $M" | shasum -a 256 -c
