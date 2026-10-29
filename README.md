# Consumer head trackers for desktop spatial audio: recorder, analysis and data

Code and data for a study that characterises consumer head trackers for desktop spatial audio. To cite them, see
`CITATION.cff`.

The study records three head trackers on one Mac clock in one session: the motion sensor of AirPods Max 2
headphones (CMHeadphoneMotionManager), and MediaPipe Face Landmarker and Apple Vision on the built-in webcam, both fed
the same frames. An iPhone running ARKit face tracking, synchronised over Wi-Fi, is the reference.

## Contents
- `headtrack/mac/`: the Mac recorder (camera, Vision, headphone motion, clock sync, protocol prompts).
- `headtrack/ios/`: the iPhone reference app (ARKit face tracking, UDP to the Mac).
- `headtrack/common/`: the wire format and the shared frame buffer. `src/core/`: clock and JSON helpers.
- `headtrack/protocols/`: the head-movement protocols (`core`, `drift`, `calib`).
- `headtrack/analysis/`: analysis (`htlib.py`), per-session report (`analyze.py`), the paper's numbers, tables and
  figures (`paper_results.py`), and the live MediaPipe tracker the recorder starts (`mp_live.py`).
- `headtrack/paper/sessions.json`: which session plays which role in the paper; `results/`: the paper's numbers
  (`numbers.json`), tables and plotted values; `make_xlsx.py` turns those into Excel workbooks.
- `runs/`: the 12 recorded sessions used in the paper (151 MB). Each folder: `airpods.csv`, `camera.csv` (Vision),
  `mediapipe.csv`, `ref.csv` (iPhone), `sync.csv` (clock-sync pings), `events.csv` (protocol steps), `session.json`,
  logs. Personal details were removed: home paths, charger serial numbers, borrowed devices' Bluetooth names.
  No camera images are included.

## Reproduce the paper's numbers
```
headtrack/setup.sh                    # Python venv with MediaPipe, downloads the face model
headtrack/.venv/bin/python headtrack/analysis/paper_results.py headtrack/paper/sessions.json
```
This rewrites `headtrack/paper/results/` (numbers.json, tables.md, figures). One session alone:
`headtrack/.venv/bin/python headtrack/analysis/analyze.py runs/<session> --plot`.

## Record a session
macOS 14 or later on Apple Silicon, headphones with head tracking, an iPhone with a TrueDepth camera on the same
Wi-Fi. `make recorder`, `make ios ...` (see the Makefile), install and start the iPhone app, then
`headtrack/run_session.sh core --speak on`.

## Licence
Code: MIT (`LICENSE`). Data in `runs/` and `headtrack/paper/results/`: CC BY 4.0 (`LICENSE-data.md`).
How to cite: `CITATION.cff`.
