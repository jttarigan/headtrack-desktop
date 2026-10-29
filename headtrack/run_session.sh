#!/bin/sh
# Records one head-tracking session for the side paper (HEADTRACK_STUDY.md):
# HeadTrackRecorder (AirPods, Mac camera + Apple Vision, iPhone reference) plus
# the MediaPipe tracker on the same camera frames. Writes runs/<UTC>_ht_<protocol>/.
#
#   headtrack/run_session.sh <protocol> [recorder options]
#   headtrack/run_session.sh pilot --speak on
#   headtrack/run_session.sh smoke --reference off
#
# <protocol> names headtrack/protocols/<protocol>.json. Recorder options are
# listed by running build/release/HeadTrackRecorder.app/Contents/MacOS/HeadTrackRecorder.
# Before the first run: headtrack/setup.sh (Python venv and MediaPipe model).
set -e
cd "$(dirname "$0")/.."
PROTO=${1:?usage: headtrack/run_session.sh <protocol> [--marker tent|calib] [--spatial off|fixed|tracked] [--photos s] [recorder options]}
shift
# --marker tent: also log the marker tent (reference angle-scale check);
# --marker calib: log the ChArUco board instead (camera calibration).
# --spatial: the Spatial Audio mode set in Control Center for the headphones
# (no public API reports it; a report says yaw recentres when audio starts
# with Fixed or Head Tracked). Written to spatial_audio.txt.
# --photos s: webcam stills (MediaPipe's frames) and recorder window shots
# every s seconds into photos/, for figures of the test process. Photo
# sessions are not measured sessions: leave them out of sessions.json.
MARKER=""
SPATIAL="not recorded"
PHOTOS=0
while [ "$1" = "--marker" ] || [ "$1" = "--spatial" ] || [ "$1" = "--photos" ]; do
    case "$1" in
        --photos) PHOTOS=${2:?--photos needs seconds} ;;
        --marker) MARKER=${2:?--marker needs tent or calib} ;;
        --spatial) SPATIAL=${2:?--spatial needs off, fixed or tracked}
                   case "$SPATIAL" in off|fixed|tracked) ;; *) echo "--spatial: off, fixed or tracked"; exit 1 ;; esac ;;
    esac
    shift 2
done
[ "$PHOTOS" = 0 ] || set -- "$@" --photos "$PHOTOS"
[ -f "headtrack/protocols/$PROTO.json" ] || { echo "no headtrack/protocols/$PROTO.json"; exit 1; }
# Protocols with targets (laser check) need the marks for this setup, made by
# headtrack/analysis/laser_targets.py from the measured geometry.
if grep -q '"target"' "headtrack/protocols/$PROTO.json"; then
    TARGETS="headtrack/protocols/${PROTO}_targets.json"
    [ -f "$TARGETS" ] || { echo "no $TARGETS: make it with headtrack/analysis/laser_targets.py"; exit 1; }
    set -- --targets "$PWD/$TARGETS" "$@"
fi
make headtrack-mac >/dev/null
OUT="$PWD/runs/$(date -u +%Y%m%dT%H%M%SZ)_ht_$PROTO"
mkdir -p "$OUT"
echo "session $OUT"
# Power source and adapter wattage (LAB.md: a 20 W adapter slows the Mac).
{ pmset -g ac; pmset -g batt; } >"$OUT/power.txt" 2>&1 || true
echo "$SPATIAL" >"$OUT/spatial_audio.txt"
# AirPods firmware, which CoreMotion does not report (headphones.json).
headtrack/.venv/bin/python headtrack/analysis/headphones.py "$OUT"
headtrack/.venv/bin/python headtrack/analysis/mp_live.py "$OUT" --photos "$PHOTOS" >"$OUT/mediapipe_log.txt" 2>&1 &
MP=$!
MK=""
if [ -n "$MARKER" ]; then
    if [ "$MARKER" = calib ]; then MKARG=--calib; else MKARG=""; fi
    headtrack/.venv/bin/python headtrack/analysis/marker_live.py "$OUT" $MKARG >"$OUT/marker_log.txt" 2>&1 &
    MK=$!
fi
# Through LaunchServices (open), so the recorder owns its permissions.
# caffeinate: display and system stay awake for long sessions.
caffeinate -di open -W -n --stdout "$OUT/console.txt" --stderr "$OUT/console.txt" \
    build/release/HeadTrackRecorder.app --args --out "$OUT" \
    --protocol "$PWD/headtrack/protocols/$PROTO.json" "$@"
# The tracker exits when the recorder closes the frame FIFO.
i=0
while kill -0 $MP 2>/dev/null && [ $i -lt 50 ]; do sleep 0.2; i=$((i + 1)); done
kill $MP 2>/dev/null || true
# The marker process exits by itself a few seconds after the frames stop.
if [ -n "$MK" ]; then
    i=0
    while kill -0 $MK 2>/dev/null && [ $i -lt 50 ]; do sleep 0.2; i=$((i + 1)); done
    kill $MK 2>/dev/null || true
fi
# Borrowed headphones' Bluetooth names carry their owner's name: remove them.
headtrack/.venv/bin/python headtrack/analysis/redact_names.py "$OUT" || true
tail -n 3 "$OUT/log.txt"
