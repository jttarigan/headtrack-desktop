"""Marker detector for the reference's angle-scale check (marker tent on the
headband, headtrack/protocols/marker_sheet.pdf; replaces the laser).

Reads the camera frames HeadTrackRecorder publishes in shared memory, like
mp_live.py, but does not use the wake-up FIFO (each FIFO write wakes only one
reader, and MediaPipe owns it). It polls the shared-memory header instead, so
MediaPipe's timing is untouched. It logs the raw corner positions of every
detected tent marker to <session>/marker.csv; poses are computed in the
analysis (htlib.track_marker), so a new camera calibration can be applied
later without recording again.

    headtrack/.venv/bin/python headtrack/analysis/marker_live.py <session dir> [--calib]

With --calib it looks for the ChArUco board instead and writes
<session>/charuco.npz (corners per frame) for marker_calib.py.
Times are CLOCK_UPTIME_RAW in ns, the recorder's clock. Exits when the
recorder stops publishing frames for --idle seconds.
"""

import argparse
import json
import mmap
import os
import struct
import sys
import time

import _posixshmem
import cv2
import numpy as np

SHM_NAME = "/eht_frames"
MAGIC = 0x31524645
HEADER = struct.Struct("<6IQ")
SLOT = struct.Struct("<4Q")
HERE = os.path.dirname(os.path.abspath(__file__))
TENT = os.path.join(HERE, "..", "protocols", "marker_tent.json")


def now_ns():
    return time.clock_gettime_ns(time.CLOCK_UPTIME_RAW)


def detector(dict_id):
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(dict_id), params)


def attach(session, wait_s):
    """Waits until the recorder has created this session's frame FIFO (so the
    shared memory is this session's, not a stale one), then maps the frames."""
    deadline = time.monotonic() + wait_s
    while not os.path.exists(os.path.join(session, "frames.fifo")):
        if time.monotonic() > deadline:
            return None
        time.sleep(0.05)
    time.sleep(0.2)
    while True:
        try:
            fd = _posixshmem.shm_open(SHM_NAME, os.O_RDONLY, 0)
            head = mmap.mmap(fd, HEADER.size, mmap.MAP_SHARED, mmap.PROT_READ)
            magic, slots, slot_bytes, width, height, stride, _ = HEADER.unpack_from(head, 0)
            head.close()
            if magic == MAGIC:
                shm = mmap.mmap(fd, HEADER.size + slots * slot_bytes, mmap.MAP_SHARED, mmap.PROT_READ)
                os.close(fd)
                return shm, slots, slot_bytes, width, height, stride
            os.close(fd)
        except (FileNotFoundError, ValueError, OSError):
            pass
        if time.monotonic() > deadline:
            return None
        time.sleep(0.05)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--calib", action="store_true", help="detect the ChArUco board instead of the tent")
    ap.add_argument("--wait", type=float, default=120.0, help="seconds to wait for the first frame")
    ap.add_argument("--idle", type=float, default=3.0, help="exit after this long without a new frame")
    ap.add_argument("--poll-ms", type=float, default=2.0)
    args = ap.parse_args()
    tent = json.load(open(TENT))
    got = attach(args.session, args.wait)
    if got is None:
        print("no shared frames after %.0f s; exiting" % args.wait)
        return 2
    shm, slots, slot_bytes, width, height, stride = got
    print("attached: %dx%d, mode %s" % (width, height, "calib" if args.calib else "tent"))

    if args.calib:
        b = tent["board"]
        board = cv2.aruco.CharucoBoard(tuple(b["squares"]), b["square_mm"] / 1000, b["marker_mm"] / 1000,
                                       cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100))
        cdet = cv2.aruco.CharucoDetector(board)
        frames = []
    else:
        det = detector(cv2.aruco.DICT_4X4_50)
        ids_wanted = {int(k) for k in tent["corners"]}
        out = open(os.path.join(args.session, "marker.csv"), "w", buffering=1 << 16)
        out.write("seq,pts_ns,delivered_ns,start_ns,done_ns,id," + ",".join("c%d%s" % (i, a) for i in range(4) for a in "xy") + "\n")

    last_seq, last_new = 0, time.monotonic()
    n_frames = n_hits = 0
    proc_ms = []
    cpu0, wall0 = time.process_time(), time.monotonic()
    while True:
        latest = HEADER.unpack_from(shm, 0)[6]
        if latest <= last_seq:
            if time.monotonic() - last_new > args.idle and last_seq:
                break
            if not last_seq and time.monotonic() - wall0 > args.wait:
                break
            time.sleep(args.poll_ms / 1000)
            continue
        off = HEADER.size + (latest % slots) * slot_bytes
        seq, pts_ns, delivered_ns, _ = SLOT.unpack_from(shm, off)
        if seq != latest:
            continue
        pixels = np.frombuffer(shm, dtype=np.uint8, count=stride * height, offset=off + SLOT.size)
        bgra = pixels.reshape(height, stride)[:, : width * 4].reshape(height, width, 4).copy()
        if SLOT.unpack_from(shm, off)[0] != latest:
            continue
        last_seq, last_new = latest, time.monotonic()
        start = now_ns()
        gray = cv2.cvtColor(bgra, cv2.COLOR_BGRA2GRAY)
        n_frames += 1
        if args.calib:
            cc, cid, _, _ = cdet.detectBoard(gray)
            if cid is not None and len(cid) >= 8:
                frames.append((pts_ns, cc.reshape(-1, 2).astype(np.float32), cid.reshape(-1).astype(np.int32)))
                n_hits += 1
        else:
            corners, ids, _ = det.detectMarkers(gray)
            done = now_ns()
            if ids is not None:
                for c, i in zip(corners, ids.reshape(-1)):
                    if int(i) in ids_wanted:
                        out.write("%d,%d,%d,%d,%d,%d,%s\n" % (latest, pts_ns, delivered_ns, start, done, int(i),
                                                             ",".join("%.3f" % v for v in c.reshape(-1))))
                        n_hits += 1
        proc_ms.append((now_ns() - start) / 1e6)

    wall = time.monotonic() - wall0
    summary = {"tool": "marker_live.py", "mode": "calib" if args.calib else "tent", "opencv": cv2.__version__,
               "width": width, "height": height, "frames": n_frames, "detections": n_hits,
               "cpu_share_of_one_core": round((time.process_time() - cpu0) / wall, 3) if wall > 0 else None,
               "processing_ms_median": round(float(np.median(proc_ms)), 3) if proc_ms else None}
    if args.calib:
        np.savez_compressed(os.path.join(args.session, "charuco.npz"),
                            pts_ns=np.array([f[0] for f in frames], dtype=np.int64),
                            corners=np.array([f[1] for f in frames], dtype=object),
                            ids=np.array([f[2] for f in frames], dtype=object),
                            width=width, height=height)
    else:
        out.close()
    with open(os.path.join(args.session, "marker_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    print("done: %s" % json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
