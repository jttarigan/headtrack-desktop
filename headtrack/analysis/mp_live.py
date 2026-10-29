"""MediaPipe head tracker for the head-tracking study (HEADTRACK_STUDY.md).

Reads the camera frames HeadTrackRecorder publishes in shared memory
(headtrack/common/frames.h), runs MediaPipe Face Landmarker on the newest
frame each time it wakes, and logs the facial transformation matrix to
<session>/mediapipe.csv. Frames that arrive while it is busy are skipped, as
in any live tracker. All times are CLOCK_UPTIME_RAW in ns, which is the
recorder's clock (mach_absolute_time).

    headtrack/.venv/bin/python headtrack/analysis/mp_live.py <session dir>

Exits when the recorder closes the frame FIFO, or if no frames come within
--wait seconds. Writes mediapipe_summary.json at the end. --photos N also
saves the frame it tracks every N s as photos/webcam_<UTC>.jpg, for figures of
the test process (not for measured sessions: the JPEG write runs in this loop).
"""

import argparse
import hashlib
import json
import mmap
import os
import struct
import sys
import time

import _posixshmem
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

SHM_NAME = "/eht_frames"
MAGIC = 0x31524645
HEADER = struct.Struct("<6IQ")  # magic, slots, slot_bytes, width, height, stride, latest_seq
SLOT = struct.Struct("<4Q")  # seq, pts_ns, delivered_ns, shared_ns
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(HERE, "..", "models", "face_landmarker.task")


def now_ns():
    return time.clock_gettime_ns(time.CLOCK_UPTIME_RAW)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--wait", type=float, default=120.0, help="seconds to wait for the first frame")
    ap.add_argument("--photos", type=float, default=0.0, help="save a frame as JPEG every N s (0 = off)")
    args = ap.parse_args()

    model_path = os.path.abspath(args.model)
    with open(model_path, "rb") as f:
        model_sha = hashlib.sha256(f.read()).hexdigest()
    landmarker = vision.FaceLandmarker.create_from_options(
        vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            output_facial_transformation_matrixes=True,
        )
    )
    log = print
    fifo_path = os.path.join(args.session, "frames.fifo")
    deadline = time.monotonic() + args.wait
    while not os.path.exists(fifo_path):
        if time.monotonic() > deadline:
            log("no frame FIFO after %.0f s; exiting" % args.wait)
            return 2
        time.sleep(0.05)
    fifo = os.open(fifo_path, os.O_RDONLY)  # blocks until the recorder opens it
    fd = _posixshmem.shm_open(SHM_NAME, os.O_RDONLY, 0)
    head = mmap.mmap(fd, HEADER.size, mmap.MAP_SHARED, mmap.PROT_READ)
    magic, slots, slot_bytes, width, height, stride, _ = HEADER.unpack_from(head, 0)
    head.close()
    if magic != MAGIC:
        log("bad shared-memory magic %08x" % magic)
        return 2
    shm = mmap.mmap(fd, HEADER.size + slots * slot_bytes, mmap.MAP_SHARED, mmap.PROT_READ)
    os.close(fd)
    log("attached: %dx%d, stride %d, %d slots, model sha256 %s" % (width, height, stride, slots, model_sha))

    out = open(os.path.join(args.session, "mediapipe.csv"), "w", buffering=1 << 16)
    out.write(
        "seq,pts_ns,delivered_ns,shared_ns,woke_ns,start_ns,done_ns,skipped,face,"
        + ",".join("m%d%d" % (r, c) for r in range(4) for c in range(4))
        + "\n"
    )
    cpu0, wall0 = time.process_time(), time.monotonic()
    last_seq, last_ts_ms = 0, -1
    n_frames = n_faces = n_skipped = n_torn = 0
    infer_ms = []
    last_flush = time.monotonic()
    photo_dir, last_photo_ns, n_photos = os.path.join(args.session, "photos"), 0, 0
    while True:
        woke = os.read(fifo, 4096)
        woke_ns = now_ns()
        if not woke:
            break  # recorder closed the FIFO
        latest = HEADER.unpack_from(shm, 0)[6]
        if latest <= last_seq:
            continue
        off = HEADER.size + (latest % slots) * slot_bytes
        seq, pts_ns, delivered_ns, shared_ns = SLOT.unpack_from(shm, off)
        if seq != latest:
            n_torn += 1
            continue
        pixels = np.frombuffer(shm, dtype=np.uint8, count=stride * height, offset=off + SLOT.size)
        bgra = pixels.reshape(height, stride)[:, : width * 4].reshape(height, width, 4).copy()
        if SLOT.unpack_from(shm, off)[0] != latest:  # overwritten while copying
            n_torn += 1
            continue
        skipped = latest - last_seq - 1 if last_seq else 0
        n_skipped += skipped
        last_seq = latest
        ts_ms = pts_ns // 1_000_000
        if ts_ms <= last_ts_ms:
            continue
        last_ts_ms = ts_ms
        start_ns = now_ns()
        if args.photos > 0 and pts_ns - last_photo_ns >= args.photos * 1e9:
            last_photo_ns = pts_ns
            os.makedirs(photo_dir, exist_ok=True)
            stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime()) + ".%03dZ" % (time.time_ns() // 1_000_000 % 1000)
            cv2.imwrite(os.path.join(photo_dir, "webcam_%s.jpg" % stamp), bgra[:, :, :3], [cv2.IMWRITE_JPEG_QUALITY, 92])
            n_photos += 1
        rgb = cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGB)
        result = landmarker.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts_ms)
        done_ns = now_ns()
        n_frames += 1
        infer_ms.append((done_ns - start_ns) / 1e6)
        if result.facial_transformation_matrixes:
            n_faces += 1
            m = np.asarray(result.facial_transformation_matrixes[0], dtype=np.float64).reshape(16)
            mat = ",".join("%.7g" % v for v in m)
            face = 1
        else:
            mat = ",".join([""] * 16)
            face = 0
        out.write(
            "%d,%d,%d,%d,%d,%d,%d,%d,%d,%s\n"
            % (latest, pts_ns, delivered_ns, shared_ns, woke_ns, start_ns, done_ns, skipped, face, mat)
        )
        if time.monotonic() - last_flush > 1.0:
            out.flush()
            last_flush = time.monotonic()
    out.close()
    cpu, wall = time.process_time() - cpu0, time.monotonic() - wall0
    infer = np.array(infer_ms) if infer_ms else np.array([np.nan])
    summary = {
        "tool": "mp_live.py",
        "mediapipe": mp.__version__,
        "opencv": cv2.__version__,
        "python": sys.version.split()[0],
        "model": os.path.relpath(model_path, os.path.join(HERE, "..", "..")),
        "model_sha256": model_sha,
        "frames_processed": n_frames,
        "frames_with_face": n_faces,
        "frames_skipped": int(n_skipped),
        "frames_torn": n_torn,
        "photos": n_photos,
        "process_cpu_s": round(cpu, 3),
        "wall_s": round(wall, 3),
        "cpu_share_of_one_core": round(cpu / wall, 3) if wall > 0 else None,
        "convert_plus_inference_ms": {
            "median": round(float(np.nanmedian(infer)), 3),
            "p95": round(float(np.nanpercentile(infer, 95)), 3),
            "max": round(float(np.nanmax(infer)), 3),
        },
    }
    with open(os.path.join(args.session, "mediapipe_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
        f.write("\n")
    log("done: %s" % json.dumps(summary))
    return 0


if __name__ == "__main__":
    sys.exit(main())
