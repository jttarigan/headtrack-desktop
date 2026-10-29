"""Marker tent pose from the corners marker_live.py logs (reference angle-scale
check; tent geometry in headtrack/protocols/marker_tent.json).

The tent's orientation comes from a perspective-n-point fit of all visible
corners with the camera's intrinsics (marker_calib.py). With both faces in
view the eight corners are not coplanar and the fit is unique (SQPnP). With
one face only, a planar fit has two candidate poses (IPPE); the one closer to
the previous accepted pose is kept, and the pose is flagged as single-face.
"""

import glob
import json
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROTOCOLS = os.path.join(HERE, "..", "protocols")
TENT_PATH = os.path.join(PROTOCOLS, "marker_tent.json")


def load_tent(path=TENT_PATH):
    t = json.load(open(path))
    return {int(k): np.asarray(v, dtype=np.float64) / 1000.0 for k, v in t["corners"].items()}  # metres


def intrinsics_path(session_dir, width, height):
    """Session's own calibration if present, else the newest one for this resolution."""
    own = os.path.join(session_dir, "camera_intrinsics.json")
    if os.path.exists(own):
        return own
    cands = sorted(glob.glob(os.path.join(PROTOCOLS, "camera_%dx%d_*.json" % (width, height))))
    return cands[-1] if cands else None


def load_intrinsics(path):
    d = json.load(open(path))
    return np.asarray(d["K"], dtype=np.float64), np.asarray(d["dist"], dtype=np.float64), d


def rotvec_angle_deg(rvec_a, rvec_b):
    Ra, _ = cv2.Rodrigues(rvec_a)
    Rb, _ = cv2.Rodrigues(rvec_b)
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def solve_pose(obs, tent, K, dist, prev_rvec=None):
    """obs: {id: 4x2 pixel corners}. Returns (rvec, tvec, rms_px, n_faces) or None.
    rvec/tvec map tent coordinates (m) to camera coordinates (OpenCV: x right,
    y down, z forward)."""
    ids = [i for i in sorted(obs) if i in tent]
    if not ids:
        return None
    obj = np.concatenate([tent[i] for i in ids]).astype(np.float64)
    img = np.concatenate([np.asarray(obs[i], dtype=np.float64).reshape(4, 2) for i in ids])
    if len(ids) >= 2:
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_SQPNP)
        if not ok:
            return None
        rvec, tvec = cv2.solvePnPRefineLM(obj, img, K, dist, rvec, tvec)
    else:
        # planar: express the face in its own plane for IPPE, then map back
        P = obj
        o = P[0]
        ex = (P[1] - P[0]) / np.linalg.norm(P[1] - P[0])
        ey0 = P[3] - P[0]
        ez = np.cross(ex, ey0)
        ez /= np.linalg.norm(ez)
        ey = np.cross(ez, ex)
        Rf = np.stack([ex, ey, ez], 1)  # face -> tent
        local = (P - o) @ Rf
        n, rvecs, tvecs, _ = cv2.solvePnPGeneric(local, img, K, dist, flags=cv2.SOLVEPNP_IPPE)
        if not n:
            return None
        best = None
        for rv, tv in zip(rvecs, tvecs):
            Rc, _ = cv2.Rodrigues(rv)  # face -> camera
            R = Rc @ Rf.T  # tent -> camera
            t = tv.reshape(3) - R @ o
            rv_t, _ = cv2.Rodrigues(R)
            proj, _ = cv2.projectPoints(obj, rv_t, t, K, dist)
            err = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, 1))))
            score = rotvec_angle_deg(rv_t, prev_rvec) if prev_rvec is not None else err
            if t[2] > 0 and (best is None or score < best[0]):
                best = (score, rv_t, t.reshape(3, 1))
        if best is None:
            return None
        rvec, tvec = best[1], best[2]
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    rms = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img) ** 2, 1))))
    return rvec.reshape(3), tvec.reshape(3), rms, len(ids)


def quat_from_rvec(rvec):
    """Rotation vector -> quaternion [x, y, z, w] (htlib convention)."""
    a = float(np.linalg.norm(rvec))
    if a < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    ax = np.asarray(rvec, dtype=np.float64) / a
    return np.concatenate([ax * np.sin(a / 2), [np.cos(a / 2)]])


def poses_from_csv(table, tent, K, dist, max_rms_px=2.0):
    """marker.csv rows (several per frame) -> per-frame arrays
    (seq, pts_ns, done_ns, q [n,4], valid, rms_px, n_faces)."""
    seqs = table["seq"].astype(np.int64)
    order = np.argsort(seqs, kind="stable")
    out = {k: [] for k in ("seq", "pts_ns", "done_ns", "q", "valid", "rms", "faces")}
    prev = None
    i = 0
    n = len(order)
    while i < n:
        j = i
        s0 = seqs[order[i]]
        obs = {}
        while j < n and seqs[order[j]] == s0:
            r = order[j]
            obs[int(table["id"][r])] = np.array([[table["c%d%s" % (k, a)][r] for a in "xy"] for k in range(4)], dtype=np.float64)
            j += 1
        r0 = order[i]
        res = solve_pose(obs, tent, K, dist, prev)
        ok = res is not None and res[2] <= max_rms_px
        out["seq"].append(int(s0))
        out["pts_ns"].append(int(table["pts_ns"][r0]))
        out["done_ns"].append(int(table["done_ns"][r0]))
        out["q"].append(quat_from_rvec(res[0]) if ok else np.full(4, np.nan))
        out["valid"].append(bool(ok))
        out["rms"].append(res[2] if res is not None else np.nan)
        out["faces"].append(res[3] if res is not None else 0)
        if ok:
            prev = res[0]
        i = j
    return {k: np.asarray(v) for k, v in out.items()}
