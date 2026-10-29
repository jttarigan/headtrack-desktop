"""Synthetic check of the marker chain: calibration, detection and tent pose.

    headtrack/.venv/bin/python headtrack/analysis/test_marker.py

Renders the printed tent and the ChArUco board into 1280x720 frames with a
known camera and known head poses, runs the same detector, calibration and
pose code as real sessions, and checks the recovered angles. Prints the error
budget used in the paper; exits non-zero if a check fails.
"""

import json
import math
import os
import sys

import cv2
import numpy as np

import marker_calib
import marker_pose

HERE = os.path.dirname(os.path.abspath(__file__))
W, H = 1280, 720
K_TRUE = np.array([[1000.0, 0, 640.0], [0, 1000.0, 360.0], [0, 0, 1]])
DIST_TRUE = np.array([0.05, -0.10, 0.0, 0.0, 0.0])
R0 = np.diag([1.0, -1.0, -1.0])  # tent frame (y up, z toward camera) -> camera frame
TENT_JSON = json.load(open(os.path.join(HERE, "..", "protocols", "marker_tent.json")))


def rot(yaw, pitch):
    y, p = math.radians(yaw), math.radians(pitch)
    Ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    Rx = np.array([[1, 0, 0], [0, math.cos(p), -math.sin(p)], [0, math.sin(p), math.cos(p)]])
    return Ry @ Rx


def face_quads():
    """Each face's outer corners (with the quiet zone) in the tent frame, m."""
    t = TENT_JSON
    c, s = math.cos(math.radians(t["face_deg"])), math.sin(math.radians(t["face_deg"]))
    fw, fh = t["face_mm"][0] / 1000, t["face_mm"][1] / 1000
    L = lambda d, y: [-d * c, y, -d * s]
    R = lambda d, y: [d * c, y, -d * s]
    return {0: np.array([L(fw, fh / 2), L(0, fh / 2), L(0, -fh / 2), L(fw, -fh / 2)]),
            1: np.array([R(0, fh / 2), R(fw, fh / 2), R(fw, -fh / 2), R(0, -fh / 2)])}


def face_image(k, px=700):
    d = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    t = TENT_JSON
    m = int(px * t["marker_mm"] / t["face_mm"][0])
    img = np.full((px, px), 255, np.uint8)
    off = (px - m) // 2
    img[off:off + m, off:off + m] = cv2.aruco.generateImageMarker(d, k, m, borderBits=1)
    return img


def render_tent(Rc, tc, K, dist):
    frame = np.full((H, W), 128, np.uint8)
    quads = face_quads()
    for k in (0, 1):
        q = quads[k]
        normal = np.cross(q[3] - q[0], q[1] - q[0])  # outward normal (toward camera at rest)
        center = q.mean(0)
        n_cam = Rc @ normal
        p_cam = Rc @ center + tc
        if np.dot(n_cam, -p_cam) <= 0:  # face turned away
            continue
        proj, _ = cv2.projectPoints(q, cv2.Rodrigues(Rc)[0], tc, K, dist)
        img = face_image(k)
        s = img.shape[0] - 1
        Hm = cv2.getPerspectiveTransform(np.float32([[0, 0], [s, 0], [s, s], [0, s]]), proj.reshape(4, 2).astype(np.float32))
        warped = cv2.warpPerspective(img, Hm, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
        mask = cv2.warpPerspective(np.full_like(img, 255), Hm, (W, H), flags=cv2.INTER_NEAREST, borderValue=0)
        frame[mask > 0] = warped[mask > 0]
    return frame


def detect(frame):
    det = marker_pose.cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50),
                                              _params())
    corners, ids, _ = det.detectMarkers(frame)
    return {} if ids is None else {int(i): c.reshape(4, 2) for c, i in zip(corners, ids.reshape(-1))}


def _params():
    p = cv2.aruco.DetectorParameters()
    p.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return p


def test_calibration():
    b = marker_calib.board()
    sq = b.getSquareLength()
    nx, ny = b.getChessboardSize()
    bw, bh = nx * sq, ny * sq
    img = b.generateImage((1000, 1400), marginSize=0, borderBits=1)
    corners, ids = [], []
    rng = np.random.default_rng(1)
    cdet = cv2.aruco.CharucoDetector(b)
    for _ in range(25):
        R = rot(rng.uniform(-35, 35), rng.uniform(-35, 35)) @ rot(0, 0)
        Rz = cv2.Rodrigues(np.array([0, 0, rng.uniform(-0.4, 0.4)]))[0]
        Rc = R0 @ R @ Rz
        tc = np.array([rng.uniform(-0.08, 0.08), rng.uniform(-0.05, 0.05), rng.uniform(0.35, 0.6)])
        # board plane: origin at its top-left corner, x right, y down in the image -> tent-like frame
        quad = np.array([[0, 0, 0], [bw, 0, 0], [bw, -bh, 0], [0, -bh, 0]]) - np.array([bw / 2, -bh / 2, 0])
        proj, _ = cv2.projectPoints(quad, cv2.Rodrigues(Rc)[0], tc, K_TRUE, DIST_TRUE)
        Hm = cv2.getPerspectiveTransform(np.float32([[0, 0], [999, 0], [999, 1399], [0, 1399]]), proj.reshape(4, 2).astype(np.float32))
        frame = np.full((H, W), 128, np.uint8)
        warped = cv2.warpPerspective(img, Hm, (W, H), borderValue=0)
        mask = cv2.warpPerspective(np.full_like(img, 255), Hm, (W, H), flags=cv2.INTER_NEAREST, borderValue=0)
        frame[mask > 0] = warped[mask > 0]
        cc, cid, _, _ = cdet.detectBoard(frame)
        if cid is not None and len(cid) >= 8:
            corners.append(cc.reshape(-1, 2))
            ids.append(cid.reshape(-1))
    rms, K, dist, n = marker_calib.calibrate(corners, ids, (W, H), b)
    fx_err = abs(K[0, 0] - K_TRUE[0, 0]) / K_TRUE[0, 0] * 100
    print("calibration: %d views, rms %.3f px, fx %.1f (true 1000, %.2f %%), cx %.1f, cy %.1f"
          % (n, rms, K[0, 0], fx_err, K[0, 2], K[1, 2]))
    return K, dist, fx_err < 1.0 and n >= 15


def test_pose(K, dist):
    tent = marker_pose.load_tent()
    tc = np.array([0.0, -0.12, 0.62])  # tent on the forehead, above the camera axis
    neutral = None
    worst = 0.0
    rows = []
    for yaw in (-60, -45, -30, -15, 0, 15, 30, 45, 60):
        for pitch in (-20, 0, 20):
            Rc = R0 @ rot(yaw, pitch)
            frame = render_tent(Rc, tc, K_TRUE, DIST_TRUE)
            obs = detect(frame)
            res = marker_pose.solve_pose(obs, tent, K, dist)
            if res is None:
                rows.append((yaw, pitch, None, 0))
                continue
            Rm, _ = cv2.Rodrigues(res[0])
            if yaw == 0 and pitch == 0:
                neutral = Rm
            rows.append((yaw, pitch, Rm, res[3]))
    errs = []
    for yaw, pitch, Rm, faces in rows:
        if Rm is None:
            print("  yaw %+3d pitch %+3d: not detected" % (yaw, pitch))
            continue
        rel = neutral.T @ Rm  # rotation from neutral, tent frame
        true = rot(yaw, pitch)
        e = math.degrees(math.acos(np.clip((np.trace(true.T @ rel) - 1) / 2, -1, 1)))
        ang = math.degrees(math.acos(np.clip((np.trace(rel) - 1) / 2, -1, 1)))
        ang_true = math.degrees(math.acos(np.clip((np.trace(true) - 1) / 2, -1, 1)))
        errs.append(e)
        worst = max(worst, e)
        print("  yaw %+3d pitch %+3d: faces %d, error %.2f deg, angle %.2f vs %.2f" % (yaw, pitch, faces, e, ang, ang_true))
    print("pose: %d of %d poses detected, median error %.2f deg, max %.2f deg" % (len(errs), len(rows), np.median(errs), worst))
    # single-face views (beyond about 40 degrees) are the weakest; both faces stay under 0.75 deg
    return len(errs) >= 21 and np.median(errs) < 0.5 and worst < 1.5


def test_noise(K, dist, sigma_px=0.3, n=300):
    """Angle noise from corner noise alone, frontal and at 30 degrees of yaw."""
    tent = marker_pose.load_tent()
    tc = np.array([0.0, -0.12, 0.62])
    rng = np.random.default_rng(2)
    out = {}
    for yaw in (0, 30):
        Rc = R0 @ rot(yaw, 0)
        rv = cv2.Rodrigues(Rc)[0]
        obs = {}
        for k, P in tent.items():
            proj, _ = cv2.projectPoints(P, rv, tc, K, dist)
            obs[k] = proj.reshape(4, 2)
        # keep only faces turned toward the camera
        es = []
        for _ in range(n):
            noisy = {k: v + rng.normal(0, sigma_px, v.shape) for k, v in obs.items()}
            res = marker_pose.solve_pose(noisy, tent, K, dist)
            es.append(marker_pose.rotvec_angle_deg(res[0], rv.reshape(3)))
        out[yaw] = float(np.sqrt(np.mean(np.square(es))))
        print("noise: corner sigma %.1f px -> rotation error rms %.3f deg at yaw %d" % (sigma_px, out[yaw], yaw))
    return all(v < 0.5 for v in out.values())


def main():
    K, dist, ok_cal = test_calibration()
    ok_pose = test_pose(K, dist)
    ok_noise = test_noise(K, dist)
    ok = ok_cal and ok_pose and ok_noise
    print("ALL PASS" if ok else "FAIL (calibration %s, pose %s, noise %s)" % (ok_cal, ok_pose, ok_noise))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
