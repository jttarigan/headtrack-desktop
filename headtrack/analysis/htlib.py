"""Analysis library for the head-tracking side paper (HEADTRACK_STUDY.md).

Turns one session folder written by headtrack/run_session.sh into numbers:
iPhone-to-Mac clock sync, each tracker's delivery timing and latency, the
fixed rotations that align it with the reference, and its agreement, noise,
drift and coverage. numpy only; the optional plot uses matplotlib.

Conventions
- Quaternions are numpy arrays [..., 4] in (x, y, z, w) order, Hamilton
  product, unit length. A tracker's q rotates vectors from its body frame
  (head or sensor) into its world frame: v_world = q v q*.
- Time: the logs hold mach_absolute_time in ns. Here times are float seconds
  on the Mac clock, relative to the session origin t0 (session.json start_ns).
  iPhone times are mapped onto the Mac clock with ClockMap (sync.csv).
- Reference: the ARKit face anchor in the iPhone camera frame. ARKit face axes:
  +x towards the face's own left (the viewer's right), +y up, +z out of the
  face. Its capture time is taken as the physical time of the pose.
- Alignment model: q_ref(t) = A q_trk(t + L) B. L is the tracker's lag (its
  sample time minus physical time); A rotates tracker-world vectors into the
  reference camera frame; B rotates head-frame vectors into the tracker body.
- Head-relative angles: q_rel = q_neutral^-1 q, decomposed as
  R = Ry(yaw) Rx(pitch) Rz(roll) in the neutral head frame, in degrees.
  yaw + = turning towards the face's own left, pitch + = nose down,
  roll + = left ear up.
- Percentiles use numpy's default (linear, "type 7"), as in src/core/stats.
"""

import csv
import itertools
import json
import math
import os
import re

import numpy as np

DEG = 180.0 / math.pi

# ---- quaternions -------------------------------------------------------------

_CONJ = np.array([-1.0, -1.0, -1.0, 1.0])


def qnormalize(q):
    q = np.asarray(q, dtype=float)
    return q / np.linalg.norm(q, axis=-1, keepdims=True)


def qconj(q):
    return np.asarray(q, dtype=float) * _CONJ


def qmul(a, b):
    """Hamilton product a b (apply b first, then a)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ax, ay, az, aw = np.moveaxis(a, -1, 0)
    bx, by, bz, bw = np.moveaxis(b, -1, 0)
    return np.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        axis=-1,
    )


def qaxis(axis, angle):
    """Rotation by angle (rad) about coordinate axis 0 = x, 1 = y, 2 = z."""
    angle = np.asarray(angle, dtype=float)
    q = np.zeros(angle.shape + (4,))
    q[..., axis] = np.sin(angle / 2)
    q[..., 3] = np.cos(angle / 2)
    return q


def qfrom_rotvec(v):
    v = np.asarray(v, dtype=float)
    ang = np.linalg.norm(v, axis=-1, keepdims=True)
    big = ang > 1e-12
    k = np.where(big, np.sin(ang / 2) / np.where(big, ang, 1.0), 0.5 - ang**2 / 48)
    return np.concatenate([v * k, np.cos(ang / 2)], axis=-1)


def qto_rotvec(q):
    """Rotation vector (axis times angle, rad), angle in [0, pi]."""
    q = np.asarray(q, dtype=float)
    q = np.where(q[..., 3:4] < 0, -q, q)
    s = np.linalg.norm(q[..., :3], axis=-1, keepdims=True)
    ang = 2 * np.arctan2(s, q[..., 3:4])
    big = s > 1e-12
    k = np.where(big, ang / np.where(big, s, 1.0), 2.0)
    return q[..., :3] * k


def qangle(a, b):
    """Geodesic angle (rad) between orientations a and b."""
    r = qmul(qconj(a), b)
    return 2 * np.arctan2(np.linalg.norm(r[..., :3], axis=-1), np.abs(r[..., 3]))


def matrix_from_quat(q):
    q = np.asarray(q, dtype=float)
    x, y, z, w = np.moveaxis(q, -1, 0)
    return np.stack(
        [
            np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
            np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
            np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1),
        ],
        axis=-2,
    )


def quat_from_matrix(R):
    """Unit quaternion of a rotation matrix (Shepperd's method), w >= 0."""
    R = np.asarray(R, dtype=float)
    shape = R.shape[:-2]
    R = R.reshape(-1, 3, 3)
    out = np.empty((len(R), 4))
    tr = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    case = np.argmax(np.stack([tr, R[:, 0, 0], R[:, 1, 1], R[:, 2, 2]], axis=1), axis=1)
    for c in range(4):
        i = case == c
        if not i.any():
            continue
        m = R[i]
        if c == 0:
            w = 0.5 * np.sqrt(1 + tr[i])
            out[i] = np.stack([(m[:, 2, 1] - m[:, 1, 2]) / (4 * w), (m[:, 0, 2] - m[:, 2, 0]) / (4 * w),
                               (m[:, 1, 0] - m[:, 0, 1]) / (4 * w), w], -1)
        elif c == 1:
            x = 0.5 * np.sqrt(1 + m[:, 0, 0] - m[:, 1, 1] - m[:, 2, 2])
            out[i] = np.stack([x, (m[:, 0, 1] + m[:, 1, 0]) / (4 * x), (m[:, 0, 2] + m[:, 2, 0]) / (4 * x),
                               (m[:, 2, 1] - m[:, 1, 2]) / (4 * x)], -1)
        elif c == 2:
            y = 0.5 * np.sqrt(1 - m[:, 0, 0] + m[:, 1, 1] - m[:, 2, 2])
            out[i] = np.stack([(m[:, 0, 1] + m[:, 1, 0]) / (4 * y), y, (m[:, 1, 2] + m[:, 2, 1]) / (4 * y),
                               (m[:, 0, 2] - m[:, 2, 0]) / (4 * y)], -1)
        else:
            z = 0.5 * np.sqrt(1 - m[:, 0, 0] - m[:, 1, 1] + m[:, 2, 2])
            out[i] = np.stack([(m[:, 0, 2] + m[:, 2, 0]) / (4 * z), (m[:, 1, 2] + m[:, 2, 1]) / (4 * z), z,
                               (m[:, 1, 0] - m[:, 0, 1]) / (4 * z)], -1)
    out = qnormalize(out)
    out = np.where(out[:, 3:4] < 0, -out, out)
    return out.reshape(shape + (4,))


def orthonormalize(M):
    """Nearest rotation matrix (SVD), for matrices that carry scale or noise."""
    U, _, Vt = np.linalg.svd(np.asarray(M, dtype=float))
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.zeros(U.shape)
    D[..., 0, 0] = 1
    D[..., 1, 1] = 1
    D[..., 2, 2] = d
    return U @ D @ Vt


def quat_from_euler_yxz(yaw, pitch, roll):
    """R = Ry(yaw) Rx(pitch) Rz(roll), angles in rad."""
    return qmul(qmul(qaxis(1, yaw), qaxis(0, pitch)), qaxis(2, roll))


def euler_yxz_from_quat(q):
    """(yaw, pitch, roll) in rad with R = Ry(yaw) Rx(pitch) Rz(roll)."""
    R = matrix_from_quat(q)
    pitch = np.arcsin(np.clip(-R[..., 1, 2], -1, 1))
    yaw = np.arctan2(R[..., 0, 2], R[..., 2, 2])
    roll = np.arctan2(R[..., 1, 0], R[..., 1, 1])
    return yaw, pitch, roll


def slerp(q0, q1, u):
    q0 = np.asarray(q0, dtype=float)
    q1 = np.asarray(q1, dtype=float)
    d = np.sum(q0 * q1, axis=-1)
    q1 = np.where(d[..., None] < 0, -q1, q1)
    d = np.abs(d)
    u = np.asarray(u, dtype=float)[..., None]
    theta = np.arccos(np.clip(d, -1, 1))[..., None]
    s = np.sin(theta)
    small = s < 1e-6
    safe = np.where(small, 1.0, s)
    w0 = np.where(small, 1 - u, np.sin((1 - u) * theta) / safe)
    w1 = np.where(small, u, np.sin(u * theta) / safe)
    return qnormalize(w0 * q0 + w1 * q1)


def qaverage(q, w=None):
    """Mean orientation: principal eigenvector of sum w q q^T (Markley et al. 2007)."""
    q = np.asarray(q, dtype=float)
    keep = np.all(np.isfinite(q), axis=-1)
    q = q[keep]
    w = np.ones(len(q)) if w is None else np.asarray(w, dtype=float)[keep]
    with np.errstate(all="ignore"):  # numpy 2 + Accelerate flags spurious matmul warnings
        M = (q * w[:, None]).T @ q
    _, vecs = np.linalg.eigh(M)
    v = vecs[:, -1]
    return -v if v[3] < 0 else v


def wrap_deg(a):
    return (np.asarray(a) + 180.0) % 360.0 - 180.0


# ---- reading logs ------------------------------------------------------------

_INT = re.compile(r"^-?\d+$")
_NS_COLS = ("t1", "t2", "t3", "t4", "seq", "t_ns")


class Table(dict):
    """CSV columns as numpy arrays: int64 for integer columns (empty -> -1),
    float64 for numeric ones (empty -> nan), object arrays for text. n = rows."""

    n = 0


def read_csv(path):
    if not os.path.exists(path):
        return None
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        return None
    header, body = rows[0], [r for r in rows[1:] if r]
    t = Table()
    t.n = len(body)
    for j, name in enumerate(header):
        col = [r[j] if j < len(r) else "" for r in body]
        filled = [c for c in col if c != ""]
        is_ns = name.endswith("_ns") or name in _NS_COLS
        if filled and all(_INT.match(c) for c in filled):
            t[name] = np.array([int(c) if c != "" else -1 for c in col], dtype=np.int64)
        elif not filled and is_ns:
            t[name] = np.full(len(col), -1, dtype=np.int64)
        else:
            try:
                t[name] = np.array([float(c) if c != "" else np.nan for c in col], dtype=float)
            except ValueError:
                t[name] = np.array(col, dtype=object)
    return t


def headphones_from_system_profiler(sp):
    """Connected Bluetooth headphones in `system_profiler SPBluetoothDataType -json` output.

    CoreMotion does not report the AirPods firmware, and Apple can change the
    motion API's behaviour with it, so each session records it. Keeps name,
    firmware and ids; drops addresses and serials (session folders may be
    published)."""
    out = []
    for ctl in sp.get("SPBluetoothDataType", []):
        for entry in ctl.get("device_connected") or []:
            for name, info in entry.items():
                if info.get("device_minorType") not in ("Headphones", "Headset"):
                    continue
                out.append({"name": name, "firmware": info.get("device_firmwareVersion"),
                            "vendor_id": info.get("device_vendorID"), "product_id": info.get("device_productID"),
                            "minor_type": info.get("device_minorType")})
    return out


class Session:
    """Raw tables of one session folder; any of them may be None."""

    def __init__(self, path):
        self.path = path
        self.meta = {}
        meta_path = os.path.join(path, "session.json")
        if os.path.exists(meta_path):
            with open(meta_path) as f:
                self.meta = json.load(f)
        self.headphones = None  # headphones.json, written by run_session.sh
        hp_path = os.path.join(path, "headphones.json")
        if os.path.exists(hp_path):
            with open(hp_path) as f:
                self.headphones = json.load(f)
        names = ("airpods", "camera", "ref", "sync", "mediapipe", "events", "marker")
        for name in names:
            setattr(self, name, read_csv(os.path.join(path, name + ".csv")))
        self.t0 = self._origin()

    def _origin(self):
        if "start_ns" in self.meta:
            return int(self.meta["start_ns"])
        cands = []
        for tab, col in ((self.airpods, "arrival_ns"), (self.camera, "pts_ns"), (self.sync, "t1"),
                         (self.events, "t_ns"), (self.ref, "recv_ns"), (self.mediapipe, "pts_ns")):
            if tab is not None and tab.n and col in tab:
                v = tab[col][tab[col] > 0]
                if len(v):
                    cands.append(int(v.min()))
        return min(cands) if cands else 0

    def seconds(self, ns):
        """Mac-clock ns (int64) -> float seconds since t0."""
        return (np.asarray(ns, dtype=np.int64) - self.t0) / 1e9

    def steps(self):
        """Protocol steps as (t_s, duration_s, text), from events.csv."""
        out = []
        ev = self.events
        if ev is None or not ev.n:
            return out
        for t, kind, detail in zip(ev["t_ns"], ev["kind"], ev["detail"]):
            if str(kind) != "step":
                continue
            m = re.match(r"^\d+/\d+ ([\d.]+)s (.*)$", str(detail))
            if m:
                out.append(((int(t) - self.t0) / 1e9, float(m.group(1)), m.group(2)))
        return out


# ---- clock sync --------------------------------------------------------------


class ClockMap:
    """iPhone clock -> Mac clock from the NTP-style ping exchange in sync.csv.

    Per ping: offset = ((t2 - t1) + (t3 - t4)) / 2 (iPhone minus Mac) and
    round trip = (t4 - t1) - (t3 - t2). The offset is fitted as a + b (t - t_ref)
    over the pings with the smallest round trips (the lowest `keep` share in
    each `window_s` window), whose one-way delays are the most symmetric.
    The worst case for a single ping's offset error is half its round trip.
    """

    def __init__(self, t1, t2, t3, t4, window_s=10.0, keep=0.2):
        t1, t2, t3, t4 = (np.asarray(v, dtype=np.int64) for v in (t1, t2, t3, t4))
        two_theta = (t2 - t1) + (t3 - t4)  # exact in int64
        rtt = (t4 - t1) - (t3 - t2)
        tm = t1 + (t4 - t1) // 2
        self.t_ref = int(tm[0])
        self.theta0 = int(np.median(two_theta)) // 2
        x = (tm - self.t_ref) / 1e9  # s
        y = (two_theta - 2 * self.theta0) / 2.0  # ns, small
        self.rtt_ms = rtt / 1e6
        sel = np.zeros(len(x), dtype=bool)
        for w0 in np.arange(x.min(), x.max() + window_s, window_s):
            i = np.where((x >= w0) & (x < w0 + window_s) & (rtt > 0))[0]
            if len(i):
                sel[i[rtt[i] <= np.percentile(rtt[i], keep * 100)]] = True
        a, b = self._fit(x[sel], y[sel])
        r = y[sel] - (a + b * x[sel])
        mad = np.median(np.abs(r - np.median(r))) * 1.4826
        if mad > 0:  # one robust pass: drop selected pings far off the line
            idx = np.where(sel)[0]
            sel[idx[np.abs(r) > 4 * mad + 1e3]] = False
            a, b = self._fit(x[sel], y[sel])
            r = y[sel] - (a + b * x[sel])
        self.a, self.b = a, b  # ns, ns per s
        self.n_pings, self.n_selected = len(x), int(sel.sum())
        self.residual_std_ms = float(np.std(r) / 1e6) if len(r) > 1 else float("nan")
        self.span_s = float(x.max() - x.min()) if len(x) else 0.0
        self.rtt_selected_ms = self.rtt_ms[sel]

    @staticmethod
    def _fit(x, y):
        if len(x) >= 2 and np.ptp(x) > 1.0:
            b, a = np.polyfit(x, y, 1)
            return float(a), float(b)
        return float(np.mean(y)) if len(y) else 0.0, 0.0

    def offset_ns(self, mac_ns):
        """Fitted iPhone-minus-Mac offset at Mac time mac_ns."""
        return self.theta0 + self.a + self.b * (np.asarray(mac_ns, dtype=np.int64) - self.t_ref) / 1e9

    def to_mac(self, iphone_ns):
        """iPhone ns -> Mac ns (int64)."""
        d = np.asarray(iphone_ns, dtype=np.int64) - self.t_ref - self.theta0  # exact
        u = (d - self.a) / (1 + self.b / 1e9)
        return self.t_ref + np.round(u).astype(np.int64)

    def report(self):
        return {
            "pings": self.n_pings,
            "selected": self.n_selected,
            "span_s": self.span_s,
            "offset_s": (self.theta0 + self.a) / 1e9,
            "drift_ppm": self.b / 1e3,
            "rtt_ms": pct(self.rtt_ms[self.rtt_ms > 0], ("min", "median", "p95", "max")),
            "residual_std_ms": self.residual_std_ms,
            # Offset error of one ping is at most half its round trip (all delay on
            # one leg). Best case: half the smallest round trip. Bound: half the
            # largest round trip among the pings the line was fitted to.
            "uncertainty_ms": float(np.min(self.rtt_ms[self.rtt_ms > 0]) / 2) if np.any(self.rtt_ms > 0) else None,
            "bound_ms": float(np.max(self.rtt_selected_ms) / 2) if len(self.rtt_selected_ms) else None,
        }


# ---- trackers as time series ---------------------------------------------------


class Track:
    """One tracker's poses. t_sample: the time the pose describes, as the
    tracker stamps it (s); t_avail: when an app could first use it (s);
    q: [n, 4]; valid: bool [n]. Sorted by t_sample."""

    def __init__(self, name, t_sample, t_avail, q, valid, detail=None):
        t_sample = np.asarray(t_sample, dtype=float)
        order = np.argsort(t_sample, kind="stable")
        self.name = name
        self.t_sample = t_sample[order]
        self.t_avail = np.asarray(t_avail, dtype=float)[order]
        q = np.asarray(q, dtype=float)[order]
        valid = np.asarray(valid, dtype=bool)[order] & np.all(np.isfinite(q), axis=-1)
        norm = np.linalg.norm(q, axis=-1)
        valid &= norm > 0.5
        q = np.where(valid[:, None], q / np.where(valid, norm, 1.0)[:, None], np.nan)
        self.q, self.valid = q, valid
        self.detail = detail or {}

    def __len__(self):
        return len(self.t_sample)


def track_airpods(s):
    a = s.airpods
    q = np.stack([a["qx"], a["qy"], a["qz"], a["qw"]], axis=-1).astype(float)
    return Track("airpods", s.seconds(a["timestamp_ns"]), s.seconds(a["arrival_ns"]), q, np.ones(a.n, bool))


def track_reference(s, clock):
    r = s.ref
    q = np.stack([r["qx"], r["qy"], r["qz"], r["qw"]], axis=-1).astype(float)
    t_cap = s.seconds(clock.to_mac(r["capture_ns"]))
    t_del = s.seconds(clock.to_mac(r["delivered_ns"]))
    return Track("reference", t_cap, t_del, q, r["tracked"].astype(int) == 1)


MEDIAPIPE_VARIANTS = ("R", "R_T")


def track_mediapipe(s, variant="R"):
    """MediaPipe's facial transformation matrix -> rotation. Variants cover the
    convention the pilot must confirm: R as logged, or its transpose (the
    inverse rotation). A mirrored camera image needs no variant: it negates yaw
    and roll, which is R conjugated by a 180-degree turn about x, and the
    alignment's A and B absorb that."""
    m = s.mediapipe
    M = np.stack([m["m%d%d" % (r, c)].astype(float) for r in range(4) for c in range(4)], -1).reshape(-1, 4, 4)
    valid = (m["face"].astype(int) == 1) & np.all(np.isfinite(M.reshape(-1, 16)), axis=-1)
    layout = "translation_in_column_3"
    if valid.any():
        col = np.nanmedian(np.abs(M[valid, :3, 3]))
        row = np.nanmedian(np.abs(M[valid, 3, :3]))
        if row > col:  # translation stored in the last row: the matrix is transposed
            M = np.swapaxes(M, -1, -2)
            layout = "translation_in_row_3_transposed"
    R = np.full((len(M), 3, 3), np.nan)
    if valid.any():
        R[valid] = orthonormalize(M[valid, :3, :3])
    if variant == "R_T":
        R = np.swapaxes(R, -1, -2)
    q = np.full((len(M), 4), np.nan)
    if valid.any():
        q[valid] = quat_from_matrix(R[valid])
    scale = np.linalg.norm(M[valid, :3, 0], axis=-1) if valid.any() else np.array([np.nan])
    return Track("mediapipe", s.seconds(m["pts_ns"]), s.seconds(m["done_ns"]), q, valid,
                 {"variant": variant, "matrix_layout": layout, "matrix_scale_median": float(np.nanmedian(scale))})


VISION_FLIPS = tuple(itertools.product((1, -1), repeat=3))


def track_vision(s, flips=(1, 1, 1)):
    """Apple Vision roll/yaw/pitch -> q = Ry(sy yaw) Rx(sp pitch) Rz(sr roll).
    Sign flips (sy, sp, sr) cover the unknown convention. Flips that differ by
    an even number of signs are equivalent under alignment (conjugation by a
    180-degree turn), so the search really decides the parity: whether one
    axis (or all three) runs the other way."""
    c = s.camera
    yaw, pitch, roll = (c[k].astype(float) for k in ("yaw_deg", "pitch_deg", "roll_deg"))
    valid = (c["face"].astype(int) == 1) & np.isfinite(yaw)
    pitch_missing = bool(valid.any() and not np.isfinite(pitch[valid]).any())
    pitch = np.where(np.isfinite(pitch), pitch, 0.0)
    roll = np.where(np.isfinite(roll), roll, 0.0)
    sy, sp, sr = flips
    q = quat_from_euler_yxz(sy * yaw / DEG, sp * pitch / DEG, sr * roll / DEG)
    return Track("vision", s.seconds(c["pts_ns"]), s.seconds(c["vision_done_ns"]), q, valid,
                 {"flips": list(flips), "pitch_missing": pitch_missing})


def track_marker(s):
    """Marker tent (marker.csv) -> Track, using the camera calibration for the
    session's resolution (marker_pose.intrinsics_path). None without one."""
    import marker_pose  # needs OpenCV; only imported for sessions with a marker log

    cam = s.meta.get("camera", {})
    ip = marker_pose.intrinsics_path(s.path, int(cam.get("width", 1280)), int(cam.get("height", 720)))
    if ip is None:
        return None
    K, dist, info = marker_pose.load_intrinsics(ip)
    p = marker_pose.poses_from_csv(s.marker, marker_pose.load_tent(), K, dist)
    return Track("marker", s.seconds(p["pts_ns"]), s.seconds(p["done_ns"]), p["q"], p["valid"],
                 {"intrinsics": os.path.basename(ip), "calib_rms_px": info.get("rms_px"),
                  "two_face_share": float(np.mean(p["faces"][p["valid"]] >= 2)) if p["valid"].any() else None,
                  "pose_rms_px_median": float(np.nanmedian(p["rms"])) if len(p["rms"]) else None})


# ---- resampling and speed ------------------------------------------------------


def resample(track, grid, max_gap, shift=0.0):
    """Pose at each grid time (on the sample-time axis, at grid + shift) by
    slerp between neighbouring valid samples. Invalid where a neighbour is
    missing or the two samples are more than max_gap seconds apart."""
    tv = track.t_sample[track.valid]
    qv = track.q[track.valid]
    Q = np.full((len(grid), 4), np.nan)
    V = np.zeros(len(grid), dtype=bool)
    if len(tv) < 2:
        return Q, V
    g = np.asarray(grid, dtype=float) + shift
    i0 = np.searchsorted(tv, g, side="right") - 1
    ok = (i0 >= 0) & (i0 < len(tv) - 1)
    i0 = np.clip(i0, 0, len(tv) - 2)
    i1 = i0 + 1
    dt = tv[i1] - tv[i0]
    ok &= (dt > 0) & (dt <= max_gap)
    u = (g - tv[i0]) / np.where(dt > 0, dt, 1.0)
    Q[ok] = slerp(qv[i0[ok]], qv[i1[ok]], u[ok])
    V[ok] = True
    return Q, V


def zoh(track, grid, max_age):
    """What an app polling at each grid time would hold: the newest valid
    sample (by sample time) that has arrived (t_avail <= t). Invalid if none
    or if it describes a moment more than max_age seconds ago.
    Returns (Q, V, age_s)."""
    m = track.valid
    ta, ts, qq = track.t_avail[m], track.t_sample[m], track.q[m]
    Q = np.full((len(grid), 4), np.nan)
    V = np.zeros(len(grid), dtype=bool)
    age = np.full(len(grid), np.nan)
    if not len(ta):
        return Q, V, age
    order = np.lexsort((ts, ta))
    ta, ts, qq = ta[order], ts[order], qq[order]
    newest = np.maximum.accumulate(np.where(ts >= np.maximum.accumulate(ts), np.arange(len(ts)), 0))
    j = np.searchsorted(ta, grid, side="right") - 1
    ok = j >= 0
    k = newest[np.clip(j, 0, None)]
    a = grid - ts[k]
    ok &= a <= max_age
    Q[ok] = qq[k[ok]]
    V[ok] = True
    age[ok] = a[ok]
    return Q, V, age


def windowed_speed(Q, V, dt, window_s):
    """Angular speed (rad/s) as the angle between poses window_s apart."""
    h = max(1, int(round(window_s / (2 * dt))))
    n = len(Q)
    sp = np.full(n, np.nan)
    ok = np.zeros(n, dtype=bool)
    if n > 2 * h:
        v = V[: n - 2 * h] & V[2 * h:]
        ang = np.full(n - 2 * h, np.nan)
        ang[v] = qangle(Q[: n - 2 * h][v], Q[2 * h:][v])
        sp[h: n - h] = ang / (2 * h * dt)
        ok[h: n - h] = v
    return sp, ok


def estimate_lag(ref, ref_ok, trk, trk_ok, dt, lag_min=-0.05, lag_max=0.5, min_overlap=2000):
    """Lag L (s) at which trk(t) best matches ref(t - L): Pearson correlation over
    the samples valid in both, for every lag on the grid (FFT sums), then a
    parabola through the peak. Returns (lag_s, peak_corr, lags_s, corr)."""
    r = np.where(ref_ok, ref, 0.0)
    s = np.where(trk_ok, trk, 0.0)
    mr, ms = ref_ok.astype(float), trk_ok.astype(float)
    n = len(r)
    nfft = 1 << int(math.ceil(math.log2(2 * n)))
    lags = np.arange(int(round(lag_min / dt)), int(round(lag_max / dt)) + 1)

    def xc(a, b):  # sum_k a[k] b[k + L]
        c = np.fft.irfft(np.conj(np.fft.rfft(a, nfft)) * np.fft.rfft(b, nfft), nfft)
        return c[lags % nfft]

    N = np.round(xc(mr, ms))
    Sr, Ss, Srs = xc(r, ms), xc(mr, s), xc(r, s)
    Srr, Sss = xc(r * r, ms), xc(mr, s * s)
    with np.errstate(invalid="ignore", divide="ignore"):
        cov = Srs - Sr * Ss / N
        corr = cov / np.sqrt((Srr - Sr**2 / N) * (Sss - Ss**2 / N))
    corr[N < min_overlap] = np.nan
    if not np.isfinite(corr).any():
        return float("nan"), float("nan"), lags * dt, corr
    p = int(np.nanargmax(corr))
    frac = 0.0
    if 0 < p < len(corr) - 1 and np.isfinite(corr[p - 1]) and np.isfinite(corr[p + 1]):
        den = corr[p - 1] - 2 * corr[p] + corr[p + 1]
        if den < 0:
            frac = 0.5 * (corr[p - 1] - corr[p + 1]) / den
    return float((lags[p] + frac) * dt), float(corr[p]), lags * dt, corr


# ---- alignment -------------------------------------------------------------------


def align(q_ref, q_trk, valid, dt, offsets_s=(0.25, 0.5, 1.0, 2.0), min_deg=3.0, max_deg=120.0):
    """Constant A, B with q_ref = A q_trk B (both series on one grid).

    B from relative rotations, which do not depend on A:
    q_ref(i)^-1 q_ref(j) = B^-1 (q_trk(i)^-1 q_trk(j)) B, so the rotation vectors
    obey r_ref = R(B)^T r_trk; Kabsch gives R(B)^T. Then A is the average of
    q_ref (q_trk B)^-1. Returns dict with A, B, residual (deg) and conditioning."""
    P, Qv = [], []
    for off in offsets_s:
        k = max(1, int(round(off / dt)))
        if len(valid) <= k:
            continue
        i = np.where(valid[:-k] & valid[k:])[0]
        rr = qto_rotvec(qmul(qconj(q_ref[i]), q_ref[i + k]))
        rt = qto_rotvec(qmul(qconj(q_trk[i]), q_trk[i + k]))
        ang = np.linalg.norm(rr, axis=-1) * DEG
        use = (ang > min_deg) & (ang < max_deg)
        P.append(rt[use])
        Qv.append(rr[use])
    if not P or sum(len(p) for p in P) < 10:
        return None
    P, Qv = np.concatenate(P), np.concatenate(Qv)
    with np.errstate(all="ignore"):  # see qaverage
        H = P.T @ Qv  # sum p q^T
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    M = Vt.T @ np.diag([1, 1, d]) @ U.T  # r_ref ~ M r_trk
    B = quat_from_matrix(M.T)
    v = np.where(valid)[0]
    A = qaverage(qmul(q_ref[v], qconj(qmul(q_trk[v], B))))
    res = qangle(q_ref[v], qmul(qmul(A, q_trk[v]), B)) * DEG
    return {
        "A": A,
        "B": B,
        "residual_deg": pct(res, ("median", "p95", "rms")),
        "pairs": int(len(P)),
        "conditioning": float(S[2] / S[0]) if S[0] > 0 else 0.0,
    }


# ---- statistics ------------------------------------------------------------------


def pct(x, keys=("median", "p95", "p99", "max")):
    """Summary of a sample; numpy linear percentiles. None for an empty sample."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    out = {"n": int(len(x))}
    for k in keys:
        if not len(x):
            out[k] = None
        elif k == "min":
            out[k] = float(x.min())
        elif k == "max":
            out[k] = float(x.max())
        elif k == "mean":
            out[k] = float(x.mean())
        elif k == "rms":
            out[k] = float(np.sqrt(np.mean(x**2)))
        elif k == "median":
            out[k] = float(np.median(x))
        elif k.startswith("p"):
            out[k] = float(np.percentile(x, float(k[1:])))
    return out


def error_stats(e):
    """Per-axis error (tracker minus reference): MAE, RMS, bias and Bland-Altman
    95% limits (bias +- 1.96 SD). Samples are autocorrelated; n is not the
    number of independent observations."""
    e = np.asarray(e, dtype=float)
    e = e[np.isfinite(e)]
    if len(e) < 2:
        return {"n": int(len(e))}
    bias, sd = float(e.mean()), float(e.std(ddof=1))
    return {"n": int(len(e)), "mae": float(np.mean(np.abs(e))), "rms": float(np.sqrt(np.mean(e**2))),
            "bias": bias, "sd": sd, "loa_low": bias - 1.96 * sd, "loa_high": bias + 1.96 * sd}


def runs(mask, t, min_dur):
    """(start, end) times of runs of True in mask lasting at least min_dur."""
    out = []
    m = np.concatenate([[False], mask, [False]]).astype(int)
    d = np.diff(m)
    for a, b in zip(np.where(d == 1)[0], np.where(d == -1)[0]):
        if t[b - 1] - t[a] >= min_dur:
            out.append((float(t[a]), float(t[b - 1])))
    return out


def head_angles(q, q_neutral):
    """Head-relative (yaw, pitch, roll) in degrees, [n, 3]."""
    y, p, r = euler_yxz_from_quat(qmul(qconj(q_neutral), q))
    return np.stack([y, p, r], axis=-1) * DEG


def delivery_stats(track):
    out = {"samples": len(track), "valid_share": float(track.valid.mean()) if len(track) else None}
    ts = np.sort(track.t_sample)
    if len(ts) >= 3:
        d = np.diff(ts) * 1e3
        med = float(np.median(d))
        out["rate_hz"] = float((len(ts) - 1) / (ts[-1] - ts[0])) if ts[-1] > ts[0] else None
        out["sample_interval_ms"] = pct(d)
        out["intervals_over_2_5x_median"] = int(np.sum(d > 2.5 * med))
        da = np.diff(np.sort(track.t_avail[np.isfinite(track.t_avail)])) * 1e3
        out["arrival_interval_ms"] = pct(da)
        out["arrivals_within_1ms_share"] = float(np.mean(da < 1.0)) if len(da) else None
    return out


def still_noise(track, q_neutral, A, B, lag, periods, margin=0.1):
    """Noise at rest from the tracker's own samples: within each still period,
    head-relative angles minus a straight-line fit, pooled RMS per axis (deg)."""
    resid = [[], [], []]
    t_phys = track.t_sample - lag
    for a, b in periods:
        sel = track.valid & (t_phys >= a + margin) & (t_phys <= b - margin)
        if sel.sum() < 5:
            continue
        q = track.q[sel]
        if A is not None:
            q = qmul(qmul(A, q), B)
        ang = head_angles(q, q_neutral)
        x = t_phys[sel] - t_phys[sel].mean()
        for k in range(3):
            y = ang[:, k]
            coef = np.polyfit(x, y, 1)
            resid[k].append(y - np.polyval(coef, x))
    if not resid[0]:
        return None
    out = {}
    for k, name in enumerate(("yaw", "pitch", "roll")):
        r = np.concatenate(resid[k])
        dof = max(1, len(r) - 2 * len(resid[k]))
        out[name] = float(np.sqrt(np.sum(r**2) / dof))
    out["periods"] = len(resid[0])
    out["samples"] = int(sum(len(r) for r in resid[0]))
    return out


YAW_BINS = ((0, 15), (15, 30), (30, 45), (45, 60), (60, 180))
PITCH_BINS = ((0, 15), (15, 30), (30, 180))


def coverage(ref_ang, ref_ok, trk_ok):
    out = {"overall": float(trk_ok[ref_ok].mean()) if ref_ok.any() else None, "by_abs_yaw": {}, "by_abs_pitch": {}}
    for key, col, bins in (("by_abs_yaw", 0, YAW_BINS), ("by_abs_pitch", 1, PITCH_BINS)):
        a = np.abs(ref_ang[:, col])
        for lo, hi in bins:
            sel = ref_ok & (a >= lo) & (a < hi)
            out[key]["%d-%d" % (lo, hi) if hi < 180 else ">%d" % lo] = {
                "share": float(trk_ok[sel].mean()) if sel.any() else None,
                "seconds": float(sel.sum() * ALIGN_DT),
            }
    return out


# ---- session analysis --------------------------------------------------------------

LATENCY_DT = 0.001  # grid for cross-correlation, s
ALIGN_DT = 0.01  # grid for alignment and errors, s
SPEED_WINDOW = 0.1  # s
STILL_DEG_S = 5.0
STILL_MIN_S = 0.5
MAX_GAP = {"reference": 0.05, "airpods": 0.06, "mediapipe": 0.1, "vision": 0.1, "marker": 0.1}
# Heading jumps (first seen in day 2's drift session, 2026-10-23): the
# headphones' attitude stepped by 84 deg between two samples while their own
# gyroscope and the reference showed the head still.
JUMP_MIN_DEG = 20.0  # attitude change between consecutive samples
JUMP_MAX_GYRO_DPS = 150.0  # fastest reported rotation rate within +-5 samples
JUMP_SKIP_S = 2.0  # attitude still settling after the stream starts


def heading_jumps(s):
    """Steps in the headphones' attitude that their own gyroscope does not
    explain: more than JUMP_MIN_DEG between consecutive samples at most 30 ms
    apart, while the fastest rotation rate within +-5 samples stays below
    JUMP_MAX_GYRO_DPS. Real head motion shows in the gyroscope (and in the
    reference); these are the headphones re-setting their heading. Fast head
    shakes (rates of 500 deg/s) give brief mismatches that leave no offset
    and are not counted. Returns [{"t_s" (seconds since t0), "deg", "gyro_dps"}]."""
    a = s.airpods
    if a is None or len(a["timestamp_ns"]) < 20:
        return []
    ta = s.seconds(a["timestamp_ns"])
    q = qnormalize(np.stack([a["qx"], a["qy"], a["qz"], a["qw"]], -1).astype(float))
    rate = np.degrees(np.linalg.norm(np.stack([a["rot_x"], a["rot_y"], a["rot_z"]], -1).astype(float), axis=1))
    step = np.degrees(np.linalg.norm(qto_rotvec(qmul(qconj(q[:-1]), q[1:])), axis=1))
    near = np.lib.stride_tricks.sliding_window_view(np.pad(rate, 5, mode="edge"), 11).max(axis=1)[:-1]
    dts = np.diff(ta)
    hit = ((step > JUMP_MIN_DEG) & (near < JUMP_MAX_GYRO_DPS) & (dts > 0) & (dts < 0.03)
           & (ta[:-1] - ta[0] > JUMP_SKIP_S))
    return [{"t_s": float(ta[i + 1]), "deg": float(step[i]), "gyro_dps": float(near[i])} for i in np.where(hit)[0]]


# ---- protocol segments -----------------------------------------------------------
#
# The protocol prompts (events.csv, kind "step") say what the head was doing.
# Latency is estimated separately on impulsive steps (snaps, shakes) and on
# smooth steps (slow sweeps): predictive sensor fusion can make smooth motion
# look earlier than it was, even negative (Meyer-Kahlen et al. 2023).

STEP_CLASS_WORDS = (
    ("impulsive", ("snap", "quick", "shake", "sharp", "jerk")),
    ("range", ("as far",)),
    ("smooth", ("slow", "sweep")),
    ("static", ("hold still", "hold", "face the screen")),
)


def step_class(text):
    """Movement class of one prompt: impulsive, range, smooth, static or None.
    Earlier classes win ('Hold still, then snap your head' is impulsive)."""
    low = text.lower()
    for cls, words in STEP_CLASS_WORDS:
        if any(w in low for w in words):
            return cls
    return None


def segment_windows(steps, margin_s=0.5):
    """class -> [(t0, t1)] in session seconds. The first margin_s of each step
    is dropped: the subject is still taking in the prompt."""
    out = {}
    for t, dur, text in steps:
        cls = step_class(text)
        if cls is None or dur <= margin_s:
            continue
        out.setdefault(cls, []).append((t + margin_s, t + dur))
    return out


def window_mask(grid, windows):
    m = np.zeros(len(grid), dtype=bool)
    for a, b in windows:
        m |= (grid >= a) & (grid <= b)
    return m


def _neutral(s, ref, still_periods):
    for t, dur, text in s.steps():
        if "hold still" in text.lower():
            sel = ref.valid & (ref.t_sample >= t + 1.0) & (ref.t_sample <= t + dur - 0.5)
            if sel.sum() >= 10:
                return qaverage(ref.q[sel]), "protocol step '%s' at %.1f s" % (text, t)
    for a, b in still_periods:
        if b - a >= 1.0:
            sel = ref.valid & (ref.t_sample >= a) & (ref.t_sample <= b)
            if sel.sum() >= 10:
                return qaverage(ref.q[sel]), "first still period at %.1f s" % a
    sel = np.where(ref.valid)[0][:60]
    return qaverage(ref.q[sel]), "first reference samples (no still period found)"


def _tracker_variants(s):
    """name -> list of candidate Tracks (convention variants) for present sources."""
    out = {}
    if s.airpods is not None and s.airpods.n >= 3:
        out["airpods"] = [track_airpods(s)]
    if s.mediapipe is not None and s.mediapipe.n >= 3 and (s.mediapipe["face"].astype(int) == 1).any():
        out["mediapipe"] = [track_mediapipe(s, v) for v in MEDIAPIPE_VARIANTS]
    if (s.camera is not None and s.camera.n >= 3 and "face" in s.camera
            and (s.camera["face"].astype(int) == 1).any() and np.isfinite(s.camera["yaw_deg"].astype(float)).any()):
        out["vision"] = [track_vision(s, f) for f in VISION_FLIPS]
    if s.marker is not None and s.marker.n >= 3:
        mk = track_marker(s)
        if mk is not None and mk.valid.sum() >= 3:
            out["marker"] = [mk]
    return out


def _internal_delays(s, name):
    """Delays measured inside one device, in ms (no cross-clock mapping)."""
    ms = lambda a, b: (np.asarray(b, dtype=np.int64) - np.asarray(a, dtype=np.int64)) / 1e6
    keys = ("median", "p95", "p99", "max", "mean")
    if name == "airpods":
        a = s.airpods
        return {"timestamp_to_arrival": pct(ms(a["timestamp_ns"], a["arrival_ns"]), keys)}
    if name == "mediapipe":
        m = s.mediapipe
        return {
            "capture_to_done": pct(ms(m["pts_ns"], m["done_ns"]), keys),
            "capture_to_delivered": pct(ms(m["pts_ns"], m["delivered_ns"]), keys),
            "delivered_to_shared": pct(ms(m["delivered_ns"], m["shared_ns"]), keys),
            "shared_to_woke": pct(ms(m["shared_ns"], m["woke_ns"]), keys),
            "woke_to_start": pct(ms(m["woke_ns"], m["start_ns"]), keys),
            "start_to_done": pct(ms(m["start_ns"], m["done_ns"]), keys),
            "frames_skipped_total": int(np.sum(m["skipped"])) if "skipped" in m else None,
        }
    if name == "vision":
        c = s.camera
        ok = c["vision_done_ns"] > 0
        return {
            "capture_to_done": pct(ms(c["pts_ns"][ok], c["vision_done_ns"][ok]), keys),
            "capture_to_delivered": pct(ms(c["pts_ns"], c["delivered_ns"]), keys),
            "vision_compute": pct(ms(c["vision_start_ns"][ok], c["vision_done_ns"][ok]), keys),
        }
    if name == "marker":
        m = s.marker
        return {"capture_to_done": pct(ms(m["pts_ns"], m["done_ns"]), keys),
                "detect": pct(ms(m["start_ns"], m["done_ns"]), keys)}
    return {}


# ---- laser check (headtrack/protocols/laser_check.md) ---------------------------
#
# Geometry: the head turns about a pivot taken midway between the ears. The
# screen centre is D mm in front of it. The laser sits h mm above the pivot and
# points along the head's forward direction. The screen may curve about a
# vertical axis (radius R mm; edges towards the viewer); pixels are evenly
# spaced along its surface. A yaw mark at arc length s from the centre lies at
# x = R sin(s/R), z = R (1 - cos(s/R)) and is hit at yaw atan2(x, D - z). A
# pitch mark on the centre column is hit at pitch p when its height above the
# centre mark is D tan(p) + h (1/cos(p) - 1). Both hold for any laser distance
# in front of the pivot.

_LASER_LABEL = re.compile(r"^(C)$|^([LRUD])(\d+(?:\.\d+)?)$")


def laser_label(label):
    """'R30' -> ('yaw', 30.0); 'L30' -> ('yaw', -30.0); 'U10' -> ('pitch', 10.0);
    'C' -> ('centre', 0.0). Right and up are positive."""
    m = _LASER_LABEL.match(label)
    if not m:
        raise ValueError("not a laser target label: %r" % label)
    if m.group(1):
        return "centre", 0.0
    d, deg = m.group(2), float(m.group(3))
    return ("yaw" if d in "LR" else "pitch"), (deg if d in "RU" else -deg)


def screen_radius(arc_mm, sagitta_mm):
    """Curvature radius (mm) from the gap at the centre under a straight edge laid
    across the ends of an arc of width arc_mm. inf for a flat screen."""
    if sagitta_mm <= 0:
        return math.inf
    lo, hi = arc_mm / math.pi, 1e9  # semicircle .. nearly flat; the gap shrinks as R grows
    if sagitta_mm > lo:
        raise ValueError("gap of %.0f mm is too large for a %.0f mm wide screen" % (sagitta_mm, arc_mm))
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mid * (1 - math.cos(arc_mm / (2 * mid))) > sagitta_mm:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _yaw_of_arc(s_mm, D, R):
    if math.isinf(R):
        return math.degrees(math.atan2(s_mm, D))
    return math.degrees(math.atan2(R * math.sin(s_mm / R), D - R * (1 - math.cos(s_mm / R))))


def _pitch_height(deg, D, h):
    p = math.radians(deg)
    return D * math.tan(p) + h * (1 / math.cos(p) - 1)


def _solve_increasing(f, target, lo, hi):
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if f(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def laser_geometry(D_mm, h_mm, sagitta_mm, screen_mm, screen_px):
    w_mm, h_scr = screen_mm
    return {"D_mm": float(D_mm), "h_mm": float(h_mm), "sagitta_mm": float(sagitta_mm),
            "radius_mm": screen_radius(w_mm, sagitta_mm), "screen_mm": [float(w_mm), float(h_scr)],
            "screen_px": [int(screen_px[0]), int(screen_px[1])],
            "centre_px": [screen_px[0] / 2.0, screen_px[1] / 2.0]}


def laser_mark_px(label, g):
    """Pixel position (x right, y down) of a target mark."""
    axis, deg = laser_label(label)
    cx, cy = g["centre_px"]
    mm_x = g["screen_mm"][0] / g["screen_px"][0]
    mm_y = g["screen_mm"][1] / g["screen_px"][1]
    if axis == "centre":
        return [cx, cy]
    if axis == "yaw":
        half = g["screen_mm"][0] / 2
        if abs(deg) >= _yaw_of_arc(half, g["D_mm"], g["radius_mm"]):
            raise ValueError("%s is off the screen" % label)
        s = _solve_increasing(lambda v: _yaw_of_arc(v, g["D_mm"], g["radius_mm"]), abs(deg), 0.0, half)
        return [cx + math.copysign(s, deg) / mm_x, cy]
    y = _pitch_height(deg, g["D_mm"], g["h_mm"])
    if abs(y) >= g["screen_mm"][1] / 2:
        raise ValueError("%s is off the screen" % label)
    return [cx, cy - y / mm_y]


def laser_mark_angle(label, px, g):
    """True head angle (deg, signed) at which the laser hits a mark drawn at px."""
    axis, _ = laser_label(label)
    cx, cy = g["centre_px"]
    if axis == "centre":
        return 0.0
    if axis == "yaw":
        s = (px[0] - cx) * g["screen_mm"][0] / g["screen_px"][0]
        return math.copysign(_yaw_of_arc(abs(s), g["D_mm"], g["radius_mm"]), s)
    y = (cy - px[1]) * g["screen_mm"][1] / g["screen_px"][1]
    return _solve_increasing(lambda p: _pitch_height(p, g["D_mm"], g["h_mm"]), y, -80.0, 80.0)


def laser_targets(labels, g):
    """Targets file content for the recorder (--targets) and the analysis."""
    marks = {}
    for lab in labels:
        px = laser_mark_px(lab, g)
        axis, _ = laser_label(lab)
        marks[lab] = {"px": [round(px[0], 1), round(px[1], 1)], "axis": axis,
                      "angle_deg": round(laser_mark_angle(lab, px, g), 3)}
    return {"geometry": g, "marks": marks}


def _step_target(text):
    m = re.match(r"^\[([A-Z0-9.]+)\] ", text)
    return m.group(1) if m else None


def laser_check(s, tracks, hold_s=3.0, max_spread_deg=1.0, min_samples=5):
    """Static accuracy of every tracker against the laser targets. tracks: name ->
    Track on the Mac clock. Each hold is the mean pose over the last hold_s of a
    step; its angle is measured from the centre pose, interpolated in time
    between the centre holds around it (so slow drift drops out). The rotation
    angle is frame-independent, so no alignment is needed. True angles come
    from the drawn pixel positions and the geometry in session.json."""
    tg = s.meta.get("targets") or {}
    g, marks = tg.get("geometry"), tg.get("marks") or {}
    if not g or not marks:
        return None
    holds = []
    for t, dur, text in s.steps():
        lab = _step_target(text)
        if lab in marks:
            holds.append((lab, t + dur - hold_s, t + dur))
    out = {"hold_s": hold_s, "max_spread_deg": max_spread_deg, "holds": len(holds), "trackers": {}}
    true = {lab: laser_mark_angle(lab, m["px"], g) for lab, m in marks.items()}
    for name, trk in tracks.items():
        rows = []  # label, t_mid, mean q, spread
        dropped = 0
        for lab, a, b in holds:
            sel = trk.valid & (trk.t_sample >= a) & (trk.t_sample <= b)
            if sel.sum() < min_samples:
                dropped += 1
                continue
            qm = qaverage(trk.q[sel])
            spread = float(np.max(qangle(trk.q[sel], qm)) * DEG)
            if spread > max_spread_deg:
                dropped += 1
                continue
            rows.append((lab, 0.5 * (a + b), qm))
        centre = [(t, q) for lab, t, q in rows if lab == "C"]
        if len(centre) < 1:
            out["trackers"][name] = {"error": "no usable centre hold"}
            continue
        ct = np.array([t for t, _ in centre])
        per = {}
        for lab, t, q in rows:
            if lab == "C":
                continue
            i = int(np.searchsorted(ct, t))
            if i == 0 or i == len(ct):
                qc = centre[0][1] if i == 0 else centre[-1][1]
            else:
                u = (t - ct[i - 1]) / (ct[i] - ct[i - 1])
                qc = slerp(centre[i - 1][1], centre[i][1], u)
            per.setdefault(lab, []).append(float(qangle(qc, q) * DEG))
        res = {"dropped_holds": dropped, "targets": {}}
        if len(centre) >= 2:
            cq = np.array([q for _, q in centre])
            res["centre_spread_deg"] = float(max(qangle(cq[i], cq[j]) * DEG
                                                 for i in range(len(cq)) for j in range(i + 1, len(cq))))
        for lab, v in sorted(per.items()):
            res["targets"][lab] = {"true_deg": abs(true[lab]), "measured_deg": float(np.mean(v)),
                                   "sd_deg": float(np.std(v)) if len(v) > 1 else None, "n": len(v)}
        for group, keep in (("yaw", lambda lab: marks[lab]["axis"] == "yaw"),
                            ("yaw_left", lambda lab: marks[lab]["axis"] == "yaw" and true[lab] < 0),
                            ("yaw_right", lambda lab: marks[lab]["axis"] == "yaw" and true[lab] > 0),
                            ("pitch", lambda lab: marks[lab]["axis"] == "pitch")):
            x = np.array([abs(true[lab]) for lab, v in per.items() if keep(lab) for _ in v])
            y = np.array([m for lab, v in per.items() if keep(lab) for m in v])
            if len(x) == 0:
                continue
            e = y - x
            res[group] = {"n": int(len(x)), "gain": float(np.sum(x * y) / np.sum(x * x)),
                          "mean_error_deg": float(e.mean()), "mae_deg": float(np.abs(e).mean()),
                          "rms_deg": float(np.sqrt(np.mean(e ** 2))), "max_abs_error_deg": float(np.abs(e).max())}
        out["trackers"][name] = res
    return out


def analyze_session(path, keep_series=False):
    """Analyse one session folder. Returns (result dict, series dict or None)."""
    s = Session(path)
    res = {"session": os.path.basename(os.path.normpath(path)), "t0_ns": s.t0, "notes": [], "trackers": {}}
    series = {} if keep_series else None

    if s.headphones is not None and "connected_headphones" in s.headphones:
        res["headphones"] = s.headphones["connected_headphones"]
    else:
        res["notes"].append("headphone firmware not recorded (headphones.json missing or failed)")

    clock = None
    if s.sync is not None and s.sync.n >= 10:
        clock = ClockMap(s.sync["t1"], s.sync["t2"], s.sync["t3"], s.sync["t4"])
        res["clock_sync"] = clock.report()
    else:
        res["notes"].append("no clock sync (sync.csv missing or under 10 pings): reference not usable")

    ref = None
    if clock is not None and s.ref is not None and s.ref.n >= 10:
        ref = track_reference(s, clock)
        r = s.ref
        res["reference"] = {
            "delivery": delivery_stats(ref),
            "tracked_share": float(ref.valid.mean()),
            "fps": float(np.nanmedian(r["fps"])) if "fps" in r else None,
            "internal_delay_ms": {
                "capture_to_delivered": pct((r["delivered_ns"] - r["capture_ns"]) / 1e6),
                "sent_to_mac_receive": pct((r["recv_ns"] - clock.to_mac(r["sent_ns"])) / 1e6),
            },
        }
        if ref.valid.sum() < 100:
            res["notes"].append("reference has under 100 tracked poses: no latency or agreement")
            ref = None
    elif s.ref is None or s.ref.n < 10:
        res["notes"].append("no reference poses (ref.csv missing or empty)")

    variants = _tracker_variants(s)
    windows = segment_windows(s.steps())
    for name, tracks in variants.items():
        res["trackers"][name] = {"delivery": delivery_stats(tracks[0]), "internal_delay_ms": _internal_delays(s, name)}
    for name in ("airpods", "mediapipe", "vision"):
        if name not in variants:
            res["notes"].append("%s: no data" % name)
    if s.meta.get("targets"):  # laser check: static accuracy against known angles
        trk = {name: tracks[0] for name, tracks in variants.items()}
        if ref is not None:
            trk["reference"] = ref
        res["laser_check"] = laser_check(s, trk)
    if ref is None:
        return res, series

    # Reference on the alignment grid: angles, speed, still periods, neutral pose.
    t_lo, t_hi = ref.t_sample[ref.valid].min(), ref.t_sample[ref.valid].max()
    g10 = np.arange(t_lo, t_hi, ALIGN_DT)
    Qr, Vr = resample(ref, g10, MAX_GAP["reference"])
    sp10, ok10 = windowed_speed(Qr, Vr, ALIGN_DT, SPEED_WINDOW)
    still = ok10 & (sp10 * DEG < STILL_DEG_S)
    periods = runs(still, g10, STILL_MIN_S)
    still_grid = np.zeros(len(g10), dtype=bool)
    for a, b in periods:
        still_grid |= (g10 >= a) & (g10 <= b)
    moving = Vr & ok10 & ~still_grid
    q_neutral, neutral_src = _neutral(s, ref, periods)
    ref_ang = np.full((len(g10), 3), np.nan)
    ref_ang[Vr] = head_angles(Qr[Vr], q_neutral)
    res["neutral"] = {"source": neutral_src, "q": q_neutral.tolist()}
    res["still_periods"] = {"count": len(periods), "seconds": float(sum(b - a for a, b in periods)),
                            "threshold_deg_s": STILL_DEG_S, "min_duration_s": STILL_MIN_S}
    res["reference"]["noise_at_rest_deg"] = still_noise(ref, q_neutral, None, None, 0.0, periods)
    res["reference"]["angle_range_deg"] = {
        k: [float(np.nanmin(ref_ang[:, i])), float(np.nanmax(ref_ang[:, i]))] if Vr.any() else None
        for i, k in enumerate(("yaw", "pitch", "roll"))
    }
    if keep_series:
        series["t"] = g10
        series["reference"] = ref_ang
        series["still"] = still_grid

    # Headphone heading jumps: the drift fit stops 1 s before the first one.
    jumps = heading_jumps(s) if "airpods" in variants else []
    res["heading_jumps"] = jumps
    drift_end = jumps[0]["t_s"] - 1.0 if jumps else None

    # Reference speed on the latency grid (capture time = physical time).
    g1 = np.arange(t_lo, t_hi, LATENCY_DT)
    Qr1, Vr1 = resample(ref, g1, MAX_GAP["reference"])
    sp1, ok1 = windowed_speed(Qr1, Vr1, LATENCY_DT, SPEED_WINDOW)

    aligned = {}
    for name, tracks in variants.items():
        out = res["trackers"][name]
        trk = tracks[0]
        gap = MAX_GAP[name]
        # Latency: speed is unchanged by A, B, mirroring or inversion, so it is
        # estimated once, before choosing the convention variant.
        Qs, Vs = resample(trk, g1, gap)
        spt, okt = windowed_speed(Qs, Vs, LATENCY_DT, SPEED_WINDOW)
        lag_s, corr_s, _, _ = estimate_lag(sp1, ok1, spt, okt, LATENCY_DT)
        internal = (trk.t_avail - trk.t_sample)[trk.valid] * 1e3
        # App view: the newest arrived pose, while the tracker is tracking.
        # Holds older than the usual age plus four sample intervals (e.g. while
        # a webcam has lost the face) are left out of this estimate.
        interval = float(np.median(np.diff(trk.t_sample[trk.valid]))) if trk.valid.sum() > 2 else 0.05
        stale = (lag_s if np.isfinite(lag_s) else 0.0) + float(np.median(internal)) / 1e3 + 4 * interval
        Qz, Vz, _ = zoh(trk, g1, max_age=stale)
        spz, okz = windowed_speed(Qz, Vz, LATENCY_DT, SPEED_WINDOW)
        lag_a, corr_a, _, _ = estimate_lag(sp1, ok1, spz, okz, LATENCY_DT)
        out["latency_ms"] = {
            "sample_time_lag": lag_s * 1e3 if np.isfinite(lag_s) else None,
            "sample_time_corr": corr_s,
            "motion_to_data": pct(lag_s * 1e3 + internal, ("median", "p95", "p99", "max", "mean"))
            if np.isfinite(lag_s) else None,
            "app_view_lag": lag_a * 1e3 if np.isfinite(lag_a) else None,
            "app_view_corr": corr_a,
            "app_view_max_age": stale * 1e3,
        }
        # Lag on the impulsive and on the smooth protocol steps alone.
        segs = {}
        for cls in ("impulsive", "smooth"):
            win = windows.get(cls)
            if not win:
                continue
            m = window_mask(g1, win)
            l, c, _, _ = estimate_lag(sp1, ok1 & m, spt, okt & m, LATENCY_DT, min_overlap=1000)
            segs[cls] = {"sample_time_lag": l * 1e3 if np.isfinite(l) else None,
                         "corr": c if np.isfinite(c) else None,
                         "seconds": float(sum(b - a for a, b in win)), "steps": len(win)}
        if segs:
            out["latency_ms"]["by_segment"] = segs
        if not np.isfinite(lag_s):
            res["notes"].append("%s: latency not found (too little overlap with the reference)" % name)
            continue

        # Alignment, choosing the convention variant with the smallest residual.
        best = None
        tried = []
        for cand in tracks:
            Qt, Vt = resample(cand, g10, gap, shift=lag_s)
            both = Vr & Vt
            al = align(Qr, Qt, both, ALIGN_DT)
            if al is None:
                continue
            tried.append({**cand.detail, "residual_median_deg": al["residual_deg"]["median"]})
            if best is None or al["residual_deg"]["median"] < best[1]["residual_deg"]["median"]:
                best = (cand, al, Qt, Vt)
        if best is None:
            res["notes"].append("%s: alignment failed (too little movement overlapping the reference)" % name)
            continue
        cand, al, Qt, Vt = best
        A, B = al["A"], al["B"]
        out["convention"] = cand.detail
        if len(tracks) > 1:
            out["convention_candidates"] = tried
        out["alignment"] = {"A": A.tolist(), "B": B.tolist(), "residual_deg": al["residual_deg"],
                            "pairs": al["pairs"], "conditioning": al["conditioning"]}

        both = Vr & Vt
        q_hat = np.full_like(Qt, np.nan)
        q_hat[both] = qmul(qmul(A, Qt[both]), B)
        trk_ang = np.full((len(g10), 3), np.nan)
        aligned[name] = (q_hat, both)
        trk_ang[both] = head_angles(q_hat[both], q_neutral)
        err = wrap_deg(trk_ang - ref_ang)
        geo = np.full(len(g10), np.nan)
        geo[both] = qangle(Qr[both], q_hat[both]) * DEG
        agreement = {}
        for subset, mask in (("all", both), ("still", both & still_grid), ("moving", both & moving)):
            agreement[subset] = {k: error_stats(err[mask, i]) for i, k in enumerate(("yaw", "pitch", "roll"))}
            agreement[subset]["geodesic"] = pct(geo[mask], ("median", "p95", "rms", "mean"))
        out["agreement_latency_compensated_deg"] = agreement

        # What an app sees (no latency compensation): newest arrived sample.
        Qa, Va, age = zoh(cand, g10, max_age=0.5)
        okA = Vr & Va
        q_app = np.full_like(Qa, np.nan)
        q_app[okA] = qmul(qmul(A, Qa[okA]), B)
        app_ang = np.full((len(g10), 3), np.nan)
        app_ang[okA] = head_angles(q_app[okA], q_neutral)
        app_err = wrap_deg(app_ang - ref_ang)
        app_geo = np.full(len(g10), np.nan)
        app_geo[okA] = qangle(Qr[okA], q_app[okA]) * DEG
        app = {"pose_age_ms": pct(age[okA] * 1e3, ("median", "p95", "p99", "max", "mean"))}
        for subset, mask in (("all", okA), ("moving", okA & moving)):
            app[subset] = {k: error_stats(app_err[mask, i]) for i, k in enumerate(("yaw", "pitch", "roll"))}
            app[subset]["geodesic"] = pct(app_geo[mask], ("median", "p95", "rms", "mean"))
        out["agreement_as_seen_by_app_deg"] = app

        out["noise_at_rest_deg"] = still_noise(cand, q_neutral, A, B, lag_s, periods)

        # Drift: slope of the yaw error over time, from still periods if there
        # are at least 10 s of them, else from every sample.
        before = (g10 < drift_end) if (name == "airpods" and drift_end is not None) else np.ones(len(g10), bool)
        use = both & still_grid & before
        src = "still periods"
        if use.sum() * ALIGN_DT < 10:
            use, src = both & before, "all samples"
        if use.sum() >= 50 and np.ptp(g10[use]) > 10:
            x = g10[use] / 60.0
            y = err[use, 0]
            slope, icpt = np.polyfit(x, y, 1)
            pred = slope * x + icpt
            ss = np.sum((y - y.mean()) ** 2)
            out["yaw_drift"] = {"deg_per_min": float(slope), "r2": float(1 - np.sum((y - pred) ** 2) / ss) if ss > 0 else None,
                                "span_min": float(np.ptp(x)), "from": src,
                                "until_heading_jump_s": drift_end if (name == "airpods" and drift_end is not None) else None}
        out["coverage"] = coverage(ref_ang, Vr, Vt)
        if keep_series:
            series[name] = trk_ang
    if "marker" in aligned:
        res["marker_check"] = marker_scale_check(g10, Qr, Vr, ref_ang, aligned, periods)
    return res, series


def marker_scale_check(t, Qr, Vr, ref_ang, aligned, periods, min_hold_s=0.5, min_pair_deg=5.0):
    """Angle scale of the reference and of each tracker against the marker
    tent. Each still period gives one held pose per source (average
    quaternion). For every pair of held poses, the rotation angle between them
    is compared with the marker's; the slope through zero is the gain (1.10:
    the source reports turns 10 % larger than the marker). Angles between two
    poses do not depend on any source's frames, the alignment or the neutral
    pose. Pairs are also split by the reference's change in yaw vs pitch."""
    qm, vm = aligned["marker"]
    sources = {"reference": (Qr, Vr)}
    sources.update({k: v for k, v in aligned.items() if k != "marker"})
    holds = []  # (period index, {source: quaternion}), marker always present
    for a, b in periods:
        m = (t >= a) & (t <= b) & vm
        if m.sum() * (t[1] - t[0]) < min_hold_s:
            continue
        h = {"marker": qaverage(qm[m])}
        for name, (q, v) in sources.items():
            mm = m & v
            if mm.sum() * (t[1] - t[0]) >= min_hold_s:
                h[name] = qaverage(q[mm])
        hy = np.nanmedian(ref_ang[m, 0]) if np.isfinite(ref_ang[m, 0]).any() else np.nan
        hp = np.nanmedian(ref_ang[m, 1]) if np.isfinite(ref_ang[m, 1]).any() else np.nan
        holds.append((h, hy, hp))
    out = {"holds": len(holds), "min_pair_deg": min_pair_deg, "sources": {}}
    for name in sources:
        rows = []
        for i in range(len(holds)):
            for j in range(i + 1, len(holds)):
                hi, hj = holds[i][0], holds[j][0]
                if name not in hi or name not in hj:
                    continue
                am = qangle(hi["marker"], hj["marker"]) * DEG
                if am < min_pair_deg:
                    continue
                dy, dp = abs(holds[i][1] - holds[j][1]), abs(holds[i][2] - holds[j][2])
                rows.append((am, qangle(hi[name], hj[name]) * DEG, dy, dp))
        if len(rows) < 5:
            out["sources"][name] = None
            continue
        r = np.array(rows)
        d = {}
        for label, m in (("all", np.ones(len(r), bool)), ("yaw_pairs", r[:, 2] > 2 * r[:, 3]),
                         ("pitch_pairs", r[:, 3] > 2 * r[:, 2])):
            if m.sum() < 5:
                d[label] = None
                continue
            x, y = r[m, 0], r[m, 1]
            g = float(np.sum(x * y) / np.sum(x * x))
            d[label] = {"gain": g, "residual_rms_deg": float(np.sqrt(np.mean((y - g * x) ** 2))),
                        "pairs": int(m.sum()), "max_deg": float(x.max())}
        out["sources"][name] = d
    return out


def to_jsonable(x):
    """numpy -> plain Python; nan and inf -> None."""
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return to_jsonable(x.tolist())
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return float(x) if math.isfinite(x) else None
    return x


def plot_series(series, png_path, title=""):
    """Optional: head-relative yaw, pitch and roll of every tracker over time."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    t = series["t"]
    for i, ax in enumerate(axes):
        for name, style in (("reference", dict(color="black", lw=1.2)), ("airpods", dict(lw=0.8)),
                            ("mediapipe", dict(lw=0.8)), ("vision", dict(lw=0.8))):
            if name in series:
                ax.plot(t, series[name][:, i], label=name, **style)
        ax.set_ylabel(("yaw", "pitch", "roll")[i] + " (deg)")
        ax.grid(alpha=0.3)
    axes[0].legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("time (s)")
    axes[0].set_title(title)
    fig.tight_layout()
    fig.savefig(png_path, dpi=110)
    plt.close(fig)
