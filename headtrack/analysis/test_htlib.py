"""Tests for htlib.py on synthetic data with known truth (milestone H0 in
HEADTRACK_STUDY.md). The end-to-end test writes a session folder in the exact
CSV formats of HeadTrackRecorder and mp_live.py, then checks that
analyze_session recovers the injected clock offset, latencies, frame
rotations, noise, drift and coverage.

    headtrack/.venv/bin/python headtrack/analysis/test_htlib.py
    headtrack/.venv/bin/python headtrack/analysis/test_htlib.py --write-session <dir>
"""

import json
import math
import os
import shutil
import sys
import tempfile
import traceback

import numpy as np

import htlib as h

DEG = h.DEG

# ---- synthetic truth -------------------------------------------------------------

START_NS = 5_000_000_000_000  # Mac uptime at session start
IPHONE_OFFSET_NS = 1_234_567_890_123  # iPhone clock minus Mac clock at START
IPHONE_DRIFT = 20e-6  # iPhone clock runs 20 ppm fast
AIRPODS_LAG = 0.0473  # physical motion -> AirPods sample timestamp, s
AIRPODS_NOISE_DEG = 0.1
AIRPODS_DRIFT_DEG_MIN = 1.5  # about the AirPods world vertical
WEBCAM_LAG = 0.012  # physical motion -> camera pts, s
MP_DELAY = 0.100  # pts -> MediaPipe done (mean), s
MP_NOISE_DEG = 1.0
VISION_NOISE_DEG = 2.0
REF_NOISE_DEG = 0.05
A_AIR = h.qmul(h.qaxis(1, math.radians(37)), h.qaxis(0, math.radians(-90)))  # AirPods z (up) -> camera y (up)
B_AIR = h.quat_from_euler_yxz(math.radians(20), math.radians(-65), math.radians(110))
A_MP = h.qmul(h.qaxis(1, math.radians(4)), h.qaxis(0, math.radians(-3)))
B_MP = h.qaxis(0, math.radians(6))
VISION_FLIPS = (-1, 1, -1)  # Vision angles = these signs times the true head angles


def mac_ns(t):
    return START_NS + np.round(np.asarray(t, dtype=float) * 1e9).astype(np.int64)


def iphone_from_mac(m):
    m = np.asarray(m, dtype=np.int64)
    return m + IPHONE_OFFSET_NS + np.round(IPHONE_DRIFT * (m - START_NS)).astype(np.int64)


def mac_from_iphone(i):
    u = (np.asarray(i, dtype=np.int64) - START_NS - IPHONE_OFFSET_NS) / (1 + IPHONE_DRIFT)
    return START_NS + np.round(u).astype(np.int64)


def ease(u):
    return (1 - np.cos(np.pi * u)) / 2


class Trajectory:
    """Smooth head motion in the reference camera frame (neutral = identity):
    10 s still at neutral, then moves to random poses with holds, sweeps and
    quick shakes. Angles in degrees on a 1 ms grid."""

    def __init__(self, duration, rng):
        keys = [(0.0, 0.0, 0.0, 0.0), (10.0, 0.0, 0.0, 0.0)]
        t, cur = 10.0, (0.0, 0.0, 0.0)
        n = 0
        self.windows = {"smooth": [], "impulsive": []}  # (start, end) s of sweeps and shakes
        while t < duration:
            n += 1
            if n % 6 == 0:  # yaw sweeps
                w0 = t
                for k in range(4):
                    t += 0.6
                    cur = ((-1) ** k * 40.0, cur[1] * 0.5, 0.0)
                    keys.append((t, *cur))
                self.windows["smooth"].append((w0, t))
            elif n % 6 == 3:  # quick shakes
                w0 = t
                for k in range(4):
                    t += 0.15
                    keys.append((t, cur[0] + (-1) ** k * 12.0, cur[1], cur[2]))
                t += 0.15
                keys.append((t, *cur))
                self.windows["impulsive"].append((w0, t))
            else:
                t += rng.uniform(0.4, 1.8)
                cur = (rng.uniform(-70, 70), rng.uniform(-30, 25), rng.uniform(-20, 20))
                keys.append((t, *cur))
            t += rng.uniform(1.0, 4.0)
            keys.append((t, *cur))
        keys = np.array(keys)
        self.t = np.arange(0.0, duration, 0.001)
        self.ang = np.zeros((len(self.t), 3))
        for i in range(len(keys) - 1):
            a, b = keys[i], keys[i + 1]
            sel = (self.t >= a[0]) & (self.t < b[0])
            u = ease((self.t[sel] - a[0]) / (b[0] - a[0]))
            self.ang[sel] = a[1:] + (b[1:] - a[1:]) * u[:, None]
        last = self.t >= keys[-1][0]
        self.ang[last] = keys[-1][1:]

    def angles(self, t):
        return np.stack([np.interp(t, self.t, self.ang[:, k]) for k in range(3)], axis=-1)

    def q(self, t):
        a = self.angles(t) / DEG
        return h.quat_from_euler_yxz(a[:, 0], a[:, 1], a[:, 2])

    def speed(self, t, dt=0.01):
        return h.qangle(self.q(t - dt / 2), self.q(t + dt / 2)) / dt


def noise(rng, n, sigma_deg):
    return h.qfrom_rotvec(rng.normal(0.0, sigma_deg / DEG, (n, 3)))


def write_csv(path, header, rows):
    with open(path, "w") as f:
        f.write(header + "\n")
        for r in rows:
            f.write(",".join(r) + "\n")


def write_synthetic_session(path, duration=240.0, seed=7):
    """Writes a session folder and returns the truth needed by the tests."""
    rng = np.random.default_rng(seed)
    os.makedirs(path, exist_ok=True)
    traj = Trajectory(duration, rng)
    truth = {"traj": traj}

    with open(os.path.join(path, "session.json"), "w") as f:
        json.dump({"tool": "synthetic", "start_ns": START_NS}, f)
    # Protocol steps as the recorder logs them: a still step, then one prompt per
    # sweep ("slowly") or shake ("quick") window, spoken 0.5 s before the motion.
    prompts = [(0.0, 10.0, "Face the screen and hold still")]
    for cls, text in (("smooth", "Turn slowly left and right"), ("impulsive", "Quick head shakes, left and right")):
        for a, b in traj.windows[cls]:
            prompts.append((a - 0.5, b - a + 0.8, text))
    prompts.sort()
    write_csv(os.path.join(path, "events.csv"), "t_ns,kind,detail",
              [[str(mac_ns(t)), "step", "%d/%d %.1fs %s" % (i + 1, len(prompts), dur, text)]
               for i, (t, dur, text) in enumerate(prompts)])

    # Clock sync pings, 10 Hz, asymmetric random one-way delays.
    t1 = mac_ns(np.arange(0.2, duration - 0.2, 0.1))
    u1, u2 = rng.random(len(t1)), rng.random(len(t1))
    d1 = np.round(1e6 + 12e6 * u1**3).astype(np.int64)
    d2 = np.round(1e6 + 4e6 * u2**2).astype(np.int64)
    t2 = iphone_from_mac(t1 + d1)
    t3 = t2 + 50_000
    t4 = mac_from_iphone(t3) + d2
    write_csv(os.path.join(path, "sync.csv"), "seq,t1,t2,t3,t4",
              [[str(i + 1), str(a), str(b), str(c), str(d)] for i, (a, b, c, d) in enumerate(zip(t1, t2, t3, t4))])

    # Reference: ARKit at 60 Hz, capture time = physical time.
    p = np.arange(0.05, duration - 0.1, 1 / 60)
    ang = traj.angles(p)
    tracked = np.abs(ang[:, 0]) < 80
    q = h.qmul(traj.q(p), noise(rng, len(p), REF_NOISE_DEG))
    cap = iphone_from_mac(mac_ns(p))
    dlv = cap + np.round(18e6 + 4e6 * rng.random(len(p))).astype(np.int64)
    sent = dlv + 300_000
    recv = mac_from_iphone(sent) + np.round(3e6 + rng.exponential(1.5e6, len(p))).astype(np.int64)
    rows = []
    for i in range(len(p)):
        qq = q[i] if tracked[i] else np.array([0, 0, 0, 1.0])
        rows.append([str(i), str(cap[i]), str(dlv[i]), str(sent[i]), str(recv[i]), str(int(tracked[i])), "60.0",
                     *("%.9f" % v for v in qq), "0.010000", "-0.020000", "-0.450000"])
    write_csv(os.path.join(path, "ref.csv"), "seq,capture_ns,delivered_ns,sent_ns,recv_ns,tracked,fps,qx,qy,qz,qw,px,py,pz",
              rows)

    # AirPods: 50 Hz timestamps, bursty batched arrivals, drift about world z.
    ts = np.arange(0.3, duration - 0.3, 0.02)
    pa = ts - AIRPODS_LAG
    drift = np.radians(AIRPODS_DRIFT_DEG_MIN / 60.0 * pa)
    qa = h.qmul(h.qmul(h.qmul(h.qaxis(2, drift), h.qconj(A_AIR)), traj.q(pa)), h.qconj(B_AIR))
    qa = h.qmul(qa, noise(rng, len(ts), AIRPODS_NOISE_DEG))
    arrival = np.zeros(len(ts))
    b, nxt = 0.3, 0
    while nxt < len(ts):
        b += rng.uniform(0.02, 0.06) if rng.random() < 0.8 else rng.uniform(0.06, 0.11)
        k = 0
        while nxt < len(ts) and ts[nxt] + 0.025 <= b:
            arrival[nxt] = b + 1e-5 * k
            nxt += 1
            k += 1
    truth["airpods"] = {"t_phys": pa, "t_avail": arrival, "mean_drift_rad": None}
    write_csv(os.path.join(path, "airpods.csv"),
              "arrival_ns,timestamp_ns,qx,qy,qz,qw,rot_x,rot_y,rot_z,grav_x,grav_y,grav_z,acc_x,acc_y,acc_z,sensor_location",
              [[str(mac_ns(arrival[i])), str(mac_ns(ts[i])), *("%.9f" % v for v in qa[i]), *(["0.000000"] * 9), "1"]
               for i in range(len(ts))])

    # Mac camera at 30 Hz: MediaPipe (90% of frames processed) and Vision.
    pts = 0.5 + np.arange(int((duration - 1.0) * 30)) / 30 + rng.normal(0, 0.0002, int((duration - 1.0) * 30))
    pc = pts - WEBCAM_LAG
    angc = traj.angles(pc)
    qc = traj.q(pc)
    mp_rows, cam_rows = [], []
    last_done = -1
    processed = rng.random(len(pts)) < 0.9
    q_mp = h.qmul(h.qmul(h.qmul(h.qconj(A_MP), qc), h.qconj(B_MP)), noise(rng, len(pts), MP_NOISE_DEG))
    R_mp = h.matrix_from_quat(q_mp)
    mp_avail, mp_phys = [], []
    for k in range(len(pts)):
        dlv_t = pts[k] + 0.020
        if processed[k]:
            shared = dlv_t + 0.0008
            woke = shared + 0.0002
            start = woke + 0.0001
            done = pts[k] + MP_DELAY + rng.uniform(-0.005, 0.005)
            face = abs(angc[k, 0]) <= 45
            if face:
                M = np.eye(4)
                M[:3, :3] = R_mp[k]
                M[:3, 3] = (1.2, -3.4, -50.0)
                mat = ",".join("%.7g" % v for v in M.reshape(16))
                mp_avail.append(done)
                mp_phys.append(pc[k])
            else:
                mat = ",".join([""] * 16)
            skipped = k - last_done - 1 if last_done >= 0 else 0
            last_done = k
            mp_rows.append([str(k + 1), str(mac_ns(pts[k])), str(mac_ns(dlv_t)), str(mac_ns(shared)), str(mac_ns(woke)),
                            str(mac_ns(start)), str(mac_ns(done)), str(skipped), str(int(face)), mat])
        v0 = dlv_t + 0.001
        v1 = v0 + 0.006 + rng.uniform(-0.001, 0.001)
        if abs(angc[k, 0]) <= 50:
            yv, pv, rv = (VISION_FLIPS[i] * angc[k, i] + rng.normal(0, VISION_NOISE_DEG) for i in range(3))
            vis = "%d,%d,1,%.4f,%.4f,%.4f,0.40000,0.30000,0.20000,0.30000,0.9900" % (mac_ns(v0), mac_ns(v1), yv, pv, rv)
        else:
            vis = "%d,%d,0,,,,,,,," % (mac_ns(v0), mac_ns(v1))
        cam_rows.append([str(k), str(mac_ns(pts[k])), str(mac_ns(dlv_t)), vis])
    write_csv(os.path.join(path, "mediapipe.csv"),
              "seq,pts_ns,delivered_ns,shared_ns,woke_ns,start_ns,done_ns,skipped,face,"
              + ",".join("m%d%d" % (r, c) for r in range(4) for c in range(4)), mp_rows)
    write_csv(os.path.join(path, "camera.csv"),
              "seq,pts_ns,delivered_ns,vision_start_ns,vision_done_ns,face,yaw_deg,pitch_deg,roll_deg,"
              "box_x,box_y,box_w,box_h,confidence", cam_rows)
    truth["mediapipe"] = {"t_phys": np.array(mp_phys), "t_avail": np.array(mp_avail)}
    return truth


def mean_age(truth_track, t_lo, t_hi, max_age):
    """Truth for the app-view lag: time-averaged age of the newest arrived
    sample, over the times it is under max_age. With bursty or skipped samples
    the age varies frame to frame, so any single lag is a summary; other
    weightings (by head speed or its rate of change) move it by 1 to 3 ms."""
    g = np.arange(t_lo, t_hi, 0.001)
    order = np.argsort(truth_track["t_avail"], kind="stable")
    ta, tp = truth_track["t_avail"][order], truth_track["t_phys"][order]
    newest = np.maximum.accumulate(tp)
    j = np.searchsorted(ta, g, side="right") - 1
    ok = j >= 0
    age = g[ok] - newest[j[ok]]
    return float(age[age < max_age].mean())


# ---- tests ---------------------------------------------------------------------------


def test_quaternion_algebra():
    rng = np.random.default_rng(1)
    a = h.qnormalize(rng.normal(size=(200, 4)))
    b = h.qnormalize(rng.normal(size=(200, 4)))
    c = h.qnormalize(rng.normal(size=(200, 4)))
    ident = h.qmul(a, h.qconj(a))
    assert np.allclose(np.abs(ident[:, 3]), 1, atol=1e-12)
    assert np.allclose(h.qmul(h.qmul(a, b), c), h.qmul(a, h.qmul(b, c)), atol=1e-12)
    assert np.allclose(h.matrix_from_quat(h.qmul(a, b)), h.matrix_from_quat(a) @ h.matrix_from_quat(b), atol=1e-12)
    back = h.quat_from_matrix(h.matrix_from_quat(a))
    assert np.all(h.qangle(a, back) < 1e-9)
    v = rng.normal(size=(200, 3))
    v *= (rng.uniform(0, math.pi * 0.999, 200) / np.linalg.norm(v, axis=1))[:, None]
    assert np.allclose(h.qto_rotvec(h.qfrom_rotvec(v)), v, atol=1e-9)
    noisy = h.matrix_from_quat(a) * 1.03 + rng.normal(0, 1e-4, (200, 3, 3))
    assert np.all(h.qangle(a, h.quat_from_matrix(h.orthonormalize(noisy))) < 1e-3)


def test_euler_round_trip():
    rng = np.random.default_rng(2)
    yaw, pitch, roll = rng.uniform(-3, 3, 500), rng.uniform(-1.5, 1.5, 500), rng.uniform(-3, 3, 500)
    y2, p2, r2 = h.euler_yxz_from_quat(h.quat_from_euler_yxz(yaw, pitch, roll))
    assert np.allclose(y2, yaw, atol=1e-9) and np.allclose(p2, pitch, atol=1e-9) and np.allclose(r2, roll, atol=1e-9)
    # Sign conventions in the ARKit face frame (+x face's left, +y up, +z out of the face).
    nose = np.array([0.0, 0.0, 1.0])
    R = h.matrix_from_quat(h.quat_from_euler_yxz(math.radians(30), 0, 0))
    assert (R @ nose)[0] > 0.4  # yaw + turns the nose towards +x, the face's own left
    R = h.matrix_from_quat(h.quat_from_euler_yxz(0, math.radians(20), 0))
    assert (R @ nose)[1] < -0.3  # pitch + is nose down
    R = h.matrix_from_quat(h.quat_from_euler_yxz(0, 0, math.radians(20)))
    assert (R @ np.array([1.0, 0, 0]))[1] > 0.3  # roll + raises the left ear


def test_slerp_and_average():
    a = h.qaxis(2, 0.0)
    b = h.qaxis(2, 1.0)
    assert np.allclose(h.slerp(a, b, 0.0), a) and np.allclose(h.slerp(a, b, 1.0), b)
    assert abs(h.qangle(a, h.slerp(a, b, 0.25)) - 0.25) < 1e-12
    assert abs(h.qangle(a, h.slerp(a, -b, 0.5)) - 0.5) < 1e-12  # sign of b does not matter
    avg = h.qaverage(np.stack([h.qaxis(1, 0.1), -h.qaxis(1, 0.3)]))
    assert abs(h.qangle(avg, h.qaxis(1, 0.2))) < 1e-3


def test_resample_and_speed():
    t = np.arange(0, 10, 0.02)
    q = h.qaxis(1, np.radians(30.0) * t)
    trk = h.Track("x", t, t, q, np.ones(len(t), bool))
    g = np.arange(1, 9, 0.001)
    Q, V = h.resample(trk, g, 0.05)
    assert V.all() and np.all(h.qangle(Q, h.qaxis(1, np.radians(30.0) * g)) < 1e-9)
    sp, ok = h.windowed_speed(Q, V, 0.001, 0.1)
    assert np.allclose(sp[ok] * DEG, 30.0, atol=1e-6)
    gap = np.ones(len(t), bool)
    gap[100:110] = False
    Q2, V2 = h.resample(h.Track("x", t, t, q, gap), g, 0.05)
    assert not V2[(g > 2.05) & (g < 2.15)].any()


def test_zoh():
    ts = np.array([0.0, 0.02, 0.04, 0.06])
    ta = np.array([0.05, 0.05, 0.09, 0.09])  # two bursts
    q = h.qaxis(2, ts)
    trk = h.Track("x", ts, ta, q, np.ones(4, bool))
    Q, V, age = h.zoh(trk, np.array([0.04, 0.05, 0.08, 0.09, 0.2]), max_age=0.1)
    assert list(V) == [False, True, True, True, False]
    assert np.allclose(age[1:4], [0.03, 0.06, 0.03])


def test_lag_estimation():
    rng = np.random.default_rng(3)
    dt = 0.001
    x = np.convolve(rng.normal(size=60000), np.hanning(301), mode="same")
    shift = 37.6e-3
    t = np.arange(len(x)) * dt
    y = np.interp(t - shift, t, x)
    ok = np.ones(len(x), bool)
    lag, peak, _, _ = h.estimate_lag(x, ok, y, ok, dt)
    assert abs(lag - shift) < 0.3e-3 and peak > 0.99, (lag, peak)


def test_app_view_lag_uniform():
    """Constant delay and even sampling: the app-view lag must equal
    sample lag + delivery delay + half a sample interval, within 1 ms."""
    rng = np.random.default_rng(5)
    traj = Trajectory(120.0, rng)
    dt = 0.001
    g = np.arange(12.0, 115.0, dt)
    ref = h.Track("ref", g, g, traj.q(g), np.ones(len(g), bool))
    p = np.arange(11.0, 116.0, 1 / 30)
    trk = h.Track("cam", p + 0.012, p + 0.112, traj.q(p), np.ones(len(p), bool))
    Qr, Vr = h.resample(ref, g, 0.05)
    sr, okr = h.windowed_speed(Qr, Vr, dt, h.SPEED_WINDOW)
    Qz, Vz, _ = h.zoh(trk, g, max_age=0.5)
    sz, okz = h.windowed_speed(Qz, Vz, dt, h.SPEED_WINDOW)
    lag, peak, _, _ = h.estimate_lag(sr, okr, sz, okz, dt)
    expected = 0.112 + 0.5 / 30
    print("    app-view lag %.2f ms (expected %.2f), peak correlation %.3f" % (lag * 1e3, expected * 1e3, peak))
    assert abs(lag - expected) < 1e-3


def test_clock_map():
    rng = np.random.default_rng(4)
    t1 = mac_ns(np.arange(0.2, 300, 0.1))
    d1 = np.round(1e6 + 12e6 * rng.random(len(t1)) ** 3).astype(np.int64)
    d2 = np.round(1e6 + 4e6 * rng.random(len(t1)) ** 2).astype(np.int64)
    t2 = iphone_from_mac(t1 + d1)
    t3 = t2 + 50_000
    t4 = mac_from_iphone(t3) + d2
    cm = h.ClockMap(t1, t2, t3, t4)
    m = mac_ns(np.linspace(0, 300, 50))
    err_ms = np.abs(cm.to_mac(iphone_from_mac(m)) - m) / 1e6
    assert err_ms.max() < 0.5, err_ms.max()
    assert abs(cm.b / 1e3 - IPHONE_DRIFT * 1e6) < 1.0, cm.b / 1e3


SESSION = {}
TEMP_DIRS = []


def temp_dir(prefix):
    d = tempfile.mkdtemp(prefix=prefix)
    TEMP_DIRS.append(d)
    return d


def session():
    if not SESSION:
        d = temp_dir("ht_synth_")
        truth = write_synthetic_session(d)
        res, series = h.analyze_session(d, keep_series=True)
        SESSION.update(dir=d, truth=truth, res=res, series=series)
    return SESSION


def angle_deg(a, b):
    return float(h.qangle(np.asarray(a), np.asarray(b)) * DEG)


def test_session_clock():
    S = session()
    s = h.Session(S["dir"])
    cm = h.ClockMap(s.sync["t1"], s.sync["t2"], s.sync["t3"], s.sync["t4"])
    m = mac_ns(np.linspace(0, 240, 50))
    err = np.abs(cm.to_mac(iphone_from_mac(m)) - m).max() / 1e6
    assert err < 0.5, err
    cs = S["res"]["clock_sync"]
    assert abs(cs["drift_ppm"] - 20.0) < 1.0, cs
    print("    clock: max mapping error %.3f ms, drift %.2f ppm (truth 20), offset error at most %.2f ms (best case %.2f)"
          % (err, cs["drift_ppm"], cs["bound_ms"], cs["uncertainty_ms"]))
    assert err < cs["bound_ms"]


def test_session_latency():
    S = session()
    tr = S["res"]["trackers"]
    la, lm = tr["airpods"]["latency_ms"], tr["mediapipe"]["latency_ms"]
    lv = tr["vision"]["latency_ms"]
    print("    sample-time lag: airpods %.2f ms (truth %.1f), mediapipe %.2f ms (truth %.1f), vision %.2f ms (truth %.1f)"
          % (la["sample_time_lag"], AIRPODS_LAG * 1e3, lm["sample_time_lag"], WEBCAM_LAG * 1e3, lv["sample_time_lag"],
             WEBCAM_LAG * 1e3))
    assert abs(la["sample_time_lag"] - AIRPODS_LAG * 1e3) < 1.0
    assert abs(lm["sample_time_lag"] - WEBCAM_LAG * 1e3) < 1.0
    assert abs(lv["sample_time_lag"] - WEBCAM_LAG * 1e3) < 1.0
    mean_mp = lm["motion_to_data"]["mean"]
    print("    mediapipe motion-to-data mean %.2f ms (truth %.1f)" % (mean_mp, (WEBCAM_LAG + MP_DELAY) * 1e3))
    assert abs(mean_mp - (WEBCAM_LAG + MP_DELAY) * 1e3) < 1.0
    t_lo, t_hi = 12.0, 230.0
    for name in ("airpods", "mediapipe"):
        lat = tr[name]["latency_ms"]
        truth = mean_age(S["truth"][name], t_lo, t_hi, lat["app_view_max_age"] / 1e3) * 1e3
        got = lat["app_view_lag"]
        print("    %s app-view lag %.2f ms (mean age of the newest arrived sample %.2f ms)" % (name, got, truth))
        assert abs(got - truth) < 2.5


def test_session_segment_lag():
    """Lag on the impulsive and on the smooth protocol steps alone (segments
    from the prompts in events.csv), each within 1.5 ms of the truth."""
    S = session()
    tr = S["res"]["trackers"]
    assert h.step_class("Hold still, then snap your head quickly to the left and back. Repeat") == "impulsive"
    assert h.step_class("Turn slowly left and right, a few times") == "smooth"
    assert h.step_class("Turn as far left as is comfortable, then as far right") == "range"
    assert h.step_class("Face the screen and hold still") == "static"
    assert h.step_class("Move around") is None
    for name, truth in (("airpods", AIRPODS_LAG), ("mediapipe", WEBCAM_LAG), ("vision", WEBCAM_LAG)):
        seg = tr[name]["latency_ms"]["by_segment"]
        parts = []
        for cls in ("impulsive", "smooth"):
            got = seg[cls]["sample_time_lag"]
            parts.append("%s %.2f ms (r %.2f, %.0f s)" % (cls, got, seg[cls]["corr"], seg[cls]["seconds"]))
            assert got is not None and abs(got - truth * 1e3) < 1.5, (name, cls, got, truth * 1e3)
        print("    %s lag by step (truth %.1f): %s" % (name, truth * 1e3, "; ".join(parts)))


def test_session_alignment():
    S = session()
    tr = S["res"]["trackers"]
    al = tr["airpods"]["alignment"]
    # Drift is absorbed into A at its mean over the aligned samples.
    t = S["series"]["t"]
    mean_drift = math.radians(AIRPODS_DRIFT_DEG_MIN / 60.0 * float(np.mean(t)))
    A_exp = h.qmul(A_AIR, h.qaxis(2, -mean_drift))
    eA, eB = angle_deg(al["A"], A_exp), angle_deg(al["B"], B_AIR)
    print("    airpods: A error %.3f deg, B error %.3f deg, residual median %.3f deg"
          % (eA, eB, al["residual_deg"]["median"]))
    assert eA < 0.5 and eB < 0.5
    am = tr["mediapipe"]["alignment"]
    eA, eB = angle_deg(am["A"], A_MP), angle_deg(am["B"], B_MP)
    print("    mediapipe: A error %.3f deg, B error %.3f deg, variant %s"
          % (eA, eB, tr["mediapipe"]["convention"]["variant"]))
    assert eA < 0.5 and eB < 0.5 and tr["mediapipe"]["convention"]["variant"] == "R"
    flips = tr["vision"]["convention"]["flips"]
    print("    vision: flips %s (truth %s; same parity = equivalent)" % (flips, list(VISION_FLIPS)))
    assert np.prod(flips) == np.prod(VISION_FLIPS)
    mae = tr["vision"]["agreement_latency_compensated_deg"]["still"]["yaw"]["mae"]
    assert mae < 2.5, mae


def test_session_noise_drift_coverage():
    S = session()
    tr = S["res"]["trackers"]
    na, nm = tr["airpods"]["noise_at_rest_deg"], tr["mediapipe"]["noise_at_rest_deg"]
    nr = S["res"]["reference"]["noise_at_rest_deg"]
    print("    noise at rest (yaw/pitch/roll): airpods %.3f/%.3f/%.3f (truth %.2f), mediapipe %.2f/%.2f/%.2f (truth %.1f), "
          "reference %.3f/%.3f/%.3f (truth %.2f)"
          % (na["yaw"], na["pitch"], na["roll"], AIRPODS_NOISE_DEG, nm["yaw"], nm["pitch"], nm["roll"], MP_NOISE_DEG,
             nr["yaw"], nr["pitch"], nr["roll"], REF_NOISE_DEG))
    for n, truth in ((na, AIRPODS_NOISE_DEG), (nm, MP_NOISE_DEG), (nr, REF_NOISE_DEG)):
        for k in ("yaw", "pitch", "roll"):
            assert 0.7 * truth < n[k] < 1.3 * truth, (k, n[k], truth)
    d = tr["airpods"]["yaw_drift"]["deg_per_min"]
    print("    airpods yaw drift %.3f deg/min (truth %.2f)" % (d, AIRPODS_DRIFT_DEG_MIN))
    assert abs(d - AIRPODS_DRIFT_DEG_MIN) < 0.1 * AIRPODS_DRIFT_DEG_MIN
    cov = tr["mediapipe"]["coverage"]["by_abs_yaw"]
    print("    mediapipe coverage by |yaw|: " + ", ".join("%s %.2f" % (k, v["share"]) for k, v in cov.items()
                                                       if v["share"] is not None))
    assert cov["0-15"]["share"] > 0.95 and cov["45-60"]["share"] < 0.15 and cov[">60"]["share"] < 0.05
    assert tr["airpods"]["coverage"]["by_abs_yaw"][">60"]["share"] > 0.95


def test_session_json_and_missing_sources():
    S = session()
    json.dumps(h.to_jsonable(S["res"]))  # must serialise
    d = temp_dir("ht_partial_")
    write_synthetic_session(d, duration=40.0, seed=9)
    for f in ("ref.csv", "mediapipe.csv"):
        os.remove(os.path.join(d, f))
    with open(os.path.join(d, "camera.csv"), "w") as f:
        f.write("seq,pts_ns,delivered_ns,vision_start_ns,vision_done_ns,face,yaw_deg,pitch_deg,roll_deg,"
                "box_x,box_y,box_w,box_h,confidence\n")
    res, _ = h.analyze_session(d)
    assert "airpods" in res["trackers"] and "rate_hz" in res["trackers"]["airpods"]["delivery"]
    assert "mediapipe" not in res["trackers"] and "vision" not in res["trackers"]
    json.dumps(h.to_jsonable(res))
    empty = temp_dir("ht_empty_")
    res, _ = h.analyze_session(empty)
    assert res["trackers"] == {}


def test_headphones_firmware():
    sp = {"SPBluetoothDataType": [{
        "device_connected": [
            {"JTAIRPDMAX2": {"device_address": "00:11:22:33:44:55", "device_firmwareVersion": "9A348",
                             "device_minorType": "Headphones", "device_productID": "0x202D",
                             "device_vendorID": "0x004C", "device_serialNumber": "X"}},
            {"Mouse": {"device_minorType": "Mouse", "device_productID": "0xB03F"}},
        ],
        "device_not_connected": [{"Other": {"device_minorType": "Headphones", "device_firmwareVersion": "1"}}],
    }]}
    hp = h.headphones_from_system_profiler(sp)
    assert hp == [{"name": "JTAIRPDMAX2", "firmware": "9A348", "vendor_id": "0x004C", "product_id": "0x202D",
                   "minor_type": "Headphones"}]
    assert h.headphones_from_system_profiler({}) == []
    d = temp_dir("ht_hp_")
    write_synthetic_session(d, duration=20.0, seed=3)
    res, _ = h.analyze_session(d)
    assert "headphones" not in res and any("firmware not recorded" in n for n in res["notes"])
    with open(os.path.join(d, "headphones.json"), "w") as f:
        json.dump({"source": "test", "connected_headphones": hp}, f)
    res, _ = h.analyze_session(d)
    assert res["headphones"][0]["firmware"] == "9A348"


def test_laser_geometry():
    R = 1500.0
    sag = R * (1 - math.cos(802 / (2 * R)))
    assert abs(h.screen_radius(802, sag) - R) < 0.01 and h.screen_radius(802, 0) == math.inf
    for sagitta in (sag, 0.0):
        g = h.laser_geometry(650, 60, sagitta, (802, 330), (3440, 1440))
        for lab in ("C", "L30", "L10", "R20", "R30", "U10", "D10"):
            px = h.laser_mark_px(lab, g)
            assert abs(h.laser_mark_angle(lab, px, g) - h.laser_label(lab)[1]) < 1e-6, (lab, sagitta)
    flat = h.laser_geometry(650, 0, 0, (802, 330), (3440, 1440))
    dx = h.laser_mark_px("R20", flat)[0] - 1720
    assert abs(dx * 802 / 3440 - 650 * math.tan(math.radians(20))) < 1e-6  # flat screen: D tan(angle)
    dy = 720 - h.laser_mark_px("U10", flat)[1]
    assert abs(dy * 330 / 1440 - 650 * math.tan(math.radians(10))) < 1e-6
    curved = h.laser_geometry(650, 0, sag, (802, 330), (3440, 1440))
    assert h.laser_mark_px("R30", curved)[0] < h.laser_mark_px("R30", flat)[0]  # edges curve towards the viewer
    for bad in ("R40", "U20"):
        try:
            h.laser_mark_px(bad, curved)
            raise AssertionError("%s should be off the screen" % bad)
        except ValueError:
            pass
    assert h.laser_label("L20") == ("yaw", -20.0) and h.laser_label("U10") == ("pitch", 10.0)
    assert h._step_target("[R30] Point at right 30 and hold") == "R30" and h._step_target("Hold still") is None


class _FakeSession:
    def __init__(self, meta, steps):
        self.meta, self._steps = meta, steps

    def steps(self):
        return self._steps


def test_laser_check():
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "protocols", "laser_check.json")) as f:
        proto = json.load(f)
    g = h.laser_geometry(650, 60, 54, (802, 330), (3440, 1440))
    tg = h.laser_targets(sorted({s["target"] for s in proto["steps"]}), g)
    steps, t = [], 1.0
    for st in proto["steps"]:
        steps.append((t, st["seconds"], "[%s] %s" % (st["target"], st["text"])))
        t += st["seconds"]
    rng = np.random.default_rng(5)
    A, B = h.qnormalize(rng.normal(size=4)), h.qnormalize(rng.normal(size=4))
    ts = np.arange(0.0, t + 1.0, 0.02)
    yaw, pitch = np.zeros(len(ts)), np.zeros(len(ts))
    for (t0, dur, text), st in zip(steps, proto["steps"]):
        axis, deg = h.laser_label(st["target"])
        sel = (ts >= t0 + 0.5) & (ts < t0 + dur + 0.5)
        if axis == "yaw":
            yaw[sel] = deg
        elif axis == "pitch":
            pitch[sel] = deg
    moving_step = steps[5]
    wob = (ts >= moving_step[0] + 2.5) & (ts < moving_step[0] + 5.0)

    def make(name, gain, drift_deg_min=0.0, wobble=False):
        y = gain * yaw + (np.where(wob, 3.0 * np.sin(2 * np.pi * 2 * ts), 0.0) if wobble else 0.0)
        q = h.quat_from_euler_yxz(y / h.DEG, gain * pitch / h.DEG, np.zeros(len(ts)))
        q = h.qmul(h.qaxis(1, drift_deg_min / 60.0 * ts / h.DEG), q)
        q = h.qmul(h.qmul(A, q), B)  # arbitrary tracker frames: the angles must not care
        return h.Track(name, ts, ts, q, np.ones(len(ts), bool))

    s = _FakeSession({"targets": tg}, steps)
    res = h.laser_check(s, {"exact": make("exact", 1.0), "low": make("low", 0.9, wobble=True),
                            "drifting": make("drifting", 1.0, drift_deg_min=2.0)})
    ex, lo, dr = (res["trackers"][k] for k in ("exact", "low", "drifting"))
    for grp in ("yaw_left", "yaw_right", "pitch"):
        assert abs(ex[grp]["gain"] - 1.0) < 1e-3 and ex[grp]["mae_deg"] < 0.02, (grp, ex[grp])
        assert abs(lo[grp]["gain"] - 0.9) < 1e-3, (grp, lo[grp])
        assert abs(dr[grp]["gain"] - 1.0) < 0.01, (grp, dr[grp])
    assert ex["dropped_holds"] == 0 and lo["dropped_holds"] == 1
    assert dr["centre_spread_deg"] > 3.0  # drift is real, yet the interpolated centre removes it
    print("    laser: gains exact %.4f, low %.4f, drifting %.4f (centre spread %.1f deg)"
          % (ex["yaw"]["gain"], lo["yaw"]["gain"], dr["yaw"]["gain"], dr["centre_spread_deg"]))


TESTS = [v for k, v in list(globals().items()) if k.startswith("test_")]


def main():
    if len(sys.argv) == 3 and sys.argv[1] == "--write-session":
        write_synthetic_session(sys.argv[2])
        print("wrote", sys.argv[2])
        return 0
    failed = 0
    for t in TESTS:
        try:
            t()
            print("  ok   %s" % t.__name__)
        except Exception:
            failed += 1
            print("  FAIL %s" % t.__name__)
            traceback.print_exc()
    for d in TEMP_DIRS:
        shutil.rmtree(d, ignore_errors=True)
    print("%d tests, %d failed" % (len(TESTS), failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
