"""Tables, figures and numbers for the side paper from a list of sessions.

    headtrack/.venv/bin/python headtrack/analysis/paper_results.py headtrack/paper/sessions.json

sessions.json lists the sessions with a label and their role: "core" (full
protocol, used for agreement, noise, range and cost), "latency" (sessions whose
impulsive steps count for latency), "drift" (the long desk-work sessions).
Optional per session: "day" (1, 2, ...), "audio" ("silent" or "music";
otherwise read from "condition") and "headphones" ("max2", the default, or
"pro1": reported as a separate tracker, AirPods Pro 1st generation). headtrack/paper/add_day.py appends a
recording day; headtrack/paper/update.sh reruns everything and lists which
numbers changed. Writes headtrack/paper/results/: numbers.json (every number the text quotes),
tables.md, fig2_turn.png ... fig5_drift.png and fig_data.json (their plotted values). Rerun after adding sessions.
"""

import datetime
import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import htlib as h

ALL_TRACKERS = ("airpods", "airpods_pro", "mediapipe", "vision")
TRACKERS = ALL_TRACKERS  # narrowed in main() to the trackers present in the sessions
LABEL = {"reference": "Reference (iPhone)", "airpods": "AirPods Max 2", "airpods_pro": "AirPods Pro (1st gen.)",
         "mediapipe": "MediaPipe", "vision": "Apple Vision"}
COLOR = {"reference": "#222222", "airpods": "#1f77b4", "airpods_pro": "#9467bd", "mediapipe": "#ff7f0e", "vision": "#2ca02c"}
# sessions.json "headphones" -> tracker name in the results (htlib always calls it "airpods")
HP_TRACKER = {"max2": "airpods", "pro1": "airpods_pro"}
YAW_BINS = ((0, 15), (15, 30), (30, 45), (45, 60), (60, 90))


def med_range(xs):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    if not xs:
        return None
    return {"median": float(np.median(xs)), "min": float(np.min(xs)), "max": float(np.max(xs)), "n": len(xs)}


def bland_altman(e):
    e = e[np.isfinite(e)]
    if len(e) < 10:
        return None
    b, sd = float(np.mean(e)), float(np.std(e, ddof=1))
    return {"bias": b, "loa_low": b - 1.96 * sd, "loa_high": b + 1.96 * sd, "mae": float(np.mean(np.abs(e))),
            "rms": float(np.sqrt(np.mean(e ** 2))), "n": int(len(e))}


def apkey(r):
    """The headphone tracker's name in this session's results."""
    return next((k for k in ("airpods", "airpods_pro") if k in r["trackers"]), "airpods")


def has(name, rows):
    return [x for x in rows if name in x[1]["trackers"]]


def audio_of(s):
    if s.get("audio"):
        return s["audio"]
    c = s.get("condition", "")
    return "music" if "music" in c else ("silent" if "silent" in c else None)


def detrend_yaw(t, e, mask):
    """AirPods yaw error minus its straight-line trend over the session: the
    agreement left once drift (reported separately) is removed."""
    m = mask & np.isfinite(e)
    if m.sum() < 50:
        return e
    k, c = np.polyfit(t[m], e[m], 1)
    return e - (k * t + c)


def load(spec_path):
    spec = json.load(open(spec_path))
    out = []
    for s in spec["sessions"]:
        res, ser = h.analyze_session(s["path"], keep_series=True)
        key = HP_TRACKER[s.get("headphones", "max2")]
        if key != "airpods":
            if "airpods" in res["trackers"]:
                res["trackers"][key] = res["trackers"].pop("airpods")
            if ser is not None and "airpods" in ser:
                ser[key] = ser.pop("airpods")
        out.append((s, res, ser))
    return spec, out


def aligned_app_view(path, res, name):
    """Tracker angles as an application sees them (arrival times), in the
    reference frame, plus the reference at capture time."""
    s = h.Session(path)
    clock = h.ClockMap(s.sync["t1"], s.sync["t2"], s.sync["t3"], s.sync["t4"])
    ref = h.track_reference(s, clock)
    q_neutral = np.array(res["neutral"]["q"])
    if name == "reference":
        ang = np.full((len(ref), 3), np.nan)
        ang[ref.valid] = h.head_angles(ref.q[ref.valid], q_neutral)
        return ref.t_sample, ang
    t = res["trackers"][name]
    if name.startswith("airpods"):
        trk = h.track_airpods(s)
    elif name == "mediapipe":
        trk = h.track_mediapipe(s, t["convention"]["variant"])
    else:
        trk = h.track_vision(s, tuple(t["convention"]["flips"]))
    A, B = np.array(t["alignment"]["A"]), np.array(t["alignment"]["B"])
    ang = np.full((len(trk), 3), np.nan)
    v = trk.valid
    ang[v] = h.head_angles(h.qmul(A, h.qmul(trk.q[v], B)), q_neutral)
    return trk.t_avail, ang


def fig1_architecture(path):
    """Block diagram of the measurement set-up (no data)."""
    fig, ax = plt.subplots(figsize=(6.3, 3.5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 56)
    ax.axis("off")

    def box(x, y, w, hgt, text, fc="#f2f2f2", ec="#444444", fs=6.6, bold=False):
        ax.add_patch(matplotlib.patches.FancyBboxPatch((x, y), w, hgt, boxstyle="round,pad=0.4", fc=fc, ec=ec, lw=0.8))
        ax.text(x + w / 2, y + hgt / 2, text, ha="center", va="center", fontsize=fs, weight="bold" if bold else "normal")

    def arrow(x0, y0, x1, y1, text="", dy=1.2, tx=None):
        ax.annotate("", (x1, y1), (x0, y0), arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#444444"))
        if text:
            ax.text((x0 + x1) / 2 if tx is None else tx, (y0 + y1) / 2 + dy, text, ha="center", va="bottom",
                    fontsize=6.5, color="#333333")

    # every source feeds the tall log box on the right with a straight arrow; labels sit left of the Mac outline
    ax.add_patch(matplotlib.patches.Rectangle((25.5, 0.5), 74, 54, fc="none", ec="#999999", lw=0.8, ls="--"))
    ax.text(26.5, 53.5, "MacBook Air: one host clock (mach_absolute_time)", fontsize=7.5, va="top", color="#555555")
    box(1, 38, 15, 8, "AirPods Max 2\nmotion sensor", fc="#dbe8f5")
    box(1, 26, 15, 8, "Built-in camera\n1280×720, 30 fps", fc="#eeeeee")
    box(1, 2, 15, 9, "iPhone 12\nARKit TrueDepth\n60 fps (reference)", fc="#e6e6e6")
    box(28, 38, 22, 8, "Core Motion handler\nsensor time + arrival time")
    box(28, 26, 22, 8, "Capture callback\nApple Vision (same frame)", fc="#dff0dd")
    box(28, 14, 22, 8, "Shared memory\n+ wake-up pipe")
    box(55, 14, 19, 8, "MediaPipe process\nFace Landmarker", fc="#fde8d4")
    box(28, 2, 22, 9, "UDP receiver\nposes at capture time\nping–pong clock sync")
    box(80, 2, 17, 44, "Logs per session:\nCSV + events,\none clock\n\n→ analysis\n(Python)", fc="#fafafa")
    arrow(16, 42, 28, 42, "Bluetooth", tx=20.5)
    arrow(16, 30, 28, 30, "frames", tx=20.5)
    arrow(16, 6.5, 28, 6.5, "Wi-Fi UDP", tx=20.5)
    arrow(50, 42, 80, 42)
    arrow(50, 30, 80, 30)
    arrow(39, 26, 39, 22)
    arrow(50, 18, 55, 18)
    arrow(74, 18, 80, 18)
    arrow(50, 6.5, 80, 6.5)
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def main():
    spec_path = sys.argv[1]
    outdir = os.path.join(os.path.dirname(spec_path), "results")
    os.makedirs(outdir, exist_ok=True)
    spec, data = load(spec_path)
    global TRACKERS
    TRACKERS = tuple(n for n in ALL_TRACKERS if any(n in r["trackers"] for _, r, _ in data))
    N = {"sessions": [{"label": s["label"], "path": s["path"], "roles": s["roles"], "condition": s.get("condition", "")}
                      for s, _, _ in data]}

    # ---- session overview and clock sync
    N["clock_sync"] = {s["label"]: {"best_ms": r["clock_sync"]["uncertainty_ms"], "bound_ms": r["clock_sync"].get("bound_ms"),
                                    "rtt_min_ms": r["clock_sync"]["rtt_ms"]["min"], "rtt_median_ms": r["clock_sync"]["rtt_ms"]["median"],
                                    "drift_ppm": r["clock_sync"]["drift_ppm"]} for s, r, _ in data}
    N["reference"] = {s["label"]: {"tracked_share": r["reference"]["tracked_share"], "rate_hz": r["reference"]["delivery"]["rate_hz"],
                                   "capture_to_delivered_ms": r["reference"]["internal_delay_ms"]["capture_to_delivered"]["median"],
                                   "phone_to_mac_ms": r["reference"]["internal_delay_ms"]["sent_to_mac_receive"]["median"],
                                   "noise_deg": r["reference"].get("noise_at_rest_deg")} for s, r, _ in data}
    N["duration_s"] = {s["label"]: float(np.ptp(ser["t"])) for s, r, ser in data}
    N["headphones"] = sorted({d["firmware"] for _, r, _ in data for d in (r.get("headphones") or [])})

    core = [(s, r, ser) for s, r, ser in data if "core" in s["roles"]]
    lat = [(s, r, ser) for s, r, ser in data if "latency" in s["roles"]]
    drift = [(s, r, ser) for s, r, ser in data if "drift" in s["roles"]]

    # ---- delivery, latency, cost
    T = {}
    for name in TRACKERS:
        core_n, lat_n, drift_n = has(name, core), has(name, lat), has(name, drift)
        d = {}
        d["rate_hz"] = med_range([r["trackers"][name]["delivery"]["rate_hz"] for _, r, _ in core_n + drift_n])
        d["valid_share"] = med_range([r["trackers"][name]["delivery"]["valid_share"] for _, r, _ in core_n])
        d["burst_share"] = med_range([r["trackers"][name]["delivery"]["arrivals_within_1ms_share"] for _, r, _ in core_n])
        d["arrival_gap_p99_ms"] = med_range([r["trackers"][name]["delivery"]["arrival_interval_ms"]["p99"] for _, r, _ in core_n])
        d["lag_impulsive_ms"] = med_range([(r["trackers"][name]["latency_ms"].get("by_segment") or {}).get("impulsive", {}).get("sample_time_lag")
                                           for _, r, _ in lat_n])
        d["lag_smooth_ms"] = med_range([(r["trackers"][name]["latency_ms"].get("by_segment") or {}).get("smooth", {}).get("sample_time_lag")
                                        for _, r, _ in lat_n])
        d["motion_to_data_median_ms"] = med_range([(r["trackers"][name]["latency_ms"]["motion_to_data"] or {}).get("median") for _, r, _ in lat_n])
        d["motion_to_data_p95_ms"] = med_range([(r["trackers"][name]["latency_ms"]["motion_to_data"] or {}).get("p95") for _, r, _ in lat_n])
        d["app_view_lag_ms"] = med_range([r["trackers"][name]["latency_ms"]["app_view_lag"] for _, r, _ in lat_n])
        idl = [r["trackers"][name]["internal_delay_ms"] for _, r, _ in core_n]
        if name.startswith("airpods"):
            d["timestamp_to_arrival_ms"] = med_range([x["timestamp_to_arrival"]["median"] for x in idl])
        else:
            d["capture_to_delivered_ms"] = med_range([x["capture_to_delivered"]["median"] for x in idl])
            d["capture_to_done_ms"] = med_range([x["capture_to_done"]["median"] for x in idl])
        T[name] = d
    # CPU: MediaPipe process alone; recorder process = capture + Vision + AirPods + logging
    cpu_mp, cpu_rec, cpu_mp_ms = [], [], []
    for s, r, _ in core:
        mp = json.load(open(os.path.join(s["path"], "mediapipe_summary.json")))
        cpu_mp.append(mp["cpu_share_of_one_core"])
        cpu_mp_ms.append(mp["convert_plus_inference_ms"]["median"])
        se = json.load(open(os.path.join(s["path"], "session.json")))["end"]
        cpu_rec.append((se["cpu_user_s"] + se["cpu_system_s"]) / se["duration_s"])
    T["cost"] = {"mediapipe_cpu_share_one_core": med_range(cpu_mp), "recorder_cpu_share_one_core": med_range(cpu_rec),
                 "vision_compute_ms": med_range([r["trackers"]["vision"]["internal_delay_ms"]["vision_compute"]["median"] for _, r, _ in core]),
                 "mediapipe_inference_ms": med_range(cpu_mp_ms),
                 "thermal": sorted({json.load(open(os.path.join(s["path"], "session.json")))["end"]["thermal"] for s, _, _ in data})}
    N["trackers"] = T

    # ---- agreement (latency compensated), Bland-Altman per axis, at rest and moving, pooled over core sessions
    AG = {}
    for name in TRACKERS:
        pooled = {"still": [[], [], []], "moving": [[], [], []]}
        for s, r, ser in core:
            if name not in ser:
                continue
            t, ref, trk, still = ser["t"], ser["reference"], ser[name], ser["still"]
            ok = np.all(np.isfinite(ref), 1) & np.all(np.isfinite(trk), 1)
            e = h.wrap_deg(trk - ref)
            if name.startswith("airpods"):
                e[:, 0] = detrend_yaw(t, e[:, 0], ok)
            # moving: reference speed above the still threshold (not still, valid)
            for sub, m in (("still", ok & still), ("moving", ok & ~still)):
                for i in range(3):
                    pooled[sub][i].append(e[m, i])
        AG[name] = {}
        for sub in ("still", "moving"):
            AG[name][sub] = {}
            geo = []
            for i, ax in enumerate(("yaw", "pitch", "roll")):
                arr = np.concatenate(pooled[sub][i]) if pooled[sub][i] else np.array([])
                AG[name][sub][ax] = bland_altman(arr)
    N["agreement_deg"] = AG
    N["agreement_note"] = "AirPods yaw detrended per session (drift reported separately); latency compensated; pooled over core sessions"
    N["geodesic_moving_median_deg"] = {name: med_range([r["trackers"][name]["agreement_latency_compensated_deg"]["moving"]["geodesic"]["median"]
                                                         for _, r, _ in has(name, core)]) for name in TRACKERS}
    N["noise_deg"] = {name: {ax: med_range([(r["trackers"][name].get("noise_at_rest_deg") or {}).get(ax) for _, r, _ in has(name, core)])
                             for ax in ("yaw", "pitch", "roll")} for name in TRACKERS}
    N["noise_deg"]["reference"] = {ax: med_range([(r["reference"].get("noise_at_rest_deg") or {}).get(ax) for _, r, _ in core])
                                   for ax in ("yaw", "pitch", "roll")}

    # ---- range: share of time each tracker has a pose, by reference |yaw|, pooled over core
    RG = {}
    for name in ("mediapipe", "vision"):
        rows = {}
        for lo, hi in YAW_BINS:
            num = den = 0
            for s, r, ser in core:
                ref, trk = ser["reference"], ser[name]
                ok = np.isfinite(ref[:, 0])
                m = ok & (np.abs(ref[:, 0]) >= lo) & (np.abs(ref[:, 0]) < hi)
                den += m.sum()
                num += (m & np.isfinite(trk[:, 0])).sum()
            rows["%d-%d" % (lo, hi)] = {"share": float(num / den) if den else None, "seconds": float(den * h.ALIGN_DT)}
        RG[name] = rows
    RG["yaw_error_by_bin"] = {}
    for name in TRACKERS:
        RG["yaw_error_by_bin"][name] = {}
        for lo, hi in YAW_BINS:
            es = []
            for s, r, ser in core:
                if name not in ser:
                    continue
                ref, trk = ser["reference"], ser[name]
                e = h.wrap_deg(trk[:, 0] - ref[:, 0])
                if name.startswith("airpods"):
                    e = detrend_yaw(ser["t"], e, np.isfinite(e))
                m = np.isfinite(e) & (np.abs(ref[:, 0]) >= lo) & (np.abs(ref[:, 0]) < hi)
                es.append(np.abs(e[m]))
            es = np.concatenate(es) if es else np.array([])
            RG["yaw_error_by_bin"][name]["%d-%d" % (lo, hi)] = float(np.median(es)) if len(es) >= 20 else None
    RG["reference_angle_range_deg"] = {s["label"]: r["reference"]["angle_range_deg"] for s, r, _ in core}
    RG["reference_tracked_share"] = {s["label"]: r["reference"]["tracked_share"] for s, r, _ in core}
    N["range"] = RG

    # ---- drift
    DR = {}
    for s, r, ser in data:
        yd = r["trackers"][apkey(r)].get("yaw_drift")
        if yd:
            DR[s["label"]] = {"deg_per_min": yd["deg_per_min"], "span_min": yd["span_min"], "r2": yd["r2"],
                              "headphones": apkey(r)}
    for s, r, ser in drift:
        t, ref, ap = ser["t"], ser["reference"], ser[apkey(r)]
        jumps = r.get("heading_jumps") or []
        t_end = min(t[-1], jumps[0]["t_s"] - 1.0) if jumps else t[-1]  # drift up to the first heading jump
        ok = np.isfinite(ref[:, 0]) & np.isfinite(ap[:, 0]) & (t <= t_end)
        e = h.wrap_deg(ap[:, 0] - ref[:, 0])
        mins = []
        for lo in np.arange(0, t_end - t[0], 60.0):
            m = ok & (t - t[0] >= lo) & (t - t[0] < lo + 60)
            mins.append(float(np.median(e[m])) if m.sum() else None)
        steps = [b - a for a, b in zip(mins, mins[1:]) if a is not None and b is not None]
        # slope over the first 5 min and the steepest 60 s window, from 1 s medians
        sec = np.arange(t[0], t_end - 1, 1.0)
        y = np.array([np.median(e[ok & (t >= a) & (t < a + 1)]) if (ok & (t >= a) & (t < a + 1)).sum() > 20 else np.nan
                      for a in sec])
        x = (sec - t[0]) / 60.0
        fin = np.isfinite(y)
        first5 = fin & (x < 5)
        slopes = [np.polyfit(x[fin & (x >= c) & (x < c + 1)], y[fin & (x >= c) & (x < c + 1)], 1)[0]
                  for c in np.arange(0, x[-1] - 1, 0.25) if (fin & (x >= c) & (x < c + 1)).sum() > 30]
        DR[s["label"] + "_per_minute"] = {"headphones": apkey(r), "median_yaw_error_deg": mins, "max_step_deg_per_min": float(np.max(steps)),
                                          "total_range_deg": float(np.nanmax([m for m in mins if m is not None]) -
                                                                   np.nanmin([m for m in mins if m is not None])),
                                          "first_5min_deg_per_min": float(np.polyfit(x[first5], y[first5], 1)[0]) if first5.sum() > 30 else None,
                                          "steepest_60s_deg_per_min": float(np.max(np.abs(slopes))) if slopes else None,
                                          "until_heading_jump_min": float((t_end - t[0]) / 60.0) if jumps else None}
    N["drift"] = DR
    # Heading jumps (htlib.heading_jumps), every session, seconds from the session's analysis start
    N["heading_jumps"] = {s["label"]: [{"t_from_start_s": j["t_s"] - ser["t"][0], "deg": j["deg"], "gyro_dps": j["gyro_dps"]}
                                       for j in (r.get("heading_jumps") or [])] for s, r, ser in data}
    # Drift checks: extra sessions (spec "drift_checks") used for the headphones' drift and delivery only;
    # they never enter `data`, so their camera timings (other load on the Mac) touch no other statistic.
    DC = []
    for c in spec.get("drift_checks", []):
        rc, sc = h.analyze_session(c["path"], keep_series=True)
        ap = rc["trackers"].get("airpods", {})
        yd, dl = ap.get("yaw_drift") or {}, ap.get("delivery", {})
        minutes = float(np.ptp(sc["t"])) / 60.0
        DC.append({"label": c["label"], "path": c["path"], "day": c.get("day"), "audio": c.get("audio"), "note": c.get("note", ""),
                   "minutes": minutes, "deg_per_min": yd.get("deg_per_min"), "r2": yd.get("r2"),
                   "rate_hz": dl.get("rate_hz"), "gaps_per_min": (dl.get("intervals_over_2_5x_median") or 0) / minutes,
                   "heading_jumps": len(rc.get("heading_jumps") or [])})
    N["drift_checks"] = DC
    # Webcam trackers' yaw trend against the reference (the drift control), sessions of 5 min or more
    WT, worst = {}, {}
    for s, r, ser in data:
        if N["duration_s"][s["label"]] < 300:
            continue
        for n in ("mediapipe", "vision"):
            yd = r["trackers"].get(n, {}).get("yaw_drift")
            if yd:
                WT.setdefault(s["label"], {})[n] = {"deg_per_min": yd["deg_per_min"], "r2": yd["r2"]}
    for c in spec.get("drift_checks", []):
        rc, _ = h.analyze_session(c["path"])
        for n in ("mediapipe", "vision"):
            yd = rc["trackers"].get(n, {}).get("yaw_drift")
            if yd:
                WT.setdefault(c["label"], {})[n] = {"deg_per_min": yd["deg_per_min"], "r2": yd["r2"]}
    for n in ("mediapipe", "vision"):
        vals = [v[n] for v in WT.values() if n in v]
        if vals:
            worst[n] = {"max_abs_deg_per_min": max(abs(v["deg_per_min"]) for v in vals), "max_r2": max(v["r2"] or 0 for v in vals)}
    N["webcam_yaw_trend"] = {"sessions": WT, "summary": worst}
    N["drift_sessions"] = med_range([DR[s["label"] + "_per_minute"]["total_range_deg"] for s, _, _ in drift])

    # ---- per session (for day-to-day and audio-condition comparisons)
    PS = []
    for s, r, ser in data:
        ap = r["trackers"].get(apkey(r), {})
        dl = ap.get("delivery", {})
        span = N["duration_s"][s["label"]] / 60.0
        row = {"label": s["label"], "day": s.get("day", 1), "audio": audio_of(s), "roles": s["roles"], "headphones": apkey(r),
               "minutes": span, "airpods_rate_hz": dl.get("rate_hz"),
               "airpods_gaps": dl.get("intervals_over_2_5x_median"),
               "airpods_gaps_per_min": (dl["intervals_over_2_5x_median"] / span) if dl.get("intervals_over_2_5x_median") is not None and span else None,
               "airpods_drift_deg_per_min": (ap.get("yaw_drift") or {}).get("deg_per_min"),
               "sync_best_ms": N["clock_sync"][s["label"]]["best_ms"], "sync_bound_ms": N["clock_sync"][s["label"]]["bound_ms"]}
        # per-session keys say "airpods" for whichever headphones the session used ("headphones")
        for name, key in ((apkey(r), "airpods"), ("mediapipe", "mediapipe"), ("vision", "vision")):
            tr = r["trackers"].get(name, {})
            row[key + "_motion_to_data_ms"] = ((tr.get("latency_ms") or {}).get("motion_to_data") or {}).get("median")
            agr = tr.get("agreement_latency_compensated_deg") or {}
            for sub in ("still", "moving"):
                row["%s_geodesic_%s_deg" % (key, sub)] = ((agr.get(sub) or {}).get("geodesic") or {}).get("median")
        PS.append(row)
    N["per_session"] = PS
    N["days"] = sorted({row["day"] for row in PS})
    # marker-tent check of the reference's angle scale (role "marker"; htlib.marker_scale_check)
    N["marker_check"] = {s["label"]: r.get("marker_check") for s, r, _ in data if "marker" in s["roles"]} or None
    # headphone delivery by audio condition, per headphone model
    N["by_audio"] = {}
    for hp in sorted({row["headphones"] for row in PS}):
        N["by_audio"][hp] = {}
        for cond in ("silent", "music"):
            rows = [row for row in PS if row["headphones"] == hp and row["audio"] == cond
                    and ("core" in row["roles"] or "drift" in row["roles"])]
            N["by_audio"][hp][cond] = {"sessions": [row["label"] for row in rows],
                                       "rate_hz": med_range([row["airpods_rate_hz"] for row in rows]),
                                       "gaps_per_min": med_range([row["airpods_gaps_per_min"] for row in rows])}

    json.dump(N, open(os.path.join(outdir, "numbers.json"), "w"), indent=2)

    # ---- figures (FD: the plotted values, saved as fig_data.json for headtrack/paper/make_xlsx.py)
    FD = {}
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

    # Fig 2: one quick turn, reference at capture time, trackers as the app receives them
    fs = spec.get("fig_turn")
    if fs:
        s0, r0, ser0 = next(x for x in data if x[0]["label"] == fs["session"])
        fig, ax = plt.subplots(figsize=(6.3, 2.8))
        t0 = ser0["t"][0] + fs["t_start"]  # t_start: seconds into the analysis grid
        w0, w1 = (t0 + x for x in fs["hold"])  # a held pose: each curve is shifted to the reference there
        views = {name: aligned_app_view(s0["path"], r0, name) for name in ("reference",) + tuple(n for n in TRACKERS if n in r0["trackers"])}
        tr, ar = views["reference"]
        ref_hold = np.nanmedian(ar[(tr >= w0) & (tr <= w1), 0])
        FD["turn"] = {"session": fs["session"], "series": []}
        for name, (t, ang) in views.items():
            off = np.nanmedian(ang[(t >= w0) & (t <= w1), 0]) - ref_hold
            m = (t >= t0) & (t <= t0 + fs["dur"]) & np.isfinite(ang[:, 0])
            FD["turn"]["series"].append({"name": name, "label": LABEL[name], "steps": name != "reference",
                                         "t_ms": ((t[m] - t0) * 1e3).tolist(), "yaw_deg": (ang[m, 0] - off).tolist()})
            ax.plot((t[m] - t0) * 1e3, ang[m, 0] - off, lw=1.6 if name == "reference" else 1.2, color=COLOR[name],
                    label=LABEL[name] + (" (capture time)" if name == "reference" else " (as received)"),
                    drawstyle="steps-post" if name != "reference" else "default")
        ax.set_xlabel("time (ms)")
        ax.set_ylabel("yaw (deg)")
        ax.legend(frameon=False, fontsize=7.5)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "fig2_turn.png"), dpi=300)
        plt.close(fig)

    # Fig 3: latency breakdown per tracker
    fig, ax = plt.subplots(figsize=(6.3, 2.4))
    names = list(TRACKERS)
    lag = [T[n]["lag_impulsive_ms"]["median"] for n in names]
    m2d = [T[n]["motion_to_data_median_ms"]["median"] for n in names]
    y = np.arange(len(names))
    ax.barh(y, m2d, color=[COLOR[n] for n in names], alpha=0.9, zorder=2)
    # each latency session as a marker, one shape per day: the camera delay changed between days
    lat_rows = [row for row in PS if "latency" in row["roles"]]
    days = sorted({row["day"] for row in lat_rows})
    shapes = ("o", "^", "s", "D")
    top = list(m2d)
    FD["latency"] = {"band_ms": [40, 75], "trackers": [{"name": n, "label": LABEL[n], "median_ms": m2d[i], "sessions": [
        {"label": row["label"], "day": row["day"], "ms": row[n + "_motion_to_data_ms"]} for row in lat_rows
        if row.get(n + "_motion_to_data_ms") is not None]} for i, n in enumerate(names)]}
    for i, n in enumerate(names):
        for d in days:
            v = [row[n + "_motion_to_data_ms"] for row in lat_rows if row["day"] == d and row.get(n + "_motion_to_data_ms") is not None]
            if v:
                ax.scatter(v, [i + (days.index(d) - (len(days) - 1) / 2) * 0.22] * len(v), s=11, marker=shapes[days.index(d) % 4],
                           facecolor="white", edgecolor="#222222", linewidth=0.7, zorder=3,
                           label=("sessions, day %d" % d) if i == 0 else None)
                top[i] = max(top[i], max(v))
        ax.text(top[i] + 2, i, "%.0f ms" % m2d[i], va="center", fontsize=8)
    if len(days) > 1:
        ax.legend(frameon=False, fontsize=7, loc="upper right")
    ax.axvspan(40, 75, color="#999999", alpha=0.18, lw=0, zorder=0)
    ax.text(57.5, -0.62, "thresholds of the most sensitive listeners", ha="center", va="bottom", fontsize=7, color="#555555")
    ax.set_yticks(y)
    ax.set_yticklabels([LABEL[n] for n in names])
    ax.set_xlabel("motion-to-data delay, median (ms)")
    ax.set_xlim(0, max(top) * 1.25)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig3_latency.png"), dpi=300)
    plt.close(fig)

    # Fig 4: range: (a) share of time with a pose, (b) median absolute yaw error, by reference |yaw|
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(6.3, 2.5))
    xs = np.arange(len(YAW_BINS))
    w = 0.38
    for k, name in enumerate(("mediapipe", "vision")):
        vals = [RG[name]["%d-%d" % b]["share"] or 0 for b in YAW_BINS]
        ax.bar(xs + (k - 0.5) * w, [100 * v for v in vals], w, color=COLOR[name], label=LABEL[name])
    ax.set_ylabel("time with a pose (%)")
    ax.set_ylim(0, 128)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_title("(a)", fontsize=9, loc="left")
    for name in TRACKERS:
        bx.plot(xs, [RG["yaw_error_by_bin"][name]["%d-%d" % b] for b in YAW_BINS], "o-", ms=3.5, color=COLOR[name],
                label=LABEL[name])
    FD["range"] = {"bins": ["%d-%d" % b for b in YAW_BINS],
                   "share_pct": {LABEL[n]: [100 * (RG[n]["%d-%d" % b]["share"] or 0) for b in YAW_BINS] for n in ("mediapipe", "vision")},
                   "yaw_error_deg": {LABEL[n]: [RG["yaw_error_by_bin"][n]["%d-%d" % b] for b in YAW_BINS] for n in TRACKERS}}
    bx.set_ylabel("median |yaw error| (deg)")
    bx.set_title("(b)", fontsize=9, loc="left")
    for a in (ax, bx):
        a.set_xticks(xs)
        a.set_xticklabels(["%d–%d" % b for b in YAW_BINS], fontsize=7.5)
        a.set_xlabel("|yaw| from the reference (deg)")
    ax.legend(frameon=False, fontsize=7.5, loc="upper center", ncol=2)
    bx.legend(frameon=False, fontsize=7.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "fig4_range.png"), dpi=300)
    plt.close(fig)

    # Fig 5: drift, one curve per drift session (all samples drawn only when there is one)
    if drift:
        fig, ax = plt.subplots(figsize=(6.3, 2.6 if len(drift) == 1 else 3.6))
        marks = ("o-", "^--", "D:", "v-.")
        FD["drift"] = []
        for k, (s, r, ser) in enumerate(drift):
            hp = apkey(r)
            t, ref, ap = ser["t"], ser["reference"], ser[hp]
            ok = np.isfinite(ref[:, 0]) & np.isfinite(ap[:, 0])
            e = h.wrap_deg(ap[:, 0] - ref[:, 0])
            tm = (t - t[0]) / 60.0
            tag = "" if len(drift) == 1 else " (%s)" % s["label"]
            if len(drift) == 1:
                ax.plot(tm[ok], e[ok] - np.median(e[ok & (tm < 1)]), lw=0.4, color=COLOR[hp], alpha=0.5)
            mins = DR[s["label"] + "_per_minute"]["median_yaw_error_deg"]
            base = mins[0]
            for j in (r.get("heading_jumps") or [])[:1]:  # the curve stops here; mark where and how far it jumped
                xj = (j["t_s"] - t[0]) / 60.0
                ax.axvline(xj, ls=":", lw=0.8, color=COLOR[hp])
                last = [m for m in mins if m is not None][-1] - base
                ax.annotate("heading jump, %.0f°" % j["deg"], (xj, last), xytext=(-3, 4), textcoords="offset points",
                            ha="right", va="bottom", rotation=90, fontsize=6.5, color=COLOR[hp])
            fd = {"session": s["label"], "headphones": LABEL[hp], "minute_mid": (np.arange(len(mins)) + 0.5).tolist(),
                  "headphones_minus_ref_deg": [m - base if m is not None else None for m in mins], "mediapipe_minus_ref_deg": None,
                  "heading_jump": [{"min": (j["t_s"] - t[0]) / 60.0, "deg": j["deg"]} for j in (r.get("heading_jumps") or [])[:1]]}
            FD["drift"].append(fd)
            ax.plot(np.arange(len(mins)) + 0.5, [m - base if m is not None else np.nan for m in mins], marks[k % 4], ms=3,
                    color=COLOR[hp], label=("%s yaw minus reference yaw, per-minute median" % LABEL[hp] if len(drift) == 1
                                            else "%s minus reference" % LABEL[hp] + tag))
            if "mediapipe" in ser:
                em = h.wrap_deg(ser["mediapipe"][:, 0] - ref[:, 0])
                okm = ok & np.isfinite(em)
                mm = [np.median(em[okm & (tm >= i) & (tm < i + 1)]) if (okm & (tm >= i) & (tm < i + 1)).any() else np.nan
                      for i in range(len(mins))]
                fd["mediapipe_minus_ref_deg"] = [float(v) if np.isfinite(v) else None for v in np.array(mm) - mm[0]]
                ax.plot(np.arange(len(mins)) + 0.5, np.array(mm) - mm[0], marks[k % 4].replace("o", "s"), ms=3,
                        color=COLOR["mediapipe"], alpha=1.0 if len(drift) == 1 else 0.7,
                        label=("MediaPipe yaw minus reference yaw, per-minute median" if len(drift) == 1
                               else "MediaPipe minus reference" + tag))
        ax.set_xlabel("time (min)")
        ax.set_ylabel("yaw difference (deg)")
        if len(drift) == 1:
            ax.legend(frameon=False, fontsize=7.5, loc="upper left")
        else:  # below the axes, so the legend never covers a curve
            ax.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2)
        fig.tight_layout()
        fig.savefig(os.path.join(outdir, "fig5_drift.png"), dpi=300)
        plt.close(fig)

    fig1_architecture(os.path.join(outdir, "fig1_architecture.png"))
    json.dump(FD, open(os.path.join(outdir, "fig_data.json"), "w"), indent=1)

    # ---- tables.md
    def f(x, nd=1):
        return "–" if x is None else ("%." + str(nd) + "f") % x

    def mr(d, nd=1, scale=1.0):
        if not d:
            return "–"
        if d["n"] == 1 or abs(d["max"] - d["min"]) < 10 ** -nd:
            return f(d["median"] * scale, nd)
        return "%s (%s to %s)" % (f(d["median"] * scale, nd), f(d["min"] * scale, nd), f(d["max"] * scale, nd))

    L = ["# Results tables (generated by paper_results.py; do not edit by hand)", ""]
    L += ["Sessions: " + "; ".join("%s = %s (%s, %.0f s)" % (s["label"], os.path.basename(s["path"]), s.get("condition", ""),
                                                           N["duration_s"][s["label"]]) for s, _, _ in data), ""]
    L += ["## Table 1. Delivery, latency and cost (median; range over sessions in brackets)", "",
          "| | %s |" % " | ".join(LABEL[n] for n in TRACKERS), "|---" * (len(TRACKERS) + 1) + "|"]
    row = lambda title, key, nd=1, sc=1.0: L.append("| %s | %s |" % (title, " | ".join(mr(T[n].get(key), nd, sc) for n in TRACKERS)))
    row("Update rate (Hz)", "rate_hz")
    row("Time with a pose (%)", "valid_share", 0, 100)
    row("Arrivals within 1 ms of the previous (%)", "burst_share", 0, 100)
    row("Gap between arrivals, p99 (ms)", "arrival_gap_p99_ms", 0)
    row("Timestamp lag behind reference, impulsive (ms)", "lag_impulsive_ms")
    row("Motion-to-data delay, median (ms)", "motion_to_data_median_ms", 0)
    row("Motion-to-data delay, p95 (ms)", "motion_to_data_p95_ms", 0)
    hps = [n for n in TRACKERS if n.startswith("airpods")]
    L.append("| Timestamp to arrival (headphones) / capture to delivery (camera), median (ms) | %s |"
             % " | ".join([mr(T[n]["timestamp_to_arrival_ms"], 0) for n in hps] +
                          [mr(T[n]["capture_to_delivered_ms"], 0) for n in ("mediapipe", "vision")]))
    L.append("| Processing per frame, median (ms) | %s%s | %s |" % ("– | " * len(hps), mr(T["cost"]["mediapipe_inference_ms"]),
                                                                    mr(T["cost"]["vision_compute_ms"])))
    L.append("| CPU, share of one core (%%) | %s%s | recorder incl. Vision %s |"
             % ("– (in recorder) | " * len(hps), mr(T["cost"]["mediapipe_cpu_share_one_core"], 0, 100),
                mr(T["cost"]["recorder_cpu_share_one_core"], 0, 100)))
    L += ["", "## Table 2. Agreement with the reference (deg): mean absolute error, Bland–Altman bias [95 % limits]", "",
          "| Tracker | State | Yaw | Pitch | Roll |", "|---|---|---|---|---|"]
    for n in TRACKERS:
        for sub in ("still", "moving"):
            cells = []
            for ax in ("yaw", "pitch", "roll"):
                b = AG[n][sub][ax]
                cells.append("–" if not b else "%s; %s [%s, %s]" % (f(b["mae"]), f(b["bias"]), f(b["loa_low"]), f(b["loa_high"])))
            L.append("| %s | %s | %s |" % (LABEL[n], sub, " | ".join(cells)))
    L += ["", "AirPods yaw: linear drift removed per session before these statistics (drift in Figure 6)."]
    off = spec.get("utc_offset_h", 7)

    def local_start(path):
        st = datetime.datetime.strptime(os.path.basename(os.path.normpath(path))[:16], "%Y%m%dT%H%M%SZ")
        return (st + datetime.timedelta(hours=off)).strftime("%H:%M")
    L += ["", "## Table 3. Headphone yaw drift against the reference per session (linear fit over still periods)", "",
          "| Session | Day | Start (UTC%+d) | Audio | Minutes | Drift (deg/min) | r² | Note |" % off, "|---|---|---|---|---|---|---|---|"]
    for s, r, ser in data:
        yd = r["trackers"].get(apkey(r), {}).get("yaw_drift")
        if not yd or "latency" in s["roles"] and "core" not in s["roles"]:
            continue  # pilot and quick-turn sessions: too short for a drift fit
        cut = yd.get("until_heading_jump_s") is not None
        L.append("| %s | %s | %s | %s | %.1f | %+.1f | %.3f | %s |" % (
            s["label"], s.get("day", 1), local_start(s["path"]), audio_of(s) or "–", yd["span_min"] if cut else N["duration_s"][s["label"]] / 60.0,
            yd["deg_per_min"], yd["r2"], "up to the heading jump" if cut else ""))
    for c in DC:
        L.append("| %s | %s | %s | %s | %.1f | %+.1f | %.3f | %s |" % (
            c["label"], c["day"], local_start(c["path"]), c["audio"] or "–", c["minutes"], c["deg_per_min"], c["r2"], c["note"]))
    if DC:
        L += ["", "Drift checks (rows %s): headphone rate %s Hz, %s gaps/min; heading jumps %d." % (
            ", ".join(c["label"] for c in DC), "/".join("%.1f" % c["rate_hz"] for c in DC),
            "/".join("%.0f" % c["gaps_per_min"] for c in DC), sum(c["heading_jumps"] for c in DC))]
    L += ["", "## Table S1. Per session (geodesic medians in deg, headphones not detrended; motion-to-data in ms, all steps)", "",
          "| Session | Day | Headphones | Audio | Min | Headphone Hz | Gaps/min | Drift (deg/min) | M2D HP / MP / Vision | Still HP / MP / V | Moving HP / MP / V | Sync best / bound (ms) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for row in PS:
        tri = lambda key: " / ".join(f(row[n + key], 0 if "motion" in key else 1) for n in ("airpods", "mediapipe", "vision"))
        L.append("| %s | %s | %s | %s | %.1f | %s | %s | %s | %s | %s | %s | %s / %s |" % (
            row["label"], row["day"], LABEL[row["headphones"]], row["audio"] or "–", row["minutes"], f(row["airpods_rate_hz"]), f(row["airpods_gaps_per_min"]),
            f(row["airpods_drift_deg_per_min"]), tri("_motion_to_data_ms"), tri("_geodesic_still_deg"), tri("_geodesic_moving_deg"),
            f(row["sync_best_ms"]), f(row["sync_bound_ms"])))
    for hp, conds in N["by_audio"].items():
        L += ["", "%s by audio (core and drift sessions): " % LABEL[hp] + "; ".join(
            "%s %s: %s Hz, %s gaps/min" % (c, ",".join(conds[c]["sessions"]) or "–", mr(conds[c]["rate_hz"]),
                                            mr(conds[c]["gaps_per_min"])) for c in ("silent", "music"))]
    open(os.path.join(outdir, "tables.md"), "w").write("\n".join(L) + "\n")
    print("wrote", outdir)


if __name__ == "__main__":
    main()
