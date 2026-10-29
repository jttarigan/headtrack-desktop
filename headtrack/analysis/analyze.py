"""Analyse one head-tracking session (HEADTRACK_STUDY.md).

    headtrack/.venv/bin/python headtrack/analysis/analyze.py runs/<session> [--plot]

Writes <session>/analysis.json (all numbers; conventions in htlib.py) and, with
--plot, <session>/analysis_angles.png. Prints a short summary. Sources that
are missing or empty are reported in "notes" and skipped.
"""

import argparse
import json
import os
import sys

import htlib as h


def f(x, fmt="%.1f"):
    return "-" if x is None else fmt % x


def summary_lines(res):
    out = ["session %s" % res["session"]]
    if res.get("headphones"):
        out.append("headphones: " + ", ".join("%s firmware %s" % (d["name"], d["firmware"])
                                              for d in res["headphones"]))
    cs = res.get("clock_sync")
    if cs:
        out.append("clock sync: %d pings (%d used), drift %s ppm, RTT min/median %s/%s ms, residual %s ms, "
                   "offset error at most %s ms (best case %s)"
                   % (cs["pings"], cs["selected"], f(cs["drift_ppm"], "%.2f"), f(cs["rtt_ms"]["min"], "%.2f"),
                      f(cs["rtt_ms"]["median"], "%.2f"), f(cs["residual_std_ms"], "%.3f"),
                      f(cs.get("bound_ms"), "%.2f"), f(cs["uncertainty_ms"], "%.2f")))
    ref = res.get("reference")
    if ref:
        d = ref["delivery"]
        n = ref.get("noise_at_rest_deg") or {}
        out.append("reference: %d poses, tracked %.0f%%, %s Hz, capture->delivered %s ms, iPhone->Mac %s ms, "
                   "noise at rest y/p/r %s/%s/%s deg"
                   % (d["samples"], 100 * ref["tracked_share"], f(d.get("rate_hz")),
                      f(ref["internal_delay_ms"]["capture_to_delivered"]["median"]),
                      f(ref["internal_delay_ms"]["sent_to_mac_receive"]["median"]),
                      f(n.get("yaw"), "%.3f"), f(n.get("pitch"), "%.3f"), f(n.get("roll"), "%.3f")))
    if "neutral" in res:
        out.append("neutral pose: %s; still periods %d (%.0f s)"
                   % (res["neutral"]["source"], res["still_periods"]["count"], res["still_periods"]["seconds"]))
    for name, t in res["trackers"].items():
        d = t["delivery"]
        line = "%s: %d samples, %s Hz, valid %.0f%%" % (name, d["samples"], f(d.get("rate_hz")),
                                                       100 * (d["valid_share"] or 0))
        lat = t.get("latency_ms")
        if lat and lat.get("sample_time_lag") is not None:
            m2d = lat["motion_to_data"] or {}
            line += ("; lag at sample time %s ms (r %.2f), motion-to-data median %s p95 %s ms, as seen by app %s ms"
                     % (f(lat["sample_time_lag"]), lat["sample_time_corr"], f(m2d.get("median")), f(m2d.get("p95")),
                        f(lat.get("app_view_lag"))))
        out.append(line)
        seg = (lat or {}).get("by_segment")
        if seg:
            out.append("    lag at sample time by protocol step: " + ", ".join(
                "%s %s ms (r %s, %.0f s in %d steps)" % (k, f(v["sample_time_lag"]), f(v["corr"], "%.2f"), v["seconds"],
                                                   v["steps"]) for k, v in seg.items()))
        if "alignment" in t:
            ag = t["agreement_latency_compensated_deg"]
            app = t["agreement_as_seen_by_app_deg"]
            n = t.get("noise_at_rest_deg") or {}
            conv = t.get("convention", {})
            conv_s = ", ".join("%s=%s" % kv for kv in conv.items()) if conv else ""
            out.append("    alignment residual median %s deg%s"
                       % (f(t["alignment"]["residual_deg"]["median"], "%.2f"), (" (%s)" % conv_s) if conv_s else ""))
            for sub in ("still", "moving"):
                a = ag[sub]
                out.append("    %s: MAE y/p/r %s/%s/%s deg, geodesic median %s p95 %s deg"
                           % (sub, f(a["yaw"].get("mae"), "%.2f"), f(a["pitch"].get("mae"), "%.2f"),
                              f(a["roll"].get("mae"), "%.2f"), f(a["geodesic"]["median"], "%.2f"),
                              f(a["geodesic"]["p95"], "%.2f")))
            out.append("    as seen by an app while moving: geodesic median %s p95 %s deg; pose age median %s ms"
                       % (f(app["moving"]["geodesic"]["median"], "%.2f"), f(app["moving"]["geodesic"]["p95"], "%.2f"),
                          f(app["pose_age_ms"]["median"])))
            line = "    noise at rest y/p/r %s/%s/%s deg" % (f(n.get("yaw"), "%.3f"), f(n.get("pitch"), "%.3f"),
                                                              f(n.get("roll"), "%.3f"))
            if "yaw_drift" in t:
                line += "; yaw drift %s deg/min over %s min" % (f(t["yaw_drift"]["deg_per_min"], "%.3f"),
                                                                 f(t["yaw_drift"]["span_min"]))
            out.append(line)
            cov = t["coverage"]["by_abs_yaw"]
            out.append("    coverage by |yaw|: " + ", ".join("%s %s" % (k, f(v["share"] and 100 * v["share"], "%.0f%%"))
                                                          for k, v in cov.items()))
    lc = res.get("laser_check")
    if lc:
        out.append("laser check: %d holds (last %.0f s of each step, spread under %.1f deg)"
                   % (lc["holds"], lc["hold_s"], lc["max_spread_deg"]))
        for name, t in lc["trackers"].items():
            if "error" in t:
                out.append("  %s: %s" % (name, t["error"]))
                continue
            parts = ["%s gain %.3f MAE %s deg (n %d)" % (g, t[g]["gain"], f(t[g]["mae_deg"], "%.2f"), t[g]["n"])
                     for g in ("yaw_left", "yaw_right", "pitch") if g in t]
            out.append("  %s: %s; centre spread %s deg, dropped %d"
                       % (name, "; ".join(parts), f(t.get("centre_spread_deg"), "%.2f"), t["dropped_holds"]))
    for n in res["notes"]:
        out.append("note: " + n)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("session")
    ap.add_argument("--plot", action="store_true", help="also write analysis_angles.png")
    args = ap.parse_args()
    if not os.path.isdir(args.session):
        print("not a directory: %s" % args.session)
        return 1
    res, series = h.analyze_session(args.session, keep_series=args.plot)
    with open(os.path.join(args.session, "analysis.json"), "w") as fp:
        json.dump(h.to_jsonable(res), fp, indent=2)
        fp.write("\n")
    print("\n".join(summary_lines(res)))
    if args.plot and series and "t" in series:
        png = os.path.join(args.session, "analysis_angles.png")
        h.plot_series(series, png, title=res["session"])
        print("plot: %s" % png)
    return 0


if __name__ == "__main__":
    sys.exit(main())
