"""Make the laser check's targets file from Jos's measurements.

    headtrack/.venv/bin/python headtrack/analysis/laser_targets.py --a-mm 540 --b-mm 110 --h-mm 60 \\
        --sagitta-mm 54 [--screen-mm 802 330] [--screen-px 3440 1440]

Writes headtrack/protocols/laser_check_targets.json, which run_session.sh hands
to the recorder (--targets) for the laser_check protocol. Measurements and
geometry: headtrack/protocols/laser_check.md; formulas: htlib.py, laser check.
"""

import argparse
import json
import os
import sys

import htlib as h

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a-mm", type=float, required=True, help="laser tip to its spot on the screen centre")
    ap.add_argument("--b-mm", type=float, required=True, help="ear canal forward to the laser tip")
    ap.add_argument("--h-mm", type=float, required=True, help="laser height above the ear canal")
    ap.add_argument("--sagitta-mm", type=float, required=True, help="gap under a straight edge at the screen centre; 0 = flat")
    ap.add_argument("--screen-mm", type=float, nargs=2, default=(802, 330), metavar=("W", "H"),
                    help="lit picture size along the surface (default: what macOS reports for the G34w-30)")
    ap.add_argument("--screen-px", type=int, nargs=2, default=(3440, 1440), metavar=("W", "H"))
    ap.add_argument("--protocol", default=os.path.join(ROOT, "headtrack/protocols/laser_check.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "headtrack/protocols/laser_check_targets.json"))
    args = ap.parse_args()

    with open(args.protocol) as f:
        proto = json.load(f)
    labels = sorted({s["target"] for s in proto["steps"] if "target" in s}, key=lambda l: (h.laser_label(l)[0], h.laser_label(l)[1]))
    g = h.laser_geometry(args.a_mm + args.b_mm, args.h_mm, args.sagitta_mm, args.screen_mm, args.screen_px)
    try:
        tg = h.laser_targets(labels, g)
    except ValueError as e:
        print("cannot place the targets: %s" % e)
        return 1
    tg["measured"] = {"a_mm": args.a_mm, "b_mm": args.b_mm, "h_mm": args.h_mm, "sagitta_mm": args.sagitta_mm,
                      "screen_mm": list(args.screen_mm)}
    tg["protocol"] = os.path.relpath(args.protocol, ROOT)
    with open(args.out, "w") as f:
        json.dump(h.to_jsonable(tg), f, indent=2)
        f.write("\n")
    R = g["radius_mm"]
    print("D %.0f mm, h %.0f mm, screen radius %s" % (g["D_mm"], g["h_mm"], "flat" if R == float("inf") else "%.0f mm" % R))
    mm_x = g["screen_mm"][0] / g["screen_px"][0]
    mm_y = g["screen_mm"][1] / g["screen_px"][1]
    cx, cy = g["centre_px"]
    for lab, m in tg["marks"].items():
        x, y = m["px"]
        print("  %-4s %6.1f deg  px (%7.1f, %6.1f)  from centre along the screen: %+6.1f mm across, %+6.1f mm up"
              % (lab, m["angle_deg"], x, y, (x - cx) * mm_x, (cy - y) * mm_y))
    print("wrote %s" % args.out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
