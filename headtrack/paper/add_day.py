"""Appends one recording day to headtrack/paper/sessions.json.

    python3 headtrack/paper/add_day.py --day 2 --silent runs/<id> --music runs/<id> \
        --drift runs/<id> [--marker runs/<id>] [--headphones max2|pro1] [--dry-run]

Labels follow day 1: core sessions C1, C2, ... (silent first, then music),
drift sessions D1, D2, ..., marker sessions M1, ... Each session folder must
hold the files the analysis needs. Run headtrack/paper/update.sh afterwards.
"""

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SPEC = os.path.join(HERE, "sessions.json")
NEEDED = ("events.csv", "ref.csv", "sync.csv", "airpods.csv", "camera.csv", "mediapipe.csv", "session.json")


def check(path):
    missing = [f for f in NEEDED if not os.path.exists(os.path.join(path, f))]
    if missing:
        sys.exit("%s: missing %s" % (path, ", ".join(missing)))
    notes = os.path.join(path, "NOTES.md")
    if os.path.exists(notes) and "not study data" in open(notes).read().lower():
        sys.exit("%s: NOTES.md says 'not study data'" % path)
    return os.path.normpath(path)


def next_label(spec, prefix):
    n = [int(s["label"][len(prefix):]) for s in spec["sessions"]
         if s["label"].startswith(prefix) and s["label"][len(prefix):].isdigit()]
    return "%s%d" % (prefix, max(n, default=0) + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", type=int, required=True)
    ap.add_argument("--silent", help="core protocol, no audio")
    ap.add_argument("--music", help="core protocol, Music.app playing")
    ap.add_argument("--drift", help="20 min desk work, music")
    ap.add_argument("--marker", help="core protocol with the marker tent")
    ap.add_argument("--headphones", default="max2", choices=("max2", "pro1"),
                    help="headphones worn in these sessions (pro1: the borrowed AirPods Pro, 1st generation)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    spec = json.load(open(SPEC))
    have = {s["path"] for s in spec["sessions"]}
    new = []
    for arg, prefix, roles, audio, cond in (
            (a.silent, "C", ["core", "latency"], "silent", "full protocol, silent"),
            (a.music, "C", ["core", "latency"], "music", "full protocol, music"),
            (a.drift, "D", ["drift"], "music", "20 min desk work, music"),
            (a.marker, "M", ["marker"], "silent", "full protocol with marker tent, silent")):
        if not arg:
            continue
        path = check(arg)
        if path in have:
            sys.exit("%s is already in sessions.json" % path)
        s = {"label": next_label({"sessions": spec["sessions"] + new}, prefix), "path": path, "roles": roles,
             "condition": cond, "day": a.day, "audio": audio, "headphones": a.headphones}
        new.append(s)
    if not new:
        sys.exit("nothing to add")
    for s in new:
        print("add %-3s day %d  %-4s %-6s %s" % (s["label"], s["day"], s["headphones"], s["audio"], s["path"]))
    if a.dry_run:
        return
    spec["sessions"] += new
    with open(SPEC, "w") as f:
        json.dump(spec, f, indent=2)
        f.write("\n")
    print("updated", SPEC)


if __name__ == "__main__":
    main()
