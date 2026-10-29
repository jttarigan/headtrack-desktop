"""Removes borrowed headphones' Bluetooth names from a session folder.

    headtrack/.venv/bin/python headtrack/analysis/redact_names.py runs/<session>

A borrowed device's Bluetooth name usually carries its owner's first name
(e.g. "<Name> - Find My"), and the recorder, the probe and headphones.json
log it. Every connected headphone name that is not one of the lab's own
devices is replaced by "[borrowed headphones]" in the session's text and JSON
files. run_session.sh calls this after each session. Prints what it changed.
"""

import glob
import json
import os
import sys

LAB_DEVICES = {"JTAIRPDMAX2"}  # the lab's own AirPods Max 2 (LAB.md)
REPLACEMENT = "[borrowed headphones]"


def main():
    session = sys.argv[1]
    hp = os.path.join(session, "headphones.json")
    names = set()
    if os.path.exists(hp):
        for d in json.load(open(hp)).get("connected_headphones", []):
            if d.get("name") and d["name"] not in LAB_DEVICES:
                names.add(d["name"])
    names |= {n for n in sys.argv[2:]}  # extra names given by hand
    if not names:
        return 0
    changed = []
    for path in sorted(glob.glob(os.path.join(session, "*"))):
        if not path.endswith((".txt", ".json", ".md", ".log")) or not os.path.isfile(path):
            continue
        text = open(path, encoding="utf-8", errors="surrogateescape").read()
        new = text
        for n in names:
            new = new.replace(n, REPLACEMENT)
            new = new.replace(json.dumps(n)[1:-1], REPLACEMENT)  # as escaped inside JSON strings
        if new != text:
            open(path, "w", encoding="utf-8", errors="surrogateescape").write(new)
            changed.append(os.path.basename(path))
    print("redacted %d borrowed headphone name(s) in: %s" % (len(names), ", ".join(changed) or "nothing"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
