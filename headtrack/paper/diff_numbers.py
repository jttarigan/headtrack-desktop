"""Lists the numbers that changed between two numbers.json files, so the
prose of manuscript.md can be updated after new sessions are added.

    python3 headtrack/paper/diff_numbers.py old.json new.json [--rel 0.02]

A value counts as changed when it moves by more than --rel (relative) and
by more than 0.05 in absolute terms (rounding in the text is 0.1 at best).
"""

import argparse
import json
import re


def flat(x, p="", out=None):
    out = {} if out is None else out
    if isinstance(x, dict):
        for k, v in x.items():
            flat(v, "%s.%s" % (p, k) if p else str(k), out)
    elif isinstance(x, list) and x and isinstance(x[0], dict):
        for i, v in enumerate(x):
            flat(v, "%s[%s]" % (p, v.get("label", i)), out)
    else:
        out[p] = x
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("old")
    ap.add_argument("new")
    ap.add_argument("--rel", type=float, default=0.02)
    a = ap.parse_args()
    o, n = flat(json.load(open(a.old))), flat(json.load(open(a.new)))
    changed, added, removed = [], sorted(set(n) - set(o)), sorted(set(o) - set(n))
    for k in sorted(set(o) & set(n)):
        x, y = o[k], n[k]
        if isinstance(x, (int, float)) and isinstance(y, (int, float)) and not isinstance(x, bool):
            if abs(y - x) > 0.05 and abs(y - x) > a.rel * max(abs(x), 1e-9):
                changed.append((k, x, y))
        elif x != y and not k.endswith(".n") and not k.startswith("sessions"):
            changed.append((k, x, y))
    skip = (".n", ".loa_low", ".loa_high", ".rms")  # follow from the others
    print("%d changed (rel > %.0f %%), %d new, %d removed\n" % (len(changed), 100 * a.rel, len(added), len(removed)))
    for k, x, y in changed:
        if k.endswith(skip):
            continue
        fx = "%.3g" % x if isinstance(x, float) else str(x)
        fy = "%.3g" % y if isinstance(y, float) else str(y)
        print("  %-70s %10s -> %s" % (k, fx[:40], fy[:40]))
    for title, keys in (("new", added), ("removed", removed)):
        if keys:
            groups = {}
            for k in keys:
                g = re.split(r"[.\[]", k)[0]
                sub = re.split(r"[.]", k)[1] if "." in k else ""
                groups.setdefault(g, set()).add(sub)
            print("\n%s: " % title + "; ".join("%s (%s)" % (g, ", ".join(sorted(s))[:120]) for g, s in sorted(groups.items())))


if __name__ == "__main__":
    main()
