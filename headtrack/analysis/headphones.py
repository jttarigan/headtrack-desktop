"""Record the connected Bluetooth headphones (name, firmware) for one session.

    headtrack/.venv/bin/python headtrack/analysis/headphones.py runs/<session>

Writes <session>/headphones.json from `system_profiler SPBluetoothDataType`
and prints one line. Never fails the session: errors go into the file.
"""

import json
import os
import subprocess
import sys

import htlib as h


def main():
    out_dir = sys.argv[1]
    rec = {"source": "system_profiler SPBluetoothDataType -json"}
    try:
        sp = subprocess.run(["system_profiler", "SPBluetoothDataType", "-json"], capture_output=True, text=True,
                            timeout=30, check=True).stdout
        rec["connected_headphones"] = h.headphones_from_system_profiler(json.loads(sp))
        line = ", ".join("%s firmware %s" % (d["name"], d["firmware"]) for d in rec["connected_headphones"])
        print("headphones: %s" % (line or "none connected"))
    except Exception as e:  # recorded, not fatal
        rec["error"] = repr(e)
        print("headphones: not recorded (%s)" % e)
    with open(os.path.join(out_dir, "headphones.json"), "w") as f:
        json.dump(rec, f, indent=1)
        f.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
