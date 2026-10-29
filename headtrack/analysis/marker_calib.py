"""Mac camera intrinsics from a ChArUco calibration session.

    headtrack/run_session.sh calib --marker calib      (records charuco.npz)
    headtrack/.venv/bin/python headtrack/analysis/marker_calib.py runs/<session>

Writes headtrack/protocols/camera_<w>x<h>_<session>.json with the camera
matrix K, the distortion coefficients and the RMS reprojection error. The
board (headtrack/protocols/marker_sheet.pdf, page 2) must have been printed at
100 % and kept flat; its square size is taken from marker_tent.json.
"""

import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROTOCOLS = os.path.join(HERE, "..", "protocols")


def board():
    t = json.load(open(os.path.join(PROTOCOLS, "marker_tent.json")))["board"]
    return cv2.aruco.CharucoBoard(tuple(t["squares"]), t["square_mm"] / 1000, t["marker_mm"] / 1000,
                                  cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_100))


def calibrate(corners, ids, size, b, max_views=60):
    """corners/ids per view -> (rms, K, dist, views used). Views are thinned
    evenly to at most max_views, since neighbouring frames are near copies."""
    views = [(c, i) for c, i in zip(corners, ids) if len(i) >= 8]
    if len(views) > max_views:
        views = [views[k] for k in np.linspace(0, len(views) - 1, max_views).astype(int)]
    obj, img = [], []
    for c, i in views:
        o, p = b.matchImagePoints(c.reshape(-1, 1, 2), i.reshape(-1, 1))
        if o is not None and len(o) >= 6:
            obj.append(o.astype(np.float32))
            img.append(p.astype(np.float32))
    rms, K, dist, _, _ = cv2.calibrateCamera(obj, img, size, None, None)
    return rms, K, dist, len(obj)


def main():
    session = sys.argv[1]
    d = np.load(os.path.join(session, "charuco.npz"), allow_pickle=True)
    w, h = int(d["width"]), int(d["height"])
    rms, K, dist, n = calibrate(list(d["corners"]), list(d["ids"]), (w, h), board())
    name = "camera_%dx%d_%s.json" % (w, h, os.path.basename(os.path.normpath(session)))
    out = {"width": w, "height": h, "K": K.tolist(), "dist": dist.reshape(-1).tolist(), "rms_px": rms,
           "views": n, "session": os.path.basename(os.path.normpath(session)), "opencv": cv2.__version__}
    with open(os.path.join(PROTOCOLS, name), "w") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    print("rms %.3f px from %d views; fx %.1f fy %.1f cx %.1f cy %.1f -> %s"
          % (rms, n, K[0, 0], K[1, 1], K[0, 2], K[1, 2], name))
    if rms > 1.0:
        print("warning: rms above 1 px; check that the board was flat and printed at 100 %")


if __name__ == "__main__":
    main()
