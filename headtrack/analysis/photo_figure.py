"""The side paper's test-process figure from a photo session (run_session.sh --photos):
(a) eight webcam crops picked by photo_crops.py, labelled with the ARKit angles of that still,
(b) iPhone screen shots at the poses of (a)'s first row, cropped to the head, and
(c) four recorder window shots from different parts of the protocol.

    headtrack/.venv/bin/python headtrack/analysis/photo_figure.py runs/<session> <out.png>

Run photo_crops.py on the session first. Run from the repository root.
"""
import sys, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

d, outf = sys.argv[1], sys.argv[2]
# (crop, label, [(word, ARKit angle whose size is shown)]); four per row
PANELS = [("01_front", "front", []),
          ("09_left_screen_edge", "left to screen edge", [("", "yaw_deg")]),
          ("14_right_screen_edge", "right to screen edge", [("", "yaw_deg")]),
          ("22_up_left", "", [("up", "pitch_deg"), ("left", "yaw_deg")]),
          ("04_tilt", "tilt to shoulder", [("", "roll_deg")]),
          ("17_up", "up", [("", "pitch_deg")]),
          ("20_down", "down", [("", "pitch_deg")]),
          ("25_down_right", "", [("down", "pitch_deg"), ("right", "yaw_deg")])]
# iPhone shots nearest the first-row webcam stills (0.9 s earlier to 0.1 s later), and the head crop (rows, columns)
IPHONE_POSES = [("ref_20261027T085824.758Z.jpg", "front"),
                ("ref_20261027T085925.264Z.jpg", "left to screen edge"),
                ("ref_20261027T085948.760Z.jpg", "right to screen edge"),
                ("ref_20261027T090029.746Z.jpg", "up and left")]
IPHONE_CROP = (slice(800, 1600), slice(300, 1100))
# mid-step shots of steps 2 (slow turns), 13 (held turn), 29 (fast turns) and 51 (head shakes)
RECORDER = ["photos/recorder_20261027T085835.939Z.png", "photos/recorder_20261027T085943.838Z.png",
            "photos/recorder_20261027T090120.736Z.png", "photos/recorder_20261027T090255.934Z.png"]
# window rows kept: the prompt, then the step counter and status lines (the black between is cut)
REC_ROWS = [(50, 130), (256, 462)]

ang = {r["crop"]: r for r in csv.DictReader(open(d + "/figure_crops/angles.csv"))}
recs = [np.vstack([im[a:b] for a, b in REC_ROWS]) for im in (mpimg.imread(d + "/" + f) for f in RECORDER)]

plt.rcParams.update({"font.size": 9})
W = 6.3
gap, lab = 0.04, 0.22             # inches: gap between images, label under a crop
tile = (W - 3 * gap) / 4          # webcam and iPhone crop
head = 0.2                        # room for a panel letter
rw = (W - gap) / 2                # width of one recorder shot
rh = rw * recs[0].shape[0] / recs[0].shape[1]
H = head + 2 * tile + 2 * lab + head + tile + lab + head + 2 * rh + gap
fig = plt.figure(figsize=(W, H))


def axes(x, y, w, h):
    """Axes at (x, y, w, h) in inches from the bottom left, no frame."""
    ax = fig.add_axes((x / W, y / H, w / W, h / H))
    ax.set_axis_off()
    return ax


def letter(x, y, s):
    fig.text(x / W, (y + 0.03) / H, s, fontsize=9, va="bottom")


def tile_at(i, top, img, label):
    r, c = divmod(i, 4)
    x, y = c * (tile + gap), top - tile - r * (tile + lab)
    axes(x, y, tile, tile).imshow(img)
    fig.text((x + tile / 2) / W, (y - 0.03) / H, label, ha="center", va="top", fontsize=8)


top = H - head
for i, (name, label, cols) in enumerate(PANELS):
    sizes = [("%s %.0f°" % (w, abs(float(ang[name][c])))).strip() for w, c in cols]
    tile_at(i, top, mpimg.imread("%s/figure_crops/%s.jpg" % (d, name)), ", ".join([label] * bool(label) + sizes))
letter(0, top, "(a)")

top -= 2 * (tile + lab) + head
for i, (f, label) in enumerate(IPHONE_POSES):
    tile_at(i, top, mpimg.imread(d + "/photos_iphone/" + f)[IPHONE_CROP], label)
letter(0, top, "(b)")

for i, rec in enumerate(recs):
    r, c = divmod(i, 2)
    axes(c * (rw + gap), (1 - r) * (rh + gap), rw, rh).imshow(rec)
letter(0, 2 * rh + gap, "(c)")
fig.savefig(outf, dpi=300)
print("%s  %.2f x %.2f in" % (outf, W, H))
