"""Printable sheet for the marker check of the reference's angle scale
(replaces the laser, decision of 2026-10-15), plus the tent's 3D model.

    headtrack/.venv/bin/python headtrack/analysis/marker_sheet.py

Writes headtrack/protocols/marker_sheet.pdf (A4, print at 100 %) and
headtrack/protocols/marker_tent.json (corner positions in the tent frame).

Page 1: the tent. Two ArUco markers (DICT_4X4_50, ids 0 and 1) on two faces
that fold into a triangular prism: the ridge points at the camera and each face
is turned 30 degrees from the frontal plane, so neither face is ever seen
exactly head-on (where a flat marker's angle is least accurate) and the eight
corners are not coplanar.
Page 2: a ChArUco board (DICT_5X5_100, so its ids never clash with the tent)
for calibrating the Mac camera.
Both pages carry a 100 mm scale bar to check that the printer did not scale.
"""

import json
import math
import os

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "..", "protocols")

A4 = (210.0, 297.0)  # mm
FACE_W, FACE_H = 70.0, 70.0  # one tent face, mm
MARKER = 54.0  # marker side incl. its black border, mm (6 modules of 9 mm)
MARGIN = (FACE_W - MARKER) / 2  # white quiet zone around each marker
FACE_DEG = 30.0  # each face turned this far from the frontal plane
BASE_W = 2 * FACE_W * math.cos(math.radians(FACE_DEG))  # back of the prism
TAB_W = 15.0
TENT_DICT = cv2.aruco.DICT_4X4_50
BOARD_DICT = cv2.aruco.DICT_5X5_100
BOARD_SQUARES = (5, 7)
BOARD_SQUARE, BOARD_MARKER = 30.0, 22.0  # mm


def tent_model():
    """Corners of each marker in the tent frame (mm): x right, y up, z toward
    the camera; origin on the ridge at mid-height. Corner order is ArUco's:
    top-left, top-right, bottom-right, bottom-left, as seen from the front."""
    c, s = math.cos(math.radians(FACE_DEG)), math.sin(math.radians(FACE_DEG))
    near, far, top, bot = MARGIN, MARGIN + MARKER, MARKER / 2, -MARKER / 2

    def left(d, y):  # face left of the ridge, d = distance from the ridge along the face
        return [-d * c, y, -d * s]

    def right(d, y):
        return [d * c, y, -d * s]

    return {
        "0": [left(far, top), left(near, top), left(near, bot), left(far, bot)],
        "1": [right(near, top), right(far, top), right(far, bot), right(near, bot)],
    }


def scale_bar(ax, x, y):
    ax.plot([x, x + 100], [y, y], color="k", lw=1.2)
    for k in range(0, 101, 10):
        ax.plot([x + k, x + k], [y, y + (3 if k % 50 else 5)], color="k", lw=0.8)
    ax.text(x + 50, y - 4, "100 mm: measure this before use (must be 100 ± 0.5 mm)", ha="center", va="top", fontsize=8)


def page(pdf, draw, landscape=False):
    w, h = (A4[1], A4[0]) if landscape else A4
    fig = plt.figure(figsize=(w / 25.4, h / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.axis("off")
    draw(ax)
    pdf.savefig(fig)
    plt.close(fig)


def draw_tent(ax):
    d = cv2.aruco.getPredefinedDictionary(TENT_DICT)
    x0, y0 = 10.0, 58.0  # strip origin (lower left), mm; page 1 is A4 landscape
    widths = [TAB_W, BASE_W, FACE_W, FACE_W]
    # layout left to right: glue tab | base (back) | face id 0 | face id 1
    xs = np.cumsum([x0] + widths)
    ax.add_patch(plt.Rectangle((x0, y0), sum(widths), FACE_H, fill=False, lw=0.8))
    for xv in xs[1:-1]:
        ax.plot([xv, xv], [y0, y0 + FACE_H], ls=(0, (4, 3)), color="k", lw=0.6)
    for k, mid in enumerate((xs[2] + FACE_W / 2, xs[3] + FACE_W / 2)):
        img = cv2.aruco.generateImageMarker(d, k, 600, borderBits=1)
        ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest",
                  extent=(mid - MARKER / 2, mid + MARKER / 2, y0 + MARGIN, y0 + MARGIN + MARKER))
        ax.text(mid, y0 + FACE_H + 2, "face %d (id %d)  ↑ top" % (k, k), ha="center", va="bottom", fontsize=8)
    ax.text(xs[0] + TAB_W / 2, y0 + FACE_H / 2, "glue", rotation=90, ha="center", va="center", fontsize=8)
    ax.text(xs[1] + BASE_W / 2, y0 + FACE_H / 2, "BASE (back, against the headband)\n%.1f mm" % BASE_W,
            ha="center", va="center", fontsize=8)
    ax.text(148.5, 200, "Marker tent for the head-tracking check (EchoHunt side paper)", ha="center", fontsize=11, weight="bold")
    lines = [
        "1. Print at 100 % (\"actual size\", no fit-to-page). Check the scale bar below.",
        "2. Glue the sheet to thin card (cereal box) before cutting, so the faces stay flat.",
        "3. Cut along the solid outline. Score and fold every dashed line the same way, printed",
        "   side outside, into a triangular tube: the line between the markers becomes the ridge.",
        "4. Glue the tab inside, under the free edge of face 1. Faces flat, edges straight.",
        "5. Check: the base should measure %.1f mm between the two folds; the ridge angle is 120°." % BASE_W,
        "6. Fix the base to the front of an elastic headband, high on the forehead at the hairline,",
        "   ridge vertical and pointing straight forward, markers upright. It must not wobble.",
        "Do not cover the eyebrows or eyes: the face trackers must still see the face.",
    ]
    for i, l in enumerate(lines):
        ax.text(12, 192 - 5.2 * i, l, fontsize=8, va="top", family="DejaVu Sans")
    scale_bar(ax, 20, 35)
    # top view of the folded tent
    c, s = math.cos(math.radians(FACE_DEG)), math.sin(math.radians(FACE_DEG))
    ox, oy = 225, 32
    pts = np.array([[-FACE_W * c, -FACE_W * s], [0, 0], [FACE_W * c, -FACE_W * s], [-FACE_W * c, -FACE_W * s]]) * 0.5
    ax.plot(ox + pts[:, 0], oy + pts[:, 1], color="k", lw=1)
    ax.annotate("camera", (ox, oy + 6), (ox, oy + 18), ha="center", fontsize=8, arrowprops=dict(arrowstyle="<-"))
    ax.text(ox, oy - 26, "top view when folded (half size): ridge toward the camera, faces 30° from frontal",
            ha="center", fontsize=8)


def draw_board(ax):
    d = cv2.aruco.getPredefinedDictionary(BOARD_DICT)
    board = cv2.aruco.CharucoBoard(BOARD_SQUARES, BOARD_SQUARE / 1000, BOARD_MARKER / 1000, d)
    wmm, hmm = BOARD_SQUARES[0] * BOARD_SQUARE, BOARD_SQUARES[1] * BOARD_SQUARE
    img = board.generateImage((int(wmm * 10), int(hmm * 10)), marginSize=0, borderBits=1)
    x0, y0 = (A4[0] - wmm) / 2, 40
    ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest", extent=(x0, x0 + wmm, y0, y0 + hmm))
    ax.text(105, 285, "Camera calibration board (ChArUco 5 × 7, 30 mm squares)", ha="center", fontsize=11, weight="bold")
    ax.text(105, 278, "Print at 100 %, glue flat on card. Hold it in front of the Mac camera when asked, at several",
            ha="center", fontsize=8.2)
    ax.text(105, 273, "distances and tilts. The board must stay flat; a bent board spoils the calibration.",
            ha="center", fontsize=8.2)
    scale_bar(ax, 55, 30)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    pdf_path = os.path.join(OUT_DIR, "marker_sheet.pdf")
    with PdfPages(pdf_path) as pdf:
        page(pdf, draw_tent, landscape=True)
        page(pdf, draw_board)
    model = {
        "units": "mm",
        "frame": "x right, y up, z toward the camera; origin on the ridge at mid-height",
        "dictionary": "DICT_4X4_50",
        "face_deg": FACE_DEG,
        "face_mm": [FACE_W, FACE_H],
        "marker_mm": MARKER,
        "base_mm": round(BASE_W, 2),
        "corners": tent_model(),
        "board": {"dictionary": "DICT_5X5_100", "squares": list(BOARD_SQUARES),
                  "square_mm": BOARD_SQUARE, "marker_mm": BOARD_MARKER},
    }
    with open(os.path.join(OUT_DIR, "marker_tent.json"), "w") as f:
        json.dump(model, f, indent=2)
        f.write("\n")
    print("wrote", pdf_path)


if __name__ == "__main__":
    main()
