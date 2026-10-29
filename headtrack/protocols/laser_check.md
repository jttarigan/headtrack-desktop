# Laser check: angle scale of the reference (and of every tracker)

Decision 11 (Jos, 2026-10-13): a laser pointer on the head, aimed at targets
drawn on the G34w-30 ultrawide. Why: the only published check of ARKit face
tracking (Parker et al. 2022) found head angles off by 15 to 23 %, so the
reference has to be checked against known angles. The laser makes the head,
not the eyes, do the aiming: looking at a mark lets the eyes take part of the
turn, which would look like a tracker error. All four trackers record during
the check, so each one gets a static accuracy figure against true angles.

Status 2026-10-13: procedure, protocol, target screen and analysis ready and
tested on synthetic data. Waiting for Jos's laser and measurements; not run.

## What to get

1. **Laser pointer, Class 2 (under 1 mW)**, pen type, red or green. Many cheap
   presentation pointers are 5 mW (Class 3R); avoid those near eyes and
   screens. If the button must be held down, a rubber band or tape keeps it
   on. Spare batteries.
2. **Elastic sports headband** (or a snug cap) to carry the laser.
3. **Tape**: electrical or masking tape, for the laser on the headband and for
   floor marks.
4. **Tape measures**: a rigid one, and a flexible sewing tape (it can follow
   the curved screen).
5. **A straight edge longer than the screen** (about 90 cm): a long ruler or a
   straight wooden strip.
6. A chair that does not swivel or roll, or one that can be locked.

## Where the laser goes

- On the headband, on the **forehead midline**, 2 to 4 cm above the
  eyebrows, lying flat and **pointing straight forward**.
- Its tip should stick out a little past the headband so hair and band do not
  block the beam. Fix it with two wraps of tape (or a zip tie); tape the
  button on.
- The AirPods Max headband must sit behind it and not press on it.
- Alignment check: head upright, look straight at the middle of the screen.
  The spot should land within about 3 cm of where you look. If not, shim the
  laser with folded tape or paper. It does not have to be perfect: every
  angle is measured from the centre reading.

## Where the tape goes

- Nothing on the monitor: the agent draws the targets on screen.
- **Floor**: mark where the chair's feet (or wheels) stand, so every session
  sits in the same place. Note the seat height.
- **Headband**: holds the laser.

## Seating

- Sit in the normal recording position: Mac camera in front, iPhone
  reference next to it, centred on the ultrawide, back against the backrest.
- Head upright and level, the laser spot should fall near the **vertical
  middle of the screen**. Adjust the seat height (or the monitor) until it
  does. The screen is only 33 cm tall, so up and down targets need room.

## What to measure (send the numbers to the agent)

All to about ±1 cm, sitting in the recording position, head upright.

1. **a**: from the laser's front tip to its spot on the screen, along the beam
   (hold the tape at the tip, run it to the spot).
2. **b**: from the ear canal opening (the small flap in front of it) forward
   to the laser's front tip, horizontally. A mirror helps.
   The head turns roughly around a point between the ears, so the distance
   that matters is **D = a + b**; the agent computes it.
3. **h**: how far the laser sits above the ear canal, vertically (used for
   the up and down targets; ±1 cm is fine).
4. **Screen curve**: lay the straight edge across the front of the screen,
   touching the left and right edges at mid-height, and measure the gap
   between the straight edge and the screen surface at the centre (mm).
5. **Picture size**: with the flexible tape following the curve, the width of
   the lit picture from its left edge to its right edge, and its height
   (mm). macOS says 802 × 330 mm; this checks it.

With D of about 65 cm and the 1500R curve, the screen covers about ±33° left
and right and ±14° up and down.

## Targets and protocol

C (centre), L10, L20, L30, R10, R20, R30 (left and right, degrees), U10, D10
(up and down). Each three times in a fixed shuffled order (seed 20261013),
5 s each, with 10 s at C before and 8 s after: 2.6 minutes
(`laser_check.json`). The current target is highlighted on screen and spoken
through the AirPods.

For each prompt: **turn the head, not the body**; keep the shoulders still;
put the spot on the mark and hold it until the next prompt. 1 cm off at 65 cm
is about 0.9°.

Safety: never at anyone's eyes; watch for reflections from glass.

## Making the targets (agent, once per setup)

```
headtrack/.venv/bin/python headtrack/analysis/laser_targets.py --a-mm A --b-mm B --h-mm H \
    --sagitta-mm GAP --screen-mm W H
```

writes `laser_check_targets.json` (geometry, and each mark's pixel position
and true angle) and prints each mark's distance from the centre along the
screen, which a tape can confirm. To look at the screen without recording:
`build/release/HeadTrackRecorder.app/Contents/MacOS/HeadTrackRecorder
--targets headtrack/protocols/laser_check_targets.json --render-targets
/tmp/targets.png`. New measurements mean a new targets file; the analysis
reads the geometry each session stored in its `session.json`.

## Each session (about 10 minutes, attended)

1. Chair on its floor marks, AirPods Max on (Bluetooth), headband with laser,
   iPhone reference running. Normal lighting.
2. `headtrack/run_session.sh laser_check --speak on`. It refuses to start
   without the targets file. The recorder covers the ultrawide with the marks
   (current one highlighted) and moves its own window to the MacBook screen.
   Space starts, Esc stops.
3. Afterwards: `headtrack/.venv/bin/python headtrack/analysis/analyze.py
   runs/<session>`; the laser results are under `laser_check`.

## Analysis (`htlib.laser_check`)

- Holds: the last 3 s of each step (step events carry the mark, e.g.
  `[R20]`). A hold where a tracker moved more than 1° is dropped for it.
- Angle of each hold: the rotation from the centre pose, with the centre
  interpolated in time between the centre holds around it, so slow drift
  (AirPods yaw) drops out. A rotation angle needs no frame alignment.
- True angle of each mark from D, h, the screen curve and its pixel position.
- Per tracker, for yaw left, yaw right and pitch: **gain** (slope through
  zero; 1.0 is perfect), mean error, MAE, RMS, max error; per mark the mean
  and spread over the three rounds; the spread of the centre holds.
- The reference's gain feeds the paper's limits and, if far from 1, a
  correction before the agreement analysis.

## Error budget (estimates, to confirm from the data)

- Aiming: about 0.5 to 1° (spot held within 0.5 to 1 cm).
- D to ±1 cm at 65 cm: ±1.5 % of each angle.
- Head shifting while turning: up to 1 to 2° at 30°, closer to the screen
  than a wall would be. ARKit's head-position estimate can correct it later.
- Roll is not checked; a laser spot cannot show it. Audio cares far less
  about roll than about yaw.
