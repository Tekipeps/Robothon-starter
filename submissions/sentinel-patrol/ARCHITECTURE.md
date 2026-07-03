# Architecture

Sentinel is a small, layered control stack over a vendored Unitree Go1. Each layer is
one focused module; data flows **command down, state up**.

```
                    ┌──────────────────────────────────────────────┐
 run_demo.py  ──►   │  Mission (engine.py)   state machine + score  │
 (render/HUD/       │     PATROL → INSPECT → (SHOVE) → finish        │
  telemetry log)    └───┬───────────────┬───────────────┬───────────┘
                        │ body pose     │ IMU accel     │ forward, yaw, brace
                        ▼               ▼               ▼
              WaypointFollower   DisturbanceReflex   TrotGait (gait.py)
              (controller.py)    (proprio.py)        (fwd, yaw, brace)
              heading P-ctrl     8 ms crouch brace     → 12 ctrl
                        ▲               ▲               │
                        │ trunk pose    │ sensordata    ▼  joint targets
                    ┌───┴───────────────┴────────────────────────────┐
                    │  MuJoCo model  (scene.py: Go1 + course +        │
                    │  IMU gyro/accel/velocimeter + 4 foot touch)     │
                    └─────────────────────────────────────────────────┘
```

## 1. Scene — `sentinel/scene.py`

`compile_scene()` loads the vendored `go1.xml` through `mujoco.MjSpec` and grows the
course programmatically: a checker floor, a **ramp berm** (a rotated up-slab, flat
top, rotated down-slab; ~9° ramps, 5.5 cm high), a **rubble field** of 14 seeded low
blocks (1.2–3 cm), two **inspection stations** (post + target panel + a scan-target
site), a finish pad, two lights, and three cameras (`head` on the trunk, `chase`
tracking the trunk, `overview`). A `CourseConfig` dataclass holds the waypoints, the
waypoint→panel inspection map, and the shove parameters, so the whole course is one
declarative object.

It also attaches the **onboard sensor suite** to the Go1's own sites: an IMU triad
(`gyro` + `accelerometer` + `velocimeter` on the trunk's `imu` site) and a **touch
sensor on each foot** — seven real MuJoCo sensors whose `sensordata` is the only
thing the reflex and the HUD's contact dots ever read.

The home keyframe from `go1.xml` survives the merge and is used to reset the robot to
a stable stance at the origin.

## 2. Gait — `sentinel/gait.py`

The Go1's 12 joints are position servos. `TrotGait.step(forward, yaw)` produces their
targets each tick:

- **Phasing.** Legs split into diagonal pairs `{FR, RL}` and `{FL, RR}` a half-cycle
  apart (trot), so two diagonal feet always support the body.
- **Foot trajectory.** Each foot follows a planar path in its leg's sagittal plane:
  *stance* sweeps the planted foot backward (propelling the body); *swing* lifts it
  (`sin` arc) and returns it to the front.
- **Inverse kinematics.** Planar 2-link IK (thigh `L1` = calf `L2` = 0.213 m) maps a
  Cartesian foot target `(fx, fz)` to `(thigh, calf)` angles; the hip-abduction joint
  stays at 0 for an upright trunk. The IK is validated against the model's home pose
  (`IK(home foot) == (0.9, −1.8)`).
- **Skid-steer.** Per-leg stride = `stride · (forward + turn_gain·yaw·side)`. Equal for
  all legs when going straight; the turn term lengthens one side and shortens the
  other. With `forward = 0` the sides stride oppositely → **pivot in place**.

The gait is open-loop in the legs — no per-foot force feedback — so steady-state
stability comes from the support pattern and ride height. A `brace` input (0–1)
lowers the ride height by up to 5 cm: the disturbance reflex's crouch.

## 2b. Proprioception & reflex — `sentinel/proprio.py`

`Proprioception` resolves the seven named sensors once and reads them from
`data.sensordata` each tick. `DisturbanceReflex` watches the **body-frame lateral
acceleration** through a 25 ms low-pass: the trot's own foot-impact spikes are
milliseconds long and filter away (steady walking stays under ~3.4 m/s² filtered),
while an external shove is *sustained* and crosses the 3.6 m/s² trigger within
~8 ms. Triggering latches a crouch envelope (30 ms attack, 0.7 s hold, 0.35 s
release) that drops the centre of mass while the legs **keep stepping** — the step
pattern is what catches the body.

Two details were measured, not assumed:

- **Crouch-only beats crouch+wide-stance.** Abducting the hips mid-push breaks the
  planar-IK stance and topples the robot; the pure crouch survives 90–110 N where
  the passive gait falls at 80 N.
- **Reafference gating.** A commanded pivot-in-place produces sustained lateral
  accelerations that look exactly like a shove, so triggering is suppressed while
  the controller itself commands an aggressive turn (`INSPECT` or |yaw| > 0.5) — a
  robot must not startle at its own motion. Over the rubble the reflex *does* fire
  on real terrain jolts, usefully hunkering the trunk down over the roughest ground.

## 3. Navigation — `sentinel/controller.py`

`WaypointFollower.command(pos, yaw)` returns `(forward, yaw)` toward the active
waypoint: proportional control on heading error, with forward speed eased by
`cos(error)` (clamped to a crawl) so the trunk stays level while turning. Waypoints
advance when the trunk enters their acceptance radius. This closes the loop the gait
lacks — without it the trot slowly drifts off course and falls at the first obstacle.

## 4. Mission — `sentinel/engine.py`

A state machine over the primitives:

- **PATROL** — follow waypoints. On arriving at a station with a panel, enter INSPECT.
- **INSPECT** — command `forward = 0` and yaw toward the panel until the heading error
  is < 0.18 rad, then hold a 1.1 s scan; the dwell restarts if the robot loses the
  panel, so a scan only counts when it's *steadily* facing the target.
- **SHOVE** — when the trunk passes `push_at_x`, apply a 100 N·0.15 s lateral force via
  `data.xfrc_applied` on the trunk. The reflex feels it through the IMU (it is never
  told the shove is coming) and the braced gait absorbs it; the report logs the
  measured detection latency.

It reads the trunk pose back from physics each tick, scores seven objectives against
the live state, and (for the demo) streams a `StepInfo` with the live objective flags,
phase, drive commands, and pose so the HUD can render the checklist and mini-map.

## 5. Presentation — `sentinel/hud.py` + `run_demo.py`

Pure-PIL overlays: a title card, a per-frame HUD (objective checklist, state + drive
bars, **live foot-contact dots fed by the touch sensors**, an amber **BRACE chip**
while the reflex is active, a route mini-map with the live robot dot, a progress bar,
a caption), an onboard **head-cam picture-in-picture** shown during each inspection,
and a final scorecard. `run_demo.py` renders the chase camera, composites the HUD,
and writes the video plus `report.json`.

## 6. Data collection — `sentinel/datalog.py`

Every physics step is logged — `qpos`/`qvel`, `ctrl`, the full sensor stream, the
mission phase, and the high-level command (`forward`, `yaw`, `brace`) — and written
as a self-describing compressed dataset (`outputs/telemetry.npz`, 500 Hz) with a
JSON summary, plus a labelled head-cam snapshot at the moment each panel scan
completes. Each run is therefore a state-action-observation dataset ready for
imitation-learning or gait-analysis work.

## Design choices

- **Position servos + scripted trot** over a learned policy: deterministic, dependency-
  free, and identical for every judge — reproducibility over raw capability.
- **Closed loop where it earns its keep:** navigation on the body pose, the brace
  reflex on the IMU — while the legs stay open-loop rather than pretending to
  per-foot force control the rig can't do.
- **Everything measured:** thresholds, the shove envelope, and gait parameters come
  from printed experiments, not tuning by assertion.
