# Architecture

Sentinel is a small, layered control stack over a vendored Unitree Go1. Each layer is
one focused module; data flows **command down, state up**.

```
                    ┌──────────────────────────────────────────────┐
 run_demo.py  ──►   │  Mission (engine.py)   state machine + score  │
 (render/HUD)       │     PATROL → INSPECT → (SHOVE) → finish        │
                    └───────┬───────────────────────────┬───────────┘
                            │ body pose                  │ forward, yaw
                            ▼                            ▼
                  WaypointFollower (controller.py)   TrotGait (gait.py)
                  heading P-control → (fwd, yaw)      (fwd, yaw) → 12 ctrl
                            ▲                            │
                            │ trunk xpos / yaw           ▼  joint targets
                    ┌───────┴────────────────────────────────────────┐
                    │  MuJoCo model  (scene.py: Go1 + course)         │
                    └────────────────────────────────────────────────┘
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

The gait is open-loop in the legs — no per-foot force feedback — so stability comes
purely from the support pattern and ride height.

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
- **SHOVE** — when the trunk passes `push_at_x`, apply a 70 N·0.15 s lateral force via
  `data.xfrc_applied` on the trunk; the gait must absorb it.

It reads the trunk pose back from physics each tick, scores seven objectives against
the live state, and (for the demo) streams a `StepInfo` with the live objective flags,
phase, drive commands, and pose so the HUD can render the checklist and mini-map.

## 5. Presentation — `sentinel/hud.py` + `run_demo.py`

Pure-PIL overlays: a title card, a per-frame HUD (objective checklist, state + drive
bars, a route mini-map with the live robot dot, a progress bar, a caption), an onboard
**head-cam picture-in-picture** shown during each inspection, and a final scorecard.
`run_demo.py` renders the chase camera, composites the HUD, and writes the video plus
`report.json`.

## Design choices

- **Position servos + scripted trot** over a learned policy: deterministic, dependency-
  free, and identical for every judge — reproducibility over raw capability.
- **Closed loop on the body, open loop on the legs:** the minimal feedback that makes
  the patrol reliable without pretending to per-foot force control the rig can't do.
- **Everything measured:** thresholds, the shove envelope, and gait parameters come
  from printed experiments, not tuning by assertion.
