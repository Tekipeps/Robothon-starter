# 🐕‍🦺 Sentinel — Autonomous Quadruped Inspection Patrol

**A Unitree Go1 walks an instrumented patrol route entirely on real foot-ground
contact** — crests a ramp berm, traverses a rubble field, pivots in place to scan
two inspection panels with its onboard head camera, shrugs off a 70 N lateral shove,
and reaches the finish pad. Every motion is produced by a hand-built trot gait
through `data.ctrl`; nothing is teleported.

```
Patrol scorecard:  100/100   (7/7 objectives)
  [✓] crest_berm        crossed the ramp berm upright
  [✓] cross_rubble      traversed the uneven low-block field
  [✓] inspect_panel_A   pivoted to face panel A and scanned (head cam)
  [✓] inspect_panel_B   pivoted to face panel B and scanned (head cam)
  [✓] push_recovery     survived a 70 N lateral shove, stayed upright
  [✓] reach_finish      reached the finish pad
  [✓] stayed_upright    never fell across the whole ~48 s mission
```

> One command — `python run_demo.py` — reproduces the narrated HUD video and
> `report.json`. No downloads: the Go1 model is vendored under `assets/`.

---

## Why this is interesting

Most dexterous-manipulation entries live or die on grasp reliability. Sentinel takes
the **locomotion** lane: a legged robot is only useful if it can *get somewhere
rough and stay upright*, so the whole submission is built around **honest, contact-
driven walking** and a closed control loop on top of it.

- **A trot gait from scratch.** The 12 leg actuators are driven by a foot-trajectory
  trot with planar 2-link inverse kinematics — diagonal leg pairs swing in anti-phase
  so the robot is always supported. It walks because the feet really push the ground.
- **Closed-loop autonomy.** A waypoint follower reads the trunk pose every tick and
  steers (proportional heading control) so the patrol holds its line over the berm and
  rubble — the open-loop gait alone drifts off course and falls.
- **A real mission, scored.** A small state machine strings primitives into a patrol:
  *trot → pivot-and-scan at each station → recover from a shove → finish*, with seven
  measurable objectives turned into a 0–100 score and an on-screen checklist.

---

## Run it

```bash
python -m pip install -r requirements.txt   # mujoco, numpy, imageio[ffmpeg], pillow
cd submissions/sentinel-patrol

python run_demo.py            # full HUD demo video (outputs/) + report.json
python run_demo.py --quick    # fast low-res smoke run
python validate.py            # self-check: files, UUID, model structure
python -m sentinel.scene      # compile the course + print a structural summary
```

A copy of the rendered demo is committed at [`demo.mp4`](demo.mp4); the scored result
is in [`report.json`](report.json).

---

## The mission & scoring

The robot starts at the origin facing +x and patrols five waypoints across the
course. Each objective is checked against the **live physics state** (trunk pose,
height, contact-driven progress) — never a scripted "success" flag.

| # | Objective | Pass criterion |
|---|---|---|
| 1 | `crest_berm` | trunk passes the berm crest (x > 2.3 m) while upright |
| 2 | `cross_rubble` | trunk clears the rubble field (x > 3.7 m) while upright |
| 3 | `inspect_panel_A` | pivots to face panel A (heading error < 0.18 rad) and holds a 1.1 s scan |
| 4 | `inspect_panel_B` | same, for panel B on the far side of the route |
| 5 | `push_recovery` | survives a 70 N · 0.15 s lateral shove without falling |
| 6 | `reach_finish` | enters the finish-pad radius |
| 7 | `stayed_upright` | trunk height never drops below 0.16 m across the run |

Score = `100 × passed / 7`.

---

## How it works

```
run_demo.py ── renders chase-cam video + HUD + head-cam PiP + scorecard
   │
   └── sentinel.engine.Mission ── state machine: PATROL · INSPECT · SHOVE
         ├── sentinel.controller.WaypointFollower  (heading → forward/yaw command)
         ├── sentinel.gait.TrotGait                (forward/yaw → 12 joint targets)
         └── sentinel.scene.compile_scene          (Go1 + berm + rubble + panels + cams)
```

- **`gait.py`** — the trot. Each foot follows a planar stance/swing trajectory; planar
  2-link IK (thigh & calf both 0.213 m) maps the Cartesian foot target to joint angles.
  A per-side stride term skid-steers the body, and with `forward = 0` the two sides
  stride oppositely so the robot **pivots in place** to face a panel.
- **`controller.py`** — proportional heading control toward the active waypoint, easing
  forward speed while turning hard so the trunk stays level.
- **`engine.py`** — the mission state machine, the scripted shove (`xfrc` on the trunk),
  objective bookkeeping, and scoring. Reads the trunk pose back from physics; the only
  things it ever *writes* are the gait's `ctrl` and the brief external shove force.
- **`scene.py`** — procedurally grows the course around the vendored Go1 via `MjSpec`:
  textured floor, a ramp-up/flat/ramp-down berm, a seeded rubble field, two inspection
  posts with scan-target sites, a finish pad, and onboard/chase/overview cameras.

---

## Robot platform

The **Unitree Go1** quadruped (MuJoCo Menagerie, BSD-3) is vendored under
[`assets/unitree_go1/`](assets/unitree_go1) — no runtime download. It has a free-joint
trunk and 12 position-servo leg actuators (per leg: hip abduction, thigh, calf). The
only change to the vendored model is an added forward-facing **head camera** on the
trunk for the inspection picture-in-picture. See
[`LICENSE-THIRDPARTY.md`](LICENSE-THIRDPARTY.md).

---

## Honest physics

- Control is **only** through `data.ctrl` (the 12 joint targets) plus one brief
  external `xfrc` shove. The trunk free joint is **never** written — the robot walks,
  climbs, and recovers purely through actuated joints and foot-ground contact.
- The gait is **open-loop in the legs** (no cheating per-foot force feedback); stability
  comes from the diagonal support pattern, and *navigation* is closed-loop on the body
  pose. The shove is a genuine external force the controller does not anticipate.
- The run is **deterministic** (seeded), so `report.json` reproduces exactly.

---

## How it maps to the rubric

- **Reproducibility** — one command, vendored model, deterministic, `validate.py` self-check.
- **MuJoCo depth** — free-joint floating base, 12 actuated joints, frictional foot
  contacts over a ramp and uneven blocks, an external disturbance force, multi-camera rig.
- **Task design** — a coherent real-world mission (autonomous site-inspection patrol)
  with seven measurable, physics-checked objectives.
- **Control** — hand-built trot gait + planar IK + closed-loop waypoint navigation + a
  mission state machine.
- **Dexterity** — *coordination* lane: four legs in a phased gait, pivot-in-place steering.
- **Engineering quality** — small, documented modules (`scene / gait / controller /
  engine / hud`), config dataclasses, a validator.
- **Presentation** — narrated HUD: live objective checklist, route mini-map, drive/turn
  bars, and an onboard head-cam inset during each scan.
- **Innovation** — a legged inspection-patrol with turn-in-place scanning and disturbance
  recovery, distinct from the manipulation-heavy field.

---

## File map

| Path | Role |
|---|---|
| `run_demo.py` | one-command entry point (video + scorecard + report.json) |
| `sentinel/scene.py` | procedural course builder (Go1 + terrain + panels + cameras) |
| `sentinel/gait.py` | trot-gait controller + planar leg IK |
| `sentinel/controller.py` | closed-loop waypoint heading controller |
| `sentinel/engine.py` | mission state machine + scoring |
| `sentinel/hud.py` | PIL video overlays (HUD, mini-map, PiP, scorecard) |
| `validate.py` | submission self-check |
| `assets/unitree_go1/` | vendored Go1 model (BSD-3) |
| `demo.mp4` · `report.json` | committed demo video + scorecard |

---

## Limitations

- The gait is a fixed-parameter trot tuned for gentle terrain (ramps, low rubble), not
  stairs or large gaps — it has no learned or reactive balance controller, so very steep
  or tall obstacles would topple it. The shove magnitude is tuned to the gait's passive
  recovery envelope (≈ 70 N); a harder hit falls.
- Steering is skid-steer heading control, not full-body MPC; it holds a line well but
  won't do precise dynamic maneuvers.

These are honest boundaries of a clean, scripted, *reproducible* controller — chosen
over a heavier learned policy so the demo runs identically for every judge.

---

## Human + AI collaboration

This entry was built by a human directing an AI coding agent — the locomotion was
de-risked first (stand → trot → steer → terrain) before any polish. The decision log,
including the dead-ends and why locomotion was chosen over manipulation, is in
[`COLLABORATION.md`](COLLABORATION.md).

*Built for Robothon 2026 · Faraday Future MuJoCo Robotics Hackathon.*
