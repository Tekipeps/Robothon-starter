# DexAssembly Cell — A Real-Physics Dexterous Assembly & QA Cell (LEAP Hand)

> Robothon 2026 · Faraday Future MuJoCo Robotics Hackathon
> **Registration UUID:** `a003f03d-ca73-4d5e-96bd-87a7a83465bf`

A **16-DOF LEAP dexterous hand** on a 4-axis gantry autonomously runs a **simulated
EV end-of-line assembly & QA station**: it sorts components into trays, raises a
part to its **eye-in-hand camera** for visual inspection, presses a **diagnostic
button**, **force-inspects a wiring-harness cable**, and uses a slender **probe tool** to actuate a recessed diagnostic micro-switch — six graded tasks, closed-loop, with tactile feedback and grasp
recovery.

### For the judges — the 60-second read

- **Honest physics is the headline.** The robot is driven **only** through position
  actuators and real contact dynamics — **the code never writes `qpos` to move
  anything**. So parts fall under gravity, rest on the bench, and a grasp holds
  *only* when the fingers physically close (confirmed by fingertip touch sensors and
  by the part actually rising). Many sim demos teleport joints; this one doesn't, and
  the demo video says so on-screen throughout.
- **It's closed-loop, and we prove it.** A domain-randomized **ablation** shows the
  adaptive controller (live-position perception + grasp recovery) beating an
  open-loop baseline **93% vs 67% — +27 pp**. That gap *is* the contribution.
- **Depth + breadth in one package.** 21 actuators, **24 sensors** (6-axis wrist
  F/T, 4 fingertip touch, joint + button), a **deformable articulated cable**, 4
  cameras, `nq=57` — and all four control modalities the rubric lists (scripted
  autonomy, closed-loop policy, teleoperation, data collection).
- **One command, no downloads.** `python run_demo.py` reproduces the video, a
  labelled RGB-D dataset, and the scorecard. The LEAP hand is vendored; the scene is
  generated procedurally; `validate.py` + 41 tests pass.

> The scenario is *themed* as an EV assembly/QA cell (the sponsor builds EVs) to make
> each task purposeful — the underlying physics and control are general-purpose.

> **Built transparently with a human in the loop.** See
> [`COLLABORATION.md`](COLLABORATION.md) for an honest log of how the human engineer
> and Claude divided the work and caught each other's mistakes (the catapulting
> grasp, the mis-aimed wrist camera, the brittle peg concept that became the tool-use station).

```
Deterministic arena:  100.0/100   (6/6 tasks)
  [PASS] part_red                placement_err 15 mm, hold_grip 29 N
  [PASS] part_green              placement_err 48 mm, hold_grip 23 N
  [PASS] inspect_sort_part_blue  raised to eye-in-hand cam, placement_err 48 mm
  [PASS] inspect_button          press_depth 19 mm
  [PASS] cable_inspect           cable deflection 87 mm, peak_wrist 362 N
  [PASS] tool_use                probe press 10.3 mm, tool lifted+re-docked, hold_grip 28 N

Domain-randomized grasp study (±12 mm / ±15%, 30 grasps/mode):
  adaptive (closed-loop)   93.3%      open-loop baseline   66.7%      gain +26.7 pp
```

---

## Robot platform

| | |
|---|---|
| **End-effector** | LEAP Hand — 16 actuated DOF, 4 fingers (index/middle/ring + opposable thumb), per-joint position servos and joint-position sensors (MuJoCo Menagerie, MIT). |
| **Arm** | Procedural **5-axis** Cartesian gantry: prismatic X / Y / Z slides + a **wrist-pitch hinge** + a wrist-yaw hinge, each a position servo. Pitch enables angled and side-approach grasps; singularity-free reach over the bench. |
| **Sensing** | 4 fingertip **touch sensors**, a **6-axis wrist force/torque sensor**, 16 hand joint encoders, a button-displacement sensor, and **4 cameras** (hero / top / side / eye-in-hand wrist). |
| **Scene** | Instrumented bench with 3 colour bins, 3 sortable parts, a spring-loaded inspection button, a recessed probe-actuated diagnostic switch, a graspable probe tool, and a **deformable articulated cable** — all built in MJCF from primitives. |

Model size: `nq=57, nv=53, nu=21` actuators, `nsensor=24`, `ncam=4`, `ngeom=143`.

---

## Task goal

A tabletop **EV end-of-line assembly & QA cell** (themed; the physics is
general-purpose). The hand must, end-to-end and autonomously:

1. **Sort** the red and green components into their colour-matched trays (pick → carry → place).
2. **Inspect-and-sort** the blue component: grasp it, **raise it to the eye-in-hand camera** and hold it steady for visual inspection, then place it in its tray.
3. **Functional-test** a spring-loaded diagnostic button and confirm the press via its displacement sensor.
4. **Force-inspect** a wiring-harness cable: deflect it elastically while monitoring the 6-axis wrist force/torque sensor.
5. **Use a tool**: grasp a slender probe from its holster and actuate a recessed diagnostic micro-switch that sits below the power-grasp cage's reliable fingertip reach.

Each task is scored against measurable physical criteria (placement error, press
depth, cable deflection, probe-switch travel), and the run reports an overall
success rate on a 0–100 scale.

| # | Task | Core challenge | Key capability exercised | Pass criterion |
|---|------|----------------|--------------------------|----------------|
| 1–2 | Sort red / green | Grasp a 48 mm cube, carry, release into a target bin | Power grasp + closed-loop position targeting + grasp recovery | placement_err < 80 mm |
| 3 | Inspect-sort blue | Lift to 270 mm eye-in-hand camera, hold steady, then bin | High-lift carry + held-offset compensation + eye-in-hand framing | raised to cam height + placement_err < 120 mm |
| 4 | Button functional test | Drive a single fingertip to a spring-loaded cap and confirm travel via displacement sensor | Single-finger pose + contact-triggered closed-loop | press_depth > 8 mm |
| 5 | Cable force inspect | Sweep a deformable cable while the wrist F/T sensor monitors the load | Deformable-body contact + force-aware manipulation | deflection > 40 mm |
| 6 | Tool use | Pick a slender probe, lower its tip into a 28 mm guarded well, actuate a recessed micro-switch | Tool-mediated manipulation: precision lives in geometry, not finger dexterity | probe_switch travel > 6 mm |

The tasks are ordered by complexity: tasks 1–2 isolate the basic grasp primitive; task 3 adds an inspection hold and a high-lift carry; task 4 switches to a single-finger pointing pose and a contact-triggered condition; task 5 introduces deformable-body dynamics and force sensing; task 6 chains two full pick-and-place sequences with a precision tool, requiring the robot to reason about a held object's reach envelope.

---

## Technical approach

### Procedural scene (`dexassembly/scene.py`)
The whole cell is generated through the **`mujoco.MjSpec`** API. The LEAP hand is
loaded from its MJCF and **attached** to the wrist body via `frame.attach_body`,
oriented (identity mount) so the palm's grasping face points down and the fingers
curl **down + inward** to cage an object beneath the palm. Gantry slides, the
bench, colour bins (tray + four walls), the probe-tool station, the spring-loaded
button (slide joint with stiffness), cameras, lights, and **fingertip touch
sensor sites** are all added programmatically. The builder also emits a static
[`scene.xml`](scene.xml) for inspection in `python -m mujoco.viewer`.

### Cartesian control + grasp primitives (`dexassembly/controllers.py`)
At pitch=0 the gantry is a pure translation + yaw, so the world position of the
grasp centre is an **affine function of the three translational controls**; the
controller calibrates the constant offset once (from the *closed*-grasp fingertip
convergence point) and inverts analytically — so `set_gantry_target(x, y, z, pitch, yaw)`
places the grasp point in the world directly, with `pitch` enabling angled approach. Five symbolic **hand poses** (`open`, `pregrasp`, `grasp`,
`pinch`, `point`) map onto the 16 LEAP actuators; the power grasp was tuned by an
automated sweep to lift a 48 mm part with a clean, stable grip.

### Closed-loop task FSM + recovery (`dexassembly/tasks.py`, `dexassembly/engine.py`)
Each task is a list of **motion primitives** (`Step`s) that command a gantry
target and/or hand pose and advance on a **condition** — a tactile/sensor trigger
(`button_pressed`) or a timed phase. Grasp targets are computed from the part's
**live position** at runtime, and the peg-placement aim is corrected by the
**measured held-object offset**. A step can carry a `verify` predicate; if it
fails (e.g. the part didn't rise after a grasp) the engine **recovers from the
last `checkpoint`** — re-sensing and re-grasping. The engine streams a per-step
callback used by the recorder and renderer, monitors peak metrics, and produces a
structured **scorecard**.

### Force-aware deformable inspection
A **deformable articulated cable** (six stiff-damped hinge segments) is swept by
the hand and bends elastically while the **6-axis wrist force/torque sensor**
monitors the load — genuine deformable-body dynamics plus force sensing.

### Robustness evaluation + ablation (`dexassembly/evaluate.py`)
The dexterous pick-and-place is run over many **domain-randomized** rollouts (part
positions ±12 mm, masses ±15%). An **ablation** compares the full closed-loop
system (live-position perception + grasp recovery) against an open-loop baseline
(fixed nominal targeting, no recovery): **93% vs 67% success — a +27 pp
gain**. Tuning this study is also how a real grasp-robustness bug was found and
fixed (over-stiff contacts were catapulting slightly off-centre parts; compliant
contacts resolved it) — the kind of issue only a randomized study surfaces.

### Data collection (`dexassembly/record.py`)
The same run is logged as a labelled dataset: `trajectory.jsonl` + `states.npz`
(time, `qpos`, `qvel`, commanded `ctrl`, fingertip forces, contact count, and the
active task/phase) plus periodic **RGB and depth** frames and a `dataset_meta.json`
schema — i.e. a ready-made imitation-learning corpus.

### Presentation (`dexassembly/hud.py`)
The demo overlays a live HUD (task/phase, **per-finger tactile bars**, grip force,
progress) and **narration captions**, cutting between the hero, side, and top
cameras; a matching `narration.srt` subtitle track is exported alongside the
video. (An **eye-in-hand wrist camera** is also defined in the model — the fourth
camera — for perception/data use.)

---

## Core features

- 16-DOF dexterous hand on a **5-DOF gantry** (x/y/z + pitch + yaw), attached via `MjSpec` — **21 actuators, 24 sensors, 4 cameras**.
- **Honest, fully-actuated physics** — control is *only* through `data.ctrl`; no `qpos` teleportation, so objects obey gravity and grasps hold through real contact.
- **Closed-loop, tactile-aware** control: contact-triggered button press, live-position grasp targeting, runtime held-offset correction, and **grasp-failure detection + recovery**.
- **Six-task graded arena** with quantitative per-task scoring and an aggregate 0–100 score.
- **Deformable articulated cable** + **6-axis wrist force/torque sensor** for force-aware inspection of a soft body.
- **Eye-in-hand inspection**: the part is raised to a wrist-mounted camera and held steady before sorting.
- **Domain-randomized evaluation** with an **adaptive-vs-open-loop ablation** that quantifies the value of perception feedback.
- **Narrated** demo video (on-screen captions + exported `narration.srt`) with live tactile/force overlays and multi-camera cuts.
- **Dual-purpose run**: a polished demo video *and* a labelled RGB-D + state/action dataset.
- **One-command reproducibility** (self-contained, no downloads) + keyboard **teleoperation** + a `validate.py` self-check.

## Highlights

- **Physics integrity.** Unlike scripts that move the robot by writing `qpos` (which makes objects float and arms penetrate surfaces), every motion here is produced by actuators and contact dynamics. Grasps succeed only when the fingers physically close — verified by the fingertip touch sensors and by the part actually rising.
- **All four control modalities** the rubric lists — scripted autonomy, closed-loop policy, teleoperation, and data collection — in one project, plus closed-loop **failure recovery**.
- **100% on the deterministic six-task arena** and a **domain-randomized robustness study** showing the closed-loop controller's advantage over an open-loop baseline.
- Small, honest engineering — affine gantry calibration, runtime held-offset correction, tactile-triggered recovery — turns a brittle open-loop script into a reliable system.

## Current limitations

- The hand mounts on a Cartesian gantry rather than a 6-DOF arm; the 5-DOF wrist (pitch + yaw) enables angled approach but not full-hemisphere reach (chosen deliberately for singularity-free, repeatable workspace over the bench).
- Inspection is a raise-and-hold to a wrist camera; true in-hand reorientation (finger-gaiting) is future work — a power grasp on a free cube was not stable enough under a large wrist rotation to ship honestly.
- Grasp poses are tuned for ~48 mm parts; very thin or very large objects need re-tuning.
- The autonomous policy is a tuned finite-state machine, not a learned policy — the recorded dataset is the intended bridge to learning one.

## Future improvements

- Swap the gantry for a Franka/UR arm and add full 6-DOF grasp approach.
- Train a visuomotor / tactile policy on the recorded RGB-D + state-action dataset (behaviour cloning, then RL fine-tuning).
- Finger-gaiting for continuous in-hand reorientation; force-servoed grasping that regulates grip to a target tactile force.
- Additional randomization axes (friction, lighting) for sim-to-real transfer.

---

## How to run

From the **repository root** (`Robothon-starter/`):

```bash
python -m pip install -r submissions/dexassembly-cell/requirements.txt
cd submissions/dexassembly-cell

# Full demo: scored arena -> HUD video + dataset + report.json (in outputs/)
python run_demo.py

# Fast smoke run (low-res, no images)
python run_demo.py --quick

# Domain-randomized robustness study + adaptive-vs-open-loop ablation
python -m dexassembly.evaluate --rollouts 20

# Validate the submission package (files, UUID, model structure)
python validate.py

# Regenerate the static scene and print a structural summary
python -m dexassembly.scene

# Inspect the cell interactively / teleoperate it (needs a display)
python -m mujoco.viewer        # then load scene.xml
python -m dexassembly.teleop   # keyboard control (see module docstring)

# Run the tests (no rendering)
python -m pytest tests -q
```

> **Runtime note.** The physics arena itself runs in ~3 s; the wall-clock time of
> `run_demo.py` is dominated by **offscreen rendering** of the ~1.2k video frames
> (seconds on a GPU, a few minutes with the CPU software renderer). Use
> `--quick` for a fast low-res smoke run, or `--no-data` to skip the RGB-D frames.
> `pytest tests` validates the full arena without any rendering.

Outputs land in `outputs/`:

| Artifact | Description |
|----------|-------------|
| `dexassembly_demo.mp4` | Annotated multi-shot demo video (≈1.5 min). |
| `report.json` | Per-task results, metrics, and the aggregate score. |
| `dataset/trajectory.jsonl`, `dataset/states.npz` | Labelled state/action trajectory. |
| `dataset/frames/rgb_*.png`, `depth_*.png` | Periodic RGB-D observations. |
| `narration.srt` | Subtitle / narration track for the demo video. |
| `evaluation.json` | Domain-randomized success rates + ablation (from `dexassembly.evaluate`). |

A copy of the rendered demo is committed at [`demo.mp4`](demo.mp4), and the scored
run at [`report.json`](report.json).

---

## Repository layout

```
dexassembly-cell/
├── run_demo.py              # one-command entry point (video + data + scorecard)
├── validate.py              # submission self-check (files, UUID, model structure)
├── scene.xml                # generated static scene (for the MuJoCo viewer)
├── registration.json        # contest UUID + project metadata
├── requirements.txt
├── README.md · EVALUATION_GUIDE.md · ARCHITECTURE.md · HARDWARE.md · COLLABORATION.md · LICENSE-THIRDPARTY.md
├── demo.mp4 · report.json   # committed demo video + scorecard
├── dexassembly/
│   ├── scene.py             # procedural MjSpec scene builder (+ LEAP attach)
│   ├── controllers.py       # Cartesian gantry control + hand-pose presets + F/T
│   ├── tasks.py             # closed-loop task FSM + arena + scoring + recovery
│   ├── engine.py            # execution engine, recovery, step callbacks, scorecard
│   ├── evaluate.py          # domain-randomized rollouts + adaptive/open-loop ablation
│   ├── record.py            # RGB-D + state/action dataset recorder
│   ├── hud.py               # demo-video HUD overlay + captions
│   └── teleop.py            # keyboard teleoperation
├── tests/                   # 41 tests across scene/controllers/tasks/pipeline
└── assets/leap_hand/        # vendored LEAP Hand MJCF (MIT) — see LICENSE-THIRDPARTY.md
```

## Scoring Rubric Alignment

This maps the submission to the eight judging criteria and tells you **exactly where
to verify each claim** — nothing here is asserted without code/artifact you can check.

**1. Runnability.** One command (`python run_demo.py`) reproduces every artifact;
**all assets are committed** (LEAP hand vendored under `assets/leap_hand/`), the scene
is generated procedurally (no runtime downloads), and the run is deterministic.
*Verify:* [`run_demo.py`](run_demo.py), [`validate.py`](validate.py) (38 checks),
`python -m pytest tests` (41 tests), [`requirements.txt`](requirements.txt).

**2. Depth of MuJoCo use.** `MjSpec` programmatic build with the hand **attached**
via `frame.attach_body`; a **deformable articulated cable** (6 hinge joints); a
**6-axis wrist force/torque** sensor + **4 fingertip touch** sensors + a button
displacement sensor (**24 sensors**); a spring-loaded **slide joint**; **elliptic
friction cones** (`impratio`); **21 position-servo actuators** (5-DOF gantry + 16
LEAP); **4 cameras**; `nq=57`. Control is **only** via `data.ctrl` — **never `qpos`
teleportation**. *Verify:* [`dexassembly/scene.py`](dexassembly/scene.py), [`scene.xml`](scene.xml).

**3. Task design.** A graded **six-task** EV assembly/QA arena with a deliberate
difficulty ladder: 2× colour-sort (baseline grasp), an eye-in-hand inspect-and-sort
(high-lift hold + camera framing), a tactile button test (single-finger + contact
trigger), a deformable-cable force inspection (soft-body dynamics + F/T sensing), and
a probe tool-use task (tool-mediated reach into a guarded well). Each task is scored
on **physical state** (placement error, press depth, cable deflection, probe-switch
travel) and the sequencing is intentional — each task exercises a capability the
previous ones do not. *Verify:* [`dexassembly/tasks.py`](dexassembly/tasks.py),
[`report.json`](report.json).

**4. Control.** Closed-loop FSM with **live-position perception**,
**tactile-triggered** grasping, **force-aware** inspection, and **grasp-failure
detection + recovery**; plus **keyboard teleoperation** and a **data-collection**
pipeline — all four modalities the rubric lists. An **ablation** quantifies the
closed-loop gain (+27 pp). *Verify:* [`dexassembly/engine.py`](dexassembly/engine.py),
[`dexassembly/teleop.py`](dexassembly/teleop.py), `evaluation.json`.

**5. Dexterous manipulation.** A 16-DOF, four-finger **power grasp** that physically
closes and **lifts** parts with **measured tactile force (~25 N)**; opposable-thumb
caging; single-finger button press; multi-finger deformable-cable manipulation.
*Verify:* [`dexassembly/controllers.py`](dexassembly/controllers.py), the demo video.

**6. Engineering quality.** Typed, documented Python package; config dataclasses;
CLI; `validate.py`; ARCHITECTURE + EVALUATION + **HARDWARE (sim-to-real)** guides; 41
tests; third-party attribution. *Verify:* [`ARCHITECTURE.md`](ARCHITECTURE.md),
[`HARDWARE.md`](HARDWARE.md), [`LICENSE-THIRDPARTY.md`](LICENSE-THIRDPARTY.md), `tests/`.

**7. Presentation.** A narrated, multi-camera demo with a title card, a closing
scorecard, a persistent **"real physics — no qpos teleport"** badge, live
tactile/force overlays, and a live **eye-in-hand** picture-in-picture; an `.srt`
subtitle track is exported. *Verify:* [`demo.mp4`](demo.mp4), `narration.srt`.

**8. Innovation.** Three intersecting claims, each verifiable:
(a) **Honest actuation as a design constraint, not a post-hoc claim** — the entire
architecture (affine Cartesian calibration, tactile-triggered conditions, held-offset
correction) was designed around the fact that `qpos` teleportation is forbidden;
the ablation measures the concrete value this adds (+27 pp over open-loop).
(b) **Tool-mediated fine manipulation on a power-grasp rig** — rather than
over-engineering finger dexterity, the probe task solves a precision reach problem
through *tool geometry*, the same strategy humans use and a pattern directly
applicable to real assembly lines.
(c) **A deployment path, not just a demo** — the 5-DOF gantry (pitch + yaw wrist)
is mapped to real off-the-shelf hardware in [`HARDWARE.md`](HARDWARE.md); the
labelled RGB-D + state/action dataset is a ready-made imitation-learning corpus; and
the human-AI workflow in [`COLLABORATION.md`](COLLABORATION.md) documents an
engineering methodology, not just a credit disclaimer.

---

## 简介（中文）

本项目 **DexAssembly Cell** 是一个完全自包含的 MuJoCo 灵巧操作单元：一只 **16 自由度
LEAP 灵巧手**安装在四轴笛卡尔龙门架上，在**闭环、带触觉反馈**的控制器驱动下，自主完成一个
以**电动车产线总装与质检**为主题的分级多任务竞技场——零件分拣、手眼相机检视、诊断按钮按压、
线束力觉检测、以及用细长探针工具触动隐藏式诊断微动开关。机器人**全程仅通过执行器（`data.ctrl`）驱动，绝不通过 `qpos`
传送关节**，因此物体遵循重力、抓取必须靠手指真实闭合才能成立。运行 `python run_demo.py`
即可一键复现：仿真、计分、带 HUD 的演示视频，以及一份带标签的 RGB-D 与状态/动作数据集。
所有资源（含 LEAP 手）均已随提交一并包含，运行时无需任何下载。六项任务成功率 **100%**。

---

*LEAP Hand model © its authors / Google DeepMind, MIT License — see
[`LICENSE-THIRDPARTY.md`](LICENSE-THIRDPARTY.md). All other content original to this submission.*
