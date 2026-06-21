# DexAssembly Cell — Closed-Loop Dexterous Micro-Assembly with a LEAP Hand

> Robothon 2026 · Faraday Future MuJoCo Robotics Hackathon
> **Registration UUID:** `a003f03d-ca73-4d5e-96bd-87a7a83465bf`

A fully **self-contained** MuJoCo cell in which a **16-DOF LEAP dexterous hand**,
mounted on a 4-axis Cartesian gantry, autonomously runs a **graded six-task
assembly arena** — colour-sorting, eye-in-hand inspection, a tactile button press,
**deformable-cable** force inspection, and peg-in-hole assembly — under a
**closed-loop, tactile-aware controller with grasp-failure recovery**, then
renders a **narrated** demo video, writes a labelled imitation-learning dataset,
and runs a **domain-randomized robustness study with an ablation**.

Everything is driven through **position actuators and real contact dynamics — the
code never teleports joints via `qpos`**, so parts fall under gravity, rest on the
bench, and grasps only hold when the fingers physically close (confirmed by the
fingertip touch sensors). Run **one command** (`python run_demo.py`) and the
project reproduces every artifact; no assets are downloaded at runtime — the LEAP
hand is committed and everything else is generated procedurally.

```
Deterministic arena:  100.0/100   (6/6 tasks)
  [PASS] part_red                placement_err 25 mm, hold_grip 25 N
  [PASS] part_green              placement_err 44 mm, hold_grip 23 N
  [PASS] inspect_sort_part_blue  raised to eye-in-hand cam, placement_err 29 mm
  [PASS] inspect_button          press_depth 19 mm
  [PASS] cable_inspect           cable deflection 89 mm
  [PASS] peg_insert              align_err 1.4 mm, peg seated

Domain-randomized grasp study (±12 mm / ±15%, 30 grasps/mode):
  adaptive (closed-loop)   80.0%      open-loop baseline   50.0%      gain +30.0 pp
```

---

## Robot platform

| | |
|---|---|
| **End-effector** | LEAP Hand — 16 actuated DOF, 4 fingers (index/middle/ring + opposable thumb), per-joint position servos and joint-position sensors (MuJoCo Menagerie, MIT). |
| **Arm** | Procedural 4-axis Cartesian gantry: prismatic X / Y / Z slides + a wrist-yaw hinge, each a position servo. Rock-solid, singularity-free reach over the bench. |
| **Sensing** | 4 fingertip **touch sensors**, a **6-axis wrist force/torque sensor**, 16 hand joint encoders, a button-displacement sensor, and **4 cameras** (hero / top / side / eye-in-hand wrist). |
| **Scene** | Instrumented bench with 3 colour bins, 3 sortable parts, a square connector peg + socket fixture, a spring-loaded inspection button, and a **deformable articulated cable** — all built in MJCF from primitives. |

Model size: `nq=55, nv=51, nu=20` actuators, `nsensor=23`, `ncam=4`, `ngeom=130`.

---

## Task goal

A tabletop **micro-assembly / kitting cell**. The hand must, end-to-end and
autonomously:

1. **Sort** the red and green parts into their colour-matched bins (pick → carry → place).
2. **Inspect-and-sort** the blue part: grasp it, **raise it to the eye-in-hand camera** and hold it for inspection, then place it in the blue bin.
3. **Press** a spring-loaded inspection button and confirm the press via its displacement sensor.
4. **Force-inspect** a deformable connector cable: deflect it elastically while monitoring the wrist force/torque sensor.
5. **Assemble** a square connector peg into its socket (peg-in-hole).

Each task is scored against measurable physical criteria (placement error, press
depth, cable deflection, socket alignment), and the run reports an overall
success rate on a 0–100 scale.

---

## Technical approach

### Procedural scene (`dexassembly/scene.py`)
The whole cell is generated through the **`mujoco.MjSpec`** API. The LEAP hand is
loaded from its MJCF and **attached** to the wrist body via `frame.attach_body`,
oriented (identity mount) so the palm's grasping face points down and the fingers
curl **down + inward** to cage an object beneath the palm. Gantry slides, the
bench, colour bins (tray + four walls), the peg/socket fixture, the spring-loaded
button (slide joint with stiffness), cameras, lights, and **fingertip touch
sensor sites** are all added programmatically. The builder also emits a static
[`scene.xml`](scene.xml) for inspection in `python -m mujoco.viewer`.

### Cartesian control + grasp primitives (`dexassembly/controllers.py`)
Because the gantry is a pure translation + yaw, the world position of the grasp
centre is an **affine function of the four gantry controls**, which the controller
calibrates once (from the *closed*-grasp fingertip convergence point) and inverts
analytically — so `set_gantry_target(x, y, z, yaw)` places the grasp point in the
world directly. Five symbolic **hand poses** (`open`, `pregrasp`, `grasp`,
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
(fixed nominal targeting, no recovery): **80% vs 50% success — a +30 pp
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

- 16-DOF dexterous hand on a 4-DOF gantry, attached via `MjSpec` — **20 actuators, 23 sensors, 4 cameras**.
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

- The hand mounts on a Cartesian gantry rather than a 6-DOF arm, so grasp approach is top-down only (chosen deliberately for reliability and singularity-free reach).
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
├── README.md · EVALUATION_GUIDE.md · ARCHITECTURE.md · LICENSE-THIRDPARTY.md
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
├── tests/                   # ~45 tests across scene/controllers/tasks/pipeline
└── assets/leap_hand/        # vendored LEAP Hand MJCF (MIT) — see LICENSE-THIRDPARTY.md
```

## Scoring rubric mapping

| Rubric criterion | Where it shows up |
|---|---|
| Runnability | One command; all assets committed; static `scene.xml`; `validate.py`; ~45 tests; deterministic. |
| MuJoCo depth | `MjSpec` build + hand attach; **deformable cable**; **6-axis wrist F/T** + 4 touch + button sensors (23 total); spring joint; elliptic friction; 20 actuators; 4 cameras; **fully actuated, no `qpos` teleport**. |
| Task design | Graded **6-task** assembly/inspection arena with real-world relevance and quantitative scoring. |
| Control | Closed-loop FSM (tactile + live-pose feedback), **failure recovery**, force-aware inspection, autonomy, teleoperation, data collection; **ablation** quantifies the gain. |
| Dexterity | 16-DOF multi-finger power grasp that physically lifts parts (~25 N tactile), opposable-thumb caging, single-finger button press, multi-finger deformable-cable manipulation. |
| Engineering quality | Typed, documented package; config dataclasses; CLI; `validate.py`; ARCHITECTURE + EVALUATION guides; ~45 tests; attribution. |
| Presentation | Narrated multi-camera demo (captions + `narration.srt`) with live tactile/force overlays and progress. |
| Innovation | Self-contained procedural dexterous cell that is simultaneously a benchmark, a demo, and a dataset generator. |

---

## 简介（中文）

本项目 **DexAssembly Cell** 是一个完全自包含的 MuJoCo 灵巧操作单元：一只 **16 自由度
LEAP 灵巧手**安装在四轴笛卡尔龙门架上，在**闭环、带触觉反馈**的控制器驱动下，自主完成一个
分级多任务装配竞技场——颜色分拣、手内翻转检视、触觉按钮按压、方形插销入孔装配。运行
`python run_demo.py` 即可一键复现：仿真、计分、带 HUD 的演示视频，以及一份带标签的
RGB-D 与状态/动作数据集。所有资源（含 LEAP 手）均已随提交一并包含，运行时无需任何下载。
五项任务成功率 **100%**。

---

*LEAP Hand model © its authors / Google DeepMind, MIT License — see
[`LICENSE-THIRDPARTY.md`](LICENSE-THIRDPARTY.md). All other content original to this submission.*
