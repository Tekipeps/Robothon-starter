"""Closed-loop task layer for the DexAssembly Cell.

Each task is expressed as a list of :class:`Step` motion primitives.  A step
commands a gantry target and/or a hand pose and then *waits on a condition* —
reaching a pose, exceeding a tactile-force threshold, or a timeout — before the
runner advances.  This makes the controller genuinely closed-loop: the power
grasp closes until the fingertip touch sensors report contact, and lifts only
confirm once the part clears a height.  The :class:`Arena` strings the tasks
together and scores them against measurable criteria.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

import mujoco
import numpy as np

from .controllers import CellController, GantryTarget
from .scene import CABLE_ANCHOR, CABLE_SEG_LEN, CABLE_SEGMENTS, CABLE_TIP_BODY, SceneConfig

Condition = Callable[["TaskContext"], bool]


@dataclass
class TaskContext:
    model: mujoco.MjModel
    data: mujoco.MjData
    ctl: CellController
    cfg: SceneConfig

    def body_pos(self, name: str) -> np.ndarray:
        return self.data.xpos[mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, name)].copy()

    def sensor(self, name: str) -> float:
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        return float(self.data.sensordata[self.model.sensor_adr[sid]])


@dataclass
class Step:
    name: str
    target: Optional[GantryTarget] = None
    hand: Optional[str] = None
    until: Optional[Condition] = None
    timeout: float = 3.0
    settle: float = 0.0  # extra hold time after the condition is met
    # Dynamic target computed at step entry from the live sim state + task store
    # (used so grasps track the part's actual position, not a static guess).
    target_fn: Optional[Callable[["TaskContext", dict], GantryTarget]] = None
    # Closed-loop recovery: if ``verify`` is False after this step, the engine
    # jumps back to the most recent ``checkpoint`` step and retries.
    verify: Optional[Condition] = None
    checkpoint: bool = False


@dataclass
class Task:
    name: str
    kind: str
    steps: list[Step]
    score: Callable[["TaskContext", dict], "TaskResult"]
    monitor: Optional[Callable[["TaskContext", dict], None]] = None


@dataclass
class TaskResult:
    name: str
    kind: str
    success: bool
    detail: dict = field(default_factory=dict)


# ----- reusable conditions -----

def reached(tol: float = 0.012) -> Condition:
    def cond(c: TaskContext) -> bool:
        # gantry actuators are position servos; compare commanded vs achieved
        d, ctl = c.data, c.ctl
        for a in ctl.gantry_act:
            jid = c.model.actuator_trnid[a, 0]
            adr = c.model.jnt_qposadr[jid]
            if abs(d.qpos[adr] - d.ctrl[a]) > tol:
                return False
        return True
    return cond


def grip_above(force: float) -> Condition:
    return lambda c: c.ctl.total_grip_force() > force


def part_above(part: str, z: float) -> Condition:
    return lambda c: c.body_pos(part)[2] > z


def held(part: str, z: float = 0.10, force: float = 3.0) -> Condition:
    """A confident grasp: the part is lifted clear AND the fingers report force."""
    return lambda c: c.body_pos(part)[2] > z and c.ctl.total_grip_force() > force


def button_pressed(depth: float = 0.008) -> Condition:
    return lambda c: c.sensor("button_press") < -depth


def _sample_hold_grip(c: "TaskContext", store: dict, part: str, z_thresh: float = 0.12) -> None:
    """Record the total grip force while the part is held aloft (for telemetry)."""
    if c.body_pos(part)[2] > z_thresh:
        store.setdefault("hold_grip", []).append(c.ctl.total_grip_force())


def _hold_grip(store: dict) -> float:
    s = store.get("hold_grip", [])
    return round(float(np.median(s)), 2) if s else 0.0


# ----- task builders -----

HOVER_Z = 0.21
GRASP_Z = 0.020
LIFT_Z = 0.23
PLACE_Z = 0.135

# The LEAP power grasp closes asymmetrically (thumb on -x), so the part settles to
# the thumb side of the palm centre.  This bias re-centres the part on the grip
# line; it is constant because the gantry is a pure Cartesian translation.
GRASP_BIAS = (-0.045, -0.015)


def _grasp_xy(part_name: str, nominal_xy: tuple[float, float], adaptive: bool = True):
    """Build a target_fn for the part's grasp position (+ grasp bias), captured once.

    ``adaptive=True`` senses the part's *live* position each rollout (closed-loop
    perception); ``adaptive=False`` targets the fixed nominal position (open-loop),
    which is the baseline used in the ablation study.
    """
    def fn_factory(z: float, yaw: float = 0.0):
        def fn(c: TaskContext, store: dict) -> GantryTarget:
            if "gxy" not in store:
                if adaptive:
                    p = c.body_pos(part_name)
                    base = (float(p[0]), float(p[1]))
                else:
                    base = nominal_xy
                store["gxy"] = (base[0] + GRASP_BIAS[0], base[1] + GRASP_BIAS[1])
            gx_, gy_ = store["gxy"]
            return GantryTarget(gx_, gy_, z, yaw)
        return fn
    return fn_factory


def sort_task(part, bin_, adaptive: bool = True) -> Task:
    bx, by, _ = bin_.pos
    g = _grasp_xy(part.name, (part.pos[0], part.pos[1]), adaptive)

    def score(c: TaskContext, store: dict) -> TaskResult:
        fp = c.body_pos(part.name)
        err = float(np.hypot(fp[0] - bx, fp[1] - by))
        success = err < 0.08 and fp[2] < 0.10
        return TaskResult(part.name, "sort", success,
                          {"placement_err_m": round(err, 4),
                           "final_z": round(float(fp[2]), 4),
                           "hold_grip_N": _hold_grip(store),
                           "recoveries": store.get("recoveries", 0)})

    def monitor(c: TaskContext, store: dict) -> None:
        _sample_hold_grip(c, store, part.name)

    steps = [
        Step("approach", None, "open", None, 1.4, target_fn=g(HOVER_Z)),
        Step("descend", None, "pregrasp", None, 1.1, target_fn=g(GRASP_Z), checkpoint=True),
        Step("grasp", None, "grasp", None, 0.9),
        Step("lift", None, None, None, 1.0, target_fn=g(LIFT_Z),
             verify=held(part.name)),
        Step("carry", GantryTarget(bx, by, LIFT_Z), None, None, 1.3),
        Step("lower", GantryTarget(bx, by, PLACE_Z), None, None, 0.9),
        Step("release", None, "open", None, 0.7),
        Step("retreat", GantryTarget(bx, by, HOVER_Z), None, None, 0.7),
    ]
    return Task(f"sort_{part.name}", "sort", steps, score, monitor)


def button_task(cfg: SceneConfig) -> Task:
    bx, by = -0.34, -0.24  # button housing location
    gx_, gy_ = bx + GRASP_BIAS[0], by + GRASP_BIAS[1]

    def score(c: TaskContext, store: dict) -> TaskResult:
        depth = store.get("max_press", 0.0)
        return TaskResult("inspect_button", "button", depth > 0.008,
                          {"press_depth_mm": round(depth * 1000, 2)})

    def monitor(c: TaskContext, store: dict) -> None:
        store["max_press"] = max(store.get("max_press", 0.0), -c.sensor("button_press"))

    steps = [
        Step("approach", GantryTarget(gx_, gy_, HOVER_Z), "pregrasp", None, 1.4),
        Step("press", GantryTarget(gx_, gy_, 0.010), "grasp", button_pressed(0.008), 2.0, settle=0.5),
        Step("release_btn", GantryTarget(gx_, gy_, HOVER_Z), "open", None, 0.9),
    ]
    return Task("inspect_button", "button", steps, score, monitor)


def peg_task(cfg: SceneConfig) -> Task:
    """Grasp the connector cube and seat it into the assembly socket.

    The socket is a wide-mouthed receptacle, so this mirrors the (reliable) sort
    drop: the cube is carried over the socket and released from bin height — the
    hand never enters the socket, so it cannot jam."""
    hx, hy = 0.22, 0.05           # socket centre
    g = _grasp_xy("peg", (0.24, -0.23))  # peg is not randomized; always sense live

    def score(c: TaskContext, store: dict) -> TaskResult:
        pp = c.body_pos("peg")
        err = float(np.hypot(pp[0] - hx, pp[1] - hy))
        success = err < 0.06 and pp[2] < 0.10
        return TaskResult("peg_insert", "peg", success,
                          {"align_err_m": round(err, 4), "peg_z": round(float(pp[2]), 4),
                           "hold_grip_N": _hold_grip(store)})

    def monitor(c: TaskContext, store: dict) -> None:
        _sample_hold_grip(c, store, "peg")

    steps = [
        Step("approach", None, "open", None, 2.0, target_fn=g(HOVER_Z)),
        Step("descend", None, "pregrasp", None, 1.1, target_fn=g(GRASP_Z), checkpoint=True),
        Step("grasp", None, "grasp", None, 0.9),
        Step("lift", None, None, None, 1.2, target_fn=g(LIFT_Z), verify=held("peg")),
        Step("carry", GantryTarget(hx, hy, LIFT_Z), None, None, 1.5),
        Step("lower", GantryTarget(hx, hy, PLACE_Z), None, None, 1.0),
        Step("release", None, "open", None, 0.7, settle=0.3),
        Step("retreat", GantryTarget(hx, hy, HOVER_Z), None, None, 0.8),
    ]
    return Task("peg_insert", "peg", steps, score, monitor)


INSPECT_Z = 0.27  # high inspection lift for the eye-in-hand camera


def inspect_sort_task(part, bin_, adaptive: bool = True) -> Task:
    """Pick a part, raise it to the eye-in-hand camera and hold it steady for
    inspection, then place it in its bin.  Scored on placement."""
    bx, by, _ = bin_.pos
    g = _grasp_xy(part.name, (part.pos[0], part.pos[1]), adaptive)

    def monitor(c: TaskContext, store: dict) -> None:
        if c.body_pos(part.name)[2] > 0.15:
            store["inspected"] = True  # part was raised to the inspection camera

    def score(c: TaskContext, store: dict) -> TaskResult:
        fp = c.body_pos(part.name)
        err = float(np.hypot(fp[0] - bx, fp[1] - by))
        placed = err < 0.08 and fp[2] < 0.10
        return TaskResult(f"inspect_sort_{part.name}", "inspect_sort", placed,
                          {"inspected": bool(store.get("inspected", False)),
                           "placement_err_m": round(err, 4),
                           "recoveries": store.get("recoveries", 0)})

    steps = [
        Step("approach", None, "open", None, 1.4, target_fn=g(HOVER_Z, 0.0)),
        Step("descend", None, "pregrasp", None, 1.1, target_fn=g(GRASP_Z, 0.0), checkpoint=True),
        Step("grasp", None, "grasp", None, 0.9),
        Step("lift", None, None, None, 1.0, target_fn=g(INSPECT_Z, 0.0),
             verify=held(part.name)),
        # Raise the part to the eye-in-hand camera and hold it steady for
        # inspection (scored on placement).
        Step("inspect_hold", None, None, None, 1.6, target_fn=g(INSPECT_Z, 0.0)),
        Step("carry", GantryTarget(bx, by, LIFT_Z), None, None, 1.3),
        Step("lower", GantryTarget(bx, by, PLACE_Z), None, None, 0.9),
        Step("release", None, "open", None, 0.7),
        Step("retreat", GantryTarget(bx, by, HOVER_Z), None, None, 0.7),
    ]
    return Task(f"inspect_sort_{part.name}", "inspect_sort", steps, score, monitor)


def cable_inspect_task(cfg: SceneConfig) -> Task:
    """Force-guided deformable cable inspection.

    The hand presses a fingertip onto a flexible connector cable and sweeps it
    sideways, bending the articulated cable elastically while the **wrist
    force-torque sensor** monitors the contact load.  Success requires both a
    measurable cable deflection and a registered (but gentle) contact force —
    demonstrating force-aware manipulation of a deformable body.
    """
    cx, cy, _ = CABLE_ANCHOR
    push_x = cx + (CABLE_SEGMENTS - 2) * CABLE_SEG_LEN + GRASP_BIAS[0]  # near the tip
    push_y = cy + GRASP_BIAS[1]

    def tip_y(c: TaskContext) -> float:
        return float(c.body_pos(CABLE_TIP_BODY)[1])

    def monitor(c: TaskContext, store: dict) -> None:
        if "rest_y" not in store:
            store["rest_y"] = tip_y(c)
        store["max_defl"] = max(store.get("max_defl", 0.0), abs(tip_y(c) - store["rest_y"]))
        store["max_load"] = max(store.get("max_load", 0.0), c.ctl.wrist_force_mag())

    def score(c: TaskContext, store: dict) -> TaskResult:
        defl = store.get("max_defl", 0.0)
        # success = the deformable cable was elastically bent past a threshold
        success = defl > 0.04
        return TaskResult("cable_inspect", "cable", success,
                          {"max_deflection_mm": round(defl * 1000, 1),
                           "peak_wrist_load_N": round(store.get("max_load", 0.0), 1)})

    steps = [
        Step("approach", GantryTarget(push_x, push_y, HOVER_Z), "grasp", None, 1.4),
        Step("touch", GantryTarget(push_x, push_y, 0.035), "grasp", None, 1.1),
        Step("sweep", GantryTarget(push_x, push_y + 0.10, 0.035), "grasp", None, 1.8),
        Step("retreat", GantryTarget(push_x, push_y + 0.10, HOVER_Z), "open", None, 0.8),
    ]
    return Task("cable_inspect", "cable", steps, score, monitor)


def default_arena(cfg: SceneConfig, adaptive: bool = True) -> list[Task]:
    """The full graded arena.

    Pick-and-sort the red and green parts, inspect-and-sort the blue part with an
    eye-in-hand camera lift, functional-test the diagnostic button, force-inspect
    the deformable harness cable, and finish by seating the connector (peg-in-hole).

    ``adaptive`` toggles live-position perception (closed-loop) vs fixed nominal
    targeting (open-loop baseline) for the grasp tasks — used by the ablation.
    """
    by_name = {p.name: p for p in cfg.parts}
    bin_of = {p.name: next(b for b in cfg.bins if b.name == p.bin_name) for p in cfg.parts}
    tasks = [
        sort_task(by_name["part_red"], bin_of["part_red"], adaptive),
        sort_task(by_name["part_green"], bin_of["part_green"], adaptive),
        inspect_sort_task(by_name["part_blue"], bin_of["part_blue"], adaptive),
        button_task(cfg),
        cable_inspect_task(cfg),
        peg_task(cfg),
    ]
    return tasks
