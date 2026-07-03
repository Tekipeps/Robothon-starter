"""Mission engine: drives the Sentinel patrol and scores it.

A small state machine strings the locomotion primitives into an autonomous
inspection patrol:

    PATROL  — follow the waypoint route across the berm and rubble.
    INSPECT — on reaching a station, pivot in place to face the panel, hold a brief
              "scan", then resume.
    (a scripted lateral SHOVE is injected mid-course to test disturbance recovery.)

It logs measurable objectives — terrain crossed, panels scanned, shove survived,
finish reached, stayed upright throughout — and turns them into a 0–100 score, the
same per-objective structure the demo's scorecard reports.  Control is only ever
through ``data.ctrl`` (the gait) and a brief external ``xfrc`` shove; the trunk pose
is read back from the live physics, never written.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import mujoco
import numpy as np

from .controller import WaypointFollower, trunk_yaw, _wrap
from .gait import TrotGait
from .proprio import DisturbanceReflex, Proprioception
from .scene import CourseConfig, compile_scene

UPRIGHT_Z = 0.16        # trunk height below this = the robot has fallen
SCAN_DWELL = 1.1        # seconds held facing a panel to count as "inspected"
FACE_TOL = 0.18         # rad heading error within which the panel is "in view"


@dataclass
class StepInfo:
    t: float
    phase: str
    pos: np.ndarray
    yaw: float
    forward: float
    yaw_cmd: float
    target: str
    inspecting: str
    progress: float
    fell: bool
    objectives: dict = field(default_factory=dict)
    brace: float = 0.0
    feet: dict = field(default_factory=dict)


@dataclass
class Mission:
    cfg: CourseConfig
    seed: int = 0
    model: mujoco.MjModel = field(init=False)
    data: mujoco.MjData = field(init=False)

    def __post_init__(self) -> None:
        self.model, self.spec = compile_scene(self.cfg)
        self.data = mujoco.MjData(self.model)
        self._trunk = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "trunk")
        self._panel_xy = {ins.name: np.asarray(ins.post_xy) for ins in self.cfg.inspections}
        self.reset()

    def reset(self) -> None:
        mujoco.mj_resetDataKeyframe(
            self.model, self.data,
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home"))
        self.gait = TrotGait(self.model)
        self.follower = WaypointFollower(self.cfg.waypoints)
        self.proprio = Proprioception(self.model)
        self.reflex = DisturbanceReflex()

    # ----- helpers -----
    def _pos(self) -> np.ndarray:
        return self.data.qpos[0:2].copy()

    def _yaw(self) -> float:
        return trunk_yaw(self.data.qpos[3:7])

    def _fallen(self) -> bool:
        return float(self.data.qpos[2]) < UPRIGHT_Z

    # ----- main loop -----
    def run(self, on_step: Optional[Callable[[StepInfo], None]] = None,
            timeout_s: float = 70.0) -> dict:
        dt = self.model.opt.timestep
        m, d = self.model, self.data

        scanned: list[str] = []
        crossed = {"berm": False, "rubble": False}
        push_done = {"fired": False, "t0": -1.0, "t_end": -1.0, "survived": True}
        ever_fell = [False]

        phase = "PATROL"
        scan_target = ""           # inspection name currently being scanned
        scan_start = 0.0
        wall0 = time.time()
        n = int(timeout_s / dt)

        for k in range(n):
            t = k * dt
            pos, yaw = self._pos(), self._yaw()

            # --- terrain progress flags (only credited while upright) ---
            if not self._fallen():
                if pos[0] > 2.30:
                    crossed["berm"] = True
                if pos[0] > 3.70:
                    crossed["rubble"] = True

            # --- scripted lateral shove as the robot leaves the berm ---
            d.xfrc_applied[self._trunk, :] = 0.0
            if not push_done["fired"] and pos[0] > self.cfg.push_at_x:
                push_done["fired"] = True
                push_done["t0"] = t
                push_done["t_end"] = t + self.cfg.push_duration
            if push_done["fired"] and t < push_done["t_end"]:
                d.xfrc_applied[self._trunk, :3] = self.cfg.push_force
                phase_push = True
            else:
                phase_push = False

            # --- state machine ---
            if phase == "PATROL":
                forward, yaw_cmd = self.follower.command(pos, yaw)
                # did we just arrive at a station that has a panel to inspect?
                just = self.follower.reached[-1] if self.follower.reached else None
                pending = self.cfg.inspect_at.get(just) if just else None
                if pending and pending not in scanned and pending != scan_target:
                    phase, scan_target, scan_start = "INSPECT", pending, t
            elif phase == "INSPECT":
                tgt = self._panel_xy[scan_target]
                desired = float(np.arctan2(tgt[1] - pos[1], tgt[0] - pos[0]))
                err = _wrap(desired - yaw)
                forward, yaw_cmd = 0.0, float(np.clip(2.0 * err, -1.0, 1.0))
                if abs(err) < FACE_TOL:
                    if t - scan_start > SCAN_DWELL:
                        scanned.append(scan_target)
                        phase, scan_target = "PATROL", ""
                else:
                    scan_start = t      # restart dwell until steadily facing it
            else:
                forward, yaw_cmd = 0.0, 0.0

            # --- proprioceptive brace reflex: the IMU feels the hit and the gait
            #     crouches while it keeps stepping (the step pattern is what
            #     catches the body).  Triggering is suppressed while the
            #     controller itself commands an aggressive pivot (reafference
            #     gating); it has no privileged knowledge of the scripted shove.
            self_induced = phase == "INSPECT" or abs(yaw_cmd) > 0.5
            brace = self.reflex.update(t, dt, float(self.proprio.accel(d)[1]),
                                       suppress=self_induced)

            self.gait.step(d, dt, forward=forward, yaw=yaw_cmd, brace=brace)
            mujoco.mj_step(m, d)

            if self._fallen():
                ever_fell[0] = True
                if phase_push:
                    push_done["survived"] = False

            live_obj = {
                "inspect_panel_A": "panel_A" in scanned,
                "crest_berm": crossed["berm"],
                "push_recovery": push_done["fired"] and push_done["survived"] and not ever_fell[0],
                "cross_rubble": crossed["rubble"],
                "inspect_panel_B": "panel_B" in scanned,
                "reach_finish": "finish" in self.follower.reached,
                "stayed_upright": not ever_fell[0],
            }
            if on_step is not None:
                on_step(StepInfo(
                    t=t, phase=("SHOVE" if phase_push else phase), pos=pos, yaw=yaw,
                    forward=forward, yaw_cmd=yaw_cmd,
                    target=(self.follower.target.name if self.follower.target else "—"),
                    inspecting=scan_target, progress=min(1.0, self.follower.idx
                                                         / max(1, len(self.cfg.waypoints))),
                    fell=self._fallen(), objectives=live_obj,
                    brace=brace, feet=self.proprio.foot_contacts(d)))

            if ever_fell[0] or (self.follower.done and phase == "PATROL"):
                break

        objectives = live_obj
        n_pass = sum(objectives.values())
        # reflex latency: time from shove onset to the IMU-triggered brace
        lat_ms = None
        after = [e for e in self.reflex.events if e >= push_done["t0"] - 1e-9]
        if push_done["fired"] and after:
            lat_ms = round(1000.0 * (after[0] - push_done["t0"]), 1)
        return {
            "n_objectives": len(objectives),
            "n_success": n_pass,
            "score_0_100": round(100.0 * n_pass / len(objectives), 2),
            "waypoints_reached": self.follower.reached,
            "panels_scanned": scanned,
            "sim_time_s": round(float(d.time), 2),
            "wall_time_s": round(time.time() - wall0, 2),
            "reflex": {
                "triggered": self.reflex.triggered,
                "events_s": self.reflex.events,
                "shove_onset_s": round(push_done["t0"], 3) if push_done["fired"] else None,
                "latency_ms": lat_ms,
                "push_force_N": float(-np.asarray(self.cfg.push_force)[1]),
            },
            "objectives": objectives,
        }
