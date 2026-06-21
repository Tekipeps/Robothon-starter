"""Low-level control for the DexAssembly Cell.

Two layers live here:

* :class:`HandPose` presets and helpers that map symbolic finger poses (open,
  pre-grasp, power grasp, pinch, pointing) onto the 16 LEAP actuators.
* :class:`CellController`, a thin Cartesian interface over the gantry.  Because the
  gantry is a pure x/y/z translation plus a wrist yaw, the world position of the
  grasp centre is an affine function of the four gantry controls, which the
  controller calibrates once at construction and then inverts analytically.

The controller exposes *targets* only; integration over time (slewing controls
toward targets) is handled by the finite-state machine in :mod:`dexassembly.tasks`.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .scene import GANTRY_ACTUATORS, HAND_ACTUATORS

# Finger actuator order (per HAND_ACTUATORS): each finger has 4 dofs.
# index/middle/ring: [mcp, rot, pip, dip]; thumb: [cmc, axl, mcp, ipl].
# Poses are given as the 16-vector of target joint angles (radians).

_OPEN = np.array([
    0.0, 0.0, 0.0, 0.0,   # index
    0.0, 0.0, 0.0, 0.0,   # middle
    0.0, 0.0, 0.0, 0.0,   # ring
    0.5, 0.4, 0.2, 0.0,   # thumb pre-positioned to oppose the fingers
], dtype=float)

# Cupped pre-grasp: fingers partly bent so the open claw descends around an object
# before the power close.
_PREGRASP = np.array([
    0.5, 0.0, 0.4, 0.3,
    0.5, 0.0, 0.4, 0.3,
    0.5, 0.0, 0.4, 0.3,
    0.6, 0.5, 0.8, 0.3,
], dtype=float)

# Power grasp: the three fingers curl down + inward, the thumb opposes them.
# Tuned by sweep (_grasp_tune.py): grip force ~4.4 N on a 48 mm part.
_GRASP = np.array([
    1.3, 0.0, 1.4, 0.7,
    1.3, 0.0, 1.4, 0.7,
    1.3, 0.0, 1.4, 0.7,
    0.9, 0.5, 1.7, 0.6,
], dtype=float)

# Precision pinch between index and thumb only.
_PINCH = np.array([
    1.0, 0.0, 1.1, 0.7,
    0.2, 0.0, 0.3, 0.2,
    0.2, 0.0, 0.3, 0.2,
    0.6, 0.6, 1.5, 0.7,
], dtype=float)

# Pointing pose: index extended down to press a button, others tucked away.
_POINT = np.array([
    1.1, 0.0, 0.2, 0.1,
    0.2, 0.0, 1.7, 1.4,
    0.2, 0.0, 1.7, 1.4,
    1.6, 0.8, 1.6, 1.0,
], dtype=float)

HAND_POSES: dict[str, np.ndarray] = {
    "open": _OPEN,
    "pregrasp": _PREGRASP,
    "grasp": _GRASP,
    "pinch": _PINCH,
    "point": _POINT,
}


@dataclass
class GantryTarget:
    x: float
    y: float
    z: float
    yaw: float = 0.0


class CellController:
    """Cartesian + finger-pose command interface for the cell."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        self.gantry_act = [self._act(n) for n in GANTRY_ACTUATORS]
        self.hand_act = [self._act(n) for n in HAND_ACTUATORS]
        # joint qpos addresses for the hand, in HAND_ACTUATORS order
        self.hand_qadr = [
            int(model.jnt_qposadr[mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, n.replace("_act", ""))])
            for n in HAND_ACTUATORS
        ]
        self.grasp_site = self._grasp_center_offset()

    # ----- id helpers -----
    def _act(self, name: str) -> int:
        i = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if i < 0:
            raise KeyError(f"actuator {name!r} not found")
        return i

    def _body(self, name: str) -> int:
        return mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)

    def _grasp_center_offset(self) -> np.ndarray:
        """World offset from the gantry control origin to the grasp centre.

        The grasp centre is where the fingertips converge in the *closed* power
        grasp.  We freeze the hand at the grasp pose with all gantry controls at
        zero, read the fingertip centroid, and use it as the Cartesian reference:
        commanding the gantry to ``target`` places that convergence point at
        ``target`` so the closing fingers cage an object sitting there.
        """
        d = mujoco.MjData(self.model)
        for k, adr in enumerate(self.hand_qadr):
            d.qpos[adr] = HAND_POSES["grasp"][k]
        mujoco.mj_forward(self.model, d)
        tips = [self._body(b) for b in
                ("lh_if_ds", "lh_mf_ds", "lh_rf_ds", "lh_th_ds")]
        centroid = np.mean([d.xpos[t] for t in tips], axis=0)
        return centroid.copy()

    # ----- commands -----
    def set_gantry_target(self, t: GantryTarget) -> None:
        """Command the grasp centre toward world (x, y, z) and wrist to ``yaw``."""
        off = self.grasp_site
        self.data.ctrl[self.gantry_act[0]] = np.clip(t.x - off[0], *self._crange(0))
        self.data.ctrl[self.gantry_act[1]] = np.clip(t.y - off[1], *self._crange(1))
        self.data.ctrl[self.gantry_act[2]] = np.clip(t.z - off[2], *self._crange(2))
        self.data.ctrl[self.gantry_act[3]] = np.clip(t.yaw, *self._crange(3))

    def set_hand_pose(self, pose: str | np.ndarray, blend: float = 1.0) -> None:
        target = HAND_POSES[pose] if isinstance(pose, str) else np.asarray(pose)
        for k, act in enumerate(self.hand_act):
            cur = self.data.ctrl[act]
            self.data.ctrl[act] = (1 - blend) * cur + blend * target[k]

    def _crange(self, k: int) -> tuple[float, float]:
        lo, hi = self.model.actuator_ctrlrange[self.gantry_act[k]]
        return float(lo), float(hi)

    # ----- readouts -----
    def grasp_center(self) -> np.ndarray:
        tips = [self._body(b) for b in
                ("lh_if_ds", "lh_mf_ds", "lh_rf_ds", "lh_th_ds")]
        return np.mean([self.data.xpos[t] for t in tips], axis=0)

    def fingertip_forces(self) -> np.ndarray:
        """Return the four fingertip touch-sensor magnitudes (N)."""
        out = []
        for b in ("lh_if_ds", "lh_mf_ds", "lh_rf_ds", "lh_th_ds"):
            sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, f"{b}_force")
            adr = self.model.sensor_adr[sid]
            out.append(float(self.data.sensordata[adr]))
        return np.array(out)

    def total_grip_force(self) -> float:
        return float(self.fingertip_forces().sum())

    def _sensor_vec(self, name: str, dim: int) -> np.ndarray:
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        adr = self.model.sensor_adr[sid]
        return self.data.sensordata[adr:adr + dim].copy()

    def wrist_wrench(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the 6-axis wrist load as (force[3], torque[3]) in N / N·m."""
        return self._sensor_vec("wrist_force", 3), self._sensor_vec("wrist_torque", 3)

    def wrist_force_mag(self) -> float:
        return float(np.linalg.norm(self._sensor_vec("wrist_force", 3)))
