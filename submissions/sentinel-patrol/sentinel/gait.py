"""Trot-gait locomotion controller for the Unitree Go1.

The Go1 is driven by 12 position actuators (4 legs x hip/thigh/calf).  This module
turns a high-level **velocity command** ``(forward_speed, yaw_rate)`` into those 12
joint targets every control tick, using a classic foot-trajectory trot:

* The four feet are split into two diagonal pairs (FR+RL and FL+RR) that swing in
  anti-phase, so three feet (or two diagonal feet) are always on the ground — the
  statically/dynamically stable gait quadrupeds actually use to trot.
* Each foot follows a planar trajectory in its leg's sagittal plane: a **stance**
  phase where the planted foot is swept backward (propelling the body forward) and
  a **swing** phase where it lifts and returns to the front.
* A small per-side stride scaling turns the body in place (skid-steer), so the same
  controller both walks and steers toward a waypoint.

Planar 2-link inverse kinematics (thigh L1, calf L2, both 0.213 m on the Go1) maps
each Cartesian foot target ``(fx, fz)`` to ``(thigh, calf)`` joint angles; the hip
abduction joint stays at zero for a straight, upright trunk.  All control is through
``data.ctrl`` — the robot walks by real foot-ground contact, never qpos teleport.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

LEGS = ("FR", "FL", "RR", "RL")
_THIGH_L = 0.213          # thigh link length (m)
_CALF_L = 0.213           # calf link length (m)

# Trot diagonal phasing: FR & RL move together; FL & RR a half-cycle later.
_PHASE = {"FR": 0.0, "FL": 0.5, "RR": 0.5, "RL": 0.0}
# Which side each leg is on (for skid-steer turning).
_SIDE = {"FR": +1.0, "RR": +1.0, "FL": -1.0, "RL": -1.0}  # +1 = right, -1 = left


def _leg_ik(fx: float, fz: float) -> tuple[float, float]:
    """Planar 2-link IK: foot target (fx forward, fz up, relative to the hip) ->
    (thigh, calf) joint angles.  fz is negative (foot below the hip)."""
    r = min(np.hypot(fx, fz), _THIGH_L + _CALF_L - 1e-3)
    cos_calf = np.clip((r * r - _THIGH_L ** 2 - _CALF_L ** 2)
                       / (2.0 * _THIGH_L * _CALF_L), -1.0, 1.0)
    calf = -np.arccos(cos_calf)                       # knee bends backward (negative)
    thigh = np.arctan2(-fx, -fz) - np.arctan2(_CALF_L * np.sin(calf),
                                              _THIGH_L + _CALF_L * np.cos(calf))
    return float(thigh), float(calf)


@dataclass
class GaitParams:
    stand_z: float = -0.26      # nominal foot height below the hip (sets ride height)
    stride: float = 0.10        # full fore-aft foot travel during stance (m)
    swing_height: float = 0.08  # peak foot lift during swing (m)
    freq: float = 2.2           # gait cycles per second
    turn_gain: float = 0.6      # how strongly yaw_rate skews the per-side stride
    brace_crouch: float = 0.05  # ride-height drop at full brace (m)
    # Stance-widening (hip abduction) was tried as part of the brace and measured
    # *worse* than a pure crouch — the abduction breaks the planar-IK stance mid
    # push — so the brace keeps the hips at zero and only crouches.
    brace_hip: float = 0.0      # hip abduction at full brace (rad); 0 = crouch-only


class TrotGait:
    """Maps ``(forward, yaw)`` velocity commands to the Go1's 12 joint targets."""

    def __init__(self, model: mujoco.MjModel, params: GaitParams | None = None):
        self.p = params or GaitParams()
        self.act = {
            lg: [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{lg}_{j}")
                 for j in ("hip", "thigh", "calf")]
            for lg in LEGS
        }
        self._t = 0.0

    def reset(self) -> None:
        self._t = 0.0

    def _foot(self, phase: float, stride: float) -> tuple[float, float]:
        """Foot (fx, fz) for a leg at the given gait ``phase`` in [0,1)."""
        if phase < 0.5:                                   # stance: sweep foot backward
            s = phase / 0.5
            fx = stride * (0.5 - s)
            fz = self.p.stand_z
        else:                                             # swing: lift and return front
            s = (phase - 0.5) / 0.5
            fx = stride * (-0.5 + s)
            fz = self.p.stand_z + self.p.swing_height * np.sin(np.pi * s)
        return fx, fz

    def step(self, data: mujoco.MjData, dt: float,
             forward: float = 1.0, yaw: float = 0.0, brace: float = 0.0) -> None:
        """Advance the gait clock by ``dt`` and write the 12 joint targets.

        ``forward`` in [0,1] scales stride (walk speed); ``yaw`` in [-1,1] skid-steers
        (positive = turn left).  ``forward=0, yaw=0`` holds a stable stand.

        ``brace`` in [0,1] is the disturbance-reflex level (from
        :class:`~sentinel.proprio.DisturbanceReflex`): it crouches the ride
        height, dropping the centre of mass so a lateral hit is absorbed instead
        of toppling the robot.
        """
        self._t += dt
        moving = abs(forward) > 1e-3 or abs(yaw) > 1e-3
        crouch = self.p.brace_crouch * brace
        for lg in LEGS:
            if not moving:
                fx, fz = 0.0, self.p.stand_z + crouch     # planted stand
            else:
                phase = (self._t * self.p.freq + _PHASE[lg]) % 1.0
                # Skid-steer as a per-side stride: ``forward`` strides all legs
                # equally; the turn term adds to the right side and subtracts from
                # the left (so +yaw turns counter-clockwise / left).  With forward=0
                # the two sides stride oppositely and the robot pivots in place —
                # used to turn and face an inspection target.
                stride = self.p.stride * (forward + self.p.turn_gain * yaw * _SIDE[lg])
                fx, fz = self._foot(phase, stride)
                fz += crouch
            thigh, calf = _leg_ik(fx, fz)
            data.ctrl[self.act[lg][0]] = -_SIDE[lg] * self.p.brace_hip * brace
            data.ctrl[self.act[lg][1]] = thigh
            data.ctrl[self.act[lg][2]] = calf
