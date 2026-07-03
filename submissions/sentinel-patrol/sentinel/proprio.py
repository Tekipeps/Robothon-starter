"""Proprioception: the robot's onboard senses, and the reflex built on them.

:class:`Proprioception` is a thin reader over the model's **real MuJoCo sensors**
— the IMU triad (gyro / accelerometer / velocimeter on the trunk ``imu`` site) and
the four foot **touch** sensors — resolving named sensor addresses once and
returning views of ``data.sensordata`` each tick.  Everything the controller
"feels" flows through here; the reflex never reads privileged simulator state.

:class:`DisturbanceReflex` is a sensor-driven brace response: when the IMU
velocimeter reports a lateral speed no normal trot produces (the signature of an
external shove), it commands a **brace** — the gait crouches and widens its stance
for a fraction of a second, dropping the centre of mass so the push is absorbed
instead of toppling the robot.  The trigger is purely proprioceptive: the reflex
does not know when (or whether) a shove is scripted, it only feels the hit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

FEET = ("FR", "FL", "RR", "RL")


class Proprioception:
    """Named access to the Go1's onboard sensor readings."""

    def __init__(self, model: mujoco.MjModel):
        def adr(name: str) -> int:
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
            if sid < 0:
                raise ValueError(f"sensor not in model: {name}")
            return int(model.sensor_adr[sid])

        self._gyro = adr("imu_gyro")
        self._accel = adr("imu_accel")
        self._vel = adr("imu_vel")
        self._touch = {f: adr(f"touch_{f}") for f in FEET}

    def gyro(self, data: mujoco.MjData) -> np.ndarray:
        """Trunk angular velocity, body frame (rad/s)."""
        return data.sensordata[self._gyro:self._gyro + 3]

    def accel(self, data: mujoco.MjData) -> np.ndarray:
        """Trunk linear acceleration, body frame, gravity included (m/s^2)."""
        return data.sensordata[self._accel:self._accel + 3]

    def body_vel(self, data: mujoco.MjData) -> np.ndarray:
        """Trunk linear velocity, body frame (m/s)."""
        return data.sensordata[self._vel:self._vel + 3]

    def lateral_speed(self, data: mujoco.MjData) -> float:
        """|body-frame sideways speed| — the disturbance signature."""
        return float(abs(data.sensordata[self._vel + 1]))

    def foot_forces(self, data: mujoco.MjData) -> dict[str, float]:
        """Normal contact force felt by each foot's touch sensor (N)."""
        return {f: float(data.sensordata[a]) for f, a in self._touch.items()}

    def foot_contacts(self, data: mujoco.MjData, thresh: float = 0.5) -> dict[str, bool]:
        return {f: float(data.sensordata[a]) > thresh for f, a in self._touch.items()}


@dataclass
class DisturbanceReflex:
    """IMU-triggered crouch brace: drop the centre of mass when a hit is felt.

    The trigger channel is the accelerometer's **body-frame lateral acceleration**
    passed through a first-order low-pass (``tau_s``).  The trot's own foot-impact
    spikes are large but only milliseconds long, so the filter flattens them
    (steady walking stays under ~3.4 m/s² filtered, the whole course included);
    an external shove is *sustained*, so the filtered signal crosses the trigger
    within a few milliseconds of the hit.  Crossing ``trigger_acc`` latches the
    brace — a fast crouch (``attack_s``) held for ``hold_s`` and released smoothly
    — which the gait maps onto a lower ride height.  Detection is purely
    proprioceptive: the reflex never knows whether or when a shove is scripted,
    it only feels the hit through the IMU.
    """

    trigger_acc: float = 3.6     # m/s² filtered lateral accel that latches the brace
    tau_s: float = 0.025         # low-pass time constant on the accel channel
    hold_s: float = 0.70         # how long the crouch is held after a trigger
    attack_s: float = 0.03       # ramp-in (fast drop into the crouch)
    release_s: float = 0.35      # ramp-out (smooth stand back up)
    events: list = field(default_factory=list)   # trigger timestamps (s)
    _t_trig: float = -1e9
    _lp: float = 0.0             # low-pass filter state

    def update(self, t: float, dt: float, lateral_accel: float,
               suppress: bool = False) -> float:
        """Advance to time ``t`` with the current IMU reading; return brace [0,1].

        ``suppress`` is reafference gating: while the controller itself commands
        an aggressive motion (a pivot-in-place), its self-induced accelerations
        must not be mistaken for an external hit, so triggering is inhibited —
        an already-latched brace still plays out its envelope.
        """
        self._lp += (dt / self.tau_s) * (float(lateral_accel) - self._lp)
        if (abs(self._lp) > self.trigger_acc and not suppress
                and t - self._t_trig > self.hold_s + self.release_s):
            # latch once and let the full brace envelope play out — the signal
            # stays above threshold for the whole hit, so re-latching here would
            # keep the brace pinned at the start of its attack ramp
            self.events.append(round(t, 4))
            self._t_trig = t
        rel = t - self._t_trig
        if rel < 0:
            return 0.0
        if rel < self.attack_s:
            return rel / self.attack_s
        if rel < self.hold_s:
            return 1.0
        if rel < self.hold_s + self.release_s:
            return 1.0 - (rel - self.hold_s) / self.release_s
        return 0.0

    @property
    def triggered(self) -> bool:
        return bool(self.events)
