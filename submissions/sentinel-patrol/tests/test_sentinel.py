"""Unit tests for the Sentinel patrol stack.

    python -m pytest tests/ -q
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sentinel.controller import WaypointFollower, _wrap, trunk_yaw
from sentinel.gait import _CALF_L, _THIGH_L, _leg_ik
from sentinel.proprio import DisturbanceReflex
from sentinel.scene import CourseConfig, Waypoint, compile_scene


# ---- gait / IK ------------------------------------------------------------

def _fk(thigh: float, calf: float) -> tuple[float, float]:
    """Planar forward kinematics matching the IK's frame."""
    fx = -(_THIGH_L * np.sin(thigh) + _CALF_L * np.sin(thigh + calf))
    fz = -(_THIGH_L * np.cos(thigh) + _CALF_L * np.cos(thigh + calf))
    return fx, fz


@pytest.mark.parametrize("fx,fz", [(0.0, -0.26), (0.05, -0.24), (-0.05, -0.30),
                                   (0.08, -0.21), (-0.03, -0.35)])
def test_leg_ik_round_trip(fx, fz):
    thigh, calf = _leg_ik(fx, fz)
    rx, rz = _fk(thigh, calf)
    assert abs(rx - fx) < 1e-6 and abs(rz - fz) < 1e-6


def test_leg_ik_clamps_unreachable():
    thigh, calf = _leg_ik(0.0, -1.0)          # beyond leg length: clamps, no NaN
    assert np.isfinite(thigh) and np.isfinite(calf)


# ---- controller ------------------------------------------------------------

def test_wrap():
    assert abs(_wrap(3 * np.pi) - np.pi) < 1e-9 or abs(_wrap(3 * np.pi) + np.pi) < 1e-9
    assert abs(_wrap(0.3) - 0.3) < 1e-12


def test_trunk_yaw_identity_quat():
    assert trunk_yaw(np.array([1.0, 0, 0, 0])) == 0.0


def test_follower_advances_and_finishes():
    f = WaypointFollower([Waypoint("a", (1.0, 0.0)), Waypoint("b", (2.0, 0.0))])
    fwd, yaw = f.command(np.array([0.0, 0.0]), 0.0)
    assert fwd > 0 and abs(yaw) < 1e-6 and not f.done
    f.command(np.array([1.0, 0.05]), 0.0)     # inside a's radius -> advance
    assert f.reached == ["a"]
    f.command(np.array([2.0, 0.0]), 0.0)
    assert f.done and f.reached == ["a", "b"]


# ---- reflex ----------------------------------------------------------------

def test_reflex_ignores_impulsive_spikes_but_latches_on_sustained_hit():
    r = DisturbanceReflex()
    dt, t = 0.002, 0.0
    for _ in range(100):                      # single-step spikes (foot impacts)
        t += dt
        r.update(t, dt, 20.0 if int(t / dt) % 25 == 0 else 0.0)
    assert not r.triggered
    for _ in range(50):                       # sustained lateral acceleration
        t += dt
        b = r.update(t, dt, 8.0)
    assert r.triggered and b > 0.0


def test_reflex_suppression_gates_trigger():
    r = DisturbanceReflex()
    dt, t = 0.002, 0.0
    for _ in range(100):
        t += dt
        r.update(t, dt, 8.0, suppress=True)
    assert not r.triggered


def test_reflex_envelope_attack_hold_release():
    r = DisturbanceReflex()
    dt, t = 0.002, 0.0
    while not r.triggered:
        t += dt
        r.update(t, dt, 8.0)
    t0 = r.events[0]
    assert r.update(t0 + r.attack_s + 0.01, dt, 0.0) == 1.0
    assert r.update(t0 + r.hold_s + r.release_s + 0.01, dt, 0.0) == 0.0


# ---- scene -----------------------------------------------------------------

def test_scene_compiles_with_sensors_and_cameras():
    import mujoco
    model, _ = compile_scene(CourseConfig())
    assert model.nu == 12
    for name in ("imu_gyro", "imu_accel", "imu_vel",
                 "touch_FR", "touch_FL", "touch_RR", "touch_RL"):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name) >= 0
    for cam in ("head", "chase", "overview"):
        assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam) >= 0
