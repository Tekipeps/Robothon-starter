"""Structural tests for the procedurally-built scene."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from dexassembly.scene import SceneConfig, build_spec, compile_scene


@pytest.fixture(scope="module")
def model():
    m, _ = compile_scene()
    return m


def _has(model, kind, name):
    return mujoco.mj_name2id(model, kind, name) >= 0


def test_dof_and_actuator_counts(model):
    assert model.nu == 20
    assert model.nq == 56
    assert model.nv == 52


def test_sensor_count(model):
    # 16 hand jointpos + 4 touch + button + probe switch + force + torque
    assert model.nsensor == 24


def test_four_cameras(model):
    assert model.ncam == 4
    for cam in ("cam_hero", "cam_top", "cam_side", "cam_wrist"):
        assert _has(model, mujoco.mjtObj.mjOBJ_CAMERA, cam), cam


def test_force_torque_sensors_present(model):
    for s in ("wrist_force", "wrist_torque"):
        assert _has(model, mujoco.mjtObj.mjOBJ_SENSOR, s), s


def test_button_sensor_present(model):
    assert _has(model, mujoco.mjtObj.mjOBJ_SENSOR, "button_press")


def test_fingertip_touch_sensors(model):
    for f in ("if", "mf", "rf", "th"):
        assert _has(model, mujoco.mjtObj.mjOBJ_SENSOR, f"lh_{f}_ds_force"), f


def test_leap_hand_attached(model):
    assert _has(model, mujoco.mjtObj.mjOBJ_BODY, "lh_palm")
    for f in ("if", "mf", "rf", "th"):
        assert _has(model, mujoco.mjtObj.mjOBJ_BODY, f"lh_{f}_ds")


def test_gantry_joints(model):
    for j in ("gx", "gy", "gz", "wrist_yaw"):
        assert _has(model, mujoco.mjtObj.mjOBJ_JOINT, j), j


def test_parts_and_bins(model):
    for name in ("part_red", "part_green", "part_blue",
                 "bin_red_floor", "bin_green_floor", "bin_blue_floor"):
        assert _has(model, mujoco.mjtObj.mjOBJ_BODY, name) or \
               _has(model, mujoco.mjtObj.mjOBJ_GEOM, name), name


def test_tool_station(model):
    assert _has(model, mujoco.mjtObj.mjOBJ_BODY, "probe_tool")
    assert _has(model, mujoco.mjtObj.mjOBJ_BODY, "diag_well")
    assert _has(model, mujoco.mjtObj.mjOBJ_SENSOR, "probe_switch")


def test_deformable_cable_segments(model):
    for i in range(6):
        assert _has(model, mujoco.mjtObj.mjOBJ_BODY, f"cable_seg_{i}"), i
        assert _has(model, mujoco.mjtObj.mjOBJ_JOINT, f"cable_j_{i}"), i


def test_elliptic_cone_and_gravity(model):
    assert model.opt.cone == mujoco.mjtCone.mjCONE_ELLIPTIC
    assert model.opt.gravity[2] < 0


def test_parts_rest_on_bench_under_gravity(model):
    data = mujoco.MjData(model)
    for _ in range(400):
        mujoco.mj_step(model, data)
    # parts should settle near the bench top (z ~ part half-extent), not fall through
    for name in ("part_red", "part_green", "part_blue"):
        z = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)][2]
        assert 0.0 < z < 0.06, (name, z)


def test_default_config_has_three_parts_and_bins():
    cfg = SceneConfig.default()
    assert len(cfg.parts) == 3 and len(cfg.bins) == 3


def test_randomized_is_deterministic_and_perturbed():
    a = SceneConfig.randomized(7)
    b = SceneConfig.randomized(7)
    c = SceneConfig.randomized(8)
    assert [p.pos for p in a.parts] == [p.pos for p in b.parts]
    assert [p.pos for p in a.parts] != [p.pos for p in c.parts]
    base = SceneConfig.default()
    for pr, pb in zip(a.parts, base.parts):
        assert abs(pr.pos[0] - pb.pos[0]) <= 0.02 + 1e-9


def test_build_spec_emits_xml():
    spec = build_spec()
    xml = spec.to_xml()
    assert "<mujoco" in xml and "lh_palm" in xml
