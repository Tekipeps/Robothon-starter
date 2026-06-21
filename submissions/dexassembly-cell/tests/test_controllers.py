"""Tests for the Cartesian controller and hand-pose presets."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from dexassembly.controllers import CellController, GantryTarget, HAND_POSES
from dexassembly.scene import compile_scene


@pytest.fixture(scope="module")
def cd():
    model, _ = compile_scene()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return CellController(model, data), model, data


def test_all_poses_are_16d():
    assert set(HAND_POSES) >= {"open", "pregrasp", "grasp", "pinch", "point"}
    for name, pose in HAND_POSES.items():
        assert np.asarray(pose).shape == (16,), name


def test_actuator_index_maps(cd):
    ctl, _, _ = cd
    assert len(ctl.gantry_act) == 4
    assert len(ctl.hand_act) == 16
    assert len(ctl.hand_qadr) == 16


def test_grasp_site_above_bench(cd):
    ctl, _, _ = cd
    assert 0.1 < ctl.grasp_site[2] < 0.4
    assert abs(ctl.grasp_site[0]) < 0.1 and abs(ctl.grasp_site[1]) < 0.1


def test_gantry_target_clips(cd):
    ctl, model, data = cd
    ctl.set_gantry_target(GantryTarget(99, 99, 99, 99))
    for a in ctl.gantry_act:
        lo, hi = model.actuator_ctrlrange[a]
        assert lo - 1e-6 <= data.ctrl[a] <= hi + 1e-6


def test_set_hand_pose_writes_ctrl(cd):
    ctl, model, data = cd
    ctl.set_hand_pose("grasp")
    cmd = np.array([data.ctrl[a] for a in ctl.hand_act])
    assert np.allclose(cmd, HAND_POSES["grasp"], atol=1e-6)


def test_hand_pose_blend(cd):
    ctl, model, data = cd
    ctl.set_hand_pose("open")
    ctl.set_hand_pose("grasp", blend=0.5)
    cmd = np.array([data.ctrl[a] for a in ctl.hand_act])
    expected = 0.5 * HAND_POSES["open"] + 0.5 * HAND_POSES["grasp"]
    assert np.allclose(cmd, expected, atol=1e-6)


def test_fingertip_forces_shape(cd):
    ctl, _, _ = cd
    assert ctl.fingertip_forces().shape == (4,)
    assert ctl.total_grip_force() >= 0.0


def test_wrist_wrench_shapes(cd):
    ctl, _, _ = cd
    f, t = ctl.wrist_wrench()
    assert f.shape == (3,) and t.shape == (3,)
    assert ctl.wrist_force_mag() >= 0.0


def test_grasp_center_is_finite(cd):
    ctl, _, _ = cd
    gc = ctl.grasp_center()
    assert gc.shape == (3,) and np.all(np.isfinite(gc))
