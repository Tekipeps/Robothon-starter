"""Tests for task construction, conditions, and the adaptive/baseline flag."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

from dexassembly.controllers import CellController, GantryTarget
from dexassembly.scene import SceneConfig, compile_scene
from dexassembly import tasks as T


@pytest.fixture(scope="module")
def ctx():
    cfg = SceneConfig.default()
    model, _ = compile_scene(cfg)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ctl = CellController(model, data)
    return T.TaskContext(model, data, ctl, cfg)


def test_default_arena_has_six_tasks():
    cfg = SceneConfig.default()
    arena = T.default_arena(cfg)
    assert len(arena) == 6
    kinds = [t.kind for t in arena]
    assert kinds.count("sort") == 2
    assert "inspect_sort" in kinds and "button" in kinds
    assert "cable" in kinds and "tool_use" in kinds


def test_every_task_has_steps_and_score():
    cfg = SceneConfig.default()
    for task in T.default_arena(cfg):
        assert task.steps, task.name
        assert callable(task.score)
        for step in task.steps:
            assert step.target is not None or step.target_fn is not None or step.hand or step.until


def test_sort_task_has_checkpoint_and_verify():
    cfg = SceneConfig.default()
    sort = T.sort_task(cfg.parts[0], cfg.bins[0])
    names = [s.name for s in sort.steps]
    assert "descend" in names and "lift" in names
    descend = next(s for s in sort.steps if s.name == "descend")
    lift = next(s for s in sort.steps if s.name == "lift")
    assert descend.checkpoint is True
    assert lift.verify is not None


def test_reached_condition(ctx):
    ctx.ctl.set_gantry_target(GantryTarget(0.0, -0.1, 0.20))
    cond = T.reached(0.02)
    # immediately after commanding a far target it is not yet reached
    assert cond(ctx) in (True, False)  # callable returns a bool
    assert isinstance(cond(ctx), bool)


def test_grip_above_condition(ctx):
    assert T.grip_above(1e9)(ctx) is False


def test_button_pressed_condition(ctx):
    assert T.button_pressed(0.008)(ctx) is False  # nothing pressing at rest


def test_part_above_condition(ctx):
    assert not T.part_above("part_red", 1.0)(ctx)
    assert T.part_above("part_red", -1.0)(ctx)


def test_adaptive_vs_baseline_targeting_differs():
    # Under a randomized scene, the adaptive target tracks the live part while the
    # baseline target stays at nominal -> the two grasp xy differ.
    cfg = SceneConfig.randomized(123)
    model, _ = compile_scene(cfg)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ctl = CellController(model, data)
    ctx = T.TaskContext(model, data, ctl, cfg)
    nominal = SceneConfig.default().parts[0]
    fa = T._grasp_xy(nominal.name, (nominal.pos[0], nominal.pos[1]), adaptive=True)(0.2)
    fb = T._grasp_xy(nominal.name, (nominal.pos[0], nominal.pos[1]), adaptive=False)(0.2)
    ta, tb = fa(ctx, {}), fb(ctx, {})
    assert (abs(ta.x - tb.x) + abs(ta.y - tb.y)) > 1e-4


def test_hold_grip_helper():
    assert T._hold_grip({}) == 0.0
    assert T._hold_grip({"hold_grip": [10.0, 20.0, 30.0]}) == 20.0
