"""Engine, recorder, HUD, evaluation, and end-to-end tests."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from dexassembly.engine import Engine, StepInfo
from dexassembly.hud import draw_hud
from dexassembly.record import DataRecorder
from dexassembly.scene import SceneConfig, compile_scene
from dexassembly.tasks import button_task, sort_task


def test_hud_returns_same_shape():
    frame = np.zeros((360, 640, 3), dtype=np.uint8)
    out = draw_hud(frame, title="t", task="x", phase="y",
                   forces=np.array([1.0, 2, 3, 4]), grip=10.0, progress=0.5,
                   score_line="arena 50%", caption="hello world")
    assert out.shape == frame.shape and out.dtype == np.uint8


def test_engine_report_structure():
    cfg = SceneConfig.default()
    rep = Engine(cfg).run(tasks=[button_task(cfg)])
    for key in ("n_tasks", "n_success", "success_rate", "score_0_100", "tasks"):
        assert key in rep
    assert rep["n_tasks"] == 1
    assert rep["tasks"][0]["name"] == "inspect_button"


def test_engine_step_callback_fires():
    cfg = SceneConfig.default()
    seen = []
    Engine(cfg).run(tasks=[button_task(cfg)], on_step=lambda i: seen.append(i))
    assert seen and isinstance(seen[0], StepInfo)
    assert 0.0 <= seen[-1].progress <= 1.0


def test_recorder_writes_dataset(tmp_path: Path):
    cfg = SceneConfig.default()
    model, _ = compile_scene(cfg)
    data = mujoco.MjData(model)
    rec = DataRecorder(model, data, tmp_path, save_images=False, sample_hz=50)
    for _ in range(300):
        mujoco.mj_step(model, data)
        rec.maybe_record("t", "k", "p", np.zeros(4))
    meta = rec.finalize()
    assert meta["n_samples"] > 0
    assert (tmp_path / "trajectory.jsonl").exists()
    assert (tmp_path / "states.npz").exists()
    npz = np.load(tmp_path / "states.npz")
    assert npz["qpos"].shape[1] == model.nq


@pytest.mark.slow
def test_single_sort_succeeds():
    cfg = SceneConfig.default()
    rep = Engine(cfg).run(tasks=[sort_task(cfg.parts[0], cfg.bins[0])])
    assert rep["tasks"][0]["success"], rep["tasks"][0]


@pytest.mark.slow
def test_full_arena_scores_100():
    rep = Engine(SceneConfig.default()).run()
    assert rep["n_tasks"] == 6
    assert rep["score_0_100"] == 100.0, rep["tasks"]


@pytest.mark.slow
def test_adaptive_beats_open_loop_under_randomization():
    from dexassembly.evaluate import evaluate
    summary = evaluate(n_rollouts=4, seed0=500)
    a = summary["modes"]["adaptive"]["mean_success_rate"]
    b = summary["modes"]["baseline_open_loop"]["mean_success_rate"]
    assert a >= b  # closed-loop perception + recovery should never hurt
    assert a >= 0.6  # adaptive stays reasonably robust under randomization
