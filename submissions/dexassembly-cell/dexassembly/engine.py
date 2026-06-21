"""Execution engine: runs closed-loop tasks and streams per-step callbacks.

The engine owns the MuJoCo model/data, a :class:`CellController`, and a list of
:class:`Task`.  :meth:`run` executes every task step, advancing on the step's
condition or timeout, and invokes an optional ``on_step`` callback each physics
step (used by the recorder and the video renderer).  It returns a structured
report with per-task results and aggregate metrics.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import mujoco
import numpy as np

from .controllers import CellController
from .scene import SceneConfig, compile_scene
from .tasks import Task, TaskContext, TaskResult, default_arena


@dataclass
class StepInfo:
    task: str
    kind: str
    phase: str
    t: float
    grip: float
    forces: np.ndarray
    progress: float  # 0..1 over the whole arena


@dataclass
class Engine:
    cfg: SceneConfig
    seed: int = 0
    max_retries: int = 1  # closed-loop grasp-failure recovery attempts per task
    model: mujoco.MjModel = field(init=False)
    data: mujoco.MjData = field(init=False)
    ctl: CellController = field(init=False)
    ctx: TaskContext = field(init=False)

    def __post_init__(self) -> None:
        self.model, _ = compile_scene(self.cfg)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)
        self.ctl = CellController(self.model, self.data)
        self.ctx = TaskContext(self.model, self.data, self.ctl, self.cfg)

    def run(
        self,
        tasks: Optional[list[Task]] = None,
        on_step: Optional[Callable[[StepInfo], None]] = None,
    ) -> dict:
        tasks = tasks or default_arena(self.cfg)
        total_steps = sum(len(t.steps) for t in tasks)
        done_steps = 0
        results: list[TaskResult] = []
        wall0 = time.time()

        for task in tasks:
            store: dict = {}
            steps = task.steps
            i, checkpoint, retries = 0, 0, 0
            while i < len(steps):
                step = steps[i]
                if step.checkpoint:
                    checkpoint = i
                self._apply(step, store)
                self._drive(task, step, done_steps, total_steps, on_step, store)
                done_steps += 1
                if (step.verify is not None and not step.verify(self.ctx)
                        and retries < self.max_retries):
                    # tactile/pose check failed -> recover from the last checkpoint
                    retries += 1
                    store["recoveries"] = retries
                    store.pop("gxy", None)  # re-sense the part on the retry
                    i = checkpoint
                    continue
                i += 1
            results.append(task.score(self.ctx, store))

        sim_time = float(self.data.time)
        succ = [r for r in results if r.success]
        report = {
            "n_tasks": len(results),
            "n_success": len(succ),
            "success_rate": round(len(succ) / max(1, len(results)), 4),
            "score_0_100": round(100.0 * len(succ) / max(1, len(results)), 2),
            "sim_time_s": round(sim_time, 2),
            "wall_time_s": round(time.time() - wall0, 2),
            "tasks": [r.__dict__ for r in results],
        }
        return report

    # ----- internals -----
    def _apply(self, step, store) -> None:
        target = step.target
        if step.target_fn is not None:
            target = step.target_fn(self.ctx, store)
        if target is not None:
            self.ctl.set_gantry_target(target)
        if step.hand is not None:
            self.ctl.set_hand_pose(step.hand)

    def _drive(self, task, step, done_steps, total_steps, on_step, store) -> bool:
        t0 = self.data.time
        met = step.until is None
        while self.data.time - t0 < step.timeout:
            mujoco.mj_step(self.model, self.data)
            self._monitor(task, store)
            self._emit(task, step, done_steps, total_steps, on_step)
            if step.until is not None and step.until(self.ctx):
                met = True
                break
        t1 = self.data.time
        while self.data.time - t1 < step.settle:
            mujoco.mj_step(self.model, self.data)
            self._monitor(task, store)
            self._emit(task, step, done_steps, total_steps, on_step)
        return met

    def _monitor(self, task, store) -> None:
        if task.monitor is not None:
            task.monitor(self.ctx, store)

    def _emit(self, task, step, done_steps, total_steps, on_step) -> None:
        if on_step is None:
            return
        on_step(StepInfo(
            task=task.name, kind=task.kind, phase=step.name,
            t=float(self.data.time), grip=self.ctl.total_grip_force(),
            forces=self.ctl.fingertip_forces(),
            progress=done_steps / max(1, total_steps),
        ))
