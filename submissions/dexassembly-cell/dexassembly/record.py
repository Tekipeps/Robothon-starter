"""Trajectory / tele-metry recorder for data-collection runs.

The recorder snapshots the full simulator state (joint positions & velocities,
commanded controls, fingertip tactile forces, contact count, and the active
task/phase label) at a fixed sample rate, plus optional RGB+depth frames from a
named camera.  At the end it writes:

* ``trajectory.jsonl`` — one JSON record per sample (states, actions, labels).
* ``states.npz``       — the same numerical streams as stacked arrays.
* ``frames/``          — periodic ``rgb_XXXX.png`` and ``depth_XXXX.png`` (optional).
* ``dataset_meta.json``— shapes, rates, and field documentation.

This makes a single demo run double as a labelled imitation-learning dataset.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np


@dataclass
class DataRecorder:
    model: mujoco.MjModel
    data: mujoco.MjData
    out_dir: Path
    sample_hz: float = 50.0
    image_hz: float = 3.0
    camera: str = "cam_hero"
    save_images: bool = True
    img_w: int = 512
    img_h: int = 288

    _rows: list[dict] = field(default_factory=list, init=False)
    _np: dict[str, list] = field(default_factory=dict, init=False)
    _renderer: mujoco.Renderer | None = field(default=None, init=False)
    _depth: mujoco.Renderer | None = field(default=None, init=False)
    _last_sample: float = field(default=-1e9, init=False)
    _last_image: float = field(default=-1e9, init=False)
    _img_idx: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.out_dir = Path(self.out_dir)
        (self.out_dir / "frames").mkdir(parents=True, exist_ok=True)
        for k in ("t", "qpos", "qvel", "ctrl", "grip_forces", "ncon"):
            self._np[k] = []
        if self.save_images:
            self._renderer = mujoco.Renderer(self.model, self.img_h, self.img_w)
            self._depth = mujoco.Renderer(self.model, self.img_h, self.img_w)
            self._depth.enable_depth_rendering()

    def maybe_record(self, task: str, kind: str, phase: str, grip_forces: np.ndarray) -> None:
        t = float(self.data.time)
        if t - self._last_sample < 1.0 / self.sample_hz:
            return
        self._last_sample = t
        row = {
            "t": round(t, 4),
            "task": task, "kind": kind, "phase": phase,
            "qpos": [round(float(x), 5) for x in self.data.qpos],
            "ctrl": [round(float(x), 5) for x in self.data.ctrl],
            "grip_forces": [round(float(x), 4) for x in grip_forces],
            "ncon": int(self.data.ncon),
        }
        self._rows.append(row)
        self._np["t"].append(t)
        self._np["qpos"].append(self.data.qpos.copy())
        self._np["qvel"].append(self.data.qvel.copy())
        self._np["ctrl"].append(self.data.ctrl.copy())
        self._np["grip_forces"].append(np.asarray(grip_forces, dtype=float))
        self._np["ncon"].append(self.data.ncon)

        if self.save_images and t - self._last_image >= 1.0 / self.image_hz:
            self._last_image = t
            self._renderer.update_scene(self.data, camera=self.camera)
            rgb = self._renderer.render()
            self._depth.update_scene(self.data, camera=self.camera)
            depth = self._depth.render()
            import imageio.v3 as iio
            iio.imwrite(self.out_dir / "frames" / f"rgb_{self._img_idx:04d}.png", rgb)
            dnorm = np.clip(depth / (depth.max() + 1e-6), 0, 1)
            iio.imwrite(self.out_dir / "frames" / f"depth_{self._img_idx:04d}.png",
                        (dnorm * 255).astype(np.uint8))
            self._img_idx += 1

    def finalize(self) -> dict:
        with open(self.out_dir / "trajectory.jsonl", "w", encoding="utf-8") as f:
            for row in self._rows:
                f.write(json.dumps(row) + "\n")
        arrays = {k: np.asarray(v) for k, v in self._np.items()}
        np.savez_compressed(self.out_dir / "states.npz", **arrays)
        meta = {
            "n_samples": len(self._rows),
            "sample_hz": self.sample_hz,
            "image_hz": self.image_hz if self.save_images else 0,
            "n_images": self._img_idx,
            "nq": int(self.model.nq), "nv": int(self.model.nv), "nu": int(self.model.nu),
            "fields": {
                "qpos": "generalized positions (gantry, hand, free objects)",
                "qvel": "generalized velocities",
                "ctrl": "commanded actuator targets (4 gantry + 16 hand)",
                "grip_forces": "fingertip touch-sensor forces [index, middle, ring, thumb] (N)",
                "ncon": "active contact count",
            },
        }
        (self.out_dir / "dataset_meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8")
        return meta
