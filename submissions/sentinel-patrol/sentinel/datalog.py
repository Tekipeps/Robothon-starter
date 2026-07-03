"""Full-rate telemetry logging: the patrol doubles as a data-collection pipeline.

Every physics step of the mission is recorded — simulation state (``qpos`` /
``qvel``), the controller's actions (``ctrl``, the high-level ``forward`` /
``yaw`` command, the reflex brace level), the complete onboard sensor stream
(IMU gyro / accelerometer / velocimeter + four foot touch sensors), and the
mission phase — and saved as a compressed ``.npz`` with a self-describing
``schema`` field, plus a small human-readable JSON summary.

That makes each run a labelled state-action-observation dataset at the full
500 Hz control rate, suitable for imitation-learning or gait-analysis work:

    import numpy as np
    log = np.load("outputs/telemetry.npz", allow_pickle=True)
    print(dict(zip(log["schema"].item()["fields"], [log[f].shape for f in
          log["schema"].item()["fields"]])))
"""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

PHASES = ("PATROL", "INSPECT", "SHOVE")


class TelemetryLog:
    """Accumulates one row per physics step; ``save()`` writes npz + summary."""

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        self.rows: dict[str, list] = {k: [] for k in (
            "t", "qpos", "qvel", "ctrl", "sensordata",
            "phase", "forward", "yaw_cmd", "brace")}
        # named sensor layout so the npz is self-describing
        self.sensor_layout = {
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i):
                [int(model.sensor_adr[i]), int(model.sensor_dim[i])]
            for i in range(model.nsensor)
        }

    def record(self, data: mujoco.MjData, *, t: float, phase: str,
               forward: float, yaw_cmd: float, brace: float) -> None:
        r = self.rows
        r["t"].append(t)
        r["qpos"].append(data.qpos.copy())
        r["qvel"].append(data.qvel.copy())
        r["ctrl"].append(data.ctrl.copy())
        r["sensordata"].append(data.sensordata.copy())
        r["phase"].append(PHASES.index(phase) if phase in PHASES else -1)
        r["forward"].append(forward)
        r["yaw_cmd"].append(yaw_cmd)
        r["brace"].append(brace)

    def save(self, npz_path: Path, summary_path: Path | None = None) -> dict:
        arrays = {k: np.asarray(v) for k, v in self.rows.items()}
        schema = {
            "fields": list(arrays),
            "rate_hz": round(1.0 / self.model.opt.timestep, 1),
            "phases": list(PHASES),
            "sensors": self.sensor_layout,
            "note": "one row per physics step; sensordata columns per 'sensors'",
        }
        np.savez_compressed(npz_path, schema=np.asarray(schema, dtype=object), **arrays)

        n = len(arrays["t"])
        summary = {
            "steps": n,
            "duration_s": round(float(arrays["t"][-1]), 2) if n else 0.0,
            "rate_hz": schema["rate_hz"],
            "fields": {k: list(arrays[k].shape) for k in arrays},
            "brace_active_frac": round(float((arrays["brace"] > 0.01).mean()), 4) if n else 0.0,
            "npz": npz_path.name,
        }
        if summary_path is not None:
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary
