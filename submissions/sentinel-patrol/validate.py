"""Self-validation for the Sentinel quadruped-patrol submission.

Checks required files, a valid registration UUID, that the procedural course
compiles, and that the expected Go1 structure (free trunk, 12 leg actuators, home
keyframe, onboard + chase + overview cameras) and the patrol props are present.
Exits non-zero if any check fails.

    python validate.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")

checks: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    checks.append((name, bool(ok), detail))


def main() -> int:
    for rel in ("README.md", "ARCHITECTURE.md", "COLLABORATION.md",
                "LICENSE-THIRDPARTY.md", "requirements.txt", "run_demo.py",
                "registration.json", "report.json", "demo.mp4",
                "sentinel/__init__.py", "sentinel/scene.py", "sentinel/gait.py",
                "sentinel/controller.py", "sentinel/engine.py", "sentinel/hud.py",
                "sentinel/proprio.py", "sentinel/datalog.py",
                "assets/unitree_go1/go1.xml", "assets/unitree_go1/LICENSE"):
        check(f"file present: {rel}", (ROOT / rel).exists())

    try:
        reg = json.loads((ROOT / "registration.json").read_text(encoding="utf-8"))
        check("registration UUID valid", bool(UUID_RE.match(reg.get("uuid", ""))),
              reg.get("uuid", ""))
        check("registration has project_name", bool(reg.get("project_name")))
    except Exception as exc:  # noqa: BLE001
        check("registration.json parses", False, str(exc))

    try:
        import mujoco
        from sentinel.scene import compile_scene
        model, _ = compile_scene()
        check("course compiles", True,
              f"nq={model.nq} nu={model.nu} ngeom={model.ngeom} ncam={model.ncam}")
        check("12 leg position actuators", model.nu == 12, str(model.nu))

        def has(kind, name):
            return mujoco.mj_name2id(model, kind, name) >= 0

        for j in ("FR_thigh_joint", "FL_calf_joint", "RR_hip_joint"):
            check(f"joint present: {j}", has(mujoco.mjtObj.mjOBJ_JOINT, j))
        for c in ("head", "chase", "overview"):
            check(f"camera present: {c}", has(mujoco.mjtObj.mjOBJ_CAMERA, c))
        check("home keyframe present", has(mujoco.mjtObj.mjOBJ_KEY, "home"))
        for s in ("panel_A_target", "panel_B_target"):
            check(f"inspection target: {s}", has(mujoco.mjtObj.mjOBJ_SITE, s))
        check("finish pad present", has(mujoco.mjtObj.mjOBJ_GEOM, "finish_pad"))
        for s in ("imu_gyro", "imu_accel", "imu_vel",
                  "touch_FR", "touch_FL", "touch_RR", "touch_RL"):
            check(f"sensor present: {s}", has(mujoco.mjtObj.mjOBJ_SENSOR, s))
    except Exception as exc:  # noqa: BLE001
        check("course compiles", False, str(exc))

    n_pass = sum(ok for _, ok, _ in checks)
    print(f"Sentinel — submission validation ({n_pass}/{len(checks)} passed)\n")
    for name, ok, detail in checks:
        extra = f"  [{detail}]" if detail else ""
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}{extra}")
    failed = len(checks) - n_pass
    if failed:
        print(f"\n{failed} check(s) FAILED")
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
