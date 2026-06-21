"""Self-validation for the DexAssembly Cell submission.

Runs a checklist over the package: required files, registration UUID, vendored
assets, that the MuJoCo model compiles, and that the expected actuators/sensors/
cameras and key bodies exist. Exits non-zero if any check fails.

Run:  python validate.py
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


def _exists(rel: str) -> None:
    check(f"file present: {rel}", (ROOT / rel).exists())


def main() -> int:
    for rel in ("README.md", "EVALUATION_GUIDE.md", "ARCHITECTURE.md",
                "LICENSE-THIRDPARTY.md", "requirements.txt", "run_demo.py",
                "scene.xml", "registration.json", "demo.mp4",
                "report.json", "evaluation.json",
                "dexassembly/scene.py", "dexassembly/controllers.py",
                "dexassembly/tasks.py", "dexassembly/engine.py",
                "dexassembly/evaluate.py", "dexassembly/record.py",
                "dexassembly/hud.py", "dexassembly/teleop.py",
                "assets/leap_hand/right_hand.xml", "assets/leap_hand/LICENSE"):
        _exists(rel)

    # registration
    try:
        reg = json.loads((ROOT / "registration.json").read_text(encoding="utf-8"))
        uuid = reg.get("uuid", "")
        check("registration.json has valid UUID", bool(UUID_RE.match(uuid)), uuid)
        check("registration has project_name", bool(reg.get("project_name")))
    except Exception as exc:  # noqa: BLE001
        check("registration.json parses", False, str(exc))

    # model compiles + structure
    try:
        import mujoco  # noqa: WPS433
        from dexassembly.scene import compile_scene
        model, _ = compile_scene()
        check("MuJoCo model compiles", True, f"nq={model.nq} nu={model.nu} nsensor={model.nsensor}")
        check("20 actuators (4 gantry + 16 hand)", model.nu == 20, str(model.nu))
        check("4 cameras", model.ncam == 4, str(model.ncam))

        def has(kind, name):
            return mujoco.mj_name2id(model, kind, name) >= 0

        for s in ("wrist_force", "wrist_torque", "button_press"):
            check(f"sensor present: {s}", has(mujoco.mjtObj.mjOBJ_SENSOR, s))
        for b in ("lh_palm", "peg", "cable_seg_5", "part_red"):
            check(f"body present: {b}", has(mujoco.mjtObj.mjOBJ_BODY, b))
        n_touch = sum(has(mujoco.mjtObj.mjOBJ_SENSOR, f"lh_{f}_ds_force")
                      for f in ("if", "mf", "rf", "th"))
        check("4 fingertip touch sensors", n_touch == 4, str(n_touch))
    except Exception as exc:  # noqa: BLE001
        check("MuJoCo model compiles", False, str(exc))

    n_pass = sum(ok for _, ok, _ in checks)
    print(f"DexAssembly Cell — submission validation ({n_pass}/{len(checks)} passed)\n")
    for name, ok, detail in checks:
        mark = "PASS" if ok else "FAIL"
        extra = f"  [{detail}]" if detail else ""
        print(f"  [{mark}] {name}{extra}")
    failed = len(checks) - n_pass
    if failed:
        print(f"\n{failed} check(s) FAILED")
        return 1
    print("\nAll checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
