"""Interactive keyboard teleoperation of the DexAssembly Cell.

Opens the MuJoCo passive viewer and lets you drive the gantry in Cartesian space
and command the hand through its grasp presets in real time.  Requires a display
/ OpenGL (it will not run on a headless machine).

Controls
--------
    W / S      gantry +x / -x            Arrow Up / Down   gantry +y / -y
    R / F      gantry +z / -z            Q / E             wrist yaw - / +
    1          open hand                 2                 pre-grasp
    3          power grasp               4                 pinch
    Space      toggle open / grasp       Backspace         reset
    Esc        quit

Run with:  python -m dexassembly.teleop
"""

from __future__ import annotations

import time

import mujoco
import mujoco.viewer
import numpy as np

from .controllers import CellController, GantryTarget
from .scene import SceneConfig, compile_scene

STEP_XY = 0.01
STEP_Z = 0.01
STEP_YAW = 0.08


def main() -> int:
    cfg = SceneConfig.default()
    model, _ = compile_scene(cfg)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ctl = CellController(model, data)

    state = {"x": 0.0, "y": -0.1, "z": 0.18, "yaw": 0.0, "hand": "open", "grasped": False}
    ctl.set_gantry_target(GantryTarget(state["x"], state["y"], state["z"]))
    ctl.set_hand_pose(state["hand"])

    def apply() -> None:
        ctl.set_gantry_target(GantryTarget(state["x"], state["y"], state["z"], state["yaw"]))
        ctl.set_hand_pose(state["hand"])

    def key_cb(key: int) -> None:
        k = chr(key) if 32 <= key < 127 else ""
        if k in ("W", "w"):
            state["x"] += STEP_XY
        elif k in ("S", "s"):
            state["x"] -= STEP_XY
        elif key == 265:  # up
            state["y"] += STEP_XY
        elif key == 264:  # down
            state["y"] -= STEP_XY
        elif k in ("R", "r"):
            state["z"] += STEP_Z
        elif k in ("F", "f"):
            state["z"] -= STEP_Z
        elif k in ("Q", "q"):
            state["yaw"] -= STEP_YAW
        elif k in ("E", "e"):
            state["yaw"] += STEP_YAW
        elif k == "1":
            state["hand"] = "open"
        elif k == "2":
            state["hand"] = "pregrasp"
        elif k == "3":
            state["hand"] = "grasp"
        elif k == "4":
            state["hand"] = "pinch"
        elif key == 32:  # space
            state["grasped"] = not state["grasped"]
            state["hand"] = "grasp" if state["grasped"] else "open"
        elif key == 259:  # backspace
            mujoco.mj_resetData(model, data)
            state.update(x=0.0, y=-0.1, z=0.18, yaw=0.0, hand="open", grasped=False)
        apply()

    print(__doc__)
    with mujoco.viewer.launch_passive(model, data, key_callback=key_cb) as viewer:
        while viewer.is_running():
            step_start = time.time()
            mujoco.mj_step(model, data)
            viewer.sync()
            dt = model.opt.timestep - (time.time() - step_start)
            if dt > 0:
                time.sleep(dt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
