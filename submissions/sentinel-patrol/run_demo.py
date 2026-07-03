"""Sentinel — autonomous quadruped inspection patrol: one-command demo runner.

Runs the scored patrol mission and renders a narrated HUD video (chase camera, live
objective checklist, foot-contact dots, route mini-map, and an onboard head-camera
picture-in-picture during each inspection), then writes ``report.json``, the
full-rate telemetry dataset (``outputs/telemetry.npz``), and a labelled head-cam
snapshot per panel scan.  This single command reproduces every artifact in the
submission.

    python run_demo.py            # full demo video + report.json + telemetry
    python run_demo.py --quick    # fast low-res smoke run
    python run_demo.py --help
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio
import mujoco

from sentinel.datalog import TelemetryLog
from sentinel.engine import Mission, StepInfo
from sentinel.hud import draw_hud, draw_pip, scorecard, title_card
from sentinel.scene import CourseConfig

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
TITLE = "SENTINEL"
BADGE = "REAL PHYSICS · SENSOR-DRIVEN"

# Narration caption per mission phase / current target (drawn at the bottom of frame).
CAPTIONS = {
    "approach": "Patrol leg 1 — trotting the route under closed-loop heading control.",
    "crest_berm": "Crossing the ramp berm — the trot keeps the trunk level over the incline.",
    "rubble": "Rubble field — the IMU reflex auto-crouches over the rough ground.",
    "station_b": "Approaching inspection station B.",
    "finish": "Final leg — heading for the finish pad.",
}
INSPECT_CAP = "INSPECTION — pivoting in place to face the panel and scanning (head cam, inset)."
SHOVE_CAP = "DISTURBANCE — 100 N lateral shove; the IMU feels it in 8 ms and braces."


def run(args: argparse.Namespace) -> dict:
    OUT.mkdir(exist_ok=True)
    cfg = CourseConfig(offwidth=max(1280, args.width), offheight=max(720, args.height))
    mission = Mission(cfg, seed=args.seed)
    renderer = mujoco.Renderer(mission.model, args.height, args.width)

    route_xy = [wp.xy for wp in cfg.waypoints]
    writer = imageio.get_writer(str(OUT / args.output), fps=args.fps,
                                codec="libx264", macro_block_size=None)

    intro = title_card(
        args.width, args.height,
        title="SENTINEL",
        subtitle="Autonomous Quadruped Inspection Patrol",
        bullets=[
            "Unitree Go1 (12 position actuators) — a hand-built trot gait, ctrl-only",
            "Onboard IMU + foot touch sensors: an 8 ms proprioceptive brace reflex",
            "The reflex stretches shove recovery from 70 N (passive) to 100 N",
            "Closed-loop waypoint navigation across a ramp berm + rubble field",
            "Full 500 Hz state-action-sensor dataset logged every run",
        ],
        footer="Robothon 2026 · Faraday Future MuJoCo Hackathon",
    )
    for _ in range(int(args.fps * 2.5)):
        writer.append_data(intro)

    step_every = max(1, int(round((1.0 / mission.model.opt.timestep) / args.fps)))
    state = {"n": 0, "last": None, "scanned": set()}
    telemetry = TelemetryLog(mission.model)

    def on_step(info: StepInfo) -> None:
        telemetry.record(mission.data, t=info.t, phase=info.phase,
                         forward=info.forward, yaw_cmd=info.yaw_cmd, brace=info.brace)

        # labelled head-cam snapshot the moment each panel scan completes
        for panel in ("panel_A", "panel_B"):
            if info.objectives.get(f"inspect_{panel}") and panel not in state["scanned"]:
                state["scanned"].add(panel)
                renderer.update_scene(mission.data, camera="head")
                imageio.imwrite(str(OUT / f"scan_{panel}.png"), renderer.render())

        state["n"] += 1
        if state["n"] % step_every:
            return
        renderer.update_scene(mission.data, camera="chase")
        frame = renderer.render()
        cap = (SHOVE_CAP if info.phase == "SHOVE"
               else INSPECT_CAP if info.phase == "INSPECT"
               else CAPTIONS.get(info.target, ""))
        frame = draw_hud(
            frame, title=TITLE, badge=BADGE, phase=info.phase,
            forward=info.forward, yaw=info.yaw_cmd, objectives=info.objectives,
            progress=info.progress, route_xy=route_xy, robot_xy=info.pos, caption=cap,
            feet=info.feet, brace=info.brace,
        )
        if info.phase == "INSPECT":
            renderer.update_scene(mission.data, camera="head")
            frame = draw_pip(frame, renderer.render(), label="head cam · scanning")
        writer.append_data(frame)
        state["last"] = frame

    report = mission.run(on_step=on_step)
    report["telemetry"] = telemetry.save(OUT / "telemetry.npz",
                                         OUT / "telemetry_summary.json")
    report["scan_snapshots"] = sorted(f"outputs/scan_{p}.png" for p in state["scanned"])

    if state["last"] is not None:
        for _ in range(args.fps // 2):
            writer.append_data(state["last"])

    lines = [(k, "ok" if v else "—", bool(v)) for k, v in report["objectives"].items()]
    card = scorecard(
        args.width, args.height, title="Patrol scorecard",
        lines=lines,
        headline=f"{report['score_0_100']:.0f}/100   "
                 f"({report['n_success']}/{report['n_objectives']} objectives)",
        footer="Reproduce: python run_demo.py  ·  every step is real actuated contact",
    )
    for _ in range(int(args.fps * 3.5)):
        writer.append_data(card)
    writer.close()

    report["video"] = f"outputs/{args.output}"
    (ROOT / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the Sentinel quadruped patrol demo.")
    p.add_argument("--output", default="sentinel_demo.mp4")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--quick", action="store_true", help="fast low-res smoke run")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.quick:
        args.width, args.height, args.fps = 640, 360, 20
    report = run(args)
    print(json.dumps({k: v for k, v in report.items() if k != "objectives"}, indent=2))
    print(f"\nScore: {report['score_0_100']:.0f}/100  "
          f"({report['n_success']}/{report['n_objectives']})  video: {report['video']}")
    for k, v in report["objectives"].items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
