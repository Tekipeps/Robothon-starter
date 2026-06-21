"""DexAssembly Cell -- autonomous demo runner.

Runs the full closed-loop arena, renders a multi-shot HUD demo video, records a
labelled trajectory dataset, and writes a JSON scorecard.  This single command
reproduces every artifact referenced in the submission.

Usage:
    python run_demo.py                 # full demo: video + data + report
    python run_demo.py --quick         # short, low-res smoke run
    python run_demo.py --no-data       # skip dataset recording
    python run_demo.py --help
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import imageio  # v2 API for streaming the video writer (flat memory)
import mujoco
import numpy as np

from dexassembly.engine import Engine, StepInfo
from dexassembly.hud import draw_hud
from dexassembly.record import DataRecorder
from dexassembly.scene import SceneConfig
from dexassembly.tasks import default_arena

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"

# Which camera each task kind is filmed from, for a multi-shot edit.
CAM_BY_KIND = {
    "sort": "cam_hero",
    "inspect_sort": "cam_side",
    "button": "cam_side",
    "peg": "cam_hero",
}
GRASP_PHASES = {"grasp", "press", "descend", "inspect_cw", "inspect_ccw", "insert"}

# Narration beat per task (drawn as a caption + exported to an SRT subtitle track).
NARRATION = {
    "part_red": "Autonomous color-sort: the 16-DOF LEAP hand grasps the red part "
                "with tactile feedback and places it in the matching bin.",
    "part_green": "Closed-loop perception re-targets to each part's live position; "
                  "the grasp closes until the fingertip touch sensors register contact.",
    "inspect_sort_part_blue": "Eye-in-hand inspection: the part is raised to the wrist "
                              "camera and held steady, then sorted into its bin.",
    "inspect_button": "A single extended finger presses the spring-loaded inspection "
                      "button, confirmed by its displacement sensor.",
    "cable_inspect": "Force-aware inspection: the hand elastically deflects the "
                     "deformable connector cable while the wrist F/T sensor monitors load.",
    "peg_insert": "Peg-in-hole assembly: the connector cube is grasped, carried, and "
                  "seated into the wide-mouth assembly socket.",
}


def _to_srt(beats: list[tuple[float, float, str]]) -> str:
    def ts(t: float) -> str:
        h, rem = divmod(t, 3600)
        m, s = divmod(rem, 60)
        ms = int((s - int(s)) * 1000)
        return f"{int(h):02d}:{int(m):02d}:{int(s):02d},{ms:03d}"
    return "\n".join(
        f"{i}\n{ts(a)} --> {ts(b)}\n{txt}\n"
        for i, (a, b, txt) in enumerate(beats, 1)
    )


def run(args: argparse.Namespace) -> dict:
    OUT.mkdir(exist_ok=True)
    cfg = SceneConfig.default()
    cfg.offwidth = max(cfg.offwidth, args.width)
    cfg.offheight = max(cfg.offheight, args.height)
    eng = Engine(cfg, seed=args.seed)
    tasks = default_arena(cfg)

    renderer = mujoco.Renderer(eng.model, args.height, args.width)
    recorder = (
        DataRecorder(eng.model, eng.data, OUT / "dataset", save_images=not args.no_images)
        if not args.no_data else None
    )

    video_path = OUT / args.output
    writer = imageio.get_writer(str(video_path), fps=args.fps, codec="libx264",
                                macro_block_size=None)
    last_frame = {"img": None}
    step_every = max(1, int(round((1.0 / eng.model.opt.timestep) / args.fps)))
    counter = {"n": 0}
    video_t = {"t": 0.0}
    beats: list[tuple[float, float, str]] = []  # (start, end, caption) in video time

    def on_step(info: StepInfo) -> None:
        if recorder is not None:
            recorder.maybe_record(info.task, info.kind, info.phase, info.forces)
        counter["n"] += 1
        if counter["n"] % step_every:
            return
        cam = CAM_BY_KIND.get(info.kind, "cam_hero")
        if info.phase in GRASP_PHASES and info.kind in ("sort", "peg"):
            cam = "cam_side"
        caption = NARRATION.get(info.task, "")
        # accumulate subtitle beats keyed on video time
        vt = video_t["t"]
        if not beats or beats[-1][2] != caption:
            if beats:
                beats[-1] = (beats[-1][0], vt, beats[-1][2])
            beats.append((vt, vt + 1.0, caption))
        video_t["t"] += 1.0 / args.fps

        renderer.update_scene(eng.data, camera=cam)
        frame = renderer.render()
        frame = draw_hud(
            frame,
            title="DexAssembly Cell -- LEAP-hand micro-assembly",
            task=info.task, phase=info.phase,
            forces=info.forces, grip=info.grip, progress=info.progress,
            score_line=f"arena {int(info.progress * 100):3d}%",
            caption=caption,
        )
        writer.append_data(frame)
        last_frame["img"] = frame

    report = eng.run(tasks=tasks, on_step=on_step)
    if beats:
        beats[-1] = (beats[-1][0], video_t["t"], beats[-1][2])
        (OUT / "narration.srt").write_text(
            _to_srt([b for b in beats if b[2]]), encoding="utf-8")
        report["narration_srt"] = str(OUT / "narration.srt")

    # outro: hold the final scored result for ~1 s, then close the stream
    if last_frame["img"] is not None:
        for _ in range(args.fps):
            writer.append_data(last_frame["img"])
    writer.close()
    report["video"] = str(video_path)
    if recorder is not None:
        report["dataset"] = recorder.finalize()

    (OUT / "report.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run the DexAssembly Cell demo.")
    p.add_argument("--output", default="dexassembly_demo.mp4", help="video filename (in outputs/)")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-data", action="store_true", help="skip dataset recording")
    p.add_argument("--no-images", action="store_true", help="record states but not frames")
    p.add_argument("--quick", action="store_true", help="fast low-res smoke run")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.quick:
        args.width, args.height, args.fps, args.no_images = 640, 360, 20, True
    report = run(args)
    summary = {k: v for k, v in report.items() if k != "tasks"}
    print(json.dumps(summary, indent=2, default=str))
    print(f"\nScore: {report['score_0_100']:.1f}/100  "
          f"({report['n_success']}/{report['n_tasks']} tasks)  "
          f"video: {report['video']}")
    for t in report["tasks"]:
        mark = "PASS" if t["success"] else "FAIL"
        print(f"  [{mark}] {t['name']:24s} {t['detail']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
