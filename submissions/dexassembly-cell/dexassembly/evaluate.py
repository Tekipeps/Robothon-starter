"""Statistical robustness evaluation for the DexAssembly Cell.

Focuses on the **dexterous pick-and-place** — the part of the pipeline where
perception feedback matters — under **domain randomization** (part position
+/- 12 mm, mass +/- 15%).  For each seed and each part it runs a fresh,
isolated rollout in two modes:

* **adaptive** — the closed-loop controller senses the part's live position and
  retries on a failed grasp, and
* **baseline_open_loop** — aims at the fixed nominal position with no recovery.

It reports per-mode success rate (mean +/- std over seeds) and the ablation gap,
i.e. how much the closed-loop perception + recovery is worth.

Run:  python -m dexassembly.evaluate --rollouts 12
Output: outputs/evaluation.json + a printed summary.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .engine import Engine
from .scene import SceneConfig
from .tasks import sort_task

OUT = Path(__file__).resolve().parent.parent / "outputs"


def _grasp_rollout(seed: int, part_name: str, adaptive: bool) -> bool:
    """Run a single isolated pick-and-place of one randomized part."""
    rnd = SceneConfig.randomized(seed)
    nominal = SceneConfig.default()
    part = next(p for p in nominal.parts if p.name == part_name)   # nominal target
    bin_ = next(b for b in nominal.bins if b.name == part.bin_name)
    eng = Engine(rnd, max_retries=1 if adaptive else 0)
    rep = eng.run(tasks=[sort_task(part, bin_, adaptive=adaptive)])
    return bool(rep["tasks"][0]["success"])


def evaluate(n_rollouts: int = 12, seed0: int = 2000) -> dict:
    parts = ["part_red", "part_green", "part_blue"]
    modes = {"adaptive": True, "baseline_open_loop": False}
    rates: dict[str, list[float]] = {m: [] for m in modes}
    per_part: dict[str, dict[str, list[bool]]] = {m: {p: [] for p in parts} for m in modes}

    for i in range(n_rollouts):
        seed = seed0 + i
        for mode, adaptive in modes.items():
            successes = []
            for part_name in parts:
                ok = _grasp_rollout(seed, part_name, adaptive)
                successes.append(ok)
                per_part[mode][part_name].append(ok)
            rates[mode].append(float(np.mean(successes)))

    summary: dict = {
        "study": "dexterous pick-and-place under domain randomization",
        "randomization": {"position_mm": 12, "mass_pct": 15},
        "n_rollouts": n_rollouts, "n_grasps_per_mode": n_rollouts * len(parts),
        "seed0": seed0, "modes": {},
    }
    for mode in modes:
        r = np.array(rates[mode])
        summary["modes"][mode] = {
            "mean_success_rate": round(float(r.mean()), 4),
            "std_success_rate": round(float(r.std()), 4),
            "per_part_success_rate": {p: round(float(np.mean(v)), 3)
                                      for p, v in per_part[mode].items()},
        }
    a = summary["modes"]["adaptive"]["mean_success_rate"]
    b = summary["modes"]["baseline_open_loop"]["mean_success_rate"]
    summary["ablation_gain_closed_loop"] = round(a - b, 4)
    return summary


def _print(s: dict) -> None:
    print(f"\nDexterous pick-and-place robustness — {s['n_rollouts']} seeds × "
          f"3 parts = {s['n_grasps_per_mode']} grasps/mode, "
          f"±{s['randomization']['position_mm']} mm / ±{s['randomization']['mass_pct']}%\n")
    for mode, m in s["modes"].items():
        print(f"[{mode}]  {m['mean_success_rate'] * 100:5.1f}% ± {m['std_success_rate'] * 100:.1f}")
        for p, rate in m["per_part_success_rate"].items():
            print(f"      {p:12s} {rate * 100:5.1f}%")
    print(f"\nClosed-loop perception + recovery gain: "
          f"+{s['ablation_gain_closed_loop'] * 100:.1f} pp\n")


def main() -> int:
    p = argparse.ArgumentParser(description="Grasp robustness evaluation + ablation.")
    p.add_argument("--rollouts", type=int, default=12)
    p.add_argument("--seed0", type=int, default=2000)
    args = p.parse_args()
    summary = evaluate(args.rollouts, args.seed0)
    OUT.mkdir(exist_ok=True)
    (OUT / "evaluation.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _print(summary)
    print(f"Wrote {OUT / 'evaluation.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
