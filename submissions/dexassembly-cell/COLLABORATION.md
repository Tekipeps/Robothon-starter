# How this was built — a Human-AI collaboration log

This submission was produced by a human engineer (project direction, judgement,
hardware/physics intuition, and final review) working with **Claude (Anthropic)**
as a pair-programmer (implementation, debugging, and iteration). We're documenting
the process honestly because *how* a human and an AI divide the work — and catch
each other's mistakes — is itself part of the result.

## Who did what

**Human (Tekena Solomon) — direction & judgement**
- Set the goal and the bar: build a MuJoCo dexterous-manipulation cell strong
  enough to place on the contest leaderboard, and decide when it was good enough.
- Steered the scenario, caught issues by *watching the rendered video* that no
  metric flagged, and made the calls the AI couldn't: scope, framing, what to ship.
- Owned the environment (Python venv, dependency install) and every commit.

**Claude — implementation & iteration**
- Wrote the procedural `MjSpec` scene, the Cartesian-gantry controller, the
  closed-loop task FSM, the engine, the evaluation/ablation harness, the HUD, the
  recorder, and the tests.
- Ran experiments (grasp-pose sweeps, contact-parameter studies) and reported
  results back for the human to judge.

## Bugs we found *together* (the interesting part)

The collaboration was most valuable where a metric said "pass" but a human eye said
"that's wrong" — and vice-versa:

1. **The catapulting grasp.** The deterministic arena scored 100%, but a
   domain-randomized study (AI's idea, human's call to run it) revealed grasps
   *flinging* slightly off-centre parts ~33% of the time. Root cause: over-stiff
   contacts storing energy and releasing it like a spring. Fix: compliant contact
   parameters (`solimp`/`solref`). Only a randomized study surfaced it.
2. **The "gray circle."** The human watched the video and reported a gray disc
   covering the screen before each pickup. The AI traced it to a wrist camera that
   was *aimed up into the palm* — so it rendered the back of the hand. We turned the
   bug into a feature: re-aimed it as a proper side-offset **eye-in-hand camera**
   and added it as a live picture-in-picture during inspection.
3. **The "rubbery peg."** The human noticed the peg behaving like rubber and being
   ejected from a deep socket. The AI diagnosed three compounding causes (a tall peg
   pivoting in-grip; a narrow socket jamming the wide hand and ejecting chaotically;
   loose in-hand rotation) and fixed all three — a shorter cube connector, a
   wide-mouth receptacle, and a steady raise-and-hold inspection.

## What we learned

- **Metrics and human eyes catch different failures.** The ablation caught the
  catapult; the human caught the camera and the rubbery peg. Neither alone was
  enough.
- **Honesty constraints make better physics.** We refused to teleport joints via
  `qpos`. That made early grasps fail loudly — which forced real fixes (grasp pose,
  tactile-triggered closing, recovery) instead of hiding the problem.
- **Tight feedback loops win.** Cheap low-res smoke renders to *look* at the result
  before committing to an expensive full render saved hours.

*Every line of code here was written in this collaboration and is reproducible from
one command; nothing is copied from a prior solution.*
