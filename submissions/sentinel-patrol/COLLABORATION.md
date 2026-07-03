# How this was built — human + AI collaboration log

Sentinel was built by a human (Tekena Solomon) directing an AI coding agent. This is
an honest account of *how the work was divided and how we caught each other's
mistakes* — the thing that actually makes a human-AI build good rather than just
fast.

## The pivotal decision: change the embodiment, not the polish

The human's goal was a top-scoring entry. An earlier direction had reached for **fine
in-hand manipulation** on a fixed dexterous-hand rig. The AI built and stress-tested
it and reported back a hard, honest finding: that embodiment's top-down power grasp
**physically cannot** do reliable fine manipulation (no wrist-torque coupling, chaotic
grip force, asymmetric release) — exactly the work the leading entries showcase. The
right move wasn't more tuning; it was a **different robot**.

So we scrapped that and chose the **locomotion lane** with a legged robot — a domain
where (a) the physics is reliable enough to land a clean, deterministic demo, and (b)
the result is visually distinct from the manipulation-heavy field. The human picked
the concept from a short menu of AI-proposed directions; the AI built it.

## How we de-risked it (riskiest thing first)

Rather than build the whole system and hope it walked, we proved the **hard part
first**, one rung at a time, and only moved up when the rung was solid:

1. **Stand** — load the vendored Go1, hold the home keyframe → confirmed it stands
   rock-steady (trunk holds 0.265 m, upright).
2. **Trot** — foot-trajectory gait + planar 2-link IK → walked 0.65 m in 4 s, upright,
   *on the first run* (the IK was verified against the model's home pose before use).
3. **Steer** — additive per-side stride for skid-steer turning, including pivot-in-place;
   fixed an inverted turn sign caught by a left/right test.
4. **Terrain** — grew a ramp berm + rubble field; the open-loop gait drifted off and
   fell, which is exactly why we added **closed-loop heading control** — then it crossed
   everything and reached all five waypoints upright.
5. **Mission** — state machine for turn-and-scan inspections + a scripted shove. The
   first shove (85 N) toppled the robot; an empirical sweep found **70 N** staggers it
   ~11 cm but recovers — so the disturbance test is honest *and* survivable.
6. **Senses, then a reflex** (the v1.1 upgrade) — real MuJoCo sensors (IMU triad + 4
   foot touch) were compiled into the model, and a disturbance reflex built on them.
   This one took *four measured iterations* to get right:
   - The obvious trigger (lateral **velocity**) was too slow — ~120 ms into a 150 ms
     shove. A 25 ms-filtered **acceleration** channel detects in 8 ms because the
     trot's own foot-impact spikes filter away while a real hit is sustained.
   - An oracle-timed A/B across brace strategies showed **crouch-only wins**;
     stance-widening — the intuitive choice — actually topples the robot mid-push,
     and stop-and-brace removes the stepping recovery that catches the body.
   - A latch bug (the trigger kept re-arming while the signal stayed high, pinning
     the brace at zero *during the hit*) was found by comparing oracle vs. live runs.
   - Pivot-in-place false-triggered the reflex, so it is suppressed while the
     controller commands an aggressive turn (**reafference gating**).

   Net result, all from printed sweeps: the survivable shove went from **70 N passive
   to 100 N braced** (mid-plateau of 90–110 N; 120 N still topples).

Each rung was a runnable check the human could see, not a claim.

## Who did what

- **Human:** set the goal, chose the locomotion concept, and pushed back when an
  earlier approach was being over-polished instead of rethought.
- **AI:** researched the rubric and the field, proposed concepts, wrote all the code,
  ran the physics experiments, reported failures plainly (the toppling shove, the
  off-course drift, the unreachable manipulation), and tuned to the measured envelope.

## Honesty guardrails we held

- **No qpos teleportation.** The robot only ever moves through `data.ctrl` and one
  brief external shove force; every result is real contact dynamics.
- **Measured, not asserted.** Objective thresholds, the shove magnitude, and the gait
  parameters are all backed by printed experiments, not hand-waving.
- **Reproducible.** Deterministic seed, vendored model, `validate.py` self-check — the
  committed `report.json` reproduces exactly with `python run_demo.py`.
