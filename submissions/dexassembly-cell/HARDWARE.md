# Sim-to-Real Channel Mapping

This document maps every simulated channel of the DexAssembly Cell onto a concrete
real-world counterpart, and — just as importantly — states plainly **what is
idealized and would not transfer as-is**. The point is not to claim this was run on
hardware (it was not). The point is that the control interface was designed around
channels that *exist on real hardware*, so the same FSM and controller could drive a
physical cell with the changes listed here and nothing more exotic.

The whole system already talks to the robot through `data.ctrl` (position commands)
and named sensors only — never by writing `qpos`. That is precisely the interface a
real robot exposes, which is what makes this mapping short.

---

## Actuators → real hardware

| Sim channel (`data.ctrl`) | Count | Real-world counterpart |
|---|---|---|
| LEAP hand finger servos | 16 | **LEAP Hand** (MIT) — 16× Dynamixel **XC330-M288-T** smart servos, position mode, commanded over a U2D2 / serial bus via the published `leap_hand_utils` stack. This is a real, buildable open-hardware hand; the MJCF we vendor is its official Menagerie model. |
| Gantry X / Y / Z prismatic slides | 3 | Three stacked **linear stages** (e.g. ball-screw or belt actuators with stepper/servo drives and position feedback). A Cartesian portal is deliberately chosen over an arm for singularity-free, repeatable reach. |
| Wrist-yaw hinge | 1 | One **rotary stage** (or a servo) at the wrist plate. |

The controller commands all 20 actuators as position targets in engineering units
(rad for joints, m for slides). The affine gantry calibration in
[`controllers.py`](dexassembly/controllers.py) maps a desired world grasp-centre to
slide setpoints — identical math on a real portal, with the calibration constant
re-measured once on the physical rig.

## Sensors → real hardware

| Sim sensor | Dim | Real-world counterpart | Transfers cleanly? |
|---|---|---|---|
| Hand joint encoders | 16 | Dynamixel present-position registers (native on the XC330). | ✅ Native. |
| 6-axis wrist force/torque | 6 | An inline F/T sensor at the wrist plate — e.g. **ATI Mini45** or **Bota Systems** SenseOne. | ✅ Standard, but needs the ~9 N hand-weight tare we already document. |
| Eye-in-hand wrist camera | 1 | A wrist-mounted depth camera — **Intel RealSense D405** is the usual close-range choice. | ✅ Common eye-in-hand setup. |
| Hero / top / side cameras | 3 | Fixed RGB cameras on the cell frame. | ✅ Trivial. |
| Fingertip touch sensors | 4 | **Not standard on a stock LEAP Hand** — would require add-on fingertip tactile pads (e.g. FSRs, or a DIGIT/GelSight-class sensor). | ⚠️ **Idealized.** See below. |
| Button / probe-switch displacement | 2 | These are *scene-prop* sensors (the device-under-test reports its own state), not robot channels — on real hardware the press is confirmed by the wrist F/T spike and the part camera instead. | ⚠️ Prop-side. |

## What is idealized (and how we'd close the gap)

Honesty first — three things would need real work before this ran on metal:

1. **Fingertip touch sensing.** The contact-triggered grasp logic assumes per-finger
   normal-force readings. A stock LEAP Hand has none. On hardware you would either add
   fingertip tactile sensors, or fall back to **motor-current-based contact detection**
   (the Dynamixels expose present-current), which the FSM could consume in place of the
   touch channel with a re-tuned threshold.
2. **Grip-force control is open-loop.** As documented in the README's limitations, the
   grasp is a tuned fixed-position power close; the resulting grip force is not
   regulated. A physical deployment would want current-based force control on the
   finger servos before handling fragile or varied parts.
3. **Contact stiffness / friction are sim values.** The compliant-contact tuning that
   fixed the catapulting grasp is calibrated in MuJoCo units; real materials would need
   their own friction and compliance identification (a planned randomization axis).

## Bill of materials (indicative)

A faithful physical rebuild is roughly: 1× LEAP Hand kit, 3× linear stages + 1 rotary
stage with drives, 1× wrist F/T sensor, 1× wrist RealSense + 3× frame cameras, a host
PC, and the device-under-test fixtures (spring button, probe holster, harness cable,
colour bins). Everything on the robot side is off-the-shelf; nothing is custom-machined.

---

*This mapping is a design artifact, not a validated hardware result. It exists to show
the sim was built against a real control interface — position-command actuators and
named sensors — rather than against simulator-only shortcuts.*
