"""Procedural course builder for the Sentinel quadruped patrol.

Loads the vendored Unitree Go1 (``assets/unitree_go1/go1.xml``) via :class:`MjSpec`
and grows an instrumented **inspection-patrol course** around it: a textured floor,
a gentle ramp-over-berm, an uneven "rubble" field of low blocks, two inspection
stations (a post + a target panel the robot scans with its head camera), a finish
pad, plus an onboard head camera and chase / overview cameras for filming.

The course is laid out along +x; the robot starts at the origin facing +x and
patrols a list of waypoints.  Everything is generated from one builder so the whole
cell is reproducible with no runtime downloads — the Go1 MJCF is committed under
``assets/`` (BSD-3, MuJoCo Menagerie).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
GO1_XML = ROOT / "assets" / "unitree_go1" / "go1.xml"


@dataclass
class Waypoint:
    name: str
    xy: tuple[float, float]
    radius: float = 0.30          # "reached" tolerance


@dataclass
class Inspection:
    name: str
    post_xy: tuple[float, float]  # where the marker post stands
    rgba: tuple[float, float, float, float]


@dataclass
class CourseConfig:
    """Layout of the patrol course (all positions in metres)."""

    offwidth: int = 1280
    offheight: int = 720

    # Ordered patrol waypoints the robot drives through.
    waypoints: list[Waypoint] = field(default_factory=lambda: [
        Waypoint("approach",   (1.15, 0.00)),
        Waypoint("crest_berm", (2.05, 0.00), radius=0.32),
        Waypoint("rubble",     (3.15, 0.00), radius=0.34),
        Waypoint("station_b",  (3.95, 0.00)),
        Waypoint("finish",     (4.70, 0.00), radius=0.28),
    ])
    inspections: list[Inspection] = field(default_factory=lambda: [
        Inspection("panel_A", (1.15,  0.62), (0.95, 0.55, 0.15, 1.0)),
        Inspection("panel_B", (3.95, -0.62), (0.20, 0.70, 0.95, 1.0)),
    ])
    # Which inspection panel the robot turns to scan on reaching each waypoint.
    inspect_at: dict = field(default_factory=lambda: {
        "approach": "panel_A", "station_b": "panel_B"})

    # A lateral shove applied to the trunk on the flat approach (disturbance test).
    # The gait's passive envelope is ~70 N (85 N topples it); with the IMU-triggered
    # crouch reflex the same patrol absorbs 100 N — chosen mid-plateau (90–110 N all
    # recover, 120 N is the reflex's limit).
    push_at_x: float = 0.70
    push_force: tuple[float, float, float] = (0.0, -100.0, 0.0)  # N, lateral (-y)
    push_duration: float = 0.15                                  # s

    @staticmethod
    def default() -> "CourseConfig":
        return CourseConfig()


# ---------------------------------------------------------------------------

def _add_materials(spec: mujoco.MjSpec) -> None:
    spec.add_texture(
        name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D,
        builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
        rgb1=[0.18, 0.20, 0.24], rgb2=[0.22, 0.24, 0.29], width=300, height=300,
    )
    spec.add_material(name="floor_mat", textures=["", "grid"],
                      texrepeat=[8, 8], reflectance=0.12)
    spec.add_material(name="ramp_mat", rgba=[0.33, 0.36, 0.42, 1.0], reflectance=0.05)
    spec.add_material(name="rubble_mat", rgba=[0.40, 0.34, 0.28, 1.0], reflectance=0.04)
    spec.add_material(name="post_mat", rgba=[0.55, 0.58, 0.62, 1.0], reflectance=0.2)
    spec.add_material(name="pad_mat", rgba=[0.18, 0.55, 0.30, 1.0], reflectance=0.05)


def _add_berm(world: mujoco.MjsBody) -> None:
    """A gentle ramp-up / flat-top / ramp-down berm the robot crests."""
    h = 0.055                                   # berm height
    ang = 0.16                                  # ramp incline (~9 deg)
    run = h / np.tan(ang)                       # horizontal run of each ramp
    cx = 2.05                                   # berm centre x
    # up-ramp (rotated slab), flat top, down-ramp
    world.add_geom(name="berm_up", type=mujoco.mjtGeom.mjGEOM_BOX,
                   size=[run / 1.4, 0.45, 0.02], pos=[cx - run, 0, h / 2],
                   euler=[0, -ang, 0], material="ramp_mat")
    world.add_geom(name="berm_top", type=mujoco.mjtGeom.mjGEOM_BOX,
                   size=[0.22, 0.45, h / 2], pos=[cx, 0, h / 2], material="ramp_mat")
    world.add_geom(name="berm_dn", type=mujoco.mjtGeom.mjGEOM_BOX,
                   size=[run / 1.4, 0.45, 0.02], pos=[cx + run, 0, h / 2],
                   euler=[0, ang, 0], material="ramp_mat")


def _add_rubble(world: mujoco.MjsBody) -> None:
    """A field of low, uneven fixed blocks — rough terrain to step over."""
    rng = np.random.default_rng(7)
    for i in range(14):
        x = 2.85 + rng.uniform(0.0, 0.85)
        y = rng.uniform(-0.34, 0.34)
        hz = rng.uniform(0.012, 0.030)          # 1.2–3 cm tall
        sx, sy = rng.uniform(0.05, 0.09, size=2)
        world.add_geom(name=f"rubble_{i}", type=mujoco.mjtGeom.mjGEOM_BOX,
                       size=[sx, sy, hz], pos=[x, y, hz],
                       euler=[0, 0, rng.uniform(0, 3.14)], material="rubble_mat")


def _add_inspection(world: mujoco.MjsBody, insp: Inspection) -> None:
    px, py = insp.post_xy
    facing = -1.0 if py > 0 else 1.0            # panel faces the path (toward y=0)
    world.add_geom(name=f"{insp.name}_post", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                   size=[0.018, 0.16], pos=[px, py, 0.16], material="post_mat")
    # the scan target panel, raised on the post, facing the patrol line
    world.add_geom(name=f"{insp.name}_panel", type=mujoco.mjtGeom.mjGEOM_BOX,
                   size=[0.005, 0.10, 0.085], pos=[px, py + facing * 0.02, 0.30],
                   rgba=list(insp.rgba))
    # a small site at the panel centre = the scan target the head camera aims at
    world.add_site(name=f"{insp.name}_target", pos=[px, py + facing * 0.03, 0.30],
                   size=[0.012, 0.012, 0.012], rgba=[insp.rgba[0], insp.rgba[1],
                                                     insp.rgba[2], 0.0])


def _add_sensors(spec: mujoco.MjSpec) -> None:
    """Onboard sensor suite, attached to the Go1's own sites.

    An IMU triad (gyro + accelerometer + velocimeter) on the trunk ``imu`` site
    feeds the disturbance-brace reflex, and a touch sensor on each foot site
    reports real foot-ground contact force (the HUD's contact dots and the
    telemetry log read these).  Everything the controller "feels" comes from
    these sensors — it never peeks at privileged simulator state for the reflex.
    """
    site = mujoco.mjtObj.mjOBJ_SITE
    spec.add_sensor(name="imu_gyro", type=mujoco.mjtSensor.mjSENS_GYRO,
                    objtype=site, objname="imu")
    spec.add_sensor(name="imu_accel", type=mujoco.mjtSensor.mjSENS_ACCELEROMETER,
                    objtype=site, objname="imu")
    spec.add_sensor(name="imu_vel", type=mujoco.mjtSensor.mjSENS_VELOCIMETER,
                    objtype=site, objname="imu")
    for foot in ("FR", "FL", "RR", "RL"):
        spec.add_sensor(name=f"touch_{foot}", type=mujoco.mjtSensor.mjSENS_TOUCH,
                        objtype=site, objname=foot)


def build_spec(cfg: CourseConfig | None = None) -> mujoco.MjSpec:
    cfg = cfg or CourseConfig.default()
    spec = mujoco.MjSpec.from_file(str(GO1_XML))
    spec.modelname = "sentinel_patrol"
    spec.visual.global_.offwidth = cfg.offwidth
    spec.visual.global_.offheight = cfg.offheight
    spec.visual.global_.azimuth = 120
    spec.visual.global_.elevation = -20
    spec.stat.extent = 2.5
    spec.stat.center = [2.3, 0.0, 0.2]

    _add_materials(spec)
    world = spec.worldbody
    world.add_light(pos=[1.5, -1.0, 2.5], dir=[-0.3, 0.4, -1.0],
                    diffuse=[0.85, 0.85, 0.9], castshadow=True)
    world.add_light(pos=[3.0, 1.0, 2.0], dir=[0.2, -0.3, -1.0], diffuse=[0.4, 0.45, 0.5])

    world.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                   size=[0, 0, 0.05], material="floor_mat")
    _add_berm(world)
    _add_rubble(world)
    for insp in cfg.inspections:
        _add_inspection(world, insp)
    # finish pad
    fx, fy = cfg.waypoints[-1].xy
    world.add_geom(name="finish_pad", type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                   size=[0.28, 0.004], pos=[fx, fy, 0.004], material="pad_mat")

    # onboard head camera (forward-looking) for the inspection picture-in-picture.
    trunk = spec.body("trunk")
    trunk.add_camera(name="head", pos=[0.30, 0.0, 0.02],
                     xyaxes=[0, -1, 0, 0, 0, 1], fovy=70)
    # chase + overview cameras tracking the trunk.
    trunk.add_camera(name="chase", pos=[-1.1, -1.3, 0.7], mode=mujoco.mjtCamLight.mjCAMLIGHT_TRACKCOM,
                     xyaxes=[0.76, -0.65, 0, 0.3, 0.35, 0.88])
    world.add_camera(name="overview", pos=[2.3, -3.4, 2.6], xyaxes=[1, 0, 0, 0, 0.6, 0.8])
    _add_sensors(spec)
    return spec


def compile_scene(cfg: CourseConfig | None = None):
    spec = build_spec(cfg)
    return spec.compile(), spec


if __name__ == "__main__":
    m, _ = compile_scene()
    print(f"sentinel course compiled: nq={m.nq} nu={m.nu} ngeom={m.ngeom} "
          f"ncam={m.ncam} nsite={m.nsite} nsensor={m.nsensor}")
