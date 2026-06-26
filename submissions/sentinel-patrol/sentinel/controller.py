"""High-level navigation: turn the robot's state into gait velocity commands.

The :class:`WaypointFollower` reads the trunk pose each control tick and produces a
``(forward, yaw)`` command for :class:`~sentinel.gait.TrotGait` that drives the robot
toward the current waypoint — proportional heading control with a forward speed that
eases off while the robot is turning hard or cresting an obstacle.  This closes the
loop the open-loop gait lacks (which otherwise drifts off course), and is what lets
the patrol hold its line across the berm and rubble.
"""

from __future__ import annotations

import numpy as np


def trunk_yaw(quat: np.ndarray) -> float:
    """World yaw (heading) of the trunk from its quaternion (w, x, y, z)."""
    w, x, y, z = quat
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _wrap(a: float) -> float:
    return float((a + np.pi) % (2.0 * np.pi) - np.pi)


class WaypointFollower:
    def __init__(self, waypoints, heading_gain: float = 1.6, cruise: float = 1.0):
        self.waypoints = waypoints
        self.heading_gain = heading_gain
        self.cruise = cruise
        self.idx = 0
        self.reached: list[str] = []

    @property
    def done(self) -> bool:
        return self.idx >= len(self.waypoints)

    @property
    def target(self):
        return None if self.done else self.waypoints[self.idx]

    def command(self, pos_xy: np.ndarray, yaw: float) -> tuple[float, float]:
        """Return ``(forward, yaw_cmd)`` toward the current waypoint, advancing the
        waypoint index when the robot enters its acceptance radius."""
        if self.done:
            return 0.0, 0.0
        wp = self.waypoints[self.idx]
        to = np.asarray(wp.xy) - pos_xy
        dist = float(np.hypot(*to))
        if dist < wp.radius:
            self.reached.append(wp.name)
            self.idx += 1
            return self.command(pos_xy, yaw)

        desired = float(np.arctan2(to[1], to[0]))
        err = _wrap(desired - yaw)
        yaw_cmd = float(np.clip(self.heading_gain * err, -1.0, 1.0))
        # ease forward speed when turning hard so the body stays upright; never below
        # a slow crawl so the robot keeps making progress.
        forward = self.cruise * float(np.clip(np.cos(err), 0.25, 1.0))
        return forward, yaw_cmd
