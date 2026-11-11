"""TCRS terrain-adaptive motion optimizer.

Transforms flat-ground biped motion clips into terrain-adapted versions:
contact/gait-phase detection, MPPI foot-swing planning against a terrain
height field, support-aware pelvis-height re-targeting, and whole-body
Jacobian IK on a MuJoCo model of the robot.

Ported from Mondo-Robotics/PMT ``TCRS`` @ 88838ca. The engine is env-free
(numpy + scipy + mujoco); the dataset-facing entry point is the
:class:`~eden.datasets.modifiers.terrain_adapt.TCRSTerrainAdapt` modifier.
Config classes (:class:`~eden.options.extensions.motion_optimization.TCRSRobotOptions`,
:class:`~eden.options.extensions.motion_optimization.TCRSPlannerOptions`) live with the
other extension options and are re-exported here for convenience.
"""

from eden.options.extensions.motion_optimization import DEFAULT_IK_CONFIG, TCRSPlannerOptions, TCRSRobotOptions
from eden.extensions.motion_optimization.tcrs.terrain_source import HeightFieldTerrain, TerrainSource


__all__ = [
    "DEFAULT_IK_CONFIG",
    "AdaptResult",
    "ClipArrays",
    "HeightFieldTerrain",
    "MotionSet",
    "TCRSMotionAdapter",
    "TCRSPlannerOptions",
    "TCRSRobotOptions",
    "TerrainSource",
]


def __getattr__(name: str):
    # The pipeline (and its mujoco import) stays lazy so the config/terrain
    # helpers import without mujoco present.
    if name in ("TCRSMotionAdapter", "ClipArrays", "AdaptResult", "MotionSet"):
        from eden.extensions.motion_optimization.tcrs import pipeline

        return getattr(pipeline, name)
    raise AttributeError(name)
