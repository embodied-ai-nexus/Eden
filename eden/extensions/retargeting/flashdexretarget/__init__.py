"""FlashDexRetarget's kinematic retargeting (Holiday-Robot/FlashDexRetarget, MIT): human hand-object demos -> a
bimanual robot-hand ``motion.pt``, without Isaac Sim.

Two stages:

1. :func:`import_demos` — a released dataset (TACO, OakInk2, HOT3D, DexYCB), read by its :mod:`eden.datasets`
   loader, -> ``<dataset>/human_demo/<clip>.npz``
   (MANO wrist and finger keypoints, object poses, z-up, resampled) and ``<dataset>/objects/<id>/visual.obj``.
2. :func:`run_kinematic_retargeting` — support disks and reject rules, convex object parts, MuJoCo physics IK onto
   the robot (``hands/flashdexretarget/<robot>/*.xml``), ``<dataset>/<robot>/motion.pt``, object SDF grids.

``eden retarget flashdex {import,kinematic}`` drives both from :mod:`eden.options.extensions.flashdexretarget`.
Re-exports are lazy, so importing the package pulls neither MuJoCo nor ``smplx``.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from eden.extensions.retargeting.flashdexretarget.importer import import_demos
    from eden.extensions.retargeting.flashdexretarget.pipeline import run_kinematic_retargeting

__all__ = ["import_demos", "run_kinematic_retargeting"]


def __getattr__(name: str):
    if name == "import_demos":
        from eden.extensions.retargeting.flashdexretarget.importer import import_demos

        return import_demos
    if name == "run_kinematic_retargeting":
        from eden.extensions.retargeting.flashdexretarget.pipeline import run_kinematic_retargeting

        return run_kinematic_retargeting
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
