"""Offline motion-optimization extensions.

Backends that take an existing motion clip and re-optimize it — as opposed to
:mod:`eden.extensions.retargeting`, whose env-bound ``RetargetingBase`` backends
map a source motion onto a live robot entity online. The members here are
env-free pipelines consumed offline (e.g. by a dataset modifier).

Current members:

- :mod:`~eden.extensions.motion_optimization.tcrs` — the TCRS terrain-adaptive
  motion optimizer (footstep planning + whole-body IK), exposed as
  :class:`~eden.extensions.motion_optimization.tcrs.pipeline.TCRSMotionAdapter`
  and driven by the ``tcrs_terrain_adapt`` dataset modifier.
"""


def __getattr__(name: str):
    # Lazy re-export so the pipeline's mujoco dependency isn't required to
    # import this package.
    if name == "TCRSMotionAdapter":
        from eden.extensions.motion_optimization.tcrs import TCRSMotionAdapter

        return TCRSMotionAdapter
    raise AttributeError(name)
