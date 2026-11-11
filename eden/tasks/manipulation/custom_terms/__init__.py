"""Custom MDP terms for the manipulation tasks, split by manager.

These mirror the terms in ``mjlab_manipulation`` (``tasks/lift/env_cfgs.py`` and
``tasks/reach/env_cfgs.py``), re-expressed against Eden's Genesis entity API.
Each module holds the terms for one manager (and ``common`` the shared
geometry/frame/state helpers); importing this package registers them all with
their respective ``*_TERM_REGISTRY`` and re-exports the public names the task
configs reference (e.g. ``custom_terms.object_ee_distance``).
"""

from eden.tasks.manipulation.custom_terms.curricula import GoalRangeCurriculum
from eden.tasks.manipulation.custom_terms.observations import (
    grasp_point_b,
    object_orientation_b,
    object_position_b,
    object_rot6d_b,
)
from eden.tasks.manipulation.custom_terms.rewards import (
    grasp_bonus,
    lift_height,
    object_ee_distance,
    object_goal_distance,
    object_is_lifted,
    reaching_target,
    settle_at_goal,
)
from eden.tasks.manipulation.custom_terms.terminations import object_fallen

__all__ = [
    # observations
    "object_position_b",
    "object_orientation_b",
    "object_rot6d_b",
    "grasp_point_b",
    # rewards
    "object_ee_distance",
    "lift_height",
    "grasp_bonus",
    "object_is_lifted",
    "object_goal_distance",
    "reaching_target",
    "settle_at_goal",
    # terminations
    "object_fallen",
    # curricula
    "GoalRangeCurriculum",
]
