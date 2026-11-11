"""Which scripted demonstrator drives which task: ``SCRIPTED_DEMO_REGISTRY``, keyed by task name.

An entry is a factory ``(env, *, seed, p_mistake) -> demonstrator`` — a keyframe
:class:`~eden.extensions.scripted_demos.controller.ScriptedPickPlaceController`, a tool-routine
:class:`~eden.extensions.scripted_demos.routines.RoutineDemonstrator`, or a two-arm
:class:`~eden.extensions.scripted_demos.multi_arm.MultiArmDemonstrator` — registered under the task's
:data:`~eden.tasks.registry.TASK_REGISTRY` name, next to the task it scripts
(``eden/tasks/<suite>/scripted_demos.py``) — list that module in ``populated_by`` below. A factory
that also takes ``privileged`` has a camera-only mode (``eden datagen --camera_only``), and one whose
episodes need longer than the generator's default declares an ``episode_length_s`` attribute.
"""

from __future__ import annotations

from eden.utils.registry import Registry

SCRIPTED_DEMO_REGISTRY = Registry(
    "SCRIPTED_DEMO",
    populated_by=(
        "eden.tasks.robosuite.scripted_demos",
        "eden.tasks.simpler.scripted_demos",
        "eden.tasks.polaris.scripted_demos",
        "eden.tasks.whiteboard.scripted_demos",
        "eden.tasks.art.calligraphy.scripted_demos",
        "eden.tasks.art.painting.scripted_demos",
    ),
)
