"""Scripted demonstrators that produce demo-like trajectories — a port of OCBench's scripted policies.

OCBench (Park et al., 2026, https://github.com/seohongpark/ocbench) replaces the rule-based,
noise-injected scripted policies of earlier benchmarks with ones that move like a person: smooth
minimum-jerk-timed splines through randomized keyframes, a per-episode speed, pauses, tilted and
off-centre grasps, retries after a failed pick, and deliberate high-level mistakes the demonstrator
then recovers from. This package ports that machinery onto Eden's batched envs:

* :mod:`~eden.extensions.scripted_demos.trajectory` — keyframe plans and their smooth evaluation.
* :mod:`~eden.extensions.scripted_demos.primitives` — the pick-and-place keyframe sampler.
* :mod:`~eden.extensions.scripted_demos.controller` — plan clock, gates, retries, and the
  tool-pose to IK-action mapping, around a task-specific subgoal choice.
* :mod:`~eden.extensions.scripted_demos.block_world` — tower-building and lifting demonstrators.
* :mod:`~eden.extensions.scripted_demos.rig` — the robot a demonstrator drives, and
  :class:`~eden.extensions.scripted_demos.rig.RigDriver`, which turns tool poses into its actions.
* :mod:`~eden.extensions.scripted_demos.tools` / :mod:`~eden.extensions.scripted_demos.routines` —
  tool routines: a scripted policy written for one robot as a generator of motions (``move_to``,
  ``move_until_contact``, ``close_gripper``, ...), run batched over every env, privileged or
  camera-only.
* :mod:`~eden.extensions.scripted_demos.multi_arm` — one routine per arm, run side by side and
  handing off through a blackboard each env's routines share.
* :mod:`~eden.extensions.scripted_demos.generate` — the batch generator behind ``eden datagen``.

A task gets a demonstrator by registering a factory in :data:`SCRIPTED_DEMO_REGISTRY` under its task
name, next to the task (``eden/tasks/robosuite/scripted_demos.py``).
"""

from eden.extensions.scripted_demos.registry import SCRIPTED_DEMO_REGISTRY

__all__ = ["SCRIPTED_DEMO_REGISTRY"]
