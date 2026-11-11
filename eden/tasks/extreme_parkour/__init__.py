"""Extreme Parkour tasks (Cheng et al. 2023, Unitree Go2).

Two phases, both trained by ``eden train`` through the RMA stack in
:mod:`eden.learning.wrappers.rsl_rl.rma` (selected by each config's ``runner_options``):

- ``extreme_parkour_teacher`` (:mod:`.teacher.config`) — the privileged
  scandots teacher.
- ``extreme_parkour_student`` (:mod:`.student.config`) — the same environment
  plus the forward depth camera (gs-madrona batch renderer), action delay and
  camera-mount randomization, for DAgger depth distillation from a teacher
  checkpoint (``eden train --task extreme_parkour_student --checkpoint <teacher>``).

Both self-register via ``@TASK_REGISTRY.register()`` (the task registry
AST-scans ``eden/tasks/**`` — no central index to update). Custom
command/observation/reward/termination/event terms live in
:mod:`eden.tasks.extreme_parkour.custom_terms`.
"""
