"""HOMIE training task: the OpenHomie G1 velocity + base-height locomotion controller.

The task config (:mod:`.config`) self-registers as ``homie``; its custom terms live in
:mod:`eden.tasks.homie.custom_terms` and its sagittal mirror in :mod:`.symmetry`. It trains with the
task-agnostic HIMLoco learner :mod:`eden.learning.wrappers.rsl_rl.him`. The frozen OpenHomie release
policy is :class:`eden.extensions.policies.homie.HomiePolicy`.
"""
