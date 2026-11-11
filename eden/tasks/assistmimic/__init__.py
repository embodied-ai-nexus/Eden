"""AssistMimic: a caregiver humanoid helps a weakened recipient up, both tracking a paired clip (SMPL).

The task configs (:mod:`.config` — ``assistmimic`` on Inter-X, ``assistmimic_bed`` on HHI-Assist) self-register
via ``@TASK_REGISTRY.register()``; the custom terms live in :mod:`eden.tasks.assistmimic.custom_terms` and the
weakened recipients in :mod:`.robots`. The two-agent trainer is example-local
(``examples/projects/mimic/assistmimic``) — the configs declare no ``runner_options``.
"""
