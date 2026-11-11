"""HUSKY humanoid-skateboarding training task (G1 23-DOF + skateboard).

The task config (:mod:`.config`) self-registers via ``@TASK_REGISTRY.register()``;
custom command/observation/reward/termination terms live in
:mod:`eden.tasks.husky.custom_terms`. The config trains with the AMP-augmented
PPO from :mod:`eden.learning.wrappers.rsl_rl.amp_ppo`, so ``eden train --task husky``
is the full HUSKY recipe.
"""
