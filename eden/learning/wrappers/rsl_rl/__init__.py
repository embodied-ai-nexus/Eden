"""Models, algorithms and runners for rsl-rl-lib 5, selected through their options' ``class_name``.

- :mod:`.simba` — a SimBaV2 actor / critic behind one encoder per observation group.
- :mod:`.muon_ppo` — PPO whose optimizer is Muon for the 2-D weights and AdamW for the rest.
- :mod:`.him` — HIMLoco (OpenHomie's ``HIMPPO``): a self-supervised history estimator feeding the
  actor, plus rollout-time symmetry and mirror losses.
- :mod:`.amp_ppo` — PPO plus an adversarial-motion-prior style reward (HUSKY).
- :mod:`.rma` — the RMA teacher (actor with estimator + scan / privileged / history encoders, PPO with
  privileged-latent regularization and DAgger) and its depth-student distillation runner (extreme parkour).

Each subpackage ships an options class subclassing the rsl_rl option it replaces (``RslRlModelOptions`` /
``RslRlPpoAlgorithmOptions`` / ``RslRlBaseRunnerOptions``) whose ``class_name`` defaults to its implementation, so a
task selects it with ``actor=<Options>(...)``, ``algorithm=<Options>(...)`` or ``runner_options=<Options>(...)``.
"""

#: The oldest rsl-rl-lib :mod:`.amp_ppo` and :mod:`.rma` run on: ``rsl_rl.utils.resolve_class`` arrived in 5.5.0,
#: ``reduce_gradients_in_buckets`` and PPO's ``grad_reduce_bucket_mb`` in 5.5.1.
MIN_RSL_RL_VERSION = (5, 5, 1)
