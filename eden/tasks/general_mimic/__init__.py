"""General-purpose motion-tracking task for small~mid scale datasets.

Designed for 10–100 motion clips (e.g. LAFAN1, a SEED subset, or a custom
processed-cache slice). Composes:

  * BeyondMimic-style **action / DR** setup (ExplicitPDController on G1
    with the canonical action scale + RSI + push / friction / COM-shift
    domain randomization).
  * ProtoMotions-aligned **reward / observation / termination** family
    (anchor-relative body pos & ori + global body lin/ang vel + soft DOF
    limit / action-rate / self-collision penalties; same numbers as the
    pretrained ``motion_tracker/g1-bones-deploy`` reference).
  * Eden's **per-motion adaptive curriculum** with per-motion metrics
    (auto-logs ``MotionWeights/*`` and ``MotionOutcome/*`` to wandb every
    iter — no extra wiring).

Construct via the registry::

    from eden.tasks import TASK_REGISTRY
    cfg = TASK_REGISTRY.build("general_mimic")

See ``eden/tasks/general_mimic/config.py`` for the full task definition.
"""
