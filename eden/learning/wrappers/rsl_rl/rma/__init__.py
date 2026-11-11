"""RMA-style teacher training and depth-student distillation on rsl-rl-lib 5 (the extreme-parkour stack).

- :mod:`.model` — :class:`~.model.RmaActorModel`: estimator + scan / privileged / history encoders + backbone over one
  flat ``[prop | scan | priv_explicit | priv_latent | history]`` observation; its default forward is the deployed
  policy, so ``eden inference`` / ``eden deploy`` need nothing extra.
- :mod:`.ppo` — :class:`~.ppo.RmaPPO`: PPO + concurrent estimator + privileged-latent regularization + DAgger
  iterations (+ an optional mirror loss), driven by rsl_rl's stock ``OnPolicyRunner``.
- :mod:`.distillation` — :class:`~.distillation.RmaDepthDistillationRunner`: the DAgger loop that distills a teacher
  checkpoint into a recurrent depth encoder + student actor. The student replays in sim (``eden inference``) only;
  it reads the env's ground-truth base velocity, and ``eden deploy`` refuses its runner.

A task selects them through the option classes re-exported here (``actor=RslRlRmaActorOptions(...)``,
``algorithm=RslRlRmaPpoAlgorithmOptions(...)``, or ``runner_options=RslRlRmaDepthDistillationRunnerOptions(...)``).
Only the options are re-exported: importing this package needs neither rsl-rl-lib >= 5.5.1 nor the models.
"""

from eden.learning.wrappers.rsl_rl.rma.options import (
    RslRlRmaActorOptions,
    RslRlRmaDepthDistillationAlgorithmOptions,
    RslRlRmaDepthDistillationRunnerOptions,
    RslRlRmaPpoAlgorithmOptions,
)

__all__ = [
    "RslRlRmaActorOptions",
    "RslRlRmaDepthDistillationAlgorithmOptions",
    "RslRlRmaDepthDistillationRunnerOptions",
    "RslRlRmaPpoAlgorithmOptions",
]
