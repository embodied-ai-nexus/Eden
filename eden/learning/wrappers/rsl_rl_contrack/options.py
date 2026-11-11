"""Runner/algorithm options for the external ``rsl_rl_contrack`` CMDP-PPO backend.

Declarative only — importing this module needs neither the backend nor rsl_rl's runner API, so the ``contrack``
task config that selects it imports on any install. The runner its ``class_name`` names
(:class:`~eden.learning.wrappers.rsl_rl_contrack.runner.ContrackOnPolicyRunner`) needs the backend and raises with
the install command when it is missing, so ``eden train --task contrack`` is the training entry point.

ConTrack (https://github.com/EmptyBlueBox/ConTrack) trains tracking policies as
a constrained MDP: rewards are split into task (``rg``), style (``rs``), and
penalty (``rp``) groups, the critic outputs two values for the ``(rg+rp, rs+rp)``
channels, and a dual variable mixes the two advantage streams with
``w_task = sigmoid(λ)`` updated toward a target task ratio ``alpha``. The reward
groups are declared on the env config (``RewardGroupOptions``); these options
carry only the learning hyperparameters. Defaults match the ConTrack release.
"""

from __future__ import annotations

from typing import Any

from eden.options.learning.clip_actions import ClipActions
from eden.options.learning.rsl_rl.rl import (
    RslRlGaussianDistributionOptions,
    RslRlModelOptions,
    RslRlModelWithDistributionOptions,
    RslRlOnPolicyRunnerOptions,
    RslRlPpoAlgorithmOptions,
)


class RslRlContrackAlgorithmOptions(RslRlPpoAlgorithmOptions):
    """PPO options extended with ConTrack's dual-controller (CMDP) parameters.

    Parameters
    ----------
    cmdp_alpha: float
        Target task ratio ``alpha`` — the constraint ``J_g >= alpha * J_g*``.
    cmdp_lambda_lr: float
        Dual step size for ``λ``.
    cmdp_warmup_iters: int
        Updates with ``w_task = 1`` (pure task reward) before the dual controller
        takes over.
    cmdp_vg_window: int
        Episodes in the running mean of the task return ``Ĵ_g``.
    cmdp_vg_star_window: int
        Updates in the running max defining the normalizer ``J_g*``.
    cmdp_lambda_init: float
        Initial ``λ`` (``0`` gives ``w_task = 0.5``).
    """

    # PPO hyperparameters (ConTrack release values).
    entropy_coef: float = 0.001
    num_learning_epochs: int = 10
    num_mini_batches: int = 16
    learning_rate: float = 5.0e-4
    # CMDP dual-controller.
    cmdp_alpha: float = 0.9
    cmdp_lambda_lr: float = 5.0e-2
    cmdp_warmup_iters: int = 0
    cmdp_vg_window: int = 100
    cmdp_vg_star_window: int = 20
    cmdp_lambda_init: float = 0.0


# A literal (not imported from the runner module) so this module imports without the backend;
# ``tests/learning/wrappers/rsl_rl_contrack/test_runner.py`` pins it against the implementation.
_CONTRACK_RUNNER_CLASS_NAME = "eden.learning.wrappers.rsl_rl_contrack.runner:ContrackOnPolicyRunner"


class RslRlContrackRunnerOptions(RslRlOnPolicyRunnerOptions):
    """On-policy runner options that select the ``rsl_rl_contrack`` backend.

    The ``actor`` / ``critic`` model options are consumed by
    ``rsl_rl_contrack.modules.ActorCritic`` (MLP-only; the critic automatically
    outputs the two CMDP value channels). Defaults mirror the ConTrack release
    hyperparameters, so ``RslRlContrackRunnerOptions()`` is training-ready.

    Parameters
    ----------
    class_name : str
        The runner ``eden train`` / ``eden inference`` build; leave at the default.
    """

    class_name: str = _CONTRACK_RUNNER_CLASS_NAME
    num_steps_per_env: int = 16
    max_iterations: int = 5000
    save_interval: int = 200
    clip_actions: ClipActions = 1.0
    experiment_name: str = "contrack"
    obs_groups: dict = {"actor": ["policy"], "critic": ["critic"]}
    actor: RslRlModelWithDistributionOptions = RslRlModelWithDistributionOptions(
        hidden_dims=[256, 256, 128],
        distribution_cfg=RslRlGaussianDistributionOptions(init_std=1.0, std_type="scalar"),
    )
    critic: RslRlModelOptions = RslRlModelOptions(hidden_dims=[256, 256, 128])
    algorithm: RslRlContrackAlgorithmOptions = RslRlContrackAlgorithmOptions()

    def model_post_init(self, context: Any) -> None:
        """Refuse what the fork cannot honour: these inherited fields would otherwise be accepted and then ignored."""
        super().model_post_init(context)
        algorithm, distribution = self.algorithm, self.actor.distribution_cfg
        default_std_range = RslRlGaussianDistributionOptions.model_fields["std_range"].default
        unsupported = {
            "actor.class_name": (self.actor.class_name, self.actor.class_name == "MLPModel"),
            "critic.class_name": (self.critic.class_name, self.critic.class_name == "MLPModel"),
            "critic.activation": (self.critic.activation, self.critic.activation == self.actor.activation),
            "actor.distribution_cfg": (
                distribution,
                isinstance(distribution, RslRlGaussianDistributionOptions)
                and distribution.class_name == "GaussianDistribution"
                and tuple(distribution.std_range) == tuple(default_std_range),
            ),
            "algorithm.class_name": (algorithm.class_name, algorithm.class_name == "PPO"),
            "algorithm.rnd_cfg": (algorithm.rnd_cfg, algorithm.rnd_cfg is None),
            "algorithm.symmetry_cfg": (algorithm.symmetry_cfg, algorithm.symmetry_cfg is None),
            "algorithm.share_cnn_encoders": (algorithm.share_cnn_encoders, not algorithm.share_cnn_encoders),
        }
        bad = {field: value for field, (value, ok) in unsupported.items() if not ok}
        if bad:
            raise ValueError(
                f"{type(self).__name__}: the rsl_rl_contrack fork cannot honour {bad}. It builds its own MLP "
                "ActorCritic (one activation for actor and critic, a Gaussian with the default std_range) and its CMDP "
                "PPO, with no RND, symmetry or CNN encoders. Leave these at their defaults."
            )
