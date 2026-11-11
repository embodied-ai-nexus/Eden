"""Weave's runner for ``weave_hoi``: a SimBaV2 actor-critic trained by Muon PPO, at ``configs/track/train.yaml``.

The model and the algorithm are task-agnostic and live in :mod:`eden.learning.wrappers.rsl_rl` (:mod:`~.simba`,
:mod:`~.muon_ppo`). What is Weave's is here: its observation sets, its per-group encoder widths and its
hyperparameters (:func:`weave_runner_options`), and :class:`RslRlWeaveRunnerOptions`, which narrows the runner's
``actor`` / ``critic`` / ``algorithm`` fields to those option classes so a saved config reloads into them rather than
into the stock ones (whose ``extra="forbid"`` would reject ``hidden_dim``, ``latent_dim``, ``weight_decay``, …).
"""

from __future__ import annotations

from eden.learning.wrappers.rsl_rl.muon_ppo import RslRlMuonPpoAlgorithmOptions
from eden.learning.wrappers.rsl_rl.simba import RslRlSimBaActorOptions, RslRlSimBaCriticOptions
from eden.options.learning.rsl_rl.rl import RslRlGaussianDistributionOptions, RslRlOnPolicyRunnerOptions

#: Weave's per-group encoder widths (``configs/track/train.yaml``): one 256-wide hidden layer for every group.
WEAVE_ENCODER_HIDDEN_DIMS: dict[str, list[int]] = {
    "ref_motion_body": [256],
    "ref_motion_object": [256],
    "object_state": [256],
    "robot_proprio": [256],
    "robot_privileged": [256],
}

#: Weave's observation sets: both networks see the reference and the object state; the actor adds noisy
#: proprioception, the critic privileged state.
WEAVE_OBS_GROUPS: dict[str, list[str]] = {
    "actor": ["ref_motion_body", "ref_motion_object", "object_state", "robot_proprio"],
    "critic": ["ref_motion_body", "ref_motion_object", "object_state", "robot_privileged"],
}


class RslRlWeaveRunnerOptions(RslRlOnPolicyRunnerOptions):
    """On-policy runner whose actor / critic / algorithm fields are narrowed to the SimBa + Muon option classes.

    Parameters
    ----------
    actor : RslRlSimBaActorOptions
        The actor model configuration.
    critic : RslRlSimBaCriticOptions
        The critic model configuration.
    algorithm : RslRlMuonPpoAlgorithmOptions
        The algorithm configuration.
    """

    actor: RslRlSimBaActorOptions = RslRlSimBaActorOptions()
    critic: RslRlSimBaCriticOptions = RslRlSimBaCriticOptions()
    algorithm: RslRlMuonPpoAlgorithmOptions = RslRlMuonPpoAlgorithmOptions()


def _weave_simba_fields() -> dict:
    """The SimBa fields Weave's actor and critic share, as fresh objects for each."""
    return dict(
        hidden_dim=512,
        expansion=4,
        latent_dim=128,
        encoder_hidden_dims={k: list(v) for k, v in WEAVE_ENCODER_HIDDEN_DIMS.items()},
        default_encoder_hidden_dims=[256],
        c_shift=3.0,
        activation="swish",  # rsl_rl's name for Weave's SiLU
        obs_normalization=True,
        # Compiled as Weave trains: on an L40S at 4096 envs the PPO update drops from 1.50 s to 0.93 s per iteration
        # (a one-time ~9 s compile on the first one).
        compile=True,
    )


def weave_runner_options(**overrides) -> RslRlWeaveRunnerOptions:
    """Weave's G1-Inspire-HOI runner (``configs/track/train.yaml``): SimBaV2 actor-critic, Muon PPO.

    Parameters
    ----------
    **overrides
        Runner-level fields to set on top (``experiment_name``, ``max_iterations``, ``wandb_project``, …).

    Returns
    -------
    RslRlWeaveRunnerOptions
        32 steps per env, Weave's observation sets and per-group encoders (one 256-wide SiLU layer to a 128-wide
        latent per group, after empirical normalisation), actor 512-wide / 1 block, critic 512-wide / 2 blocks (both
        trunks ``torch.compile``-d, as Weave trains), expansion 4, embedder shift 3.0, scalar init std 1.0, lr 1e-4
        adaptive to KL 0.01, entropy 5e-4, 5 epochs x 4 minibatches, gamma 0.99, lambda 0.95, clip 0.2 with clipped
        value loss, grad-norm 1.0, AdamW weight decay 0.01, Muon weight decay 0.1.

    Every Weave value is set here rather than left to the option classes' defaults, which are shared with other
    tasks and may change.
    """
    fields = dict(
        num_steps_per_env=32,
        save_interval=100,
        max_iterations=100_000,
        obs_groups={k: list(v) for k, v in WEAVE_OBS_GROUPS.items()},
        actor=RslRlSimBaActorOptions(
            num_blocks=1,
            distribution_cfg=RslRlGaussianDistributionOptions(init_std=1.0, std_type="scalar"),
            **_weave_simba_fields(),
        ),
        critic=RslRlSimBaCriticOptions(num_blocks=2, **_weave_simba_fields()),
        algorithm=RslRlMuonPpoAlgorithmOptions(
            learning_rate=1.0e-4,
            schedule="adaptive",
            desired_kl=0.01,
            entropy_coef=5.0e-4,
            num_learning_epochs=5,
            num_mini_batches=4,
            gamma=0.99,
            lam=0.95,
            clip_param=0.2,
            use_clipped_value_loss=True,
            value_loss_coef=1.0,
            max_grad_norm=1.0,
            weight_decay=0.01,
            muon_weight_decay=0.1,  # torch.optim.Muon's default, which Weave leaves
        ),
    )
    fields.update(overrides)
    return RslRlWeaveRunnerOptions(**fields)
