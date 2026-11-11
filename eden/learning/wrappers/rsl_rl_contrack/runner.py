"""Runner for the external ``rsl_rl_contrack`` backend (ConTrack's CMDP-PPO fork of rsl_rl).

``rsl_rl_contrack`` is a standalone RL library installed separately (it is not an Eden extra).
:class:`ContrackOnPolicyRunner` is what
:class:`~eden.learning.wrappers.rsl_rl_contrack.options.RslRlContrackRunnerOptions`'s ``class_name`` names, so
``eden train`` / ``eden inference`` build it. Importing this module raises with the install command when the backend
is missing — ``eden train`` resolves the runner before building the env, so that happens in seconds.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

from eden.envs.wrappers.rsl_rl_env import CheckpointRetentionMixin
from eden.learning.wrappers.rsl_rl_contrack.env import ContrackVecEnvWrapper

try:
    from rsl_rl_contrack.runners import OnPolicyRunner as _ContrackForkRunner
except ImportError as e:
    raise ImportError(
        "The ConTrack CMDP-PPO backend `rsl_rl_contrack` is not installed (it is a standalone fork of rsl_rl, not an "
        "Eden extra). Install it, then re-run training:\n"
        "    pip install 'git+https://github.com/Kashu7100/rsl_rl_contrack@7b6c043f3ceea45e8afc8175ba74ba3876510e25'\n"
        "Building and stepping the env (`eden smoke --task contrack`) does not need it."
    ) from e

if TYPE_CHECKING:
    from eden.envs.wrappers.rsl_rl_env import RslRlVecEnvWrapper
    from eden.learning.wrappers.rsl_rl_contrack.options import RslRlContrackRunnerOptions

#: ``class_name`` string ``eden train`` resolves to :class:`ContrackOnPolicyRunner`.
CONTRACK_RUNNER_CLASS_NAME = "eden.learning.wrappers.rsl_rl_contrack.runner:ContrackOnPolicyRunner"


def contrack_train_cfg(options: RslRlContrackRunnerOptions) -> dict:
    """The fork's ``train_cfg`` dict, assembled directly from the runner options.

    No rsl-rl-lib version translation — the fork has a single, known API.
    """
    # Eden's obs_groups convention names the actor set "actor"; the contrack
    # runner (rsl_rl v5 lineage) calls it "policy".
    obs_groups = {("policy" if k == "actor" else k): list(v) for k, v in options.obs_groups.items()}
    if not obs_groups:
        obs_groups = {"policy": ["policy"], "critic": ["critic"]}

    distribution = options.actor.distribution_cfg
    algorithm = options.algorithm
    return {
        "num_steps_per_env": options.num_steps_per_env,
        "save_interval": options.save_interval,
        "keep_last_checkpoints": options.keep_last_checkpoints,
        "obs_groups": obs_groups,
        "logger": options.logger,
        "wandb_project": options.wandb_project,
        "neptune_project": options.neptune_project,
        "experiment_name": options.experiment_name or "contrack",
        "run_name": options.run_name,
        "policy": {
            "class_name": "ActorCritic",
            "actor_hidden_dims": list(options.actor.hidden_dims),
            "critic_hidden_dims": list(options.critic.hidden_dims),
            "activation": options.actor.activation,
            "actor_obs_normalization": options.actor.obs_normalization,
            "critic_obs_normalization": options.critic.obs_normalization,
            "init_noise_std": distribution.init_std if distribution is not None else 1.0,
            "noise_std_type": distribution.std_type if distribution is not None else "scalar",
        },
        "algorithm": {
            "class_name": "PPO",
            "num_learning_epochs": algorithm.num_learning_epochs,
            "num_mini_batches": algorithm.num_mini_batches,
            "clip_param": algorithm.clip_param,
            "gamma": algorithm.gamma,
            "lam": algorithm.lam,
            "value_loss_coef": algorithm.value_loss_coef,
            "entropy_coef": algorithm.entropy_coef,
            "learning_rate": algorithm.learning_rate,
            "max_grad_norm": algorithm.max_grad_norm,
            "use_clipped_value_loss": algorithm.use_clipped_value_loss,
            "schedule": algorithm.schedule,
            "desired_kl": algorithm.desired_kl,
            "normalize_advantage_per_mini_batch": algorithm.normalize_advantage_per_mini_batch,
            "cmdp_alpha": algorithm.cmdp_alpha,
            "cmdp_lambda_lr": algorithm.cmdp_lambda_lr,
            "cmdp_warmup_iters": algorithm.cmdp_warmup_iters,
            "cmdp_vg_window": algorithm.cmdp_vg_window,
            "cmdp_vg_star_window": algorithm.cmdp_vg_star_window,
            "cmdp_lambda_init": algorithm.cmdp_lambda_init,
        },
    }


class ContrackOnPolicyRunner(CheckpointRetentionMixin, _ContrackForkRunner):
    """The fork's ``OnPolicyRunner`` behind Eden's runner signature, with Eden's checkpoint retention.

    ``eden train`` hands it a plain :class:`~eden.envs.wrappers.rsl_rl_env.RslRlVecEnvWrapper` and the
    version-translated runner dict; the fork needs neither. It re-wraps the same built env in
    :class:`ContrackVecEnvWrapper` (which adds the CMDP reward-group extras the fork's PPO reads) and rebuilds the
    fork's ``train_cfg`` from the env's runner options. Checkpoint retention matters because the fork saves
    ``model_<it>.pt`` the same way rsl_rl does, and ``keep_last_checkpoints`` would otherwise be silently ignored.
    """

    def __init__(self, env: RslRlVecEnvWrapper, train_cfg: dict, log_dir: str | None = None, device: str = "cpu"):
        if not isinstance(env, ContrackVecEnvWrapper):
            env = ContrackVecEnvWrapper(env.unwrapped, env.runner_options)
        super().__init__(env, contrack_train_cfg(env.runner_options), log_dir=log_dir, device=device)
