"""Algorithm options for :class:`~eden.learning.wrappers.rsl_rl.amp_ppo.ppo.AmpPPO`.

Declarative only — importing this module pulls in neither rsl_rl's algorithm API nor the discriminator, so a task
config that uses it still imports under any rsl_rl Eden supports. The algorithm its ``class_name`` names needs
rsl-rl-lib >= 5.5.1 and says so when imported without it.
"""

from __future__ import annotations

from eden.options.learning.rsl_rl.rl import RslRlPpoAlgorithmOptions

# A literal (not imported from the algorithm module) so this module stays importable without rsl-rl-lib >= 5.5.1;
# ``tests/learning/wrappers/rsl_rl/amp_ppo/test_ppo.py`` pins it against the implementation.
_AMP_PPO_CLASS_NAME = "eden.learning.wrappers.rsl_rl.amp_ppo.ppo:AmpPPO"


class RslRlAmpPpoAlgorithmOptions(RslRlPpoAlgorithmOptions):
    """PPO plus an adversarial motion prior (AMP) style reward; defaults are the HUSKY release's.

    The env must publish the observation group ``amp_obs_group`` as ``[joint positions x amp_num_frames (oldest
    first), style gate]``: the style reward is added only where the gate is set (> 0.5), and every window is banked as
    a policy sample for the discriminator.

    Parameters
    ----------
    class_name : str
        The algorithm class rsl_rl resolves; leave at the default.
    amp_motion_file : str
        Asset path (resolved by ``eden.utils.assets.get_asset_path``) of the expert ``.npz`` bundle: ``motion_<i>``
        arrays of shape (T, frame_dim) plus ``fps`` and ``num_motions``. ``frame_dim`` must equal the joint-position
        width of the AMP observation group.
    amp_obs_group : str
        Observation group carrying the window and the gate.
    amp_num_frames : int
        Frames per window; must equal the group's joint-position ``history_length``.
    amp_reward_coef : float
        Scale inside the LSGAN style reward ``coef * clamp(1 - 0.25 (d - 1)^2, 0)``.
    amp_reward_scale : float
        Extra multiplier on the style reward before it is added to the env reward.
    amp_replay_size : int
        Capacity of the policy-window replay buffer.
    amp_discr_hidden_dims : list[int]
        Hidden widths of the spectral-norm discriminator MLP.
    amp_grad_pen_scale : float
        Gradient-penalty weight at the expert samples.
    amp_trunk_weight_decay : float
        Weight decay of the discriminator trunk.
    amp_head_weight_decay : float
        Weight decay of the discriminator head.
    amp_warm_start : bool
        Allow loading a checkpoint without AMP state (one trained as plain PPO): its actor-critic loads and the
        discriminator starts fresh. Off, such a checkpoint is refused rather than resumed under a changed reward.
    """

    class_name: str = _AMP_PPO_CLASS_NAME
    amp_motion_file: str
    amp_obs_group: str = "amp"
    amp_num_frames: int = 5
    amp_reward_coef: float = 5.0
    amp_reward_scale: float = 0.02
    amp_replay_size: int = 100_000
    amp_discr_hidden_dims: list[int] = [256, 256]
    amp_grad_pen_scale: float = 5.0
    amp_trunk_weight_decay: float = 1e-3
    amp_head_weight_decay: float = 1e-1
    amp_warm_start: bool = False

    def for_inference(self) -> RslRlPpoAlgorithmOptions:
        """Stock PPO with the same PPO fields — what replays an AMP checkpoint (the AMP state only shapes training)."""
        fields = {name: getattr(self, name) for name in RslRlPpoAlgorithmOptions.model_fields if name != "class_name"}
        return RslRlPpoAlgorithmOptions(**fields)

    def preflight(self) -> None:
        """Load the expert clips (``eden train`` calls this before building the env), so a bad file fails first."""
        from eden.learning.wrappers.rsl_rl.amp_ppo.amp import AmpMotionSampler

        AmpMotionSampler(self.amp_motion_file, self.amp_num_frames, "cpu")
