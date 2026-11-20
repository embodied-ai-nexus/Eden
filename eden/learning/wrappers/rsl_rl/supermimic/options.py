"""Options of the SuperMimic learner. Declarative only: importing this module imports no learner code."""

from __future__ import annotations

from typing import Literal

from pydantic import model_validator

from eden.options.learning.rsl_rl.rl import RslRlBaseRunnerOptions
from eden.options.options import ConfigurableOptions


class WorldModelOptions(ConfigurableOptions):
    """The dynamics model and its supervised fit on replay-buffer windows.

    Parameters
    ----------
    hidden_dims : tuple[int, ...]
        MLP widths.
    lr : float
        Learning rate.
    num_batches : int
        Optimizer steps per update, one batch each.
    batch_size : int
        Windows per batch.
    horizon : int
        Steps per window, rolled out autoregressively from the window's first state.
    nominal_error_root_pos, nominal_error_root_ori, nominal_error_joint_pos, nominal_error_joint_vel : float
        Nominal error of each channel group of the state loss; a group is weighted by ``1 / nominal_error^2``.
    """

    hidden_dims: tuple[int, ...] = (2048, 2048, 2048)
    lr: float = 2e-4
    num_batches: int = 10
    batch_size: int = 2048
    horizon: int = 8
    nominal_error_root_pos: float = 0.01
    nominal_error_root_ori: float = 0.02
    nominal_error_joint_pos: float = 0.1
    nominal_error_joint_vel: float = 3.14


class PolicyOptions(ConfigurableOptions):
    """The actor, the critic and their update through imagined rollouts of the world model.

    Parameters
    ----------
    algorithm : {"actor_critic", "bptt"}
        ``"actor_critic"`` bootstraps the imagined return with a critic fit to TD(lambda) targets of the same
        rollouts; ``"bptt"`` uses the imagined return alone.
    lr : float
        Learning rate of the actor and the critic.
    num_batches : int
        Actor steps per update, one imagined batch each.
    batch_size : int
        Imagined rollouts per batch.
    horizon : int
        Imagined steps per rollout.
    gamma, lam : float
        Discount and TD(lambda) parameter (``bptt``: ``gamma`` only, default 1).
    survival_masking : bool
        Weight each imagined reward by the probability of not having terminated before it.
    beta2 : float
        Adam's second-moment decay for the actor.
    grad_clip_value : float
        Element-wise clip of the actor's gradient, after its global norm is clipped to 1.
    termination_softness : float
        Width of the sigmoid that turns the task's termination thresholds into a termination probability.
    latent_dim : int
        Width of the tanh-bounded goal latent.
    actor_hidden_dims, critic_hidden_dims : tuple[int, ...]
        MLP widths of each encoder and the decoder, and of the critic.
    latent_bc_weight : float
        Weight of the loss pulling each non-motion encoder's latent towards the motion encoder's.
    observation : {"state", "proprio"}
        The actor reads the model state (``"state"``) or the deployable proprioception with sensor noise
        (``"proprio"``).
    history_length : int
        Proprioception frames the actor reads; above 1 needs ``observation="proprio"``.
    slot_mix : dict[str, float]
        Share of the batch each goal interface trains on (``"motion"``, ``"velocity"``).
    """

    algorithm: Literal["actor_critic", "bptt"] = "actor_critic"
    lr: float = 2e-4
    num_batches: int = 1
    batch_size: int = 4096
    horizon: int = 32
    gamma: float = 0.99
    lam: float = 0.95
    survival_masking: bool = True
    beta2: float = 0.95
    grad_clip_value: float = 0.1
    termination_softness: float = 0.05
    latent_dim: int = 32
    actor_hidden_dims: tuple[int, ...] = (2048, 1024, 512, 256)
    critic_hidden_dims: tuple[int, ...] = (256, 128)
    latent_bc_weight: float = 0.1
    observation: Literal["state", "proprio"] = "state"
    history_length: int = 1
    slot_mix: dict[str, float] = {"motion": 1.0, "velocity": 0.0}

    @model_validator(mode="after")
    def _check(self) -> PolicyOptions:
        if self.history_length > 1 and self.observation != "proprio":
            raise ValueError(
                f"history_length={self.history_length} stacks proprioception frames, so it needs "
                "observation='proprio' (got observation='state')."
            )
        if self.termination_softness <= 0.0:
            raise ValueError(f"termination_softness must be positive, got {self.termination_softness}.")
        return self


class VelocityGoalOptions(ConfigurableOptions):
    """The velocity-command goal interface.

    Parameters
    ----------
    lin_vel_x_range, lin_vel_y_range, ang_vel_z_range : tuple[float, float]
        Command ranges, body frame.
    window_frames : int
        Reference frames averaged around the current one; 0 or 1 reads the current frame only.
    command_noise : tuple[float, float, float]
        Half-width of the uniform noise added to each command component every step.
    """

    lin_vel_x_range: tuple[float, float] = (-1.5, 1.5)
    lin_vel_y_range: tuple[float, float] = (-0.5, 0.5)
    ang_vel_z_range: tuple[float, float] = (-3.14, 3.14)
    window_frames: int = 25
    command_noise: tuple[float, float, float] = (0.5, 0.5, 0.785)


class ReturnSamplerOptions(ConfigurableOptions):
    """Reset sampling that concentrates resets on the motion bins where the policy earns a low return.

    Parameters
    ----------
    enabled : bool
        Drive the motion term's reset distribution; ``False`` leaves it uniform.
    gamma : float
        Discount of the measured return.
    horizon : int
        Env steps between updates of the bin values.
    memory : int
        Samples a bin's value averages over before older samples start to fade.
    steepness : float
        Softmax gain on ``value / v_max``.
    explore : float
        Share of resets spread uniformly over the bins.
    keep : float
        Share of resets on the bins in proportion to the summed squared focus they received.
    """

    enabled: bool = True
    gamma: float = 0.99
    horizon: int = 20
    memory: int = 256
    steepness: float = 25.0
    explore: float = 0.5
    keep: float = 0.1

    @model_validator(mode="after")
    def _check(self) -> ReturnSamplerOptions:
        if not 0.0 < self.gamma < 1.0:
            raise ValueError(f"return sampler gamma must be in (0, 1), got {self.gamma}.")
        if self.explore + self.keep >= 1.0:
            raise ValueError(f"explore ({self.explore}) + keep ({self.keep}) must leave a share for the focus.")
        return self


class SimContextOptions(ConfigurableOptions):
    """The randomized parameters the world model is conditioned on, named by the task's event terms.

    Parameters
    ----------
    foot_friction_event : str | None
        A ``RandomizeFrictionRatio`` event; its first link's friction is read (one draw shared by the feet).
    com_event : str | None
        A ``RandomizeComShift`` event on one link.
    encoder_bias_event : str | None
        A ``RandomizeDofsPosOffset`` event on the joint position controller.
    """

    foot_friction_event: str | None = "foot_friction"
    com_event: str | None = "base_com"
    encoder_bias_event: str | None = "encoder_bias"


class SuperMimicRunnerOptions(RslRlBaseRunnerOptions):
    """SuperMimic: a motion tracker trained through one learned, differentiable world model.

    Each iteration steps the simulator ``num_steps_per_env`` times with the deterministic policy, storing every
    transition in a replay buffer. Every ``num_envs / (num_batches * batch_size * horizon) * replay_ratio`` env
    steps on average, the world model fits replay-buffer windows and the policy improves on rollouts imagined by the
    frozen world model from replay-buffer states.

    Parameters
    ----------
    class_name : str
        The runner.
    motion_term : str
        The task's :class:`~eden.tasks.supermimic.motion.SuperMimicMotion` term.
    action_term : str
        The joint position controller.
    actor_observation_group : str
        The proprioception group, ``[projected_gravity, base_ang_vel, q - q0 (biased), qd, last_action]``.
    state_observation_group : str
        The group holding the robot's full state ``[root_pos, root_quat, q, root_lin_vel_w, root_ang_vel_w, qd]``.
    anchor_pos_termination, anchor_ori_termination : str
        Termination terms whose thresholds set the world model's termination probability.
    clip_end_termination : str
        Termination term that ends an episode at its clip's last frame.
    buffer_size : int
        Replay-buffer steps per env.
    replay_ratio : float
        Replayed world-model transitions per collected one.
    """

    class_name: Literal["SuperMimicRunner"] = "SuperMimicRunner"
    num_steps_per_env: int = 10
    max_iterations: int = 100_000
    save_interval: int = 100
    motion_term: str = "default"
    action_term: str = "joint_pos"
    actor_observation_group: str = "policy"
    state_observation_group: str = "full_state"
    anchor_pos_termination: str = "anchor_pos"
    anchor_ori_termination: str = "anchor_ori"
    clip_end_termination: str = "clip_end"
    buffer_size: int = 1000
    replay_ratio: float = 16.0
    world_model: WorldModelOptions = WorldModelOptions()
    policy: PolicyOptions = PolicyOptions()
    velocity_goal: VelocityGoalOptions = VelocityGoalOptions()
    return_sampler: ReturnSamplerOptions = ReturnSamplerOptions()
    sim_context: SimContextOptions = SimContextOptions()
