"""The world model as an environment: imagined, differentiable rollouts that follow the reference motion."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

from eden.learning.wrappers.rsl_rl.supermimic.representation import motion_goal
from eden.utils.geom import inv_transform_by_quat

if TYPE_CHECKING:
    from collections.abc import Callable

    from eden.learning.wrappers.rsl_rl.supermimic.goals import GoalSpec
    from eden.learning.wrappers.rsl_rl.supermimic.models import Dynamics
    from eden.learning.wrappers.rsl_rl.supermimic.replay_buffer import ReplayBuffer
    from eden.learning.wrappers.rsl_rl.supermimic.representation import GenCoordRepresentation, GoalWindow
    from eden.utils.motion_reference.library import MotionLibrary


def actor_input(
    goals: dict[str, torch.Tensor], history: torch.Tensor, critic: torch.Tensor | None = None
) -> TensorDict:
    """The actor's input: the observation ``history`` ``(B, H, obs_dim)``, one ``goal_<name>`` per goal interface and,
    in imagination, the critic's input.
    """
    obs = {"history": history} | {f"goal_{name}": goal for name, goal in goals.items()}
    if critic is not None:
        obs["critic"] = critic
    return TensorDict(obs, batch_size=[history.shape[0]], device=history.device)


class WorldModelEnv:
    """Steps the learned dynamics, advances the reference one frame per step and grades the predicted state.

    A step grades the predicted state against the reference frame it advances to, with the reward and termination
    the simulator computes. Past a clip's end the reference holds the last frame, velocities included; the simulator
    truncates the episode there instead, so the steps of a rollout after the clip's end have no counterpart in it.
    Termination is soft: the anchor height and tilt errors relative to the task's termination thresholds pass
    through a sigmoid of width ``termination_softness``.

    Parameters
    ----------
    dynamics : Dynamics
        The world model.
    representation : GenCoordRepresentation
        Its state, observation and reward.
    library : MotionLibrary
        The reference motions.
    fk : torch.nn.Module
        Forward kinematics of the tracked bodies, the anchor first.
    goal_specs : list[GoalSpec]
        The goal interfaces trained, the motion interface first.
    goal_window : GoalWindow
        The reference frames the motion goal holds.
    observation : {"state", "proprio"}
        What the actor reads.
    history_length : int
        Proprioception frames the actor reads.
    anchor_pos_threshold, anchor_ori_threshold : float
        The task's anchor height and tilt termination thresholds.
    termination_softness : float
        Width of the termination sigmoid, in units of the thresholds.
    encoder_bias : Callable[[torch.Tensor], torch.Tensor | None] | None
        Each joint's encoder bias from a normalized simulation context (``SimContext.encoder_bias``), added to the
        joint reading of the imagined proprioception. ``None`` for a task without an encoder bias.
    """

    def __init__(
        self,
        dynamics: Dynamics,
        representation: GenCoordRepresentation,
        library: MotionLibrary,
        fk: torch.nn.Module,
        goal_specs: list[GoalSpec],
        goal_window: GoalWindow,
        observation: str,
        history_length: int,
        anchor_pos_threshold: float,
        anchor_ori_threshold: float,
        termination_softness: float,
        encoder_bias: Callable[[torch.Tensor], torch.Tensor | None] | None = None,
    ):
        if goal_specs[0].name != "motion":
            raise ValueError(f"the first goal interface must be 'motion', got {[s.name for s in goal_specs]}.")
        self.dynamics = dynamics
        self.representation = representation
        self.library = library
        self.fk = fk
        self.goal_specs = goal_specs
        self.goal_window = goal_window
        self.observation = observation
        self.history_length = history_length
        self.anchor_pos_threshold = anchor_pos_threshold
        self.anchor_ori_threshold = anchor_ori_threshold
        self.termination_softness = termination_softness
        self.encoder_bias = encoder_bias
        # Gravity's direction, on the model's device: created inside a step it would be a host-to-device copy in the
        # captured CUDA graph of the policy update.
        self._down = torch.tensor([0.0, 0.0, -1.0], device=representation.default_joint_pos.device)

    def reference_bodies(self, frames: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """World ``(pos, quat, lin_vel, ang_vel)`` of the reference's tracked bodies at ``frames``."""
        lib = self.library
        return self.fk(
            lib.root_pos_w[frames],
            lib.root_quat_w[frames],
            lib.joint_pos[frames],
            lib.root_lin_vel_w[frames],
            lib.root_ang_vel_w[frames],
            lib.joint_vel[frames],
        )

    def reset(self, start: TensorDict) -> TensorDict:
        """Start every rollout at a replay-buffer state."""
        rep = self.representation
        self.aux_state = rep.aux_state(start["state"]).detach()
        self.rep_state = rep.represent(start["state"]).detach()
        self.last_action = start["prev_action"].detach()
        self.sim_ctx = start["sim_ctx"]
        self.frames = start["frame"]
        self._update_goals()
        if self.history_length > 1:
            self._past = self._warm_start_history(start)
        self._latest = self._sense()
        return self.observations()

    def step(self, action: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor]:
        """``(observations, reward, termination probability)`` after ``action``."""
        self.aux_state, self.rep_state = self.dynamics(self.rep_state, action, self.aux_state, self.sim_ctx)
        self.frames = torch.minimum(self.frames + 1, self.library.last_frame(self.frames))
        self._update_goals()
        target = self.reference_bodies(self.frames)
        reward = self.representation.reward(self.aux_state, self.rep_state, action, self.last_action, target)
        term_prob = self._termination_probability(target)
        self.last_action = action
        if self.history_length > 1:
            self._past = torch.cat([self._past[:, 1:], self._latest.unsqueeze(1)], dim=1)
        self._latest = self._sense()
        return self.observations(), reward, term_prob

    def observations(self) -> TensorDict:
        history = self._latest.unsqueeze(1)
        if self.history_length > 1:
            history = torch.cat([self._past, history], dim=1)
        critic = torch.cat([self.rep_state, self._wm_goal, self.sim_ctx], dim=-1)
        return actor_input(self._goals, history, critic)

    def _sense(self) -> torch.Tensor:
        if self.observation == "state":
            return self.rep_state
        bias = self.encoder_bias(self.sim_ctx) if self.encoder_bias is not None else None
        reading = self.representation.proprio(self.rep_state, self.last_action, bias)
        return self.representation.add_proprio_noise(reading)

    def _update_goals(self) -> None:
        self._wm_goal, actor_goal = motion_goal(self.library, self.frames, self.aux_state, self.goal_window)
        self._goals = {"motion": actor_goal}
        for spec in self.goal_specs[1:]:
            self._goals[spec.name] = spec.goal(self.library, self.frames, self.aux_state)

    def _termination_probability(self, target: tuple[torch.Tensor, ...]) -> torch.Tensor:
        ref_pos, ref_quat = target[0][:, 0], target[1][:, 0]
        root_pos, root_quat = self.aux_state[..., :3], self.aux_state[..., 3:7]
        height_error = (ref_pos[..., 2] - root_pos[..., 2]).abs()
        down = self._down.expand(*ref_quat.shape[:-1], 3)
        tilt_error = (
            inv_transform_by_quat(down, ref_quat)[..., 2] - inv_transform_by_quat(down, root_quat)[..., 2]
        ).abs()
        error = torch.maximum(height_error / self.anchor_pos_threshold, tilt_error / self.anchor_ori_threshold)
        return torch.sigmoid((error - 1.0) / self.termination_softness)

    def _warm_start_history(self, start: TensorDict) -> torch.Tensor:
        """The sampled predecessor frames, noised; slots before the episode start share the oldest real frame's
        noise, as the simulator's history repeats its first frame.
        """
        noisy = self.representation.add_proprio_noise(start["sensor_window"][:, :-1])
        pad = start["pad_mask"][:, :-1]
        first = (~pad).int().argmax(dim=1)
        shared = noisy[torch.arange(len(noisy), device=noisy.device), first]
        return torch.where(pad.unsqueeze(-1), shared.unsqueeze(1), noisy)


def sample_starts(buffer: ReplayBuffer, history_length: int, batch_size: int) -> TensorDict:
    """Start states for :meth:`WorldModelEnv.reset`. With a history, each also carries ``sensor_window``, the frames
    ending at it, and ``pad_mask``, which of them repeat the episode's first frame.
    """
    if history_length == 1:
        return buffer.sample(1, batch_size).squeeze(1)
    window, padded = buffer.sample_causal(history_length, batch_size)
    return window[:, -1].set("sensor_window", window["actor_obs"]).set("pad_mask", padded)
