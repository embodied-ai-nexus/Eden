"""The SuperMimic training loop: collect with the policy, fit the world model, improve the policy in imagination."""

from __future__ import annotations

import os
import statistics
import time
from collections import deque
from typing import TYPE_CHECKING, Any

import torch
from tensordict import TensorDict

import eden as en
from eden.envs.wrappers.rsl_rl_env import EDEN_RUNNER_MIXINS
from eden.learning.wrappers.rsl_rl.supermimic.goals import motion_goal_spec, velocity_goal_spec
from eden.learning.wrappers.rsl_rl.supermimic.learners import PolicyLearner, WorldModelLearner, load_optimizer_state
from eden.learning.wrappers.rsl_rl.supermimic.models import Dynamics, MultiEncoderActor, build_mlp
from eden.learning.wrappers.rsl_rl.supermimic.options import SuperMimicRunnerOptions
from eden.learning.wrappers.rsl_rl.supermimic.replay_buffer import FieldSpec, ReplayBuffer
from eden.learning.wrappers.rsl_rl.supermimic.representation import GenCoordRepresentation, GoalWindow
from eden.learning.wrappers.rsl_rl.supermimic.reset_sampler import ReturnSampler
from eden.learning.wrappers.rsl_rl.supermimic.sim_context import SimContext
from eden.learning.wrappers.rsl_rl.supermimic.task_terms import (
    check_joint_and_anchor_order,
    check_motion_term,
    check_observation_groups,
    task_reward_spec,
    termination_threshold,
)
from eden.learning.wrappers.rsl_rl.supermimic.world_model_env import WorldModelEnv, actor_input, sample_starts

if TYPE_CHECKING:
    from eden.envs.wrappers.rsl_rl_env import RslRlVecEnvWrapper


class _Ratio:
    """How many updates are due at each env step for ``ratio`` updates per step on average."""

    def __init__(self, ratio: float):
        self._ratio = ratio
        self._prev: float | None = None

    def __call__(self, step: int) -> int:
        if self._prev is None:
            self._prev = step
            return 1
        repeats = int((step - self._prev) * self._ratio)
        self._prev += repeats / self._ratio
        return repeats


def _same_device(a: torch.device, b: torch.device) -> bool:
    """Whether ``a`` and ``b`` name one device: ``cuda`` without an index is the current CUDA device."""
    if a.type != b.type:
        return False
    if a.type != "cuda":
        return True
    current = torch.cuda.current_device()
    return (current if a.index is None else a.index) == (current if b.index is None else b.index)


class _Logger:
    """Episode statistics of the simulator and the learners' losses, to the console and TensorBoard or W&B.

    Holds what Eden's runner mixins read: ``log_dir``, ``disable_logs``, ``rewbuffer``, ``lenbuffer``,
    ``tot_timesteps`` and ``tot_time``. The writer opens on :meth:`open`, which only a training run calls.
    """

    def __init__(self, log_dir: str | None, options: SuperMimicRunnerOptions, num_envs: int, device: torch.device):
        self.log_dir = log_dir
        self._options = options
        self.disable_logs = log_dir is None or en.distributed.get_global_rank() != 0
        self.rewbuffer: deque[float] = deque(maxlen=100)
        self.lenbuffer: deque[float] = deque(maxlen=100)
        self.tot_timesteps = 0
        self.tot_time = 0.0
        self._reward_sum = torch.zeros(num_envs, device=device)
        self._length = torch.zeros(num_envs, device=device)
        self._episode_logs: list[dict] = []
        self._writer = None
        self._wandb = None
        self._owns_wandb_run = False

    def open(self) -> None:
        """Open the W&B run or the TensorBoard writer, once."""
        if self.disable_logs or self._writer is not None or self._wandb is not None:
            return
        options = self._options
        if options.logger == "wandb":
            import wandb

            run_name = options.run_name or None
            self._owns_wandb_run = wandb.run is None
            self._wandb = wandb.run or wandb.init(
                project=options.wandb_project, name=run_name, dir=self.log_dir, config=options.dict()
            )
        else:
            from torch.utils.tensorboard import SummaryWriter

            self._writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

    def process_env_step(self, rewards: torch.Tensor, dones: torch.Tensor, extras: dict) -> None:
        if self.disable_logs:
            return
        if extras.get("log"):
            self._episode_logs.append(extras["log"])
        self._reward_sum += rewards
        self._length += 1
        done = dones.nonzero(as_tuple=False)[:, 0]
        if done.numel():
            self.rewbuffer.extend(self._reward_sum[done].tolist())
            self.lenbuffer.extend(self._length[done].tolist())
            self._reward_sum[done] = 0
            self._length[done] = 0

    def log(self, it: int, total_it: int, steps: int, collect_time: float, learn_time: float, losses: dict) -> None:
        if self.disable_logs:
            return
        self.tot_timesteps += steps
        self.tot_time += collect_time + learn_time
        scalars = {f"Loss/{k}": float(v) for k, v in losses.items()}
        for key in dict.fromkeys(k for log in self._episode_logs for k in log):
            values = [
                torch.as_tensor(log[key], dtype=torch.float).reshape(-1) for log in self._episode_logs if key in log
            ]
            scalars[key if "/" in key else f"Episode/{key}"] = float(torch.cat(values).mean())
        self._episode_logs.clear()
        if self.rewbuffer:
            scalars["Train/mean_reward"] = statistics.mean(self.rewbuffer)
            scalars["Train/mean_episode_length"] = statistics.mean(self.lenbuffer)
        fps = int(steps / max(collect_time + learn_time, 1e-9))
        scalars |= {"Perf/total_fps": fps, "Perf/collection_time": collect_time, "Perf/learning_time": learn_time}
        if self._wandb is not None:
            self._wandb.log(scalars, step=it)
        else:
            for key, value in scalars.items():
                self._writer.add_scalar(key, value, it)
        width, pad = 80, 40
        lines = [f"{'#' * width}", f"\033[1m{f' Learning iteration {it}/{total_it} '.center(width)}\033[0m", ""]
        lines += [f"{f'{key}:':>{pad}} {value:.4f}" for key, value in scalars.items()]
        print("\n".join(lines))

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
        # A run the caller started before the runner (``wandb.run``) is theirs to finish.
        if self._wandb is not None and self._owns_wandb_run:
            self._wandb.finish()
        self._writer = self._wandb = None


class _SuperMimicRunner:
    """Trains a motion tracker through one learned, differentiable world model.

    One iteration steps every env ``num_steps_per_env`` times with the deterministic policy and stores each
    transition. Updates run at ``num_envs / (num_batches * batch_size * horizon) * replay_ratio`` per env step on
    average; one update fits the world model on ``num_batches`` batches of replay-buffer windows, then improves the
    policy on rollouts imagined from replay-buffer states.

    Parameters
    ----------
    env : RslRlVecEnvWrapper
        The task env, a ``supermimic`` config or one with the same motion term, observations and terms.
    train_cfg : dict | SuperMimicRunnerOptions
        The runner options.
    log_dir : str | None
        Where checkpoints and logs go; ``None`` for a runner that only runs a policy.
    device : str
        The training device.
    """

    #: Why ``eden export`` cannot write this runner's policy; the command exits with it before building anything.
    export_unsupported = (
        "eden export does not support the SuperMimic runner: its policy reads each env's reference frame from the "
        "motion library and, with policy.observation='state', the simulator's full state, neither of which an "
        "exported graph has. Replay the checkpoint with `eden inference --task supermimic --checkpoint <path>`."
    )

    def __init__(self, env: RslRlVecEnvWrapper, train_cfg: Any, log_dir: str | None = None, device: str = "cuda"):
        options = train_cfg if isinstance(train_cfg, SuperMimicRunnerOptions) else SuperMimicRunnerOptions(**train_cfg)
        self.env = env
        self.options = options
        self.cfg = options.dict()
        self.log_dir = log_dir
        self.device = torch.device(env.device if device == "auto" else device)
        if not _same_device(self.device, torch.device(env.device)):
            raise ValueError(
                f"The SuperMimic runner trains on the simulator's device: runner_options.device resolves to "
                f"{str(self.device)!r}, the env is on {str(torch.device(env.device))!r}. Its motion library and "
                "simulation context are read on the env's device every step. Set runner_options.device='auto'."
            )
        self.current_learning_iteration = 0
        base = env.unwrapped
        self.num_envs = env.num_envs
        self.motion = check_motion_term(base, options.motion_term)
        controller = base.action_manager.get_term(options.action_term)
        robot = base.entities[controller.entity_name]
        check_joint_and_anchor_order(controller, robot, self.motion, options.action_term, options.motion_term)
        if options.clip_end_termination not in base.termination_manager.active_terms:
            raise ValueError(
                f"runner_options.clip_end_termination={options.clip_end_termination!r}, but the task has no such "
                f"termination term. Its terms: {list(base.termination_manager.active_terms)}."
            )

        self._build_world_model(base, controller, robot)
        check_observation_groups(base, options, controller.entity_name, self.representation)
        self._build_policy(base)
        self.sampler = self._build_return_sampler()

        self._stores_sensor = options.policy.observation == "proprio" and options.policy.history_length > 1
        #: Allocated by :meth:`learn`; a runner built to replay a checkpoint holds no transitions.
        self.replay_buffer: ReplayBuffer | None = None
        self._history: torch.Tensor | None = None
        self._pending: list[tuple[torch.Tensor, torch.Tensor, TensorDict]] = []
        #: Reset bins of a loaded checkpoint whose return sampler does not fit this run's library (:meth:`load`).
        self._stale_sampler_bins: int | None = None
        self.logger = _Logger(log_dir, options, self.num_envs, self.device)

    def _build_world_model(self, base, controller, robot) -> None:
        """The state representation, the simulation context and the dynamics model with its learner."""
        options, num_actions = self.options, self.env.num_actions
        dofs = robot.find_named_dofs(list(controller.dofs_name))
        reward = task_reward_spec(base, options.reward_terms, self.motion.fk.link_names)
        self.representation = GenCoordRepresentation(
            fk=self.motion.fk,
            default_joint_pos=robot.get_default_dofs_pos(dofs)[0],
            soft_joint_pos_limits=robot.soft_dofs_pos_limits[0, dofs.columns],
            dt=base.dt,
            decimation=base.env_options.decimation,
            num_actions=num_actions,
            reward=reward,
            proprio_noise=options.policy.proprio_noise,
        ).to(self.device)
        self.sim_context = SimContext(base, options.sim_context, options.action_term)
        self.max_step_reward = base.dt * reward.max_step_rate
        action_scale = torch.as_tensor(controller.effective_scale, device=self.device, dtype=torch.float)
        self.dynamics = Dynamics(
            self.representation,
            num_actions,
            self.sim_context.dim,
            action_scale.expand(1, num_actions)[0],
            self.representation.default_joint_pos,
            options.world_model.hidden_dims,
        ).to(self.device)
        self.world_model_learner = WorldModelLearner(
            self.dynamics, self.representation, options.world_model, str(self.device)
        )

    def _build_policy(self, base) -> None:
        """The actor, the critic (``actor_critic`` only), the imagination env and the policy learner."""
        options, policy, representation = self.options, self.options.policy, self.representation
        anchor = self.motion.fk.link_names[0]
        window = GoalWindow(policy.goal_frames, policy.goal_frame_stride)
        specs = [
            motion_goal_spec(representation.num_joints, window),
            velocity_goal_spec(options.velocity_goal, self.device),
        ]
        self.actor = MultiEncoderActor(
            obs_dim=representation.rep_state_dim if policy.observation == "state" else representation.proprio_dim,
            action_dim=self.env.num_actions,
            goal_specs=specs,
            slot_mix=policy.slot_mix,
            latent_dim=policy.latent_dim,
            hidden_dims=policy.actor_hidden_dims,
            latent_bc_weight=policy.latent_bc_weight,
            history_length=policy.history_length,
        ).to(self.device)
        self.active_specs = [spec for spec in specs if policy.slot_mix.get(spec.name, 0.0) > 0.0]
        self.critic = None
        if policy.algorithm == "actor_critic":
            critic_dim = representation.rep_state_dim + window.dims(representation.num_joints)[0] + self.sim_context.dim
            self.critic = build_mlp(critic_dim, policy.critic_hidden_dims, 1).to(self.device)
        self.wm_env = WorldModelEnv(
            dynamics=self.dynamics,
            representation=representation,
            library=self.motion.library,
            fk=self.motion.fk,
            goal_specs=self.active_specs,
            goal_window=window,
            observation=policy.observation,
            history_length=policy.history_length,
            anchor_pos_threshold=termination_threshold(
                base, options.anchor_pos_termination, "anchor_pos_termination", "bad_anchor_pos_z_only", anchor
            ),
            anchor_ori_threshold=termination_threshold(
                base, options.anchor_ori_termination, "anchor_ori_termination", "bad_anchor_ori", anchor
            ),
            termination_softness=policy.termination_softness,
            encoder_bias=self.sim_context.encoder_bias,
            compile=self.device.type == "cuda",
        )
        self.policy_learner = PolicyLearner(
            self.actor, self.critic, self.wm_env, policy, self.max_step_reward, str(self.device)
        )

    def _build_return_sampler(self) -> ReturnSampler | None:
        options = self.options
        if not options.return_sampler.enabled:
            return None
        if self.motion.reset_sampling != "external":
            raise ValueError(
                f"The return sampler sets the reset distribution of motion term {options.motion_term!r}, which "
                f"samples resets with reset_sampling={self.motion.reset_sampling!r}. Set its reset_sampling to "
                "'external', or disable the sampler (runner_options.return_sampler.enabled=False)."
            )
        return ReturnSampler(
            options.return_sampler,
            self.motion.bin_count,
            self.motion.library.num_frames,
            self.num_envs,
            self.max_step_reward,
            self.device,
        )

    def _make_replay_buffer(self) -> ReplayBuffer:
        num_actions = self.env.num_actions
        schema = {
            "state": FieldSpec(self.representation.full_state_dim),
            "action": FieldSpec(num_actions),
            "prev_action": FieldSpec(num_actions),
            "frame": FieldSpec(dtype=torch.long),
            "terminated": FieldSpec(dtype=torch.bool),
            "truncated": FieldSpec(dtype=torch.bool),
            "sim_ctx": FieldSpec(self.sim_context.dim),
        }
        if self._stores_sensor:
            schema["actor_obs"] = FieldSpec(self.representation.proprio_dim)
        return ReplayBuffer(schema, self.options.buffer_size, self.num_envs, self.device)

    def _observe(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(full state, proprioception, frames, sim context)`` of every env now."""
        o = self.options
        return (
            obs[o.state_observation_group].to(self.device),
            obs[o.actor_observation_group].to(self.device),
            self.env.unwrapped.motion_manager.get_library_frame(o.motion_term),
            self.sim_context.read(),
        )

    def _sensed(self, sensor: torch.Tensor, training: bool) -> torch.Tensor:
        """The simulator's proprioception as the actor reads it: with sensor noise while training, as it is at
        inference.
        """
        return self.representation.add_proprio_noise(sensor) if training else sensor

    def _policy_input(
        self, state: torch.Tensor, sensor: torch.Tensor, frames: torch.Tensor, training: bool = True
    ) -> TensorDict:
        """The actor's input. While training it holds the goal of every interface with a share of the batch; at
        inference the motion goal alone.
        """
        aux = self.representation.aux_state(state)
        specs = self.active_specs if training else [s for s in self.active_specs if s.name == "motion"]
        goals = {spec.name: spec.goal(self.motion.library, frames, aux) for spec in specs}
        if self.options.policy.observation == "state":
            history = self.representation.represent(state).unsqueeze(1)
        elif self.options.policy.history_length == 1:
            history = self._sensed(sensor, training).unsqueeze(1)
        else:
            history = self._history
        return actor_input(goals, history)

    def _seed_history(self, obs: TensorDict, training: bool = True) -> None:
        if self._stores_sensor:
            seed = self._sensed(obs[self.options.actor_observation_group].to(self.device), training)
            self._history = seed.unsqueeze(1).repeat(1, self.options.policy.history_length, 1)

    def _advance_history(self, sensor: torch.Tensor, dones: torch.Tensor, training: bool = True) -> None:
        """Append the next frame; an env that just reset repeats it over its whole history."""
        if not self._stores_sensor:
            return
        frame = self._sensed(sensor, training)
        shifted = torch.cat([self._history[:, 1:], frame.unsqueeze(1)], dim=1)
        self._history = torch.where(dones.bool().view(-1, 1, 1), frame.unsqueeze(1), shifted)

    def _collect(self, obs: TensorDict) -> TensorDict:
        """One env step with the policy, stored in the replay buffer; returns the next observations."""
        base = self.env.unwrapped
        with torch.inference_mode():
            state, sensor, frames, sim_ctx = self._observe(obs)
            actor_in = self._policy_input(state, sensor, frames)
            self.actor.eval()
            actions = self.actor(actor_in)
            prev_action = base.action_manager.action.clone()
            next_obs, rewards, dones, extras = self.env.step(actions)
            truncated = extras["time_outs"].to(self.device, torch.bool)
            step = {
                "state": state,
                "action": actions,
                "prev_action": prev_action,
                "frame": frames,
                "terminated": dones.bool() & ~truncated,
                "truncated": truncated,
                "sim_ctx": sim_ctx,
            }
            if self._stores_sensor:
                step["actor_obs"] = sensor
            self.replay_buffer.insert(TensorDict(step, batch_size=[self.num_envs]))
            self._pending.append((state.clone(), actions.clone(), actor_in.clone()))
            self._advance_history(next_obs[self.options.actor_observation_group].to(self.device), dones)
            self.logger.process_env_step(rewards, dones, extras)
            if self.sampler is not None:
                clip_end = base.termination_manager.get_term_dones(self.options.clip_end_termination)
                weights = self.sampler.record(
                    frames, rewards, dones.bool() & ~truncated, clip_end, truncated & ~clip_end
                )
                if weights is not None:
                    self.motion.set_reset_distribution(weights)
        self._flush_normalization()
        return next_obs

    def _raise_if_not_finite(self) -> None:
        """Raise if this iteration collected a non-finite value (``runner_options.check_for_nan``).

        Every collected state and action is folded into the dynamics' running normalizers, and every reward into
        the return sampler's bin values, where one non-finite value stays. Reading those few statistics once per
        iteration finds it without an operation or a host sync in the collection loop.
        """
        dynamics = self.dynamics
        tracked = {
            "state": (dynamics.state_normalizer.mean, dynamics.state_normalizer.var),
            "action": (dynamics.action_normalizer.mean, dynamics.action_normalizer.var),
        }
        if self.sampler is not None:
            tracked["reward"] = (self.sampler.value,)
        finite = torch.stack([torch.stack([torch.isfinite(t).all() for t in ts]).all() for ts in tracked.values()])
        bad = [name for name, ok in zip(tracked, finite.tolist(), strict=True) if not ok]
        if bad:
            raise ValueError(
                f"Non-finite values in the collected {bad} during iteration {self.current_learning_iteration}: "
                "they entered the replay buffer and the running statistics. The simulation or the policy "
                "diverged; runner_options.check_for_nan=False skips this check."
            )

    def _flush_normalization(self) -> None:
        states, actions, actor_inputs = zip(*self._pending, strict=True)
        self.world_model_learner.update_normalization(torch.cat(states), torch.cat(actions))
        self.actor.train()
        for actor_in in actor_inputs:
            self.actor.update_normalization(actor_in)
        self._pending.clear()

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        """Run ``num_learning_iterations`` iterations from the current one. Episodes start where the reset sampler
        puts them, so ``init_at_random_ep_len`` has no effect.
        """
        del init_at_random_ep_len
        options, world_model, policy = self.options, self.options.world_model, self.options.policy
        if self._stale_sampler_bins is not None:
            raise ValueError(
                f"The loaded checkpoint's return sampler holds {self._stale_sampler_bins} reset bins, but this "
                f"run's motion library has {self.sampler.bin_count} ({self.motion.library.num_frames} frames): the "
                "checkpoint was trained on another library, clip list or reset_bin_length_s. Resume on the library "
                "it was trained on, or set runner_options.return_sampler.enabled=False to continue with uniform "
                "resets."
            )
        if self.replay_buffer is None:
            self.replay_buffer = self._make_replay_buffer()
        else:
            # A second call: the env reset below ends the episodes of the steps already stored.
            self.replay_buffer.end_episodes()
            if self.sampler is not None:
                self.sampler.restart_horizon()
        self.logger.open()
        try:
            obs, _ = self.env.reset()
            self._seed_history(obs)
            updates_due = _Ratio(
                self.num_envs
                / (world_model.num_batches * world_model.batch_size * world_model.horizon)
                * options.replay_ratio
            )
            start = self.current_learning_iteration
            total = start + num_learning_iterations
            step = start * options.num_steps_per_env
            for it in range(start, total):
                t0, learn_time = time.time(), 0.0
                losses: dict[str, torch.Tensor] = {}
                for _ in range(options.num_steps_per_env):
                    obs = self._collect(obs)
                    for _ in range(updates_due(step)):
                        # With few envs, every stored window may hold an episode end.
                        if not self.replay_buffer.has_window(world_model.horizon):
                            continue
                        t1 = time.time()
                        windows = (
                            self.replay_buffer.sample(world_model.horizon, world_model.batch_size)
                            for _ in range(world_model.num_batches)
                        )
                        losses.update(self.world_model_learner.update(windows))
                        starts = (
                            sample_starts(self.replay_buffer, policy.history_length, policy.batch_size)
                            for _ in range(policy.num_batches)
                        )
                        losses.update(self.policy_learner.update(starts))
                        learn_time += time.time() - t1
                    step += 1
                if options.check_for_nan:
                    self._raise_if_not_finite()
                self.current_learning_iteration = it + 1
                if self.sampler is not None:
                    losses["reset_sampler/top1_prob"] = self.motion.reset_distribution.max()
                env_steps = options.num_steps_per_env * self.num_envs
                self.logger.log(it, total, env_steps, time.time() - t0 - learn_time, learn_time, losses)
                if self.log_dir is not None and (it + 1) % options.save_interval == 0:
                    self.save(os.path.join(self.log_dir, f"model_{it + 1}.pt"))
            if self.log_dir is not None:
                self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))
        finally:
            self.logger.close()

    def _layout(self) -> dict[str, Any]:
        """What fixes the shapes of the networks, stored with a checkpoint and compared when one is loaded."""
        policy, representation = self.options.policy, self.representation
        return {
            "runner_options.policy.observation": policy.observation,
            "runner_options.policy.history_length": policy.history_length,
            "runner_options.policy.goal_frames": policy.goal_frames,
            "runner_options.policy.goal_frame_stride": policy.goal_frame_stride,
            "runner_options.policy.latent_dim": policy.latent_dim,
            "runner_options.policy.actor_hidden_dims": tuple(policy.actor_hidden_dims),
            "runner_options.policy.critic_hidden_dims": tuple(policy.critic_hidden_dims),
            "runner_options.world_model.hidden_dims": tuple(self.options.world_model.hidden_dims),
            "joints": representation.num_joints,
            "actions": self.env.num_actions,
            "tracked links": len(self.motion.fk.link_names),
            "simulation context columns": self.sim_context.dim,
        }

    def save(self, path: str, infos: dict | None = None) -> None:
        if en.distributed.get_global_rank() != 0:
            return
        checkpoint = {
            "actor": self.actor.state_dict(),
            "dynamics": self.dynamics.state_dict(),
            "critic": None if self.critic is None else self.critic.state_dict(),
            "world_model_optimizers": [o.state_dict() for o in self.world_model_learner.optimizers],
            "actor_optimizers": [o.state_dict() for o in self.policy_learner.actor_optimizers],
            "critic_optimizers": [o.state_dict() for o in self.policy_learner.critic_optimizers],
            "return_sampler": None if self.sampler is None else self.sampler.state_dict(),
            "layout": self._layout(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        torch.save(checkpoint, path)

    def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None) -> dict | None:
        """Restore a checkpoint :meth:`save` wrote. ``load_optimizer=False`` leaves the optimizers as they are; the
        networks, the iteration and the return sampler's bin values are always restored, and the motion term gets
        the restored reset distribution.

        A checkpoint trained on another motion library has another number of reset bins. Its policy still replays,
        so the load keeps the networks and leaves the sampler as it is; :meth:`learn` then raises.
        """
        checkpoint = torch.load(path, map_location=map_location or self.device, weights_only=False)
        if (self.critic is None) != (checkpoint["critic"] is None):
            trained = "bptt" if checkpoint["critic"] is None else "actor_critic"
            raise ValueError(
                f"{path} was trained with policy.algorithm={trained!r}, but this runner has "
                f"policy.algorithm={self.options.policy.algorithm!r}: one has a critic and the other does not. "
                f"Set runner_options.policy.algorithm={trained!r} to load it."
            )
        # A checkpoint written before the layout record existed is checked by the state dicts alone.
        saved_layout, layout = checkpoint.get("layout"), self._layout()
        if saved_layout is not None and saved_layout != layout:
            differ = {
                key: (saved_layout.get(key), layout.get(key))
                for key in saved_layout.keys() | layout.keys()
                if saved_layout.get(key) != layout.get(key)
            }
            detail = "; ".join(f"{key}: checkpoint {a!r}, this run {b!r}" for key, (a, b) in sorted(differ.items()))
            raise ValueError(
                f"{path} was trained with other network inputs or sizes than this runner builds ({detail}). The "
                "runner_options.* entries are options to set; the others follow from the task's robot, tracked "
                "links and randomization events."
            )
        if (self.sampler is None) != (checkpoint["return_sampler"] is None) and self.sampler is not None:
            en.logger.warning(
                f"{path} holds no return sampler state (it was trained with return_sampler.enabled=False): the "
                "reset distribution starts uniform."
            )
        self.actor.load_state_dict(checkpoint["actor"])
        self.dynamics.load_state_dict(checkpoint["dynamics"])
        if self.critic is not None:
            self.critic.load_state_dict(checkpoint["critic"])
        self._stale_sampler_bins = None
        if self.sampler is not None and checkpoint["return_sampler"] is not None:
            saved_bins = checkpoint["return_sampler"]["value"].numel()
            if saved_bins != self.sampler.bin_count:
                self._stale_sampler_bins = saved_bins
            else:
                self.sampler.load_state_dict(checkpoint["return_sampler"])
                if float(self.sampler.effort.sum()) > 0.0:
                    self.motion.set_reset_distribution(self.sampler.distribution())
        if load_optimizer:
            groups = (
                (self.world_model_learner.optimizers, "world_model_optimizers"),
                (self.policy_learner.actor_optimizers, "actor_optimizers"),
                (self.policy_learner.critic_optimizers, "critic_optimizers"),
            )
            for optimizers, key in groups:
                for optimizer, state in zip(optimizers, checkpoint[key], strict=True):
                    load_optimizer_state(optimizer, state)
        self.current_learning_iteration = checkpoint["iter"]
        return checkpoint["infos"]

    def get_inference_policy(self, device: str | torch.device | None = None):
        """The deterministic policy on the env's observations: it reads each env's reference frame from the motion
        term. Every env follows the motion goal, whatever ``slot_mix`` the actor trained on, and the proprioception
        carries no sensor noise. A proprioception history starts from the first observation it is given.
        """
        self.actor.eval()
        self._history = None

        @torch.inference_mode()
        def policy(obs: TensorDict) -> torch.Tensor:
            state, sensor, frames, _ = self._observe(obs)
            if self._stores_sensor:
                if self._history is None:
                    self._seed_history(obs, training=False)
                else:
                    self._advance_history(sensor, self.env.unwrapped.reset_buf, training=False)
            return self.actor(self._policy_input(state, sensor, frames, training=False), interface="motion")

        return policy


class SuperMimicRunner(*EDEN_RUNNER_MIXINS, _SuperMimicRunner):
    """:class:`_SuperMimicRunner` with Eden's ``progress.json``, agent log theme and checkpoint retention.

    Named by ``SuperMimicRunnerOptions.class_name``, so ``eden train`` / ``eden inference`` build it through
    :func:`~eden.envs.wrappers.rsl_rl_env.make_rsl_rl_runner`.
    """
