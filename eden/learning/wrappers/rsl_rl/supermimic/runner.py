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
from eden.learning.wrappers.rsl_rl.supermimic.learners import PolicyLearner, WorldModelLearner
from eden.learning.wrappers.rsl_rl.supermimic.models import Dynamics, MultiEncoderActor, build_mlp
from eden.learning.wrappers.rsl_rl.supermimic.options import SuperMimicRunnerOptions
from eden.learning.wrappers.rsl_rl.supermimic.replay_buffer import FieldSpec, ReplayBuffer
from eden.learning.wrappers.rsl_rl.supermimic.representation import (
    GenCoordRepresentation,
    check_task_reward,
    goal_dims,
)
from eden.learning.wrappers.rsl_rl.supermimic.reset_sampler import ReturnSampler
from eden.learning.wrappers.rsl_rl.supermimic.sim_context import SimContext
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


class _Logger:
    """Episode statistics of the simulator and the learners' losses, to the console and TensorBoard or W&B.

    Holds what Eden's runner mixins read: ``log_dir``, ``disable_logs``, ``rewbuffer``, ``lenbuffer``,
    ``tot_timesteps`` and ``tot_time``.
    """

    def __init__(self, log_dir: str | None, options: SuperMimicRunnerOptions, num_envs: int, device: torch.device):
        self.log_dir = log_dir
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
        if self.disable_logs:
            return
        if options.logger == "wandb":
            import wandb

            run_name = options.run_name or None
            self._wandb = wandb.run or wandb.init(
                project=options.wandb_project, name=run_name, dir=log_dir, config=options.dict()
            )
        else:
            from torch.utils.tensorboard import SummaryWriter

            self._writer = SummaryWriter(log_dir=log_dir, flush_secs=10)

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
        if self._wandb is not None:
            self._wandb.finish()


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

    def __init__(self, env: RslRlVecEnvWrapper, train_cfg: Any, log_dir: str | None = None, device: str = "cuda"):
        if en.distributed.is_distributed():
            raise RuntimeError(
                "The SuperMimic runner trains on one GPU; under torchrun every rank would train its own copy. "
                "Launch `eden train --task supermimic` without torchrun."
            )
        options = train_cfg if isinstance(train_cfg, SuperMimicRunnerOptions) else SuperMimicRunnerOptions(**train_cfg)
        self.env = env
        self.options = options
        self.cfg = options.dict()
        self.log_dir = log_dir
        self.device = torch.device(env.device if device == "auto" else device)
        self.current_learning_iteration = 0
        base = env.unwrapped
        self.num_envs = env.num_envs

        self.motion = base.motion_manager.get_term(options.motion_term)
        self.library = self.motion.library
        controller = base.action_manager.get_term(options.action_term)
        robot = base.entities[controller.entity_name]
        dofs = robot.find_named_dofs(list(controller.dofs_name))
        if list(controller.dofs_name) != list(robot.dofs_name) or list(self.motion.fk.joint_names) != list(
            robot.dofs_name
        ):
            raise ValueError(
                f"SuperMimic reads the robot's joints in one order: the controller {options.action_term!r} "
                f"({list(controller.dofs_name)}), the motion term's joints ({self.motion.fk.joint_names}) and the robot "
                f"({list(robot.dofs_name)}) must all list every joint in the robot's dofs_name order."
            )
        num_actions = env.num_actions
        self.representation = GenCoordRepresentation(
            fk=self.motion.fk,
            default_joint_pos=robot.get_default_dofs_pos(dofs)[0],
            soft_joint_pos_limits=robot.soft_dofs_pos_limits[0, dofs.columns],
            dt=base.dt,
            decimation=base.env_options.decimation,
            num_actions=num_actions,
        ).to(self.device)
        self.sim_context = SimContext(base, options.sim_context, options.action_term)

        rewards = base.reward_manager
        reward_terms = {}
        for name in rewards.active_terms:
            term = rewards.get_term_options(name)
            std = (term.params or {}).get("std") if getattr(term, "params", None) else getattr(term, "std", None)
            reward_terms[name] = (term.weight, std)
        check_task_reward(reward_terms)
        self.max_step_reward = base.dt * sum(w for w, _ in reward_terms.values() if w > 0)
        terminations = base.termination_manager
        thresholds = [
            terminations.get_term_options(n).params["threshold"]
            for n in (options.anchor_pos_termination, options.anchor_ori_termination)
        ]

        scale = controller.effective_scale
        scale = torch.as_tensor(scale, device=self.device, dtype=torch.float).expand(1, num_actions)[0]
        self.dynamics = Dynamics(
            self.representation,
            num_actions,
            self.sim_context.dim,
            scale,
            self.representation.default_joint_pos,
            options.world_model.hidden_dims,
        ).to(self.device)
        self.world_model_learner = WorldModelLearner(
            self.dynamics, self.representation, options.world_model, str(self.device)
        )

        p = options.policy
        specs = [motion_goal_spec(self.representation.num_joints), velocity_goal_spec(options.velocity_goal)]
        self.actor = MultiEncoderActor(
            obs_dim=self.representation.rep_state_dim if p.observation == "state" else self.representation.proprio_dim,
            action_dim=num_actions,
            goal_specs=specs,
            slot_mix=p.slot_mix,
            latent_dim=p.latent_dim,
            hidden_dims=p.actor_hidden_dims,
            latent_bc_weight=p.latent_bc_weight,
            history_length=p.history_length,
        ).to(self.device)
        self.active_specs = [s for s in specs if p.slot_mix.get(s.name, 0.0) > 0.0]
        critic_dim = (
            self.representation.rep_state_dim + goal_dims(self.representation.num_joints)[0] + self.sim_context.dim
        )
        self.critic = (
            build_mlp(critic_dim, p.critic_hidden_dims, 1).to(self.device) if p.algorithm == "actor_critic" else None
        )
        self.wm_env = WorldModelEnv(
            self.dynamics,
            self.representation,
            self.library,
            self.motion.fk,
            self.active_specs,
            p.observation,
            p.history_length,
            *thresholds,
            p.termination_softness,
        )
        if self.device.type == "cuda":
            self.wm_env.step = torch.compile(self.wm_env.step)
            self.wm_env.reset = torch.compile(self.wm_env.reset)
        self.policy_learner = PolicyLearner(
            self.actor, self.critic, self.wm_env, p, self.max_step_reward, str(self.device)
        )

        self.sampler = None
        if options.return_sampler.enabled:
            if self.motion.reset_sampling != "external":
                raise ValueError(
                    f"The return sampler sets the reset distribution of motion term {options.motion_term!r}, which "
                    f"samples resets with reset_sampling={self.motion.reset_sampling!r}. Set its reset_sampling to "
                    "'external', or disable the sampler (runner_options.return_sampler.enabled=False)."
                )
            self.sampler = ReturnSampler(
                options.return_sampler,
                self.motion.bin_count,
                self.library.num_frames,
                self.num_envs,
                self.max_step_reward,
                self.device,
            )

        self._stores_sensor = p.observation == "proprio" and p.history_length > 1
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
        self.replay_buffer = ReplayBuffer(schema, options.buffer_size, self.num_envs, self.device)
        self._history: torch.Tensor | None = None
        self._pending: list[tuple[torch.Tensor, torch.Tensor, TensorDict]] = []
        self.logger = _Logger(log_dir, options, self.num_envs, self.device)

    def _observe(self, obs: TensorDict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """``(full state, proprioception, frames, sim context)`` of every env now."""
        o = self.options
        return (
            obs[o.state_observation_group].to(self.device),
            obs[o.actor_observation_group].to(self.device),
            self.motion.frame_index.clone(),
            self.sim_context.read(),
        )

    def _policy_input(self, state: torch.Tensor, sensor: torch.Tensor, frames: torch.Tensor) -> TensorDict:
        aux = self.representation.aux_state(state)
        goals = {spec.name: spec.goal(self.library, frames, aux) for spec in self.active_specs}
        if self.options.policy.observation == "state":
            history = self.representation.represent(state).unsqueeze(1)
        elif self.options.policy.history_length == 1:
            history = self.representation.add_proprio_noise(sensor).unsqueeze(1)
        else:
            history = self._history
        return actor_input(goals, history)

    def _seed_history(self, obs: TensorDict) -> None:
        if self._stores_sensor:
            seed = self.representation.add_proprio_noise(obs[self.options.actor_observation_group].to(self.device))
            self._history = seed.unsqueeze(1).repeat(1, self.options.policy.history_length, 1)

    def _advance_history(self, sensor: torch.Tensor, dones: torch.Tensor) -> None:
        """Append the noised next frame; an env that just reset repeats it over its whole history."""
        if not self._stores_sensor:
            return
        frame = self.representation.add_proprio_noise(sensor)
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
        o, wm, p = self.options, self.options.world_model, self.options.policy
        obs, _ = self.env.reset()
        self._seed_history(obs)
        updates_due = _Ratio(self.num_envs / wm.num_batches / wm.batch_size / wm.horizon * o.replay_ratio)
        start = self.current_learning_iteration
        total = start + num_learning_iterations
        step = start * o.num_steps_per_env
        losses: dict[str, torch.Tensor] = {}
        for it in range(start, total):
            t0, learn_time = time.time(), 0.0
            for _ in range(o.num_steps_per_env):
                obs = self._collect(obs)
                for _ in range(updates_due(step)):
                    if self.replay_buffer.num_transitions < wm.horizon:
                        continue
                    t1 = time.time()
                    windows = (self.replay_buffer.sample(wm.horizon, wm.batch_size) for _ in range(wm.num_batches))
                    losses.update(self.world_model_learner.update(windows))
                    starts = (
                        sample_starts(self.replay_buffer, p.history_length, p.batch_size) for _ in range(p.num_batches)
                    )
                    losses.update(self.policy_learner.update(starts))
                    learn_time += time.time() - t1
                step += 1
            self.current_learning_iteration = it + 1
            if self.sampler is not None:
                losses["reset_sampler/top1_prob"] = self.motion.reset_distribution.max()
            self.logger.log(
                it, total, o.num_steps_per_env * self.num_envs, time.time() - t0 - learn_time, learn_time, losses
            )
            if self.log_dir is not None and (it + 1) % o.save_interval == 0:
                self.save(os.path.join(self.log_dir, f"model_{it + 1}.pt"))
        if self.log_dir is not None:
            self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))
        self.logger.close()

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
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        torch.save(checkpoint, path)

    def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None) -> dict | None:
        checkpoint = torch.load(path, map_location=map_location or self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint["actor"])
        self.dynamics.load_state_dict(checkpoint["dynamics"])
        if self.critic is not None and checkpoint["critic"] is not None:
            self.critic.load_state_dict(checkpoint["critic"])
        if load_optimizer:
            groups = (
                (self.world_model_learner.optimizers, "world_model_optimizers"),
                (self.policy_learner.actor_optimizers, "actor_optimizers"),
                (self.policy_learner.critic_optimizers, "critic_optimizers"),
            )
            for optimizers, key in groups:
                for optimizer, state in zip(optimizers, checkpoint[key], strict=True):
                    optimizer.load_state_dict(state)
            if self.sampler is not None and checkpoint["return_sampler"] is not None:
                self.sampler.load_state_dict(checkpoint["return_sampler"])
        self.current_learning_iteration = checkpoint["iter"]
        return checkpoint["infos"]

    def get_inference_policy(self, device: str | torch.device | None = None):
        """The deterministic policy on the env's observations: it reads each env's reference frame from the motion
        term. A proprioception history starts from the first observation it is given.
        """
        self.actor.eval()
        self._history = None

        @torch.inference_mode()
        def policy(obs: TensorDict) -> torch.Tensor:
            state, sensor, frames, _ = self._observe(obs)
            if self._stores_sensor:
                if self._history is None:
                    self._seed_history(obs)
                else:
                    self._advance_history(sensor, self.env.unwrapped.reset_buf)
            return self.actor(self._policy_input(state, sensor, frames))

        return policy


class SuperMimicRunner(*EDEN_RUNNER_MIXINS, _SuperMimicRunner):
    """:class:`_SuperMimicRunner` with Eden's ``progress.json``, agent log theme and checkpoint retention."""


def make_supermimic_runner(
    env: RslRlVecEnvWrapper, checkpoint: str | None = None, log_dir: str | None = None
) -> SuperMimicRunner:
    """Build the SuperMimic runner on ``env`` from its ``runner_options``, optionally loading ``checkpoint``."""
    options = env.runner_options
    if not isinstance(options, SuperMimicRunnerOptions):
        raise TypeError(f"make_supermimic_runner takes SuperMimicRunnerOptions, got {type(options).__name__}.")
    device = str(env.device) if options.device == "auto" else options.device
    runner = SuperMimicRunner(env, options, log_dir=log_dir or en.log_dir, device=device)
    if checkpoint:
        runner.load(checkpoint)
    return runner
