"""DAgger depth distillation of an RMA teacher into a depth-camera student (rsl-rl-lib >= 5.5.1).

Port of the vision distillation loop from "Extreme Parkour with Legged Robots" (Cheng et al. 2023). The frozen
teacher (:class:`~eden.learning.wrappers.rsl_rl.rma.model.RmaActorModel`, history-latent path, true scandots, ground
truth privileged-explicit chunk) labels every step; the student — ``depth_actor`` (a copy of the teacher's actor net)
fed by a recurrent ``depth_encoder`` whose latent replaces the scan latent — acts *and drives the environment* from
its depth latent and its own prediction of the delta-yaw slots; one optimizer step per iteration regresses
``student -> teacher`` actions plus the yaw prediction, backpropagating through the whole iteration's graph.

That loop fits neither rsl_rl's ``Distillation`` (which stores observations and trains on mini-batches) nor its runner
loop (which collects under ``torch.inference_mode``), so this module ships its own runner,
:class:`RmaDepthDistillationRunner` — what
:class:`~eden.learning.wrappers.rsl_rl.rma.options.RslRlRmaDepthDistillationRunnerOptions`'s ``class_name`` names, so
``eden train --task extreme_parkour_student --checkpoint <teacher model_<it>.pt>`` runs it.

Cadence contract: the depth encoder runs only on control steps where the env refreshed the depth observation
(``common_step_counter % update_interval == 0``); between refreshes the cached latent/yaw are reused. Every
observation term of the student's groups that declares an ``update_interval`` must run at the same cadence (checked
against the live env at construction), and ``num_steps_per_env`` must be a positive multiple of it so an iteration
boundary never splits a cadence window (the update would otherwise backprop into the previous iteration's freed graph).

Checkpoints carry the teacher (``actor_state_dict``, estimator included — the same key a teacher checkpoint has) plus
``depth_actor_state_dict`` / ``depth_encoder_state_dict`` / ``depth_optimizer_state_dict``. ``load()`` accepts
either: a teacher checkpoint starts a fresh distillation (student actor re-copied from the teacher), a student
checkpoint resumes one. Only a student checkpoint (or a student trained in this process) replays:
:meth:`RmaDepthDistillationRunner.get_inference_policy` refuses an untrained depth encoder.

Like the source, the student reads the privileged-explicit chunk (base linear velocity) as the env provides it — in
sim, the ground truth — in distillation and in replay alike. It is an in-sim policy: ``eden deploy`` refuses this
runner, and a hardware port would have to substitute the frozen teacher's estimator for that chunk (in distillation
too, so the student trains on what it would see).
"""

from __future__ import annotations

import copy
import os
import time
from collections.abc import Callable

import torch
from tensordict import TensorDict
from torch import nn

from eden.envs.wrappers.rsl_rl_env import AgentIterationLogMixin, CheckpointRetentionMixin, ProgressFileMixin
from eden.learning.wrappers.rsl_rl.rma.depth_encoder import RecurrentDepthEncoder
from eden.learning.wrappers.rsl_rl.rma.model import RmaActorModel
from eden.utils import distributed as eden_dist

from rsl_rl.runners import OnPolicyRunner  # noqa: E402  (the model import above holds the rsl-rl-lib version guard)
from rsl_rl.utils import check_nan, resolve_class, resolve_obs_groups  # noqa: E402

#: ``class_name`` string ``eden train`` resolves to :class:`RmaDepthDistillationRunner`.
RMA_DEPTH_DISTILLATION_RUNNER_CLASS_NAME = "eden.learning.wrappers.rsl_rl.rma.distillation:RmaDepthDistillationRunner"
#: ``class_name`` string the runner resolves to :class:`RmaDepthDistillation`.
RMA_DEPTH_DISTILLATION_CLASS_NAME = "eden.learning.wrappers.rsl_rl.rma.distillation:RmaDepthDistillation"

#: The observation groups the student reads beyond the actor set, by their algorithm keyword.
_STUDENT_GROUP_KEYS = {"depth_obs_group": "depth_camera", "yaw_gate_obs_group": "delta_yaw_ok"}


class RmaDepthDistillation:
    """The student stage's modules and optimizer; :meth:`update` applies one DAgger regression.

    Holds the frozen ``teacher`` (an :class:`RmaActorModel`, kept whole — estimator included — so a student checkpoint
    stays self-contained), the trainable ``depth_encoder`` and the trainable ``depth_actor`` (initialized as a copy of
    the teacher's actor net). The rollout loop lives in :class:`RmaDepthDistillationRunner`. The keyword arguments are
    documented on :class:`~eden.learning.wrappers.rsl_rl.rma.options.RslRlRmaDepthDistillationAlgorithmOptions`.
    """

    def __init__(
        self,
        teacher: RmaActorModel,
        *,
        depth_obs_group: str = _STUDENT_GROUP_KEYS["depth_obs_group"],
        yaw_gate_obs_group: str = _STUDENT_GROUP_KEYS["yaw_gate_obs_group"],
        yaw_slots: tuple[int, int] | list[int] = (6, 8),
        update_interval: int = 5,
        num_pretrain_iters: int = 0,
        gru_hidden_dim: int = 512,
        learning_rate: float = 1.0e-3,
        max_grad_norm: float = 1.0,
        yaw_scale: float = 1.5,
        device: str = "cpu",
        multi_gpu_cfg: dict | None = None,
    ) -> None:
        if multi_gpu_cfg is not None:
            raise ValueError("RmaDepthDistillation is single-GPU; launch without torchrun.")
        start, stop = yaw_slots
        if not (0 <= start < stop <= teacher.dims.num_prop and stop - start == 2):
            raise ValueError(
                f"yaw_slots={tuple(yaw_slots)} must name the 2 (delta_yaw, delta_next_yaw) slots inside the "
                f"{teacher.dims.num_prop}-d proprioception chunk; the yaw head predicts exactly 2 values."
            )
        self.device = device
        self.dims = teacher.dims
        self.depth_obs_group = depth_obs_group
        self.yaw_gate_obs_group = yaw_gate_obs_group
        self.yaw_slice = slice(start, stop)
        self.update_interval = update_interval
        self.num_pretrain_iters = num_pretrain_iters
        self.yaw_scale = yaw_scale
        self.max_grad_norm = max_grad_norm
        self.teacher = teacher.to(device).requires_grad_(False)
        self.teacher_loaded = False
        # Whether the depth encoder / actor hold a trained student (loaded, or updated in this process).
        self.student_ready = False
        # The depth latent substitutes for the teacher's scan latent, so its width is the scan latent's width.
        self.depth_encoder = RecurrentDepthEncoder(
            num_prop=self.dims.num_prop, latent_dim=teacher.net.scan_latent_dim, gru_hidden_dim=gru_hidden_dim
        ).to(device)
        self.depth_actor = copy.deepcopy(self.teacher.net).requires_grad_(True).to(device)
        self.optimizer = torch.optim.Adam(
            [*self.depth_actor.parameters(), *self.depth_encoder.parameters()], lr=learning_rate
        )

    @staticmethod
    def construct_algorithm(obs: TensorDict, env, cfg: dict, device: str) -> RmaDepthDistillation:
        """Build the teacher from ``cfg["actor"]`` (it must match the teacher checkpoint) and the student around it."""
        alg_class, alg_cfg = resolve_class(cfg["algorithm"])
        actor_class, actor_cfg = resolve_class(cfg["actor"])
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], ["actor"])
        for key, default in _STUDENT_GROUP_KEYS.items():
            group = alg_cfg.get(key, default)
            if group not in obs.keys():
                raise ValueError(
                    f"Depth distillation reads the observation group algorithm.{key}={group!r}, which the env does "
                    f"not provide (groups: {sorted(obs.keys())}). Is the env the depth-student task?"
                )
        teacher = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **actor_cfg)
        if not isinstance(teacher, RmaActorModel):
            raise TypeError(
                "Depth distillation needs an RmaActorModel teacher (actor=RslRlRmaActorOptions(...)), got "
                f"{type(teacher).__name__}."
            )
        return alg_class(teacher, device=device, **alg_cfg, multi_gpu_cfg=cfg["multi_gpu"])

    def encode(self, depth_frame: torch.Tensor, flat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """One depth-encoder pass: ``(depth latent, scaled 2-d yaw prediction)``.

        The proprioception chunk is fed with the yaw slots hidden (the student must recover heading from depth); the
        Tanh-bounded yaw head is stretched by ``yaw_scale``. Shared by the training loop and the inference policy so
        the encoding contract cannot drift between them.
        """
        prop = flat[:, self.dims.prop_slice].clone()
        prop[:, self.yaw_slice] = 0.0
        latent_and_yaw = self.depth_encoder(depth_frame, prop)
        return latent_and_yaw[:, :-2], self.yaw_scale * latent_and_yaw[:, -2:]

    def teacher_actions(self, obs: TensorDict) -> torch.Tensor:
        """The teacher's label: history latent, true scandots, ground-truth privileged-explicit chunk (no estimator)."""
        with torch.no_grad():
            return self.teacher(obs, hist_encoding=True, estimate=False)

    def student_actions(self, flat: torch.Tensor, depth_latent: torch.Tensor) -> torch.Tensor:
        """The student's action from the (yaw-substituted) flat observation and the depth latent."""
        return self.depth_actor(flat, hist_encoding=True, scandots_latent=depth_latent)

    def update(self, action_diffs: torch.Tensor, yaw_errors: torch.Tensor) -> dict[str, float]:
        """One optimizer step on the iteration's ``teacher - student`` actions and ``true - predicted`` yaw errors.

        Both losses are mean per-sample L2 norms; the gradient clip covers the actor only (faithful to the source).
        """
        action_loss = action_diffs.norm(p=2, dim=1).mean()
        yaw_loss = yaw_errors.norm(p=2, dim=1).mean()
        loss = action_loss + yaw_loss
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.depth_actor.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.student_ready = True
        return {"depth_actor": action_loss.item(), "yaw": yaw_loss.item(), "total": loss.item()}

    def train_mode(self) -> None:
        self.teacher.eval()  # frozen
        self.depth_actor.train()
        self.depth_encoder.train()

    def eval_mode(self) -> None:
        self.teacher.eval()
        self.depth_actor.eval()
        self.depth_encoder.eval()

    def save(self) -> dict:
        return {
            "actor_state_dict": self.teacher.state_dict(),
            "depth_actor_state_dict": self.depth_actor.state_dict(),
            "depth_encoder_state_dict": self.depth_encoder.state_dict(),
            "depth_optimizer_state_dict": self.optimizer.state_dict(),
        }

    def load(self, loaded_dict: dict, *, load_optimizer: bool = True) -> bool:
        """Load the teacher, and the student halves when present; ``True`` when it resumed a student."""
        self.teacher.load_state_dict(loaded_dict["actor_state_dict"])
        self.teacher_loaded = True
        if "depth_actor_state_dict" not in loaded_dict:
            self.depth_actor.load_state_dict(self.teacher.net.state_dict())  # a fresh distillation from the teacher
            return False
        self.depth_actor.load_state_dict(loaded_dict["depth_actor_state_dict"])
        self.depth_encoder.load_state_dict(loaded_dict["depth_encoder_state_dict"])
        if load_optimizer:
            self.optimizer.load_state_dict(loaded_dict["depth_optimizer_state_dict"])
        self.student_ready = True
        return True


class RmaDepthDistillationRunner(ProgressFileMixin, AgentIterationLogMixin, CheckpointRetentionMixin, OnPolicyRunner):
    """Runs :class:`RmaDepthDistillation`'s DAgger loop; see the module docstring for the loop and its contracts."""

    alg: RmaDepthDistillation

    #: Read by ``eden train`` before it builds the env: this runner cannot start without a teacher.
    checkpoint_required = (
        "Depth distillation needs a teacher: pass --checkpoint <the teacher run's model_<it>.pt> "
        "(or --resume / --checkpoint <a student checkpoint> to continue a distillation)."
    )

    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        if eden_dist.is_distributed():
            raise RuntimeError(
                "Depth distillation is single-GPU: it has no parameter broadcast or gradient reduction, so a "
                "distributed launch would train an independent student per rank. Launch without torchrun."
            )
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)
        interval, steps = self.alg.update_interval, self.cfg["num_steps_per_env"]
        if steps <= 0 or steps % interval != 0:
            raise ValueError(
                f"num_steps_per_env ({steps}) must be a positive multiple of algorithm.update_interval ({interval}), "
                "so an iteration boundary never splits a depth-refresh window."
            )
        self._validate_env_cadence()

    def _validate_env_cadence(self) -> None:
        """Fail fast if a refresh-cadenced observation term runs off the encoder's cadence."""
        manager = self.env.unwrapped.observation_manager
        group_terms = manager.group_term_names
        interval = self.alg.update_interval
        cadences: dict[str, int] = {}
        for group in [*self.cfg["obs_groups"]["actor"], self.alg.depth_obs_group]:
            for name in group_terms[group]:
                term_interval = getattr(manager.get_term(name), "update_interval", None)
                if term_interval is not None:
                    cadences[f"{group}/{name}"] = term_interval
        if not any(key.startswith(f"{self.alg.depth_obs_group}/") for key in cadences):
            raise ValueError(
                f"No term of the '{self.alg.depth_obs_group}' group declares an update_interval, so the depth "
                "encoder's cadence cannot be matched to the frame refresh. The depth term must render every "
                f"update_interval steps (algorithm.update_interval={interval})."
            )
        mismatched = {key: value for key, value in cadences.items() if value != interval}
        if mismatched:
            raise ValueError(
                f"algorithm.update_interval={interval} does not match the env's refresh cadence of {mismatched}; the "
                "encoder would run off the frame / delta-yaw refresh and silently degrade distillation. Set them equal."
            )

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        """The vision DAgger loop; ``init_at_random_ep_len`` spreads the envs' episode clocks, as rsl_rl's loop does."""
        if not self.alg.teacher_loaded:
            raise ValueError(self.checkpoint_required)
        alg, env_unwrapped = self.alg, self.env.unwrapped
        interval = alg.update_interval
        check_for_nan = self.cfg.get("check_for_nan", True)
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )
        obs = self.env.get_observations().to(self.device)
        alg.train_mode()
        self.logger.init_logging_writer()
        depth_latent: torch.Tensor | None = None
        yaw_pred: torch.Tensor | None = None

        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        for it in range(start_it, total_it):
            start = time.time()
            action_diffs: list[torch.Tensor] = []
            yaw_errors: list[torch.Tensor] = []
            gate_fracs: list[torch.Tensor] = []  # 0-d tensors; one host sync per iteration
            # NO torch.inference_mode (unlike PPO): the student's actions carry the graph the update backpropagates.
            for _ in range(self.cfg["num_steps_per_env"]):
                # Clone: the yaw substitution must not leak into the env-owned observation buffer.
                flat = alg.teacher.get_flat_obs(obs).clone()
                gate = obs[alg.yaw_gate_obs_group].squeeze(-1) > 0.5
                if depth_latent is None or env_unwrapped.common_step_counter % interval == 0:
                    depth_latent, yaw_pred = alg.encode(obs[alg.depth_obs_group], flat)
                    yaw_errors.append(flat[:, alg.yaw_slice] - yaw_pred)
                # The label comes BEFORE the yaw substitution: the teacher sees the true delta-yaw.
                teacher_actions = alg.teacher_actions(obs)
                gate_fracs.append(gate.float().mean())
                # The student sees its own (detached) yaw prediction wherever the gate allows; the yaw head trains
                # only through the auxiliary loss, as in the source.
                flat[gate, alg.yaw_slice] = yaw_pred.detach()[gate]
                student_actions = alg.student_actions(flat, depth_latent)
                action_diffs.append(teacher_actions - student_actions)

                stepped = teacher_actions if it < alg.num_pretrain_iters else student_actions
                obs, rewards, dones, extras = self.env.step(stepped.detach().to(self.env.device))
                if check_for_nan:
                    check_nan(obs, rewards, dones)
                obs, rewards, dones = obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                self.logger.process_env_step(rewards, dones, extras, None)

            stop = time.time()
            collect_time, start = stop - start, stop

            loss_dict = alg.update(torch.cat(action_diffs), torch.cat(yaw_errors))
            loss_dict["delta_yaw_ok_frac"] = torch.stack(gate_fracs).mean().item()
            alg.depth_encoder.detach_hidden()
            # Force a fresh encoder pass at the next iteration's first step: reusing the latent across the update
            # would backprop into the freed graph.
            depth_latent = yaw_pred = None

            learn_time = time.time() - start
            self.current_learning_iteration = it
            self.logger.log(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
                learning_rate=alg.optimizer.param_groups[0]["lr"],
                # Distillation is deterministic — no exploration noise to report.
                action_std=torch.zeros(self.env.num_actions, device=self.device),
                rnd_weight=None,
            )
            if self.logger.writer is not None and self._save_due(it, start_it):
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))

        if self.logger.writer is not None:
            self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))
            self.logger.stop_logging_writer()

    def _save_due(self, it: int, start_it: int) -> bool:
        """The source's staged save schedule: denser early, sparser as the run matures.

        Faithful-port quirk: like the source, the stage is picked run-relative (``it - start_it``) while the modulus
        runs on the absolute iteration — a resumed run re-enters the dense stage.
        """
        rel, base = it - start_it, self.cfg["save_interval"]
        if rel < 2500:
            return it % base == 0
        if rel < 5000:
            return it % (2 * base) == 0
        return it % (5 * base) == 0

    def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None) -> dict | None:
        """Load a teacher checkpoint (fresh distillation) or a student checkpoint (resume)."""
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        if "actor_state_dict" not in loaded_dict:
            raise ValueError(
                f"'{path}' has no 'actor_state_dict' (keys: {sorted(loaded_dict)}), so it is not an RMA teacher or "
                "student checkpoint of the rsl-rl-lib 5 layout. An rsl_rl 3 checkpoint ('model_state_dict') does not "
                "load here: retrain the teacher with `eden train --task extreme_parkour_teacher`."
            )
        if self.alg.load(loaded_dict, load_optimizer=load_optimizer):
            self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict.get("infos")

    def get_inference_policy(self, device: str | None = None) -> Callable[[TensorDict], torch.Tensor]:
        """The deployed student: depth encoder every ``update_interval`` steps, its yaw in the yaw slots.

        The yaw prediction replaces the yaw slots unconditionally (the source's play behavior — the gate is a
        training-time aid), and the privileged-explicit chunk stays as the env provides it (in-sim ground truth; a
        hardware deployment would substitute the teacher's estimator here).
        """
        alg, env_unwrapped = self.alg, self.env.unwrapped
        if not alg.student_ready:
            raise RuntimeError(
                "The depth student has not been trained: this runner loaded a teacher checkpoint (no "
                "depth_actor_state_dict / depth_encoder_state_dict), so its depth encoder is randomly initialized and "
                "would drive the robot from a random latent and a random yaw. Replay a student checkpoint (a model_<it>.pt "
                "written by `eden train --task extreme_parkour_student`), or replay the teacher itself with "
                "--task extreme_parkour_teacher."
            )
        alg.eval_mode()
        if device is not None:
            alg.depth_actor.to(device)
            alg.depth_encoder.to(device)
        alg.depth_encoder.reset_hidden()
        state: dict[str, torch.Tensor | None] = {"latent": None, "yaw": None}

        def act(obs: TensorDict) -> torch.Tensor:
            with torch.no_grad():
                flat = alg.teacher.get_flat_obs(obs).clone()
                if state["latent"] is None or env_unwrapped.common_step_counter % alg.update_interval == 0:
                    state["latent"], state["yaw"] = alg.encode(obs[alg.depth_obs_group], flat)
                flat[:, alg.yaw_slice] = state["yaw"]
                return alg.student_actions(flat, state["latent"])

        return act
