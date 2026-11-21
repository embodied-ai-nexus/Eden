"""The world-model fit and the policy update through imagined rollouts, each captured as one CUDA graph."""

from __future__ import annotations

import functools
from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict
from torch import nn

if TYPE_CHECKING:
    from eden.learning.wrappers.rsl_rl.supermimic.models import Dynamics
    from eden.learning.wrappers.rsl_rl.supermimic.options import PolicyOptions, WorldModelOptions
    from eden.learning.wrappers.rsl_rl.supermimic.representation import GenCoordRepresentation
    from eden.learning.wrappers.rsl_rl.supermimic.world_model_env import WorldModelEnv


def build_optimizers(
    modules: Sequence[nn.Module], lr: float, betas: tuple[float, float] = (0.9, 0.999), device: str = "cpu"
) -> list[torch.optim.Optimizer]:
    """Muon on the hidden Linear weights of every ``nn.Sequential`` in ``modules`` (all but the first and last),
    fused capturable Adam on the other parameters, both at ``lr``.

    Not :class:`~eden.learning.wrappers.rsl_rl.muon_ppo.ppo.MuonAdamW`, which splits by parameter rank and decays
    the weights with AdamW: a captured update needs Adam's ``capturable`` state and a tensor learning rate.
    """
    hidden = [
        linear.weight
        for module in modules
        for seq in module.modules()
        if isinstance(seq, nn.Sequential)
        for linear in [m for m in seq if isinstance(m, nn.Linear)][1:-1]
    ]
    hidden_ids = {id(p) for p in hidden}
    rest = [p for m in modules for p in m.parameters() if id(p) not in hidden_ids]
    on_cuda = torch.device(device).type == "cuda"
    # A tensor learning rate on the GPU, so a captured update reads it in place.
    adam_lr = torch.tensor(lr, device=device) if on_cuda else lr
    adam = torch.optim.Adam(rest, lr=adam_lr, betas=betas, fused=on_cuda, capturable=on_cuda)
    if not hidden:
        return [adam]
    if not hasattr(torch.optim, "Muon"):
        raise ImportError(
            f"torch.optim.Muon is unavailable in torch {torch.__version__} ({torch.__file__}); SuperMimic optimizes "
            "the hidden layers of its networks with it and needs torch >= 2.10. Upgrade torch."
        )
    return [adam, torch.optim.Muon(hidden, lr=lr, adjust_lr_fn="match_rms_adamw")]


def optimize(
    optimizers: Sequence[torch.optim.Optimizer],
    loss: torch.Tensor,
    params: Iterable[torch.Tensor],
    max_norm: float,
    clip_value: float = 0.0,
) -> torch.Tensor:
    """``zero_grad -> backward -> clip the global norm (and each element) -> step``; returns the gradient norm."""
    for optimizer in optimizers:
        optimizer.zero_grad(set_to_none=True)
    loss.backward()
    params = list(params)
    grad_norm = torch.nn.utils.clip_grad_norm_(params, max_norm=max_norm)
    if clip_value > 0:
        torch.nn.utils.clip_grad_value_(params, clip_value)
    for optimizer in optimizers:
        optimizer.step()
    return grad_norm


def graph_state(modules: Iterable[nn.Module], optimizers: Iterable[torch.optim.Optimizer]) -> list[torch.Tensor]:
    """The tensors a captured update reads that Python could rebind: parameters, buffers, optimizer state and
    learning-rate tensors.
    """
    state: list[torch.Tensor] = []
    for module in modules:
        state += [*module.parameters(), *module.buffers()]
    for optimizer in optimizers:
        state += [t for s in optimizer.state.values() for t in s.values() if torch.is_tensor(t)]
        state += [g["lr"] for g in optimizer.param_groups if torch.is_tensor(g["lr"])]
    return state


def load_optimizer_state(optimizer: torch.optim.Optimizer, saved: dict) -> None:
    """Load an optimizer ``state_dict`` into the state tensors the optimizer already holds.

    ``Optimizer.load_state_dict`` replaces those tensors, which a recorded :class:`CapturedUpdate` would keep
    reading at their old addresses. An optimizer that has not stepped yet holds no state, and takes the saved one
    as it is.
    """
    if not optimizer.state:
        optimizer.load_state_dict(saved)
        return
    params = [p for group in optimizer.param_groups for p in group["params"]]
    for index, param in enumerate(params):
        current = optimizer.state[param]
        for key, value in saved["state"].get(index, {}).items():
            if torch.is_tensor(value) and torch.is_tensor(current.get(key)):
                current[key].copy_(value)
            else:
                current[key] = value
    for group, saved_group in zip(optimizer.param_groups, saved["param_groups"], strict=True):
        for key, value in saved_group.items():
            if key == "params":
                continue
            if torch.is_tensor(group.get(key)):
                group[key].copy_(torch.as_tensor(value))
            else:
                group[key] = value


@functools.cache
def _capture_stream_and_pool(device: torch.device):
    return torch.cuda.Stream(device), torch.cuda.graph_pool_handle()


class CapturedUpdate:
    """Runs ``fn`` eagerly for its first ``warmup`` calls, then records it once as a CUDA graph and replays it.

    ``fn`` must be capture-safe: no host syncs, fixed shapes, and every tensor it reads besides its inputs updated in
    place. Python values it reads are fixed at capture, and each replay overwrites the outputs it returns. All
    captured updates on a device record on one stream into one memory pool and replay in the order they were
    recorded. A replay raises if a tensor in ``state()`` moved, so a checkpoint's optimizer state is loaded into
    the existing tensors (:func:`load_optimizer_state`). On the CPU ``fn`` always runs eagerly.
    """

    def __init__(
        self,
        fn: Callable[[TensorDict], dict[str, torch.Tensor]],
        state: Callable[[], list[torch.Tensor]],
        warmup: int = 3,
    ):
        self._fn = fn
        self._state = state
        self._warmup = warmup
        self._inputs: TensorDict | None = None
        self._graph: torch.cuda.CUDAGraph | None = None
        self._identity: list[int] = []
        self._outputs: dict[str, torch.Tensor] = {}

    def __call__(self, inputs: TensorDict) -> dict[str, torch.Tensor]:
        state = self._state()
        if not state[0].is_cuda:
            return self._fn(inputs)
        with torch.cuda.device(state[0].device):
            stream, pool = _capture_stream_and_pool(state[0].device)
            if self._inputs is None:
                self._inputs = inputs.clone()
            else:
                self._inputs.update_(inputs)
            if self._warmup > 0:
                self._warmup -= 1
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    outputs = self._fn(self._inputs)
                torch.cuda.current_stream().wait_stream(stream)
                return outputs
            if self._graph is None:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=pool, stream=stream):
                    self._outputs = self._fn(self._inputs)
                self._graph, self._identity = graph, [x.data_ptr() for x in self._state()]
            elif [x.data_ptr() for x in state] != self._identity:
                raise RuntimeError("a tensor read by a captured update moved; update it in place instead.")
            self._graph.replay()
            torch.cuda.current_stream().synchronize()
            return self._outputs


class WorldModelLearner:
    """Fits the dynamics to replay-buffer windows: an autoregressive rollout of each window's actions from its first
    state, scored per channel group with Bryson's rule (weight ``1 / nominal_error^2``).
    """

    def __init__(
        self, dynamics: Dynamics, representation: GenCoordRepresentation, options: WorldModelOptions, device: str
    ):
        self.dynamics = dynamics
        self.representation = representation
        self.options = options
        self.params = list(dynamics.mlp.parameters())
        self.optimizers = build_optimizers([dynamics.mlp], options.lr, device=device)
        self.group_weights = {
            name: 1.0 / getattr(options, f"nominal_error_{name}") ** 2 for name in representation.loss_groups
        }
        compile_ = functools.partial(torch.compile, disable=torch.device(device).type != "cuda")
        self._rollout = compile_(self._rollout_eager)
        self._losses = compile_(self._losses_eager)
        self._represent = compile_(representation.represent)
        self._update = CapturedUpdate(self._update_from, lambda: graph_state([dynamics], self.optimizers))

    @torch.no_grad()
    def update_normalization(self, states: torch.Tensor, actions: torch.Tensor) -> None:
        """Fold freshly collected states and raw actions into the dynamics' input normalizers."""
        for normalizer in self.dynamics.normalizers():
            normalizer.train()
        self.dynamics.state_normalizer.update(self.representation.represent(states))
        self.dynamics.action_normalizer.update(self.dynamics.target(actions))
        for normalizer in self.dynamics.normalizers():
            normalizer.eval()

    def update(self, windows: Iterable[TensorDict]) -> dict[str, torch.Tensor]:
        self.dynamics.mlp.train()
        self.dynamics.mlp.requires_grad_(True)
        return self._update(torch.stack([w.select("state", "action", "sim_ctx") for w in windows]))

    def _update_from(self, batches: TensorDict) -> dict[str, torch.Tensor]:
        stats = []
        for batch in batches.unbind(0):
            states = batch["state"]
            rep_states = self._represent(states)
            aux_states = self.representation.aux_state(states)
            pred_rep, pred_aux = self._rollout(
                rep_states[:, 0], batch["action"], aux_states[:, 0], batch["sim_ctx"][:, 0]
            )
            loss, group_mse = self._losses(pred_rep, pred_aux, rep_states[:, 1:], aux_states[:, 1:])
            optimize(self.optimizers, loss, self.params, max_norm=1.0)
            stats.append(torch.cat([loss.unsqueeze(0), group_mse]).detach())
        means = torch.stack(stats).mean(0)
        names = ["state", *self.representation.loss_groups]
        return {f"world_model/{name}": value for name, value in zip(names, means, strict=True)}

    def _rollout_eager(
        self, rep_state: torch.Tensor, actions: torch.Tensor, aux_state: torch.Tensor, sim_ctx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predicted states after each of the window's actions but the last, whose outcome is not stored."""
        reps, auxes = [], []
        for k in range(actions.shape[1] - 1):
            aux_state, rep_state = self.dynamics(rep_state, actions[:, k], aux_state, sim_ctx)
            reps.append(rep_state)
            auxes.append(aux_state)
        return torch.stack(reps, dim=1), torch.stack(auxes, dim=1)

    def _losses_eager(
        self, pred_rep: torch.Tensor, pred_aux: torch.Tensor, rep: torch.Tensor, aux: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The weighted state loss, a width-weighted mean over every channel, and the plain MSE of each group."""
        pred = self.representation.loss_blocks(pred_rep, pred_aux)
        true = self.representation.loss_blocks(rep, aux)
        sq = {name: (pred[name] - true[name]) ** 2 for name in pred}
        group_mse = torch.stack([e.mean() for e in sq.values()])
        widths = torch.stack([group_mse.new_full((), e.shape[-1]) for e in sq.values()])
        weights = torch.stack([group_mse.new_full((), self.group_weights[name]) for name in sq])
        return (group_mse * weights * widths).sum() / widths.sum(), group_mse


def gae_returns(
    values: torch.Tensor, last_value: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor, gamma: float, lam: float
) -> torch.Tensor:
    """TD(lambda) returns of a time-major rollout; a soft done ``p`` discounts the next state by ``(1 - p) gamma``."""
    returns = torch.empty_like(values)
    advantage = torch.zeros_like(last_value)
    next_value = last_value
    for t in reversed(range(values.shape[0])):
        not_done = 1.0 - dones[t]
        delta = rewards[t] + not_done * gamma * next_value - values[t]
        advantage = delta + not_done * gamma * lam * advantage
        returns[t] = advantage + values[t]
        next_value = values[t]
    return returns


class PolicyLearner:
    """Improves the actor by backpropagation through imagined rollouts of the frozen world model.

    The objective is the rollout's discounted reward, each step weighted by the probability of having survived to it,
    plus, for ``algorithm="actor_critic"``, the critic's value of the last state discounted the same way. The critic
    then fits TD(lambda) targets of the same rollouts (5 epochs of 4 minibatches). Values never exceed ``v_max``, the
    return of the largest reward every step. ``algorithm="bptt"`` has no critic and sums the rewards undiscounted.
    """

    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module | None,
        wm_env: WorldModelEnv,
        options: PolicyOptions,
        max_step_reward: float,
        device: str,
    ):
        self.actor = actor
        self.critic = critic
        self.wm_env = wm_env
        self.options = options
        self.gamma = options.gamma if critic is not None else 1.0
        #: The largest discounted return, which bounds the critic.
        self.v_max = max_step_reward / (1.0 - options.gamma) if critic is not None else None
        self.actor_params = list(actor.parameters())
        self.actor_optimizers = build_optimizers([actor], options.lr, betas=(0.9, options.beta2), device=device)
        self.critic_optimizers = build_optimizers([critic], options.lr, device=device) if critic is not None else []
        on_cuda = torch.device(device).type == "cuda"
        self._rollout_actor = torch.compile(actor) if on_cuda else actor
        self._update = CapturedUpdate(self._update_from, self._graph_state)

    def update(self, starts: Iterable[TensorDict]) -> dict[str, torch.Tensor]:
        self.actor.train()
        self.wm_env.dynamics.eval()
        self.wm_env.dynamics.requires_grad_(False)
        return self._update(torch.stack(list(starts)))

    def _update_from(self, starts: TensorDict) -> dict[str, torch.Tensor]:
        totals: dict[str, torch.Tensor] = {}
        for start in starts.unbind(0):
            for key, value in self._train_batch(start).items():
                totals[key] = totals[key] + value if key in totals else value
        n = starts.shape[0]
        reward = totals.pop("return") / self.options.horizon / n / self.wm_env.representation.dt
        return {"policy/reward": reward} | {f"policy/{k}": v / n for k, v in totals.items()}

    def _train_batch(self, start: TensorDict) -> dict[str, torch.Tensor]:
        o = self.options
        obs = self.wm_env.reset(start)
        ret, aux_loss, survival = 0.0, 0.0, 1.0
        rollout: list[TensorDict] = []
        for t in range(o.horizon):
            with torch.autocast("cuda", torch.bfloat16, enabled=obs.device.type == "cuda"):
                action = self._rollout_actor(obs).float()
            next_obs, reward, term_prob = self.wm_env.step(action)
            step = {"critic": obs["critic"], "reward": reward, "done": term_prob}
            rollout.append(TensorDict(step, batch_size=reward.shape).detach())
            ret = ret + self.gamma**t * survival * reward
            aux_loss = aux_loss + self.actor.auxiliary_loss()
            if o.survival_masking:
                survival = survival * (1 - term_prob)
            obs = next_obs
        stats: dict[str, torch.Tensor] = {}
        if self.critic is not None:
            self.critic.requires_grad_(False)
            value = self.critic(obs["critic"]).clamp(max=self.v_max).squeeze(-1)
            alive = survival if o.survival_masking else 1.0 - term_prob
            ret = ret + self.gamma**o.horizon * value * alive
        loss = (-ret + aux_loss).mean()
        stats["grad_norm"] = optimize(self.actor_optimizers, loss, self.actor_params, 1.0, o.grad_clip_value)
        stats["return"] = ret.mean().detach()
        stats["latent_bc_loss"] = self.actor.auxiliary_loss().mean().detach()
        if self.critic is not None:
            stats["value_loss"] = self._fit_critic(rollout, obs)
        return stats

    def _fit_critic(self, rollout: list[TensorDict], last_obs: TensorDict) -> torch.Tensor:
        steps = torch.stack(rollout)
        with torch.no_grad():
            values = self.critic(steps["critic"])
            last_value = self.critic(last_obs["critic"])
        returns = gae_returns(
            values,
            last_value,
            steps["reward"].unsqueeze(-1),
            steps["done"].unsqueeze(-1),
            self.options.gamma,
            self.options.lam,
        ).clamp(max=self.v_max)
        obs, returns = steps["critic"].flatten(0, 1), returns.flatten(0, 1)
        # Not torch.randperm: it draws small permutations on the CPU, which a captured CUDA graph would freeze.
        perm = torch.rand(len(returns) // 4 * 4, device=returns.device).argsort()
        self.critic.requires_grad_(True)
        losses = []
        for _ in range(5):
            for idx in perm.view(4, -1):
                loss = (returns[idx] - self.critic(obs[idx])).pow(2).mean()
                optimize(self.critic_optimizers, loss, self.critic.parameters(), max_norm=1.0)
                losses.append(loss.detach())
        return torch.stack(losses).mean()

    def _graph_state(self) -> list[torch.Tensor]:
        modules = [self.actor, self.wm_env.dynamics]
        state = graph_state(modules, self.actor_optimizers)
        if self.critic is not None:
            state += graph_state([self.critic], self.critic_optimizers)
        return state
