"""AssistMimic's role-separated actor, initialized from PHC, and its critic.

AssistMimic (``amp_network_pnn_multi_builder.py`` with ``weight_share: false``, ``residual_mode: false``) gives
each role its own copy of PHC's primitive-0 MLP. Every layer is copied from the checkpoint; the first layer
is widened to the agent's full observation, with PHC's weights on the leading ``self + imitation`` columns
and zeros on the auxiliary ones. At initialization each role's action is therefore exactly PHC's action for
the PHC part of its observation — provided the observation is normalized with PHC's running statistics,
which this module also loads (aux columns: mean 0, variance 1, as AssistMimic does).
"""

from __future__ import annotations

import functools

import torch
from rl_games.algos_torch import model_builder
from rl_games.algos_torch.network_builder import NetworkBuilder
from torch import nn

from assist_training import ROLES

PHC_UNITS = (2048, 1536, 1024, 1024, 512, 512)
#: Key prefix of PHC's primitive-0 actor in an rl_games PHC checkpoint (``phc_3`` or :func:`export_phc`'s).
PHC_ACTOR_PREFIX = "a2c_network.pnn.actors.0."


def mlp(in_dim: int, out_dim: int, units=PHC_UNITS) -> nn.Sequential:
    layers, d = [], in_dim
    for u in units:
        layers += [nn.Linear(d, u), nn.SiLU()]
        d = u
    layers.append(nn.Linear(d, out_dim))
    return nn.Sequential(*layers)


class FrozenObsNormalizer(nn.Module):
    """rl_games ``RunningMeanStd`` at eval: ``clamp((x - mean) / sqrt(var + 1e-5), -5, 5)``.

    Frozen: PHC's statistics carry a count of ~1e11 samples, so rl_games' updates never move them either.
    """

    def __init__(self, dim: int):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ((x - self.mean) / torch.sqrt(self.var + 1e-5)).clamp(-5.0, 5.0)


class AssistActorCritic(nn.Module):
    """Two role actors (caregiver = role 0, recipient = role 1) with fixed-sigma Gaussians, and one critic.

    Also the ``a2c_network`` of rl_games' ``continuous_a2c_logstd`` model (:class:`AssistNetworkBuilder`):
    :meth:`forward` takes rl_games' ``input_dict`` and returns ``(mu, logstd, value, None)``. The model's own
    input normalization stays off (``normalize_input=False``) — this module normalizes with PHC's frozen
    statistics, which rl_games' running statistics would drift away from over a run.

    Rows may come in any order: each is sent to its role's actor by its ``RoleFlag`` observation column
    (``role_column``), so it does not matter how rl_games lays out or slices the batch. The rows are sorted by
    role and each actor takes one half — no row mask, whose row count would be a GPU->CPU sync — so a batch
    must hold as many caregiver rows as recipient ones. Every batch rl_games makes does when each env is two
    agents and ``minibatch_size`` is a multiple of ``2 * horizon_length`` (a minibatch is whole agents' horizons,
    agents interleaved by env); a batch that does not trips a device-side assert.

    Parameters
    ----------
    obs_dim : int
        Per-agent observation size.
    action_dim : int
        Per-agent action size.
    role_column : int
        Column of the observation holding the role flag (0 caregiver, 1 recipient).
    input_dim : int | None
        Feed only the first ``input_dim`` observation columns to the networks (PHC adaptation: PHC's 934).
        ``None``: all of them.
    init_sigma : float
        Fixed log-std (AssistMimic ``sigma_init: -2.9``).
    """

    def __init__(
        self, obs_dim: int, action_dim: int, role_column: int, input_dim: int | None = None, init_sigma: float = -2.9
    ):
        super().__init__()
        if not 0 <= role_column < obs_dim:
            raise ValueError(
                f"AssistActorCritic: role_column={role_column} is outside the {obs_dim}-column observation."
            )
        self.obs_dim, self.action_dim, self.role_column = obs_dim, action_dim, role_column
        self.input_dim = obs_dim if input_dim is None else input_dim
        self.normalizer = FrozenObsNormalizer(self.input_dim)
        self.actors = nn.ModuleList([mlp(self.input_dim, action_dim), mlp(self.input_dim, action_dim)])
        self.critic = mlp(self.input_dim, 1)
        self.register_buffer("log_std", torch.full((action_dim,), init_sigma))
        # The checkpoint the shared normalizer's statistics came from (``load_phc``), if any.
        self._normalizer_source: str | None = None

    def load_phc(self, checkpoint_path: str, phc_obs_dim: int, roles: tuple[int, ...] = (0, 1)) -> dict:
        """Copy PHC's ``pnn.actors.0`` and running statistics into the ``roles`` actors; return a summary.

        ``checkpoint_path`` is an rl_games PHC checkpoint (``output/HumanoidIm/phc_3/Humanoid.pth``) or one written
        by :func:`export_phc` after adapting PHC to Genesis, which has the same layout.

        Both roles share one observation normalizer, so loading the roles from two checkpoints is only valid
        when they carry the same running statistics (two exports of one ``adapt`` run do); otherwise the
        second load would silently re-normalize the first role's actor, and this raises instead.
        """
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = ckpt["model"]
        prefix = PHC_ACTOR_PREFIX
        phc_layers = sorted({int(k[len(prefix) :].split(".")[0]) for k in state if k.startswith(prefix)})
        ours = [i for i, m in enumerate(self.actors[0]) if isinstance(m, nn.Linear)]
        if phc_layers != ours:
            raise ValueError(
                f"{checkpoint_path}: PHC actor Linear layers {phc_layers} do not line up with ours {ours}; "
                f"expected PHC's im_pnn_big units {PHC_UNITS}."
            )
        first_w = state[f"{prefix}0.weight"]
        if first_w.shape[1] != phc_obs_dim:
            raise ValueError(
                f"{checkpoint_path}: PHC actor takes {first_w.shape[1]} inputs, but the observation's PHC slice is "
                f"{phc_obs_dim} (358 self + 576 imitation for the 24-body SMPL humanoid). A SMPL-X checkpoint "
                "(e.g. phc_x_pnn, 2026 inputs) does not fit this robot; use output/HumanoidIm/phc_3/Humanoid.pth."
            )
        rms = ckpt["running_mean_std"]
        mean, var = rms["running_mean"].float(), rms["running_var"].float()
        if self._normalizer_source is not None:
            loaded_mean = self.normalizer.mean[:phc_obs_dim].cpu()
            loaded_var = self.normalizer.var[:phc_obs_dim].cpu()
            if not (torch.allclose(mean, loaded_mean) and torch.allclose(var, loaded_var)):
                raise ValueError(
                    f"{checkpoint_path}: its observation running mean/var differ from those of "
                    f"{self._normalizer_source}, loaded before it. The two role actors share one normalizer, so "
                    "each would see the other's statistics. Initialize both roles from checkpoints of one source "
                    "(both exports of the same `main.py adapt` run, or the same phc_3) rather than mixing them."
                )
        with torch.no_grad():
            for r in roles:
                actor = self.actors[r]
                for i in ours:
                    w, b = state[f"{prefix}{i}.weight"], state[f"{prefix}{i}.bias"]
                    if i == 0:
                        actor[i].weight.zero_()
                        actor[i].weight[:, :phc_obs_dim].copy_(w)
                    else:
                        actor[i].weight.copy_(w)
                    actor[i].bias.copy_(b)
            self.normalizer.mean.zero_()
            self.normalizer.var.fill_(1.0)
            self.normalizer.mean[:phc_obs_dim].copy_(mean)
            self.normalizer.var[:phc_obs_dim].copy_(var)
        self._normalizer_source = checkpoint_path
        return {"epoch": ckpt.get("epoch"), "layers": phc_layers, "rms_count": float(rms["count"])}

    def load_phc_pair(self, paths: tuple[str, str], phc_obs_dim: int) -> None:
        """:meth:`load_phc` the caregiver's actor from ``paths[0]`` and the recipient's from ``paths[1]``."""
        for role, (name, path) in enumerate(zip(ROLES, paths)):
            print(f"PHC {name}:", self.load_phc(path, phc_obs_dim, roles=(role,)))

    def _inputs(self, obs: torch.Tensor) -> torch.Tensor:
        # A non-finite observation (NaN the solver left in an env) is read as zeros, not passed on to poison the
        # whole network in the next update.
        return self.normalizer(torch.nan_to_num(obs[:, : self.input_dim], nan=0.0, posinf=0.0, neginf=0.0))

    def act_mean(self, obs: torch.Tensor) -> torch.Tensor:
        """Deterministic action of each row of ``obs`` ``(B, obs_dim)``, by its role's actor."""
        return self._act_mean(obs, self._inputs(obs))

    def _act_mean(self, obs: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """:meth:`act_mean` of rows whose network inputs ``x`` are already normalized."""
        order = torch.argsort(obs[:, self.role_column], stable=True)
        half = obs.shape[0] // 2
        role = obs[order, self.role_column]
        torch._assert_async(
            (obs.shape[0] % 2 == 0) & (role[half - 1] < 0.5) & (role[half] > 0.5),
            "AssistActorCritic: the batch does not hold as many caregiver rows as recipient ones",
        )
        xs = x[order]
        mu_sorted = torch.cat([self.actors[0](xs[:half]), self.actors[1](xs[half:])], dim=0)
        return torch.empty_like(mu_sorted).index_copy_(0, order, mu_sorted)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(self._inputs(obs)).squeeze(-1)

    @property
    def std(self) -> torch.Tensor:
        return self.log_std.exp()

    # --- rl_games a2c_network interface ----------------------------------------------------------------------------
    def forward(self, input_dict: dict):
        obs = input_dict["obs"]
        x = self._inputs(obs)
        mu = self._act_mean(obs, x)
        return mu, self.log_std.expand_as(mu), self.critic(x), None

    def is_rnn(self) -> bool:
        return False

    def get_default_rnn_state(self):
        return None

    def get_value_layer(self):
        return None

    def get_aux_loss(self):
        return None


class AssistNetworkBuilder(NetworkBuilder):
    """rl_games network builder of :class:`AssistActorCritic`, registered as ``"assistmimic"`` (:func:`register`).

    Parameters
    ----------
    role_column, input_dim
        As in :class:`AssistActorCritic`.
    phc : tuple[str, str] | None
        Checkpoints to initialize the caregiver and the recipient actor from (:meth:`AssistActorCritic.load_phc`);
        ``None`` keeps the random initialization (a run resumed from, or played from, an rl_games checkpoint).
    phc_obs_dim : int
        Size of the observation's PHC slice.
    """

    def __init__(self, role_column: int, input_dim: int | None, phc: tuple[str, str] | None, phc_obs_dim: int):
        super().__init__()
        self.role_column, self.input_dim, self.phc, self.phc_obs_dim = role_column, input_dim, phc, phc_obs_dim

    def load(self, params) -> None:
        self.params = params

    def build(self, name, **kwargs) -> AssistActorCritic:
        net = AssistActorCritic(kwargs["input_shape"][0], kwargs["actions_num"], self.role_column, self.input_dim)
        if self.phc is not None:
            net.load_phc_pair(self.phc, self.phc_obs_dim)
        return net


def register(role_column: int, input_dim: int | None, phc: tuple[str, str] | None, phc_obs_dim: int) -> None:
    """Register :class:`AssistNetworkBuilder` with rl_games as network ``"assistmimic"`` (before the agent is built)."""
    model_builder.register_network(
        "assistmimic", functools.partial(AssistNetworkBuilder, role_column, input_dim, phc, phc_obs_dim)
    )


def load_policy(path: str, obs_dim: int, role_column: int, input_dim: int | None, device) -> AssistActorCritic:
    """The :class:`AssistActorCritic` of an rl_games checkpoint written by ``main.py adapt`` / ``train``."""
    state = torch.load(path, map_location="cpu", weights_only=False)["model"]
    prefix = "a2c_network."
    net = {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
    policy = AssistActorCritic(obs_dim, net["log_std"].numel(), role_column, input_dim)
    policy.load_state_dict(net)
    return policy.to(device).eval()


def export_phc(policy: AssistActorCritic, role: int, path: str, phc_obs_dim: int) -> None:
    """Write role ``role``'s actor as a PHC-layout checkpoint (``pnn.actors.0`` + ``running_mean_std``).

    The first layer keeps only the PHC observation columns, so :meth:`AssistActorCritic.load_phc` reads it
    back exactly like the original ``phc_3`` checkpoint.
    """
    actor = policy.actors[role]
    model = {}
    for i, m in enumerate(actor):
        if isinstance(m, nn.Linear):
            w = m.weight.detach().cpu()
            model[f"{PHC_ACTOR_PREFIX}{i}.weight"] = w[:, :phc_obs_dim].clone() if i == 0 else w.clone()
            model[f"{PHC_ACTOR_PREFIX}{i}.bias"] = m.bias.detach().cpu().clone()
    rms = {
        "running_mean": policy.normalizer.mean[:phc_obs_dim].detach().cpu().double(),
        "running_var": policy.normalizer.var[:phc_obs_dim].detach().cpu().double(),
        "count": torch.tensor(1.25e11, dtype=torch.float64),
    }
    torch.save({"model": model, "running_mean_std": rms, "epoch": None}, path)
