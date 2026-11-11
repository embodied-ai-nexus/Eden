"""The RMA actor as an rsl-rl-lib >= 5.5.1 model: estimator + scan / privileged / history encoders + backbone.

rsl_rl 5 builds the actor and the critic as two independent models, each reading its own observation set. The RMA
critic is a plain MLP over the full flat observation (rsl_rl's own ``MLPModel``); the actor is :class:`RmaActorModel`,
which owns everything the deployed policy needs — including the base-velocity estimator — so a checkpoint's
``actor_state_dict`` alone is a deployable policy.

Its ``forward`` takes two switches beyond rsl_rl's contract, both defaulting to the *deployment* composition, so any
caller that only knows rsl_rl's interface (``eden inference``, ``eden deploy``, the runner's inference policy) gets
the policy as deployed:

``hist_encoding`` (default ``True``)
    Adaptation latent from the proprioception-history encoder; ``False`` uses the privileged encoder instead (what
    PPO trains through).
``estimate`` (default ``True``)
    Overwrite the privileged-explicit chunk with the estimator's prediction from proprioception; ``False`` keeps the
    ground truth the env provides (what PPO trains on).

:class:`~eden.learning.wrappers.rsl_rl.rma.ppo.RmaPPO` drives the switches during training.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from tensordict import TensorDict

from eden.learning.wrappers.rsl_rl import MIN_RSL_RL_VERSION
from eden.learning.wrappers.rsl_rl.rma.dims import RmaObsDims
from eden.learning.wrappers.rsl_rl.rma.networks import Estimator, RmaActor
from eden.options.learning.rsl_rl.translate import require_rsl_rl_version

require_rsl_rl_version(
    MIN_RSL_RL_VERSION,
    __name__,
    "the separate actor/critic model API and rsl_rl.utils.resolve_class; translating the runner dict to an older "
    "rsl_rl cannot carry a custom model class",
)

from rsl_rl.models.mlp_model import MLPModel  # noqa: E402  (after the version guard, on purpose)
from rsl_rl.utils import resolve_class, resolve_nn_activation  # noqa: E402

#: ``class_name`` string rsl_rl resolves to :class:`RmaActorModel`.
RMA_ACTOR_MODEL_CLASS_NAME = "eden.learning.wrappers.rsl_rl.rma.model:RmaActorModel"


class RmaActorModel(MLPModel):
    """RMA actor over one flat ``[prop | scan | priv_explicit | priv_latent | history]`` observation.

    Subclasses rsl_rl's ``MLPModel`` for its distribution plumbing (``output_*``, log-prob, KL, entropy) and its
    recurrent no-ops, but replaces the trunk: there is no ``self.mlp``, and observation normalization is refused —
    the task's terms carry hand-tuned fixed scales, and the chunks are sliced by position.

    Parameters
    ----------
    obs, obs_groups, obs_set, output_dim
        rsl_rl's model arguments; the groups of ``obs_set`` concatenate to the flat layout.
    num_prop, num_scan, num_priv_explicit, num_priv_latent, num_hist : int
        Chunk widths (:class:`~eden.learning.wrappers.rsl_rl.rma.dims.RmaObsDims`).
    scan_encoder_dims, priv_encoder_dims : list[int]
        Encoder widths; the last entry of each is its latent width.
    hidden_dims : list[int]
        Actor backbone hidden widths.
    activation : str
        Activation everywhere but the scan latent (Tanh) and the estimator output (linear).
    estimator_hidden_dims : list[int]
        Hidden widths of the base-velocity estimator.
    tanh_backbone_output : bool
        Squash the action mean with Tanh.
    obs_normalization : bool
        Must be ``False``.
    distribution_cfg : dict
        The output distribution (a state-independent Gaussian for the RMA teacher).
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        num_prop: int = 53,
        num_scan: int = 132,
        num_priv_explicit: int = 9,
        num_priv_latent: int = 29,
        num_hist: int = 10,
        scan_encoder_dims: tuple[int, ...] | list[int] = (128, 64, 32),
        priv_encoder_dims: tuple[int, ...] | list[int] = (64, 20),
        hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
        activation: str = "elu",
        estimator_hidden_dims: tuple[int, ...] | list[int] = (128, 64),
        tanh_backbone_output: bool = False,
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
    ) -> None:
        nn.Module.__init__(self)  # not MLPModel.__init__: it would build the plain MLP trunk this model replaces
        if obs_normalization:
            raise ValueError(
                "RmaActorModel(obs_normalization=True): the RMA actor slices its observation by position and trains "
                "on the task's fixed term scales; set obs_normalization=False on the actor options."
            )
        if distribution_cfg is None:
            raise ValueError(
                "RmaActorModel needs a distribution_cfg (the RMA teacher's actor is stochastic); pass "
                "RslRlGaussianDistributionOptions(init_std=1.0, std_type='scalar')."
            )
        self.obs_groups, self.obs_dim = self._get_obs_dim(obs, obs_groups, obs_set)
        self.dims = RmaObsDims(num_prop, num_scan, num_priv_explicit, num_priv_latent, num_hist)
        if self.obs_dim != self.dims.total:
            raise ValueError(
                f"The '{obs_set}' observation set {self.obs_groups} is {self.obs_dim}-d, but the RMA actor options "
                f"describe {self.dims.layout()}. Make the task's {obs_set} groups emit that layout, or set the "
                "actor options' num_prop / num_scan / num_priv_explicit / num_priv_latent / num_hist to the task's."
            )
        self.obs_normalization = False
        self.obs_normalizer = nn.Identity()

        dist_class, dist_cfg = resolve_class(distribution_cfg)
        self.distribution = dist_class(output_dim, **dist_cfg)
        if self.distribution.input_dim != output_dim:
            raise ValueError(
                f"RmaActorModel's backbone emits the {output_dim}-d action mean only; {dist_class.__name__} needs "
                f"{self.distribution.input_dim} inputs. Use a state-independent Gaussian distribution."
            )
        act = resolve_nn_activation(activation)
        self.net = RmaActor(
            self.dims,
            output_dim,
            list(scan_encoder_dims),
            list(priv_encoder_dims),
            list(hidden_dims),
            act,
            tanh_backbone_output=tanh_backbone_output,
        )
        self.estimator = Estimator(num_prop, num_priv_explicit, list(estimator_hidden_dims), act)

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state=None,
        stochastic_output: bool = False,
        *,
        hist_encoding: bool = True,
        estimate: bool = True,
    ) -> torch.Tensor:
        """Action (mean, or a sample with ``stochastic_output``); the switches are described in the module docstring."""
        mean = self.mean(self.get_flat_obs(obs), hist_encoding=hist_encoding, estimate=estimate)
        if stochastic_output:
            self.distribution.update(mean)
            return self.distribution.sample()
        return self.distribution.deterministic_output(mean)

    def mean(self, flat: torch.Tensor, *, hist_encoding: bool, estimate: bool) -> torch.Tensor:
        """Action mean from the flat observation (no distribution update)."""
        if estimate:
            flat = self.with_estimate(flat)
        return self.net(flat, hist_encoding)

    def with_estimate(self, flat: torch.Tensor) -> torch.Tensor:
        """A copy of ``flat`` whose privileged-explicit chunk is the estimator's prediction (detached).

        Detached: the policy loss never trains the estimator — it has its own regression loss and optimizer.
        """
        d = self.dims
        flat = flat.clone()
        flat[:, d.priv_explicit_slice] = self.estimator(flat[:, d.prop_slice]).detach()
        return flat

    def get_flat_obs(self, obs: TensorDict) -> torch.Tensor:
        """The concatenated observation groups of this model's set."""
        return torch.cat([obs[group] for group in self.obs_groups], dim=-1)

    def policy_parameters(self):
        """Every parameter but the estimator's — what the PPO optimizer and its gradient clip cover."""
        return (p for name, p in self.named_parameters() if not name.startswith("estimator."))

    def get_latent(self, obs, masks=None, hidden_state=None) -> torch.Tensor:
        return self.get_flat_obs(obs)

    def as_jit(self) -> nn.Module:
        """A TorchScript-compatible deployment policy over the flat observation."""
        return _ExportedRmaActor(self)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        """An ONNX-exportable deployment policy over the flat observation.

        ``verbose`` is unused and optional: rsl-rl-lib 5.5.2 calls ``as_onnx()``, earlier 5.x ``as_onnx(verbose=...)``.
        """
        return _ExportedRmaActor(self)


class _ExportedRmaActor(nn.Module):
    """The deployment composition (estimator injected, history latent) on a pre-concatenated observation."""

    is_recurrent: bool = False

    def __init__(self, model: RmaActorModel) -> None:
        super().__init__()
        d = model.dims
        self.net = copy.deepcopy(model.net)
        self.estimator = copy.deepcopy(model.estimator)
        self.prop_end = d.num_prop
        self.pe_start = d.priv_explicit_slice.start
        self.pe_end = d.priv_explicit_slice.stop
        self.input_size = d.total

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        estimate = self.estimator(x[:, : self.prop_end])
        x = torch.cat([x[:, : self.pe_start], estimate, x[:, self.pe_end :]], dim=-1)
        return self.net(x, True)

    @torch.jit.export
    def reset(self) -> None:
        pass

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        return ["obs"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]
