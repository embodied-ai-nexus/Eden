"""rsl_rl model wrapping the SimBaV2 backbone behind per-group encoders (Weave's ``SimBaActorCritic``).

Weave (rsl_rl 3.x) builds one ``ActorCritic`` holding both networks. rsl_rl >= 5 splits actor and critic into two
independent models, each reading its own observation set, so the port is a single :class:`SimBaModel` used twice:

* the actor: ``SimBaModel`` over the ``actor`` observation set with a Gaussian ``distribution_cfg`` — Weave's
  state-independent scalar std;
* the critic: ``SimBaModel`` over the ``critic`` set, deterministic, output width 1.

Each is ``EmpiricalNormalization`` over the concatenated groups -> :class:`~.networks.GroupEncoder` (one MLP per
group) -> :class:`~.networks.SimBa`. That is exactly Weave's ``build_group_backbone``, including the normaliser
sitting *before* the group split, so normalisation statistics are per-feature over the whole concatenated vector.

It subclasses rsl_rl's ``MLPModel`` and only swaps the trunk (``self.mlp``), so everything else — the forward
contract, the distribution, ``update_normalization``, and JIT / ONNX export (which deep-copy ``obs_normalizer`` and
``mlp``) — is rsl_rl's own and follows whichever rsl_rl >= 5 Eden resolves.

Point an actor/critic ``class_name`` at :data:`SIMBA_MODEL_CLASS_NAME`; the option classes in
:mod:`eden.learning.wrappers.rsl_rl.simba.options` do that for you.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from tensordict import TensorDict

from eden.options.learning.rsl_rl.translate import detect_rsl_rl_major_version
from eden.learning.wrappers.rsl_rl.simba.networks import GroupEncoder, SimBa

if detect_rsl_rl_major_version() < 5:
    raise ImportError(
        "eden.learning.wrappers.rsl_rl.simba.model needs rsl-rl-lib >= 5 (the separate actor/critic `MLPModel` "
        f"API); the installed major version is {detect_rsl_rl_major_version()}. Upgrade with "
        "`uv pip install 'rsl-rl-lib>=5'` — translating the runner dict to v3/v4 cannot carry a custom model class."
    )

from rsl_rl.models.mlp_model import MLPModel  # noqa: E402  (after the version guard, on purpose)
from rsl_rl.utils import resolve_nn_activation, unpad_trajectories  # noqa: E402

#: ``class_name`` string rsl_rl resolves to :class:`SimBaModel`.
SIMBA_MODEL_CLASS_NAME = "eden.learning.wrappers.rsl_rl.simba.model:SimBaModel"


class SimBaModel(MLPModel):
    """Per-group MLP encoders feeding a SimBaV2 backbone, as an rsl_rl >= 5 model.

    Parameters
    ----------
    obs : TensorDict
        A batch of observations; only the group widths are read.
    obs_groups : dict[str, list[str]]
        Observation set -> groups, as resolved by rsl_rl.
    obs_set : str
        The set this model reads (``"actor"`` or ``"critic"``).
    output_dim : int
        Output width (action dimension, or 1 for the critic).
    hidden_dim : int
        Width of the SimBaV2 hypersphere.
    num_blocks : int
        Number of SimBaV2 residual blocks.
    expansion : int
        Inner-layer width multiplier of each block.
    latent_dim : int
        Output width of every group encoder.
    encoder_hidden_dims : dict[str, list[int]]
        Per-group hidden widths of the group encoders. A key must name an observation group the env provides; a
        group of this set without a key uses ``default_encoder_hidden_dims``.
    default_encoder_hidden_dims : list[int]
        Hidden widths for groups not in ``encoder_hidden_dims``.
    c_shift : float
        Input shift of the SimBaV2 embedder.
    compile : bool
        ``torch.compile`` the trunk's forward (inductor, fullgraph, static shapes), as Weave does in training. The
        compiled function is kept outside the module state, so ``state_dict`` keys and export are unchanged.
    hidden_dims : list[int] | tuple[int, ...]
        Inherited from rsl_rl's ``MLPModel``; SimBa has no plain MLP trunk, so it must be empty.
    activation : str
        Activation between the layers of the group encoders (Weave: SiLU, which rsl_rl names ``"swish"``).
    obs_normalization : bool
        Normalise the concatenated observation with ``EmpiricalNormalization``.
    distribution_cfg : dict | None
        Output distribution (actor), or ``None`` for a deterministic output (critic).
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dim: int = 512,
        num_blocks: int = 2,
        expansion: int = 4,
        latent_dim: int = 128,
        encoder_hidden_dims: dict[str, list[int]] | None = None,
        default_encoder_hidden_dims: list[int] | tuple[int, ...] = (256,),
        c_shift: float = 3.0,
        compile: bool = False,
        hidden_dims: list[int] | tuple[int, ...] = (),
        activation: str = "swish",
        obs_normalization: bool = True,
        distribution_cfg: dict | None = None,
    ) -> None:
        if len(hidden_dims) != 0:
            raise ValueError(
                f"SimBaModel got hidden_dims={list(hidden_dims)}, but SimBa has no plain MLP trunk to size. Set the "
                "backbone with hidden_dim / num_blocks / expansion and the per-group encoders with "
                "encoder_hidden_dims, and leave hidden_dims empty."
            )
        encoder_hidden_dims = dict(encoder_hidden_dims or {})
        unknown = sorted(set(encoder_hidden_dims) - set(obs.keys()))
        if unknown:
            raise ValueError(
                f"SimBaModel encoder_hidden_dims names observation groups {unknown} that the env does not provide; "
                f"available groups: {sorted(obs.keys())}."
            )
        # rsl_rl's MLPModel builds the normaliser, the distribution and a plain MLP trunk; the trunk is replaced
        # below, so build it as small as possible (a 1-wide hidden layer; rsl_rl's MLP needs at least one) rather
        # than at the MLPModel default size.
        super().__init__(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            hidden_dims=[1],
            activation=activation,
            obs_normalization=obs_normalization,
            distribution_cfg=distribution_cfg,
        )
        trunk_output_dim = self.distribution.input_dim if self.distribution is not None else output_dim
        group_dims = [obs[group].shape[-1] for group in self.obs_groups]
        specs = [list(encoder_hidden_dims.get(group, default_encoder_hidden_dims)) for group in self.obs_groups]
        encoder = GroupEncoder(group_dims, specs, latent_dim, resolve_nn_activation(activation))
        self.mlp = nn.Sequential(
            encoder, SimBa(encoder.out_features, trunk_output_dim, hidden_dim, num_blocks, expansion, c_shift)
        )
        if self.distribution is not None:
            self.distribution.init_mlp_weights(self.mlp)
        self._compile_trunk = compile
        self._compiled_trunk = None

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: Any = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        if not self._compile_trunk:
            return super().forward(obs, masks, hidden_state, stochastic_output)
        if self._compiled_trunk is None:
            self._compiled_trunk = torch.compile(
                self.mlp.forward, backend="inductor", mode="default", fullgraph=True, dynamic=False
            )
        obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
        latent = self.get_latent(obs, masks, hidden_state)
        trunk_output = self._compiled_trunk(latent)
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(trunk_output)
                return self.distribution.sample()
            return self.distribution.deterministic_output(trunk_output)
        return trunk_output

    def __getstate__(self) -> dict[str, Any]:
        # A compiled function is neither picklable nor deep-copyable; rebuild it lazily on the copy instead.
        state = dict(super().__getstate__())
        state["_compiled_trunk"] = None
        return state
