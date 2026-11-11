"""Actor / critic options for :class:`~eden.learning.wrappers.rsl_rl.simba.model.SimBaModel`.

Declarative only — importing this module does not pull in rsl_rl's model API, so a task config that uses it still
imports under any rsl_rl Eden supports. The model its ``class_name`` names needs rsl-rl-lib >= 5 and says so when
imported without it. The defaults are Weave's G1-Inspire-HOI networks.
"""

from __future__ import annotations

from eden.options.learning.rsl_rl.rl import (
    RslRlGaussianDistributionOptions,
    RslRlModelOptions,
    RslRlModelWithDistributionOptions,
)

# A literal (not imported from the model module) so this module stays importable without rsl-rl-lib >= 5;
# ``tests/learning/wrappers/rsl_rl/simba/test_model.py`` pins it against the implementation.
_SIMBA_MODEL_CLASS_NAME = "eden.learning.wrappers.rsl_rl.simba.model:SimBaModel"


class _SimBaFields(RslRlModelOptions):
    """SimBaV2 fields shared by the actor and critic options.

    Parameters
    ----------
    class_name : str
        The model class rsl_rl resolves; leave at the default.
    hidden_dim : int
        Width of the SimBaV2 hypersphere.
    num_blocks : int
        Number of SimBaV2 residual blocks.
    expansion : int
        Inner-layer width multiplier of each block.
    latent_dim : int
        Output width of every per-group encoder.
    encoder_hidden_dims : dict[str, list[int]]
        Per-group encoder hidden widths, keyed by observation-group name.
    default_encoder_hidden_dims : list[int]
        Encoder hidden widths for a group without an ``encoder_hidden_dims`` entry.
    c_shift : float
        Input shift of the SimBaV2 embedder.
    compile : bool
        ``torch.compile`` the trunk forward (Weave compiles in training).
    hidden_dims : list[int]
        Unused by SimBa and must stay empty (inherited from the MLP model options).
    activation : str
        Activation inside the per-group encoders.
    obs_normalization : bool
        Empirical normalisation of the concatenated observation.
    """

    class_name: str = _SIMBA_MODEL_CLASS_NAME
    hidden_dim: int = 512
    num_blocks: int = 2
    expansion: int = 4
    latent_dim: int = 128
    encoder_hidden_dims: dict[str, list[int]] = {}
    default_encoder_hidden_dims: list[int] = [256]
    c_shift: float = 3.0
    compile: bool = False
    hidden_dims: list[int] = []
    activation: str = "swish"
    obs_normalization: bool = True

    def model_post_init(self, context) -> None:
        super().model_post_init(context)
        if self.hidden_dims:
            raise ValueError(
                f"{type(self).__name__} got hidden_dims={self.hidden_dims}, but SimBa has no plain MLP trunk. Size "
                "the backbone with hidden_dim / num_blocks / expansion and the encoders with encoder_hidden_dims."
            )


class RslRlSimBaCriticOptions(_SimBaFields):
    """Critic: a deterministic SimBaV2 value network (Weave: 2 blocks).

    Parameters
    ----------
    num_blocks : int
        Number of SimBaV2 residual blocks.
    """

    num_blocks: int = 2


class RslRlSimBaActorOptions(_SimBaFields, RslRlModelWithDistributionOptions):
    """Actor: a SimBaV2 policy with a state-independent Gaussian std (Weave: 1 block, scalar std 1.0).

    Parameters
    ----------
    num_blocks : int
        Number of SimBaV2 residual blocks.
    distribution_cfg : RslRlGaussianDistributionOptions
        Output distribution; Weave's is a learned per-action scalar std initialised at 1.0.
    """

    num_blocks: int = 1
    distribution_cfg: RslRlGaussianDistributionOptions = RslRlGaussianDistributionOptions(
        init_std=1.0, std_type="scalar"
    )
