"""rsl_rl learning configuration options."""

from eden.options.learning.rsl_rl.distillation import (
    RslRlDistillationAlgorithmOptions,
    RslRlDistillationRunnerOptions,
)
from eden.options.learning.rsl_rl.rl import (
    RslRlActorCNNOptions,
    RslRlActorOptions,
    RslRlActorRNNOptions,
    RslRlBaseRunnerOptions,
    RslRlCNNModelOptions,
    RslRlCriticCNNOptions,
    RslRlCriticOptions,
    RslRlCriticRNNOptions,
    RslRlDistributionOptions,
    RslRlGaussianDistributionOptions,
    RslRlHeteroscedasticGaussianDistributionOptions,
    RslRlModelOptions,
    RslRlModelWithDistributionOptions,
    RslRlOnPolicyRunnerOptions,
    RslRlPpoAlgorithmOptions,
    RslRlRNNModelOptions,
)
from eden.options.learning.rsl_rl.rnd import RslRlRndOptions
from eden.options.learning.rsl_rl.symmetry import RslRlSymmetryOptions
from eden.options.learning.rsl_rl.translate import detect_rsl_rl_major_version, translate_runner_dict

__all__ = [
    # Shared
    "RslRlRndOptions",
    "RslRlSymmetryOptions",
    # Distribution
    "RslRlDistributionOptions",
    "RslRlGaussianDistributionOptions",
    "RslRlHeteroscedasticGaussianDistributionOptions",
    # Models (canonical)
    "RslRlModelOptions",
    "RslRlModelWithDistributionOptions",
    "RslRlRNNModelOptions",
    "RslRlCNNModelOptions",
    "RslRlActorRNNOptions",
    "RslRlActorCNNOptions",
    # Actor / Critic (role names)
    "RslRlActorOptions",
    "RslRlCriticOptions",
    "RslRlCriticRNNOptions",
    "RslRlCriticCNNOptions",
    # Algorithm / Runner
    "RslRlPpoAlgorithmOptions",
    "RslRlBaseRunnerOptions",
    "RslRlOnPolicyRunnerOptions",
    # Distillation
    "RslRlDistillationAlgorithmOptions",
    "RslRlDistillationRunnerOptions",
    # Translation
    "detect_rsl_rl_major_version",
    "translate_runner_dict",
]
