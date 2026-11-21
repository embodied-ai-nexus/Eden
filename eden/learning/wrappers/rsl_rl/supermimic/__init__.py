"""SuperMimic: a motion tracker trained through one learned, differentiable world model.

:class:`SuperMimicRunnerOptions` as a task's ``runner_options`` makes ``eden train`` build
:class:`~eden.learning.wrappers.rsl_rl.supermimic.runner.SuperMimicRunner`. Importing this package imports the options
only.
"""

from eden.learning.wrappers.rsl_rl.supermimic.options import (
    PolicyOptions,
    ReturnSamplerOptions,
    RewardTermsOptions,
    SimContextOptions,
    SuperMimicRunnerOptions,
    VelocityGoalOptions,
    WorldModelOptions,
)

__all__ = [
    "PolicyOptions",
    "ReturnSamplerOptions",
    "RewardTermsOptions",
    "SimContextOptions",
    "SuperMimicRunnerOptions",
    "VelocityGoalOptions",
    "WorldModelOptions",
]
