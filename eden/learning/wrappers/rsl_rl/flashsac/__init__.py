"""FlashSAC (off-policy, github.com/Holiday-Robot/rsl_rl_flashsac) behind Eden's rsl_rl wrapper.

:class:`RslRlFlashSacRunnerOptions` as a task's ``runner_options`` makes ``eden train`` build
:class:`~eden.learning.wrappers.rsl_rl.flashsac.runner.FlashSacVecEnvWrapper` and ``rsl_rl_flashsac``'s
``OffPolicyRunner``. The options import without the backend; training needs
``uv sync --inexact --extra flashsac --group flashsac-git``.
"""

from eden.learning.wrappers.rsl_rl.flashsac.options import (
    RslRlFlashSacActorOptions,
    RslRlFlashSacAlgorithmOptions,
    RslRlFlashSacCriticOptions,
    RslRlFlashSacRunnerOptions,
)

__all__ = [
    "RslRlFlashSacActorOptions",
    "RslRlFlashSacAlgorithmOptions",
    "RslRlFlashSacCriticOptions",
    "RslRlFlashSacRunnerOptions",
]
