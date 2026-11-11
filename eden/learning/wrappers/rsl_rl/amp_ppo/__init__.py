"""PPO with an adversarial motion prior (AMP) style reward on rsl-rl-lib 5.

Select it with :class:`RslRlAmpPpoAlgorithmOptions` as the runner's ``algorithm``. Only the option class is re-exported
here: rsl_rl imports :class:`~.ppo.AmpPPO` when it resolves its ``class_name``, so importing this package does not
need rsl-rl-lib >= 5.5.1.
"""

from eden.learning.wrappers.rsl_rl.amp_ppo.options import RslRlAmpPpoAlgorithmOptions

__all__ = ["RslRlAmpPpoAlgorithmOptions"]
