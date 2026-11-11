"""HIMLoco — OpenHomie's learner (HIMPPO) — on rsl-rl-lib 3–5.

:class:`HIMPPO` (rollout-time symmetry, actor + critic mirror losses), :class:`HIMActorCritic`
(the actor sees the newest frame plus :class:`HIMEstimator`'s velocity and dynamics latent) and the
doubled :class:`HIMRolloutStorage`. Select it with :class:`HIMPPOAlgorithmOptions` as the runner's
``algorithm``; rsl-rl-lib's own ``OnPolicyRunner`` builds and drives it.
"""

from eden.learning.wrappers.rsl_rl.him.actor_critic import HIMActorCritic
from eden.learning.wrappers.rsl_rl.him.estimator import HIMEstimator
from eden.learning.wrappers.rsl_rl.him.options import HIMPPOAlgorithmOptions
from eden.learning.wrappers.rsl_rl.him.ppo import HIMPPO
from eden.learning.wrappers.rsl_rl.him.storage import HIMRolloutStorage

__all__ = ["HIMPPO", "HIMActorCritic", "HIMEstimator", "HIMPPOAlgorithmOptions", "HIMRolloutStorage"]
