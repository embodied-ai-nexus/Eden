"""SimBaV2 actor-critic on rsl-rl-lib 5: one MLP encoder per observation group feeding a SimBaV2 backbone.

Select it with :class:`RslRlSimBaActorOptions` / :class:`RslRlSimBaCriticOptions` as the runner's ``actor`` /
``critic``. Only the option classes are re-exported here: rsl_rl imports :class:`~.model.SimBaModel` when it resolves
their ``class_name``, so importing this package does not need rsl-rl-lib >= 5.
"""

from eden.learning.wrappers.rsl_rl.simba.options import RslRlSimBaActorOptions, RslRlSimBaCriticOptions

__all__ = ["RslRlSimBaActorOptions", "RslRlSimBaCriticOptions"]
