"""ConTrack's constrained-MDP PPO, trained through the external ``rsl_rl_contrack`` fork of rsl_rl.

The ``contrack`` task selects it with :class:`RslRlContrackRunnerOptions` as its ``runner_options``, whose
``class_name`` names :class:`~.runner.ContrackOnPolicyRunner`. Only the option classes are re-exported here: importing
the runner needs the backend (``pip install 'git+https://github.com/Kashu7100/rsl_rl_contrack@7b6c043f…'``), the
options do not.

- :mod:`.options` — runner / algorithm options (PPO + the CMDP dual controller).
- :mod:`.env` — the VecEnv wrapper that adds the ``rg`` / ``rs`` / ``rp`` reward-group extras.
- :mod:`.runner` — the fork's runner behind Eden's runner signature.
"""

from eden.learning.wrappers.rsl_rl_contrack.options import RslRlContrackAlgorithmOptions, RslRlContrackRunnerOptions

__all__ = ["RslRlContrackAlgorithmOptions", "RslRlContrackRunnerOptions"]
