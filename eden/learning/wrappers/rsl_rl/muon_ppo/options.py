"""Algorithm options for :class:`~eden.learning.wrappers.rsl_rl.muon_ppo.ppo.MuonPPO`.

Declarative only — importing this module pulls in neither rsl_rl's algorithm API nor ``torch.optim.Muon``, so a task
config that uses it still imports under any rsl_rl Eden supports. The algorithm its ``class_name`` names needs
rsl-rl-lib >= 5 and torch >= 2.10, and says so when imported without them.
"""

from __future__ import annotations

from eden.options.learning.rsl_rl.rl import RslRlPpoAlgorithmOptions

# A literal (not imported from the algorithm module) so this module stays importable without rsl-rl-lib >= 5;
# ``tests/learning/wrappers/rsl_rl/muon_ppo/test_ppo.py`` pins it against the implementation.
_MUON_PPO_CLASS_NAME = "eden.learning.wrappers.rsl_rl.muon_ppo.ppo:MuonPPO"


class RslRlMuonPpoAlgorithmOptions(RslRlPpoAlgorithmOptions):
    """PPO with Muon (2-D weights) + AdamW (the rest), Weave's hyperparameters as defaults.

    Parameters
    ----------
    class_name : str
        The algorithm class rsl_rl resolves; leave at the default.
    learning_rate : float
        Initial learning rate of both optimizers (then adapted to ``desired_kl``).
    entropy_coef : float
        Entropy bonus coefficient.
    weight_decay : float
        AdamW weight decay.
    muon_weight_decay : float
        Muon weight decay (Weave leaves ``torch.optim.Muon``'s default, 0.1).
    """

    class_name: str = _MUON_PPO_CLASS_NAME
    learning_rate: float = 1.0e-4
    entropy_coef: float = 5.0e-4
    weight_decay: float = 0.01
    muon_weight_decay: float = 0.1
