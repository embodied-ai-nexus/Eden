"""PPO with a Muon + AdamW optimizer pair on rsl-rl-lib 5.

Select it with :class:`RslRlMuonPpoAlgorithmOptions` as the runner's ``algorithm``. Only the option class is
re-exported here: rsl_rl imports :class:`~.ppo.MuonPPO` when it resolves its ``class_name``, so importing this
package needs neither rsl-rl-lib >= 5 nor ``torch.optim.Muon``.
"""

from eden.learning.wrappers.rsl_rl.muon_ppo.options import RslRlMuonPpoAlgorithmOptions

__all__ = ["RslRlMuonPpoAlgorithmOptions"]
