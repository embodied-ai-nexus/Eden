"""PPO with a Muon + AdamW optimizer pair, ported from Weave's ``MuonPPO``.

Weave (github.com/xiaohu-art/Weave, ``g1_hoi_learning/algorithms/{optimizers,ppo/algorithm}.py``) keeps rsl_rl's
PPO objective unchanged and swaps only the optimizer: every 2-D weight goes to ``torch.optim.Muon`` (with
``adjust_lr_fn="match_rms_adamw"``, so one learning rate means the same step size for both), everything else —
biases, SimBaV2 scalers, the action std — to ``AdamW``. The adaptive-KL schedule writes the learning rate into every
``param_group``, and :class:`MuonAdamW` exposes both optimizers' groups, so the schedule drives both.

Behavioural differences against Weave that come from the rsl_rl version rather than this port (Weave pins 3.x,
this runs on rsl_rl >= 5):

* rsl_rl 5 clips the actor's and the critic's gradient norms separately (each to ``max_grad_norm``); rsl_rl 3
  clipped the joint actor-critic norm.
* The adaptive-KL learning-rate floor is 1e-5 in rsl_rl 5 (1e-6 in Weave's copy); at Weave's 1e-4 start it binds
  only after a long run of KL overshoots.
* Weave's update loop also logged ``clip_fraction`` and ``kl``; only ``explained_variance`` is kept here (it is
  computable before the update without re-implementing it).

``torch.optim.Muon`` needs torch >= 2.10.
"""

from __future__ import annotations

import warnings
from typing import Any

import torch
import torch.nn as nn

from eden.options.learning.rsl_rl.translate import detect_rsl_rl_major_version

if detect_rsl_rl_major_version() < 5:
    raise ImportError(
        "eden.learning.wrappers.rsl_rl.muon_ppo.ppo needs rsl-rl-lib >= 5 (PPO over separate actor/critic models); the "
        f"installed major version is {detect_rsl_rl_major_version()}. Upgrade with `uv pip install 'rsl-rl-lib>=5'`."
    )

from rsl_rl.algorithms import PPO  # noqa: E402  (after the version guard, on purpose)

#: ``class_name`` string rsl_rl resolves to :class:`MuonPPO`.
MUON_PPO_CLASS_NAME = "eden.learning.wrappers.rsl_rl.muon_ppo.ppo:MuonPPO"


class MuonAdamW(torch.optim.Optimizer):
    """Muon for 2-D weights and AdamW for every other parameter, behind one optimizer interface.

    ``param_groups`` is the concatenation of both optimizers' groups, so writing ``group["lr"]`` (what rsl_rl's
    adaptive schedule does) reaches both, and ``state_dict`` / ``load_state_dict`` round-trip both.

    Parameters
    ----------
    modules : list[nn.Module]
        Modules whose parameters to optimize; a parameter shared between them is registered once.
    lr : float
        Learning rate of both optimizers.
    weight_decay : float
        AdamW weight decay.
    muon_weight_decay : float
        Muon weight decay. Weave leaves it at ``torch.optim.Muon``'s default (0.1).
    ignore_frozen : bool
        Skip parameters with ``requires_grad=False``.
    """

    def __init__(
        self,
        modules: list[nn.Module],
        lr: float,
        weight_decay: float = 0.01,
        muon_weight_decay: float = 0.1,
        ignore_frozen: bool = True,
    ) -> None:
        if not hasattr(torch.optim, "Muon"):
            raise ImportError(
                f"torch.optim.Muon is unavailable in torch {torch.__version__} ({torch.__file__}); MuonPPO needs "
                "torch >= 2.10. Upgrade torch, or train with the stock PPO algorithm options instead."
            )
        seen: set[int] = set()
        self.muon_params: list[nn.Parameter] = []
        self.adamw_params: list[nn.Parameter] = []
        for module in modules:
            for param in module.parameters():
                if id(param) in seen or (ignore_frozen and not param.requires_grad):
                    continue
                seen.add(id(param))
                (self.muon_params if param.dim() == 2 else self.adamw_params).append(param)
        if not self.muon_params and not self.adamw_params:
            raise ValueError(
                "MuonAdamW got no trainable parameters: every parameter of the given modules has "
                "requires_grad=False (pass ignore_frozen=False to register frozen ones anyway)."
            )

        self.optimizers: list[torch.optim.Optimizer] = []
        if self.muon_params:
            self.optimizers.append(
                torch.optim.Muon(
                    self.muon_params, lr=lr, weight_decay=muon_weight_decay, adjust_lr_fn="match_rms_adamw"
                )
            )
        if self.adamw_params:
            self.optimizers.append(torch.optim.AdamW(self.adamw_params, lr=lr, weight_decay=weight_decay))

        super().__init__([*self.muon_params, *self.adamw_params], defaults={})
        self.param_groups = [group for opt in self.optimizers for group in opt.param_groups]

    @torch.no_grad()
    def step(self, closure=None):  # noqa: D102 -- torch.optim.Optimizer API
        loss = None
        for i, opt in enumerate(self.optimizers):
            result = opt.step(closure if i == 0 else None)
            loss = result if loss is None else loss
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:  # noqa: D102 -- torch.optim.Optimizer API
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict[str, Any]:  # noqa: D102 -- torch.optim.Optimizer API
        return {"class": type(self).__name__, "optimizers": [opt.state_dict() for opt in self.optimizers]}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:  # noqa: D102 -- torch.optim.Optimizer API
        if "optimizers" not in state_dict:
            raise ValueError(
                f"MuonAdamW.load_state_dict got a single-optimizer state (keys {sorted(state_dict)}), i.e. a "
                "checkpoint trained with stock PPO's Adam. Resume it with the algorithm it was trained with, or "
                "load it with load_cfg={'optimizer': False} to keep only the weights."
            )
        saved = state_dict["optimizers"]
        if len(saved) != len(self.optimizers):
            warnings.warn(
                f"MuonAdamW state holds {len(saved)} optimizers but this instance has {len(self.optimizers)}; "
                "loading the matching prefix only.",
                stacklevel=2,
            )
        for opt, opt_state in zip(self.optimizers, saved):
            opt.load_state_dict(opt_state)
        # ``Optimizer.load_state_dict`` replaces each optimizer's group dicts, so re-gather them: a stale list would
        # leave the adaptive schedule writing lr into groups no optimizer reads (and PPO.load reading the old lr).
        self.param_groups = [group for opt in self.optimizers for group in opt.param_groups]


class MuonPPO(PPO):
    """rsl_rl PPO whose Adam is replaced by :class:`MuonAdamW` over the actor and critic.

    Parameters
    ----------
    *args
        Forwarded to ``rsl_rl.algorithms.PPO``.
    weight_decay : float
        AdamW weight decay (Weave: 0.01).
    muon_weight_decay : float
        Muon weight decay (Weave: torch's default, 0.1).
    **kwargs
        Forwarded to ``rsl_rl.algorithms.PPO``.
    """

    def __init__(self, *args: Any, weight_decay: float = 0.01, muon_weight_decay: float = 0.1, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.optimizer = MuonAdamW(
            [self.actor, self.critic],
            lr=self.learning_rate,
            weight_decay=weight_decay,
            muon_weight_decay=muon_weight_decay,
        )

    def update(self) -> dict[str, float]:
        """Run the stock PPO update, and log the value function's explained variance (Weave's extra metric).

        Returns
        -------
        dict[str, float]
            rsl_rl's loss dict plus ``explained_variance``.
        """
        with torch.no_grad():
            values = self.storage.values.flatten()
            returns = self.storage.returns.flatten()
            explained_variance = (1.0 - (returns - values).var() / (returns.var() + 1e-8)).item()
        loss_dict = super().update()
        loss_dict["explained_variance"] = explained_variance
        return loss_dict
