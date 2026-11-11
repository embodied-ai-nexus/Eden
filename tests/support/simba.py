"""SimBa actor / critic models and a Muon PPO built from Eden's own option classes, over synthetic observation groups.

What a task does to train with them, minus the env: narrow a runner's ``actor`` / ``critic`` / ``algorithm`` fields
to the option classes, translate it for the installed rsl_rl, and let rsl_rl resolve each ``class_name``.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict

from eden.learning.wrappers.rsl_rl.muon_ppo import RslRlMuonPpoAlgorithmOptions
from eden.learning.wrappers.rsl_rl.simba import RslRlSimBaActorOptions, RslRlSimBaCriticOptions
from eden.options.learning.rsl_rl.rl import RslRlOnPolicyRunnerOptions
from eden.options.learning.rsl_rl.translate import translate_runner_dict

#: Width of each synthetic observation group.
GROUP_DIMS = {"reference": 7, "target": 5, "object": 4, "proprio": 6, "privileged": 9}
#: The actor and the critic share three groups and each reads one of its own.
OBS_GROUPS = {
    "actor": ["reference", "target", "object", "proprio"],
    "critic": ["reference", "target", "object", "privileged"],
}
NUM_ACTIONS = 3


class SimBaMuonRunnerOptions(RslRlOnPolicyRunnerOptions):
    """A runner narrowed to the SimBa models and Muon PPO, at their option classes' defaults."""

    actor: RslRlSimBaActorOptions = RslRlSimBaActorOptions()
    critic: RslRlSimBaCriticOptions = RslRlSimBaCriticOptions()
    algorithm: RslRlMuonPpoAlgorithmOptions = RslRlMuonPpoAlgorithmOptions()


def group_obs(n: int = 8, seed: int = 0) -> TensorDict:
    """A batch of ``n`` random observations, one entry per :data:`GROUP_DIMS` group."""
    g = torch.Generator().manual_seed(seed)
    return TensorDict({k: torch.randn(n, d, generator=g) for k, d in GROUP_DIMS.items()}, batch_size=[n])


def runner_dict() -> dict:
    """:class:`SimBaMuonRunnerOptions` over :data:`OBS_GROUPS`, translated for the installed rsl_rl."""
    return translate_runner_dict(SimBaMuonRunnerOptions(obs_groups=OBS_GROUPS).dict())


def simba_models(obs: TensorDict, *, compile: bool = False, **actor_overrides):
    """The SimBa actor and critic over ``obs``, eager unless ``compile``."""
    from tests.support.rsl_rl import resolve_class

    runner = runner_dict()
    actor_cls, actor_kwargs = resolve_class({**runner["actor"], "compile": compile, **actor_overrides})
    critic_cls, critic_kwargs = resolve_class({**runner["critic"], "compile": compile})
    actor = actor_cls(obs, runner["obs_groups"], "actor", NUM_ACTIONS, **actor_kwargs)
    critic = critic_cls(obs, runner["obs_groups"], "critic", 1, **critic_kwargs)
    return actor, critic
