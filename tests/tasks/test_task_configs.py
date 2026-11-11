"""Cross-task config properties: tasks that splat one ``_Observations`` dict into two groups must not alias terms."""

from __future__ import annotations

import pytest

from eden.options import ObservationTermOptions
from eden.tasks.benchmark.reacher.config import ReacherConfig
from eden.tasks.homie.config import HomieLocomotionConfig
from eden.tasks.manipulation.dex_lift.config import ManipulationDexLiftConfig
from eden.tasks.manipulation.lift_box.config import ManipulationLiftBoxConfig
from eden.tasks.manipulation.lift_box_rgb.config import ManipulationLiftBoxRGBConfig
from eden.tasks.manipulation.reach_droid.config import ManipulationReachDroidConfig
from eden.tasks.mochi.fold_the_towel.config import MochiFoldTheTowelConfig
from eden.tasks.mochi.route_the_cable.config import MochiRouteTheCableConfig
from eden.tasks.mochi.wipe_the_bar.config import MochiWipeTheBarConfig


def _terms(group) -> dict:
    """Term options in an observation group, minus the ``_option_*`` string metadata."""
    return {k: v for k, v in (group.__pydantic_extra__ or {}).items() if not isinstance(v, str)}


@pytest.mark.parametrize(
    "config_cls",
    [
        ReacherConfig,
        ManipulationReachDroidConfig,
        ManipulationDexLiftConfig,
        ManipulationLiftBoxConfig,
        ManipulationLiftBoxRGBConfig,
        MochiWipeTheBarConfig,
        MochiFoldTheTowelConfig,
        MochiRouteTheCableConfig,
        HomieLocomotionConfig,
    ],
    ids=lambda c: c.__name__,
)
def test_editing_a_policy_term_leaves_the_critic_untouched(config_cls):
    """A TaskMod or CLI edit to one group must not change what the other observes."""
    obs = config_cls().observation_options
    for group_name in obs.term_keys():
        for name, term in _terms(getattr(obs, group_name)).items():
            assert isinstance(term, ObservationTermOptions), f"{group_name}.{name} is a {type(term).__name__}"

    policy, critic = _terms(obs.policy), _terms(obs.critic)
    shared = [name for name in policy if name in critic]
    assert shared
    for name in shared:
        before = (critic[name].scale, dict(critic[name].params or {}))
        # A probe no task ships; a list scale (a per-component one) is replaced whole.
        policy[name].scale = 7.5
        if policy[name].params:
            policy[name].params["__probe__"] = 1
        assert (critic[name].scale, dict(critic[name].params or {})) == before, f"critic.{name} changed"
