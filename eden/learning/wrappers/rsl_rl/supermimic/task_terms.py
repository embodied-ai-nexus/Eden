"""What the world model reads off the task's reward and termination terms.

The world model grades a predicted state with its own implementation of the task's reward and termination. The
functions here check that each term the runner options name is the term that implementation computes, on the links
the forward kinematics returns, and read its weight, kernel width or threshold. A task whose terms differ would train
the policy on another objective than the one the simulator logs, so every mismatch raises at runner construction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from eden.learning.wrappers.rsl_rl.supermimic.representation import RewardSpec

if TYPE_CHECKING:
    from collections.abc import Sequence

    from eden.envs.base import RLEnvBase
    from eden.learning.wrappers.rsl_rl.supermimic.options import RewardTermsOptions

#: Role in :class:`RewardTermsOptions` -> the registered reward term the world model computes for it.
REWARD_TERM_KINDS: dict[str, str] = {
    "anchor_pos": "motion_link_position_tracking_exp",
    "anchor_ori": "motion_link_orientation_tracking_exp",
    "body_pos": "motion_relative_body_position_error_exp",
    "body_ori": "motion_relative_body_orientation_error_exp",
    "body_lin_vel": "motion_global_body_linear_velocity_error_exp",
    "body_ang_vel": "motion_global_body_angular_velocity_error_exp",
    "action_rate": "action_rate_l2",
    "joint_limit": "dofs_pos_limits",
}
_TRACKING_ROLES = ("anchor_pos", "anchor_ori", "body_pos", "body_ori", "body_lin_vel", "body_ang_vel")


def term_params(manager, term_name: str, option: str) -> dict[str, Any]:
    """The registered name (``"name"``), weight and parameters of ``manager``'s term ``term_name``.

    A class term's fields and a function term's ``params`` read alike. ``option`` is the runner option that named
    the term, for the message when the task has no such term.
    """
    if term_name not in manager.active_terms:
        raise ValueError(
            f"runner_options.{option}={term_name!r}, but the task's {type(manager).__name__} has no such term. "
            f"Its terms: {list(manager.active_terms)}. Set runner_options.{option} to one of them."
        )
    values = manager.get_term_options(term_name).dict()
    return values | (values.pop("params", None) or {})


def _require(params: dict[str, Any], key: str, term_name: str, option: str) -> Any:
    if key not in params:
        raise ValueError(
            f"The world model reads {key!r} of term {term_name!r} (runner_options.{option}), a {params['name']!r} "
            f"term, which has no such parameter. Its parameters: {sorted(params)}."
        )
    return params[key]


def _check_kind(params: dict[str, Any], kind: str, term_name: str, option: str) -> None:
    if params["name"] != kind:
        raise ValueError(
            f"The world model computes term {term_name!r} (runner_options.{option}) as a {kind!r} term, but the "
            f"task registers it as {params['name']!r}. Point runner_options.{option} at a {kind!r} term of the "
            "task."
        )


def task_reward_spec(env: RLEnvBase, names: RewardTermsOptions, links_name: Sequence[str]) -> RewardSpec:
    """The reward the world model computes, read off the task's reward terms.

    ``links_name`` are the links the forward kinematics returns, the anchor first. Raises unless the task's reward
    terms are exactly the ones ``names`` lists, each of the kind in :data:`REWARD_TERM_KINDS`, tracking those links
    at the current reference frame.
    """
    manager = env.reward_manager
    links, anchor = list(links_name), links_name[0]
    roles = names.dict()
    weights: dict[str, float] = {}
    stds: dict[str, float] = {}
    for role, kind in REWARD_TERM_KINDS.items():
        term_name, option = roles[role], f"reward_terms.{role}"
        params = term_params(manager, term_name, option)
        _check_kind(params, kind, term_name, option)
        weights[role] = float(params["weight"])
        if role not in _TRACKING_ROLES:
            continue
        stds[role] = float(_require(params, "std", term_name, option))
        expected = {"links_name": [anchor] if role.startswith("anchor") else links}
        if role in ("body_pos", "body_ori"):
            expected["anchor_link_name"] = anchor
        if "frame_offsets" in params:
            expected["frame_offsets"] = (0,)
        got = {key: _as_plain(_require(params, key, term_name, option)) for key in expected}
        differ = {key: (got[key], want) for key, want in expected.items() if got[key] != _as_plain(want)}
        if differ:
            detail = "; ".join(f"{key}={got} (the world model uses {want})" for key, (got, want) in differ.items())
            raise ValueError(
                f"Reward term {term_name!r} (runner_options.{option}) differs from what the world model computes: "
                f"{detail}. The world model tracks the motion term's links {links} with {anchor!r} as the anchor."
            )
    extra = sorted(set(manager.active_terms) - set(roles.values()))
    if extra:
        raise ValueError(
            f"The task has reward terms {extra} that the world model does not compute, so the policy would train "
            f"on another reward than the simulator logs. The world model computes {sorted(roles.values())} "
            "(runner_options.reward_terms). Remove the other terms from the task's reward_options."
        )
    return RewardSpec(
        tracking=tuple((weights[role], stds[role]) for role in _TRACKING_ROLES),
        action_rate=weights["action_rate"],
        joint_limit=weights["joint_limit"],
    )


def termination_threshold(env: RLEnvBase, term_name: str, option: str, kind: str, anchor: str) -> float:
    """The ``threshold`` of the task's termination term ``term_name``, a ``kind`` term on the ``anchor`` link."""
    params = term_params(env.termination_manager, term_name, option)
    _check_kind(params, kind, term_name, option)
    link = _require(params, "anchor_link_name", term_name, option)
    if link != anchor:
        raise ValueError(
            f"Termination term {term_name!r} (runner_options.{option}) watches link {link!r}, but the world model "
            f"terminates on the motion term's first link {anchor!r}. Set its anchor_link_name to {anchor!r}."
        )
    if params.get("threshold_adaptive"):
        raise ValueError(
            f"Termination term {term_name!r} (runner_options.{option}) sets threshold_adaptive=True, which widens "
            "the threshold for low reference poses. The world model terminates on the fixed threshold, so the "
            "policy would train on another termination than the simulator applies. Set threshold_adaptive=False."
        )
    return float(_require(params, "threshold", term_name, option))


def _as_plain(value: Any) -> Any:
    return list(value) if isinstance(value, (list, tuple)) else value
