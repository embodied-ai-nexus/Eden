"""The runner's contract with the task: what it checks at construction and what it reads off the task's terms.

The world model steps and grades a predicted state with its own implementation of the task's observations, reward
and termination. The functions here check that the motion term, the two observation groups and each reward and
termination term the runner options name are the ones that implementation computes, and read the weights, kernel
widths and thresholds. A task that differs would train the policy on another objective than the one the simulator
logs, so every mismatch raises at runner construction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from eden.learning.wrappers.rsl_rl.supermimic.motion import SuperMimicMotion
from eden.learning.wrappers.rsl_rl.supermimic.representation import PROPRIO_TERMS, RewardSpec
from eden.learning.wrappers.rsl_rl.supermimic.terms import DofsPosBiased, RootJointState

if TYPE_CHECKING:
    from collections.abc import Sequence

    from eden.entities.rigid import RigidEntity
    from eden.envs.base import RLEnvBase
    from eden.learning.wrappers.rsl_rl.supermimic.options import RewardTermsOptions, SuperMimicRunnerOptions
    from eden.learning.wrappers.rsl_rl.supermimic.representation import GenCoordRepresentation
    from eden.learning.wrappers.rsl_rl.supermimic.sim_context import SimContext
    from eden.managers.action_manager import ActionTerm

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
#: Tolerance (s) on the reference frame interval against the env dt.
_FRAME_INTERVAL_TOLERANCE_S = 1e-9
#: Proprioception term (``representation.PROPRIO_TERMS``) -> the registered observation term the world model's
#: imagined reading computes for it. The joint-position term depends on the encoder bias and is checked apart.
ACTOR_TERM_KINDS: dict[str, str] = {
    "projected_gravity": "projected_gravity",
    "base_ang_vel": "base_ang_vel",
    "joint_vel": "dofs_vel",
    "last_action": "last_action",
}
#: Options of an observation term that change its value after ``compute``, each with the value that leaves it as it
#: is. The world model decodes the state group and imagines the actor group from the raw terms.
_UNMODIFIED_TERM = {"scale": 1.0, "clip": None, "modifiers": []}
_TRACKING_ROLES = ("anchor_pos", "anchor_ori", "body_pos", "body_ori", "body_lin_vel", "body_ang_vel")


def check_motion_term(env: RLEnvBase, term_name: str) -> SuperMimicMotion:
    """The task's motion term ``term_name``, checked to play one library frame per control step on every env."""
    motion = env.motion_manager.get_term(term_name)
    if not isinstance(motion, SuperMimicMotion):
        raise TypeError(
            f"runner_options.motion_term={term_name!r} is a {type(motion).__name__}, but the SuperMimic runner "
            "needs a SuperMimicMotion term (eden.learning.wrappers.rsl_rl.supermimic.motion): it reads the term's "
            "motion library and forward kinematics and sets its reset distribution. Configure the task's motion "
            "term with SuperMimicMotion.configure(...)."
        )
    manager_options = env.motion_options
    frame_interval = tuple(manager_options.frame_interval_s_range)
    problems = []
    if motion.num_assigned_envs != env.num_envs:
        problems.append(
            f"motion term {term_name!r} plays {motion.num_assigned_envs} of the {env.num_envs} envs (motion terms: "
            f"{env.motion_manager.active_terms}); give the task one motion term"
        )
    if manager_options.data_start_from != "current_time":
        problems.append(f"motion_options.data_start_from={manager_options.data_start_from!r}; set it to 'current_time'")
    if motion.freeze_at_timestep_zero_prob > 0.0:
        problems.append(
            f"motion term {term_name!r} has freeze_at_timestep_zero_prob={motion.freeze_at_timestep_zero_prob}, "
            "which holds some envs on their first frame; set it to 0"
        )
    if manager_options.resync_on_motion_end:
        problems.append(
            "motion_options.resync_on_motion_end=True, which restarts a finished clip from the base term's start "
            "sampling instead of the reset distribution; set it to False"
        )
    if any(abs(value - env.dt) > _FRAME_INTERVAL_TOLERANCE_S for value in frame_interval):
        problems.append(
            f"motion_options.frame_interval_s_range={frame_interval}; set both ends to the env dt {env.dt:g}"
        )
    if problems:
        raise ValueError(
            "The world model grades a predicted state against the library frame at each env's clock, one frame per "
            "control step, and the simulator's reward must read the same frame: " + "; ".join(problems) + "."
        )
    return motion


def check_joint_and_anchor_order(
    controller: ActionTerm, robot: RigidEntity, motion: SuperMimicMotion, action_term: str, motion_term: str
) -> None:
    """Raise unless the controller, the motion term and the robot list the joints in one order and the motion
    term's first link is the robot's base link.
    """
    robot_joints = list(robot.dofs_name)
    if list(controller.dofs_name) != robot_joints or list(motion.fk.joint_names) != robot_joints:
        raise ValueError(
            f"SuperMimic reads the robot's joints in one order: the controller {action_term!r} "
            f"({list(controller.dofs_name)}), the motion term's joints ({motion.fk.joint_names}) and the robot "
            f"({robot_joints}) must all list every joint in the robot's dofs_name order."
        )
    links = motion.fk.link_names
    if links[0] != robot.base_link_name:
        raise ValueError(
            f"The world model anchors its reward and termination on the robot's root pose, so motion term "
            f"{motion_term!r} must list the base link {robot.base_link_name!r} first in target_links_name, got "
            f"{links[0]!r} (links: {list(links)}). Put {robot.base_link_name!r} first and use it as the "
            "anchor_link_name of the task's reward and termination terms."
        )


def check_observation_groups(
    env: RLEnvBase,
    options: SuperMimicRunnerOptions,
    entity_name: str,
    representation: GenCoordRepresentation,
    sim_context: SimContext,
) -> None:
    """Raise unless the two observation groups the runner reads hold what it decodes.

    The state group is one :class:`~eden.learning.wrappers.rsl_rl.supermimic.terms.RootJointState` of the
    controlled entity. The actor group is the proprioception ``[projected_gravity, base_ang_vel, q - q0, qd,
    last_action]``, flat and without history, which ``policy.proprio_noise`` and the world model's imagined
    reading follow term by term. Its joint term is a
    :class:`~eden.learning.wrappers.rsl_rl.supermimic.terms.DofsPosBiased` exactly when the simulation context
    carries an encoder bias, which the imagined reading then adds.
    """
    observations = env.observation_manager
    names, dims = observations.group_term_names, observations.group_obs_term_dim
    for option in ("state_observation_group", "actor_observation_group"):
        group = getattr(options, option)
        if group not in names:
            raise ValueError(
                f"runner_options.{option}={group!r}, but the task has no such observation group. Its groups: "
                f"{list(names)}."
            )
    group = options.state_observation_group
    terms = [term for term_group, _, term in observations.iter_terms() if term_group == group]
    if len(terms) != 1 or not isinstance(terms[0], RootJointState) or terms[0].entity_name != entity_name:
        raise ValueError(
            f"The runner decodes observation group {group!r} (runner_options.state_observation_group) as the "
            f"generalized state of {entity_name!r}, so the group must hold one RootJointState term on that entity, "
            f"got terms {names[group]}. Configure it as RootJointState.configure(entity_name={entity_name!r})."
        )
    problems = _modified_term_problems(env, group, [(names[group][0], terms[0])])
    if problems:
        raise ValueError(
            f"The runner decodes observation group {group!r} (runner_options.state_observation_group) from the raw "
            "state: " + "; ".join(problems) + "."
        )
    group = options.actor_observation_group
    widths = [dim[-1] if len(dim) == 1 else dim for dim in dims[group]]
    expected = representation.proprio_term_widths()
    if widths != expected or observations.group_obs_dim[group] != (sum(expected),):
        raise ValueError(
            f"The runner reads observation group {group!r} (runner_options.actor_observation_group) as the "
            f"proprioception {list(PROPRIO_TERMS)} with term widths {expected}, concatenated and without history. "
            f"The group holds terms {names[group]} with shapes {dims[group]} and group shape "
            f"{observations.group_obs_dim[group]}. policy.proprio_noise and the world model's imagined reading "
            "follow that order term by term."
        )
    terms = [(name, term) for term_group, name, term in observations.iter_terms() if term_group == group]
    problems = _modified_term_problems(env, group, terms)
    for role, (name, term) in zip(PROPRIO_TERMS, terms, strict=True):
        params = _plain_params(term.options)
        if role == "joint_pos":
            problems += _joint_term_problems(name, term, params, options, sim_context)
        elif params["name"] != ACTOR_TERM_KINDS[role]:
            problems.append(f"term {name!r} ({role}) is a {params['name']!r} term, not {ACTOR_TERM_KINDS[role]!r}")
        if params.get("entity_name", entity_name) != entity_name:
            problems.append(f"term {name!r} ({role}) reads entity {params['entity_name']!r}, not {entity_name!r}")
        if role == "base_ang_vel" and params.get("frame") != "body":
            problems.append(f"term {name!r} ({role}) has frame={params.get('frame')!r}; set frame='body'")
    if problems:
        raise ValueError(
            f"Observation group {group!r} (runner_options.actor_observation_group) differs from the proprioception "
            f"the world model imagines, {list(PROPRIO_TERMS)}: " + "; ".join(problems) + "."
        )


def _joint_term_problems(
    name: str, term: Any, params: dict[str, Any], options: SuperMimicRunnerOptions, sim_context: SimContext
) -> list[str]:
    """How the actor's joint-position term differs from the reading the world model imagines.

    The term is a :class:`~eden.learning.wrappers.rsl_rl.supermimic.terms.DofsPosBiased` on the runner's controller
    exactly when the simulation context carries an encoder bias, which the imagined reading then adds.
    """
    event = options.sim_context.encoder_bias_event
    biased = isinstance(term, DofsPosBiased)
    plain = params["name"] == "dofs_pos" and params.get("offset_from_default") is True
    if biased and not sim_context.has_encoder_bias:
        problem = (
            f"is a DofsPosBiased, which reads the encoder bias, but runner_options.sim_context.encoder_bias_event "
            f"is {event!r}, so the imagined reading has none; name the task's RandomizeDofsPosOffset event there, "
            "or use a joint-position term without the bias"
        )
    elif not biased and sim_context.has_encoder_bias:
        problem = (
            f"is a {params['name']!r} term, which reads no encoder bias, but runner_options.sim_context."
            f"encoder_bias_event={event!r} adds one to the imagined reading; use DofsPosBiased for the term, or set "
            "encoder_bias_event to None"
        )
    elif biased and term.action_term_name != options.action_term:
        problem = (
            f"reads the offset of action term {term.action_term_name!r}, not the runner's {options.action_term!r} "
            "(runner_options.action_term)"
        )
    elif not biased and not plain:
        problem = (
            f"is a {params['name']!r} term with offset_from_default={params.get('offset_from_default')!r}, but the "
            "imagined reading is q - q0; use DofsPos.configure(offset_from_default=True) for a task without an "
            "encoder bias"
        )
    else:
        return []
    return [f"term {name!r} (joint_pos) {problem}"]


def _modified_term_problems(env: RLEnvBase, group: str, terms: list[tuple[str, Any]]) -> list[str]:
    """How the terms of observation group ``group`` differ from their raw values: a scale, a clip or a modifier on
    a term, or the group's corruption (the runner adds the sensor noise itself, to both readings).
    """
    problems = []
    if getattr(env.observation_options, group).enable_corruption:
        problems.append(
            "the group has enable_corruption=True, which adds the terms' noise in the simulator only; set it to "
            "False (the runner adds runner_options.policy.proprio_noise to the actor's reading)"
        )
    for name, term in terms:
        values = term.options.dict()
        changed = {key: values.get(key) for key, raw in _UNMODIFIED_TERM.items() if values.get(key, raw) != raw}
        if changed:
            problems.append(f"term {name!r} sets {changed}, which the world model does not apply; remove them")
    return problems


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
    return _plain_params(manager.get_term_options(term_name))


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


def _plain_params(term_options: Any) -> dict[str, Any]:
    """A term's options as one dict: a class term's fields and a function term's ``params`` read alike."""
    values = term_options.dict()
    return values | (values.pop("params", None) or {})


def _as_plain(value: Any) -> Any:
    return list(value) if isinstance(value, (list, tuple)) else value
