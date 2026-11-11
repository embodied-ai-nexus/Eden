"""Per-embodiment strategy classes for :class:`GalaxeaVlaPolicy`.

GalaxeaVLA (G0.5) is a mixture policy: a single checkpoint can serve multiple
Galaxea embodiments, routed server-side by the ``embodiment_type`` field in the
observation payload.  ``galaxea_r1lite`` and ``galaxea_r1pro`` share the wire
schema (``left_arm`` / ``left_gripper`` / ``right_arm`` / ``right_gripper`` state
and action parts, three cameras) but not its widths: R1 Lite has 6-DOF arms
(``configs/data/r1lite.yaml``), R1 Pro 7-DOF arms (``configs/data/r1pro.yaml``).
Each setup declares its ``part_layout``; ``action_dim``, the proprio slicing and
the action decoding all follow from it.

Each setup owns:

* ``observation_options`` / ``action_options`` — classmethod factories that wire
  the Eden env (cameras + joint state; arm PD controllers + ``GripperDecisionController``
  gripper terms that make the open/close decision).
* ``pack_observations`` — Eden obs tensors → the server's ``raw_obs`` dict.
* ``unpack_action_parts`` — one step's action parts (decoded from the scheduler's
  row) → Eden controller dict.

.. warning::
   The DOF-name defaults below are **placeholders**.  Set ``*_dofs_name`` /
   ``*_gripper_dof_name`` (and entity names) to match your Eden robot's
   URDF/MJCF when wiring the env.  See the manual follow-ups in the policy
   module docstring.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import torch

from eden.extensions.policies.adapters import (
    DROID_ARM_DOFS,
    JointPositionAdapter,
    check_droid_arm_term,
    check_env_gripper_term,
    check_gripper_state_matches_action,
    check_joint_pass_through,
    check_plain_observation_group,
    dofs_pos_joint_names,
    make_droid_joint_position_actions,
    make_droid_joint_state_group,
    make_image_groups,
)
from eden.extensions.policies.client import PolicySetupBase
from eden.utils.image import to_chw_uint8
from eden.utils.torch import broadcast_to_envs, to_numpy_unbatched

if TYPE_CHECKING:
    from eden.extensions.policies.galaxeavla.policy import GalaxeaVlaPolicy
    from eden.options.managers.actions import ActionTermOptions
    from eden.options.managers.observations import ObservationGroupOptions


# ---------------------------------------------------------------------------
# Placeholder DOF names — override per your Eden robot definition.
# ---------------------------------------------------------------------------

_R1_LEFT_GRIPPER_DOF = "left_gripper_joint"
_R1_RIGHT_GRIPPER_DOF = "right_gripper_joint"

# ---------------------------------------------------------------------------
# R1 gripper wire units
# ---------------------------------------------------------------------------
# GalaxeaVLA's R1 data and R1-Lite client carry the gripper as the Galaxea HDAS gripper stroke: the raw
# ``position`` of /hdas/feedback_gripper_{left,right} (state) and /motion_target/target_position_gripper_*
# (action), which OpenGalaxea/GalaxeaLeRobotToolkit episode_parser/util.py copies unscaled into
# ``observation.state.*_gripper`` / ``action.*_gripper``. Open-Galaxea/open-galaxea G0/docs/dataset.md:
# "gripper state, 0-close and 100-open"; GalaxeaVLA experiments/r1lite ros2_bridge.reset publishes 0 to
# "close grippers" and scheduler._binarize_gripper_inplace maps a high value to 100. R1 Pro uses the same
# topics (experiments/r1pro/core/communication/robot_topics.py).

#: HDAS stroke of a fully closed R1 gripper.
R1_GRIPPER_CLOSED_STROKE = 0.0
#: HDAS stroke of a fully open R1 gripper.
R1_GRIPPER_OPEN_STROKE = 100.0

#: Default DROID gripper thresholds, shared by the policy fields and ``action_options``: binary closes
#: above ``gripper_close_threshold``; continuous-snap snaps closedness above ``gripper_snap_threshold``
#: (droid-franka-client: 0.8).
DROID_GRIPPER_THRESHOLD_DEFAULTS = {"gripper_close_threshold": 0.5, "gripper_snap_threshold": 0.8}
#: The threshold each DROID ``gripper_decode_mode`` does not read, and the mode that reads it.
_DROID_UNREAD_THRESHOLD = {
    "binary": ("gripper_snap_threshold", "continuous-snap"),
    "continuous-snap": ("gripper_close_threshold", "binary"),
}


def _dof_position(name: str, value) -> float:
    """``value`` as a Python float: ``GripperDecisionController`` takes a float or a dict, never an int."""
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise TypeError(
            f"{name} must be a gripper DOF position (a number), got {value!r} ({type(value).__name__}). "
            f"Fix: pass {name}=<the DOF position, e.g. 0.04>."
        )
    return float(value)


def _droid_gripper_decision(mode: str, close_threshold: float, snap_threshold: float) -> dict:
    """The ``GripperDecisionController`` decision fields of DROID ``gripper_decode_mode`` ``mode``.

    Only the threshold the mode reads is set; the other stays at the term's default, which
    :func:`check_env_gripper_term` requires of every field it is not given.
    """
    continuous = mode == "continuous-snap"
    decision = dict(input="closedness", continuous=continuous, on_missing="measured" if continuous else "hold")
    if continuous:
        decision["snap_threshold"] = snap_threshold
    else:
        decision["threshold"] = close_threshold
    return decision


def _refuse_unread_droid_threshold(mode: str, values: dict[str, float], where: str) -> None:
    """Raise when ``values`` changes the threshold DROID ``gripper_decode_mode`` ``mode`` does not read."""
    field, reader = _DROID_UNREAD_THRESHOLD[mode]
    if values[field] != DROID_GRIPPER_THRESHOLD_DEFAULTS[field]:
        raise ValueError(
            f"{where}: {field}={values[field]!r} has no effect under gripper_decode_mode={mode!r}; only "
            f"gripper_decode_mode={reader!r} reads it.\n"
            f"Fix: drop {field}= from {where}, or pass gripper_decode_mode={reader!r} if you meant that rule."
        )


def _r1_placeholder_arm_dofs(side: str, width: int) -> list[str]:
    return [f"{side}_arm_joint{i}" for i in range(1, width + 1)]


# Images go out at native resolution as CHW uint8: the server validates ``ndim == 3`` and resizes to the
# checkpoint's training resolution itself.

#: Fixed zero placeholder the DROID checkpoint expects in the third image slot.
_DUMMY_WRIST_RIGHT_SHAPE = (3, 224, 224)


# ---------------------------------------------------------------------------
# Reply helpers
# ---------------------------------------------------------------------------


def _squeeze_action_vector(value, expected_dim: int, embodiment_type: str, name: str) -> np.ndarray:
    """Coerce a server action part to a flat ``(expected_dim,)`` float32 vector.

    A leading batch / chunk dim is cut to its first row, as ``_squeeze_action_vector`` in the reference client
    (``droid-franka-client/eval/main.py``) does. Unlike the reference, a flat vector of another length is refused even
    when it is a multiple of ``expected_dim``: ``serve_policy.py`` serves one step of its chunk per reply, so such a
    vector is another width than this embodiment's part (a misrouted checkpoint), never a flattened chunk.
    """
    shape = np.shape(value)
    arr = np.squeeze(np.asarray(value, dtype=np.float32))
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if arr.ndim > 1:
        # The first row of a leading batch / chunk dim; an empty one has none and is refused below.
        arr = arr.reshape(-1, arr.shape[-1])[0] if arr.size else arr.reshape(-1)
    if arr.shape[0] != expected_dim:
        raise ValueError(
            f"GalaxeaVLA {embodiment_type!r} reply part {name!r} has shape {shape}, but one step of the part is "
            f"{expected_dim} wide (a leading batch / chunk dim is accepted). Fix: check that the server routes "
            f"embodiment_type={embodiment_type!r} to a checkpoint trained on it. Do NOT trim or pad the part to fit: "
            "the robot would execute another part's values (or another embodiment's) without an error."
        )
    return arr


def _get_action_part(action: dict, keys: tuple[str, ...]):
    """Return the first present alias, or ``None`` when the head dropped the key."""
    for key in keys:
        if key in action:
            return action[key]
    return None


# ---------------------------------------------------------------------------
# Env checks
# ---------------------------------------------------------------------------


def _subject(setup: str | None) -> str:
    """The client and setup every construction refusal names."""
    return f"GalaxeaVlaPolicy.policy_setup={setup!r}"


def _arm_factory(setup: str | None) -> str:
    """The ``action_options`` call that builds ``setup``'s arm terms, for a refusal's fix."""
    return f"GalaxeaVlaPolicy.action_options(policy_setup={setup!r}, ...)"


def _arm_term(env, name: str, part: str, setup: str | None):
    """The env's arm action term ``name``, refused unless it is exactly an ``ImplicitPDController``.

    GalaxeaVLA returns absolute joint positions (rad). Exact type, not ``isinstance``: a subclass (an EMA-filtered
    target, ...) reinterprets the same row. The caller then checks the term executes the targets as they are
    (:func:`~eden.extensions.policies.adapters.check_joint_pass_through` or ``check_droid_arm_term``).
    """
    from eden.managers.terms.actions.joint_actions import ImplicitPDController

    terms = env.action_manager.terms
    term = terms.get(name)
    if type(term) is not ImplicitPDController:
        found = "missing" if term is None else f"of type {type(term).__name__}"
        raise ValueError(
            f"{_subject(setup)} sends absolute {part} joint targets (rad) to action term {name!r}, which must be "
            f"exactly an ImplicitPDController (a subclass reinterprets the same row), but the env's {name!r} is "
            f"{found} (action terms: {sorted(terms)}). Fix: build the env's action_options from {_arm_factory(setup)}."
        )
    return term


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class SetupBase(PolicySetupBase):
    """Abstract strategy for one Galaxea embodiment.

    The server answers each request with a dict of named action *parts* and may omit a
    part its autoregressive head was not confident about. The shared scheduler moves
    numeric rows, so :meth:`decode_response` lays the parts out in ``part_layout`` order
    and marks an omitted part with ``NaN``; :meth:`unpack_actions` turns the row back into
    the parts dict (dropping NaN parts) before decoding.

    Only the ``optional_parts`` may be omitted; any other missing part raises, and so does
    a part the server *sent* with a non-finite value (that is a model failure, never an
    omission), so a ``NaN`` in a row always means "absent from the reply". A reply that
    carries none of the parts raises too: it is no prediction at all.
    """

    #: ``embodiment_type`` string used for mixture routing on the server.
    embodiment_type: ClassVar[str] = "galaxea_r1lite"
    #: ``(part_name, width)`` in row order; ``action_dim`` is their total width.
    part_layout: ClassVar[tuple[tuple[str, int], ...]] = ()
    #: Parts the server may leave out of a reply; every other part is required.
    optional_parts: ClassVar[frozenset[str]] = frozenset()
    #: Alias keys the server may use instead of a canonical part name.
    part_aliases: ClassVar[dict[str, tuple[str, ...]]] = {}
    #: ``GalaxeaVlaPolicy`` fields this setup does not read; setting one to a non-default value raises.
    unused_policy_fields: ClassVar[tuple[str, ...]] = ()
    #: The env action term each part's reply drives.
    action_terms: ClassVar[dict[str, str]] = {}
    #: The ``proprio`` term holding a part's state, where it is not named after the part.
    proprio_terms: ClassVar[dict[str, str]] = {}
    #: ``GripperState.normalize`` of the gripper state ``observation_options`` reads.
    gripper_state_normalize: ClassVar[str] = "openness"
    #: Wire type of the request's ``frequency`` (the reference client's).
    frequency_type: ClassVar[type] = float

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        cls.action_dim = sum(width for _, width in cls.part_layout) or None

    @classmethod
    def part_width(cls, name: str) -> int:
        return dict(cls.part_layout)[name]

    @classmethod
    def _check_dofs_name(
        cls, field: str, dofs_name: list[str] | tuple[str, ...], part: str, hint: str = ""
    ) -> list[str]:
        """``dofs_name`` as a list, or raise when it does not name exactly the joints of the server's ``part``."""
        dofs_name, width = list(dofs_name), cls.part_width(part)
        if len(dofs_name) != width:
            raise ValueError(
                f"{cls.__name__}: {field} has {len(dofs_name)} names, but the {part!r} part of "
                f"{cls.embodiment_type!r} is {width} joints. Given: {dofs_name}.{hint}\n"
                f"Fix: pass {field}=[<the {width} arm joint names, in the checkpoint's joint order>]; a gripper "
                "DOF goes in its own *gripper_dof_name argument, not in the arm list.\n"
                "Do NOT pad or trim the list to make it fit: the server would read one joint's position as "
                "another part (e.g. an arm joint as the gripper) without any error."
            )
        return dofs_name

    @abstractmethod
    def pack_observations(self, obs: dict) -> dict:
        """Convert Eden obs tensors to the WebSocket server ``raw_obs`` dict."""

    @abstractmethod
    def unpack_action_parts(self, action: dict) -> dict[str, torch.Tensor]:
        """Convert one step's action parts (canonical names, omitted optional parts absent) to a controller dict."""

    def decode_response(self, response) -> np.ndarray:
        """``{"action": {part: array}, "need_obs": bool}`` -> ``(1, D)`` NaN-padded row."""
        if not isinstance(response, dict) or "action" not in response:
            raise KeyError(
                "GalaxeaVLA reply must be a dict with an 'action' key; "
                f"got {sorted(response) if isinstance(response, dict) else type(response).__name__}"
            )
        return self.parts_to_row(response["action"])[np.newaxis]

    def parts_to_row(self, action: dict) -> np.ndarray:
        """Lay the parts dict out as one ``(D,)`` row; omitted optional parts become ``NaN``."""
        accepted = {name: (name, *self.part_aliases.get(name, ())) for name, _ in self.part_layout}
        if all(_get_action_part(action, keys) is None for keys in accepted.values()):
            raise KeyError(
                f"GalaxeaVLA {self.embodiment_type!r} reply carries none of its action parts (accepted keys: "
                f"{[key for keys in accepted.values() for key in keys]}); its action holds "
                f"{sorted(str(k) for k in action)}. A reply without a single part is no prediction: every part "
                "would keep its previous target and the robot would freeze without an error. Fix: check that the "
                f"server routes embodiment_type={self.embodiment_type!r} to a checkpoint trained on it, and its log "
                "for 'Dropped absent action keys'. Do NOT treat it as a step on which every part was omitted."
            )
        cols = []
        for name, width in self.part_layout:
            keys = accepted[name]
            value = _get_action_part(action, keys)
            if value is None:
                if name not in self.optional_parts:
                    raise KeyError(
                        f"GalaxeaVLA {self.embodiment_type!r} reply has no {name!r} action (accepted keys: "
                        f"{list(keys)}); its action holds {sorted(str(k) for k in action)}. Only "
                        f"{sorted(self.optional_parts)} may be omitted. Fix: check that the server routes "
                        f"embodiment_type={self.embodiment_type!r} to a checkpoint trained on it. Do NOT hold the "
                        "previous target instead: the arm would freeze silently for the rest of the episode."
                    )
                cols.append(np.full(width, np.nan, dtype=np.float32))
                continue
            vector = _squeeze_action_vector(value, width, self.embodiment_type, name)
            if not np.isfinite(vector).all():
                raise ValueError(
                    f"GalaxeaVLA {self.embodiment_type!r} reply part {name!r} is not finite: {vector.tolist()}.\n"
                    "The server sent the key, so this is a model / de-normalization failure, not an omitted "
                    "part (an omitted part is an absent key).\n"
                    "Fix: check the served checkpoint and its dataset statistics for this embodiment.\n"
                    "Do NOT route it through the omitted-part path (hold / measured gripper): that would hide "
                    "the blow-up and keep the episode running on a stale command."
                )
            cols.append(vector)
        return np.concatenate(cols).astype(np.float32)

    @classmethod
    def proprio_term(cls, part: str) -> str:
        """The env ``proprio`` term holding ``part``'s state."""
        return cls.proprio_terms.get(part, part)

    @classmethod
    def _arm_kwargs(cls, part: str) -> str:
        """The factory arguments that name ``part``'s joints, for error messages."""
        return f"{part}_entity_name= / {part}_dofs_name="

    def _check_env_proprio(self, env) -> None:
        """Fail at construction unless the env's ``proprio`` group is what :meth:`pack_observations` cuts by column.

        The group must be the plain current reading, its terms the ``part_layout`` parts in that order and width (a
        reordered group of the same width would be cut silently wrong), each arm the positions of exactly the joints
        its action term drives, in that order, and each gripper a ``GripperState`` of
        :attr:`gripper_state_normalize` over its action term's open / closed positions. Runs after the action-term
        checks, so every action term is there.
        """
        from eden.managers.terms.observations.manipulation import GripperState
        from eden.managers.terms.observations.proprio import DofsPos

        manager = getattr(env, "observation_manager", None)
        if manager is None or getattr(env, "action_manager", None) is None:
            return
        setup = getattr(self.policy, "policy_setup", None)
        fix = self._proprio_fix()
        check_plain_observation_group(env, "proprio", width=self.action_dim, fix=fix)

        def layout(terms) -> str:
            return ", ".join(f"{name}({', '.join(str(d) for d in dims)})" for name, dims in terms)

        expected = [(self.proprio_term(part), (width,)) for part, width in self.part_layout]
        actual = list(
            zip(manager.group_term_names["proprio"], (tuple(dims) for dims in manager.group_obs_term_dim["proprio"]))
        )
        if actual != expected:
            raise ValueError(
                f"{_subject(setup)} cuts obs['proprio'] by column into [{layout(expected)}], but the env's "
                f"'proprio' group is [{layout(actual)}]: the server would read one part's values as another's.\n"
                f"Fix: {fix}"
            )

        published = {name: term for group, name, term in manager.iter_terms() if group == "proprio"}
        actions = env.action_manager.terms
        problems, arms = [], []
        for part, _ in self.part_layout:
            name, action_name = self.proprio_term(part), self.action_terms[part]
            state, action = published[name], actions[action_name]
            if part.endswith("_gripper"):
                normalize = self.gripper_state_normalize
                if not isinstance(state, GripperState) or state.normalize != normalize:
                    found = (
                        f"a GripperState with normalize={state.normalize!r}"
                        if isinstance(state, GripperState)
                        else f"a {type(state).__name__}"
                    )
                    raise ValueError(
                        f"{_subject(setup)} sends the gripper's measured {normalize}, read from a "
                        f"GripperState(normalize={normalize!r}) term at 'proprio/{name}', but the env has {found} "
                        f"there.\nFix: {fix}"
                    )
                check_gripper_state_matches_action(env, group="proprio", term=name, action_term=action_name)
                continue
            if type(state) is not DofsPos:
                problems.append(f"'proprio/{name}' is a {type(state).__name__}, not the DofsPos joint positions")
            elif state.offset_from_default:
                problems.append(f"'proprio/{name}' has offset_from_default=True (the server reads absolute rad)")
            # The joints the term reads, resolved: its dofs_name may be None (every DOF) or patterns.
            elif (reads := (state.entity_name, dofs_pos_joint_names(env, state))) != (
                drives := (action.entity_name, list(action.dofs_name))
            ):
                problems.append(
                    f"'proprio/{name}' reads {reads[0]!r} {reads[1]} "
                    f"while {action_name!r} drives {drives[0]!r} {drives[1]}"
                )
            else:
                continue
            arms.append(part)
        if problems:
            raise ValueError(
                f"{_subject(setup)} sends each arm's measured joint positions and applies the reply's targets "
                f"to the same joints in the same order, but {'; '.join(problems)}. A swapped or reordered joint list "
                "sends one joint's angle as another's without an error.\n"
                f"Fix: pass the same {', '.join(self._arm_kwargs(part) for part in arms)} to "
                f"GalaxeaVlaPolicy.observation_options(policy_setup={setup!r}, ...) and "
                f"GalaxeaVlaPolicy.action_options(policy_setup={setup!r}, ...)."
            )

    def _proprio_fix(self) -> str:
        """The fix for an ``obs['proprio']`` of the wrong width: this setup's own observation group."""
        setup = getattr(self.policy, "policy_setup", None)
        return f"build the env's observations with GalaxeaVlaPolicy.observation_options(policy_setup={setup!r}, ...)."

    def split_parts(self, row: np.ndarray, what: str, fix: str = "") -> dict[str, np.ndarray]:
        """Cut a flat ``(D,)`` vector into ``{part: slice}`` in ``part_layout`` order; ``fix`` ends the width error."""
        row = np.asarray(row, dtype=np.float32).reshape(-1)
        if row.shape[0] != self.action_dim:
            layout = ", ".join(f"{name}({width})" for name, width in self.part_layout)
            raise ValueError(
                f"{type(self).__name__} ({self.embodiment_type!r}) expects a {self.action_dim}-D {what} "
                f"[{layout}], got {row.shape[0]} values.{f' Fix: {fix}' if fix else ''}"
            )
        parts: dict[str, np.ndarray] = {}
        col = 0
        for name, width in self.part_layout:
            parts[name] = row[col : col + width]
            col += width
        return parts

    def row_to_parts(self, row: np.ndarray) -> dict[str, np.ndarray]:
        """Inverse of :meth:`parts_to_row`: ``NaN`` parts (omitted by the server) are dropped."""
        return {name: value for name, value in self.split_parts(row, "action row").items() if not np.isnan(value).any()}

    def unpack_actions(self, raw_action) -> dict[str, torch.Tensor]:
        """One ``(D,)`` scheduler row -> the parts the server sent that step -> controller dict."""
        return self.unpack_action_parts(self.row_to_parts(raw_action))


# ---------------------------------------------------------------------------
# R1 Lite (canonical bimanual setup)
# ---------------------------------------------------------------------------


class GalaxeaR1LiteSetup(SetupBase):
    """Galaxea R1 Lite: dual 6-DOF arms + two grippers, three cameras (``configs/data/r1lite.yaml``).

    * State and action share one layout, ``[left_arm, left_gripper, right_arm, right_gripper]``
      (``part_layout``); the env's ``proprio`` group must be that concatenation.
    * Arm actions are **absolute joint positions** (the server undoes the
      training-time ``RelativeJointTransform`` before returning them). The server drops any
      part its AR head did not predict (``serve_policy.py``: "each client fills/holds missing
      keys itself") and upstream's R1-Lite client publishes only the parts it received, so an
      omitted arm holds its previous target of the episode, or, before its first, the
      measured joint positions (``obs['proprio']``). The client resolves the hold itself: the
      arm's ``ImplicitPDController`` would drive the joints to a ``NaN`` target.
    * Grippers travel on the wire as the Galaxea HDAS gripper stroke, ``0`` = closed …
      ``100`` = open (:data:`R1_GRIPPER_CLOSED_STROKE` / :data:`R1_GRIPPER_OPEN_STROKE`), in
      both directions. The state is the measured openness of the Eden gripper DOF, scaled to
      that stroke; a returned stroke becomes closedness ``(open - stroke) / (open - closed)``
      for the env's ``GripperDecisionController``, which drives the DOF continuously between
      ``gripper_open_pos`` and ``gripper_close_pos``, as upstream's R1-Lite client publishes
      the stroke as a position target (its ``--binarize-gripper`` is off by default).
    * A gripper the server omitted is sent as ``NaN``: the term keeps its previous target within
      an episode (upstream publishes only the parts it received) and, before its first command
      of an episode, commands the measured position.
    * Every request carries the full observation (upstream's R1-Lite scheduler does not
      use ``need_obs``).
    """

    embodiment_type: ClassVar[str] = "galaxea_r1lite"
    part_layout = (("left_arm", 6), ("left_gripper", 1), ("right_arm", 6), ("right_gripper", 1))
    # serve_policy.py drops every action key the AR head did not predict, arms included.
    optional_parts = frozenset({"left_arm", "left_gripper", "right_arm", "right_gripper"})
    unused_policy_fields = ("gripper_close_threshold", "gripper_decode_mode", "gripper_snap_threshold", "debug_gripper")
    action_terms = {
        "left_arm": "left_arm_controller",
        "left_gripper": "left_gripper_controller",
        "right_arm": "right_arm_controller",
        "right_gripper": "right_gripper_controller",
    }
    # GalaxeaVLA's R1-Lite client sends no frequency; serve_policy.py reads float(raw_obs.get("frequency", 15)).
    frequency_type = float

    def __init__(self, policy: "GalaxeaVlaPolicy") -> None:
        super().__init__(policy)
        #: Per arm part, the target last commanded this episode: what an omitted arm holds.
        self._arm_targets: dict[str, np.ndarray] = {}
        #: Per arm part, this step's measured joint positions: what an omitted arm holds before its first target.
        self._measured_arms: dict[str, np.ndarray] = {}
        self._check_env_arm_terms(policy.env)
        for name in ("left_gripper_controller", "right_gripper_controller"):
            check_env_gripper_term(
                policy.env, name, input="closedness", continuous=True, snap_threshold=None, on_missing="hold"
            )
        self._check_env_proprio(policy.env)

    def _check_env_arm_terms(self, env) -> None:
        """Fail at construction unless each arm term applies this embodiment's absolute arm targets as they are."""
        if getattr(env, "action_manager", None) is None:
            return
        setup = getattr(self.policy, "policy_setup", None)
        for side in ("left", "right"):
            name, part = f"{side}_arm_controller", f"{side}_arm"
            width = self.part_width(part)
            term = _arm_term(env, name, part, setup)
            if term.action_dim != width:
                raise ValueError(
                    f"Action term {name!r} commands {term.action_dim} DOFs, but {_subject(setup)} sends a {width}-joint "
                    f"{part}. Fix: build the env with GalaxeaVlaPolicy.action_options(policy_setup={setup!r}, "
                    f"{side}_arm_dofs_name=[<{width} names>]), or pick the policy_setup whose arm width matches your "
                    f"robot (R1 arm widths: {_r1_arm_widths()})."
                )
            check_joint_pass_through(
                term,
                term_name=name,
                subject=_subject(setup),
                fix=_arm_factory(setup),
                commands=f"absolute {part} joint targets (rad)",
            )

    @classmethod
    def _arm_dofs_name(cls, side: str, dofs_name: list[str] | None) -> list[str]:
        if dofs_name is None:
            return _r1_placeholder_arm_dofs(side, cls.part_width(f"{side}_arm"))
        hint = f" R1 arm widths per policy_setup: {_r1_arm_widths()}."
        return cls._check_dofs_name(f"{side}_arm_dofs_name", dofs_name, f"{side}_arm", hint)

    @classmethod
    def observation_options(
        cls,
        *,
        gripper_open_pos: float,
        gripper_close_pos: float,
        head_camera_name: str = "head_camera",
        left_wrist_camera_name: str = "left_wrist_camera",
        right_wrist_camera_name: str = "right_wrist_camera",
        left_arm_entity_name: str = "robot",
        left_arm_dofs_name: list[str] | None = None,
        left_gripper_entity_name: str = "robot",
        left_gripper_dof_name: str = _R1_LEFT_GRIPPER_DOF,
        right_arm_entity_name: str = "robot",
        right_arm_dofs_name: list[str] | None = None,
        right_gripper_entity_name: str = "robot",
        right_gripper_dof_name: str = _R1_RIGHT_GRIPPER_DOF,
    ) -> dict[str, "ObservationGroupOptions"]:
        """Three camera groups and a ``proprio`` group laid out as ``part_layout``.

        ``*_arm_dofs_name`` must name exactly the setup's arm width (6 for R1 Lite, 7 for
        R1 Pro), in the checkpoint's joint order. Each gripper is reported as its openness
        (``gripper_open_pos`` → 1, ``gripper_close_pos`` → 0, the Eden DOF positions of
        *your* robot's open and closed gripper; required, as no Eden asset fixes them), which
        :meth:`pack_observations` scales to the HDAS stroke the checkpoint was trained on.
        """
        from eden.managers.terms.observations.camera import rgb_image
        from eden.managers.terms.observations.manipulation import GripperState
        from eden.managers.terms.observations.proprio import DofsPos
        from eden.options.managers.observations import ObservationGroupOptions
        from eden.terms import ObsTerm

        left_arm_dofs_name = cls._arm_dofs_name("left", left_arm_dofs_name)
        right_arm_dofs_name = cls._arm_dofs_name("right", right_arm_dofs_name)
        openness = dict(
            normalize="openness",
            open_pos=_dof_position("gripper_open_pos", gripper_open_pos),
            close_pos=_dof_position("gripper_close_pos", gripper_close_pos),
        )

        return {
            "head_image": ObservationGroupOptions(
                rgb=ObsTerm.configure(func=rgb_image, params={"camera_name": head_camera_name}),
            ),
            "left_wrist_image": ObservationGroupOptions(
                rgb=ObsTerm.configure(func=rgb_image, params={"camera_name": left_wrist_camera_name}),
            ),
            "right_wrist_image": ObservationGroupOptions(
                rgb=ObsTerm.configure(func=rgb_image, params={"camera_name": right_wrist_camera_name}),
            ),
            # [left_arm, left_gripper_openness(1), right_arm, right_gripper_openness(1)] in part_layout order.
            "proprio": ObservationGroupOptions(
                left_arm=DofsPos.configure(entity_name=left_arm_entity_name, dofs_name=left_arm_dofs_name),
                left_gripper=GripperState.configure(
                    entity_name=left_gripper_entity_name, dofs_name=[left_gripper_dof_name], **openness
                ),
                right_arm=DofsPos.configure(entity_name=right_arm_entity_name, dofs_name=right_arm_dofs_name),
                right_gripper=GripperState.configure(
                    entity_name=right_gripper_entity_name, dofs_name=[right_gripper_dof_name], **openness
                ),
            ),
        }

    @classmethod
    def action_options(
        cls,
        *,
        gripper_open_pos: float,
        gripper_close_pos: float,
        left_arm_entity_name: str = "robot",
        left_arm_dofs_name: list[str] | None = None,
        left_gripper_entity_name: str = "robot",
        left_gripper_dof_name: str = _R1_LEFT_GRIPPER_DOF,
        right_arm_entity_name: str = "robot",
        right_arm_dofs_name: list[str] | None = None,
        right_gripper_entity_name: str = "robot",
        right_gripper_dof_name: str = _R1_RIGHT_GRIPPER_DOF,
    ) -> dict[str, "ActionTermOptions"]:
        """Absolute joint-position arm controllers + one continuous ``GripperDecisionController`` per gripper.

        ``*_arm_dofs_name`` must name exactly the setup's arm width. The client turns the
        server's HDAS stroke into closedness; each gripper term drives its DOF to
        ``gripper_open_pos + (gripper_close_pos - gripper_open_pos) * closedness`` (the Eden DOF
        positions of *your* robot's open and closed gripper; required, as no Eden asset fixes
        them) and holds its target on an omitted gripper (``NaN``).
        """
        from eden.constants import ReferenceSource
        from eden.managers.terms.actions.binary_actions import GripperDecisionController
        from eden.managers.terms.actions.joint_actions import ImplicitPDController

        left_arm_dofs_name = cls._arm_dofs_name("left", left_arm_dofs_name)
        right_arm_dofs_name = cls._arm_dofs_name("right", right_arm_dofs_name)
        open_pos = _dof_position("gripper_open_pos", gripper_open_pos)
        close_pos = _dof_position("gripper_close_pos", gripper_close_pos)

        def gripper(entity_name: str, dof_name: str):
            return GripperDecisionController.configure(
                entity_name=entity_name,
                dofs_name=[dof_name],
                open_action=open_pos,
                close_action=close_pos,
                input="closedness",
                continuous=True,
                on_missing="hold",
            )

        return dict(
            left_arm_controller=ImplicitPDController.configure(
                entity_name=left_arm_entity_name,
                dofs_name=left_arm_dofs_name,
                reference_source=ReferenceSource.ZERO,
            ),
            right_arm_controller=ImplicitPDController.configure(
                entity_name=right_arm_entity_name,
                dofs_name=right_arm_dofs_name,
                reference_source=ReferenceSource.ZERO,
            ),
            left_gripper_controller=gripper(left_gripper_entity_name, left_gripper_dof_name),
            right_gripper_controller=gripper(right_gripper_entity_name, right_gripper_dof_name),
        )

    # -- runtime obs/action conversion ------------------------------------

    def reset(self) -> None:
        super().reset()
        # A new episode: an omitted arm holds where it is, never the previous episode's target.
        self._arm_targets.clear()
        self._measured_arms.clear()

    def _state_parts(self, obs: dict) -> dict[str, np.ndarray]:
        """``obs['proprio']`` cut into its ``part_layout`` parts."""
        return self.split_parts(
            to_numpy_unbatched(obs["proprio"]).astype(np.float32), "obs['proprio']", fix=self._proprio_fix()
        )

    def observe(self, obs: dict) -> None:
        state = self._state_parts(obs)
        self._measured_arms = {f"{side}_arm": state[f"{side}_arm"] for side in ("left", "right")}

    def pack_observations(self, obs: dict) -> dict:
        p = self.policy
        state = self._state_parts(obs)
        for side in ("left", "right"):
            # A PD-driven finger can overshoot its nominal range slightly: clamp before scaling to the stroke.
            openness = np.clip(state[f"{side}_gripper"], 0.0, 1.0)
            state[f"{side}_gripper"] = (
                R1_GRIPPER_CLOSED_STROKE + (R1_GRIPPER_OPEN_STROKE - R1_GRIPPER_CLOSED_STROKE) * openness
            ).astype(np.float32)
        images = {
            "head_rgb": to_chw_uint8(obs["head_image"]),
            "left_wrist_rgb": to_chw_uint8(obs["left_wrist_image"]),
            "right_wrist_rgb": to_chw_uint8(obs["right_wrist_image"]),
        }
        return {
            "images": images,
            "state": state,
            "task": p.task_description,
            "frequency": self.frequency_type(p.control_frequency),
            "embodiment_type": self.embodiment_type,
        }

    def unpack_action_parts(self, action: dict) -> dict[str, torch.Tensor]:
        p = self.policy
        device, dtype = p.env.device, torch.float32
        n = p.env.num_envs

        out: dict[str, torch.Tensor] = {}
        # Arm parts are absolute joint-position targets. An omitted arm holds its previous target of the episode,
        # else the joints' measured positions (observe()); never NaN, which its ImplicitPDController would apply.
        for side in ("left", "right"):
            part = f"{side}_arm"
            if part in action:
                target = np.array(action[part], dtype=np.float32)  # a copy: the held target outlives the row
            elif part in self._arm_targets:
                target = self._arm_targets[part]
            else:
                target = self._measured_arms[part]
            self._arm_targets[part] = target
            arm = torch.as_tensor(target, device=device, dtype=dtype)
            out[f"{side}_arm_controller"] = broadcast_to_envs(arm, n)

        # Gripper: HDAS stroke -> closedness for the env's GripperDecisionController. An omitted gripper is
        # NaN, the term's own "no command" (hold the previous target; the measured position before the
        # episode's first command), so a reset never replays the previous episode's target.
        for side in ("left", "right"):
            stroke = float(action[f"{side}_gripper"][0]) if f"{side}_gripper" in action else float("nan")
            closedness = (R1_GRIPPER_OPEN_STROKE - stroke) / (R1_GRIPPER_OPEN_STROKE - R1_GRIPPER_CLOSED_STROKE)
            out[f"{side}_gripper_controller"] = torch.full((n, 1), closedness, device=device, dtype=dtype)
        return out


# ---------------------------------------------------------------------------
# R1 Pro (same parts, 7-DOF arms; different mixture tag)
# ---------------------------------------------------------------------------


class GalaxeaR1ProSetup(GalaxeaR1LiteSetup):
    """Galaxea R1 Pro: R1 Lite's parts with **7-DOF** arms (``configs/data/r1pro.yaml``), routed as ``galaxea_r1pro``.

    ``r1pro.yaml`` declares the same four state and action keys (no torso or chassis
    part) and the same three camera keys; only the arm widths differ.
    """

    embodiment_type: ClassVar[str] = "galaxea_r1pro"
    part_layout = (("left_arm", 7), ("left_gripper", 1), ("right_arm", 7), ("right_gripper", 1))


# ---------------------------------------------------------------------------
# DROID Franka (single arm)
# ---------------------------------------------------------------------------


class GalaxeaDroidFrankaSetup(SetupBase):
    """Galaxea G0.5 on the DROID Franka embodiment (``embodiment_type='Droid_Franka'``).

    Wire contract — see ``GalaxeaVLA/experiments/droid/PROTOCOL.md`` and the
    reference client ``droid-franka-client/eval/main.py``:

    * request ``images`` = ``exterior_image`` / ``wrist_image`` (CHW uint8, native
      resolution — the server resizes to the checkpoint's ``shape_meta``) plus a
      fixed zero ``dummy_wrist_right``;
    * request ``state`` = ``right_arm`` (7 joint angles, rad) and ``right_gripper``,
      which is gripper **openness** (``1 - closedness``);
    * the request is the empty dict ``{}`` while the server serves cached chunk steps:
      after a reply with ``need_obs=False`` the next request carries no observation;
      after one with ``need_obs`` true or absent, and after a reset / task change, it
      carries the full observation (``need_obs = bool(response.get("need_obs", True))``,
      re-armed to ``True`` per episode, as the reference does);
    * response ``action`` = ``right_arm`` (alias ``joint_position``) absolute joint
      targets in radians, and ``right_gripper`` (alias ``gripper``) openness.  The
      gripper key is **omitted** when the model was not confident about it; a reply
      without an arm raises, as in the reference. One deviation: a part of another width
      raises even when it is a multiple of the expected one (14 arm values), which the
      reference cuts to its first 7 (see ``_squeeze_action_vector``).

    Decoding reuses :class:`~eden.extensions.policies.adapters.JointPositionAdapter`
    (as ``pi``'s ``droid_jointpos`` and ``cosmos3`` do). The gripper is sent on as
    closedness (``1 - openness``, ``NaN`` when omitted) to the env's
    ``GripperDecisionController`` (see :meth:`action_options`). In the default
    ``continuous-snap`` mode that reproduces droid-franka-client ``eval/main.py``: the
    closedness drives the fingers continuously, values above 0.8 snap closed, and an
    omitted key commands the measured gripper position. ``binary`` mode thresholds at
    ``gripper_close_threshold`` and holds the previous target on an omitted key.
    """

    embodiment_type: ClassVar[str] = "Droid_Franka"
    part_layout = (("right_arm", 7), ("right_gripper", 1))
    optional_parts = frozenset({"right_gripper"})
    part_aliases = {"right_arm": ("joint_position",), "right_gripper": ("gripper",)}
    action_terms = {"right_arm": "ee_controller", "right_gripper": "gripper_controller"}
    # make_droid_joint_state_group's term names.
    proprio_terms = {"right_arm": "joint_pos", "right_gripper": "gripper"}
    gripper_state_normalize = "closedness"
    # droid-franka-client eval/main.py sends DROID_CONTROL_FREQUENCY = 15; PROTOCOL.md: "frequency": <int>.
    frequency_type = int

    @classmethod
    def _arm_kwargs(cls, part: str) -> str:
        return "arm_entity_name= / arm_dofs_name="

    def __init__(self, policy: "GalaxeaVlaPolicy") -> None:
        super().__init__(policy)
        self.gripper_decode_mode = str(policy.gripper_decode_mode)
        if self.gripper_decode_mode not in _DROID_UNREAD_THRESHOLD:
            raise ValueError(
                f"{_subject(policy.policy_setup)}: gripper_decode_mode={self.gripper_decode_mode!r} is not one of "
                f"{sorted(_DROID_UNREAD_THRESHOLD)}. Fix: configure(gripper_decode_mode='continuous-snap'), the "
                "droid-franka-client rule, or 'binary', and build the env with the same gripper_decode_mode in "
                "GalaxeaVlaPolicy.action_options(policy_setup='droid_franka', ...)."
            )
        thresholds = {name: getattr(policy, name) for name in DROID_GRIPPER_THRESHOLD_DEFAULTS}
        _refuse_unread_droid_threshold(self.gripper_decode_mode, thresholds, where="GalaxeaVlaPolicy.configure()")
        self.gripper_snap_threshold = float(policy.gripper_snap_threshold)
        if not 0.0 <= self.gripper_snap_threshold <= 1.0:
            raise ValueError(
                f"{_subject(policy.policy_setup)}: gripper_snap_threshold={self.gripper_snap_threshold!r} is not a "
                "closedness in [0, 1]. Fix: configure(gripper_snap_threshold=0.8), droid-franka-client's value, and "
                "pass the same value to GalaxeaVlaPolicy.action_options(policy_setup='droid_franka', ...)."
            )
        if getattr(policy.env, "action_manager", None) is not None:
            setup = getattr(policy, "policy_setup", None)
            # The checkpoint is trained on DROID data: one column per DROID joint, in DROID's order.
            check_droid_arm_term(
                _arm_term(policy.env, "ee_controller", "right_arm", setup),
                term_name="ee_controller",
                subject=_subject(setup),
                fix=f"{_arm_factory(setup)}, passing the rig's arm_entity_name= / arm_dofs_name=",
                commands="absolute right_arm joint targets (rad)",
            )
        # Arm targets pass through; the server's gripper openness is converted to closedness
        # (``1 - openness``, or NaN when the AR head omitted the key) and forwarded to the env's
        # GripperDecisionController, which holds the previous target on NaN, seeds from the
        # measured fingers on a first NaN, closes when closedness exceeds
        # ``gripper_close_threshold`` (binary) or snaps above ``gripper_snap_threshold`` (continuous-snap).
        self.adapter = JointPositionAdapter(
            n_arm=self.part_width("right_arm"),
            gripper_threshold=policy.gripper_close_threshold,
            debug_gripper=policy.debug_gripper,
        )
        check_env_gripper_term(
            policy.env,
            "gripper_controller",
            **_droid_gripper_decision(
                self.gripper_decode_mode, float(policy.gripper_close_threshold), self.gripper_snap_threshold
            ),
        )
        self._check_env_proprio(policy.env)
        #: Whether the next request carries the observation (the reference client's ``need_obs``).
        self.need_obs = True

    def reset(self) -> None:
        super().reset()
        # The server's chunk cache was just cleared (``__reset__``), so it needs an observation again.
        self.need_obs = True

    # -- env wiring -------------------------------------------------------

    @classmethod
    def observation_options(
        cls,
        camera_name: str = "camera",
        wrist_camera_name: str = "wrist_camera",
        arm_entity_name: str = "robot",
        arm_dofs_name: list[str] | None = None,
        gripper_entity_name: str | None = None,
        gripper_dof_name: str = "left_driver_joint",
        gripper_close_pos: float = 0.8,
    ) -> dict[str, "ObservationGroupOptions"]:
        """``image`` + ``wrist_image`` camera groups and an 8-D ``proprio`` group.

        ``proprio`` is ``[joint_pos(7), gripper_closedness(1)]`` — the same DROID
        joint-state contract OpenPI's ``droid_jointpos`` setup uses (shared group builder).
        """
        arm_dofs_name = cls._droid_arm_dofs_name(arm_dofs_name)
        groups = make_image_groups(camera_name, use_wrist_image=True, wrist_camera_name=wrist_camera_name)
        groups["proprio"] = make_droid_joint_state_group(
            arm_entity_name,
            arm_dofs_name,
            gripper_entity_name=gripper_entity_name,
            gripper_dof_name=gripper_dof_name,
            gripper_close_pos=_dof_position("gripper_close_pos", gripper_close_pos),
        )
        return groups

    @classmethod
    def action_options(
        cls,
        arm_entity_name: str = "robot",
        arm_dofs_name: list[str] | None = None,
        gripper_entity_name: str | None = None,
        gripper_dof_name: str = "left_driver_joint",
        gripper_close_pos: float = 0.8,
        gripper_close_threshold: float = DROID_GRIPPER_THRESHOLD_DEFAULTS["gripper_close_threshold"],
        gripper_decode_mode: str = "continuous-snap",
        gripper_snap_threshold: float = DROID_GRIPPER_THRESHOLD_DEFAULTS["gripper_snap_threshold"],
    ) -> dict[str, "ActionTermOptions"]:
        """Absolute joint-position arm controller + ``GripperDecisionController`` fed gripper closedness.

        The server returns absolute arm joint targets, so the arm controller is an
        ``ImplicitPDController`` with a zero reference. The client converts the server's openness
        to closedness (``1 - openness``). ``continuous-snap`` (default, as the droid-franka-client
        reference) drives the fingers continuously, snapping closedness above
        ``gripper_snap_threshold`` to fully closed, and commands the measured closedness when the
        server omits the key (``NaN``). ``binary`` closes when closedness exceeds
        ``gripper_close_threshold`` and keeps the previous target on an omitted key. Each mode reads
        only its own threshold; changing the other raises. Keep these equal to the policy's
        attributes of the same name (checked at policy construction).
        """
        if gripper_decode_mode not in _DROID_UNREAD_THRESHOLD:
            raise ValueError(
                f"gripper_decode_mode must be one of {sorted(_DROID_UNREAD_THRESHOLD)}; got {gripper_decode_mode!r}."
            )
        thresholds = dict(
            gripper_close_threshold=gripper_close_threshold, gripper_snap_threshold=gripper_snap_threshold
        )
        _refuse_unread_droid_threshold(
            gripper_decode_mode, thresholds, where="GalaxeaVlaPolicy.action_options(policy_setup='droid_franka', ...)"
        )
        return make_droid_joint_position_actions(
            arm_entity_name,
            cls._droid_arm_dofs_name(arm_dofs_name),
            gripper_entity_name=gripper_entity_name,
            gripper_dof_name=gripper_dof_name,
            gripper_close_pos=_dof_position("gripper_close_pos", gripper_close_pos),
            **_droid_gripper_decision(gripper_decode_mode, gripper_close_threshold, gripper_snap_threshold),
        )

    @classmethod
    def _droid_arm_dofs_name(cls, arm_dofs_name: list[str] | None) -> list[str]:
        if arm_dofs_name is None:
            return list(DROID_ARM_DOFS)
        return cls._check_dofs_name("arm_dofs_name", arm_dofs_name, "right_arm")

    # -- runtime obs/action conversion ------------------------------------

    def pack_observations(self, obs: dict) -> dict:
        if not self.need_obs:
            # The previous reply came from a chunk the server still has steps of: the reference sends {}.
            return {}
        p = self.policy
        # [joints(7), closedness(1)], cut by part_layout: any other width raises rather than slicing silently.
        proprio = self.split_parts(
            to_numpy_unbatched(obs["proprio"]).astype(np.float32), "obs['proprio']", fix=self._proprio_fix()
        )
        # The PD-controlled driver can overshoot its nominal [0, close_pos]
        # range slightly. Clamp normalized closedness before converting to the
        # Galaxea/DROID training convention (openness = 1 - closedness).
        closedness = np.clip(proprio["right_gripper"], 0.0, 1.0).astype(np.float32)

        return {
            "images": {
                "exterior_image": to_chw_uint8(obs["image"]),
                "wrist_image": to_chw_uint8(obs["wrist_image"]),
                "dummy_wrist_right": np.zeros(_DUMMY_WRIST_RIGHT_SHAPE, dtype=np.uint8),
            },
            "state": {
                "right_arm": proprio["right_arm"],
                # The checkpoint is trained on openness, not closedness.
                "right_gripper": (1.0 - closedness).astype(np.float32),
            },
            "task": p.task_description,
            # A whole number: GalaxeaVlaPolicy refuses any other control_frequency for an int wire type.
            "frequency": self.frequency_type(p.control_frequency),
            "embodiment_type": self.embodiment_type,
        }

    def decode_response(self, response) -> np.ndarray:
        if isinstance(response, dict):
            # droid-franka-client eval/main.py: need_obs = bool(response.get("need_obs", True)). Read before the
            # parts are decoded, so a reply that fails to decode cannot leave the previous reply's value behind.
            self.need_obs = bool(response.get("need_obs", True))
        return super().decode_response(response)

    def unpack_action_parts(self, action: dict) -> dict[str, torch.Tensor]:
        # right_arm is always present (parts_to_row raises on a reply without it).
        arm_abs = np.asarray(action["right_arm"], dtype=np.float32)

        if "right_gripper" not in action:
            # The AR head dropped the gripper this chunk: NaN tells the env's
            # GripperDecisionController there is no command; it commands the measured gripper
            # (continuous-snap, as the reference client) or keeps its previous target (binary).
            openness = float("nan")
            gripper_source = "omitted"
        else:
            openness = float(action["right_gripper"][0])
            gripper_source = "server_prediction"

        # The checkpoint predicts openness; gripper_close_threshold is a closedness threshold.
        closedness = 1.0 - openness
        row = np.concatenate([arm_abs, np.asarray([closedness], dtype=np.float32)])
        result = self.adapter(row, env=self.policy.env)
        if self.adapter.last_diagnostics is not None:
            self.adapter.last_diagnostics["gripper_source"] = gripper_source
            self.adapter.last_diagnostics["gripper_openness"] = openness
            self.adapter.last_diagnostics["gripper_decode_mode"] = self.gripper_decode_mode
            self.adapter.last_diagnostics["gripper_snap_threshold"] = self.gripper_snap_threshold
        return result


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SETUP_REGISTRY: dict[str, type[SetupBase]] = {
    "galaxea_r1lite": GalaxeaR1LiteSetup,
    "galaxea_r1pro": GalaxeaR1ProSetup,
    "droid_franka": GalaxeaDroidFrankaSetup,
}


def _r1_arm_widths() -> str:
    """``"galaxea_r1lite=6, galaxea_r1pro=7"``: the arm width of every R1 setup, for error messages."""
    return ", ".join(
        f"{key}={dict(cls.part_layout).get('left_arm')}"
        for key, cls in SETUP_REGISTRY.items()
        if issubclass(cls, GalaxeaR1LiteSetup)
    )
