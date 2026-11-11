"""Per-setup strategy classes for PiPolicy.

Each ``SetupBase`` subclass encapsulates all setup-specific logic:

* ``observation_options`` / ``action_options`` — classmethod factory helpers
  called by ``PiPolicy`` to wire the Eden environment.
* ``pack_observations`` — converts Eden obs tensors to the server payload dict.
* An adapter (from :mod:`eden.extensions.policies.adapters`, or one of this module's
  setup-specific ones) performs action unpacking (rotation conventions, unit conversion,
  client-side velocity integration, ...). A gripper scalar is forwarded raw; the env's
  gripper term applies the setup's rule.
* ``reset`` — clears any internal state held by the adapter.

Adding a new setup means subclassing ``SetupBase`` (or ``_EEDeltaBase``),
implementing ``_pack_state`` / ``pack_observations``, choosing / constructing
an adapter, and registering the class in :data:`SETUP_REGISTRY`.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import torch

from eden.extensions.policies.adapters import (
    DROID_ARM_DOFS,
    DROID_CONTROL_HZ,
    DROID_CONTROL_TIMING,
    DROID_CONTROL_TIMING_NOTE,
    WIDOWX_DOFS,
    ActionAdapter,
    EEDeltaAdapter,
    JointPositionAdapter,
    JointVelocityAdapter,
    bridge_rpy,
    check_control_rate,
    check_droid_arm_term,
    check_droid_joint_state,
    check_env_gripper_term,
    check_gripper_state_matches_action,
    check_joint_pass_through,
    check_libero_camera_resolution,
    check_plain_observation_group,
    libero_task_action_options,
    libero_training_view,
    make_droid_gripper_action,
    make_droid_joint_position_actions,
    make_droid_joint_state_group,
    make_ee_pose_group,
    make_image_groups,
    make_widowx_ee_delta_actions,
    robosuite_axis_angle,
)
from eden.extensions.policies.client import PolicySetupBase
from eden.options.robots.aloha_vx300s import Aloha
from eden.options.robots.libero import LIBERO_ROBOT_ENTITY_NAME, LiberoPanda
from eden.utils.image import prepare_image_padded
from eden.utils.torch import broadcast_to_envs, to_numpy_unbatched

if TYPE_CHECKING:
    from eden.envs.base import EnvBase
    from eden.extensions.policies.pi.policy import PiPolicy
    from eden.managers.action_manager import ActionTerm
    from eden.options.managers.actions import ActionTermOptions
    from eden.options.managers.observations import ObservationGroupOptions


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: DROID's ``RobotEnv(action_space="joint_velocity")`` turns a unit joint velocity into a joint
#: delta of ``RobotIKSolver.max_joint_delta`` rad per control step (``joint_velocity_to_delta``)
#: and steps at ``RobotEnv.control_hz``; the arm is then sent to ``q + delta``:
#: https://github.com/droid-dataset/droid/blob/main/droid/robot_ik/robot_ik_solver.py
#: https://github.com/droid-dataset/droid/blob/main/droid/franka/robot.py (``update_joints``)
DROID_MAX_JOINT_DELTA: float = 0.2
DROID_CONTROL_PERIOD_S: float = 1.0 / DROID_CONTROL_HZ
#: A unit DROID joint velocity in rad/s: ``0.2`` rad per ``1/15`` s step.
DROID_RAD_PER_S_PER_UNIT: float = DROID_MAX_JOINT_DELTA * DROID_CONTROL_HZ
#: openpi ``examples/droid/main.py`` clips every action dim to ``[-1, 1]`` before ``env.step``.
#: That makes DROID's own direction-preserving rescale (``v / max|v|`` when ``max|v| > 1``) a
#: no-op, so the element-wise clip is the whole of the robot-side bound:
#: https://github.com/Physical-Intelligence/openpi/blob/main/examples/droid/main.py
DROID_ACTION_CLIP: float = 1.0

#: BridgeData's and SimplerEnv's WidowX end-effector frame, ``ee_gripper_link``, in ``gripper_link``'s frame
#: (identity rotation). ManiSkill2_real2sim's ``WidowX`` reports ``eef_pos`` from ``ee_gripper_link``
#: (``mani_skill2_real2sim/agents/robots/widowx.py``), which the Interbotix wx250s URDF puts ``ee_arm`` 0.043 +
#: ``ee_bar`` 0.023 + ``ee_gripper`` 0.027575 m along ``gripper_link``'s +x. Eden's ``wx250s.xml`` has no such
#: body (its finger links sit at x = 0.066 = 0.043 + 0.023), so the proprio adds it as a tool offset; without
#: it every state xyz is ~9 cm off the training distribution.
SIMPLER_WIDOWX_EE_OFFSET_POS: tuple[float, float, float] = (0.093575, 0.0, 0.0)
#: The link of Eden's ``wx250s.xml`` that carries ``ee_gripper_link`` at :data:`SIMPLER_WIDOWX_EE_OFFSET_POS`.
SIMPLER_WIDOWX_EE_LINK: str = "gripper_link"
#: SimplerEnv's WidowX finger travel ``(closed, open)`` in metres: ManiSkill2_real2sim's ``wx250s.urdf`` limits the
#: ``left_finger`` / ``right_finger`` joints to ``lower=0.015`` / ``upper=0.037``, and its
#: ``WidowX.get_gripper_closedness`` reports ``(qmax - q) / (qmax - qmin)`` clamped at >= 0, so the Bridge state's
#: openness is ``(q - 0.015) / 0.022``.
#: Eden's ``wx250s.xml`` lets the fingers close further (0.01), which this clips to 0 like SimplerEnv's clamp.
SIMPLER_WIDOWX_FINGER_RANGE: tuple[float, float] = (0.015, 0.037)

#: ALOHA arm joints per side, in Eden's ``Aloha`` DOF naming (``<side>/<joint>``).
_ALOHA_ARM_JOINTS = ("waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate")


def _aloha_arm_dofs(side: str) -> list[str]:
    return [f"{side}/{joint}" for joint in _ALOHA_ARM_JOINTS]


def _aloha_finger_dofs(side: str) -> list[str]:
    return [f"{side}/left_finger", f"{side}/right_finger"]


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class SetupBase(PolicySetupBase):
    """Abstract strategy for a single pi0 environment configuration.

    Concrete subclasses own:

    * ``pack_observations`` — setup-specific obs → server payload.
    * ``adapter`` — an object with ``__call__(raw_action, *, env)`` and
      ``reset()`` that converts a raw server action array into an Eden
      ``{controller_name: torch.Tensor}`` dict.

    Env wiring (``observation_options`` / ``action_options``) stays as
    classmethod factories so the caller can build the env without a live
    ``PiPolicy`` instance.

    A setup whose server reads a wrist image on every request sets ``wrist_image_required``
    (``PiPolicy.use_wrist_image`` then defaults to True for it) and names the server-side reader
    in ``_WRIST_IMAGE_READER`` / the payload key in ``_WRIST_IMAGE_KEY`` for the error message.

    ``policy_fields`` lists the setup-specific ``PiPolicy`` fields (``droid_*``, ``aloha_*``) the
    setup reads. A setup-specific field that the selected setup does not read must keep its
    default: construction refuses one set to anything else, since the setup would ignore it.

    Every action row is one step of the upstream client's control loop (``control_hz``), so construction also
    refuses an env whose control period ``env.dt`` is another one.
    """

    #: The setup-specific ``PiPolicy`` fields this setup reads; any other setup's must stay at their defaults.
    policy_fields: ClassVar[frozenset[str]] = frozenset()
    #: The server reads the wrist image on every request; building without it is refused.
    wrist_image_required: ClassVar[bool] = False
    _WRIST_IMAGE_READER: ClassVar[str] = ""
    _WRIST_IMAGE_KEY: ClassVar[str] = ""
    #: Width of the concatenated ``proprio`` group :meth:`pack_observations` slices by column; construction refuses
    #: an env whose group is another width or not the plain current reading (:func:`check_plain_observation_group`).
    #: ``None`` checks only that the group exists.
    _PROPRIO_WIDTH: ClassVar[int | None] = None
    #: The upstream client that sends ``prompt`` with every request, named in the error raised when
    #: ``task_description`` is unset. ``None``: the prompt is optional and omitted when unset.
    _PROMPT_SENT_BY: ClassVar[str | None] = None
    #: The upstream control rate in Hz: each action row is one step of ``1 / control_hz`` s, which construction
    #: requires of ``env.dt`` (:meth:`_check_control_period`). ``_CONTROL_HZ_SOURCE`` names where upstream sets it.
    control_hz: ClassVar[float]
    _CONTROL_HZ_SOURCE: ClassVar[str]
    #: ``(sim_dt as typed, decimation)`` with ``sim_dt * decimation = 1 / control_hz``, the refusal's example fix;
    #: ``_CONTROL_TIMING_NOTE`` is appended to it (e.g. the ``sim_substeps`` a Robotiq 2F85 needs), or empty.
    _CONTROL_TIMING: ClassVar[tuple[str, int]]
    _CONTROL_TIMING_NOTE: ClassVar[str] = ""

    def __init__(self, policy: "PiPolicy") -> None:
        super().__init__(policy)
        self._refuse_ignored_policy_fields(policy)
        if self.wrist_image_required and not policy.use_wrist_image:
            raise ValueError(
                f"PiPolicy.policy_setup={policy.policy_setup!r} needs use_wrist_image=True: OpenPI's "
                f"{self._WRIST_IMAGE_READER} reads {self._WRIST_IMAGE_KEY!r} on every request, so the server would "
                f"fail on the first inference; got use_wrist_image={policy.use_wrist_image!r}. Fix: leave "
                "PiPolicy.use_wrist_image unset (it defaults to True for this setup) and build the env with "
                f"PiPolicy.observation_options(policy_setup={policy.policy_setup!r})."
            )
        self._check_observation_groups(policy)
        self._check_control_period(policy)

    def _check_control_period(self, policy: "PiPolicy") -> None:
        """Refuse an env whose control period ``env.dt`` is not the upstream client's ``1 / control_hz`` s."""
        check_control_rate(
            policy.env,
            control_hz=self.control_hz,
            subject=f"PiPolicy.policy_setup={policy.policy_setup!r}",
            source=self._CONTROL_HZ_SOURCE,
            timing=self._CONTROL_TIMING,
            timing_note=self._CONTROL_TIMING_NOTE,
        )

    def _check_observation_groups(self, policy: "PiPolicy") -> None:
        """Refuse an env without a group :meth:`pack_observations` reads, rather than a ``KeyError`` at the first step.

        Every setup reads ``image`` and ``proprio``, and ``wrist_image`` when ``use_wrist_image``. The image groups
        are camera frames, so only their presence is checked; ``proprio`` is sliced by column, so a setup that declares
        ``_PROPRIO_WIDTH`` also has its width and plainness checked.
        """
        manager = getattr(policy.env, "observation_manager", None)
        if manager is None:
            return
        groups = sorted(manager.group_term_names)
        needed = ["image", "proprio"] + (["wrist_image"] if policy.use_wrist_image else [])
        missing = [group for group in needed if group not in groups]
        setup = policy.policy_setup
        if not missing:
            if self._PROPRIO_WIDTH is not None:
                check_plain_observation_group(
                    policy.env,
                    "proprio",
                    width=self._PROPRIO_WIDTH,
                    fix=f"build the env's 'proprio' group from PiPolicy.observation_options(policy_setup={setup!r}).",
                )
            return
        wrist = ", use_wrist_image=True" if "wrist_image" in missing else ""
        or_drop_wrist = (
            ", or PiPolicy.configure(use_wrist_image=False) if the checkpoint takes no wrist image"
            if missing == ["wrist_image"] and not self.wrist_image_required
            else ""
        )
        raise ValueError(
            f"PiPolicy.policy_setup={setup!r} packs observation groups {needed} into every request "
            f"(use_wrist_image={policy.use_wrist_image!r}), but the env has no {missing}; its groups: {groups}. "
            f"Fix: build the env's observation_options from PiPolicy.observation_options(policy_setup={setup!r}"
            f"{wrist}){or_drop_wrist}. Do not feed the server a placeholder (zeros, a copy of another camera) "
            "instead: the checkpoint would act on an input it was never trained on."
        )

    def _refuse_ignored_policy_fields(self, policy: "PiPolicy") -> None:
        """Raise when a setup-specific field another setup reads was set, under this setup, off its default."""
        registry = type(policy).SETUP_REGISTRY
        setup_specific = set().union(*(setup.policy_fields for setup in registry.values()))
        ignored = {
            name: getattr(policy, name)
            for name in sorted(setup_specific - self.policy_fields)
            if getattr(policy, name) != getattr(type(policy), name)
        }
        if not ignored:
            return
        given = ", ".join(f"{name}={value!r}" for name, value in ignored.items())
        readers = ", ".join(
            f"{name}: {sorted(key for key, setup in registry.items() if name in setup.policy_fields)}"
            for name in ignored
        )
        defaults = ", ".join(f"{name}={getattr(type(policy), name)!r}" for name in ignored)
        has, it = ("has", "it") if len(ignored) == 1 else ("have", "them")
        raise ValueError(
            f"PiPolicy {given} {has} no effect under policy_setup={policy.policy_setup!r}, which does not read "
            f"{it}; the setups that do: {readers}. Fix: drop {', '.join(ignored)} from PiPolicy.configure(...) "
            f"(defaults: {defaults}), or select the policy_setup you meant. Do not leave {it} set expecting an "
            "effect: this setup never reads it, so the run would not do what the config says."
        )

    def _prompt(self) -> dict[str, str]:
        """``{"prompt": task_description}``; ``{}`` when unset and optional, a raise when unset and required."""
        p = self.policy
        if p.task_description is not None:
            return {"prompt": p.task_description}
        if self._PROMPT_SENT_BY is None:
            return {}
        raise ValueError(
            f"PiPolicy.policy_setup={p.policy_setup!r} sends 'prompt' with every request, as {self._PROMPT_SENT_BY} "
            "does, but PiPolicy.task_description is None. Fix: policy.set_task('<instruction>') before the first "
            "step, or PiPolicy.configure(task_description='<instruction>'). Do not fill in a generic instruction: "
            "the checkpoint is language-conditioned and acts on whatever it is told."
        )

    def _prepare_image(self, image: torch.Tensor | np.ndarray) -> np.ndarray:
        """One camera frame -> the server's padded ``(h, w, 3)`` uint8 image (orientation unchanged)."""
        return prepare_image_padded(image, self.policy.image_size)

    def _wrist_image(self, obs: dict, payload_key: str) -> np.ndarray:
        """The padded wrist frame; a missing ``wrist_image`` group raises (``use_wrist_image`` asked for it).

        Construction already refuses an env without the group; this guards an observation dict built by hand.
        """
        p = self.policy
        if "wrist_image" not in obs:
            raise KeyError(
                f"PiPolicy.policy_setup={p.policy_setup!r} sends {payload_key!r} (use_wrist_image=True) but the "
                f"observation has no 'wrist_image' group; got groups {sorted(obs)}. Fix: build the env with "
                f"PiPolicy.observation_options(policy_setup={p.policy_setup!r}, use_wrist_image=True)."
            )
        return self._prepare_image(obs["wrist_image"])


def _env_action_term(policy: "PiPolicy", term_name: str, expected: type, *, fix: str) -> "ActionTerm":
    """The env's built action term ``term_name``, refused unless it is exactly an ``expected``.

    ``fix`` is the ``PiPolicy.action_options(...)`` call that builds the right term.
    """
    terms = policy.env.action_manager.terms
    term = terms.get(term_name)
    if term is None:
        raise ValueError(
            f"PiPolicy.policy_setup={policy.policy_setup!r} drives action term {term_name!r}, but the env has "
            f"no such term; its action terms are {sorted(terms)}. Fix: build the env's action_options from {fix}."
        )
    if type(term) is not expected:
        raise TypeError(
            f"PiPolicy.policy_setup={policy.policy_setup!r} needs action term {term_name!r} of type "
            f"{expected.__name__}, but the env built {type(term).__name__}. Fix: build the env's action_options "
            f"from {fix}."
        )
    return term


# ---------------------------------------------------------------------------
# Image + state setups (widowx_bridge, google_robot, libero)
# ---------------------------------------------------------------------------


class _ImageStateSetup(SetupBase):
    """``observation/image`` (+ ``observation/wrist_image``) + an 8-D ``observation/state`` payload.

    Subclasses implement :meth:`_pack_state` for their proprio encoding.
    """

    action_dim: int = 7
    _PROMPT_SENT_BY = "the Bridge / Fractal (SimplerEnv) clients, with the episode's instruction,"
    #: ``[xyz(3), quat_wxyz(4), gripper(1)]``, which :meth:`_pack_state` re-encodes per server.
    _PROPRIO_WIDTH = 8
    primary_image_key: str = "observation/image"
    wrist_image_key: str = "observation/wrist_image"
    state_key: str = "observation/state"

    def pack_observations(self, obs: dict) -> dict:
        p = self.policy
        primary_img = self._prepare_image(obs["image"])

        proprio = to_numpy_unbatched(obs["proprio"])
        state = self._pack_state(proprio)

        payload = {
            self.primary_image_key: primary_img,
            self.state_key: state.astype(np.float32),
            **self._prompt(),
        }
        if p.use_wrist_image:
            payload[self.wrist_image_key] = self._wrist_image(obs, self.wrist_image_key)
        return payload

    @abstractmethod
    def _pack_state(self, proprio: np.ndarray) -> np.ndarray:
        """Convert the Eden proprio vector to the state vector sent to the server."""


#: The env's EE-delta arm term, which :class:`EEDeltaAdapter` writes ``[dxyz, drot]`` into.
_EE_DELTA_ARM_TERM = "ee_controller"


class _EEDeltaBase(_ImageStateSetup):
    """Shared obs / action structure for the WidowX EE-delta setups (7-D ``[dxyz, drot, gripper]`` actions).

    Subclasses pick the server's rotation convention (``_ROTATION``) and gripper rule
    (``_GRIPPER_*``) and implement :meth:`_pack_state` for their proprio encoding.
    """

    pred_action_horizon = 5
    exec_horizon = 4

    #: Fractal latch length for the gripper term (Google-robot checkpoints); 0 = plain binarize.
    _STICKY_NUM_REPEAT: int = 0
    #: How the GripperDecisionController reads the server's gripper scalar (see the setups).
    _GRIPPER_INPUT: str = "openness"
    #: Rotation convention of the server's delta rows (see :class:`EEDeltaAdapter`).
    _ROTATION: str = "euler"

    def __init__(self, policy: "PiPolicy") -> None:
        super().__init__(policy)
        self.adapter = EEDeltaAdapter(rotation=self._ROTATION)
        self._check_ee_controller(policy)
        check_env_gripper_term(
            policy.env,
            "gripper_controller",
            input=self._GRIPPER_INPUT,
            sticky_repeats=self._STICKY_NUM_REPEAT,
        )

    def _check_ee_controller(self, policy: "PiPolicy") -> None:
        """Refuse an ``ee_controller`` that would not move the end effector by the server's ``[dxyz, drot]``.

        :class:`EEDeltaAdapter` rotates the delta from the controlled entity's base frame into the world frame and
        writes it into ``ee_controller``. That term must:

        * be exactly a :class:`DifferentialIKController`: a joint term would take the deltas as joint targets;
        * control the point the ``proprio`` group reports (entity, link, and ``ee_local_pos`` = the pose's
          ``offset_pos``): the server's delta moves that point. SimplerEnv's controller moves and turns
          ``ee_gripper_link``, so a term on ``gripper_link``'s origin would shift it by ``drot x 9.4 cm`` on every
          rotation row;
        * have neither task-space weight zero: a zero weight drops the server's translation or rotation rows;
        * have one scale per half: the term scales (and ``tanh``-squashes) the delta per *world* axis after the
          adapter rotated it, so a per-axis scale would act on the world's axes instead of the server's base axes.
          The ``tanh`` itself commutes with the rotation only to first order (the gap is third order in the delta,
          negligible at Bridge's centimetre steps).

        The scale's value, damping and ``max_dq`` are tuning and are not checked.
        """
        from eden.managers.terms.actions.task_space_actions import DifferentialIKController
        from eden.managers.terms.observations.manipulation import EndEffectorPose

        setup = policy.policy_setup
        fix = f"PiPolicy.action_options(policy_setup={setup!r})"
        term = _env_action_term(policy, _EE_DELTA_ARM_TERM, DifferentialIKController, fix=fix)
        driven = (term.entity_name, term.ee_link_name, tuple(term.ee_local_pos))
        for name, observed in _proprio_terms(policy.env):
            if not isinstance(observed, EndEffectorPose):
                continue
            reported = (observed.entity_name, observed.ee_link_name, tuple(observed.offset_pos))
            if reported == driven:
                continue
            reported_entity, reported_link, reported_offset = reported
            driven_entity, driven_link, driven_offset = driven
            raise ValueError(
                f"PiPolicy.policy_setup={setup!r} sends the pose of the point {reported_offset} on "
                f"{reported_entity!r} link {reported_link!r} (proprio term {name!r}), the point the server's "
                f"[dxyz, drot] moves, but action term {_EE_DELTA_ARM_TERM!r} moves the point {driven_offset} on "
                f"{driven_entity!r} link {driven_link!r} (ee_local_pos), so the arm would apply every delta to "
                "another point than the server computed it for. Fix: pass the same entity_name, ee_link_name and "
                f"ee_offset_pos to PiPolicy.observation_options(policy_setup={setup!r}, ...) and "
                f"PiPolicy.action_options(policy_setup={setup!r}, ...), or leave all three at their defaults "
                f"(entity_name='robot', ee_link_name={SIMPLER_WIDOWX_EE_LINK!r}, "
                f"ee_offset_pos={SIMPLER_WIDOWX_EE_OFFSET_POS}: SimplerEnv's ee_gripper_link)."
            )
        dropped = [
            f"{field}=0.0 (drops the server's {rows} rows)"
            for field, rows in (("position_weight", "translation"), ("orientation_weight", "rotation"))
            if getattr(term, field) == 0.0
        ]
        if dropped:
            raise ValueError(
                f"PiPolicy.policy_setup={setup!r} sends a 6-D end-effector delta [dx, dy, dz, drx, dry, drz], but "
                f"action term {_EE_DELTA_ARM_TERM!r} has {', '.join(dropped)}, so the arm would ignore part of "
                f"every action. Fix: build the env's action_options from {fix} (position_weight=1.0, "
                "orientation_weight=1.0). Do not zero a weight to calm a jittering wrist: raise the term's damping "
                "or set max_dq instead, which keep the server's command."
            )
        anisotropic = [
            f"{field}={getattr(term, field)!r}"
            for field, half in (("position_scale", slice(0, 3)), ("orientation_scale", slice(3, 6)))
            if not bool(torch.all(term.effective_scale[half] == term.effective_scale[half][0]))
        ]
        if anisotropic:
            raise ValueError(
                f"PiPolicy.policy_setup={setup!r} rotates the server's base-frame delta into the world frame before "
                f"action term {_EE_DELTA_ARM_TERM!r} scales it per world axis, but the term has "
                f"{', '.join(anisotropic)}, so on a rotated base each axis of the server's delta would be scaled by "
                f"another axis's factor. Fix: build the env's action_options from {fix} with one scalar scale "
                "(e.g. scale=0.5). Do not pre-scale the server's rows in the client to compensate: which world axis "
                "a base axis lands on depends on the base's orientation."
            )

    @classmethod
    def observation_options(
        cls,
        camera_name: str = "camera",
        entity_name: str = "robot",
        ee_link_name: str = SIMPLER_WIDOWX_EE_LINK,
        ee_offset_pos: tuple[float, float, float] = SIMPLER_WIDOWX_EE_OFFSET_POS,
        gripper_dof_name: str = "left_finger",
        open_gripper_pos: float = SIMPLER_WIDOWX_FINGER_RANGE[1],
        closed_gripper_pos: float = SIMPLER_WIDOWX_FINGER_RANGE[0],
        gripper_entity_name: str | None = None,
        gripper_close_pos: float = 0.8,
        use_wrist_image: bool = False,
        wrist_camera_name: str = "wrist_camera",
    ) -> dict[str, "ObservationGroupOptions"]:
        """``ee_offset_pos`` is the reported point in ``ee_link_name``'s frame: SimplerEnv's ``ee_gripper_link``
        on the WidowX's ``gripper_link`` by default (:data:`SIMPLER_WIDOWX_EE_OFFSET_POS`).

        With the WidowX's own finger (``gripper_entity_name=None``) the gripper state is SimplerEnv's
        openness ``1 - get_gripper_closedness()``: ``(q - closed_gripper_pos) / (open_gripper_pos -
        closed_gripper_pos)`` clipped to ``[0, 1]`` (:data:`SIMPLER_WIDOWX_FINGER_RANGE`). A separately
        mounted Robotiq reads ``1 - q / gripper_close_pos``.
        """
        from eden.managers.terms.observations.manipulation import GripperState
        from eden.options.managers.observations import ObservationGroupOptions

        groups = make_image_groups(camera_name, use_wrist_image, wrist_camera_name)
        proprio = make_ee_pose_group(
            entity_name,
            ee_link_name,
            gripper_entity_name=gripper_entity_name,
            gripper_dof_name=gripper_dof_name,
            open_gripper_pos=open_gripper_pos,
            gripper_close_pos=gripper_close_pos,
            offset_pos=ee_offset_pos,
        )
        if gripper_entity_name is None:
            # make_ee_pose_group's WidowX openness closes at q = 0; SimplerEnv's at the URDF's lower finger limit.
            proprio = ObservationGroupOptions(
                ee_pose=proprio.ee_pose,
                gripper=GripperState.configure(
                    entity_name=entity_name,
                    dofs_name=[gripper_dof_name],
                    normalize="openness",
                    open_pos=open_gripper_pos,
                    close_pos=closed_gripper_pos,
                    clip=(0.0, 1.0),
                ),
            )
        groups["proprio"] = proprio
        return groups

    @classmethod
    def action_options(
        cls,
        entity_name: str = "robot",
        ee_link_name: str = SIMPLER_WIDOWX_EE_LINK,
        ee_offset_pos: tuple[float, float, float] = SIMPLER_WIDOWX_EE_OFFSET_POS,
        dofs_name: list[str] | None = None,
        scale: float = 0.5,
    ) -> dict[str, "ActionTermOptions"]:
        """``ee_offset_pos`` is the point the IK moves, in ``ee_link_name``'s frame: the point
        :meth:`observation_options` reports (SimplerEnv's ``ee_gripper_link`` by default).
        """
        # The raw server gripper scalar is forwarded; the GripperDecisionController decides with
        # the rule the setup declares in ``_GRIPPER_INPUT`` at SimplerEnv's fixed 0.5 threshold.
        actions = make_widowx_ee_delta_actions(
            entity_name=entity_name,
            ee_link_name=ee_link_name,
            dofs_name=dofs_name if dofs_name is not None else list(WIDOWX_DOFS),
            scale=scale,
            gripper_input=cls._GRIPPER_INPUT,
            gripper_sticky_repeats=cls._STICKY_NUM_REPEAT,
        )
        arm = actions[_EE_DELTA_ARM_TERM]
        actions[_EE_DELTA_ARM_TERM] = arm.model_copy(update={"ee_local_pos": tuple(ee_offset_pos)})
        return actions


# ---------------------------------------------------------------------------
# widowx_bridge
# ---------------------------------------------------------------------------


class WidowxBridgeSetup(_EEDeltaBase):
    """Bridge/WidowX EE-delta setup.

    Proprioception: 8-D ``[xyz, rpy, pad, gripper_openness]``.
    The EE rotation is re-expressed relative to the Bridge top-down frame.
    Server actions: 7-D ``[dx, dy, dz, drx_euler, dry_euler, drz_euler, gripper]``. The gripper
    openness is forwarded raw; the env's GripperDecisionController (``input="openness"``) opens it
    when the openness is above 0.5 and closes it otherwise.
    """

    control_hz = 5.0
    _CONTROL_HZ_SOURCE = "SimplerEnv steps its Bridge envs (ManiSkill2_real2sim control_freq=5)"
    _CONTROL_TIMING = ("0.005", 40)
    _CONTROL_TIMING_NOTE = (
        " as Eden's SIMPLER Bridge tasks do; with a Robotiq 2F85 mounted (gripper_entity_name) also set "
        "sim_substeps=5 to keep its substep at 1 ms"
    )

    def _pack_state(self, proprio: np.ndarray) -> np.ndarray:
        pos, quat_wxyz, gripper_openness = proprio[:3], proprio[3:7], float(proprio[7])
        return np.concatenate([pos, bridge_rpy(quat_wxyz), np.zeros(1), [gripper_openness]])


# ---------------------------------------------------------------------------
# google_robot
# ---------------------------------------------------------------------------


class GoogleRobotSetup(_EEDeltaBase):
    """Fractal/Google-Robot EE-delta setup.

    Proprioception: 8-D ``[xyz, quat_xyzw, gripper_closedness]``.
    Server actions: EE-delta (Euler) + gripper openness; the env's GripperDecisionController
    applies the fractal sticky latch (10-step hold) the checkpoint was trained with.
    """

    _STICKY_NUM_REPEAT = 10  # the GripperDecisionController's fractal latch (see action_options)
    control_hz = 3.0
    _CONTROL_HZ_SOURCE = "SimplerEnv steps its Google-robot envs (ManiSkill2_real2sim control_freq=3)"
    _CONTROL_TIMING = ("1/300", 100)
    _CONTROL_TIMING_NOTE = (
        "; with a Robotiq 2F85 mounted (gripper_entity_name) also set sim_substeps=4 to keep its substep under 1 ms"
    )

    def _pack_state(self, proprio: np.ndarray) -> np.ndarray:
        pos, quat_wxyz, gripper_openness = proprio[:3], proprio[3:7], float(proprio[7])
        quat_xyzw = np.roll(quat_wxyz, -1)
        gripper_closedness = 1.0 - gripper_openness
        return np.concatenate([pos, quat_xyzw, [gripper_closedness]])


# ---------------------------------------------------------------------------
# libero
# ---------------------------------------------------------------------------

#: The LIBERO task's action term names (``eden.tasks.libero.config``); :class:`LiberoSetup`
#: builds the env from that task's own terms and checks these at construction.
_LIBERO_ARM_TERM = "osc_pose"
_LIBERO_GRIPPER_TERM = "gripper"


class _RobosuiteOSCAdapter(ActionAdapter):
    """7-D robosuite ``OSC_POSE`` row -> the LIBERO task's ``osc_pose`` (6) and ``gripper`` (1) terms.

    openpi ``examples/libero/main.py`` steps the server's row through ``env.step`` unchanged, so
    both slices are forwarded as-is: the OSC term clips the 6 pose dims to ``[-1, 1]`` and scales
    them (0.05 m / 0.5 rad, robosuite's ``output_max``), and :class:`LiberoPandaGripperController`
    steps the fingers by ``sign(gripper)`` (``+1`` closes, ``-1`` opens, robosuite's ``PandaGripper``).
    """

    row_width = 7

    def __call__(self, raw_action: np.ndarray, *, env: "EnvBase") -> dict[str, torch.Tensor]:
        row = torch.tensor(self._check_row(raw_action), dtype=torch.float32, device=env.device)
        return {
            _LIBERO_ARM_TERM: broadcast_to_envs(row[:6], env.num_envs),
            _LIBERO_GRIPPER_TERM: broadcast_to_envs(row[6:], env.num_envs),
        }


class LiberoSetup(_ImageStateSetup):
    """LIBERO setup, matching openpi ``examples/libero/main.py`` end to end.

    https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/main.py

    Env wiring: the stock ``libero`` task's own action terms (``osc_pose``:
    :class:`OperationalSpaceController`, ``gripper``: :class:`LiberoPandaGripperController`),
    built by that task's builder so the two never drift; cameras default to the task's
    ``agentview`` / ``eye_in_hand``, which must render at openpi's 256x256
    (:data:`~eden.extensions.policies.adapters.LIBERO_CAMERA_RESOLUTION`): the stock task renders
    128x128, so override their ``cam_resolution`` in its ``cameras_options``.
    Construction refuses an env whose ``image`` / ``wrist_image`` camera renders at another size.

    Proprioception: 8-D ``[robot0_eef_pos(3), quat2axisangle(robot0_eef_quat)(3), robot0_gripper_qpos(2)]``
    as robosuite reports it, in the world frame:

    * ``robot0_eef_pos`` is the ``grip_site`` (``LiberoPanda.grip_site_pos_local`` on the hand link);
    * ``robot0_eef_quat`` is the *hand body* (``robot0_right_hand``) orientation, not the grip site's
      (robosuite v1.4 ``SingleArm`` reads ``get_body_xquat(robot_model.eef_name)``), so no offset
      rotation is applied. Eden's wxyz quaternion goes through :func:`robosuite_axis_angle`, the
      wxyz form of robosuite's ``quat2axisangle`` on its xyzw quaternion;
    * ``robot0_gripper_qpos`` is ``[q1, q2]`` with robosuite's second finger in ``[-0.04, 0]``;
      Eden's Menagerie Panda reports both fingers positive, so :meth:`_pack_state` negates the second
      (the ``proprio`` group stays a plain reading, which construction checks).

    Server actions: 7-D robosuite ``OSC_POSE`` ``[dpos(3), daxis_angle(3), gripper]``, forwarded
    unchanged (see :class:`_RobosuiteOSCAdapter`).

    OpenPI's ``LiberoInputs`` reads ``observation/wrist_image`` on every request, so the wrist image
    is sent by default and building without it is refused. Both images are mirrored left-right
    before resize-with-pad (see :meth:`_prepare_image`).
    """

    # openpi examples/libero/main.py: replan_steps=5, asserting the chunk has at least that many rows.
    pred_action_horizon = 5
    exec_horizon = 5

    _PROMPT_SENT_BY = "openpi examples/libero/main.py (the task's language instruction)"
    control_hz = 20.0
    _CONTROL_HZ_SOURCE = "openpi examples/libero/main.py steps LIBERO (robosuite's default control_freq=20)"
    _CONTROL_TIMING = ("0.002", 25)
    _CONTROL_TIMING_NOTE = " as Eden's libero task does"
    #: ``[grip_site_pos(3), hand_quat_wxyz(4), q1(1), q2(1)]``, which :meth:`_pack_state` re-encodes for robosuite.
    _PROPRIO_WIDTH = 9
    wrist_image_required = True
    _WRIST_IMAGE_READER = "LiberoInputs"
    _WRIST_IMAGE_KEY = "observation/wrist_image"

    def __init__(self, policy: "PiPolicy") -> None:
        from eden.managers.terms.actions.task_space_actions import OperationalSpaceController
        from eden.tasks.libero.custom_terms.actions import LiberoPandaGripperController

        super().__init__(policy)
        fix = "PiPolicy.action_options(policy_setup='libero') (the libero task's own action terms)"
        _env_action_term(policy, _LIBERO_ARM_TERM, OperationalSpaceController, fix=fix)
        _env_action_term(policy, _LIBERO_GRIPPER_TERM, LiberoPandaGripperController, fix=fix)
        check_libero_camera_resolution(
            policy.env,
            subject="PiPolicy.policy_setup='libero'",
            trained_on="openpi's LIBERO checkpoints were trained on 256x256 renders (examples/libero/main.py: "
            "LIBERO_ENV_RESOLUTION = 256)",
            wrong_fix="Do not change PiPolicy.image_size instead: that is the 224x224 the server's model takes, after "
            "resize-with-pad, and does not restore the detail a smaller render lacks.",
        )
        self.adapter = _RobosuiteOSCAdapter()

    @classmethod
    def observation_options(
        cls,
        camera_name: str = "agentview",
        use_wrist_image: bool = True,
        wrist_camera_name: str = "eye_in_hand",
    ) -> dict[str, "ObservationGroupOptions"]:
        from eden.managers.terms.observations.manipulation import EndEffectorPose
        from eden.managers.terms.observations.proprio import DofsPos
        from eden.options.managers.observations import ObservationGroupOptions

        groups = make_image_groups(camera_name, use_wrist_image, wrist_camera_name)
        # [grip_site_pos(3), hand_quat_wxyz(4), q1(1), q2(1)] in the world frame; _pack_state negates q2.
        groups["proprio"] = ObservationGroupOptions(
            eef_pose=EndEffectorPose.configure(
                entity_name=LIBERO_ROBOT_ENTITY_NAME,
                ee_link_name=LiberoPanda.ee_links_name[0],
                frame="world",
                offset_pos=LiberoPanda.grip_site_pos_local,
                rotation="quat",
            ),
            gripper_qpos=DofsPos.configure(
                entity_name=LIBERO_ROBOT_ENTITY_NAME,
                dofs_name=list(LiberoPanda.gripper_dofs_name),
            ),
        )
        return groups

    @classmethod
    def action_options(cls) -> dict[str, "ActionTermOptions"]:
        """The ``libero`` task's own action terms (``osc_pose`` + ``gripper``), from the task's builder."""
        return libero_task_action_options()

    def _prepare_image(self, image: torch.Tensor | np.ndarray) -> np.ndarray:
        """Mirror the frame left-right (:func:`libero_training_view`), then resize-with-pad: the view openpi's
        LIBERO checkpoints were trained on (the ``physical-intelligence/libero`` frames show it mirrored too).
        """
        return prepare_image_padded(libero_training_view(image), self.policy.image_size)

    def _pack_state(self, proprio: np.ndarray) -> np.ndarray:
        # proprio: [pos(3), quat_wxyz(4), q1(1), q2(1)] = 9-D
        pos = proprio[:3]
        quat_wxyz = proprio[3:7]
        # robosuite's second finger is mirrored ([-0.04, 0]); Eden's Menagerie Panda reports both positive.
        gripper_qpos = proprio[7:9] * np.array([1.0, -1.0])
        axis_angle = robosuite_axis_angle(quat_wxyz)
        return np.concatenate([pos, axis_angle, gripper_qpos])  # 8-D


# ---------------------------------------------------------------------------
# droid
# ---------------------------------------------------------------------------

#: The env's DROID arm term, shared by every DROID setup's action options and adapter.
_DROID_ARM_TERM = "ee_controller"
_DROID_N_ARM = 7


def _droid_arm_term(policy: "PiPolicy", expected: type, *, fix: str) -> "ActionTerm":
    """The env's 7-joint DROID arm term, taking the client's commands unchanged, in the proprio's joint order."""
    from eden.managers.terms.actions.joint_actions import ImplicitVelocityController

    setup = policy.policy_setup
    subject = f"PiPolicy.policy_setup={setup!r}"
    term = _env_action_term(policy, _DROID_ARM_TERM, expected, fix=fix)
    if expected is ImplicitVelocityController:
        commands = "joint velocities, already converted to rad/s,"
        wrong_fix = (
            "Do not rescale the server's rows to cancel it: the client already applies DROID's own conversion "
            f"({DROID_RAD_PER_S_PER_UNIT:g} rad/s per unit), which the term must pass on as it is."
        )
    else:
        commands = "absolute joint targets"
        wrong_fix = (
            "Do not pre-scale the server's rows to cancel it: the server's state and actions share one joint space."
        )
    check_droid_arm_term(
        term, term_name=_DROID_ARM_TERM, subject=subject, fix=fix, commands=commands, wrong_fix=wrong_fix
    )
    arm_dofs = list(term.dofs_name)
    check_droid_joint_state(
        policy.env,
        term,
        group="proprio",
        subject=subject,
        fix=f"pass the same arm_entity_name and arm_dofs_name to PiPolicy.observation_options(policy_setup={setup!r}, "
        f"arm_entity_name={term.entity_name!r}, arm_dofs_name={arm_dofs!r}) and PiPolicy.action_options("
        f"policy_setup={setup!r}, arm_entity_name={term.entity_name!r}, arm_dofs_name={arm_dofs!r})",
    )
    return term


def _proprio_terms(env: "EnvBase") -> list[tuple[str, object]]:
    """``(name, term)`` of the env's ``proprio`` group in column order; empty without an observation manager."""
    manager = getattr(env, "observation_manager", None)
    if manager is None:
        return []
    return [(name, term) for group, name, term in manager.iter_terms() if group == "proprio"]


class DroidSetup(SetupBase):
    """DROID joint-velocity setup.

    Server actions: ``(T, 8)`` = 7 joint velocities in DROID's unit action space + 1 gripper
    scalar binarized at 0.5 by the env's ``GripperDecisionController`` (openpi
    ``examples/droid/main.py``, which steps ``RobotEnv(action_space="joint_velocity")``).

    The arm rows are converted exactly as the DROID robot does: openpi's client clips every dim
    to ``[-1, 1]`` (:data:`DROID_ACTION_CLIP`, which leaves DROID's own ``v / max|v|`` rescale
    with nothing to do), and ``RobotIKSolver.joint_velocity_to_delta`` turns a unit velocity into
    :data:`DROID_MAX_JOINT_DELTA` = 0.2 rad per 15 Hz control step. ``droid_arm_controller_kind``
    picks how that reaches Eden:

    * ``"velocity"`` (default): an :class:`ImplicitVelocityController` tracks the joint velocity
      ``0.2 * v / (1/15 s) = 3 * v`` rad/s (:data:`DROID_RAD_PER_S_PER_UNIT`);
    * ``"velocity_integrate"``: an :class:`ImplicitPDController` (zero reference) is sent DROID's own
      position target ``q + 0.2 * v`` once per policy step, ``q`` read from the joints the env's arm
      term drives, in its ``dofs_name`` order.

    Both are defined per DROID control step, so construction refuses an env whose control period
    ``env.dt`` is not 1/15 s, and an ``ee_controller`` that is not the term the selected kind needs,
    that rescales or offsets the command, or that drives the arm joints in another order than the
    ``proprio`` group reports them (the server's row ``i`` is joint ``i`` of the state it was sent).
    """

    action_dim: int = 8
    # openpi examples/droid/main.py executes open_loop_horizon=8 actions per query. The chunk length
    # depends on the checkpoint (pi05 15, pi0-FAST 10, velocity 10); 8 is only the executed floor, so
    # pass the checkpoint's pred_action_horizon to check for truncated chunks or to ensemble.
    pred_action_horizon = 8
    exec_horizon = 8

    policy_fields = frozenset(
        {
            "droid_arm_controller_kind",
            "droid_velocity_lag",
            "droid_velocity_compensation",
            "droid_gripper_threshold",
            "droid_debug_gripper",
        }
    )
    _PROMPT_SENT_BY = "openpi examples/droid/main.py (the operator's instruction)"
    control_hz = DROID_CONTROL_HZ
    _CONTROL_HZ_SOURCE = "DROID's RobotEnv does (control_hz=15)"
    _CONTROL_TIMING = DROID_CONTROL_TIMING
    _CONTROL_TIMING_NOTE = DROID_CONTROL_TIMING_NOTE
    #: ``[joint_pos(7), gripper_closedness(1)]``.
    _PROPRIO_WIDTH = 8
    wrist_image_required = True
    _WRIST_IMAGE_READER = "DroidInputs"
    _WRIST_IMAGE_KEY = "observation/wrist_image_left"

    def __init__(self, policy: "PiPolicy") -> None:
        from eden.managers.terms.observations.manipulation import GripperState

        super().__init__(policy)
        self.adapter = self._make_adapter(policy)
        check_env_gripper_term(
            policy.env,
            "gripper_controller",
            input="closedness",
            continuous=False,
            threshold=float(policy.droid_gripper_threshold),
        )
        # The proprio's gripper closedness must span the stroke the gripper term commands.
        for name, term in _proprio_terms(policy.env):
            if isinstance(term, GripperState):
                check_gripper_state_matches_action(
                    policy.env, group="proprio", term=name, action_term="gripper_controller"
                )

    def _make_adapter(self, policy: "PiPolicy") -> JointVelocityAdapter:
        """DROID unit velocities -> rad/s commands, or ``q + 0.2 * v`` targets (``droid_arm_controller_kind``)."""
        from eden.managers.terms.actions.joint_actions import ImplicitPDController, ImplicitVelocityController

        kind = policy.droid_arm_controller_kind
        expected = {"velocity": ImplicitVelocityController, "velocity_integrate": ImplicitPDController}
        if kind not in expected:
            raise ValueError(
                f"PiPolicy.droid_arm_controller_kind={kind!r} is not a DROID arm controller kind; expected one of "
                f"{sorted(expected)}. Fix: PiPolicy.configure(droid_arm_controller_kind='velocity'), with the env "
                f"built from PiPolicy.action_options(policy_setup={policy.policy_setup!r}, "
                "arm_controller_kind='velocity')."
            )
        arm = _droid_arm_term(
            policy,
            expected[kind],
            fix=f"PiPolicy.action_options(policy_setup={policy.policy_setup!r}, arm_controller_kind={kind!r})",
        )
        return JointVelocityAdapter(
            n_arm=_DROID_N_ARM,
            # Integration reads the joints the arm term drives, in its column order.
            entity_name=arm.entity_name,
            arm_dofs_name=list(arm.dofs_name),
            mode=kind,
            # target = q + DROID_CONTROL_PERIOD_S * DROID_RAD_PER_S_PER_UNIT * v = q + 0.2 * v per policy step.
            dt=DROID_CONTROL_PERIOD_S,
            action_scale=DROID_RAD_PER_S_PER_UNIT,
            clip=DROID_ACTION_CLIP,
            velocity_lag=policy.droid_velocity_lag,
            velocity_compensation=policy.droid_velocity_compensation,
            # The filter runs once per policy action, which SetupBase pins to DROID's period.
            velocity_lag_dt=DROID_CONTROL_PERIOD_S,
            gripper_threshold=policy.droid_gripper_threshold,
            debug_gripper=policy.droid_debug_gripper,
        )

    @classmethod
    def observation_options(
        cls,
        camera_name: str = "camera",
        arm_entity_name: str = "robot",
        arm_dofs_name: list[str] | None = None,
        gripper_entity_name: str | None = None,
        gripper_dof_name: str = "left_driver_joint",
        gripper_close_pos: float = 0.8,
        use_wrist_image: bool = True,
        wrist_camera_name: str = "wrist_camera",
    ) -> dict[str, "ObservationGroupOptions"]:
        if arm_dofs_name is None:
            arm_dofs_name = list(DROID_ARM_DOFS)

        groups = make_image_groups(camera_name, use_wrist_image, wrist_camera_name)
        groups["proprio"] = make_droid_joint_state_group(
            arm_entity_name,
            arm_dofs_name,
            gripper_entity_name=gripper_entity_name,
            gripper_dof_name=gripper_dof_name,
            gripper_close_pos=gripper_close_pos,
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
        arm_controller_kind: str = "velocity",
        gripper_threshold: float = 0.5,
    ) -> dict[str, "ActionTermOptions"]:
        from eden.managers.terms.actions.joint_actions import ImplicitVelocityController

        # Raw closedness in; the env's GripperDecisionController binarizes at gripper_threshold.
        gripper = dict(gripper_close_pos=gripper_close_pos, threshold=gripper_threshold)
        # "velocity_integrate" integrates the policy's velocities into position targets client-side
        # instead of driving Genesis's velocity actuator.
        if arm_controller_kind == "velocity_integrate":
            return make_droid_joint_position_actions(
                arm_entity_name,
                arm_dofs_name,
                gripper_entity_name=gripper_entity_name,
                gripper_dof_name=gripper_dof_name,
                **gripper,
            )
        if arm_controller_kind != "velocity":
            raise ValueError(
                "DroidSetup.action_options: arm_controller_kind must be 'velocity' or 'velocity_integrate', "
                f"got {arm_controller_kind!r}"
            )
        return {
            _DROID_ARM_TERM: ImplicitVelocityController.configure(
                entity_name=arm_entity_name,
                dofs_name=list(DROID_ARM_DOFS) if arm_dofs_name is None else arm_dofs_name,
            ),
            "gripper_controller": make_droid_gripper_action(
                gripper_entity_name if gripper_entity_name is not None else arm_entity_name,
                gripper_dof_name,
                **gripper,
            ),
        }

    def pack_observations(self, obs: dict) -> dict:
        primary_img = self._prepare_image(obs["image"])
        proprio = to_numpy_unbatched(obs["proprio"])
        joint_pos = proprio[:7].astype(np.float32)
        gripper_closedness = proprio[7:8].astype(np.float32)

        return {
            "observation/exterior_image_1_left": primary_img,
            "observation/joint_position": joint_pos,
            "observation/gripper_position": gripper_closedness,
            **self._prompt(),
            self._WRIST_IMAGE_KEY: self._wrist_image(obs, self._WRIST_IMAGE_KEY),
        }


# ---------------------------------------------------------------------------
# droid_jointpos
# ---------------------------------------------------------------------------


class DroidJointPositionSetup(DroidSetup):
    """DROID joint-position setup.

    The server receives the same observation payload as :class:`DroidSetup`.
    Its action chunk is interpreted as absolute arm joint-position targets
    after OpenPI's output transforms convert model deltas back to positions,
    so the env's ``ee_controller`` must be an :class:`ImplicitPDController` with a zero reference.

    The chunk's rows are joint waypoints 1/15 s apart, so, as for :class:`DroidSetup`, construction
    refuses an env whose control period ``env.dt`` is not 1/15 s. PolaRiS steps it at 15 Hz
    (``droid_cfg.py``: ``sim.dt = 1/120``, ``decimation = 8``); Eden's PolaRiS tasks step every 20 ms,
    which would replay each chunk 3.3x too fast, and need their ``env_options`` set to 15 Hz.

    Its targets go straight to the PD, so it reads none of :class:`DroidSetup`'s velocity fields:
    ``droid_arm_controller_kind``, ``droid_velocity_lag`` and ``droid_velocity_compensation`` must
    keep their defaults.
    """

    policy_fields = frozenset({"droid_gripper_threshold", "droid_debug_gripper"})

    def _make_adapter(self, policy: "PiPolicy") -> JointPositionAdapter:
        from eden.managers.terms.actions.joint_actions import ImplicitPDController

        _droid_arm_term(policy, ImplicitPDController, fix="PiPolicy.action_options(policy_setup='droid_jointpos')")
        return JointPositionAdapter(
            n_arm=_DROID_N_ARM,
            gripper_threshold=policy.droid_gripper_threshold,
            debug_gripper=policy.droid_debug_gripper,
        )

    @classmethod
    def action_options(
        cls,
        arm_entity_name: str = "robot",
        arm_dofs_name: list[str] | None = None,
        gripper_entity_name: str | None = None,
        gripper_dof_name: str = "left_driver_joint",
        gripper_close_pos: float = 0.8,
        gripper_threshold: float = 0.5,
    ) -> dict[str, "ActionTermOptions"]:
        # Raw closedness in; the env's GripperDecisionController binarizes at gripper_threshold.
        return make_droid_joint_position_actions(
            arm_entity_name,
            arm_dofs_name,
            gripper_entity_name=gripper_entity_name,
            gripper_dof_name=gripper_dof_name,
            gripper_close_pos=gripper_close_pos,
            threshold=gripper_threshold,
        )


# ---------------------------------------------------------------------------
# aloha
# ---------------------------------------------------------------------------

#: One action term per arm and per gripper, in the server row's order.
_ALOHA_TERMS = ("left_arm_controller", "left_gripper_controller", "right_arm_controller", "right_gripper_controller")


class _AlohaAdapter(ActionAdapter):
    """14-D ``[left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]`` -> four action terms.

    The arm slices are absolute joint targets; each gripper scalar is the normalised finger
    position, which the env's ``GripperDecisionController`` (``input="openness"``,
    ``continuous=True``) un-normalises between the finger's close and open positions.
    """

    row_width = 14

    def __call__(self, raw_action: np.ndarray, *, env: "EnvBase") -> dict[str, torch.Tensor]:
        row = torch.tensor(self._check_row(raw_action), dtype=torch.float32, device=env.device)
        slices = (row[:6], row[6:7], row[7:13], row[13:14])
        return {name: broadcast_to_envs(part, env.num_envs) for name, part in zip(_ALOHA_TERMS, slices)}


class AlohaSetup(SetupBase):
    """ALOHA bimanual joint-position setup, following openpi's ``examples/aloha_sim`` / gym_aloha.

    https://github.com/Physical-Intelligence/openpi/blob/main/examples/aloha_sim/env.py
    https://github.com/huggingface/gym-aloha/blob/main/gym_aloha/tasks/sim.py
    https://github.com/huggingface/gym-aloha/blob/main/gym_aloha/constants.py

    State and actions are 14-D ``[left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]``,
    arm joints in radians. openpi's ``AlohaInputs`` / ``AlohaOutputs`` (``adapt_to_pi=True``) flip
    joint signs and convert the gripper to and from pi's angular space on the server, so the client
    speaks gym_aloha's convention on both ends:

    * state gripper: ``PUPPET_GRIPPER_POSITION_NORMALIZE_FN`` of the left finger's position
      (``0`` = closed, ``1`` = open), as ``BimanualViperXTask.get_qpos`` reports it;
    * action gripper: the same normalised position, which ``before_step`` un-normalises
      (``PUPPET_GRIPPER_POSITION_UNNORMALIZE_FN``) into a target for both fingers.

    Eden's :class:`Aloha` model (Menagerie-derived) places its finger joints differently from
    gym_aloha's ViperX MJCF (both fingers in ``[-0.001, 0.04]`` here, gym_aloha's
    ``PUPPET_GRIPPER_POSITION_CLOSE/OPEN`` = 0.01844 / 0.058 m), so the normalisation spans the model's
    own ``Aloha.close_dofs_pos`` / ``Aloha.open_dofs_pos``; both Eden fingers take the same target
    (gym_aloha's ``[g, -g]`` is its mirrored second finger).

    Images are sent as a CHW dict: ``cam_high`` (gym_aloha's ``top`` camera, the only one
    ``examples/aloha_sim`` sends) + ``cam_left_wrist`` when ``use_wrist_image``. ``prompt`` is sent
    only when ``task_description`` is set: ``examples/aloha_sim`` sends none and the server's
    ``InjectDefaultPrompt`` fills in its config's ``default_prompt``.

    Chunking: ``examples/aloha_sim/main.py`` wraps the client in ``ActionChunkBroker(action_horizon=10)``,
    so the first 10 rows of each chunk (``pi0_aloha_sim`` returns 50) are executed before re-querying.

    Each arm term must be an :class:`ImplicitPDController` that takes the server's absolute joint
    targets unchanged (``ReferenceSource.ZERO``, unit scale, zero offset); construction refuses another.
    """

    action_dim: int = 14
    # openpi examples/aloha_sim/main.py: ActionChunkBroker(action_horizon=10).
    pred_action_horizon = 10
    exec_horizon = 10

    policy_fields = frozenset({"aloha_left_camera_name"})
    control_hz = 50.0
    _CONTROL_HZ_SOURCE = "openpi examples/aloha_sim steps gym_aloha (constants.py: DT = 0.02)"
    # gym_aloha's MJCF keeps MuJoCo's default 0.002 s timestep: 10 physics steps per control step.
    _CONTROL_TIMING = ("0.002", 10)
    #: ``[left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)]``.
    _PROPRIO_WIDTH = 14

    def __init__(self, policy: "PiPolicy") -> None:
        from eden.managers.terms.actions.joint_actions import ImplicitPDController

        super().__init__(policy)
        fix = "PiPolicy.action_options(policy_setup='aloha')"
        for side in ("left", "right"):
            name = f"{side}_arm_controller"
            arm = _env_action_term(policy, name, ImplicitPDController, fix=fix)
            check_joint_pass_through(
                arm,
                term_name=name,
                subject="PiPolicy.policy_setup='aloha'",
                fix=fix,
                commands="absolute joint targets",
                wrong_fix="Do not pre-scale the server's rows to cancel it: the server's state and actions share one "
                "joint space.",
            )
            check_env_gripper_term(policy.env, f"{side}_gripper_controller", input="openness", continuous=True)
        self.adapter = _AlohaAdapter()

    @classmethod
    def observation_options(
        cls,
        camera_name: str = "camera",
        entity_name: str = "robot",
        use_wrist_image: bool = False,
        wrist_camera_name: str = "wrist_camera",
    ) -> dict[str, "ObservationGroupOptions"]:
        from eden.managers.terms.observations.manipulation import GripperState
        from eden.managers.terms.observations.proprio import DofsPos
        from eden.options.managers.observations import ObservationGroupOptions

        groups = make_image_groups(camera_name, use_wrist_image, wrist_camera_name)
        terms = {}
        for side in ("left", "right"):
            finger = _aloha_finger_dofs(side)[0]
            terms[f"{side}_arm"] = DofsPos.configure(entity_name=entity_name, dofs_name=_aloha_arm_dofs(side))
            terms[f"{side}_gripper"] = GripperState.configure(
                entity_name=entity_name,
                dofs_name=[finger],
                normalize="openness",
                open_pos=Aloha.open_dofs_pos[finger],
                close_pos=Aloha.close_dofs_pos[finger],
            )
        # 14-D [left_arm(6), left_gripper(1), right_arm(6), right_gripper(1)], in declaration order.
        groups["proprio"] = ObservationGroupOptions(**terms)
        return groups

    @classmethod
    def action_options(cls, entity_name: str = "robot") -> dict[str, "ActionTermOptions"]:
        from eden.constants import ReferenceSource
        from eden.managers.terms.actions.binary_actions import GripperDecisionController
        from eden.managers.terms.actions.joint_actions import ImplicitPDController

        terms = {}
        for side in ("left", "right"):
            fingers = _aloha_finger_dofs(side)
            terms[f"{side}_arm_controller"] = ImplicitPDController.configure(
                entity_name=entity_name, dofs_name=_aloha_arm_dofs(side), reference_source=ReferenceSource.ZERO
            )
            # Openness in [0, 1] -> close + (open - close) * openness on both fingers.
            terms[f"{side}_gripper_controller"] = GripperDecisionController.configure(
                entity_name=entity_name,
                dofs_name=fingers,
                open_action=Aloha.open_dofs_pos[fingers[0]],
                close_action=Aloha.close_dofs_pos[fingers[0]],
                input="openness",
                continuous=True,
            )
        return terms

    def pack_observations(self, obs: dict) -> dict:
        p = self.policy
        base_img = self._prepare_image(obs["image"])
        images: dict[str, np.ndarray] = {"cam_high": base_img.transpose(2, 0, 1)}  # CHW

        if p.use_wrist_image:
            images[p.aloha_left_camera_name] = self._wrist_image(obs, p.aloha_left_camera_name).transpose(2, 0, 1)

        proprio = to_numpy_unbatched(obs["proprio"])
        # openpi examples/aloha_sim sends no prompt; the server's InjectDefaultPrompt fills its default_prompt.
        return {"images": images, "state": proprio.astype(np.float32), **self._prompt()}


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SETUP_REGISTRY: dict[str, type[SetupBase]] = {
    "widowx_bridge": WidowxBridgeSetup,
    "google_robot": GoogleRobotSetup,
    "libero": LiberoSetup,
    "droid": DroidSetup,
    "droid_jointpos": DroidJointPositionSetup,
    "aloha": AlohaSetup,
}
