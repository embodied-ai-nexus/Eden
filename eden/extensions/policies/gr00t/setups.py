"""Per-setup strategy classes for Gr00tPolicy.

Each :class:`SetupBase` subclass owns the per-embodiment serialization contract
expected by the GR00T inference server:

* ``observation_options`` / ``action_options`` — classmethod factories used to
  wire the Eden environment.
* ``pack_observations`` — convert Eden obs tensors into the nested
  ``{"video": ..., "state": ..., "language": ...}`` payload of the PolicyServer's
  ``get_action`` request (each leaf carries a leading ``(B, T, ...)``
  axis, ``B = num_envs``). :func:`to_sim_wrapper_request` flattens it for a server
  launched with ``--use-sim-policy-wrapper``.
* ``unpack_actions`` — convert one row of the server's action chunk
  ``(D,)`` numpy array into the ``{controller: torch.Tensor}`` dict consumed by
  Eden's action manager.

The setups below mirror NVIDIA's prebuilt MODALITY_CONFIGS:

* :class:`WidowxSimplerSetup`  — ``simpler_env_widowx``
* :class:`LiberoSetup`         — ``libero_sim``
* :class:`OxeDroidSetup`       — ``oxe_droid_relative_eef_relative_joint``

Shared helpers (image groups, WidowX EE-delta actions, ``EEDeltaAdapter``, …)
come from :mod:`eden.extensions.policies.adapters`.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import torch

import eden as en
from eden.extensions.policies.adapters import (
    DROID_ARM_DOFS,
    DROID_CONTROL_HZ,
    DROID_FLANGE_FROM_FR3_ATTACHMENT_EULER,
    GRIPPER_DECISION_FIELDS,
    WIDOWX_DOFS,
    EEDeltaAdapter,
    JointPositionAdapter,
    bridge_rpy,
    check_env_gripper_term,
    check_gripper_state_matches_action,
    check_control_rate,
    check_droid_arm_term,
    check_droid_joint_state,
    check_libero_camera_resolution,
    check_plain_observation_group,
    libero_task_action_options,
    libero_training_view,
    make_droid_joint_position_actions,
    make_droid_joint_state_group,
    make_image_groups,
    make_widowx_ee_delta_actions,
    robosuite_axis_angle,
)
from eden.extensions.policies.client import (
    FrameHistory,
    ModalitySchema,
    PolicySetupBase,
    concat_action_streams,
    validate_modality_config,
)
from eden.options.robots.libero import LIBERO_ROBOT_ENTITY_NAME, LiberoPanda
from eden.utils.geom import R_to_quat, quat_from_euler_xyz, quat_mul
from eden.utils.image import prepare_image_padded, prepare_image_resized, to_uint8_hwc
from eden.utils.torch import broadcast_to_envs, to_numpy_unbatched

if TYPE_CHECKING:
    from eden.envs.base import EnvBase
    from eden.extensions.policies.gr00t.policy import Gr00tPolicy
    from eden.options.managers.actions import ActionTermOptions
    from eden.options.managers.observations import ObservationGroupOptions


# ---------------------------------------------------------------------------
# Request format: bare Gr00tPolicy vs. Gr00tSimPolicyWrapper
# ---------------------------------------------------------------------------


def to_sim_wrapper_request(payload: dict) -> dict:
    """Nested ``Gr00tPolicy`` request -> the flat one ``Gr00tSimPolicyWrapper`` takes.

    ``{"video": {k: v}}`` becomes ``{"video.k": v}`` (``state`` likewise), and each language key
    maps to its ``(B,)`` batch of strings instead of the nested ``(B, T=1)`` lists, as
    ``Gr00tSimPolicyWrapper._get_action`` (``gr00t/policy/gr00t_policy.py``) unpacks them.
    """
    flat = {f"{modality}.{key}": value for modality in ("video", "state") for key, value in payload[modality].items()}
    flat.update({key: [texts[0] for texts in batch] for key, batch in payload["language"].items()})
    return flat


# What a strict server says when the request's format is the other one: the sim wrapper asserts
# ``Video key 'video.<key>' must be in observation`` and a bare Gr00tPolicy asserts
# ``Observation must contain a 'video' key`` (``check_observation`` in gr00t/policy/gr00t_policy.py).
_WRAPPER_EXPECTED_FLAT = re.compile(r"(Video|State) key '(video|state)\.")
_BARE_EXPECTED_NESTED = re.compile(r"Observation must contain a '(video|state|language)' key")


def wire_format_hint(server_message: str, *, use_sim_policy_wrapper: bool) -> str | None:
    """What to flip when ``server_message`` says the request came in the other format; ``None`` otherwise."""
    if not use_sim_policy_wrapper and _WRAPPER_EXPECTED_FLAT.search(server_message):
        return (
            "The server was launched with --use-sim-policy-wrapper (it takes flat 'video.<key>' / 'state.<key>' "
            "keys), but this client sent the nested request. Fix: Gr00tPolicy.configure(..., "
            "use_sim_policy_wrapper=True), or relaunch run_gr00t_server.py without --use-sim-policy-wrapper. "
            "Do not rename the setup's modality keys: they are right, only the request format differs."
        )
    if use_sim_policy_wrapper and _BARE_EXPECTED_NESTED.search(server_message):
        return (
            "The server was launched without --use-sim-policy-wrapper (it takes the nested "
            "{'video': ..., 'state': ..., 'language': ...} request), but this client sent the flat sim-wrapper "
            "request. Fix: Gr00tPolicy.configure(..., use_sim_policy_wrapper=False), or relaunch "
            "run_gr00t_server.py with --use-sim-policy-wrapper."
        )
    return None


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class SetupBase(PolicySetupBase):
    """Strategy for a single GR00T embodiment configuration."""

    # Default chunk-execution policy. Subclasses override.
    pred_action_horizon: int = 8
    exec_horizon: int = 4

    # Default image_size used by ``Gr00tPolicy`` when the policy attribute is
    # left at None (so each setup can match the size its training pipeline
    # used). Subclasses override.
    default_image_size: tuple[int, int] = (224, 224)

    # The order in which per-key (B, T, 1) action streams are concatenated to
    # produce a single (B, T, D) action tensor. Subclasses override.
    action_keys: tuple[str, ...] = ()
    # Modality keys this setup packs; checked against the server's get_modality_config.
    video_keys: tuple[str, ...] = ()
    state_keys: tuple[str, ...] = ()
    language_key: str = ""
    # Frames per video key and their spacing in control steps (``delta_indices``).
    video_horizon: int = 1
    video_stride: int = 1

    #: Whether the upstream README launches this embodiment's server with
    #: ``--use-sim-policy-wrapper``; the default of ``Gr00tPolicy.use_sim_policy_wrapper``.
    upstream_sim_policy_wrapper: ClassVar[bool] = False
    #: ``Gr00tPolicy`` fields only this setup reads (see ``Gr00tPolicy._SETUP_SPECIFIC_FIELDS``).
    policy_fields: ClassVar[frozenset[str]] = frozenset()
    #: Observation groups ``pack_observations`` reads, as its own ``observation_options()`` builds them:
    #: group -> concatenated per-env dims, ``None`` for a camera group (any ``(H, W, C)`` image).
    observation_groups: ClassVar[dict[str, tuple[int, ...] | None]] = {}

    def check_env_observations(self, env: "EnvBase") -> None:
        """Raise unless ``env`` publishes the observation groups (and proprio widths) this setup reads.

        Checked at construction, so an env built from another layout (a stock task's ``policy`` group)
        fails here with the fix rather than as a ``KeyError`` on the first step.
        """
        published = env.observation_manager.group_obs_dim
        wrong = []
        for group, dims in self.observation_groups.items():
            found = published.get(group)
            if not (isinstance(found, tuple) and (len(found) == 3 if dims is None else found == dims)):
                got = "missing" if found is None else f"got {found}"
                wrong.append(f"{group!r} ({'(H, W, C) image' if dims is None else dims}, {got})")
        if wrong:
            setup = self.policy.policy_setup
            raise ValueError(
                f"Gr00tPolicy.policy_setup={setup!r} reads observation groups {', '.join(wrong)}; the env publishes {published}. "
                f"Fix: build the env's observation_options with ObservationManagerOptions(**Gr00tPolicy."
                f"observation_options(policy_setup={setup!r}, ...)), naming the env's cameras through its "
                "camera_name arguments. Do not rename the env's existing groups to match: their terms compute "
                "a different state layout."
            )

    #: Rate the checkpoint's training episodes were recorded at: one action row per control step. ``exec_horizon``
    #: and ``video_stride`` count env steps, so the env must step at exactly ``1 / control_hz``;
    #: ``control_hz_source`` names where upstream sets it.
    control_hz: ClassVar[float]
    control_hz_source: ClassVar[str]

    def check_env_control_rate(self, env: "EnvBase") -> None:
        """Raise unless ``env.dt`` is the control period the checkpoint was trained at (``1 / control_hz``)."""
        check_control_rate(
            env,
            control_hz=self.control_hz,
            subject=f"Gr00tPolicy.policy_setup={self.policy.policy_setup!r}",
            source=self.control_hz_source,
        )

    def modality_schema(self) -> ModalitySchema:
        """The schema this setup packs and decodes, for :func:`validate_modality_config`."""
        return ModalitySchema(
            video_keys=tuple(self.video_keys),
            state_keys=tuple(self.state_keys),
            action_keys=tuple(self.action_keys),
            language_key=self.language_key,
            action_horizon=self.policy.pred_action_horizon,
            video_horizon=int(self.video_horizon),
            video_stride=int(self.video_stride),
        )

    def validate_server_config(self, config: dict) -> None:
        validate_modality_config(config, self.modality_schema(), setup_name=type(self).__name__)

    # -- inference -----------------------------------------------------

    def language(self) -> dict[str, list[list[str]]]:
        """The nested language entry; a missing instruction raises rather than reaching the server."""
        task = self.policy.task_description
        if task is None:
            raise ValueError(
                "Gr00tPolicy has no task_description, and every GR00T checkpoint is conditioned on one. "
                "Fix: call policy.set_task('<instruction>') before policy.step(), or "
                "Gr00tPolicy.configure(..., task_description='<instruction>')."
            )
        return {self.language_key: [[task]]}

    def decode_response(self, response) -> np.ndarray:
        """``[action_dict, info]`` reply -> ``(T, D)`` chunk in ``action_keys`` order.

        The sim wrapper replies ``action.<key>``; a bare ``Gr00tPolicy`` the bare key.
        """
        action_dict, _info = response
        prefix = "action." if self.policy.use_sim_policy_wrapper else ""
        return concat_action_streams(action_dict, tuple(f"{prefix}{key}" for key in self.action_keys))


#: The arm action term the WidowX and OXE-DROID setups drive (the name their ``action_options()`` give it).
_ARM_TERM = "ee_controller"


def _require_arm_term(env: "EnvBase", setup: str, expected: type, *, why: str, wrong_fix: str):
    """The env's ``ee_controller`` when it is exactly an ``expected`` term (what the setup's action_options build).

    Exact type, not ``isinstance``: a subclass (an EMA-filtered joint target, ...) reinterprets the same row.
    """
    terms = env.action_manager.terms
    term = terms.get(_ARM_TERM)
    if type(term) is not expected:
        found = "missing" if term is None else f"of type {type(term).__name__}"
        raise ValueError(
            f"Gr00tPolicy.policy_setup={setup!r} needs action term {_ARM_TERM!r} of type {expected.__name__} exactly ({why}), but the "
            f"env's {_ARM_TERM!r} is {found} (action terms: {sorted(terms)}). Fix: build the env's action_options "
            f"with ActionManagerOptions(**Gr00tPolicy.action_options(policy_setup={setup!r})), passing the rig's "
            f"entity / DOF names as keyword arguments. {wrong_fix}"
        )
    return term


def _check_gripper_term_decides_as_built(setup: type[SetupBase], env: "EnvBase") -> None:
    """Raise unless the env's ``gripper_controller`` decides open/close exactly as ``setup.action_options()``'s does.

    Every :data:`GRIPPER_DECISION_FIELDS` entry is compared, so a task's own threshold, hysteresis, hold, latch or
    continuous mode cannot replace the upstream client's plain rule. The decision fields do not depend on the
    factory's entity / DOF arguments, so the default ``action_options()`` stands for any rig.
    """
    built = setup.action_options()["gripper_controller"]
    check_env_gripper_term(
        env, "gripper_controller", **{name: getattr(built, name) for name in GRIPPER_DECISION_FIELDS}
    )


# ---------------------------------------------------------------------------
# simpler_env_widowx
# ---------------------------------------------------------------------------


_WIDOWX_LANGUAGE_KEY = "annotation.human.action.task_description"
_WIDOWX_VIDEO_KEY = "image_0"

#: SimplerEnv's WidowX end-effector frame, ``ee_gripper_link``, in ``gripper_link``'s frame (identity
#: rotation). ManiSkill2_real2sim's ``WidowX`` reports ``eef_pos`` from ``ee_gripper_link``
#: (``mani_skill2_real2sim/agents/robots/widowx.py``), which the Interbotix wx250s URDF puts ``ee_arm`` 0.043 +
#: ``ee_bar`` 0.023 + ``ee_gripper`` 0.027575 m along ``gripper_link``'s +x. Eden's ``wx250s.xml`` has no such
#: body (its finger links sit at x = 0.066 = 0.043 + 0.023), so the proprio adds it as a tool offset; without
#: it every state x, y, z is ~9 cm off the training distribution.
SIMPLER_WIDOWX_EE_OFFSET_POS: tuple[float, float, float] = (0.093575, 0.0, 0.0)

#: SimplerEnv's WidowX finger travel ``(closed, open)``: ManiSkill2_real2sim's ``wx250s.urdf`` limits both finger
#: joints to ``lower=0.015 upper=0.037``, and ``WidowX.get_gripper_closedness`` is ``(upper - q) / (upper - lower)``
#: clamped at ``>= 0``, so the state's ``1 - closedness`` openness is 0 at 0.015 and 1 at 0.037. Eden's
#: ``wx250s.xml`` fingers travel down to 0.01, so ``q / 0.037`` would report a closed gripper as ~0.27 open.
SIMPLER_WIDOWX_FINGER_RANGE: tuple[float, float] = (0.015, 0.037)
#: SimplerEnv WidowX/Bridge control rate: ManiSkill2_real2sim's bridge scenes set ``control_freq = 5``
#: (``envs/custom_scenes/put_on_in_scene.py``); Eden's stock SIMPLER tasks step 0.005 s x 40 = 0.2 s.
_SIMPLER_WIDOWX_CONTROL_HZ = 5.0


class WidowxSimplerSetup(SetupBase):
    """SimplerEnv WidowX (Bridge) embodiment for GR00T.

    Mirrors ``PiPolicy``'s ``widowx_bridge`` setup (:class:`eden.extensions.policies.pi.setups.WidowxBridgeSetup`):

    * Proprioception: Eden returns 8-D ``[pos(3), quat_wxyz(4), gripper_openness(1)]``;
      we re-express the rotation in the Bridge top-down frame and split into
      per-key scalars ``x, y, z, roll, pitch, yaw, pad, gripper``.
    * Server actions: 7-D per row, ``[dx, dy, dz, droll, dpitch, dyaw, gripper]``, executed by a
      ``DifferentialIKController``. The gripper openness is forwarded raw; the env's
      ``GripperDecisionController`` (``input="openness"``) closes at or below 0.5, with no latch.
    * Gripper state: SimplerEnv's openness over the finger travel :data:`SIMPLER_WIDOWX_FINGER_RANGE`.
    * Rate: 5 Hz, as SimplerEnv's bridge scenes.

    Upstream's SimplerEnv README launches this server with ``--use-sim-policy-wrapper``.
    """

    pred_action_horizon: int = 8  # MODALITY_CONFIGS["simpler_env_widowx"]["action"].delta_indices length
    # examples/SimplerEnv/README.md runs the WidowX rollout with --n-action-steps 4.
    exec_horizon: int = 4
    # Matches gr00t/eval/sim/SimplerEnv/simpler_env.py:WidowXBridgeEnv.
    default_image_size: tuple[int, int] = (256, 256)
    action_keys: tuple[str, ...] = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
    action_dim: int = 7
    video_keys: tuple[str, ...] = (_WIDOWX_VIDEO_KEY,)
    state_keys: tuple[str, ...] = ("x", "y", "z", "roll", "pitch", "yaw", "pad", "gripper")
    language_key: str = _WIDOWX_LANGUAGE_KEY
    upstream_sim_policy_wrapper: ClassVar[bool] = True
    control_hz: ClassVar[float] = _SIMPLER_WIDOWX_CONTROL_HZ
    control_hz_source: ClassVar[str] = "SimplerEnv steps its Bridge envs (ManiSkill2_real2sim control_freq=5)"
    # [pos(3), quat_wxyz(4), gripper_openness(1)]
    observation_groups: ClassVar[dict[str, tuple[int, ...] | None]] = {"image": None, "proprio": (8,)}

    def __init__(self, policy: "Gr00tPolicy") -> None:
        super().__init__(policy)
        from eden.managers.terms.actions.task_space_actions import DifferentialIKController

        _require_arm_term(
            policy.env,
            policy.policy_setup,
            DifferentialIKController,
            why="the rows are Bridge end-effector deltas",
            wrong_fix="Do not convert the deltas to joint targets in the client: the IK term's position_scale / "
            "orientation_scale give them their units.",
        )
        # Gripper openness is forwarded raw; upstream's ``WidowXBridgeEnv._postprocess_gripper`` is a plain
        # ``openness > 0.5`` with no latch or hysteresis, which is what action_options()'s term decides.
        _check_gripper_term_decides_as_built(type(self), policy.env)
        self.adapter = EEDeltaAdapter(rotation="euler")

    def check_env_observations(self, env: "EnvBase") -> None:
        """The base check, then ``proprio`` as the plain current reading :meth:`pack_observations` slices by column.

        Its gripper openness is not compared with ``gripper_controller``'s targets: SimplerEnv's state spans the URDF
        finger travel (:data:`SIMPLER_WIDOWX_FINGER_RANGE`) while the command closes the fingers to 0, by design.
        """
        super().check_env_observations(env)
        check_plain_observation_group(
            env,
            "proprio",
            fix="build 'proprio' with Gr00tPolicy.observation_options(policy_setup='simpler_env_widowx')['proprio'] "
            "as returned, adding no history_length, scale, clip, noise or modifiers. Do not undo the transform in "
            "pack_observations: the server expects SimplerEnv's raw end-effector state.",
        )

    @classmethod
    def observation_options(
        cls,
        camera_name: str = "camera",
        entity_name: str = "robot",
        ee_link_name: str = "gripper_link",
        ee_offset_pos: tuple[float, float, float] = SIMPLER_WIDOWX_EE_OFFSET_POS,
        gripper_dof_name: str = "left_finger",
        open_gripper_pos: float = SIMPLER_WIDOWX_FINGER_RANGE[1],
        closed_gripper_pos: float = SIMPLER_WIDOWX_FINGER_RANGE[0],
    ) -> dict[str, "ObservationGroupOptions"]:
        """``ee_offset_pos`` is the reported point in ``ee_link_name``'s frame: SimplerEnv's ``ee_gripper_link``
        on the WidowX's ``gripper_link`` by default (:data:`SIMPLER_WIDOWX_EE_OFFSET_POS`).

        The gripper state is SimplerEnv's openness of the WidowX fingers on ``entity_name``,
        ``(q - closed_gripper_pos) / (open_gripper_pos - closed_gripper_pos)`` clipped to ``[0, 1]``
        (:data:`SIMPLER_WIDOWX_FINGER_RANGE`). There is no separately mounted gripper: ``action_options()``
        drives the WidowX fingers on the arm entity.
        """
        from eden.managers.terms.observations.manipulation import EndEffectorPose, GripperState
        from eden.options.managers.observations import ObservationGroupOptions

        groups = make_image_groups(camera_name, use_wrist_image=False, wrist_camera_name="")
        groups["proprio"] = ObservationGroupOptions(
            ee_pose=EndEffectorPose.configure(
                entity_name=entity_name, ee_link_name=ee_link_name, rotation="quat", offset_pos=tuple(ee_offset_pos)
            ),
            gripper=GripperState.configure(
                entity_name=entity_name,
                dofs_name=[gripper_dof_name],
                normalize="openness",
                open_pos=open_gripper_pos,
                close_pos=closed_gripper_pos,
                clip=(0.0, 1.0),
            ),
        )
        return groups

    @classmethod
    def action_options(
        cls,
        entity_name: str = "robot",
        ee_link_name: str = "gripper_link",
        dofs_name: list[str] | None = None,
        scale: float = 0.5,
    ) -> dict[str, "ActionTermOptions"]:
        return make_widowx_ee_delta_actions(
            entity_name=entity_name,
            ee_link_name=ee_link_name,
            dofs_name=dofs_name if dofs_name is not None else list(WIDOWX_DOFS),
            scale=scale,
        )

    # -- packing -------------------------------------------------------

    def pack_observations(self, obs: dict) -> dict:
        p = self.policy
        image_uint8 = prepare_image_resized(obs["image"], p.image_size)  # (H, W, 3)

        proprio = to_numpy_unbatched(obs["proprio"])  # 8-D
        pos, quat_wxyz, gripper_openness = proprio[:3], proprio[3:7], float(proprio[7])
        rpy = bridge_rpy(quat_wxyz)
        # state order: x, y, z, roll, pitch, yaw, pad, gripper
        state_scalars = {
            "x": float(pos[0]),
            "y": float(pos[1]),
            "z": float(pos[2]),
            "roll": float(rpy[0]),
            "pitch": float(rpy[1]),
            "yaw": float(rpy[2]),
            "pad": 0.0,
            "gripper": gripper_openness,
        }
        state = {
            k: np.asarray([[[v]]], dtype=np.float32)  # (1, T=1, 1)
            for k, v in state_scalars.items()
        }

        video = {
            _WIDOWX_VIDEO_KEY: image_uint8[np.newaxis, np.newaxis],  # (1, T=1, H, W, 3)
        }
        return {"video": video, "state": state, "language": self.language()}


# ---------------------------------------------------------------------------
# libero_sim
# ---------------------------------------------------------------------------


_LIBERO_LANGUAGE_KEY = "annotation.human.action.task_description"
# The stock ``libero`` task's action terms (eden/tasks/libero/config.py), which this setup drives.
_LIBERO_ARM_TERM = "osc_pose"
_LIBERO_GRIPPER_TERM = "gripper"
#: Post-compute ``scale`` of the ``proprio`` group's ``gripper_qpos`` term: robosuite's ``robot0_gripper_qpos``
#: reports the second finger negative (``[+q1, -q2]``) where Eden's Panda reports both fingers positive.
_LIBERO_GRIPPER_QPOS_SCALE: tuple[float, float] = (1.0, -1.0)
#: LIBERO control rate: robosuite's ``control_freq=20`` default, which LIBERO's ``OffScreenRenderEnv`` keeps;
#: Eden's stock ``libero`` task steps 0.002 s x 25 = 0.05 s.
_LIBERO_CONTROL_HZ = 20.0


def check_libero_action_terms(env: "EnvBase") -> None:
    """Raise unless ``env`` carries the stock ``libero`` task's ``osc_pose`` / ``gripper`` terms exactly.

    Exact type, not ``isinstance``, and every option field, not only the action units: a subclass, another
    control point (``ee_local_pos``), gains or nullspace reference executes the same OSC_POSE row differently.
    """
    from eden.managers.terms.actions.task_space_actions import OperationalSpaceController
    from eden.tasks.libero.custom_terms.actions import LiberoPandaGripperController

    terms = env.action_manager.terms
    fix = (
        "Fix: build the env's action_options with "
        "ActionManagerOptions(**Gr00tPolicy.action_options(policy_setup='libero_sim')), which are the stock 'libero' "
        "task's osc_pose / gripper terms."
    )
    for name, expected in (
        (_LIBERO_ARM_TERM, OperationalSpaceController),
        (_LIBERO_GRIPPER_TERM, LiberoPandaGripperController),
    ):
        term = terms.get(name)
        if type(term) is not expected:
            found = "missing" if term is None else f"of type {type(term).__name__}"
            raise ValueError(
                f"Gr00tPolicy.policy_setup='libero_sim' drives action term {name!r} as a {expected.__name__} exactly, but the env's "
                f"{name!r} is {found} (action terms: {sorted(terms)}). {fix}"
            )
    for name, stock in libero_task_action_options().items():
        # JSON mode on both sides: a config saved and reloaded (eden inference replays the run's config.json)
        # carries ee_local_pos / ee_local_quat as lists where the builder has tuples, and enums as strings.
        env_fields = getattr(env.config.action_options, name).model_dump(mode="json")
        stock_fields = stock.model_dump(mode="json")
        changed = {f: (env_fields.get(f), v) for f, v in stock_fields.items() if env_fields.get(f) != v}
        if changed:
            details = ", ".join(
                f"{f}={env_value!r} (stock {stock_value!r})" for f, (env_value, stock_value) in changed.items()
            )
            raise ValueError(
                f"Action term {name!r} differs from the stock 'libero' task's: {details}. GR00T's LIBERO checkpoints "
                f"emit robosuite OSC_POSE rows recorded under exactly that controller (action units, control point, "
                f"gains), so every field must stay at the stock value. {fix} Do not rescale the rows in the client "
                "instead: the server's action units are fixed by the dataset."
            )


class LiberoSetup(SetupBase):
    """LIBERO Panda embodiment for GR00T, as Isaac-GR00T's ``gr00t/eval/sim/LIBERO/libero_env.py`` runs it.

    * State: robosuite's ``robot0_eef_pos`` (the grip site, world frame), the axis-angle of
      ``robot0_eef_quat`` (robosuite 1.4 reads it off the ``right_hand`` *body*, which is Eden's
      ``hand`` link, not the grip site) and ``robot0_gripper_qpos``, whose second finger is
      negative (``[+q1, -q2]``; Eden's Panda reports both fingers positive).
    * Actions: 7-D robosuite ``OSC_POSE`` inputs ``[dx, dy, dz, drx, dry, drz, gripper]`` executed by
      the stock ``libero`` task's ``osc_pose`` and ``gripper`` terms. GR00T's gripper is the dataset's
      ``[0, 1]`` (1 = open); like libero_env.py it becomes robosuite's ``-sign(2g - 1)`` (-1 opens,
      +1 closes, 0.5 holds).
    * Images: both cameras mirrored left-right (:func:`libero_training_view`) before the resize; the cameras
      must render at :data:`~eden.extensions.policies.adapters.LIBERO_CAMERA_RESOLUTION`, as upstream's eval does.
    * Rate: 20 Hz, robosuite's ``control_freq``.

    Upstream's LIBERO README launches this server with ``--use-sim-policy-wrapper``.
    """

    # The served LIBERO_PANDA modality config's action delta_indices length (checked against the server on connect).
    pred_action_horizon: int = 16
    # examples/LIBERO/README.md runs the LIBERO rollout with --n-action-steps 8.
    exec_horizon: int = 8
    default_image_size: tuple[int, int] = (256, 256)
    action_keys: tuple[str, ...] = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
    action_dim: int = 7
    video_keys: tuple[str, ...] = ("image", "wrist_image")
    state_keys: tuple[str, ...] = ("x", "y", "z", "roll", "pitch", "yaw", "gripper")
    language_key: str = _LIBERO_LANGUAGE_KEY
    upstream_sim_policy_wrapper: ClassVar[bool] = True
    control_hz: ClassVar[float] = _LIBERO_CONTROL_HZ
    control_hz_source: ClassVar[str] = "Isaac-GR00T's LIBERO eval steps robosuite (its default control_freq=20)"
    # [grip_site_pos_w(3), hand_quat_wxyz(4), q_left, -q_right]
    observation_groups: ClassVar[dict[str, tuple[int, ...] | None]] = {
        "image": None,
        "wrist_image": None,
        "proprio": (9,),
    }

    def __init__(self, policy: "Gr00tPolicy") -> None:
        super().__init__(policy)
        check_libero_action_terms(policy.env)

    def check_env_observations(self, env: "EnvBase") -> None:
        """The base check, ``proprio`` as the plain current reading with robosuite's finger sign flip, then the image
        groups' cameras at upstream's 256x256 (an upsampled render is off-distribution).
        """
        super().check_env_observations(env)
        # pack_observations slices proprio by column, and the [+q1, -q2] gripper layout lives only in the
        # gripper_qpos term's scale: a same-width group without it (pi's LIBERO proprio negates q2 in the
        # client instead) would send a gripper state outside the training distribution.
        check_plain_observation_group(
            env,
            "proprio",
            scales={"gripper_qpos": _LIBERO_GRIPPER_QPOS_SCALE},
            fix="build 'proprio' with Gr00tPolicy.observation_options(policy_setup='libero_sim')['proprio'] as "
            "returned, adding no history_length, scale, clip, noise or modifiers and keeping gripper_qpos's "
            f"scale={list(_LIBERO_GRIPPER_QPOS_SCALE)}. Do not negate the second finger in pack_observations: the "
            "group already sends robosuite's [+q1, -q2].",
        )
        check_libero_camera_resolution(
            env,
            subject="Gr00tPolicy.policy_setup='libero_sim'",
            trained_on="GR00T's LIBERO checkpoints were trained on 256x256 renders (gr00t/eval/sim/LIBERO/libero_env.py: "
            "camera_heights=256, camera_widths=256)",
            wrong_fix="Do not rely on Gr00tPolicy.image_size instead: resizing a smaller render up to 256x256 does not "
            "restore the detail it lacks.",
        )

    @classmethod
    def observation_options(
        cls,
        camera_name: str = "agentview",
        wrist_camera_name: str = "eye_in_hand",
    ) -> dict[str, "ObservationGroupOptions"]:
        """``image`` / ``wrist_image`` (the stock task's camera names) and robosuite's proprio on Eden's LIBERO
        Panda.
        """
        from eden.managers.terms.observations.manipulation import EndEffectorPose
        from eden.managers.terms.observations.proprio import DofsPos
        from eden.options.managers.observations import ObservationGroupOptions

        groups = make_image_groups(camera_name, use_wrist_image=True, wrist_camera_name=wrist_camera_name)
        # [grip_site_pos_w(3), hand_quat_w(4) wxyz, q_left, -q_right]: robosuite's eef_pos / eef_quat /
        # gripper_qpos. Only the position moves to the grip site; robosuite's eef_quat is the hand body's.
        groups["proprio"] = ObservationGroupOptions(
            ee_pose=EndEffectorPose.configure(
                entity_name=LIBERO_ROBOT_ENTITY_NAME,
                ee_link_name=LiberoPanda.ee_links_name[0],
                frame="world",
                offset_pos=LiberoPanda.grip_site_pos_local,
                rotation="quat",
            ),
            # The second finger's sign flip is the manager's post-compute ``scale``. Not GripperState(scale=...):
            # that term also applies its own ``scale`` field, so a -1 is applied twice and cancels.
            gripper_qpos=DofsPos.configure(
                entity_name=LIBERO_ROBOT_ENTITY_NAME,
                dofs_name=list(LiberoPanda.gripper_dofs_name),
                scale=list(_LIBERO_GRIPPER_QPOS_SCALE),
            ),
        )
        return groups

    @classmethod
    def action_options(cls) -> dict[str, "ActionTermOptions"]:
        """The stock ``libero`` task's ``osc_pose`` / ``gripper`` terms (:func:`libero_task_action_options`)."""
        return libero_task_action_options()

    def pack_observations(self, obs: dict) -> dict:
        p = self.policy
        image = prepare_image_resized(libero_training_view(obs["image"]), p.image_size)
        wrist_image = prepare_image_resized(libero_training_view(obs["wrist_image"]), p.image_size)

        # (9,) [grip_site_pos_w(3), hand_quat_wxyz(4), q_left, -q_right], checked at construction.
        proprio = to_numpy_unbatched(obs["proprio"]).astype(np.float32)
        pos = proprio[:3]
        # robosuite's quat2axisangle, as Isaac-GR00T's LIBERO env computes the state (not as_rotvec: its
        # angle is canonicalised to [0, pi], which differs whenever w < 0).
        axis_angle = robosuite_axis_angle(proprio[3:7]).astype(np.float32)
        gripper = proprio[7:9]

        state_values = {
            "x": pos[0:1],
            "y": pos[1:2],
            "z": pos[2:3],
            "roll": axis_angle[0:1],
            "pitch": axis_angle[1:2],
            "yaw": axis_angle[2:3],
            "gripper": gripper,
        }
        state = {key: value.reshape(1, 1, -1) for key, value in state_values.items()}
        video = {
            "image": image[np.newaxis, np.newaxis],
            "wrist_image": wrist_image[np.newaxis, np.newaxis],
        }
        return {"video": video, "state": state, "language": self.language()}

    def unpack_actions(self, raw_action: np.ndarray) -> dict[str, torch.Tensor]:
        row = np.asarray(raw_action, dtype=np.float32)  # (7,): action_dim is checked on every chunk
        env = self.policy.env
        # libero_env.py: normalize_gripper_action (2g - 1, then sign) and invert_gripper_action (negate).
        gripper = -np.sign(2.0 * row[6:7] - 1.0)
        return {
            _LIBERO_ARM_TERM: broadcast_to_envs(torch.tensor(row[:6], device=env.device), env.num_envs),
            _LIBERO_GRIPPER_TERM: broadcast_to_envs(torch.tensor(gripper, device=env.device), env.num_envs),
        }


# ---------------------------------------------------------------------------
# oxe_droid_relative_eef_relative_joint
# ---------------------------------------------------------------------------


_OXE_DROID_LANGUAGE_KEY = "annotation.language.language_instruction"

# DROID end-effector frame correction baked into the OXE-DROID training data, ``DROID_EEF_ROTATION_CORRECT``
# of Isaac-GR00T's ``examples/DROID/main_gr00t.py``: the model expects ``R_corrected = R_flange @ CORRECT``.
_DROID_EEF_ROTATION_CORRECT_QUAT = R_to_quat(torch.tensor([[[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]]))


def droid_eef_correction_quat(
    tool_frame_offset_euler: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> tuple[float, float, float, float]:
    """Wxyz quaternion of ``TOOL_OFFSET @ DROID_CORRECT`` for ``EndEffectorPose.frame_correction_quat``.

    ``rot6d`` of ``R_eden @ TOOL_OFFSET @ DROID_CORRECT`` is what the OXE-DROID checkpoint was trained on, with
    ``R_eden @ TOOL_OFFSET`` the DROID flange rotation and ``TOOL_OFFSET`` extrinsic xyz Euler angles (TFG
    ``from_euler``, scipy ``"xyz"``).

    Isaac-GR00T's own DROID client (``examples/DROID/main_gr00t.py`` lines 57-68, copying
    ``gr00t/data/state_action/droid_frame.py`` line 57, at 944fe70) rebuilds this rotation from DROID's Euler
    angles with scipy's intrinsic ``"XYZ"``, a different rotation for the same angles, although its docstring
    (lines 45-46) calls that equivalent to TFG ``from_euler``. The checkpoint's statistics follow ``"xyz"``:
    ``nvidia/GR00T-N1.7-3B`` ``oxe_droid_relative_eef_relative_joint.state.eef_9d`` has rot6d mean
    ``[0.059, 0.112, -0.717, 0.732, -0.026, 0.051]``, and rebuilding ``lerobot/droid_1.0.1``
    ``data/chunk-000/file-000.parquet`` (1074 episodes) gives a rot6d mean / std / q01 / q99 RMSE of
    0.014 / 0.012 / 0.031 / 0.036 under ``"xyz"`` against 0.094 / 0.014 / 0.160 / 0.061 under ``"XYZ"``.
    """
    offset_quat = quat_from_euler_xyz(*torch.tensor([tool_frame_offset_euler], dtype=torch.float32).T)
    return tuple(float(v) for v in quat_mul(offset_quat, _DROID_EEF_ROTATION_CORRECT_QUAT)[0])


class OxeDroidSetup(SetupBase):
    """OXE-DROID (relative EEF + relative joint) embodiment.

    State sent to the server:

    * ``eef_9d`` — current EE pose ``[xyz, rot6d]`` (9-D) in the
      DROID-corrected frame (``EndEffectorPose`` with ``droid_eef_correction_quat``).
    * ``gripper_position`` — scalar in ``[0, 1]`` (**1 = closed**) broadcast from
      the gripper DOF position, matching DROID's robot-state convention.
    * ``joint_position`` — 7 arm DOF positions (radians).

    Action streams returned by the server:

    * ``eef_9d`` (relative) — *unused on output* (state-only here; switch to
      this stream if the env's arm controller is task-space).
    * ``gripper_position`` (absolute closedness) — forwarded raw to the env's
      ``GripperDecisionController``, which closes on a strict ``> 0.5`` per row.
    * ``joint_position`` — the server converts the relative prediction to absolute
      joint targets against the state it was sent, so the client forwards them
      unchanged to an ``ImplicitPDController`` with a zero reference
      (:class:`~eden.extensions.policies.adapters.JointPositionAdapter`; its ``last_diagnostics``
      describe the executed row).

    Images are letterboxed to ``(180, 320)`` with ``resize_with_pad`` exactly as
    NVIDIA's DROID client does (``examples/DROID/main_gr00t.py`` ``RESOLUTION``);
    the server's eval transform pads that to a square, as in training.

    Rate: 15 Hz, ``DROID_CONTROL_FREQUENCY`` of ``main_gr00t.py`` (DROID's data-collection rate).
    """

    pred_action_horizon: int = 40  # delta_indices=list(range(40))
    # NVIDIA's DROID client re-queries every open_loop_horizon=15 steps (examples/DROID/main_gr00t.py).
    exec_horizon: int = 15
    # Order matters: decode_response concatenates the streams in this order.
    # eef_9d (9) + gripper_position (1) + joint_position (7) = 17 columns.
    action_keys: tuple[str, ...] = ("eef_9d", "gripper_position", "joint_position")
    action_dim: int = 17
    video_keys: tuple[str, ...] = ("exterior_image_1_left", "wrist_image_left")
    state_keys: tuple[str, ...] = ("eef_9d", "gripper_position", "joint_position")
    language_key: str = _OXE_DROID_LANGUAGE_KEY
    # examples/DROID/main_gr00t.py DROID_CONTROL_FREQUENCY.
    control_hz: ClassVar[float] = DROID_CONTROL_HZ
    control_hz_source: ClassVar[str] = (
        "examples/DROID/main_gr00t.py steps DROID's RobotEnv (DROID_CONTROL_FREQUENCY = 15)"
    )
    # eef_9d = [xyz, rot6d]; joint_state = [q(7), gripper_closedness]
    observation_groups: ClassVar[dict[str, tuple[int, ...] | None]] = {
        "image": None,
        "wrist_image": None,
        "eef_9d": (9,),
        "joint_state": (8,),
    }
    policy_fields: ClassVar[frozenset[str]] = frozenset({"video_horizon", "video_stride", "debug_gripper"})

    # Isaac-GR00T's DROID client letterboxes every camera to (180, 320) with
    # resize_with_pad (examples/DROID/main_gr00t.py RESOLUTION, utils.py).
    default_image_size: tuple[int, int] = (180, 320)

    # Number of video frames to send per modality key. Must match the running
    # server's modality config (``len(video.delta_indices)``). The stock OXE-DROID
    # pretrain uses ``delta_indices=[-15, 0]`` (T=2). Override
    # ``Gr00tPolicy.video_horizon`` to 1 if your fine-tune collapses to a single
    # frame.
    video_horizon: int = 2
    # Stride (in control steps) between successive video frames when
    # ``video_horizon > 1``. Mirrors negative ``delta_indices`` like
    # ``[-15, 0]`` (=> stride 15). Frames are recorded every control step in
    # :meth:`observe`, as the reference client fills its frame buffer.
    video_stride: int = 15

    # Gripper: NVIDIA's DROID client (``examples/DROID/main_gr00t.py``) binarizes each
    # executed action's ``gripper_position`` with a strict ``> 0.5``. The chunk is forwarded
    # raw and the env's GripperDecisionController applies exactly that rule.

    def __init__(self, policy: "Gr00tPolicy") -> None:
        from eden.managers.terms.actions.joint_actions import ImplicitPDController

        super().__init__(policy)
        # The policy's video_horizon / video_stride override the checkpoint defaults when set.
        if policy.video_horizon is not None:
            self.video_horizon = int(policy.video_horizon)
        if policy.video_stride is not None:
            self.video_stride = int(policy.video_stride)
        self.debug_gripper = bool(policy.debug_gripper)
        self.last_input_diagnostics: dict[str, float] | None = None
        # Absolute joint targets + raw closedness, labelled "close" above NVIDIA's strict 0.5 in the diagnostics.
        self.adapter = JointPositionAdapter(
            n_arm=len(DROID_ARM_DOFS), gripper_threshold=0.5, debug_gripper=self.debug_gripper
        )
        # joint_position is RELATIVE in the checkpoint's action config, and the server's
        # StateActionProcessor.unapply_action adds it to the sent joint_position state before replying
        # (gr00t/data/state_action/state_action_processor.py); main_gr00t.py then steps
        # RobotEnv(action_space="joint_position") with the reply as is. So the term executes absolute targets as is.
        arm = _require_arm_term(
            policy.env,
            policy.policy_setup,
            ImplicitPDController,
            why="the server replies absolute joint targets",
            wrong_fix="Do not add the joint state to the targets in the client: the server already did.",
        )
        check_droid_arm_term(
            arm,
            term_name=_ARM_TERM,
            subject=f"Gr00tPolicy.policy_setup={policy.policy_setup!r}",
            fix=f"ActionManagerOptions(**Gr00tPolicy.action_options(policy_setup={policy.policy_setup!r})), passing "
            "the rig's entity / gripper names as keyword arguments",
        )
        # Closedness is forwarded raw; main_gr00t.py's strict ``> 0.5`` per row (no hysteresis, hold or continuous
        # mode) is what action_options()'s term decides.
        _check_gripper_term_decides_as_built(type(self), policy.env)
        # Per-key rolling history of the raw uint8 renders, one frame per control step (see observe()).
        self._history: dict[str, FrameHistory] = {
            key: FrameHistory(self.video_horizon, self.video_stride)
            for key in ("exterior_image_1_left", "wrist_image_left")
        }

    def reset(self) -> None:
        super().reset()
        for history in self._history.values():
            history.reset()
        self.last_input_diagnostics = None

    def check_env_observations(self, env: "EnvBase") -> None:
        """The base check, then the two state groups as the plain current reading of the arm and gripper the action
        terms drive: ``joint_state`` reads ``ee_controller``'s joints in its order, and its closedness spans
        ``gripper_controller``'s stroke.
        """
        super().check_env_observations(env)
        setup = self.policy.policy_setup
        for group in ("eef_9d", "joint_state"):
            check_plain_observation_group(
                env,
                group,
                fix=f"build {group!r} with Gr00tPolicy.observation_options(policy_setup={setup!r})[{group!r}] as "
                "returned, adding no history_length, scale, clip, noise or modifiers. Do not undo the transform in "
                "pack_observations: the server expects DROID's raw robot state.",
            )
        arm = env.action_manager.terms[_ARM_TERM]
        check_droid_joint_state(
            env,
            arm,
            group="joint_state",
            subject=f"Gr00tPolicy.policy_setup={setup!r}",
            fix=f"build it from Gr00tPolicy.observation_options(policy_setup={setup!r}, entity_name={arm.entity_name!r}, "
            f"arm_dofs_name={list(arm.dofs_name)!r}) and pass the same entity_name / arm_dofs_name to "
            "Gr00tPolicy.action_options(...)",
        )
        # The state's closedness and the command's targets come from separate observation_options() /
        # action_options() arguments (gripper_entity_name, gripper_close_pos): they must describe one gripper.
        check_gripper_state_matches_action(env, group="joint_state", term="gripper", action_term="gripper_controller")

    @classmethod
    def observation_options(
        cls,
        camera_name: str = "camera",
        wrist_camera_name: str = "wrist_camera",
        entity_name: str = "robot",
        ee_link_name: str = "attachment",
        arm_dofs_name: list[str] | None = None,
        gripper_dof_name: str = "left_driver_joint",
        gripper_entity_name: str | None = None,
        gripper_close_pos: float = 0.8,
        eef_tool_frame_offset_euler: tuple[float, float, float] = DROID_FLANGE_FROM_FR3_ATTACHMENT_EULER,
    ) -> dict[str, "ObservationGroupOptions"]:
        """Camera groups plus ``eef_9d`` and ``joint_state``.

        ``eef_9d`` must be the pose of DROID's ``cartesian_position`` frame, the Panda flange
        (``panda_link8``: 0.107 m along link7, no rotation). The defaults give exactly that on
        Eden's FR3 (``Droid`` robot): its ``attachment`` link sits at the flange origin rotated
        +135 deg about z, so ``eef_tool_frame_offset_euler = (0, 0, -3*pi/4)`` undoes it. Pass a
        different link/offset for another arm asset.
        """
        from eden.managers.terms.observations.manipulation import EndEffectorPose
        from eden.options.managers.observations import ObservationGroupOptions

        if arm_dofs_name is None:
            arm_dofs_name = list(DROID_ARM_DOFS)

        groups = make_image_groups(camera_name, use_wrist_image=True, wrist_camera_name=wrist_camera_name)

        # eef_9d: [xyz, rot6d] of R_eden @ TOOL_OFFSET @ DROID_CORRECT (Isaac-GR00T compute_eef_9d).
        groups["eef_9d"] = ObservationGroupOptions(
            pose=EndEffectorPose.configure(
                entity_name=entity_name,
                ee_link_name=ee_link_name,
                rotation="rot6d",
                frame_correction_quat=droid_eef_correction_quat(tuple(eef_tool_frame_offset_euler)),
            ),
        )
        # joint_position + gripper_position (closedness, matching DROID
        # ``robot_state["gripper_position"]`` where 1 = closed).
        groups["joint_state"] = make_droid_joint_state_group(
            entity_name,
            arm_dofs_name,
            gripper_entity_name=gripper_entity_name,
            gripper_dof_name=gripper_dof_name,
            gripper_close_pos=gripper_close_pos,
        )
        return groups

    @classmethod
    def action_options(
        cls,
        entity_name: str = "robot",
        arm_dofs_name: list[str] | None = None,
        gripper_dof_name: str = "left_driver_joint",
        gripper_entity_name: str | None = None,
        gripper_close_pos: float = 0.8,
    ) -> dict[str, "ActionTermOptions"]:
        # Raw closedness in [0, 1]; the GripperDecisionController binarizes at 0.5 as NVIDIA's client does.
        return make_droid_joint_position_actions(
            entity_name,
            arm_dofs_name,
            gripper_entity_name=gripper_entity_name,
            gripper_dof_name=gripper_dof_name,
            gripper_close_pos=gripper_close_pos,
            threshold=0.5,
        )

    # -- packing -------------------------------------------------------

    @staticmethod
    def _raw_frames(obs: dict) -> dict[str, np.ndarray]:
        """The two renders as ``(H, W, 3)`` uint8, per video key (both groups are checked at construction)."""
        return {
            "exterior_image_1_left": to_uint8_hwc(obs["image"]),
            "wrist_image_left": to_uint8_hwc(obs["wrist_image"]),
        }

    def observe(self, obs: dict) -> None:
        """Record one raw frame per control step (the reference ``frame_buffer`` appends every step).

        ``video.delta_indices=[-15, 0]`` are control-step offsets; recording only when the
        chunk drains would stretch the window by ``exec_horizon``. Frames are letterboxed in
        :meth:`_video`, only the ``video_horizon`` a query sends. The trade: the history holds
        ``(video_horizon - 1) * video_stride + 1`` raw renders per camera (16 at the defaults, about
        45 MB per 1280x720 camera) instead of 180x320 letterboxed ones, to skip two resizes per step.
        """
        if self.video_horizon <= 1:
            return
        for key, frame in self._raw_frames(obs).items():
            self._history[key].append(frame)

    def _video(self, obs: dict) -> dict[str, np.ndarray]:
        """``(T, H, W, 3)`` letterboxed frames per video key: the current frame, or the strided window
        :meth:`observe` recorded. Missing history is padded with the oldest frame.
        """
        if self.video_horizon <= 1:
            frames = {key: frame[np.newaxis] for key, frame in self._raw_frames(obs).items()}
        else:
            frames = {key: history.window() for key, history in self._history.items()}
        size = self.policy.image_size
        return {
            key: np.stack([prepare_image_padded(frame, size) for frame in window]) for key, window in frames.items()
        }

    def pack_observations(self, obs: dict) -> dict:
        video = self._video(obs)
        ext_seq = video["exterior_image_1_left"][np.newaxis]  # (1, T, H, W, 3)
        wrist_seq = video["wrist_image_left"][np.newaxis]

        eef_9d = to_numpy_unbatched(obs["eef_9d"]).astype(np.float32)  # (9,)
        joint_state = to_numpy_unbatched(obs["joint_state"]).astype(np.float32)  # (8,)
        joint_pos = joint_state[:7]
        gripper_closedness = float(joint_state[7])  # 1 = closed (DROID convention)
        self.last_input_diagnostics = {
            "state_gripper_closedness": gripper_closedness,
            "eef_x": float(eef_9d[0]),
            "eef_y": float(eef_9d[1]),
            "eef_z": float(eef_9d[2]),
            "joint_min": float(np.min(joint_pos)),
            "joint_max": float(np.max(joint_pos)),
        }
        if self.debug_gripper:
            en.logger.info(
                "[gr00t droid input] "
                f"state_grip_closed={gripper_closedness:.4f} "
                f"eef_xyz=({eef_9d[0]:.3f},{eef_9d[1]:.3f},{eef_9d[2]:.3f}) "
                f"joint_range=({np.min(joint_pos):.3f},{np.max(joint_pos):.3f})"
            )

        state = {
            "eef_9d": eef_9d.reshape(1, 1, 9),
            "gripper_position": np.asarray([[[gripper_closedness]]], dtype=np.float32),
            "joint_position": joint_pos.reshape(1, 1, 7),
        }
        video = {
            "exterior_image_1_left": ext_seq,
            "wrist_image_left": wrist_seq,
        }
        return {"video": video, "state": state, "language": self.language()}

    def unpack_actions(self, raw_action: np.ndarray) -> dict[str, torch.Tensor]:
        """``[eef_9d(9), gripper_position(1), joint_position(7)]`` -> joint targets + raw closedness.

        Isaac-GR00T's server-side ``_decode_action`` already turns the RELATIVE prediction into
        ABSOLUTE targets against the sent state (``state_action_processor.py``
        ``_convert_to_absolute_action``), so ``joint_position`` must not get ``current_q`` added again.
        Each executed row carries its own closedness (``main_gr00t.py``: no chunk averaging or latching).
        """
        row = np.asarray(raw_action, dtype=np.float32)
        return self.adapter(np.concatenate([row[10:17], row[9:10]]), env=self.policy.env)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


SETUP_REGISTRY: dict[str, type[SetupBase]] = {
    "simpler_env_widowx": WidowxSimplerSetup,
    "libero_sim": LiberoSetup,
    "oxe_droid_relative_eef_relative_joint": OxeDroidSetup,
}
