"""Cosmos3Policy — NVIDIA Cosmos3-Nano-Policy-DROID remote inference client.

``Cosmos3Policy`` is a thin WebSocket client (``transport="openpi_websocket"``)
for the Cosmos-Framework RoboLab action policy server
(``cosmos_framework.scripts.action_policy_server_robolab``). That server speaks
OpenPI's msgpack+NumPy WebSocket protocol via ``WebsocketPolicyServer``:

* on connection it sends an (empty) metadata dict;
* each client message is an observation dict;
* each response is ``{"action": ndarray, ...}`` — note the key is ``action``
  (singular), unlike OpenPI's pi0 servers which return ``actions``.

This integration targets the released **Cosmos3-Nano-Policy-DROID** checkpoint,
which serves the DROID Franka embodiment in the ``joint_pos`` action space:

* **Observation** — language ``prompt``, ``observation/joint_position`` (7),
  ``observation/gripper_position`` (1, closedness), plus camera images.
* **Action** — chunk ``(T, 8)`` = 7 absolute arm joint targets + 1 gripper
  scalar.  The model predicts absolute joint targets; the server drops the
  conditioning-history rows and re-applies its ``1 - x`` gripper convention, so
  the gripper scalar the client receives is *closedness* (``> 0.5`` ⇒ close),
  matching the closedness we send in ``observation/gripper_position``.

Camera payload
--------------
The released checkpoint is trained on a RoboArena "concat view": the wrist camera on top and
two third-person exterior views, from opposite sides of the robot, side by side on the bottom.
The client sends the three views under the RoboArena keys
(``observation/wrist_image_left``, ``observation/exterior_image_1_left``,
``observation/exterior_image_2_left``) and the server composes them. These keys are the only
camera input this client sends: the server reads ``observation/image`` as an *already composed*
concat view (it resizes it to 540x640 and describes its top row as the wrist camera), so a lone
exterior frame under that key would be misread.

The second exterior view is an explicit choice, checked when the policy is constructed:

* ``secondary_camera_name="<camera>"`` renders it from a second exterior camera (the
  ``image_2`` observation group; pass the same name to :meth:`Cosmos3Policy.observation_options`).
  This is what upstream RoboLab does — it always sends two distinct over-shoulder cameras.
* ``duplicate_exterior_view=True`` repeats the primary exterior into the second slot, for a
  scene with a single exterior camera (e.g. ``robotarena360``). Upstream never does this, so the
  composed image is off the checkpoint's training distribution.

The env must step at DROID's 15 Hz (``env.dt == 1/15`` s): every row of the 32-row chunk is a
joint waypoint 1/15 s after the previous one, and the policy refuses another control period at
construction, along with an arm term, camera groups or joint state other than its own
:meth:`~Cosmos3Policy.action_options` / :meth:`~Cosmos3Policy.observation_options` build.

Action unpacking reuses :class:`~eden.extensions.policies.adapters.JointPositionAdapter`
and observation wiring reuses the shared DROID joint-state group — the
DROID joint-position action/observation contract is identical to OpenPI's
``droid_jointpos`` setup. The rate, arm-term and joint-state refusals are the shared
:func:`~eden.extensions.policies.adapters.check_control_rate`,
:func:`~eden.extensions.policies.adapters.check_droid_arm_term` and
:func:`~eden.extensions.policies.adapters.check_droid_joint_state` that the pi and GR00T
DROID setups run.
"""

from __future__ import annotations

from typing import ClassVar, TYPE_CHECKING

import numpy as np
import torch

from eden.extensions.policies.base import POLICY_REGISTRY, PolicyBase
from eden.extensions.policies.adapters import (
    DROID_ARM_DOFS,
    DROID_CONTROL_HZ,
    DROID_CONTROL_TIMING,
    DROID_CONTROL_TIMING_NOTE,
    JointPositionAdapter,
    check_control_rate,
    check_droid_arm_term,
    check_droid_joint_state,
    check_env_gripper_term,
    check_gripper_state_matches_action,
    check_plain_observation_group,
    make_droid_joint_position_actions,
    make_droid_joint_state_group,
    make_image_groups,
)
from eden.extensions.policies.client import ClientMixin
from eden.utils.image import prepare_image_padded
from eden.utils.torch import to_numpy_unbatched
from eden.options.extensions.policies import RemotePolicyOptions
from eden.utils.common import ConfigurableMixin

if TYPE_CHECKING:
    from eden.managers.terms.actions.joint_actions import ImplicitPDController
    from eden.options.managers.actions import ActionTermOptions
    from eden.options.managers.observations import ObservationGroupOptions

#: The env's absolute-joint arm term, as :meth:`Cosmos3Policy.action_options` names it.
_ARM_TERM = "ee_controller"
#: The client every construction refusal names.
_SUBJECT = "Cosmos3Policy"


@POLICY_REGISTRY.register()
class Cosmos3Policy(ClientMixin, PolicyBase, ConfigurableMixin[RemotePolicyOptions]):
    """Cosmos3-Nano-Policy-DROID remote inference client (WebSocket).

    All class attributes are configurable via ``Cosmos3Policy.configure(...)``.
    """

    action_dim: ClassVar[int | None] = 8  # ``[joint_position (7), gripper (1)]``

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    host: str = "localhost"
    port: int | None = 8000
    api_key: str | None = None
    # Set to None to disable WebSocket keepalive pings (recommended — Cosmos3
    # diffusion inference is slow and would otherwise trip ping timeouts).
    ws_ping_timeout: float | None = None
    transport: str = "openpi_websocket"
    # The RoboLab server returns the chunk under "action" (singular).
    response_action_key: str = "action"

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    # The server returns a 32-step action chunk by default.  We execute
    # ``exec_horizon`` of those open-loop before re-querying.  The official
    # RoboLab client (NVlabs/RoboLab policies/cosmos3/client.py) sets
    # OPEN_LOOP_HORIZON = 32, i.e. it executes the FULL chunk (~2.1s at 15Hz)
    # before re-querying -- the released benchmark numbers were produced that
    # way, so these defaults match it.  Shorter horizons re-plan mid-chunk,
    # which is off-distribution for this diffusion policy.
    pred_action_horizon: int | None = 32
    exec_horizon: int | None = 32

    # ------------------------------------------------------------------
    # Image preprocessing
    # ------------------------------------------------------------------

    # Per-view size before server-side RoboArena concat composition.  The
    # RoboLab DROID server stacks wrist (top) with two half-size exterior views
    # (bottom), so 360x640 views compose to the released server's 540x640
    # default input geometry.
    image_size: tuple[int, int] = (360, 640)

    # ------------------------------------------------------------------
    # Second exterior view (exactly one of the two; see the module docstring)
    # ------------------------------------------------------------------

    # The camera the env's ``image_2`` observation group renders; sent as
    # ``exterior_image_2_left``.
    secondary_camera_name: str | None = None
    # Repeat the primary exterior into ``exterior_image_2_left`` instead (single-exterior scenes;
    # off-distribution for the checkpoint).
    duplicate_exterior_view: bool = False

    # ------------------------------------------------------------------
    # Gripper decision (checked against the env's GripperDecisionController)
    # ------------------------------------------------------------------

    gripper_threshold: float = 0.5
    debug_gripper: bool = False

    # ------------------------------------------------------------------

    def _post_configure(self) -> None:
        self._check_second_exterior_choice()
        check_control_rate(
            self.env,
            control_hz=DROID_CONTROL_HZ,
            subject=_SUBJECT,
            source="DROID's RobotEnv does (control_hz=15) in the data Cosmos3-Nano-Policy-DROID was trained on",
            timing=DROID_CONTROL_TIMING,
            timing_note=DROID_CONTROL_TIMING_NOTE,
        )
        arm = self._check_arm_term()
        self._check_env_observation_groups(arm)
        self._adapter = JointPositionAdapter(
            n_arm=7,
            arm_key=_ARM_TERM,
            gripper_threshold=self.gripper_threshold,
            debug_gripper=self.debug_gripper,
        )
        check_env_gripper_term(
            self.env,
            "gripper_controller",
            input="closedness",
            continuous=False,
            threshold=float(self.gripper_threshold),
        )
        # observation_options(gripper_close_pos=...) and action_options(gripper_close_pos=...) are separate
        # arguments: a stroke set in one only would skew the closedness sent with every request.
        check_gripper_state_matches_action(self.env, group="proprio", term="gripper", action_term="gripper_controller")

    def _check_second_exterior_choice(self) -> None:
        """Require exactly one source for ``exterior_image_2_left`` (see the module docstring)."""
        if self.secondary_camera_name is not None and self.duplicate_exterior_view:
            raise ValueError(
                f"Cosmos3Policy got both secondary_camera_name={self.secondary_camera_name!r} and "
                "duplicate_exterior_view=True; they are two sources for the same exterior_image_2_left slot. "
                "Drop duplicate_exterior_view when the scene has a second exterior camera (what the checkpoint was "
                "trained on), or set secondary_camera_name=None to duplicate the primary exterior."
            )
        if self.secondary_camera_name is None and not self.duplicate_exterior_view:
            raise ValueError(
                "Cosmos3Policy needs a second exterior view: Cosmos3-Nano-Policy-DROID was trained on two "
                "third-person cameras from opposite sides of the robot, and upstream RoboLab always sends two "
                "distinct ones. Fix: Cosmos3Policy.configure(secondary_camera_name='<camera>') with the same "
                "secondary_camera_name passed to Cosmos3Policy.observation_options(); or, for a scene with a "
                "single exterior camera, Cosmos3Policy.configure(duplicate_exterior_view=True) to repeat it "
                "(off-distribution for the checkpoint)."
            )

    def _check_arm_term(self) -> "ImplicitPDController":
        """The env's ``ee_controller``, refused unless it applies the server's absolute DROID joint targets as they are."""
        from eden.managers.terms.actions.joint_actions import ImplicitPDController

        terms = self.env.action_manager.terms
        term = terms.get(_ARM_TERM)
        fix = "Cosmos3Policy.action_options(), passing the rig's arm / gripper names as keyword arguments"
        if type(term) is not ImplicitPDController:
            found = "missing" if term is None else f"of type {type(term).__name__}"
            raise ValueError(
                f"{_SUBJECT} sends absolute DROID joint targets to action term {_ARM_TERM!r}, which must be exactly an "
                f"ImplicitPDController (a subclass reinterprets the same row), but the env's {_ARM_TERM!r} is {found} "
                f"(action terms: {sorted(terms)}). Fix: build the env's action_options from {fix}."
            )
        check_droid_arm_term(term, term_name=_ARM_TERM, subject=_SUBJECT, fix=fix)
        return term

    def _check_env_observation_groups(self, arm: "ImplicitPDController") -> None:
        """Refuse an env whose observation groups were declared for another setup of this policy.

        ``observation_options`` takes ``secondary_camera_name`` (and the arm DOFs) separately from
        ``configure``, so the two can disagree: an env without ``image_2`` under a two-exterior
        policy would raise at the first inference, an ``image_2`` rendering another camera than
        the policy names (or the primary exterior again) would silently send the wrong view, and
        an env with ``image_2`` under a duplicating policy would render a camera nobody sends.
        """
        group_dims = self.env.observation_manager.group_obs_dim
        published = sorted(group_dims)
        required = {"image", "wrist_image", "proprio"}
        if self.secondary_camera_name is not None:
            required.add("image_2")
        missing = sorted(required - set(published))
        if missing:
            raise ValueError(
                f"Cosmos3Policy(secondary_camera_name={self.secondary_camera_name!r}) reads the observation "
                f"group(s) {missing}, but this env publishes {published}. Build the env from "
                f"Cosmos3Policy.observation_options(secondary_camera_name={self.secondary_camera_name!r}), "
                "which declares the exterior 'image', 'wrist_image' and 'proprio' groups, and 'image_2' for "
                "the second exterior camera."
            )
        self._check_observation_terms(sorted(required), arm)

        cameras = self._rendered_cameras()
        if self.secondary_camera_name is None:
            if "image_2" in published:
                raise ValueError(
                    f"This env renders a second exterior camera, {cameras.get('image_2')!r}, as its 'image_2' "
                    "observation group, but Cosmos3Policy has secondary_camera_name=None and "
                    "duplicate_exterior_view=True, so it would send the primary exterior twice and never read "
                    f"'image_2'. Fix: Cosmos3Policy.configure(secondary_camera_name={cameras.get('image_2')!r}) "
                    "without duplicate_exterior_view; or rebuild the env from "
                    "Cosmos3Policy.observation_options() without secondary_camera_name, so it has no 'image_2'."
                )
            return
        if cameras.get("image_2") != self.secondary_camera_name:
            raise ValueError(
                f"Cosmos3Policy has secondary_camera_name={self.secondary_camera_name!r}, but this env's 'image_2' "
                f"observation group renders camera {cameras.get('image_2')!r}. Pass the same "
                f"secondary_camera_name to Cosmos3Policy.observation_options() and Cosmos3Policy.configure()."
            )
        if cameras.get("image") == self.secondary_camera_name:
            raise ValueError(
                f"Cosmos3Policy's secondary_camera_name={self.secondary_camera_name!r} is also the primary "
                "exterior camera the env's 'image' group renders, so both exterior slots would carry the same "
                "view. Name a second exterior camera in observation_options(secondary_camera_name=...) and "
                "configure(); for a scene with one exterior camera, use duplicate_exterior_view=True and no "
                "secondary_camera_name, which states the duplication."
            )

    def _check_observation_terms(self, groups_read: list[str], arm: "ImplicitPDController") -> None:
        """Refuse groups that hold other terms than :meth:`observation_options` builds, or a ``proprio`` it can't slice.

        Both checks are needed. A width check alone would pass the 8-D ``[ee_pos, ee_quat, openness]``
        group of the EE-delta clients as DROID joint state, or a joint state in another order than the
        arm term's; a term check alone would pass the right terms behind a history window, ``scale`` or
        ``clip``, which :meth:`pack_observations`' ``proprio[:7]`` / ``proprio[7:8]`` slices would send
        as the current joint state.
        """
        from eden.managers.terms.observations.camera import rgb_image

        groups: dict[str, dict] = {}
        for group, name, term in self.env.observation_manager.iter_terms():
            groups.setdefault(group, {})[name] = getattr(term, "name", type(term).__name__)
        broken = [
            f"{group!r} holds {groups[group]}"
            for group in groups_read
            if group != "proprio" and groups[group] != {"rgb": rgb_image.__name__}
        ]
        if broken:
            published = {group: list(terms) for group, terms in groups.items()}
            raise ValueError(
                f"{_SUBJECT} sends each camera group's single 'rgb' rgb_image term as one RoboArena view, but "
                f"{'; '.join(broken)}. The env publishes {published}. Fix: build the env's observation_options with "
                "ObservationManagerOptions(**Cosmos3Policy.observation_options(...)), with the same "
                "secondary_camera_name as Cosmos3Policy.configure(). Do not rename the env's own groups or terms to "
                "match: they compute another layout."
            )
        check_droid_joint_state(
            self.env,
            arm,
            group="proprio",
            subject=_SUBJECT,
            fix=f"pass arm_entity_name={arm.entity_name!r} and arm_dofs_name={list(arm.dofs_name)!r} to both "
            "Cosmos3Policy.observation_options(...) and Cosmos3Policy.action_options(...), keeping the other "
            "arguments you build them with (gripper_entity_name, secondary_camera_name, ...)",
        )
        check_plain_observation_group(
            self.env,
            "proprio",
            width=8,
            fix="build the env's 'proprio' group from Cosmos3Policy.observation_options(...) as it is, and set no "
            "history_length, scale, clip or modifiers on it: Cosmos3Policy sends proprio[:7] as "
            "observation/joint_position and proprio[7:8] as observation/gripper_position, the current raw state.",
        )

    def _rendered_cameras(self) -> dict[str, str]:
        """``{group: camera_name}`` for each built observation term that renders a named camera."""
        cameras = {}
        for group, _, term in self.env.observation_manager.iter_terms():
            camera = (getattr(term, "params", None) or {}).get("camera_name")
            if camera is not None:
                cameras[group] = camera
        return cameras

    def _reset_local(self) -> None:
        """Clear the adapter's per-episode state (task change / episode boundary)."""
        self._adapter.reset()

    # ------------------------------------------------------------------
    # Env wiring
    # ------------------------------------------------------------------

    @classmethod
    def observation_options(
        cls,
        camera_name: str = "camera",
        wrist_camera_name: str = "wrist_camera",
        secondary_camera_name: str | None = None,
        arm_entity_name: str = "robot",
        arm_dofs_name: list[str] | None = None,
        gripper_entity_name: str | None = None,
        gripper_dof_name: str = "left_driver_joint",
        gripper_close_pos: float = 0.8,
    ) -> dict[str, ObservationGroupOptions]:
        """Wire the DROID joint-state proprio + exterior/wrist camera groups.

        Mirrors OpenPI's ``droid_jointpos`` observation contract: an 8-D
        ``proprio`` vector ``[joint_pos(7), gripper(1)]`` plus an ``image``
        (exterior) and ``wrist_image`` group.  When ``secondary_camera_name``
        is given, an ``image_2`` group carries the second exterior view; pass
        the same value to ``Cosmos3Policy.configure()`` (the policy checks the
        two agree when it is constructed).
        """
        from eden.managers.terms.observations.camera import rgb_image
        from eden.options.managers.observations import ObservationGroupOptions
        from eden.terms import ObsTerm

        if arm_dofs_name is None:
            arm_dofs_name = list(DROID_ARM_DOFS)

        groups = make_image_groups(camera_name, use_wrist_image=True, wrist_camera_name=wrist_camera_name)
        if secondary_camera_name is not None:
            groups["image_2"] = ObservationGroupOptions(
                rgb=ObsTerm.configure(func=rgb_image, params={"camera_name": secondary_camera_name}),
            )

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
        gripper_threshold: float = 0.5,
    ) -> dict[str, ActionTermOptions]:
        """Absolute joint-position arm controller + ``GripperDecisionController``.

        The server returns absolute arm joint targets, so the arm controller is
        an ``ImplicitPDController`` with a zero reference (targets are applied
        directly). The gripper term receives the server's closedness and binarizes
        it at ``gripper_threshold`` (``> threshold`` closes).
        """
        return make_droid_joint_position_actions(
            arm_entity_name,
            arm_dofs_name,
            gripper_entity_name=gripper_entity_name,
            gripper_dof_name=gripper_dof_name,
            arm_key=_ARM_TERM,
            gripper_close_pos=gripper_close_pos,
            threshold=gripper_threshold,
        )

    # ------------------------------------------------------------------
    # Observation packing
    # ------------------------------------------------------------------

    def pack_observations(self, obs: dict) -> dict:
        """Convert Eden obs tensors to the Cosmos3 RoboLab server payload dict."""
        if self.task_description is None:
            raise ValueError(
                "Cosmos3Policy sends 'prompt' with every request, but Cosmos3Policy.task_description is None. Fix: "
                "policy.set_task('<instruction>') before the first step, or "
                "Cosmos3Policy.configure(task_description='<instruction>'). Do not fill in a generic instruction: "
                "the checkpoint is language-conditioned and acts on whatever it is told."
            )
        proprio = to_numpy_unbatched(obs["proprio"])
        joint_pos = proprio[:7].astype(np.float32)
        gripper_closedness = proprio[7:8].astype(np.float32)

        payload: dict = {
            "prompt": self.task_description,
            "observation/joint_position": joint_pos,
            "observation/gripper_position": gripper_closedness,
        }

        # RoboArena concat view: wrist (top) + two exterior views (bottom), composed by the
        # server. The groups were checked against this policy at construction.
        exterior_img = prepare_image_padded(obs["image"], self.image_size)
        if self.secondary_camera_name is not None:
            exterior2_img = prepare_image_padded(obs["image_2"], self.image_size)
        else:  # duplicate_exterior_view=True
            exterior2_img = exterior_img
        payload["observation/wrist_image_left"] = prepare_image_padded(obs["wrist_image"], self.image_size)
        payload["observation/exterior_image_1_left"] = exterior_img
        payload["observation/exterior_image_2_left"] = exterior2_img

        return payload

    def unpack_actions(self, raw_action: np.ndarray) -> dict[str, torch.Tensor]:
        """Decode one ``(1, 8)`` action row → Eden controller dict.

        Layout: ``[joint_pos(7), gripper(1)]``. Gripper convention: the server
        already re-applies its ``1 - x`` transform, so the scalar is closedness
        (``> 0.5`` ⇒ CLOSE). Delegated to ``JointPositionAdapter``.
        """
        return self._adapter(np.asarray(raw_action, dtype=np.float32).reshape(-1), env=self.env)
