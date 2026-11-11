"""MolmoAct2 — vision-language manipulation policy (allenai/MolmoAct2-*).

Client for the FastAPI inference server at
https://github.com/allenai/molmoact2/blob/main/examples/droid/host_server_droid.py. Wire protocol
(``json_numpy``-encoded POST to ``/act``)::

    request:
        {
          "external_cam":  ndarray(H, W, 3) uint8 RGB,
          "wrist_cam":     ndarray(H, W, 3) uint8 RGB,
          "instruction":   str,
          "state":         ndarray(8,) float32  [q1..q7, gripper],
          "num_steps":     int  (optional, default 10),
          "enable_cuda_graph": bool (optional),
        }
    response:
        {"actions": ndarray(N, 8) float32, "dt_ms": float}

MolmoAct2-DROID is **absolute joint-pose control**: ``actions[:, :7]`` are
target joint positions (rad) for Franka joints 1..7, and ``actions[:, 7]`` is
a gripper scalar in roughly ``[0, 1]`` (binarized at 0.5 by default). Each reply is a
15-row chunk (``norm_stats.json`` ``franka_droid``: ``action_horizon: 15``,
``n_action_steps: 15``) of joint waypoints 1/15 s apart, executed whole at DROID's 15 Hz:
the env must step every 1/15 s. The rate, arm-term and joint-state refusals at construction are the
shared :func:`~eden.extensions.policies.adapters.check_control_rate`,
:func:`~eden.extensions.policies.adapters.check_droid_arm_term` and
:func:`~eden.extensions.policies.adapters.check_droid_joint_state` every DROID client runs.

The server has no ``/set_task`` or ``/reset`` endpoints (``set_task_endpoint`` and
``reset_endpoint`` are ``None``), so ``set_task`` / ``reset_server`` only clear the
local action queue. Transport: ``transport="http_json_numpy"``.
"""

from __future__ import annotations

import math
from typing import ClassVar, TYPE_CHECKING

import numpy as np
import torch

from eden.extensions.policies.base import POLICY_REGISTRY, PolicyBase
from eden.extensions.policies.client import ClientMixin
from eden.extensions.policies.adapters import (
    DROID_ARM_DOFS,
    DROID_CONTROL_HZ,
    DROID_CONTROL_TIMING,
    DROID_CONTROL_TIMING_NOTE,
    check_control_rate,
    check_droid_arm_term,
    check_droid_joint_state,
    check_env_gripper_term,
    check_gripper_state_matches_action,
    check_plain_observation_group,
    make_droid_joint_position_actions,
    make_droid_joint_state_group,
)
from eden.utils.image import to_uint8_hwc
from eden.utils.torch import to_numpy_unbatched
from eden.managers.terms.observations.camera import rgb_image
from eden.options.extensions.policies import RemotePolicyOptions
from eden.options.managers.observations import ObservationGroupOptions
from eden.terms import ObsTerm
from eden.utils.common import ConfigurableMixin

if TYPE_CHECKING:
    from eden.managers.terms.actions.joint_actions import ImplicitPDController
    from eden.options.managers.actions import ActionTermOptions

#: The env's absolute-joint arm term, as :meth:`MolmoAct2Policy.action_options` names it.
_ARM_TERM = "arm_controller"
#: Who the construction refusals name (the client has a single setup).
_SUBJECT = "MolmoAct2Policy"
#: The ``action_options`` call a refusal of the arm term points to.
_ACTION_FIX = "MolmoAct2Policy.action_options(), passing the rig's arm / gripper names as keyword arguments"


@POLICY_REGISTRY.register()
class MolmoAct2Policy(ClientMixin, PolicyBase, ConfigurableMixin[RemotePolicyOptions]):
    """Client for the MolmoAct2 DROID inference server (absolute joint control)."""

    action_dim: ClassVar[int | None] = 8  # ``[joint_position (7), gripper (1)]``

    host: str = "localhost"
    port: int | None = 8000
    act_endpoint: str = "/act"
    # The server has no /set_task or /reset endpoint: those calls only clear local state.
    set_task_endpoint: str | None = None
    reset_endpoint: str | None = None
    transport: str = "http_json_numpy"
    response_action_key: str = "actions"
    # MolmoAct2-DROID's norm_stats.json (franka_droid): action_horizon 15, n_action_steps 15. A shorter
    # reply is an error (ClientMixin's short_chunk="error"); the whole chunk runs before re-querying.
    pred_action_horizon: int | None = 15
    request_timeout: float = 120.0

    # Inference knobs forwarded to the server (see DEFAULT_NUM_STEPS in
    # host_server_droid.py).
    num_steps: int = 10
    enable_cuda_graph: bool = False

    # Gripper decode. MolmoAct2-DROID outputs gripper closedness in [0, 1]
    # (0 = open, 1 = closed) — see norm_stats.json `franka_droid` action_stats
    # (gripper mask=False, min=0, max=1) and processor_molmoact2.py:417.
    # It is forwarded raw to the env's GripperDecisionController, which binarizes it at
    # ``gripper_threshold`` (or follows it continuously with binarize_gripper=False).
    binarize_gripper: bool = True
    gripper_threshold: float = 0.5
    # Set True only if the served checkpoint emits the inverted convention (openness).
    flip_gripper: bool = False

    def _post_configure(self) -> None:
        check_control_rate(
            self.env,
            control_hz=DROID_CONTROL_HZ,
            subject=_SUBJECT,
            source="DROID's RobotEnv does (control_hz=15) in the data MolmoAct2-DROID was trained on",
            timing=DROID_CONTROL_TIMING,
            timing_note=DROID_CONTROL_TIMING_NOTE,
        )
        arm = self._check_arm_term()
        self._check_observation_groups(arm)
        # Every decision field action_options() sets; the shared check requires the rest at their defaults.
        check_env_gripper_term(
            self.env,
            "gripper_controller",
            input="openness" if self.flip_gripper else "closedness",
            threshold=float(self.gripper_threshold),
            continuous=not self.binarize_gripper,
        )

    def _check_arm_term(self) -> "ImplicitPDController":
        """Return the env's ``arm_controller``, refused unless it applies the absolute DROID joint targets unchanged."""
        from eden.managers.terms.actions.joint_actions import ImplicitPDController

        terms = self.env.action_manager.terms
        term = terms.get(_ARM_TERM)
        if type(term) is not ImplicitPDController:
            found = "missing" if term is None else f"of type {type(term).__name__}"
            raise ValueError(
                f"{_SUBJECT} sends absolute DROID joint targets to action term {_ARM_TERM!r}, which must be exactly an "
                f"ImplicitPDController (a subclass reinterprets the same row), but the env's {_ARM_TERM!r} is {found} "
                f"(action terms: {sorted(terms)}). Fix: build the env's action_options from {_ACTION_FIX}."
            )
        check_droid_arm_term(term, term_name=_ARM_TERM, subject=_SUBJECT, fix=_ACTION_FIX)
        return term

    def _check_observation_groups(self, arm: "ImplicitPDController") -> None:
        """Refuse an env that does not publish the two camera images and DROID's 8-D joint state.

        Checked at construction, so an env built from another layout (a stock task's groups, or the
        8-D ``[ee_pos, ee_quat, openness]`` group of the EE-delta clients) fails here with the fix rather
        than as a ``KeyError`` on the first step, or by sending an end-effector pose as joint state.
        """
        groups: dict[str, dict] = {}
        for group, name, term in self.env.observation_manager.iter_terms():
            groups.setdefault(group, {})[name] = term
        image = groups.get("image", {})
        broken = []
        cameras = {name: getattr(term, "name", type(term).__name__) for name, term in image.items()}
        if cameras != {"external": rgb_image.__name__, "wrist": rgb_image.__name__}:
            broken.append(f"'image' must hold rgb_image terms 'external' and 'wrist', but holds {cameras}")
        image_options = getattr(self.env.config.observation_options, "image", None)
        if getattr(image_options, "concatenate_terms", False):
            # pack_observations reads obs['image']['external'] and obs['image']['wrist'] as separate frames.
            broken.append("'image' has concatenate_terms=True, which fuses the two cameras into one tensor")
        if broken:
            published = {group: list(terms) for group, terms in groups.items()}
            raise ValueError(
                f"{_SUBJECT} sends the 'external' and 'wrist' camera images as two separate frames, but the env's "
                f"observations do not match: {'; '.join(broken)}. The env publishes {published}. Fix: build the env's "
                "observation_options with ObservationManagerOptions(**MolmoAct2Policy.observation_options(...)), "
                "naming its cameras through external_camera / wrist_camera. Do not rename the env's own groups to "
                "match: their terms compute another layout."
            )
        # The server reads state[:7] as q1..q7 and replies targets for those joints in that order.
        check_droid_joint_state(
            self.env,
            arm,
            group="proprio",
            subject=_SUBJECT,
            fix=f"build the env's 'proprio' group from MolmoAct2Policy.observation_options(arm_entity_name="
            f"{arm.entity_name!r}, arm_dofs_name={list(arm.dofs_name)!r}, ...), the arm the env's {_ARM_TERM!r} drives, "
            "keeping the other arguments you build it with (external_camera, gripper_entity_name, ...)",
        )
        # The server reads state[:7] as q1..q7 and state[7] as closedness: this step's values, as computed.
        check_plain_observation_group(
            self.env,
            "proprio",
            width=8,
            fix="build 'proprio' with MolmoAct2Policy.observation_options(...)['proprio'] as returned, adding no "
            "history_length, scale, clip, noise or modifiers. Do not rescale the state in "
            "pack_observations to compensate: the server expects DROID's raw joint state.",
        )
        # The state's closedness and the command's targets come from separate observation_options() /
        # action_options() arguments (gripper_entity_name, gripper_close_pos): they must describe one gripper.
        check_gripper_state_matches_action(self.env, group="proprio", term="gripper", action_term="gripper_controller")

    # ------------------------------------------------------------------
    # Payload
    # ------------------------------------------------------------------

    def pack_observations(
        self,
        obs: dict[str, torch.Tensor | dict[str, torch.Tensor]],
    ) -> dict:
        if self.task_description is None:
            raise ValueError(
                "MolmoAct2Policy sends 'instruction' with every request, but MolmoAct2Policy.task_description is "
                "None. Fix: policy.set_task('<instruction>') before the first step, or "
                "MolmoAct2Policy.configure(task_description='<instruction>'). Do not fill in a generic "
                "instruction: the checkpoint is language-conditioned and acts on whatever it is told."
            )
        missing = [group for group in ("image", "proprio") if group not in obs]
        if missing:
            raise KeyError(
                f"MolmoAct2Policy reads the observation groups ['image', 'proprio'], but obs has no {missing} (its "
                f"groups are {sorted(obs)}). Fix: pass the observations of an env built from "
                "MolmoAct2Policy.observation_options(...) as they are. Do not fill a missing group with zeros: the "
                "server would act on a blank view or a zero joint state."
            )
        image_group = obs["image"]
        if not isinstance(image_group, dict):
            raise TypeError(
                f"MolmoAct2Policy expects obs['image'] to be a dict with 'external' and 'wrist' keys, but it is a "
                f"{type(image_group).__name__}. Fix: build the env's 'image' group from "
                "MolmoAct2Policy.observation_options(...), which keeps the two cameras apart (concatenate_terms=False)."
            )
        if "external" not in image_group or "wrist" not in image_group:
            raise KeyError(
                "MolmoAct2Policy sends obs['image']['external'] and obs['image']['wrist'] as its two views, but "
                f"obs['image'] holds {sorted(image_group)}. Fix: build the env's 'image' group from "
                "MolmoAct2Policy.observation_options(external_camera=..., wrist_camera=...), which names the env's "
                "cameras. Do not send one camera as both views: the checkpoint was trained on two."
            )

        proprio = to_numpy_unbatched(obs["proprio"]).astype(np.float32)
        if proprio.shape != (8,):
            raise ValueError(f"MolmoAct2 expects 8-D proprio [q1..q7, gripper], got {proprio.shape}.")

        payload = {
            "external_cam": to_uint8_hwc(image_group["external"]),
            "wrist_cam": to_uint8_hwc(image_group["wrist"]),
            "instruction": self.task_description,
            "state": proprio,
            "num_steps": int(self.num_steps),
        }
        if self.enable_cuda_graph:
            payload["enable_cuda_graph"] = True
        return payload

    # ------------------------------------------------------------------
    # Action decoding — (N, 8) abs joint targets + gripper scalar
    # ------------------------------------------------------------------

    def accept_chunk(self, chunk: np.ndarray) -> None:
        """Refuse a chunk with any non-finite arm target before its first row executes, then queue it.

        Every row of the reply is scanned, including any past the 15 that execute: a non-finite value anywhere
        means the server's output is broken, so the rows before it are not trusted either.
        """
        chunk = self._normalize_action_shape(np.asarray(chunk, dtype=np.float32))
        self._check_action_width(chunk, self.action_dim, schema=type(self).__name__)
        # Arm columns only, on the whole (num_envs, T, 8) chunk: a NaN gripper alone holds the gripper.
        bad = ~np.isfinite(chunk[:, :, :7]).all(axis=2)
        if bad.any():
            env_idx, row_idx = (int(i) for i in np.argwhere(bad)[0])
            raise ValueError(
                f"MolmoAct2 server returned non-finite arm joint targets for env {env_idx} in row {row_idx} of its "
                f"{chunk.shape[1]}-row chunk: {[round(float(v), 6) for v in chunk[env_idx, row_idx]]} ([q1..q7, "
                f"gripper], rad). The whole chunk is refused before any row reaches the arm term {_ARM_TERM!r} "
                "(control_dofs_pos). Fix: check the served checkpoint and its norm_stats.json (franka_droid) on the "
                "server; a NaN gripper alone is allowed (it holds the gripper). Do not replace the bad values with "
                "the current or a default pose, or execute the rows before it: the whole chunk was predicted from "
                "the same broken output."
            )
        super().accept_chunk(chunk)

    def unpack_actions(self, raw_action: np.ndarray) -> dict[str, torch.Tensor]:
        """Decode an ``(num_envs, 8)`` MolmoAct2-DROID action row.

        ``raw_action[:, :7]`` are absolute Franka joint targets in rad and are
        passed straight through to the arm ``ImplicitPDController`` (scale=1).
        ``raw_action[:, 7]`` is gripper **closedness in ``[0, 1]``** (0=open,
        1=closed), forwarded raw to the env's ``GripperDecisionController``, which
        binarizes it at ``gripper_threshold`` (or follows it with ``binarize_gripper=False``).

        Stores ``self.last_diagnostics`` with the raw gripper scalar, threshold, and the intent
        the env's gripper term derives from it, so callers can record/plot gripper traces.
        """
        device = self.env.device
        arm = torch.as_tensor(raw_action[:, :7], dtype=torch.float32, device=device)
        # The raw closedness goes to the env's GripperDecisionController, configured by
        # action_options() with the same flip / threshold / binarize semantics.
        gripper = torch.as_tensor(raw_action[:, 7:8], dtype=torch.float32, device=device)

        raw_g = float(raw_action[0, 7])
        self.last_diagnostics = {
            "raw_gripper_action": raw_g,
            "gripper_threshold": float(self.gripper_threshold),
            "gripper_intent": self._gripper_intent(raw_g),
            "gripper_command": raw_g,
            "binarize_gripper": bool(self.binarize_gripper),
            "flip_gripper": bool(self.flip_gripper),
        }
        return {_ARM_TERM: arm, "gripper_controller": gripper}

    def _gripper_intent(self, raw: float) -> str:
        """What the env's ``GripperDecisionController`` (built by :meth:`action_options`) does with ``raw``.

        Mirrors the term's rule rather than re-deriving a closedness: ``input="openness"`` opens iff
        the openness is strictly above the threshold (so it closes on ``raw <= threshold``),
        ``input="closedness"`` closes iff the closedness is strictly above it. With
        ``binarize_gripper=False`` the term follows the clipped closedness, so there is no binary
        intent; a NaN is a missing command, which the term holds.
        """
        if math.isnan(raw):
            return "hold"
        if not self.binarize_gripper:
            return "continuous"
        if self.flip_gripper:
            return "open" if raw > self.gripper_threshold else "close"
        return "close" if raw > self.gripper_threshold else "open"

    # ------------------------------------------------------------------
    # Env options — DROID Franka
    # ------------------------------------------------------------------

    @classmethod
    def action_options(
        cls,
        arm_entity_name: str = "robot",
        arm_dofs_name: list[str] | None = None,
        gripper_entity_name: str | None = None,
        gripper_dof_name: str = "left_driver_joint",
        gripper_close_pos: float = 0.8,
        gripper_threshold: float = 0.5,
        binarize_gripper: bool = True,
        flip_gripper: bool = False,
    ) -> dict[str, ActionTermOptions]:
        """Absolute joint arm controller + ``GripperDecisionController`` fed the server's closedness.

        ``flip_gripper`` treats the scalar as openness instead; ``binarize_gripper=False``
        drives the fingers continuously by the (clipped) closedness. Keep these equal to the
        policy's attributes of the same name (checked at policy construction).
        """
        return make_droid_joint_position_actions(
            arm_entity_name,
            arm_dofs_name,
            gripper_entity_name=gripper_entity_name,
            gripper_dof_name=gripper_dof_name,
            arm_key=_ARM_TERM,
            gripper_close_pos=gripper_close_pos,
            input="openness" if flip_gripper else "closedness",
            threshold=gripper_threshold,
            continuous=not binarize_gripper,
        )

    @classmethod
    def observation_options(
        cls,
        external_camera: str = "camera",
        wrist_camera: str = "wrist_camera",
        arm_entity_name: str = "robot",
        arm_dofs_name: list[str] | None = None,
        gripper_entity_name: str | None = None,
        gripper_dof_name: str = "left_driver_joint",
        gripper_close_pos: float = 0.8,
    ) -> dict[str, ObservationGroupOptions]:
        if arm_dofs_name is None:
            arm_dofs_name = list(DROID_ARM_DOFS)

        image_group = ObservationGroupOptions(
            concatenate_terms=False,
            external=ObsTerm.configure(
                func=rgb_image,
                params={"camera_name": external_camera},
            ),
            wrist=ObsTerm.configure(
                func=rgb_image,
                params={"camera_name": wrist_camera},
            ),
        )

        return dict(
            image=image_group,
            proprio=make_droid_joint_state_group(
                arm_entity_name,
                arm_dofs_name,
                gripper_entity_name=gripper_entity_name,
                gripper_dof_name=gripper_dof_name,
                gripper_close_pos=gripper_close_pos,
            ),
        )
