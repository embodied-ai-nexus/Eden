"""Octo vision-language-action policy client."""

from __future__ import annotations
from typing import ClassVar, TYPE_CHECKING

import numpy as np
import torch

from eden.extensions.policies.adapters import (
    WIDOWX_DOFS,
    EEDeltaAdapter,
    check_env_gripper_term,
    make_widowx_ee_delta_actions,
)
from eden.extensions.policies.base import POLICY_REGISTRY, PolicyBase
from eden.extensions.policies.client import ClientMixin
from eden.managers.terms.observations.camera import rgb_image
from eden.options.extensions.policies import RemotePolicyOptions
from eden.options.managers.observations import ObservationGroupOptions
from eden.terms import ObsTerm
from eden.utils.common import ConfigurableMixin
from eden.utils.image import to_uint8_hwc

if TYPE_CHECKING:
    from eden.options.managers.actions import ActionTermOptions


@POLICY_REGISTRY.register()
class OctoPolicy(ClientMixin, PolicyBase, ConfigurableMixin[RemotePolicyOptions]):
    """
    Octo: A transformer-based robot manipulation policy.

    Communicates with a remote Octo policy server (RobotArena/src/server/server_octo.py):

      POST /set_task?task_description=<str>
      POST /reset
      POST /act  {"instruction": str, "image": list}
              -> {"raw_action": {"world_vector": list, "rotation_delta": list, "open_gripper": list},
                  "action":     {"world_vector": list, "rot_axangle": list,
                                 "gripper": list, "terminate_episode": list}}

    Action layout sent to Eden: [world_vector (3), rotation_delta (3), gripper (1)]
      - [:6] -> ee_controller  (DifferentialIKController; the Euler delta becomes a rotation vector)
      - [6]  -> gripper_controller (from action["gripper"], server-binarized +1=open/-1=close, forwarded raw)

    Gripper convention: server +1=open / -1=close, forwarded raw; the env's
    GripperDecisionController (input="open_sign") turns the sign into open/close.
    """

    action_dim: ClassVar[int | None] = 7  # ``[world_vector (3), rotation_delta (3), gripper (1)]``

    host: str = "localhost"
    port: int | None = 9010
    camera_name: str = "camera"
    # The Octo server returns one action per request and expects the task as a URL
    # query parameter on /set_task and a bare POST on /reset (RobotArena server_octo.py;
    # /reset calls model.reset(task), clearing the observation history).
    chunk_mode: str = "single"
    set_task_endpoint: str | None = "/set_task"
    reset_endpoint: str | None = "/reset"
    set_task_as_query: bool = True
    reset_with_payload: bool = False

    # The raw rotation_delta is extrinsic-xyz Euler; the IK controller takes a rotation vector.
    _adapter: ClassVar[EEDeltaAdapter] = EEDeltaAdapter(rotation="euler")

    def _post_connect(self) -> None:
        # The gripper sign is forwarded raw; only a GripperDecisionController(input="open_sign") reads it right.
        check_env_gripper_term(self.env, "gripper_controller", input="open_sign")

    # ------------------------------------------------------------------
    # Response decoding
    # ------------------------------------------------------------------

    def decode_response(self, response: dict) -> np.ndarray:
        """Flatten Octo's two-dict reply into one ``(1, 1, 7)`` action row.

        Uses raw_action["world_vector"] + raw_action["rotation_delta"] as the
        6-D EE delta. RobotArena's Genesis default_test.py also uses the raw
        Euler rotation_delta for its target-pose transform, while action["gripper"]
        carries the processed WidowX open/close convention. The processed
        action["world_vector"] / action["rot_axangle"] are not a substitute: the
        server scales both by its action_scale, and rot_axangle is axis-angle.
        """
        try:
            raw_action = response["raw_action"]
            action = response["action"]
            world_vector = raw_action["world_vector"]
            rotation_delta = raw_action["rotation_delta"]
            gripper = action["gripper"]
        except (KeyError, TypeError) as e:
            if isinstance(response, dict):
                got = {k: list(v) if isinstance(v, dict) else type(v).__name__ for k, v in response.items()}
            else:
                got = type(response).__name__
            raise KeyError(
                "Octo reply must carry raw_action{world_vector, rotation_delta} and action{gripper}, as "
                f"RobotArena's server_octo.py /act sends them; got {got}. Do not fall back to "
                "action['world_vector'] / action['rot_axangle']: the server scales both by its action_scale "
                "and rot_axangle is axis-angle, not the raw Euler rotation_delta this client converts."
            ) from e

        flat_action = np.concatenate(
            [
                np.asarray(world_vector, dtype=np.float32).reshape(-1),  # (3,) translation delta
                np.asarray(rotation_delta, dtype=np.float32).reshape(-1),  # (3,) Euler rotation delta
                np.asarray(gripper, dtype=np.float32).reshape(-1)[:1],  # (1,) binarized +1=open/-1=close
            ]
        )  # (7,)
        if flat_action.shape != (7,):
            raise ValueError(f"Expected Octo action shape (7,), got {flat_action.shape}: {flat_action}")
        return self._normalize_action_shape(flat_action)

    # ------------------------------------------------------------------
    # Env options
    # ------------------------------------------------------------------

    @classmethod
    def action_options(cls, entity_name: str = "robot") -> dict[str, ActionTermOptions]:
        # The WidowX configuration Octo is trained on. Octo emits +1 = open / -1 = close; the
        # GripperDecisionController reads that sign directly.
        return make_widowx_ee_delta_actions(
            entity_name, "gripper_link", list(WIDOWX_DOFS), 0.5, gripper_input="open_sign"
        )

    @classmethod
    def observation_options(cls, camera_name: str = "camera", **kwargs) -> dict[str, ObservationGroupOptions]:
        # Extra kwargs are accepted for polymorphic call sites and ignored.
        return dict(
            image=ObservationGroupOptions(
                concatenate_terms=False,
                rgb=ObsTerm.configure(
                    func=rgb_image,
                    params={"camera_name": camera_name},
                ),
            ),
        )

    def pack_observations(
        self,
        obs: dict[str, torch.Tensor | dict[str, torch.Tensor]],
    ) -> dict:
        """Convert Eden observations to the Octo server payload."""
        image_obs = obs["image"]
        image: torch.Tensor = image_obs["rgb"] if isinstance(image_obs, dict) else image_obs
        return {
            "instruction": self.task_description,
            # (H, W, 3) uint8: drops the num_envs=1 batch axis and an RGBA render's alpha; a float image raises.
            "image": to_uint8_hwc(image).tolist(),
        }

    def unpack_actions(self, raw_action: np.ndarray) -> dict[str, torch.Tensor]:
        """Convert a ``(1, 7)`` (or ``(7,)``) action ``[world_vector (3), rotation_delta (3), gripper (1)]``.

        Returns tensors with a leading batch axis of 1. ``OctoPolicy`` keeps
        ``ClientMixin.supports_batch`` at the default ``False`` because the
        upstream Octo HTTP server accepts a single image per ``/act`` request.

        The Euler rotation_delta reaches the IK controller as the rotation vector of the
        same rotation (:class:`EEDeltaAdapter`). The gripper sign (+1 open / -1 close) is
        forwarded raw; the env's ``GripperDecisionController(input="open_sign")`` interprets it.
        """
        return self._adapter(np.asarray(raw_action, dtype=np.float32).reshape(-1), env=self.env)
