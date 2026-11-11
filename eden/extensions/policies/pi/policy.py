"""PiPolicy — OpenPI (pi0 / pi0-FAST) remote inference client for Eden.

``PiPolicy`` is a thin WebSocket client built on
:class:`~eden.extensions.policies.client.SetupClientMixin`:

1. ``transport="openpi_websocket"`` speaks the OpenPI msgpack WebSocket protocol
   (no ``openpi_client`` dependency).
2. The shared ``ActionChunkScheduler`` executes ``exec_horizon`` of every
   ``pred_action_horizon``-long chunk open-loop before re-querying.
3. All setup-specific logic (obs packing, action unpacking, env wiring) lives
   in a :class:`~eden.extensions.policies.pi.setups.SetupBase` subclass
   selected by ``policy_setup``.

Supported ``policy_setup`` values
----------------------------------
widowx_bridge
    EE-delta (7-D).  8-D state ``[xyz, rpy, pad, gripper_openness]``.
    Image keys ``observation/image`` + optional ``observation/wrist_image``.

google_robot
    EE-delta (7-D).  8-D state ``[xyz, quat_xyzw, gripper_closedness]``.
    The env's GripperDecisionController applies the fractal sticky latch.

libero
    robosuite OSC_POSE (7-D) on the stock ``libero`` task's own action terms.  8-D state
    ``[eef_pos, axis_angle(eef_quat), gripper_qpos]`` in the world frame, as
    https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/main.py builds it.
    Image keys ``observation/image`` + ``observation/wrist_image`` (required).

droid
    Joint-velocity (8-D: 7 arm velocities + 1 gripper) in DROID's unit action space: 0.2 rad per
    15 Hz step, i.e. 3 rad/s per unit (ImplicitVelocityController), or the ``q + 0.2 v`` position
    target (``droid_arm_controller_kind="velocity_integrate"``, ImplicitPDController).
    Image keys ``observation/exterior_image_1_left`` + ``observation/wrist_image_left`` (required).

droid_jointpos
    Joint-position (8-D: 7 absolute arm targets + 1 gripper).  ImplicitPDController
    with zero reference.  Same observation payload as ``droid``.

aloha
    Bimanual joint-position (14-D) with normalised grippers, gym_aloha's convention.
    Images sent as a CHW dict (``cam_high``, optional ``cam_left_wrist``).  10 rows executed per
    query, as ``examples/aloha_sim/main.py``'s ``ActionChunkBroker(action_horizon=10)``.

The ``droid_*`` and ``aloha_*`` fields are read only by the setups named in their prefix; setting one
off its default under another setup raises at construction.

See :mod:`eden.extensions.policies.pi.setups` for the per-setup details.
"""

from __future__ import annotations
from typing import ClassVar

from eden.extensions.policies.base import POLICY_REGISTRY, PolicyBase
from eden.extensions.policies.client import PolicySetupBase, SetupClientMixin
from eden.extensions.policies.pi.setups import SETUP_REGISTRY
from eden.options.extensions.policies import RemotePolicyOptions
from eden.utils.common import ConfigurableMixin


@POLICY_REGISTRY.register()
class PiPolicy(SetupClientMixin, PolicyBase, ConfigurableMixin[RemotePolicyOptions]):
    """OpenPI (pi0 / pi0-FAST) remote inference client.

    All class attributes are configurable via ``PiPolicy.configure(...)``.
    """

    SETUP_REGISTRY: ClassVar[dict[str, type[PolicySetupBase]]] = SETUP_REGISTRY

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    host: str = "localhost"
    port: int | None = 8000
    api_key: str | None = None
    # Set to None to disable WebSocket keepalive pings (recommended for
    # slow first-step JIT compilation that would otherwise time out).
    ws_ping_timeout: float | None = None
    transport: str = "openpi_websocket"
    # OpenPI servers return the chunk under "actions" (plural).
    response_action_key: str = "actions"

    # ------------------------------------------------------------------
    # Setup selection
    # ------------------------------------------------------------------

    policy_setup: str = "widowx_bridge"

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    # None until set_task(): setups whose upstream client always sends a prompt refuse to send a
    # request without one; ALOHA omits it, so the server applies its own default_prompt.
    task_description: str | None = None
    # None takes the selected setup's default: the reference client's replan / open-loop horizon
    # (libero: replan_steps=5, droid: open_loop_horizon=8, aloha: ActionChunkBroker(action_horizon=10)).
    pred_action_horizon: int | None = None
    exec_horizon: int | None = None
    # Optional test-time smoothing for overlapping action chunks. Zero keeps
    # the stock OpenPI execution path bit-for-bit unchanged. When enabled, the
    # unused tail of the previous prediction is linearly ensembled into the
    # beginning of the next chunk.
    chunk_temporal_ensemble_steps: int = 0
    chunk_temporal_ensemble_dims: int | None = None

    # ------------------------------------------------------------------
    # Image preprocessing
    # ------------------------------------------------------------------

    image_size: tuple[int, int] = (224, 224)

    # ------------------------------------------------------------------
    # Cameras
    # ------------------------------------------------------------------

    # None takes the setup's default: True for the setups whose server reads a wrist image on
    # every request (libero, droid, droid_jointpos), False otherwise.
    use_wrist_image: bool | None = None
    # Aloha-specific wrist camera key sent inside the images dict
    aloha_left_camera_name: str = "cam_left_wrist"

    # ------------------------------------------------------------------
    # DROID-specific
    # ------------------------------------------------------------------

    droid_arm_controller_kind: str = "velocity"
    # Optional first-order lag on the commanded joint velocity, modelling the
    # real DROID controller's bandwidth (~210-320 ms) that Genesis's velocity
    # actuator (~11-48 ms) lacks; its dc_gain also corrects the steady-state
    # tracking gain. Velocity path only; None = no-op. Both filters run at DROID's 1/15 s
    # action period, which the setup requires of the env.
    droid_velocity_lag: dict | None = None
    droid_velocity_compensation: dict | None = None
    droid_gripper_threshold: float = 0.5
    droid_debug_gripper: bool = False

    def _post_configure(self) -> None:
        if self.use_wrist_image is None:
            self.use_wrist_image = self._setup_class(self.policy_setup).wrist_image_required
        super()._post_configure()
