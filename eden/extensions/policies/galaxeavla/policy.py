"""GalaxeaVLA (G0.5) remote inference client for Eden.

``GalaxeaVlaPolicy`` is a thin **WebSocket** client for the GalaxeaVLA policy
server (``scripts/serve_policy.py`` in the GalaxeaVLA repo), built on
``transport="galaxea_websocket"``: a synchronous ``websockets.sync.client``
connection with an msgpack codec whose record layout matches OpenPI's numpy hook
(``{"__ndarray__": True, "data": bytes, "dtype": str, "shape": tuple}``, str keys).
The server's ``g05`` package is not needed.

Protocol
--------
* On connect the server sends a metadata handshake ``{"action_steps": N}``.
* Each :meth:`step` sends one request and receives a **single-step** action dict
  (R1 Lite shown; R1 Pro's arms are 7-D)::

      {"action": {"left_arm": (6,), "left_gripper": (1,),
                  "right_arm": (6,), "right_gripper": (1,)},
       "need_obs": bool}

  The server owns the chunk cache (``chunk_mode="server"``): it recomputes only
  when its internal chunk is exhausted and otherwise serves cached steps
  (open-loop execution), so there is no client-side action queue.
  ``droid_franka`` follows the reference ``droid-franka-client``: after a reply
  with ``need_obs=False`` the next request is ``{}``, otherwise (and after a reset)
  it is the full observation. The R1 setups send the full observation every tick,
  as upstream's R1-Lite scheduler does (the server ignores it on cached steps).
* The ``action`` dict omits every part the model was not confident about. R1: an
  omitted part keeps its previous target of the episode (before its first, the
  measured position), as upstream's R1-Lite client publishes only the parts it
  received. ``droid_franka``: the gripper commands its measured closedness; an
  omitted arm raises, as in the reference client. A reply with none of the parts,
  or a part sent with a non-finite value, raises.
* :meth:`reset` sends ``{"__reset__": True}`` to clear the server's chunk cache
  at episode boundaries.

Supported ``policy_setup`` values
----------------------------------
galaxea_r1lite
    Dual 6-DOF arms (absolute joint-position targets) + two 1-DOF grippers,
    three cameras (head + left/right wrist).  See
    :mod:`eden.extensions.policies.galaxeavla.setups`.
galaxea_r1pro
    The same parts and cameras with dual **7-DOF** arms (GalaxeaVLA
    ``configs/data/r1pro.yaml``).
droid_franka
    Single 7-DOF Franka arm (absolute joint-position targets) + parallel-jaw
    gripper, exterior + wrist cameras.  Matches the ``Droid_Franka`` embodiment
    served by ``GalaxeaVLA/experiments/droid/start_server.sh``.

Manual follow-ups (cannot be derived from the server repo)
----------------------------------------------------------
* Set the arm/gripper **DOF names** and camera names in
  ``observation_options`` / ``action_options`` to match your Eden robot.
* R1 setups: pass ``gripper_open_pos`` / ``gripper_close_pos``, the Eden DOF positions
  of your robot's open and closed gripper, to both ``observation_options`` and
  ``action_options`` (required). The wire side is fixed: the Galaxea HDAS gripper
  stroke, 0 = closed … 100 = open (see ``R1_GRIPPER_*_STROKE`` in
  :mod:`~eden.extensions.policies.galaxeavla.setups`), driven continuously as upstream's
  R1-Lite client does. ``droid_franka`` follows the droid-franka-client reference
  (openness on the wire, continuous-snap at 0.8 by default).
"""

from __future__ import annotations

import math
from typing import ClassVar

import numpy as np

from eden.extensions.policies.adapters import check_control_rate
from eden.extensions.policies.base import POLICY_REGISTRY, PolicyBase
from eden.extensions.policies.client import PolicySetupBase, SetupClientMixin
from eden.extensions.policies.galaxeavla.setups import DROID_GRIPPER_THRESHOLD_DEFAULTS, SETUP_REGISTRY
from eden.options.extensions.policies import RemotePolicyOptions
from eden.utils.common import ConfigurableMixin


@POLICY_REGISTRY.register()
class GalaxeaVlaPolicy(SetupClientMixin, PolicyBase, ConfigurableMixin[RemotePolicyOptions]):
    """GalaxeaVLA (G0.5) remote WebSocket inference client.

    All class attributes are configurable via ``GalaxeaVlaPolicy.configure(...)``.
    """

    SETUP_REGISTRY: ClassVar[dict[str, type[PolicySetupBase]]] = SETUP_REGISTRY

    # -- connection -------------------------------------------------------
    host: str = "localhost"
    port: int | None = 8765
    api_key: str | None = None
    ws_timeout_s: float = 60.0
    transport: str = "galaxea_websocket"
    # scripts/serve_policy.py clears its chunk cache on a {"__reset__": True} message.
    reset_endpoint: str | None = "__reset__"
    # The server serves one action per request from its own chunk cache.
    chunk_mode: str = "server"
    short_chunk: str = "accept"

    # -- setup selection --------------------------------------------------
    policy_setup: str = "galaxea_r1lite"

    # -- inference --------------------------------------------------------
    task_description: str = "pick up the object"
    control_frequency: float = 15.0

    # -- gripper decoding (droid_franka only; R1 grippers are continuous HDAS strokes) --
    # binary only: close when closedness exceeds it (continuous-snap does not read it).
    gripper_close_threshold: float = DROID_GRIPPER_THRESHOLD_DEFAULTS["gripper_close_threshold"]
    # DROID decoding modes: continuous-snap (default, the droid-franka-client reference:
    # eval/main.py sends clip(1 - openness) as a position and snaps > 0.8 to closed, and
    # sends the measured gripper position when the server omits the key); binary
    # thresholds closedness at gripper_close_threshold and holds on an omitted key.
    # Read by droid_franka only, like gripper_close_threshold: setting one under an R1 setup raises.
    gripper_decode_mode: str = "continuous-snap"
    # continuous-snap only (binary does not read it).
    gripper_snap_threshold: float = DROID_GRIPPER_THRESHOLD_DEFAULTS["gripper_snap_threshold"]
    debug_gripper: bool = False  # log every gripper decision (JointPositionAdapter)

    #: ``ClientMixin`` knobs no GalaxeaVLA setup reads, with the reason; a non-default value raises.
    _UNREAD_CLIENT_FIELDS: ClassVar[dict[str, str]] = {
        **dict.fromkeys(
            ("pred_action_horizon", "exec_horizon", "chunk_temporal_ensemble_steps", "chunk_temporal_ensemble_dims"),
            "the server owns the action chunk and answers every request with one step, so the client never "
            "buffers, cuts or ensembles a chunk (the chunk length is serve_policy.py --action_steps)",
        ),
        "image_size": "images go out at native resolution and the server resizes them to the checkpoint's shape_meta",
        "response_action_key": "the setups decode the reply's 'action' dict by part name",
    }

    def _post_configure(self) -> None:
        setup_cls = self._setup_class(self.policy_setup)
        self._refuse_unread_fields(setup_cls)
        self._check_control_frequency(setup_cls)
        super()._post_configure()

    def _refuse_unread_fields(self, setup_cls: type[PolicySetupBase]) -> None:
        """Raise when a field the selected setup (or every setup) ignores was changed from its default."""
        unread = {
            name: getattr(self, name)
            for name in self._UNREAD_CLIENT_FIELDS
            if getattr(self, name) != getattr(type(self), name)
        }
        if unread:
            given = ", ".join(f"{name}={value!r}" for name, value in unread.items())
            reasons = "; ".join(sorted({self._UNREAD_CLIENT_FIELDS[name] for name in unread}))
            raise ValueError(
                f"{type(self).__name__}: {given} has no effect under any policy_setup: {reasons}.\n"
                f"Fix: drop {sorted(unread)} from configure()."
            )
        ignored = {
            name: getattr(self, name)
            for name in setup_cls.unused_policy_fields
            if getattr(self, name) != getattr(type(self), name)
        }
        if ignored:
            given = ", ".join(f"{name}={value!r}" for name, value in ignored.items())
            readers = {
                name: sorted(key for key, cls in self.SETUP_REGISTRY.items() if name not in cls.unused_policy_fields)
                for name in ignored
            }
            raise ValueError(
                f"{type(self).__name__}: {given} has no effect under policy_setup={self.policy_setup!r}; "
                f"only these setups read them: {readers}.\n"
                f"Fix: drop {sorted(ignored)} from configure(), or select the policy_setup you meant."
            )

    def _check_control_frequency(self, setup_cls: type[PolicySetupBase]) -> None:
        """Refuse a ``control_frequency`` the env does not step at, or one the setup's wire type cannot carry.

        The server is told ``control_frequency`` and serves one step of its plan per request, and the client
        sends one request per ``env.step``: an env stepping at another rate plays the plan faster or slower.
        """
        frequency = self.control_frequency
        if (
            isinstance(frequency, bool)
            or not isinstance(frequency, (int, float, np.integer, np.floating))
            or not math.isfinite(frequency)
            or frequency <= 0
        ):
            raise ValueError(
                f"{type(self).__name__}.control_frequency must be a positive number of Hz, got {frequency!r}.\n"
                "Fix: configure(control_frequency=15.0), the rate the checkpoint was trained at (GalaxeaVLA's R1-Lite "
                "config.toml and droid-franka-client both run at 15 Hz)."
            )
        if getattr(setup_cls, "frequency_type", float) is int and not float(frequency).is_integer():
            raise ValueError(
                f"{type(self).__name__}.policy_setup={self.policy_setup!r} sends 'frequency' as an int, as "
                f"droid-franka-client does (GalaxeaVLA experiments/droid/PROTOCOL.md), but "
                f"control_frequency={frequency!r} is not a whole number of Hz.\n"
                "Fix: configure(control_frequency=15) and build the env at that rate.\n"
                "Do NOT round it in the client: the server would be told another rate than the env runs at."
            )
        check_control_rate(
            self.env,
            control_hz=float(frequency),
            subject=f"{type(self).__name__}.policy_setup={self.policy_setup!r}",
            source=f"control_frequency={frequency!r} tells the server, which serves one step of its plan per request",
            wrong_fix="Do not change control_frequency to match the env instead: the server feeds it to the model as a "
            "condition, so it must stay the rate the checkpoint was trained at (GalaxeaVLA's R1-Lite config.toml and "
            "droid-franka-client both run at 15 Hz).",
        )
