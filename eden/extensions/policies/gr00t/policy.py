"""Gr00tPolicy — NVIDIA Isaac-GR00T remote-inference client for Eden.

Talks to ``gr00t.policy.server_client.PolicyServer`` (see Isaac-GR00T's
``gr00t/eval/run_gr00t_server.py``) over the
ZMQ + msgpack-numpy transport that ships with Isaac-GR00T
(``transport="zmq_msgpack"``, serializer ``msgpack_numpy``). The wire protocol
is implemented in :mod:`eden.utils.policy_client.transport`, so only
``zmq`` and ``msgpack_numpy`` are required, not the ``gr00t`` package.

Endpoints used:

  * ``get_action({"observation": ..., "options": None})`` ->
    ``[action_dict, info]`` where ``action_dict`` carries one
    ``(B, T, d)`` array per action key (e.g. ``x``, ``y``, ``eef_9d``); ``info`` is kept in
    ``last_response_info``.
  * ``reset({"options": None})``
  * ``ping`` / ``kill``

Two request formats exist, chosen by how the server was launched. Without
``--use-sim-policy-wrapper`` the server is a bare ``Gr00tPolicy``: the observation is nested
(``{"video": {key: ...}, "state": {key: ...}, "language": {key: [[text]]}}``) and the reply is
keyed by the bare action key. With it, ``Gr00tSimPolicyWrapper`` takes flat ``video.<key>`` /
``state.<key>`` arrays and the language key with a ``[text]`` batch, and replies
``action.<key>``. ``use_sim_policy_wrapper`` selects the format; ``None`` takes the launch the
upstream README documents for the setup.

Per-embodiment serialization (state layout, image keys, action concatenation
order, controller wiring) is delegated to a :class:`SetupBase` subclass
selected by ``policy_setup``. See :mod:`eden.extensions.policies.gr00t.setups`.

Currently supported setups
--------------------------
``simpler_env_widowx``
    SimplerEnv WidowX/Bridge — 7-D EE-delta; the gripper openness goes to the env's
    ``GripperDecisionController``. Upstream launches its server with the sim wrapper.
``libero_sim``
    LIBERO Panda — robosuite's world-frame grip-site state, and robosuite ``OSC_POSE`` actions
    executed by the stock ``libero`` task's own ``osc_pose`` / ``gripper`` terms, as
    Isaac-GR00T's LIBERO env does. Upstream launches its server with the sim wrapper.
``oxe_droid_relative_eef_relative_joint``
    OXE-DROID — absolute arm joint targets (the server converts its relative
    prediction against the sent state) + gripper closedness binarized at a strict
    ``> 0.5`` by the env term; eef_9d action stream is ignored on output (state-only).
    Images letterboxed to (180, 320) and a required wrist view, as NVIDIA's DROID client.
    Upstream launches the DROID server without the sim wrapper.
"""

from __future__ import annotations
from typing import Any, ClassVar

import numpy as np

from eden.extensions.policies.base import POLICY_REGISTRY, PolicyBase
from eden.extensions.policies.client import PolicySetupBase, SetupClientMixin
from eden.extensions.policies.gr00t.setups import SETUP_REGISTRY, to_sim_wrapper_request, wire_format_hint
from eden.options.extensions.policies import RemotePolicyOptions
from eden.utils.common import ConfigurableMixin
from eden.utils.policy_client.errors import ServerError


@POLICY_REGISTRY.register()
class Gr00tPolicy(SetupClientMixin, PolicyBase, ConfigurableMixin[RemotePolicyOptions]):
    """NVIDIA Isaac-GR00T remote-inference client.

    All class attributes are configurable via :meth:`Gr00tPolicy.configure`.
    """

    SETUP_REGISTRY: ClassVar[dict[str, type[PolicySetupBase]]] = SETUP_REGISTRY
    #: Fields only some setups read: the union of every setup's ``SetupBase.policy_fields``. A
    #: non-default value under a setup that ignores it raises instead of being silently dropped.
    _SETUP_SPECIFIC_FIELDS: ClassVar[tuple[str, ...]] = tuple(
        sorted(frozenset().union(*(cls.policy_fields for cls in SETUP_REGISTRY.values())))
    )

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------

    host: str = "localhost"
    port: int | None = 5555
    timeout_ms: int = 60000
    api_token: str | None = None
    transport: str = "zmq_msgpack"
    # Isaac-GR00T's PolicyServer registers a "reset" endpoint (gr00t/policy/server_client.py).
    reset_endpoint: str | None = "reset"
    # GR00T setups execute min(exec_horizon, T) rows and never reject a short chunk.
    short_chunk: str = "accept"

    # ------------------------------------------------------------------
    # Setup selection and wire format
    # ------------------------------------------------------------------

    policy_setup: str = "simpler_env_widowx"
    # True: the server was launched with ``--use-sim-policy-wrapper`` (flat ``video.<key>`` /
    # ``state.<key>`` request, ``action.<key>`` reply); False: a bare Gr00tPolicy (nested request).
    # None takes the launch the upstream README documents for the setup.
    use_sim_policy_wrapper: bool | None = None

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    # Set with set_task() (or here); a step without it raises rather than sending no instruction.
    task_description: str | None = None
    # Optional overrides; leave as None to use the setup's defaults.
    pred_action_horizon: int | None = None
    exec_horizon: int | None = None

    # ------------------------------------------------------------------
    # Image preprocessing
    # ------------------------------------------------------------------

    # Set to None to let the selected setup pick its training-time default
    # (e.g. WidowxSimplerSetup uses (256, 256) to match the reference
    # SimplerEnv eval). Set explicitly to override per-instance.
    image_size: tuple[int, int] | None = None

    # ------------------------------------------------------------------
    # OXE-DROID only (rejected under the other setups)
    # ------------------------------------------------------------------

    # None means "use the setup's modality default". Override only when the running GR00T server
    # was launched with a custom modality config.
    video_horizon: int | None = None
    video_stride: int | None = None
    # Log each executed row's gripper decision (official ``> 0.5`` rule) and the state sent.
    debug_gripper: bool = False

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _post_configure(self) -> None:
        setup_cls = self._setup_class(self.policy_setup)
        for name in self._SETUP_SPECIFIC_FIELDS:
            value = getattr(self, name)
            if name in setup_cls.policy_fields or value == getattr(type(self), name):
                continue
            readers = sorted(key for key, cls in self.SETUP_REGISTRY.items() if name in cls.policy_fields)
            raise ValueError(
                f"Gr00tPolicy.{name}={value!r} is read only by policy_setup {readers}; "
                f"policy_setup={self.policy_setup!r} ignores it. Fix: drop {name}= from Gr00tPolicy.configure(...), "
                f"or select one of {readers}."
            )
        if self.use_sim_policy_wrapper is None:
            self.use_sim_policy_wrapper = setup_cls.upstream_sim_policy_wrapper
        super()._post_configure()
        # After the setup's own action-term checks: the observation groups its pack_observations reads, then
        # the control rate that exec_horizon / video_stride (counted in env steps) assume.
        self._setup.check_env_observations(self.env)
        self._setup.check_env_control_rate(self.env)

    # ------------------------------------------------------------------
    # Wire format
    # ------------------------------------------------------------------

    def observe(self, obs: dict) -> None:
        """Refuse an observation dict without a group the setup reads, then run the setup's per-step hook.

        Construction checks the env publishes every group in ``observation_groups``; this catches a dict the caller
        built by hand (a dropped ``wrist_image``, a renamed group) on the step it arrives, naming the fix.
        """
        missing = [group for group in self._setup.observation_groups if group not in obs]
        if missing:
            raise KeyError(
                f"Gr00tPolicy.step() got observation groups {sorted(obs)}, without {missing}, which "
                f"policy_setup={self.policy_setup!r} reads. Fix: pass the dict env.reset() / env.step() return for an "
                f"env built from Gr00tPolicy.observation_options(policy_setup={self.policy_setup!r}). Do not fill a "
                "missing view with a copy of another camera: the checkpoint's schema has a distinct view per key."
            )
        super().observe(obs)

    def pack_observations(self, obs: dict) -> dict:
        payload = super().pack_observations(obs)
        return to_sim_wrapper_request(payload) if self.use_sim_policy_wrapper else payload

    def infer_payload(self, payload: Any) -> np.ndarray:
        """Round-trip as the base does; a server rejecting the request's format gets the flag to flip named."""
        try:
            return super().infer_payload(payload)
        except ServerError as e:
            hint = wire_format_hint(str(e), use_sim_policy_wrapper=self.use_sim_policy_wrapper)
            if hint is None:
                raise
            raise ServerError(f"{e}\n{hint}") from e
