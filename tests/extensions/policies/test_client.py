"""Tests for ``eden.extensions.policies.client``: the chunk scheduler, the server modality-config check
and the options surface shared by every server-backed client policy.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from eden.extensions.policies.adapters import WIDOWX_DOFS, JointVelocityAdapter
from eden.extensions.policies.base import PolicyBase
from eden.extensions.policies.client import (
    ActionChunkScheduler,
    ClientMixin,
    ModalitySchema,
    PolicySetupBase,
    SetupClientMixin,
    validate_modality_config,
)
from eden.options.extensions.policies import RemotePolicyOptions
from eden.utils.common import ConfigurableMixin
from tests.extensions.policies.conftest import policy_env
from tests.support.policy_servers import HttpPolicyServer

# ---------------------------------------------------------------------------
# ActionChunkScheduler: chunk queuing, horizons, temporal ensembling
# ---------------------------------------------------------------------------


def _chunk(t: int, d: int = 2, offset: float = 0.0) -> np.ndarray:
    return np.arange(t * d, dtype=np.float32).reshape(1, t, d) + offset


def test_client_mode_executes_exec_horizon_rows_then_needs_inference():
    s = ActionChunkScheduler(num_envs=1, pred_action_horizon=5, exec_horizon=3)
    assert s.needs_inference
    s.accept(_chunk(5))
    assert not s.needs_inference
    rows = [s.pop() for _ in range(3)]
    np.testing.assert_array_equal(np.stack(rows)[:, 0], _chunk(5)[0, :3])
    assert s.needs_inference
    assert s.action_metadata == {
        "chunk_index": 0,
        "chunk_action_index": 2,
        "predicted_chunk_length": 5,
        "executed_chunk_length": 3,
    }


def test_no_horizons_executes_the_whole_chunk():
    s = ActionChunkScheduler(num_envs=1, short_chunk="accept")
    s.accept(_chunk(4))
    assert [s.pop().shape for _ in range(4)] == [(1, 2)] * 4
    assert s.needs_inference


def test_short_chunk_error_and_accept():
    strict = ActionChunkScheduler(num_envs=1, pred_action_horizon=5, exec_horizon=2)
    with pytest.raises(RuntimeError, match="pred_action_horizon=5"):
        strict.accept(_chunk(3))
    lenient = ActionChunkScheduler(num_envs=1, pred_action_horizon=5, exec_horizon=4, short_chunk="accept")
    lenient.accept(_chunk(3))
    assert len(lenient.queues[0]) == 3  # min(exec, T)


@pytest.mark.parametrize("mode", ["server", "single"])
def test_server_and_single_modes_always_need_inference(mode):
    s = ActionChunkScheduler(num_envs=1, mode=mode, short_chunk="accept")
    s.accept(_chunk(1))
    assert s.needs_inference  # even with a row queued
    s.pop()
    with pytest.raises(RuntimeError, match="No queued action"):
        s.pop()


@pytest.mark.parametrize("mode", ["server", "single"])
def test_server_and_single_modes_reject_multi_row_chunks(mode):
    s = ActionChunkScheduler(num_envs=1, mode=mode, short_chunk="accept")
    with pytest.raises(ValueError, match="one action per inference call"):
        s.accept(_chunk(4))
    assert not s.queues[0]


def test_temporal_ensemble_blends_previous_tail_into_new_head_and_spares_dims():
    """Matches PiPolicy's ensembling: weights K/(K+1) .. 1/(K+1) on the old tail."""
    s = ActionChunkScheduler(num_envs=1, pred_action_horizon=5, exec_horizon=3, ensemble_steps=2, ensemble_dims=1)
    first = _chunk(5)  # rows 0..4; unused tail = rows 3, 4
    s.accept(first)
    for _ in range(3):
        s.pop()
    second = _chunk(5, offset=100.0)
    s.accept(second)
    row0, row1, row2 = (s.pop()[0] for _ in range(3))
    w = np.array([2 / 3, 1 / 3], dtype=np.float32)
    # dim 0 is ensembled, dim 1 (beyond ensemble_dims) is the new prediction unchanged
    assert row0[0] == pytest.approx(w[0] * first[0, 3, 0] + (1 - w[0]) * second[0, 0, 0])
    assert row1[0] == pytest.approx(w[1] * first[0, 4, 0] + (1 - w[1]) * second[0, 1, 0])
    assert row0[1] == second[0, 0, 1] and row1[1] == second[0, 1, 1]
    np.testing.assert_array_equal(row2, second[0, 2])
    assert s.last_temporal_ensemble_steps == 2


def test_clear_drops_queue_and_ensemble_tail():
    s = ActionChunkScheduler(num_envs=1, pred_action_horizon=5, exec_horizon=3, ensemble_steps=2)
    s.accept(_chunk(5))
    s.clear()
    assert s.needs_inference and s.chunk_index == -1
    s.accept(_chunk(5, offset=7.0))
    np.testing.assert_array_equal(s.pop()[0], _chunk(5, offset=7.0)[0, 0])  # no blending after clear


def test_raw_chunk_is_queued_alongside_for_diagnostics():
    s = ActionChunkScheduler(num_envs=1, short_chunk="accept")
    latched = _chunk(2)
    raw = _chunk(2, offset=0.5)
    s.accept(latched, raw_chunk=raw)
    s.pop()
    np.testing.assert_array_equal(s.last_raw_row, raw[:, 0])


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        (dict(mode="bogus"), "chunk_mode"),
        (dict(short_chunk="maybe"), "short_chunk"),
        (dict(pred_action_horizon=2, exec_horizon=3), "exec_horizon"),
        (dict(ensemble_steps=1), "requires both"),
        (dict(pred_action_horizon=5, exec_horizon=4, ensemble_steps=2), "exceeds"),
        (dict(ensemble_dims=0), "ensemble_dims"),
    ],
)
def test_invalid_configuration_fails_at_construction(kwargs, match):
    with pytest.raises(ValueError, match=match):
        ActionChunkScheduler(num_envs=1, **kwargs)


def test_chunk_shape_is_validated():
    s = ActionChunkScheduler(num_envs=1, short_chunk="accept")
    with pytest.raises(ValueError, match=r"\(1, T, D\)"):
        s.accept(np.zeros((2, 3, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="empty"):
        s.accept(np.zeros((1, 0, 4), dtype=np.float32))


# ---------------------------------------------------------------------------
# validate_modality_config: what the server advertises vs. the setup's schema
# ---------------------------------------------------------------------------


def _gr00t_config(*, video, state, action, language, action_len=8, video_delta=(0,)):
    return {
        "video": {"modality_keys": list(video), "delta_indices": list(video_delta)},
        "state": {"modality_keys": list(state), "delta_indices": [0]},
        "action": {"modality_keys": list(action), "delta_indices": list(range(action_len))},
        "language": {"modality_keys": [language], "delta_indices": [0]},
    }


_WIDOWX_SCHEMA = ModalitySchema(
    video_keys=("image_0",),
    state_keys=("x", "y", "z", "roll", "pitch", "yaw", "pad", "gripper"),
    action_keys=("x", "y", "z", "roll", "pitch", "yaw", "gripper"),
    language_key="annotation.human.action.task_description",
    action_horizon=8,
)


def test_matching_modality_config_passes_and_empty_config_is_skipped():
    cfg = _gr00t_config(
        video=["image_0"],
        state=["x", "y", "z", "roll", "pitch", "yaw", "pad", "gripper"],
        action=["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
        language="annotation.human.action.task_description",
    )
    validate_modality_config(cfg, _WIDOWX_SCHEMA, setup_name="widowx")
    validate_modality_config({}, _WIDOWX_SCHEMA, setup_name="widowx")


@pytest.mark.parametrize(
    ("override", "match"),
    [
        (dict(video=["primary"]), "video keys"),
        (dict(state=["x", "y", "z"]), "state keys"),
        (dict(action=["gripper", "x", "y", "z", "roll", "pitch", "yaw"]), "order matters"),
        (dict(action_len=16), "action horizon"),
        (dict(language="annotation.language.language_instruction"), "language key"),
    ],
)
def test_each_mismatch_is_named(override, match):
    base = dict(
        video=["image_0"],
        state=["x", "y", "z", "roll", "pitch", "yaw", "pad", "gripper"],
        action=["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
        language="annotation.human.action.task_description",
    )
    base.update(override)
    with pytest.raises(ValueError, match=match):
        validate_modality_config(_gr00t_config(**base), _WIDOWX_SCHEMA, setup_name="widowx")


def test_video_window_horizon_and_stride_are_checked():
    schema = ModalitySchema(
        video_keys=("primary", "wrist"),
        state_keys=("s",),
        action_keys=("a",),
        language_key="l",
        video_horizon=4,
        video_stride=2,
    )
    ok = _gr00t_config(video=["primary", "wrist"], state=["s"], action=["a"], language="l", video_delta=(-6, -4, -2, 0))
    validate_modality_config(ok, schema, setup_name="rldx")
    with pytest.raises(ValueError, match="video horizon"):
        validate_modality_config(
            _gr00t_config(video=["primary", "wrist"], state=["s"], action=["a"], language="l", video_delta=(-15, 0)),
            schema,
            setup_name="rldx",
        )
    with pytest.raises(ValueError, match="video stride"):
        validate_modality_config(
            _gr00t_config(
                video=["primary", "wrist"], state=["s"], action=["a"], language="l", video_delta=(-9, -6, -3, 0)
            ),
            schema,
            setup_name="rldx",
        )


def test_malformed_config_names_missing_modalities():
    with pytest.raises(ValueError, match="no 'action' modality"):
        validate_modality_config({"video": {"modality_keys": ["image_0"]}}, _WIDOWX_SCHEMA, setup_name="widowx")


# ---------------------------------------------------------------------------
# Options surface
# ---------------------------------------------------------------------------


class _OnlySetup(PolicySetupBase):
    @classmethod
    def observation_options(cls, **kwargs):
        return {}

    @classmethod
    def action_options(cls, **kwargs):
        return {}

    def pack_observations(self, obs):
        return obs


class _OtherSetup(_OnlySetup):
    pass


class _SetupPolicy(SetupClientMixin, PolicyBase, ConfigurableMixin[RemotePolicyOptions]):
    SETUP_REGISTRY = {"only": _OnlySetup, "other": _OtherSetup}
    policy_setup: str = "only"


def test_class_level_constants_are_not_configure_parameters():
    names = _SetupPolicy.get_parameter_names()
    assert {"SETUP_REGISTRY", "supports_batch", "action_dim"}.isdisjoint(names)
    json.loads(_SetupPolicy.configure().model_dump_json())  # the options serialise
    with pytest.raises(ValueError, match="supports_batch"):
        _SetupPolicy.configure(supports_batch=True)


def test_an_unknown_policy_setup_lists_the_clients_setups():
    with pytest.raises(
        ValueError,
        match=r"_SetupPolicy.policy_setup='nope' is not a setup of this client. Available: \['only', 'other'\]",
    ):
        _SetupPolicy.action_options(policy_setup="nope")


# ---------------------------------------------------------------------------
# ClientMixin lifecycle: the smallest concrete client on a real env, against a local HTTP server
# ---------------------------------------------------------------------------


class _EchoPolicy(ClientMixin, PolicyBase, ConfigurableMixin[RemotePolicyOptions]):
    """The payload is the observation dict; a server action row passes through as ``ctrl``."""

    short_chunk: str = "accept"

    @classmethod
    def action_options(cls):
        return {}

    @classmethod
    def observation_options(cls):
        return {}

    def pack_observations(self, obs):
        return {"obs": obs}

    def unpack_actions(self, raw_action):
        return {"ctrl": torch.as_tensor(raw_action)}


class _RoutedPolicy(_EchoPolicy):
    """A client whose server exposes /set_task and /reset (open-pi-zero / Octo style)."""

    set_task_endpoint: str | None = "/set_task"
    reset_endpoint: str | None = "/reset"


def _client(cls, server, *, num_envs: int = 1, **configure):
    env = policy_env("widowx", {}, num_envs=num_envs)
    return cls(env, cls.configure(host=server.host, port=server.port, **configure))


def _routes(server) -> list[str]:
    return [record["url"] for record in server.records]


def test_unknown_transport_key_fails_at_construction_listing_available():
    with pytest.raises(ValueError, match="Available"):
        _EchoPolicy(policy_env("widowx", {}), _EchoPolicy.configure(transport="carrier_pigeon"))


def test_full_url_host_with_port_none_is_used_verbatim():
    env = policy_env("widowx", {})
    policy = _EchoPolicy(env, _EchoPolicy.configure(host="https://example.com/api/", port=None))
    assert policy._transport.base_url == "https://example.com/api"
    with pytest.raises(ValueError, match="embeds port 9000"):
        _EchoPolicy(env, _EchoPolicy.configure(host="http://node:9000", port=8000))


def test_set_task_reset_and_step_reach_the_servers_routes():
    with HttpPolicyServer(lambda call, record: {"action": [[1.0, 2.0]]}) as server:
        policy = _client(_RoutedPolicy, server, reset_payload_key="instruction")
        policy.set_task("do it")
        policy.reset_server()
        out = policy.step({"x": 1})

    assert server.records == [
        {"url": "set_task", "json": {"task_description": "do it"}},
        {"url": "reset", "json": {"instruction": "do it"}},
        {"url": "act", "json": {"obs": {"x": 1}}},
    ]
    torch.testing.assert_close(out["ctrl"], torch.tensor([[1.0, 2.0]]))
    assert policy.last_response_info == {}


def test_default_client_has_no_server_reset_or_set_task_api():
    with HttpPolicyServer() as server:
        policy = _client(_EchoPolicy, server)
        policy.set_task("t")
        policy.reset_server()
        policy.reset()
    assert server.records == [] and policy.task_description == "t"


def test_last_response_info_keeps_everything_but_the_action_chunk():
    with HttpPolicyServer(lambda call, record: {"action": [[0.0]], "trace": [1, 2], "dt_ms": 3.0}) as server:
        policy = _client(_EchoPolicy, server)
        policy.step({})
    assert policy.last_response_info == {"trace": [1, 2], "dt_ms": 3.0}


def test_missing_action_key_names_the_key_and_the_reply():
    with HttpPolicyServer(lambda call, record: {"actions": [[0.0]]}) as server:
        policy = _client(_EchoPolicy, server)
        with pytest.raises(KeyError, match="response_action_key"):
            policy.step({})


def test_reset_sends_a_directly_assigned_task_description():
    with HttpPolicyServer() as server:
        policy = _client(_RoutedPolicy, server)
        policy.task_description = "open the drawer"
        policy.reset_server()
    assert server.records == [{"url": "reset", "json": {"task_description": "open the drawer"}}]


def test_reset_resets_the_server_only_when_every_served_env_resets():
    with HttpPolicyServer() as server:
        policy = _client(_RoutedPolicy, server)
        policy.reset(torch.tensor([0]))
        policy.reset(None)
        policy.reset(torch.tensor([], dtype=torch.long))  # nothing done: no server call
    assert _routes(server) == ["reset", "reset"]


def test_a_batched_client_without_a_per_env_reset_is_refused_at_construction():
    """The env resets its envs one by one, so the shared-state default reset would raise at the first episode end."""

    class _Batched(_RoutedPolicy):
        supports_batch = True

    with HttpPolicyServer() as server:
        with pytest.raises(ValueError, match="serves 3 envs but keeps ClientMixin.reset"):
            _client(_Batched, server, num_envs=3)
    assert server.records == []


def test_the_default_reset_still_refuses_a_partial_reset_it_is_handed():
    class _Batched(_RoutedPolicy):
        supports_batch = True

        def reset(self, envs_idx=None):  # a per-env scheme that defers whole-batch resets to the default
            super().reset(envs_idx)

    with HttpPolicyServer() as server:
        policy = _client(_Batched, server, num_envs=3)
        with pytest.raises(NotImplementedError, match="1 of 3 envs"):
            policy.reset(torch.tensor([1]))
        assert server.records == []  # the other envs' server state is left alone
        policy.reset(torch.tensor([True, True, True]))
    assert _routes(server) == ["reset"]


@pytest.mark.parametrize(
    ("envs_idx", "match"),
    [(torch.tensor([3]), "out of range"), ([-1], "out of range"), (torch.tensor([True, False]), "entries for 1 envs")],
)
def test_reset_rejects_invalid_indices(envs_idx, match):
    with HttpPolicyServer() as server:
        policy = _client(_RoutedPolicy, server)
        with pytest.raises(ValueError, match=match):
            policy.reset(envs_idx)
    assert server.records == []


def test_a_client_with_a_fixed_row_width_rejects_a_chunk_of_another_width():
    class _TwoWide(_EchoPolicy):
        action_dim = 2

    assert "action_dim" not in _TwoWide.get_parameter_names()
    with HttpPolicyServer(lambda call, record: {"action": [[0.0, 0.0, 0.0]]}) as server:
        policy = _client(_TwoWide, server)
        with pytest.raises(ValueError, match="_TwoWide expects 2-D action rows, the server returned 3-D"):
            policy.step({})


@pytest.mark.parametrize(("field", "value"), [("chunk_mode", "bogus"), ("short_chunk", "maybe")])
def test_chunk_options_are_validated_at_configure(field, value):
    with pytest.raises(ValueError, match=field):
        _EchoPolicy.configure(**{field: value})


# ---------------------------------------------------------------------------
# SetupClientMixin: adapter binding and the raw-chunk hand-off
# ---------------------------------------------------------------------------


class _IntegratingSetup(_OnlySetup):
    """Integrates joint velocities on the WidowX arm, with one joint name misspelled."""

    def __init__(self, policy):
        super().__init__(policy)
        self.adapter = JointVelocityAdapter(
            n_arm=6, arm_dofs_name=[*WIDOWX_DOFS[:-1], "wrist_rotat"], mode="velocity_integrate"
        )


class _IntegratingPolicy(_SetupPolicy):
    SETUP_REGISTRY = {"integrating": _IntegratingSetup}
    policy_setup: str = "integrating"


def test_a_setups_adapter_resolves_its_joints_when_the_policy_is_built():
    """A misspelled arm joint fails at construction, naming it, not on the first integrated step."""
    with HttpPolicyServer() as server:
        with pytest.raises(ValueError, match="wrist_rotat"):
            _client(_IntegratingPolicy, server)
    assert server.records == []


class _RawChunkSetup(_OnlySetup):
    """Keeps the untouched server rows for the first reply only, as a setup post-processing that reply would."""

    def decode_response(self, response):
        chunk = np.asarray(response["action"], dtype=np.float32)
        if response.get("keep_raw"):
            self.last_raw_action_chunk = chunk + 100.0
        return chunk

    def unpack_actions(self, raw_action):
        return {"ctrl": torch.as_tensor(raw_action)}


class _RawChunkPolicy(_SetupPolicy):
    SETUP_REGISTRY = {"raw": _RawChunkSetup}
    policy_setup: str = "raw"


def test_a_setups_raw_chunk_is_handed_over_once_and_never_paired_with_a_later_chunk():
    replies = {1: {"action": [[1.0]], "keep_raw": True}, 2: {"action": [[2.0], [3.0]]}}
    with HttpPolicyServer(lambda call, record: replies[call]) as server:
        policy = _client(_RawChunkPolicy, server)
        policy.step({})
        assert policy._scheduler.last_raw_row[0].tolist() == [101.0]
        assert policy._setup.last_raw_action_chunk is None
        policy.step({})  # a two-row chunk: the stale one-row raw chunk would fail the shape check
    assert policy._scheduler.last_raw_row[0].tolist() == [2.0]
