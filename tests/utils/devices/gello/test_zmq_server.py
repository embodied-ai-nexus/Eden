"""``GelloZMQServer`` over a built Panda follower, with an in-memory fake of ``pyzmq`` as the transport.

pyzmq is an optional dependency (and absent offline), so the module's ``zmq`` binding is swapped for
:class:`FakeZmq`: REQ/REP sockets that really deliver messages between threads. Requests go through
the server's own serve thread, so a test sees exactly what a GELLO client would.
"""

from __future__ import annotations

import pickle
import queue

import numpy as np
import pytest
import torch

import eden.utils.devices.gello.zmq_server as gello_zmq_module
from eden.options import EnvOptions, SceneOptions
from eden.options.robots import FrankaEmikaPanda
from eden.utils.configs import EdenConfig
from eden.utils.devices.gello.zmq_server import GelloZMQServer
from tests.support.envs import build_env

ARM_DOFS = [f"joint{i}" for i in range(1, 8)]
GRIPPER_DOFS = ["finger_joint1", "finger_joint2"]
ARM_POS = [0.1, 0.2, 0.3, -1.4, 0.5, 1.6, 0.7]
ARM_VEL = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07]
#: Panda fingers open at 0.04 and close at 0.0, so 0.01 each is 3/4 of the way closed.
FINGER_POS = [0.01, 0.01]
GRIPPER_POSITION = 0.75


# ---------------------------------------------------------------------------
# in-memory pyzmq
# ---------------------------------------------------------------------------


class FakeZmq:
    """The slice of the ``zmq`` module the server uses: contexts, REQ/REP sockets, two exceptions."""

    REQ, REP, LINGER, RCVTIMEO, SNDTIMEO = "REQ", "REP", "LINGER", "RCVTIMEO", "SNDTIMEO"

    class Again(Exception):
        """``zmq.Again``: a receive timed out."""

    class ZMQError(Exception):
        """``zmq.ZMQError``: the socket is gone."""

    def __init__(self):
        self.bound: dict[str, _Socket] = {}
        self.contexts: list[_Context] = []
        self.bind_error: Exception | None = None

    def Context(self):  # noqa: N802 — pyzmq's spelling
        context = _Context(self)
        self.contexts.append(context)
        return context


class _Context:
    def __init__(self, zmq: FakeZmq):
        self.zmq = zmq
        self.sockets: list[_Socket] = []
        self.terminated = False

    def socket(self, kind):
        sock = _Socket(self.zmq, kind)
        self.sockets.append(sock)
        return sock

    def term(self):
        self.terminated = True


class _Socket:
    def __init__(self, zmq: FakeZmq, kind: str):
        self.zmq, self.kind = zmq, kind
        self.options: dict[str, int] = {}
        self.inbox: queue.Queue = queue.Queue()
        self.endpoint: str | None = None
        self.closed = False
        self._reply_to: _Socket | None = None

    def setsockopt(self, option, value):
        self.options[option] = value

    def bind(self, endpoint):
        if self.zmq.bind_error is not None:
            raise self.zmq.bind_error
        if endpoint in self.zmq.bound:
            raise FakeZmq.ZMQError(f"Address already in use: {endpoint}")
        self.zmq.bound[endpoint] = self
        self.endpoint = endpoint

    def connect(self, endpoint):
        self.endpoint = endpoint

    def send(self, payload: bytes):
        if self.kind == FakeZmq.REQ:
            peer = self.zmq.bound.get(self.endpoint)
            if peer is not None:
                peer.inbox.put((payload, self))
        else:
            self._reply_to.inbox.put((payload, None))

    def recv(self) -> bytes:
        if self.closed:
            raise FakeZmq.ZMQError("socket closed")
        try:
            payload, sender = self.inbox.get(timeout=self.options.get(FakeZmq.RCVTIMEO, 5000) / 1000)
        except queue.Empty:
            raise FakeZmq.Again() from None
        self._reply_to = sender
        return payload

    def close(self, linger=None):
        self.closed = True
        if self.zmq.bound.get(self.endpoint) is self:
            del self.zmq.bound[self.endpoint]


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def env():
    env = build_env(
        EdenConfig(env_options=EnvOptions(num_envs=1), scene_options=SceneOptions(robot=FrankaEmikaPanda()))
    )
    robot = env.entities["robot"]
    _, arm_idx = robot.find_named_dofs_idx_local(ARM_DOFS, preserve_order=True)
    _, gripper_idx = robot.find_named_dofs_idx_local(GRIPPER_DOFS, preserve_order=True)
    robot.set_dofs_pos(torch.tensor([ARM_POS + FINGER_POS]), dofs_idx_local=[*arm_idx, *gripper_idx])
    robot.set_dofs_vel(torch.tensor([ARM_VEL]), dofs_idx_local=arm_idx)
    return env


@pytest.fixture
def zmq(monkeypatch):
    fake = FakeZmq()
    monkeypatch.setattr(gello_zmq_module, "ZMQ_AVAILABLE", True)
    monkeypatch.setattr(gello_zmq_module, "zmq", fake, raising=False)
    return fake


@pytest.fixture
def make_server(env, zmq):
    servers: list[GelloZMQServer] = []

    def make(*, gripper: bool = True, **kwargs) -> GelloZMQServer:
        if gripper:
            kwargs.update(gripper_entity_name="robot", gripper_dofs_name=GRIPPER_DOFS)
        server = GelloZMQServer(env, "robot", ARM_DOFS, "hand", host="127.0.0.1", port=6001, **kwargs)
        servers.append(server)
        return server

    yield make
    for server in servers:
        server.stop()


def _rpc(zmq: FakeZmq, server: GelloZMQServer, request) -> object:
    """One request/reply round trip, as a GELLO client makes it."""
    client = zmq.Context().socket(zmq.REQ)
    client.connect(server.endpoint)
    client.send(pickle.dumps(request))
    return pickle.loads(client.recv())


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------


def test_requires_pyzmq(env, monkeypatch):
    monkeypatch.setattr(gello_zmq_module, "ZMQ_AVAILABLE", False)
    with pytest.raises(ImportError, match="pyzmq"):
        GelloZMQServer(env, "robot", ARM_DOFS, "hand")


def test_the_gripper_needs_exactly_two_dofs(make_server):
    with pytest.raises(AssertionError, match="2 DOFs"):
        make_server(gripper=False, gripper_entity_name="robot", gripper_dofs_name=["finger_joint1"])


def test_endpoint_is_tcp_host_port(make_server):
    server = make_server()
    assert (server.host, server.port, server.endpoint) == ("127.0.0.1", 6001, "tcp://127.0.0.1:6001")


# ---------------------------------------------------------------------------
# the RPC surface
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("gripper, num_dofs", [(True, 8), (False, 7)])
def test_observations_report_the_followers_state(env, zmq, make_server, gripper, num_dofs):
    """The gripper is one normalised DOF: 0 open, 1 closed, from the entity's open/close poses."""
    server = make_server(gripper=gripper)
    server.start()

    assert _rpc(zmq, server, {"method": "num_dofs"}) == num_dofs
    obs = _rpc(zmq, server, {"method": "get_observations"})
    hand = env.entities["robot"].get_link("hand")
    ee_pose = np.concatenate(
        [hand.get_pos(relative=False)[0].cpu().numpy(), hand.get_quat(relative=False)[0].cpu().numpy()]
    )
    np.testing.assert_allclose(obs["ee_pos_quat"], ee_pose, atol=1e-6)
    if gripper:
        np.testing.assert_allclose(obs["joint_positions"], [*ARM_POS, GRIPPER_POSITION], atol=1e-5)
        np.testing.assert_allclose(obs["joint_velocities"], [*ARM_VEL, 0.0], atol=1e-5)
        assert float(obs["gripper_position"]) == pytest.approx(GRIPPER_POSITION, abs=1e-5)
    else:
        np.testing.assert_allclose(obs["joint_positions"], ARM_POS, atol=1e-5)
        assert obs["gripper_position"] is None
    np.testing.assert_allclose(_rpc(zmq, server, {"method": "get_joint_positions"}), obs["joint_positions"])


def test_the_command_starts_at_the_follower_pose_and_follows_the_leader(zmq, make_server):
    server = make_server()
    np.testing.assert_allclose(server.get_command().numpy(), [*ARM_POS, GRIPPER_POSITION], atol=1e-5)

    server.start()
    command = np.array([1.0, 0.8, 0.6, 0.4, 0.2, 0.0, 0.5, 0.3], dtype=np.float32)
    assert _rpc(zmq, server, {"method": "command_joint_state", "args": {"joint_state": command}}) is None
    np.testing.assert_array_equal(server.get_command().numpy(), command)
    np.testing.assert_array_equal(server.get_raw_data().numpy(), command)


@pytest.mark.parametrize(
    "request_, error",
    [
        (["not", "a", "dict"], "must be a dict"),
        ({"method": "unknown"}, "Invalid method: unknown"),
        ({"method": "command_joint_state"}, "'args' field is required"),
        ({"method": "command_joint_state", "args": {}}, "'joint_state' argument is required"),
        ({"method": "command_joint_state", "args": {"joint_state": np.zeros(7)}}, "must have length 8"),
        ({"method": "command_joint_state", "args": {"joint_state": [0, 1, 2, 3, 4, 5, 6, np.nan]}}, "finite"),
    ],
)
def test_a_bad_request_is_answered_with_its_error(zmq, make_server, eden_caplog, request_, error):
    """The serve loop answers every request, so a malformed one cannot wedge the client's REQ socket."""
    server = make_server(verbose=True)
    server.start()
    response = _rpc(zmq, server, request_)
    assert set(response) == {"error"} and error in response["error"]
    assert f"Gello RPC processing error: {response['error']}" in eden_caplog.text
    np.testing.assert_allclose(server.get_command().numpy(), [*ARM_POS, GRIPPER_POSITION], atol=1e-5)


# ---------------------------------------------------------------------------
# lifecycle
# ---------------------------------------------------------------------------


def test_start_is_idempotent_stop_is_final(zmq, make_server):
    server = make_server()
    server.start()
    server.start()  # already running: no second bind
    assert list(zmq.bound) == [server.endpoint]
    rep = zmq.bound[server.endpoint]
    assert rep.kind == zmq.REP and rep.options[zmq.LINGER] == 0

    server.stop()
    server.stop()
    assert zmq.bound == {}, "the REP socket outlived stop()"
    assert all(ctx.terminated for ctx in zmq.contexts) and all(s.closed for c in zmq.contexts for s in c.sockets)
    with pytest.raises(RuntimeError, match="cannot be restarted"):
        server.start()


def test_a_failed_bind_fails_start_and_releases_the_socket(zmq, make_server):
    zmq.bind_error = OSError("port busy")
    server = make_server()
    with pytest.raises(RuntimeError, match="Failed to start Gello RPC server") as exc:
        server.start()
    assert isinstance(exc.value.__cause__, OSError)
    (context,) = zmq.contexts
    assert context.terminated and all(sock.closed for sock in context.sockets)
