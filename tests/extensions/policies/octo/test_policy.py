"""``OctoPolicy`` against a local HTTP server standing in for the Octo policy server."""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pytest
import torch
from scipy.spatial.transform import Rotation

import eden as en
from eden.envs.base import EnvBase
from eden.extensions.policies.octo import OctoPolicy
from eden.options import EnvOptions
from eden.options.managers.actions import ActionManagerOptions
from eden.utils.configs import EdenConfig
from tests.extensions.policies.conftest import GOLDEN_TASK, assert_matches_golden, policy_env, record_golden
from tests.extensions.policies.conftest import image as golden_image
from tests.support.envs import build_env
from tests.support.policy_servers import HttpPolicyServer


def _config(num_envs: int) -> EdenConfig:
    """A WidowX built with Octo's own action terms: the policy refuses an env without its gripper term."""
    return EdenConfig(
        env_options=EnvOptions(num_envs=num_envs),
        scene_options=en.scenes.Empty(robot=en.robots.WidowX250()),
        action_options=ActionManagerOptions(**OctoPolicy.action_options()),
    )


@pytest.fixture(scope="module")
def env():
    return build_env(_config(num_envs=1))


class _OctoServer(ThreadingHTTPServer):
    """Records every request and answers from ``status`` / ``act_response`` / ``delay_s``."""

    def __init__(self):
        super().__init__(("127.0.0.1", 0), _Handler)
        self.requests: list[dict] = []
        self.status = 200
        self.delay_s = 0.0
        self.act_response: dict = {}


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802 — http.server's spelling
        url = urlparse(self.path)
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else b""
        self.server.requests.append(
            {"path": url.path, "query": parse_qs(url.query), "json": json.loads(body) if body else None}
        )
        time.sleep(self.server.delay_s)
        payload = json.dumps(self.server.act_response if url.path == "/act" else {}).encode()
        self.send_response(self.server.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


@pytest.fixture
def server():
    srv = _OctoServer()
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv
    srv.shutdown()
    srv.server_close()


@pytest.fixture
def policy(env, server):
    host, port = server.server_address
    return OctoPolicy(env=env, options=OctoPolicy.configure(host=host, port=port, request_timeout=0.5))


def test_configure_accepts_both_the_client_and_the_policy_fields(env):
    """Without the merged ``RemotePolicyOptions`` base one parent's fields would be rejected."""
    assert OctoPolicy.configure(checkpoint_path="some/ckpt.pt").checkpoint_path == "some/ckpt.pt"
    policy = OctoPolicy(env=env, options=OctoPolicy.configure(host="h.example", port=8080))
    assert (policy.host, policy.port) == ("h.example", 8080)


def test_more_than_one_env_is_refused():
    """The Octo server takes one image per request (``supports_batch=False``)."""
    env = EnvBase.from_config(_config(num_envs=2), show_viewer=False)  # construction reads num_envs; no build needed
    with pytest.raises(ValueError, match="num_envs"):
        OctoPolicy(env=env, options=OctoPolicy.configure())


def test_set_task_sends_the_description_as_a_query_parameter(policy, server):
    policy.set_task("pick up the cube")
    (request,) = server.requests
    assert request == {"path": "/set_task", "query": {"task_description": ["pick up the cube"]}, "json": None}
    assert policy.task_description == "pick up the cube"


def test_step_posts_the_image_and_instruction_and_routes_the_action(policy, server):
    """World vector + rotation delta drive the EE, the Euler delta as the rotation vector the IK controller
    solves for (what the server's own rot_axangle carries at action_scale=1); the gripper sign is forwarded
    raw (server +1 = open) for the env's GripperDecisionController(input="open_sign") to read.
    """
    server.act_response = {
        "raw_action": {"world_vector": [0.1, 0.2, 0.3], "rotation_delta": [0.4, 0.5, 0.6], "open_gripper": [1.0]},
        "action": {"world_vector": [0.1, 0.2, 0.3], "rot_axangle": [0.0] * 3, "gripper": [1.0]},
    }
    policy.task_description = "do thing"
    image = torch.zeros(1, 2, 2, 3, dtype=torch.uint8)
    image[0, 0, 0, 0] = 255

    out = policy.step({"image": {"rgb": image}})

    (request,) = server.requests
    assert request["path"] == "/act"
    assert request["json"]["instruction"] == "do thing"
    assert request["json"]["image"] == [[[255, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]]]  # batch axis dropped
    rotvec = Rotation.from_euler("xyz", [0.4, 0.5, 0.6]).as_rotvec().tolist()
    torch.testing.assert_close(out["ee_controller"].cpu(), torch.tensor([[0.1, 0.2, 0.3, *rotvec]]), atol=1e-6, rtol=0)
    torch.testing.assert_close(out["gripper_controller"].cpu(), torch.tensor([[1.0]]))


def test_a_reply_without_raw_action_names_the_layout_and_the_wrong_fallback(policy, server):
    """The processed action dict is scaled by the server's action_scale and carries axis-angle, so it is not a
    stand-in for raw_action; the error shows what the reply did carry.
    """
    server.act_response = {"action": {"world_vector": [0.1, 0.2, 0.3], "rot_axangle": [0.0] * 3, "gripper": [1.0]}}
    policy.task_description = "do thing"
    with pytest.raises(KeyError, match=r"got \{'action': \['world_vector', 'rot_axangle', 'gripper'\]\}") as e:
        policy.step({"image": {"rgb": torch.zeros(1, 2, 2, 3, dtype=torch.uint8)}})
    assert "Do not fall back to action['world_vector'] / action['rot_axangle']" in str(e.value)


def test_an_unbatched_image_is_sent_as_is(policy):
    payload = policy.pack_observations({"image": torch.zeros(2, 3, 3, dtype=torch.uint8)})
    assert len(payload["image"]) == 2 and len(payload["image"][0]) == 3


def test_an_rgba_render_is_sent_as_rgb_and_a_float_image_is_refused(policy):
    """The madrona/Nyx renderers return RGBA; a float image would be truncated by a uint8 cast, so it raises."""
    rgba = torch.full((1, 2, 2, 4), 7, dtype=torch.uint8)
    assert policy.pack_observations({"image": rgba})["image"] == [[[7, 7, 7]] * 2] * 2
    with pytest.raises(TypeError, match="Expected a uint8 image, got dtype float32"):
        policy.pack_observations({"image": torch.full((1, 2, 2, 3), 0.5)})


def test_an_http_error_is_a_runtime_error(policy, server):
    server.status = 500
    with pytest.raises(RuntimeError, match="Policy server request"):
        policy.set_task("x")


def test_a_slow_server_is_a_connection_error(policy, server):
    server.delay_s = 1.0
    # A server that is up but slow is told apart from an absent one: the message names the timeout to raise.
    with pytest.raises(ConnectionError, match="did not answer within request_timeout"):
        policy.set_task("x")


def test_an_absent_server_is_a_connection_error(env, server):
    host, port = server.server_address
    server.shutdown()
    server.server_close()
    policy = OctoPolicy(env=env, options=OctoPolicy.configure(host=host, port=port, request_timeout=0.5))
    with pytest.raises(ConnectionError, match="Cannot reach policy server"):
        policy.set_task("x")


def test_observation_options_wire_the_camera_and_drop_unknown_kwargs():
    """Polymorphic call sites pass ``entity_name`` and more; the camera name must still reach the term."""
    options = OctoPolicy.observation_options(camera_name="my_cam", entity_name="r", spurious=42)
    assert options["image"].rgb.params["camera_name"] == "my_cam"
    assert set(OctoPolicy.action_options()) == {"ee_controller", "gripper_controller"}


def _golden_reply(call, record):
    world_vector, rotation = [0.01 * call, -0.02, 0.03], [0.0, 0.1 * call, -0.05]
    gripper = [1.0 if call % 2 else -1.0]
    return {
        "raw_action": {"world_vector": world_vector, "rotation_delta": rotation, "open_gripper": gripper},
        "action": {
            "world_vector": world_vector,
            "rot_axangle": rotation,
            "gripper": gripper,
            "terminate_episode": [0.0],
        },
    }


def test_wire_contract_and_controller_outputs_match_the_golden_record():
    with HttpPolicyServer(_golden_reply) as server:
        env = policy_env("widowx", OctoPolicy.action_options())
        policy = OctoPolicy(env, OctoPolicy.configure(host=server.host, port=server.port))
        policy.set_task(GOLDEN_TASK)
        got = record_golden(policy, server, lambda s: {"image": {"rgb": golden_image(h=4, w=4, seed=s)}}, steps=3)
    assert_matches_golden("octo", got)


def test_the_client_declares_exactly_its_official_servers_reset_api():
    # RobotArena's server_octo.py serves /set_task and /reset.
    options = OctoPolicy.configure()
    assert (options.set_task_endpoint, options.reset_endpoint) == ("/set_task", "/reset")
