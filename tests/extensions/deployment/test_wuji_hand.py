"""``WujiHandDeployment`` on a built Wuji hand env, over a scripted transport; the SDK transports over fake SDKs.

Two properties of the hands shape this backend: both SDKs speak 20 angles in one finger-major order
whatever the side, so the backend scatters the deployment DOF order into it; and gains are
programmed once at connect, so every option that acts through per-command gains is refused.
"""

from __future__ import annotations

import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from eden.extensions.deployment import resolve_deployment_for
from eden.extensions.deployment import wuji_hand as wh
from eden.extensions.deployment.utils.state import RobotCommand
from eden.options.robots.wuji_hand1 import WujiHand1_L, WujiHand1_R
from eden.options.robots.wuji_hand2beta2 import WujiHand2Beta2_L, WujiHand2Beta2_R, WujiHand2Beta2WithMount_L
from tests.support.deploy import ScriptedHandTransport, assert_dofs_looked_up_by_name, deployments, robot_env

N = wh.NUM_JOINTS
# Reversed: both naming schemes are declared in SDK order, so only a permuted deployment order makes
# the scatter observable.
H1_DOFS = list(reversed(WujiHand1_R().dofs_name))
H2_DOFS = list(reversed(WujiHand2Beta2_L().dofs_name))


def _command(dofs_pos) -> RobotCommand:
    zeros = np.zeros(N)
    return RobotCommand(dofs_pos=np.asarray(dofs_pos, dtype=np.float64), dofs_vel=zeros, dofs_torque=zeros)


@pytest.fixture(scope="module")
def envs():
    """``envs(generation)``: a right Hand 1 or a left Hand 2 env, built on first use."""
    robots = {"hand1": lambda: WujiHand1_R(dofs_name=H1_DOFS), "hand2": lambda: WujiHand2Beta2_L(dofs_name=H2_DOFS)}
    built = {}

    def get(generation):
        if generation not in built:
            built[generation] = robot_env(robots[generation]())
        return built[generation]

    return get


def _deployer(generation):
    @pytest.fixture(name=generation)
    def fixture(envs):
        """``make(transport=None, cls=WujiHandDeployment, **options)``: a deployment on this env, on ``transport``."""
        with deployments(envs(generation)) as make:
            yield lambda transport=None, cls=wh.WujiHandDeployment, **opts: _connected(make(cls, **opts), transport)

    return fixture


hand1, hand2 = _deployer("hand1"), _deployer("hand2")


def _connected(dep, transport):
    if transport is not None:
        dep.use_transport(transport)
        dep.connect()
    return dep


# --- the joint table (no env) ----------------------------------------------------------------


@pytest.mark.parametrize("robot_cls", [WujiHand1_R, WujiHand1_L, WujiHand2Beta2_R, WujiHand2Beta2_L])
def test_every_wuji_robot_maps_onto_each_sdk_joint_once_in_declaration_order(robot_cls):
    assert [wh.WUJI_JOINT_IDX[name] for name in robot_cls().dofs_name] == list(range(N))


def test_sdk_joint_ids_map_finger_major():
    # wuji_sdk nids are 1-based with stride 5 per finger; the fifth node of a finger is not a joint.
    assert [wh._nid_index(n) for n in (1, 4, 5, 6, 24, 25)] == [0, 3, None, 4, 19, None]
    # Both naming schemes name the same joint (upstream ``WUJI_HAND2_TASK_NAMES``).
    assert wh.WUJI_JOINT_IDX["r_index_finger_mcp_abd"] == wh.WUJI_JOINT_IDX["right_finger2_joint2"] == 5


@pytest.mark.parametrize("robot_cls", [WujiHand1_R, WujiHand2Beta2WithMount_L])
def test_wuji_robots_resolve_to_this_backend(robot_cls):
    assert resolve_deployment_for(robot_cls()) is wh.WujiHandDeployment


# --- construction ----------------------------------------------------------------------------


@pytest.mark.parametrize(("generation", "side"), [("hand1", "right"), ("hand2", "left")])
def test_side_and_generation_come_from_the_robot(request, generation, side):
    dep = request.getfixturevalue(generation)()
    assert (dep.side, dep.hand) == (side, generation)


def test_an_unrecognised_robot_must_name_its_generation(hand1, monkeypatch):
    monkeypatch.setattr(wh, "_WujiHand1", type("NotThisRobot", (), {}))
    with pytest.raises(ValueError, match=r"configure\(hand='hand1'\)"):
        hand1()
    assert hand1(hand="hand2").hand == "hand2"


def test_mixed_sides_are_refused_naming_the_joints(hand1, monkeypatch):
    monkeypatch.setitem(wh._JOINT_SIDE, H1_DOFS[0], "left")
    with pytest.raises(ValueError, match="mixes right- and left-hand joints"):
        hand1()


def test_a_joint_mapped_twice_names_the_duplicate_and_the_gap(hand1, monkeypatch):
    monkeypatch.setitem(wh.WUJI_JOINT_IDX, "right_finger5_joint4", 0)  # thumb joint1's slot, twice
    with pytest.raises(ValueError, match=r"SDK joints \[19\] have no joint") as exc:
        hand1()
    assert "right_finger1_joint1" in str(exc.value) and "right_finger5_joint4" in str(exc.value)


@pytest.mark.parametrize(
    ("options", "match"),
    [
        ({"stream_mode": "Thread"}, r"stream_mode='Thread' is not one of \('sync', 'thread'\)"),
        ({"hand": "Hand2"}, r"hand='Hand2' is not one of"),
        ({"servo_hz": 0.0}, "servo_hz must be a positive number"),
        ({"effort_limit_a": -1.0}, "effort_limit_a must be a positive number"),
        ({"vel_lowpass_hz": 0.0}, "vel_lowpass_hz must be a positive number"),
        ({"init_dofs_kp_scale": 0.5}, "init_dofs_kp_scale=0.5"),
        ({"disable_joints": ("l_thumb_*",)}, "Remove disable_joints"),
        ({"mit_kp": 2.0, "deploy_kp_kd_scale": 0.5}, "leave deploy_kp_kd_scale at 1.0"),
        ({"mit_kp": {"l_thumb_.*": 1.0, "l_(index|middle|ring)_finger_.*": 1.0, "l_pinky_mcp.*": 1.0}}, "l_pinky_dip"),
    ],
    ids=[
        "stream_mode",
        "hand",
        "servo_hz",
        "effort_limit",
        "vel_lowpass",
        "init_gain_scale",
        "disable_joints",
        "gains_and_scale",
        "gains_leave_a_joint_out",
    ],
)
def test_option_values_are_checked_at_construction(hand2, options, match):
    with pytest.raises(ValueError, match=match):
        hand2(**options)


@pytest.mark.parametrize(
    "options", [{"deploy_kp_kd_scale": 0.5}, {"mit_kp": 1.0}, {"mit_kd": {".*": 0.1}}], ids=["scale", "kp", "kd"]
)
def test_hand1_refuses_gains_it_cannot_receive(hand1, options):
    with pytest.raises(ValueError, match=r"configure\(effort_limit_a=0\.3\)"):
        hand1(**options)


# --- gains -----------------------------------------------------------------------------------


def test_hand2_programs_the_nominal_gains_by_default(hand2):
    gains = hand2(deploy_kp_kd_scale=0.5)._make_transport().gains
    np.testing.assert_allclose(gains.kp, [25.0] * N)  # default_dofs_kp 50 x 0.5
    np.testing.assert_allclose(gains.kd, [2.5] * N)
    assert gains.effort_limit_a == [1.0] * N


def test_hand2_per_joint_gains_reach_the_sdk_in_its_order(hand2):
    kp = {name: float(i) for i, name in enumerate(WujiHand2Beta2_L().dofs_name)}  # SDK index as the value
    gains = hand2(mit_kp=kp, mit_kd=0.1, effort_limit_a=1.5)._make_transport().gains
    np.testing.assert_allclose(gains.kp, np.arange(N))
    assert gains.kd == [0.1] * N
    assert gains.effort_limit_a == [1.5] * N


@pytest.mark.parametrize(
    "options, kp, kd", [({"deploy_kp_kd_scale": 0.5}, 25.0, 2.5), ({"mit_kp": 2.0, "mit_kd": 0.1}, 2.0, 0.1)]
)
def test_hand2_reports_the_gains_it_programs_for_the_deploy_to_compare_slots_with(hand2, options, kp, kd):
    """``eden deploy`` warns about a slot trained at other gains than these; they are the policy's units."""
    dep = hand2(**options)
    programmed = dep.programmed_gains()
    np.testing.assert_allclose(programmed[0], [kp] * N)
    np.testing.assert_allclose(programmed[1], [kd] * N)
    np.testing.assert_allclose(dep.to_sdk(programmed[0]), dep._make_transport().gains.kp)


def test_hand1_reports_no_gains_its_position_loop_could_be_compared_by(hand1):
    assert hand1().programmed_gains() is None


# --- the control tick ------------------------------------------------------------------------


def test_targets_are_scattered_to_the_sdk_and_reads_gathered_back(hand1):
    sdk = np.arange(N, dtype=np.float64) / 10.0
    transport = ScriptedHandTransport(reads=[sdk])
    dep = hand1(transport)
    assert dep.sdk_joint_names == WujiHand1_R().dofs_name  # what a transport names joints by
    dep.send_payload(_command(dep.from_sdk(sdk)))
    np.testing.assert_allclose(transport.sent[-1], sdk)
    np.testing.assert_allclose(dep.read_state().dofs_pos, sdk[::-1])  # deployment order is the SDK's, reversed


def test_velocity_is_a_finite_difference(hand1):
    dep = hand1(ScriptedHandTransport(reads=[np.zeros(N), np.full(N, 0.02)], clock=[1.0, 1.02]))
    np.testing.assert_array_equal(dep.read_state().dofs_vel, 0.0)  # nothing to difference yet
    np.testing.assert_allclose(dep.read_state().dofs_vel, 1.0)


def test_velocity_low_pass_and_a_stalled_clock(hand1):
    q0, q1 = np.zeros(N), np.full(N, 0.02)
    dep = hand1(ScriptedHandTransport(reads=[q0, q1, q1], clock=[1.0, 1.02, 1.02]), vel_lowpass_hz=100.0)
    dep.read_state()
    a = 0.02 / (0.02 + 1.0 / (2.0 * np.pi * 100.0))
    np.testing.assert_allclose(dep.read_state().dofs_vel, a)
    np.testing.assert_allclose(dep.read_state().dofs_vel, a)  # a stalled clock keeps the estimate, never divides
    dep.reset_velocity_filter()
    np.testing.assert_array_equal(dep._vel, 0.0)


def test_limits_clip_every_target_including_ease_goals(hand1):
    transport = ScriptedHandTransport()
    dep = hand1(transport, clip_targets_to_joint_limits=True)
    lower, upper = dep._lower, dep._upper
    dep.send_payload(_command(np.full(N, 50.0)))
    dep.ease_to(np.full(N, -50.0), duration_s=0.0, rate_hz=100.0)
    np.testing.assert_allclose(dep.from_sdk(transport.sent[-1]), upper)
    np.testing.assert_allclose(dep.from_sdk(transport.eased[-1]), lower)


def test_target_lpf_smooths_the_sent_targets(hand1):
    transport = ScriptedHandTransport()
    dep = hand1(transport, target_lpf=0.5)
    dep.send_payload(_command(np.zeros(N)))
    dep.send_payload(payload := _command(np.full(N, 0.2)))
    np.testing.assert_allclose(transport.sent, [np.zeros(N), np.full(N, 0.1)])  # warm start, then a*new+(1-a)*prev
    np.testing.assert_allclose(payload.dofs_pos, 0.1)  # a debug record logs what was sent


def test_init_ramps_from_the_first_measured_pose_to_the_default_and_holds(hand1):
    """Re-reading the pose each tick would chase a sagging finger; the start is latched once."""
    start, sagged = np.full(N, 0.4), np.full(N, 0.1)
    transport = ScriptedHandTransport(reads=[start, sagged])
    dep = hand1(transport, init_steps=4)
    dep.reset()  # boots in STOP, measuring the start pose
    dep.request_init()
    sent_before = len(transport.sent)
    for _ in range(6):
        dep.step(None)
    transport.sent = transport.sent[sent_before:]
    default = dep.default_dof_pos
    got = [dep.from_sdk(t) for t in transport.sent]
    for sent, alpha in zip(got, [0.0, 0.25, 0.5, 0.75, 1.0, 1.0], strict=True):
        np.testing.assert_allclose(sent, start + alpha * (default - start), atol=1e-12)


def test_step_reads_and_syncs_the_scene_before_publishing_commands(hand1):
    """The order the port's sim-to-sim runs were validated with: a command grades this tick's state."""
    calls = []

    class Recording(wh.WujiHandDeployment):
        def sync_scene(self) -> None:
            calls.append(("sync_scene", float(self.state.dofs_pos[0])))

        def _update_commands(self, advance_reference: bool) -> None:
            calls.append(("commands", int(self._env.episode_length_buf[0])))

    transport = ScriptedHandTransport(reads=[np.full(N, 0.3)])
    dep = hand1(transport, cls=Recording)
    dep.action_to_payload = lambda action: _command(np.full(N, 0.1))
    dep.request_start()
    dep.step(torch.zeros(N))  # the START tick holds
    calls.clear()
    clock0 = int(dep._env.episode_length_buf[0])

    dep.step(torch.zeros(N))
    np.testing.assert_allclose(transport.sent[-1], 0.1)
    # The scene is never stepped on this path: the episode clock advances here, once per tick.
    assert calls == [("sync_scene", 0.3), ("commands", clock0 + 1)]


@pytest.mark.parametrize(("stream_mode", "paced_by_the_base"), [("sync", False), ("thread", True)])
def test_sync_streaming_is_the_loop_pacing_so_the_rate_limiter_is_not_stacked_on_it(
    hand1, stream_mode, paced_by_the_base
):
    assert (hand1(sync=True, stream_mode=stream_mode)._rate is not None) is paced_by_the_base


def test_a_hand_fault_stops_the_run_before_anything_is_sent(hand1):
    transport = ScriptedHandTransport(fault="joint nid=3 error 0x0001")
    with pytest.raises(wh.DeployStop, match="nid=3"):
        hand1(transport).step(torch.zeros(N))
    assert transport.sent == []


def test_stop_on_fault_off_tolerates_a_hardware_fault(hand1):
    hand1(ScriptedHandTransport(fault="joint nid=3 error 0x0001"), stop_on_fault=False).check_safety()


@pytest.mark.parametrize(
    "move",
    [
        lambda dep: dep.check_safety(),
        lambda dep: (dep.request_init(), dep.step(None)),
        lambda dep: dep.step(None),
        lambda dep: dep.ease_to(np.zeros(N), 1.0, 100.0),
    ],
    ids=["policy_tick", "init", "hold", "ease_to"],
)
def test_stale_feedback_stops_every_motion_even_with_stop_on_fault_off(hand1, move):
    """A hand must never move on a frozen reading, whatever ``stop_on_fault`` says."""
    transport = ScriptedHandTransport(stale="feed stale")
    with pytest.raises(wh.DeployStop, match="feedback lost: feed stale"):
        move(hand1(transport, stop_on_fault=False))
    assert transport.sent == transport.eased == []


def test_ease_to_checks_the_hand_before_every_waypoint(hand1):
    """Feedback lost after the ease has started stops it: the check is not only made up front."""
    transport = ScriptedHandTransport()
    answers = iter([None, "feed stale"])  # fresh for the up-front check, stale by the first waypoint
    transport.stale = lambda: next(answers)
    with pytest.raises(wh.DeployStop, match="feedback lost: feed stale"):
        hand1(transport).ease_to(np.zeros(N), 1.0, 100.0)
    assert transport.eased == []


def test_target_lpf_restarts_from_an_ease_goal(hand1):
    """The command after an ease filters from where the hand was eased to, not back toward the old target."""
    transport = ScriptedHandTransport()
    dep = hand1(transport, target_lpf=0.5)
    dep.send_payload(_command(np.zeros(N)))
    dep.ease_to(np.full(N, 0.4), duration_s=0.0, rate_hz=100.0)
    dep.send_payload(_command(np.full(N, 0.4)))
    np.testing.assert_allclose(transport.sent[-1], 0.4)  # not 0.2, half-way back to the pre-ease 0.0


def test_a_tick_with_nothing_to_send_holds_for_the_period(hand1):
    """In "sync" the send is the loop's only wait; a tick that sends nothing must still take one period."""
    transport = ScriptedHandTransport()
    dep = hand1(transport)
    dep.request_start()
    dep.step(None)  # the START tick holds
    transport.sent.clear()
    transport.holds = 0
    dep.step(None)  # no action, and nothing written through write_dofs_targets
    assert transport.sent == [] and transport.holds == 1


def test_reset_restarts_the_episode_clock_and_the_velocity_estimate(hand1):
    # step, step, then reset's read.
    transport = ScriptedHandTransport(reads=[np.zeros(N), np.full(N, 0.5), np.ones(N)])
    transport.times = [1.0, 1.02, 100.0]
    dep = hand1(transport)
    dep.action_to_payload = lambda action: _command(np.zeros(N))
    dep.request_start()
    dep.step(torch.zeros(N))
    dep.step(torch.zeros(N))
    assert int(dep._env.episode_length_buf[0]) > 0 and np.all(dep.state.dofs_vel > 0)
    dep.reset()
    assert int(dep._env.episode_length_buf[0]) == 0
    # The pose moved by 0.5 across the reset; differencing it across the reset is not a velocity.
    np.testing.assert_array_equal(dep.state.dofs_vel, 0.0)


def test_the_episode_clock_is_the_policys_and_starts_at_the_handover(hand1):
    """A warm-up gated on the clock (the reorient task's) must still be running at START, however long the
    operator held and INITed before it.
    """
    dep = hand1(ScriptedHandTransport(), init_steps=3)
    dep.reset()
    for _ in range(4):
        dep.step(None)  # STOP hold
    dep.request_init()
    for _ in range(5):
        dep.step(None)  # the INIT ramp, then its hold
    assert int(dep._env.episode_length_buf[0]) == 0, "nothing counts before START"

    dep.request_start()
    dep.action_to_payload = lambda action: _command(np.zeros(N))
    dep.step(torch.zeros(N))
    assert int(dep._env.episode_length_buf[0]) == 0, "the START tick holds; its read is the policy's first"
    dep.step(torch.zeros(N))
    assert int(dep._env.episode_length_buf[0]) == 1, "the first policy tick"
    dep.step(torch.zeros(N))
    dep.request_stop()
    dep.step(None)
    assert int(dep._env.episode_length_buf[0]) == 2, "STOP freezes it"
    dep.request_start()
    dep.step(torch.zeros(N))
    assert int(dep._env.episode_length_buf[0]) == 0, "and every START restarts it"


def test_a_transport_cannot_be_swapped_once_connected(hand1):
    dep = hand1(ScriptedHandTransport())
    with pytest.raises(RuntimeError, match="before connect"):
        dep.use_transport(ScriptedHandTransport())


def test_a_transport_must_stream_as_the_deployment_is_configured(hand1):
    with pytest.raises(ValueError, match=r"streams 'thread'.*stream_mode='sync'.*dep\.stream"):
        hand1(ScriptedHandTransport(stream=wh.StreamOptions(mode="thread")))


def test_close_releases_the_transport_even_when_teardown_raises(hand1, monkeypatch):
    transport = ScriptedHandTransport()
    dep = hand1(transport)
    monkeypatch.setattr(dep, "_dump_record", lambda: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError, match="disk full"):
        dep.close()
    assert not transport.connected


# --- streaming -------------------------------------------------------------------------------


class _Recorder(wh.HandTransport):
    """Streams for real, recording every write; a write takes ``write_s``, and the hand always reads zeros."""

    def __init__(self, write_s: float = 0.0, **stream):
        super().__init__(wh.StreamOptions(**{"mode": "sync", "servo_hz": 200.0, "interp_s": 0.02, **stream}))
        self.sent = []
        self.write_s = write_s

    def read(self):
        return np.zeros(N)

    def write(self, qpos):
        self.sent.append(qpos.copy())
        time.sleep(self.write_s)

    def disconnect(self):
        pass


def test_sync_streaming_interpolates_over_the_period():
    t = _Recorder()
    t.command(np.ones(N))
    # 4 sub-targets at 200 Hz over 20 ms, ending on the target (upstream run_policy's streaming).
    np.testing.assert_allclose([s[0] for s in t.sent], [0.25, 0.5, 0.75, 1.0])
    bad = np.zeros(N)
    bad[3] = np.inf
    with pytest.raises(ValueError, match=r"non-finite joint target for \['sdk\[3\]'\]"):
        t.command(bad)
    with pytest.raises(ValueError, match="20 joint angles"):
        t.command(np.ones(12))
    t.close()  # a reconnect streams from the pose it then measures, not from the last target
    t.sent.clear()
    t.command(np.ones(N))
    assert t.sent[0][0] == pytest.approx(0.25)


def test_sync_ticks_last_one_period_absorbing_the_time_between_commands():
    """Upstream anchors each period before inference, so inference must not lengthen the tick."""
    t = _Recorder(write_s=0.002, interp_s=0.1)
    t0 = time.monotonic()
    t.command(np.ones(N))
    time.sleep(0.03)  # inference
    t.command(np.zeros(N))
    assert time.monotonic() - t0 == pytest.approx(0.2, abs=0.015)
    t = _Recorder(interp_s=1 / 30)
    t.command(np.ones(N))
    assert len(t.sent) == 7  # round(200 Hz x 33.3 ms), not truncated to 6 sub-targets (30 ms)


def test_thread_streaming_ramps_then_holds_and_surfaces_a_write_error():
    t = _Recorder(mode="thread", servo_hz=1000.0, interp_s=0.01)
    t.command(np.ones(N))
    time.sleep(0.05)
    assert t.sent[-1][0] == pytest.approx(1.0)  # ramp over, holding the target
    assert 0.0 <= min(s[0] for s in t.sent) < 1.0

    def fail(qpos):
        raise OSError("bus down")

    t.write = fail
    time.sleep(0.02)
    with pytest.raises(RuntimeError, match="servo stream stopped: bus down"):
        t.command(np.ones(N))
    with pytest.raises(RuntimeError, match="servo stream stopped: bus down"):
        t.ease_to(np.zeros(N), 0.05, 100.0, "cosine")
    t.stop_servo()


def test_ease_profiles_are_checked():
    t = _Recorder(servo_hz=100.0)
    with pytest.raises(ValueError, match=r"one of \('cosine', 'smoothstep'\)"):
        t.ease_to(np.ones(N), 0.01, 100.0, "linear")
    assert t.sent == []
    t.ease_to(np.ones(N), 0.04, 100.0, "smoothstep")
    assert [round(float(s[0]), 3) for s in t.sent] == [0.156, 0.5, 0.844, 1.0]


def test_ease_to_stops_where_its_guard_raises():
    t = _Recorder(servo_hz=100.0)
    calls = []

    def guard():
        calls.append(len(t.sent))
        if len(calls) == 3:
            raise wh.DeployStop("hand feedback lost: feed stale")

    with pytest.raises(wh.DeployStop, match="feed stale"):
        t.ease_to(np.ones(N), 0.04, 100.0, "smoothstep", guard=guard)
    assert calls == [0, 1, 2]  # before every waypoint
    assert [round(float(s[0]), 3) for s in t.sent] == [0.156, 0.5]
    t.sent.clear()
    t.command(np.ones(N))  # streams on from the last waypoint written, not from the unreached goal
    np.testing.assert_allclose([s[0] for s in t.sent], [0.75, 1.0])


def test_hold_restreams_the_last_target_for_one_period():
    t = _Recorder(interp_s=0.05)
    t0 = time.monotonic()
    t.hold()
    assert t.sent == []  # nothing commanded yet, so nothing to hold: it only waits
    assert time.monotonic() - t0 == pytest.approx(0.05, abs=0.015)
    t.command(np.full(N, 0.3))
    t.sent.clear()
    t.hold()
    np.testing.assert_allclose(t.sent, np.full((10, N), 0.3))  # 200 Hz x 50 ms, all the held target


def test_hand1_read_timeout_stops_the_run_and_stays_stale(monkeypatch):
    """Each Hand 1 read is a bus round-trip; one that times out is lost feedback, not a crash."""
    timeouts = []

    def read_joint_actual_position(timeout):
        timeouts.append(timeout)
        raise TimeoutError("no response")

    hand = SimpleNamespace(
        write_joint_effort_limit=lambda amps: None,
        write_joint_enabled=lambda on: None,
        realtime_controller=lambda **kwargs: SimpleNamespace(__enter__=SimpleNamespace, __exit__=lambda *exc: None),
        read_joint_actual_position=read_joint_actual_position,
    )
    lowpass = SimpleNamespace(LowPass=lambda cutoff_freq: None)
    monkeypatch.setitem(sys.modules, "wujihandpy", SimpleNamespace(Hand=lambda: hand, filter=lowpass))
    t = wh.Hand1Transport(wh.StreamOptions(), stale_after_s=0.2)
    t.connect()
    with pytest.raises(wh.DeployStop, match=r"feedback lost: read_joint_actual_position\(timeout=0.2\).*TimeoutError"):
        t.read()
    assert timeouts == [0.2] and "TimeoutError: no response" in t.stale()


def test_hand1_is_de_energized_when_its_controller_fails_to_open(monkeypatch):
    enabled = []

    def realtime_controller(**kwargs):
        raise OSError("controller busy")

    hand = SimpleNamespace(
        write_joint_effort_limit=lambda amps: None,
        write_joint_enabled=enabled.append,
        realtime_controller=realtime_controller,
    )
    lowpass = SimpleNamespace(LowPass=lambda cutoff_freq: None)
    monkeypatch.setitem(sys.modules, "wujihandpy", SimpleNamespace(Hand=lambda: hand, filter=lowpass))
    with pytest.raises(OSError, match="controller busy"):
        wh.Hand1Transport(wh.StreamOptions()).connect()
    assert enabled == [True, False]


# --- Hand 2 over a fake wuji_sdk ---------------------------------------------------------------


class FakeWujiSdk:
    """``wuji_sdk`` as far as :class:`Hand2Transport` uses it: a bus of devices, one hand's joint streams."""

    GLOVE, HAND = "WujiGlove", "WujiHand2"

    def __init__(self, devices, online=N, reporting=None):
        self.devices = devices
        self.online = online
        self.reporting = online if reporting is None else reporting
        self.connected: list[str] = []
        self.disconnected: list[str] = []
        self.enabled = False
        self.mit_error: Exception | None = None
        self.sent: list[list[float]] = []
        self.diag_cb = None
        sdk = self

        class _Value:
            def __init__(self, get=None, on_set=None):
                self._get, self._on_set = get, on_set

            def get(self):
                return self._get

            def set(self, value):
                if self._on_set:
                    self._on_set(value)

        class _Stream:
            def __init__(self, kind):
                self.kind = kind

            def subscribe_with_callback(self, cb):
                if self.kind == "states":  # the online joints report straight away
                    cb(SimpleNamespace(joints=[SimpleNamespace(nid=5 * (i // 4) + i % 4 + 1, position=0.1 * i)
                                               for i in range(sdk.reporting)]))  # fmt: skip
                else:
                    sdk.diag_cb = cb
                return SimpleNamespace(close=lambda: None)

            def publish(self):
                return SimpleNamespace(send=lambda cmds: sdk.sent.append([c.q for c in cmds]))

        def _mit_set(value):
            if sdk.mit_error:
                raise sdk.mit_error

        class Hand:
            serial_number = "SN-HAND"

            def online_joints_count(self):
                return _Value(sdk.online)

            def effort_limit(self):
                return _Value()

            def mit_params(self):
                return _Value(on_set=_mit_set)

            def joint_states(self):
                return _Stream("states")

            def joint_diagnostics(self):
                return _Stream("diag")

            def joint_command(self):
                return _Stream("cmd")

            def enable(self):
                sdk.enabled = True

            def disable(self):
                sdk.enabled = False

            def describe_error(self, code):
                return {
                    "severity": "warning" if code == 1 else "error",
                    "name": f"E{code}",
                    "desc": "",
                    "resolution": "",
                }

        class Manager:
            def scan(self):
                return sdk.devices

            def connect(self, sn, device_name, options):
                sdk.connected.append((sn, device_name))
                return Hand()

            def disconnect(self, name):
                sdk.disconnected.append(name)

            def disconnect_all(self):
                raise AssertionError("disconnect_all would also drop every other wuji_sdk device")

        self.module = SimpleNamespace(
            SdkManager=SimpleNamespace(instance=Manager),
            ConnectOptions=lambda enable_bridge: None,
            MitParam=lambda kp, kd: (kp, kd),
            JointCommand=lambda q, dq, tau: SimpleNamespace(q=q),
            DeviceType=SimpleNamespace(WujiGlove=self.GLOVE, WujiHand2=self.HAND),
        )

    @classmethod
    def device(cls, sn, kind):
        return SimpleNamespace(sn=sn, ip="10.0.0.2", transport_type="usb", device_type=kind)


@pytest.fixture
def fake_sdk(monkeypatch):
    def install(devices=(("SN-HAND", FakeWujiSdk.HAND),), online=N, reporting=None):
        sdk = FakeWujiSdk([FakeWujiSdk.device(sn, kind) for sn, kind in devices], online, reporting)
        monkeypatch.setitem(sys.modules, "wuji_sdk", sdk.module)
        monkeypatch.setattr(wh, "_HAND2_ARM_DELAY_S", 0.0)
        return sdk

    return install


def _hand2_transport(**kwargs) -> wh.Hand2Transport:
    gains = wh.Hand2Gains(kp=[1.0] * N, kd=[0.1] * N, effort_limit_a=[1.0] * N)
    return wh.Hand2Transport(wh.StreamOptions(), gains, **kwargs)


def test_hand2_autoselect_skips_a_glove_on_the_bus(fake_sdk):
    sdk = fake_sdk(devices=[("SN-GLOVE", FakeWujiSdk.GLOVE), ("SN-HAND", FakeWujiSdk.HAND)])
    t = _hand2_transport()
    t.connect()
    assert [sn for sn, _ in sdk.connected] == ["SN-HAND"] and sdk.enabled
    np.testing.assert_allclose(t.read(), 0.1 * np.arange(N))
    assert t.stale() is None and t.fault() is None


def test_hand2_autoselect_refuses_two_hands_listing_them(fake_sdk):
    fake_sdk(devices=[("A", FakeWujiSdk.HAND), ("B", FakeWujiSdk.HAND)])
    with pytest.raises(RuntimeError, match=r"found 2 Wuji hands.*sn=A.*sn=B.*serial_number="):
        _hand2_transport().connect()


def test_each_hand_disconnects_only_its_own_connection(fake_sdk):
    sdk = fake_sdk()
    a, b = _hand2_transport(serial_number="SN-HAND"), _hand2_transport(serial_number="SN-HAND")
    a.connect()
    b.connect()
    assert a.device_name != b.device_name
    a.close()
    assert sdk.disconnected == [a.device_name]


def test_an_incomplete_hand_is_refused_and_released(fake_sdk):
    sdk = fake_sdk(online=16)
    t = _hand2_transport()
    with pytest.raises(RuntimeError, match=r"16/20 joints online.*require_all_joints=False"):
        t.connect()
    assert sdk.disconnected == [t.device_name] and not sdk.enabled


def test_a_failed_gain_write_releases_the_hand(fake_sdk):
    sdk = fake_sdk()
    sdk.mit_error = OSError("mit_params rejected")
    t = _hand2_transport()
    with pytest.raises(OSError, match="mit_params rejected"):
        t.connect()
    assert sdk.disconnected == [t.device_name]


def test_joints_that_never_report_time_out_the_connect_by_name(fake_sdk):
    sdk = fake_sdk(reporting=18)
    t = _hand2_transport(feedback_timeout_s=0.05, joint_names=WujiHand2Beta2_L().dofs_name)
    with pytest.raises(
        RuntimeError, match=r"for joints \['l_pinky_pip', 'l_pinky_dip'\].*connect_timeout_s=.*require_all"
    ):
        t.connect()
    assert sdk.disconnected == [t.device_name] and not sdk.enabled


def test_a_partial_hand_watches_the_joints_that_report(fake_sdk):
    """Joints that never report are not stale; a watched joint that stops reporting is."""
    fake_sdk(online=16)
    t = _hand2_transport(require_all_joints=False, joint_names=WujiHand2Beta2_L().dofs_name)
    t.connect()
    assert t._watched.sum() == 16 and t.stale() is None
    t._joint_t[3] -= 10.0
    assert t.stale() == "joint feedback stale for ['l_thumb_ip']"
    t._last_state_t -= 10.0
    assert "joint_states feed stale" in t.stale()


def test_hand2_faults_dismiss_warnings_and_latch_errors(fake_sdk):
    sdk = fake_sdk()
    t = _hand2_transport()
    t.connect()
    sdk.diag_cb(SimpleNamespace(joints=[SimpleNamespace(nid=2, mcu_temp_c_fb=40.0, error_code_current=1)]))
    assert t.fault() is None  # a warning is logged once, then dismissed
    sdk.diag_cb(SimpleNamespace(joints=[SimpleNamespace(nid=7, mcu_temp_c_fb=41.0, error_code_current=9)]))
    assert t.fault() == "joint 'sdk[5]' (nid=7) error 0x0009 E9 [error]  ()"
    assert t.temp_summary() == "40.0-41.0 C (mean 40.5)"


def test_hand2_a_later_warning_does_not_unlatch_an_error(fake_sdk):
    sdk = fake_sdk()
    t = _hand2_transport()
    t.connect()
    sdk.diag_cb(SimpleNamespace(joints=[SimpleNamespace(nid=7, mcu_temp_c_fb=40.0, error_code_current=9)]))
    sdk.diag_cb(SimpleNamespace(joints=[SimpleNamespace(nid=7, mcu_temp_c_fb=40.0, error_code_current=1)]))
    assert t.fault() == "joint 'sdk[5]' (nid=7) error 0x0009 E9 [error]  ()"


def test_dofs_are_looked_up_by_name_beside_the_sdk_joint_map(hand2):
    assert_dofs_looked_up_by_name(hand2(ScriptedHandTransport()), WujiHand2Beta2_L())
