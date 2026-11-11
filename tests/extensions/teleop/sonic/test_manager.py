"""``eden.extensions.teleop.sonic.manager``: the operator's CLI, bridge launch argv, and input-loss rule."""

from __future__ import annotations

import socket
import sys
import time

import pytest

from eden.extensions.policies.sonic import wire
from eden.extensions.teleop.sonic.manager import SonicTeleopOperator, _bridge_argv, build_arg_parser, input_lost
from eden.extensions.teleop.sonic.state_machine import StreamMode
from eden.options.robots.unitree_g1 import G1Dofs29Dex3
from eden.utils.assets import get_asset_path
from tests.support.sonic import smpl_fk
from tests.utils.devices.pico.conftest import FakeSdk


@pytest.mark.parametrize(
    "mode, bridge_died, body_stale, lost",
    [
        (StreamMode.POSE, False, True, True),
        (StreamMode.PLANNER_VR_3PT, False, True, True),
        # Joystick modes ride on the controllers: a tracker occlusion must not end walking.
        (StreamMode.PLANNER, False, True, False),
        (StreamMode.PLANNER_FROZEN_UPPER_BODY, False, True, False),
        (StreamMode.PLANNER, True, False, True),  # a dead bridge freezes every mode
        (StreamMode.OFF, True, True, False),  # nothing engaged, nothing to hand off
    ],
)
def test_input_loss_is_judged_on_the_stream_the_mode_reads(mode, bridge_died, body_stale, lost):
    assert input_lost(mode, bridge_died=bridge_died, body_stale=body_stale) is lost


def test_bridge_argv_is_a_pose_only_bridge_in_this_venv():
    # The private argv builder is the observable form: the public path (setup()) hands it
    # to subprocess.Popen, which launches a real bridge process.
    args = build_arg_parser().parse_args(["--bridge-port", "7001", "--bridge-fps", "90"])
    argv = _bridge_argv(args)
    assert argv[0] == sys.executable
    assert argv[1:3] == ["-m", "eden.utils.devices.pico.bridge"]
    assert "--no-video" in argv  # never binds the frame port (5556 = the operator's own PUB)
    assert argv[argv.index("--pose-port") + 1] == "7001"
    assert argv[argv.index("--fps") + 1] == "90.0"


def test_defaults_use_the_bridge_off_the_sonic_ports():
    args = build_arg_parser().parse_args([])
    assert not args.sdk and not args.no_bridge_start and args.bridge_host == ""
    assert args.bridge_port not in (args.port, args.feedback_port)
    assert args.bridge_fps >= 100.0  # must outpace the ~90 Hz device body rate


# ---------------------------------------------------------------------------
# The operator end to end: real setup / step / teardown over a fake PICO SDK
# ---------------------------------------------------------------------------


class _ScriptedSdk(FakeSdk):
    """The PICO SDK with scripted buttons and a live body stream (the hardware boundary).

    While ``streaming`` the device timestamp ticks at the headset's ~90 Hz (on the wall clock, so
    the reader idles between frames as it does on a real device); clearing it freezes the stream
    (tracking lost).
    """

    FRAME_NS = 11_000_000

    def __init__(self):
        super().__init__(body=True)
        self.buttons = {"a": False, "b": False, "x": False, "y": False}
        self.streaming = True
        self._frozen_stamp = None

    def get_time_stamp_ns(self):
        now = time.perf_counter_ns() // self.FRAME_NS * self.FRAME_NS  # the latest whole device frame
        if self.streaming:
            return now
        if self._frozen_stamp is None:
            self._frozen_stamp = now
        return self._frozen_stamp

    def get_A_button(self):
        return self.buttons["a"]

    def get_B_button(self):
        return self.buttons["b"]

    def get_X_button(self):
        return self.buttons["x"]

    def get_Y_button(self):
        return self.buttons["y"]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def operator_rig(monkeypatch):
    """A set-up ``SonicTeleopOperator`` on a scripted SDK, and a SUB standing in for the robot's command socket.

    Yields ``(operator, sdk, robot, teardown)``: ``robot()`` drains what has reached the robot so
    far as ``(topic, parsed)`` pairs; ``teardown()`` runs the operator's teardown once (the
    fixture runs it at the end if the test did not).
    """
    zmq = pytest.importorskip("zmq")
    pytest.importorskip("msgpack")
    # setup() builds the G1 FK scene and the SMPL-X FK from HF assets: skip before anything starts if either is missing.
    try:
        get_asset_path(G1Dofs29Dex3().file)
    except FileNotFoundError as e:
        pytest.skip(f"G1 with-hand asset unavailable: {e}")
    smpl_fk()
    sdk = _ScriptedSdk()
    monkeypatch.setitem(sys.modules, "xrobotoolkit_sdk", sdk)
    port = _free_port()
    options = SonicTeleopOperator.configure(
        sdk=True, no_service_start=True, port=port, robot_host="127.0.0.1", feedback_port=_free_port()
    )
    operator = SonicTeleopOperator(options)
    sub = zmq.Context.instance().socket(zmq.SUB)
    sub.setsockopt(zmq.SUBSCRIBE, b"")
    sub.connect(f"tcp://127.0.0.1:{port}")
    received = []

    def robot():
        while True:
            try:
                msg = sub.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                return received
            for topic, parse in (
                (wire.COMMAND_TOPIC, wire.parse_command),
                (wire.PLANNER_TOPIC, wire.parse_planner),
            ):
                if msg.startswith(topic.encode()):
                    received.append((topic, parse(wire.unpack_message(msg, topic))))

    torn_down = False

    def teardown():
        nonlocal torn_down
        if not torn_down:
            torn_down = True
            operator.teardown()

    try:
        operator.setup()
        time.sleep(0.3)  # the SUB's join: messages published before it are dropped
        yield operator, sdk, robot, teardown
    finally:
        teardown()
        sub.close(linger=0)


def _hold_buttons(operator, sdk, **buttons):
    """Hold ``buttons`` for one operator tick, then release them for one (the state machine fires on edges)."""
    sdk.buttons.update({k: True for k in buttons})
    operator.step()
    sdk.buttons.update(dict.fromkeys(buttons, False))
    operator.step()


def _commands(received):
    return [m for topic, m in received if topic == wire.COMMAND_TOPIC]


def test_losing_body_tracking_in_pose_mode_hands_the_robot_to_the_planner(operator_rig, clock):
    """Frozen input left a POSE-mode robot holding the last streamed pose for as long as it ran."""
    operator, sdk, robot, _ = operator_rig
    _hold_buttons(operator, sdk, a=True, b=True, x=True, y=True)  # engage -> PLANNER
    _hold_buttons(operator, sdk, a=True, x=True)  # -> POSE
    for _ in range(5):
        operator.step()
    assert _commands(robot())[-1].planner is False  # the robot is on the pose stream

    sdk.streaming = False  # tracking lost: no new body frames
    clock(operator._reader.STALE_TIMEOUT + 1.0)  # private: the reader is built by setup(); its timeout is the knob
    deadline = time.monotonic() + 5.0
    while True:
        last = _commands(robot())[-1]
        if last.planner and not last.start:  # the hand-off, not the engage command
            break
        assert time.monotonic() < deadline, "the operator never handed the robot back to the planner"
        operator.step()
    planner_msgs = [m for topic, m in robot() if topic == wire.PLANNER_TOPIC]
    idle = planner_msgs[-1]
    assert idle.mode == 0 and idle.vr_position is None and idle.upper_body_position is None


def test_the_combo_again_disengages_and_the_robot_keeps_balancing_on_the_planner(operator_rig):
    """A+B+X+Y while engaged hands the robot to the planner's IDLE; nothing ever sends it ``stop``.

    The reference sends ``stop`` there, which ends the robot process (damping only, then limp) —
    an unsupported real robot falls. The operator also keeps running, so the combo re-engages.
    """
    operator, sdk, robot, teardown = operator_rig
    _hold_buttons(operator, sdk, a=True, b=True, x=True, y=True)  # engage -> PLANNER
    _hold_buttons(operator, sdk, a=True, x=True)  # -> POSE
    received_before = len(robot())

    _hold_buttons(operator, sdk, a=True, b=True, x=True, y=True)  # disengage
    time.sleep(0.2)

    assert operator._sm.mode == StreamMode.OFF and not operator.should_stop()
    after = robot()[received_before:]
    assert [(m.planner, m.start, m.stop) for topic, m in after if topic == wire.COMMAND_TOPIC] == [(True, False, False)]
    idle = [m for topic, m in after if topic == wire.PLANNER_TOPIC]
    assert idle and idle[-1].mode == 0 and idle[-1].vr_position is None

    _hold_buttons(operator, sdk, a=True, b=True, x=True, y=True)  # re-engage
    assert operator._sm.mode == StreamMode.PLANNER
    teardown()
    time.sleep(0.2)
    assert not any(cmd.stop for cmd in _commands(robot()))


def test_re_engaging_a_robot_left_on_the_planner_keeps_its_facing(operator_rig):
    """The robot stayed engaged, so it re-captures no heading: the facing it idles with carries over."""
    operator, sdk, robot, _ = operator_rig
    _hold_buttons(operator, sdk, a=True, b=True, x=True, y=True)  # engage -> PLANNER
    operator._planner_streamer.yaw_accumulator.update(1.0, 0.5)  # the operator turned the robot
    turned = list(operator._planner_streamer.yaw_accumulator.heading)

    _hold_buttons(operator, sdk, a=True, b=True, x=True, y=True)  # disengage
    _hold_buttons(operator, sdk, a=True, b=True, x=True, y=True)  # re-engage

    assert list(operator._planner_streamer.yaw_accumulator.heading) == turned


def test_exiting_while_engaged_hands_the_robot_to_the_planner(operator_rig):
    """Ctrl-C (or an exception) mid-session: teardown releases the robot instead of leaving it on the pose stream."""
    operator, sdk, robot, teardown = operator_rig
    _hold_buttons(operator, sdk, a=True, b=True, x=True, y=True)  # engage -> PLANNER
    _hold_buttons(operator, sdk, a=True, x=True)  # -> POSE: the robot follows the pose stream
    received_before = len(robot())
    assert _commands(robot())[-1].planner is False
    teardown()
    time.sleep(0.2)
    after = robot()[received_before:]  # only what teardown sent
    assert [(m.planner, m.start, m.stop) for topic, m in after if topic == wire.COMMAND_TOPIC] == [(True, False, False)]
    idle = [m for topic, m in after if topic == wire.PLANNER_TOPIC]
    assert len(idle) == 1 and idle[0].mode == 0 and idle[0].vr_position is None
