"""The deployment safety layer, on a built two-joint arm through the in-memory ``LoopbackDeployment``.

The arm's joints have ranges ``j1`` [-1, 1] and ``j2`` [-0.5, 0.5]. The loopback reports the robot at the last
commanded pose unless a test scripts a reading, so a test that wants a lagging joint or a link out of its box
queues that reading. The clock for ``max_read_interval_s`` is the one double besides the wire.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from eden.extensions.deployment import DeployStop
from eden.options.extensions.deployment import DeploySafetyOptions, LinkBoxOptions
from tests.support.deploy import deployments, robot_env, robot_state, start_policy

LIMITED_ARM_MJCF = """
<mujoco model="limited_arm">
  <compiler angle="radian"/>
  <worldbody>
    <body name="base" pos="0 0 1.0">
      <geom name="root" type="sphere" size="0.05" mass="1.0"/>
      <body name="link1" pos="0 0 0">
        <joint name="j1" type="hinge" axis="0 1 0" limited="true" range="-1 1"/>
        <geom name="g1" type="capsule" fromto="0 0 0 0.3 0 0" size="0.03" mass="0.5"/>
        <body name="link2" pos="0.3 0 0">
          <joint name="j2" type="hinge" axis="0 1 0" limited="true" range="-0.5 0.5"/>
          <geom name="g2" type="capsule" fromto="0 0 0 0.3 0 0" size="0.03" mass="0.5"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


@pytest.fixture(scope="module")
def arm_env(tmp_path_factory):
    from eden.options.entities import RobotOptions

    path: Path = tmp_path_factory.mktemp("limited_arm") / "limited_arm.xml"
    path.write_text(LIMITED_ARM_MJCF)
    robot = RobotOptions(
        file=str(path),
        dofs_name=["j1", "j2"],
        default_dofs_pos={".*": 0.0},
        default_dofs_kp={".*": 50.0},
        default_dofs_kd={".*": 2.0},
        is_fixed_base=True,
    )
    return robot_env(robot)


@pytest.fixture
def deploy(arm_env):
    """``deploy(**safety)``: a connected, reset loopback deployment of the arm with those safety options."""
    with deployments(arm_env) as make:

        def build(**safety):
            dep = make(safety=DeploySafetyOptions(**safety), connect=True)
            dep.reset()
            return dep

        yield build


def drive(dep, *targets) -> list[np.ndarray]:
    """Hand the policy's targets (j1, j2) to the deployment, one control step each; return what was sent."""
    for target in targets:
        dep.write_dofs_targets(torch.tensor(target, dtype=torch.float32), ["j1", "j2"])
        dep.step(None)
    return [command.dofs_pos for command in dep.sent[-len(targets) :]]


def link2_z(arm_env) -> float:
    """Where ``link2`` sits on the built arm at its default pose."""
    robot = arm_env.entities["robot"]
    _, (idx,) = robot.find_named_links_idx_local(["link2"])
    return float(robot.get_links_pos([idx])[0, 0, 2])


# ---------------------------------------------------------------------------
# the filter, on the ticks the policy drives
# ---------------------------------------------------------------------------


def test_the_policys_targets_are_clamped_into_the_joint_range_shrunk_by_the_margin(deploy):
    dep = deploy(dofs_pos_margin=0.1)
    start_policy(dep)

    (sent,) = drive(dep, [5.0, -5.0])

    np.testing.assert_allclose(sent, [0.9, -0.4])
    np.testing.assert_allclose(dep.state.extra["safety.target_raw"], [5.0, -5.0])


def test_a_target_moves_at_most_its_cap_from_the_measured_pose_per_tick(deploy):
    """Only the joint the cap names is capped; the loopback robot reaches each target, so j1 ramps."""
    dep = deploy(max_dofs_step_from_measured={"j1": 0.1})
    start_policy(dep)

    sent = drive(dep, [0.5, 0.3], [0.5, 0.3], [0.5, 0.3])

    np.testing.assert_allclose(sent, [[0.1, 0.3], [0.2, 0.3], [0.3, 0.3]], atol=1e-6)


def test_a_target_moves_at_most_its_cap_from_the_previous_send_when_the_robot_lags(deploy, arm_env):
    """The measured cap would let j1 go to 0.25 from the stuck reading of 0.2; the send cap holds it to 0.15."""
    dep = deploy(max_dofs_step_from_measured=0.05, max_dofs_step_from_sent=0.05)
    start_policy(dep)
    drive(dep, [0.1, 0.0], [0.1, 0.0])
    dep.readings.append(robot_state(2, dofs_pos=[0.2, 0.0]))  # a reading after the next send

    drive(dep, [0.5, 0.0])
    (sent,) = drive(dep, [0.5, 0.0])

    np.testing.assert_allclose(sent, [0.2, 0.0], atol=1e-6)
    assert dep.sent[-2].dofs_pos[0] == pytest.approx(0.15)


def test_when_the_two_caps_disagree_the_target_stays_near_the_measured_pose(deploy):
    """A robot pushed back from its last target: the send cap would allow 0.08, the measured cap holds it to 0.05."""
    dep = deploy(max_dofs_step_from_measured=0.05, max_dofs_step_from_sent=0.02)
    start_policy(dep)
    drive(dep, *[[0.1, 0.0]] * 5)  # the send cap ramps j1 to 0.1 and the loopback robot follows
    dep.readings.append(robot_state(2, dofs_pos=[0.0, 0.0]))  # read after the next send

    drive(dep, [0.1, 0.0])
    (sent,) = drive(dep, [0.1, 0.0])

    np.testing.assert_allclose(sent, [0.05, 0.0], atol=1e-6)


def test_the_range_clamp_holds_over_the_measured_cap(deploy):
    """A joint read past the shrunk range is not followed there: the target goes to the range's end."""
    dep = deploy(dofs_pos_margin=0.1, max_dofs_step_from_measured=0.02)
    start_policy(dep)
    dep.readings.append(robot_state(2, dofs_pos=[1.0, 0.0]))
    drive(dep, [0.0, 0.0])

    (sent,) = drive(dep, [2.0, 0.0])

    np.testing.assert_allclose(sent, [0.9, 0.0], atol=1e-6)


def test_the_hold_is_not_filtered(deploy):
    """The STOP hold is the state machine's own command; the filter acts on the policy's alone."""
    dep = deploy(dofs_pos_margin=0.3)
    dep.readings.append(robot_state(2, dofs_pos=[0.9, 0.0]))
    dep.step(None)
    dep.request_stop()  # the next hold latches the pose just read
    dep.step(None)

    np.testing.assert_allclose(dep.sent[-1].dofs_pos, [0.9, 0.0])
    assert np.isnan(dep.state.extra["safety.target_raw"]).all()


# ---------------------------------------------------------------------------
# the guard, on the ticks the robot is moved
# ---------------------------------------------------------------------------


def test_a_joint_that_falls_behind_its_target_stops_the_run_naming_it(deploy):
    dep = deploy(max_dofs_tracking_error={"j2": 0.1})
    start_policy(dep)
    dep.readings.append(robot_state(2, dofs_pos=[0.0, -0.3]))

    with pytest.raises(DeployStop, match=r"joint 'j2' is 0\.300 from its target \(max_dofs_tracking_error 0\.1"):
        drive(dep, [0.0, 0.0])

    assert dep.state.extra["safety.stopped"][0] == 1.0
    np.testing.assert_allclose(dep.state.extra["safety.tracking_error"], [0.0, 0.3])


def test_the_hold_records_a_limit_crossed_and_does_not_stop(deploy):
    """A robot that boots away from its hold target can be put right by hand; INIT then moves it, and is checked."""
    dep = deploy(max_dofs_tracking_error=0.1)
    dep.step(None)
    dep.readings.append(robot_state(2, dofs_pos=[0.4, 0.0]))

    dep.step(None)

    assert dep.state.extra["safety.stopped"][0] == 0.0
    assert dep.state.extra["safety.tracking_error"][0] == pytest.approx(0.4)


def test_the_init_ramp_is_checked(deploy):
    dep = deploy(max_dofs_tracking_error=0.1)
    dep.request_init()
    dep.step(None)
    dep.readings.append(robot_state(2, dofs_pos=[0.4, 0.0]))

    with pytest.raises(DeployStop, match="while the INIT ramp moved the robot: joint 'j1'"):
        dep.step(None)


def test_a_policy_that_keeps_asking_past_a_limit_stops_the_run_on_the_count(deploy):
    dep = deploy(limit_excess_steps=3, limit_excess_tolerance=0.05)
    start_policy(dep)
    drive(dep, [1.2, 0.0], [1.2, 0.0], [0.0, 0.0])  # two ticks past j1's range, then back: the count restarts
    drive(dep, [1.2, 0.0], [1.2, 0.0])

    with pytest.raises(
        DeployStop,
        match=r"on 3 tick\(s\) in a row \(limit_excess_steps=3\); on the last, joint 'j1' was 0\.200 past \[-1, 1\]",
    ):
        drive(dep, [1.2, 0.0])


def test_a_target_within_the_tolerance_past_a_limit_does_not_count(deploy):
    dep = deploy(limit_excess_steps=1, limit_excess_tolerance=0.05)
    start_policy(dep)

    drive(dep, [1.04, 0.0], [1.04, 0.0])

    assert dep.state.extra["safety.stopped"][0] == 0.0


def test_a_gap_between_readings_stops_the_run(deploy):
    times = iter([0.0, 0.02, 0.04, 0.5])
    dep = deploy(max_read_interval_s=0.1)
    dep._safety.clock = lambda: next(times)
    start_policy(dep)  # 0.0 (the handover tick)
    drive(dep, [0.0, 0.0], [0.0, 0.0])  # 0.02, 0.04

    with pytest.raises(DeployStop, match=r"0\.460 s between two readings \(max_read_interval_s=0\.1\)"):
        drive(dep, [0.0, 0.0])


def test_a_link_that_leaves_its_box_stops_the_run_naming_the_link_and_the_box(deploy, arm_env):
    z = link2_z(arm_env)
    box = LinkBoxOptions(links=("link2",), lower=(None, None, z - 0.05), upper=(None, None, z + 0.05))
    dep = deploy(link_boxes={"bench": box})
    start_policy(dep)
    drive(dep, [0.0, 0.0])
    assert dep.state.extra["safety.box.bench"][0] == pytest.approx(0.05, abs=1e-4)

    with pytest.raises(DeployStop, match=r"link 'link2' of 'robot' at \(.*\) m is outside box 'bench'"):
        drive(dep, [0.5, 0.0])  # 0.3 m from j1: link2 moves 0.14 m in z

    assert dep.state.extra["safety.box.bench"][0] < 0.0


def test_every_limit_crossed_on_the_stopping_tick_is_named(deploy, arm_env):
    z = link2_z(arm_env)
    box = LinkBoxOptions(links=("link2",), lower=(None, None, z - 0.05), upper=(None, None, z + 0.05))
    dep = deploy(link_boxes={"bench": box}, max_dofs_tracking_error=0.1)
    start_policy(dep)
    dep.readings.append(robot_state(2, dofs_pos=[0.5, 0.0]))

    with pytest.raises(DeployStop) as exc:
        drive(dep, [0.0, 0.0])

    assert "joint 'j1' is 0.500 from its target" in str(exc.value)
    assert "outside box 'bench'" in str(exc.value)


# ---------------------------------------------------------------------------
# a misnamed joint, link or entity raises when the deployment is built
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "safety, match",
    [
        ({"max_dofs_step_from_sent": {"elbow": 0.1}}, "elbow"),
        (
            {"link_boxes": {"b": LinkBoxOptions(entity_name="hand", links=("palm",))}},
            r"entity_name='hand', which the scene does not have.*The deployment syncs: \['robot'\]",
        ),
        ({"link_boxes": {"b": LinkBoxOptions(links=("palm",))}}, "palm"),
        ({"dofs_pos_margin": 0.6}, r"leaves no range on joint\(s\) \['j2'\]"),
    ],
)
def test_a_name_or_margin_the_robot_cannot_take_raises_at_construction(arm_env, safety, match):
    with deployments(arm_env) as make, pytest.raises(ValueError, match=match):
        make(safety=DeploySafetyOptions(**safety))


@pytest.mark.parametrize(
    "build, match",
    [
        (lambda: DeploySafetyOptions(dofs_pos_margin=float("nan")), "dofs_pos_margin must be >= 0, got nan"),
        (lambda: DeploySafetyOptions(limit_excess_tolerance=float("nan")), "limit_excess_tolerance must be >= 0"),
        (lambda: LinkBoxOptions(links=("link2",), upper=(None, None, float("nan"))), "z bound is NaN"),
    ],
)
def test_a_nan_limit_is_refused(build, match):
    """NaN compares False both ways, so a check written as ``< 0`` would let it through to the clamp."""
    with pytest.raises(ValueError, match=match):
        build()


def test_a_dict_is_read_as_safety_options_and_a_misspelled_field_is_refused(arm_env):
    with deployments(arm_env) as make:
        dep = make(safety={"dofs_pos_margin": 0.1})
        assert dep.safety == DeploySafetyOptions(dofs_pos_margin=0.1)
        with pytest.raises(ValueError, match="Did you mean 'dofs_pos_margin'"):
            make(safety={"dofs_pos_margins": 0.1})


def test_the_layer_reads_the_built_robots_joint_range(deploy):
    lower, upper = deploy().dofs_pos_limits()

    np.testing.assert_allclose(lower, [-1.0, -0.5])
    np.testing.assert_allclose(upper, [1.0, 0.5])
