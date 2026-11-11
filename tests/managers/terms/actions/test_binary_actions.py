"""The binary gripper controller in ``eden/managers/terms/actions/binary_actions.py``."""

from __future__ import annotations

import pytest
import torch

from eden.managers.terms.actions.binary_actions import BinaryJointController
from tests.managers.terms.conftest import PANDA_KP
from tests.support.envs import control_force_over_one_step

pytestmark = pytest.mark.requires_genesis

FINGERS = ["finger_joint1", "finger_joint2"]
OPEN, CLOSED = 0.04, 0.0  # the Panda's finger travel; it starts open


def _gripper(env, make_term, **params):
    return make_term(
        BinaryJointController,
        env,
        entity_name="robot",
        dofs_name=FINGERS,
        open_action=OPEN,
        close_action=CLOSED,
        **params,
    )


@pytest.mark.parametrize(
    ("actions", "scale"),
    [
        (torch.tensor([[True], [False]]), 1.0),
        (torch.tensor([[True], [False]]), 0.0),  # a bool bypasses the scale: True*0 must not read as "open"
        (torch.tensor([[0.5], [-0.5]]), 1.0),  # a float closes above zero
        (torch.tensor([[0.3], [-0.3]]), 2.0),  # after scaling
    ],
    ids=["bool", "bool-ignores-scale", "float", "float-scaled"],
)
def test_the_first_env_closes_and_the_second_opens(panda_env, make_term, actions, scale):
    term = _gripper(panda_env, make_term, scale=scale)
    term.compute(actions)
    torch.testing.assert_close(term.processed_action, torch.tensor([[CLOSED] * 2, [OPEN] * 2]))


def test_the_gripper_drives_its_fingers_to_the_target(panda_env, make_term):
    robot = panda_env.entities["robot"]
    term = _gripper(panda_env, make_term)
    term.compute(torch.tensor([[True], [False]]))
    term.apply_actions()
    force = control_force_over_one_step(panda_env, robot)[:, [robot.dofs_name.index(name) for name in FINGERS]]
    torch.testing.assert_close(force, torch.tensor([[PANDA_KP * (CLOSED - OPEN)] * 2, [0.0, 0.0]]))


def test_per_joint_targets_land_on_this_terms_own_positions(panda_env, make_term):
    """Dict keys are resolved against the term's ``dofs_name``, which here lists the fingers in reverse."""
    term = make_term(
        BinaryJointController,
        panda_env,
        entity_name="robot",
        dofs_name=FINGERS[::-1],
        open_action={"finger_joint1": OPEN, "finger_joint2": 0.03},
        close_action={"finger_joint*": CLOSED},
    )
    term.compute(torch.tensor([[True], [False]]))
    torch.testing.assert_close(term.processed_action, torch.tensor([[CLOSED] * 2, [0.03, OPEN]]))


# ``pin.1`` is passive; read as a regex, its exact name would also select the controlled ``pin_1``.
_PIN_MJCF = """<mujoco model="pins"><compiler angle="radian" autolimits="true"/><worldbody>
  <body name="post" pos="0 0 0.4"><geom type="box" size="0.04 0.04 0.02" mass="1"/>
    <body name="p1" pos="0 0 0.05"><joint name="pin.1" axis="0 0 1" range="-1 1"/>
      <geom type="cylinder" size="0.02 0.01" mass="0.1"/>
      <body name="p2" pos="0 0 0.03"><joint name="pin_1" axis="0 0 1" range="-1 1"/>
        <geom type="cylinder" size="0.02 0.01" mass="0.1"/></body></body></body>
</worldbody></mujoco>"""


@pytest.fixture(scope="module")
def pin_env(tmp_path_factory):
    from eden.options import EntityOptions, EnvOptions, SceneOptions
    from eden.utils.configs import EdenConfig
    from tests.support.envs import build_env

    path = tmp_path_factory.mktemp("pins") / "pins.xml"
    path.write_text(_PIN_MJCF)
    pins = EntityOptions(file=str(path), is_fixed_base=True, is_articulated=True, is_actuated=True, dofs_name=["pin_1"])
    return build_env(EdenConfig(env_options=EnvOptions(num_envs=1), scene_options=SceneOptions(robot=pins)))


def test_a_key_naming_a_joint_outside_the_term_exactly_selects_nothing(pin_env, make_term):
    with pytest.raises(ValueError, match="Not all query patterns are matched"):
        make_term(BinaryJointController, pin_env, entity_name="robot", dofs_name=["pin_1"], close_action={"pin.1": 0.5})


def test_the_open_and_close_targets_are_readable_in_this_terms_dof_order(panda_env, make_term):
    term = make_term(
        BinaryJointController,
        panda_env,
        entity_name="robot",
        dofs_name=FINGERS[::-1],
        open_action={"finger_joint1": OPEN, "finger_joint2": 0.03},
        close_action=CLOSED,
    )
    # Column i is dofs_name[i], the joint genesis_dofs_idx[i] drives.
    torch.testing.assert_close(term.open_targets, torch.tensor([[0.03, OPEN]]))
    torch.testing.assert_close(term.close_targets, torch.tensor([[CLOSED, CLOSED]]))
