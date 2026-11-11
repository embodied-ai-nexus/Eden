"""The joint-space controllers in ``eden/managers/terms/actions/joint_actions.py``.

Each controller is built against a real Panda at kp = 10, kd = 1 on every DOF (see ``panda_env``),
and what it commands is read back through ``get_dofs_control_force``: ``kp * (target - pos) - kd * vel``
for a position target, ``kd * (target - vel)`` for a velocity target, the torque itself for a force.
The arm is at its default pose and at rest unless a test moves it.
"""

from __future__ import annotations

import math

import pytest
import torch

from eden.constants import ReferenceSource
from eden.managers.action_manager import ActionManager
from eden.managers.modifiers.actions.actuators import ActionDelay, EnvelopeClip, TorqueOffset
from eden.managers.terms.actions.joint_actions import (
    ExplicitPDController,
    ExplicitVelocityController,
    ImplicitPDController,
    ImplicitVelocityController,
    NullJointAction,
    VelocityFeedforwardPDController,
    is_joint_position_controller,
)
from eden.managers.terms.actions.task_space_actions import DifferentialIKController
from eden.managers.terms.events.domain_rand import RandomizeTorqueNoise
from eden.options.managers.actions import ActionManagerOptions
from eden.options.robots.unitree_g1 import G1_DOFS29_NAME
from tests.managers.terms.conftest import PANDA_KD, PANDA_KP
from tests.support.envs import control_force_over_one_step
from tests.support.motion import build_mimic_env, manager_options, set_reference, term_options, write_library

pytestmark = pytest.mark.requires_genesis

DECIMATION = 4  # physics steps per control step in the drive loops below
TUCKED = {"joint4": -2.27, "joint6": 2.27}  # the two Panda DOFs with a non-zero default position


@pytest.fixture
def controller(panda_env, make_term):
    """``controller(cls, *dofs, **params)``: a controller on the Panda's ``dofs``, built standalone."""

    def build(cls=ImplicitPDController, *dofs, **params):
        return make_term(cls, panda_env, entity_name="robot", dofs_name=list(dofs or ("joint1", "joint2")), **params)

    return build


def _commanded(env, term):
    """Apply ``term`` and read back the control force on each of its DOFs, in its ``dofs_name`` order."""
    robot = env.entities["robot"]
    term.apply_actions()
    return control_force_over_one_step(env, robot)[:, [robot.dofs_name.index(name) for name in term.dofs_name]]


def _rows(*rows):
    return torch.tensor(rows, dtype=torch.float32)


# ---------------------------------------------------------------------------
# The position target: raw * scale + offset + reference
# ---------------------------------------------------------------------------


def test_implicit_pd_commands_the_action_on_top_of_the_default_pose(panda_env, controller):
    """``ReferenceSource.DEFAULT`` is the class default: the action is a displacement from the default pose."""
    term = controller(ImplicitPDController, *TUCKED)
    assert term.reference_source == ReferenceSource.DEFAULT
    actions = _rows([0.1, 0.2], [0.3, -0.1])
    term.compute(actions)
    torch.testing.assert_close(term.processed_action, actions + torch.tensor(list(TUCKED.values())))
    torch.testing.assert_close(_commanded(panda_env, term), PANDA_KP * actions)


def test_per_dof_scale_and_offset_and_their_inverse(controller):
    """``(processed - effective_offset) / effective_scale`` recovers the raw action (a policy's own-target obs)."""
    term = controller(scale={"joint1": 2.0, "joint2": 0.5}, offset={"joint1": 1.0, "joint2": -1.0})
    actions = _rows([10.0, 10.0], [1.0, -2.0])
    term.compute(actions)
    torch.testing.assert_close(term.processed_action, _rows([21.0, 4.0], [3.0, -2.0]))
    torch.testing.assert_close((term.processed_action - term.effective_offset) / term.effective_scale, actions)


def test_effective_offset_is_the_resolved_value_not_the_configured_field(controller):
    term = controller(ImplicitPDController, *TUCKED)
    assert term.offset == 0.0
    torch.testing.assert_close(term.effective_offset.expand(2, -1), _rows(*[list(TUCKED.values())] * 2))


def test_an_unset_scale_applies_none(controller):
    term = controller(reference_source=ReferenceSource.ZERO)
    assert term.scale is None and term.effective_scale == 1.0
    term.compute(_rows([0.5, -0.5], [0.0, 1.0]))
    torch.testing.assert_close(term.processed_action, _rows([0.5, -0.5], [0.0, 1.0]))


@pytest.mark.parametrize(
    ("source", "offset", "expected"),
    [
        (ReferenceSource.ZERO, 0.0, [[1.0, 2.0]]),
        (ReferenceSource.DEFAULT, {"joint4": 0.1, "joint6": 0.0}, [[1.0 + 0.1 - 2.27, 2.0 + 2.27]]),
        ("default", 0.0, [[1.0 - 2.27, 2.0 + 2.27]]),  # the StrEnum value as a plain string
    ],
    ids=["zero", "default-plus-offset", "default-as-string"],
)
def test_reference_source_selects_what_the_action_is_added_to(controller, source, offset, expected):
    term = controller(ImplicitPDController, *TUCKED, reference_source=source, offset=offset)
    term.compute(_rows([1.0, 2.0], [1.0, 2.0]))
    torch.testing.assert_close(term.processed_action, _rows(*expected * 2))


def test_scale_ratio_is_a_fraction_of_each_dofs_limit_range(panda_env, make_term, controller):
    robot = panda_env.entities["robot"]
    term = controller(ImplicitPDController, "joint1", "joint4", reference_source=ReferenceSource.ZERO, scale_ratio=0.25)
    lower, upper = robot.get_dofs_limit()
    span = (upper - lower)[:, [0, 3]]
    term.compute(torch.ones(2, 2))
    torch.testing.assert_close(term.processed_action, 0.25 * span)
    # An unlimited hinge falls back to a 2 pi range.
    spin = make_term(
        ImplicitPDController,
        panda_env,
        entity_name="wheel",
        dofs_name=["spin"],
        reference_source=ReferenceSource.ZERO,
        scale_ratio=0.5,
    )
    spin.compute(torch.ones(2, 1))
    torch.testing.assert_close(spin.processed_action, torch.full((2, 1), math.pi))


def test_scale_ratio_rejects_a_bool(controller):
    with pytest.raises(ValueError, match="unsupported scale_ratio type"):
        controller(scale_ratio=True)


# ---------------------------------------------------------------------------
# The other controllers
# ---------------------------------------------------------------------------


def test_explicit_pd_commands_the_pd_torque(panda_env, controller, restore_state):
    robot = restore_state(panda_env.entities["robot"])
    vel = torch.zeros(2, 9)
    vel[:, :2] = 0.5
    robot.set_dofs_vel(vel)
    term = controller(ExplicitPDController)
    term.compute(_rows([0.1, 0.2], [0.3, -0.1]))
    torch.testing.assert_close(_commanded(panda_env, term), PANDA_KP * _rows([0.1, 0.2], [0.3, -0.1]) - PANDA_KD * 0.5)


def test_implicit_velocity_commands_the_action_as_a_velocity(panda_env, controller):
    term = controller(ImplicitVelocityController)
    term.compute(_rows([1.0, 2.0], [3.0, -4.0]))
    torch.testing.assert_close(_commanded(panda_env, term), PANDA_KD * _rows([1.0, 2.0], [3.0, -4.0]))


def test_explicit_velocity_gain_is_the_force_range_times_the_velocity_gain(panda_env, controller):
    """joint5 / joint6 take ±12 N·m, so kp = 24 * 0.5 = 12 N·m per rad/s of error."""
    term = controller(ExplicitVelocityController, "joint5", "joint6", velocity_gain=0.5)
    term.compute(_rows([0.5, -0.25], [0.1, 0.0]))
    torch.testing.assert_close(_commanded(panda_env, term), 12.0 * _rows([0.5, -0.25], [0.1, 0.0]))


def test_velocity_feedforward_pd_splits_the_action_into_position_and_velocity_halves(
    panda_env, controller, restore_state
):
    """With ``ReferenceSource.DELTA`` the position half rides on the pose captured at compute()."""
    robot = restore_state(panda_env.entities["robot"])
    pos = robot.get_dofs_pos().clone()
    pos[:, [3, 5]] = torch.tensor([-2.0, 2.0])
    robot.set_dofs_pos(pos)
    term = controller(
        VelocityFeedforwardPDController, *TUCKED, reference_source=ReferenceSource.DELTA, scale=0.5, velocity_scale=2.0
    )
    term.compute(_rows([1.0, 2.0, 0.1, -0.1], [0.0, 0.0, 0.0, 0.0]))
    torch.testing.assert_close(term.processed_action[0], torch.tensor([-1.5, 3.0, 0.2, -0.2]))
    # tau = kp * (target - pos) + kd * target_vel
    torch.testing.assert_close(
        _commanded(panda_env, term)[0], PANDA_KP * torch.tensor([0.5, 1.0]) + PANDA_KD * torch.tensor([0.2, -0.2])
    )


def test_velocity_feedforward_dict_scale_covers_the_position_half_only(controller):
    term = controller(
        VelocityFeedforwardPDController,
        reference_source=ReferenceSource.ZERO,
        scale={"joint1": 0.5, "joint2": 0.25},
        velocity_scale=2.0,
    )
    assert term.effective_scale.shape == (1, 2)
    term.compute(_rows([1.0, 2.0, 0.1, -0.1], [1.0, 2.0, 0.1, -0.1]))
    torch.testing.assert_close(term.processed_action, _rows([0.5, 0.5, 0.2, -0.2], [0.5, 0.5, 0.2, -0.2]))


def test_null_joint_action_holds_the_default_of_each_of_its_dofs(panda_env, controller, restore_state):
    """It reads the entity's columns for its DOFs, not the first k (a subset used to pick up joint1's)."""
    robot = restore_state(panda_env.entities["robot"])
    pos = robot.get_dofs_pos().clone()
    pos[:, 5] = 2.0  # joint6 off its 2.27 default
    robot.set_dofs_pos(pos)
    term = controller(NullJointAction, *TUCKED)
    torch.testing.assert_close(term.processed_action, _rows(*[list(TUCKED.values())] * 2))
    torch.testing.assert_close(_commanded(panda_env, term), _rows([0.0, PANDA_KP * 0.27], [0.0, PANDA_KP * 0.27]))


def test_null_joint_action_draws_no_policy_output(controller):
    """Pinning its DOFs, it reserves no action columns; ``num_dofs`` stays, since it still commands them."""
    term = controller(NullJointAction, "joint1", "joint4")
    assert term.action_dim == 0 and term.num_dofs == 2
    assert term.action.shape == (2, 0) and term.prev_action.shape == (2, 0)


@pytest.mark.parametrize(
    ("cls", "reference_source"),
    [
        (ImplicitVelocityController, ReferenceSource.MOTION_REF),
        (ImplicitVelocityController, ReferenceSource.ACCUMULATE),
        (VelocityFeedforwardPDController, ReferenceSource.ACCUMULATE),  # overrides compute(): would never integrate
    ],
)
def test_a_reference_source_the_controller_cannot_honour_is_rejected(controller, cls, reference_source):
    with pytest.raises(ValueError, match="reference_source|accumulate"):
        controller(cls, reference_source=reference_source)


def test_is_joint_position_controller_separates_position_from_velocity_controllers(controller):
    class Subclassed(ImplicitPDController):
        pass

    for cls in (ImplicitPDController, ExplicitPDController, VelocityFeedforwardPDController, Subclassed):
        assert is_joint_position_controller(cls), cls
    for cls in (ImplicitVelocityController, ExplicitVelocityController, NullJointAction, DifferentialIKController):
        assert not is_joint_position_controller(cls), cls
    assert is_joint_position_controller(controller())


# ---------------------------------------------------------------------------
# ReferenceSource.DELTA / ReferenceSource.ACCUMULATE
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("cls", [ImplicitPDController, ExplicitPDController])
def test_delta_adds_the_pose_captured_at_compute(panda_env, controller, restore_state, cls):
    robot = restore_state(panda_env.entities["robot"])
    pos = robot.get_dofs_pos().clone()
    pos[:, [3, 5]] = torch.tensor([-2.0, 2.0])
    robot.set_dofs_pos(pos)
    term = controller(
        cls, *TUCKED, reference_source=ReferenceSource.DELTA, scale=0.25, offset={"joint4": 0.1, "joint6": -0.1}
    )
    term.compute(_rows([2.0, 4.0], [2.0, 4.0]))
    torch.testing.assert_close(term.processed_action, _rows([-1.4, 2.9], [-1.4, 2.9]))


def test_delta_explicit_pd_passes_its_torque_through_the_modifier(panda_env, install_manager):
    """A ``TorqueOffset`` the noise event fills adds to the PD torque; the controller's reset clears it."""
    arm = ExplicitPDController.configure(
        entity_name="robot",
        dofs_name=["joint1", "joint2"],
        reference_source=ReferenceSource.DELTA,
        modifier=TorqueOffset.configure(),
    )
    manager = install_manager(panda_env, "action_manager", ActionManager(panda_env, ActionManagerOptions(arm=arm)))
    term = manager.get_term("arm")
    term.compute(torch.zeros(2, 2))
    baseline = _commanded(panda_env, term)
    noise = RandomizeTorqueNoise(panda_env, RandomizeTorqueNoise.configure(action_term_name="arm", rfi_scale=5.0))
    noise.build()
    noise.compute()
    assert not torch.allclose(_commanded(panda_env, term), baseline)
    term.reset()
    torch.testing.assert_close(_commanded(panda_env, term), baseline)


def test_accumulate_integrates_onto_a_persistent_target_clamped_to_the_soft_limits(panda_env, controller):
    """joint2's soft limits are ±1.5865 rad; the manager-facing action stays the raw delta."""
    term = controller(reference_source=ReferenceSource.ACCUMULATE, scale=0.5)
    soft = panda_env.entities["robot"].soft_dofs_pos_limits[0, 1, 1].item()
    term.compute(_rows([1.0, 2.0], [0.5, -2.0]))
    torch.testing.assert_close(term.processed_action, _rows([0.5, 1.0], [0.25, -1.0]))
    term.compute(_rows([1.0, 2.0], [0.5, -2.0]))
    torch.testing.assert_close(term.processed_action, _rows([1.0, soft], [0.5, -soft]))
    torch.testing.assert_close(term.action, _rows([1.0, 2.0], [0.5, -2.0]))

    term.reset(torch.tensor([0]))  # env 0 restarts from the default pose; env 1 keeps its target
    term.compute(torch.zeros(2, 2))
    torch.testing.assert_close(term.processed_action, _rows([0.0, 0.0], [0.5, -soft]))


def test_accumulate_rejects_a_static_offset(controller):
    """It would be re-added to the running target every step."""
    with pytest.raises(ValueError, match="offset"):
        controller(reference_source=ReferenceSource.ACCUMULATE, offset=0.5)


# ---------------------------------------------------------------------------
# set_dofs_pos_offset: the controller-owned per-env offset joint-zero randomization writes
# ---------------------------------------------------------------------------


def test_dofs_pos_offset_adds_to_the_target_and_survives_reset(controller):
    term = controller(ImplicitPDController, *TUCKED, offset=0.1)
    term.set_dofs_pos_offset(_rows([0.01, 0.02]), envs_idx=torch.tensor([1]))
    expected = _rows([-2.17, 2.37], [-2.16, 2.39])
    for _ in range(2):
        term.compute(torch.zeros(2, 2))
        torch.testing.assert_close(term.processed_action, expected)
        torch.testing.assert_close(term.effective_offset, expected)
        term.reset()


def test_dofs_pos_offset_is_added_to_an_accumulated_target_not_integrated(controller):
    term = controller(reference_source=ReferenceSource.ACCUMULATE, scale=0.5)
    term.set_dofs_pos_offset(torch.full((2, 2), 0.1))
    for _ in range(3):
        term.compute(torch.zeros(2, 2))
        torch.testing.assert_close(term.processed_action, torch.full((2, 2), 0.1))


def test_dofs_pos_offset_is_refused_by_a_velocity_controller_and_checked_for_width(controller):
    with pytest.raises(ValueError, match="commands joint velocities"):
        controller(ImplicitVelocityController).set_dofs_pos_offset(torch.zeros(2, 2))
    with pytest.raises(ValueError, match=r"offset for 3 DOFs, but the term controls 2"):
        controller().set_dofs_pos_offset(torch.zeros(2, 3))


# ---------------------------------------------------------------------------
# dict_keys_scope: one entity-wide scale / offset dict shared by terms that split the entity
# ---------------------------------------------------------------------------

WHOLE_ARM_SCALE = {"joint1": 2.0, "joint2": 0.5, "joint[34]": 3.0}  # written for the whole arm, as a preset is


def test_a_key_for_another_terms_dofs_raises_by_default_naming_the_scope_switch(controller):
    with pytest.raises(ValueError, match=r"(?s)`scale` key 'joint\[34\]' matches only DOFs.*dict_keys_scope='entity'"):
        controller(scale=WHOLE_ARM_SCALE)


def test_entity_scope_skips_keys_for_the_entitys_other_dofs(controller):
    torch.testing.assert_close(
        controller(scale=WHOLE_ARM_SCALE, dict_keys_scope="entity").effective_scale, _rows([2.0, 0.5])
    )
    offset = controller(
        reference_source=ReferenceSource.ZERO, offset={"joint1": 0.1, "joint[34]": 5.0}, dict_keys_scope="entity"
    )
    torch.testing.assert_close(offset.effective_offset, _rows([0.1, 0.0]))


# Joint names that read as patterns: ``pin.1`` is also a regex matching ``pin_1``, ``arm(2)`` one that does not
# match itself.
_PATTERN_LIKE_MJCF = """<mujoco model="pattern_like"><compiler angle="radian" autolimits="true"/><worldbody>
  <body name="post" pos="0 0 0.4"><geom type="box" size="0.04 0.04 0.02" mass="1"/>
    <body name="p1" pos="0 0 0.05"><joint name="pin.1" axis="0 0 1" range="-1 1"/>
      <geom type="cylinder" size="0.02 0.01" mass="0.1"/>
      <body name="p2" pos="0 0 0.03"><joint name="pin_1" axis="0 0 1" range="-1 1"/>
        <geom type="cylinder" size="0.02 0.01" mass="0.1"/>
        <body name="p3" pos="0 0 0.03"><joint name="arm(2)" axis="0 0 1" range="-1 1"/>
          <geom type="cylinder" size="0.02 0.01" mass="0.1"/></body></body></body></body>
</worldbody></mujoco>"""


@pytest.fixture(scope="module")
def pattern_like_env(tmp_path_factory):
    from eden.options import EntityOptions, EnvOptions, SceneOptions
    from eden.utils.configs import EdenConfig
    from tests.support.envs import build_env

    asset = tmp_path_factory.mktemp("pattern_like") / "pattern_like.xml"
    asset.write_text(_PATTERN_LIKE_MJCF)
    return build_env(
        EdenConfig(
            env_options=EnvOptions(num_envs=1),
            scene_options=SceneOptions(
                robot=EntityOptions(
                    file=str(asset),
                    is_fixed_base=True,
                    is_articulated=True,
                    is_actuated=True,
                    dofs_name=["pin.1", "pin_1", "arm(2)"],
                    default_dofs_kp={".*": 5.0},
                    default_dofs_kd={".*": 0.2},
                ),
                # ``pin.1`` passive: the entity has it but does not control it.
                passive_pin=EntityOptions(
                    file=str(asset),
                    is_fixed_base=True,
                    is_articulated=True,
                    is_actuated=True,
                    dofs_name=["pin_1", "arm(2)"],
                    default_dofs_kp={".*": 5.0},
                    default_dofs_kd={".*": 0.2},
                    default_root_pos=(0.0, 1.0, 0.0),
                ),
            ),
        )
    )


def test_a_key_that_is_exactly_a_dof_name_is_taken_literally(pattern_like_env, make_term):
    """Read as patterns, ``pin.1`` (another term's DOF) would also scale ``pin_1``, and ``arm(2)`` would match no DOF."""
    term = make_term(
        ImplicitPDController,
        pattern_like_env,
        entity_name="robot",
        dofs_name=["pin_1", "arm(2)"],
        scale={"pin.1": 9.0, "arm(2)": 0.5},
        dict_keys_scope="entity",
    )
    torch.testing.assert_close(term.effective_scale, _rows([1.0, 0.5]))


def test_a_passive_joints_exact_name_is_literal_too(pattern_like_env, make_term):
    """Read as a pattern, ``pin.1`` (a joint the entity does not control) would scale ``pin_1``."""
    with pytest.raises(ValueError, match=r"`scale` key 'pin\.1' matches no DOF of the entity"):
        make_term(
            ImplicitPDController, pattern_like_env, entity_name="passive_pin", dofs_name=["pin_1"], scale={"pin.1": 9.0}
        )


@pytest.mark.parametrize("scope", ["term", "entity"])
@pytest.mark.parametrize("field", ["scale", "offset"])
def test_a_key_matching_no_dof_of_the_entity_raises_under_either_scope(controller, field, scope):
    with pytest.raises(ValueError, match=rf"`{field}` key 'elbow_\.\*' matches no DOF of the entity"):
        controller(dict_keys_scope=scope, **{field: {"joint1": 1.0, "elbow_.*": 2.0}})


def test_an_unknown_dict_keys_scope_raises_at_build(controller):
    with pytest.raises(ValueError, match=r"dict_keys_scope must be 'term' or 'entity', got 'entities'"):
        controller(dict_keys_scope="entities")


# ---------------------------------------------------------------------------
# ActionDelay counts physics steps inside apply_actions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cls",
    [
        ImplicitPDController,
        ExplicitPDController,
        VelocityFeedforwardPDController,
        ImplicitVelocityController,
        ExplicitVelocityController,
    ],
)
def test_a_delayed_step_input_reaches_the_actuator_exactly_k_physics_steps_later(panda_env, controller, cls):
    """``k = 6`` is 1.5 control steps at decimation 4, a lag only a physics-step count can express.

    A delay advanced per ``compute()`` would land this input 24 physics steps late. The
    command is read at the actuator, so each controller's ``apply_actions()`` must route through the
    delay; ``VelocityFeedforwardPDController`` switching in one jump shows both halves move together.
    """
    k = 6
    term = controller(cls, modifier=ActionDelay.configure(delay_range_steps=(k, k)))
    held = torch.full((2, term.action_dim), 0.1)
    fresh = torch.full((2, term.action_dim), -0.4)
    step_input_at = 2
    commanded = []
    for control_step in range(5):
        term.compute(held if control_step < step_input_at else fresh)
        for _ in range(DECIMATION):
            commanded.append(_commanded(panda_env, term).clone())
    arrival = step_input_at * DECIMATION + k
    assert [not torch.equal(c, commanded[0]) for c in commanded].index(True) == arrival
    assert all(torch.equal(c, commanded[-1]) for c in commanded[arrival:])
    torch.testing.assert_close(term.processed_action, fresh)  # what the policy issued, not the delayed command


# ---------------------------------------------------------------------------
# A modifier reads the controller's own DOFs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dofs", [["joint3", "joint4"], ["joint6", "joint4"]], ids=["contiguous", "gappy-reordered"])
def test_envelope_clip_bounds_the_target_around_each_controlled_dofs_own_position(controller, dofs):
    """At rest the torque envelope allows ``±0.5 N·m``, so ``q ± 0.5 / kp = q ± 0.05`` about each DOF's position.

    The controller hands the modifier its ``dofs_idx_local``; a modifier that read other DOFs' state or gains
    (joint1 and joint2 sit at 0, joint4 and joint6 at ∓2.27) would clip about the wrong position.
    """
    envelope = EnvelopeClip.configure(
        driving_torque_limit=0.5, braking_torque_limit=0.5, full_torque_speed=1.0, no_load_speed=2.0
    )
    term = controller(ImplicitPDController, *dofs, modifier=envelope)
    home = {"joint3": 0.0, **TUCKED}
    q = torch.tensor([home[name] for name in dofs])
    term.compute(_rows([1.0, -1.0], [-1.0, 1.0]))
    torch.testing.assert_close(term.processed_action, q + 0.05 * _rows([1.0, -1.0], [-1.0, 1.0]))


# ---------------------------------------------------------------------------
# ReferenceSource.MOTION_REF: a residual on the motion reference's current frame
# ---------------------------------------------------------------------------

MOTION_DOFS = [name for name in G1_DOFS29_NAME if name != "waist_yaw_joint"]  # one G1 DOF left out of the reference


@pytest.fixture(scope="module")
def mimic_env(tmp_path_factory):
    """A G1 (``robot``) tracked on every DOF but ``waist_yaw_joint``, a box, and an untracked Panda (``arm``)."""
    from eden.options.entities import BoxOptions
    from eden.options.robots.franka_emika_panda import FrankaEmikaPanda
    from eden.options.robots.unitree_g1 import G1Dofs29

    root = write_library(tmp_path_factory.mktemp("motion_ref"), [20])
    return build_mimic_env(
        root,
        num_envs=2,
        scene={
            "robot": G1Dofs29(),
            "object": BoxOptions(size=(0.1, 0.1, 0.1), default_root_pos=(1.0, 0.0, 0.05)),
            "arm": FrankaEmikaPanda(default_root_pos=(0.0, 2.0, 0.0)),
        },
        motion_options=manager_options(terms={"default": term_options(root, dofs={"robot": MOTION_DOFS})}),
    )


@pytest.fixture
def reference(mimic_env):
    """The reference's current-frame DOF positions: ``0.01 j`` in env 0 and ``-0.01 j`` in env 1, by column ``j``."""
    values = 0.01 * torch.arange(len(MOTION_DOFS), dtype=torch.float32)
    set_reference(mimic_env, "robot", dofs_pos=torch.stack([values, -values]).unsqueeze(1))
    return lambda name: torch.tensor([0.01, -0.01]) * MOTION_DOFS.index(name)


@pytest.mark.parametrize(
    ("cls", "params", "static"),
    [
        (ImplicitPDController, {"scale": 0.25}, [0.0, 0.0]),
        (ImplicitPDController, {"offset": {"left_knee_joint": 0.1, "left_hip_pitch_joint": -0.1}}, [0.1, -0.1]),
        (ExplicitPDController, {"scale": 0.5}, [0.0, 0.0]),
    ],
    ids=["implicit-scaled", "implicit-static-offset", "explicit"],
)
def test_motion_ref_adds_the_reference_dofs_in_the_controllers_order(
    mimic_env, make_term, reference, cls, params, static
):
    """The controller's DOFs are a reordered subset of the reference's: each reads its own column."""
    dofs = ["left_knee_joint", "left_hip_pitch_joint"]
    term = make_term(
        cls, mimic_env, entity_name="robot", dofs_name=dofs, reference_source=ReferenceSource.MOTION_REF, **params
    )
    actions = _rows([2.0, 4.0], [-2.0, 0.0])
    term.compute(actions)
    scale = params.get("scale", 1.0)
    expected = actions * scale + torch.tensor(static) + torch.stack([reference(name) for name in dofs], dim=-1)
    torch.testing.assert_close(term.processed_action, expected)


@pytest.mark.parametrize(
    ("env_name", "entity", "dofs", "error", "match"),
    [
        ("panda_env", "robot", ["joint1"], RuntimeError, "MimicRLEnvBase"),
        ("mimic_env", "arm", ["joint1"], ValueError, "not tracked by any motion term"),
        (
            "mimic_env",
            "robot",
            ["left_knee_joint", "waist_yaw_joint"],
            ValueError,
            "not present in the motion reference",
        ),
    ],
    ids=["plain-env", "untracked-entity", "dof-outside-the-reference"],
)
def test_motion_ref_needs_a_tracked_entity_on_a_mimic_env(request, make_term, env_name, entity, dofs, error, match):
    env = request.getfixturevalue(env_name)
    with pytest.raises(error, match=match):
        make_term(
            ImplicitPDController, env, entity_name=entity, dofs_name=dofs, reference_source=ReferenceSource.MOTION_REF
        )


def test_motion_ref_resolves_the_drivers_only_and_derives_the_followers_from_them(mimic_env, make_term, reference):
    """A follower outside the reference (``waist_yaw_joint``) is fine: it reads its driver's target, not a column."""
    term = make_term(
        ImplicitPDController,
        mimic_env,
        entity_name="robot",
        dofs_name=["left_knee_joint"],
        reference_source=ReferenceSource.MOTION_REF,
        mimic_dofs={"waist_yaw_joint": ("left_knee_joint", 0.5, 0.1)},
    )
    term.compute(_rows([2.0], [-2.0]))
    knee = 2.0 * torch.tensor([1.0, -1.0]) + reference("left_knee_joint")
    torch.testing.assert_close(term.processed_action, torch.stack([knee, 0.5 * knee + 0.1], dim=-1))


# ---------------------------------------------------------------------------
# mimic_dofs: followers commanded from their drivers' targets
# ---------------------------------------------------------------------------

#: The finger's coupling: ``b`` follows ``a`` at twice its angle, 0.1 rad on.
FINGER_MIMIC = {"b": ("a", 2.0, 0.1)}


@pytest.fixture(scope="module")
def finger_env(tmp_path_factory):
    """Two fixed palms: ``finger`` with ``b`` free to be coupled in software, ``tagged`` with ``b`` URDF-mimicked."""
    from eden.options import EntityOptions, EnvOptions, SceneOptions
    from eden.utils.configs import EdenConfig
    from tests.managers.terms.conftest import write_mimic_finger_urdf
    from tests.support.envs import build_env

    root = tmp_path_factory.mktemp("finger")

    def finger(name, pos, mimic_tag):
        return EntityOptions(
            file=write_mimic_finger_urdf(root / f"{name}.urdf", mimic_tag=mimic_tag),
            is_fixed_base=True,
            is_articulated=True,
            is_actuated=True,
            default_root_pos=pos,
            dofs_name=["a", "b", "c"],
            default_dofs_kp={".*": 5.0},
            default_dofs_kd={".*": 0.2},
            default_dofs_armature={".*": 0.01},
        )

    return build_env(
        EdenConfig(
            env_options=EnvOptions(num_envs=2, sim_dt=0.005, sim_substeps=1),
            scene_options=SceneOptions(
                finger=finger("finger", (0.0, 0.0, 1.0), False), tagged=finger("tagged", (1.0, 0.0, 1.0), True)
            ),
        )
    )


@pytest.fixture
def finger_term(finger_env, make_term):
    """``finger_term(cls=ImplicitPDController, **params)``: a controller on the finger's drivers ``a``, ``c``."""

    def build(cls=ImplicitPDController, **params):
        params = {"dofs_name": ["a", "c"], "mimic_dofs": FINGER_MIMIC, **params}
        return make_term(cls, finger_env, entity_name="finger", **params)

    return build


def test_a_follower_takes_no_action_column_but_is_commanded_after_the_drivers(finger_term):
    term = finger_term()
    assert (term.action_dim, term.num_dofs) == (2, 3)
    assert term.dofs_name == ["a", "c", "b"]
    assert term.mimic_follower_dofs_name == ["b"]
    assert tuple(term.action.shape) == (2, 2) and tuple(term.processed_action.shape) == (2, 3)


@pytest.mark.parametrize("cls", [ImplicitPDController, ExplicitPDController])
def test_the_follower_target_is_its_drivers_full_target_times_the_multiplier_plus_the_offset(finger_term, cls):
    """``a``'s target is ``raw * scale + offset``; ``b``'s is that times 2, plus 0.1."""
    term = finger_term(cls, scale=0.5, offset={"a": 0.2, "c": 0.0}, reference_source=ReferenceSource.ZERO)
    term.compute(_rows([0.4, -0.2], [-0.8, 0.6]))
    a = torch.tensor([0.4 * 0.5 + 0.2, -0.8 * 0.5 + 0.2])
    c = torch.tensor([-0.1, 0.3])
    torch.testing.assert_close(term.processed_action, torch.stack([a, c, 2.0 * a + 0.1], dim=-1))
    # effective_* stay driver-wide, so the usual inverse still recovers the action.
    torch.testing.assert_close(
        (term.processed_action[:, :2] - term.effective_offset) / term.effective_scale, _rows([0.4, -0.2], [-0.8, 0.6])
    )


def test_the_explicit_controller_drives_the_follower_at_its_own_gain(finger_env, finger_term, restore_state):
    """The cached gains widen with the command: ``b``'s torque is ``kp * (target - pos)`` at rest."""
    restore_state(finger_env.entities["finger"])
    term = finger_term(ExplicitPDController, reference_source=ReferenceSource.ZERO)
    term.compute(_rows([0.3, 0.0], [0.3, 0.0]))
    term.apply_actions()
    finger = finger_env.entities["finger"]
    torque = control_force_over_one_step(finger_env, finger)[:, finger.dofs_name.index("b")]
    torch.testing.assert_close(torque, torch.full((2,), 5.0 * (2.0 * 0.3 + 0.1)))


def test_delta_captures_the_drivers_and_the_follower_derives_from_the_result(finger_env, finger_term, restore_state):
    finger = restore_state(finger_env.entities["finger"])
    finger.set_dofs_pos(torch.tensor([[0.2, 0.9, -0.3]]).expand(2, -1).contiguous())
    term = finger_term(reference_source=ReferenceSource.DELTA, scale=0.5)
    term.compute(_rows([0.2, 0.2], [0.2, 0.2]))
    a, c = 0.2 + 0.1, -0.3 + 0.1
    torch.testing.assert_close(term.processed_action, _rows([a, c, 2.0 * a + 0.1], [a, c, 2.0 * a + 0.1]))


def test_dofs_pos_offset_is_per_commanded_dof_and_added_after_the_coupling(finger_term):
    """The follower's offset is its own: it is not multiplied through the coupling."""
    term = finger_term(reference_source=ReferenceSource.ZERO)
    term.set_dofs_pos_offset(_rows([0.01, 0.02, 0.03], [0.01, 0.02, 0.03]))
    term.compute(_rows([0.1, 0.0], [0.1, 0.0]))
    torch.testing.assert_close(term.processed_action, _rows([0.11, 0.02, 0.33], [0.11, 0.02, 0.33]))
    torch.testing.assert_close(term.effective_offset, _rows([0.01, 0.02], [0.01, 0.02]))


def test_the_coupled_finger_tracks_in_simulation(finger_env, finger_term, restore_state):
    """Stepped under the implicit PD, the follower settles at twice its driver's angle, 0.1 rad on."""
    finger = restore_state(finger_env.entities["finger"])
    term = finger_term(reference_source=ReferenceSource.ZERO)
    term.compute(_rows([0.4, -0.3], [0.4, -0.3]))
    for _ in range(300):
        term.apply_actions()
        finger_env.scene.step()
    pos = finger.get_dofs_pos()
    a, b = pos[:, finger.dofs_name.index("a")], pos[:, finger.dofs_name.index("b")]
    torch.testing.assert_close(a, torch.full((2,), 0.4), atol=0.03, rtol=0.0)
    torch.testing.assert_close(b, 2.0 * a + 0.1, atol=0.03, rtol=0.0)


@pytest.mark.parametrize(
    ("params", "match"),
    [
        ({"dofs_name": ["a", "b", "c"]}, "both a `mimic_dofs` follower and in `dofs_name`"),
        ({"dofs_name": ["c"]}, "'a', which is not in this term's `dofs_name`"),
        ({"mimic_dofs": {"nope": ("a", 1.0, 0.0)}}, "follower 'nope' is not a DOF of the entity"),
        ({"mimic_dofs": {"b": ("a", 1.0, 0.0), "c": ("b", 1.0, 0.0)}, "dofs_name": ["a"]}, "itself a follower"),
        ({"mimic_dofs": {"b": ("a", float("nan"), 0.0)}}, "non-finite"),
        ({"mimic_dofs": {"b": ("a", 1.0)}}, r"must be \(driver, multiplier, offset\)"),
    ],
    ids=["follower-is-driver", "driver-not-an-action-column", "unknown-follower", "chained", "nan", "two-tuple"],
)
def test_a_misdeclared_coupling_is_refused_at_build(finger_term, params, match):
    with pytest.raises(ValueError, match=match):
        finger_term(**params)


@pytest.mark.parametrize(
    "cls", [ImplicitVelocityController, ExplicitVelocityController, VelocityFeedforwardPDController]
)
def test_mimic_dofs_is_refused_where_the_action_is_not_one_position_per_dof(finger_term, cls):
    params = {} if cls is VelocityFeedforwardPDController else {"reference_source": ReferenceSource.ZERO}
    with pytest.raises(ValueError, match="joint position controllers"):
        finger_term(cls, **params)


def test_a_follower_the_urdf_already_mimics_is_refused_naming_both_fixes(finger_env, make_term):
    """Genesis made ``tagged``'s ``<mimic>`` an equality; a PD target on ``b`` would fight it."""
    assert any(eq.name == "mimic_b_to_a" for eq in finger_env.entities["tagged"]._entity.equalities)
    with pytest.raises(ValueError, match=r"joint equalities \['mimic_b_to_a'\].*<mimic> tags stripped"):
        make_term(ImplicitPDController, finger_env, entity_name="tagged", dofs_name=["a", "c"], mimic_dofs=FINGER_MIMIC)


def test_another_term_commanding_a_follower_is_an_overlap(finger_env):
    arm = ImplicitPDController.configure(entity_name="finger", dofs_name=["a"], mimic_dofs=FINGER_MIMIC)
    other = ImplicitPDController.configure(entity_name="finger", dofs_name=["b", "c"])
    with pytest.raises(ValueError, match=r"'coupled' \(ImplicitPDController\) and 'other'.*\['b'\]"):
        ActionManager(finger_env, ActionManagerOptions(coupled=arm, other=other))


def test_a_coupled_term_cannot_be_deployed_yet(finger_env, finger_term):
    from eden.extensions.deployment.base import build_term_action_maps

    term = finger_term()
    with pytest.raises(ValueError, match=r"uses `mimic_dofs`: \['b'\]"):
        build_term_action_maps({"hand": term}, ["a", "b", "c"], finger_env.device, entity_name="finger")
