"""Tests for ``ActionManager`` and the ``ActionTerm`` base: the action vector's layout, compute/apply/reset,
and which term sets are accepted at construction.

The scene is a Franka Panda driven by two real ``ImplicitPDController`` terms (the arm with
``reference_source=ZERO``, so its target is the raw action; the fingers offset from their 0.04 m
default), two copies of a two-hinge wheel (identical DOF names on two entities) and a free-floating
body with two hinges.
"""

from __future__ import annotations

import pytest
import torch

from eden.constants import ReferenceSource
from eden.managers.action_manager import ACTION_TERM_REGISTRY, ActionManager, ActionTerm
from eden.managers.terms.actions.binary_actions import BinaryJointController, GripperDecisionController
from eden.managers.terms.actions.free_base_actions import FreeBasePoseController
from eden.managers.terms.actions.joint_actions import ImplicitPDController, NullJointAction
from eden.managers.terms.actions.welding_actions import SuctionCupWelding
from eden.options import ActionManagerOptions, EntityOptions, EnvOptions, SceneOptions
from eden.options.robots import FrankaEmikaPanda
from eden.utils.configs import EdenRLConfig
from tests.support.envs import build_env

pytestmark = pytest.mark.requires_genesis

NUM_ENVS = 2
ARM = [f"joint{i}" for i in range(1, 8)]
FINGERS = ["finger_joint1", "finger_joint2"]
FREE_JOINT = "floating_base_joint"

WHEEL_MJCF = """
<mujoco model="wheel">
  <compiler angle="radian"/>
  <worldbody>
    <body name="hub">
      <joint name="spin" type="hinge" axis="0 0 1" limited="false"/>
      <geom type="cylinder" size="0.1 0.02" mass="1.0"/>
      <body name="flap_link" pos="0.2 0 0">
        <joint name="flap" type="hinge" axis="0 1 0" range="-1 1" limited="true"/>
        <geom type="box" size="0.05 0.05 0.01" mass="0.2"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""
FLOATER_MJCF = f"""
<mujoco model="floater">
  <compiler angle="radian"/>
  <worldbody>
    <body name="base">
      <freejoint name="{FREE_JOINT}"/>
      <geom type="box" size="0.1 0.1 0.1" mass="1.0"/>
      <body name="arm" pos="0.2 0 0">
        <joint name="hinge_a" type="hinge" axis="0 0 1" range="-1 1"/>
        <geom type="box" size="0.05 0.05 0.05" mass="0.2"/>
        <body name="forearm" pos="0.1 0 0">
          <joint name="hinge_b" type="hinge" axis="0 0 1" range="-1 1"/>
          <geom type="box" size="0.05 0.05 0.05" mass="0.2"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


def _options() -> ActionManagerOptions:
    return ActionManagerOptions(
        arm=ImplicitPDController.configure(entity_name="robot", dofs_name=ARM, reference_source=ReferenceSource.ZERO),
        gripper=ImplicitPDController.configure(entity_name="robot", dofs_name=FINGERS),
    )


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("assets")
    (root / "wheel.xml").write_text(WHEEL_MJCF)
    (root / "floater.xml").write_text(FLOATER_MJCF)
    gains = {"default_dofs_kp": {".*": 10.0}, "default_dofs_kd": {".*": 1.0}}
    wheel = dict(
        file=str(root / "wheel.xml"),
        is_fixed_base=True,
        is_articulated=True,
        is_actuated=True,
        dofs_name=["spin", "flap"],
        default_dofs_pos={"spin": 0.0, "flap": 0.0},
        **gains,
    )
    return build_env(
        EdenRLConfig(
            env_options=EnvOptions(num_envs=NUM_ENVS),
            scene_options=SceneOptions(
                robot=FrankaEmikaPanda(),
                wheel=EntityOptions(default_root_pos=(3.0, 0.0, 0.5), **wheel),
                wheel_1=EntityOptions(default_root_pos=(-3.0, 0.0, 0.5), **wheel),
                floater=EntityOptions(
                    file=str(root / "floater.xml"),
                    is_fixed_base=False,
                    is_articulated=True,
                    is_actuated=True,
                    default_root_pos=(0.0, 3.0, 1.0),
                    dofs_name=[".*"],
                    **gains,
                ),
            ),
            action_options=_options(),
        ),
        reset=True,
    )


@pytest.fixture
def manager(env):
    """The env's own action manager, reset."""
    env.action_manager.reset()
    return env.action_manager


def _actions(*rows) -> torch.Tensor:
    return torch.tensor(rows, dtype=torch.float32)


def test_the_action_vector_is_laid_out_term_by_term_in_declaration_order(manager):
    action = torch.arange(9, dtype=torch.float32).expand(NUM_ENVS, 9) * 0.01

    manager.compute(action)

    assert (manager.total_action_dim, manager.action_term_dim) == (9, [7, 2])
    torch.testing.assert_close(manager.action, action)
    torch.testing.assert_close(manager.prev_action, torch.zeros(NUM_ENVS, 9))
    # What each term commands, read off the public `processed_action`.
    torch.testing.assert_close(manager.processed_actions["arm"], action[:, :7])
    torch.testing.assert_close(manager.processed_actions["gripper"], 0.04 + action[:, 7:])


def test_a_dict_action_writes_only_its_terms_columns_and_shifts_its_prev_action(manager):
    first, second = _actions([0.001, 0.002], [0.003, 0.004]), _actions([0.005, 0.006], [0.007, 0.008])

    manager.compute({"gripper": first})
    manager.compute({"gripper": second})

    torch.testing.assert_close(manager.action[:, 7:], second)
    torch.testing.assert_close(manager.prev_action[:, 7:], first)
    torch.testing.assert_close(manager.action[:, :7], torch.zeros(NUM_ENVS, 7))
    torch.testing.assert_close(manager.get_term("gripper").action, second)
    torch.testing.assert_close(manager.get_term("gripper").prev_action, first)


def test_a_wrong_width_action_names_the_per_term_layout(manager):
    """Two totals alone do not say which term owns which columns."""
    with pytest.raises(ValueError, match=r"Invalid action shape, expected: 9, received: 10.*\[arm=7, gripper=2\]"):
        manager.compute(torch.zeros(NUM_ENVS, 10))


def test_a_command_of_the_wrong_width_is_named_once_per_term(env):
    """Genesis' opaque "Invalid input shape" names neither the term nor the DOFs; this error names both.

    Checked once, since the width is structural; a shape-less command is Genesis' to handle.
    """
    fresh = ActionManager(env=env, options=_options())
    arm, gripper = fresh.get_term("arm"), fresh.get_term("gripper")

    with pytest.raises(ValueError, match=r"ImplicitPDController on entity 'robot' .* width 2, but it controls 7 DOFs"):
        arm.verify_command_shape(torch.zeros(NUM_ENVS, 2))
    arm.verify_command_shape(torch.zeros(NUM_ENVS, 2))  # already checked: no second raise

    gripper.verify_command_shape(0.0)
    gripper.verify_command_shape(torch.zeros(NUM_ENVS, 99))  # the scalar consumed the one check


def test_in_smoke_mode_a_bad_dict_action_names_the_term(manager):
    """Under pytest (``PYTEST_VERSION``) this is smoke mode, where bare torch/KeyErrors become named ones."""
    with pytest.raises(ValueError, match=r"Action dict for term 'gripper' has width 3, but the term expects 2"):
        manager.compute({"gripper": torch.zeros(NUM_ENVS, 3)})
    with pytest.raises(KeyError, match=r"Unknown action term 'hand'. Active action terms: \['arm', 'gripper'\]"):
        manager.compute({"hand": torch.zeros(NUM_ENVS, 2)})


def test_outside_smoke_mode_the_dict_action_goes_unchecked(env, monkeypatch):
    """Real training pays nothing per step for it: an unknown key is the bare ``KeyError`` again."""
    monkeypatch.setenv("EDEN_SMOKE", "0")  # read once, at construction
    training = ActionManager(env=env, options=_options())

    with pytest.raises(KeyError) as excinfo:
        training.compute({"hand": torch.zeros(NUM_ENVS, 2)})
    assert excinfo.value.args == ("hand",)


def test_reset_clears_the_given_rows_of_the_manager_and_of_every_term(manager):
    manager.compute(torch.full((NUM_ENVS, 9), 0.01))
    manager.compute(torch.full((NUM_ENVS, 9), 0.02))

    manager.reset(envs_idx=torch.tensor([0]))

    assert manager.action[:, 0].tolist() == pytest.approx([0.0, 0.02])
    assert manager.prev_action[:, 0].tolist() == pytest.approx([0.0, 0.01])
    for name in manager.active_terms:
        assert manager.get_term(name).action[:, 0].tolist() == pytest.approx([0.0, 0.02]), name


def test_applied_actions_drive_the_joints_toward_the_commanded_target(env):
    """Through the real step: compute, ``apply_actions`` every physics substep, and the joints move."""
    env.reset()
    robot = env.entities["robot"]
    start = robot.get_dofs_pos()[:, 0].clone()
    action = torch.zeros(NUM_ENVS, 9)
    action[:, :7] = robot.get_dofs_pos()[:, :7]
    action[:, 0] += 0.3  # joint1, 0.3 rad from where it stands

    for _ in range(10):
        env.step(action)

    moved = robot.get_dofs_pos()[:, 0] - start
    assert (moved > 0.15).all(), moved
    env.reset()


def test_processed_actions_maps_every_term_and_has_no_singular_twin(manager):
    """Plural on the manager, singular on the term: ``manager.processed_action[:, i]`` must be an
    ``AttributeError``, not a ``KeyError`` from a tensor-style slice on a dict.
    """
    processed = manager.processed_actions

    assert set(processed) == set(manager.active_terms)
    for name, target in processed.items():
        assert target is manager.get_term(name).processed_action
        assert target.shape == (NUM_ENVS, manager.get_term(name).num_dofs)
    assert not hasattr(manager, "processed_action")


# ---------------------------------------------------------------------------
# Which term sets construct
# ---------------------------------------------------------------------------


def _manager(env, **terms):
    return ActionManager(env=env, options=ActionManagerOptions(**terms))


def _pd(entity="robot", dofs=(), cls=ImplicitPDController):
    return cls.configure(entity_name=entity, dofs_name=list(dofs))


def test_a_pinned_term_takes_no_slice_of_the_action_vector(env):
    """``total_action_dim`` is what the RL wrappers publish as ``num_actions``: pinned DOFs must not widen it."""
    manager = _manager(
        env, pinned=_pd(dofs=["joint1", "joint4"], cls=NullJointAction), driven=_pd(dofs=["joint2", "joint3"])
    )
    assert manager.total_action_dim == 2 and manager.action_term_dim == [0, 2]
    assert manager.total_dofs_dim == 4  # all four DOFs are still commanded
    manager.compute(torch.zeros(2, 2))
    manager.apply_actions()
    assert manager.processed_actions["pinned"].shape == (2, 2)


def test_a_term_that_neither_draws_actions_nor_commands_dofs_is_rejected(env):
    """A ``dofs_name``-less joint term would build, train and deploy as a silent no-op."""
    with pytest.raises(ValueError, match="command no DOFs"):
        _manager(env, driven=_pd(dofs=["joint1"]), forgotten=_pd())
    with pytest.raises(ValueError, match="command no DOFs"):
        _manager(env, pinned=_pd(cls=NullJointAction))


def test_a_term_that_only_draws_or_only_commands_is_accepted(env):
    """``NullJointAction`` commands DOFs from no action columns; a weld owns an action column and no DOFs."""
    pinned = _manager(env, pinned=_pd(dofs=["joint1"], cls=NullJointAction))
    assert (pinned.total_action_dim, pinned.total_dofs_dim) == (0, 1)
    weld = _manager(env, weld=SuctionCupWelding.configure(entity_name="robot", ee_link_name="hand"))
    assert (weld.total_action_dim, weld.total_dofs_dim) == (1, 0)


@ACTION_TERM_REGISTRY.register()
class SubstepPush(ActionTerm):
    """A disturbance that acts from ``apply_actions`` on every physics sub-step: no action column, no DOF."""

    _acts_without_action_ = True

    def compute(self, actions: torch.Tensor) -> None:
        pass

    def apply_actions(self) -> None:
        force = torch.tensor([0.0, 0.0, 5.0]).expand(self.num_envs, 1, 3)
        self._env.rigid_solver.apply_links_external_wrench(force=force, links_idx=[self.entity.entity.base_link_idx])


@ACTION_TERM_REGISTRY.register()
class UndeclaredSubstepPush(SubstepPush):
    _acts_without_action_ = False


def test_a_term_that_acts_without_action_is_accepted_only_when_it_declares_so(env):
    push = _manager(env, push=SubstepPush.configure(entity_name="floater"))
    assert (push.total_action_dim, push.total_dofs_dim) == (0, 0)
    push.compute(torch.zeros(NUM_ENVS, 0))
    push.apply_actions()
    with pytest.raises(ValueError, match="command no DOFs") as exc:
        _manager(env, push=UndeclaredSubstepPush.configure(entity_name="floater"))
    assert "`_acts_without_action_ = True`" in str(exc.value) and "Do NOT set" in str(exc.value)


def test_a_term_naming_a_multi_dof_joint_is_rejected(env):
    """One name must mean one DOF: a free joint resolved to 36 columns commanding its last DOF six times over."""
    with pytest.raises(ValueError, match="free or ball joint") as exc:
        _manager(env, base=_pd(entity="floater", dofs=[FREE_JOINT]))
    message = str(exc.value)
    assert "ImplicitPDController" in message and f"'{FREE_JOINT}'" in message
    assert "(6 DOFs, resolved to 36 columns)" in message


def test_an_unknown_entity_name_lists_the_entities_the_scene_has(env):
    """Not a bare ``KeyError: 'robt'``, which names neither the typo's fix nor the scene."""
    with pytest.raises(KeyError, match=r"No scene entity named 'robt'; this env has \[.*'robot'.*\]"):
        _manager(env, arm=_pd(entity="robt", dofs=["joint1"]))


@pytest.mark.parametrize(
    ("term", "subject"),
    [
        (
            GripperDecisionController.configure(entity_name="robot", dofs_name=FINGERS, input="percent"),
            "GripperDecisionController on entity 'robot': input='percent'",
        ),
        (
            FreeBasePoseController.configure(entity_name="floater", dofs_name=["hinge_a"]),
            "FreeBasePoseController on entity 'floater' drives the entity's free joint",
        ),
    ],
    ids=["gripper_decision", "free_base_pose"],
)
def test_a_configuration_error_names_the_term_class_and_its_entity(env, term, subject):
    """An action term opens its configuration errors with ``term_subject``: ``"<Class> on entity '<name>'"``."""
    with pytest.raises(ValueError) as exc:
        _manager(env, term=term)
    assert str(exc.value).startswith(subject)


def test_a_lone_term_on_a_multi_dof_joint_is_not_reported_as_overlapping_itself(env):
    with pytest.raises(ValueError) as exc:
        _manager(env, base=_pd(entity="floater", dofs=[".*"]))
    assert "overlap on the DOFs" not in str(exc.value) and "free or ball joint" in str(exc.value)


def test_single_dof_joints_of_a_floating_base_entity_build(env):
    manager = _manager(
        env,
        arm=_pd(entity="floater", dofs=["hinge_a"]),
        hold=_pd(entity="floater", dofs=["hinge_b"], cls=NullJointAction),
    )
    assert manager.total_dofs_dim == 2


def test_two_terms_owning_the_same_dof_are_rejected(env):
    """``control_dofs_*`` replaces a DOF's target, so overlapping ownership is last-writer-wins."""
    with pytest.raises(ValueError, match="overlap on the DOFs") as exc:
        _manager(env, dofs_pos_controller=_pd(dofs=ARM), arms=_pd(dofs=["joint4", "joint5"], cls=NullJointAction))
    message = str(exc.value)
    assert "'dofs_pos_controller'" in message and "'arms'" in message
    assert "'joint4', 'joint5'" in message and "joint1" not in message  # only the shared DOFs


def test_the_overlap_report_names_every_term_involved(env):
    with pytest.raises(ValueError, match="overlap on the DOFs") as exc:
        _manager(
            env,
            first=_pd(dofs=["joint1", "joint2"]),
            second=_pd(dofs=["joint2", "joint3"]),
            third=_pd(dofs=["joint2"], cls=NullJointAction),
        )
    message = str(exc.value)
    assert message.count("'first'") == 2 and "'second'" in message and "'third'" in message


@pytest.mark.parametrize(
    "terms",
    [
        {
            "arm": _pd(dofs=ARM),
            "gripper": BinaryJointController.configure(entity_name="robot", dofs_name=["finger_joint1"]),
        },
        {"a": _pd(entity="wheel", dofs=["flap"]), "b": _pd(entity="wheel_1", dofs=["flap"])},  # ownership is per entity
        {"arm": _pd(dofs=ARM), "weld": SuctionCupWelding.configure(entity_name="robot", ee_link_name="hand")},
    ],
    ids=["arm-and-gripper", "same-name-two-entities", "zero-dof-weld"],
)
def test_disjoint_ownership_is_accepted(env, terms):
    manager = _manager(env, **terms)
    assert manager.total_dofs_dim == sum(len(t.dofs_name) for t in terms.values())
