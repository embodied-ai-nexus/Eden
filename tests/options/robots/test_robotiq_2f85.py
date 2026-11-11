"""``Robotiq2f85`` built: the grasp compensation reaches the solver, and a 5 cm box is pinched and held.

Genesis solves the four-bar loop closures softly, drops the MJCF springref and has no runtime tendon, so
the stock asset penetrates and limit-cycles instead of pinching (see the class docstring).
"""

from __future__ import annotations

import genesis as gs
import numpy as np
import pytest
import torch

from eden.options import BoxOptions, EnvOptions, PlaneOptions, SceneOptions
from eden.options.materials import RigidMaterialOptions
from eden.options.robots import Robotiq2f85
from eden.utils.configs import EdenConfig
from tests.options.robots.conftest import build_scene
from tests.support.envs import build_env

pytestmark = pytest.mark.requires_genesis

_DRIVERS = ["left_driver_joint", "right_driver_joint"]
_PASSIVE = ["left_spring_link_joint", "left_follower", "right_spring_link_joint", "right_follower_joint"]
_SPRING_LINKS = ["left_spring_link_joint", "right_spring_link_joint"]
# Substep dt 0.001 so the stiffened timeconst 0.002 survives Genesis' 2 * substep_dt clamp.
_STIFF_STEP = dict(sim_dt=0.004, sim_substeps=4)


def _gripper_env(**overrides):
    return build_scene(
        **_STIFF_STEP, plane=PlaneOptions(), gripper=Robotiq2f85(default_root_pos=(0.0, 0.0, 0.3), **overrides)
    )


@pytest.fixture(scope="module")
def env():
    return _gripper_env()


def _dofs(gripper, names):
    return gripper.find_named_dofs_idx_local(names)[1]


def test_joint_dissipation_drive_cap_and_spring_servo_reach_the_solver(env):
    gripper = env.entities["gripper"]
    drivers, passive, springs = (_dofs(gripper, names) for names in (_DRIVERS, _PASSIVE, _SPRING_LINKS))
    expected = [
        (gripper.get_dofs_damping(dofs_idx_local=drivers), 1.0),
        (gripper.get_dofs_damping(dofs_idx_local=passive), 0.05),
        (gripper.get_dofs_frictionloss(dofs_idx_local=drivers), 0.1),
        (gripper.get_dofs_frictionloss(dofs_idx_local=passive), 0.01),
        (gripper.get_dofs_armature(dofs_idx_local=passive), 0.002),
        # MuJoCo's tendon forcerange 5 N x wrap coef 0.485.
        (gripper.get_dofs_force_range(dofs_idx_local=drivers)[1], 2.4),
        (-gripper.get_dofs_force_range(dofs_idx_local=drivers)[0], 2.4),
        # The springref Genesis drops, re-anchored as a kp servo at the reference.
        (gripper.get_dofs_stiffness(dofs_idx_local=springs), 0.0),
        (gripper.get_dofs_kp(dofs_idx_local=springs), 0.05),
    ]
    for got, value in expected:
        assert torch.allclose(got, torch.tensor(value)), (got, value)


def test_loop_closures_and_pad_contacts_are_stiffened_and_nothing_else(env):
    gripper, solver = env.entities["gripper"], env.rigid_solver
    assert solver.n_equalities == 3  # 2 connects + 1 driver coupling
    eq_sol = solver.get_sol_params(eqs_idx=np.arange(3))
    assert torch.allclose(eq_sol[..., 0], torch.tensor(0.002))  # timeconst survived the clamp
    assert torch.allclose(eq_sol[..., 3], torch.tensor(0.999))  # dmax

    pads = [g.idx for g in gripper.geoms if g.link.name.endswith("pad")]
    others = [g.idx for g in gripper.geoms if not g.link.name.endswith("pad")]
    assert pads
    pad_sol = solver.get_sol_params(geoms_idx=pads)
    assert torch.allclose(pad_sol[..., 0], torch.tensor(0.002)) and torch.allclose(pad_sol[..., 3], torch.tensor(0.999))
    assert not torch.allclose(solver.get_sol_params(geoms_idx=others)[..., 3], torch.tensor(0.999))


@pytest.mark.parametrize(
    "overrides, match",
    [
        # The drivers carry no spring in the MJCF, so re-anchoring one would be a silent no-op servo.
        (dict(default_dofs_spring_ref={"left_driver_joint": 1.0}), "spring"),
        (dict(default_geoms_sol_params={"no_such_link": (0.002, 1.0, 0.95, 0.999, 0.001, 0.5, 2.0)}), "no_such_link"),
    ],
)
def test_a_misaimed_compensation_fails_at_build(overrides, match):
    with pytest.raises(ValueError, match=match):
        _gripper_env(**overrides)


def test_sol_params_must_be_a_seven_tuple():
    with pytest.raises(ValueError):
        Robotiq2f85(default_equalities_sol_params=(0.002, 1.0))


@pytest.mark.slow
def test_the_gripper_pinches_a_box_without_chatter_and_holds_it_against_a_lift_load():
    """The SpaceMouse-teleop validation scene: before the tuning the pads penetrated a 5 cm box and the
    grip limit-cycled (~0.15 rad driver swing) without ever holding it.

    ``constraint_timeconst=0.002`` stiffens the box side of each pad contact as well, since Genesis mixes a
    pair's sol params 50/50. The gripper hangs fingers-down; the arm's lift is emulated by pulling the box
    up at twice its weight.
    """
    env = build_env(
        EdenConfig(
            env_options=EnvOptions(num_envs=1, **_STIFF_STEP, constraint_timeconst=0.002, enable_multi_contact=True),
            scene_options=SceneOptions(
                plane=PlaneOptions(),
                box=BoxOptions(
                    size=(0.05, 0.05, 0.05),
                    default_root_pos=(0.0, 0.0, 0.025),
                    material=RigidMaterialOptions(friction=2.0),
                ),
                # Fingers down (180 deg about x); at base z = 0.18 the closed pads centre on the box.
                gripper=Robotiq2f85(
                    default_root_pos=(0.0, 0.0, 0.18),
                    default_root_quat=(0.0, 1.0, 0.0, 0.0),
                    collision_link_patterns=[".*pad"],
                ),
            ),
        )
    )
    gripper, box = env.entities["gripper"], env.entities["box"]
    drivers = _dofs(gripper, _DRIVERS)
    rest = torch.tensor([0.0, 0.0, 0.025], device=env.device)

    def driver_qpos():
        return gripper.get_dofs_pos(dofs_idx_local=drivers)[0]

    gripper.control_dofs_pos(torch.tensor([0.8, 0.8], device=env.device), dofs_idx_local=drivers)
    for _ in range(250):  # 1.0 s
        env.scene.step()
    stall = driver_qpos()
    # Stopped at the box surface: short of the 0.8 target (crush-through), past free-hang contact.
    assert bool((stall > 0.15).all() and (stall < 0.7).all()), f"drivers did not stall on the box: {stall}"

    trace = []
    for _ in range(125):
        env.scene.step()
        trace.append(driver_qpos())
    chatter = (torch.stack(trace).amax(dim=0) - torch.stack(trace).amin(dim=0)).max()
    assert chatter < 0.02, f"grip is oscillating: driver peak-to-peak {chatter:.4f} rad"
    assert torch.linalg.vector_norm(box.get_pos()[0] - rest) < 0.01, "the box was ejected or crushed"

    # Genesis clears external forces every step, so the 1 g net upward load is re-applied each step.
    box_links = [link.idx for link in box.links]
    pull = torch.tensor([[0.0, 0.0, 2.0 * 9.81 * float(box.get_mass())]], device=env.device)
    for _ in range(250):
        env.rigid_solver.apply_links_external_wrench(force=pull, links_idx=box_links, ref=gs.link_ref_frame.link_COM)
        env.scene.step()
    assert torch.linalg.vector_norm(box.get_pos()[0] - rest) < 0.02, "the box was pulled out of the grip"
