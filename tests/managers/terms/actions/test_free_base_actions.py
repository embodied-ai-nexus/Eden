"""The free-body pose controller in ``eden/managers/terms/actions/free_base_actions.py``: ``FreeBasePoseController``.

It flies a free body: a zero action holds the placed pose, the
target is read in the authored frame of a body Genesis re-origins onto its CoM, an orientation at the Euler
singularity tracks, and a body with no floating base is refused at build.
"""

import math

import numpy as np
import pytest
import torch
from genesis.utils.geom import inv_quat

from eden.managers.terms.actions import FreeBasePoseController
from eden.options import BoxOptions, EntityOptions, EnvOptions, PlaneOptions, SceneOptions
from eden.options.materials import RigidMaterialOptions
from eden.utils.configs import ActionManagerOptions, EdenConfig
from eden.utils.geom import axis_angle_from_quat, quat_from_axis_angle, quat_mul
from eden.utils.misc import get_assets_dir
from tests.support.envs import build_env

pytestmark = pytest.mark.requires_genesis

# A mesh whose centre of mass is off its origin: the solver frame Genesis positions is not the authored frame.
HAND_STL = get_assets_dir("grippers", "franka_hand", "assets", "hand.stl")

REST_POS = (0.3, -0.2, 0.6)
HANGING_POS = (-0.3, 0.2, 0.6)
# Palm-down: a half turn about (1, 0, -1), at the pitch = 90 degrees singularity of intrinsic xyz Euler angles.
SINGULAR_QUAT = (0.0, math.sqrt(0.5), 0.0, -math.sqrt(0.5))
GAINS = {"position_kp": 2000.0, "position_kd": 80.0, "orientation_kp": 40.0, "orientation_kd": 2.0}


def _free_body(pos, gravity_compensation):
    return EntityOptions(
        file=HAND_STL,
        convexify=True,
        is_fixed_base=False,
        default_root_pos=pos,
        default_root_quat=SINGULAR_QUAT,
        material=RigidMaterialOptions(gravity_compensation=gravity_compensation),
    )


def _config(entity, **term_kwargs):
    return EdenConfig(
        env_options=EnvOptions(num_envs=2, sim_dt=0.005, decimation=10),
        scene_options=SceneOptions(plane=PlaneOptions(), body=entity),
        action_options=ActionManagerOptions(
            pose=FreeBasePoseController.configure(entity_name="body", **{**GAINS, **term_kwargs})
        ),
    )


@pytest.fixture(scope="module")
def env():
    """``body`` is carried by gravity compensation; ``hanging`` leaves its weight to the position loop."""
    scales = {"position_scale": 0.1, "orientation_scale": 0.3}
    return build_env(
        EdenConfig(
            env_options=EnvOptions(num_envs=2, sim_dt=0.005, decimation=10),
            scene_options=SceneOptions(
                plane=PlaneOptions(),
                body=_free_body(REST_POS, gravity_compensation=1.0),
                hanging=_free_body(HANGING_POS, gravity_compensation=0.0),
            ),
            action_options=ActionManagerOptions(
                pose=FreeBasePoseController.configure(entity_name="body", **GAINS, **scales),
                hang=FreeBasePoseController.configure(entity_name="hanging", **GAINS, **scales),
            ),
        ),
        reset=True,
    )


def _drive(env, action, n_steps=20):
    """Hold ``action`` on ``body`` (and zero on ``hanging``) for ``n_steps`` control steps."""
    command = torch.as_tensor(action, dtype=torch.float32, device=env.device).expand(env.num_envs, 6).contiguous()
    for _ in range(n_steps):
        env.step({"pose": command, "hang": torch.zeros_like(command)})


def _pose_error(env, pos, quat, name="body"):
    """Distance in metres and angle in radians from the entity's authored-frame pose to ``(pos, quat)``."""
    body = env.entities[name]
    target_pos = torch.as_tensor(pos, dtype=torch.float32, device=env.device).expand(env.num_envs, 3)
    target_quat = torch.as_tensor(quat, dtype=torch.float32, device=env.device).expand(env.num_envs, 4)
    turn = axis_angle_from_quat(quat_mul(body.get_quat(relative=True), inv_quat(target_quat)))
    return float((body.get_pos(relative=True) - target_pos).norm(dim=-1).max()), float(turn.norm(dim=-1).max())


def test_a_zero_action_holds_the_pose_the_body_was_placed_in(env):
    env.reset()
    _drive(env, [0.0] * 6, n_steps=40)
    moved, turned = _pose_error(env, REST_POS, SINGULAR_QUAT)
    assert moved < 5.0e-4
    assert turned < math.radians(1.0)


def test_the_body_reaches_the_pose_the_action_asks_for(env):
    """Translation and rotation are scaled offsets on the default pose, about world axes."""
    env.reset()
    _drive(env, [1.0, -0.5, 0.25, 0.0, 0.0, 0.0])
    moved, turned = _pose_error(env, tuple(np.array(REST_POS) + np.array([0.1, -0.05, 0.025])), SINGULAR_QUAT)
    assert moved < 2.0e-3
    assert turned < math.radians(1.0)  # translating does not tip it


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_a_rotation_tracks_through_the_euler_singularity(env, axis):
    env.reset()
    action = [0.0] * 6
    action[3 + axis] = 1.0  # 0.3 rad about one world axis
    _drive(env, action)
    turn = torch.zeros(3, device=env.device)
    turn[axis] = 0.3
    target = quat_mul(
        quat_from_axis_angle(turn[None]), torch.as_tensor(SINGULAR_QUAT, dtype=torch.float32, device=env.device)[None]
    )[0].tolist()
    moved, turned = _pose_error(env, REST_POS, target)
    assert turned < math.radians(2.0)
    assert moved < 5.0e-3  # it turns about the authored origin, not about the centre of mass


def test_without_gravity_compensation_the_body_hangs_below_its_target(env):
    """The controller alone holds it up, so its weight shows as steady-state droop of about m g / kp."""
    env.reset()
    _drive(env, [0.0] * 6, n_steps=40)
    hanging = env.entities["hanging"]
    droop = float((HANGING_POS[2] - hanging.get_pos(relative=True)[:, 2]).max())
    assert droop == pytest.approx(float(hanging.get_mass()) * 9.81 / GAINS["position_kp"], rel=0.5)
    assert droop > 1.0e-4


def test_the_gains_land_on_the_free_joint(env):
    """Nothing else writes them: Genesis leaves a free joint at zero gain with an unbounded force range."""
    body = env.entities["body"]
    base = body.find_free_base_dofs_idx_local()
    kp, kd = body.get_dofs_kp(dofs_idx_local=base)[0], body.get_dofs_kd(dofs_idx_local=base)[0]
    np.testing.assert_allclose(kp.cpu().numpy(), [GAINS["position_kp"]] * 3 + [GAINS["orientation_kp"]] * 3)
    np.testing.assert_allclose(kd.cpu().numpy(), [GAINS["position_kd"]] * 3 + [GAINS["orientation_kd"]] * 3)
    lower, upper = body.get_dofs_force_range(dofs_idx_local=base)
    assert torch.isfinite(lower).all() and torch.isfinite(upper).all()


def test_a_fixed_base_is_refused_at_build():
    fixed_box = BoxOptions(size=(0.1, 0.1, 0.1), default_root_pos=(0.0, 0.0, 0.5), is_fixed_base=True)
    with pytest.raises(Exception, match="free joint|floating base"):
        build_env(_config(fixed_box))


def test_dofs_name_is_refused_at_build():
    """The term drives the free joint; naming joints would silently mean something else."""
    with pytest.raises(ValueError, match="takes no `dofs_name`"):
        build_env(_config(_free_body(REST_POS, gravity_compensation=1.0), dofs_name=["x"]))
