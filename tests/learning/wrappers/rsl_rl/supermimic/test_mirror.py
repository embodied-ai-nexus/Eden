"""The left-right mirror of world-model batches: its name rules, and its invariants on the G1."""

import copy

import pytest
import torch

from eden.learning.wrappers.rsl_rl.supermimic.learners import WorldModelLearner
from eden.learning.wrappers.rsl_rl.supermimic.mirror import (
    context_permutation,
    joint_signs,
    mirror_name,
    mirror_permutation,
)
from eden.learning.wrappers.rsl_rl.supermimic.options import WorldModelOptions


def test_a_name_swaps_its_side_and_a_name_without_one_is_kept():
    assert mirror_name("left_hip_roll_joint") == "right_hip_roll_joint"
    assert mirror_name("right_ankle_roll_link") == "left_ankle_roll_link"
    assert mirror_name("waist_yaw_joint") == "waist_yaw_joint"
    assert mirror_name("upright_joint") == "upright_joint"  # "right" inside another word is no side


def test_roll_and_yaw_joints_change_sign_and_the_waists_too():
    names = ["left_hip_pitch_joint", "left_hip_roll_joint", "left_knee_joint", "waist_yaw_joint", "waist_roll_joint"]
    assert joint_signs(names) == [1.0, -1.0, 1.0, -1.0, -1.0]


def test_each_joint_maps_to_the_joint_of_the_other_side():
    names = ["left_knee_joint", "waist_pitch_joint", "right_knee_joint"]
    assert mirror_permutation(names, "joint") == [2, 1, 0]


def test_a_joint_without_a_partner_is_named():
    with pytest.raises(ValueError, match=r"\['left_knee_joint'\] have no partner \(\['right_knee_joint'\]\) among the"):
        mirror_permutation(["left_knee_joint", "waist_pitch_joint"], "joint")


def test_a_context_column_of_an_unknown_kind_is_named():
    with pytest.raises(ValueError, match=r"no rule for simulation context columns of kind \['payload_mass'\]"):
        context_permutation([("friction", "left_foot"), ("payload_mass", "torso")], [])


def test_context_columns_take_the_other_side_and_the_lateral_offset_its_negative():
    columns = [
        ("friction", "left_foot"),
        ("friction", "right_foot"),
        ("com_x", "torso"),
        ("com_y", "torso"),
        ("com_z", "torso"),
        ("encoder_bias", "left_hip_roll_joint"),
        ("encoder_bias", "right_hip_roll_joint"),
    ]
    perm, sign = context_permutation(columns, ["left_hip_roll_joint", "right_hip_roll_joint"])
    assert perm == [1, 0, 2, 3, 4, 6, 5]
    assert sign == [1.0, 1.0, 1.0, -1.0, 1.0, -1.0, -1.0]


@pytest.fixture(scope="module")
def mirror(runner):
    return runner.world_model_learner.mirror


def _states(runner, count: int = 64) -> torch.Tensor:
    """Random full states within the joint limits."""
    torch.manual_seed(0)
    rep = runner.representation
    lo, hi = rep.soft_joint_pos_limits.unbind(-1)
    joints = rep.num_joints
    return torch.cat(
        [
            torch.randn(count, 3),
            torch.nn.functional.normalize(torch.randn(count, 4), dim=-1),
            lo + torch.rand(count, joints) * (hi - lo),
            torch.randn(count, 6 + joints),
        ],
        dim=-1,
    )


requires_genesis = pytest.mark.requires_genesis


@requires_genesis
def test_mirroring_twice_is_the_identity(runner, mirror):
    state = _states(runner)
    torch.testing.assert_close(mirror.state(mirror.state(state)), state, atol=0, rtol=0)
    assert (mirror.state(state) - state).abs().max() > 0.1
    action = torch.randn(8, runner.env.num_actions)
    torch.testing.assert_close(mirror.action(mirror.action(action)), action, atol=1e-5, rtol=0)
    context = torch.rand(8, runner.sim_context.dim) * 2 - 1
    torch.testing.assert_close(mirror.sim_context(mirror.sim_context(context)), context, atol=1e-6, rtol=0)


@requires_genesis
def test_the_links_of_a_mirrored_state_are_the_mirror_image_of_their_partner_links(runner, mirror):
    """Forward kinematics commutes with the mirror, with left and right links swapped."""
    fk, joints = runner.motion.fk, runner.representation.num_joints
    state = _states(runner)
    link_perm = torch.tensor(mirror_permutation(fk.link_names, "tracked link"))
    assert link_perm.tolist() != list(range(len(fk.link_names)))

    def links(full_state):
        return fk(*full_state.split((3, 4, joints, 3, 3, joints), dim=-1))

    pos, quat, lin, ang = links(mirror.state(state))
    e_pos, e_quat, e_lin, e_ang = mirror.bodies(links(state), link_perm)
    torch.testing.assert_close(pos, e_pos, atol=1e-4, rtol=0)
    torch.testing.assert_close(lin, e_lin, atol=1e-3, rtol=0)
    torch.testing.assert_close(ang, e_ang, atol=1e-3, rtol=0)
    # q and -q are one rotation.
    torch.testing.assert_close((quat * e_quat).sum(-1).abs(), torch.ones(quat.shape[:-1]), atol=1e-5, rtol=0)


@requires_genesis
def test_the_model_state_of_a_mirrored_state_is_the_mirror_image_of_the_model_state(runner, mirror):
    """Root height, and gravity and twist in the root frame: y of a vector and x, z of the angular velocity flip."""
    rep = runner.representation
    state = _states(runner)
    z, g_b, v_b, w_b, q, qd = rep.split_rep(rep.represent(state))
    m_z, m_g_b, m_v_b, m_w_b, m_q, m_qd = rep.split_rep(rep.represent(mirror.state(state)))
    flip_y, flip_xz = torch.tensor([1.0, -1.0, 1.0]), torch.tensor([-1.0, 1.0, -1.0])
    torch.testing.assert_close(m_z, z)
    torch.testing.assert_close(m_g_b, g_b * flip_y, atol=1e-5, rtol=0)
    torch.testing.assert_close(m_v_b, v_b * flip_y, atol=1e-5, rtol=0)
    torch.testing.assert_close(m_w_b, w_b * flip_xz, atol=1e-5, rtol=0)
    torch.testing.assert_close(m_q, mirror.joints(q))
    torch.testing.assert_close(m_qd, mirror.joints(qd))


@requires_genesis
def test_a_mirrored_action_commands_the_mirror_image_of_the_joint_target(runner, mirror):
    dynamics = runner.dynamics
    action = torch.randn(8, runner.env.num_actions)
    torch.testing.assert_close(
        dynamics.target(mirror.action(action)), mirror.joints(dynamics.target(action)), atol=1e-5, rtol=0
    )


@requires_genesis
def test_a_mirrored_context_swaps_the_feet_and_negates_the_lateral_offset_and_the_bias_of_flipped_joints(
    runner, mirror
):
    context = runner.sim_context
    columns = context.columns
    ctx = torch.rand(4, context.dim) * 2 - 1
    mirrored = mirror.sim_context(ctx)
    left, right = (
        columns.index(("friction", "left_ankle_roll_link")),
        columns.index(("friction", "right_ankle_roll_link")),
    )
    torch.testing.assert_close(mirrored[:, left], ctx[:, right])
    torch.testing.assert_close(mirrored[:, right], ctx[:, left])
    for axis, sign in (("com_x", 1.0), ("com_y", -1.0), ("com_z", 1.0)):
        column = columns.index((axis, "torso_link"))
        torch.testing.assert_close(mirrored[:, column], sign * ctx[:, column])
    # The encoder bias in radians mirrors like the joints.
    torch.testing.assert_close(
        context.encoder_bias(mirrored), mirror.joints(context.encoder_bias(ctx)), atol=1e-7, rtol=0
    )


@requires_genesis
def test_a_mapping_that_the_kinematics_contradict_is_refused_with_the_link(runner, mirror):
    limits = runner.representation.soft_joint_pos_limits
    mirror.check_kinematics(runner.motion.fk, limits)
    wrong = copy.copy(mirror)
    wrong.joint_sign = mirror.joint_sign.clone()
    wrong.joint_sign[list(runner.motion.fk.joint_names).index("left_hip_roll_joint")] = 1.0  # as a pitch joint
    with pytest.raises(ValueError, match=r"link '\w+' of a mirrored state lies up to \d\.\d+ m from the mirror image"):
        wrong.check_kinematics(runner.motion.fk, limits)


@requires_genesis
def test_the_world_model_fits_the_mirror_image_of_the_share_of_each_batch(runner, mirror):
    """A fully mirrored batch scores as the batch mirrored by hand, and a half mirrored batch keeps its second half."""
    runner.learn(num_learning_iterations=2)
    options = WorldModelOptions(hidden_dims=(32,), batch_size=8, num_batches=1, horizon=4)
    windows = runner.replay_buffer.sample(options.horizon, options.batch_size)
    by_hand = windows.clone()
    by_hand["state"] = mirror.state(windows["state"])
    by_hand["action"] = mirror.action(windows["action"])
    by_hand["sim_ctx"] = mirror.sim_context(windows["sim_ctx"])

    def loss(batch, share):
        learner = WorldModelLearner(
            copy.deepcopy(runner.dynamics),
            runner.representation,
            options.model_copy(update={"mirror_share": share}),
            "cpu",
            mirror if share else None,
        )
        return learner.update([batch])["world_model/state"]

    plain, mirrored = loss(windows, 0.0), loss(by_hand, 0.0)
    assert abs(plain - mirrored) > 1e-6  # the dynamics are not symmetric, so the two batches score differently
    torch.testing.assert_close(loss(windows, 1.0), mirrored, rtol=1e-5, atol=0)
    half = torch.cat([by_hand[:4], windows[4:]])
    torch.testing.assert_close(loss(windows, 0.5), loss(half, 0.0), rtol=1e-5, atol=0)


@requires_genesis
def test_the_simulator_steps_a_mirrored_state_under_the_mirrored_action_to_the_mirror_image(supermimic_library):
    """The physical symmetry the augmentation relies on, with both feet on the ground: env 1 starts on the mirror
    image of env 0's pose, takes the mirrored action, and lands on the mirror image of env 0's next state.
    """
    from eden.envs.wrappers.rsl_rl_env import RslRlVecEnvWrapper, make_rsl_rl_runner
    from tests.learning.wrappers.rsl_rl.supermimic.conftest import small_learner
    from tests.support.supermimic import supermimic_config

    cfg = small_learner(supermimic_config(supermimic_library, rsi_noise=False))
    cfg.env_options.num_envs = 2
    # One context for both envs: the mirror of a context is tested apart, and here only the state differs.
    events = cfg.event_options
    events.foot_friction.friction_range = (0.8, 0.8)
    events.base_com.com_shift_x_range = events.base_com.com_shift_y_range = (0.0, 0.0)
    events.base_com.com_shift_z_range = events.encoder_bias.dofs_pos_range = (0.0, 0.0)
    env = RslRlVecEnvWrapper.from_config(cfg)
    mirror = make_rsl_rl_runner(env).world_model_learner.mirror
    joints = mirror.num_joints
    obs, _ = env.reset()
    assert env.unwrapped.motion_manager.env_origins is None  # the state below is written without an origin

    torch.manual_seed(0)
    start = obs["full_state"][0].clone()
    start[7 : 7 + joints] += 0.15 * torch.randn(joints)
    start[3:7] = torch.nn.functional.normalize(start[3:7] + 0.05 * torch.randn(4), dim=-1)
    start[7 + joints :] = 0.0
    poses = torch.stack([start, mirror.state(start)])[:, : 7 + joints]
    env.unwrapped.entities["robot"].set_qpos(poses, zero_velocity=True)
    action = 0.5 * torch.randn(joints)
    obs, _, dones, _ = env.step(torch.stack([action, mirror.action(action)]))
    assert not dones.any()

    state, image = obs["full_state"][1], mirror.state(obs["full_state"][0])
    moved = (obs["full_state"][0] - start).abs()
    assert moved[7 : 7 + joints].max() > 0.05 and moved[7 + joints :].max() > 1.0
    torch.testing.assert_close(state[: 7 + joints], image[: 7 + joints], atol=5e-4, rtol=0)
    torch.testing.assert_close(state[7 + joints :], image[7 + joints :], atol=5e-2, rtol=0)
