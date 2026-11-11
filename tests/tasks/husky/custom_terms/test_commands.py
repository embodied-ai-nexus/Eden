"""``SkateCommand``: the transition-blend geometry, and its reset/post_reset split on a real HUSKY env."""

from __future__ import annotations

import math

import pytest
import torch

from eden.tasks.husky.custom_terms.commands import _bezier, _slerp


def test_bezier_hits_its_endpoints_and_arcs_up_by_the_offset():
    start, end = torch.zeros(2, 3, 3), torch.ones(2, 3, 3)
    torch.testing.assert_close(_bezier(start, end, torch.zeros(2), offset=0.2), start)
    torch.testing.assert_close(_bezier(start, end, torch.ones(2), offset=0.2), end)
    mid = _bezier(start, end, torch.full((2,), 0.5), offset=0.2)
    # Halfway: the mean of the endpoints, plus half the arc offset on z.
    torch.testing.assert_close(mid[..., :2], torch.full((2, 3, 2), 0.5))
    torch.testing.assert_close(mid[..., 2], torch.full((2, 3), 0.6))


def test_slerp_hits_its_endpoints_and_takes_the_short_path():
    yaw = math.pi / 2
    q0 = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])
    q1 = torch.tensor([[[math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]]])
    half = torch.tensor([[[math.cos(yaw / 4), 0.0, 0.0, math.sin(yaw / 4)]]])
    torch.testing.assert_close(_slerp(q0, q1, torch.zeros(1)), q0)
    torch.testing.assert_close(_slerp(q0, q1, torch.ones(1)), q1)
    torch.testing.assert_close(_slerp(q0, q1, torch.full((1,), 0.5)), half, atol=1e-5, rtol=1e-5)
    # The antipodal representation of the same rotation must blend the same way.
    torch.testing.assert_close(_slerp(q0, -q1, torch.full((1,), 0.5)).abs(), half.abs(), atol=1e-5, rtol=1e-5)


@pytest.fixture
def skate(husky_env):
    """The live ``SkateCommand`` after a few steps, with the skateboard of envs 1 and 3 moved under the robot.

    Tracked poses are in the board frame, so the move changes what a refresh would compute for those envs only.
    """
    env = husky_env
    env.reset()
    for _ in range(3):
        env.step(torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device))
    board = env.entities["skateboard"]
    moved = torch.tensor([1, 3], device=env.device)
    pos = board.get_pos()[moved]
    pos[:, 0] += 0.3
    board.set_pos(pos, envs_idx=moved)
    yield env.command_manager.get_term("skate"), moved
    env.reset()


@pytest.mark.requires_genesis
def test_reset_zeroes_the_contact_state_and_leaves_the_pose_caches_to_post_reset(skate):
    """``reset()`` runs mid-sequence, while other managers may still move the scene; only ``post_reset`` reads it."""
    term, moved = skate
    live = torch.tensor([0, 2], device=moved.device)
    before = term.tracked_pos_b.clone()

    term.reset(moved)
    assert not term.foot_ground_force[moved].any()  # unknowable until the next physics step
    assert not term.illegal_contact[moved].any()
    assert torch.equal(term.tracked_pos_b, before)

    term.post_reset(moved)
    assert not torch.allclose(term.tracked_pos_b[moved], before[moved])
    assert torch.equal(term.tracked_pos_b[live], before[live])  # the hook is index-scoped
    assert not term.in_transition[moved].any()


@pytest.mark.requires_genesis
def test_post_reset_under_inference_mode_leaves_the_caches_writable(skate):
    """The manager fan-out is ``@torch.inference_mode()``: every cache write must be in place into a build-time
    buffer, or the next write outside inference mode raises.
    """
    term, moved = skate
    with torch.inference_mode():
        term.post_reset(moved)
    term.post_reset(torch.tensor([0, 2], device=moved.device))  # raises on an inference-tensor cache
    for cache in (term.tracked_pos_b, term.tracked_quat_b, term.feet_marker_dis, term.left_foot_site_z):
        assert not cache.is_inference()


@pytest.fixture(scope="module")
def held_env():
    """Two envs at the HUSKY control rate holding a ``ConstantSkateCommand``; env 0 is told to stand still."""
    from eden.options import CommandManagerOptions, EnvOptions, ObservationGroupOptions, ObservationManagerOptions
    from eden.options.entities import BoxOptions, SceneOptions
    from eden.tasks.husky.custom_terms.commands import ConstantSkateCommand
    from eden.tasks.husky.custom_terms.observations import skate_phase
    from eden.terms import ObsTerm
    from eden.utils.configs import EdenRLConfig
    from tests.support.envs import build_env

    env = build_env(
        EdenRLConfig(
            env_options=EnvOptions(num_envs=2, sim_dt=0.005, decimation=4, episode_length_s=1e9),
            scene_options=SceneOptions(box=BoxOptions(size=(0.1, 0.1, 0.1), default_root_pos=(0.0, 0.0, 0.05))),
            command_options=CommandManagerOptions(skate=ConstantSkateCommand.configure()),
            observation_options=ObservationManagerOptions(
                policy=ObservationGroupOptions(phase=ObsTerm.configure(func=skate_phase))
            ),
        )
    )
    env.command_manager.get_term("skate").set_command(torch.tensor([[0.05, 0.0], [1.0, 0.3]]))
    return env


def _phase(steps: int) -> torch.Tensor:
    return (torch.tensor(steps).float() * (0.02 / 6.0)) % 1.0


@pytest.mark.requires_genesis
def test_held_command_survives_steps_and_resets(held_env):
    term = held_env.command_manager.get_term("skate")
    held_env.reset()
    for _ in range(3):
        held_env.step()
    torch.testing.assert_close(term.command, torch.tensor([[0.05, 0.0], [1.0, 0.3]]))


@pytest.mark.requires_genesis
def test_phase_clock_ticks_once_per_step_and_reads_zero_after_reset(held_env):
    term = held_env.command_manager.get_term("skate")
    obs, _ = held_env.reset()
    assert not obs["policy"].any()
    for _ in range(3):
        obs, *_ = held_env.step()
    # The observation carries the tick of the step that produced it. The still env pinned on its first
    # tick (step 1 is a half-cycle boundary), so it never leaves the push stance.
    assert torch.equal(obs["policy"][:, 0], torch.stack([_phase(0), _phase(3)]))
    assert torch.equal(term.phase, obs["policy"][:, 0])


@pytest.mark.requires_genesis
def test_a_still_env_pins_back_to_zero_at_the_half_cycle_boundary(held_env):
    term = held_env.command_manager.get_term("skate")
    held_env.reset()
    half_cycle = int(6.0 / 2.0 / held_env.dt)
    term.phase_length_buf[:] = half_cycle  # the next tick lands on the boundary
    held_env.step()
    assert term.phase_length_buf.tolist() == [0, half_cycle + 1]  # env 0 is still, env 1 rides on
    assert torch.equal(term.phase, torch.stack([_phase(0), _phase(half_cycle + 1)]))


@pytest.mark.requires_genesis
def test_rewind_restarts_the_clock_and_keeps_the_command(held_env):
    """A deployment rewinds at its reset, its START and every policy switch, where nothing else resets the term."""
    term = held_env.command_manager.get_term("skate")
    held_env.reset()
    for _ in range(2):
        held_env.step()
    held_env.command_manager.rewind_reference()
    assert not term.phase_length_buf.any() and not term.phase.any()
    torch.testing.assert_close(term.command, torch.tensor([[0.05, 0.0], [1.0, 0.3]]))


@pytest.mark.requires_genesis
def test_a_held_command_of_the_wrong_width_is_refused_with_its_layout(held_env):
    term = held_env.command_manager.get_term("skate")
    with pytest.raises(ValueError, match=r"\[lin_vel_x, relative_heading\], shaped \(2,\)"):
        term.set_command(torch.tensor([1.0, 0.0, 0.0]))
