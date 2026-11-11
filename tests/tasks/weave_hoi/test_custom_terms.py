"""The Weave task terms: the geometry helpers against closed forms, and each term on a built two-object task.

Term behaviours are driven through real state — the reference contact labels in the motion buffer, the object
moved clear of the hands — rather than stand-ins, so each assertion is what the policy would be scored on.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from eden.tasks.weave_hoi.custom_terms import (
    SDF_GRID_BOUND,
    load_surface_points,
    nearest_surface_points,
    sample_sdf_trilinear,
)
from eden.utils.geom import transform_by_quat
from tests.tasks.weave_hoi.conftest import TEST_OBJECTS, build_weave_env

_HANDS = ("left_hand", "right_hand")


# ── geometry helpers (no build) ────────────────────────────────────────────────────────────────────


def _sphere_grid(res: int, radius: float) -> torch.Tensor:
    axis = torch.linspace(-SDF_GRID_BOUND, SDF_GRID_BOUND, res)
    x, y, z = torch.meshgrid(axis, axis, axis, indexing="ij")
    return torch.sqrt(x**2 + y**2 + z**2) - radius


def test_the_sdf_is_sampled_trilinearly_on_x_y_z_grid_axes():
    """A plane SDF is linear, so trilinear interpolation is exact anywhere inside the grid."""
    axis = torch.linspace(-SDF_GRID_BOUND, SDF_GRID_BOUND, 9)
    x, y, z = torch.meshgrid(axis, axis, axis, indexing="ij")
    grids = torch.stack([x - 0.3 * y + 2.0 * z, -x])  # two objects: the second is picked per env
    points = torch.tensor([[[0.1, 0.2, -0.33], [1.0, -1.1, 0.7]]] * 2)
    got = sample_sdf_trilinear(grids, torch.tensor([0, 1]), points, SDF_GRID_BOUND)
    p = points[0]
    torch.testing.assert_close(got[0], p[:, 0] - 0.3 * p[:, 1] + 2.0 * p[:, 2])
    torch.testing.assert_close(got[1], -p[:, 0])


def test_a_query_outside_the_grid_reads_the_boundary():
    grid = _sphere_grid(33, 0.5)[None]
    inside = sample_sdf_trilinear(grid, torch.tensor([0]), torch.tensor([[[SDF_GRID_BOUND, 0.0, 0.0]]]), SDF_GRID_BOUND)
    outside = sample_sdf_trilinear(grid, torch.tensor([0]), torch.tensor([[[9.0, 0.0, 0.0]]]), SDF_GRID_BOUND)
    torch.testing.assert_close(outside, inside)
    torch.testing.assert_close(inside, torch.tensor([[SDF_GRID_BOUND - 0.5]]), atol=1e-5, rtol=0)


def test_the_sphere_sdf_reads_its_distance_to_within_a_voxel():
    grid = _sphere_grid(65, 0.5)[None]
    points = (torch.rand(1, 64, 3, generator=torch.Generator().manual_seed(0)) * 2 - 1) * 0.8  # inside the grid
    got = sample_sdf_trilinear(grid, torch.tensor([0]), points, SDF_GRID_BOUND)
    torch.testing.assert_close(got, points.norm(dim=-1) - 0.5, atol=0.02, rtol=0)


def test_the_nearest_surface_point_is_the_brute_force_argmin():
    generator = torch.Generator().manual_seed(0)
    surface, query = torch.randn(3, 50, 3, generator=generator), torch.randn(3, 7, 3, generator=generator)
    got = nearest_surface_points(surface, query)
    brute = surface[torch.arange(3)[:, None], ((query[:, :, None] - surface[:, None]) ** 2).sum(-1).argmin(-1)]
    torch.testing.assert_close(got, brute)


# ── on the built task ──────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def env(weave_root):
    return build_weave_env(weave_root, ",".join(TEST_OBJECTS), num_envs=4)


@pytest.fixture
def fresh(env):
    env.reset()
    yield env
    env.reset()


def _set_labels(env, value: int) -> None:
    """Overwrite the current-frame reference contact label of every hand link."""
    for hand in _HANDS:
        env.motion_manager.data[hand].contact_label[:, 0] = value


def _move_object_clear(env) -> None:
    """Park the object 10 m away and step once, so the contact buffers see no hand-object contact."""
    obj = env.entities["object"]
    obj.set_pos(obj.get_pos() + torch.tensor([10.0, 0.0, 0.0], device=env.device))
    env.scene.step()


@pytest.mark.slow
@pytest.mark.requires_genesis
def test_the_light_variant_is_rescaled_to_its_own_density(env):
    """One material serves the bank (200 kg/m^3); largebox is Weave's 25 kg/m^3 box, ~0.9 kg rather than ~7 kg."""
    obj = env.entities["object"]
    mass = obj.get_links_inertial_mass().reshape(env.num_envs, -1).sum(-1)
    variant = obj.env_variant_idx
    largebox, smallbox = mass[variant == 1], mass[variant == 0]
    assert torch.allclose(largebox, torch.full_like(largebox, 0.876), atol=0.02)
    assert torch.allclose(smallbox, torch.full_like(smallbox, 2.338), atol=0.05)


@pytest.mark.slow
@pytest.mark.requires_genesis
def test_the_nearest_surface_vectors_are_as_long_as_the_tip_to_surface_distance(fresh, weave_root):
    """Rotation-free check: however the offsets are framed, their length is the distance to the nearest sample."""
    env = fresh
    term = env.observation_manager.get_term("object_nearest_point")
    got = term.compute().reshape(env.num_envs, -1, 3).norm(dim=-1)
    obj = env.entities["object"]
    surface = load_surface_points(list(TEST_OBJECTS), weave_root, env.device)[obj.env_variant_idx]
    surface_w = (
        transform_by_quat(surface, obj.get_quat()[:, None].expand(-1, surface.shape[1], -1)) + obj.get_pos()[:, None]
    )
    tips = torch.cat([env.entities[n].get_links_pos(ls_idx_local=idx) for n, idx in term._links_idx.items()], dim=1)
    expected = torch.cdist(tips, surface_w).min(-1).values
    torch.testing.assert_close(got, expected, atol=1e-4, rtol=0)


@pytest.mark.slow
@pytest.mark.requires_genesis
def test_the_bps_descriptor_ignores_where_the_object_is_but_not_how_it_is_turned(fresh):
    env = fresh
    term = env.observation_manager.get_term("object_bps_sdf")
    before = term.compute()
    assert before.shape == (env.num_envs, 128)
    obj = env.entities["object"]
    obj.set_pos(obj.get_pos() + torch.tensor([0.3, -0.2, 0.1], device=env.device))
    torch.testing.assert_close(term.compute(), before)
    yaw = torch.tensor([[np.cos(0.4), 0.0, 0.0, np.sin(0.4)]], dtype=torch.float32, device=env.device)
    from eden.utils.geom import quat_mul

    obj.set_quat(quat_mul(yaw.expand(env.num_envs, -1), obj.get_quat()), relative=False)
    assert not torch.allclose(term.compute(), before, atol=1e-3)


@pytest.mark.slow
@pytest.mark.requires_genesis
@pytest.mark.parametrize("label, expected", [(-1, 1.0), (1, 0.0), (0, 0.0)])
def test_the_contact_reward_scores_labelled_links_and_skips_the_rest(fresh, label, expected):
    """Hands clear of the object: every "don't touch" is met, every "touch" missed, and no label scores nothing."""
    env = fresh
    _move_object_clear(env)
    _set_labels(env, label)
    reward = env.reward_manager.get_term("contact").compute()
    torch.testing.assert_close(reward, torch.full_like(reward, expected))


@pytest.mark.slow
@pytest.mark.requires_genesis
def test_the_opposition_reward_is_gated_on_the_reference_and_zero_with_all_digits_on_one_side(fresh):
    env = fresh
    term = env.reward_manager.get_term("hand_opposition")
    _set_labels(env, 0)
    torch.testing.assert_close(term.compute(), torch.zeros(env.num_envs, device=env.device))
    _set_labels(env, 1)
    near = term.compute()
    assert ((near >= 0) & (near <= 1)).all()
    # 10 m away, every digit looks at the object from the same side: nothing opposes anything.
    obj = env.entities["object"]
    obj.set_pos(obj.get_pos() + torch.tensor([10.0, 0.0, 0.0], device=env.device))
    far = term.compute()
    assert (far < 0.01).all(), far


@pytest.mark.slow
@pytest.mark.requires_genesis
def test_lost_contact_counts_consecutive_lost_steps_and_clears_on_any_other(fresh):
    env = fresh
    term = env.termination_manager.get_term("lost_contact")
    term.reset()
    _move_object_clear(env)
    _set_labels(env, 1)
    fired = [bool(term.compute().any()) for _ in range(term.max_lost_steps)]
    assert fired == [False] * (term.max_lost_steps - 1) + [True]
    _set_labels(env, -1)  # contact no longer expected: the counter clears
    assert not term.compute().any()
    assert (term._lost == 0).all()
    _set_labels(env, 1)
    assert not term.compute().any()  # one lost step again, far from the limit
    term.reset()
    assert (term._lost == 0).all()


@pytest.mark.slow
@pytest.mark.requires_genesis
def test_feet_air_time_pays_a_landing_after_a_long_enough_swing_only(fresh):
    env = fresh
    env.scene.step()
    term = env.reward_manager.get_term("feet_air_time")
    robot = env.entities["robot"]
    loaded = robot.get_links_net_contact_force()[:, term._feet_idx].norm(dim=-1) > term.contact_force_threshold
    if not loaded.any():
        pytest.skip("no foot is on the ground at this reference frame")
    term.reset()
    term._air_time.fill_(0.5)  # every foot has been airborne 0.5 s
    torch.testing.assert_close(term.compute(), loaded.float().sum(-1))
    assert (term._air_time[loaded] == 0).all()  # a landing restarts the swing clock
    torch.testing.assert_close(term.compute(), torch.zeros(env.num_envs, device=env.device))  # still down: no pay
    term.reset()
    term._air_time.fill_(0.2)  # below Weave's 0.3 s
    torch.testing.assert_close(term.compute(), torch.zeros(env.num_envs, device=env.device))


@pytest.mark.slow
@pytest.mark.requires_genesis
def test_feet_slide_is_the_planar_foot_speed_of_loaded_feet(fresh):
    env = fresh
    env.scene.step()
    term = env.reward_manager.get_term("feet_slide")
    speed = env.entities["robot"].get_links_vel(ls_idx_local=term._feet_idx)[..., :2].norm(dim=-1)
    term.reset()
    term._history.fill_(100.0)  # loaded throughout the history window
    torch.testing.assert_close(term.compute(), speed.sum(-1), atol=1e-6, rtol=0)
    term.reset()
    threshold = term.force_threshold
    term.force_threshold = float("inf")  # nothing counts as loaded
    try:
        torch.testing.assert_close(term.compute(), torch.zeros(env.num_envs, device=env.device))
    finally:
        term.force_threshold = threshold
        term.reset()


def test_the_surface_files_have_weaves_512_samples(weave_root):
    surface = load_surface_points(list(TEST_OBJECTS), weave_root, "cpu")
    assert surface.shape == (2, 512, 3)
    assert os.path.isfile(os.path.join(weave_root, "objects", "geometry", "bps_128.npy"))
