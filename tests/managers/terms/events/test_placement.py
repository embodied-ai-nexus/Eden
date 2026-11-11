"""The support-surface samplers in ``eden/managers/terms/events/placement.py``: pure tensors.

A row with no valid support point falls back to a uniform draw over the valid points and is flagged in the
returned mask, instead of ``torch.multinomial`` raising.
"""

import pytest
import shapely.geometry
import torch

from eden.managers.terms.events.placement import (
    Gaussian2D,
    constant_xy_sampler,
    gaussian_sampler,
    range_xy_sampler,
    uniform_sampler,
)
from eden.utils.mesh import ExtendedSupportData

NUM_POINTS = 4  # sample points at x = -0.1, -0.033, 0.033, 0.1 on a flat +z plane through the origin


def _support(valid_mask):
    num_envs = valid_mask.shape[0]
    points = torch.zeros(NUM_POINTS, 2)
    points[:, 0] = torch.linspace(-0.1, 0.1, NUM_POINTS)
    normal = torch.zeros(num_envs, 3)
    normal[:, 2] = 1.0
    return ExtendedSupportData(
        polygon=shapely.geometry.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]),
        transform=torch.eye(4).expand(num_envs, 4, 4).contiguous(),
        normal=normal,
        valid_mask=valid_mask,
        sample_points=points,
        num_sample_points=NUM_POINTS,
    )


def _zero_extent_asset(num_envs):
    return torch.zeros(num_envs, 2, 3)


_SAMPLERS = {
    "uniform": lambda support, aabb: uniform_sampler([support], aabb),
    "gaussian": lambda support, aabb: gaussian_sampler(
        [support], aabb, gaussian_weights=Gaussian2D(mu_x=0.0, mu_y=0.0, var_x=1.0, var_y=1.0).get_probs
    ),
}


@pytest.mark.parametrize("sampler", list(_SAMPLERS))
@pytest.mark.parametrize("num_envs", [1, 7, 64])
def test_rows_with_no_valid_support_are_flagged_not_raised(sampler, num_envs):
    """Env 0 keeps the support alive through the ``.any()`` filter; every other row is all-invalid."""
    valid_mask = torch.zeros(num_envs, dtype=torch.bool)
    valid_mask[0] = True

    pos, direction, out_mask = _SAMPLERS[sampler](_support(valid_mask), _zero_extent_asset(num_envs))

    assert pos.shape == direction.shape == (num_envs, 3)
    assert out_mask.tolist() == valid_mask.tolist()


@pytest.mark.parametrize("num_envs", [1, 4])
def test_a_valid_row_whose_gaussian_weights_underflow_falls_back_to_uniform(num_envs):
    """A guard keyed on ``~valid_mask`` alone misses an all-zero probability row on a *valid* env."""
    valid_mask = torch.ones(num_envs, dtype=torch.bool)
    narrow = Gaussian2D(mu_x=0.0, mu_y=0.0, var_x=1e-6, var_y=1e-6).get_probs

    pos, _, out_mask = gaussian_sampler([_support(valid_mask)], _zero_extent_asset(num_envs), gaussian_weights=narrow)

    assert pos.shape == (num_envs, 3)
    assert out_mask.all()


@pytest.mark.parametrize("num_envs", [1, 7])
def test_the_underflow_fallback_never_lands_in_a_hole(num_envs):
    """The fallback draws over the valid points only; an all-ones fallback could pick a point in a hole."""
    holes = torch.zeros(num_envs, 1, 2, 3)
    holes[:, 0, 0] = torch.tensor([-0.2, -0.5, -1.0])  # the hole covers x in [-0.2, 0.0)
    holes[:, 0, 1] = torch.tensor([0.0, 0.5, 1.0])

    pos, _, out_mask = gaussian_sampler(
        [_support(torch.ones(num_envs, dtype=torch.bool))],
        _zero_extent_asset(num_envs),
        holes=holes,
        gaussian_weights=lambda pts: torch.zeros(pts.shape[:-1], dtype=torch.float32),
    )

    assert out_mask.all()
    assert torch.all(pos[:, 0] >= 0.0)


# --- support_idx: which surface the xy samplers place onto -------------------------------------------------


def _level_surface(height, num_envs=2, valid=True):
    """A level +z surface at ``height``, valid in every env (or in none)."""
    support = _support(torch.full((num_envs,), valid, dtype=torch.bool))
    support.transform = support.transform.clone()
    support.transform[:, 2, 3] = height
    return support


_XY_SAMPLERS = {
    "constant": lambda surfaces, **kw: constant_xy_sampler(surfaces, _zero_extent_asset(2), 2, pos_xy=(0.1, 0.2), **kw),
    "range": lambda surfaces, **kw: range_xy_sampler(
        surfaces, _zero_extent_asset(2), 2, range_x=(0.0, 0.1), range_y=(0.0, 0.1), **kw
    ),
}


@pytest.mark.parametrize("sampler", list(_XY_SAMPLERS))
@pytest.mark.parametrize(("support_idx", "height"), [(None, 0.1), (0, 0.1), (1, 0.3), (-1, 0.3), (-2, 0.1)])
def test_support_idx_picks_a_surface_like_a_list_index(sampler, support_idx, height):
    """``-1`` is the last surface (it used to be clamped to the first)."""
    surfaces = [_level_surface(0.1), _level_surface(0.3)]
    kwargs = {} if support_idx is None else {"support_idx": support_idx}

    pos, _, _ = _XY_SAMPLERS[sampler](surfaces, **kwargs)

    assert torch.allclose(pos[:, 2], torch.full((2,), height))


@pytest.mark.parametrize("sampler", list(_XY_SAMPLERS))
def test_by_default_the_first_surface_that_can_hold_the_object_is_used(sampler):
    surfaces = [_level_surface(0.1, valid=False), _level_surface(0.3)]

    pos, _, _ = _XY_SAMPLERS[sampler](surfaces)

    assert torch.allclose(pos[:, 2], torch.full((2,), 0.3))


@pytest.mark.parametrize("sampler", list(_XY_SAMPLERS))
def test_an_explicit_index_counts_every_surface_so_it_does_not_shift_with_the_object(sampler):
    """Index 2 is the third surface whether or not the second can hold this object."""
    surfaces = [_level_surface(0.1), _level_surface(0.2, valid=False), _level_surface(0.3)]

    pos, _, _ = _XY_SAMPLERS[sampler](surfaces, support_idx=2)

    assert torch.allclose(pos[:, 2], torch.full((2,), 0.3))


@pytest.mark.parametrize("sampler", list(_XY_SAMPLERS))
def test_an_explicit_index_on_a_surface_that_cannot_hold_the_object_raises(sampler):
    surfaces = [_level_surface(0.1), _level_surface(0.2, valid=False)]
    with pytest.raises(ValueError, match=r"support_idx=1 names a surface that cannot hold this object.*1: z=0\.200 m"):
        _XY_SAMPLERS[sampler](surfaces, support_idx=1)


@pytest.mark.parametrize("sampler", list(_XY_SAMPLERS))
@pytest.mark.parametrize("support_idx", [2, -3])
def test_an_out_of_range_support_idx_raises_listing_the_surfaces(sampler, support_idx):
    surfaces = [_level_surface(0.1), _level_surface(0.3)]
    match = rf"support_idx={support_idx} is out of range.*2 surface\(s\): 0: z=0\.100 m.*1: z=0\.300 m"
    with pytest.raises(IndexError, match=match):
        _XY_SAMPLERS[sampler](surfaces, support_idx=support_idx)


@pytest.mark.parametrize("sampler", list(_XY_SAMPLERS))
def test_a_support_with_no_surface_that_can_hold_the_object_raises(sampler):
    with pytest.raises(IndexError, match="no surface that can hold this object.*0: z=0\\.100 m.*cannot hold"):
        _XY_SAMPLERS[sampler]([_level_surface(0.1, valid=False)])
