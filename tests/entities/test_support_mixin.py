"""Support-surface extraction: which links contribute, how many points each surface samples, and overhang culling."""

from __future__ import annotations

import numpy as np
import pytest
import torch
import trimesh

from eden.entities.support_mixin import SupportSurfaceMixin
from eden.options import BoxOptions, EntityOptions, EnvOptions, PlaneOptions, SceneOptions
from eden.utils.configs import EdenConfig
from tests.support.envs import build_env

pytestmark = pytest.mark.requires_genesis

# 1 m x 1 m tops, so a polygon has unit area and its sample count is ``support_num_sample_points``.
_TABLE_SIZE = (1.0, 1.0, 0.1)
_DENSE_SAMPLE_POINTS = 1500

# A fixed two-level station: a 1 m x 1 m top at z = 0.5 over a shelf of the same size at z = 0.2.
_TOP_Z, _SHELF_Z = 0.5, 0.2
_STATION_MJCF = f"""<mujoco model="station"><worldbody>
  <body name="station_base">
    <body name="table_top" pos="0 0 {_TOP_Z - 0.02}"><geom type="box" size="0.5 0.5 0.02"/></body>
    <body name="shelf" pos="0 0 {_SHELF_Z - 0.02}"><geom type="box" size="0.5 0.5 0.02"/></body>
  </body>
</worldbody></mujoco>"""


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    asset = tmp_path_factory.mktemp("support") / "station.xml"
    asset.write_text(_STATION_MJCF)
    station = dict(file=str(asset), is_fixed_base=True, is_support_enabled=True, links_to_keep=["table_top", "shelf"])
    table = dict(size=_TABLE_SIZE, fixed=True, is_support_enabled=True)
    return build_env(
        EdenConfig(
            env_options=EnvOptions(num_envs=1),
            scene_options=SceneOptions(
                plane=PlaneOptions(),
                station=EntityOptions(**station, default_root_pos=(3.0, 3.0, 0.0)),
                station_top=EntityOptions(
                    **station, default_root_pos=(6.0, 3.0, 0.0), support_links_name=["table_top"]
                ),
                table_default=BoxOptions(**table, default_root_pos=(3.0, 0.0, 0.05)),
                table_dense=BoxOptions(
                    **table, default_root_pos=(6.0, 0.0, 0.05), support_num_sample_points=_DENSE_SAMPLE_POINTS
                ),
            ),
        )
    )


def _surface_heights(entity) -> list[float]:
    return sorted(round(float(s.transform[0, 2, 3]), 3) for s in entity.current_support(minimum_clearance=0.0))


def test_every_link_contributes_by_default_and_support_links_name_restricts_them(env):
    assert _surface_heights(env.entities["station"]) == [_SHELF_Z, _TOP_Z]
    assert _surface_heights(env.entities["station_top"]) == [_TOP_Z]


def test_an_unknown_support_link_raises_listing_the_links_there_are(env):
    """A typo must not silently fall back to "every link"; every missing name is reported at once."""
    links = env.entities["station"].links
    with pytest.raises(ValueError, match=r"\['missing', 'top'\].*available links.*'shelf'.*'table_top'"):
        SupportSurfaceMixin._select_support_links(links, ["table_top", "top", "missing"])


def test_support_sampling_options_reach_a_primitive(env):
    """``support_*`` used to live on ``EntityOptions`` only, so a box used as a table could not configure it."""

    def sampled_points(name):
        return sum(s.num_sample_points for s in env.entities[name].current_support(minimum_clearance=0.0))

    assert sampled_points("table_default") == BoxOptions.model_fields["support_num_sample_points"].default
    assert sampled_points("table_dense") == _DENSE_SAMPLE_POINTS


# ── the overhang filter: samples under occluding geometry are dropped ──────────────────────────────
# Guards the Riverway franka station's "cube spawns inside the franka_control_box" bug.

DOWN = np.array([0.0, 0.0, -1.0])


def _floor():
    """A 1 x 1 m slab whose top is the z=0 plane, footprint x, y in [-0.5, 0.5]."""
    floor = trimesh.creation.box(extents=(1.0, 1.0, 0.02))
    floor.apply_translation((0.0, 0.0, -0.01))
    return floor


def _floor_with_shelf():
    """The floor with a 0.3 x 0.3 x 0.1 m shelf standing on it over x in [-0.4, -0.1], y in [-0.15, 0.15]."""
    shelf = trimesh.creation.box(extents=(0.3, 0.3, 0.1))
    shelf.apply_translation((-0.25, 0.0, 0.05))
    return trimesh.util.concatenate([_floor(), shelf])


def _kept(points, mesh, transform=None):
    """Filter 2-D samples of the z=0 polygon (identity polygon transform) against ``mesh``."""
    transform = np.eye(4) if transform is None else transform
    return SupportSurfaceMixin._drop_occluded_samples(
        np.asarray(points, dtype=float).reshape(-1, 2),
        polygon_transform=transform,
        occlusion_mesh=mesh,
        occlusion_gravity=DOWN,
    )


@pytest.mark.parametrize("transform", [np.eye(4), torch.eye(4).unsqueeze(0)], ids=["numpy", "torch_batched"])
def test_points_under_an_overhang_are_dropped(transform):
    """Samples clear of the shelf survive; samples under its footprint do not.

    The polygon transform is stored as a ``(1, 4, 4)`` torch tensor in production, so that form is accepted too.
    """
    clear = [[0.3, 0.0], [0.4, 0.4], [0.0, -0.4]]
    under = [[-0.2, 0.0], [-0.3, -0.1], [-0.15, 0.1]]
    np.testing.assert_allclose(_kept(clear + under, _floor_with_shelf(), transform), clear)


def test_points_whose_ray_misses_the_mesh_are_dropped():
    """Fail-safe: a polygon vertex a hair outside its facet (a mesh-edge degeneracy) is treated as occluded."""
    on = [[0.0, 0.0], [0.3, 0.3]]
    np.testing.assert_allclose(_kept(on + [[5.0, 5.0], [-5.0, 0.0]], _floor()), on)


def test_no_samples_in_means_none_out():
    assert _kept([], _floor_with_shelf()).shape == (0, 2)
