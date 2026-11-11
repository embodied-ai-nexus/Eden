"""``eden.entities.fem.FEMEntity``: the COM-offset, centroid and vertex-position setters, on a built FEM body."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from eden.options import BoxOptions, EnvOptions, SceneOptions
from eden.options.materials import FEMElasticMaterialOptions
from eden.utils.configs import EdenConfig
from tests.support.envs import build_env

NUM_ENVS = 2


@pytest.fixture(scope="module")
def cube():
    env = build_env(
        EdenConfig(
            env_options=EnvOptions(num_envs=NUM_ENVS, sim_dt=1e-3, decimation=1),
            scene_options=SceneOptions(
                cube=BoxOptions(
                    size=(0.2, 0.2, 0.2), default_root_pos=(0.0, 0.0, 1.0), material=FEMElasticMaterialOptions()
                )
            ),
        )
    )
    cube = env.entities["cube"]
    yield cube, cube.get_vertex_positions().clone()
    env.destroy()


def test_set_com_offset_translates_every_vertex_from_its_initial_position(cube):
    """The documented contract: an offset from the *build-time* pose, per env or shared by all — never accumulated."""
    entity, rest = cube
    per_env = torch.tensor([[0.1, 0.0, 0.0], [0.0, -0.2, 0.0]], device=rest.device)
    entity.set_com_offset(per_env)
    torch.testing.assert_close(entity.get_vertex_positions(), rest + per_env[:, None], atol=1e-5, rtol=0)
    entity.set_com_offset(torch.tensor([0.0, 0.0, 0.3], device=rest.device))
    torch.testing.assert_close(entity.get_vertex_positions(), rest + torch.tensor([0.0, 0.0, 0.3], device=rest.device))


@pytest.mark.parametrize(
    "as_input", [list, lambda v: np.asarray(v, dtype=np.float64), torch.tensor], ids=["list", "numpy", "host_tensor"]
)
def test_set_com_offset_moves_a_host_offset_to_the_sim_device(cube, as_input):
    """A literal offset (no ``device=``) is host-side: ``eden.init`` sets no torch default device."""
    entity, rest = cube
    entity.set_com_offset(as_input([0.0, 0.0, 0.1]))
    torch.testing.assert_close(entity.get_vertex_positions(), rest + torch.tensor([0.0, 0.0, 0.1], device=rest.device))


def test_set_com_offset_restores_the_undeformed_shape(cube):
    """The offset applies to the build-time vertices, so a deformation written before it is discarded."""
    entity, rest = cube
    deformed = rest.clone()
    deformed[:, 0] += 0.05  # drag one vertex away from the rest shape
    entity.set_vertex_positions(deformed)
    offset = torch.tensor([0.0, 0.1, 0.0], device=rest.device)
    entity.set_com_offset(offset)
    torch.testing.assert_close(entity.get_vertex_positions(), rest + offset, atol=1e-5, rtol=0)


def test_set_pos_places_the_vertex_centroid_where_get_pos_reads_it(cube):
    """``set_pos``'s ``(3,)`` / ``(B, 3)`` forms are absolute, like on every other entity kind — not offsets."""
    entity, rest = cube
    target = torch.tensor([[0.5, 0.0, 0.3], [-0.2, 0.4, 0.6]], device=rest.device)
    entity.set_pos(target)
    torch.testing.assert_close(entity.get_pos(), target, atol=1e-5, rtol=0)
    rest_shape = rest - rest.mean(dim=1, keepdim=True)
    torch.testing.assert_close(entity.get_vertex_positions(), rest_shape + target[:, None], atol=1e-5, rtol=0)


def test_set_vertex_positions_writes_absolute_vertices(cube):
    entity, rest = cube
    tiled = rest[0] + 0.05
    entity.set_vertex_positions(tiled)
    torch.testing.assert_close(entity.get_vertex_positions(), tiled.expand_as(rest).contiguous())
    batched = rest.clone()
    batched[1] += 0.1
    entity.set_vertex_positions(batched)
    torch.testing.assert_close(entity.get_vertex_positions(), batched)


@pytest.mark.parametrize(
    ("setter", "shape", "match"),
    [
        ("set_com_offset", "vertices", "set_vertex_positions"),  # per-vertex data handed to the offset setter
        ("set_com_offset", (NUM_ENVS, 5, 3), "set_com_offset expects"),
        ("set_com_offset", (NUM_ENVS, 2), "set_com_offset expects"),
        ("set_vertex_positions", (3,), "set_com_offset"),  # an offset handed to the vertex setter
        ("set_vertex_positions", "one_vertex_too_many", "set_vertex_positions expects"),
    ],
    ids=["vertices_as_offset", "rank3_offset", "bad_trailing_dim", "offset_as_vertices", "wrong_vertex_count"],
)
def test_a_tensor_of_the_other_meaning_is_rejected_naming_the_right_setter(cube, setter, shape, match):
    """Genesis reads ``set_position`` by rank; the explicit setters refuse the rank that means the other thing."""
    entity, rest = cube
    n = entity.n_vertices
    shape = {"vertices": (n, 3), "one_vertex_too_many": (n + 1, 3)}.get(shape, shape)
    with pytest.raises(ValueError, match=match):
        getattr(entity, setter)(torch.zeros(shape, device=rest.device))


@pytest.mark.parametrize("setter", ["set_com_offset", "set_vertex_positions"])
def test_per_env_addressing_is_refused(cube, setter):
    """Genesis FEM entities cannot update a subset of envs, so ``envs_idx`` raises rather than being dropped."""
    entity, rest = cube
    value = torch.zeros(3, device=rest.device) if setter == "set_com_offset" else rest[0]
    with pytest.raises(NotImplementedError, match="per-env"):
        getattr(entity, setter)(value, envs_idx=torch.tensor([0], device=rest.device))


def test_set_quat_is_refused(cube):
    entity, rest = cube
    with pytest.raises(NotImplementedError, match="set_quat"):
        entity.set_quat(torch.tensor([1.0, 0.0, 0.0, 0.0], device=rest.device))


def test_the_pose_is_the_vertex_centroid_with_an_identity_orientation(cube):
    """Shared pose terms read any entity's pose, so a soft body reports its vertex centroid and no rotation."""
    entity, _ = cube
    centroid = entity.get_vertex_positions().mean(dim=1)
    torch.testing.assert_close(entity.get_pos(), centroid)
    expected = torch.eye(4, device=centroid.device).repeat(NUM_ENVS, 1, 1)
    expected[:, :3, 3] = centroid
    torch.testing.assert_close(entity.get_T(), expected)


@pytest.mark.parametrize("getter", ["get_pos", "get_quat"])
def test_relative_true_is_refused_naming_the_getter(cube, getter):
    """Shared pose terms pass ``relative=False`` to every entity kind; only a re-origined rigid body can answer True."""
    entity, _ = cube
    with pytest.raises(ValueError, match=rf"FEMEntity\.{getter}\(relative=True\) is not meaningful"):
        getattr(entity, getter)(relative=True)
