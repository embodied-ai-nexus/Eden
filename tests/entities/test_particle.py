"""``eden.entities.particle.ParticleEntity``: particle selection and constraints on a built MPM body."""

from __future__ import annotations

import genesis as gs
import pytest
import torch

from eden.entities.particle import ParticleEntity
from eden.options import BoxOptions, EnvOptions, PlaneOptions, SceneOptions
from eden.options.materials import MPMElasticMaterialOptions, RigidMaterialOptions
from eden.utils.configs import EdenConfig
from tests.support.envs import build_env

# The top slab of the MPM cube (z 0.275-0.425), where the rigid handle is pinned.
SLAB = ((-0.08, -0.08, 0.40), (0.08, 0.08, 0.43))


@pytest.fixture(scope="module")
def env():
    """An elastic MPM cube under a rigid handle, as ``examples/recipes/physics/coupler_rigid_mpm.py`` builds it."""
    env = build_env(
        EdenConfig(
            env_options=EnvOptions(
                num_envs=1,
                sim_dt=2e-3,
                sim_substeps=20,
                decimation=1,
                mpm_options=gs.options.MPMOptions(
                    lower_bound=(-0.5, -0.5, 0.0), upper_bound=(0.5, 0.5, 1.0), grid_density=64
                ),
            ),
            scene_options=SceneOptions(
                plane=PlaneOptions(),
                handle=BoxOptions(
                    size=(0.12, 0.12, 0.05),
                    default_root_pos=(0.0, 0.0, 0.45),
                    material=RigidMaterialOptions(needs_coup=False),
                ),
                cube=BoxOptions(
                    size=(0.15, 0.15, 0.15),
                    default_root_pos=(0.0, 0.0, 0.35),
                    material=MPMElasticMaterialOptions(E=5e4, nu=0.3, rho=1000.0),
                ),
            ),
        )
    )
    yield env
    env.destroy()


def test_an_mpm_body_is_a_particle_entity_that_refuses_to_be_posed(env):
    """Position and rotation are baked into the morph; setting them after build raises rather than being dropped."""
    cube = env.entities["cube"]
    assert isinstance(cube, ParticleEntity)
    with pytest.raises(NotImplementedError, match="set_pos"):
        cube.set_pos(torch.zeros(1, 3, device=env.device))
    with pytest.raises(NotImplementedError, match="set_quat"):
        cube.set_quat(torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=env.device))


def test_constrained_particles_follow_the_link_they_are_pinned_to(env):
    """Particles selected by ``get_particles_in_bbox`` and pinned with ``set_particle_constraints`` ride the handle.

    The handle does not couple with the MPM solver (``needs_coup=False``), so only the constraint can carry the
    cube: lifted 5 cm, the handle drags the pinned slab up with it instead of the cube falling away.
    """
    cube, handle = env.entities["cube"], env.entities["handle"]
    mask = cube.get_particles_in_bbox(*SLAB)
    pinned = torch.as_tensor(mask, device=env.device).reshape(-1).bool()
    assert 0 < int(pinned.sum()) < pinned.numel()

    cube.set_particle_constraints(mask, handle.links[0].idx, stiffness=1e5)
    z0 = cube.entity.get_particles_pos().reshape(-1, 3)[:, 2].clone()
    for step in range(50):
        z = 0.45 + 0.05 * (step + 1) / 50
        handle.set_qpos(torch.tensor([[0.0, 0.0, z, 1.0, 0.0, 0.0, 0.0]], device=env.device))
        env.scene.step()
    rise = cube.entity.get_particles_pos().reshape(-1, 3)[:, 2] - z0

    assert float(rise[pinned].mean()) > 0.03
