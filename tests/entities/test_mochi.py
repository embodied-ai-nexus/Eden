"""Build tests for the Mochi entity wrappers (rigid + deformable) against a real Genesis scene.

The session fixture initializes Genesis at single precision, where the MochiSolver warns about its default
1e9 Pa/m contact stiffness — these scenes lower ``penalty_coefficient`` so the warning is not a spurious
part of the test output. Precision is not what is under test here; the wrappers are.
"""

import numpy as np
import pytest
import torch
import trimesh

from eden.engine.solvers.mochi.options import MochiOptions
from eden.envs.base import EnvBase
from eden.options import EnvOptions, RodOptions, SceneOptions
from eden.options.entities import BoxOptions, EntityOptions, PlaneOptions, SphereOptions
from eden.options.materials import (
    MochiElasticMaterialOptions,
    MochiRigidMaterialOptions,
    MochiRodMaterialOptions,
)

SOFT_CONTACT = {"penalty_coefficient": 1e7}


@pytest.mark.requires_genesis
@pytest.mark.slow
def test_mochi_rigid_and_soft_entities_build_and_step():
    """A batched scene of a rigid box, a tetrahedral solid and a rod builds, steps, and reads back per-env."""
    env = EnvBase(
        env_options=EnvOptions(
            num_envs=2,
            sim_dt=1.0 / 60.0,
            mochi_options=MochiOptions(n_newton_iterations=4),
        ),
        scene_options=SceneOptions(
            plane=PlaneOptions(material=MochiRigidMaterialOptions(**SOFT_CONTACT)),
            box=BoxOptions(
                size=(0.2, 0.2, 0.2),
                default_root_pos=(0.6, 0.0, 0.3),
                material=MochiRigidMaterialOptions(**SOFT_CONTACT),
            ),
            blob=SphereOptions(
                radius=0.1,
                default_root_pos=(0.0, 0.0, 0.4),
                material=MochiElasticMaterialOptions(E=5e4, **SOFT_CONTACT),
                morph={"maxvolume": 2e-4, "nobisect": False},
            ),
            rope=RodOptions(
                points=np.stack([np.linspace(-0.2, 0.2, 8), np.full(8, -0.6), np.full(8, 0.3)], axis=-1),
                radius=0.01,
                material=MochiRodMaterialOptions(**SOFT_CONTACT),
            ),
        ),
    )
    env.build()
    assert env.mochi_solver.is_active

    box, blob, rope = env.entities["box"], env.entities["blob"], env.entities["rope"]
    assert blob.get_vertex_positions().shape == (2, blob.n_vertices, 3)
    assert blob.get_vertex_velocities().shape == (2, blob.n_vertices, 3)
    assert blob.get_quat().shape == (2, 4) and float(blob.get_mass()) > 0.0
    assert rope.is_rod and rope.n_vertices == 8
    assert box.get_pos().shape == (2, 3)

    # A pinned vertex set is a Dirichlet condition: those rows must not move while the body falls.
    top = np.flatnonzero(blob.init_positions[:, 2] > blob.init_positions[:, 2].max() - 1e-3)
    assert len(top) > 0
    blob.set_vertices_fixed(top)
    pinned_before = blob.get_vertex_positions()[:, top].clone()
    for _ in range(3):
        env.scene.step()
    assert float((blob.get_vertex_positions()[:, top] - pinned_before).abs().max()) < 1e-9
    assert float(box.get_pos()[0, 2]) < 0.3  # the unpinned box fell


@pytest.mark.requires_genesis
@pytest.mark.slow
def test_mochi_articulated_entity_reads_dof_info():
    """An articulated Mochi robot builds through RigidEntity.post_build and its per-DOF info reads back.

    The readback is the part that breaks first: ``post_build`` reads gains, stiffness, damping, armature and
    the force range from the solver, and per-env DOF info is a ``MochiOptions`` setting, not an
    ``EnvOptions`` one.
    """
    import eden as en

    env = EnvBase(
        env_options=EnvOptions(
            num_envs=2,
            sim_dt=0.01,
            mochi_options=MochiOptions(n_newton_iterations=4),
        ),
        scene_options=SceneOptions(
            plane=PlaneOptions(material=MochiRigidMaterialOptions(**SOFT_CONTACT)),
            franka=en.robots.FrankaEmikaPanda(material=MochiRigidMaterialOptions(**SOFT_CONTACT)),
        ),
    )
    env.build()
    franka = env.entities["franka"]
    assert franka.default_dofs_kp.shape == (1, 9)
    assert franka.get_dofs_kp().shape == (2, 9)
    assert franka.get_dofs_kp()[0, 0] > 0.0
    for getter in (franka.get_dofs_stiffness, franka.get_dofs_damping, franka.get_dofs_armature):
        assert getter().shape == (2, 9)
    assert franka.default_dofs_force_limits.shape == (1, 9)
    env.scene.step()
    assert franka.get_dofs_pos().shape == (2, 9)


@pytest.mark.requires_genesis
@pytest.mark.slow
def test_verify_build_rejects_a_massless_mochi_body():
    """Mochi link masses are fixed at build, so ``_verify_build`` raises instead of patching the mass."""
    env = EnvBase(
        env_options=EnvOptions(num_envs=1, sim_dt=1.0 / 60.0, mochi_options=MochiOptions()),
        scene_options=SceneOptions(
            plane=PlaneOptions(material=MochiRigidMaterialOptions(**SOFT_CONTACT)),
            speck=BoxOptions(
                size=(0.01, 0.01, 0.01),
                default_root_pos=(0.0, 0.0, 0.3),
                material=MochiRigidMaterialOptions(rho=1e-3, **SOFT_CONTACT),
            ),
        ),
    )
    with pytest.raises(ValueError, match="MochiSolver cannot patch"):
        env.build()


_TET_VERTS = np.array([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.1, 0.0], [0.0, 0.0, 0.1]])


def _write_one_tet(tmp_path, suffix: str) -> str:
    """One tetrahedron as a tetgen pair (the '.ele' always lower case, as the loader opens it) or a legacy VTK grid."""
    path = str(tmp_path / f"tet{suffix}")
    with open(path, "w") as f:
        if suffix.lower() == ".node":
            f.write("4 3 0 0\n" + "".join(f"{i} {x} {y} {z}\n" for i, (x, y, z) in enumerate(_TET_VERTS)))
        else:
            points = "".join(f"{x} {y} {z}\n" for x, y, z in _TET_VERTS)
            f.write(f"# vtk DataFile Version 3.0\ntet\nASCII\nDATASET UNSTRUCTURED_GRID\nPOINTS 4 double\n{points}")
            f.write("CELLS 1 5\n4 0 1 2 3\nCELL_TYPES 1\n10\n")
    if suffix.lower() == ".node":
        with open(str(tmp_path / "tet.ele"), "w") as f:
            f.write("1 4 0\n0 0 1 2 3\n")
    return path


@pytest.mark.requires_genesis
@pytest.mark.slow
@pytest.mark.parametrize("suffix", [".NODE", ".VTK"])
def test_an_upper_case_tetrahedral_mesh_file_builds_a_scaled_mochi_solid(tmp_path, suffix):
    """Suffixes match case-insensitively from validation to the solver, and the morph scale applies per axis."""
    env = EnvBase(
        env_options=EnvOptions(num_envs=1, sim_dt=1.0 / 60.0, mochi_options=MochiOptions()),
        scene_options=SceneOptions(
            plane=PlaneOptions(material=MochiRigidMaterialOptions(**SOFT_CONTACT)),
            tet=EntityOptions(
                file=_write_one_tet(tmp_path, suffix),
                scale=(1.0, 2.0, 0.5),
                default_root_pos=(0.0, 0.0, 0.3),
                material=MochiElasticMaterialOptions(E=5e4, **SOFT_CONTACT),
            ),
        ),
    )
    env.build()
    tet = env.entities["tet"]
    assert tet.n_vertices == 4
    extent = np.ptp(tet.get_vertex_positions()[0].cpu().numpy(), axis=0)
    np.testing.assert_allclose(extent, (0.1, 0.2, 0.05), rtol=1e-6)  # the session runs Genesis at fp32


@pytest.mark.requires_genesis
@pytest.mark.slow
def test_set_visual_mesh_after_set_rest_positions_raises():
    """The render mesh is placed through the morph, so it cannot be embedded after the body has moved.

    Getting this backwards is silent: the embedding extrapolates far outside the tetrahedra and the body
    renders as nothing while the physics stays correct.
    """
    env = EnvBase(
        env_options=EnvOptions(num_envs=1, sim_dt=1.0 / 60.0, mochi_options=MochiOptions()),
        scene_options=SceneOptions(
            plane=PlaneOptions(material=MochiRigidMaterialOptions(**SOFT_CONTACT)),
            blob=BoxOptions(
                size=(0.1, 0.1, 0.1),
                default_root_pos=(0.0, 0.0, 0.3),
                material=MochiElasticMaterialOptions(E=5e4, **SOFT_CONTACT),
            ),
        ),
    )

    def move_then_skin():
        blob = env.entities["blob"]
        blob.set_rest_positions(blob.init_positions + np.array([0.0, 0.0, 0.2]))
        with pytest.raises(RuntimeError, match="must be called before set_rest_positions"):
            blob.set_visual_mesh(file="unused.obj")

    env.register_pre_build_hook(move_then_skin)
    env.build()


_CARRIER_MJCF = """<mujoco model="carrier">
  <worldbody>
    <body name="base" pos="0 0 0.5">
      <body name="carrier">
        <joint name="slide" type="slide" axis="1 0 0" range="-0.5 0.5"/>
        <geom type="box" size="0.05 0.05 0.01" mass="1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""
# The block hangs 4 cm under the carrier link's origin: its top face (the one nearest the link) is tied.
_BLOCK_SIZE = (0.06, 0.04, 0.03)
_BLOCK_MOUNT = (0.0, 0.0, -0.04)


def _mount_scene(
    tmp_path,
    *,
    block_material=None,
    mount=_BLOCK_MOUNT,
    link="carrier",
    block_file=None,
    block_quat=(1.0, 0.0, 0.0, 0.0),
) -> EnvBase:
    from eden.options.entities import EntityOptions
    from eden.options.materials import MochiShellMaterialOptions
    from eden.utils.procedural_meshes import box_tet_lattice_file, grid_sheet_file

    is_shell = isinstance(block_material, MochiShellMaterialOptions)
    if block_file is None:
        block_file = (
            grid_sheet_file(_BLOCK_SIZE[:2], (3, 2)) if is_shell else box_tet_lattice_file(_BLOCK_SIZE, (3, 2, 2))
        )

    carrier = tmp_path / "carrier.xml"
    carrier.write_text(_CARRIER_MJCF)
    return EnvBase(
        env_options=EnvOptions(num_envs=2, sim_dt=0.02, mochi_options=MochiOptions(n_newton_iterations=8)),
        scene_options=SceneOptions(
            plane=PlaneOptions(material=MochiRigidMaterialOptions(**SOFT_CONTACT)),
            carrier=EntityOptions(
                file=str(carrier),
                is_fixed_base=True,
                is_articulated=True,
                is_actuated=True,
                dofs_name=["slide"],
                # The build pose has the slide at zero; the block must still end up under the moved link.
                default_dofs_pos={"slide": 0.1},
                default_dofs_kp={"slide": 1e4},
                default_dofs_kd={"slide": 2e2},
                material=MochiRigidMaterialOptions(has_gravity=False, **SOFT_CONTACT),
            ),
            block=EntityOptions(
                file=block_file,
                default_root_quat=block_quat,
                material=block_material or MochiElasticMaterialOptions(E=5e4, rho=200.0, **SOFT_CONTACT),
            ),
            attachments_dict={"block": ("carrier", link, mount, (1.0, 0.0, 0.0, 0.0))},
        ),
    )


@pytest.mark.requires_genesis
@pytest.mark.slow
def test_attachments_dict_mounts_a_soft_solid_on_a_link_and_carries_it(tmp_path):
    """The face nearest the link is tied to it; the body follows the link from the build pose and as it moves."""
    env = _mount_scene(tmp_path)
    env.build()
    block, carrier = env.entities["block"], env.entities["carrier"]
    link_pos = carrier.get_link("carrier").get_pos()
    assert float(link_pos[0, 0]) == pytest.approx(0.1, abs=1e-6)

    def top_face():
        verts = block.get_vertex_positions()
        return verts[:, verts[0, :, 2] > verts[0, :, 2].max() - 1e-4]

    # Carried from the build pose (slide at 0) to the default pose (slide at 0.1) by post_build.
    expected_top_z = 0.5 + _BLOCK_MOUNT[2] + 0.5 * _BLOCK_SIZE[2]
    np.testing.assert_allclose(block.get_pos()[:, 0].cpu(), 0.1, atol=1e-6)
    np.testing.assert_allclose(top_face()[..., 2].cpu(), expected_top_z, atol=1e-6)

    for _ in range(30):
        env.scene.step()
    # Hanging under gravity, the tied face stays within a millimetre and the free bottom sags below it.
    assert float((top_face()[..., 2] - expected_top_z).abs().max()) < 1e-3
    assert float(block.get_pos()[0, 2]) == pytest.approx(0.5 + _BLOCK_MOUNT[2], abs=5e-3)

    carrier.control_dofs_pos(torch.full((2, 1), 0.2, device=env.device))
    for _ in range(60):
        env.scene.step()
    np.testing.assert_allclose(block.get_pos()[:, 0].cpu(), 0.2, atol=5e-3)


@pytest.mark.requires_genesis
@pytest.mark.slow
def test_attachments_dict_mounts_an_off_centre_rotated_mesh_at_its_authored_offset(tmp_path):
    """The morph rotates a mesh about its vertex centroid; the mount must undo that, not a rotation about the origin.

    A lattice 3 cm off its file origin, under a 90 degree root rotation, lands with every vertex at its authored
    offset under the link: 3 cm off in x and y if the rotation is undone about the wrong point.
    """
    from eden.utils.procedural_meshes import box_tet_lattice

    verts, tets = box_tet_lattice(_BLOCK_SIZE, (3, 2, 2))
    verts = verts + np.array([0.03, 0.0, 0.0])
    points = "".join(f"{float(x)!r} {float(y)!r} {float(z)!r}\n" for x, y, z in verts)
    cells = "".join(f"4 {a} {b} {c} {d}\n" for a, b, c, d in tets)
    block_file = tmp_path / "off_centre_block.vtk"
    block_file.write_text(
        f"# vtk DataFile Version 2.0\nblock\nASCII\nDATASET UNSTRUCTURED_GRID\nPOINTS {len(verts)} double\n{points}"
        f"CELLS {len(tets)} {5 * len(tets)}\n{cells}CELL_TYPES {len(tets)}\n" + "10\n" * len(tets)
    )
    half_turn_z = (np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4))
    env = _mount_scene(tmp_path, block_file=str(block_file), block_quat=half_turn_z)
    env.build()

    # The carrier link sits at (0.1, 0, 0.5) unrotated after post_build; the mount quat is identity.
    expected = verts + np.asarray(_BLOCK_MOUNT) + np.array([0.1, 0.0, 0.5])
    actual = env.entities["block"].get_vertex_positions()[0].cpu().numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-5)


@pytest.mark.requires_genesis
def test_attachments_dict_refuses_a_shell_and_a_centred_mount(tmp_path):
    from eden.options.materials import MochiShellMaterialOptions

    with pytest.raises(NotImplementedError, match="mounts tetrahedral solids"):
        _mount_scene(tmp_path, block_material=MochiShellMaterialOptions(**SOFT_CONTACT))
    with pytest.raises(ValueError, match="no face of it is nearest the link"):
        _mount_scene(tmp_path, mount=(0.0, 0.0, 0.0))


@pytest.mark.requires_genesis
def test_attachments_dict_names_the_links_when_the_mount_link_is_missing(tmp_path):
    env = _mount_scene(tmp_path, link="carier")
    with pytest.raises(KeyError, match=r"has no link 'carier'.*\['base', 'carrier'\]|has no link 'carier'"):
        env.build()


def _skin_file(tmp_path, extent: float) -> str:
    path = str(tmp_path / "skin.obj")
    trimesh.creation.box(extents=(extent,) * 3).subdivide().subdivide().subdivide().export(path)
    return path


@pytest.mark.requires_genesis
@pytest.mark.slow
def test_contact_skin_carries_the_contact(tmp_path):
    """A solid contacts through its skin, not its tetrahedra: a cube with a 20% larger skin rests higher by the margin."""
    size = 0.1
    cube = {"size": (size, size, size), "morph": {"maxvolume": 5e-4, "nobisect": False}}
    material = MochiElasticMaterialOptions(E=1e5, nu=0.45, **SOFT_CONTACT)
    env = EnvBase(
        env_options=EnvOptions(num_envs=1, sim_dt=1.0 / 60.0, mochi_options=MochiOptions(n_newton_iterations=8)),
        scene_options=SceneOptions(
            plane=PlaneOptions(material=MochiRigidMaterialOptions(**SOFT_CONTACT)),
            bare=BoxOptions(default_root_pos=(0.0, 0.0, 0.5 * size + 0.02), material=material, **cube),
            skinned=BoxOptions(default_root_pos=(0.5, 0.0, 0.5 * size + 0.04), material=material, **cube),
        ),
    )
    env.register_pre_build_hook(lambda: env.entities["skinned"].set_contact_skin(_skin_file(tmp_path, 1.2 * size)))
    env.build()
    for _ in range(120):
        env.scene.step()

    bare_bottom = float(env.entities["bare"].get_vertex_positions()[0, :, 2].min())
    skinned_bottom = float(env.entities["skinned"].get_vertex_positions()[0, :, 2].min())
    assert skinned_bottom - bare_bottom == pytest.approx(0.1 * size, abs=2e-3)


@pytest.mark.requires_genesis
@pytest.mark.slow
def test_contact_skin_and_set_rest_positions_refuse_each_other(tmp_path):
    """The solver embeds the skin only at build, so it cannot follow a rest-geometry move in either call order."""
    blob = {"size": (0.1, 0.1, 0.1), "material": MochiElasticMaterialOptions(E=5e4, **SOFT_CONTACT)}
    env = EnvBase(
        env_options=EnvOptions(num_envs=1, sim_dt=1.0 / 60.0, mochi_options=MochiOptions()),
        scene_options=SceneOptions(
            plane=PlaneOptions(material=MochiRigidMaterialOptions(**SOFT_CONTACT)),
            moved=BoxOptions(default_root_pos=(0.0, 0.0, 0.3), **blob),
            skinned=BoxOptions(default_root_pos=(0.5, 0.0, 0.3), **blob),
        ),
    )
    skin = _skin_file(tmp_path, 0.11)

    def both_orders():
        moved, skinned = env.entities["moved"], env.entities["skinned"]
        moved.set_rest_positions(moved.init_positions + np.array([0.0, 0.0, 0.2]))
        with pytest.raises(RuntimeError, match=r"set_contact_skin\(\) cannot follow set_rest_positions"):
            moved.set_contact_skin(skin)
        skinned.set_contact_skin(skin)
        with pytest.raises(RuntimeError, match=r"set_rest_positions\(\) cannot follow set_contact_skin"):
            skinned.set_rest_positions(skinned.init_positions + np.array([0.0, 0.0, 0.2]))

    env.register_pre_build_hook(both_orders)
    env.build()
