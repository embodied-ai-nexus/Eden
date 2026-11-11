"""Host the MochiSolver in a scene of stock Genesis.

Upstream Genesis has no seam for a solver it does not ship: ``Simulator.__init__`` builds a fixed list of solvers,
``Scene.add_entity`` decides a surface's defaults from a closed chain of ``isinstance`` checks on the material, and the
visualizer names each solver it draws. This module stands in for the edits a Genesis fork would make, one scene at a
time and never on a class, so a scene without a Mochi entity runs exactly the upstream code:

* :func:`install_mochi_solver` builds the solver and appends it to the simulator's solvers. From then on the scene takes
  Mochi materials in ``scene.add_entity``, holds the solver as ``scene.mochi_solver``, draws its bodies with the
  rasterizer, restores them on a reset and frees the solver's arrays with the scene, as a scene of the fork would. It
  refuses ``scene.export``, which the fork allows: ``gs.Scene.load`` rebuilds a scene on stock Genesis, without the
  solver;
* :func:`create_mochi_scene` is ``gs.Scene(..., mochi_options=...)`` of the fork: a new scene, installed;
* :func:`add_mochi_entity` is what ``scene.add_entity`` does for a Mochi material;
* :func:`find_mochi_solver` returns the solver of a scene, or None.

What the solver itself needs beyond this (registering its error flags, clearing them when an environment is reset)
lives in :class:`~eden.engine.solvers.mochi.mochi_solver.MochiSolver`.
"""

import collections.abc
import types
from typing import TYPE_CHECKING

import genesis as gs
import genesis.utils.mesh as mu
import numpy as np
import trimesh
from genesis.engine.scene import Scene
from genesis.engine.simulator import Simulator
from genesis.ext import pyrender
from genesis.options.morphs import Morph
from genesis.utils.misc import qd_to_numpy
from genesis.vis.rasterizer_context import RasterizerContext
from genesis.vis.visualizer import Visualizer

from eden.engine._surface import require_members
from eden.engine.solvers.mochi import materials as mochi_materials
from eden.engine.solvers.mochi.mochi_solver import MochiSolver
from eden.engine.solvers.mochi.options import MochiOptions

# The instance attributes this module writes or reads (`Simulator._solvers`, `Simulator._entities`, `Scene._desc`,
# `Scene.options`) are not class members; tests/engine/solvers/mochi/test_hosting.py pins what they do.
require_members(
    __name__,
    {
        Scene: ("add_entity", "export", "is_built", "sim", "visualizer"),
        Simulator: ("cur_step_global", "destroy", "n_entities", "reset", "solvers"),
        Visualizer: ("batch_renderer", "context", "raytracer"),
        RasterizerContext: ("_rigid_solvers", "add_node", "create_node_seg", "on_fem", "update_fem"),
    },
)

if TYPE_CHECKING:
    from genesis.engine.entities.base_entity import Entity
    from genesis.engine.materials.base import Material
    from genesis.options.surfaces import Surface

_SOFT_MATERIALS = (mochi_materials.Elastic, mochi_materials.Shell, mochi_materials.Rod)


def find_mochi_solver(scene: Scene) -> MochiSolver | None:
    """The MochiSolver installed in ``scene``, or None if no Mochi entity was ever added to it."""
    for solver in scene.sim.solvers:
        if isinstance(solver, MochiSolver):
            return solver
    return None


def install_mochi_solver(scene: Scene, options: MochiOptions | None = None) -> MochiSolver:
    """Build a MochiSolver for ``scene`` and register it with its simulator, its visualizer and its reset.

    Call it once per scene, before the scene is built and before any Mochi entity is added.

    Parameters
    ----------
    scene : Scene
        The unbuilt scene. Its renderer must be the rasterizer, the only one the solver is bound to.
    options : MochiOptions | None
        The solver's options. Their unset ``dt`` and ``gravity`` come from the scene's ``SimOptions``.

    Returns
    -------
    MochiSolver
        The installed solver, also reachable as ``scene.mochi_solver``.
    """
    if scene.is_built:
        gs.raise_exception("The MochiSolver must be installed before the scene is built.")
    if find_mochi_solver(scene) is not None:
        gs.raise_exception("This scene already has a MochiSolver: install it once, then add every Mochi entity to it.")
    renderer = scene.visualizer.raytracer or scene.visualizer.batch_renderer
    if renderer is not None:
        gs.raise_exception(
            f"The MochiSolver cannot be drawn by {type(renderer).__name__}: Eden binds it to the rasterizer only, "
            "which every camera of this scene would then skip. Pass `renderer=gs.renderers.Rasterizer()`, or render a "
            "Mochi scene with a Nyx camera (examples/recipes/mochi/_nyx.py)."
        )

    sim = scene.sim
    # SceneOptions fills the values a solver's options leave unset (dt, gravity) from SimOptions, for the solvers it
    # declares only.
    options = (options or MochiOptions()).model_copy_from(scene.options.sim)
    solver = MochiSolver(scene, sim, options)
    sim._solvers.append(solver)
    # The name a scene of the fork holds the solver under, which the code written for the fork reads.
    scene.mochi_solver = solver
    _bind_add_entity(scene)
    _bind_reset(sim, solver)
    _bind_destroy(sim, solver)
    _bind_rasterizer(scene, solver)
    _bind_export(scene)
    return solver


def create_mochi_scene(mochi_options: MochiOptions | None = None, **scene_kwargs) -> Scene:
    """A new ``gs.Scene(**scene_kwargs)`` with a MochiSolver installed (see :func:`install_mochi_solver`).

    Parameters
    ----------
    mochi_options : MochiOptions | None
        The solver's options, which a scene of the fork took as ``gs.Scene(mochi_options=...)``.
    **scene_kwargs
        Everything else ``gs.Scene`` takes.

    Returns
    -------
    Scene
        The unbuilt scene, holding the solver as ``scene.mochi_solver``.
    """
    scene = gs.Scene(**scene_kwargs)
    install_mochi_solver(scene, mochi_options)
    return scene


def add_mochi_entity(
    scene: Scene,
    morph: Morph,
    material: "Material",
    surface: "Surface | None" = None,
    visualize_contact: bool = False,
    vis_mode: str | None = None,
    name: str | None = None,
) -> "Entity":
    """``Scene.add_entity`` for a Mochi material, on a scene :func:`install_mochi_solver` was called on.

    The upstream ``Scene.add_entity`` cannot take one: its surface defaults end in a bare exception for the deformable
    materials, and ``Simulator._add_entity`` hands a ``Rigid`` to the RigidSolver, which declares the most derived
    class of the two that claim it. An installed scene's ``add_entity`` routes Mochi materials here.
    """
    solver = find_mochi_solver(scene)
    if solver is None:
        gs.raise_exception("No MochiSolver in this scene: call `install_mochi_solver(scene, options)` first.")
    if scene.is_built:
        gs.raise_exception("Entities can be added only before the scene is built.")
    if not isinstance(material, mochi_materials.Base):
        gs.raise_exception(f"`add_mochi_entity` takes a Mochi material, got {type(material).__name__}.")
    if visualize_contact:
        gs.raise_exception(
            "`visualize_contact` draws the RigidSolver's contacts only; a Mochi entity has none to draw."
        )

    if surface is None:
        surface = gs.surfaces.Default()
    if isinstance(morph, collections.abc.Iterable) and not isinstance(morph, Morph):
        morph = tuple(morph)
    if isinstance(morph[0] if isinstance(morph, tuple) else morph, (gs.morphs.Box, gs.morphs.Cylinder)):
        surface.smooth = False
    if surface.double_sided is None:
        surface.double_sided = isinstance(material, mochi_materials.Shell)
    if vis_mode is not None:
        surface.vis_mode = vis_mode
    if surface.vis_mode is None:
        surface.vis_mode = "visual"
    allowed = ("visual",) if isinstance(material, _SOFT_MATERIALS) else ("visual", "collision", "sdf")
    if surface.vis_mode not in allowed:
        gs.raise_exception(
            f"Unsupported `surface.vis_mode` for material {type(material).__name__}: '{surface.vis_mode}'. Expected "
            f"one of: {list(allowed)}."
        )
    # Mochi contact samples the authored collision triangles, so a file mesh is kept as it is unless asked otherwise.
    # The rigid-entity description would otherwise convexify it, since a `Mochi.Rigid` is a `Rigid`.
    for each_morph in morph if isinstance(morph, tuple) else (morph,):
        if isinstance(each_morph, gs.morphs.FileMorph) and each_morph.convexify is None:
            each_morph.convexify = False

    sim = scene.sim
    entity = solver.add_entity(sim.n_entities, material, morph, surface, visualize_contact, name=name, desc=None)
    sim._entities.append(entity)
    if entity.desc is not None:
        scene._desc.entities.append(entity.desc)
    return entity


def _bind_add_entity(scene: Scene) -> None:
    """Route the Mochi materials ``scene.add_entity`` is given to :func:`add_mochi_entity`, the others upstream."""
    add_entity = scene.add_entity

    def add_entity_with_mochi(
        morph, material=None, surface=None, visualize_contact=False, vis_mode=None, name=None
    ) -> "Entity":
        if isinstance(material, mochi_materials.Base):
            return add_mochi_entity(scene, morph, material, surface, visualize_contact, vis_mode, name)
        return add_entity(morph, material, surface, visualize_contact, vis_mode, name)

    scene.add_entity = add_entity_with_mochi


def _bind_reset(sim, solver: MochiSolver) -> None:
    """Restore the MochiSolver on a reset even when it holds deformable bodies only.

    ``Simulator.reset`` restores a solver only while it holds entities of the kinematic tree, so the state of a scene
    of deformable bodies alone would survive every reset.
    """
    reset = sim.reset
    i_solver = len(sim._solvers) - 1

    def reset_with_deformables_only(state, envs_idx=None):
        reset(state, envs_idx)
        if solver.n_entities == 0 and solver.is_active:
            solver.set_state(0, state.solvers_state[i_solver], envs_idx)

    sim.reset = reset_with_deformables_only


def _bind_destroy(sim, solver: MochiSolver) -> None:
    """Free the MochiSolver's arrays with its scene, which ``Simulator.destroy`` skips (see ``MochiSolver.data``)."""
    destroy = sim.destroy

    def destroy_with_mochi():
        destroy()
        solver.destroy_arrays()

    sim.destroy = destroy_with_mochi


def _bind_export(scene: Scene) -> None:
    """Refuse to export ``scene``, which ``gs.Scene.load`` would rebuild without its MochiSolver."""

    def export(self, path):
        gs.raise_exception(
            f"Cannot export {path!s}: this scene hosts Eden's MochiSolver, and `gs.Scene.load` rebuilds a scene on "
            "stock Genesis, which has no MochiSolver, so every Mochi rigid body would load back as a RigidSolver body. "
            "Rebuild the scene from the code or the Eden config that made it instead."
        )

    scene.export = types.MethodType(export, scene)


def _bind_rasterizer(scene: Scene, solver: MochiSolver) -> None:
    """Draw the MochiSolver's bodies with the rasterizer of ``scene`` (its cameras and its interactive viewer).

    The rasterizer draws the rigid-like bodies of the solvers ``_rigid_solvers`` yields, and the deformable ones of the
    FEMSolver; each gains the MochiSolver here, on this scene's instances. The visualizer refreshes the visual geoms of
    every active solver derived from ``KinematicSolver``, the MochiSolver included, on its own.
    """
    context = scene.visualizer.context
    rigid_solvers = context._rigid_solvers

    def _rigid_solvers(self):
        yield from rigid_solvers()
        if solver.is_active:
            yield solver

    on_fem, update_fem = context.on_fem, context.update_fem

    def _soft_vverts(self):
        vverts_pos, _, _ = solver.get_soft_state_render(self.sim.cur_substep_local)
        return qd_to_numpy(vverts_pos, self.rendered_envs_idx, transpose=True)

    def on_fem_with_mochi(self):
        on_fem()
        if not solver.has_soft:
            return
        vverts_all = _soft_vverts(self)
        for entity in solver.soft_entities:
            if entity.surface.vis_mode != "visual":
                continue
            for i_g, vgeom in enumerate(entity.vgeoms):
                visual = mu.surface_uvs_to_trimesh_visual(vgeom.surface, uvs=vgeom.uvs, n_verts=vgeom.n_vverts)
                seg_key = (entity.idx, i_g) if self.segmentation_level == "geom" else entity.idx
                vverts = vverts_all[:, vgeom.vvert_start : vgeom.vvert_end]
                for env_i, i_b in enumerate(self.rendered_envs_idx):
                    mesh = trimesh.Trimesh(vverts[env_i], vgeom.vmesh.faces, process=False)
                    mesh.visual = visual
                    node = pyrender.Mesh.from_trimesh(
                        mesh, smooth=vgeom.surface.smooth, double_sided=vgeom.surface.double_sided, envs=env_i
                    )
                    static_node = self.add_node(node)
                    self.static_nodes[(i_b, vgeom.uid)] = static_node
                    self.create_node_seg(seg_key, static_node)

    def update_fem_with_mochi(self):
        update_fem()
        if not solver.has_soft:
            return
        vverts_all = _soft_vverts(self)
        for entity in solver.soft_entities:
            if entity.surface.vis_mode != "visual":
                continue
            for vgeom in entity.vgeoms:
                vverts = vverts_all[:, vgeom.vvert_start : vgeom.vvert_end]
                for env_i, i_b in enumerate(self.rendered_envs_idx):
                    node = self.static_nodes[(i_b, vgeom.uid)]
                    render_verts = vverts[env_i].astype(np.float32, copy=False)
                    update_data = self._scene.reorder_vertices(node, render_verts)
                    self.jit.update_buffer(node, "pos", update_data)
                    normal_data = self.jit.update_normal(node, update_data)
                    if normal_data is not None:
                        self.jit.update_buffer(node, "normal", normal_data)

    context._rigid_solvers = types.MethodType(_rigid_solvers, context)
    context.on_fem = types.MethodType(on_fem_with_mochi, context)
    context.update_fem = types.MethodType(update_fem_with_mochi, context)
