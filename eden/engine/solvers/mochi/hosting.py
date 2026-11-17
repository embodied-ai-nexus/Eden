"""Host the MochiSolver in a scene of stock Genesis.

Upstream Genesis has no seam for a solver it does not ship: ``Simulator.__init__`` builds a fixed list of solvers,
``Scene.add_entity`` decides a surface's defaults from a closed chain of ``isinstance`` checks on the entity spec, and
the visualizer names each solver it draws. This module stands in for the edits a Genesis fork would make, one scene at
a time and never on a class, so a scene without a Mochi entity runs exactly the upstream code:

* :func:`install_mochi_solver` builds the solver, appends it to the simulator's solvers and maps the Mochi entity specs
  (:mod:`~eden.engine.solvers.mochi.specs`) to it in the simulator's dispatch. From then on the scene takes Mochi
  materials and specs in ``scene.add_entity``, holds the solver as ``scene.mochi_solver`` and draws its bodies with the
  rasterizer. The simulator saves, restores and frees the solver's arrays as it does a stock solver's (see
  ``MochiSolver.data``). It refuses ``scene.export``, ``scene.save_checkpoint``, a trajectory recording and a
  pickle: Genesis rebuilds a scene from any of them on stock Genesis, without the solver;
* :func:`create_mochi_scene` is ``gs.Scene(..., mochi_options=...)`` of the fork: a new scene, installed;
* :func:`add_mochi_entity` is what ``scene.add_entity`` does for a Mochi material or spec;
* :func:`find_mochi_solver` returns the solver of a scene, or None.

What the solver itself needs beyond this (registering its error flags, clearing them when an environment is reset)
lives in :class:`~eden.engine.solvers.mochi.mochi_solver.MochiSolver`.
"""

import types
from typing import TYPE_CHECKING

import genesis as gs
import genesis.utils.mesh as mu
import numpy as np
import trimesh
from genesis.engine.scene import Scene
from genesis.engine.simulator import Simulator
from genesis.ext import pyrender
from genesis.utils.misc import qd_to_numpy
from genesis.vis.rasterizer_context import RasterizerContext
from genesis.vis.visualizer import Visualizer

from eden.engine._surface import require_members
from eden.engine.solvers.mochi import materials as mochi_materials
from eden.engine.solvers.mochi import specs as mochi_specs
from eden.engine.solvers.mochi.mochi_solver import MochiSolver
from eden.engine.solvers.mochi.options import MochiOptions

# The instance attributes this module writes or reads (`Simulator._solvers`, `Simulator._solvers_by_spec_cls`,
# `Scene.options`) are not class members; tests/engine/solvers/mochi/test_hosting.py pins what they do.
require_members(
    __name__,
    {
        Scene: (
            "__reduce__",
            "add_entity",
            "export",
            "is_built",
            "save_checkpoint",
            "sim",
            "start_recording",
            "visualizer",
        ),
        Simulator: ("add_entity", "solvers"),
        Visualizer: ("batch_renderer", "context", "raytracer"),
        RasterizerContext: (
            "_rigid_solvers",
            "add_node",
            "build",
            "create_node_seg",
            "on_fem",
            "update_fem",
            "update_link_frame",
            "update_rigid",
        ),
    },
)

if TYPE_CHECKING:
    from genesis.engine.entities.base_entity import Entity
    from genesis.options.morphs import Morph
    from genesis.options.surfaces import Surface

_MOCHI_SPEC_CLASSES = (mochi_specs.MochiRigidSpec, mochi_specs.MochiSoftSpec)


def find_mochi_solver(scene: Scene) -> MochiSolver | None:
    """The MochiSolver installed in ``scene``, or None if no Mochi entity was ever added to it."""
    for solver in scene.sim.solvers:
        if isinstance(solver, MochiSolver):
            return solver
    return None


def install_mochi_solver(scene: Scene, options: MochiOptions | None = None) -> MochiSolver:
    """Build a MochiSolver for ``scene`` and register it with its simulator and its visualizer.

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
    solver = MochiSolver(sim, options)
    sim._solvers.append(solver)
    # `Simulator.add_entity` hands a spec to the solver its class maps to, the nearest along the MRO of the class, so
    # a Mochi spec reaches this solver ahead of the RigidSolver `gs.entities.Rigid` maps to.
    for spec_cls in _MOCHI_SPEC_CLASSES:
        sim._solvers_by_spec_cls[spec_cls] = solver
    # The name a scene of the fork holds the solver under, which the code written for the fork reads.
    scene.mochi_solver = solver
    _bind_add_entity(scene)
    _bind_rasterizer(scene, solver)
    _bind_scene_files(scene)
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
    morph: "Morph | None" = None,
    material: "mochi_materials.Base | None" = None,
    surface: "Surface | None" = None,
    visualize_contact: bool | None = None,
    vis_mode: str | None = None,
    name: str | None = None,
    *,
    spec: "mochi_specs.MochiRigidSpec | mochi_specs.MochiSoftSpec | None" = None,
) -> "Entity":
    """``Scene.add_entity`` for a Mochi material or spec, on a scene :func:`install_mochi_solver` was called on.

    The upstream ``Scene.add_entity`` cannot take one: it resolves a Mochi material to the spec of the Genesis solver
    it derives from (``MochiRigid`` to ``gs.entities.Rigid``, of the RigidSolver) or to none, and defaults the surface
    of a deformable Mochi body to the particles of a fluid. An installed scene's ``add_entity`` routes them here.
    """
    solver = find_mochi_solver(scene)
    if solver is None:
        gs.raise_exception("No MochiSolver in this scene: call `install_mochi_solver(scene, options)` first.")
    if scene.is_built:
        gs.raise_exception("Entities can be added only before the scene is built.")
    if spec is None:
        if not isinstance(material, mochi_materials.Base):
            gs.raise_exception(f"`add_mochi_entity` takes a Mochi material, got {type(material).__name__}.")
        if vis_mode is not None:
            surface = (gs.surfaces.Default() if surface is None else surface).model_copy(update={"vis_mode": vis_mode})
        spec = mochi_specs.mochi_spec(morph, material, surface, visualize_contact, name)
    elif any(value is not None for value in (morph, material, surface, visualize_contact, vis_mode, name)):
        gs.raise_exception("Pass either an entity spec or individual entity arguments, never both.")
    elif not isinstance(spec, _MOCHI_SPEC_CLASSES):
        gs.raise_exception(
            "`add_mochi_entity` takes a spec of `eden.engine.solvers.mochi.specs` (MochiRigidSpec or MochiSoftSpec), "
            f"got {type(spec).__name__}."
        )
    if spec.morph is None:
        gs.raise_exception("Entity construction requires a morph.")

    # The surface resolves on a copy, since other entities may share it, as upstream's `Scene.add_entity` does
    surface = gs.surfaces.Default() if spec.surface is None else spec.surface.model_copy()
    morph = spec.morphs[0]
    if isinstance(morph, (gs.morphs.Box, gs.morphs.Cylinder)):
        surface.smooth = False
    if surface.double_sided is None:
        surface.double_sided = isinstance(spec.material, mochi_materials.Shell)
    if surface.vis_mode is None:
        surface.vis_mode = "visual"
    allowed = ("visual",) if isinstance(spec, mochi_specs.MochiSoftSpec) else ("visual", "collision", "sdf")
    if surface.vis_mode not in allowed:
        gs.raise_exception(
            f"Unsupported `surface.vis_mode` for material {type(spec.material).__name__}: '{surface.vis_mode}'. "
            f"Expected one of: {list(allowed)}."
        )
    # Mochi contact samples the authored collision triangles, so a file mesh is kept as it is unless asked otherwise.
    # The rigid-entity description would otherwise convexify it, since a `MochiRigid` is a `Rigid`. The morph resolves
    # in place, as upstream's does.
    for each_morph in spec.morphs:
        if isinstance(each_morph, gs.morphs.FileMorph) and each_morph.convexify is None:
            each_morph.convexify = False

    return scene.sim.add_entity(surface, spec)


def _bind_add_entity(scene: Scene) -> None:
    """Route the Mochi materials and specs ``scene.add_entity`` is given to :func:`add_mochi_entity`, the others
    upstream.
    """
    add_entity = scene.add_entity

    def add_entity_with_mochi(
        morph=None, material=None, surface=None, visualize_contact=None, vis_mode=None, name=None, *, spec=None
    ) -> "Entity":
        args = (morph, material, surface, visualize_contact, vis_mode, name)
        if isinstance(spec, _MOCHI_SPEC_CLASSES) or (spec is None and isinstance(material, mochi_materials.Base)):
            return add_mochi_entity(scene, *args, spec=spec)
        if spec is not None and isinstance(getattr(spec, "material", None), mochi_materials.Base):
            gs.raise_exception(
                f"gs.entities.{type(spec).__name__} hands its entity to a Genesis solver, which cannot simulate the "
                f"Mochi material {type(spec.material).__name__}. Pass a spec of `eden.engine.solvers.mochi.specs` "
                "instead: `MochiRigidSpec` for a `MochiRigid`, `MochiSoftSpec` for the deformable materials."
            )
        return add_entity(*args, spec=spec)

    scene.add_entity = add_entity_with_mochi


def _bind_scene_files(scene: Scene) -> None:
    """Refuse to write ``scene`` to a file or a pickle, which Genesis would rebuild without its MochiSolver.

    ``gs.Scene.load``, ``gs.Scene.load_checkpoint``, ``load_trajectory`` and unpickling create the scene from its
    description on stock Genesis, so every Mochi rigid body would load back as a RigidSolver body. The state itself is
    checkpointable: ``scene.get_state()`` / ``scene.reset(state)`` and the simulator's ``__getstate__`` /
    ``__setstate__`` restore it within the process.
    """

    def refuse(action):
        gs.raise_exception(
            f"Cannot {action}: this scene hosts Eden's MochiSolver, and Genesis rebuilds a scene from a file or a "
            "pickle on stock Genesis, which has no MochiSolver, so every Mochi rigid body would load back as a "
            "RigidSolver body. Rebuild the scene from the code or the Eden config that made it instead; within this "
            "process, `state = scene.get_state()` and `scene.reset(state)` save and restore it."
        )

    scene.export = types.MethodType(lambda self, path: refuse(f"export {path!s}"), scene)
    scene.save_checkpoint = types.MethodType(lambda self, path: refuse(f"save a checkpoint to {path!s}"), scene)
    scene.start_recording = types.MethodType(
        lambda self, rec_options: refuse(f"record a trajectory to {getattr(rec_options, 'filename', rec_options)!s}"),
        scene,
    )
    # `pickle` looks `__reduce__` up on the instance once the class defines its own, as Scene does
    scene.__reduce__ = types.MethodType(lambda self: refuse("pickle it"), scene)


def _bind_rasterizer(scene: Scene, solver: MochiSolver) -> None:
    """Draw the MochiSolver's bodies with the rasterizer of ``scene`` (its cameras and its interactive viewer).

    The rasterizer draws the rigid-like bodies of the solvers ``_rigid_solvers`` yields, and the deformable ones of the
    FEMSolver; each gains the MochiSolver here, on this scene's instances. The visual geoms it draws are placed by the
    solver itself (``MochiSolver._update_vgeoms``): Genesis' visualizer no longer refreshes them.

    A setter between two steps syncs the rasterizer with the nodes of the solvers it changed alone
    (``RasterizerContext._updates_solvers``, filled at the build), so the update of the rigid-like and the deformable
    nodes also runs for the MochiSolver.
    """
    context = scene.visualizer.context
    rigid_solvers = context._rigid_solvers

    def _rigid_solvers(self):
        yield from rigid_solvers()
        if solver.is_active:
            yield solver

    on_fem, update_fem = context.on_fem, context.update_fem

    def _soft_vverts(self):
        vverts_pos, _, _ = solver.get_soft_state_render()
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

    build = context.build

    def build_with_mochi(self, *args, **kwargs):
        build(*args, **kwargs)
        drawing_mochi = (self.update_link_frame, self.update_rigid, self.update_fem)
        self._updates_solvers = tuple(
            (update, (*update_solvers, solver) if update in drawing_mochi else update_solvers)
            for update, update_solvers in self._updates_solvers
        )

    context._rigid_solvers = types.MethodType(_rigid_solvers, context)
    context.build = types.MethodType(build_with_mochi, context)
    context.on_fem = types.MethodType(on_fem_with_mochi, context)
    context.update_fem = types.MethodType(update_fem_with_mochi, context)
