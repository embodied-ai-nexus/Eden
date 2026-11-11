"""Teach the Nyx plugin about the Mochi solver (not a runnable example).

``gs_nyx_plugin`` bridges Genesis to the Nyx path tracer one solver at a time: it reads rigid poses
off ``scene.rigid_solver`` and deformable render vertices off the FEM / PBD / MPM / SPH solvers. A
Mochi scene is all-Mochi — every body, rigid or soft, belongs to the MochiSolver Eden hosts
(``find_mochi_solver(scene)``), and the rigid solver holds nothing — so without this module the
plugin exports the scene at its *morph* poses and never updates it. The render comes out as a still
frame of the initial configuration, with no error anywhere.

The two things the bridge needs are already shaped like what the plugin reads:

* ``MochiSolver`` derives from ``KinematicSolver``, so it carries the same ``n_vgeoms`` / ``vgeoms``
  / ``dyn_state.vgeoms.pos|quat`` the plugin reads off the rigid solver. It only has to be handed
  that solver instead — plus ``update_vgeoms()`` before each read, which the rigid path gets for
  free from the visualizer and an offscreen Nyx render never runs.
* ``MochiSolver.get_soft_state_render(f)`` returns render-vertex positions in the same
  ``(n_vverts, B)`` layout as ``FEMSolver.get_state_render(f)``, so the deformable bodies fit the
  plugin's FEM slot behind a small adapter. It returns no UVs or faces — those live on the visual
  geoms — so the one-off upload of those is rebuilt here.

Mochi rigid bodies are ``RigidEntity`` subclasses, so they already sort into the exporter's rigid
group and export the way any rigid entity does -- except that the bridge sends a URDF down the
per-vgeom path instead of the sub-scene one. The sub-scene hands Nyx the source URDF to load for
itself, which means the entity's material override never reaches the meshes inside it and every link
Genesis merged is one Nyx cannot place. Per vgeom, both problems are gone: each visual geom is its
own instance carrying its own material, placed from the pose the solver reports, and that is already
the path the plugin uses for an MJCF.

This is a shim over plugin internals, written against ``gs-nyx-plugin`` 0.1.10; it belongs upstream
in ``nyx-for-genesis`` once the plugin learns about the Mochi solver. Call :func:`install` once,
after ``eden.init()`` and before the scene is built.
"""

import os

import genesis as gs
import numpy as np
import torch

from eden.engine.solvers.mochi.hosting import find_mochi_solver
from eden.engine.solvers.mochi.mochi_soft_entity import MochiSoftEntity
from eden.engine.solvers.mochi.morphs import Rod
from eden.utils.misc import get_cache_dir


class _MochiSoftSolver:
    """``FEMSolver``-shaped view of the Mochi solver's deformable bodies.

    Exposes only what ``NyxPyRenderer`` reads off ``scene.fem_solver``: the render-vertex and
    render-face counts that size the shared deformable buffer, the entity list that orders the
    reference table, and the per-step render state.
    """

    def __init__(self, mochi_solver):
        self._mochi = mochi_solver
        # The counts below are the solver's totals over *every* soft entity, while the instances the
        # exporter writes cover only the ones drawn as surfaces. A body in any other vis_mode would
        # silently shift every entity after it in the shared vertex buffer, so reject it here rather
        # than render the scene scrambled.
        for entity in mochi_solver.soft_entities:
            if entity.surface.vis_mode != "visual":
                raise ValueError(
                    f"Mochi deformable entity {entity.idx} has surface.vis_mode="
                    f"'{entity.surface.vis_mode}'. The Nyx bridge draws deformables as surface "
                    f"meshes, so every Mochi soft body must use vis_mode='visual'."
                )

    @property
    def is_active(self):
        return self._mochi.has_soft

    @property
    def entities(self):
        return self._mochi.soft_entities

    @property
    def n_vverts(self):
        return self._mochi.n_soft_vverts

    @property
    def n_vfaces(self):
        return self._mochi.n_soft_vfaces

    @property
    def _sim(self):
        return self._mochi._sim

    def get_state_render(self, f):
        return self._mochi.get_soft_state_render(f)


class _MochiScene:
    """``scene`` proxy handing the plugin the Mochi solver wherever it asks for a rigid / FEM one.

    Both the exporter and the renderer take the scene as their first constructor argument and reach
    the solvers through it, so substituting one object covers every read.
    """

    def __init__(self, scene):
        object.__setattr__(self, "_scene", scene)
        object.__setattr__(self, "mochi_solver", find_mochi_solver(scene))
        object.__setattr__(self, "_soft", _MochiSoftSolver(self.mochi_solver))

    def __getattr__(self, name):
        if name == "rigid_solver":
            return self.mochi_solver
        if name == "fem_solver":
            return self._soft
        return getattr(self._scene, name)


def constant_env_map(brightness: float, multiplier: float = 1.0):
    """A uniform-colour Nyx environment map: the scene's ambient term, and its background.

    Nyx's env-map slot wants a Radiance ``.hdr`` — hand it Genesis' own ``indoor_bright.png`` and it
    takes the asset without a word, then renders its built-in sky instead. Eden already caches a
    constant-colour ``.hdr`` for its Nyx splat renderer, which is exactly this; reuse the encoder
    rather than keeping a second copy of it here.

    Parameters
    ----------
    brightness : float
        Radiance of every texel, in [0, 1]. Around 0.25 reads as a soft studio fill; 0.0 is a black
        background lit by the explicit lights alone.
    multiplier : float, optional
        Scale Nyx applies to the map on top of ``brightness``.
    """
    from gs_nyx import nyx_py_sdk as nps

    from eden.envs.nyx_splat import NyxSplatEnvBase

    env_map = nps.EnvironmentMapAsset()
    env_map.texture = NyxSplatEnvBase._default_constant_env_map_uri(brightness)
    env_map.multiplier = multiplier
    return env_map


def studio_env_map(path: str | None = None, width: int = 1024, height: int = 512):
    """Write a studio panorama as a Radiance ``.hdr``, and return it as a Nyx environment map.

    The panorama is longitude-latitude, top row up. It has a dark floor, a charcoal backdrop rising to a
    lighter band at the horizon, and two softboxes high overhead, a bright key and a dimmer fill. Only the
    backdrop shows behind the scene. The softboxes show up as reflections in glossy surfaces instead.
    The map sits at infinity, so the same one suits a millimetre-scale scene and a metre-scale one.

    Parameters
    ----------
    path : str, optional
        Where the ``.hdr`` is written: by default ``studio_<width>x<height>.hdr`` in Eden's cache, rewritten on every
        call (a few megabytes, and it follows this code).
    width : int
        Panorama width in texels (twice the height).
    height : int
        Panorama height in texels.

    Returns
    -------
    gs_nyx.nyx_py_sdk.EnvironmentMapAsset
        The map, at a unit multiplier.
    """
    from gs_nyx import nyx_py_sdk as nps

    # Imported for its side effect: without it, assigning `EnvironmentMapAsset.texture` segfaults in the SDK
    # (gs-nyx 0.1.6, deterministically). `constant_env_map` survives only because it imports this too.
    import eden.envs.nyx_splat  # noqa: F401

    if path is None:
        path = os.path.join(get_cache_dir("nyx"), f"studio_{width}x{height}.hdr")

    lon = (np.arange(width) + 0.5) / width * 2.0 * np.pi - np.pi
    lat = np.pi / 2 - (np.arange(height) + 0.5) / height * np.pi
    lon, lat = np.meshgrid(lon, lat)

    backdrop = 0.012 + 0.05 * np.exp(-((lat / 0.22) ** 2))  # a soft glow at the horizon
    radiance = np.where(lat < 0.0, 0.010, backdrop)[..., None] * np.array([1.0, 1.0, 1.05])

    def softbox(lon0, lat0, half_lon, half_lat, value, tint):
        # A rounded rectangle on the sphere, its edge feathered over a few texels.
        d = np.maximum(np.abs(np.angle(np.exp(1j * (lon - lon0)))) / half_lon, np.abs(lat - lat0) / half_lat)
        return np.clip((1.0 - d) * 12.0, 0.0, 1.0)[..., None] * value * np.array(tint)

    radiance = radiance + softbox(-0.6, 0.85, 0.55, 0.22, 10.0, (1.0, 0.97, 0.92))
    radiance = radiance + softbox(2.3, 0.70, 0.35, 0.18, 4.0, (0.85, 0.92, 1.0))
    _write_hdr(path, radiance.astype(np.float32))

    env_map = nps.EnvironmentMapAsset()
    env_map.texture = path
    env_map.multiplier = 1.0
    return env_map


def _write_hdr(path: str, rgb: np.ndarray) -> None:
    """Encode ``(H, W, 3)`` linear radiance as an uncompressed (flat) Radiance RGBE file."""
    peak = rgb.max(axis=-1)
    mantissa, exponent = np.frexp(peak)
    scale = np.where(peak > 1e-32, mantissa * 256.0 / np.maximum(peak, 1e-32), 0.0)
    rgbe = np.zeros((*rgb.shape[:2], 4), dtype=np.uint8)
    rgbe[..., :3] = np.clip(rgb * scale[..., None], 0, 255).astype(np.uint8)
    rgbe[..., 3] = np.where(peak > 1e-32, exponent + 128, 0).astype(np.uint8)
    # A flat scanline must not open with (2, 2): a reader takes that as the start of a run-length one.
    rgbe[:, 0, 0] = np.where((rgbe[:, 0, 0] == 2) & (rgbe[:, 0, 1] == 2), 3, rgbe[:, 0, 0])
    height, width = rgb.shape[:2]
    # Written beside the target and renamed over it, so a concurrent run never reads half a file.
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "wb") as f:
        f.write(f"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n-Y {height} +X {width}\n".encode())
        f.write(rgbe.tobytes())
    os.replace(tmp, path)


def install():
    """Patch ``gs_nyx_plugin`` in place so a Mochi scene renders, and animates, under Nyx."""
    from gs_nyx_plugin import nyx_renderer, nyx_scene_exporter, nyx_scene_utils

    if getattr(nyx_renderer, "_eden_mochi_bridge", False):
        return

    # A Mochi deformable body is a dynamic mesh, exported and indexed exactly where a FEM body is.
    # `is_dynamic_mesh_entity` calls the module global patched here, so only the exporter's own
    # by-name imports have to be refreshed on top.
    is_fem_entity = nyx_scene_utils.is_fem_entity
    nyx_scene_utils.is_fem_entity = lambda entity: is_fem_entity(entity) or isinstance(entity, MochiSoftEntity)
    nyx_scene_exporter.is_fem_entity = nyx_scene_utils.is_fem_entity
    nyx_scene_exporter.is_dynamic_mesh_entity = nyx_scene_utils.is_dynamic_mesh_entity

    # Export a Mochi deformable on its own account, not on its morph's. The stock test admits a
    # deformable only when its morph is a file mesh or a primitive, which covers a solid or a shell but
    # not a rod: `Rod` is a polyline and inherits `Morph` alone. An unexported entity is
    # dropped silently and then surfaces as a `TypeError` on a None UUID, deep in the renderer's
    # reference table, naming nothing.
    is_exportable_entity = nyx_scene_utils.is_exportable_entity
    nyx_scene_utils.is_exportable_entity = lambda entity: (
        (entity.morph.visualization and isinstance(entity, MochiSoftEntity)) or is_exportable_entity(entity)
    )
    nyx_scene_exporter.is_exportable_entity = nyx_scene_utils.is_exportable_entity

    # Send a URDF down the per-vgeom path, and name its instances the way the plugin names an MJCF's
    # so the renderer's reference table can address them one by one.
    should_export_at_geom_level = nyx_scene_utils.should_export_at_geom_level
    nyx_scene_utils.should_export_at_geom_level = lambda entity: (
        isinstance(entity.morph, gs.morphs.URDF) or should_export_at_geom_level(entity)
    )
    nyx_scene_exporter.should_export_at_geom_level = nyx_scene_utils.should_export_at_geom_level
    nyx_scene_exporter.NyxSceneExporter._build_vgeom_instances = _name_urdf_vgeom_instances(
        nyx_scene_exporter.NyxSceneExporter._build_vgeom_instances
    )
    nyx_renderer.NyxPyRenderer._build_reference_table = _for_mochi_scene(
        nyx_renderer.NyxPyRenderer._build_reference_table,
        _address_urdf_vgeoms(nyx_renderer.NyxPyRenderer._build_reference_table),
    )

    nyx_scene_exporter.NyxSceneExporter._build_entity_instance = _smooth_rods(
        nyx_scene_exporter.NyxSceneExporter._build_entity_instance
    )
    nyx_scene_exporter.NyxSceneExporter.__init__ = _exporter_init(nyx_scene_exporter.NyxSceneExporter.__init__)
    nyx_renderer.NyxPyRenderer.__init__ = _renderer_init(nyx_renderer.NyxPyRenderer.__init__)
    # `install()` patches the plugin process-wide, so both of these leave any scene the bridge did
    # not take over on the stock path.
    nyx_renderer.NyxPyRenderer._upload_static_deformable_data = _for_mochi_scene(
        nyx_renderer.NyxPyRenderer._upload_static_deformable_data, _upload_static_deformable_data
    )
    nyx_renderer.NyxPyRenderer._update_geometry_tensors = _refresh_mochi_vgeoms(
        nyx_renderer.NyxPyRenderer._update_geometry_tensors
    )
    nyx_renderer._eden_mochi_bridge = True


def _exporter_init(init):
    def wrapper(self, scene, *args, **kwargs):
        if find_mochi_solver(scene) is not None:
            scene = _MochiScene(scene)
        return init(self, scene, *args, **kwargs)

    return wrapper


def _smooth_rods(build_entity_instance):
    """Shade a rod's tube with interpolated normals, not one flat normal per facet.

    The plugin smooths only the meshes it reconstructs from MPM / SPH particles; every other dynamic mesh
    renders faceted. A rod's tube is a smooth surface tessellated coarsely around its circumference
    (``n_cross_section_segments``), so faceted it reads as a twisted prism.
    """

    def wrapper(self, entity, instance_idx):
        end = build_entity_instance(self, entity, instance_idx)
        if isinstance(entity.morph, Rod):
            instance = self._scene_asset.get_instance(instance_idx)
            instance.smooth = True
            self._scene_asset.set_instance(instance_idx, instance)
        return end

    return wrapper


def _vgeom_node_name(vgeom) -> str:
    """The plugin's own per-vgeom node name: the owning link, then the vgeom's place within it."""
    return f"{vgeom.link.name}_{vgeom.link.vgeoms.index(vgeom)}"


def _name_urdf_vgeom_instances(build_vgeom_instances):
    """Name a URDF's vgeom instances, and re-export their meshes bare. Both are MJCF-only upstream.

    Nyx loads a mesh asset's own embedded materials in preference to the instance's override, so a
    visual geom pointing at its source ``.gltf`` renders with whatever that file says — which for a
    texture Nyx cannot decode is black, whatever the entity's surface asked for. Re-exported the way
    the plugin already re-exports an MJCF's geoms, the mesh carries no material of its own and the
    override is what shades it.
    """

    def wrapper(self, entity, instance_idx):
        end = build_vgeom_instances(self, entity, instance_idx)
        if isinstance(entity.morph, gs.morphs.URDF):
            for idx, vgeom in zip(range(instance_idx, end), entity.vgeoms, strict=True):
                instance = self._scene_asset.get_instance(idx)
                instance.name = _vgeom_node_name(vgeom)
                instance.mesh_uri = self._export_vgeom_to_obj(vgeom)
                self._scene_asset.set_instance(idx, instance)
        return end

    return wrapper


def _address_urdf_vgeoms(build_reference_table):
    """Point a URDF's references at those per-vgeom instances instead of at sub-scene link nodes."""

    def wrapper(self, update_desc, pair_list):
        build_reference_table(self, update_desc, pair_list)
        for ref_idx in range(update_desc.numRigidGeom):
            vgeom = self._rigid_solver.vgeoms[ref_idx]
            if not isinstance(vgeom.entity.morph, gs.morphs.URDF):
                continue
            reference = update_desc.get_reference(ref_idx)
            reference.nodeName = _vgeom_node_name(vgeom)
            update_desc.set_reference(ref_idx, reference)

    return wrapper


def _renderer_init(init):
    def wrapper(self, scene, *args, **kwargs):
        if find_mochi_solver(scene) is not None:
            scene = _MochiScene(scene)
        return init(self, scene, *args, **kwargs)

    return wrapper


def _for_mochi_scene(stock, mochi):
    """Dispatch a renderer method on whether the bridge took this scene over."""

    def wrapper(self, *args, **kwargs):
        method = mochi if isinstance(self._scene, _MochiScene) else stock
        return method(self, *args, **kwargs)

    return wrapper


def _refresh_mochi_vgeoms(update_geometry_tensors):
    def wrapper(self, *args, **kwargs):
        if isinstance(self._scene, _MochiScene):
            # The visual geoms follow the links, and only the visualizer refreshes them — an
            # offscreen Nyx render never runs it, so without this every frame would repeat the
            # pose the scene was built at.
            self._scene.mochi_solver.update_vgeoms()
        return update_geometry_tensors(self, *args, **kwargs)

    return wrapper


def _upload_static_deformable_data(self):
    """Upload the Mochi deformables' UVs and triangles into the plugin's FEM section.

    Replaces the stock method wholesale rather than extending it: in a Mochi scene the FEM, PBD and
    MPM solvers it otherwise reads are all inactive, so the deformable section holds Mochi bodies
    alone. ``get_soft_state_render`` carries only positions, so both static arrays are assembled
    here from the visual geoms, in the solver's own vertex order.
    """
    if self._num_deform_verts == 0:
        return

    device = self._deformable_uvs_backend.device
    uvs, faces = [], []
    for entity in self._fem_solver.entities:
        for vgeom in entity.vgeoms:
            uvs.append(
                torch.zeros((vgeom.n_vverts, 2), dtype=torch.float32, device=device)
                if vgeom.uvs is None
                else torch.as_tensor(vgeom.uvs, dtype=torch.float32, device=device)
            )
            # The vmesh faces index into the vgeom; the shared buffer is indexed solver-globally.
            faces.append(torch.as_tensor(vgeom.vmesh.faces, dtype=torch.int32, device=device) + vgeom.vvert_start)

    self._deformable_uvs_backend[: self._num_fem_verts].copy_(torch.cat(uvs))
    # One row of triangle indices per environment, all holding the same topology.
    self._deformable_indices_backend[:, : self._num_fem_tris * 3].copy_(torch.cat(faces).reshape(-1))
