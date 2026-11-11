"""Viewer / authoring commands: view, arrange, snap, gif."""

from __future__ import annotations

import argparse
import difflib
import os
import numpy as np


def _splat_bbox(ply_path: str) -> tuple[np.ndarray, float]:
    """Return ``(center, max_extent)`` of a gaussian-splat PLY's vertex AABB.

    Reads only the ``x``/``y``/``z`` columns by computing their offsets within
    each vertex record, so it's cheap even for splats with millions of points.
    """
    from eden.utils.splat_ply_converter import _parse_ply_header

    _, properties, vertex_count, header_end_offset = _parse_ply_header(ply_path)

    type_sizes = {"float": 4, "double": 8, "uchar": 1, "int": 4, "uint": 4, "short": 2, "ushort": 2}
    name_to_offset: dict[str, int] = {}
    offset = 0
    for dtype, name in properties:
        name_to_offset[name] = offset
        offset += type_sizes.get(dtype, 4)
    vertex_size = offset

    if not all(k in name_to_offset for k in ("x", "y", "z")):
        # Splat without xyz — fall back to origin with unit extent.
        return np.zeros(3), 1.0

    raw = np.memmap(ply_path, dtype=np.uint8, mode="r", offset=header_end_offset)
    raw = raw[: vertex_count * vertex_size].reshape(vertex_count, vertex_size)
    xyz = np.empty((vertex_count, 3), dtype=np.float32)
    for i, axis in enumerate(("x", "y", "z")):
        col_off = name_to_offset[axis]
        xyz[:, i] = np.frombuffer(raw[:, col_off : col_off + 4].tobytes(), dtype=np.float32)

    lo = xyz.min(axis=0)
    hi = xyz.max(axis=0)
    return (lo + hi) * 0.5, float((hi - lo).max())


def view_splat(ply_path: str, multiplier: float = 1.0, env_brightness: float = 0.5) -> None:
    """Visualize a gaussian-splat ``.ply`` with the Nyx path tracer.

    Builds a minimal Genesis scene (just a far-away plane for TLAS allocation
    plus a Nyx camera sensor), exposes WASD navigation through Genesis's own
    viewer, and renders the corresponding Nyx frame into a cv2 window each
    tick. Press ESC to quit, SPACE to pause/resume.
    """
    import time

    import cv2
    import genesis as gs

    if not os.path.isfile(ply_path):
        raise FileNotFoundError(f"PLY file not found: {ply_path}")

    gs.init(backend=gs.gpu, logging_level="warning")

    from gs_nyx import nyx_py_renderer as npr
    from gs_nyx import nyx_py_sdk as nps
    from gs_nyx_plugin.nyx_camera_options import NyxCameraOptions

    from eden.envs.nyx_splat import NyxSplatEnvBase, write_nyx_camera_profile
    from eden.utils.splat_ply_converter import convert_to_sh3_ply

    center, extent = _splat_bbox(ply_path)
    cam_distance = max(extent * 1.2, 0.5)
    cam_pos = tuple((center + np.array([0.0, -cam_distance, 0.5 * cam_distance])).tolist())
    cam_lookat = tuple(center.tolist())
    res = (1280, 720)

    # Nyx's PLY importer requires SH-degree-3 (``f_rest_0..44``); SH-0 splats
    # are silently rejected. ``convert_to_sh3_ply`` pads with zeros and is a
    # no-op when the file already has the rest coefficients.
    converted = convert_to_sh3_ply(ply_path)
    if converted != ply_path:
        print(f"padded SH-0 splat to SH-3 at: {converted}")

    light_field = nps.LightFieldAsset()
    light_field.uri = converted
    light_field.type = nps.ELightFieldType.GaussianField
    light_field.multiplier = float(multiplier)

    env_map = nps.EnvironmentMapAsset()
    env_map.texture = NyxSplatEnvBase._default_constant_env_map_uri(env_brightness)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.05),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=cam_pos,
            camera_lookat=cam_lookat,
            camera_fov=60,
            res=res,
        ),
        show_viewer=True,
    )
    # Nyx requires at least one geometry instance to allocate a TLAS for shadows/AO.
    scene.add_entity(morph=gs.morphs.Plane(pos=(0.0, 0.0, -1000.0)))

    nyx_cam = scene.add_sensor(
        NyxCameraOptions(
            res=res,
            pos=cam_pos,
            lookat=cam_lookat,
            up=(0.0, 0.0, 1.0),
            spp=4,
            denoise=True,
            near=0.05,
            far=max(cam_distance * 10, 100.0),
            render_mode=npr.ERenderMode.FastPathTracer,
            camera_profile=write_nyx_camera_profile(60.0, "Reinhard"),  # 60 deg: the Genesis viewer's camera_fov
            light_fields=[light_field],
            env_maps=(env_map,),
        )
    )

    scene.build(n_envs=1)
    scene.step()

    window = "Eden Splat Viewer (Nyx)"
    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)
    print(f"viewing {ply_path}  bbox center={center}  extent={extent:.3f}  multiplier={multiplier}")
    print("WASD/QE in the Genesis viewer to move the camera. ESC in the cv2 window quits.")

    last_render = 0.0
    try:
        while True:
            now = time.time()
            if now - last_render < 0.05:
                continue

            viewer = scene.visualizer.viewer
            if viewer is not None:
                nyx_cam.update_camera_pose(
                    pos=tuple(viewer.camera_pos),
                    lookat=tuple(viewer.camera_lookat),
                )

            data = nyx_cam.read()
            rgb = data.rgb[0].cpu().numpy()
            cv2.imshow(window, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            if cv2.waitKey(1) == 27:
                break
            last_render = now
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()


def _composite_robots() -> dict[str, type]:
    """Name -> class for every registered composite rig, keyed by class name *and* registry key."""
    from eden.options.robots import COMPOSITE_ROBOT_REGISTRY

    return {name: cls for key, cls in COMPOSITE_ROBOT_REGISTRY for name in (cls.__name__, key)}


def _find_entity_options(entity_name: str):
    """Look up an EntityOptions or composite-rig Options class by class name or registry key.

    Single entities are resolved by :func:`eden.options.catalog.find_entity_options`, which
    reads a cached index (rebuilt only when the scanned option sources change) and imports
    just the single module defining the match. Accepts class names (``RoboLabYcbBanana``) and
    registry keys (``robolab/ycb/banana``); recurses into subpackages like
    ``eden.options.objects.robolab``.

    Composite rigs are not :class:`~eden.options.entities.BaseEntityOptions` subclasses, so that
    index never sees them; they resolve from
    :data:`~eden.options.robots.COMPOSITE_ROBOT_REGISTRY` instead, by class name (``OliWuji``) or
    registry key (``oli_wuji``). The two namespaces *can* collide, and the single entity wins, so
    adding a rig can never quietly change what an existing name resolves to; reach a shadowed rig
    by its class name, which is unique.
    """
    from eden.options.catalog import find_entity_options

    try:
        return find_entity_options(entity_name)
    except ValueError as err:
        composites = _composite_robots()
        if entity_name in composites:
            return composites[entity_name]
        # The catalog's "did you mean" only knows single entities, so a near-miss on a rig
        # name would otherwise draw unrelated suggestions (or none at all).
        close = difflib.get_close_matches(entity_name, sorted(composites), n=5)
        if not close:
            raise
        suggestions = ", ".join(close)
        raise ValueError(f"{err} Composite rigs matching: {suggestions}.") from err


def _declaration_reaches_base(opts) -> bool:
    """Whether setting ``is_fixed_base`` on ``opts`` pins or frees the entity it builds.

    Genesis has no ``fixed`` switch for MJCF: the file's own root joint decides, and the entity
    raises at build on a declaration that disagrees. So a viewer leaves a plain MJCF on its class's
    declaration — a free one (a G1) cannot be pinned, a jointless one (a Panda) cannot be freed —
    while a family that rewrites its MJCF from the flag (``EntityOptions.mjcf_follows_is_fixed_base``:
    the robosuite / LIBERO objects) is pinned or freed like a URDF.
    """
    return not getattr(opts, "file", "").endswith(".xml") or opts.mjcf_follows_is_fixed_base


def _viewer_scene_entities(options_cls, *, show_collision: bool, fixed_base: bool, hold_pose: bool):
    """Instantiate a resolved Options class for a viewer scene.

    Returns ``(scene entities keyed by scene name, actuated entity name -> its DOF names)``.

    A composite rig declares none of the per-entity viewer knobs — no ``vis_mode``,
    ``is_fixed_base`` or ``material``, and ``extra="forbid"`` rejects them — and must be placed
    under its own ``primary_name``. So the knobs go to its *sub*-entities (only the root is
    pinned; welded children stay free-base so their parent link keeps posing them) and the rig
    goes in whole, leaving the env build loop to expand it and wire the welds. Pinning skips an
    MJCF that cannot be pinned (see :func:`_declaration_reaches_base`). ``hold_pose`` gravity-compensates
    anything with DOFs so slider targets don't sag, set *on* an existing material so a rig that
    declares its own friction keeps it.
    """
    from eden.options.entities import CompositeEntityOptions
    from eden.options.materials import RigidMaterialOptions

    vis_mode = "collision" if show_collision else "visual"

    def dofs_of(opts) -> list[str]:
        return list(getattr(opts, "dofs_name", None) or [])

    def prepare(opts, *, pin: bool) -> None:
        opts.vis_mode = vis_mode
        if pin and _declaration_reaches_base(opts):
            opts.is_fixed_base = True
        if hold_pose and dofs_of(opts):
            if opts.material is None:
                opts.material = RigidMaterialOptions(gravity_compensation=1.0)
            elif hasattr(opts.material, "gravity_compensation"):
                opts.material.gravity_compensation = 1.0

    if issubclass(options_cls, CompositeEntityOptions):
        rig = options_cls()
        # ``entity_options()`` hands back the rig's own sub-options uncopied, so configuring
        # them here configures the rig; ``expand()`` deep-copies at build time.
        subs = rig.entity_options()
        for name, sub in subs.items():
            prepare(sub, pin=fixed_base and name == rig.primary_name)
        return {rig.primary_name: rig}, {n: dofs_of(s) for n, s in subs.items() if dofs_of(s)}

    opts = options_cls()
    prepare(opts, pin=fixed_base)
    return {"obj": opts}, ({"obj": dofs_of(opts)} if dofs_of(opts) else {})


def show_entity(entity_name: str, show_collision: bool = False):
    import torch

    import eden as en
    from eden.constants import ReferenceSource
    from eden.envs.base import EnvBase
    from eden.managers.terms.actions.joint_actions import ImplicitPDController
    from eden.options import ActionManagerOptions, EnvOptions, SceneOptions
    from eden.utils.devices.gui_slider import SliderGUI

    entity_options = _find_entity_options(entity_name)

    en.init(debug=True)

    # Anything with actuated DOFs — a robot, an articulated object, or every actuated part of a
    # composite rig — gets an ImplicitPDController and a slider. Plain meshes and primitives
    # fall through to the passive visualizer loop.
    scene_entities, dofs_by_entity = _viewer_scene_entities(
        entity_options, show_collision=show_collision, fixed_base=True, hold_pose=True
    )

    action_options = (
        ActionManagerOptions(
            **{
                f"{name}_joints": ImplicitPDController.configure(
                    entity_name=name,
                    dofs_name=dofs,
                    reference_source=ReferenceSource.ZERO,
                    scale=1.0,
                )
                for name, dofs in dofs_by_entity.items()
            }
        )
        if dofs_by_entity
        else None
    )

    env = EnvBase(
        env_options=EnvOptions(num_envs=1, show_world_frame=True),
        scene_options=SceneOptions(**scene_entities),
        action_options=action_options,
        show_viewer=True,
    )
    env.build()
    env.reset()

    if not dofs_by_entity:
        while True:
            env.scene.visualizer.update()
        return

    # One flat slider panel over every actuated entity, remembering each one's span so the
    # reading can be split back into per-entity actions.
    labels: list[str] = []
    limits, init_pos, spans = [], [], {}
    for name, dofs in dofs_by_entity.items():
        entity = env.entities[name]
        _, dof_idx = entity.find_named_dofs_idx_local(dofs, preserve_order=True)
        lo, hi = entity.get_dofs_limit(dof_idx)
        limits.append(torch.stack([lo, hi], dim=-1)[0].cpu())
        init_pos.append(entity.get_dofs_pos(dof_idx).squeeze(0).cpu())
        # Qualify the labels only when there is more than one body to attribute them to.
        prefix = f"{name}." if len(dofs_by_entity) > 1 else ""
        spans[name] = slice(len(labels), len(labels) + len(dofs))
        labels += [f"{prefix}{dof}" for dof in dofs]

    slider = SliderGUI(
        viewer=env.scene.viewer,
        dofs_name=labels,
        dofs_pos_limits=torch.cat(limits),
        initial_position=torch.cat(init_pos),
        reset_callback=env.reset,
    )

    while env.scene.viewer.is_alive():
        action = slider.get_command().to(env.device)
        env.step({f"{name}_joints": action[:, span] for name, span in spans.items()})

    slider.close()


def arrange(
    entity_names: list[str],
    show_collision: bool = False,
    output_path: str | None = None,
    spacing: float = 1.0,
):
    """Interactively arrange entities with ImGui sliders (base 6-DoF + joints).

    Each entity is spawned free-base on a ground plane (an MJCF whose file declares no
    free joint stays welded, see :func:`_declaration_reaches_base`); sliders kinematically
    place the base (xyz + roll/pitch/yaw) and every articulated joint. The
    panel also offers drop-to-support-surface, overlap checking, physics settling, and
    exports the final arrangement as a ``SceneOptions`` snippet (printed on
    exit, and written to ``output_path`` if given).
    """
    import eden as en
    from eden.entities.rigid import RigidEntity
    from eden.envs.base import EnvBase
    from eden.options import EnvOptions, SceneOptions
    from eden.options.entities import CompositeEntityOptions, PlaneOptions
    from eden.utils.gui_arrange import ArrangeGUI
    from eden.utils.misc import to_snake_case

    # Resolve options classes and derive unique scene keys (MicrowaveOptions -> microwave).
    entities_options: dict[str, tuple[str, object]] = {}
    for i, cls_name in enumerate(entity_names):
        entity_options = _find_entity_options(cls_name)
        if issubclass(entity_options, CompositeEntityOptions):
            # This command owns one free-base entity per scene key and exports a flat
            # SceneOptions snippet; placing a welded rig's parts independently breaks its welds.
            raise ValueError(
                f"'eden arrange' does not support composite rigs; '{cls_name}' is one. Arrange "
                f"its parts as individual robots, or view the assembled rig with 'eden view "
                f"{cls_name}'."
            )
        base_key = to_snake_case(cls_name.removesuffix("Options")) or "entity"
        key, n = base_key, 1
        while key in entities_options:
            key = f"{base_key}_{n}"
            n += 1
        vis_mode = "collision" if show_collision else "visual"
        opts = entity_options(vis_mode=vis_mode)
        if _declaration_reaches_base(opts):
            # Through the constructor, where a primitive reconciles ``fixed`` with ``is_fixed_base``.
            opts = entity_options(is_fixed_base=False, vis_mode=vis_mode)
        # Spread entities along x so they don't spawn on top of each other.
        px, py, pz = opts.default_root_pos
        opts.default_root_pos = (px + i * spacing, py, pz)
        entities_options[key] = (cls_name, opts)

    en.init(debug=True)

    env = EnvBase(
        # Default keybinds are disabled because the ImGui overlay supersedes them, and their
        # re-registration breaks the overlay's Rebuild Scene (genesis forces the same combination
        # for its enable_gui flow).
        env_options=EnvOptions(num_envs=1, show_world_frame=True, enable_default_keybinds=False),
        scene_options=SceneOptions(
            plane=PlaneOptions(),
            **{key: opts for key, (_, opts) in entities_options.items()},
        ),
        show_viewer=True,
    )
    env.build()
    env.reset()

    for key in entities_options:
        if not isinstance(env.entities[key], RigidEntity):
            raise ValueError(f"'eden arrange' only supports rigid entities; '{key}' is not one.")

    gui = ArrangeGUI(
        scene=env.scene,
        entities={key: (cls_name, env.entities[key]) for key, (cls_name, _) in entities_options.items()},
        output_path=output_path,
    )

    while env.scene.viewer.is_alive():
        gui.update()
    gui.export()


def snap_entity(entity_name: str, show_collision: bool = False, output_path: str | None = None):
    import torch

    import eden as en
    from eden.envs.base import EnvBase
    from eden.options import EnvOptions, SceneOptions
    from eden.options.camera import CameraOptions, CamerasOptions

    entity_options = _find_entity_options(entity_name)

    en.init(debug=True)

    scene_entities, _ = _viewer_scene_entities(
        entity_options, show_collision=show_collision, fixed_base=True, hold_pose=False
    )

    env = EnvBase(
        env_options=EnvOptions(num_envs=1, show_world_frame=True),
        scene_options=SceneOptions(**scene_entities),
        cameras_options=CamerasOptions(
            camera=CameraOptions(),
        ),
        show_viewer=False,
    )
    env.build()

    cam = env.cameras["camera"]
    # Frame every body the target produced — a composite rig expands into several, and framing
    # only its root would crop the hands off.
    corners = torch.cat([e.get_AABB()[0].cpu() for e in env.entities.values()])  # (2 * n, 3)
    lo, hi = corners.min(0).values.numpy(), corners.max(0).values.numpy()
    center = (lo + hi) * 0.5  # (3,)
    extent = hi - lo  # (3,)
    cam.set_pose(
        pos=center + extent * np.array([2.0, 2.0, 1.0]),
        lookat=center,
    )
    cam.snapshot(output_path)


def mp4_to_gif(video_path: str):
    import subprocess

    gif_path = video_path.replace(".mp4", ".gif")
    subprocess.run(
        f'ffmpeg -i {video_path} -vf "fps=10,scale=640:-1:flags=lanczos,split[s0][s1];[s0]palettegen[p];[s1][p]paletteuse" {gif_path}',
        shell=True,
    )


def _run_gif(args) -> None:
    mp4_to_gif(args.video_path)


def _run_view(args) -> None:
    if args.target.lower().endswith(".ply") and os.path.isfile(args.target):
        view_splat(args.target, multiplier=args.multiplier, env_brightness=args.env_brightness)
    else:
        show_entity(args.target, args.collision)


def _run_arrange(args) -> None:
    arrange(args.entities, args.collision, args.output, args.spacing)


def _run_snap(args) -> None:
    snap_entity(args.entity_name, args.collision, args.output)


def _configure_gif(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("video_path", type=str, help="Path to video file")


def _configure_view(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("target", type=str, help="Entity name, or path to a .ply gaussian splat")
    parser.add_argument("-c", "--collision", action="store_true", help="Show collision geometry (entity mode)")
    parser.add_argument(
        "-m",
        "--multiplier",
        type=float,
        default=1.0,
        help="LightFieldAsset.multiplier for splat radiance (splat mode)",
    )
    parser.add_argument(
        "--env-brightness",
        type=float,
        default=0.5,
        help="Constant-color env map brightness in [0, 1] (splat mode)",
    )


def _configure_arrange(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "entities",
        type=str,
        nargs="+",
        help="EntityOptions class names to arrange (e.g. FrankaEmikaPanda YCB003CrackerBox)",
    )
    parser.add_argument("-c", "--collision", action="store_true", help="Show collision geometry")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Also write the exported config snippet to this file",
    )
    parser.add_argument(
        "--spacing",
        type=float,
        default=1.0,
        help="Initial x-spacing between spawned entities (default: 1.0 m)",
    )


def _configure_snap(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("entity_name", type=str, help="Name of the entity to snapshot")
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        help="Output path for the snapshot",
        default="snapshot.png",
    )
    parser.add_argument("-c", "--collision", action="store_true", help="Show collision geometry")
