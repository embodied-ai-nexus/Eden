"""Materialize Genesis-ready MJCF descriptors from benchmark asset files.

The benchmark object options (LIBERO, robosuite) ship the upstream MJCF files verbatim and derive
what Genesis needs at load time: the serialized ``file`` keeps naming the source, so configs stay
portable, while :meth:`~eden.options.entities.EntityOptions.resolve_runtime_file` hands the entity a
transformed copy from one per-process runtime directory (content-addressed, cached, removed at
interpreter exit).

The transforms here fix what Genesis' MJCF loader does differently from MuJoCo / robosuite:

* :func:`clone_collision_geoms_for_rendering` — Genesis makes every geom with ``contype`` /
  ``conaffinity`` a collision-only proxy drawn with the grey ``Collision()`` surface, and only
  ``contype="0" conaffinity="0"`` geoms carry their material. robosuite-style objects declare one
  dual-purpose geom per part (robosuite clones the visual twin programmatically), so without a
  twin every object renders grey whatever its material.
* :func:`flatten_cube_textures` — Genesis maps only ``type="2d"`` textures onto geoms without
  texture coordinates; a ``cube`` (or ``skybox``) texture ends up as garbage UVs. The material is
  given the texture's mean colour instead, so brass reads as brass and dark wood as dark wood.
* :func:`declare_radians` — MuJoCo reads an angle (a joint ``range``, an ``axisangle`` or
  ``euler``) in degrees unless the file's ``<compiler>`` says ``angle="radian"``. robosuite's object
  files declare no compiler: robosuite merges them into its world model, which declares radians,
  and their angles are written in radians. Loaded on their own they would compile as degrees — the
  door's hinge range ``0 0.4`` would become 0.4 degrees and the door could not open.
* :func:`extract_wrapped_object_body`, :func:`add_freejoint_to_root_body`,
  :func:`remove_freejoint_from_root_body` — robosuite wraps an object in ``<body><body
  name="object">`` and adds the free joint at load time; here the wrapper is unwrapped and the
  free joint follows ``is_fixed_base``.
"""

from __future__ import annotations

import atexit
import hashlib
import os
import shutil
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Callable, Iterable
from functools import partial
from pathlib import Path

from eden.utils.assets import get_asset_path

#: An in-place XML transform: ``(root, source_dir) -> modified``.
XmlTransform = Callable[[ET.Element, Path], bool]

_XML_CACHE: dict[tuple[object, ...], str] = {}

# ONE runtime directory per process, removed at interpreter exit. Deriving the file name from the
# cache key instead of taking a fresh ``mkdtemp`` per asset is what bounds this: a scene
# materializes a descriptor per object, workspace and fixture, and a per-asset temp directory would
# leave dozens of never-deleted ``/tmp`` entries behind on every run (thousands across a benchmark
# replay that runs one process per task). Content-addressed names also keep two variants of one
# source file (fixed- vs floating-base) from colliding on their shared basename.
_RUNTIME_DIR: Path | None = None


def runtime_dir() -> Path:
    """Return this process's runtime-descriptor directory, creating it on first use."""
    global _RUNTIME_DIR
    if _RUNTIME_DIR is None:
        directory = Path(tempfile.mkdtemp(prefix="eden_mjcf_"))
        # ignore_errors: interpreter shutdown must not fail over a cache directory a user or a
        # tmp reaper already removed.
        atexit.register(shutil.rmtree, directory, ignore_errors=True)
        _RUNTIME_DIR = directory
    return _RUNTIME_DIR


def runtime_xml_path(cache_key: tuple[object, ...], source_name: str) -> Path:
    """Return the runtime path for one materialized variant, keeping the source file extension.

    The extension is load-bearing: ``Entity._create_morph_from_options`` picks its Genesis morph
    from it, so a digest-only name would be rejected as an unknown file type.
    """
    digest = hashlib.sha256(repr(cache_key).encode("utf-8")).hexdigest()[:16]
    return runtime_dir() / f"{digest}__{source_name}"


def _compiler_dirs(root: ET.Element) -> tuple[str, str]:
    compiler = root.find("compiler")
    if compiler is None:
        return "", ""
    return compiler.get("meshdir", ""), compiler.get("texturedir", "")


def resolve_asset_paths(root: ET.Element, source_dir: Path) -> None:
    """Make mesh and texture references independent of the derived XML's location."""
    asset = root.find("asset")
    if asset is None:
        return
    mesh_dir, texture_dir = _compiler_dirs(root)
    for element in asset:
        if element.tag == "mesh":
            compiler_dir = mesh_dir
        elif element.tag == "texture":
            compiler_dir = texture_dir
        else:
            continue
        file_attribute = element.get("file")
        if file_attribute and not os.path.isabs(file_attribute):
            element.set("file", str((source_dir / compiler_dir / file_attribute).resolve()))


def transform_identity(transform: XmlTransform) -> tuple[str, ...]:
    """A stable, hashable name for one transform, the arguments bound into it included.

    This is what makes the cache safe. Two pipelines over the same source file produce different
    XML, so the cache has to tell them apart, and deriving the key from the pipeline itself is the
    only way that stays true as pipelines are added: a hand-written label has to be *remembered*,
    and a second variant reusing an existing label would silently be served the first one's
    descriptor — a wrong-dynamics bug on a scene that builds and steps fine. Bound arguments are
    part of the identity because they change the output: LIBERO and robosuite differ only in the
    keywords they bind to :func:`clone_collision_geoms_for_rendering`, one of which decides whether
    the visual twin carries mass.

    Arguments are identified by ``repr`` rather than by value, so an unhashable bound argument
    cannot break the cache key.
    """
    if isinstance(transform, partial):
        return (
            *transform_identity(transform.func),
            *(repr(argument) for argument in transform.args),
            *(f"{key}={value!r}" for key, value in sorted(transform.keywords.items())),
        )
    return (f"{transform.__module__}.{transform.__qualname__}",)


def materialize_xml(
    file_path: str,
    *,
    transforms: Iterable[XmlTransform],
    registry: str | None,
    dataset: str | None,
    local_dir: str | None,
) -> str:
    """Apply ``transforms`` to the asset at ``file_path`` and return a runtime descriptor path.

    Returns ``file_path`` itself when the asset cannot be located or no transform changed it. The
    result is cached per (resolved source, mtime, size, pipeline), where the pipeline is derived
    from ``transforms`` by :func:`transform_identity` — so a caller cannot forget to declare what
    makes its output different.
    """
    # ``get_asset_path`` returns None for an asset it cannot locate, so the None check has to come
    # before ``Path(...)`` — otherwise pathlib raises a TypeError that names neither the asset nor
    # this fallback.
    resolved = get_asset_path(file=file_path, registry=registry, dataset=dataset, local_dir=local_dir)
    if resolved is None:
        return file_path
    source_path = Path(resolved)
    if not source_path.is_file():
        return file_path

    transforms = tuple(transforms)
    stat = source_path.stat()
    pipeline = tuple(transform_identity(transform) for transform in transforms)
    cache_key = (str(source_path.resolve()), stat.st_mtime_ns, stat.st_size, pipeline)
    cached_path = _XML_CACHE.get(cache_key)
    if cached_path is not None:
        if Path(cached_path).is_file():
            return cached_path
        del _XML_CACHE[cache_key]

    root = ET.parse(source_path).getroot()
    is_modified = False
    for transform in transforms:
        if transform(root, source_path.parent):
            is_modified = True
    if not is_modified:
        return file_path

    resolve_asset_paths(root, source_path.parent)
    serialized = ET.tostring(root, encoding="utf-8")
    target_path = runtime_xml_path(cache_key, source_path.name)
    # Write-then-rename so a reader never sees a half-written descriptor: the in-memory cache is
    # per-process, but the runtime file can be re-materialized after deletion (and Genesis may be
    # loading a sibling variant from this same directory on another thread).
    staging_path = target_path.with_name(f"{target_path.name}.{os.getpid()}.tmp")
    staging_path.write_bytes(serialized)
    os.replace(staging_path, target_path)
    _XML_CACHE[cache_key] = str(target_path)
    return str(target_path)


# --- transforms -----------------------------------------------------------------------------------


#: Attributes that would place or orient a wrapper body. The unwrap deletes the wrapper, so any of
#: these on it would silently move the object; none of the shipped descriptors carries one.
_WRAPPER_POSE_ATTRIBUTES = ("pos", "quat", "euler", "axisangle", "xyaxes", "zaxis")


def extract_wrapped_object_body(root: ET.Element, _source_dir: Path, *, strict: bool = True) -> bool:
    """Unwrap robosuite's ``<body><body name="object">`` into one root body named ``main``.

    With ``strict`` (the LIBERO contract) a descriptor without exactly that shape raises; without
    it the transform is a no-op on any other layout, so hand-authored composites pass through.

    A wrapper carrying a pose of its own raises either way: the wrapper is deleted here, so its
    transform would be dropped and the object would sit somewhere other than where its own file
    puts it — with nothing to notice, since the scene still builds. Composing it into the child
    instead would mean re-implementing MuJoCo's five orientation spellings; refusing is the honest
    answer until a descriptor actually needs one.
    """
    worldbody = root.find("worldbody")
    if worldbody is None:
        if strict:
            raise ValueError("object descriptor is missing <worldbody>")
        return False
    outer_bodies = worldbody.findall("./body")
    wrapper = outer_bodies[0] if len(outer_bodies) == 1 else None
    source_body = wrapper.find("./body[@name='object']") if wrapper is not None else None
    if source_body is None:
        if strict:
            raise ValueError(
                "object descriptor must be one wrapper body with a direct child body named 'object', "
                f"got {len(outer_bodies)} top-level bodies"
            )
        return False

    posed = {key: wrapper.get(key) for key in _WRAPPER_POSE_ATTRIBUTES if wrapper.get(key) is not None}
    if posed:
        raise ValueError(
            f"the wrapper body around <body name='object'> carries a pose ({posed}); unwrapping deletes the "
            f"wrapper, which would move the object by that transform. Bake it into the 'object' body instead."
        )

    wrapper.remove(source_body)
    for child in list(worldbody):
        worldbody.remove(child)
    source_body.set("name", "main")
    worldbody.append(source_body)
    return True


def _root_body(root: ET.Element) -> ET.Element | None:
    worldbody = root.find("worldbody")
    return None if worldbody is None else worldbody.find("body")


def declare_radians(root: ET.Element, _source_dir: Path) -> bool:
    """Make the file's ``<compiler>`` read angles in radians, as the world robosuite merges its objects into does."""
    compiler = root.find("compiler")
    if compiler is None:
        compiler = ET.Element("compiler")
        root.insert(0, compiler)
    if compiler.get("angle") == "radian":
        return False
    compiler.set("angle", "radian")
    return True


def add_freejoint_to_root_body(root: ET.Element, _source_dir: Path) -> bool:
    """Give the root body a free joint unless it already has a joint (robosuite adds it at load time)."""
    body = _root_body(root)
    if body is None or body.find("freejoint") is not None or body.find("joint") is not None:
        return False
    body.insert(0, ET.Element("freejoint", {"name": "root_joint"}))
    return True


def remove_freejoint_from_root_body(root: ET.Element, _source_dir: Path) -> bool:
    """Drop the root body's free joint, pinning the object to the world (``is_fixed_base=True``)."""
    body = _root_body(root)
    if body is None:
        return False
    joints = body.findall("freejoint") + [j for j in body.findall("joint") if j.get("type") == "free"]
    for joint in joints:
        body.remove(joint)
    return bool(joints)


_RENDER_SIGNATURE_ATTRIBUTES = (
    "type",
    "size",
    "pos",
    "quat",
    "axisangle",
    "euler",
    "xyaxes",
    "zaxis",
    "fromto",
    "mesh",
    "hfield",
    "class",
)


def geom_render_signature(geom: ET.Element) -> tuple[str | None, ...]:
    """The attributes that decide what a geom looks like — two geoms with equal signatures render alike."""
    return tuple(geom.get(attribute) for attribute in _RENDER_SIGNATURE_ATTRIBUTES)


def is_visual_only_geom(geom: ET.Element) -> bool:
    """Whether Genesis will treat ``geom`` as a visual geom (no collision bits)."""
    return geom.get("contype") == "0" and geom.get("conaffinity") == "0"


#: Mass MuJoCo accepts on a render-only geom. Not zero: MuJoCo rejects a massless geom on a body it
#: has to derive inertia for, and this is small enough to round away against any real part.
_TWIN_MASS = "1e-8"


def _visual_twin(element: ET.Element, name: str, *, massless: bool) -> ET.Element:
    """A render-only copy of collision geom ``element``, carrying its material.

    With ``massless`` the twin weighs nothing, which is what a render-only geom should weigh.
    ``contype`` / ``conaffinity`` turn off *collision*, not inertia: MuJoCo's
    ``inertiafromgeom="auto"`` sums every geom inside ``inertiagrouprange`` (0-5 by default), so a
    twin inheriting the original's ``density`` / ``mass`` doubles the body's mass and inertia — and
    Genesis reads that compiled mass straight through. Neither benchmark's objects declare an
    ``<inertial>``, so every one of them is affected. robosuite's own geom duplicator drops the
    mass too.

    Without it the twin keeps whatever the original declared, doubling the body. That is the mass
    LIBERO's descriptors compile to, and :func:`clone_collision_geoms_for_rendering` lets LIBERO
    keep it: correcting those masses changes the dynamics of the 130-task benchmark and invalidates
    policies trained against it.
    """
    visual = ET.Element("geom", element.attrib)
    visual.set("contype", "0")
    visual.set("conaffinity", "0")
    if massless:
        visual.attrib.pop("density", None)
        visual.set("mass", _TWIN_MASS)
    visual.set("name", name)
    return visual


def clone_collision_geoms_for_rendering(
    root: ET.Element,
    _source_dir: Path,
    *,
    groups: tuple[int, ...] | None = (1, 2),
    require_material: bool = True,
    massless_twin: bool = True,
) -> bool:
    """Give every collision geom that Genesis would draw grey a visual-only twin carrying its look.

    A collision geom whose render signature already matches a visual-only sibling is simply moved
    to group 3 (MuJoCo's hidden group). Otherwise, when the geom is in ``groups`` (``None`` = any
    group) and — with ``require_material`` — declares a material, a
    ``contype="0" conaffinity="0"`` copy named ``<name>_vis`` is inserted after it and the original
    moves to group 3.

    ``massless_twin`` decides whether that copy carries inertia (see :func:`_visual_twin`); it is
    the correct default and a switch only so LIBERO can keep its compiled masses. robosuite's
    contract is ``groups=None, require_material=False`` — its dual-purpose geoms sit in group 0
    and may carry a bare ``rgba`` rather than a material; LIBERO's is
    ``groups=(1, 2), require_material=True, massless_twin=False``.
    """
    parents = list(root.iter("body"))
    if (worldbody := root.find("worldbody")) is not None:
        parents.append(worldbody)

    modified = False
    unnamed = 0  # robosuite's nut / plate geoms carry no name; MuJoCo rejects two geoms both called "_vis"
    for parent in parents:
        visual_signatures = {
            geom_render_signature(element)
            for element in parent
            if element.tag == "geom" and is_visual_only_geom(element)
        }
        # Rebuild the child list rather than inserting by index: a twin belongs directly after its
        # original, which is what appending both in order says, with no offset bookkeeping to get wrong.
        children: list[ET.Element] = []
        for element in parent:
            children.append(element)
            if element.tag != "geom" or is_visual_only_geom(element):
                continue
            group = int(element.get("group", "0"))
            if group in (0, 1, 2) and geom_render_signature(element) in visual_signatures:
                element.set("group", "3")  # an exact visual counterpart exists; just hide this one
                modified = True
                continue
            if require_material and element.get("material") is None:
                continue
            if groups is not None and group not in groups:
                continue

            name = element.get("name")
            if not name:
                unnamed += 1
                name = f"geom{unnamed}"
            children.append(_visual_twin(element, f"{name}_vis", massless=massless_twin))
            element.set("group", "3")
            modified = True
        parent[:] = children
    return modified


def _mean_texture_rgb(path: Path) -> tuple[float, float, float]:
    import numpy as np
    from PIL import Image

    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float64).reshape(-1, 3).mean(axis=0) / 255.0
    return (float(rgb[0]), float(rgb[1]), float(rgb[2]))


def flatten_cube_textures(root: ET.Element, source_dir: Path) -> bool:
    """Replace ``cube`` / ``skybox`` textures by their mean colour on the materials that use them.

    Genesis maps only ``2d`` textures onto texcoord-less geoms (MuJoCo's object-linear projection);
    a cube texture gives garbage UVs — the speckled grey robosuite's brass nuts came out as. The
    material keeps its alpha and drops the texture reference; the texture element is removed.
    """
    asset = root.find("asset")
    if asset is None:
        return False
    _, texture_dir = _compiler_dirs(root)
    flattened: dict[str, tuple[float, float, float]] = {}
    for texture in list(asset.findall("texture")):
        if texture.get("type", "cube") not in ("cube", "skybox") or not texture.get("file"):
            continue
        name = texture.get("name")
        if not name:
            continue
        file_attribute = texture.get("file")
        path = Path(file_attribute) if os.path.isabs(file_attribute) else source_dir / texture_dir / file_attribute
        flattened[name] = _mean_texture_rgb(path)
        asset.remove(texture)
    if not flattened:
        return False
    for material in asset.findall("material"):
        rgb = flattened.get(material.get("texture", ""))
        if rgb is None:
            continue
        alpha = material.get("rgba", "1 1 1 1").split()[3]
        material.set("rgba", f"{rgb[0]:.4f} {rgb[1]:.4f} {rgb[2]:.4f} {alpha}")
        del material.attrib["texture"]
        material.attrib.pop("texrepeat", None)
        material.attrib.pop("texuniform", None)
    return True
