"""Materialize Genesis-ready MJCF descriptors for LIBERO assets.

The generic pipeline (runtime directory, caching, path resolution, the wrapper / free-joint /
visual-twin transforms) lives in :mod:`eden.options.objects.mjcf_transforms`; this module keeps
what is LIBERO's: the two articulated assets whose compiler block only
becomes valid after robosuite's programmatic edits, and the procedural workspace table.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from functools import partial
from pathlib import Path

from eden.options.objects.mjcf_transforms import (
    XmlTransform,
    add_freejoint_to_root_body,
    clone_collision_geoms_for_rendering,
    extract_wrapped_object_body,
    materialize_xml,
)

#: LIBERO's contract for the shared visual-twin transform, spelled out rather than left to the
#: defaults: only a geom that declares a material and sits in MuJoCo's group 1 or 2 gets a twin,
#: and the twin keeps the original's density. robosuite's objects need
#: ``groups=None, require_material=False`` instead, so the difference is worth naming here.
#:
#: ``massless_twin=False`` is the one that is *not* a preference. A twin inheriting the original's
#: density doubles the body's compiled mass (MuJoCo's ``inertiafromgeom="auto"`` sums both), which
#: is what every LIBERO descriptor compiles to — the akita_black_bowl weighs 2.424 kg where its
#: geometry says 1.212 kg. Correcting it changes the dynamics of the 130-task benchmark and
#: invalidates policies trained against it, so it is pinned here. robosuite takes the default. See
#: :func:`~eden.options.objects.mjcf_transforms._visual_twin`.
_clone_for_rendering = partial(
    clone_collision_geoms_for_rendering, groups=(1, 2), require_material=True, massless_twin=False
)

# These assets declare an inertia group that only exists after robosuite's programmatic additions.
_ASSETS_NEEDING_SANITIZING = ("articulated_objects/flat_stove.xml", "articulated_objects/microwave.xml")


def _materialize_xml(
    file_path: str,
    *,
    transforms: tuple[XmlTransform, ...],
    registry: str | None,
    dataset: str | None,
    local_dir: str,
) -> str:
    """Apply one transform pipeline and write one process-local runtime file.

    ``local_dir`` is the caller's ``LiberoObjectOptions.resolve_local_dir()``.
    """
    if file_path in _ASSETS_NEEDING_SANITIZING:
        transforms = (_sanitize_compiler_and_textures, *transforms)
    return materialize_xml(
        file_path,
        transforms=transforms,
        registry=registry,
        dataset=dataset,
        local_dir=local_dir,
    )


def _prepare_object_xml(
    file_path: str,
    *,
    is_fixed_base: bool,
    registry: str | None,
    dataset: str | None,
    local_dir: str,
) -> str:
    """Materialize one LIBERO object descriptor for Genesis."""
    transforms: tuple[XmlTransform, ...] = (extract_wrapped_object_body, _clone_for_rendering)
    if not is_fixed_base:
        transforms += (add_freejoint_to_root_body,)
    return _materialize_xml(
        file_path,
        transforms=transforms,
        registry=registry,
        dataset=dataset,
        local_dir=local_dir,
    )


def _prepare_workspace_table_xml(
    file_path: str,
    *,
    table_body_name: str,
    table_top_pos_local: tuple[float, float, float],
    table_full_size: tuple[float, float, float],
    table_friction: tuple[float, float, float],
    registry: str | None,
    dataset: str | None,
    local_dir: str,
) -> str:
    """Materialize the procedural table authored by LIBERO's table arenas."""
    return _materialize_xml(
        file_path,
        transforms=(
            partial(
                _configure_workspace_table,
                table_body_name=table_body_name,
                table_top_pos_local=table_top_pos_local,
                table_full_size=table_full_size,
                table_friction=table_friction,
            ),
            _clone_for_rendering,
        ),
        registry=registry,
        dataset=dataset,
        local_dir=local_dir,
    )


def _mjcf_vector(values: tuple[float, ...]) -> str:
    return " ".join(f"{value:.12g}" for value in values)


def _configure_workspace_table(
    root: ET.Element,
    _source_dir: Path,
    *,
    table_body_name: str,
    table_top_pos_local: tuple[float, float, float],
    table_full_size: tuple[float, float, float],
    table_friction: tuple[float, float, float],
) -> bool:
    if any(value <= 0.0 for value in table_full_size):
        raise ValueError(f"LIBERO table size must be positive, got {table_full_size!r}")

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("LIBERO procedural workspace scene is missing <worldbody>")
    body = worldbody.find(f"./body[@name='{table_body_name}']")
    if body is None:
        raise ValueError(f"LIBERO procedural workspace scene has no root body named {table_body_name!r}")

    required_children = {
        "collision": body.find("./geom[@name='table_collision']"),
        "visual": body.find("./geom[@name='table_visual']"),
        "top": body.find("./site[@name='table_top']"),
        **{f"leg_{index}": body.find(f"./geom[@name='table_leg{index}_visual']") for index in range(1, 5)},
    }
    missing = [name for name, element in required_children.items() if element is None]
    if missing:
        raise ValueError(f"LIBERO procedural workspace body {table_body_name!r} is missing elements {missing!r}")

    half_size = tuple(value / 2.0 for value in table_full_size)
    body.set(
        "pos",
        _mjcf_vector(
            (
                table_top_pos_local[0],
                table_top_pos_local[1],
                table_top_pos_local[2] - half_size[2],
            )
        ),
    )

    collision = required_children["collision"]
    visual = required_children["visual"]
    top = required_children["top"]
    assert collision is not None and visual is not None and top is not None
    collision.set("size", _mjcf_vector(half_size))
    collision.set("friction", _mjcf_vector(table_friction))
    visual.set("size", _mjcf_vector(half_size))
    top.set("pos", _mjcf_vector((0.0, 0.0, half_size[2])))

    leg_half_height = (table_top_pos_local[2] - half_size[2]) / 2.0
    if leg_half_height <= 0.0:
        raise ValueError(
            "LIBERO table top position must place positive-height legs, "
            f"got table_top_pos_local={table_top_pos_local!r}, table_full_size={table_full_size!r}"
        )
    signs = ((1.0, 1.0), (-1.0, 1.0), (-1.0, -1.0), (1.0, -1.0))
    for index, (x_sign, y_sign) in enumerate(signs, start=1):
        x = x_sign * (half_size[0] - 0.1) if half_size[0] > 0.2 else 0.0
        y = y_sign * (half_size[1] - 0.1) if half_size[1] > 0.2 else 0.0
        leg = required_children[f"leg_{index}"]
        assert leg is not None
        leg.set("pos", _mjcf_vector((x, y, -leg_half_height)))
        leg.set("size", _mjcf_vector((0.025, leg_half_height)))
    return True


def _prepare_scene_xml(
    file_path: str,
    *,
    registry: str | None,
    dataset: str | None,
    local_dir: str,
) -> str:
    """Materialize one fixed LIBERO workspace descriptor for Genesis."""
    return _materialize_xml(
        file_path,
        transforms=(_clone_for_rendering,),
        registry=registry,
        dataset=dataset,
        local_dir=local_dir,
    )


def _sanitize_compiler_and_textures(root: ET.Element, _source_dir: Path) -> bool:
    """Remove the invalid inertia range and preserve latest-wins textures."""
    compiler = root.find("compiler")
    if compiler is not None and compiler.get("inertiagrouprange") is not None:
        del compiler.attrib["inertiagrouprange"]

    asset = root.find("asset")
    if asset is None:
        return True

    texture_name_map: dict[str, str] = {}
    duplicate_counters: dict[str, int] = {}
    for element in asset:
        if element.tag == "texture":
            original_name = element.get("name")
            if not original_name:
                continue
            if original_name in texture_name_map:
                duplicate_counters[original_name] = duplicate_counters.get(original_name, 0) + 1
                resolved_name = f"{original_name}__dup_{duplicate_counters[original_name]}"
                element.set("name", resolved_name)
            else:
                resolved_name = original_name
            texture_name_map[original_name] = resolved_name
        elif element.tag == "material":
            texture_name = element.get("texture")
            if texture_name and texture_name in texture_name_map:
                element.set("texture", texture_name_map[texture_name])
    return True
