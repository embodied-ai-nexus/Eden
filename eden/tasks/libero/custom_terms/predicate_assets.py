"""Predicate-relevant metadata parsed from LIBERO MJCF assets for metric terms."""

from __future__ import annotations

import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from types import MappingProxyType

import numpy as np
from genesis.utils.geom import transform_pos_quat_by_trans_quat

from eden.options.objects.libero import LIBERO_OBJECT_REGISTRY


@dataclass(frozen=True)
class _SiteMetadata:
    """Source-MJCF metadata expressed in the nearest dynamic body frame."""

    anchor_body_name: str | None
    pos_local: tuple[float, float, float]
    quat_local: tuple[float, float, float, float]
    size: tuple[float, ...]
    site_type: str
    joint_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class _AssetPredicateMetadata:
    """Predicate-relevant metadata parsed from one LIBERO asset descriptor."""

    sites: Mapping[str, _SiteMetadata]
    joint_names: tuple[str, ...] = ()


def _float_tuple(text: str | None, default: tuple[float, ...]) -> tuple[float, ...]:
    return default if text is None else tuple(float(value) for value in text.split())


def _normalize_quat_tuple(
    quat: tuple[float, float, float, float],
    *,
    label: str,
) -> tuple[float, float, float, float]:
    norm = sum(value * value for value in quat) ** 0.5
    if norm == 0.0:
        raise ValueError(f"Zero-length quaternion in {label}")
    return tuple(value / norm for value in quat)


def _compose_pose_tuple(
    parent_pos: tuple[float, float, float],
    parent_quat: tuple[float, float, float, float],
    pos_local: tuple[float, float, float],
    quat_local: tuple[float, float, float, float],
) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
    position, quat = transform_pos_quat_by_trans_quat(
        np.asarray(pos_local),
        np.asarray(quat_local),
        np.asarray(parent_pos),
        np.asarray(parent_quat),
    )
    position_tuple = tuple(float(value) for value in position)
    quat_tuple = tuple(float(value) for value in quat)
    return position_tuple, _normalize_quat_tuple(quat_tuple, label="composed MJCF pose")


def _element_quat(element: ET.Element, *, label: str) -> tuple[float, float, float, float]:
    unsupported = [name for name in ("axisangle", "euler", "xyaxes", "zaxis") if element.get(name) is not None]
    if unsupported:
        raise NotImplementedError(f"{label} uses unsupported MJCF orientation attribute {unsupported[0]!r}")
    quat = _float_tuple(element.get("quat"), (1.0, 0.0, 0.0, 0.0))
    if len(quat) != 4:
        raise ValueError(f"Invalid quaternion metadata for {label}: {quat}")
    return _normalize_quat_tuple(quat, label=label)


def _parse_asset_body(
    body: ET.Element,
    sites: dict[str, _SiteMetadata],
    all_joint_names: list[str],
    *,
    anchor_body_name: str | None,
    anchor_joint_names: tuple[str, ...],
    fixed_position: tuple[float, float, float],
    fixed_quat: tuple[float, float, float, float],
    is_entity_root: bool,
    is_top_level: bool,
) -> None:
    body_name = body.get("name")
    joint_names = tuple(joint.get("name") for joint in body.findall("./joint") if joint.get("name") is not None)
    all_joint_names.extend(joint_names)
    starts_dynamic_frame = is_top_level or bool(body.findall("./joint")) or body.find("./freejoint") is not None
    if is_entity_root:
        current_anchor = None
        current_joint_names = joint_names
        current_position = (0.0, 0.0, 0.0)
        current_quat = (1.0, 0.0, 0.0, 0.0)
    elif starts_dynamic_frame:
        if body_name is None:
            raise ValueError("A non-root LIBERO MJCF dynamic body containing predicate sites must have a name")
        current_anchor = body_name
        current_joint_names = joint_names
        current_position = (0.0, 0.0, 0.0)
        current_quat = (1.0, 0.0, 0.0, 0.0)
    else:
        body_position = _float_tuple(body.get("pos"), (0.0, 0.0, 0.0))
        if len(body_position) != 3:
            raise ValueError(f"Invalid pose metadata for LIBERO body {body_name!r}: pos={body_position}")
        body_quat = _element_quat(body, label=f"LIBERO body {body_name!r}")
        current_anchor = anchor_body_name
        current_joint_names = anchor_joint_names
        current_position, current_quat = _compose_pose_tuple(
            fixed_position,
            fixed_quat,
            body_position,
            body_quat,
        )

    for site in body.findall("./site"):
        name = site.get("name")
        if name is None:
            continue
        if name in sites:
            raise ValueError(f"LIBERO asset contains duplicate site name {name!r}")
        position = _float_tuple(site.get("pos"), (0.0, 0.0, 0.0))
        if len(position) != 3:
            raise ValueError(f"Invalid position metadata for LIBERO site {name!r}: {position}")
        quat = _element_quat(site, label=f"LIBERO site {name!r}")
        position, quat = _compose_pose_tuple(current_position, current_quat, position, quat)
        sites[name] = _SiteMetadata(
            anchor_body_name=current_anchor,
            pos_local=position,
            quat_local=quat,
            size=_float_tuple(site.get("size"), ()),
            site_type=site.get("type", "sphere"),
            joint_names=current_joint_names,
        )

    for child in body.findall("./body"):
        _parse_asset_body(
            child,
            sites,
            all_joint_names,
            anchor_body_name=current_anchor,
            anchor_joint_names=current_joint_names,
            fixed_position=current_position,
            fixed_quat=current_quat,
            is_entity_root=False,
            is_top_level=False,
        )


@cache
def _load_asset_predicate_metadata_cached(
    path_string: str,
    _mtime_ns: int,
    _size_bytes: int,
) -> _AssetPredicateMetadata:
    path = Path(path_string)
    root = ET.parse(path).getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"LIBERO descriptor has no worldbody: {path}")
    sites: dict[str, _SiteMetadata] = {}
    all_joint_names: list[str] = []
    top_level_bodies = worldbody.findall("./body")
    for body in top_level_bodies:
        _parse_asset_body(
            body,
            sites,
            all_joint_names,
            anchor_body_name=None,
            anchor_joint_names=(),
            fixed_position=(0.0, 0.0, 0.0),
            fixed_quat=(1.0, 0.0, 0.0, 0.0),
            is_entity_root=len(top_level_bodies) == 1,
            is_top_level=True,
        )
    return _AssetPredicateMetadata(sites=MappingProxyType(sites), joint_names=tuple(all_joint_names))


def _load_asset_predicate_metadata(category_name: str) -> _AssetPredicateMetadata:
    """Load and cache predicate metadata for the registered LIBERO asset of ``category_name``."""
    try:
        options_cls = LIBERO_OBJECT_REGISTRY.get(category_name)
    except KeyError as exc:
        raise KeyError(f"No Eden LIBERO asset is registered for category {category_name!r}") from exc
    options = options_cls()
    runtime_file = options.resolve_runtime_file()
    path = Path(runtime_file)
    if not path.is_file():
        path = Path(options.fetch_asset(runtime_file))
    if not path.is_file():
        raise FileNotFoundError(f"Resolved LIBERO descriptor does not exist: {path}")
    stat = path.stat()
    return _load_asset_predicate_metadata_cached(str(path), stat.st_mtime_ns, stat.st_size)
