"""Every registered robot: the link names it declares exist in its asset, and its feet survive link merging.

A static lint over ``ee_links_name`` / ``feet_links_name`` (both Franka classes shipped a copy-pasted
``"ee_link"`` neither MJCF defines). The name lint does not model Genesis' fixed-link merge, and it only reads
assets already on disk -- ``eden/assets`` or the shared Hugging Face cache -- skipping the rest visibly
rather than reaching the network.
"""

from __future__ import annotations

import glob
import os
from xml.etree import ElementTree

import pytest
from huggingface_hub.constants import HF_HUB_CACHE

from eden.options import robots  # noqa: F401 -- populates ROBOT_REGISTRY
from eden.options.catalog import ROBOT_REGISTRY
from eden.options.robots.unitree_go2 import Go2, Go2W
from eden.utils.misc import get_assets_dir

_LINK_NAME_FIELDS = ("ee_links_name", "feet_links_name")
_DECLARATIONS = sorted(
    (
        (f"{key}.{field}", cls, field)
        for key, cls in ROBOT_REGISTRY
        for field in _LINK_NAME_FIELDS
        if getattr(cls, field)
    ),
    key=lambda item: item[0],
)


def _default(cls, name: str):
    field = cls.model_fields.get(name)
    return field.default if field is not None else None


def _find_under(root: str, file: str) -> str | None:
    """The local-lookup half of ``get_asset_path``: ``file`` is the repo path below ``root``."""
    candidate = os.path.join(root, file)
    return candidate if os.path.isfile(candidate) else None


def _cached_asset_path(cls) -> str | None:
    """Where ``cls``'s asset already is on disk, or ``None``. Never downloads."""
    file = _default(cls, "file")
    if not isinstance(file, str) or not file:
        return None
    if os.path.isfile(file):
        return file
    roots = [_default(cls, "local_dir") or get_assets_dir()]
    registry, dataset = _default(cls, "registry"), _default(cls, "dataset")
    if registry and dataset:
        # The hub keeps its own copy, shared across worktrees; a fresh worktree's eden/assets is empty.
        roots += glob.glob(os.path.join(HF_HUB_CACHE, f"datasets--{registry}--{dataset}", "snapshots", "*"))
    return next((found for root in roots if (found := _find_under(root, file)) is not None), None)


def _declared_body_names(path: str) -> set[str]:
    """Every MJCF ``<body>`` / URDF ``<link>`` name in ``path`` and the files it ``<include>``s."""
    names: set[str] = set()
    seen: set[str] = set()
    pending = [path]
    while pending:
        current = os.path.normpath(pending.pop())
        if current in seen or not os.path.isfile(current):
            continue
        seen.add(current)
        root = ElementTree.parse(current).getroot()
        for tag in ("body", "link"):
            names.update(el.get("name") for el in root.iter(tag) if el.get("name"))
        pending += [
            os.path.join(os.path.dirname(current), i.get("file")) for i in root.iter("include") if i.get("file")
        ]
    return names


@pytest.mark.parametrize("field", _LINK_NAME_FIELDS)
def test_the_lint_has_robots_to_check(field):
    assert any(f == field for _, _, f in _DECLARATIONS), f"no registered robot declares `{field}`"


@pytest.mark.parametrize("key, cls, field", _DECLARATIONS, ids=[key for key, _, _ in _DECLARATIONS])
def test_declared_link_names_exist_in_the_asset(key, cls, field):
    path = _cached_asset_path(cls)
    if path is None:
        pytest.skip(f"{key}: asset {_default(cls, 'file')!r} is not cached locally; build this robot once to cache it")
    declared = _declared_body_names(path)
    missing = [name for name in getattr(cls, field) if name not in declared]
    assert not missing, f"{cls.__name__}.{field} names {missing}, not in {os.path.basename(path)}: {sorted(declared)}"


# ── feet_links_name: what the deploy-time velocity estimator stands on ────────────────────────────

_FOOTED = sorted(((key, cls) for key, cls in ROBOT_REGISTRY if cls.feet_links_name), key=lambda item: item[0])


def test_the_catalog_has_footed_robots():
    assert _FOOTED


@pytest.mark.parametrize("key, cls", _FOOTED, ids=[key for key, _ in _FOOTED])
def test_every_declared_foot_survives_the_fixed_link_merge(key, cls):
    """A merged-away foot surfaces only as ``get_link`` failing inside the estimator at deploy time."""
    assert [name for name in cls.feet_links_name if name not in cls().links_to_keep] == []


def test_the_wheeled_go2_declares_no_feet_though_it_keeps_the_links():
    """Its ``*_foot`` links are driven wheels: a rolling contact breaks the leg-odometry stance assumption."""
    feet = ("FL_foot", "RL_foot", "FR_foot", "RR_foot")
    assert (tuple(Go2W.feet_links_name), tuple(Go2.feet_links_name)) == ((), feet)
    assert set(feet) <= set(Go2W().links_to_keep)
