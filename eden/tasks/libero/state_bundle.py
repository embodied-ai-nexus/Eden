"""Simulator-independent LIBERO fixed-state bundle loader."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import numpy as np
from pydantic import BaseModel, Field, model_validator

_SemanticName = Annotated[str, Field(pattern=r"^[A-Za-z0-9_]+$")]


class _EntityDescriptor(BaseModel):
    semantic_name: _SemanticName
    root_link_name: str | None = Field(default=None, min_length=1)
    joint_names: list[_SemanticName] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_joint_names(self):
        if len(self.joint_names) != len(set(self.joint_names)):
            raise ValueError(f"Duplicate semantic joint name on entity {self.semantic_name!r}")
        return self


class _StateBundleManifest(BaseModel):
    task_name: str = Field(min_length=1)
    init_ids: list[int] = Field(min_length=1)
    entities: list[_EntityDescriptor]

    @model_validator(mode="after")
    def _validate_manifest(self):
        if len(set(self.init_ids)) != len(self.init_ids):
            raise ValueError("init_ids must be unique")
        if any(init_id < 0 for init_id in self.init_ids):
            raise ValueError("init_ids must be non-negative")
        names = [entity.semantic_name for entity in self.entities]
        if len(names) != len(set(names)):
            raise ValueError("Entity semantic_name values must be unique")
        return self


class LiberoStateBundle:
    """Validated manifest plus numeric arrays loaded with ``allow_pickle=False``."""

    def __init__(self, manifest: _StateBundleManifest, arrays: dict[str, np.ndarray]):
        self.manifest = manifest
        self.arrays = arrays
        self._row_by_init_id = {init_id: row for row, init_id in enumerate(manifest.init_ids)}

    def rows_for_init_ids(self, init_ids: list[int]) -> np.ndarray:
        try:
            return np.asarray([self._row_by_init_id[init_id] for init_id in init_ids], dtype=np.int64)
        except KeyError as exc:
            raise KeyError(f"Fixed-state bundle has no init_id={exc.args[0]}") from exc


def load_state_bundle(
    bundle_path: str | Path,
    *,
    expected_task_name: str | None = None,
) -> LiberoStateBundle:
    """Load and fully validate one ``task.json`` plus ``states.npz`` bundle."""
    manifest_path = Path(bundle_path)
    if manifest_path.is_dir():
        manifest_path = manifest_path / "task.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"LIBERO state manifest not found: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8") as stream:
        raw = json.load(stream)
    manifest = _StateBundleManifest.model_validate(raw)

    if expected_task_name is not None and manifest.task_name != expected_task_name:
        raise ValueError(f"State bundle task mismatch: expected {expected_task_name!r}, got {manifest.task_name!r}")
    states_path = manifest_path.parent / "states.npz"
    if not states_path.is_file():
        raise FileNotFoundError(f"LIBERO numeric state file not found: {states_path}")

    state_count = len(manifest.init_ids)
    expected: dict[str, tuple[np.dtype, tuple[int, ...]]] = {}
    for entity in manifest.entities:
        prefix = f"entity/{entity.semantic_name}"
        expected.update(
            {
                f"{prefix}/root_pos_world": (np.dtype(np.float64), (state_count, 3)),
                f"{prefix}/root_quat_wxyz_world": (np.dtype(np.float64), (state_count, 4)),
                f"{prefix}/root_lin_vel_world": (np.dtype(np.float64), (state_count, 3)),
                f"{prefix}/root_ang_vel_body": (np.dtype(np.float64), (state_count, 3)),
            }
        )
        for joint_name in entity.joint_names:
            joint_prefix = f"{prefix}/joint/{joint_name}"
            expected[f"{joint_prefix}/qpos"] = (np.dtype(np.float64), (state_count, 1))
            expected[f"{joint_prefix}/qvel"] = (np.dtype(np.float64), (state_count, 1))
    arrays: dict[str, np.ndarray] = {}
    with np.load(states_path, allow_pickle=False) as archive:
        actual_keys = set(archive.files)
        expected_keys = set(expected)
        if actual_keys != expected_keys:
            raise ValueError(
                f"State array keys mismatch: missing={sorted(expected_keys - actual_keys)}, "
                f"unexpected={sorted(actual_keys - expected_keys)}"
            )
        for key, (dtype, shape) in expected.items():
            value = np.asarray(archive[key])
            if value.dtype != dtype:
                raise TypeError(f"Array {key!r} must have dtype {dtype}, got {value.dtype}")
            if value.shape != shape:
                raise ValueError(f"Array {key!r} must have shape {shape}, got {value.shape}")
            if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
                raise ValueError(f"Array {key!r} contains NaN or infinity")
            arrays[key] = value.copy()

    for entity in manifest.entities:
        key = f"entity/{entity.semantic_name}/root_quat_wxyz_world"
        norms = np.linalg.norm(arrays[key], axis=1)
        if not np.all(np.abs(norms - 1.0) <= 1e-6):
            raise ValueError(f"Array {key!r} contains zero or non-unit quaternions")

    return LiberoStateBundle(manifest, arrays)


__all__ = [
    "LiberoStateBundle",
    "load_state_bundle",
]
