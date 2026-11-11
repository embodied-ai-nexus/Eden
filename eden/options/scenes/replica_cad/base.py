"""ReplicaCAD scene options built from upstream scene JSON."""

from __future__ import annotations

import numpy as np
from genesis.utils.geom import transform_quat_by_quat

from eden.options.entities import SceneOptions
from eden.options.objects.replica_cad import (
    REPLICA_CAD_STAGE_REGISTRY,
    REPLICA_CAD_OBJECT_REGISTRY,
    ReplicaCADObjectOptions,
)
from eden.utils.configs import load_json
from eden.options.materials import (
    KinematicMaterialOptions,
)


# Positions: (x, y, z)_hab → (x, -z, y)_eden  (same as _hab_to_eden_pos)
# Rotations: q_eden = _Q_M ⊗ q_hab ⊗ _Q_M_CONJ
_Q_M = np.array([np.sqrt(2) / 2, np.sqrt(2) / 2, 0.0, 0.0])  # (w, x, y, z)
_Q_M_CONJ = np.array([np.sqrt(2) / 2, -np.sqrt(2) / 2, 0.0, 0.0])


def _build_scene_dict_from_json(json_path: str) -> dict[str, ReplicaCADObjectOptions]:
    scene_data = load_json(json_path)
    scene_dict: dict[str, ReplicaCADObjectOptions] = {}

    def _hab_to_eden_pos(translation: list) -> tuple:
        """Convert Habitat Y-up to Eden Z-up: (x, y, z) -> (x, -z, y)."""
        x, y, z = translation
        return (x, -z, y)

    def _hab_to_eden_quat_zup(rotation: list) -> tuple:
        """Convert a Habitat Y-up rotation to Eden Z-up for Z-up-native assets.

        Most Replica-CAD GLBs are stored with Z-up vertices (non-standard convention),
        so Genesis loads them as-is (`file_meshes_are_zup=True`, no auto-conversion).
        These assets have identity mesh-alignment (up=(0,0,1), front=(1,0,0)).

        With `KinematicEntity.set_quat(relative=True)`, `default_root_quat` is composed
        on top of the alignment quaternion.  For identity alignment the result equals the
        desired absolute world quaternion, so we just need the frame-converted rotation:

            q_eden = q_M ⊗ q_hab ⊗ q_M_conj
        """
        q_hab = np.array(rotation)
        return tuple(transform_quat_by_quat(_Q_M_CONJ, transform_quat_by_quat(q_hab, _Q_M)).tolist())

    name_counts: dict[str, int] = {}

    def _unique_name(base_name: str) -> str:
        if base_name not in name_counts:
            name_counts[base_name] = 0
            return base_name
        name_counts[base_name] += 1
        return f"{base_name}_{name_counts[base_name]}"

    # Stage (Z-up native, identity alignment)
    stage_info = scene_data.get("stage_instance", {})
    if stage_info:
        template = stage_info["template_name"]
        stage_key = template.split("/")[-1]
        translation = stage_info.get("translation", [0, 0, 0])
        rotation = stage_info.get("rotation", [1, 0, 0, 0])

        stage_cls = REPLICA_CAD_STAGE_REGISTRY.get(stage_key)
        kwargs: dict = dict(
            is_fixed_base=True,
            default_root_pos=_hab_to_eden_pos(translation),
            default_root_quat=_hab_to_eden_quat_zup(rotation),
        )
        scene_dict["stage"] = stage_cls(**kwargs)

    # Object instances
    for obj in scene_data.get("object_instances", []):
        template = obj["template_name"]
        name = template.split("/")[-1]
        translation = obj.get("translation", [0, 0, 0])
        rotation = obj.get("rotation", [1, 0, 0, 0])

        obj_cls = REPLICA_CAD_OBJECT_REGISTRY.get(name)
        kwargs = dict(
            is_fixed_base=True,
            default_root_pos=_hab_to_eden_pos(translation),
            default_root_quat=_hab_to_eden_quat_zup(rotation),
            material=KinematicMaterialOptions(),
        )
        scene_dict[_unique_name(name)] = obj_cls(**kwargs)

    # Articulated object instances
    for obj in scene_data.get("articulated_object_instances", []):
        template = obj["template_name"]
        name = template.split("/")[-1]
        translation = obj.get("translation", [0, 0, 0])
        rotation = obj.get("rotation", [1, 0, 0, 0])
        scale = obj.get("uniform_scale", 1.0)

        obj_cls = REPLICA_CAD_OBJECT_REGISTRY.get(name)
        kwargs = dict(
            is_fixed_base=True,
            default_root_pos=_hab_to_eden_pos(translation),
            default_root_quat=_hab_to_eden_quat_zup(rotation),
            scale=scale,
            material=KinematicMaterialOptions(),
        )
        scene_dict[_unique_name(name)] = obj_cls(**kwargs)

    return scene_dict


def ReplicaCADSceneOptions(config_path: str, **kwargs) -> SceneOptions:
    """ReplicaCAD scene assembled from an upstream scene JSON.

    Returns a plain :class:`SceneOptions` with the stage and object instances
    parsed from ``config_path`` added as named children, so the config
    round-trips through validated load.
    """
    return SceneOptions(
        **_build_scene_dict_from_json(config_path),
        **kwargs,
    )
