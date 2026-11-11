"""ARCTIC dataset object options (articulated interaction objects)."""

from __future__ import annotations

from typing import ClassVar

from genesis.typing import UnitVec4FType, Vec3FType

from eden.options.entities import EntityOptions
from eden.utils.registry import Registry

# NOTE: a registry used for datasets/arctic to load the corresponding object in reference motion
ARCTIC_OBJECTS_REGISTRY = Registry("ARCTIC_OBJECTS")


class ArcticObjectOptions(EntityOptions):
    """Arctic object options."""

    up: tuple[int, int, int] = (0, 0, 1)
    front: tuple[int, int, int] = (1, 0, 0)
    is_articulated: bool = True

    dofs_name: list[str] = ["rotation"]
    default_dofs_kp: dict[str, float] = {"*": 1000.0}
    default_dofs_kd: dict[str, float] = {"*": 100.0}

    dataset: str = "eden_arctic"
    assets_subdir: ClassVar[str] = "arctic"


@ARCTIC_OBJECTS_REGISTRY.register(name="box")
class ArcticBox(ArcticObjectOptions):
    file: str = "box/box.urdf"

    # TODO: center the box at the origin
    default_root_pos: Vec3FType = (0.0, 0.0, 0.0723354468809288)
    default_root_quat: UnitVec4FType = (
        -0.6413,
        0.2875,
        0.6467,
        -0.2964,
    )


@ARCTIC_OBJECTS_REGISTRY.register(name="ketchup")
class ArcticKetchup(ArcticObjectOptions):
    file: str = "ketchup/ketchup.urdf"

    # TODO: set the default root pos and quat


@ARCTIC_OBJECTS_REGISTRY.register(name="mixer")
class ArcticMixer(ArcticObjectOptions):
    file: str = "mixer/mixer.urdf"

    # TODO: set the default root pos and quat
    # default_root_pos: tuple[float, float, float] = (0.0346, -0.1051,  1.0767)
    # default_root_quat: tuple[float, float, float, float] = (-0.5380, -0.3588,  0.6063, -0.4628)


@ARCTIC_OBJECTS_REGISTRY.register(name="notebook")
class ArcticNotebook(ArcticObjectOptions):
    file: str = "notebook/notebook.urdf"

    # TODO: set the default root pos and quat


@ARCTIC_OBJECTS_REGISTRY.register(name="waffleiron")
class ArcticWaffleiron(ArcticObjectOptions):
    file: str = "waffleiron/waffleiron.urdf"

    # TODO: set the default root pos and quat
