"""robosuite asset options: the Rethink mount and the manipulation-benchmark objects.

The object files are robosuite's own (``robosuite/models/assets/objects``, MIT) plus MJCF
re-authored from its procedural composites (``pot_with_handles``, ``hammer``, the ToolHang parts,
the bins and lid), hosted in the ``Kashu7100/eden_robosuite`` dataset and mirrored locally under
``robosuite/`` in the asset directory (:func:`eden.utils.misc.get_assets_dir`). They are stored
verbatim; what Genesis needs is derived at load time by :meth:`RobosuiteObjectOptions.resolve_runtime_file` (see
:mod:`eden.options.objects.mjcf_transforms` for why each step exists):

1. angles are read in radians, the unit robosuite's world model compiles its objects in;
   robosuite's files declare no ``<compiler>`` of their own, and MuJoCo's default is degrees;
2. robosuite's ``<body><body name="object">`` wrapper is unwrapped (hand-authored composites pass
   through unchanged);
3. every collision geom gets a visual-only twin, so objects render with their material instead of
   Genesis' grey collision surface;
4. ``cube`` textures, which Genesis cannot map, become the material's mean colour;
5. the root free joint follows ``is_fixed_base`` — added for a free object (unless the root body
   already declares a joint of its own), removed for a fixture.

Objects are registered in :data:`ROBOSUITE_OBJECT_REGISTRY` under robosuite's names (``milk``,
``square_nut``, ``door``, ...).
"""

from __future__ import annotations

from functools import partial
from typing import ClassVar

from eden.options.entities import EntityOptions, MetadataOptions
from eden.options.objects.mjcf_transforms import (
    XmlTransform,
    add_freejoint_to_root_body,
    clone_collision_geoms_for_rendering,
    declare_radians,
    extract_wrapped_object_body,
    flatten_cube_textures,
    materialize_xml,
    remove_freejoint_from_root_body,
)
from eden.utils.registry import Registry

ROBOSUITE_REGISTRY = "Kashu7100"
ROBOSUITE_DATASET = "eden_robosuite"


class RethinkMount(EntityOptions):
    """Rethink Robotics pedestal robosuite mounts every arm on (also LIBERO's mounted families).

    The root body sits 0.01 m above the arm mounting surface, with the pedestal's feet ~0.91 m below
    it, so a mount at ``default_root_pos`` ``(x, y, 0.922)`` stands on the floor with its top at
    0.912 m — where robosuite puts ``link0``, and where LIBERO welds the arm (0.01 m below the root).
    """

    registry: str = "Kashu7100"
    dataset: str = "eden_assets"
    file: str = "mounts/rethink_mount/rethink_mount.xml"
    default_root_pos: tuple[float, float, float] = (0.0, 0.0, 0.922)
    default_root_quat: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    is_fixed_base: bool = True
    is_articulated: bool = False
    is_actuated: bool = False
    is_support_enabled: bool = False
    links_to_keep: list[str] = ["base"]


def prepare_robosuite_object_xml(
    file_path: str,
    *,
    is_fixed_base: bool,
    registry: str | None,
    dataset: str | None,
    local_dir: str | None,
) -> str:
    """Materialize one robosuite object descriptor for Genesis (see the module docstring)."""
    transforms: tuple[XmlTransform, ...] = (
        declare_radians,
        partial(extract_wrapped_object_body, strict=False),
        # groups=None / require_material=False: robosuite's dual-purpose geoms sit in group 0 and
        # some carry a bare ``rgba`` rather than a material. massless_twin is spelled out because
        # LIBERO deliberately differs (it keeps the masses its descriptors have always compiled to),
        # and the two call sites should read as a pair.
        partial(clone_collision_geoms_for_rendering, groups=None, require_material=False, massless_twin=True),
        flatten_cube_textures,
        remove_freejoint_from_root_body if is_fixed_base else add_freejoint_to_root_body,
    )
    return materialize_xml(
        file_path,
        transforms=transforms,
        registry=registry,
        dataset=dataset,
        local_dir=local_dir,
    )


class RobosuiteObjectOptions(EntityOptions):
    """Base options for one file of the robosuite asset set (``objects/can.xml``, ...)."""

    registry: str = ROBOSUITE_REGISTRY
    dataset: str = ROBOSUITE_DATASET
    assets_subdir: ClassVar[str] = "robosuite"
    metadata: ClassVar[MetadataOptions] = MetadataOptions(benchmark="robosuite")
    mjcf_follows_is_fixed_base: ClassVar[bool] = True

    def resolve_runtime_file(self) -> str:
        """Derive the Genesis-ready MJCF without changing the serialized source file."""
        return prepare_robosuite_object_xml(
            self.file,
            is_fixed_base=self.is_fixed_base,
            registry=self.registry,
            dataset=self.dataset,
            local_dir=self.resolve_local_dir(),
        )


class RobosuiteFixtureOptions(RobosuiteObjectOptions):
    """A robosuite object pinned to the world (bins, the ToolHang stand, the door frame)."""

    is_fixed_base: bool = True


ROBOSUITE_OBJECT_REGISTRY = Registry("ROBOSUITE_OBJECT")


# --- mesh objects (robosuite/models/assets/objects) ------------------------------------------------


@ROBOSUITE_OBJECT_REGISTRY.register(name="milk")
class RobosuiteMilk(RobosuiteObjectOptions):
    file: str = "objects/milk.xml"


@ROBOSUITE_OBJECT_REGISTRY.register(name="bread")
class RobosuiteBread(RobosuiteObjectOptions):
    file: str = "objects/bread.xml"


@ROBOSUITE_OBJECT_REGISTRY.register(name="cereal")
class RobosuiteCereal(RobosuiteObjectOptions):
    file: str = "objects/cereal.xml"


@ROBOSUITE_OBJECT_REGISTRY.register(name="can")
class RobosuiteCan(RobosuiteObjectOptions):
    file: str = "objects/can.xml"


@ROBOSUITE_OBJECT_REGISTRY.register(name="bottle")
class RobosuiteBottle(RobosuiteObjectOptions):
    file: str = "objects/bottle.xml"


@ROBOSUITE_OBJECT_REGISTRY.register(name="lemon")
class RobosuiteLemon(RobosuiteObjectOptions):
    file: str = "objects/lemon.xml"


# --- primitive-geom objects (robosuite XMLs) -------------------------------------------------------


@ROBOSUITE_OBJECT_REGISTRY.register(name="square_nut")
class RobosuiteSquareNut(RobosuiteObjectOptions):
    file: str = "objects/square-nut.xml"


@ROBOSUITE_OBJECT_REGISTRY.register(name="round_nut")
class RobosuiteRoundNut(RobosuiteObjectOptions):
    file: str = "objects/round-nut.xml"


@ROBOSUITE_OBJECT_REGISTRY.register(name="plate_with_hole")
class RobosuitePlateWithHole(RobosuiteObjectOptions):
    """The 34 cm plate with a 10 cm square hole TwoArmPegInHole welds to one hand (free base, so it can be welded)."""

    file: str = "objects/plate-with-hole.xml"


@ROBOSUITE_OBJECT_REGISTRY.register(name="door")
class RobosuiteDoor(RobosuiteFixtureOptions):
    """``door_lock.xml``: frame, hinged panel and a sprung handle on its own ``latch_joint`` (robosuite ``use_latch=True``)."""

    file: str = "objects/door_lock.xml"
    is_articulated: bool = True
    dofs_name: list[str] = ["hinge", "latch_joint"]


@ROBOSUITE_OBJECT_REGISTRY.register(name="door_no_latch")
class RobosuiteDoorNoLatch(RobosuiteFixtureOptions):
    """``door.xml``: the same door with a rigid handle (robosuite ``use_latch=False``)."""

    file: str = "objects/door.xml"
    is_articulated: bool = True
    dofs_name: list[str] = ["hinge"]


# --- composites re-authored from robosuite's procedural objects ------------------------------------


@ROBOSUITE_OBJECT_REGISTRY.register(name="pick_place_bin1")
class RobosuitePickPlaceBin1(RobosuiteFixtureOptions):
    """PickPlace's source bin (``bins_arena.xml`` ``bin1``: slab + four walls, light wood)."""

    file: str = "objects/bin1.xml"


@ROBOSUITE_OBJECT_REGISTRY.register(name="pick_place_bin2")
class RobosuitePickPlaceBin2(RobosuiteFixtureOptions):
    """PickPlace's target bin (``bin2``: slab, four walls and the two quadrant dividers, dark wood)."""

    file: str = "objects/bin2.xml"


@ROBOSUITE_OBJECT_REGISTRY.register(name="pot_with_handles")
class RobosuitePotWithHandles(RobosuiteObjectOptions):
    """TwoArmLift's ``PotWithHandlesObject``: an open 14 cm box with a green (+y) and a blue (-y) handle."""

    file: str = "objects/pot_with_handles.xml"


@ROBOSUITE_OBJECT_REGISTRY.register(name="hammer")
class RobosuiteHammer(RobosuiteObjectOptions):
    """TwoArmHandover's ``HammerObject`` at the midpoint of robosuite's size ranges."""

    file: str = "objects/hammer.xml"


@ROBOSUITE_OBJECT_REGISTRY.register(name="hammer_payload")
class RobosuiteHammerPayload(RobosuiteObjectOptions):
    """TwoArmTransport's payload hammer (``handle_radius 0.015``, ``handle_length 0.20``)."""

    file: str = "objects/hammer_payload.xml"


@ROBOSUITE_OBJECT_REGISTRY.register(name="transport_bin")
class RobosuiteTransportBin(RobosuiteFixtureOptions):
    """TwoArmTransport's ``Bin`` (30 cm, transparent walls); the slab is its own ``base`` link for contact checks."""

    file: str = "objects/bin.xml"


@ROBOSUITE_OBJECT_REGISTRY.register(name="lid")
class RobosuiteLid(RobosuiteObjectOptions):
    """TwoArmTransport's ``Lid``: a transparent 30 cm top with a dark-wood handle."""

    file: str = "objects/lid.xml"


@ROBOSUITE_OBJECT_REGISTRY.register(name="tool_hang_stand")
class RobosuiteToolHangStand(RobosuiteFixtureOptions):
    """ToolHang's ``StandWithMount``: base slab and a 12 mm square socket (a fixture; robosuite's 8 kg base never moves)."""

    file: str = "objects/tool_hang_stand.xml"


@ROBOSUITE_OBJECT_REGISTRY.register(name="tool_hang_frame")
class RobosuiteToolHangFrame(RobosuiteObjectOptions):
    """ToolHang's ``HookFrame``: the L-shaped hook with its grip block and insertion tip."""

    file: str = "objects/tool_hang_frame.xml"


@ROBOSUITE_OBJECT_REGISTRY.register(name="tool_hang_wrench")
class RobosuiteToolHangWrench(RobosuiteObjectOptions):
    """ToolHang's ``RatchetingWrenchObject``: handle, two ring heads and the grip block."""

    file: str = "objects/tool_hang_wrench.xml"


__all__ = [
    "ROBOSUITE_DATASET",
    "ROBOSUITE_OBJECT_REGISTRY",
    "RethinkMount",
    "RobosuiteBottle",
    "RobosuiteBread",
    "RobosuiteCan",
    "RobosuiteCereal",
    "RobosuiteDoor",
    "RobosuiteDoorNoLatch",
    "RobosuiteFixtureOptions",
    "RobosuiteHammer",
    "RobosuiteHammerPayload",
    "RobosuiteLemon",
    "RobosuiteLid",
    "RobosuiteMilk",
    "RobosuiteObjectOptions",
    "RobosuitePickPlaceBin1",
    "RobosuitePickPlaceBin2",
    "RobosuitePlateWithHole",
    "RobosuitePotWithHandles",
    "RobosuiteRoundNut",
    "RobosuiteSquareNut",
    "RobosuiteToolHangFrame",
    "RobosuiteToolHangStand",
    "RobosuiteToolHangWrench",
    "RobosuiteTransportBin",
    "prepare_robosuite_object_xml",
]
