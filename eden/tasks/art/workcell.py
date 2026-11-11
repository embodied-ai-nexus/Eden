"""The art tasks' workcell: a Panda on a desk, a sheet of paper in front of it, a brush or a pen in its Franka Hand.

Shared by the calligraphy tasks (``eden.tasks.art.calligraphy``) and the painting tasks (``eden.tasks.art.painting``).
Layout (world frame, metres; the robot faces ``+x``):

* The desk's top is :data:`DESK_HEIGHT` above the floor; the Panda's base stands on it at the origin of ``x, y``.
* The sheet lies on the desk on a felt pad, its face :data:`PAPER_THICKNESS` above the desk, centred at
  :data:`SHEET_CENTER`. The sheet frame is ``(u, v)`` from that centre: ``u`` to the robot's right (world ``-y``),
  ``v`` up the page, away from the robot (world ``+x``); into the paper is world ``-z``.
* The tool is welded in the Franka Hand, the fingers closed on its handle. Unlike the whiteboard's marker it is not
  held by friction: over an episode a friction grip creeps — a pen held that way turned 29 degrees about its axis
  and slid 5 mm out of the pinch in 50 s — and a pen's shading depends on how its nib is turned.

The tools (:data:`TOOLS`; tool frame: ``+z`` from the back end toward the tip, origin where the hairs or the nib
leave the tool):

* The **brush**: a Chinese writing brush, a 10 mm bamboo handle and ferrule, 13.5 cm long, and a 30 mm tuft of
  hairs (9 mm thick at the ferrule: RoboPaint's ``bristle_m`` and ``bristle_r``), held 10 cm up from the ferrule.
  It points straight down.
* The **pen**: a 10 mm holder, 11.5 cm long, and a 22 mm nib, held 8 cm up from the nib. It leans
  :data:`PEN_TILT_DEG` from upright along round hand's slant, its nib's slit along the downstrokes, so they come out
  shaded and the upstrokes hairline. Leaning so, the holder's rim clears the paper by 12 mm when the tip is on it,
  so the tines can be pressed their full 3.5 mm.
* The **paintbrush**: a round painting brush, a 10 mm handle and ferrule, 15 cm long, and a 15 mm tuft 8 mm thick
  (RoboPaint's acrylic ``bristle_m`` and ``bristle_r``), held 11.5 cm up from the ferrule, so that its tip is where
  the Chinese brush's is. It points straight down.

The brushes are held 35 mm from their back end and the pen likewise, as the whiteboard's marker is: the back end
then clears the palm. The hairs, the tuft and the nib are drawn but do not collide (``contype="0"
conaffinity="0"``): what they mark is the task's tool model's, read from where the tip would be undeformed. The rest
of each tool collides, so pressing deeper than the hairs are long drives the ferrule into the paper.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal

import numpy as np
from genesis.utils import geom as gu

from eden.options.entities import BoxOptions, EntityOptions, PlaneOptions
from eden.options.materials import RigidMaterialOptions
from eden.options.robots import PandaFrankaHand
from eden.utils.misc import get_cache_dir
from eden.utils.plane_grid import PlaneGrid
from eden.utils.procedural_meshes import write_atomically

ToolKind = Literal["brush", "pen", "paintbrush"]

# --- the desk and the sheet ------------------------------------------------------------------------------------------
DESK_HEIGHT = 0.75
DESK_SIZE = (1.2, 1.4, DESK_HEIGHT)  # depth (x), width (y), height
DESK_CENTER_X = 0.25
#: The felt pad and the paper on it, as one box: the paper's face is this far above the desk.
PAPER_THICKNESS = 0.003
SHEET_CENTER = (0.50, 0.0, DESK_HEIGHT + PAPER_THICKNESS)
SHEET_U = (0.0, -1.0, 0.0)
SHEET_V = (1.0, 0.0, 0.0)
PAPER_FRICTION = 0.5
#: The paper's colour (RGB in ``[0, 1]``) as the scene draws it.
PAPER_COLOR = (0.96, 0.94, 0.88)


def sheet_grid(size: tuple[float, float], cell: float) -> PlaneGrid:
    """The grid over a ``size = (width, height)`` sheet centred on the sheet frame's origin, in ``cell``-m cells."""
    cols, rows = (math.ceil(round(extent / cell, 6)) for extent in size)
    return PlaneGrid(-cols * cell / 2, rows * cell / 2, cell, rows, cols)


# --- the tools -----------------------------------------------------------------------------------------------------
TOOL_RADIUS = 0.005
#: The tool's lateral axis in its own frame: the way a pen's tines spread (the Franka Hand's finger axis).
TOOL_LATERAL = (0.0, 1.0, 0.0)
#: The pen leans this far from upright (the tines then meet the paper at 45 degrees, as a hand holds a pen).
PEN_TILT_DEG = 45.0
#: Round hand's slant: downstrokes lean 55 degrees from the line. The pen's slit is held along it.
SLANT_DEG = 55.0
#: The arm's start puts the tip this far above the sheet's centre, the hand at its tool's ``hand_quat``.
HOME_STANDOFF = 0.03


@dataclass(frozen=True)
class ToolSpec:
    """Everything about a held tool's body: one entry of :data:`TOOLS` per :data:`ToolKind`.

    What the tool marks is not here: that is the task's tool model (``eden.tasks.art.calligraphy.tools``,
    ``eden.tasks.art.painting.paint``), read from where :attr:`tip` is.

    Parameters
    ----------
    back, tip, grip, mass : float
        The tool in its own frame (m, kg; see the module docstring): its back end, its undeformed tip, the grip
        point between the pads, and its mass.
    geoms : str
        Its MJCF geoms; the hairs, the tuft or the nib are visual only (``contype="0" conaffinity="0"``).
    home : Mapping[str, float]
        The Panda's joints at the start: the tip :data:`HOME_STANDOFF` above the sheet's centre, the hand at
        ``hand_quat``.
    hand_quat : tuple[float, float, float, float]
        The Franka Hand's working orientation (wxyz): its approach axis is the tool's.
    draw_speed : float
        The scripted demonstrators' default speed (m/s) on and near the paper.
    """

    back: float
    tip: float
    grip: float
    mass: float
    geoms: str
    home: Mapping[str, float] = field(hash=False)  # a mapping has no hash; the spec stays hashable without it
    hand_quat: tuple[float, float, float, float]
    draw_speed: float

    @property
    def tip_ahead_of_grip(self) -> float:
        """How far the undeformed tip is ahead of the pads' midpoint along the tool's axis."""
        return self.tip - self.grip


def _sheet_to_world(u: float, v: float) -> np.ndarray:
    return u * np.asarray(SHEET_U) + v * np.asarray(SHEET_V)


def pen_lateral() -> tuple[float, float]:
    """The pen's lateral axis on the sheet, ``(u, v)``: across round hand's slant, pointing right and down."""
    slant = math.radians(SLANT_DEG)
    return (math.sin(slant), -math.cos(slant))


def _quat_of_axes(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[float, float, float, float]:
    return tuple(float(q) for q in gu.R_to_quat(np.stack([x, y, z], axis=1)))


def _downward_hand_quat() -> tuple[float, float, float, float]:
    """A brush points straight down, the hand's ``x`` along world ``+x``."""
    x, z = np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, -1.0])
    return _quat_of_axes(x, np.cross(z, x), z)


def _pen_hand_quat() -> tuple[float, float, float, float]:
    """The pen points up the page along round hand's slant, leaning :data:`PEN_TILT_DEG` from upright, the hand's
    ``y`` (the fingers' axis, and the tool's :data:`TOOL_LATERAL`) level, along :func:`pen_lateral`: the hand's 20 cm
    width stays off the paper.
    """
    slant, tilt = math.radians(SLANT_DEG), math.radians(PEN_TILT_DEG)
    up_slant = _sheet_to_world(math.cos(slant), math.sin(slant))
    z = math.sin(tilt) * up_slant + math.cos(tilt) * np.array([0.0, 0.0, -1.0])
    y = -_sheet_to_world(*pen_lateral())  # either way spreads the tines; this one the arm reaches easily
    return _quat_of_axes(np.cross(y, z), y, z)


def _brush_geoms(back: float, tuft_length: float, tuft_radius: float, *, handle_rgba: str, tuft_rgba: str) -> str:
    """A handle and ferrule from ``back`` to the tool's origin, and a visual tuft ``tuft_length`` long past it: a
    cylinder for its first half and a tapering ellipsoid for the rest.
    """
    body_half = (-0.012 - back) / 2
    half = tuft_length / 4
    return f"""
      <geom name="handle" type="cylinder" size="{TOOL_RADIUS} {body_half}" pos="0 0 {back + body_half}"
            rgba="{handle_rgba}"/>
      <geom name="ferrule" type="cylinder" size="0.0055 0.006" pos="0 0 -0.006"
            rgba="0.12 0.10 0.09 1"/>
      <geom name="hairs" type="cylinder" size="{tuft_radius} {half}" pos="0 0 {half}" contype="0" conaffinity="0"
            rgba="{tuft_rgba}"/>
      <geom name="hair_tip" type="ellipsoid" size="{tuft_radius} {tuft_radius} {2 * half}" pos="0 0 {2 * half}"
            contype="0" conaffinity="0" rgba="{tuft_rgba}"/>"""


def _pen_geoms(back: float, tip: float) -> str:
    body_half = -back / 2
    return f"""
      <geom name="holder" type="cylinder" size="{TOOL_RADIUS} {body_half}" pos="0 0 {back + body_half}"
            rgba="0.10 0.10 0.12 1"/>
      <geom name="nib" type="box" size="0.0004 0.002 {tip / 2}" pos="0 0 {tip / 2}" contype="0"
            conaffinity="0" rgba="0.75 0.72 0.65 1"/>"""


#: The Panda's joints with a downward brush's tip :data:`HOME_STANDOFF` above the sheet's centre. The Chinese brush
#: and the paintbrush share them: their tips are as far ahead of the grip.
_DOWNWARD_HOME = MappingProxyType(
    {
        "joint1": 0.0,
        "joint2": 0.1223,
        "joint3": 0.0,
        "joint4": -2.3944,
        "joint5": 0.0,
        "joint6": 2.5168,
        "joint7": 0.7854,
    }
)

TOOLS: dict[ToolKind, ToolSpec] = {
    "brush": ToolSpec(
        back=-0.135,
        tip=0.030,
        grip=-0.10,
        mass=0.018,
        geoms=_brush_geoms(-0.135, 0.030, 0.0045, handle_rgba="0.62 0.47 0.27 1", tuft_rgba="0.06 0.06 0.06 1"),
        home=_DOWNWARD_HOME,
        hand_quat=_downward_hand_quat(),
        draw_speed=0.02,
    ),
    "pen": ToolSpec(
        back=-0.115,
        tip=0.022,
        grip=-0.08,
        mass=0.012,
        geoms=_pen_geoms(-0.115, 0.022),
        home=MappingProxyType(
            {
                "joint1": 0.2941,
                "joint2": 0.1411,
                "joint3": -0.1170,
                "joint4": -2.6209,
                "joint5": -1.9526,
                "joint6": 2.5853,
                "joint7": 0.2862,
            }
        ),
        hand_quat=_pen_hand_quat(),
        # Slower than the brush: at 2 cm/s the arm's IK lag left *minimum*'s first dot at 74 % covered, under
        # lettering's 75 % stroke floor, for 1 arm reset noise in 10; at 1.5 cm/s all 10 passed (about 58 s of 120).
        draw_speed=0.015,
    ),
    "paintbrush": ToolSpec(
        back=-0.15,
        tip=0.015,
        grip=-0.115,
        mass=0.016,
        geoms=_brush_geoms(-0.15, 0.015, 0.004, handle_rgba="0.80 0.20 0.16 1", tuft_rgba="0.30 0.22 0.15 1"),
        home=_DOWNWARD_HOME,
        hand_quat=_downward_hand_quat(),
        draw_speed=0.03,
    ),
}


def tool_file(kind: ToolKind) -> str:
    """Path of the MJCF of tool ``kind``, written to Eden's cache (rewritten per call).

    One free body named ``tool``, which the scene welds into the hand; the hairs, the tuft or the nib are visual
    geoms only.
    """
    if kind not in TOOLS:
        raise ValueError(f"tool_file: unknown tool {kind!r}; the tools are {list(TOOLS)} (art.workcell.TOOLS).")
    spec = TOOLS[kind]
    text = f"""<mujoco model="tool_{kind}">
  <compiler angle="radian"/>
  <worldbody>
    <body name="tool">
      <freejoint name="tool_free"/>
      <inertial pos="0 0 {spec.back / 2}" mass="{spec.mass}" diaginertia="5e-5 5e-5 3e-7"/>{spec.geoms}
    </body>
  </worldbody>
</mujoco>
"""
    path = f"{get_cache_dir('art')}/{kind}.xml"
    write_atomically(path, text)
    return path


class HeldToolOptions(EntityOptions):
    """The held tool ``tool`` (a :data:`ToolKind`), one body, welded into the hand by the scene.

    Its MJCF is generated (:func:`tool_file`) when the env builds the entity, not when a task config is defined:
    ``file`` stays a name and :meth:`resolve_runtime_file` hands the builder the written file.
    """

    tool: ToolKind = "brush"
    file: str = "held_tool.xml"
    default_armature: float = 0.0

    def resolve_runtime_file(self) -> str:
        return tool_file(self.tool)


# --- the robot -------------------------------------------------------------------------------------------------------
ROBOT_BASE = (0.0, 0.0, DESK_HEIGHT)


def panda_rig(kind: ToolKind) -> PandaFrankaHand:
    """The Panda + Franka Hand on the desk at its tool's :attr:`ToolSpec.home`, its fingers closed on the tool's
    handle.
    """
    rig = PandaFrankaHand(base_pos=ROBOT_BASE, arm_name="robot", gripper_name="robot_gripper")
    rig.arm.default_dofs_pos = dict(TOOLS[kind].home)
    rig.preset_hold(2 * TOOL_RADIUS, clearance=0.0005)
    return rig


#: The Franka Hand's ``hand`` link in the Panda's ``attachment`` link (wxyz): turned 180 degrees about their shared
#: approach axis.
HAND_IN_ATTACHMENT = (0.0, 0.0, 0.0, 1.0)


def tool_mount(kind: ToolKind) -> tuple[float, float, float]:
    """Where the tool's frame sits in the Panda's ``attachment`` link: on the approach axis, the grip point
    (:attr:`ToolSpec.grip`) between the pads. Its orientation is the hand's, :data:`HAND_IN_ATTACHMENT`.
    """
    return (0.0, 0.0, PandaFrankaHand().tcp_offset[2] - TOOLS[kind].grip)


def tool_tip_in_attachment(kind: ToolKind) -> tuple[float, float, float]:
    """The tool's undeformed tip in the Panda's ``attachment`` link: the control point of the tasks' IK action."""
    return (0.0, 0.0, PandaFrankaHand().tcp_offset[2] + TOOLS[kind].tip_ahead_of_grip)


def scene(kind: ToolKind, sheet: tuple[float, float]) -> dict:
    """The workcell's entities, keyed by scene name, with tool ``kind`` and a ``sheet = (width, height)`` sheet,
    and the weld that mounts the tool in the hand: splat into ``SceneOptions``.
    """
    width, height = sheet
    rig = panda_rig(kind)
    return {
        "plane": PlaneOptions(),
        "robot": rig,
        "desk": BoxOptions(
            size=DESK_SIZE,
            fixed=True,
            default_root_pos=(DESK_CENTER_X, 0.0, DESK_HEIGHT / 2),
            color=(0.55, 0.42, 0.30, 1.0),
        ),
        # The paper on its felt pad: `width` along u (world -y), `height` along v (world +x).
        "sheet": BoxOptions(
            size=(height, width, PAPER_THICKNESS),
            fixed=True,
            default_root_pos=(SHEET_CENTER[0], SHEET_CENTER[1], DESK_HEIGHT + PAPER_THICKNESS / 2),
            material=RigidMaterialOptions(friction=PAPER_FRICTION),
            color=(*PAPER_COLOR, 1.0),
        ),
        "tool": HeldToolOptions(tool=kind),
        # Welded to the arm's flange link, as the hand itself is: a DOF-less body welded to the hand instead (one
        # attachment further down) reads an all-zero pose after build and reset, until the first physics step.
        "attachments_dict": {"tool": (rig.arm_name, rig.ee_link, tool_mount(kind), HAND_IN_ATTACHMENT)},
    }
