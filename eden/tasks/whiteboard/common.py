"""The whiteboard workcell: a Panda on a stand, facing a classroom whiteboard, holding a whiteboard marker.

The board, the marker and the ink law are RoboWorld's ``whiteboard`` environment (``gym-anything/robotics``);
the robot is not. The original's G1 humanoid stands at the board and must pick a capped marker out of the tray
and uncap it; here a Panda with its Franka Hand already holds the uncapped marker, by friction only, and the
tasks are the original's three drawing tasks (see ``config.py``).

Layout (world frame, metres; the robot faces ``+x``):

* The board is the original's MooreCo 8 x 4 ft board: 2.4384 x 1.2192 m, bottom edge 0.9144 m above the floor.
  Its writing face is the plane ``x = BOARD_FACE_X``. Every drawing is centred at :data:`DRAWING_CENTER`, the
  original's ``(centre_x, centre_z) = (-0.05, 1.28)``: 1.28 m high and 5 cm right of the board's middle.
* The board frame of a drawing is ``(u, v)`` from :data:`DRAWING_CENTER`: ``u`` along :data:`BOARD_U` (the
  robot's right, world ``-y``), ``v`` along :data:`BOARD_V` (up). :data:`BOARD_NORMAL` points into the board.
* The Panda stands on a :data:`STAND_HEIGHT` stand 0.50 m from the board, its base right in front of the
  drawing centre. Searched, not copied: the original's height puts the drawing centre in a Panda's wrist
  singularity (the hand straight ahead of the shoulder), and one continuous IK family has to reach the whole
  0.6 x 0.36 m table — this one does, with every joint at least 0.12 rad inside its limits.

The marker is the original's STAEDTLER 351 B: a 14 mm barrel (16.2 g uncapped) and a 5 x 2 x 7.5 mm chisel nib.
The original bonds an elastic nib (0.9 MPa) to the barrel; here the nib is a body on a 2 mm slide along the
barrel's axis with the spring that solid nib has along it, ``E * A / L`` = 1200 N/m, so the press force is read
off the solver's nib-board contact rather than a stiff rigid contact. The hand holds the barrel 35 mm from its
back end (the back end then clears the palm), the nib pitched 20 degrees below the board's normal and its chisel
turned 45 degrees, so horizontal and vertical lines come out equally wide.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np
from genesis.utils import geom as gu

from eden.options.entities import BoxOptions, EntityOptions, PlaneOptions
from eden.options.materials import RigidMaterialOptions
from eden.options.robots import PandaFrankaHand
from eden.utils.misc import get_cache_dir
from eden.utils.procedural_meshes import write_atomically

# --- the board (the original's MooreCo board, its placement and its drawing centre) ---------------------------
BOARD_SIZE = (2.4384, 1.2192)  # width, height
BOARD_BOTTOM = 0.9144
BOARD_THICKNESS = 0.0111125
BOARD_FACE_X = 0.50
#: Where every drawing is centred: the original's (centre_x, centre_z) = (-0.05, 1.28), 5 cm right of mid-board.
DRAWING_CENTER = (BOARD_FACE_X, 0.0, 1.28)
BOARD_U = (0.0, -1.0, 0.0)
BOARD_V = (0.0, 0.0, 1.0)
BOARD_NORMAL = (1.0, 0.0, 0.0)  # into the board
_BOARD_CENTER = (BOARD_FACE_X + BOARD_THICKNESS / 2, 0.05, BOARD_BOTTOM + BOARD_SIZE[1] / 2)
#: Porcelain against the nib: the nib's 0.35 is the pair's friction (Genesis takes the larger of the two).
BOARD_FRICTION = 0.2

#: The original's marker colours (digital appearance, as its report says, not spectrophotometry).
COLORS = {
    "black": (0.025, 0.031, 0.042),
    "red": (0.72, 0.028, 0.045),
    "blue": (0.024, 0.10, 0.63),
    "green": (0.02, 0.34, 0.12),
}

# --- the marker ------------------------------------------------------------------------------------------------
# Marker frame: +z from the back end toward the nib. The barrel runs from MARKER_BACK to the collar; the nib's
# root sits at 55.5 mm (the original's), so the uncompressed tip is at NIB_TIP.
MARKER_RADIUS = 0.007
MARKER_MASS = 0.0162
MARKER_BACK = -0.0705
NIB_ROOT = 0.0555
NIB_SIZE = (0.005, 0.002, 0.0075)  # chisel width, thickness, length
NIB_TIP = NIB_ROOT + NIB_SIZE[2]
#: The chisel's half-width and half-thickness: the ink footprint's semi-axes (the original's 2.5 x 1 mm ellipse).
NIB_SEMI_AXES = (NIB_SIZE[0] / 2, NIB_SIZE[1] / 2)
#: Nib spring along the barrel (N/m): the original's 0.9 MPa nib, E * A / L over its 5 x 2 mm section, 7.5 mm long.
NIB_STIFFNESS = 0.9e6 * NIB_SIZE[0] * NIB_SIZE[1] / NIB_SIZE[2]
NIB_TRAVEL = 0.002
#: Nib slide damping (N s/m). Above the original's 0.45 N s/m contact damping: the 0.2 g nib on its 1200 N/m spring
#: rings at ~390 Hz, and this keeps it settled at the 2.5 ms substep.
NIB_DAMPING = 2.0
NIB_FRICTION = 0.35
BARREL_FRICTION = 0.45
#: Contact solver parameters of the barrel (timeconst, dampratio, dmin, dmax, width, mid, power): the floor time
#: constant at the 2.5 ms substep, as robosuite's ports give a grasped tool (``GRASPED_SOL_PARAMS``). On Genesis'
#: default 0.02 s the pinch creeps: the barrel slid 2.5 mm back into the hand under a 1 N press and forward again in
#: the air, on a grip that holds 70 N by friction.
MARKER_SOL_PARAMS = (0.005, 1.0, 0.95, 0.999, 0.001, 0.5, 2.0)
#: Where the pads hold the barrel, in the marker frame: 35 mm from its back end, which then clears the palm.
GRIP_Z = MARKER_BACK + 0.035
#: The nib tip, uncompressed, ahead of the pads' midpoint along the marker's axis.
NIB_AHEAD_OF_GRIP = NIB_TIP - GRIP_Z


MarkerColor = Literal["black", "red", "blue", "green"]


def marker_file(color: str) -> str:
    """Path of the marker MJCF in ``color`` (one of :data:`COLORS`), written to Eden's cache (rewritten per call).

    One free body (barrel and collar) and the nib on its spring slide; the nib geom is named ``nib``, which is how
    the ink metric tells the nib's contacts from the barrel's.
    """
    if color not in COLORS:
        raise ValueError(f"marker_file: unknown marker colour {color!r}; the markers are {list(COLORS)}.")
    r, g, b = COLORS[color]
    barrel_half = (NIB_ROOT - 0.010 - MARKER_BACK) / 2
    collar_half = 0.005
    text = f"""<mujoco model="whiteboard_marker_{color}">
  <compiler angle="radian"/>
  <worldbody>
    <body name="marker">
      <freejoint name="marker_free"/>
      <inertial pos="0 0 -0.008" mass="{MARKER_MASS}" diaginertia="2.3e-5 2.3e-5 4.0e-7"/>
      <geom name="barrel" type="cylinder" size="{MARKER_RADIUS} {barrel_half}" pos="0 0 {MARKER_BACK + barrel_half}"
            friction="{BARREL_FRICTION}" rgba="{r} {g} {b} 1"/>
      <geom name="collar" type="cylinder" size="0.004 {collar_half}" pos="0 0 {NIB_ROOT - collar_half}"
            friction="{BARREL_FRICTION}" rgba="0.9 0.9 0.9 1"/>
      <body name="nib" pos="0 0 {NIB_ROOT + NIB_SIZE[2] / 2}">
        <joint name="nib_slide" type="slide" axis="0 0 1" range="{-NIB_TRAVEL} 0" stiffness="{NIB_STIFFNESS}"
               damping="{NIB_DAMPING}" armature="0.0001"/>
        <inertial pos="0 0 0" mass="0.0002" diaginertia="1e-9 1e-9 1e-9"/>
        <geom name="nib" type="box" size="{NIB_SIZE[0] / 2} {NIB_SIZE[1] / 2} {NIB_SIZE[2] / 2}"
              friction="{NIB_FRICTION}" rgba="{r} {g} {b} 1"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""
    path = f"{get_cache_dir('whiteboard')}/marker_{color}.xml"
    write_atomically(path, text)
    return path


class WhiteboardMarkerOptions(EntityOptions):
    """The whiteboard marker whose ink is ``ink`` (one of :data:`COLORS`), a free body (parked until the reset puts
    it in the hand). ``ink`` also colours the barrel and nib; the entity's own ``color`` (a surface RGBA) is unused.

    Its MJCF is generated (:func:`marker_file`) when the env builds the entity, not when a task config is
    defined or imported: ``file`` stays a name, the source identity a saved config carries, and
    :meth:`resolve_runtime_file` hands the builder the written file.
    """

    ink: MarkerColor = "black"
    file: str = "whiteboard_marker.xml"
    is_fixed_base: bool = False
    default_armature: float = 0.0

    def resolve_runtime_file(self) -> str:
        return marker_file(self.ink)


# --- the robot --------------------------------------------------------------------------------------------------
STAND_HEIGHT = 0.63
ROBOT_BASE = (0.0, 0.0, STAND_HEIGHT)
#: The Franka Hand's fingers: its rated 70 N continuous grasp, 35 N a finger. Commanded shut on the 14 mm barrel the
#: fingers stop ~6 mm short of closure, so the servo is stiff enough to saturate there (at 2000 N/m it squeezed 12 N).
#: The pads' friction is 1.0, with torsional friction over a 10 mm patch, which resists the marker pivoting between
#: the pads (the nib's drag acts 10 cm ahead of them), and their contacts are as stiff as the barrel's.
FINGER_FORCE = 35.0
FINGER_KP = 10000.0
PAD_FRICTION = 1.0
PAD_TORSIONAL_FRICTION = 0.01
#: The hand's orientation while writing (:func:`hand_quat`): its approach axis (the marker's) along +x pitched
#: 20 degrees nib down, rolled 45 degrees about it so the chisel lies diagonally on the board.
HAND_PITCH_DEG = 20.0
HAND_ROLL_DEG = 45.0
#: The arm's start: the nib 30 mm off the board in front of the drawing centre, at :func:`hand_quat`.
HOME = {
    "joint1": 0.2503,
    "joint2": -1.0446,
    "joint3": -0.1761,
    "joint4": -2.5179,
    "joint5": -0.4675,
    "joint6": 2.6796,
    "joint7": -1.2364,
}
HOME_STANDOFF = 0.03


def hand_quat() -> tuple[float, float, float, float]:
    """The writing orientation of the Franka Hand (wxyz): see :data:`HAND_PITCH_DEG`, :data:`HAND_ROLL_DEG`."""
    pointing = gu.axis_angle_to_quat(np.array(math.pi / 2), np.array([0.0, 1.0, 0.0]))  # hand z -> world +x
    roll = gu.axis_angle_to_quat(np.array(math.radians(HAND_ROLL_DEG)), np.array([0.0, 0.0, 1.0]))
    pitch = gu.axis_angle_to_quat(np.array(math.radians(HAND_PITCH_DEG)), np.array([0.0, 1.0, 0.0]))
    q = gu.transform_quat_by_quat(gu.transform_quat_by_quat(roll, pointing), pitch)
    return tuple(float(x) for x in q)


def whiteboard_rig() -> PandaFrankaHand:
    """The Panda + Franka Hand rig at :data:`ROBOT_BASE`, at :data:`HOME`, with the Franka Hand's rated grasp."""
    rig = PandaFrankaHand(base_pos=ROBOT_BASE, arm_name="robot", gripper_name="robot_gripper")
    rig.arm.default_dofs_pos = dict(HOME)
    fingers = rig.gripper_dofs
    rig.gripper.default_dofs_kp = dict.fromkeys(fingers, FINGER_KP)
    rig.gripper.default_dofs_kd = dict.fromkeys(fingers, 0.0)
    rig.gripper.default_dofs_damping = dict.fromkeys(fingers, 100.0)
    rig.gripper.default_dofs_force_limits = dict.fromkeys(fingers, FINGER_FORCE)
    rig.gripper.default_geoms_friction = {".*_gripper_pad": PAD_FRICTION}
    rig.gripper.default_geoms_friction_torsional = {".*_gripper_pad": PAD_TORSIONAL_FRICTION}
    # Genesis averages a contact pair's solver parameters: the barrel's stiffness needs the pads' to match.
    rig.gripper.default_geoms_sol_params = {".*_gripper_pad": MARKER_SOL_PARAMS}
    return rig


def whiteboard_scene(color: str) -> dict:
    """The workcell's entities, keyed by scene name, with the marker in ``color``: splat into ``SceneOptions``."""
    return {
        "plane": PlaneOptions(),
        "robot": whiteboard_rig(),
        "stand": BoxOptions(
            size=(0.30, 0.30, STAND_HEIGHT),
            fixed=True,
            default_root_pos=(ROBOT_BASE[0] - 0.05, ROBOT_BASE[1], STAND_HEIGHT / 2),
            color=(0.35, 0.36, 0.38, 1.0),
        ),
        "board": BoxOptions(
            size=(BOARD_THICKNESS, *BOARD_SIZE),
            fixed=True,
            default_root_pos=_BOARD_CENTER,
            material=RigidMaterialOptions(friction=BOARD_FRICTION),
            color=(0.965, 0.965, 0.965, 1.0),
        ),
        # Parked on the floor until the reset puts it in the hand (PlaceMarkerInGrasp).
        "marker": WhiteboardMarkerOptions(
            ink=color,
            default_geoms_sol_params={"marker": MARKER_SOL_PARAMS},
            default_root_pos=(0.3, 0.6, 0.01),
        ),
    }
