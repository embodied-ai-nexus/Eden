"""The bimanual YAM workcell every Mochi deformable task shares, and the tool-centre helpers their terms use.

The tasks are Eden's versions of RoboSimStudio's three demos (``examples/08_demos``), which run two i2RT YAM arms
on a bench. Every position here is in RoboSimStudio's work-surface frame: the origin at the middle of the bench
top, +x away from the arms, +y toward the left arm and z up from the bench top, which is the ground plane. The
arms stand level with the bench 0.3475 m behind its middle, 0.62 m apart (RoboSimStudio's ``yam_i2rt``
workcell), so a demo's constant reads the same here.

The robot is Eden's :class:`~eden.options.robots.i2rt_yam.Yam` on the Mochi solver, gravity-free as
RoboSimStudio's arms are (gravity compensation is baked into its MJCF). Its servo gains are RoboSimStudio's, not
the asset options' sim-to-real heuristics: the arm at the i2RT MJCF's kp 300 / kv 30, and the jaws at whatever
squeeze the task's demo calibrated (Mochi clamps no drive force, so a squeeze cap is a gain: force = kp x the
finger's remaining travel).
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET

import torch
from genesis.utils.geom import transform_by_quat

import eden as en
from eden.options.entities import PlaneOptions
from eden.options.materials import MochiRigidMaterialOptions
from eden.utils.assets import get_asset_path
from eden.utils.procedural_meshes import derived_file_path, write_atomically

# Penalty stiffness of every contact (Pa/m): Mochi conditions it well in single precision, so the tasks
# train under `eden train`'s default fp32 (Mochi's 1e9 default wants `en.init(precision="64")`).
CONTACT_STIFFNESS = 1e7
# Half-width (m) of the contact pressure ramp of a part the pads squeeze, and of the YAM where they squeeze it (taken
# from the collider side of a pair). Mochi's 5 mm default spreads the ramp over 10 mm: closed on the sponge, the pads
# sank 6 mm into it before the pressure caught the squeeze; at 1 mm on both they stop within 2 mm. Not on the cable
# task's YAMs: sharper jaws drag the thin cable into a loop as they let go at the port.
CONTACT_RAMP = 1e-3

YAM_ARM_DOFS = [f"joint{i}" for i in range(1, 7)]
YAM_FINGER_DOFS = ["left_finger", "right_finger"]
# Finger travel: the left finger opens toward +, the right toward -.
YAM_FINGERS_OPEN = {"left_finger": 0.037524, "right_finger": -0.037524}
YAM_FINGERS_CLOSED = {"left_finger": 0.0, "right_finger": 0.0}
# How fast a finger travels (m/s), its target ramping toward open or closed at this rate (custom_terms.grasp's
# RateLimitedGripper). i2RT publishes the jaws' 95 mm stroke but no speed; driven straight to the end of their
# travel instead, the servos closed at 0.1-1.9 m/s a finger, set by each task's squeeze stiffness.
YAM_FINGER_SPEED = 0.06
# The point between the fingertip pads, in link_6's frame (measured on the built arm: the pads' contact
# patches on a gripped sponge centre here). The jaws close along link_6's x.
YAM_TCP_OFFSET = (0.0, -0.044, 0.1305)
# link_6 pointing straight down with the jaws closing along world x (wxyz).
YAM_HAND_DOWN = (0.0, 1.0, 0.0, 0.0)

# The arm bases in the work-surface frame: RoboSimStudio's yam_i2rt mounts.
YAM_BASES = {"right": (-0.3475, -0.31, 0.0), "left": (-0.3475, 0.31, 0.0)}
# The YAM's base plate reaches 10 mm below its mount: a mount this high stands the plate on the ground plane.
YAM_PLATE_DEPTH = 0.010
# Ready postures, from Genesis IK: tool centre 0.15 m over the mount's height at (-0.10, -/+0.20), pointing down,
# the jaws closing along world x ("jaws_x") or y ("jaws_y"). An episode starts here rather than at RoboSimStudio's
# home (gripper pointing forward), since the policy holds the hand's tilt.
YAM_READY = {
    ("right", "jaws_x"): (0.2606, 1.5136, 1.3619, -1.4192, 0.0002, 1.8315),
    ("right", "jaws_y"): (0.3608, 1.7250, 1.5839, -1.4296, 0.0, 0.3608),
    ("left", "jaws_x"): (-0.5566, 1.6477, 1.4971, -1.4203, 0.0, 1.0142),
    ("left", "jaws_y"): (-0.3608, 1.7250, 1.5839, -1.4296, 0.0, -0.3608),
}

# RoboSimStudio's i2RT MJCF servos: arm kp 300 / kv 30.
YAM_ARM_KP = 300.0
YAM_ARM_KD = 30.0


def yam_mochi_file() -> str:
    """The YAM's MJCF without its fingertip micro-spheres, written once to Eden's cache.

    Each finger carries six 0.6 mm collision spheres at its tip (for MuJoCo's point contacts). Mochi samples every
    collision triangle, and a tessellated sphere is 960 samples: the twelve made up 55 % of the arm's 20,800, all
    where the hand meets the part, and a policy pushing the hand into the sponge slowed the step sevenfold. The
    pads' boxes, which the spheres sit on, carry the grip without them.
    """
    return without_collision_spheres(get_asset_path("manipulators/i2rt_yam/yam.xml"))


def without_collision_spheres(source: str) -> str:
    """A cached copy of the MJCF ``source`` without its ``sphere_collision`` geoms; return its path.

    The copy reads its meshes from the source's directory by absolute path, so it is one file per source file
    (:func:`~eden.utils.procedural_meshes.derived_file_path`), rewritten on every call: another checkout gets a copy
    of its own, and an updated asset is never shadowed by a stale one.
    """
    stem = os.path.splitext(os.path.basename(source))[0]
    path = derived_file_path("mochi_assets", source, f"{stem}_no_collision_spheres.xml")
    root = ET.parse(source).getroot()
    compiler = root.find("compiler")
    compiler.set("meshdir", os.path.join(os.path.dirname(os.path.abspath(source)), compiler.get("meshdir", "")))
    for parent in root.iter():
        for geom in list(parent.findall("geom")):
            if geom.get("class") == "sphere_collision":
                parent.remove(geom)
    write_atomically(path, ET.tostring(root, encoding="unicode"))
    return path


def yam(
    side: str,
    *,
    ready: str,
    finger_kp: float,
    finger_kd: float,
    arm_kp: float = YAM_ARM_KP,
    arm_kd: float = YAM_ARM_KD,
    friction: float = 0.9,
    base_height: float = 0.0,
    fingers_open: bool = True,
    penalty_threshold: float = 1e-3,
    contact_ramp: float | None = None,
) -> en.robots.Yam:
    """One YAM of the workcell, on the Mochi solver, at its mount and ready posture.

    Parameters
    ----------
    side : {"right", "left"}
        Which mount (:data:`YAM_BASES`).
    ready : {"jaws_x", "jaws_y"}
        The starting posture (:data:`YAM_READY`).
    finger_kp, finger_kd : float
        The jaws' servo: closed on a part ``w`` wide, each finger squeezes with ``finger_kp * w / 2``. How fast the
        jaws travel is the gripper action's to set (:data:`YAM_FINGER_SPEED`), not the servo's.
    arm_kp, arm_kd : float
        The six arm joints' servo.
    friction : float
        The arm's contact friction; Mochi combines two bodies' frictions by their geometric mean.
    base_height : float
        The mount's height over the ground plane (m). The base plate reaches :data:`YAM_PLATE_DEPTH` below it, so
        under that the plate sinks into the plane.
    fingers_open : bool
        Whether the jaws start open (else shut, as a hand that only presses).
    penalty_threshold : float
        How far out from the arm's surface a soft body's samples start to be pushed (m): a rod's radius, so the
        jaws feel a cable's thickness and not its centreline.
    contact_ramp : float, optional
        The arm's contact pressure ramp half-width (m), :data:`CONTACT_RAMP` for jaws that squeeze a solid; Mochi's
        default when None.
    """
    if (side, ready) not in YAM_READY:
        raise ValueError(f"No YAM ready posture for side={side!r}, ready={ready!r}; available: {sorted(YAM_READY)}.")
    arm = YAM_READY[(side, ready)]
    return en.robots.Yam(
        file=yam_mochi_file(),
        # Gravity-free: RoboSimStudio's YAMs are gravity-compensated, and Mochi has no compensation term.
        material=MochiRigidMaterialOptions(
            penalty_coefficient=CONTACT_STIFFNESS,
            has_gravity=False,
            friction=friction,
            penalty_threshold=penalty_threshold,
            **({} if contact_ramp is None else {"penalty_smoothing_half_distance": contact_ramp}),
        ),
        default_root_pos=(*YAM_BASES[side][:2], base_height),
        default_dofs_pos={**dict(zip(YAM_ARM_DOFS, arm)), **(YAM_FINGERS_OPEN if fingers_open else YAM_FINGERS_CLOSED)},
        default_dofs_kp={**dict.fromkeys(YAM_ARM_DOFS, arm_kp), **dict.fromkeys(YAM_FINGER_DOFS, finger_kp)},
        default_dofs_kd={**dict.fromkeys(YAM_ARM_DOFS, arm_kd), **dict.fromkeys(YAM_FINGER_DOFS, finger_kd)},
        # Mochi clamps no drive force: the asset's effort limits would be rejected.
        default_dofs_force_limits={},
    )


def bench(friction: float = 0.9, penalty_threshold: float = 1e-3) -> PlaneOptions:
    """The bench top as the ground plane, a Mochi rigid body like everything else in a Mochi scene.

    ``penalty_threshold`` is the distance at which contact pressure starts to build. A rod touches through
    its centreline, so a cable lies on top of the bench only when this is its radius.
    """
    return PlaneOptions(
        material=MochiRigidMaterialOptions(
            penalty_coefficient=CONTACT_STIFFNESS, friction=friction, penalty_threshold=penalty_threshold
        )
    )


def link_point(link, offset: torch.Tensor) -> torch.Tensor:
    """World position of the point fixed at ``offset`` in ``link``'s frame, ``(num_envs, 3)``."""
    quat = link.get_quat()
    return link.get_pos() + transform_by_quat(offset.expand_as(quat[:, :3]), quat)
