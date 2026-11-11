"""Frame constants and initial poses for the in-hand cube-reorientation task.

Ported from ``wuji-mjlab``'s ``reorient_constants.py``. Every value below transfers from
the upstream MJCF *unchanged*, which is worth stating because it is not obvious: the
upstream ``{side}_mjlab.xml`` places the palm at the origin unrotated, with the fingers
along +Z and the palm normal along +X -- exactly the frame of the URDF Eden loads, and
exactly Eden's default ``up``/``front``, so the load-time alignment is the identity here
(pinned in ``tests/options/robots/test_wuji_hand1.py``).

The tag frame is the AprilTag mounted on the wrist of the real hand. The policy sees the
cube pose and its goal *in tag frame*, so the same observation is reproducible on hardware
from a single camera detection -- which is why the goal is composed palm -> tag -> world
rather than being sampled directly in world.
"""

import math

# ---------------------------------------------------------------------------
# Hand root pose -- palm rotated -90 deg about Y so the palm faces up
# ---------------------------------------------------------------------------
# Eden loads the hand fingers-up / palm-+X. R_y(-90 deg) maps the palm normal (+X) onto
# world +Z and the finger direction (+Z) onto world -X, giving a palm-up hand with the
# fingers held out horizontally -- the pose a person reorienting a cube in one hand holds.
# Same quaternion the upstream constants file uses.
REORIENT_HAND_ROOT_POS = (0.0, 0.0, 0.5)
REORIENT_HAND_ROOT_QUAT = (0.70710678, 0.0, -0.70710678, 0.0)

# ---------------------------------------------------------------------------
# Cube initial pose
# ---------------------------------------------------------------------------
# Cube spawn expressed in the PALM LINK frame: 6 cm along the palm normal (+X, i.e. above
# the palm surface) and 9.7 cm out along the fingers (+Z), which seats a 54 mm cube inside
# the fingertip cage. Upstream hardcodes the equivalent world position; keeping it
# palm-relative and composing through the live palm pose at reset makes it correct under
# both the per-env scene offset and the randomized wrist pitch.
REORIENT_CUBE_IN_PALM_POS = (0.0599, 0.0100, 0.0967)

# World spawn for env 0 at the nominal hand pose -- only the declared morph position; the
# reset event immediately re-places the cube from the live palm pose.
REORIENT_CUBE_SPAWN_POS = (-0.0967, 0.0100, 0.5599)

# 54 mm cube, 120 g (upstream ``cube.xml``: box half-size 0.027, mass 0.120). The cube itself
# is :class:`~eden.options.objects.mujoco_playground.DexCube`, whose MJCF owns both numbers;
# these mirror them for the terms that need them as scalars (the cage margins, the mass-DR
# baseline), and ``tests/tasks/reorient/test_constants.py`` pins the mirror to the asset.
REORIENT_CUBE_EDGE = 0.054
REORIENT_CUBE_MASS = 0.120

# ---------------------------------------------------------------------------
# Goal marker
# ---------------------------------------------------------------------------
# A kinematic twin of the cube, drawn at the goal orientation so a rendered rollout shows
# what the policy is chasing. Positioned in the PALM frame like the cube's own spawn (+X is
# the palm normal, +Z runs out along the fingers), which is what keeps it correct under the
# per-env scene offset -- a world position would stack every env's marker in one place.
# 11 cm further along the palm normal than the cube puts it directly above it: clear of the fingertips, and
# close enough that a camera framing the hand still holds both cubes.
REORIENT_GOAL_IN_PALM_POS = (0.1699, 0.0100, 0.0967)

# ---------------------------------------------------------------------------
# Tag (wrist marker) <-> palm_link rigid transform
# ---------------------------------------------------------------------------
# tag_in_palm: pos (0.0262, 0, -0.0563), quat = R_y(+90 deg).
TAG_IN_PALM_POS = (0.0262, 0.0, -0.0563)
TAG_IN_PALM_QUAT = (math.cos(math.radians(45.0)), 0.0, math.sin(math.radians(45.0)), 0.0)

# ---------------------------------------------------------------------------
# Hand joint angles (upstream "active" keyframe -- a light grasp on the cube)
# ---------------------------------------------------------------------------
# Side-agnostic regex keys so one table serves both chiralities. Distinct from the
# manufacturer home keyframe on ``_WujiHand1``: this is the pose the reorientation policy
# starts every episode from, with the fingers already curled around the cube.
REORIENT_DOFS_POS: dict[str, float] = {
    ".*_finger1_joint1": 0.8,
    ".*_finger1_joint2": -0.0215,
    ".*_finger1_joint3": 0.285,
    ".*_finger1_joint4": 0.582,
    ".*_finger2_joint1": 0.441,
    ".*_finger2_joint2": 0.163,
    ".*_finger2_joint3": 0.822,
    ".*_finger2_joint4": 0.494,
    ".*_finger3_joint1": 0.448,
    ".*_finger3_joint2": -0.0832,
    ".*_finger3_joint3": 0.883,
    ".*_finger3_joint4": 0.305,
    ".*_finger4_joint1": 0.594,
    ".*_finger4_joint2": -0.268,
    ".*_finger4_joint3": 1.13,
    ".*_finger4_joint4": 0.18,
    ".*_finger5_joint1": 1.2,
    ".*_finger5_joint2": -0.228,
    ".*_finger5_joint3": 0.794,
    ".*_finger5_joint4": 0.407,
}

# Per-pattern reset jitter, applied as an offset on ``REORIENT_DOFS_POS``: joints 1/3/4
# start 0.1-0.3 rad more extended than the keyframe, so the fingers have to close onto the
# cube rather than beginning already wrapped around it.
REORIENT_RESET_DOFS_POS_RANGE: dict[str, tuple[float, float]] = {
    ".*_joint1": (-0.3, -0.1),
    ".*_joint2": (0.0, 0.0),
    ".*_joint3": (-0.3, -0.1),
    ".*_joint4": (-0.3, -0.1),
}

# ---------------------------------------------------------------------------
# Left-hand mirroring
# ---------------------------------------------------------------------------
# The left URDF is the right one reflected through the XZ plane, but its joint *axes* were
# chosen to preserve each joint's limit range rather than its anatomical direction. The
# consequence, measured joint by joint rather than assumed: the same value produces the
# mirror-equivalent motion on every joint EXCEPT the four abduction joints of fingers 2-5,
# whose sign is inverted -- so a right-hand grasp keyframe transplanted verbatim splays the
# left hand's fingers away from the thumb instead of around the cube, and it drops the cube
# immediately. Pinned in ``tests/tasks/reorient/test_constants.py``.
MIRRORED_DOFS = ".*_finger[2-5]_joint2"


def dofs_pos_for_side(side: str) -> dict[str, float]:
    """The reorient keyframe for ``side``, mirrored from the right-hand pose if needed.

    Parameters
    ----------
    side : str
        ``"right"`` or ``"left"``.

    Returns
    -------
    dict[str, float]
        DOF-name pattern to angle [rad].
    """
    if side == "right":
        return dict(REORIENT_DOFS_POS)
    if side != "left":
        raise ValueError(f"side must be 'right' or 'left', got {side!r}.")
    return {
        name: -value if name.endswith("_joint2") and "_finger1_" not in name else value
        for name, value in REORIENT_DOFS_POS.items()
    }


def cube_in_palm_for_side(side: str) -> tuple[float, float, float]:
    """The cube spawn in the palm frame for ``side``.

    The palm-frame y offset leans the cube toward the thumb, so it flips with the hand --
    the mirror plane is XZ.

    Parameters
    ----------
    side : str
        ``"right"`` or ``"left"``.

    Returns
    -------
    tuple[float, float, float]
        Spawn position in the palm frame.
    """
    if side not in ("right", "left"):
        raise ValueError(f"side must be 'right' or 'left', got {side!r}.")
    x, y, z = REORIENT_CUBE_IN_PALM_POS
    return (x, y if side == "right" else -y, z)


def goal_in_palm_for_side(side: str) -> tuple[float, float, float]:
    """The goal marker's position in the palm frame for ``side``.

    Mirrored the same way as :func:`cube_in_palm_for_side` so the marker hangs over the
    cube rather than over the other side of the palm.

    Parameters
    ----------
    side : str
        ``"right"`` or ``"left"``.

    Returns
    -------
    tuple[float, float, float]
        Marker position in the palm frame.
    """
    if side not in ("right", "left"):
        raise ValueError(f"side must be 'right' or 'left', got {side!r}.")
    x, y, z = REORIENT_GOAL_IN_PALM_POS
    return (x, y if side == "right" else -y, z)


# ---------------------------------------------------------------------------
# Link patterns (side-agnostic -- the ``--hand`` TaskMod swaps only explicit name lists)
# ---------------------------------------------------------------------------
PALM_LINK = ".*_palm_link"
# The massless fingertip FRAMES -- upstream reads its ``*_finger[1-5]_tip`` sites here, and
# these URDF links sit at the same points. Used for fingertip velocity (slide penalty) and
# for the privileged fingertip-position observation.
TIP_LINKS = ".*_finger[1-5]_tip_link"
# The distal phalanx that actually carries the fingertip collision geometry. Contact
# queries must use this, not TIP_LINKS: a frame with no geoms never reports a contact.
TIP_COLLISION_LINKS = ".*_finger[1-5]_link4"
DISTAL_LINKS = ".*_finger[1-5]_link[34]"
PROXIMAL_LINKS = ".*_finger[1-5]_link[12]"
# Every link that bounds the fingertip "cage" the cube must stay inside.
CAGE_LINKS = [PALM_LINK, ".*_finger[1-5]_link[1-4]"]
# Upstream ``UNDESIRED_OBJECT_CONTACT_BODIES``: the palm and the proximal two-thirds of
# each finger. The cube touching these means it is resting on the hand rather than being
# held in the fingertips.
# Transcribed link-for-link from upstream: note finger5 (the little finger) contributes
# only link1/link2 -- its link3 is short enough that cube contact there is normal.
UNDESIRED_CONTACT_LINKS = [
    PALM_LINK,
    ".*_finger1_link1",
    ".*_finger[2-4]_link[123]",
    ".*_finger5_link[12]",
]
