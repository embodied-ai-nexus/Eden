"""Embodiment presets for OTRetarget: G1 and LimX Oli, with or without Wuji hands; SMPL-X or SOMA demonstrators.

A preset states what the paper calls manual pre-assignment (Sec. III-B, D):

* ``correspondences`` — which human body parts each group of robot links is matched to; the transport
  plan is solved per group between the two surface clouds in the T-pose.
* ``style_links`` — the robot links whose world orientation (and FK-derived position) tracks a human
  joint frame, through an alignment offset measured in the T-pose.
* ``tpose`` — the robot joint values that put it in the human body model's rest T-pose (arms along
  the lateral axis, palms down, thumbs forward, fingers extended). It is a reference posture only,
  never executed, so it may lie outside a joint's limits (the G1 elbow does).

Presets name human parts with SMPL-X joints
(:data:`~eden.extensions.retargeting.otretarget.human.SMPLX_JOINT_NAMES`); :func:`for_body_model`
translates one to the SOMA rig (BONES-SEED), whose fingers carry an extra metacarpal joint.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

_HALF_PI = 0.5 * math.pi
_FINGERS = ("index", "middle", "ring", "pinky")
_SIDES = ("left", "right")

Group = tuple[tuple[str, ...], tuple[str, ...]]


@dataclass(frozen=True)
class RigPreset:
    """Everything embodiment-specific the retargeter needs (see the module docstring).

    Attributes
    ----------
    entities : tuple[str, ...]
        Scene entity names of the rig, root first.
    tpose : dict[str, float]
        Joint values of the T-pose; unlisted joints are zero.
    correspondences : tuple[Group, ...]
        ``((human parts...), (robot links...))`` groups.
    style_links : dict[str, str]
        ``{robot link: human joint}``; the first entry is the rig root, aligned to ``human_root``.
    human_root : str
        The human joint the rig root is aligned with.
    robot_probe_density : dict[str, float]
        ``{link regex: points per m^2}``.
    human_probe_density : dict[str, float]
        ``{joint regex: points per m^2}``.
    self_collision_exclude : tuple[tuple[str, str], ...]
        Link-name regex pairs left out of capsule self-collision.
    style_link_weights : dict[str, float]
        Per tracked link multipliers of the style weights (``{link regex: weight}``).
    interaction_weights : tuple[float, float, float], optional
        ``(lambda_d, lambda_x, lambda_n)`` for robot probes; None keeps the retargeter's (the paper's) values.
    joint_limits : dict[str, tuple[float, float]]
        ``{joint regex: (lower, upper)}`` the retargeted motion may use, within the joints' own limits.
    palms : dict[str, str | list[str]]
        ``{robot palm link: human wrist joint}`` (or ``[frame joint, joints...]``, see ``HandGrasp.palms``); a rig
        with palms gets the ``hand_grasp`` add-on
        (:class:`~eden.extensions.retargeting.otretarget.grasp.HandGrasp`).
    fingertips : dict[str, str]
        ``{robot fingertip link: human distal-phalanx joint}``.
    hand_options : dict
        Further ``HandGrasp`` options (``closure``, ``cold_start_palm_weight``, ...).
    """

    entities: tuple[str, ...]
    tpose: dict[str, float]
    correspondences: tuple[Group, ...]
    style_links: dict[str, str]
    human_root: str = "pelvis"
    robot_probe_density: dict[str, float] = field(default_factory=dict)
    human_probe_density: dict[str, float] = field(default_factory=dict)
    self_collision_exclude: tuple[tuple[str, str], ...] = ()
    style_link_weights: dict[str, float] = field(default_factory=dict)
    interaction_weights: tuple[float, float, float] | None = None
    joint_limits: dict[str, tuple[float, float]] = field(default_factory=dict)
    palms: dict[str, str | list[str]] = field(default_factory=dict)
    fingertips: dict[str, str] = field(default_factory=dict)
    hand_options: dict = field(default_factory=dict)

    def retargeter_options(self) -> dict:
        """The preset as :class:`~eden.extensions.retargeting.otretarget.OTRetargeter` ``configure`` keywords: a legged
        rig on the floor (``floor_stance``), and its hands when it has palms (``hand_grasp``).
        """
        addons = {"floor_stance": {}}
        if self.palms:
            addons["hand_grasp"] = dict(palms=dict(self.palms), fingertips=dict(self.fingertips), **self.hand_options)
        options = dict(
            entity_name=self.entities[0],
            hand_entity_names=list(self.entities[1:]),
            correspondences=[[list(h), list(r)] for h, r in self.correspondences],
            style_links=dict(self.style_links),
            human_root_joint=self.human_root,
            tpose=dict(self.tpose),
            robot_probe_density=dict(self.robot_probe_density),
            human_probe_density=dict(self.human_probe_density),
            self_collision_exclude=[list(p) for p in self.self_collision_exclude],
            style_link_weights=dict(self.style_link_weights),
            joint_limits={k: list(v) for k, v in self.joint_limits.items()},
            addons=addons,
        )
        if self.interaction_weights is not None:
            options["interaction_weights"] = list(self.interaction_weights)
        return options


# -- probe densities: the paper's on the body, raised on the hands -----------------------------
#: 2000 pts/m^2 (Sec. III-B) would leave a phalanx one or two probes, too few for a per-segment
#: transport plan, so hands and fingers are sampled denser on both sides.
_HAND_DENSITY = 40000.0
_HUMAN_DENSITY = {r"(left|right)_(wrist|index\d|middle\d|ring\d|pinky\d|thumb\d)": _HAND_DENSITY, r".*": 2000.0}
#: The forearm's wrist links touch what the hand holds as often as the palm does; at the body's 2000 pts/m^2 a
#: thin pole sinks 6 mm into them between probes.
_WRIST_DENSITY = 10000.0
_WUJI_DENSITY = {r"[lr]_.*": _HAND_DENSITY, r".*_wrist_.*": _WRIST_DENSITY, r".*": 2000.0}
_BODY_DENSITY = {r".*": 2000.0}

# -- shared groups ---------------------------------------------------------------------------
_HEAD = ("neck", "head", "jaw", "left_eye", "right_eye")
_TORSO = ("spine1", "spine2", "spine3", "left_collar", "right_collar")


def _fingers(side: str) -> tuple[str, ...]:
    return tuple(f"{side}_{f}{i}" for f in (*_FINGERS, "thumb") for i in (1, 2, 3))


def _legs() -> tuple[Group, ...]:
    groups = []
    for s in _SIDES:
        groups += [
            ((f"{s}_hip",), (f"{s}_hip_pitch_link", f"{s}_hip_roll_link", f"{s}_hip_yaw_link")),
            ((f"{s}_knee",), (f"{s}_knee_link",)),
            ((f"{s}_ankle", f"{s}_foot"), (f"{s}_ankle_pitch_link", f"{s}_ankle_roll_link")),
        ]
    return tuple(groups)


_LEG_STYLE = {
    f"{s}_{link}": f"{s}_{joint}"
    for s in _SIDES
    for link, joint in (("hip_yaw_link", "hip"), ("knee_link", "knee"), ("ankle_roll_link", "ankle"))
}


def _wuji_hand(side: str, prefix: str) -> tuple[Group, ...]:
    """Finger groups of one Wuji Hand2 beta2: proximal (+abduction link), middle, distal (+tip) phalanges."""
    groups = [
        ((f"{side}_thumb1",), (f"{prefix}_thumb_proximal", f"{prefix}_thumb_proximal_abd")),
        ((f"{side}_thumb2",), (f"{prefix}_thumb_middle",)),
        ((f"{side}_thumb3",), (f"{prefix}_thumb_distal", f"{prefix}_thumb_tip")),
    ]
    for f in _FINGERS:
        link = f"{prefix}_{f}_finger" if f != "pinky" else f"{prefix}_pinky"
        groups += [
            ((f"{side}_{f}1",), (f"{link}_proximal", f"{link}_proximal_abd")),
            ((f"{side}_{f}2",), (f"{link}_middle",)),
            ((f"{side}_{f}3",), (f"{link}_distal", f"{link}_tip")),
        ]
    return tuple(groups)


def _wuji_style(side: str, prefix: str) -> dict[str, str]:
    style = {
        f"{prefix}_thumb_proximal_abd": f"{side}_thumb1",
        f"{prefix}_thumb_middle": f"{side}_thumb2",
        f"{prefix}_thumb_distal": f"{side}_thumb3",
    }
    for f in _FINGERS:
        link = f"{prefix}_{f}_finger" if f != "pinky" else f"{prefix}_pinky"
        style |= {
            f"{link}_proximal_abd": f"{side}_{f}1",
            f"{link}_middle": f"{side}_{f}2",
            f"{link}_distal": f"{side}_{f}3",
        }
    return style


# -- G1 ------------------------------------------------------------------------------------------
_G1_TPOSE = {
    "left_shoulder_roll_joint": _HALF_PI,
    "right_shoulder_roll_joint": -_HALF_PI,
    # The G1 elbow is bent 90 deg at zero; +90 deg straightens it (beyond its limit, see module doc).
    "left_elbow_joint": _HALF_PI,
    "right_elbow_joint": _HALF_PI,
}


def _g1_groups(wuji: bool) -> tuple[Group, ...]:
    groups: list[Group] = [
        (("pelvis",), ("pelvis",)),
        ((*_TORSO, *_HEAD), ("waist_yaw_link", "waist_roll_link", "torso_link")),
        *_legs(),
    ]
    for s in _SIDES:
        groups += [
            ((f"{s}_shoulder",), (f"{s}_shoulder_pitch_link", f"{s}_shoulder_roll_link", f"{s}_shoulder_yaw_link")),
            ((f"{s}_elbow",), (f"{s}_elbow_link", f"{s}_wrist_roll_link", f"{s}_wrist_pitch_link")),
        ]
        if wuji:
            # The human palm onto the Wuji palm alone; the forearm-end wrist link and the adapter mount go with the
            # forearm. Each group's two clouds are normalised to unit size on their own: with those two links in it, the
            # 10 cm human palm was spread over 17 cm of robot starting 7-9 cm up the forearm. The mount took the palm
            # centre's targets, the forearm end the heel's, the Wuji palm only the knuckle line, and a pole held across
            # the palm was retargeted onto the mount with the fingers curled on air.
            groups[-1] = (groups[-1][0], (*groups[-1][1], f"{s}_wrist_yaw_link", f"{s[0]}_mount"))
            groups.append(((f"{s}_wrist",), (f"{s[0]}_wrist",)))
            groups += _wuji_hand(s, s[0])
        else:
            # The moulded G1 hand rides on the last wrist link: it takes the whole human hand.
            groups.append(((f"{s}_wrist", *_fingers(s)), (f"{s}_wrist_yaw_link",)))
    return tuple(groups)


def _g1_style(wuji: bool) -> dict[str, str]:
    style = {"pelvis": "pelvis", "torso_link": "spine3", **_LEG_STYLE}
    for s in _SIDES:
        style |= {
            f"{s}_shoulder_yaw_link": f"{s}_shoulder",
            f"{s}_wrist_roll_link": f"{s}_elbow",
            f"{s}_wrist_yaw_link": f"{s}_wrist",
        }
    if wuji:
        style |= _wuji_style("left", "l") | _wuji_style("right", "r")
    return style


# -- Oli -----------------------------------------------------------------------------------------
_OLI_TPOSE = {"left_shoulder_roll_joint": _HALF_PI, "right_shoulder_roll_joint": -_HALF_PI}


def _oli_groups(wuji: bool) -> tuple[Group, ...]:
    groups: list[Group] = [
        (("pelvis",), ("base_link",)),
        (_TORSO, ("waist_yaw_link", "waist_roll_link", "waist_pitch_link")),
        (_HEAD, ("head_yaw_link", "head_pitch_link")),
        *_legs(),
    ]
    for s in _SIDES:
        groups += [
            ((f"{s}_shoulder",), (f"{s}_shoulder_pitch_link", f"{s}_shoulder_roll_link", f"{s}_shoulder_yaw_link")),
            ((f"{s}_elbow",), (f"{s}_elbow_link", f"{s}_wrist_yaw_link", f"{s}_wrist_pitch_link")),
        ]
        if wuji:
            # The human palm onto the Wuji palm alone (see _g1_groups).
            groups[-1] = (groups[-1][0], (*groups[-1][1], f"{s}_wrist_roll_link", f"{s}_hand_mount", f"{s[0]}_mount"))
            groups.append(((f"{s}_wrist",), (f"{s[0]}_wrist",)))
            groups += _wuji_hand(s, s[0])
        else:
            # The moulded Oli hand rides on the last wrist link: it takes the whole human hand.
            groups.append(((f"{s}_wrist", *_fingers(s)), (f"{s}_wrist_roll_link",)))
    return tuple(groups)


def _oli_style(wuji: bool) -> dict[str, str]:
    style = {"base_link": "pelvis", "waist_pitch_link": "spine3", "head_pitch_link": "head", **_LEG_STYLE}
    for s in _SIDES:
        style |= {
            f"{s}_shoulder_yaw_link": f"{s}_shoulder",
            f"{s}_wrist_yaw_link": f"{s}_elbow",
            f"{s}_wrist_roll_link": f"{s}_wrist",
        }
    if wuji:
        style |= _wuji_style("left", "l") | _wuji_style("right", "r")
    return style


_WUJI_RIG = ("robot", "left_hand", "right_hand")


def _wuji_tips() -> dict[str, str]:
    tips = {}
    for side in _SIDES:
        p = side[0]
        for f in ("thumb", *_FINGERS):
            link = f"{p}_thumb_tip" if f == "thumb" else (f"{p}_pinky_tip" if f == "pinky" else f"{p}_{f}_finger_tip")
            tips[link] = f"{side}_{f}3"
    return tips


_WUJI_PALMS = {"l_wrist": "left_wrist", "r_wrist": "right_wrist"}
#: Grasp closure and a side-correct cold start (``HandGrasp.closure``, ``cold_start_palm_weight``): the
#: OMOMO hands hold poles loosely with the pole sunk into the palm, so copied, the robot fingers hovered 2-4 cm off and
#: the thumb stayed on the fingers' side of the pole (round a tripod leg, 20-50 deg from the other contacts; human
#: 110-160 deg). With closure, contacts reach half way round a thin part in 58-60% of the frames where the human's
#: do (26-30% without). The palm-pulled second cold start halves the G1 frames holding an object behind the hand
#: (12% -> 6%). The side re-approach (``side_reapproach``) brings a hand round a pole that slid off its curled
#: fingertips onto its back: on 65 OMOMO clips with such hands, wrong-side contact frames 0.28 -> 0.26 (G1),
#: 0.31 -> 0.26 (Oli).
_WUJI_HAND_OPTIONS = {
    "closure": 0.06,
    "closure_weight": 4000.0,
    "cold_start_palm_weight": 1000.0,
    "side_reapproach": True,
}
#: Side-by-side Wuji fingers (index/middle/ring/pinky) of one hand: their phalanx capsules overlap as soon as
#: the fingers curl, which would lock the hand open. Thumb-finger, hand-body and hand-hand pairs stay checked.
_WUJI_FINGERS = r"(index_finger|middle_finger|ring_finger|pinky)"
_WUJI_SELF_COLLISION_EXCLUDE = (
    (rf"l_{_WUJI_FINGERS}_.*", rf"l_{_WUJI_FINGERS}_.*"),
    (rf"r_{_WUJI_FINGERS}_.*", rf"r_{_WUJI_FINGERS}_.*"),
)

#: Wuji phalanges track the human finger orientations at 0.3 of the body links' style weight. The style cost
#: prices an angle the same on every link, but turning a 3 cm phalanx by 30 deg moves its tip 1.5 cm where a
#: thigh's knee moves 20 cm: at full weight the demonstrated finger curl outweighs the palm's contact with the
#: object, and on a robot hand of other proportions the fingertips stop on a pole the human held in the palm
#: (tripod: robot palm 3.5 cm off the leg; at 0.25 the hand depth error halves). Much lower and released fingers
#: barely reopen: at 0.1 an Oli hand stayed curled round a wooden chair's backrest after the human let go and
#: carried the chair 45 cm off its demonstrated path (3 cm at 0.3).
_WUJI_STYLE_WEIGHTS = {r"[lr]_(thumb|index_finger|middle_finger|ring_finger|pinky)_.*": 0.3}

#: ``(lambda_d, lambda_x, lambda_n)`` with Wuji hands: twice the paper's depth weight (256). A Wuji hand is larger
#: than a human hand, so the demonstrated witnesses and contact planes of the human palm and fingers cannot all be met
#: at once round a thin object; at the paper's balance those placement terms held the hand 2-4 cm off a 1.7 cm tripod
#: leg (the depth pull on the elbow, -3.4, cancelled by witness +1.75 and normal +1.45). Over 36 OMOMO runs on both
#: rigs, x2 raised hand Jaccard 0.74 -> 0.80 (G1) and 0.71 -> 0.78 (Oli) and cut the hand depth error 6.6 -> 5.1
#: and 7.8 -> 5.6 mm, with placement error, penetration and posture unchanged.
_WUJI_INTERACTION_WEIGHTS = (512.0, 625.0, 2500.0)

#: Wuji finger and thumb flexion stops 0.25 rad (14 deg) backwards, where the OMOMO demonstrators' fingers stop (1st
#: percentile of their MCP and PIP bends; the extreme is 28 deg). The hand's own range is 60 deg backwards, and pressed
#: onto an object by the contact terms the fingers used it: on 40% of runs a finger bent back more than 20 deg in over a
#: fifth of the frames.
_WUJI_JOINT_LIMITS = {
    r"[lr]_(index_finger|middle_finger|ring_finger|pinky)_(mcp_flex|pip|dip)": (-0.25, 10.0),
    r"[lr]_thumb_(mcp|ip)": (-0.25, 10.0),
}

#: Unitree G1 29-DOF + Wuji hands (:class:`~eden.options.robots.composite.G1Wuji`).
G1_WUJI = RigPreset(
    _WUJI_RIG,
    _G1_TPOSE,
    _g1_groups(True),
    _g1_style(True),
    "pelvis",
    _WUJI_DENSITY,
    _HUMAN_DENSITY,
    _WUJI_SELF_COLLISION_EXCLUDE,
    _WUJI_STYLE_WEIGHTS,
    _WUJI_INTERACTION_WEIGHTS,
    _WUJI_JOINT_LIMITS,
    _WUJI_PALMS,
    _wuji_tips(),
    _WUJI_HAND_OPTIONS,
)
#: LimX Oli + Wuji hands (:class:`~eden.options.robots.composite.OliWuji`).
OLI_WUJI = RigPreset(
    _WUJI_RIG,
    _OLI_TPOSE,
    _oli_groups(True),
    _oli_style(True),
    "pelvis",
    _WUJI_DENSITY,
    _HUMAN_DENSITY,
    _WUJI_SELF_COLLISION_EXCLUDE,
    _WUJI_STYLE_WEIGHTS,
    _WUJI_INTERACTION_WEIGHTS,
    _WUJI_JOINT_LIMITS,
    _WUJI_PALMS,
    _wuji_tips(),
    _WUJI_HAND_OPTIONS,
)
#: Unitree G1 29-DOF with its moulded hands (``G1Dofs29``), body-only retargeting.
G1 = RigPreset(("robot",), _G1_TPOSE, _g1_groups(False), _g1_style(False), "pelvis", _BODY_DENSITY, _BODY_DENSITY)
#: LimX Oli with its moulded hands (``Oli``), body-only retargeting.
OLI = RigPreset(("robot",), _OLI_TPOSE, _oli_groups(False), _oli_style(False), "pelvis", _BODY_DENSITY, _BODY_DENSITY)

#: Presets by name.
PRESETS: dict[str, RigPreset] = {"g1_wuji": G1_WUJI, "oli_wuji": OLI_WUJI, "g1": G1, "oli": OLI}


# -- SOMA rig (BONES-SEED) -------------------------------------------------------------------------
def _soma(side: str, name: str) -> str:
    return ("Left" if side == "left" else "Right") + name


def _smplx_to_soma_parts() -> dict[str, tuple[str, ...]]:
    """SMPL-X part -> the SOMA joints whose skin covers the same surface."""
    parts: dict[str, tuple[str, ...]] = {
        "pelvis": ("Root", "Hips"),
        "spine1": ("Spine1",),
        "spine2": ("Spine2",),
        "spine3": ("Chest",),
        "neck": ("Neck1", "Neck2"),
        "head": ("Head", "HeadEnd"),
        "jaw": ("Jaw",),
        "left_eye": ("LeftEye",),
        "right_eye": ("RightEye",),
    }
    for s in _SIDES:
        parts |= {
            f"{s}_collar": (_soma(s, "Shoulder"),),
            f"{s}_shoulder": (_soma(s, "Arm"),),
            f"{s}_elbow": (_soma(s, "ForeArm"),),
            # SOMA's finger chains start with a metacarpal joint, which skins the palm.
            f"{s}_wrist": (_soma(s, "Hand"), *(_soma(s, f"Hand{f.capitalize()}1") for f in _FINGERS)),
            f"{s}_hip": (_soma(s, "Leg"),),
            f"{s}_knee": (_soma(s, "Shin"),),
            f"{s}_ankle": (_soma(s, "Foot"),),
            f"{s}_foot": (_soma(s, "ToeBase"), _soma(s, "ToeEnd")),
            f"{s}_thumb1": (_soma(s, "HandThumb1"),),
            f"{s}_thumb2": (_soma(s, "HandThumb2"),),
            f"{s}_thumb3": (_soma(s, "HandThumb3"), _soma(s, "HandThumbEnd")),
        }
        for f in _FINGERS:
            c = f.capitalize()
            parts |= {
                f"{s}_{f}1": (_soma(s, f"Hand{c}2"),),
                f"{s}_{f}2": (_soma(s, f"Hand{c}3"),),
                f"{s}_{f}3": (_soma(s, f"Hand{c}4"), _soma(s, f"Hand{c}End")),
            }
    return parts


SMPLX_TO_SOMA_PARTS = _smplx_to_soma_parts()
#: SMPL-X joint -> the SOMA joint whose frame plays its role in the style cost (the first of its parts; ``Hips`` for
#: the pelvis).
SMPLX_TO_SOMA_JOINT = {k: (v[1] if k == "pelvis" else v[0]) for k, v in SMPLX_TO_SOMA_PARTS.items()}


def for_body_model(preset: RigPreset, body_model: str) -> RigPreset:
    """The preset with its human side expressed in another body model's joints.

    Parameters
    ----------
    preset : RigPreset
        A preset in SMPL-X joint names.
    body_model : {"smplx", "soma"}
        Target body model.
    """
    if body_model == "smplx":
        return preset
    if body_model != "soma":
        raise ValueError(f"for_body_model: body_model={body_model!r}; expected 'smplx' or 'soma'.")
    groups = tuple(
        (tuple(j for p in human for j in SMPLX_TO_SOMA_PARTS[p]), robot) for human, robot in preset.correspondences
    )
    style = {link: SMPLX_TO_SOMA_JOINT[joint] for link, joint in preset.style_links.items()}
    dense = any(d > 2000.0 for d in preset.human_probe_density.values())
    density = {r"(Left|Right)Hand.*": _HAND_DENSITY, r".*": 2000.0} if dense else dict(_BODY_DENSITY)
    palms = {link: list(SMPLX_TO_SOMA_PARTS[joint]) for link, joint in preset.palms.items()}  # the hand + metacarpals
    tips = {link: SMPLX_TO_SOMA_JOINT[joint] for link, joint in preset.fingertips.items()}
    return replace(
        preset,
        correspondences=groups,
        style_links=style,
        human_root="Hips",
        human_probe_density=density,
        palms=palms,
        fingertips=tips,
    )
