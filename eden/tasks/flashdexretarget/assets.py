"""What the task builds from a FlashDexRetarget dataset and robot: the robot entity, its drive parameters, and the
object variants a ``motion.pt`` references.

Read once when the config is built (by :class:`~eden.tasks.flashdexretarget.config.FlashDexRetargetDataMod`), never
per step. The robot's joint order, drive gains and palm frame come from its MJCF — the model the kinematic
retargeting solved on, whose joint order ``motion.pt``'s ``joint_pos`` columns follow — and the simulated robot is the
matching ``*_spec`` URDF.
"""

from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from eden.extensions.retargeting.flashdexretarget.clips import FINGERS
from eden.extensions.retargeting.flashdexretarget.objects import object_urdf_path
from eden.extensions.retargeting.flashdexretarget.scene import ROBOT_ASSET_DIR, robot_xml

SIDES = ("right", "left")
#: The simulated robot file per (robot, sides) under :data:`ROBOT_ASSET_DIR`.
_URDF = {
    ("xhand", ("right", "left")): "xhand/urdf/xhand_bimanual_spec.urdf",
    ("xhand", ("right",)): "xhand/urdf/xhand_right_spec.urdf",
    ("xhand", ("left",)): "xhand/urdf/xhand_left_spec.urdf",
    ("sharpa", ("right", "left")): "sharpa/urdf/sharpa_bimanual.urdf",
}


#: Per robot, the links (side prefix dropped) whose contact with the object is observed, palm first.
_CONTACT_LINKS = {
    "xhand": (
        "hand_thumb_bend_link",
        "hand_thumb_rota_link1",
        "hand_thumb_rota_link2",
        "hand_index_bend_link",
        "hand_index_rota_link1",
        "hand_index_rota_link2",
        "hand_mid_link1",
        "hand_mid_link2",
        "hand_ring_link1",
        "hand_ring_link2",
        "hand_pinky_link1",
        "hand_pinky_link2",
    ),
    "sharpa": tuple(
        f"{f}_{p}"
        for f, parts in (
            ("thumb", ("MC", "PP", "DP")),
            ("index", ("MCP_VL", "PP", "MP", "DP")),
            ("middle", ("MCP_VL", "PP", "MP", "DP")),
            ("ring", ("MCP_VL", "PP", "MP", "DP")),
            ("pinky", ("MC", "MCP_VL", "PP", "MP", "DP")),
        )
        for p in parts
    ),
}
#: Per robot, (robot link without the side prefix, MANO joint) pairs: the hand keypoints the observations compare.
_BODY_MAP = {
    "xhand": (
        ("hand_thumb_bend_link", "thumb_proximal"),
        ("hand_thumb_rota_link1", "thumb_intermediate"),
        ("hand_thumb_rota_link2", "thumb_distal"),
        ("hand_index_bend_link", "index_proximal"),
        ("hand_index_rota_link1", "index_proximal"),
        ("hand_index_rota_link2", "index_distal"),
        ("hand_mid_link1", "middle_proximal"),
        ("hand_mid_link2", "middle_distal"),
        ("hand_ring_link1", "ring_proximal"),
        ("hand_ring_link2", "ring_distal"),
        ("hand_pinky_link1", "pinky_proximal"),
        ("hand_pinky_link2", "pinky_distal"),
    ),
    "sharpa": tuple(
        (f"{f}_{p}", f"{f}_{m}")
        for f, pairs in (
            ("thumb", (("MC", "proximal"), ("PP", "intermediate"), ("DP", "distal"))),
            *(
                (f, (("MCP_VL", "proximal"), ("MP", "intermediate"), ("DP", "distal")))
                for f in ("index", "middle", "ring", "pinky")
            ),
        )
        for p, m in pairs
    ),
}
#: A link touches the object when its origin is within its collision reach plus this skin of the object surface.
CONTACT_SKIN = 0.015


@dataclass(frozen=True)
class RobotSpec:
    """A FlashDexRetarget robot as the task needs it.

    Attributes
    ----------
    file : str
        The URDF the scene loads, relative to the asset store.
    joint_names : tuple[str, ...]
        Actuated joints in MJCF order, the column order of ``motion.pt``'s ``joint_pos``.
    kp, kd, damping, armature, frictionloss, force_limit : dict[str, float]
        Per joint, from the MJCF position actuators and joints.
    limits : dict[str, tuple[float, float]]
        Per joint, the MJCF range: FlashDexRetarget's soft limits (no inward margin).
    palm_link : dict[str, str]
        Per side, the link carrying the palm site.
    palm_quat : dict[str, tuple[float, float, float, float]]
        Per side, the palm site's orientation in that link's frame (wxyz); it sits at the link origin.
    tip_links : dict[str, tuple[str, ...]]
        Per side, the link carrying each fingertip tracking site (``track_hand_<side>_<finger>_tip``), thumb to
        pinky.
    tip_offsets : dict[str, tuple[tuple[float, float, float], ...]]
        Per side, each tracking site's position in its link's frame (from the MJCF: the URDF's own tip frames sit
        elsewhere).
    wrist_trans, wrist_rot, fingers : dict[str, tuple[str, ...]]
        Per side, the forearm slide joints, forearm hinge joints and finger joints.
    contact_links : dict[str, tuple[str, ...]]
        Per side, the links whose contact with the object is observed, palm first.
    contact_margins : dict[str, tuple[float, ...]]
        Per side and contact link, the largest distance (m) of its origin from the object surface at which it can
        touch it: its collision reach (geom offset + bounding radius, from the MJCF) plus :data:`CONTACT_SKIN`.
    body_map : dict[str, tuple[tuple[str, str], ...]]
        Per side, (robot link, MANO joint) pairs compared as hand keypoints.
    """

    file: str
    joint_names: tuple[str, ...]
    kp: dict[str, float]
    kd: dict[str, float]
    damping: dict[str, float]
    armature: dict[str, float]
    frictionloss: dict[str, float]
    force_limit: dict[str, float]
    limits: dict[str, tuple[float, float]]
    palm_link: dict[str, str]
    palm_quat: dict[str, tuple[float, float, float, float]]
    tip_links: dict[str, tuple[str, ...]]
    tip_offsets: dict[str, tuple[tuple[float, float, float], ...]]
    wrist_trans: dict[str, tuple[str, ...]]
    wrist_rot: dict[str, tuple[str, ...]]
    fingers: dict[str, tuple[str, ...]]
    contact_links: dict[str, tuple[str, ...]]
    contact_margins: dict[str, tuple[float, ...]]
    body_map: dict[str, tuple[tuple[str, str], ...]]


def _collision_reach(model) -> dict[str, float]:
    """Per MJCF body, the largest ``|geom offset| + bounding radius`` over its collision geoms."""
    import mujoco

    reach: dict[str, float] = {}
    for g in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if not (model.geom_contype[g] or model.geom_conaffinity[g] or name.startswith("collision")):
            continue
        body = model.body(model.geom_bodyid[g]).name
        reach[body] = max(reach.get(body, 0.0), float(np.linalg.norm(model.geom_pos[g]) + model.geom_rbound[g]))
    return reach


def _urdf_link_names(urdf: Path) -> set[str]:
    import xml.etree.ElementTree as ET

    return {link.get("name") for link in ET.parse(urdf).getroot().iter("link")}


def _urdf_parent_link(urdf_links: set[str], urdf: Path, mjcf_body: str, side: str, *, palm: bool = False) -> str:
    """The URDF name of MJCF body ``mjcf_body``: the same name, except that the converter renames the palm body of
    each hand to ``<side>_hand_link`` (``palm=True``). Any other body must keep its name.
    """
    if mjcf_body in urdf_links:
        return mjcf_body
    if palm and f"{side}_hand_link" in urdf_links:
        return f"{side}_hand_link"
    raise ValueError(
        f"No link of {urdf} matches MJCF body {mjcf_body!r} ({side} hand{', palm' if palm else ''}); its links: "
        f"{sorted(urdf_links)}."
    )


@functools.cache
def robot_spec(robot: str, sides: tuple[str, ...]) -> RobotSpec:
    """The :class:`RobotSpec` of ``robot`` for ``sides`` (``("right", "left")``, ``("right",)`` or ``("left",)``)."""
    import mujoco

    from eden.utils.assets import get_asset_path

    key = (robot, tuple(sides))
    if key not in _URDF:
        raise ValueError(
            f"No FlashDexRetarget robot {robot!r} for hands {list(sides)}; available: "
            f"{sorted(f'{r} {list(s)}' for r, s in _URDF)}."
        )
    urdf = Path(get_asset_path(f"{ROBOT_ASSET_DIR}/{_URDF[key]}"))
    model = mujoco.MjModel.from_xml_path(str(robot_xml(robot, tuple(sides))))
    names = tuple(model.joint(model.actuator_trnid[a, 0]).name for a in range(model.nu))
    order = [model.joint(j).name for j in range(model.njnt) if model.joint(j).name in names]
    if order != list(names):
        raise ValueError(f"{robot}: the MJCF actuators do not follow its joint order; motion.pt assumes they do.")

    kp, kd, damping, armature, frictionloss, force, limits = {}, {}, {}, {}, {}, {}, {}
    for a in range(model.nu):
        j = model.actuator_trnid[a, 0]
        name, dof = model.joint(j).name, model.jnt_dofadr[j]
        kp[name] = float(model.actuator_gainprm[a, 0])
        kd[name] = float(-model.actuator_biasprm[a, 2])
        damping[name] = float(model.dof_damping[dof])
        armature[name] = float(model.dof_armature[dof])
        frictionloss[name] = float(model.dof_frictionloss[dof])
        force[name] = float(np.abs(model.actuator_forcerange[a]).max()) if model.actuator_forcelimited[a] else 1e6
        limits[name] = (float(model.jnt_range[j, 0]), float(model.jnt_range[j, 1]))

    reach = _collision_reach(model)
    urdf_links = _urdf_link_names(urdf)
    palm_link, palm_quat, tip_links, tip_offsets, wrist_trans, wrist_rot, fingers = {}, {}, {}, {}, {}, {}, {}
    contact_links, contact_margins, body_map = {}, {}, {}
    for side in sides:
        site = model.site(f"{side}_palm")
        if np.linalg.norm(site.pos) > 1e-9:
            raise ValueError(f"{robot}: the {side} palm site is off its body's origin ({site.pos}); unsupported.")
        palm_body = model.body(site.bodyid[0]).name
        palm_link[side] = _urdf_parent_link(urdf_links, urdf, palm_body, side, palm=True)
        palm_quat[side] = tuple(float(x) for x in site.quat)
        tips = [model.site(f"track_hand_{side}_{f}_tip") for f in FINGERS]
        tip_links[side] = tuple(_urdf_parent_link(urdf_links, urdf, model.body(t.bodyid[0]).name, side) for t in tips)
        tip_offsets[side] = tuple(tuple(float(x) for x in t.pos) for t in tips)
        pre = "R" if side == "right" else "L"
        wrist_trans[side] = tuple(f"{pre}_forearm_pos_{a}_joint" for a in "xyz")
        wrist_rot[side] = tuple(f"{pre}_forearm_rot_{a}_joint" for a in "zxy")
        side_joints = [n for n in names if n.startswith((f"{side}_", f"{pre}_"))]
        fingers[side] = tuple(n for n in side_joints if n not in wrist_trans[side] + wrist_rot[side])
        links = (palm_body, *(f"{side}_{n}" for n in _CONTACT_LINKS[robot]))
        unreachable = [b for b in links if b not in reach]
        if unreachable:
            raise ValueError(
                f"{robot}: contact links {unreachable} have no collision geom in the MJCF, so no contact reach; the "
                f"bodies with one: {sorted(reach)}. Fix _CONTACT_LINKS[{robot!r}]."
            )
        contact_links[side] = (palm_link[side], *links[1:])
        contact_margins[side] = tuple(reach[b] + CONTACT_SKIN for b in links)
        body_map[side] = tuple((f"{side}_{r}", m) for r, m in _BODY_MAP[robot])
    return RobotSpec(
        file=f"{ROBOT_ASSET_DIR}/{_URDF[key]}",
        joint_names=names,
        kp=kp,
        kd=kd,
        damping=damping,
        armature=armature,
        frictionloss=frictionloss,
        force_limit=force,
        limits=limits,
        palm_link=palm_link,
        palm_quat=palm_quat,
        tip_links=tip_links,
        tip_offsets=tip_offsets,
        wrist_trans=wrist_trans,
        wrist_rot=wrist_rot,
        fingers=fingers,
        contact_links=contact_links,
        contact_margins=contact_margins,
        body_map=body_map,
    )


@dataclass(frozen=True)
class MotionSummary:
    """What the scene needs from a ``motion.pt``: its hands, its object slots and the URDFs of their objects.

    Attributes
    ----------
    path : Path
        The ``motion.pt``.
    sides : tuple[str, ...]
        Its hands, in its order.
    shared : bool
        Every clip has both hands on one object (one object entity), as opposed to one object per hand.
    slot_urdfs : dict[str, list[str]]
        Per object entity (``object`` when shared, else ``object_<side>``), the URDF of every slot's object, in slot
        order.
    slot_labels : list[str]
        One label per slot, the per-clip label ``env_motion_groups`` matches.
    robot : str | None
        The robot it was retargeted to (``motion.pt``'s ``robot``); None for a file from before it carried one.
    fps : float | None
        Its frame rate (``motion.pt``'s ``fps``); None for a file from before it carried one.
    """

    path: Path
    sides: tuple[str, ...]
    shared: bool
    slot_urdfs: dict[str, list[str]]
    slot_labels: list[str]
    robot: str | None = None
    fps: float | None = None

    @property
    def object_entities(self) -> dict[str, str]:
        """Side -> the object entity that side's hand holds."""
        return {s: "object" if self.shared else f"object_{s}" for s in self.sides}


def _clip_shared(packed: dict, starts: torch.Tensor, frames: torch.Tensor, i: int) -> bool:
    a, b = int(starts[i]), int(starts[i] + frames[i])
    same_pos = torch.allclose(packed["obj_right_pos"][a:b], packed["obj_left_pos"][a:b], atol=1e-5)
    same_rot = torch.allclose(packed["obj_right_rotmat"][a:b], packed["obj_left_rotmat"][a:b], atol=1e-5)
    return bool(same_pos and same_rot)


@functools.cache
def motion_summary(motion: str) -> MotionSummary:
    """Read the hands, object slots and object URDFs of ``motion`` (a ``motion.pt``)."""
    path = Path(motion).expanduser().resolve()
    if path.is_dir():
        raise ValueError(f"{path} is a directory; pass the motion file inside it, {path / 'motion.pt'}.")
    if not path.is_file():
        raise FileNotFoundError(
            f"No motion file at {path}. Build one with `eden retarget flashdex kinematic --dataset <dataset> "
            "--robot <robot>` and pass <dataset>/<robot>/motion.pt."
        )
    packed = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    sides = tuple(s for s in SIDES if f"mano_{s}_wrist_pos" in packed)
    if not sides:
        raise ValueError(f"{path} holds no mano_right_* / mano_left_* arrays; is it a FlashDexRetarget motion.pt?")
    shared = False
    if len(sides) == 2:
        starts, frames = packed["length_starts"], packed["motion_num_frames"]
        flags = [_clip_shared(packed, starts, frames, i) for i in range(len(frames))]
        if any(flags) and not all(flags):
            raise ValueError(
                f"{path} mixes clips with both hands on one object ({sum(flags)}) and clips with one object per hand "
                f"({len(flags) - sum(flags)}); the task takes one kind per motion file. Rebuild it from one kind "
                "(e.g. HOT3D's segment modes B or C) or split it by `motion_filename`."
            )
        shared = all(flags)
    entity_sides = sides[:1] if shared else sides
    slot_urdfs, label_parts = {}, []
    for side in entity_sides:
        dirs, scales = packed[f"{side}_object_mesh_dirs"], packed[f"{side}_object_mesh_scales"]
        urdfs = []
        for d, s in zip(dirs, scales):
            # Relative to the motion file's directory: ../objects/<id> for <dataset>/<robot>/motion.pt.
            obj_dir = Path(os.path.normpath(path.parent / d))
            urdf = object_urdf_path(obj_dir, s)
            if not urdf.is_file():
                raise FileNotFoundError(
                    f"{urdf} missing; write the object URDFs with `eden retarget flashdex kinematic --dataset "
                    f"{obj_dir.parent.parent} --robot {packed.get('robot', '<robot>')}` (steps.urdf)."
                )
            urdfs.append(str(urdf))
        slot_urdfs["object" if shared else f"object_{side}"] = urdfs
        label_parts.append([f"{Path(d).name}*{float(s):g}" for d, s in zip(dirs, scales)])
    slot_labels = ["|".join(p) for p in zip(*label_parts)]
    fps = float(packed["fps"]) if "fps" in packed else None
    return MotionSummary(path, sides, shared, slot_urdfs, slot_labels, packed.get("robot"), fps)
