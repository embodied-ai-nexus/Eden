"""The IK model: the robot MJCF with collisions only through explicit pairs, the floor and support disks, and one
free body per object built from its convex parts.
"""

from __future__ import annotations
from typing import TYPE_CHECKING

import hashlib
from pathlib import Path

import mujoco
import numpy as np

from eden.extensions.retargeting.flashdexretarget.convex import convex_parts
from eden.utils.assets import get_asset_path

if TYPE_CHECKING:
    from eden.options.extensions.flashdexretarget import FDRIKSceneOptions

#: Robot MJCFs (``bimanual.xml``, ``right.xml``, ``left.xml``) under the asset store, one folder per robot.
ROBOT_ASSET_DIR = "hands/flashdexretarget"


def robot_xml(robot: str, sides: tuple[str, ...]) -> Path:
    """The robot's MJCF for ``sides`` (``bimanual.xml`` for both hands), fetched with its meshes when missing."""
    name = "bimanual.xml" if len(sides) == 2 else f"{sides[0]}.xml"
    return Path(get_asset_path(f"{ROBOT_ASSET_DIR}/{robot}/{name}"))


def _mesh_files(spec: mujoco.MjSpec, xml: Path) -> list[Path]:
    """The mesh files ``spec`` (loaded from ``xml``) reads, resolved."""
    meshdir = xml.parent / (spec.meshdir or "")
    return [Path(m.file) if Path(m.file).is_absolute() else (meshdir / m.file).resolve() for m in spec.meshes]


def robot_sha1(robot: str, sides: tuple[str, ...]) -> str:
    """Digest of the robot model the IK builds for ``sides``: its MJCF and every mesh it reads (an updated asset
    redoes the clips retargeted onto the old one).
    """
    xml = robot_xml(robot, sides)
    h = hashlib.sha1(xml.read_bytes())
    for path in sorted(set(_mesh_files(mujoco.MjSpec.from_file(str(xml)), xml))):
        h.update(path.name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def name_id(m: mujoco.MjModel, objtype: mujoco.mjtObj, name: str, where: str | Path) -> int:
    """``mj_name2id`` that raises on a missing name (MuJoCo returns -1, which numpy reads as the last element)."""
    i = mujoco.mj_name2id(m, objtype, name)
    if i < 0:
        kind = objtype.name.removeprefix("mjOBJ_").lower()
        count = {"body": m.nbody, "joint": m.njnt, "site": m.nsite, "geom": m.ngeom}[kind]
        names = [n for n in (mujoco.mj_id2name(m, objtype, k) for k in range(count)) if n]
        raise ValueError(
            f"No {kind} named {name!r} in the IK model of {where}; its {kind}s are {names}. The pipeline expects "
            f"FlashDexRetarget's robot MJCFs (hands/flashdexretarget/<robot>/), whose names it reads."
        )
    return i


def _ik_dynamics(spec: mujoco.MjSpec, cfg: FDRIKSceneOptions) -> None:
    """Heavy joints, no damping, critically damped stiff PD: kv = 2 sqrt(kp M_ii) at qpos0."""
    for j in spec.joints:
        j.armature = cfg.armature
        j.damping = np.zeros_like(j.damping)  # (3,) polynomial damping in MuJoCo >= 3.10
        j.actfrclimited = 0
    model = spec.compile()
    for a, act in enumerate(spec.actuators):
        jid = model.actuator_trnid[a, 0]
        kp = cfg.kp_wrist if "forearm" in model.joint(jid).name else cfg.kp_finger
        kv = 2.0 * np.sqrt(kp * model.dof_M0[model.jnt_dofadr[jid]])  # M_ii at qpos0, armature included
        act.gainprm[0] = kp
        act.biasprm[0], act.biasprm[1], act.biasprm[2] = 0.0, -kp, -kv
        act.forcelimited = 0


def build(
    robot: str,
    sides: tuple[str, ...],
    objects: dict[str, tuple[str, float]],
    pool: Path,
    cfg: FDRIKSceneOptions,
    support_disks: np.ndarray | None = None,
    shared: bool = False,
) -> mujoco.MjSpec:
    """The IK model's spec.

    Parameters
    ----------
    robot : str
        Robot folder under :data:`ROBOT_ASSET_DIR`.
    sides : tuple[str, ...]
        Hands of the clip.
    objects : dict[str, tuple[str, float]]
        Side -> (object id, mesh scale).
    pool : Path
        The dataset directory holding ``objects/<id>/convex/<i>.obj``.
    cfg : FDRIKSceneOptions
        Model settings.
    support_disks : np.ndarray | None
        (K, 5) disks ``[x, y, z_top, r, h]``.
    shared : bool
        Both hands hold one object (no object-object pairs).

    Returns
    -------
    mujoco.MjSpec
        The spec, not compiled.
    """
    xml = robot_xml(robot, sides)
    spec = mujoco.MjSpec.from_file(str(xml))
    for mesh, path in zip(spec.meshes, _mesh_files(spec, xml)):
        mesh.file = str(path)
    spec.meshdir = ""
    _ik_dynamics(spec, cfg)

    spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05])
    statics = ["floor"]
    for k, (x, y, z_top, r, h) in enumerate(
        np.asarray(support_disks if support_disks is not None else np.zeros((0, 5))).reshape(-1, 5)
    ):
        h = max(float(h), cfg.desk_depth)
        spec.worldbody.add_geom(
            name=f"support_{k}", type=mujoco.mjtGeom.mjGEOM_CYLINDER, size=[r, h / 2, 0], pos=[x, y, z_top - h / 2]
        )
        statics.append(f"support_{k}")

    obj_geoms: dict[str, list[str]] = {}
    for side in sides:
        oid, scale = objects[side]
        parts = convex_parts(Path(pool) / "objects" / oid)
        if not parts:
            raise FileNotFoundError(
                f"No convex parts in {Path(pool) / 'objects' / oid / 'convex'}; run the pipeline's convex step "
                "(steps.convex=true) first."
            )
        body = spec.worldbody.add_body(name=f"{side}_object")
        body.add_joint(
            name=f"{side}_object_joint",
            type=mujoco.mjtJoint.mjJNT_FREE,
            armature=cfg.object_armature,
            frictionloss=cfg.object_frictionloss,
        )
        obj_geoms[side] = []
        for p in parts:
            spec.add_mesh(name=f"{side}_{p.stem}", file=str(p.resolve()), scale=[scale] * 3)
            body.add_geom(
                name=f"{side}_object_{p.stem}",
                type=mujoco.mjtGeom.mjGEOM_MESH,
                meshname=f"{side}_{p.stem}",
                contype=0,
                conaffinity=0,
                density=cfg.object_density,
            )
            obj_geoms[side].append(f"{side}_object_{p.stem}")
        body.add_site(name=f"{side}_object", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.01, 0.02, 0.03])

    def pair(a: str, b: str) -> None:
        # solref and friction come from retarget.contact: the IK model overrides every contact's (mjENBL_OVERRIDE).
        spec.add_pair(name=f"{a}_{b}", geomname1=a, geomname2=b, condim=3)

    hand = [g.name for g in spec.geoms if g.name.startswith("collision_hand_")]
    if not hand:
        raise ValueError(
            f"{xml} has no geom named collision_hand_*, so the hands would collide with nothing; its geoms are "
            f"{[g.name for g in spec.geoms if g.name][:20]}. The pipeline expects FlashDexRetarget's robot MJCFs."
        )
    for h in hand:
        for s in statics:
            pair(s, h)
    for geoms in obj_geoms.values():
        for g in geoms:
            for h in hand + statics:
                pair(h, g)
    if len(sides) == 2 and not shared:
        for a in obj_geoms["right"]:
            for b in obj_geoms["left"]:
                pair(a, b)
    # Fingertip-link self-collision ("_0" links): across the hands and within each hand.
    done: set[tuple[str, str]] = set()
    for a in (h for h in hand if "0" in h):
        side = "right" if "right" in a else "left"
        for b in hand:
            other = "left" if side == "right" else "right"
            cross = len(sides) == 2 and other in b and ("0" in b or "1" in b)
            same = side in b and "0" in b
            if b != a and (cross or same) and (a, b) not in done and (b, a) not in done:
                pair(a, b)
                done.add((a, b))
    return spec
