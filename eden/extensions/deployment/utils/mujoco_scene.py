"""The built env read as MuJoCo data: what the MuJoCo sim-to-sim backends carry over from Genesis.

Every reader here takes the built env (or one of its entities) and returns plain data -- a pose, a
:class:`~eden.extensions.deployment.utils.mujoco_sim.MujocoFloor`, the
:class:`~eden.extensions.deployment.utils.mujoco_rig.RigPart` / ``RigProp`` descriptions -- read off
the built solver, so a task's startup events are in it. The engine wrappers
(:mod:`~eden.extensions.deployment.utils.mujoco_sim`, :mod:`~eden.extensions.deployment.utils.mujoco_rig`)
stay free of the env, and the deployments that join the two keep only their lifecycle and checks.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np
import torch

import eden as en
from eden.extensions.deployment.utils.mujoco_rig import RigPart, RigProp, root_body_name
from eden.extensions.deployment.utils.mujoco_sim import MujocoFloor
from eden.options.entities import BoxOptions, PlaneOptions, SphereOptions, TerrainOptions
from eden.options.materials import KinematicMaterialOptions

if TYPE_CHECKING:
    from eden.envs.base import EnvBase
    from eden.options.entities import CompositeEntityOptions, RobotOptions


def f64(tensor) -> np.ndarray:
    return tensor.detach().cpu().double().numpy()


def is_kinematic(options) -> bool:
    """Whether a scene entity is kinematic: moved by the env (a goal marker), never simulated, no contacts."""
    return isinstance(getattr(options, "material", None), KinematicMaterialOptions)


def entity_pose(env: EnvBase, name: str) -> tuple[np.ndarray, np.ndarray]:
    """The authored-frame world pose of entity ``name`` in env 0 -- the frame its asset's root body is in MuJoCo."""
    entity = env.entities[name]
    return f64(entity.get_pos(relative=True)[0]), f64(entity.get_quat(relative=True)[0])


def link_pose(sim, link: str) -> tuple[torch.Tensor, torch.Tensor]:
    """The world pose of a sim entity's link ``link`` in env 0, as float64 CPU tensors."""
    _, idx = sim.find_named_links_idx_local([link])
    pos = sim.get_links_pos(ls_idx_local=idx)[0, 0]
    quat = sim.get_links_quat(ls_idx_local=idx)[0, 0]
    return pos.double().cpu(), quat.double().cpu()


def geoms_sol_params(geoms) -> np.ndarray:
    """The contact parameters Genesis simulates each of ``geoms`` with, one row per geom: ``(n, 7)``.

    Read off the built solver, so a ``set_contact_sol_params`` startup event is in them, and as
    each geom states them: Genesis floors the time constant only on a contact's mix of two geoms
    (``collider/contact.py``), which MuJoCo does too (``MujocoRig._apply_contacts``), so no floor is
    applied here.
    """
    return np.stack([f64(geom.get_sol_params()).reshape(-1) for geom in geoms])


def read_floor(env: EnvBase) -> MujocoFloor:
    """The floor MuJoCo adds, standing for the scene's ground as the env built it.

    A ground plane gives it its pose and its friction and solver parameters. A scene without one
    keeps MuJoCo's plain floor at ``z = 0``, and so does one with several, which cannot all be one
    floor; a terrain lends it its contact but not its shape, which a plane cannot be. Both shortfalls
    are the floor's :attr:`~eden.extensions.deployment.utils.mujoco_sim.MujocoFloor.caveat`, said by
    the deployment once the floor is in the model -- not here, as an asset with a plane of its own
    keeps that one and the floor read here goes unused.
    """
    from eden.utils.geom import transform_by_quat

    scene = env.config.scene_options
    grounds = [
        key
        for key in scene.keys()
        if isinstance(getattr(scene, key, None), (PlaneOptions, TerrainOptions)) and key in env.entities
    ]
    if len(grounds) > 1:
        return MujocoFloor(
            caveat=f"the scene has {len(grounds)} grounds ({grounds}) and MuJoCo one floor, its plain one at z = 0 "
            "with its default contact."
        )
    if not grounds:
        return MujocoFloor()
    (name,) = grounds
    geoms = env.entities[name]._entity.geoms
    if not geoms:  # a ground built without collision
        return MujocoFloor()
    if len(geoms) != 1:
        raise NotImplementedError(
            f"The ground '{name}' built {len(geoms)} collision geoms, and the MuJoCo floor stands for one: Genesis "
            "has built a plane or a terrain's height field as one geom so far, and read_floor (deployment/utils/"
            "mujoco_scene.py) knows no other. Deploying without the ground's pose and contact is not a fix."
        )
    (ground,) = geoms
    friction = float(f64(ground.get_friction()))
    sol_params = tuple(geoms_sol_params([ground])[0].tolist())
    if not isinstance(getattr(scene, name), PlaneOptions):
        return MujocoFloor(
            friction=friction,
            sol_params=sol_params,
            caveat=f"the ground '{name}' is a terrain, and the MuJoCo floor a flat plane at z = 0, without the "
            "terrain's shape.",
        )
    pos = f64(ground.get_pos(relative=False)).reshape(-1, 3)[0]
    quat = f64(ground.get_quat(relative=False)).reshape(-1, 4)[0]
    # Genesis' plane faces the normal its geom data states, in the geom's frame.
    normal = transform_by_quat(np.asarray(ground.data, dtype=np.float64)[None, :3], quat)[0]
    return MujocoFloor(pos=tuple(pos.tolist()), normal=tuple(normal.tolist()), friction=friction, sol_params=sol_params)


def say_floor_caveat(floor: MujocoFloor | None) -> None:
    """Warn how the floor in the model falls short of the scene's ground, if it does (its ``caveat``).

    The deployments call it with the floor their compiled model holds -- ``None`` when the asset kept
    a plane of its own -- so a floor read but never added says nothing.
    """
    if floor is not None and floor.caveat:
        en.logger.warning(f"[deploy] MuJoCo: {floor.caveat}")


def kinematic_markers(env: EnvBase) -> list[tuple[str, Any]]:
    """The scene's kinematic spheres and boxes (goal markers), which a MuJoCo recording draws at their env poses."""
    scene = env.config.scene_options
    return [
        (key, getattr(scene, key))
        for key in scene.keys()
        if key in env.entities
        and isinstance(getattr(scene, key), (SphereOptions, BoxOptions))
        and is_kinematic(getattr(scene, key))
    ]


def marker_poses(env: EnvBase, markers: list[tuple[str, Any]]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Where the env has each of ``markers`` (from :func:`kinematic_markers`) now: what a recording draws this tick."""
    return {name: entity_pose(env, name) for name, _ in markers}


def unsupported_prop(entity, options) -> str | None:
    """Why the env entity ``entity`` (built from ``options``) cannot be a MuJoCo prop, or ``None`` when it can."""
    import genesis as gs

    if is_kinematic(options):
        return "kinematic (a marker the env moves; nothing to simulate)"
    if not isinstance(options, BoxOptions):
        file = getattr(options, "file", None) or ""
        if not file.lower().endswith((".urdf", ".xml")):
            kind = f"a mesh file ({file})" if file else type(options).__name__
            return f"{kind}, not a box or a URDF / MJCF body"
    joints = [
        joint.name for joint in entity._entity.joints if joint.type not in (gs.JOINT_TYPE.FREE, gs.JOINT_TYPE.FIXED)
    ]
    if joints:
        return f"articulated by joints {joints}, whose positions would not reach the env"
    return None


def prop_inertial(entity, root_pos: np.ndarray, root_quat: np.ndarray):
    """An entity's whole rigid-body inertia as the env built it, in the frame of its asset's root body.

    Summed over its links in the world, each link's own (Genesis' ``mass`` / ``inertial_pos`` /
    ``inertial_quat`` / ``inertia``, in that link's frame), then expressed in the root body's
    authored frame -- the frame the MuJoCo body carrying it is in, whatever frame Genesis
    re-origins a free body's links onto.
    """
    from genesis.utils.geom import quat_to_R

    links_pos = f64(entity.get_links_pos())[0]
    links_quat = f64(entity.get_links_quat())[0]
    masses, coms, tensors = [], [], []
    for link, pos, quat in zip(entity._entity.links, links_pos, links_quat, strict=True):
        desc = link.desc
        rotation = quat_to_R(quat)
        masses.append(float(desc.mass))
        coms.append(pos + rotation @ np.asarray(desc.inertial_pos, dtype=np.float64))
        frame = rotation @ quat_to_R(np.asarray(desc.inertial_quat, dtype=np.float64))
        tensors.append(frame @ np.asarray(desc.inertia, dtype=np.float64) @ frame.T)
    mass = sum(masses)
    com = sum(m * c for m, c in zip(masses, coms, strict=True)) / mass
    inertia = np.zeros((3, 3))
    for m, c, tensor in zip(masses, coms, tensors, strict=True):
        d = c - com
        inertia += tensor + m * (float(d @ d) * np.eye(3) - np.outer(d, d))  # parallel-axis shift to the com
    to_root = quat_to_R(np.asarray(root_quat, dtype=np.float64)).T
    return mass, to_root @ (com - root_pos), to_root @ inertia @ to_root.T


def rig_props(env: EnvBase, homes: dict[str, tuple[np.ndarray, np.ndarray]]) -> list[RigProp]:
    """Each prop named in ``homes`` (where it spawned) as the env built it: shape, freedom, inertia and contact."""
    scene = env.config.scene_options
    props = []
    for name, (pos, quat) in homes.items():
        options = getattr(scene, name)
        entity = env.entities[name]
        # Free or fixed as the env built it: a free joint is what lets it move there.
        free = bool(entity.free_joints)
        # A box is MuJoCo's own primitive; anything else is loaded from its asset.
        is_box = isinstance(options, BoxOptions)
        geoms = entity._entity.geoms
        props.append(
            RigProp(
                name,
                pos,
                quat,
                free,
                half_size=0.5 * np.asarray(options.size) if is_box else None,
                asset_path=None if is_box else options.fetch_asset(),
                friction=getattr(getattr(options, "material", None), "friction", None),
                inertial=prop_inertial(entity, pos, quat) if free else None,
                sol_params=geoms_sol_params(geoms) if geoms else None,
            )
        )
    return props


class EnvPart(NamedTuple):
    """One deployed part of a composite rig, as the env holds it: its sim entity, deployed DOFs and robot options."""

    sim: Any
    dofs_name: list[str]
    options: RobotOptions


def rig_parts(rig: CompositeEntityOptions, parts: dict[str, EnvPart]) -> list[RigPart]:
    """Each jointed part of the composite ``rig`` as the env built it: its pose or weld, joints and contact."""
    from eden.utils.geom import inv_quat, quat_mul, transform_by_quat

    attachments = rig.attachments()
    described = []
    for name, part in parts.items():
        options, sim = part.options, part.sim
        # The asset's own root body, which is what the rig attaches -- not necessarily the env
        # entity's base link (Genesis gives an MJCF a world link of its own at the origin).
        pos, quat = link_pose(sim, root_body_name(options.fetch_asset()))
        parent = parent_body = None
        if name in attachments:
            parent, parent_body = attachments[name][0], attachments[name][1]
            if parent not in parts:
                raise NotImplementedError(
                    f"Part '{name}' is welded to '{parent}', which has no joints and is not deployed; the MuJoCo rig "
                    f"assembles jointed parts only. Parts: {list(parts)}."
                )
            ppos, pquat = link_pose(parts[parent].sim, parent_body)
            inv = inv_quat(pquat[None])
            pos = transform_by_quat((pos - ppos)[None], inv)[0]
            quat = quat_mul(inv, quat[None])[0]
        order = [sim.dofs_name.index(dof) for dof in part.dofs_name]
        material = options.material
        link_friction = {}
        for pattern, value in (options.default_geoms_friction or {}).items():
            for link in sim.find_named_links_idx_local(pattern)[0]:
                link_friction[link] = float(value)
        described.append(
            RigPart(
                name=name,
                asset_path=options.fetch_asset(),
                dofs_name=list(part.dofs_name),
                pos=f64(pos),
                quat=f64(quat),
                parent=parent,
                parent_body=parent_body,
                floating=parent is None and not options.is_fixed_base,
                # As configured, not as built: under the euler integrator the built armature already folds
                # in ``kd * substep_dt``, which the rig adds itself (``MujocoRig.set_implicit_damping``).
                armature=f64(sim.default_dofs_armature)[0][order],
                damping=f64(sim.get_dofs_damping()).reshape(-1)[order],
                gravity_compensation=float(getattr(material, "gravity_compensation", 0.0) or 0.0),
                friction=getattr(material, "friction", None),
                link_friction=link_friction,
                link_sol_params={link.name: geoms_sol_params(link.geoms) for link in sim._entity.links if link.geoms},
            )
        )
    return described
