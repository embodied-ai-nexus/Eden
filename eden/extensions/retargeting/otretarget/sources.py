"""Demonstration readers: dataset files to a :class:`HumanSurfaceMotion` plus :class:`InteractionObject` tracks.

* :func:`load_grab_sequence` — a raw GRAB capture (SMPL-X with a personal template, PCA hands, the object and
  the table), z-up, 120 fps.
* :func:`mirror_demonstration` — a left-handed SMPL-X demonstration made right-handed (body and objects).
* :func:`load_omomo_sequence` — OMOMO in its canonical SMPL-X form (the ``sequences_canonical/`` layout
  of the ``eden_omomo`` release): ``human.npz`` (156-d SMPL-X poses with finger articulation, betas,
  trans, gender) and ``object.npz`` (axis-angle + translation of ``objects/<name>/<name>.obj``), y-up.
* :func:`load_soma_bvh` — a SOMA-skeleton BVH (BONES-SEED) with the SOMA neutral body skinned onto it.
"""

from __future__ import annotations

import os

import numpy as np
import trimesh

from eden.datasets.utils.body_model import body_model_arrays
from eden.extensions.retargeting.otretarget.human import (
    SMPLX_JOINT_NAMES,
    Y_UP_TO_Z_UP,
    HumanSurfaceMotion,
    smplx_motion,
)
from eden.extensions.retargeting.otretarget.retargeter import InteractionObject
from eden.utils.geom import R_to_quat, quat_from_axis_angle, quat_to_R

#: SMPL-X joints whose skinned vertices make up the feet.
SMPLX_FOOT_JOINTS = ("left_ankle", "left_foot", "right_ankle", "right_foot")

#: SMPL-X joints whose skinned vertices make up the hands (wrists and fingers).
SMPLX_HAND_JOINTS = tuple(
    n for n in SMPLX_JOINT_NAMES if any(part in n for part in ("wrist", "index", "middle", "ring", "pinky", "thumb"))
)


def load_omomo_sequence(
    sequence_dir: str,
    objects_dir: str,
    smplx_dir: str,
    fps: float = 30.0,
    flat_hand_mean: bool = True,
    ground_human: bool = True,
) -> tuple[HumanSurfaceMotion, InteractionObject]:
    """Read one canonical OMOMO sequence.

    Parameters
    ----------
    sequence_dir : str
        ``.../sequences_canonical/<sub>_<object>_<clip>``.
    objects_dir : str
        ``.../objects`` holding ``<name>/<name>.obj`` (metres).
    smplx_dir : str
        Directory with ``SMPLX_{NEUTRAL,MALE,FEMALE}.npz``.
    fps : float
        The release's frame rate (30).
    flat_hand_mean : bool
        Hand pose convention. The release's hand poses are offsets from a flat hand: read with the mean
        hand added, the fingers of a box-carrying subject curl ~2 cm into the box.
    ground_human : bool
        Move the human vertically so its feet stand on z = 0 (:meth:`HumanSurfaceMotion.floor_offset`).
        Some subjects' body fits float above the floor the objects rest on (sub3: ~4.8 cm median over the
        clip, with the box at z = 0). OTRetarget works at native scale and treats the floor as an
        interaction channel, so it would faithfully copy that gap onto the robot's feet. The objects are
        not moved: they are already consistent with the floor.

    Returns
    -------
    human : HumanSurfaceMotion
    obj : InteractionObject
        The object with its demonstrated trajectory, z-up.
    """
    human_npz = np.load(os.path.join(sequence_dir, "human.npz"), allow_pickle=True)
    object_npz = np.load(os.path.join(sequence_dir, "object.npz"), allow_pickle=True)
    gender = str(human_npz["gender"]).lower()
    model = os.path.join(smplx_dir, f"SMPLX_{gender.upper()}.npz")
    if not os.path.isfile(model):
        raise FileNotFoundError(
            f"load_omomo_sequence: {sequence_dir} is a {gender!r} subject but {model} does not exist; "
            f"point smplx_dir at the folder holding SMPLX_{{NEUTRAL,MALE,FEMALE}}.npz."
        )
    human = smplx_motion(
        model,
        human_npz["poses"],
        human_npz["trans"],
        human_npz["betas"],
        fps,
        flat_hand_mean=flat_hand_mean,
        up_axis="y",
    )
    if ground_human:
        human = human.translated(np.array([0.0, 0.0, -human.floor_offset(SMPLX_FOOT_JOINTS)]))
    name = str(object_npz["name"])
    mesh = trimesh.load(os.path.join(objects_dir, name, f"{name}.obj"), force="mesh", process=False)
    rot = Y_UP_TO_Z_UP[None] @ quat_to_R(quat_from_axis_angle(np.asarray(object_npz["angles"], dtype=np.float64)))
    pos = object_npz["trans"] @ Y_UP_TO_Z_UP.T
    poses = np.concatenate([pos, R_to_quat(rot)], axis=1)
    if len(poses) != human.num_frames:
        raise ValueError(
            f"load_omomo_sequence: {sequence_dir} has {human.num_frames} human frames but {len(poses)} object frames."
        )
    return human, InteractionObject(name=name, mesh=mesh, poses=poses)


def load_grab_sequence(
    sequence_path: str,
    grab_root: str,
    smplx_dir: str,
    start: int = 0,
    stop: int | None = None,
    step: int = 4,
    ground: bool = True,
    table_thickness: float = 0.03,
) -> tuple[HumanSurfaceMotion, list[InteractionObject]]:
    """Read one raw GRAB capture (``grab/<subject>/<object>_<intent>.npz``) with its object and table.

    The raw release is used, not a re-processed copy: its object poses fit the ``contact_meshes/*.ply`` scans
    as shipped (InterAct's re-centred ``objects/*.obj`` are off by up to 1.4 cm against the same poses), and it
    carries the table. Conventions, as in GRAB's own body and object models:

    * z-up world, metres, 120 fps; ``step`` frames are skipped (4 -> 30 fps).
    * Body: SMPL-X with the subject's personal template (``vtemp``) and ``betas`` = 0. ``fullpose`` already
      holds the 45-d hand poses expanded from the 24 PCA components *with* the mean hand added, so it is read
      as ``flat_hand_mean=True``.
    * Object and table: ``v_world = v_local @ R(global_orient) + transl``, i.e. the column-vector rotation is
      ``R(global_orient).T``.

    The table mesh is a 5.5 mm slab. It is replaced by a ``table_thickness`` plate with the same footprint and
    the same top face (tilt included), held fixed at its first-frame pose: GRAB's table does not move.

    Parameters
    ----------
    sequence_path : str
        ``.../grab/<subject>/<name>.npz``.
    grab_root : str
        The folder the npz's mesh paths are relative to (holding ``tools/``).
    smplx_dir : str
        Directory with ``SMPLX_{MALE,FEMALE}.npz``.
    start, stop, step : int
        Raw (120 fps) frame range and stride.
    ground : bool
        Move the whole scene (body, object, table) vertically so the feet stand on z = 0. Body, object and
        table come from one marker capture, so they move together; moving only the body (as for OMOMO) would
        open a gap between the hand and the object.
    table_thickness : float
        Thickness of the table plate (m).

    Returns
    -------
    human : HumanSurfaceMotion
    objects : list[InteractionObject]
        The object, then the fixed ``table`` plate.
    """
    r = np.load(sequence_path, allow_pickle=True)
    n_raw = int(r["n_frames"])
    idx = np.arange(start, n_raw if stop is None else min(stop, n_raw), step)
    if len(idx) < 2:
        raise ValueError(
            f"load_grab_sequence: frames [{start}, {stop}) step {step} of {sequence_path} leave {len(idx)} frame(s); "
            f"the capture has {n_raw} raw frames."
        )
    gender = str(r["gender"]).lower()
    model = os.path.join(smplx_dir, f"SMPLX_{gender.upper()}.npz")
    if not os.path.isfile(model):
        raise FileNotFoundError(
            f"load_grab_sequence: {sequence_path} is a {gender!r} subject but {model} does not exist; "
            f"point smplx_dir at the folder holding SMPLX_{{MALE,FEMALE}}.npz."
        )
    body = r["body"].item()
    vtemp = np.asarray(trimesh.load(os.path.join(grab_root, body["vtemp"]), process=False).vertices)
    params = body["params"]
    human = smplx_motion(
        model,
        params["fullpose"][idx],
        params["transl"][idx],
        np.zeros(10),
        float(r["framerate"]) / step,
        flat_hand_mean=True,
        up_axis="z",
        v_template=vtemp,
    )

    def track(entry: dict) -> np.ndarray:
        aa = np.asarray(entry["params"]["global_orient"][idx], dtype=np.float64)
        rot = np.transpose(quat_to_R(quat_from_axis_angle(aa)), (0, 2, 1))
        return np.concatenate([entry["params"]["transl"][idx], R_to_quat(rot)], axis=1)

    obj_entry, table_entry = r["object"].item(), r["table"].item()
    mesh = trimesh.load(os.path.join(grab_root, obj_entry["object_mesh"]), force="mesh", process=False)
    obj = InteractionObject(name=str(r["obj_name"]), mesh=mesh, poses=track(obj_entry))

    table_pose = track(table_entry)
    drift = np.abs(table_pose[:, :3] - table_pose[0, :3]).max()
    if drift > 0.005:
        raise ValueError(
            f"load_grab_sequence: the table of {sequence_path} moves {drift * 100:.1f} cm over frames "
            f"[{start}, {idx[-1] + 1}); it is modelled as a fixed plate, so this capture is not supported."
        )
    slab = trimesh.load(os.path.join(grab_root, table_entry["table_mesh"]), force="mesh", process=False)
    R0 = quat_to_R(quat_from_axis_angle(np.asarray(table_entry["params"]["global_orient"][idx[0]], dtype=np.float64))).T
    thin = int(np.argmin(slab.extents))  # the slab's thickness axis, vertical in the world
    up = np.sign(R0[2, thin])
    lo, hi = slab.bounds
    top = hi[thin] if up > 0 else lo[thin]
    extents = slab.extents.copy()
    extents[thin] = table_thickness
    plate = trimesh.creation.box(extents=extents)
    centre = 0.5 * (lo + hi)
    centre[thin] = top - up * 0.5 * table_thickness
    plate.apply_translation(centre)
    table = InteractionObject(name="table", mesh=plate, poses=np.tile(table_pose[0], (len(idx), 1)), fixed=True)

    if ground:
        lift = np.array([0.0, 0.0, -human.floor_offset(SMPLX_FOOT_JOINTS)])
        human = human.translated(lift)
        for o in (obj, table):
            o.poses = o.poses.copy()
            o.poses[:, :3] += lift
    return human, [obj, table]


def smplx_mirror_maps(model_path: str, tolerance: float = 1e-4) -> tuple[np.ndarray, np.ndarray]:
    """SMPL-X's left/right correspondences: the mirror partner of every vertex and every joint.

    The model's mean template is symmetric about its sagittal plane (x = 0): each vertex reflected through it
    lands on another vertex (within 15 um), or on itself on the midline. The partners are found that way and
    checked to form an involution.

    Parameters
    ----------
    model_path : str
        An ``SMPLX_<GENDER>.npz``.
    tolerance : float
        Largest allowed gap (m) between a reflected vertex and its partner.

    Returns
    -------
    vertex_mirror : np.ndarray
        ``(10475,)`` partner of each vertex.
    joint_mirror : np.ndarray
        ``(55,)`` partner of each joint (``left_*`` <-> ``right_*``).
    """
    from scipy.spatial import cKDTree

    template = body_model_arrays(str(model_path))["v_template"].astype(np.float64)
    gap, vertex_mirror = cKDTree(template).query(template * np.array([-1.0, 1.0, 1.0]))
    if gap.max() > tolerance or not np.array_equal(vertex_mirror[vertex_mirror], np.arange(len(template))):
        raise ValueError(
            f"smplx_mirror_maps: {model_path}'s template is not mirror-symmetric about x = 0 (largest gap "
            f"{gap.max() * 1e3:.3f} mm, tolerance {tolerance * 1e3:.3f} mm); it is not an SMPL-X model."
        )
    names = list(SMPLX_JOINT_NAMES)
    swap = {"left_": "right_", "right_": "left_"}
    joint_mirror = np.array(
        [names.index(next((swap[p] + n[len(p) :] for p in swap if n.startswith(p)), n)) for n in names]
    )
    return vertex_mirror.astype(np.int64), joint_mirror


#: SMPL-X's sagittal reflection in a joint frame: its joints' bind frames are the model frame, x lateral.
SMPLX_LOCAL_MIRROR = np.diag([-1.0, 1.0, 1.0])


def mirror_object(
    obj: InteractionObject, normal: np.ndarray, point: np.ndarray, local_mirror: np.ndarray = SMPLX_LOCAL_MIRROR
) -> InteractionObject:
    """The mirror image of an object track through a world plane (the object of a mirrored demonstration).

    ``v_world = R v + p`` reflected is ``(M R L)(L v) + M p + s``: the mesh is reflected in its own frame by
    ``L`` (face winding reversed, so it stays outward-facing) and the pose becomes ``M R L``, a proper rotation.
    A chiral object becomes its mirror twin; a symmetric one comes out the same shape.

    Parameters
    ----------
    obj : InteractionObject
        The object track to mirror.
    normal, point : np.ndarray
        ``(3,)`` the world mirror plane.
    local_mirror : np.ndarray
        ``(3, 3)`` any reflection in the object frame.
    """
    n = np.asarray(normal, dtype=np.float64)
    n = n / np.linalg.norm(n)
    M = np.eye(3) - 2.0 * np.outer(n, n)
    shift = 2.0 * float(np.dot(np.asarray(point, dtype=np.float64), n)) * n
    L = np.asarray(local_mirror, dtype=np.float64)

    def reflect_mesh(mesh: trimesh.Trimesh | None) -> trimesh.Trimesh | None:
        if mesh is None:
            return None
        return trimesh.Trimesh(np.asarray(mesh.vertices) @ L.T, np.asarray(mesh.faces)[:, ::-1], process=False)

    rot = M[None] @ quat_to_R(obj.poses[:, 3:]) @ L
    pos = obj.poses[:, :3] @ M + shift
    return InteractionObject(
        name=obj.name,
        mesh=reflect_mesh(obj.mesh),
        poses=np.concatenate([pos, R_to_quat(rot)], axis=1),
        demo_mesh=reflect_mesh(obj.demo_mesh),
        fixed=obj.fixed,
        weight=obj.weight,
    )


def mirror_demonstration(
    human: HumanSurfaceMotion, objects: list[InteractionObject], smplx_model_path: str
) -> tuple[HumanSurfaceMotion, list[InteractionObject]]:
    """Mirror an SMPL-X demonstration and its objects through the demonstrator's first-frame sagittal plane.

    The plane holds the pelvis and the pelvis' forward and up directions at frame 0, so the mirrored person
    starts where the original did, facing the same way, with every left/right role swapped.
    """
    vertex_mirror, joint_mirror = smplx_mirror_maps(smplx_model_path)
    pelvis = human.joint_index("pelvis")
    lateral = human.joint_rot[0, pelvis][:, 0].copy()  # SMPL-X model x: the body's left
    lateral[2] = 0.0
    if np.linalg.norm(lateral) < 1e-6:
        raise ValueError("mirror_demonstration: the pelvis' lateral axis is vertical at frame 0; no sagittal plane.")
    point = human.joint_pos[0, pelvis]
    mirrored = human.mirrored(vertex_mirror, joint_mirror, SMPLX_LOCAL_MIRROR, lateral, point)
    return mirrored, [mirror_object(o, lateral, point) for o in objects]


#: SOMA-rig joints whose skinned vertices make up the feet.
SOMA_FOOT_JOINTS = (
    "LeftFoot",
    "LeftToeBase",
    "LeftToeEnd",
    "RightFoot",
    "RightToeBase",
    "RightToeEnd",
)


def load_soma_bvh(
    bvh_path: str,
    soma_model_path: str,
    fps: float = 30.0,
    ground_human: bool = True,
) -> HumanSurfaceMotion:
    """Read a SOMA-skeleton BVH (BONES-SEED) and skin the SOMA neutral body onto it.

    The BVH drives the SOMA rig's 78 joints directly (same joint names and joint frames); the neutral
    mesh of ``SOMA_neutral.npz`` is scaled to the actor's overall bone length and skinned to the BVH's
    world joint transforms, so every limb, foot included, follows the captured skeleton.

    Parameters
    ----------
    bvh_path : str
        A SOMA-rig ``.bvh`` (y-up, centimetres).
    soma_model_path : str
        ``SOMA_neutral.npz`` (from the ``nvidia/soma-x`` release).
    fps : float
        Output frame rate; the capture rate must be an integer multiple of it.
    ground_human : bool
        Move the motion vertically so the feet stand on z = 0 (:meth:`HumanSurfaceMotion.floor_offset`).

    Returns
    -------
    HumanSurfaceMotion
    """
    from scipy.sparse import csc_matrix

    from eden.datasets.utils.body_model import SkinnedBodyModel
    from eden.datasets.utils.bvh import forward_kinematics_matrices, parse_bvh
    from eden.extensions.retargeting.otretarget.human import CANONICAL_TO_ROBOT

    model = np.load(soma_model_path, allow_pickle=True)
    names = [str(n) for n in model["joint_names"]]
    parents = model["joint_parent_ids"].astype(np.int64).copy()
    parents[0] = -1
    bvh = parse_bvh(bvh_path)
    missing = [n for n in names if n not in bvh.joint_names]
    if missing:
        raise ValueError(
            f"load_soma_bvh: {bvh_path} lacks SOMA joints {missing}; it is not a SOMA-rig BVH "
            f"(its joints: {bvh.joint_names[:10]}...)."
        )
    step = bvh.framerate / fps
    if abs(step - round(step)) > 1e-3 or round(step) < 1:
        raise ValueError(
            f"load_soma_bvh: the capture runs at {bvh.framerate:.3f} fps, not an integer multiple of fps={fps}; "
            f"pick an fps that divides it (e.g. {bvh.framerate / 4:g})."
        )
    order = [bvh.joint_names.index(n) for n in names]
    pos, rot = forward_kinematics_matrices(bvh)
    pos, rot = pos[:, order], rot[:, order]
    rot, pos = rot[:: int(round(step))], pos[:: int(round(step))]

    bind = model["bind_pose_world"].astype(np.float64)
    bind_rot, bind_pos = bind[:, :3, :3], bind[:, :3, 3]
    # Scale the neutral body to the actor: ratio of total bone length, BVH offsets vs. SOMA bind.
    bvh_len = sum(np.linalg.norm(bvh.joints[bvh.joint_names.index(n)].offset) for n in names[2:])
    soma_len = sum(np.linalg.norm(bind_pos[j] - bind_pos[parents[j]]) for j in range(2, len(names)))
    scale = bvh_len / soma_len
    weights = csc_matrix(
        (model["skinning_weights_data"], model["skinning_weights_indices"], model["skinning_weights_indptr"]),
        shape=tuple(model["skinning_weights_shape"]),
    ).toarray()
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-12)
    cm = 0.01
    body = SkinnedBodyModel(
        rest_vertices=model["mean"].astype(np.float64) * scale * cm,
        faces=model["triangles"].astype(np.int64),
        skin_weights=weights,
        parents=parents,
        bind_rot=bind_rot,
        bind_pos=bind_pos * scale * cm,
        joint_names=names,
    )
    vertices = body.skin(rot, pos * cm)
    joint_pos = pos * cm @ Y_UP_TO_Z_UP.T
    joint_rot = Y_UP_TO_Z_UP[None, None] @ rot
    vertices = (vertices.astype(np.float64) @ Y_UP_TO_Z_UP.T).astype(np.float32)

    # Rest pose for the transport plan: the rig's own T-pose (its bind pose is an A-pose).
    tpose = model["t_pose_world"].astype(np.float64)
    t_rot, t_pos = tpose[:, :3, :3], tpose[:, :3, 3] * scale * cm
    rest_v = body.skin(t_rot[None], t_pos[None])[0].astype(np.float64) @ CANONICAL_TO_ROBOT.T
    rest_p = t_pos @ CANONICAL_TO_ROBOT.T
    lift = -rest_v[:, 2].min()
    human = HumanSurfaceMotion(
        vertices=vertices,
        faces=body.faces,
        skin_weights=weights,
        joint_names=names,
        joint_pos=joint_pos,
        joint_rot=joint_rot,
        rest_vertices=rest_v + np.array([0.0, 0.0, lift]),
        rest_joint_pos=rest_p + np.array([0.0, 0.0, lift]),
        rest_joint_rot=CANONICAL_TO_ROBOT[None] @ t_rot,
        fps=float(fps),
    )
    if ground_human:
        human = human.translated(np.array([0.0, 0.0, -human.floor_offset(SOMA_FOOT_JOINTS)]))
    return human
