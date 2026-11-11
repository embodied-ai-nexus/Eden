"""Convert an AssistMimic paired ``.pkl`` (SMPL-X, 52 joints) into the SMPL (24-body) reference ``.npz``.

---
tier: project
capabilities: [task.mimic, dataset.inter_x, dataset.hhi_assist, retarget.human]
hardware: cpu
optional_deps: []
run: python examples/projects/mimic/assistmimic/scripts/convert_pair_motion.py --dataset inter-x --pkl <interx.pkl> --smpl-dir <data/smpl> -o data/interx.npz
---

Runs in a PHC / AssistMimic environment (``poselib``, ``smpl_sim`` and the SMPL-X body model), not in
Eden's ``uv`` env, because it reproduces AssistMimic's motion-library preprocessing exactly:

    cd /path/to/AssistMimic
    python /path/to/Eden/examples/projects/mimic/assistmimic/scripts/convert_pair_motion.py \
        --dataset inter-x --pkl sample_data/interx_processed_fixed_v9_cluster_ids_1_n_clusters_10.pkl \
        --smpl-dir data/smpl -o /path/to/Eden/examples/projects/mimic/assistmimic/data/interx_cluster1.npz

What it does, per ``<clip>_caregiver`` / ``<clip>_recipient`` key:

1. ``root_trans_offset`` is height-fixed the way the task that reads this dataset configures
   ``MotionLibSMPL`` — that is what ``--dataset`` selects, and it is **not** cosmetic: Inter-X's
   ``HumanoidImInterxHelpUp`` overrides ``fix_height`` to ``FixHeightMode.full_fix`` (the clip is dropped
   until its lowest vertex touches the floor), while HHI-Assist's ``HumanoidAssistBed`` inherits
   ``HumanoidIm``'s ``FixHeightMode.pene_fix``, which *keeps* the recipient at the height of the bed they
   were recorded on. Converting a bed clip with ``full_fix`` lays the recipient on the floor 30 cm below
   the bed, and the humanoid is then ejected on the first step.
2. The global joint rotations are moved from the raw SMPL frame (AssistMimic's SMPL-X robot is not
   upright-start) to PHC's upright-start frame, and those of the 22 body joints shared by
   SMPL-X and SMPL are kept; SMPL's ``L_Hand`` / ``R_Hand`` (absent in SMPL-X) inherit the wrist rotation,
   i.e. an identity local rotation.
3. Forward kinematics on the SMPL skeleton of ``smpl_humanoid_phc.xml`` (``PhcSmplHumanoid``'s asset) with PHC's
   ``poselib`` gives the body positions, rotations, velocities and angular velocities. (The hinge velocities
   are derived in Eden from the local rotations, ``AssistMimicPairMotion``.)

Quaternions are written **wxyz** (Eden's convention); PHC/poselib is xyzw internally.
"""

import argparse
import os
import sys
import types

import numpy as np

sys.modules.setdefault("cv2", types.ModuleType("cv2"))
for _alias, _builtin in {
    "bool": bool,
    "int": int,
    "float": float,
    "complex": complex,
    "object": object,
    "unicode": str,
    "str": str,
}.items():
    if not hasattr(np, _alias):
        setattr(np, _alias, _builtin)

import joblib  # noqa: E402
import torch  # noqa: E402
from scipy.spatial.transform import Rotation as sRot  # noqa: E402

_UPRIGHT_BASE = sRot.from_quat([0.5, 0.5, 0.5, 0.5])

#: ``MotionLibSMPL``'s ``fix_height`` as each AssistMimic task configures it: ``HumanoidImInterxHelpUp``
#: overrides it to ``full_fix``, ``HumanoidAssistBed`` inherits ``HumanoidIm``'s ``pene_fix``.
FIX_HEIGHT_MODE = {"inter-x": "full", "hhi-assist": "pene"}

SMPLX_MUJOCO_NAMES = [
    "Pelvis", "L_Hip", "L_Knee", "L_Ankle", "L_Toe", "R_Hip", "R_Knee", "R_Ankle", "R_Toe",
    "Torso", "Spine", "Chest", "Neck", "Head",
    "L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist",
    "L_Index1", "L_Index2", "L_Index3", "L_Middle1", "L_Middle2", "L_Middle3",
    "L_Pinky1", "L_Pinky2", "L_Pinky3", "L_Ring1", "L_Ring2", "L_Ring3", "L_Thumb1", "L_Thumb2", "L_Thumb3",
    "R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist",
    "R_Index1", "R_Index2", "R_Index3", "R_Middle1", "R_Middle2", "R_Middle3",
    "R_Pinky1", "R_Pinky2", "R_Pinky3", "R_Ring1", "R_Ring2", "R_Ring3", "R_Thumb1", "R_Thumb2", "R_Thumb3",
]  # fmt: skip
SMPL_MUJOCO_NAMES = [
    "Pelvis", "L_Hip", "L_Knee", "L_Ankle", "L_Toe", "R_Hip", "R_Knee", "R_Ankle", "R_Toe",
    "Torso", "Spine", "Chest", "Neck", "Head",
    "L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist", "L_Hand",
    "R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist", "R_Hand",
]  # fmt: skip
# SMPL's hand joints do not exist in SMPL-X: take the wrist's global rotation (identity local rotation).
_SOURCE_NAME = {"L_Hand": "L_Wrist", "R_Hand": "R_Wrist"}
SMPL_FROM_SMPLX = [SMPLX_MUJOCO_NAMES.index(_SOURCE_NAME.get(n, n)) for n in SMPL_MUJOCO_NAMES]


def xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    return np.concatenate([q[..., 3:4], q[..., :3]], axis=-1)


def fix_trans_height(pose_aa, trans, gender_beta, mesh_parser, mode, frame_check=30):
    """AssistMimic's ``MotionLibSMPL.fix_trans_height``, ``FixHeightMode.full_fix`` or ``pene_fix``.

    Re-implemented rather than imported: ``assistmimic.utils`` pulls in Isaac Gym, which must be imported
    before torch. Two AssistMimic quirks are reproduced rather than corrected, because the reference heights
    the trained policies expect are the ones these lines produce:

    * ``pene_fix`` indexes the **vertex** array with *joint* indices (``vertices[:, ankle_ids]``), so the
      "ankle height" it tests is the height of two arbitrary vertices near the pelvis, and its head height
      likewise reads vertex ``joint_names.index("Head")``.
    * Its ankle test is run on raw vertices while every other quantity has the SMPL root ``offset`` removed.

    Parameters
    ----------
    pose_aa : torch.Tensor
        Axis-angle pose, ``(T, 156)``.
    trans : torch.Tensor
        Root translation to fix in place, ``(T, 3)``.
    gender_beta : torch.Tensor
        ``[gender, *betas]`` as ``MotionLibSMPL`` passes it.
    mesh_parser : smpl_sim.smpllib.smpl_parser.SMPLX_Parser
        Body model the vertices come from.
    mode : str
        ``"full"`` (Inter-X) or ``"pene"`` (HHI-Assist).
    frame_check : int
        Leading frames the fix is measured over (AssistMimic's calibration phase).

    Returns
    -------
    torch.Tensor
        ``trans`` with its z shifted.
    """
    with torch.no_grad():
        betas = gender_beta[1:]
        vertices, joints = mesh_parser.get_joints_verts(pose_aa[:frame_check], betas[None,], trans[:frame_check])
        offset = joints[:, 0] - trans[:frame_check]
        corrected_z = (vertices - offset[:, None])[..., -1]  # (frame_check, n_verts)
        if mode == "full":
            diff_fix = corrected_z.min(dim=-1).values.min()
        elif mode == "pene":
            ankle_ids = [mesh_parser.joint_names.index("L_Ankle"), mesh_parser.joint_names.index("R_Ankle")]
            head_id = mesh_parser.joint_names.index("Head")
            ankle_min_height = vertices[:, ankle_ids][..., -1].min(dim=-1).values.min()
            # AssistMimic subtracts a ``(T, 1, 3)`` offset from a ``(T, 3)`` slice, which broadcasts into a
            # ``(T, T)`` grid; its minimum is this, and the trained references depend on that value.
            head_height = vertices[:, head_id, -1].min() - offset[:, -1].max()
            if ankle_min_height > 0.30 and head_height < 0.80:
                # Lying on a raised surface: put the clip back at the recorded height of the bed.
                diff_fix = (corrected_z[:, ankle_ids].min(dim=-1).values - 0.6).min()
            else:
                min_height = corrected_z.min(dim=-1).values.min()
                diff_fix = min_height if min_height < 0 else torch.zeros_like(min_height)
        else:
            raise ValueError(f"fix_trans_height: mode {mode!r} is not 'full' or 'pene'.")
        trans[..., -1] -= diff_fix
    return trans


def _root_translation(entry: dict):
    """The root translation ``MotionLibSMPL.load_motion_with_skeleton`` picks, quirk included.

    It runs ``entry["trans"].clone()`` under a bare ``except`` that falls back to ``root_trans_offset``, so
    which of the two a clip is played at comes down to how its ``trans`` happens to have been pickled: the
    HHI-Assist pkls store a ``torch.Tensor`` and are played at ``trans``, the Inter-X ones a ``numpy.ndarray``
    (no ``.clone()``) and are played at ``root_trans_offset``. The two differ by ~0.35 m for a recipient lying
    on the bed, which is the whole height of the bed — so this is not a detail to tidy up.
    """
    try:
        return entry["trans"].clone()
    except AttributeError:
        return entry["root_trans_offset"]


#: Where Eden caches ``PhcSmplHumanoid``'s MJCF: its asset directory, ``$EDEN_ASSETS_DIR`` else
#: ``$XDG_CACHE_HOME/eden/assets`` (``~/.cache/eden/assets``). This script runs without Eden importable, so it
#: restates ``eden.utils.misc.get_assets_dir()``'s rule rather than calling it.
_EDEN_ASSETS_DIR = os.environ.get("EDEN_ASSETS_DIR") or os.path.join(
    os.environ.get("XDG_CACHE_HOME") or os.path.join("~", ".cache"), "eden", "assets"
)
DEFAULT_XML = os.path.join(os.path.expanduser(_EDEN_ASSETS_DIR), "humanoids", "smpl_humanoid", "smpl_humanoid_phc.xml")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=sorted(FIX_HEIGHT_MODE),
        help="Which AssistMimic task reads this pkl — it picks the height fix (see the module docstring)",
    )
    parser.add_argument("--pkl", type=str, required=True, help="AssistMimic paired pkl")
    parser.add_argument("--smpl-dir", type=str, required=True, help="Directory with SMPLX_NEUTRAL.* body models")
    parser.add_argument(
        "--xml",
        type=str,
        default=DEFAULT_XML,
        help="SMPL humanoid MJCF whose skeleton the reference is retargeted onto (default: PhcSmplHumanoid's asset "
        "in Eden's asset cache)",
    )
    parser.add_argument("--assistmimic-root", type=str, default=".", help="AssistMimic checkout (for poselib)")
    parser.add_argument("-o", "--output", type=str, required=True, help="Output .npz")
    args = parser.parse_args()

    sys.path.insert(0, os.path.abspath(args.assistmimic_root))
    from poselib.poselib.skeleton.skeleton3d import SkeletonMotion, SkeletonState, SkeletonTree
    from smpl_sim.smpllib.smpl_parser import SMPLX_Parser

    if not os.path.isfile(args.xml):
        raise SystemExit(
            f"--xml {args.xml!r} does not exist. Pass PhcSmplHumanoid's MJCF: `huggingface-cli download --repo-type "
            "dataset Kashu7100/eden_assets humanoids/smpl_humanoid/smpl_humanoid_phc.xml --local-dir <dir>`, then "
            "--xml <dir>/humanoids/smpl_humanoid/smpl_humanoid_phc.xml (any Eden run that builds the task also leaves "
            "it under Eden's asset directory: `python -c 'from eden.utils.misc import get_assets_dir; "
            "print(get_assets_dir())'`)."
        )
    sk_tree = SkeletonTree.from_mjcf(args.xml)
    assert list(sk_tree.node_names) == SMPL_MUJOCO_NAMES, sk_tree.node_names
    parser_n = SMPLX_Parser(
        model_path=args.smpl_dir,
        gender="neutral",
        use_pca=False,
        create_transl=False,
        flat_hand_mean=True,
        num_betas=20,
    )
    gender_beta = torch.zeros(21)  # AssistMimic: no shape variation -> neutral, zero betas

    data = joblib.load(args.pkl)
    clips = sorted({k.rsplit("_", 1)[0] for k in data if k.endswith(("_caregiver", "_recipient"))})
    out = {"clips": np.array(clips), "body_names": np.array(SMPL_MUJOCO_NAMES)}
    per_role = {role: {k: [] for k in ("gts", "grs", "lrs", "gvs", "gavs")} for role in ("caregiver", "recipient")}
    lengths, fps_all = [], []
    for clip in clips:
        n_frames = None
        for role in ("caregiver", "recipient"):
            entry = data[f"{clip}_{role}"]
            trans = torch.as_tensor(np.asarray(_root_translation(entry)), dtype=torch.float32).clone()
            pose_aa = torch.as_tensor(np.asarray(entry["pose_aa"]), dtype=torch.float32)
            trans = fix_trans_height(pose_aa, trans, gender_beta, parser_n, FIX_HEIGHT_MODE[args.dataset])
            quat_global = np.asarray(entry["pose_quat_global"])[:, SMPL_FROM_SMPLX]
            # AssistMimic's SMPL-X robot has ``has_upright_start: False``, so these rotations are in the raw SMPL
            # frame; the SMPL ``phc_3`` robot is upright-start. PHC's converter maps between the two with
            # ``R_upright = R_raw * R([0.5, 0.5, 0.5, 0.5]).inv()`` (xyzw).
            quat_global = (sRot.from_quat(quat_global.reshape(-1, 4)) * _UPRIGHT_BASE.inv()).as_quat()
            quat_global = torch.as_tensor(quat_global.reshape(-1, len(SMPL_MUJOCO_NAMES), 4), dtype=torch.float32)
            sk_state = SkeletonState.from_rotation_and_root_translation(sk_tree, quat_global, trans, is_local=False)
            fps = int(entry.get("fps", 30))
            motion = SkeletonMotion.from_skeleton_state(sk_state, fps)
            arrays = {
                "gts": motion.global_translation.numpy(),
                "grs": xyzw_to_wxyz(motion.global_rotation.numpy()),
                "lrs": xyzw_to_wxyz(motion.local_rotation.numpy()),
                "gvs": motion.global_velocity.numpy(),
                "gavs": motion.global_angular_velocity.numpy(),
            }
            if n_frames is None:
                n_frames = arrays["gts"].shape[0]
            elif n_frames != arrays["gts"].shape[0]:
                raise ValueError(f"{clip}: caregiver has {n_frames} frames, recipient {arrays['gts'].shape[0]}")
            for k, v in arrays.items():
                per_role[role][k].append(v.astype(np.float32))
        lengths.append(n_frames)
        fps_all.append(fps)
    out["num_frames"] = np.array(lengths, dtype=np.int64)
    out["fps"] = np.array(fps_all, dtype=np.float32)
    for role, arrays in per_role.items():
        for k, v in arrays.items():
            out[f"{role}_{k}"] = np.concatenate(v, axis=0)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez_compressed(args.output, **out)
    print(f"wrote {args.output}: {len(clips)} clips, {sum(lengths)} frames/role")


if __name__ == "__main__":
    main()
