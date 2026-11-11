"""Weave G1 + Inspire human-object interaction reference motions (OMOMO retargeted), with their objects.

`Weave <https://github.com/xiaohu-art/Weave>`_ retargets OMOMO human-object clips onto a Unitree G1 with two Inspire
RH56DFQ hands and publishes them on the HuggingFace dataset ``appolyn/Weave``::

    reference/{train,test}/<object>_<N>clip[_var5].npz   # N clips of one object, concatenated along frames
    objects/<object>/{<object>.obj, surface.npy, sdf_128.npz}
    objects/geometry/bps_128.npy

Each npz holds every clip of one object back to back (split by ``motion_lengths``), at 50 Hz, in Isaac Lab's
articulation order of the merged G1 + Inspire URDF: 53 joints, 54 bodies, plus the manipulated object's root pose
and velocity and a tri-state per-body contact label. This loader splits each file into one trajectory per clip and
each trajectory into the four entities of the :class:`~eden.options.robots.G1Inspire` scene:

* ``robot`` — the 29 body DOFs and the 30 G1 links, based at the pelvis;
* ``left_hand`` / ``right_hand`` — each hand's 12 DOFs and 13 links: its ``hand_base_link`` (which Weave's merged
  URDF folds into ``*_wrist_yaw_link``, so its pose is rebuilt from the wrist through the rig's fixed mount) and the
  12 finger links. Named as the hand entity names them, without Weave's ``L_`` / ``R_`` prefix. Based at the hand
  base link, flagged ``is_free_floating=False`` (a welded hand has no base of its own to track);
* ``object`` — the object's root pose and velocity (``entity_kind="rigid_object"``), in the frame of its mesh
  origin. Build the scene object with ``align=False`` so the simulation reports that same frame.

The per-body ``contact_label`` (``+1`` touch the object, ``-1`` don't, ``0`` unconstrained) lands in each entity's
``contact_label`` field, row-aligned with its links; the hand base link takes its wrist's label, since that is the
body the palm belongs to in Weave. Each clip's object name is per-clip metadata (``"objects"``), which is what
``MotionTermOptions.env_motion_groups`` groups envs by.

Recorded velocities are kept (``recalculate_velocities=False``): the source is already at the 50 Hz control rate, and
re-differencing would move the reference off the one Weave trained on. The body-frame ``links_*_b`` fields are not
derived (no tracking term reads them), which keeps the 19.6 h train split around 12 GB on the GPU.
"""

from __future__ import annotations

import glob
import os
from typing import Any, ClassVar, Literal

import numpy as np
import torch

import eden as en
from eden.datasets.base import _READING_MOTION_FILE, DATASET_REGISTRY, DatasetBase, _ShortClipError
from eden.options.robots.composite.g1_inspire import (
    G1_INSPIRE_LEFT_MOUNT_QUAT,
    G1_INSPIRE_MOUNT_OFFSET,
    G1_INSPIRE_RIGHT_MOUNT_QUAT,
)
from eden.utils.geom import transform_by_quat, transform_quat_by_quat
from eden.utils.motion_reference.motion_data import BatchMotionSequence, MotionSequence
from eden.utils.string import columns_by_name
from eden.utils.torch import numpy_to_tensor

#: The nine OMOMO objects Weave publishes clips for, in the order ``G1-Inspire-HOI-v0`` lists them.
WEAVE_OBJECTS: tuple[str, ...] = (
    "clothesstand",
    "floorlamp",
    "largebox",
    "largetable",
    "smallbox",
    "smalltable",
    "tripod",
    "whitechair",
    "woodchair",
)

WEAVE_FRAMERATE = 50.0

_BODY_JOINTS: tuple[str, ...] = (
    "left_hip_pitch",
    "right_hip_pitch",
    "waist_yaw",
    "left_hip_roll",
    "right_hip_roll",
    "waist_roll",
    "left_hip_yaw",
    "right_hip_yaw",
    "waist_pitch",
    "left_knee",
    "right_knee",
    "left_shoulder_pitch",
    "right_shoulder_pitch",
    "left_ankle_pitch",
    "right_ankle_pitch",
    "left_shoulder_roll",
    "right_shoulder_roll",
    "left_ankle_roll",
    "right_ankle_roll",
    "left_shoulder_yaw",
    "right_shoulder_yaw",
    "left_elbow",
    "right_elbow",
    "left_wrist_roll",
    "right_wrist_roll",
    "left_wrist_pitch",
    "right_wrist_pitch",
    "left_wrist_yaw",
    "right_wrist_yaw",
)
_HAND_JOINT_TIERS: tuple[tuple[str, ...], ...] = (
    ("index_proximal", "middle_proximal", "pinky_proximal", "ring_proximal", "thumb_proximal_yaw"),
    ("index_intermediate", "middle_intermediate", "pinky_intermediate", "ring_intermediate", "thumb_proximal_pitch"),
    ("thumb_intermediate",),
    ("thumb_distal",),
)
#: The npz joint axis: Isaac Lab's breadth-first articulation order of the merged G1 + Inspire URDF — the 29 body
#: joints, then the hands tier by tier, left before right within a tier.
WEAVE_JOINT_NAMES: tuple[str, ...] = (
    *(f"{n}_joint" for n in _BODY_JOINTS),
    *(f"{s}_{n}_joint" for tier in _HAND_JOINT_TIERS for s in ("L", "R") for n in tier),
)

#: The 30 G1 links, first on the npz body axis (index 0 is the pelvis).
WEAVE_ROBOT_LINK_NAMES: tuple[str, ...] = (
    "pelvis",
    "left_hip_pitch_link",
    "right_hip_pitch_link",
    "waist_yaw_link",
    "left_hip_roll_link",
    "right_hip_roll_link",
    "waist_roll_link",
    "left_hip_yaw_link",
    "right_hip_yaw_link",
    "torso_link",
    "left_knee_link",
    "right_knee_link",
    "left_shoulder_pitch_link",
    "right_shoulder_pitch_link",
    "left_ankle_pitch_link",
    "right_ankle_pitch_link",
    "left_shoulder_roll_link",
    "right_shoulder_roll_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_shoulder_yaw_link",
    "right_shoulder_yaw_link",
    "left_elbow_link",
    "right_elbow_link",
    "left_wrist_roll_link",
    "right_wrist_roll_link",
    "left_wrist_pitch_link",
    "right_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
)
_HAND_LINK_TIERS: tuple[tuple[str, ...], ...] = (
    ("index_proximal", "middle_proximal", "pinky_proximal", "ring_proximal", "thumb_proximal_base"),
    ("index_intermediate", "middle_intermediate", "pinky_intermediate", "ring_intermediate", "thumb_proximal"),
)
#: The full 54-body npz axis: the G1 links, then the hands tier by tier (left before right within a tier).
WEAVE_BODY_NAMES: tuple[str, ...] = (
    *WEAVE_ROBOT_LINK_NAMES,
    *(f"{s}_{n}" for tier in _HAND_LINK_TIERS for s in ("L", "R") for n in tier),
    "L_thumb_intermediate",
    "R_thumb_intermediate",
    "L_thumb_distal",
    "R_thumb_distal",
)

#: Hand entity -> (Weave's name prefix, wrist side, hand base link orientation in the wrist frame).
_SIDES: dict[str, tuple[str, str, tuple[float, float, float, float]]] = {
    "left_hand": ("L", "left", G1_INSPIRE_LEFT_MOUNT_QUAT),
    "right_hand": ("R", "right", G1_INSPIRE_RIGHT_MOUNT_QUAT),
}


def _unprefixed(names, prefix: str) -> list[str]:
    """The names starting with Weave's ``<prefix>_``, as the hand entity names them (prefix dropped)."""
    return [n.removeprefix(f"{prefix}_") for n in names if n.startswith(f"{prefix}_")]


def _hand_link_names(prefix: str) -> list[str]:
    """A hand entity's motion links: its base link, then its 12 finger links in npz order."""
    return ["hand_base_link", *_unprefixed(WEAVE_BODY_NAMES, prefix)]


def _hand_dofs_name(prefix: str) -> list[str]:
    return _unprefixed(WEAVE_JOINT_NAMES, prefix)


# Per-frame npz keys sliced per clip (the per-clip keys are motion_lengths / motion_names / object_names / fps).
_PER_FRAME_KEYS: tuple[str, ...] = (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
    "object_pos_w",
    "object_quat_w",
    "object_lin_vel_w",
    "object_ang_vel_w",
    "contact_label",
)

_BODY_AXIS = "Weave's npz body axis (WEAVE_BODY_NAMES)"
_JOINT_AXIS = "Weave's npz joint axis (WEAVE_JOINT_NAMES)"
#: The npz body-axis rows of the G1 links, in ``motion_links_name["robot"]`` order.
_ROBOT_ROWS = torch.tensor(columns_by_name(WEAVE_BODY_NAMES, WEAVE_ROBOT_LINK_NAMES, what=_BODY_AXIS))
#: Hand entity -> (its wrist's npz body-axis row, its finger links' rows on that axis in ``motion_links_name`` order
#: after the base link, its DOFs' npz joint-axis columns in ``motion_dofs_name`` order).
_HAND_COLUMNS: dict[str, tuple[int, torch.Tensor, torch.Tensor]] = {
    key: (
        columns_by_name(WEAVE_BODY_NAMES, [f"{side}_wrist_yaw_link"], what=_BODY_AXIS)[0],
        torch.tensor(
            columns_by_name(WEAVE_BODY_NAMES, [f"{prefix}_{n}" for n in _hand_link_names(prefix)[1:]], what=_BODY_AXIS)
        ),
        torch.tensor(
            columns_by_name(WEAVE_JOINT_NAMES, [f"{prefix}_{n}" for n in _hand_dofs_name(prefix)], what=_JOINT_AXIS)
        ),
    )
    for key, (prefix, side, _mount_quat) in _SIDES.items()
}


@DATASET_REGISTRY.register()
class WeaveG1InspireHOIDataset(DatasetBase):
    """Weave G1-Inspire-HOI reference clips: a G1 + two Inspire RH56DFQ hands tracking an OMOMO object.

    Parameters
    ----------
    split : str
        ``"train"`` (7,869 clips, 19.6 h) or ``"test"`` (1,605 clips, 3.7 h; five approach variants per clip).
    objects : list[str] | None
        Load only these objects' files (a subset of :data:`WEAVE_OBJECTS`); ``None`` loads all nine. Also narrows
        the download.
    """

    _applies_frame_window: ClassVar[bool] = True

    registry: str = "appolyn"
    dataset: str = "Weave"
    split: Literal["train", "test"] = "test"
    objects: list[str] | None = None
    recalculate_velocities: bool = False
    _cache_identity_fields: ClassVar[tuple[str, ...]] = ("split", "objects")

    motion_sequence_keys: ClassVar[tuple[str, ...]] = ("robot", "left_hand", "right_hand", "object")
    motion_dofs_name: ClassVar[dict[str, list[str]]] = {
        "robot": list(WEAVE_JOINT_NAMES[:29]),
        "left_hand": _hand_dofs_name("L"),
        "right_hand": _hand_dofs_name("R"),
        "object": [],
    }
    motion_links_name: ClassVar[dict[str, list[str]]] = {
        "robot": list(WEAVE_ROBOT_LINK_NAMES),
        "left_hand": _hand_link_names("L"),
        "right_hand": _hand_link_names("R"),
        "object": [],
    }
    _PER_CLIP_METADATA_KEYS: ClassVar[tuple[str, ...]] = ("objects", "motion_names")

    def __init__(self, options, device: torch.device):
        split = getattr(options, "split", None) or type(self).split
        if split not in ("train", "test"):
            raise ValueError(f"WeaveG1InspireHOIDataset.split must be 'train' or 'test', got {split!r}.")
        objects = getattr(options, "objects", None)
        unknown = sorted(set(objects or ()) - set(WEAVE_OBJECTS))
        if unknown:
            raise ValueError(
                f"WeaveG1InspireHOIDataset.objects has unknown object(s) {unknown}; Weave publishes {list(WEAVE_OBJECTS)}."
            )
        if objects is not None and not objects:
            raise ValueError("WeaveG1InspireHOIDataset.objects is empty; pass None to load all nine objects.")
        # The download narrows to this split's reference files of the selected objects, plus their geometry. Those
        # are options, so the patterns are set per instance — before the base __init__ downloads — rather than as
        # the class-level layout most loaders declare; the base download (and its per-file lock) does the fetching.
        names = tuple(objects) if objects is not None else WEAVE_OBJECTS
        self._hf_allow_patterns = (
            *(f"reference/{split}/{obj}_*.npz" for obj in names),
            *(f"objects/{obj}/*" for obj in names),
            "objects/geometry/*",
        )
        super().__init__(options, device)

    # ------------------------------------------------------------------ files

    def _object_names(self) -> tuple[str, ...]:
        return tuple(self.objects) if self.objects is not None else WEAVE_OBJECTS

    def _get_all_motion_files(self, data_path: str, extension: str = ".npz") -> list[str]:
        """One pseudo-path ``<npz>/<motion_name>`` per clip, so include/exclude filters select clips, not files."""
        self._npz_by_object: dict[str, str] = {}
        for obj in self._object_names():
            matches = sorted(glob.glob(os.path.join(data_path, "reference", self.split, f"{obj}_*clip*.npz")))
            if len(matches) != 1:
                raise FileNotFoundError(
                    f"Expected one reference file for object {obj!r} under "
                    f"{os.path.join(data_path, 'reference', self.split)!r}, found {matches}. raw_data_path must be "
                    f"the root of an appolyn/Weave snapshot (holding reference/ and objects/), or None to download it."
                )
            self._npz_by_object[obj] = matches[0]
        files = []
        for obj, npz in self._npz_by_object.items():
            with np.load(npz, allow_pickle=False) as data:
                files += [f"{npz}/{name}" for name in data["motion_names"].tolist()]
        return files

    # ------------------------------------------------------------------ prepare

    def prepare(self) -> tuple[dict[str, BatchMotionSequence], dict[str, Any]]:
        wanted = set(self.all_motion_files)  # after include/exclude filters
        sequences: dict[str, list[MotionSequence]] = {key: [] for key in self.motion_sequence_keys}
        objects: list[str] = []
        motion_names: list[str] = []
        kept_files: list[str] = []
        short: list[_ShortClipError] = []
        window = self._frame_window()
        for obj, npz in self._npz_by_object.items():
            en.logger.info(f"Weave: reading {npz}")
            with np.load(npz, allow_pickle=False) as raw:
                data = {k: raw[k] for k in raw.files}
            fps = float(np.asarray(data["fps"]).reshape(-1)[0])
            starts = np.concatenate([[0], np.cumsum(data["motion_lengths"])])
            if starts[-1] != data["joint_pos"].shape[0]:
                raise ValueError(
                    f"{npz}: motion_lengths sum to {starts[-1]} frames but the arrays hold {data['joint_pos'].shape[0]}."
                )
            for i, name in enumerate(data["motion_names"].tolist()):
                pseudo = f"{npz}/{name}"
                if pseudo not in wanted:
                    continue
                clip = {k: data[k][starts[i] : starts[i + 1]][window] for k in _PER_FRAME_KEYS}
                token = _READING_MOTION_FILE.set(pseudo)  # names the clip in a short-clip message
                try:
                    split = self._split_clip(clip, fps)
                except _ShortClipError as e:
                    short.append(e)
                    continue
                finally:
                    _READING_MOTION_FILE.reset(token)
                for key, seq in split.items():
                    sequences[key].append(seq)
                objects.append(str(data["object_names"][i]))
                motion_names.append(str(name))
                kept_files.append(pseudo)
        self._skip_short_clips(short, len(kept_files) + len(short))
        # prepare() owns the clip order; _initialize stores all_motion_files as the per-trajectory metadata.
        self.all_motion_files = kept_files

        motion_data: dict[str, BatchMotionSequence] = {}
        for key in self.motion_sequence_keys:
            batch = BatchMotionSequence.from_motion_sequences(
                sequences[key],
                num_dofs=len(self.motion_dofs_name[key]),
                num_links=len(self.motion_links_name[key]),
            )
            batch.is_free_floating = key in ("robot", "object")
            batch.entity_kind = {"robot": "humanoid", "object": "rigid_object"}.get(key, "hand")
            motion_data[key] = batch.to(self.device)
        return motion_data, {"objects": objects, "motion_names": motion_names}

    def _split_clip(self, clip: dict[str, np.ndarray], fps: float) -> dict[str, MotionSequence]:
        """One clip's arrays -> a ``MotionSequence`` per entity (CPU tensors)."""
        t = {k: numpy_to_tensor(v, dtype=torch.float32) for k, v in clip.items()}
        label = t["contact_label"].round().to(torch.int8)
        pack = dict(framerate=fps, derive_body_frame=False)
        out = {
            "robot": self._pack_motion_sequence(
                base_pos_w=t["body_pos_w"][:, 0].clone(),
                base_quat_w=t["body_quat_w"][:, 0].clone(),
                base_lin_vel_w=t["body_lin_vel_w"][:, 0].clone(),
                base_ang_vel_w=t["body_ang_vel_w"][:, 0].clone(),
                dofs_pos=t["joint_pos"][:, :29].clone(),
                dofs_vel=t["joint_vel"][:, :29].clone(),
                links_pos_w=t["body_pos_w"][:, _ROBOT_ROWS],
                links_quat_w=t["body_quat_w"][:, _ROBOT_ROWS],
                links_lin_vel_w=t["body_lin_vel_w"][:, _ROBOT_ROWS],
                links_ang_vel_w=t["body_ang_vel_w"][:, _ROBOT_ROWS],
                contact_label=label[:, _ROBOT_ROWS],
                **pack,
            ),
            "object": self._pack_motion_sequence(
                base_pos_w=t["object_pos_w"],
                base_quat_w=t["object_quat_w"],
                base_lin_vel_w=t["object_lin_vel_w"],
                base_ang_vel_w=t["object_ang_vel_w"],
                **pack,
            ),
        }
        for key, (*_, mount_quat) in _SIDES.items():
            wrist, finger_rows, dof_cols = _HAND_COLUMNS[key]
            base_pos, base_quat, base_lin, base_ang = _hand_base_state(
                t["body_pos_w"][:, wrist],
                t["body_quat_w"][:, wrist],
                t["body_lin_vel_w"][:, wrist],
                t["body_ang_vel_w"][:, wrist],
                mount_quat,
            )
            out[key] = self._pack_motion_sequence(
                base_pos_w=base_pos,
                base_quat_w=base_quat,
                base_lin_vel_w=base_lin,
                base_ang_vel_w=base_ang,
                dofs_pos=t["joint_pos"][:, dof_cols],
                dofs_vel=t["joint_vel"][:, dof_cols],
                links_pos_w=torch.cat([base_pos[:, None], t["body_pos_w"][:, finger_rows]], dim=1),
                links_quat_w=torch.cat([base_quat[:, None], t["body_quat_w"][:, finger_rows]], dim=1),
                links_lin_vel_w=torch.cat([base_lin[:, None], t["body_lin_vel_w"][:, finger_rows]], dim=1),
                links_ang_vel_w=torch.cat([base_ang[:, None], t["body_ang_vel_w"][:, finger_rows]], dim=1),
                contact_label=torch.cat([label[:, [wrist]], label[:, finger_rows]], dim=1),
                **pack,
            )
        return out


def weave_object_dir(obj: str, root: str | None = None) -> str:
    """The directory holding Weave object ``obj``'s ``<obj>.obj`` mesh, ``surface.npy`` and ``sdf_128.npz``.

    Parameters
    ----------
    obj : str
        One of :data:`WEAVE_OBJECTS`.
    root : str | None
        The root of an ``appolyn/Weave`` snapshot (the dataset's ``raw_data_path``). ``None`` fetches just this
        object's geometry into the HuggingFace cache (a no-op once present).
    """
    if obj not in WEAVE_OBJECTS:
        raise ValueError(f"Unknown Weave object {obj!r}; Weave publishes {list(WEAVE_OBJECTS)}.")
    if root is None:
        from eden.utils.assets import _snapshot_download_locked

        root = _snapshot_download_locked(
            repo_type="dataset",
            repo_id=f"{WeaveG1InspireHOIDataset.registry}/{WeaveG1InspireHOIDataset.dataset}",
            allow_patterns=[f"objects/{obj}/*", "objects/geometry/*"],
        )
    path = os.path.join(root, "objects", obj)
    if not os.path.isfile(os.path.join(path, f"{obj}.obj")):
        raise FileNotFoundError(f"No mesh for Weave object {obj!r} at {os.path.join(path, obj + '.obj')!r}.")
    return path


def _hand_base_state(
    wrist_pos: torch.Tensor,
    wrist_quat: torch.Tensor,
    wrist_lin: torch.Tensor,
    wrist_ang: torch.Tensor,
    mount_quat: tuple[float, float, float, float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """World pose and velocity of a hand base link, rigidly mounted on its ``*_wrist_yaw_link``."""
    n = wrist_pos.shape[0]
    offset = torch.tensor(G1_INSPIRE_MOUNT_OFFSET, dtype=wrist_pos.dtype).expand(n, 3)
    mount_quat = torch.tensor(mount_quat, dtype=wrist_pos.dtype).expand(n, 4)
    lever = transform_by_quat(offset, wrist_quat)
    base_pos = wrist_pos + lever
    base_quat = transform_quat_by_quat(mount_quat, wrist_quat)
    base_lin = wrist_lin + torch.cross(wrist_ang, lever, dim=-1)
    return base_pos, base_quat, base_lin, wrist_ang.clone()
