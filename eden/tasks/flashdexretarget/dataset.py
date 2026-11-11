"""A FlashDexRetarget ``motion.pt`` as Eden motion sequences.

``eden retarget flashdex kinematic`` packs every kept clip of a dataset into one ``motion.pt``: per-frame arrays
concatenated over clips (clip ``i`` covers ``length_starts[i]`` to ``length_starts[i] + motion_num_frames[i]``). This
loader splits it into one trajectory per clip and each trajectory into the task's sequences:

* ``robot`` — the robot hand's joint positions and velocities (``joint_pos`` / ``joint_vel``, the robot MJCF's joint
  order); fixed base, so its base pose is the identity.
* ``mano_<side>`` — the human hand of that side: base = the MANO wrist (position, rotation, velocities), links = the
  20 MANO finger joints and tips in :data:`~eden.extensions.retargeting.flashdexretarget.clips.MANO_JOINT_NAMES`
  order. No scene entity follows it; the rewards and observations read it as a reference.
* ``object_<side>`` — the object that hand holds: its pose and velocities (the frame of its mesh origin).

Each clip's object slot (``"<id>*<scale>"``, both hands' joined by ``|``) is per-clip metadata (``"objects"``), which
``MotionTermOptions.env_motion_groups`` groups the envs by, one object variant each.
"""

from __future__ import annotations
from typing import Any, ClassVar

import os

import torch

import eden as en
from eden.datasets.base import DATASET_REGISTRY, DatasetBase
from eden.extensions.retargeting.flashdexretarget.clips import MANO_JOINT_NAMES
from eden.tasks.flashdexretarget.assets import SIDES, motion_summary, robot_spec
from eden.utils.geom import R_to_quat
from eden.utils.motion_reference.motion_data import BatchMotionSequence, MotionSequence

#: The rate ``eden retarget flashdex import`` resamples to, and the task's control rate.
FDR_FRAMERATE = 60.0


@DATASET_REGISTRY.register()
class FlashDexRetargetDataset(DatasetBase):
    """Clips of a FlashDexRetarget ``motion.pt`` (one hand or both, with their objects).

    Parameters
    ----------
    raw_data_path : str
        The ``motion.pt`` (a file, not a directory).
    robot : str
        The robot the motion was retargeted to (``xhand`` / ``sharpa``): names the ``joint_pos`` columns.
    """

    robot: str = "xhand"
    target_framerate: float = FDR_FRAMERATE
    # Keep the pipeline's velocities (finite differences, Gaussian-smoothed): re-differencing would move the
    # reference off the one the rewards were tuned on.
    recalculate_velocities: bool = False
    _cache_identity_fields: ClassVar[tuple[str, ...]] = ("robot",)

    # Every key this loader can produce; an instance narrows it to the motion file's hands (see __init__).
    motion_sequence_keys: ClassVar[tuple[str, ...]] = (
        "robot",
        *(f"mano_{s}" for s in SIDES),
        *(f"object_{s}" for s in SIDES),
    )
    motion_dofs_name: ClassVar[dict[str, list[str]]] = {}
    motion_links_name: ClassVar[dict[str, list[str]]] = {}
    _PER_CLIP_METADATA_KEYS: ClassVar[tuple[str, ...]] = ("objects", "motion_names")

    def _initialize(self):
        if self.raw_data_path is None:
            raise ValueError(
                "FlashDexRetargetDataset needs raw_data_path: the motion.pt `eden retarget flashdex kinematic` wrote."
            )
        summary = motion_summary(self.raw_data_path)
        joints = list(robot_spec(self.robot, summary.sides).joint_names)
        # The hands and joint names depend on the motion file and the robot, so they are set per instance, here
        # rather than in __init__: the options mixin mirrors the class-level values back onto the instance first.
        self.motion_sequence_keys = (
            "robot",
            *(f"mano_{s}" for s in summary.sides),
            *(f"object_{s}" for s in summary.sides),
        )
        self.motion_dofs_name = {
            "robot": joints,
            **{f"mano_{s}": [] for s in summary.sides},
            **{f"object_{s}": [] for s in summary.sides},
        }
        self.motion_links_name = {
            "robot": [],
            **{f"mano_{s}": list(MANO_JOINT_NAMES) for s in summary.sides},
            **{f"object_{s}": [] for s in summary.sides},
        }
        self._summary = summary
        super()._initialize()

    def _get_all_motion_files(self, data_path: str, extension: str = ".pt") -> list[str]:
        """One pseudo-path ``<motion.pt>/<clip>`` per clip, so include/exclude filters select clips."""
        if os.path.isdir(data_path):
            raise ValueError(
                f"FlashDexRetargetDataset.raw_data_path must be the motion.pt file, got the directory {data_path}; "
                f"pass {os.path.join(data_path, 'motion.pt')}."
            )
        names = torch.load(data_path, map_location="cpu", weights_only=True, mmap=True)["motion_filename"]
        return [f"{data_path}/{name}" for name in names]

    def prepare(self) -> tuple[dict[str, BatchMotionSequence], dict[str, Any]]:
        en.logger.info(f"FlashDexRetarget: reading {self.raw_data_path}")
        packed = torch.load(self.raw_data_path, map_location="cpu", weights_only=True)
        wanted = set(self.all_motion_files)
        sides = self._summary.sides
        sequences: dict[str, list[MotionSequence]] = {key: [] for key in self.motion_sequence_keys}
        objects, names, kept = [], [], []
        starts, frames = packed["length_starts"].tolist(), packed["motion_num_frames"].tolist()
        slot = packed[f"motion_object_slot_{sides[0]}"].tolist()
        for i, name in enumerate(packed["motion_filename"]):
            pseudo = f"{self.raw_data_path}/{name}"
            if pseudo not in wanted:
                continue
            rows = slice(starts[i], starts[i] + frames[i])
            for key, seq in self._split_clip(packed, rows, sides).items():
                sequences[key].append(seq)
            objects.append(self._summary.slot_labels[slot[i]])
            names.append(str(name))
            kept.append(pseudo)
        self.all_motion_files = kept

        motion_data: dict[str, BatchMotionSequence] = {}
        for key in self.motion_sequence_keys:
            batch = BatchMotionSequence.from_motion_sequences(
                sequences[key],
                num_dofs=len(self.motion_dofs_name[key]),
                num_links=len(self.motion_links_name[key]),
            )
            batch.is_free_floating = key != "robot"
            batch.entity_kind = {"robot": "hand"}.get(key, "human" if key.startswith("mano_") else "rigid_object")
            motion_data[key] = batch.to(self.device)
        return motion_data, {"objects": objects, "motion_names": names}

    def _split_clip(self, packed: dict, rows: slice, sides: tuple[str, ...]) -> dict[str, MotionSequence]:
        T = rows.stop - rows.start
        # The motion's own rate (resampled to target_framerate); a file from before motion.pt carried it is at 60 Hz.
        pack = dict(framerate=self._summary.fps or FDR_FRAMERATE, derive_body_frame=False)
        identity = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(T, 4).clone()
        out = {
            "robot": self._pack_motion_sequence(
                base_pos_w=torch.zeros(T, 3),
                base_quat_w=identity,
                base_lin_vel_w=torch.zeros(T, 3),
                base_ang_vel_w=torch.zeros(T, 3),
                dofs_pos=packed["joint_pos"][rows].float(),
                dofs_vel=packed["joint_vel"][rows].float(),
                **pack,
            )
        }
        for s in sides:
            joints = packed[f"mano_{s}_joints"][rows].float()
            names = list(packed.get(f"mano_{s}_joint_names", MANO_JOINT_NAMES))
            order = [names.index(n) for n in MANO_JOINT_NAMES]
            out[f"mano_{s}"] = self._pack_motion_sequence(
                base_pos_w=packed[f"mano_{s}_wrist_pos"][rows].float(),
                base_quat_w=R_to_quat(packed[f"mano_{s}_wrist_rot"][rows].float()),
                base_lin_vel_w=packed[f"mano_{s}_wrist_vel"][rows].float(),
                base_ang_vel_w=packed[f"mano_{s}_wrist_angvel"][rows].float(),
                links_pos_w=joints[:, order],
                links_quat_w=identity[:, None].expand(T, len(order), 4).clone(),
                links_lin_vel_w=packed[f"mano_{s}_joints_vel"][rows].float()[:, order],
                links_ang_vel_w=torch.zeros(T, len(order), 3),
                **pack,
            )
            out[f"object_{s}"] = self._pack_motion_sequence(
                base_pos_w=packed[f"obj_{s}_pos"][rows].float(),
                base_quat_w=R_to_quat(packed[f"obj_{s}_rotmat"][rows].float()),
                base_lin_vel_w=packed[f"obj_{s}_vel"][rows].float(),
                base_ang_vel_w=packed[f"obj_{s}_angvel"][rows].float(),
                **pack,
            )
        return out
