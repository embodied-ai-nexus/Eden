"""ARCTIC bimanual hand-object interaction dataset."""

from __future__ import annotations
from typing import TYPE_CHECKING, ClassVar, Any

import os
import numpy as np
from genesis.utils.geom import xyz_to_quat

from eden.datasets.base import DatasetBase, DATASET_REGISTRY
from eden.datasets.constants import MANO_KEYPOINTS_NAME
from eden.utils.motion_reference.motion_data import BatchMotionSequence
from eden.options.objects.arctic import ARCTIC_OBJECTS_REGISTRY

if TYPE_CHECKING:
    from eden.utils.motion_reference.motion_data import MotionSequence


@DATASET_REGISTRY.register()
class ArcticDataset(DatasetBase):
    """Dataset class for the Arctic dataset.

    Arctic dataset contains sequences with a left hand, a right hand, and an object.
    Each of these entities is represented by a separate MotionSequence.

    References
    ----------
    - https://huggingface.co/datasets/Kashu7100/eden_arctic
    """

    registry: str = "Kashu7100"
    dataset: str = "eden_arctic"

    file_extension: ClassVar[str] = ".npy"
    motion_sequence_keys: ClassVar[tuple[str, ...]] = (
        "object",
        "left_hand",
        "right_hand",
    )
    motion_dofs_name: ClassVar[dict[str, list[str]]] = {
        "object": ["rotation"],
        "left_hand": [],
        "right_hand": [],
    }
    motion_links_name: ClassVar[dict[str, list[str]]] = {
        "object": ["bottom", "top"],
        "left_hand": list(MANO_KEYPOINTS_NAME),
        "right_hand": list(MANO_KEYPOINTS_NAME),
    }

    def prepare(self) -> tuple[dict[str, BatchMotionSequence], dict[str, Any]]:
        """
        Prepare the Arctic dataset by reading every raw ``.npy`` file in ``all_motion_files``.

        Returns
        -------
        tuple[dict[str, BatchMotionSequence], dict[str, Any]]
            The object, left-hand and right-hand sequences by ``motion_sequence_keys``, and the per-clip
            ``objects`` metadata.
        """
        # Parallel CPU read; bulk H2D move once everything is packed.
        all_motion_sequences = self._parallel_read_motion_files(
            self.all_motion_files,
            self._read_single_sequence,
            desc="Arctic: reading raw .npy",
        )

        all_obj_sequences = BatchMotionSequence.from_motion_sequences(
            [obj_seq for obj_seq, _, _, _ in all_motion_sequences],
            num_links=len(self.motion_links_name["object"]),
        ).to(self.device)
        all_left_hand_sequences = BatchMotionSequence.from_motion_sequences(
            [left_hand_seq for _, left_hand_seq, _, _ in all_motion_sequences]
        ).to(self.device)
        all_right_hand_sequences = BatchMotionSequence.from_motion_sequences(
            [right_hand_seq for _, _, right_hand_seq, _ in all_motion_sequences]
        ).to(self.device)

        metadata = {
            "objects": [meta["object"] for _, _, _, meta in all_motion_sequences],
        }

        data = dict(
            zip(
                self.motion_sequence_keys,
                (all_obj_sequences, all_left_hand_sequences, all_right_hand_sequences),
                strict=True,
            )
        )

        return data, metadata

    def _read_single_sequence(
        self, file_path: str
    ) -> tuple[MotionSequence, MotionSequence, MotionSequence, dict[str, Any]]:
        if not file_path.endswith(".npy"):
            raise ValueError(f"Unsupported file type: {file_path}")

        data = np.load(file_path, allow_pickle=True).item()
        fps = data.get("fps", 30.0)

        # 1. Object Motion
        obj_seq = self._read_object_sequence(data, fps)
        # 2. Left Hand Motion
        left_hand_seq = self._read_hand_sequence(data, "left", fps)
        # 3. Right Hand Motion
        right_hand_seq = self._read_hand_sequence(data, "right", fps)

        obj_name = os.path.basename(file_path).split("_use_")[0]
        assert obj_name in ARCTIC_OBJECTS_REGISTRY, f"Object name {obj_name} not found in ARCTIC_OBJECTS_REGISTRY"

        return (
            obj_seq,
            left_hand_seq,
            right_hand_seq,
            {"object": obj_name},
        )

    def _read_object_sequence(self, data: dict, fps: int | float):
        """Pack the object's root pose and articulation angle into a MotionSequence."""
        # CPU-only conversion; prepare() bulk-moves the packed batch.
        params = data.get("params", {})
        obj_pos = params.get("obj_trans")
        obj_quat = params.get("obj_quat")
        obj_dofs_pos = params.get("obj_arti")

        if obj_pos is None or obj_quat is None:
            raise ValueError("Object position or quaternion is missing in the data")

        obj_seq = self._pack_motion_sequence(
            framerate=fps,
            base_pos_w=obj_pos,
            base_quat_w=obj_quat,
            dofs_pos=obj_dofs_pos[:, None] if obj_dofs_pos is not None else None,
        )

        return obj_seq

    def _read_hand_sequence(self, data: dict, side: str, fps: int | float):
        """Pack one hand's MANO keypoints (root = keypoint 0) and palm orientation into a MotionSequence."""
        configs = data.get(f"configs_{side}")
        world_coord = data.get("world_coord", {})

        if configs is None:
            raise ValueError(
                f"Missing hand configuration 'configs_{side}' in ARCTIC sample. Available keys: {list(data.keys())}"
            )

        palm_rpy = np.stack(
            [
                configs["palm_roll_joint"],
                configs["palm_pitch_joint"],
                configs["palm_yaw_joint"],
            ],
            axis=-1,
        )

        # NOTE: The base position is the position of the 0th joint of MANO hand
        joints_pos_w = world_coord.get(f"joints.{side}")  # (num_frames, 21, 3)
        if joints_pos_w is None:
            raise ValueError(
                f"Missing joints data 'joints.{side}' in ARCTIC sample world_coord. "
                f"Available world_coord keys: {list(world_coord.keys())}"
            )

        hand_seq = self._pack_motion_sequence(
            framerate=fps,
            base_pos_w=joints_pos_w[:, 0],  # (num_frames, 3)
            base_quat_w=xyz_to_quat(palm_rpy, rpy=True),  # (num_frames, 4)
            links_pos_w=joints_pos_w,  # (num_frames, 21, 3)
        )

        return hand_seq
