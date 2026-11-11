"""OMOMO object-manipulation dataset (Holosoma variant)."""

from __future__ import annotations

import os
from typing import ClassVar, Any

import numpy as np
import torch

import eden as en
from eden.datasets.base import DatasetBase, DATASET_REGISTRY
from eden.utils.geom import xyzw_to_wxyz
from eden.datasets.constants import SMPLH_KEYPOINTS_NAME
from eden.options.objects.omomo import OMOMO_OBJECTS_REGISTRY
from eden.utils.motion_reference.motion_data import BatchMotionSequence, MotionSequence
from eden.utils.torch import to_numpy

OMOMO_FRAMERATE = 30.0

# ---------------------------------------------------------------------------
# InterMimic tensor layout constants
# ---------------------------------------------------------------------------
_INTERMIMIC_JOINT_OFFSET = 162
_INTERMIMIC_NUM_JOINTS = 52
_INTERMIMIC_OBJECT_OFFSET = 318


@DATASET_REGISTRY.register()
class OMOMOHolosomaDataset(DatasetBase):
    """
    Dataset class for the OMOMO data processed by Holosoma pipeline.

    Each sequence is stored as an individual ``.pt`` file containing an
    InterMimic-format tensor of shape ``(T, 591)``.  The tensor encodes:

    * offset 0-2: root position (3)
    * offset 3-6: root quaternion in wxyz (4)
    * offset 162-317: 52 SMPL-H joint world positions (52 x 3)
    * offset 318-324: object pose [x, y, z, qx, qy, qz, qw] (7)

    Unlike the original OMOMO pickle format, no FK is required — joint
    positions are pre-computed.  The 52 joints follow the
    :data:`SMPLH_KEYPOINTS_NAME` order.
    """

    _applies_frame_window: ClassVar[bool] = True

    registry: str = "Kashu7100"
    dataset: str = "eden_omomo_holosoma"

    file_extension: ClassVar[str] = ".pt"
    motion_sequence_keys: ClassVar[tuple[str, ...]] = ("human", "object")
    motion_dofs_name: ClassVar[dict[str, list[str]]] = {
        "human": [],
        "object": [],
    }
    motion_links_name: ClassVar[dict[str, list[str]]] = {
        "human": list(SMPLH_KEYPOINTS_NAME),
        "object": [],
    }

    def prepare(self) -> tuple[dict[str, BatchMotionSequence], dict[str, Any]]:
        en.logger.info(f"Loading {len(self.all_motion_files)} sequences from {self.raw_data_path}")

        # Parallel CPU read; bulk H2D moves once everything is packed.
        pairs = self._parallel_read_motion_files(
            self.all_motion_files,
            self._read_single_sequence,
            desc="OMOMO: reading raw .pt",
        )
        human_sequences: list[MotionSequence] = [h for h, _ in pairs]
        object_sequences: list[MotionSequence] = [o for _, o in pairs]

        # Object name parsing is cheap and order-sensitive, do it sequentially
        # against the (already-ordered) file list.
        object_names: list[str] = []
        for file_path in self.all_motion_files:
            # filename format: sub{id}_{object_name}_{clip_index}.pt
            seq_name = os.path.splitext(os.path.basename(file_path))[0]
            obj_name = "_".join(seq_name.split("_")[1:-1])
            assert obj_name in OMOMO_OBJECTS_REGISTRY, f"Object name '{obj_name}' not found in OMOMO_OBJECTS_REGISTRY"
            object_names.append(obj_name)

        motion_data = {
            "human": BatchMotionSequence.from_motion_sequences(
                human_sequences,
                num_links=len(self.motion_links_name["human"]),
            ).to(self.device),
            "object": BatchMotionSequence.from_motion_sequences(
                object_sequences,
            ).to(self.device),
        }

        metadata: dict[str, Any] = {
            "objects": object_names,
            "human_height": {
                "sub1": np.float32(1.6637051),
                "sub2": np.float32(1.8438001),
                "sub3": np.float32(1.9411048),
                "sub4": np.float32(1.7904495),
                "sub5": np.float32(1.8171062),
                "sub6": np.float32(1.7621962),
                "sub7": np.float32(1.8078182),
                "sub8": np.float32(1.8449197),
                "sub9": np.float32(1.753231),
                "sub10": np.float32(1.6509807),
                "sub11": np.float32(1.7666422),
                "sub12": np.float32(1.8030452),
                "sub13": np.float32(1.8110343),
                "sub14": np.float32(1.8263087),
                "sub15": np.float32(1.6846002),
                "sub16": np.float32(1.7229546),
                "sub17": np.float32(1.7366357),
            },
        }
        return motion_data, metadata

    def _read_single_sequence(self, file_path: str) -> tuple[MotionSequence, MotionSequence]:
        # CPU-only conversion; prepare() bulk-moves the packed batch.
        # `weights_only=True`: every file in the Holosoma export is a bare float32
        # tensor (verified across the full 4421-file `eden_omomo_holosoma` set), so
        # the pickle escape hatch buys nothing and only widens the trust boundary on
        # a hub-downloaded artifact. A file that needs it fails loudly here.
        data = torch.load(file_path, map_location="cpu", weights_only=True)
        data = to_numpy(data)[self._frame_window()]

        # -- Human --
        base_pos_w = data[:, :3]  # (T, 3)
        base_quat_w = data[:, 3:7]  # (T, 4) wxyz

        j_end = _INTERMIMIC_JOINT_OFFSET + _INTERMIMIC_NUM_JOINTS * 3
        links_pos_w = data[:, _INTERMIMIC_JOINT_OFFSET:j_end].reshape(-1, _INTERMIMIC_NUM_JOINTS, 3)  # (T, 52, 3)

        human_seq = self._pack_motion_sequence(
            framerate=OMOMO_FRAMERATE,
            base_pos_w=base_pos_w,
            base_quat_w=base_quat_w,
            links_pos_w=links_pos_w,
        )

        # -- Object --
        # Raw layout at offset 318: [x, y, z, qx, qy, qz, qw]
        raw_obj = data[:, _INTERMIMIC_OBJECT_OFFSET : _INTERMIMIC_OBJECT_OFFSET + 7]
        obj_pos = raw_obj[:, :3]  # (T, 3)
        # Convert [qx, qy, qz, qw] -> [qw, qx, qy, qz] (wxyz)
        obj_quat_wxyz = xyzw_to_wxyz(raw_obj[:, 3:7])  # (T, 4)

        obj_seq = self._pack_motion_sequence(
            framerate=OMOMO_FRAMERATE,
            base_pos_w=obj_pos,
            base_quat_w=obj_quat_wxyz,
        )

        return human_seq, obj_seq
