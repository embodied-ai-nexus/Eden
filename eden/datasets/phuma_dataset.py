"""PHUMA motion dataset."""

from __future__ import annotations
from typing import TYPE_CHECKING, ClassVar, Any

import numpy as np

import eden as en
from eden.datasets.base import DatasetBase, DATASET_REGISTRY
from eden.options.robots.unitree_g1 import G1_DOFS29_NAME, G1_DOFS29_TRACKED_LINKS_NAME
from eden.utils.geom import xyzw_to_wxyz
from eden.utils.motion_reference.motion_data import BatchMotionSequence

if TYPE_CHECKING:
    from eden.utils.motion_reference.motion_data import MotionSequence


@DATASET_REGISTRY.register()
class PHUMADataset(DatasetBase):
    """
    Dataset class for the PHUMA dataset.

    https://huggingface.co/datasets/DAVIAN-Robotics/PHUMA. Unset, ``raw_data_path`` downloads its ``data.zip`` into
    the Hugging Face cache and unzips it there on first use; set it to an already unpacked copy instead.
    """

    _applies_frame_window: ClassVar[bool] = True
    # Caches without the frame-window marker cut drop_first_n_frames correctly (a long drop_last_n_frames could wrap).
    _drops_cut_before_marker: ClassVar[tuple[str, ...]] = ("drop_first_n_frames",)

    registry: str = "DAVIAN-Robotics"
    dataset: str = "PHUMA"
    # The repo is a single ``data.zip``, so there is nothing to narrow at
    # download time; the g1 / h1_2 split only exists once it is unpacked.
    _hf_data_subdir: ClassVar[str] = "data/g1"

    file_extension: ClassVar[str] = ".npy"
    motion_sequence_keys: ClassVar[tuple[str, ...]] = ("g1",)
    motion_dofs_name: ClassVar[dict[str, list[str]]] = {"g1": G1_DOFS29_NAME}
    motion_links_name: ClassVar[dict[str, list[str]]] = {"g1": G1_DOFS29_TRACKED_LINKS_NAME}

    def _post_download_process(self, local_dir: str) -> str:
        """
        Unzip data.zip to local_dir, once (see ``DatasetBase._extract_once``).

        data.zip -> data/
        ├── g1/          # Humanoid configuration g1
        └── h1_2/        # Humanoid configuration h1_2

        Returns the extraction root; the base class narrows it to
        ``_hf_data_subdir`` (``data/g1``).
        """
        self._extract_once(local_dir, "data.zip", "data")
        return local_dir

    def prepare(self) -> tuple[dict[str, BatchMotionSequence], dict[str, Any]]:
        """
        Prepare the PHUMA dataset.

        Reads the ~73k per-motion .npy files in parallel on CPU, packs them
        with a single torch.cat per field, then moves the batched sequence to
        ``self.device`` for the Genesis-based forward-velocity FK pass.
        """
        return self._prepare_single_key(
            self._read_single_sequence,
            desc="PHUMA: reading raw .npy",
            robot=en.robots.G1Dofs29(dofs_name=self.motion_dofs_name["g1"]),
        )

    def _read_single_sequence(self, file_path: str) -> MotionSequence:
        # Read into CPU tensors; the batched sequence is moved to the dataset
        # device once after all motions are packed (see prepare()). This both
        # parallelises cleanly across threads and avoids fragmented H2D copies.
        data = np.load(file_path, allow_pickle=True).item()
        window = self._frame_window()
        return self._pack_motion_sequence(
            framerate=data["fps"],
            base_pos_w=data["root_trans"][window],
            base_quat_w=xyzw_to_wxyz(data["root_ori"][window]),
            dofs_pos=data["dof_pos"][window],
        )
