"""BVH/CSV G1 dataset for the BeyondMimic task."""

from __future__ import annotations

import hashlib
import json
import os
from typing import TYPE_CHECKING, ClassVar, Any

import numpy as np
import torch

from eden.datasets.base import DatasetBase, DATASET_REGISTRY, cache_identity
from eden.utils.assets import SAMPLE_DATA_REGISTRY, SAMPLE_DATA_REPO, sample_data_path
from eden.utils.misc import get_cache_dir
from eden.utils.motion_reference.motion_data import BatchMotionSequence

if TYPE_CHECKING:
    from eden.utils.motion_reference.motion_data import MotionSequence

# The one clip in ``beyondmimic/`` of the shared example-data repo, which the task defaults to.
DEFAULT_CLIP = "G1_gangnam_style_V01_bvh_60hz.csv"


def _default_processed_path(dataset: BvhCsvG1Dataset) -> str:
    """The processed-motion cache file for one raw source and one set of prepare settings.

    Keyed, not fixed, because :meth:`DatasetBase._initialize` takes the cache fast path *before*
    it looks at ``raw_data_path``: a fixed path would let ``--raw_data_path /my/clips`` load
    whatever the previous run prepared, with nothing logged. The cache root is per user, so
    "the previous run" spans every checkout and venv on the machine.

    The key is every input that decides the prepared bytes, which is exactly what
    :func:`~eden.datasets.base.cache_identity` reports plus the two inputs with checks of their
    own — the raw source, which nothing else records, and ``target_framerate``. Both of those
    latter checks are hard errors, so leaving them out of the key would not make a stale cache
    load; it would make two legitimate commands mutually exclusive, each rejecting the file the
    other wrote. Reading the rest off ``cache_identity`` keeps a newly added prepare-time field
    from silently missing the key.

    Parameters
    ----------
    dataset : BvhCsvG1Dataset
        Dataset whose settings key the file. Read after the options are resolved, so a
        ``--override`` of any keyed field lands in the key.

    Returns
    -------
    str
        Absolute path under the per-user Eden cache directory.
    """
    # ``is None``, not truthiness: ``--raw_data_path ""`` (an unset shell variable) must not
    # collapse onto the fetched default's key, which the fast path would then serve without ever
    # looking at the empty raw path. ``BeyondMimicDataMod.apply`` rejects the empty string
    # outright; this keeps the derivation honest on its own.
    source = (
        f"hf://{SAMPLE_DATA_REGISTRY}/{SAMPLE_DATA_REPO}/beyondmimic/{DEFAULT_CLIP}"
        if dataset.raw_data_path is None
        else os.path.abspath(os.path.expanduser(dataset.raw_data_path))
    )
    key = {
        "source": source,
        # ``float()`` so 50 and 50.0 — the same rate — cannot key two cache files.
        "target_framerate": float(dataset.target_framerate),
        **cache_identity(dataset),
    }
    digest = hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest()[:12]
    return get_cache_dir("beyondmimic", f"processed-{digest}.pt")


@DATASET_REGISTRY.register()
class BvhCsvG1Dataset(DatasetBase):
    """Dataset that loads BVH CSV files for G1 robot.

    CSV columns: base_pos(3), base_quat_xyzw(4), dofs_pos(29) = 36 total.
    """

    # No ``registry`` / ``dataset``: a *format* reader, not a corpus, so there is no repo to
    # download whole. ``raw_data_path=None`` resolves one named clip instead — see
    # :meth:`_download_from_hf`.

    input_fps: int = 60
    frame_range: tuple[int, int] | None = None
    # Both slice/retime the clip in ``_read_single_sequence``, so a cache built with one value
    # is not the cache for another — and the README tells users to set ``frame_range`` (row 1 of
    # the sample CSV is a retargeting artifact), which is exactly when this bites.
    _cache_identity_fields: ClassVar[tuple[str, ...]] = ("input_fps", "frame_range")

    file_extension: ClassVar[str] = ".csv"
    motion_sequence_keys: ClassVar[tuple[str, ...]] = ("g1",)
    motion_dofs_name: ClassVar[dict[str, list[str]]] = {
        "g1": [
            "left_hip_pitch_joint",
            "left_hip_roll_joint",
            "left_hip_yaw_joint",
            "left_knee_joint",
            "left_ankle_pitch_joint",
            "left_ankle_roll_joint",
            "right_hip_pitch_joint",
            "right_hip_roll_joint",
            "right_hip_yaw_joint",
            "right_knee_joint",
            "right_ankle_pitch_joint",
            "right_ankle_roll_joint",
            "waist_yaw_joint",
            "waist_roll_joint",
            "waist_pitch_joint",
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ],
    }
    # Same 30-link list as AMASSRetargetedG1Dataset (authored breadth-first from the MJCF).
    # This is a fixed name list resolved by name end-to-end, so its order is canonical, not
    # tied to Genesis's internal link ordering (which changed BFS->DFS in genesis 1.2.0).
    motion_links_name: ClassVar[dict[str, list[str]]] = {
        "g1": [
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
        ],
    }

    def _download_from_hf(self) -> str:
        """Resolve ``raw_data_path=None`` to the task's bundled sample clip.

        One named file out of the shared example-data repo, not a repo of its own: this class
        reads a format and the clip belongs to the task. The base implementation would need a
        ``registry`` / ``dataset``, and pointing those at the *shared* repo would fetch it
        whole and fold every ``.csv`` anyone adds to it into this task's motion library.

        ``DatasetBase._initialize`` calls this from inside its rank-0 gate, so under DDP only
        rank 0 fetches and the others take the prepared cache.
        """
        return sample_data_path("beyondmimic", DEFAULT_CLIP)

    def _initialize(self):
        """Derive the default cache path, then initialize normally.

        Here rather than in ``BeyondMimicDataMod.apply`` because a task mod runs *before*
        ``--override``, so a key built there would miss every setting the CLI can only reach
        that way — ``frame_range`` above all, which the sample clip's README calls a required
        trim and which :meth:`load_cached` rejects a mismatch on. Deriving it once the options
        are resolved also means an overridden ``raw_data_path`` keys its own cache instead of
        loading the sample clip's.

        The options keep ``processed_data_path=None`` so the saved config stays replayable on
        another machine; the resolved path is logged by ``_initialize`` on load and save.
        """
        if self.processed_data_path is None:
            self.processed_data_path = _default_processed_path(self)
        super()._initialize()

    def prepare(self) -> tuple[dict[str, BatchMotionSequence], dict[str, Any]]:
        all_motion_sequences = self._parallel_read_motion_files(
            self.all_motion_files, self._read_single_sequence, desc="BvhCsvG1: reading .csv"
        )
        batch = BatchMotionSequence.from_motion_sequences(
            all_motion_sequences,
            num_links=len(self.motion_links_name["g1"]),
        )

        from eden.options.robots.unitree_g1 import G1Dofs29

        self._forward_velocity_kinematics(batch, G1Dofs29(), link_names=self.motion_links_name["g1"])

        return {"g1": batch}, {}

    def _read_single_sequence(self, file_path: str) -> MotionSequence:
        data = np.loadtxt(file_path, delimiter=",")
        data = torch.as_tensor(data, dtype=torch.float32, device=self.device)

        if self.frame_range is not None:
            start, end = self.frame_range
            data = data[start:end]

        base_pos = data[:, 0:3]
        base_quat_xyzw = data[:, 3:7]
        # Convert xyzw → wxyz
        base_quat_wxyz = torch.cat([base_quat_xyzw[:, 3:4], base_quat_xyzw[:, 0:3]], dim=-1)
        dofs_pos = data[:, 7:]

        return self._pack_motion_sequence(
            framerate=self.input_fps,
            base_pos_w=base_pos,
            base_quat_w=base_quat_wxyz,
            dofs_pos=dofs_pos,
        )
