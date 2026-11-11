"""Shared by the dataset tests: a small real ``DatasetBase`` subclass over ``.npz`` clips written into ``tmp_path``,
a synthetic MANO model, and box meshes.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest
import torch
import trimesh

from eden.datasets.base import DatasetBase
from eden.utils.motion_reference.motion_data import BatchMotionSequence, MotionSequence

DOFS_NAME = ["a", "b", "c"]
FRAMERATE = 50.0  # the dataset's target_framerate, so clips are not retimed


class ClipDataset(DatasetBase):
    """One motion key, ``g1``, read from ``.npz`` clips holding ``dofs_pos`` / ``base_pos_w`` / ``base_quat_w``.

    ``frame_range`` and ``input_fps`` are prepare-time settings recorded in the cache identity, like
    ``BvhCsvG1Dataset``'s; ``frame_range`` really slices each clip.
    """

    motion_sequence_keys: ClassVar[tuple[str, ...]] = ("g1",)
    motion_dofs_name: ClassVar[dict[str, list[str]]] = {"g1": DOFS_NAME}
    motion_links_name: ClassVar[dict[str, list[str]]] = {"g1": []}
    _cache_identity_fields: ClassVar[tuple[str, ...]] = ("frame_range", "input_fps")

    frame_range: tuple[int, int] | None = None
    input_fps: int = 60

    def prepare(self):
        clips = self._parallel_read_motion_files(self.all_motion_files, self._read_single_sequence)
        return {"g1": BatchMotionSequence.from_motion_sequences(clips, num_links=0)}, {}

    def _read_single_sequence(self, path: str) -> MotionSequence:
        data = np.load(path)
        window = slice(*self.frame_range) if self.frame_range is not None else slice(None)
        return self._pack_motion_sequence(
            framerate=float(data["framerate"]),
            base_pos_w=torch.from_numpy(data["base_pos_w"][window]),
            base_quat_w=torch.from_numpy(data["base_quat_w"][window]),
            dofs_pos=torch.from_numpy(data["dofs_pos"][window]) if "dofs_pos" in data.files else None,
        )


def write_clip(
    path: Path, dofs_pos: np.ndarray | None = None, *, num_frames: int = 8, framerate: float = FRAMERATE
) -> Path:
    """Write one clip; ``dofs_pos`` defaults to ``arange`` values, and ``None`` columns means a clip without DOFs.

    Parameters
    ----------
    path : Path
        Where to write the ``.npz``; parent directories are created.
    dofs_pos : np.ndarray | None
        ``(T, num_dofs)`` joint positions, or ``None`` for ``T = num_frames`` of ``arange`` data over
        ``DOFS_NAME``. An array with zero columns writes a clip with no ``dofs_pos`` at all.
    num_frames : int
        Frames when ``dofs_pos`` is ``None``.
    framerate : float
        The clip's recorded rate; anything but ``FRAMERATE`` makes the dataset retime it.

    Returns
    -------
    Path
        ``path``.
    """
    if dofs_pos is None:
        dofs_pos = np.arange(num_frames * len(DOFS_NAME), dtype=np.float32).reshape(num_frames, -1) * 0.01
    dofs_pos = np.asarray(dofs_pos, dtype=np.float32)
    frames = dofs_pos.shape[0]
    quat = np.zeros((frames, 4), np.float32)
    quat[:, 0] = 1.0
    fields = {"framerate": np.float32(framerate), "base_pos_w": np.zeros((frames, 3), np.float32), "base_quat_w": quat}
    if dofs_pos.shape[1]:
        fields["dofs_pos"] = dofs_pos
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **fields)
    return path


HUB_COMMIT = "0" * 40


class OfflineHub:
    """A Hugging Face cache in ``tmp_path`` with the client held offline, so a fetch resolves to a seeded snapshot.

    ``calls`` records the keyword arguments of every fetch — how a test sees what a dataset asked the Hub
    for (``allow_patterns``, ``repo_id``), since offline the client hands back the whole snapshot.
    """

    def __init__(self, cache: Path):
        self.cache = cache
        self.calls: list[dict] = []

    def snapshot(self, repo_id: str) -> Path:
        """Create ``repo_id``'s dataset snapshot directory and return it, for the test to fill."""
        folder = self.cache / f"datasets--{repo_id.replace('/', '--')}"
        snapshot = folder / "snapshots" / HUB_COMMIT
        snapshot.mkdir(parents=True)
        (folder / "refs").mkdir()
        (folder / "refs" / "main").write_text(HUB_COMMIT)
        return snapshot


@pytest.fixture
def hub(tmp_path, monkeypatch) -> OfflineHub:
    """An :class:`OfflineHub`. The network boundary is the Hub client: each call is recorded, then made for real."""
    import huggingface_hub.constants

    from eden.utils import assets

    hub = OfflineHub(tmp_path / "hub")
    hub.cache.mkdir()
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_CACHE", str(hub.cache))
    monkeypatch.setattr(huggingface_hub.constants, "HF_HUB_OFFLINE", True)
    client = assets.snapshot_download

    def recording_client(**kwargs):
        hub.calls.append(kwargs)
        return client(**kwargs)

    monkeypatch.setattr(assets, "snapshot_download", recording_client)
    return hub


@pytest.fixture
def make_dataset(tmp_path):
    """Build a ``ClipDataset`` (or subclass) on the CPU: ``make_dataset(cls=ClipDataset, **options)``."""

    def make(cls=ClipDataset, **options):
        return cls(options=cls.configure(**options), device=torch.device("cpu"))

    return make


#: Vertices of the synthetic MANO template.
MANO_VERTS = 778


def _mano_template() -> np.ndarray:
    return np.random.default_rng(0).uniform(-0.05, 0.05, (MANO_VERTS, 3))


@pytest.fixture(scope="session")
def mano_dir(tmp_path_factory) -> Path:
    """A real MANO-format ``.pkl`` pair (the actual MANO models are licence-gated): no pose blend shapes, joint ``k``
    sits on template vertex ``k`` and vertex ``i`` follows joint ``i % 16``, so at zero pose and zero ``betas`` the
    layer returns the template exactly and joint 0 (the wrist) is ``template[0]``. The shape blend shapes and the
    mean hand are not zero, so a root offset or a mean pose that is dropped shows up.
    """
    # smplx (the dev group) is the reference the tests build their synthetic releases with; this .pkl holds plain
    # numpy arrays, so Eden reads it without chumpy.
    pytest.importorskip("smplx", reason="the reference body-model implementation (the dev dependency group)")
    out = tmp_path_factory.mktemp("mano")
    J = np.zeros((16, MANO_VERTS))
    J[np.arange(16), np.arange(16)] = 1.0
    weights = np.zeros((MANO_VERTS, 16))
    weights[np.arange(MANO_VERTS), np.arange(MANO_VERTS) % 16] = 1.0
    parents = [4294967295, 0, 1, 2, 0, 4, 5, 0, 7, 8, 0, 10, 11, 0, 13, 14]
    data = {
        "v_template": _mano_template(),
        "shapedirs": np.random.default_rng(1).normal(0.0, 0.01, (MANO_VERTS, 3, 10)),
        "posedirs": np.zeros((MANO_VERTS, 3, 135)),
        "J_regressor": J,
        "weights": weights,
        "kintree_table": np.array([parents, list(range(16))], dtype=np.int64),
        "f": np.array([[0, 1, 2]]),
        # A permutation, not the identity, so a PCA expansion that is skipped shows up in the output.
        "hands_components": np.eye(45)[::-1].copy(),
        "hands_mean": np.linspace(-0.2, 0.2, 45),
    }
    for side in ("RIGHT", "LEFT"):
        with open(out / f"MANO_{side}.pkl", "wb") as f:
            pickle.dump(data, f)
    return out


@pytest.fixture(scope="session")
def mano_template() -> np.ndarray:
    return _mano_template()


def write_box(path: Path, extents=(0.1, 0.1, 0.1), scale: float = 1.0) -> trimesh.Trimesh:
    """An axis-aligned box centred on its frame's origin, written in ``scale`` units per metre."""
    mesh = trimesh.creation.box(extents=extents)
    out = mesh.copy()
    out.apply_scale(scale)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.export(path)
    return mesh


def pose_track(pos: np.ndarray) -> np.ndarray:
    """(T, 4, 4) identity-rotation poses at ``pos`` (T, 3)."""
    out = np.tile(np.eye(4), (len(pos), 1, 1))
    out[:, :3, 3] = pos
    return out
