"""Synthetic motion clips on disk, a real ``DatasetBase`` that reads them, and mimic envs built over them."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from eden.datasets.base import DATASET_REGISTRY, DatasetBase
from eden.options.robots.unitree_g1 import G1_DOFS29_NAME
from eden.utils.motion_reference.motion_data import BatchMotionSequence

FRAMERATE = 50.0
ROBOT_DOFS = list(G1_DOFS29_NAME)
ROBOT_LINKS = ["pelvis", "torso_link", "left_ankle_roll_link", "right_ankle_roll_link"]
OBJECT_LINKS = ["box_baselink"]
#: Sequence key -> (dof names, link names). Keys double as the env entity names.
SEQUENCES = {"robot": (ROBOT_DOFS, ROBOT_LINKS), "object": ([], OBJECT_LINKS)}
ARM_DOFS = [f"joint{i}" for i in range(1, 8)] + ["finger_joint1", "finger_joint2"]  # FrankaEmikaPanda
#: The G1 and the box, plus a fixed-base Panda (``arm``) whose sequence is marked fixed-base.
TRACKING_SEQUENCES = SEQUENCES | {"arm": (ARM_DOFS, ["link7"])}
_FIELDS = ("base_pos_w", "base_quat_w", "dofs_pos", "links_pos_w", "links_quat_w")
_OPTIONAL_FIELDS = (
    "base_lin_vel_w",
    "base_ang_vel_w",
    "dofs_vel",
    "links_lin_vel_w",
    "links_ang_vel_w",
    "contact_points_w",
    "contact_points_valid",
    "contact_label",
)


class SyntheticMotionDataset(DatasetBase):
    """Reads the clips :func:`write_clip` writes: one ``.npz`` per clip, arrays keyed ``<key>.<field>``.

    Velocities a clip does not carry are finite-differenced by ``DatasetBase``, as for any
    position-only dataset. A ``<key>.is_free_floating`` scalar in the first clip marks that
    sequence fixed-base. Clips written with a ``label`` (see :func:`write_clip`) carry it as the
    per-clip ``"objects"`` metadata, which ``MotionTermOptions.env_motion_groups`` groups by.
    """

    _PER_CLIP_METADATA_KEYS = ("objects",)

    motion_sequence_keys = tuple(SEQUENCES)
    motion_dofs_name = {key: dofs for key, (dofs, _) in SEQUENCES.items()}
    motion_links_name = {key: links for key, (_, links) in SEQUENCES.items()}

    def prepare(self) -> tuple[dict[str, BatchMotionSequence], dict[str, Any]]:
        clips = [np.load(path) for path in sorted(self.all_motion_files)]
        self.all_motion_files = sorted(self.all_motion_files)
        sequences = {}
        for key in self.motion_sequence_keys:
            packed = []
            for clip in clips:

                def field(name):
                    name = f"{key}.{name}"
                    return torch.as_tensor(clip[name], device=self.device) if name in clip else None

                packed.append(
                    self._pack_motion_sequence(
                        float(clip["framerate"]),
                        field("base_pos_w"),
                        field("base_quat_w"),
                        **{name: field(name) for name in _FIELDS[2:] + _OPTIONAL_FIELDS},
                    )
                )
            batch = BatchMotionSequence.from_motion_sequences(packed, num_links=len(self.motion_links_name[key]))
            if f"{key}.is_free_floating" in clips[0]:
                batch.is_free_floating = bool(clips[0][f"{key}.is_free_floating"])
            sequences[key] = batch
        metadata = {"objects": [str(clip["label"]) for clip in clips]} if all("label" in c for c in clips) else {}
        return sequences, metadata


if "synthetic_motion_dataset" not in DATASET_REGISTRY:
    DATASET_REGISTRY.register(SyntheticMotionDataset)


def dataset_for(sequences: dict[str, tuple[list[str], list[str]]]) -> str:
    """The registry name of a :class:`SyntheticMotionDataset` over ``sequences``, registered on first use.

    ``sequences`` maps each key to its ``(dof names, link names)``, like :data:`SEQUENCES`.
    """
    if sequences == SEQUENCES:
        return "synthetic_motion_dataset"
    name = "synthetic_motion_dataset_" + "_".join(sequences)
    if name not in DATASET_REGISTRY:
        attrs = {
            "motion_sequence_keys": tuple(sequences),
            "motion_dofs_name": {key: list(dofs) for key, (dofs, _) in sequences.items()},
            "motion_links_name": {key: list(links) for key, (_, links) in sequences.items()},
        }
        class_name = "SyntheticMotionDataset" + "".join(key.title().replace("_", "") for key in sequences)
        DATASET_REGISTRY.register(type(class_name, (SyntheticMotionDataset,), attrs))
    return name


def robot_dofs_at(clip: int, frame: int) -> torch.Tensor:
    """The default ``robot.dofs_pos`` row of ``frame`` in ``clip``: ``0.01 c + 0.001 f + 0.0001 j``."""
    return 0.01 * clip + 0.001 * frame + 0.0001 * torch.arange(len(ROBOT_DOFS), dtype=torch.float32)


def robot_base_pos_at(clip: int, frame: int) -> torch.Tensor:
    """The default ``robot.base_pos_w`` of ``frame`` in ``clip``: ``(0.01 f, 0, 0.8 + 0.01 c)``."""
    return torch.tensor([0.01 * frame, 0.0, 0.8 + 0.01 * clip])


def _default_arrays(clip: int, num_frames: int, sequences) -> dict[str, np.ndarray]:
    frames = np.arange(num_frames, dtype=np.float32)
    arrays = {}
    for key, (dofs, links) in sequences.items():
        offset = 0.0 if key == "robot" else 1.0
        base_pos = np.stack(
            [offset + 0.01 * frames, np.zeros(num_frames), np.full(num_frames, 0.8 + 0.01 * clip)], axis=-1
        )
        identity = np.zeros((num_frames, 4))
        identity[:, 0] = 1.0
        arrays[f"{key}.base_pos_w"] = base_pos
        arrays[f"{key}.base_quat_w"] = identity
        arrays[f"{key}.dofs_pos"] = 0.01 * clip + 0.001 * frames[:, None] + 0.0001 * np.arange(len(dofs))[None]
        # Link l sits 0.1 l to the side of the base.
        link_offsets = np.stack([np.zeros(len(links)), 0.1 * np.arange(len(links)), np.zeros(len(links))], axis=-1)
        arrays[f"{key}.links_pos_w"] = base_pos[:, None, :] + link_offsets[None]
        arrays[f"{key}.links_quat_w"] = np.repeat(identity[:, None, :], len(links), axis=1)
    return {name: arr.astype(np.float32) for name, arr in arrays.items()}


def write_clip(
    path: Path,
    num_frames: int,
    *,
    clip: int = 0,
    framerate: float = FRAMERATE,
    overrides: dict[str, np.ndarray] | None = None,
    sequences=SEQUENCES,
    label: str | None = None,
) -> Path:
    """Write one clip with the default ``(clip, frame)``-encoded content, then apply ``overrides``.

    Parameters
    ----------
    path : Path
        Destination ``.npz`` file.
    num_frames : int
        Clip length in frames.
    clip : int
        Index encoded into the default values (see :func:`robot_dofs_at`).
    framerate : float
        Source framerate; the dataset retimes to ``target_framerate`` when they differ.
    overrides : dict[str, np.ndarray] | None
        ``<key>.<field>`` arrays replacing (or adding to) the defaults.
    sequences : dict
        The sequences to write, as in :func:`dataset_for`.
    label : str | None
        The clip's per-clip ``"objects"`` label; ``None`` writes none.

    Returns
    -------
    Path
        ``path``.
    """
    arrays = _default_arrays(clip, num_frames, sequences)
    arrays.update(overrides or {})
    if label is not None:
        arrays["label"] = np.array(label)
    np.savez(path, framerate=np.array(framerate), **arrays)
    return path


def write_library(
    root: Path,
    lengths: list[int],
    *,
    framerate: float = FRAMERATE,
    sequences=SEQUENCES,
    labels: list[str] | None = None,
) -> Path:
    """Write clips ``clip_00.npz``, ``clip_01.npz``, ... of the given lengths under ``root``; returns ``root``.

    ``labels`` gives each clip its per-clip ``"objects"`` label (see :func:`write_clip`).
    """
    root.mkdir(parents=True, exist_ok=True)
    for i, num_frames in enumerate(lengths):
        label = labels[i] if labels is not None else None
        write_clip(
            root / f"clip_{i:02d}.npz", num_frames, clip=i, framerate=framerate, sequences=sequences, label=label
        )
    return root


def load_dataset(root: Path, **dataset_kwargs) -> SyntheticMotionDataset:
    """Load the library under ``root`` as a standalone dataset on CPU."""
    from eden.options.datasets import DatasetOptions

    options = DatasetOptions(name="synthetic_motion_dataset", raw_data_path=str(root), **dataset_kwargs)
    return SyntheticMotionDataset(options, device=torch.device("cpu"))


def term_options(
    root: Path,
    *,
    entities: tuple[str, ...] | None = None,
    dofs: dict[str, list[str]] | None = None,
    links: dict[str, list[str]] | None = None,
    dataset: dict | None = None,
    sequences=SEQUENCES,
    **term_kwargs,
):
    """``MotionTermOptions`` over the library at ``root``, tracking ``entities`` (default: every sequence).

    Every DOF and link is targeted unless ``dofs`` / ``links`` name a subset for a key; ``dataset``
    holds extra ``DatasetOptions`` fields; ``sequences`` is what the library was written with.
    """
    from eden.options.datasets import DatasetOptions
    from eden.options.managers.motions import MotionTermOptions

    # Every dataset key needs its targets, tracked or not: the term resolves them for each sequence.
    dofs = {key: list(names) for key, (names, _) in sequences.items()} | (dofs or {})
    links = {key: list(names) for key, (_, names) in sequences.items()} | (links or {})
    return MotionTermOptions(
        name="motion_term",
        dataset_options=DatasetOptions(name=dataset_for(sequences), raw_data_path=str(root), **(dataset or {})),
        entities_map={name: name for name in (entities or tuple(sequences))},
        target_dofs_name=dofs,
        target_links_name=links,
        **term_kwargs,
    )


def manager_options(root: Path | None = None, *, terms: dict | None = None, **manager_kwargs):
    """``MotionManagerOptions`` with one ``default`` term over ``root``, or the given ``terms``."""
    from eden.options.managers.motions import MotionManagerOptions

    manager_kwargs.setdefault("frame_interval_s_range", (1.0 / FRAMERATE, 1.0 / FRAMERATE))
    manager_kwargs.setdefault("data_start_from", "current_time")
    terms = terms if terms is not None else {"default": term_options(root)}
    return MotionManagerOptions(**manager_kwargs, **terms)


def make_manager(env, root: Path | None = None, *, terms: dict | None = None, **manager_kwargs):
    """A standalone ``MotionManager`` over the library at ``root``, run against the built ``env``."""
    from eden.managers.motion_manager import MotionManager

    return MotionManager(env, manager_options(root, terms=terms, **manager_kwargs))


def build_mimic_env(
    root: Path,
    *,
    num_envs: int = 4,
    motion_options=None,
    scene: dict | None = None,
    env_options: dict | None = None,
    reset: bool = True,
    **config_kwargs,
):
    """Build a ``MimicRLEnvBase`` over the library at ``root``.

    The scene is a G1 (``robot``) and a free box (``object``) unless ``scene`` gives the entity
    options; ``env_options`` adds ``EnvOptions`` fields and ``config_kwargs`` go to ``EdenMimicConfig``
    (``termination_options``, ``event_options``, ...).
    """
    import eden as en
    from eden.options.entities import BoxOptions
    from eden.options.envs import EnvOptions
    from eden.options.robots.unitree_g1 import G1Dofs29
    from eden.utils.configs import EdenMimicConfig
    from tests.support.envs import build_env

    if scene is None:
        scene = {"robot": G1Dofs29(), "object": BoxOptions(size=(0.1, 0.1, 0.1), default_root_pos=(1.0, 0.0, 0.05))}
    config = EdenMimicConfig(
        env_options=EnvOptions(num_envs=num_envs, **(env_options or {})),
        scene_options=en.scenes.Empty(**scene),
        motion_options=motion_options if motion_options is not None else manager_options(root),
        **config_kwargs,
    )
    return build_env(config, reset=reset)


def build_tracking_env(root: Path, *, num_frames: int = 9, **config_kwargs):
    """A mimic env for the tracking terms: G1 (``robot``), box (``object``) and fixed Panda (``arm``).

    The library at ``root`` must hold :data:`TRACKING_SEQUENCES` (see :func:`write_tracking_library`);
    the reference buffers hold ``num_frames`` frames starting at the current time, which tests
    overwrite through :func:`set_reference`.
    """
    from eden.options.entities import BoxOptions
    from eden.options.robots.franka_emika_panda import FrankaEmikaPanda
    from eden.options.robots.unitree_g1 import G1Dofs29

    scene = {
        "robot": G1Dofs29(),
        "object": BoxOptions(size=(0.1, 0.1, 0.1), default_root_pos=(1.0, 0.0, 0.05)),
        "arm": FrankaEmikaPanda(default_root_pos=(0.0, 2.0, 0.0)),
    }
    motion = manager_options(terms={"default": term_options(root, sequences=TRACKING_SEQUENCES)}, num_frames=num_frames)
    return build_mimic_env(root, scene=scene, motion_options=motion, **config_kwargs)


def write_tracking_library(root: Path) -> Path:
    """One 20-frame clip of :data:`TRACKING_SEQUENCES`, with the arm's sequence marked fixed-base."""
    root.mkdir(parents=True, exist_ok=True)
    write_clip(
        root / "clip_00.npz", 20, sequences=TRACKING_SEQUENCES, overrides={"arm.is_free_floating": np.array(False)}
    )
    return root


def set_reference(env, entity: str, **fields: torch.Tensor) -> None:
    """Overwrite fields of ``env.motion_manager.data[entity]`` (broadcast to the buffer's shape).

    The terms under test read this buffer; writing it directly states the reference a test wants
    instead of reverse-engineering a clip and a playback time that would produce it.
    """
    ref = env.motion_manager.data[entity]
    for name, value in fields.items():
        buffer = getattr(ref, name)
        buffer.copy_(torch.as_tensor(value, dtype=buffer.dtype, device=buffer.device).expand_as(buffer))


def pin_clips(term, motion_ids, start_s=0.0) -> None:
    """Put each of ``term``'s envs on a known clip and start phase.

    Selection and start phase are sampled at random by design and have no public setter; the
    tests that need a known ``(clip, frame)`` write the two per-env buffers here, and only here.
    """
    device = term._assigned_env_motion_selection.device
    term._assigned_env_motion_selection[:] = torch.as_tensor(motion_ids, device=device)
    term._motion_buffer_start_time_s[:] = torch.as_tensor(start_s, dtype=torch.float32, device=device)
