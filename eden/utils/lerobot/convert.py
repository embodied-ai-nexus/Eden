"""Convert an Eden teleoperation HDF5 recording into a LeRobot v3 dataset.

Used as a library:

    from eden.utils.lerobot import convert_hdf5_to_lerobot
    convert_hdf5_to_lerobot([...])  # argparse-style argv

or via CLI: ``eden lerobot convert --hdf5 ... --output-dir ...``.
"""

from __future__ import annotations

import importlib.util
import shutil
import sys
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from eden.utils.file_handler.base import DEMO_NAME_RE as _DEMO_NAME_PATTERN
from eden.utils.format import ANSI_BOLD, ANSI_CYAN, ANSI_DIM, ANSI_GREEN, ANSI_YELLOW, ColorHelpFormatter, colorize


def _normalize_timestamps(timestamps: np.ndarray) -> np.ndarray:
    timestamps = np.asarray(timestamps)
    if timestamps.ndim == 2 and timestamps.shape[1] == 1:
        timestamps = timestamps[:, 0]
    if timestamps.ndim != 1:
        raise ValueError(f"Expected timestamps to have shape (T,) or (T, 1), but got {timestamps.shape}.")
    return timestamps.astype(np.float32, copy=False)


def _demo_sort_key(name: str) -> tuple[int, int | str, str]:
    match = _DEMO_NAME_PATTERN.fullmatch(name)
    if match is not None:
        return (0, int(match.group(1)), name)
    return (1, name, name)


def _list_demos(hdf5_file: h5py.File) -> list[str]:
    if "data" not in hdf5_file:
        raise ValueError("Expected a top-level 'data' group in the HDF5 file.")

    demos = sorted(
        (name for name in hdf5_file["data"].keys() if _DEMO_NAME_PATTERN.fullmatch(name)),
        key=_demo_sort_key,
    )
    if not demos:
        raise ValueError(
            "No valid demo groups found under /data in the HDF5 file. Expected names matching 'demo_<number>'."
        )
    return demos


def _infer_fps(timestamps: np.ndarray) -> int:
    timestamps = _normalize_timestamps(timestamps)
    if len(timestamps) < 2:
        raise ValueError("Need at least two timestamps to infer FPS.")

    deltas = np.diff(timestamps.astype(np.float64))
    valid_deltas = deltas[np.isfinite(deltas) & (deltas > 0)]
    if len(valid_deltas) == 0:
        raise ValueError("Could not infer FPS because timestamp deltas are invalid.")

    mean_dt = float(np.mean(valid_deltas))
    fps = int(round(1.0 / mean_dt))
    if fps <= 0:
        raise ValueError(f"Inferred non-positive FPS from timestamps: mean_dt={mean_dt}")
    return fps


def _demo_feature_dims(demo_group: h5py.Group) -> tuple[int, int, int, int, int]:
    """Return (qpos_dim, dofs_vel_dim, pos_dim, quat_dim, action_dim) for a demo group."""
    qpos = demo_group["state/qpos"]
    dofs_vel = demo_group["state/dofs_vel"]
    pos = demo_group["state/pos"]
    quat = demo_group["state/quat"]
    action = demo_group["action"]
    return (qpos.shape[1], dofs_vel.shape[1], pos.shape[1], quat.shape[1], action.shape[1])


def _build_features(hdf5_file: h5py.File, demos: list[str]) -> dict[str, dict[str, Any]]:
    qpos_dim, dofs_vel_dim, pos_dim, quat_dim, action_dim = _demo_feature_dims(hdf5_file[f"data/{demos[0]}"])

    # StateRecorder emits root (base-link) pose, so these are root_pos / root_quat —
    # not end-effector pose — one 3 + 4 per recorded entity (two for a two-arm task, concatenated
    # in order). Asserting shape catches silent changes in the recorder schema.
    num_roots = pos_dim // 3
    if pos_dim == 0 or pos_dim % 3:
        raise ValueError(f"Expected state/pos shape (T, 3 * entities) for root positions, got dim={pos_dim}.")
    if quat_dim != 4 * num_roots:
        raise ValueError(
            f"Expected state/quat shape (T, {4 * num_roots}) for {num_roots} root quaternion(s) (wxyz) to go with "
            f"state/pos dim={pos_dim}, got dim={quat_dim}."
        )
    roots = [""] if num_roots == 1 else [str(i) for i in range(num_roots)]

    # Refuse to mix demos with mismatched feature dims — the LeRobot writer expects a
    # single fixed schema, and silent truncation/padding would produce a corrupt dataset.
    expected = (qpos_dim, dofs_vel_dim, pos_dim, quat_dim, action_dim)
    for demo in demos[1:]:
        actual = _demo_feature_dims(hdf5_file[f"data/{demo}"])
        if actual != expected:
            raise ValueError(
                f"Feature dim mismatch in {demo}: "
                f"expected (qpos, dofs_vel, pos, quat, action) = {expected}, got {actual}."
            )

    state_dim = qpos_dim + dofs_vel_dim + pos_dim + quat_dim
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": (int(state_dim),),
            "names": [
                *[f"qpos_{i}" for i in range(qpos_dim)],
                *[f"dofs_vel_{i}" for i in range(dofs_vel_dim)],
                *[f"root{root}_pos_{axis}" for root in roots for axis in ("x", "y", "z")],
                *[f"root{root}_quat_{c}" for root in roots for c in ("w", "x", "y", "z")],
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (int(action_dim),),
            "names": [f"action_{i}" for i in range(action_dim)],
        },
    }


def _looks_like_lerobot_dataset(path: Path) -> bool:
    # A LeRobot dataset root always contains a meta/info.json. Without this marker we
    # refuse to delete, so a typo in --output-dir cannot wipe an unrelated directory.
    return (path / "meta" / "info.json").is_file()


#: How to install what the converter imports: lerobot 0.4 or later, the first to write LeRobot v3 datasets, with its
#: ``dataset`` extra, where lerobot 0.6+ keeps its dataset code (datasets, pandas, pyarrow, av, torchcodec) instead of
#: its core. ``-U``, because without it pip leaves an older lerobot that is already installed in place.
LEROBOT_INSTALL = 'pip install -U "lerobot[dataset]>=0.4"'


def _import_lerobot_dataset() -> type:
    """``LeRobotDataset``, or an ``ImportError`` that names the install command when lerobot is absent, predates the
    ``lerobot.datasets`` package, or predates LeRobot v3. A lerobot without its ``dataset`` extra raises its own error
    naming that extra.
    """
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
    except ModuleNotFoundError as e:
        if e.name is None or not (e.name == "lerobot" or e.name.startswith("lerobot.")):
            raise  # a module lerobot itself imports is missing: its own message says which
        # Asked of the import system, not read off e.name, which names the child when an import of lerobot is blocked.
        try:
            installed = importlib.util.find_spec("lerobot") is not None
        except ValueError:  # already imported, without a spec
            installed = True
        found = f"the installed lerobot has no {e.name}" if installed else "lerobot is not installed"
        raise ImportError(
            f"eden lerobot convert writes LeRobot v3 datasets through lerobot's LeRobotDataset, but {found}. "
            f"{'Upgrade' if installed else 'Install'} it with `{LEROBOT_INSTALL}`: a plain `pip install lerobot` "
            "leaves out the dataset code the converter needs. No Eden extra installs it."
        ) from e
    if not hasattr(LeRobotDataset, "finalize"):  # lerobot < 0.4: LeRobot v2.1, which has no finalize()
        version = getattr(sys.modules.get("lerobot"), "__version__", "unknown")
        raise ImportError(
            "eden lerobot convert writes LeRobot v3 datasets, which lerobot 0.4 introduced, but the installed "
            f"lerobot ({version}) predates them and would fail only after writing every episode. "
            f"Upgrade it with `{LEROBOT_INSTALL}`."
        )
    return LeRobotDataset


def convert(
    hdf5_path: Path,
    output_dir: Path,
    repo_id: str,
    robot_type: str,
    task: str,
    force: bool = False,
    image_writer_threads: int = 4,
) -> None:
    """Convert an HDF5 recording into a LeRobot v3 dataset on disk."""
    # Before --force deletes anything: a missing or outdated lerobot must not cost the user their old dataset.
    LeRobotDataset = _import_lerobot_dataset()

    if not hdf5_path.is_file():
        raise FileNotFoundError(f"Input HDF5 file not found: {hdf5_path}")

    if output_dir.exists():
        if not force:
            raise FileExistsError(f"Output directory already exists: {output_dir}")
        if output_dir.is_symlink() or not output_dir.is_dir():
            raise ValueError(f"Refusing to --force overwrite non-directory path: {output_dir}")
        if any(output_dir.iterdir()) and not _looks_like_lerobot_dataset(output_dir):
            raise ValueError(
                f"Refusing to --force overwrite {output_dir}: it is not empty and does not look "
                "like a LeRobot dataset (missing meta/info.json). Delete it manually if intended."
            )
        shutil.rmtree(output_dir)

    with h5py.File(hdf5_path, "r") as hdf5_file:
        demos = _list_demos(hdf5_file)
        features = _build_features(hdf5_file, demos)
        fps = _infer_fps(hdf5_file[f"data/{demos[0]}/timestamp"][:])

        dataset = LeRobotDataset.create(
            repo_id=repo_id,
            root=output_dir,
            fps=fps,
            robot_type=robot_type,
            features=features,
            use_videos=True,
            image_writer_threads=image_writer_threads,
        )

        for demo in demos:
            demo_group = hdf5_file[f"data/{demo}"]
            # Bulk-load each demo's arrays once instead of per-step h5 indexing.
            actions = np.asarray(demo_group["action"][:], dtype=np.float32)
            timestamps = _normalize_timestamps(demo_group["timestamp"][:])
            qpos = np.asarray(demo_group["state/qpos"][:], dtype=np.float32)
            dofs_vel = np.asarray(demo_group["state/dofs_vel"][:], dtype=np.float32)
            pos = np.asarray(demo_group["state/pos"][:], dtype=np.float32)
            quat = np.asarray(demo_group["state/quat"][:], dtype=np.float32)

            expected_lengths = {
                "action": actions.shape[0],
                "timestamp": timestamps.shape[0],
                "qpos": qpos.shape[0],
                "dofs_vel": dofs_vel.shape[0],
                "pos": pos.shape[0],
                "quat": quat.shape[0],
            }
            if len(set(expected_lengths.values())) != 1:
                raise ValueError(f"Length mismatch in {demo}: {expected_lengths}")

            states = np.concatenate([qpos, dofs_vel, pos, quat], axis=1)
            length = int(actions.shape[0])
            for t in range(length):
                dataset.add_frame(
                    {
                        "observation.state": states[t],
                        "action": actions[t],
                        "task": task,
                    }
                )

            dataset.save_episode()

        dataset.finalize()


def convert_hdf5_to_lerobot(argv: list[str] | None = None):
    """CLI entry point: ``eden lerobot convert ...``."""
    from argparse import ArgumentParser

    description = (
        f"{colorize('Convert an Eden HDF5 recording into a LeRobot v3 dataset.', ANSI_BOLD, ANSI_CYAN)}\n"
        "\n"
        "Reads `state/qpos`, `state/dofs_vel`, `state/pos`, `state/quat`, `action`, and\n"
        "`timestamp` from each `demo_*` group under `/data` (the schema produced by\n"
        f"{colorize('StateRecorder', ANSI_GREEN)} + {colorize('ActionRecorder', ANSI_GREEN)}), "
        "then writes a LeRobotDataset rooted at\n"
        f"{colorize('--output-dir', ANSI_GREEN)}."
    )
    epilog = (
        f"{colorize('Examples:', ANSI_BOLD, ANSI_YELLOW)}\n"
        f"  {colorize('# Convert a teleop recording with the default robot_type:', ANSI_DIM)}\n"
        "  eden lerobot convert \\\n"
        f"      {colorize('--hdf5', ANSI_GREEN)} eden_teleoperation/teleoperation.hdf5 \\\n"
        f"      {colorize('--output-dir', ANSI_GREEN)} lerobot_output \\\n"
        f"      {colorize('--repo-id', ANSI_GREEN)} local/teleop \\\n"
        f"      {colorize('--force', ANSI_GREEN)}"
    )
    parser = ArgumentParser(
        prog="eden lerobot convert",
        description=description,
        epilog=epilog,
        formatter_class=ColorHelpFormatter,
    )
    parser.add_argument("--hdf5", type=Path, required=True, help="Path to the input HDF5 file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where the LeRobot dataset will be created.",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default="local/eden_teleoperation",
        help="LeRobot dataset repo_id.",
    )
    parser.add_argument(
        "--robot-type",
        type=str,
        default="eden",
        help="Robot type recorded in LeRobot metadata.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="teleoperate the robot",
        help="Task string stored for every frame.",
    )
    parser.add_argument(
        "--image-writer-threads",
        type=int,
        default=4,
        help="Number of image writer threads used by LeRobot when encoding videos.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite --output-dir if it already exists (must be empty or a LeRobot dataset).",
    )
    args = parser.parse_args(argv)

    convert(
        hdf5_path=args.hdf5,
        output_dir=args.output_dir,
        repo_id=args.repo_id,
        robot_type=args.robot_type,
        task=args.task,
        force=args.force,
        image_writer_threads=args.image_writer_threads,
    )
