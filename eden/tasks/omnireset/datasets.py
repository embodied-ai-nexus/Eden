"""UWLab's recorded OmniReset reset states, as a :class:`~eden.utils.buffers.reset_states.ResetStateBuffer`.

A UWLab file (``resets_<family>.pt``) holds ``initial_state`` rows of Isaac Lab's ``InteractiveScene.get_state``:
per articulation ``root_pose`` / ``root_velocity`` / ``joint_position`` / ``joint_velocity``, per rigid object
``root_pose`` / ``root_velocity``, positions relative to the env origin and quaternions xyzw (the file says so in
``quat_convention``). The conversion:

- quaternions become wxyz; positions stay as they are (Eden's envs share one world frame);
- the robot's joint columns, in :data:`~.constants.ISAAC_JOINT_ORDER`, are stored per joint name. Their values are
  PhysX's joint coordinates, which the converted robot reproduces (:class:`~eden.options.robots.OmniResetUR5e2f85`);
- the robot base, its mount plate and the receptive object are fixed bodies UWLab places per row, so they are
  stored as fixed roots (the scene builds them with ``batch_fixed_verts``); the table, which never moves, is
  checked against the scene's and dropped;
- the insertive object's velocity is Isaac Lab's centre-of-mass velocity; the buffer wants its authored origin's,
  ``v_origin = v_com + w x (p_origin - p_com)``.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from eden.tasks.omnireset.constants import (
    DATASETS_REVISION,
    INSERTIVE,
    ISAAC_JOINT_ORDER,
    MOUNT,
    RECEPTIVE,
    RESET_FAMILIES,
    ROBOT,
    TABLE_POS,
    TABLE_QUAT,
    UWLAB_REPO,
    Pair,
)
from eden.utils.buffers.reset_states import EntityStates, ResetStateBuffer
from eden.utils.geom import transform_by_quat, xyzw_to_wxyz

#: Version of :func:`load_family`'s output, part of the cached buffer's name (:func:`buffer_path`). Bump it with any
#: change to what the conversion writes, so caches made by the old code are not read back.
CONVERSION_VERSION = 1


def _stack(rows: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack(rows).to(torch.float32)


def _pose(state: dict) -> tuple[torch.Tensor, torch.Tensor]:
    pose = _stack(state["root_pose"])
    return pose[:, :3], xyzw_to_wxyz(pose[:, 3:])


def _require_at_rest(state: dict, what: str, path: str) -> None:
    speed = _stack(state["root_velocity"]).abs().max().item()
    if speed > 1e-6:
        raise ValueError(f"{path}: {what} is fixed in the scene but moves in the file (|v| up to {speed:.3g}).")


def load_family(path: str, pair: Pair, insertive_com: Sequence[float]) -> ResetStateBuffer:
    """One UWLab ``resets_<family>.pt`` as a buffer; ``insertive_com`` is the insertive object's CoM in its frame."""
    data = torch.load(path, map_location="cpu", weights_only=True)
    if data.get("quat_convention") != "xyzw":
        raise ValueError(
            f"{path} has quat_convention={data.get('quat_convention')!r}, expected 'xyzw'. Use the Isaac Lab 3 datasets "
            f"(Datasets/OmniReset_isaaclab3 on the isaaclab3 branch of {UWLAB_REPO})."
        )
    state = data["initial_state"]
    robot, objects = state["articulation"]["robot"], state["rigid_object"]

    table_pos, table_quat = _pose(objects["table"])
    expected_pos, expected_quat = torch.tensor(TABLE_POS), torch.tensor(TABLE_QUAT)
    if (table_pos - expected_pos).abs().max() > 1e-4 or (table_quat - expected_quat).abs().max() > 1e-3:
        raise ValueError(f"{path}: the table is not at the scene's pose {TABLE_POS} / {TABLE_QUAT} in every row.")
    for name in ("receptive_object", "ur5_metal_support"):
        _require_at_rest(objects[name], name, path)
    if _stack(robot["root_velocity"]).abs().max() > 1e-6:
        raise ValueError(f"{path}: the robot base moves in the file, but it is fixed in the scene.")

    robot_pos, robot_quat = _pose(robot)
    mount_pos, mount_quat = _pose(objects["ur5_metal_support"])
    receptive_pos, receptive_quat = _pose(objects["receptive_object"])
    insertive_pos, insertive_quat = _pose(objects["insertive_object"])
    velocity = _stack(objects["insertive_object"]["root_velocity"])
    lin_com, ang = velocity[:, :3], velocity[:, 3:]
    com_offset_w = transform_by_quat(
        torch.tensor(insertive_com, dtype=torch.float32).expand_as(lin_com), insertive_quat
    )
    lin_origin = lin_com - torch.cross(ang, com_offset_w, dim=-1)

    return ResetStateBuffer(
        {
            ROBOT: EntityStates(
                root_pos=robot_pos,
                root_quat=robot_quat,
                joints_name=ISAAC_JOINT_ORDER,
                joints_n_dofs=(1,) * len(ISAAC_JOINT_ORDER),
                dofs_pos=_stack(robot["joint_position"]),
                dofs_vel=_stack(robot["joint_velocity"]),
            ),
            MOUNT: EntityStates(root_pos=mount_pos, root_quat=mount_quat),
            RECEPTIVE: EntityStates(root_pos=receptive_pos, root_quat=receptive_quat),
            INSERTIVE: EntityStates(
                root_pos=insertive_pos, root_quat=insertive_quat, root_lin_vel=lin_origin, root_ang_vel=ang
            ),
        }
    )


def buffer_path(pair_name: str, families: Sequence[str] = RESET_FAMILIES) -> str:
    """The converted reset buffer of ``PAIRS[pair_name]``'s ``families``, converting (and downloading) on first use.

    Cached at :func:`buffer_cache_path`. Processes converting at once (DDP ranks) each write a complete file,
    atomically.
    """
    import os

    from eden.tasks.omnireset.constants import PAIRS

    path = buffer_cache_path(pair_name, families)
    if not os.path.exists(path):
        convert(PAIRS[pair_name], families).save(path)
    return path


def buffer_cache_path(pair_name: str, families: Sequence[str] = RESET_FAMILIES) -> str:
    """Where :func:`buffer_path` caches ``PAIRS[pair_name]``'s ``families``, whether or not it exists yet.

    Under Eden's cache dir: the buffer is derived data, rebuilt from UWLab's files rather than hosted. The file name
    carries everything the conversion's output depends on (the dataset revision, :data:`CONVERSION_VERSION` and the
    pair's conversion inputs), so a changed input converts afresh instead of reading a stale buffer.
    """
    import hashlib

    from eden.tasks.omnireset.constants import PAIRS
    from eden.utils.misc import get_cache_dir

    unknown = [family for family in families if family not in RESET_FAMILIES]
    if pair_name not in PAIRS or unknown or not families:
        raise ValueError(
            f"No OmniReset reset states for pair {pair_name!r} / families {list(families)}. "
            f"Pairs: {list(PAIRS)}; families: {list(RESET_FAMILIES)}."
        )
    pair = PAIRS[pair_name]
    inputs = hashlib.sha1(repr((pair.dataset_dir, pair.insertive_com)).encode()).hexdigest()[:8]
    name = f"{pair_name}-{'-'.join(families)}-v{CONVERSION_VERSION}-{inputs}.pt"
    return get_cache_dir("omnireset", DATASETS_REVISION[:12], name)


def convert(pair: Pair, families: Sequence[str] = RESET_FAMILIES) -> ResetStateBuffer:
    """Download ``pair``'s reset families at :data:`~.constants.DATASETS_REVISION` and merge them, one category each."""
    from huggingface_hub import hf_hub_download

    buffers = {}
    for family in families:
        path = hf_hub_download(UWLAB_REPO, pair.dataset_file(family), repo_type="dataset", revision=DATASETS_REVISION)
        buffers[family] = load_family(path, pair, pair.insertive_com)
    return ResetStateBuffer.from_families(buffers)
