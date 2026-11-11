"""``omnireset.datasets.load_family``: a UWLab reset file becomes a buffer in Eden's frames and joint names.

The files are written here in UWLab's layout (Isaac Lab ``InteractiveScene.get_state`` rows, lists of
per-row tensors, xyzw quaternions), so the oracle is the format itself, not a downloaded release.
"""

from __future__ import annotations

import pytest
import torch

from eden.tasks.omnireset.constants import INSERTIVE, ISAAC_JOINT_ORDER, MOUNT, PAIRS, RECEPTIVE, ROBOT, TABLE_POS
from eden.tasks.omnireset.datasets import load_family

PEG = PAIRS["peg"]
TABLE_XYZW = [0.0, 0.0, -0.70710678, 0.70710678]
# A quarter turn about z, xyzw then wxyz.
QUARTER_XYZW, QUARTER_WXYZ = [0.0, 0.0, 0.70710678, 0.70710678], [0.70710678, 0.0, 0.0, 0.70710678]


def _rows(*values):
    return [torch.tensor(v, dtype=torch.float32) for v in values]


def _write(
    path, *, insertive_velocity=(0.0,) * 6, table_pos=TABLE_POS, receptive_velocity=(0.0,) * 6, convention="xyzw"
):
    """A two-row UWLab file; row 1 turns the robot base and the insertive object a quarter about z."""
    joints = [[0.1 * i + 0.01 * row for i in range(12)] for row in range(2)]
    state = {
        "articulation": {
            "robot": {
                "root_pose": _rows([0.0, -0.039, 0.0, 0, 0, 0, 1], [0.01, -0.05, 0.005, *QUARTER_XYZW]),
                "root_velocity": _rows([0.0] * 6, [0.0] * 6),
                "joint_position": _rows(*joints),
                "joint_velocity": _rows(*[[-v for v in row] for row in joints]),
            }
        },
        "rigid_object": {
            "insertive_object": {
                "root_pose": _rows([0.4, -0.1, 0.002, 0, 0, 0, 1], [0.5, 0.2, 0.1, *QUARTER_XYZW]),
                "root_velocity": _rows([0.0] * 6, list(insertive_velocity)),
            },
            "receptive_object": {
                "root_pose": _rows([0.4, 0.1, 0.0022, 0, 0, 0, 1], [0.45, 0.0, 0.0022, 0, 0, 0, 1]),
                "root_velocity": _rows([0.0] * 6, list(receptive_velocity)),
            },
            "table": {
                "root_pose": _rows([*table_pos, *TABLE_XYZW], [*table_pos, *TABLE_XYZW]),
                "root_velocity": _rows([0.0] * 6, [0.0] * 6),
            },
            "ur5_metal_support": {
                "root_pose": _rows([0.0, -0.039, -0.0129, 0, 0, 0, 1], [0.01, -0.05, -0.0079, 0, 0, 0, 1]),
                "root_velocity": _rows([0.0] * 6, [0.0] * 6),
            },
        },
    }
    torch.save({"initial_state": state, "quat_convention": convention}, path)
    return path


def test_rows_arrive_wxyz_with_joints_named_in_isaac_order(tmp_path):
    buffer = load_family(str(_write(tmp_path / "f.pt")), PEG, insertive_com=(0.0, 0.0, 0.0))
    robot = buffer.entities[ROBOT]
    assert robot.joints_name == ISAAC_JOINT_ORDER and robot.joints_n_dofs == (1,) * 12
    torch.testing.assert_close(robot.dofs_pos[1], torch.tensor([0.1 * i + 0.01 for i in range(12)]))
    torch.testing.assert_close(robot.root_quat[1], torch.tensor(QUARTER_WXYZ))
    torch.testing.assert_close(buffer.entities[INSERTIVE].root_quat[1], torch.tensor(QUARTER_WXYZ))
    torch.testing.assert_close(buffer.entities[INSERTIVE].root_pos[1], torch.tensor([0.5, 0.2, 0.1]))


def test_fixed_bodies_carry_a_pose_and_no_velocity_and_the_table_is_dropped(tmp_path):
    buffer = load_family(str(_write(tmp_path / "f.pt")), PEG, insertive_com=(0.0, 0.0, 0.0))
    assert set(buffer.entities) == {ROBOT, MOUNT, RECEPTIVE, INSERTIVE}
    for name in (ROBOT, MOUNT, RECEPTIVE):
        assert buffer.entities[name].has_root and buffer.entities[name].root_lin_vel is None, name


def test_insertive_velocity_moves_from_the_centre_of_mass_to_the_authored_origin(tmp_path):
    # Spinning at 2 rad/s about z with the CoM 1 cm along the body x, which row 1 turns onto world y.
    path = _write(tmp_path / "f.pt", insertive_velocity=(0.0, 0.0, 0.0, 0.0, 0.0, 2.0))
    buffer = load_family(str(path), PEG, insertive_com=(0.01, 0.0, 0.0))
    # v_origin = v_com + w x (p_origin - p_com) = (0, 0, 2) x (0, -0.01, 0) = (0.02, 0, 0)
    torch.testing.assert_close(buffer.entities[INSERTIVE].root_lin_vel[1], torch.tensor([0.02, 0.0, 0.0]))
    torch.testing.assert_close(buffer.entities[INSERTIVE].root_ang_vel[1], torch.tensor([0.0, 0.0, 2.0]))


@pytest.mark.parametrize(
    "overrides, match",
    [
        ({"convention": "wxyz"}, r"quat_convention='wxyz', expected 'xyzw'.*isaaclab3"),
        ({"table_pos": (0.5, 0.0, -0.881)}, r"the table is not at the scene's pose"),
        ({"receptive_velocity": (0.1, 0.0, 0.0, 0.0, 0.0, 0.0)}, r"receptive_object is fixed in the scene but moves"),
    ],
    ids=["isaaclab2-quaternions", "table-elsewhere", "moving-receptive"],
)
def test_a_file_that_does_not_fit_the_scene_raises(tmp_path, overrides, match):
    with pytest.raises(ValueError, match=match):
        load_family(str(_write(tmp_path / "f.pt", **overrides)), PEG, insertive_com=(0.0, 0.0, 0.0))


def test_the_cached_buffer_is_reused_and_converted_afresh_when_a_conversion_input_changes(tmp_path, monkeypatch):
    import dataclasses

    import huggingface_hub

    from eden.tasks.omnireset import datasets

    downloads = []

    def hf_hub_download(repo_id, filename, **kwargs):  # the network boundary: serve a file in UWLab's layout
        downloads.append(filename)
        return str(_write(tmp_path / f"download{len(downloads)}.pt"))

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", hf_hub_download)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    families = ("ObjectAnywhereEEAnywhere",)

    first = datasets.buffer_path("peg", families)
    assert datasets.buffer_path("peg", families) == first and len(downloads) == 1, "a cached buffer is reused"

    monkeypatch.setitem(PAIRS, "peg", dataclasses.replace(PEG, insertive_com=(0.0, 0.0, 0.01)))
    moved_com = datasets.buffer_path("peg", families)
    assert moved_com != first and len(downloads) == 2, "a re-pinned centre of mass must not read the old buffer"

    monkeypatch.setattr(datasets, "CONVERSION_VERSION", datasets.CONVERSION_VERSION + 1)
    assert datasets.buffer_path("peg", families) not in (first, moved_com) and len(downloads) == 3
