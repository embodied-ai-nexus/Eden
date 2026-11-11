"""``ResetStateBuffer``: its schema checks, row operations and file format.

Capture from a live scene and the restore are tested with the event that consumes the buffer, in
``tests/managers/terms/events/test_state_buffer.py``.
"""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from eden.utils.buffers.reset_states import FORMAT, EntityStates, ResetStateBuffer


def _cube(n: int, offset: float = 0.0) -> EntityStates:
    pos = torch.arange(3 * n, dtype=torch.float32).reshape(n, 3) + offset
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(n, 1)
    return EntityStates(root_pos=pos, root_quat=quat)


def _arm(n: int, offset: float = 0.0) -> EntityStates:
    return EntityStates(
        joints_name=("shoulder", "wrist"),
        joints_n_dofs=(1, 3),
        dofs_pos=torch.full((n, 4), offset),
        dofs_vel=torch.zeros(n, 4),
    )


def _buffer(n: int = 3, offset: float = 0.0) -> ResetStateBuffer:
    return ResetStateBuffer(
        {"cube": _cube(n, offset), "arm": _arm(n, offset)}, columns={"goal": torch.full((n, 3), offset + 0.5)}
    )


def test_num_states_is_the_shared_row_count():
    assert _buffer(5).num_states == 5


@pytest.mark.parametrize(
    "states, match",
    [
        (EntityStates(root_pos=torch.zeros(2, 3)), "only one of root_pos / root_quat"),
        (
            EntityStates(
                root_lin_vel=torch.zeros(2, 3), joints_name=("j",), joints_n_dofs=(1,), dofs_pos=torch.zeros(2, 1)
            ),
            "root velocities but no root pose",
        ),
        (EntityStates(joints_name=("j",), joints_n_dofs=(1,)), r"names joints \['j'\] but has no dofs_pos"),
        (EntityStates(joints_name=("j", "j"), joints_n_dofs=(1, 1), dofs_pos=torch.zeros(2, 2)), "more than once"),
        (
            EntityStates(joints_name=("j",), joints_n_dofs=(2,), dofs_pos=torch.zeros(2, 1)),
            "dofs_pos has width 1, expected 2",
        ),
        (EntityStates(root_pos=torch.zeros(2, 3), root_quat=torch.zeros(2, 3)), "root_quat has width 3, expected 4"),
        (EntityStates(root_pos=torch.zeros(2, 3), root_quat=torch.zeros(3, 4)), "root_quat has 3 rows"),
        (EntityStates(), "holds no column"),
        (
            EntityStates(
                root_pos=torch.tensor([[0.0, 0.0, 0.0], [0.0, float("nan"), 0.0]]), root_quat=torch.eye(4)[:2]
            ),
            r"root_pos holds NaN or inf in 1 row\(s\), first \[1\]",
        ),
        (
            EntityStates(
                root_pos=torch.zeros(2, 3), root_quat=torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 0.0, 0.0]])
            ),
            r"root_quat is not unit-norm in 1 row\(s\), first \[1\] \(norms \[1.4142\]\).*root_quat.norm",
        ),
        (
            EntityStates(root_pos=torch.tensor([[0, 0, 1], [0, 0, 1]]), root_quat=torch.eye(4)[:2]),
            r"root_pos has dtype torch.int64; entity columns hold floats.*torch.tensor\(\[\[0.0, 0.0, 1.0\]\]\)",
        ),
    ],
    ids=[
        "half-pose",
        "vel-without-pose",
        "joints-without-dofs",
        "repeated-joint",
        "dof-width",
        "quat-width",
        "rows",
        "empty",
        "non-finite",
        "non-unit-quat",
        "integer-column",
    ],
)
def test_malformed_entity_states_raise_naming_the_entity(states, match):
    with pytest.raises(ValueError, match=f"entity 'thing'.*{match}|{match}"):
        ResetStateBuffer({"thing": states})


def test_columns_that_disagree_on_rows_are_listed():
    with pytest.raises(ValueError, match=r"entity 'cube': 3, column 'goal': 2"):
        ResetStateBuffer({"cube": _cube(3)}, columns={"goal": torch.zeros(2, 3)})


def test_category_out_of_range_names_the_families():
    with pytest.raises(ValueError, match=r"span \[0, 2\].*2 category_names \['reach', 'grasp'\]"):
        ResetStateBuffer({"cube": _cube(3)}, category=torch.tensor([0, 1, 2]), category_names=("reach", "grasp"))


def test_select_gathers_every_column_at_the_rows():
    picked = _buffer(4).select([3, 1])
    torch.testing.assert_close(picked.entities["cube"].root_pos, _buffer(4).entities["cube"].root_pos[[3, 1]])
    assert picked.num_states == 2


def test_select_reads_a_bool_mask_as_the_rows_it_marks():
    picked = _buffer(4).select(torch.tensor([False, True, False, True]))
    torch.testing.assert_close(picked.entities["cube"].root_pos, _buffer(4).entities["cube"].root_pos[[1, 3]])


@pytest.mark.parametrize(
    "rows, match",
    [
        (torch.tensor([True, False, True]), r"a bool mask of shape \(3,\) for a buffer of 4 rows"),
        (torch.tensor([0.0, 2.0]), r"row indices must be integers or a bool mask, got dtype torch.float32"),
    ],
    ids=["mask-of-another-length", "float-indices"],
)
@pytest.mark.parametrize("operation", ["select", "write_rows"])
def test_row_selectors_refuse_what_would_pick_other_rows(operation, rows, match):
    buffer = _buffer(4)
    with pytest.raises(ValueError, match=rf"ResetStateBuffer.{operation}: {match}"):
        if operation == "select":
            buffer.select(rows)
        else:
            buffer.write_rows(rows, _buffer(2, offset=100.0))


def test_write_rows_reads_a_bool_mask_as_the_slots_it_marks_in_order():
    """Read as indices, the mask [True, False, True, False] would write slots 1, 0, 1, 0."""
    ring = _buffer(4)
    ring.write_rows(torch.tensor([True, False, True, False]), _buffer(2, offset=100.0))
    torch.testing.assert_close(ring.entities["arm"].dofs_pos[:, 0], torch.tensor([100.0, 0.0, 100.0, 0.0]))


def test_write_rows_bumps_the_generation():
    ring = _buffer(4)
    ring.write_rows(torch.tensor([1]), _buffer(1))
    assert ring.generation == 1


def test_joint_names_given_as_lists_share_rows_with_captured_ones():
    """``save`` writes lists and capture builds tuples; either way the layout is the same."""
    listed = replace(_arm(2), joints_name=["shoulder", "wrist"], joints_n_dofs=[1, 3])
    assert ResetStateBuffer({"arm": listed}).layout() == ResetStateBuffer({"arm": _arm(2)}).layout()


def test_write_rows_overwrites_only_the_named_slots():
    ring = _buffer(4, offset=0.0)
    ring.write_rows(torch.tensor([2, 0]), _buffer(2, offset=100.0))
    dofs = ring.entities["arm"].dofs_pos[:, 0]
    torch.testing.assert_close(dofs, torch.tensor([100.0, 0.0, 100.0, 0.0]))
    torch.testing.assert_close(ring.columns["goal"][:, 0], torch.tensor([100.5, 0.5, 100.5, 0.5]))


@pytest.mark.parametrize(
    "rows_idx, match",
    [
        ([1, 4], r"Out of range: \[4\]; repeated: none"),
        ([-1, 0], r"Out of range: \[-1\]; repeated: none"),
        ([2, 2], r"Out of range: none; repeated: \[2\]"),
    ],
    ids=["past-the-end", "negative", "repeated"],
)
def test_write_rows_refuses_slots_outside_the_buffer_or_written_twice(rows_idx, match):
    """Unchecked, CUDA turns an out-of-range slot into a device-side assert that names no buffer."""
    ring = _buffer(4)
    with pytest.raises(ValueError, match=rf"distinct and in \[0, 4\)\. {match}.*% buffer.num_states"):
        ring.write_rows(torch.tensor(rows_idx), _buffer(2, offset=100.0))
    torch.testing.assert_close(ring.entities["arm"].dofs_pos, _buffer(4).entities["arm"].dofs_pos)


def test_write_rows_refuses_rows_of_another_layout_listing_each_difference():
    moving_cube = ResetStateBuffer(
        {"cube": replace(_cube(1), root_lin_vel=torch.zeros(1, 3))}, columns={"goal": torch.zeros(1, 3)}
    )
    with pytest.raises(ValueError) as raised:
        _buffer(4).write_rows(torch.tensor([0]), moving_cube)
    message = str(raised.value)
    assert "entity 'arm': only in the first" in message
    assert (
        "entity 'cube': columns [root_pos, root_quat], joints [none] vs columns [root_pos, root_quat, root_lin_vel]"
        in message
    )
    assert "column 'goal'" not in message  # what agrees is not listed


def test_write_rows_casts_a_float_column_harvested_at_another_precision():
    """A ``precision='64'`` env's buffer takes rows captured, or saved, at 32 bits, as its entity columns do."""
    ring = _buffer(2).to("cpu", dtype=torch.float64)
    ring.write_rows(torch.tensor([1]), _buffer(1, offset=100.0))
    assert ring.columns["goal"].dtype == torch.float64
    torch.testing.assert_close(ring.columns["goal"][:, 0], torch.tensor([0.5, 100.5], dtype=torch.float64))


def test_write_rows_refuses_a_float_column_into_an_integer_one():
    """Casting would truncate a float column into an integer one."""
    cells = ResetStateBuffer({"cube": _cube(2)}, columns={"cell": torch.zeros(2, dtype=torch.long)})
    with pytest.raises(ValueError, match=r"column 'cell': \(\(\), 'int64'\) vs \(\(\), 'float'\)"):
        cells.write_rows(torch.tensor([0]), ResetStateBuffer({"cube": _cube(1)}, columns={"cell": torch.zeros(1)}))


def test_entity_order_does_not_change_the_layout():
    forward = ResetStateBuffer({"cube": _cube(2), "arm": _arm(2)})
    backward = ResetStateBuffer({"arm": _arm(2), "cube": _cube(2)})
    assert forward.layout_differences(backward) == []


def test_from_families_labels_each_row_with_its_family():
    combined = ResetStateBuffer.from_families({"reaching": _buffer(2), "near_goal": _buffer(3, offset=10.0)})
    assert combined.category_names == ("reaching", "near_goal")
    assert combined.category.tolist() == [0, 0, 1, 1, 1]
    assert combined.entities["arm"].dofs_pos[:, 0].tolist() == [0.0, 0.0, 10.0, 10.0, 10.0]


def test_concat_refuses_buffers_with_different_joints():
    other = ResetStateBuffer(
        {"cube": _cube(2), "arm": EntityStates(joints_name=("elbow",), joints_n_dofs=(1,), dofs_pos=torch.zeros(2, 1))},
        columns={"goal": torch.zeros(2, 3)},
    )
    with pytest.raises(ValueError, match="buffer 1 has a different layout"):
        ResetStateBuffer.concat([_buffer(2), other])


def test_save_then_load_round_trips_every_column(tmp_path):
    original = ResetStateBuffer.from_families({"a": _buffer(2), "b": _buffer(1, offset=7.0)})
    path = tmp_path / "nested" / "states.pt"
    original.save(path)
    loaded = ResetStateBuffer.load(path)
    assert loaded.layout() == original.layout()
    for name, states in original.entities.items():
        for field_name, value in states.tensors().items():
            torch.testing.assert_close(getattr(loaded.entities[name], field_name), value)
    torch.testing.assert_close(loaded.columns["goal"], original.columns["goal"])
    assert loaded.category.tolist() == original.category.tolist()
    assert [p.name for p in path.parent.iterdir()] == ["states.pt"]  # the temporary file was moved, not left


def test_load_refuses_a_missing_file_naming_how_to_write_one(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"ResetStateBuffer.save\(path\)"):
        ResetStateBuffer.load(tmp_path / "absent.pt")


def test_load_refuses_a_raw_state_dict(tmp_path):
    path = tmp_path / "raw.pt"
    torch.save({"entities": {"cube": {"pos": torch.zeros(1, 3)}}}, path)
    with pytest.raises(ValueError, match=f"format None, expected '{FORMAT}'"):
        ResetStateBuffer.load(path)


def test_load_refuses_another_format_version(tmp_path):
    path = tmp_path / "future.pt"
    _buffer(1).save(path)
    payload = torch.load(path, weights_only=True)
    payload["version"] = 99
    torch.save(payload, path)
    with pytest.raises(ValueError, match="format version 99"):
        ResetStateBuffer.load(path)


def test_to_casts_float_columns_and_keeps_the_category_integer():
    moved = ResetStateBuffer.from_families({"a": _buffer(2)}).to("cpu", dtype=torch.float64)
    assert moved.entities["cube"].root_pos.dtype == torch.float64
    assert moved.category.dtype == torch.int64
