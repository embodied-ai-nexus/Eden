"""Tests for ``eden.managers.motion_manager``: ``MotionTerm``, ``MotionManager`` and ``sample_start_times``.

Every manager here runs a real synthetic clip library (``tests.support.motion``) against
one built mimic env; clip ``c`` frame ``f`` holds ``robot_dofs_at(c, f)``, so a test states exactly
which frame each env's reference must show.
"""

from __future__ import annotations

import json
import re

import numpy as np
import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from eden.datasets.base import DATASET_REGISTRY
from eden.managers.motion_manager import MotionTerm, sample_start_times
from eden.utils.buffers.cbt import ConcatBatchTensor
from eden.utils.motion_reference.motion_data import BatchMotionSequence, MotionReferenceData, MotionReferenceState
from tests.support.motion import (
    FRAMERATE,
    ROBOT_DOFS,
    ROBOT_LINKS,
    SyntheticMotionDataset,
    build_mimic_env,
    load_dataset,
    make_manager,
    pin_clips,
    robot_base_pos_at,
    robot_dofs_at,
    term_options,
    write_clip,
    write_library,
)

# Clips 0/1/2 of the module library: 20/30/40 frames = 0.4/0.6/0.8 s at 50 Hz.
LENGTHS = [20, 30, 40]
DT = 1.0 / FRAMERATE


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    return write_library(tmp_path_factory.mktemp("library"), LENGTHS)


@pytest.fixture(scope="module")
def env(library):
    return build_mimic_env(library)


def _dofs(manager, frame: int = 0) -> torch.Tensor:
    return manager.data["robot"].dofs_pos[:, frame].cpu()


def _expected_dofs(clips, frames) -> torch.Tensor:
    return torch.stack([robot_dofs_at(c, f) for c, f in zip(clips, frames)])


def _expected_base_pos(clips, frames) -> torch.Tensor:
    return torch.stack([robot_base_pos_at(c, f) for c, f in zip(clips, frames)])


def _clip1_manager(env, library, **term_kwargs):
    """A manager whose resets always land on clip 1, so a reset does not change which clip an env plays."""
    return make_manager(env, terms={"default": term_options(library, motion_weights=[0.0, 1.0, 0.0], **term_kwargs)})


# Half a frame into a frame, so float32 time sums never land on a frame boundary.
HALF = 0.5 * DT


# ---------------------------------------------------------------------------
# MotionTerm: env-id mapping, clip lengths, trajectory subsets
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "selector, expected",
    [
        (None, [0, 1, 2, 3]),
        (slice(4, 8), [0, 1, 2, 3]),
        (torch.tensor([5, 7]), [1, 3]),
        ([4, 6], [0, 2]),
    ],
)
def test_int_selectors_map_into_the_terms_env_slice(env, library, selector, expected):
    term = make_manager(env, library).get_term("default")
    term.set_envs_idx_assignments(slice(4, 8))

    assert term.envs_idx_to_assigned_ids(selector).tolist() == expected


def test_bool_selectors_stay_masks_sliced_to_the_terms_envs(env, library):
    term = make_manager(env, library).get_term("default")
    term.set_envs_idx_assignments(slice(4, 8))
    global_mask = torch.zeros(10, dtype=torch.bool)
    global_mask[[4, 6, 9]] = True  # env 9 lies outside the term and drops

    assert term.envs_idx_to_assigned_ids(global_mask).tolist() == [True, False, True, False]
    assert term.envs_idx_is_assigned(torch.tensor([0, 4, 7, 8])).tolist() == [False, True, True, False]


def test_motion_lengths_follow_the_env_clip_and_start(env, library):
    term = make_manager(env, library).get_term("default")
    pin_clips(term, [0, 1, 2, 0], start_s=[0.0, 0.1, 0.2, 0.3])

    torch.testing.assert_close(term.complete_motion_lengths.cpu(), torch.tensor([0.4, 0.6, 0.8, 0.4]))
    torch.testing.assert_close(term.assigned_motion_lengths.cpu(), torch.tensor([0.4, 0.5, 0.6, 0.1]))


@pytest.fixture(scope="module")
def ten_clip_library(tmp_path_factory):
    return write_library(tmp_path_factory.mktemp("ten_clips"), [5] * 10)


@pytest.mark.parametrize(
    "second, expected_globals",
    [(torch.tensor([0, 2]), [1, 5]), (slice(1, 3), [3, 5])],
    ids=["ids", "slice"],
)
def test_enable_trajectories_composes_local_ids_and_keeps_weights_global(
    env, ten_clip_library, second, expected_globals
):
    manager = make_manager(env, ten_clip_library)
    term = manager.get_term("default")

    term.enable_trajectories(torch.tensor([1, 3, 5, 7, 9]))
    term.enable_trajectories(second)  # local ids into the 5 kept above
    term.set_envs_idx_assignments(slice(0, 4))
    pin_clips(term, [0, 1, 0, 1])
    manager.fill_data(torch.arange(4))

    assert term.num_trajectories == 2
    # `_local_to_global` / `_motion_weights` are the sharding contract itself: weights stay sized to the
    # global motion-id space so a DDP all_reduce matches shapes across ranks.
    assert term._local_to_global.tolist() == expected_globals
    assert term._motion_weights.shape == (10,)
    # Local id i plays global clip expected_globals[i].
    g0, g1 = expected_globals
    torch.testing.assert_close(_dofs(manager), _expected_dofs([g0, g1, g0, g1], [0] * 4))


def test_a_full_length_permuted_target_order_reorders_the_reference(env, library):
    """A target list naming every DOF in another order reorders the columns, not just the recorded names."""
    reversed_dofs = ROBOT_DOFS[::-1]
    manager = make_manager(env, terms={"default": term_options(library, dofs={"robot": reversed_dofs})})
    pin_clips(manager.get_term("default"), [0, 0, 0, 0], start_s=0.5 * DT)

    manager.fill_data(torch.arange(4))

    assert manager.get_target_dofs_name("robot") == reversed_dofs
    # Column k must hold the joint the manager says it holds.
    expected = robot_dofs_at(0, 0)[[ROBOT_DOFS.index(name) for name in reversed_dofs]]
    torch.testing.assert_close(_dofs(manager)[0], expected)


def test_a_full_length_permuted_target_links_order_reorders_the_reference(env, library):
    """The link twin of the test above: link ``l`` sits ``0.1 l`` to the side of the base in the dataset."""
    reversed_links = ROBOT_LINKS[::-1]
    manager = make_manager(env, terms={"default": term_options(library, links={"robot": reversed_links})})
    term = manager.get_term("default")
    pin_clips(term, [0, 0, 0, 0], start_s=0.5 * DT)

    manager.fill_data(torch.arange(4))

    assert term.target_links_name["robot"] == reversed_links
    # Row k must hold the link the term says it holds.
    data = manager.data["robot"]
    side = (data.links_pos_w[0, 0, :, 1] - data.base_pos_w[0, 0, 1]).cpu()
    expected = 0.1 * torch.tensor([float(ROBOT_LINKS.index(name)) for name in reversed_links])
    torch.testing.assert_close(side, expected)


def test_a_permuted_target_order_keeps_the_datasets_names_on_their_columns(env, library):
    """The term's dataset relabels its own names with the reorder, so its name-keyed helpers stay right."""
    reversed_dofs, reversed_links = ROBOT_DOFS[::-1], ROBOT_LINKS[::-1]
    options = term_options(library, dofs={"robot": reversed_dofs}, links={"robot": reversed_links})
    data_source = make_manager(env, terms={"default": options}).get_term("default").data_source

    assert data_source.motion_dofs_name["robot"] == reversed_dofs
    assert data_source.motion_links_name["robot"] == reversed_links
    assert data_source.compute_dofs_pos_range("robot") == load_dataset(library).compute_dofs_pos_range("robot")


# ---------------------------------------------------------------------------
# MotionTerm: sampling a clip and a start phase on reset
# ---------------------------------------------------------------------------


def test_reset_resamples_only_the_given_envs(env, library):
    manager = make_manager(env, terms={"default": term_options(library, motion_weights=[0.0, 0.0, 1.0])})
    term = manager.get_term("default")
    pin_clips(term, [0, 0, 0, 0])

    manager.reset(torch.tensor([], dtype=torch.long))
    manager.fill_data(torch.arange(4))
    torch.testing.assert_close(_dofs(manager), _expected_dofs([0, 0, 0, 0], [0] * 4))

    manager.reset(torch.tensor([0, 1, 2]))  # weight only on clip 2
    manager.fill_data(torch.arange(4))
    torch.testing.assert_close(_dofs(manager), _expected_dofs([2, 2, 2, 0], [0] * 4))


def test_middle_range_start_is_that_fraction_of_the_clip(env, library):
    manager = make_manager(
        env, library, terms={"default": term_options(library, motion_start_from_middle_range=(0.5, 0.5))}
    )
    term = manager.get_term("default")
    pin_clips(term, [0, 1, 2, 0])

    manager.reset(torch.arange(4))
    torch.testing.assert_close(term.assigned_motion_lengths.cpu(), 0.5 * term.complete_motion_lengths.cpu())


@pytest.mark.parametrize("prob, all_zero", [(1.0, True), (0.0, False)])
def test_start_at_timestep_zero_prob_overrides_the_sampled_start(env, library, prob, all_zero):
    torch.manual_seed(0)
    manager = make_manager(
        env,
        library,
        terms={
            "default": term_options(
                library, motion_start_from_middle_range=(0.5, 1.0), start_at_timestep_zero_prob=prob
            )
        },
    )
    term = manager.get_term("default")

    manager.reset(torch.arange(4))

    started_at_zero = term.assigned_motion_lengths.cpu() == term.complete_motion_lengths.cpu()
    assert bool(started_at_zero.all()) is all_zero
    assert bool(started_at_zero.any()) is all_zero


def test_concat_motion_bins_starts_stay_inside_the_clip(env, tmp_path):
    # A bin longer than every clip is the case that once produced zero bins per motion.
    root = write_library(tmp_path, [7, 12, 30])
    for bin_length_s in (0.1, 9999.0):
        manager = make_manager(
            env,
            terms={
                "default": term_options(
                    root, env_starting_stub_sampling_strategy="concat_motion_bins", motion_bin_length_s=bin_length_s
                )
            },
        )
        term = manager.get_term("default")
        for _ in range(50):
            manager.reset(torch.arange(4))
            start = term.complete_motion_lengths - term.assigned_motion_lengths
            assert (start >= 0).all() and (start <= term.complete_motion_lengths - DT + 1e-6).all()


def test_concat_motion_bins_without_a_bin_length_fails_at_construction(env, library):
    with pytest.raises(ValueError, match="concat_motion_bins.*motion_bin_length_s"):
        make_manager(
            env, terms={"default": term_options(library, env_starting_stub_sampling_strategy="concat_motion_bins")}
        )


def test_current_motion_weights_are_each_envs_clip_weight(env, library):
    manager = make_manager(env, terms={"default": term_options(library, motion_weights=[1.0, 2.0, 3.0])})
    term = manager.get_term("default")
    pin_clips(term, [2, 0, 1, 2])

    assert term.get_current_motion_weights(torch.tensor([0, 1])).tolist() == [3.0, 1.0]
    assert term.get_current_motion_weights(None).tolist() == [3.0, 1.0, 2.0, 3.0]


def test_motion_weights_load_from_a_file(env, library, tmp_path):
    path = tmp_path / "weights.json"
    path.write_text(json.dumps({"motion_weights": [1.0, 2.0, 3.0]}))
    term = make_manager(env, terms={"default": term_options(library, motion_weights=str(path))}).get_term("default")
    pin_clips(term, [0, 1, 2, 0])

    assert term.get_current_motion_weights(None).tolist() == [1.0, 2.0, 3.0, 1.0]


def test_motion_weights_of_the_wrong_length_fail_at_construction(env, library):
    with pytest.raises(ValueError, match="has length 2, but the dataset has 3 trajectories"):
        make_manager(env, terms={"default": term_options(library, motion_weights=[1.0, 2.0])})


def test_the_retired_clip_joint_ref_flag_fails_at_construction(env, library):
    """Nothing ever read it; the reset clamps to the soft limits on its own and the reference is never
    clipped, so a config that sets it must hear so instead of believing it is on.
    """
    with pytest.raises(ValueError, match=r"invalid: \[.clip_joint_ref_to_robot_limits.\]"):
        make_manager(env, terms={"default": term_options(library, clip_joint_ref_to_robot_limits=True)})


# ---------------------------------------------------------------------------
# Filling the reference buffers
# ---------------------------------------------------------------------------


def test_fill_data_reads_each_envs_clip_at_its_frames(env, library):
    manager = make_manager(env, library, num_frames=3)
    pin_clips(manager.get_term("default"), [0, 1, 2, 0], start_s=[HALF, 0.1 + HALF, 0.2 + HALF, 0.3 + HALF])

    manager.fill_data(torch.arange(4))

    ref = manager.data["robot"]
    starts = [0, 5, 10, 15]
    for k in range(3):
        torch.testing.assert_close(_dofs(manager, k), _expected_dofs([0, 1, 2, 0], [s + k for s in starts]))
    torch.testing.assert_close(ref.base_pos_w[3, 2].cpu(), robot_base_pos_at(0, 17))
    assert ref.validity.all()
    # The object is filled from its own sequence (offset 1 m in x), not the robot's.
    torch.testing.assert_close(manager.data["object"].base_pos_w[1, 0].cpu(), torch.tensor([1.05, 0.0, 0.81]))


def test_buffers_are_sized_per_entity(env, library):
    manager = make_manager(env, library, num_frames=3)

    assert manager.data["robot"].dofs_pos.shape == (4, 3, len(ROBOT_DOFS))
    assert manager.data["robot"].links_pos_w.shape == (4, 3, 4, 3)
    assert manager.reference_frame["robot"].dofs_pos.shape == (4, 1, len(ROBOT_DOFS))
    assert manager.get_init_reference_state()["robot"].dofs_pos.shape == (4, len(ROBOT_DOFS))
    assert manager.data["object"].dofs_pos.shape == (4, 3, 0)


def test_frames_past_the_clip_end_hold_the_last_pose_at_rest(env, library):
    manager = make_manager(env, library, num_frames=3)
    pin_clips(manager.get_term("default"), [0, 0, 0, 0], start_s=18.5 * DT)  # frames 18, 19, 20 of 20

    manager.fill_data(torch.arange(4))

    ref = manager.data["robot"]
    assert ref.validity[0].tolist() == [True, True, False]
    torch.testing.assert_close(ref.dofs_pos[0, 2].cpu(), robot_dofs_at(0, 19))
    # The finite-differenced velocity is non-zero on the last valid frame and zeroed past it.
    torch.testing.assert_close(ref.dofs_vel[0, 1].cpu(), torch.full((len(ROBOT_DOFS),), 0.05))
    torch.testing.assert_close(ref.base_lin_vel_w[0, 1].cpu(), torch.tensor([0.5, 0.0, 0.0]))
    for field in ("dofs_vel", "base_lin_vel_w", "base_ang_vel_w", "links_lin_vel_w", "links_ang_vel_w"):
        assert not getattr(ref, field)[:, 2].any(), field
    assert (ref.time_to_target_frame[:, 2] == -1.0).all()


def test_one_frame_interval_starts_the_window_one_frame_ahead(env, library):
    manager = make_manager(env, library, data_start_from="one_frame_interval")
    pin_clips(manager.get_term("default"), [1, 1, 1, 1], start_s=HALF)

    manager.fill_data(torch.arange(4))

    torch.testing.assert_close(_dofs(manager), _expected_dofs([1] * 4, [1] * 4))


def test_frame_interval_is_resampled_within_its_range_on_reset(env, library):
    constant = make_manager(env, library, num_frames=3)
    constant.reset()
    constant.fill_data(torch.arange(4))
    torch.testing.assert_close(constant.data["robot"].time_to_target_frame[0].cpu(), torch.tensor([0.0, DT, 2 * DT]))

    ranged = make_manager(env, library, num_frames=3, frame_interval_s_range=(0.04, 0.08))
    ranged.reset()
    ranged.fill_data(torch.arange(4))
    t2t = ranged.data["robot"].time_to_target_frame.cpu()
    assert ((t2t[:, 1] >= 0.04) & (t2t[:, 1] <= 0.08)).all()
    torch.testing.assert_close(t2t[:, 2], 2 * t2t[:, 1])


def test_compute_origins_are_added_once_per_fill(env, tmp_path):
    """Contact-less clips never rewrite the contact buffer, so the origin must not pile up on it."""
    contact = {
        "robot.contact_points_w": torch.zeros(10, 4, 1, 3).numpy(),
        "robot.contact_points_valid": torch.ones(10, 4, 1, dtype=torch.bool).numpy(),
    }
    with_contact = tmp_path / "contact"
    with_contact.mkdir()
    write_clip(with_contact / "clip_00.npz", 10, overrides=contact)
    without_contact = write_library(tmp_path / "plain", [10])
    origins = torch.tensor([[10.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])

    for root, has_contact in ((with_contact, True), (without_contact, False)):
        term = make_manager(env, root).get_term("default")
        pin_clips(term, [0, 0, 0, 0])
        buffer = MotionReferenceData.make_empty(4, 2, len(ROBOT_DOFS), 4)
        for _ in range(3):
            term.compute("robot", torch.arange(4), torch.zeros(4, 2), origins, buffer)

        torch.testing.assert_close(buffer.base_pos_w[:, 0].cpu(), robot_base_pos_at(0, 0) + origins)
        torch.testing.assert_close(buffer.links_pos_w[:, 0, 0].cpu(), robot_base_pos_at(0, 0) + origins)
        expected_points = origins if has_contact else torch.zeros(4, 3)
        torch.testing.assert_close(buffer.contact_points_w[:, 0, 0, 0].cpu(), expected_points)


# A distinct origin per env, so an origin shifted onto the wrong env's rows shows.
ORIGINS = torch.tensor([[10.0, 0.0, 0.0], [0.0, 20.0, 0.0], [0.0, 0.0, 30.0], [1.0, 2.0, 3.0]])


def _buffer_tensors(manager) -> dict[str, torch.Tensor]:
    return {
        f"{entity}.{name}": value
        for entity, data in manager.data.items()
        for name, value in vars(data).items()
        if isinstance(value, torch.Tensor)
    }


def _buffers(manager) -> dict[str, torch.Tensor]:
    return {name: value.clone() for name, value in _buffer_tensors(manager).items()}


def _sentinel(value: torch.Tensor):
    return False if value.dtype == torch.bool else 7


def _sentinel_buffers(manager) -> None:
    """Overwrite every buffer field of every entity with a value no fill writes."""
    for value in _buffer_tensors(manager).values():
        value.fill_(_sentinel(value))


def _pin_every_term(manager, start_s) -> None:
    """Put every env on clip 0 of its term (5 frames, 0.1 s) at ``start_s[env]``."""
    for term in manager.terms.values():
        pin_clips(term, 0, start_s=start_s[term.assigned_env_slice])


# Env 2's last frame and all of env 3's lie past the end of the clip, so velocities are zeroed and times
# set to -1 there; envs 0 and 1 stay within it.
STARTS_PAST_THE_END = torch.tensor([0.0, 0.03, 0.07, 0.2])
TERM_SPLITS = pytest.mark.parametrize("term_counts", [[1], [1, 3]], ids=["one term", "terms split across the envs"])


class _BufferWrites(TorchDispatchMode):
    """Record the op of every in-place or ``out=`` write into the storage of one of ``manager``'s buffers."""

    def __init__(self, manager) -> None:
        super().__init__()
        # An empty buffer has no storage to tell apart from any other empty tensor's.
        self._storages = {
            value.untyped_storage().data_ptr() for value in _buffer_tensors(manager).values() if value.numel()
        }
        self.ops: list[str] = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        name = func._schema.name.removeprefix("aten::")
        target = None
        if isinstance(kwargs.get("out"), torch.Tensor):
            target, name = kwargs["out"], f"{name}(out=)"
        elif name.endswith("_") and args and isinstance(args[0], torch.Tensor):
            target = args[0]
        if target is not None and target.numel() and target.untyped_storage().data_ptr() in self._storages:
            self.ops.append(name)
        return func(*args, **kwargs)


@TERM_SPLITS
def test_the_fill_of_every_env_writes_what_a_fill_through_their_ids_writes(env, tmp_path, term_counts, monkeypatch):
    """The fill of every env hands each term its env slice and gathers into the buffer in place."""
    manager = make_manager(env, terms=_libraries(tmp_path, term_counts), num_frames=3)
    _pin_every_term(manager, STARTS_PAST_THE_END)
    monkeypatch.setattr(env, "env_origins", ORIGINS, raising=False)

    _sentinel_buffers(manager)
    manager.fill_data()
    every_env = _buffers(manager)
    _sentinel_buffers(manager)
    manager.fill_data(torch.arange(4))
    through_ids = _buffers(manager)

    assert every_env.keys() == through_ids.keys()
    for name, value in every_env.items():
        torch.testing.assert_close(value, through_ids[name], rtol=0, atol=0, msg=name)
    validity = manager.data["robot"].validity
    assert validity.any() and not validity.all()


@TERM_SPLITS
@pytest.mark.parametrize("envs_idx", [None, torch.arange(4)], ids=["every env", "through ids"])
def test_a_fill_moves_each_envs_world_positions_by_its_own_origin(env, tmp_path, term_counts, envs_idx, monkeypatch):
    manager = make_manager(env, terms=_libraries(tmp_path, term_counts), num_frames=3)
    _pin_every_term(manager, STARTS_PAST_THE_END)
    manager.fill_data(envs_idx)
    at_the_world_origin = _buffers(manager)

    monkeypatch.setattr(env, "env_origins", ORIGINS, raising=False)
    manager.fill_data(envs_idx)

    shifted = _buffers(manager)
    positions = {f"{entity}.{field}" for entity in manager.data for field in ("base_pos_w", "links_pos_w")}
    for name in positions:
        moved = shifted[name] - at_the_world_origin[name]
        torch.testing.assert_close(moved, ORIGINS.view(4, *[1] * (moved.ndim - 2), 3).expand_as(moved), msg=name)
    # Every other field ignores the origin.
    for name in at_the_world_origin.keys() - positions:
        torch.testing.assert_close(shifted[name], at_the_world_origin[name], rtol=0, atol=0, msg=name)


@TERM_SPLITS
def test_a_fill_of_some_envs_leaves_the_other_envs_rows_alone(env, tmp_path, term_counts):
    manager = make_manager(env, terms=_libraries(tmp_path, term_counts), num_frames=3)
    _pin_every_term(manager, STARTS_PAST_THE_END)
    _sentinel_buffers(manager)

    # Envs 0 and 2 lie in different terms when the terms split the envs.
    manager.fill_data(torch.tensor([0, 2]))

    for name, value in _buffers(manager).items():
        assert (value[[1, 3]] == _sentinel(value)).all(), name
    robot = manager.data["robot"]
    assert robot.validity[[0, 2], 0].all() and not robot.validity[2, 2]
    assert robot.time_to_target_frame[2, 2] == -1.0


def test_a_fill_of_some_envs_writes_the_buffers_only_through_their_ids(env, library):
    """A reset's fill finishes the gathered rows, not every env's: no pass over a whole buffer per reset."""
    manager = make_manager(env, library, num_frames=3)
    with _BufferWrites(manager) as recorded:
        manager.fill_data(torch.tensor([1, 3]))

    assert set(recorded.ops) == {"index_put_"}


def test_the_fill_of_every_env_gathers_into_the_buffers(env, library):
    """No gathered copy is written back: the fill of every env makes no ``index_put_``."""
    manager = make_manager(env, library, num_frames=3)
    with _BufferWrites(manager) as recorded:
        manager.fill_data()

    assert "index_select(out=)" in recorded.ops
    assert "index_put_" not in recorded.ops


def test_init_reference_state_is_the_start_frame_shifted_by_the_origin(env, library):
    manager = make_manager(env, library)
    term = manager.get_term("default")
    pin_clips(term, [0, 1, 2, 0], start_s=[HALF, 0.1 + HALF, 0.2 + HALF, 0.3 + HALF])

    full = manager.get_init_reference_state()["robot"]
    subset = manager.get_init_reference_state(torch.tensor([1, 3]))["robot"]

    torch.testing.assert_close(full.dofs_pos.cpu(), _expected_dofs([0, 1, 2, 0], [0, 5, 10, 15]))
    torch.testing.assert_close(subset.dofs_pos.cpu(), _expected_dofs([1, 0], [5, 15]))

    state = MotionReferenceState.make_empty(4, len(ROBOT_DOFS))
    term.fill_init_reference_state("robot", torch.tensor([1, 2]), ORIGINS, state)
    torch.testing.assert_close(state.base_pos_w[[1, 2]].cpu(), _expected_base_pos([1, 2], [5, 10]) + ORIGINS[[1, 2]])
    term.fill_init_reference_state("robot", torch.tensor([1, 2]), None, state)
    torch.testing.assert_close(state.base_pos_w[[1, 2]].cpu(), _expected_base_pos([1, 2], [5, 10]))


def test_init_reference_state_for_named_entities_is_theirs_alone(env, library):
    manager = make_manager(env, library)

    full = manager.get_init_reference_state()
    named = manager.get_init_reference_state(torch.tensor([1, 3]), entity_names=["object"])

    assert list(named) == ["object"]
    torch.testing.assert_close(named["object"].base_pos_w, full["object"].base_pos_w[[1, 3]])


@pytest.mark.parametrize(
    "entity_names, error, match",
    [
        (["robot", "ghost"], ValueError, r"no motion term tracks \['ghost'\]; tracked entities: \['object', 'robot'\]"),
        ("robot", TypeError, r"got the string 'robot'; pass \['robot'\]"),
    ],
    ids=["untracked", "bare-string"],
)
@pytest.mark.parametrize("getter", ["get_init_reference_state", "get_reset_reference_state"])
def test_init_reference_state_refuses_names_it_cannot_gather(env, library, entity_names, error, match, getter):
    manager = make_manager(env, library)

    with pytest.raises(error, match=match), manager.resetting(torch.tensor([0])):
        getattr(manager, getter)(torch.tensor([0]), entity_names=entity_names)


def test_reset_rewinds_the_clock_and_fill_data_refreshes_only_the_reset_envs(env, library):
    manager = _clip1_manager(env, library)
    manager.reset()
    pin_clips(manager.get_term("default"), [1, 1, 1, 1], start_s=HALF)
    for _ in range(5):
        manager.compute(DT)
    torch.testing.assert_close(_dofs(manager), _expected_dofs([1] * 4, [5] * 4))

    manager.reset(torch.tensor([0, 2]))
    manager.fill_data(torch.tensor([0, 2]))
    torch.testing.assert_close(_dofs(manager), _expected_dofs([1] * 4, [0, 5, 0, 5]))

    # The untouched clocks kept running; the reset ones restarted.
    manager.compute(DT)
    torch.testing.assert_close(_dofs(manager), _expected_dofs([1] * 4, [1, 6, 1, 6]))


@pytest.mark.parametrize(
    "data_start_from, window_frame, gathers",
    [("current_time", 0, ["compute", "compute"]), ("one_frame_interval", 1, ["init", "init", "compute", "compute"])],
)
def test_a_reset_gathers_the_init_state_off_its_window_when_the_window_starts_there(
    env, library, monkeypatch, data_start_from, window_frame, gathers
):
    """The window's first frame *is* the init state when it starts at the current time.

    So inside ``resetting`` the first init-state request fills the reset envs' window (one ``compute`` per
    tracked entity) and reads the state off it, and the exit does not fill it again. A window a frame
    ahead still gathers the init state on its own, and is filled on exit.
    """
    manager = make_manager(env, library, data_start_from=data_start_from)
    term = manager.get_term("default")
    reset = torch.tensor([1, 3])
    manager.reset(reset)
    pin_clips(term, [0, 1, 2, 0], start_s=[HALF, 0.1 + HALF, 0.2 + HALF, 0.3 + HALF])
    calls = []
    for name, label in (("compute", "compute"), ("fill_init_reference_state", "init")):
        original = getattr(term, name)

        def counted(*args, _label=label, _original=original, **kwargs):
            calls.append(_label)
            return _original(*args, **kwargs)

        monkeypatch.setattr(term, name, counted)

    with manager.resetting(reset):
        seeded = manager.get_reset_reference_state(reset, entity_names=["robot"])["robot"]
        manager.get_reset_reference_state(reset, entity_names=["object"])

    assert calls == gathers
    torch.testing.assert_close(seeded.dofs_pos.cpu(), _expected_dofs([1, 0], [5, 15]))
    torch.testing.assert_close(_dofs(manager)[[1, 3]], _expected_dofs([1, 0], [5 + window_frame, 15 + window_frame]))


def test_resetting_fills_the_window_on_exit_when_nothing_asks_and_not_after_an_error(env, library):
    manager = make_manager(env, library)
    term = manager.get_term("default")
    manager.reset(torch.arange(4))
    pin_clips(term, [0, 1, 2, 0], start_s=HALF)

    with pytest.raises(RuntimeError, match="boom"), manager.resetting(torch.tensor([0])):
        raise RuntimeError("boom")
    assert (_dofs(manager)[0] == 0).all()  # the fresh buffer: an aborted reset fills nothing
    with manager.resetting(torch.tensor([0, 2])):
        pass

    torch.testing.assert_close(_dofs(manager)[[0, 2]], _expected_dofs([0, 2], [0, 0]))
    # And the scope closed: a request outside one gathers, so it serves every env, not frame 0 of a window.
    gathered = manager.get_reset_reference_state(torch.arange(4))["robot"]
    torch.testing.assert_close(gathered.dofs_pos.cpu(), _expected_dofs([0, 1, 2, 0], [0] * 4))


def test_inside_a_reset_the_init_state_of_an_env_that_is_not_resetting_is_still_gathered(env, library):
    """Only ``get_reset_reference_state`` reads the window: frame 0 of a running env's window is mid-episode."""
    manager = make_manager(env, library, data_start_from="current_time")
    term = manager.get_term("default")
    manager.reset(torch.arange(4))
    pin_clips(term, [0, 1, 2, 0], start_s=HALF)
    for _ in range(5):
        manager.compute(DT)
    manager.reset(torch.tensor([1, 3]))

    with manager.resetting(torch.tensor([1, 3])):
        running = manager.get_init_reference_state(torch.tensor([0, 2]))["robot"]

    torch.testing.assert_close(running.dofs_pos.cpu(), _expected_dofs([0, 2], [0, 0]))
    torch.testing.assert_close(_dofs(manager)[[0, 2]], _expected_dofs([0, 2], [5, 5]))  # their windows ran on


def test_the_ended_episodes_validity_is_taken_as_the_manager_resets(env, library):
    """A reset-time reader grades the ended episode off it: the window itself moves on to the new one."""
    manager = make_manager(env, library)
    term = manager.get_term("default")
    pin_clips(term, [0, 0, 1, 1], start_s=[1.0, HALF, 1.0, HALF])  # envs 0 and 2 have played their clip out
    manager.fill_data(torch.arange(4))
    with pytest.raises(ValueError, match=r"nothing recorded it\. Call motion_manager\.record_ended_episode_validity"):
        manager.ended_episode_validity("robot")
    manager.record_ended_episode_validity("robot")

    manager.reset(torch.tensor([0, 1, 2]))
    pin_clips(term, [0, 0, 1, 1], start_s=HALF)
    manager.fill_data(torch.arange(4))

    assert manager.ended_episode_validity("robot").tolist() == [False, True, False, False]  # env 3 never reset
    assert manager.data["robot"].validity[:, 0].all()


def test_reference_frame_tracks_the_current_time(env, library):
    manager = make_manager(env, library, num_frames=3)
    pin_clips(manager.get_term("default"), [2, 2, 2, 2])
    manager.compute(0.05)
    manager.compute(0.04)  # 0.09 s: frame 4

    torch.testing.assert_close(manager.reference_frame["robot"].dofs_pos[:, 0].cpu(), _expected_dofs([2] * 4, [4] * 4))


def test_rewind_to_start_replays_the_assigned_clip_from_frame_zero(env, library):
    manager = make_manager(env, library)
    pin_clips(manager.get_term("default"), [0, 1, 2, 0], start_s=0.1 + HALF)
    manager.compute(DT)

    manager.rewind_to_start(torch.arange(4))
    manager.fill_data(torch.arange(4))

    torch.testing.assert_close(_dofs(manager), _expected_dofs([0, 1, 2, 0], [0] * 4))


def test_frame_validity_boundary_matches_the_resync_wrap_check(env, library):
    """``_past_clip_end`` (the resync wrap check) must be ``~validity`` of the fill, bit for bit."""
    term = make_manager(env, library).get_term("default")
    data = term.data_source.motion_sequences["robot"]
    # Starts in steps of a third of a frame across the end of clip 0 (20 frames).
    for start_frame in torch.arange(18.0, 21.0, 1.0 / 3.0):
        pin_clips(term, [0, 0, 0, 0], start_s=float(start_frame) / FRAMERATE)
        assigned = torch.arange(4)
        sample_ts = torch.zeros(4, 1)
        frame, valid = term._frame_index_and_validity(assigned, sample_ts, data)
        assert torch.equal(term._past_clip_end(assigned, sample_ts, data), ~valid)
        assert bool(valid[0, 0]) is (float(frame[0, 0]) < 20)


def test_fields_with_their_own_clip_split_are_read_by_that_split(env, library):
    """A field whose per-clip split differs from ``buffer_length`` is gathered by its own offsets."""
    manager = make_manager(env, library)
    term = manager.get_term("default")
    seq = term.data_source.motion_sequences["robot"]
    # Same row total, one frame moved from clip 0 to clip 1.
    shifted = [19, 31, 40]
    seq.links_pos_w = ConcatBatchTensor(tensors=[torch.full((n, 4, 3), float(i)) for i, n in enumerate(shifted)])
    pin_clips(term, [0, 1, 2, 1], start_s=[0.0, 0.0, 0.0, 30 * DT])

    manager.fill_data(torch.arange(4))

    ref = manager.data["robot"]
    assert ref.links_pos_w[:, 0, 0, 0].tolist() == [0.0, 1.0, 2.0, 1.0]
    torch.testing.assert_close(_dofs(manager), _expected_dofs([0, 1, 2, 1], [0, 0, 0, 29]))


def test_an_empty_clip_reads_as_invalid_instead_of_out_of_bounds(env, library):
    manager = make_manager(env, library)
    term = manager.get_term("default")
    empty = BatchMotionSequence.make_empty(buffer_lengths=[0, 30, 40], num_dofs=len(ROBOT_DOFS), num_links=4)
    empty.framerate = torch.full((3,), FRAMERATE)
    term.data_source.motion_sequences["robot"] = empty
    pin_clips(term, [0, 1, 2, 1])

    manager.fill_data(torch.arange(4))

    assert manager.data["robot"].validity[:, 0].tolist() == [False, True, True, True]


# ---------------------------------------------------------------------------
# freeze_at_timestep_zero_prob
# ---------------------------------------------------------------------------


def test_freeze_holds_only_clocks_at_zero_and_rerolls_every_step(env, library):
    manager = _clip1_manager(env, library, freeze_at_timestep_zero_prob=1.0)
    term = manager.get_term("default")
    pin_clips(term, [1, 1, 1, 1], start_s=HALF)

    manager.compute(DT)
    torch.testing.assert_close(_dofs(manager), _expected_dofs([1] * 4, [0] * 4))

    term.freeze_at_timestep_zero_prob = 0.0  # re-rolled per step, not latched
    manager.compute(DT)
    term.freeze_at_timestep_zero_prob = 1.0
    manager.compute(DT)  # past t=0 now, so nothing freezes
    torch.testing.assert_close(_dofs(manager), _expected_dofs([1] * 4, [2] * 4))

    manager.reset(torch.tensor([0, 1]))
    manager.compute(DT)
    torch.testing.assert_close(_dofs(manager), _expected_dofs([1] * 4, [0, 0, 3, 3]))


def test_freeze_probability_is_each_envs_own_terms(env, library):
    """Two terms at 0.5 over disjoint envs freeze each env with p=0.5, not the 0.75 of AND-ed rolls."""
    torch.manual_seed(0)
    manager = make_manager(
        env,
        terms={
            name: term_options(library, motion_weights=[0.0, 1.0, 0.0], freeze_at_timestep_zero_prob=0.5)
            for name in ("a", "b")
        },
    )
    for term in manager.terms.values():
        pin_clips(term, [1] * term.num_assigned_envs, start_s=HALF)

    frozen = torch.zeros(4)
    trials = 1000
    for _ in range(trials):
        manager.reset()
        manager.compute(DT)
        manager.fill_data(torch.arange(4))
        frozen += torch.isclose(_dofs(manager)[:, 0], robot_dofs_at(1, 0)[0]).float()

    assert ((frozen / trials - 0.5).abs() < 0.07).all(), frozen / trials


# ---------------------------------------------------------------------------
# Several terms
# ---------------------------------------------------------------------------


def _libraries(tmp_path, counts):
    return {f"motion_{i}": term_options(write_library(tmp_path / f"lib_{i}", [5] * n)) for i, n in enumerate(counts)}


def test_envs_split_across_terms_in_proportion_to_their_trajectories(env, tmp_path):
    """Largest-remainder split with a floor of one env each; the old floor-then-trim overflowed to 5 of 4."""
    manager = make_manager(env, terms=_libraries(tmp_path, [1, 1, 6]))

    assert [term.assigned_env_slice for term in manager.terms.values()] == [slice(0, 1), slice(1, 2), slice(2, 4)]


def test_terms_left_without_envs_are_dropped(env, tmp_path):
    manager = make_manager(env, terms=_libraries(tmp_path, [1] * 5))

    assert manager.active_terms == ["motion_0", "motion_1", "motion_2", "motion_3"]
    manager.reset()
    manager.compute(DT)
    manager.reset(torch.tensor([0]))
    assert manager.get_init_reference_state(torch.arange(4))["robot"].dofs_pos.shape == (4, len(ROBOT_DOFS))
    assert manager.reference_frame["robot"].validity.all()


def test_summary_lists_each_term_and_its_trajectories(env, library):
    summary = make_manager(env, library).summary()

    assert "MotionManager" in summary
    assert re.search(r"default\s*\|\s*3\b", summary), summary  # the row, boxed (human) or not (agent)


# ---------------------------------------------------------------------------
# Motion playback phase
# ---------------------------------------------------------------------------


def test_motion_phase_is_the_current_frame_over_the_clip(env, library):
    manager = make_manager(env, library)
    pin_clips(manager.get_term("default"), [1, 1, 0, 0], start_s=[14.5 * DT, 0.0, 19.5 * DT, 0.0])

    torch.testing.assert_close(manager.get_motion_phase("robot").cpu(), torch.tensor([14 / 29, 0.0, 1.0, 0.0]))


def test_motion_phase_of_an_untracked_entity_names_the_tracked_ones(env, library):
    manager = make_manager(env, library)

    with pytest.raises(ValueError, match=r"'robtot' is not tracked.*\['object', 'robot'\]"):
        manager.get_motion_phase("robtot")


# ---------------------------------------------------------------------------
# Splitting the library across DDP ranks (single-process: the assignment is called per rank)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "method, rank, world_size, expected",
    [("Even", 0, 2, [0, 2, 4, 6]), ("Segment", 1, 2, [4, 5, 6, 7]), ("Even", 0, 1, list(range(8)))],
)
def test_rank_split_keeps_the_ranks_trajectories_and_global_weights(env, tmp_path, method, rank, world_size, expected):
    root = write_library(tmp_path, [5] * 8)
    manager = make_manager(
        env, mp_split_method=method, terms={"default": term_options(root, adaptive_motion_weights=True)}
    )
    term = manager.get_term("default")

    manager._assign_motion_trajectories_for_this_process(local_rank=rank, world_size=world_size)

    assert term.num_trajectories == len(expected)
    assert term._local_to_global.tolist() == expected
    assert term._motion_weights.shape == (8,)


def test_rank_split_leaving_an_adaptive_term_empty_raises(env, tmp_path):
    root = write_library(tmp_path, [5, 5])
    manager = make_manager(
        env, mp_split_method="Even", terms={"default": term_options(root, adaptive_motion_weights=True)}
    )

    with pytest.raises(ValueError, match="zero local trajectories"):
        manager._assign_motion_trajectories_for_this_process(local_rank=2, world_size=4)


def test_rank_split_drops_an_empty_non_adaptive_term(env, tmp_path):
    root = write_library(tmp_path, [5, 5])
    manager = make_manager(env, root, mp_split_method="Even")

    manager._assign_motion_trajectories_for_this_process(local_rank=2, world_size=4)

    assert "default" not in manager.terms


# ---------------------------------------------------------------------------
# Symmetric augmentation (mirroring across the x-z plane)
# ---------------------------------------------------------------------------

# Left/right pairs of G1 legs, and the matching ankle links.
_MIRROR_DOFS = ["left_hip_pitch_joint", "right_hip_pitch_joint", "left_hip_roll_joint", "right_hip_roll_joint"]
_MIRROR = dict(
    symmetric_augmentation=True,
    symmetric_augmentation_joint_mapping=[1, 0, 3, 2],
    symmetric_augmentation_joint_reverse_buf=[1.0, 1.0, -1.0, -1.0],
    symmetric_augmentation_links_mapping=[0, 1, 3, 2],
)


def _mirror_manager(env, library, **overrides):
    options = _MIRROR | overrides
    return make_manager(
        env, terms={"default": term_options(library, entities=("robot",), dofs={"robot": _MIRROR_DOFS})}, **options
    )


@pytest.mark.parametrize(
    "field, buffer, expected",
    [
        ("dofs_pos", [[1.0, 2.0, 3.0, 4.0]], [[2.0, 1.0, -4.0, -3.0]]),
        ("base_pos_w", [[1.0, 2.0, 3.0]], [[1.0, -2.0, 3.0]]),
        ("base_ang_vel_w", [[1.0, 2.0, 3.0]], [[-1.0, 2.0, -3.0]]),
        ("base_quat_w", [[1.0, 2.0, 3.0, 4.0]], [[1.0, -2.0, 3.0, -4.0]]),
        (
            "links_pos_w",
            [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [0.0, 1.0, 2.0]]],
            [[[1.0, -2.0, 3.0], [4.0, -5.0, 6.0], [0.0, -1.0, 2.0], [7.0, -8.0, 9.0]]],
        ),
        (
            "links_quat_w",
            [[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]],
            [[[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [5.0, -6.0, 7.0, -8.0], [1.0, -2.0, 3.0, -4.0]]],
        ),
        (
            "links_lin_vel_w",
            [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]],
            [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [4.0, -5.0, 6.0], [1.0, -2.0, 3.0]]],
        ),
        (
            "links_ang_vel_w",
            [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]],
            [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-4.0, 5.0, -6.0], [-1.0, 2.0, -3.0]]],
        ),
    ],
)
def test_mirror_primitives(env, library, field, buffer, expected):
    manager = _mirror_manager(env, library)

    out = manager._mirror_rows(field, torch.tensor(buffer))

    torch.testing.assert_close(out, torch.tensor(expected))


def _mirror_mask(manager, mask):
    # Sampled per env at random on every reset; no public setter.
    manager._env_symmetric_augmentation_mask[:] = torch.tensor(mask)


def test_mirrored_envs_hold_the_mirror_of_the_reference(env, library):
    plain = make_manager(
        env, terms={"default": term_options(library, entities=("robot",), dofs={"robot": _MIRROR_DOFS})}
    )
    mirrored = _mirror_manager(env, library)
    for manager in (plain, mirrored):
        pin_clips(manager.get_term("default"), [0, 1, 2, 0], start_s=0.1)
    _mirror_mask(mirrored, [True, False, True, False])

    plain.fill_data(torch.arange(4))
    mirrored.fill_data(torch.arange(4))

    a, b = plain.data["robot"], mirrored.data["robot"]
    torch.testing.assert_close(b.dofs_pos[[1, 3]], a.dofs_pos[[1, 3]])
    torch.testing.assert_close(b.dofs_pos[[0, 2]], a.dofs_pos[[0, 2]][..., [1, 0, 3, 2]] * torch.tensor([1, 1, -1, -1]))
    torch.testing.assert_close(b.links_pos_w[0, :, 2, 1], -a.links_pos_w[0, :, 3, 1])


def test_augmentation_needs_every_mapping_configured(env, library):
    plain = make_manager(
        env, terms={"default": term_options(library, entities=("robot",), dofs={"robot": _MIRROR_DOFS})}
    )
    incomplete = _mirror_manager(env, library, symmetric_augmentation_links_mapping=None)
    for manager in (plain, incomplete):
        pin_clips(manager.get_term("default"), [1, 1, 1, 1])
    _mirror_mask(incomplete, [True] * 4)

    plain.fill_data(torch.arange(4))
    incomplete.fill_data(torch.arange(4))

    torch.testing.assert_close(incomplete.data["robot"].dofs_pos, plain.data["robot"].dofs_pos)


def test_init_reference_state_mirrors_only_masked_envs(env, library):
    manager = _mirror_manager(env, library)
    _mirror_mask(manager, [True, True, False, False])
    state = MotionReferenceState.make_empty(4, 4)
    state.dofs_pos[:] = torch.arange(4.0)
    state.base_pos_w[:] = torch.tensor([1.0, 2.0, 3.0])
    state.base_ang_vel_w[:] = torch.tensor([1.0, 2.0, 3.0])

    manager._symmetric_augment_reference_data(state, torch.arange(4))

    torch.testing.assert_close(state.dofs_pos, torch.tensor([[1.0, 0.0, -3.0, -2.0]] * 2 + [[0.0, 1.0, 2.0, 3.0]] * 2))
    assert state.base_pos_w[:, 1].tolist() == [-2.0, -2.0, 2.0, 2.0]
    # The angular velocity mirrors too (it once did not).
    torch.testing.assert_close(state.base_ang_vel_w[0], torch.tensor([-1.0, 2.0, -3.0]))
    torch.testing.assert_close(state.base_ang_vel_w[2], torch.tensor([1.0, 2.0, 3.0]))


def test_augmentation_skips_fields_the_reference_lacks(env, library):
    manager = _mirror_manager(env, library)
    _mirror_mask(manager, [True] * 4)
    data = MotionReferenceData(dofs_pos=torch.ones(4, 1, 4), validity=torch.ones(4, 1, dtype=torch.bool))

    manager._symmetric_augment_reference_data(data, torch.arange(4))

    torch.testing.assert_close(data.dofs_pos[0, 0], torch.tensor([1.0, 1.0, -1.0, -1.0]))


# ---------------------------------------------------------------------------
# MotionTerm / MotionTermOptions declare the same config surface twice
# ---------------------------------------------------------------------------


def test_term_parameters_and_option_fields_match():
    from eden.options.managers.base import ManagerTermOptions
    from eden.options.managers.motions import MotionTermOptions

    option_fields = set(MotionTermOptions.model_fields) - set(ManagerTermOptions.model_fields)
    assert set(MotionTerm.get_parameter_names()) == option_fields


def test_term_and_option_defaults_match():
    """``configure()`` reads the term's class defaults, a direct ``MotionTermOptions`` the field's."""
    from eden.options.managers.motions import MotionTermOptions

    mismatches = {}
    for name in MotionTerm.get_parameter_names():
        field = MotionTermOptions.model_fields[name]
        option_default = field.default_factory() if field.default_factory is not None else field.default
        if option_default != getattr(MotionTerm, name):
            mismatches[name] = (getattr(MotionTerm, name), option_default)
    assert not mismatches


# ---------------------------------------------------------------------------
# sample_start_times
# ---------------------------------------------------------------------------


def _bin_weights(batch_sizes, weights=None) -> ConcatBatchTensor:
    cbt = ConcatBatchTensor(batch_sizes=batch_sizes, data_shape=tuple(), device="cpu")
    for i, size in enumerate(batch_sizes):
        cbt[i] = torch.ones(size) / size if weights is None else torch.as_tensor(weights[i], dtype=torch.float)
    return cbt


@pytest.mark.parametrize("uniform_bins", [False, True])
@pytest.mark.parametrize(
    "batch_sizes, bin_length_s, num_envs", [([10, 20, 5], 0.5, 10), ([3, 100], 0.1, 6), ([8], 0.5, 500)]
)
def test_start_times_stay_inside_each_motions_bins(batch_sizes, bin_length_s, num_envs, uniform_bins):
    motion_ids = torch.arange(num_envs) % len(batch_sizes)

    starts = sample_start_times(motion_ids, _bin_weights(batch_sizes), "cpu", bin_length_s, uniform_bins=uniform_bins)

    upper = torch.tensor(batch_sizes, dtype=torch.float)[motion_ids] * bin_length_s
    assert (starts >= 0).all() and (starts <= upper + 1e-6).all()


def test_all_bin_weight_on_one_bin_starts_every_env_in_it():
    starts = sample_start_times(
        torch.zeros(1000, dtype=torch.long), _bin_weights([10], [[0] * 5 + [1] + [0] * 4]), "cpu", 1.0
    )

    assert ((starts >= 5.0) & (starts < 6.0)).all()


def test_uniform_fast_path_draws_the_same_distribution_as_the_multinomial():
    torch.manual_seed(0)
    batch_sizes = [5, 8, 3]
    motion_ids = torch.randint(0, 3, (60_000,))
    for uniform_bins in (False, True):
        starts = sample_start_times(motion_ids, _bin_weights(batch_sizes), "cpu", 1.0, uniform_bins=uniform_bins)
        for m, num_bins in enumerate(batch_sizes):
            counts = torch.bincount(starts[motion_ids == m].floor().long(), minlength=num_bins).float()
            torch.testing.assert_close(
                counts / counts.sum(), torch.full((num_bins,), 1.0 / num_bins), atol=0.02, rtol=0
            )


def test_no_motion_ids_sample_nothing_and_a_missing_bin_length_raises():
    assert sample_start_times(torch.tensor([], dtype=torch.long), _bin_weights([5]), "cpu", 1.0).shape == (0,)
    with pytest.raises(ValueError, match="motion_bin_length_s must be provided"):
        sample_start_times(torch.tensor([0]), _bin_weights([5]), "cpu", None)


@pytest.mark.parametrize("uniform_bins", [False, True])
def test_start_times_clamp_to_each_clips_last_frame(tmp_path, uniform_bins):
    """A clip shorter than one bin once drew starts past its end: the zero-quat NaN on first reset."""
    dataset = load_dataset(write_library(tmp_path, [7, 300]))
    motion_ids = torch.tensor([0, 1] * 1000)

    starts = sample_start_times(
        motion_ids, _bin_weights([1, 1]), "cpu", 1.0, data_source=dataset, uniform_bins=uniform_bins
    )

    assert (starts[motion_ids == 0] <= 6 * DT + 1e-6).all()
    assert (starts[motion_ids == 1] < 1.0).all() and (starts[motion_ids == 1] > 0.5).any()


# ---------------------------------------------------------------------------
# env_motion_groups: each env samples only the clips of the variant it holds
# ---------------------------------------------------------------------------
#
# A G1 and a heterogeneous box bank of two sizes ("a", "b") over five envs (uneven blocks), with clips
# labelled a/b/a/b. An env holding the "a" box must only ever play clips 0 and 2.

GROUP_LENGTHS = [20, 30, 40, 25]
GROUP_LABELS = ["a", "b", "a", "b"]
VARIANT_LABELS = ["a", "b"]


def _box_bank():
    from eden.options.entities import BoxOptions, GroupedEntityOptions

    return GroupedEntityOptions(
        grouped_entities=[BoxOptions(size=(0.1, 0.1, 0.1)), BoxOptions(size=(0.2, 0.2, 0.2))],
        default_root_pos=(1.0, 0.0, 0.1),
    )


def _groups(**overrides):
    from eden.options.managers.motions import MotionGroupOptions

    return MotionGroupOptions(**({"entity_name": "object", "variant_labels": VARIANT_LABELS} | overrides))


@pytest.fixture(scope="module")
def grouped_library(tmp_path_factory):
    return write_library(tmp_path_factory.mktemp("grouped"), GROUP_LENGTHS, labels=GROUP_LABELS)


@pytest.fixture(scope="module")
def grouped_env(grouped_library):
    from eden.options.robots.unitree_g1 import G1Dofs29
    from tests.support.motion import manager_options

    return build_mimic_env(
        grouped_library,
        num_envs=5,
        scene={"robot": G1Dofs29(), "object": _box_bank()},
        motion_options=manager_options(terms={"default": term_options(grouped_library, env_motion_groups=_groups())}),
    )


def _grouped_manager(env, library, **term_kwargs):
    term_kwargs.setdefault("env_motion_groups", _groups())
    return make_manager(env, terms={"default": term_options(library, **term_kwargs)})


def _assert_each_env_plays_its_variant(env, term):
    variants = env.entities["object"].env_variant_idx.tolist()
    played = [GROUP_LABELS[i] for i in term._assigned_env_motion_selection.tolist()]
    assert played == [VARIANT_LABELS[v] for v in variants], (played, variants)


def test_the_bank_puts_both_variants_in_uneven_blocks(grouped_env):
    assert sorted(set(grouped_env.entities["object"].env_variant_idx.tolist())) == [0, 1]


@pytest.mark.parametrize("strategy", ["independent", "concat_motion_bins"])
def test_every_draw_stays_in_the_envs_variant_group(grouped_env, grouped_library, strategy):
    term_kwargs = {"env_starting_stub_sampling_strategy": strategy}
    if strategy == "concat_motion_bins":
        term_kwargs["motion_bin_length_s"] = 0.1
    manager = _grouped_manager(grouped_env, grouped_library, **term_kwargs)
    term = manager.get_term("default")
    _assert_each_env_plays_its_variant(grouped_env, term)  # the initial assignment
    for i in range(30):
        mask = torch.rand(5) < 0.5
        term.reset(mask if i % 2 else torch.nonzero(mask).flatten().to(term._assigned_env_motion_selection.device))
        _assert_each_env_plays_its_variant(grouped_env, term)
    # Both clips of each group get drawn: the restriction narrows the library, it does not pin a clip.
    seen = set()
    for _ in range(30):
        term.reset(None)
        seen |= set(term._assigned_env_motion_selection.tolist())
    assert seen == {0, 1, 2, 3}


def test_a_group_whose_weights_are_all_zero_falls_back_to_uniform_within_it(grouped_env, grouped_library):
    term = _grouped_manager(grouped_env, grouped_library, motion_weights=[1.0, 0.0, 1.0, 0.0]).get_term("default")
    seen_b = set()
    for _ in range(20):
        term.reset(None)
        _assert_each_env_plays_its_variant(grouped_env, term)
        seen_b |= {c for c in term._assigned_env_motion_selection.tolist() if GROUP_LABELS[c] == "b"}
    assert seen_b == {1, 3}


def test_a_clip_end_resync_keeps_each_env_on_its_group(grouped_library):
    import eden as en
    from eden.managers.terms.events.motion import ReferenceStateInitialization
    from eden.options.managers.events import EventManagerOptions
    from eden.options.robots.unitree_g1 import G1Dofs29
    from tests.support.motion import manager_options

    env = build_mimic_env(
        grouped_library,
        num_envs=5,
        scene={"robot": G1Dofs29(), "object": _box_bank()},
        motion_options=manager_options(
            terms={"default": term_options(grouped_library, env_motion_groups=_groups())}, resync_on_motion_end=True
        ),
        event_options=EventManagerOptions(
            reset_robot=ReferenceStateInitialization.configure(
                mode=en.EventMode.RESET, entity_name="robot", dofs_position_range=(0.0, 0.0)
            )
        ),
    )
    term = env.motion_manager.get_term("default")
    before = term._assigned_env_motion_selection.clone()
    for _ in range(3 * max(GROUP_LENGTHS)):  # every clip wraps at least twice
        env.motion_manager.compute(DT)
    assert torch.equal(term._assigned_env_motion_selection, before)
    _assert_each_env_plays_its_variant(env, term)


@pytest.mark.parametrize(
    "groups, error, match",
    [
        (
            {"variant_labels": ["a"]},
            ValueError,
            r"clips labelled \{'b': 2\}.*no variant of entity 'object' matches \(variant_labels=\['a'\]\)",
        ),
        ({"variant_labels": ["a", "b", "c"]}, ValueError, r"variant label\(s\) \['c'\].*match no clip"),
        ({"variant_labels": ["a", "b", "a"]}, ValueError, r"3 labels but entity 'object' has 2 variants"),
        ({"entity_name": "robot"}, TypeError, r"'robot' names a G1Dofs29 entity.*heterogeneous"),
        ({"entity_name": "table"}, KeyError, r"No scene entity named 'table'.*env_motion_groups"),
        ({"metadata_key": "subjects"}, KeyError, r"no per-clip metadata 'subjects'"),
    ],
)
def test_a_misconfigured_group_fails_at_construction_naming_the_value(
    grouped_env, grouped_library, groups, error, match
):
    with pytest.raises(error, match=match):
        _grouped_manager(grouped_env, grouped_library, env_motion_groups=_groups(**groups))


def test_groups_refuse_symmetric_augmentation(grouped_env, grouped_library):
    with pytest.raises(ValueError, match=r"cannot be combined with env_motion_groups.*\['default'\]"):
        make_manager(
            grouped_env,
            terms={"default": term_options(grouped_library, env_motion_groups=_groups())},
            symmetric_augmentation=True,
        )


# ---------------------------------------------------------------------------
# contact_label reaches the per-env reference buffers
# ---------------------------------------------------------------------------


def test_the_reference_carries_each_envs_contact_labels_frame_by_frame(env, tmp_path):
    rng = np.random.default_rng(0)
    labels = {c: rng.integers(-1, 2, size=(n, len(ROBOT_LINKS))).astype(np.int8) for c, n in enumerate(LENGTHS)}
    for c, n in enumerate(LENGTHS):
        write_clip(tmp_path / f"clip_{c:02d}.npz", n, clip=c, overrides={"robot.contact_label": labels[c]})
    manager = make_manager(env, tmp_path, num_frames=3)
    term = manager.get_term("default")
    pin_clips(term, [0, 1, 2, 1], start_s=[HALF, 2 * DT + HALF, 5 * DT + HALF, HALF])
    manager.fill_data(torch.arange(4))

    buffer = manager.data["robot"].contact_label.cpu()
    assert buffer.dtype == torch.int8 and buffer.shape == (env.num_envs, 3, len(ROBOT_LINKS))
    for e, (clip, start) in enumerate([(0, 0), (1, 2), (2, 5), (1, 0)]):
        assert torch.equal(buffer[e], torch.as_tensor(labels[clip][start : start + 3])), e


class NoBodyFrameMotionDataset(SyntheticMotionDataset):
    """Drops the body-frame link fields, as a large-library dataset does to save memory."""

    def prepare(self):
        sequences, metadata = super().prepare()
        for seq in sequences.values():
            for field in ("links_pos_b", "links_quat_b", "links_lin_vel_b", "links_ang_vel_b"):
                setattr(seq, field, None)
        return sequences, metadata


if "no_body_frame_motion_dataset" not in DATASET_REGISTRY:
    DATASET_REGISTRY.register(NoBodyFrameMotionDataset)


def test_a_library_without_body_frame_link_fields_still_sizes_its_link_buffers(env, library):
    from eden.options.datasets import DatasetOptions

    options = term_options(library)
    options.dataset_options = DatasetOptions(name="no_body_frame_motion_dataset", raw_data_path=str(library))
    manager = make_manager(env, terms={"default": options})
    pin_clips(manager.get_term("default"), [2, 2, 2, 2])
    manager.fill_data(torch.arange(4))

    assert manager.data["robot"].links_pos_w.shape[-2] == len(ROBOT_LINKS)
    assert manager.get_term("default").data_source.motion_sequences["robot"].links_pos_b is None
    assert manager.data["robot"].links_pos_w[0, 0].abs().sum() > 0
