"""Tests for ``eden.envs.graded_graph``: the CUDA graph ``RLEnvBase.step`` replays its graded phase from
(``EnvOptions.cuda_graph_options``).

The graph itself only exists on CUDA, and this suite holds Genesis on ``gs.cpu``, so the scenarios that build one
run in a process of their own (``tests/support/graded_graph.py``) and skip without a GPU. The helpers its
validation rests on are exercised here directly.
"""

from __future__ import annotations

import json
from collections import OrderedDict
import pathlib
import subprocess
import sys
import types

import pytest
import torch

from eden.envs.base import env_class_for_config
from eden.envs.graded_graph import (
    BakedHostReadError,
    _address,
    _reachable_tensors,
    _ReadRecorder,
    _same_bits,
    _value_at,
    _walked,
    _walked_class,
    _write_back,
    _WriteRecorder,
)
from eden.options import BoxOptions, CudaGraphOptions, EnvOptions, SceneOptions
from eden.utils.configs import EdenConfig, EdenRLConfig
from tests.support.envs import build_env

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="a CUDA graph needs a CUDA device")


def _run(*argv: str) -> dict:
    repo = pathlib.Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, "-m", "tests.support.graded_graph", *argv],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    line = next((line for line in result.stdout.splitlines() if line.startswith("RESULT ")), None)
    assert line is not None, f"graded_graph {argv} printed no RESULT line; stderr tail:\n{result.stderr[-3000:]}"
    return json.loads(line.removeprefix("RESULT "))


def test_a_cpu_env_refuses_the_graph_at_build():
    config = EdenRLConfig(
        env_options=EnvOptions(num_envs=2, cuda_graph_options=CudaGraphOptions()),
        scene_options=SceneOptions(box=BoxOptions(size=(0.1, 0.1, 0.1), default_root_pos=(0.0, 0.0, 0.5))),
    )
    with pytest.raises(ValueError, match=r"needs a CUDA device.*cuda_graph_options=None.*backend=gs\.cuda"):
        build_env(config)


def test_an_env_with_no_graded_phase_refuses_the_graph_at_construction():
    config = EdenConfig(
        env_options=EnvOptions(num_envs=2, cuda_graph_options=CudaGraphOptions()),
        scene_options=SceneOptions(box=BoxOptions(size=(0.1, 0.1, 0.1), default_root_pos=(0.0, 0.0, 0.5))),
    )
    with pytest.raises(ValueError, match=r"EnvBase is not an RLEnvBase.*cuda_graph_options=None"):
        env_class_for_config(config).from_config(config, show_viewer=False)


@requires_cuda
@pytest.mark.slow
@pytest.mark.parametrize("task", ["velocity_tracking_flat", "beyondmimic"])
def test_a_task_replays_its_graded_phase_as_its_eager_step_computes_it(task):
    """Every capture and the replays after it are checked bit for bit against eager runs. A reward weight moved
    mid-run (what a curriculum does) is read from the device, so the one graph replays it without re-capturing.
    """
    assert _run("task", task) == {"captures": 1, "weight_applied": True}


@requires_cuda
@pytest.mark.slow
@pytest.mark.parametrize("scenario", ["field_change", "rebind", "rebind_holder", "env_constant", "term_replaced"])
def test_a_baked_in_value_or_a_rebound_buffer_changed_after_validation_re_captures(scenario):
    """Long after the replays were validated: a configured field changed, a buffer the graph reads rebound to a
    new tensor (or the dict holding it replaced, its own binding untouched), an env constant a term may read
    (``max_episode_length``) changed, or a term swapped in its manager for another object. Each re-captures, and
    the new capture is validated again.
    """
    assert _run(scenario) == {"captures": 2, "error": None}


@requires_cuda
@pytest.mark.slow
def test_a_one_off_change_right_after_the_capture_re_captures_once():
    """A curriculum switching stage early in a run is not a re-capture storm."""
    assert _run("one_off") == {"captures": 2, "error": None}


@requires_cuda
@pytest.mark.slow
def test_a_field_changed_on_every_step_refuses_to_re_capture_every_step_naming_it():
    result = _run("storm")
    assert result["captures"] == 3  # the capture, then the two re-captures a window allows
    assert "reward term 'probe' field 'scale' changed again" in result["error"]
    assert "cuda_graph_options=None" in result["error"]


@requires_cuda
@pytest.mark.slow
def test_a_term_branching_on_the_step_counter_fails_the_capture_naming_its_line():
    """A replay would keep the branch the capture took, whichever step the counter later reaches."""
    result = _run("counter_read")
    assert "capturing the graded phase failed at tests/support/graded_graph.py:" in result["error"]
    assert "(counter_gated_reward)" in result["error"]
    assert "env.common_step_counter was read" in result["error"]
    assert "cuda_graph_options=None" in result["error"]


@requires_cuda
@pytest.mark.slow
def test_work_a_capture_does_not_record_fails_the_first_validated_replay():
    """A write outside torch's stream runs once, at capture time. The capture step cannot see that (the write
    ran right before the replay read it); a validated replay runs the graph before the eager phase, so its
    replay reads the capture step's value and disagrees.
    """
    result = _run("side_stream")
    assert result["captures"] == 1
    assert "on a replay" in result["error"]
    assert "env.reward_manager._step_raw_reward" in result["error"]


@requires_cuda
@pytest.mark.slow
def test_state_no_eden_attribute_reaches_advances_once_per_step():
    """The validations undo a tensor the walk cannot reach as well, even one allocated under inference mode, so
    neither the capture step nor a validated replay advances it twice.
    """
    assert _run("hidden_state") == {"captures": 1, "error": None, "counts": [12.0]}


@requires_cuda
@pytest.mark.slow
def test_a_validation_leaves_a_cached_fresh_tensor_as_the_eager_run_wrote_it():
    """Genesis pads a fresh tensor in place and serves it to every later query of the step. Undoing that write
    would hand the step's later readers (an observation, at phase 19) the unpadded values.
    """
    assert _run("fresh_cache") == {"captures": 1, "stale_cache_steps": []}


@requires_cuda
@pytest.mark.slow
def test_a_float_sum_over_contacts_validates_since_every_graded_run_reads_the_padded_form():
    """The eager runs that validate a capture read every contact slot, as the capture does.

    A capture cannot read the live contact count back, so it reads every slot. A sum over the trimmed axis in
    the eager runs could round differently and fail a correct term.
    """
    result = _run("contacts")
    assert result["error"] is None
    assert result["captures"] == 1
    assert len(result["widths"]) == 1


@requires_cuda
@pytest.mark.slow
def test_a_term_hiding_state_in_python_fails_the_capture_steps_validation():
    """Its counter advances once for the eager run and once for the capture, so the replay grades another value."""
    result = _run("python_state")
    assert result["captures"] == 1
    assert "on the capture step" in result["error"]
    assert "disagrees with an eager run of the same step" in result["error"]
    assert "env.reward_manager._step_raw_reward" in result["error"]


@requires_cuda
@pytest.mark.slow
def test_a_term_reading_a_value_back_fails_the_capture_naming_its_line():
    result = _run("host_read")
    assert "capturing the graded phase" in result["error"]
    assert "tests/support/graded_graph.py:" in result["error"]  # the term's own line, not the manager's
    assert "(host_read_reward)" in result["error"]
    assert "cuda_graph_options=None" in result["error"]


class TestWriteRecorder:
    """What a validation undoes after its eager run: every write, in place or through ``out=``."""

    def test_writing_the_saved_values_back_last_first_restores_every_tensor(self):
        buffer = torch.arange(6.0)
        view = buffer[2:4]
        out = torch.zeros(3)
        with _WriteRecorder() as writes:
            buffer.mul_(2.0)
            view.add_(100.0)  # a second write, through a view of the first target
            torch.add(buffer[:3], 1.0, out=out)
        assert len(writes.saved) == 3
        for tensor, saved in reversed(writes.saved):
            tensor.copy_(saved)
        assert torch.equal(buffer, torch.arange(6.0))
        assert torch.equal(out, torch.zeros(3))

    def test_reads_are_not_recorded(self):
        buffer = torch.ones(4)
        with _WriteRecorder() as writes:
            (buffer * 2.0).sum()
        assert writes.saved == []

    def test_a_tensor_the_recorded_run_allocated_is_not_recorded(self):
        """A validation undoes only what existed before its eager run: a fresh tensor it wrote may sit in a cache
        (Genesis pads its contact query in place), and rolling it back would serve that cache stale values.
        """
        persistent = torch.zeros(4)
        with _WriteRecorder() as writes:
            persistent[1:].add_(1.0)
            fresh = torch.zeros(4)
            fresh.fill_(-1.0)
            fresh[:2].add_(1.0)  # through a view of it too
            torch.add(persistent, 1.0, out=fresh)
        assert len(writes.saved) == 1
        target, before = writes.saved[0]
        assert target.data_ptr() == persistent[1:].data_ptr()
        assert torch.equal(before, torch.zeros(3))

    def test_a_tensor_written_twice_through_one_view_is_written_once(self):
        buffer = torch.zeros(4)
        with _WriteRecorder() as writes:
            buffer.add_(1.0)
            buffer.mul_(2.0)
            buffer[1:].add_(1.0)
        assert len(writes.saved) == 3
        assert len(writes.written()) == 2  # the whole buffer, and the view of its tail


def test_writing_back_restores_a_tensor_allocated_under_inference_mode():
    """``RewardManager.compute`` runs under inference mode, so a buffer a reward term allocates there is an inference
    tensor, which refuses an in-place write outside it.
    """
    with torch.inference_mode():
        state = torch.zeros(3)
        with _WriteRecorder() as writes:
            state.add_(1.0)
    _write_back(reversed(writes.saved))
    assert torch.equal(state, torch.zeros(3))


def test_reading_the_step_counter_during_a_capture_raises():
    """The counter changes every step, so a graph would replay the count it read at capture."""
    config = EdenRLConfig(
        env_options=EnvOptions(num_envs=2),
        scene_options=SceneOptions(box=BoxOptions(size=(0.1, 0.1, 0.1), default_root_pos=(0.0, 0.0, 0.5))),
    )
    env = env_class_for_config(config).from_config(config, show_viewer=False)
    with env.graded_capture_scope():  # what GradedPhaseGraph captures in
        with env.graded_capture_scope():  # a nested scope leaves the outer one open
            pass
        with pytest.raises(BakedHostReadError, match=r"env\.common_step_counter was read.*cuda_graph_options=None"):
            env.common_step_counter
    with pytest.raises(RuntimeError, match="capture failed"), env.graded_capture_scope():
        raise RuntimeError("capture failed")  # a failed capture leaves the env stepping eagerly, counter readable
    env.common_step_counter += 1  # outside a capture it reads and writes as a plain int
    assert env.common_step_counter == 1


def test_same_bits_counts_matching_nans_as_equal_and_any_other_difference_as_not():
    nan = torch.tensor([1.0, float("nan")])
    assert _same_bits(nan, nan.clone())
    assert not _same_bits(nan, torch.tensor([1.0, 2.0]))
    assert not _same_bits(torch.zeros(2), torch.zeros(2, dtype=torch.float64))


class TestReadRecorder:
    def test_it_records_every_tensor_an_op_reads_and_writes_through_views_and_lists(self):
        a, b, out = torch.ones(4), torch.ones(4), torch.zeros(6)
        with _ReadRecorder() as reads:
            torch.cat([a[:2], b], out=out)
        assert {t.untyped_storage().data_ptr() for t in (a, b, out)} <= reads.storages


def test_an_address_names_where_and_how_a_tensor_is_read():
    buffer = torch.zeros(4, 3)
    container = {"view": buffer[:, 0]}
    assert _address(_value_at(container, "view")) == _address(buffer[:, 0])  # a new view of the same memory
    assert _value_at(container, "missing") is None
    container["view"] = buffer[:, 1]
    assert _address(_value_at(container, "view")) != _address(buffer[:, 0])


def test_a_binding_is_read_from_a_dict_a_sequence_or_an_attribute():
    class Holder:
        pass

    holder = Holder()
    holder.items = (1, 2)
    assert _value_at({"a": 1}, "a") == 1
    assert _value_at([1, 2], 1) == 2 and _value_at([1, 2], 5) is None
    assert _value_at(holder, "items") == (1, 2) and _value_at(holder, "missing") is None


class _Helper:
    """A helper defined outside Eden (here, in its tests): the walk follows it as it follows Eden's own objects."""


def _class_in(module_name: str, filename: str, monkeypatch) -> type:
    """A fresh class defined in a stand-in module ``module_name`` whose source is ``filename``."""
    module = types.ModuleType(module_name)
    module.__file__ = filename
    monkeypatch.setitem(sys.modules, module_name, module)
    return type("Helper", (), {"__module__": module_name})


def test_the_walk_enters_every_object_but_the_simulator_s_torch_s_the_stdlib_s_and_the_env(monkeypatch):
    """The boundary is the package a class is defined in, not where it is installed.

    An editable Genesis checkout is still the simulator (the walk would reach its per-step query caches, which it
    empties every step), and a task package installed into site-packages is walked like Eden's own code.
    """
    from eden.managers.reward_manager import RewardManager

    assert _walked_class(RewardManager)
    assert _walked_class(_Helper)
    assert _walked_class(
        _class_in("installed_tasks.helpers", "/venv/site-packages/installed_tasks/helpers.py", monkeypatch)
    )
    assert not _walked_class(_class_in("genesis.stand_in", "/home/dev/Genesis/genesis/stand_in.py", monkeypatch))
    assert not _walked_class(torch.nn.Linear)
    assert not _walked_class(OrderedDict)
    assert not _walked(_Helper)  # a class, not an object
    env = env_class_for_config(EdenConfig()).from_config(EdenConfig(), show_viewer=False)
    assert not _walked(env)  # every term's back-reference to it


@requires_cuda
def test_the_walk_reaches_a_tensor_however_deep_it_is_held_and_enters_each_object_once():
    """No depth stops the walk short: a binding below it would go unwatched."""
    root = _Helper()
    holder = root
    for _ in range(30):
        holder.next = _Helper()
        holder.back = root  # a cycle at every level
        holder = holder.next
    holder.buffer = torch.zeros(1, device="cuda")

    found = _reachable_tensors(root)

    assert list(found) == ["env" + ".next" * 30 + ".buffer"]
    assert len(found["env" + ".next" * 30 + ".buffer"].chain) == 30
