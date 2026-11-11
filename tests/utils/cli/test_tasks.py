"""Tests for ``eden.utils.cli.tasks``: the ``eden train`` / ``inference`` / ``export`` / ``deploy`` command helpers.

Four built envs back them: a reacher trained for one real iteration (checkpoint replay, run manifests), a G1
under a joint PD controller (the deploy policy slots), three one-joint arms with velocity commands wrapped
for rsl_rl (command substitution and the ``--async`` loop), and the envs a ``--policy`` with no ``--task``
gets. Deployments are the in-memory ``LoopbackDeployment`` of ``tests/extensions/deployment/conftest.py``.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import pathlib
import stat
import sys
import time
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest
import torch

import eden as en
from eden import actions, observations
from eden.constants import ReferenceSource
from eden.envs.base import EnvBase
from eden.extensions.policies.base import PolicyBase
from eden.extensions.policies.beyondmimic_moves.policy import BeyondMimicMovesPolicy
from eden.extensions.policies.holosoma.policy import HolosomaLocomotionPolicy, holosoma_g1_robot
from eden.extensions.policies.sonic.policy import SonicPolicy
from eden.learning.wrappers.rsl_rl import MIN_RSL_RL_VERSION
from eden.managers import COMMAND_TERM_REGISTRY
from eden.managers.terms.commands.velocity import UniformVelocityCommand
from eden.managers.terms.events import domain_rand
from eden.options import (
    ActionManagerOptions,
    EnvOptions,
    EventManagerOptions,
    ObservationGroupOptions,
    ObservationManagerOptions,
    SceneOptions,
)
from eden.options.learning.rsl_rl.translate import detect_rsl_rl_version
from eden.options.robots.unitree_g1 import G1Dofs23, G1Dofs29
from eden.options.robots.unitree_go1 import Go1
from eden.tasks import TASK_REGISTRY
from eden.tasks.parser import get_task_argparser
from eden.terms import EventTerm, ObsTerm
from eden.utils.cli import tasks
from eden.utils.cli.tasks import (
    _actor_obs_groups,
    _add_shared_policy_cli_args,
    _base_lin_vel_readers,
    _cap_base_command,
    _check_base_lin_vel_source,
    _check_deployable_runner,
    _check_rsl_rl_runner,
    _clone_obs,
    _config_from_checkpoint,
    _detect_checkpoint_framework,
    _EpisodeMetrics,
    _explicit_task_mods,
    _extension_command_options,
    _extension_scene_options,
    _plan_velocity_command,
    _policy_spec,
    _replay_log_root,
    _resolve_step_budget,
    _resolve_velocity_command_term,
    _run_sim2sim_loop,
    _set_policy_command,
    _task_terms_driving_policy_command,
    _velocity_command_ranges,
    deploy_task,
)
from eden.utils.configs import EdenConfig, EdenRLConfig
from eden.utils.warning import warn_once

from eden.extensions.deployment.utils.operator_input import CMD_START
from tests.support.deploy import (
    FirmwareGainsLoopbackDeployment,
    LoopbackDeployment,
    ProgrammedGainsLoopbackDeployment,
    deployments,
    OperatorLoopbackDeployment,
)
from tests.support.envs import build_env


def _shared_parser(argv, command="inference"):
    parser = get_task_argparser(description="t", argv=argv)
    _add_shared_policy_cli_args(parser, command=command)
    return parser


def test_a_mistyped_task_is_named_even_while_another_task_name_clashes(task_package, isolated_task_registry, capsys):
    """The clash belongs to its own name's lookups; the typo's error still names the typo and its close match."""
    clashing = "from eden.tasks import TASK_REGISTRY\n@TASK_REGISTRY.register(name='dup')\nclass {}:\n    pass\n"
    task_package("clashing_cli_pkg", {"a.py": clashing.format("AConfig"), "b.py": clashing.format("BConfig")})
    isolated_task_registry.add_search_path("clashing_cli_pkg")
    with pytest.warns(UserWarning, match="'dup' is registered in 2 locations"), pytest.raises(SystemExit):
        tasks.smoke_task(["--task", "reachr"])
    err = capsys.readouterr().err
    assert "No task named 'reachr' found in task registry. Did you mean 'reacher'?" in err
    assert '[project.entry-points."eden.tasks"]' in err


@pytest.mark.parametrize("command", ["inference", "deploy"])
@pytest.mark.parametrize("task", ["velocity_tracking_flat", "move_forward"])
def test_a_robot_parametric_task_keeps_its_own_robot_flag(task, command):
    """Its ``TaskMod`` registers ``--robot``; re-adding it raised "conflicting option string" at parser build."""
    args = _shared_parser(["--task", task], command=command).parse_args(["--task", task])
    assert hasattr(args, "robot")


@pytest.mark.parametrize("command", ["inference", "deploy"])
@pytest.mark.parametrize("flag", ["--robot", "--entity_name", "--base_command", "--motion_file"])
def test_any_shared_flag_a_mod_owns_is_left_to_the_mod(flag, command):
    parser = argparse.ArgumentParser()
    parser.add_argument(flag, dest="owned", default="mod_default")  # what a TaskMod would have added

    _add_shared_policy_cli_args(parser, command=command)

    assert parser.parse_args([]).owned == "mod_default"


def test_a_task_less_policy_run_still_gets_the_shared_robot_flag():
    assert _shared_parser(["--policy", "holosoma"]).parse_args(["--robot", "H1"]).robot == "H1"
    assert _shared_parser([]).parse_args([]).robot == "G1Dofs29"


# ===========================================================================
# eden train / inference: checkpoint replay, run manifests, the task-less policy env
# ===========================================================================


def _args(**fields) -> argparse.Namespace:
    """What the ``eden inference``/``deploy`` parsers hand these helpers."""
    return argparse.Namespace(
        **{"checkpoint": None, "config": None, "override": None, "run_name": "", "task": None, **fields}
    )


def _run_with_config(tmp_path, task: str = "reacher") -> str:
    """A run dir holding ``task``'s ``config.json`` and a checkpoint; return the checkpoint path."""
    run = tmp_path / f"26Aug15T120000-{task}"
    run.mkdir()
    TASK_REGISTRY.get(task)().save_as_json(str(run / "config.json"))
    (run / "model_100.pt").write_bytes(b"")
    return str(run / "model_100.pt")


def _edit_sibling(checkpoint: str, edit) -> None:
    sibling = pathlib.Path(checkpoint).with_name("config.json")
    data = json.loads(sibling.read_text())
    edit(data)
    sibling.write_text(json.dumps(data))


def _warning_text(caplog) -> str:
    return "\n".join(r.getMessage() for r in caplog.records if r.levelname == "WARNING")


@pytest.fixture
def restore_run_globals(monkeypatch):
    """``en.init`` inside a test rebinds the run globals; put the session's back afterwards."""
    for name in ("_log_dir", "_run_id", "_run_timestamp", "logger"):
        monkeypatch.setattr(en, name, getattr(en, name))


# ---------------------------------------------------------------------------
# A checkpoint replays its own config
# ---------------------------------------------------------------------------


def test_a_checkpoint_replays_its_sibling_config_as_its_own_class(tmp_path):
    loaded = _config_from_checkpoint(_args(checkpoint=_run_with_config(tmp_path)))

    assert type(loaded) is TASK_REGISTRY.get("reacher")


def test_no_sibling_replay_without_a_checkpoint_a_sibling_or_when_a_saved_config_wins(tmp_path):
    """``None`` tells the caller to rebuild from ``--task`` instead."""
    loose = tmp_path / "loose"
    loose.mkdir()
    (loose / "model_1.pt").write_bytes(b"")
    saved = tmp_path / "explicit.json"
    TASK_REGISTRY.get("reacher")().save_as_json(str(saved))

    assert _config_from_checkpoint(_args()) is None
    assert _config_from_checkpoint(_args(checkpoint=str(loose / "model_1.pt"))) is None
    assert _config_from_checkpoint(_args(checkpoint=_run_with_config(tmp_path), config=str(saved))) is None


def test_a_partial_config_and_overrides_layer_onto_the_sibling(tmp_path):
    partial = tmp_path / "partial.yaml"
    partial.write_text("env_options: {num_envs: 3}\n")

    loaded = _config_from_checkpoint(
        _args(
            checkpoint=_run_with_config(tmp_path),
            config=str(partial),
            override=["env_options: {episode_length_s: 2.5}"],
        )
    )

    assert (loaded.env_options.num_envs, loaded.env_options.episode_length_s) == (3, 2.5)


def test_a_replay_resets_the_training_only_env_options_unless_an_override_sets_them_again(tmp_path):
    """A run trained with ``cuda_graph_options`` must still replay on a CPU host, where building with it raises."""
    checkpoint = _run_with_config(tmp_path)
    _edit_sibling(
        checkpoint, lambda data: data["config"]["env_options"].update(cuda_graph_options={"validation_steps": 5})
    )

    assert _config_from_checkpoint(_args(checkpoint=checkpoint)).env_options.cuda_graph_options is None
    kept = _config_from_checkpoint(
        _args(checkpoint=checkpoint, override=["env_options: {cuda_graph_options: {validation_steps: 4}}"])
    )
    assert kept.env_options.cuda_graph_options.validation_steps == 4


def test_task_mod_flags_cannot_apply_to_a_replay_and_say_so(tmp_path):
    with pytest.raises(SystemExit, match="cannot apply to a replay"):
        _config_from_checkpoint(_args(task="reacher", difficulty="hard", checkpoint=_run_with_config(tmp_path)))


@pytest.fixture
def reacher_registered_with_hard_default():
    """``reacher`` registered again under a second name, its ``--difficulty`` defaulting to ``hard``."""
    from eden.tasks.benchmark.reacher.config import ReacherDifficultyMod

    name = "reacher_hard_default"
    TASK_REGISTRY.register(
        TASK_REGISTRY.get("reacher"), name=name, modifiers=(ReacherDifficultyMod(difficulty="hard"),)
    )
    yield name
    TASK_REGISTRY._registry.pop(name, None)
    TASK_REGISTRY._modifiers.pop(name, None)


def test_a_mod_flag_is_explicit_only_away_from_the_registered_default(reacher_registered_with_hard_default):
    """The baseline is the registered mod instance (argparse's default), not the signature default,
    or every unset flag of an overridden mod would abort a plain replay.
    """
    task = reacher_registered_with_hard_default

    assert _explicit_task_mods(_args(task="reacher", difficulty="medium")) == []
    assert _explicit_task_mods(_args(task="reacher", difficulty="hard")) == ["--difficulty"]
    assert _explicit_task_mods(_args(task=task, difficulty="hard")) == []
    assert _explicit_task_mods(_args(task=task, difficulty="medium")) == ["--difficulty"]
    assert _explicit_task_mods(_args()) == []


# ---------------------------------------------------------------------------
# The --task identity check on the checkpoint's config picks logs/<task>/
# ---------------------------------------------------------------------------


def test_the_log_root_is_the_task_when_there_is_nothing_to_replay(tmp_path, eden_caplog):
    loose = tmp_path / "loose"
    loose.mkdir()
    (loose / "model_1.pt").write_bytes(b"")

    assert _replay_log_root(_args(task="reacher"), task_less="x") == "logs/reacher"
    assert (
        _replay_log_root(_args(task="reacher", checkpoint=str(loose / "model_1.pt")), task_less="x") == "logs/reacher"
    )
    assert _replay_log_root(_args(), task_less="logs/deploy") == "logs/deploy"  # an extension-policy-only run
    assert not eden_caplog.records


def test_a_mislabelled_task_follows_the_checkpoints_own_task_with_a_warning(tmp_path, eden_caplog):
    """``--resume latest`` and sibling replay later resolve under ``logs/<task>/``: the durable half of the mistake."""
    checkpoint = _run_with_config(tmp_path, "move_forward")

    assert _replay_log_root(_args(task="reacher", checkpoint=checkpoint), task_less="x") == "logs/move_forward"
    warning = _warning_text(eden_caplog)
    # Role-correct, not merely naming both: a swapped message would pass an `in` check on the names alone.
    assert "belongs to task 'move_forward'" in warning and "not --task 'reacher'" in warning
    assert "logs/move_forward/ rather than logs/reacher/" in warning
    assert type(_config_from_checkpoint(_args(task="reacher", checkpoint=checkpoint))) is TASK_REGISTRY.get(
        "move_forward"
    )


def test_the_log_root_stays_quiet_for_the_same_task_an_alias_or_a_winning_saved_config(
    tmp_path, eden_caplog, task_registry_alias
):
    """One class under two names is one task: a reverse lookup of the saved class would find only the first."""
    task_registry_alias(TASK_REGISTRY.get("reacher"), "reacher_alias")
    own = _run_with_config(tmp_path)
    other = _run_with_config(tmp_path, "move_forward")
    explicit = tmp_path / "explicit.json"
    TASK_REGISTRY.get("reacher")().save_as_json(str(explicit))

    assert _replay_log_root(_args(task="reacher", checkpoint=own), task_less="x") == "logs/reacher"
    assert _replay_log_root(_args(task="reacher_alias", checkpoint=own), task_less="x") == "logs/reacher_alias"
    assert (
        _replay_log_root(_args(task="reacher", checkpoint=other, config=str(explicit)), task_less="x") == "logs/reacher"
    )
    assert not eden_caplog.records


def test_a_saved_class_no_task_owns_warns_and_keeps_the_task(tmp_path, eden_caplog):
    checkpoint = _run_with_config(tmp_path)
    _edit_sibling(checkpoint, lambda data: data["_meta"].update(_option_class_="GoneConfig"))

    assert _replay_log_root(_args(task="reacher", checkpoint=checkpoint), task_less="x") == "logs/reacher"
    assert "GoneConfig" in _warning_text(eden_caplog) and "ReacherConfig" in _warning_text(eden_caplog)


def test_a_save_with_no_class_tag_is_never_warned_about_on_a_guess(tmp_path, eden_caplog):
    checkpoint = _run_with_config(tmp_path)
    _edit_sibling(checkpoint, lambda data: [data["_meta"].pop(k, None) for k in ("_option_module_", "_option_class_")])

    assert _replay_log_root(_args(task="reacher", checkpoint=checkpoint), task_less="x") == "logs/reacher"
    assert not eden_caplog.records


@pytest.mark.parametrize(
    ("command", "task_less"), [("inference_task", "logs/inference"), ("deploy_task", "logs/deploy")]
)
def test_both_commands_route_their_log_root_through_the_identity_check(command, task_less):
    """Read off the source: reaching ``en.init`` at runtime means real argv plus a Genesis init, and a revert
    of either call site to ``f"logs/{args.task}"`` would leave the helper passing its own tests unused.
    """
    tree = ast.parse(pathlib.Path(tasks.__file__).read_text())
    func = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == command)
    call = next(
        n
        for n in ast.walk(func)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "init"
        and isinstance(n.func.value, ast.Name)
        and n.func.value.id == "en"
    )
    log_root = next(kw.value for kw in call.keywords if kw.arg == "log_root_path")

    assert isinstance(log_root, ast.Call) and getattr(log_root.func, "id", None) == "_replay_log_root", ast.dump(
        log_root
    )
    assert [a.id for a in log_root.args if isinstance(a, ast.Name)] == ["args"]
    assert {kw.arg: getattr(kw.value, "value", None) for kw in log_root.keywords} == {"task_less": task_less}


# ---------------------------------------------------------------------------
# _detect_checkpoint_framework: key names only
# ---------------------------------------------------------------------------


def test_the_framework_is_read_off_the_checkpoints_keys(tmp_path):
    layouts = {
        "rsl_rl": {"actor_state_dict": {}, "critic_state_dict": {}},
        "rsl_rl_legacy": {"model_state_dict": {"actor.0.weight": torch.zeros(1)}},
        "rl_games": {"model": {}, "epoch": 3},
        "tensor": torch.zeros(2),
    }
    for name, payload in layouts.items():
        torch.save(payload, tmp_path / f"{name}.pt")

    detected = {name: _detect_checkpoint_framework(str(tmp_path / f"{name}.pt")) for name in layouts}
    assert detected == {"rsl_rl": "rsl_rl", "rsl_rl_legacy": "rsl_rl", "rl_games": "rl_games", "tensor": None}
    assert _detect_checkpoint_framework(str(tmp_path / "missing.pt")) is None


def test_framework_detection_maps_the_checkpoint_rather_than_deserializing_it(tmp_path, monkeypatch):
    """``eden inference`` sniffs before the runner load; a full read would add a copy of the weights."""
    path = str(tmp_path / "model_1.pt")
    torch.save({"actor_state_dict": {"mlp.0.weight": torch.zeros(1024)}, "critic_state_dict": {}}, path)
    mapped: list[bool] = []
    real_load = torch.load

    def recording_load(file, *args, **kwargs):
        if str(file) == path:
            mapped.append(bool(kwargs.get("mmap")))
        return real_load(file, *args, **kwargs)

    monkeypatch.setattr(torch, "load", recording_load)

    assert _detect_checkpoint_framework(path) == "rsl_rl"
    assert mapped and all(mapped), mapped


# ---------------------------------------------------------------------------
# eden train: config.json, run.json, --resume
# ---------------------------------------------------------------------------

_ONE_ITERATION = ["--cpu", "--num_envs", "2", "--override", "runner_options: {max_iterations: 1, num_steps_per_env: 2}"]


@pytest.fixture(scope="module")
def trained_reacher(tmp_path_factory):
    """One real ``eden train --task reacher`` run, in a scratch working directory."""
    cwd = tmp_path_factory.mktemp("train")
    saved = {name: getattr(en, name) for name in ("_log_dir", "_run_id", "_run_timestamp", "logger")}
    previous = os.getcwd()
    os.chdir(cwd)
    try:
        tasks.train_task(["--task", "reacher", "--run_name", "arm-A", *_ONE_ITERATION])
        yield cwd, pathlib.Path(en.get_log_dir()), en.get_run_id(), en.get_run_timestamp()
    finally:
        os.chdir(previous)
        for name, value in saved.items():
            setattr(en, name, value)


def test_train_writes_the_config_and_a_manifest_recording_the_base_seed(trained_reacher):
    cwd, log_dir, run_id, timestamp = trained_reacher

    assert log_dir.parent == cwd / "logs" / "reacher" and "-arm-A-" in log_dir.name
    manifest = json.loads((log_dir / "run.json").read_text())
    config = json.loads((log_dir / "config.json").read_text())["config"]
    assert (manifest["task"], manifest["run_id"], manifest["timestamp"]) == ("reacher", run_id, timestamp)
    assert manifest["seed"] == config["runner_options"]["seed"] == 7100  # the base, as configured
    assert (manifest["world_size"], manifest["run_name"]) == (1, "arm-A")
    assert list(log_dir.glob("model_*.pt"))


def test_resume_forks_a_new_run_from_the_latest_and_records_its_parent(
    trained_reacher, monkeypatch, restore_run_globals
):
    cwd, parent, _, _ = trained_reacher
    monkeypatch.chdir(cwd)
    monkeypatch.setenv("WANDB_MODE", "disabled")
    monkeypatch.delenv("WANDB_RUN_GROUP", raising=False)

    tasks.train_task(
        ["--task", "reacher", "--resume", "latest", "--logger", "wandb", "--wandb_group", "ablate1", *_ONE_ITERATION]
    )

    child = pathlib.Path(en.get_log_dir())
    assert child != parent and child.parent == parent.parent
    assert json.loads((child / "run.json").read_text())["resumed_from"] == parent.name
    assert os.environ["WANDB_RUN_GROUP"] == "ablate1"


def test_resume_and_checkpoint_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        tasks.train_task(["--task", "reacher", "--resume", "latest", "--checkpoint", "x.pt", "--cpu"])


def test_export_writes_the_graph_and_its_spec_beside_the_checkpoint(trained_reacher, monkeypatch, restore_run_globals):
    cwd, log_dir, _, _ = trained_reacher
    monkeypatch.chdir(cwd)
    checkpoint = sorted(log_dir.glob("model_*.pt"))[-1]

    tasks.export_task(["--task", "reacher", "--cpu", "--checkpoint", str(checkpoint), "--format", "jit"])

    out = log_dir / "exported" / checkpoint.stem
    assert sorted(path.name for path in out.iterdir()) == ["policy.pt", "policy_spec.json"]
    spec = json.loads((out / "policy_spec.json").read_text())
    assert (spec["provenance"]["task"], list(spec["model"])) == ("reacher", ["jit"])
    assert spec["actions"]["terms"][0]["dofs_name"] == [f"joint{i}" for i in range(1, 7)]


def test_export_refuses_an_unknown_format_before_building_anything(tmp_path, capsys):
    checkpoint = tmp_path / "model_0.pt"
    checkpoint.touch()

    with pytest.raises(SystemExit):
        tasks.export_task(["--task", "reacher", "--cpu", "--checkpoint", str(checkpoint), "--format", "onnx,tflite"])

    assert "--format 'onnx,tflite' names ['tflite']; choose from onnx, jit" in capsys.readouterr().err


def test_export_names_a_missing_onnxruntime_before_building_anything(tmp_path, capsys, monkeypatch):
    checkpoint = tmp_path / "model_0.pt"
    checkpoint.touch()  # never loaded: the check comes first
    monkeypatch.setitem(sys.modules, "onnxruntime", None)

    with pytest.raises(SystemExit):
        tasks.export_task(["--task", "reacher", "--cpu", "--checkpoint", str(checkpoint), "--format", "onnx"])

    assert "Exporting the ONNX graph needs onnxruntime" in capsys.readouterr().err


def test_a_run_json_write_failure_warns_but_does_not_stop_training(tmp_path, eden_caplog, restore_run_globals):
    """Provenance, not a training input: the unlocked ``config.json`` is the recoverable source of truth."""
    en.init(debug=True, initialize_genesis=False, log_root_path=str(tmp_path))
    log_dir = pathlib.Path(en.get_log_dir())
    log_dir.chmod(stat.S_IRUSR | stat.S_IXUSR)  # the filesystem refuses the write
    try:
        tasks._write_run_manifest(TASK_REGISTRY.get("reacher")(), _args(task="reacher"))
    finally:
        log_dir.chmod(stat.S_IRWXU)

    assert not (log_dir / "run.json").exists()
    assert "run.json" in _warning_text(eden_caplog)


@pytest.mark.requires_genesis
def test_the_deploy_env_is_built_without_reseeding_the_process(tmp_path, restore_run_globals):
    """Deploy mirrors a live robot: no run to reproduce, so no reseeding behind the operator."""
    from eden.envs.wrappers.rsl_rl_env import RslRlVecEnvWrapper

    en.init(debug=True, initialize_genesis=False, log_root_path=str(tmp_path))
    draws = []
    for build in (
        lambda config: tasks._build_deploy_env(config, None, "g1", "robot", show_viewer=False, merged_cmd_options={}),
        lambda config: RslRlVecEnvWrapper.from_config(config, eval_mode=True, apply_seed=False),
        lambda config: RslRlVecEnvWrapper.from_config(config, eval_mode=True),
    ):
        config = TASK_REGISTRY.get("reacher")()
        config.env_options.num_envs = 1
        torch.manual_seed(0)
        build(config)
        draws.append(torch.rand(1).item())

    assert (pathlib.Path(en.get_log_dir()) / "deploy_config.json").exists()
    assert draws[0] == draws[1] != draws[2]  # deploy leaves the RNG as an unseeded build does


# ---------------------------------------------------------------------------
# A --policy with no --task builds its env from the policy's own spec
# ---------------------------------------------------------------------------


def _built_entity_names(scene_options) -> set[str]:
    """Entity names the env build would register, composites expanded as ``EnvBase`` does."""
    from eden.options.entities import CompositeEntityOptions

    names: set[str] = set()
    for key in scene_options.keys():
        if key == "attachments_dict":
            continue
        value = getattr(scene_options, key)
        names.update(value.expand()[0] if isinstance(value, CompositeEntityOptions) else [key])
    return names


@pytest.mark.parametrize(
    ("module", "cls_name", "reads"),
    [
        ("eden.extensions.policies.simtoolreal.policy", "SimToolRealPolicy", {"robot", "hand", "object"}),
        ("eden.extensions.policies.husky.policy", "HuskyPolicy", {"robot", "skateboard"}),
    ],
)
def test_the_scene_carries_every_entity_the_policy_reads(module, cls_name, reads):
    """Only the robot used to be spawned, so a policy reading anything else died one step in."""
    from eden.options.entities import EntityOptionsLike

    cls = getattr(pytest.importorskip(module), cls_name)

    assert reads <= _built_entity_names(_extension_scene_options(cls, cls.robot_options(), "robot"))
    assert all(isinstance(value, EntityOptionsLike) for value in cls.scene_entity_options().values())


def test_a_robot_only_policy_gets_the_robot_on_a_plane():
    from eden.options.entities import PlaneOptions

    cls = pytest.importorskip("eden.extensions.policies.holosoma.policy").HolosomaLocomotionPolicy
    scene = _extension_scene_options(cls, cls.robot_options(), "robot")

    assert cls.scene_entity_options() == {}
    assert _built_entity_names(scene) == {"robot", "plane"} and type(scene.plane) is PlaneOptions


def test_a_policy_may_replace_the_ground_but_not_take_the_robots_slot():
    from eden.options.entities import TerrainOptions

    holosoma = pytest.importorskip("eden.extensions.policies.holosoma.policy").HolosomaLocomotionPolicy
    simtoolreal = pytest.importorskip("eden.extensions.policies.simtoolreal.policy").SimToolRealPolicy

    class _TerrainPolicy(holosoma):
        @classmethod
        def scene_entity_options(cls):
            return {"plane": TerrainOptions()}

    scene = _extension_scene_options(_TerrainPolicy, holosoma.robot_options(), "robot")
    assert type(scene.plane) is TerrainOptions and _built_entity_names(scene) == {"robot", "plane"}
    with pytest.raises(SystemExit, match="robot's own slot"):
        _extension_scene_options(simtoolreal, simtoolreal.robot_options(), "object")


def test_spec_hook_parameters_are_filled_from_the_policys_attributes():
    """A policy that renamed its hand must reach it, or the action term drives one entity while ``step()``
    reads another; ``--entity_name`` still wins, and a non-attribute parameter keeps its default.
    """
    simtoolreal = pytest.importorskip("eden.extensions.policies.simtoolreal.policy").SimToolRealPolicy

    class _RenamedHand(simtoolreal):
        hand_name = "left_hand"

    assert _policy_spec(_RenamedHand, "action_options", "robot")["hand_dofs_pos_controller"].entity_name == "left_hand"
    assert _policy_spec(_RenamedHand, "action_options", "arm")["dofs_pos_controller"].entity_name == "arm"
    assert _policy_spec(_RenamedHand, "scene_entity_options", "robot")["object"].vis_mode == "visual"


def test_command_options_is_the_clis_to_fill_not_the_policys_attributes():
    """Which is why ``beyondmimic_moves`` resolves ``clip_name or cls.clip_name`` inside its own hook."""
    from eden.extensions.policies.base import PolicyBase

    class _ClipPolicy(PolicyBase):
        clip_name = "B_DadDance"

        @classmethod
        def get_parameter_names(cls):
            return ["clip_name"]

        @classmethod
        def command_options(cls, clip_name=None):
            return {"clip_name_seen": clip_name}

        @classmethod
        def observation_options(cls, clip_name=None):
            return {"clip_name_seen": clip_name}

    assert _extension_command_options(_ClipPolicy, "robot", None)["clip_name_seen"] is None
    assert _policy_spec(_ClipPolicy, "observation_options", "robot")["clip_name_seen"] == "B_DadDance"


class _CommandPolicy:
    """A command-driven policy's ``set_command`` and per-env buffer, without the ONNX weights a real one loads."""

    def __init__(self, width: int | None):
        self._commands = None if width is None else torch.zeros(1, width)
        self.received = None

    def set_command(self, command):
        self.received = command
        if self._commands is not None:
            self._commands[:] = command


@pytest.mark.parametrize("width, expected", [(2, [1.0, 0.5]), (3, [1.0, 0.5, -0.2]), (4, [1.0, 0.5, -0.2, 0.0])])
def test_the_base_command_is_resized_to_the_policys_buffer(width, expected):
    """HUSKY takes 2, holosoma 3, HOMIE 4: one ``--base_command`` fits all three."""
    policy = _CommandPolicy(width)

    _set_policy_command(policy, (1.0, 0.5, -0.2))

    assert policy.received.tolist() == pytest.approx(expected)


def test_a_policy_with_no_command_buffer_gets_the_command_and_one_warning(eden_caplog):
    """Legitimate, so non-fatal; the warning is what would have made HUSKY's ``_command`` typo visible."""
    warn_once.cache_clear()
    policy = _CommandPolicy(None)

    _set_policy_command(policy, (1.0, 0.5, -0.2))
    _set_policy_command(_CommandPolicy(None), (1.0, 0.5, -0.2))
    _set_policy_command(object(), (1.0, 0.5, -0.2))  # no set_command at all: left alone

    assert policy.received.tolist() == pytest.approx([1.0, 0.5, -0.2])
    warnings = [r.message for r in eden_caplog.records if "_commands" in r.message]
    assert len(warnings) == 1 and "_CommandPolicy" in warnings[0]


def _husky_policy():
    from eden.extensions.policies.husky.policy import HuskyPolicy

    return HuskyPolicy


def _amo_policy():
    from eden.extensions.policies.amo.policy import AMOPolicy

    return AMOPolicy


@pytest.mark.parametrize(
    "task, policy, expected",
    [
        ("husky", _husky_policy, ["skate"]),  # the task's SkateCommand is reused: it drives the policy
        (None, _husky_policy, []),  # task-less: the policy's own ConstantSkateCommand holds --base_command
        ("reacher", _husky_policy, []),  # no 'skate' on the task: the policy's own term is merged in
        ("husky", _amo_policy, []),  # reads the env but takes no command: --base_command never reached it
        ("husky", lambda: HolosomaLocomotionPolicy, []),  # raw state: set_command feeds its own buffer
        ("husky_holding", _husky_policy, []),  # the task's 'skate' holds a command: set_command reaches it
    ],
)
def test_only_a_reused_task_term_keeps_the_command_from_an_env_reading_policy(task, policy, expected):
    from eden.tasks.husky.custom_terms.commands import ConstantSkateCommand

    cls = policy()
    config = None if task is None else TASK_REGISTRY.build(task.removesuffix("_holding"))
    if task == "husky_holding":
        config.command_options.skate = ConstantSkateCommand.configure()
    cmd_options = _extension_command_options(cls, "robot", None)

    assert _task_terms_driving_policy_command(config, cls, cmd_options) == expected


def test_inference_refuses_a_base_command_an_env_reading_policy_could_not_observe_under_a_task():
    """Under ``--task`` the task's command terms are what such a policy observes; refused before anything is built."""
    with pytest.raises(
        SystemExit,
        match=r"--base_command \(1.0, 0.0, 0.0\) cannot reach HuskyPolicy under --task .*\['skate'\]",
    ):
        tasks._run_inference_policy(
            TASK_REGISTRY.build("husky"),
            "husky",
            show_viewer=False,
            record=False,
            max_steps=1,
            entity_name="robot",
            robot_name="G1Dofs29",
            motion_file=None,
            base_command=(1.0, 0.0, 0.0),
        )


def _registered_policies():
    """Every ``--policy`` the CLI can resolve, imported the way it imports them."""
    import importlib
    import pkgutil

    import eden.extensions.policies as policies_pkg
    from eden.extensions.policies.base import POLICY_REGISTRY

    for module in [found.name for found in pkgutil.iter_modules(policies_pkg.__path__) if found.ispkg]:
        try:
            importlib.import_module(f"eden.extensions.policies.{module}.policy")
        except Exception:  # noqa: BLE001, S110 -- an optional dependency missing is not this test's subject
            pass
    return [(key, POLICY_REGISTRY.get(key)) for key in sorted(POLICY_REGISTRY.keys())]


@pytest.mark.parametrize("key, cls", _registered_policies(), ids=lambda x: x if isinstance(x, str) else "")
def test_a_policy_that_reads_a_command_term_also_declares_it(key, cls):
    """A task-less env has only the command terms the policy declares; reading another is an
    ``AttributeError`` at env build (the Oli loco policy shipped that way).
    """
    if cls.robot_options() is None:
        pytest.skip(f"{cls.__name__} declares no robot; it cannot own a task-less env.")
    read: dict[str, set[str]] = {}
    for group_name, group in _policy_spec(cls, "observation_options", "robot").items():
        for term_key in group.keys():
            term = getattr(group, term_key)
            if getattr(term, "name", None) == "generated_commands":
                read.setdefault(group_name, set()).add((term.params or {}).get("command_name", "command"))
    declared = (
        set(_extension_command_options(cls, "robot", motion_file="unused.npz", clip_name=None)) if read else set()
    )

    missing = {group: names - declared for group, names in read.items() if names - declared}
    assert not missing, f"{cls.__name__}.observation_options reads {missing}, which command_options does not declare"


# ===========================================================================
# eden deploy: policy slots
# ===========================================================================


KNEES = ["left_knee_joint", "right_knee_joint"]
#: G1Dofs29's stock knee angle, which the env below spawns at.
STOCK_KNEE = 0.3


def _config() -> EdenRLConfig:
    """A stock G1 at 50 Hz, spelled ``0.01 x 2`` so it differs from EnvOptions' own ``0.005 x 4`` default."""
    return EdenRLConfig(
        env_options=EnvOptions(num_envs=1, sim_dt=0.01, decimation=2),
        scene_options=SceneOptions(robot=G1Dofs29()),
        action_options=ActionManagerOptions(
            dofs_pos_controller=actions.ImplicitPDController.configure(
                entity_name="robot", dofs_name=G1Dofs29().dofs_name
            )
        ),
        observation_options=ObservationManagerOptions(
            policy=ObservationGroupOptions(
                history_length=3,
                dofs_pos=observations.DofsPos.configure(entity_name="robot"),
            )
        ),
    )


@pytest.fixture(scope="module")
def g1_env():
    return build_env(_config(), reset=True)


@pytest.fixture
def deploy(g1_env):
    """Return ``deploy(**options)``: an in-memory deployment on ``g1_env``, undone after the test."""
    with deployments(g1_env) as make:
        yield lambda **options: make(LoopbackDeployment, **options)


# ---------------------------------------------------------------------------
# policies with a chosen combination of spec hooks
# ---------------------------------------------------------------------------


def _knee_pd(reference_source, scale=None):
    return actions.ImplicitPDController.configure(
        entity_name="robot", dofs_name=KNEES, reference_source=reference_source, scale=scale
    )


class _KneePolicy(PolicyBase):
    """A 50 Hz knee policy emitting a fixed raw ``action``; subclasses change one hook each."""

    rate: ClassVar[dict] = {"sim_dt": 0.005, "decimation": 4}
    knee_pose: ClassVar[float | None] = None
    reference: ClassVar[ReferenceSource] = ReferenceSource.ZERO
    scale: ClassVar[float | None] = None
    obs_groups: ClassVar[dict] = {}
    action = torch.zeros(1, 2)

    @classmethod
    def env_options(cls):
        return dict(cls.rate)

    @classmethod
    def robot_options(cls):
        return None if cls.knee_pose is None else G1Dofs29(default_dofs_pos={"*_knee_joint": cls.knee_pose})

    @classmethod
    def action_options(cls, entity_name: str = "robot"):
        return {"dofs_pos_controller": _knee_pd(cls.reference, cls.scale)}

    @classmethod
    def observation_options(cls, entity_name: str = "robot"):
        return dict(cls.obs_groups)

    def step(self, obs):
        return {"dofs_pos_controller": self.action}


def _policy(name, **attrs):
    return type(name, (_KneePolicy,), attrs)


def _obs_group():
    return {"policy": ObservationGroupOptions(last_action=ObsTerm.configure(func=observations.last_action))}


# ---------------------------------------------------------------------------
# slot order and spec resolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "ext, ckpt, has_base, expected",
    [
        (["E0", "E1"], ["C0", "C1"], False, ["C0", "C1", "E0", "E1"]),
        (["BASE", "E1"], ["C0", "C1"], True, ["BASE", "C0", "C1", "E1"]),  # the base stays slot 1
        (["BASE"], [], True, ["BASE"]),
        (["E0"], [], False, ["E0"]),
        ([], ["C0"], False, ["C0"]),
    ],
)
def test_order_policy_slots(ext, ckpt, has_base, expected):
    assert tasks._order_policy_slots(ext, ckpt, has_base=has_base) == expected


@pytest.mark.parametrize(
    "base, extensions, expected, has_base",
    [
        ("none", ["sonic"], [("extension:sonic", SonicPolicy)], False),
        (
            "holosoma",
            ["sonic"],
            [("holosoma base (loco)", HolosomaLocomotionPolicy), ("extension:sonic", SonicPolicy)],
            True,
        ),
        # Listed again via --policy: one slot, deduped by resolved class, not by name.
        ("holosoma", ["holosoma"], [("extension:holosoma", HolosomaLocomotionPolicy)], False),
    ],
)
def test_resolve_extension_specs(base, extensions, expected, has_base):
    assert tasks._resolve_extension_specs(base, extensions) == (expected, has_base)


@pytest.mark.parametrize(
    "base, config, robot_name, expected",
    [
        ("none", EdenConfig(scene_options=SceneOptions(robot=Go1())), "go1", "none"),
        ("auto", EdenConfig(scene_options=SceneOptions(robot=G1Dofs29())), "g1", "holosoma"),
        ("auto", EdenConfig(scene_options=SceneOptions(robot=Go1())), "go1", "none"),
        ("holosoma", EdenConfig(scene_options=SceneOptions(robot=G1Dofs29())), "g1", "holosoma"),
        ("auto", None, "G1Dofs29", "holosoma"),  # task-less: the --robot class decides
        ("auto", None, "Go1", "none"),
    ],
)
def test_resolve_base_policy_gates_holosoma_on_a_g1(base, config, robot_name, expected):
    assert tasks._resolve_base_policy(base, config, robot_name, "robot") == expected


def test_forcing_holosoma_onto_another_robot_exits():
    with pytest.raises(SystemExit, match="requires a G1Dofs29"):
        tasks._resolve_base_policy("holosoma", EdenConfig(scene_options=SceneOptions(robot=Go1())), "go1", "robot")


# ---------------------------------------------------------------------------
# _env_owning_slot
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "specs, has_base, owner",
    [
        # The bug: slot 1 was the Holosoma base, which publishes no observation groups.
        ([("holosoma base (loco)", HolosomaLocomotionPolicy), ("extension:bm", BeyondMimicMovesPolicy)], False, 1),
        # The requested policy's robot fixes the stance and gains, not the auto-inserted base's.
        ([("holosoma base (loco)", HolosomaLocomotionPolicy), ("extension:sonic", SonicPolicy)], True, 1),
        ([("holosoma base (loco)", HolosomaLocomotionPolicy)], True, 0),
        ([("first", SonicPolicy), ("second", HolosomaLocomotionPolicy)], False, 0),
        # Reading the env outranks being requested: a wrong layout is unusable, wrong gains merely wrong.
        (
            [
                ("holosoma base (loco)", HolosomaLocomotionPolicy),
                ("extension:sonic", SonicPolicy),
                ("extension:bm", BeyondMimicMovesPolicy),
            ],
            True,
            2,
        ),
        ([], False, None),
    ],
)
def test_env_owning_slot(specs, has_base, owner):
    assert tasks._env_owning_slot(specs, config=None, has_base=has_base) == owner


def test_the_same_policy_twice_yields_one_owner():
    """An owner named by class would make both ``--policy sonic`` slots one."""
    specs = [("extension:sonic", SonicPolicy), ("extension:sonic", SonicPolicy)]
    assert tasks._env_owning_slot(specs, config=None, has_base=False) == 0


@pytest.mark.parametrize(
    "specs, config, error",
    [
        ([("a", BeyondMimicMovesPolicy), ("b", BeyondMimicMovesPolicy)], None, "one at a time"),
        ([("bm", BeyondMimicMovesPolicy)], _config(), "without --task"),  # it would read the task's observations
    ],
)
def test_env_owning_slot_refuses_what_one_env_cannot_serve(specs, config, error):
    with pytest.raises(SystemExit, match=error):
        tasks._env_owning_slot(specs, config=config, has_base=False)


# ---------------------------------------------------------------------------
# _check_policy_fits_env
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "attrs",
    [
        {"rate": {"sim_dt": 0.002, "decimation": 10}},  # 50 Hz, spelled differently
        {"knee_pose": STOCK_KNEE, "reference": ReferenceSource.DEFAULT},  # pattern pose == the spawn pose
        {"knee_pose": 0.669, "reference": ReferenceSource.ZERO},  # absolute targets off its own defaults
        {"rate": {}},  # a policy declaring no rate is not held to EnvOptions' default one
    ],
    ids=["same_rate", "same_pose", "absolute_targets", "no_declared_rate"],
)
def test_a_policy_that_fits_the_env_passes(g1_env, attrs):
    tasks._check_policy_fits_env(g1_env, "robot", _policy("Fits", **attrs))


@pytest.mark.parametrize(
    "attrs, error",
    [
        ({"rate": {"sim_dt": 0.002, "decimation": 15}}, "33.3 Hz"),
        ({"knee_pose": 0.669, "reference": ReferenceSource.DEFAULT}, "left_knee_joint"),  # folded into the offset
        ({"knee_pose": 0.669, "reference": ReferenceSource.ACCUMULATE}, "left_knee_joint"),  # integrated from it
        ({"knee_pose": 0.669, "obs_groups": _obs_group()}, "left_knee_joint"),  # dofs_pos_rel subtracts it
    ],
    ids=["rate", "default_reference", "accumulate", "declares_observations"],
)
def test_a_policy_trained_elsewhere_is_refused(g1_env, attrs, error):
    with pytest.raises(SystemExit, match=error) as exc:
        tasks._check_policy_fits_env(g1_env, "robot", _policy("Misfit", **attrs))
    if error == "33.3 Hz":
        assert "50.0 Hz" in str(exc.value)


#: A fixed base with a three-DOF ball joint, which ``dofs_name`` (and so the pose resolver) names once per DOF.
_BALL_MJCF = """<mujoco model="ball"><compiler angle="radian" autolimits="true"/><worldbody>
  <body name="anchor" pos="0 0 0.4"><geom type="box" size="0.04 0.04 0.02" mass="1"/>
    <body name="socket" pos="0 0 0.05"><joint name="ball_joint" type="ball"/>
      <geom type="capsule" size="0.02" fromto="0 0 0 0.08 0 0" mass="0.1"/>
      <body name="spindle" pos="0.08 0 0"><joint name="spin_joint" axis="0 0 1" range="-0.4 0.4"/>
        <geom type="cylinder" size="0.02 0.01" mass="0.1"/></body></body></body>
</worldbody></mujoco>"""


@pytest.fixture(scope="module")
def ball_env(tmp_path_factory):
    from eden.options.entities import EntityOptions

    path = tmp_path_factory.mktemp("ball") / "ball.xml"
    path.write_text(_BALL_MJCF)
    robot = EntityOptions(
        file=str(path), is_fixed_base=True, is_articulated=True, dofs_name=["ball_joint", "spin_joint"]
    )
    return build_env(EdenConfig(env_options=EnvOptions(num_envs=2), scene_options=SceneOptions(robot=robot)))


@pytest.mark.parametrize("spin, fits", [(0.0, True), (0.3, False)], ids=["same_pose", "spin_differs"])
def test_a_ball_joint_in_the_pose_is_checked_dof_by_dof(ball_env, spin, fits):
    """Each of the ball joint's DOFs is compared at its own column, rather than its repeated name looked up again."""
    pose = {"ball_joint": 0.0, "spin_joint": spin}
    policy = _policy(
        "Ball",
        reference=ReferenceSource.DEFAULT,
        robot_options=classmethod(lambda cls: SimpleNamespace(default_dofs_pos=pose)),  # all the check reads
    )
    if fits:
        tasks._check_policy_fits_env(ball_env, "robot", policy)
    else:
        with pytest.raises(SystemExit, match="spin_joint: robot spawns at \\+0.0000, policy trained at \\+0.3000"):
            tasks._check_policy_fits_env(ball_env, "robot", policy)


# ---------------------------------------------------------------------------
# _slot_controllers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "term, error",
    [
        (actions.ImplicitVelocityController.configure(entity_name="robot", dofs_name=KNEES), "not a joint position"),
        (
            actions.VelocityFeedforwardPDController.configure(entity_name="robot", dofs_name=KNEES),
            "feedforward half would be dropped",
        ),
    ],
)
def test_a_controller_a_slot_cannot_carry_is_refused(g1_env, term, error):
    """A slot writes the payload's joint-position field only."""
    policy = _policy("NotPositions", action_options=classmethod(lambda cls, entity_name="robot": {"c": term}))
    with pytest.raises(SystemExit, match=error):
        tasks._slot_controllers(g1_env, "robot", policy)


def test_a_position_controller_is_a_built_private_copy_of_the_policys_term(g1_env):
    controllers = tasks._slot_controllers(g1_env, "robot", _policy("Zero"))
    controller = controllers["dofs_pos_controller"]
    assert controller is not g1_env.action_manager.get_term("dofs_pos_controller")
    assert (controller.dofs_name, controller.action_dim) == (KNEES, 2)
    controller.compute(torch.tensor([[0.25, -0.25]], device=g1_env.device))
    assert controller.processed_action[0].tolist() == pytest.approx([0.25, -0.25])


# ---------------------------------------------------------------------------
# _extension_policy_slot
# ---------------------------------------------------------------------------


def _knee_targets(dep) -> list[float]:
    payload = dep.action_to_payload(None)
    return [float(payload.dofs_pos[dep.dofs_name.index(knee)]) for knee in KNEES]


def test_the_slot_converts_the_raw_action_with_the_policys_own_controller(g1_env, deploy):
    """BeyondMimic Moves emits a normalised action; the robot must receive ``raw * scale + default pose``."""
    policy_cls = _policy(
        "Normalised", reference=ReferenceSource.DEFAULT, scale=0.35, action=torch.tensor([[2.0, -2.0]])
    )
    controllers = tasks._slot_controllers(g1_env, "robot", policy_cls)  # before the swap: it resolves DOF names
    dep = deploy()
    slot = tasks._extension_policy_slot(
        dep, "bm", policy_cls(g1_env, policy_cls.configure()), controllers, (0.0, 0.0, 0.0)
    )
    slot.act(None)
    assert _knee_targets(dep) == pytest.approx([STOCK_KNEE + 0.7, STOCK_KNEE - 0.7], abs=1e-6)


def test_the_env_owner_hands_its_action_to_the_deployment_instead(g1_env, deploy):
    """Its action must reach the env's action manager, or ``last_action`` reads zeros for ever."""
    policy_cls = _policy("Owner", action=torch.tensor([[0.5, 0.5]]))
    controllers = tasks._slot_controllers(g1_env, "robot", policy_cls)
    dep = deploy()
    slot = tasks._extension_policy_slot(
        dep, "bm", policy_cls(g1_env, policy_cls.configure()), controllers, (0.0,) * 3, owns_env=True
    )
    assert slot.act(None)["dofs_pos_controller"].tolist() == [[0.5, 0.5]]
    with pytest.raises(RuntimeError, match="before anything wrote joint targets"):
        dep.action_to_payload(None)  # the owner slot wrote nothing itself


class _ObsReading(_KneePolicy):
    """Reads the observation dict, so it raises on any input the probe cannot supply."""

    reads_env_observations = True

    def step(self, obs):
        return {"dofs_pos_controller": self.action * obs["policy"].sum()}


class _Raising(_KneePolicy):
    def step(self, obs):
        raise RuntimeError("Got invalid dimensions for input: command_obs")


@pytest.fixture
def probe_obs(g1_env):
    return tasks._probe_observations(g1_env)


def _slot(g1_env, deploy, policy, **kwargs):
    controllers = tasks._slot_controllers(g1_env, "robot", type(policy))
    return tasks._extension_policy_slot(deploy(), "bm", policy, controllers, (0.0,) * 3, **kwargs)


def test_an_action_of_the_wrong_width_is_refused_before_the_robot_is_enabled(g1_env, deploy, probe_obs):
    policy = _ObsReading(g1_env, _ObsReading.configure())
    policy.action = torch.zeros(1, 3)  # three values for a two-DOF term
    with pytest.raises(SystemExit, match="expected 2"):
        _slot(g1_env, deploy, policy, owns_env=True, probe_obs=probe_obs)


class _RaisingReader(_Raising):
    reads_env_observations = True


def test_failing_on_the_envs_real_observations_is_refused_at_startup(g1_env, deploy, probe_obs):
    """OmniXtreme against a clip its ONNX model does not take: known before connect, so refused there."""
    with pytest.raises(SystemExit, match="could not step on the observations this env publishes"):
        _slot(g1_env, deploy, _RaisingReader(g1_env, _RaisingReader.configure()), owns_env=True, probe_obs=probe_obs)


def test_a_policy_that_discards_the_dict_is_warned_about_not_refused(g1_env, deploy, probe_obs, eden_caplog):
    """Its real inputs (entity, command manager) are not in their run-time state before connect."""
    assert _slot(g1_env, deploy, _Raising(g1_env, _Raising.configure()), probe_obs=probe_obs) is not None
    assert "Could not pre-validate 'bm' step contract" in eden_caplog.text


@pytest.mark.parametrize("owns_env", [True, False])
def test_an_unprobeable_policy_is_checked_on_its_first_step(g1_env, deploy, owns_env):
    policy = _ObsReading(g1_env, _ObsReading.configure())
    policy.action = torch.zeros(1, 3)
    slot = _slot(g1_env, deploy, policy, owns_env=owns_env, probe_obs=None)
    with pytest.raises(SystemExit, match="expected 2"):
        slot.act({"policy": torch.ones(1, 3)})


def test_a_well_formed_action_passes_and_the_check_does_not_repeat(g1_env, deploy):
    """Widths are structural, and a deploy step is a hard real-time budget."""
    policy = _ObsReading(g1_env, _ObsReading.configure())
    policy.action = torch.full((1, 2), 0.5)
    slot = _slot(g1_env, deploy, policy, owns_env=True, probe_obs=None)
    obs = {"policy": torch.ones(1, 3)}
    assert slot.act(obs)["dofs_pos_controller"][0].tolist() == pytest.approx([1.5, 1.5])
    policy.action = torch.zeros(1, 3)  # wrong now: passes through, as a mid-run shape change always has
    assert slot.act(obs)["dofs_pos_controller"].shape[-1] == 3


# ---------------------------------------------------------------------------
# _probe_observations
# ---------------------------------------------------------------------------


def _set_joints(g1_env, value: float | None) -> None:
    """Every joint to ``value``, or back to the spawn pose for ``None``."""
    robot = g1_env.entities["robot"]
    _, idx = robot.find_named_dofs_idx_local(robot.dofs_name, preserve_order=True)
    pos = robot.default_dofs_pos.clone() if value is None else torch.full((1, len(idx)), value, device=g1_env.device)
    robot.set_dofs_pos(pos, dofs_idx_local=idx)


def test_the_probe_leaves_no_frame_in_the_history_window(g1_env):
    """The first compute backfills the window with the spawn pose; the probe must undo that."""
    g1_env.observation_manager.reset()  # task-less, the probe runs on managers built but never reset
    _set_joints(g1_env, 0.1)
    probe = tasks._probe_observations(g1_env)
    _set_joints(g1_env, 0.2)
    window = g1_env.observation_manager.compute(update_history=True)["policy"]
    try:
        torch.testing.assert_close(probe["policy"], torch.full_like(probe["policy"], 0.1))  # a detached copy
        torch.testing.assert_close(window, torch.full_like(window, 0.2))  # the probe's frame did not outlive it
    finally:
        _set_joints(g1_env, None)
        g1_env.observation_manager.reset()


def test_an_env_publishing_nothing_probes_with_none():
    unbuilt = EnvBase.from_config(EdenConfig(scene_options=SceneOptions(robot=G1Dofs29())), show_viewer=False)
    assert tasks._probe_observations(unbuilt) is None


# ---------------------------------------------------------------------------
# _build_deploy_env
# ---------------------------------------------------------------------------


def test_a_task_less_deploy_without_a_policy_exits_with_the_fix():
    """``--checkpoint`` alone carries a network, not a scene."""
    with pytest.raises(SystemExit) as exc:
        tasks._build_deploy_env(None, None, "Go2", "robot", False, {})
    assert "--policy" in str(exc.value) and "--task" in str(exc.value)


# ---------------------------------------------------------------------------
# _strip_training_randomization: by what a term is, not its key
# ---------------------------------------------------------------------------


def _joint_zero_offset():
    """A joint-zero draw far from the nominal zero offset, under the key beyondmimic gives it."""
    return domain_rand.RandomizeDofsPosOffset.configure(
        entity_name="robot", action_term_name="joint_pos_controller", dofs_pos_range=(0.1, 0.2)
    )


#: The config keys deploy once stripped by name, whatever term they held; each must still be stripped.
_KEYS_DEPLOY_STRIPPED_BY_NAME = frozenset(
    {
        "push_robot",
        "erfi_50",
        "randomize_torque_noise",
        "randomize_motor_strength",
        "randomize_kp_kd_gains",
        "randomize_friction",
        "randomize_com",
        "randomize_mass_shift",
        "randomize_link_mass_scale",
        "randomize_backlash_epsilon",
        "encoder_bias",
        "randomize_startup_dofs_pos_bias",
        "reset_base",
        "reset_dofs",
    }
)


def test_deploy_strips_training_randomization_under_whatever_key_the_task_chose():
    config = TASK_REGISTRY.get("reacher")()
    config.event_options = EventManagerOptions(
        dofs_default_pos=_joint_zero_offset(),
        foot_friction=domain_rand.RandomizeFrictionRatio.configure(entity_name="robot"),
        place_robot=EventTerm.configure(func=en.events.reset_base_state_uniform, params={"entity_name": "robot"}),
        reset_robot=EventTerm.configure(func=en.events.reset_dofs_by_scale, params={"entity_name": "robot"}),
        stiffen_contact=EventTerm.configure(
            func=en.events.set_contact_sol_params,
            mode="startup",
            params={"sol_params": (0.02, 1.0, 0.9, 0.95, 0.001, 0.5, 2.0), "entity_links": {"robot": None}},
        ),
    )

    assert tasks._strip_training_randomization(config) == {
        "dofs_default_pos": "randomize_dofs_pos_offset",
        "foot_friction": "randomize_friction_ratio",
        "place_robot": "reset_base_state_uniform",
        "reset_robot": "reset_dofs_by_scale",
    }
    assert config.event_options.term_keys() == ("stiffen_contact",)


def test_deploy_resets_the_robot_to_its_default_but_still_places_the_objects():
    """Object placement is scene setup: the robosuite cube is placed as every other placement term places."""
    config = TASK_REGISTRY.build("manipulation_lift_box")

    assert tasks._strip_training_randomization(config) == {"reset_robot_dofs": "reset_dofs_by_offset"}
    assert "place_cube" in config.event_options.term_keys()


def test_every_domain_randomization_term_is_stripped_for_deploy():
    """A term added to ``domain_rand`` reaches deploy stripped, or this names it."""
    terms = {
        obj
        for obj in vars(domain_rand).values()
        if isinstance(obj, type)
        and issubclass(obj, domain_rand.EventTerm)
        and obj.__module__ == domain_rand.__name__
        and not obj.__name__.startswith("_")
    }
    assert terms - set(tasks._deploy_disabled_event_classes()) == set()


def test_the_randomized_resets_are_read_off_the_resets_module():
    assert tasks._deploy_disabled_reset_functions() == {
        "reset_base_state_uniform",
        "reset_dofs_by_offset",
        "reset_dofs_by_scale",
        "reset_dofs_by_scale_and_offset",
    }


@pytest.mark.parametrize("task", ["beyondmimic", "general_mimic"])
def test_the_mimic_tasks_deploy_with_no_domain_randomization(task):
    """Both store their joint-zero draw as ``dofs_default_pos``, a key no list of names had."""
    config = TASK_REGISTRY.build(task)

    assert tasks._strip_training_randomization(config) == {
        "base_com_shift": "randomize_com_shift",
        "dofs_default_pos": "randomize_dofs_pos_offset",
        "foot_friction": "randomize_friction_ratio",
        "push_robot": "push_by_velocity",
    }


@pytest.mark.parametrize(
    "task",
    [
        "velocity_tracking_flat",
        "velocity_tracking_terrain",
        "homie",
        "extreme_parkour_teacher",
        "extreme_parkour_student",
        "reorient",
    ],
)
def test_every_key_deploy_stripped_before_is_still_stripped(task):
    config = TASK_REGISTRY.build(task)
    keyed = _KEYS_DEPLOY_STRIPPED_BY_NAME & set(config.event_options.term_keys())

    assert keyed and keyed <= tasks._strip_training_randomization(config).keys()


@pytest.mark.requires_genesis
def test_eden_deploy_strips_the_task_config_before_building(tmp_path, monkeypatch, restore_run_globals):
    monkeypatch.chdir(tmp_path)
    deployed = []
    monkeypatch.setattr(tasks, "_run_deploy", lambda config, **_: deployed.append(config))

    tasks.deploy_task(["--task", "velocity_tracking_flat", "--cpu", "--checkpoint", str(tmp_path / "model_0.pt")])

    (config,) = deployed
    assert not _KEYS_DEPLOY_STRIPPED_BY_NAME & set(config.event_options.term_keys())


@pytest.mark.requires_genesis
def test_the_deploy_env_commands_the_nominal_joint_zero(tmp_path, restore_run_globals):
    en.init(debug=True, initialize_genesis=False, log_root_path=str(tmp_path))
    config = TASK_REGISTRY.get("reacher")()
    config.env_options.num_envs = 1
    config.event_options = EventManagerOptions(dofs_default_pos=_joint_zero_offset())
    tasks._strip_training_randomization(config)

    env = tasks._build_deploy_env(config, None, "g1", "robot", show_viewer=False, merged_cmd_options={})
    try:
        controller = env.unwrapped.action_manager.get_term("joint_pos_controller")
        assert torch.count_nonzero(controller.effective_offset) == 0  # DELTA control: the nominal offset is zero
    finally:
        env.close()


# ---------------------------------------------------------------------------
# each slot's own PD gains
# ---------------------------------------------------------------------------


class _SwitchingDeployment(OperatorLoopbackDeployment):
    """An operator who presses START at boot, switches to slot 2 after the second control step and presses
    Ctrl-C after the fourth.

    Records the command each control step sent in :attr:`stepped`.
    """

    def __init__(self, env, options) -> None:
        super().__init__(env, options)
        self.stepped = []

    def reset(self):
        obs = super().reset()
        self.operator.press(CMD_START)
        return obs

    def step(self, action=None):
        obs = super().step(action)
        self.stepped.append(self.sent[-1])
        if len(self.stepped) == 2:
            self.request_policy_switch(1)
        if len(self.stepped) == 4:
            raise KeyboardInterrupt
        return obs


def _knee_slot(g1_env, dep, label, robot):
    policy_cls = _policy("Knees", action=torch.tensor([[0.1, 0.1]]))
    controllers = tasks._slot_controllers(g1_env, "robot", policy_cls)
    policy = policy_cls(g1_env, policy_cls.configure())
    return tasks._extension_policy_slot(dep, label, policy, controllers, (0.0,) * 3, robot=robot)


def test_switching_from_sonic_to_holosoma_sends_holosomas_gains_on_the_next_command(g1_env):
    """SONIC's hip pitch is a 7520-22, Holosoma's a 7520-14: the safety slot must not run at SONIC's ~2.4x kp."""
    with deployments(g1_env) as make:
        dep = make(_SwitchingDeployment, gain_ramp_steps=0)
        sonic = _knee_slot(g1_env, dep, "extension:sonic", SonicPolicy.robot_options())
        holosoma = _knee_slot(g1_env, dep, "holosoma base (loco)", holosoma_g1_robot())
        sonic_kp, sonic_kd = dep.wire_gains(SonicPolicy.robot_options())
        holosoma_kp, holosoma_kd = dep.wire_gains(holosoma_g1_robot())
        assert not np.allclose(sonic_kp, holosoma_kp), "the two policies were trained at different gains"

        tasks._deploy_control_loop(g1_env, dep, [sonic, holosoma], "robot", False)

    assert dep.sent == dep.stepped, "reset() sends nothing; the robot is first commanded by a control step"
    for payload in dep.stepped[:2]:
        np.testing.assert_array_equal(payload.dofs_kp, sonic_kp)
        np.testing.assert_array_equal(payload.dofs_kd, sonic_kd)
    for payload in dep.stepped[2:]:
        np.testing.assert_array_equal(payload.dofs_kp, holosoma_kp)
        np.testing.assert_array_equal(payload.dofs_kd, holosoma_kd)


class _RestartRecordingDeployment(_SwitchingDeployment):
    """Records each observation :meth:`restart_observations` hands the loop."""

    def __init__(self, env, options) -> None:
        super().__init__(env, options)
        self.restarted = []

    def restart_observations(self):
        obs = super().restart_observations()
        self.restarted.append(obs)
        return obs


def test_the_slot_switched_in_acts_first_on_observations_restarted_after_the_switch(g1_env):
    """Not on the frame the outgoing slot last acted on, with the windows it filled and its last action."""
    with deployments(g1_env) as make:
        dep = make(_RestartRecordingDeployment, gain_ramp_steps=0)
        sonic = _knee_slot(g1_env, dep, "extension:sonic", SonicPolicy.robot_options())
        holosoma = _knee_slot(g1_env, dep, "holosoma base (loco)", holosoma_g1_robot())
        seen = {sonic.label: [], holosoma.label: []}
        for slot in (sonic, holosoma):
            slot.act = lambda obs, act=slot.act, label=slot.label: seen[label].append(obs) or act(obs)

        tasks._deploy_control_loop(g1_env, dep, [sonic, holosoma], "robot", False)

    assert len(dep.restarted) == 1, "one switch, one restart"
    assert seen[holosoma.label][0] is dep.restarted[0]
    assert all(obs is not dep.restarted[0] for obs in seen[sonic.label])


def test_a_pinned_gain_map_that_leaves_out_a_joint_is_refused_before_any_deployment_exists():
    """No deployment is passed: the refusal lands before the entity swap and before a device is opened."""
    partial = G1Dofs29(default_dofs_kp={"*_hip_pitch_joint": 200.0})
    with pytest.raises(SystemExit) as exc:
        tasks._check_pinned_gains([("holosoma", holosoma_g1_robot()), ("extension:mine", partial)], G1Dofs29())
    message = str(exc.value)
    assert message.startswith("Extension policy 'extension:mine' pins a robot")
    assert "default_dofs_kp gives no value for ['left_hip_roll_joint'" in message


def test_every_shipped_kind_of_pinned_robot_passes_the_gain_check():
    """Full maps (SONIC, Holosoma), no robot, and an empty map (G1Dofs23) that runs at the deployed robot's."""
    robots = [
        ("sonic", SonicPolicy.robot_options()),
        ("holosoma", holosoma_g1_robot()),
        ("none", None),
        ("g1_23", G1Dofs23()),
    ]
    tasks._check_pinned_gains(robots, G1Dofs29())


def test_the_gain_check_leaves_an_undeployable_composite_to_the_errors_that_name_it():
    """SimToolReal's KukaSharpa has no dofs_name; the backend lookup says it has no deployment, so must this not preempt it."""
    composite = pytest.importorskip("eden.extensions.policies.simtoolreal.policy").SimToolRealPolicy.robot_options()
    assert getattr(composite, "dofs_name", None) is None
    partial = G1Dofs29(default_dofs_kp={"*_hip_pitch_joint": 200.0})  # would be refused against a real robot
    tasks._check_pinned_gains([("extension:sim_tool_real", composite), ("extension:mine", partial)], composite)


class _FirmwareSwitchingDeployment(_SwitchingDeployment):
    sends_command_gains = False


def test_a_backend_that_cannot_switch_gains_boots_and_switches_on_its_own_gains(g1_env):
    """``set_gains`` raises on such a backend, so the loop must leave the gains alone at boot and on a switch."""
    with deployments(g1_env) as make:
        dep = make(_FirmwareSwitchingDeployment)
        slots = [
            _knee_slot(g1_env, dep, "extension:sonic", SonicPolicy.robot_options()),
            _knee_slot(g1_env, dep, "holosoma base (loco)", holosoma_g1_robot()),
        ]
        robots_kp, _ = dep.wire_gains()

        tasks._deploy_control_loop(g1_env, dep, slots, "robot", False)

    assert len(dep.stepped) == 4, "the loop ran through the switch to the operator's Ctrl-C"
    for payload in dep.sent:
        np.testing.assert_array_equal(payload.dofs_kp, robots_kp)


@pytest.mark.parametrize("options", [{}, {"deploy_kp_kd_scale": 0.5}], ids=["stock", "scaled"])
def test_under_a_task_each_policy_trained_at_other_gains_than_the_task_robot_is_named(g1_env, eden_caplog, options):
    """The task's stock G1 sets every slot's gains: SONIC's 7520-22 hip pitch does not run; Holosoma's matches."""
    pinned = [("holosoma base (loco)", holosoma_g1_robot()), ("extension:sonic", SonicPolicy.robot_options())]
    with deployments(g1_env) as make:
        tasks._warn_on_task_robot_gains(make(LoopbackDeployment, **options), [*pinned, ("pins none", None)])

    warnings = [r.message for r in eden_caplog.records if r.levelname == "WARNING" and "--task robot's" in r.message]
    assert [w.split("'")[1] for w in warnings] == ["extension:sonic", "extension:sonic"]
    assert "--task robot's kp," in warnings[0] and "--task robot's kd," in warnings[1]
    assert all("left_hip_pitch_joint (own" in w and "task " in w and "without --task" in w for w in warnings)


def _gain_slots(dep):
    return [
        tasks._DeploySlot("holosoma base (loco)", None, gains=dep.wire_gains(holosoma_g1_robot())),
        tasks._DeploySlot("extension:sonic", None, gains=dep.wire_gains(SonicPolicy.robot_options())),
    ]


def _warnings(eden_caplog) -> list[str]:
    return [r.message for r in eden_caplog.records if r.levelname == "WARNING" and "than the boot slot" in r.message]


@pytest.mark.parametrize(
    "options",
    [{}, {"deploy_kp_kd_scale": 0.5, "disable_joints": ("left_hip_pitch_joint",)}],
    ids=["stock", "scaled_and_disabled"],
)
def test_a_backend_that_cannot_switch_gains_names_each_slot_that_disagrees_with_the_boot_slot(
    g1_env, eden_caplog, options
):
    """Holosoma boots; SONIC's 7520-22 hip pitch disagrees with it, and the robot can run only one of the two."""
    with deployments(g1_env) as make:
        dep = make(FirmwareGainsLoopbackDeployment, **options)
        tasks._warn_on_unswitchable_slot_gains(dep, _gain_slots(dep))

    warnings = _warnings(eden_caplog)
    assert [w.split(" was trained at another ")[1][:2] for w in warnings] == ["kp", "kd"]
    assert all(w.startswith("[deploy] 'extension:sonic'") and "right_hip_pitch_joint (" in w for w in warnings)
    # --disable_joints zeroes a joint for every slot alike, so it is no disagreement.
    disabled = bool(options)
    assert all(("left_hip_pitch_joint (" in w) is not disabled for w in warnings)


@pytest.mark.parametrize("backend", ["sends_gains", "cannot_switch"])
def test_nothing_to_warn_about_when_gains_switch_or_the_slots_agree(g1_env, eden_caplog, backend):
    with deployments(g1_env) as make:
        if backend == "sends_gains":
            dep = make(LoopbackDeployment)
            slots = _gain_slots(dep)
        else:
            dep = make(FirmwareGainsLoopbackDeployment)
            slots = [_gain_slots(dep)[0], tasks._DeploySlot("checkpoint", None, gains=dep.wire_gains())]
        tasks._warn_on_unswitchable_slot_gains(dep, slots)
    assert _warnings(eden_caplog) == []


@pytest.mark.parametrize(
    "labels, warned",
    [
        (["extension:sonic"], ["extension:sonic"]),
        (["extension:sonic", "holosoma base (loco)"], ["extension:sonic"]),
        (["holosoma base (loco)"], []),
    ],
    ids=["lone_slot", "boot_slot", "agrees"],
)
def test_a_backend_that_reports_its_programmed_gains_names_every_slot_that_will_not_run_at_its_own(
    g1_env, eden_caplog, labels, warned
):
    """The robot runs the stock G1's gains; SONIC's 7520-22 hip pitch does not, whether it boots or runs alone."""
    with deployments(g1_env) as make:
        dep = make(ProgrammedGainsLoopbackDeployment)
        by_label = {slot.label: slot for slot in _gain_slots(dep)}
        tasks._warn_on_unswitchable_slot_gains(dep, [by_label[label] for label in labels])

    warnings = [r.message for r in eden_caplog.records if r.levelname == "WARNING" and " runs — " in r.message]
    assert [w.split("'")[1] for w in warnings] == [label for label in warned for _ in ("kp", "kd")]
    assert all("right_hip_pitch_joint (own" in w and "robot " in w and "gain options to them" in w for w in warnings)


# ===========================================================================
# eden deploy / inference: velocity commands and the real-time loop
# ===========================================================================


ARM_MJCF = """
<mujoco model="arm">
  <worldbody>
    <body name="base" pos="0 0 1.0">
      <geom type="sphere" size="0.05" mass="1.0"/>
      <body name="link1">
        <joint name="j1" type="hinge" axis="0 1 0"/>
        <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.03" mass="0.5"/>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


@COMMAND_TERM_REGISTRY.register()
class GaitVelocityCommand(UniformVelocityCommand):
    """A task's own extension of the velocity command, which a held command would silently replace."""


def _runner_options(obs_groups=None):
    from eden.options.learning.rsl_rl import (
        RslRlActorOptions,
        RslRlCriticOptions,
        RslRlOnPolicyRunnerOptions,
        RslRlPpoAlgorithmOptions,
    )

    return RslRlOnPolicyRunnerOptions(
        actor=RslRlActorOptions(),
        critic=RslRlCriticOptions(),
        algorithm=RslRlPpoAlgorithmOptions(),
        obs_groups=obs_groups or {},
    )


def _velocity(entity_name: str, cls=UniformVelocityCommand, **ranges):
    return cls.configure(
        entity_name=entity_name,
        resampling_time_range=(3.0, 8.0),
        **{"lin_vel_x_range": (-1.0, 1.0), "lin_vel_y_range": (-1.0, 1.0), "ang_vel_z_range": (-0.5, 0.5), **ranges},
    )


@pytest.fixture(scope="module")
def arm_env(tmp_path_factory):
    """Two envs of three one-joint arms: ``robot`` with one velocity command, ``twin`` with two, ``legacy`` with a subclass.

    The policy group reads ``base_lin_vel`` under the key ``velocity``; the critic reads it under its own
    name. Episodes time out after three control steps and publish one metric.
    """
    from eden import observations, terminations
    from eden.envs.wrappers.rsl_rl_env import RslRlVecEnvWrapper
    from eden.managers.terms.metrics.articulation import joint_angle_at_target
    from eden.options import ObservationGroupOptions
    from eden.options.entities import RobotOptions
    from eden.terms import DoneTerm, MetricTerm, ObsTerm
    from eden.utils.configs import (
        ActionManagerOptions,
        CommandManagerOptions,
        EdenRLConfig,
        EnvOptions,
        MetricManagerOptions,
        ObservationManagerOptions,
        SceneOptions,
        TerminationManagerOptions,
    )
    from eden import actions

    asset = tmp_path_factory.mktemp("cli") / "arm.xml"
    asset.write_text(ARM_MJCF)

    def arm(x: float) -> RobotOptions:
        return RobotOptions(
            file=str(asset),
            dofs_name=["j1"],
            default_dofs_pos={"j1": 0.0},
            default_dofs_kp={".*": 10.0},
            default_dofs_kd={".*": 1.0},
            default_root_pos=(x, 0.0, 1.0),
            is_fixed_base=True,
        )

    config = EdenRLConfig(
        env_options=EnvOptions(num_envs=2, episode_length_s=0.06),
        scene_options=SceneOptions(robot=arm(0.0), twin=arm(1.0), legacy=arm(2.0)),
        action_options=ActionManagerOptions(
            joint_pos=actions.ImplicitPDController.configure(entity_name="robot", dofs_name=["j1"])
        ),
        command_options=CommandManagerOptions(
            velocity_command=_velocity("robot"),
            pinned_command=_velocity(
                "twin", lin_vel_x_range=(0.5, 0.5), lin_vel_y_range=(0.0, 0.0), ang_vel_z_range=(0.0, 0.0)
            ),
            asymmetric_command=_velocity("twin", lin_vel_x_range=(-1.0, 3.0)),
            gait_command=_velocity("legacy", cls=GaitVelocityCommand),
        ),
        observation_options=ObservationManagerOptions(
            policy=ObservationGroupOptions(
                velocity=ObsTerm.configure(func=observations.base_lin_vel, params={"entity_name": "robot"}),
                gyro=ObsTerm.configure(func=observations.base_ang_vel, params={"entity_name": "robot"}),
            ),
            critic=ObservationGroupOptions(
                base_lin_vel=ObsTerm.configure(func=observations.base_lin_vel, params={"entity_name": "robot"}),
            ),
        ),
        termination_options=TerminationManagerOptions(
            time_out=DoneTerm.configure(func=terminations.time_out, time_out=True)
        ),
        metric_options=MetricManagerOptions(
            at_rest=MetricTerm.configure(
                func=joint_angle_at_target, params={"entity_name": "robot", "joint_name": "j1", "target_angle": 0.0}
            )
        ),
        runner_options=_runner_options(),
    )
    return RslRlVecEnvWrapper.from_config(config)


# ---------------------------------------------------------------------------
# Which velocity command term the CLI substitutes
# ---------------------------------------------------------------------------


def test_the_exact_velocity_command_is_substituted(arm_env):
    name, term = _resolve_velocity_command_term(arm_env.unwrapped, "robot", "constant")
    assert name == "velocity_command"
    assert term is arm_env.unwrapped.command_manager.get_term("velocity_command")


def test_velocity_command_task_substitutes_nothing_and_trips_no_check(arm_env, eden_caplog):
    """It is the escape hatch both errors recommend, so it cannot raise either of them."""
    for entity_name in ("robot", "twin", "legacy"):
        assert _resolve_velocity_command_term(arm_env.unwrapped, entity_name, "task") == (None, None)
    assert "subclasses" not in eden_caplog.text


def test_a_subclassed_command_is_left_running(arm_env, eden_caplog):
    """A subclass adds behaviour a held command would drop: a subclass is not the class."""
    assert _resolve_velocity_command_term(arm_env.unwrapped, "legacy", "constant") == (None, None)
    assert "subclasses UniformVelocityCommand" in eden_caplog.text
    with pytest.raises(SystemExit, match="cannot substitute"):
        _resolve_velocity_command_term(arm_env.unwrapped, "legacy", "keyboard")


def test_two_velocity_commands_on_one_entity_are_refused(arm_env):
    with pytest.raises(SystemExit, match="driven by 2"):
        _resolve_velocity_command_term(arm_env.unwrapped, "twin", "constant")


def test_substituted_ranges_come_from_the_replaced_term_or_the_scale():
    """No term to read (a task-less ``--policy`` deploy) still yields a bound."""
    assert _velocity_command_ranges(None, ())["lin_vel_x_range"] == (-1.0, 1.0)
    assert _velocity_command_ranges(None, (0.3, 0.2, 0.1))["ang_vel_z_range"] == (-0.1, 0.1)


def test_substituted_ranges_default_to_the_replaced_terms(arm_env):
    term = arm_env.unwrapped.command_manager.get_term("velocity_command")
    assert _velocity_command_ranges(term, ())["ang_vel_z_range"] == (-0.5, 0.5)


# ---------------------------------------------------------------------------
# The ceiling on --base_command
# ---------------------------------------------------------------------------


def _term(arm_env, name):
    return arm_env.unwrapped.command_manager.get_term(name)


def test_the_scale_caps_base_command_on_a_taskless_deploy(eden_caplog):
    """``--base_command`` reaches an extension policy with nothing in between, so the scale is the only bound."""
    assert _cap_base_command((3.0, 0.0, 0.0), None, (0.5, 0.5, 0.5)) == pytest.approx((0.5, 0.0, 0.0))
    assert "exceeds this deploy's per-axis ceiling" in eden_caplog.text


def test_nothing_caps_base_command_when_nothing_declares_a_ceiling(eden_caplog):
    assert _cap_base_command((3.0, 0.0, 0.0), None, ()) == (3.0, 0.0, 0.0)
    assert "ceiling" not in eden_caplog.text


def test_the_replaced_terms_ranges_cap_base_command(arm_env):
    assert _cap_base_command((3.0, 0.0, -2.0), _term(arm_env, "velocity_command"), ()) == pytest.approx(
        (1.0, 0.0, -0.5)
    )


def test_a_pinned_axis_is_a_ceiling_not_a_command(arm_env, eden_caplog):
    """``move_forward`` pins vx=0.5 for its own policy; clamping into that would make "stand" 0.5 m/s for another."""
    pinned = _term(arm_env, "pinned_command")
    assert _cap_base_command((0.0, 0.0, 0.0), pinned, ()) == pytest.approx((0.0, 0.0, 0.0))
    assert "ceiling" not in eden_caplog.text
    assert _cap_base_command((2.0, 0.0, 0.0), pinned, ()) == pytest.approx((0.5, 0.0, 0.0))


def test_an_asymmetric_range_caps_the_magnitude_on_both_signs(arm_env):
    asymmetric = _term(arm_env, "asymmetric_command")
    assert _cap_base_command((-3.0, 0.0, 0.0), asymmetric, ()) == pytest.approx((-3.0, 0.0, 0.0)), "magnitude"
    assert _cap_base_command((-4.0, 0.0, 0.0), asymmetric, ()) == pytest.approx((-3.0, 0.0, 0.0)), "still a ceiling"


def _plan(arm_env, mode, base_command, scale=()):
    return _plan_velocity_command(arm_env, "robot", mode, base_command, scale, "ps4")


def test_the_plan_caps_once_for_every_consumer(arm_env):
    """The extension policies get ``--base_command`` in their constructor, so a later cap reaches nothing."""
    plan = _plan(arm_env.unwrapped, "constant", (3.0, 0.0, -2.0))
    assert plan.base_command == pytest.approx((1.0, 0.0, -0.5))
    assert plan.overrides["velocity_command"].command_value == pytest.approx((1.0, 0.0, -0.5))


def test_the_plan_keeps_the_scale_when_there_is_no_term_to_substitute(arm_env):
    """A task-less ``--policy`` deploy: the scale is the only bound, and it must survive the plan."""
    plan = _plan_velocity_command(arm_env.unwrapped, "nobody", "constant", (3.0, 0.0, 0.0), (0.5, 0.5, 0.5), "ps4")
    assert plan.base_command == pytest.approx((0.5, 0.0, 0.0))
    assert plan.overrides == {} and plan.term_name is None


def test_the_plan_leaves_velocity_command_task_alone(arm_env):
    plan = _plan(arm_env.unwrapped, "task", (3.0, 0.0, 0.0))
    assert plan.base_command == pytest.approx((3.0, 0.0, 0.0))
    assert plan.overrides == {}


def test_every_consumer_reads_the_command_off_the_plan():
    """``_run_deploy`` still holds the raw ``base_command``; a source check, since a real deploy needs a robot."""
    import inspect
    import re

    from eden.utils.cli.tasks import _run_deploy

    source = re.sub(r"\s+", "", inspect.getsource(_run_deploy))
    assert "_construct_extension_policy(env.unwrapped,entity_name,cls,plan.base_command)" in source
    assert "controllers,plan.base_command," in source


# ---------------------------------------------------------------------------
# Refusing a base_lin_vel nothing supplies
# ---------------------------------------------------------------------------


def _wrapped(arm_env, obs_groups):
    from eden.envs.wrappers.rsl_rl_env import RslRlVecEnvWrapper

    return RslRlVecEnvWrapper(arm_env.unwrapped, _runner_options(obs_groups))


@pytest.mark.parametrize(
    ("obs_groups", "checkpoints", "actor"),
    [
        ({}, ["run.pt"], ["policy"]),
        ({"critic": ["critic"]}, ["run.pt"], ["policy"]),
        ({"policy": ["proprio", "cmd"], "critic": ["critic"]}, ["run.pt"], ["proprio", "cmd"]),
        ({}, [], []),
    ],
    ids=["default", "critic-only", "named", "no-checkpoint"],
)
def test_the_actor_is_fed_rsl_rls_policy_groups(arm_env, obs_groups, checkpoints, actor):
    """``obs_groups["policy"]`` when the runner names it, else the env group ``policy``; nothing without a checkpoint."""
    assert _actor_obs_groups(_wrapped(arm_env, obs_groups), checkpoints) == actor


def test_readers_are_matched_on_the_registry_key_in_the_groups_the_actor_reads(arm_env):
    """``velocity`` is a ``base_lin_vel`` term under another name; the critic's copy feeds no actor."""
    from eden.extensions.policies.holosoma.policy import HolosomaLocomotionPolicy
    from eden.extensions.policies.limx_oli_loco.policy import OliVelocityLocoPolicy

    extensions = [("extension:oli", OliVelocityLocoPolicy), ("extension:holosoma", HolosomaLocomotionPolicy)]

    readers = _base_lin_vel_readers(arm_env.unwrapped, extensions, "robot", ["policy"])

    assert readers == [
        "env observation 'policy/velocity'",
        "extension:oli (OliVelocityLocoPolicy) observation 'policy/base_lin_vel'",
    ]
    assert _base_lin_vel_readers(arm_env.unwrapped, [], "robot", ["critic"]) == [
        "env observation 'critic/base_lin_vel'"
    ]
    assert _base_lin_vel_readers(arm_env.unwrapped, [], "robot", []) == []


class _GroundTruthLoopbackDeployment(LoopbackDeployment):
    """Publishes ``base_lin_vel`` every step, as the MuJoCo backend does."""

    def _publishes_base_lin_vel(self) -> bool:
        return True


def test_a_reader_with_no_source_is_refused_naming_the_flag(arm_env):
    with deployments(arm_env.unwrapped) as make:
        blind, informed = make(LoopbackDeployment), make(_GroundTruthLoopbackDeployment)
        _check_base_lin_vel_source(blind, [])  # nothing reads it: nothing to check
        with pytest.raises(SystemExit, match=r"(?s)policy/base_lin_vel.*--estimate_lin_vel"):
            _check_base_lin_vel_source(blind, ["env observation 'policy/base_lin_vel'"])
        _check_base_lin_vel_source(informed, ["env observation 'policy/base_lin_vel'"])


# ---------------------------------------------------------------------------
# The deploy argparse layer (fails before ``en.init()`` and any env build)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["--menagerie"], "--menagerie"),
        (["--mujoco"], "Provide --task"),
        (["--imu_noise", "0.3", "0.05"], "requires --mujoco"),
        (["--imu_bias", "0", "0", "0.1", "0", "0", "0"], "requires --mujoco"),
        (["--imu_noise_seed", "3"], "requires --mujoco"),
        (["--mujoco", "--imu_noise", "-0.1", "0.05"], "--imu_noise takes standard deviations"),
        (["--sim", "mujoco"], "unrecognized arguments"),
        (["--sim_scene", "menagerie"], "unrecognized arguments"),
        (["--async"], "--async"),
        (["--mujoco", "--async"], "Provide --task"),
        (["--prop_noise", "0.003", "0.035"], "--prop_noise acts on a MuJoCo sim-to-sim run"),
        (["--prop_camera", "30", "0.04"], "--prop_camera acts on a MuJoCo sim-to-sim run"),
        (["--prop_noise_seed", "7"], "--prop_noise_seed acts on a MuJoCo sim-to-sim run"),
        (["--mujoco_record", "out.mp4"], "--mujoco_record acts on a MuJoCo sim-to-sim run"),
        (["--mujoco_seconds", "6"], "--mujoco_seconds acts on a MuJoCo sim-to-sim run"),
        (["--mujoco", "--prop_noise", "-0.003", "0.035"], "--prop_noise takes standard deviations"),
        (["--mujoco", "--prop_camera", "0", "0.04"], "--prop_camera takes a frame rate > 0"),
        (["--mujoco", "--prop_camera", "30", "-0.04"], "--prop_camera takes a frame rate > 0"),
        (["--mujoco", "--mujoco_seconds", "0"], "--mujoco_seconds must be > 0"),
        (["--mujoco_sim_dt", "0.005"], "--mujoco_sim_dt sets MuJoCo's timestep; it requires --mujoco"),
        (["--mujoco", "--mujoco_sim_dt", "0"], "--mujoco_sim_dt must be > 0"),
    ],
    ids=[
        "menagerie-needs-mujoco",
        "mujoco-is-a-switch",
        "imu-noise-needs-mujoco",
        "imu-bias-needs-mujoco",
        "imu-seed-needs-mujoco",
        "negative-imu-noise",
        "old-sim-flag",
        "old-sim-scene-flag",
        "async-needs-mujoco",
        "async-is-a-switch",
        "prop-noise-needs-mujoco",
        "prop-camera-needs-mujoco",
        "prop-noise-seed-needs-mujoco",
        "mujoco-record-needs-mujoco",
        "mujoco-seconds-needs-mujoco",
        "negative-prop-noise",
        "zero-camera-rate",
        "negative-camera-latency",
        "zero-mujoco-seconds",
        "mujoco-sim-dt-needs-mujoco",
        "zero-mujoco-sim-dt",
    ],
)
def test_deploy_rejects_a_bad_flag_combination_before_building_anything(capsys, argv, message):
    with pytest.raises(SystemExit):
        deploy_task(argv)
    assert message in capsys.readouterr().err


_RIG_FLAGS = ["--prop_noise", "0.003", "0.035", "--prop_camera", "30", "0.04", "--prop_noise_seed", "7"]


@pytest.mark.requires_genesis
def test_the_mujoco_flags_reach_the_deployment_by_name(tmp_path, monkeypatch, restore_run_globals):
    from eden.extensions.deployment.mujoco_composite import MujocoCompositeDeployment

    monkeypatch.chdir(tmp_path)
    deployed = []
    monkeypatch.setattr(tasks, "_run_deploy", lambda config, **kwargs: deployed.append(kwargs))
    checkpoint = str(tmp_path / "model_0.pt")
    tasks.deploy_task(
        ["--task", "velocity_tracking_flat", "--cpu", "--checkpoint", checkpoint, "--mujoco", *_RIG_FLAGS]
        + ["--mujoco_sim_dt", "0.005", "--mujoco_record", "lift.mp4", "--mujoco_seconds", "6"]
    )

    kwargs = tasks._rig_option_kwargs(deployed[0]["rig_opts"], MujocoCompositeDeployment, "robot")
    assert kwargs == {
        "prop_pos_noise": 0.003,
        "prop_rot_noise": 0.035,
        "prop_camera_hz": 30.0,
        "prop_latency_s": 0.04,
        "prop_noise_seed": 7,
    }
    assert set(kwargs) <= set(MujocoCompositeDeployment.get_parameter_names()), "every one a real option"

    # The timestep, the recording and the run length are either MuJoCo backend's, a single robot's too.
    from eden.extensions.deployment.mujoco import MujocoDeployment

    assert (deployed[0]["mujoco_sim_dt"], deployed[0]["mujoco_record"], deployed[0]["mujoco_seconds"]) == (
        0.005,
        "lift.mp4",
        6.0,
    )
    for deployment_cls in (MujocoDeployment, MujocoCompositeDeployment):
        declared = set(deployment_cls.get_parameter_names())
        assert {"sim_dt", "record_video", "policy_seconds"} <= declared, deployment_cls.__name__


def test_a_composite_rig_flag_on_a_single_robots_mujoco_exits_naming_it():
    from eden.extensions.deployment.mujoco import MujocoDeployment

    rig_opts = {"--prop_noise": {"prop_pos_noise": 0.003}, "--prop_noise_seed": {"prop_noise_seed": None}}
    with pytest.raises(SystemExit, match=r"--prop_noise acts on a composite rig's MuJoCo.*Drop --prop_noise"):
        tasks._rig_option_kwargs(rig_opts, MujocoDeployment, "go2")
    assert tasks._rig_option_kwargs({"--prop_noise_seed": {"prop_noise_seed": None}}, MujocoDeployment, "go2") == {}, (
        "a flag not passed is not a flag given"
    )


# ---------------------------------------------------------------------------
# ``eden inference --async``: the real-time loop
# ---------------------------------------------------------------------------


def _zero_policy(obs):
    return torch.zeros(2, 1)


def test_the_loop_paces_the_sim_to_wall_clock(arm_env):
    start = time.perf_counter()
    before = arm_env.unwrapped.common_step_counter
    _run_sim2sim_loop(arm_env, _zero_policy, max_steps=10)
    assert arm_env.unwrapped.common_step_counter - before == 10
    assert time.perf_counter() - start >= 0.9 * 10 * arm_env.unwrapped.dt


def test_a_slow_policy_leaves_the_sim_repeating_its_latest_action(arm_env):
    calls = []

    def slow_policy(obs):
        calls.append(obs)
        time.sleep(0.05)  # 2.5 control periods
        return torch.zeros(2, 1)

    before = arm_env.unwrapped.common_step_counter
    _run_sim2sim_loop(arm_env, slow_policy, max_steps=20)
    assert arm_env.unwrapped.common_step_counter - before == 20
    assert len(calls) < 20, "the sim kept stepping while inference ran"


def test_a_policy_exception_reaches_the_caller(arm_env):
    def broken_policy(obs):
        raise RuntimeError("policy exploded")

    with pytest.raises(RuntimeError, match="policy exploded"):
        _run_sim2sim_loop(arm_env, broken_policy, max_steps=5)


def test_the_loop_stops_at_the_episode_cap_and_reports_the_metrics(arm_env, capsys):
    """Both envs time out together after three steps: one episode each, which is the cap of 2 over 2 envs."""
    before = arm_env.unwrapped.common_step_counter
    _run_sim2sim_loop(arm_env, _zero_policy, max_episodes=2)
    assert arm_env.unwrapped.common_step_counter - before == 3
    out = capsys.readouterr().out
    assert "averaged over 2 episode(s), the first 1 of each env" in out
    assert "Metrics/at_rest" in out


def test_each_env_counts_its_first_episodes_up_to_the_cap_and_the_rollout_waits_for_the_slowest(arm_env):
    """Env 0 finishes three short episodes while env 1 runs one long one: with a cap of one each, env 0's second and
    third are left out, and the rollout is done only once env 1's has finished. Stopping at two resets over all envs
    would instead have counted env 0 twice and never seen env 1's.
    """
    manager = arm_env.unwrapped.metric_manager
    metrics = _EpisodeMetrics(num_envs=2, episodes_per_env=1)
    for dones in ([True, False], [True, False], [True, True]):
        assert not metrics.done
        mask = torch.tensor(dones, device=manager.device)
        manager.reset(mask)
        metrics.update(mask, manager)
    assert metrics.done
    assert metrics.n_episodes == 2
    assert metrics.finished_per_env.tolist() == [3, 1]
    values, _ = manager.last_episode()
    assert metrics.n_averaged == 2
    assert metrics.sums["Metrics/at_rest"] == pytest.approx(float(values["Metrics/at_rest"].sum()))


def test_a_raw_env_counts_its_timed_out_episodes(arm_env, capsys):
    """A raw RLEnvBase returns terminations and time-outs apart; the loop reads the env's reset mask, so the episodes
    that time out after three steps still end the rollout and reach the means.
    """
    env = arm_env.unwrapped
    before = env.common_step_counter
    _run_sim2sim_loop(env, _zero_policy, max_episodes=2)
    assert env.common_step_counter - before == 3
    assert "averaged over 2 episode(s), the first 1 of each env" in capsys.readouterr().out


def test_a_budget_that_runs_out_before_the_cap_warns(arm_env, eden_caplog):
    _run_sim2sim_loop(arm_env, _zero_policy, max_steps=2, max_episodes=2)
    assert "2 of 2 env(s) fell short" in eden_caplog.text


def test_without_a_cap_every_finished_episode_counts():
    metrics = _EpisodeMetrics(num_envs=2)
    metrics.update(torch.tensor([True, False]), None)
    metrics.update(torch.tensor([True, True]), None)
    metrics.update(torch.tensor([False, False]), None)
    assert metrics.n_episodes == 3
    assert not metrics.done


def test_the_step_budget_is_steps_else_episodes_else_one_episode(arm_env):
    """The episode budget lets every env finish its cap at full length; the loop still stops once they have."""
    length = arm_env.unwrapped.max_episode_length
    assert _resolve_step_budget(arm_env, 42, None) == 42
    assert _resolve_step_budget(arm_env, None, None) == length
    assert _resolve_step_budget(arm_env, None, 3) == math.ceil(3 / 2) * length  # each env's cap, at full length


def test_clone_obs_decouples_the_buffers_it_is_handed():
    obs = torch.zeros(2, 3)
    cloned = _clone_obs(obs)
    obs.fill_(7.0)
    assert torch.all(cloned == 0)

    obs_dict = {"obs": torch.zeros(2), "aux": "passthrough"}
    cloned_dict = _clone_obs(obs_dict)
    obs_dict["obs"].fill_(7.0)
    assert torch.all(cloned_dict["obs"] == 0) and cloned_dict["aux"] == "passthrough"


# ---------------------------------------------------------------------------
# `eden train`'s pre-build runner check and `eden deploy`'s custom-runner refusal
# ---------------------------------------------------------------------------

_needs_rsl_rl_5 = pytest.mark.skipif(
    detect_rsl_rl_version() < MIN_RSL_RL_VERSION, reason="the custom runners and algorithms need rsl-rl-lib >= 5.5.1"
)


@pytest.fixture
def distributed_launch(monkeypatch):
    from eden.utils import distributed

    monkeypatch.setattr(distributed, "is_distributed", lambda: True)


@_needs_rsl_rl_5
def test_the_depth_student_exits_without_a_checkpoint_and_starts_with_one():
    config = TASK_REGISTRY.build("extreme_parkour_student")
    with pytest.raises(SystemExit, match=r"Depth distillation needs a teacher: pass --checkpoint"):
        _check_rsl_rl_runner(config, checkpoint=None)
    _check_rsl_rl_runner(config, checkpoint="teacher/model_100.pt")


@_needs_rsl_rl_5
def test_a_single_gpu_algorithm_exits_under_torchrun_before_the_build(distributed_launch):
    with pytest.raises(SystemExit, match=r"RmaPPO trains on one GPU.*without torchrun"):
        _check_rsl_rl_runner(TASK_REGISTRY.build("extreme_parkour_teacher"), checkpoint=None)


@_needs_rsl_rl_5
def test_a_runner_without_the_ddp_mixin_is_refused_under_torchrun(distributed_launch):
    pytest.importorskip("rsl_rl_contrack")
    with pytest.raises(ValueError, match=r"ContrackOnPolicyRunner does not compose .*DDPRunnerMixin"):
        _check_rsl_rl_runner(TASK_REGISTRY.build("contrack"), checkpoint=None)


@_needs_rsl_rl_5
def test_changing_class_name_alone_is_refused_naming_the_fields_and_the_options_class():
    husky = TASK_REGISTRY.build("husky")
    husky.runner_options.algorithm.class_name = "PPO"
    with pytest.raises(ValueError, match=r"does not accept \['amp_discr_hidden_dims', .*RslRlAmpPpoAlgorithmOptions"):
        _check_rsl_rl_runner(husky, checkpoint=None)

    reacher = TASK_REGISTRY.build("reacher")
    reacher.runner_options.algorithm.class_name = "eden.learning.wrappers.rsl_rl.amp_ppo.ppo:AmpPPO"
    with pytest.raises(ValueError, match=r"requires \['amp_motion_file'\], but RslRlPpoAlgorithmOptions has no such"):
        _check_rsl_rl_runner(reacher, checkpoint=None)


@_needs_rsl_rl_5
def test_a_model_import_path_that_names_nothing_is_refused_before_the_build():
    config = TASK_REGISTRY.build("reacher")
    config.runner_options.actor.class_name = "eden.learning.wrappers.rsl_rl.rma.model:NoSuchModel"
    with pytest.raises(ValueError, match=r"runner_options\.actor\.class_name='.*:NoSuchModel': no such attribute"):
        _check_rsl_rl_runner(config, checkpoint=None)


#: Tasks whose runner imports an optional backend, which the check then (rightly) refuses when it is absent.
_RUNNER_BACKENDS = {"contrack": "rsl_rl_contrack"}


@_needs_rsl_rl_5
@pytest.mark.parametrize("task", TASK_REGISTRY.list_tasks())
def test_every_registered_task_passes_the_pre_build_runner_check(task):
    """The check `eden train` runs before the build accepts every shipped task's own runner and algorithm.

    It mirrors how rsl_rl constructs them, so an algorithm that builds itself differently (HIMPPO's own
    ``construct_algorithm``) must not be refused for fields only that constructor reads — a refusal here means the
    task cannot be trained at all. The config class is checked as declared, without its task mods: a mod may need
    data to apply at all (flashdexretarget's ``--motion``).
    """
    config = TASK_REGISTRY.get(task)()
    if "runner_options" not in type(config).model_fields or tasks._is_rl_games(config):
        pytest.skip(f"{task} has no rsl_rl runner")
    if task in _RUNNER_BACKENDS:
        pytest.importorskip(_RUNNER_BACKENDS[task])
    _check_rsl_rl_runner(config, checkpoint="model_100.pt")  # a distillation's teacher; a resume for the rest


def test_a_flashsac_runner_is_left_to_its_own_factory_not_refused_as_a_bare_name():
    from eden.envs.wrappers.rsl_rl_env import rsl_rl_runner_class
    from eden.learning.wrappers.rsl_rl.flashsac.options import RslRlFlashSacRunnerOptions

    config = TASK_REGISTRY.build("reacher").model_copy(update={"runner_options": RslRlFlashSacRunnerOptions()})
    with pytest.raises(ValueError, match=r"'OffPolicyRunner' is neither a stock runner name"):
        rsl_rl_runner_class(config.runner_options.class_name)  # what the check would hit without the exclusion
    _check_rsl_rl_runner(config, checkpoint=None)  # `eden train` builds it through make_flashsac_runner


def test_deploy_refuses_a_task_whose_runner_is_named_by_import_path():
    with pytest.raises(
        SystemExit, match=r"ContrackConfig trains with runner_options\.class_name='eden\.learning.*eden inference"
    ):
        _check_deployable_runner(TASK_REGISTRY.build("contrack"))
    _check_deployable_runner(TASK_REGISTRY.build("husky"))  # a stock runner (its AMP algorithm replays as PPO)
