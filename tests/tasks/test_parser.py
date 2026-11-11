"""Tests for ``eden.tasks.parser``: TaskMod fields as argparse flags, and the ``--config`` / ``--override`` chain."""

from __future__ import annotations

import argparse
import importlib
import json
from typing import Literal

import pytest
import yaml

from eden.tasks import TASK_REGISTRY
from eden.tasks.parser import (
    _add_mod_field_to_parser,
    _collect_modifier_kwargs,
    get_task_argparser,
    get_task_config_from_args,
)
from eden.tasks.registry import TaskMod


class _LabelMod(TaskMod):
    def __init__(self, label: Literal["a", "b", "c"] = "a", delta: int = 0, items: list[int] = ()):
        super().__init__()
        self.label, self.delta, self.items = label, delta, items

    def apply(self, config):
        return config


# ── TaskMod fields → argparse ───────────────────────────────────────────────


def test_literal_field_becomes_choices():
    parser = argparse.ArgumentParser()
    _add_mod_field_to_parser(parser, _LabelMod(), "label", Literal["a", "b", "c"], help_text="Pick a label.")
    assert parser.parse_args(["--label", "b"]).label == "b"
    assert "Pick a label." in parser.format_help()
    with pytest.raises(SystemExit):
        parser.parse_args(["--label", "z"])


def test_int_field_is_typed_and_defaults_to_the_instance_value():
    parser = argparse.ArgumentParser()
    _add_mod_field_to_parser(parser, _LabelMod(delta=5), "delta", int)
    assert parser.parse_args([]).delta == 5
    assert parser.parse_args(["--delta", "7"]).delta == 7


def test_prefix_renames_the_flag():
    parser = argparse.ArgumentParser()
    _add_mod_field_to_parser(parser, _LabelMod().with_prefix("left"), "label", Literal["a", "b", "c"])
    assert parser.parse_args(["--left_label", "c"]).left_label == "c"


def test_unsupported_annotation_raises():
    with pytest.raises(TypeError, match="unsupported annotation"):
        _add_mod_field_to_parser(argparse.ArgumentParser(), _LabelMod(), "items", list[int])


class _ReservedMod(TaskMod):
    def __init__(self, task: str = "", override: str = ""):
        self.task, self.override = task, override


@pytest.mark.parametrize("field", ["task", "override"])
def test_mod_field_on_a_reserved_flag_raises(field):
    with pytest.raises(ValueError, match="reserved by get_task_argparser"):
        _add_mod_field_to_parser(argparse.ArgumentParser(), _ReservedMod(), field, str)


def test_task_argparser_adds_the_selected_tasks_mod_flags():
    parser = get_task_argparser("test", default_task_name="reacher")
    assert "Reaching difficulty" in parser.format_help()  # the Annotated help text
    args = parser.parse_args(["--task", "reacher", "--difficulty", "hard"])
    assert args.difficulty == "hard"
    args.unrelated = "ignore_me"
    assert _collect_modifier_kwargs(args, "reacher") == {"difficulty": "hard"}


def test_task_argparser_skips_mod_flags_for_an_unknown_task():
    ns = get_task_argparser("test", default_task_name="not_a_real_task").parse_args(["--task", "not_a_real_task"])
    assert ns.task == "not_a_real_task"
    assert not hasattr(ns, "difficulty")


@pytest.mark.parametrize(
    ("theme", "colored"),
    [("human", True), ("agent", False), ("auto", True)],
    ids=["human", "agent-default", "auto-with-force-color"],
)
def test_task_argparser_help_colorizes_per_the_log_theme(monkeypatch, theme, colored):
    """``--help`` follows ``EDEN_LOG_THEME``, not the TTY."""
    monkeypatch.setenv("EDEN_LOG_THEME", theme)
    monkeypatch.setenv("FORCE_COLOR", "1")
    monkeypatch.delenv("NO_COLOR", raising=False)
    formatted = get_task_argparser("test", default_task_name="reacher").format_help()
    assert ("\033[" in formatted) is colored, f"theme={theme}: {formatted[:200]!r}"


# ── --config / --override ───────────────────────────────────────────────────


def _config_from_argv(argv: list[str]):
    """Run ``argv`` through the real parser → config funnel, as the CLI subcommands do."""
    return get_task_config_from_args(get_task_argparser("test", argv=argv).parse_args(argv))


def _saved(tmp_path, task: str, name: str) -> str:
    path = tmp_path / name
    TASK_REGISTRY.build(task).save_as_json(str(path), lock_config=False)
    return str(path)


def test_override_takes_yaml_or_json_repeatably_later_winning():
    cfg = _config_from_argv(
        [
            "--task",
            "reacher",
            "--override",
            '{"env_options": {"num_envs": 64}}',
            "--override",
            "env_options: {num_envs: 128, episode_length_s: 3.0}",
        ]
    )
    assert (cfg.env_options.num_envs, cfg.env_options.episode_length_s) == (128, 3.0)


def test_each_flag_names_the_other_when_handed_its_input():
    """A path in ``--override`` points at ``--config``, and an inline mapping in ``--config`` at ``--override``."""
    with pytest.raises(ValueError, match=r"not a mapping.*use --config"):
        _config_from_argv(["--task", "reacher", "--override", "overrides.json"])
    for inline in ("env_options: {num_envs: 64}", '{"env_options": {"num_envs": 64}}'):
        with pytest.raises(ValueError, match=r"--config takes a .*path.*Pass it to --override"):
            _config_from_argv(["--task", "reacher", "--config", inline])


@pytest.mark.parametrize(
    "name, body, expected",
    [
        ("override.txt", "{}", "Unsupported config file extension"),
        ("list.yaml", "- 1\n- 2\n", "does not hold a config mapping"),
        ("bad.json", "{not json", "Expecting property name"),
    ],
)
def test_unreadable_config_file_keeps_its_own_error(tmp_path, name, body, expected):
    """The ``--override`` redirect fires only on a positive match, never on a real path."""
    path = tmp_path / name
    path.write_text(body)
    with pytest.raises(ValueError, match=expected):
        _config_from_argv(["--task", "reacher", "--config", str(path)])


def test_saved_config_replaces_the_build_and_recovers_its_class(tmp_path):
    reference = TASK_REGISTRY.build("reacher")
    reference.env_options.num_envs = 7
    path = tmp_path / "config.json"
    reference.save_as_json(str(path), lock_config=False)

    cfg = _config_from_argv(["--task", "reacher", "--config", str(path), "--override", "env_options: {num_envs: 11}"])
    assert type(cfg) is type(reference)
    assert cfg.env_options.num_envs == 11  # --override lands on top of --config

    unchanged = _config_from_argv(["--task", "reacher", "--config", str(path), "--difficulty", "medium"])
    assert unchanged.env_options.num_envs == 7  # a mod flag left at its default is not a collision


@pytest.mark.parametrize("suffix", [".yaml", ".yml", ".json"])
def test_mod_then_partial_config_then_override_apply_in_that_order(tmp_path, suffix):
    """``--difficulty hard`` sets 6.0, a partial ``--config`` 5.0, ``--override`` 4.0; file-only keys survive."""
    body = {"env_options": {"episode_length_s": 5.0, "num_envs": 3}}
    partial = tmp_path / f"partial{suffix}"
    partial.write_text(json.dumps(body) if suffix == ".json" else yaml.safe_dump(body))

    cfg = _config_from_argv(
        [
            "--task",
            "reacher",
            "--difficulty",
            "hard",
            "--config",
            str(partial),
            "--override",
            "env_options: {episode_length_s: 4.0}",
        ]
    )
    assert cfg.env_options.episode_length_s == 4.0
    assert cfg.env_options.num_envs == 3
    assert type(cfg).__name__ == "ReacherConfig"


def test_saved_config_with_an_explicit_task_mod_flag_is_a_hard_error(tmp_path):
    with pytest.raises(ValueError, match=r"saved config.*--difficulty"):
        _config_from_argv(
            ["--task", "reacher", "--config", _saved(tmp_path, "reacher", "config.json"), "--difficulty", "hard"]
        )


@pytest.mark.parametrize(
    ("task", "saved_task", "match"),
    [
        ("reacher", "move_forward", "Re-run with --task move_forward"),
        # Task identity is an exact class match, not load_from_file's subclass rule:
        # ManipulationLiftBoxRGBConfig extends ManipulationLiftBoxConfig.
        ("manipulation_lift_box", "manipulation_lift_box_rgb", "Re-run with --task manipulation_lift_box_rgb"),
        ("manipulation_lift_box_rgb", "manipulation_lift_box", "was saved from"),
    ],
)
def test_saved_config_from_another_task_is_a_hard_error(tmp_path, task, saved_task, match):
    with pytest.raises(ValueError, match=match):
        _config_from_argv(["--task", task, "--config", _saved(tmp_path, saved_task, "other.json")])


def test_a_registry_that_cannot_name_the_owner_degrades_the_hint_not_the_identity_error(
    tmp_path, task_package, isolated_task_registry
):
    """A registry failure while looking up the owning task must not replace the mismatch message."""
    clashing = (
        "from eden.tasks import TASK_REGISTRY\nfrom eden.utils.configs import EdenConfig\n"
        "@TASK_REGISTRY.register(name='dup')\nclass {}(EdenConfig):\n    pass\n"
    )
    task_package("broken_tasks_pkg", {"a.py": clashing.format("AConfig"), "b.py": clashing.format("BConfig")})
    isolated_task_registry.add_search_path("broken_tasks_pkg")
    saved = tmp_path / "clashing.json"
    importlib.import_module("broken_tasks_pkg.a").AConfig().save_as_json(str(saved), lock_config=False)
    # The owner lookup the hint makes raises: the saved class is one of the two registering 'dup'.
    with pytest.warns(UserWarning), pytest.raises(ValueError, match="'dup' is registered in 2 locations"):
        isolated_task_registry.task_for_config_qualname("broken_tasks_pkg.a.AConfig")

    with pytest.raises(ValueError, match=r"was saved from.*No registered task builds it"):
        _config_from_argv(["--task", "reacher", "--config", str(saved)])
