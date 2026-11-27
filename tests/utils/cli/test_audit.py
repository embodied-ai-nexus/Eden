"""Tests for ``eden.utils.cli.audit``: how ``eden audit`` picks the entity and its physics, and the command end to end.

The checks themselves are tested in ``tests/utils/test_asset_audit.py``; here the command runs in-process through
``eden.__main__.main`` on a small MJCF.
"""

import json
import sys

import pytest

import eden.__main__ as m
from eden.options.entities import EntityOptions
from eden.utils.cli.audit import _resolve_audit_target

FREE_BOX = """
<mujoco model="box">
  <worldbody>
    <body name="box" pos="0 0 0.5">
      <freejoint/>
      <geom type="box" size="0.05 0.04 0.03" mass="0.2"/>
    </body>
  </worldbody>
</mujoco>
"""


def _run_main(monkeypatch, *argv):
    """Run ``eden <argv>`` in-process and return the ``SystemExit`` code, or ``None`` if it returned."""
    monkeypatch.setattr(sys, "argv", ["eden", *argv])
    try:
        m.main()
    except SystemExit as exit_:
        return exit_.code
    return None


class TestResolveTarget:
    def test_a_path_with_an_asset_suffix_loads_as_a_file(self, tmp_path):
        path = tmp_path / "box.xml"
        path.write_text(FREE_BOX)
        options = _resolve_audit_target(str(path), None)
        assert type(options) is EntityOptions
        assert options.is_fixed_base is False

    def test_an_existing_file_with_another_suffix_is_refused_as_a_file(self, tmp_path):
        """Not looked up as an entity name, which would answer with unrelated suggestions."""
        path = tmp_path / "model.sdf"
        path.write_text("")
        with pytest.raises(ValueError, match="not an asset file Eden loads"):
            _resolve_audit_target(str(path), None)

    def test_a_missing_file_is_not_looked_up_as_a_name(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="arm.urdf"):
            _resolve_audit_target(str(tmp_path / "arm.urdf"), None)

    def test_an_entity_name_resolves_to_its_options_class(self):
        options = _resolve_audit_target("ur5e", None)
        assert type(options).__name__ == "UR5e"
        assert _resolve_audit_target("ur5e", True).is_fixed_base is True

    def test_a_urdf_preset_can_be_freed(self):
        assert _resolve_audit_target("ur5e", False).is_fixed_base is False

    def test_an_mjcf_that_declares_its_base_refuses_the_flag(self):
        with pytest.raises(ValueError, match=r"--fixed-base cannot change 'g1_dofs29'.*Drop --fixed-base"):
            _resolve_audit_target("g1_dofs29", True)

    def test_a_composite_rig_names_its_parts(self):
        with pytest.raises(TypeError, match=r"composite rig.*'UR5e', 'Robotiq2f85'"):
            _resolve_audit_target("ur5e_2f85", None)


def test_eden_audit_prints_the_report_and_writes_json(monkeypatch, tmp_path, capsys):
    path = tmp_path / "box.xml"
    path.write_text(FREE_BOX)
    out = tmp_path / "report.json"

    assert _run_main(monkeypatch, "audit", str(path), "--seconds", "1.0", "--json", str(out)) is None

    printed = capsys.readouterr().out
    assert f"Audit of {path}" in printed
    assert "PASS  settle" in printed
    report = json.loads(out.read_text())
    assert report["failed"] is False
    assert report["target"] == str(path)
    assert {check["name"] for check in report["checks"]} >= {"collision", "mass", "settle", "hold"}


def test_eden_audit_rejects_both_base_flags(monkeypatch, tmp_path):
    path = tmp_path / "box.xml"
    path.write_text(FREE_BOX)
    assert _run_main(monkeypatch, "audit", str(path), "--fixed-base", "--free-base") == 2


def test_eden_audit_exits_1_when_a_check_fails(monkeypatch, tmp_path, capsys):
    """A box whose only geom collides with nothing has no collision geometry, which fails ``collision``."""
    path = tmp_path / "ghost.xml"
    path.write_text(FREE_BOX.replace('mass="0.2"/>', 'mass="0.2" contype="0" conaffinity="0"/>'))

    assert _run_main(monkeypatch, "audit", str(path)) == 1
    assert "FAIL  collision" in capsys.readouterr().out


def test_eden_audit_task_runs_at_the_tasks_physics(monkeypatch, tmp_path):
    """``reacher`` steps at sim_dt=0.01 with two substeps, not EnvOptions' 0.005 and one."""
    path = tmp_path / "box.xml"
    path.write_text(FREE_BOX)
    out = tmp_path / "report.json"

    assert (
        _run_main(monkeypatch, "audit", str(path), "--task", "reacher", "--seconds", "1.0", "--json", str(out)) is None
    )

    physics = json.loads(out.read_text())["physics"]
    assert physics["sim_dt"] == 0.01
    assert physics["sim_substeps"] == 2


@pytest.mark.parametrize("seconds", ["-1", "0", "0.25", "nan"])
def test_eden_audit_refuses_a_run_too_short_to_grade_as_a_usage_error(monkeypatch, tmp_path, capsys, seconds):
    """``--seconds -1`` used to simulate one step and report the box, still in the air, as at rest."""
    path = tmp_path / "box.xml"
    path.write_text(FREE_BOX)
    assert _run_main(monkeypatch, "audit", str(path), "--seconds", seconds) == 2
    assert "AuditThresholds:" in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv, message",
    [
        ([], "name an entity or asset file to audit, or pass --task"),
        (["--task", "reacherr"], "reacher"),
        (["ur5e", "--entity", "robot"], "--entity picks a scene entity of --task"),
        (["--task", "reacher", "--fixed-base"], "--fixed-base / --free-base apply to a target"),
        (["--task", "reacher", "--entity", "nope"], "has no scene entity 'nope'; its entities are"),
        (["ur5e_2f85"], "composite rig"),
        (["NoSuchEntityAnywhere"], "No entity options found"),
        (["ur5e", "--json", "/no/such/dir/report.json"], "does not exist"),
    ],
)
def test_eden_audit_reports_argument_errors_as_usage_errors(monkeypatch, capsys, argv, message):
    """Every one is refused by argparse (exit 2) before Eden initializes, naming the problem."""
    assert _run_main(monkeypatch, "audit", *argv) == 2
    assert message in capsys.readouterr().err


def test_eden_audit_takes_the_shared_task_flags(monkeypatch, tmp_path):
    """``--override`` reaches the task's env_options exactly as it does for ``eden smoke``."""
    path = tmp_path / "box.xml"
    path.write_text(FREE_BOX)
    out = tmp_path / "report.json"

    argv = ["audit", str(path), "--task", "reacher", "--override", "env_options: {sim_dt: 0.004}", "--seconds", "1.0"]
    assert _run_main(monkeypatch, *argv, "--json", str(out)) is None

    assert json.loads(out.read_text())["physics"]["sim_dt"] == 0.004


def test_eden_audit_task_alone_audits_the_tasks_robot_at_its_physics(monkeypatch, tmp_path):
    """No target: the scene entity and the physics both come from the config reacher builds."""
    out = tmp_path / "report.json"

    assert _run_main(monkeypatch, "audit", "--task", "reacher", "--cpu", "--json", str(out)) is None

    report = json.loads(out.read_text())
    assert report["target"] == "reacher:robot"
    assert report["physics"] == {"sim_dt": 0.01, "sim_substeps": 2, "integrator": "approximate_implicitfast"}
    assert {c["name"]: c["status"] for c in report["checks"]}["hold"] in ("pass", "warn")


class TestWeldedTaskEntities:
    """A part a task welds to an arm declares a free base for its weld, which used to crash the audit."""

    def test_welds_come_from_rigs_and_from_attachments_dict(self):
        from eden.tasks import TASK_REGISTRY
        from eden.utils.cli.audit import _welds

        rig_task = TASK_REGISTRY.build("robosuite_lift", robot="ur5e_2f85")
        assert _welds(rig_task.scene_options) == {"robot_gripper": "robot"}
        attached = TASK_REGISTRY.build("manipulation_lift_box")
        assert _welds(attached.scene_options) == {"gripper": "robot"}

    @pytest.mark.parametrize(
        "task, entity, kind, parent",
        [
            ("robosuite_lift", "robot_gripper", "FrankaHand", "robot"),  # plain MJCF without a free joint
            ("manipulation_dex_lift", "hand", "XHand1_L", "robot"),  # URDF
            ("robosuite_two_arm_peg_in_hole", "peg", "CylinderOptions", "robot0"),  # primitive
            ("robosuite_two_arm_peg_in_hole", "hole", "RobosuitePlateWithHole", "robot1"),  # MJCF following the flag
        ],
    )
    def test_a_welded_part_gets_back_a_fixed_base(self, task, entity, kind, parent):
        from eden.tasks import TASK_REGISTRY
        from eden.utils.cli.audit import _task_entity_options

        config = TASK_REGISTRY.build(task)
        declared = config.scene_options.named_entities()[entity]
        assert type(declared).__name__ == kind
        assert declared.is_fixed_base is False

        options, label = _task_entity_options(config, task, entity)
        assert options.is_fixed_base is True
        assert getattr(options, "fixed", True) is True
        assert label == f"{task}:{entity} (welded to {parent}; audited alone with a fixed base)"
        assert declared.is_fixed_base is False  # the task's config is not modified

    def test_an_unwelded_entity_is_audited_as_declared(self):
        from eden.tasks import TASK_REGISTRY
        from eden.utils.cli.audit import _task_entity_options

        config = TASK_REGISTRY.build("reacher")
        options, label = _task_entity_options(config, "reacher", "robot")
        assert options is config.scene_options.named_entities()["robot"]
        assert label == "reacher:robot"

    @pytest.mark.parametrize(
        "argv",
        [
            ["--task", "robosuite_lift", "--robot", "ur5e_2f85", "--entity", "robot_gripper"],  # the documented example
            ["--task", "manipulation_lift_box", "--entity", "gripper"],  # attachments_dict
        ],
    )
    def test_eden_audit_runs_on_a_welded_part(self, monkeypatch, tmp_path, argv):
        out = tmp_path / "report.json"

        assert _run_main(monkeypatch, "audit", *argv, "--cpu", "--json", str(out)) is None

        report = json.loads(out.read_text())
        assert "audited alone with a fixed base" in report["target"]
        statuses = {c["name"]: c["status"] for c in report["checks"]}
        assert statuses["settle"] == "skip"
        assert statuses["hold"] in ("pass", "warn", "skip")
        assert not report["failed"]
