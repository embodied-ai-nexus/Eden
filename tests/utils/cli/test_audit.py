"""Tests for ``eden.utils.cli.audit``: how ``eden audit`` resolves its target, and the command end to end.

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
        _run_main(monkeypatch, "audit", str(path), "--task", "reacher", "--seconds", "0.5", "--json", str(out)) is None
    )

    physics = json.loads(out.read_text())["physics"]
    assert physics["sim_dt"] == 0.01
    assert physics["sim_substeps"] == 2
