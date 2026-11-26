"""Tests for ``eden.utils.asset_audit``: each check against a small MJCF built for real.

Every asset is written to a temporary directory and audited once per module (a build costs about a second on the
CPU); the tests then read the report. Each defect is planted on purpose, so a check that stops seeing it fails here.
"""

import json

import pytest

from eden.options.entities import EntityOptions
from eden.utils.asset_audit import (
    AuditReport,
    AuditThresholds,
    audit_entity,
    mjcf_has_free_root,
    options_for_file,
)

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

# Upright two-link arm on a fixed base: authored armature, comparable masses, room to move.
CLEAN_ARM = """
<mujoco model="clean_arm">
  <worldbody>
    <body name="base" pos="0 0 0.1">
      <geom type="box" size="0.05 0.05 0.05" mass="2"/>
      <body name="upper" pos="0 0 0.1">
        <joint name="shoulder" type="hinge" axis="0 1 0" range="-1.5 1.5" armature="0.01"/>
        <geom type="capsule" fromto="0 0 0.05 0 0 0.3" size="0.03" mass="1"/>
        <body name="fore" pos="0 0 0.35">
          <joint name="elbow" type="hinge" axis="0 1 0" range="-1.5 1.5" armature="0.01"/>
          <geom type="capsule" fromto="0 0 0.05 0 0 0.3" size="0.02" mass="0.5"/>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""

# The same arm with planted defects: no armature anywhere, a 200x mass ratio at the elbow and a 5000x one at the
# wrist, a 1e-6 kg tip, and a forearm whose geom reaches back down into the base box (base and forearm are not
# adjacent, so they collide; the 6 cm offset keeps it clear of the upper arm).
SLOPPY_ARM = """
<mujoco model="sloppy_arm">
  <worldbody>
    <body name="base" pos="0 0 0.1">
      <geom type="box" size="0.05 0.05 0.05" mass="2"/>
      <body name="upper" pos="0 0 0.1">
        <joint name="shoulder" type="hinge" axis="0 1 0" range="-1.5 1.5"/>
        <geom type="capsule" fromto="0 0 0.05 0 0 0.3" size="0.03" mass="1"/>
        <body name="fore" pos="0 0 0.35">
          <joint name="elbow" type="hinge" axis="0 1 0" range="-1.5 1.5"/>
          <geom type="capsule" fromto="0.06 0 0 0.06 0 -0.42" size="0.02" mass="0.005"/>
          <body name="tip" pos="0 0 0.3">
            <joint name="wrist" type="hinge" axis="0 1 0" range="-1 1"/>
            <geom type="sphere" size="0.01" mass="1e-6"/>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
</mujoco>
"""


# Clean arm whose forearm is long enough to reach the base box once the elbow folds back (3.0 rad). At the zero
# pose the forearm points straight up and touches nothing, so Genesis keeps the base/forearm pair; at the default
# pose it collides.
FOLDED_ARM = CLEAN_ARM.replace(
    'range="-1.5 1.5" armature="0.01"/>\n          <geom type="capsule" fromto="0 0 0.05 0 0 0.3"',
    'range="-3.1 3.1" armature="0.01"/>\n          <geom type="capsule" fromto="0 0 0.05 0 0 0.45"',
)


def _write(tmp_path_factory, name: str, text: str) -> str:
    path = tmp_path_factory.mktemp("assets") / f"{name}.xml"
    path.write_text(text)
    return str(path)


@pytest.fixture(scope="module")
def free_box_path(tmp_path_factory) -> str:
    return _write(tmp_path_factory, "box", FREE_BOX)


@pytest.fixture(scope="module")
def clean_arm_path(tmp_path_factory) -> str:
    return _write(tmp_path_factory, "clean_arm", CLEAN_ARM)


@pytest.fixture(scope="module")
def free_box_report(free_box_path) -> AuditReport:
    return audit_entity(options_for_file(free_box_path))


@pytest.fixture(scope="module")
def clean_arm_report(clean_arm_path) -> AuditReport:
    options = options_for_file(clean_arm_path).model_copy(
        update={
            "dofs_name": ["shoulder", "elbow"],
            "default_dofs_kp": {".*": 300.0},
            "default_dofs_kd": {".*": 20.0},
        }
    )
    return audit_entity(options)


@pytest.fixture(scope="module")
def sloppy_arm_report(tmp_path_factory) -> AuditReport:
    return audit_entity(options_for_file(_write(tmp_path_factory, "sloppy_arm", SLOPPY_ARM)))


class TestFreeBox:
    def test_drops_settles_and_passes(self, free_box_report):
        settle = free_box_report.check("settle")
        assert settle.status == "pass", settle.message
        assert settle.details["max_penetration"] < AuditThresholds().max_penetration
        assert not free_box_report.failed

    def test_skips_what_a_single_free_body_cannot_have(self, free_box_report):
        for name in ("mass_ratio", "armature", "joint_limits", "self_contact", "hold"):
            assert free_box_report.check(name).status == "skip", name

    def test_without_collision_it_fails_and_skips_the_drop(self, free_box_path):
        report = audit_entity(options_for_file(free_box_path).model_copy(update={"collision": False}))
        assert report.check("collision").status == "fail"
        assert report.check("settle").status == "skip"
        assert report.check("hold").status == "skip"
        assert report.failed


class TestCleanArm:
    def test_every_static_check_passes(self, clean_arm_report):
        for name in ("collision", "mass", "mass_ratio", "armature", "joint_limits", "self_contact"):
            check = clean_arm_report.check(name)
            assert check.status == "pass", f"{name}: {check.message}"

    def test_holds_its_default_pose_with_its_gains(self, clean_arm_report):
        hold = clean_arm_report.check("hold")
        assert hold.status == "pass", hold.message
        assert set(hold.details["max_drift"]) == {"shoulder", "elbow"}
        assert clean_arm_report.check("settle").status == "skip"

    def test_without_gains_hold_is_skipped_with_the_fix(self, clean_arm_path):
        hold = audit_entity(options_for_file(clean_arm_path)).check("hold")
        assert hold.status == "skip"
        assert "default_dofs_kp" in hold.message


class TestSloppyArm:
    def test_names_every_dof_genesis_filled_armature_for(self, sloppy_arm_report):
        armature = sloppy_arm_report.check("armature")
        assert armature.status == "warn"
        assert armature.details["filled"] == ["shoulder", "elbow", "wrist"]
        assert "default_dofs_armature" in armature.message

    def test_flags_the_light_tip_and_the_mass_ratio(self, sloppy_arm_report):
        mass = sloppy_arm_report.check("mass")
        assert mass.status == "warn" and "tip" in mass.message
        ratio = sloppy_arm_report.check("mass_ratio")
        assert ratio.status == "warn"
        assert ratio.details["worst_pair"] == ["fore", "tip"]
        assert "upper/fore" in ratio.message

    def test_names_the_pair_genesis_dropped_for_overlapping_at_zero(self, sloppy_arm_report):
        """The forearm overlaps the base at the zero pose, so Genesis never collides them; only the probe sees it."""
        contact = sloppy_arm_report.check("self_contact")
        assert contact.details["pairs"] == []
        assert contact.details["dropped_pairs"] == [["base", "fore"]]
        assert "base/fore" in contact.message and "enable_neutral_collision" in contact.message

    def test_warnings_do_not_fail_the_report(self, sloppy_arm_report):
        assert not sloppy_arm_report.failed


def test_armature_set_in_the_options_is_not_reported_as_filled(tmp_path):
    path = tmp_path / "sloppy_arm.xml"
    path.write_text(SLOPPY_ARM)
    options = options_for_file(str(path)).model_copy(
        update={"dofs_name": ["shoulder", "elbow", "wrist"], "default_dofs_armature": {"shoulder|elbow": 0.1}}
    )
    armature = audit_entity(options).check("armature")
    assert armature.details["filled"] == ["wrist"]


def test_report_serializes_and_formats(free_box_report):
    data = json.loads(json.dumps(free_box_report.to_dict()))
    assert data["failed"] is False
    assert [c["name"] for c in data["checks"]] == [c.name for c in free_box_report.checks]
    text = free_box_report.format()
    assert text.splitlines()[0].startswith("Audit of ")
    assert "PASS  settle" in text


def test_unknown_check_name_lists_the_checks(free_box_report):
    with pytest.raises(KeyError, match="settle"):
        free_box_report.check("bogus")


class TestOptionsForFile:
    def test_mjcf_base_follows_the_file(self, free_box_path, clean_arm_path):
        assert mjcf_has_free_root(free_box_path)
        assert not mjcf_has_free_root(clean_arm_path)
        assert options_for_file(free_box_path).is_fixed_base is False
        assert options_for_file(clean_arm_path).is_fixed_base is True

    def test_explicit_base_wins(self, clean_arm_path):
        assert options_for_file(clean_arm_path, fixed_base=False).is_fixed_base is False

    def test_type_free_joint_counts_as_free(self, tmp_path):
        path = tmp_path / "typed.xml"
        path.write_text(FREE_BOX.replace("<freejoint/>", '<joint name="root" type="free"/>'))
        assert mjcf_has_free_root(str(path))

    def test_missing_file_raises_with_the_path(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="nope.urdf"):
            options_for_file(str(tmp_path / "nope.urdf"))

    def test_unknown_suffix_lists_the_known_ones(self, tmp_path):
        path = tmp_path / "notes.txt"
        path.write_text("")
        with pytest.raises(ValueError, match=r"\.urdf"):
            options_for_file(str(path))

    def test_returns_entity_options_with_an_absolute_path(self, free_box_path):
        options = options_for_file(free_box_path)
        assert isinstance(options, EntityOptions)
        assert options.file == free_box_path


def test_folded_default_pose_collides_with_the_base(tmp_path):
    assert FOLDED_ARM != CLEAN_ARM
    path = tmp_path / "folded_arm.xml"
    path.write_text(FOLDED_ARM)
    options = options_for_file(str(path)).model_copy(
        update={"dofs_name": ["shoulder", "elbow"], "default_dofs_pos": {"elbow": 3.0}}
    )
    contact = audit_entity(options).check("self_contact")
    assert contact.status == "warn"
    assert ["base", "fore"] in contact.details["pairs"]
    assert contact.details["dropped_pairs"] == []
