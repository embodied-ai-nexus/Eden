"""Tests for ``eden.utils.asset_audit``: each check against a small MJCF built for real.

Every asset is written to a temporary directory and audited once per module (a build costs about a second on the
CPU); the tests then read the report. Each defect is planted on purpose, so a check that stops seeing it fails here.
"""

import json
from typing import ClassVar

import pytest

from eden.options import EnvOptions
from eden.options.actuators import ActuatorSpecOptions
from eden.options.entities import MESH_FILE_SUFFIXES, TET_MESH_FILE_SUFFIXES, EntityOptions
from eden.options.robots import ROBOT_REGISTRY
from eden.utils.asset_audit import (
    ASSET_FILE_SUFFIXES,
    LANDING_SECONDS,
    AuditReport,
    AuditThresholds,
    HoldRun,
    SettleRun,
    audit_entity,
    grade_hold,
    grade_settle,
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
        assert settle.details["rest_penetration"] < AuditThresholds().max_penetration
        assert settle.details["peak_penetration"] >= settle.details["rest_penetration"]
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
        assert set(hold.details["drift"]) == {"shoulder", "elbow"}
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
        assert contact.details["dropped_pairs"] == {"base/fore": [1, 1]}
        assert "base/fore (1 of 1 shape pairs)" in contact.message and "enable_neutral_collision" in contact.message

    def test_the_dropped_pair_does_not_depend_on_the_default_pose(self, tmp_path):
        """Swung 1.4 rad away, the forearm clears the base at the default pose; Genesis dropped the pair at build."""
        path = tmp_path / "sloppy_arm.xml"
        path.write_text(SLOPPY_ARM)
        options = options_for_file(str(path)).model_copy(
            update={"dofs_name": ["shoulder", "elbow", "wrist"], "default_dofs_pos": {"shoulder": 1.4}}
        )
        contact = audit_entity(options).check("self_contact")
        assert contact.details["pairs"] == []
        assert contact.details["dropped_pairs"] == {"base/fore": [1, 1]}

    def test_a_task_that_keeps_neutral_pairs_reports_none_dropped(self, tmp_path):
        path = tmp_path / "sloppy_arm.xml"
        path.write_text(SLOPPY_ARM)
        report = audit_entity(options_for_file(str(path)), env_options=EnvOptions(enable_neutral_collision=True))
        contact = report.check("self_contact")
        assert contact.details["dropped_pairs"] == {}
        assert "base/fore" in contact.details["pairs"]

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


def test_report_serializes_and_renders(free_box_report, tmp_path):
    path = tmp_path / "report.json"
    free_box_report.to_json(str(path))
    data = json.loads(path.read_text())
    assert data == json.loads(json.dumps(free_box_report.as_dict()))
    assert data["failed"] is False
    assert [c["name"] for c in data["checks"]] == [c.name for c in free_box_report.checks]
    text = free_box_report.render()
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

    @pytest.mark.parametrize("name", ["notes.txt", "model.mjcf", "model.dae"])
    def test_a_suffix_the_loader_cannot_build_lists_the_known_ones(self, tmp_path, name):
        path = tmp_path / name
        path.write_text("")
        with pytest.raises(ValueError, match=r"not an asset file Eden loads.*\.urdf"):
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
    assert "base/fore" in contact.details["pairs"]
    assert contact.details["dropped_pairs"] == {}


class TestPhysics:
    def test_defaults_to_env_options_defaults_and_records_them(self, free_box_report):
        defaults = EnvOptions()
        assert free_box_report.physics == {
            "sim_dt": defaults.sim_dt,
            "sim_substeps": defaults.sim_substeps,
            "integrator": defaults.integrator,
        }
        assert f"sim_dt={defaults.sim_dt}" in free_box_report.render().splitlines()[0]

    def test_runs_at_the_given_env_options_without_modifying_them(self, free_box_path):
        env_options = EnvOptions(num_envs=4096, sim_dt=0.01, sim_substeps=2)
        report = audit_entity(options_for_file(free_box_path), env_options=env_options)
        assert report.physics["sim_dt"] == 0.01
        assert report.physics["sim_substeps"] == 2
        assert env_options.num_envs == 4096

    def test_a_closed_loop_gripper_gets_the_substeps_it_declares(self):
        gripper = ROBOT_REGISTRY.get("robotiq2f85")
        report = audit_entity(gripper())
        dt = report.physics["sim_dt"] / report.physics["sim_substeps"]
        assert dt <= gripper.max_substep_dt + 1e-12


@pytest.mark.parametrize("key", ["go1", "oli", "franka_emika_panda"])
def test_known_good_presets_do_not_fail(key):
    """Presets the repo trains and deploys. An unpowered legged robot lands hard; only resting depth is graded."""
    report = audit_entity(ROBOT_REGISTRY.get(key)(), target=key)
    failed = [f"{c.name}: {c.message}" for c in report.checks if c.status == "fail"]
    assert not failed, failed


def test_file_suffixes_are_the_ones_the_entity_loader_builds():
    surface_meshes = tuple(s for s in MESH_FILE_SUFFIXES if s not in TET_MESH_FILE_SUFFIXES)
    assert set(ASSET_FILE_SUFFIXES) == {".urdf", ".xml", ".usd", ".usda", ".usdc", ".usdz", *surface_meshes}


class TestThresholds:
    def test_defaults_leave_the_rest_window_after_the_landing(self):
        thresholds = AuditThresholds()
        assert thresholds.seconds >= thresholds.min_seconds
        assert thresholds.min_seconds == pytest.approx(
            thresholds.fall_seconds + LANDING_SECONDS + thresholds.rest_window
        )

    @pytest.mark.parametrize("seconds", [0.25, 0.8])
    def test_a_run_whose_rest_window_would_hold_the_landing_is_refused_with_the_minimum(self, seconds):
        with pytest.raises(ValueError, match=r"too short to grade settle.*seconds >= 0\.85.*--seconds 0\.9"):
            AuditThresholds(seconds=seconds)

    def test_a_shorter_rest_window_admits_a_shorter_run(self):
        assert AuditThresholds(seconds=0.7, rest_window=0.05).seconds == 0.7

    @pytest.mark.parametrize("field", ["seconds", "rest_window", "max_penetration", "drop_height"])
    @pytest.mark.parametrize("value", [0.0, -1.0, float("nan")])
    def test_every_limit_must_be_positive(self, field, value):
        with pytest.raises(ValueError, match=rf"must be a positive number; got {field}="):
            AuditThresholds(**{field: value})


def test_genesis_still_keeps_its_collision_candidates_where_self_contact_reads_them(tmp_path):
    """``self_contact`` diffs ``Collider._valid_collision_pairs`` across two builds; a rename upstream breaks it here.

    Built with the neutral filter, the sloppy arm keeps no candidate between its base and forearm; built without
    it, it does.
    """
    from eden.utils.asset_audit import AUDIT_ENTITY_NAME, _build_audit_env, _collision_pairs, _link_pair

    path = tmp_path / "sloppy_arm.xml"
    path.write_text(SLOPPY_ARM)
    options = options_for_file(str(path))
    pairs = {}
    for keep in (False, True):
        env = _build_audit_env(options.model_copy(deep=True), enable_neutral_collision=keep)
        try:
            entity = env.entities[AUDIT_ENTITY_NAME]
            pairs[keep] = {_link_pair(env, entity, a, b) for a, b in _collision_pairs(env).tolist()}
        finally:
            env.destroy()
    assert ("base", "fore") not in pairs[False]
    assert ("base", "fore") in pairs[True]


def test_a_partly_dropped_link_pair_is_reported_with_its_shape_counts():
    """Genesis drops 6 shape pairs between the YAM's fingers at build and keeps the rest, so they still collide."""
    contact = audit_entity(ROBOT_REGISTRY.get("yam")()).check("self_contact")
    dropped, total = contact.details["dropped_pairs"]["lf_down/rf_down"]
    assert dropped == 6
    assert total > dropped


class _TenthArmature(ActuatorSpecOptions):
    """An actuator spec whose reflected armature is exactly Genesis' 0.1 fill."""

    ROTOR_INERTIA: ClassVar[float] = 0.1
    GEAR_RATIO: ClassVar[float] = 1.0


class _SpecArm(EntityOptions):
    dofs_spec: ClassVar[dict] = {"shoulder|elbow": _TenthArmature}


def test_armature_from_a_dofs_spec_is_configured_even_at_the_fill_value(tmp_path):
    """The spec's ARMATURE() is 0.1, the fill value, but the options chose it, so only the wrist was filled."""
    path = tmp_path / "sloppy_arm.xml"
    path.write_text(SLOPPY_ARM)
    options = _SpecArm(
        file=str(path),
        is_fixed_base=True,
        is_articulated=True,
        is_actuated=True,
        dofs_name=["shoulder", "elbow", "wrist"],
    )
    armature = audit_entity(options).check("armature")
    assert armature.details["armature"]["shoulder"] == pytest.approx(0.1)
    assert armature.details["filled"] == ["wrist"]


def test_hold_grades_the_pandas_fingers_as_slides():
    hold = audit_entity(ROBOT_REGISTRY.get("franka_emika_panda")()).check("hold")
    assert hold.status == "pass", hold.message
    assert hold.details["linear_dofs"] == ["finger_joint1", "finger_joint2"]


def _settle(**overrides) -> SettleRun:
    """A settle run that passes at rest; each test changes what it is about."""
    values = {
        "first_non_finite_s": None,
        "top_z": 0.1,
        "max_speed": 1.0,
        "max_ang_speed": 5.0,
        "peak_penetration": 0.002,
        "rest_penetration": 0.001,
        "tail_speed": 0.0,
        "tail_ang_speed": 0.0,
        "travel_ratio": 1.0,
    }
    return SettleRun(**{**values, **overrides})


class TestGradeSettle:
    @pytest.mark.parametrize(
        "overrides, status, message",
        [
            ({}, "pass", "at rest after 2.0 s, 1.0 mm deep"),
            ({"first_non_finite_s": 0.37}, "fail", "non-finite at 0.370 s"),
            ({"top_z": -0.2}, "fail", "fell through the ground plane"),
            ({"max_speed": 25.0}, "fail", "exploded: peak link speed 25 m/s"),
            ({"max_ang_speed": 250.0}, "fail", "exploded"),
            ({"rest_penetration": 0.02, "peak_penetration": 0.02}, "fail", "rests 20.0 mm deep"),
            ({"peak_penetration": 0.03}, "pass", "30.0 mm on impact"),  # the landing is reported, not graded
            ({"tail_speed": 0.4, "travel_ratio": 0.1}, "warn", "jitters in place"),
            ({"tail_ang_speed": 3.0, "travel_ratio": 0.1}, "warn", "jitters in place"),
            ({"tail_speed": 0.4, "travel_ratio": 0.9}, "pass", "still moving steadily"),
            ({"tail_speed": 0.01, "travel_ratio": 0.05}, "pass", "at rest"),  # contact noise under rest_speed
        ],
    )
    def test_each_verdict(self, overrides, status, message):
        check = grade_settle(_settle(**overrides), AuditThresholds())
        assert (check.status, check.name) == (status, "settle")
        assert message in check.message


class TestGradeHold:
    @pytest.mark.parametrize(
        "drift, speed, linear, status, message",
        [
            ({"j1": 0.01, "f1": 0.001}, {"j1": 1.0, "f1": 0.1}, {"f1"}, "pass", "worst f1 within 0.001 m"),
            ({"j1": 0.2}, {"j1": 1.0}, set(), "warn", "j1 (0.2 rad)"),
            # 3 cm on a gripper finger passed the old shared 0.05 limit; in metres it is far past 5 mm.
            ({"j1": 0.01, "f1": 0.03}, {"j1": 1.0, "f1": 0.1}, {"f1"}, "warn", "f1 (0.03 m)"),
            ({"j1": 0.01}, {"j1": 80.0}, set(), "fail", "j1 (80 rad/s)"),
            # 6 m/s is far under the hinge limit of 50 rad/s and past the slide limit of 5 m/s.
            ({"f1": 0.001}, {"f1": 6.0}, {"f1"}, "fail", "f1 (6 m/s)"),
        ],
    )
    def test_each_dof_is_graded_in_its_own_unit(self, drift, speed, linear, status, message):
        check = grade_hold(HoldRun(finite=True, drift=drift, speed=speed, linear=frozenset(linear)), AuditThresholds())
        assert check.status == status
        assert message in check.message

    def test_non_finite_state_fails(self):
        check = grade_hold(HoldRun(finite=False, drift={"j1": 0.0}, speed={"j1": 0.0}), AuditThresholds())
        assert check.status == "fail"
        assert "non-finite" in check.message
