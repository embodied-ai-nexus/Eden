"""``XArm7XHand1``: the welded xArm7 + XHand1 describes the same chain as ConTrack's merged URDF.

ConTrack ships arm and hand in one file; Eden welds its own ``Xarm7`` and ``XHand1_L``/``XHand1_R``, and the
released 19-DOF reference clips stay valid only while the two chains agree. The golden
(``tests/data/contrack_xarm_xhand_golden.npz``) is the merged URDF's joint table, extracted once from
``EmptyBlueBox/ConTrack@f464773``; the licensed asset itself is not vendored.
"""

from __future__ import annotations

import functools
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

import tests

from eden.options.robots.composite import XArm7XHand1_L, XArm7XHand1_R
from eden.options.robots.composite.xarm_xhand import _MOUNT_OFFSET
from eden.utils.assets import get_asset_path

GOLDEN = Path(tests.__file__).parent / "data" / "contrack_xarm_xhand_golden.npz"

# The constant per-link frame rotation between ConTrack's XHand revision and Eden's:
# 180 deg about (1, ±1, 0)/sqrt(2). Right-multiplying a link frame by it maps one
# convention onto the other, which is why link positions agree while link quaternions
# differ by exactly this rotation.
FRAME_CONVENTION = {
    "right": np.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]),
    "left": np.array([[0.0, -1.0, 0.0], [-1.0, 0.0, 0.0], [0.0, 0.0, -1.0]]),
}

# ConTrack's URDF writes the mount rotation as rpy="3.14 0 ±1.57", truncating pi and pi/2 —
# 1.781 mrad off the exact rotation, which the composite mounts at instead. That truncation is
# the ONLY thing standing between the two chains, so it is what both tolerances here measure.
# Applied elementwise to rotation matrices, where a small angle shows up as a deviation of
# roughly the same magnitude.
CONTRACK_TRUNCATION_ATOL = 2e-3
# 1.781 mrad over the hand's ~0.2 m reach (worst observed: 0.37 mm). Why the composite accepts
# this rather than reproducing ConTrack's truncated frame: see the module docstring of
# ``eden/options/robots/composite/xarm_xhand.py``.
TRUNCATION_POS_TOL = 5e-4

ARM_JOINTS = tuple(f"joint{i}" for i in range(1, 8))
HAND_JOINTS = (
    "thumb_bend_joint",
    "thumb_rota_joint1",
    "thumb_rota_joint2",
    "index_bend_joint",
    "index_joint1",
    "index_joint2",
    "mid_joint1",
    "mid_joint2",
    "ring_joint1",
    "ring_joint2",
    "pinky_joint1",
    "pinky_joint2",
)

RIGS = {"right": XArm7XHand1_R, "left": XArm7XHand1_L}


# ---------------------------------------------------------------------------
# URDF parsing + forward kinematics (independent of Genesis: these tests are
# about the asset descriptions, so they must not depend on a built scene)
# ---------------------------------------------------------------------------


def _rpy_to_R(rpy: np.ndarray) -> np.ndarray:
    r, p, y = rpy
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx  # URDF rpy convention


def _T(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = p
    return M


#: The composite's weld, as a transform: ``link_eef`` -> the hand's base link.
MOUNT = _T(np.eye(3), np.array(_MOUNT_OFFSET))


def _axis_angle_to_R(axis: np.ndarray, angle: float) -> np.ndarray:
    norm = np.linalg.norm(axis)
    if norm == 0.0:
        return np.eye(3)
    a = axis / norm
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def _parse_joints(urdf_path: str) -> dict[str, dict]:
    """``{joint name: {type, parent, child, origin (4x4), axis (3,)}}``."""
    joints = {}
    for j in ET.parse(urdf_path).getroot().findall("joint"):
        origin = j.find("origin")
        xyz = np.zeros(3) if origin is None else np.array([float(v) for v in origin.get("xyz", "0 0 0").split()])
        rpy = np.zeros(3) if origin is None else np.array([float(v) for v in origin.get("rpy", "0 0 0").split()])
        axis = j.find("axis")
        joints[j.get("name")] = dict(
            type=j.get("type"),
            parent=j.find("parent").get("link"),
            child=j.find("child").get("link"),
            origin=_T(_rpy_to_R(rpy), xyz),
            axis=np.zeros(3) if axis is None else np.array([float(v) for v in axis.get("xyz").split()]),
        )
    return joints


@functools.lru_cache(maxsize=None)
def _golden_joints(side: str) -> dict[str, dict]:
    """The merged URDF's joint table, in the same shape :func:`_parse_joints` returns.

    Cached, so the returned dicts are shared across tests — read them, never mutate them.
    """
    d = np.load(GOLDEN, allow_pickle=False)
    joints = {}
    for i, name in enumerate(d[f"{side}_name"]):
        origin = d[f"{side}_origin"][i]
        joints[str(name)] = dict(
            type=str(d[f"{side}_type"][i]),
            parent=str(d[f"{side}_parent"][i]),
            child=str(d[f"{side}_child"][i]),
            origin=_T(_rpy_to_R(origin[3:]), origin[:3]),
            axis=d[f"{side}_axis"][i],
        )
    return joints


def _fk(joints: dict[str, dict], q: dict[str, float], root_pose: np.ndarray) -> dict[str, np.ndarray]:
    """World pose of every link reachable from the root, given joint angles ``q``."""
    # Sets, not a list: a root link parents many joints (the palm parents five fingers), and
    # unpacking a set of one also asserts the file is a single kinematic tree.
    (root,) = {j["parent"] for j in joints.values()} - {j["child"] for j in joints.values()}
    poses = {root: root_pose}
    pending = dict(joints)
    while pending:
        ready = [name for name, j in pending.items() if j["parent"] in poses]
        assert ready, f"unreachable links: {sorted(pending)}"
        for name in ready:
            j = pending.pop(name)
            local = j["origin"]
            if j["type"] in ("revolute", "continuous"):
                local = local @ _T(_axis_angle_to_R(j["axis"], q.get(name, 0.0)), np.zeros(3))
            poses[j["child"]] = poses[j["parent"]] @ local
    return poses


def _fk_both(side, arm_joints, hand_joints, q) -> tuple[dict, dict, dict]:
    """``(merged, arm, hand)`` link poses at identical joint values.

    The composite chain is the arm FK continued through the mount into the hand FK; the
    merged chain is ConTrack's single file. Same ``q`` feeds both — the hand joint names and
    sign conventions are shared, so a joint value means the same rotation in each.
    """
    arm = _fk(arm_joints, q, np.eye(4))
    hand = _fk(hand_joints[side], q, arm["link_eef"] @ MOUNT)
    return _fk(_golden_joints(side), q, np.eye(4)), arm, hand


def _random_configs(rig, seed: int, count: int):
    """``count`` random 19-DOF configurations keyed by joint name."""
    names = list(ARM_JOINTS) + list(rig.hand_dofs)
    rng = np.random.default_rng(seed)
    for _ in range(count):
        yield {name: float(v) for name, v in zip(names, rng.uniform(-2.0, 2.0, len(names)), strict=True)}


@pytest.fixture(scope="module")
def arm_joints() -> dict[str, dict]:
    return _parse_joints(get_asset_path("arms/ufactory_xarm7/xarm.urdf"))


@pytest.fixture(scope="module")
def hand_joints() -> dict[str, dict[str, dict]]:
    return {side: _parse_joints(get_asset_path(RIGS[side]().hand.file)) for side in RIGS}


# ---------------------------------------------------------------------------
# The composite's own declaration
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("side", ["right", "left"])
def test_composite_expands_to_arm_plus_welded_hand(side):
    rig = RIGS[side]()
    assert set(rig.entity_options()) == {"robot", "hand"}
    # A 4-tuple weld: an explicit mounting transform on the arm's flange link.
    assert rig.attachments() == {"hand": ("robot", "link_eef", _MOUNT_OFFSET, (1.0, 0.0, 0.0, 0.0))}
    # 19 DOFs total, split 7 (arm) / 12 (hand) — the clip's qpos width.
    assert tuple(rig.arm_dofs) == ARM_JOINTS
    assert tuple(rig.hand_dofs) == tuple(f"{side}_hand_{s}" for s in HAND_JOINTS)
    # The hand must be free-based for Genesis `attach` to absorb it into the arm's tree.
    assert rig.hand.is_fixed_base is False
    assert rig.arm.is_fixed_base is True
    assert rig.fingertip_links == rig.grasp_links and len(rig.grasp_links) == 5


@pytest.mark.parametrize("side", ["right", "left"])
def test_composite_restates_contrack_actuator_config(side):
    """ConTrack's IsaacLab gains, not Eden's stock Xarm7 / XHand1 defaults."""
    rig = RIGS[side]()
    assert set(rig.arm.default_dofs_kp.values()) == {1000.0}
    assert set(rig.arm.default_dofs_kd.values()) == {50.0}
    assert set(rig.arm.default_dofs_force_limits.values()) == {50.0}
    assert set(rig.hand.default_dofs_kp.values()) == {100.0}
    assert set(rig.hand.default_dofs_kd.values()) == {10.0}
    assert set(rig.hand.default_dofs_force_limits.values()) == {1.0}
    # ConTrack's home pose, not Eden's tabletop-ready Xarm7 default (joint4=pi/3).
    assert rig.arm.default_dofs_pos == dict(zip(ARM_JOINTS, (0.0, 0.0, 0.0, 0.22, 3.14, 1.33, 0.0), strict=True))


# ---------------------------------------------------------------------------
# Equivalence to ConTrack's merged URDF
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("side", ["right", "left"])
def test_arm_chain_is_identical_to_the_merged_urdf(side, arm_joints):
    """Eden's xArm7 asset IS ConTrack's arm: same origins and axes, bit for bit."""
    golden = _golden_joints(side)
    for name in ARM_JOINTS:
        assert arm_joints[name]["parent"] == golden[name]["parent"]
        assert arm_joints[name]["child"] == golden[name]["child"]
        np.testing.assert_array_equal(arm_joints[name]["origin"], golden[name]["origin"])
        np.testing.assert_array_equal(arm_joints[name]["axis"], golden[name]["axis"])


@pytest.mark.parametrize("side", ["right", "left"])
def test_mount_offset_equals_contrack_joint_eef(side):
    """``joint_eef`` composed with the frame-convention delta reduces to the mount offset.

    ConTrack welds the hand at ``link7`` with a rotation; Eden's XHand1 assets already
    carry that rotation in their base frame, so the composite's mount is a pure
    translation with an identity quaternion.
    """
    joint_eef = _golden_joints(side)["joint_eef"]
    assert joint_eef["parent"] == "link7"
    assert joint_eef["child"] == f"{side}_hand_link"
    # link_eef is a zero-offset fixed child of link7, so T(link7->hand) == T(link_eef->hand).
    residual = joint_eef["origin"] @ _T(FRAME_CONVENTION[side].T, np.zeros(3))
    np.testing.assert_allclose(residual[:3, :3], np.eye(3), atol=CONTRACK_TRUNCATION_ATOL)
    np.testing.assert_allclose(residual[:3, 3], _MOUNT_OFFSET, atol=1e-12)


@pytest.mark.parametrize("side", ["right", "left"])
def test_link_eef_is_a_zero_offset_child_of_link7(side, arm_joints):
    """The claim the mount transform relies on: attaching at ``link_eef`` == at ``link7``."""
    joint_eef = arm_joints["joint_eef"]
    assert (joint_eef["parent"], joint_eef["child"], joint_eef["type"]) == ("link7", "link_eef", "fixed")
    np.testing.assert_array_equal(joint_eef["origin"], np.eye(4))


@pytest.mark.parametrize("side", ["right", "left"])
def test_every_link_position_matches_the_merged_urdf(side, arm_joints, hand_joints):
    """FK the welded pair and ConTrack's merged chain at identical joint values.

    The strongest statement of equivalence: for random 19-DOF configurations, every shared
    link sits in the same place. Arm links agree exactly; hand links agree to
    :data:`TRUNCATION_POS_TOL`, which is ConTrack's truncated mount rotation and nothing
    else — swap :data:`MOUNT` for ``_golden_joints(side)["joint_eef"]["origin"] @
    _T(FRAME_CONVENTION[side].T, zeros)`` and the bound drops to 292 nm. Orientations are
    checked separately below: they differ by the constant frame-convention rotation.
    """
    worst_pos = 0.0
    for q in _random_configs(RIGS[side](), seed=0, count=25):
        merged, arm, hand = _fk_both(side, arm_joints, hand_joints, q)
        assert set(hand) <= set(merged), f"unmatched hand links: {sorted(set(hand) - set(merged))}"
        # ``link_eef`` alone has no counterpart in the merged file; assert the count so this
        # filter can never quietly shrink the comparison to a vacuous subset.
        shared = {link: pose for link, pose in (arm | hand).items() if link in merged}
        assert len(shared) == len(merged) == 38
        for link, pose in shared.items():
            worst_pos = max(worst_pos, np.linalg.norm(pose[:3, 3] - merged[link][:3, 3]))
    # Non-vacuous: the observed worst is ~0.35 mm against a 0.5 mm bound.
    assert worst_pos < TRUNCATION_POS_TOL, f"worst link position mismatch {worst_pos:.3e} m"


@pytest.mark.parametrize("side", ["right", "left"])
def test_hand_link_orientations_differ_by_a_constant_rotation(side, arm_joints, hand_joints):
    """Pins the one real deviation: hand link *frames* are rotated by a constant 180 deg.

    Positions, joint values, and dynamics are unaffected, but a term that reads a hand
    link's quaternion is not frame-comparable with the source implementation. Pinned here
    so a future asset swap that silently changes the convention is caught.
    """
    for q in _random_configs(RIGS[side](), seed=1, count=10):
        merged, _, hand = _fk_both(side, arm_joints, hand_joints, q)
        for link, pose in hand.items():
            residual = pose[:3, :3].T @ merged[link][:3, :3]
            np.testing.assert_allclose(residual, FRAME_CONVENTION[side], atol=CONTRACK_TRUNCATION_ATOL)


@pytest.mark.parametrize("side", ["right", "left"])
def test_hand_joint_limits_regression(side):
    """The two joints whose range Eden's later XHand1 revision tightened.

    ConTrack allows 1.745 rad on both thumb rotations; Eden's asset stops at 1.570. The
    bundled clips peak at 1.505 rad, so the reference stays reachable — this pins the
    delta so it is a deliberate, visible difference rather than a silent one.
    """
    urdf = get_asset_path(RIGS[side]().hand.file)
    uppers = {
        j.get("name"): float(j.find("limit").get("upper"))
        for j in ET.parse(urdf).getroot().findall("joint")
        if j.find("limit") is not None and j.find("limit").get("upper")
    }
    for joint in (f"{side}_hand_thumb_rota_joint1", f"{side}_hand_thumb_rota_joint2"):
        assert uppers[joint] == pytest.approx(1.57)
