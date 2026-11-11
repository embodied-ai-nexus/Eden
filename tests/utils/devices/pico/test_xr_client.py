"""``eden.utils.devices.pico.xr_client``: the OpenXR -> robot basis change, ``PicoXrClient`` over a fake SDK,
and ``load_sdk``'s preload of the PC Service library.

The basis change is pinned against the documented axis map, not recomputed from the matrices the
implementation uses. ``load_sdk`` is driven at the import system and the filesystem: a finder that
fails to import ``xrobotoolkit_sdk`` until a real shared library is loaded, and ``EDEN_PXREA_SDK_DIR``
pointing at a ``tmp_path`` directory with or without that library.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
from genesis.utils.geom import quat_to_R

import eden.utils.devices.pico.xr_client as XC
from eden.utils.devices.pico.xr_client import (
    BODY_JOINT_NAMES,
    load_sdk,
    xr_points_to_robot,
    xr_pose_to_pos_quat,
    xr_pose_to_se3,
    xr_vel6_to_robot,
)
from tests.utils.devices.pico.conftest import (
    PXREA_LIB,
    SDK_MODULE,
    FakeSdk,
    SdkImportFinder,
    copy_loadable_library,
    library_is_loaded,
)


def _identity_pose(pos):
    """Service pose (x, y, z, qx, qy, qz, qw) at ``pos`` with no rotation."""
    return [*pos, 0.0, 0.0, 0.0, 1.0]


@pytest.mark.parametrize(
    "xr_axis, robot_axis",
    [
        ((1.0, 0.0, 0.0), (0.0, -1.0, 0.0)),  # XR right      -> robot -y (right of forward)
        ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),  # XR up         -> robot +z
        ((0.0, 0.0, -1.0), (1.0, 0.0, 0.0)),  # XR forward    -> robot +x
    ],
)
def test_position_axes_map_to_the_robot_convention(xr_axis, robot_axis):
    mat = xr_pose_to_se3(_identity_pose(xr_axis))
    np.testing.assert_allclose(mat[:3, 3], robot_axis, atol=1e-12)
    np.testing.assert_allclose(xr_points_to_robot([xr_axis])[0], robot_axis, atol=1e-12)


def test_yaw_about_xr_up_becomes_yaw_about_robot_z():
    """A 90 deg turn about the operator's up axis must be a 90 deg turn about world z."""
    half = np.sqrt(0.5)
    pose = [0.0, 0.0, 0.0, 0.0, half, 0.0, half]  # xyzw: +90 deg about XR +y (up)
    pq = xr_pose_to_pos_quat(pose)
    assert pq is not None
    expected = quat_to_R(np.array([half, 0.0, 0.0, half]))  # wxyz: +90 deg about robot +z
    np.testing.assert_allclose(quat_to_R(pq[1]), expected, atol=1e-9)


def test_body_frame_axes_are_re_expressed_too():
    """The conjugation must rotate the pose's own axes, not just its origin."""
    mat = xr_pose_to_se3(_identity_pose((0.0, 0.0, 0.0)))
    # The basis change of the identity is the identity.
    np.testing.assert_allclose(mat, np.eye(4), atol=1e-12)

    # +90 deg about XR +x (the operator's right) is a pitch-up: it must come out
    # as +90 deg about robot -y, i.e. robot forward (+x) tips to robot up (+z)
    # and robot up tips to backward (-x).
    half = np.sqrt(0.5)
    rot = xr_pose_to_se3([0.0, 0.0, 0.0, half, 0.0, 0.0, half])[:3, :3]
    np.testing.assert_allclose(rot @ np.array([1.0, 0.0, 0.0]), [0.0, 0.0, 1.0], atol=1e-9)
    np.testing.assert_allclose(rot @ np.array([0.0, 0.0, 1.0]), [-1.0, 0.0, 0.0], atol=1e-9)


def test_degenerate_quaternion_reports_untracked():
    mat = xr_pose_to_se3([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(mat, np.zeros((4, 4)))
    assert xr_pose_to_pos_quat([1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0]) is None


def test_pose_shape_is_validated():
    with pytest.raises(ValueError, match="7-element"):
        xr_pose_to_se3([0.0, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match=r"\(N, 3\)"):
        xr_points_to_robot([1.0, 2.0, 3.0])


def test_velocity6_rotates_both_halves_into_robot_frame():
    # XR +x (right) -> robot -y; XR -z (forward) -> robot +x. Linear and angular
    # halves each rotate by R_ROBOT_XR.
    out = xr_vel6_to_robot([1.0, 0.0, 0.0, 0.0, 0.0, -1.0])
    np.testing.assert_allclose(out[:3], [0.0, -1.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(out[3:], [1.0, 0.0, 0.0], atol=1e-12)


def test_velocity6_shape_is_validated():
    with pytest.raises(ValueError, match="6-element"):
        xr_vel6_to_robot([0.0, 0.0, 0.0])


_POSE = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
_V = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]


def test_trackers_keyed_by_role_falling_back_to_serial(fake_client):
    sdk = FakeSdk(trackers=[("PA_WAIST", _POSE, _V, _V), ("PB_UNKNOWN", _POSE, _V, _V)])
    client = fake_client(sdk, tracker_roles={"PA_WAIST": "waist"})
    trackers = client.trackers()
    assert set(trackers) == {"waist", "PB_UNKNOWN"}  # mapped -> role, unmapped -> serial
    assert trackers["waist"]["serial"] == "PA_WAIST"
    assert trackers["waist"]["role"] == "waist"
    assert trackers["PB_UNKNOWN"]["role"] is None
    # velocity is rotated into the robot frame (XR +x -> robot -y)
    np.testing.assert_allclose(trackers["waist"]["vel"][:3], [0.0, -1.0, 0.0], atol=1e-12)


def test_supports_up_to_five_trackers(fake_client):
    sdk = FakeSdk(trackers=[(f"S{i}", _POSE, _V, _V) for i in range(5)])
    client = fake_client(sdk)
    assert len(client.trackers()) == 5  # 2 legs + waist + 2 wrists, etc.


def test_body_joints_named_dict_with_derivatives(fake_client):
    client = fake_client(FakeSdk(body=True))
    body = client.body_joints()
    assert set(body) == set(BODY_JOINT_NAMES)
    for joint in ("left_wrist", "right_wrist", "left_ankle", "right_foot", "pelvis"):
        assert len(body[joint]["pos"]) == 3 and len(body[joint]["quat"]) == 4
        assert len(body[joint]["vel"]) == 6 and len(body[joint]["acc"]) == 6


def test_body_joints_none_without_full_body(fake_client):
    assert fake_client(FakeSdk(body=False)).body_joints() is None


def test_state_publishes_trackers_and_named_body(fake_client):
    sdk = FakeSdk(trackers=[("PA_WAIST", _POSE, _V, _V)], body=True)
    state = fake_client(sdk, tracker_roles={"PA_WAIST": "waist"}).state()
    assert state["device"] == "pico" and state["ready"] is True
    assert "waist" in state["trackers"]
    assert state["body_ready"] is True
    # joint 20 (left_wrist) has XR pos (0, 0, 2.0) -> robot (-2, 0, 0) via R_ROBOT_XR
    np.testing.assert_allclose(state["body"]["left_wrist"]["pos"], [-2.0, 0.0, 0.0], atol=1e-12)
    assert "body_joint_pos" not in state  # replaced by the named `body` dict


def test_state_includes_raw_body_passthrough(fake_client):
    sdk = FakeSdk(body=True)
    state = fake_client(sdk).state()
    # Exact nested equality: no frame rotation, no xyzw->wxyz flip, no
    # untracked-joint substitution — byte-for-byte what the SDK reported.
    assert state["body_raw"] == sdk.get_body_joints_pose()
    assert state["body_raw_timestamp_ns"] == sdk.get_time_stamp_ns()


def test_state_omits_raw_body_when_unavailable(fake_client):
    state = fake_client(FakeSdk(body=False)).state()
    assert "body_raw" not in state and "body_raw_timestamp_ns" not in state


def test_state_without_derived_body_keeps_raw_and_controllers(fake_client):
    sdk = FakeSdk(trackers=[("PA_WAIST", _POSE, _V, _V)], body=True)
    state = fake_client(sdk, tracker_roles={"PA_WAIST": "waist"}).state(derived_body=False)
    # The lossless passthrough and the controller keys survive; the derived
    # per-joint dict and trackers are skipped; body_ready reflects the raw read.
    assert state["body_raw"] == sdk.get_body_joints_pose()
    assert state["body_ready"] is True
    assert "body" not in state and "trackers" not in state
    assert "left_trigger_value" in state and "right_thumbstick_value" in state
    assert fake_client(FakeSdk(body=False)).state(derived_body=False)["body_ready"] is False


def test_state_derived_body_matches_direct_body_joints(fake_client):
    # state() reuses its single raw pose read for the derived dict — the result
    # must equal a direct body_joints() call on the same SDK data.
    sdk = FakeSdk(body=True)
    client = fake_client(sdk)
    assert client.state()["body"] == client.body_joints()


# ---------------------------------------------------------------------------
# load_sdk: libPXREARobotSDK preload
# ---------------------------------------------------------------------------

_HINT = "XRoboToolkit-PC-Service-Pybind"


def test_load_sdk_preloads_the_service_library_and_retries(sdk_import, tmp_path, monkeypatch):
    """The first import fails on the library; ``load_sdk`` loads it from EDEN_PXREA_SDK_DIR and imports again."""
    lib = copy_loadable_library(tmp_path / "sdk")
    monkeypatch.setenv("EDEN_PXREA_SDK_DIR", str(lib.parent))
    finder = sdk_import(SdkImportFinder(links_against=lib))
    assert not library_is_loaded(lib)

    sdk = load_sdk()
    assert sdk is sys.modules[SDK_MODULE]
    assert library_is_loaded(lib)
    assert finder.attempts == 2 and len(finder.errors) == 1  # one linkage failure, then the retry imported


def test_load_sdk_hints_when_the_service_library_is_absent(sdk_import, tmp_path, monkeypatch):
    # The built-in search directory is read, not patched: where the PC Service is installed, the
    # library is never absent, so this case cannot occur on that machine.
    installed = [d for d in XC._PXREA_SEARCH_DIRS if os.path.exists(os.path.join(d, PXREA_LIB))]
    if installed:
        pytest.skip(f"the PC Service library is installed under {installed[0]}, so it cannot be absent here")
    (tmp_path / "sdk").mkdir()
    monkeypatch.setenv("EDEN_PXREA_SDK_DIR", str(tmp_path / "sdk"))
    finder = sdk_import(SdkImportFinder(links_against=tmp_path / "sdk" / PXREA_LIB))

    with pytest.raises(ImportError, match=_HINT) as excinfo:
        load_sdk()
    assert finder.attempts == 1  # nothing to preload, so no retry
    assert excinfo.value.__cause__ is finder.errors[0]


def test_load_sdk_skips_preload_for_a_missing_module(tmp_path, monkeypatch):
    """A plain missing module is not a linkage error: the library must stay unloaded."""
    lib = copy_loadable_library(tmp_path / "sdk")
    monkeypatch.setenv("EDEN_PXREA_SDK_DIR", str(lib.parent))
    monkeypatch.setitem(sys.modules, SDK_MODULE, None)  # ``import`` raises ModuleNotFoundError

    with pytest.raises(ImportError, match=_HINT) as excinfo:
        load_sdk()
    assert isinstance(excinfo.value.__cause__, ModuleNotFoundError)
    assert not library_is_loaded(lib)


def test_load_sdk_hints_when_the_retry_also_fails(sdk_import, tmp_path, monkeypatch):
    """The library loads but the module still does not link: the hint chains the retry's error."""
    lib = copy_loadable_library(tmp_path / "sdk")
    monkeypatch.setenv("EDEN_PXREA_SDK_DIR", str(lib.parent))
    finder = sdk_import(SdkImportFinder(links_against=None))

    with pytest.raises(ImportError, match=_HINT) as excinfo:
        load_sdk()
    assert library_is_loaded(lib)
    assert finder.attempts == 2
    assert excinfo.value.__cause__ is finder.errors[1]
