"""``eden.extensions.policies.sonic.band``: the elastic band's wrench and its keys."""

from __future__ import annotations

import numpy as np
import pytest
from genesis.utils import geom as gu

from eden.extensions.policies.sonic.band import ElasticBand


def test_band_starts_enabled_with_reference_gains():
    band = ElasticBand()
    assert band.enable is True  # the runbook's "press 9 to drop" depends on this
    assert (band.kp_pos, band.kd_pos, band.kp_ang, band.kd_ang) == (10000.0, 1000.0, 1000.0, 10.0)
    np.testing.assert_array_equal(band.point, [0, 0, 1])
    assert band.length == 0.0


def test_wrench_position_spring_and_damping():
    band = ElasticBand()
    force, torque = band.wrench(
        pos=np.array([0.1, -0.2, 0.7]),
        quat_wxyz=np.array([1.0, 0, 0, 0]),
        lin_vel=np.array([0.5, 0.0, -1.0]),
        ang_vel=np.zeros(3),
    )
    # f = kp*(point - pos + [0,0,length]) - kd*v
    np.testing.assert_allclose(force, [10000 * -0.1 - 1000 * 0.5, 10000 * 0.2, 10000 * 0.3 + 1000 * 1.0])
    np.testing.assert_allclose(torque, 0.0)


def test_wrench_length_offset_raises_anchor():
    band = ElasticBand()
    band.length = 0.5
    force, _ = band.wrench(np.array([0, 0, 1.0]), np.array([1.0, 0, 0, 0]), np.zeros(3), np.zeros(3))
    np.testing.assert_allclose(force, [0, 0, 10000 * 0.5])


def test_wrench_orientation_spring():
    band = ElasticBand()
    quat = gu.axis_angle_to_quat(np.array(0.3), np.array([0.0, 1.0, 0.0]))  # 0.3 rad pitch
    _, torque = band.wrench(np.array([0, 0, 1.0]), quat, np.zeros(3), np.array([0.0, 2.0, 0.0]))
    # tau = -kp_ang * rotvec - kd_ang * w = [0, -1000*0.3 - 10*2, 0]
    np.testing.assert_allclose(torque, [0, -1000 * 0.3 - 10 * 2.0, 0], atol=1e-9)


def test_wrench_identity_orientation_no_torque():
    band = ElasticBand()
    _, torque = band.wrench(np.zeros(3), np.array([1.0, 0, 0, 0]), np.zeros(3), np.zeros(3))
    np.testing.assert_allclose(torque, 0.0, atol=1e-12)


def test_keys_toggle_and_adjust_length():
    band = ElasticBand()
    assert band.handle_key("9") is True
    assert band.enable is False
    assert band.handle_key("9") is True
    assert band.enable is True
    assert band.handle_key("7") is True
    assert band.length == pytest.approx(-0.1)
    assert band.handle_key("8") is True
    assert band.handle_key("8") is True
    assert band.length == pytest.approx(0.1)
    assert band.handle_key("z") is False
