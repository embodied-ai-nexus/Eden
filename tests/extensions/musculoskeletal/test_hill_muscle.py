"""Unit tests for MuJoCo-exact Hill muscle curves and MJCF joint-torque stripping."""

from __future__ import annotations

import numpy as np
import pytest

from eden.extensions.musculoskeletal.hill_muscle import (
    MuscleParams,
    dense_actuator_moment,
    muscle_activation_dot,
    muscle_actuator_force,
    step_muscle_activation,
    strip_muscles_for_genesis,
)

_TENDON_MUSCLE_XML = """
<mujoco model="two_muscle_elbow">
  <option timestep="0.002"/>
  <worldbody>
    <site name="s_flex_orig" pos="0 0 0.08"/>
    <site name="s_ext_orig" pos="0 0 -0.08"/>
    <body name="forearm" pos="0 0 0">
      <joint name="elbow" type="hinge" axis="0 1 0" limited="true" range="-1.2 1.2"/>
      <geom type="capsule" fromto="0 0 0 0.28 0 0" size="0.025" mass="1.2"/>
      <site name="s_flex_ins" pos="0.06 0 0.03"/>
      <site name="s_ext_ins" pos="0.06 0 -0.03"/>
    </body>
  </worldbody>
  <tendon>
    <spatial name="t_flex"><site site="s_flex_orig"/><site site="s_flex_ins"/></spatial>
    <spatial name="t_ext"><site site="s_ext_orig"/><site site="s_ext_ins"/></spatial>
  </tendon>
  <actuator>
    <muscle name="flexor" tendon="t_flex" force="350" lengthrange="0.05 0.14"/>
    <muscle name="extensor" tendon="t_ext" force="300" lengthrange="0.05 0.14"/>
  </actuator>
</mujoco>
"""


def test_numpy_and_torch_hill_curves_match() -> None:
    torch = pytest.importorskip("torch")
    params = MuscleParams(
        tau_act=np.array([0.01, 0.012]),
        tau_deact=np.array([0.04, 0.045]),
        tausmooth=np.array([0.0, 0.02]),
        act_limited=np.array([True, True]),
        act_range=np.array([[0.0, 1.0], [0.0, 1.0]]),
        length_range=np.array([[0.08, 0.16], [0.10, 0.22]]),
        acc0=np.array([1.0, 1.0]),
        gain_prm=np.array(
            [
                [0.75, 1.05, 320.0, 200.0, 0.5, 1.6, 1.5, 1.3, 1.4],
                [0.75, 1.05, -1.0, 240.0, 0.5, 1.6, 1.5, 1.3, 1.4],
            ]
        ),
        bias_prm=np.array(
            [
                [0.75, 1.05, 320.0, 200.0, 0.5, 1.6, 1.5, 1.3, 1.4],
                [0.75, 1.05, -1.0, 240.0, 0.5, 1.6, 1.5, 1.3, 1.4],
            ]
        ),
    )
    rng = np.random.default_rng(3)
    L_np = rng.uniform(0.07, 0.23, size=(6, 2))
    V_np = rng.normal(scale=0.15, size=(6, 2))
    act_np = rng.uniform(0.05, 0.95, size=(6, 2))
    ctrl_np = rng.uniform(0.0, 1.0, size=(6, 2))

    f_np = muscle_actuator_force(L_np, V_np, act_np, params)
    next_np, _ = step_muscle_activation(act_np, ctrl_np, params, dt=0.002, substeps=3)

    f_t = muscle_actuator_force(
        torch.as_tensor(L_np, dtype=torch.float64),
        torch.as_tensor(V_np, dtype=torch.float64),
        torch.as_tensor(act_np, dtype=torch.float64),
        params,
    )
    next_t, _ = step_muscle_activation(
        torch.as_tensor(act_np, dtype=torch.float64),
        torch.as_tensor(ctrl_np, dtype=torch.float64),
        params,
        dt=0.002,
        substeps=3,
    )
    np.testing.assert_allclose(f_t.numpy(), f_np, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(next_t.numpy(), next_np, atol=1e-12, rtol=1e-12)


def test_mujoco_parity_and_strip_muscles_for_genesis() -> None:
    mujoco = pytest.importorskip("mujoco")
    spec = mujoco.MjSpec.from_string(_TENDON_MUSCLE_XML)
    model = spec.compile()
    data = mujoco.MjData(model)
    data.qpos[0] = 0.25
    data.qvel[0] = -0.3
    data.act[:] = [0.65, 0.2]
    data.ctrl[:] = [0.9, 0.1]
    mujoco.mj_forward(model, data)

    params = MuscleParams.from_model(model)
    np.testing.assert_allclose(muscle_activation_dot(data.ctrl, data.act, params), data.act_dot, atol=1e-12)
    f_mus = muscle_actuator_force(data.actuator_length, data.actuator_velocity, data.act, params)
    np.testing.assert_allclose(f_mus, data.actuator_force, atol=1e-12)
    moment = dense_actuator_moment(model, data)
    np.testing.assert_allclose(moment.T @ f_mus, data.qfrc_actuator, atol=1e-12)

    stripped = strip_muscles_for_genesis(spec, add_joint_motors=True).compile()
    assert stripped.ntendon == 0
    assert stripped.na == 0
    assert stripped.nu == 1
    assert stripped.nsite == model.nsite
