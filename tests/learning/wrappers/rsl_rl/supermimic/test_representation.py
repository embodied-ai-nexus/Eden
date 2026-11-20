"""The generalized-coordinate state: its integration and its loss."""

import math

import pytest
import torch

from eden.learning.wrappers.rsl_rl.supermimic.representation import (
    REWARD_TERMS,
    GenCoordRepresentation,
    check_task_reward,
    quat_integrate,
)
from eden.utils.geom import quat_from_euler_xyz

J = 2


def _representation(decimation: int = 4) -> GenCoordRepresentation:
    return GenCoordRepresentation(
        fk=torch.nn.Identity(),
        default_joint_pos=torch.zeros(J),
        soft_joint_pos_limits=torch.tensor([[-1.0, 1.0]] * J),
        dt=0.02,
        decimation=decimation,
        num_actions=J,
    )


def _full_state(root_quat, lin=(0.0, 0.0, 0.0), ang=(0.0, 0.0, 0.0), q=(0.1, -0.2), qd=(1.0, 2.0)):
    return torch.cat(
        [
            torch.tensor([[0.0, 0.0, 0.7]]),
            root_quat,
            torch.tensor([q]),
            torch.tensor([lin]),
            torch.tensor([ang]),
            torch.tensor([qd]),
        ],
        dim=-1,
    )


def test_a_zero_prediction_coasts_at_constant_velocity():
    rep = _representation()
    yaw = quat_from_euler_xyz(torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([math.pi / 2]))
    full = _full_state(yaw, lin=(1.0, 0.0, 0.0))
    aux, state = rep.apply_prediction(rep.aux_state(full), rep.represent(full), torch.zeros(1, rep.prediction_dim))
    torch.testing.assert_close(aux[:, :3], torch.tensor([[0.02, 0.0, 0.7]]))
    torch.testing.assert_close(state[:, 10:12], torch.tensor([[0.1 + 0.02, -0.2 + 0.04]]))
    torch.testing.assert_close(state[:, 4:7], torch.tensor([[0.0, -1.0, 0.0]]), atol=1e-6, rtol=0)


def test_a_velocity_change_moves_the_pose_by_the_substep_share():
    """Four substeps under constant acceleration move the pose by (4 + 1) / (2 * 4) of the change times dt."""
    rep = _representation(decimation=4)
    full = _full_state(torch.tensor([[1.0, 0.0, 0.0, 0.0]]), qd=(0.0, 0.0))
    prediction = torch.zeros(1, rep.prediction_dim)
    prediction[0, 6] = 1.0
    _, state = rep.apply_prediction(rep.aux_state(full), rep.represent(full), prediction)
    torch.testing.assert_close(state[0, 10], torch.tensor(0.1 + 5 / 8 * 0.02))
    torch.testing.assert_close(state[0, 12], torch.tensor(1.0))


def test_the_root_orientation_loss_is_the_squared_angle():
    rep = _representation()
    angle = 0.05
    q0 = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    q1 = quat_integrate(q0, torch.tensor([[0.0, angle / 0.02, 0.0]]), 0.02)
    a = rep.loss_blocks(torch.zeros(1, rep.rep_state_dim), torch.cat([torch.zeros(1, 3), q0], -1))["root_ori"]
    b = rep.loss_blocks(torch.zeros(1, rep.rep_state_dim), torch.cat([torch.zeros(1, 3), q1], -1))["root_ori"]
    torch.testing.assert_close(((a - b) ** 2).sum(), torch.tensor(angle**2), rtol=1e-3, atol=0)


def test_a_task_reward_the_world_model_does_not_compute_is_refused():
    terms = dict(REWARD_TERMS)
    check_task_reward(terms)
    terms["joint_limit"] = (-10.0, None)
    with pytest.raises(ValueError, match=r"joint_limit: \(weight, std\) \(-10.0, None\) != \(-2.0, None\)"):
        check_task_reward(terms)
    del terms["joint_limit"]
    with pytest.raises(ValueError, match="but the task has"):
        check_task_reward(terms)
