"""``manipulation_lift_box`` restored from a reset-state buffer captured off its own resets (``--reset_buffer``).

The full loop on the real task: the Droid arm (fixed base), the welded Robotiq 2F85 (closed loop, passive
followers outside ``dofs_name``) and the free cube are captured mid-episode, saved, and restored by a second
build of the task with the mod.
"""

from __future__ import annotations

import pytest
import torch

from eden.tasks import TASK_REGISTRY
from eden.utils.buffers.reset_states import ResetStateBuffer, capture_reset_states
from tests.support.envs import build_env

pytestmark = pytest.mark.requires_genesis

NUM_ENVS = 2
RESTORED = ("robot", "gripper", "cube")


def _lift_box(**mods):
    return TASK_REGISTRY.build("manipulation_lift_box", **mods).with_overrides_from_dict(
        {"env_options": {"num_envs": NUM_ENVS}}
    )


def _raw_state(env, name: str) -> tuple[torch.Tensor, torch.Tensor]:
    entity = env.entities[name]
    return (
        entity.get_qpos(qs_idx_local=list(range(entity.n_qs))).clone().cpu(),
        entity.get_dofs_vel(dofs_idx_local=list(range(entity.n_dofs))).clone().cpu(),
    )


@pytest.fixture(scope="module")
def captured(tmp_path_factory):
    """Mid-episode states of the stock task: the arm moved by random actions, the gripper closing."""
    env = build_env(_lift_box(), reset=True)
    torch.manual_seed(0)
    for _ in range(10):
        env.step(torch.rand(NUM_ENVS, env.action_manager.total_action_dim, device=env.device) * 2 - 1)
    path = tmp_path_factory.mktemp("lift_reset_buffer") / "states.pt"
    capture_reset_states(env, RESTORED).save(path)
    return path, {name: _raw_state(env, name) for name in RESTORED}


def test_lift_box_resets_onto_the_captured_states(captured):
    path, before = captured
    env = build_env(_lift_box(reset_buffer=str(path)), reset=True)
    rows = env.event_manager.get_term("restore_reset_state").current_config.cpu()
    assert ResetStateBuffer.load(path).entities["gripper"].joints_name  # the gripper's passive joints are stored
    for name in RESTORED:
        qpos, vel = _raw_state(env, name)
        if name == "cube":  # the free root's quaternion is equal up to sign
            sign = torch.where((qpos[:, 3:7] * before[name][0][rows, 3:7]).sum(-1, keepdim=True) < 0, -1.0, 1.0)
            qpos = torch.cat([qpos[:, :3], qpos[:, 3:7] * sign, qpos[:, 7:]], dim=1)
        torch.testing.assert_close(qpos, before[name][0][rows], atol=1e-5, rtol=0, msg=f"{name} qpos")
        torch.testing.assert_close(vel, before[name][1][rows], atol=1e-4, rtol=0, msg=f"{name} dofs velocity")
