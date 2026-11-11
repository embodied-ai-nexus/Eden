"""CUDA-resident stand-ins for the command terms' GPU-sync budget tests, and nothing else.

The suite runs Genesis on the CPU, so no built env lives on CUDA (``docs/guides/testing.md``). What the
terms compute is tested against real envs in the test modules.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch


def _cuda_env(num_envs: int, **fields) -> SimpleNamespace:
    """The env surface a command term reads, on CUDA; skips without CUDA."""
    if not torch.cuda.is_available():
        pytest.skip("counting GPU->CPU syncs needs CUDA")
    return SimpleNamespace(num_envs=num_envs, device=torch.device("cuda"), dt=0.02, entities={}, **fields)


class _CudaRobot:
    """Takes the two reset-teleport writes and discards them."""

    def __init__(self, num_dofs: int):
        self.soft_dofs_pos_limits = torch.tensor([[[-10.0, 10.0]] * num_dofs], device="cuda")

    def set_qpos(self, *args, **kwargs):
        pass

    def set_dofs_vel(self, *args, **kwargs):
        pass


@pytest.fixture
def cuda_motion_command():
    """A ``MotionCommand`` in its post-``build()`` state on CUDA: 4 envs, one 50-frame random clip, 5 bins."""
    num_envs, frames, links, dofs, bins = 4, 50, 2, 3, 5
    env = _cuda_env(
        num_envs,
        episode_length_buf=torch.ones(num_envs, dtype=torch.long, device="cuda"),
        termination_manager=SimpleNamespace(terminated=torch.zeros(num_envs, dtype=torch.bool, device="cuda")),
    )
    from eden.managers.terms.commands.mimic import MotionCommand

    term = MotionCommand(
        env,
        MotionCommand.configure(entity_name="robot", sampling_mode="adaptive", resampling_time_range=(1.0e9, 1.0e9)),
    )
    quat = torch.rand(1, frames, links, 4, device="cuda")
    term.motion = SimpleNamespace(
        dofs_pos=torch.rand(1, frames, dofs, device="cuda"),
        dofs_vel=torch.rand(1, frames, dofs, device="cuda"),
        links_pos_w=torch.rand(1, frames, links, 3, device="cuda"),
        links_quat_w=quat / quat.norm(dim=-1, keepdim=True),
        links_lin_vel_w=torch.rand(1, frames, links, 3, device="cuda"),
        links_ang_vel_w=torch.rand(1, frames, links, 3, device="cuda"),
        time_step_total=frames,
        frame_total=torch.tensor([frames], device="cuda"),
    )
    term.robot = _CudaRobot(dofs)
    term.num_clips = 1
    term.base_link_index = 0
    term.motion_id = torch.zeros(num_envs, dtype=torch.long, device="cuda")
    term.frame_total = term.motion.frame_total[term.motion_id]
    term.time_steps = torch.zeros(num_envs, dtype=torch.long, device="cuda")
    # Genesis' zero-copy path: every row, the mask, and a range on each DOF axis.
    term._reset_through_mask = True
    term._reset_dofs_idx = range(6 + dofs)
    term._reset_qs_idx = range(7 + dofs)
    term.bin_count = bins
    term.bin_failed_count = torch.tensor([1.0, 0.0, 2.0, 0.0, 1.0], device="cuda")
    term.bin_starts_count = torch.full((bins,), 4.0, device="cuda")
    term._current_bin_failed = torch.zeros(bins, device="cuda")
    term._current_bin_starts = torch.zeros(bins, device="cuda")
    term.start_bin = torch.zeros(num_envs, dtype=torch.long, device="cuda")
    term.kernel = torch.ones(1, device="cuda")
    term._init_sampling_stats()
    return term


class _HeadingRobot:
    """Reports a zero heading: the one entity read a velocity command's heading control makes."""

    def __init__(self, num_envs: int):
        self.heading = torch.zeros(num_envs, device="cuda")

    def get_heading(self, envs_idx=None):
        return self.heading


@pytest.fixture
def cuda_env():
    """The four-env CUDA env, with a ``robot`` that reports a heading (for ``UniformVelocityCommand``)."""
    env = _cuda_env(4)
    env.entities["robot"] = _HeadingRobot(4)
    return env
