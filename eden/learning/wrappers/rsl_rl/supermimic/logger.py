"""The run's console, TensorBoard and W&B output."""

from __future__ import annotations

import statistics
from collections import deque
from typing import TYPE_CHECKING

import torch

import eden as en

if TYPE_CHECKING:
    from eden.learning.wrappers.rsl_rl.supermimic.options import SuperMimicRunnerOptions


class RunLogger:
    """Episode statistics of the simulator and the learners' losses, to the console and TensorBoard or W&B.

    Holds what Eden's runner mixins read: ``log_dir``, ``disable_logs``, ``rewbuffer``, ``lenbuffer``,
    ``tot_timesteps`` and ``tot_time``. The writer opens on :meth:`open`, which only a training run calls.
    """

    def __init__(self, log_dir: str | None, options: SuperMimicRunnerOptions, num_envs: int, device: torch.device):
        self.log_dir = log_dir
        self._options = options
        self.disable_logs = log_dir is None or en.distributed.get_global_rank() != 0
        self.rewbuffer: deque[float] = deque(maxlen=100)
        self.lenbuffer: deque[float] = deque(maxlen=100)
        self.tot_timesteps = 0
        self.tot_time = 0.0
        self._reward_sum = torch.zeros(num_envs, device=device)
        self._length = torch.zeros(num_envs, device=device)
        self._episode_logs: list[dict] = []
        self._writer = None
        self._wandb = None
        self._owns_wandb_run = False

    def open(self) -> None:
        """Open the W&B run or the TensorBoard writer, once."""
        if self.disable_logs or self._writer is not None or self._wandb is not None:
            return
        options = self._options
        if options.logger == "wandb":
            import wandb

            run_name = options.run_name or None
            self._owns_wandb_run = wandb.run is None
            self._wandb = wandb.run or wandb.init(
                project=options.wandb_project, name=run_name, dir=self.log_dir, config=options.dict()
            )
        else:
            from torch.utils.tensorboard import SummaryWriter

            self._writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)

    def process_env_step(self, rewards: torch.Tensor, dones: torch.Tensor, extras: dict) -> None:
        if self.disable_logs:
            return
        if extras.get("log"):
            self._episode_logs.append(extras["log"])
        self._reward_sum += rewards
        self._length += 1
        done = dones.nonzero(as_tuple=False)[:, 0]
        if done.numel():
            self.rewbuffer.extend(self._reward_sum[done].tolist())
            self.lenbuffer.extend(self._length[done].tolist())
            self._reward_sum[done] = 0
            self._length[done] = 0

    def log(self, it: int, total_it: int, steps: int, collect_time: float, learn_time: float, losses: dict) -> None:
        if self.disable_logs:
            return
        self.tot_timesteps += steps
        self.tot_time += collect_time + learn_time
        scalars = {f"Loss/{k}": float(v) for k, v in losses.items()}
        for key in dict.fromkeys(k for log in self._episode_logs for k in log):
            values = [
                torch.as_tensor(log[key], dtype=torch.float).reshape(-1) for log in self._episode_logs if key in log
            ]
            scalars[key if "/" in key else f"Episode/{key}"] = float(torch.cat(values).mean())
        self._episode_logs.clear()
        if self.rewbuffer:
            scalars["Train/mean_reward"] = statistics.mean(self.rewbuffer)
            scalars["Train/mean_episode_length"] = statistics.mean(self.lenbuffer)
        fps = int(steps / max(collect_time + learn_time, 1e-9))
        scalars |= {"Perf/total_fps": fps, "Perf/collection_time": collect_time, "Perf/learning_time": learn_time}
        if self._wandb is not None:
            self._wandb.log(scalars, step=it)
        else:
            for key, value in scalars.items():
                self._writer.add_scalar(key, value, it)
        width, pad = 80, 40
        lines = [f"{'#' * width}", f"\033[1m{f' Learning iteration {it}/{total_it} '.center(width)}\033[0m", ""]
        lines += [f"{f'{key}:':>{pad}} {value:.4f}" for key, value in scalars.items()]
        print("\n".join(lines))

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
        # A run the caller started before the runner (``wandb.run``) is theirs to finish.
        if self._wandb is not None and self._owns_wandb_run:
            self._wandb.finish()
        self._writer = self._wandb = None
