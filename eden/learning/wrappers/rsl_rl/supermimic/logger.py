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
            from torch.utils.tensorboard.writer import FileWriter

            self._writer = FileWriter(self.log_dir, flush_secs=10)

    def process_env_step(self, rewards: torch.Tensor, dones: torch.Tensor, extras: dict) -> None:
        if self.disable_logs:
            return
        if extras.get("log"):
            self._episode_logs.append(extras["log"])
        self._reward_sum += rewards
        self._length += 1
        done = dones.bool()
        # One host read per step: every env's sum, length and done flag, filtered on the host.
        reward_sum, length, ended = torch.stack([self._reward_sum, self._length, done.float()]).cpu()
        ended = ended.bool()
        if ended.any():
            self.rewbuffer.extend(reward_sum[ended].tolist())
            self.lenbuffer.extend(length[ended].tolist())
            self._reward_sum.masked_fill_(done, 0)
            self._length.masked_fill_(done, 0)

    def log(
        self,
        it: int,
        total_it: int,
        steps: int,
        collect_time: float,
        learn_time: float,
        losses: dict,
        normalized_return: float | None = None,
    ) -> None:
        """Write one iteration. ``normalized_return`` is the mean, over the episodes that ended in it, of the
        episode's summed reward over its length times the largest step reward; ``None`` if no episode ended.
        """
        if self.disable_logs:
            return
        self.tot_timesteps += steps
        self.tot_time += collect_time + learn_time
        # One host read for every loss and episode statistic: each is a 0-dim tensor on the device.
        on_device = {f"Loss/{k}": torch.as_tensor(v, dtype=torch.float) for k, v in losses.items()}
        for key in dict.fromkeys(k for log in self._episode_logs for k in log):
            values = [
                torch.as_tensor(log[key], dtype=torch.float).reshape(-1) for log in self._episode_logs if key in log
            ]
            on_device[key if "/" in key else f"Episode/{key}"] = torch.cat(values).mean()
        device = self._reward_sum.device
        read = (
            torch.stack([v.detach().reshape(()).to(device) for v in on_device.values()]).tolist() if on_device else []
        )
        scalars = dict(zip(on_device, read, strict=True))
        self._episode_logs.clear()
        if self.rewbuffer:
            scalars["Train/mean_reward"] = statistics.mean(self.rewbuffer)
            scalars["Train/mean_episode_length"] = statistics.mean(self.lenbuffer)
        if normalized_return is not None:
            scalars["Train/mean_normalized_return"] = normalized_return
        fps = int(steps / max(collect_time + learn_time, 1e-9))
        scalars |= {"Perf/total_fps": fps, "Perf/collection_time": collect_time, "Perf/learning_time": learn_time}
        if self._wandb is not None:
            self._wandb.log(scalars, step=it)
        else:
            self._write_scalars(scalars, it)
        width, pad = 80, 40
        lines = [f"{'#' * width}", f"\033[1m{f' Learning iteration {it}/{total_it} '.center(width)}\033[0m", ""]
        lines += [f"{f'{key}:':>{pad}} {value:.4f}" for key, value in scalars.items()]
        print("\n".join(lines))

    def _write_scalars(self, scalars: dict[str, float], step: int) -> None:
        """Write an iteration's scalars to TensorBoard as one event holding every tag. ``SummaryWriter.add_scalar``
        writes one event per scalar, each appended with an open and a close of the file, which is slow on a network
        file system.
        """
        from tensorboard.compat.proto.summary_pb2 import Summary
        from torch.utils.tensorboard.summary import scalar

        values = [value for key, x in scalars.items() for value in scalar(key, x).value]
        self._writer.add_summary(Summary(value=values), step)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
        # A run the caller started before the runner (``wandb.run``) is theirs to finish.
        if self._wandb is not None and self._owns_wandb_run:
            self._wandb.finish()
        self._writer = self._wandb = None
