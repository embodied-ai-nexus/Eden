"""Per-env circular replay buffer with episode-aware window sampling."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch
from tensordict import TensorDict


@dataclass(frozen=True)
class FieldSpec:
    """A stored field: ``dim`` 0 for a scalar per step, else a vector of that width."""

    dim: int = 0
    dtype: torch.dtype = torch.float32

    def shape(self, num_envs: int, buffer_size: int) -> tuple[int, ...]:
        return (num_envs, buffer_size, self.dim) if self.dim else (num_envs, buffer_size)


class ReplayBuffer:
    """``buffer_size`` steps per env, as a ``(num_envs, buffer_size)`` :class:`TensorDict`.

    The schema needs boolean ``terminated`` and ``truncated`` fields: a stored step with either ends its episode, and
    sampled windows never cross that end.
    """

    def __init__(self, schema: dict[str, FieldSpec], buffer_size: int, num_envs: int, device: str | torch.device):
        if not {"terminated", "truncated"} <= schema.keys():
            raise ValueError(f"the replay buffer schema needs 'terminated' and 'truncated', got {sorted(schema)}.")
        self.buffer_size = buffer_size
        self.num_envs = num_envs
        self.device = device
        self.num_transitions = 0
        self._step = 0
        #: Writes so far, and per sequence length the valid windows with the stamp they were counted at.
        self._writes = 0
        self._windows: dict[int, tuple[int, torch.Tensor, torch.Tensor, bool]] = {}
        self._storage = TensorDict(
            {
                name: torch.zeros(s.shape(num_envs, buffer_size), dtype=s.dtype, device=device)
                for name, s in schema.items()
            },
            batch_size=[num_envs, buffer_size],
        )

    def insert(self, step: TensorDict) -> None:
        """Store one step of every env."""
        if set(step.keys()) != set(self._storage.keys()):
            raise ValueError(
                f"step fields {sorted(step.keys())} differ from the schema {sorted(self._storage.keys())}."
            )
        self._storage[:, self._step] = step
        self._writes += 1
        self._step = (self._step + 1) % self.buffer_size
        self.num_transitions = min(self.buffer_size, self.num_transitions + 1)

    def end_episodes(self) -> None:
        """Mark the newest stored step of every env as truncating its episode, so no window crosses it.

        For a caller that resets the envs between two stored steps.
        """
        if self.num_transitions:
            self._storage["truncated"][:, (self._step - 1) % self.buffer_size] = True
            self._writes += 1

    def _base(self) -> int:
        """The slot of the oldest stored step: slot = (base + age) % buffer_size."""
        return self._step if self.num_transitions == self.buffer_size else 0

    def _valid_windows(self, sequence_length: int) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """The windows of ``sequence_length`` steps with no episode end before their last step: each env's count
        ``(num_envs,)``, the cumulative counts by age of the first step, flattened and offset per env for
        :meth:`sample`'s search, and whether there is any window. Counted once per write.
        """
        L, E, T, n = sequence_length, self.num_envs, self.buffer_size, self.num_transitions
        if L > T:
            raise ValueError(f"sequence_length {L} exceeds buffer_size {T}.")
        # A window of one step holds no step before its last, so the one-step windows change with the number of
        # stored steps alone: once the buffer is full they are counted for good.
        stamp = n if L == 1 else self._writes
        cached = self._windows.get(L)
        if cached is not None and cached[0] == stamp:
            return cached[1:]
        ages = torch.arange(T, device=self.device)
        ended = (self._storage["terminated"] | self._storage["truncated"])[:, (self._base() + ages) % T].int()
        # Window [a, a+L-1] has no end before its last step iff the end count over [a, a+L-2] is zero.
        ends = ended.cumsum(1, dtype=torch.int32)
        c = torch.cat([ended.new_zeros(E, 1), ends, ended.new_zeros(E, L - 1)], dim=1)
        valid = (c[:, L - 1 : L - 1 + T] == c[:, :T]) & (ages + L - 1 < n)
        csum = valid.cumsum(dim=1, dtype=torch.int32)
        total = csum[:, -1].clone()
        # Env e's counts lie in [0, T]: shifted by e (T + 1) they sort after env e - 1's and before env e + 1's.
        flat = (csum + torch.arange(E, device=self.device)[:, None] * (T + 1)).view(-1)
        self._windows[L] = (stamp, total, flat, bool((total > 0).any()))
        return self._windows[L][1:]

    def has_window(self, sequence_length: int) -> bool:
        """Whether :meth:`sample` has a window of ``sequence_length`` steps to draw."""
        return self._valid_windows(sequence_length)[2]

    def sample(
        self,
        sequence_length: int,
        batch_size: int,
        fields: Sequence[str] | None = None,
        num_batches: int | None = None,
    ) -> TensorDict:
        """``(batch_size, sequence_length)`` windows of consecutive steps that no episode end interrupts.

        A window may end on a terminating or truncating step but holds none before its last step. An env is drawn
        uniformly among those with a valid window, and a window uniformly within its env, so a window of an env
        with few valid windows is drawn more often than one of an env with many. ``fields`` names the stored fields
        to return, every one by default. ``num_batches`` returns ``(num_batches, batch_size, sequence_length)``, the
        windows that many successive calls would draw.
        """
        L, B, T, n = sequence_length, batch_size, self.buffer_size, self.num_transitions
        base, dev = self._base(), self.device
        total, flat, any_window = self._valid_windows(L)
        if not any_window:
            raise RuntimeError(
                f"no window of {L} steps without an episode end among {n} stored steps. Check has_window({L}) "
                "before sampling."
            )
        weights = (total > 0).float()
        # One batch after the other, so the random stream is that of one call per batch.
        draws = [
            (torch.multinomial(weights, B, replacement=True), torch.rand(B, device=dev))
            for _ in range(1 if num_batches is None else num_batches)
        ]
        env_idx, uniform = (torch.cat(column) if len(draws) > 1 else column[0] for column in zip(*draws, strict=True))
        u = (uniform * total[env_idx].float()).long()
        # The first age of env_idx whose cumulative count exceeds u, by one search of the flattened counts.
        start = torch.searchsorted(flat, u + env_idx * (T + 1), right=True) - env_idx * T
        slots = (base + start[:, None] + torch.arange(L, device=dev)) % T
        storage = self._storage if fields is None else self._storage.select(*fields)
        windows = storage[env_idx[:, None], slots]
        return windows if num_batches is None else windows.reshape(num_batches, B, L)

    def sample_causal(self, sequence_length: int, batch_size: int) -> tuple[TensorDict, torch.Tensor]:
        """Windows of ``sequence_length`` steps ending at uniformly drawn stored steps, and their padding mask.

        The newest step is drawn as in a one-step sample. Its predecessors fill the earlier slots; slots before its
        episode's start, or before the oldest stored step, repeat the episode's first step, and the returned mask
        is ``True`` there.
        """
        L, B, E, T, n = sequence_length, batch_size, self.num_envs, self.buffer_size, self.num_transitions
        if L > T:
            raise ValueError(f"sequence_length {L} exceeds buffer_size {T}.")
        if n < 1:
            raise RuntimeError("sample_causal on an empty replay buffer.")
        base, dev = self._base(), self.device
        env_idx = torch.randint(E, (B,), device=dev)
        current = torch.randint(n, (B,), device=dev)
        ages = current[:, None] + (torch.arange(L, device=dev) - (L - 1))
        ended = (self._storage["terminated"] | self._storage["truncated"])[
            env_idx[:, None], (base + ages.clamp(min=0)) % T
        ]
        valid = (ages >= 0) & ~ended.bool()
        valid[:, -1] = True
        reachable = torch.cummin(valid.flip(-1).int(), dim=-1).values.flip(-1).bool()
        first = torch.where(reachable, ages, current[:, None]).amin(dim=-1, keepdim=True)
        slots = (base + torch.where(reachable, ages, first)) % T
        return self._storage[env_idx[:, None], slots], ~reachable
