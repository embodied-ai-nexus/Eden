"""Random-seed utilities."""

import os
import random
from typing import Annotated

import numpy as np
import torch
from pydantic import AfterValidator


def set_random_seed(seed: int, torch_deterministic: bool = False) -> None:
    """Seed Python's ``random``, numpy and torch (every device), and set ``PYTHONHASHSEED``.

    ``PYTHONHASHSEED`` only reaches subprocesses started afterwards; this interpreter's hash seed is fixed at
    startup. ``torch_deterministic`` turns on ``torch.use_deterministic_algorithms(True, warn_only=True)``.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    # Ref: https://docs.pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(seed)  # Seed RNG for all devices.
    # torch.manual_seed above already covers CUDA; the explicit calls are belt-and-braces, gated on
    # availability so CPU-only / MPS builds stay portable.
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if torch_deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def validate_seed(seed: int) -> int:
    """Reject a negative seed, including a ``-1`` meant as "draw a random seed".

    Eden has no such sentinel, and it could not honour one: :func:`eden.init` seeds numpy
    through ``gs.init(seed=LOCAL_RANK)`` before any config is read, so a draw would come off a
    stream just seeded to a constant and return the *same* value on every run. Pass an
    explicit seed (``--seed $RANDOM`` in a sweep script) so the value is recorded rather than
    invented.

    Parameters
    ----------
    seed : int
        The configured seed.

    Returns
    -------
    int
        The same seed, as an ``int``.

    Raises
    ------
    ValueError
        If ``seed`` is negative.
    """
    if seed < 0:
        raise ValueError(
            f"seed must be >= 0, got {seed}. Eden has no `-1` 'draw a random seed' sentinel: "
            "eden.init() seeds numpy via gs.init() before any config is read, so such a draw would "
            "return the same value on every run. Pass an explicit seed instead — "
            "e.g. `--seed $RANDOM` if you want a fresh one per launch."
        )
    return int(seed)


#: A validated non-negative seed. Use it for any config field naming a seed so a stale
#: ``-1`` fails at config load with :func:`validate_seed`'s message, not silently at run time.
Seed = Annotated[int, AfterValidator(validate_seed)]


def apply_run_seed(seed: int) -> int:
    """Apply a run's seed to every RNG — the single owner of run seeding.

    Call it once, *before* the environment is constructed, so the build, its first reset and
    every domain-randomization draw come from the seeded RNGs. The training wrappers'
    ``from_config`` do exactly that, and rsl-rl-lib never applies ``runner_options.seed``
    itself.

    Under DDP each rank offsets the configured base by its global rank, so ranks sample
    distinct env randomization and action noise instead of duplicating rank 0's rollout.

    ``seed`` is the run's **base**, and stays the base: the applied value is returned and
    logged, never written back into the config. A config field that is both the input and
    the output of this function would compound the offset every time it ran — rank ``r``
    getting ``base + 2r`` on the second call, invisibly, since rank 0 is unaffected. Keeping
    it read-only also keeps one meaning for ``seed`` everywhere: ``config.json`` and
    ``run.json`` record the base, and ``run.json``'s ``world_size`` pins the rest
    (rank ``r`` used ``seed + r``), so the base plus the topology reproduces every rank.

    Parameters
    ----------
    seed : int
        The run's configured base seed. Must be non-negative.

    Returns
    -------
    int
        The seed applied on this rank (``base + global_rank``).

    Raises
    ------
    ValueError
        If ``seed`` is negative (see :func:`validate_seed`).
    """
    import eden as en
    from eden.utils import distributed as eden_dist

    base = validate_seed(seed)
    rank = eden_dist.get_global_rank()
    applied = base + rank
    en.logger.info(f"Setting seed: {applied}" + (f" (base {base} + rank {rank})" if rank else ""))
    set_random_seed(applied)
    return applied
