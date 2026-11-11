"""Count the GPU->CPU syncs of an auto-resetting ``RLEnvBase.step`` of a registered task, on CUDA.

Run as ``python -m tests.support.step_syncs TASK [--wrap] [--num-envs N]`` from the repo root; prints
one line, ``RESULT {"task": ..., "wrap": ..., "counts": [...], "sites": {...}}``: the syncs of each of
5 steps after 5 of warm-up, four envs timing out in every one so the reset path runs, and the call
sites they came from (the innermost Eden or Genesis frame of each). A process of its own because the
test suite initialises Genesis on ``gs.cpu``, where nothing syncs, and Genesis cannot be
re-initialised in-process.

A task with a motion manager has its clock rewound before every step, so no clip ends by chance and
the count does not hang on where the seeded clips start. ``--wrap`` instead pushes two envs past
their clip's end in every step, so the motion-end resync (``resync_on_motion_end``) resamples and
teleports them; it refuses a task without that resync.
"""

from __future__ import annotations

import argparse
import json
import traceback
import warnings
from collections import Counter

#: The envs forced to time out, and the ones pushed past their clip's end with ``--wrap``.
RESET_ENVS, WRAP_ENVS = slice(0, 4), slice(8, 10)


def main(task: str, *, wrap: bool = False, num_envs: int = 16) -> None:
    import eden as en  # before genesis and torch (DDP import order)
    import genesis as gs

    en.init(backend=gs.gpu, logging_level="WARNING")
    import torch

    from eden.envs.base import env_class_for_config
    from eden.tasks import TASK_REGISTRY

    cfg = TASK_REGISTRY.build(task)
    cfg.env_options.num_envs = num_envs
    env = env_class_for_config(cfg).from_config(cfg)
    env.build()
    env.reset(seed=0)
    motion = getattr(env, "motion_manager", None)
    if wrap and not (motion is not None and motion._resync.enabled):
        raise SystemExit(f"--wrap needs a task whose motion manager resyncs at a clip's end; {task!r} does not")
    action = torch.zeros(num_envs, env.action_manager.total_action_dim, device=env.device)
    sites: Counter[str] = Counter()
    count = 0

    def record(message, *args, **kwargs) -> None:
        nonlocal count
        if "called a synchronizing CUDA operation" not in str(message):
            return
        count += 1
        frames = [f for f in traceback.extract_stack()[:-1] if "/eden/" in f.filename or "/genesis/" in f.filename]
        if frames:
            frame = frames[-1]
            sites[f"{frame.filename.rsplit('/', 2)[-2]}/{frame.filename.rsplit('/', 1)[-1]}:{frame.lineno}"] += 1

    def step() -> int:
        nonlocal count
        env.episode_length_buf[RESET_ENVS] = int(env.max_episode_length) - 1
        if motion is not None:
            motion._timestamp.zero_()
            if wrap:
                motion._timestamp[WRAP_ENVS] = 1e6  # past any clip's end
        count = 0
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("warn")
        try:
            env.step(action)
        finally:
            torch.cuda.set_sync_debug_mode("default")
        assert bool(env.reset_buf[RESET_ENVS].all()), "the forced timeouts did not reset"
        if wrap:
            assert bool((motion._timestamp[WRAP_ENVS] < 1e6).all()), "the envs pushed past their clip did not wrap"
        return count

    warnings.showwarning = record
    warnings.simplefilter("always")
    with torch.inference_mode():
        for _ in range(5):  # first-call reads (cached index ranges, lengths) are not per-step costs
            step()
        sites.clear()
        counts = [step() for _ in range(5)]
    result = {"task": task, "wrap": wrap, "counts": counts, "sites": dict(sites)}
    print("RESULT " + json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("task")
    parser.add_argument("--wrap", action="store_true", help="push two envs past their clip's end every step")
    parser.add_argument("--num-envs", type=int, default=16)
    args = parser.parse_args()
    main(args.task, wrap=args.wrap, num_envs=args.num_envs)
