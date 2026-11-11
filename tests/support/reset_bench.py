"""Time ``MimicRLEnvBase._reset_idx`` on CUDA for a G1 plus ``M`` free boxes, each seeded by its own RSI term.

Run as ``python -m tests.support.reset_bench M [NUM_ENVS] [NUM_RESET]`` from the repo root; prints one line,
``RESULT {"entities": ..., "reset_idx_ms": ...}``, the median of 200 resets of ``NUM_RESET`` random envs
after 20 of warm-up, each timed with ``torch.cuda.synchronize()`` on both sides. A process of its own because
the test suite initialises Genesis on ``gs.cpu`` and Genesis cannot be re-initialised in-process.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path


def main(num_boxes: int, num_envs: int = 2048, num_reset: int = 26) -> None:
    import eden as en  # before genesis and torch (DDP import order)
    import genesis as gs

    en.init(backend=gs.gpu, logging_level="WARNING")
    import torch

    from eden.managers.terms.events.motion import ObjectReferenceStateInitialization, ReferenceStateInitialization
    from eden.options.entities import BoxOptions
    from eden.options.managers.events import EventManagerOptions
    from eden.options.robots.unitree_g1 import G1Dofs29
    from tests.support.motion import (
        OBJECT_LINKS,
        ROBOT_DOFS,
        ROBOT_LINKS,
        build_mimic_env,
        manager_options,
        term_options,
        write_clip,
    )

    boxes = [f"box{i}" for i in range(num_boxes)]
    sequences = {"robot": (ROBOT_DOFS, ROBOT_LINKS)} | {box: ([], OBJECT_LINKS) for box in boxes}
    root = Path(tempfile.mkdtemp(prefix="reset_bench"))
    for clip, frames in enumerate([20, 30]):
        write_clip(root / f"clip_{clip:02d}.npz", frames, clip=clip, sequences=sequences)
    reset = en.EventMode.RESET
    events = EventManagerOptions(
        reset_robot=ReferenceStateInitialization.configure(mode=reset, entity_name="robot"),
        **{f"reset_{box}": ObjectReferenceStateInitialization.configure(mode=reset, entity_name=box) for box in boxes},
    )
    scene = {"robot": G1Dofs29()} | {
        box: BoxOptions(size=(0.1, 0.1, 0.1), default_root_pos=(1.0 + i, 0.0, 0.05)) for i, box in enumerate(boxes)
    }
    env = build_mimic_env(
        root,
        num_envs=num_envs,
        scene=scene,
        event_options=events,
        motion_options=manager_options(terms={"default": term_options(root, sequences=sequences)}),
    )

    generator = torch.Generator(device="cpu").manual_seed(0)

    def median_ms(repeats: int) -> float:
        times = []
        for _ in range(repeats):
            mask = torch.zeros(num_envs, dtype=torch.bool)
            mask[torch.randperm(num_envs, generator=generator)[:num_reset]] = True
            mask = mask.to(env.device)
            torch.cuda.synchronize()
            start = time.perf_counter()
            env._reset_idx(mask)
            env._dispatch_post_reset()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
        return sorted(times)[repeats // 2] * 1e3

    median_ms(20)
    print("RESULT " + json.dumps({"entities": 1 + num_boxes, "reset_idx_ms": median_ms(200)}), flush=True)


if __name__ == "__main__":
    main(*(int(arg) for arg in sys.argv[1:]))
