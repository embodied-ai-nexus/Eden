r"""General-mimic — inference helper with ghost overlay + sequential motion exhaustion.

---
tier: project
capabilities: [task.mimic, robot.g1, dataset.lafan1]
hardware: cpu
optional_deps: []
run: uv run python examples/projects/mimic/general_mimic/main.py inference --model /path/to/general_mimic.pt
---

Plays every motion in the configured library exactly once, end-to-end (no
episode-length clipping), with a translucent G1 ghost rendering the
reference clip alongside the actor. Companion to ``eden train --task
general_mimic`` — training is unchanged, this script only adds the inference
visualisation features.

Usage::

    python examples/projects/mimic/general_mimic/main.py inference \\
        --model /path/to/general_mimic.pt \\
        --raw-data-path /path/to/LAFAN1_Retargeting_Dataset/g1 \\
        --include-pattern '^walk' --record --vis
"""

from __future__ import annotations

import argparse
import os

import eden as en  # must be imported before genesis to isolate GPUs for DDP
import genesis as gs
import torch
from tqdm import tqdm

from eden.datasets.base import DATASET_REGISTRY
from eden.envs.wrappers.rsl_rl_env import RslRlVecEnvWrapper, make_rsl_rl_runner
from eden.options import GHOST_DEFAULTS, EventManagerOptions
from eden.options.camera import CameraOptions, CamerasOptions
from eden.options.scenes.empty import EmptySceneOptions
from eden.options.robots.unitree_g1 import G1Dofs29
from eden.tasks import TASK_REGISTRY
from eden.managers.terms.events.motion import ReferenceStateInitialization


def _build_inference_config(args: argparse.Namespace):
    """GeneralMimicConfig with ghost entity, no clipping, no DR — eval-only."""
    # When the user didn't pin a processed_data_path, route the dataset cache
    # under the run's log dir. The cache must be wired in BEFORE the config
    # is built because ConfigurableOptions instances lock themselves after
    # construction, so we can't mutate ``processed_data_path`` later. The
    # preload step below uses this same path to materialise the cache before
    # the viewer env is built (see _preload_dataset for the GL-context
    # reasoning).
    cache_path = args.processed_data_path
    if cache_path is None:
        cache_path = os.path.join(en.log_dir, "motion_cache.pt")

    cfg = TASK_REGISTRY.build(
        "general_mimic",
        raw_data_path=args.raw_data_path,
        processed_data_path=cache_path,
        include_pattern=args.include_pattern,
        exclude_pattern=args.exclude_pattern,
        include_list_file=args.include_list_file,
        exclude_list_file=args.exclude_list_file,
        # The adaptive curriculum is a training-time concept; force it off
        # so motion sampling is uniform and our sequential override below
        # isn't fighting per-iter reweighting.
        adaptive_motion_weights=False,
    )

    cfg.env_options.num_envs = 1
    # Huge episode horizon so `time_out` never fires before `motion_ends`
    # — every clip plays to its natural end.
    cfg.env_options.episode_length_s = 1e9

    cfg.motion_options.default.adaptive_sampling = False
    cfg.motion_options.default.motion_start_from_middle_range = (0.0, 0.0)

    # Drop all randomization (push, COM shift, friction, default-pos jitter)
    # so the rollout is deterministic. Keep RSI so each motion still spawns
    # the robot at the clip's start pose — without it the policy would
    # always reset to the default joint configuration.
    cfg.event_options = EventManagerOptions(
        reset_from_motion=ReferenceStateInitialization.configure(
            mode=en.EventMode.RESET,
            entity_name="robot",
            base_joint_name="floating_base_joint",
            pose_range={},
            velocity_range={},
            dofs_position_range=(0.0, 0.0),
            priority=100,
        ),
    )

    # Add the ghost as a sibling entity in the scene. GHOST_DEFAULTS gives
    # it a fixed base, translucent material, and visual-only collision so
    # it floats over the actor without interfering with physics.
    cfg.scene_options = EmptySceneOptions(
        robot=G1Dofs29(),
        ghost_g1=G1Dofs29(**GHOST_DEFAULTS),
    )

    if args.record:
        cfg.cameras_options = CamerasOptions(rec=CameraOptions(follow_entity_name="robot"))
    if hasattr(cfg.runner_options, "logger"):
        cfg.runner_options.logger = "tensorboard"
    if args.cpu:
        cfg.runner_options.device = "cpu"

    return cfg


def _install_sequential_motion_sampler(motion_term, total_motions: int) -> list[int]:
    """Replace random motion sampling with a deterministic 0,1,2,... walk.

    Returns a one-element list holding the count of motions consumed so the
    caller can detect when the library is exhausted (cleaner than juggling a
    closure variable in the main loop). Wraps the original sampler so the
    bin-curriculum start-time bookkeeping still runs — we only override the
    *which-motion* choice, not the rest of the reset path.
    """
    consumed = [0]
    original = motion_term._sample_assigned_env_starting_stub

    def sequential(envs_idx=None):
        original(envs_idx)
        assigned = motion_term.envs_idx_to_assigned_ids(envs_idx)
        for i in range(len(assigned)):
            motion_id = consumed[0] % total_motions
            motion_term._assigned_env_motion_selection[assigned[i]] = motion_id
            motion_term._motion_buffer_start_time_s[assigned[i]] = 0.0
            consumed[0] += 1

    motion_term._sample_assigned_env_starting_stub = sequential
    return consumed


def _push_ghost(env, ghost_entity_name: str) -> None:
    """Snap the ghost to the current motion-reference pose."""
    ref = env.unwrapped.motion_manager.reference_frame["robot"]
    ghost = env.unwrapped.entities[ghost_entity_name]
    # MotionReferenceData fields carry a (num_envs, num_frames, ...) layout;
    # we're at num_envs=1 / num_frames=1 so frame 0 is the current target.
    ghost.set_pos(ref.base_pos_w[:, 0])
    ghost.set_quat(ref.base_quat_w[:, 0])
    ghost.set_dofs_pos(ref.dofs_pos[:, 0])


def _preload_dataset(cfg, device: torch.device) -> None:
    """Materialize the motion cache before the viewer env is constructed.

    Datasets that lack pre-computed link velocities (e.g. LAFAN1) call
    ``DatasetBase._forward_velocity_kinematics`` from inside ``prepare()``,
    which spawns a temporary headless ``gs.Scene`` to forward-integrate FK.
    If that runs *after* the inference env's pyrender viewer thread is
    live, the second scene's build pulls the GL context out from under the
    viewer and the next help-text draw raises ``OpenGL.error.Error: no
    valid context``. Instantiating the dataset here forces ``prepare()`` to
    run before any viewer exists, and the resulting cache lets the
    follow-up ``MotionManager.__init__`` inside ``env.build()`` hit the
    fast cache path instead of re-running ``prepare()``.
    """
    ds_options = cfg.motion_options.default.dataset_options
    ds_cls = DATASET_REGISTRY.get(ds_options.name)
    en.logger.info(f"Preloading dataset (cache: {ds_options.processed_data_path})")
    ds = ds_cls(ds_options, device)
    del ds  # release; the cache is on disk and the env will mmap it back


def inference(args: argparse.Namespace) -> None:
    if not gs._initialized:
        en.init(
            backend=gs.cpu if args.cpu else gs.gpu,
            log_root_path="logs/general_mimic",
        )

    cfg = _build_inference_config(args)
    cfg.save_as_json()

    device = torch.device("cpu" if args.cpu else "cuda")
    _preload_dataset(cfg, device)

    env = RslRlVecEnvWrapper.from_config(cfg, show_viewer=args.vis, eval_mode=True)
    runner = make_rsl_rl_runner(env, checkpoint=args.model)
    policy = runner.get_inference_policy(device=env.device)

    motion_term = env.unwrapped.motion_manager._terms["default"]
    total_motions = motion_term.num_trajectories
    en.logger.info(f"Sequential playback over {total_motions} motion(s)")
    consumed = _install_sequential_motion_sampler(motion_term, total_motions)

    rec = env.unwrapped.cameras["rec"] if args.record else None
    if rec is not None:
        out_dir = os.path.dirname(args.model) or "."
        out_path = os.path.join(out_dir, "general_mimic_inference.mp4")
        rec.start_recording(out_path)

    # reset() invokes the patched sampler once → motion 0 is now assigned.
    obs, _ = env.reset()
    _push_ghost(env, "ghost_g1")

    completed = 0
    pbar = tqdm(total=total_motions, desc="motions")
    # Hard step cap so a stuck policy can't run forever. ``--steps`` is the
    # absolute ceiling; under normal operation ``motion_ends`` fires well
    # before this — the per-clip length is bounded by ``data_source``.
    for _ in range(args.steps):
        with torch.no_grad():
            action = policy(obs)
        obs, _, dones, _ = env.step(action)
        _push_ghost(env, "ghost_g1")
        if rec is not None:
            rec.render_rgb()

        if bool(dones[0].item()):
            completed += 1
            pbar.update(1)
            # The patched sampler has already advanced the assignment for
            # the post-reset step. ``consumed[0]`` counts samples *served*,
            # so once it reaches ``total_motions`` every clip has been
            # started — let the in-flight one finish, then stop.
            if completed >= total_motions:
                break
    pbar.close()

    if rec is not None:
        rec.stop_recording()
        en.logger.info(f"Recording written to {out_path}")

    en.logger.info(f"Played {completed}/{total_motions} motions (sampler served {consumed[0]})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="General-mimic inference visualizer.")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    infer = subparsers.add_parser("inference", help="Run inference with ghost overlay across every motion.")
    infer.add_argument("--model", type=str, required=True, help="Path to a trained rsl_rl checkpoint.")
    infer.add_argument(
        "--raw-data-path",
        type=str,
        default=None,
        help="Directory with retargeted-G1 motion files. None → HF auto-download.",
    )
    infer.add_argument("--processed-data-path", type=str, default=None, help="Optional .pt cache path.")
    infer.add_argument("--include-pattern", type=str, default=None, help="Regex on basenames; keep matching clips.")
    infer.add_argument("--exclude-pattern", type=str, default=None, help="Regex on basenames; drop matching clips.")
    infer.add_argument(
        "--include-list-file",
        type=str,
        default=None,
        help="Path to a YAML/JSON file enumerating motion files to keep (basename-stem match).",
    )
    infer.add_argument(
        "--exclude-list-file",
        type=str,
        default=None,
        help="Path to a YAML/JSON file enumerating motion files to drop (basename-stem match).",
    )
    infer.add_argument("-r", "--record", action="store_true", help="Record an mp4 alongside the checkpoint.")
    infer.add_argument("-v", "--vis", action="store_true", help="Show the Genesis viewer window.")
    infer.add_argument("-c", "--cpu", action="store_true", help="Use CPU backend.")
    infer.add_argument(
        "-s",
        "--steps",
        type=int,
        default=200_000,
        help="Hard ceiling on env steps (motion_ends usually trips well before this).",
    )

    args = parser.parse_args()
    if args.command == "inference":
        inference(args)
    else:
        parser.print_help()
