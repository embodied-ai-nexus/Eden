"""
MimicRLEnvBase BeyondMimic — motion imitation task.

---
tier: project
capabilities: [task.mimic, robot.g1]
hardware: cpu
optional_deps: []
run: uv run python examples/projects/mimic/mimic_beyondmimic/main.py train
---

Single-GPU training::

    python main.py train --num-envs 4096

Inference::

    python main.py inference --model <checkpoint.pt>

"""

import argparse
import os

import eden as en  # must be before genesis to isolate GPUs for DDP
import genesis as gs
import torch
from tqdm import tqdm
from eden.envs.wrappers.rsl_rl_env import RslRlVecEnvWrapper, make_rsl_rl_runner
from eden.managers.terms.events.motion import ReferenceStateInitialization
from eden.options.camera import CameraOptions
from eden.options import GHOST_DEFAULTS, EventManagerOptions
from eden.tasks import TASK_REGISTRY
from eden.tasks.beyondmimic.config import BeyondMimicConfig
from eden.utils.assets import find_latest_checkpoint

_SAMPLE_DIR = os.path.join(os.path.dirname(__file__), "../beyondmimic/sample/")


def _make_config(args: argparse.Namespace) -> BeyondMimicConfig:
    """Build a fully-configured BeyondMimicConfig.

    Routes through ``TASK_REGISTRY.build`` so the registered
    ``BeyondMimicDataMod`` runs and populates ``motion_options`` from the
    requested data paths. Equivalent to::

        eden train --task beyondmimic --raw_data_path <args.data_dir>

    ``processed_data_path`` is deliberately left unset so the mod derives it from the raw
    source. Pinning one fixed ``.cache/processed.pt`` here would make ``--data-dir /my/clips``
    load whatever the previous run had prepared: ``DatasetBase.load`` takes the cache fast path
    before it looks at ``raw_data_path``, so the run would train on the sample clip with nothing
    logged. The mod's keyed default prevents that — pinning a path here opts back out of it.
    """
    return TASK_REGISTRY.build("beyondmimic", raw_data_path=args.data_dir)


def train(args: argparse.Namespace):
    en.init(
        backend=gs.cpu if args.cpu else gs.gpu,
        log_root_path="logs/mimic_beyondmimic",
        performance_mode=True,
    )

    config = _make_config(args)
    config.env_options.num_envs = args.num_envs

    if args.wandb_project:
        config.runner_options.wandb_project = args.wandb_project

    if args.adaptive_sampling:
        # Bin-level adaptive start-time curriculum (bounded per-START-bin failure
        # rate; see MotionTerm._update_adaptive_bin_weights). Off by default —
        # this flag enables the validation run against the uniform baseline.
        term = config.motion_options.default
        term.adaptive_sampling = True
        term.adaptive_uniform_ratio = args.adaptive_uniform_ratio
        term.adaptive_failure_rate_max_over_mean = args.adaptive_max_over_mean
        en.logger.info(
            f"adaptive_sampling=True (uniform_ratio={term.adaptive_uniform_ratio}, "
            f"max_over_mean={term.adaptive_failure_rate_max_over_mean}, "
            f"bin_length_s={term.motion_bin_length_s})"
        )

    agent_config = config.runner_options

    if args.cpu:
        agent_config.device = "cpu"

    config.save_as_json()

    # `RslRlVecEnvWrapper.from_config` (not MimicRLEnvBase + a hand-rolled wrap): it
    # dispatches to the same env class, builds, and applies runner_options.seed.
    wrapper = RslRlVecEnvWrapper.from_config(config)

    runner = make_rsl_rl_runner(wrapper)
    runner.add_git_repo_to_log(__file__)

    runner.learn(num_learning_iterations=agent_config.max_iterations, init_at_random_ep_len=True)


def inference(args: argparse.Namespace):
    if not gs._initialized:
        en.init(
            backend=gs.cpu if args.cpu else gs.gpu,
            log_root_path="logs/mimic_beyondmimic",
        )

    config = _make_config(args)

    config.env_options.episode_length_s = 1e9
    config.env_options.num_envs = 1

    # Add ghost entity for visualisation
    from eden.options.robots.unitree_g1 import G1Dofs29

    config.scene_options.ghost_g1 = G1Dofs29(**GHOST_DEFAULTS)

    # Drop all randomization (push, COM shift, friction, default-pos jitter) so the
    # rollout is deterministic, but KEEP a zeroed RSI RESET term: it seeds the robot
    # onto the clip's start pose each reset, and motion-end resync
    # (resync_on_motion_end=True, the beyondmimic task default) reuses it to teleport
    # the robot at clip wrap. Wiping the whole event config here removes RSI too,
    # which both crashes resync at the first wrap and resets the robot to its default
    # joint config instead of the clip start. Mirrors general_mimic/main.py.
    config.event_options = EventManagerOptions(
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

    # Deterministic eval replay: always start the reference at frame 0. Training
    # uses motion_start_from_middle_range=(0,1) for the start-time curriculum,
    # which otherwise makes the ghost begin mid-clip and re-randomise the start on
    # every reset (motion-end/fall) — the "starts from the middle, then restarts"
    # artifact. start_at_timestep_zero_prob=1.0 forces frame 0 on each reset.
    config.motion_options.default.motion_start_from_middle_range = (0.0, 0.0)
    config.motion_options.default.start_at_timestep_zero_prob = 1.0

    agent_cfg = config.runner_options
    agent_cfg.logger = "tensorboard"  # never wandb-log a checkpoint replay

    if args.record:
        config.cameras_options.rec = CameraOptions()

    if args.cpu:
        agent_cfg.device = "cpu"

    config.save_as_json()

    wrapper = RslRlVecEnvWrapper.from_config(config, show_viewer=args.vis, eval_mode=True)
    env = wrapper.unwrapped

    runner = make_rsl_rl_runner(wrapper, checkpoint=args.model)
    runner.add_git_repo_to_log(__file__)

    policy = runner.get_inference_policy(device=wrapper.device)

    # Get total motion length from the data source
    motion_term = env.motion_manager._terms["default"]
    total_frames = int(motion_term.data_source.buffer_length[0].item())

    obs, _ = wrapper.reset()
    if args.record:
        rec = env.cameras["rec"]
        rec.start_recording(os.path.join(en.log_dir, "inference.mp4"))

    # Visualise ghost entity from motion reference
    ghost_entity = env.entities.get("ghost_g1", None)

    for i in tqdm(range(total_frames - 1)):
        with torch.no_grad():
            raw_action = policy(obs)
        obs, _, _, _ = wrapper.step(raw_action)

        # Update ghost pose from motion data
        if ghost_entity is not None:
            ref = env.motion_manager.data["robot"]
            ghost_entity.set_pos(ref.base_pos_w[:, 0])
            ghost_entity.set_quat(ref.base_quat_w[:, 0])
            ghost_entity.set_dofs_pos(ref.dofs_pos[:, 0])

        if args.record:
            rec.render_rgb()

    if args.record:
        rec.stop_recording()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MimicRLEnvBase BeyondMimic motion imitation task.")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # --- train subcommand ---
    train_parser = subparsers.add_parser("train", help="Train the motion imitation policy.")
    train_parser.add_argument("--data-dir", type=str, default=_SAMPLE_DIR, help="Directory containing BVH CSV files")
    train_parser.add_argument("-c", "--cpu", action="store_true", help="Use CPU backend")
    train_parser.add_argument("-b", "--num-envs", type=int, default=4096, help="Number of environments")
    train_parser.add_argument(
        "--adaptive-sampling", action="store_true", help="Enable bin-level adaptive start-time sampling"
    )
    train_parser.add_argument(
        "--adaptive-uniform-ratio", type=float, default=0.1, help="Uniform-mix floor for adaptive sampling"
    )
    train_parser.add_argument(
        "--adaptive-max-over-mean",
        type=float,
        default=10.0,
        help="Per-bin failure-rate cap (x per-motion mean) for adaptive sampling",
    )
    train_parser.add_argument(
        "--wandb-project", type=str, default=None, help="Override the wandb project (default: config value)"
    )

    # --- inference subcommand ---
    inference_parser = subparsers.add_parser("inference", help="Run inference on trained model.")
    inference_parser.add_argument("--model", type=str, required=True, help="Path to the model checkpoint .pt file")
    inference_parser.add_argument(
        "--data-dir", type=str, default=_SAMPLE_DIR, help="Directory containing BVH CSV files"
    )
    inference_parser.add_argument("-r", "--record", action="store_true", help="Record video")
    inference_parser.add_argument("-v", "--vis", action="store_true", help="Show viewer")
    inference_parser.add_argument("-c", "--cpu", action="store_true", help="Use CPU backend")

    args = parser.parse_args()

    if args.command == "train":
        train(args)
        # Chain into inference through the real parser: a hand-built Namespace has to
        # restate argparse's dest names, and drifts from them silently.
        argv = ["inference", "--record", "--data-dir", args.data_dir]
        argv += ["--model", find_latest_checkpoint(en.log_dir, return_path=True)]
        if args.cpu:
            argv.append("--cpu")
        inference(parser.parse_args(argv))
    elif args.command == "inference":
        inference(args)
