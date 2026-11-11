"""Dataset commands: scripted demonstration generation (datagen) and conversion (lerobot).

Handlers lazily import their backends so a plain ``eden --help`` never pays
for them.
"""

from __future__ import annotations


def _run_lerobot_convert(extra_args: list[str]) -> None:
    from eden.utils.lerobot import convert_hdf5_to_lerobot

    convert_hdf5_to_lerobot(extra_args)


def _run_datagen(extra_args: list[str]) -> None:
    """Generate scripted, demo-like demonstrations of a registered task (``eden datagen``)."""
    from eden.tasks.parser import get_task_argparser, get_task_config_from_args
    from eden.utils.cli.tasks import _validate_task_arg

    parser = get_task_argparser(
        description="Roll a task's scripted demonstrator out in parallel envs and record the episodes to HDF5 "
        "(OCBench-style: smooth, randomly paced, with retries and deliberate mistakes).",
        argv=extra_args,
    )
    parser.add_argument("--num_episodes", type=int, default=100, help="Episodes to record (default: 100).")
    parser.add_argument("--num_envs", type=int, default=16, help="Parallel envs (default: 16).")
    parser.add_argument(
        "--output_dir", type=str, default="data/demos", help="Directory of the dataset (default: data/demos)."
    )
    parser.add_argument(
        "--filename", type=str, default=None, help="Dataset file name without extension (default: the task name)."
    )
    parser.add_argument(
        "--export",
        choices=("all", "succeeded", "separate"),
        default="all",
        help="Keep every episode (each tagged with its success), only the successful ones, or the failed ones in "
        "a separate '<filename>_failed' file (default: all).",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed of the env and the demonstrator (default: 0).")
    parser.add_argument(
        "--p_mistake",
        type=float,
        default=None,
        help="Probability a subgoal is a deliberate mistake (default: the task demonstrator's own).",
    )
    parser.add_argument(
        "--episode_length_s",
        type=float,
        default=None,
        help="Episode time limit in seconds (default: the task demonstrator's own, else 90).",
    )
    parser.add_argument(
        "--camera_only",
        action="store_true",
        help="Run the demonstrator without the simulator's object state: it locates objects through the task's "
        "cameras (demonstrators that support it only, e.g. robosuite_pick_place).",
    )
    parser.add_argument("--show_viewer", action="store_true", help="Open the Genesis viewer.")
    args = parser.parse_args(extra_args)
    _validate_task_arg(args, parser)
    if args.num_envs <= 0 or args.num_episodes <= 0:
        parser.error("--num_envs and --num_episodes must be positive integers.")

    import genesis as gs

    import eden as en
    from eden.constants import DatasetExportMode
    from eden.extensions.scripted_demos.generate import generate_demos

    en.init(backend=gs.cpu if args.cpu else gs.gpu, log_root_path=f"logs/datagen/{args.task}", performance_mode=True)
    config = get_task_config_from_args(args, run_name=args.run_name)
    config.env_options.num_envs = args.num_envs
    report = generate_demos(
        config,
        args.task,
        num_episodes=args.num_episodes,
        output_dir=args.output_dir,
        filename=args.filename or args.task,
        export_mode={
            "all": DatasetExportMode.EXPORT_ALL,
            "succeeded": DatasetExportMode.EXPORT_SUCCEEDED_ONLY,
            "separate": DatasetExportMode.EXPORT_SUCCEEDED_FAILED_IN_SEPARATE_FILES,
        }[args.export],
        seed=args.seed,
        p_mistake=args.p_mistake,
        episode_length_s=args.episode_length_s,
        camera_only=args.camera_only,
        show_viewer=args.show_viewer,
    )
    print(f"✅ {report.render()}")
