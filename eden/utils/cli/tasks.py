"""Task-execution commands: train, inference, export, deploy, preprocess, smoke."""

from __future__ import annotations

import importlib
import os
import sys
from typing import Any, Literal, NamedTuple
import numpy as np

from eden.utils.distribution import install_hint


_MAX_DEPLOY_SLOTS = 9
_GHOST_SPHERE_RADIUS = 0.04
_GHOST_COLOR = (0.2, 0.7, 1.0, 0.5)  # translucent cyan
_DEFAULT_POLICY_INFERENCE_STEPS = 1000


def smoke_task(extra_args: list[str]) -> None:
    """Build a registered task and step it a few times with no action (end-to-end smoke check)."""
    from eden.tasks.parser import get_task_argparser, get_task_config_from_args

    parser = get_task_argparser(
        description="Build a task by name and step it a few times — a one-call end-to-end sanity check.",
        argv=extra_args,
    )
    parser.add_argument("--num_envs", type=int, default=2, help="Number of parallel envs to build (default: 2).")
    parser.add_argument("--steps", type=int, default=2, help="Number of no-action steps to run (default: 2).")
    parser.add_argument("--eval_mode", action="store_true", help="Build in eval mode (training/randomization off).")
    parser.add_argument("--show_viewer", action="store_true", help="Open the Genesis viewer.")
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Profile per-term compute cost and write term_cost.json (instead of a plain sanity step).",
    )
    parser.add_argument(
        "--profile_warmup", type=int, default=10, help="Warmup steps before measuring, with --profile (default: 10)."
    )
    parser.add_argument("--profile_steps", type=int, default=50, help="Measured steps, with --profile (default: 50).")
    args = parser.parse_args(extra_args)
    _validate_task_arg(args, parser)
    if args.num_envs <= 0:
        parser.error("--num_envs must be a positive integer.")
    if args.steps <= 0:
        parser.error("--steps must be a positive integer.")
    if args.profile and (args.profile_warmup < 0 or args.profile_steps <= 0):
        parser.error("--profile_warmup must be >= 0 and --profile_steps must be a positive integer.")

    import eden as en
    import genesis as gs

    en.init(
        backend=gs.gpu if not args.cpu else gs.cpu,
        log_root_path=f"logs/smoke/{args.task}",
        performance_mode=True,
    )

    config = get_task_config_from_args(args, run_name=args.run_name)

    if args.profile:
        from eden.utils.profiling import run_smoke_profile

        report, path = run_smoke_profile(
            config,
            num_envs=args.num_envs,
            warmup=args.profile_warmup,
            measure=args.profile_steps,
            eval_mode=args.eval_mode,
        )
        print(report.render())
        print(f"📊 wrote {path}")
        return

    from eden.utils.smoke import build_and_step

    build_and_step(
        config,
        num_envs=args.num_envs,
        num_steps=args.steps,
        eval_mode=args.eval_mode,
        show_viewer=args.show_viewer,
    )
    print(f"✅ smoke OK: built '{args.task}' with {args.num_envs} env(s) and ran {args.steps} step(s).")


def _validate_task_arg(args, parser):
    """Ensure ``args.task`` is set and registered; print a helpful error otherwise."""
    from eden.tasks import TASK_REGISTRY

    if not args.task:
        parser.error("--task is required (run 'eden list' to see available tasks).")
    if args.task not in TASK_REGISTRY:
        # Not list_tasks(): it raises for a clash on any other name, which would hide this one's typo.
        parser.error(TASK_REGISTRY.missing_task_message(args.task))


def _explicit_task_mods(args) -> list[str]:
    """The ``--`` flags for task mods this Namespace sets away from their default.

    Namespace-shaped adapter over :func:`eden.tasks.parser.explicit_task_mods`, which is
    where the baseline rule lives (and which the saved-``--config`` guard shares).
    """
    from eden.tasks.parser import _collect_modifier_kwargs, explicit_task_mods

    task = getattr(args, "task", None) or ""
    return explicit_task_mods(task, _collect_modifier_kwargs(args, task))


def _checkpoint_sibling_config(args) -> str | None:
    """Path of the ``config.json`` beside ``--checkpoint`` that a replay would load, or ``None``.

    The one place the question "does this run replay a saved config?" is answered, so the
    log-root decision (:func:`_replay_log_root`, which runs before ``eden.init``) and the load
    itself (:func:`_config_from_checkpoint`, which runs after it) cannot drift apart on it.

    Parameters
    ----------
    args: argparse.Namespace
        The parsed ``eden inference`` / ``eden deploy`` arguments.

    Returns
    -------
    str | None
        The sibling path, or ``None`` when there is no ``--checkpoint``, when ``--config``
        names a *saved* config (an explicit one wins over the sibling), or when no sibling
        ``config.json`` exists (a loose or legacy checkpoint).
    """
    from eden.tasks.parser import read_config_arg
    from eden.utils.configs import is_full_config

    if not getattr(args, "checkpoint", None):
        return None
    if getattr(args, "config", None) and is_full_config(read_config_arg(args.config)):
        return None  # an explicit saved config replaces the checkpoint's own; let the rebuild path apply it
    sibling = os.path.join(os.path.dirname(os.path.abspath(args.checkpoint)), "config.json")
    return sibling if os.path.exists(sibling) else None


def _replay_log_root(args, *, task_less: str) -> str:
    """The ``logs/<task>`` root for ``eden inference`` / ``eden deploy``, checked against the checkpoint.

    A checkpoint's sibling ``config.json`` is authoritative for a replay
    (:func:`_config_from_checkpoint`), so ``--task`` is only a label — and a wrong label is
    durable in a way the config object is not: the run is written to *another* task's
    ``logs/<task>/``, which ``eden train --resume latest`` and checkpoint-sibling replay both
    resolve later, so one mislabelled run quietly pollutes another task's history. The
    saved config's own task therefore wins the directory, and the mismatch is reported.

    A warning, not the hard error a mismatched saved ``--config`` gets: there the user
    handed the config over by name, here the sibling was picked up automatically, and replaying a
    checkpoint whose config class is no longer a *registered task* is a legitimate thing to do
    (the class need only still be importable — a renamed or moved one raises ``ImportError`` from
    the load itself, in :meth:`EdenConfig.load_from_file`).

    Identity is read import-free, off class names alone, and in the forward direction: the saved
    class is compared to the one ``--task`` *declares*
    (:meth:`~eden.tasks.registry.TaskRegistry.config_qualname_for_task`), never to a reverse
    lookup of the saved class. Nothing stops one config class being registered under two names,
    and a reverse lookup returns only the first of them — so an alias would read as a mismatch and
    this helper would relocate a correctly-labelled run. The reverse lookup is used only *after* a
    mismatch is established, to name the directory.

    Parameters
    ----------
    args: argparse.Namespace
        The parsed ``eden inference`` / ``eden deploy`` arguments.
    task_less: str
        Log root for a run with no ``--task`` at all (an extension-policy-only run).

    Returns
    -------
    str
        The log root to hand :func:`eden.init`.
    """
    import eden as en

    from eden.tasks import TASK_REGISTRY
    from eden.utils.configs import read_config_file, saved_config_qualname

    task = getattr(args, "task", None)
    if not task:
        return task_less
    sibling = _checkpoint_sibling_config(args)
    saved = saved_config_qualname(read_config_file(sibling)) if sibling else None
    built = TASK_REGISTRY.config_qualname_for_task(task)
    if saved is None or built is None or saved == built:
        # Nothing to check against (no replay to do, or a legacy save with no class tag), or it
        # checks out. ``built`` is None only for an unregistered ``--task``, which the caller has
        # already rejected — and a helper that only picks a directory must never warn on a guess.
        return f"logs/{task}"
    owner = TASK_REGISTRY.task_for_config_qualname(saved)
    if owner is None:
        en.logger.warning(
            f"⚠️ The checkpoint's own config ~~<{sibling}>~~ was saved from '{saved}', which is not "
            f"what --task {task} builds ('{built}') and which no registered task declares. There is "
            f"no better directory to name, so this run stays under logs/{task}/ — check that is "
            f"where you want it. If that class was renamed or moved, loading it raises next."
        )
        return f"logs/{task}"
    en.logger.warning(
        f"⚠️ The checkpoint's own config ~~<{sibling}>~~ belongs to task '{owner}', not --task "
        f"'{task}' — and that config is authoritative for a replay, so this run *is* '{owner}'. "
        f"Logging under logs/{owner}/ rather than logs/{task}/, which --resume and checkpoint-sibling "
        f"replay resolve later. Re-run with --task {owner} to silence this."
    )
    return f"logs/{owner}"


def _config_from_checkpoint(args):
    """Load the checkpoint's own sibling ``config.json`` (the exact trained config), or None.

    Identity-coupled replay/deploy: a checkpoint at ``<run>/model_<n>.pt`` was
    trained with ``<run>/config.json``, so ``eden inference`` / ``eden deploy`` load that
    exact config by default rather than rebuilding it from ``--task`` + mods (which would
    silently substitute the task's *current* defaults for whatever the checkpoint was
    trained with). Returns ``None`` — signalling the caller to fall back to the rebuild
    path — in every case :func:`_checkpoint_sibling_config` finds no sibling to load.

    A partial ``--config`` and any ``--override`` still apply on top. Explicit task-mod flags,
    by contrast, cannot apply to a saved config and would be silently ignored, so they are a
    hard error here (fail fast, never mask a caller mistake) rather than a warning. A
    ``--task`` that disagrees with the saved config is reported by :func:`_replay_log_root`,
    which owns the durable half of that mistake.

    The env's training-only fields (``EnvOptions.training_only_fields``, e.g. ``cuda_graph_options``) are
    reset to their defaults first: they only speed up or instrument training, and one may not build where the
    replay runs (``cuda_graph_options`` on a host without an NVIDIA GPU). ``--config`` / ``--override`` apply
    after, so either can set one again.
    """
    import eden as en

    from eden.tasks.parser import _apply_run_name, overrides_from_args, read_config_arg
    from eden.utils.configs import EdenConfig

    sibling = _checkpoint_sibling_config(args)
    if sibling is None:
        return None
    mods = _explicit_task_mods(args)
    if mods:
        sys.exit(
            f"Task mod flag(s) {', '.join(mods)} cannot apply to a replay: the checkpoint's own "
            f"config ({sibling}) is authoritative and the mods would be silently ignored. Drop them "
            f"to replay the trained config, or pass a saved config to --config to supply your own."
        )
    config = EdenConfig.load_from_file(sibling)
    reset = config.env_options.reset_training_only_fields()
    if reset:
        en.logger.info(f"🧩 Reset the training-only env_options {', '.join(reset)} for the replay.")
    if getattr(args, "config", None):
        # A *saved* --config would have suppressed the sibling above, so this is a partial.
        config = config.with_overrides_from_dict(read_config_arg(args.config))
    for overrides in overrides_from_args(args):
        config = config.with_overrides_from_dict(overrides)
    _apply_run_name(config, getattr(args, "run_name", ""))
    en.logger.info(f"🧩 Loaded the checkpoint's own config ~~<{sibling}>~~ (--config/--override on top).")
    return config


def _add_shared_policy_cli_args(parser, *, command: Literal["inference", "deploy"]) -> None:
    """Add the argparse options shared by ``eden inference`` and ``eden deploy``.

    ``command`` is ``"inference"`` or ``"deploy"``. Both subcommands take the same viewer /
    scene-entity / extension-policy plumbing options with identical flag names, types,
    defaults and metavars — defined once here so they can't drift — and only the help
    wording differs, so it is selected per ``command``. ``--policy`` itself is deliberately
    NOT shared (inference accepts a single policy, deploy ``append``s a list of switchable
    slots), so each subcommand declares it.
    """
    deploy = command == "deploy"
    viewer_help = (
        "Open the Genesis viewer to mirror the hardware state during deployment."
        if deploy
        else "Open the Genesis viewer window during inference."
    )
    entity_help = (
        "Scene entity to deploy (defaults to the canonical 'robot'). A composite rig placed under this name "
        "(an arm with a hand welded on, e.g. FrankaXHand) deploys every part, each through its own backend; "
        "name another part of it (e.g. 'hand') to deploy that part alone."
        if deploy
        else "Scene entity the extension policy controls (defaults to the canonical 'robot')."
    )
    base_command_help = (
        "Constant velocity command for command-driven extension policies"
        + (", including the base loco policy" if deploy else "")
        + " (default: stand)."
    )
    robot_help = (
        f"Robot options class from eden.robots for task-less --policy {'deploys' if deploy else 'runs'} "
        "(default: G1Dofs29). Ignored when --task is given."
    )
    motion_help = "Reference motion npz for motion-tracking extension policies (e.g. sonic). " + (
        "Required when such a policy is loaded; the motion restarts from frame 0 each time you switch to that policy."
        if deploy
        else "Required when such a policy is loaded."
    )

    def add_if_absent(*flags: str, **kwargs) -> None:
        # get_task_argparser runs before this helper, so a TaskMod may already own one of
        # these flag names — robot-parametric tasks (move_forward, velocity_tracking_*) expose their
        # own --robot, and that binding is what actually rewires the scene. Let the mod's version
        # win and skip the shared one; re-adding it would raise argparse's opaque "conflicting
        # option string" at parser-build time. Every shared flag is extension-policy
        # plumbing that's ignored or overridden when --task is given, so deduping is always safe.
        if not any(flag in parser._option_string_actions for flag in flags):
            parser.add_argument(*flags, **kwargs)

    add_if_absent("--vis", "--show_viewer", action="store_true", dest="show_viewer", help=viewer_help)
    add_if_absent("--entity_name", type=str, default="robot", help=entity_help)
    add_if_absent(
        "--base_command",
        nargs=3,
        type=float,
        default=(0.0, 0.0, 0.0),
        metavar=("VX", "VY", "WZ"),
        help=base_command_help,
    )
    add_if_absent("--robot", type=str, default="G1Dofs29", help=robot_help)
    add_if_absent("--motion_file", default=None, help=motion_help)


def train_task(extra_args: list[str]) -> None:
    """``eden train --task <name> [task-mod flags]`` — train a registered task.

    Forwards ``extra_args`` (everything after ``train``) to
    :func:`eden.tasks.parser.get_task_argparser`. The argparser also uses
    ``extra_args`` (not ``sys.argv``) for its ``--task`` peek, so the same
    arg slice drives both mod-flag registration and the final parse.
    Dispatches into the right framework (rsl_rl by default, rl_games when
    the task's ``runner_options`` is an ``RlGamesPpoRunnerOptions``).
    """
    from eden.tasks.parser import get_task_argparser, get_task_config_from_args

    parser = get_task_argparser(
        description="Train a registered task with TaskMod-driven CLI.",
        argv=extra_args,
    )
    parser.add_argument(
        "--logger",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "neptune", "wandb"],
        help="rsl_rl logger backend (ignored for rl_games).",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=None,
        help="Override the task's training num_envs.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        metavar="REF",
        help="Resume from a prior run of this task, resolving REF under logs/<task>/: "
        "'latest' (most recently modified), a run id, or a run_name substring. Loads that "
        "run's latest checkpoint and forks a new run dir recording its parent. Mutually "
        "exclusive with --checkpoint.",
    )
    parser.add_argument(
        "--wandb_group",
        type=str,
        default=None,
        help="wandb group for this run (defaults to the task name). Ablation arms sharing "
        "a group are compared together in the wandb UI. Ignored unless --logger wandb.",
    )
    args = parser.parse_args(extra_args)
    _validate_task_arg(args, parser)
    if args.num_envs is not None and args.num_envs <= 0:
        parser.error("--num_envs must be a positive integer.")
    if args.resume and args.checkpoint:
        parser.error("--resume and --checkpoint are mutually exclusive (both pick a checkpoint to load).")

    import eden as en
    import genesis as gs

    # Resolve --resume before building anything so a bad ref fails fast. A resume forks a
    # fresh run dir (en.init below) and records its parent in run.json, so wandb/TB
    # continuity is not silently hijacked.
    resume_checkpoint, resumed_from = None, None
    if args.resume:
        from eden.utils.assets import find_latest_checkpoint, resolve_run_dir

        parent_dir = resolve_run_dir(f"logs/{args.task}", args.resume)
        resume_checkpoint = find_latest_checkpoint(parent_dir)
        resumed_from = os.path.basename(os.path.normpath(parent_dir))
        en.logger.info(
            f"↩️  Resuming from ~~<{resumed_from}>~~ ({os.path.basename(resume_checkpoint)}) into a fresh run"
        )

    en.init(
        backend=gs.gpu if not args.cpu else gs.cpu,
        log_root_path=f"logs/{args.task}",
        run_name=args.run_name,
        performance_mode=True,
    )

    config = get_task_config_from_args(args, run_name=args.run_name)
    # Tasks with no in-Eden runner are trained by a standalone script (or only evaluated), not `eden train`.
    # Must fire before any config.runner_options access below (logger/seed/save).
    if config.runner_options is None:
        sys.exit(
            f"Task '{args.task}' declares no runner_options, so `eden train` has no RL backend for it. "
            f"{_no_runner_hint(config)}"
        )
    if args.num_envs is not None:
        config.env_options.num_envs = args.num_envs
    if hasattr(config.runner_options, "logger"):
        config.runner_options.logger = args.logger
    # Group runs in wandb (its natural home for ablation arms; the run_name folded into the
    # log-dir basename above already names each run). Defaults to the task, or --wandb_group
    # to cluster an ablation across tasks. wandb reads WANDB_RUN_GROUP natively, so this
    # needs no rsl_rl/rl_games plumbing; setdefault leaves an explicit env value untouched.
    if args.logger == "wandb":
        os.environ.setdefault("WANDB_RUN_GROUP", args.wandb_group or args.task)
    # config.json/run.json record the *base* seed exactly as configured — it is concrete by
    # construction (negatives are rejected at config load) and `apply_run_seed` adds the
    # per-rank offset later, inside the wrapper's from_config.
    # Main rank only: all ranks share this log_dir, so writing from each would race. Workers
    # build the env from the in-memory config, not the file.
    checkpoint = args.checkpoint or resume_checkpoint  # mutually exclusive (validated above)
    if not _is_rl_games(config):
        _check_rsl_rl_runner(config, checkpoint)
    if en.distributed.is_main_process():
        config.save_as_json()
        _write_run_manifest(config, args, resumed_from=resumed_from)

    if _is_rl_games(config):
        _run_train_rl_games(config, checkpoint=checkpoint)
    else:
        _run_train_rsl_rl(config, checkpoint=checkpoint)


def inference_task(extra_args: list[str]) -> None:
    """``eden inference --task <name> --checkpoint <path> [task-mod flags]``.

    Loads a trained checkpoint and runs the eval loop for ``episode_length_s`` (or
    ``--steps`` / ``--episodes``), through rsl_rl or rl_games — whichever the task's
    ``runner_options`` selects, after :func:`_check_checkpoint_framework` confirms the
    checkpoint was saved by it.

    Frozen extension policies (``POLICY_REGISTRY``) can be rolled out in sim
    instead of a checkpoint via ``--policy NAME`` — the sim counterpart of
    ``eden deploy --policy``. With ``--task``, the policy runs on the task's
    env (the policy's command terms are merged in); without one, the env is
    built from the policy's own spec methods on an empty scene with the
    ``--robot`` robot.

    With ``--async`` (a.k.a. ``--sim2sim``) the rollout runs deployment-style
    instead: policy inference moves to an async background thread and the sim
    loop is paced to wall-clock ``env.dt``, repeating the last action whenever
    inference hasn't finished — a sim-side dry run of ``eden deploy``. This
    applies to both checkpoints and ``--policy`` extension policies. See
    :func:`_run_sim2sim_loop`.
    """
    from eden.tasks.parser import get_task_argparser, get_task_config_from_args

    parser = get_task_argparser(
        description="Run inference for a registered task or a frozen extension policy "
        "(add --async for a deployment-style, wall-clock-paced rollout).",
        argv=extra_args,
    )
    _add_shared_policy_cli_args(parser, command="inference")
    parser.add_argument(
        "--rec",
        action="store_true",
        help="Record an mp4 via a 'rec' camera (saved under the run's log dir).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Cap inference at N env steps (default: one episode, i.e. max_episode_length).",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Count each env's first ceil(N / num_envs) episodes, running until every env has finished them: "
        "ceil(N / num_envs) * num_envs episodes in all, at least N (stopping at N resets over all envs would leave "
        "out the longer episodes still running).",
    )
    parser.add_argument(
        "--num_envs",
        type=int,
        default=1,
        help="Number of parallel eval envs (default: 1).",
    )
    parser.add_argument(
        "--async",
        "--sim2sim",
        action="store_true",
        dest="run_async",
        help="Run deployment-style: policy inference in an async background thread, sim "
        "paced to wall-clock dt, last action repeated when inference lags — a sim-side dry "
        "run of 'eden deploy'. Works with both checkpoints and --policy.",
    )
    parser.add_argument(
        "--policy",
        "--extension_policy",
        default=None,
        dest="extension_policy",
        metavar="NAME",
        help="Frozen extension policy to roll out in sim instead of a checkpoint. "
        "Accepts the package name from eden/extensions/policies (e.g. 'holosoma', "
        "'homie'), the POLICY_REGISTRY key, or the key without its '_policy' suffix. "
        "With --policy, --task and --checkpoint become optional: a task-less run "
        "builds the env from the policy's own spec on the --robot scene.",
    )
    parser.add_argument(
        "--clip_name",
        default=None,
        metavar="CLIP",
        help="Clip to track for per-clip extension policies (e.g. beyondmimic_moves). "
        "Ignored by policies that don't take one; they use their default clip.",
    )
    args = parser.parse_args(extra_args)
    if args.extension_policy:
        if args.checkpoint:
            parser.error("--policy and --checkpoint are mutually exclusive; 'eden deploy' supports both as slots.")
        if args.episodes is not None:
            parser.error("--episodes is not supported with --policy (extension policies run a fixed --steps budget).")
        if args.task:
            _validate_task_arg(args, parser)
    else:
        _validate_task_arg(args, parser)
        if not args.checkpoint:
            parser.error("--checkpoint is required for 'eden inference' (or pass --policy).")
    if args.steps is not None and args.steps <= 0:
        parser.error("--steps must be a positive integer.")
    if args.episodes is not None and args.episodes <= 0:
        parser.error("--episodes must be a positive integer.")
    if args.num_envs <= 0:
        parser.error("--num_envs must be a positive integer.")

    import eden as en
    import genesis as gs

    en.init(
        backend=gs.gpu if not args.cpu else gs.cpu,
        log_root_path=_replay_log_root(args, task_less="logs/inference"),
        performance_mode=True,
    )

    # Prefer the checkpoint's own config.json (the exact trained config) over rebuilding
    # from --task + mods; falls back to the rebuild when there's no sibling config.json.
    config = (
        (_config_from_checkpoint(args) or get_task_config_from_args(args, run_name=args.run_name))
        if args.task
        else None
    )
    if config is not None:
        config.env_options.num_envs = args.num_envs
        if hasattr(config.runner_options, "logger"):
            config.runner_options.logger = "tensorboard"  # never wandb-log a checkpoint replay
        if args.rec:
            config.add_recording_camera()

    if args.extension_policy:
        _run_inference_policy(
            config,
            args.extension_policy,
            show_viewer=args.show_viewer,
            record=args.rec,
            max_steps=args.steps or _DEFAULT_POLICY_INFERENCE_STEPS,
            entity_name=args.entity_name,
            robot_name=args.robot,
            motion_file=args.motion_file,
            clip_name=args.clip_name,
            base_command=tuple(args.base_command),
            loop_fn=_run_sim2sim_loop if args.run_async else None,
        )
        return

    config.save_as_json()

    if config.runner_options is None:
        sys.exit(
            f"Task '{args.task}' declares no runner_options, so `eden inference` has no runner to replay it. "
            f"{_no_runner_hint(config)}"
        )

    configured = _check_checkpoint_framework(config, args.checkpoint)
    runner_fn = _run_eval_rl_games if configured == "rl_games" else _run_eval_rsl_rl
    runner_fn(
        config,
        args.checkpoint,
        show_viewer=args.show_viewer,
        record=args.rec,
        max_steps=args.steps,
        max_episodes=args.episodes,
        loop_fn=_run_sim2sim_loop if args.run_async else None,
    )


def export_task(extra_args: list[str]) -> None:
    """``eden export --task <name> --checkpoint <path> [--format onnx,jit] [--out DIR]``.

    Writes the checkpoint's policy as a graph a runtime outside Eden can load (ONNX, TorchScript) and a
    ``policy_spec.json`` describing its observations, actions and control rate (:mod:`eden.learning.export`).
    The env is built from the checkpoint's own ``config.json``, as ``eden inference`` builds it, with one env in
    eval mode and without the event terms that would randomize what the spec records as nominal: the spec is read
    off it, and every graph is checked against the policy on its observations.
    """
    from eden.learning.export import (
        EXPORT_FORMATS,
        check_export_dependencies,
        disable_contract_randomization,
        export_policy,
    )
    from eden.tasks.parser import get_task_argparser, get_task_config_from_args

    parser = get_task_argparser(
        description="Export a trained checkpoint's policy (ONNX / TorchScript) and its policy_spec.json "
        "for a runtime outside Eden.",
        argv=extra_args,
    )
    parser.add_argument(
        "--format",
        default=",".join(EXPORT_FORMATS),
        help=f"Comma-separated formats to write, from {', '.join(EXPORT_FORMATS)} (default: all of them).",
    )
    parser.add_argument(
        "--out",
        default=None,
        metavar="DIR",
        help="Directory to write into (default: exported/<checkpoint name>/ beside the checkpoint).",
    )
    parser.add_argument(
        "--no_verify",
        action="store_true",
        help="Skip checking each exported graph against the policy on the env's observations "
        f"(the ONNX check needs onnxruntime: {install_hint('onnx')}).",
    )
    args = parser.parse_args(extra_args)
    _validate_task_arg(args, parser)
    if not args.checkpoint:
        parser.error("--checkpoint is required for 'eden export'.")
    if not os.path.isfile(args.checkpoint):
        parser.error(f"--checkpoint {args.checkpoint!r} is not a file.")
    formats = tuple(fmt.strip() for fmt in args.format.split(",") if fmt.strip())
    unknown = sorted(set(formats) - set(EXPORT_FORMATS))
    if unknown or not formats:
        parser.error(
            f"--format {args.format!r} names {unknown or 'nothing'}; choose from {', '.join(EXPORT_FORMATS)} "
            f"(comma-separated, e.g. --format onnx)."
        )
    try:
        check_export_dependencies(formats, verify=not args.no_verify)
    except ImportError as exc:
        parser.error(str(exc))
    out_dir = args.out or os.path.join(
        os.path.dirname(os.path.abspath(args.checkpoint)),
        "exported",
        os.path.splitext(os.path.basename(args.checkpoint))[0],
    )

    import eden as en
    import genesis as gs

    en.init(
        backend=gs.gpu if not args.cpu else gs.cpu,
        log_root_path=_replay_log_root(args, task_less="logs/export"),
        performance_mode=True,
    )
    config = _config_from_checkpoint(args) or get_task_config_from_args(args, run_name=args.run_name)
    if config.runner_options is None:
        sys.exit(
            f"Task '{args.task}' declares no runner_options, so there is no Eden runner to load its checkpoint "
            f"into. {_no_runner_hint(config)}"
        )
    if _check_checkpoint_framework(config, args.checkpoint) == "rl_games":
        sys.exit(
            "eden export reads rsl_rl checkpoints only: an rl_games checkpoint has no export hooks to build the "
            "graph from. Export it with rl_games' own tooling, or train the task with rsl_rl (its default runner)."
        )
    config.env_options.num_envs = 1
    if hasattr(config.runner_options, "logger"):
        config.runner_options.logger = "tensorboard"  # never wandb-log an export
    if disabled := disable_contract_randomization(config):
        en.logger.info(f"📦 Disabled event terms {disabled}: they randomize the default pose the spec records.")

    env, runner = _rsl_rl_env_and_runner(config, args.checkpoint, inference=True, eval_mode=True)
    files = export_policy(
        runner,
        env,
        out_dir,
        formats=formats,
        task=args.task,
        checkpoint=args.checkpoint,
        verify=not args.no_verify,
    )
    checked = "written" if args.no_verify else "checked against the policy and written"
    en.logger.info(f"📦 Exported policy {checked}: " + ", ".join(f"~~<{path}>~~" for path in files.values()))


def deploy_task(extra_args: list[str]) -> None:
    """``eden deploy --task <name> --checkpoint <path> [task-mod flags]``.

    Run a trained policy on real hardware. The deployment backend is picked
    automatically from the robot bound to the task's scene entity via
    :func:`eden.extensions.deployment.resolve_deployment_for` — to support a
    new robot, add a single ``bind_deployment(...)`` line in its deployment
    module.

    By default (G1 robots) the frozen Holosoma loco policy is preloaded as
    slot 1 and is the policy active on boot: after INIT → START the robot
    stands (or walks ``--base_command``) under a known-good controller, and
    the operator switches to the checkpoint slot(s) — and back — via the
    remote Select button or keyboard ``n`` / ``1``-``9``. Pass
    ``--base_policy none`` to restore checkpoint-as-slot-1.

    Frozen extension policies (``POLICY_REGISTRY``) can be preloaded as
    additional slots via ``--policy NAME``. Either ``--task`` or ``--policy``
    must be given: with a task, extension policies ride along on the task's
    env and ``--checkpoint`` becomes optional; without a task, the env is
    built on an empty scene with the ``--robot`` robot from the spec of the
    loaded policy that reads it, else of the first policy ``--policy`` named
    (:func:`_env_owning_slot`) — checkpoints then require ``--task``.

    Each slot converts its own actions, through the policy's own controller
    (:func:`_slot_controllers`) or, for the env-owning policy, through the env's
    action manager, so a policy is not required to emit absolute targets.

    The function disables training-only randomization (push, friction /
    COM / encoder-bias randomization) and forces ``num_envs=1`` so the
    policy drives a single physical robot. The sim env is still built (with
    the viewer optionally open) so it can mirror robot state for
    visualization while the policy runs on hardware.
    """
    from eden.tasks.parser import get_task_argparser, get_task_config_from_args

    parser = get_task_argparser(
        description="Deploy a trained task policy to a real robot.",
        argv=extra_args,
    )
    _add_shared_policy_cli_args(parser, command="deploy")
    parser.add_argument(
        "--deploy_kp_kd_scale",
        type=float,
        default=1.0,
        help="Scale on default Kp/Kd at deployment. Ramp up from 0.2 → 1.0 when first commissioning.",
    )
    parser.add_argument(
        "--gain_ramp_steps",
        type=int,
        default=None,
        help="Control steps over which a policy-slot switch ramps Kp/Kd to the new slot's own "
        "(backend default 50 = 1 s at 50 Hz; 0 jumps). Each slot runs at the gains its policy was "
        "trained at, and a stiffness step between two of them can knock the robot over.",
    )
    parser.add_argument(
        "--estimate_lin_vel",
        action="store_true",
        help="Estimate base linear velocity onboard (contact-aided Kalman filter fusing IMU "
        "acceleration with leg-kinematics odometry — the legged_control/BeyondMimic estimator). "
        "Observation terms reading base_lin_vel then see the estimate instead of zeros. "
        "Contact is detected from joint torques (tau_est) via the foot Jacobian.",
    )
    parser.add_argument(
        "--feet_links",
        nargs="+",
        default=None,
        metavar="LINK",
        help="Foot link name(s) the --estimate_lin_vel filter hangs its leg odometry and contact "
        "detection on (e.g. left_ankle_roll_link right_ankle_roll_link). Defaults to the robot's "
        "own feet (RobotOptions.feet_links_name), which the deployable legged robots declare, so "
        "this is only needed for a robot that declares none — a wheeled or retargeting-only body "
        "— or to estimate off a different pair of links.",
    )
    parser.add_argument(
        "--velocity_command",
        choices=("constant", "keyboard", "gamepad", "task"),
        default="constant",
        help="What drives the deployed SE(2) velocity command, if the task declares one. A "
        "policy deployed from its training config otherwise keeps generating the command it "
        "was TRAINED on -- a fresh random (vx, vy, wz) every few seconds -- which on a robot "
        "reads as it wandering off on its own. 'constant' (default) substitutes a term holding "
        "--base_command, so the robot stands until told otherwise. 'keyboard' (arrows for "
        "vx/vy, , / . for yaw; letters and digits belong to the operator state machine and "
        "pynput listens globally) and 'gamepad' (left stick vx/vy, right stick X yaw) hand it "
        "to an operator. 'task' keeps the training generator. Full deflection is the fastest "
        "command the replaced term declares, and the operator cannot exceed it. All of it "
        "works on hardware and with --mujoco.",
    )
    parser.add_argument(
        "--gamepad_layout",
        choices=("ps4", "xbox"),
        default="ps4",
        help="Axis scaling for --velocity_command gamepad (PS4 sticks report 0..255, Xbox "
        "-32768..32767). Default: ps4.",
    )
    parser.add_argument(
        "--velocity_command_scale",
        nargs=3,
        type=float,
        default=None,
        metavar=("VX", "VY", "WZ"),
        help="Symmetric per-axis limit for the substituted velocity command, overriding the "
        "trained ranges of the term it replaces. Lower it to bound the operator (or a constant "
        "--base_command) below what the policy was trained for while commissioning; it is also "
        "the only bound available on a task-less --policy deploy, which has no term to read.",
    )
    parser.add_argument(
        "--extra_checkpoint",
        action="append",
        default=None,
        metavar="PATH",
        help="Additional policy checkpoint(s) to preload for runtime switching "
        "(repeatable). Slots are numbered in load order: the base policy (if any) "
        "is slot 1, then --checkpoint, then extras. Switch live via the G1 remote "
        "Select button or keyboard n / 1-9 (9 policies max).",
    )
    parser.add_argument(
        "--base_policy",
        choices=("auto", "holosoma", "none"),
        default="auto",
        help="Frozen base/safety policy preloaded as slot 1 and active on boot. "
        "'auto' (default) loads the Holosoma G1 velocity-tracking loco policy only when the "
        "deploy robot is a G1 29-DOF (G1Dofs29), and loads nothing otherwise (--checkpoint "
        "becomes slot 1). 'holosoma' forces it (errors on a non-G1 robot); 'none' disables it. "
        "With Holosoma, the robot stands (or walks --base_command) under a known-good "
        "controller until you switch to a checkpoint slot, and you can always switch back.",
    )
    parser.add_argument(
        "--policy",
        "--extension_policy",
        action="append",
        default=None,
        dest="extension_policy",
        metavar="NAME",
        help="Extension policy to preload as an extra switchable slot (repeatable). "
        "Accepts the package name from eden/extensions/policies (e.g. 'holosoma', "
        "'homie'), the POLICY_REGISTRY key, or the key without its '_policy' suffix. "
        "Each slot converts through its own controller, so any joint-position controller "
        "deploys, scaled and pose-offset ones included. Policies that track a reference "
        "motion (e.g. 'sonic') also need --motion_file; their command terms are merged "
        "into the env. With --policy, --task and --checkpoint become optional: a task-less "
        "deploy builds the env from the spec of whichever loaded policy reads it "
        "(the first policy's, when none does) on the --robot scene.",
    )
    parser.add_argument(
        "--disable_joints",
        nargs="+",
        default=None,
        metavar="NAME",
        help="Joint name(s) or fnmatch patterns (e.g. 'left_wrist_*' '*_ankle_*_joint') to disable: "
        "send kp=kd=0 so the joint goes free/limp (the policy still commands it, the motor ignores it). "
        "For bench debugging — isolate a buzzing joint or free the arms. Do NOT disable load-bearing leg "
        "joints on an unsupported robot; it will collapse.",
    )
    parser.add_argument(
        "--ignore_action_terms",
        nargs="+",
        default=None,
        metavar="TERM",
        help="Action term name(s) whose command the payload has no channel for (e.g. a suction "
        "term's vacuum bit) and that you drive over your own channel. RobotCommand carries only "
        "joint dofs_pos/vel/torque/kp/kd, so such a term cannot be sent; naming it here "
        "acknowledges that and lets the deploy proceed. Without it the deployment refuses to "
        "start, rather than dropping the command silently on the robot. Names are validated: a "
        "term the payload does carry is rejected here (it would not stop being sent) — use "
        "--disable_joints to leave joints undriven.",
    )
    parser.add_argument(
        "--debug_record",
        action="store_true",
        help="Record per-step commanded targets (pos/kp/kd) + measured proprioception "
        "(joint pos/vel/torque, base quat/ang-vel/lin-acc) and write a .npz on exit. "
        "For offline diagnosis (commanded-vs-measured, torque saturation, limit cycles).",
    )
    parser.add_argument(
        "--debug_record_path",
        type=str,
        default=None,
        help="Destination .npz for --debug_record (default: <log_dir>/deploy_record.npz).",
    )
    parser.add_argument(
        "--debug_record_flush_steps",
        type=int,
        default=None,
        help="Steps per streamed --debug_record chunk (backend default 100 = 2 s at 50 Hz). The "
        "recording is written as the run goes, so this is the most a Ctrl-C or a crash can cost; "
        "chunks left by a run that never closed are read back by "
        "eden.extensions.deployment.utils.record.load_deploy_record.",
    )
    parser.add_argument(
        "--target_lpf",
        type=float,
        default=1.0,
        help="First-order low-pass on commanded joint targets: sent = a*new + (1-a)*prev "
        "(a in (0,1]; 1.0 = off). Attenuates high-frequency command content (e.g. a single-frame "
        "policy's half-rate limit cycle / motor whine) while leaving kp/kd — and standing "
        "stiffness — intact. Try 0.5 / 0.3 with full --deploy_kp_kd_scale.",
    )
    parser.add_argument(
        "--part_option",
        action="append",
        default=None,
        metavar="PART.OPTION=VALUE",
        help="Composite rigs only: set one option of one part's backend, e.g. "
        "--part_option robot.robot_ip=172.16.0.2 --part_option hand.protocol=EtherCAT. Repeatable. "
        "VALUE is read as the option's declared type: a str option keeps it as typed, a bool takes "
        "true/false, a number or a tuple is read as a Python literal; a value of another type is refused.",
    )
    parser.add_argument(
        "--mujoco",
        action="store_true",
        help="Deploy against MuJoCo instead of hardware (sim-to-sim): run the exact deploy loop "
        "against a MuJoCo model of the robot — a robot-agnostic port of the idea behind "
        "unitree_mujoco, useful to validate a checkpoint against an independent physics engine "
        "before touching a real robot. Overrides the robot→backend binding, so it works for any "
        "--task / --robot. Combine with --vis to open MuJoCo's viewer.",
    )
    parser.add_argument(
        "--gantry",
        action="store_true",
        help="With --mujoco, hang the robot from a simulated overhead hoist — the sim "
        "counterpart of the bring-up rig, so a biped stays upright while you commission a "
        "controller instead of toppling whenever control pauses. The STOP hold latches and holds "
        "its joint pose, but no joint-space hold can balance a biped — the ankles would have to "
        "out-stiffen m*g*h about the ankle, and on a G1 they are about a quarter of it — so an "
        "unsupported robot topples the moment nothing balances it. A tension-only rope to a "
        "fixed anchor, "
        "attached above the centre of mass so it gives a real restoring torque rather than "
        "just cancelling weight. SIM-ONLY: it is the rig, not the robot — anything that only "
        "stands with --gantry does not stand.",
    )
    parser.add_argument(
        "--gantry_body",
        nargs="+",
        default=None,
        metavar="LINK",
        help="Bodies the --gantry ropes are tied to, one rope each. Default: the robot's own "
        "declared mount links — both shoulders for a humanoid that names them (Oli does), so it "
        "hangs from two points like a real spreader bar; a robot that declares none hangs from "
        "its base by one rope. Two points resist roll as a couple, which one rope cannot. "
        "Naming a link the model does not have lists the ones it does.",
    )
    parser.add_argument(
        "--gantry_attach",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Attachment point within each rope's body frame, metres. Default depends on the rig: "
        "the link origin for declared mount links (a shoulder origin sits on the pitch axis at "
        "the top of the torso and does not move as the arm swings), or 0 0 0.25 for a single "
        "base rope, which needs the height to have any moment arm at all.",
    )
    parser.add_argument(
        "--gantry_height",
        type=float,
        default=None,
        metavar="Z",
        help="World height of the --gantry anchor (default: 1 m above the attachment point). "
        "Raise it to take up rope and lift the robot, lower it to set the robot down.",
    )
    parser.add_argument(
        "--gantry_step",
        type=float,
        default=None,
        help="Metres the operator's + / - keys move the --gantry anchor per press (default 0.05).",
    )
    parser.add_argument(
        "--gantry_no_draw",
        action="store_true",
        help="Do not draw the --gantry ropes and their force arrows in the --vis viewer. They "
        "are drawn by default: a rope quietly carrying the robot looks exactly like a robot "
        "that is standing, right up until you unrig it.",
    )
    parser.add_argument(
        "--gantry_force_scale",
        type=float,
        default=None,
        help="Metres of --gantry force arrow per newton (default 1e-3, so 1 kN draws as 1 m).",
    )
    parser.add_argument(
        "--gantry_slack",
        type=float,
        default=None,
        help="Metres of slack in the --gantry rope. 0 (default) is taut and holds station; a "
        "positive value is a safety tether that does nothing until the robot drops that far.",
    )
    parser.add_argument(
        "--gantry_support",
        type=float,
        default=None,
        help="Scale on the --gantry rope force, 0-1 (default 1). Ramp down across runs to hand "
        "the robot's weight back to its own controller without re-rigging.",
    )
    parser.add_argument(
        "--imu_noise",
        nargs=2,
        type=float,
        default=None,
        metavar=("ACC", "GYRO"),
        help="With --mujoco, add white noise to the simulated IMU: per-axis standard deviation of "
        "the accelerometer (m/s^2) and the gyro (rad/s). MuJoCo's IMU is otherwise exact, so an "
        "--estimate_lin_vel run cannot show the drift a hardware IMU produces when foot contact "
        "is lost. Try 0.3 0.05.",
    )
    parser.add_argument(
        "--imu_bias",
        nargs=6,
        type=float,
        default=None,
        metavar=("AX", "AY", "AZ", "GX", "GY", "GZ"),
        help="With --mujoco, add a constant body-frame bias to the simulated accelerometer (m/s^2, "
        "first three) and gyro (rad/s, last three).",
    )
    parser.add_argument(
        "--imu_noise_seed",
        type=int,
        default=None,
        help="Seed for --imu_noise, so a noisy --mujoco run can be repeated.",
    )
    parser.add_argument(
        "--mujoco_props",
        nargs="*",
        default=None,
        metavar="ENTITY",
        help="With --mujoco on a composite rig: the scene entities to simulate besides the rig (default: every "
        "colliding one but the ground). Each must be a box or a rigid URDF / MJCF body with no joints; name only "
        "those to leave the rest out, or give none to simulate the rig alone.",
    )
    parser.add_argument(
        "--prop_noise",
        nargs=2,
        type=float,
        default=None,
        metavar=("POS", "ROT"),
        help="With --mujoco on a composite rig: what the policy reads of every free prop is a camera's pose "
        "estimate, not MuJoCo's exact pose -- per-axis position noise (m) and a rotation error (rad), as standard "
        "deviations, with a random quaternion sign. An AprilTag cube: 0.003 0.035.",
    )
    parser.add_argument(
        "--prop_camera",
        nargs=2,
        type=float,
        default=None,
        metavar=("HZ", "LATENCY_S"),
        help="With --mujoco on a composite rig: the free props' observed poses come at the camera's frame rate, "
        "held between frames, each LATENCY_S old when it arrives (rounded to control ticks). E.g. 30 0.04.",
    )
    parser.add_argument(
        "--prop_noise_seed",
        type=int,
        default=None,
        help="Seed for --prop_noise, so a noisy --mujoco run can be repeated.",
    )
    parser.add_argument(
        "--mujoco_record",
        type=str,
        default=None,
        metavar="MP4",
        help="With --mujoco: render the rollout from MuJoCo, from START on, to this mp4 (two views side by side). "
        "Lockstep only (not with --async). Headless, set MUJOCO_GL=egl.",
    )
    parser.add_argument(
        "--mujoco_seconds",
        type=float,
        default=None,
        help="With --mujoco: end the run once the policy has run this many seconds since its latest START, as "
        "KILL does -- a headless rollout ends by itself.",
    )
    parser.add_argument(
        "--mujoco_sim_dt",
        type=float,
        default=None,
        metavar="SECONDS",
        help="With --mujoco, MuJoCo's timestep (default: a single robot keeps its asset's; a composite rig takes the "
        "largest of at most 2 ms that divides the control period; a single robot's asset timestep that does not "
        "divide it gives way to the largest under it that does). In lockstep a given one must divide the control "
        "period. On a composite rig, Genesis' substep (sim_dt / sim_substeps) floors a contact's time constant as "
        "Genesis does.",
    )
    parser.add_argument(
        "--menagerie",
        action="store_true",
        help="With --mujoco, load the robot's community-tuned MuJoCo Menagerie model "
        "(armature/damping/contacts tuned for MuJoCo — higher-fidelity sim-to-sim) from HF "
        "instead of its own Genesis training asset. Errors if the robot has no Menagerie "
        "mapping or the download fails.",
    )
    parser.add_argument(
        "--on_motion_end",
        choices=("continue", "stand", "switch"),
        default=None,
        help="What a motion-imitation deploy does once the clip reaches its last frame. "
        "'continue' (default) keeps running the policy against a reference that has stopped "
        "moving. 'switch' hands control to --motion_end_slot through the same queue the "
        "operator's remote/keyboard uses -- point it at a loco/stand policy (e.g. --policy "
        "limx_oli_loco at zero command) and the robot keeps its balance after the clip. "
        "'stand' instead disengages the policy and ramps the joints to the default stance over "
        "--stand_steps; that holds a shape, not a stand, so a biped topples within seconds -- "
        "it suits a quadruped or a fixed-base arm. All three run off the motion clock and send "
        "ordinary wire commands, so hardware and --mujoco behave identically.",
    )
    parser.add_argument(
        "--motion_end_slot",
        type=int,
        default=None,
        help="1-based slot --on_motion_end switch hands to, numbered like the operator keys 1-9 "
        "(default 1). With --base_policy none, slot 1 is --checkpoint and the --policy "
        "extensions follow, so a single extension policy is slot 2.",
    )
    parser.add_argument(
        "--policy_checkpoint",
        action="append",
        default=None,
        metavar="NAME=PATH",
        help="Point an extension policy at a local checkpoint instead of its released weights, "
        "e.g. --policy_checkpoint limx_oli_loco=/path/model.pt. Repeatable. Required for a "
        "policy that ships no default checkpoint.",
    )
    parser.add_argument(
        "--stand_steps",
        type=int,
        default=None,
        help="Control cycles the --on_motion_end stand ramp takes (default 100 = 2s at 50 Hz). "
        "Longer is gentler on hardware.",
    )
    parser.add_argument(
        "--init_steps",
        type=int,
        default=None,
        help="Control cycles the INIT ramp takes to reach the start pose before the policy takes "
        "over (default 250 = 5 s at 50 Hz, on hardware and --mujoco alike; 0 disables the ramp). "
        "Every backend runs the INIT state.",
    )
    parser.add_argument(
        "--init_target",
        choices=("default", "motion", "motion_rsi"),
        default=None,
        help="Where the INIT ramp drives the joints before the policy takes over. 'default' "
        "(default) targets the robot's default_dofs_pos. 'motion' targets frame 0 of the tracked "
        "clip, so a motion-imitation policy starts from the pose it was trained to continue from "
        "instead of 100+ degrees away -- a joint-space ramp over the wire, identical on hardware "
        "and --mujoco. It is NOT automatically safe: frame 0 of a crouched clip is an open-loop "
        "squat with only the joint PD holding the base, and it can topple -- ramp slowly via "
        "--init_steps and support the robot the first time. 'motion_rsi' additionally teleports "
        "the base to frame 0; that is sim-only and --mujoco is the only backend that accepts it.",
    )
    parser.add_argument(
        "--async",
        action="store_true",
        dest="mujoco_async",
        help="With --mujoco, run the physics on its own thread in wall-clock time, like a robot's "
        "controller, instead of in lockstep with the control loop: the simulated robot keeps "
        "executing the last command while the policy is late and publishes state at 1/sim_dt "
        "regardless of when the policy reads it. Reproduces control-period overruns, command "
        "latency and state age the way hardware does; the default lockstep mode is "
        "deterministic and unaffected by the loop's timing. The physics thread reports if it "
        "cannot keep up with wall time.",
    )
    parser.add_argument(
        "--publish_rate",
        type=float,
        default=None,
        help="LimX Oli / Unitree G1 / Unitree Go2: rate at which the held wire command is re-sent between "
        "control ticks (backend defaults are each vendor's own low-level rate -- 1000 Hz for Oli, 500 Hz for "
        "the G1 and Go2). The Oli and G1 evaluate the PD of their parallel-linkage joints (ankle and waist "
        "roll/pitch) per received packet, so 50 Hz (one packet per tick) makes their damping act on a "
        "20 ms-old velocity and the linkage rings at 25 Hz; the Go2 has no such joints and re-sends for "
        "vendor-rate parity only. Pass 0 to send one packet per tick, for A/B only.",
    )
    parser.add_argument(
        "--power_protect_factor",
        type=int,
        default=None,
        help="Unitree Go1 only: Safety.PowerProtect factor, 1..10 == 10%%..100%% of rated motor "
        "power (backend default 10 = 100%%, no derating). LOWER it for cautious bench bring-up "
        "with the robot supported — but a factor set too low clamps the gait's torque and trips "
        "'Error: Power Protection' (the SDK examples' value of 1 = 10%% starves locomotion "
        "entirely). Ignored by non-Go1 backends.",
    )
    args = parser.parse_args(extra_args)
    if args.power_protect_factor is not None and not 1 <= args.power_protect_factor <= 10:
        parser.error("--power_protect_factor must be an integer in 1..10.")
    # The flags that only a MuJoCo sim-to-sim run reads: flag -> (its argparse attribute, what it does).
    mujoco_only = {
        "menagerie": ("menagerie", "selects the MuJoCo model to load"),
        "gantry": ("gantry", "is a simulated rig"),
        "async": ("mujoco_async", "runs the MuJoCo physics on its own thread"),
        "mujoco_sim_dt": ("mujoco_sim_dt", "sets MuJoCo's timestep"),
        "mujoco_props": ("mujoco_props", "picks the scene entities MuJoCo simulates"),
        **{name: (name, "corrupts the simulated IMU") for name in ("imu_noise", "imu_bias", "imu_noise_seed")},
        **{
            name: (name, "acts on a MuJoCo sim-to-sim run")
            for name in ("prop_noise", "prop_camera", "prop_noise_seed", "mujoco_record", "mujoco_seconds")
        },
    }
    for flag, (attr, what) in mujoco_only.items():
        value = getattr(args, attr)
        if value is not None and value is not False and not args.mujoco:
            parser.error(f"--{flag} {what}; it requires --mujoco.")
    for name in (
        "gantry_body",
        "gantry_attach",
        "gantry_height",
        "gantry_step",
        "gantry_force_scale",
        "gantry_slack",
        "gantry_support",
    ):
        if getattr(args, name) is not None and not args.gantry:
            parser.error(f"--{name} tunes the --gantry rope; it requires --gantry.")
    if args.gantry_no_draw and not args.gantry:
        parser.error("--gantry_no_draw turns off the --gantry overlay; it requires --gantry.")
    if args.gantry_force_scale is not None and args.gantry_force_scale <= 0:
        parser.error("--gantry_force_scale must be positive.")
    if args.gantry_step is not None and args.gantry_step <= 0:
        parser.error("--gantry_step must be positive.")
    if args.gantry_slack is not None and args.gantry_slack < 0:
        parser.error("--gantry_slack must be non-negative.")
    if args.gantry_support is not None and not 0.0 <= args.gantry_support <= 1.0:
        parser.error("--gantry_support must be in 0..1.")
    if args.init_steps is not None and args.init_steps < 0:
        parser.error("--init_steps must be a non-negative integer (0 disables the ramp).")
    if args.stand_steps is not None and args.stand_steps <= 0:
        parser.error("--stand_steps must be a positive integer.")
    if args.stand_steps is not None and args.on_motion_end != "stand":
        parser.error("--stand_steps sets the length of the --on_motion_end stand ramp; it requires it.")
    if args.motion_end_slot is not None and args.on_motion_end != "switch":
        parser.error("--motion_end_slot names the slot --on_motion_end switch hands to; it requires it.")
    if args.motion_end_slot is not None and not 1 <= args.motion_end_slot <= _MAX_DEPLOY_SLOTS:
        parser.error(f"--motion_end_slot must be in 1..{_MAX_DEPLOY_SLOTS}.")
    for spec in args.policy_checkpoint or []:
        if "=" not in spec:
            parser.error(f"--policy_checkpoint takes NAME=PATH; got {spec!r}.")
    if args.policy_checkpoint and not args.extension_policy:
        parser.error("--policy_checkpoint redirects a --policy extension policy; none was given.")
    if args.imu_noise is not None and min(args.imu_noise) < 0:
        parser.error("--imu_noise takes standard deviations; both must be >= 0.")
    try:
        parts_options = _parse_part_options(args.part_option or ())
    except ValueError as exc:
        parser.error(str(exc))
    # The backends refuse these values too; checked here as well so a bad flag fails before the env is built.
    if args.prop_noise is not None and min(args.prop_noise) < 0:
        parser.error("--prop_noise takes standard deviations; both must be >= 0.")
    if args.prop_camera is not None and (args.prop_camera[0] <= 0 or args.prop_camera[1] < 0):
        parser.error("--prop_camera takes a frame rate > 0 and a latency >= 0.")
    if args.mujoco_seconds is not None and args.mujoco_seconds <= 0:
        parser.error("--mujoco_seconds must be > 0.")
    if args.mujoco_sim_dt is not None and args.mujoco_sim_dt <= 0:
        parser.error("--mujoco_sim_dt must be > 0.")
    if parts_options and args.mujoco:
        parser.error("--part_option configures a composite rig's hardware backends; --mujoco has none.")
    if args.feet_links and not args.estimate_lin_vel:
        parser.error("--feet_links only feeds the base-linear-velocity estimator; it requires --estimate_lin_vel.")
    if args.velocity_command_scale and args.velocity_command == "task":
        parser.error(
            "--velocity_command_scale bounds the substituted command; it cannot apply to --velocity_command task."
        )
    if args.velocity_command_scale and any(v <= 0.0 for v in args.velocity_command_scale):
        parser.error("--velocity_command_scale takes three positive limits (VX VY WZ).")
    if args.gamepad_layout != "ps4" and args.velocity_command != "gamepad":
        parser.error("--gamepad_layout only applies to --velocity_command gamepad.")
    if tuple(args.base_command) != (0.0, 0.0, 0.0) and args.velocity_command in ("keyboard", "gamepad"):
        parser.error(
            "--base_command is the constant a substituted command holds; --velocity_command "
            f"{args.velocity_command} hands that command to the operator instead, so it would be "
            "ignored. Drop one of the two."
        )
    extension_policies = args.extension_policy or []
    if not args.task and not extension_policies:
        parser.error("Provide --task and/or --policy (an extension policy from POLICY_REGISTRY).")
    if args.task:
        _validate_task_arg(args, parser)
    elif args.checkpoint or args.extra_checkpoint:
        parser.error("--checkpoint requires --task (rsl_rl checkpoints load against the task's env).")
    if not args.checkpoint and args.extra_checkpoint:
        parser.error("--extra_checkpoint requires --checkpoint.")
    if not args.checkpoint and not extension_policies and args.base_policy == "none":
        parser.error("Provide at least one policy: --checkpoint, --policy, or --base_policy holosoma.")

    checkpoints = [ckpt for ckpt in [args.checkpoint, *(args.extra_checkpoint or [])] if ckpt]
    num_slots = len(checkpoints) + len(extension_policies) + (1 if args.base_policy != "none" else 0)
    if num_slots > _MAX_DEPLOY_SLOTS:
        parser.error(
            f"At most {_MAX_DEPLOY_SLOTS} policy slots (base + checkpoints + extensions) "
            "are supported for deploy switching."
        )

    import eden as en
    import genesis as gs

    en.init(
        backend=gs.gpu if not args.cpu else gs.cpu,
        log_root_path=_replay_log_root(args, task_less="logs/deploy"),
        performance_mode=True,
    )

    if args.task:
        # Prefer the checkpoint's own config.json (the exact trained config); fall back to
        # rebuilding from --task + mods when there's no sibling config.json.
        config = _config_from_checkpoint(args) or get_task_config_from_args(args, run_name=args.run_name)
        if _is_rl_games(config):
            sys.exit("'eden deploy' currently supports rsl_rl checkpoints only.")
        _strip_training_randomization(config)
        config.env_options.num_envs = 1
        config.env_options.episode_length_s = 1e9
        if hasattr(config.runner_options, "logger"):
            config.runner_options.logger = "tensorboard"
        # The deploy config snapshot is saved inside _run_deploy, after the
        # extension policies' command terms are merged in.
    else:
        # Task-less deploy: the env is built from the env-owning extension policy's
        # own spec methods inside _run_deploy (_env_owning_slot).
        config = None

    _run_deploy(
        config,
        checkpoints=checkpoints,
        show_viewer=args.show_viewer,
        entity_name=args.entity_name,
        deploy_kp_kd_scale=args.deploy_kp_kd_scale,
        gain_ramp_steps=args.gain_ramp_steps,
        base_policy=args.base_policy,
        base_command=tuple(args.base_command),
        extension_policies=extension_policies,
        robot_name=args.robot,
        motion_file=args.motion_file,
        estimate_lin_vel=args.estimate_lin_vel,
        feet_links=tuple(args.feet_links or ()),
        velocity_command=args.velocity_command,
        gamepad_layout=args.gamepad_layout,
        velocity_command_scale=tuple(args.velocity_command_scale or ()),
        disable_joints=tuple(args.disable_joints or ()),
        ignore_action_terms=tuple(args.ignore_action_terms or ()),
        debug_record=args.debug_record,
        debug_record_path=args.debug_record_path,
        debug_record_flush_steps=args.debug_record_flush_steps,
        target_lpf=args.target_lpf,
        power_protect_factor=args.power_protect_factor,
        publish_rate=args.publish_rate,
        mujoco=args.mujoco,
        menagerie=args.menagerie,
        mujoco_props=args.mujoco_props,
        init_steps=args.init_steps,
        init_target=args.init_target,
        on_motion_end=args.on_motion_end,
        stand_steps=args.stand_steps,
        motion_end_slot=args.motion_end_slot,
        policy_checkpoints=tuple(args.policy_checkpoint or ()),
        gantry=args.gantry,
        gantry_opts={
            "gantry_body": None if args.gantry_body is None else tuple(args.gantry_body),
            "gantry_attach": None if args.gantry_attach is None else tuple(args.gantry_attach),
            "gantry_height": args.gantry_height,
            "gantry_step": args.gantry_step,
            "gantry_draw": False if args.gantry_no_draw else None,
            "gantry_force_scale": args.gantry_force_scale,
            "gantry_slack": args.gantry_slack,
            "gantry_support": args.gantry_support,
        },
        mujoco_async=args.mujoco_async,
        mujoco_sim_dt=args.mujoco_sim_dt,
        parts_options=parts_options,
        imu_opts={
            "imu_acc_noise": None if args.imu_noise is None else args.imu_noise[0],
            "imu_gyro_noise": None if args.imu_noise is None else args.imu_noise[1],
            "imu_acc_bias": None if args.imu_bias is None else tuple(args.imu_bias[:3]),
            "imu_gyro_bias": None if args.imu_bias is None else tuple(args.imu_bias[3:]),
            "imu_noise_seed": args.imu_noise_seed,
        },
        rig_opts={
            "--prop_noise": {
                "prop_pos_noise": None if args.prop_noise is None else args.prop_noise[0],
                "prop_rot_noise": None if args.prop_noise is None else args.prop_noise[1],
            },
            "--prop_camera": {
                "prop_camera_hz": None if args.prop_camera is None else args.prop_camera[0],
                "prop_latency_s": None if args.prop_camera is None else args.prop_camera[1],
            },
            "--prop_noise_seed": {"prop_noise_seed": args.prop_noise_seed},
        },
        mujoco_record=args.mujoco_record,
        mujoco_seconds=args.mujoco_seconds,
    )


def _parse_part_options(specs) -> dict[str, dict]:
    """``--part_option PART.OPTION=VALUE`` specs as ``{part: {option: "VALUE"}}``, for
    ``CompositeDeployment.parts_options``.

    Every value stays the string typed: only the part's backend knows an option's type, so
    ``CompositeDeployment`` converts each against its declaration -- a ``str`` option keeps
    ``12345678`` as a serial number, an ``int`` one reads ``2``.
    """
    parsed: dict[str, dict] = {}
    for spec in specs:
        key, sep, raw = spec.partition("=")
        part, dot, option = key.partition(".")
        if not sep or not dot or not part or not option:
            raise ValueError(f"--part_option takes PART.OPTION=VALUE (e.g. hand.protocol=EtherCAT); got {spec!r}.")
        parsed.setdefault(part, {})[option] = raw
    return parsed


def _deploy_disabled_event_classes() -> tuple[type, ...]:
    """Training-only event terms ``eden deploy`` disables: every term of ``domain_rand``.

    Most only perturb the deploy env's own physics, but not all: :class:`RandomizeDofsPosOffset` and
    :class:`RandomizeStartupDofsPosBias` move every position target the robot is sent. A subclass is disabled with
    its base, so a task-local subclass that sets a pose the task needs, rather than randomizing one, must not
    derive from these.
    """
    from eden.managers.terms.events import domain_rand as dr

    return (
        dr.ERFI50,
        dr.ApplyExternalForce,
        dr.ApplyExternalTorque,
        dr.PushByVelocity,
        dr.PushSetPlanarVelocity,
        dr.RandomizeComShift,
        dr.RandomizeConstantTorqueKick,
        dr.RandomizeDeadbandEpsilon,
        dr.RandomizeDofsPosOffset,
        dr.RandomizeFrictionRatio,
        dr.RandomizeGearBacklash,
        dr.RandomizeKpKdGains,
        dr.RandomizeLinkMassScale,
        dr.RandomizeMassShift,
        dr.RandomizeMotorStrength,
        dr.RandomizeStartupDofsPos,
        dr.RandomizeStartupDofsPosBias,
        dr.RandomizeTorqueNoise,
        dr.SetRandomDofsPos,
    )


def _deploy_disabled_reset_functions() -> frozenset[str]:
    """Registry keys of the reset functions ``eden deploy`` disables on a robot: every term of ``resets``.

    Each draws a random initial state. Function terms have no class to test, so they are matched by key.
    """
    import inspect

    from eden.managers.event_manager import EVENT_TERM_REGISTRY
    from eden.managers.terms.events import resets

    return frozenset(
        name
        for name, obj in vars(resets).items()
        if inspect.isfunction(obj) and obj.__module__ == resets.__name__ and name in EVENT_TERM_REGISTRY
    )


def _strip_training_randomization(config) -> dict[str, str]:
    """Disable ``config``'s training-only event terms for deploy, log them, and return them as ``{key: term name}``.

    A term is matched by what it is, never by the key a task stores it under: the same
    :class:`RandomizeDofsPosOffset` is ``encoder_bias`` in one task and ``dofs_default_pos`` in another.
    Disabled: every domain-randomization term, and a randomized reset (:func:`_deploy_disabled_reset_functions`)
    of a robot entity, since the deploy env mirrors the robot's *actual* state. Object placement is scene setup,
    not robot state, and stays on, as do the ``placement`` terms.
    """
    import eden as en
    from eden.managers.event_manager import EVENT_TERM_REGISTRY
    from eden.options.entities import RobotOptions

    if config.event_options is None:
        return {}
    disabled_classes = _deploy_disabled_event_classes()
    reset_functions = _deploy_disabled_reset_functions()
    robots = {
        name for name, entity in config.scene_options.named_entities().items() if isinstance(entity, RobotOptions)
    }
    stripped: dict[str, str] = {}
    for key in config.event_options.term_keys():
        options = getattr(config.event_options, key)
        if options.name in reset_functions:
            disable = (options.params or {}).get("entity_name") in robots
        else:
            term = EVENT_TERM_REGISTRY.get(options.name)
            disable = isinstance(term, type) and issubclass(term, disabled_classes)
        if disable:
            config.event_options.disable_term(key)
            stripped[key] = options.name

    remaining = list(config.event_options.term_keys())
    if stripped:
        en.logger.info(f"[deploy] Stripped randomization event terms: {stripped}")
    if remaining:
        en.logger.info(f"[deploy] Remaining active event terms: {remaining}")
    else:
        en.logger.info("[deploy] No event terms remain active.")
    return stripped


class _DeploySlot:
    """One switchable deploy policy slot.

    ``act(obs)`` returns the action to feed ``dep.step`` — a raw tensor for
    rsl_rl checkpoints, or ``None`` when the slot already wrote absolute
    targets via ``dep.write_dofs_targets``. ``on_activate()`` is the handover
    hook run when the operator switches to this slot (reset internal policy
    state, re-arm commands — the pattern from ``examples/policies/policy_switch/main.py``).

    ``gains`` is the ``(kp, kd)`` pair the slot is driven at, in deployment DOF order and
    ready for the wire (``DeploymentBase.wire_gains``): the policy's own, which a switch to
    the slot ramps to (:func:`_deploy_control_loop`).
    """

    def __init__(self, label: str, act, on_activate=None, *, gains: tuple[np.ndarray, np.ndarray]) -> None:
        self.label = label
        self.act = act
        self.on_activate = on_activate or (lambda: None)
        self.gains = gains


def _resolve_extension_policy(name: str):
    """Resolve ``name`` to its ``POLICY_REGISTRY`` class, importing extension modules lazily.

    Accepted spellings, in resolution order:

    - the full registry key (``holosoma_locomotion_policy``),
    - the key without its ``_policy`` suffix (``holosoma_locomotion``),
    - the extension package name from ``eden.extensions.policies``
      (``holosoma``, ``sonic``, ...) — valid while the package registers
      exactly one policy; ambiguous packages list their keys instead.

    Extension policies register on import of their ``policy`` module (the
    ``eden.extensions.policies`` namespace package has no eager imports), so an
    unresolved name triggers a sweep over ``eden.extensions.policies.<pkg>.policy``
    — packages whose name prefixes the requested key first, the rest as a
    fallback. Import failures (missing optional deps) only surface if the name
    stays unresolved.
    """
    import pkgutil

    import eden as en
    import eden.extensions.policies as policies_pkg
    from eden.extensions.policies.base import POLICY_REGISTRY

    def _lookup():
        for key in (name, f"{name}_policy"):
            if key in POLICY_REGISTRY.keys():
                return POLICY_REGISTRY.get(key)
        return None

    # Only packages that ship a ``policy`` module are extension policies; shared
    # infrastructure packages (``client``) live beside them and are skipped. Checked on
    # the filesystem so the probe never imports a package (importing pulls optional deps).
    subpackages = [
        m.name
        for m in pkgutil.iter_modules(policies_pkg.__path__)
        if m.ispkg and os.path.isfile(os.path.join(m.module_finder.path, m.name, "policy.py"))
    ]
    errors: dict[str, Exception] = {}

    def _import(pkg: str) -> None:
        try:
            importlib.import_module(f"eden.extensions.policies.{pkg}.policy")
        except Exception as exc:  # a broken/optional extension must not block the others
            errors[pkg] = exc

    if (cls := _lookup()) is not None:
        return cls

    if name in subpackages:
        _import(name)
        prefix = f"eden.extensions.policies.{name}."
        matches = sorted(k for k in POLICY_REGISTRY.keys() if POLICY_REGISTRY.get(k).__module__.startswith(prefix))
        if len(matches) == 1:
            return POLICY_REGISTRY.get(matches[0])
        if len(matches) > 1:
            sys.exit(f"Extension package '{name}' registers multiple policies; pick one of: {matches}")

    for pkg in sorted(subpackages, key=lambda p: not name.startswith(p)):
        _import(pkg)
        if (cls := _lookup()) is not None:
            return cls
    for pkg, exc in errors.items():
        en.logger.warning(f"[policy] Could not import extension policy package '{pkg}': {exc}")
    sys.exit(
        f"Unknown extension policy '{name}'. Registered policies: {sorted(POLICY_REGISTRY.keys())}; "
        f"package names are also accepted: {sorted(subpackages)}"
    )


def _policy_spec(cls, method: str, entity_name: str) -> dict:
    """Call a policy spec classmethod, filling its parameters from the policy's own attributes.

    A parameter naming one of the policy's configurable attributes gets that attribute, so a
    name is declared once: a hook restating it as a literal (``action_options(cls,
    hand_name="hand")``) would not follow a policy that renamed it. ``entity_name`` is the
    caller's, so it wins; anything else keeps its own default.
    """
    import inspect

    fn = getattr(cls, method)
    params = inspect.signature(fn).parameters
    declared = set(cls.get_parameter_names())  # the policy's configurable attributes
    kwargs = {name: getattr(cls, name) for name in params if name in declared}
    if "entity_name" in params:
        kwargs["entity_name"] = entity_name
    return fn(**kwargs)


def _command_options_accepts(cls, name: str) -> bool:
    """Whether the policy's ``command_options`` signature accepts keyword ``name``.

    Single source of truth for the clip-name plumbing so the command term and the policy
    instance can't disagree on whether a policy is a per-clip tracker (and silently desync).
    """
    import inspect

    method = getattr(cls, "command_options", None)
    if method is None:
        return False
    return name in inspect.signature(method).parameters


def _extension_command_options(cls, entity_name: str, motion_file: str | None, clip_name: str | None = None) -> dict:
    """Build a policy's command term options, plumbing the CLI params its signature accepts.

    Motion-tracking policies (e.g. SONIC) declare a command term that needs a
    reference motion — ``--motion_file`` is forwarded when the policy's
    ``command_options`` signature takes one, and demanded when it does.
    Per-clip trackers (e.g. beyondmimic_moves) take ``clip_name`` instead and
    resolve the file themselves; ``--clip_name`` is forwarded when given and the
    signature accepts it (otherwise the policy's default clip is used). The
    returned terms must exist on the env's command manager or the policy
    crashes mid-run on the robot.
    """
    import inspect

    method = getattr(cls, "command_options", None)
    if method is None:
        return {}
    params = inspect.signature(method).parameters
    kwargs = {}
    if "entity_name" in params:
        kwargs["entity_name"] = entity_name
    if "motion_file" in params:
        if motion_file is None:
            sys.exit(f"Extension policy '{cls.__name__}' tracks a reference motion; pass --motion_file.")
        kwargs["motion_file"] = motion_file
    if _command_options_accepts(cls, "clip_name") and clip_name is not None:
        kwargs["clip_name"] = clip_name
    return method(**kwargs)


def _merge_command_terms(config, cmd_options: dict, context: str) -> None:
    """Merge extension-policy command terms into the task config in place.

    Extension policies (e.g. SONIC's motion reference) need their command terms
    present on the env's command manager or the policy crashes mid-run on the
    robot. Terms the task already declares are kept as-is; missing ones are added.
    ``context`` is the log tag ("deploy" / "inference").
    """
    import eden as en

    existing_terms = set(config.command_options.term_keys()) if config.command_options is not None else set()
    for term_name, term_options in cmd_options.items():
        if term_name in existing_terms:
            en.logger.info(f"[{context}] Env already defines command term '{term_name}'; reusing it.")
        else:
            setattr(config.command_options, term_name, term_options)


def _task_terms_driving_policy_command(config, cls, cmd_options: dict) -> list[str]:
    """The task's command terms that generate the command an extension policy would otherwise take.

    A policy that reads the env's observations observes its command through the env's command terms.
    When the task already declares one of the policy's terms, :func:`_merge_command_terms` reuses the
    task's. If that term samples its own command (no ``set_command``, as the training task's
    ``SkateCommand`` on ``--task husky --policy husky``), the task's generator drives the policy, as
    it would a checkpoint trained on it, and the policy has no command it can hold.
    Empty without a task, for a raw-state policy (its ``set_command`` feeds its own buffer), for a
    policy that takes no command (``--base_command`` never reached it), when the task lacks the
    policy's terms (the policy's own are merged in and take the command as without a task), and when
    the task's term can hold a command itself.
    """
    from eden.managers import COMMAND_TERM_REGISTRY

    if config is None or not getattr(cls, "reads_env_observations", False) or not hasattr(cls, "set_command"):
        return []
    if config.command_options is None:
        return []
    existing_terms = set(config.command_options.term_keys())
    return [
        name
        for name in cmd_options
        if name in existing_terms
        and not hasattr(COMMAND_TERM_REGISTRY.get(getattr(config.command_options, name).name), "set_command")
    ]


def _set_policy_command(policy, command) -> None:
    """Set an extension policy's command, padding/truncating to its expected width.

    The CLI exposes a 3-tuple ``--base_command`` (vx, vy, wz), but some policies
    take a different width — HOMIE's 4-D ``[vx, vy, wz, height_delta]``, HUSKY's
    2-D ``[lin_vel_x, relative_heading]``. Pad the missing trailing entries with
    zeros (HOMIE's ``height_delta=0`` is the default standing height, the natural
    default) and truncate if a policy wants fewer. The width is read from the
    policy's ``_commands`` buffer; policies without ``set_command`` are left
    untouched.

    A policy with no ``_commands`` buffer gets the command at the CLI's own width, with a
    warning: that width is a guess, and a policy whose command is narrower (HUSKY's 2-wide one
    handed a 3-vector) fails inside ``torch`` a frame later with nothing naming the cause. A
    ``set_command`` that needs no per-env buffer is legitimate, so it cannot be fatal.
    """
    import torch

    from eden.utils.warning import warn_once

    if not hasattr(policy, "set_command"):
        return
    command = list(command)
    buffer = getattr(policy, "_commands", None)
    if buffer is None:
        warn_once(
            f"[policy] '{type(policy).__name__}' takes a command (it defines set_command) but "
            f"exposes no `_commands` buffer to size it from; passing --base_command through at "
            f"width {len(command)}. If the policy expects another width, name its per-env command "
            "buffer `_commands` as the in-tree command-driven policies do."
        )
        policy.set_command(torch.tensor(command, dtype=torch.float32, device=getattr(policy, "device", None)))
        return
    width = buffer.shape[-1]
    if len(command) < width:
        command = command + [0.0] * (width - len(command))
    elif len(command) > width:
        command = command[:width]
    policy.set_command(torch.tensor(command, dtype=torch.float32, device=buffer.device))


def _extension_scene_options(spec_cls, robot, entity_name: str):
    """Scene for a task-less policy run: the policy's robot plus the entities its ``step()`` reads.

    A policy whose observations are defined against something other than its own robot (the
    tool SimToolReal manipulates, HUSKY's skateboard) declares it in ``scene_entity_options()``;
    without them the policy builds fine and then dies resolving the entity on its first step.
    Separate from :func:`_build_extension_env` so the scene is checkable without a GPU.
    """
    import eden as en

    extra = _policy_spec(spec_cls, "scene_entity_options", entity_name)
    if entity_name in extra:
        sys.exit(
            f"{spec_cls.__name__}.scene_entity_options() declares an entity named "
            f"'{entity_name}', which is the robot's own slot in the scene. Rename that entity. "
            "(A `plane` entity is fine — it replaces the flat ground, e.g. with terrain.)"
        )
    if extra:
        en.logger.info(f"[policy] {spec_cls.__name__} adds scene entities: {sorted(extra)}.")
    return en.scenes.Empty(**{entity_name: robot}, **extra)


def _build_extension_env(
    spec_cls, robot_name: str, entity_name: str, show_viewer: bool, command_options: dict, cameras_options=None
):
    """Build a minimal env from an extension policy's spec methods (task-less deploy / inference).

    Mirrors the policy examples (e.g. ``examples/policies/holosoma/main.py``): an empty
    scene with the policy's robot and the other entities it reads, plus ``spec_cls``'s
    env / action / observation options and the merged command terms of *all* loaded
    extension policies.

    ``spec_cls`` is the run's *env-owning* policy (:func:`_env_owning_slot`), not
    necessarily the slot-1 one: the env publishes one observation layout and runs one action
    manager, so they have to be the layout and manager of whichever loaded policy actually
    reads them.

    The robot comes from ``spec_cls.robot_options()`` when the policy pins one
    (HOMIE/Holosoma bake in their training default pose + PD gains — using a
    stock ``G1Dofs29`` here spawns the wrong pose/gains and the robot collapses).
    ``--robot`` only chooses the ``eden.robots`` class for policies that don't.
    """
    import eden as en
    from eden.envs.base import RLEnvBase
    from eden.options import ActionManagerOptions, CommandManagerOptions, EnvOptions, ObservationManagerOptions

    robot = spec_cls.robot_options()
    if robot is None:
        robot_cls = getattr(en.robots, robot_name, None)
        if robot_cls is None:
            available = sorted(n for n in dir(en.robots) if not n.startswith("_") and n[:1].isupper())
            sys.exit(f"Unknown robot '{robot_name}' (not found in eden.robots). Available: {available}")
        robot = robot_cls()
    else:
        en.logger.info(f"[policy] Using {type(robot).__name__} configured by {spec_cls.__name__} (--robot ignored).")

    env = RLEnvBase(
        env_options=EnvOptions(
            num_envs=1,
            episode_length_s=1e9,
            **spec_cls.env_options(),
        ),
        scene_options=_extension_scene_options(spec_cls, robot, entity_name),
        action_options=ActionManagerOptions(**_policy_spec(spec_cls, "action_options", entity_name)),
        observation_options=ObservationManagerOptions(**_policy_spec(spec_cls, "observation_options", entity_name)),
        command_options=CommandManagerOptions(**command_options) if command_options else None,
        cameras_options=cameras_options,
        show_viewer=show_viewer,
        eval_mode=True,
    )
    env.build()
    return env


def _slot_controllers(env, entity_name: str, cls) -> dict:
    """Build a private instance of every action term the policy declares.

    A policy's ``step()`` speaks whatever action convention it was trained in: Holosoma
    emits absolute joint targets, BeyondMimic Moves emits the normalised action mjlab
    trained, to be scaled per joint and offset onto the robot's default pose. The slot
    therefore owns the policy's own controller and converts with it every step
    (``compute(raw)`` → ``processed_action``), which is what lets any controller deploy
    rather than only the absolute-target ones.

    *Private* because the env's action manager belongs to the task's rsl_rl checkpoint (or,
    task-less, to whichever policy built the env): routing a second policy's actions through
    it would silently apply the first policy's scaling and offset.

    Built for every slot, the env-owning one included, whose controllers are duplicates of
    the terms its own env action manager holds: they are what the pre-flight DOF and width
    checks in :func:`_extension_policy_slot` read, and that slot converts through the action
    manager rather than through these.

    Only joint-position PD controllers are accepted. The one channel a slot writes
    (``write_dofs_targets``) is the payload's joint-*position* field, so a velocity or torque
    term's command would be sent as positions with nothing to report it, and a
    :class:`VelocityFeedforwardPDController`'s feedforward half would be dropped. Widening
    that is a payload question, not a controller one: it means a slot writing the ``dofs_vel``
    and ``dofs_torque`` fields the backends already carry.

    Must run *before* the deployment is constructed: ``build()`` resolves DOF names against
    the sim entity, and the deployment swaps that entity for a ``RobotStateEntity``. The
    handles cached here are repointed at the live robot afterwards, by
    :func:`_repoint_slot_controllers`.
    """
    from eden.managers.action_manager import ACTION_TERM_REGISTRY
    from eden.managers.terms.actions.joint_actions import (
        ExplicitPDController,
        ImplicitPDController,
        VelocityFeedforwardPDController,
    )

    controllers = {}
    for term_name, options in _policy_spec(cls, "action_options", entity_name).items():
        term_cls = ACTION_TERM_REGISTRY.get(options.name)
        if issubclass(term_cls, VelocityFeedforwardPDController):
            sys.exit(
                f"Extension policy '{cls.__name__}' action term '{term_name}' is a {term_cls.__name__}, "
                "which commands a joint velocity alongside its position target. A deploy slot writes the "
                "payload's joint-position field only, so the feedforward half would be dropped in silence. "
                "Deploying it means teaching a slot to write the payload's dofs_vel field as well."
            )
        if not issubclass(term_cls, (ImplicitPDController, ExplicitPDController)):
            sys.exit(
                f"Extension policy '{cls.__name__}' action term '{term_name}' is a {term_cls.__name__}, "
                f"whose command is not a joint position. A deploy slot writes the payload's joint-position "
                "field only, so this policy cannot be deployed."
            )
        controller = term_cls(env=env, options=options)
        controller.build()
        controllers[term_name] = controller
    return controllers


def _check_policy_fits_env(env, entity_name: str, cls) -> None:
    """Fail fast when the env a policy is about to ride on is not the one it was trained for.

    The env is built once — from the task, or from the env-owning policy's spec
    (:func:`_env_owning_slot`) — and every other slot inherits it. Two of its properties are
    part of a policy's control path rather than scenery, and a mismatch in either is
    silently-wrong control, not a crash:

    - **The robot's default pose.** A controller whose reference is the default pose folds
      ``default_dofs_pos`` into its offsets — ``DEFAULT`` into the static offset, ``ACCUMULATE``
      into the target it integrates from — so the wrong pose shifts the whole control frame.
      A policy that builds proprioception at all subtracts a trained stance from the measured
      joint positions on the way in, whether the env computes that for it (a ``dofs_pos_rel``
      observation term) or it does its own subtraction off a baked-in constant
      (``HolosomaPolicy._build_obs``), so declaring observation terms puts the pose in its
      control path too. Checked only when one of those holds: a policy emitting absolute
      targets off its own defaults and reading nothing pose-relative is entitled to any
      spawn pose.

      Note ``declares_obs_terms`` below is deliberately *not* ``reads_env_observations``.
      That flag answers a different question — whose observation layout the env publishes
      (:func:`_env_owning_slot`) — and the two sets differ: Holosoma and HOMIE declare
      observation terms, read none of them at deploy, and still care about the spawn pose.
    - **The control rate.** A frozen policy is trained at a fixed control dt and has no way
      to notice it is being stepped at another one. Only checked against a rate the policy
      actually declares: one that names neither ``sim_dt`` nor ``decimation`` has claimed no
      rate, and reading :class:`EnvOptions`' defaults as its own would refuse it in the name
      of a number it never gave.
    """
    from eden.constants import ReferenceSource
    from eden.options import EnvOptions
    from eden.utils.string import first_few, resolve_matching_names_values

    name = cls.__name__
    spec_env = cls.env_options()
    env_options = env.config.env_options
    if "sim_dt" in spec_env or "decimation" in spec_env:
        # A half-declared rate is filled from the same defaults ``_build_extension_env`` would
        # build this policy's own env with, so the rate named below is the one it would run at.
        defaults = EnvOptions()
        policy_dt = spec_env.get("sim_dt", defaults.sim_dt) * spec_env.get("decimation", defaults.decimation)
        env_dt = env_options.sim_dt * env_options.decimation
        if abs(policy_dt - env_dt) > 1e-9:
            sys.exit(
                f"Extension policy '{name}' declares a {1 / policy_dt:.1f} Hz control rate but this deploy "
                f"runs at {1 / env_dt:.1f} Hz (sim_dt={env_options.sim_dt:g} × decimation="
                f"{env_options.decimation}), which comes from the task, or from the env-owning policy when "
                "there is no task. Deploy it on its own, alongside a policy of the same rate, or — when the "
                "slot that disagrees is the base policy the CLI added itself — with --base_policy none."
            )

    robot = cls.robot_options()
    declares_obs_terms = bool(_policy_spec(cls, "observation_options", entity_name))
    pose_referenced = (ReferenceSource.DEFAULT, ReferenceSource.ACCUMULATE)
    folds_default_pose = any(
        getattr(options, "reference_source", None) in pose_referenced
        for options in _policy_spec(cls, "action_options", entity_name).values()
    )
    # Absent on a composite (SimToolReal's arm + hand), whose joints belong to its parts and
    # have no single-entity pose to compare against — nothing to check, as opposed to a check
    # being skipped. An empty dict is a robot that pins no pose and passes the loop below on
    # its own.
    default_pose = getattr(robot, "default_dofs_pos", None) if robot is not None else None
    if default_pose is None or not (declares_obs_terms or folds_default_pose):
        return
    if not isinstance(default_pose, dict):
        # ``RobotOptions.default_dofs_pos`` is declared ``dict[str, float]``, so reaching here
        # means a policy handed back something the options class does not describe. Say so
        # rather than skipping in silence: the check exists because a wrong pose is
        # silently-wrong control, and quietly not running it is that same failure.
        sys.exit(
            f"Extension policy '{name}' declares default joint positions as "
            f"{type(default_pose).__name__}, not the dict[str, float] RobotOptions documents; "
            "the deploy's spawn-pose check cannot read it."
        )

    entity = env.entities[entity_name]
    try:
        idx, names, values = resolve_matching_names_values(default_pose, entity.dofs_name)
    except ValueError as exc:
        sys.exit(f"Extension policy '{name}' declares default joint positions this robot does not have: {exc}")
    # Every env's row, at the resolver's columns (one per DOF, so a ball joint's repeated name is no problem): per
    # DOF, the env furthest from the trained pose is the one reported.
    rows = entity.get_default_dofs_pos().tolist()
    mismatched = []
    for column, dof, value in zip(idx, names, values):
        spawns = [row[column] for row in rows]
        deviations = [abs(spawn - float(value)) for spawn in spawns]
        if max(deviations) > 1e-6:
            actual = spawns[deviations.index(max(deviations))]
            mismatched.append(f"{dof}: robot spawns at {actual:+.4f}, policy trained at {float(value):+.4f}")
    if mismatched:
        sys.exit(
            f"Extension policy '{name}' operates relative to the joint positions it was trained in, and "
            f"this deploy's robot spawns in another pose: {first_few(mismatched)}. The robot comes from "
            "the task, or from the env-owning policy when there is no task. Deploy it on its own, "
            "alongside a policy of the same stance, or — when the slot that disagrees is the base policy "
            "the CLI added itself — with --base_policy none."
        )


def _actor_obs_groups(env, checkpoints: list[str]) -> list[str]:
    """The env observation groups a ``--checkpoint`` actor is fed, or none when no checkpoint is deployed.

    rsl_rl concatenates ``runner_dict["obs_groups"]["policy"]`` into the actor's input and, when
    the config names no ``policy`` set, takes the env group called ``policy``
    (``rsl_rl.utils.resolve_obs_groups``). Every other group on the manager -- the critic's,
    typically -- is computed on the robot but read by nothing, so what it observes cannot
    reach the actuators and must not decide whether the deploy may start.
    """
    if not checkpoints:
        return []
    return list(env.runner_dict.get("obs_groups", {}).get("policy") or ["policy"])


def _base_lin_vel_readers(env, ext_specs, entity_name: str, actor_groups: list[str]) -> list[str]:
    """Every observation term on this deploy that feeds a policy the base linear velocity, by owner.

    Two places declare observation terms that will be computed on the robot and fed to a
    policy: the env's observation manager, of which only ``actor_groups`` (the checkpoint
    actor's input, :func:`_actor_obs_groups`) reach a network at deploy time, and each
    extension policy's ``observation_options`` -- Holosoma-style policies read the entity
    directly, and that declaration is the only statement of what they read. Both are matched
    on the term's registry key, ``base_lin_vel``, not on the group key it was given.
    """
    readers: list[str] = []
    manager = env.observation_manager
    for group in actor_groups:
        terms = manager._group_obs_terms[group]
        for key, term in zip(manager._group_obs_term_names[group], terms, strict=True):
            if getattr(getattr(term, "_options", None), "name", None) == "base_lin_vel":
                readers.append(f"env observation '{group}/{key}'")
    for label, cls in ext_specs:
        for group, options in _policy_spec(cls, "observation_options", entity_name).items():
            for key in options.keys():
                if getattr(getattr(options, key), "name", None) == "base_lin_vel":
                    readers.append(f"{label} ({cls.__name__}) observation '{group}/{key}'")
    return readers


def _check_base_lin_vel_source(dep, readers: list[str]) -> None:
    """Refuse a deploy whose observations read a base linear velocity nothing supplies.

    ``RobotStateEntity`` serves zeros for it unless the onboard estimator is on or the backend
    publishes it (:attr:`DeploymentBase.provides_base_lin_vel`). An IMU measures no velocity,
    so on every hardware backend that means ``--estimate_lin_vel``; a policy stepped on zeros
    there runs, badly, with nothing on the log to say why.
    """
    if not readers or dep.provides_base_lin_vel:
        return
    from eden.utils.string import first_few

    sys.exit(
        f"{first_few(readers)} read the base linear velocity, but {type(dep).__name__} publishes none and "
        "the onboard estimator is off, so they would read zeros for the whole run -- silently-wrong "
        "control, not a crash. Pass --estimate_lin_vel (and --feet_links <left> <right> if the robot "
        "declares no feet links); on --mujoco, keep provide_ground_truth_base on a floating-base robot."
    )


def _construct_extension_policy(env, entity_name: str, cls, command):
    """Instantiate a resolved extension policy class as a deploy slot.

    Returns ``(policy, {term_name: controller})`` — the policy's own controllers, which the
    slot runs its raw action through before writing the result to the robot. See
    :func:`_slot_controllers` for why the slot owns them and :func:`_check_policy_fits_env`
    for what the env has to provide.

    Must run *before* the deployment is constructed: policies resolve DOF / link indices by
    name against the sim entity in ``__init__``, and the deployment swaps that entity for a
    ``RobotStateEntity`` (no name lookups).
    """
    import eden as en

    name = cls.__name__
    _check_policy_fits_env(env, entity_name, cls)
    controllers = _slot_controllers(env, entity_name, cls)

    try:
        policy = cls(env=env, options=cls.configure())
    except Exception:
        en.logger.error(
            f"[deploy] Constructing extension policy '{name}' failed — it may require env terms "
            "the task config does not provide (e.g. a motion command term)."
        )
        raise
    _set_policy_command(policy, command)
    return policy, controllers


def _repoint_slot_controllers(constructed, sim_entity, live_entity) -> None:
    """Point each slot's controllers at real robot state after the deployment's entity swap.

    The controllers are built against the sim entity (they resolve DOF names, which a
    ``RobotStateEntity`` cannot do), so a controller whose reference source re-reads the
    entity every step — ``ReferenceSource.DELTA``, or an ``EnvelopeClip`` modifier clipping
    the target to a torque envelope — would otherwise read *simulated* joint positions while
    driving the real robot. Same treatment, through the same identity-based scan and the same
    modifier expansion, that :meth:`DeploymentBase._repoint_entity_refs` gives the env's own
    terms — these are private to the slot, so that pass does not reach them.
    """
    import eden as en
    from eden.extensions.deployment.base import cached_entity_refs, with_owned_modifiers

    terms = [term for _, _, controllers in constructed for term in controllers.values()]
    refs = cached_entity_refs(with_owned_modifiers(terms), sim_entity)
    for term, attr in refs:
        setattr(term, attr, live_entity)
    if refs:
        en.logger.info(
            f"[deploy] Repointed {len(refs)} slot-controller entity reference(s) to real robot state: "
            f"{[f'{type(term).__name__}.{attr}' for term, attr in refs]}"
        )


def _repoint_swapped_slot_controllers(constructed, sim_entities: dict, entities) -> None:
    """:func:`_repoint_slot_controllers` for every entity the deployment swapped, not only ``--entity_name``'s.

    ``sim_entities`` are the env's entities before the deployment was constructed. A composite
    rig's deployment swaps one per part, and a slot's controller may hold any of them -- the hand
    of an arm + hand rig, under its own entity name.
    """
    for name, sim_entity in sim_entities.items():
        if entities[name] is not sim_entity:
            _repoint_slot_controllers(constructed, sim_entity, entities[name])


def _probe_observations(env):
    """The observation dict the env publishes right now, to probe a policy's step contract with.

    ``step(None)`` is enough for a policy that reads robot state off the entity, but every
    policy that reads the *dict* raises on it — so a ``None`` probe would skip exactly the
    policies whose action contract most needs checking, leaving them to fail mid-run with the
    robot enabled. Feeding them the real dict lets the probe run.

    Has to run *before* the deployment's entity swap: afterwards the terms read a
    ``RobotStateEntity`` that holds no reading until ``connect()``.

    Computing leaves state behind, and a pre-flight check must leave none.
    ``update_history=False`` is a snapshot only once a history buffer *exists*; on the first
    call there is none, so ``compute`` initialises each one and backfills the whole window —
    here with a frame read off the simulated robot at its spawn pose. So the observation manager
    is reset afterwards, the same way the policy is reset after its own probe step, and the
    returned dict is detached from the buffers first.

    This is hygiene for the probe, not the guarantee the deploy relies on: the probe runs only
    ``if ext_specs``, and the same seeding arrives without it from the training wrapper's own
    ``self.unwrapped.reset()``. ``DeploymentBase.state_to_observation`` clears it on the
    deployment's first read, whatever the path.

    Returns ``None`` if the env publishes nothing or cannot compute — the probe then falls back
    to ``step(None)``, and the first control step validates what neither could.
    """
    import eden as en

    manager = getattr(env, "observation_manager", None)
    if manager is None:
        return None
    try:
        obs = manager.compute(update_history=False)
    except Exception as exc:  # noqa: BLE001 — this only shapes the probe input, it is not the check
        en.logger.warning(f"[deploy] Could not compute probe observations ({exc}); probing with None instead.")
        return None
    # Detached before the reset below, which would otherwise zero it under the probe's feet: a
    # group carrying history is returned *as* the buffer that was just initialised.
    probe = {
        name: {term: value.clone() for term, value in group.items()} if isinstance(group, dict) else group.clone()
        for name, group in obs.items()
    }
    manager.reset()
    return probe


def _validate_step_action(label: str, action: dict, controllers: dict) -> None:
    """Check what a policy's ``step()`` emitted against the terms it declared, naming the offender.

    Called from the pre-flight probe *and*, once, on the first real control step, because
    neither route out of :func:`_extension_policy_slot` can count on the probe having run: a
    policy that raises on every synthesizable input reaches the robot unprobed. Unchecked, an
    undeclared term arrives at ``ActionManager.compute`` as a bare ``KeyError`` (its named
    check is smoke-only) and a wrong width as a bare torch broadcast error — both mid-run,
    which for a deployment means with the robot already enabled.
    """
    unknown = sorted(set(action) - set(controllers))
    if unknown:
        sys.exit(
            f"Extension policy '{label}' step() emitted action term(s) {unknown} not declared in its "
            f"action_options() ({sorted(controllers)})."
        )
    for term_name, term_action in action.items():
        width, expected = term_action.shape[-1], controllers[term_name].action_dim
        if width != expected:
            sys.exit(
                f"Extension policy '{label}' step() emitted {width} action values for term '{term_name}', "
                f"expected {expected} (its action_options dofs)."
            )


def _extension_policy_slot(
    dep,
    label: str,
    policy,
    controllers: dict,
    command,
    command_terms=(),
    *,
    operator=None,
    owns_env: bool = False,
    probe_obs=None,
    robot=None,
) -> _DeploySlot:
    """Wrap a constructed extension policy as a deploy slot.

    Two routes from the policy's raw action to the robot, and which one a slot takes is
    :func:`_env_owning_slot`'s answer:

    - ``owns_env``: hand the action to the deployment, which runs the env's action manager
      exactly as it does for a checkpoint. The manager holds *this* policy's terms (the env
      was built from its spec), and running it is what keeps the env-computed observations
      the policy reads back — ``last_action`` above all — in step with what the robot was
      commanded. Converting privately here instead leaves ``last_action`` reading zeros
      forever, which is enough on its own to walk a G1 into the floor.
    - otherwise: convert with this policy's own controller (``controllers``, see
      :func:`_slot_controllers`) and write the targets straight to the robot, leaving an
      action manager that belongs to another policy out of the path.

    ``command_terms`` are the env command terms this policy declared (e.g.
    SONIC's motion reference); on every switch-in their clocks are rewound to
    frame 0 *before* the policy reset, so the policy re-enters at the motion
    start instead of wherever the clock drifted while another slot was active
    (the handover pattern from ``examples/policies/policy_switch/main.py``).

    With ``operator`` (an :class:`_OperatorCommandReader`, i.e. ``--velocity_command
    keyboard|gamepad``) the slot re-reads the device every step rather than holding the
    constant ``command`` it was armed with, so one stick drives whichever slot is active —
    the base loco policy and a task checkpoint alike — instead of only the one whose
    command lives on the env.

    ``robot`` is the policy's pinned ``robot_options()``, resolved by the caller while the env
    still held the sim entity; its PD gains are the slot's. ``None`` — a policy pinning
    no robot, or any policy under ``--task``, whose robot sets every slot's gains — runs at the
    deployed robot's.
    """
    import eden as en
    from eden.utils.string import first_few

    controlled = {dof for controller in controllers.values() for dof in controller.dofs_name}
    missing = sorted(controlled - set(dep.dofs_name))
    if missing:
        sys.exit(f"Extension policy '{label}' controls DOFs the robot does not have: {first_few(missing)}")
    # Only the direct route is silent about this. The deployment already warns about the DOFs
    # its action-term maps leave unreached, which is the same question for the owner slot.
    uncontrolled = sorted(set(dep.dofs_name) - controlled)
    if uncontrolled and not owns_env:
        en.logger.warning(
            f"[deploy] Extension policy '{label}' leaves {len(uncontrolled)} DOF(s) uncontrolled "
            f"(they hold the previous slot's targets): {first_few(uncontrolled)}"
        )

    # Probe the step contract once now, off-robot: a policy whose step() emits a term not
    # declared in action_options(), or an action of the wrong width, would otherwise fail
    # mid-run inside the control loop. ``probe_obs`` is the env's own observation dict
    # (:func:`_probe_observations`) so the policies that read one can be probed too. Probing
    # advances the policy's internal state one step against the sim-default pose, so it is
    # reset right after (and again via on_activate at boot / switch-in).
    #
    # A raise is a refusal only for a policy that consumes the dict. For that one the input is
    # exactly what it gets every control step, so failing on it is the policy failing on its
    # real input, and letting it through only moves the same failure to a moment when the robot
    # is enabled. A policy that discards the argument (``del obs``, and most in-tree ones do)
    # reads its state off the entity and the command manager instead, and those are not in
    # their run-time condition yet: the probe runs before ``dep.connect()`` and, on the
    # task-less path, before any ``reset()`` at all — ``env.build()`` loads the managers but
    # does not reset them. A raise there says nothing about whether the policy can run, so it
    # stays a warning and the first control step carries the check.
    probed = False
    reads_obs = getattr(policy, "reads_env_observations", False)
    try:
        probe = policy.step(probe_obs)
    except Exception as exc:  # noqa: BLE001 — a policy may reject any input we can synthesize
        if probe_obs is not None and reads_obs:
            sys.exit(
                f"Extension policy '{label}' could not step on the observations this env publishes: "
                f"{type(exc).__name__}: {exc}\nThat is the dict it is handed every control step, so "
                "the run would fail the same way once the robot was enabled. Check that the policy, "
                "its --clip_name / --motion_file and the env it owns agree."
            )
        en.logger.warning(
            f"[deploy] Could not pre-validate '{label}' step contract ({exc}); the first control step "
            "checks it instead."
        )
    else:
        _validate_step_action(label, probe, controllers)
        policy.reset()
        probed = True

    def _act(obs):
        nonlocal probed
        if operator is not None:
            _set_policy_command(policy, operator.command())
        action = policy.step(obs)
        if not probed:
            # The probe could not run, so this is the contract's first check. Once: the widths
            # are structural, and a deploy step is a hard-real-time budget.
            probed = True
            _validate_step_action(label, action, controllers)
        if owns_env:
            # Returned, not written: the control loop passes it to ``dep.step``, which runs it
            # through the env's action manager.
            return action
        for term_name, term_action in action.items():
            controller = controllers.get(term_name)
            if controller is None:
                raise KeyError(
                    f"Extension policy '{label}' step() emitted action term '{term_name}', which it does "
                    f"not declare in action_options() ({sorted(controllers)})."
                )
            controller.compute(term_action)
            dep.write_dofs_targets(controller.processed_action[0], controller.dofs_name)
        return None

    def _activate():
        for term in command_terms:
            clock = getattr(term, "time_steps", None)
            if clock is not None:
                clock.zero_()
        policy.reset()
        _set_policy_command(policy, operator.command() if operator is not None else command)

    return _DeploySlot(label, _act, _activate, gains=dep.wire_gains(robot))


def _check_pinned_gains(pinned_robots: list[tuple[str, Any]], env_robot) -> None:
    """Refuse a slot whose pinned robot's gains do not resolve onto the deployed robot's DOFs.

    Runs before the deployment is constructed, so the refusal lands before the entity swap and
    before an operator device is opened — ``DeploymentBase.wire_gains`` would otherwise
    raise the same error only once both have happened.

    ``env_robot`` is the env's options for the deployed entity. One with no ``dofs_name`` is a
    composite rig, which ``CompositeDeployment`` lays out from its parts -- or refuses in its own
    words, as it does SimToolReal's ``KukaSharpa``, whose parts no backend is bound to; this check
    steps aside for it rather than fail first on the missing attribute.

    Under ``--task`` a slot runs at the task robot's gains, not its pinned ones, but the pinned
    ones are still what :func:`_warn_on_task_robot_gains` compares it with, so they must be whole.
    """
    from eden.extensions.deployment.base import resolve_pinned_gains

    dofs_name = getattr(env_robot, "dofs_name", None)
    if dofs_name is None:
        return
    for label, robot in pinned_robots:
        try:
            resolve_pinned_gains(robot, dofs_name)
        except ValueError as exc:
            sys.exit(f"Extension policy '{label}' pins a robot this deploy cannot run at its own gains: {exc}")


#: Relative tolerance under which two slots' gains on a joint count as the same gain.
_GAIN_RTOL = 1e-4


def _gain_differences(dofs_name, values, other) -> list[tuple[str, float, float]]:
    """``(joint, value, other_value)`` for every joint where two gain arrays (both kp, or both kd) disagree."""
    return [
        (name, float(a), float(b))
        for name, a, b in zip(dofs_name, values, other, strict=True)
        if not np.isclose(a, b, rtol=_GAIN_RTOL)
    ]


def _differing_joints(dofs_name, gains, reference, ref_label: str) -> list[tuple[str, list[str]]]:
    """``(gain, ["<joint> (own <v>, <ref_label> <r>)", ...])`` for kp then kd, where two ``(kp, kd)`` pairs
    disagree.
    """
    differing = []
    for gain, own, ref in zip(("kp", "kd"), gains, reference, strict=True):
        joints = [f"{name} (own {a:.2f}, {ref_label} {b:.2f})" for name, a, b in _gain_differences(dofs_name, own, ref)]
        if joints:
            differing.append((gain, joints))
    return differing


def _warn_on_task_robot_gains(dep, pinned_robots: list[tuple[str, Any]]) -> None:
    """Warn for each extension policy a ``--task`` runs at other gains than its pinned robot's.

    With ``--task`` the task's robot sets the PD gains of every slot, extension policies
    included: a task robot is how a deploy retunes gains for the hardware, so a policy's
    ``robot_options()`` must not silently take that back. A policy trained at other gains then
    runs off them, which the operator has to hear at startup rather than infer from a fall.
    """
    import eden as en
    from eden.utils.string import first_few

    task_gains = dep.wire_gains()
    for label, robot in pinned_robots:
        for gain, joints in _differing_joints(dep.dofs_name, dep.wire_gains(robot), task_gains, "task"):
            en.logger.warning(
                f"[deploy] '{label}' runs at the --task robot's {gain}, not the one it was trained at — "
                f"{first_few(joints)}. With --task the task's robot sets every slot's gains; to run this policy at "
                "its own, give the task's robot those gains, or deploy it without --task."
            )


def _warn_on_unswitchable_slot_gains(dep, slots: list[_DeploySlot]) -> None:
    """Warn for each slot whose gains the robot will not run, on a backend that cannot switch gains.

    A slot switch ramps the commanded kp/kd to the new slot's own (:func:`_deploy_control_loop`),
    which reaches the motors only on a backend that sends each command's gains
    (``DeploymentBase.sends_command_gains``). The Wuji and XHand hands and the Franka program
    theirs once at connect or run their own loop, so one gain set runs whichever slot is active.

    Where the backend can say what that set is in a policy's units
    (``DeploymentBase.programmed_gains``: Wuji Hand 2's MIT gains), every slot is compared with
    it, a lone one and the boot slot included. Where it cannot (XHand firmware units, Franka
    ``arm_stiffness``, Hand 1's position loop), the slots are compared with the boot slot, since
    of two slots trained at different gains at most one can run at its own.
    ``deploy_kp_kd_scale`` and ``--disable_joints`` apply to every slot alike, so neither reads
    as a disagreement.

    On a composite rig only the joints of the parts that run their own gains are compared
    (``DeploymentBase.command_gain_dofs``): a switch does bring the new slot's gains to the rest.
    """
    import eden as en
    from eden.utils.string import first_few

    fixed = ~np.asarray(dep.command_gain_dofs(), dtype=bool)
    if not fixed.any():
        return
    programmed = dep.programmed_gains()
    if programmed is None and len(slots) < 2:
        return
    dofs_name = [name for name, is_fixed in zip(dep.dofs_name, fixed, strict=True) if is_fixed]

    def _on_fixed(gains):
        return tuple(np.asarray(gain)[fixed] for gain in gains)

    backend = type(dep).__name__
    if programmed is not None:
        reference = programmed
        compared = slots
        what = f"{backend} runs"
        ref_label = "robot"
        consequence = (
            f"{backend} programs one gain set at connect, so this slot runs at it whichever slot is active. To run "
            f"it at its own, set {backend}'s gain options to them, and deploy slots trained at other gains separately."
        )
    else:
        reference = slots[0].gains
        compared = slots[1:]
        what = f"the boot slot '{slots[0].label}'"
        ref_label = "boot"
        consequence = (
            f"{backend} does not take gains per command, so the robot runs one gain set whichever slot is active "
            "and at most one of the two runs at its own. Deploy them separately."
        )
    for slot in compared:
        for gain, joints in _differing_joints(dofs_name, _on_fixed(slot.gains), _on_fixed(reference), ref_label):
            en.logger.warning(
                f"[deploy] '{slot.label}' was trained at another {gain} than {what} — {first_few(joints)} — "
                f"but {consequence}"
            )


def _resolve_base_policy(base_policy: str, config, robot_name: str, entity_name: str) -> str:
    """Resolve the ``--base_policy`` choice against the robot the deploy actually runs on.

    Holosoma is a G1 29-DOF (``G1Dofs29``) velocity-tracking loco policy, so it is only
    valid when the deploy robot is a ``G1Dofs29``. The robot is the task's bound scene
    entity (``--task``) or the ``--robot`` class (task-less).

    - ``auto`` (default): load Holosoma for a G1Dofs29, drop it (return ``none``) otherwise.
    - ``holosoma``: force it; a non-G1 robot is a hard error (29-DOF policy can't drive it).
    - ``none``: unchanged.
    """
    import eden as en
    from eden.options.robots.unitree_g1 import G1Dofs29

    if base_policy == "none":
        return "none"

    if config is not None:
        robot_options = getattr(config.scene_options, entity_name, None)
    else:
        # Task-less deploy: the env robot is the env-owning policy's pinned robot if it has
        # one, else the --robot class (see _build_extension_env). Only --robot is read here,
        # so a policy pinning a non-G1 robot (limx_oli_loco's Oli) under the default
        # --robot G1Dofs29 still reads as a G1.
        robot_cls = getattr(en.robots, robot_name, None)
        robot_options = robot_cls() if robot_cls is not None else None
    is_g1 = isinstance(robot_options, G1Dofs29)
    robot_label = type(robot_options).__name__ if robot_options is not None else robot_name

    if is_g1:
        return "holosoma"
    if base_policy == "holosoma":
        sys.exit(
            f"--base_policy holosoma requires a G1Dofs29 robot, but the deploy robot is "
            f"'{robot_label}'. Use --base_policy none (or auto) for non-G1 robots."
        )
    en.logger.info(
        f"[deploy] base_policy=auto: robot '{robot_label}' is not a G1Dofs29 — no Holosoma "
        "base policy loaded (--checkpoint / --policy is slot 1)."
    )
    return "none"


def _resolve_extension_specs(base_policy: str, extension_policies) -> tuple[list[tuple[str, type]], bool]:
    """Resolve --policy / base-policy names to ``(label, class)`` specs, deduped by class.

    The base policy is just the holosoma extension pinned to slot 1 / boot-active; listing it
    again via --policy (under any accepted spelling) keeps a single slot, so dedupe by resolved
    class rather than by name. Returns ``(ext_specs, has_base)``.
    """
    ext_specs: list[tuple[str, type]] = [
        (f"extension:{name}", _resolve_extension_policy(name)) for name in extension_policies
    ]
    has_base = False
    if base_policy == "holosoma":
        base_cls = _resolve_extension_policy("holosoma")
        if base_cls not in {cls for _, cls in ext_specs}:
            ext_specs.insert(0, ("holosoma base (loco)", base_cls))
            has_base = True
    return ext_specs, has_base


def _env_owning_slot(ext_specs, config, has_base: bool) -> int | None:
    """Pick the slot whose spec the deploy env is built from.

    The env is built once and every other slot inherits it, so this single choice fixes the
    observation layout, the action manager, the control cadence, and the standing pose every
    slot is driven at. (Not the PD gains: each slot carries its pinned robot's own, and a switch
    ramps to them — :func:`_deploy_control_loop`. The env robot's are only the fallback for a
    slot that pins none, a checkpoint's among them.)

    Two rules, in order:

    1. **A policy that reads the env's observations owns it.** ``step(obs)`` can only be driven
       by the layout the policy was trained on, so such a policy
       (``PolicyBase.reads_env_observations``) owns the env wherever it sits in the slot
       order — building from slot 1 regardless would hand BeyondMimic Moves the Holosoma
       base policy's env, which publishes no observation groups at all.
    2. **Otherwise the first policy the operator named wins**, ahead of the Holosoma base that
       :func:`_resolve_extension_specs` inserts for safety without being asked. A policy
       reading raw robot state rides along on any env's *observations*, but not on another's
       stance (:func:`_check_policy_fits_env` refuses that), and the policy the operator asked
       for is the one whose env they mean — a plain ``eden deploy --policy sonic`` runs in
       SONIC's env, not the auto-inserted base's. With no ``--policy`` at all the base is the
       only spec there is.

    The two rules can name different slots — ``--policy sonic --policy beyondmimic_moves``
    gives the env to BeyondMimic under rule 1. Rule 1 wins because an observation layout is
    unusable when wrong.

    Returns the *index* into ``ext_specs``, not the spec: two slots can hold the same policy
    class (``--policy sonic --policy sonic``), and an owner identified by class would make
    both of them owners. ``None`` when no extension policy is loaded at all.
    """
    import eden as en

    if not ext_specs:
        return None
    consumers = [i for i, (_, cls) in enumerate(ext_specs) if getattr(cls, "reads_env_observations", False)]
    if len(consumers) > 1:
        sys.exit(
            f"Policies {[ext_specs[i][0] for i in consumers]} each read the env's observations, and one env "
            "publishes one observation layout. Deploy them one at a time."
        )
    if not consumers:
        return 1 if has_base and len(ext_specs) > 1 else 0
    label, cls = ext_specs[consumers[0]]
    if config is not None:
        sys.exit(
            f"Policy '{label}' reads the env's observations, so it has to own the env — but --task "
            f"{type(config).__name__} owns it here, and would feed {cls.__name__} the task's observations "
            "instead of the ones it was trained on. Deploy the policy without --task."
        )
    en.logger.info(f"[deploy] Env built from '{label}': it reads the env's observations.")
    return consumers[0]


def _build_deploy_env(
    config, env_owner_cls, robot_name: str, entity_name: str, show_viewer: bool, merged_cmd_options: dict
):
    """Build the eval env for deployment, merging the extension policies' command terms into it.

    With a task config, snapshots the *deploy* config under a distinct filename so it can't be
    mistaken for a training-time ``config.json``. Saved after the command-term merge so the
    snapshot rebuilds the env that actually ran. (The deploy env's ``num_envs=1`` /
    randomization-free properties come from the ``eval_mode=True`` build below, not from this
    function — the snapshot just preserves whatever was built.)

    Without a task config there is no scene to read, so the env is built from
    ``env_owner_cls`` — the env-owning policy's own spec, per :func:`_env_owning_slot`. That
    choice reaches every other slot: its observation layout, its action manager, its control
    cadence (which :func:`_check_policy_fits_env` holds the other slots to), and the robot
    whose stance goes on the wire — and whose PD gains do for any slot that pins no robot of
    its own. A deploy that loaded no extension policy has no such spec, so it is refused here
    rather than indexed for — ``--checkpoint`` supplies a network but not the scene to run it in.
    """
    import eden as en
    from eden.envs.wrappers.rsl_rl_env import RslRlVecEnvWrapper

    if config is not None:
        _merge_command_terms(config, merged_cmd_options, context="deploy")
        config.save_as_json(os.path.join(en.log_dir, "deploy_config.json"))
        # `apply_seed=False`: deploy mirrors a live robot, so there is no run to reproduce and
        # no reason for env construction to reseed the process's RNGs behind the operator.
        return RslRlVecEnvWrapper.from_config(config, show_viewer=show_viewer, eval_mode=True, apply_seed=False)
    if env_owner_cls is None:
        sys.exit(
            "A task-less deploy builds its env from an extension policy's spec, and none was "
            "loaded: --checkpoint supplies a network but not the scene, robot or control rate to "
            f"run it in (--robot {robot_name} only names the robot class). Add --policy <name> for "
            "a frozen policy's own env, or --task <name> to deploy the checkpoint in its task's env."
        )
    return _build_extension_env(env_owner_cls, robot_name, entity_name, show_viewer, merged_cmd_options)


def _check_deployable_runner(config) -> None:
    """Exit, before anything is built, when the task's checkpoints load only into its own custom runner.

    :func:`_load_checkpoint_slots` loads every checkpoint into rsl_rl's stock ``OnPolicyRunner`` and runs its actor.
    A runner named by import path (``runner_options.class_name="module:Class"``) owns its checkpoint layout and its
    inference policy — the ConTrack fork's actor-critic, the extreme-parkour depth student's encoder cadence — so
    loading one of its checkpoints that way fails, at best after the deployment has taken the robot entity over.
    """
    class_name = getattr(config.runner_options, "class_name", None) or "OnPolicyRunner"
    if ":" in class_name:
        sys.exit(
            f"eden deploy loads checkpoints into rsl_rl's stock OnPolicyRunner and runs its actor, but "
            f"{type(config).__name__} trains with runner_options.class_name={class_name!r}, a custom runner that owns "
            "its checkpoint layout and inference policy, so eden deploy cannot run its checkpoints. Nothing was built "
            "or connected. Replay them in sim with `eden inference --task <task> --checkpoint <path>`."
        )


def _load_checkpoint_slots(env, dep, checkpoints: list[str]) -> list[_DeploySlot]:
    """Load every checkpoint into an independent policy slot, *before* any hardware contact.

    If a checkpoint is malformed (key mismatch, shape mismatch, unknown layout the migration
    helper couldn't fix), we want to abort with the files untouched rather than leaving the
    hardware in a half-connected state. Loading goes through ``load_runner_checkpoint`` —
    the same version-tolerant path as inference (``make_rsl_rl_runner``), which handles
    legacy-checkpoint migration and the ``strict`` kwarg difference across rsl-rl-lib
    versions.

    ``runner.load`` mutates the runner's policy module in place, so each loaded checkpoint is
    deep-copied into an independent module before the next load overwrites it. A checkpoint was
    trained in this env, so its slot runs at the env robot's PD gains (``dep.wire_gains()``).
    """
    import copy

    import eden as en
    from rsl_rl.runners import OnPolicyRunner

    from eden.envs.wrappers.rsl_rl_env import load_runner_checkpoint
    from eden.learning.wrappers.rsl_rl.flashsac.options import RslRlFlashSacRunnerOptions

    ckpt_slots: list[_DeploySlot] = []
    if not checkpoints:
        return ckpt_slots
    if isinstance(env.runner_options, RslRlFlashSacRunnerOptions):
        raise SystemExit(
            f"Deploying a FlashSAC policy (runner_options {type(env.runner_options).__name__}) is not supported: its "
            "tanh actions are scaled by the training wrapper, which the deploy loop bypasses. Run it in simulation "
            "with `eden inference --task <task> --checkpoint <ckpt>`, or deploy a PPO policy."
        )
    # The replay dict: a training-only algorithm extension (AMP-PPO) is not built on the robot host.
    runner = OnPolicyRunner(env, env.inference_runner_dict, log_dir=en.log_dir, device=env.device)
    for ckpt in checkpoints:
        load_runner_checkpoint(runner, ckpt)
        policy = copy.deepcopy(runner.get_inference_policy(device=env.device))
        # Best-effort handover hook: recurrent actor-critics expose ``reset()``
        # (clears hidden state); for MLPs it's a no-op. The inference policy is a
        # bound method, so the module hangs off ``__self__``.
        module = getattr(policy, "__self__", None)
        ckpt_slots.append(_DeploySlot(ckpt, policy, on_activate=getattr(module, "reset", None), gains=dep.wire_gains()))
    return ckpt_slots


def _switch_slot_gains(dep, slot: _DeploySlot) -> None:
    """Ramp the commanded kp/kd to ``slot``'s own on a live switch, logging the largest change the robot will feel.

    A no-op on a backend that does not take gains per command (``dep.sends_command_gains``),
    where ``set_gains`` would raise: the robot runs its own whichever slot is active, which
    :func:`_warn_on_unswitchable_slot_gains` reports at startup.
    """
    import eden as en

    if not dep.sends_command_gains:
        return
    changes = [
        (gain, *change)
        for gain, new, old in zip(("kp", "kd"), slot.gains, (dep.default_dof_kp, dep.default_dof_kd), strict=True)
        for change in _gain_differences(dep.dofs_name, new, old)
    ]
    dep.set_gains(*slot.gains)
    if changes:
        gain, name, new, old = max(changes, key=lambda change: abs(change[2] - change[3]))
        steps = dep.gain_ramp_steps
        how = f"Ramping kp/kd over {steps} control steps" if steps else "Switching kp/kd at once"
        en.logger.info(
            f"[deploy] {how} to the gains '{slot.label}' was trained at: {len(changes)} gain(s) change, "
            f"the largest {name} {gain} {old:.2f} -> {new:.2f}."
        )


def _deploy_control_loop(
    env, dep, slots: list[_DeploySlot], entity_name: str, show_viewer: bool, *, operator=None
) -> None:
    """Connect to hardware and drive the active policy until interrupted.

    Handles live slot switches (remote Select / keyboard) and re-arms the active policy on
    each START handover. With --vis on a motion-imitation task, overlays the reference motion
    as a translucent "ghost" skeleton (the tracked link positions) so the operator sees the
    target the policy is chasing vs. the solid robot.

    Each slot runs at its own PD gains (``_DeploySlot.gains``, :func:`_switch_slot_gains`). The
    boot slot's are set before ``dep.connect()``, so the robot is never commanded at any other;
    a switch ramps the commanded kp/kd to the new slot's over ``gain_ramp_steps`` control steps,
    because a stiffness step is enough to knock a robot over.
    """
    import torch

    import eden as en

    active = 0
    en.logger.info(f"[deploy] Loaded {len(slots)} policy slot(s); boot-active: {slots[0].label}")
    if len(slots) > 1:
        for i, slot in enumerate(slots):
            en.logger.info(f"[deploy]   slot {i + 1}: {slot.label}")
        en.logger.info("[deploy] Switch live via remote Select / keyboard n / 1-9.")

    def _apply_switches() -> bool:
        """Drain the deployment's pending policy-switch requests and update ``active``; whether it changed."""
        nonlocal active
        switched = False
        for req in dep.consume_policy_switches():
            target = (active + 1) % len(slots) if req == "next" else req
            if not isinstance(target, int) or not (0 <= target < len(slots)):
                en.logger.warning(f"[deploy] Ignoring out-of-range policy switch request: {req!r}")
                continue
            if target != active:
                active = target
                slots[active].on_activate()
                _switch_slot_gains(dep, slots[active])
                # Restart any motion-imitation reference at frame 0 so the new policy's clip plays from the
                # beginning, aligned with the robot's current pose at the switch (no-op for non-mimic envs).
                dep.reset_motion_reference()
                switched = True
                en.logger.info(f"[deploy] Switched to policy [{active + 1}/{len(slots)}]: {slots[active].label}")
        return switched

    motion_manager = getattr(env.unwrapped, "motion_manager", None)
    draw_ghost = show_viewer and motion_manager is not None
    ghost_node = None
    if draw_ghost:
        en.logger.info(f"[deploy] --vis: drawing motion reference ghost for entity '{entity_name}'.")

    if dep.sends_command_gains:
        dep.set_gains(*slots[active].gains, ramp_steps=0)  # nothing is on the wire yet, so no ramp
    dep.connect()
    try:
        obs = dep.reset()
        slots[active].on_activate()
        while True:
            # Re-arm the active slot at the START handover: START re-captures
            # the yaw offset, so any policy state accumulated while stepping
            # during STOP/hold references the pre-START world frame.
            if dep.consume_start_events():
                slots[active].on_activate()
                en.logger.info(f"[deploy] START — re-armed active policy: {slots[active].label}")
            if _apply_switches():
                # The switched-in slot reads the env's windows afresh, as at START, not frames another
                # slot drove (a raw-state slot ignores them).
                obs = dep.restart_observations()
            with torch.no_grad():
                raw_action = slots[active].act(obs)
            # The step synced the reading onto the sim entity (RobotStateEntity.update), which is what the viewer shows.
            obs = dep.step(raw_action)
            if show_viewer:
                if draw_ghost:
                    ghost_node = _draw_motion_ghost(env.unwrapped, motion_manager, entity_name, ghost_node)
                env.unwrapped.scene.visualizer.update()
    except KeyboardInterrupt:
        en.logger.info("[deploy] Interrupted — closing deployment.")
    except SystemExit as exc:
        # A slot refusing its own policy's action on the first control step
        # (:func:`_validate_step_action`). Logged here because ``sys.exit``'s own message
        # reaches stderr only once the interpreter unwinds, by which point Genesis' teardown
        # has scrolled it away — and re-raised so the ``finally`` below still settles the
        # robot and the exit code survives.
        en.logger.error(f"[deploy] {exc} Closing deployment.")
        raise
    finally:
        # Nested so the operator's device is released even if the transport teardown throws;
        # dep.close() still goes first, because settling the robot matters more than a
        # keyboard listener. A no-op unless the reader owns a standalone term — a substituted
        # one is closed and restored by dep.close().
        try:
            dep.close()
        finally:
            if operator is not None:
                operator.close()


def _apply_policy_checkpoints(specs: tuple[str, ...], ext_specs) -> None:
    """Redirect ``NAME=PATH`` extension policies at local checkpoints before construction.

    ``PolicyBase.use_local_checkpoint`` also clears the md5 pinned to the released weights,
    which an unreleased file would otherwise fail. Matching is by the same spellings
    ``--policy`` accepts, so the two flags can be written the same way.
    """
    import os

    if not specs:
        return
    by_cls = {cls: label for label, cls in ext_specs}
    for spec in specs:
        name, _, path = spec.partition("=")
        if not os.path.isfile(path):
            sys.exit(f"--policy_checkpoint {name}: no such file: {path}")
        cls = _resolve_extension_policy(name.strip())
        if cls not in by_cls:
            sys.exit(
                f"--policy_checkpoint names '{name}', which is not among the loaded --policy "
                f"extensions ({sorted(by_cls.values())})."
            )
        cls.use_local_checkpoint(path)


def _order_policy_slots(
    ext_slots: list[_DeploySlot], ckpt_slots: list[_DeploySlot], has_base: bool
) -> list[_DeploySlot]:
    """Order the deploy slots: base policy (if any) first / boot-active, then checkpoints,
    then the remaining extension policies.

    ``has_base`` means ``ext_slots[0]`` is the holosoma base loco policy (pinned to slot 1);
    checkpoints are inserted after it so they don't displace the boot-active base.
    """
    num_base = 1 if has_base else 0
    return ext_slots[:num_base] + ckpt_slots + ext_slots[num_base:]


_DEVICE_VELOCITY_COMMANDS = ("keyboard", "gamepad")


def _resolve_velocity_command_term(env, entity_name: str, mode: str):
    """The SE(2) velocity command term to substitute for ``entity_name``, as ``(name, term)``.

    ``(None, None)`` when there is nothing to substitute — a motion-imitation task, a
    task-less ``--policy`` deploy, or ``--velocity_command task``. Two candidates for one
    entity is an error: which one an operator drives would otherwise be an arbitrary
    dict-ordering choice, and the other would keep resampling at random behind their back.

    ``--velocity_command task`` returns before any of that. Nothing is substituted in that
    mode, so there is nothing to disambiguate and nothing to refuse — the env's own terms
    drive the robot exactly as they did in training, however many of them there are. It is
    also the escape hatch the two errors below recommend, which it cannot be if it trips
    them itself.

    It also finds nothing on an env a *live* deployment is already driving, whose term is a
    substitute rather than the generator. That case is not diagnosed here because the entity
    swap hits it first and harder — a second deployment wraps the ``RobotStateEntity`` in
    another one — and because the swap is undone by ``close()``, so a sequential
    construct/close/construct works.

    Matched by **exact type**, not ``isinstance``. A subclass of ``UniformVelocityCommand``
    exists to add behaviour — extra state, a gait phase, a second output — and none of that
    survives being replaced by a term that only holds a velocity; swapping it would discard
    the task's own logic silently. So a subclass is left running and said so, and asking for
    an operator device on one is refused rather than honoured wrongly. (The same reasoning
    ``EdenConfig.load_from_file`` documents for task identity: a subclass is not the class.)
    """
    import eden as en
    from eden.managers.terms.commands.velocity import UniformVelocityCommand

    manager = getattr(env, "command_manager", None)
    if manager is None or mode == "task":
        return None, None
    candidates = [
        (name, term)
        for name, term in manager.terms.items()
        if isinstance(term, UniformVelocityCommand) and term.entity_name == entity_name
    ]
    exact = [(name, term) for name, term in candidates if type(term) is UniformVelocityCommand]

    if len(exact) > 1:
        sys.exit(
            f"Cannot resolve the velocity command to deploy: entity '{entity_name}' is driven by "
            f"{len(exact)} UniformVelocityCommand terms ({[n for n, _ in exact]}). Deploy a config "
            "with one, or pass --velocity_command task and drive it yourself."
        )
    if exact:
        return exact[0]

    if candidates:
        subclasses = [f"{name} ({type(term).__name__})" for name, term in candidates]
        if mode in _DEVICE_VELOCITY_COMMANDS:
            sys.exit(
                f"--velocity_command {mode} cannot substitute {subclasses}: a UniformVelocityCommand "
                "*subclass* carries behaviour a held command would drop. Pass --velocity_command task "
                "and drive the term yourself."
            )
        en.logger.warning(
            f"[deploy] Leaving {subclasses} in place: it subclasses UniformVelocityCommand, so "
            "substituting a held command would drop whatever behaviour it adds. The task's own "
            "command generator drives the robot — pass --base_command through the task instead."
        )
    return None, None


#: Bound on a substituted velocity command when nothing else declares one — a task-less
#: ``--policy`` deploy, where the extension policy publishes no trained range anywhere Eden
#: can read. Deliberately modest; ``--velocity_command_scale`` is the way past it.
_DEFAULT_VELOCITY_RANGES = {
    "lin_vel_x_range": (-1.0, 1.0),
    "lin_vel_y_range": (-1.0, 1.0),
    "ang_vel_z_range": (-1.0, 1.0),
}


def _velocity_command_ranges(term, scale: tuple[float, ...]) -> dict:
    """Per-axis ranges for a substituted velocity command: ``scale``, the replaced term's, or the default.

    Clamping the substitute to the ranges the *trained* term declared is what keeps an
    operator — or a ``--base_command`` typo — inside the distribution the policy learned.
    ``--velocity_command_scale`` overrides them symmetrically, which is how you bound a robot
    below its trained speed while commissioning, and the only bound available when there is
    no term to read.

    Note what the term's own ranges are for a task with a command curriculum: the *initial*
    ones. The curriculum widens them at runtime and the saved config holds the starting
    values, so the ceiling is the curriculum's first stage until ``scale`` raises it.
    """
    if scale:
        return {
            "lin_vel_x_range": (-scale[0], scale[0]),
            "lin_vel_y_range": (-scale[1], scale[1]),
            "ang_vel_z_range": (-scale[2], scale[2]),
        }
    if term is None:
        return dict(_DEFAULT_VELOCITY_RANGES)
    return {
        "lin_vel_x_range": tuple(term.lin_vel_x_range),
        "lin_vel_y_range": tuple(term.lin_vel_y_range),
        "ang_vel_z_range": tuple(term.ang_vel_z_range),
    }


def _cap_base_command(base_command, term, scale: tuple[float, ...]):
    """``--base_command`` capped at the deploy's declared speed ceiling, per axis.

    ``--base_command`` reaches two consumers and only one of them bounds it. The substituted
    :class:`~eden.managers.terms.commands.velocity.ConstantVelocityCommand` trims it into the
    trained ranges itself; every command-driven **extension policy** takes it raw through
    ``set_command``. Without this, a value past the deploy's ceiling drives the checkpoint
    slot at the trimmed speed and the base loco policy at the typed one — and on a task-less
    ``--policy`` deploy, where there is no term at all, ``--velocity_command_scale`` bounds
    nothing whatsoever despite being documented as the only bound available there.

    A symmetric *ceiling* (``|v| <= max(|lo|, |hi|)``), not the range clamp the term applies
    to itself, because the two consumers are not the same policy. A range belongs to the task
    that declared it, and a task that pinned an axis — ``move_forward``'s
    ``lin_vel_x_range=(0.5, 0.5)`` — is saying what *its own* policy needs, not commanding a
    foreign base loco policy to walk when the operator asked it to stand. What does carry
    across every slot is the magnitude: nothing here should be driven faster than the fastest
    command this deploy declares.

    Capped when the operator asked for a limit (``scale``) or when a replaced term declares
    the range it was trained on. With neither — a task-less ``--policy`` deploy that passed
    no ``--velocity_command_scale`` — the command is returned untouched: nothing declares a
    ceiling, and :data:`_DEFAULT_VELOCITY_RANGES` is a fallback for an operator *device*, not a
    licence to cap a flag nothing has ever capped.

    Only ``constant`` mode reaches here. An operator mode has no ``--base_command`` (argparse
    refuses the combination) and seeds its term with a placeholder the device overwrites on
    the first ``compute``, so there is nothing to cap and the message would point at a value
    nothing ever holds.
    """
    import eden as en

    if not scale and term is None:
        return base_command
    ranges = _velocity_command_ranges(term, scale)
    ceiling = tuple(
        max(abs(lo), abs(hi))
        for lo, hi in (ranges["lin_vel_x_range"], ranges["lin_vel_y_range"], ranges["ang_vel_z_range"])
    )
    typed = tuple(float(v) for v in base_command)
    capped = tuple(min(max(v, -limit), limit) for v, limit in zip(typed, ceiling))
    if capped != typed:
        en.logger.warning(
            f"[deploy] --base_command {list(typed)} exceeds this deploy's per-axis ceiling "
            f"(vx {ceiling[0]}, vy {ceiling[1]}, wz {ceiling[2]}); using {list(capped)} instead."
        )
    return capped


def _velocity_command_options(mode: str, entity_name: str, ranges: dict, base_command, gamepad_layout: str):
    """Configured substitute term for ``--velocity_command <mode>``."""
    from eden.managers.terms.commands.velocity import ConstantVelocityCommand, OperatorVelocityCommand

    if mode in _DEVICE_VELOCITY_COMMANDS:
        return OperatorVelocityCommand.configure(
            entity_name=entity_name,
            device_name=mode,
            device_options={"layout": gamepad_layout} if mode == "gamepad" else {},
            **ranges,
        )
    return ConstantVelocityCommand.configure(entity_name=entity_name, command_value=tuple(base_command), **ranges)


class _VelocityCommandPlan(NamedTuple):
    """What a deploy decides about the SE(2) velocity command before any policy exists.

    ``term_name`` / ``ranges`` steer the operator reader, ``overrides`` is what the deployment
    substitutes (empty when there is nothing to substitute), and ``base_command`` is the
    ceiling-capped command every consumer must use.

    Produced in one step by :func:`_plan_velocity_command` and read in several, which is the
    point. As a bare reassignment among the dozen other things ``_run_deploy`` does, a capped
    command is easy to compute and then not use — handing a policy the unbounded value while
    the bound sits in a local nobody reads. A plan cannot be half-used.
    """

    term_name: str | None
    ranges: dict
    base_command: tuple[float, ...]
    overrides: dict


def _plan_velocity_command(env, entity_name: str, mode: str, base_command, scale, gamepad_layout: str):
    """Resolve which velocity command a deploy installs, and bound the command it will hold.

    Runs before a single policy is constructed, for two reasons. A mistake here — an ambiguous
    term, an operator device with nothing to drive — then raises with nothing swapped and no
    input device opened. And every command-driven extension policy is handed ``--base_command``
    in its *constructor*, so the ceiling has to be applied before any of them exist; see
    :func:`_cap_base_command` for what that ceiling is and why it is not the term's range.
    """
    term_name, term = _resolve_velocity_command_term(env, entity_name, mode)
    ranges = _velocity_command_ranges(term, scale)
    if mode == "constant":
        base_command = _cap_base_command(base_command, term, scale)
    overrides = {}
    if term is not None:
        overrides[term_name] = _velocity_command_options(mode, entity_name, ranges, base_command, gamepad_layout)
    return _VelocityCommandPlan(term_name, ranges, tuple(base_command), overrides)


class _OperatorCommandReader:
    """The live ``(vx, vy, wz)`` for extension-policy slots to follow.

    With a task, the substituted term is already on the command manager and ``dep.step``
    ticks it, so this just reads it — one device, whichever slot is active. Without a task
    (a ``--policy`` deploy) there is no manager term to substitute, so it owns a standalone
    :class:`~eden.managers.terms.commands.velocity.OperatorVelocityCommand` and ticks that
    instead; either way exactly one input device exists.
    """

    def __init__(self, term, *, owns_term: bool) -> None:
        self._term = term
        self._owns_term = owns_term

    @classmethod
    def for_deploy(cls, env, entity_name: str, mode: str, term_name: str | None, ranges: dict, gamepad_layout: str):
        """Build a reader, or ``None`` when no operator device is in play."""
        if mode not in _DEVICE_VELOCITY_COMMANDS:
            return None
        if term_name is not None:
            return cls(env.command_manager.get_term(term_name), owns_term=False)
        from eden.managers.terms.commands.velocity import OperatorVelocityCommand

        options = _velocity_command_options(mode, "", ranges, (0.0, 0.0, 0.0), gamepad_layout)
        term = OperatorVelocityCommand(env, options)
        term.build()
        return cls(term, owns_term=True)

    def command(self) -> list[float]:
        if self._owns_term:
            self._term.compute(dt=0.0)
        return self._term.command[0].tolist()

    def close(self) -> None:
        if self._owns_term:
            self._term.close()


def _rig_option_kwargs(rig_opts: dict | None, deployment_cls: type, entity_name: str) -> dict[str, Any]:
    """The composite rig's MuJoCo options the operator passed, keyed by the flag that set them, as configure kwargs.

    A flag whose options ``deployment_cls`` does not declare (a single robot's ``MujocoDeployment``)
    exits naming it, rather than being dropped.
    """
    kwargs: dict[str, Any] = {}
    for flag, options in (rig_opts or {}).items():
        given = {name: value for name, value in options.items() if value is not None}
        if not given:
            continue
        if not set(given) <= set(deployment_cls.get_parameter_names()):
            sys.exit(
                f"{flag} acts on a composite rig's MuJoCo (MujocoCompositeDeployment), and '{entity_name}' deploys "
                f"through {deployment_cls.__name__}, which has no such option. Drop {flag}."
            )
        kwargs.update(given)
    return kwargs


def _run_deploy(
    config,
    *,
    checkpoints: list[str],
    show_viewer: bool,
    entity_name: str,
    deploy_kp_kd_scale: float,
    gain_ramp_steps: int | None = None,
    base_policy: str = "none",
    base_command: tuple[float, float, float] = (0.0, 0.0, 0.0),
    extension_policies: list[str] = (),
    robot_name: str = "G1Dofs29",
    motion_file: str | None = None,
    estimate_lin_vel: bool = False,
    feet_links: tuple[str, ...] = (),
    velocity_command: str = "constant",
    gamepad_layout: str = "ps4",
    velocity_command_scale: tuple[float, ...] = (),
    disable_joints: tuple[str, ...] = (),
    ignore_action_terms: tuple[str, ...] = (),
    debug_record: bool = False,
    debug_record_path: str | None = None,
    debug_record_flush_steps: int | None = None,
    target_lpf: float = 1.0,
    power_protect_factor: int | None = None,
    publish_rate: float | None = None,
    mujoco: bool = False,
    menagerie: bool = False,
    init_steps: int | None = None,
    init_target: str | None = None,
    on_motion_end: str | None = None,
    stand_steps: int | None = None,
    motion_end_slot: int | None = None,
    policy_checkpoints: tuple[str, ...] = (),
    gantry: bool = False,
    gantry_opts: dict | None = None,
    mujoco_async: bool = False,
    mujoco_sim_dt: float | None = None,
    mujoco_record: str | None = None,
    mujoco_seconds: float | None = None,
    parts_options: dict | None = None,
    mujoco_props: list[str] | None = None,
    imu_opts: dict | None = None,
    rig_opts: dict | None = None,
) -> None:
    import eden as en
    from eden.extensions.deployment import DEPLOYMENT_REGISTRY, resolve_deployment_for
    from eden.extensions.deployment.base import scene_entity_options
    from eden.options.entities import CompositeEntityOptions

    # In sim-to-sim mode the robot's physics run in-process, so the Genesis env is only
    # ever a state mirror; drive the viewer from the sim backend (MuJoCo's own window)
    # instead of opening a second Genesis viewer that shows the same rollout.
    genesis_show_viewer = show_viewer and not mujoco

    # The single-robot MuJoCo backend's own flags, which a rig's MuJoCo backend does not run.
    mujoco_flags = {
        "--menagerie": menagerie,
        "--async": mujoco_async,
        "--gantry": gantry,
        "--imu_noise / --imu_bias": any(value is not None for value in (imu_opts or {}).values()),
    }
    # Before anything is built when a task says what the scene holds; a --policy's env is checked once built.
    if config is not None:
        _check_composite_deploy(
            config.scene_options, entity_name, mujoco=mujoco, parts_options=parts_options, mujoco_flags=mujoco_flags
        )
        if checkpoints:
            _check_deployable_runner(config)

    base_policy = _resolve_base_policy(base_policy, config, robot_name, entity_name)
    if not checkpoints and not extension_policies and base_policy == "none":
        sys.exit(
            "No policy to deploy: the robot is not a G1Dofs29 (so no Holosoma base policy) "
            "and neither --checkpoint nor --policy was given. Provide --checkpoint or --policy."
        )
    ext_specs, has_base = _resolve_extension_specs(base_policy, extension_policies)
    _apply_policy_checkpoints(policy_checkpoints, ext_specs)
    # Settled before anything is built: it decides which policy's spec the env comes from, and
    # refuses the arrangements no single env can serve.
    env_owner_idx = _env_owning_slot(ext_specs, config, has_base)
    env_owner_cls = ext_specs[env_owner_idx][1] if env_owner_idx is not None else None

    # Command terms the extension policies need on the env (e.g. SONIC's
    # motion reference). Resolved per slot so each slot's activation can rewind
    # its own clocks, and merged into whichever env gets built below — a
    # missing term would otherwise only surface as a crash mid-run on the
    # robot, after the operator switches to that slot.
    ext_cmd_options = {label: _extension_command_options(cls, entity_name, motion_file) for label, cls in ext_specs}
    merged_cmd_options: dict = {}
    for cmd_options in ext_cmd_options.values():
        for term_name, term_options in cmd_options.items():
            merged_cmd_options.setdefault(term_name, term_options)

    env = _build_deploy_env(config, env_owner_cls, robot_name, entity_name, genesis_show_viewer, merged_cmd_options)
    # The entities as built, before the deployment swaps the ones it drives (one per part, on a rig).
    sim_entities = dict(env.unwrapped.entities)
    # A composite placed under --entity_name deploys whole; anything else is the one env entity of that name.
    scene_options = env.unwrapped.config.scene_options
    if config is None:
        _check_composite_deploy(
            scene_options, entity_name, mujoco=mujoco, parts_options=parts_options, mujoco_flags=mujoco_flags
        )
    robot_options = getattr(scene_options, entity_name, None)
    if not isinstance(robot_options, CompositeEntityOptions):
        robot_options = scene_entity_options(scene_options, entity_name)

    # Resolve each slot's pinned robot now, while the env still holds the sim entity: this calls
    # into policy code, and the deployment constructed below swaps that entity, after which a
    # raise would land outside the rollback guarding it. Their gain maps are checked here
    # too, for the same reason.
    pinned_robots = [(label, cls.robot_options()) for label, cls in ext_specs]
    _check_pinned_gains(pinned_robots, robot_options)

    def _resolve_command_terms(label: str) -> list:
        terms = []
        for term_name in ext_cmd_options[label]:
            try:
                terms.append(env.unwrapped.command_manager.get_term(term_name))
            except (AttributeError, KeyError) as exc:
                sys.exit(f"Extension policy '{label}' needs command term '{term_name}' on the env: {exc}")
        return terms

    # Everything the velocity command decides, before a policy exists to be handed it.
    plan = _plan_velocity_command(
        env.unwrapped, entity_name, velocity_command, base_command, velocity_command_scale, gamepad_layout
    )

    # Extension policies resolve DOF/entity handles in __init__, so construct
    # them against the sim entity before the deployment swap below.
    constructed = [
        (label, *_construct_extension_policy(env.unwrapped, entity_name, cls, plan.base_command))
        for label, cls in ext_specs
    ]
    # Captured here, from the simulated robot, for the same reason the policies are constructed
    # here: after the deployment's entity swap the observation terms read a robot that has no
    # reading until connect(), and the step-contract probe has to run before either.
    probe_obs = _probe_observations(env.unwrapped) if ext_specs else None

    if (
        not plan.overrides
        and velocity_command in _DEVICE_VELOCITY_COMMANDS
        and not any(hasattr(policy, "set_command") for _, policy, _ in constructed)
    ):
        sys.exit(
            f"--velocity_command {velocity_command} has nothing to drive: this env declares no "
            f"SE(2) velocity command term (UniformVelocityCommand) for entity '{entity_name}', "
            "and no loaded policy takes a command. Deploy a velocity-tracking task "
            "(e.g. --task velocity_tracking_flat) or a command-driven --policy."
        )

    if mujoco:
        # A rig runs through the composite deployment itself, every part on one MuJoCo model of the rig.
        rig = isinstance(robot_options, CompositeEntityOptions)
        deployment_cls = DEPLOYMENT_REGISTRY.get("mujoco_composite_deployment" if rig else "mujoco_deployment")
        en.logger.info(f"[deploy] sim-to-sim via {deployment_cls.__name__} (robot binding overridden by --mujoco).")
    else:
        deployment_cls = resolve_deployment_for(robot_options)
        en.logger.info(f"[deploy] Using {deployment_cls.__name__} for {type(robot_options).__name__}")
    # Backend-specific options are only accepted by the backends that declare them
    # (configure() rejects unknown params), so include them conditionally.
    configure_kwargs = dict(
        entity_name=entity_name,
        sync=True,
        connect_timeout_s=10.0,
        deploy_kp_kd_scale=deploy_kp_kd_scale,
        # Only when given: the backend's own default (DeploymentBase.gain_ramp_steps) stays the one source of it.
        **({} if gain_ramp_steps is None else {"gain_ramp_steps": gain_ramp_steps}),
        estimate_lin_vel=estimate_lin_vel,
        # Empty means unset: the deployment then falls back to RobotOptions.feet_links_name.
        feet_links=feet_links,
        command_overrides=plan.overrides,
        disable_joints=disable_joints,
        ignore_action_terms=ignore_action_terms,
        debug_record=debug_record,
        debug_record_path=debug_record_path,
        **({} if debug_record_flush_steps is None else {"debug_record_flush_steps": debug_record_flush_steps}),
        target_lpf=target_lpf,
    )
    # The MuJoCo backend renders in its own viewer window; route --vis to it.
    if mujoco and "show_viewer" in deployment_cls.get_parameter_names():
        configure_kwargs["show_viewer"] = show_viewer
    if mujoco and "mjcf_source" in deployment_cls.get_parameter_names():
        configure_kwargs["mjcf_source"] = "menagerie" if menagerie else "eden"
    if mujoco and gantry and "gantry" in deployment_cls.get_parameter_names():
        configure_kwargs["gantry"] = True
        # Only the flags the operator actually passed; the rest keep the backend's defaults.
        for name, value in (gantry_opts or {}).items():
            if value is not None:
                configure_kwargs[name] = value
    if mujoco_props is not None:
        if "props" not in deployment_cls.get_parameter_names():
            sys.exit(
                f"--mujoco_props picks the props a composite rig's MuJoCo simulates, and '{entity_name}' is a single "
                f"robot, whose MuJoCo ({deployment_cls.__name__}) simulates the robot alone. Drop --mujoco_props."
            )
        configure_kwargs["props"] = tuple(mujoco_props)
    configure_kwargs.update(_rig_option_kwargs(rig_opts, deployment_cls, entity_name))
    if mujoco and "async_physics" in deployment_cls.get_parameter_names():
        configure_kwargs["async_physics"] = mujoco_async
    # Both MuJoCo backends declare sim_dt and record_video; policy_seconds is every simulator's.
    for name, value in (("sim_dt", mujoco_sim_dt), ("record_video", mujoco_record), ("policy_seconds", mujoco_seconds)):
        if mujoco and value is not None:
            configure_kwargs[name] = value
    if mujoco:
        # Only the flags the operator actually passed; the rest keep the backend's (exact) IMU.
        for name, value in (imu_opts or {}).items():
            if value is not None and name in deployment_cls.get_parameter_names():
                configure_kwargs[name] = value
    # A rig's parts each run their own backend, and a flag only some of them declare goes to those.
    part_backends = (
        deployment_cls.part_backends(robot_options)
        if isinstance(robot_options, CompositeEntityOptions) and not mujoco
        else {}
    )
    parts_options = {part: dict(options) for part, options in (parts_options or {}).items()}

    if power_protect_factor is not None:
        _route_backend_flag(
            "power_protect_factor",
            {"power_protect_factor": power_protect_factor},
            deployment_cls,
            part_backends,
            configure_kwargs,
            parts_options,
        )
    for name, value in (
        ("init_steps", init_steps),
        ("init_target", init_target),
        ("on_motion_end", on_motion_end),
        ("stand_steps", stand_steps),
        ("motion_end_slot", motion_end_slot),
    ):
        if value is None:
            continue
        if name in deployment_cls.get_parameter_names():
            configure_kwargs[name] = value
        else:
            en.logger.warning(f"[deploy] --{name} is not supported by {deployment_cls.__name__}; ignoring.")
    if publish_rate is not None:
        # 0 on the command line means "one packet per tick"; the hardware examples that
        # expose the same flag share this translation.
        from eden.extensions.deployment.utils.republisher import publish_rate_configure_kwargs

        _route_backend_flag(
            "publish_rate",
            publish_rate_configure_kwargs(publish_rate),
            deployment_cls,
            part_backends,
            configure_kwargs,
            parts_options,
        )
    if parts_options:
        configure_kwargs["parts_options"] = parts_options

    # Instantiating the deployment swaps the sim entity for a ``RobotStateEntity``
    # wrapper but does *not* touch the wire. Hardware contact is deferred to
    # ``dep.connect()`` below, after we've verified every policy loads.
    dep = deployment_cls(env.unwrapped, deployment_cls.configure(**configure_kwargs))
    # Before the wire: an observation fed to a policy that reads a velocity this deploy never supplies.
    _check_base_lin_vel_source(
        dep, _base_lin_vel_readers(env.unwrapped, ext_specs, entity_name, _actor_obs_groups(env, checkpoints))
    )
    # The swap has happened: the slots' controllers still hold the sim entities they resolved
    # their DOF names against.
    _repoint_swapped_slot_controllers(constructed, sim_entities, env.unwrapped.entities)

    # After construction: with a task the reader follows the term the deployment just
    # substituted, so the stick drives whichever slot is active rather than only the one
    # whose command lives on the env.
    operator = _OperatorCommandReader.for_deploy(
        env.unwrapped, entity_name, velocity_command, plan.term_name, plan.ranges, gamepad_layout
    )
    ext_slots = [
        _extension_policy_slot(
            dep,
            label,
            policy,
            controllers,
            plan.base_command,
            _resolve_command_terms(label),
            operator=operator,
            owns_env=config is None and slot_idx == env_owner_idx,
            probe_obs=probe_obs,
            # With --task the task's robot sets every slot's gains (_warn_on_task_robot_gains).
            robot=pinned_robot if config is None else None,
        )
        for slot_idx, ((label, policy, controllers), (_, pinned_robot)) in enumerate(
            zip(constructed, pinned_robots, strict=True)
        )
    ]

    ckpt_slots = _load_checkpoint_slots(env, dep, checkpoints)
    slots = _order_policy_slots(ext_slots, ckpt_slots, has_base)
    if config is not None:
        _warn_on_task_robot_gains(dep, pinned_robots)
    _warn_on_unswitchable_slot_gains(dep, slots)

    _deploy_control_loop(env, dep, slots, entity_name, genesis_show_viewer, operator=operator)


def _route_backend_flag(
    flag: str,
    kwargs: dict,
    deployment_cls,
    part_backends: dict,
    configure_kwargs: dict,
    parts_options: dict,
) -> None:
    """Hand a backend-specific flag's ``configure`` kwargs to the backend, or to every rig part whose backend
    declares them.

    ``eden deploy`` flags like ``--publish_rate`` exist on some backends only. On a composite rig
    (``part_backends``, from ``CompositeDeployment.part_backends``) they go into ``parts_options``
    of every part that declares them -- the Oli's ``publish_rate`` on an ``OliWuji`` -- under an
    explicit ``--part_option`` for the same option, which is the more specific. A flag nothing
    declares is dropped with a warning naming every backend asked.
    """
    import eden as en

    if set(kwargs) <= set(deployment_cls.get_parameter_names()):
        configure_kwargs.update(kwargs)
        return
    takers = [part for part, cls in part_backends.items() if set(kwargs) <= set(cls.get_parameter_names())]
    for part in takers:
        for name, value in kwargs.items():
            parts_options.setdefault(part, {}).setdefault(name, value)
    if takers:
        en.logger.info(f"[deploy] --{flag} applies to part(s) {takers}.")
    else:
        backends = [deployment_cls.__name__] + [cls.__name__ for cls in part_backends.values()]
        en.logger.warning(f"[deploy] --{flag} is not supported by {' / '.join(backends)}; ignoring.")


def _check_composite_deploy(
    scene_options, entity_name: str, *, mujoco: bool, parts_options: dict | None, mujoco_flags: dict | None = None
) -> None:
    """Exit unless what sits under ``entity_name`` can take ``--part_option`` and the MuJoCo flags given.

    A composite rig deploys whole: on hardware each part through its own backend, under ``--mujoco``
    each part on the rig's one MuJoCo model (``MujocoCompositeDeployment``). That backend does not
    run ``mujoco_flags`` (``{"--menagerie": True, ...}``, the single-robot MuJoCo's options), and a
    flag a backend does not declare would be dropped in silence -- so a set one is refused here.
    ``--part_option`` configures a rig's parts, so on a single robot it would configure nothing.
    """
    from eden.options.entities import CompositeEntityOptions

    deployed = getattr(scene_options, entity_name, None)
    if isinstance(deployed, CompositeEntityOptions):
        if mujoco and (unsupported := [flag for flag, given in (mujoco_flags or {}).items() if given]):
            sys.exit(
                f"{' / '.join(unsupported)} {'is' if len(unsupported) == 1 else 'are'} not supported by --mujoco on "
                f"'{entity_name}', a composite rig ({type(deployed).__name__}: {', '.join(deployed.entity_names)}), which "
                "runs every part on one MuJoCo model of the rig. Drop it, or pass --entity_name <part> to run one part "
                "alone in the single-robot MuJoCo backend."
            )
    elif parts_options:
        sys.exit(
            f"--part_option configures the parts of a composite rig, and '{entity_name}' is a single robot. Drop it."
        )


def _draw_motion_ghost(env, motion_manager, entity_name: str, prev_node):
    """Draw the current motion-reference tracked-link positions as a ghost.

    Clears only the previous ghost node (not other debug draws) and returns the new node handle. Returns
    ``None`` if the reference has no link positions.
    """
    if prev_node is not None:
        env.clear_debug_object(prev_node)
    ref = motion_manager.reference_frame.get(entity_name)
    links_pos_w = getattr(ref, "links_pos_w", None) if ref is not None else None
    if links_pos_w is None:
        return None
    # links_pos_w is (num_envs, num_frames, num_links, 3); take env 0, current frame.
    poss = links_pos_w[0, 0].detach().cpu().numpy()
    return env.draw_debug_spheres(poss, radius=_GHOST_SPHERE_RADIUS, color=_GHOST_COLOR)


def _run_inference_policy(
    config,
    policy_name: str,
    *,
    show_viewer: bool,
    record: bool,
    max_steps: int,
    entity_name: str,
    robot_name: str,
    motion_file: str | None,
    clip_name: str | None = None,
    base_command: tuple[float, float, float],
    loop_fn=None,
) -> None:
    """Roll out a frozen extension policy in sim — no checkpoint, no hardware.

    Mirrors the policy examples (``examples/policies/*/main.py``): the policy's
    ``step(obs)`` output feeds the env's action manager, so unlike deploy there
    is no absolute-target contract — scaled/offset controllers work too. With
    ``config`` (a task) the task's env hosts the policy and the policy's
    command terms are merged in; without one the env is built from the
    policy's own spec methods on an empty ``robot_name`` scene.

    ``loop_fn`` (``eden inference --async --policy``) swaps the synchronous rollout
    for the wall-clock-paced :func:`_run_sim2sim_loop` — policy inference moves to
    a background thread and the last action is repeated when it lags, the
    deployment-style dry run of ``eden deploy --policy``. As in deploy the policy
    is reset once at boot and not rewound on episode boundaries.
    """
    import torch
    from tqdm import tqdm

    import eden as en
    from eden.options.camera import CameraOptions, CamerasOptions

    cls = _resolve_extension_policy(policy_name)
    cmd_options = _extension_command_options(cls, entity_name, motion_file, clip_name)
    task_terms = _task_terms_driving_policy_command(config, cls, cmd_options)
    if task_terms and any(base_command):
        sys.exit(
            f"--base_command {tuple(base_command)} cannot reach {cls.__name__} under --task "
            f"{type(config).__name__}: the policy observes its command through the task's own command "
            f"term(s) {task_terms}, which generate it. Drop --base_command to run on the task's commands, or "
            "drop --task to hold a command through the policy's own command term."
        )

    if config is not None:
        _merge_command_terms(config, cmd_options, context="inference")
        # Saved after the command-term merge so the snapshot rebuilds the env
        # that actually ran.
        config.save_as_json()

        from eden.envs.base import env_class_for_config

        env = env_class_for_config(config).from_config(config, show_viewer=show_viewer, eval_mode=True)
        env.build()
    else:
        cameras_options = (
            CamerasOptions(
                rec=CameraOptions(cam_pos=(1.5, 1.0, 2.0), cam_lookat=(0.0, 0.0, 1.0), follow_entity_name=entity_name)
            )
            if record
            else None
        )
        env = _build_extension_env(cls, robot_name, entity_name, show_viewer, cmd_options, cameras_options)

    # Forward --clip_name to the policy instance using the same signal as the command term
    # (_command_options_accepts) so the two resolve the same clip; the hasattr guard keeps
    # configure() from getting an unknown kwarg.
    forward_clip = clip_name is not None and _command_options_accepts(cls, "clip_name") and hasattr(cls, "clip_name")
    policy_overrides = {"clip_name": clip_name} if forward_clip else {}
    policy = cls(env=env, options=cls.configure(**policy_overrides))
    if task_terms:
        en.logger.info(
            f"[inference] {cls.__name__} observes the task's command term(s) {task_terms}; --base_command is not used."
        )
    else:
        _set_policy_command(policy, base_command)

    if loop_fn is not None:
        # Deployment-style async rollout (eden inference --async --policy). The shared loop
        # owns env.reset() / recording / pacing; we only need the policy's boot
        # reset (zeroes its history buffers) before handing off policy.step.
        env.reset()
        policy.reset()
        loop_fn(env, policy.step, record=record, max_steps=max_steps, max_episodes=None)
        return

    rec = env.cameras["rec"] if record else None
    if rec is not None:
        rec.start_recording()

    obs, _ = env.reset()
    policy.reset()
    with torch.no_grad():
        for _ in tqdm(range(max_steps), desc="inference"):
            obs, _, terminated, timeouts, _ = env.step(policy.step(obs))
            # Task envs terminate/auto-reset mid-run; rewind the policy's
            # per-env state (last-action buffers, gait phase, ...) to match.
            dones = terminated | timeouts
            done_idx = dones.nonzero(as_tuple=False).flatten()
            if done_idx.numel():
                policy.reset(done_idx)
            if rec is not None:
                rec.render_rgb()

    if rec is not None:
        rec.stop_recording()


def preprocess_task(extra_args: list[str]) -> None:
    """``eden preprocess --task <name>`` — build the task's dataset cache.

    Walks the task's ``motion_options`` tree, instantiates every configured
    ``DatasetBase`` once, and triggers the ``prepare()`` + ``save_cached()``
    chain. Use this to pre-warm the v2 mmap cache before launching training,
    especially when the modifier stack includes expensive offline steps like
    ``SmoothStartTransition`` (per-clip IK).

    For libraries large enough that single-process IK preprocessing is too
    slow, pass ``--num_batches N --batch_idx K`` and launch one process per
    GPU (``CUDA_VISIBLE_DEVICES=0 ... batch_idx=0 ... &``); each process
    handles a contiguous slice of the clip list.  Caches land at
    ``processed_data_path``-keyed paths (one per batch when ``num_batches > 1``).
    """
    from eden.tasks.parser import get_task_argparser, get_task_config_from_args

    parser = get_task_argparser(
        description="Build the v2 dataset cache for a registered task.",
        argv=extra_args,
    )
    parser.add_argument(
        "--num_batches",
        type=int,
        default=1,
        help="Split the clip list into this many contiguous batches. >1 enables "
        "per-process sharding; launch one process per GPU.",
    )
    parser.add_argument(
        "--batch_idx",
        type=int,
        default=0,
        help="Which batch (0-based, < --num_batches) this process owns. Default: 0.",
    )
    args = parser.parse_args(extra_args)
    _validate_task_arg(args, parser)
    if args.num_batches < 1:
        parser.error("--num_batches must be >= 1.")
    if args.batch_idx < 0 or args.batch_idx >= args.num_batches:
        parser.error(f"--batch_idx must be in [0, {args.num_batches}).")

    import eden as en
    import genesis as gs

    # GPU by default: preprocessing runs Genesis FK / IK over one env per clip,
    # which is intractable on CPU at multi-thousand-clip scale (the docstring's
    # per-GPU sharding recipe assumes a CUDA backend). --cpu opts out.
    en.init(backend=gs.cpu if args.cpu else gs.gpu, performance_mode=False)

    config = get_task_config_from_args(args, run_name=args.run_name)
    motion_options = getattr(config, "motion_options", None)
    if motion_options is None:
        parser.error(f"Task '{args.task}' has no motion_options — nothing to preprocess.")
        return  # for type-checkers; parser.error exits

    # Walk all configured MotionTerm entries; each carries a dataset_options.
    dataset_options_list = []
    extras = getattr(motion_options, "model_extra", None) or {}
    for term_options in extras.values():
        dataset_options = getattr(term_options, "dataset_options", None)
        if dataset_options is not None:
            dataset_options_list.append(dataset_options)

    if not dataset_options_list:
        en.logger.warning(
            f"Task '{args.task}' has motion_options but no MotionTerm declares "
            f"a dataset_options — nothing to preprocess."
        )
        return

    import torch

    # Importing eden.datasets.base first executes the eden.datasets package
    # __init__, which imports every dataset module and populates DATASET_REGISTRY.
    from eden.datasets.base import DATASET_REGISTRY

    for dataset_options in dataset_options_list:
        dataset_cls = DATASET_REGISTRY.get(dataset_options.name)
        en.logger.info(f"Preprocessing dataset '{dataset_options.name}'...")
        ds = dataset_cls(options=dataset_options, device=torch.device("cpu"))

        if args.num_batches > 1:
            # Split the loaded clip list into N contiguous slices and keep
            # only the requested one. ``select_trajectories`` rebuilds the
            # per-key BatchMotionSequence in place.
            n_clips = len(ds)
            base = n_clips // args.num_batches
            rem = n_clips % args.num_batches
            start = args.batch_idx * base + min(args.batch_idx, rem)
            end = start + base + (1 if args.batch_idx < rem else 0)
            keep_idx = list(range(start, end))
            if not keep_idx:
                en.logger.warning(
                    f"Batch {args.batch_idx}/{args.num_batches} is empty (library has {n_clips} clips); skipping save."
                )
                continue
            en.logger.info(f"Batch {args.batch_idx}/{args.num_batches}: clips [{start}, {end}) out of {n_clips}")
            ds.select_trajectories(keep_idx)
            # Save to a batch-suffixed path so multiple processes don't race
            # on the same file. Read off the instance, not the options: a dataset
            # may derive its own default cache path (BvhCsvG1Dataset keys one by
            # the settings that shape the prepared motion), and that derived path
            # is the one ``_initialize`` will look for on next load.
            if ds.processed_data_path:
                suffix = f".batch_{args.batch_idx}_of_{args.num_batches}"
                root, ext = os.path.splitext(ds.processed_data_path)
                batch_path = f"{root}{suffix}{ext or '.pt'}"
                ds.save_cached(batch_path)
                en.logger.info(f"Saved batch cache to {batch_path}")
        elif ds.processed_data_path:
            # Single-process path saves to the canonical location (which is what
            # the dataset's ``_initialize`` will look for on next load).
            ds.save_cached(ds.processed_data_path)


def _is_rl_games(config) -> bool:
    """Whether ``config.runner_options`` is the rl_games variant."""
    from eden.options.learning.rl_games import RlGamesPpoRunnerOptions

    return isinstance(config.runner_options, RlGamesPpoRunnerOptions)


def _write_run_manifest(config, args, resumed_from: str | None = None) -> None:
    """Write the run's ``run.json`` reproducibility manifest into ``eden.log_dir``.

    Best-effort: run.json is provenance, not a training input, so an I/O failure warns
    loudly but must not abort a training run (unlike the config.json save, whose failure
    is worth stopping for). The unlocked config.json is the recoverable source of truth.

    Records the RL backend the run is about to train with, so the manifest carries the
    version of the library that writes the checkpoints — the piece a cross-machine replay
    needs and that eden/genesis provenance alone cannot supply.
    """
    import eden as en

    from eden.utils.run_manifest import build_run_manifest, write_run_manifest

    runner_options = getattr(config, "runner_options", None)
    try:
        # Read the lazily-created log dir once, and inside the guard: the path recorded in
        # the manifest and the path written to must be the same string, and resolving it
        # both creates the directory and can raise — which must not abort the run either.
        log_dir = en.log_dir
        manifest = build_run_manifest(
            log_dir=log_dir,
            run_id=en.get_run_id(),
            timestamp=en.get_run_timestamp(),
            task=args.task,
            seed=getattr(runner_options, "seed", None),
            world_size=en.distributed.get_world_size(),
            run_name=getattr(runner_options, "run_name", "") or "",
            argv=list(sys.argv),
            resumed_from=resumed_from,
            rl_backend="rl_games" if _is_rl_games(config) else "rsl_rl",
        )
        write_run_manifest(log_dir, manifest)
    except Exception as exc:
        en.logger.warning(f"Could not write run manifest (run.json): {exc}")


def _check_rsl_rl_runner(config, checkpoint: str | None) -> None:
    """Resolve the task's runner and algorithm before anything is built or written, and exit if they cannot start.

    Everything here would otherwise surface only after the scene build (minutes at thousands of envs):

    - the runner class — a typo, a missing backend, or one that cannot train under this distributed launch
      (:func:`~eden.envs.wrappers.rsl_rl_env.rsl_rl_runner_class`), or that cannot start without a checkpoint (a
      distillation needs its teacher) — exits before a run dir is written;
    - for a stock runner, every model/algorithm ``class_name`` given as an import path, and (rsl-rl-lib >= 5, for a
      subclass of rsl_rl's ``PPO`` / ``Distillation``) whether the algorithm class accepts the fields its options
      carry (:func:`_check_rsl_rl_algorithm`);
    - the algorithm options' own ``preflight()``, when they define one (AMP-PPO loads its expert clips there).

    FlashSAC's ``OffPolicyRunner`` is not checked here: :func:`_rsl_rl_env_and_runner` builds it through its own
    factory, never through ``rsl_rl_runner_class``, which would refuse its bare non-stock name.
    """
    from eden.envs.wrappers.rsl_rl_env import STOCK_RUNNER_NAMES, import_class_path, rsl_rl_runner_class
    from eden.learning.wrappers.rsl_rl.flashsac.options import RslRlFlashSacRunnerOptions

    options = config.runner_options
    if isinstance(options, RslRlFlashSacRunnerOptions):
        return
    class_name = getattr(options, "class_name", None) or "OnPolicyRunner"
    runner_cls = rsl_rl_runner_class(class_name)
    if checkpoint is None and getattr(runner_cls, "checkpoint_required", None):
        sys.exit(runner_cls.checkpoint_required)
    if class_name in STOCK_RUNNER_NAMES:
        # A custom runner owns how its options map onto its classes; a stock one hands them to rsl_rl as-is.
        for field in ("actor", "critic", "student", "teacher"):
            model_name = getattr(getattr(options, field, None), "class_name", None)
            field_name = f"runner_options.{field}.class_name"
            if model_name and ":" in model_name and import_class_path(model_name, field_name)[1] is None:
                raise ValueError(f"{field_name}={model_name!r}: no such attribute.")
        if getattr(options, "algorithm", None) is not None:
            _check_rsl_rl_algorithm(options)
    preflight = getattr(getattr(options, "algorithm", None), "preflight", None)
    if preflight is not None:
        preflight()


def _check_rsl_rl_algorithm(options) -> None:
    """Resolve a stock runner's ``algorithm.class_name`` and check it can take the fields its options carry.

    The field check models rsl_rl's ``PPO.construct_algorithm`` / ``Distillation.construct_algorithm``, which pass
    the translated algorithm dict to the class as keywords, so it covers their subclasses only. A class outside that
    hierarchy (HIMPPO) builds itself in its own ``construct_algorithm`` from the fields it reads.
    """
    import eden as en
    from eden.envs.wrappers.rsl_rl_env import import_class_path
    from eden.options.learning.rsl_rl.translate import detect_rsl_rl_major_version, translate_runner_dict

    algorithm = options.algorithm
    name = algorithm.class_name
    if ":" in name:
        alg_cls = import_class_path(name, "runner_options.algorithm.class_name")[1]
        if alg_cls is None:
            raise ValueError(f"runner_options.algorithm.class_name={name!r}: no such attribute.")
    elif detect_rsl_rl_major_version() >= 5:
        from rsl_rl.utils import resolve_callable

        alg_cls = resolve_callable(name)
    else:
        return  # rsl-rl-lib < 5 resolves its bare names itself, from a differently shaped dict
    if en.distributed.is_distributed() and getattr(alg_cls, "distributed_unsupported", None):
        sys.exit(alg_cls.distributed_unsupported)
    if detect_rsl_rl_major_version() < 5:
        return
    from rsl_rl.algorithms import PPO, Distillation

    if not (isinstance(alg_cls, type) and issubclass(alg_cls, (PPO, Distillation))):
        # Its own ``construct_algorithm`` (HIMPPO's) picks its fields out of the runner dict and checks them there;
        # only PPO's and Distillation's pass the whole algorithm dict to ``__init__`` as keywords.
        return
    accepted, required = _accepted_keywords(alg_cls)
    # What rsl_rl's ``PPO.construct_algorithm`` passes as keywords: the translated dict, less what it consumes itself.
    passed = set(translate_runner_dict(options.dict(), algorithm)["algorithm"]) - {"class_name", "share_cnn_encoders"}
    unknown = sorted(passed - accepted) if accepted is not None else []
    missing = sorted(required - passed - {"device", "multi_gpu_cfg"})
    if missing:
        raise ValueError(
            f"runner_options.algorithm.class_name={name!r} names {alg_cls.__module__}.{alg_cls.__qualname__}, which "
            f"requires {missing}, but {type(algorithm).__name__} has no such field. Give the runner the options class "
            f"made for {alg_cls.__name__} (its learning package exports one) rather than changing class_name alone."
        )
    if unknown:
        options_cls = type(algorithm)
        default = options_cls.model_fields["class_name"].default
        raise ValueError(
            f"runner_options.algorithm.class_name={name!r} names {alg_cls.__module__}.{alg_cls.__qualname__}, which "
            f"does not accept {unknown}: those are fields of {options_cls.__name__}, whose class_name is {default!r}. "
            f"To train {alg_cls.__name__}, give the runner the options class made for it (plain PPO: "
            f"RslRlPpoAlgorithmOptions) rather than changing class_name alone; to keep {options_cls.__name__}, leave "
            f"class_name={default!r}."
        )


def _accepted_keywords(cls) -> tuple[set[str] | None, set[str]]:
    """``(keyword names cls(...) accepts or None if any, keyword-only names it requires)``, through ``**kwargs``."""
    import inspect

    names: set[str] = set()
    required: set[str] = set()
    for klass in cls.__mro__:
        init = klass.__dict__.get("__init__")
        if init is None:
            continue
        params = inspect.signature(init).parameters.values()
        names |= {p.name for p in params if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)}
        required |= {p.name for p in params if p.kind is p.KEYWORD_ONLY and p.default is p.empty}
        if not any(p.kind is p.VAR_KEYWORD for p in params):
            return names, required
    return None, required


def _rsl_rl_env_and_runner(config, checkpoint: str | None, *, inference: bool = False, **env_kwargs):
    """The rsl_rl env wrapper and runner ``config.runner_options`` names: rsl_rl_flashsac's ``OffPolicyRunner`` for
    :class:`RslRlFlashSacRunnerOptions` (on a wrapper that also hands it the action bounds and the pre-reset
    observation of timed-out envs), else rsl_rl's on-policy / distillation runner.

    ``inference`` builds a runner only to run the policy (no FlashSAC replay buffer); ``env_kwargs`` go to
    ``from_config``.
    """
    from eden.envs.wrappers.rsl_rl_env import RslRlVecEnvWrapper, make_rsl_rl_runner
    from eden.learning.wrappers.rsl_rl.flashsac.options import RslRlFlashSacRunnerOptions

    if isinstance(config.runner_options, RslRlFlashSacRunnerOptions):
        from eden.learning.wrappers.rsl_rl.flashsac.runner import FlashSacVecEnvWrapper, make_flashsac_runner

        env = FlashSacVecEnvWrapper.from_config(config, **env_kwargs)
        return env, make_flashsac_runner(env, checkpoint=checkpoint, inference=inference)
    env = RslRlVecEnvWrapper.from_config(config, **env_kwargs)
    return env, make_rsl_rl_runner(env, checkpoint=checkpoint, for_inference=inference)


def _run_train_rsl_rl(config, checkpoint: str | None = None) -> None:
    # No seeding here: `RslRlVecEnvWrapper.from_config` applies runner_options.seed before
    # it builds the env, so this path and the programmatic examples share one owner.
    env, runner = _rsl_rl_env_and_runner(config, checkpoint)
    # The wrapper's options, not the config's: FlashSAC sizes max_iterations for the env count it was built with.
    runner.learn(num_learning_iterations=env.runner_options.max_iterations, init_at_random_ep_len=True)


def _run_train_rl_games(config, checkpoint: str | None = None) -> None:
    # Use ``runner.run({"train": True, ...})`` rather than constructing the
    # agent directly: run_train() also handles _restore (checkpoint resume),
    # _override_sigma, optional torch.compile (params.config.torch_compile),
    # and profiling (args.profile). Bypassing it would silently drop all four.
    from rl_games.torch_runner import Runner

    from eden.envs.wrappers.rl_games_env import RlGamesVecEnvWrapper

    env = RlGamesVecEnvWrapper.from_config(config)
    runner = Runner()
    runner.load(env.runner_dict)
    env.configure_runner(runner)
    runner.run({"train": True, "checkpoint": checkpoint or ""})


def _no_runner_hint(config) -> str:
    """Where to go for a task without ``runner_options``: its own scripts, found by the task package's name."""
    package = type(config).__module__.removeprefix("eden.tasks.").split(".")[0]
    return (
        "Such a task is trained by a standalone script, or only evaluated (e.g. by a client policy); "
        f"look for its scripts with `eden examples list --search {package}`."
    )


def _check_checkpoint_framework(config, checkpoint: str) -> str:
    """The framework ``config`` trains with (``"rsl_rl"`` / ``"rl_games"``); exits if ``checkpoint`` was saved by
    the other.
    """
    configured = "rl_games" if _is_rl_games(config) else "rsl_rl"
    detected = _detect_checkpoint_framework(checkpoint)
    if detected is not None and detected != configured:
        sys.exit(
            f"checkpoint at {checkpoint!r} looks like a {detected} checkpoint, "
            f"but the task config selected {configured}. Re-run with the task mod "
            f"that selects {detected} (e.g. '--runner {detected}' for tasks that "
            f"expose it), or supply a {configured} checkpoint."
        )
    return configured


def _detect_checkpoint_framework(path: str) -> str | None:
    """Guess which RL framework saved a checkpoint, or ``None`` if unclear.

    rsl_rl saves ``actor_state_dict`` / ``critic_state_dict`` (current
    format) or a single combined ``model_state_dict`` with ``actor.*`` /
    ``critic.*`` keys (legacy format, auto-converted inside
    ``make_rsl_rl_runner``). rl_games saves a ``model`` state_dict
    alongside ``epoch`` / ``optimizer``. Used to fail fast on framework
    mismatches before the runner blows up deep inside ``load`` with an
    opaque KeyError.

    Detection needs key names only, so it reads the checkpoint's index
    (:func:`eden.utils.torch.read_checkpoint_index`) rather than
    deserializing the weights it is about to hand to the runner's own load.
    """
    from eden.utils.torch import read_checkpoint_index

    try:
        state = read_checkpoint_index(path)
    except Exception:
        return None
    if state is None:
        return None
    if "actor_state_dict" in state:
        return "rsl_rl"
    if "model_state_dict" in state and isinstance(state["model_state_dict"], dict):
        keys = state["model_state_dict"].keys()
        if any(k.startswith("actor.") or k.startswith("critic.") for k in keys):
            return "rsl_rl"
    if "model" in state and ("epoch" in state or "optimizer" in state):
        return "rl_games"
    return None


def _run_eval_rsl_rl(
    config,
    checkpoint: str,
    show_viewer: bool = False,
    record: bool = False,
    max_steps: int | None = None,
    max_episodes: int | None = None,
    loop_fn=None,
) -> None:
    env, runner = _rsl_rl_env_and_runner(config, checkpoint, inference=True, eval_mode=True, show_viewer=show_viewer)
    policy = runner.get_inference_policy(device=env.device)
    loop_fn = loop_fn or _run_eval_loop
    loop_fn(env, policy, record=record, max_steps=max_steps, max_episodes=max_episodes)


def _run_eval_rl_games(
    config,
    checkpoint: str,
    show_viewer: bool = False,
    record: bool = False,
    max_steps: int | None = None,
    max_episodes: int | None = None,
    loop_fn=None,
) -> None:
    from rl_games.torch_runner import Runner

    from eden.envs.wrappers.rl_games_env import RlGamesVecEnvWrapper

    env = RlGamesVecEnvWrapper.from_config(config, eval_mode=True, show_viewer=show_viewer)
    runner = Runner()
    runner.load(env.runner_dict)
    env.configure_runner(runner)
    player = runner.create_player()
    # Eden hands rl_games batched, on-device tensor obs; tell the player so it
    # skips the unsqueeze/squeeze and numpy-cast paths in get_action/env_step.
    player.has_batch_dimension = True
    player.is_tensor_obses = True
    player.restore(checkpoint)
    player.reset()

    def policy_fn(obs):
        actor_obs = obs["obs"] if isinstance(obs, dict) else obs
        return player.get_action(actor_obs, is_deterministic=True)

    loop_fn = loop_fn or _run_eval_loop
    loop_fn(env, policy_fn, record=record, max_steps=max_steps, max_episodes=max_episodes)


class _EpisodeMetrics:
    """Per-metric mean over finished episodes, read per env from ``metric_manager.last_episode()``.

    With an episode cap (``--episodes N``) every env counts its first ``ceil(N / num_envs)`` episodes and the rollout
    runs until each env has finished them. Stopping at ``N`` resets over all envs instead would leave out the episodes
    still running then, the longer ones, so a metric that a long episode scores differently on (a success rate when
    failures end episodes early) would be biased. Without a cap every finished episode counts. Episodes Genesis
    flagged as corrupt count toward the cap but not toward the means.

    :meth:`update` takes the env's own ``reset_buf``, the ``(num_envs,)`` mask ``metric_manager.reset()`` wrote
    ``last_episode()`` through, not a wrapper's ``dones``: a raw ``RLEnvBase`` returns its terminations and time-outs
    apart, and rl_games' multi-agent wrapper repeats ``dones`` once per agent.
    """

    def __init__(self, num_envs: int, episodes_per_env: int | None = None) -> None:
        import torch

        self.episodes_per_env = episodes_per_env
        self.finished_per_env = torch.zeros(num_envs, dtype=torch.long)
        self.sums: dict[str, float] = {}
        self.n_episodes = 0
        self.n_averaged = 0

    def update(self, reset_buf, metric_manager) -> None:
        """Count the episodes ``reset_buf`` ended on this step, up to each env's cap."""
        import torch

        dones = reset_buf.bool().cpu()
        if not bool(dones.any()):
            return
        counted = dones.clone()
        if self.episodes_per_env is not None:
            counted &= self.finished_per_env < self.episodes_per_env
        self.finished_per_env += dones.long()
        self.n_episodes += int(counted.sum())
        if metric_manager is None:
            return
        values, healthy = metric_manager.last_episode()
        if not values:
            return
        averaged = counted.to(healthy.device) & healthy
        # One device-to-host copy for every metric and the count, rather than one per metric.
        columns = list(values.values())
        rows = torch.stack([*columns, torch.ones_like(columns[0])], dim=1)
        *sums, n = rows[averaged].sum(dim=0).tolist()
        for key, value in zip(values, sums):
            self.sums[key] = self.sums.get(key, 0.0) + value
        self.n_averaged += int(n)

    @property
    def done(self) -> bool:
        """Every env has finished its cap of episodes (never, without one)."""
        return self.episodes_per_env is not None and bool((self.finished_per_env >= self.episodes_per_env).all())

    def report(self) -> None:
        import eden as en

        if self.episodes_per_env is not None and not self.done:
            short = int((self.finished_per_env < self.episodes_per_env).sum())
            en.logger.warning(
                f"[inference] The step budget ran out before every env finished its {self.episodes_per_env} "
                f"episode(s): {short} of {len(self.finished_per_env)} env(s) fell short, so the means below leave "
                "out their longest episodes and lean toward the short ones. Raise --steps, or lower --episodes."
            )
        if not self.sums:
            return
        per_env = f", the first {self.episodes_per_env} of each env" if self.episodes_per_env is not None else ""
        left_out = self.n_episodes - self.n_averaged
        unhealthy = f"; {left_out} more left out, Genesis flagged their env corrupt" if left_out else ""
        print(f"\n=== Inference metrics (averaged over {self.n_averaged} episode(s){per_env}{unhealthy}) ===")
        for key in sorted(self.sums):
            avg = self.sums[key] / max(self.n_averaged, 1)
            print(f"  {key:<32s} {avg: .4f}")


def _episodes_per_env(env, max_episodes: int | None) -> int | None:
    """``--episodes`` as a per-env cap: ``ceil(max_episodes / num_envs)``."""
    import math

    return None if max_episodes is None else max(1, math.ceil(max_episodes / max(1, int(env.num_envs))))


def _resolve_step_budget(env, max_steps: int | None, max_episodes: int | None) -> int:
    """Step budget: explicit --steps wins; else enough for every env to finish its ``--episodes`` cap
    (:func:`_episodes_per_env`) at ``max_episode_length`` steps each, an upper bound (the loop stops once
    they have); else one episode worth of steps.
    """
    episode_len = max(1, int(env.unwrapped.max_episode_length))
    if max_steps is not None:
        return max_steps
    if max_episodes is not None:
        return _episodes_per_env(env, max_episodes) * episode_len
    return episode_len


def _run_eval_loop(
    env,
    policy_fn,
    *,
    record: bool = False,
    max_steps: int | None = None,
    max_episodes: int | None = None,
) -> None:
    """Backend-agnostic inference loop shared by the rsl_rl and rl_games paths."""
    import torch
    from tqdm import tqdm

    rec = env.unwrapped.cameras["rec"] if record else None
    if rec is not None:
        rec.start_recording()

    metrics = _EpisodeMetrics(env.num_envs, _episodes_per_env(env, max_episodes))
    step_budget = _resolve_step_budget(env, max_steps, max_episodes)

    # rsl_rl env.reset() returns (obs, extras); rl_games returns just obs.
    reset_out = env.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    with torch.no_grad():
        for _ in tqdm(range(step_budget), desc="inference"):
            obs = env.step(policy_fn(obs))[0]
            metrics.update(env.unwrapped.reset_buf, env.unwrapped.metric_manager)
            if metrics.done:
                break
            if rec is not None:
                rec.render_rgb()

    if rec is not None:
        rec.stop_recording()

    metrics.report()


def _clone_obs(obs):
    """Copy an observation (tensor, or dict of tensors) before handing it to
    the inference thread, so the policy never reads a buffer the env is
    rewriting in place on the next control tick.
    """
    import torch

    if isinstance(obs, dict):
        return {k: _clone_obs(v) for k, v in obs.items()}
    if isinstance(obs, torch.Tensor):
        return obs.clone()
    return obs


def _run_sim2sim_loop(
    env,
    policy_fn,
    *,
    record: bool = False,
    max_steps: int | None = None,
    max_episodes: int | None = None,
) -> None:
    """Async real-time eval loop mimicking on-robot deployment (``eden inference --async``).

    Decouples the policy from the sim the way a deployed controller is
    decoupled from the hardware:

    - Policy inference runs in a background thread. It always picks up the
      *freshest* observation (intermediate ones are dropped, never queued)
      and publishes the action when it finishes.
    - The control loop steps the env with the latest completed action and
      paces itself to wall-clock ``env.dt`` by sleeping off whatever time
      the physics step left over. When inference is slower than the control
      period the previous action is repeated — exactly as a real robot holds
      the last received target.

    Steps where the action was reused (stale) and steps whose compute alone
    exceeded the control period (overruns: real-time pacing was not met) are
    counted and reported at the end alongside the achieved real-time factor.
    """
    import threading
    import time

    import torch
    from tqdm import tqdm

    import eden as en

    control_dt = float(env.unwrapped.dt)
    en.logger.info(
        f"[sim2sim] Pacing sim to wall clock at dt={control_dt * 1e3:.1f} ms "
        f"({1.0 / control_dt:.1f} Hz) with policy inference in an async thread."
    )

    rec = env.unwrapped.cameras["rec"] if record else None
    if rec is not None:
        rec.start_recording()

    metrics = _EpisodeMetrics(env.num_envs, _episodes_per_env(env, max_episodes))
    step_budget = _resolve_step_budget(env, max_steps, max_episodes)

    # Shared state between the control loop (this thread) and the inference
    # worker. ``version`` counts published actions so the control loop can
    # tell a fresh action from a stale one.
    lock = threading.Lock()
    obs_ready = threading.Event()
    stop = threading.Event()
    shared = {"obs": None, "action": None, "version": 0, "exc": None}

    def _infer_worker() -> None:
        try:
            while not stop.is_set():
                if not obs_ready.wait(timeout=0.05):
                    continue
                with lock:
                    obs = shared["obs"]
                    obs_ready.clear()
                with torch.no_grad():
                    action = policy_fn(obs)
                with lock:
                    shared["action"] = action
                    shared["version"] += 1
        except Exception as exc:  # noqa: BLE001 — surfaced in the control loop
            shared["exc"] = exc
            stop.set()

    # rsl_rl env.reset() returns (obs, extras); rl_games returns just obs.
    reset_out = env.reset()
    obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    with lock:
        shared["obs"] = _clone_obs(obs)
        obs_ready.set()

    worker = threading.Thread(target=_infer_worker, name="sim2sim-inference", daemon=True)
    worker.start()

    n_steps = 0
    n_overruns = 0
    n_stale = 0
    last_version = 0
    busy_time = 0.0
    try:
        # Wait for the first action before engaging the control loop — a
        # deployment doesn't start stepping before the policy produced a
        # target either.
        while shared["version"] == 0:
            if shared["exc"] is not None:
                raise shared["exc"]
            time.sleep(1e-3)

        wall_start = time.perf_counter()
        for _ in tqdm(range(step_budget), desc="sim2sim"):
            t0 = time.perf_counter()
            if shared["exc"] is not None:
                raise shared["exc"]
            with lock:
                action = shared["action"]
                version = shared["version"]
            if version == last_version:
                n_stale += 1
            last_version = version

            obs = env.step(action)[0]
            with lock:
                shared["obs"] = _clone_obs(obs)
                obs_ready.set()
            n_steps += 1

            metrics.update(env.unwrapped.reset_buf, env.unwrapped.metric_manager)
            if metrics.done:
                break
            if rec is not None:
                rec.render_rgb()

            elapsed = time.perf_counter() - t0
            busy_time += elapsed
            if elapsed > control_dt:
                n_overruns += 1
            else:
                time.sleep(control_dt - elapsed)
        wall_time = time.perf_counter() - wall_start
    finally:
        stop.set()
        worker.join(timeout=1.0)
        if rec is not None:
            rec.stop_recording()

    if n_steps:
        en.logger.info(
            f"[sim2sim] {n_steps} control step(s) @ dt={control_dt * 1e3:.1f} ms | "
            f"mean step compute {busy_time / n_steps * 1e3:.1f} ms | "
            f"{shared['version']} policy inference(s) | "
            f"{n_stale} stale-action step(s) | {n_overruns} overrun(s) | "
            f"real-time factor {n_steps * control_dt / max(wall_time, 1e-9):.2f}x"
        )
    metrics.report()
