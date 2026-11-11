"""Eden command-line interface entry point.

Declares the ``COMMANDS`` table — one :class:`Command` per subcommand, wiring
each to a handler in the area modules under :mod:`eden.utils.cli` (grouped by
area: ``tasks``, ``terms``, ``examples``, ``viewers``, ``devices``,
``dataset``, ``pose``, ``retarget``, ``teleop``, ``maintenance``) — then
builds the argparse tree and dispatches over it via the primitives in
:mod:`eden.utils.cli.registry`.

This is a curated in-tree table, *not* an :class:`eden.utils.registry.Registry`:
commands are instances, the set is closed, and there's no plugin discovery.

To add a subcommand: write the handler in its area module, then add one
:class:`~eden.utils.cli.registry.Command` entry to ``COMMANDS`` below — that
class documents the three shapes an entry can take.
"""

from __future__ import annotations

import argparse
import sys

import eden
from eden._logging import LOG_THEME_ENV_VAR
from eden.utils.cli.dataset import _run_datagen, _run_lerobot_convert
from eden.utils.cli.devices import (
    _run_gello_calibrate,
    _run_gello_leader,
    _run_metaquest_bridge,
    _run_pico_bridge,
    _run_wuji_monitor,
    _run_wuji_scan,
)
from eden.utils.cli.examples import (
    _configure_examples_list,
    _configure_examples_smoke,
    _run_examples_list,
    _run_examples_smoke,
)
from eden.utils.cli.maintenance import _run_clean, _run_list
from eden.utils.cli.pose import _run_gvhmr
from eden.utils.cli.registry import Command, _add_command, _dispatch
from eden.utils.cli.retarget import (
    _run_dex,
    _run_flashdex_import,
    _run_flashdex_kinematic,
    _run_omniretarget,
    _run_otretarget,
    _run_soma,
)
from eden.utils.cli.tasks import (
    deploy_task,
    export_task,
    inference_task,
    preprocess_task,
    smoke_task,
    train_task,
)
from eden.utils.cli.teleop import _run_sonic_operator, _run_sonic_robot
from eden.utils.cli.terms import _configure_terms, _run_terms
from eden.utils.cli.viewers import (
    _configure_arrange,
    _configure_gif,
    _configure_snap,
    _configure_view,
    _run_arrange,
    _run_gif,
    _run_snap,
    _run_view,
)

COMMANDS: tuple[Command, ...] = (
    Command("list", "List all available tasks", handler=_run_list),
    Command(
        "terms",
        "List every registered term and the parameters it accepts "
        "(e.g. 'eden terms --kind reward', 'eden terms --json')",
        handler=_run_terms,
        configure=_configure_terms,
    ),
    Command(
        "examples",
        "Browse the runnable examples catalog under examples/",
        subcommands=(
            Command(
                "list",
                "List/filter example scripts by tier, capability, or hardware "
                "(e.g. 'eden examples list --tier recipe --capability sensor')",
                handler=_run_examples_list,
                configure=_configure_examples_list,
            ),
            Command(
                "smoke",
                "Smoke-run one example headless in a few steps (EDEN_SMOKE); "
                "refuses examples needing hardware/devices/missing deps",
                handler=_run_examples_smoke,
                configure=_configure_examples_smoke,
            ),
        ),
    ),
    Command(
        "train",
        "Train a registered task. Use 'eden train --task <name> --help' for task options.",
        handler=train_task,
        forwards_extra=True,
    ),
    Command(
        "inference",
        "Run inference for a registered task with a trained checkpoint, or roll out a "
        "frozen extension policy in sim (--policy). Add --async for a deployment-style, "
        "wall-clock-paced rollout (a sim-side dry run of 'eden deploy').",
        handler=inference_task,
        forwards_extra=True,
    ),
    Command(
        "export",
        "Export a trained checkpoint's policy as ONNX / TorchScript plus a policy_spec.json "
        "(observations, actions, control rate) for a runtime outside Eden.",
        handler=export_task,
        forwards_extra=True,
    ),
    Command(
        "deploy",
        "Deploy a trained task policy on real hardware (robot picked from the task's scene), "
        "or against a MuJoCo model of that robot for sim-to-sim validation (--mujoco).",
        handler=deploy_task,
        forwards_extra=True,
    ),
    Command(
        "preprocess",
        "Build the v2 dataset cache for a registered task. Supports --num_batches/--batch_idx for multi-GPU sharding.",
        handler=preprocess_task,
        forwards_extra=True,
    ),
    Command(
        "smoke",
        "Build a registered task and step it a few times with no action — a quick "
        "end-to-end sanity check. Use 'eden smoke --task <name> --help' for options.",
        handler=smoke_task,
        forwards_extra=True,
    ),
    Command(
        "datagen",
        "Generate scripted, demo-like demonstrations of a task into an HDF5 dataset (OCBench-style "
        "demonstrators). Use 'eden datagen --task <name> --help' for options.",
        handler=_run_datagen,
        forwards_extra=True,
    ),
    Command(
        "gello",
        "Gello commands",
        subcommands=(
            Command(
                "calibrate",
                "Calibrate the Gello leader arm (pass --help after 'calibrate' for options)",
                handler=_run_gello_calibrate,
                forwards_extra=True,
            ),
            Command(
                "leader",
                "Read the Gello leader arm and send the joint states to a GelloZMQServer "
                "(pass --help after 'leader' for options)",
                handler=_run_gello_leader,
                forwards_extra=True,
            ),
        ),
    ),
    Command(
        "lerobot",
        "LeRobot dataset commands",
        subcommands=(
            Command(
                "convert",
                "Convert an Eden HDF5 recording into a LeRobot v3 dataset (pass --help after 'convert' for options)",
                handler=_run_lerobot_convert,
                forwards_extra=True,
            ),
        ),
    ),
    Command(
        "metaquest",
        "Meta Quest WebXR teleoperation commands",
        subcommands=(
            Command(
                "bridge",
                "Run the televuer XR bridge (requires the 'metaquest' extra; pass --help after 'bridge' for options)",
                handler=_run_metaquest_bridge,
                forwards_extra=True,
            ),
        ),
    ),
    Command(
        "pico",
        "PICO (XRoboToolkit) teleoperation commands",
        subcommands=(
            Command(
                "bridge",
                "Run the PICO XR bridge — publishes headset/controller tracking and streams the "
                "stereo view back to the headset (requires the 'pico' extra; pass --help after "
                "'bridge' for options)",
                handler=_run_pico_bridge,
                forwards_extra=True,
            ),
        ),
    ),
    Command(
        "teleop",
        "Teleoperation pipelines (operator station + robot process)",
        subcommands=(
            Command(
                "sonic",
                "GEAR-SONIC whole-body teleoperation (PICO)",
                subcommands=(
                    Command(
                        "robot",
                        "Run the robot process: Genesis G1 body + SONIC policy + planner; listens for the "
                        "operator's ZMQ commands (pass --help after 'robot' for options)",
                        handler=_run_sonic_robot,
                        forwards_extra=True,
                    ),
                    Command(
                        "operator",
                        "Run the operator station: PICO body tracking, mode state machine, calibration, "
                        "and streaming to the robot process (pass --help after 'operator' for options)",
                        handler=_run_sonic_operator,
                        forwards_extra=True,
                    ),
                ),
            ),
        ),
    ),
    Command(
        "pose",
        "Human pose / motion estimation (video → SMPL motion)",
        subcommands=(
            Command(
                "gvhmr",
                "World-grounded SMPL-X motion from monocular video via GVHMR (pass --help after 'gvhmr' for options)",
                handler=_run_gvhmr,
                forwards_extra=True,
            ),
        ),
    ),
    Command(
        "retarget",
        "Motion retargeting (human → robot)",
        subcommands=(
            Command(
                "omniretarget",
                "Robot-only OMOMO → G1 retargeting via OmniRetarget (pass --help after 'omniretarget' for options)",
                handler=_run_omniretarget,
                forwards_extra=True,
            ),
            Command(
                "otretarget",
                "Joint robot + object retargeting from surface interactions via OTRetarget "
                "(OMOMO hands/body/object, BONES-SEED BVH; pass --help after 'otretarget' for options)",
                handler=_run_otretarget,
                forwards_extra=True,
            ),
            Command(
                "soma",
                "Batch SMPL/SMPL-H/SMPL-X → robot retargeting via SomaRetargeter (pass --help after 'soma' for options)",
                handler=_run_soma,
                forwards_extra=True,
            ),
            Command(
                "dex",
                "Offline MANO-keypoint → dexterous-hand position retargeting via DexRetargeter "
                "(pass --help after 'dex' for options)",
                handler=_run_dex,
                forwards_extra=True,
            ),
            Command(
                "flashdex",
                "FlashDexRetarget: human hand-object demos (TACO / OakInk2 / HOT3D / DexYCB) → bimanual robot-hand "
                "motion.pt",
                subcommands=(
                    Command(
                        "import",
                        "Import a released dataset into <dataset>/human_demo/ and objects/ "
                        "(pass --help after 'import' for options)",
                        handler=_run_flashdex_import,
                        forwards_extra=True,
                    ),
                    Command(
                        "kinematic",
                        "Support disks, convex parts, MuJoCo physics IK and motion.pt for one robot "
                        "(pass --help after 'kinematic' for options)",
                        handler=_run_flashdex_kinematic,
                        forwards_extra=True,
                    ),
                ),
            ),
        ),
    ),
    Command(
        "wuji",
        "Wuji Glove teleoperation commands",
        subcommands=(
            Command(
                "scan",
                "List every Wuji Glove on the bus (requires the 'wuji' extra)",
                handler=_run_wuji_scan,
                forwards_extra=True,
            ),
            Command(
                "monitor",
                "Watch a Wuji Glove's live hand pose and tactile sheet in the terminal — the "
                "first check after plugging one in (requires the 'wuji' extra; pass --help "
                "after 'monitor' for options)",
                handler=_run_wuji_monitor,
                forwards_extra=True,
            ),
        ),
    ),
    Command("gif", "Convert mp4 video to gif", handler=_run_gif, configure=_configure_gif),
    Command(
        "view",
        "View an entity by name, or a gaussian-splat .ply by path",
        handler=_run_view,
        configure=_configure_view,
    ),
    Command(
        "arrange",
        "Interactively arrange entities with ImGui sliders (base 6-DoF + joints) and export a SceneOptions snippet",
        handler=_run_arrange,
        configure=_configure_arrange,
    ),
    Command("snap", "Take a snapshot of the scene/entity", handler=_run_snap, configure=_configure_snap),
    Command("clean", "Clean up logs", handler=_run_clean),
)


def main():
    """Eden CLI entry point: build the parser from ``COMMANDS`` and dispatch."""
    # prog="eden" so `python -m eden` prints `usage: eden ...`, not `usage: __main__.py` —
    # the module form is the documented fallback when the console script isn't on PATH,
    # and it matches the `prog="eden <sub>"` the subcommands already set.
    parser = argparse.ArgumentParser(prog="eden", description="Eden - Robot Learning Environments")
    parser.add_argument(
        "--log-theme",
        choices=("auto", "human", "agent"),
        default=None,
        help=(
            "How this command renders its console output: agent (the default — no ANSI, no "
            "greeting box), human (color), auto (color on a TTY, honoring NO_COLOR/FORCE_COLOR). "
            "Goes before the subcommand ('eden --log-theme human train ...'), because the "
            "forwarding subcommands hand everything after their own flags to the task parser. "
            "Sets EDEN_LOG_THEME for this process and anything it launches; setting that "
            "variable directly is the way to pick a theme for a whole shell."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    parsers: dict[Command, argparse.ArgumentParser] = {}
    for cmd in COMMANDS:
        _add_command(subparsers, cmd, parsers)

    args, extra_args = parser.parse_known_args()

    # Before anything renders. Exported rather than threaded through every handler: the theme has
    # to reach `eden.init()` (whose `logger_theme=None` reads it), the argparse help formatters,
    # and any subprocess a command spawns — one variable does all three, and the flag is then
    # exactly the documented environment variable with a shorter spelling. An explicit
    # `--log-theme` beats an inherited EDEN_LOG_THEME, which is the usual precedence for a flag.
    if args.log_theme is not None:
        # `set_theme` exports the variable (for anything this process launches) and applies it.
        # This process itself built `eden.logger` back at `import eden`, and imported the config
        # modules (which seed Genesis' palette) before argv was parsed, so the flag has to be
        # *applied* as well as exported -- otherwise every record up to `eden.init()` (the
        # `--resume` line, the config dump) still renders in the inherited theme.
        eden.logger.set_theme(args.log_theme)

    # A global flag on the root parser is only reachable before the subcommand, and every
    # command after it either rejects it as unknown or forwards it to a task parser that does.
    # Both of those name the flag and nothing else, so say where it goes — this is the one
    # mistake the placement guarantees someone will make.
    # Matched on the prefix, not the exact spelling: argparse accepts any unambiguous abbreviation
    # of a long option, so someone who typed `--log-th` where it works keeps typing it where it
    # does not. `len > 2` excludes a bare `--`, which every prefix check otherwise matches.
    misplaced = [
        arg for arg in extra_args if len(prefix := arg.split("=", 1)[0]) > 2 and "--log-theme".startswith(prefix)
    ]
    if misplaced:
        # `args.command` is None when argparse took the subcommand slot for something else; an
        # error message that says `eden None ...` sends the reader looking for a command by that
        # name, so name the slot instead of printing the miss.
        subcommand = args.command or "<command>"
        parser.error(
            f"{misplaced[0]} came after the subcommand, where it belongs to "
            f"'eden {subcommand}' rather than to eden. Put it first: "
            f"`eden --log-theme <auto|human|agent> {subcommand} ...`, or set "
            f"{LOG_THEME_ENV_VAR} for the whole shell."
        )

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    cmd = next(c for c in COMMANDS if c.name == args.command)

    # Forwarding commands and groups parse leftover args themselves; for everyone
    # else, unknown args are a user error (so typos like `eden view r --collison`
    # aren't silently dropped) — mirrors argparse's parse_args behavior.
    if extra_args and not cmd.consumes_extra:
        parser.error(f"unrecognized arguments: {' '.join(extra_args)}")

    _dispatch(cmd, args, extra_args, parsers)


if __name__ == "__main__":
    # Usage:
    #   eden list
    #   eden view <entity_name>
    #   eden arrange <EntityOptionsClass> [<EntityOptionsClass> ...]
    main()
