"""Roll a task's scripted demonstrator out in parallel envs and record every episode (``eden datagen``).

OCBench's ``data_gen_scripts/generate_ocbench.py``, on Eden's recorder: the env records itself
(:class:`~eden.managers.recorder_manager.RecorderManager`), so a demonstration is written in the same
HDF5 layout as a teleoperated one and ``eden lerobot convert`` reads either.

A demonstrator is either a keyframe controller
(:class:`~eden.extensions.scripted_demos.controller.ScriptedPickPlaceController`) or a tool routine
(:class:`~eden.extensions.scripted_demos.routines.RoutineDemonstrator`, or one per arm composed by a
:class:`~eden.extensions.scripted_demos.multi_arm.MultiArmDemonstrator`); all expose ``reset`` /
``update_subgoals`` / ``act``. A two-arm task records both arms' state, side by side. A routine
demonstrator also keeps a per-episode trace, and the report carries the first failed episode's —
the place to start when the success rate is low.

An episode ends when the demonstrator has finished the task — its success is the task's own
success metric at that step — or when the env ends it (time limit, an object off the table). Every
ended episode is exported with its ``success`` flag; ``export_mode`` decides whether the failed
ones are kept. Generation stops once ``num_episodes`` have ended; the episodes still running then
are dropped rather than written cut short.
"""

from __future__ import annotations

import inspect
import math
import os
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import genesis as gs
import torch

import eden as en
from eden.constants import DatasetExportMode
from eden.extensions.scripted_demos.registry import SCRIPTED_DEMO_REGISTRY
from eden.extensions.scripted_demos.multi_arm import MultiArmDemonstrator
from eden.extensions.scripted_demos.rig import find_gripper_rigs
from eden.extensions.scripted_demos.routines import RoutineDemonstrator
from eden.managers.terms.recorders import (
    ActionRecorder,
    InitialStateRecorder,
    ObservationRecorder,
    RLStepRecorder,
    StateRecorder,
)
from eden.options.managers.recorders import RecorderManagerOptions
from eden.utils.file_handler.base import FILE_HANDLER_REGISTRY
from eden.utils.file_handler.hdf5_file_handler import HDF5FileHandler

if TYPE_CHECKING:
    from eden.utils.configs import EdenRLConfig

#: Default time limit [s] of a generated episode: room for a slow demonstrator to recover from a
#: mistake and a failed pick. The tasks' own limits are sized for RL rollouts, not for this. A
#: demonstrator factory that needs longer declares it as its ``episode_length_s`` attribute.
DEFAULT_EPISODE_LENGTH_S = 90.0


@dataclass
class GenerationReport:
    """What a generation run produced.

    Attributes
    ----------
    dataset_path : str
        The dataset file the successful (or all) episodes went to.
    episodes : int
        Episodes that ended (exported or not, per ``export_mode``).
    successes : int
        Of those, the ones the task's success metric passed.
    cut_short : int
        Episodes that ended unsuccessfully before the demonstrator finished: the time limit, or a
        task termination (an object off the table). A task that ends its episode *on* success
        (Wipe) cuts nothing short.
    steps : int
        Control steps recorded across all ended episodes.
    wall_s : float
        Wall-clock duration of the rollout.
    subgoals : dict[str, int]
        Subgoals started, by kind (``goal``, ``clear``, ``mistake``) — episodes still running when
        generation stopped included. For a routine demonstrator: motions ended, by ``<tool>:<status>``.
    failure_trace : list[str]
        A routine demonstrator's trace of the first episode that ended unsuccessfully (empty for a
        keyframe demonstrator, or when every episode succeeded).
    """

    dataset_path: str
    episodes: int = 0
    successes: int = 0
    cut_short: int = 0
    steps: int = 0
    wall_s: float = 0.0
    subgoals: dict[str, int] = field(default_factory=dict)
    failure_trace: list[str] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return self.successes / self.episodes if self.episodes else 0.0

    def render(self) -> str:
        return (
            f"{self.episodes} episodes, {self.successes} successful ({100 * self.success_rate:.1f} %), "
            f"{self.cut_short} cut short; {self.steps} steps in {self.wall_s:.0f} s; "
            f"subgoals {self.subgoals} -> {self.dataset_path}"
        )


def recorder_options(
    output_dir: str, filename: str, export_mode: DatasetExportMode, *, state_entity: str | list[str]
) -> RecorderManagerOptions:
    """What a demonstration records: observations, actions, rewards / dones / timestamps, the arm's state, and each
    episode's initial state.

    ``obs/<group>`` and ``action`` are the behavior-cloning pairs; ``state/*`` is ``state_entity``'s
    joint and root state — every arm's, concatenated in order, when it lists several — which
    ``eden lerobot convert`` reads as ``observation.state``; ``initial_state`` lets an episode be
    replayed from its first frame.
    """
    return RecorderManagerOptions(
        file_handler_options=HDF5FileHandler.configure(),
        dataset_export_dir_path=output_dir,
        dataset_filename=filename,
        dataset_export_mode=export_mode,
        override=True,
        observations=ObservationRecorder.configure(),
        actions=ActionRecorder.configure(),
        rl_step=RLStepRecorder.configure(),
        state=StateRecorder.configure(entity_name=state_entity),
        initial_state=InitialStateRecorder.configure(),
    )


def generate_demos(
    config: EdenRLConfig,
    task_name: str,
    *,
    num_episodes: int,
    output_dir: str,
    filename: str = "demos",
    export_mode: DatasetExportMode = DatasetExportMode.EXPORT_ALL,
    seed: int = 0,
    p_mistake: float | None = None,
    episode_length_s: float | None = None,
    camera_only: bool = False,
    show_viewer: bool = False,
) -> GenerationReport:
    """Generate ``num_episodes`` scripted demonstrations of ``task_name`` into ``output_dir/filename``.

    Parameters
    ----------
    config : EdenRLConfig
        The task config (mods applied); its recorder options and episode length are replaced.
    task_name : str
        Key of the task's demonstrator in :data:`~eden.extensions.scripted_demos.registry.SCRIPTED_DEMO_REGISTRY`.
    num_episodes : int
        Episodes to end before stopping.
    output_dir, filename : str
        Where the dataset goes (``filename`` without extension).
    export_mode : DatasetExportMode
        Keep every episode, the successful ones only, or both in separate files. Not
        ``EXPORT_NONE``: successes are counted off what is exported.
    seed : int
        Seeds the env and the demonstrator.
    p_mistake : float, optional
        Probability a subgoal is a deliberate mistake; the demonstrator's default when None.
    episode_length_s : float, optional
        Time limit [s] of an episode; the demonstrator's own ``episode_length_s`` (or
        :data:`DEFAULT_EPISODE_LENGTH_S`) when None.
    camera_only : bool
        Run the demonstrator without the simulator's object state (``privileged=False``): it sees
        through the task's cameras, which then render every env (``env_separate_rigid``). Only
        demonstrators that take a ``privileged`` argument can.
    show_viewer : bool
        Open the Genesis viewer.

    Returns
    -------
    GenerationReport
        Counts and where the dataset went.
    """
    if export_mode == DatasetExportMode.EXPORT_NONE:
        raise ValueError("export_mode=EXPORT_NONE would generate demonstrations and write none of them.")
    if num_episodes <= 0:
        raise ValueError(f"num_episodes must be a positive integer, got {num_episodes}.")
    if task_name not in SCRIPTED_DEMO_REGISTRY:
        available = ", ".join(sorted(SCRIPTED_DEMO_REGISTRY.keys())) or "none"
        raise KeyError(
            f"Task '{task_name}' has no scripted demonstrator. Tasks with one: {available}. A demonstrator is a factory "
            f"registered in SCRIPTED_DEMO_REGISTRY under the task's name (see eden/tasks/robosuite/scripted_demos.py)."
        )
    from eden.envs.base import env_class_for_config

    factory = SCRIPTED_DEMO_REGISTRY.get(task_name)
    factory_kwargs = {"seed": seed}
    if p_mistake is not None:
        factory_kwargs["p_mistake"] = p_mistake
    if camera_only:
        if "privileged" not in inspect.signature(factory).parameters:
            camera_capable = sorted(
                name  # the registry iterates (name, factory) pairs
                for name, candidate in SCRIPTED_DEMO_REGISTRY
                if "privileged" in inspect.signature(candidate).parameters
            )
            raise ValueError(
                f"Task '{task_name}''s demonstrator reads the simulator's object state and has no camera-only mode; "
                f"tasks that do: {camera_capable or 'none'}. Drop --camera_only for this task."
            )
        factory_kwargs["privileged"] = False
        if config.env_options.num_envs > 1 and not config.env_options.env_separate_rigid:
            # A camera shared by the envs renders env 0 only; a camera-only demonstrator would see one scene everywhere.
            config.env_options.env_separate_rigid = True
            en.logger.info("--camera_only: rendering every env's own view (env_options.env_separate_rigid = True).")
    if episode_length_s is None:
        episode_length_s = getattr(factory, "episode_length_s", DEFAULT_EPISODE_LENGTH_S)
    if not episode_length_s > 0:
        raise ValueError(f"episode_length_s must be a positive number of seconds, got {episode_length_s}.")
    config.env_options.episode_length_s = episode_length_s
    arms = [rig.arm_name for rig in find_gripper_rigs(config.scene_options)]
    if not arms:
        raise TypeError(
            f"Task '{task_name}' holds no arm + two-finger gripper a scripted demonstrator could drive; scene entities: "
            f"{list(config.scene_options.keys())}."
        )
    state_entity = arms[0] if len(arms) == 1 else arms
    config.recorder_options = recorder_options(output_dir, filename, export_mode, state_entity=state_entity)
    env = env_class_for_config(config).from_config(config, show_viewer=show_viewer, eval_mode=True)
    env.build()
    env.reset(seed=seed)
    if env.metric_manager is None or not hasattr(env.metric_manager, "success_buf"):
        raise ValueError(
            f"Task '{task_name}' declares no success metric, so a generated episode cannot be labelled; give it a "
            f"metric term with contributes_to_success=True."
        )

    controller = factory(env, **factory_kwargs)
    controller.reset()

    recorder = env.recorder_manager
    report = GenerationReport(dataset_path=recorder_dataset_path(config.recorder_options))
    episode_steps = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    # The generator's own time limit: a task need not declare a time_out termination (SIMPLER's
    # declare none), and a demonstrator that cannot finish must not hold its env forever.
    max_episode_steps = math.ceil(episode_length_s / env.dt)
    # Successful episodes exported per env so far: an env-ended episode's success is read off the
    # exporter, since the env clears its success buffer when it resets the env inside the step.
    exported_successes = [0] * env.num_envs
    recorder.start_recording()
    start = time.time()
    try:
        while report.episodes < num_episodes:
            done = controller.update_subgoals()
            timed_out = (episode_steps >= max_episode_steps) & ~done
            report.cut_short += int(timed_out.sum())
            ending = done | timed_out
            if bool(ending.any()):
                recorder.set_episode_success(ending, env.metric_manager.success_buf[ending])
                _keep_failure_trace(report, controller, ending & ~env.metric_manager.success_buf)
                _tally(report, ending, episode_steps)
                env.reset(ending)
                _count_exported_successes(recorder, ending, exported_successes)
                controller.reset(ending)
                episode_steps[ending] = 0
                if report.episodes >= num_episodes:
                    break

            try:
                env.step(controller.act())
            except gs.GenesisException as error:
                if "not finite" not in str(error) or not config.env_options.use_gjk_collision:
                    raise
                raise RuntimeError(
                    f"Genesis's GJK narrowphase produced a non-finite contact while generating '{task_name}' demos. "
                    f"Thin convex collision geoms (a WidowX's finger pads, say) pressed against an object trigger "
                    f"it, so a long scripted rollout eventually does. Rerun with MPR collision: eden datagen "
                    f"--task {task_name} --override 'env_options: {{use_gjk_collision: false}}'. Catching this "
                    f"and resetting the env does not help: the solver state of every env is invalid after it."
                ) from error
            episode_steps += 1

            ended = env.reset_buf
            if bool(ended.any()):  # the env has already reset these, and exported them with their success
                succeeded = _count_exported_successes(recorder, ended, exported_successes)
                report.cut_short += int((ended & ~succeeded).sum())
                _keep_failure_trace(report, controller, ended & ~succeeded)
                _tally(report, ended, episode_steps)
                controller.reset(ended)
                episode_steps[ended] = 0
    finally:
        recorder.end_recording(export_unfinished=False)
        # Counted by the exporter: the env clears its success buffer when it resets an env, so the
        # flag an env-ended episode carried survives only in what was exported.
        report.successes = recorder.exported_successful_episode_count
        recorder.close()
    report.wall_s = time.time() - start
    report.subgoals = dict(controller.subgoal_counts)
    en.logger.info(f"Scripted demos for '{task_name}': {report.render()}")
    if report.failure_trace:
        en.logger.info("Trace of the first unsuccessful episode:\n" + "\n".join(report.failure_trace))
    return report


def recorder_dataset_path(options: RecorderManagerOptions) -> str:
    """The file a recorder with ``options`` writes its (successful) episodes to."""
    handler = FILE_HANDLER_REGISTRY.get(options.file_handler_options.name)()
    return handler.resolve_path(os.path.join(options.dataset_export_dir_path, options.dataset_filename))


def _keep_failure_trace(report: GenerationReport, controller, failed: torch.Tensor) -> None:
    """Keep a routine (or multi-arm) demonstrator's trace of the first of ``failed`` (an ``(N,)`` mask), if none is
    kept yet.
    """
    if (
        report.failure_trace
        or not isinstance(controller, (RoutineDemonstrator, MultiArmDemonstrator))
        or not bool(failed.any())
    ):
        return
    env_id = int(failed.nonzero(as_tuple=False)[0])
    report.failure_trace = [f"env {env_id}:"] + controller.trace(env_id)


def _count_exported_successes(recorder, ended: torch.Tensor, exported_successes: list[int]) -> torch.Tensor:
    """``(N,)`` bool: which of ``ended``'s just-exported episodes were successful; updates ``exported_successes``.

    Read off the exporter's per-env count rather than the env's success buffer, which the env clears
    when it resets an env — after a task terminates the episode *on* success (Wipe), the buffer
    already reads False.
    """
    succeeded = torch.zeros_like(ended)
    for env_id in ended.nonzero(as_tuple=False).flatten().tolist():
        count = recorder.get_exported_successful_episode_count(env_id)
        succeeded[env_id] = count > exported_successes[env_id]
        exported_successes[env_id] = count
    return succeeded


def _tally(report: GenerationReport, ended: torch.Tensor, episode_steps: torch.Tensor) -> None:
    report.episodes += int(ended.sum())
    report.steps += int(episode_steps[ended].sum())
