"""``eden.envs.wrappers.rsl_rl_env``: the VecEnv wrapper, Eden's rsl_rl runners, and checkpoint loading.

Everything runs the installed rsl-rl-lib against the ``tests/envs`` pendulum task; checkpoints are real
files saved by a real runner (or rewritten from one into an older layout).
"""

from __future__ import annotations

import json
import logging
import os
import pathlib
from importlib import metadata

import gymnasium as gym
import pytest
import torch

from eden.envs.wrappers import rsl_rl_env
from eden.envs.wrappers.rsl_rl_env import (
    AgentIterationLogMixin,
    DDPDistillationRunner,
    DDPOnPolicyRunner,
    MotionWeightCurriculumMixin,
    MimicDistillationRunner,
    MimicOnPolicyRunner,
    RslRlVecEnvWrapper,
    _audit_rsl_rl_checkpoint_keys,
    _detect_std_type,
    _maybe_convert_legacy_checkpoint,
    _rsl_rl_checkpoint_format_error,
    _runner_uses_split_policy,
    compact_iteration_log,
    load_runner_checkpoint,
    make_rsl_rl_runner,
    rsl_rl_runner_class,
)
from eden.utils.configs import EdenConfig, EdenRLConfig
from eden.utils.warning import warn_once
from tests.envs.conftest import trace_config
from tests.envs.wrappers.conftest import rsl_rl_runner_options
from tests.support.motion import build_mimic_env, manager_options, pin_clips, term_options, write_library

INSTALLED = metadata.version("rsl-rl-lib")


@pytest.fixture(scope="module")
def wrapper(pendulum_file, tmp_path_factory) -> RslRlVecEnvWrapper:
    config = trace_config(pendulum_file, str(tmp_path_factory.mktemp("recordings")))
    config.runner_options = rsl_rl_runner_options()
    return RslRlVecEnvWrapper.from_config(config)


@pytest.fixture
def runner(wrapper, tmp_path):
    return make_rsl_rl_runner(wrapper, log_dir=str(tmp_path / "run"))


def _progress(runner) -> dict:
    return json.loads(pathlib.Path(runner.logger.log_dir, "progress.json").read_text())


def _draw_after_from_config(config, **kwargs) -> float:
    """Build through the wrapper and return the next torch draw: a fingerprint of the RNG state."""
    RslRlVecEnvWrapper.from_config(config, **kwargs)
    return torch.rand(1).item()


# ---------------------------------------------------------------------------
# The wrapper
# ---------------------------------------------------------------------------


def test_a_config_without_runner_options_is_pointed_at_rlenvbase(wrapper):
    """Both entry points guard, and a plain ``EdenConfig`` (no such field) gets the same message."""
    with pytest.raises(ValueError, match="requires `runner_options`"):
        RslRlVecEnvWrapper(wrapper.unwrapped, options=None)
    for config in (EdenRLConfig(), EdenConfig()):
        with pytest.raises(ValueError, match=r"no `runner_options`.*RLEnvBase\.from_config"):
            RslRlVecEnvWrapper.from_config(config)


def test_step_reports_dones_as_terminated_or_truncated_and_time_outs_only_where_not_terminated(wrapper):
    """rsl_rl bootstraps V-targets from ``extras["time_outs"]``, which must be present on every step,
    whatever ``dones`` says: an env that terminated on the step it timed out is terminal, so it is left out.
    """
    env = wrapper.unwrapped
    ending = env.termination_manager.get_term("ends").ending
    ending[:] = torch.tensor([True, False])  # env 0 terminates ...
    env.episode_length_buf[:] = env.max_episode_length - 1  # ... and both time out
    try:
        _, _, dones, extras = wrapper.step(torch.zeros(env.num_envs, 1))
    finally:
        ending[:] = False

    assert dones.dtype == torch.long and dones.tolist() == [1, 1]
    assert extras["time_outs"].tolist() == [False, True]

    _, _, dones, extras = wrapper.step(torch.zeros(env.num_envs, 1))
    assert dones.tolist() == [0, 0] and extras["time_outs"].tolist() == [False, False]


def test_observation_spaces_follow_each_groups_layout(wrapper):
    """A concatenated group is one Box; a non-concatenated one keeps every term (the loop once kept the last)."""
    policy = wrapper.single_observation_space["policy"]
    critic = wrapper.single_observation_space["critic"]

    assert isinstance(policy, gym.spaces.Box) and policy.shape == (6,)  # (trace obs + dofs_pos) x history 3
    assert isinstance(critic, gym.spaces.Dict)
    assert {name: space.shape for name, space in critic.spaces.items()} == {"dofs_pos": (1,), "dofs_vel": (1,)}


def test_config_is_the_envs_config_carrying_the_wrappers_runner_options(wrapper):
    """A wrapper built directly holds runner options the base env never saw; ``cfg`` is what rsl_rl reads."""
    options = rsl_rl_runner_options(num_steps_per_env=8)
    direct = RslRlVecEnvWrapper(wrapper.unwrapped, options)

    assert type(direct.config) is type(wrapper.unwrapped.config)
    assert direct.config.runner_options is options
    assert direct.cfg.runner_options is options
    assert direct.config.env_options is wrapper.unwrapped.env_options


# ---------------------------------------------------------------------------
# Run seeding: from_config owns it, before the env exists
# ---------------------------------------------------------------------------


def test_from_config_seeds_the_run_from_runner_options_before_building(rl_config):
    same = [_draw_after_from_config(rl_config(rsl_rl_runner_options(seed=4242))) for _ in range(2)]
    other = _draw_after_from_config(rl_config(rsl_rl_runner_options(seed=4243)))

    assert same[0] == same[1]
    assert other != same[0]


def test_from_config_offsets_the_seed_by_the_ddp_rank_once_and_leaves_the_config_alone(rl_config, monkeypatch):
    """Rank ``r`` draws ``base + r``; building twice from one config must not compound it to ``base + 2r``."""
    rank_zero_equivalent = _draw_after_from_config(rl_config(rsl_rl_runner_options(seed=4244)))
    monkeypatch.setenv("RANK", "2")
    config = rl_config(rsl_rl_runner_options(seed=4242))

    assert [_draw_after_from_config(config) for _ in range(2)] == [rank_zero_equivalent] * 2
    assert config.runner_options.seed == 4242


def test_apply_seed_false_leaves_the_process_rng_alone(rl_config):
    """Hardware deployment mirrors a live robot; reseeding the process behind the operator is a side effect."""
    draws = []
    for seed in (4242, 999):
        torch.manual_seed(0)
        draws.append(_draw_after_from_config(rl_config(rsl_rl_runner_options(seed=seed)), apply_seed=False))

    assert draws[0] == draws[1]


# ---------------------------------------------------------------------------
# Eden's runners
# ---------------------------------------------------------------------------


def test_progress_file_tracks_every_iteration_and_ends_completed(wrapper, runner):
    """``progress.json`` is rewritten each ``logger.log`` and stamped ``completed`` once ``learn`` returns."""
    ending = wrapper.unwrapped.termination_manager.get_term("ends").ending
    ending[:] = True  # every step ends every episode, so the reward and length buffers fill
    try:
        runner.learn(num_learning_iterations=2)
    finally:
        ending[:] = False

    progress = _progress(runner)
    assert progress["status"] == "completed"
    assert (progress["iteration"], progress["total_iterations"]) == (1, 2)
    assert progress["total_timesteps"] == 2 * 4 * wrapper.num_envs
    assert (progress["mean_reward"], progress["best_mean_reward"], progress["mean_episode_length"]) == (0.0, 0.0, 1.0)
    assert progress["elapsed_s"] > 0.0 and isinstance(progress["updated_at"], float)
    assert runner.logger.log.__name__ == "log"  # the wrap is scoped to learn()


def test_progress_file_stays_running_when_learn_fails(wrapper, runner):
    """A crash never reaches the terminal write: ``running`` with a stale ``updated_at`` is the crashed signal."""
    term = wrapper.unwrapped.termination_manager.get_term("ends")
    term.raise_at_step = wrapper.unwrapped.common_step_counter + 4 + 2  # into the second iteration's rollout
    try:
        with pytest.raises(RuntimeError, match="as the test asked"):
            runner.learn(num_learning_iterations=3)
    finally:
        term.raise_at_step = None

    progress = _progress(runner)
    assert progress["status"] == "running" and progress["iteration"] == 0
    assert progress["mean_reward"] is None  # no episode finished: null, not a crash
    assert runner.logger.log.__name__ == "log"


def test_progress_file_is_not_written_without_a_log_dir_or_on_a_logging_disabled_rank(wrapper, tmp_path):
    silent = MimicOnPolicyRunner(wrapper, wrapper.runner_dict, log_dir=None, device="cpu")
    silent.learn(num_learning_iterations=1)  # an eval runner: nothing to write, nothing to fail

    disabled = make_rsl_rl_runner(wrapper, log_dir=str(tmp_path))
    disabled.logger.disable_logs = True  # what rsl_rl sets on a non-zero DDP rank
    disabled.learn(num_learning_iterations=1)
    assert not (tmp_path / "progress.json").exists()


def test_a_non_zero_ddp_rank_runner_accumulates_no_episode_extras(wrapper, tmp_path, monkeypatch):
    """rsl_rl's rollout loop appends ``extras["log"]`` per step; on a rank that never logs, nothing may
    clear them, so they must not accumulate there (GPU tensors pinned for the whole run).
    """
    monkeypatch.setenv("WORLD_SIZE", "2")
    monkeypatch.setenv("RANK", "1")
    runner = make_rsl_rl_runner(wrapper, log_dir=str(tmp_path))
    assert isinstance(runner, DDPOnPolicyRunner)
    runner.logger.init_logging_writer()

    for _ in range(50):
        runner.logger.process_env_step(torch.zeros(2), torch.zeros(2), {"log": {"x": torch.ones(2)}})

    assert runner.logger.ep_extras == []


def _saved_checkpoints(wrapper, log_dir, **runner_overrides) -> list[str]:
    """Train 3 iterations saving every one, and list the ``model_*.pt`` left in ``log_dir``, oldest first."""
    runner_dict = {**wrapper.runner_dict, "save_interval": 1, **runner_overrides}
    MimicOnPolicyRunner(wrapper, runner_dict, log_dir=str(log_dir), device="cpu").learn(num_learning_iterations=3)
    return [name for _, name in rsl_rl_env.model_checkpoints(str(log_dir))]


def test_keep_last_checkpoints_leaves_only_the_newest_on_disk(wrapper, tmp_path):
    every = _saved_checkpoints(wrapper, tmp_path / "all", keep_last_checkpoints=None)
    assert len(every) >= 3  # one per iteration, and the final save

    assert _saved_checkpoints(wrapper, tmp_path / "two", keep_last_checkpoints=2) == every[-2:]
    assert _saved_checkpoints(wrapper, tmp_path / "default") == every[-1:]  # the newest only, by default


def test_a_stock_runner_name_maps_to_edens_runner_and_an_import_path_to_that_class():
    assert rsl_rl_runner_class("OnPolicyRunner") is MimicOnPolicyRunner
    assert rsl_rl_runner_class("eden.envs.wrappers.rsl_rl_env:DDPOnPolicyRunner") is DDPOnPolicyRunner


@pytest.mark.parametrize(
    "class_name, message",
    [
        ("eden.no_such_module:Runner", r"no module 'eden.no_such_module'"),
        (
            "eden.envs.wrappers.rsl_rl_env:NoSuchRunner",
            r"no such attribute\. Runner classes defined in 'eden.envs.wrappers.rsl_rl_env': \[.*'MimicOnPolicyRunner'",
        ),
        (
            "eden.envs.wrappers.rsl_rl_env:CheckpointRetentionMixin",
            r"lacks the runner interface \(load, get_inference_policy\)",
        ),
    ],
    ids=["missing_module", "missing_attribute", "not_a_runner"],
)
def test_a_bad_runner_import_path_names_the_value_and_what_exists(class_name, message):
    with pytest.raises(ValueError, match=message):
        rsl_rl_runner_class(class_name)


@pytest.mark.parametrize(
    "class_name, message",
    [
        ("OnPolicyRuner", r"'OnPolicyRuner' is neither a stock runner name \('OnPolicyRunner', 'DistillationRunner'\)"),
        (
            "eden.envs.wrappers.rsl_rl_env.DDPOnPolicyRunner",
            r"with a colon before the class: 'eden\.envs\.wrappers\.rsl_rl_env:DDPOnPolicyRunner'",
        ),
    ],
    ids=["typo", "dotted_path"],
)
def test_a_bare_name_that_is_not_a_stock_runner_is_refused_not_mapped_to_one(class_name, message):
    with pytest.raises(ValueError, match=message):
        rsl_rl_runner_class(class_name)


def test_under_a_distributed_launch_an_import_path_runner_must_compose_the_ddp_mixin(monkeypatch):
    from eden.utils import distributed

    monkeypatch.setattr(distributed, "is_distributed", lambda: True)
    assert rsl_rl_runner_class("OnPolicyRunner") is DDPOnPolicyRunner
    assert rsl_rl_runner_class("eden.envs.wrappers.rsl_rl_env:DDPOnPolicyRunner") is DDPOnPolicyRunner
    with pytest.raises(ValueError, match=r"MimicOnPolicyRunner does not compose .*DDPRunnerMixin.*without torchrun"):
        rsl_rl_runner_class("eden.envs.wrappers.rsl_rl_env:MimicOnPolicyRunner")


def test_the_inference_runner_dict_swaps_a_training_only_algorithm_for_the_one_it_extends(wrapper):
    from eden.learning.wrappers.rsl_rl.amp_ppo import RslRlAmpPpoAlgorithmOptions

    algorithm = RslRlAmpPpoAlgorithmOptions(amp_motion_file="objects/skateboard/husky_amp_push_motions.npz", gamma=0.9)
    amp = RslRlVecEnvWrapper(wrapper.unwrapped, wrapper.runner_options.model_copy(update={"algorithm": algorithm}))

    assert amp.runner_dict["algorithm"]["class_name"] == algorithm.class_name
    replay = amp.inference_runner_dict["algorithm"]
    assert replay["class_name"] == "PPO" and replay["gamma"] == 0.9
    assert not any(key.startswith("amp_") for key in replay)
    assert wrapper.inference_runner_dict == wrapper.runner_dict  # nothing to swap: the training dict itself


@pytest.fixture
def offline_wandb(tmp_path, monkeypatch):
    """Offline wandb: runs live in ``tmp_path``, and any run a test leaves open is finished after it."""
    wandb = pytest.importorskip("wandb")
    monkeypatch.setenv("WANDB_MODE", "offline")
    monkeypatch.setenv("WANDB_DIR", str(tmp_path))
    monkeypatch.setenv("WANDB_SILENT", "true")
    yield wandb
    if wandb.run is not None:
        wandb.finish()


@pytest.mark.parametrize("logger", ["wandb", {"class_name": "WandbLogWriter", "project_name": "eden-test"}])
def test_while_wandb_may_still_be_uploading_one_checkpoint_more_stays(wrapper, tmp_path, offline_wandb, logger):
    """A wandb run reads each saved checkpoint through a symlink in the background.

    The newest's predecessor may still be in flight, so it is only deleted at the save after next. rsl_rl takes the
    logger by name or, from 5.5, as a dict.
    """
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    runner = MimicOnPolicyRunner(wrapper, {**wrapper.runner_dict, "logger": logger}, log_dir=str(run_dir), device="cpu")
    offline_wandb.init(project="eden-test")
    for it in (1, 2, 3):
        runner.save(str(run_dir / f"model_{it}.pt"))

    assert sorted(p.name for p in run_dir.glob("model_*.pt")) == ["model_2.pt", "model_3.pt"]


def test_a_wandb_run_ends_with_only_its_newest_checkpoint_once_learn_has_finished_the_uploads(
    wrapper, tmp_path, offline_wandb
):
    runner_dict = {**wrapper.runner_dict, "logger": "wandb", "wandb_project": "eden-test", "save_interval": 1}
    runner = MimicOnPolicyRunner(wrapper, runner_dict, log_dir=str(tmp_path / "run"), device="cpu")

    runner.learn(num_learning_iterations=3)

    assert offline_wandb.run is None  # rsl_rl's learn finished the run, so every upload is done
    assert len(rsl_rl_env.model_checkpoints(str(tmp_path / "run"))) == 1


def test_a_later_checkpoint_left_in_the_log_dir_never_costs_the_run_its_newest(wrapper, tmp_path, eden_caplog):
    """A reused log dir (or a rollback resume) holds a later iteration.

    The run's own checkpoints are ranked without it, so the one just written survives; the later one is kept but
    reported once.
    """
    (tmp_path / "model_5000.pt").touch()

    assert _saved_checkpoints(wrapper, tmp_path) == ["model_2.pt", "model_5000.pt"]
    reported = [r for r in eden_caplog.records if r.levelno == logging.WARNING and "model_5000.pt" in r.getMessage()]
    assert len(reported) == 1


def test_a_resume_from_an_earlier_checkpoint_prunes_only_below_it(wrapper, tmp_path, eden_caplog):
    for name in ("model_1.pt", "model_9.pt"):
        (tmp_path / name).touch()
    runner = MimicOnPolicyRunner(wrapper, wrapper.runner_dict, log_dir=str(tmp_path), device="cpu")

    runner.save(str(tmp_path / "model_5.pt"))

    assert [name for _, name in rsl_rl_env.model_checkpoints(str(tmp_path))] == ["model_5.pt", "model_9.pt"]
    assert "model_9.pt" in eden_caplog.text


def test_a_save_outside_the_model_checkpoint_naming_prunes_nothing(wrapper, tmp_path):
    """An export into the run directory (``runner.save("export.pt")``) is not one of the run's checkpoints."""
    runner = MimicOnPolicyRunner(
        wrapper, {**wrapper.runner_dict, "keep_last_checkpoints": 1}, log_dir=str(tmp_path), device="cpu"
    )
    runner.save(str(tmp_path / "model_1.pt"))
    runner.save(str(tmp_path / "model_2.pt"))
    assert sorted(p.name for p in tmp_path.glob("model_*.pt")) == ["model_2.pt"]

    runner.save(str(tmp_path / "export.pt"))

    assert sorted(p.name for p in tmp_path.glob("*.pt")) == ["export.pt", "model_2.pt"]


# ---------------------------------------------------------------------------
# Checkpoint loading: the current layout, the legacy combined one, and provenance
# ---------------------------------------------------------------------------


def _perturbed_actor_weights(runner) -> dict[str, torch.Tensor]:
    """Snapshot the actor's weights, then scramble them in place so a load is visible."""
    state = {k: v.clone() for k, v in runner.alg.actor.state_dict().items()}
    with torch.no_grad():
        for param in runner.alg.actor.parameters():
            param.add_(1.0)
    return state


def _assert_actor_is(runner, state):
    for key, value in runner.alg.actor.state_dict().items():
        torch.testing.assert_close(value, state[key], msg=key)


def _write_legacy(runner, path, *, std_key: str = "std") -> torch.Tensor:
    """Rewrite ``runner``'s weights into the pre-split combined layout; return the raw std written."""
    actor, critic = runner.alg.actor.state_dict(), runner.alg.critic.state_dict()
    std = torch.tensor([0.5])
    combined = {f"actor.{k[4:]}": v for k, v in actor.items() if k.startswith("mlp.")}
    combined |= {f"actor_obs_normalizer.{k[15:]}": v for k, v in actor.items() if k.startswith("obs_normalizer.")}
    combined |= {f"critic.{k[4:]}": v for k, v in critic.items() if k.startswith("mlp.")}
    combined |= {f"critic_obs_normalizer.{k[15:]}": v for k, v in critic.items() if k.startswith("obs_normalizer.")}
    combined[std_key] = std if std_key == "std" else torch.log(std)
    torch.save(
        {"model_state_dict": combined, "optimizer_state_dict": runner.alg.optimizer.state_dict(), "iter": 7}, path
    )
    return std


def _write_run_json(directory, version: str, distribution: str = "rsl-rl-lib") -> None:
    rl_library = {"backend": "rsl_rl", "distribution": distribution, "version": version}
    (directory / "run.json").write_text(json.dumps({"schema_version": 2, "rl_library": rl_library}))


def _warnings(caplog, needle: str) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING and needle in r.getMessage()]


def test_each_runner_gets_its_own_runner_dict(wrapper, runner, tmp_path):
    """rsl-rl-lib 3.x to 5.4 pop ``class_name`` out of the dicts a runner is given (5.4's PPO already has, by now);
    neither a second runner nor a dict read off the wrapper must see that.
    """
    runner.cfg["algorithm"].pop("class_name", None)
    runner.cfg["actor"].pop("class_name", None)
    shallow = {**wrapper.runner_dict, "save_interval": 1}  # what a test or a script passes on: nested entries shared
    shallow["algorithm"].pop("class_name")
    second = make_rsl_rl_runner(wrapper, log_dir=str(tmp_path / "second"))  # raised KeyError: 'class_name' on 5.4
    assert type(second.alg).__name__ == "PPO"
    assert wrapper.runner_dict["algorithm"]["class_name"] == "PPO"


def test_a_checkpoint_the_runner_saved_loads_back(runner, tmp_path):
    path = str(tmp_path / "model_1.pt")
    runner.save(path)
    saved = _perturbed_actor_weights(runner)

    load_runner_checkpoint(runner, path)

    _assert_actor_is(runner, saved)


@pytest.fixture
def distiller(wrapper):
    """A real distillation runner (``student`` / ``teacher``) on the same task."""
    from eden.options.learning.rsl_rl import (
        RslRlCriticOptions,
        RslRlDistillationAlgorithmOptions,
        RslRlDistillationRunnerOptions,
    )

    options = RslRlDistillationRunnerOptions(
        device="cpu",
        num_steps_per_env=4,
        logger="tensorboard",
        check_for_nan=False,
        obs_groups={"student": ["policy"], "teacher": ["policy"]},
        student=rsl_rl_runner_options().actor,
        teacher=RslRlCriticOptions(hidden_dims=[8]),
        algorithm=RslRlDistillationAlgorithmOptions(),
    )
    return make_rsl_rl_runner(RslRlVecEnvWrapper(wrapper.unwrapped, options), log_dir=None)


def test_both_split_algorithms_take_the_split_layout_path(runner, distiller):
    """The probe once looked for ``alg.actor`` only, routing every distillation runner past conversion and audit."""
    assert _runner_uses_split_policy(runner) and _runner_uses_split_policy(distiller)


def test_a_distillation_runner_converts_a_legacy_checkpoint_into_its_teacher(runner, distiller, tmp_path, eden_caplog):
    """Bootstrapping a teacher from an RL run's legacy checkpoint: the student's std_type drives the conversion."""
    path = str(tmp_path / "rl_teacher.pt")
    _write_legacy(runner, path)
    teacher_weights = {k: v for k, v in runner.alg.actor.state_dict().items() if not k.startswith("distribution.")}
    student_before = {k: v.clone() for k, v in distiller.alg.student.state_dict().items()}

    load_runner_checkpoint(distiller, path)

    assert os.path.exists(tmp_path / "rl_teacher.log.converted.pt")
    assert distiller.alg.teacher_loaded
    teacher = distiller.alg.teacher.state_dict()
    assert teacher.keys() == teacher_weights.keys()
    for key, value in teacher_weights.items():
        torch.testing.assert_close(teacher[key], value, msg=key)
    for key, value in distiller.alg.student.state_dict().items():  # an RL checkpoint loads only the teacher
        torch.testing.assert_close(value, student_before[key], msg=key)
    # The RL actor's action noise has nowhere to go on a deterministic teacher; that is no drift.
    assert [r.getMessage() for r in eden_caplog.records if r.levelno >= logging.WARNING] == []


def _write_legacy_distillation(distiller, path) -> None:
    """Rewrite ``distiller``'s weights into rsl-rl-lib 3.x's ``StudentTeacher`` layout: one ``model_state_dict``."""
    combined = {"std": torch.tensor([0.5])}
    for head in ("student", "teacher"):
        for k, v in getattr(distiller.alg, head).state_dict().items():
            if k.startswith("mlp."):
                combined[f"{head}.{k[4:]}"] = v
            elif k.startswith("obs_normalizer."):
                combined[f"{head}_obs_normalizer.{k[15:]}"] = v
    torch.save({"model_state_dict": combined, "optimizer_state_dict": {}, "iter": 7}, path)


def test_a_legacy_distillation_checkpoint_is_refused_before_anything_is_converted_or_loaded(distiller, tmp_path):
    """Converting it once dropped every ``student.*`` / ``teacher.*`` key and set ``teacher_loaded`` on an untrained teacher."""
    path = str(tmp_path / "student_3x.pt")
    _write_legacy_distillation(distiller, path)
    teacher_before = {k: v.clone() for k, v in distiller.alg.teacher.state_dict().items()}

    with pytest.raises(RuntimeError) as excinfo:
        load_runner_checkpoint(distiller, path)

    message = str(excinfo.value)
    assert message.startswith(f"Legacy checkpoint {path} is an rsl-rl-lib 3.x distillation checkpoint")
    assert "student.* (" in message and "teacher_obs_normalizer.* (" in message
    assert "Nothing was converted or loaded" in message and "replays under rsl-rl-lib 3.x" in message
    assert not list(tmp_path.glob("*.converted.pt"))
    assert not distiller.alg.teacher_loaded
    for key, value in distiller.alg.teacher.state_dict().items():
        torch.testing.assert_close(value, teacher_before[key], msg=key)


def test_a_legacy_checkpoint_with_keys_the_converter_cannot_map_is_refused(runner, tmp_path):
    """A recurrent 3.x policy's ``memory_a.*`` was once dropped with a warning, loading a partly random policy."""
    path = str(tmp_path / "recurrent.pt")
    _write_legacy(runner, path)
    state = torch.load(path, weights_only=False)
    state["model_state_dict"]["memory_a.rnn.weight_ih_l0"] = torch.zeros(4, 2)
    torch.save(state, path)

    with pytest.raises(RuntimeError) as excinfo:
        load_runner_checkpoint(runner, path)

    message = str(excinfo.value)
    assert message.startswith(f"Legacy checkpoint {path} has 'model_state_dict' keys the legacy converter")
    assert "memory_a.* (1 key)." in message and "actor_obs_normalizer.*" in message
    assert "not an rsl-rl-lib 4.x/5.x version to install" in message
    assert not list(tmp_path.glob("*.converted.pt"))


def test_a_cached_conversion_does_not_bypass_the_refusal(runner, tmp_path):
    """An older converter dropped ``memory_a.*`` and cached the result; reusing that cache must not skip the check."""
    path = str(tmp_path / "recurrent.pt")
    _write_legacy(runner, path)
    state = torch.load(path, weights_only=False)
    state["model_state_dict"]["memory_a.rnn.weight_ih_l0"] = torch.zeros(4, 2)
    torch.save(state, path)
    # What that converter left beside the source: a loadable split-layout file, newer than it.
    stale = str(tmp_path / f"recurrent.{_detect_std_type(runner.cfg)}.converted.pt")
    runner.save(stale)
    os.utime(stale, (os.path.getmtime(path) + 10,) * 2)

    with pytest.raises(RuntimeError, match=r"keys the legacy converter does not recognise: memory_a\.\* \(1 key\)"):
        load_runner_checkpoint(runner, path)


def test_a_distillation_checkpoint_reloads_with_its_student_and_teacher_audited(distiller, tmp_path, eden_caplog):
    """The audit once checked ``actor``/``critic`` only: three false warnings, and the real heads unaudited."""
    path = str(tmp_path / "model_3.pt")
    distiller.save(path)
    saved = torch.load(path, weights_only=False)
    del saved["student_state_dict"]["mlp.0.bias"]
    torch.save(saved, path)
    with torch.no_grad():
        for param in distiller.alg.teacher.parameters():
            param.add_(1.0)

    load_runner_checkpoint(distiller, path)

    (student,) = _warnings(eden_caplog, "checkpoint key audit")
    assert student.startswith("[rsl_rl] student checkpoint key audit") and "mlp.0.bias" in student
    assert not _warnings(eden_caplog, "Checkpoint has no")  # no head the runner lacks, and none it has
    for key, value in distiller.alg.teacher.state_dict().items():
        torch.testing.assert_close(value, saved["teacher_state_dict"][key], msg=key)


def test_a_legacy_checkpoint_is_converted_beside_the_original_and_loads(runner, tmp_path):
    """Pre-split Eden runs saved one ``model_state_dict``; the converted copy is what rsl_rl loads."""
    path = str(tmp_path / "ckpt.pt")
    expected = {k: v.clone() for k, v in runner.alg.actor.state_dict().items()}
    std = _write_legacy(runner, path)
    original = open(path, "rb").read()
    _perturbed_actor_weights(runner)

    load_runner_checkpoint(runner, path)

    assert os.path.exists(tmp_path / "ckpt.log.converted.pt")  # the task's std_type, in the name
    assert open(path, "rb").read() == original  # never modified in place
    expected["distribution.log_std_param"] = torch.log(std)  # raw std -> log for a 'log' distribution
    _assert_actor_is(runner, expected)


@pytest.mark.parametrize(
    ("std_key", "std_type", "param"),
    [
        ("std", "scalar", "distribution.std_param"),
        ("std", "log", "distribution.log_std_param"),
        ("log_std", "scalar", "distribution.std_param"),
        ("log_std", "log", "distribution.log_std_param"),
    ],
)
def test_legacy_conversion_writes_raw_std_to_whichever_parameter_the_destination_learns(
    runner, tmp_path, std_key, std_type, param
):
    """Raw std in a ``log_std_param`` samples NaN on the first action; the cache is keyed per std_type."""
    path = str(tmp_path / "ckpt.pt")
    std = _write_legacy(runner, path, std_key=std_key)

    converted = torch.load(_maybe_convert_legacy_checkpoint(path, std_type=std_type), weights_only=False)

    assert set(converted) >= {"actor_state_dict", "critic_state_dict"} and "model_state_dict" not in converted
    actor = converted["actor_state_dict"]
    assert {k for k in actor if k.startswith("distribution.")} == {param}
    torch.testing.assert_close(actor[param], std if std_type == "scalar" else torch.log(std))
    assert "mlp.0.weight" in actor and "mlp.0.weight" in converted["critic_state_dict"]


def test_legacy_conversion_is_reused_until_the_original_changes(runner, tmp_path):
    path = str(tmp_path / "ckpt.pt")
    _write_legacy(runner, path)
    first = _maybe_convert_legacy_checkpoint(path, std_type="scalar")
    written = os.path.getmtime(first)

    older = os.path.getmtime(path) - 60
    os.utime(path, (older, older))
    assert _maybe_convert_legacy_checkpoint(path, std_type="scalar") == first
    assert os.path.getmtime(first) == written  # reused, not rewritten

    newer = written + 3600  # a re-save of the original invalidates the copy
    os.utime(path, (newer, newer))
    assert _maybe_convert_legacy_checkpoint(path, std_type="scalar") == first
    assert os.path.getmtime(first) > written


def test_legacy_conversion_passes_a_current_checkpoint_through_and_refuses_an_unknown_std_type(runner, tmp_path):
    path = str(tmp_path / "model.pt")
    runner.save(path)

    assert _maybe_convert_legacy_checkpoint(path, std_type="scalar") == path
    assert not list(tmp_path.glob("*.converted.pt"))
    with pytest.raises(ValueError, match="std_type must be 'scalar' or 'log'"):
        _maybe_convert_legacy_checkpoint(path, std_type="bogus")


@pytest.mark.parametrize(
    ("runner_dict", "std_type"),
    [
        ({"actor": {"distribution_cfg": {"std_type": "scalar"}}}, "scalar"),  # rsl-rl 5
        ({"actor": {"noise_std_type": "log"}}, "log"),  # rsl-rl 4: distribution_cfg flattened
        ({"policy": {"noise_std_type": "log"}}, "log"),  # rsl-rl 3
        ({"student": {"distribution_cfg": {"std_type": "log"}}, "teacher": {}}, "log"),  # distillation
        ({"student": {"noise_std_type": "log"}, "teacher": {}}, "log"),  # distillation, rsl-rl 4
        # A legacy ``std`` lands in the teacher, so a teacher with a distribution decides.
        ({"student": {"noise_std_type": "log"}, "teacher": {"noise_std_type": "scalar"}}, "scalar"),
        ({}, None),
        ({"actor": "not-a-dict"}, None),
        ({"actor": {}}, None),
    ],
)
def test_detect_std_type_reads_every_translated_layout(runner_dict, std_type):
    assert _detect_std_type(runner_dict) == std_type


def test_a_runner_dict_without_std_type_refuses_a_legacy_checkpoint_and_passes_a_current_one(
    wrapper, tmp_path, eden_caplog
):
    """Guessing mis-maps ``std`` silently; skipping once surfaced as a version mismatch naming the wrong fix."""
    runner_dict = json.loads(json.dumps(wrapper.runner_dict))
    del runner_dict["actor"]["distribution_cfg"]["std_type"]
    runner = MimicOnPolicyRunner(wrapper, runner_dict, log_dir=None, device="cpu")
    legacy, current = str(tmp_path / "legacy.pt"), str(tmp_path / "current.pt")
    _write_legacy(runner, legacy)
    runner.save(current)

    with pytest.raises(RuntimeError) as excinfo:
        load_runner_checkpoint(runner, legacy)
    message = str(excinfo.value)
    assert message.startswith(f"Checkpoint {legacy} is in the pre-split legacy layout")
    assert "none of the heads ['actor', 'teacher', 'student']" in message
    assert "Heads present: ['actor']" in message and "critic, device" in message
    assert "not an rsl-rl-lib version mismatch" in message
    assert not list(tmp_path.glob("*.converted.pt"))

    load_runner_checkpoint(runner, current)  # nothing to convert, so nothing to say
    assert [r.getMessage() for r in eden_caplog.records if r.levelno >= logging.WARNING] == []


def test_the_key_audit_names_what_strict_false_would_silently_skip(runner, tmp_path, eden_caplog):
    """``runner.load(strict=False)`` leaves a missing key at random init; the audit makes it visible."""
    path = str(tmp_path / "model.pt")
    runner.save(path)
    saved = torch.load(path, weights_only=False)
    saved["actor_state_dict"]["stale_buffer"] = torch.zeros(3)
    del saved["actor_state_dict"]["mlp.0.bias"]
    torch.save(saved, path)

    load_runner_checkpoint(runner, path)

    (actor,) = _warnings(eden_caplog, "actor checkpoint key audit")
    assert "missing in ckpt: 1, unexpected in ckpt: 1" in actor
    assert "mlp.0.bias" in actor and "stale_buffer" in actor
    assert not _warnings(eden_caplog, "critic checkpoint key audit")
    assert _audit_rsl_rl_checkpoint_keys(runner, path, "cpu") == sorted(saved)  # names only, never the weights


def test_a_checkpoint_missing_a_top_level_key_names_the_file_both_versions_and_its_keys(runner, tmp_path, eden_caplog):
    """rsl_rl's bare ``KeyError: 'actor_state_dict'`` names nothing; the replacement names everything."""
    path = tmp_path / "model_15000.pt"
    runner.save(str(path))
    saved = torch.load(path, weights_only=False)
    del saved["actor_state_dict"]
    torch.save(saved, path)
    _write_run_json(tmp_path, "4.0.0")

    with pytest.raises(RuntimeError) as excinfo:
        load_runner_checkpoint(runner, str(path))

    message = str(excinfo.value)
    assert message.startswith(f"Checkpoint {path} was written by rsl-rl-lib 4.0.0 and has top-level keys [")
    assert f"(version {INSTALLED}) expects 'actor_state_dict'" in message
    assert "critic_state_dict" in message
    assert isinstance(excinfo.value.__cause__, KeyError)
    assert _warnings(eden_caplog, "no 'actor_state_dict'")  # the audit warned before the load failed


def test_the_format_error_leads_with_the_callers_path_when_a_converted_copy_was_loaded(runner, tmp_path):
    """The keys reported are the copy's, so which file they came from must be unambiguous."""
    source = str(tmp_path / "model_500.pt")
    _write_legacy(runner, source)
    loaded = _maybe_convert_legacy_checkpoint(source, std_type="log")

    error = _rsl_rl_checkpoint_format_error(KeyError("student_state_dict"), source=source, loaded=loaded, keys=None)

    assert str(error).startswith(f"Checkpoint {source} (loaded via the converted copy at {loaded}) records no ")
    assert "actor_state_dict" in str(error)


def test_a_keyerror_for_a_key_the_checkpoint_has_is_not_dressed_up(runner, tmp_path):
    """Only a missing *top-level* key is a format mismatch; a deeper KeyError is someone else's bug."""
    path = str(tmp_path / "model.pt")
    runner.save(path)

    assert _rsl_rl_checkpoint_format_error(KeyError("actor_state_dict"), source=path, loaded=path, keys=None) is None


def test_the_pre_load_inspections_map_the_checkpoint_and_the_error_path_reads_nothing(runner, tmp_path, monkeypatch):
    """Eden reads key names by mapping (``mmap=True``); only rsl_rl's own load deserializes the weights.

    A full read in the format probe or the key audit would take the weights off disk twice more per load,
    and the error builder reuses the audit's key names rather than reading the file again.
    """
    path = tmp_path / "model_1.pt"
    runner.save(str(path))
    saved = torch.load(path, weights_only=False)
    del saved["actor_state_dict"]
    torch.save(saved, path)
    reads: list[bool] = []
    real_load = torch.load

    def recording_load(file, *args, **kwargs):
        if str(file) == str(path):
            reads.append(bool(kwargs.get("mmap")))
        return real_load(file, *args, **kwargs)

    monkeypatch.setattr(torch, "load", recording_load)

    with pytest.raises(RuntimeError, match="expects 'actor_state_dict'"):
        load_runner_checkpoint(runner, str(path))

    assert reads and reads[:-1] == [True] * (len(reads) - 1), reads  # Eden's inspections: mapped
    assert reads[-1] is False  # rsl_rl's own load, and nothing after it


def test_a_recorded_rsl_rl_version_that_differs_from_the_installed_one_warns_once(runner, tmp_path, eden_caplog):
    path = str(tmp_path / "model_100.pt")
    runner.save(path)
    _write_run_json(tmp_path, "4.0.0")
    warn_once.cache_clear()

    load_runner_checkpoint(runner, path)
    load_runner_checkpoint(runner, path)

    (warning,) = _warnings(eden_caplog, "rsl-rl-lib version mismatch")
    assert f"trained with 4.0.0, {INSTALLED} is installed" in warning


@pytest.mark.parametrize(
    "run_json",
    [{"version": INSTALLED}, None, {"version": "1.6.1", "distribution": "rl-games"}],
    ids=["same-version", "no-run-json", "rl-games-run"],
)
def test_no_drift_warning_without_a_differing_rsl_rl_version_on_record(runner, tmp_path, eden_caplog, run_json):
    path = str(tmp_path / "model_100.pt")
    runner.save(path)
    if run_json is not None:
        _write_run_json(tmp_path, **run_json)
    warn_once.cache_clear()

    load_runner_checkpoint(runner, path)

    assert not _warnings(eden_caplog, "version mismatch")


def test_a_mistyped_checkpoint_path_gets_no_provenance_claim(runner, tmp_path, eden_caplog):
    """``run.json`` describes the run's *other* checkpoints; "trained with X, loading anyway" about a file
    that does not exist points the reader at a version problem when the problem is the path.
    """
    runner.save(str(tmp_path / "model_100.pt"))
    _write_run_json(tmp_path, "4.0.0")
    warn_once.cache_clear()

    with pytest.raises(FileNotFoundError):
        load_runner_checkpoint(runner, str(tmp_path / "model_9999.pt"))
    assert not _warnings(eden_caplog, "version mismatch")


class _Rsl3Runner:
    """An rsl-rl-lib 3.x runner, which the installed 5.x cannot provide: one ``alg.policy`` module, and a
    ``load`` with no ``strict`` that reads ``model_state_dict`` by key. Records what it was handed.
    """

    def __init__(self):
        self.alg = type("Alg", (), {"policy": torch.nn.Linear(1, 1)})()
        self.cfg: dict = {}
        self.device = "cpu"
        self.loaded: list[str] = []

    def load(self, path, load_optimizer=True, map_location=None):
        self.loaded.append(path)
        return torch.load(path, map_location=map_location, weights_only=False)["model_state_dict"]


def test_a_single_module_runner_loads_its_native_checkpoint_untouched(tmp_path):
    path = str(tmp_path / "native.pt")
    torch.save({"model_state_dict": {"policy.weight": torch.zeros(1)}}, path)
    runner = _Rsl3Runner()

    load_runner_checkpoint(runner, path)

    assert runner.loaded == [path]
    assert not list(tmp_path.glob("*.converted.pt"))


def test_a_split_checkpoint_under_a_single_module_runner_names_both_versions(runner, tmp_path):
    """The headline case: a checkpoint written by 5.x, replayed under 3.x."""
    path = tmp_path / "model_15000.pt"
    runner.save(str(path))
    _write_run_json(tmp_path, INSTALLED)

    with pytest.raises(RuntimeError, match=rf"was written by rsl-rl-lib {INSTALLED}.*expects 'model_state_dict'"):
        load_runner_checkpoint(_Rsl3Runner(), str(path))


# ---------------------------------------------------------------------------
# MotionWeightCurriculumMixin: the per-iteration motion-weight curriculum
#
# The env and its motion manager are real; rsl-rl's ``learn`` loop and algorithm are a small stand-in
# for the third-party training loop (a real runner needs actions and observations this mimic env
# does not declare), and ``wandb`` is a recording fake of the service.
# ---------------------------------------------------------------------------


class _Alg:
    def __init__(self):
        self.updates = 0

    def update(self):
        self.updates += 1
        return {"loss": 0.0}


class _Loop:
    """rsl-rl's ``learn``: one ``alg.update`` per iteration."""

    def learn(self, num_iters, fail=False):
        if fail:
            raise RuntimeError("boom")
        for _ in range(num_iters):
            self.alg.update()


class _Runner(MotionWeightCurriculumMixin, _Loop):
    def __init__(self, mimic_env):
        self.env = mimic_env
        self.alg = _Alg()
        self.current_learning_iteration = 0


class _Wandb:
    """The two ``wandb`` attributes the mixin touches: ``run`` and ``log``."""

    class Histogram:
        def __init__(self, sequence):
            self.sequence = sequence

    def __init__(self, active=True):
        self.run = object() if active else None
        self.logged = []

    def log(self, data, *, step=None):
        self.logged.append((dict(data), step))


@pytest.fixture(scope="module")
def mixin_library(tmp_path_factory):
    return write_library(tmp_path_factory.mktemp("mixin"), [50, 50])


@pytest.fixture(scope="module")
def mimic_env(mixin_library):
    motion = manager_options(terms={"default": term_options(mixin_library, adaptive_motion_weights=True)})
    return build_mimic_env(mixin_library, motion_options=motion)


def _weights(mimic_env):
    return mimic_env.motion_manager.get_term("default")._motion_weights.clone()


def _record_a_failure_on(mimic_env, motion: int):
    term = mimic_env.motion_manager.get_term("default")
    pin_clips(term, [motion] * 4)
    term._curriculum.record_motion_outcomes(
        torch.tensor([], dtype=torch.long), torch.tensor([0]), torch.zeros(0), torch.zeros(1), 0.02
    )


def test_every_iteration_drains_the_curriculum_and_logs_to_wandb(mimic_env, monkeypatch):
    wandb = _Wandb()
    monkeypatch.setattr(rsl_rl_env, "_wandb", wandb)
    runner = _Runner(mimic_env)
    _record_a_failure_on(mimic_env, 1)

    runner.learn(num_iters=3)

    assert runner.alg.updates == 3
    assert _weights(mimic_env).tolist() == [0.0, 1.0]  # the drained failure put the weight on motion 1
    assert [step for _, step in wandb.logged] == [0, 1, 2]
    assert all("MotionWeights/entropy" in data for data, _ in wandb.logged)
    # The hook is detached once learn() returns: a direct update drains nothing.
    runner.alg.update()
    assert len(wandb.logged) == 3


def test_the_hook_is_detached_after_learn_raises(mimic_env, monkeypatch):
    wandb = _Wandb()
    monkeypatch.setattr(rsl_rl_env, "_wandb", wandb)
    runner = _Runner(mimic_env)

    with pytest.raises(RuntimeError, match="boom"):
        runner.learn(num_iters=1, fail=True)
    runner.alg.update()

    assert wandb.logged == []


@pytest.mark.parametrize("wandb", [None, _Wandb(active=False)], ids=["not-installed", "no-active-run"])
def test_without_a_wandb_run_the_curriculum_still_drains(mimic_env, monkeypatch, wandb):
    monkeypatch.setattr(rsl_rl_env, "_wandb", wandb)
    before = _weights(mimic_env)
    _record_a_failure_on(mimic_env, int(before.argmin()))

    _Runner(mimic_env).learn(num_iters=1)

    assert not torch.equal(_weights(mimic_env), before)
    assert wandb is None or wandb.logged == []


def test_a_manager_without_adaptive_terms_is_left_alone(mixin_library, monkeypatch):
    wandb = _Wandb()
    monkeypatch.setattr(rsl_rl_env, "_wandb", wandb)
    mimic_env = build_mimic_env(mixin_library)
    runner = _Runner(mimic_env)
    original = runner.alg.update

    runner.learn(num_iters=2)

    assert runner.alg.update == original and wandb.logged == []


@pytest.mark.parametrize(
    "runner_cls", [MimicOnPolicyRunner, MimicDistillationRunner, DDPOnPolicyRunner, DDPDistillationRunner]
)
def test_every_on_policy_runner_steps_the_curriculum(runner_cls):
    """It is not in ``EDEN_RUNNER_MIXINS`` (an off-policy iteration is not a PPO rollout), so each of these adds it,
    ahead of rsl_rl's runner: placed after it, rsl_rl's ``learn`` would never call the mixin's.
    """
    from rsl_rl.runners import DistillationRunner, OnPolicyRunner

    mro = runner_cls.__mro__
    base = DistillationRunner if issubclass(runner_cls, DistillationRunner) else OnPolicyRunner
    assert MotionWeightCurriculumMixin in mro and mro.index(MotionWeightCurriculumMixin) < mro.index(base)


# ---------------------------------------------------------------------------
# The agent-theme iteration log
# ---------------------------------------------------------------------------


def _rsl_rl_iteration_block(it: int = 2, total: int = 3000) -> str:
    """What ``rsl_rl.utils.logger.Logger.log`` prints for one iteration, in its own layout."""
    pad, width = 40, 80
    block = f"{'#' * width}\n"
    block += f"\033[1m{f' Learning iteration {it}/{total} '.center(width)}\033[0m \n\n"
    block += f"{'Total steps:':>{pad}} 4608 \n"
    block += f"{'Steps per second:':>{pad}} 3536 \n"
    block += f"{'Mean value loss:':>{pad}} 5.3846\n"
    block += f"{'Mean reward:':>{pad}} -19.79\n"
    block += f"{'Episode_Reward/track_linear_velocity:':>{pad}} 0.0138\n"
    block += f"{'Episode_Reward_Raw/track_linear_velocity:':>{pad}} 0.0069\n"
    block += f"{'Episode_Reward/upright:':>{pad}} 0.0314\n"
    block += f"{'Episode_Termination/fell_over:':>{pad}} 2.4762\n"
    block += f"{'-' * width}\n"
    block += f"{'Time elapsed:':>{pad}} 00:00:07\n"
    block += f"{'ETA:':>{pad}} 00:00:00\n"
    return block + "\n"  # rsl_rl's print() adds one more


@pytest.fixture
def log_theme():
    """``log_theme(name)`` rethemes ``eden.logger`` through its public setter; the theme and variable come back after."""
    import eden as en
    from eden._logging import LOG_THEME_ENV_VAR

    previous_theme, previous_env = en.logger.theme, os.environ.get(LOG_THEME_ENV_VAR)
    try:
        yield en.logger.set_theme
    finally:
        en.logger.set_theme(previous_theme)
        if previous_env is None:
            os.environ.pop(LOG_THEME_ENV_VAR, None)
        else:
            os.environ[LOG_THEME_ENV_VAR] = previous_env


def test_compact_iteration_log_keeps_every_number_and_drops_the_layout():
    """One scalar line, one line per ``/`` group, no padding, rules or ANSI, and no number lost."""
    assert compact_iteration_log(_rsl_rl_iteration_block()) == (
        "[it 2/3000] total_steps=4608 steps_per_second=3536 value_loss=5.3846 reward=-19.79"
        " time_elapsed=00:00:07 eta=00:00:00\n"
        "  Episode_Reward: track_linear_velocity=0.0138 upright=0.0314\n"
        "  Episode_Reward_Raw: track_linear_velocity=0.0069\n"
        "  Episode_Termination: fell_over=2.4762\n"
    )


def test_compact_iteration_log_passes_foreign_lines_and_foreign_layouts_through():
    """A wandb notice follows the block verbatim; text without the iteration header comes back unchanged."""
    out = compact_iteration_log(_rsl_rl_iteration_block() + "wandb: Syncing run foo\n")
    assert out.endswith("\nwandb: Syncing run foo\n"), out

    foreign = "Iteration 2 of 3000\n   reward = -19.79\n"
    assert compact_iteration_log(foreign) == foreign


def test_compact_iteration_log_quotes_a_value_holding_whitespace():
    """Each ``key=value`` stays one field: rsl_rl 5 prints a timedelta (``1 day, 1:01:01``) past a day."""
    pad = 40
    block = f"{' Learning iteration 2/3000 '.center(80)}\n"
    block += f"{'Run name:':>{pad}} " + 'my "best" run\n'
    block += f"{'Time elapsed:':>{pad}} 1 day, 1:01:01\n"
    block += f"{'ETA:':>{pad}} 0:00:07\n"
    block += f"{'Episode_Reward/upright:':>{pad}} 1 2\n"

    assert compact_iteration_log(block) == (
        '[it 2/3000] run_name="my \\"best\\" run" time_elapsed="1 day, 1:01:01" eta=0:00:07\n'
        '  Episode_Reward: upright="1 2"\n'
    )


@pytest.mark.parametrize("theme", ["agent", "human"])
def test_a_training_iteration_prints_compact_only_under_the_agent_theme(runner, capsys, log_theme, theme):
    """Eden's runner rewrites rsl_rl's per-iteration block under ``agent`` and leaves it alone under ``human``.

    Either way ``logger.log`` is restored once ``learn`` returns.
    """
    log_theme(theme)
    runner.learn(num_learning_iterations=1)

    out = capsys.readouterr().out
    if theme == "agent":
        assert "[it " in out and "Learning iteration" not in out and "\033[" not in out, out
    else:
        assert "Learning iteration" in out
    assert runner.logger.log.__name__ == "log"


def test_the_capture_leaves_another_threads_output_alone(capsys, log_theme):
    """Only the thread running ``Logger.log`` is captured; another thread's print reaches the console verbatim.

    rsl_rl's loop is stood in for (``_Loop``/``_IterationLogger``): a real runner cannot print from a
    second thread in the middle of its own ``log`` call, which is the case under test.
    """
    import threading

    class _IterationLogger:
        def log(self):
            print(_rsl_rl_iteration_block(), end="")
            other = threading.Thread(target=print, args=("from the viewer thread: 12345678901234567890123456789: x",))
            other.start()
            other.join()

    class _Loop:
        def learn(self):
            self.logger.log()

    class _Runner(AgentIterationLogMixin, _Loop):
        def __init__(self):
            self.logger = _IterationLogger()

    log_theme("agent")
    _Runner().learn()

    assert capsys.readouterr().out == (
        "from the viewer thread: 12345678901234567890123456789: x\n" + compact_iteration_log(_rsl_rl_iteration_block())
    )
