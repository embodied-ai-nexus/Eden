"""The supermimic task config and its data flags."""

import pytest

from eden.tasks import TASK_REGISTRY


@pytest.mark.parametrize("flag", ["raw_data_path", "processed_data_path", "clip_list"])
def test_an_empty_data_flag_raises(flag):
    with pytest.raises(ValueError, match=f"--{flag} was given an empty path"):
        TASK_REGISTRY.build("supermimic", **{flag: ""})


@pytest.mark.parametrize(
    ("dataset", "name"), [("lafan1", "lafan1_root_joints_g1_dataset"), ("bones_seed", "seed_root_joints_g1_dataset")]
)
def test_the_dataset_flag_selects_a_root_and_joints_library_at_the_control_rate(dataset, name):
    cfg = TASK_REGISTRY.build("supermimic", dataset=dataset, clip_list="clips.txt")
    options = cfg.motion_options.default.dataset_options
    assert options.name == name
    assert options.include_list_file == "clips.txt"
    assert options.target_framerate == pytest.approx(50.0)


def _update_rounds_per_env_step(cfg) -> float:
    runner, world_model = cfg.runner_options, cfg.runner_options.world_model
    windows = world_model.num_batches * world_model.batch_size * world_model.horizon
    return cfg.env_options.num_envs / windows * runner.replay_ratio


def test_the_task_trains_with_the_learners_default_options():
    from eden.learning.wrappers.rsl_rl.supermimic import SuperMimicRunnerOptions

    assert TASK_REGISTRY.build("supermimic").runner_options == SuperMimicRunnerOptions(experiment_name="supermimic")


def test_the_defaults_are_blinkmimics_recipe_at_the_papers_update_rate():
    cfg = TASK_REGISTRY.build("supermimic")
    runner = cfg.runner_options
    world_model, policy, sampler = runner.world_model, runner.policy, runner.return_sampler
    assert (world_model.hidden_dims, world_model.num_batches, world_model.lr) == ((1024, 1024, 1024), 5, 4e-4)
    assert world_model.mirror_share == 0.5
    assert (policy.batch_size, policy.lr, policy.lr_multiplier) == (2048, 4e-4, 4.0)
    assert (policy.lr_decay_return, policy.lr_decay_deadline, policy.lr_decay_iterations) == (0.75, 350, 75)
    assert (policy.ema_decay, policy.reference_action) == (0.995, True)
    assert _update_rounds_per_env_step(cfg) == pytest.approx(0.4)
    # 30% of the resets before falls; explore, focus and keep split the rest 0.5 / 0.4 / 0.1, as in the paper.
    focus = 1.0 - sampler.explore - sampler.keep - sampler.failure
    assert sampler.failure == 0.3
    assert (sampler.explore / 0.7, focus / 0.7, sampler.keep / 0.7) == pytest.approx((0.5, 0.4, 0.1))
