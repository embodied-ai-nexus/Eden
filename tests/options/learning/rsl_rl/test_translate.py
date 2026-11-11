"""``eden.options.learning.rsl_rl.translate``: the v5 runner dict rewritten for rsl-rl-lib 3.x and 4.x."""

from __future__ import annotations

import copy

import pytest

from eden.options.learning.rsl_rl.translate import translate_runner_dict_to_v3, translate_runner_dict_to_v4


def _v5_runner_dict() -> dict:
    return {
        "seed": 7100,
        "device": "cuda:0",
        "num_steps_per_env": 24,
        "max_iterations": 100,
        "obs_groups": {"actor": ["policy"], "critic": ["policy"]},
        "clip_actions": None,
        "save_interval": 50,
        "check_for_nan": True,
        "experiment_name": "test",
        "run_name": "",
        "logger": "tensorboard",
        "class_name": "OnPolicyRunner",
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [256, 128],
            "activation": "elu",
            "obs_normalization": True,
            "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "log"},
        },
        "critic": {"class_name": "MLPModel", "hidden_dims": [512, 256], "activation": "elu", "obs_normalization": True},
        "algorithm": {
            "class_name": "PPO",
            "learning_rate": 1e-3,
            "share_cnn_encoders": False,
            "rnd_cfg": None,
            "symmetry_cfg": None,
        },
    }


def test_v3_merges_actor_and_critic_into_one_policy_with_noise_params():
    out = translate_runner_dict_to_v3(_v5_runner_dict())

    assert "actor" not in out and "critic" not in out
    policy = out["policy"]
    assert policy["class_name"] == "ActorCritic"
    assert (policy["actor_hidden_dims"], policy["critic_hidden_dims"]) == ([256, 128], [512, 256])
    assert (policy["init_noise_std"], policy["noise_std_type"]) == (1.0, "log")
    assert policy["actor_obs_normalization"] is policy["critic_obs_normalization"] is True
    assert "policy" in out["obs_groups"] and "actor" not in out["obs_groups"]
    assert "empirical_normalization" in out
    assert "check_for_nan" not in out and "share_cnn_encoders" not in out["algorithm"]


@pytest.mark.parametrize(
    ("actor_class", "extra", "expected_class", "expected_extra"),
    [
        (
            "RNNModel",
            {"rnn_type": "lstm", "rnn_hidden_dim": 128, "rnn_num_layers": 2},
            "ActorCriticRecurrent",
            {"rnn_type": "lstm", "rnn_hidden_dim": 128, "rnn_num_layers": 2},
        ),
        # v3's ActorCriticCNN has no `cnn_cfg`: forwarded bare it lands in **kwargs and the model has no CNN.
        (
            "CNNModel",
            {"cnn_cfg": {"output_channels": [16]}},
            "ActorCriticCNN",
            {"actor_cnn_cfg": {"output_channels": [16]}, "critic_cnn_cfg": {"output_channels": [16]}},
        ),
    ],
    ids=["rnn", "cnn"],
)
def test_v3_picks_the_actor_critic_class_for_the_model_kind(actor_class, extra, expected_class, expected_extra):
    runner = _v5_runner_dict()
    runner["actor"] |= {"class_name": actor_class, **extra}
    runner["critic"] |= extra

    policy = translate_runner_dict_to_v3(runner)["policy"]

    assert policy["class_name"] == expected_class
    assert {k: policy[k] for k in expected_extra} == expected_extra
    assert "cnn_cfg" not in policy


def test_v4_flattens_the_distribution_into_the_actor():
    out = translate_runner_dict_to_v4(_v5_runner_dict())

    assert "distribution_cfg" not in out["actor"]
    assert (out["actor"]["stochastic"], out["actor"]["init_noise_std"], out["actor"]["noise_std_type"]) == (
        True,
        1.0,
        "log",
    )
    assert out["actor"]["state_dependent_std"] is False
    assert "check_for_nan" not in out


def test_v4_maps_a_heteroscedastic_distribution_to_state_dependent_std_and_strips_a_critic_one():
    runner = _v5_runner_dict()
    runner["actor"]["distribution_cfg"] = {"class_name": "HeteroscedasticGaussianDistribution", "std_type": "scalar"}
    runner["critic"]["distribution_cfg"] = {"class_name": "GaussianDistribution"}

    out = translate_runner_dict_to_v4(runner)

    assert out["actor"]["state_dependent_std"] is True and out["actor"]["stochastic"] is True
    assert "distribution_cfg" not in out["critic"]


def _v5_distillation_runner_dict() -> dict:
    runner = _v5_runner_dict()
    actor, critic = runner.pop("actor"), runner.pop("critic")
    return runner | {
        "class_name": "DistillationRunner",
        "obs_groups": {"student": ["policy"], "teacher": ["privileged"]},
        "student": actor,
        "teacher": critic,
        "algorithm": {"class_name": "Distillation", "learning_rate": 1e-3, "loss_type": "mse"},
    }


def test_v4_flattens_a_distillation_runners_student_and_teacher():
    """rsl-rl 4.x's DistillationRunner splats each model dict into ``MLPModel``, which has no ``distribution_cfg``."""
    out = translate_runner_dict_to_v4(_v5_distillation_runner_dict())

    assert out["student"] == {
        "class_name": "MLPModel",
        "hidden_dims": [256, 128],
        "activation": "elu",
        "obs_normalization": True,
        "stochastic": True,
        "init_noise_std": 1.0,
        "noise_std_type": "log",
        "state_dependent_std": False,
    }
    assert out["teacher"] == {
        "class_name": "MLPModel",
        "hidden_dims": [512, 256],
        "activation": "elu",
        "obs_normalization": True,
        "stochastic": False,
    }
    assert "actor" not in out and "critic" not in out  # a distillation dict gains no PPO heads
    assert "check_for_nan" not in out


@pytest.mark.parametrize(
    ("translate", "model"), [(translate_runner_dict_to_v3, "policy"), (translate_runner_dict_to_v4, "actor")]
)
def test_no_distribution_means_no_noise_params(translate, model):
    runner = _v5_runner_dict()
    runner["actor"]["distribution_cfg"] = None

    out = translate(runner)[model]

    assert "init_noise_std" not in out and "noise_std_type" not in out
    if model == "actor":
        assert out["stochastic"] is False


@pytest.mark.parametrize("translate", [translate_runner_dict_to_v3, translate_runner_dict_to_v4])
def test_translation_leaves_its_input_alone(translate):
    runner = _v5_runner_dict()
    original = copy.deepcopy(runner)

    translate(runner)

    assert runner == original


@pytest.mark.parametrize("translate", [translate_runner_dict_to_v3, translate_runner_dict_to_v4])
def test_the_runner_level_keep_last_checkpoints_survives_translation(translate):
    """Eden's checkpoint retention reads it off the runner's ``cfg`` whatever rsl_rl major built it."""
    assert translate(_v5_runner_dict() | {"keep_last_checkpoints": 1})["keep_last_checkpoints"] == 1


def test_an_already_translated_dict_passes_through():
    v3 = {"policy": {"class_name": "ActorCritic"}, "algorithm": {"class_name": "PPO"}}
    v4 = _v5_runner_dict() | {"actor": {"class_name": "MLPModel", "stochastic": True, "init_noise_std": 1.0}}

    assert translate_runner_dict_to_v3(v3)["policy"]["class_name"] == "ActorCritic"
    assert translate_runner_dict_to_v4(v4)["actor"] == {
        "class_name": "MLPModel",
        "stochastic": True,
        "init_noise_std": 1.0,
    }


def test_a_module_needing_a_newer_rsl_rl_names_the_upgrade_not_a_missing_name(monkeypatch):
    from eden.options.learning.rsl_rl import translate

    monkeypatch.setattr(translate, "detect_rsl_rl_version", lambda: (5, 4, 2))
    with pytest.raises(
        ImportError,
        match=r"^my\.module needs rsl-rl-lib >= 5\.5\.1 \(resolve_class\); the installed version is 5\.4\.2\. "
        r"Upgrade with `uv pip install 'rsl-rl-lib>=5\.5\.1'`\.$",
    ):
        translate.require_rsl_rl_version((5, 5, 1), "my.module", "resolve_class")
    translate.require_rsl_rl_version((5, 4), "my.module", "resolve_class")  # new enough: no error


def test_the_installed_rsl_rl_version_is_a_comparable_release_tuple():
    from importlib.metadata import version

    from eden.options.learning.rsl_rl.translate import detect_rsl_rl_major_version, detect_rsl_rl_version

    release = detect_rsl_rl_version()
    assert release[0] == detect_rsl_rl_major_version()
    assert ".".join(map(str, release)) == version("rsl-rl-lib").split("+")[0]
