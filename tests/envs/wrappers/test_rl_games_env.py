"""``eden.envs.wrappers.rl_games_env``: construction guards, run seeding, checkpoint retention and the
multi-agent (``agents``) layout.

Seeding and retention run on the ``tests/envs`` pendulum task; the multi-agent tests on two pendulums, ``left``
and ``right``, each one agent with an observation group, a reward group and an action term of its name.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("rl_games", reason="the rl-games extra is not installed")

from pydantic import ValidationError  # noqa: E402

from eden.envs.wrappers.rl_games_env import RlGamesVecEnvWrapper, rl_games_checkpoints  # noqa: E402
from eden.envs.wrappers.rsl_rl_env import RslRlVecEnvWrapper  # noqa: E402
from eden.options.learning.rl_games import RlGamesNetworkOptions, RlGamesPpoRunnerOptions  # noqa: E402
from eden.utils.configs import EdenConfig, EdenRLConfig  # noqa: E402
from tests.envs.conftest import EnvsTraceTermination, pendulum_options  # noqa: E402
from tests.envs.wrappers.conftest import rsl_rl_runner_options  # noqa: E402


def _draw_after_from_config(wrapper_cls, config, **kwargs) -> float:
    wrapper_cls.from_config(config, **kwargs)
    return torch.rand(1).item()


def test_a_config_without_runner_options_is_pointed_at_rlenvbase():
    """``__init__`` reads ``options.clip_actions`` at once, which used to die on ``None`` naming nothing."""
    with pytest.raises(ValueError, match="requires `runner_options`"):
        RlGamesVecEnvWrapper(env=None, options=None)
    for config in (EdenRLConfig(), EdenConfig()):
        with pytest.raises(ValueError, match=r"no `runner_options`.*RLEnvBase\.from_config"):
            RlGamesVecEnvWrapper.from_config(config)


def test_from_config_seeds_before_building_like_the_rsl_rl_wrapper(rl_config, monkeypatch):
    """rl_games seeds in ``Runner.load``, after the env is built and reset, so the wrapper seeds first.

    The rank offset matches: at ``RANK=1`` seed 77 must seed exactly as the rsl_rl wrapper does with 78.
    """
    reference = _draw_after_from_config(RslRlVecEnvWrapper, rl_config(rsl_rl_runner_options(seed=78)))
    monkeypatch.setenv("RANK", "1")
    config = rl_config(RlGamesPpoRunnerOptions(seed=77, device="cpu"))

    assert _draw_after_from_config(RlGamesVecEnvWrapper, config) == reference
    assert config.runner_options.seed == 77


def test_apply_seed_false_leaves_the_process_rng_alone(rl_config):
    draws = []
    for seed in (77, 999):
        torch.manual_seed(0)
        config = rl_config(RlGamesPpoRunnerOptions(seed=seed, device="cpu"))
        draws.append(_draw_after_from_config(RlGamesVecEnvWrapper, config, apply_seed=False))

    assert draws[0] == draws[1]


# ----------------------------------------------------------------------------------------------
# Multi-agent: `RlGamesPpoRunnerOptions.agents`
# ----------------------------------------------------------------------------------------------

AGENTS = ["left", "right"]
#: The reward weight of each agent's `action_l2` group, distinct so a swapped reward shows.
REWARD_WEIGHTS = {"left": 1.0, "right": 2.0}


def _two_pendulum_config(pendulum_file: str):
    """Two pendulums; ``left``'s observation group is concatenated, ``right``'s a per-term dict of the same width."""
    from eden import observations, rewards, terminations
    from eden.managers.terms.actions.joint_actions import ImplicitPDController
    from eden.options import (
        ActionManagerOptions,
        EnvOptions,
        ObservationGroupOptions,
        ObservationManagerOptions,
        PlaneOptions,
        RewardGroupOptions,
        RewardManagerOptions,
        SceneOptions,
        TerminationManagerOptions,
    )
    from eden.terms import DoneTerm, RewardTerm

    def pendulum(y: float, hinge: float):
        return pendulum_options(pendulum_file).model_copy(
            update={"default_root_pos": (0.0, y, 0.0), "default_dofs_pos": {"hinge": hinge}}
        )

    def group(name: str, concatenate: bool):
        return ObservationGroupOptions(
            concatenate_terms=concatenate,
            dofs_pos=observations.DofsPos.configure(entity_name=name),
            dofs_vel=observations.DofsVel.configure(entity_name=name),
        )

    return EdenRLConfig(
        env_options=EnvOptions(num_envs=3, decimation=2, episode_length_s=100.0),
        scene_options=SceneOptions(plane=PlaneOptions(), left=pendulum(-0.5, 0.3), right=pendulum(0.5, -0.6)),
        action_options=ActionManagerOptions(
            **{name: ImplicitPDController.configure(entity_name=name, dofs_name=["hinge"]) for name in AGENTS}
        ),
        observation_options=ObservationManagerOptions(left=group("left", True), right=group("right", False)),
        reward_options=RewardManagerOptions(
            **{
                name: RewardGroupOptions(
                    **{
                        f"{name}_action": RewardTerm.configure(
                            func=rewards.action_l2, weight=REWARD_WEIGHTS[name], params={"action_term_name": name}
                        )
                    }
                )
                for name in AGENTS
            }
        ),
        termination_options=TerminationManagerOptions(
            ends=EnvsTraceTermination.configure(),
            timeout=DoneTerm.configure(func=terminations.time_out, time_out=True),
        ),
        runner_options=RlGamesPpoRunnerOptions(agents=AGENTS, device="cpu"),
    )


@pytest.fixture(scope="module")
def agents_wrapper(pendulum_file):
    """A built two-pendulum env behind a two-agent wrapper, shared by the module."""
    return RlGamesVecEnvWrapper.from_config(_two_pendulum_config(pendulum_file))


def _group_obs(obs_dict: dict, group: str, n: int) -> torch.Tensor:
    value = obs_dict[group]
    if isinstance(value, dict):
        return torch.cat([t.reshape(n, -1) for t in value.values()], dim=-1)
    return value.reshape(n, -1)


def _assert_interleaved(obs: torch.Tensor, env) -> None:
    n = env.num_envs
    assert obs.shape == (n * len(AGENTS), 2)
    for e in range(n):
        for i, agent in enumerate(AGENTS):
            torch.testing.assert_close(obs[e * len(AGENTS) + i], _group_obs(env.obs_buf, agent, n)[e])


@pytest.mark.requires_genesis
def test_the_spaces_and_env_info_are_per_agent(agents_wrapper):
    info = agents_wrapper.get_env_info()

    assert info["agents"] == agents_wrapper.get_number_of_agents() == 2
    assert info["observation_space"].shape == (2,)  # dofs_pos + dofs_vel of one pendulum
    assert info["action_space"].shape == (1,) and agents_wrapper.num_actions == 1
    assert "state_space" not in info


@pytest.mark.requires_genesis
def test_reset_interleaves_the_agents_env_major(agents_wrapper):
    """Row ``e * A + i`` is agent ``i`` of env ``e``; the two pendulums start at distinct hinge angles."""
    obs = agents_wrapper.reset()

    _assert_interleaved(obs, agents_wrapper.unwrapped)
    assert not torch.equal(obs[0], obs[1])


@pytest.mark.requires_genesis
def test_step_routes_each_row_to_its_agent_and_pays_its_group_reward(agents_wrapper):
    env = agents_wrapper.unwrapped
    n, a = env.num_envs, len(AGENTS)
    agents_wrapper.reset()
    # Row e*A+i carries 0.1 * (10 e + i + 1): every (env, agent) pair gets a distinct action.
    actions = 0.1 * torch.arange(1, n * a + 1, dtype=torch.float).reshape(n, a).add(
        torch.arange(n).unsqueeze(1) * (10 - a)
    ).reshape(n * a, 1)

    obs, rew, dones, extras = agents_wrapper.step(actions)

    for i, agent in enumerate(AGENTS):
        own = actions[i::a, 0]
        torch.testing.assert_close(env.action_manager.get_term(agent).action[:, 0], own)
        torch.testing.assert_close(rew[i::a], env.reward_manager.get_group_reward(agent))
        torch.testing.assert_close(rew[i::a], REWARD_WEIGHTS[agent] * own.square() * agents_wrapper.dt)
    assert rew.shape == dones.shape == extras["time_outs"].shape == (n * a,)
    _assert_interleaved(obs, env)


@pytest.mark.requires_genesis
def test_dones_and_time_outs_repeat_for_each_agent_of_an_env(agents_wrapper):
    env = agents_wrapper.unwrapped
    ends = env.termination_manager.get_term("ends")
    agents_wrapper.reset()
    zeros = torch.zeros(env.num_envs * len(AGENTS), 1)

    ends.ending[:] = torch.tensor([False, True, False])
    _, _, dones, extras = agents_wrapper.step(zeros)
    assert dones.tolist() == [0, 0, 1, 1, 0, 0]
    assert extras["time_outs"].tolist() == [False] * 6

    ends.ending[:] = False
    env.episode_length_buf[2] = env.max_episode_length - 1
    _, _, dones, extras = agents_wrapper.step(zeros)
    assert dones.tolist() == [0, 0, 0, 0, 1, 1]
    assert extras["time_outs"].tolist() == [False, False, False, False, True, True]


@pytest.mark.requires_genesis
def test_an_env_that_times_out_and_terminates_in_one_step_is_not_a_time_out(agents_wrapper):
    """rl_games bootstraps the value of every ``time_outs`` row; a terminated env has none to bootstrap."""
    env = agents_wrapper.unwrapped
    ends = env.termination_manager.get_term("ends")
    agents_wrapper.reset()
    zeros = torch.zeros(env.num_envs * len(AGENTS), 1)

    ends.ending[:] = torch.tensor([False, False, True])
    env.episode_length_buf[1:] = env.max_episode_length - 1
    _, _, dones, extras = agents_wrapper.step(zeros)
    ends.ending[:] = False
    assert dones.tolist() == [0, 0, 1, 1, 1, 1]
    assert extras["time_outs"].tolist() == [False, False, True, True, False, False]

    single = RlGamesVecEnvWrapper(agents_wrapper.env, RlGamesPpoRunnerOptions(device="cpu"))
    ends.ending[:] = torch.tensor([False, False, True])
    env.episode_length_buf[1:] = env.max_episode_length - 1
    _, _, dones, extras = single.step(torch.zeros(env.num_envs, 2))
    ends.ending[:] = False
    assert dones.tolist() == [0, 1, 1]
    assert extras["time_outs"].tolist() == [False, True, False]


@pytest.mark.requires_genesis
def test_a_reward_group_no_agent_owns_is_refused_at_construction(pendulum_file):
    from eden import rewards
    from eden.options import RewardGroupOptions
    from eden.terms import RewardTerm

    config = _two_pendulum_config(pendulum_file)
    config.reward_options.shared = RewardGroupOptions(
        alive=RewardTerm.configure(func=rewards.action_l2, weight=1.0, params={"action_term_name": "left"})
    )
    with pytest.raises(ValueError, match=r"leaves the reward groups \['shared'\] to no agent"):
        RlGamesVecEnvWrapper.from_config(config)


@pytest.mark.requires_genesis
def test_without_agents_the_same_env_is_one_agent(agents_wrapper):
    wrapper = RlGamesVecEnvWrapper(agents_wrapper.env, RlGamesPpoRunnerOptions(device="cpu"))

    assert wrapper.get_env_info()["agents"] == wrapper.get_number_of_agents() == 1
    assert wrapper.num_actions == 2
    assert wrapper.reset().shape == (wrapper.num_envs, 2)  # the first observation group, `left`


@pytest.mark.requires_genesis
@pytest.mark.parametrize(
    ("agents", "match"),
    [
        (
            ["left", "middle"],
            r"(?s)names 'middle', which is not an observation group.*observation groups: \['left', 'right'\]",
        ),
        (
            ["right", "left"],
            r"does not match the action terms \['left', 'right'\]: the terms are declared in a different order",
        ),
        (["left"], r"the action terms \['right'\] belong to no agent"),
    ],
)
def test_agents_the_env_cannot_back_are_refused_at_construction(agents_wrapper, agents, match):
    with pytest.raises(ValueError, match=match):
        RlGamesVecEnvWrapper(agents_wrapper.env, RlGamesPpoRunnerOptions(agents=agents, device="cpu"))


def test_agents_and_obs_groups_are_mutually_exclusive():
    with pytest.raises(ValidationError, match="mutually exclusive"):
        RlGamesPpoRunnerOptions(agents=AGENTS, obs_groups={"actor": ["left"]})
    with pytest.raises(ValidationError, match=r"names \['left'\] more than once"):
        RlGamesPpoRunnerOptions(agents=["left", "left"])


def test_the_wrapper_rechecks_agents_assigned_past_the_validator():
    options = RlGamesPpoRunnerOptions(agents=AGENTS)
    options.obs_groups = {"actor": ["left"]}  # assignment does not re-run the model validator

    with pytest.raises(ValueError, match="mutually exclusive"):
        RlGamesVecEnvWrapper(env=None, options=options)


def test_the_full_batch_default_counts_every_agent():
    from eden.options.learning.rl_games import translate_runner_dict

    options = RlGamesPpoRunnerOptions(agents=AGENTS, horizon_length=4, device="cpu")
    config = translate_runner_dict(options.dict(), num_envs=3)["params"]["config"]

    assert config["minibatch_size"] == 4 * 3 * 2
    assert "agents" not in config


def _runner(rl_config, *, agent_builder=None, **overrides):
    """A loaded rl_games ``Runner`` for 3 epochs of the pendulum, saving every epoch, with episodes short enough to end.

    rl_games' periodic save only fires once an episode has finished. ``agent_builder``, when given, is registered as
    ``a2c_continuous`` before ``configure_runner``, as a caller with its own agent would.
    """
    from rl_games.torch_runner import Runner

    options = RlGamesPpoRunnerOptions(
        **{
            "device": "cpu",
            "experiment_name": "pendulum",
            "max_epochs": 3,
            "save_frequency": 1,
            "horizon_length": 4,
            "mini_epochs": 1,
            "network": RlGamesNetworkOptions(units=[8]),
            **overrides,
        }
    )
    config = rl_config(options)
    config.env_options.episode_length_s = 0.05
    env = RlGamesVecEnvWrapper.from_config(config)
    runner = Runner()
    if agent_builder is not None:
        runner.algo_factory.register_builder("a2c_continuous", agent_builder)
    runner.load(env.runner_dict)
    env.configure_runner(runner)
    return runner


def _trained_checkpoints(rl_config, run_root, monkeypatch, **overrides) -> list[tuple[int, str]]:
    """Train 3 epochs in ``run_root`` and list the periodic checkpoints left, as ``(epoch, filename)``."""
    run_root.mkdir()
    monkeypatch.chdir(run_root)  # rl_games writes into ``./runs/<name>_<timestamp>/nn``
    _runner(rl_config, **overrides).run({"train": True, "checkpoint": ""})
    (nn_dir,) = run_root.glob("runs/*/nn")
    return rl_games_checkpoints(str(nn_dir), "pendulum")


def test_keep_last_checkpoints_leaves_only_the_newest_on_disk(rl_config, tmp_path, monkeypatch):
    """At ``max_epochs`` rl_games saves twice at one epoch (periodic, then final as ``_rew__<reward>_``): the final
    one supersedes the other, so ``keep`` counts epochs and the last save always survives.
    """
    every = _trained_checkpoints(rl_config, tmp_path / "all", monkeypatch, keep_last_checkpoints=None)
    assert [epoch for epoch, _ in every] == [2, 3, 3]  # no episode has ended by epoch 1, so it saves nothing

    two = _trained_checkpoints(rl_config, tmp_path / "two", monkeypatch, keep_last_checkpoints=2)
    assert [epoch for epoch, _ in two] == [2, 3] and "_rew__" in two[-1][1]

    newest = _trained_checkpoints(rl_config, tmp_path / "default", monkeypatch)
    assert [epoch for epoch, _ in newest] == [3] and "_rew__" in newest[0][1]


def _agent(runner):
    return runner.algo_factory.create(runner.algo_name, base_name="run", params=runner.params)


def test_a_save_outside_the_periodic_naming_prunes_nothing(rl_config, tmp_path, monkeypatch):
    """The best-reward checkpoint ``<name>.pth`` (and any export) is not one of the periodic checkpoints."""
    monkeypatch.chdir(tmp_path)
    agent = _agent(_runner(rl_config, keep_last_checkpoints=1))
    nn_dir = tmp_path / agent.nn_dir

    for epoch in (1, 2):
        agent.save(str(nn_dir / f"last_pendulum_ep_{epoch}_rew_0.5"))
    agent.save(str(nn_dir / "pendulum"))

    assert sorted(p.name for p in nn_dir.glob("*.pth")) == ["last_pendulum_ep_2_rew_0.5.pth", "pendulum.pth"]


def test_a_later_epoch_left_in_the_directory_never_costs_the_save_its_file(rl_config, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    agent = _agent(_runner(rl_config, keep_last_checkpoints=1))
    nn_dir = tmp_path / agent.nn_dir
    (nn_dir / "last_pendulum_ep_1_rew_0.5.pth").touch()
    (nn_dir / "last_pendulum_ep_9_rew_0.5.pth").touch()

    agent.save(str(nn_dir / "last_pendulum_ep_5_rew_0.5"))

    assert [epoch for epoch, _ in rl_games_checkpoints(str(nn_dir), "pendulum")] == [5, 9]


def test_an_agent_a_caller_registered_is_kept_and_still_prunes(rl_config, tmp_path, monkeypatch):
    """``configure_runner`` wraps the registered ``a2c_continuous`` builder instead of replacing it."""
    from rl_games.algos_torch.a2c_continuous import A2CAgent

    class CustomAgent(A2CAgent):
        pass

    monkeypatch.chdir(tmp_path)
    agent = _agent(_runner(rl_config, agent_builder=lambda **kwargs: CustomAgent(**kwargs)))
    assert isinstance(agent, CustomAgent)

    nn_dir = tmp_path / agent.nn_dir
    for epoch in (1, 2):
        agent.save(str(nn_dir / f"last_pendulum_ep_{epoch}_rew_0.5"))

    assert [epoch for epoch, _ in rl_games_checkpoints(str(nn_dir), "pendulum")] == [2]
