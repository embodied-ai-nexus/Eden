"""Interactive reward plots on a real rollout."""

from contextlib import closing

import numpy as np
import pytest

pytest.importorskip("viser", minversion="1.1.0")

from eden.extensions.visualization.reward_viewer import RewardViewer
from eden.options import BoxOptions, EnvOptions, RewardManagerOptions, RewardTermOptions, SceneOptions
from eden.options.managers.terminations import TerminationManagerOptions, TerminationTermOptions
from eden.utils.configs import EdenConfig, EdenRLConfig
from tests.support.envs import build_env


def test_plots_weights_impulses_and_episode_boundaries():
    config = EdenRLConfig(
        env_options=EnvOptions(num_envs=2, sim_dt=0.01, decimation=2, episode_length_s=0.06),
        scene_options=SceneOptions(box=BoxOptions(size=(0.1, 0.1, 0.1), default_root_pos=(0, 0, 1))),
        reward_options=RewardManagerOptions(
            height=RewardTermOptions(
                name="base_height_l2", weight=-1, params={"entity_name": "box", "target_height": 0}
            )
        ),
        termination_options=TerminationManagerOptions(timeout=TerminationTermOptions(name="time_out", time_out=True)),
    )
    env = build_env(config, reset=True, eval_mode=True)
    try:
        with closing(RewardViewer(env, port=0)) as viewer:
            viewer.update()
            viewer.playing.value = True
            viewer.env_index.value = 1
            box = env.entities["box"]
            viewer._impulses.put((1, box.links[0].idx, (0.1, 0, 0)))
            term = env.reward_manager.get_term("height")
            _, weight, set_weight = viewer._weights[0]
            assert set_weight.disabled
            # Viser sends every keystroke: the partial entries neither change the weight nor get overwritten, and
            # the enabled button marks the edit as not yet applied.
            for typed in ("", "-", "-0", "-2"):
                weight.value = typed
                assert term.weight == -1.0
                assert not set_weight.disabled
            assert weight.value == "-2"
            set_weight.value = True
            assert (weight.value, set_weight.disabled) == ("-2.0", True)
            _, reward, *_ = viewer.step(None)
            assert reward[1].item() == pytest.approx(-2 * box.get_pos()[1, 2].item() ** 2 * env.dt, rel=1e-5)
            assert box.get_vel()[0, 0].item() == pytest.approx(0)
            mass = box.get_links_inertial_mass()[1, 0].item()
            assert box.get_vel()[1, 0].item() == pytest.approx(0.1 / mass, rel=1e-4)
            assert term.weight == -2.0
            folder, plot = viewer._plots["height"]
            assert folder.label == f"height  {plot.data[1][-1]:.4g}"
            np.testing.assert_allclose(plot.data[1], [reward[1].item() / env.dt])
            viewer.step(None)
            np.testing.assert_allclose(plot.data[0], [env.dt, 2 * env.dt])
            viewer.env_index.value = 0
            viewer.step(None)
            np.testing.assert_allclose(plot.data[0], [3 * env.dt])
            viewer.step(None)
            np.testing.assert_allclose(plot.data[0], [env.dt])
            weight.value = "heavy"
            set_weight.value = True
            assert term.weight == -2.0
            assert (weight.value, set_weight.disabled) == ("-2.0", True)
    finally:
        env.destroy()


def test_an_env_without_rewards_is_refused():
    env = build_env(
        EdenConfig(env_options=EnvOptions(num_envs=1), scene_options=SceneOptions(box=BoxOptions(size=(0.1, 0.1, 0.1))))
    )
    try:
        with pytest.raises(TypeError, match=r"need an RL env with a reward manager, but got EnvBase"):
            RewardViewer(env, port=0)
    finally:
        env.destroy()


def test_an_env_whose_reward_terms_all_have_weight_zero_is_refused():
    config = EdenRLConfig(
        env_options=EnvOptions(num_envs=1),
        scene_options=SceneOptions(box=BoxOptions(size=(0.1, 0.1, 0.1))),
        reward_options=RewardManagerOptions(
            height=RewardTermOptions(name="base_height_l2", weight=0.0, params={"entity_name": "box"})
        ),
    )
    env = build_env(config)
    try:
        with pytest.raises(
            ValueError, match=r"but EdenRLConfig builds none: a reward term of weight 0\.0 is not built"
        ):
            RewardViewer(env, port=0)
    finally:
        env.destroy()
