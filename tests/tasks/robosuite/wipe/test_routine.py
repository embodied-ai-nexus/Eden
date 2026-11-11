"""The wiping routine: where it aims, and a real run that wipes every marker safely on the robosuite table."""

from __future__ import annotations

# isort: off
import eden  # noqa: F401 - isolate the DDP rank before importing Torch
# isort: on

import numpy as np

from eden.extensions.scripted_demos.registry import SCRIPTED_DEMO_REGISTRY
from eden.tasks import TASK_REGISTRY
from eden.tasks.robosuite.wipe.routine import AIM_JITTER, _aim
from tests.support.envs import build_env

#: robosuite's ``pressure_threshold_max``: above it the task pays its excess-force penalty.
EXCESS_FORCE = 60.0


def test_the_pad_aims_within_the_jitter_of_a_marker_at_the_asked_height():
    rng = np.random.default_rng(0)
    marker = np.array([0.1, -0.2, 0.901])
    for _ in range(100):
        x, y, z = _aim(rng, marker, 0.95)
        assert max(abs(x - 0.1), abs(y + 0.2)) <= AIM_JITTER
        assert z == 0.95


def test_the_routine_wipes_every_marker_without_an_unsafe_step_or_excess_force():
    config = TASK_REGISTRY.build("robosuite_wipe")
    config.env_options.num_envs = 2
    config.env_options.episode_length_s = 60.0
    # Every marker wiped ends the task's episode (and resets the markers); read the result first.
    del config.termination_options.success
    env = build_env(config)
    env.reset(seed=0)
    demo = SCRIPTED_DEMO_REGISTRY.get("robosuite_wipe")(env, seed=0, p_mistake=0.0)
    demo.reset()
    metrics = env.metric_manager
    peak_force = np.zeros(2)
    for _ in range(900):
        if bool(demo.update_subgoals().all()):
            break
        env.step(demo.act())
        assert not env.reset_buf.any(), "an arm link on the table or a joint at its limit ended the episode"
        peak_force = np.maximum(peak_force, metrics.get_value("gripper_force").cpu().numpy())

    assert demo.done.all()
    assert (metrics.get_value("markers_wiped") == 1.0).all()
    assert (peak_force > 1.0).all(), "the pad never pressed on the table"
    assert (peak_force < EXCESS_FORCE).all(), peak_force
