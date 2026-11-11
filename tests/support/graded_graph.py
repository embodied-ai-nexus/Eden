"""Drive ``EnvOptions.cuda_graph_options`` on CUDA and report what it did, for ``tests/envs/test_graded_graph.py``.

Run as ``python -m tests.support.graded_graph SCENARIO [TASK]`` from the repo root; prints one line,
``RESULT {...}``. A process of its own because the test suite initialises Genesis on ``gs.cpu``, where a CUDA
graph cannot exist, and Genesis cannot be re-initialised in-process. Scenarios:

* ``task TASK`` — step a registered task in graph mode, four envs timing out every step so the reset path runs
  between replays, then double a reward term's weight (as a curriculum would) and step on. Reports how many
  captures ran (a weight is read from the device, so it needs none) and whether the doubled weight reached
  the replayed reward. Every capture, and the replays after it, are checked against eager runs by the graph
  itself, so finishing without an error is the equivalence result.
* ``python_state`` — a box env whose reward reads a Python counter it advances in ``compute()``. Reports the
  error a validated replay raises.
* ``host_read`` — a box env whose reward reads a value back with ``.item()``. Reports the capture error.
* ``field_change`` — a box env whose reward has a configured ``scale``, changed once, long after the capture.
* ``rebind`` — a box env whose reward reads a buffer of its own, rebound to a new tensor long after the
  capture (as a ``load_state_dict`` might).
* ``rebind_holder`` — a box env whose reward reads a buffer from a dict of its own, the whole dict replaced long
  after the capture: the buffer's own binding (in the old dict) never changes.
* ``env_constant`` — the ``field_change`` reward, with ``env.max_episode_length`` changed long after the capture.
* ``term_replaced`` — the ``field_change`` reward, swapped in its manager for a copy of itself long after the
  capture: the graph would neither replay the new term nor watch its fields.
* ``one_off`` — the ``field_change`` reward with its ``scale`` changed once, on the step right after the capture.
* ``storm`` — the ``field_change`` reward with its ``scale`` changed on every step after the capture.
* ``counter_read`` — a box env whose reward branches on ``env.common_step_counter``. Reports the capture error.
* ``side_stream`` — a box env whose reward reads a buffer it writes on a CUDA stream of its own, as a kernel
  launched outside torch's stream would: a capture does not record that write. Reports the validation error.
* ``hidden_state`` — a box env whose reward counts its calls in a tensor no Eden attribute reaches, allocated
  lazily inside ``compute()`` (so under ``RewardManager.compute``'s inference mode). Reports the count after
  12 steps: each validation has to put that tensor back as well, or the count runs ahead of the steps.
* ``fresh_cache`` — a box env whose reward pads a fresh tensor in place and leaves it in a per-step cache, as
  Genesis' contact query does. Reports the steps after which the cache held anything but the box height.
* ``contacts`` — a box resting on a plane, whose reward sums the contact forces over the contact axis. Reports
  the contact-axis widths the reward saw (the warm-up, the eager runs of each validation and the capture all
  read the padded form) and any error: a float sum over a trimmed axis in the eager runs and a padded one in
  the graph can differ in its last bits.
"""

from __future__ import annotations

import argparse
import json

#: The envs forced to time out in every step of the ``task`` scenario.
RESET_ENVS = slice(0, 4)

#: The ``fresh_cache`` reward's per-step cache, emptied before every step as Genesis empties its query cache.
FRESH_CACHE: dict = {}


def _box_env(reward_name: str, num_envs: int = 16, on_a_plane: bool = False):
    from eden.envs.base import env_class_for_config
    from eden.options import BoxOptions, CudaGraphOptions, EnvOptions, PlaneOptions, SceneOptions
    from eden.options.managers.rewards import RewardManagerOptions, RewardTermOptions
    from eden.utils.configs import EdenRLConfig

    if on_a_plane:  # resting on it, so every step has contacts
        scene = SceneOptions(
            plane=PlaneOptions(), box=BoxOptions(size=(0.1, 0.1, 0.1), default_root_pos=(0.0, 0.0, 0.05))
        )
    else:
        scene = SceneOptions(box=BoxOptions(size=(0.1, 0.1, 0.1), default_root_pos=(0.0, 0.0, 0.5)))
    config = EdenRLConfig(
        env_options=EnvOptions(num_envs=num_envs, cuda_graph_options=CudaGraphOptions()),
        scene_options=scene,
        reward_options=RewardManagerOptions(probe=RewardTermOptions(name=reward_name)),
    )
    env = env_class_for_config(config).from_config(config, show_viewer=False)
    env.build()
    env.reset()
    return env


def _register_probe_terms() -> None:
    from types import SimpleNamespace

    import torch

    from eden.managers.reward_manager import REWARD_TERM_REGISTRY, RewardTerm

    @REWARD_TERM_REGISTRY.register()
    class PythonCounterReward(RewardTerm):
        """Breaks the term contract: its value depends on a Python int it advances in ``compute()``."""

        def build(self) -> None:
            self._calls = 0

        def compute(self, envs_idx=None) -> torch.Tensor:
            self._calls += 1
            return torch.full((self.num_envs,), float(self._calls), device=self.device)

    @REWARD_TERM_REGISTRY.register()
    def host_read_reward(env) -> torch.Tensor:
        height = env.entities["box"].get_pos()[:, 2]
        return height * height.max().item()

    @REWARD_TERM_REGISTRY.register()
    class ScaledHeightReward(RewardTerm):
        """A configured field a capture bakes in as a constant."""

        scale: float = 1.0

        def compute(self, envs_idx=None) -> torch.Tensor:
            return self._env.entities["box"].get_pos()[:, 2] * self.scale

    @REWARD_TERM_REGISTRY.register()
    class OffsetHeightReward(RewardTerm):
        """A buffer of its own the graph reads where it lives."""

        def build(self) -> None:
            self._offset = torch.zeros(self.num_envs, device=self.device)

        def compute(self, envs_idx=None) -> torch.Tensor:
            return self._env.entities["box"].get_pos()[:, 2] + self._offset

    @REWARD_TERM_REGISTRY.register()
    class HeldOffsetReward(RewardTerm):
        """A buffer the graph reads, held in a dict of its own."""

        def build(self) -> None:
            self._held = {"offset": torch.zeros(self.num_envs, device=self.device)}

        def compute(self, envs_idx=None) -> torch.Tensor:
            return self._env.entities["box"].get_pos()[:, 2] + self._held["offset"]

    @REWARD_TERM_REGISTRY.register()
    def counter_gated_reward(env) -> torch.Tensor:
        height = env.entities["box"].get_pos()[:, 2]
        return 2.0 * height if env.common_step_counter % 5 == 0 else height

    @REWARD_TERM_REGISTRY.register()
    class SideStreamReward(RewardTerm):
        """Reads a buffer it writes as a foreign kernel would (Warp, Quadrants): on a stream of its own, unseen by
        torch's dispatch. A capture runs that write once instead of recording it, and a validation cannot undo it.
        """

        def build(self) -> None:
            self._steps = torch.zeros(self.num_envs, device=self.device)
            self._side = torch.cuda.Stream(device=self.device)

        def compute(self, envs_idx=None) -> torch.Tensor:
            from torch.utils._python_dispatch import _disable_current_modes

            # Under a capture the side-stream copy runs at once, unrecorded, and no cross-stream sync is legal
            # (torch.cuda.graph synchronized the device on entry, so the step count is already written; the
            # replay launches after the capture ends, long after this 16-element copy).
            capturing = torch.cuda.is_current_stream_capturing()
            if not capturing:
                self._side.wait_stream(torch.cuda.current_stream(self.device))
            with _disable_current_modes(), torch.cuda.stream(self._side):
                self._steps.copy_(self._env.episode_length_buf)
            if not capturing:
                torch.cuda.current_stream(self.device).wait_stream(self._side)
            return self._steps + 0.0

    @REWARD_TERM_REGISTRY.register()
    class HiddenCounterReward(RewardTerm):
        """Keeps its state where no Eden attribute reaches (``SimpleNamespace`` is not walked), allocated on its
        first call, so under the reward manager's inference mode, and advanced in place.
        """

        def build(self) -> None:
            self._hidden = SimpleNamespace(count=None)

        def compute(self, envs_idx=None) -> torch.Tensor:
            if self._hidden.count is None:
                self._hidden.count = torch.zeros(self.num_envs, device=self.device)
            self._hidden.count.add_(1.0)
            return self._hidden.count * 1.0

    @REWARD_TERM_REGISTRY.register()
    def fresh_cache_reward(env) -> torch.Tensor:
        height = env.entities["box"].get_pos()[:, 2]
        padded = torch.zeros_like(height)
        padded.add_(height)  # a fresh tensor written in place, then kept by a cache
        FRESH_CACHE["height"] = padded
        return padded * 1.0

    @REWARD_TERM_REGISTRY.register()
    class ContactForceReward(RewardTerm):
        """Sums the box's contact-force magnitudes over the contact axis.

        A float reduction whose bits depend on how many slots it runs over. Records each width it saw (the
        eager runs and the capture; a replay runs no Python).
        """

        def build(self) -> None:
            self.widths: list[int] = []

        def compute(self, envs_idx=None) -> torch.Tensor:
            box = self._env.entities["box"]
            contacts = box.get_contacts(with_entity=self._env.entities["plane"])
            self.widths.append(contacts["valid_mask"].shape[-1])
            return torch.sum(torch.norm(contacts["force_a"], dim=-1) * contacts["valid_mask"], dim=-1)


BOX_REWARDS = {
    "python_state": "python_counter_reward",
    "host_read": "host_read_reward",
    "field_change": "scaled_height_reward",
    "rebind": "offset_height_reward",
    "rebind_holder": "held_offset_reward",
    "env_constant": "scaled_height_reward",
    "term_replaced": "scaled_height_reward",
    "one_off": "scaled_height_reward",
    "storm": "scaled_height_reward",
    "counter_read": "counter_gated_reward",
    "side_stream": "side_stream_reward",
    "hidden_state": "hidden_counter_reward",
    "fresh_cache": "fresh_cache_reward",
    "contacts": "contact_force_reward",
}


def _change_late(env, scenario: str) -> None:
    """What changes, once, long after the capture, in the scenarios that re-capture.

    ``field_change``, ``rebind``, ``rebind_holder``, ``env_constant`` and ``term_replaced``.
    """
    import copy

    import torch

    term = env.reward_manager.get_term("probe")
    for _ in range(env.env_options.cuda_graph_options.recapture_window + 10):
        env.step(torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device))
    if scenario == "field_change":
        term.scale = 2.0
    elif scenario == "env_constant":
        env.max_episode_length += 1
    elif scenario == "term_replaced":
        env.reward_manager.terms["probe"] = copy.copy(term)
    elif scenario == "rebind_holder":
        term._held = {"offset": torch.ones(env.num_envs, device=env.device)}
    else:
        term._offset = torch.ones(env.num_envs, device=env.device)


def _step_until_error(env, steps: int = 12, every_step=None) -> str | None:
    import torch

    action = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
    try:
        for _ in range(steps):
            env.step(action)
            if every_step is not None:
                every_step()
    except RuntimeError as exc:
        return str(exc)
    return None


def main(scenario: str, task: str | None) -> None:
    import eden as en  # before genesis and torch (DDP import order)
    import genesis as gs

    en.init(backend=gs.gpu, logging_level="WARNING")
    import torch

    from eden.envs.graded_graph import GradedPhaseGraph

    captures = 0
    capture = GradedPhaseGraph._capture

    def counting_capture(self, *args):
        nonlocal captures
        captures += 1
        return capture(self, *args)

    GradedPhaseGraph._capture = counting_capture

    if scenario == "task":
        from eden.envs.base import env_class_for_config
        from eden.options import CudaGraphOptions
        from eden.tasks import TASK_REGISTRY

        cfg = TASK_REGISTRY.build(task)
        cfg.env_options.num_envs = 16
        cfg.env_options.cuda_graph_options = CudaGraphOptions()
        env = env_class_for_config(cfg).from_config(cfg)
        env.build()
        env.reset(seed=0)
        generator = torch.Generator(device=env.device).manual_seed(0)
        dim = env.action_manager.total_action_dim
        rewards = env.reward_manager
        term = next(iter(rewards._terms.values()))
        with torch.inference_mode():
            for step in range(16):
                if step == 10:
                    term.weight = 2.0 * term.weight
                env.episode_length_buf[RESET_ENVS] = int(env.max_episode_length) - 1
                env.step(torch.rand(env.num_envs, dim, device=env.device, generator=generator) * 2 - 1)
        # The last replay weighted the first term's raw value by the doubled weight, read from the device.
        weight_applied = bool(
            torch.equal(rewards._step_reward[:, 0], rewards._step_raw_reward[:, 0] * term.weight)
            and rewards._weights[0].item() == term.weight
        )
        print("RESULT " + json.dumps({"captures": captures, "weight_applied": weight_applied}))
    else:
        _register_probe_terms()
        env = _box_env(BOX_REWARDS[scenario], on_a_plane=scenario == "contacts")
        term = env.reward_manager.get_term("probe")
        if scenario in ("field_change", "rebind", "rebind_holder", "env_constant", "term_replaced"):
            _change_late(env, scenario)
        if scenario == "storm":
            error = _step_until_error(env, steps=8, every_step=lambda: setattr(term, "scale", term.scale + 1.0))
        elif scenario == "one_off":
            steps = 0

            def change_after_the_capture() -> None:
                nonlocal steps
                steps += 1
                if steps == env.env_options.cuda_graph_options.warmup_steps + 1:
                    term.scale = 2.0

            error = _step_until_error(env, every_step=change_after_the_capture)
        elif scenario == "hidden_state":
            error = _step_until_error(env)
            counts = term._hidden.count.unique().tolist()
            print("RESULT " + json.dumps({"captures": captures, "error": error, "counts": counts}))
            return
        elif scenario == "contacts":
            error = _step_until_error(env)
            print("RESULT " + json.dumps({"captures": captures, "error": error, "widths": sorted(set(term.widths))}))
            return
        elif scenario == "fresh_cache":
            import torch

            action = torch.zeros(env.num_envs, env.action_manager.total_action_dim, device=env.device)
            stale = []
            for step in range(1, 13):
                FRESH_CACHE.clear()
                env.step(action)
                cached = FRESH_CACHE.get("height")
                if cached is not None and not torch.equal(cached, env.entities["box"].get_pos()[:, 2]):
                    stale.append(step)
            print("RESULT " + json.dumps({"captures": captures, "stale_cache_steps": stale}))
            return
        else:
            error = _step_until_error(env)
        print("RESULT " + json.dumps({"captures": captures, "error": error}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("scenario", choices=("task", *BOX_REWARDS))
    parser.add_argument("task", nargs="?")
    args = parser.parse_args()
    main(args.scenario, args.task)
