"""Live reward plots and interactive disturbances during a policy rollout."""

import math
import time
from collections import deque
from functools import partial
from queue import SimpleQueue

import genesis as gs
import numpy as np
import torch
from viser import uplot

import eden as en
from eden.envs.base import RLEnvBase
from eden.extensions.visualization.viser import ViserViewer


class RewardViewer(ViserViewer):
    def __init__(self, env, *, host="localhost", port=8080):
        if not isinstance(getattr(env, "unwrapped", env), RLEnvBase):
            raise TypeError(
                f"Reward plots need an RL env with a reward manager, but got {type(env).__name__}. "
                "Pass a --task whose config subclasses EdenRLConfig."
            )
        if not env.unwrapped.reward_manager.active_terms:
            raise ValueError(
                f"Reward plots need a reward term to plot, but {type(env.unwrapped.config).__name__} builds none: "
                "a reward term of weight 0.0 is not built. Give a term in reward_options a nonzero weight, or roll "
                "out without the reward viewer."
            )
        super().__init__(env.unwrapped, host=host, port=port)
        self._step_env = env.step
        # Wall-clock seconds spent waiting for Play and drawing the plots and the scene. A rollout's timing report
        # subtracts this, so it measures the policy and the sim rather than the viewer.
        self.viewer_time = 0.0

    def _setup_gui(self):
        manager = self.env.reward_manager

        gui = self.server.gui
        self.playing = gui.add_checkbox("Play", initial_value=False)
        self.env_index = gui.add_number(
            "Environment", 0, min=0, max=self.env.num_envs - 1, step=1, visible=self.env.num_envs > 1
        )
        self._history = deque(maxlen=200)
        self._last_env = None
        self._impulses = SimpleQueue()
        links = {
            f"{name}/{link.name}": link.idx
            for name, entity in self.env.entities.items()
            if entity.solver is self.env.rigid_solver
            for link in entity.links
            if not link.is_fixed
        }
        if links:
            with gui.add_folder("Disturbance"):
                target = gui.add_dropdown("Link", tuple(links))
                impulse = gui.add_vector3("World impulse (N s)", (1.0, 0.0, 0.0), step=0.01)
                push = gui.add_button("Apply impulse")

                @push.on_click
                def _(_):
                    self._impulses.put((self.env_index.value, links[target.value], impulse.value))

        self._weights = []
        self._plots = {}
        with gui.add_folder("Weighted rewards per second"):
            for name, term in manager.iter_terms():
                with gui.add_folder(name) as folder:
                    # Viser sends a text field's value on every keystroke, so the weight changes only on the button:
                    # a half-typed "-" or "-0" must neither be rejected under the cursor nor rescore any step. The
                    # button is enabled only while the field differs from the weight in force, so an edit that was
                    # never applied stays visible.
                    weight = gui.add_text("Weight", str(term.weight), hint="Type a number, then click Set weight.")
                    set_weight = gui.add_button("Set weight", disabled=True)
                    weight.on_update(partial(self._mark_weight_edit, term, set_weight))
                    set_weight.on_click(partial(self._set_weight, name, term, weight, set_weight))
                    self._weights.append((term, weight, set_weight))
                    plot = gui.add_uplot(
                        data=(np.empty(0), np.empty(0)),
                        series=(uplot.Series(label="Time (s)"), uplot.Series(label="Rate", stroke="#3b82f6")),
                        scales={"x": uplot.Scale(time=False)},
                        aspect=2.8,
                    )
                self._plots[name] = folder, plot
        self._setup_visualization_gui()

    @staticmethod
    def _mark_weight_edit(term, button, event):
        button.disabled = event.target.value == str(term.weight)

    def _set_weight(self, name, term, field, button, _event):
        try:
            weight = float(field.value)
        except ValueError:
            weight = math.nan
        if not math.isfinite(weight):
            en.logger.warning(
                f"Reward weight {field.value!r} for term '{name}' is not a finite number; keeping {term.weight}."
            )
        else:
            term.weight = weight
        # Show the weight in force in its canonical spelling, which also disables the button until the next edit.
        field.value = str(term.weight)
        button.disabled = True

    def step(self, action, *, pace=True):
        if not self._built:
            self.update()
        wait_start = time.monotonic()
        waiting = False
        while True:
            index = self.env_index.value
            if self._last_env != index:
                self._history.clear()
                self._last_env = index
                for name, (folder, plot) in self._plots.items():
                    folder.label = name
                    plot.data = (np.empty(0), np.empty(0))
            if self.playing.value:
                break
            if not waiting:
                en.logger.info(f"Reward plots: waiting for Play at http://{self.host}:{self.server.get_port()}")
                waiting = True
            time.sleep(0.02)

        start = time.monotonic()
        self.viewer_time += start - wait_start
        episode_step = self.env.episode_length_buf[index] + 1
        while not self._impulses.empty():
            env_index, link_index, impulse = self._impulses.get_nowait()
            force = torch.tensor(impulse, device=self.env.device, dtype=gs.tc_float) / self.env.env_options.sim_dt
            self.env.rigid_solver.apply_links_external_wrench(
                force=force.reshape(1, 1, 3),
                links_idx=[link_index],
                envs_idx=[env_index],
                ref=gs.link_ref_frame.link_COM,
                local=False,
            )

        result = self._step_env(action)
        rates = self.env.reward_manager.step_reward_rates[index]
        # This read waits for the step's device work, so the clock below starts once the sim has finished and
        # times only the drawing. The per-link scene refresh in update() makes a device round trip per link.
        episode_step, *rates = torch.cat((episode_step.view(1).to(rates.dtype), rates)).tolist()
        draw_start = time.monotonic()
        if episode_step == 1:
            self._history.clear()
        self._history.append((episode_step * self.env.dt, *rates))
        times, *values = np.asarray(self._history).T
        with self.server.atomic():
            self.update()
            for (name, (folder, plot)), series in zip(self._plots.items(), values, strict=True):
                folder.label = f"{name}  {series[-1]:.4g}"
                plot.data = (times, series)
        self.viewer_time += time.monotonic() - draw_start
        if pace:
            time.sleep(max(0.0, self.env.dt - (time.monotonic() - start)))
        return result
