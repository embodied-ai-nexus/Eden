"""Keyboard teleoperation of a Franka Panda with live task-metric terms plotted in real time.

---
tier: recipe
capabilities: [teleop.keyboard, robot.franka]
hardware: cpu
optional_deps: []
run: uv run python examples/recipes/teleop/keyboard/teleoperation_keyboard_metrics.py -v
---

Extends the basic keyboard teleoperation example with a pick-and-place task:
    * Pick up the apple
    * Place it on the plate

Metric terms track each sub-goal independently and a SequentialMetricTerm
chains them into an overall task progress score.  A separate matplotlib
window plots all metric values in real-time so you can visually verify the
metric functions while teleoperating.

Controls (same as base example):
    WASDQE  - translate end-effector
    IJKLUO  - rotate end-effector
    SPACE   - toggle gripper open/close
    X       - reset environment
    ESC     - quit

Usage:
    python examples/recipes/teleop/keyboard/teleoperation_keyboard_metrics.py -v         # GPU + viewer
    python examples/recipes/teleop/keyboard/teleoperation_keyboard_metrics.py -v -c      # CPU + viewer
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import deque
import logging

import genesis as gs

import eden as en
from eden.constants import EventMode, MetricDirection, MetricMode
from eden.envs.base import EnvBase
from eden.managers.terms.metrics import MetricTermAND, MetricTermNOT
from eden.terms import EventTerm, MetricTerm
from eden.managers.terms.actions.binary_actions import BinaryJointController
from eden.managers.terms.actions.task_space_actions import (
    DifferentialIKController,
    OperationalSpaceController,
)
from eden.managers.terms.events.placement import place_constant_xy, place_uniform
from eden.managers.terms.metrics.proximity import (
    object_a_is_on_b,
    entity_distance_xy,
    entity_near_entity_xy,
    EeNearEntity,
)
from eden.managers.terms.metrics.grasp import object_lifted, IsGrasping
from eden.managers.terms.metrics.sequence import SequentialMetricTerm
from eden.options.managers.metrics import PhaseOptions
from eden.options import (
    ActionManagerOptions,
    EnvOptions,
    EventManagerOptions,
    SceneOptions,
)
from eden.options.entities import EntityOptions, PrimitiveOptions
from eden.options.managers.metrics import MetricManagerOptions
from eden.options.materials import RigidMaterialOptions
from eden.utils.devices.keyboard import SE3Keyboard


# ---------------------------------------------------------------------------
# Live metric plotter (runs in a separate thread)
# ---------------------------------------------------------------------------


class LiveMetricPlotter:
    """Non-blocking matplotlib window that plots metric time-series.

    All matplotlib calls run on the main thread.  Call :meth:`start` once
    to create the figure, :meth:`update` each step to push data, and
    :meth:`tick` periodically to redraw the window.
    """

    def __init__(self, metric_names: list[str], maxlen: int = 500):
        self._names = metric_names
        self._maxlen = maxlen
        self._buffers: dict[str, deque] = {n: deque(maxlen=maxlen) for n in metric_names}
        self._buffers["success"] = deque(maxlen=maxlen)
        self._buffers["progress"] = deque(maxlen=maxlen)
        self._step_buf: deque = deque(maxlen=maxlen)
        self._step = 0
        self._fig = None
        self._axes = None
        self._lines_terms = {}
        self._line_prog = None
        self._line_succ = None
        self._line_dist = None

    def start(self):
        """Create the matplotlib figure (must be called from the main thread)."""
        import matplotlib

        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt

        plt.ion()
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        fig.suptitle("Live Task Metrics  (pick apple → place on plate)")
        fig.canvas.manager.set_window_title("Eden Metric Monitor")

        ax_terms = axes[0]
        ax_terms.set_ylabel("Metric Value")
        ax_terms.set_ylim(-0.05, 1.15)
        ax_terms.set_title("Individual Metric Terms")
        for name in self._names:
            (line,) = ax_terms.plot([], [], label=name)
            self._lines_terms[name] = line
        ax_terms.legend(loc="upper left", fontsize="small", ncol=2)

        ax_prog = axes[1]
        ax_prog.set_ylabel("Progress")
        ax_prog.set_ylim(-0.05, 1.15)
        ax_prog.set_title("Task Progress (SequentialMetricTerm)")
        (self._line_prog,) = ax_prog.plot([], [], color="tab:blue", label="progress")
        (self._line_succ,) = ax_prog.plot([], [], color="tab:green", linestyle="--", label="success")
        ax_prog.legend(loc="upper left", fontsize="small")

        ax_dist = axes[2]
        ax_dist.set_ylabel("Distance (m)")
        ax_dist.set_xlabel("Step")
        ax_dist.set_title("Apple ↔ Plate XY Distance")
        (self._line_dist,) = ax_dist.plot([], [], color="tab:orange", label="apple_plate_dist_xy")
        ax_dist.legend(loc="upper left", fontsize="small")

        plt.tight_layout()
        self._fig = fig
        self._axes = axes
        self._plt = plt

    def update(self, metric_values: dict[str, float], success: bool, progress: float):
        """Push new values (called from the main loop)."""
        self._step += 1
        self._step_buf.append(self._step)
        for name in self._names:
            self._buffers[name].append(metric_values.get(name, 0.0))
        self._buffers["success"].append(float(success))
        self._buffers["progress"].append(progress)

    def tick(self):
        """Redraw the plot (must be called from the main thread)."""
        if self._fig is None or len(self._step_buf) == 0:
            return
        try:
            steps = list(self._step_buf)
            term_data = {n: list(self._buffers[n]) for n in self._names}
            progress_data = list(self._buffers["progress"])
            success_data = list(self._buffers["success"])

            for name, line in self._lines_terms.items():
                line.set_data(steps, term_data[name])

            self._line_prog.set_data(steps, progress_data)
            self._line_succ.set_data(steps, success_data)

            if "apple_plate_dist_xy" in term_data:
                self._line_dist.set_data(steps, term_data["apple_plate_dist_xy"])
                if term_data["apple_plate_dist_xy"]:
                    self._axes[2].set_ylim(0, max(max(term_data["apple_plate_dist_xy"]) * 1.1, 0.1))

            for ax in self._axes:
                ax.set_xlim(steps[0], steps[-1] + 1)
                ax.relim()

            self._fig.canvas.draw_idle()
            self._fig.canvas.flush_events()
            self._plt.pause(0.001)
        except Exception as exc:
            en.logger.warning(f"Plotter tick failed: {exc}")

    def stop(self):
        if self._fig is not None:
            self._plt.close(self._fig)
            self._fig = None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

# Thresholds (tuned for the Franka + work_table scene)
EE_NEAR_THRESHOLD = 0.05  # generous — "hand" link is at the wrist, not fingertips
GRASP_FORCE_THRESHOLD = 0.5  # min contact force per gripper pad to count as grasping
LIFT_MARGIN = 0.05  # how far above resting height counts as "lifted"
PLACE_THRESHOLD = 0.03  # XY distance for apple-on-plate detection

# End-effector link names — use the gripper pads (fingertip surfaces) for
# a tighter proximity check in the sequential metric.
EE_LINK_NAMES = ["left_gripper_pad", "right_gripper_pad"]


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--cpu", action="store_true", help="Run on CPU instead of the GPU")
    parser.add_argument("-v", "--vis", action="store_true", help="Open the interactive viewer")
    parser.add_argument(
        "--action",
        type=str,
        default="diffik",
        choices=["diffik", "osc"],
        help="End-effector controller to drive the arm with",
    )
    parser.add_argument("--no-plot", action="store_true", help="Disable live metric plot")
    args = parser.parse_args()

    en.init(backend=gs.cpu if args.cpu else gs.gpu, debug=True)

    # Enable DEBUG logging for the sequential metric module so we can see
    # per-phase evaluation details.  INFO-level logs phase transitions.
    logging.getLogger("eden.managers.terms.metrics.sequence").setLevel(logging.DEBUG)

    # ------------------------------------------------------------------
    # Build the environment first (without metrics) so we can read the
    # apple's resting height and set thresholds dynamically.
    # ------------------------------------------------------------------
    env = EnvBase(
        env_options=EnvOptions(
            num_envs=1,
            env_spacing=(1.5, 1.5),
            enable_default_keybinds=False,
        ),
        scene_options=SceneOptions(
            table=EntityOptions(
                file="objects/tables/work_table/work_table.obj",
                is_fixed_base=True,
                is_support_enabled=True,
            ),
            plate=EntityOptions(
                file="objects/plates/86d4d4f8-ab98-4c48-9630-dc5c7710733b.glb",
                is_fixed_base=False,
            ),
            apple=EntityOptions(
                file="objects/apples/013_apple/textured.obj",
                is_fixed_base=False,
            ),
            floor=PrimitiveOptions(
                morph=gs.morphs.Plane(),
                is_support_enabled=True,
                default_root_pos=(0.0, 0.0, -0.5),
            ),
            robot=en.robots.FrankaEmikaPanda(
                material=RigidMaterialOptions(gravity_compensation=1.0),
            ),
        ),
        event_options=EventManagerOptions(
            place_table=EventTerm.configure(
                mode=EventMode.STARTUP,
                func=place_constant_xy,
                params={
                    "entity_name": "table",
                    "support_entity_name": "floor",
                    "pos_xy": (0.45, 0.0),
                },
                priority=0,
            ),
            place_plate=EventTerm.configure(
                mode=EventMode.RESET,
                func=place_uniform,
                params={
                    "entity_name": "plate",
                    "support_entity_name": "table",
                    "samples_per_area": 100,
                },
                priority=1,
            ),
            place_apple=EventTerm.configure(
                mode=EventMode.RESET,
                func=place_uniform,
                params={
                    "entity_name": "apple",
                    "support_entity_name": "table",
                    "samples_per_area": 100,
                },
                priority=2,
            ),
        ),
        action_options=ActionManagerOptions(
            ee_controller={
                "diffik": DifferentialIKController,
                "osc": OperationalSpaceController,
            }[args.action].configure(
                entity_name="robot",
                ee_link_name="hand",
                dofs_name=[
                    "joint1",
                    "joint2",
                    "joint3",
                    "joint4",
                    "joint5",
                    "joint6",
                    "joint7",
                ],
                position_scale=0.05,
                orientation_scale=0.05,
            ),
            gripper_controller=BinaryJointController.configure(
                entity_name="robot",
                dofs_name=["finger_joint1", "finger_joint2"],
                open_action=0.04,
                close_action=0.0,
            ),
        ),
        show_viewer=args.vis,
    )

    env.build()
    env.reset()

    # ------------------------------------------------------------------
    # Read the apple's resting height to compute a meaningful lift_height
    # ------------------------------------------------------------------
    apple_rest_z = env.entities["apple"].get_pos()[0, 2].item()
    lift_height = apple_rest_z + LIFT_MARGIN

    en.logger.info(f"Apple resting z = {apple_rest_z:.4f}  →  lift_height = {lift_height:.4f}")

    # ------------------------------------------------------------------
    # Configure metrics with calibrated thresholds.
    #
    # The five individual metrics use metric_mode=RESET so they are
    # tracked (plotted) but do NOT gate the MetricManager's success_buf.
    # Only task_progress (INTERVAL mode) determines success — it flips
    # to 1.0 when all 4 sequential phases are complete.
    # ------------------------------------------------------------------
    metric_options = MetricManagerOptions(
        # -- Individual metrics (for monitoring / plotting only) --
        # metric_mode=RESET means they are computed every step but
        # do not count toward the INTERVAL-based success check.
        ee_near_apple=EeNearEntity.configure(
            robot_name="robot",
            ee_link_names=EE_LINK_NAMES,
            entity_name="apple",
            threshold=EE_NEAR_THRESHOLD,
            metric_mode=MetricMode.RESET,
        ),
        is_grasping=IsGrasping.configure(
            robot_name="robot",
            left_gripper_link_name="left_gripper_pad",
            right_gripper_link_name="right_gripper_pad",
            force_threshold=GRASP_FORCE_THRESHOLD,
            metric_mode=MetricMode.RESET,
        ),
        apple_lifted=MetricTerm.configure(
            func=object_lifted,
            params={"entity_name": "apple", "min_height": lift_height},
            metric_mode=MetricMode.RESET,
        ),
        apple_near_plate=MetricTerm.configure(
            func=entity_near_entity_xy,
            params={
                "entity_a_name": "apple",
                "entity_b_name": "plate",
                "threshold": PLACE_THRESHOLD,
            },
            metric_mode=MetricMode.RESET,
        ),
        apple_plate_dist_xy=MetricTerm.configure(
            func=entity_distance_xy,
            params={"entity_a_name": "apple", "entity_b_name": "plate"},
            direction=MetricDirection.LIB,
            success_threshold=PLACE_THRESHOLD,
            metric_mode=MetricMode.RESET,
        ),
        # -- Composite sequential metric (only INTERVAL term → drives success) --
        task_progress=SequentialMetricTerm.configure(
            phases=[
                # Phase 0: reach — gripper pads near the apple
                PhaseOptions(
                    name="reach",
                    ref="ee_near_apple",
                    hold=True,
                    hold_until="grasp",
                ),
                # Phase 1: grasp — contact force on both gripper pads
                # hold until "place": releasing the apple before placement reverts here
                PhaseOptions(
                    name="grasp",
                    ref="is_grasping",
                    hold=True,
                    hold_until="move_to_plate",
                ),
                # Phase 2: lift — apple above its resting height
                # hold until "place": dropping the apple before placement reverts here
                PhaseOptions(
                    name="lift",
                    ref="apple_lifted",
                    hold=True,
                    hold_until="move_to_plate",
                    hold_guard="grasp",
                ),
                # Phase 3: move to plate — apple is above the plate
                PhaseOptions(
                    name="move_to_plate",
                    term=MetricTerm.configure(
                        func=object_a_is_on_b,
                        params={
                            "entity_a_name": "apple",
                            "entity_b_name": "plate",
                            "xy_threshold": PLACE_THRESHOLD,
                            "height_threshold": 0.05,
                            "height_diff": 0.1,
                        },
                        metric_mode=MetricMode.RESET,
                    ),
                    hold=True,
                    hold_until="place",
                ),
                # Phase 4: place — apple horizontally aligned with the plate
                # Once placed, grasp and lift holds expire — you can release safely
                PhaseOptions(
                    name="place",
                    term=MetricTermAND.configure(
                        terms=[
                            MetricTerm.configure(
                                func=object_a_is_on_b,
                                params={
                                    "entity_a_name": "apple",
                                    "entity_b_name": "plate",
                                    "xy_threshold": PLACE_THRESHOLD,
                                    "height_threshold": 0.05,
                                    "height_diff": 0.1,
                                },
                                metric_mode=MetricMode.RESET,
                            ),
                            MetricTermNOT.configure(
                                term=IsGrasping.configure(
                                    robot_name="robot",
                                    left_gripper_link_name="left_gripper_pad",
                                    right_gripper_link_name="right_gripper_pad",
                                    force_threshold=GRASP_FORCE_THRESHOLD,
                                    metric_mode=MetricMode.RESET,
                                )
                            ),
                        ]
                    ),
                ),
            ],
            allow_skip=False,
        ),
    )

    # Attach metrics to the environment's metric manager
    from eden.managers.metric_manager import MetricManager

    env.metric_manager = MetricManager(env, metric_options)
    env.metric_options = metric_options

    en.logger.info(env.action_manager.summary())
    en.logger.info(env.metric_manager.summary())

    # ------------------------------------------------------------------
    # Helper to update dynamic thresholds after each reset
    # ------------------------------------------------------------------
    def _update_thresholds():
        """Recalibrate lift height after each reset (apple resting z varies
        slightly with random placement).
        """
        z = env.entities["apple"].get_pos()[0, 2].item()
        h = z + LIFT_MARGIN
        # Standalone apple_lifted term (public .params attribute)
        env.metric_manager.get_term("apple_lifted").params["min_height"] = h
        # Phase 2 ("lift") inside SequentialMetricTerm — use get_phase_term()
        task_term = env.metric_manager.get_term("task_progress")
        task_term.get_phase_params("lift")["min_height"] = h  # func-based term has .params

    # ------------------------------------------------------------------
    # Live plotter
    # ------------------------------------------------------------------
    individual_metric_names = [
        "ee_near_apple",
        "is_grasping",
        "apple_lifted",
        "apple_near_plate",
        "apple_plate_dist_xy",
    ]

    plotter = None
    if not args.no_plot:
        try:
            plotter = LiveMetricPlotter(individual_metric_names, maxlen=1000)
            plotter.start()
            en.logger.info("Live metric plot window launched.")
        except Exception as e:
            en.logger.warning(f"Could not start live plot (matplotlib/Tk unavailable): {e}")
            plotter = None

    # ------------------------------------------------------------------
    # Keyboard setup
    # ------------------------------------------------------------------
    keyboard = SE3Keyboard(key_bindings=SE3Keyboard.wasd_key_bindings())

    # NOTE: keyboard callbacks are invoked while holding the pynput
    # listener lock.  Heavy work (env.reset, threshold recalibration)
    # blocks the listener and causes key events to be dropped.  Instead,
    # set a flag here and handle the actual reset in the main loop.
    _reset_requested = [False]

    def _on_reset(on_press):
        if on_press:
            _reset_requested[0] = True

    keyboard.add_callback("x", _on_reset)

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    en.logger.info(
        "\n=== Pick-and-Place Task ===\n"
        "Objective: pick up the apple and place it on the plate.\n"
        "Phases: reach → grasp (hold) → lift (hold) → place\n"
        "  Dropping the apple reverts progress to the failing phase.\n"
        "Press X to reset, ESC to quit.\n"
    )

    step_count = 0
    PRINT_EVERY = 100  # print status line every N steps

    try:
        while True:
            action = keyboard.get_command()
            action = {
                "ee_controller": action[:6].unsqueeze(0).repeat(env.num_envs, 1).to(env.device),
                "gripper_controller": action[6:].unsqueeze(0).repeat(env.num_envs, 1).to(env.device),
            }
            env.step(action)
            step_count += 1

            # Handle deferred reset (see _on_reset callback above)
            if _reset_requested[0]:
                _reset_requested[0] = False
                env.reset()
                keyboard.reset()
                _update_thresholds()

            # Read metric values
            mm = env.metric_manager
            if mm is not None:
                metric_vals = {}
                for name in individual_metric_names:
                    metric_vals[name] = mm._step_metric[name][0].item()

                task_term = mm.get_term("task_progress")
                num_phases = max(len(task_term.phase_names), 1)
                cur_phase = task_term._phase_buf[0].item()
                progress = cur_phase / num_phases
                success = bool(mm.success_buf[0].item())

                # Feed the plotter and redraw
                if plotter is not None:
                    plotter.update(metric_vals, success=success, progress=progress)
                    plotter.tick()

                # Periodic status line to stdout
                if step_count % PRINT_EVERY == 0:
                    vals = "  ".join(f"{k}={v:.2f}" for k, v in metric_vals.items())
                    apple_z = env.entities["apple"].get_pos()[0, 2].item()
                    en.logger.info(
                        f"[step {step_count}] phase={cur_phase}/{num_phases} "
                        f"progress={progress:.2f} success={success}  "
                        f"apple_z={apple_z:.4f}  {vals}"
                    )

    except KeyboardInterrupt:
        pass
    finally:
        if plotter is not None:
            plotter.stop()
        en.logger.info("Done.")


if __name__ == "__main__":
    main()
