"""Send the same sinusoidal actions to sim and real XHand left hand side-by-side.

---
tier: recipe
capabilities: [robot.xhand, deploy.real_robot]
hardware: real-robot
optional_deps: []
run: uv run python examples/recipes/deployment/real_xhand.py
---

The sim hand steps with a full physics solve while the real hand executes
the same position targets via the XHand deployment backend.  A live
matplotlib plot streams per-joint position error so you can see where
sim and real diverge.

Usage:
    uv run examples/recipes/deployment/real_xhand.py
    uv run examples/recipes/deployment/real_xhand.py --protocol RS485 --serial-port /dev/ttyUSB0
"""

import argparse
from collections import deque

import genesis as gs
import numpy as np
import torch
import matplotlib.pyplot as plt

import eden as en
from eden.envs.base import RLEnvBase
from eden.extensions.deployment.robotera_xhand import RoboTeraXHandDeployment
from eden.extensions.deployment.utils.state import RobotStateEntity
from eden.managers.terms.actions.joint_actions import ExplicitPDController
from eden.options import EnvOptions, ActionManagerOptions
from eden.options.robots.robotera_xhand import XHand1_L
from eden.options.scenes.empty import EmptySceneOptions

CONTROL_FREQ = 30  # Hz
DURATION_S = 30.0
PLOT_WINDOW = 500
PLOT_EVERY = 2


def build_action_sequence(num_dofs: int, num_steps: int, dt: float) -> np.ndarray:
    """Generate a smooth sinusoidal sweep that exercises each finger."""
    t = np.arange(num_steps) * dt
    actions = np.zeros((num_steps, num_dofs), dtype=np.float32)
    for j in range(num_dofs):
        freq = 0.3 + 0.1 * j  # slightly different freq per joint
        phase = j * np.pi / num_dofs
        actions[:, j] = 0.4 * np.sin(2 * np.pi * freq * t + phase)
    return actions


def main():
    parser = argparse.ArgumentParser(description="Sim vs real XHand comparison")
    parser.add_argument(
        "--protocol", type=str, default="RS485", choices=["EtherCAT", "RS485"], help="Hardware bus the hand is wired on"
    )
    parser.add_argument("--serial-port", type=str, default="/dev/ttyUSB0", help="Serial device the hand is attached to")
    parser.add_argument("--hand-id", type=int, default=0, help="Hardware id of the hand on the bus")
    parser.add_argument(
        "-t", "--seconds", type=float, default=DURATION_S, help="Seconds to run the sim/real comparison"
    )
    parser.add_argument("-c", "--cpu", action="store_true", help="Run on CPU instead of the GPU")
    args = parser.parse_args()

    en.init(backend=gs.cpu if args.cpu else gs.gpu, debug=True)

    control_dt = 1.0 / CONTROL_FREQ
    num_steps = int(args.seconds / control_dt)

    # -- Build environment with a single left hand ----------------------------
    env = RLEnvBase(
        env_options=EnvOptions(
            num_envs=1,
            sim_dt=control_dt,
            decimation=1,
        ),
        scene_options=EmptySceneOptions(robot=XHand1_L(is_fixed_base=True)),
        action_options=ActionManagerOptions(
            hand_controller=ExplicitPDController.configure(
                entity_name="robot",
                dofs_name=list(XHand1_L().dofs_name),
                scale=1.0,
            ),
        ),
        show_viewer=True,
    )
    env.build()

    sim_hand = env.entities["robot"]
    num_dofs = sim_hand.num_dofs

    # -- Connect the deployment to the real hand ------------------------------
    dep = RoboTeraXHandDeployment(
        env,
        RoboTeraXHandDeployment.configure(
            entity_name="robot",
            control_freq=CONTROL_FREQ,
            sync=True,
            connect_timeout_s=10.0,
            protocol=args.protocol,
            serial_port=args.serial_port,
            hand_id=args.hand_id,
        ),
    )
    dep.connect()

    state_entity: RobotStateEntity = dep._state_entity

    # -- Pre-compute action sequence ------------------------------------------
    actions = build_action_sequence(num_dofs, num_steps, control_dt)

    # -- Live plot setup ------------------------------------------------------
    plt.ion()
    fig, (ax_pos, ax_err) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    fig.canvas.manager.set_window_title("XHand Sim vs Real")
    fig.suptitle("XHand Left — Sim vs Real Joint Positions")

    hist_step: deque[int] = deque(maxlen=PLOT_WINDOW)
    hist_sim_pos: deque[np.ndarray] = deque(maxlen=PLOT_WINDOW)
    hist_real_pos: deque[np.ndarray] = deque(maxlen=PLOT_WINDOW)
    hist_max_err: deque[float] = deque(maxlen=PLOT_WINDOW)

    # -- Reset: the hand boots in STOP, holding; the operator INITs then STARTs it ----------
    dep.reset()
    en.logger.info("[xhand] Holding. Press i+Enter to ramp to the default pose, then s+Enter to play the sequence.")
    while not dep.policy_active:
        dep.step(None)  # the hold, then the INIT ramp, until START

    # -- Main loop: same action to both sim and real -------------------------
    for step in range(num_steps):
        action = torch.from_numpy(actions[step]).unsqueeze(0).to(env.device)

        # Sim: step the environment with physics
        env.step(action)
        sim_pos = sim_hand.get_dofs_pos()[0].detach().cpu().numpy()

        # Real: send the same target via deployment
        dep.step(action)
        real_pos = state_entity.get_dofs_pos()[0].detach().cpu().numpy()

        # -- Collect history --------------------------------------------------
        hist_step.append(step)
        hist_sim_pos.append(sim_pos.copy())
        hist_real_pos.append(real_pos.copy())
        hist_max_err.append(float(np.max(np.abs(sim_pos - real_pos))))

        # -- Update plot ------------------------------------------------------
        if step % PLOT_EVERY == 0 and len(hist_step) > 1:
            x = np.array(hist_step)
            sim_arr = np.vstack(hist_sim_pos)
            real_arr = np.vstack(hist_real_pos)

            ax_pos.clear()
            for j in range(num_dofs):
                color = plt.cm.tab20(j / num_dofs)
                ax_pos.plot(x, sim_arr[:, j], "-", color=color, alpha=0.6)
                ax_pos.plot(x, real_arr[:, j], "--", color=color, alpha=0.6)
            ax_pos.set_ylabel("rad")
            ax_pos.set_title("Joint Positions (solid=sim, dashed=real)")
            ax_pos.grid(True, alpha=0.3)

            ax_err.clear()
            ax_err.plot(x, np.array(hist_max_err), "r-")
            ax_err.set_ylabel("max |sim - real| (rad)")
            ax_err.set_xlabel("step")
            ax_err.set_title("Max Joint Position Error")
            ax_err.grid(True, alpha=0.3)

            fig.tight_layout()
            fig.canvas.draw_idle()
            fig.canvas.flush_events()

    # -- Cleanup --------------------------------------------------------------
    dep.close()
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
