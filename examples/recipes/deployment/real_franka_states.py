"""Compare sim vs real sensor readings on a Franka Emika Panda to verify convention consistency.

---
tier: recipe
capabilities: [robot.franka, deploy.real_robot]
hardware: real-robot
optional_deps: []
run: LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 uv run python examples/recipes/deployment/real_franka_states.py
---

Reads real arm state from the Franka Emika Panda via panda-py, mirrors the
joint angles in sim, and prints a side-by-side comparison for every quantity
that observation terms use.  Flags any mismatch above a configurable tolerance.

The Franka is a fixed-base manipulator with no IMU, so the base-quaternion /
projected-gravity / angular-velocity sections from the G1 viewer are replaced
by an end-effector pose readout and a Franka-Hand gripper width readout.

Usage:
    LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6 uv run examples/recipes/deployment/real_franka_states.py
"""

import genesis as gs
import numpy as np
import torch
from collections import deque
import matplotlib.pyplot as plt

import eden as en
from eden.envs.base import RLEnvBase
from eden.extensions.deployment.franka_panda import FrankaPandaDeployment
from eden.extensions.deployment.utils.state import RobotStateEntity
from eden.options import EnvOptions
from eden.options.robots.franka_emika_panda import FrankaEmikaPanda
from eden.options.scenes.empty import EmptySceneOptions

PRINT_EVERY = 50  # print comparison every N steps
WARN_TOL = 1e-2  # flag values whose absolute diff exceeds this
PLOT_EVERY = 2  # refresh live plot every N steps
PLOT_WINDOW = 500  # rolling history length (steps)
ROBOT_IP = "192.168.1.1"


# ---------------------------------------------------------------------------
# Pretty-printing helpers
# ---------------------------------------------------------------------------


def _header(title: str) -> str:
    return f"\n  {title}\n  {'-' * len(title)}"


def _compare(
    label: str,
    real: torch.Tensor,
    sim: torch.Tensor,
    names: list[str] | None = None,
) -> str:
    """Return a formatted comparison table string."""
    real_np = real.detach().cpu().numpy().flatten()
    sim_np = sim.detach().cpu().numpy().flatten()
    diff = np.abs(real_np - sim_np)

    lines = [_header(label)]
    if names and len(names) == len(real_np):
        lines.append(f"  {'name':<30s} {'real':>10s} {'sim':>10s} {'diff':>10s}")
        for n, r, s, d in zip(names, real_np, sim_np, diff):
            flag = " <--" if d > WARN_TOL else ""
            lines.append(f"  {n:<30s} {r:>10.4f} {s:>10.4f} {d:>10.6f}{flag}")
    else:
        lines.append(f"  {'idx':<6s} {'real':>10s} {'sim':>10s} {'diff':>10s}")
        for i, (r, s, d) in enumerate(zip(real_np, sim_np, diff)):
            flag = " <--" if d > WARN_TOL else ""
            lines.append(f"  {i:<6d} {r:>10.4f} {s:>10.4f} {d:>10.6f}{flag}")

    lines.append(f"  max |diff| = {diff.max():.6f}")
    return "\n".join(lines)


def _show(label: str, val: torch.Tensor, names: list[str] | None = None) -> str:
    """Return a single-column table (for values with no sim counterpart)."""
    val_np = val.detach().cpu().numpy().flatten()
    lines = [_header(label)]
    if names and len(names) == len(val_np):
        lines.append(f"  {'name':<30s} {'value':>10s}")
        for n, v in zip(names, val_np):
            lines.append(f"  {n:<30s} {v:>10.4f}")
    else:
        for i, v in enumerate(val_np):
            lines.append(f"  {i:<6d} {v:>10.4f}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    en.init(backend=gs.cpu, debug=True)

    env = RLEnvBase(
        env_options=EnvOptions(
            num_envs=1,
            sim_dt=0.02,
            decimation=1,
            show_world_frame=True,
        ),
        scene_options=EmptySceneOptions(
            robot=FrankaEmikaPanda(),
        ),
        show_viewer=True,
    )
    env.build()

    # Keep a reference to the sim entity *before* the deployment swaps it
    # with a RobotStateEntity (so we can read from both).
    sim_robot = env.entities["robot"]

    dep = FrankaPandaDeployment(
        env,
        FrankaPandaDeployment.configure(
            entity_name="robot",
            sync=True,
            connect_timeout_s=10.0,
            robot_ip=ROBOT_IP,
        ),
    )
    dep.connect()
    # Only the sim scene is reset: this example reads the arm (``read_state``) and never sends it a
    # command, so nothing moves.
    env.reset()

    state_entity: RobotStateEntity = dep._state_entity
    device = env.device
    dofs_name = list(sim_robot.dofs_name)

    # Live plot setup: stream EE pose, joint summary, and error/gripper width.
    plt.ion()
    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    fig.canvas.manager.set_window_title("Franka Real vs Sim Live Streams")
    fig.suptitle("Franka Data Streams (Real vs Sim)")
    ax_ee, ax_joint, ax_err = axes

    hist_step: deque[int] = deque(maxlen=PLOT_WINDOW)
    hist_ee_real: deque[np.ndarray] = deque(maxlen=PLOT_WINDOW)
    hist_gripper: deque[float] = deque(maxlen=PLOT_WINDOW)
    hist_joint_real: deque[np.ndarray] = deque(maxlen=PLOT_WINDOW)
    hist_joint_sim: deque[np.ndarray] = deque(maxlen=PLOT_WINDOW)
    hist_err: deque[np.ndarray] = deque(maxlen=PLOT_WINDOW)

    step = 0
    while True:
        robot_state = dep.read_state()

        # --- Mirror real joint state into sim (for viewer) -------------------
        # Franka is fixed-base so we skip set_quat; only joint DOFs are
        # physically meaningful to mirror.
        real_dofs_pos_np = robot_state.dofs_pos
        real_dofs_vel_np = robot_state.dofs_vel

        sim_robot.set_dofs_pos(real_dofs_pos_np)
        sim_robot.set_dofs_vel(real_dofs_vel_np)

        # --- Update the deployment wrapper -----------------------------------
        state_entity.update(robot_state)

        # --- Periodic comparison ---------------------------------------------
        step += 1
        dofs_pos_real = state_entity.get_dofs_pos()
        dofs_pos_sim = sim_robot.get_dofs_pos()
        dofs_vel_real = state_entity.get_dofs_vel()
        dofs_vel_sim = sim_robot.get_dofs_vel()

        # Extra signals only the real robot exposes.
        O_T_EE = robot_state.extra.get("O_T_EE")
        ee_pos_real_np = O_T_EE[:3, 3] if O_T_EE is not None else np.zeros(3)
        gripper_width = robot_state.extra.get("gripper_width", float("nan"))
        gripper_is_grasped = robot_state.extra.get("gripper_is_grasped", False)
        robot_mode = robot_state.extra.get("robot_mode", "?")

        hist_step.append(step)
        hist_ee_real.append(np.asarray(ee_pos_real_np, dtype=np.float64))
        hist_gripper.append(float(gripper_width) if gripper_width == gripper_width else 0.0)
        hist_joint_real.append(
            np.array(
                [
                    float(dofs_pos_real.mean().item()),
                    float(dofs_vel_real.mean().item()),
                ]
            )
        )
        hist_joint_sim.append(
            np.array(
                [
                    float(dofs_pos_sim.mean().item()),
                    float(dofs_vel_sim.mean().item()),
                ]
            )
        )
        hist_err.append(
            np.array(
                [
                    float(torch.max(torch.abs(dofs_pos_real - dofs_pos_sim)).item()),
                    float(torch.max(torch.abs(dofs_vel_real - dofs_vel_sim)).item()),
                ]
            )
        )

        if step % PLOT_EVERY == 0 and len(hist_step) > 1:
            x = np.asarray(hist_step)
            ee_real_np = np.vstack(hist_ee_real)
            gripper_np = np.asarray(hist_gripper)
            joint_real_np = np.vstack(hist_joint_real)
            joint_sim_np = np.vstack(hist_joint_sim)
            err_np = np.vstack(hist_err)

            ax_ee.clear()
            ax_ee.plot(x, ee_real_np[:, 0], "r-", label="ee x")
            ax_ee.plot(x, ee_real_np[:, 1], "g-", label="ee y")
            ax_ee.plot(x, ee_real_np[:, 2], "b-", label="ee z")
            ax_ee.set_ylabel("m")
            ax_ee.set_title("End-Effector Position (base frame)")
            ax_ee.grid(True, alpha=0.3)
            ax_ee.legend(loc="upper left", ncol=3, fontsize=8)

            ax_joint.clear()
            ax_joint.plot(x, joint_real_np[:, 0], "c-", label="mean dof_pos real")
            ax_joint.plot(x, joint_sim_np[:, 0], "c--", label="mean dof_pos sim")
            ax_joint.plot(x, joint_real_np[:, 1], "m-", label="mean dof_vel real")
            ax_joint.plot(x, joint_sim_np[:, 1], "m--", label="mean dof_vel sim")
            ax_joint.set_ylabel("rad / rad/s")
            ax_joint.set_title("Joint Stream Overview (mean across DOFs)")
            ax_joint.grid(True, alpha=0.3)
            ax_joint.legend(loc="upper left", ncol=2, fontsize=8)

            ax_err.clear()
            ax_err.plot(x, err_np[:, 0], label="max |dof_pos diff|")
            ax_err.plot(x, err_np[:, 1], label="max |dof_vel diff|")
            ax_err.plot(x, gripper_np, "k-", alpha=0.6, label="gripper width (m)")
            ax_err.set_xlabel("step")
            ax_err.set_ylabel("abs error / m")
            ax_err.set_title("Real vs Sim Error + Gripper Width")
            ax_err.grid(True, alpha=0.3)
            ax_err.legend(loc="upper left", ncol=2, fontsize=8)

            fig.tight_layout()
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(0.001)

        if step % PRINT_EVERY == 0:
            out: list[str] = []
            out.append(f"\n{'=' * 64}")
            out.append(f"  Convention comparison   (step {step}, mode={robot_mode})")
            out.append(f"{'=' * 64}")

            # 1. End-effector position -----------------------------------------
            ee_pos_real_t = torch.as_tensor(ee_pos_real_np, dtype=torch.float32, device=device)
            out.append(
                _show(
                    "End-Effector Position (m, base frame from O_T_EE)",
                    ee_pos_real_t,
                    names=["x", "y", "z"],
                )
            )

            # 2. Gripper -------------------------------------------------------
            gripper_t = torch.tensor(
                [float(gripper_width) if gripper_width == gripper_width else float("nan")],
                dtype=torch.float32,
                device=device,
            )
            out.append(
                _show(
                    f"Gripper Width  (is_grasped={gripper_is_grasped})",
                    gripper_t,
                    names=["width(m)"],
                )
            )

            # 3. DOF positions --------------------------------------------------
            out.append(
                _compare(
                    "DOF Positions (rad for joints, m for fingers)",
                    dofs_pos_real,
                    dofs_pos_sim,
                    names=dofs_name,
                )
            )

            # 4. DOF velocities -------------------------------------------------
            out.append(
                _compare(
                    "DOF Velocities (rad/s for joints, m/s for fingers)",
                    dofs_vel_real,
                    dofs_vel_sim,
                    names=dofs_name,
                )
            )

            # 5. DOF torques (real only — sim has no matching setter) ----------
            #    For Franka this is tau_J: link-side measured joint torque.
            out.append(
                _show(
                    "DOF Torques (Nm) — real only (tau_J)",
                    state_entity.get_dofs_force(),
                    names=dofs_name,
                )
            )

            print("\n".join(out))

        env.step()


if __name__ == "__main__":
    main()
