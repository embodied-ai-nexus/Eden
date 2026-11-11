"""Keyboard SE(3) teleoperation of a Franka Emika Panda with demo recording, replay, and a Viser viewer.

---
tier: recipe
capabilities: [teleop.keyboard, robot.franka, render.viser]
hardware: cpu
optional_deps: [viser]
run: uv run python examples/recipes/teleop/keyboard/teleoperation_keyboard_viser_recorder.py -v
---
"""

import time
from tqdm import tqdm
from argparse import ArgumentParser

import genesis as gs
import torch

import eden as en
from eden.constants import EventMode
from eden.envs.base import EnvBase
from eden.extensions.visualization.viser import ViserViewer
from eden.terms import EventTerm
from eden.managers.terms.actions.binary_actions import BinaryJointController
from eden.managers.terms.actions.task_space_actions import (
    DifferentialIKController,
    OperationalSpaceController,
)
from eden.managers.terms.recorders import ActionRecorder, InitialStateRecorder
from eden.managers.terms.events.placement import place_constant_xy, place_uniform
from eden.options import ActionManagerOptions, EnvOptions, EventManagerOptions, SceneOptions, RecorderManagerOptions
from eden.options.entities import EntityOptions, PlaneOptions
from eden.options.materials import RigidMaterialOptions
from eden.utils.devices.keyboard import SE3Keyboard


def main():
    """Run keyboard teleoperation with recording and replay.

    Usage:
        python teleoperation_keyboard_viser_recorder.py [-c] [-v] [--action {diffik,osc}]

    Keyboard controls:
        - Numpad/arrow keys + space: SE(3) end-effector and gripper commands.
        - o: start recording and reset to a new episode.
        - x: stop recording and reset environment.
        - r: replay the latest recorded demo.

    GUI controls:
        - Viser "reset" button also triggers replay of the latest demo.
    """
    parser = ArgumentParser()
    parser.add_argument("-c", "--cpu", action="store_true", help="Run on CPU instead of the GPU")
    parser.add_argument("-v", "--vis", action="store_true", help="Open the interactive viewer")
    parser.add_argument(
        "--action",
        type=str,
        default="diffik",
        choices=["diffik", "osc"],
        help="End-effector controller to drive the arm with",
    )
    args = parser.parse_args()

    en.init(backend=gs.cpu if args.cpu else gs.gpu, debug=True)

    env = EnvBase(
        env_options=EnvOptions(num_envs=6, env_spacing=(1.5, 1.5), enable_default_keybinds=False),
        scene_options=SceneOptions(
            table=EntityOptions(
                file="objects/tables/work_table/work_table.obj", is_fixed_base=True, is_support_enabled=True
            ),
            plate=EntityOptions(
                file="objects/plates/86d4d4f8-ab98-4c48-9630-dc5c7710733b.glb",
                is_fixed_base=False,
            ),
            apple=EntityOptions(
                file="objects/apples/013_apple/textured.obj",
                is_fixed_base=False,
            ),
            floor=PlaneOptions(
                is_support_enabled=True,
                default_root_pos=(0.0, 0.0, -0.5),
            ),
            robot=en.robots.FrankaEmikaPanda(
                material=RigidMaterialOptions(
                    gravity_compensation=1.0,
                )
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
                dofs_name=[
                    "finger_joint1",
                    "finger_joint2",
                ],
                open_action=0.04,
                close_action=0.0,
            ),
        ),
        recorder_options=RecorderManagerOptions(
            dataset_export_dir_path="test",
            action_recorder=ActionRecorder.configure(),
            initial_state_recorder=InitialStateRecorder.configure(),
        ),
        show_viewer=args.vis,
    )
    env.build()

    vis = ViserViewer(env, port=8080)
    vis.build()

    env.reset()
    recorder_manager = env.recorder_manager

    en.logger.info(env.action_manager.summary())
    keyboard = SE3Keyboard(key_bindings=SE3Keyboard.wasd_key_bindings())
    key_events = {"replay": False, "record": False, "reset": False}

    def request_replay(on_press):
        if on_press:
            key_events["replay"] = True

    def request_record(on_press):
        if on_press:
            key_events["record"] = True

    def request_reset(on_press):
        if on_press:
            key_events["reset"] = True

    keyboard.add_callback("x", request_reset)
    keyboard.add_callback("r", request_replay)
    keyboard.add_callback("o", request_record)

    demo = None
    replay = False
    replay_env_id = 0
    pbar = None

    def apply_recorded_initial_state(episode, target_env_id: int) -> bool:
        initial_entities = episode.get("initial_state/entities", index=0)
        if initial_entities is None:
            return False

        target_env_idx = torch.tensor([target_env_id], device=env.device)

        for entity_name, entity_state in initial_entities.items():
            entity = env.entities.get(entity_name, None)
            if entity is None:
                continue

            # Restore joint angles first (set_qpos with skip_forward=False triggers a
            # physics forward step). For non-fixed entities the free-joint portion of
            # qpos may be stale if the recording happened after set_pos/set_quat without
            # an intervening physics step, so we override with pos/quat afterwards.
            qpos = entity_state.get("qpos", None)
            if qpos is not None:
                qpos_value = qpos.unsqueeze(0) if getattr(qpos, "ndim", 0) == 1 else qpos
                entity.set_qpos(qpos_value, envs_idx=target_env_idx, zero_velocity=False, skip_forward=False)

            dofs_vel = entity_state.get("dofs_vel", None)
            if dofs_vel is not None:
                dofs_vel_value = dofs_vel.unsqueeze(0) if getattr(dofs_vel, "ndim", 0) == 1 else dofs_vel
                entity.set_dofs_vel(dofs_vel_value, envs_idx=target_env_idx, skip_forward=True)

            # Restore Cartesian root pose last — this reliably sets the world position
            # for free-floating objects (plate, apple) whose pos/quat are authoritative.
            # Skip fixed-base entities without batch_fixed_verts: Genesis requires
            # that flag to set env-specific pos on entities with fixed links.
            if not (entity.is_fixed_base and not entity.morph.batch_fixed_verts):
                pos = entity_state.get("pos", None)
                if pos is not None:
                    pos_value = pos.unsqueeze(0) if getattr(pos, "ndim", 0) == 1 else pos
                    entity.set_pos(pos_value, envs_idx=target_env_idx)

                quat = entity_state.get("quat", None)
                if quat is not None:
                    quat_value = quat.unsqueeze(0) if getattr(quat, "ndim", 0) == 1 else quat
                    entity.set_quat(quat_value, envs_idx=target_env_idx)

        return True

    while True:
        if replay and demo is not None:
            if pbar is None:
                total = demo.size("action")
                pbar = tqdm(total=total, desc="Replaying demo", leave=False)

            action = demo.get("action", sequential=True)

            if action is None:
                replay = False
                demo = None
                env.reset()

                if pbar is not None:
                    pbar.close()
                    pbar = None

                print("Replay finished. Switching to keyboard control.")
                continue

            full_action = torch.zeros(env.num_envs, action.shape[-1], device=env.device)
            full_action[replay_env_id] = action
            action = full_action
            pbar.update(1)

        else:
            if pbar is not None:
                pbar.close()
                pbar = None

            action = keyboard.get_command()
            action = {
                "gripper_controller": action[6:].unsqueeze(0).repeat(env.num_envs, 1).to(env.device),
                "ee_controller": action[:6].unsqueeze(0).repeat(env.num_envs, 1).to(env.device),
            }

        env.step(action)

        gui_info = vis.update()

        if key_events["record"]:
            key_events["record"] = False
            replay = False
            demo = None
            if pbar is not None:
                pbar.close()
                pbar = None
            recorder_manager.start_recording(reset_scene=True)
            print("Started a new recording episode.")

        if key_events["reset"]:
            key_events["reset"] = False
            replay = False
            demo = None
            if recorder_manager.is_recording:
                recorder_manager.end_recording()
                print("Recording saved.")
            env.reset()
            if pbar is not None:
                pbar.close()
                pbar = None
            print("Stopped recording and reset environment.")

        if gui_info["reset"] or key_events["replay"]:
            key_events["replay"] = False
            if recorder_manager.is_recording:
                recorder_manager.end_recording()
                print("Recording saved.")
            if recorder_manager.demo_count <= 0:
                print("No recorded demos available for replay yet.")
                time.sleep(1 / 30)
                continue

            replay = True
            # Load env 0's demo from the latest export batch (each batch has num_envs demos)
            demo_name = f"demo_{recorder_manager.demo_count - env.num_envs}"
            demo = recorder_manager.load_episode(demo_name, device=env.device)

            if demo is None:
                replay = False
                print(f"Failed to load demo: {demo_name}")
                time.sleep(1 / 30)
                continue

            demo_env_id = int(demo.env_id) if demo.env_id is not None else 0
            replay_env_id = demo_env_id if 0 <= demo_env_id < env.num_envs else 0
            env.reset()
            if not apply_recorded_initial_state(demo, replay_env_id):
                print("No recorded initial state found; replaying from reset pose.")

            if pbar is not None:
                pbar.close()
            total = demo.size("action")
            pbar = tqdm(total=total, desc=f"Replaying {demo_name}", leave=False)

            print(f"Loaded demo: {demo_name} (replay env: {replay_env_id})")

        time.sleep(1 / 30)


if __name__ == "__main__":
    main()
