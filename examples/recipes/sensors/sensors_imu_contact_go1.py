"""IMU + per-foot Contact sensors attached to a Go1 quadruped.

---
tier: recipe
capabilities: [sensor.imu, sensor.contact, robot.go1]
hardware: cpu
optional_deps: []
run: uv run python examples/recipes/sensors/sensors_imu_contact_go1.py -v
---
"""

from argparse import ArgumentParser

import genesis as gs

import eden as en
from eden.envs.base import EnvBase
from eden.options import EnvOptions, PlaneOptions, SceneOptions
from eden.options.sensors import SensorOptions, SensorsOptions


def main():
    parser = ArgumentParser()
    parser.add_argument("-c", "--cpu", action="store_true", help="Run on CPU instead of the GPU")
    parser.add_argument("-v", "--vis", action="store_true", help="Open the interactive viewer")
    parser.add_argument("-s", "--steps", type=int, default=100, help="Number of steps to run")
    args = parser.parse_args()

    en.init(backend=gs.cpu if args.cpu else gs.gpu, debug=True)

    env = EnvBase(
        env_options=EnvOptions(
            num_envs=4,
            env_spacing=(1.5, 1.5),
            show_FPS=False,
        ),
        scene_options=SceneOptions(
            floor=PlaneOptions(),
            robot=en.robots.Go1(),
        ),
        sensors_options=SensorsOptions(
            imu=SensorOptions(
                sensor=gs.sensors.IMU(
                    delay=0.01,
                    acc_noise=(0.01, 0.01, 0.01),
                    draw_debug=args.vis,
                ),
                attach_entity_name="robot",
            ),
            contact_fl=SensorOptions(
                sensor=gs.sensors.Contact(draw_debug=args.vis),
                attach_entity_name="robot",
                attach_link_name="FL_foot",
            ),
            contact_fr=SensorOptions(
                sensor=gs.sensors.Contact(draw_debug=args.vis),
                attach_entity_name="robot",
                attach_link_name="FR_foot",
            ),
            contact_rl=SensorOptions(
                sensor=gs.sensors.Contact(draw_debug=args.vis),
                attach_entity_name="robot",
                attach_link_name="RL_foot",
            ),
            contact_rr=SensorOptions(
                sensor=gs.sensors.Contact(draw_debug=args.vis),
                attach_entity_name="robot",
                attach_link_name="RR_foot",
            ),
        ),
        show_viewer=args.vis,
    )

    env.build()

    env.reset()

    step = 0
    while step < args.steps or args.vis:
        env.step()
        step += 1
        if en.is_smoke() and step >= 5:
            break

        print(f"-----step {step}-----")
        for sensor_name in env.sensors:
            print(sensor_name)
            print(env.sensors[sensor_name].read())


if __name__ == "__main__":
    main()
