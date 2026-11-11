"""Mochi Newton's cradle: a five-ball elastic impact chain under implicit integration (BDF2 vs backward Euler).

---
tier: recipe
capabilities: [solver.mochi]
hardware: gpu-fp64
optional_deps: []
run: uv run python examples/recipes/mochi/newton_cradle.py -v
---

Eden port of the Genesis fork's ``examples/mochi/newton_cradle.py``. Five 1 kg, 5 cm-radius balls hang from
1 m strings with a 2 mm surface gap; the first is released from 45 degrees and its momentum should travel
through the pack and eject only the last ball. The penalty contact is elastic (``normal_viscous_damping=0``),
so what decides the fidelity of the chain is the time discretization: backward Euler at plain 60 Hz steps
transfers ~20% of the swing to the last ball, BDF2 with 0.33 ms substeps (the defaults here) ~97%. The
substep count is ``EnvOptions.sim_substeps`` — never ``MochiOptions.dt`` — like every other Eden solver.
Each pendulum is its own single-body MJCF entity: contact between the links of one entity is disabled, so a
cradle loaded as one articulated asset would swing straight through itself.

    uv run python examples/recipes/mochi/newton_cradle.py -v
    uv run python examples/recipes/mochi/newton_cradle.py -v --substeps 1 --integrator backward_euler
"""

import argparse
import math
import tempfile

import genesis as gs
import numpy as np
from _mochi_common import VideoRecorder, init, recording_cameras, write_text

import eden as en
from eden.engine.solvers.mochi.options import MochiOptions
from eden.envs.base import EnvBase
from eden.options import CamerasOptions, EnvOptions, SceneOptions
from eden.options.entities import BaseEntityOptions, BoxOptions, EntityOptions
from eden.options.materials import MochiRigidMaterialOptions

CAM_POS, CAM_LOOKAT = (0.0, -2.35, 0.25), (0.0, 0.0, -0.5)
BALL_X = (-0.204, -0.102, 0.0, 0.102, 0.204)
INITIAL_ANGLE = math.radians(45.0)


def ball_xml(i: int, x: float) -> str:
    return f"""<mujoco model="cradle_ball_{i}">
  <compiler angle="radian" coordinate="local" inertiafromgeom="false"/>
  <worldbody>
    <body name="ball_{i}" pos="{x} 0 0">
      <joint name="string_{i}" type="hinge" axis="0 1 0" damping="0" armature="0" frictionloss="0" limited="false"/>
      <inertial pos="0 0 -1" mass="1" diaginertia="0.001 0.001 0.001"/>
      <geom name="ball_{i}" type="sphere" pos="0 0 -1" size="0.05" rgba="{0.05 + 0.025 * i} {0.2 + 0.15 * i} {0.9 - 0.1375 * i} 1"/>
      <geom name="string_visual_{i}" type="capsule" fromto="0 0 -0.05 0 0 -0.95" size="0.004"
            contype="0" conaffinity="0" density="0" rgba="0.55 0.45 0.35 1"/>
    </body>
  </worldbody>
</mujoco>
"""


def build(
    *,
    substeps: int = 50,
    integrator: str = "bdf2",
    cameras_options: CamerasOptions | None = None,
    show_viewer: bool = False,
    ball_surface=None,
    frame_surface=None,
    **decor: BaseEntityOptions,
) -> EnvBase:
    """Build the cradle scene; ``newton_cradle_nyx.py`` renders this same scene with Nyx.

    Parameters
    ----------
    substeps : int
        Implicit substeps per 60 Hz frame.
    integrator : str
        ``bdf2`` or ``backward_euler``.
    cameras_options : eden.options.CamerasOptions | None
        Cameras to build with the scene.
    show_viewer : bool
        Open the interactive viewer.
    ball_surface : gs.surfaces.Surface | None
        Surface of every pendulum entity; None keeps the MJCF's own ``rgba``.
    frame_surface : gs.surfaces.Surface | None
        Surface of the top rail; None keeps it flat grey.
    **decor : BaseEntityOptions
        Extra visual-only entities (no collision), all on a Mochi material like the rest of the scene.

    Returns
    -------
    eden.envs.base.EnvBase
        The built environment, every pendulum at rest.
    """
    with tempfile.TemporaryDirectory(prefix="eden_mochi_cradle_") as directory:
        balls = {
            f"ball_{i}": EntityOptions(
                file=write_text(directory, f"ball_{i}.xml", ball_xml(i, x)),
                is_articulated=True,
                is_fixed_base=True,
                dofs_name=[f"string_{i}"],
                material=MochiRigidMaterialOptions(friction=0.01),
                surface=ball_surface,
            )
            for i, x in enumerate(BALL_X)
        }
        env = EnvBase(
            env_options=EnvOptions(
                num_envs=1,
                sim_dt=1.0 / 60.0,
                sim_substeps=substeps,
                mochi_options=MochiOptions(integrator=integrator),
            ),
            scene_options=SceneOptions(
                frame=BoxOptions(
                    size=(0.648, 0.03, 0.03),
                    is_fixed_base=True,
                    collision=False,
                    material=MochiRigidMaterialOptions(),
                    surface=frame_surface or gs.surfaces.Default(color=(0.3, 0.3, 0.3)),
                ),
                **balls,
                **decor,
            ),
            cameras_options=cameras_options,
            show_viewer=show_viewer,
        )
        env.build()  # the XML files must still exist here: Genesis parses them at build
    return env


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-v", "--vis", action="store_true", help="Open the interactive viewer")
    parser.add_argument("-c", "--cpu", action="store_true", help="Run on the CPU backend instead of the GPU")
    parser.add_argument("-s", "--steps", type=int, default=600, help="Number of simulation steps")
    parser.add_argument("-r", "--record", action="store_true", help="Record a video of the run")
    parser.add_argument(
        "-o", "--output-dir", type=str, default="renders/mochi", help="Directory the recorded video is written to"
    )
    parser.add_argument("--substeps", type=int, default=50, help="Implicit substeps per 60 Hz frame")
    parser.add_argument(
        "--integrator", choices=("bdf2", "backward_euler"), default="bdf2", help="Time integration scheme"
    )
    args = parser.parse_args()
    init(args)
    steps = 5 if en.is_smoke() else args.steps

    env = build(
        substeps=args.substeps,
        integrator=args.integrator,
        cameras_options=recording_cameras(args, pos=CAM_POS, lookat=CAM_LOOKAT, res=(1280, 720)),
        show_viewer=args.vis,
    )
    balls = [env.entities[f"ball_{i}"] for i in range(len(BALL_X))]
    balls[0].set_dofs_pos(np.array([[INITIAL_ANGLE]]))

    def angles() -> np.ndarray:
        return np.array([float(ball.get_dofs_pos()[0, 0]) for ball in balls])

    recorder = VideoRecorder(env, args, "newton_cradle", fps=60)
    peak = np.abs(angles())
    for _ in range(steps):
        env.scene.step()
        peak = np.maximum(peak, np.abs(angles()))
        recorder.frame()
    recorder.close()

    peak_deg = np.degrees(peak)
    print(f"Peak swing angles [deg]: {np.array2string(peak_deg, precision=2)}")
    print(f"Last-ball transfer ratio: {peak_deg[-1] / math.degrees(INITIAL_ANGLE):.3f}")


if __name__ == "__main__":
    main()
