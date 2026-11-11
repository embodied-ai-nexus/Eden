"""Mochi RJ45 insertion: a plug driven into a socket past a sprung latch, trailing a deformable cable.

---
tier: recipe
capabilities: [solver.mochi]
hardware: gpu-fp64
optional_deps: []
run: uv run python examples/recipes/mochi/contacts_rj45_plug.py -v
---

Eden port of Newton's ``newton.examples contacts_rj45_plug`` (newton-physics/newton, Apache-2.0), with the
cable rebuilt as a real Kirchhoff rod. Newton approximates the cable by a chain of rigid capsules with a
bending stiffness, teleports its first four bodies onto the plug every substep and swing-corrects each
capsule's orientation afterwards so the collision geometry follows the deformed centerline. Mochi assembles
the rod's nodes and segment twists, the two rigid connector bodies, their joints and the penalty contact
into *one* implicit Newton system per step, so the cable is a discrete elastic rod (stretch, bend, twist)
tied to the plug by a stiff constraint inside that same solve -- no teleport kernel, no orientation fix-up,
and no filter for the nodes buried in the plug (they are constrained, not colliding). The rod also collides
with itself, which Newton's capsule chain cannot: the source centerline crosses over its own coil.

Newton drives the plug with a mouse gizmo. Here it hangs off the world by three prismatic joints and the
insertion axis is position-controlled along a ramp, so the run is reproducible and prints a metric: the
plug rides in on a 0.1 mm clearance, the latch is pressed ~0.15 rad into its window, and past ~24.5 mm it
springs back onto the socket's front lip -- the click.

Contact lives at the scale of the latch, not of the scene, so the penalty ramp is shrunk to 0.01 mm /
0.02 mm and the socket's distance field is sampled at 0.1 mm cells, as ``bolt_nut_self_screw`` does for the
thread. ``_rj45.py`` holds the scene and carries the rest of the reasoning.

    uv run python examples/recipes/mochi/contacts_rj45_plug.py -v
    uv run python examples/recipes/mochi/contacts_rj45_plug.py -v --no-cable
"""

import argparse

import _rj45
import genesis as gs
from _mochi_common import VideoRecorder, init, newton_iterations, recording_cameras

import eden as en

CAM_POS, CAM_LOOKAT, CAM_FOV = (0.118, -0.118, 0.076), (-0.015, -0.040, 0.007), 35.0
CAM_RES = (960, 540)
SURFACES = {
    "connector": gs.surfaces.Default(color=(0.88, 0.88, 0.90)),
    "socket": gs.surfaces.Default(color=(0.30, 0.55, 0.75)),
    "cable": gs.surfaces.Default(color=(0.12, 0.12, 0.15)),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-v", "--vis", action="store_true", help="Open the interactive viewer")
    parser.add_argument("-c", "--cpu", action="store_true", help="Run on the CPU backend instead of the GPU")
    parser.add_argument("-s", "--steps", type=int, default=260, help="Number of simulation steps")
    parser.add_argument("-r", "--record", action="store_true", help="Record a video of the run")
    parser.add_argument(
        "-o", "--output-dir", type=str, default="renders/mochi", help="Directory the recorded video is written to"
    )
    parser.add_argument("--no-cable", action="store_true", help="Insert the bare connector, without the cable")
    args = parser.parse_args()
    init(args)
    steps = 5 if en.is_smoke() else args.steps

    env = _rj45.build(
        with_cable=not args.no_cable,
        surfaces=SURFACES,
        cameras_options=recording_cameras(args, pos=CAM_POS, lookat=CAM_LOOKAT, res=CAM_RES, fov=CAM_FOV),
        show_viewer=args.vis,
    )
    connector = env.entities["connector"]

    recorder = VideoRecorder(env, args, "contacts_rj45_plug")
    for i_step in range(steps):
        _rj45.drive(env, i_step)
        env.scene.step()
        if i_step % 20 == 0 or i_step == steps - 1:
            qpos = connector.get_qpos()[0]
            cable_state = "" if args.no_cable else f" {_rj45.cable_metrics(env.entities['cable'])}"
            commanded = -_rj45.PLUG_Y_OFFSET * min(i_step / _rj45.INSERT_STEPS, 1.0)
            print(
                f"step {i_step:4d}: insertion={float(qpos[1]) * 1e3:6.3f} mm (commanded {commanded * 1e3:5.2f}) "
                f"latch={float(qpos[3]):+.4f} rad{cable_state} newton iterations={newton_iterations(env)}"
            )
        recorder.frame()
    recorder.close()


if __name__ == "__main__":
    main()
