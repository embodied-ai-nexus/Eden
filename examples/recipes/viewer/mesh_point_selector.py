"""Select points on a rigid mesh by clicking in the Genesis viewer.

---
tier: recipe
capabilities: [render.viewer, robot.leap_hand]
hardware: gpu
optional_deps: []
run: uv run python examples/recipes/viewer/mesh_point_selector.py
---

Builds an Eden env with a fixed dexterous hand and registers the
``MeshPointSelectorPlugin`` on the native Genesis viewer. Left-*click* the hand's
meshes to toggle selection points -- each point is stored in its link's local
frame so it tracks the link as it moves. Left-*drag* still orbits the camera
(middle drag pans, right drag / scroll zooms). Close the viewer to write the
selections (link index, local position, local normal) to a CSV file.

    uv run python examples/recipes/viewer/mesh_point_selector.py
    uv run python examples/recipes/viewer/mesh_point_selector.py --cpu -o points.csv
    uv run python examples/recipes/viewer/mesh_point_selector.py --visual-geom  # click the visual mesh
"""

from argparse import ArgumentParser

import genesis as gs
import numpy as np

import eden as en
from eden.envs.base import EnvBase
from eden.extensions.visualization.viewer_plugins import MeshPointSelectorPlugin
from eden.options import EnvOptions, SceneOptions
from eden.options.entities import PlaneOptions
from eden.options.materials import RigidMaterialOptions


def main():
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--cpu", action="store_true", help="Run on CPU instead of the GPU")
    parser.add_argument("-o", "--output", default="selected_points.csv", help="CSV path for the selected points.")
    parser.add_argument(
        "--visual-geom",
        action="store_true",
        help="Select points on the visual mesh instead of the collision one.",
    )
    args = parser.parse_args()

    en.init(backend=gs.cpu if args.cpu else gs.gpu, debug=True)

    env = EnvBase(
        # A click is a single world ray, so the selector requires a single environment.
        env_options=EnvOptions(num_envs=1, show_world_frame=True),
        scene_options=SceneOptions(
            floor=PlaneOptions(),
            hand=en.robots.LeapHand_R(
                is_fixed_base=True,
                default_root_pos=(0.0, 0.0, 0.2),
                # Only an entity opting into visual raycasting is hittable with --visual-geom;
                # the collision cast reaches every entity regardless.
                material=RigidMaterialOptions(use_visual_raycasting=args.visual_geom),
                # Draw what is being clicked, so a miss is visibly a miss.
                vis_mode="visual" if args.visual_geom else "collision",
            ),
        ),
        show_viewer=True,
    )

    # A viewer plugin must be registered before the scene is built. `env.scene`
    # is a `gs.Scene` created in `EnvBase.__init__`, so its viewer is available
    # here, before `env.build()`.
    env.scene.viewer.add_plugin(
        MeshPointSelectorPlugin(
            sphere_radius=0.004,
            grid_snap=(-1.0, 0.01, 0.01),  # snap y/z to 1 cm; leave x free
            output_file=args.output,
            use_visual_geom=args.visual_geom,
        )
    )

    env.build()
    env.reset()

    # Frame the small fixed hand -- the default viewer camera sits too far back.
    # set_camera_pose needs float32 arrays (tuples / float64 trip its numba kernel).
    env.scene.viewer.set_camera_pose(
        pos=np.array([0.35, 0.35, 0.45], dtype=np.float32),
        lookat=np.array([0.0, 0.0, 0.2], dtype=np.float32),
    )

    en.logger.info(
        f"Left-click the hand to toggle points; left-drag orbits, middle pans, right/scroll zooms. "
        f"Close the viewer to save selections to '{args.output}'."
    )
    try:
        while env.scene.viewer.is_alive():
            env.step()
    except gs.GenesisException as e:
        # The viewer can be closed between the is_alive() check and scene.step();
        # that surfaces as "Viewer closed." -- exit cleanly, re-raise anything else.
        if "Viewer closed" not in str(e):
            raise


if __name__ == "__main__":
    main()
