"""Load and visualize the LBM-eval Riverway dual-Franka station.

---
tier: recipe
capabilities: [scene.placement, robot.franka, term.event]
hardware: gpu
optional_deps: []
run: uv run python examples/recipes/scenes/lbm_eval.py
---

Builds the bimanual ``LBMEvalSceneOptions`` with two Franka Panda arms mounted on
the station's gantry base plates (left at the +x site, right at the -y site).
The scene re-poses each robot automatically — the user-supplied
``default_root_pos`` / ``default_root_quat`` are overridden inside the scene
constructor so the arms land flush on the plates with the correct yaw.

A handful of LBM-eval graspable objects (apple, cutting board, mug holder,
cereal box) are added to the scene and a ``place_uniform`` event term
samples each one onto the station's table top at reset. The station's
``support_links_name=["station_table_top"]`` (set on ``LBMRiverwayStation``)
restricts placement to the table top, so the event doesn't drop objects
onto auxiliary gantry plates or arm-cart tops.

Run with::

    uv run python examples/recipes/scenes/lbm_eval.py
    uv run python examples/recipes/scenes/lbm_eval.py --no-vis  # headless smoke-test
"""

import argparse

import genesis as gs

import eden as en
from eden.constants import EventMode
from eden.envs.base import EnvBase
from eden.managers.terms.events.placement import place_uniform
from eden.options import EnvOptions, EventManagerOptions
from eden.options.objects.lbm_eval import (
    LBMDorhorsWoodMugHolder,
    LBMDrakeFlakesCerealBox,
    LBMGalaApple,
    LBMOxoPlasticCuttingBoard,
)
from eden.options.scenes.lbm_eval import LBMEvalSceneOptions
from eden.terms import EventTerm


def _place_on_station(entity_name: str) -> EventTerm:
    """``place_uniform`` reset event that samples ``entity_name`` onto the
    station's table-top support polygon.
    """
    return EventTerm.configure(
        mode=EventMode.RESET,
        func=place_uniform,
        params={
            "entity_name": entity_name,
            "support_entity_name": "station",
            "samples_per_area": 100,
        },
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-vis", dest="vis", action="store_false", default=True, help="Run headless instead of opening the viewer"
    )
    args = parser.parse_args()

    en.init(backend=gs.gpu)

    env = EnvBase(
        env_options=EnvOptions(num_envs=1, episode_length_s=100.0),
        scene_options=LBMEvalSceneOptions(
            # left_robot / right_robot default to FrankaEmikaPanda() with a
            # ±π/4 splay on joint1 so the two arms don't collide at init.
            # A few representative graspable objects. Their default_root_pos
            # is irrelevant — the place_uniform event below resamples each
            # of them onto the station's table top at every reset.
            apple=LBMGalaApple(),
            cutting_board=LBMOxoPlasticCuttingBoard(),
            mug_holder=LBMDorhorsWoodMugHolder(),
            cereal_box=LBMDrakeFlakesCerealBox(),
        ),
        event_options=EventManagerOptions(
            place_apple=_place_on_station("apple"),
            place_cutting_board=_place_on_station("cutting_board"),
            place_mug_holder=_place_on_station("mug_holder"),
            place_cereal_box=_place_on_station("cereal_box"),
        ),
        show_viewer=args.vis and not en.is_smoke(),
    )
    env.build()
    env.reset()

    print("Loaded LBM-eval Riverway station with two Franka arms.")
    print(f"  entities: {sorted(env.entities)}")
    print(f"  left robot base:  {env.entities['left_robot'].get_pos().cpu().numpy().flatten()}")
    print(f"  right robot base: {env.entities['right_robot'].get_pos().cpu().numpy().flatten()}")

    horizon = 2000 if not en.is_smoke() else 5
    reset_every = 200
    for i in range(horizon):
        env.scene.step()
        if (i + 1) % reset_every == 0:
            env.reset()


if __name__ == "__main__":
    main()
