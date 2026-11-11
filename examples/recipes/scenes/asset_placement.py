"""Asset placement tutorial: drop objects onto a multi-link support surface.

---
tier: recipe
capabilities: [scene.placement, robot.franka, term.event]
hardware: cpu
optional_deps: []
run: uv run python examples/recipes/scenes/asset_placement.py
---

Walks through wiring a ``place_uniform`` event term against the Riverway
franka station — a real multi-link asset whose support extraction needs
both the ``support_links_name`` filter (so only the table top is treated as
a placement target, not the auxiliary gantry plates / arm-cart tops) and
the occlusion-aware sample-point filter (so objects can't spawn inside
overhanging collision geometry).

Two run modes:

* default — ``LBMRiverwayStation`` ships with ``support_links_name=
  ["station_table_top"]``, so every placement lands on the table top.
* ``--all-links`` — clears the filter so every horizontal facet of the
  station (table top + gantry plates + arm-cart tops + bottom frame)
  becomes a placement candidate. Useful for verifying the
  ``support_links_name`` plumbing and the occlusion filter together.

Usage::

    uv run python examples/recipes/scenes/asset_placement.py                # default, vis on
    uv run python examples/recipes/scenes/asset_placement.py --cpu --no-vis # headless smoke
    uv run python examples/recipes/scenes/asset_placement.py --all-links    # legacy path
"""

from argparse import ArgumentParser

import genesis as gs

import eden as en
from eden.constants import EventMode
from eden.envs.base import EnvBase
from eden.managers.terms.events.placement import place_uniform
from eden.options import EnvOptions, EventManagerOptions, SceneOptions
from eden.options.entities import BoxOptions, PlaneOptions
from eden.options.objects.lbm_eval import LBMRiverwayStation
from eden.terms import EventTerm


def main():
    parser = ArgumentParser()
    parser.add_argument("-c", "--cpu", action="store_true", help="Run on CPU instead of the GPU")
    parser.add_argument(
        "--no-vis", dest="vis", action="store_false", default=True, help="Run headless instead of opening the viewer"
    )
    parser.add_argument(
        "--all-links",
        action="store_true",
        help="Disable the support_links_name filter — every horizontal surface on the "
        "station (table top + gantry plates + arm-cart tops + bottom frame) becomes a "
        "placement target. The occlusion filter still drops samples whose ray hits an "
        "overhang, so placements are still collision-free.",
    )
    args = parser.parse_args()

    en.init(backend=gs.cpu if args.cpu else gs.gpu, debug=True)

    # LBMRiverwayStation ships with the table-top-only filter. For the
    # contrast demo, clear it (and the matching links_to_keep entry) so
    # every horizontal facet of the station contributes to placement.
    station_kwargs: dict = {}
    if args.all_links:
        station_kwargs = dict(support_links_name=[], links_to_keep=[])

    env = EnvBase(
        env_options=EnvOptions(num_envs=8),
        scene_options=SceneOptions(
            plane=PlaneOptions(),
            station=LBMRiverwayStation(**station_kwargs),
            cube=BoxOptions(size=(0.04, 0.04, 0.04), is_fixed_base=False),
        ),
        event_options=EventManagerOptions(
            place_cube=EventTerm.configure(
                mode=EventMode.RESET,
                func=place_uniform,
                params={
                    "entity_name": "cube",
                    "support_entity_name": "station",
                    "samples_per_area": 100,
                },
            ),
        ),
        show_viewer=args.vis and not en.is_smoke(),
    )
    env.build()
    env.reset()

    cube_pos = env.entities["cube"].get_pos().cpu().numpy()
    print(f"\ncube xyz after place_uniform (num_envs={env.num_envs}, --all-links={args.all_links}):")
    for i, p in enumerate(cube_pos):
        print(f"  env {i}: x={p[0]:+.4f}  y={p[1]:+.4f}  z={p[2]:.4f}")
    print(
        f"  z range: [{cube_pos[:, 2].min():.4f}, {cube_pos[:, 2].max():.4f}]"
        f"  (table-top-only → all near 0.6529; --all-links → multiple plateau heights)\n"
    )

    if args.vis:
        for _ in range(5 if en.is_smoke() else 1000):
            env.scene.step()


if __name__ == "__main__":
    main()
