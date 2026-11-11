"""Simulate heterogeneous YCB object variants across parallel environments.

---
tier: recipe
capabilities: [scene.heterogeneous]
hardware: cpu
optional_deps: []
run: uv run python examples/recipes/heterogeneous/heterogeneous_sim.py -v
---
"""

from argparse import ArgumentParser
import genesis as gs
import eden as en

from eden.options.entities import EntityOptions, GroupedEntityOptions, PlaneOptions
from eden.options.objects.ycb import (
    YCB013Apple,
    YCB072AToyAirplane,
    YCB065CCups,
    YCB077RubiksCube,
)
from eden.envs.base import EnvBase
from eden.options import EnvOptions, SceneOptions, EventManagerOptions
from eden.terms import EventTerm

from eden.managers.terms.events.placement import (
    place_constant_xy,
    place_uniform,
    PlaceGaussian,
)
from eden.constants import EventMode


def main():
    parser = ArgumentParser()
    parser.add_argument("-c", "--cpu", action="store_true", help="Run on CPU instead of the GPU")
    parser.add_argument("-v", "--vis", action="store_true", help="Open the interactive viewer")
    parser.add_argument(
        "--sampler",
        type=str,
        default="uniform",
        choices=["gaussian", "uniform"],
        help="Distribution to sample the per-environment parameters from",
    )
    args = parser.parse_args()

    en.init(backend=gs.cpu if args.cpu else gs.gpu, debug=True)

    if args.sampler == "gaussian":
        sampler = PlaceGaussian.configure(
            mode=EventMode.RESET,
            entity_name="obj",
            support_entity_name="table",
            mu_x=0.0,
            mu_y=-0.5,
            var_x=0.01,
            var_y=0.01,
        )
    elif args.sampler == "uniform":
        sampler = EventTerm.configure(
            mode=EventMode.RESET,
            func=place_uniform,
            params={
                "entity_name": "obj",
                "support_entity_name": "table",
                "samples_per_area": 100,
            },
        )

    env = EnvBase(
        env_options=EnvOptions(
            num_envs=12,
            env_spacing=(1.5, 1.5),
        ),
        scene_options=SceneOptions(
            table=EntityOptions(
                file="objects/tables/work_table/work_table.obj",
                is_fixed_base=True,
                is_support_enabled=True,
                support_sample_method="uniform",  # NOTE: you can also use "grid"
            ),
            obj=GroupedEntityOptions(
                grouped_entities=[
                    YCB077RubiksCube(),
                    YCB013Apple(),
                    YCB072AToyAirplane(),
                    YCB065CCups(),
                ],
                material=None,
                is_fixed_base=False,
            ),
            floor=PlaneOptions(
                is_support_enabled=True,
                default_root_pos=(0.0, 0.0, 1.0),
            ),
        ),
        event_options=EventManagerOptions(
            place_table=EventTerm.configure(
                mode=EventMode.STARTUP,
                func=place_constant_xy,
                params={
                    "entity_name": "table",
                    "support_entity_name": "floor",
                    "pos_xy": (0.0, 0.0),
                },
            ),
            place_plate=sampler,
        ),
        show_viewer=args.vis,
    )

    env.build()
    env.reset()

    for i in range(5 if en.is_smoke() else 1000):
        env.step()

        if i % 20 == 0:
            env.reset()


if __name__ == "__main__":
    main()
