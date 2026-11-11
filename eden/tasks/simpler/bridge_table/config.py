"""SIMPLER Bridge 'table' task configurations."""

from __future__ import annotations

import eden as en
from eden.terms import MetricTerm
from eden.managers.terms import PlaceFromLayouts
from eden.managers.terms.metrics.proximity import object_a_is_on_b
from eden.options.camera import CamerasOptions
from eden.options.managers.events import EventManagerOptions
from eden.options.managers.metrics import MetricManagerOptions
from eden.options.entities import SceneOptions
from eden.options.objects.simpler import (
    SimplerSpoon,
    SimplerTableCloth,
    SimplerCarrot,
    SimplerPlateObjaverseLarger,
    SimplerGreenCube3cm,
    SimplerYellowCube3cm,
)
from eden.tasks import TASK_REGISTRY
from eden.tasks.simpler.base_config import DEFAULT_ROBOT, SimplerBridgeBaseConfig, simpler_camera

#: SIMPLER's robot base in its own world: (0.147, 0.028), turned to face -x. Eden's world is that base's
#: frame (the arm at the origin, the camera at SIMPLER's base-relative mount), so a SIMPLER point (X, Y) is
#: (0.147 - X, 0.028 - Y) here and a SIMPLER yaw is 180 deg less.
_SIMPLER_BASE_XY = (0.147, 0.028)
#: The middle of SIMPLER's object grids, in SIMPLER's world.
_SIMPLER_GRID_CENTRE = (-0.16, 0.0)


def simpler_layouts(
    half_edges: tuple[float, ...], source_yaws_deg: tuple[float, ...]
) -> list[dict[str, tuple[float, float, float]]]:
    """SIMPLER's table layouts (``PutOnBridgeInSceneEnv`` subclasses) in Eden's frame.

    For each half edge, the source and the target sit on two different corners of a square around the grid
    centre: every ordered pair of corners, so the two never overlap. Each pair comes once per source yaw
    (degrees, SIMPLER's world); the target keeps SIMPLER's identity orientation. SIMPLER enumerates exactly
    these, pair-major, for its episode ids.
    """
    corners = ((-1, -1), (-1, 1), (1, -1), (1, 1))
    layouts = []
    for half in half_edges:
        points = [(_SIMPLER_GRID_CENTRE[0] + cx * half, _SIMPLER_GRID_CENTRE[1] + cy * half) for cx, cy in corners]
        for i, source in enumerate(points):
            for j, target in enumerate(points):
                if i == j:
                    continue
                for yaw in source_yaws_deg:
                    layouts.append({"source": _to_eden(source, yaw), "target": _to_eden(target, 0.0)})
    return layouts


def _to_eden(point: tuple[float, float], yaw_deg: float) -> tuple[float, float, float]:
    x, y = _SIMPLER_BASE_XY[0] - point[0], _SIMPLER_BASE_XY[1] - point[1]
    return (round(x, 6), round(y, 6), yaw_deg % 360.0 - 180.0)  # yaw_deg - 180, wrapped into [-180, 180)


def _place_objects(layouts: list[dict[str, tuple[float, float, float]]]) -> EventManagerOptions:
    return EventManagerOptions(place_objects=PlaceFromLayouts.configure(layouts=layouts, support_entity_name="plane"))


class BridgeTableBaseConfig(SimplerBridgeBaseConfig):
    """Source object onto target object, both free-standing on the table.

    Places nothing itself: each task sets ``event_options = _place_objects(simpler_layouts(...))`` with its
    SIMPLER grid, and ``tests/tasks/simpler/test_bridge_table_config.py`` fails a registered table task
    that does not place both ``source`` and ``target``.
    """

    cameras_options: CamerasOptions = CamerasOptions(
        camera=simpler_camera(
            # SIMPLER's real-evaluation camera for these tasks (WidowX "3rd_view_camera", base-mounted).
            cam_pos=(0.0, -0.16, 0.36),
            cam_lookat=(1.0, 0.393, -0.725),
            background_image="real_inpainting/bridge_real_eval_1.png",
            background_entities=["plane"],
        )
    )

    metric_options: MetricManagerOptions = MetricManagerOptions(
        placement=MetricTerm.configure(
            func=object_a_is_on_b,
            params={
                "entity_a_name": "source",
                "entity_b_name": "target",
            },
        ),
    )


@TASK_REGISTRY.register()
class SimplerPutSpoonOnTableClothConfig(BridgeTableBaseConfig):
    scene_options: SceneOptions = en.scenes.Empty(
        is_support_enabled=True,
        robot=DEFAULT_ROBOT,
        source=SimplerSpoon(),
        target=SimplerTableCloth(),
    )

    # PutSpoonOnTableClothInScene: a 15 cm square; the spoon along SIMPLER's x or turned 90 deg.
    event_options: EventManagerOptions = _place_objects(simpler_layouts((0.075,), (0.0, 90.0)))

    language_instructions: list[str] = ["put spoon on table cloth"]


@TASK_REGISTRY.register()
class SimplerPutCarrotOnPlateConfig(BridgeTableBaseConfig):
    scene_options: SceneOptions = en.scenes.Empty(
        is_support_enabled=True,
        robot=DEFAULT_ROBOT,
        source=SimplerCarrot(),
        target=SimplerPlateObjaverseLarger(),
    )

    # PutCarrotOnPlateInScene: a 15 cm square; the carrot turned 180 or -90 deg.
    event_options: EventManagerOptions = _place_objects(simpler_layouts((0.075,), (180.0, -90.0)))

    language_instructions: list[str] = ["put carrot on plate"]


@TASK_REGISTRY.register()
class SimplerStackGreenCubeOnYellowCubeConfig(BridgeTableBaseConfig):
    scene_options: SceneOptions = en.scenes.Empty(
        is_support_enabled=True,
        robot=DEFAULT_ROBOT,
        source=SimplerGreenCube3cm(),
        target=SimplerYellowCube3cm(),
    )

    # StackGreenCubeOnYellowCubeInScene: a 10 cm and a 20 cm square, cubes square to SIMPLER's axes.
    event_options: EventManagerOptions = _place_objects(simpler_layouts((0.05, 0.1), (0.0,)))

    language_instructions: list[str] = ["put green cube on yellow cube"]
