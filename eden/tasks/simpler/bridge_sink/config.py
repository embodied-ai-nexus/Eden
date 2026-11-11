"""SIMPLER Bridge 'sink' task configurations."""

from __future__ import annotations

import genesis.utils.geom as gu
import numpy as np

import eden as en
from eden.constants import EventMode
from eden.terms import EventTerm, MetricTerm
from eden.managers.terms import place_range_xy
from eden.managers.terms.metrics.containment import object_in_region
from eden.options.camera import CamerasOptions
from eden.options.managers.events import EventManagerOptions
from eden.options.managers.metrics import MetricManagerOptions
from eden.options.entities import SceneOptions
from eden.options.objects.simpler import SimplerSink, SimplerEggplant
from eden.tasks import TASK_REGISTRY
from eden.tasks.simpler.base_config import DEFAULT_ROBOT, SimplerBridgeBaseConfig, simpler_camera

#: The sink's pose and scale, aligned by eye to the backdrop (``real_inpainting/bridge_sink.png``) through the
#: task camera: the mesh's basket compartment under the photo's dish rack, its basin under the photo's basin.
#: Kept level: a few degrees of pitch match the photo's perspective a little better, but tilt the basin floor
#: and the eggplant rolls out of the routine's reach. The layout below follows the pose and the scale.
SINK_POS = (0.305, -0.064, 0.0)
SINK_QUAT = tuple(float(v) for v in gu.xyz_to_quat(np.array([0.0, 0.0, 0.0]), rpy=True, degrees=True))
SINK_SCALE = 1.15

#: The scale the layout offsets below were measured at.
_LAYOUT_SCALE = 1.15


def _on_sink(low: tuple[float, float, float], high: tuple[float, float, float]):
    """A world-frame box from one measured relative to the sink's root (at yaw 0 and ``_LAYOUT_SCALE``).

    The box's centre moves with the sink's pose and scale; it stays axis-aligned, which holds for the few
    degrees of tilt and yaw a backdrop fit needs.
    """
    low, high = np.array(low), np.array(high)
    ratio = SINK_SCALE / _LAYOUT_SCALE
    centre = np.array(SINK_POS) + gu.transform_by_quat((low + high) / 2 * ratio, np.array(SINK_QUAT))
    half = (high - low) / 2 * ratio
    return tuple(float(v) for v in centre - half), tuple(float(v) for v in centre + half)


# The sink's layout, measured off its collision geometry: the basket compartment on the robot's left, the
# basin on its right, a rim between them.
#: The basket's floor box: where an object's centre rests when it lies in the basket.
BASKET_REGION = _on_sink((-0.10, 0.065, 0.06), (0.06, 0.165, 0.11))
#: The basin floor's height.
BASIN_FLOOR_Z = _on_sink((0.0, 0.0, 0.05), (0.0, 0.0, 0.05))[0][2]
#: Where the reset puts an object's centre on the basin floor: its middle, clear of the walls by an
#: eggplant's half length. A third of the eggplants roll 4-6 cm toward -y off their spawn pose, so the
#: range keeps to the basin's +y half: a rolled one stops >= 1.5 cm clear of the wall, room for a finger.
_SPAWN_LOW, _SPAWN_HIGH = _on_sink((-0.06, -0.085, 0.0), (0.0, -0.02, 0.0))
BASIN_SPAWN_X, BASIN_SPAWN_Y = (_SPAWN_LOW[0], _SPAWN_HIGH[0]), (_SPAWN_LOW[1], _SPAWN_HIGH[1])
#: Highest point of the rims between the basin and the basket.
SINK_RIM_Z = _on_sink((0.0, 0.0, 0.12), (0.0, 0.0, 0.12))[0][2]


class BridgeSinkBaseConfig(SimplerBridgeBaseConfig):
    """Source object into the sink basin, which is itself part of the backdrop."""

    scene_options: SceneOptions = en.scenes.Empty(
        is_support_enabled=True,
        robot=DEFAULT_ROBOT,
        target=SimplerSink(default_root_pos=SINK_POS, default_root_quat=SINK_QUAT, scale=SINK_SCALE),
    )

    cameras_options: CamerasOptions = CamerasOptions(
        camera=simpler_camera(
            # SIMPLER's sink camera (WidowXSinkCameraSetup "3rd_view_camera", base-mounted).
            cam_pos=(-0.003, -0.21, 0.39),
            cam_lookat=(0.997, 0.24, -0.66),
            background_image="real_inpainting/bridge_sink.png",
            background_entities=["plane", "target"],
        )
    )

    event_options: EventManagerOptions = EventManagerOptions(
        place_source=EventTerm.configure(
            mode=EventMode.RESET,
            func=place_range_xy,
            params={
                "entity_name": "source",
                "support_entity_name": "target",
                # On the basin floor, in world frame: facet-local ranges on the sink's support surface land
                # behind the basin's back wall (with (0.03, 0.12) x (-0.04, 0.1), 2 of 32 eggplants settled in it).
                "range_x": BASIN_SPAWN_X,
                "range_y": BASIN_SPAWN_Y,
            },
        ),
    )


@TASK_REGISTRY.register()
class SimplerPutEggplantInBasketConfig(BridgeSinkBaseConfig):
    scene_options: SceneOptions = en.scenes.Empty(
        is_support_enabled=True,
        robot=DEFAULT_ROBOT,
        source=SimplerEggplant(),
        target=SimplerSink(
            default_root_pos=SINK_POS, default_root_quat=SINK_QUAT, scale=SINK_SCALE, is_support_enabled=True
        ),
    )

    language_instructions: list[str] = ["put eggplant in basket"]

    metric_options: MetricManagerOptions = MetricManagerOptions(
        placement=MetricTerm.configure(
            func=object_in_region,
            params={
                "entity_name": "source",
                "region_min": list(BASKET_REGION[0]),
                "region_max": list(BASKET_REGION[1]),
            },
        ),
    )
