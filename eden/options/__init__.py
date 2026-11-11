"""Eden options package.

Genesis-native ``Options`` classes are used as-is — Eden does not patch them. They keep
genesis' own ``strict`` / ``extra="forbid"`` validation, so a misspelled kwarg
(``gs.morphs.Box(sizee=1)``) fails loudly, and they carry no live type-metadata extras:
``serialize_obj_with_metadata`` stamps ``_option_module_``/``_option_class_`` at dump time
from ``__module__``/``__class__``, which is all the load path needs to recover the concrete
class.
"""

from eden.options.camera import CameraOptions, CameraOverlayOptions, CamerasOptions
from eden.options.entities import (
    GHOST_DEFAULTS,
    BoxOptions,
    CompositeEntityOptions,
    CylinderOptions,
    EntityOptions,
    GroupedEntityOptions,
    PlaneOptions,
    PrimitiveOptions,
    RodOptions,
    SceneOptions,
    SphereOptions,
    TerrainOptions,
)
from eden.options.envs import CudaGraphOptions, EnvOptions
from eden.options.managers.base import ManagerOptions
from eden.options.managers.actions import ActionManagerOptions, ActionTermOptions
from eden.options.managers.commands import CommandManagerOptions, CommandTermOptions
from eden.options.managers.curricula import (
    CurriculumManagerOptions,
    CurriculumTermOptions,
)
from eden.options.managers.events import EventManagerOptions, EventTermOptions
from eden.options.managers.metrics import MetricManagerOptions, MetricTermOptions
from eden.options.managers.motions import MotionGroupOptions, MotionManagerOptions
from eden.options.managers.observations import (
    NoiseOptions,
    ObservationGroupOptions,
    ObservationManagerOptions,
    ObservationTermOptions,
)
from eden.options.managers.rewards import RewardGroupOptions, RewardManagerOptions, RewardTermOptions
from eden.options.managers.terminations import (
    TerminationManagerOptions,
    TerminationTermOptions,
)
from eden.options.managers.recorders import RecorderManagerOptions, RecorderTermOptions
from eden.options.renderer import BatchRendererOptions, RayTracerOptions
from eden.options.sensors import SensorOptions, SensorsOptions
from eden.options.splat import SplatOptions

__all__ = [
    "ManagerOptions",
    "ActionManagerOptions",
    "ActionTermOptions",
    "CommandManagerOptions",
    "CommandTermOptions",
    "CurriculumManagerOptions",
    "CurriculumTermOptions",
    "CudaGraphOptions",
    "EnvOptions",
    "SplatOptions",
    "CameraOptions",
    "CamerasOptions",
    "CameraOverlayOptions",
    "SensorOptions",
    "SensorsOptions",
    "EntityOptions",
    "GroupedEntityOptions",
    "PlaneOptions",
    "BoxOptions",
    "SphereOptions",
    "CylinderOptions",
    "RodOptions",
    "PrimitiveOptions",
    "TerrainOptions",
    "GHOST_DEFAULTS",
    "SceneOptions",
    "CompositeEntityOptions",
    "EventManagerOptions",
    "EventTermOptions",
    "ObservationManagerOptions",
    "ObservationTermOptions",
    "ObservationGroupOptions",
    "NoiseOptions",
    "RewardGroupOptions",
    "RewardManagerOptions",
    "RewardTermOptions",
    "TerminationManagerOptions",
    "TerminationTermOptions",
    "MetricManagerOptions",
    "MetricTermOptions",
    "RecorderManagerOptions",
    "RecorderTermOptions",
    "MotionGroupOptions",
    "MotionManagerOptions",
    "BatchRendererOptions",
    "RayTracerOptions",
]
