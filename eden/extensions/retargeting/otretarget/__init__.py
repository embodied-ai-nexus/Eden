"""OTRetarget: joint robot and object motion retargeting via optimal transport (arXiv 2609.36602).

``OTRetargeter`` needs the ``ot-retargeter`` extra (ProxQP and threadpoolctl); the geometry, body-model
and dataset helpers below import without it.
"""

from eden.extensions.retargeting.otretarget.human import HumanSurfaceMotion, smplx_motion
from eden.extensions.retargeting.otretarget.presets import PRESETS, RigPreset, for_body_model
from eden.extensions.retargeting.otretarget.retargeter import InteractionObject, OTRetargeter, RetargetResult

__all__ = [
    "HumanSurfaceMotion",
    "InteractionObject",
    "OTRetargeter",
    "PRESETS",
    "RetargetResult",
    "RigPreset",
    "for_body_model",
    "smplx_motion",
]
