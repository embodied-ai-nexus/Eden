"""Real-robot deployment backends.

Importing this package imports each deployment module so that:

1. ``@DEPLOYMENT_REGISTRY.register()`` decorators run and populate the
   string→class registry.
2. ``bind_deployment(robot_cls, deployment_name)`` calls run and populate
   the robot→deployment binding map consulted by
   :func:`eden.extensions.deployment.base.resolve_deployment_for`.

``eden deploy --task <name>`` relies on (2) to pick the right backend
automatically from the task's bound robot. The robot-agnostic MuJoCo
sim-to-sim backend (``eden deploy ... --mujoco``) is looked up by name
instead, bypassing the binding map so any robot can be routed through it.
A composite rig (an arm with a hand welded on) resolves to the
``composite_deployment`` backend, which drives each part through its own binding.
"""

from eden.extensions.deployment.base import (
    DEPLOYMENT_REGISTRY,
    DeploymentBase,
    bind_deployment,
    resolve_deployment_for,
)

# Importing each module triggers its registry + binding side-effects. The
# imports are guarded inside ``try`` so a missing optional dependency (e.g.
# the Unitree SDK) doesn't break the whole package import; the missing
# backend simply won't be available at resolve time.
try:
    from eden.extensions.deployment import unitree_g1  # noqa: F401
except ImportError:
    pass
try:
    from eden.extensions.deployment import unitree_go2  # noqa: F401
except ImportError:
    pass
try:
    from eden.extensions.deployment import unitree_go1  # noqa: F401
except ImportError:
    pass
try:
    from eden.extensions.deployment import robotera_xhand  # noqa: F401
except ImportError:
    pass
try:
    from eden.extensions.deployment import franka_panda  # noqa: F401
except ImportError:
    pass
try:
    from eden.extensions.deployment import limx_oli  # noqa: F401
except ImportError:
    pass
# No optional dependency at module scope (both SDKs are imported at connect), so no ImportError guard:
# one could only hide a real break as a silently missing backend.
from eden.extensions.deployment import wuji_hand  # noqa: F401, E402

try:
    from eden.extensions.deployment import mujoco  # noqa: F401
except ImportError:
    pass
# Drives a composite rig's parts through their own backends above; no SDK of its own.
from eden.extensions.deployment import composite  # noqa: F401, E402

try:
    # The composite rig on MuJoCo (``eden deploy --mujoco`` on a composite); mujoco, as above.
    from eden.extensions.deployment import mujoco_composite  # noqa: F401
except ImportError:
    pass


__all__ = [
    "DEPLOYMENT_REGISTRY",
    "DeploymentBase",
    "bind_deployment",
    "resolve_deployment_for",
]
