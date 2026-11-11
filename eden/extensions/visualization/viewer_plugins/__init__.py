"""Interactive Genesis viewer plugins for Eden.

Reusable ``genesis.vis.viewer_plugins`` subclasses that add mouse-driven
interaction to the Genesis viewer (point selection, ...). Register one with
``scene.viewer.add_plugin(...)`` before ``scene.build()``.
"""

from eden.extensions.visualization.viewer_plugins.mesh_point_selector import (
    MeshPointSelectorPlugin,
    SelectedPoint,
)

__all__ = ["MeshPointSelectorPlugin", "SelectedPoint"]
