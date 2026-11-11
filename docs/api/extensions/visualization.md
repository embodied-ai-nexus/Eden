# Visualization

`ViserViewer` is a browser-based viewer, built on [Viser](https://viser.studio), for inspecting an env's scene and
link frames outside the Genesis viewer. It can also record a rollout to a self-contained `.html` page (or a `.viser`
file) that replays in a browser. Install it with the `vis` extra (`pip install 'eden-world[vis]'`, or
`uv sync --inexact --extra vis` in a checkout). The [sensors and rendering guide](../../guides/sensors-and-rendering.md)
shows recording; `examples/recipes/rendering/viser_recording.py` is a runnable example.

```python
from eden.extensions.visualization.viser import ViserViewer
```

`eden/extensions/visualization/` has no `__init__.py`, so the docs build finds the class below through its
`eden/extensions` search path, and any module path it shows reads `visualization.viser`. Import it as above.

::: visualization.viser.ViserViewer
    options:
      show_bases: true

## See also

- [Sensors and rendering guide](../../guides/sensors-and-rendering.md)
