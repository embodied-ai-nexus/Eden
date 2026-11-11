"""Structural check of the Genesis internals an engine addon builds on.

The sensors override ``SimpleSensor``'s per-step hooks and ride ``SharedSensorContext``'s lifecycle, and the
MochiSolver subclasses ``KinematicSolver`` and is hosted through the simulator and the rasterizer; none of it is public
Genesis API. Genesis is pinned to an exact release, so that surface only moves on a deliberate bump; this turns "the pin
moved and an addon quietly stopped working" into an error at import that names the addon.

It checks that members *exist*, which catches a rename or a removal. A module's own ``import`` statements already
cover module-level names, so only class members are declared. Behaviour — for one, that ``SensorManager.step`` calls
``context.update()`` before the per-type loop — is pinned by ``tests/engine/sensors/test_audio.py`` instead.
"""

import genesis as gs


def require_members(addon: str, required: dict[type, tuple[str, ...]]) -> None:
    """Raise naming every class member in ``required`` that this Genesis does not provide.

    Parameters
    ----------
    addon : str
        Module name of the calling addon, so the error says which one needs re-syncing.
    required : dict[type, tuple[str, ...]]
        Genesis class to the names of the members the caller overrides, calls or reads on it.

    Raises
    ------
    ImportError
        If any member is missing. Every miss is listed at once alongside the installed Genesis version, so a pin
        bump yields one report rather than a fix-and-rerun loop.
    """
    missing = [f"{cls.__name__}.{name}" for cls, names in required.items() for name in names if not hasattr(cls, name)]
    if missing:
        raise ImportError(
            f"{addon} is built against Genesis internals that genesis-world {gs.__version__} does not provide: "
            f"{', '.join(missing)}. The Genesis pin in pyproject.toml most likely moved; re-sync {addon} against "
            "the new Genesis, or pin Genesis back."
        )
