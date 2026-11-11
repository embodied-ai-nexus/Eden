"""Genesis engine components that upstream Genesis does not ship.

A component belongs here when it *is* Genesis code — it subclasses a Genesis engine class and runs inside
``scene.step()`` — but lives in Eden because upstream does not carry it: a sensor from a research fork, say.
Eden-side wrappers of Genesis objects (``eden.entities``) and their options (``eden.options``) do not belong here.

The package is a plain home, not a framework. There is no registry and nothing to install:

- **Sensors** (:mod:`eden.engine.sensors`) need no Genesis change at all. ``Sensor.__init_subclass__`` registers
  any subclass into ``SensorManager.SENSOR_TYPES_MAP`` whichever package defines it — the same seam Genesis' own
  out-of-tree ``gs_nyx_plugin`` uses — and Eden's ``SensorOptions.sensor`` accepts any Genesis sensor options.
  Importing the sensor's module is the whole installation, and a saved config re-imports it by itself through the
  ``_option_module_`` path the config loader already follows. ``import eden`` never imports this package.
- **Solvers** (:mod:`eden.engine.solvers`) are hosted one scene at a time. Genesis builds its solvers from a
  hardcoded list, so a solver here ships a ``hosting`` module that appends it to one scene's simulator and binds,
  on that scene's instances only, what upstream names solver by solver (entity dispatch, rendering, reset). A
  scene that never installs it runs the upstream code.
- **Couplers, constraint solvers** are not hosted here: they change a solver every scene uses, which cannot be
  done without replacing Genesis code for every scene in the process. Pin a Genesis fork that carries it and wrap
  it in ``eden.entities`` / ``eden.options`` instead, or push it upstream.

Rules for a component in this package, each learned the hard way (``docs/guides/engine-addons.md``):

- The options class lives under ``eden.*``, never in a Genesis fork, so a saved config opens on stock Genesis.
- The module docstring states provenance: the upstream source it was ported from, every deviation, and the
  limitation that kept it out of Genesis.
- Genesis internals it relies on beyond its own imports are declared once per module through
  :func:`eden.engine._surface.require_members`, and the *behaviour* it relies on is pinned by a test, so
  a Genesis pin bump fails there rather than degrading a scene silently.
"""
