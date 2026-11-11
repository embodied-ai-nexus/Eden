# Core Utilities

The machinery the rest of Eden is built on: registries, the configurable mixin that binds an options class to its
runtime object, parameter descriptions, multi-GPU helpers, logging and run identity, profiling, and small string and
formatting helpers.

## Package entry point

`import eden` isolates one GPU per DDP rank before CUDA is initialized, so import Eden before `genesis` and `torch`.
It imports neither itself; `eden.init()` does, and every process that builds a scene calls it. The built-in term
modules and the options namespaces (`eden.rewards`, `eden.robots`, `eden.scenes`, ...) load lazily on first use.

::: eden
    options:
      show_root_heading: false
      members:
        - init
        - get_log_dir
        - get_run_id
        - get_run_timestamp

## Registries

A `Registry` maps a name to an object. Keys default to `to_snake_case(obj.__name__)`. A `TermRegistry` (every term
kind) takes no `name=`: a term's key is always its own snake-cased name, because that is what `.configure()` stamps
into `options.name` and what the manager looks it up by. A lookup of a missing key raises `KeyError` with the closest
match and the available keys.

::: eden.utils.registry
    options:
      show_root_heading: false

## Configurable objects

`ConfigurableMixin[Options]` binds a runtime class (a term, a deployment, a dataset, ...) to its options class:
`Cls.configure(**kwargs)` returns validated options, and constructing the object from them mirrors the options onto
it.

::: eden.utils.common
    options:
      show_root_heading: false

## Parameter descriptions

How a configurable's parameters are described, shared by `eden terms`, the [term catalog](../terms/catalog.md) and
the `configure()` errors so all three word a parameter identically.

::: eden.utils.params
    options:
      show_root_heading: false

## Multi-GPU (DDP)

Rank and world-size queries and process-group setup. Also reachable as `eden.distributed`. See the
[multi-GPU guide](../../guides/ddp.md).

::: eden.utils.distributed
    options:
      show_root_heading: false

## Run identity and provenance

::: eden.utils.run_manifest
    options:
      show_root_heading: false

::: eden.utils.distribution
    options:
      show_root_heading: false

## Smoke runs

::: eden.utils.smoke
    options:
      show_root_heading: false

## Profiling

Per-term cost profilers for the step loop.

::: eden.utils.profiling.term_profiler
    options:
      show_root_heading: false

::: eden.utils.profiling.step_profiler
    options:
      show_root_heading: false

::: eden.utils.profiling.core
    options:
      show_root_heading: false

## Strings, formatting and warnings

`term_subject(term)` renders the `"<Class> on entity '<name>'"` subject an entity-bound term opens its errors with
(see [Error Messages](../../guides/error-messages.md)).

::: eden.utils.string
    options:
      show_root_heading: false

::: eden.utils.misc
    options:
      show_root_heading: false

::: eden.utils.format
    options:
      show_root_heading: false

::: eden.utils.warning
    options:
      show_root_heading: false

::: eden.utils.ndarray_field
    options:
      show_root_heading: false

## See also

- [Terms guide](../../guides/terms.md) — registries and `configure()`
- [Error Messages](../../guides/error-messages.md)
