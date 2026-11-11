"""Warning helpers: emitting a warning *once* where the same one would otherwise repeat."""

import functools

# Top-level self-import is safe on the package-init chain (same pattern as
# eden.utils.misc): `eden` is already in sys.modules during `import eden`,
# and `en.*` attributes are only accessed at call time.
import eden as en


@functools.lru_cache(maxsize=128)
def warn_once(message: str) -> None:
    """Warn once per distinct text — for a diagnostic a loop would otherwise repeat verbatim.

    Put the varying part (class name, version, path) in the message and the dedup follows it.
    The cache is bounded because it keys on the whole string; ``cache_clear()`` resets it.
    """
    en.logger.warning(message)
