"""``eden.engine._surface.require_members``: the import-time check of the Genesis internals an engine addon uses."""

from __future__ import annotations

import pytest

from eden.engine._surface import require_members


def test_missing_genesis_member_names_the_sensor_and_every_miss():
    """A pin bump that renames a hook fails at import, naming the sensor module and all misses at once."""

    class _Moved:
        kept = 1

    require_members("my.sensor", {_Moved: ("kept",)})
    with pytest.raises(ImportError, match=r"my\.sensor.*_Moved\.gone, _Moved\.also_gone"):
        require_members("my.sensor", {_Moved: ("kept", "gone", "also_gone")})
