"""``eden.options.sensors``: named-link fields are checked against the Genesis sensor they configure."""

from __future__ import annotations

import genesis as gs
import pytest

from eden.options.sensors import SensorOptions


@pytest.mark.parametrize("field", ["filter_link_names", "track_link_names"])
def test_link_names_for_a_sensor_without_the_matching_field_are_rejected(field):
    """An IMU has neither ``filter_link_idx`` nor ``track_link_idx``, so naming links for one is a config error."""
    with pytest.raises(ValueError, match=field.replace("_names", "_idx")):
        SensorOptions(sensor=gs.sensors.IMU(), attach_entity_name="robot", **{field: ["robot"]})
