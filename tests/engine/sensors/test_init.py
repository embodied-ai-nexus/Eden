"""``eden.engine.sensors``: importing the package installs its sensors into Genesis' dispatch map."""

from __future__ import annotations

import pytest
from genesis.engine.sensors.sensor_manager import SensorManager

from eden.engine.sensors import (
    ActuationAudio,
    ActuationAudioSensor,
    ContactAudio,
    ContactAudioSensor,
    PianoAudio,
    PianoAudioSensor,
    SpatialAudio,
    SpatialAudioSensor,
)


@pytest.mark.parametrize(
    ("options", "sensor"),
    [
        (SpatialAudio, SpatialAudioSensor),
        (ActuationAudio, ActuationAudioSensor),
        (ContactAudio, ContactAudioSensor),
        (PianoAudio, PianoAudioSensor),
    ],
)
def test_out_of_tree_sensor_registers_without_patching_genesis(options, sensor):
    """An Eden-defined sensor lands in Genesis' own ``SENSOR_TYPES_MAP``, keyed on its Eden options class."""
    assert SensorManager.SENSOR_TYPES_MAP[options] is sensor
