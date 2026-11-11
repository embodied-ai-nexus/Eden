"""Sensor engine addons: Genesis sensors that stock Genesis does not ship.

Importing this package installs every sensor below. Installation is just the import: a
``Sensor`` subclass registers itself into ``SensorManager.SENSOR_TYPES_MAP`` from
``__init_subclass__`` no matter which package defines it, and ``scene.add_sensor`` resolves
through that map — so an Eden-side sensor is reachable from Eden's own ``SensorsOptions`` with
no change to Genesis. See :mod:`eden.engine` for what belongs here and what does not.

Currently installed, all three stages of one audio pipeline:

- :class:`~eden.engine.sensors.actuation_audio.ActuationAudio` — motor and gearbox noise per
  actuated DOF (synthesis).
- :class:`~eden.engine.sensors.contact_audio.ContactAudio` — taps, ring-down and scraping of one
  link's contacts, voiced by the material of what it touches (synthesis).
- :class:`~eden.engine.sensors.piano_audio.PianoAudio` — the struck strings of a keyboard entity, voiced from
  the motion of its key joints (synthesis).
- :class:`~eden.engine.sensors.audio.AudioContext` — the registry sources publish into
  (propagation).
- :class:`~eden.engine.sensors.spatial_audio.SpatialAudio` — a microphone that renders the
  registry with distance attenuation, propagation delay, and Doppler (rendering).
"""

from eden.engine.sensors.actuation_audio import (
    ActuationAudio,
    ActuationAudioProperties,
    ActuationAudioSensor,
    ActuationVoice,
)
from eden.engine.sensors.audio import AudioContext, AudioSource
from eden.engine.sensors.contact_audio import ContactAudio, ContactAudioProperties, ContactAudioSensor, ContactVoice
from eden.engine.sensors.piano_audio import PianoAudio, PianoAudioSensor, PianoVoice
from eden.engine.sensors.spatial_audio import SpatialAudio, SpatialAudioSensor

__all__ = [
    "ActuationAudio",
    "ActuationAudioProperties",
    "ActuationAudioSensor",
    "ActuationVoice",
    "AudioContext",
    "AudioSource",
    "ContactAudio",
    "ContactAudioProperties",
    "ContactAudioSensor",
    "ContactVoice",
    "PianoAudio",
    "PianoAudioSensor",
    "PianoVoice",
    "SpatialAudio",
    "SpatialAudioSensor",
]
