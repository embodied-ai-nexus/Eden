"""Eden config hierarchy (EdenConfig/EdenRLConfig/EdenMimicConfig) and serialization.

- :class:`EdenConfig` — base config for all environments (``EnvBase``).
- :class:`EdenRLConfig` — adds reward, termination, command, curriculum, and runner options.
- :class:`EdenMimicConfig` — adds motion options for motion-imitation tasks.

Root config classes use Pydantic ``extra="forbid"`` (so misspelled fields raise), while
inner ``ManagerOptions`` / ``SceneOptions`` keep ``extra="allow"`` to hold dynamic
term/entity children. Saved configs use a ``_meta``/``config`` envelope; nested Options
recover their type from ``_option_module_`` / ``_option_class_`` keys.

The root config recovers its type the same way: :meth:`EdenConfig.load_from_file` reads the
saved class out of ``_meta`` and reconstructs *that* class, so a generic
``EdenConfig.load_from_file(path)`` returns the concrete task config it was saved from.
The requested class still constrains the result — the saved class must be a subclass of it —
and files with no class metadata load as the requested class. Round-tripping a config through
save/load is therefore type-preserving and idempotent (pinned by
``tests/utils/test_configs.py::test_every_task_config_round_trips_to_a_fixed_point``).
"""

import json
import os
from enum import Enum
from functools import lru_cache
from types import UnionType
from typing import Union, get_args, get_origin

import numpy as np
import torch
import yaml
from genesis.options.renderers import RendererOptions
from pydantic import BaseModel, TypeAdapter, ValidationError

import eden as en
from eden.options import (
    ActionManagerOptions,
    CameraOptions,
    CamerasOptions,
    CommandManagerOptions,
    CurriculumManagerOptions,
    EnvOptions,
    EventManagerOptions,
    MetricManagerOptions,
    MotionManagerOptions,
    ObservationManagerOptions,
    RecorderManagerOptions,
    RewardManagerOptions,
    SceneOptions,
    SensorsOptions,
    TerminationManagerOptions,
)
from eden.options._metadata import OPTION_METADATA_KEYS
from eden.options.options import ConfigurableOptions, Options
from eden.options.splat import NyxSplatOptions, SplatOptions
from eden.utils.misc import describe_provenance_change, get_eden_provenance, get_package_provenance


def _is_namedtuple_class(cls: type) -> bool:
    return bool(isinstance(cls, type) and cls is not tuple and issubclass(cls, tuple) and hasattr(cls, "_fields"))


def read_config_file(path: str) -> dict:
    """Read a ``.json`` / ``.yaml`` / ``.yml`` config file into a raw dict.

    The single file-reading primitive behind :meth:`EdenConfig.load_from_file` and
    :meth:`EdenConfig.with_overrides_from_file`, so both dispatch on the extension the
    same way and report an unsupported one identically.

    Parameters
    ----------
    path: str
        The path to read.

    Returns
    -------
    dict
        The file's contents, exactly as parsed — envelope and all, if it has one.

    Raises
    ------
    ValueError
        If the extension is not ``.json``, ``.yaml``, or ``.yml``, or if the file does not
        hold a mapping (a bare scalar or list is never a config).
    """
    path_lower = path.lower()
    if path_lower.endswith(".json"):
        with open(path, "r") as f:
            data = json.load(f)
    elif path_lower.endswith((".yaml", ".yml")):
        with open(path, "r") as f:
            data = yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported config file extension: {path}. Use .json, .yaml, or .yml")

    if not isinstance(data, dict):
        raise ValueError(f"{path} does not hold a config mapping — parsed as {type(data).__name__}.")
    return data


def is_full_config(data: dict) -> bool:
    """Whether ``data`` is a *saved* config (replaces a base) rather than a partial override.

    A save from :meth:`EdenConfig.save_as_file` carries its class and version tags — under a
    ``_meta`` envelope, or (legacy flat saves) alongside the fields. A hand-written partial
    override never does, which is what makes the two forms distinguishable by content instead
    of by flag: ``eden train --config`` reads this to decide whether the file replaces the
    task's config or layers onto it.
    """
    if "_meta" in data and "config" in data:
        return True
    return any(key in data for key in (*OPTION_METADATA_KEYS, "eden_version", "genesis_version"))


def saved_config_qualname(data: dict) -> str | None:
    """The ``module.Class`` a saved config records, or ``None`` when it records no class.

    Reads the same tags :meth:`EdenConfig.load_from_data` reconstructs from, without importing
    anything — so a caller can compare identities (``eden train --config``, checking that a
    saved config is the one its ``--task`` builds) and report a mismatch by name even when the
    saved class is not importable here.

    Parameters
    ----------
    data: dict
        A saved config's contents, in either the ``_meta``/``config`` envelope or the legacy
        flat layout. Not modified.

    Returns
    -------
    str | None
        The dotted qualname, or ``None`` for a file with no class metadata (legacy saves and
        hand-written full configs, which load as whatever class is asked for).
    """
    meta = data.get("_meta") if isinstance(data.get("_meta"), dict) and "config" in data else data
    module_name, class_name = (meta.get(key) for key in OPTION_METADATA_KEYS)
    return f"{module_name}.{class_name}" if module_name and class_name else None


def overrides_from_str(text: str) -> dict:
    """Parse one ``--override`` argument — an inline YAML (or JSON) mapping — into a dict.

    YAML flow style keeps the shell-facing form short and quote-free
    (``'env_options: {num_envs: 64}'``); JSON is a subset, so the explicit form works too.
    Note that PyYAML implements YAML 1.1 implicit typing, where ``off`` / ``no`` / ``yes``
    parse as booleans — quote any value that should stay a string.

    Parameters
    ----------
    text: str
        The inline mapping.

    Returns
    -------
    dict
        The parsed override data, ready for :meth:`EdenConfig.with_overrides_from_dict`.

    Raises
    ------
    ValueError
        If ``text`` is not parseable, or parses to something other than a mapping — the
        common case being a file path, which belongs to ``--config``.
    """
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise ValueError(f"Could not parse override {text!r} as an inline YAML mapping: {e}") from e

    if not isinstance(data, dict):
        raise ValueError(
            f"Override {text!r} is not a mapping — it parsed as {type(data).__name__}. Write it as "
            f"'env_options: {{num_envs: 64}}'. To load overrides from a .json/.yaml file, use --config."
        )
    return data


class EdenConfig(ConfigurableOptions):
    """
    Base configuration for all Eden environments (EnvBase).

    Parameters
    ----------
    env_options: EnvOptions
        Environment options (sim dt, num_envs, etc.).
    scene_options: SceneOptions
        Scene options (entities, attachments).
    observation_options: ObservationManagerOptions
        Observation manager options.
    action_options: ActionManagerOptions
        Action manager options.
    event_options: EventManagerOptions
        Event manager options.
    cameras_options: CamerasOptions
        Cameras options.
    sensors_options: SensorsOptions
        Sensors options.
    metric_options: MetricManagerOptions
        Metric manager options.
    recorder_options: RecorderManagerOptions
        Recorder manager options.
    renderer_options: RendererOptions
        Renderer options.
    language_instructions: list[str]
        Language instructions for the task.
    """

    env_options: EnvOptions = EnvOptions()
    scene_options: SceneOptions = SceneOptions()
    observation_options: ObservationManagerOptions = ObservationManagerOptions()
    action_options: ActionManagerOptions | None = ActionManagerOptions()
    event_options: EventManagerOptions = EventManagerOptions()
    cameras_options: CamerasOptions | None = CamerasOptions()
    sensors_options: SensorsOptions | None = SensorsOptions()
    metric_options: MetricManagerOptions | None = MetricManagerOptions()
    recorder_options: RecorderManagerOptions | None = RecorderManagerOptions()
    renderer_options: RendererOptions | None = RendererOptions()

    language_instructions: list[str] = []

    def check_bindings(self) -> None:
        """Raise if a field this config leaves for a :class:`~eden.tasks.registry.TaskMod` to fill is unset.

        :meth:`eden.envs.base.EnvBase.from_config` calls this before it constructs the env. A task
        whose robot (or data) binding lives in a TaskMod ships the robot-specific fields as empty
        sentinels, and a config that reaches an env without the mod -- ``TASK_REGISTRY.get(name)()``,
        a subclass registered with ``modifiers=()``, a subclass of an unregistered base -- otherwise
        fails deep in ``env.build()`` on a pattern error that names neither the field nor the mod,
        or does not fail at all where an empty link list resolves to no links. The base checks
        nothing; a config with mod-bound fields overrides this to name them and the mod
        (``VelocityTrackingBaseConfig`` is the worked example).
        """

    def recording_camera(self) -> CameraOptions:
        """The camera a recording of this task renders, added as ``rec`` by :meth:`add_recording_camera`.

        ``eden inference --rec`` adds it; so does any script that records the task and calls
        :meth:`add_recording_camera`. A method rather than a ``cameras_options`` entry so that only a recording run
        builds it: a camera in the training config costs every training run its setup (and whatever its overlays
        allocate) though nothing renders it, and a checkpoint replays the config it was trained with. The default
        follows the ``robot`` entity from 1.5 m out, which suits a locomoting robot; a task whose subject is
        elsewhere, or whose recording paints state the scene does not render (a camera overlay,
        :mod:`eden.entities.camera_overlays`), overrides it.
        """
        return CameraOptions(cam_pos=(1.5, 1.0, 2.0), cam_lookat=(0.0, 0.0, 1.0), follow_entity_name="robot")

    def add_recording_camera(self) -> None:
        """Add :meth:`recording_camera` to this config as its ``rec`` camera, in place, unless it already has one.

        A ``rec`` camera the config already has (a ``cameras_options: {rec: {...}}`` override) is kept: replacing it
        would silently discard the override. It replaces the task's camera whole, overlays included, so the overlays
        it lacks are named in a warning.
        """
        if self.cameras_options is None:
            self.cameras_options = CamerasOptions()
        declared = getattr(self.cameras_options, "rec", None)
        if declared is None:
            self.cameras_options.rec = self.recording_camera()
            return
        # By value: an overlay read back from a saved config is equal to the one written (its parameters are given their
        # declared types again, so a tuple is not a list).
        dropped = [overlay for overlay in self.recording_camera().overlays if overlay not in declared.overlays]
        if dropped:
            en.logger.warning(
                f"The config declares its own `rec` camera, which replaces {type(self).__name__}.recording_camera() "
                f"whole, so the recording leaves out that camera's overlays {[overlay.name for overlay in dropped]} "
                "(task state the scene does not render, such as ink). To keep them, add them to "
                "cameras_options.rec.overlays, e.g. the entries of config.recording_camera().overlays."
            )

    def _serialize(self) -> dict:
        """
        Serialize the config with metadata envelope.

        The ``_meta`` key is placed under a dedicated namespace to avoid collisions with config field names.
        """
        config_data = {}
        for key in type(self).model_fields:
            config_data[key] = serialize_obj_with_metadata(getattr(self, key))
        return {
            "_meta": {
                "eden": get_eden_provenance(),
                "genesis": get_package_provenance("genesis-world"),
                "_option_module_": self.__module__,
                "_option_class_": self.__class__.__name__,
            },
            "config": config_data,
        }

    def save_as_file(self, path: str, lock_config: bool = True):
        """
        Save the config as either JSON or YAML file with metadata envelope.

        Parameters
        ----------
        path: str
            The path to save the config.
        lock_config: bool
            Whether to lock the config after saving to prevent further modifications.
        """
        file_type = path.split(".")[-1].lower()
        if file_type not in ["json", "yaml", "yml"]:
            raise ValueError(f"Unsupported config file extension: {file_type}. Use .json, .yaml, or .yml")

        result = self._serialize()

        if lock_config:
            self.lock()

        dirpath = os.path.dirname(path)
        if dirpath:
            os.makedirs(dirpath, exist_ok=True)
        with open(path, "w") as f:
            if file_type == "json":
                json.dump(result, f, indent=4)
            else:
                yaml.safe_dump(result, f, sort_keys=False)

    def save_as_json(self, path: str | None = None, lock_config: bool = True):
        """
        Save the config as a JSON file with metadata envelope.

        Parameters
        ----------
        path: str | None
            The path to save the config. If not provided, the config will be saved to the log directory.
        lock_config: bool
            Whether to lock the config after saving to prevent further modifications.

        Raises
        ------
        ValueError
            If ``path`` does not end in ``.json`` (case-insensitive). Nothing is written and the
            config is left unlocked.
        """
        path = path or f"{en.log_dir}/config.json"
        # Not an ``assert``: ``python -O`` strips those, and ``save_as_file`` dispatches on the
        # suffix — so ``save_as_json("cfg.txt")`` would silently write a file whose extension
        # ``load_from_file`` refuses. Matched case-insensitively, as ``save_as_file`` is.
        if not path.lower().endswith(".json"):
            raise ValueError(
                f"save_as_json expects a path ending in '.json', got '{path}'. "
                f"Use save_as_yaml for YAML, or save_as_file to dispatch on the extension."
            )
        self.save_as_file(path, lock_config)

    def save_as_yaml(self, path: str | None = None, lock_config: bool = True):
        """
        Save the config as a YAML file with metadata envelope.

        Parameters
        ----------
        path: str | None
            The path to save the config. If not provided, the config will be saved to the log directory.
        lock_config: bool
            Whether to lock the config after saving to prevent further modifications.

        Raises
        ------
        ValueError
            If ``path`` does not end in ``.yaml`` or ``.yml`` (case-insensitive). Nothing is
            written and the config is left unlocked.
        """
        path = path or f"{en.log_dir}/config.yaml"
        # See ``save_as_json`` — same ``-O``-stripped assert, same silent-wrong-extension outcome.
        if not path.lower().endswith((".yaml", ".yml")):
            raise ValueError(
                f"save_as_yaml expects a path ending in '.yaml' or '.yml', got '{path}'. "
                f"Use save_as_json for JSON, or save_as_file to dispatch on the extension."
            )
        self.save_as_file(path, lock_config)

    @classmethod
    def load_from_file(cls, path: str) -> "EdenConfig":
        """
        Load the config from a JSON or YAML file, reconstructing its saved class.

        The class the file was saved from is recorded in its metadata, so
        ``EdenConfig.load_from_file("logs/.../config.json")`` returns the concrete task config
        (``ReacherConfig``) rather than a bare ``EdenConfig`` — which is what makes a generic
        loader like ``eden train --config <a saved config.json>`` work at all: a base
        ``EdenConfig`` has nowhere to put ``reward_options`` and rejects the file outright.

        ``cls`` still constrains the result: the saved class must be ``cls`` or a subclass of it,
        so ``EdenMimicConfig.load_from_file(<a non-mimic config>)`` fails loud instead of quietly
        handing back something that is not an ``EdenMimicConfig``. Files with no class metadata
        (legacy saves) load as ``cls``.

        Parameters
        ----------
        path: str
            The path to load the config from.

        Returns
        -------
        EdenConfig
            An instance of the saved class (a ``cls`` instance when the file records no class).
        """
        return cls.load_from_data(read_config_file(path), path)

    @classmethod
    def load_from_data(cls, data: dict, source: str = "<data>") -> "EdenConfig":
        """Reconstruct a saved config from already-read data — see :meth:`load_from_file`.

        Split out so a caller that has to *inspect* the data before deciding what to do with it
        (``eden train --config``, which reads :func:`is_full_config` to tell a full save from a
        partial override) can reconstruct without reading the file a second time.

        Parameters
        ----------
        data: dict
            The file's contents, in either the ``_meta``/``config`` envelope or the legacy flat
            layout. Consumed destructively — pass a copy to keep it.
        source: str
            Where ``data`` came from, for error messages.

        Returns
        -------
        EdenConfig
            An instance of the saved class (a ``cls`` instance when the data records no class).
        """
        meta = data.pop("_meta", None)
        if meta is not None and "config" in data:
            # Envelope format: {"_meta": {...}, "config": {...}}
            data = data["config"]
        else:
            # Legacy flat format: metadata sat alongside the fields. It carries the same
            # class tags, so it resolves the saved class the same way.
            meta = {key: data.pop(key, None) for key in (*OPTION_METADATA_KEYS, "eden_version", "genesis_version")}

        options = to_options(data, cls._saved_config_class(meta, source))

        for key, name, current in (
            ("eden", "Eden", get_eden_provenance()),
            ("genesis", "Genesis", get_package_provenance("genesis-world")),
        ):
            if change := describe_provenance_change(meta.get(key), current):
                en.logger.warning(f"{name} differs from the one that saved {source}: {change}.")

        return options

    @classmethod
    def _saved_config_class(cls, meta: dict, path: str) -> type["EdenConfig"]:
        """The config class ``path`` was saved from, validated against the requested ``cls``.

        Falls back to ``cls`` when the file records no class — pre-metadata saves, and
        hand-written config files, must keep loading.

        Raises
        ------
        ImportError
            If the recorded class cannot be imported. Reconstructing a *different* class in its
            place would drop every field only the saved one declares — and, for a subclass that
            merely overrides defaults, would silently drop its validators and ``model_post_init``
            instead, producing a config that looks right and behaves differently. Neither is worth
            the narrow compatibility win, so this fails loud with the qualname and a way forward —
            deliberately even for an overrides-only subclass, defined somewhere unimportable, loaded
            via its base class. Do not "fix" it with a fallback.
        TypeError
            If the recorded class is not a subclass of ``cls``. Loading it anyway would return
            an object the caller's own type annotation says is impossible.
        """
        module_name, class_name = (meta.get(key) for key in OPTION_METADATA_KEYS)
        if not module_name or not class_name:
            return cls
        if (module_name, class_name) == (cls.__module__, cls.__name__):
            # Already the requested class. Short-circuit rather than re-import it: a config class
            # defined inside a function or a ``python my_task.py`` ``__main__`` is not importable
            # by name, and refusing to load it *with itself* would help nobody.
            return cls

        qualname = f"{module_name}.{class_name}"
        try:
            saved_cls = get_options_class_from_qualname(qualname)
        except (ImportError, AttributeError) as e:
            hint = (
                "it was defined in a script run directly (`__main__`), so load the file from that "
                "same script, or move the class into an importable module"
                if module_name == "__main__"
                else f"import the module that defines it before loading, or load with {class_name} itself"
            )
            raise ImportError(f"{path} was saved from '{qualname}', which cannot be imported ({e}) — {hint}.") from e

        if not (isinstance(saved_cls, type) and issubclass(saved_cls, cls)):
            raise TypeError(
                f"{path} was saved from '{qualname}', which is not a subclass of the requested "
                f"{cls.__name__}. Load it with {class_name} (or a base class of it, e.g. EdenConfig)."
            )
        return saved_cls

    def with_overrides_from_file(self, path: str) -> "EdenConfig":
        """Create a new config by merging this config with values from a partial config file.

        The current config is not modified.

        Parameters
        ----------
        path: str
            The path to the override file (.json, .yaml, or .yml).

        Returns
        -------
        EdenConfig
            A new config instance based on this config with overrides applied.
        """
        return self.with_overrides_from_dict(read_config_file(path))

    def with_overrides_from_dict(self, data_dict: dict) -> "EdenConfig":
        merged = _deep_merge(self._to_dict(), data_dict)
        return to_options(merged, self)

    def _to_dict(self) -> dict:
        """Export config to a dict with option metadata (same shape as saved JSON/YAML)."""
        result = {}
        for key in type(self).model_fields:
            result[key] = serialize_obj_with_metadata(getattr(self, key))
        return result


class EdenRLConfig(EdenConfig):
    """
    Configuration for RL environments (RLEnvBase).

    Extends EdenConfig with reward, termination, command, curriculum,
    and runner options needed for reinforcement learning.

    Parameters
    ----------
    reward_options: RewardManagerOptions
        Reward manager options.
    termination_options: TerminationManagerOptions
        Termination manager options.
    command_options: CommandManagerOptions
        Command manager options.
    curriculum_options: CurriculumManagerOptions
        Curriculum manager options.
    runner_options: ConfigurableOptions | None
        Runner/training algorithm options. Framework-agnostic base type;
        use RslRlOnPolicyRunnerOptions, RlGamesPpoRunnerOptions, etc.
    """

    reward_options: RewardManagerOptions | None = RewardManagerOptions()
    termination_options: TerminationManagerOptions | None = TerminationManagerOptions()
    command_options: CommandManagerOptions | None = CommandManagerOptions()
    curriculum_options: CurriculumManagerOptions | None = CurriculumManagerOptions()
    runner_options: ConfigurableOptions | None = None


def _with_splat(camera: CameraOptions) -> CameraOptions:
    """``camera`` with a :class:`~eden.entities.camera_overlays.SplatComposite` first in its overlays, showing the splat
    behind empty space. A camera shows a splat env's splat only when it declares one.
    """
    # Deferred: the overlays module pulls in the camera-overlay stack (PIL, the overlay registry), which only a
    # splat recording camera needs.
    from eden.entities.camera_overlays import SplatComposite

    return camera.model_copy(update={"overlays": [SplatComposite.configure(), *camera.overlays]})


class EdenSplatConfig(EdenConfig):
    """
    Configuration for non-RL gaussian-splat environments (:class:`~eden.envs.splat.SplatEnvBase`).

    A sibling of :class:`EdenRLConfig` (like :class:`EdenNyxSplatConfig`): a splat scene with no
    reward/termination/command/curriculum surface — viewer / teleop / inference / data collection.
    Whether the splat is rendered per-env or broadcast is a *camera* choice, not a config-leaf one
    (see :class:`~eden.options.splat.SplatOptions`), so this leaf covers both single-env and batched
    non-RL rendering. For RL training on a splat scene, use :class:`EdenSplatRLConfig`.

    ``splat_options`` is a root field rather than a member of ``scene_options`` because it selects
    the environment class — :func:`~eden.envs.base.env_class_for_config` dispatches on this config
    type, and a field buried in the scene would leave the dispatcher unable to see it.

    Parameters
    ----------
    splat_options: SplatOptions | None
        Splat renderer options for photorealistic background rendering (shared / per-env Gaussians,
        minibatch, camera mode).
    """

    splat_options: SplatOptions | None = None

    def recording_camera(self) -> CameraOptions:
        """:meth:`EdenConfig.recording_camera` with the splat composited behind empty space. A task whose splat
        stands in for a sim surface (a table plane) overrides it to name that entity in the ``SplatComposite``.
        """
        return _with_splat(super().recording_camera())


class EdenSplatRLConfig(EdenRLConfig):
    """
    Configuration for RL gaussian-splat environments (:class:`~eden.envs.splat.SplatRLEnvBase`).

    The RL counterpart of :class:`EdenSplatConfig`: the same gsplat splat background plus the full
    reward/termination/command/curriculum/runner surface of :class:`EdenRLConfig`. The env↔env
    split mirrors ``EnvBase`` / ``RLEnvBase``; the single-vs-batched rendering choice is orthogonal
    and lives on the camera (:class:`~eden.options.splat.SplatOptions.camera_mode`).

    Parameters
    ----------
    splat_options: SplatOptions | None
        Splat renderer options for photorealistic background rendering (shared / per-env Gaussians,
        minibatch, camera mode).
    """

    splat_options: SplatOptions | None = None

    def recording_camera(self) -> CameraOptions:
        """:meth:`EdenConfig.recording_camera` with the splat composited behind empty space, as for
        :class:`EdenSplatConfig`.
        """
        return _with_splat(super().recording_camera())


class EdenNyxSplatConfig(EdenConfig):
    """
    Configuration for Nyx path-traced splat environments (NyxSplatEnvBase).

    A sibling of :class:`EdenSplatConfig`, mirroring the env classes: both render splats, but
    through different backends with different options types. (Nyx has no batched path.)

    Parameters
    ----------
    splat_options: NyxSplatOptions | None
        Nyx splat renderer options for path-traced background rendering.
    """

    splat_options: NyxSplatOptions | None = None


class EdenMimicConfig(EdenRLConfig):
    """
    Configuration for motion imitation environments (MimicRLEnvBase).

    Extends EdenRLConfig with motion manager options for motion imitation tasks.

    Parameters
    ----------
    motion_options: MotionManagerOptions
        Motion manager options.
    """

    motion_options: MotionManagerOptions | None = None


# ------------------------------------------------------------
# --------------------- Helper Functions ---------------------
# ------------------------------------------------------------


def _deep_merge(base: dict, override: dict) -> dict:
    """Merge override into base recursively. Joins dicts but not lists.

    A nested dict whose serialized Options type (``_option_class_`` / ``_option_module_``)
    differs from the base's is a *type-changing* override: its keys describe a different
    schema and cannot be merged onto the old type's keys — the join would leave stale
    fields from the base type that the new type rejects on load (e.g. overriding a
    ``GroupedEntityOptions`` scene entity with a ``BoxOptions`` would keep ``grouped_entities``,
    which ``BoxOptions`` forbids). Replace wholesale in that case.
    """
    result = dict(base)
    for k, v in override.items():
        if (
            k in result
            and isinstance(result[k], dict)
            and isinstance(v, dict)
            and not _option_type_changed(result[k], v)
        ):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def _option_type_changed(base: dict, override: dict) -> bool:
    """True if both dicts tag a concrete Options type and those tags differ.

    A partial override that omits the type tags (same-type field tweak) merges key by key;
    only an override that explicitly names a *different* ``_option_class_`` / ``_option_module_``
    triggers a replace.
    """
    return any(
        base.get(key) is not None and override.get(key) is not None and base[key] != override[key]
        for key in OPTION_METADATA_KEYS
    )


def serialize_obj_with_metadata(obj):
    """Recursively serialize an object, capturing Options type metadata.

    Captures ``_option_module_`` and ``_option_class_`` metadata from Options
    instances. This works by using ``model_dump()`` for data but accessing actual
    Option objects to extract their metadata.

    NamedTuple values (e.g. Genesis ``TemperatureProperties``) subclass ``tuple`` and are
    serialized as plain lists of fields in field order; ``to_options`` reconstructs them
    using the surrounding Pydantic field annotations on load.
    """
    if isinstance(obj, Options):
        # A field left at a None default is dumped as absent, not as an explicit None: loading would
        # otherwise mark it set, and Genesis reads set-ness — a solver whose options merely list `dt`
        # among `model_fields_set` divides the sim dt by it (`MochiOptions(dt=None)` would crash the build).
        unset_none = {name for name in type(obj).model_fields if name not in obj.model_fields_set}
        unset_none = {name for name in unset_none if getattr(obj, name, None) is None}
        result: dict = obj.model_dump(exclude=unset_none)

        # Stamp type metadata at dump time. The getattr fallback covers extra="forbid" options
        # (which don't carry it in __pydantic_extra__), so entities stored as unannotated scene
        # children still recover their concrete class on load even though they forbid extras.
        result["_option_module_"] = getattr(obj, "_option_module_", obj.__module__)
        result["_option_class_"] = getattr(obj, "_option_class_", obj.__class__.__name__)

        # Now recursively process each field value to capture nested Options metadata
        for field_name in list(result.keys()):
            if field_name not in OPTION_METADATA_KEYS:
                # Try to get the actual object using getattr to handle all cases
                # (including fields not in __dict__ due to defaults, properties, etc)
                try:
                    actual_value = getattr(obj, field_name)
                    # If the actual value is an Options object, recursively serialize it
                    if isinstance(actual_value, Options):
                        result[field_name] = serialize_obj_with_metadata(actual_value)
                    else:
                        # Recurse over the live field value so container fields
                        # (for example list[Union[...Options]]) keep subclass data
                        # that may be erased by model_dump() schema narrowing.
                        result[field_name] = serialize_obj_with_metadata(actual_value)
                except AttributeError:
                    # Field doesn't exist, keep the dumped value and recurse
                    result[field_name] = serialize_obj_with_metadata(result[field_name])

        return result
    elif isinstance(obj, BaseModel):
        # For other BaseModel instances, serialize and recurse
        result = obj.model_dump()
        for key, value in list(result.items()):
            result[key] = serialize_obj_with_metadata(value)
        return result
    elif isinstance(obj, dict):
        return {key: serialize_obj_with_metadata(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [serialize_obj_with_metadata(item) for item in obj]
    elif isinstance(obj, Enum):
        return obj.value
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy().tolist()
    # in case of np/torch scalars, cast to plain numeric types
    elif isinstance(obj, bool):
        return bool(obj)
    elif isinstance(obj, int):
        return int(obj)
    elif isinstance(obj, float):
        return float(obj)
    return obj


def get_options_class_from_qualname(qualname: str):
    """
    Dynamically import and retrieve an Options class from its fully qualified name.

    Parameters
    ----------
    qualname : str
        The fully qualified name of the Options class (e.g., "eden.options.managers.actions.ActionTermOptions")

    Returns
    -------
    Type[Options]
        The Options class

    Raises
    ------
    ImportError
        If the module cannot be imported
    AttributeError
        If the class cannot be found in the module
    """
    import importlib

    # Split the qualname into module path and class name
    parts = qualname.rsplit(".", 1)
    if len(parts) != 2:
        raise ValueError(f"Invalid qualified name: {qualname}. Expected format: 'module.path.ClassName'")

    module_path, class_name = parts

    # Import the module
    try:
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(f"Could not import module '{module_path}': {e}")

    # Get the class from the module
    try:
        return getattr(module, class_name)
    except AttributeError:
        raise AttributeError(f"Class '{class_name}' not found in module '{module_path}'")


# --- to_options helpers --------------------------------------------------------
# ``_convert_value`` / ``_construct_options`` are mutually recursive — they drive the whole
# reconstruction.


def _is_dynamic_config(value) -> bool:
    """Check if value is a dict carrying ``_option_module_`` / ``_option_class_`` metadata."""
    if isinstance(value, dict):
        return all(key in value for key in OPTION_METADATA_KEYS)
    return False


def _is_options_class(value) -> bool:
    return isinstance(value, type) and issubclass(value, Options)


def _enum_class_from_annotation(annotation):
    """Return the ``Enum`` subclass named by ``annotation`` (unwrapping ``Optional``/``Union``), else None.

    Returns the *first* enum found in a union: a field annotated with two enum types whose
    serialized values overlap (e.g. ``EventMode | MetricMode``) would coerce to the first arm.
    No built-in field does this — avoid multi-enum unions in options, or the load is ambiguous
    anyway (the dump keeps only ``.value``).
    """
    if isinstance(annotation, type) and issubclass(annotation, Enum):
        return annotation
    if get_origin(annotation) in (Union, UnionType):
        for arg in get_args(annotation):
            enum_cls = _enum_class_from_annotation(arg)
            if enum_cls is not None:
                return enum_cls
    return None


@lru_cache(maxsize=None)
def _type_adapter(annotation) -> TypeAdapter:
    """Cached ``TypeAdapter`` for an annotation (config annotations are few and hashable)."""
    return TypeAdapter(annotation)


def _validate_against_annotation(value, annotation):
    """Recover a serialized enum leaf to its declared enum via the field's own pydantic validator.

    Genesis-native options load through ``model_construct`` (validation skipped, to tolerate
    list-for-tuple on strict genesis vector fields — see ``_safe_model_construct``), so an enum leaf
    would otherwise stay a bare ``str``. Running the annotation's ``TypeAdapter`` restores the type
    and — unlike a bare ``enum_cls(value)`` cast — validates it exactly as live construction would.
    Lenient: a value that does not validate warns and is kept raw (an unknown enum value shouldn't
    sink the config).
    """
    try:
        return _type_adapter(annotation).validate_python(value)
    except (ValidationError, ValueError, TypeError):
        en.logger.warning(f"Could not validate {value!r} against {annotation}; leaving value as-is.")
        return value


def _option_class_from_annotation(annotation, value=None):
    if annotation is None:
        return None
    if _is_options_class(annotation):
        return annotation

    origin = get_origin(annotation)
    if origin in (Union, UnionType):
        # A union that explicitly admits a dict form — bare ``dict`` (e.g. the
        # ``morph: Morph | dict | None`` escape hatch) or parameterized ``dict[K, V]`` —
        # declares a plain dict to be a legal raw value for the field. Dicts that are
        # serialized Options carry ``_option_class_`` metadata and are resolved before
        # annotation dispatch, so a metadata-less dict here must stay a dict — rebuilding it
        # via ``model_construct`` on the Options arm would fabricate every field as "set"
        # (e.g. a ``{"pos": ...}`` morph override coming back as a full ``Morph`` whose
        # ``quat=None`` then clobbers the entity default).
        if isinstance(value, dict) and any(arg is dict or get_origin(arg) is dict for arg in get_args(annotation)):
            return None
        option_classes = []
        for arg in get_args(annotation):
            option_cls = _option_class_from_annotation(arg)
            if option_cls is not None and option_cls not in option_classes:
                option_classes.append(option_cls)
        if len(option_classes) == 1:
            return option_classes[0]
        # Multiple candidates (e.g. `SurfaceOptions | Surface`): a dict with class
        # metadata is handled before we get here, so this is a legacy metadata-less
        # dict. Disambiguate by field-key overlap — the class whose declared fields
        # best cover the dict's keys.
        if len(option_classes) > 1 and isinstance(value, dict):
            keys = set(value) - set(OPTION_METADATA_KEYS)
            return max(option_classes, key=lambda cls: len(keys & set(getattr(cls, "model_fields", {}))))
    return None


def _item_annotation(annotation):
    origin = get_origin(annotation)
    if origin in (list, tuple):
        args = get_args(annotation)
        if args:
            return args[0]
    return None


def _value_annotation(annotation):
    origin = get_origin(annotation)
    if origin is dict:
        args = get_args(annotation)
        if len(args) == 2:
            return args[1]
    return None


def _key_annotation(annotation):
    origin = get_origin(annotation)
    if origin is dict:
        args = get_args(annotation)
        if len(args) == 2:
            return args[0]
    return None


def _coerce_dict_key(key, key_type):
    if key_type is int and isinstance(key, str):
        try:
            return int(key)
        except ValueError:
            return key
    return key


def _construct_options(options_cls: type[Options], value: dict):
    option_dict = {}
    for key, item in value.items():
        field_info = options_cls.model_fields.get(key)
        field_annotation = field_info.annotation if field_info is not None else None
        option_dict[key] = _convert_value(item, field_annotation)
    return _safe_model_construct(options_cls, option_dict)


def _convert_value(value, target_annotation=None):
    """Recursively convert values, handling dynamic configs and nested structures."""
    # Handle dynamic config instances
    if isinstance(value, dict) and _is_dynamic_config(value):
        option_module = value.get("_option_module_", None)
        option_class = value.get("_option_class_", None)
        ResolvedClass = get_options_class_from_qualname(f"{option_module}.{option_class}")
        return _construct_options(ResolvedClass, value)

    target_options_cls = _option_class_from_annotation(target_annotation, value)
    if isinstance(value, dict) and target_options_cls is not None:
        return _construct_options(target_options_cls, value)

    # Handle dicts (may contain dynamic configs)
    elif isinstance(value, dict):
        nested_annotation = _value_annotation(target_annotation)
        kt = _key_annotation(target_annotation)
        return {_coerce_dict_key(k, kt): _convert_value(v, nested_annotation) for k, v in value.items()}
    # Handle lists (may contain nested dynamic configs; NamedTuple rows load via target_annotation)
    elif isinstance(value, (list, tuple)):
        if target_annotation is not None and _is_namedtuple_class(target_annotation):
            seq = value
            fields = target_annotation._fields
            if len(seq) == len(fields):
                return target_annotation(*(_convert_value(v, None) for v in seq))
        nested_annotation = _item_annotation(target_annotation)
        return [_convert_value(item, nested_annotation) for item in value]
    # Coerce serialized enum values (dumped as their ``.value``) back to the enum instance. Load
    # reconstructs options via ``model_construct`` (validators skipped — see ``_safe_model_construct``),
    # so declared enum fields would otherwise come back as bare ``str``. Lenient (strict=False): an
    # out-of-range value warns and stays raw rather than failing the whole load.
    else:
        enum_cls = _enum_class_from_annotation(target_annotation)
        if enum_cls is not None and not isinstance(value, enum_cls) and value is not None:
            return _validate_against_annotation(value, target_annotation)
        return value


def to_options(config: dict, base_options: Options | type[Options]):
    """
    Reconstruct the original Options instance from serialized data.

    Eden options (config root classes included) are rebuilt with validation, so an unknown key on
    an ``extra="forbid"`` class raises, while ``extra="allow"`` classes (ManagerOptions,
    SceneOptions, etc.) keep unknown keys as their dynamic term/entity children. Genesis-native
    options skip validation, and an ``extra="forbid"`` one drops unknown keys with a warning — see
    :func:`_safe_model_construct`.

    Parameters
    ----------
    config: dict
        The serialized field data.
    base_options: Options | type[Options]
        The target type, as either a class or an instance to take the class of. Only the
        *type* is read, so pass the class when there is nothing to instantiate it from —
        a task config's ``__init__`` can be expensive and its result is discarded here.
    """
    options_cls = base_options if isinstance(base_options, type) else type(base_options)

    # Convert all config attributes
    res = {}
    for key, value in config.items():
        field_info = options_cls.model_fields.get(key)
        field_annotation = field_info.annotation if field_info is not None else None
        res[key] = _convert_value(value, field_annotation)

    return _safe_model_construct(options_cls, res)


def _safe_model_construct(cls: type, data: dict):
    """Build ``cls`` from an already type-recovered ``data`` dict, validating eden options on load.

    ``_convert_value`` has already reconstructed nested Options instances and enums; this turns
    that dict into the ``cls`` instance.

    - **eden ``ConfigurableOptions``** (``strict=False``) are built via *validating* construction
      (``cls.model_validate``): the JSON round-trip coerces (list→tuple, str→enum,
      nested-list→``FloatArray`` ndarray) and a bad value / unknown field / stale term param
      **fails loud at load**, naming the class and the field, not silently deep in ``build()``.
      This is the only thing that turns a saved matrix back into an ``np.ndarray``, so a
      ``FloatArray`` field belongs on a ``ConfigurableOptions`` subclass. No pre-filtering of
      unknown keys — an ``extra="forbid"`` class lets ``model_validate`` raise on a typo'd/stale
      top-level field (naming it), rather than dropping it and loading the default;
      ``extra="allow"`` classes keep unknown keys as their dynamic children. Type-recovery tags are
      stripped first (re-seeded by ``ConfigurableOptions.__init__``). This requires eden options to
      be plain declarative schemas
      — scene *presets* are factory functions returning a base ``SceneOptions``, not
      builder-``__init__`` subclasses, so nothing here needs non-persisted construction args.
    - **genesis-native Options** stay on ``model_construct`` (validation skipped): they are
      ``strict=True`` and their fixed-size vector fields would reject the JSON list form. Since
      ``model_construct`` can't validate, unknown/legacy keys on a ``forbid`` genesis option are
      dropped with a warning — a genesis (third-party) schema change must not hard-fail a load.
      Nothing validates these fields, so a ``FloatArray`` declared on one would load back as a
      plain ``list``.
    """
    is_eden = issubclass(cls, ConfigurableOptions)
    # Type-recovery tags are internal metadata, never real fields — drop them for both eden
    # (re-seeded by ``ConfigurableOptions.__init__``) and genesis-native options, so they never
    # reach ``model_validate``/``model_construct`` nor the unknown-key warning below.
    data = {k: v for k, v in data.items() if k not in OPTION_METADATA_KEYS}
    if is_eden:
        # model_validate enforces the class's own extra policy: forbid -> raises loud on an
        # unknown field; allow -> keeps it as a dynamic child. Fail-fast, no silent drop.
        return cls.model_validate(data)
    if cls.model_config.get("extra") == "forbid":
        known_fields = set(cls.model_fields.keys())
        dropped = {k for k in data if k not in known_fields}
        if dropped:
            en.logger.warning(f"Dropping unknown keys when loading {cls.__name__}: {dropped}")
        data = {k: v for k, v in data.items() if k in known_fields}
    return cls.model_construct(**data)


def load_json(path: str):
    import json

    with open(path, "r") as f:
        data = json.load(f)
    return data
