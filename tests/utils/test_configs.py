"""``eden.utils.configs``: saving, loading, locking and overriding Eden configs."""

from __future__ import annotations

import json
import os
from typing import Optional

import genesis as gs
import numpy as np
import pytest
import torch
import yaml
from pydantic import ValidationError

import eden.terms  # noqa: F401 -- stamps each kind's registry onto its options class
from eden.constants import EventMode, MetricDirection, NoiseOperation
from eden.engine.solvers.mochi.options import MochiOptions
from eden.envs.base import EnvBase
from eden.managers.modifiers.observations import ExponentialMovingAverage, ObservationDelay
from eden.managers.reward_manager import REWARD_TERM_REGISTRY
from eden.options import (
    BoxOptions,
    CylinderOptions,
    EnvOptions,
    GroupedEntityOptions,
    ObservationGroupOptions,
    ObservationTermOptions,
    PlaneOptions,
    SceneOptions,
    SensorsOptions,
    SphereOptions,
)
from eden.options.camera import CameraOptions, CamerasOptions
from eden.options.managers.actions import ActionTermOptions
from eden.options.managers.commands import CommandManagerOptions, CommandTermOptions
from eden.options.managers.events import EventManagerOptions, EventTermOptions
from eden.options.managers.metrics import MetricManagerOptions, MetricTermOptions
from eden.options.managers.observations import NoiseOptions, ObservationManagerOptions
from eden.options.managers.rewards import RewardGroupOptions, RewardManagerOptions, RewardTermOptions
from eden.options.options import ConfigurableOptions
from eden.options.sensors import SensorOptions
from eden.utils.configs import (
    EdenConfig,
    EdenMimicConfig,
    EdenRLConfig,
    EdenSplatConfig,
    EdenSplatRLConfig,
    _convert_value,
    _deep_merge,
    _safe_model_construct,
    get_options_class_from_qualname,
    is_full_config,
    overrides_from_str,
    read_config_file,
    saved_config_qualname,
    serialize_obj_with_metadata,
    to_options,
)
from eden.utils.misc import get_eden_provenance, get_package_provenance


# Named so pytest does not try to collect it (a ``Test*`` pydantic model warns).
class RoundtripConfig(EdenRLConfig):
    env_options: EnvOptions = EnvOptions(num_envs=1)
    test_param: str = "initial"
    test_vec: list[float] = [1.0, 2.0, 3.0]


def _saved(tmp_path, config=None, name: str = "cfg.json") -> str:
    path = str(tmp_path / name)
    (config or RoundtripConfig()).save_as_file(path, lock_config=False)
    return path


_CLASS_TAGS = ("_option_module_", "_option_class_")


def _without_class_tags(mapping: dict) -> None:
    for key in _CLASS_TAGS:
        mapping.pop(key)


def _rewrite(path: str, edit) -> None:
    with open(path) as f:
        raw = json.load(f)
    edit(raw)
    with open(path, "w") as f:
        json.dump(raw, f)


# ---------------------------------------------------------------------------
# Save, lock, load
# ---------------------------------------------------------------------------


def test_save_locks_the_config_and_the_load_reproduces_the_file(tmp_path):
    config = RoundtripConfig()
    config.env_options.num_envs = 10
    config.test_param = "modified"
    config.action_options.custom_controller = ActionTermOptions(entity_name="robot", dofs_name=[])
    config.action_options.custom_controller.dynamic_param = 123  # term options take extras
    path = tmp_path / "config.json"

    config.save_as_json(str(path), lock_config=True)

    for set_field in (
        lambda: setattr(config.env_options, "num_envs", 20),
        lambda: setattr(config, "test_param", "changed"),
        lambda: setattr(config.action_options.custom_controller, "dynamic_param", 456),
    ):
        with pytest.raises(RuntimeError, match="locked"):
            set_field()
    raw = json.loads(path.read_text())
    assert set(raw) == {"_meta", "config"}
    assert (raw["_meta"]["eden"], raw["_meta"]["genesis"]) == (
        get_eden_provenance(),
        get_package_provenance("genesis-world"),
    )

    loaded = RoundtripConfig.load_from_file(str(path))
    assert (loaded.env_options.num_envs, loaded.test_param) == (10, "modified")
    assert loaded.action_options.custom_controller.dynamic_param == 123
    resaved = tmp_path / "resaved.json"
    loaded.save_as_json(str(resaved), lock_config=False)
    assert json.loads(resaved.read_text()) == raw


def test_yaml_round_trips_numpy_scalars_enums_and_tensors_without_python_tags(tmp_path):
    """``!!python/...`` tags would make the file unloadable by ``yaml.safe_load``."""
    config = RoundtripConfig()
    config.env_options.env_spacing = (np.float64(2.5), np.float64(3.5))
    config.observation_options.policy = ObservationGroupOptions(
        terms_order=["dofs_pos"],
        dofs_pos=ObservationTermOptions(noise=NoiseOptions(operation=NoiseOperation.SCALE)),
    )
    config.test_vec = torch.tensor([1.0, 2.0, 3.0])
    path = tmp_path / "config.yaml"

    config.save_as_yaml(str(path), lock_config=False)

    assert "!!python" not in path.read_text()
    loaded = RoundtripConfig.load_from_file(str(path))
    assert tuple(loaded.env_options.env_spacing) == (2.5, 3.5)
    assert loaded.observation_options.policy.dofs_pos.noise.operation == NoiseOperation.SCALE
    assert loaded.test_vec == [1.0, 2.0, 3.0]


@pytest.mark.parametrize(
    ("save", "bad_name"),
    [
        ("save_as_json", "config.txt"),
        ("save_as_json", "config.yaml"),
        ("save_as_json", "config"),
        ("save_as_yaml", "config.txt"),
        ("save_as_yaml", "config.json"),
        ("save_as_yaml", "config"),
    ],
)
def test_save_refuses_a_path_of_the_other_format_before_any_side_effect(tmp_path, save, bad_name):
    """This was an ``assert``: ``python -O`` dropped it and YAML landed under a JSON-named call."""
    config = RoundtripConfig()
    path = tmp_path / bad_name

    with pytest.raises(ValueError) as excinfo:
        getattr(config, save)(str(path))

    assert save in str(excinfo.value) and bad_name in str(excinfo.value)
    assert ".json" in str(excinfo.value) or ".yaml" in str(excinfo.value)
    assert not path.exists()
    config.test_param = "still unlocked"


@pytest.mark.parametrize(("name", "save"), [("config.JSON", "save_as_json"), ("config.YML", "save_as_yaml")])
def test_extensions_match_case_insensitively(tmp_path, name, save):
    path = str(tmp_path / name)
    getattr(RoundtripConfig(test_param="upper"), save)(path, lock_config=False)

    assert RoundtripConfig.load_from_file(path).test_param == "upper"


def test_an_unsupported_extension_or_a_non_mapping_is_not_a_config(tmp_path):
    txt = tmp_path / "config.txt"
    txt.write_text("{}")
    listy = tmp_path / "list.yaml"
    listy.write_text("- 1\n- 2\n")

    for load in (RoundtripConfig.load_from_file, read_config_file):
        with pytest.raises(ValueError, match="Unsupported config file extension"):
            load(str(txt))
    with pytest.raises(ValueError, match="does not hold a config mapping"):
        read_config_file(str(listy))


def test_load_fails_fast_on_a_corrupt_field_or_an_unknown_key(tmp_path):
    """Validated load (``model_validate``): a stale or typo'd key must not fall back to a default."""
    corrupt = _saved(tmp_path, name="corrupt.json")
    _rewrite(corrupt, lambda raw: raw["config"]["env_options"].update(num_envs="not-an-int"))
    stale = _saved(tmp_path, name="stale.json")
    _rewrite(stale, lambda raw: raw["config"].update(BOGUS_KEY="x"))

    with pytest.raises(ValidationError, match="num_envs"):
        RoundtripConfig.load_from_file(corrupt)
    with pytest.raises(ValidationError, match="BOGUS_KEY"):
        RoundtripConfig.load_from_file(stale)
    with pytest.raises(ValidationError, match="extra_forbidden"):
        EdenConfig(unknown_field="value")


# ---------------------------------------------------------------------------
# Which class a load reconstructs
# ---------------------------------------------------------------------------


def test_load_reconstructs_the_class_the_file_was_saved_from(tmp_path):
    """``eden train --config <saved>`` loads through ``EdenConfig``; an RL task must still come back whole."""
    path = _saved(tmp_path, RoundtripConfig(test_param="concrete"))

    loaded = EdenConfig.load_from_file(path)

    assert type(loaded) is RoundtripConfig and loaded.test_param == "concrete"
    assert saved_config_qualname(read_config_file(path)) == f"{RoundtripConfig.__module__}.RoundtripConfig"


def test_load_refuses_a_class_the_file_was_not_saved_from(tmp_path):
    with pytest.raises(TypeError, match="not a subclass of the requested EdenMimicConfig"):
        EdenMimicConfig.load_from_file(_saved(tmp_path))


def test_load_raises_on_an_unimportable_saved_class(tmp_path):
    """Reconstructing some other class instead would drop every field only the saved one declares."""
    path = _saved(tmp_path)
    _rewrite(path, lambda raw: raw["_meta"].update(_option_module_="nowhere.at.all"))

    with pytest.raises(ImportError, match="nowhere.at.all.RoundtripConfig"):
        EdenConfig.load_from_file(path)


def test_a_class_defined_in_a_function_loads_through_itself(tmp_path):
    """Its qualname does not resolve, but the caller passed the class, so nothing needs looking up."""

    class LocalConfig(EdenRLConfig):
        local_field: int = 7

    path = _saved(tmp_path, LocalConfig(local_field=9))
    with pytest.raises(AttributeError):
        get_options_class_from_qualname(saved_config_qualname(read_config_file(path)))

    assert LocalConfig.load_from_file(path).local_field == 9


def test_files_without_class_metadata_load_as_the_requested_class(tmp_path):
    """Pre-metadata saves and hand-written files; the pre-envelope flat layout carries the tags inline."""
    untagged = _saved(tmp_path, RoundtripConfig(test_param="legacy"), "untagged.json")
    _rewrite(untagged, lambda raw: _without_class_tags(raw["_meta"]))
    flat = tmp_path / "flat.json"
    envelope = json.loads(open(_saved(tmp_path, RoundtripConfig(test_param="flat"), "env.json")).read())
    flat.write_text(json.dumps({**envelope["config"], **{k: envelope["_meta"][k] for k in _CLASS_TAGS}}))

    assert RoundtripConfig.load_from_file(untagged).test_param == "legacy"
    loaded = EdenConfig.load_from_file(str(flat))
    assert type(loaded) is RoundtripConfig and loaded.test_param == "flat"


def test_saved_and_hand_written_files_are_told_apart(tmp_path):
    """A save carries class metadata (envelope or flat); a partial override does not."""
    partial = tmp_path / "partial.json"
    partial.write_text(json.dumps({"env_options": {"num_envs": 64}}))
    saved = read_config_file(_saved(tmp_path))

    assert is_full_config(saved) is True
    assert is_full_config({"_option_class_": "RoundtripConfig", "env_options": {}}) is True
    assert is_full_config(read_config_file(str(partial))) is False
    assert saved_config_qualname({"_option_module_": "m", "_option_class_": "C"}) == "m.C"
    assert saved_config_qualname({"env_options": {"num_envs": 4}}) is None
    assert "_meta" in saved  # reading the qualname does not consume the envelope


def test_load_from_data_is_load_from_file_without_the_read(tmp_path):
    path = _saved(tmp_path, RoundtripConfig(test_param="data"))

    loaded = EdenConfig.load_from_data(read_config_file(path), path)

    assert type(loaded) is RoundtripConfig and loaded.test_param == "data"


# ---------------------------------------------------------------------------
# Overrides
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fmt", ["json", "yaml", "dict"])
def test_an_override_returns_a_new_mutable_config_and_leaves_the_locked_source_alone(tmp_path, fmt):
    """The override path reconstructs (serialize, merge, validate), so a locked source yields an unlocked copy."""
    config = RoundtripConfig(test_param="unchanged")
    config.env_options.num_envs = 8
    config.lock()
    override = {"env_options": {"num_envs": 64}}
    if fmt == "dict":
        new = config.with_overrides_from_dict(override)
    else:
        path = tmp_path / f"override.{fmt}"
        path.write_text(json.dumps(override) if fmt == "json" else yaml.safe_dump(override))
        new = config.with_overrides_from_file(str(path))

    assert type(new) is RoundtripConfig and type(new.env_options) is EnvOptions
    assert (new.env_options.num_envs, new.test_param) == (64, "unchanged")
    assert config.env_options.num_envs == 8
    new.test_param = "editable"


def test_an_override_keeps_an_unset_none_field_of_genesis_options_unset():
    """Genesis reads set-ness: a solver whose options list ``dt`` as set divides the sim dt by it (a None)."""
    config = EdenConfig(env_options=EnvOptions(mochi_options=MochiOptions(n_newton_iterations=3)))

    new = config.with_overrides_from_dict({"env_options": {"num_envs": 2}})

    mochi = new.env_options.mochi_options
    assert mochi.n_newton_iterations == 3 and mochi.dt is None
    assert "dt" not in mochi.model_fields_set
    assert "dt" not in serialize_obj_with_metadata(config.env_options.mochi_options)


def test_an_override_may_change_an_options_class():
    config = EdenConfig(sensors_options=SensorsOptions(contact=SensorOptions(sensor=gs.sensors.Contact())))

    new = config.with_overrides_from_dict(
        {
            "sensors_options": {
                "contact": {"sensor": {"_option_module_": "genesis.options.sensors", "_option_class_": "ContactForce"}}
            }
        }
    )

    assert type(new.sensors_options.contact.sensor).__name__ == "ContactForce"
    assert type(config.sensors_options.contact.sensor).__name__ == "Contact"


def test_a_type_changing_override_replaces_the_entry_instead_of_merging_into_it():
    """Merged, a ``BoxOptions`` override of a group would keep ``grouped_entities``, which ``BoxOptions`` forbids."""
    base = {
        "cube": {
            "_option_module_": "eden.options.entities",
            "_option_class_": "GroupedEntityOptions",
            "grouped_entities": [{"_option_class_": "MeshOptions"}],
            "default_root_pos": [0.3, 0.0, 0.6],
        }
    }

    changed = _deep_merge(base, {"cube": {"_option_class_": "BoxOptions", "size": [0.1, 0.1, 0.1]}})
    merged = _deep_merge(base, {"cube": {"default_root_pos": [0.4, 0.0, 0.6]}})

    assert changed["cube"] == {"_option_class_": "BoxOptions", "size": [0.1, 0.1, 0.1]}
    assert merged["cube"] == {**base["cube"], "default_root_pos": [0.4, 0.0, 0.6]}


@pytest.mark.parametrize(
    ("text", "error"),
    [
        ("env_options: {num_envs: 64}", None),
        ('{"env_options": {"num_envs": 64}}', None),
        ("overrides.json", r"not a mapping.*use --config"),  # a path is the reflex mistake
        ("env_options: {num_envs: 64", "inline YAML mapping"),
    ],
)
def test_overrides_from_str_takes_yaml_flow_or_json_and_names_the_flag_for_a_file(text, error):
    if error is None:
        assert overrides_from_str(text) == {"env_options": {"num_envs": 64}}
    else:
        with pytest.raises(ValueError, match=error):
            overrides_from_str(text)


# ---------------------------------------------------------------------------
# Field types that must survive the file
# ---------------------------------------------------------------------------


def test_genesis_options_round_trip_to_their_concrete_classes(tmp_path):
    """Genesis options stay ``extra="forbid"`` and unpatched: the class is stamped at dump time."""
    config = EdenConfig(
        scene_options=SceneOptions(
            plane=PlaneOptions(surface=gs.surfaces.Plastic()),
            box=BoxOptions(size=(0.2, 0.3, 0.4)),
            group=GroupedEntityOptions(
                grouped_entities=[SphereOptions(radius=0.2), CylinderOptions(radius=0.2, height=0.3)]
            ),
        ),
    )
    path = _saved(tmp_path, config)

    surface = json.loads(open(path).read())["config"]["scene_options"]["plane"]["surface"]
    assert (surface["_option_module_"], surface["_option_class_"]) == ("genesis.options.surfaces", "Plastic")
    loaded = EdenConfig.load_from_file(path).scene_options
    assert type(loaded.plane.surface) is gs.surfaces.Plastic
    assert tuple(loaded.box.size) == (0.2, 0.3, 0.4)
    sphere, cylinder = loaded.group.grouped_entities
    assert (type(sphere), sphere.radius) == (SphereOptions, 0.2)
    assert (type(cylinder), cylinder.height) == (CylinderOptions, 0.3)
    with pytest.raises(gs.GenesisException, match="colour"):  # Genesis' own validation is intact
        gs.surfaces.Plastic(colour=(1.0, 0.0, 0.0))


def test_a_genesis_surface_saved_without_class_metadata_rebuilds_from_the_field_type(tmp_path):
    path = _saved(
        tmp_path,
        EdenConfig(scene_options=SceneOptions(plane=PlaneOptions(surface=gs.surfaces.Default(double_sided=True)))),
    )
    _rewrite(path, lambda raw: _without_class_tags(raw["config"]["scene_options"]["plane"]["surface"]))

    surface = EdenConfig.load_from_file(path).scene_options.plane.surface

    assert isinstance(surface, gs.surfaces.Surface) and surface.double_sided is True


@pytest.mark.requires_genesis
def test_genesis_sensor_options_survive_yaml_and_build_the_same_sensor(tmp_path):
    """``TemperatureGrid``'s NamedTuple dict values and ``track_link_names`` resolution."""
    from genesis.options.sensors.options import TemperatureGrid, TemperatureProperties

    config = EdenConfig(
        env_options=EnvOptions(num_envs=1),
        scene_options=SceneOptions(
            box=BoxOptions(size=(0.2, 0.2, 0.2)), sphere1=SphereOptions(radius=0.1), sphere2=SphereOptions(radius=0.2)
        ),
        sensors_options=SensorsOptions(
            temp_grid=SensorOptions(
                sensor=TemperatureGrid(
                    entity_idx=0,
                    link_idx_local=0,
                    properties_dict={
                        -1: TemperatureProperties(22.0, 150.0, 2000.0, 500.0, 0.85),
                        0: TemperatureProperties(45.0, 200.0, 3000.0, 1000.0, 0.4),
                    },
                    grid_size=(1, 1, 1),
                    ambient_temperature=22.0,
                    convection_coefficient=0.0,
                ),
                attach_entity_name="box",
                attach_link_name="box_baselink",
            ),
            surface_distance=SensorOptions(
                sensor=gs.sensors.SurfaceDistanceProbe(
                    probe_local_pos=[(0.0, 0.0, 0.0)], probe_radius=1.0, track_link_idx=(0, 0)
                ),
                attach_entity_name="box",
                attach_link_name="box_baselink",
                track_link_names=["sphere1", "sphere2"],
            ),
        ),
    )
    path = tmp_path / "sensors.yaml"
    config.save_as_yaml(str(path), lock_config=False)

    raw = yaml.safe_load(path.read_text())["config"]["sensors_options"]["temp_grid"]["sensor"]["properties_dict"]
    assert raw[-1] == [22.0, 150.0, 2000.0, 500.0, 0.85]
    loaded = EdenConfig.load_from_file(str(path))
    props = loaded.sensors_options.temp_grid.sensor.properties_dict
    assert props[-1] == TemperatureProperties(22.0, 150.0, 2000.0, 500.0, 0.85)
    assert type(props[0]) is TemperatureProperties and props[0].conductivity == 200.0
    assert loaded.sensors_options.surface_distance.track_link_names == ["sphere1", "sphere2"]

    built = []
    for cfg in (config, loaded):
        env = EnvBase.from_config(cfg)
        env.build()
        built.append(env.sensors["surface_distance"]._options.sensor.track_link_idx)
    assert built[0] == built[1]


def test_a_morph_kwargs_dict_round_trips_verbatim(tmp_path):
    """Rebuilt as a base ``Morph``, it would fabricate ``quat=None`` and defaults that clobber the entity's own."""
    path = _saved(
        tmp_path,
        EdenConfig(scene_options=SceneOptions(cube=BoxOptions(size=(0.1, 0.1, 0.1), morph={"pos": (1.0, 2.0, 3.0)}))),
    )

    cube = EdenConfig.load_from_file(path).scene_options.cube

    assert list(cube.morph) == ["pos"]
    kwargs = cube._collect_morph_kwargs(size=(0.1, 0.1, 0.1))
    assert (tuple(kwargs["pos"]), tuple(kwargs["quat"])) == ((1.0, 2.0, 3.0), (1.0, 0.0, 0.0, 0.0))
    assert "offset_pos" not in kwargs and "requires_jac_and_IK" not in kwargs


def test_a_prebuilt_morph_round_trips_to_its_concrete_class(tmp_path):
    prebuilt = gs.morphs.Box(size=(0.2, 0.2, 0.2), pos=(4.0, 5.0, 6.0))
    path = _saved(
        tmp_path, EdenConfig(scene_options=SceneOptions(cube=BoxOptions(size=(0.1, 0.1, 0.1), morph=prebuilt)))
    )

    cube = EdenConfig.load_from_file(path).scene_options.cube

    assert type(cube.morph) is gs.morphs.Box and tuple(cube.morph.pos) == (4.0, 5.0, 6.0)
    assert tuple(cube.get_morph().pos) == (4.0, 5.0, 6.0)


def test_a_union_with_a_dict_arm_keeps_a_bare_dict_a_dict():
    assert _convert_value({"pos": (1.0, 2.0, 3.0)}, BoxOptions | dict | None) == {"pos": [1.0, 2.0, 3.0]}
    assert _convert_value({"a": 1.0}, BoxOptions | dict[str, float] | None) == {"a": 1.0}
    assert type(_convert_value({"size": (0.2, 0.2, 0.2)}, BoxOptions)) is BoxOptions


def test_enum_fields_round_trip_as_enums(tmp_path):
    """Loaded as bare ``str`` they compare equal to the enum and break every ``is`` check."""
    config = EdenRLConfig()
    config.event_options = EventManagerOptions(reset_foo=EventTermOptions(mode=EventMode.RESET, name="x"))
    config.metric_options = MetricManagerOptions(m1=MetricTermOptions(direction=MetricDirection.HIB, name="m"))

    loaded = EdenRLConfig.load_from_file(_saved(tmp_path, config))

    assert loaded.event_options.reset_foo.mode is EventMode.RESET
    assert loaded.metric_options.m1.direction is MetricDirection.HIB


def test_enum_coercion_handles_optional_and_falls_back_to_the_raw_value():
    assert _convert_value("reset", Optional[EventMode]) is EventMode.RESET
    assert _convert_value("not_a_mode", EventMode) == "not_a_mode"
    assert _convert_value(None, Optional[EventMode]) is None


def test_camera_options_round_trip_with_their_float_matrices(tmp_path):
    """A forbid leaf inside an allow container, and ``FloatArray`` fields rebuilt as float arrays."""
    intrinsic = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]])  # ints, as json writes 600.0 back
    extrinsic = np.eye(4)
    extrinsic[:3, 3] = [1.0, 2.0, 3.0]
    config = EdenConfig(
        cameras_options=CamerasOptions(
            front=CameraOptions(cam_intrinsic=intrinsic, cam_extrinsic=extrinsic, cam_resolution=(320, 240)),
            back=CameraOptions(),
        )
    )

    cameras = EdenConfig.load_from_file(_saved(tmp_path, config)).cameras_options

    assert type(cameras.front) is CameraOptions and tuple(cameras.front.cam_resolution) == (320, 240)
    assert cameras.front.cam_intrinsic.dtype.kind == "f" and np.allclose(cameras.front.cam_intrinsic, intrinsic)
    assert cameras.front.cam_extrinsic.shape == (4, 4) and np.allclose(cameras.front.cam_extrinsic, extrinsic)
    assert cameras.back.cam_intrinsic is None and cameras.back.cam_extrinsic is None


def test_a_matrix_of_the_wrong_shape_fails_at_load_naming_the_class_and_field():
    with pytest.raises(ValidationError, match=r"for CameraOptions\ncam_intrinsic\n.*expected shape \(3, 3\)"):
        to_options({"cam_intrinsic": [[1.0, 0.0], [0.0, 1.0]]}, CameraOptions)


def test_splat_pcds_survive_a_round_trip(tmp_path):
    """``SplatOptions`` collects named ``PCDOptions`` children into ``pcds``; the load must re-absorb them."""
    from eden.options.splat import PCDOptions, SplatOptions

    ply = tmp_path / "bg.ply"
    ply.write_bytes(b"")
    splat = SplatOptions(background=PCDOptions(path=str(ply), scale=2.0))

    reloaded = _convert_value(serialize_obj_with_metadata(splat), SplatOptions)

    assert set(reloaded.pcds) == {"background"} and reloaded.pcds["background"].scale == 2.0


# ---------------------------------------------------------------------------
# Every registered task
# ---------------------------------------------------------------------------


def _registered_tasks():
    from eden.tasks import TASK_REGISTRY

    return sorted(TASK_REGISTRY.list_tasks())


@pytest.mark.parametrize("task_name", _registered_tasks())
def test_every_task_config_round_trips_to_a_fixed_point(task_name, tmp_path):
    """Loads as its own class, also through ``EdenConfig``, and re-saving the loaded config is stable.

    The second and third saves are compared: the first validating load may normalize (a quaternion
    validator re-rounds the last ULP), after which the round trip must not drift.
    """
    from eden.tasks import TASK_REGISTRY

    cfg_cls = TASK_REGISTRY.get(task_name)
    first = _saved(tmp_path, cfg_cls(), "first.json")

    loaded = cfg_cls.load_from_file(first)
    assert type(loaded) is cfg_cls
    assert type(EdenConfig.load_from_file(first)) is cfg_cls
    second = _saved(tmp_path, loaded, "second.json")
    third = _saved(tmp_path, cfg_cls.load_from_file(second), "third.json")
    assert json.loads(open(second).read())["config"] == json.loads(open(third).read())["config"]


@pytest.mark.parametrize("task_name", _registered_tasks())
def test_every_task_keeps_an_override_through_reconstruction(task_name):
    """A task that builds fields in ``__init__`` clobbers ``--config`` overrides on reconstruction.

    The value is checked, not the key: a clobbering ``__init__`` reassigns ``scene_options.cube``, leaving the
    key present while the override's type and size are gone. Declare the field as a default, per the reacher template.
    The class is built without its task mods, which reconstruction does not re-run and which may need data to apply at
    all (flashdexretarget's ``--motion``).
    """
    from eden.options.materials import MochiMaterialOptions
    from eden.tasks import TASK_REGISTRY

    config = TASK_REGISTRY.get(task_name)()
    cube = {"_option_module_": "eden.options.entities", "_option_class_": "BoxOptions", "size": [0.1, 0.1, 0.1]}
    if any(isinstance(e.material, MochiMaterialOptions) for e in config.scene_options.named_entities().values()):
        # A Mochi scene refuses a rigid entity, so its cube is kinematic; every other task keeps the default material.
        cube["material"] = {"_option_module_": "eden.options.materials", "_option_class_": "KinematicMaterialOptions"}
    merged = config.with_overrides_from_dict({"scene_options": {"cube": cube}})

    assert type(merged.scene_options.cube) is BoxOptions, f"{task_name} reverted scene_options.cube on reconstruction"
    assert tuple(merged.scene_options.cube.size) == (0.1, 0.1, 0.1)


# ---------------------------------------------------------------------------
# Manager options through a saved config and through overrides
# ---------------------------------------------------------------------------


@REWARD_TERM_REGISTRY.register()
def configs_test_const(env, value: float = 1.0):
    return value


def _const(weight: float, value: float) -> RewardTermOptions:
    return RewardTermOptions(name="configs_test_const", weight=weight, params={"value": value})


def test_observation_modifiers_and_nan_policy_survive_a_saved_config(tmp_path):
    from eden import observations

    config = EdenConfig(
        observation_options=ObservationManagerOptions(
            policy=ObservationGroupOptions(
                x=observations.DofsPos.configure(
                    entity_name="robot",
                    modifiers=[
                        ExponentialMovingAverage.configure(alpha=0.2, backfill=False),
                        ObservationDelay.configure(control_steps=3),
                    ],
                ),
                nan_policy="sanitize",
            )
        )
    )
    path = str(tmp_path / "config.json")
    config.save_as_json(path, lock_config=False)

    group = EdenConfig.load_from_file(path).observation_options.policy

    assert group.nan_policy == "sanitize"
    ema, delay = group.x.modifiers
    assert (ema.name, ema.alpha, ema.backfill) == ("exponential_moving_average", 0.2, False)
    assert (delay.name, delay.control_steps) == ("observation_delay", 3)


def test_mixed_flat_and_grouped_rewards_survive_a_saved_config_with_their_types(tmp_path):
    cfg = EdenRLConfig()
    cfg.reward_options = RewardManagerOptions(alive=_const(1.0, 2.0), rg=RewardGroupOptions(track=_const(2.0, 3.0)))
    path = str(tmp_path / "cfg.json")
    cfg.save_as_file(path, lock_config=False)

    loaded = EdenRLConfig.load_from_file(path).reward_options

    assert isinstance(loaded.alive, RewardTermOptions)
    assert isinstance(loaded.rg, RewardGroupOptions)
    assert (loaded.rg.track.weight, loaded.rg.track.params) == (2.0, {"value": 3.0})


def test_a_dict_override_reaches_a_term_inside_a_group_and_names_the_group_for_a_flat_spelling():
    """``--config`` deep-merges into the dump; the pre-groups flat spelling must point at the owning group."""
    cfg = EdenRLConfig()
    cfg.reward_options = RewardManagerOptions(rg=RewardGroupOptions(track=_const(2.0, 3.0)))

    overridden = cfg.with_overrides_from_dict({"reward_options": {"rg": {"track": {"weight": 5.0}}}})

    assert isinstance(overridden.reward_options.rg, RewardGroupOptions)
    assert overridden.reward_options.rg.track.weight == 5.0
    with pytest.raises(TypeError, match="inside group 'rg'"):
        cfg.with_overrides_from_dict({"reward_options": {"track": {"weight": 5.0}}})


def test_class_term_extras_survive_a_saved_config(tmp_path):
    """The load re-runs construction-time validation, so the extras must come back intact."""
    cfg = EdenRLConfig()
    cfg.command_options = CommandManagerOptions(
        base_velocity=CommandTermOptions(
            name="uniform_velocity_command",
            entity_name="robot",
            lin_vel_x_range=(-1.0, 1.0),
            resampling_time_range=(2.0, 4.0),
        )
    )
    path = os.path.join(tmp_path, "cfg.json")
    cfg.save_as_file(path, lock_config=False)

    term = EdenRLConfig.load_from_file(path).command_options.base_velocity

    assert term.name == "uniform_velocity_command"
    assert tuple(term.lin_vel_x_range) == (-1.0, 1.0)


def test_a_stale_parameter_in_a_saved_config_fails_at_load_as_it_does_live():
    with pytest.raises(ValueError, match="stale_param"):
        _safe_model_construct(RewardTermOptions, {"name": "upright", "stale_param": 3.0})
    with pytest.raises(ValueError, match="stale_param"):
        RewardTermOptions(name="upright", stale_param=3.0)


# ---------------------------------------------------------------------------
# The annotation-driven coercions of ``to_options``
# ---------------------------------------------------------------------------


class _Probe(ConfigurableOptions):
    counts: dict[int, float] = {}
    names: dict[str, int] = {}
    env: EnvOptions | None = None
    anything: ConfigurableOptions | None = None
    many: list[EnvOptions] = []
    items: tuple[int, ...] = ()


def _through_json(options: ConfigurableOptions) -> dict:
    """What a saved config looks like when read back: JSON turns every dict key into a string."""
    return json.loads(json.dumps(serialize_obj_with_metadata(options)))


def test_a_json_round_trip_restores_the_declared_types():
    original = _Probe(
        counts={3: 1.5, 7: 2.0},
        names={"5": 1},
        env=EnvOptions(num_envs=3),
        many=[EnvOptions(num_envs=2)],
        items=(1, 2),
    )

    loaded = to_options(_through_json(original), _Probe)

    assert loaded.counts == {3: 1.5, 7: 2.0}  # int keys, back from JSON's strings
    assert loaded.names == {"5": 1}  # a str-keyed dict is left alone
    assert isinstance(loaded.env, EnvOptions) and loaded.env.num_envs == 3  # through the Optional
    assert [type(e) for e in loaded.many] == [EnvOptions] and loaded.many[0].num_envs == 2
    assert loaded.items == (1, 2)


def test_a_nested_config_is_rebuilt_as_the_class_its_metadata_names():
    """A field typed as a base class holds whichever options class was saved into it."""
    loaded = to_options(_through_json(_Probe(anything=EnvOptions(num_envs=4))), _Probe)
    assert type(loaded.anything) is EnvOptions and loaded.anything.num_envs == 4


def test_a_key_that_cannot_be_coerced_reaches_validation_and_is_named():
    saved = _through_json(_Probe(counts={3: 1.5}))
    saved["counts"]["not-an-int"] = 4.0

    with pytest.raises(ValidationError, match="counts.not-an-int"):
        to_options(saved, _Probe)


def test_add_recording_camera_adds_the_configs_own_recording_camera():
    """``eden inference --rec`` adds the config's ``recording_camera()``, which a task overrides (the whiteboard
    tasks paint their ink) and every other config inherits: the default framing that follows the robot.
    """
    config = EdenConfig(cameras_options=None)
    config.add_recording_camera()
    assert config.cameras_options.rec == EdenConfig().recording_camera()
    assert config.cameras_options.rec.follow_entity_name == "robot"


@pytest.mark.parametrize("config_class", [EdenSplatConfig, EdenSplatRLConfig])
def test_a_splat_configs_recording_camera_declares_its_splat(config_class):
    """A splat env's camera shows the splat only when it declares a SplatComposite, so a splat config's recording
    camera does, first, behind empty space; the framing is the default one.
    """
    camera = config_class().recording_camera()
    (composite,) = camera.overlays
    assert (composite.name, composite.entities) == ("splat_composite", [])
    assert camera.follow_entity_name == EdenConfig().recording_camera().follow_entity_name


def test_add_recording_camera_keeps_a_rec_camera_the_config_declares():
    """A ``cameras_options: {rec: ...}`` override is the user's framing; replacing it would silently discard it."""
    config = EdenConfig(cameras_options=CamerasOptions(rec=CameraOptions(cam_pos=(9.0, 0.0, 1.0))))
    config.add_recording_camera()
    assert tuple(config.cameras_options.rec.cam_pos) == (9.0, 0.0, 1.0)


def test_a_declared_rec_camera_that_drops_the_tasks_overlays_is_warned_about(eden_caplog):
    """A declared ``rec`` camera replaces the task's recording camera whole: its ink would silently be missing."""
    from eden.tasks import TASK_REGISTRY

    config = TASK_REGISTRY.build("whiteboard_write_text")
    config.cameras_options.rec = CameraOptions(cam_pos=(9.0, 0.0, 1.0))
    config.add_recording_camera()
    assert "leaves out that camera's overlays ['plane_paint_overlay']" in eden_caplog.text

    eden_caplog.clear()
    config.cameras_options.rec = CameraOptions(cam_pos=(9.0, 0.0, 1.0), overlays=config.recording_camera().overlays)
    config.add_recording_camera()
    assert "leaves out" not in eden_caplog.text


def test_a_declared_rec_camera_loaded_from_a_saved_config_keeps_the_tasks_overlays_without_a_warning(eden_caplog):
    """``eden inference --rec`` saves its config with the ``rec`` camera it added; read back, the same overlay must
    still count as kept (adding it again would paint it twice).
    """
    from eden.tasks import TASK_REGISTRY

    config = TASK_REGISTRY.build("whiteboard_write_text")
    saved = config.recording_camera()
    config.cameras_options.rec = CameraOptions.model_validate_json(saved.model_dump_json())
    assert config.cameras_options.rec.overlays == saved.overlays
    config.add_recording_camera()
    assert "leaves out" not in eden_caplog.text

    config.lock()  # as ``save_as_json`` leaves it
    config.add_recording_camera()
    assert "leaves out" not in eden_caplog.text
