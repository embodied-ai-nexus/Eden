"""``ConfigurableOptions``: extra-forbid messages, lock / unlock / clone / copy, and polymorphic dumps."""

from __future__ import annotations

import copy
import json
import threading

import pytest
from pydantic import ValidationError

import eden.terms  # noqa: F401 -- stamps each kind's registry onto its options class
from eden.options import (
    BoxOptions,
    CylinderOptions,
    EnvOptions,
    GroupedEntityOptions,
    ObservationGroupOptions,
    ObservationTermOptions,
    SceneOptions,
    SphereOptions,
)
from eden.options.managers.rewards import RewardManagerOptions, RewardTermOptions
from eden.options.options import ConfigurableOptions
from eden.options.sensors import SensorOptions
from eden.utils.configs import EdenRLConfig


class _Config(EdenRLConfig):
    env_options: EnvOptions = EnvOptions(num_envs=1)
    test_param: str = "initial"
    test_vec: list[float] = [1.0, 2.0, 3.0]


# ---------------------------------------------------------------------------
# extra="forbid": a typo raises where it was written, and says what was meant
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("options_cls", [EnvOptions, BoxOptions, SensorOptions])
def test_a_misspelled_field_is_rejected(options_cls):
    with pytest.raises(ValidationError, match="extra_forbidden"):
        options_cls(bogus_field=1)


def test_the_rejection_lists_the_fields_and_suggests_the_near_match():
    with pytest.raises(ValidationError, match="extra_forbidden") as excinfo:
        EnvOptions(sim_dt=0.01, decimatoin=4)
    message = str(excinfo.value)
    assert "not a field of EnvOptions" in message
    assert "Did you mean 'decimation'?" in message
    assert "Valid fields:" in message and "sim_dt: float" in message

    with pytest.raises(ValidationError) as excinfo:
        BoxOptions(totally_unrelated_xyz=1)
    assert "Did you mean" not in str(excinfo.value) and "Valid fields:" in str(excinfo.value)


def test_a_real_type_error_beside_a_typo_is_left_as_pydantic_reports_it():
    with pytest.raises(ValidationError, match="int_parsing"):
        EnvOptions(decimation="not-an-int", bogus_field=1)


def test_a_nested_typo_is_attributed_to_the_nested_class():
    class _Child(ConfigurableOptions):
        child_field: int = 1

    class _Parent(ConfigurableOptions):
        child: _Child = _Child()
        parent_field: int = 0

    with pytest.raises(ValidationError) as excinfo:
        _Parent(child={"bogus_nested": 2})
    message = str(excinfo.value)
    assert "is not a field of _Child" in message and "is not a field of _Parent" not in message
    assert "child.bogus_nested" in message

    with pytest.raises(ValidationError) as excinfo:
        _Parent(bogus_top=9, child={"bogus_nested": 2})
    assert "is not a field of _Parent" in str(excinfo.value) and "is not a field of _Child" in str(excinfo.value)


def test_dynamic_containers_still_take_named_children():
    scene = SceneOptions(cube=BoxOptions())
    group = ObservationGroupOptions(my_term=ObservationTermOptions())

    assert "cube" in scene.keys() and type(scene.cube) is BoxOptions
    assert type(group.my_term) is ObservationTermOptions


# ---------------------------------------------------------------------------
# lock / unlock / clone / deepcopy
# ---------------------------------------------------------------------------


def _is_locked(options) -> bool:
    """Whether assigning to ``options`` raises; reads no private state."""
    name = next(iter(type(options).model_fields))
    try:
        setattr(options, name, getattr(options, name))
    except RuntimeError as exc:
        assert "locked" in str(exc)
        return True
    return False


def test_lock_propagates_down_and_unlock_recurses_into_children_lists_and_extras():
    config = _Config()
    grouped = GroupedEntityOptions(
        grouped_entities=[SphereOptions(radius=0.2), CylinderOptions(radius=0.2, height=0.3)]
    )
    scene = SceneOptions(cube=BoxOptions(size=(0.05, 0.05, 0.05)), group=grouped)

    for options, children in ((config, [config.env_options]), (scene, [scene.cube, *grouped.grouped_entities])):
        options.lock()
        assert _is_locked(options) and all(_is_locked(child) for child in children)
        options.unlock()
        assert not _is_locked(options) and not any(_is_locked(child) for child in children)


def test_a_locked_options_object_equals_the_same_values_unlocked():
    """The lock is a mode, not a value: ``save_as_json`` locks a config, and a value check against it (does the declared
    ``rec`` camera already carry these overlays?) must not answer no because one side is locked.
    """
    locked, fresh = _Config(), _Config()
    locked.lock()
    assert locked == fresh and locked.env_options == fresh.env_options
    fresh.env_options.num_envs += 1
    assert locked != fresh


def test_a_field_missing_from_an_options_object_does_not_equal_one_set_to_none():
    """The deprecated ``copy()`` can leave a field out of ``__dict__``; that is not the field's value being None."""
    from eden.options.camera import CameraOptions

    with_none, missing = CameraOptions(cam_near=None), CameraOptions(cam_near=None)
    del missing.__dict__["cam_near"]
    assert with_none == CameraOptions(cam_near=None) and with_none != missing


def test_lock_and_unlock_are_idempotent():
    options = EnvOptions(num_envs=8)
    options.unlock()
    options.lock()
    options.lock()
    assert _is_locked(options)
    options.unlock()
    options.unlock()
    options.num_envs = 16
    assert options.num_envs == 16


def test_a_child_locks_and_unlocks_its_own_subtree_only():
    """No upward walk, and a redundant parent ``lock()`` does not re-lock a child unlocked on its own."""
    config = _Config()
    config.lock()
    config.env_options.unlock()
    config.lock()

    config.env_options.num_envs = 42
    assert _is_locked(config)

    other = _Config()
    other.env_options.lock()
    other.test_param = "still editable"
    assert _is_locked(other.env_options)


@pytest.mark.parametrize("locked", [False, True])
def test_deepcopy_is_an_independent_snapshot_with_the_same_lock_state(locked):
    """Copying a locked source used to raise; the copy must not alias the source either way."""
    config = _Config()
    config.env_options.num_envs = 8
    if locked:
        config.lock()

    clone = copy.deepcopy(config)

    assert _is_locked(clone) is locked and _is_locked(clone.env_options) is locked
    assert clone.env_options is not config.env_options
    if not locked:
        clone.env_options.num_envs = 64
        assert config.env_options.num_envs == 8


def test_deepcopy_copies_extras_and_shares_what_cannot_be_copied():
    scene = SceneOptions(cube=BoxOptions(size=(0.05, 0.05, 0.05)))
    unpicklable = threading.Lock()
    scene.__pydantic_extra__["_unpicklable"] = unpicklable

    clone = copy.deepcopy(scene)

    assert clone.cube is not scene.cube
    assert clone.__pydantic_extra__["_unpicklable"] is unpicklable


def test_a_deepcopied_scene_enumerates_its_own_entities():
    """``keys`` is a cached closure over its owner; a copied cache would answer for the source.

    ``EnvBase`` builds what ``scene_options.keys()`` lists, so an entity added to ``env.config.clone()``
    after a build would silently not be built.
    """
    scene = SceneOptions(cube=BoxOptions(size=(1.0, 1.0, 1.0)))
    assert "cube" in scene.keys()  # noqa: SIM118 -- materializes the cache

    clone = copy.deepcopy(scene)
    clone.sphere = SphereOptions(radius=0.1)

    assert "sphere" in clone.keys()  # noqa: SIM118
    assert "sphere" not in scene.keys()  # noqa: SIM118


def test_deepcopy_does_not_rerun_model_post_init():
    """A copy is a snapshot, not a reconstruction that could write through a locked child."""
    calls = []

    class _Child(ConfigurableOptions):
        value: int = 0

    class _Parent(ConfigurableOptions):
        value: int = 0
        child: _Child = _Child()

        def model_post_init(self, context) -> None:
            calls.append(id(self))
            self.child.value = self.value

    source = _Parent(value=1)
    source.value = 2  # a legal state the derived child value no longer matches
    source.lock()
    calls.clear()

    clone = copy.deepcopy(source)

    assert calls == []
    assert (clone.value, clone.child.value) == (2, 1)


@pytest.mark.parametrize(("unlock", "expected"), [(False, True), (True, False)])
def test_clone_of_a_locked_config_keeps_or_drops_the_lock_and_the_type(unlock, expected):
    source = _Config()
    source.test_vec = [1.0, 2.0, 3.0]
    source.lock()

    clone = source.clone(unlock=unlock)

    assert type(clone) is _Config and type(clone.env_options) is EnvOptions
    assert _is_locked(clone) is expected and _is_locked(clone.env_options) is expected
    assert _is_locked(source) and _is_locked(source.env_options)
    if unlock:
        clone.test_vec[0] = 99.0
        assert source.test_vec[0] == 1.0  # deep: the list is not aliased


# ---------------------------------------------------------------------------
# Dumps are duck-typed: a parent's dump keeps what a subclass adds
# ---------------------------------------------------------------------------


def _reacher_runner_options():
    from eden.tasks import TASK_REGISTRY

    return TASK_REGISTRY.get("reacher")().runner_options


def test_a_parent_dump_keeps_the_fields_a_subclass_value_adds():
    """Declared-type serialization would drop ``std_type`` from an ``actor.distribution_cfg`` holding a
    Gaussian subclass, as it would across ~300 polymorphic fields, silently.
    """
    options = _reacher_runner_options()
    child = options.actor.distribution_cfg
    assert type(child) is not type(options.actor).model_fields["distribution_cfg"].annotation

    assert options.model_dump()["actor"]["distribution_cfg"] == child.model_dump()
    assert json.loads(options.model_dump_json())["actor"]["distribution_cfg"]["std_type"] == child.std_type


def test_narrowing_is_available_to_a_caller_that_asks_for_it():
    narrowed = _reacher_runner_options().model_dump(serialize_as_any=False)["actor"]["distribution_cfg"]

    assert "std_type" not in narrowed


# ---------------------------------------------------------------------------
# model_copy / updated / copy.copy
# ---------------------------------------------------------------------------
def test_a_forbid_options_copy_update_with_a_typo_raises_like_construction():
    """Pydantic wrote the typo into ``__dict__`` and nothing ever raised, not even build."""
    with pytest.raises(ValueError, match=r"'num_evns' is not a field of EnvOptions. Did you mean 'num_envs'\?"):
        EnvOptions().model_copy(update={"num_evns": 4})
    assert EnvOptions().model_copy(update={"num_envs": 4}).num_envs == 4
    assert EnvOptions().updated(num_envs=4).num_envs == 4


def test_a_deep_model_copy_works():
    """Pydantic calls ``__deepcopy__()`` with no memo; the override used to require one."""
    source = ObservationTermOptions(name="base_lin_vel", params={"entity_name": "robot"})

    copied = source.model_copy(deep=True)

    assert copied.params == source.params and copied.params is not source.params


def test_updated_is_an_editable_deep_copy_even_of_a_locked_source_and_model_copy_keeps_the_lock():
    source = ObservationTermOptions(name="base_lin_vel", params={"entity_name": "robot"})
    source.lock()

    copied = source.updated(scale=2.0)
    copied.scale = 3.0  # editable

    assert copied.params == source.params and copied.params is not source.params
    assert source._locked and source.scale == 1.0
    assert source.model_copy(update={"scale": 2.0})._locked

    # Unlocking reaches the children of a locked tree too.
    manager_options = RewardManagerOptions(upright=RewardTermOptions(name="upright", weight=2.0))
    manager_options.lock()
    variant = manager_options.updated()
    variant.upright.weight = 0.0
    assert manager_options.upright.weight == 2.0 and manager_options.upright._locked


def test_a_shallow_copy_recomputes_the_keys_it_enumerates():
    """``copy.copy`` and ``model_copy`` share ``__copy__``; a carried-over ``keys`` cache hid added children."""
    manager_options = RewardManagerOptions(upright=RewardTermOptions(name="upright"))
    manager_options.keys()

    copied = copy.copy(manager_options)
    copied.alive = RewardTermOptions(name="upright")

    assert "alive" in copied.keys() and "alive" not in manager_options.keys()
