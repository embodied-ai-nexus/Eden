"""The domain-randomization events in ``eden/managers/terms/events/domain_rand.py``.

The events randomize a real Panda's controllers, gains and DOFs, and push a free cube. Each test
wires in the action terms it needs as a real ``ActionManager`` (``install_manager``) and puts the
scene back afterwards (``env.reset()``, plus the gains and defaults a reset does not restore).
"""

from __future__ import annotations

import ast
import math
import re

import genesis.utils.geom as gu
import numpy as np
import pytest
import torch

from eden.constants import EventMode, ReferenceSource
from eden.managers.action_manager import ActionManager
from eden.managers.event_manager import EventManager
from eden.managers.modifiers.actions.actuators import (
    Compose,
    ConstantTorqueKick,
    Deadband,
    GearBacklash,
    MotorStrength,
    TorqueOffset,
)
from eden.managers.terms.actions.joint_actions import (
    ExplicitPDController,
    ImplicitPDController,
    ImplicitVelocityController,
    NullJointAction,
)
from eden.managers.terms.actions.task_space_actions import OperationalSpaceController
from eden.managers.terms.events.domain_rand import (
    ERFI50,
    ApplyExternalForce,
    ApplyExternalTorque,
    PushByVelocity,
    RandomizeComShift,
    RandomizeConstantTorqueKick,
    RandomizeDeadbandEpsilon,
    RandomizeDofsPosOffset,
    RandomizeGearBacklash,
    RandomizeKpKdGains,
    RandomizeLinkMassScale,
    RandomizeMassShift,
    RandomizeMotorStrength,
    RandomizeStartupDofsPos,
    RandomizeStartupDofsPosBias,
    RandomizeTorqueNoise,
)
from eden.options.managers.actions import ActionManagerOptions
from eden.options.managers.events import EventManagerOptions
from tests.managers.terms.conftest import PANDA_ARM, PANDA_KD, PANDA_KP, panda_options, wheel_options
from tests.support.envs import build_env, control_force_over_one_step

pytestmark = pytest.mark.requires_genesis

NUM_ENVS = 8
ARM3 = PANDA_ARM[:3]  # joint1..3: default position 0, so a DEFAULT-reference target is the action itself
KP = PANDA_KP


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    """A Panda (``robot``), the two-hinge wheel (``wheel``) and a free 0.2 kg cube (``cube``), in 8 envs.

    DOF and link info are batched, so gains and inertials can be randomized per env.
    """
    from eden.options import BoxOptions, EnvOptions, SceneOptions
    from eden.utils.configs import EdenConfig

    return build_env(
        EdenConfig(
            env_options=EnvOptions(num_envs=NUM_ENVS, batch_dofs_info=True, batch_links_info=True),
            scene_options=SceneOptions(
                robot=panda_options(),
                wheel=wheel_options(tmp_path_factory.mktemp("wheel") / "wheel.xml"),
                cube=BoxOptions(size=(0.1, 0.1, 0.1), default_root_pos=(0.0, 3.0, 2.0)),
            ),
        )
    )


@pytest.fixture(scope="module")
def unbatched_env():
    """One box, with neither DOF nor link info batched."""
    from eden.options import BoxOptions, EnvOptions, SceneOptions
    from eden.utils.configs import EdenConfig

    return build_env(
        EdenConfig(
            env_options=EnvOptions(num_envs=1, batch_dofs_info=False, batch_links_info=False),
            scene_options=SceneOptions(robot=BoxOptions(size=(0.1, 0.1, 0.1))),
        )
    )


@pytest.fixture(autouse=True)
def restore(env):
    """Put back what a test may change: the scene state, the Panda's gains, link masses and every entity's defaults."""
    robot = env.entities["robot"]
    kp, kd = robot.get_dofs_kp().clone(), robot.get_dofs_kd().clone()
    defaults = {
        name: entity.default_dofs_pos.clone()  # the bias writes into it in place
        for name, entity in env.entities.items()
        if isinstance(getattr(entity, "default_dofs_pos", None), torch.Tensor)
    }
    yield
    for name, value in defaults.items():
        env.entities[name].default_dofs_pos = value
    robot.set_dofs_kp(kp)
    robot.set_dofs_kd(kd)
    robot.set_mass_shift(torch.zeros(NUM_ENVS, robot.num_links), ls_idx_local=torch.arange(robot.num_links))
    env.reset()


@pytest.fixture
def actions(env, install_manager):
    """``actions(**terms)``: a real action manager holding ``terms``, wired in as ``env.action_manager``."""

    def install(**terms):
        return install_manager(env, "action_manager", ActionManager(env, ActionManagerOptions(**terms)))

    return install


def _pd(modifier=None, dofs=ARM3, cls=ExplicitPDController, entity="robot", **params):
    if modifier is not None:
        params["modifier"] = modifier
    return cls.configure(entity_name=entity, dofs_name=list(dofs), **params)


def _event(env, cls, **params):
    event = cls(env, cls.configure(**params))
    event.build()
    return event


def _torque(env, manager, action):
    """The joint1..3 torques the action manager commands for ``action`` (the error, at the default pose)."""
    manager.compute(torch.as_tensor(action, dtype=torch.float32).expand(NUM_ENVS, -1).clone())
    manager.apply_actions()
    return control_force_over_one_step(env, env.entities["robot"])[:, :3]


def _envs(*idx):
    return torch.tensor(idx)


# ---------------------------------------------------------------------------
# The reset-mode actuator randomizers: one modifier buffer each
#
# The modifier's per-env buffer is the event's whole output, so the family tests read it directly;
# what each buffer does to the commanded torque is pinned once per modifier further down.
# ---------------------------------------------------------------------------

ACTUATOR_DR = [
    pytest.param(RandomizeMotorStrength, MotorStrength, "motor_strength", 1.0, id="motor_strength"),
    pytest.param(RandomizeGearBacklash, GearBacklash, "backlash", 0.0, id="gear_backlash"),
    pytest.param(RandomizeConstantTorqueKick, ConstantTorqueKick, "torque_kick", 0.0, id="torque_kick"),
    pytest.param(RandomizeDeadbandEpsilon, Deadband, "deadband_epsilon", 0.0, id="deadband"),
]
DR_RANGE = (2.0, 3.0)  # excludes every baseline, so "in range" means "randomized"
DECOY = {MotorStrength: Deadband.configure(deadband_epsilon=0.0)}  # a neutral modifier of another type


def _dr_setup(env, actions, event_cls, modifier_cls, field, baseline, modifier=None):
    manager = actions(arm=_pd(modifier or modifier_cls.configure(**{field: baseline})))
    event = _event(env, event_cls, action_term_name="arm", **{f"{field}_range": DR_RANGE})
    return manager, event, lambda: getattr(event.modifier, f"_{field}")


@pytest.mark.parametrize(("event_cls", "modifier_cls", "field", "baseline"), ACTUATOR_DR)
def test_reset_draws_every_env_and_dof_from_the_range(env, actions, event_cls, modifier_cls, field, baseline):
    _, event, buffer = _dr_setup(env, actions, event_cls, modifier_cls, field, baseline)
    event.reset()
    drawn = buffer()
    assert ((drawn >= DR_RANGE[0]) & (drawn <= DR_RANGE[1])).all()
    assert drawn.unique().numel() > 1
    first = drawn.clone()
    event.reset()
    assert not torch.equal(buffer(), first)  # every reset redraws


@pytest.mark.parametrize(("event_cls", "modifier_cls", "field", "baseline"), ACTUATOR_DR)
@pytest.mark.parametrize(
    "envs_idx", [_envs(0, 2, 5), torch.tensor([True, False, True, False, False, True, False, False])]
)
def test_a_partial_reset_redraws_only_the_selected_envs(
    env, actions, event_cls, modifier_cls, field, baseline, envs_idx
):
    """The auto-reset hands a bool mask, a manual reset an index tensor: either selects the same envs."""
    _, event, buffer = _dr_setup(env, actions, event_cls, modifier_cls, field, baseline)
    event.reset(envs_idx)
    selected = torch.zeros(NUM_ENVS, dtype=torch.bool)
    selected[envs_idx] = True
    assert (buffer()[selected] >= DR_RANGE[0]).all()
    assert (buffer()[~selected] == baseline).all()


@pytest.mark.parametrize(("event_cls", "modifier_cls", "field", "baseline"), ACTUATOR_DR)
def test_the_draw_survives_the_env_reset_order(env, actions, install_manager, event_cls, modifier_cls, field, baseline):
    """The reset-mode event phase runs before ``action_manager.reset()`` restores the modifier baseline.

    A draw written there was restored away before the episode ran, leaving the randomization inert;
    the events write in ``reset()``, which runs after the restore. ``compute()`` is a no-op.
    """
    actions(arm=_pd(modifier_cls.configure(**{field: baseline})))
    options = EventManagerOptions(dr=event_cls.configure(action_term_name="arm", **{f"{field}_range": DR_RANGE}))
    events = install_manager(env, "event_manager", EventManager(env, options))
    (event,) = [term for _, name, term in events.iter_terms() if name == "dr"]
    buffer = getattr(event.modifier, f"_{field}")
    event.compute(slice(None))
    assert (buffer == baseline).all()
    env.reset()
    buffer = getattr(event.modifier, f"_{field}")
    assert ((buffer >= DR_RANGE[0]) & (buffer <= DR_RANGE[1])).all()


@pytest.mark.parametrize(("event_cls", "modifier_cls", "field", "baseline"), ACTUATOR_DR)
@pytest.mark.parametrize(
    ("extra", "match"), [({"mode": "interval"}, "mode"), ({"min_step_count_between_reset": 5}, "min_step_count")]
)
def test_only_the_plain_reset_mode_is_accepted(env, event_cls, modifier_cls, field, baseline, extra, match):
    """``compute()`` is a no-op, so any other mode or a throttle would randomize nothing."""
    with pytest.raises(ValueError, match=match):
        event_cls(env, event_cls.configure(action_term_name="arm", **extra))


MODIFIER_LOOKUP = [
    *ACTUATOR_DR,
    pytest.param(ERFI50, TorqueOffset, "torque_offset", 0.0, id="erfi50"),
    pytest.param(RandomizeTorqueNoise, TorqueOffset, "torque_offset", 0.0, id="torque_noise"),
]


@pytest.mark.parametrize(("event_cls", "modifier_cls", "field", "baseline"), MODIFIER_LOOKUP)
def test_the_modifier_is_found_inside_a_compose_chain(env, actions, event_cls, modifier_cls, field, baseline):
    """The decoy comes first: a lookup returning the head of the chain would pick the wrong modifier."""
    decoy = DECOY.get(modifier_cls, MotorStrength.configure(motor_strength=1.0))
    actions(arm=_pd(Compose.configure(modifiers=[decoy, modifier_cls.configure(**{field: baseline})])))
    event = _event(env, event_cls, action_term_name="arm")
    assert type(event.modifier) is modifier_cls


@pytest.mark.parametrize(("event_cls", "modifier_cls", "field", "baseline"), MODIFIER_LOOKUP)
@pytest.mark.parametrize("chain", ["absent-from-compose", "no-modifier"])
def test_build_raises_naming_the_missing_modifier(env, actions, event_cls, modifier_cls, field, baseline, chain):
    """Skipping would leave a config that reads as randomizing training without it."""
    decoy = DECOY.get(modifier_cls, MotorStrength.configure(motor_strength=1.0))
    actions(arm=_pd(Compose.configure(modifiers=[decoy]) if chain == "absent-from-compose" else None))
    with pytest.raises(ValueError, match=modifier_cls.__name__):
        _event(env, event_cls, action_term_name="arm")


@pytest.mark.parametrize(
    ("event_cls", "modifier_cls", "field"),
    [
        (RandomizeGearBacklash, GearBacklash, "backlash"),
        (RandomizeConstantTorqueKick, ConstantTorqueKick, "torque_kick"),
        (RandomizeDeadbandEpsilon, Deadband, "deadband_epsilon"),
    ],
)
def test_apply_as_ratio_scales_the_configured_per_dof_baseline(env, actions, event_cls, modifier_cls, field):
    actions(arm=_pd(modifier_cls.configure(**{field: [0.01, 0.02, 0.04]})))
    event = _event(env, event_cls, action_term_name="arm", apply_as_ratio=True, **{f"{field}_range": (10.0, 10.0)})
    event.reset()
    torch.testing.assert_close(getattr(event.modifier, f"_{field}"), torch.tensor([[0.1, 0.2, 0.4]] * NUM_ENVS))


@pytest.mark.parametrize(
    ("event_cls", "modifier_cls", "field"),
    [
        (RandomizeGearBacklash, GearBacklash, "backlash"),
        (RandomizeConstantTorqueKick, ConstantTorqueKick, "torque_kick"),
    ],
)
def test_a_negative_draw_is_clamped_to_zero(env, actions, event_cls, modifier_cls, field):
    actions(arm=_pd(modifier_cls.configure(**{field: 0.0})))
    event = _event(env, event_cls, action_term_name="arm", **{f"{field}_range": (-0.5, -0.1)})
    event.reset()
    assert (getattr(event.modifier, f"_{field}") == 0.0).all()


# The effect of each randomized buffer on what the controller commands, in the one env it was drawn for.


def test_randomized_motor_strength_scales_the_torque(env, actions):
    manager = actions(arm=_pd(MotorStrength.configure(motor_strength=1.0)))
    _event(env, RandomizeMotorStrength, action_term_name="arm", motor_strength_range=(0.5, 0.5)).reset(_envs(0))
    torque = _torque(env, manager, [0.1, 0.2, 0.3])
    torch.testing.assert_close(torque[0], 0.5 * KP * torch.tensor([0.1, 0.2, 0.3]))
    torch.testing.assert_close(torque[1], KP * torch.tensor([0.1, 0.2, 0.3]))


def test_a_randomized_torque_kick_adds_to_the_torque(env, actions):
    manager = actions(arm=_pd(ConstantTorqueKick.configure(torque_kick=0.0)))
    _event(env, RandomizeConstantTorqueKick, action_term_name="arm", torque_kick_range=(0.2, 0.2)).reset(_envs(0))
    torque = _torque(env, manager, [0.01] * 3)
    torch.testing.assert_close(torque[0], torch.full((3,), KP * 0.01 + 0.2))
    torch.testing.assert_close(torque[1], torch.full((3,), KP * 0.01))


def test_a_randomized_deadband_zeroes_the_torque_inside_it(env, actions):
    manager = actions(arm=_pd(Deadband.configure(deadband_epsilon=0.0)))
    _event(env, RandomizeDeadbandEpsilon, action_term_name="arm", deadband_epsilon_range=(10.0, 10.0)).reset(_envs(0))
    torque = _torque(env, manager, [0.5] * 3)
    torch.testing.assert_close(torque[0], torch.zeros(3))
    torch.testing.assert_close(torque[1], torch.full((3,), KP * 0.5))


def test_randomized_backlash_offsets_the_position_target(env, actions):
    manager = actions(arm=_pd(GearBacklash.configure(backlash=0.0)))
    _event(env, RandomizeGearBacklash, action_term_name="arm", backlash_range=(0.2, 0.2)).reset(_envs(0))
    manager.compute(torch.zeros(NUM_ENVS, 3))
    manager.compute(torch.ones(NUM_ENVS, 3))  # a positive move takes up the backlash on the positive side
    target = manager.processed_actions["arm"]
    torch.testing.assert_close(target[0], torch.full((3,), 1.2))
    torch.testing.assert_close(target[1], torch.ones(3))


# ---------------------------------------------------------------------------
# RandomizeKpKdGains
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("envs_idx", [None, _envs(2, 5)], ids=["all", "subset"])
def test_kp_kd_gains_are_the_defaults_scaled_by_a_draw_from_the_range(env, envs_idx):
    robot = env.entities["robot"]
    event = _event(env, RandomizeKpKdGains, entity_name="robot", kp_range=(0.5, 0.6), kd_range=(0.7, 0.8))
    event.compute(envs_idx=envs_idx)
    selected = torch.zeros(NUM_ENVS, dtype=torch.bool)
    selected[slice(None) if envs_idx is None else envs_idx] = True
    kp, kd = robot.get_dofs_kp(), robot.get_dofs_kd()
    assert ((kp[selected] >= 0.5 * KP) & (kp[selected] <= 0.6 * KP)).all()
    assert ((kd[selected] >= 0.7 * PANDA_KD) & (kd[selected] <= 0.8 * PANDA_KD)).all()
    assert kp[selected].unique().numel() > 1
    assert (kp[~selected] == KP).all() and (kd[~selected] == PANDA_KD).all()


PANDA_DOFS = PANDA_ARM + ["finger_joint1", "finger_joint2"]


@pytest.mark.parametrize(
    "dofs",
    [
        pytest.param(PANDA_DOFS, id="every-dof"),
        pytest.param(ARM3, id="dof-subset"),
        pytest.param(PANDA_DOFS[::-1], id="reordered"),
    ],
)
def test_kp_kd_gains_on_a_controller_change_its_torque_not_the_entity(env, actions, dofs):
    """Each of the controller's DOFs gets its own entity gain, doubled.

    A draw laid out over the entity's DOFs would make a subset controller raise and a reordered one
    swap gains between DOFs. Distinct per-DOF gains tell the columns apart.
    """
    robot = env.entities["robot"]
    manager = actions(arm=_pd(dofs=dofs))
    gains = torch.arange(1.0, 10.0)
    robot.set_dofs_kp(gains.expand(NUM_ENVS, -1))
    _event(env, RandomizeKpKdGains, entity_name="robot", action_term_name="arm", kp_range=(2.0, 2.0)).compute()
    manager.compute(torch.full((NUM_ENVS, len(dofs)), 0.1))
    manager.apply_actions()
    cols = [robot.dofs_name.index(name) for name in dofs]
    torch.testing.assert_close(
        control_force_over_one_step(env, robot)[:, cols], (2.0 * gains[cols] * 0.1).expand(NUM_ENVS, -1)
    )
    assert (robot.get_dofs_kp() == gains).all()


@pytest.mark.parametrize("mode", [EventMode.STARTUP, EventMode.RESET, EventMode.INTERVAL])
def test_a_kp_kd_draw_on_a_controller_survives_the_env_reset(env, actions, install_manager, mode):
    """The controller's reset re-reads its gains from the untouched entity after the draw.

    A reset-mode draw was undone within the reset that drew it, and a startup or interval draw by the next one;
    the event writes its draw back in ``reset()``, after the re-read.
    """
    manager = actions(arm=_pd())
    timing = {"interval_range_s": (0.02, 0.02)} if mode == EventMode.INTERVAL else {}
    gains = RandomizeKpKdGains.configure(mode=mode, action_term_name="arm", kp_range=(2.0, 2.0), **timing)
    events = install_manager(env, "event_manager", EventManager(env, EventManagerOptions(gains=gains)))
    (event,) = [term for _, name, term in events.iter_terms() if name == "gains"]
    if mode != EventMode.RESET:  # a reset-mode draw is the env reset's own
        event.compute(torch.ones(NUM_ENVS, dtype=torch.bool) if mode == EventMode.INTERVAL else None)
    for _ in range(2):
        env.reset()
        torch.testing.assert_close(_torque(env, manager, [0.1] * 3), torch.full((NUM_ENVS, 3), 2.0 * KP * 0.1))
    assert (env.entities["robot"].get_dofs_kp() == KP).all()


@pytest.mark.parametrize(
    ("mode", "factor"),
    [pytest.param(EventMode.RESET, 2.0, id="drawn"), pytest.param(EventMode.INTERVAL, 1.0, id="not-yet-drawn")],
)
def test_a_kp_kd_draw_on_a_controller_scales_the_entitys_current_gains(env, actions, install_manager, mode, factor):
    """The controller path keeps factors, not gains, so it composes with an entity-path gain draw.

    Each reset scales the gains the controller just re-read from the entity; an env the controller event has not
    drawn yet (an interval event before it fires) keeps the entity's gains as they are, not the build-time ones.
    """
    manager = actions(arm=_pd())
    timing = {"interval_range_s": (10.0, 10.0)} if mode == EventMode.INTERVAL else {}
    options = EventManagerOptions(
        entity_gains=RandomizeKpKdGains.configure(kp_range=(3.0, 3.0), kd_range=(1.0, 1.0)),
        arm_gains=RandomizeKpKdGains.configure(mode=mode, action_term_name="arm", kp_range=(2.0, 2.0), **timing),
    )
    install_manager(env, "event_manager", EventManager(env, options))
    env.reset()
    robot = env.entities["robot"]
    assert (robot.get_dofs_kp() == 3.0 * KP).all()
    torch.testing.assert_close(manager.get_term("arm")._kp, robot.get_dofs_kp()[:, :3] * factor)
    torch.testing.assert_close(_torque(env, manager, [0.1] * 3), torch.full((NUM_ENVS, 3), factor * 3.0 * KP * 0.1))


@pytest.mark.parametrize(
    ("term", "match"),
    [
        pytest.param(
            _pd(cls=ImplicitPDController),
            r"'arm' is of type ImplicitPDController, which has none.*set action_term_name=None",
            id="no-gain-cache",
        ),
        pytest.param(
            _pd(dofs=["flap"], entity="wheel"),
            r"'arm' controls entity 'wheel'.*Set entity_name='wheel'",
            id="other-entity",
        ),
    ],
)
def test_kp_kd_gains_reject_a_controller_they_cannot_write_at_build(env, actions, term, match):
    actions(arm=term)
    with pytest.raises(ValueError, match=match):
        _event(env, RandomizeKpKdGains, entity_name="robot", action_term_name="arm")


@pytest.mark.parametrize(
    ("cls", "flag"),
    [
        (RandomizeKpKdGains, "batch_dofs_info"),
        (RandomizeComShift, "batch_links_info"),
        (RandomizeMassShift, "batch_links_info"),
        (RandomizeLinkMassScale, "batch_links_info"),
    ],
)
def test_randomizers_needing_batched_info_refuse_an_unbatched_env(unbatched_env, cls, flag):
    """A ``ValueError`` at construction, not an ``assert`` that ``python -O`` strips."""
    with pytest.raises(ValueError, match=f"{flag}=True"):
        cls(unbatched_env, cls.configure())


# ---------------------------------------------------------------------------
# Link inertial randomization
# ---------------------------------------------------------------------------

SCALE_RANGE = (0.8, 1.2)
# Every Panda link but ``link0``, which is fixed to the world: its mass is never read, and the terms refuse it.
MOVING = ("link[1-7]", "hand", "*_finger", "*_pad")


def _mass_scale(env, links=MOVING):
    return _event(
        env, RandomizeLinkMassScale, entity_name="robot", links_name=list(links), mass_scale_range=SCALE_RANGE
    )


def test_mass_scale_draws_per_link_and_per_env_within_the_range(env):
    """The nominal it scales is each link's own mass (``get_mass()`` is the entity total, which would die at build)."""
    robot = env.entities["robot"]
    base = robot.get_links_inertial_mass()
    _mass_scale(env).compute()
    shift = robot.get_mass_shift()
    assert (shift >= base * (SCALE_RANGE[0] - 1.0) - 1e-6).all()
    assert (shift <= base * (SCALE_RANGE[1] - 1.0) + 1e-6).all()
    assert shift[:, int(base[0].argmax())].std() > 0.0  # not one factor reused across the batch


def test_mass_scale_leaves_unselected_envs_and_unnamed_links_alone(env):
    robot = env.entities["robot"]
    _mass_scale(env).compute(envs_idx=_envs(1, 3))
    shift = robot.get_mass_shift()
    others = [i for i in range(NUM_ENVS) if i not in (1, 3)]
    assert (shift[others] == 0.0).all() and (shift[[1, 3]] != 0.0).any()

    robot.set_mass_shift(torch.zeros_like(shift), ls_idx_local=torch.arange(robot.num_links))
    _, fingers = robot.find_named_links_idx_local(["*finger*"])
    _mass_scale(env, links=["*finger*"]).compute()
    shift = robot.get_mass_shift()
    untouched = [i for i in range(robot.num_links) if i not in fingers]
    assert (shift[:, untouched] == 0.0).all() and (shift[:, list(fingers)] != 0.0).any()


def test_mass_scale_rejects_a_non_positive_scale(env):
    with pytest.raises(ValueError, match="strictly positive"):
        _event(env, RandomizeLinkMassScale, entity_name="robot", links_name=list(MOVING), mass_scale_range=(0.0, 1.1))


def test_mass_shift_refuses_a_range_that_could_zero_a_link_and_applies_a_safe_one(env):
    """The stock (-1, 3) over every moving link could drive the 15 g fingers non-positive."""
    with pytest.raises(ValueError, match="non-positive"):
        _event(env, RandomizeMassShift, entity_name="robot", links_name=list(MOVING), mass_shift_range=(-1.0, 3.0))
    _event(
        env, RandomizeMassShift, entity_name="robot", links_name=list(MOVING), mass_shift_range=(0.05, 0.1)
    ).compute()
    shift = env.entities["robot"].get_mass_shift()[:, 1:]
    assert (shift >= 0.05 - 1e-6).all() and (shift <= 0.1 + 1e-6).all()


INERTIAL_DR = pytest.mark.parametrize(
    "cls, what",
    [(RandomizeMassShift, "mass"), (RandomizeLinkMassScale, "mass"), (RandomizeComShift, "COM")],
    ids=["mass_shift", "mass_scale", "com_shift"],
)


@INERTIAL_DR
def test_inertial_randomizers_refuse_a_link_fixed_to_the_world_and_name_the_moving_ones(env, cls, what):
    """``link0`` never moves, so a randomized mass or COM on it changes no dynamics yet reads back as set."""
    fixed = rf"matched \['link0'\], which are fixed to the world, so the solver never reads their {what}:"
    with pytest.raises(ValueError, match=fixed) as caught:
        _event(env, cls, entity_name="robot", links_name=["link0", "link1", "hand"])
    assert "Fix: links_name=['link1', 'hand']" in str(caught.value)
    assert "as a tensor" in str(caught.value)  # the wrong fix, named


@INERTIAL_DR
def test_inertial_randomizers_offer_the_entitys_moving_links_when_only_fixed_ones_matched(env, cls, what):
    fix = r"Fix: name moving links instead; links_name=\['\(\?!link0\$\)\.\*'\] selects all of them: link1, link2, "
    with pytest.raises(ValueError, match=fix + r".*right_gripper_pad\."):
        _event(env, cls, entity_name="robot", links_name=["link0"])


@pytest.mark.parametrize(
    "links_name, suggested",
    [
        (["*"], ["(?!link0$).*"]),  # the wildcard that used to be the advice: everything but link0
        (["link0", "left_*"], ["*left_*"]),
        (["link0", "link1", "hand"], ["link1", "hand"]),  # no pattern is shorter than the names
    ],
    ids=["wildcard", "common-substring", "explicit"],
)
def test_the_suggested_links_name_pastes_back_and_selects_exactly_the_moving_links(env, links_name, suggested):
    with pytest.raises(ValueError, match="selects exactly the moving links it matched") as caught:
        _event(env, RandomizeLinkMassScale, entity_name="robot", links_name=links_name)
    pasted = ast.literal_eval(re.search(r"Fix: links_name=(\[.*?\]) selects", str(caught.value)).group(1))
    assert pasted == suggested
    robot = env.entities["robot"]
    matched, _ = robot.find_named_links_idx_local(links_name)
    event = _event(env, RandomizeLinkMassScale, entity_name="robot", links_name=pasted)
    assert event.ls_idx_local.tolist() == [robot.get_link(name).idx_local for name in matched if name != "link0"]


# A free root whose box sits 0.3 m off its origin, with a fixed 10 g child: one body of two links.
_TOOL_MJCF = """<mujoco model="tool"><worldbody>
  <body name="root">
    <freejoint/><geom type="box" size="0.02 0.02 0.02" pos="0.3 0 0" mass="1.0"/>
    <body name="handle" pos="0 0.1 0"><geom type="box" size="0.01 0.01 0.01" mass="0.01"/></body>
  </body>
</worldbody></mujoco>"""
INERTIAL_ENVS = 4


@pytest.fixture(scope="module")
def inertial_env(tmp_path_factory):
    """The tool aligned (Genesis' default for it) and with ``align=False``, plus a primitive cube; link info batched."""
    from eden.options import BoxOptions, EntityOptions, EnvOptions, SceneOptions
    from eden.utils.configs import EdenConfig

    path = tmp_path_factory.mktemp("tool") / "tool.xml"
    path.write_text(_TOOL_MJCF)
    return build_env(
        EdenConfig(
            env_options=EnvOptions(num_envs=INERTIAL_ENVS, batch_links_info=True),
            scene_options=SceneOptions(
                tool=EntityOptions(file=str(path), default_root_pos=(0.0, 0.0, 0.5)),
                unaligned_tool=EntityOptions(file=str(path), default_root_pos=(1.0, 0.0, 0.5), align=False),
                cube=BoxOptions(size=(0.1, 0.1, 0.1), default_root_pos=(2.0, 0.0, 0.5)),
            ),
        )
    )


@pytest.mark.parametrize(
    ("cls", "entity", "params", "remedy"),
    [
        (RandomizeComShift, "tool", {}, "`EntityOptions(..., align=False)`"),
        (RandomizeComShift, "cube", {}, "BoxOptions, which Genesis always aligns"),
        (RandomizeMassShift, "tool", {"mass_shift_range": (0.001, 0.002)}, "`EntityOptions(..., align=False)`"),
        (RandomizeLinkMassScale, "tool", {}, "`EntityOptions(..., align=False)`"),
    ],
)
def test_inertial_randomizers_refuse_an_aligned_body_at_build(inertial_env, cls, entity, params, remedy):
    """Genesis does not support these writes on an aligned body.

    It raises on the first write on CPU and lets it through unchecked on GPU (its guard skips a device index tensor
    outside debug mode), so the event refuses them at build, naming the fix.
    """
    with pytest.raises(ValueError, match="anchored that body's frame") as exc:
        _event(inertial_env, cls, entity_name=entity, links_name=["*"], **params)
    assert remedy in str(exc.value) and "batch_links_info=True` does not help" in str(exc.value)


def test_the_mass_of_a_single_link_aligned_body_stays_writable(inertial_env):
    """Genesis' exemption: rescaling the one link of an aligned body leaves its anchor where it is."""
    cube = inertial_env.entities["cube"]
    _event(inertial_env, RandomizeLinkMassScale, entity_name="cube", links_name=["*"]).compute()
    assert (cube.get_mass_shift() != 0.0).any()
    cube.set_mass_shift(torch.zeros(INERTIAL_ENVS, 1), ls_idx_local=torch.arange(1))


def test_align_false_makes_a_free_bodys_com_writable_per_env(inertial_env):
    _event(
        inertial_env,
        RandomizeComShift,
        entity_name="unaligned_tool",
        links_name=["root"],
        com_shift_x_range=(-0.05, 0.05),
        com_shift_y_range=(-0.05, 0.05),
        com_shift_z_range=(-0.05, 0.05),
    ).compute()
    shift = inertial_env.entities["unaligned_tool"].get_COM_shift(ls_idx_local=[0])
    assert shift.shape == (INERTIAL_ENVS, 1, 3)
    assert (shift.abs() <= 0.05 + 1e-6).all() and shift[:, 0, 0].std() > 0.0


# ---------------------------------------------------------------------------
# PushByVelocity
# ---------------------------------------------------------------------------


def test_a_zero_push_leaves_a_tilted_spinning_base_spinning_about_the_same_world_axis(env):
    """A free joint's angular DOFs are body-frame: the push rotates the world angular velocity in and out."""
    cube = env.entities["cube"]
    tilt = torch.tensor(gu.xyz_to_quat(np.array([25.0, -15.0, 40.0]), rpy=True, degrees=True), dtype=torch.float32)
    cube.set_quat(tilt.expand(NUM_ENVS, 4).clone())
    lin, omega = torch.tensor([[0.1, 0.2, -0.1]] * NUM_ENVS), torch.tensor([[0.5, -0.3, 0.8]] * NUM_ENVS)
    cube.set_dofs_vel(torch.cat([lin, gu.inv_transform_by_quat(omega, cube.get_quat())], dim=1))
    _event(env, PushByVelocity, entity_name="cube").compute()  # an empty velocity_range: a zero push
    torch.testing.assert_close(cube.get_ang(), omega, atol=1e-4, rtol=0)
    torch.testing.assert_close(cube.get_vel(), lin, atol=1e-5, rtol=0)


def test_the_mask_form_pushes_exactly_the_due_envs_as_the_row_form_does(env):
    """The sync-free form (read every row, let Genesis apply the mask) against the row gather.

    ``_write_every_row`` is the term's only switch between the two, so the test flips it.
    """
    cube = env.entities["cube"]
    push = _event(
        env,
        PushByVelocity,
        mode="interval",
        interval_range_s=(1.0, 1.0),
        entity_name="cube",
        velocity_range={"x": (-0.5, 0.5), "yaw": (-0.7, 0.7)},
    )
    assert push._write_every_row  # a free body takes the fast form
    due = torch.tensor([True, False, True, False, False, True, False, False])
    start = torch.tensor([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]] * NUM_ENVS)
    pushed = {}
    for write_every_row in (True, False):
        cube.set_dofs_vel(start, dofs_idx_local=range(6))
        push._write_every_row = write_every_row
        torch.manual_seed(0)
        push.compute(envs_idx=due)
        pushed[write_every_row] = cube.get_dofs_vel(dofs_idx_local=range(6)).clone()
    torch.testing.assert_close(pushed[True], pushed[False])
    assert torch.equal(pushed[True][~due], start[~due]) and not torch.allclose(pushed[True][due], start[due])


# ---------------------------------------------------------------------------
# ApplyExternalForce / ApplyExternalTorque
# ---------------------------------------------------------------------------


def _pushed(env, event, envs_idx=None):
    """Which envs' cube one physics step accelerates upward (``+z``) or spins about ``z``."""
    cube = env.entities["cube"]
    event.compute(envs_idx=envs_idx)
    vel, ang = cube.get_vel().clone(), cube.get_ang().clone()
    env.scene.step()  # the wrench applies for one step; gravity alone would slow the cube by 0.049 m/s
    return (cube.get_vel() - vel)[:, 2] > 0.0, (cube.get_ang() - ang)[:, 2] > 0.1


def _force(env, **params):
    return _event(env, ApplyExternalForce, entity_name="cube", links_name=[".*"], force_z_range=(10.0, 10.0), **params)


@pytest.mark.parametrize(("prob", "expected"), [(1.0, [True] * NUM_ENVS), (0.0, [False] * NUM_ENVS)])
def test_the_external_force_reaches_the_envs_its_probability_selects(env, prob, expected):
    assert _pushed(env, _force(env, prob=prob))[0].tolist() == expected


def test_a_fractional_probability_draws_each_env_independently(env):
    """Seeded: with ``prob=0.5`` some envs are pushed and some not, and an env outside the selection never is."""
    torch.manual_seed(0)
    subset = _envs(0, 1, 2, 3, 4, 5)
    pushed, _ = _pushed(env, _force(env, prob=0.5), envs_idx=subset)
    assert pushed[:6].any() and not pushed[:6].all()
    assert not pushed[6:].any()


def test_the_external_torque_spins_the_body(env):
    event = _event(env, ApplyExternalTorque, entity_name="cube", links_name=[".*"], torque_z_range=(1.0, 1.0))
    assert _pushed(env, event)[1].all()


@pytest.mark.parametrize("cls", [ApplyExternalForce, ApplyExternalTorque])
@pytest.mark.parametrize("prob", [-0.1, 1.5])
def test_a_probability_outside_the_unit_interval_raises(env, cls, prob):
    with pytest.raises(ValueError, match="prob"):
        _event(env, cls, entity_name="cube", links_name=[".*"], prob=prob)


@pytest.mark.parametrize("ref", ["link_origin", "link_com"])
def test_each_supported_reference_frame_reaches_the_solver(env, ref):
    """Genesis rejects string frames, so the configured name is resolved to ``gs.link_ref_frame`` at build."""
    assert _pushed(env, _force(env, ref=ref))[0].all()


def test_an_unknown_reference_frame_raises_at_build_naming_the_supported_ones(env):
    """``root_com`` was dropped by Genesis for wrenches."""
    with pytest.raises(ValueError, match=r"'root_com'.*link_com.*link_origin"):
        _force(env, ref="root_com")


# ---------------------------------------------------------------------------
# RandomizeDofsPosOffset
# ---------------------------------------------------------------------------


def _offset_event(env, **params):
    return _event(env, RandomizeDofsPosOffset, **{"entity_name": "robot", "dofs_pos_range": (0.25, 0.25), **params})


@pytest.fixture
def split_robot(actions):
    """Two position controllers splitting the Panda's arm, as a per-limb variant does, plus distractors:
    a position controller on another entity and a velocity controller on this one.
    """
    manager = actions(
        legs=_pd(dofs=["joint1", "joint2"], cls=ImplicitPDController, reference_source=ReferenceSource.ZERO),
        upper=_pd(dofs=["joint3", "joint4"], cls=ImplicitPDController, reference_source=ReferenceSource.ZERO),
        other=_pd(dofs=["flap"], cls=ImplicitPDController, entity="wheel"),
        velocity=_pd(dofs=["joint5"], cls=ImplicitVelocityController),
    )
    return {name: manager.get_term(name) for name in ("legs", "upper", "other", "velocity")}


def test_offset_on_every_position_controller_of_the_entity_by_default(env, split_robot):
    """A split robot keeps the bias on all its joints without naming a term."""
    event = _offset_event(env)
    assert event.action_term_name is None
    event.compute()
    for name in ("legs", "upper"):
        torch.testing.assert_close(split_robot[name].effective_offset, torch.full((NUM_ENVS, 2), 0.25))
    assert (split_robot["other"].effective_offset == 0.0).all()


def test_offset_on_exactly_the_named_controllers(env, split_robot):
    _offset_event(env, action_term_name=["upper"]).compute()
    torch.testing.assert_close(split_robot["upper"].effective_offset, torch.full((NUM_ENVS, 2), 0.25))
    assert (split_robot["legs"].effective_offset == 0.0).all()


def test_offset_adds_to_the_default_fold_and_the_static_offset_and_is_redrawn(env, actions):
    """It replaces neither; a second draw replaces the first instead of accumulating."""
    robot = env.entities["robot"]
    manager = actions(
        default=_pd(dofs=["joint4", "joint5"], cls=ImplicitPDController, offset=0.1),
        zero=_pd(dofs=["joint6"], cls=ImplicitPDController, reference_source=ReferenceSource.ZERO),
    )
    event = _offset_event(env)
    fold = robot.default_dofs_pos[:, [3, 4]]
    for _ in range(2):
        event.compute()
        torch.testing.assert_close(manager.get_term("default").effective_offset, fold + 0.1 + 0.25)
        torch.testing.assert_close(manager.get_term("zero").effective_offset, torch.full((NUM_ENVS, 1), 0.25))


def test_a_partial_offset_draw_touches_only_the_selected_envs(env, actions):
    """The controller owns the default fold; an event keeping its own copy raised an IndexError here."""
    manager = actions(ctrl=_pd(dofs=["joint4", "joint6"], cls=ImplicitPDController))
    fold = env.entities["robot"].default_dofs_pos[:, [3, 5]]
    _offset_event(env, dofs_pos_range=(0.5, 0.5)).compute()
    _offset_event(env, dofs_pos_range=(99.0, 99.0)).compute(envs_idx=_envs(0, 2))
    offset = manager.get_term("ctrl").effective_offset - fold
    assert (offset[[0, 2]] == 99.0).all() and (offset[[1, 3, 4, 5, 6, 7]] == 0.5).all()


@pytest.mark.parametrize("offset_first", [True, False])
def test_offset_composes_with_the_startup_bias_in_either_order(env, actions, offset_first):
    manager = actions(ctrl=_pd(cls=ImplicitPDController))
    fold = env.entities["robot"].default_dofs_pos[:, :3].clone()
    bias = _event(env, RandomizeStartupDofsPosBias, action_terms_name=["ctrl"], dofs_pos_range=(0.1, 0.1))
    jitter = _offset_event(env, dofs_pos_range=(0.02, 0.02))
    for event in (jitter, bias) if offset_first else (bias, jitter):
        event.compute()
    torch.testing.assert_close(manager.get_term("ctrl").effective_offset, fold + 0.12)


@pytest.mark.parametrize(
    ("action_term_name", "match"),
    [
        ("velocity", r"'velocity' is a ImplicitVelocityController, not a joint position controller"),
        ("other", r"'other' controls entity 'wheel', not 'robot'"),
        ([], r"action_term_name is an empty list"),
        (["legs", "upper", "legs"], r"lists \['legs'\] more than once"),
    ],
)
def test_offset_rejects_a_bad_selection_at_build(env, split_robot, action_term_name, match):
    with pytest.raises(ValueError, match=match):
        _offset_event(env, action_term_name=action_term_name)


def test_offset_on_an_entity_without_position_controllers_raises(env, actions):
    actions(velocity=_pd(dofs=["joint1"], cls=ImplicitVelocityController))
    with pytest.raises(ValueError, match=r"(?s)on entity 'robot'.*has none.*'velocity': 'ImplicitVelocityController'"):
        _offset_event(env)


# ---------------------------------------------------------------------------
# RandomizeStartupDofsPos
# ---------------------------------------------------------------------------


def test_startup_dofs_pos_writes_absolute_positions(env):
    _event(env, RandomizeStartupDofsPos, entity_name="wheel", dofs_pos_range=(0.7, 0.7)).compute()
    torch.testing.assert_close(env.entities["wheel"].get_dofs_pos(), torch.full((NUM_ENVS, 2), 0.7))


@pytest.mark.parametrize(("ratio", "flap"), [(0.0, -1.0), (0.5, 0.0), (1.0, 1.0)])
def test_startup_dofs_pos_as_a_ratio_spans_each_dofs_limits(env, ratio, flap):
    """The limited flap spans its ±1 rad; the unlimited spin falls back to a 2 pi range from -pi."""
    _event(
        env, RandomizeStartupDofsPos, entity_name="wheel", dofs_pos_range=(ratio, ratio), apply_as_ratio=True
    ).compute()
    expected = torch.tensor([[-math.pi + ratio * 2 * math.pi, flap]] * NUM_ENVS)
    torch.testing.assert_close(env.entities["wheel"].get_dofs_pos(), expected, atol=1e-5, rtol=0.0)


def test_startup_dofs_pos_as_a_ratio_writes_only_the_selected_envs(env):
    """Per-env limits and a subset broadcast together (a shape mismatch regression)."""
    robot = env.entities["robot"]
    before = robot.get_dofs_pos().clone()
    _event(env, RandomizeStartupDofsPos, entity_name="robot", dofs_pos_range=(1.0, 1.0), apply_as_ratio=True).compute(
        envs_idx=_envs(0, 2)
    )
    _, upper = robot.get_dofs_limit()
    torch.testing.assert_close(robot.get_dofs_pos()[[0, 2]], upper[[0, 2]])
    torch.testing.assert_close(robot.get_dofs_pos()[[1, 3]], before[[1, 3]])


# ---------------------------------------------------------------------------
# ERFI50: random actuation offset (RAO) in some envs, random force injection (RFI) in the rest
# ---------------------------------------------------------------------------


def _erfi(env, actions, erfi_ratio):
    manager = actions(arm=_pd(TorqueOffset.configure(torque_offset=0.0)))
    event = _event(env, ERFI50, action_term_name="arm", erfi_ratio=erfi_ratio)
    event.reset()
    return manager, event


UPPER = 7.0 / 9.0  # the default bound


def test_rao_is_one_constant_offset_per_env_for_the_whole_episode(env, actions):
    """What the controller adds, read at zero error: equal across DOFs, unchanged by compute()."""
    manager, event = _erfi(env, actions, erfi_ratio=1.0)
    event.compute()
    first = _torque(env, manager, [0.0] * 3)
    event.compute()
    torch.testing.assert_close(_torque(env, manager, [0.0] * 3), first)
    torch.testing.assert_close(first, first[:, :1].expand(-1, 3))
    assert ((first >= 0.0) & (first <= 3.0 * UPPER + 1e-6)).all()


def test_rfi_redraws_independent_per_dof_noise_every_compute(env, actions):
    manager, event = _erfi(env, actions, erfi_ratio=0.0)
    event.compute()
    first = _torque(env, manager, [0.0] * 3)
    event.compute()
    second = _torque(env, manager, [0.0] * 3)
    assert not torch.allclose(first, second) and not torch.allclose(first[:, 0], first[:, 1])
    assert ((first >= -UPPER) & (first <= UPPER)).all()


def test_the_erfi_draw_survives_the_env_reset_order(env, actions, install_manager):
    """``action_manager.reset()`` restores the TorqueOffset baseline first; ERFI50's reset writes after it."""
    manager = actions(arm=_pd(TorqueOffset.configure(torque_offset=0.0)))
    events = EventManagerOptions(
        erfi=ERFI50.configure(action_term_name="arm", erfi_ratio=1.0, interval_range_s=(0.02, 0.02))
    )
    install_manager(env, "event_manager", EventManager(env, events))
    env.reset()
    offset = _torque(env, manager, [0.0] * 3)
    assert (offset > 0).all()  # every env RAO: one positive offset per env


def test_erfi_partial_reset_redraws_only_the_selected_envs(env, actions):
    manager, event = _erfi(env, actions, erfi_ratio=1.0)
    event.compute()
    before = _torque(env, manager, [0.0] * 3)
    event.erfi_ratio = 0.0  # the redrawn envs switch to RFI
    event.reset(envs_idx=_envs(0, 2))
    event.compute(envs_idx=_envs(0, 2))
    after = _torque(env, manager, [0.0] * 3)
    torch.testing.assert_close(after[[1, 3, 4, 5, 6, 7]], before[[1, 3, 4, 5, 6, 7]])
    assert not torch.allclose(after[[0, 2]], before[[0, 2]])


def test_erfi_compute_through_a_bool_mask_writes_only_the_due_envs(env, actions):
    """The interval tick hands a bool mask; only those envs get this step's noise."""
    manager, event = _erfi(env, actions, erfi_ratio=0.0)
    event.compute()
    before = _torque(env, manager, [0.0] * 3)
    due = torch.tensor([False, True, False, True, False, True, False, False])
    event.compute(envs_idx=due)
    after = _torque(env, manager, [0.0] * 3)
    torch.testing.assert_close(after[~due], before[~due])
    assert (after[due] != before[due]).all()


# ---------------------------------------------------------------------------
# RandomizeStartupDofsPosBias: a per-env calibration drift baked into the entity defaults
# ---------------------------------------------------------------------------


def _bias(env, **params):
    return _event(env, RandomizeStartupDofsPosBias, **{"entity_name": "robot", "dofs_pos_range": (0.1, 0.1), **params})


def test_bias_shifts_the_entity_defaults_once(env, actions):
    actions()
    robot = env.entities["robot"]
    before = robot.default_dofs_pos.expand(NUM_ENVS, -1).clone()
    event = _bias(env)
    for _ in range(2):  # idempotent: a startup term applies its draw once
        event.compute()
        torch.testing.assert_close(robot.default_dofs_pos, before + 0.1)


def test_bias_moves_each_listed_controller_on_its_own_dofs_in_lockstep(env, actions):
    """The fold is shifted by the same delta, not re-folded from the shifted defaults (that would add it twice)."""
    manager = actions(legs=_pd(dofs=["joint1", "joint2"]), arm=_pd(dofs=["joint3", "joint4", "joint5"]))
    robot = env.entities["robot"]
    folds = {"legs": robot.default_dofs_pos[:, :2].clone(), "arm": robot.default_dofs_pos[:, 2:5].clone()}
    _bias(env, action_terms_name=["legs", "arm"]).compute()
    for name, fold in folds.items():
        torch.testing.assert_close(manager.get_term(name).effective_offset, fold.expand(NUM_ENVS, -1) + 0.1)


def test_bias_moves_a_mimic_controllers_drivers_only(env, actions):
    """Its `_offset` spans the drivers, the action columns; a follower's target follows its driver's."""
    manager = actions(ctrl=_pd(dofs=["joint1", "joint2"], mimic_dofs={"joint3": ("joint2", 1.0, 0.0)}))
    fold = env.entities["robot"].default_dofs_pos[:, :2].clone()
    _bias(env, action_terms_name=["ctrl"]).compute()
    torch.testing.assert_close(manager.get_term("ctrl").effective_offset, fold.expand(NUM_ENVS, -1) + 0.1)


def test_bias_survives_a_per_reset_offset_on_top(env, actions):
    manager = actions(ctrl=_pd())
    robot = env.entities["robot"]
    before = robot.default_dofs_pos[:, :3].expand(NUM_ENVS, -1).clone()
    _bias(env, action_terms_name=["ctrl"]).compute()
    _offset_event(env, action_term_name="ctrl", dofs_pos_range=(0.0, 0.0)).compute()
    torch.testing.assert_close(manager.get_term("ctrl").effective_offset, before + 0.1)
    torch.testing.assert_close(robot.default_dofs_pos[:, :3], before + 0.1)


def test_bias_accepts_controllers_that_do_not_snapshot_the_defaults(env, actions):
    """A ZERO-source controller never folded the defaults, and OSC refreshes its reference at reset."""
    actions(
        zero=_pd(dofs=["finger_joint1"], reference_source=ReferenceSource.ZERO),
        osc=OperationalSpaceController.configure(entity_name="robot", dofs_name=PANDA_ARM, ee_link_name="hand"),
    )
    before = env.entities["robot"].default_dofs_pos.expand(NUM_ENVS, -1).clone()
    _bias(env).compute()
    torch.testing.assert_close(env.entities["robot"].default_dofs_pos, before + 0.1)


@pytest.mark.parametrize(
    ("terms", "listed", "match"),
    [
        ({"ctrl": _pd(reference_source=ReferenceSource.ZERO)}, ["ctrl"], "reference_source"),
        ({"wrist": _pd(dofs=["flap"], entity="wheel")}, ["wrist"], "targets entity"),
        ({"ctrl": _pd()}, ["ctrl", "ctrl"], "duplicate"),
        ({"null": _pd(cls=NullJointAction)}, ["null"], "not a `_JointPDControllerBase`"),
        (
            {"listed": _pd(dofs=["joint1"]), "forgotten": _pd(dofs=["joint2", "joint3"])},
            ["listed"],
            "forgotten.*not listed",
        ),
        ({"null": _pd(cls=NullJointAction)}, [], "caches `default_dofs_pos`"),
    ],
    ids=["zero-source-listed", "other-entity", "duplicate", "not-pd", "unlisted-default-pd", "null-on-entity"],
)
def test_bias_rejects_a_controller_whose_cache_would_go_stale(env, actions, terms, listed, match):
    """Any cache of the pre-bias defaults left out of the lockstep update fails loud at build."""
    actions(**terms)
    with pytest.raises(ValueError, match=match):
        _bias(env, action_terms_name=listed)
