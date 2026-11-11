"""Tests for ``eden.managers.motion.resync``: an env whose clip ends is teleported onto a fresh start.

One mimic env with ``resync_on_motion_end`` and reference-state-initialization events for the G1 and
the box; the clips carry explicit (tilted-base) velocities so the teleported qvel is checkable.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from genesis.utils.geom import inv_transform_by_quat

import eden as en
from eden.managers.event_manager import EVENT_TERM_REGISTRY
from eden.utils.misc import to_snake_case
from eden.managers.terms.events.motion import (
    ObjectReferenceStateInitialization,
    ReferenceStateInitialization,
    ReferenceStateInitializationBase,
)
from tests.support.motion import (
    FRAMERATE,
    ROBOT_DOFS,
    build_mimic_env,
    manager_options,
    pin_clips,
    term_options,
    write_clip,
)

FRAMES = 10
DT = 1.0 / FRAMERATE


class CustomReferenceStateInitialization(ReferenceStateInitialization):
    """A user subclass: the resync must find it by ``isinstance``, not by class name."""


class NoTeleportReferenceStateInitialization(ReferenceStateInitializationBase):
    """An RSI flavour that forgot to implement ``_stage_teleport``."""

    def compute(self, envs_idx=None):
        pass


for _term_cls in (CustomReferenceStateInitialization, NoTeleportReferenceStateInitialization):
    if to_snake_case(_term_cls.__name__) not in EVENT_TERM_REGISTRY:
        EVENT_TERM_REGISTRY.register()(_term_cls)


def _tilted(num_frames: int, roll: float) -> np.ndarray:
    return np.tile([math.cos(roll / 2), math.sin(roll / 2), 0.0, 0.0], (num_frames, 1)).astype(np.float32)


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    root = tmp_path_factory.mktemp("resync")
    for clip in range(2):
        write_clip(
            root / f"clip_{clip:02d}.npz",
            FRAMES,
            clip=clip,
            overrides={
                "robot.base_quat_w": _tilted(FRAMES, 0.3),
                "robot.base_lin_vel_w": np.tile([0.5, 0.0, 0.0], (FRAMES, 1)).astype(np.float32),
                "robot.base_ang_vel_w": np.tile([0.0, 0.0, 1.0], (FRAMES, 1)).astype(np.float32),
                "robot.dofs_vel": np.full((FRAMES, len(ROBOT_DOFS)), 0.05, dtype=np.float32),
                "object.base_quat_w": _tilted(FRAMES, -0.4),
                "object.base_lin_vel_w": np.tile([0.0, 0.2, 0.0], (FRAMES, 1)).astype(np.float32),
                "object.base_ang_vel_w": np.tile([0.0, 0.0, 2.0], (FRAMES, 1)).astype(np.float32),
                "object.dofs_vel": np.zeros((FRAMES, 0), dtype=np.float32),
            },
        )
    return root


def _motion_options(root, **manager_kwargs):
    term = term_options(root, motion_start_from_middle_range=(0.0, 0.5), dataset={"recalculate_velocities": False})
    return manager_options(terms={"default": term}, resync_on_motion_end=True, **manager_kwargs)


def _events(**extra):
    from eden.options.managers.events import EventManagerOptions

    return EventManagerOptions(
        reset_robot=ReferenceStateInitialization.configure(
            mode=en.EventMode.RESET, entity_name="robot", dofs_position_range=(0.0, 0.0)
        ),
        reset_object=ObjectReferenceStateInitialization.configure(mode=en.EventMode.RESET, entity_name="object"),
        **extra,
    )


@pytest.fixture(scope="module")
def env(library):
    from eden.terms import EventTerm

    return build_mimic_env(
        library,
        motion_options=_motion_options(library),
        event_options=_events(
            reset_again=CustomReferenceStateInitialization.configure(
                mode=en.EventMode.RESET, entity_name="robot", dofs_position_range=(0.0, 0.0)
            ),
            nudge_object=EventTerm.configure(
                func=en.events.reset_base_state_uniform,
                mode="reset",
                params={"entity_name": "object", "pos_x_range": (0.0, 0.0), "pos_y_range": (0.0, 0.0)},
            ),
        ),
    )


def _wrap_envs_1_and_3(env):
    """Put envs 0 and 2 at the start of clip 0 and envs 1 and 3 on the last frame, then step the clock."""
    env.reset()
    pin_clips(env.motion_manager.get_term("default"), [0, 0, 0, 0], start_s=[0.5 * DT, (FRAMES - 0.5) * DT] * 2)
    robot, obj = env.entities["robot"], env.entities["object"]
    before = robot.get_qpos().clone(), obj.get_qpos().clone()
    env.motion_manager.compute(env.dt)
    return before


def test_a_quiet_step_teleports_nothing(env):
    """The teleport runs every step through the wrap mask: with no env wrapped, it writes nothing."""
    env.reset()
    pin_clips(env.motion_manager.get_term("default"), [0, 1, 0, 1], start_s=0.5 * DT)
    entities = [env.entities["robot"], env.entities["object"]]
    before = [(e.get_qpos().clone(), e.get_dofs_vel().clone()) for e in entities]
    start_times = env.motion_manager.get_term("default")._motion_buffer_start_time_s.clone()

    env.motion_manager.compute(env.dt)

    for entity, (qpos, vel) in zip(entities, before):
        assert torch.equal(entity.get_qpos(), qpos)
        assert torch.equal(entity.get_dofs_vel(), vel)
    assert torch.equal(env.motion_manager.get_term("default")._motion_buffer_start_time_s, start_times)


def test_a_wrapped_env_restarts_inside_its_clip_and_keeps_its_episode(env):
    episode_steps = env.episode_length_buf.clone()
    _wrap_envs_1_and_3(env)

    ref = env.motion_manager.data["robot"]
    assert ref.validity[:, 0].all()  # the wrapped envs no longer read past the clip end
    phase = env.motion_manager.get_motion_phase("robot").cpu()
    assert (phase[[1, 3]] <= 0.5).all()
    assert torch.equal(env.episode_length_buf, episode_steps)
    assert not env.termination_manager.terminated.any()


def test_the_robot_is_teleported_onto_the_new_reference_with_its_velocity(env):
    (robot_before, _) = _wrap_envs_1_and_3(env)

    robot = env.entities["robot"]
    ref = env.motion_manager.data["robot"]
    qpos = robot.get_qpos()
    assert torch.equal(qpos[[0, 2]], robot_before[[0, 2]])  # not wrapped: untouched
    for e in (1, 3):
        torch.testing.assert_close(qpos[e, :3], ref.base_pos_w[e, 0])
        torch.testing.assert_close(qpos[e, 3:7], ref.base_quat_w[e, 0])
        torch.testing.assert_close(qpos[e, 7:], ref.dofs_pos[e, 0])
    base_dofs = robot.find_free_base_dofs_idx_local()
    base_vel = robot.get_dofs_vel(dofs_idx_local=base_dofs)[[1, 3]].cpu()
    # Linear base DOFs are world-frame; the free joint's angular DOFs are body-frame.
    torch.testing.assert_close(base_vel[:, :3], torch.tensor([[0.5, 0.0, 0.0]] * 2))
    expected_ang_b = inv_transform_by_quat(torch.tensor([[0.0, 0.0, 1.0]] * 2), ref.base_quat_w[[1, 3], 0].cpu())
    torch.testing.assert_close(base_vel[:, 3:], expected_ang_b)
    # set_qpos after set_dofs_vel must not zero the velocity it follows.
    torch.testing.assert_close(robot.get_dofs_vel()[[1, 3]].cpu(), torch.full((2, len(ROBOT_DOFS)), 0.05))


def test_the_free_object_is_teleported_too(env):
    (_, object_before) = _wrap_envs_1_and_3(env)

    obj = env.entities["object"]
    ref = env.motion_manager.data["object"]
    qpos = obj.get_qpos()
    torch.testing.assert_close(qpos[[0, 2]], object_before[[0, 2]])
    torch.testing.assert_close(qpos[[1, 3], :3], ref.base_pos_w[[1, 3], 0])
    torch.testing.assert_close(qpos[[1, 3], 3:7], ref.base_quat_w[[1, 3], 0])
    vel = obj.get_dofs_vel()[[1, 3]].cpu()
    torch.testing.assert_close(vel[:, :3], torch.tensor([[0.0, 0.2, 0.0]] * 2))
    expected_ang_b = inv_transform_by_quat(torch.tensor([[0.0, 0.0, 2.0]] * 2), ref.base_quat_w[[1, 3], 0].cpu())
    torch.testing.assert_close(vel[:, 3:], expected_ang_b)


def test_every_rsi_flavour_and_subclass_is_found_by_type(env):
    terms = env.motion_manager._resync._reference_init_terms()

    assert [type(t) for t in terms] == [
        ReferenceStateInitialization,
        ObjectReferenceStateInitialization,
        CustomReferenceStateInitialization,
    ]


@pytest.mark.parametrize(
    "extra_events, match",
    [
        (None, "requires at least one ReferenceStateInitialization"),
        (
            {"ghost": NoTeleportReferenceStateInitialization.configure(mode=en.EventMode.RESET)},
            r"do not implement _stage_teleport\(\): \['NoTeleportReferenceStateInitialization'\]",
        ),
    ],
    ids=["no-rsi-term", "rsi-without-teleport"],
)
def test_resync_without_a_way_to_teleport_fails_at_env_construction(library, extra_events, match):
    events = {} if extra_events is None else {"event_options": _events(**extra_events)}
    with pytest.raises(RuntimeError, match=match):
        build_mimic_env(library, motion_options=_motion_options(library), **events)
