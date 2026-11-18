"""``ResetFromStateBuffer``: a reset restores each env to a stored row, and a bad buffer raises at build.

One scene holds every entity kind the restore distinguishes: a free body re-origined onto a centre of mass
0.3 m from its authored origin (``tool``), a free base carrying a hinge (``swing``), a fixed base carrying a
ball joint and a hinge (``floppy``), a box (``cube``), and a fixed box posed per env (``post``, built with
``batch_fixed_verts``), and a group of fixed boxes, a variant per env block (``shelf``). Two more hold free
bodies no row can store: two in one entity (``twins``), and one
inside a per-env fixed base that also carries a hinge (``rack``). The oracles are Genesis' own getters: the
authored frame through ``relative=True``, and the raw generalized coordinates and DOF velocities.
"""

from __future__ import annotations

import math

import genesis as gs
import genesis.utils.geom as gu
import pytest
import torch

from eden.managers.terms.events.state_buffer import ResetFromStateBuffer
from eden.utils.buffers.reset_states import EntityStates, ResetStateBuffer, capture_reset_states
from tests.support.envs import build_env

pytestmark = pytest.mark.requires_genesis

NUM_ENVS = 4
ENTITIES = ("tool", "swing", "floppy", "cube")

TOOL_MJCF = """<mujoco model="tool"><worldbody>
  <body name="root"><freejoint/><geom type="box" size="0.02 0.02 0.02" pos="0.3 0 0" mass="1.0"/>
    <body name="handle" pos="0 0.1 0"><geom type="box" size="0.01 0.01 0.01" mass="0.01"/></body></body>
</worldbody></mujoco>"""
SWING_MJCF = """<mujoco model="swing"><compiler angle="radian"/><worldbody>
  <body name="hub"><freejoint/><geom type="box" size="0.05 0.05 0.05" mass="1.0"/>
    <body name="pendulum" pos="0 0 -0.1"><joint name="swing_joint" type="hinge" axis="0 1 0" range="-1 1"/>
      <geom type="capsule" size="0.02" fromto="0 0 0 0 0 -0.3" mass="0.2"/></body></body>
</worldbody></mujoco>"""
FLOPPY_MJCF = """<mujoco model="floppy"><compiler angle="radian" autolimits="true"/><worldbody>
  <body name="anchor" pos="-0.6 -0.6 0.4"><geom type="box" size="0.04 0.04 0.02" density="500"/>
    <body name="socket" pos="0 0 0.05"><joint name="ball_joint" type="ball"/>
      <geom type="capsule" size="0.02" fromto="0 0 0 0.08 0 0" density="500"/>
      <body name="spindle" pos="0.08 0 0"><joint name="free_spin_joint" axis="0 0 1"/>
        <geom type="cylinder" size="0.02 0.01" density="500"/></body></body></body>
</worldbody></mujoco>"""
TWINS_MJCF = """<mujoco model="twins"><worldbody>
  <body name="left"><freejoint name="left_free"/><geom type="box" size="0.02 0.02 0.02" mass="0.1"/></body>
  <body name="right" pos="0.1 0 0"><freejoint name="right_free"/><geom type="box" size="0.02 0.02 0.02" mass="0.1"/></body>
</worldbody></mujoco>"""
RACK_MJCF = """<mujoco model="rack"><compiler angle="radian"/><worldbody>
  <body name="frame"><geom type="box" size="0.05 0.05 0.02" mass="1.0"/>
    <body name="arm" pos="0 0 0.05"><joint name="rack_hinge" axis="0 0 1"/>
      <geom type="box" size="0.08 0.01 0.01" mass="0.1"/></body></body>
  <body name="loose" pos="0 0 0.3"><freejoint name="loose_free"/><geom type="box" size="0.02 0.02 0.02" mass="0.1"/></body>
</worldbody></mujoco>"""


def _yaw_pitch(yaw: float, pitch: float) -> list[float]:
    return [float(v) for v in gu.xyz_to_quat(torch.tensor([0.0, pitch, yaw]), rpy=True)]


def _authored_buffer() -> ResetStateBuffer:
    """Four hand-written states, every one moving: the "generated" source, with known values to check against."""
    rows = torch.arange(NUM_ENVS, dtype=torch.float32)[:, None]
    tool_quat = torch.tensor([_yaw_pitch(0.4 * i, 0.3 - 0.2 * i) for i in range(NUM_ENVS)])
    return ResetStateBuffer(
        {
            "tool": EntityStates(
                root_pos=torch.tensor([-0.8, -0.8, 1.0]) + 0.1 * rows,
                root_quat=tool_quat,
                root_lin_vel=torch.tensor([0.2, -0.1, 0.3]) * (rows + 1),
                root_ang_vel=torch.tensor([0.5, 1.0, -2.0]) - rows,
            ),
            "swing": EntityStates(
                root_pos=torch.tensor([0.8, -0.8, 1.2]) + 0.05 * rows,
                root_quat=torch.tensor([_yaw_pitch(-0.3 * i, 0.0) for i in range(NUM_ENVS)]),
                root_lin_vel=torch.tensor([0.0, 0.3, -0.2]) * rows,
                root_ang_vel=torch.tensor([0.0, 0.0, 1.5]).repeat(NUM_ENVS, 1),
                joints_name=("swing_joint",),
                joints_n_dofs=(1,),
                dofs_pos=0.2 * rows - 0.3,
                dofs_vel=0.5 - 0.25 * rows,
            ),
            "floppy": EntityStates(
                joints_name=("ball_joint", "free_spin_joint"),
                joints_n_dofs=(3, 1),
                dofs_pos=torch.cat([0.1 * rows, -0.2 * rows, 0.3 * rows, 1.0 + rows], dim=1),
                dofs_vel=torch.cat([torch.full((NUM_ENVS, 3), 0.4), 2.0 - rows], dim=1),
            ),
            "cube": EntityStates(
                root_pos=torch.tensor([0.8, 0.8, 1.0]) - 0.05 * rows,
                root_quat=torch.tensor([_yaw_pitch(0.25 * i, 0.0) for i in range(NUM_ENVS)]),
                root_lin_vel=torch.tensor([0.1, 0.0, 0.0]) * rows,
                root_ang_vel=torch.tensor([0.0, 0.2, 0.0]) * rows,
            ),
            "post": EntityStates(
                root_pos=torch.tensor([0.0, 0.8, 0.2]) + torch.cat([0.1 * rows, -0.05 * rows, 0.02 * rows], dim=1),
                root_quat=torch.tensor([_yaw_pitch(0.5 * i, 0.0) for i in range(NUM_ENVS)]),
            ),
        },
        columns={"goal": torch.tensor([0.1, 0.2, 0.3]) + rows},
    )


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    from eden.options import BoxOptions, EntityOptions, EnvOptions, SceneOptions
    from eden.options.entities import GroupedEntityOptions
    from eden.utils.configs import EdenConfig, EventManagerOptions

    assets = tmp_path_factory.mktemp("state_buffer")
    for name, text in (
        ("tool", TOOL_MJCF),
        ("swing", SWING_MJCF),
        ("floppy", FLOPPY_MJCF),
        ("twins", TWINS_MJCF),
        ("rack", RACK_MJCF),
    ):
        (assets / f"{name}.xml").write_text(text)
    buffer_path = assets / "authored.pt"
    _authored_buffer().save(buffer_path)
    return build_env(
        EdenConfig(
            env_options=EnvOptions(num_envs=NUM_ENVS),
            scene_options=SceneOptions(
                tool=EntityOptions(file=str(assets / "tool.xml"), default_root_pos=(-0.8, -0.8, 0.5)),
                swing=EntityOptions(file=str(assets / "swing.xml"), default_root_pos=(0.8, -0.8, 1.0)),
                floppy=EntityOptions(file=str(assets / "floppy.xml"), is_fixed_base=True),
                cube=BoxOptions(size=(0.1, 0.1, 0.1), default_root_pos=(0.8, 0.8, 0.5)),
                post=BoxOptions(
                    size=(0.05, 0.05, 0.2), fixed=True, batch_fixed_verts=True, default_root_pos=(0, 0.8, 0.2)
                ),
                twins=EntityOptions(file=str(assets / "twins.xml"), default_root_pos=(0.0, -0.8, 0.5)),
                rack=EntityOptions(
                    file=str(assets / "rack.xml"),
                    is_fixed_base=True,
                    batch_fixed_verts=True,
                    default_root_pos=(0.6, 0, 0.1),
                ),
                shelf=GroupedEntityOptions(
                    is_fixed_base=True,
                    grouped_entities=[
                        BoxOptions(size=size, batch_fixed_verts=True, default_root_pos=(-0.6, 0.6, 0.05))
                        for size in ((0.05, 0.05, 0.05), (0.1, 0.05, 0.05))
                    ],
                ),
            ),
            event_options=EventManagerOptions(restore=ResetFromStateBuffer.configure(path=str(buffer_path))),
        )
    )


@pytest.fixture
def restore(env):
    """The built event term; the buffer it owns is put back after a test rewrites its rows."""
    term = env.event_manager.get_term("restore")
    yield term
    term.buffer.write_rows(torch.arange(NUM_ENVS), _authored_buffer().to(term.buffer.device, dtype=torch.float32))


def _raw_state(env) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    """Every entity's full generalized coordinates and DOF velocities: the restore's ground truth."""
    state = {}
    for name in ENTITIES:
        entity = env.entities[name]
        state[name] = (
            entity.get_qpos(qs_idx_local=list(range(entity.n_qs))).clone(),
            entity.get_dofs_vel(dofs_idx_local=list(range(entity.n_dofs))).clone(),
        )
    return state


def _assert_quat_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    """Equal up to the sign a quaternion carries for free."""
    sign = torch.where((actual * expected).sum(-1, keepdim=True) < 0, -1.0, 1.0)
    torch.testing.assert_close(actual * sign, expected, atol=1e-5, rtol=0)


def _joint(entity, name: str):
    return next(joint for joint in entity.joints if joint.name == name)


def test_a_reset_lands_each_env_on_its_row_in_the_authored_frame(env, restore):
    """Root pose and velocity are the authored origin's, read back through Genesis' relative=True getters."""
    env.reset()
    buffer, rows = _authored_buffer(), restore.current_config.cpu()
    tool, swing, floppy = env.entities["tool"], env.entities["swing"], env.entities["floppy"]
    expected = buffer.entities["tool"]
    torch.testing.assert_close(tool.get_pos(relative=True).cpu(), expected.root_pos[rows], atol=1e-5, rtol=0)
    _assert_quat_close(tool.get_quat(relative=True).cpu(), expected.root_quat[rows])
    # The authored origin sits 0.3 m off the centre of mass, so its velocity carries the w x d transport term.
    authored_vel = tool.get_links_vel(ls_idx_local=[0], relative=True)[:, 0].cpu()
    torch.testing.assert_close(authored_vel, expected.root_lin_vel[rows], atol=1e-4, rtol=0)
    torch.testing.assert_close(
        tool.get_links_ang(ls_idx_local=[0])[:, 0].cpu(), expected.root_ang_vel[rows], atol=1e-4, rtol=0
    )

    hinge = _joint(swing, "swing_joint").dofs_idx_local
    torch.testing.assert_close(swing.get_dofs_pos(dofs_idx_local=hinge).cpu(), buffer.entities["swing"].dofs_pos[rows])
    torch.testing.assert_close(swing.get_dofs_vel(dofs_idx_local=hinge).cpu(), buffer.entities["swing"].dofs_vel[rows])

    ball = _joint(floppy, "ball_joint")
    _assert_quat_close(
        floppy.get_qpos(qs_idx_local=list(ball.qs_idx_local)).cpu(),
        gu.xyz_to_quat(buffer.entities["floppy"].dofs_pos[rows, :3]),
    )
    spin = _joint(floppy, "free_spin_joint").dofs_idx_local
    torch.testing.assert_close(
        floppy.get_dofs_pos(dofs_idx_local=spin).cpu(), buffer.entities["floppy"].dofs_pos[rows, 3:]
    )
    torch.testing.assert_close(restore.current_column("goal").cpu(), buffer.columns["goal"][rows])
    post = env.entities["post"]
    torch.testing.assert_close(
        post.get_pos(relative=True).cpu(), buffer.entities["post"].root_pos[rows], atol=1e-5, rtol=0
    )
    _assert_quat_close(post.get_quat(relative=True).cpu(), buffer.entities["post"].root_quat[rows])


def test_capture_save_load_and_reset_restores_the_raw_state(env, restore, tmp_path):
    """The harvest path: capture the live sim, write it into the buffer's slots, reset back onto it."""
    env.reset()
    for _ in range(5):  # let the bodies fall, swing and spin, so every state is in motion
        env.scene.step()
    before = _raw_state(env)
    post_before = env.entities["post"].get_pos(relative=True).cpu(), env.entities["post"].get_quat(relative=True).cpu()
    captured = capture_reset_states(env, (*ENTITIES, "post"), columns={"goal": restore.buffer.columns["goal"].clone()})
    assert captured.entities["post"].root_lin_vel is None, "a fixed base was captured with a root velocity"
    captured.save(tmp_path / "harvest.pt")
    restore.buffer.write_rows(torch.arange(NUM_ENVS), ResetStateBuffer.load(tmp_path / "harvest.pt"))

    env.reset()
    rows = restore.current_config.cpu()
    assert len(set(rows.tolist())) > 1, "uniform draws over 4 rows for 4 envs all landed on one row"
    after = _raw_state(env)
    for name in ENTITIES:
        qpos_before, vel_before = before[name][0].cpu()[rows], before[name][1].cpu()[rows]
        qpos_after, vel_after = after[name][0].cpu(), after[name][1].cpu()
        if not env.entities[name].has_immobile_base:
            _assert_quat_close(qpos_after[:, 3:7], qpos_before[:, 3:7])
            qpos_after, qpos_before = (
                torch.cat([qpos_after[:, :3], qpos_after[:, 7:]], 1),
                torch.cat([qpos_before[:, :3], qpos_before[:, 7:]], 1),
            )
        if name == "floppy":  # the ball joint's quaternion (qs 0:4) is equal up to sign
            _assert_quat_close(qpos_after[:, :4], qpos_before[:, :4])
            qpos_after, qpos_before = qpos_after[:, 4:], qpos_before[:, 4:]
        torch.testing.assert_close(qpos_after, qpos_before, atol=1e-5, rtol=0, msg=f"{name} qpos")
        torch.testing.assert_close(vel_after, vel_before, atol=1e-4, rtol=0, msg=f"{name} dofs velocity")
    post = env.entities["post"]
    torch.testing.assert_close(post.get_pos(relative=True).cpu(), post_before[0][rows], atol=1e-5, rtol=0)
    _assert_quat_close(post.get_quat(relative=True).cpu(), post_before[1][rows])


@pytest.fixture(params=[True, False], ids=["zero-copy", "genesis-kernel-path"])
def accessor_path(request):
    """Genesis' accessors on their zero-copy views, then on the kernel path they take without them."""
    if request.param:
        yield
    else:
        with gs.use_zerocopy.disabled():
            yield


@pytest.mark.usefixtures("accessor_path")
def test_a_mask_reset_writes_only_the_masked_envs(env, restore):
    """Off Genesis' zero-copy path (``GS_ENABLE_ZEROCOPY=0``; hibernation and a relative write onto a morph offset
    land there too) its base-pose setters refuse every env's row under a mask, so the fixed root is written as ids.
    """
    env.reset()
    env.scene.step()
    before = _raw_state(env)
    env.entities["post"].set_pos(torch.tensor([[0.0, 1.5, 0.2]] * NUM_ENVS, device=env.device), relative=True)
    post_before = env.entities["post"].get_pos(relative=True).cpu()
    mask = torch.tensor([True, False, False, True], device=env.device)
    restore.compute(mask)
    after, buffer = _raw_state(env), _authored_buffer()
    for name in ENTITIES:
        torch.testing.assert_close(after[name][0][~mask], before[name][0][~mask], msg=f"{name} moved outside the mask")
    rows = restore.current_config.cpu()[mask.cpu()]
    torch.testing.assert_close(
        env.entities["cube"].get_pos(relative=True).cpu()[mask.cpu()],
        buffer.entities["cube"].root_pos[rows],
        atol=1e-5,
        rtol=0,
    )
    post = env.entities["post"].get_pos(relative=True).cpu()
    torch.testing.assert_close(post[mask.cpu()], buffer.entities["post"].root_pos[rows], atol=1e-5, rtol=0)
    torch.testing.assert_close(post[~mask.cpu()], post_before[~mask.cpu()], msg="post moved outside the mask")


def _swing(**overrides) -> EntityStates:
    fields = dict(
        root_pos=torch.zeros(1, 3),
        root_quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
        joints_name=("swing_joint",),
        joints_n_dofs=(1,),
        dofs_pos=torch.zeros(1, 1),
    )
    return EntityStates(**{**fields, **overrides})


@pytest.mark.parametrize(
    "entities, error, match",
    [
        (
            {"ghost": _swing()},
            ValueError,
            r"(?s)restores entity .ghost., which is not in the scene.*scene entities: tool",
        ),
        (
            {"floppy": EntityStates(joints_name=("free_spin_joint",), joints_n_dofs=(1,), dofs_pos=torch.zeros(1, 1))},
            ValueError,
            r"on the entity but not stored: \[.ball_joint.\]",
        ),
        ({"swing": _swing(joints_name=("elbow",))}, ValueError, r"stored but not on the entity: \['elbow'\]"),
        (
            {"swing": _swing(joints_n_dofs=(2,), dofs_pos=torch.zeros(1, 2))},
            ValueError,
            r"joint 'swing_joint' has 1 DOFs, but the buffer stores 2",
        ),
        (
            {"swing": _swing(root_pos=None, root_quat=None)},
            ValueError,
            r"no root pose for entity 'swing', which has a free base",
        ),
        (
            {
                "floppy": EntityStates(
                    root_pos=torch.zeros(1, 3),
                    root_quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
                    joints_name=("ball_joint", "free_spin_joint"),
                    joints_n_dofs=(3, 1),
                    dofs_pos=torch.zeros(1, 4),
                )
            },
            ValueError,
            r"root pose for entity 'floppy', but 'floppy' is fixed-base without batch_fixed_verts=True.*Or set "
            r"batch_fixed_verts=True on 'floppy'",
        ),
        (
            {
                "post": EntityStates(
                    root_pos=torch.zeros(1, 3),
                    root_quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
                    root_lin_vel=torch.zeros(1, 3),
                )
            },
            ValueError,
            r"root velocity for entity 'post', which is fixed-base",
        ),
        (
            {"twins": EntityStates(root_pos=torch.zeros(1, 3), root_quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]]))},
            ValueError,
            r"entity 'twins' has 2 free joints \['left_free', 'right_free'\].*split each free body",
        ),
        (
            {"rack": EntityStates(joints_name=("rack_hinge",), joints_n_dofs=(1,), dofs_pos=torch.zeros(1, 1))},
            ValueError,
            r"no root pose for entity 'rack', which is fixed-base with batch_fixed_verts=True",
        ),
        (
            {
                "rack": EntityStates(
                    root_pos=torch.zeros(1, 3),
                    root_quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
                    joints_name=("rack_hinge",),
                    joints_n_dofs=(1,),
                    dofs_pos=torch.zeros(1, 1),
                )
            },
            ValueError,
            r"entity 'rack' is fixed-base, yet holds free joint\(s\) \['loose_free'\].*its own scene entity",
        ),
    ],
    ids=[
        "unknown-entity",
        "missing-joint",
        "unknown-joint",
        "dof-count",
        "free-base-without-root",
        "root-on-fixed-base",
        "velocity-on-fixed-base",
        "two-free-bodies",
        "per-env-fixed-base-without-root",
        "free-body-in-a-fixed-base",
    ],
)
def test_a_buffer_that_does_not_fit_the_scene_raises_at_build(env, tmp_path, entities, error, match):
    path = tmp_path / "bad.pt"
    ResetStateBuffer(entities).save(path)
    term = ResetFromStateBuffer(env=env, options=ResetFromStateBuffer.configure(path=str(path)))
    with pytest.raises(error, match=match):
        term.build()


@pytest.mark.parametrize(
    "name, match",
    [("twins", r"entity 'twins' has 2 free joints"), ("rack", r"entity 'rack' is fixed-base, yet holds free joint")],
)
def test_capture_refuses_a_free_body_no_row_can_hold(env, name, match):
    """A row stores one root per entity: any other free body would keep the previous episode's pose on a restore."""
    with pytest.raises(ValueError, match=f"capture_reset_states: {match}"):
        capture_reset_states(env, [name])


def test_capture_reads_only_the_envs_a_mask_selects(env):
    """A ``(num_envs,)`` bool mask, like ``reset_buf``, picks envs; read as indices it would be envs 0 and 1."""
    cube = env.entities["cube"]
    pos = torch.tensor([0.8, 0.8, 0.5], device=env.device) + 0.1 * torch.arange(NUM_ENVS, device=env.device)[:, None]
    cube.set_pos(pos, relative=True)
    mask = torch.tensor([False, True, False, True], device=env.device)
    captured = capture_reset_states(env, ["cube"], envs_idx=mask)
    torch.testing.assert_close(captured.entities["cube"].root_pos.cpu(), pos[mask].cpu(), atol=1e-5, rtol=0)


def test_a_fixed_base_with_a_variant_per_env_block_round_trips_its_root(env, tmp_path):
    """A group's ``morph`` is its variants' list; the per-env root is stored and restored all the same."""
    shelf = env.entities["shelf"]
    pos = torch.tensor([-0.6, 0.6, 0.05]) + torch.tensor([0.0, 0.1, 0.0]) * torch.arange(NUM_ENVS)[:, None]
    shelf.set_pos(pos.to(env.device), relative=True)
    captured = capture_reset_states(env, ["shelf"])
    torch.testing.assert_close(captured.entities["shelf"].root_pos.cpu(), pos, atol=1e-5, rtol=0)
    captured.save(tmp_path / "shelf.pt")
    shelf.set_pos(torch.tensor([[-0.6, 0.6, 0.05]] * NUM_ENVS, device=env.device), relative=True)
    term = ResetFromStateBuffer(env=env, options=ResetFromStateBuffer.configure(path=str(tmp_path / "shelf.pt")))
    term.build()
    term.compute(None)
    rows = term.current_config.cpu()
    torch.testing.assert_close(shelf.get_pos(relative=True).cpu(), pos[rows], atol=1e-5, rtol=0)


def test_capture_copies_the_task_columns(env):
    """A live tensor passed as a column (a command's buffer) must not carry the captured rows along as it changes."""
    goal = torch.zeros(NUM_ENVS, 3, device=env.device)
    captured = capture_reset_states(env, ["cube"], columns={"goal": goal})
    goal += 1.0
    torch.testing.assert_close(captured.columns["goal"].cpu(), torch.zeros(NUM_ENVS, 3))


def test_no_path_raises_naming_the_fix(env):
    with pytest.raises(ValueError, match=r"needs path=.*ResetFromStateBuffer.configure\(path="):
        ResetFromStateBuffer(env=env, options=ResetFromStateBuffer.configure()).build()


def test_a_sampler_name_needs_a_curriculum_manager(env, tmp_path):
    path = tmp_path / "ok.pt"
    ResetStateBuffer({"swing": _swing()}).save(path)
    term = ResetFromStateBuffer(env=env, options=ResetFromStateBuffer.configure(path=str(path), sampler="sgs"))
    with pytest.raises(ValueError, match="names sampler='sgs', but this env has no curriculum manager"):
        term.build()


def test_current_column_names_the_columns_on_a_miss(restore):
    with pytest.raises(KeyError, match="no task column 'target'. Columns: goal"):
        restore.current_column("target")


def test_offsets_hold_for_a_body_turned_far_from_its_build_pose(env, restore):
    """The authored->solver offset is read once at build; it must not depend on the pose read then."""
    tool = env.entities["tool"]
    upside_down = torch.tensor([[0.0, 1.0, 0.0, 0.0]] * NUM_ENVS)
    restore.buffer.write_rows(
        torch.arange(NUM_ENVS),
        ResetStateBuffer(
            {
                **{n: s for n, s in _authored_buffer().entities.items() if n != "tool"},
                "tool": EntityStates(
                    root_pos=torch.tensor([[-0.8, -0.8, 1.0]] * NUM_ENVS),
                    root_quat=upside_down,
                    root_lin_vel=torch.zeros(NUM_ENVS, 3),
                    root_ang_vel=torch.zeros(NUM_ENVS, 3),
                ),
            },
            columns={"goal": torch.zeros(NUM_ENVS, 3)},
        ),
    )
    env.reset()
    torch.testing.assert_close(
        tool.get_pos(relative=True).cpu(), torch.tensor([[-0.8, -0.8, 1.0]] * NUM_ENVS), atol=1e-5, rtol=0
    )
    _assert_quat_close(tool.get_quat(relative=True).cpu(), upside_down)
    # The centre of mass sits along the authored +x (1 kg at 0.3 m, the 0.01 kg handle at x = 0), which the flip
    # about x leaves at +x.
    com_x = 0.3 * 1.0 / 1.01
    assert math.isclose(float(tool.get_pos()[0, 0] - tool.get_pos(relative=True)[0, 0]), com_x, abs_tol=1e-4)


def test_a_state_stored_without_velocities_restores_at_rest(env, tmp_path):
    """A generated state at rest leaves its velocities out; they read as zero, not as the last episode's."""
    path = tmp_path / "at_rest.pt"
    at_rest = EntityStates(root_pos=torch.tensor([[0.8, 0.8, 1.0]]), root_quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    ResetStateBuffer({"cube": at_rest}).save(path)
    term = ResetFromStateBuffer(env=env, options=ResetFromStateBuffer.configure(path=str(path)))
    term.build()
    env.reset()  # the authored rows set the cube moving in every env but env 0
    term.compute(None)
    cube = env.entities["cube"]
    torch.testing.assert_close(cube.get_dofs_vel(dofs_idx_local=list(range(6))).cpu(), torch.zeros(NUM_ENVS, 6))
    torch.testing.assert_close(cube.get_pos(relative=True).cpu(), torch.tensor([[0.8, 0.8, 1.0]] * NUM_ENVS))


def test_a_state_at_rest_restores_whatever_torch_default_dtype_a_script_set(env, tmp_path):
    """The velocities a row leaves out are filled in the solver's float, not torch's default dtype.

    ``en.init`` makes the two agree, but a script can set the default apart. The restore must not depend on that
    global: ``torch.cross`` does not promote a float32 operand to meet a float64 one.
    """
    path = tmp_path / "tool_at_rest.pt"
    pose = torch.tensor([[-0.8, -0.8, 1.0]]), torch.tensor([_yaw_pitch(0.5, 0.2)])
    ResetStateBuffer({"tool": EntityStates(root_pos=pose[0], root_quat=pose[1])}).save(path)
    previous = torch.get_default_dtype()
    torch.set_default_dtype(torch.float32 if previous == torch.float64 else torch.float64)
    try:
        term = ResetFromStateBuffer(env=env, options=ResetFromStateBuffer.configure(path=str(path)))
        term.build()
        term.compute(None)
    finally:
        torch.set_default_dtype(previous)
    tool = env.entities["tool"]
    torch.testing.assert_close(tool.get_pos(relative=True).cpu(), pose[0].expand(NUM_ENVS, 3), atol=1e-5, rtol=0)
    torch.testing.assert_close(tool.get_dofs_vel(dofs_idx_local=list(range(6))).cpu(), torch.zeros(NUM_ENVS, 6))


def _families(tmp_path) -> str:
    path = tmp_path / "families.pt"
    ResetStateBuffer.from_families(
        {"near": ResetStateBuffer({"swing": _swing()}), "far": ResetStateBuffer({"swing": _swing()})}
    ).save(path)
    return str(path)


def test_category_weights_pick_the_weighted_sampler(env, tmp_path):
    term = ResetFromStateBuffer(
        env=env, options=ResetFromStateBuffer.configure(path=_families(tmp_path), category_weights={"far": 1.0})
    )
    term.build()
    term.compute(torch.arange(NUM_ENVS, device=env.device))
    assert (term.current_config == 1).all(), "category 'near' has no weight, so every env starts from row 1 ('far')"


def test_category_weights_follow_rows_relabelled_by_a_harvest(env, tmp_path):
    """Rows written into the term's buffer carry their own family; the weighted draw must use it, not the old one."""
    term = ResetFromStateBuffer(
        env=env, options=ResetFromStateBuffer.configure(path=_families(tmp_path), category_weights={"far": 1.0})
    )
    term.build()
    swapped = ResetStateBuffer(
        {"swing": _swing(**{name: value.repeat(2, 1) for name, value in _swing().tensors().items()})},
        category=torch.tensor([1, 0]),
        category_names=("near", "far"),
    )
    term.buffer.write_rows(torch.tensor([0, 1]), swapped)
    term.compute(torch.arange(NUM_ENVS, device=env.device))
    assert (term.current_config == 0).all(), "row 0 is now the only 'far' row"


@pytest.mark.parametrize(
    "labelled, overrides, match",
    [
        (
            True,
            {"category_weights": {"close": 1.0}},
            r"category_weights names \['close'\].*Its categories: \['near', 'far'\]",
        ),
        (True, {"category_weights": {"far": 1.0}, "sampler": "sgs"}, r"sets both sampler='sgs' and category_weights"),
        (False, {"category_weights": {"far": 1.0}}, r"the buffer labels no categories"),
    ],
    ids=["unknown-category", "with-a-sampler", "unlabelled-buffer"],
)
def test_category_weights_that_do_not_fit_raise_at_build(env, tmp_path, labelled, overrides, match):
    if labelled:
        path = _families(tmp_path)
    else:
        path = str(tmp_path / "plain.pt")
        ResetStateBuffer({"swing": _swing()}).save(path)
    term = ResetFromStateBuffer(env=env, options=ResetFromStateBuffer.configure(path=path, **overrides))
    with pytest.raises(ValueError, match=match):
        term.build()
