"""``LinVelKalmanEstimator``: the contact-aided base-velocity filter, on analytic legs and on a real G1.

Most tests drive the filter through ``_KinematicLegs``, an analytic kinematics model (identity foot
Jacobians, feet pinned in the world) that makes each scenario exact; a real entity cannot hold a
foot still while its base moves. The last two pin the same contact model against a real Genesis G1
and MuJoCo's ground-truth contact force.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from eden.extensions.deployment.base import resolve_dofs_values
from eden.extensions.deployment.utils.lin_vel_estimator import LinVelKalmanEstimator
from eden.options.robots.unitree_g1 import G1Dofs29

DT = 0.02
FEET = ("left_foot", "right_foot")
FOOT_WORLD = {
    "left_foot": np.array([0.0, 0.1, 0.0]),
    "right_foot": np.array([0.0, -0.1, 0.0]),
}
BASE_MASS = 40.0 / (0.12 * 9.81)
# Motor torque holding against an 80 N upward GRF (tau = -J^T f with J = I):
# the estimator maps it back to fz = +80, well above the 40 N latch threshold.
STANCE_TAU = np.array([0.0, 0.0, -80.0])


class _KinematicLegs:
    """Analytic kinematics: two massless legs whose foot Jacobians are identity.

    The base moves along a scripted trajectory while the feet stay pinned at
    fixed world points; joint velocities are set so leg odometry is exactly
    consistent (``J @ dq = -v_base`` for stance feet).
    """

    def __init__(self, n_leg: int = 3) -> None:
        self.n_leg = n_leg
        self.dofs_name = [f"{side}{j}" for side in "lr" for j in range(n_leg)]
        self.foot_world = {name: pos.copy() for name, pos in FOOT_WORLD.items()}
        base = SimpleNamespace(idx=0, parent_idx=-1, name="pelvis", joints=[])
        feet_links = [
            SimpleNamespace(
                idx=1 + i,
                parent_idx=0,
                name=name,
                joints=[SimpleNamespace(name=self.dofs_name[n_leg * i + j], n_dofs=1) for j in range(n_leg)],
            )
            for i, name in enumerate(FEET)
        ]
        self._entity = SimpleNamespace(
            links=[base, *feet_links],
            get_link=lambda name: {lk.name: lk for lk in [base, *feet_links]}[name],
        )
        self.base_pos = np.zeros(3)

    @property
    def base_link_name(self):
        # Mirrors RigidEntity.base_link_name, which is the root of the kinematic tree.
        return self._entity.links[0].name

    def get_link(self, name):
        return self._entity.get_link(name)

    def find_named_links_idx_local(self, names, preserve_order=True):
        order = ["pelvis", *FEET]
        return names, [order.index(n) for n in names]

    def get_links_pos(self, ls_idx_local=None, ref="link_origin"):
        # Massless legs (``get_links_inertial_mass``), so where their centres of mass sit never
        # matters: ``ref="link_com"`` reads the origins too.
        rows = [self.base_pos, self.foot_world["left_foot"], self.foot_world["right_foot"]]
        pos = np.stack([rows[i] for i in ls_idx_local])
        return torch.as_tensor(pos, dtype=torch.float64).unsqueeze(0)

    def get_links_inertial_mass(self, ls_idx_local=None):
        # All the weight in the base, so the default latch threshold (12% of m g) is 40 N.
        masses = torch.tensor([[BASE_MASS, 0.0, 0.0]], dtype=torch.float64)
        return masses if ls_idx_local is None else masses[:, list(ls_idx_local)]

    def get_jacobian(self, link):
        J = torch.zeros(1, 6, 6 + 2 * self.n_leg, dtype=torch.float64)
        first = 0 if link.name == "left_foot" else self.n_leg
        for j in range(min(self.n_leg, 6)):
            J[0, j, 6 + first + j] = 1.0
        return J


def _make(entity=None, **kw):
    return LinVelKalmanEstimator(entity or _KinematicLegs(), feet_links=FEET, dt=DT, foot_sole_offset=0.0, **kw)


def _step(est, entity, v_base, *, contact=(True, True), quat=(1.0, 0.0, 0.0, 0.0), yaw_tag=0.0):
    """One estimator update with kinematics consistent with base velocity ``v_base``."""
    dq = np.concatenate([-v_base if contact[0] else np.zeros(3), -v_base if contact[1] else np.zeros(3)])
    tau = np.concatenate([STANCE_TAU if contact[0] else np.zeros(3), STANCE_TAU if contact[1] else np.zeros(3)])
    vel = est.update(
        np.array(quat),
        np.zeros(3),  # gyro
        np.array([0.0, 0.0, 9.81]),  # accelerometer at constant velocity: specific force only
        dq,
        tau,
        yaw_tag=yaw_tag,
    )
    entity.base_pos = entity.base_pos + v_base * DT
    return vel


def test_body_to_world_rotation_is_not_transposed():
    """The estimator must rotate body-frame IMU input *into* world, not out of it.

    Every other test runs at identity orientation, where ``R`` and ``R.T`` agree, so a transposed
    rotation (or an xyzw quaternion) would pass them all. The filter corrects the foot lever arm
    with ``w x r`` in world frame, and a transposed rotation flips that correction's sign.
    """
    yaw90 = np.array([np.cos(np.pi / 4), 0.0, 0.0, np.sin(np.pi / 4)])

    # Single-foot contact: with both feet down the stub's feet sit at (0, +-0.1, 0)
    # about the base, so the two `w x r` terms are equal and opposite and cancel
    # exactly -- the estimate is then insensitive to the gyro and proves nothing.
    # A body +y gyro is the axis that yaw actually moves: R maps it to world -x,
    # R.T to world +x, giving opposite `w x r` on the stance foot.
    def run(gyro):
        est = _make()
        out = None
        for _ in range(30):
            out = est.update(
                yaw90,
                gyro,
                np.array([0.0, 0.0, 9.81]),
                np.concatenate([np.zeros(3), np.zeros(3)]),
                np.concatenate([STANCE_TAU, np.zeros(3)]),  # left foot only
                yaw_tag=0.0,
            )
        return out

    spun, still = run(np.array([0.0, 0.9, 0.0])), run(np.zeros(3))
    # The estimator must actually consume the world-frame gyro...
    assert not np.allclose(spun, still, atol=1e-6), "estimate is insensitive to the gyro"
    # ...with the body->world sense, not its inverse. Under this pose the whole
    # estimate collapses onto +z, and swapping R for R.T flips its sign exactly
    # (+0.0678 -> -0.0678), so pinning the sign is what rejects a transpose.
    assert spun[2] > 1e-3, f"body->world rotation looks transposed (vz={spun[2]:+.4f}, expected > 0)"


def test_stationary_estimate_is_zero():
    entity = _KinematicLegs()
    est = _make(entity)
    for _ in range(50):
        vel = _step(est, entity, np.zeros(3))
    assert np.allclose(vel, 0.0, atol=1e-6)


def test_constant_velocity_tracked_through_leg_odometry():
    entity = _KinematicLegs()
    est = _make(entity)
    v = np.array([0.5, -0.2, 0.0])
    for _ in range(100):
        vel = _step(est, entity, v)
    assert np.allclose(vel, v, atol=0.02), vel


def test_swing_feet_barely_pull_the_estimate():
    entity = _KinematicLegs()
    est = _make(entity)
    for _ in range(50):
        _step(est, entity, np.zeros(3))
    # Both feet lose contact and report bogus zero leg motion while the base
    # "moves": the gated filter must stay near the IMU-predicted velocity (0)
    # instead of trusting the leg odometry.
    # The support gate is off: the stub's accelerometer keeps reading 1 g, which with both feet
    # up is the case the gate exists to catch, not swing.
    est._support_accel = 0.0
    v = np.array([0.5, 0.0, 0.0])
    for _ in range(5):
        vel = _step(est, entity, v, contact=(False, False))
    assert np.linalg.norm(vel) < 0.1, vel
    assert not est.contacts.any()


def test_contact_hysteresis():
    entity = _KinematicLegs()
    est = _make(entity, contact_force_threshold=40.0, contact_release_threshold=20.0)
    J = [np.eye(3), np.eye(3)]

    def tau(fz: float) -> np.ndarray:
        # Uniform per-leg torques: with an identity Jacobian the estimated
        # force is fz on every axis regardless of the chain's DOF ordering.
        # Negative torque <-> positive (upward) estimated GRF.
        return np.full(6, -fz)

    est._detect_contacts(J, tau(30.0))
    assert not est.contacts.any()  # below the latch threshold
    est._detect_contacts(J, tau(50.0))
    assert est.contacts.all()
    est._detect_contacts(J, tau(30.0))
    assert est.contacts.all()  # latched: above the release threshold
    est._detect_contacts(J, tau(10.0))
    assert not est.contacts.any()


def test_yaw_tag_change_reinitializes():
    entity = _KinematicLegs()
    est = _make(entity)
    v = np.array([0.5, 0.0, 0.0])
    for _ in range(100):
        _step(est, entity, v)
    assert np.allclose(est.vel, v, atol=0.02)
    vel = _step(est, entity, v, yaw_tag=1.0)  # new frame -> filter restarts from scratch
    assert np.linalg.norm(vel) < np.linalg.norm(v)


def test_requires_feet_links():
    with pytest.raises(ValueError, match="at least one foot"):
        LinVelKalmanEstimator(_KinematicLegs(), feet_links=(), dt=DT)


# ---------------------------------------------------------------------------
# Contact detection that survives a straight-knee stance
# ---------------------------------------------------------------------------


def _update(est, *, acc=(0.0, 0.0, 9.81), tau=None, dq=None):
    """One update of a stationary base, with the given accelerometer reading, torques and joint velocities."""
    n = len(est._entity.dofs_name)
    return est.update(
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.zeros(3),
        np.asarray(acc),
        np.zeros(n) if dq is None else dq,
        np.zeros(n) if tau is None else tau,
    )


def _load(left: float = 0.0, right: float = 0.0) -> np.ndarray:
    """Leg torques under which the stub's feet read ``left`` / ``right`` newtons of vertical load."""
    return np.concatenate([np.array([0.0, 0.0, -left]), np.array([0.0, 0.0, -right])])


def test_the_latch_threshold_defaults_to_a_fraction_of_the_robots_weight():
    est = _make()
    assert est._contact_on == pytest.approx(0.12 * BASE_MASS * 9.81)  # 40 N for this stub
    assert est._contact_off == pytest.approx(0.5 * est._contact_on)
    explicit = _make(contact_force_threshold=70.0)
    assert (explicit._contact_on, explicit._contact_off) == (70.0, 35.0)


def test_a_six_dof_leg_solves_the_foot_wrench_so_an_ankle_moment_is_not_read_as_load():
    """The G1 failure: a loaded foot read as airborne once the ankles carried the moment.

    A flat foot carries a moment as well as a force. Solved as a 3-D point force, the moment
    the ankle motors hold leaks into the force through the Jacobian's coupling -- on the G1 it
    read a foot carrying ~150 N as -190 to -380 N. Here the leg's linear rows couple joint 4
    (an ankle pitch) into vertical motion; a 100 N load with a 400 Nm ankle moment then reads
    100 N from the 6-D solve and -60 N from the 3-D one.
    """
    est = _make(_KinematicLegs(n_leg=6))
    assert est._wrench_rows == [6, 6]
    J = np.zeros((6, 6))
    J[:3, :3] = np.eye(3)
    J[3:, 3:] = np.eye(3)
    J[2, 4] = -0.5  # the ankle pitch also moves the foot vertically
    wrench = np.array([0.0, 0.0, 100.0, 0.0, 400.0, 0.0])
    tau = -J.T @ wrench  # quasi-static: tau_motor = -J^T w, in the leg's own DOF order
    dofs_torque = np.zeros(12)
    for leg_dofs in est._leg_dofs:
        dofs_torque[leg_dofs] = tau
    est._detect_contacts([J, J], dofs_torque)
    np.testing.assert_allclose(est.forces[:, 2], 100.0, atol=1e-6)
    assert est.contacts.all()

    point_force = -np.linalg.solve(J[:3] @ J[:3].T, J[:3] @ tau)
    assert point_force[2] < 0.0  # what the 3-D solve reads for the same loaded foot


def test_a_short_leg_keeps_the_point_force_solve():
    """A quadruped's 3-DOF leg ends in a point foot, which holds no moment."""
    assert _make(_KinematicLegs(n_leg=3))._wrench_rows == [3, 3]


def test_support_gate_latches_the_most_loaded_foot_while_the_imu_reads_support():
    est = _make(support_gate_ticks=3)
    for tick in range(3):
        _update(est, tau=_load(left=5.0, right=30.0))  # both below the 40 N latch
        assert est.contacts.tolist() == ([False, True] if tick == 2 else [False, False]), tick
    assert est.forces[1, 2] == pytest.approx(30.0)  # the gate latches; the load stays honest

    _update(est, acc=(0.0, 0.0, 0.0), tau=_load(left=5.0, right=30.0))  # free fall
    assert not est.contacts.any()


def test_support_gate_waits_for_the_contact_loss_itself_to_persist():
    """A long stance must not pre-charge the gate: it counts ticks with *no foot latched*."""
    est = _make(support_gate_ticks=3)
    for _ in range(20):
        _update(est, tau=_load(left=80.0, right=80.0))  # the IMU reads support throughout
    _update(est)  # both feet lift: tick 1 with none latched
    _update(est, tau=_load(right=30.0))  # tick 2: a load the gate would accept, but too soon
    assert not est.contacts.any()
    _update(est, tau=_load(right=30.0))  # tick 3
    assert est.contacts.tolist() == [False, True]


def test_a_hanging_robot_is_not_anchored_on_a_dangling_foot(eden_caplog):
    """On a hoist the IMU reads 1 g with both feet in the air, reading no load.

    Latching the lowest foot there would feed the estimate the motion of a leg swinging free;
    the gate needs load evidence, so the robot falls through to the warned fallback instead.
    """
    entity = _KinematicLegs()
    entity.foot_world["right_foot"][2] = -0.05
    est = _make(entity, support_gate_ticks=1, max_no_contact_time=0.2)
    swing = np.concatenate([np.zeros(3), np.array([0.8, 0.0, 0.0])])  # the right leg is being swung
    with eden_caplog.at_level("WARNING"):
        for _ in range(100):  # 2 s hanging
            vel = _update(est, dq=swing)
    assert not est.contacts.any()
    assert np.linalg.norm(vel) < 0.1, vel
    assert any("no foot in contact" in r.getMessage() for r in eden_caplog.records)


def test_a_leg_swung_under_a_hoist_is_not_anchored_on_either(eden_caplog):
    """A policy driving the legs of a hanging robot: swing dynamics read as load, but the foot moves.

    The swing foot of a closed-loop dance reads a median 38 N, p90 100 N, from its dynamics alone --
    past the release threshold the gate asks for -- so load evidence cannot tell it from a stance
    foot. Its world speed can.
    """
    entity = _KinematicLegs()
    entity.foot_world["right_foot"][2] = -0.05
    est = _make(entity, support_gate_ticks=1, max_no_contact_time=0.2)
    swing = np.concatenate([np.zeros(3), np.array([0.8, 0.0, 0.0])])  # the right leg is being swung
    with eden_caplog.at_level("WARNING"):
        for _ in range(100):
            vel = _update(est, tau=_load(right=35.0), dq=swing)  # below the latch, above the release
    assert not est.contacts.any()
    assert np.linalg.norm(vel) < 0.1, vel
    assert any("no foot in contact" in r.getMessage() for r in eden_caplog.records)


def test_the_speed_check_ignores_a_velocity_from_before_a_reinitialization():
    """On a yaw re-capture (every START) the previous estimate is in the old frame: the base counts as still."""
    entity = _KinematicLegs()
    entity.foot_world["left_foot"][2] = 0.15  # the stance foot is on a higher step
    est = _make(entity)
    _update(est, tau=_load(left=150.0))
    est._x[3:6] = [2.0, 0.0, 0.0]  # what the filter held in the old frame
    est.update(
        np.array([1.0, 0.0, 0.0, 0.0]),
        np.zeros(3),
        np.array([0.0, 0.0, 9.81]),
        np.zeros(6),
        _load(left=150.0),
        yaw_tag=1.0,
    )
    assert est.contacts.tolist() == [True, False]


def test_support_gate_leaves_a_detected_foot_alone():
    entity = _KinematicLegs()
    entity.foot_world["right_foot"][2] = -0.03  # lower, but within the height gate of the left
    est = _make(entity, support_gate_ticks=1)
    _update(est, tau=_load(left=80.0))  # the left foot carries load
    assert est.contacts.tolist() == [True, False]


def test_a_loaded_foot_above_another_loaded_foot_is_not_latched():
    """Swing-leg dynamics read as load in the 6-D solve; the loaded foot beneath is the one standing."""
    entity = _KinematicLegs()
    entity.foot_world["left_foot"][2] = 0.10  # 10 cm above the right foot
    stance = _load(left=80.0, right=80.0)
    est = _make(entity, max_contact_height=0.05)
    _update(est, tau=stance)
    assert est.contacts.tolist() == [False, True]
    assert est.forces[0, 2] == pytest.approx(80.0)  # the load is still reported, just not trusted

    ungated = _make(entity, max_contact_height=None)
    _update(ungated, tau=stance)
    assert ungated.contacts.all()


def test_the_stance_foot_on_a_higher_step_is_latched():
    """Stairs: the upper foot carries the robot while the lower one swings, unloaded."""
    entity = _KinematicLegs()
    entity.foot_world["left_foot"][2] = 0.15  # on the next step up
    est = _make(entity, support_gate_ticks=1)
    _update(est, tau=_load(left=150.0))
    assert est.contacts.tolist() == [True, False]


def test_a_raised_foot_that_is_moving_is_not_latched():
    """With no loaded foot beneath it, a raised foot counts only while it is still in the world."""
    entity = _KinematicLegs()
    entity.foot_world["left_foot"][2] = 0.15
    est = _make(entity, support_gate_ticks=5)
    moving = np.concatenate([np.array([1.0, 0.0, 0.0]), np.zeros(3)])  # the left foot moves at 1 m/s
    _update(est, tau=_load(left=150.0), dq=moving)
    assert not est.contacts.any()


def test_support_gate_can_be_disabled():
    est = _make(support_accel_ratio=0.0, support_gate_ticks=1)
    for _ in range(5):
        _update(est)
    assert not est.contacts.any()


def test_no_contact_fallback_bounds_imu_drift_and_warns_once(eden_caplog):
    """With no foot down, a biased accelerometer must not run the velocity away."""
    kw = dict(support_accel_ratio=0.0, max_no_contact_time=0.2, no_contact_velocity_decay=0.2)
    biased = (1.0, 0.0, 9.81)  # a 1 m/s^2 forward bias
    est = _make(**kw)
    with eden_caplog.at_level("INFO"):
        for _ in range(150):  # 3 s airborne
            vel = _update(est, acc=biased)
    # Integrated open loop the bias alone reaches 3 m/s; the fallback holds it near zero.
    assert np.linalg.norm(vel) < 0.1, vel
    warnings = [r for r in eden_caplog.records if "no foot in contact" in r.getMessage()]
    assert len(warnings) == 1 and warnings[0].levelname == "WARNING"
    assert "0.20 s" in warnings[0].getMessage()

    with eden_caplog.at_level("INFO"):
        _update(est, tau=_load(left=80.0, right=80.0))
    assert any("contact regained" in r.getMessage() for r in eden_caplog.records)


def test_no_contact_shorter_than_the_limit_is_left_to_the_imu():
    """A jump is legitimately IMU-only: the fallback must not touch a short flight."""
    est = _make(support_accel_ratio=0.0, max_no_contact_time=0.5)
    for _ in range(10):
        _update(est, tau=_load(left=80.0, right=80.0))
    for _ in range(20):  # 0.4 s of flight, reading the 1 m/s^2 of a push-off
        vel = _update(est, acc=(1.0, 0.0, 9.81))
    assert not est._fallback_warned
    assert vel[0] > 0.2, vel  # still integrating the accelerometer


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"support_gate_ticks": 0}, "support_gate_ticks"),
        ({"max_no_contact_time": 0.0}, "max_no_contact_time"),
        ({"no_contact_velocity_decay": -1.0}, "no_contact_velocity_decay"),
        ({"max_contact_height": 0.0}, "max_contact_height"),
        ({"max_raised_foot_speed": 0.0}, "max_raised_foot_speed"),
    ],
)
def test_bad_fallback_options_raise_at_construction(kwargs, match):
    with pytest.raises(ValueError, match=match):
        _make(**kwargs)


# ---------------------------------------------------------------------------
# A straight-kneed G1 stance, against MuJoCo's contact force
# ---------------------------------------------------------------------------
#
# The ankle-pitch motors carry the foot's moment in that stance. The robot hangs from the
# simulated shoulder gantry so it can hold the pose open loop, with about two thirds of its
# weight still on its feet.

CONTROL_DT = 0.02
SETTLE_TICKS = 50  # default stance, then a 100-tick ramp to the straight-knee pose
RAMP_TICKS = 100
HOLD_TICKS = 150
STRAIGHT_KNEE = {"hip_pitch": -0.1, "knee": 0.2, "ankle_pitch": -0.1}  # knees 0.2 rad, as on the robot


def _foot_contact_force(sim, foot_bodies: list[int]) -> np.ndarray:
    """World-frame contact force the floor exerts on each foot, (n_feet, 3)."""
    import mujoco

    model, data = sim.model, sim.data
    out = np.zeros((len(foot_bodies), 3))
    f6 = np.zeros(6)
    for c in range(data.ncon):
        con = data.contact[c]
        b1, b2 = model.geom_bodyid[con.geom1], model.geom_bodyid[con.geom2]
        for i, body in enumerate(foot_bodies):
            if body in (b1, b2):
                mujoco.mj_contactForce(model, data, c, f6)
                force = con.frame.reshape(3, 3).T @ f6[:3]
                out[i] += force if b2 == body else -force
    return out


@pytest.fixture(scope="module")
def straight_knee_stance():
    """Run the MuJoCo stance once; return per-tick readings plus the true foot contact force."""
    pytest.importorskip("mujoco")
    from eden.extensions.deployment.utils.mujoco_sim import MujocoSim
    from eden.utils.assets import get_asset_path

    robot = G1Dofs29()
    dofs = list(robot.dofs_name)
    sim = MujocoSim(get_asset_path(robot.file), dofs)
    q0 = resolve_dofs_values(robot.default_dofs_pos, dofs)
    kp = resolve_dofs_values(robot.default_dofs_kp, dofs)
    kd = resolve_dofs_values(robot.default_dofs_kd, dofs)
    sim.set_implicit_damping(kd)
    sim.reset(q0, base_pos=[0.0, 0.0, 0.8])
    sim.reset(q0, base_pos=[0.0, 0.0, 0.8 - sim.lowest_geom_z() + 0.001])  # feet on the floor
    sim.set_gantry(
        enabled=True,
        mounts=(("left_shoulder_pitch_link", (0.0, 0.0, 0.0)), ("right_shoulder_pitch_link", (0.0, 0.0, 0.0))),
    )

    q1 = q0.copy()
    for i, name in enumerate(dofs):
        for joint, value in STRAIGHT_KNEE.items():
            if name.endswith(f"_{joint}_joint"):
                q1[i] = value
    import mujoco

    foot_bodies = [mujoco.mj_name2id(sim.model, mujoco.mjtObj.mjOBJ_BODY, f) for f in G1Dofs29.feet_links_name]
    n_substeps = max(1, round(CONTROL_DT / sim.sim_dt))
    ticks = []
    for k in range(SETTLE_TICKS + RAMP_TICKS + HOLD_TICKS):
        alpha = np.clip((k - SETTLE_TICKS) / RAMP_TICKS, 0.0, 1.0)
        target = q0 + alpha * (q1 - q0)
        sim.step_pd(target, np.zeros_like(target), kp, kd, np.zeros_like(target), n_substeps)
        ticks.append({**sim.read(), "true_force": _foot_contact_force(sim, foot_bodies)})
    return dofs, ticks


@pytest.fixture(scope="module")
def g1_entity():
    """A Genesis G1: the kinematics oracle the deploy loop gives the estimator."""
    from eden.options import EnvOptions, SceneOptions
    from eden.utils.configs import EdenConfig
    from tests.support.envs import build_env

    config = EdenConfig(env_options=EnvOptions(num_envs=1, sim_dt=0.005), scene_options=SceneOptions(robot=G1Dofs29()))
    return build_env(config).entities["robot"]


def _run_estimator(entity, dofs, ticks):
    assert list(entity.dofs_name) == dofs
    est = LinVelKalmanEstimator(entity, feet_links=G1Dofs29.feet_links_name, dt=CONTROL_DT)
    contacts, forces = [], []
    for r in ticks:
        # What RobotStateEntity.update does: sync orientation and joints into the sim entity's FK.
        entity.set_quat(torch.as_tensor(r["base_quat"], dtype=torch.float32).unsqueeze(0))
        entity.set_dofs_pos(torch.as_tensor(r["dofs_pos"], dtype=torch.float32).unsqueeze(0))
        entity.set_dofs_vel(torch.as_tensor(r["dofs_vel"], dtype=torch.float32).unsqueeze(0))
        est.update(r["base_quat"], r["base_ang_vel"], r["base_lin_acc"], r["dofs_vel"], r["dofs_torque"])
        contacts.append(est.contacts.copy())
        forces.append(est.forces.copy())
    return np.array(contacts), np.array(forces)


@pytest.mark.slow
@pytest.mark.requires_genesis
def test_the_stance_is_straight_kneed_and_ankle_loaded(straight_knee_stance):
    """Guard the scenario itself: if the robot fell or the knees bent, the next test proves nothing."""
    dofs, ticks = straight_knee_stance
    hold = ticks[-HOLD_TICKS:]
    knees = [dofs.index(n) for n in ("left_knee_joint", "right_knee_joint")]
    ankles = [dofs.index(n) for n in ("left_ankle_pitch_joint", "right_ankle_pitch_joint")]
    assert np.all(np.array([r["dofs_pos"][knees] for r in hold]) < 0.25)
    assert np.all(np.array([r["dofs_torque"][ankles] for r in hold]) > 3.0)  # the ankles carry the moment
    true_fz = np.array([r["true_force"][:, 2] for r in hold])
    assert np.all(true_fz > 80.0), true_fz.min()  # both feet loaded, well above the 40 N latch
    assert min(r["base_pos"][2] for r in hold) > 0.7  # still standing


@pytest.mark.slow
@pytest.mark.requires_genesis
def test_both_feet_stay_latched_with_their_true_load(straight_knee_stance, g1_entity):
    """On hardware a straight-kneed G1 read both feet airborne, as the point-force model read ~32 N here."""
    dofs, ticks = straight_knee_stance
    contacts, forces = _run_estimator(g1_entity, dofs, ticks)
    hold = slice(-HOLD_TICKS, None)
    assert contacts[hold].all(), f"feet unlatched on {np.count_nonzero(~contacts[hold])} foot-ticks of the stance"

    true_fz = np.array([r["true_force"][:, 2] for r in ticks])[hold]
    # Quasi-static, so within a few percent of the simulator's contact force. Without the leg's
    # own weight compensated it reads ~9% low; without the foot moment, ~70% low.
    np.testing.assert_allclose(forces[hold][..., 2].mean(0), true_fz.mean(0), rtol=0.05)
