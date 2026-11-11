"""Tests for ``eden.extensions.retargeting.dex_retargeter.DexRetargeter`` on a fixed-base XHand1.

The ``env`` fixture comes from this directory's ``conftest.py``.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("nlopt")  # the optional ``dex-retargeter`` extra

from eden.extensions.retargeting.dex_retargeter import DexRetargeter  # noqa: E402


FINGER_TIPS = [
    "right_hand_thumb_rota_tip",
    "right_hand_index_rota_tip",
    "right_hand_mid_tip",
    "right_hand_ring_tip",
    "right_hand_pinky_tip",
]
WRIST = "right_hand_link"


REFERENCE_TIPS = np.array(
    [[0.05, 0.00, 0.10], [0.00, 0.00, 0.13], [0.00, -0.02, 0.12], [0.00, -0.04, 0.11], [0.00, -0.06, 0.10]],
    dtype=np.float32,
)


@pytest.mark.parametrize(
    ("kwargs", "kind"),
    [
        (
            {
                "type": "vector",
                "target_origin_link_names": [WRIST] * 5,
                "target_task_link_names": FINGER_TIPS,
                "target_link_human_indices": np.array([[0] * 5, [4, 8, 12, 16, 20]]),
                "scaling_factor": 1.5,
            },
            "VECTOR",
        ),
        (
            {
                "type": "position",
                "target_link_names": FINGER_TIPS,
                "target_link_human_indices": np.array([4, 8, 12, 16, 20]),
            },
            "POSITION",
        ),
    ],
    ids=["vector", "position"],
)
def test_one_solve_gives_a_finite_qpos_for_every_dof(env, kwargs, kind):
    retargeter = DexRetargeter(
        env=env, options=DexRetargeter.configure(entity_name="robot", low_pass_alpha=1.0, **kwargs)
    )
    qpos = retargeter.retarget(REFERENCE_TIPS)
    assert qpos.shape == (12,) and np.isfinite(qpos).all()
    assert retargeter.retargeting_type == kind


def _dexpilot(env):
    return DexRetargeter(
        env=env,
        options=DexRetargeter.configure(
            entity_name="robot",
            type="dexpilot",
            finger_tip_link_names=FINGER_TIPS,
            wrist_link_name=WRIST,
            scaling_factor=1.5,
            low_pass_alpha=1.0,
        ),
    )


def test_the_dexpilot_solver_actually_moves_the_hand(env, eden_caplog):
    """A raise inside the nlopt objective is caught and returns ``last_qpos``, which passes any finiteness check."""
    retargeter = _dexpilot(env)
    n_pairs = retargeter.optimizer.target_link_human_indices.shape[1]
    ref = np.random.default_rng(0).normal(scale=0.03, size=(n_pairs, 3)).astype(np.float32)

    start = retargeter.last_qpos.copy()
    qpos = retargeter.retarget(ref)

    assert "nlopt solver failure" not in eden_caplog.text
    assert np.abs(retargeter.last_qpos - start).max() > 1e-6, "the solver returned its initial guess"
    assert qpos.shape == (12,) and np.isfinite(qpos).all()
    assert retargeter.retargeting_type == "DEXPILOT"


def test_objective_tensors_all_land_on_the_optimizers_device(env):
    """On a GPU one stray CPU tensor makes every solve raise, silently freezing the hand.

    The suite runs on ``gs.cpu``, where a stray tensor is invisible, so the optimizer is given the ``meta``
    device and the objective's closure is inspected: it is the only place those tensors are observable.
    """
    retargeter = _dexpilot(env)
    optimizer = retargeter.optimizer
    n_pairs = optimizer.target_link_human_indices.shape[1]
    ref = np.random.default_rng(0).normal(scale=0.03, size=(n_pairs, 3)).astype(np.float32)
    fixed = np.zeros(len(optimizer.idx_fixed), dtype=np.float32)

    previous = optimizer.device
    try:
        optimizer.device = torch.device("meta")
        objective = optimizer.get_objective_function(ref, fixed, retargeter.last_qpos)
    finally:
        optimizer.device = previous

    captured = dict(zip(objective.__code__.co_freevars, (c.cell_contents for c in objective.__closure__ or ())))
    tensors = {name: v for name, v in captured.items() if isinstance(v, torch.Tensor)}
    assert tensors, "expected the objective's closure to capture tensors"
    strays = {name: str(t.device) for name, t in tensors.items() if t.device.type != "meta"}
    assert not strays, f"these ignore the optimizer's device (pass device=self.device, not from_numpy): {strays}"


def test_retarget_motion_batch(env):
    retargeter = DexRetargeter(
        env=env,
        options=DexRetargeter.configure(
            entity_name="robot",
            type="vector",
            target_origin_link_names=[WRIST, WRIST],
            target_task_link_names=FINGER_TIPS[:2],
            target_link_human_indices=np.array([[0, 0], [4, 8]]),
            low_pass_alpha=1.0,
        ),
    )
    ref_seq = np.random.default_rng(0).normal(scale=0.05, size=(4, 2, 3)).astype(np.float32)
    qpos_seq = retargeter.retarget_motion(ref_seq)
    assert qpos_seq.shape == (4, 12)
    assert np.isfinite(qpos_seq).all()


def test_position_wrong_ref_shape_raises(env):
    """The full 21-keypoint MANO array instead of the 5 fingertips used to be a broadcast error inside SLSQP."""
    retargeter = DexRetargeter(
        env=env,
        options=DexRetargeter.configure(
            entity_name="robot",
            type="position",
            target_link_names=FINGER_TIPS,
            target_link_human_indices=np.array([16, 17, 18, 19, 20]),
            low_pass_alpha=1.0,
        ),
    )
    full_21 = np.random.default_rng(0).normal(size=(21, 3)).astype(np.float32)
    with pytest.raises(ValueError, match=r"PositionOptimizer: target_pos shape"):
        retargeter.retarget(full_21)


def test_position_fingertips_reach_round_trip_targets(env):
    """FK a reachable qpos into fingertip targets and solve back to them.

    Position retargeting is non-convex and SLSQP lands in a 1-3 cm local minimum; the 5 cm bound catches
    a frame swap or gradient sign error, which puts the error past 10 cm.
    """
    retargeter = DexRetargeter(
        env=env,
        options=DexRetargeter.configure(
            entity_name="robot",
            type="position",
            target_link_names=FINGER_TIPS,
            target_link_human_indices=np.array([0, 1, 2, 3, 4]),
            low_pass_alpha=1.0,
        ),
    )

    robot = env.entities["robot"]
    target_links = retargeter.optimizer._target_links

    # Small reachable qpos perturbation from neutral.
    rng = np.random.default_rng(0)
    target_qpos = rng.uniform(-0.2, 0.2, size=12).astype(np.float32)
    robot.set_qpos(torch.as_tensor(target_qpos, device=robot.device).unsqueeze(0))
    targets = np.stack([link.get_pos().squeeze(0).cpu().numpy() for link in target_links])

    qpos = retargeter.retarget(targets)
    robot.set_qpos(torch.as_tensor(qpos, device=robot.device).unsqueeze(0))
    achieved = np.stack([link.get_pos().squeeze(0).cpu().numpy() for link in target_links])

    err = np.linalg.norm(achieved - targets, axis=-1)
    assert err.max() < 0.05, f"fingertip max err {err.max() * 1000:.2f} mm exceeds 5 cm; per-finger (mm): {err * 1000}"


def test_invalid_link_name_raises(env):
    with pytest.raises(ValueError, match="Link .* not in entity"):
        DexRetargeter(
            env=env,
            options=DexRetargeter.configure(
                entity_name="robot",
                type="vector",
                target_origin_link_names=["does_not_exist"],
                target_task_link_names=[FINGER_TIPS[0]],
                target_link_human_indices=np.array([[0], [4]]),
            ),
        )
