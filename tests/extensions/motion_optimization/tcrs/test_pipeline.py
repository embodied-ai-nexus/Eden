"""``TCRSMotionAdapter.adapt`` on a stair height field: contacts land on the treads, and the output matches a golden."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from eden.extensions.motion_optimization.tcrs.demo import make_demo_clip
from eden.extensions.motion_optimization.tcrs.pipeline import ClipArrays, TCRSMotionAdapter, _capsule_clearance
from eden.options.extensions.motion_optimization import TCRSPlannerOptions, TCRSRobotOptions
from eden.options.robots.unitree_g1 import G1_DOFS29_NAME

GOLDEN = Path(__file__).resolve().parents[3] / "data" / "tcrs_stair_golden.npz"

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def stair_adapt(g1_mjcf, stair_terrain):
    """One deterministic adapt shared by both tests: ``(result, terrain)``.

    0.3 m treads fit the G1 foot (heel -0.06 m to midfoot probe +0.15 m), so footholds resolve to full support
    rather than straddling an edge; ``lowpass_cutoff_hz=0`` keeps stance frames on the latched foothold exactly.
    """
    terrain = stair_terrain(step_depth=0.3)
    fps, dofs_pos, links_pos, links_quat = make_demo_clip(num_frames=100)
    adapter = TCRSMotionAdapter(model_path=str(g1_mjcf), params=TCRSPlannerOptions(lowpass_cutoff_hz=0.0))
    clip = ClipArrays(fps=fps, dofs_pos=dofs_pos, links_pos_w=links_pos, links_quat_w=links_quat)
    return adapter.adapt(clip, terrain), terrain


def test_contact_frames_sit_on_the_tread_under_the_foot(stair_adapt):
    result, terrain = stair_adapt
    spec = TCRSRobotOptions()
    for motion_set in (result.raw, result.ghost, result.optimized):
        assert motion_set.dofs_pos.shape == (100, len(G1_DOFS29_NAME))
        assert np.isfinite(motion_set.dofs_pos).all() and np.isfinite(motion_set.links_pos_w).all()
    np.testing.assert_allclose(np.linalg.norm(result.optimized.links_quat_w, axis=-1), 1.0, atol=1e-6)
    assert result.ik_bad_frame_count == 0

    # The sole goes on the highest surface under the ankle and its toe/heel probes; the walk is straight along +x.
    for foot_index, contact in (
        (spec.left_foot_index, result.left_contact),
        (spec.right_foot_index, result.right_contact),
    ):
        assert contact.any()
        for frame in np.flatnonzero(contact):
            x, y, z = result.optimized.links_pos_w[frame, foot_index]
            support = max(
                terrain.height_at(float(x + dx), float(y)) for dx in (0.0, spec.toe_offset[0], spec.heel_offset[0])
            )
            assert abs(z - (support + spec.foot_nominal_z)) < 0.03, f"frame {frame}: z={z:.3f} support={support:.3f}"


def test_matches_the_stair_golden(stair_adapt):
    """Guards solver refactors: drift beyond float round-off in foot placement, leg angles or IK error shows here."""
    result, _ = stair_adapt
    golden = np.load(GOLDEN)
    spec = TCRSRobotOptions()

    for foot_index, mask in (
        (spec.left_foot_index, golden["left_contact"]),
        (spec.right_foot_index, golden["right_contact"]),
    ):
        assert mask.any()
        delta = np.linalg.norm(
            result.optimized.links_pos_w[mask, foot_index] - golden["links_pos_w"][mask, foot_index], axis=-1
        )
        assert delta.mean() < 5e-3, f"mean contact-foot drift {delta.mean():.4f} m"
        assert delta.max() < 2e-2, f"max contact-foot drift {delta.max():.4f} m"

    dofs = list(G1_DOFS29_NAME)
    legs = [dofs.index(name) for name in spec.left_leg_dofs_name + spec.right_leg_dofs_name]
    leg_p95 = np.percentile(np.abs(result.optimized.dofs_pos[:, legs] - golden["dofs_pos"][:, legs]), 95)
    assert leg_p95 < 0.05, f"leg joint p95 drift {leg_p95:.4f} rad"
    # No worse than the fixture's p95 IK error plus slack (mm).
    assert np.percentile(result.ik_pos_errors, 95) <= float(np.percentile(golden["ik_pos_errors"], 95)) * 1.25 + 1.0


def test_capsule_clearance_is_the_true_segment_distance_minus_the_radii():
    # The pipeline's own closest-points helper never recomputed one segment's parameter after clamping the other's,
    # and overstated the distance of about a quarter of random segment pairs (by up to 1.7 m at unit scale): a
    # self-collision read as clear. It now uses eden.utils.geom.segment_closest_points.
    rng = np.random.default_rng(0)
    s = np.linspace(0.0, 1.0, 801)
    for p0, p1, q0, q1 in rng.normal(size=(50, 4, 3)):
        on_p = p0 + s[:, None] * (p1 - p0)
        on_q = q0 + s[:, None] * (q1 - q0)
        brute = np.linalg.norm(on_p[:, None] - on_q[None], axis=-1).min()
        clearance, _, _ = _capsule_clearance(p0, p1, 0.05, q0, q1, 0.02)
        assert brute - 2e-3 <= clearance + 0.07 <= brute + 1e-12
