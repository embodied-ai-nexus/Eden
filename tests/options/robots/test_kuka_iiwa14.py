"""``Iiwa14`` built: the joint damping and effort limit its MJCF omits, and what a partial override does to them.

Without them Genesis loads an undamped arm with an unbounded drive, which diverged ~74 deg at the
shoulder from KUKA's URDF over a 200-step PD rollout (0.002 deg with them).
"""

from __future__ import annotations

import pytest
import torch

from eden.options.robots import Iiwa14
from tests.options.robots.conftest import build_scene

pytestmark = pytest.mark.requires_genesis


@pytest.fixture(scope="module")
def env():
    return build_scene(
        arm=Iiwa14(),
        retuned=Iiwa14(
            default_root_pos=(1.0, 0.0, 0.0),
            default_dofs_damping={"joint1": 2.0},
            default_dofs_force_limits={"joint1": 120.0},
        ),
    )


def test_the_class_damping_and_effort_limit_reach_the_solver(env):
    arm = env.entities["arm"]
    assert arm.num_dofs == 7
    assert torch.allclose(arm.get_dofs_damping(), torch.tensor(0.5))
    lower, upper = arm.get_dofs_force_range()
    assert torch.allclose(upper, torch.tensor(300.0)) and torch.allclose(lower, torch.tensor(-300.0))


def test_a_partial_override_replaces_the_class_default_rather_than_merging(env):
    """The six unnamed joints fall back to the MJCF's 0 damping and unbounded force, with no warning."""
    arm = env.entities["retuned"]
    i = arm.dofs_name.index("joint1")
    others = [j for j in range(7) if j != i]
    damping, upper = arm.get_dofs_damping()[0], arm.get_dofs_force_range()[1][0]
    assert (damping[i].item(), upper[i].item()) == (2.0, 120.0)
    assert torch.equal(damping[others], torch.zeros(6, device=damping.device))
    assert torch.isinf(upper[others]).all()
