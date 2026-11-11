"""The released OmniReset experts' I/O contract: which checkpoint, and what it observes in which order."""

from __future__ import annotations

#: The actor's observation terms and widths, in order: the ``target_terms`` of an Isaac Lab 3 expert's
#: ``uwlab_observation_layout`` (UWLab's ``ObservationsCfg.PolicyCfg`` declaration order).
POLICY_OBSERVATION_LAYOUT = (
    ("prev_actions", 7),
    ("joint_pos", 12),
    ("end_effector_pose", 6),
    ("insertive_asset_pose", 6),
    ("receptive_asset_pose", 6),
    ("insertive_asset_in_receptive_asset_frame", 6),
)

#: The actor's action width: 6 end-effector deltas (``arm``), then the gripper.
POLICY_ACTION_DIM = 7

#: Each pair's released expert (seed 42) inside ``UW-Lab/uwlab-assets`` at
#: ``eden.tasks.omnireset.constants.DATASETS_REVISION`` (its ``main`` holds an older Isaac Lab 2 layout under the
#: same names), and its md5. Every expert has the same network and I/O contract.
EXPERT_CHECKPOINTS = {
    pair: (f"Policies/OmniReset/state_based_experts/{pair}_state_rl_expert_seed42.pt", md5)
    for pair, md5 in {
        "peg": "dbbd7d0f4038627e49ddb277817091d1",
    }.items()
}
