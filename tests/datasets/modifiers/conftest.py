"""A ``BatchMotionSequence`` factory with distinct, deterministic values for the dataset-modifier tests."""

import pytest
import torch

from eden.utils.motion_reference.motion_data import BatchMotionSequence


def _ramp(rows, cols, scale):
    return torch.arange(rows * cols, dtype=torch.float32).reshape(rows, cols) * scale


@pytest.fixture
def make_sequence():
    """``make_sequence(buffer_lengths=(5, 3), num_dofs=4, num_links=2)``: ramps in every base and DOF field."""

    def make(buffer_lengths=(5, 3), num_dofs=4, num_links=2):
        seq = BatchMotionSequence.make_empty(
            buffer_lengths=list(buffer_lengths), num_dofs=num_dofs, num_links=num_links
        )
        total = int(sum(buffer_lengths))
        seq.base_pos_w.concatenated_tensor.copy_(_ramp(total, 3, 1.0))
        seq.base_lin_vel_w.concatenated_tensor.copy_(_ramp(total, 3, 0.1))
        seq.base_ang_vel_w.concatenated_tensor.copy_(_ramp(total, 3, 0.01))
        seq.dofs_pos.concatenated_tensor.copy_(_ramp(total, num_dofs, 0.01))
        seq.dofs_vel.concatenated_tensor.copy_(_ramp(total, num_dofs, 0.001))
        return seq

    return make


@pytest.fixture
def g1_clips():
    """``g1_clips(base_pos_zs, num_links=0, T=8, hip_roll=0.0)``: G1 standing clips, one per root height.

    Each clip holds G1's default IK pose at 50 fps with zero velocities; ``hip_roll`` opens both hips.
    """
    import numpy as np

    from eden.options.robots.unitree_g1 import G1_IK_PROFILE
    from eden.utils.motion_reference.bipedal_ik import _resolve_default_qpos
    from eden.utils.motion_reference.motion_data import MotionSequence

    def make(base_pos_zs, num_links=0, T=8, hip_roll=0.0):
        default = _resolve_default_qpos(G1_IK_PROFILE).copy()
        default[7 + 1], default[7 + 7] = hip_roll, -hip_roll
        num_dofs = default.shape[0] - 7
        seqs = []
        for base_z in base_pos_zs:
            qpos = np.tile(default, (T, 1)).astype(np.float32)
            qpos[:, 2] = base_z
            fields = dict(
                framerate=torch.tensor(50.0),
                buffer_length=torch.tensor(T),
                base_pos_w=torch.from_numpy(qpos[:, :3]),
                base_quat_w=torch.from_numpy(qpos[:, 3:7]),
                dofs_pos=torch.from_numpy(qpos[:, 7:]),
                dofs_vel=torch.zeros(T, num_dofs),
                base_lin_vel_w=torch.zeros(T, 3),
                base_ang_vel_w=torch.zeros(T, 3),
            )
            if num_links:
                identity = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(T, num_links, 4).clone()
                for name in (
                    "links_pos_w",
                    "links_lin_vel_w",
                    "links_ang_vel_w",
                    "links_pos_b",
                    "links_lin_vel_b",
                    "links_ang_vel_b",
                ):
                    fields[name] = torch.zeros(T, num_links, 3)
                fields["links_quat_w"], fields["links_quat_b"] = identity, identity.clone()
            seqs.append(MotionSequence(**fields))
        return BatchMotionSequence.from_motion_sequences(seqs, num_links=num_links)

    return make


def with_body_model(seq: BatchMotionSequence, model: str = "mano_right") -> BatchMotionSequence:
    """``seq`` with identity body-model poses and a shape whose first coefficient is each frame's index in its clip."""
    from eden.utils.buffers.cbt import ConcatBatchTensor

    lengths = seq.buffer_length.tolist()
    total = int(sum(lengths))
    pose = torch.zeros(total, 15, 4)
    pose[..., 0] = 1.0
    shape = torch.zeros(total, 10)
    shape[:, 0] = torch.cat([torch.arange(n, dtype=torch.float32) for n in lengths])
    seq.body_pose_quat = ConcatBatchTensor.from_concatenated(pose, batch_sizes=lengths)
    seq.body_shape = ConcatBatchTensor.from_concatenated(shape, batch_sizes=lengths)
    seq.body_model = (model,) * len(lengths)
    return seq
