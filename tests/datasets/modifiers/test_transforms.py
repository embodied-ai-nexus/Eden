"""Tests for ``eden.datasets.modifiers.transforms``: spatial transforms and their composition."""

import pytest
import torch

from eden.datasets.modifiers import Compose, MirrorLR, Rotate, Translate
from eden.utils.motion_reference.motion_data import BatchMotionSequence

TOL = 1e-5


def _single_point(x=1.0, y=0.0, z=0.0):
    seq = BatchMotionSequence.make_empty(buffer_lengths=[1], num_dofs=0, num_links=0)
    seq.base_pos_w.concatenated_tensor.copy_(torch.tensor([[x, y, z]]))
    seq.base_lin_vel_w.concatenated_tensor.copy_(torch.tensor([[x, y, z]]))
    return seq


def test_translate_shifts_positions_and_leaves_velocities(make_sequence):
    seq = make_sequence()
    pos, vel = seq.base_pos_w.concatenated_tensor.clone(), seq.base_lin_vel_w.concatenated_tensor.clone()

    out = Translate(options=Translate.configure(offset=(0.1, -0.2, 0.3))).apply_to_sequence(seq)

    torch.testing.assert_close(out.base_pos_w.concatenated_tensor, pos + torch.tensor([0.1, -0.2, 0.3]))
    assert torch.equal(out.base_lin_vel_w.concatenated_tensor, vel)


@pytest.mark.parametrize(
    ("cls", "kwargs"),
    [(Translate, {"offset": (0.0, 0.0, 0.0)}), (Rotate, {"angle_deg": 0.0})],
    ids=["zero-offset", "zero-angle"],
)
def test_an_identity_transform_is_a_noop(make_sequence, cls, kwargs):
    seq = make_sequence()
    before = seq.base_pos_w.concatenated_tensor.clone()
    cls(options=cls.configure(**kwargs)).apply_to_sequence(seq)
    assert torch.equal(seq.base_pos_w.concatenated_tensor, before)


def test_rotate_90_about_z_turns_positions_and_velocities():
    out = Rotate(options=Rotate.configure(angle_deg=90.0)).apply_to_sequence(_single_point())
    torch.testing.assert_close(out.base_pos_w.concatenated_tensor, torch.tensor([[0.0, 1.0, 0.0]]), atol=TOL, rtol=0)
    torch.testing.assert_close(
        out.base_lin_vel_w.concatenated_tensor, torch.tensor([[0.0, 1.0, 0.0]]), atol=TOL, rtol=0
    )


def test_rotate_keeps_quaternions_unit(make_sequence):
    seq = make_sequence()
    quats = torch.randn(seq.base_quat_w.concatenated_tensor.shape)
    seq.base_quat_w.concatenated_tensor.copy_(quats / quats.norm(dim=-1, keepdim=True))

    Rotate(options=Rotate.configure(angle_deg=37.0, axis=(0.2, 0.3, 0.9))).apply_to_sequence(seq)

    torch.testing.assert_close(seq.base_quat_w.concatenated_tensor.norm(dim=-1), torch.ones(8))


def test_rotate_refuses_a_zero_axis(make_sequence):
    with pytest.raises(ValueError, match="non-zero"):
        Rotate(options=Rotate.configure(angle_deg=45.0, axis=(0.0, 0.0, 0.0))).apply_to_sequence(make_sequence())


def test_compose_applies_in_order():
    """Translate-then-rotate takes (1, 0, 0) to (-1, 1, 0); rotate-then-translate to (0, 2, 0)."""
    tr, rot = Translate.configure(offset=(0.0, 1.0, 0.0)), Rotate.configure(angle_deg=90.0)

    a = Compose(options=Compose.configure(modifiers=[tr, rot])).apply({"r": _single_point()})["r"]
    b = Compose(options=Compose.configure(modifiers=[rot, tr])).apply({"r": _single_point()})["r"]

    torch.testing.assert_close(a.base_pos_w[0][0], torch.tensor([-1.0, 1.0, 0.0]), atol=TOL, rtol=0)
    torch.testing.assert_close(b.base_pos_w[0][0], torch.tensor([0.0, 2.0, 0.0]), atol=TOL, rtol=0)


def test_compose_of_inverse_transforms_is_the_identity(make_sequence):
    seq = make_sequence()
    pos, vel = seq.base_pos_w.concatenated_tensor.clone(), seq.base_lin_vel_w.concatenated_tensor.clone()
    mods = [
        Translate.configure(offset=(0.5, 1.0, -0.2)),
        Rotate.configure(angle_deg=37.0),
        Rotate.configure(angle_deg=-37.0),
        Translate.configure(offset=(-0.5, -1.0, 0.2)),
    ]

    Compose(options=Compose.configure(modifiers=mods)).apply({"r": seq})

    torch.testing.assert_close(seq.base_pos_w.concatenated_tensor, pos, atol=1e-4, rtol=0)
    torch.testing.assert_close(seq.base_lin_vel_w.concatenated_tensor, vel, atol=1e-4, rtol=0)


def test_compose_keys_restrict_every_child(make_sequence):
    robot, obj = make_sequence(), make_sequence()
    robot_z, obj_pos = robot.base_pos_w.concatenated_tensor[..., 2].clone(), obj.base_pos_w.concatenated_tensor.clone()

    out = Compose(
        options=Compose.configure(
            modifiers=[Translate.configure(offset=(0.0, 0.0, 1.0))], motion_sequence_keys=("robot",)
        )
    ).apply({"robot": robot, "object": obj})

    torch.testing.assert_close(out["robot"].base_pos_w.concatenated_tensor[..., 2], robot_z + 1.0)
    assert torch.equal(out["object"].base_pos_w.concatenated_tensor, obj_pos)


def test_mirror_flips_the_mirror_axis_and_keeps_quaternions_unit(make_sequence):
    seq = make_sequence()
    pos = seq.base_pos_w.concatenated_tensor.clone()

    out = MirrorLR(options=MirrorLR.configure(axis=1)).apply_to_sequence(seq)

    torch.testing.assert_close(out.base_pos_w.concatenated_tensor, pos * torch.tensor([1.0, -1.0, 1.0]))
    torch.testing.assert_close(out.base_quat_w.concatenated_tensor.norm(dim=-1), torch.ones(8))


def test_mirror_swaps_the_paired_dofs_and_links(make_sequence):
    seq = make_sequence(buffer_lengths=(2,), num_dofs=4, num_links=3)
    seq.dofs_pos.concatenated_tensor[:] = torch.tensor([0.0, 1.0, 2.0, 3.0])
    seq.links_pos_w.concatenated_tensor[:] = torch.tensor([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0], [20.0, 20.0, 20.0]])

    out = MirrorLR(options=MirrorLR.configure(axis=1, dofs_swap=((0, 2),), links_swap=((0, 2),))).apply_to_sequence(seq)

    assert out.dofs_pos.concatenated_tensor[0].tolist() == [2.0, 1.0, 0.0, 3.0]
    torch.testing.assert_close(out.links_pos_w.concatenated_tensor[0, 0], torch.tensor([20.0, -20.0, 20.0]))


def test_mirror_refuses_an_axis_that_is_not_x_y_or_z(make_sequence):
    with pytest.raises(ValueError, match="axis must be 0, 1, or 2"):
        MirrorLR(options=MirrorLR.configure(axis=3)).apply_to_sequence(make_sequence())


def test_mirroring_refuses_a_sequence_with_body_model_parameters(make_sequence):
    """A mirrored MANO pose is the other hand's model, which no sign flip of the stored rotations gives."""
    from tests.datasets.modifiers.conftest import with_body_model

    mirror = MirrorLR(options=MirrorLR.configure(axis=1))
    with pytest.raises(ValueError, match="MirrorLR cannot apply to 'hand': it carries body-model parameters"):
        mirror.apply({"hand": with_body_model(make_sequence())})
    mirror.apply({"robot": make_sequence()})  # without them, as before
