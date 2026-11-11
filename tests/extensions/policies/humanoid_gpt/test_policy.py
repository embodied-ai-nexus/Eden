"""``HumanoidGptPolicy``'s checkpoint selection and the observation group its spec declares (no ONNX bundle needed)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from eden.extensions.policies.humanoid_gpt.constants import HGPT_DEFAULT_DOFS_POS, NUM_DOFS, OBS_DIM
from eden.extensions.policies.humanoid_gpt.custom_terms.observations import HgptReference
from eden.extensions.policies.humanoid_gpt.policy import _MLP_CHECKPOINT, _TRANSFORMER_CHECKPOINT, HumanoidGptPolicy
from eden.managers.observation_manager import ObservationManager
from eden.options.managers.observations import ObservationGroupOptions, ObservationManagerOptions


@pytest.mark.parametrize(
    "policy_type, checkpoint_path, md5_hash, expected",
    [
        ("mlp", None, None, (_MLP_CHECKPOINT, "mlp-md5")),
        ("transformer", None, None, (_TRANSFORMER_CHECKPOINT, "trans-md5")),
        # --ckpt-path: a local path with its md5 cleared, whichever type it is.
        ("mlp", "/tmp/local.onnx", None, ("/tmp/local.onnx", None)),
        ("transformer", "/tmp/trans.onnx", None, ("/tmp/trans.onnx", None)),
        # an explicit path is used as-is, regardless of policy_type
        ("transformer", "custom.onnx", "custom-md5", ("custom.onnx", "custom-md5")),
    ],
)
def test_select_checkpoint(policy_type, checkpoint_path, md5_hash, expected):
    selected = HumanoidGptPolicy._select_checkpoint(
        policy_type, checkpoint_path, md5_hash, _MLP_CHECKPOINT, "mlp-md5", _TRANSFORMER_CHECKPOINT, "trans-md5"
    )
    assert selected == expected


@pytest.mark.parametrize("policy_type, history", [("transformer", 16), ("mlp", None)])
def test_spec_stacks_a_frame_history_only_for_the_transformer(policy_type, history):
    policy_cls = type("LocalHumanoidGptPolicy", (HumanoidGptPolicy,), {"policy_type": policy_type})
    group = policy_cls.observation_options()["policy"]
    assert group.history_length == history
    assert list(group.term_keys()) == ["gyro", "gvec", "dofs_pos", "dofs_vel", "last_action", "reference"]


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    """The env a task-less ``eden inference --policy humanoid_gpt`` builds for the Transformer, on a short clip."""
    from eden.utils.cli.tasks import _build_extension_env, _extension_command_options

    frames = 30
    t = np.arange(frames, dtype=np.float32) / 50.0
    quat = np.zeros((frames, 2, 4), np.float32)
    quat[..., 0] = 1.0
    pos = np.zeros((frames, 2, 3), np.float32)
    pos[:, 0] = np.stack([0.2 * t, 0.0 * t, 0.79 + 0.0 * t], -1)
    pos[:, 1] = pos[:, 0] + [0.0, 0.0, 0.2]
    motion_file = tmp_path_factory.mktemp("humanoid_gpt") / "motion.npz"
    np.savez(
        motion_file,
        joint_pos=np.broadcast_to(np.asarray(HGPT_DEFAULT_DOFS_POS, np.float32), (frames, NUM_DOFS)),
        joint_vel=np.zeros((frames, NUM_DOFS), np.float32),
        body_pos_w=pos,
        body_quat_w=quat,
        body_lin_vel_w=np.zeros((frames, 2, 3), np.float32),
        body_ang_vel_w=np.zeros((frames, 2, 3), np.float32),
        body_names=np.array(["pelvis", "torso_link"]),
    )
    command = _extension_command_options(HumanoidGptPolicy, "robot", str(motion_file))
    return _build_extension_env(HumanoidGptPolicy, "G1Dofs29", "robot", False, command)


@pytest.mark.requires_genesis
def test_transformer_history_is_backfilled_then_rolls_newest_last(env):
    """The first frame after a reset fills all K slots (no zero padding), as the source's ``reset_history`` does."""
    obs, _ = env.reset()
    window = obs["policy"]
    assert window.shape == (1, 16, OBS_DIM)
    assert torch.equal(window, window[:, :1].expand_as(window))

    action = torch.full((1, NUM_DOFS), 0.1)
    obs, *_ = env.step({"dofs_pos_controller": action})
    rolled = obs["policy"]
    torch.testing.assert_close(rolled[:, :-1], window[:, 1:], rtol=0, atol=0)
    last_action = slice(3 + 3 + 2 * NUM_DOFS, 3 + 3 + 3 * NUM_DOFS)  # after gyro, gvec, q - nominal, qd
    torch.testing.assert_close(rolled[:, -1, last_action], action)


def test_observation_options_takes_the_knobs_without_touching_the_class():
    """The example once switched variants by assigning ``HumanoidGptPolicy.policy_type``; the knobs are arguments now."""
    group = HumanoidGptPolicy.observation_options(policy_type="mlp", ref_ema_alpha=None)["policy"]
    assert group.history_length is None
    assert group.reference.ema_alpha is None
    assert HumanoidGptPolicy.policy_type == "transformer"
    assert HumanoidGptPolicy.observation_options()["policy"].reference.ema_alpha == HumanoidGptPolicy.ref_ema_alpha


@pytest.mark.requires_genesis
@pytest.mark.parametrize(
    "overrides, message",
    [({"policy_type": "mlp"}, "policy_type='mlp'"), ({"ref_ema_alpha": 0.5}, "ref_ema_alpha")],
)
def test_a_policy_configured_unlike_its_env_is_refused(env, tmp_path, overrides, message):
    """Refused before the network loads, so a placeholder checkpoint is enough."""
    checkpoint = tmp_path / "unused.onnx"
    checkpoint.write_bytes(b"")
    policy_cls = type("LocalHumanoidGptPolicy", (HumanoidGptPolicy,), {})
    policy_cls.use_local_checkpoint(str(checkpoint))
    with pytest.raises(ValueError, match=message):
        policy_cls(env=env, options=policy_cls.configure(**overrides))


def _reference_manager(env, **reference) -> ObservationManager:
    return ObservationManager(
        env=env,
        options=ObservationManagerOptions(
            policy=ObservationGroupOptions(reference=HgptReference.configure(**reference))
        ),
    )


@pytest.mark.requires_genesis
def test_the_reference_block_describes_the_root_link_it_names(env):
    """Resolved by name on the command's links: the clip's torso rides 0.2 m above its pelvis."""
    env.reset()
    heights = {
        root: _reference_manager(env, root_link_name=root, ema_alpha=None).compute()["policy"][0, NUM_DOFS].item()
        for root in ("pelvis", "torso_link")
    }
    assert heights == pytest.approx({"pelvis": 0.79, "torso_link": 0.99})


@pytest.mark.requires_genesis
def test_a_reference_root_the_motion_does_not_track_is_refused_at_build(env):
    with pytest.raises(
        ValueError, match=r"tracks links \['pelvis', 'torso_link'\], not the root 'left_ankle_roll_link'"
    ):
        _reference_manager(env, root_link_name="left_ankle_roll_link")
