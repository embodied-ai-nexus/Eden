"""The contact observation terms in ``eden/managers/terms/observations/contact.py``."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from eden.managers.terms.observations.contact import (
    ContactForceNorm,
    GroupedContactFlags,
    motion_contact_flags,
    motion_future_contact_label,
)
from tests.managers.terms.conftest import write_stack_urdf
from tests.support.motion import build_mimic_env, manager_options, set_reference, term_options, write_clip

pytestmark = pytest.mark.requires_genesis


@pytest.mark.parametrize(
    ("links_name", "match"),
    [([], "matched no links"), (["nonexistent_link"], "Not all query patterns are matched")],
    ids=["empty", "unmatched"],
)
def test_contact_force_norm_rejects_a_bad_link_selection_at_build(contact_env, make_term, links_name, match):
    with pytest.raises(ValueError, match=match):
        make_term(ContactForceNorm, contact_env, entity_name="stack", links_name=links_name)


def test_contact_force_norm_reads_the_selected_links_in_the_order_given(contact_env, make_term):
    """One column per named link: the loaded ``foot`` carries the stack's weight, ``torso`` nothing."""
    forces = contact_env.entities["stack"].get_links_net_contact_force()
    term = make_term(ContactForceNorm, contact_env, entity_name="stack", links_name=["torso", "foot"])
    out = term.compute()
    torch.testing.assert_close(out, torch.norm(forces[:, [2, 0]], dim=-1))
    assert (out[:, 0] == 0.0).all() and (out[:, 1] > 40.0).all()


def test_contact_force_norm_follows_a_swapped_entity(contact_env, make_term):
    """The entity is resolved per ``compute()``, so a deployment-style swap of ``env.entities`` is seen."""
    term = make_term(ContactForceNorm, contact_env, entity_name="stack", links_name=["foot"])
    stack = contact_env.entities["stack"]
    before = term.compute().clone()
    contact_env.entities["stack"] = contact_env.entities["cube_a"]  # a 0.125 kg cube resting on the plane
    try:
        after = term.compute()
    finally:
        contact_env.entities["stack"] = stack
    torch.testing.assert_close(after, torch.norm(contact_env.entities["cube_a"].get_links_net_contact_force(), dim=-1))
    assert (after < 2.0).all() and (before > 40.0).all()


# ---------------------------------------------------------------------------
# Contact flags between link groups, simulated and demonstrated
# ---------------------------------------------------------------------------

SEQUENCES = {"robot": ([], ["box_baselink"]), "object": ([], ["bottom", "top"])}


@pytest.fixture(scope="module")
def mimic_env(tmp_path_factory):
    """A box (``robot``) come to rest on the top link of a fixed two-link stack (``object``), with a demo clip."""
    from eden.options.entities import BoxOptions, EntityOptions

    root = tmp_path_factory.mktemp("contact_obs")
    contact = {
        "robot.contact_points_w": np.zeros((20, 1, 2, 3), np.float32),
        "robot.contact_points_valid": np.zeros((20, 1, 2), bool),
        "robot.contact_label": np.zeros((20, 1), np.int8),
    }
    write_clip(root / "clip_00.npz", 20, sequences=SEQUENCES, overrides=contact)
    env = build_mimic_env(
        root,
        num_envs=2,
        scene={
            "robot": BoxOptions(size=(0.1, 0.1, 0.1), default_root_pos=(0.0, 0.0, 0.3)),
            "object": EntityOptions(
                file=write_stack_urdf(root / "stack.urdf"), links_to_keep=["top"], is_fixed_base=True
            ),
        },
        motion_options=manager_options(terms={"default": term_options(root, sequences=SEQUENCES)}, num_frames=3),
    )
    for _ in range(30):
        env.step()
    return env


@pytest.mark.parametrize("object_links, expected", [([], [0.0, 1.0]), (["top", "bottom"], [1.0, 0.0])])
def test_grouped_contact_flags_are_the_link_pair_contacts_as_floats(mimic_env, object_links, expected):
    term = GroupedContactFlags(
        mimic_env, GroupedContactFlags.configure(links_name=["box_baselink"], object_links_name=object_links)
    )
    term.build()

    out = term.compute()

    assert out.dtype == torch.float32
    assert out.tolist() == [expected] * 2


def test_grouped_contact_flags_need_links(mimic_env):
    with pytest.raises(ValueError, match="links_name is required"):
        GroupedContactFlags(mimic_env, GroupedContactFlags.configure()).build()


def test_motion_contact_flags_are_the_demo_flags_of_the_named_links(mimic_env):
    set_reference(mimic_env, "robot", contact_points_valid=torch.tensor([[[[False, True]]], [[[True, True]]]]))

    assert motion_contact_flags(mimic_env, entity_name="robot", links_name=["box_baselink"]).tolist() == [
        [0.0, 1.0],
        [1.0, 1.0],
    ]


@pytest.mark.parametrize("threshold, expected", [(None, [0.0, 1.0]), (1.0, [0.0, 1.0]), (5.0, [0.0, 0.0])])
def test_grouped_contact_flags_can_require_a_contact_force(mimic_env, threshold, expected):
    """The resting 0.2 kg box presses ~2 N on ``top``: a 1 N sensor sees it, a 5 N one does not."""
    term = GroupedContactFlags(
        mimic_env, GroupedContactFlags.configure(links_name=["box_baselink"], force_threshold=threshold)
    )
    term.build()

    assert term.compute().tolist() == [expected] * 2


def test_motion_future_contact_labels_are_the_demo_labels_at_the_offset_frames(mimic_env):
    labels = torch.tensor([[[1], [-1], [0]], [[0], [1], [-1]]], dtype=torch.int8)  # (env, frame, link)
    set_reference(mimic_env, "robot", contact_label=labels)

    out = motion_future_contact_label(mimic_env, entity_name="robot", frame_offsets=(2, 0))
    structured = motion_future_contact_label(
        mimic_env, entity_name="robot", frame_offsets=(1,), links_name=["box_baselink"], flatten=False
    )

    assert out.dtype == torch.float32 and out.tolist() == [[0.0, 1.0], [-1.0, 0.0]]
    assert structured.shape == (2, 1, 1) and structured[:, 0, 0].tolist() == [-1.0, 1.0]


def test_motion_future_contact_labels_need_a_labelled_reference(mimic_env):
    with pytest.raises(ValueError, match="does not populate 'contact_label'"):
        motion_future_contact_label(mimic_env, entity_name="object", frame_offsets=(0,))
