"""Tests for ``eden.extensions.retargeting.otretarget.presets`` against the real rigs and body models."""

from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("proxsuite")
pytest.importorskip("threadpoolctl")

import eden as en
from eden.extensions.retargeting.otretarget import PRESETS, OTRetargeter, for_body_model
from eden.extensions.retargeting.otretarget.human import SMPLX_JOINT_NAMES
from eden.options import EnvOptions, SceneOptions
from eden.options.robots.composite import G1Wuji, OliWuji
from eden.utils.configs import EdenConfig
from tests.support.envs import build_env

RIGS = {"g1_wuji": G1Wuji, "oli_wuji": OliWuji, "g1": en.robots.G1Dofs29, "oli": en.robots.Oli}
#: The SOMA rig's joints come from its model file (nvidia/soma-x); point this at SOMA_neutral.npz.
SOMA_MODEL = os.environ.get("EDEN_SOMA_MODEL", "")


@pytest.mark.slow
@pytest.mark.parametrize("name", sorted(PRESETS))
def test_preset_builds_on_its_rig_and_every_probed_link_has_a_human_part(name):
    env = build_env(EdenConfig(env_options=EnvOptions(num_envs=1), scene_options=SceneOptions(robot=RIGS[name]())))
    # Construction validates every link, joint and T-pose name of the preset against the built rig.
    ret = OTRetargeter(env, OTRetargeter.configure(**PRESETS[name].retargeter_options()))
    grouped = {link for _, links in PRESETS[name].correspondences for link in links}
    probed = {ret.kin.link_names[c] for c in np.unique(ret.robot_probes.carrier)}
    assert probed <= grouped, f"links with surface but no human counterpart: {sorted(probed - grouped)}"


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_preset_human_side_names_smplx_joints(name):
    preset = PRESETS[name]
    parts = {p for human, _ in preset.correspondences for p in human} | set(preset.style_links.values())
    assert parts <= set(SMPLX_JOINT_NAMES), sorted(parts - set(SMPLX_JOINT_NAMES))
    # Each human part is matched to one group only.
    flat = [p for human, _ in preset.correspondences for p in human]
    assert len(flat) == len(set(flat))


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_soma_translation_names_soma_joints(name):
    if not SOMA_MODEL or not os.path.isfile(SOMA_MODEL):
        pytest.skip(f"SOMA model not found at {SOMA_MODEL} (set EDEN_SOMA_MODEL)")
    soma = {str(n) for n in np.load(SOMA_MODEL)["joint_names"]}
    preset = for_body_model(PRESETS[name], "soma")
    parts = {p for human, _ in preset.correspondences for p in human} | set(preset.style_links.values())
    assert parts <= soma, sorted(parts - soma)
    assert preset.human_root == "Hips"
