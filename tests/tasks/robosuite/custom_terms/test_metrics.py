"""``shared_per_metric_pass``: a success metric's children reuse what its sibling metrics computed in the same pass.

ToolHang's ``success`` ANDs the tests of ``frame_assembled`` and ``tool_on_frame``, and TwoArmTransport's
``transported`` those of ``payload_in_target`` and ``trash_in_trash``; each test (two of them contact
queries) used to run twice a step.
"""

from __future__ import annotations

# isort: off
import eden  # noqa: F401 - isolate the DDP rank before importing Torch
# isort: on

import pytest
import torch

from eden.entities.rigid import RigidEntity
from eden.tasks import TASK_REGISTRY
from tests.support.envs import build_env

#: Distinct contact queries one metric pass of each task makes, every test of its success metric
#: being one its sibling metrics already ran: ToolHang's ring-on-bar and gripper-on-wrench,
#: TwoArmTransport's payload-in-target and trash-in-trash.
_CONTACT_QUERIES = {"robosuite_tool_hang": 2, "robosuite_two_arm_transport": 2}


@pytest.fixture(scope="module", params=sorted(_CONTACT_QUERIES))
def task(request):
    """``(task name, built env)``."""
    config = TASK_REGISTRY.build(request.param)
    config.env_options.num_envs = 2
    return request.param, build_env(config, reset=True)


@pytest.mark.requires_genesis
def test_a_shared_metric_pass_logs_what_each_metric_computes_on_its_own(task):
    """Each term evaluated alone, outside a pass (where nothing is shared), is the independent reference."""
    _, env = task
    manager = env.metric_manager
    with torch.inference_mode():
        manager.reset()  # so each buffer holds exactly this pass's value
        shared = {name: value.clone() for name, value in manager.compute().items()}
        for name in manager.active_terms:
            alone = manager.get_term(name).compute().float()
            assert torch.equal(shared[name], alone), f"metric {name!r}: pass {shared[name]} != alone {alone}"


@pytest.mark.requires_genesis
def test_each_contact_query_runs_once_per_metric_pass(task, monkeypatch):
    task_name, env = task
    queries = []
    get_contacts = RigidEntity.get_contacts

    def counted(entity, *args, **kwargs):
        queries.append(entity.name)
        return get_contacts(entity, *args, **kwargs)

    monkeypatch.setattr(RigidEntity, "get_contacts", counted)
    with torch.inference_mode():
        env.metric_manager.compute()

    assert len(queries) == _CONTACT_QUERIES[task_name], queries
