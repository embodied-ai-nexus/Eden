"""Welding action terms for suction-cup and parallel-jaw attach/detach."""

from __future__ import annotations

from typing import TYPE_CHECKING

import genesis as gs
import torch

from eden.managers.action_manager import ACTION_TERM_REGISTRY, ActionTerm
from eden.managers.base import ConfigurableTermBase
from eden.managers.terms.utils import parallel_jaw_is_grasping
from eden.utils.misc import is_env_mask
from eden.utils.string import term_subject

if TYPE_CHECKING:
    from eden.envs.base import EnvBase
    from eden.options.managers.actions import ActionTermOptions


class _WeldingBase(ActionTerm):
    """A special ActionTerm that does not require any dofs for control."""

    entity_name: str = ""
    ee_link_name: str = ""
    ignore_entities_name: list[str] = []
    ignore_links_idx: list[int] = []

    def __init__(self, env: EnvBase, options: ActionTermOptions):
        ConfigurableTermBase.__init__(self, env=env, options=options)

        assert not self.dofs_name, "This ActionTerm does not require any dofs for control."

        self._entity = None
        self.dofs_name = []
        self.dofs_idx_local = torch.tensor([], dtype=gs.tc_int, device=self.device)
        self._n_dofs = 0

        self._raw_action: torch.Tensor | None = None
        self._prev_action: torch.Tensor | None = None
        self._processed_action: torch.Tensor | None = None

        self._ee_link_idx = None
        self._welded = None
        self._ignore_links_idx: torch.Tensor | None = None

    def build(self) -> None:
        self._entity = self.resolve_entity(self.entity_name)

        self._raw_action = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._prev_action = torch.zeros(self.num_envs, self.action_dim, device=self.device)
        self._processed_action = torch.zeros(self.num_envs, device=self.device)

        ee_link = self._entity.get_link(self.ee_link_name)
        assert ee_link.n_geoms > 0, (
            "End-effector link must have at least one geom, but got 0; make sure the link is not a site."
        )
        self._ee_link_idx = ee_link.idx
        self._welded = torch.zeros(self.num_envs, dtype=gs.tc_bool, device=self.device)
        n_links = self._env.rigid_solver.n_links
        bad_links_idx = [idx for idx in self.ignore_links_idx if not 0 <= idx < n_links]
        if bad_links_idx:
            raise ValueError(
                f"{term_subject(self)}: ignore_links_idx {bad_links_idx} are not rigid-solver link indices; "
                f"the scene has {n_links} links, so each index must be in [0, {n_links - 1}].\n"
                f"ignore_links_idx takes solver-global indices, not entity-local ones. "
                f"Fix: exclude a whole entity with ignore_entities_name=['<entity>'] "
                f"(entities: {list(self._env.entities.keys())}), or one link with "
                f"env.entities['<entity>'].get_link('<link>').idx.\n"
                f"Do not drop or clamp the out-of-range values: an index that matches nothing leaves the link "
                f"it was meant to exclude weldable."
            )
        ignore_entities = [self.resolve_entity(name) for name in self.ignore_entities_name]
        ignore_links_idx = set(self.ignore_links_idx).union(
            link.idx for ignore_entity in ignore_entities for link in ignore_entity.links
        )
        self._ignore_links_idx = torch.tensor(sorted(ignore_links_idx), dtype=gs.tc_int, device=self.device)

    def _weldable(self, links_idx: torch.Tensor) -> torch.Tensor:
        """Drop the ignored links from ``links_idx``, a 1-D tensor of solver link indices."""
        return links_idx[~torch.isin(links_idx, self._ignore_links_idx)]

    def compute(self, actions: torch.Tensor) -> None:
        self._prev_action[:] = self._raw_action
        self._raw_action[:] = actions
        if actions.dtype == torch.bool:
            # true: suction, false: release. squeeze(-1) drops only the action-dim (1) so a
            # single-env (1, 1) action stays (1,) rather than collapsing to a 0-d scalar.
            self._processed_action = actions.squeeze(-1)
        else:
            # true: suction, false: release
            self._processed_action = actions.squeeze(-1) > 0

        # NOTE: remove weld constraints if the action is release
        weld_const_info = self._env.rigid_solver.get_weld_constraints(as_tensor=True, to_torch=True)
        link_a = weld_const_info["link_a"]
        link_b = weld_const_info["link_b"]
        objects_link_idx = torch.unique(
            torch.cat(
                [
                    link_a[link_b == self._ee_link_idx],
                    link_b[link_a == self._ee_link_idx],
                ]
            )
        )
        unweld_mask = ~self._processed_action & self._welded
        if not unweld_mask.any():
            return
        for link_idx in objects_link_idx:
            # NOTE: ensure the object is welded before deleting the weld constraint
            envs_unweld = unweld_mask & (
                ((link_a == link_idx) & (link_b == self._ee_link_idx)).any(dim=1)
                | ((link_b == link_idx) & (link_a == self._ee_link_idx)).any(dim=1)
            )
            self._env.rigid_solver.delete_weld_constraint(link_idx, self._ee_link_idx, envs_idx=envs_unweld)
            self._welded[envs_unweld] = False

    def reset(self, envs_idx: slice | torch.Tensor | None = None) -> None:
        # The release below ANDs envs_idx into a per-env "welded here" row, so every selector must be a
        # bool mask: an index tensor would AND bitwise and be read back as env indices.
        if not is_env_mask(envs_idx):
            envs_mask = torch.zeros(self.num_envs, dtype=gs.tc_bool, device=self.device)
            envs_mask[slice(None) if envs_idx is None else envs_idx] = True
            envs_idx = envs_mask

        self._prev_action[envs_idx] = 0.0
        self._raw_action[envs_idx] = 0.0

        weld_const_info = self._env.rigid_solver.get_weld_constraints(as_tensor=True, to_torch=True)
        link_a = weld_const_info["link_a"]
        link_b = weld_const_info["link_b"]
        objects_link_idx = torch.unique(
            torch.cat(
                [
                    link_a[link_b == self._ee_link_idx],
                    link_b[link_a == self._ee_link_idx],
                ]
            )
        )
        for link_idx in objects_link_idx:
            envs_unweld = envs_idx & (
                ((link_a == link_idx) & (link_b == self._ee_link_idx)).any(dim=1)
                | ((link_b == link_idx) & (link_a == self._ee_link_idx)).any(dim=1)
            )
            if not envs_unweld.any():
                continue
            self._env.rigid_solver.delete_weld_constraint(link_idx, self._ee_link_idx, envs_idx=envs_unweld)
        # Every selected env is free to weld again, whether or not a constraint was left to delete above:
        # ``env.reset`` resets the scene before the action terms, and Genesis' full ``set_state`` drops every
        # weld added at run time, so by now the constraint is usually gone. Clearing only the envs it found
        # would leave ``_welded`` True there, and that env would never weld again.
        self._welded[envs_idx] = False


@ACTION_TERM_REGISTRY.register()
class SuctionCupWelding(_WeldingBase):
    """Collision-aware welding of an object link to the suction-cup end-effector link.

    When the action is applied and the end-effector link is in contact with an object link,
    the end-effector link and the object link are welded together.

    Parameters
    ----------
    entity_name: str
        The name of the entity to control.
    ee_link_name: str
        The name of the end-effector link.
    ignore_entities_name: list[str]
        The names of the entities to exclude from welding (e.g., "table", "floor").
    ignore_links_idx: list[int]
        The rigid-solver (scene-global) link indices to exclude from welding, e.g.
        ``env.entities["table"].get_link("top").idx``; an entity-local ``idx_local`` excludes the wrong link.

    Note
    ----
    action is a boolean tensor, True: suction, False: release
    if float is used, >0: suction, <=0: release
    """

    # The action *is* the vacuum on/off bit: no other term supplies it, so a deployment that
    # cannot carry it must refuse rather than drop it (see ActionTerm.sim_surrogate).
    _sim_surrogate_ = False

    @property
    def action_dim(self) -> int:
        return 1

    def apply_actions(self) -> None:
        contact_info = self._entity.get_contacts(exclude_self_contact=True)
        link_a = contact_info["link_a"]
        link_b = contact_info["link_b"]
        valid_mask = contact_info["valid_mask"]
        # NOTE: get object links in contact
        objects_link_idx = torch.unique(
            torch.cat(
                [
                    link_a[(link_b == self._ee_link_idx) & valid_mask],
                    link_b[(link_a == self._ee_link_idx) & valid_mask],
                ]
            )
        )
        suction_mask = self._processed_action & ~self._welded
        if not suction_mask.any():
            return
        for link_idx in self._weldable(objects_link_idx):
            # NOTE: only add weld constraint if the object is not welded and the action is true
            envs_weld = suction_mask & (
                ((link_a == link_idx) & (link_b == self._ee_link_idx) & valid_mask).any(dim=1)
                | ((link_b == link_idx) & (link_a == self._ee_link_idx) & valid_mask).any(dim=1)
            )
            if not envs_weld.any():
                continue
            self._env.rigid_solver.add_weld_constraint(link_idx, self._ee_link_idx, envs_idx=envs_weld)
            self._welded[envs_weld] = True


@ACTION_TERM_REGISTRY.register()
class ParallelJawWelding(_WeldingBase):
    """Weld the link a parallel-jaw gripper is squeezing to the end-effector link.

    While the action is on, an env with no weld welds a link to ``ee_link_name`` once **both** pads
    press that same link with contact force along the closing direction above
    ``grasp_force_threshold`` (:func:`~eden.managers.terms.utils.parallel_jaw_is_grasping` with
    ``object_link_idx``). A link touching only one pad is never welded, and neither are two links
    each pressed by one pad: a grasp that spans two links of a multi-link object welds nothing.
    Releasing the action deletes the weld. An env holds at most one weld; if several links pass in
    the same step, the one with the lowest solver index is welded.

    The force test is sign-agnostic, so both pads scraping the same surface while closing can pass
    it: list support surfaces (table, floor) in ``ignore_entities_name``.

    Parameters
    ----------
    entity_name: str
        The name of the entity to control. (e.g., "robot")
    ee_link_name: str
        The name of the end-effector link. (e.g., "hand")
    ignore_entities_name: list[str]
        The names of the entities to exclude from welding (e.g., "table", "floor").
    ignore_links_idx: list[int]
        The rigid-solver (scene-global) link indices to exclude from welding, e.g.
        ``env.entities["table"].get_link("top").idx``; an entity-local ``idx_local`` excludes the wrong link.
    left_gripper_link_name: str
        The name of the left gripper link. (e.g., "left_gripper_pad")
    right_gripper_link_name: str
        The name of the right gripper link. (e.g., "right_gripper_pad")
    grasp_force_threshold: float
        Minimum projected contact force (N) on **each** pad from the candidate link; ``>= 0``.
    """

    left_gripper_link_name: str = ""
    right_gripper_link_name: str = ""
    grasp_force_threshold: float = 1.0

    # The weld only stands in for sim contact friction: it fires *because* the jaws already closed
    # under a separate joint term, which is what holds the object on the real gripper. Nothing for a
    # deployment to send (see ActionTerm.sim_surrogate) — unlike SuctionCupWelding above.
    _sim_surrogate_ = True

    def __init__(self, env: EnvBase, options: ActionTermOptions):
        super().__init__(env=env, options=options)
        self._left_gripper_link_idx: int = -1
        self._right_gripper_link_idx: int = -1
        self._gripper_idx_local: list[int] = [-1, -1]

    def build(self) -> None:
        super().build()
        if not self.grasp_force_threshold >= 0.0:
            # The force test compares |projected force| > threshold, so a negative threshold passes every
            # contact, and NaN passes none: either way the weld no longer depends on a grasp.
            raise ValueError(
                f"{term_subject(self)}: grasp_force_threshold={self.grasp_force_threshold!r} must be >= 0 "
                "(newtons of contact force each pad must carry from the squeezed link). Use 0.0 to weld "
                "on any two-pad contact with that link."
            )

        self._left_gripper_link_idx = self._entity.get_link(self.left_gripper_link_name).idx
        self._right_gripper_link_idx = self._entity.get_link(self.right_gripper_link_name).idx
        self._gripper_idx_local = [
            self._entity.get_link(self.left_gripper_link_name).idx_local,
            self._entity.get_link(self.right_gripper_link_name).idx_local,
        ]

    @property
    def action_dim(self) -> int:
        return 1

    def apply_actions(self) -> None:
        # Only an env whose action is on and that holds no weld yet can weld this step.
        if not (self._processed_action & ~self._welded).any():
            return
        contact_info = self._entity.get_contacts(exclude_self_contact=True)
        link_a = contact_info["link_a"]
        link_b = contact_info["link_b"]
        valid_mask = contact_info["valid_mask"]

        # Candidates: every external link either pad touches. The grasp is checked per candidate below,
        # from that link's contacts alone, so contacts with different links never add up to one grasp.
        pad_a = (link_a == self._left_gripper_link_idx) | (link_a == self._right_gripper_link_idx)
        pad_b = (link_b == self._left_gripper_link_idx) | (link_b == self._right_gripper_link_idx)
        objects_link_idx = torch.unique(torch.cat([link_a[pad_b & valid_mask], link_b[pad_a & valid_mask]]))
        for link_idx in self._weldable(objects_link_idx):
            # Both pads carry force above the threshold (>= 0, checked in build) from this link, which
            # implies both touch it. Re-read _welded each time: an env welds at most one link per step.
            squeezed = parallel_jaw_is_grasping(
                self._entity,
                contact_info,
                self._left_gripper_link_idx,
                self._right_gripper_link_idx,
                self._gripper_idx_local,
                self.grasp_force_threshold,
                object_link_idx=link_idx,
            )
            envs_weld = self._processed_action & ~self._welded & squeezed
            if not envs_weld.any():
                continue

            self._env.rigid_solver.add_weld_constraint(link_idx, self._ee_link_idx, envs_idx=envs_weld)
            self._welded[envs_weld] = True
