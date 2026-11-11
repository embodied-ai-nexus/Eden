"""Base classes for policy extensions.

``ClientMixin`` (server-backed policies) lives in :mod:`eden.extensions.policies.client`.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Any, ClassVar
from collections.abc import Mapping
from types import MappingProxyType
from abc import abstractmethod
from hashlib import md5

import torch

from eden.utils.common import ConfigurableMixin
from eden.options.extensions.policies import PolicyOptions
from eden.utils.assets import get_asset_path
from eden.utils.misc import resolve_local_dir, resolve_named_entity
from eden.utils.registry import Registry

if TYPE_CHECKING:
    from eden.envs.base import EnvBase
    from eden.options.managers.actions import ActionTermOptions
    from eden.options.managers.commands import CommandTermOptions
    from eden.options.managers.observations import ObservationGroupOptions
    from eden.entities.base import Entity
    from eden.options.entities import EntityOptionsLike, RobotOptions


POLICY_REGISTRY = Registry("POLICY")

__all__ = ["POLICY_REGISTRY", "PolicyBase"]


class PolicyBase(ConfigurableMixin[PolicyOptions]):
    """
    Base class for policies.

    Any object that takes observations and returns actions are defined as a policy.

    The ``*_options`` classmethods below describe the world the policy was trained in, and a
    task-less run (``eden inference|deploy --policy``) calls them for you. In
    :meth:`action_options`, :meth:`observation_options` and :meth:`scene_entity_options`, a
    parameter named after one of the policy's configurable attributes is *filled from it*
    (``hand_name`` gets ``self.hand_name``), so a name is declared once; ``entity_name`` is
    the caller's (``--entity_name`` wins) and anything else keeps its own default. Parameters
    there therefore share a namespace with the class attributes — a knob that must keep its
    literal default needs a name no attribute uses. :meth:`command_options` is filled by the
    CLI instead (``entity_name``, ``motion_file``, ``--clip_name``), and
    :meth:`robot_options` / :meth:`env_options` take no arguments at all.

    :attr:`reads_env_observations` says which side of the env boundary the policy's
    ``step()`` sits on, and a deploy is arranged around the answer.

    Where a policy's files come from (``registry``, ``dataset``, ``local_dir``, ``assets_subdir``) is fixed by
    the class, not configured per instance: a lookup that runs before any instance exists (a clip catalogue in
    :meth:`command_options`, a file an observation term loads) must read the same place ``__init__`` does. To
    fetch from a mirror (an offline cluster, a shared read-only cache), set ``EDEN_ASSETS_DIR``, or set
    ``local_dir`` on the class before anything reads it: ``MyPolicy.local_dir = "/data/mirror"``, as
    :meth:`use_local_checkpoint` redirects a checkpoint.

    Parameters
    ----------
    checkpoint_path: str | None
        The path to the checkpoint file.
    """

    #: Whether ``step(obs)`` consumes the observation dict the env's managers produce, rather
    #: than reading robot state off the entity itself.
    #:
    #: It decides how ``eden deploy`` arranges the run, so it is a contract, not a hint. A
    #: policy that reads the dict **owns the deploy env**: the env is built from *its*
    #: observation groups and action terms, and its action is routed through the env's action
    #: manager, which is what keeps env-computed observations (``last_action`` above all) in
    #: step with what the robot was actually commanded. Only one such policy can be loaded at
    #: a time, and not alongside a task, since one env publishes one observation layout.
    #:
    #: Policies that read raw state instead (Holosoma, SONIC, …) leave this False and ride
    #: along as extra slots on whatever env the run has, converting their own actions.
    reads_env_observations: bool = False

    checkpoint_path: str | None = None
    md5_hash: str | None = None
    # Where the policy's files come from: class constants (see the class docstring), never saved in a config.
    registry: ClassVar[str | None] = "Kashu7100"
    dataset: ClassVar[str | None] = "eden_policies"
    # Where they are cached; None means ``assets_subdir`` under the asset root (see resolve_local_dir).
    local_dir: ClassVar[str | None] = None
    # The subdirectory of the asset root that an unset ``local_dir`` means.
    assets_subdir: ClassVar[str] = "policies"
    _non_parameter_names: ClassVar[frozenset[str]] = frozenset({"registry", "dataset", "local_dir", "assets_subdir"})
    _retired_params: ClassVar[Mapping[str, str]] = MappingProxyType(
        dict.fromkeys(
            ("registry", "dataset", "local_dir"),
            "where a policy's files come from is fixed by its class, so a lookup made before the policy exists "
            "reads the same place. Drop the option; to fetch from a mirror, export EDEN_ASSETS_DIR=/data/mirror "
            "or set it on the class (MyPolicy.local_dir = '/data/mirror') before building anything",
        )
    )

    @classmethod
    def resolve_local_dir(cls) -> str:
        """Where this policy's files are cached: ``local_dir``, else ``assets_subdir`` under the asset root.

        Resolved on each call, so ``EDEN_ASSETS_DIR`` moves it. To fetch a file, call :meth:`fetch_asset`.
        """
        return resolve_local_dir(cls.local_dir, cls.assets_subdir)

    @classmethod
    def fetch_asset(cls, file: str) -> str:
        """The local path of ``file`` from this policy's Hub dataset, downloaded if missing.

        Fetches with the class's ``registry``, ``dataset`` and :meth:`resolve_local_dir`, so it gives the same
        path on the class (a lookup before any env exists) as on an instance. Call this rather than passing
        ``local_dir`` (``None`` by default) to ``get_asset_path`` yourself. Unlike ``EntityOptions.fetch_asset``,
        ``file`` is required: a policy has no single file of its own.
        """
        return get_asset_path(file, registry=cls.registry, dataset=cls.dataset, local_dir=cls.resolve_local_dir())

    def __init__(self, env: EnvBase, options: PolicyOptions):
        super().__init__(options)
        self.env = env

        if self.checkpoint_path is not None:
            self.checkpoint_path = self.fetch_asset(self.checkpoint_path)
        self._verify_md5(self.checkpoint_path, self.md5_hash)

    @classmethod
    def use_local_checkpoint(cls, path: str, *, checkpoint_attr: str = "checkpoint_path") -> None:
        """Point a checkpoint at a local file and drop the hash pinned to the released one.

        The two go together: ``md5_hash`` pins the *released* weights, so setting the path
        without clearing it makes ``_verify_md5`` reject the very file the caller asked
        for. Use this for every ``--ckpt-path``-style flag rather than assigning the path
        by hand.

        Parameters
        ----------
        path: str
            Local checkpoint to load instead of the released one. Not hashed: an
            unreleased file has no pinned digest to check it against.
        checkpoint_attr: str
            Which checkpoint to redirect, for policies that load more than one (sonic's
            ``decoder_checkpoint_path``, omnixtreme's ``residual_``/``fk_``). The hash
            attribute is derived from it by the naming convention every policy follows —
            ``<prefix>checkpoint_path`` is pinned by ``<prefix>md5_hash`` — and both are
            required to exist, so a policy that breaks the convention fails here rather
            than silently keeping a stale pin.
        """
        if not checkpoint_attr.endswith("checkpoint_path"):
            raise ValueError(
                f"checkpoint_attr must end with 'checkpoint_path' so its md5 attribute can be derived; "
                f"got {checkpoint_attr!r} on {cls.__name__}."
            )
        hash_attr = f"{checkpoint_attr.removesuffix('checkpoint_path')}md5_hash"
        for attr in (checkpoint_attr, hash_attr):
            if not hasattr(cls, attr):
                raise AttributeError(
                    f"{cls.__name__} has no `{attr}`. `use_local_checkpoint` redirects "
                    f"`{checkpoint_attr}` and clears `{hash_attr}` together; both must exist."
                )
        setattr(cls, checkpoint_attr, path)
        setattr(cls, hash_attr, None)

    def _verify_md5(self, path: str, expected: str | None) -> None:
        if expected is None:
            return
        with open(path, "rb") as f:
            actual = md5(f.read()).hexdigest()
        if actual != expected:
            raise ValueError(f"MD5 hash mismatch for {path}: expected {expected}, got {actual}")

    def require_entity(self, name: str) -> Entity:
        """Resolve a scene entity the policy reads, failing loud when the scene has no such entity.

        Call this in ``__init__`` for every entity ``step()`` touches, so a scene missing one
        raises at construction rather than deep in the first step. In ``step()`` read
        ``self.env.entities[name]`` instead of caching the handle: deployment hot-swaps it.
        Resolves against the mapping, like a term's ``resolve_entity``.
        """
        return resolve_named_entity(
            self.env.entities,
            name,
            hint=(
                f"A task-less run builds the policy's own workspace from "
                f"{type(self).__name__}.scene_entity_options(); with --task, the task's scene "
                f"must declare '{name}'."
            ),
        )

    def __call__(self, obs: dict[str, torch.Tensor | dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        return self.step(obs)

    @classmethod
    def robot_options(cls) -> RobotOptions | None:
        """Return a robot configured for this policy, or ``None`` for the generic robot.

        Frozen policies are usually trained against a specific default joint
        pose and PD-gain set (e.g. OpenHomie's). Task-less rollout/deploy builds
        the scene from this hook so the robot spawns in the pose and gains the
        policy expects; returning ``None`` falls back to the caller-chosen
        ``eden.robots`` class with its stock defaults.
        """
        return None

    @classmethod
    def scene_entity_options(cls) -> dict[str, EntityOptionsLike]:
        """Return the non-robot scene entities the policy reads, keyed by scene entity name (optional).

        A policy that only reads its own robot needs none — the default. One whose observation
        is defined against something else (SimToolReal's tool and table, HUSKY's skateboard)
        cannot run on a robot-only scene, so a task-less run builds these alongside
        :meth:`robot_options`; with ``--task`` the task's scene must declare them, which
        :meth:`require_entity` checks at construction. A ``plane`` entity replaces the flat
        ground (e.g. with terrain) rather than adding one.
        """
        return {}

    @classmethod
    def env_options(cls) -> dict[str, Any]:
        """Set up required environment parameters for the policy, if any (optional)."""
        return {}

    @classmethod
    @abstractmethod
    def action_options(cls) -> dict[str, ActionTermOptions]:
        """Set up required action group options for the policy (required)."""
        raise NotImplementedError

    @classmethod
    def command_options(cls) -> dict[str, CommandTermOptions]:
        """Set up required command group options for the policy, if any (optional)."""
        return {}

    @classmethod
    @abstractmethod
    def observation_options(cls) -> dict[str, ObservationGroupOptions]:
        """Set up required observation group options for the policy (required)."""
        raise NotImplementedError

    @abstractmethod
    def step(self, obs: dict[str, torch.Tensor | dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Get action from policy."""
        raise NotImplementedError

    def reset(self, envs_idx: (slice | torch.Tensor) | None = None) -> None:
        """Reset per-env policy state at episode boundaries.

        Override this in subclasses that hold per-env state across steps —
        e.g. heading offsets cached on the first step, last-action histories
        fed back as observations, attention KV caches, action-chunk queues.
        The default is a no-op, which is right for stateless policies.

        When used as a teacher / motion generator in long-horizon rollouts the
        owner is expected to call ``policy.reset(envs_idx)`` whenever the
        underlying env resets the corresponding envs.

        Parameters
        ----------
        envs_idx : slice | torch.Tensor | None, optional
            Indices of environments to reset. ``None`` means all environments.
        """
        return None
