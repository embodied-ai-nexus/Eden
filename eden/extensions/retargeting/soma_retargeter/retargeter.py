"""Eden ``RetargetingBase`` wrapper around the ``eden-soma-retargeter`` library.

Retargets human motion — SMPL / SMPL-H / SMPL-X (fitted via py-soma-x) or
native SOMA (arrays / ``save_soma_npz`` files / SOMA-skeleton BVH) — onto a
floating-base robot (Unitree G1 29-DOF, LimX Oli 31-DOF), returning
Genesis-compatible qpos.

Usage mirrors ``DexRetargeter`` / ``OmniRetargeter``:

.. code-block:: python

    retargeter = SomaRetargeter(
        env=env,
        options=SomaRetargeter.configure(
            entity_name="robot",
            body_model="smplx",
            body_model_path="/path/to/SMPLX_NEUTRAL.npz",
            target_fps=30.0,
        ),
    )
    qpos = retargeter.retarget_motion(poses=..., trans=..., betas=..., source_fps=120.0)
    qpos_batch = retargeter.retarget_motions([clip0, clip1, ...])

**Why a worker subprocess.** Retargeting runs in a persistent Genesis-free worker
process; this class streams clips to it and permutes results into
``entity.dofs_name`` order. The isolation is precautionary — it keeps the Warp
fit's CUDA context and its several GB off the ones Genesis holds — and is an
internal detail: the public API is the same in-process loop as the other
retargeters.
"""

from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch

from eden.extensions.retargeting import RetargetingBase
from eden.extensions.retargeting.soma_retargeter._backend import check_backend_available
from eden.extensions.retargeting.soma_retargeter._protocol import recv_message, send_message
from eden.options.extensions.retargeting import RetargeterOptions

if TYPE_CHECKING:
    from eden.envs.base import EnvBase


def build_dof_permutation(source_joint_names: Sequence[str], entity_dofs_name: Sequence[str]) -> np.ndarray:
    """Permutation ``perm`` such that ``out[:, perm] = src`` maps each source DOF to its entity slot.

    Resolves purely by name (independent of Genesis's internal link/DOF order,
    which changed BFS->DFS in genesis 1.2.0). A trailing ``_dof`` suffix on
    source names is tolerated. Raises ``ValueError`` on a length mismatch or a
    missing joint, listing both name sets.
    """
    entity = list(entity_dofs_name)
    entity_index = {n: i for i, n in enumerate(entity)}
    # Strip the CSV '_dof' suffix only when doing so resolves a name.
    src = [
        n if n in entity_index else (n[:-4] if n.endswith("_dof") and n[:-4] in entity_index else n)
        for n in source_joint_names
    ]
    if len(src) != len(entity):
        raise ValueError(
            f"Retargeter emits {len(src)} DOFs but entity has {len(entity)}. Source: {src}. Entity: {entity}."
        )
    missing = [n for n in src if n not in entity_index]
    if missing:
        raise ValueError(f"Retargeter DOFs not in entity.dofs_name: {missing}. Entity DOFs: {entity}.")
    perm = [entity_index[n] for n in src]
    if len(set(perm)) != len(perm):
        raise ValueError(f"Retargeter DOF names are not a bijection onto entity.dofs_name (duplicates). Source: {src}.")
    return np.array(perm, dtype=np.int64)


class SomaRetargeter(RetargetingBase):
    """SMPL/SMPL-H/SMPL-X/SOMA -> robot retargeter (library-backed).

    Parameters
    ----------
    body_model : {"smpl", "smplh", "smplx", "soma"}
        Source motion representation. The smpl family is fitted to the SOMA rig
        via py-soma-x; ``"soma"`` consumes SOMA motion directly.
    body_model_path : str
        Path to the body-model file (``.npz``/``.pkl``) or a directory holding
        per-gender ``<MODEL>_<GENDER>.{npz,pkl}`` files. Required for the smpl
        family; unused for ``"soma"``.
    gender : {"neutral", "male", "female"}
        Default gender; a per-call ``gender=`` on ``retarget_motion`` overrides it.
    target_fps : float
        FPS every clip is resampled to before retargeting.
    snap_to_ground : bool
        Post-process each clip with one constant vertical offset so the lowest
        point of the robot's collision geometry over the whole motion touches
        the ground plane (z = 0) — lifts clips whose feet sink below the floor,
        lowers floating ones. Sweeps the clip through the entity's forward
        kinematics; the entity's qpos is restored afterwards (velocities zeroed).
    device : str
        Torch/warp device for the worker ("cuda", "cuda:1", "cpu").
    enable_procedural_transforms : bool
        Fit through py-soma-x's expanded twist-joint rig instead of the public
        78-joint SOMA rig. Off by default (the library defaults it *on*) because
        it requires ``SOMA_procedural_transforms.json`` / ``SOMA_template_rig.usda``,
        which the public ``nvidia/soma-x`` asset repo does not ship — turning it on
        without them raises ``FileNotFoundError`` naming the missing file. Both
        modes emit the same 78-joint interface, so the mode never changes what
        callers do; the fitted rotations do differ, since procedural mode refines
        the skin the fit is matched against. Ignored by ``body_model="soma"``
        (no fit).
    robot : str
        Library identifier of the target robot ("unitree_g1", "limx_oli", or any
        robot the library ships a ``configs/<robot>/`` bundle for). Must match
        the entity referenced by ``entity_name``: the library emits that robot's
        joint names, which are permuted into ``entity.dofs_name`` by name. For
        robots whose library config bundles no downloadable model (e.g.
        "limx_oli"), the entity's own resolved model file is forwarded as
        ``robot_model_path``, so both sides use the same asset.
    retargeter_kwargs : dict
        Extra keyword arguments forwarded verbatim to the library
        ``MotionRetargeter`` (e.g. ``fit_chunk``, ``forward_chunk``,
        ``ik_config_overrides``, ``fit_kwargs``). A ``robot_model_path`` here
        overrides the entity-derived one.
    """

    body_model: Literal["smpl", "smplh", "smplx", "soma"] = "smplx"
    body_model_path: str = ""
    gender: Literal["neutral", "male", "female"] = "neutral"
    target_fps: float = 30.0
    snap_to_ground: bool = False
    device: str = "cuda"
    enable_procedural_transforms: bool = False
    robot: str = "unitree_g1"
    retargeter_kwargs: dict = {}

    def __init__(self, env: "EnvBase", options: RetargeterOptions) -> None:
        # Set before anything that can raise, so __del__ -> close() is always safe.
        self._proc: subprocess.Popen | None = None
        self._perm: np.ndarray | None = None
        self._perm_names: tuple[str, ...] | None = None

        super().__init__(env, options)

        if self.entity.has_immobile_base:
            reason = "is welded onto another entity" if self.entity.is_attached else "is fixed-base"
            raise ValueError(
                f"SomaRetargeter requires a floating-base entity (returns qpos with "
                f"[base_pos(3), base_quat(4), joints(N)]); '{self.entity.name}' {reason} and has "
                f"no base DOFs."
            )
        check_backend_available(self.body_model)

        # Built (and validated) before the worker exists, so a bad option costs no process.
        self._start_worker(self._init_message())

    # ------------------------------------------------------------------ worker

    def _init_message(self) -> dict:
        """The worker's init frame — keyword arguments for the library ``MotionRetargeter``."""
        if "enable_procedural_transforms" in self.retargeter_kwargs:
            # A duplicate here would win over the option (retargeter_kwargs splats last),
            # silently ignoring what the caller configured. One home for one setting.
            raise ValueError(
                "enable_procedural_transforms is a SomaRetargeter option; setting it through "
                "retargeter_kwargs would override the option silently. Pass it to "
                "SomaRetargeter.configure() instead."
            )
        kwargs = {
            "body_model": self.body_model,
            "body_model_path": self.body_model_path or None,
            "gender": self.gender,
            "target_fps": self.target_fps,
            "device": self.device,
            "robot": self.robot,
            "enable_procedural_transforms": self.enable_procedural_transforms,
            **self.retargeter_kwargs,
        }
        # Robots without a library-bundled model (everything but unitree_g1) retarget
        # against the entity's own resolved model file, guaranteeing the library's
        # joint names are a bijection onto entity.dofs_name.
        if self.robot != "unitree_g1" and "robot_model_path" not in kwargs:
            kwargs["robot_model_path"] = self._entity_model_path()
        return kwargs

    def _start_worker(self, init: dict) -> None:
        # Fresh process => clean CUDA context. Its stdout carries the binary
        # protocol; stderr is inherited. Prepend our sys.path so the worker
        # resolves the same `eden`/`soma_retargeter` (e.g. worktree checkouts);
        # an empty entry would be read as cwd, so drop those.
        env = os.environ.copy()
        parts = [p for p in sys.path if p]
        if env.get("PYTHONPATH"):
            parts.append(env["PYTHONPATH"])
        env["PYTHONPATH"] = os.pathsep.join(parts)
        self._proc = subprocess.Popen(
            [sys.executable, "-m", "eden.extensions.retargeting.soma_retargeter._worker"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            env=env,
        )
        send_message(self._proc.stdin, init)
        ack = self._read_response()
        if not ack.get("ok"):
            self.close()
            raise RuntimeError(f"soma retargeter worker failed to start:\n{ack.get('trace', ack.get('error'))}")

    def _entity_model_path(self) -> str:
        """The entity's resolved model file (the exact asset Genesis built it from)."""
        file = getattr(getattr(self.entity, "morph", None), "file", None)
        if not isinstance(file, str) or not file:
            raise ValueError(
                f"robot={self.robot!r} needs the target entity's model file, but entity "
                f"{self.entity_name!r} has no file-based morph. Pass robot_model_path via "
                "retargeter_kwargs instead."
            )
        return file

    def _read_response(self) -> dict:
        if self._proc is None:
            raise RuntimeError("SomaRetargeter worker is not running (was it closed?).")
        try:
            msg = recv_message(self._proc.stdout)
        except (EOFError, OSError) as exc:  # worker died mid-frame
            rc = self._worker_died()
            raise RuntimeError(f"soma retargeter worker crashed (code {rc}).") from exc
        if msg is None:  # clean EOF
            rc = self._worker_died()
            raise RuntimeError(f"soma retargeter worker exited unexpectedly (code {rc}).")
        return msg

    def _worker_died(self) -> int | None:
        """Reap the dead worker and clear the handle so later calls fail cleanly."""
        proc, self._proc = self._proc, None
        if proc is None:
            return None
        try:
            return proc.wait(timeout=5)
        except Exception:  # noqa: BLE001
            proc.kill()
            return None

    def _run(self, clips: list[dict]) -> list[dict]:
        if self._proc is None:
            raise RuntimeError("SomaRetargeter worker is not running (was it closed?).")
        try:
            send_message(self._proc.stdin, {"clips": clips})
        except (BrokenPipeError, OSError) as exc:  # worker died before/while reading
            rc = self._worker_died()
            raise RuntimeError(f"soma retargeter worker crashed (code {rc}).") from exc
        resp = self._read_response()
        if not resp.get("ok"):
            raise RuntimeError(f"soma retargeting failed:\n{resp.get('trace', resp.get('error'))}")
        return resp["results"]

    def close(self) -> None:
        """Terminate the worker subprocess (idempotent)."""
        proc, self._proc = getattr(self, "_proc", None), None
        if proc is None:
            return
        try:
            if proc.stdin is not None:
                proc.stdin.close()
            proc.wait(timeout=10)
        except Exception:  # noqa: BLE001 - best-effort teardown
            proc.kill()

    def __del__(self):
        self.close()

    # ------------------------------------------------------------------ public

    @property
    def joint_names(self) -> list[str]:
        """Full DOF name list of the underlying entity (``dofs_name`` order)."""
        return list(self.entity.dofs_name)

    def retarget_motion(
        self,
        poses: np.ndarray | None = None,
        trans: np.ndarray | None = None,
        transl: np.ndarray | None = None,
        betas: np.ndarray | None = None,
        gender: str | None = None,
        source_fps: float | None = None,
        **extra,
    ) -> np.ndarray:
        """Retarget one clip to Genesis qpos ``(T, 7 + N)`` in ``entity.dofs_name`` order.

        Field names follow SOMA-X conventions. For the smpl family pass
        ``poses`` / ``trans`` / ``betas`` / ``source_fps`` (+ optional
        ``gender``); for native SOMA arrays pass ``poses`` / ``transl`` /
        ``source_fps``. Returned qpos is ``[x, y, z, qw, qx, qy, qz, joints...]``,
        Z-up, wxyz quaternion, radians.
        """
        clip: dict = {"source_fps": source_fps, **extra}
        if poses is not None:
            clip["poses"] = poses
        if trans is not None:
            clip["trans"] = trans
        if transl is not None:
            clip["transl"] = transl
        if betas is not None:
            clip["betas"] = betas
        if gender is not None:
            clip["gender"] = gender
        clip = {k: v for k, v in clip.items() if v is not None}
        return self._to_qpos(self._run([clip])[0])

    def retarget_bvh(self, bvh_path) -> np.ndarray:
        """Retarget a SOMA-skeleton BVH file (e.g. BONES-SEED) to qpos."""
        return self._to_qpos(self._run([{"bvh_path": str(bvh_path)}])[0])

    def retarget_motions(self, clips: list[dict]) -> list[np.ndarray]:
        """Retarget many clips in one batched IK solve (one env per clip).

        Each clip is the keyword dict of a single-clip call
        (``{"poses", "trans", "source_fps", ...}`` / ``{"poses", "transl", ...}``
        / ``{"bvh_path": ...}``).
        """
        return [self._to_qpos(r) for r in self._run(clips)]

    # ----------------------------------------------------------------- internal

    def _to_qpos(self, result: dict) -> np.ndarray:
        names = tuple(result["joint_names"])
        if self._perm is None or self._perm_names != names:
            self._perm = build_dof_permutation(names, self.entity.dofs_name)
            self._perm_names = names
        dof = np.empty_like(result["dof_pos"])
        dof[:, self._perm] = result["dof_pos"]
        qpos = np.concatenate([result["root_pos"], result["root_quat"], dof], axis=1).astype(np.float32, copy=False)
        return self._snap_to_ground(qpos) if self.snap_to_ground else qpos

    def _snap_to_ground(self, qpos: np.ndarray) -> np.ndarray:
        """Shift the whole clip vertically so its lowest collision vertex touches z=0.

        The offset is constant over the clip, preserving the motion's dynamics.
        Uses the entity as forward-kinematics scratch: its qpos is restored
        afterwards, with velocities zeroed.
        """
        entity = self.entity
        saved = entity.get_qpos().clone()
        mins = []
        try:
            for frame in torch.as_tensor(qpos, device=entity.device):
                entity.set_qpos(frame)
                mins.append(entity.entity.get_verts()[..., 2].min())
        finally:
            entity.set_qpos(saved)
        qpos[:, 2] -= torch.stack(mins).min().item()
        return qpos
