"""Read the body-model pickles that hold ``chumpy`` arrays (MANO, SMPL, SMPL-H) without chumpy.

Those releases pickle ``shapedirs`` as a ``chumpy.ch.Ch`` (SMPL) or a ``chumpy.reordering.Select`` over one (MANO,
SMPL-H). chumpy itself is unmaintained, its PyPI release does not build on a current numpy, and a published wheel cannot
depend on the git commit that does, so Eden reads the two classes itself: :class:`ChumpyFreeUnpickler` for Eden's own
loader (:func:`eden.datasets.utils.body_model.body_model_arrays`), and :func:`install_chumpy_pickle_modules` for a
third-party loader that calls a plain ``pickle.load`` (smplx, in the SOMA retargeter's worker process).

numpy-only on purpose: the SOMA worker imports it and must stay free of Genesis.
"""

import pickle
import sys
import types

import numpy as np


class _PickledCh:
    """A ``chumpy.ch.Ch`` read back from a model pickle, without chumpy: the array its ``x`` holds.

    chumpy pickles a ``Ch`` as its ``__dict__``, so restoring that dict is the whole unpickle; the expression-graph
    bookkeeping that comes with it (``_cache``, ``_parents``, ...) is carried along and never read.
    """

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)

    # numpy 2's contract, which np.array implements for an ndarray: copy=True copies, copy=False refuses to, None copies
    # only when the dtype asks it to. A Ch's value is its own x, so np.array(stand_in) must copy for the caller to own it.
    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        return np.array(self._value(), dtype=dtype, copy=copy)

    def _value(self) -> np.ndarray:
        return np.asarray(self.x)

    # A third-party loader handed these (smplx, through install_chumpy_pickle_modules) reads the shape and slices
    # before converting, as it would a chumpy array. A slice is a copy, as chumpy's own is a new expression: a view
    # would let the caller write into x.
    @property
    def shape(self) -> tuple[int, ...]:
        return self._value().shape

    def __getitem__(self, key) -> np.ndarray:
        return self._value()[key].copy()


class _PickledSelect(_PickledCh):
    """A ``chumpy.reordering.Select``: ``a``'s entries at the flat indices ``idxs``, as chumpy's ``compute_r`` reads
    them (MANO and SMPL-H keep their first 10 shape components this way).
    """

    def _value(self) -> np.ndarray:
        selected = np.asarray(self.a).ravel()[np.asarray(self.idxs)]
        shape = getattr(self, "preferred_shape", None)
        return selected if shape is None else selected.reshape(shape)


#: The chumpy classes a body-model pickle references, by ``(module, name)``. These two are every chumpy class in the
#: MANO (v1.2), SMPL (v1.1) and SMPL-H (MANO release) ``.pkl`` files.
_CHUMPY_CLASSES: dict[tuple[str, str], type] = {
    ("chumpy.ch", "Ch"): _PickledCh,
    ("chumpy.reordering", "Select"): _PickledSelect,
}


class ChumpyFreeUnpickler(pickle.Unpickler):
    """Unpickles a body-model ``.pkl`` with :data:`_CHUMPY_CLASSES` standing in for chumpy's, so the file reads
    without ``chumpy`` installed (its PyPI release does not build on a current numpy). Every other class resolves as
    ``pickle.load`` would resolve it.
    """

    def find_class(self, module: str, name: str):
        if module == "chumpy" or module.startswith("chumpy."):
            cls = _CHUMPY_CLASSES.get((module, name))
            if cls is None:
                known = ", ".join(f"{m}.{n}" for m, n in _CHUMPY_CLASSES)
                raise pickle.UnpicklingError(
                    f"This body-model pickle holds a {module}.{name}, a chumpy class Eden does not read (it reads "
                    f"{known}: every chumpy class in the MANO, SMPL and SMPL-H releases). Use the model's .npz "
                    f"instead (put it next to the .pkl), or convert the .pkl to an .npz with the same keys. "
                    f"Installing chumpy does not help: Eden never imports it."
                )
            return cls
        return super().find_class(module, name)


def install_chumpy_pickle_modules() -> None:
    """Make ``chumpy.ch.Ch`` / ``chumpy.reordering.Select`` resolve to Eden's stand-ins, process-wide.

    For a third-party loader that reads a model ``.pkl`` with a plain ``pickle.load`` (smplx, under the SOMA
    retargeter), which a custom unpickler cannot reach. It registers ``chumpy``, ``chumpy.ch`` and
    ``chumpy.reordering`` in ``sys.modules``, shadowing an installed chumpy for the rest of the process, so only a
    process that exists to run such a loader calls it (the SOMA worker). The stand-ins read every MANO, SMPL and
    SMPL-H release bit-identically to chumpy. A chumpy the process has already imported is left in place; a
    ``sys.modules["chumpy"] = None`` (an import blocked on purpose) is not an import, and is replaced.
    """
    if sys.modules.get("chumpy") is not None:
        return
    modules = {name: types.ModuleType(name) for name in ("chumpy", "chumpy.ch", "chumpy.reordering")}
    for (module, name), cls in _CHUMPY_CLASSES.items():
        setattr(modules[module], name, cls)
    modules["chumpy"].ch = modules["chumpy.ch"]
    modules["chumpy"].reordering = modules["chumpy.reordering"]
    sys.modules.update(modules)
