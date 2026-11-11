"""Tests for ``eden.utils.chumpy_pickle``: body-model pickles that hold chumpy arrays read without chumpy."""

from __future__ import annotations

import os
import pickle
import subprocess
import sys
import types

import numpy as np
import pytest

from eden.utils.chumpy_pickle import ChumpyFreeUnpickler


class Chumpy:
    """A chumpy object to pickle: its class (``"Ch"`` or ``"Select"``) and its ``__dict__``."""

    def __init__(self, kind: str, **state):
        self.kind, self.state = kind, state


def chumpy_pickle(monkeypatch, objects: dict) -> bytes:
    """Pickle ``objects`` the way chumpy does, without chumpy: each :class:`Chumpy` value becomes an instance of a
    class that pickle records as ``chumpy.ch.Ch`` / ``chumpy.reordering.Select``, its state the given ``__dict__``
    (chumpy carries its graph bookkeeping along, so a ``_cache`` rides with it here too).

    The stand-in modules exist only while pickling; ``sys.modules["chumpy"] = None`` afterwards makes any import of
    chumpy during the read raise.
    """
    modules = {name: types.ModuleType(name) for name in ("chumpy", "chumpy.ch", "chumpy.reordering")}
    classes = {}
    for module, name in (("chumpy.ch", "Ch"), ("chumpy.reordering", "Select")):
        cls = type(name, (), {"__module__": module})
        setattr(modules[module], name, cls)
        classes[name] = cls
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    def build(value):
        if not isinstance(value, Chumpy):
            return value
        obj = classes[value.kind].__new__(classes[value.kind])
        obj.__dict__.update({k: build(v) for k, v in value.state.items()}, _cache={})
        return obj

    data = pickle.dumps({k: build(v) for k, v in objects.items()}, protocol=2)
    monkeypatch.setitem(sys.modules, "chumpy", None)
    for name in ("chumpy.ch", "chumpy.reordering"):
        monkeypatch.delitem(sys.modules, name)
    return data


def _read(data: bytes) -> dict:
    import io

    return ChumpyFreeUnpickler(io.BytesIO(data), encoding="latin1").load()


def test_a_ch_reads_as_the_array_it_holds(monkeypatch):
    shapedirs = np.random.default_rng(0).normal(size=(5, 3, 4))
    data = chumpy_pickle(monkeypatch, {"shapedirs": Chumpy("Ch", x=shapedirs), "f": np.arange(6)})
    read = _read(data)
    np.testing.assert_array_equal(np.asarray(read["shapedirs"]), shapedirs)
    assert np.asarray(read["shapedirs"]).dtype == shapedirs.dtype
    np.testing.assert_array_equal(read["f"], np.arange(6))


def test_a_select_reads_its_sources_flat_indices_in_its_preferred_shape(monkeypatch):
    """MANO and SMPL-H keep the first 10 of their shape components as a Select over the full Ch."""
    full = np.random.default_rng(1).normal(size=(4, 3, 20))
    idxs = np.arange(full.size).reshape(full.shape)[..., :10]
    select = Chumpy("Select", a=Chumpy("Ch", x=full), idxs=idxs, preferred_shape=(4, 3, 10))
    read = _read(chumpy_pickle(monkeypatch, {"shapedirs": select}))
    np.testing.assert_array_equal(np.asarray(read["shapedirs"]), full[..., :10])


def test_the_stand_ins_answer_shape_and_slicing_as_a_third_party_loader_asks(monkeypatch):
    """The way smplx reads them: ``shapedirs.shape[-1]``, then ``shapedirs[:, :, :num_betas]``, before converting."""
    x = np.random.default_rng(2).normal(size=(5, 3, 4))
    ch = _read(chumpy_pickle(monkeypatch, {"shapedirs": Chumpy("Ch", x=x)}))["shapedirs"]
    assert ch.shape == (5, 3, 4)
    np.testing.assert_array_equal(ch[:, :, :2], x[:, :, :2])
    assert np.array(ch, dtype=np.float32).dtype == np.float32


@pytest.mark.parametrize("kind", ["Ch", "Select"])
def test_np_array_and_a_slice_are_the_callers_own(monkeypatch, kind):
    """The ``__array__`` contract numpy 2 sets: a caller writing into ``np.array(stand_in)`` or a slice of it must not
    change what the next read of the stand-in sees.
    """
    x = np.random.default_rng(3).normal(size=(5, 3, 4))
    obj = Chumpy("Ch", x=x)
    if kind == "Select":
        obj = Chumpy("Select", a=obj, idxs=np.arange(x.size).reshape(x.shape), preferred_shape=x.shape)
    stand_in = _read(chumpy_pickle(monkeypatch, {"shapedirs": obj}))["shapedirs"]

    np.array(stand_in)[...] = 0.0
    stand_in[:2][...] = 0.0
    np.testing.assert_array_equal(np.asarray(stand_in), x)
    with pytest.raises(ValueError):
        np.array(stand_in, dtype=np.float32, copy=False)


def test_a_chumpy_class_outside_the_releases_names_itself_and_the_npz_fix(monkeypatch):
    monkeypatch.setitem(sys.modules, "chumpy", types.ModuleType("chumpy"))
    monkeypatch.setitem(sys.modules, "chumpy.ch_ops", ops := types.ModuleType("chumpy.ch_ops"))
    ops.add = type("add", (), {"__module__": "chumpy.ch_ops"})
    data = pickle.dumps({"v_template": ops.add()}, protocol=2)
    with pytest.raises(pickle.UnpicklingError, match=r"chumpy\.ch_ops\.add.*chumpy\.ch\.Ch.*model's \.npz.*does not"):
        _read(data)


@pytest.mark.parametrize("before", ["", "sys.modules['chumpy'] = None"], ids=["fresh", "import-blocked"])
def test_installed_modules_let_a_plain_pickle_load_read_a_chumpy_pickle(monkeypatch, tmp_path, before):
    """The SOMA worker's path: smplx calls ``pickle.load`` itself. Run in a child, since the install is
    process-wide. A ``None`` entry (an import blocked on purpose) is not an imported chumpy, so it is replaced.
    """
    x = np.arange(24.0).reshape(2, 3, 4)
    path = tmp_path / "SMPL_NEUTRAL.pkl"
    path.write_bytes(chumpy_pickle(monkeypatch, {"shapedirs": Chumpy("Ch", x=x)}))
    script = (
        "import pickle, sys, numpy as np\n"
        f"{before}\n"
        "from eden.utils.chumpy_pickle import install_chumpy_pickle_modules\n"
        "install_chumpy_pickle_modules()\n"
        f"d = pickle.load(open({str(path)!r}, 'rb'), encoding='latin1')\n"
        "print(np.asarray(d['shapedirs']).sum(), type(d['shapedirs']).__module__)\n"
    )
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True).stdout.split()
    assert out == [str(x.sum()), "eden.utils.chumpy_pickle"]


# smplx's own read of a real release, through the stand-ins (the SOMA worker's path), against smplx built from the
# arrays Eden reads with ChumpyFreeUnpickler. The releases are licence-gated: point these at a folder holding one.
# The official SMPL-H .pkl carries no hand-PCA basis, which smplx.create reads even with PCA off, so it is not here.
_REAL_RELEASES = [
    ("EDEN_SMPL_DIR", "smpl_neutral", "smpl", {}, "body_pose", 69),
    ("EDEN_MANO_DIR", "mano_right", "mano", {"is_rhand": True, "use_pca": False}, "hand_pose", 45),
]


@pytest.mark.parametrize(("env", "name", "model_type", "kwargs", "pose_key", "pose_dim"), _REAL_RELEASES)
def test_smplx_reads_a_real_release_through_the_installed_stand_ins_as_eden_reads_it(
    tmp_path, env, name, model_type, kwargs, pose_key, pose_dim
):
    smplx = pytest.importorskip("smplx", reason="the reference body-model implementation (the dev dependency group)")
    import torch
    from eden.datasets.utils.body_model import body_model_arrays, body_model_file

    try:
        path = body_model_file(os.environ.get(env) or None, name)
    except (ValueError, FileNotFoundError):
        pytest.skip(f"set {env} to a folder holding the {name} model file")
    if path.suffix != ".pkl":
        pytest.skip(f"{path} is not a .pkl: nothing pickles chumpy objects")
    rng = np.random.default_rng(9)
    inputs = {"betas": rng.normal(size=(1, 10)), pose_key: rng.normal(0.0, 0.3, (1, pose_dim))}
    np.savez(tmp_path / "inputs.npz", **inputs)
    script = (
        "import sys, numpy as np, torch, smplx\n"
        "from eden.utils.chumpy_pickle import install_chumpy_pickle_modules\n"
        "install_chumpy_pickle_modules()\n"
        f"model = smplx.create({str(path)!r}, model_type={model_type!r}, num_betas=10, **{kwargs!r})\n"
        f"inputs = np.load({str(tmp_path / 'inputs.npz')!r})\n"
        "with torch.no_grad():\n"
        "    out = model(**{k: torch.tensor(inputs[k]).float() for k in inputs.files})\n"
        f"np.save({str(tmp_path / 'vertices.npy')!r}, out.vertices.numpy())\n"
    )
    subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True)

    model_cls = {"smpl": smplx.SMPL, "mano": smplx.MANO}[model_type]
    struct = smplx.utils.Struct(**body_model_arrays(path))
    model = model_cls(str(path), data_struct=struct, num_betas=10, **kwargs)
    with torch.no_grad():
        expected = model(**{k: torch.tensor(v).float() for k, v in inputs.items()}).vertices.numpy()
    np.testing.assert_array_equal(np.load(tmp_path / "vertices.npy"), expected)
