"""Asset resolution: dependency parsers, the local-first Hub fetch and its dep walk, sample data, and run dirs.

The Hub is the one boundary faked here: ``FakeHub`` replaces ``huggingface_hub.snapshot_download``
with a copy out of a local repo directory, and records the patterns each fetch asked for.
"""

from __future__ import annotations

import datetime
import fnmatch
import os
import pathlib
import re
import shutil

import PIL.Image
import pytest
import trimesh

from eden.utils import assets
from eden.utils.assets import (
    _extract_asset_paths_from_dae,
    _extract_asset_paths_from_mtl,
    _extract_asset_paths_from_obj,
    _extract_asset_paths_from_urdf,
    _extract_asset_paths_from_xml,
    _get_asset_dependencies,
    find_latest_checkpoint,
    model_checkpoints,
    resolve_run_dir,
)
from eden.utils.misc import get_cache_dir

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]


def _write(tmp_path, name: str, content: str) -> str:
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    return str(path)


def _norm(*paths: str) -> set[str]:
    """The parsers return OS-normalized relative paths."""
    return {os.path.normpath(p) for p in paths}


class FakeHub:
    """``snapshot_download`` against a local directory: copies the files matching ``allow_patterns``.

    Defined at module level so it pickles into a spawned DDP worker.
    """

    def __init__(self, repo_dir):
        self.repo_dir = pathlib.Path(repo_dir)
        self.requests: list[list[str]] = []

    def __call__(self, *, repo_type, repo_id, allow_patterns=None, local_dir=None):
        patterns = [allow_patterns] if isinstance(allow_patterns, str) else list(allow_patterns)
        self.requests.append(patterns)
        for path in self.repo_dir.rglob("*"):
            rel = path.relative_to(self.repo_dir).as_posix()
            if path.is_file() and any(fnmatch.fnmatch(rel, p) for p in patterns):
                target = pathlib.Path(local_dir, rel)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(path, target)
        return local_dir

    def list_files(self, repo_id: str | None = None) -> list[str]:
        """``HfApi.list_repo_files``: every file path in the repo."""
        return sorted(p.relative_to(self.repo_dir).as_posix() for p in self.repo_dir.rglob("*") if p.is_file())


def _install_hub(monkeypatch, repo_dir) -> FakeHub:
    """Fake both Hub calls ``eden.utils.assets`` makes: the fetch and the listing its errors quote."""
    fake = FakeHub(repo_dir)
    monkeypatch.setattr(assets, "snapshot_download", fake)
    monkeypatch.setattr(assets, "_list_repo_files", fake.list_files)
    return fake


# --- dependency parsers ------------------------------------------------------------------------

_XML_MESHDIR = """<mujoco model="t"><compiler meshdir="meshes"/><asset>
  <mesh name="m1" file="model1.stl"/><mesh name="m2" file="model2.obj"/><mesh name="m3" file="model3.STL"/>
</asset></mujoco>"""
_XML_TEXTURES = """<mujoco model="t"><asset>
  <texture file="textures/wood.png" type="cube" name="wood"/>
  <texture file="textures/metal.png" type="cube" name="metal"/>
  <texture type="skybox" name="sky"/>
  <mesh name="m" file="mesh.stl"/>
</asset></mujoco>"""
_XML_TEXTUREDIR = """<mujoco model="t"><compiler texturedir="textures"/><asset>
  <texture file="wood.png" type="cube" name="wood"/>
</asset></mujoco>"""
_XML_HFIELD = '<mujoco model="t"><asset><hfield name="h" file="terrain.png" size="50 50 1 0.1"/></asset></mujoco>'
_XML_TRAVERSAL = """<mujoco model="t"><compiler angle="radian" meshdir="../../"/><asset>
  <mesh name="link0" file="arms/franka/assets/link0.stl"/><mesh name="hand" file="grippers/hand/assets/hand.stl"/>
</asset></mujoco>"""


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        # meshdir prefixes every mesh; the extension's case is preserved.
        pytest.param(_XML_MESHDIR, _norm("meshes/model1.stl", "meshes/model2.obj", "meshes/model3.STL"), id="meshdir"),
        # A <texture> without a file attr (skybox) contributes nothing.
        pytest.param(_XML_TEXTURES, _norm("textures/wood.png", "textures/metal.png", "mesh.stl"), id="textures"),
        pytest.param(_XML_TEXTUREDIR, _norm("textures/wood.png"), id="texturedir"),
        pytest.param(_XML_HFIELD, {"terrain.png"}, id="hfield"),
        # A meshdir of ../../ is collapsed into the path, not left dangling or dropped.
        pytest.param(
            _XML_TRAVERSAL, _norm("../../arms/franka/assets/link0.stl", "../../grippers/hand/assets/hand.stl"), id="up"
        ),
        # Unparseable XML is not an error the dep walk should propagate.
        pytest.param("invalid xml content <><", set(), id="malformed"),
    ],
)
def test_xml_dependencies(tmp_path, content, expected):
    assert _extract_asset_paths_from_xml(_write(tmp_path, "model.xml", content)) == expected


def test_urdf_dependencies_strip_package_and_file_schemes(tmp_path):
    content = """<robot name="r">
  <link name="base"><visual><geometry><mesh filename="package://robot/meshes/base.stl"/></geometry></visual>
    <collision><geometry><mesh filename="meshes/base_collision.stl"/></geometry></collision></link>
  <link name="l1"><visual><geometry><mesh filename="file://meshes/link1.obj"/></geometry></visual></link>
</robot>"""
    assert _extract_asset_paths_from_urdf(_write(tmp_path, "robot.urdf", content)) == _norm(
        "robot/meshes/base.stl", "meshes/base_collision.stl", "meshes/link1.obj"
    )


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        pytest.param("mtllib material.mtl\nv 0 0 0\nusemtl default\nf 1 2 3\n", {"material.mtl"}, id="single"),
        pytest.param("mtllib a.mtl\nmtllib subdir/b.mtl\n", _norm("a.mtl", "subdir/b.mtl"), id="multiple"),
        pytest.param("mtllib material.mtl # a comment\n", {"material.mtl"}, id="inline-comment"),
        pytest.param("# no materials\nv 0 0 0\n", set(), id="none"),
    ],
)
def test_obj_dependencies(tmp_path, content, expected):
    assert _extract_asset_paths_from_obj(_write(tmp_path, "mesh.obj", content)) == expected


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        pytest.param(
            "newmtl d\nmap_Kd textures/diffuse.png\nmap_Ks textures/specular.png\nbump textures/normal.png\n",
            _norm("textures/diffuse.png", "textures/specular.png", "textures/normal.png"),
            id="map-and-bump",
        ),
        # Directive options (-bm, -o ...) precede the filename and are skipped over.
        pytest.param(
            "newmtl m\nmap_bump -bm 2.0 normal_map.png\nmap_Kd -o 0.5 0.5 0.0 diffuse.jpg\n",
            {"normal_map.png", "diffuse.jpg"},
            id="directive-options",
        ),
        pytest.param("newmtl d\nmap_Kd diffuse.png # tex\nbump normal.png # bump\n", {"diffuse.png", "normal.png"}),
        pytest.param("newmtl plain\nKa 0 0 0\nKd 0.8 0.8 0.8\n", set(), id="no-textures"),
    ],
)
def test_mtl_dependencies(tmp_path, content, expected):
    assert _extract_asset_paths_from_mtl(_write(tmp_path, "material.mtl", content)) == expected


_DAE_14 = """<?xml version="1.0" encoding="utf-8"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
  <library_images>
    <image id="Image_0_005_png" name="Image_0_005_png"><init_from>Image_0.005.png</init_from></image>
    <image id="bump"><init_from>file://textures/my%20bump.jpg</init_from></image>
  </library_images>
  <library_effects><effect id="e"><profile_COMMON>
    <image id="local"><init_from>local.png</init_from></image>
    <newparam sid="s"><surface type="2D"><init_from>Image_0_005_png</init_from></surface></newparam>
  </profile_COMMON></effect></library_effects>
</COLLADA>"""
_DAE_15 = """<COLLADA xmlns="http://www.collada.org/2008/03/COLLADASchema" version="1.5.0">
  <library_images><image id="i"><init_from><ref>textures/diffuse.png</ref></init_from></image></library_images>
</COLLADA>"""


def _textured_dae(image_ref: str) -> str:
    """One triangle, textured by the image ``image_ref`` names: the smallest COLLADA trimesh textures."""
    return f"""<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
  <library_images><image id="img"><init_from>{image_ref}</init_from></image></library_images>
  <library_effects><effect id="fx"><profile_COMMON>
    <newparam sid="surf"><surface type="2D"><init_from>img</init_from></surface></newparam>
    <newparam sid="samp"><sampler2D><source>surf</source></sampler2D></newparam>
    <technique sid="t"><lambert><diffuse><texture texture="samp" texcoord="uv"/></diffuse></lambert></technique>
  </profile_COMMON></effect></library_effects>
  <library_materials><material id="mat"><instance_effect url="#fx"/></material></library_materials>
  <library_geometries><geometry id="g"><mesh>
    <source id="p"><float_array id="pa" count="9">0 0 0 1 0 0 0 1 0</float_array><technique_common>
      <accessor source="#pa" count="3" stride="3"><param name="X" type="float"/><param name="Y" type="float"/>
      <param name="Z" type="float"/></accessor></technique_common></source>
    <source id="u"><float_array id="ua" count="6">0 0 1 0 0 1</float_array><technique_common>
      <accessor source="#ua" count="3" stride="2"><param name="S" type="float"/><param name="T" type="float"/>
      </accessor></technique_common></source>
    <vertices id="v"><input semantic="POSITION" source="#p"/></vertices>
    <triangles material="m" count="1"><input semantic="VERTEX" source="#v" offset="0"/>
      <input semantic="TEXCOORD" source="#u" offset="0" set="0"/><p>0 1 2</p></triangles>
  </mesh></geometry></library_geometries>
  <library_visual_scenes><visual_scene id="s"><node id="n"><instance_geometry url="#g"><bind_material>
    <technique_common><instance_material symbol="m" target="#mat"/></technique_common>
  </bind_material></instance_geometry></node></visual_scene></library_visual_scenes>
  <scene><instance_visual_scene url="#s"/></scene>
</COLLADA>"""


@pytest.mark.parametrize(
    ("content", "expected"),
    [
        # Each <image>, in <library_images> or effect-local, by its <init_from> text as written: trimesh
        # opens it verbatim, so a file:// URI is neither stripped nor percent-decoded. The effect's
        # <surface><init_from> names the image id, not a file, and is not a dependency.
        pytest.param(_DAE_14, _norm("Image_0.005.png", "file://textures/my%20bump.jpg", "local.png"), id="collada-1.4"),
        # pycollada reads COLLADA 1.4 only: a 1.5 <init_from><ref> never loads, so it is not fetched.
        pytest.param(_DAE_15, set(), id="collada-1.5-ref"),
        pytest.param(
            '<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema"><library_images/></COLLADA>',
            set(),
            id="no-images",
        ),
        pytest.param("invalid xml content <><", set(), id="malformed"),
    ],
)
def test_dae_dependencies(tmp_path, content, expected):
    assert _extract_asset_paths_from_dae(_write(tmp_path, "textured.dae", content)) == expected


@pytest.mark.parametrize(
    ("name", "content", "expected"),
    [
        pytest.param("model.xml", '<mujoco><asset><mesh name="m" file="m.stl"/></asset></mujoco>', {"m.stl"}, id="xml"),
        pytest.param(
            "robot.urdf",
            '<robot name="r"><link name="b"><visual><geometry><mesh filename="b.stl"/></geometry></visual>'
            "</link></robot>",
            {"b.stl"},
            id="urdf",
        ),
        pytest.param("mesh.obj", "mtllib material.mtl\n", {"material.mtl"}, id="obj"),
        pytest.param("material.mtl", "newmtl d\nmap_Kd diffuse.png\n", {"diffuse.png"}, id="mtl"),
        pytest.param("textured.dae", _textured_dae("diffuse.png"), {"diffuse.png"}, id="dae"),
        pytest.param("duck.node", "1899 3 0 0\n", {"duck.ele"}, id="tetgen"),
        # An extension with no parser simply has no dependencies.
        pytest.param("notes.txt", "test content", set(), id="unhandled"),
    ],
)
def test_get_asset_dependencies_dispatches_on_extension(tmp_path, name, content, expected):
    assert _get_asset_dependencies(_write(tmp_path, name, content)) == expected


def test_a_missing_file_has_no_dependencies(tmp_path):
    """A referenced-but-absent file must not abort the recursive dep walk."""
    assert _get_asset_dependencies(str(tmp_path / "does" / "not" / "exist.xml")) == set()


# --- get_asset_path -----------------------------------------------------------------------------


@pytest.fixture
def repo(tmp_path):
    """A three-layer asset: robot.xml -> mesh.obj -> mat.mtl -> tex.png."""
    root = tmp_path / "repo"
    _write(root, "robot/robot.xml", '<mujoco><asset><mesh name="m" file="mesh.obj"/></asset></mujoco>')
    _write(root, "robot/mesh.obj", "mtllib mat.mtl\nv 0 0 0\n")
    _write(root, "robot/mat.mtl", "newmtl d\nmap_Kd tex.png\n")
    _write(root, "robot/tex.png", "png")
    return root


@pytest.fixture
def hub(repo, monkeypatch):
    return _install_hub(monkeypatch, repo)


def test_a_cold_fetch_downloads_the_file_then_one_batch_per_dependency_layer(hub, tmp_path):
    local = tmp_path / "local"
    path = assets.get_asset_path("robot/robot.xml", registry="org", dataset="repo", local_dir=str(local))

    assert path == str(local / "robot" / "robot.xml")
    assert all((local / "robot" / f).is_file() for f in ("robot.xml", "mesh.obj", "mat.mtl", "tex.png"))
    assert hub.requests == [
        ["robot/robot.xml"],
        ["robot/mesh.obj"],
        ["robot/mat.mtl"],
        ["robot/tex.png"],
    ]


def _carrot_repo(root: pathlib.Path, image_ref: str) -> pathlib.Path:
    """A SIMPLER-shaped object: object.urdf -> textured.dae -> the image ``image_ref`` names."""
    _write(
        root,
        "models/carrot/object.urdf",
        '<robot name="c"><link name="b"><visual><geometry><mesh filename="textured.dae"/></geometry></visual></link></robot>',
    )
    _write(root, "models/carrot/textured.dae", _textured_dae(image_ref))
    return root / "models" / "carrot"


def _save_png(path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    PIL.Image.new("RGB", (2, 2), (255, 128, 0)).save(path)


def _is_textured(dae_path: pathlib.Path) -> bool:
    """Whether trimesh — Genesis' COLLADA loader — finds the mesh's texture."""
    scene = trimesh.load(str(dae_path), force="scene", process=False)
    return all(geom.visual.material.baseColorTexture is not None for geom in scene.geometry.values())


@pytest.mark.parametrize(
    ("image_ref", "image_in_repo"),
    [
        pytest.param("Image_0.005.png", "Image_0.005.png", id="beside"),
        pytest.param("textures/diffuse.png", "textures/diffuse.png", id="subdir"),
        # trimesh opens these by basename, beside the COLLADA file; the walk fetches the same file.
        pytest.param("file://textures/diffuse.png", "diffuse.png", id="file-uri"),
        pytest.param("/home/author/diffuse.png", "diffuse.png", id="absolute"),
    ],
)
def test_a_collada_visual_fetches_the_image_trimesh_textures_it_with(monkeypatch, tmp_path, image_ref, image_in_repo):
    """The SIMPLER carrot, spoon, eggplant and table cloth rendered white until the walk read COLLADA."""
    _save_png(_carrot_repo(tmp_path / "repo", image_ref) / image_in_repo)
    hub = _install_hub(monkeypatch, tmp_path / "repo")

    local = tmp_path / "local"
    assets.get_asset_path("models/carrot/object.urdf", registry="org", dataset="repo", local_dir=str(local))

    assert _is_textured(local / "models" / "carrot" / "textured.dae")
    assert not any(pattern.startswith("/") for request in hub.requests for pattern in request), hub.requests


def test_a_ref_the_repo_lacks_warns_once_naming_it_and_is_requested_again(monkeypatch, tmp_path, eden_caplog):
    _carrot_repo(tmp_path / "repo", "gone.png")
    hub = _install_hub(monkeypatch, tmp_path / "repo")
    local = tmp_path / "local"

    assets.get_asset_path("models/carrot/object.urdf", registry="org", dataset="repo", local_dir=str(local))
    assert hub.requests[-1] == ["models/carrot/gone.png"]
    (warning,) = [r.getMessage() for r in eden_caplog.records if "not delivered" in r.getMessage()]
    assert "'gone.png'" in warning and "textured.dae" in warning and "'org/repo'" in warning

    # Nothing remembers the miss, so the file is fetched once it is uploaded (or the Hub is back); the
    # warning is not repeated meanwhile.
    _save_png(tmp_path / "repo" / "models" / "carrot" / "gone.png")
    assets.get_asset_path("models/carrot/object.urdf", registry="org", dataset="repo", local_dir=str(local))
    assert hub.requests[-1] == ["models/carrot/gone.png"]
    assert (local / "models" / "carrot" / "gone.png").is_file()
    assert sum("not delivered" in r.getMessage() for r in eden_caplog.records) == 1


def test_two_objects_naming_one_absolute_texture_each_fetch_their_own(monkeypatch, tmp_path, eden_caplog):
    """An author-machine path names a different file beside each referrer; one's miss must not skip the other."""
    root = tmp_path / "repo"
    for name in ("a", "b"):
        _write(
            root,
            f"models/{name}/object.urdf",
            '<robot name="o"><link name="l"><visual><geometry><mesh filename="textured.dae"/></geometry></visual>'
            "</link></robot>",
        )
        _write(root, f"models/{name}/textured.dae", _textured_dae("/home/author/tex.png"))
    _save_png(root / "models" / "b" / "tex.png")
    _install_hub(monkeypatch, root)
    local = tmp_path / "local"

    assets.get_asset_path("models/a/object.urdf", registry="org", dataset="repo", local_dir=str(local))
    assets.get_asset_path("models/b/object.urdf", registry="org", dataset="repo", local_dir=str(local))

    assert _is_textured(local / "models" / "b" / "textured.dae")
    (warning,) = [r.getMessage() for r in eden_caplog.records if "not delivered" in r.getMessage()]
    assert "models/a/tex.png" in warning and "/home/author" not in warning.split("upload the file")[1]


def test_a_warm_cache_is_resolved_without_touching_the_hub(hub, repo, tmp_path):
    local = tmp_path / "local"
    shutil.copytree(repo, local)

    assert assets.get_asset_path("robot/robot.xml", registry="org", dataset="repo", local_dir=str(local)) == str(
        local / "robot" / "robot.xml"
    )
    assert hub.requests == []


def test_a_missing_dependency_is_fetched_even_when_the_file_itself_is_local(hub, repo, tmp_path):
    local = tmp_path / "local"
    shutil.copytree(repo, local)
    (local / "robot" / "tex.png").unlink()

    assets.get_asset_path("robot/robot.xml", registry="org", dataset="repo", local_dir=str(local))

    assert hub.requests == [["robot/tex.png"]]
    assert (local / "robot" / "tex.png").is_file()


@pytest.fixture
def shared_cache(tmp_path, monkeypatch):
    """A repo, and a local_dir that also caches another dataset (as ``eden/assets`` holds ``ycb/``)."""
    root = tmp_path / "repo"
    _write(root, "objects/apples/013_apple/textured.obj", "v 0 0 0\n")
    _write(root, "models/big_eggplant/object.urdf", '<robot name="r"/>')
    _write(root, "models/eggplant/object.urdf", '<robot name="r"/>')
    local = tmp_path / "local"
    _write(local, "ycb/models/013_apple/textured.obj", "v 1 1 1\n")
    return _install_hub(monkeypatch, root), local


def test_a_repo_path_resolves_cold_then_warm_beside_another_datasets_cache(shared_cache):
    hub, local = shared_cache
    file = "objects/apples/013_apple/textured.obj"

    cold = assets.get_asset_path(file, registry="org", dataset="repo", local_dir=str(local))
    requests = len(hub.requests)
    warm = assets.get_asset_path(file, registry="org", dataset="repo", local_dir=str(local))

    assert cold == warm == str(local / file)
    assert len(hub.requests) == requests, "the warm cache must resolve without the Hub"


@pytest.mark.parametrize(
    ("file", "fix"),
    [
        # ycb/models/013_apple/textured.obj, another dataset's file, also ends in it.
        pytest.param("013_apple/textured.obj", "file='objects/apples/013_apple/textured.obj'", id="suffix"),
        pytest.param("eggplant/object.urdf", "file='models/eggplant/object.urdf'", id="suffix-of-a-substring"),
        pytest.param(
            "object.urdf", "['models/big_eggplant/object.urdf', 'models/eggplant/object.urdf']", id="bare-name"
        ),
    ],
)
def test_a_tail_raises_naming_the_repo_paths_that_end_in_it(shared_cache, file, fix):
    _, local = shared_cache
    with pytest.raises(FileNotFoundError, match=re.escape(fix)):
        assets.get_asset_path(file, registry="org", dataset="repo", local_dir=str(local))


def test_a_path_inside_local_dir_resolves_as_its_repo_path(shared_cache):
    hub, local = shared_cache
    inside = str(local / "models" / "eggplant" / "object.urdf")

    # Not fetched yet: the path below local_dir is the repo path that is fetched.
    assert assets.get_asset_path(inside, registry="org", dataset="repo", local_dir=str(local)) == inside
    assert hub.requests == [["models/eggplant/object.urdf"]]
    # Fetched: it resolves from the cache, without the Hub.
    assert assets.get_asset_path(inside, registry="org", dataset="repo", local_dir=str(local)) == inside
    assert len(hub.requests) == 1


def test_a_cached_path_inside_local_dir_heals_a_missing_dependency(hub, repo, tmp_path):
    """Named by its cached path, the asset still gets its dependency walk, as its repo path would."""
    local = tmp_path / "local"
    shutil.copytree(repo, local)
    (local / "robot" / "tex.png").unlink()

    path = str(local / "robot" / "robot.xml")
    assert assets.get_asset_path(path, registry="org", dataset="repo", local_dir=str(local)) == path

    assert hub.requests == [["robot/tex.png"]]
    assert (local / "robot" / "tex.png").is_file()


def test_an_absolute_path_outside_local_dir_that_does_not_exist_fails_without_the_hub(hub, tmp_path):
    typo = str(tmp_path / "elsewhere" / "robot.xml")

    with pytest.raises(FileNotFoundError, match=re.escape(f"No such local file: {typo!r}")):
        assets.get_asset_path(typo, registry="org", dataset="repo", local_dir=str(tmp_path / "local"))
    assert hub.requests == []


def test_an_unreachable_hub_is_named_as_the_cause_not_the_path(hub, monkeypatch, tmp_path):
    """Offline, ``snapshot_download`` hands back local_dir without raising; the error must not blame ``file``."""

    def unreachable(repo_id):
        raise ConnectionError("Failed to resolve 'huggingface.co'\n(details)")

    monkeypatch.setattr(assets, "_list_repo_files", unreachable)
    with pytest.raises(FileNotFoundError) as raised:
        assets.get_asset_path("robot/absent.xml", registry="org", dataset="repo", local_dir=str(tmp_path / "local"))

    message = str(raised.value)
    assert "cannot be reached (ConnectionError: Failed to resolve 'huggingface.co')" in message
    assert "do not rewrite `file`" in message and "\n" not in message


def test_a_repo_file_the_fetch_did_not_deliver_is_not_called_missing_from_the_repo(repo, monkeypatch, tmp_path):
    hub = _install_hub(monkeypatch, repo)
    monkeypatch.setattr(assets, "snapshot_download", lambda **kwargs: kwargs["local_dir"])  # delivers nothing

    with pytest.raises(FileNotFoundError, match=r"is in the Hub repo 'org/repo', but the fetch did not deliver it"):
        assets.get_asset_path("robot/robot.xml", registry="org", dataset="repo", local_dir=str(tmp_path / "local"))
    assert hub.requests == []


def test_a_name_with_glob_characters_is_fetched_literally(monkeypatch, tmp_path):
    root = tmp_path / "repo"
    _write(root, "textures/wood[1].png", "literal")
    _write(root, "textures/wood1.png", "what the unescaped pattern would match")
    _install_hub(monkeypatch, root)
    local = tmp_path / "local"

    path = assets.get_asset_path("textures/wood[1].png", registry="org", dataset="repo", local_dir=str(local))

    assert pathlib.Path(path).read_text() == "literal"
    assert not (local / "textures" / "wood1.png").exists()


def test_a_dependency_with_glob_characters_is_fetched_literally(monkeypatch, tmp_path):
    root = tmp_path / "repo"
    _write(root, "m/mesh.obj", "mtllib mat.mtl\n")
    _write(root, "m/mat.mtl", "newmtl x\nmap_Kd wood[1].png\n")
    _write(root, "m/wood[1].png", "literal")
    _write(root, "m/wood1.png", "what the unescaped pattern would match")
    _install_hub(monkeypatch, root)
    local = tmp_path / "local"

    assets.get_asset_path("m/mesh.obj", registry="org", dataset="repo", local_dir=str(local))

    assert (local / "m" / "wood[1].png").read_text() == "literal"
    assert not (local / "m" / "wood1.png").exists()


def test_an_asset_the_repo_does_not_have_raises_naming_it(hub, tmp_path):
    with pytest.raises(FileNotFoundError, match="nope.xml"):
        assets.get_asset_path("nope.xml", registry="org", dataset="repo", local_dir=str(tmp_path / "local"))


def _fetch_on_one_rank(rank, world_size, init_file, repo_dir, local_dirs, out_dir):
    """Resolve the asset on this rank inside a live 2-rank gloo group with a 20 s collective timeout."""
    import torch.distributed as dist

    os.environ.update(RANK=str(rank), WORLD_SIZE=str(world_size), LOCAL_RANK=str(rank))
    dist.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=20),
    )
    try:
        from eden.utils import assets as rank_assets

        rank_assets.snapshot_download = FakeHub(repo_dir)
        path = rank_assets.get_asset_path("robot/robot.xml", registry="org", dataset="repo", local_dir=local_dirs[rank])
        pathlib.Path(out_dir, f"rank_{rank}").write_text(path)
    finally:
        dist.destroy_process_group()


def test_the_hub_fetch_is_not_a_collective(repo, tmp_path):
    """A rank with a warm cache never reaches the fetch, so a barrier there would hang the cold rank.

    Rank 0 is warm, rank 1 is cold. With any collective inside the fetch, rank 1 times out.
    Rank caches diverge whenever ``$HOME`` is node-local or a scratch cache was partly evicted.
    """
    import torch.multiprocessing as mp

    warm, cold, out = tmp_path / "warm", tmp_path / "cold", tmp_path / "out"
    shutil.copytree(repo, warm)
    out.mkdir()
    mp.spawn(
        _fetch_on_one_rank,
        args=(2, str(tmp_path / "ddp_init"), str(repo), [str(warm), str(cold)], str(out)),
        nprocs=2,
        join=True,
    )
    assert (out / "rank_0").read_text() == str(warm / "robot" / "robot.xml")
    assert (out / "rank_1").read_text() == str(cold / "robot" / "robot.xml")


# --- the download root -------------------------------------------------------------------------


def test_eden_assets_dir_moves_the_default_download(hub, user_assets_dir, monkeypatch, tmp_path):
    monkeypatch.setenv("EDEN_ASSETS_DIR", str(tmp_path / "shared"))

    path = assets.get_asset_path("robot/robot.xml", registry="org", dataset="repo")

    assert path == str(tmp_path / "shared" / "robot" / "robot.xml")
    assert (tmp_path / "shared" / "robot" / "tex.png").is_file()
    assert not user_assets_dir.exists()


def test_without_the_env_var_the_default_download_goes_to_the_user_cache(hub, user_assets_dir):
    path = assets.get_asset_path("robot/robot.xml", registry="org", dataset="repo")

    assert path == str(user_assets_dir / "robot" / "robot.xml")


@pytest.fixture
def read_only():
    """Make a directory tree read-only (``read_only(path)``); restored after the test so ``tmp_path`` cleans up."""
    made: list[pathlib.Path] = []

    def make(directory: pathlib.Path) -> pathlib.Path:
        directory.mkdir(parents=True, exist_ok=True)
        tree = [directory, *(p for p in directory.rglob("*") if p.is_dir())]
        for path in tree:
            path.chmod(0o555)
        made.extend(tree)
        if os.access(directory, os.W_OK):
            pytest.skip("running as root: permission bits do not stop writes")
        return directory

    yield make
    for path in made:
        path.chmod(0o755)


def test_an_unwritable_download_root_raises_naming_it_and_the_env_var(
    hub, user_assets_dir, read_only, monkeypatch, tmp_path
):
    root = read_only(tmp_path / "read_only")
    monkeypatch.setenv("EDEN_ASSETS_DIR", str(root))

    with pytest.raises(PermissionError) as raised:
        assets.get_asset_path("robot/robot.xml", registry="org", dataset="repo")

    message = str(raised.value)
    assert message.startswith(f"Cannot download assets into {root}: writing {root / 'robot'} failed")
    assert "set by EDEN_ASSETS_DIR" in message and "export EDEN_ASSETS_DIR=/path/you/can/write" in message
    assert isinstance(raised.value.__cause__, PermissionError)


def test_a_root_that_cannot_be_created_raises_naming_it(hub, user_assets_dir, read_only, monkeypatch, tmp_path):
    root = read_only(tmp_path / "read_only") / "assets"
    monkeypatch.setenv("EDEN_ASSETS_DIR", str(root))

    with pytest.raises(PermissionError, match=rf"writing {re.escape(str(root))} failed \(PermissionError: Permission"):
        assets.get_asset_path("robot/robot.xml", registry="org", dataset="repo")


def test_an_unwritable_explicit_local_dir_says_the_env_var_does_not_move_it(hub, user_assets_dir, read_only, tmp_path):
    local_dir = read_only(tmp_path / "read_only")

    with pytest.raises(PermissionError) as raised:
        assets.get_asset_path("robot/robot.xml", registry="org", dataset="repo", local_dir=str(local_dir))

    assert "explicit local_dir" in str(raised.value) and "EDEN_ASSETS_DIR does not move it" in str(raised.value)


def test_an_unwritable_explicit_local_dir_is_still_named_when_the_env_var_is_relative(
    hub, user_assets_dir, read_only, monkeypatch, tmp_path
):
    """Building the message reads the root, which a relative EDEN_ASSETS_DIR refuses: both problems are reported."""
    local_dir = read_only(tmp_path / "read_only")
    monkeypatch.setenv("EDEN_ASSETS_DIR", "relative")

    with pytest.raises(PermissionError) as raised:
        assets.get_asset_path("robot/robot.xml", registry="org", dataset="repo", local_dir=str(local_dir))

    message = str(raised.value)
    assert message.startswith(f"Cannot download assets into {local_dir}: writing")
    assert "Pass a local_dir you can write" in message and "EDEN_ASSETS_DIR='relative' is a relative path" in message


def test_a_read_only_prefetched_cache_with_a_ref_the_repo_lacks_still_resolves(
    hub, repo, user_assets_dir, read_only, monkeypatch, tmp_path, eden_caplog
):
    """A shared cluster cache is read-only. A dangling ref is fetched on every resolve and writes nothing."""
    cache = tmp_path / "shared"
    shutil.copytree(repo, cache)
    (cache / "robot" / "mat.mtl").write_text("newmtl d\nmap_Kd tex.png\nmap_Ks gone.png\n")
    read_only(cache)
    monkeypatch.setenv("EDEN_ASSETS_DIR", str(cache))

    assert assets.get_asset_path("robot/robot.xml", registry="org", dataset="repo") == str(
        cache / "robot" / "robot.xml"
    )
    assert hub.requests == [["robot/gone.png"]]
    assert any("'gone.png'" in r.getMessage() for r in eden_caplog.records)


# --- sample data -------------------------------------------------------------------------------


def test_sample_data_is_mirrored_into_the_user_cache_not_the_package(repo, monkeypatch, tmp_path):
    """Tasks write beside their clip (contrack caches a mesh there), so it must be writable and stable."""
    import eden

    _write(repo, "contrack/clip.h5", "clip")
    fake = _install_hub(monkeypatch, repo)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "xdg"))

    path = pathlib.Path(assets.sample_data_path("contrack", "clip.h5"))

    assert path == pathlib.Path(get_cache_dir("eden_example"), "contrack", "clip.h5")
    assert path.is_relative_to(tmp_path / "xdg")
    assert not path.is_relative_to(pathlib.Path(eden.__file__).resolve().parent)
    # One named file out of the task's own directory, never the whole shared repo.
    assert fake.requests == [["contrack/clip.h5"]]


def test_each_sample_data_directory_is_a_registered_task_read_by_one_file():
    """The shared repo keeps one directory per task, named for the task that reads it.

    A lint over the literal ``sample_data_path("<task>", ...)`` call sites under ``eden/``.
    """
    from eden.tasks import TASK_REGISTRY

    call = re.compile(r"""sample_data_path\(\s*["']([^"']+)["']""")
    owner: dict[str, str] = {}
    for path in sorted((_REPO_ROOT / "eden").rglob("*.py")):
        for task in call.findall(path.read_text()):
            file = str(path.relative_to(_REPO_ROOT))
            assert owner.setdefault(task, file) == file, f"{file} and {owner[task]} both read {task!r}"
    assert owner, "no sample_data_path call sites found"
    unregistered = sorted(set(owner) - set(TASK_REGISTRY.list_tasks()))
    assert not unregistered, f"sample data directories that are not registered task names: {unregistered}"


def _default_clips() -> list[tuple[str, str]]:
    from eden.tasks.beyondmimic.dataset import DEFAULT_CLIP
    from eden.tasks.contrack.config import _DEFAULT_CLIP

    return [("beyondmimic", DEFAULT_CLIP), ("contrack", _DEFAULT_CLIP)]


@pytest.mark.parametrize("task, filename", _default_clips())
def test_the_published_default_clip_also_ships_in_the_examples(task, filename):
    """Examples read the checkout copy while task defaults resolve the Hub one; the names must agree."""
    shipped = _REPO_ROOT / "examples/projects/mimic"
    assert list(shipped.rglob(filename)), f"{task}'s default clip {filename!r} no longer ships under {shipped}"


# --- resolve_run_dir: what --resume names ---------------------------------------------------------


def _runs(tmp_path, *names) -> str:
    root = tmp_path / "logs" / "reacher"
    root.mkdir(parents=True)
    for name in names:
        (root / name).mkdir()
    return str(root)


def test_latest_is_the_most_recently_modified_not_the_last_by_name(tmp_path):
    """``%y%b%d`` names (Aug, Dec, ...) do not sort chronologically as strings."""
    root = _runs(tmp_path, "26Aug15T120000-aaaa1111", "26Dec15T130000-bbbb2222")
    os.utime(os.path.join(root, "26Aug15T120000-aaaa1111"), (1000, 1000))
    os.utime(os.path.join(root, "26Dec15T130000-bbbb2222"), (2000, 2000))
    assert os.path.basename(resolve_run_dir(root, "latest")) == "26Dec15T130000-bbbb2222"

    os.utime(os.path.join(root, "26Aug15T120000-aaaa1111"), (3000, 3000))
    assert os.path.basename(resolve_run_dir(root, "latest")) == "26Aug15T120000-aaaa1111"


@pytest.mark.parametrize(
    ("ref", "expected"), [("bbbb2222", "26Aug15T120001-arm_b-bbbb2222"), ("arm_a", "26Aug15T120000-arm_a-aaaa1111")]
)
def test_a_run_id_or_a_unique_run_name_substring_picks_one_run(tmp_path, ref, expected):
    root = _runs(tmp_path, "26Aug15T120000-arm_a-aaaa1111", "26Aug15T120001-arm_b-bbbb2222")

    assert os.path.basename(resolve_run_dir(root, ref)) == expected


def test_an_ambiguous_or_unmatched_reference_raises(tmp_path):
    root = _runs(tmp_path, "26Aug15T120000-arm_a-aaaa1111", "26Aug15T120001-arm_b-bbbb2222")

    with pytest.raises(ValueError, match="ambiguous"):
        resolve_run_dir(root, "arm_")
    with pytest.raises(FileNotFoundError, match="matched no run"):
        resolve_run_dir(root, "zzzz9999")


def test_a_task_never_trained_is_told_to_train_first(tmp_path):
    with pytest.raises(FileNotFoundError, match="train the task first"):
        resolve_run_dir(str(tmp_path / "logs" / "nope"), "latest")
    with pytest.raises(FileNotFoundError, match="train the task first"):
        resolve_run_dir(_runs(tmp_path), "latest")


def test_model_checkpoints_lists_a_runs_checkpoints_by_iteration_not_by_name(tmp_path):
    for name in ("model_10.pt", "model_9.pt", "model_100.pt", "model_best.pt", "model_9.pt.tmp", "config.json"):
        (tmp_path / name).write_bytes(b"")

    assert model_checkpoints(str(tmp_path)) == [(9, "model_9.pt"), (10, "model_10.pt"), (100, "model_100.pt")]
    assert find_latest_checkpoint(str(tmp_path)) == str(tmp_path / "model_100.pt")
