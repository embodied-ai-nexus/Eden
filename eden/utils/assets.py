"""Asset download (local-first, lock-serialized) and dependency extraction from model files."""

import errno
import os
import glob
import re
import xml.etree.ElementTree as ET

import yaml
import json
from huggingface_hub import HfApi, snapshot_download

from eden.utils.misc import ASSETS_DIR_ENV_VAR, get_assets_dir, get_cache_dir, resolve_local_dir
from eden.utils.warning import warn_once


def _is_within(path: str, root: str) -> bool:
    path, root = os.path.abspath(path), os.path.abspath(root)
    return path == root or path.startswith(root + os.sep)


#: The ``OSError`` errnos that mean "this directory cannot be written", as opposed to a full disk or a bad path.
_NOT_WRITABLE_ERRNOS = frozenset({errno.EACCES, errno.EPERM, errno.EROFS})


def _not_writable_message(local_dir: str, error: OSError) -> str:
    """Why a download into ``local_dir`` failed to write, and which setting moves it."""
    failure = f"Cannot download assets into {local_dir}: writing {error.filename} failed ({type(error).__name__}: {error.strerror})."
    try:
        root = get_assets_dir()
    except ValueError as root_error:  # A relative EDEN_ASSETS_DIR, so local_dir was an explicit one.
        return f"{failure} Pass a local_dir you can write. Separately: {root_error}"
    reason = (
        f"set by {ASSETS_DIR_ENV_VAR}"
        if os.environ.get(ASSETS_DIR_ENV_VAR)
        else f"the per-user default, chosen because {ASSETS_DIR_ENV_VAR} is unset"
    )
    if _is_within(local_dir, root):
        fix = (
            f"It is under Eden's asset download root {root} ({reason}). Move the root to a directory you can "
            f"write: export {ASSETS_DIR_ENV_VAR}=/path/you/can/write"
        )
    else:
        fix = (
            f"It is an explicit local_dir, outside Eden's asset download root {root} ({reason}), so "
            f"{ASSETS_DIR_ENV_VAR} does not move it (a config saved on another machine carries that machine's "
            f"paths). Pass a local_dir you can write, or leave local_dir unset to download under {root}"
        )
    return f"{failure} {fix}."


def _snapshot_download_locked(**kwargs) -> str:
    """The one place Eden fetches from the Hub. Deliberately *not* a collective.

    Concurrent DDP ranks are safe here because ``huggingface_hub`` takes a per-file lock
    under ``.cache/huggingface/download/``: one rank downloads, the rest block on the lock
    and then read the finished file.

    A rank barrier would be wrong, not merely redundant. Callers reach this *conditionally* —
    :func:`get_asset_path` returns before it whenever the file is already local — so a rank
    whose cache is warm never arrives, and a collective would hang every rank whose cache is
    cold. That is reachable whenever rank caches can diverge: multi-node with node-local
    ``$HOME``, a partially-evicted scratch cache, a job resumed on new hardware.

    An unwritable ``local_dir`` is reported when the fetch first writes to it, naming the path and
    ``EDEN_ASSETS_DIR``. It is not checked up front: a read-only, pre-fetched cache must keep working,
    and a fetch for a file the repo lacks (a dangling texture ref the dependency walk asks for on every
    resolve) writes nothing. Without a ``local_dir`` the fetch goes to the Hugging Face cache, which
    ``HF_HOME`` places.
    """
    try:
        return snapshot_download(**kwargs)
    except OSError as error:
        local_dir, written = kwargs.get("local_dir"), error.filename
        if (
            error.errno in _NOT_WRITABLE_ERRNOS
            and isinstance(written, str)
            and local_dir
            and _is_within(written, local_dir)
        ):
            raise PermissionError(_not_writable_message(local_dir, error)) from error
        raise


def _extract_asset_paths_from_xml(xml_path: str) -> set[str]:
    """Extract file dependencies (meshes, textures, hfields) from a MuJoCo XML file.

    Resolves paths using the ``meshdir`` and ``texturedir`` compiler directives,
    and normalizes ``../`` traversals so the returned paths are clean relative paths.

    Parameters
    ----------
    xml_path : str
        Path to the XML file.

    Returns
    -------
    set[str]
        Asset file paths relative to the XML file directory.
    """
    asset_paths: set[str] = set()

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Get directory overrides from <compiler> element
        compiler = root.find("compiler")
        meshdir = compiler.get("meshdir", "") if compiler is not None else ""
        texturedir = compiler.get("texturedir", "") if compiler is not None else ""

        asset_section = root.find("asset")
        if asset_section is not None:
            # Mesh files
            for mesh in asset_section.findall("mesh"):
                mesh_file = mesh.get("file")
                if mesh_file:
                    raw = os.path.join(meshdir, mesh_file) if meshdir else mesh_file
                    asset_paths.add(os.path.normpath(raw))

            # Texture files (only those with a `file` attribute)
            for texture in asset_section.findall("texture"):
                tex_file = texture.get("file")
                if tex_file:
                    raw = os.path.join(texturedir, tex_file) if texturedir else tex_file
                    asset_paths.add(os.path.normpath(raw))

            # Heightfield files
            for hfield in asset_section.findall("hfield"):
                hfield_file = hfield.get("file")
                if hfield_file:
                    asset_paths.add(os.path.normpath(hfield_file))

    except (ET.ParseError, FileNotFoundError, PermissionError):
        pass

    return asset_paths


def _extract_asset_paths_from_urdf(urdf_path: str) -> set[str]:
    """Extract mesh file paths from a URDF file.

    Handles ``package://`` and ``file://`` URI prefixes by stripping them.

    Parameters
    ----------
    urdf_path : str
        Path to the URDF file.

    Returns
    -------
    set[str]
        Mesh file paths relative to the URDF file directory.
    """
    asset_paths: set[str] = set()

    try:
        tree = ET.parse(urdf_path)
        root = tree.getroot()

        for mesh in root.findall(".//mesh"):
            mesh_filename = mesh.get("filename")
            if mesh_filename:
                # Strip URI prefixes
                if mesh_filename.startswith("package://"):
                    mesh_filename = mesh_filename[len("package://") :]
                elif mesh_filename.startswith("file://"):
                    mesh_filename = mesh_filename[len("file://") :]
                asset_paths.add(os.path.normpath(mesh_filename))

    except (ET.ParseError, FileNotFoundError, PermissionError):
        pass

    return asset_paths


def _extract_asset_paths_from_obj(obj_path: str) -> set[str]:
    """Extract material library (.mtl) file paths from a Wavefront OBJ file.

    Parameters
    ----------
    obj_path : str
        Path to the OBJ file.

    Returns
    -------
    set[str]
        Material file paths relative to the OBJ file directory.
    """
    asset_paths: set[str] = set()

    try:
        with open(obj_path, "r") as f:
            for line in f:
                line = line.split("#", 1)[0].strip()
                if line.startswith("mtllib "):
                    mtl_file = line[len("mtllib ") :].strip()
                    if mtl_file:
                        asset_paths.add(os.path.normpath(mtl_file))
    except (FileNotFoundError, PermissionError, UnicodeDecodeError):
        pass

    return asset_paths


def _extract_asset_paths_from_mtl(mtl_path: str) -> set[str]:
    """Extract texture file paths from a Wavefront MTL file.

    Parameters
    ----------
    mtl_path : str
        Path to the MTL file.

    Returns
    -------
    set[str]
        Texture file paths relative to the MTL file directory.
    """
    asset_paths: set[str] = set()
    texture_directives = {
        "map_ka",
        "map_kd",
        "map_ks",
        "map_ns",
        "map_d",
        "map_bump",
        "bump",
        "disp",
        "decal",
        "map_pr",
        "map_pm",
        "norm",
    }

    try:
        with open(mtl_path, "r") as f:
            for line in f:
                line = line.split("#", 1)[0]
                parts = line.split()
                if len(parts) >= 2 and parts[0].lower() in texture_directives:
                    # Texture path is the last token (options may precede it)
                    tex_file = parts[-1]
                    asset_paths.add(os.path.normpath(tex_file))
    except (FileNotFoundError, PermissionError, UnicodeDecodeError):
        pass

    return asset_paths


def _extract_asset_paths_from_gltf(gltf_path: str) -> set[str]:
    """Extract external buffer and image URIs from a glTF JSON file.

    glTF JSONs reference their binary buffer(s) and texture image(s) via
    ``buffers[*].uri`` and ``images[*].uri``. Data URIs (``data:...``) are
    skipped because they're inlined. ``.glb`` files self-contain everything,
    so they're not handled here.

    Parameters
    ----------
    gltf_path : str
        Path to the glTF JSON file.

    Returns
    -------
    set[str]
        Asset file paths relative to the glTF file's directory.
    """
    asset_paths: set[str] = set()

    try:
        with open(gltf_path, "r") as f:
            doc = json.load(f)
    except (FileNotFoundError, PermissionError, UnicodeDecodeError, json.JSONDecodeError):
        return asset_paths

    for section in ("buffers", "images"):
        for entry in doc.get(section, []) or []:
            uri = entry.get("uri") if isinstance(entry, dict) else None
            if not uri or uri.startswith("data:"):
                continue
            asset_paths.add(os.path.normpath(uri))

    return asset_paths


def _extract_asset_paths_from_dae(dae_path: str) -> set[str]:
    """Extract texture image paths from a COLLADA (.dae) file.

    Returns the paths the loader opens, since only those are worth fetching. Genesis loads
    COLLADA through trimesh and pycollada: each ``<image>`` (in ``<library_images>`` or an
    effect's ``<profile_COMMON>``) names its file by its ``<init_from>`` text, which trimesh
    opens verbatim relative to the COLLADA file, then by basename. So the text is returned as
    written — not percent-decoded, not stripped of ``file://`` (the loader does neither; a
    decoded path would fetch a file it never reads) — and the dep walk's basename fallback
    covers absolute and ``file://`` paths the loader reaches by basename.

    Not read: an effect's ``<surface><init_from>``, which names an image *id*, not a file; and
    COLLADA 1.5's ``<init_from><ref>``, which pycollada (1.4 only) never loads.

    Parameters
    ----------
    dae_path : str
        Path to the COLLADA file.

    Returns
    -------
    set[str]
        Image file paths relative to the COLLADA file directory.
    """
    asset_paths: set[str] = set()

    try:
        root = ET.parse(dae_path).getroot()
    except (ET.ParseError, FileNotFoundError, PermissionError):
        return asset_paths

    for image in root.findall(".//{*}image"):
        image_file = (image.findtext("{*}init_from") or "").strip()
        if image_file:
            asset_paths.add(os.path.normpath(image_file))

    return asset_paths


def _get_asset_dependencies(file_path: str) -> set[str]:
    """Get file dependencies (meshes, textures, materials) from a scene/model file.

    Parameters
    ----------
    file_path : str
        Path to the XML, URDF, OBJ, MTL, glTF, COLLADA, or tetgen ``.node`` file.

    Returns
    -------
    set[str]
        Asset file paths relative to the descriptor file's directory.
    """
    if not os.path.exists(file_path):
        return set()

    file_lower = file_path.lower()
    if file_lower.endswith(".xml"):
        return _extract_asset_paths_from_xml(file_path)
    elif file_lower.endswith(".urdf"):
        return _extract_asset_paths_from_urdf(file_path)
    elif file_lower.endswith(".obj"):
        return _extract_asset_paths_from_obj(file_path)
    elif file_lower.endswith(".mtl"):
        return _extract_asset_paths_from_mtl(file_path)
    elif file_lower.endswith(".gltf"):
        return _extract_asset_paths_from_gltf(file_path)
    elif file_lower.endswith(".dae"):
        return _extract_asset_paths_from_dae(file_path)
    elif file_lower.endswith(".node"):
        # A tetgen mesh is a pair: the Mochi soft-body loader opens the '.ele' beside the '.node' by its stem.
        return {os.path.splitext(os.path.basename(file_path))[0] + ".ele"}
    else:
        return set()


def _resolve_dep(dep_path: str, file_dir: str, file_dir_relative: str, asset_path: str) -> list[tuple[str, str]]:
    """Resolve a dependency path to a list of (dep_pattern, dep_local_path) candidates.

    Returns the path relative to the referring file's directory as the primary
    candidate. When the dependency path contains subdirectories (e.g.
    ``assets/base.dae``), a basename-only fallback is also included so that
    OBJ files with absolute-style mtllib paths are handled correctly.

    The caller should try candidates in order: use the first whose local path
    exists, or include all patterns when requesting a download.
    """
    if file_dir_relative != ".":
        dep_pattern = os.path.join(file_dir_relative, dep_path)
    else:
        dep_pattern = dep_path
    dep_pattern = os.path.normpath(dep_pattern)
    dep_local_path = os.path.join(asset_path, dep_pattern)

    primary = (dep_pattern, dep_local_path)

    if os.sep in dep_path:
        basename = os.path.basename(dep_path)
        fallback_local = os.path.join(file_dir, basename)
        # If the basename already exists locally, use it directly
        if os.path.exists(fallback_local):
            fb_pattern = os.path.relpath(fallback_local, asset_path)
            return [(fb_pattern, fallback_local)]
        # Otherwise return both primary and basename fallback for download
        if file_dir_relative != ".":
            fb_pattern = os.path.join(file_dir_relative, basename)
        else:
            fb_pattern = basename
        fb_pattern = os.path.normpath(fb_pattern)
        fb_local_path = os.path.join(asset_path, fb_pattern)
        if fb_pattern != dep_pattern:
            return [primary, (fb_pattern, fb_local_path)]

    return [primary]


def _warn_dependency_not_delivered(
    repo_id: str, local_dir: str, referrer: str, dep_path: str, patterns: list[str]
) -> None:
    """Warn, once per process, that a fetch did not deliver a file ``referrer`` names.

    A warning, not an error: Genesis still loads an asset with a dangling ref (a texture renders
    untextured), and third-party assets ship such refs — absolute paths from the author's machine.
    Nothing remembers the miss, so a resolve after the file is uploaded (or the Hub is reachable
    again) fetches it.
    """
    warn_once(
        f"Asset dependency not delivered: {referrer} names {dep_path!r}, but fetching {patterns} from the "
        f"Hub repo {repo_id!r} produced no file (the repo lacks it, or the Hub was unreachable — see any "
        f"huggingface_hub warning above), so the asset loads without it: a missing texture renders "
        f"untextured. Fix the asset, not the cache: upload the file to {repo_id!r} at one of {patterns}, or "
        f"correct the reference in {referrer}. Copying the file into {local_dir} fixes this machine only."
    )


def _list_repo_files(repo_id: str) -> list[str]:
    """Every file path in the dataset repo ``repo_id`` (one Hub call)."""
    return HfApi().list_repo_files(repo_id, repo_type="dataset")


def _not_fetched_message(file: str, repo_file: str, repo_id: str, local_dir: str) -> str:
    """Why fetching ``repo_file`` delivered nothing: the Hub unreachable, the file not in the repo, or neither."""
    try:
        files = _list_repo_files(repo_id)
    except Exception as error:  # noqa: BLE001 — the failure *is* the diagnosis; the caller raises either way
        detail = (str(error).splitlines() or [""])[0]
        return (
            f"Asset {file!r} is not in {local_dir} and could not be fetched: the Hub repo {repo_id!r} cannot be "
            f"reached ({type(error).__name__}: {detail}). Offline (HF_HUB_OFFLINE=1) or no network? Fetch it once "
            f"online and the cache serves it after. The path is not the problem: do not rewrite `file`."
        )
    if repo_file in files:
        return (
            f"Asset {file!r} is in the Hub repo {repo_id!r}, but the fetch did not deliver it into {local_dir}: see "
            f"the huggingface_hub warning above (an interrupted download, or no write access to {local_dir})."
        )
    ending_in = sorted(path for path in files if path.endswith("/" + repo_file))
    if len(ending_in) == 1:
        fix = f"Pass its full repo path: file={ending_in[0]!r}."
    elif ending_in:
        fix = f"Pass the full repo path of the one you mean; repo files ending in it: {ending_in[:5]}."
    else:
        fix = "Pass the file's full path in that repo."
    return (
        f"Asset {file!r} is not a file in the Hub repo {repo_id!r}. {fix} A bare file name or a path suffix is "
        f"not looked up: another dataset's cache can sit under the same local_dir, so a tail could name its file."
    )


def get_asset_path(
    file: str,
    registry: str | None = "Kashu7100",
    dataset: str | None = "eden_assets",
    local_dir: str | None = None,
) -> str:
    """Resolve an asset to a local file, fetching it and its dependencies from the Hub if needed.

    ``file`` is one of:

    - a path inside ``local_dir``, cached or not: the repo path below ``local_dir``, resolved as that
      repo path is (so a missing dependency of a cached file is fetched too);
    - any other existing local path, returned as is (an absolute one that does not exist raises,
      without a fetch);
    - the file's full path in the ``registry/dataset`` Hub repo, e.g. ``models/eggplant/object.urdf``.

    ``local_dir`` defaults to :func:`eden.utils.misc.get_assets_dir`, the download root ``EDEN_ASSETS_DIR``
    moves. A repo path is matched exactly, never by its tail. ``local_dir`` can hold other datasets' caches
    (the download root holds ``ycb/``, ``simpler/``, ...), so a bare name or a path suffix could name
    another dataset's file; the error for one lists the repo paths that end in it.

    With ``registry`` or ``dataset`` set to ``None`` nothing is fetched: ``file`` is returned if it exists
    locally, else ``None``.
    """
    local_dir = resolve_local_dir(local_dir)
    if registry is None or dataset is None:
        return file if os.path.exists(file) else None

    asset_path = local_dir
    repo_id = f"{registry}/{dataset}"
    local_dir_abs = os.path.abspath(local_dir)
    file_abs = os.path.abspath(file)
    if file_abs.startswith(local_dir_abs + os.sep) and not os.path.isdir(file_abs):
        # A path inside local_dir is the cached copy of the repo path below it, so it resolves
        # (and has its dependencies fetched) exactly as that repo path does.
        repo_file = os.path.relpath(file_abs, local_dir_abs)
    elif os.path.exists(file):
        return file
    elif os.path.isabs(file):
        raise FileNotFoundError(
            f"No such local file: {file!r}. An absolute path is read from disk, never fetched; only a path "
            f"inside local_dir ({local_dir}) or a repo path names a file of the Hub repo {repo_id!r}."
        )
    else:
        repo_file = file

    matched_file_path = os.path.join(local_dir, repo_file)
    if not os.path.isfile(matched_file_path):
        # Escaped: ``allow_patterns`` are fnmatch patterns, and a name is matched literally.
        _snapshot_download_locked(
            repo_type="dataset",
            repo_id=repo_id,
            allow_patterns=[glob.escape(repo_file)],
            local_dir=local_dir,
        )
        if not os.path.isfile(matched_file_path):
            raise FileNotFoundError(_not_fetched_message(file, repo_file, repo_id, local_dir))

    # Download asset dependencies (meshes, textures, materials, etc.)
    # Walk the dep graph layer by layer, BFS: every file at the current
    # depth is parsed for refs, then ALL missing refs are downloaded in a
    # single batched ``snapshot_download`` call before advancing to the
    # next layer. URDFs in particular reference dozens of ``.obj`` files,
    # which each reference one ``.mtl`` — batching avoids one HF round-trip
    # per file (laptop URDF: 1 + 37 + 37 + textures ≈ 4 batched calls instead
    # of ~100 single-file ones).
    checked_files: set[str] = {matched_file_path}
    current_layer: list[str] = [matched_file_path]

    while current_layer:
        next_missing_patterns: list[str] = []
        next_local_paths: list[str] = []
        already_local: list[str] = []
        # (referring file, ref, candidates) for each ref this layer's fetch asks for.
        requested: list[tuple[str, str, list[tuple[str, str]]]] = []

        for current_file in current_layer:
            dep_paths = _get_asset_dependencies(current_file)
            if not dep_paths:
                continue

            file_dir = os.path.dirname(current_file)
            file_dir_relative = os.path.relpath(file_dir, asset_path)

            for dep_path in dep_paths:
                candidates = _resolve_dep(dep_path, file_dir, file_dir_relative, asset_path)
                found_local = False
                for _dep_pattern, dep_local_path in candidates:
                    if os.path.exists(dep_local_path):
                        if dep_local_path not in checked_files:
                            already_local.append(dep_local_path)
                        found_local = True
                        break
                if not found_local:
                    requested.append((current_file, dep_path, candidates))
                    for dep_pattern, dep_local_path in candidates:
                        # An absolute ref's primary candidate is a host path, not a repo path.
                        if not os.path.isabs(dep_pattern):
                            next_missing_patterns.append(dep_pattern.replace(os.sep, "/"))
                            next_local_paths.append(dep_local_path)

        # Dedup patterns + their matched local paths, preserving order.
        seen: set[str] = set()
        unique_patterns: list[str] = []
        unique_local_paths: list[str] = []
        for pattern, local_path in zip(next_missing_patterns, next_local_paths):
            if pattern not in seen:
                seen.add(pattern)
                unique_patterns.append(pattern)
                unique_local_paths.append(local_path)

        if unique_patterns:
            _snapshot_download_locked(
                repo_type="dataset",
                repo_id=repo_id,
                allow_patterns=[glob.escape(pattern) for pattern in unique_patterns],
                local_dir=local_dir,
            )
        for referrer, dep_path, candidates in requested:
            if not any(os.path.exists(dep_local_path) for _, dep_local_path in candidates):
                patterns = [p.replace(os.sep, "/") for p, _ in candidates if not os.path.isabs(p)]
                _warn_dependency_not_delivered(repo_id, local_dir, referrer, dep_path, patterns)

        next_layer: list[str] = []
        for p in already_local:
            if p not in checked_files:
                checked_files.add(p)
                next_layer.append(p)
        for p in unique_local_paths:
            if p not in checked_files and os.path.exists(p):
                checked_files.add(p)
                next_layer.append(p)
        current_layer = next_layer

    return matched_file_path


# The HuggingFace dataset repo holding the sample data the in-tree tasks default to, one
# directory per task. A task's default clip cannot live under ``examples/``: that tree is not
# packaged, so a checkout-relative default resolves inside ``site-packages/`` on a
# pip-installed Eden and names a file that cannot exist.
SAMPLE_DATA_REGISTRY = "Kashu7100"
SAMPLE_DATA_REPO = "eden_example"


def sample_data_path(task: str, filename: str) -> str:
    """Resolve one of ``task``'s sample files, fetching it from HuggingFace on first use.

    The one way in to the shared example-data repo, so a checkout and an installed wheel
    behave identically. The task name *is* the directory holding that task's files, which
    stops a task widening its fetch to another's.

    Parameters
    ----------
    task : str
        Registered task name, and the directory holding that task's files in the repo.
    filename : str
        File to resolve, relative to that directory. Only this one file is downloaded.

    Returns
    -------
    str
        Local path to the file.
    """
    return get_asset_path(
        f"{task}/{filename}",
        registry=SAMPLE_DATA_REGISTRY,
        dataset=SAMPLE_DATA_REPO,
        # A directory of its own rather than the asset root (the default): tasks write beside their
        # clip — contrack caches the exported object mesh in a ``.cache/`` there. It must also be
        # stable, not a HuggingFace snapshot dir whose name tracks the revision, or those caches
        # cold-start every run.
        local_dir=get_cache_dir(SAMPLE_DATA_REPO),
    )


def find_latest_checkpoint(log_dir: str, return_path: bool = True) -> str:
    if not os.path.exists(log_dir):
        raise FileNotFoundError(f"Log directory {log_dir} not found")
    if not os.path.isdir(log_dir):
        raise NotADirectoryError(f"Log directory {log_dir} is not a directory")
    if not os.listdir(log_dir):
        raise FileNotFoundError(f"Log directory {log_dir} is empty")
    if not any(filename.endswith(".pt") for filename in os.listdir(log_dir)):
        raise FileNotFoundError(f"No model checkpoints in {log_dir}")

    candidates = model_checkpoints(log_dir)
    if not candidates:
        raise FileNotFoundError(f"No model checkpoints matching 'model_{{itr}}.pt' in {log_dir}")
    if return_path:
        return os.path.join(log_dir, candidates[-1][1])
    else:
        return candidates[-1][1]


_MODEL_CHECKPOINT = re.compile(r"^model_(\d+)\.pt$")


def model_checkpoints(log_dir: str) -> list[tuple[int, str]]:
    """The run's ``model_<iteration>.pt`` checkpoints in ``log_dir``, as ``(iteration, filename)``, oldest first."""
    found = ((_MODEL_CHECKPOINT.match(name), name) for name in os.listdir(log_dir))
    return sorted((int(match.group(1)), name) for match, name in found if match)


def resolve_run_dir(runs_root: str, ref: str) -> str:
    """Resolve a ``--resume`` reference to a run directory under ``runs_root``.

    ``runs_root`` is a task's run root (``logs/<task>``); each child is a run directory
    named ``<timestamp>[-<run_name>]-<id>``. ``ref`` selects one of them:

    - ``"latest"`` — the most recently modified run directory;
    - an exact run id — the trailing ``-<id>`` segment of a run's basename;
    - a substring — matched against the whole basename (e.g. part of a run_name).

    Fails loud, listing the available runs, when nothing matches or a substring is
    ambiguous, so a typo never silently resumes the wrong run. This is the run-directory
    analogue of :func:`find_latest_checkpoint` (which selects a checkpoint *within* one
    run); callers chain the two: ``find_latest_checkpoint(resolve_run_dir(root, ref))``.

    Parameters
    ----------
    runs_root : str
        The task's run root, e.g. ``logs/<task>``.
    ref : str
        ``"latest"``, an exact run id, or a basename substring.

    Returns
    -------
    str
        Path (joined onto ``runs_root``) of the resolved run directory.

    Raises
    ------
    FileNotFoundError
        If ``runs_root`` has no run directories, or ``ref`` matches none.
    ValueError
        If ``ref`` is a substring matching more than one run.
    """
    if not os.path.isdir(runs_root):
        raise FileNotFoundError(f"No runs directory {runs_root!r} to resume from — train the task first.")
    run_dirs = sorted(d for d in os.listdir(runs_root) if os.path.isdir(os.path.join(runs_root, d)))
    if not run_dirs:
        raise FileNotFoundError(f"No runs under {runs_root!r} to resume from — train the task first.")

    def _full(name: str) -> str:
        return os.path.join(runs_root, name)

    if ref == "latest":
        # Not max(run_dirs): the "%y%b%dT..." timestamp prefix (Aug/Sep/...) does not sort
        # chronologically as a string, so pick the most recently written run by mtime.
        return _full(max(run_dirs, key=lambda d: os.path.getmtime(_full(d))))

    # Exact run-id match (the trailing base-36 segment); ids are unique, so 0 or 1 hit.
    id_matches = [d for d in run_dirs if d.rsplit("-", 1)[-1] == ref]
    if id_matches:
        return _full(id_matches[0])

    sub_matches = [d for d in run_dirs if ref in d]
    if len(sub_matches) == 1:
        return _full(sub_matches[0])
    if len(sub_matches) > 1:
        raise ValueError(
            f"--resume {ref!r} is ambiguous under {runs_root!r}; it matches "
            + ", ".join(sub_matches)
            + ". Use 'latest', a full run id, or a longer substring."
        )
    raise FileNotFoundError(
        f"--resume {ref!r} matched no run under {runs_root!r}. Available runs: " + ", ".join(run_dirs)
    )


def load_text(filepath: str) -> dict:
    try:
        if filepath.endswith(".yaml") or filepath.endswith(".yml"):
            with open(filepath, "r") as f:
                return yaml.safe_load(f)
        elif filepath.endswith(".json"):
            with open(filepath, "r") as f:
                return json.load(f)
        else:
            raise ValueError(f"Unsupported file type: {filepath}")
    except OSError as e:
        # Provide a clearer error message while preserving the original exception context.
        raise FileNotFoundError(f"Unable to open file '{filepath}': {e}") from e
