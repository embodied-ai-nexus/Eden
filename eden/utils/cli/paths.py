"""Shared ``--src`` / ``--out`` file plumbing for the batch CLI commands.

Used by the drivers that map one input file to one output file (``eden retarget``,
``eden pose``): resolve ``--src`` into a file list, then map each input under ``--out``,
mirroring the source tree and refusing to overwrite the input.

Pure ``pathlib`` — no Eden/Genesis imports, so ``--help`` works before any backend is touched.
"""

from __future__ import annotations

from pathlib import Path


def iter_src_files(src: Path, pattern: str) -> tuple[bool, Path, list[Path]]:
    """Resolve ``--src`` into ``(single, src_root, files)``; error on missing / empty / wrong suffix."""
    suffix = Path(pattern).suffix  # "*.npz" and "**/*.npz" alike -> ".npz"
    if src.is_file():
        if src.suffix != suffix:
            raise SystemExit(f"--src file {src} is not a {suffix} file (expected pattern {pattern!r}).")
        return True, src.parent, [src]
    if src.is_dir():
        files = sorted(src.rglob(pattern))
        if not files:
            raise SystemExit(f"No files matching {pattern!r} found under {src}")
        return False, src, files
    raise SystemExit(f"--src path does not exist: {src}")


def resolve_out(out: Path, src_root: Path, src_file: Path, out_suffix: str, *, single: bool) -> Path:
    """Map one input file to its output path, refusing to overwrite the input."""
    # A suffix is the only hint that --out names a file rather than a directory, and it is a weak
    # one: a run directory called `runs/v1.2` has suffix ".2". An existing directory settles it.
    if out.suffix and not out.is_dir():
        if not single:
            raise SystemExit(f"--out {out} looks like a file, but --src is a directory; pass a directory for --out.")
        # np.savez / pickle append the suffix; normalize now so skip-check and log agree.
        resolved = out.with_suffix(out_suffix)
    else:
        resolved = out / src_file.relative_to(src_root).with_suffix(out_suffix)
    if resolved.resolve() == src_file.resolve():
        raise SystemExit(f"Refusing to write output onto its input {src_file}; choose a different --out.")
    return resolved
