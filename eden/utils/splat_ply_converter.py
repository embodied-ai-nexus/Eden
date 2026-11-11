"""Utility to convert gaussian splat PLY files to a renderer-compatible format.

The Nyx light-field importer requires SH degree 3 PLY files with
``f_rest_0..f_rest_44``. Some PLY files (e.g. from WorldLabs) only carry SH
degree 0 (``f_dc_0..f_dc_2``, no ``f_rest_*``). This utility pads the missing
SH coefficients with zeros so the resulting PLY is accepted by the renderer.
"""

from __future__ import annotations

import os
import tempfile


def _parse_ply_header(file_path: str):
    """Parse PLY header and return (header_lines, properties, vertex_count, header_end_offset)."""
    properties = []
    vertex_count = 0
    header_lines = []

    with open(file_path, "rb") as f:
        while True:
            line = f.readline().decode("ascii").strip()
            header_lines.append(line)
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
            elif line.startswith("property"):
                parts = line.split()
                properties.append((parts[1], parts[2]))  # (type, name)
            if line == "end_header":
                header_end_offset = f.tell()
                break

    return header_lines, properties, vertex_count, header_end_offset


def needs_sh_padding(file_path: str) -> bool:
    """Check if a PLY file needs SH coefficient padding for renderer compatibility."""
    _, properties, _, _ = _parse_ply_header(file_path)
    prop_names = {name for _, name in properties}
    has_f_rest = any(name.startswith("f_rest_") for name in prop_names)
    has_f_dc = "f_dc_0" in prop_names
    return has_f_dc and not has_f_rest


def convert_to_sh3_ply(input_path: str, output_path: str | None = None, sh_degree: int = 3) -> str:
    """Convert a gaussian splat PLY to SH-degree-3 format by padding SH coefficients.

    The input must be a binary little-endian PLY (the output header always declares that format).

    Parameters
    ----------
    input_path : str
        Path to the input PLY file.
    output_path : str, optional
        Path to write the output PLY. If None, writes to a temp file.
    sh_degree : int
        Target SH degree. Default 3 (requires f_rest_0..f_rest_44).

    Returns
    -------
    str
        Path to the output PLY file (same as input if no conversion needed).
    """
    if not needs_sh_padding(input_path):
        return input_path

    _, properties, vertex_count, header_end_offset = _parse_ply_header(input_path)

    num_sh_rest = 3 * (sh_degree + 1) ** 2 - 3  # 45 for degree 3

    new_properties = []
    for dtype, name in properties:
        new_properties.append((dtype, name))
        if name == "f_dc_2":
            for i in range(num_sh_rest):
                new_properties.append(("float", f"f_rest_{i}"))

    type_sizes = {"float": 4, "double": 8, "uchar": 1, "int": 4, "uint": 4, "short": 2, "ushort": 2}
    old_vertex_size = sum(type_sizes.get(dtype, 4) for dtype, _ in properties)
    padding_size = num_sh_rest * 4

    insert_byte_offset = 0
    for dtype, name in properties:
        insert_byte_offset += type_sizes.get(dtype, 4)
        if name == "f_dc_2":
            break

    new_header_lines = []
    new_header_lines.append("ply")
    new_header_lines.append("format binary_little_endian 1.0")
    new_header_lines.append(f"element vertex {vertex_count}")
    for dtype, name in new_properties:
        new_header_lines.append(f"property {dtype} {name}")
    new_header_lines.append("end_header")
    new_header = "\n".join(new_header_lines) + "\n"

    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".ply", prefix="splat_sh3_")
        os.close(fd)

    zero_padding = b"\x00" * padding_size

    with open(input_path, "rb") as fin, open(output_path, "wb") as fout:
        fin.seek(header_end_offset)
        fout.write(new_header.encode("ascii"))
        for _ in range(vertex_count):
            vertex_data = fin.read(old_vertex_size)
            fout.write(vertex_data[:insert_byte_offset])
            fout.write(zero_padding)
            fout.write(vertex_data[insert_byte_offset:])

    return output_path
