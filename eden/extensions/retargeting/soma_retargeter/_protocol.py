"""Length-prefixed pickle framing shared by the SomaRetargeter and its worker.

Kept in its own module so importing the extension package does not pull in the
worker's ``__main__`` path (which would double-execute under ``python -m``).
"""

from __future__ import annotations

import pickle
import struct

_HEADER = struct.Struct(">Q")


def send_message(stream, obj) -> None:
    """Write one length-prefixed pickle frame and flush."""
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    stream.write(_HEADER.pack(len(data)))
    stream.write(data)
    stream.flush()


def recv_message(stream):
    """Read one length-prefixed pickle frame, or ``None`` at clean EOF."""
    header = _read_exact(stream, _HEADER.size, allow_eof=True)
    if header is None:
        return None
    (length,) = _HEADER.unpack(header)
    body = _read_exact(stream, length, allow_eof=False)
    return pickle.loads(body)


def _read_exact(stream, n: int, *, allow_eof: bool):
    buf = bytearray()
    while len(buf) < n:
        chunk = stream.read(n - len(buf))
        if not chunk:
            if allow_eof and not buf:
                return None
            raise EOFError("truncated message")
        buf.extend(chunk)
    return bytes(buf)
