"""Genesis-free worker process that hosts the library ``MotionRetargeter``.

The fit is kept out of the Genesis process (see the :mod:`.retargeter` module docstring), so
retargeting runs here and results are streamed back to the in-process
:class:`SomaRetargeter` over the :mod:`_protocol` framing.

Protocol (one framed pickle object per message):
    parent -> worker : init dict of ``MotionRetargeter`` kwargs (see
                       :meth:`SomaRetargeter._init_message`), then
                       ``{"clips": [clip, ...]}`` requests, each ``clip`` the
                       keyword dict of one library ``retarget*`` call
    worker -> parent : ``{"ok": True}`` after init, then per-request
                       ``{"ok": True, "results": [...]}`` or
                       ``{"ok": False, "error": str, "trace": str}``
"""

from __future__ import annotations

import os
import sys

# The binary protocol travels over the original stdout. Duplicate it to a
# private fd and point fd 1 at stderr BEFORE any other import, so import-time
# prints can never desync the protocol. Only ever run as ``python -m`` (never
# imported), so the process-wide redirect is safe.
_PROTO_OUT = os.fdopen(os.dup(1), "wb")
os.dup2(2, 1)

from eden.extensions.retargeting.soma_retargeter._protocol import recv_message, send_message  # noqa: E402


def main() -> None:
    proto_out = _PROTO_OUT
    proto_in = sys.stdin.buffer

    import traceback

    init = recv_message(proto_in)
    if init is None:
        return
    try:
        # Imported inside the try so an ImportError reaches the parent as an init
        # failure instead of killing the worker before it can ack.
        from soma_retargeter.retargeting import MotionRetargeter

        from eden.utils.chumpy_pickle import install_chumpy_pickle_modules

        # smplx reads an SMPL / SMPL-H .pkl with a plain pickle.load, which needs chumpy's classes importable; this
        # process exists to run that fit, so it takes Eden's stand-ins process-wide.
        install_chumpy_pickle_modules()

        retargeter = MotionRetargeter(**init)
        send_message(proto_out, {"ok": True})
    except Exception as exc:  # noqa: BLE001 - report any init failure to the parent
        send_message(proto_out, {"ok": False, "error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()})
        return

    while True:
        req = recv_message(proto_in)
        if req is None:
            break
        try:
            results = retargeter.retarget_batch(req["clips"])
            send_message(proto_out, {"ok": True, "results": results})
        except Exception as exc:  # noqa: BLE001 - a bad clip must not kill the worker
            send_message(
                proto_out, {"ok": False, "error": f"{type(exc).__name__}: {exc}", "trace": traceback.format_exc()}
            )


if __name__ == "__main__":
    main()
