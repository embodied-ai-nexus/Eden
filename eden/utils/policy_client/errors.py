"""Error hierarchy for request/reply clients (``eden.utils.policy_client``).

``TransportError`` also derives from :class:`ConnectionError` and every error derives from
:class:`RuntimeError`, so callers that already distinguish "server unreachable" from
"protocol failure" with those builtins keep working.

::

    ClientError (RuntimeError)
    ├── TransportError (…, ConnectionError)   server unreachable / connection dropped
    │   └── TransportTimeoutError             request timed out
    ├── ServerError                           server answered with an error (HTTP 4xx/5xx, {"error": …})
    └── SerializationError                    payload or reply could not be (de)serialised
"""

from __future__ import annotations


class ClientError(RuntimeError):
    """Base class for every remote-policy client failure."""


class TransportError(ClientError, ConnectionError):
    """The policy server could not be reached or the connection was lost."""


class TransportTimeoutError(TransportError):
    """A request to the policy server timed out."""


class ServerError(ClientError):
    """The policy server answered, but with an error."""


class SerializationError(ClientError):
    """A payload or reply could not be encoded or decoded."""
