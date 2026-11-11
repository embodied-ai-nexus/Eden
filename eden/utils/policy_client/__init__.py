"""Request/reply client plumbing: transports, serialisers and their error hierarchy.

Used by server-backed policies (:mod:`eden.extensions.policies.client`); nothing here
depends on policies, so teleoperation bridges or deployment backends can reuse it.
"""

from eden.utils.policy_client.errors import (
    ClientError,
    SerializationError,
    ServerError,
    TransportError,
    TransportTimeoutError,
)
from eden.utils.policy_client.serializer import SERIALIZER_REGISTRY, SerializerBase, make_serializer, require_module
from eden.utils.policy_client.transport import (
    TRANSPORT_REGISTRY,
    TransportBase,
    TransportConfig,
    make_transport,
    server_url,
)

__all__ = [
    "SERIALIZER_REGISTRY",
    "TRANSPORT_REGISTRY",
    "ClientError",
    "SerializationError",
    "SerializerBase",
    "ServerError",
    "TransportBase",
    "TransportConfig",
    "TransportError",
    "TransportTimeoutError",
    "make_serializer",
    "make_transport",
    "require_module",
    "server_url",
]
