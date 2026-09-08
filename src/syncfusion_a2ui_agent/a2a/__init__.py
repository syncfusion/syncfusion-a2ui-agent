"""Google A2A SDK glue — lifecycle, sessions, streaming, task execution."""

from .transport import A2ATransport, start_server

__all__ = ["A2ATransport", "start_server"]
