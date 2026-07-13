from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from prism.serve.protocol import PolicyRequest


class PolicyBackend(Protocol):
    """Model-agnostic inference boundary used by the benchmark server."""

    @property
    def metadata(self) -> Mapping[str, Any]:
        ...

    def infer(self, request: PolicyRequest) -> Mapping[str, Any] | Any:
        ...
