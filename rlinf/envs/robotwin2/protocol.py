# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Wire protocol shared by the RoboTwin 2.0 bridge client and server."""

from __future__ import annotations

from typing import Any

PROTOCOL_VERSION = 1
AUTHKEY = b"rlinf-robotwin2-v1"


class BridgeProtocolError(RuntimeError):
    """Raised when a bridge response violates the request/response contract."""


def make_request(request_id: int, command: str, **payload: Any) -> dict[str, Any]:
    """Build one versioned request."""
    return {
        "protocol_version": PROTOCOL_VERSION,
        "request_id": request_id,
        "command": command,
        "payload": payload,
    }


def make_response(
    request: dict[str, Any], *, result: Any = None, error: str | None = None
) -> dict[str, Any]:
    """Build one response correlated with ``request``."""
    return {
        "protocol_version": PROTOCOL_VERSION,
        "request_id": request.get("request_id"),
        "ok": error is None,
        "result": result,
        "error": error,
    }


def validate_response(response: Any, request_id: int) -> Any:
    """Validate a response and return its result payload."""
    if not isinstance(response, dict):
        raise BridgeProtocolError(
            f"Bridge returned {type(response).__name__}, expected a dictionary."
        )
    if response.get("protocol_version") != PROTOCOL_VERSION:
        raise BridgeProtocolError(
            "RoboTwin 2.0 bridge protocol version mismatch: "
            f"expected {PROTOCOL_VERSION}, got {response.get('protocol_version')}."
        )
    if response.get("request_id") != request_id:
        raise BridgeProtocolError(
            "RoboTwin 2.0 bridge response id mismatch: "
            f"expected {request_id}, got {response.get('request_id')}."
        )
    if not response.get("ok", False):
        raise BridgeProtocolError(str(response.get("error") or "Unknown bridge error"))
    return response.get("result")
