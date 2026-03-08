from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal, NotRequired, TypedDict

ReasoningRole = Literal["subagent", "primary"]


@dataclass(slots=True)
class ReasoningInput:
    role: ReasoningRole
    system_prompt: str
    user_prompt: str
    metadata: dict[str, Any] = field(default_factory=dict)
    request_payload: dict[str, Any] = field(default_factory=dict)
    structured_schema: dict[str, Any] | None = None


@dataclass(slots=True)
class ReasoningResult:
    role: ReasoningRole
    model_name: str
    output_text: str
    usage: dict[str, int] = field(default_factory=dict)
    response_metadata: dict[str, Any] = field(default_factory=dict)
    request_metadata: dict[str, Any] = field(default_factory=dict)
    structured_output: dict[str, Any] = field(default_factory=dict)


class ReasoningState(TypedDict):
    role: ReasoningRole
    system_prompt: str
    user_prompt: str
    request_metadata: dict[str, Any]
    request_payload: NotRequired[dict[str, Any]]
    structured_schema: NotRequired[dict[str, Any]]
    model_name: NotRequired[str]
    output_text: NotRequired[str]
    usage: NotRequired[dict[str, int]]
    response_metadata: NotRequired[dict[str, Any]]
    structured_output: NotRequired[dict[str, Any]]
