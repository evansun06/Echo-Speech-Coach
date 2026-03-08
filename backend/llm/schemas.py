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


@dataclass(slots=True)
class ReasoningResult:
    role: ReasoningRole
    model_name: str
    output_text: str
    usage: dict[str, int] = field(default_factory=dict)
    response_metadata: dict[str, Any] = field(default_factory=dict)
    request_metadata: dict[str, Any] = field(default_factory=dict)


class ReasoningState(TypedDict):
    role: ReasoningRole
    system_prompt: str
    user_prompt: str
    request_metadata: dict[str, Any]
    model_name: NotRequired[str]
    output_text: NotRequired[str]
    usage: NotRequired[dict[str, int]]
    response_metadata: NotRequired[dict[str, Any]]
