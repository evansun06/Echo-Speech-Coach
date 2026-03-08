from __future__ import annotations

from functools import lru_cache
from typing import Any

from .coach_graph import build_reasoning_graph, run_reasoning_graph
from .provider import ChatModelConfig, build_chat_model, build_reasoning_models
from .schemas import ReasoningInput, ReasoningResult


@lru_cache(maxsize=1)
def _compiled_reasoning_graph():
    """Build and cache a compiled reasoning graph instance for reuse."""
    models = build_reasoning_models()
    return build_reasoning_graph(models=models)


def clear_reasoning_graph_cache() -> None:
    """Clear the cached compiled graph so new model config is picked up."""
    _compiled_reasoning_graph.cache_clear()


def run_reasoning(*, reasoning_input: ReasoningInput, graph: Any | None = None) -> ReasoningResult:
    """Run a reasoning request on either an injected graph or the cached default."""
    target_graph = graph or _compiled_reasoning_graph()
    return run_reasoning_graph(graph=target_graph, reasoning_input=reasoning_input)


def run_subagent_reasoning(
    *,
    system_prompt: str,
    user_prompt: str,
    metadata: dict[str, Any] | None = None,
    graph: Any | None = None,
) -> ReasoningResult:
    """Convenience wrapper for running reasoning with the subagent role."""
    return run_reasoning(
        reasoning_input=ReasoningInput(
            role="subagent",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            metadata=dict(metadata or {}),
        ),
        graph=graph,
    )


def run_subagent_structured_reasoning(
    *,
    system_prompt: str,
    user_prompt: str,
    structured_schema: dict[str, Any],
    metadata: dict[str, Any] | None = None,
    request_payload: dict[str, Any] | None = None,
    graph: Any | None = None,
) -> ReasoningResult:
    """Run subagent reasoning and request a structured model response payload."""
    return run_reasoning(
        reasoning_input=ReasoningInput(
            role="subagent",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            metadata=dict(metadata or {}),
            request_payload=dict(request_payload or {}),
            structured_schema=dict(structured_schema),
        ),
        graph=graph,
    )


def run_primary_reasoning(
    *,
    system_prompt: str,
    user_prompt: str,
    metadata: dict[str, Any] | None = None,
    graph: Any | None = None,
) -> ReasoningResult:
    """Convenience wrapper for running reasoning with the primary role."""
    return run_reasoning(
        reasoning_input=ReasoningInput(
            role="primary",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            metadata=dict(metadata or {}),
        ),
        graph=graph,
    )


def run_primary_structured_reasoning(
    *,
    system_prompt: str,
    user_prompt: str,
    structured_schema: dict[str, Any],
    metadata: dict[str, Any] | None = None,
    request_payload: dict[str, Any] | None = None,
    graph: Any | None = None,
) -> ReasoningResult:
    """Run primary reasoning and request a structured model response payload."""
    return run_reasoning(
        reasoning_input=ReasoningInput(
            role="primary",
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            metadata=dict(metadata or {}),
            request_payload=dict(request_payload or {}),
            structured_schema=dict(structured_schema),
        ),
        graph=graph,
    )


@lru_cache(maxsize=1)
def _cached_chat_model() -> ChatModelConfig:
    """Build and cache the configured Gemini chat model."""
    return build_chat_model()


def clear_chat_model_cache() -> None:
    """Clear the cached chat model so updated settings are picked up."""
    _cached_chat_model.cache_clear()


def _normalize_stream_chunk_content(content: Any) -> str:
    """Convert streamed chunk content into plain text without stripping spacing."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
                continue
            if isinstance(item, dict):
                maybe_text = item.get("text")
                if isinstance(maybe_text, str):
                    parts.append(maybe_text)
        return "".join(parts)
    if content is None:
        return ""
    return str(content)


def stream_chat_response_tokens(
    *,
    system_prompt: str,
    user_prompt: str,
    metadata: dict[str, Any] | None = None,
    model_config: ChatModelConfig | None = None,
):
    """Yield streamed Gemini chat text chunks for one system+user prompt pair."""
    from langchain_core.messages import HumanMessage, SystemMessage

    resolved_model_config = model_config or _cached_chat_model()
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    stream_config = {"metadata": dict(metadata)} if metadata else None
    chunk_stream = (
        resolved_model_config.model.stream(messages, config=stream_config)
        if stream_config is not None
        else resolved_model_config.model.stream(messages)
    )
    for chunk in chunk_stream:
        token = _normalize_stream_chunk_content(getattr(chunk, "content", ""))
        if token:
            yield token
