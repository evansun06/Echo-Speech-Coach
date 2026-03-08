from __future__ import annotations

from functools import lru_cache
from typing import Any

from .coach_graph import build_reasoning_graph, run_reasoning_graph
from .provider import build_reasoning_models
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
