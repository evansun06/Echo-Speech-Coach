from __future__ import annotations

from functools import partial
from typing import Any

from .provider import ReasoningModels, get_reasoning_model
from .schemas import ReasoningInput, ReasoningResult, ReasoningState


def _normalize_response_content(content: Any) -> str:
    """Convert provider response content into a trimmed plain-text string."""
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                if item.strip():
                    parts.append(item.strip())
                continue
            if isinstance(item, dict):
                maybe_text = item.get("text")
                if isinstance(maybe_text, str) and maybe_text.strip():
                    parts.append(maybe_text.strip())
        return "\n".join(parts).strip()

    if content is None:
        return ""
    return str(content).strip()

def _normalize_usage(response: Any) -> dict[str, int]:
    """Extract token usage counters from provider response metadata."""
    usage: dict[str, int] = {}
    usage_metadata = getattr(response, "usage_metadata", None)
    if isinstance(usage_metadata, dict):
        for key in ("input_tokens", "output_tokens", "total_tokens"):
            value = usage_metadata.get(key)
            if isinstance(value, int):
                usage[key] = value
        if usage:
            return usage

    response_metadata = getattr(response, "response_metadata", None)
    if isinstance(response_metadata, dict):
        token_usage = response_metadata.get("token_usage")
        if isinstance(token_usage, dict):
            for key in ("input_tokens", "output_tokens", "total_tokens"):
                value = token_usage.get(key)
                if isinstance(value, int):
                    usage[key] = value
    return usage


def _normalize_response_metadata(response: Any) -> dict[str, Any]:
    """Return a shallow dict copy of response metadata when available."""
    response_metadata = getattr(response, "response_metadata", None)
    if not isinstance(response_metadata, dict):
        return {}
    return dict(response_metadata)


def _invoke_reasoning_model(
    state: ReasoningState, *, models: ReasoningModels
) -> ReasoningState:
    """Run the selected role model for the current state and enrich the state."""
    from langchain_core.messages import HumanMessage, SystemMessage

    role = state["role"]
    model = get_reasoning_model(models, role)
    response = model.invoke(
        [
            SystemMessage(content=state["system_prompt"]),
            HumanMessage(content=state["user_prompt"]),
        ]
    )

    model_name = (
        models.subagent_model_name
        if role == "subagent"
        else models.primary_model_name
    )
    return {
        **state,
        "model_name": model_name,
        "output_text": _normalize_response_content(getattr(response, "content", "")),
        "usage": _normalize_usage(response),
        "response_metadata": _normalize_response_metadata(response),
    }


def build_reasoning_graph(*, models: ReasoningModels):
    """Build and compile the single-step LangGraph used for reasoning calls."""
    from langgraph.graph import END, StateGraph

    graph_builder = StateGraph(ReasoningState)
    graph_builder.add_node(
        "invoke_reasoning_model",
        partial(_invoke_reasoning_model, models=models),
    )
    graph_builder.set_entry_point("invoke_reasoning_model")
    graph_builder.add_edge("invoke_reasoning_model", END)
    return graph_builder.compile()


def run_reasoning_graph(*, graph: Any, reasoning_input: ReasoningInput) -> ReasoningResult:
    """Execute the compiled reasoning graph for one request and map output to result."""
    initial_state: ReasoningState = {
        "role": reasoning_input.role,
        "system_prompt": reasoning_input.system_prompt,
        "user_prompt": reasoning_input.user_prompt,
        "request_metadata": dict(reasoning_input.metadata),
    }
    final_state: ReasoningState = graph.invoke(initial_state)
    return ReasoningResult(
        role=reasoning_input.role,
        model_name=final_state.get("model_name", ""),
        output_text=final_state.get("output_text", ""),
        usage=dict(final_state.get("usage", {})),
        response_metadata=dict(final_state.get("response_metadata", {})),
        request_metadata=dict(reasoning_input.metadata),
    )
