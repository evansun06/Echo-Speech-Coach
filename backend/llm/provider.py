from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from django.conf import settings

from .schemas import ReasoningRole

DEFAULT_SUBAGENT_MODEL = "gemini-2.5-flash"
DEFAULT_PRIMARY_MODEL = "gemini-2.5-flash"
DEFAULT_CHAT_MODEL = "gemini-2.5-flash"
DEFAULT_SUBAGENT_TEMPERATURE = 0.2
DEFAULT_PRIMARY_TEMPERATURE = 0.1
DEFAULT_CHAT_TEMPERATURE = 0.2


class ModelConfigurationError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ReasoningModels:
    subagent: Any
    primary: Any
    subagent_model_name: str
    primary_model_name: str


def _clean_string(value: Any) -> str:
    """Normalize an optional config value into a stripped string."""
    if not isinstance(value, str):
        return ""
    return value.strip()


def _resolve_temperature(value: Any, fallback: float) -> float:
    """Return a float temperature, falling back when no value is provided."""
    if value is None:
        return fallback
    return float(value)


def build_reasoning_models(
    *,
    api_key: str | None = None,
    subagent_model: str | None = None,
    primary_model: str | None = None,
    subagent_temperature: float | None = None,
    primary_temperature: float | None = None,
) -> ReasoningModels:
    """Create configured subagent and primary Gemini chat model clients."""
    resolved_api_key = _clean_string(
        settings.GEMINI_API_KEY if api_key is None else api_key
    )
    resolved_subagent_model = _clean_string(
        getattr(settings, "GEMINI_SUBAGENT_MODEL", DEFAULT_SUBAGENT_MODEL)
        if subagent_model is None
        else subagent_model
    )
    resolved_primary_model = _clean_string(
        getattr(settings, "GEMINI_PRIMARY_MODEL", DEFAULT_PRIMARY_MODEL)
        if primary_model is None
        else primary_model
    )
    resolved_subagent_temperature = _resolve_temperature(
        subagent_temperature
        if subagent_temperature is not None
        else getattr(
            settings,
            "GEMINI_SUBAGENT_TEMPERATURE",
            DEFAULT_SUBAGENT_TEMPERATURE,
        ),
        DEFAULT_SUBAGENT_TEMPERATURE,
    )
    resolved_primary_temperature = _resolve_temperature(
        primary_temperature
        if primary_temperature is not None
        else getattr(
            settings,
            "GEMINI_PRIMARY_TEMPERATURE",
            DEFAULT_PRIMARY_TEMPERATURE,
        ),
        DEFAULT_PRIMARY_TEMPERATURE,
    )

    if not resolved_api_key:
        raise ModelConfigurationError("Missing Gemini API key. Set GEMINI_API_KEY.")
    if not resolved_subagent_model:
        raise ModelConfigurationError(
            "Missing subagent model ID. Set GEMINI_SUBAGENT_MODEL."
        )
    if not resolved_primary_model:
        raise ModelConfigurationError(
            "Missing primary model ID. Set GEMINI_PRIMARY_MODEL."
        )

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except Exception as exc:  # pragma: no cover - import-path validation
        raise ModelConfigurationError(
            "langchain-google-genai is required to build reasoning models."
        ) from exc

    subagent = ChatGoogleGenerativeAI(
        model=resolved_subagent_model,
        google_api_key=resolved_api_key,
        temperature=resolved_subagent_temperature,
    )
    primary = ChatGoogleGenerativeAI(
        model=resolved_primary_model,
        google_api_key=resolved_api_key,
        temperature=resolved_primary_temperature,
    )
    return ReasoningModels(
        subagent=subagent,
        primary=primary,
        subagent_model_name=resolved_subagent_model,
        primary_model_name=resolved_primary_model,
    )


def get_reasoning_model(models: ReasoningModels, role: ReasoningRole):
    """Select the model instance matching the requested reasoning role."""
    if role == "subagent":
        return models.subagent
    if role == "primary":
        return models.primary
    raise ValueError(f"Unsupported reasoning role: {role}")


@dataclass(frozen=True, slots=True)
class ChatModelConfig:
    model: Any
    model_name: str
    temperature: float


def build_chat_model(
    *,
    api_key: str | None = None,
    model: str | None = None,
    temperature: float | None = None,
) -> ChatModelConfig:
    """Create a configured Gemini chat model client for session chat streaming."""
    resolved_api_key = _clean_string(
        settings.GEMINI_API_KEY if api_key is None else api_key
    )
    resolved_model = _clean_string(
        getattr(settings, "GEMINI_CHAT_MODEL", DEFAULT_CHAT_MODEL)
        if model is None
        else model
    )
    resolved_temperature = _resolve_temperature(
        temperature
        if temperature is not None
        else getattr(
            settings,
            "GEMINI_CHAT_TEMPERATURE",
            DEFAULT_CHAT_TEMPERATURE,
        ),
        DEFAULT_CHAT_TEMPERATURE,
    )

    if not resolved_api_key:
        raise ModelConfigurationError("Missing Gemini API key. Set GEMINI_API_KEY.")
    if not resolved_model:
        raise ModelConfigurationError("Missing chat model ID. Set GEMINI_CHAT_MODEL.")

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except Exception as exc:  # pragma: no cover - import-path validation
        raise ModelConfigurationError(
            "langchain-google-genai is required to build chat models."
        ) from exc

    configured_model = ChatGoogleGenerativeAI(
        model=resolved_model,
        google_api_key=resolved_api_key,
        temperature=resolved_temperature,
    )
    return ChatModelConfig(
        model=configured_model,
        model_name=resolved_model,
        temperature=resolved_temperature,
    )
