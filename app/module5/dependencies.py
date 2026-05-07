"""Dependency setup for module 5 memory workflows."""

from __future__ import annotations

from langchain.chat_models import init_chat_model
from langchain_core.language_models.chat_models import BaseChatModel

from app.config.settings import Settings, get_settings
from app.module5.schemas import DEFAULT_MODEL, DEFAULT_MODEL_PROVIDER, ChatModelConfig

PROVIDER_API_KEYS = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}
PROVIDER_API_KEY_FIELDS = {
    "anthropic": "anthropic_api_key",
    "openai": "openai_api_key",
}


def prepare_environment(settings: Settings | None = None) -> Settings:
    """Return configured settings for CLI and service usage.

    Args:
        settings: Optional settings override for tests.

    Returns:
        Runtime settings loaded through Pydantic.
    """
    return settings or get_settings()


def get_chat_model_config(
    model: str | None = None,
    model_provider: str | None = None,
    settings: Settings | None = None,
) -> ChatModelConfig:
    """Resolve chat model configuration from arguments or environment.

    Args:
        model: Optional explicit chat model name.
        model_provider: Optional explicit LangChain model provider.
        settings: Optional settings override for tests.

    Returns:
        Resolved chat model configuration.
    """
    resolved_settings = prepare_environment(settings)
    config = ChatModelConfig(
        model=model or resolved_settings.langchain_chat_model or DEFAULT_MODEL,
        model_provider=model_provider
        or resolved_settings.langchain_model_provider
        or DEFAULT_MODEL_PROVIDER,
    )
    validate_model_provider(config.model_provider)
    return config


def validate_model_provider(model_provider: str) -> None:
    """Validate that the provider is explicitly supported by this module.

    Args:
        model_provider: LangChain model provider name.

    Raises:
        ValueError: If the provider is unknown.
    """
    if model_provider not in PROVIDER_API_KEYS:
        supported_providers = sorted(PROVIDER_API_KEYS)
        raise ValueError(
            f"Unsupported model provider '{model_provider}'. "
            f"Supported providers: {', '.join(supported_providers)}."
        )


def get_required_api_key_name(model_provider: str | None = None) -> str | None:
    """Return the expected API key variable for known hosted providers.

    Args:
        model_provider: LangChain model provider name.

    Returns:
        Environment variable name for supported hosted providers.
    """
    provider = model_provider or get_chat_model_config().model_provider
    validate_model_provider(provider)
    return PROVIDER_API_KEYS.get(provider)


def has_model_credentials(
    model_provider: str | None = None,
    settings: Settings | None = None,
) -> bool:
    """Return whether required credentials are present for the provider.

    Args:
        model_provider: Optional LangChain model provider name.
        settings: Optional settings override for tests.

    Returns:
        True when the provider API key is set.
    """
    resolved_settings = prepare_environment(settings)
    provider = (
        model_provider
        or get_chat_model_config(settings=resolved_settings).model_provider
    )
    validate_model_provider(provider)
    return bool(getattr(resolved_settings, PROVIDER_API_KEY_FIELDS[provider]))


def get_model_api_key(
    model_provider: str,
    settings: Settings | None = None,
) -> str:
    """Return the configured API key for a supported model provider.

    Args:
        model_provider: LangChain model provider name.
        settings: Optional settings override for tests.

    Returns:
        API key value for the provider.
    """
    validate_model_provider(model_provider)
    resolved_settings = prepare_environment(settings)
    return getattr(resolved_settings, PROVIDER_API_KEY_FIELDS[model_provider])


def get_chat_model(
    model: str | None = None,
    model_provider: str | None = None,
    settings: Settings | None = None,
) -> BaseChatModel:
    """Build the chat model used by the module 5 graph.

    Args:
        model: Optional chat model name.
        model_provider: Optional LangChain model provider name.
        settings: Optional settings override for tests.

    Returns:
        Configured LangChain chat model.
    """
    resolved_settings = prepare_environment(settings)
    config = get_chat_model_config(
        model=model,
        model_provider=model_provider,
        settings=resolved_settings,
    )
    return init_chat_model(
        model=config.model,
        model_provider=config.model_provider,
        api_key=get_model_api_key(config.model_provider, settings=resolved_settings),
    )
