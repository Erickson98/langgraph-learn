"""Tests for module 4 dependency helpers."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from app.config.settings import Settings
from app.module4.dependencies import (
    get_chat_model,
    get_chat_model_config,
    get_model_api_key,
    get_required_api_key_name,
    has_model_credentials,
    validate_model_provider,
)


class Module4DependencyTests(unittest.TestCase):
    """Verify module 4 model and provider configuration helpers."""

    def test_get_chat_model_config_uses_settings_defaults(self) -> None:
        """Model config should resolve from shared settings."""
        settings = Settings(
            _env_file=None,
            langchain_chat_model="claude-3-5-haiku-latest",
            langchain_model_provider="anthropic",
        )

        config = get_chat_model_config(settings=settings)

        self.assertEqual(config.model, "claude-3-5-haiku-latest")
        self.assertEqual(config.model_provider, "anthropic")

    def test_validate_model_provider_rejects_unknown_provider(self) -> None:
        """Unknown providers should fail early with a clear error."""
        with self.assertRaisesRegex(ValueError, "Unsupported model provider"):
            validate_model_provider("opneai")

    def test_required_api_key_name_is_provider_specific(self) -> None:
        """Provider names should map to their expected environment variables."""
        self.assertEqual(get_required_api_key_name("openai"), "OPENAI_API_KEY")
        self.assertEqual(get_required_api_key_name("anthropic"), "ANTHROPIC_API_KEY")

    def test_has_model_credentials_reads_settings(self) -> None:
        """Credential checks should use Pydantic settings values."""
        self.assertTrue(
            has_model_credentials(
                "openai",
                settings=Settings(_env_file=None, openai_api_key="test-key"),
            )
        )
        self.assertFalse(
            has_model_credentials(
                "anthropic",
                settings=Settings(_env_file=None, anthropic_api_key=""),
            )
        )

    def test_get_model_api_key_returns_configured_key(self) -> None:
        """API key lookup should return the provider-specific settings value."""
        self.assertEqual(
            get_model_api_key(
                "openai",
                settings=Settings(_env_file=None, openai_api_key="test-key"),
            ),
            "test-key",
        )

    def test_get_chat_model_uses_langchain_initializer(self) -> None:
        """Chat model construction should go through LangChain's abstraction."""
        settings = Settings(
            _env_file=None,
            langchain_chat_model="gpt-4o-mini",
            langchain_model_provider="openai",
            openai_api_key="test-key",
        )

        with patch("app.module4.dependencies.init_chat_model") as init_chat_model:
            get_chat_model(settings=settings)

        init_chat_model.assert_called_once_with(
            model="gpt-4o-mini",
            model_provider="openai",
            api_key="test-key",
        )


if __name__ == "__main__":
    unittest.main()
