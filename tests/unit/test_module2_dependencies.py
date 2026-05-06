"""Tests for module 2 dependency helpers."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from app.config.settings import Settings
from app.module2.dependencies import (
    get_chat_model,
    get_chat_model_config,
    get_model_api_key,
    get_required_api_key_name,
    has_model_credentials,
    validate_model_provider,
)
from app.module2.schemas import DEFAULT_MODEL, DEFAULT_MODEL_PROVIDER


class Module2DependencyTests(unittest.TestCase):
    """Verify provider-neutral model setup for module 2."""

    def build_settings(self, **overrides: str | bool) -> Settings:
        """Build isolated test settings.

        Args:
            overrides: Settings values to override.

        Returns:
            Isolated settings for dependency tests.
        """
        defaults = {
            "langchain_chat_model": DEFAULT_MODEL,
            "langchain_model_provider": DEFAULT_MODEL_PROVIDER,
            "openai_api_key": "",
            "anthropic_api_key": "",
            "run_live_llm_tests": False,
        }
        return Settings(_env_file=None, **(defaults | overrides))

    def test_get_chat_model_config_uses_defaults(self) -> None:
        """Chat model config should use module defaults."""
        config = get_chat_model_config(settings=self.build_settings())

        self.assertEqual(config.model, DEFAULT_MODEL)
        self.assertEqual(config.model_provider, DEFAULT_MODEL_PROVIDER)

    def test_get_chat_model_config_prefers_explicit_values(self) -> None:
        """Explicit model values should override settings."""
        config = get_chat_model_config(
            model="claude-3-5-haiku-latest",
            model_provider="anthropic",
            settings=self.build_settings(),
        )

        self.assertEqual(config.model, "claude-3-5-haiku-latest")
        self.assertEqual(config.model_provider, "anthropic")

    def test_provider_credentials_are_checked_by_provider(self) -> None:
        """Credential checks should follow the selected provider."""
        empty_settings = self.build_settings()
        self.assertFalse(has_model_credentials("openai", settings=empty_settings))
        self.assertFalse(has_model_credentials("anthropic", settings=empty_settings))

        anthropic_settings = self.build_settings(anthropic_api_key="test-key")
        self.assertTrue(has_model_credentials("anthropic", settings=anthropic_settings))

    def test_get_required_api_key_name_for_known_providers(self) -> None:
        """Known hosted providers should map to their key names."""
        self.assertEqual(get_required_api_key_name("openai"), "OPENAI_API_KEY")
        self.assertEqual(get_required_api_key_name("anthropic"), "ANTHROPIC_API_KEY")

    def test_get_model_api_key_uses_settings(self) -> None:
        """Model construction should receive the key from settings."""
        settings = self.build_settings(openai_api_key="test-openai-key")

        self.assertEqual(
            get_model_api_key("openai", settings=settings), "test-openai-key"
        )

    def test_unknown_provider_fails_early(self) -> None:
        """Unknown providers should fail before LangChain initialization."""
        with self.assertRaisesRegex(ValueError, "Unsupported model provider 'opneai'"):
            validate_model_provider("opneai")

        with self.assertRaisesRegex(ValueError, "Unsupported model provider 'opneai'"):
            get_chat_model_config(model_provider="opneai")

    def test_get_chat_model_uses_langchain_initializer(self) -> None:
        """The dependency layer should use LangChain's provider-neutral factory."""
        sentinel_model = object()

        with (
            patch(
                "app.module2.dependencies.init_chat_model",
                return_value=sentinel_model,
            ) as init_chat_model,
            patch(
                "app.module2.dependencies.get_settings",
                return_value=self.build_settings(anthropic_api_key="test-key"),
            ),
        ):
            model = get_chat_model(
                model="claude-3-5-haiku-latest",
                model_provider="anthropic",
            )

        self.assertIs(model, sentinel_model)
        init_chat_model.assert_called_once_with(
            model="claude-3-5-haiku-latest",
            model_provider="anthropic",
            api_key="test-key",
        )


if __name__ == "__main__":
    unittest.main()
