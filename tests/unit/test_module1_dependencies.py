"""Tests for module 1 dependency helpers."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from app.config.settings import Settings
from app.module1.dependencies import (
    get_chat_model,
    get_chat_model_config,
    get_model_api_key,
    get_required_api_key_name,
    has_model_credentials,
    validate_model_provider,
)
from app.module1.schemas import (
    DEFAULT_MODEL,
    DEFAULT_MODEL_PROVIDER,
)


class Module1DependencyTests(unittest.TestCase):
    """Verify environment loading behavior."""

    def build_settings(self, **overrides: str | bool) -> Settings:
        """Build test settings without inheriting process environment values.

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
        """Chat model config should default through LangChain settings."""
        config = get_chat_model_config(settings=self.build_settings())

        self.assertEqual(config.model, DEFAULT_MODEL)
        self.assertEqual(config.model_provider, DEFAULT_MODEL_PROVIDER)

    def test_get_chat_model_config_uses_settings(self) -> None:
        """Settings should configure provider-agnostic model setup."""
        config = get_chat_model_config(
            settings=Settings(
                _env_file=None,
                openai_api_key="",
                anthropic_api_key="",
                langchain_chat_model="claude-3-5-haiku-latest",
                langchain_model_provider="anthropic",
            )
        )

        self.assertEqual(config.model, "claude-3-5-haiku-latest")
        self.assertEqual(config.model_provider, "anthropic")

    def test_get_chat_model_config_prefers_explicit_values(self) -> None:
        """Explicit model values should override environment variables."""
        config = get_chat_model_config(
            model="claude-3-5-haiku-latest",
            model_provider="anthropic",
            settings=Settings(
                _env_file=None,
                openai_api_key="",
                anthropic_api_key="",
                langchain_chat_model="gpt-4o-mini",
                langchain_model_provider="openai",
            ),
        )

        self.assertEqual(config.model, "claude-3-5-haiku-latest")
        self.assertEqual(config.model_provider, "anthropic")

    def test_provider_credentials_are_checked_by_provider(self) -> None:
        """Credential checks should follow the configured provider."""
        empty_settings = self.build_settings()
        self.assertFalse(has_model_credentials("openai", settings=empty_settings))
        self.assertFalse(has_model_credentials("anthropic", settings=empty_settings))

        anthropic_settings = self.build_settings(anthropic_api_key="test-key")
        self.assertTrue(has_model_credentials("anthropic", settings=anthropic_settings))

    def test_get_required_api_key_name_for_known_providers(self) -> None:
        """Known hosted providers should map to their expected key names."""
        self.assertEqual(get_required_api_key_name("openai"), "OPENAI_API_KEY")
        self.assertEqual(get_required_api_key_name("anthropic"), "ANTHROPIC_API_KEY")

    def test_get_model_api_key_uses_settings(self) -> None:
        """Model construction should use API keys loaded through settings."""
        settings = self.build_settings(openai_api_key="test-openai-key")

        self.assertEqual(
            get_model_api_key("openai", settings=settings), "test-openai-key"
        )

    def test_unknown_provider_fails_early(self) -> None:
        """Unknown providers should fail with an actionable error."""
        with self.assertRaisesRegex(ValueError, "Unsupported model provider 'opneai'"):
            validate_model_provider("opneai")

        with self.assertRaisesRegex(ValueError, "Unsupported model provider 'opneai'"):
            get_chat_model_config(model_provider="opneai")

    def test_get_chat_model_uses_langchain_initializer(self) -> None:
        """The dependency layer should use LangChain's provider-neutral factory."""
        sentinel_model = object()

        with (
            patch(
                "app.module1.dependencies.init_chat_model",
                return_value=sentinel_model,
            ) as init_chat_model,
            patch(
                "app.module1.dependencies.get_settings",
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

    def test_settings_loads_env_without_overwriting_existing(self) -> None:
        """BaseSettings should load .env values and preserve environment overrides."""
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            env_path.write_text(
                "\n".join(
                    [
                        "# comment",
                        "OPENAI_API_KEY='from-file'",
                        "ANTHROPIC_API_KEY='from-file-anthropic'",
                        'EXAMPLE_VALUE="quoted"',
                        "SPACED_VALUE='value with spaces'",
                        "IGNORED_LINE",
                    ]
                )
            )

            with patch.dict(
                os.environ,
                {"OPENAI_API_KEY": "already-set"},
                clear=True,
            ):
                settings = Settings(_env_file=env_path)

                self.assertEqual(settings.openai_api_key, "already-set")
                self.assertEqual(settings.anthropic_api_key, "from-file-anthropic")


if __name__ == "__main__":
    unittest.main()
