"""Tests for module 4 text helpers."""

from __future__ import annotations

import unittest

from app.module4.services.text import clip, dedupe, message_content_to_text, slugify


class Module4TextTests(unittest.TestCase):
    """Verify module 4 text normalization helpers."""

    def test_clip_shortens_to_word_boundary(self) -> None:
        """Long text should be clipped without splitting the last word."""
        self.assertEqual(clip("alpha beta gamma", limit=12), "alpha beta...")

    def test_dedupe_preserves_order(self) -> None:
        """De-duplication should preserve first-seen order."""
        self.assertEqual(dedupe(["a", "b", "a", "", "c"]), ["a", "b", "c"])

    def test_message_content_to_text_handles_structured_blocks(self) -> None:
        """Structured message content should become readable plain text."""
        content = [{"type": "text", "text": "hello"}, {"content": "world"}, "done"]

        self.assertEqual(message_content_to_text(content), "hello\nworld\ndone")

    def test_slugify_returns_stable_section_key(self) -> None:
        """Section titles should become stable lowercase slugs."""
        self.assertEqual(slugify("Risk & Adoption!"), "risk-adoption")
        self.assertEqual(slugify("!!!"), "section")


if __name__ == "__main__":
    unittest.main()
