"""Text normalization helpers for module 4."""

from __future__ import annotations

import re


def clip(text: str, limit: int = 1400) -> str:
    """Clip long text to a word boundary.

    Args:
        text: Text to clip.
        limit: Maximum character length.

    Returns:
        Clipped text.
    """
    normalized = text.strip()
    if len(normalized) <= limit:
        return normalized
    return normalized[:limit].rsplit(" ", 1)[0] + "..."


def dedupe(items: list[str]) -> list[str]:
    """Return items in original order without duplicates.

    Args:
        items: Values to de-duplicate.

    Returns:
        Unique values in first-seen order.
    """
    seen: set[str] = set()
    output: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            output.append(item)
    return output


def message_content_to_text(content: object) -> str:
    """Normalize LangChain message content into plain text.

    Args:
        content: Message content returned by a chat model.

    Returns:
        Text representation of the message content.
    """
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                value = item.get("text", item.get("content", item))
                parts.append(str(value))
            else:
                parts.append(str(item))
        return "\n".join(parts)

    return str(content)


def slugify(text: str) -> str:
    """Convert text into a stable section key.

    Args:
        text: Text to slugify.

    Returns:
        URL-style slug.
    """
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug or "section"
