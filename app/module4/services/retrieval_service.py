"""Retrieval helpers for module 4 research sections."""

from __future__ import annotations

from pydantic import SecretStr
from langchain_community.document_loaders import WikipediaLoader

try:
    from langchain_tavily import TavilySearch
except ImportError:  # pragma: no cover
    TavilySearch = None

from app.config.settings import Settings
from app.logging import get_logger
from app.module4.schemas import RetrievalContext
from app.module4.services.text import clip

logger = get_logger(__name__)


def search_wikipedia(
    query: str,
    *,
    enabled: bool = True,
    load_max_docs: int = 2,
    label_prefix: str = "W",
) -> RetrievalContext:
    """Search Wikipedia for background context.

    Args:
        query: Wikipedia-style query.
        enabled: Whether Wikipedia retrieval should run.
        load_max_docs: Maximum documents to load.
        label_prefix: Prefix used to label returned sources.

    Returns:
        Retrieved context blocks and source labels.
    """
    if not enabled:
        return RetrievalContext(
            context_blocks=["Wikipedia search disabled for this request."],
            source_items=[],
        )

    try:
        docs = WikipediaLoader(query=query, load_max_docs=load_max_docs).load()
    except Exception:
        logger.warning("Wikipedia lookup failed for query=%s", query, exc_info=True)
        return RetrievalContext(
            context_blocks=["Wikipedia lookup failed; continuing without that source."],
            source_items=[],
        )

    blocks: list[str] = []
    sources: list[str] = []
    for index, document in enumerate(docs, start=1):
        label = f"{label_prefix}{index}"
        source = document.metadata.get("source", query)
        blocks.append(f"[{label}] Source: {source}\n{clip(document.page_content)}")
        sources.append(f"[{label}] {source}")

    if not blocks:
        blocks.append(f"Wikipedia returned no useful results for '{query}'.")

    return RetrievalContext(context_blocks=blocks, source_items=sources)


def search_web(
    query: str,
    *,
    settings: Settings,
    enabled: bool = True,
    max_results: int = 3,
    label_prefix: str = "T",
) -> RetrievalContext:
    """Search the web through Tavily when configured.

    Args:
        query: Search-engine style query.
        settings: Runtime settings with optional Tavily key.
        enabled: Whether web retrieval should run.
        max_results: Maximum search results.
        label_prefix: Prefix used to label returned sources.

    Returns:
        Retrieved context blocks and source labels.
    """
    if not enabled:
        return RetrievalContext(
            context_blocks=["Web search disabled for this request."],
            source_items=[],
        )

    if TavilySearch is None:
        return RetrievalContext(
            context_blocks=["Tavily package is not installed; skipping web search."],
            source_items=[],
        )

    if not settings.tavily_api_key:
        return RetrievalContext(
            context_blocks=["TAVILY_API_KEY is not set; skipping web search."],
            source_items=[],
        )

    try:
        tavily = TavilySearch(
            max_results=max_results,
            tavily_api_key=SecretStr(settings.tavily_api_key),
        )
        data = tavily.invoke({"query": query})
        results = data.get("results", data)
    except Exception:
        logger.warning("Web search failed for query=%s", query, exc_info=True)
        return RetrievalContext(
            context_blocks=["Web search failed; continuing without that source."],
            source_items=[],
        )

    blocks: list[str] = []
    sources: list[str] = []
    for index, item in enumerate(results, start=1):
        label = f"{label_prefix}{index}"
        url = item.get("url", "unknown-url")
        content = clip(item.get("content", ""))
        if content:
            blocks.append(f"[{label}] Source: {url}\n{content}")
            sources.append(f"[{label}] {url}")

    if not blocks:
        blocks.append(f"Web search returned no useful results for '{query}'.")

    return RetrievalContext(context_blocks=blocks, source_items=sources)
