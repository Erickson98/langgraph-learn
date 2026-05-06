"""Module 1 arithmetic assistant package."""

from __future__ import annotations

import warnings

from langchain_core._api.deprecation import LangChainPendingDeprecationWarning

warnings.filterwarnings(
    "ignore",
    message="The default value of `allowed_objects` will change.*",
    category=LangChainPendingDeprecationWarning,
    module="langgraph.checkpoint.serde.encrypted",
)
