"""Test package for langgraph learning modules."""

from __future__ import annotations

import os
import warnings

from langchain_core._api.deprecation import LangChainPendingDeprecationWarning

os.environ["LANGCHAIN_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGSMITH_TRACING_V2"] = "false"

warnings.filterwarnings(
    "ignore",
    category=LangChainPendingDeprecationWarning,
)
