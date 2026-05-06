"""Shared logging helpers."""

from __future__ import annotations

import logging

LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"


def configure_logging(level: str = "INFO") -> None:
    """Configure application logging.

    Args:
        level: Logging level name.
    """
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=LOG_FORMAT,
    )


def get_logger(name: str) -> logging.Logger:
    """Return a logger for the given module.

    Args:
        name: Logger name, usually `__name__`.

    Returns:
        Configured logger instance.
    """
    return logging.getLogger(name)
