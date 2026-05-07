"""Arithmetic tools used by the module 3 breakpoint demos."""

from __future__ import annotations


def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


def subtract(a: int, b: int) -> int:
    """Subtract ``b`` from ``a``."""
    return a - b


def divide(a: int, b: int) -> float:
    """Divide ``a`` by ``b``.

    Args:
        a: Dividend.
        b: Divisor.

    Returns:
        Division result.

    Raises:
        ValueError: If the divisor is zero.
    """
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b


ARITHMETIC_TOOLS = [add, multiply, subtract, divide]
