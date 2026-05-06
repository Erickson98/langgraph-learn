"""Arithmetic tools exposed to the module 1 LangGraph assistant."""

from __future__ import annotations


def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: First integer.
        b: Second integer.

    Returns:
        Sum of the two integers.
    """
    return a + b


def multiply(a: int, b: int) -> int:
    """Multiplies a and b.

    Args:
        a: First integer.
        b: Second integer.

    Returns:
        Product of the two integers.
    """
    return a * b


def divide(a: int, b: int) -> float:
    """Divides a by b.

    Args:
        a: Numerator.
        b: Denominator.

    Returns:
        Quotient of a divided by b.

    Raises:
        ValueError: If b is zero.
    """
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b


def subtract(a: int, b: int) -> int:
    """Subtracts b from a.

    Args:
        a: First integer.
        b: Integer to subtract.

    Returns:
        Difference of a and b.
    """
    return a - b


def power(a: int, b: int) -> int:
    """Raises a to the power of b.

    Args:
        a: Base integer.
        b: Exponent integer.

    Returns:
        a raised to the b power.
    """
    return a**b


def modulo(a: int, b: int) -> int:
    """Returns the remainder of a divided by b.

    Args:
        a: First integer.
        b: Divisor integer.

    Returns:
        Remainder after division.

    Raises:
        ValueError: If b is zero.
    """
    if b == 0:
        raise ValueError("Cannot take modulo by zero.")
    return a % b


def floor_divide(a: int, b: int) -> int:
    """Divides a by b and returns the integer quotient.

    Args:
        a: First integer.
        b: Divisor integer.

    Returns:
        Integer quotient after division.

    Raises:
        ValueError: If b is zero.
    """
    if b == 0:
        raise ValueError("Cannot floor divide by zero.")
    return a // b


def absolute_value(a: int) -> int:
    """Returns the absolute value of a.

    Args:
        a: Input integer.

    Returns:
        Absolute value of a.
    """
    return abs(a)


ARITHMETIC_TOOLS = [
    add,
    multiply,
    divide,
    subtract,
    power,
    modulo,
    floor_divide,
    absolute_value,
]
