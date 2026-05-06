"""Tests for module 1 arithmetic tools."""

from __future__ import annotations

import unittest

from app.module1.services.tools import (
    ARITHMETIC_TOOLS,
    absolute_value,
    add,
    divide,
    floor_divide,
    modulo,
    multiply,
    power,
    subtract,
)


class Module1ToolsTests(unittest.TestCase):
    """Verify arithmetic tool behavior."""

    def test_arithmetic_operations(self) -> None:
        """Tools should return expected arithmetic results."""
        self.assertEqual(add(2, 3), 5)
        self.assertEqual(multiply(6, 7), 42)
        self.assertEqual(divide(8, 2), 4)
        self.assertEqual(subtract(10, 4), 6)
        self.assertEqual(power(2, 5), 32)
        self.assertEqual(modulo(43, 5), 3)
        self.assertEqual(floor_divide(43, 5), 8)
        self.assertEqual(absolute_value(-9), 9)

    def test_division_tools_reject_zero_divisor(self) -> None:
        """Division-style tools should fail clearly on zero divisors."""
        with self.assertRaisesRegex(ValueError, "divide by zero"):
            divide(1, 0)

        with self.assertRaisesRegex(ValueError, "modulo by zero"):
            modulo(1, 0)

        with self.assertRaisesRegex(ValueError, "floor divide by zero"):
            floor_divide(1, 0)

    def test_tool_registry_contains_all_tools(self) -> None:
        """The graph tool registry should expose every arithmetic helper."""
        tool_names = {tool.__name__ for tool in ARITHMETIC_TOOLS}

        self.assertEqual(
            tool_names,
            {
                "absolute_value",
                "add",
                "divide",
                "floor_divide",
                "modulo",
                "multiply",
                "power",
                "subtract",
            },
        )


if __name__ == "__main__":
    unittest.main()
