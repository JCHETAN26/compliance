"""Exact money arithmetic for LLM budgets.

Amounts are integer micro-dollars (1 micro-USD = 1e-6 USD). Prices are USD per million
tokens, so ``tokens * price`` is already in micro-dollars. Every conversion from a
computed (possibly fractional) amount rounds **up**, so rounding can never free budget
that was actually consumed.
"""

from __future__ import annotations

from decimal import ROUND_CEILING, Decimal

MICRO_PER_USD = 1_000_000


def _dec(value: float | int | str | Decimal) -> Decimal:
    if isinstance(value, Decimal):
        return value
    # str() keeps the shortest exact decimal form of a float (0.1 -> "0.1").
    return Decimal(str(value))


def usd_to_micro_ceil(usd: float | int | str | Decimal) -> int:
    amount = _dec(usd)
    if amount.is_nan() or amount.is_infinite() or amount < 0:
        raise ValueError(f"invalid USD amount {usd!r}")
    return int((amount * MICRO_PER_USD).to_integral_value(rounding=ROUND_CEILING))


def micro_to_usd(micro: int) -> float:
    return micro / MICRO_PER_USD


def token_cost_micro(
    input_tokens: int,
    output_tokens: int,
    input_usd_per_mtok: float | Decimal,
    output_usd_per_mtok: float | Decimal,
) -> int:
    """Cost in micro-dollars, rounded up."""
    if input_tokens < 0 or output_tokens < 0:
        raise ValueError("token counts must be non-negative")
    exact = input_tokens * _dec(input_usd_per_mtok) + output_tokens * _dec(output_usd_per_mtok)
    return int(exact.to_integral_value(rounding=ROUND_CEILING))


def format_usd(micro: int) -> str:
    return f"${micro / MICRO_PER_USD:.6f}" if micro else "$0"


__all__ = [
    "MICRO_PER_USD",
    "format_usd",
    "micro_to_usd",
    "token_cost_micro",
    "usd_to_micro_ceil",
]
