"""Explain the arithmetic behind a released sales comparison."""

from __future__ import annotations

from dataclasses import dataclass

from sme_growth.analytics.contracts import SalesPerformance


@dataclass(frozen=True, slots=True)
class SalesComponents:
    transaction_effect_aed: float
    ticket_effect_aed: float
    verdict: str


def component_verdict(count_effect: float, ticket_effect: float) -> str:
    """Describe observed accounting effects without implying a behavioral cause."""

    if count_effect < 0 and ticket_effect < 0:
        if count_effect == ticket_effect:
            return "Fewer purchases and a lower average ticket each contributed about the same amount to the sales drop."
        larger = "fewer purchases" if count_effect < ticket_effect else "a lower average ticket"
        return f"{larger.capitalize()} mattered more, though both contributed to the sales drop."
    if count_effect < 0:
        return "Fewer purchases pulled sales down, but the average ticket offset some or all of that."
    if ticket_effect < 0:
        return "A lower average ticket pulled sales down, but purchase count offset some or all of that."
    return "Neither purchase count nor average ticket pulled sales down in this comparison."


def analyze_sales_components(sales: SalesPerformance) -> SalesComponents | None:
    """Split the sales delta symmetrically between purchase count and average ticket.

    This is an accounting decomposition, not a causal estimate. The midpoint
    formula avoids assigning the interaction entirely to either component.
    """

    if sales.status != "available" or any(value is None for value in (
        sales.baseline_sales_aed, sales.current_sales_aed,
        sales.baseline_transactions, sales.current_transactions,
        sales.transaction_count_change_pct, sales.average_ticket_change_pct,
    )):
        return None
    before_count = sales.baseline_transactions
    after_count = sales.current_transactions
    if before_count <= 0 or after_count <= 0:
        return None
    before_ticket = sales.baseline_sales_aed / before_count
    after_ticket = sales.current_sales_aed / after_count
    count_effect = round((after_count - before_count) * (before_ticket + after_ticket) / 2, 2)
    ticket_effect = round(sales.current_sales_aed - sales.baseline_sales_aed - count_effect, 2)

    return SalesComponents(count_effect, ticket_effect, component_verdict(count_effect, ticket_effect))
