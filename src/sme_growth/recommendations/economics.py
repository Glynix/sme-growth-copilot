"""Transparent scenario arithmetic; these are assumptions, not forecasts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ImpactAssumptions:
    addressable_count: int
    reach_rate: float
    response_rate_low: float
    response_rate_high: float
    visits_per_response: float
    spend_per_visit_aed: float
    incentive_rate: float
    all_qualifying_redemption_rate_low: float
    all_qualifying_redemption_rate_high: float
    response_meaning: str


@dataclass(frozen=True, slots=True)
class EstimatedImpactRange:
    assumptions: ImpactAssumptions
    incremental_sales_low_aed: float
    incremental_sales_high_aed: float
    campaign_cost_low_aed: float
    campaign_cost_high_aed: float
    net_additional_sales_low_aed: float
    net_additional_sales_high_aed: float
    evidence_type: str = "assumption_based_scenario"
    note: str = (
        "The range is not a confidence interval. Gross additional sales use assumed "
        "response, visits and spend; incentives apply to all qualifying redemptions. "
        "Net additional sales subtract those discounts; profit and ROI are not modeled. "
        "Historical spend is context, not recoverable revenue."
    )


def calculate_impact(assumptions: ImpactAssumptions) -> EstimatedImpactRange:
    """Calculate additional sales after all assumed qualifying discounts."""

    if assumptions.addressable_count < 0 or assumptions.visits_per_response < 0 or assumptions.spend_per_visit_aed < 0:
        raise ValueError("Counts, visits and spend must be non-negative")
    rates = (
        assumptions.reach_rate, assumptions.response_rate_low, assumptions.response_rate_high,
        assumptions.incentive_rate,
        assumptions.all_qualifying_redemption_rate_low,
        assumptions.all_qualifying_redemption_rate_high,
    )
    if any(rate < 0 or rate > 1 for rate in rates):
        raise ValueError("Rates must be between zero and one")
    if assumptions.response_rate_low > assumptions.response_rate_high:
        raise ValueError("Response range is reversed")
    if assumptions.all_qualifying_redemption_rate_low > assumptions.all_qualifying_redemption_rate_high:
        raise ValueError("Redemption range is reversed")

    reached = assumptions.addressable_count * assumptions.reach_rate
    per_responder_sales = assumptions.visits_per_response * assumptions.spend_per_visit_aed
    sales_low = reached * assumptions.response_rate_low * per_responder_sales
    sales_high = reached * assumptions.response_rate_high * per_responder_sales
    incentive_base = reached * per_responder_sales * assumptions.incentive_rate
    cost_low = incentive_base * assumptions.all_qualifying_redemption_rate_low
    cost_high = incentive_base * assumptions.all_qualifying_redemption_rate_high
    net_low = sales_low - cost_high
    net_high = sales_high - cost_low
    return EstimatedImpactRange(
        assumptions=assumptions,
        incremental_sales_low_aed=round(sales_low, 2),
        incremental_sales_high_aed=round(sales_high, 2),
        campaign_cost_low_aed=round(cost_low, 2),
        campaign_cost_high_aed=round(cost_high, 2),
        net_additional_sales_low_aed=round(net_low, 2),
        net_additional_sales_high_aed=round(net_high, 2),
    )
