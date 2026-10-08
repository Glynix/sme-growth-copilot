"""Assemble a single allowlisted observed-facts object for a merchant."""

from __future__ import annotations

from datetime import date

import pandas as pd

from sme_growth.analytics.contracts import MerchantFacts
from sme_growth.analytics.customers import get_lapsed_customer_summary, get_repeat_customer_analysis
from sme_growth.analytics.peers import get_peer_benchmark
from sme_growth.analytics.performance import get_daypart_analysis, get_sales_performance


def get_merchant_facts(
    transactions: pd.DataFrame, merchants: pd.DataFrame,
    program_eligibility: pd.DataFrame, merchant_id: str, as_of_date: date,
) -> MerchantFacts:
    """Compose safe observed facts; this does not recommend an action."""

    focal = merchants.loc[merchants["merchant_id"].eq(merchant_id), "merchant_name"]
    if len(focal) != 1:
        raise ValueError("Merchant context is unavailable")
    sales = get_sales_performance(transactions, merchant_id, as_of_date)
    dayparts = get_daypart_analysis(transactions, merchant_id, as_of_date)
    repeat = get_repeat_customer_analysis(transactions, merchant_id, as_of_date)
    lapsed = get_lapsed_customer_summary(transactions, program_eligibility, merchant_id, as_of_date)
    peers = get_peer_benchmark(transactions, merchants, merchant_id, as_of_date)
    weekday_lunch = None
    lunch_lost = None
    driver = None
    driver_delta = None
    if dayparts.segments is not None:
        lunch = next(segment for segment in dayparts.segments if segment.segment == "weekday_lunch")
        weekday_lunch = lunch.transaction_count_change_pct
        lunch_lost = max(0, lunch.baseline_transactions - lunch.current_transactions)
        declining = [segment for segment in dayparts.segments if segment.sales_delta_aed < 0]
        if declining:
            largest = min(declining, key=lambda segment: segment.sales_delta_aed)
            driver = largest.segment
            driver_delta = largest.sales_delta_aed
    return MerchantFacts(
        merchant_name=str(focal.iloc[0]),
        as_of_date=as_of_date.isoformat(),
        baseline_window=sales.meta.baseline_window,
        current_window=sales.meta.current_window,
        qualification_window=lapsed.qualification_window,
        evidence_type="observed_synthetic_transactions_and_simulated_eligibility",
        sales_change_pct=sales.sales_change_pct,
        transaction_count_change_pct=sales.transaction_count_change_pct,
        average_ticket_change_pct=sales.average_ticket_change_pct,
        largest_driver=driver,
        largest_driver_sales_delta_aed=driver_delta,
        weekday_lunch_change_pct=weekday_lunch,
        weekday_lunch_transactions_lost=lunch_lost,
        repeat_customer_change_pct=repeat.repeat_customers.change_pct,
        peer_change_pct=peers.median_sales_change_pct,
        peer_count=peers.peer_count,
        lapsed_candidate_customers=lapsed.lapsed_candidate_customers,
        lapsed_regular_customers=lapsed.eligible_lapsed_customers,
        historical_lapsed_spend=lapsed.historical_monthly_spend_aed,
    )
