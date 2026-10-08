"""Private token joins that return aggregate customer trends only."""

from __future__ import annotations

from datetime import date

import pandas as pd

from sme_growth.analytics.common import comparison_meta, percent_change, settled_transactions
from sme_growth.analytics.contracts import CustomerTrend, LapsedCustomerSummary, RepeatCustomerAnalysis, Window
from sme_growth.audiences.private_rules import select_private_lapsed_lunch


MIN_PUBLIC_CUSTOMERS = 10
MIN_ELIGIBLE_LAPSED = 50


def _trend(baseline: int, current: int) -> CustomerTrend:
    if baseline < MIN_PUBLIC_CUSTOMERS or current < MIN_PUBLIC_CUSTOMERS:
        return CustomerTrend(None, None, None, "suppressed")
    return CustomerTrend(baseline, current, percent_change(current, baseline), "available")


def get_repeat_customer_analysis(
    transactions: pd.DataFrame, merchant_id: str, as_of_date: date,
) -> RepeatCustomerAnalysis:
    """Count distinct new and previously observed customers in fixed windows."""

    meta = comparison_meta(as_of_date)
    settled = settled_transactions(transactions)
    merchant = settled[
        settled["merchant_id"].eq(merchant_id)
        & settled["transaction_datetime"].lt(pd.Timestamp(as_of_date))
    ]
    first_seen = merchant.groupby("synthetic_customer_id")["transaction_datetime"].min()

    def cohorts(start: str, end: str) -> tuple[int, int]:
        start_at, end_at = pd.Timestamp(start), pd.Timestamp(end)
        seen = merchant[
            merchant["transaction_datetime"].ge(start_at)
            & merchant["transaction_datetime"].lt(end_at)
        ]["synthetic_customer_id"].unique()
        first = first_seen.reindex(seen)
        return int(first.ge(start_at).sum()), int(first.lt(start_at).sum())

    base_new, base_repeat = cohorts(meta.baseline_window.start, meta.baseline_window.end_exclusive)
    current_new, current_repeat = cohorts(meta.current_window.start, meta.current_window.end_exclusive)
    return RepeatCustomerAnalysis(
        meta, _trend(base_new, current_new), _trend(base_repeat, current_repeat),
    )


def get_lapsed_customer_summary(
    transactions: pd.DataFrame, program_eligibility: pd.DataFrame,
    merchant_id: str, as_of_date: date,
) -> LapsedCustomerSummary:
    """Summarize regulars absent for 45+ days and their eligible subset.

    This is an aggregate diagnostic calculation, not an Audience Builder:
    no membership, handle, or individual eligibility result is returned.
    """

    cohort = select_private_lapsed_lunch(transactions, program_eligibility, merchant_id, as_of_date)
    window = Window(cohort.qualification_start.isoformat(), cohort.qualification_end_exclusive.isoformat())

    def suppressed(note: str) -> LapsedCustomerSummary:
        return LapsedCustomerSummary(
            as_of_date.isoformat(), window, 45, 3, 2,
            None, None, None, "suppressed", note,
        )

    if not cohort.program_data_complete:
        return suppressed("Program eligibility is unavailable or incomplete.")
    if len(cohort.candidates) < MIN_PUBLIC_CUSTOMERS:
        return suppressed("Lapsed regular summary has too few contributors.")
    if len(cohort.eligible_members) < MIN_ELIGIBLE_LAPSED or len(cohort.candidates) - len(cohort.eligible_members) < MIN_PUBLIC_CUSTOMERS:
        return suppressed("Audience or its complement is below the release threshold.")
    return LapsedCustomerSummary(
        as_of_date.isoformat(), window, 45, 3, 2,
        len(cohort.candidates), len(cohort.eligible_members), round(cohort.historical_monthly_spend_aed, 2),
        "available",
        "Historical cohort spend is a 90-day net-sales total divided by three; it is context, not recoverable revenue.",
    )
