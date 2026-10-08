"""Observed sales, weekday, and fixed daypart comparisons."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from sme_growth.analytics.common import (
    comparison_meta, contributors_safe, merchant_window, percent_change,
    settled_transactions,
)
from sme_growth.analytics.contracts import BreakdownResult, SalesPerformance, SegmentComparison, WeeklySalesPulse, Window


WEEKDAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
DAYPARTS = (
    "weekday_breakfast", "weekday_lunch", "weekday_afternoon",
    "weekday_evening", "weekend_all",
)


def get_sales_performance(transactions: pd.DataFrame, merchant_id: str, as_of_date: date) -> SalesPerformance:
    """Compare two complete 28-day periods with identical weekday composition."""

    meta = comparison_meta(as_of_date)
    settled = settled_transactions(transactions)
    baseline = merchant_window(settled, merchant_id, meta.baseline_window)
    current = merchant_window(settled, merchant_id, meta.current_window)
    if not contributors_safe(baseline) or not contributors_safe(current):
        return SalesPerformance(meta, None, None, None, None, None, None, None, None, None,
                                "suppressed", "Insufficient distinct contributors in one or both periods.")
    baseline_sales = float(baseline["net_sales_aed"].sum())
    current_sales = float(current["net_sales_aed"].sum())
    baseline_count = len(baseline)
    current_count = len(current)
    baseline_ticket = baseline_sales / baseline_count
    current_ticket = current_sales / current_count
    return SalesPerformance(
        meta, round(baseline_sales, 2), round(current_sales, 2),
        percent_change(current_sales, baseline_sales),
        baseline_count, current_count, percent_change(current_count, baseline_count),
        round(baseline_ticket, 2), round(current_ticket, 2),
        percent_change(current_ticket, baseline_ticket),
        "available", "Net sales exclude declines and full refunds; partial refunds reduce sales. Changes are observed, not causal.",
    )


def get_weekly_sales_pulse(
    transactions: pd.DataFrame, merchant_id: str, as_of_date: date,
) -> WeeklySalesPulse:
    """Compare the last complete week with the prior week and prior four-week average.

    All five fixed seven-day source cells must be safe. No arbitrary date cuts
    or peer, audience, or recommendation data are released here.
    """

    if not isinstance(as_of_date, date):
        raise TypeError("as_of_date must be an explicit date")
    windows = tuple(
        Window(
            (as_of_date - timedelta(days=7 * (index + 1))).isoformat(),
            (as_of_date - timedelta(days=7 * index)).isoformat(),
        )
        for index in range(5)
    )
    settled = settled_transactions(transactions)
    weeks = tuple(merchant_window(settled, merchant_id, window) for window in windows)
    prior_four = Window(windows[4].start, windows[1].end_exclusive)

    def suppressed(note: str) -> WeeklySalesPulse:
        return WeeklySalesPulse(
            as_of_date.isoformat(), windows[0], windows[1], prior_four,
            None, None, None, "suppressed", note,
        )

    if not all(contributors_safe(week) for week in weeks):
        return suppressed("A source week has fewer than 10 distinct contributors.")

    sales = [float(week["net_sales_aed"].sum()) for week in weeks]
    prior_four_average = sum(sales[1:]) / 4
    if sales[1] <= 0 or prior_four_average <= 0:
        return suppressed("The comparison week has insufficient sales evidence.")
    return WeeklySalesPulse(
        as_of_date.isoformat(), windows[0], windows[1], prior_four,
        round(sales[0], 2), percent_change(sales[0], sales[1]),
        percent_change(sales[0], prior_four_average), "available",
        "Observed net sales only. The four-week reference is divided by four to compare weekly amounts; this pulse does not drive recommendations.",
    )


def _weekday_labels(frame: pd.DataFrame) -> pd.Series:
    return frame["transaction_datetime"].dt.weekday.map(dict(enumerate(WEEKDAYS)))


def _daypart_labels(frame: pd.DataFrame) -> pd.Series:
    timestamps = frame["transaction_datetime"]
    weekday = timestamps.dt.weekday < 5
    hour = timestamps.dt.hour
    labels = pd.Series("weekend_all", index=frame.index, dtype="object")
    labels.loc[weekday & hour.lt(11)] = "weekday_breakfast"
    labels.loc[weekday & hour.ge(11) & hour.lt(15)] = "weekday_lunch"
    labels.loc[weekday & hour.ge(15) & hour.lt(18)] = "weekday_afternoon"
    labels.loc[weekday & hour.ge(18)] = "weekday_evening"
    return labels


def _fixed_breakdown(
    transactions: pd.DataFrame, merchant_id: str, as_of_date: date,
    labels: tuple[str, ...], labeler,
) -> BreakdownResult:
    meta = comparison_meta(as_of_date)
    settled = settled_transactions(transactions)
    baseline = merchant_window(settled, merchant_id, meta.baseline_window).copy()
    current = merchant_window(settled, merchant_id, meta.current_window).copy()
    baseline["_segment"] = labeler(baseline)
    current["_segment"] = labeler(current)
    if not contributors_safe(baseline) or not contributors_safe(current):
        return BreakdownResult(meta, None, "suppressed", "Insufficient contributors for this fixed breakdown.")
    segments = []
    for label in labels:
        before = baseline[baseline["_segment"].eq(label)]
        after = current[current["_segment"].eq(label)]
        before_complement = baseline[baseline["_segment"].ne(label)]
        after_complement = current[current["_segment"].ne(label)]
        # Suppress the entire fixed partition if any cell or its complement
        # is unsafe. This prevents recovery from a published total.
        if not all(contributors_safe(frame) for frame in (before, after, before_complement, after_complement)):
            return BreakdownResult(meta, None, "suppressed", "A small cell or related complement makes this breakdown unavailable.")
        before_sales = float(before["net_sales_aed"].sum())
        after_sales = float(after["net_sales_aed"].sum())
        segments.append(SegmentComparison(
            label, round(before_sales, 2), round(after_sales, 2),
            round(after_sales - before_sales, 2), len(before), len(after),
            percent_change(len(after), len(before)),
        ))
    return BreakdownResult(meta, tuple(segments), "available",
                           "Fixed, disjoint segments; sales deltas describe observed contributions, not causes.")


def get_day_of_week_analysis(transactions: pd.DataFrame, merchant_id: str, as_of_date: date) -> BreakdownResult:
    """Return the seven fixed weekday cells, or suppress the full breakdown."""

    return _fixed_breakdown(transactions, merchant_id, as_of_date, WEEKDAYS, _weekday_labels)


def get_daypart_analysis(transactions: pd.DataFrame, merchant_id: str, as_of_date: date) -> BreakdownResult:
    """Return fixed disjoint dayparts, including weekday 11:00-15:00 lunch."""

    return _fixed_breakdown(transactions, merchant_id, as_of_date, DAYPARTS, _daypart_labels)
