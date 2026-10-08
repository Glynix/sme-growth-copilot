"""Fixed day, week and calendar-month sales overview with safe aggregate peers."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from sme_growth.analytics.common import contributors_safe, merchant_window, percent_change, settled_transactions
from sme_growth.analytics.contracts import DailySalesPoint, OverviewPeriod, PeerDailySalesPoint, SalesOverview, Window
from sme_growth.contracts import MIN_PEER_MERCHANTS


def _matching_peer_ids(merchants: pd.DataFrame, merchant_id: str) -> set[str]:
    required = {"merchant_id", "merchant_category", "merchant_location", "merchant_size"}
    if not required <= set(merchants.columns):
        raise ValueError("Merchant schema is incomplete")
    focal = merchants.loc[merchants["merchant_id"].eq(merchant_id)]
    if len(focal) != 1:
        return set()
    merchant = focal.iloc[0]
    return set(merchants.loc[
        merchants["merchant_category"].eq(merchant["merchant_category"])
        & merchants["merchant_location"].eq(merchant["merchant_location"])
        & merchants["merchant_size"].eq(merchant["merchant_size"])
        & merchants["merchant_id"].ne(merchant_id),
        "merchant_id",
    ])


def _peer_change(
    settled: pd.DataFrame, peer_ids: set[str], current_window: Window, baseline_window: Window,
) -> tuple[float | None, int | None]:
    if len(peer_ids) < MIN_PEER_MERCHANTS:
        return None, None
    peers = settled.loc[settled["merchant_id"].isin(peer_ids)]
    current = peers.loc[
        peers["transaction_datetime"].ge(pd.Timestamp(current_window.start))
        & peers["transaction_datetime"].lt(pd.Timestamp(current_window.end_exclusive))
    ]
    baseline = peers.loc[
        peers["transaction_datetime"].ge(pd.Timestamp(baseline_window.start))
        & peers["transaction_datetime"].lt(pd.Timestamp(baseline_window.end_exclusive))
    ]
    active_ids = set(current["merchant_id"]) & set(baseline["merchant_id"])
    if len(active_ids) < MIN_PEER_MERCHANTS:
        return None, None
    current = current.loc[current["merchant_id"].isin(active_ids)]
    baseline = baseline.loc[baseline["merchant_id"].isin(active_ids)]
    if not contributors_safe(current) or not contributors_safe(baseline):
        return None, None
    change = percent_change(float(current["net_sales_aed"].sum()), float(baseline["net_sales_aed"].sum()))
    return (change, len(active_ids)) if change is not None else (None, None)


def _period(
    settled: pd.DataFrame, merchant_id: str, peer_ids: set[str],
    current_window: Window, baseline_window: Window,
) -> OverviewPeriod:
    current = merchant_window(settled, merchant_id, current_window)
    baseline = merchant_window(settled, merchant_id, baseline_window)
    safe = contributors_safe(current) and contributors_safe(baseline)
    sales = round(float(current["net_sales_aed"].sum()), 2) if safe else None
    change = percent_change(sales, float(baseline["net_sales_aed"].sum())) if safe else None
    peer_change, peer_count = _peer_change(settled, peer_ids, current_window, baseline_window)
    return OverviewPeriod(
        current_window, baseline_window, sales, change, peer_change, peer_count,
        "available" if change is not None else "suppressed",
        "available" if peer_change is not None else "suppressed",
    )


def _daily_trend(
    settled: pd.DataFrame, merchant_id: str, window: Window,
) -> tuple[DailySalesPoint, ...] | None:
    month = merchant_window(settled, merchant_id, window)
    start = date.fromisoformat(window.start)
    end = date.fromisoformat(window.end_exclusive)
    points = []
    for offset in range((end - start).days):
        day = start + timedelta(days=offset)
        cell = month.loc[month["transaction_datetime"].dt.date.eq(day)]
        complement = month.loc[month["transaction_datetime"].dt.date.ne(day)]
        if not contributors_safe(cell) or not contributors_safe(complement):
            return None
        points.append(DailySalesPoint(day.isoformat(), round(float(cell["net_sales_aed"].sum()), 2)))
    return tuple(points)


def _peer_daily_trend(
    settled: pd.DataFrame, peer_ids: set[str], window: Window, baseline_window: Window,
) -> tuple[tuple[PeerDailySalesPoint, ...] | None, int | None]:
    if len(peer_ids) < MIN_PEER_MERCHANTS:
        return None, None
    peers = settled.loc[settled["merchant_id"].isin(peer_ids)]
    month = peers.loc[
        peers["transaction_datetime"].ge(pd.Timestamp(window.start))
        & peers["transaction_datetime"].lt(pd.Timestamp(window.end_exclusive))
    ].copy()
    baseline = peers.loc[
        peers["transaction_datetime"].ge(pd.Timestamp(baseline_window.start))
        & peers["transaction_datetime"].lt(pd.Timestamp(baseline_window.end_exclusive))
    ]
    current_counts = month.groupby("merchant_id")["synthetic_customer_id"].nunique()
    baseline_counts = baseline.groupby("merchant_id")["synthetic_customer_id"].nunique()
    cohort = sorted(
        set(current_counts[current_counts >= 10].index)
        & set(baseline_counts[baseline_counts >= 10].index)
    )
    if len(cohort) < MIN_PEER_MERCHANTS:
        return None, None
    month = month.loc[month["merchant_id"].isin(cohort)]
    month["_date"] = month["transaction_datetime"].dt.date
    start = date.fromisoformat(window.start)
    end = date.fromisoformat(window.end_exclusive)
    points = []
    for offset in range((end - start).days):
        day = start + timedelta(days=offset)
        cell = month.loc[month["_date"].eq(day)]
        complement = month.loc[month["_date"].ne(day)]
        if (cell["merchant_id"].nunique() < MIN_PEER_MERCHANTS
                or complement["merchant_id"].nunique() < MIN_PEER_MERCHANTS
                or not contributors_safe(cell) or not contributors_safe(complement)):
            return None, None
        per_peer_sales = cell.groupby("merchant_id")["net_sales_aed"].sum().reindex(cohort, fill_value=0)
        points.append(PeerDailySalesPoint(day.isoformat(), round(float(per_peer_sales.median()), 2)))
    return tuple(points), len(cohort)


def get_sales_overview(
    transactions: pd.DataFrame, merchants: pd.DataFrame, merchant_id: str, as_of_date: date,
) -> SalesOverview:
    """Release fixed aggregate comparisons and all-or-nothing daily trend."""

    if not isinstance(as_of_date, date):
        raise TypeError("as_of_date must be an explicit date")
    current_month_end = as_of_date.replace(day=1)
    current_month_start = (current_month_end - timedelta(days=1)).replace(day=1)
    baseline_month_start = (current_month_start - timedelta(days=1)).replace(day=1)
    windows = (
        (
            Window((as_of_date - timedelta(days=1)).isoformat(), as_of_date.isoformat()),
            Window((as_of_date - timedelta(days=8)).isoformat(), (as_of_date - timedelta(days=7)).isoformat()),
        ),
        (
            Window((as_of_date - timedelta(days=7)).isoformat(), as_of_date.isoformat()),
            Window((as_of_date - timedelta(days=14)).isoformat(), (as_of_date - timedelta(days=7)).isoformat()),
        ),
        (
            Window(current_month_start.isoformat(), current_month_end.isoformat()),
            Window(baseline_month_start.isoformat(), current_month_start.isoformat()),
        ),
    )
    settled = settled_transactions(transactions)
    peer_ids = _matching_peer_ids(merchants, merchant_id)
    day, week, month = (
        _period(settled, merchant_id, peer_ids, current, baseline)
        for current, baseline in windows
    )
    trend = _daily_trend(settled, merchant_id, windows[2][0])
    peer_trend, peer_trend_count = _peer_daily_trend(settled, peer_ids, windows[2][0], windows[2][1])
    return SalesOverview(
        as_of_date.isoformat(), day, week, month, windows[2][0], trend,
        "available" if trend is not None else "suppressed",
        peer_trend, peer_trend_count, "available" if peer_trend is not None else "suppressed",
    )
