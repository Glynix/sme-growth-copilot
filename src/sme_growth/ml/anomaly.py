"""Transparent merchant anomaly signal from observed historical changes."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from sme_growth.analytics.common import settled_transactions
from sme_growth.analytics.contracts import PeerBenchmark
from sme_growth.analytics.performance import get_sales_performance


@dataclass(frozen=True, slots=True)
class MerchantAnomalySignal:
    as_of_date: str
    current_window_start: str
    current_window_end_exclusive: str
    observed_sales_change_pct: float | None
    historical_median_change_pct: float | None
    historical_robust_spread_pct: float | None
    peer_gap_percentage_points: float | None
    unusually_weak: bool | None
    status: str
    evidence_type: str = "observed_comparative_signal"
    note: str = "A weak signal is an unusual comparison, not a proven cause or forecast."


def get_merchant_anomaly_signal(
    transactions: pd.DataFrame, merchant_id: str, as_of_date: date,
    peer_benchmark: PeerBenchmark | None = None,
) -> MerchantAnomalySignal:
    """Compare latest 28-day growth with older non-overlapping 28-day growth."""

    sales = get_sales_performance(transactions, merchant_id, as_of_date)
    observed = sales.sales_change_pct
    empty = MerchantAnomalySignal(
        as_of_date.isoformat(), sales.meta.current_window.start,
        sales.meta.current_window.end_exclusive, observed,
        None, None, None, None, "insufficient_evidence",
    )
    if observed is None:
        return empty
    settled = settled_transactions(transactions)
    merchant = settled[settled["merchant_id"].eq(merchant_id)]
    block_sales: list[float | None] = []
    for block in range(13):
        end = as_of_date - timedelta(days=block * 28)
        start = end - timedelta(days=28)
        rows = merchant[
            merchant["transaction_datetime"].ge(pd.Timestamp(start))
            & merchant["transaction_datetime"].lt(pd.Timestamp(end))
        ]
        block_sales.append(float(rows["net_sales_aed"].sum()) if rows["synthetic_customer_id"].nunique() >= 10 else None)
    older_changes = [
        (block_sales[index] / block_sales[index + 1] - 1) * 100
        for index in range(1, 12)
        if block_sales[index] is not None and block_sales[index + 1] is not None and block_sales[index + 1] > 0
    ]
    if len(older_changes) < 6:
        return empty
    historical_median = float(np.median(older_changes))
    robust_spread = float(1.4826 * np.median(np.abs(np.asarray(older_changes) - historical_median)))
    peer_change = peer_benchmark.median_sales_change_pct if peer_benchmark and peer_benchmark.status == "available" else None
    peer_gap = observed - peer_change if peer_change is not None else None
    weak_vs_history = observed <= historical_median - max(2 * robust_spread, 5.0)
    weak_vs_peers = observed <= -5 and peer_gap is not None and peer_gap <= -5
    return MerchantAnomalySignal(
        as_of_date.isoformat(), sales.meta.current_window.start,
        sales.meta.current_window.end_exclusive, observed,
        round(historical_median, 2), round(robust_spread, 2),
        round(peer_gap, 2) if peer_gap is not None else None,
        bool(weak_vs_history or weak_vs_peers), "available",
    )
