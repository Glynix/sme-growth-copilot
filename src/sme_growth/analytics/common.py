"""Internal transaction preparation and fixed comparison windows."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from sme_growth.analytics.contracts import ComparisonMeta, Window


REQUIRED_TRANSACTION_COLUMNS = {
    "merchant_id", "synthetic_customer_id", "transaction_datetime", "amount",
    "refund_amount", "transaction_status", "currency",
}
PRIVACY_MIN_CONTRIBUTORS = 10


def comparison_meta(as_of_date: date) -> ComparisonMeta:
    if not isinstance(as_of_date, date):
        raise TypeError("as_of_date must be an explicit date")
    return ComparisonMeta(
        as_of_date=as_of_date.isoformat(),
        baseline_window=Window((as_of_date - timedelta(days=56)).isoformat(), (as_of_date - timedelta(days=28)).isoformat()),
        current_window=Window((as_of_date - timedelta(days=28)).isoformat(), as_of_date.isoformat()),
    )


def settled_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    """Keep positive-net AED purchases with complete required values.

    Declines and fully refunded purchases do not count as sales or purchases.
    A partial refund reduces net sales while retaining one purchase. Missing
    or invalid rows are excluded rather than treated as zero.
    """

    missing = REQUIRED_TRANSACTION_COLUMNS - set(transactions.columns)
    if missing:
        raise ValueError(f"Transaction schema is incomplete: {', '.join(sorted(missing))}")
    dates = pd.to_datetime(transactions["transaction_datetime"], errors="coerce")
    amount = pd.to_numeric(transactions["amount"], errors="coerce")
    refund = pd.to_numeric(transactions["refund_amount"], errors="coerce")
    net = amount - refund
    valid = (
        dates.notna()
        & amount.notna() & refund.notna()
        & amount.gt(0) & refund.ge(0) & refund.le(amount) & net.gt(0)
        & transactions["currency"].eq("AED")
        & transactions["transaction_status"].isin(("completed", "refunded"))
        & transactions["merchant_id"].notna()
        & transactions["synthetic_customer_id"].notna()
    )
    result = transactions.loc[valid, ["merchant_id", "synthetic_customer_id"]].copy()
    result["transaction_datetime"] = dates.loc[valid]
    result["net_sales_aed"] = net.loc[valid]
    return result


def merchant_window(frame: pd.DataFrame, merchant_id: str, window: Window) -> pd.DataFrame:
    return frame.loc[
        frame["merchant_id"].eq(merchant_id)
        & frame["transaction_datetime"].ge(pd.Timestamp(window.start))
        & frame["transaction_datetime"].lt(pd.Timestamp(window.end_exclusive))
    ]


def contributors_safe(frame: pd.DataFrame) -> bool:
    return frame["synthetic_customer_id"].nunique() >= PRIVACY_MIN_CONTRIBUTORS


def percent_change(current: float, baseline: float) -> float | None:
    return round((current / baseline - 1) * 100, 2) if baseline > 0 else None
