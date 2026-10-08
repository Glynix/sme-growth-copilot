"""Safe peer median for a fixed category/location/size group."""

from __future__ import annotations

from datetime import date

import pandas as pd

from sme_growth.analytics.common import comparison_meta, settled_transactions
from sme_growth.analytics.contracts import PeerBenchmark


MIN_PEERS = 10


def get_peer_benchmark(
    transactions: pd.DataFrame, merchants: pd.DataFrame,
    merchant_id: str, as_of_date: date,
) -> PeerBenchmark:
    """Return an aggregate median only when at least ten safe peers qualify."""

    meta = comparison_meta(as_of_date)
    required = {"merchant_id", "merchant_category", "merchant_location", "merchant_size"}
    if not required <= set(merchants.columns):
        raise ValueError("Merchant schema is incomplete")
    focal = merchants[merchants["merchant_id"].eq(merchant_id)]
    if len(focal) != 1:
        return PeerBenchmark(meta, "unavailable", None, None, None, None, "suppressed", "Merchant context is unavailable.")
    merchant = focal.iloc[0]
    label = f"{merchant['merchant_category']} · {merchant['merchant_location']} · {merchant['merchant_size']}"
    peer_ids = set(merchants.loc[
        merchants["merchant_category"].eq(merchant["merchant_category"])
        & merchants["merchant_location"].eq(merchant["merchant_location"])
        & merchants["merchant_size"].eq(merchant["merchant_size"])
        & merchants["merchant_id"].ne(merchant_id),
        "merchant_id",
    ])
    if len(peer_ids) < MIN_PEERS:
        return PeerBenchmark(meta, label, None, None, None, None, "suppressed", "Fewer than ten comparable merchants.")
    settled = settled_transactions(transactions)
    peer_tx = settled[settled["merchant_id"].isin(peer_ids)]
    baseline = peer_tx[
        peer_tx["transaction_datetime"].ge(pd.Timestamp(meta.baseline_window.start))
        & peer_tx["transaction_datetime"].lt(pd.Timestamp(meta.baseline_window.end_exclusive))
    ]
    current = peer_tx[
        peer_tx["transaction_datetime"].ge(pd.Timestamp(meta.current_window.start))
        & peer_tx["transaction_datetime"].lt(pd.Timestamp(meta.current_window.end_exclusive))
    ]
    base_counts = baseline.groupby("merchant_id")["synthetic_customer_id"].nunique()
    current_counts = current.groupby("merchant_id")["synthetic_customer_id"].nunique()
    safe_ids = set(base_counts[base_counts >= 10].index) & set(current_counts[current_counts >= 10].index)
    base_sales = baseline[baseline["merchant_id"].isin(safe_ids)].groupby("merchant_id")["net_sales_aed"].sum()
    current_sales = current[current["merchant_id"].isin(safe_ids)].groupby("merchant_id")["net_sales_aed"].sum()
    changes = ((current_sales / base_sales) - 1).dropna() * 100
    if len(changes) < MIN_PEERS:
        return PeerBenchmark(meta, label, None, None, None, None, "suppressed", "Fewer than ten peers have sufficient data in both periods.")
    return PeerBenchmark(
        meta, label, len(changes), round(float(changes.median()), 2),
        round(float(changes.quantile(.25)), 2), round(float(changes.quantile(.75)), 2),
        "available", "Median and quartiles summarize peer changes; no peer identities or individual results are released.",
    )
