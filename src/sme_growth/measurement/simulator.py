"""Private deterministic event simulation and one aggregate-only release."""

from __future__ import annotations

import hashlib
import sqlite3
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from sme_growth.contracts import AggregateOutcome, MIN_OUTCOME_CONTRIBUTORS


ATTRIBUTION_NOTE = (
    "Simulated campaign-associated purchases only. Submission is not delivery; "
    "these descriptive sales are not incremental lift. No holdout was used."
)


def _event_id(campaign_ref: str, version: int, token: str, visit: int) -> str:
    return hashlib.sha256(f"{campaign_ref}|{version}|{token}|{visit}".encode()).hexdigest()


def _private_members(
    connection: sqlite3.Connection, merchant_id: str, audience_id: str, audience_version: int,
) -> tuple[str, ...]:
    rows = connection.execute("""
        SELECT m.synthetic_customer_id
        FROM audience_members AS m
        JOIN audience_snapshots AS s
          ON s.audience_id = m.audience_id AND s.audience_version = m.audience_version
        WHERE s.merchant_id = ? AND s.audience_id = ? AND s.audience_version = ?
        ORDER BY m.synthetic_customer_id
    """, (merchant_id, audience_id, audience_version)).fetchall()
    return tuple(row[0] for row in rows)


def simulate_aggregate_outcome(
    path: Path, *, campaign_ref: str, campaign_version: int, merchant_id: str,
    audience_id: str, audience_version: int, eligible_count: int,
    reporting_start: datetime, reporting_end: datetime,
    offer_percent: int, spend_per_visit_aed: Decimal,
) -> AggregateOutcome:
    """Write private synthetic events once, then release only a safe whole-campaign total."""

    with sqlite3.connect(path) as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS simulated_purchase_events (
                event_id TEXT PRIMARY KEY,
                campaign_ref TEXT NOT NULL,
                campaign_version INTEGER NOT NULL,
                synthetic_customer_id TEXT NOT NULL,
                amount_cents INTEGER NOT NULL,
                incentive_cents INTEGER NOT NULL
            )
        """)
        members = _private_members(connection, merchant_id, audience_id, audience_version)
        if len(members) != eligible_count:
            raise ValueError("Private audience snapshot is unavailable")
        # Fixed 14% simulated response; ordering is stable and private. This is
        # a demonstration mechanism, not a prediction or attribution method.
        ranked = sorted(members, key=lambda token: hashlib.sha256(
            f"{campaign_ref}|{campaign_version}|{token}".encode()
        ).digest())
        purchasers = ranked[:round(eligible_count * .14)]
        amount_cents = int((spend_per_visit_aed * 100).to_integral_value())
        incentive_cents = int((Decimal(amount_cents) * Decimal(offer_percent) / 100).to_integral_value())
        connection.executemany("""
            INSERT OR IGNORE INTO simulated_purchase_events
            (event_id, campaign_ref, campaign_version, synthetic_customer_id, amount_cents, incentive_cents)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            (_event_id(campaign_ref, campaign_version, token, visit), campaign_ref,
             campaign_version, token, amount_cents, incentive_cents)
            for token in purchasers for visit in (1, 2)
        ))
        row = connection.execute("""
            SELECT COUNT(DISTINCT synthetic_customer_id), SUM(amount_cents), SUM(incentive_cents)
            FROM simulated_purchase_events WHERE campaign_ref = ? AND campaign_version = ?
        """, (campaign_ref, campaign_version)).fetchone()
    purchaser_count = int(row[0])
    safe = purchaser_count >= MIN_OUTCOME_CONTRIBUTORS and eligible_count - purchaser_count >= MIN_OUTCOME_CONTRIBUTORS
    if not safe:
        return AggregateOutcome(campaign_ref, reporting_start, reporting_end, None, None, None,
                                True, ATTRIBUTION_NOTE)
    sales = Decimal(int(row[1])) / 100
    cost = Decimal(int(row[2])) / 100
    return AggregateOutcome(campaign_ref, reporting_start, reporting_end, purchaser_count,
                            sales, cost, False, ATTRIBUTION_NOTE)
