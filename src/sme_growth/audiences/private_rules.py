"""Private behavioral matching and simulated program permission checks.

Never return these membership sets through the UI, an export, or an LLM.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import pandas as pd

from sme_growth.analytics.common import settled_transactions
from sme_growth.synthetic.generator import RETENTION_PURPOSE, SIMULATED_CHANNEL


REQUIRED_PROGRAM_COLUMNS = {
    "synthetic_customer_id", "enrolled", "permission_state", "purpose", "channel",
    "valid_until", "revoked_at", "frequency_cap_ok", "suppressed",
}
RULE_CODE = "lapsed_weekday_lunch_v1"
SAFE_DEFINITION = (
    "At least 3 settled visits, including 2 weekday lunches (11:00-15:00), "
    "in the 90 days ending 45 days before the as-of date; no later observed "
    "visit; enrolled with valid simulated retention/issuer-app permission, "
    "no revocation or suppression, and frequency-cap availability."
)


@dataclass(repr=False)
class PrivateLapsedCohort:
    as_of_date: date
    qualification_start: date
    qualification_end_exclusive: date
    candidates: frozenset[str] = field(repr=False)
    eligible_members: frozenset[str] = field(repr=False)
    historical_monthly_spend_aed: float = field(repr=False)
    program_data_complete: bool = False


def eligible_program_members(
    candidates: frozenset[str], program_eligibility: pd.DataFrame | None,
    as_of_date: date,
) -> frozenset[str]:
    """Apply current purpose, channel and suppression checks to private tokens."""

    if program_eligibility is None or not REQUIRED_PROGRAM_COLUMNS <= set(program_eligibility.columns):
        return frozenset()
    program = program_eligibility[
        program_eligibility["synthetic_customer_id"].isin(candidates)
    ].drop_duplicates(subset="synthetic_customer_id", keep=False)
    valid_until = pd.to_datetime(program["valid_until"], errors="coerce")
    return frozenset(program.loc[
        program["enrolled"].eq(True)
        & program["permission_state"].eq("granted")
        & program["purpose"].eq(RETENTION_PURPOSE)
        & program["channel"].eq(SIMULATED_CHANNEL)
        & valid_until.ge(pd.Timestamp(as_of_date))
        & program["revoked_at"].isna()
        & program["frequency_cap_ok"].eq(True)
        & program["suppressed"].eq(False),
        "synthetic_customer_id",
    ])


def select_private_lapsed_lunch(
    transactions: pd.DataFrame,
    program_eligibility: pd.DataFrame | None,
    merchant_id: str,
    as_of_date: date,
) -> PrivateLapsedCohort:
    """Select members privately; unknown, malformed and duplicate records fail closed."""

    start = as_of_date - timedelta(days=135)
    cutoff = as_of_date - timedelta(days=45)
    settled = settled_transactions(transactions)
    merchant = settled[
        settled["merchant_id"].eq(merchant_id)
        & settled["transaction_datetime"].lt(pd.Timestamp(as_of_date))
    ]
    qualifying = merchant[
        merchant["transaction_datetime"].ge(pd.Timestamp(start))
        & merchant["transaction_datetime"].lt(pd.Timestamp(cutoff))
    ]
    visits = qualifying.groupby("synthetic_customer_id").size()
    times = qualifying["transaction_datetime"]
    lunches = qualifying[
        (times.dt.weekday < 5) & times.dt.hour.ge(11) & times.dt.hour.lt(15)
    ].groupby("synthetic_customer_id").size()
    last_seen = merchant.groupby("synthetic_customer_id")["transaction_datetime"].max()
    candidates = frozenset(
        set(visits[visits.ge(3)].index)
        & set(lunches[lunches.ge(2)].index)
        & set(last_seen[last_seen.lt(pd.Timestamp(cutoff))].index)
    )
    if program_eligibility is None or not REQUIRED_PROGRAM_COLUMNS <= set(program_eligibility.columns):
        return PrivateLapsedCohort(as_of_date, start, cutoff, candidates, frozenset(), 0.0, False)

    eligible = eligible_program_members(candidates, program_eligibility, as_of_date)
    spend = float(qualifying.loc[
        qualifying["synthetic_customer_id"].isin(eligible), "net_sales_aed",
    ].sum()) / 3
    return PrivateLapsedCohort(as_of_date, start, cutoff, candidates, eligible, spend, True)
