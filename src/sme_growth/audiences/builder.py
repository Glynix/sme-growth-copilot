"""Private local Audience Builder; only an aggregate summary crosses out."""

from __future__ import annotations

import hashlib
import sqlite3
import uuid
from datetime import date, datetime, time, timedelta
from pathlib import Path

import pandas as pd

from sme_growth.analytics.common import settled_transactions
from sme_growth.audiences.private_rules import (
    RULE_CODE, SAFE_DEFINITION, eligible_program_members, select_private_lapsed_lunch,
)
from sme_growth.contracts import AudienceSummary, MIN_ELIGIBLE_AUDIENCE, MIN_OUTCOME_CONTRIBUTORS
from sme_growth.ml.retention import PrivateRetentionModel, select_private_top_risk_regulars


MODEL_RULE_CODE = "top_risk_active_regulars_v1"
MODEL_SAFE_DEFINITION = (
    "Highest predicted 30-day non-return risk among the top 20% of active regulars "
    "(at least 3 settled visits in 90 days and a visit within 45 days); "
    "enrolled with valid simulated retention/issuer-app permission, no revocation "
    "or suppression, and frequency-cap availability. Model risk is not offer response."
)


class PrivateAudienceStore:
    """SQLite boundary for private membership; place it under `runtime/`."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.path) as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS audience_snapshots (
                    audience_id TEXT NOT NULL,
                    audience_version INTEGER NOT NULL,
                    merchant_id TEXT NOT NULL,
                    rule_code TEXT NOT NULL,
                    as_of_date TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    membership_digest TEXT NOT NULL,
                    eligible_count INTEGER NOT NULL,
                    PRIMARY KEY (audience_id, audience_version)
                )
            """)
            connection.execute("""
                CREATE TABLE IF NOT EXISTS audience_members (
                    audience_id TEXT NOT NULL,
                    audience_version INTEGER NOT NULL,
                    synthetic_customer_id TEXT NOT NULL,
                    PRIMARY KEY (audience_id, audience_version, synthetic_customer_id)
                )
            """)

    def _save_snapshot(
        self, merchant_id: str, as_of_date: date, eligible_members: frozenset[str],
        expires_at: datetime,
        rule_code: str = RULE_CODE,
    ) -> tuple[str, int]:
        """Persist or reuse one scoped version; never return the membership."""

        audience_id = "aud_" + uuid.uuid5(
            uuid.NAMESPACE_URL, f"sme-growth-local-simulator/{merchant_id}/{rule_code}",
        ).hex
        members = sorted(eligible_members)
        digest = hashlib.sha256("\n".join(members).encode("utf-8")).hexdigest()
        with sqlite3.connect(self.path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            latest = connection.execute(
                """SELECT audience_version, as_of_date, expires_at, membership_digest
                   FROM audience_snapshots WHERE audience_id = ? AND merchant_id = ?
                   ORDER BY audience_version DESC LIMIT 1""",
                (audience_id, merchant_id),
            ).fetchone()
            if latest and latest[1:] == (as_of_date.isoformat(), expires_at.isoformat(), digest):
                return audience_id, int(latest[0])
            version = int(latest[0]) + 1 if latest else 1
            connection.execute(
                """INSERT INTO audience_snapshots
                   (audience_id, audience_version, merchant_id, rule_code, as_of_date,
                    expires_at, membership_digest, eligible_count)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (audience_id, version, merchant_id, rule_code, as_of_date.isoformat(),
                 expires_at.isoformat(), digest, len(members)),
            )
            connection.executemany(
                """INSERT INTO audience_members
                   (audience_id, audience_version, synthetic_customer_id) VALUES (?, ?, ?)""",
                ((audience_id, version, token) for token in members),
            )
        return audience_id, version


def _expiry(as_of_date: date, eligible_members: frozenset[str], program: pd.DataFrame) -> datetime:
    default_expiry = datetime.combine(as_of_date + timedelta(days=7), time.min)
    valid = pd.to_datetime(program.loc[
        program["synthetic_customer_id"].isin(eligible_members), "valid_until",
    ], errors="coerce")
    earliest = valid.min()
    if pd.isna(earliest):
        return datetime.combine(as_of_date, time.min)
    # Program `valid_until` is an inclusive date in the synthetic fixture.
    return min(default_expiry, earliest.to_pydatetime().replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1))


def build_eligible_audience(
    transactions: pd.DataFrame, program_eligibility: pd.DataFrame | None,
    merchant_id: str, as_of_date: date, store: PrivateAudienceStore,
) -> AudienceSummary:
    """Return a merchant-scoped opaque handle and safe count, never members."""

    cohort = select_private_lapsed_lunch(transactions, program_eligibility, merchant_id, as_of_date)
    snapshot = datetime.combine(as_of_date, time.min)
    blocked = AudienceSummary(
        audience_id=None, audience_version=None, safe_definition=SAFE_DEFINITION,
        eligible_count=None, snapshot_at=snapshot, expires_at=snapshot,
        activation_eligible=False, suppression_reason="Insufficient safe or valid eligible audience.",
    )
    if not cohort.program_data_complete:
        return blocked
    if len(cohort.eligible_members) < MIN_ELIGIBLE_AUDIENCE:
        return blocked
    if len(cohort.candidates) - len(cohort.eligible_members) < MIN_OUTCOME_CONTRIBUTORS:
        return blocked
    assert program_eligibility is not None
    expires_at = _expiry(as_of_date, cohort.eligible_members, program_eligibility)
    if expires_at <= snapshot:
        return blocked
    audience_id, version = store._save_snapshot(merchant_id, as_of_date, cohort.eligible_members, expires_at)
    return AudienceSummary(
        audience_id=audience_id, audience_version=version,
        safe_definition=SAFE_DEFINITION, eligible_count=len(cohort.eligible_members),
        snapshot_at=snapshot, expires_at=expires_at, activation_eligible=True,
    )


def build_model_risk_audience(
    model: PrivateRetentionModel, transactions: pd.DataFrame,
    program_eligibility: pd.DataFrame | None, merchant_id: str,
    as_of_date: date, store: PrivateAudienceStore,
) -> AudienceSummary:
    """Version a private top-quintile model audience after program checks."""

    snapshot = datetime.combine(as_of_date, time.min)
    blocked = AudienceSummary(
        None, None, MODEL_SAFE_DEFINITION, None, snapshot, snapshot, False,
        "Insufficient model, safe audience size, or valid program eligibility.",
        "estimated_model_and_simulated_program_eligibility",
    )
    if program_eligibility is None:
        return blocked
    candidates = select_private_top_risk_regulars(model, transactions, merchant_id, as_of_date)
    if not candidates:
        return blocked
    eligible = eligible_program_members(candidates, program_eligibility, as_of_date)
    if len(eligible) < MIN_ELIGIBLE_AUDIENCE or len(candidates) - len(eligible) < MIN_OUTCOME_CONTRIBUTORS:
        return blocked
    expires_at = _expiry(as_of_date, eligible, program_eligibility)
    if expires_at <= snapshot:
        return blocked
    settled = settled_transactions(transactions)
    historical = settled.loc[
        settled["merchant_id"].eq(merchant_id)
        & settled["synthetic_customer_id"].isin(eligible)
        & settled["transaction_datetime"].ge(pd.Timestamp(as_of_date - timedelta(days=90)))
        & settled["transaction_datetime"].lt(pd.Timestamp(as_of_date))
    ]
    historical_spend = (
        round(float(historical["net_sales_aed"].sum()) / 3, 2)
        if historical["synthetic_customer_id"].nunique() >= MIN_OUTCOME_CONTRIBUTORS else None
    )
    audience_id, version = store._save_snapshot(merchant_id, as_of_date, eligible, expires_at, MODEL_RULE_CODE)
    return AudienceSummary(
        audience_id, version, MODEL_SAFE_DEFINITION, len(eligible), snapshot, expires_at,
        True, evidence_type="estimated_model_and_simulated_program_eligibility",
        historical_spend_aed_per_30d_equivalent=historical_spend,
    )
