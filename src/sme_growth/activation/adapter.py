"""Local-only mock partner boundary. No network, issuer access, or messaging."""

from __future__ import annotations

import sqlite3
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


RECEIPT_MESSAGE = "Submitted to simulator. No customers contacted."


@dataclass(frozen=True, slots=True)
class ApprovedCampaign:
    campaign_ref: str
    campaign_version: int
    audience_ref: str
    audience_version: int
    approval_ref: str
    approved_definition_hash: str
    idempotency_key: str
    simulation: bool = True


@dataclass(frozen=True, slots=True)
class SubmissionReceipt:
    campaign_ref: str
    campaign_version: int
    status: str
    message: str
    simulation: bool = True


class MockActivationAdapter(Protocol):
    def submit(self, campaign: ApprovedCampaign) -> SubmissionReceipt | None:
        """None means the response was lost and the outcome must be reconciled."""

    def reconcile(self, idempotency_key: str) -> SubmissionReceipt | None: ...

    def accept(self, idempotency_key: str) -> None: ...

    def mark_simulated_dispatched(self, idempotency_key: str) -> None: ...


class LocalMockPartnerAdapter:
    """Persistent simulator; optional one-shot faults exist only for tests."""

    def __init__(self, path: Path, *, lose_next_response: bool = False, fail_next_submission: bool = False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lose_next_response = lose_next_response
        self.fail_next_submission = fail_next_submission
        with sqlite3.connect(self.path) as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS partner_submissions (
                    idempotency_key TEXT PRIMARY KEY,
                    campaign_ref TEXT NOT NULL,
                    campaign_version INTEGER NOT NULL,
                    status TEXT NOT NULL
                )
            """)

    def reconcile(self, idempotency_key: str) -> SubmissionReceipt | None:
        with sqlite3.connect(self.path) as connection:
            row = connection.execute(
                "SELECT campaign_ref, campaign_version, status FROM partner_submissions WHERE idempotency_key = ?",
                (idempotency_key,),
            ).fetchone()
        if row is None:
            return None
        return SubmissionReceipt(row[0], int(row[1]), row[2], RECEIPT_MESSAGE)

    def submit(self, campaign: ApprovedCampaign) -> SubmissionReceipt | None:
        try:
            with sqlite3.connect(self.path) as connection:
                approved = connection.execute("""
                    SELECT status, approval_ref, approved_definition_hash, idempotency_key, definition_json
                    FROM campaign_versions WHERE campaign_ref = ? AND campaign_version = ?
                """, (campaign.campaign_ref, campaign.campaign_version)).fetchone()
        except sqlite3.OperationalError:
            raise ValueError("Exact persisted approval is required for simulator submission") from None
        if approved is None or approved[0] not in ("approved", "submission_unknown"):
            raise ValueError("Exact persisted approval is required for simulator submission")
        definition = json.loads(approved[4])
        if (
            not campaign.simulation
            or approved[1] != campaign.approval_ref
            or approved[2] != campaign.approved_definition_hash
            or approved[3] != campaign.idempotency_key
            or definition["audience_id"] != campaign.audience_ref
            or int(definition["audience_version"]) != campaign.audience_version
        ):
            raise ValueError("Simulator submission does not match the persisted approval")
        existing = self.reconcile(campaign.idempotency_key)
        if existing is not None:
            return existing
        status = "failed" if self.fail_next_submission else "submitted"
        self.fail_next_submission = False
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                "INSERT OR IGNORE INTO partner_submissions VALUES (?, ?, ?, ?)",
                (campaign.idempotency_key, campaign.campaign_ref, campaign.campaign_version, status),
            )
        if self.lose_next_response:
            self.lose_next_response = False
            return None
        return self.reconcile(campaign.idempotency_key)

    def _transition(self, idempotency_key: str, expected: str, target: str) -> None:
        with sqlite3.connect(self.path) as connection:
            row = connection.execute(
                "SELECT status FROM partner_submissions WHERE idempotency_key = ?", (idempotency_key,),
            ).fetchone()
            if row is None or row[0] not in (expected, target):
                raise ValueError("Simulator submission is not ready for this transition")
            connection.execute(
                "UPDATE partner_submissions SET status = ? WHERE idempotency_key = ? AND status = ?",
                (target, idempotency_key, expected),
            )

    def accept(self, idempotency_key: str) -> None:
        self._transition(idempotency_key, "submitted", "accepted")

    def mark_simulated_dispatched(self, idempotency_key: str) -> None:
        self._transition(idempotency_key, "accepted", "simulated_dispatched")
