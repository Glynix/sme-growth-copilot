"""Version-bound approval and persistent local simulation lifecycle."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from dataclasses import asdict, dataclass, replace
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from pathlib import Path

import pandas as pd

from sme_growth.activation.adapter import (
    ApprovedCampaign, LocalMockPartnerAdapter, MockActivationAdapter,
    RECEIPT_MESSAGE, SubmissionReceipt,
)
from sme_growth.audiences.builder import PrivateAudienceStore, build_eligible_audience, build_model_risk_audience
from sme_growth.contracts import AggregateOutcome, AudienceSummary, MIN_ELIGIBLE_AUDIENCE
from sme_growth.measurement.simulator import simulate_aggregate_outcome
from sme_growth.ml.retention import train_retention_model
from sme_growth.recommendations.economics import calculate_impact
from sme_growth.recommendations.engine import Opportunity
from sme_growth.synthetic.generator import SIMULATED_CHANNEL


class CampaignError(ValueError):
    """Only static, merchant-safe explanations should be raised."""


WEEKDAY_LUNCH_DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday")
WEEKDAY_LUNCH_WINDOW = "11:00–15:00 weekday lunch"
ALL_DAYS = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
ALL_PURCHASES_WINDOW = "all purchases"


@dataclass(frozen=True, slots=True)
class CampaignDefinition:
    merchant_id: str
    opportunity_id: str
    audience_id: str
    audience_version: int
    offer_percent: int
    start_date: date
    end_date: date
    channel: str
    budget_aed: Decimal
    offer_days: tuple[str, ...] = WEEKDAY_LUNCH_DAYS
    offer_window: str = WEEKDAY_LUNCH_WINDOW

    @property
    def offer_terms(self) -> str:
        if self.offer_days == ALL_DAYS and self.offer_window == ALL_PURCHASES_WINDOW:
            return f"{self.offer_percent}% off all purchases, any day"
        return f"{self.offer_percent}% off {self.offer_window}; {', '.join(self.offer_days)} only"

    def canonical(self) -> dict[str, object]:
        return {
            "merchant_id": self.merchant_id, "opportunity_id": self.opportunity_id,
            "audience_id": self.audience_id, "audience_version": self.audience_version,
            "offer_percent": self.offer_percent, "offer_days": list(self.offer_days),
            "offer_window": self.offer_window, "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(), "channel": self.channel,
            "budget_aed": f"{self.budget_aed:.2f}",
        }

    def definition_hash(self) -> str:
        payload = json.dumps(self.canonical(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class CampaignState:
    campaign_ref: str
    campaign_version: int
    definition: CampaignDefinition
    definition_hash: str
    status: str
    approval_ref: str | None
    estimated_cost_high_aed: Decimal
    events: tuple[str, ...]
    receipt: SubmissionReceipt | None
    outcome: AggregateOutcome | None
    simulation: bool = True


def _definition_from_json(raw: str) -> CampaignDefinition:
    data = json.loads(raw)
    return CampaignDefinition(
        merchant_id=data["merchant_id"], opportunity_id=data["opportunity_id"],
        audience_id=data["audience_id"], audience_version=int(data["audience_version"]),
        offer_percent=int(data["offer_percent"]), start_date=date.fromisoformat(data["start_date"]),
        end_date=date.fromisoformat(data["end_date"]), channel=data["channel"],
        budget_aed=Decimal(data["budget_aed"]), offer_days=tuple(data["offer_days"]),
        offer_window=data["offer_window"],
    )


def _outcome_from_json(raw: str | None) -> AggregateOutcome | None:
    if raw is None:
        return None
    data = json.loads(raw)
    return AggregateOutcome(
        campaign_ref=data["campaign_ref"],
        reporting_start=datetime.fromisoformat(data["reporting_start"]),
        reporting_end=datetime.fromisoformat(data["reporting_end"]),
        purchasers=data["purchasers"],
        campaign_associated_sales_aed=(Decimal(data["campaign_associated_sales_aed"])
                                        if data["campaign_associated_sales_aed"] is not None else None),
        campaign_cost_aed=(Decimal(data["campaign_cost_aed"])
                           if data["campaign_cost_aed"] is not None else None),
        suppressed=bool(data["suppressed"]), attribution_note=data["attribution_note"],
    )


class CampaignService:
    """Private data and state stay behind this backend-only service."""

    def __init__(
        self, path: Path, transactions: pd.DataFrame, program_eligibility: pd.DataFrame,
        merchant_id: str, as_of_date: date, adapter: MockActivationAdapter | None = None,
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.transactions = transactions
        self.program_eligibility = program_eligibility
        self.merchant_id = merchant_id
        self.as_of_date = as_of_date
        self.audiences = PrivateAudienceStore(self.path)
        self.adapter = adapter or LocalMockPartnerAdapter(self.path)
        with sqlite3.connect(self.path) as connection:
            connection.execute("""
                CREATE TABLE IF NOT EXISTS campaign_heads (
                    campaign_ref TEXT PRIMARY KEY,
                    merchant_id TEXT NOT NULL,
                    current_version INTEGER NOT NULL
                )
            """)
            connection.execute("""
                CREATE TABLE IF NOT EXISTS campaign_versions (
                    campaign_ref TEXT NOT NULL,
                    campaign_version INTEGER NOT NULL,
                    merchant_id TEXT NOT NULL,
                    definition_json TEXT NOT NULL,
                    definition_hash TEXT NOT NULL,
                    status TEXT NOT NULL,
                    approval_ref TEXT,
                    approved_definition_hash TEXT,
                    idempotency_key TEXT NOT NULL UNIQUE,
                    estimated_cost_high_cents INTEGER NOT NULL,
                    receipt_status TEXT,
                    outcome_json TEXT,
                    PRIMARY KEY (campaign_ref, campaign_version)
                )
            """)
            connection.execute("""
                CREATE TABLE IF NOT EXISTS campaign_events (
                    campaign_ref TEXT NOT NULL,
                    campaign_version INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    PRIMARY KEY (campaign_ref, campaign_version, event_type)
                )
            """)

    def _campaign_ref(self, opportunity_id: str) -> str:
        return "camp_" + uuid.uuid5(
            uuid.NAMESPACE_URL, f"sme-growth-local-simulator/{self.merchant_id}/{opportunity_id}",
        ).hex

    def _current_audience(self, opportunity_id: str) -> AudienceSummary:
        if opportunity_id == "model_retention_offer":
            try:
                model = train_retention_model(self.transactions, self.as_of_date)
                return build_model_risk_audience(
                    model, self.transactions, self.program_eligibility,
                    self.merchant_id, self.as_of_date, self.audiences,
                )
            except ValueError as error:
                raise CampaignError("The model audience is unavailable. Refresh the recommendation.") from error
        if opportunity_id != "weekday_lunch_winback":
            raise CampaignError("This campaign opportunity is unavailable.")
        return build_eligible_audience(
            self.transactions, self.program_eligibility, self.merchant_id,
            self.as_of_date, self.audiences,
        )

    def _audience_matches(self, definition: CampaignDefinition) -> bool:
        current = self._current_audience(definition.opportunity_id)
        dispatch_at = datetime.combine(self.as_of_date + timedelta(days=1), time(hour=10))
        return (
            current.activation_eligible and current.eligible_count is not None
            and current.eligible_count >= MIN_ELIGIBLE_AUDIENCE
            and current.audience_id == definition.audience_id
            and current.audience_version == definition.audience_version
            and current.expires_at > dispatch_at
            and datetime.combine(definition.end_date + timedelta(days=1), time.min) <= current.expires_at
        )

    def _validate_review(self, definition: CampaignDefinition, opportunity: Opportunity) -> Decimal:
        if (definition.merchant_id != self.merchant_id
                or definition.opportunity_id != opportunity.opportunity_id
                or definition.opportunity_id not in ("weekday_lunch_winback", "model_retention_offer")):
            raise CampaignError("This campaign does not match the selected opportunity.")
        if opportunity.audience_id != definition.audience_id or opportunity.audience_version != definition.audience_version:
            raise CampaignError("The audience changed. Refresh the opportunity and review it again.")
        current_audience = self._current_audience(definition.opportunity_id)
        if (opportunity.eligible_count != current_audience.eligible_count
                or opportunity.estimated_impact_range.assumptions.addressable_count != current_audience.eligible_count):
            raise CampaignError("The opportunity is stale. Refresh its audience and estimate.")
        if opportunity.as_of_date != self.as_of_date.isoformat() or not self._audience_matches(definition):
            raise CampaignError("The audience is no longer eligible. Review a refreshed campaign.")
        if definition.channel != SIMULATED_CHANNEL:
            raise CampaignError("Only the listed simulated offer and channel are available.")
        if definition.opportunity_id == "weekday_lunch_winback":
            if definition.offer_percent not in (10, 15):
                raise CampaignError("Only the listed simulated offer and channel are available.")
            if definition.offer_days != WEEKDAY_LUNCH_DAYS or definition.offer_window != WEEKDAY_LUNCH_WINDOW:
                raise CampaignError("Offer timing must match the reviewed weekday lunch terms.")
        elif (definition.offer_percent != 10 or definition.offer_days != ALL_DAYS
              or definition.offer_window != ALL_PURCHASES_WINDOW):
            raise CampaignError("The retention offer must be 10% off all purchases, any day.")
        if definition.start_date < self.as_of_date + timedelta(days=1) or definition.end_date < definition.start_date:
            raise CampaignError("Choose valid campaign dates after the data as-of date.")
        if not definition.budget_aed.is_finite() or definition.budget_aed <= 0:
            raise CampaignError("Enter a positive campaign budget.")
        assumptions = replace(
            opportunity.estimated_impact_range.assumptions,
            incentive_rate=definition.offer_percent / 100,
        )
        high_cost = Decimal(str(calculate_impact(assumptions).campaign_cost_high_aed))
        if definition.budget_aed < high_cost:
            raise CampaignError("Budget is below the scenario's high estimated campaign cost.")
        return high_cost

    def _event(self, connection: sqlite3.Connection, ref: str, version: int, event: str) -> None:
        connection.execute(
            "INSERT OR IGNORE INTO campaign_events VALUES (?, ?, ?)", (ref, version, event),
        )

    def review(self, definition: CampaignDefinition, opportunity: Opportunity) -> CampaignState:
        """Persist an exact safe definition; any edit produces a new version."""

        high_cost = self._validate_review(definition, opportunity)
        ref = self._campaign_ref(definition.opportunity_id)
        digest = definition.definition_hash()
        with sqlite3.connect(self.path) as connection:
            connection.execute("BEGIN IMMEDIATE")
            head = connection.execute(
                "SELECT current_version FROM campaign_heads WHERE campaign_ref = ? AND merchant_id = ?",
                (ref, self.merchant_id),
            ).fetchone()
            if head is not None:
                last = connection.execute(
                    "SELECT definition_hash FROM campaign_versions WHERE campaign_ref = ? AND campaign_version = ?",
                    (ref, head[0]),
                ).fetchone()
                if last and last[0] == digest:
                    version = int(head[0])
                    return self.get_state(ref, version)
            version = int(head[0]) + 1 if head else 1
            key = "idem_" + uuid.uuid5(uuid.NAMESPACE_URL, f"{ref}/{version}/{digest}").hex
            connection.execute("""
                INSERT INTO campaign_versions
                (campaign_ref, campaign_version, merchant_id, definition_json, definition_hash,
                 status, idempotency_key, estimated_cost_high_cents)
                VALUES (?, ?, ?, ?, ?, 'reviewed', ?, ?)
            """, (ref, version, self.merchant_id,
                  json.dumps(definition.canonical(), sort_keys=True), digest, key,
                  int(high_cost * 100)))
            connection.execute("""
                INSERT INTO campaign_heads (campaign_ref, merchant_id, current_version)
                VALUES (?, ?, ?)
                ON CONFLICT(campaign_ref) DO UPDATE SET current_version = excluded.current_version
            """, (ref, self.merchant_id, version))
            self._event(connection, ref, version, "reviewed")
        return self.get_state(ref, version)

    def _row(self, ref: str, version: int) -> sqlite3.Row:
        with sqlite3.connect(self.path) as connection:
            connection.row_factory = sqlite3.Row
            row = connection.execute("""
                SELECT v.* FROM campaign_versions AS v JOIN campaign_heads AS h
                ON h.campaign_ref = v.campaign_ref AND h.current_version = v.campaign_version
                WHERE v.campaign_ref = ? AND v.campaign_version = ?
                  AND v.merchant_id = ? AND h.merchant_id = ?
            """, (ref, version, self.merchant_id, self.merchant_id)).fetchone()
        if row is None:
            raise CampaignError("This campaign review is no longer current. Review it again.")
        return row

    def get_state(self, ref: str, version: int) -> CampaignState:
        row = self._row(ref, version)
        with sqlite3.connect(self.path) as connection:
            events = tuple(item[0] for item in connection.execute(
                "SELECT event_type FROM campaign_events WHERE campaign_ref = ? AND campaign_version = ? ORDER BY rowid",
                (ref, version),
            ))
        receipt = (SubmissionReceipt(ref, version, "submitted", RECEIPT_MESSAGE)
                   if row["receipt_status"] == "submitted" else None)
        return CampaignState(
            ref, version, _definition_from_json(row["definition_json"]), row["definition_hash"],
            row["status"], row["approval_ref"],
            Decimal(row["estimated_cost_high_cents"]) / 100, events,
            receipt, _outcome_from_json(row["outcome_json"]),
        )

    def latest_state(self, opportunity_id: str) -> CampaignState | None:
        ref = self._campaign_ref(opportunity_id)
        with sqlite3.connect(self.path) as connection:
            row = connection.execute(
                "SELECT current_version FROM campaign_heads WHERE campaign_ref = ? AND merchant_id = ?",
                (ref, self.merchant_id),
            ).fetchone()
        return self.get_state(ref, int(row[0])) if row else None

    def approve(self, ref: str, version: int, current_definition: CampaignDefinition, *, explicit: bool) -> CampaignState:
        if not explicit:
            raise CampaignError("Confirm the exact campaign before approving the simulation.")
        row = self._row(ref, version)
        if current_definition.definition_hash() != row["definition_hash"]:
            raise CampaignError("Campaign details changed. Review the new version before approval.")
        if row["status"] not in ("reviewed", "approved"):
            raise CampaignError("This campaign is not awaiting approval.")
        if not self._audience_matches(current_definition):
            raise CampaignError("The audience changed or expired. Review a refreshed campaign.")
        if row["status"] == "approved":
            return self.get_state(ref, version)
        approval = "approval_" + uuid.uuid5(
            uuid.NAMESPACE_URL, f"{ref}/{version}/{row['definition_hash']}",
        ).hex
        with sqlite3.connect(self.path) as connection:
            connection.execute("""
                UPDATE campaign_versions SET status = 'approved', approval_ref = ?,
                    approved_definition_hash = ?
                WHERE campaign_ref = ? AND campaign_version = ? AND status = 'reviewed'
            """, (approval, row["definition_hash"], ref, version))
            self._event(connection, ref, version, "approved")
        return self.get_state(ref, version)

    def _set_status(self, ref: str, version: int, status: str, *, receipt_status: str | None = None) -> None:
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                "UPDATE campaign_versions SET status = ?, receipt_status = COALESCE(?, receipt_status) "
                "WHERE campaign_ref = ? AND campaign_version = ?",
                (status, receipt_status, ref, version),
            )
            self._event(connection, ref, version, status)

    def submit(self, ref: str, version: int, current_definition: CampaignDefinition) -> CampaignState:
        """A lost response is reconciled by the stored idempotency key."""

        row = self._row(ref, version)
        if current_definition.definition_hash() != row["definition_hash"]:
            raise CampaignError("Campaign details changed. Review and approve the new version.")
        if row["status"] in ("submitted", "accepted", "simulated_dispatched", "completed"):
            return self.get_state(ref, version)
        if row["status"] not in ("approved", "submission_unknown") or row["approved_definition_hash"] != row["definition_hash"] or not row["approval_ref"]:
            raise CampaignError("Exact campaign approval is required before submission.")
        receipt = self.adapter.reconcile(row["idempotency_key"])
        if not self._audience_matches(current_definition):
            self._set_status(
                ref, version, "blocked",
                receipt_status="submitted" if receipt is not None and receipt.status != "failed" else None,
            )
            return self.get_state(ref, version)
        approved = ApprovedCampaign(
            ref, version, current_definition.audience_id, current_definition.audience_version,
            row["approval_ref"], row["approved_definition_hash"], row["idempotency_key"],
        )
        if receipt is None:
            try:
                receipt = self.adapter.submit(approved)
            except Exception:
                # The simulator may have accepted the request before the local
                # response failed. Never issue a fresh idempotency key here.
                receipt = None
        if receipt is None:
            self._set_status(ref, version, "submission_unknown")
        elif receipt.status == "failed":
            self._set_status(ref, version, "failed", receipt_status="failed")
        else:
            self._set_status(ref, version, "submitted", receipt_status="submitted")
        return self.get_state(ref, version)

    def advance(self, ref: str, version: int, current_definition: CampaignDefinition) -> CampaignState:
        """Simulate acceptance, private dispatch, and aggregate completion."""

        row = self._row(ref, version)
        if current_definition.definition_hash() != row["definition_hash"]:
            raise CampaignError("Campaign details changed. Review and approve the new version.")
        if row["status"] == "completed":
            return self.get_state(ref, version)
        if row["status"] == "submission_unknown":
            state = self.submit(ref, version, current_definition)
            if state.status != "submitted":
                return state
            row = self._row(ref, version)
        if row["status"] not in ("submitted", "accepted", "simulated_dispatched"):
            raise CampaignError("A submitted simulation is required before results are available.")
        key = row["idempotency_key"]
        try:
            if row["status"] == "submitted":
                self.adapter.accept(key)
                self._set_status(ref, version, "accepted")
            if row["status"] != "simulated_dispatched" and not self._audience_matches(current_definition):
                self._set_status(ref, version, "blocked")
                return self.get_state(ref, version)
            if row["status"] != "simulated_dispatched":
                self.adapter.mark_simulated_dispatched(key)
                self._set_status(ref, version, "simulated_dispatched")
            with sqlite3.connect(self.path) as connection:
                audience_count = connection.execute("""
                    SELECT eligible_count FROM audience_snapshots
                    WHERE audience_id = ? AND audience_version = ? AND merchant_id = ?
                """, (current_definition.audience_id, current_definition.audience_version,
                      self.merchant_id)).fetchone()
            if audience_count is None:
                self._set_status(ref, version, "blocked")
                return self.get_state(ref, version)
            outcome = simulate_aggregate_outcome(
                self.path, campaign_ref=ref, campaign_version=version,
                merchant_id=self.merchant_id, audience_id=current_definition.audience_id,
                audience_version=current_definition.audience_version,
                eligible_count=int(audience_count[0]),
                reporting_start=datetime.combine(current_definition.start_date, time.min),
                reporting_end=datetime.combine(current_definition.end_date + timedelta(days=1), time.min),
                offer_percent=current_definition.offer_percent,
                spend_per_visit_aed=Decimal("60"),
            )
            payload = asdict(outcome)
            payload["reporting_start"] = outcome.reporting_start.isoformat()
            payload["reporting_end"] = outcome.reporting_end.isoformat()
            payload["campaign_associated_sales_aed"] = (
                str(outcome.campaign_associated_sales_aed) if outcome.campaign_associated_sales_aed is not None else None
            )
            payload["campaign_cost_aed"] = str(outcome.campaign_cost_aed) if outcome.campaign_cost_aed is not None else None
            with sqlite3.connect(self.path) as connection:
                connection.execute(
                    "UPDATE campaign_versions SET status = 'completed', outcome_json = ? "
                    "WHERE campaign_ref = ? AND campaign_version = ?",
                    (json.dumps(payload), ref, version),
                )
                self._event(connection, ref, version, "completed")
        except Exception:
            self._set_status(ref, version, "failed")
        return self.get_state(ref, version)
